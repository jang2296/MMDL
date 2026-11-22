"""Pinned vLLM 0.11.0 backend for the accelerated MMMU protocol."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
from collections.abc import Iterable, Iterator
from typing import Any

import psutil
import torch
from transformers import AutoConfig, AutoProcessor

from mmdl.evaluation.backends.input_preparation import prepare_protocol_inputs
from mmdl.runtime.artifacts import digest
from mmdl.runtime.contracts import MODEL_REVISION, EXECUTION_PROFILES, SUBMISSION_PROFILES, validate_profile_model


VLLM_VERSION = "0.11.0"
MAX_MODEL_LEN = 36864  # 32768 output + measured maximum 2655 input; never truncate inputs.
MAX_IMAGES_PER_PROMPT = 5


def _vllm() -> tuple[Any, Any, str]:
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - covered by the isolated vLLM environment.
        raise RuntimeError("vLLM backend requires env/requirements-vllm.lock") from exc
    version = getattr(vllm, "__version__", "unknown")
    if version != VLLM_VERSION:
        raise RuntimeError(f"Pinned accelerated protocol requires vLLM {VLLM_VERSION}, found {version}")
    return LLM, SamplingParams, version


def _final_only_output_kind() -> Any:
    """Load the vLLM enum only in the isolated accelerated environment."""
    try:
        from vllm.sampling_params import RequestOutputKind
    except ImportError as exc:  # pragma: no cover - requires the isolated vLLM environment.
        raise RuntimeError("vLLM backend requires env/requirements-vllm.lock") from exc
    return RequestOutputKind.FINAL_ONLY


def _device_memory_used_bytes() -> int | None:
    """vLLM allocates in worker processes, so query the device rather than Torch's allocator."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.splitlines()[0].strip()) * 1024**2
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


KV_RESERVATION_RULE = (
    "admit requests in submission order while active < max_num_seqs and the sum over active requests of "
    "ceil((input_tokens + max_new_tokens) / block_size) * block_size stays <= "
    "floor(kv_safety_fraction * actual_kv_token_capacity); the next request waits, nothing skips ahead")


class KVAdmission:
    """Reserve each admitted request's whole context in the measured KV pool, so vLLM never preempts."""

    def __init__(self, *, usable_tokens: int, block_size: int, slots: int, max_new_tokens: int):
        if any(type(n) is not int or n < 1 for n in (usable_tokens, block_size, slots, max_new_tokens)):
            raise ValueError("Invalid KV admission limits")
        self.usable_tokens, self.block_size = usable_tokens, block_size
        self.slots, self.max_new_tokens = slots, max_new_tokens
        self.reserved: dict[str, int] = {}
        self.max_concurrent = self.steps = self.active_step_sum = 0

    def need(self, input_tokens: int) -> int:
        return math.ceil((input_tokens + self.max_new_tokens) / self.block_size) * self.block_size

    def admit(self, request_id: str, input_tokens: int) -> bool:
        need = self.need(input_tokens)
        if need > self.usable_tokens:
            raise ValueError(f"Request {request_id} can never fit the measured KV pool: {need} > {self.usable_tokens}")
        if len(self.reserved) >= self.slots or sum(self.reserved.values()) + need > self.usable_tokens:
            return False
        self.reserved[request_id] = need
        self.max_concurrent = max(self.max_concurrent, len(self.reserved))
        return True

    def release(self, request_id: str) -> None:
        del self.reserved[request_id]

    def observe_step(self) -> None:
        self.steps += 1
        self.active_step_sum += len(self.reserved)

    def summary(self) -> dict[str, Any]:
        return {"rule": KV_RESERVATION_RULE, "usable_tokens": self.usable_tokens, "block_size": self.block_size,
                "max_num_seqs": self.slots, "max_new_tokens": self.max_new_tokens, "engine_steps": self.steps,
                "max_concurrent_requests": self.max_concurrent,
                "mean_concurrent_requests_per_step": self.active_step_sum / self.steps if self.steps else None}


def allocated_kv_capacity(engine: Any, version: str, *, max_model_len: int,
                          slots: int, safety_fraction: float = .90, admission: str = "fixed") -> dict[str, Any]:
    """Read the same live scheduler blocks as v17; reserve each complete context."""
    if version != VLLM_VERSION or type(slots) is not int or slots < 1:
        raise ValueError("Unsupported KV engine version or slot count")
    try:
        cache = engine.engine_core.engine_core.scheduler.kv_cache_config
        sizes = {group.kv_cache_spec.block_size for group in cache.kv_cache_groups}
        if len(sizes) != 1:
            raise ValueError("Mixed KV block sizes")
        block_size, blocks = sizes.pop(), cache.num_blocks
    except (AttributeError, TypeError) as exc:
        raise ValueError("Actual allocated KV block structure unavailable") from exc
    if any(type(n) is not int or n < 1 for n in (blocks, block_size, max_model_len)):
        raise ValueError("Invalid allocated KV capacity")
    if not 0 < safety_fraction <= .90:
        raise ValueError("KV safety fraction cannot exceed .90")
    reserved = math.ceil(max_model_len / block_size) * block_size
    tokens, required = blocks * block_size, slots * reserved
    usable = math.floor(safety_fraction * tokens)
    if admission == "kv_reservation":
        # Concurrency is admitted per request from measured input lengths; one maximum context must fit.
        value = dict(version="submission-kv-reservation-v1", admission=admission, rule=KV_RESERVATION_RULE,
                     vllm_version=version, slots=slots, max_model_len=max_model_len, actual_blocks=blocks,
                     block_size=block_size, actual_kv_token_capacity=tokens, usable_tokens=usable,
                     safety_fraction=safety_fraction, reserved_per_request=reserved, required_tokens=reserved,
                     full_context_requests=usable // reserved)
        value["pass"] = reserved <= usable
        if not value["pass"]:
            raise ValueError(f"Insufficient actual KV capacity for one maximum request: {reserved} > {usable}")
        return value
    if admission != "fixed":
        raise ValueError(f"Unknown KV admission rule: {admission}")
    result = dict(version="submission-kv-v1", vllm_version=version, slots=slots,
                  max_model_len=max_model_len, actual_blocks=blocks, block_size=block_size,
                  actual_kv_token_capacity=tokens, usable_tokens=usable,
                  reserved_per_request=reserved, required_tokens=required,
                  safe_slots=usable // reserved, pass_=required <= usable)
    result["pass"] = result.pop("pass_")
    if not result["pass"]:
        raise ValueError(f"Insufficient actual KV capacity: {required} > {usable}; re-register both arms")
    return result


def observe_reservation_engine(backend: Any, mode: str, slots: int, max_model_len: int) -> dict[str, Any]:
    """Live-engine checks of v17 observe_engine for a measured-admission profile (its own context length)."""
    engine = backend.llm.llm_engine
    config = engine.vllm_config
    model, scheduler, compilation = config.model_config, config.scheduler_config, config.compilation_config
    graph_mode = getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode))
    sampler = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.sampler
    value = {"version": "kv-reservation-engine-observed-v1", "engine_mode": mode,
             "vllm_version": backend.vllm_version, "enforce_eager": model.enforce_eager,
             "dtype": str(model.dtype), "quantization": model.quantization,
             "logprobs_mode": model.logprobs_mode, "sampler_logprobs_mode": sampler.logprobs_mode,
             "max_model_len": model.max_model_len, "slots": scheduler.max_num_seqs,
             "compilation_config": {"level": compilation.level, "cudagraph_mode": graph_mode,
                                    "cudagraph_capture_sizes": compilation.cudagraph_capture_sizes},
             "allocated_kv": allocated_kv_capacity(engine, backend.vllm_version, max_model_len=model.max_model_len,
                                                   slots=scheduler.max_num_seqs, admission="kv_reservation")}
    if (mode not in ("cuda_graph", "eager") or type(model.enforce_eager) is not bool
            or model.enforce_eager != (mode == "eager")
            or str(model.dtype) not in ("bfloat16", "torch.bfloat16") or model.quantization is not None
            or model.logprobs_mode != "raw_logprobs" or sampler.logprobs_mode != "raw_logprobs"
            or model.max_model_len != max_model_len or scheduler.max_num_seqs != slots
            or (mode == "cuda_graph" and graph_mode in ("NONE", "None", "0"))):
        raise ValueError("KV-reservation engine: effective engine config differs from requested execution")
    return value | {"sha256": digest(value)}


class VLLMBackend:
    """Bounded vLLM inference with a separately frozen evaluation protocol."""

    def __init__(self, model_path: str, cfg: dict[str, Any], hw: dict[str, Any], kind: str = "base",
                 base_path: str | None = None, *, rehearsal: bool = False):
        if hw["placement"] != "gpu_only":
            raise ValueError("vLLM accelerated protocol requires the GPU-only hardware profile")
        diagnostic = (cfg.get("protocol_id") == "input-recovery-v1"
                      and cfg["generation"].get("max_new_tokens") == 8192)
        submission = cfg.get("protocol_id") in SUBMISSION_PROFILES
        if submission and not rehearsal:
            validate_profile_model(cfg, kind)
            if cfg["execution"] != EXECUTION_PROFILES[cfg["protocol_id"]]:
                raise ValueError("Submission execution differs from its named fixed profile")
        if kind == "unmerged-lora" and not (submission or rehearsal):
            raise ValueError("Unmerged LoRA requires a named submission profile")
        self.admission = cfg["execution"].get("admission", "fixed")
        if self.admission not in ("fixed", "kv_reservation") or (self.admission != "fixed" and not submission):
            raise ValueError("Measured KV admission requires its named submission profile")
        allowed_slots = (tuple(range(1, 65)) if self.admission == "kv_reservation" else
                         tuple(range(1, 10)) if submission or rehearsal else ((2, 4, 8) if diagnostic else (1, 2)))
        if (cfg["execution"]["backend"] != "vllm"
                or type(cfg["execution"]["batch_size"]) is not int
                or cfg["execution"]["batch_size"] not in allowed_slots):
            raise ValueError("vLLM requires a validated protocol with one or two active requests")
        if kind == "adapter":
            raise ValueError("vLLM accelerated protocol does not load an unmerged adapter")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with native BF16 is required")
        if kind in {"full", "merged", "unmerged-lora"}:
            if base_path is None:
                raise ValueError("Derived model requires its pinned base path")
            reference = AutoConfig.from_pretrained(base_path, local_files_only=True).to_dict()
            candidate = AutoConfig.from_pretrained(model_path, local_files_only=True).to_dict()
            for key in ("model_type", "text_config", "vision_config"):
                if candidate[key] != reference[key]:
                    raise ValueError("Derived checkpoint changed the original model structure")
        processor_path = model_path if kind == "base" else base_path
        assert processor_path is not None
        self.cfg = cfg
        self.rehearsal = rehearsal
        self.max_images_per_prompt = cfg["execution"].get("max_images_per_prompt", MAX_IMAGES_PER_PROMPT)
        self.enforce_eager = cfg["execution"].get("enforce_eager", True)
        self.gpu_memory_utilization = cfg["execution"].get("gpu_memory_utilization", .90)
        self.capacity_receipt = None
        self.observed_execution = None
        if self.max_images_per_prompt not in (5, 7) or type(self.enforce_eager) is not bool:
            raise ValueError("Only registered five/seven-image eager/graph profiles are allowed")
        if submission or rehearsal or kind == "unmerged-lora":
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            if os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] != "0":
                raise ValueError("Submission validation and LoRA require an in-process vLLM engine")
        self.max_model_len = cfg["execution"].get("max_model_len", MAX_MODEL_LEN)
        self.batch_size = cfg["execution"]["batch_size"]
        self.mm_processor_kwargs: dict[str, Any] = {
            "size": {
                "shortest_edge": cfg["image"]["min_pixels"],
                "longest_edge": cfg["image"]["max_pixels"],
            }
        }
        if cfg["image"].get("preprocessing") == "qwen_vl_utils_0_0_14":
            self.mm_processor_kwargs["do_resize"] = False
        self.processor = AutoProcessor.from_pretrained(
            processor_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
        )
        self.processor.image_processor.size = {
            "shortest_edge": cfg["image"]["min_pixels"],
            "longest_edge": cfg["image"]["max_pixels"],
        }
        self.tokenizer_special_tokens = {
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        self.model_config = AutoConfig.from_pretrained(
            model_path if kind != "adapter" else base_path,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
        )
        LLM, self._sampling_params_cls, self.vllm_version = _vllm()
        self._request_output_kind = _final_only_output_kind()
        start = time.monotonic()
        engine_kwargs = dict(
            model=model_path,
            tokenizer=processor_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_num_seqs=self.batch_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            quantization=None,
            generation_config="auto",
            mm_processor_kwargs=self.mm_processor_kwargs,
            enable_prefix_caching=False,
            async_scheduling=False,
            enforce_eager=self.enforce_eager,
            limit_mm_per_prompt={"image": self.max_images_per_prompt, "video": 0},
            trust_remote_code=False,
        )
        if submission or rehearsal:
            engine_kwargs["logprobs_mode"] = "raw_logprobs"
            engine_kwargs["disable_log_stats"] = False
        if kind == "unmerged-lora":
            from fine_tuning.notebook_rl.proof_eval import attach_lora, lora_llm
            self.llm = lora_llm(LLM, engine_kwargs)
            attach_lora(self, Path(model_path))
        else:
            self.llm = LLM(**engine_kwargs)
        if submission or rehearsal:
            self.capacity_receipt = allocated_kv_capacity(
                self.llm.llm_engine, self.vllm_version, max_model_len=self.max_model_len,
                slots=self.batch_size, safety_fraction=cfg["execution"].get("kv_safety_fraction", .90),
                **({"admission": self.admission} if self.admission != "fixed" else {}))
        if self.admission == "kv_reservation":
            receipt = self.capacity_receipt
            assert receipt is not None
            self.kv_admission_limits = dict(usable_tokens=receipt["usable_tokens"], block_size=receipt["block_size"],
                                            slots=self.batch_size, max_new_tokens=cfg["generation"]["max_new_tokens"])
        if submission and not rehearsal:
            mode = "eager" if self.enforce_eager else "cuda_graph"
            if self.admission == "kv_reservation":
                # v17 observe_engine is a pinned reference source fixed to 40,960-token full-context slots.
                self.observed_execution = observe_reservation_engine(self, mode, self.batch_size, self.max_model_len)
            else:
                from fine_tuning.error_note.v17_eval_engine import observe_engine
                self.observed_execution = observe_engine(self, mode, self.batch_size)
        self.engine_default_sampling_params = repr(self.llm.get_default_sampling_params())
        self.load_seconds = time.monotonic() - start

    def describe(self) -> dict[str, Any]:
        return {
            "model_load_seconds": self.load_seconds,
            "device_map": {"": "cuda:0"},
            "placement": {"gpu_only": True},
            "generation_config": self._generation_settings(),
            "engine_default_sampling_params": self.engine_default_sampling_params,
            "engine_tokenizer_special_tokens": self.tokenizer_special_tokens,
            "presence_penalty": {
                "implementation": "vllm-native-generated-only",
                "value": self.cfg["generation"]["presence_penalty"],
                "order": "before temperature/top_k/top_p",
            },
            "vllm": {
                "version": self.vllm_version,
                "async_scheduling": False,
                "max_num_seqs": self.batch_size,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "prefix_caching": False,
                "tensor_parallel_size": 1,
                "quantization": None,
                "generation_config": "auto",
                "mm_processor_kwargs": self.mm_processor_kwargs,
                "enforce_eager": self.enforce_eager,
                "limit_mm_per_prompt": {"image": self.max_images_per_prompt, "video": 0},
                "allocated_kv": self.capacity_receipt,
                **({"admission": KV_RESERVATION_RULE} if getattr(self, "admission", "fixed") != "fixed" else {}),
                "observed_execution": self.observed_execution,
                "rehearsal": self.rehearsal,
                "timing_policy": "batch_wall_divided_by_size (legacy generate_batch); "
                                 "engine_step_wall_divided_by_active_requests (generate_stream)",
            },
            "processor_tensor_hash_scope": "pinned_cpu_processor_reference_only",
            "actual_engine_pixel_tensors_verified": False,
            "image_processor": self.processor.image_processor.to_dict(),
            "chat_template_sha256": digest(self.processor.chat_template),
            "available_ram_bytes_at_load": psutil.virtual_memory().available,
        }

    def _generation_settings(self) -> dict[str, Any]:
        values = dict(self.cfg["generation"])
        values.pop("seed", None)
        values.pop("seed_policy", None)
        return values | {"max_model_len": self.max_model_len}

    def _sampling_params(self, seed: int, *, final_only: bool = False) -> Any:
        generation = self.cfg["generation"]
        values = dict(
            n=1,
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            top_k=generation["top_k"],
            repetition_penalty=generation["repetition_penalty"],
            presence_penalty=generation["presence_penalty"],
            max_tokens=generation["max_new_tokens"],
            seed=seed,
        )
        if final_only:
            values["output_kind"] = self._request_output_kind
        return self._sampling_params_cls(**values)

    def _prepare_request(self, request: dict[str, Any], *, final_only: bool) -> tuple[dict[str, Any], dict[str, Any], Any]:
        if not 1 <= len(request["images"]) <= getattr(self, "max_images_per_prompt", MAX_IMAGES_PER_PROMPT):
            raise ValueError("Request image count exceeds the registered suite limit")
        item, engine_images = prepare_protocol_inputs(
            self.processor, self.model_config, request["messages"], request["images"], self.cfg.get("image", {}))
        if item["input_tokens"] + self.cfg["generation"]["max_new_tokens"] > self.max_model_len:
            raise ValueError("Input plus max_new_tokens exceeds vLLM max_model_len; refusing truncation")
        prompt = {"prompt": item["chat_prompt"], "multi_modal_data": {"image": engine_images}}
        item["passed_image_count"] = len(engine_images)
        return item, prompt, self._sampling_params(request["seed"], final_only=final_only)

    def _row(self, request: dict[str, Any], item: dict[str, Any], params: Any, output: Any,
             *, generation_seconds: float, timing_policy: str, request_latency_seconds: float | None = None,
             batch_generation_seconds: float | None = None, batch_size_actual: int | None = None,
             observed_device_memory_used: int | None = None,
             memory_measurement: str = "completion_device_snapshot_not_true_peak") -> dict[str, Any]:
        actual_prompt_ids = list(output.prompt_token_ids)
        if actual_prompt_ids != item["prompt_token_ids"]:
            raise RuntimeError("vLLM prompt token IDs differ from the pinned CPU processor reference")
        if len(output.outputs) != 1:
            raise RuntimeError("vLLM did not return exactly one sampled completion")
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        row = {
            "sample_id": request["sample_id"],
            "raw_response": self.processor.tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "generated_token_ids": token_ids,
            "generated_tokens": len(token_ids),
            "input_tokens": item["input_tokens"],
            "source_image_count": len(request["images"]),
            "passed_image_count": item.get("passed_image_count", len(request["images"])),
            "lora_request_id": getattr(getattr(self, "lora_request", None), "lora_int_id", None),
            "image_grid_thw": item["image_grid_thw"],
            "input_tensors": item["input_tensors"],
            "input_tensors_scope": "pinned_cpu_processor_reference_only",
            "input_sha256": item["input_sha256"],
            "chat_prompt": item["chat_prompt"],
            "actual_engine_prompt_token_ids": actual_prompt_ids,
            "actual_engine_prompt_token_ids_sha256": digest(actual_prompt_ids),
            "requested_sampling_params": repr(params),
            "generation_seconds": generation_seconds,
            "generation_timing_policy": timing_policy,
            "finish_reason": completion.finish_reason or "unknown",
            "stop_reason": getattr(completion, "stop_reason", None),
            "peak_vram_allocated_bytes": None,
            "peak_vram_reserved_bytes": None,
            "allocator_measurement": "worker_allocator_unavailable",
            "observed_device_memory_used_bytes": observed_device_memory_used
            if observed_device_memory_used is not None else _device_memory_used_bytes(),
            "memory_measurement": memory_measurement,
            "actual_engine_pixel_tensors_verified": False,
            "processed_image_sizes": item.get("processed_image_sizes"),
            "image_preprocessing": self.cfg.get("image", {}).get("preprocessing", "hf_processor"),
        }
        if request_latency_seconds is not None:
            row["request_latency_seconds"] = request_latency_seconds
        if batch_generation_seconds is not None:
            row["batch_generation_seconds"] = batch_generation_seconds
        if batch_size_actual is not None:
            row["batch_size_actual"] = batch_size_actual
        return row

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not 1 <= len(requests) <= self.batch_size:
            raise ValueError(f"vLLM batch must contain 1..{self.batch_size} requests")
        prepared_requests = [self._prepare_request(request, final_only=False) for request in requests]
        prepared = [item for item, _, _ in prepared_requests]
        prompts = [prompt for _, prompt, _ in prepared_requests]
        start = time.monotonic()
        sampling_params = [params for _, _, params in prepared_requests]
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        batch_seconds = time.monotonic() - start
        device_memory_used = _device_memory_used_bytes()
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned a different number of requests")
        return [
            self._row(request, item, params, output,
                      generation_seconds=batch_seconds / len(requests),
                      timing_policy="batch_wall_divided_by_size",
                      batch_generation_seconds=batch_seconds, batch_size_actual=len(requests),
                      observed_device_memory_used=device_memory_used,
                      memory_measurement="post_batch_device_snapshot_not_true_peak")
            for request, item, params, output in zip(requests, prepared, sampling_params, outputs, strict=True)
        ]

    def generate_stream(self, requests: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Continuously refill the validated slot count, yielding only completions."""
        iterator = iter(requests)
        active: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        exhausted = False
        engine = self.llm.llm_engine
        limits = getattr(self, "kv_admission_limits", None)
        admission = KVAdmission(**limits) if limits else None
        self.last_admission = admission
        held: tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any] | None = None

        def fill_slots() -> None:
            nonlocal exhausted, held
            while not exhausted and len(active) < self.batch_size:
                if held is None:
                    try:
                        request = next(iterator)
                    except StopIteration:
                        exhausted = True
                        return
                    request_id = request["sample_id"]
                    if not isinstance(request_id, str) or not request_id or request_id in seen:
                        raise ValueError("Continuous vLLM requests require unique non-empty string sample_id values")
                    held = (request, *self._prepare_request(request, final_only=True))
                request, item, prompt, params = held
                request_id = request["sample_id"]
                # The prepared head request waits for KV room; later requests never skip ahead of it.
                if admission is not None and not admission.admit(request_id, item["input_tokens"]):
                    return
                held = None
                enqueue_time = time.monotonic()
                # Let vLLM use its wall-clock arrival timestamp; our latency clock is monotonic.
                engine.add_request(request_id, prompt, params)
                active[request_id] = {
                    "request": request, "item": item, "params": params, "enqueue_time": enqueue_time,
                    "generation_seconds": 0.0,
                }
                seen.add(request_id)

        try:
            fill_slots()
            while active:
                deadline_check = getattr(self, "_check_deadline", None)
                if deadline_check is not None:
                    deadline_check()
                active_count = len(active)
                if admission is not None:
                    admission.observe_step()
                step_start = time.monotonic()
                outputs = engine.step()
                step_share = (time.monotonic() - step_start) / active_count
                for running_state in active.values():
                    running_state["generation_seconds"] += step_share
                completed_at = time.monotonic()
                for output in outputs:
                    request_id = output.request_id
                    state = active.get(request_id)
                    if state is None:
                        raise RuntimeError(f"vLLM returned an unknown continuous request ID: {request_id}")
                    if output.finished:
                        row = self._row(
                            state["request"], state["item"], state["params"], output,
                            generation_seconds=state["generation_seconds"],
                            timing_policy="engine_step_wall_divided_by_active_requests",
                            request_latency_seconds=completed_at - state["enqueue_time"],
                        )
                        del active[request_id]
                        if admission is not None:
                            admission.release(request_id)
                        # Persist each valid completion before checking any later output.
                        yield row
                fill_slots()
        finally:
            if active:
                handling_error = sys.exc_info()[0] is not None
                try:
                    engine.abort_request(list(active))
                except Exception as exc:
                    if not handling_error:
                        raise
                    warnings.warn(f"Failed to abort outstanding vLLM requests: {exc}", RuntimeWarning)
