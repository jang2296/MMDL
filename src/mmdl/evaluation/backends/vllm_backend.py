"""Pinned vLLM 0.11.0 backend for the accelerated MMMU protocol."""

from __future__ import annotations

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
from mmdl.runtime.contracts import MODEL_REVISION


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


class VLLMBackend:
    """Bounded vLLM inference with a separately frozen evaluation protocol."""

    def __init__(self, model_path: str, cfg: dict[str, Any], hw: dict[str, Any], kind: str = "base",
                 base_path: str | None = None):
        if hw["placement"] != "gpu_only":
            raise ValueError("vLLM accelerated protocol requires the GPU-only hardware profile")
        if cfg["execution"]["backend"] != "vllm" or cfg["execution"]["batch_size"] not in (1, 2):
            raise ValueError("vLLM requires a validated protocol with one or two active requests")
        if kind == "adapter":
            raise ValueError("vLLM accelerated protocol does not load an unmerged adapter")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with native BF16 is required")
        if kind in {"full", "merged"}:
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
        self.llm = LLM(
            model=model_path,
            tokenizer=processor_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_num_seqs=self.batch_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.90,
            quantization=None,
            generation_config="auto",
            mm_processor_kwargs=self.mm_processor_kwargs,
            enable_prefix_caching=False,
            async_scheduling=False,
            enforce_eager=True,
            limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT, "video": 0},
            trust_remote_code=False,
        )
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
                "gpu_memory_utilization": 0.90,
                "prefix_caching": False,
                "tensor_parallel_size": 1,
                "quantization": None,
                "generation_config": "auto",
                "mm_processor_kwargs": self.mm_processor_kwargs,
                "enforce_eager": True,
                "limit_mm_per_prompt": {"image": MAX_IMAGES_PER_PROMPT, "video": 0},
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
        item, engine_images = prepare_protocol_inputs(
            self.processor, self.model_config, request["messages"], request["images"], self.cfg.get("image", {}))
        if item["input_tokens"] + self.cfg["generation"]["max_new_tokens"] > self.max_model_len:
            raise ValueError("Input plus max_new_tokens exceeds vLLM max_model_len; refusing truncation")
        prompt = {"prompt": item["chat_prompt"], "multi_modal_data": {"image": engine_images}}
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
        """Continuously refill up to two vLLM slots, yielding only completed requests."""
        iterator = iter(requests)
        active: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        exhausted = False
        engine = self.llm.llm_engine

        def fill_slots() -> None:
            nonlocal exhausted
            while not exhausted and len(active) < self.batch_size:
                try:
                    request = next(iterator)
                except StopIteration:
                    exhausted = True
                    return
                request_id = request["sample_id"]
                if not isinstance(request_id, str) or not request_id or request_id in seen:
                    raise ValueError("Continuous vLLM requests require unique non-empty string sample_id values")
                item, prompt, params = self._prepare_request(request, final_only=True)
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
                active_count = len(active)
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
