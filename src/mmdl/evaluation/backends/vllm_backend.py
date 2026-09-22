"""Pinned vLLM 0.11.0 backend for the accelerated MMMU protocol."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import psutil
import torch
from transformers import AutoConfig, AutoProcessor

from mmdl.evaluation.backends.input_preparation import prepare_inputs
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
    """Batch-two vLLM inference without changing model, processor, or sampling recipe."""

    def __init__(self, model_path: str, cfg: dict[str, Any], hw: dict[str, Any], kind: str = "base",
                 base_path: str | None = None):
        if hw["placement"] != "gpu_only":
            raise ValueError("vLLM accelerated protocol requires the GPU-only hardware profile")
        if cfg["execution"]["backend"] != "vllm" or cfg["execution"]["batch_size"] != 2:
            raise ValueError("vLLM accelerated protocol requires backend=vllm and batch_size=2")
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
        self.max_model_len = MAX_MODEL_LEN
        self.batch_size = cfg["execution"]["batch_size"]
        self.mm_processor_kwargs = {
            "size": {
                "shortest_edge": cfg["image"]["min_pixels"],
                "longest_edge": cfg["image"]["max_pixels"],
            }
        }
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
                "timing_policy": "batch_wall_divided_by_size",
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

    def _sampling_params(self, seed: int) -> Any:
        generation = self.cfg["generation"]
        return self._sampling_params_cls(
            n=1,
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            top_k=generation["top_k"],
            repetition_penalty=generation["repetition_penalty"],
            presence_penalty=generation["presence_penalty"],
            max_tokens=generation["max_new_tokens"],
            seed=seed,
        )

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not 1 <= len(requests) <= self.batch_size:
            raise ValueError(f"vLLM batch must contain 1..{self.batch_size} requests")
        prepared = [prepare_inputs(self.processor, self.model_config, request["messages"], request["images"])
                    for request in requests]
        for item in prepared:
            if item["input_tokens"] + self.cfg["generation"]["max_new_tokens"] > self.max_model_len:
                raise ValueError("Input plus max_new_tokens exceeds vLLM max_model_len; refusing truncation")
        prompts = [
            {"prompt": item["chat_prompt"], "multi_modal_data": {"image": request["images"]}}
            for item, request in zip(prepared, requests, strict=True)
        ]
        start = time.monotonic()
        sampling_params = [self._sampling_params(request["seed"]) for request in requests]
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        batch_seconds = time.monotonic() - start
        device_memory_used = _device_memory_used_bytes()
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned a different number of requests")
        rows = []
        for request, item, params, output in zip(requests, prepared, sampling_params, outputs, strict=True):
            actual_prompt_ids = list(output.prompt_token_ids)
            if actual_prompt_ids != item["prompt_token_ids"]:
                raise RuntimeError("vLLM prompt token IDs differ from the pinned CPU processor reference")
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM did not return exactly one sampled completion")
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            raw = self.processor.tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            rows.append({
                "sample_id": request["sample_id"],
                "raw_response": raw,
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
                "generation_seconds": batch_seconds / len(requests),
                "generation_timing_policy": "batch_wall_divided_by_size",
                "batch_generation_seconds": batch_seconds,
                "batch_size_actual": len(requests),
                "finish_reason": completion.finish_reason or "unknown",
                "stop_reason": getattr(completion, "stop_reason", None),
                "peak_vram_allocated_bytes": None,
                "peak_vram_reserved_bytes": None,
                "allocator_measurement": "worker_allocator_unavailable",
                "observed_device_memory_used_bytes": device_memory_used,
                "memory_measurement": "post_batch_device_snapshot_not_true_peak",
                "actual_engine_pixel_tensors_verified": False,
            })
        return rows
