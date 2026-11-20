"""Explicit local NF4 policy; the existing BF16 protocol and loader stay separate.

The caller must verify the original snapshot files and the separate execution/data
approval before loading. This module supplies neither approval nor an RL objective.
"""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fine_tuning.modules import BASE_ID, BASE_REVISION, inspect_modules, tensor_sha256

from .checkpoint import file_hash
from .policy import ARMS, PolicyBundle, assert_base_unchanged, configure_policy

MERGERS = (
    "model.visual.merger",
    "model.visual.deepstack_merger_list.0",
    "model.visual.deepstack_merger_list.1",
    "model.visual.deepstack_merger_list.2",
)
PINNED_VERSIONS = {"torch": "2.8.0+cu128", "transformers": "4.57.1", "peft": "0.17.1",
                   "accelerate": "1.11.0", "bitsandbytes": "0.48.2"}
_CACHE_FORMAT = "notebook-rl-nf4-base-v2"
_CACHE_METADATA = "nf4_base.json"
_STATE_KEYS = {"absmax", "quant_map", "nested_absmax", "nested_quant_map",
               "quant_state.bitsandbytes__nf4"}


def quantization_contract() -> dict:
    """A new protocol must explicitly bind this format, never inherit BF16 identity."""
    return {"method": "bitsandbytes", "load_in_4bit": True, "quant_type": "nf4",
            "compute_dtype": "bfloat16", "use_double_quant": True, "storage_dtype": "uint8",
            "linear_input_dtype": "bfloat16",
            "excluded_modules": [*MERGERS, "lm_head"]}


def _versions() -> dict[str, str]:
    actual = {name: version(name) for name in PINNED_VERSIONS}
    if actual != PINNED_VERSIONS:
        raise ValueError("NF4 policy: exact separately pinned NF4 runtime required")
    return actual


def _validate_protocol(model_path: Path, arm: str, protocol: dict, device_index: int, *,
                       require_snapshot: bool = True) -> None:
    if (type(device_index) is not int or device_index < 0 or arm not in ARMS or arm == "MOD-FML"
            or not model_path.is_dir() or require_snapshot and model_path.name != BASE_REVISION):
        raise ValueError("NF4 policy: pinned local snapshot, explicit CUDA index and Vision/LLM arm required")
    if (protocol.get("initial_checkpoint", {}).get("model_id") != BASE_ID
            or protocol["initial_checkpoint"].get("revision") != BASE_REVISION
            or protocol.get("quantization") != quantization_contract()
            or protocol.get("model", {}).get("modules_to_save") != list(MERGERS)
            or protocol.get("forbidden") not in (
                {"additional_sft": True, "teacher_or_upper_model_labels": True, "benchmark_training": True},
                {"additional_sft": False, "teacher_or_upper_model_labels": True, "benchmark_training": True})):
        raise ValueError("NF4 policy: explicit separate NF4/identity/forbidden/Merger contract required")


def _inspect_quantized_base(base: Any, device: torch.device) -> list[str]:
    from bitsandbytes.nn import Linear4bit, Params4bit

    inventory = inspect_modules(base)
    if (base.config.model_type != "qwen3_vl" or not getattr(base, "is_loaded_in_4bit", False)
            or len(inventory["targets"]["L"]) != 252 or len(inventory["targets"]["V"]) != 48
            or set(inventory["mergers"]) != set(MERGERS)
            or any(p.device != device for p in base.parameters())):
        raise ValueError("NF4 policy: full pinned architecture must reside on the selected CUDA device")
    quantized = []
    for name, module in base.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        excluded = name == "lm_head" or any(name.startswith(root + ".") for root in MERGERS)
        if excluded:
            if isinstance(module, Linear4bit) or module.weight.dtype != torch.bfloat16:
                raise ValueError(f"NF4 policy: excluded module must retain BF16 source weights: {name}")
        else:
            if (not isinstance(module, Linear4bit) or not isinstance(module.weight, Params4bit)
                    or module.weight.dtype != torch.uint8 or module.weight.requires_grad
                    or module.weight.quant_type != "nf4" or not module.weight.compress_statistics
                    or module.weight.quant_storage != torch.uint8 or not module.weight.bnb_quantized
                    or module.compute_dtype != torch.bfloat16):
                raise ValueError(f"NF4 policy: missing/incorrect actual NF4 module: {name}")
            quantized.append(name)
    if not set(inventory["targets"]["L"] + inventory["targets"]["V"]) <= set(quantized):
        raise ValueError("NF4 policy: both adapter target groups must have quantized frozen bases")
    return quantized


def _cuda_device(device_index: int) -> torch.device:
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise ValueError("NF4 policy: actual CUDA/BF16 required; no CPU/offload fallback")
    with torch.cuda.device(device_index):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("NF4 policy: native BF16 support required")
    return torch.device("cuda", device_index)


def _bf16_linear_input(module: nn.Module, inputs: tuple) -> tuple:
    if not inputs or not isinstance(inputs[0], torch.Tensor) or not inputs[0].is_floating_point():
        raise ValueError("NF4 policy: floating Linear4bit input required")
    # bnb 0.48.2 otherwise changes frozen bias.data to a FP32 activation's dtype.
    return inputs if inputs[0].dtype == torch.bfloat16 else (inputs[0].to(torch.bfloat16), *inputs[1:])


def _register_bf16_inputs(base: Any) -> None:
    from bitsandbytes.nn import Linear4bit

    for module in base.modules():
        if isinstance(module, Linear4bit):
            module.register_forward_pre_hook(_bf16_linear_input)


def quantization_state_sha256(model: Any) -> dict:
    """Hash every scale/code and packed shape/dtype/blocksize, beyond weight bytes."""
    from bitsandbytes.nn import Linear4bit

    base = model.get_base_model() if hasattr(model, "peft_config") else model
    states = {}
    for name, module in base.named_modules():
        if not isinstance(module, Linear4bit):
            continue
        canonical = name.removesuffix(".base_layer")
        state = getattr(module.weight, "quant_state", None)
        if (canonical in states or state is None or state.quant_type != "nf4"
                or state.dtype != torch.bfloat16 or state.blocksize != 64 or not state.nested
                or module.quant_state is not state):
            raise ValueError("NF4 policy: complete, unambiguous NF4 double-quantization state required")
        packed = state.as_dict(packed=True)
        if (set(packed) != _STATE_KEYS
                or any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
                       for value in packed.values())):
            raise ValueError("NF4 policy: incomplete/nonfinite serialized quantization state")
        states[canonical] = {key: {"sha256": tensor_sha256(value), "shape": list(value.shape),
                                  "dtype": str(value.dtype)} for key, value in packed.items()}
    if not states:
        raise ValueError("NF4 policy: no actual quantization state")
    return states


def assert_quantized_base_unchanged(bundle: PolicyBundle) -> None:
    """Call around updates/checkpoints/reloads in addition to ordinary BF16 checks."""
    assert_base_unchanged(bundle)
    provenance = bundle.inventory.get("quantization", {})
    if (provenance.get("contract") != quantization_contract()
            or provenance.get("versions") != _versions()
            or quantization_state_sha256(bundle.model) != provenance.get("state_sha256")):
        raise ValueError("NF4 policy: frozen quantization state or runtime provenance changed")
    from bitsandbytes.nn import Linear4bit

    for module in bundle.model.modules():
        if isinstance(module, Linear4bit) and _bf16_linear_input not in module._forward_pre_hooks.values():
            raise ValueError("NF4 policy: explicit BF16 input hook missing")


def build_quantized_policy(model_path: Path, arm: str, protocol: dict, *,
                           device_index: int = 0) -> PolicyBundle:
    """Load original local BF16 source into NF4 on one CUDA device, without offload.

    All Vision/LLM adapters and four Merger masters use the existing policy setup.
    The immutable reference here is adapter-disabled NF4 F0, not the BF16 baseline.
    """
    model_path = Path(model_path)
    _validate_protocol(model_path, arm, protocol, device_index)
    return _build_quantized_policy(model_path, arm, protocol, device_index)


def _build_quantized_policy(model_path: Path, arm: str, protocol: dict, device_index: int) -> PolicyBundle:
    versions = _versions()
    device = _cuda_device(device_index)
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8, llm_int8_skip_modules=[*MERGERS, "lm_head"],
        llm_int8_enable_fp32_cpu_offload=False)
    base = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path),
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": device_index}, quantization_config=config)
    quantized_modules = _inspect_quantized_base(base, device)
    states = quantization_state_sha256(base)
    _register_bf16_inputs(base)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, min_pixels=protocol["image"]["min_pixels"],
        max_pixels=protocol["image"]["max_pixels"])
    # prepare_model_for_kbit_training would promote all BF16 base weights to FP32.
    bundle = configure_policy(base, processor, arm, protocol)
    if (any(p.dtype != torch.float32 for params in bundle.params.values() for p in params)
            or any(p.device != device for p in bundle.model.parameters())):
        raise ValueError("NF4 policy: all adapter/Merger masters must be FP32 on the selected CUDA device")
    bundle.inventory["quantization"] = {"contract": quantization_contract(),
        "versions": versions,
        "modules": quantized_modules, "state_sha256": states,
        "reference": "adapter_disabled_nf4_f0"}
    assert_quantized_base_unchanged(bundle)
    return bundle


def _frozen_state_dict(bundle: PolicyBundle) -> dict[str, torch.Tensor]:
    state = {}
    for name, tensor in bundle.model.get_base_model().state_dict().items():
        if ".lora_" in name or ".modules_to_save." in name:
            continue
        canonical = name.replace(".base_layer.", ".").replace(".original_module.", ".")
        if canonical in state:
            raise ValueError("NF4 cache: duplicate canonical state key")
        state[canonical] = tensor
    if not set(bundle.base_sha256) <= state.keys():
        raise ValueError("NF4 cache: incomplete frozen parameter state")
    for name, components in bundle.inventory["quantization"]["state_sha256"].items():
        for key, row in components.items():
            value = state.get(name + ".weight." + key)
            if (value is None or tensor_sha256(value) != row["sha256"]
                    or list(value.shape) != row["shape"] or str(value.dtype) != row["dtype"]):
                raise ValueError("NF4 cache: missing/changed quantization tensor")
    return state


def save_quantized_base(bundle: PolicyBundle, destination: Path) -> dict:
    """Save immutable NF4 F0 and processor, including original (untrained) Mergers.

    Adapter and optimizer states belong in the separate RL checkpoint. No adapter
    merge or dequantization occurs, and an existing destination is never replaced.
    """
    assert_quantized_base_unchanged(bundle)
    if bundle.processor is None:
        raise ValueError("NF4 cache: complete processor required")
    state = _frozen_state_dict(bundle)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    base = bundle.model.get_base_model()
    from huggingface_hub import save_torch_state_dict

    # Transformers 4.57.1 discards explicit state_dict whenever hf_quantizer exists.
    save_torch_state_dict(state, destination, safe_serialization=True, max_shard_size="2GB",
                          shared_tensors_to_discard=getattr(base, "_tied_weights_keys", None))
    serialized = _serialized_state_sha256(destination)
    if (set(serialized) != set(state)
            or any(serialized[name] != {"sha256": tensor_sha256(value), "shape": list(value.shape),
                                       "dtype": str(value.dtype)} for name, value in state.items())):
        raise ValueError("NF4 cache: serialized state differs from canonical frozen tensors")
    base.config.save_pretrained(str(destination))
    base.generation_config.save_pretrained(str(destination))
    bundle.processor.save_pretrained(str(destination))
    files = list(destination.iterdir())
    if (not files or any(not p.is_file() or p.is_symlink() for p in files)
            or not any(p.suffix == ".safetensors" for p in files)):
        raise ValueError("NF4 cache: complete flat safetensors export required")
    metadata = {"format": _CACHE_FORMAT, "base_id": BASE_ID, "base_revision": BASE_REVISION,
        "quantization": bundle.inventory["quantization"], "base_sha256": bundle.base_sha256,
        "serialized_state_sha256": serialized,
        "files": {p.name: file_hash(p) for p in sorted(files)}}
    wire = json.dumps(metadata, sort_keys=True, allow_nan=False, indent=2) + "\n"
    with (destination / _CACHE_METADATA).open("x", encoding="utf-8") as stream:
        stream.write(wire)
    with (destination / (_CACHE_METADATA + ".sha256")).open("x", encoding="ascii") as stream:
        stream.write(file_hash(destination / _CACHE_METADATA) + "\n")
    assert_quantized_base_unchanged(bundle)
    return metadata


def _serialized_state_sha256(cache: Path) -> dict:
    from safetensors import safe_open

    states = {}
    for path in sorted(cache.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if (name in states or ".lora_" in name or ".modules_to_save." in name
                        or ".original_module." in name or ".base_layer." in name):
                    raise ValueError("NF4 cache: noncanonical/duplicate frozen state key")
                value = stream.get_tensor(name)
                states[name] = {"sha256": tensor_sha256(value), "shape": list(value.shape),
                                "dtype": str(value.dtype)}
    return states


def read_quantized_base_metadata(cache: Path) -> dict:
    """Reject partial exports, foreign identity, path tricks and changed payloads."""
    cache = Path(cache)
    if not cache.is_dir() or cache.is_symlink():
        raise ValueError("NF4 cache: actual standalone directory required")
    paths = list(cache.iterdir())
    if (any(not p.is_file() or p.is_symlink() for p in paths)
            or not (cache / _CACHE_METADATA).is_file()
            or not (cache / (_CACHE_METADATA + ".sha256")).is_file()
            or (cache / (_CACHE_METADATA + ".sha256")).read_text(encoding="ascii").strip()
            != file_hash(cache / _CACHE_METADATA)):
        raise ValueError("NF4 cache: incomplete or changed metadata/files")
    metadata = json.loads((cache / _CACHE_METADATA).read_text(encoding="utf-8"))
    if (set(metadata) != {"format", "base_id", "base_revision", "quantization", "base_sha256",
                         "serialized_state_sha256", "files"}
            or metadata["format"] != _CACHE_FORMAT or metadata["base_id"] != BASE_ID
            or metadata["base_revision"] != BASE_REVISION
            or metadata["quantization"].get("contract") != quantization_contract()
            or metadata["quantization"].get("versions") != _versions()
            or set(metadata["files"]) | {_CACHE_METADATA, _CACHE_METADATA + ".sha256"}
            != {p.name for p in paths}
            or any(Path(name).name != name or file_hash(cache / name) != expected
                   for name, expected in metadata["files"].items())
            or not any(name.endswith(".safetensors") for name in metadata["files"])
            or _serialized_state_sha256(cache) != metadata["serialized_state_sha256"]):
        raise ValueError("NF4 cache: identity/runtime/payload checksum mismatch")
    return metadata


def load_quantized_policy(cache: Path, arm: str, protocol: dict, *,
                          device_index: int = 0) -> PolicyBundle:
    """Rebuild fresh NF4 F0 from a verified cache, before restoring RL trainables."""
    cache = Path(cache)
    metadata = read_quantized_base_metadata(cache)
    _validate_protocol(cache, arm, protocol, device_index, require_snapshot=False)
    bundle = _build_quantized_policy(cache, arm, protocol, device_index)
    if (bundle.base_sha256 != metadata["base_sha256"]
            or bundle.inventory["quantization"] != metadata["quantization"]
            or read_quantized_base_metadata(cache) != metadata):
        raise ValueError("NF4 cache: reconstructed frozen base or quantization differs")
    assert_quantized_base_unchanged(bundle)
    return bundle
