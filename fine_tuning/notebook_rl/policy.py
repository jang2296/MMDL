"""Selective policy gradients with an immutable adapter-disabled F0 reference.

Only generated actions reach completion_logprobs. No gold labels, SFT forward,
in-place merge, quantization, or automatic device/precision fallback is used.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fine_tuning.modules import BASE_ID, BASE_REVISION, inspect_modules, parameter_group, tensor_sha256

ARMS = frozenset(("MOD-FML", "MOD-FVML", "MOD-RV", "MOD-RR", "MOD-RS"))


def enable_deterministic_execution() -> dict:
    """Require strict CUDA reproducibility, never relax gradient comparisons.

    See https://docs.pytorch.org/docs/2.8/notes/randomness.html.
    This must precede the first CUDA/cuBLAS operation in every native worker.
    """
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        if torch.cuda.is_initialized():
            raise ValueError("policy: configure deterministic cuBLAS before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace != ":4096:8":
        raise ValueError("policy: deterministic execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {"algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


class VisionRoute:
    def __init__(self, vision_lora_params: list[nn.Parameter]):
        self.params = vision_lora_params
        self.vision_on = bool(vision_lora_params)

    def apply(self, vision_on: bool) -> None:
        if type(vision_on) is not bool or vision_on and not self.params:
            raise ValueError("policy: route requires an installed Vision adapter")
        self.vision_on = vision_on
        for parameter in self.params:
            parameter.requires_grad_(vision_on)


@dataclass
class PolicyBundle:
    model: Any
    processor: Any
    inventory: dict
    params: dict[str, list[nn.Parameter]]
    route: VisionRoute
    base_sha256: dict[str, str]


def _canonical_frozen(bundle: PolicyBundle) -> Iterator[tuple[str, torch.Tensor]]:
    for name, parameter in bundle.model.get_base_model().named_parameters():
        if ".lora_" in name or ".modules_to_save." in name:
            continue
        yield name.replace(".base_layer.", ".").replace(".original_module.", "."), parameter


def configure_policy(base: Any, processor: Any, arm: str, protocol: dict, *,
                     enforce_counts: bool = True) -> PolicyBundle:
    """Also used with a tiny real Qwen architecture in CPU tests, never tiny weights in GPU runs."""
    from peft import LoraConfig, get_peft_model

    if arm not in ARMS or hasattr(base, "peft_config"):
        raise ValueError("policy: recognized arm and a fresh base required")
    cfg = protocol["model"]["lora"]
    if cfg != {"r": 16, "alpha": 32, "dropout": 0.0, "bias": "none"}:
        raise ValueError("policy: unsupported adapter contract; use a separately reviewed protocol")
    inventory = inspect_modules(base)
    if enforce_counts and (len(inventory["targets"]["L"]) != 252 or len(inventory["targets"]["V"]) != 48
                           or set(inventory["mergers"]) != set(protocol["model"]["modules_to_save"])):
        raise ValueError("policy: actual modules differ from the pinned architecture")
    expected_base = {name: tensor_sha256(p) for name, p in base.named_parameters()}
    base.requires_grad_(False)
    targets = inventory["targets"]["L"] + (inventory["targets"]["V"] if arm != "MOD-FML" else [])
    model: Any = get_peft_model(base, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0,
        bias="none", target_modules=targets, modules_to_save=inventory["mergers"], task_type="CAUSAL_LM"))
    for root in inventory["mergers"]:
        wrapper: Any = model.get_base_model().get_submodule(root)
        wrapper.modules_to_save["default"].float()
        wrapper.original_module.requires_grad_(False)
    params: dict[str, list[nn.Parameter]] = {"L": [], "M": [], "V": []}
    rows, trainable_ids = [], set()
    for name, parameter in model.named_parameters():
        group = parameter_group(name, inventory)
        allowed = ".modules_to_save.default." in name if group == "M" else ".lora_" in name
        parameter.requires_grad_(allowed)
        if allowed:
            if id(parameter) in trainable_ids:
                raise ValueError("policy: overlapping trainable partitions")
            trainable_ids.add(id(parameter))
            params[group].append(parameter)
            rows.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
                         "group": group, "dtype": str(parameter.dtype)})
    counts = {g: sum(p.numel() for p in ps) for g, ps in params.items()}
    expected = dict(protocol["expected_trainable"])
    if arm == "MOD-FML":
        expected["V"] = 0
    if enforce_counts and counts != expected:
        raise ValueError(f"policy: trainable count mismatch {counts}")
    if not params["L"] or not params["M"] or bool(params["V"]) != (arm != "MOD-FML"):
        raise ValueError("policy: empty or unintended partition")
    inventory.update(arm=arm, trainable=rows, trainable_counts=counts,
                     base_id=BASE_ID, base_revision=BASE_REVISION)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    bundle = PolicyBundle(model, processor, inventory, params, VisionRoute(params["V"]), expected_base)
    assert_base_unchanged(bundle)
    return bundle


def build_policy(model_path: Path, arm: str, protocol: dict) -> PolicyBundle:
    from transformers import AutoProcessor

    from fine_tuning.train import load_base

    initial = protocol["initial_checkpoint"]
    if initial["model_id"] != BASE_ID or initial["revision"] != BASE_REVISION:
        raise ValueError("policy: pinned model identity required")
    # File hashes are checked by the deployment gate before this potentially large load.
    base = load_base(model_path)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=protocol["image"]["min_pixels"], max_pixels=protocol["image"]["max_pixels"])
    return configure_policy(base, processor, arm, protocol)


def _logprobs(bundle: PolicyBundle, sample_inputs: dict, completion_len: int, chunk: int) -> torch.Tensor:
    allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if set(sample_inputs) - allowed or not {"input_ids", "attention_mask"} <= sample_inputs.keys():
        raise ValueError("policy: only actual token/image inputs are allowed; no labels or cached states")
    ids, mask = sample_inputs["input_ids"], sample_inputs["attention_mask"]
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1
            or not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or not bool((mask == 1).all())
            or type(completion_len) is not int or not 0 < completion_len < ids.shape[1]
            or type(chunk) is not int or chunk <= 0):
        raise ValueError("policy: unpadded microbatch-one with a nonempty prompt/action suffix required")
    if ("pixel_values" in sample_inputs) != ("image_grid_thw" in sample_inputs):
        raise ValueError("policy: image pixels/grid must be paired")
    base = bundle.model.get_base_model()
    device = next(base.parameters()).device
    if any(v.device != device for v in sample_inputs.values()):
        raise ValueError("policy: explicit input/model device placement required")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast:
        # Call the multimodal body directly to avoid allocating full-vocabulary logits
        # at all prompt positions. Adapter layers and all four merger wrappers remain active.
        hidden = base.model(**sample_inputs, use_cache=False, return_dict=True).last_hidden_state
        start = ids.shape[1] - completion_len
        actions = ids[:, start:]
        states = hidden[:, start - 1:-1, :]
        pieces = []
        def project(h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
            logits = base.lm_head(h).float()
            return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        for offset in range(0, completion_len, chunk):
            h, tokens = states[:, offset:offset + chunk], actions[:, offset:offset + chunk]
            # Without recomputation, chunking would still retain all vocabulary logits
            # for backward. This checkpoint bounds those retained activations.
            lp = checkpoint(project, h, tokens, use_reentrant=False) if torch.is_grad_enabled() else project(h, tokens)
            pieces.append(lp)
        result = torch.cat(pieces, dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("policy: nonfinite action log probability")
    return result


def completion_logprobs(bundle: PolicyBundle, sample_inputs: dict, completion_len: int, *, chunk: int = 1024) -> torch.Tensor:
    bundle.route.apply(bundle.route.vision_on)
    return _logprobs(bundle, sample_inputs, completion_len, chunk)


def reference_logprobs(bundle: PolicyBundle, sample_inputs: dict, completion_len: int, *,
                       chunk: int = 1024) -> torch.Tensor:
    try:
        with torch.no_grad(), bundle.model.disable_adapter():
            return _logprobs(bundle, sample_inputs, completion_len, chunk)
    finally:
        # PEFT re-enables all adapters on exit, including a previously blocked Vision branch.
        bundle.route.apply(bundle.route.vision_on)


def assert_base_unchanged(bundle: PolicyBundle) -> None:
    actual = {name: tensor_sha256(parameter) for name, parameter in _canonical_frozen(bundle)}
    if actual != bundle.base_sha256:
        raise ValueError("policy: frozen base/reference changed")


def merged_named_tensors(bundle: PolicyBundle) -> Iterator[tuple[str, torch.Tensor]]:
    """Independent BF16 tensors; never mutate PEFT/base or reuse an in-place merge."""
    base = bundle.model.get_base_model()
    seen = set()
    for name, parameter in base.named_parameters():
        if ".lora_" in name or ".original_module." in name:
            continue
        canonical = name.replace(".base_layer.", ".").replace(".modules_to_save.default.", ".")
        if canonical in seen or canonical not in bundle.base_sha256:
            raise ValueError("policy: duplicate/unexpected export tensor")
        seen.add(canonical)
        with torch.no_grad():
            value = parameter.detach().float()
            if ".base_layer.weight" in name:
                layer = base.get_submodule(name.removesuffix(".base_layer.weight"))
                if layer.merged:
                    raise ValueError("policy: in-place merged adapter forbidden")
                value = value + layer.get_delta_weight("default").float()
            yield canonical, value.to(dtype=torch.bfloat16).clone()
    if seen != set(bundle.base_sha256):
        raise ValueError("policy: incomplete exported parameter set")
