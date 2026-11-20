"""G3 fresh-F0 comparison at the actual completion-prediction positions.

Observes the production policy/reference lm_head calls; the independent raw HF
forward supplies all vocabulary logits at those same positions. No model loading,
optimizer update, label loss, sampling or approval happens in this component.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch

from fine_tuning.modules import tensor_sha256
from . import policy
from .checkpoint import capture_rng, restore_rng


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G3: " + message)


def _input_fingerprint(inputs: dict, length: int, device: torch.device) -> dict:
    _require(set(inputs) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"},
             "only actual image/token inputs allowed, never labels or cache")
    _require(all(isinstance(t, torch.Tensor) and t.device == device and not t.requires_grad
                 and t.grad is None for t in inputs.values()), "explicit input device and no input gradients required")
    ids, mask, grid, pixels = (inputs[k] for k in
                             ("input_ids", "attention_mask", "image_grid_thw", "pixel_values"))
    _require(ids.dtype == torch.long and ids.ndim == 2 and ids.shape[0] == 1
             and mask.shape == ids.shape and bool((mask == 1).all())
             and type(length) is int and 1 <= length < ids.shape[1]
             and grid.dtype == torch.long and grid.ndim == 2 and grid.shape[1] == 3
             and grid.shape[0] in (1, 2) and bool((grid > 0).all()) and bool((grid[:, 0] == 1).all())
             and pixels.ndim == 2 and pixels.dtype in (torch.float32, torch.bfloat16)
             and bool(torch.isfinite(pixels).all()), "unpadded still-image completion probe required")
    return {"completion_len": length, "tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype),
        "sha256": tensor_sha256(t)} for k, t in sorted(inputs.items())}}


def _hashes(model) -> dict:
    return {name: tensor_sha256(p) for name, p in model.named_parameters()}


def _compare(expected: torch.Tensor, actual: torch.Tensor, name: str) -> dict:
    _require(expected.shape == actual.shape and expected.dtype == actual.dtype
             and bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all()),
             name + " tensor shape/dtype/finite mismatch")
    before, after = tensor_sha256(expected), tensor_sha256(actual)
    difference = float((expected - actual).abs().max())
    _require(before == after, name + " differs from fresh F0")
    return {"shape": list(expected.shape), "dtype": str(expected.dtype), "max_abs": difference,
            "expected_sha256": before, "actual_sha256": after, "bit_equal": True}


def _observed_logits(bundle, inputs: dict, length: int, *, reference: bool):
    values = []
    def observe(module, args, output):
        values.append(output.detach().float().cpu())
    handle = bundle.model.get_base_model().lm_head.register_forward_hook(observe)
    try:
        with torch.no_grad():
            lp = (policy.reference_logprobs(bundle, inputs, length) if reference else
                  policy.completion_logprobs(bundle, inputs, length, chunk=1024))
        _require(bool(values), "actual production lm_head was not observed")
        return torch.cat(values, dim=1), lp.detach().cpu()
    finally:
        handle.remove()


def _measure_reference(bundle, fresh, probes: list[tuple[str, dict, int]], *, initial: bool) -> dict:
    """Private tiny-model seam; never certifies a pretrained model or GPU run."""
    from transformers import Qwen3VLForConditionalGeneration

    base = bundle.model.get_base_model()
    _require(type(initial) is bool and type(fresh) is Qwen3VLForConditionalGeneration
             and fresh is not base and not hasattr(fresh, "peft_config"), "separate unadapted fresh HF model required")
    _require(len(probes) == 2 and all(isinstance(p[0], str) and p[0] for p in probes)
             and len({p[0] for p in probes}) == 2, "two distinct bound probes required")
    device = next(base.parameters()).device
    _require(all(p.device == device and p.grad is None for m in (base, fresh) for p in m.parameters()),
             "same explicit device and a clean gradient boundary required")
    if hasattr(fresh, "hf_device_map"):
        _require(not fresh.hf_device_map, "no offloaded fresh model")
    fingerprints = {pid: _input_fingerprint(inputs, length, device) for pid, inputs, length in probes}
    _require(sorted(inputs["image_grid_thw"].shape[0] for _, inputs, _ in probes) == [1, 2],
             "single- and two-image probes required")
    fresh_hashes, policy_hashes = _hashes(fresh), _hashes(bundle.model)
    _require(fresh_hashes == bundle.base_sha256, "fresh weights differ from original reference")
    policy.assert_base_unchanged(bundle)
    parameters = [p for model in (bundle.model, fresh) for p in model.parameters()]
    gradient_tensors = parameters + [t for _, inputs, _ in probes for t in inputs.values()]
    flags = [(p, p.requires_grad) for p in parameters]
    modes = [(m, m.training) for model in (bundle.model, fresh) for m in model.modules()]
    rope = [(body, body.rope_deltas,
             body.rope_deltas.detach().clone() if isinstance(body.rope_deltas, torch.Tensor) else None)
            for body in (base.model, fresh.model)]
    rng, route = capture_rng(), bundle.route.vision_on
    reference_binding = dict(bundle.base_sha256)
    rows = []
    try:
        fresh.train(bundle.model.training)
        for pid, inputs, length in probes:
            positions = torch.arange(inputs["input_ids"].shape[1] - length - 1,
                                     inputs["input_ids"].shape[1] - 1, device=device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with torch.no_grad(), autocast:
                # Standard independent HF forward, not adapter disabling or the RL helper.
                logits = fresh(**inputs, logits_to_keep=positions, use_cache=False, return_dict=True).logits.float()
                lp = logits.log_softmax(-1).gather(-1, inputs["input_ids"][:, -length:].unsqueeze(-1)).squeeze(-1)
                expected, expected_lp = logits.cpu(), lp.cpu()
            del logits, lp
            measured, measured_lp = _observed_logits(bundle, inputs, length, reference=True)
            row = {"id": pid, "input": fingerprints[pid], "reference_logits": _compare(expected, measured, "reference logits"),
                   "reference_logprobs": _compare(expected_lp, measured_lp, "reference log probabilities")}
            if initial:
                actual, actual_lp = _observed_logits(bundle, inputs, length, reference=False)
                row["policy_logits"] = _compare(expected, actual, "initial policy logits")
                row["policy_logprobs"] = _compare(expected_lp, actual_lp, "initial policy log probabilities")
            _require(bundle.route.vision_on == route and all(p.requires_grad == flag for p, flag in flags),
                     "reference context did not restore requires_grad/route")
            rows.append(row)
        _require(all(t.grad is None for t in gradient_tensors), "unexpected gradients left by forward measurement")
        _require(fingerprints == {pid: _input_fingerprint(inputs, length, device) for pid, inputs, length in probes},
                 "probe inputs mutated during measurement")
        _require(fresh_hashes == _hashes(fresh) and policy_hashes == _hashes(bundle.model)
                 and reference_binding == bundle.base_sha256, "forward mutated parameters/reference binding")
        policy.assert_base_unchanged(bundle)
    finally:
        # The entry boundary requires None: only gradients created here are cleared.
        for tensor in gradient_tensors:
            tensor.grad = None
        for p, flag in flags:
            p.requires_grad_(flag)
        bundle.route.vision_on = route
        for module, mode in modes:
            module.training = mode
        for body, old, value in rope:
            if isinstance(old, torch.Tensor):
                with torch.no_grad():
                    old.copy_(value)
            body.rope_deltas = old
        restore_rng(rng)
    return {"max_abs": 0.0, "frozen_sha_unchanged": True, "requires_grad_restored": True,
            "probe_count": len(rows), "probes": rows, "device": str(device), "base_sha256": reference_binding,
            "native_execution": False, "training_allowed": False, "gpu_gates_certified": False}


def measure_reference(bundle, fresh, probes: list[tuple[str, dict, int]], protocol: dict,
                      *, stage: str, engine=None) -> dict:
    """Measure only INITIAL or POST_SYNC; the runner must bind both real stages.

    Caller owns and loads the separate, file-verified F0. This does not manufacture
    a second phase from a single successful measurement or modify any weights.
    """
    from .gate_weights import _pinned_config, require_idle_engine
    from .gpu_gates import measure_inventory

    _require(stage in ("INITIAL", "POST_SYNC"), "explicit measurement stage required")
    measure_inventory(bundle, protocol)
    _pinned_config(bundle.model.config)
    _pinned_config(fresh.config)
    device = next(bundle.model.parameters()).device
    _require(device.type == "cuda" and device.index is not None
             and all(p.dtype == torch.bfloat16 for p in fresh.parameters()), "actual pinned BF16 CUDA models required")
    runtime = None
    if stage == "POST_SYNC":
        runtime = require_idle_engine(engine)
        _require(runtime["policy_version"] > 0 and engine._synchronized is True
                 and runtime["device"] == str(device), "positive completed synchronization on the same device required")
    else:
        _require(engine is None, "initial reference check precedes rollout engine allocation")
    result = _measure_reference(bundle, fresh, probes, initial=stage == "INITIAL")
    if runtime is not None:
        _require(require_idle_engine(engine) == runtime, "engine changed during reference measurement")
    result.update(stage=stage, native_execution=True, runtime=runtime,
        limitations=["All vocabulary logits at probe completion-prediction positions, not every prompt position",
                     "Caller must bind fresh checkpoint files and INITIAL/POST_SYNC stages in the full gate runner",
                     "No performance improvement, optimizer update, or complete GPU/training approval"])
    result["initial_max_abs" if stage == "INITIAL" else "post_sync_max_abs"] = result.pop("max_abs")
    return result
