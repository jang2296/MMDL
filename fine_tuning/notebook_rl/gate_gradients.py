"""G9 gradient-route observations on a caller-owned policy; no optimizer update.

The hard-freeze control directly disables Vision parameter gradients and bypasses
the routing wrapper through policy._logprobs. It is independent of route.apply.
Signed coefficients act only on the supplied generated suffix; no gold or CE.
"""
from __future__ import annotations

import json
import math

import torch

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from . import checkpoint, policy


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G9: " + message)


def _fingerprints(inputs: dict) -> dict:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                   "sha256": tensor_sha256(value)} for name, value in inputs.items()}


def _capture(bundle, names: dict[int, str], retain: set[str]) -> tuple[dict, dict]:
    summary, gradients = {}, {}
    installed = {id(p) for group in bundle.params.values() for p in group}
    _require(all(p.grad is None for p in bundle.model.parameters() if id(p) not in installed),
             "frozen parameter acquired a gradient")
    for group, parameters in bundle.params.items():
        rows, squared_norm = [], 0.
        for parameter in parameters:
            name, grad = names[id(parameter)], parameter.grad
            if grad is None:
                rows.append({"name": name, "grad_none": True})
                continue
            _require(not grad.is_sparse and bool(torch.isfinite(grad).all()), "nonfinite/sparse gradient: " + name)
            value = grad.detach().to(device="cpu", copy=True)
            norm = float(torch.linalg.vector_norm(value.float()))
            _require(math.isfinite(norm), "nonfinite gradient norm: " + name)
            squared_norm += norm**2
            rows.append({"name": name, "grad_none": False, "dtype": str(value.dtype),
                         "shape": list(value.shape), "norm": norm, "sha256": tensor_sha256(value)})
            if group in retain:
                gradients[name] = value
        summary[group] = {"norm": math.sqrt(squared_norm), "parameters": rows,
                          "all_none": all(row["grad_none"] for row in rows),
                          "all_present": bool(rows) and all(not row["grad_none"] for row in rows)}
    return summary, gradients


def _match(expected: dict, actual: dict, *, comparison: str = "GRADIENT") -> dict:
    _require(set(expected) == set(actual) and bool(expected), "incomplete gradient comparison")
    rows = []
    for name in sorted(expected):
        reference, observed = expected[name], actual[name]
        descriptors = {"comparison": comparison, "name": name,
            "expected_shape": list(reference.shape), "actual_shape": list(observed.shape),
            "expected_dtype": str(reference.dtype), "actual_dtype": str(observed.dtype)}
        _require(observed.shape == reference.shape and observed.dtype == reference.dtype,
                 "gradient mismatch: " + json.dumps(descriptors, sort_keys=True))
        difference = (observed.double() - reference.double()).abs()
        relative = difference / torch.where(reference != 0, reference.double().abs(), 1.)
        rows.append({"name": name, "max_abs_error": float(difference.max()),
                     "max_relative_error": float(relative.max()),
                     "zero_reference_nonzero_actual": int(((reference == 0) & (observed != 0)).sum()),
                     "expected_norm": float(torch.linalg.vector_norm(reference.double())),
                     "actual_norm": float(torch.linalg.vector_norm(observed.double()))})
        _require(bool(torch.allclose(observed, reference, rtol=1e-6, atol=0)),
                 "gradient mismatch: " + json.dumps(descriptors | rows[-1], sort_keys=True))
    return {"match": True, "rtol": 1e-6, "atol": 0., "parameters": rows,
            "max_abs_error": max(row["max_abs_error"] for row in rows),
            "max_relative_error": max(row["max_relative_error"] for row in rows)}


def _measure_routes(bundle, sample_inputs: dict, completion_len: int, *, native_cuda: bool = False,
                    chunk: int = 1024) -> dict:
    """Private tiny-CPU seam; native_cuda is reserved for the strict public path."""
    named = dict(bundle.model.named_parameters())
    parameters = list(named.values())
    _require(bool(parameters) and all(p.grad is None for p in parameters), "pre-existing gradients must not be erased")
    _require(not torch.is_inference_mode_enabled(), "autograd cannot run in inference_mode")
    _require(set(sample_inputs) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
             and all(isinstance(value, torch.Tensor) and not value.requires_grad and value.grad is None
                     for value in sample_inputs.values()),
             "detached token/image inputs only; no labels, gold, cached states or input gradients")
    ids = sample_inputs["input_ids"]
    _require(ids.ndim == 2 and ids.shape[0] == 1 and type(completion_len) is int
             and 2 <= completion_len < ids.shape[1], "at least two generated suffix actions required")
    device = parameters[0].device
    _require(all(p.device == device for p in parameters)
             and all(value.device == device for value in sample_inputs.values()), "one explicit model/input device required")
    _require(not native_cuda or device.type == "cuda", "native probe requires actual CUDA policy tensors")
    groups = bundle.params
    installed = [p for group in groups.values() for p in group]
    _require(set(groups) == {"L", "M", "V"} and groups["L"] and groups["M"]
             and len({id(p) for p in installed}) == len(installed)
             and {id(p) for p in installed} <= {id(p) for p in parameters}, "complete disjoint installed partitions required")
    has_vision = bool(groups["V"])
    _require(has_vision == (bundle.inventory["arm"] != "MOD-FML")
             and type(bundle.route.vision_on) is bool
             and {id(p) for p in bundle.route.params} == {id(p) for p in groups["V"]}, "Vision route/arm mismatch")
    _require(all(not p.requires_grad for p in parameters if id(p) not in {id(q) for q in installed})
             and all(p.requires_grad for p in groups["L"] + groups["M"])
             and all(p.requires_grad == bundle.route.vision_on for p in groups["V"]), "initial gradient flags differ from route")
    policy.assert_base_unchanged(bundle)
    names = {id(p): name for name, p in named.items()}
    trainable_before = {names[id(p)]: tensor_sha256(p) for p in installed}
    base_before = dict(bundle.base_sha256)
    fingerprints = _fingerprints(sample_inputs)
    flags = [(p, p.requires_grad) for p in parameters]
    modes = [(module, module.training) for module in bundle.model.modules()]
    route_on = bundle.route.vision_on
    body = bundle.model.get_base_model().model
    had_rope = hasattr(body, "rope_deltas")
    rope = getattr(body, "rope_deltas", None)
    rope_value = rope.detach().clone() if isinstance(rope, torch.Tensor) else None
    rng = checkpoint.capture_rng()

    def clear():
        for parameter in parameters:
            parameter.grad = None

    def restore_rng_and_rope():
        checkpoint.restore_rng(rng)
        if had_rope:
            if isinstance(rope, torch.Tensor):
                with torch.no_grad():
                    rope.copy_(rope_value)
            body.rope_deltas = rope
        elif hasattr(body, "rope_deltas"):
            delattr(body, "rope_deltas")

    def backward(*, hard_freeze=False):
        # The independent control never invokes completion_logprobs/route.apply.
        values = (policy._logprobs(bundle, sample_inputs, completion_len, chunk) if hard_freeze
                  else policy.completion_logprobs(bundle, sample_inputs, completion_len, chunk=chunk))
        coefficients = torch.tensor([1. if i % 2 == 0 else -1. for i in range(completion_len)], device=device)
        objective = (values[0] * coefficients).sum() / completion_len
        _require(bool(torch.isfinite(objective)), "nonfinite signed action objective")
        objective.backward()
        return float(objective.detach())

    def nonvision(summary):
        _require(all(summary[group]["all_present"] and summary[group]["norm"] > 0 for group in ("L", "M")),
                 "LLM/Merger gradients must be finite, present and nonzero")

    try:
        with torch.enable_grad():
            bundle.model.train()
            bundle.route.apply(False)
            restore_rng_and_rope()
            off_loss = backward()
            off, off_gradients = _capture(bundle, names, {"L", "M"})
            _require(off["V"]["all_none"], "route OFF Vision gradient must be None")
            nonvision(off)
            clear()

            # Separate same-path backward repeatability from route-control equivalence.
            bundle.route.apply(False)
            restore_rng_and_rope()
            off_repeat_loss = backward()
            off_repeat, off_repeat_gradients = _capture(bundle, names, {"L", "M"})
            _require(off_repeat["V"]["all_none"], "repeat route OFF Vision gradient must be None")
            nonvision(off_repeat)
            off_repeat_comparison = _match(off_gradients, off_repeat_gradients, comparison="OFF_REPEAT")
            del off_repeat_gradients
            clear()

            # Keep the route flag ON but freeze V parameters directly. The raw
            # scorer bypasses routing, so a broken route wrapper is not the control.
            bundle.route.vision_on = has_vision
            for parameter in groups["V"]:
                parameter.requires_grad_(False)
            restore_rng_and_rope()
            frozen_loss = backward(hard_freeze=True)
            frozen, frozen_gradients = _capture(bundle, names, {"L", "M"})
            _require(frozen["V"]["all_none"], "hard-frozen Vision gradient must be None")
            nonvision(frozen)
            frozen_comparison = _match(off_gradients, frozen_gradients, comparison="OFF_HARD_FROZEN")
            del off_gradients, frozen_gradients
            clear()

            on, mixed, mixed_comparison = None, None, None
            on_loss, mixed_loss = None, None
            if has_vision:
                bundle.route.apply(True)
                restore_rng_and_rope()
                on_loss = backward()
                on, on_gradients = _capture(bundle, names, {"V"})
                _require(on["V"]["all_present"] and on["V"]["norm"] > 0, "route ON Vision gradients must be nonzero")
                clear()
                bundle.route.apply(True)
                restore_rng_and_rope()
                mixed_loss = backward()
                bundle.route.apply(False)
                mixed_loss += backward()
                mixed, mixed_gradients = _capture(bundle, names, {"V"})
                mixed_comparison = _match(on_gradients, mixed_gradients)
                nonvision(mixed)
            report = {"vision_off_grad_none": off["V"]["all_none"],
                "nonvision_grad_finite_nonzero": all(off[g]["all_present"] and off[g]["norm"] > 0 for g in ("L", "M")),
                "hard_freeze_gradient_match": frozen_comparison["match"],
                "off_repeat_gradient_match": off_repeat_comparison["match"],
                "vision_on_grad_nonzero": on["V"]["norm"] > 0 if on is not None else None,
                "mixed_on_only": mixed_comparison["match"] if mixed_comparison is not None else None,
                "vision_applicability": "MEASURED" if has_vision else "NOT_APPLICABLE_MOD_FML",
                "vision_on_status": "MEASURED" if has_vision else "NOT_APPLICABLE_MOD_FML",
                "mixed_status": "MEASURED" if has_vision else "NOT_APPLICABLE_MOD_FML",
                "vision_na_reason": None if has_vision else "NO_INSTALLED_VISION_ADAPTERS",
                "vision_parameter_count": sum(p.numel() for p in groups["V"]),
                "device": str(device), "cuda_policy_execution": device.type == "cuda",
                "pinned_architecture_counts": False,
                "training_allowed": False, "gpu_gates_certified": False, "optimizer_updated": False,
                "objective": "fixed alternating signed generated-action logprobs / suffix length",
                "hard_freeze_control": "direct V requires_grad=False; policy._logprobs bypasses route wrapper",
                "completion_len": completion_len, "input_fingerprints": fingerprints,
                "off": off, "off_repeat": off_repeat, "hard_frozen": frozen, "on": on, "mixed": mixed,
                "objective_values": {"off": off_loss, "off_repeat": off_repeat_loss,
                                     "hard_frozen": frozen_loss, "on": on_loss, "mixed": mixed_loss},
                "off_repeat_comparison": off_repeat_comparison,
                "hard_freeze_comparison": frozen_comparison, "mixed_comparison": mixed_comparison,
                "limitations": ["Signed synthetic action objective, not a training update or learned routing result",
                    "One caller-supplied image/action suffix; no data eligibility or complete GPU gate approval",
                    "CPU helper fixtures do not establish pretrained 4B or CUDA behavior"]}
    finally:
        clear()
        bundle.route.vision_on = route_on
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training
        restore_rng_and_rope()
        policy.assert_base_unchanged(bundle)
        _require(bundle.base_sha256 == base_before, "frozen hash binding changed")
        _require({names[id(p)]: tensor_sha256(p) for p in installed} == trainable_before,
                 "installed parameter values changed during gradient measurement")
        _require(_fingerprints(sample_inputs) == fingerprints, "input tensors changed during gradient measurement")
    report.update(hf_base_unchanged=True, installed_parameters_unchanged=True,
                  temporary_gradients_cleared=True, caller_state_restored=True)
    return report


def measure_gradient_routes(bundle, sample_inputs: dict, completion_len: int, *, protocol: dict) -> dict:
    """G9 component for a live pinned CUDA policy; allocates no model or optimizer."""
    from .gate_weights import _pinned_config
    from .gpu_gates import measure_inventory

    _require(type(bundle) is policy.PolicyBundle, "actual PolicyBundle required")
    _require(all(p.grad is None for p in bundle.model.parameters()), "pre-existing gradients must not be erased")
    devices = {p.device for p in bundle.model.parameters()}
    _require(len(devices) == 1 and next(iter(devices)).type == "cuda", "actual CUDA policy required; no CPU fallback")
    _require(bundle.inventory["base_id"] == BASE_ID and bundle.inventory["base_revision"] == BASE_REVISION,
             "pinned model identity required")
    _pinned_config(bundle.model.config)
    inventory = measure_inventory(bundle, protocol)
    _require(inventory["pinned_architecture_counts"], "pinned 4B live architecture required")
    result = _measure_routes(bundle, sample_inputs, completion_len, native_cuda=True,
                             chunk=protocol["model"]["logprob_chunk"])
    result.update(model_id=BASE_ID, model_revision=BASE_REVISION,
                  pinned_architecture_counts=True, arm=bundle.inventory["arm"])
    return result
