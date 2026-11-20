"""G4 evidence around one real reward-based update of a caller-owned policy.

This operation advances weights, AdamW and scheduler once. It neither rolls back
an update nor synchronizes the sampler, including when a post-update check fails.
"""
from __future__ import annotations

import math

import torch

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from fine_tuning.train import parameter_probes, probe_indices
from . import checkpoint, policy, updates


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G4: " + message)


def _base_hashes(bundle) -> dict:
    result = {name: tensor_sha256(p) for name, p in policy._canonical_frozen(bundle)}
    _require(result == bundle.base_sha256, "frozen base/reference changed")
    return result


def _masters(bundle) -> dict:
    installed = checkpoint._parameters(bundle)
    rows = {row["name"]: row for row in bundle.inventory["trainable"] if row["group"] == "M"}
    _require(len(rows) == 24 and {id(installed[n]) for n in rows} == {id(p) for p in bundle.params["M"]},
             "complete 24-tensor Merger master partition required")
    roots = bundle.inventory["mergers"]
    result = {}
    for name, row in rows.items():
        p = installed[name]
        matches = [root for root in roots if name.removeprefix("base_model.model.").startswith(
            root + ".modules_to_save.default.")]
        _require(len(matches) == 1 and p.requires_grad and p.dtype == torch.float32,
                 "distinct active FP32 Merger master required")
        indices = probe_indices(p.numel(), p.device)
        result[name] = {"root": matches[0], "shape": list(p.shape), "dtype": str(p.dtype),
            "numel": p.numel(), "device": str(p.device), "sha256": tensor_sha256(p),
            "probe_indices": indices.cpu().tolist(), "probe_sha256": tensor_sha256(p.detach().flatten()[indices])}
    _require(len(roots) == 4 and all(sum(row["root"] == root for row in result.values()) == 6 for root in roots),
             "all four six-tensor Merger masters required")
    return result


def _measure_update(bundle, optimizer, scheduler, trees, images, preparation, protocol: dict,
                    algorithm: str, *, step: int, seed: int, vision_disabled: bool = False) -> dict:
    """Private actual tiny-CPU seam; never a pretrained-model/GPU certificate."""
    _require(type(step) is int and step >= 0 and type(seed) is int, "integer update cursor/seed required")
    _require(not torch.is_inference_mode_enabled() and torch.is_grad_enabled(), "actual autograd context required")
    named = dict(bundle.model.named_parameters())
    _require(bool(named) and all(p.grad is None for p in named.values()),
             "pre-existing gradients must not be erased")
    devices = {p.device for p in named.values()}
    _require(len(devices) == 1, "single explicit policy device required")
    checkpoint._validate_optimizer(bundle, optimizer, scheduler, protocol, step)
    master_before, base_before = _masters(bundle), _base_hashes(bundle)
    installed = {id(p) for ps in bundle.params.values() for p in ps}
    # A temporarily route-OFF adapter is not a frozen base probe.
    probes = {name: probe for name, probe in parameter_probes(bundle.model, bundle.inventory).items()
              if id(named[name]) not in installed}
    _require(bool(probes), "fixed frozen probes required")
    binding_before = dict(bundle.base_sha256)
    state_before = {name: float(optimizer.state[p]["step"]) if p in optimizer.state else 0.
                    for name, p in named.items() if id(p) in installed}
    # Existing production RL loss, routing, MIS masks, backward and optimizer path.
    update = updates.optimizer_step(bundle, optimizer, scheduler, trees, images, preparation,
        protocol, algorithm, step=step, seed=seed, vision_disabled=vision_disabled)
    master_after, base_after = _masters(bundle), _base_hashes(bundle)
    _require(bundle.base_sha256 == binding_before and base_after == base_before, "frozen hash binding changed")
    frozen_probes = {}
    for name, probe in probes.items():
        p = named[name]
        observed = tensor_sha256(p.detach().flatten()[probe["indices"]])
        frozen_probes[name] = {"indices": probe["indices"].cpu().tolist(),
                               "before_sha256": probe["sha256"], "after_sha256": observed}
        _require(observed == probe["sha256"] and not p.requires_grad, "fixed frozen probe changed: " + name)
    _require(update["status"] == "UPDATED_UNSYNCHRONIZED" and update["step_before"] == step
             and update["step_after"] == step + 1 and update["engine_sync_required"] is True,
             "no completed actual optimizer update; no-op is not G4 evidence")
    checkpoint._validate_optimizer(bundle, optimizer, scheduler, protocol, step + 1)
    _require(all(p.grad is None for p in bundle.model.parameters()), "update left uncleared gradients")
    _require(math.isfinite(update["grad_norm"]["M"]) and update["grad_norm"]["M"] > 0,
             "Merger update needs finite nonzero actual gradients")
    _require(set(master_after) == set(master_before), "Merger inventory changed during update")
    changed, probe_changed, optimizer_steps = [], [], {}
    for name, before in master_before.items():
        after = master_after[name]
        _require({k: v for k, v in before.items() if k not in ("sha256", "probe_sha256")}
                 == {k: v for k, v in after.items() if k not in ("sha256", "probe_sha256")},
                 "Merger descriptor changed: " + name)
        p = named[name]
        _require(p in optimizer.state and float(optimizer.state[p]["step"]) == state_before[name] + 1,
                 "Merger AdamW counter did not advance exactly once: " + name)
        optimizer_steps[name] = {"before": state_before[name], "after": float(optimizer.state[p]["step"])}
        if before["sha256"] != after["sha256"]:
            changed.append(name)
        if before["probe_sha256"] != after["probe_sha256"]:
            probe_changed.append(name)
    _require(bool(changed) and bool(probe_changed), "FP32 Merger masters/probes did not change")
    return {"merger_changed": bool(changed) and bool(probe_changed), "frozen_unchanged": base_after == base_before,
        "status": "UPDATED_UNSYNCHRONIZED", "update": update, "engine_sync_required": True,
        "device": str(next(iter(devices))), "cuda_policy_execution": next(iter(devices)).type == "cuda",
        "pinned_architecture_counts": False, "master_dtype": "float32", "merger_count": 4,
        "master_tensor_count": len(master_before), "master_numel": sum(row["numel"] for row in master_before.values()),
        "master_before": master_before, "master_after": master_after, "changed_tensors": changed,
        "changed_probe_tensors": probe_changed, "frozen_probes": frozen_probes,
        "master_optimizer_steps": optimizer_steps, "algorithm": algorithm, "seed": seed,
        "protocol_sha256": protocol["sha256"],
        "base_before": base_before, "base_after": base_after,
        "training_allowed": False, "gpu_gates_certified": False,
        "limitations": ["One actual caller-owned RL optimizer update; weights/state are not rolled back on failure",
            "Sampler remains unsynchronized; this is not G3 post-sync or G5 evidence",
            "Frozen probes are bounded samples; frozen base/reference hashes cover every canonical tensor",
            "CPU tiny fixtures do not establish pretrained 4B or CUDA behavior"]}


def measure_update(bundle, optimizer, scheduler, trees, images, preparation, protocol: dict,
                   algorithm: str, *, step: int, seed: int, vision_disabled: bool = False) -> dict:
    """Perform and measure ONE real update on a live pinned CUDA policy, without sync."""
    from .gate_weights import _pinned_config
    from .gpu_gates import measure_inventory

    _require(type(bundle) is policy.PolicyBundle, "actual PolicyBundle required")
    devices = {p.device for p in bundle.model.parameters()}
    _require(len(devices) == 1 and next(iter(devices)).type == "cuda", "actual CUDA policy required; no CPU fallback")
    _require(bundle.inventory["base_id"] == BASE_ID and bundle.inventory["base_revision"] == BASE_REVISION,
             "pinned model identity required")
    _pinned_config(bundle.model.config)
    inventory = measure_inventory(bundle, protocol)
    _require(inventory["pinned_architecture_counts"], "pinned 4B live architecture required")
    result = _measure_update(bundle, optimizer, scheduler, trees, images, preparation, protocol, algorithm,
                             step=step, seed=seed, vision_disabled=vision_disabled)
    result.update(model_id=BASE_ID, model_revision=BASE_REVISION, arm=bundle.inventory["arm"],
                  pinned_architecture_counts=True)
    return result
