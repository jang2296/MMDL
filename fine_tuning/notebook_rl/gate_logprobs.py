"""G7 measurements from generated sampler actions and the actual HF policy.

One invocation measures one stage, not both sides of a synchronization. Engine-
validated sampler returns are journaled before the G7 comparison. This does not change
protocol thresholds, train weights, approve data, or allocate another engine.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Callable

import torch

from . import policy
from .checkpoint import capture_rng, restore_rng
from .engines import RolloutRequest, SamplingSettings, _completion
from .gate_reference import _hashes, _input_fingerprint
from .losses import mis_mask


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G7: " + message)


def _distribution(values: list[float]) -> dict:
    tensor = torch.tensor(values, dtype=torch.float64)
    _require(bool(values) and bool(torch.isfinite(tensor).all()), "finite nonempty distribution required")
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": float(tensor.mean()), "p50": float(torch.quantile(tensor, .5)),
            "p95": float(torch.quantile(tensor, .95)), "p995": float(torch.quantile(tensor, .995))}


def _limits(threshold: float, delta_seq: float) -> None:
    _require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 < threshold <= .02
             and type(delta_seq) in (int, float) and math.isfinite(delta_seq) and 0 <= delta_seq <= .05,
             "explicit bounded probability thresholds required")


def _summarize(rows: list[dict], threshold: float, delta_seq: float) -> dict:
    _limits(threshold, delta_seq)
    _require(len(rows) >= 32, "at least 32 actual trajectories required")
    absolute, means, sums, per_trajectory = [], [], [], []
    masked = 0
    for row in rows:
        old = torch.tensor(row["hf_logprobs"], dtype=torch.float64).reshape(1, -1)
        sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float64).reshape(1, -1)
        _require(old.shape == sampler.shape and old.numel() == len(row["completion_token_ids"])
                 and old.numel() > 0 and bool(torch.isfinite(old).all()) and bool(torch.isfinite(sampler).all())
                 and bool((old <= 1e-5).all()) and bool((sampler <= 1e-5).all()),
                 "finite, action-aligned log probabilities required")
        difference = old - sampler
        values = difference.abs().flatten().tolist()
        mean_abs = sum(values) / len(values)
        _require(mean_abs <= threshold, "trajectory mean absolute mismatch exceeds predeclared threshold")
        keep, mean_delta = mis_mask(old, sampler, torch.ones_like(old, dtype=torch.bool), delta_seq)
        masked += int(not bool(keep[0]))
        means.append(float(mean_delta[0]))
        sums.append(float(difference.sum()))
        absolute.extend(values)
        per_trajectory.append({"request_id": row["request_id"], "seed": row["seed"],
            "tokens": len(values), "mean_abs": mean_abs, "max_abs": max(values),
            "sequence_sum_delta": sums[-1], "sequence_mean_delta": means[-1], "mis_kept": bool(keep[0])})
    synthetic = torch.full((1, 5), -2.0, dtype=torch.float64)
    synthetic_keep, _ = mis_mask(synthetic, synthetic - .1, torch.ones_like(synthetic, dtype=torch.bool), delta_seq)
    _require(not bool(synthetic_keep[0]), "0.1-nat synthetic mismatch was not masked")
    mean_distribution = _distribution([abs(value) for value in means])
    return {"trajectories": len(rows), "tokens": len(absolute), "mean_abs": sum(absolute) / len(absolute),
        "max_abs": max(absolute), "token_abs_delta": _distribution(absolute),
        "sequence_sum_delta": _distribution(sums), "sequence_mean_abs_delta": mean_distribution,
        "delta_seq": delta_seq, "delta_seq_candidate": min(.05, 2 * mean_distribution["p995"]),
        "quantile_method": "linear_float64", "masked_fraction": masked / len(rows),
        "synthetic_0_1_masked": True, "per_trajectory": per_trajectory}


def _measure_logprobs(bundle, engine, requests: list[RolloutRequest], *, threshold: float,
                      delta_seq: float, record: Callable[[dict], None]) -> dict:
    """Private CPU seam: real tiny HF sampling is possible, never native evidence."""
    _limits(threshold, delta_seq)
    _require(callable(record) and bool(requests) and len({r.request_id for r in requests}) == len(requests)
             and sum(r.n for r in requests) >= 32 and {len(r.images) for r in requests} == {1, 2},
             "journal, distinct requests, 32 trajectories and single/two-image coverage required")
    _require(engine._synchronized is True, "completed sampler synchronization required")
    parameters = list(bundle.model.parameters())
    _require(all(p.grad is None for p in parameters), "clean gradient boundary required")
    version = engine.policy_version
    device = next(bundle.model.parameters()).device
    flags = [(p, p.requires_grad) for p in parameters]
    modes = [(m, m.training) for m in bundle.model.modules()]
    body = bundle.model.get_base_model().model
    rope = body.rope_deltas
    rope_value = rope.detach().clone() if isinstance(rope, torch.Tensor) else None
    route, rng = bundle.route.vision_on, capture_rng()
    before = _hashes(bundle.model)
    policy.assert_base_unchanged(bundle)
    rows = []
    input_tensors: dict[int, torch.Tensor] = {}
    try:
        for request in requests:
            # Freeze the prepared prompt before generating; retry IDs stay literal.
            prepared_ids = list(engine.prepare(request).token_ids)
            try:
                results = engine.sample_with_settings([request], SamplingSettings())
            except BaseException as exc:
                # Native engine checks happen before its public return. Do not
                # claim we recovered a malformed backend payload we never saw.
                record({"kind": "G7_SAMPLER_ERROR", "policy_version": version,
                        "request_id": request.request_id, "error_type": type(exc).__name__,
                        "backend_invalid_payload_preserved": False})
                raise
            record({"kind": "G7_ENGINE_RETURN", "policy_version": version,
                    "request_id": request.request_id, "seeds": list(request.seeds),
                    "returns": [asdict(result) for result in results]})
            _require(len(results) == 1 and results[0].request_id == request.request_id
                     and results[0].prompt_token_ids == prepared_ids
                     and len(results[0].completions) == request.n,
                     "actual sampler prompt/request/completion count mismatch")
            for seed, completion in zip(request.seeds, results[0].completions, strict=True):
                _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
                _require(len(completion.token_ids) <= request.max_new_tokens, "completion exceeds reserved budget")
                inputs = engine.model_inputs(request, completion.token_ids, device)
                fingerprint = _input_fingerprint(inputs, len(completion.token_ids), device)
                # Retain only the current action's tensors, not 32 high-resolution
                # inputs on the GPU. Earlier inputs already passed the clean check.
                input_tensors = {id(t): t for t in inputs.values()}
                _require(inputs["input_ids"][0].tolist() == prepared_ids + completion.token_ids,
                         "HF input differs from actual sampled prompt plus actions")
                with torch.no_grad():
                    hf = policy.completion_logprobs(bundle, inputs, len(completion.token_ids))[0].detach().cpu().tolist()
                _require(_input_fingerprint(inputs, len(completion.token_ids), device) == fingerprint,
                         "HF forward mutated inputs")
                row = {"request_id": request.request_id, "seed": seed, "policy_version": version,
                    "prompt_token_ids": prepared_ids, "completion_token_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason, "sampler_logprobs": list(completion.logprobs),
                    "hf_logprobs": hf, "input": fingerprint}
                record({"kind": "G7_HF_COMPARISON", **row})
                rows.append(row)
        _require(engine.policy_version == version and engine._synchronized is True,
                 "sampler version/synchronization changed during measurement")
        _require(all(p.grad is None and p.requires_grad == flag for p, flag in flags)
                 and bundle.route.vision_on == route, "forward changed gradient/route state")
        _require(_hashes(bundle.model) == before, "forward changed policy parameters")
        policy.assert_base_unchanged(bundle)
        result = _summarize(rows, threshold, delta_seq)
    finally:
        for tensor in input_tensors.values():
            tensor.grad = None
            tensor.requires_grad_(False)
        for p, flag in flags:
            p.grad = None
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        bundle.route.vision_on = route
        if isinstance(rope, torch.Tensor):
            with torch.no_grad():
                rope.copy_(rope_value)
        body.rope_deltas = rope
        restore_rng(rng)
    return result | {"policy_version": version, "device": str(device), "rows": rows,
        "native_execution": False, "training_allowed": False, "gpu_gates_certified": False,
        "thresholds_modified": False, "weights_updated": False,
        "raw_return_scope": "ENGINE_VALIDATED_RETURNS_ONLY"}


def measure_logprobs(bundle, engine, requests: list[RolloutRequest], protocol: dict, *,
                     stage: str, record: Callable[[dict], None]) -> dict:
    """Measure INITIAL or POST_SYNC from the caller-owned pinned native engine."""
    from .gate_weights import _pinned_config, require_idle_engine
    from .gpu_gates import measure_inventory

    _require(stage in ("INITIAL", "POST_SYNC"), "explicit stage required")
    measure_inventory(bundle, protocol)
    _pinned_config(bundle.model.config)
    runtime = require_idle_engine(engine, require_push=stage == "POST_SYNC")
    device = next(bundle.model.parameters()).device
    _require(device.type == "cuda" and str(device) == runtime["device"]
             and (runtime["policy_version"] == 0 if stage == "INITIAL" else runtime["policy_version"] > 0),
             "same actual CUDA device and stage-specific policy version required")
    result = _measure_logprobs(bundle, engine, requests,
        threshold=protocol["engine"]["probe_max_mean_abs_logprob_diff"],
        delta_seq=protocol["loss"]["mismatch"]["delta_seq"], record=record)
    _require(require_idle_engine(engine, require_push=stage == "POST_SYNC") == runtime,
             "native runtime changed during measurement")
    result.update(stage=stage, runtime=runtime, native_execution=True,
        limitations=["Single-stage measurement; both INITIAL and POST_SYNC must be independently bound",
                     "Initial delta_seq_candidate is evidence for explicit protocol freezing, not an automatic edit",
                     "Engine-internal validation failures retain an error journal, not unavailable backend payloads",
                     "No full GPU/data approval or learned self-correction claim"])
    if stage != "INITIAL":
        result.pop("delta_seq_candidate")
    result["step0_mean_abs" if stage == "INITIAL" else "post_sync_mean_abs"] = result.pop("mean_abs")
    return result
