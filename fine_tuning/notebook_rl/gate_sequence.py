"""Remaining live gate stages around the session's one shared RL update.

These functions measure existing components; they never grant data, spending or
training approval. Fixed short probes are not model-accuracy evaluations.
"""
from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path

from . import checkpoint, gate_cache, gate_gradients, gate_images, gate_logprobs
from . import gate_numerics, gate_reference, gate_tokens, gate_weights, gpu_gates, launch
from .rollout import seeds


def _record(name, value, record, should_stop):
    record({"kind": "GPU_GATE_COMPONENT", "gate": name, "measurement": value})
    if should_stop():
        raise TimeoutError("gate sequence deadline after " + name)
    return value


def _reference(bundle, preparation, probes, model_path, protocol, *, stage, engine=None):
    import torch

    from fine_tuning.train import load_base

    rng = checkpoint.capture_rng()
    fresh, inputs = None, []
    device = next(bundle.model.parameters()).device
    if device.type != "cuda":
        raise ValueError("gate sequence: actual CUDA required; no CPU reference fallback")
    manifest = launch.read_json(launch.ROOT/protocol["initial_checkpoint"]["manifest"])
    weight_bytes = sum(row["bytes"] for row in manifest["files"] if row["path"].endswith(".safetensors"))
    if torch.cuda.mem_get_info(device)[0] <= weight_bytes:
        raise ValueError("gate sequence: no room for separate fresh F0; do not offload or reduce KV")
    try:
        fresh = load_base(Path(model_path)).to(device)
        fresh.requires_grad_(False)
        inputs = [(p.request.request_id, preparation.model_inputs(p.request, p.completion_ids, device),
                   len(p.completion_ids)) for p in probes]
        return gate_reference.measure_reference(bundle, fresh, inputs, protocol, stage=stage, engine=engine)
    finally:
        inputs.clear()
        del fresh
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint.restore_rng(rng)


def before_engine(bundle, preparation, probes, model_path, protocol, *, record, should_stop):
    if should_stop():
        raise TimeoutError("gate sequence deadline before allocation")
    measured = {}
    measured["G1"] = _record("G1", gpu_gates.measure_inventory(bundle, protocol), record, should_stop)
    measured["G3_INITIAL"] = _record("G3_INITIAL", _reference(bundle, preparation, probes, model_path,
        protocol, stage="INITIAL"), record, should_stop)
    device = next(bundle.model.parameters()).device
    probe = next(p for p in probes if len(p.request.images) == 2)
    inputs = preparation.model_inputs(probe.request, probe.completion_ids, device)
    try:
        measured["G9"] = _record("G9", gate_gradients.measure_gradient_routes(bundle, inputs,
            len(probe.completion_ids), protocol=protocol), record, should_stop)
    finally:
        del inputs
    measured["G10"] = _record("G10", gate_numerics.measure_numerics(device), record, should_stop)
    return measured


def _logprob_requests(probes, version, seed):
    # 32 actual short trajectories at each stage; not a training batch override.
    return [replace(p.request, request_id=p.request.request_id+f":G7:{version}", n=16,
        seeds=seeds(p.request.request_id, version, 1, "G7", 16, master_seed=seed)) for p in probes]


def before_update(bundle, engine, probes, protocol, runtime, *, seed, record, should_stop):
    if should_stop():
        raise TimeoutError("gate sequence deadline before engine probes")
    actual = gate_weights.require_idle_engine(engine, require_push=False)
    if (actual["device"] != f"cuda:{runtime['device']['index']}" or actual["policy_version"] != 0
            or not actual["in_process"] or runtime["model_file_count"] != 13):
        raise ValueError("gate sequence: runtime and native initial engine differ")
    measured = {"G0": _record("G0", {"lock_match": True, "cuda": True, "bf16_matmul": True,
        "model_executor": True, "vllm_multiprocessing": False, "runtime": runtime,
        "engine_runtime": actual, "provenance": "successful runtime_check and live require_idle_engine"},
        record, should_stop)}
    requests = [p.request for p in probes]
    measured["G2"] = _record("G2", gate_images.measure_image_parity(engine, requests), record, should_stop)
    measured["G7_INITIAL"] = _record("G7_INITIAL", gate_logprobs.measure_logprobs(bundle, engine,
        _logprob_requests(probes, 0, seed), protocol, stage="INITIAL", record=record), record, should_stop)
    measured["G8"] = _record("G8", gate_tokens.measure_tokens(engine, requests,
        feedback="Continue this unscored token-boundary probe; no correctness feedback is provided.",
        retry_max_new_tokens=32, record=record), record, should_stop)
    measured["G6_BEFORE"] = _record("G6_BEFORE", gate_cache.measure_current(engine, probes,
        stage="BEFORE", record=record), record, should_stop)
    return measured


def after_update(bundle, engine, preparation, probes, model_path, protocol, *, seed, record, should_stop):
    if should_stop():
        raise TimeoutError("gate sequence deadline after shared update")
    if engine.policy_version != 1 or preparation.policy_version != 1:
        raise ValueError("gate sequence: post measurements must observe shared step1, never live step2")
    measured = {}
    measured["G5"] = _record("G5", gate_weights.measure_weight_parity(bundle, engine), record, should_stop)
    measured["G6_AFTER"] = _record("G6_AFTER", gate_cache.measure_current(engine, probes,
        stage="AFTER", record=record), record, should_stop)
    measured["G7_POST_SYNC"] = _record("G7_POST_SYNC", gate_logprobs.measure_logprobs(bundle, engine,
        _logprob_requests(probes, 1, seed), protocol, stage="POST_SYNC", record=record), record, should_stop)
    measured["G3_POST_SYNC"] = _record("G3_POST_SYNC", _reference(bundle, preparation, probes,
        model_path, protocol, stage="POST_SYNC", engine=engine), record, should_stop)
    return measured


def combine(measured, shared, cache, replay, protocol, scope):
    """Validate all observed components, without manufacturing approval receipts."""
    from .approval import _gpu_gates

    initial, post = measured["G3_INITIAL"], measured["G3_POST_SYNC"]
    if (initial["stage"] != "INITIAL" or post["stage"] != "POST_SYNC"
            or initial["base_sha256"] != post["base_sha256"]
            or not all(v["native_execution"] is True for v in (initial, post, cache, replay))):
        raise ValueError("gate sequence: bound initial/post native observations required")
    g3 = {"initial_max_abs": initial["initial_max_abs"], "post_sync_max_abs": post["post_sync_max_abs"],
        "frozen_sha_unchanged": initial["frozen_sha_unchanged"] and post["frozen_sha_unchanged"],
        "requires_grad_restored": initial["requires_grad_restored"] and post["requires_grad_restored"],
        "stages": {"INITIAL": initial, "POST_SYNC": post}}
    first, second = measured["G7_INITIAL"], measured["G7_POST_SYNC"]
    if (first["stage"] != "INITIAL" or second["stage"] != "POST_SYNC"
            or first["policy_version"] != 0 or second["policy_version"] != 1
            or not all(v["native_execution"] is True and v["trajectories"] >= 32
                       and v["delta_seq"] == protocol["loss"]["mismatch"]["delta_seq"] for v in (first, second))):
        raise ValueError("gate sequence: two independently measured logprob stages required")
    g7 = {"trajectories": first["trajectories"]+second["trajectories"],
        "step0_mean_abs": first["step0_mean_abs"], "post_sync_mean_abs": second["post_sync_mean_abs"],
        "delta_seq": first["delta_seq"], "synthetic_0_1_masked": first["synthetic_0_1_masked"] and second["synthetic_0_1_masked"],
        "stages": {"INITIAL": first, "POST_SYNC": second}}
    values = {k: measured[k] for k in ("G0", "G1", "G2", "G5", "G8", "G9", "G10")}
    errors = replay["logit_errors"]
    expected = {f"{i}.{kind}.{field}" for i in (0, 1) for kind in ("policy", "reference")
                for field in ("logits", "logprobs")}
    if (replay.get("passed") is not True or replay.get("fresh_process") is not True
            or set(errors) != expected or replay.get("different_fields") != []):
        raise ValueError("gate sequence: complete fresh-process checkpoint comparison required")
    g12 = replay | {"fresh_process_reload": replay["fresh_process"],
        **{kind+"_logits_equal": all(errors[f"{i}.{kind}.{field}"]["bit_equal"] is True
            for i in (0, 1) for field in ("logits", "logprobs")) for kind in ("policy", "reference")}}
    values.update(G3=g3, G4=shared["G11"]["G4"], G6=cache, G7=g7,
                  G11=shared["G11"], G12=g12, G13=shared["G13"])
    _gpu_gates({k: {"measurements": v} for k, v in values.items()}, scope, protocol)
    return values
