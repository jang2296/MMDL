"""Bounded shared-policy HF smoke for 24GB GPUs, never general training approval.

One live BF16 policy alternates generation and RL; no vLLM copy, fixed 10GiB KV
reservation or merged-weight push. Native-vLLM launch/gates remain separate.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import approval, gate_session, launch, measure
from .dataset import SPLITS, _bytes, _write, load_images
from .schema import ProblemTree, from_json, load_protocol, to_json
from .vault import _safe_root

FORMAT = "notebook-rl-shared-hf-smoke-job-v1"
PURPOSE = "BOUNDED_SHARED_HF_RL_SMOKE_ONLY"
PROFILE = "hf_shared_policy_24gb_v1"
KEYS = gate_session.KEYS | {"problem_ids", "algorithm"}


def memory_floor(protocol: dict, weight_bytes: int) -> int:
    """Conservative persistent allowance, NOT an activation/peak-fit certificate."""
    if type(weight_bytes) is not int or weight_bytes <= 0:
        raise ValueError("shared smoke: positive measured weight size required")
    counts = protocol["expected_trainable"]
    if set(counts) != {"L", "M", "V"} or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("shared smoke: actual pinned trainable counts required")
    # Four FP32 copies allow for masters, gradients and both Adam moments.
    return weight_bytes + 16*sum(counts.values())


def _shape(job: dict) -> None:
    if (type(job) is not dict or set(job) != KEYS or job["format"] != FORMAT
            or type(job["runtime_scope"]) is not dict
            or job["runtime_scope"].get("purpose") != PURPOSE
            or job["arm"] != "MOD-RV" or job["algorithm"] not in ("ALG-NB3", "ALG-NB+D")
            or type(job["problem_ids"]) is not list or len(job["problem_ids"]) != 10
            or any(type(p) is not str or not p for p in job["problem_ids"])
            or len(set(job["problem_ids"])) != 10):
        raise ValueError("shared smoke: explicit ten-problem routed RL smoke required")


def validate_job(job: dict, protocol: dict) -> None:
    _shape(job)
    # Reuse original path/identity/probe syntax without granting its execution mode.
    syntax = {k: job[k] for k in gate_session.KEYS}
    syntax.update(format=gate_session.FORMAT,
                  runtime_scope=job["runtime_scope"] | {"purpose": gate_session.PURPOSE})
    gate_session.validate_job(syntax, protocol)


def preflight(job: dict, protocol: dict) -> dict:
    validate_job(job, protocol)
    checked = approval.validate_for_gate_data(Path(job["build_root"]),
        **{k: job[k] for k in (*measure.DESCRIPTORS, "expected_measurement_scope")})
    if not gate_session._eligible(checked):
        return checked
    probes = gate_session._probe_actors(job, checked)
    actors = {a.problem_id: a for a in gate_session._actors(checked)}
    chosen = job["problem_ids"]
    if (not set(chosen) <= actors.keys() or not {a.problem_id for a in probes.values()} <= set(chosen)
            or len({checked["structures"][p] for p in chosen}) != 10):
        raise ValueError("shared smoke: ten eligible distinct structures and bound probes required")
    return checked


def _binding(job: dict, protocol: dict, checked: dict) -> dict:
    from .checkpoint import digest
    result = launch.checkpoint_binding(checked, protocol, job["commit"])
    # Execution identity is distinct; old vLLM or pre-note checkpoints cannot resume here.
    result["code_sha256"] = digest({"base": result["code_sha256"], "profile": PROFILE,
        "problem_ids": job["problem_ids"], "algorithm": job["algorithm"], "final_audit_note": True})
    return result


def _final_notes(batch, actors, image_root, engine, prompts, cfg, should_stop):
    from .rollout import append_final_notebook
    trees = list(batch.trees)
    for row in batch.discarded:
        if "tree" in row:
            trees.append(from_json(ProblemTree, row["tree"]))
    for tree in trees:
        if "final_notebook" in tree.notebook:
            continue
        owned = tree.problem_id not in batch.images
        images = load_images(actors[tree.problem_id], image_root) if owned else batch.images[tree.problem_id]
        try:
            note = append_final_notebook(tree, images, engine, prompts, cfg, should_stop=should_stop)
            if note["status"] == "BUDGET_STOP":
                batch.status = "BUDGET_STOP"
        finally:
            if owned:
                for image in images:
                    image.close()
    saved = {t.problem_id: t for t in trees}
    for row in batch.discarded:
        if "tree" in row:
            row["tree"] = to_json(saved[row["problem_id"]])
    return {t.problem_id: t.notebook["final_notebook"]["status"] for t in trees}


def _prepare_hf_dependencies() -> dict:
    # HF probes optional SDK availability on import; actual SDK imports stay blocked.
    from peft import LoraConfig, get_peft_model
    from peft.import_utils import is_bnb_available
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # PEFT lazily imports optional bnb compiler dependencies during adapter construction.
    if is_bnb_available():
        from importlib import import_module
        import_module("bitsandbytes")

    return {value.__name__: value.__module__ for value in (
        LoraConfig, get_peft_model, AutoProcessor, Qwen3VLForConditionalGeneration)}


def execute(job: dict, protocol: dict, checked: dict) -> dict:
    if not gate_session._eligible(checked) or preflight(job, protocol) != checked:
        raise ValueError("shared smoke: fresh complete data eligibility required before model allocation")
    runtime = launch.runtime_check(job | {"expected_gpu_scope": job["runtime_scope"]}, protocol)
    import numpy as np
    import torch
    from fine_tuning.modules import tensor_sha256
    from . import batches, checkpoint
    from .guards import install_guards
    from .policy import assert_base_unchanged, build_policy
    from .prompts import load_prompts
    from .rollout import RolloutConfig
    from .shared_policy_engine import SharedPolicyRolloutEngine
    from .trainer import synchronize
    from .updates import optimizer_step
    from .verifier_service import VerifierService

    runtime["trusted_hf_imports"] = _prepare_hf_dependencies()
    root, output = Path(job["build_root"]), Path(job["output"])
    binding = _binding(job, protocol, checked)
    _safe_root(output.parent, create=True)
    output.mkdir(mode=0o700, exist_ok=False)
    _write(output/"launch.json", _bytes({"job": job, "binding": binding, "runtime": runtime,
        "capture_pid": os.getpid(),
        "input_binding": checked["input_binding"], "profile": PROFILE, "general_training_allowed": False})+b"\n")
    result: dict[str, Any] = {"status": "INCOMPLETE", "profile": PROFILE, "binding": binding,
        "optimizer_updates": 0, "attempted_ids": [], "fresh_reload_verified": False,
        "final_note_statuses": {}, "discarded_problem_ids": [],
        "native_vllm_certified": False, "model_improvement_verified": False,
        "pod_billing_stopped": False, "general_training_allowed": False}
    forbidden = [Path(p) for p in job["forbidden_roots"]]
    install_guards(forbidden_roots=forbidden,
        vault_files=[root/s/name for s in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
        +[launch.descriptor(job[k]) for k in measure.DESCRIPTORS], allow_vault=False, allow_benchmark=False)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.cuda.reset_peak_memory_stats(runtime["device"]["index"])
    bundle = optimizer = engine = None
    try:
        bundle = build_policy(Path(job["model_path"]), job["arm"], protocol)
        bundle.model.to(f"cuda:{runtime['device']['index']}")
        from .gpu_gates import measure_inventory
        _write(output/"inventory.json", _bytes(measure_inventory(bundle, protocol))+b"\n")
        optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
        engine = SharedPolicyRolloutEngine(bundle, protocol["image"], protocol["context_limit"],
                                          logprob_chunk=protocol["model"]["logprob_chunk"])
        prompts = load_prompts(launch.HERE/"prompts_v1.yaml")
        if prompts["sha256"] != protocol["prompts_sha256"]:
            raise ValueError("shared smoke: prompt bytes changed")
        by_id = {a.problem_id: a for a in gate_session._actors(checked) if a.problem_id in job["problem_ids"]}
        structures = {pid: checked["structures"][pid] for pid in by_id}
        order = batches.family_order(list(by_id.values()), job["seed"])
        cursor, step = 0, 0
        def should_stop():
            return time.time() >= runtime["deadline_timestamp"]-120
        with ExitStack() as stack:
            probes = launch.load_probes(job["sync_probes"], job["probe_scope"], root/"images", prompts, stack)
            verifier = stack.enter_context(VerifierService(root/"train", forbidden_roots=forbidden))
            if verifier.metadata["vault_manifest_sha256"] != launch.read_json(root/"manifest.yaml")["split_manifests"]["train"]:
                raise ValueError("shared smoke: verifier split identity changed")
            _write(output/"initial_sync.json", _bytes({"probes": synchronize(
                bundle, engine, engine, probes, protocol, 0, push=False)})+b"\n")
            journal_index = 0
            def record(kind, value):
                nonlocal journal_index
                _write(output/f"raw-{journal_index:05d}.json", _bytes({"kind": kind, "raw": to_json(value)})+b"\n")
                journal_index += 1
            while cursor < len(order) and step < 2 and not should_stop():
                batch = batches.collect_batch(by_id, structures, order, cursor,
                    lambda actor: load_images(actor, root/"images"), engine, verifier, prompts,
                    protocol, job["algorithm"], should_stop=should_stop, seed=job["seed"],
                    on_tree=lambda t: record("TREE", t), on_group=lambda g: record("GROUP", g))
                try:
                    cfg = replace(RolloutConfig.from_protocol(protocol, job["algorithm"], master_seed=job["seed"]),
                                  final_notebook=True)
                    # Extra audit calls are explicitly bound by this profile, not hidden in the old protocol.
                    result["final_note_statuses"].update(_final_notes(
                        batch, by_id, root/"images", engine, prompts, cfg, should_stop))
                    _write(output/f"batch-{cursor:03d}.json", _bytes({"attempted": batch.attempted,
                        "discarded": batch.discarded, "diagnosis": batch.diagnosis, "status": batch.status,
                        "trees": [json.loads(to_json(t)) for t in batch.trees]})+b"\n")
                    result["attempted_ids"].extend(batch.attempted)
                    result["discarded_problem_ids"].extend(r["problem_id"] for r in batch.discarded)
                    cursor = batch.cursor
                    if batch.status == "BUDGET_STOP" or should_stop():
                        result["status"] = "BUDGET_STOP"
                        break
                    if not batch.trees:
                        continue
                    update = optimizer_step(bundle, optimizer, scheduler, batch.trees, batch.images, engine,
                        protocol, job["algorithm"], step=step, seed=job["seed"])
                    _write(output/f"update-{step:03d}.json", _bytes(update)+b"\n")
                    if update["status"] != "UPDATED_UNSYNCHRONIZED":
                        result["status"] = update["status"]
                        break
                    step = update["step_after"]
                    result["optimizer_updates"] = step
                    _write(output/f"sync-{step:03d}.json", _bytes({"probes": synchronize(
                        bundle, engine, engine, probes, protocol, step, push=True)})+b"\n")
                finally:
                    for values in batch.images.values():
                        for image in values:
                            image.close()
            assert_base_unchanged(bundle)
            result["cohort_complete"] = cursor == len(order)
            if result["status"] == "INCOMPLETE":
                result["status"] = ("CAPTURED_PENDING_FRESH_RELOAD" if step and cursor == len(order)
                                    else "NO_LEARNING_SIGNAL" if cursor == len(order) else "BUDGET_OR_STEP_STOP")
            if step:
                destination = output/"checkpoint"
                checkpoint.save(bundle, optimizer, scheduler, step,
                    {"epoch": 0, "position": cursor, "problem_ids": order,
                     "discarded_problem_ids": result["discarded_problem_ids"],
                     "sync_pending": not engine._synchronized},
                    checkpoint.capture_rng(), destination, protocol=protocol, binding=binding)
                _write(output/"reload_expected.json", _bytes({"binding": binding, "step": step,
                    "parameter_sha256": {n: tensor_sha256(p) for n, p in bundle.model.named_parameters()}})+b"\n")
                result["checkpoint"] = str(destination)
    except BaseException as exc:
        result.update(status="FAILED", error_type=type(exc).__name__)
        raise
    finally:
        index = runtime["device"]["index"]
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(index),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(index))
        _write(output/"result.json", _bytes(result)+b"\n")
        del engine, optimizer, bundle
        gc.collect()
        torch.cuda.empty_cache()
    return result


def reload_check(job: dict, protocol: dict) -> dict:
    """Run as a separate CLI process AFTER capture released its only model copy."""
    from . import checkpoint
    from fine_tuning.modules import tensor_sha256
    _shape(job)
    output = _safe_root(Path(job["output"]))
    closed = launch.read_json(output/"launch.json")
    result = launch.read_json(output/"result.json")
    expected = launch.read_json(output/"reload_expected.json")
    import psutil
    capture_pid = closed.get("capture_pid")
    if type(capture_pid) is not int or capture_pid <= 0 or capture_pid == os.getpid() or psutil.pid_exists(capture_pid):
        raise ValueError("shared smoke: capture process must exit before fresh-process reload")
    fresh_binding = _binding(job, protocol, {"code_sha256": approval.code_hashes(),
                                             "input_binding": closed["input_binding"]})
    if (closed.get("job") != job or closed.get("binding") != fresh_binding
            or result.get("binding") != fresh_binding or result.get("profile") != PROFILE
            or result.get("attempted_ids") is None
            or len(result["attempted_ids"]) != 10 or set(result["attempted_ids"]) != set(job["problem_ids"])
            or result.get("status") != "CAPTURED_PENDING_FRESH_RELOAD"
            or result.get("cohort_complete") is not True or expected.get("binding") != closed.get("binding")
            or type(expected.get("step")) is not int or not 1 <= expected["step"] <= 2
            or expected.get("step") != result.get("optimizer_updates") or result.get("checkpoint") != str(output/"checkpoint")):
        raise ValueError("shared smoke: complete closed capture required; no failed-sync resume")
    launch.runtime_check(job | {"expected_gpu_scope": job["runtime_scope"]}, protocol)
    bundle, _, _, step, cursor = checkpoint.load_resume(Path(job["model_path"]), output/"checkpoint",
        protocol, binding=expected["binding"], device=f"cuda:{job['runtime_scope']['device']['index']}")
    actual = {n: tensor_sha256(p) for n, p in bundle.model.named_parameters()}
    if (step != expected["step"] or actual != expected["parameter_sha256"]
            or cursor.get("sync_pending") is not False or cursor.get("epoch") != 0
            or cursor.get("position") != 10 or len(cursor.get("problem_ids", [])) != 10
            or set(cursor["problem_ids"]) != set(job["problem_ids"])
            or cursor.get("discarded_problem_ids") != result.get("discarded_problem_ids")):
        raise ValueError("shared smoke: fresh-process weights or optimizer step differ")
    verified = {"status": "SHARED_HF_SMOKE_RELOAD_VERIFIED", "profile": PROFILE,
        "optimizer_updates": step, "cohort_complete": True, "fresh_reload_verified": True,
        "native_vllm_certified": False, "model_improvement_verified": False,
        "general_training_allowed": False, "pod_billing_stopped": False}
    _write(output/"reload_verified.json", _bytes(verified)+b"\n")
    return verified


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--reload-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        job = launch.read_json(args.job)
        protocol = load_protocol(launch.HERE/"protocol_v1.yaml")
        if args.reload_check:
            print(json.dumps(reload_check(job, protocol), allow_nan=False))
            return 0
        checked = preflight(job, protocol)
        if args.check or not gate_session._eligible(checked):
            print(json.dumps({"status": checked["status"], "blockers": checked["blockers"],
                "profile": PROFILE, "gpu_fit_certified": False, "training_executed": False}))
            return 0 if gate_session._eligible(checked) else 2
        result = execute(job, protocol, checked)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["status"] == "CAPTURED_PENDING_FRESH_RELOAD" else 2
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
            "profile": PROFILE, "training_complete": False, "pod_billing_stopped": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
