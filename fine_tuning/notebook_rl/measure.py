"""Run fixed-F0 candidate measurements; never approve data or start training.

Requires eligible data, a clean exact-SHA MMDL_test clone and the live financial
watchdog. This is the candidate/retry component, not complete Phase A or G0–G13.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

from . import approval, launch
from .dataset import SPLITS, _bytes, _sha, _write
from .schema import load_protocol
from .vault import _read, _safe_root

FORMAT = "notebook-rl-candidate-measurement-job-v1"
DESCRIPTORS = ("index", "source_registry", "evidence", "data_report")
KEYS = {"format", "build_root", *DESCRIPTORS, "expected_measurement_scope",
        "runtime_scope", "model_path", "seed", "output", "deadline_utc", "commit", "forbidden_roots"}


def validate_job(job: dict, protocol: dict) -> None:
    from fine_tuning.budget import utc

    if (type(job) is not dict or set(job) != KEYS or job["format"] != FORMAT
            or type(job["seed"]) is not int or job["seed"] not in protocol["seeds"]
            or not isinstance(job["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", job["commit"])):
        raise ValueError("measure: fixed candidate job required; no subset, cap or training override")
    utc(job["deadline_utc"])
    for key in DESCRIPTORS:
        launch.descriptor(job[key])
    for key in ("build_root", "model_path", "output"):
        if not isinstance(job[key], str) or not Path(job[key]).is_absolute():
            raise ValueError("measure: explicit absolute paths required")
    if (type(job["forbidden_roots"]) is not list or not job["forbidden_roots"]
            or any(not isinstance(p, str) or not Path(p).is_absolute() for p in job["forbidden_roots"])):
        raise ValueError("measure: explicit benchmark roots required even if unmounted")
    output = Path(job["output"])
    protected = [launch.ROOT, Path(job["build_root"]), Path(job["model_path"]),
                 *(Path(p) for p in job["forbidden_roots"]), *(launch.descriptor(job[k]) for k in DESCRIPTORS)]
    if (output.exists() or any(p.is_symlink() for p in (output, *output.parents))
            or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
                   for p in protected)):
        raise ValueError("measure: new private output must not overlap protected inputs/checkout")
    scope = job["runtime_scope"]
    if (type(scope) is not dict or set(scope) != {"purpose", "runtime_id", "device", "environment_lock_sha256"}
            or scope["purpose"] != "FROZEN_WEIGHT_MEASUREMENT_ONLY"
            or not isinstance(scope["runtime_id"], str) or not scope["runtime_id"]
            or not isinstance(scope["environment_lock_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", scope["environment_lock_sha256"]) is None):
        raise ValueError("measure: explicit measurement runtime scope required; not GPU gate approval")
    device = scope["device"]
    if (type(device) is not dict or set(device) != {"type", "uuid", "name", "index", "total_memory_bytes", "driver_version"}
            or device["type"] != "cuda" or type(device["index"]) is not int or device["index"] < 0
            or type(device["total_memory_bytes"]) is not int or device["total_memory_bytes"] <= 0
            or any(not isinstance(device[k], str) or not device[k] for k in ("uuid", "name", "driver_version"))):
        raise ValueError("measure: expected physical GPU identity required")


def preflight(job: dict, protocol: dict) -> dict:
    validate_job(job, protocol)
    return approval.validate_for_measurement(Path(job["build_root"]),
        **{k: job[k] for k in (*DESCRIPTORS, "expected_measurement_scope")})


def execute(job: dict, protocol: dict, validated: dict) -> dict:
    if (validated.get("status") != "VALIDATED_FOR_MEASUREMENT"
            or validated.get("measurement_allowed") is not True or validated.get("training_allowed") is not False):
        raise ValueError("measure: data eligibility required before model allocation")
    if preflight(job, protocol) != validated:
        raise ValueError("measure: inputs changed before execution")
    binding = launch.checkpoint_binding(validated, protocol, job["commit"])
    runtime = launch.runtime_check(job | {"expected_gpu_scope": job["runtime_scope"]}, protocol)
    # Imports here keep --help/--check CPU-only and avoid constructing a trainer.
    from .engines import VLLMRolloutEngine
    from .guards import install_guards, prepare_vllm_schema_dependency
    from .phase_a import measure_candidates

    root, output = Path(job["build_root"]), Path(job["output"])
    _safe_root(output.parent, create=True)
    output.mkdir(mode=0o700, exist_ok=False)
    _write(output/"launch.json", _bytes({"job": job, "binding": binding, "runtime": runtime,
        "approval_binding": validated["input_binding"], "training_allowed": False,
        "status": "VALIDATED_NOT_YET_MEASURED"}) + b"\n")
    result = {"status": "INCOMPLETE", "training_executed": False, "phase_a_completed": False,
        "gpu_gates_certified": False, "pod_billing_stopped": False, "engine_shutdown": "NOT_STARTED",
        "launch_sha256": _sha(_read(output/"launch.json")), "binding": binding}
    engine = None
    try:
        result["schema_dependency_guard"] = prepare_vllm_schema_dependency()
        forbidden = [Path(p) for p in job["forbidden_roots"]]
        install_guards(forbidden_roots=forbidden,
            vault_files=[root/s/name for s in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
            + [launch.descriptor(job[k]) for k in DESCRIPTORS], allow_vault=False, allow_benchmark=False)
        cfg = protocol["engine"]["vllm"]
        engine = VLLMRolloutEngine(Path(job["model_path"]), protocol["image"], cfg["kv_cache_memory_bytes"],
            cfg["max_num_seqs"], protocol["context_limit"])
        if type(engine.policy_version) is not int or engine.policy_version != 0:
            raise ValueError("measure: fresh original F0 required")
        summary = measure_candidates(root, output/"candidates", engine=engine, protocol=protocol,
            input_binding=binding, deadline_timestamp=runtime["deadline_timestamp"], seed=job["seed"],
            forbidden_roots=forbidden)
        summary_path = output/"candidates"/"summary.json"
        if (type(engine.policy_version) is not int or engine.policy_version != 0
                or summary != launch.read_json(summary_path)
                or summary.get("status") not in ("MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL", "INCOMPLETE")):
            raise ValueError("measure: fixed policy and actual closed candidate ledger required")
        result.update(status=summary["status"], candidate_summary_sha256=_sha(_read(summary_path)))
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        raise
    finally:
        if engine is not None:
            try:
                # vLLM v0.11.0 InprocClient.shutdown owns its EngineCore only.
                engine.llm.llm_engine.engine_core.shutdown()
                result["engine_shutdown"] = "COMPLETED"
            except BaseException as exc:
                result.update(status="INCOMPLETE", engine_shutdown="FAILED", shutdown_error_type=type(exc).__name__)
                _write(output/"result.json", _bytes(result)+b"\n")
                raise
        _write(output/"result.json", _bytes(result)+b"\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        job = launch.read_json(args.job)
        protocol = load_protocol(launch.HERE/"protocol_v1.yaml")
        checked = preflight(job, protocol)
        if args.check or checked["status"] != "VALIDATED_FOR_MEASUREMENT":
            print(json.dumps({"status": checked["status"], "blockers": checked["blockers"],
                "training_executed": False, "gpu_runtime_checked": False, "phase_a_completed": False}))
            return 0 if checked["status"] == "VALIDATED_FOR_MEASUREMENT" else 2
        result = execute(job, protocol, checked)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["status"] == "MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL" else 2
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
            "training_executed": False, "phase_a_completed": False, "pod_billing_stopped": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
