"""Run an approved notebook-RL job in a clean RunPod checkout under its watchdog.

This command does not rent a Pod, manufacture approvals, download data, or change
the protocol. --check validates prerequisites only. --execute requires actual
data/GPU evidence and the owned financial watchdog before loading model weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from . import approval
from .dataset import SPLITS, _bytes, _sha, _write, load_images
from .schema import ActorItem, from_json, load_protocol
from .resources import resource_path
from .vault import _read, _safe_root, _unique

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESCRIPTORS = ("index", "source_registry", "evidence", "data_report", "gpu_report", "f0_report", "sync_probes")
KEYS = {"format", "build_root", "selection_path", "phase_a_report", *DESCRIPTORS,
        "expected_gpu_scope", "model_path", "arm", "algorithm", "seed", "output",
        "deadline_utc", "max_steps", "resume", "commit", "forbidden_roots"}


def read_json(path: Path) -> dict:
    def nonfinite(_):
        raise ValueError("launch: nonfinite JSON")
    value = json.loads(_read(Path(path)), object_pairs_hook=_unique, parse_constant=nonfinite)
    if type(value) is not dict:
        raise ValueError("launch: JSON object required")
    return value


def descriptor(value: dict) -> Path:
    if (type(value) is not dict or set(value) != {"path", "file_sha256"}
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or not isinstance(value["file_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["file_sha256"])):
        raise ValueError("launch: use file descriptors, never inline private index/evidence")
    return Path(value["path"])


def read_descriptor(value: dict) -> dict:
    path = descriptor(value)
    raw = _read(path)
    if _sha(raw) != value["file_sha256"]:
        raise ValueError("launch: artifact bytes differ from descriptor")
    result = read_json(path)
    if _sha(_read(path)) != value["file_sha256"]:
        raise ValueError("launch: artifact changed while reading")
    return result


def validate_job(job: dict, protocol: dict) -> None:
    from fine_tuning.budget import utc

    if (set(job) != KEYS or job["format"] != "notebook-rl-training-job-v1"
            or job["arm"] not in protocol["arms"] or job["arm"] == "MOD-RS"
            or job["algorithm"] not in protocol["algorithms"]
            or type(job["seed"]) is not int or job["seed"] not in protocol["seeds"]
            or type(job["max_steps"]) is not int or job["max_steps"] < 1
            or not isinstance(job["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", job["commit"])):
        raise ValueError("launch: invalid frozen job identity/limits; MOD-RS needs its own reviewed policy")
    utc(job["deadline_utc"])
    for key in DESCRIPTORS:
        descriptor(job[key])
    for key in ("build_root", "model_path", "selection_path", "phase_a_report", "output"):
        if not isinstance(job[key], str) or not Path(job[key]).is_absolute():
            raise ValueError("launch: explicit absolute private/model paths required")
    if job["resume"] is not None and (not isinstance(job["resume"], str) or not Path(job["resume"]).is_absolute()):
        raise ValueError("launch: explicit checkpoint path required")
    if (not isinstance(job["forbidden_roots"], list) or not job["forbidden_roots"]
            or any(not isinstance(p, str) or not Path(p).is_absolute() for p in job["forbidden_roots"])):
        raise ValueError("launch: explicit benchmark roots required even if unmounted")
    output = Path(job["output"])
    if (any(p.is_symlink() for p in (output, *output.parents))
            or any(output.resolve().is_relative_to(p.resolve()) for p in
                   (ROOT, Path(job["model_path"]), Path(job["build_root"])))):
        raise ValueError("launch: output must be outside checkout/model/data")
    if type(job["expected_gpu_scope"]) is not dict or job["expected_gpu_scope"].get("arm") != job["arm"]:
        raise ValueError("launch: actual GPU gate arm must match training arm")


def preflight(job: dict, protocol: dict) -> dict:
    validate_job(job, protocol)
    return approval.validate_for_training(Path(job["build_root"]), Path(job["selection_path"]),
        phase_a_report=Path(job["phase_a_report"]), index=job["index"], source_registry=job["source_registry"],
        evidence=job["evidence"], data_report=job["data_report"], gpu_report=job["gpu_report"],
        expected_gpu_scope=job["expected_gpu_scope"])


def checkpoint_binding(validated: dict, protocol: dict, commit: str) -> dict:
    """Shared with the pre-learning F0 monitor; runtime GPU evidence is not model identity."""
    from .checkpoint import digest
    return {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
        "code_sha256": digest({"commit": commit, "sources": validated["code_sha256"]}),
        "data_sha256": digest({k: v for k, v in validated["input_binding"].items()
            if k not in ("data_report_sha256", "gpu_report_sha256")})}


def runtime_check(job: dict, protocol: dict) -> dict:
    """Read actual checkout, ancestor watchdog, lock, pinned weights and GPU identity."""
    from fine_tuning.budget import effective_deadline, utc

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=30).strip()
    if (git("remote", "get-url", "origin") not in
            ("git@github.com:jang2296/MMDL_test.git", "https://github.com/jang2296/MMDL_test.git")
            or git("rev-parse", "HEAD") != job["commit"]
            or git("status", "--porcelain", "--untracked-files=all")):
        raise ValueError("launch: exact clean MMDL_test clone required; no source overlay")
    scope = job["expected_gpu_scope"]
    if not scope.get("runtime_id") or os.environ.get("RUNPOD_POD_ID") != scope["runtime_id"]:
        raise ValueError("launch: actual owned Pod identity differs from GPU gate")
    psutil = import_module("psutil")
    watcher_id = int(os.environ.get("MMDL_BUDGET_WATCHDOG_PID", "0"))
    parents = {p.pid: p for p in psutil.Process().parents()}
    if watcher_id not in parents:
        raise ValueError("launch: owned financial watchdog must be a live ancestor")
    watcher = parents[watcher_id]
    command = watcher.cmdline()
    if "fine_tuning.budget" not in command:
        raise ValueError("launch: unexpected watchdog command")
    def argument(name):
        key = "--" + name
        if command.count(key) != 1:
            raise ValueError("launch: ambiguous watchdog budget")
        return command[command.index(key)+1]
    if argument("owned-pod-id") != scope["runtime_id"]:
        raise ValueError("launch: watchdog Pod mismatch")
    financial = effective_deadline(utc(argument("gpu-cutoff-utc")), utc(argument("billing-start-utc")),
        float(argument("balance-usd")), float(argument("reserve-usd")), float(argument("hourly-usd")))
    maximum = float(argument("max-seconds")) if "--max-seconds" in command else 3600.
    deadline = utc(job["deadline_utc"]).timestamp()
    outer_end = min(financial.timestamp(), watcher.create_time()+maximum)-120
    if not math.isfinite(maximum) or maximum <= 0 or not time.time()+120 < deadline <= outer_end:
        raise ValueError("launch: job deadline must retain watchdog/financial recovery reserve")
    lock = resource_path(protocol["environment"]["lock"], project_root=ROOT)
    if _sha(_read(lock)) != scope["environment_lock_sha256"]:
        raise ValueError("launch: lock bytes changed")
    for line in lock.read_text().splitlines():
        if not line or line.startswith(("#", "--")):
            continue
        name, expected = line.split("==", 1)
        if version(name) != expected:
            raise ValueError(f"launch: exact installed lock mismatch: {name}")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, capture_output=True, timeout=60)
    from .checkpoint import file_hash
    manifest = read_json(resource_path(protocol["initial_checkpoint"]["manifest"], project_root=ROOT))
    model_root = Path(job["model_path"])
    if model_root.name != protocol["initial_checkpoint"]["revision"]:
        raise ValueError("launch: exact pinned HF snapshot path required")
    weight_bytes = 0
    for item in manifest["files"]:
        path = model_root / item["path"]
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("launch: pinned model/processor bytes changed")
        if path.suffix == ".safetensors":
            weight_bytes += item["bytes"]
    import torch
    from .policy import enable_deterministic_execution

    determinism = enable_deterministic_execution()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("launch: actual CUDA BF16 required; no precision/offload fallback")
    index = scope["device"]["index"]
    torch.cuda.set_device(index)
    properties = torch.cuda.get_device_properties(index)
    physical = subprocess.check_output(["nvidia-smi", f"--id={index}", "--query-gpu=uuid,name,driver_version",
                                       "--format=csv,noheader,nounits"], text=True, timeout=15).strip().split(", ")
    actual = dict(type="cuda", uuid=physical[0], name=physical[1], driver_version=physical[2],
                  index=index, total_memory_bytes=properties.total_memory)
    if actual != scope["device"]:
        raise ValueError("launch: live GPU does not match measured gate")
    free, _ = torch.cuda.mem_get_info(index)
    measuring = job.get("format") == "notebook-rl-candidate-measurement-job-v1"
    if measuring and scope.get("purpose") != "FROZEN_WEIGHT_MEASUREMENT_ONLY":
        raise ValueError("launch: measurement-only runtime scope required")
    # Frozen Phase A loads one vLLM copy; training still loads HF plus vLLM.
    shared_smoke = job.get("format") == "notebook-rl-shared-hf-smoke-job-v1"
    if shared_smoke:
        from .shared_smoke import PURPOSE, memory_floor
        if scope.get("purpose") != PURPOSE:
            raise ValueError("launch: explicit shared-policy bounded smoke authority required")
        weight_copies, minimum = 1, memory_floor(protocol, weight_bytes)
    else:
        weight_copies = 1 if measuring else 2
        minimum = weight_copies*weight_bytes+protocol["engine"]["vllm"]["kv_cache_memory_bytes"]
    if free <= minimum:
        raise ValueError("launch: persistent memory floor exceeds free memory; no automatic shrink")
    a = torch.ones((8, 8), device=f"cuda:{index}", dtype=torch.bfloat16)
    if not torch.equal(a@a, torch.full_like(a, 8)):
        raise ValueError("launch: actual BF16 matmul failed")
    return {"device": actual, "watchdog_pid": watcher_id, "deadline_timestamp": deadline,
            "free_memory_bytes": free, "minimum_weights_and_kv_bytes": minimum,
            "memory_fit_certified": False, "weight_copies": weight_copies,
            "model_file_count": len(manifest["files"]), "determinism": determinism,
            "execution_profile": "hf_shared_policy_24gb_v1" if shared_smoke else "vllm_original"}


def load_probes(value: dict, scope: dict, image_root: Path, prompts: dict, stack: ExitStack):
    from .engines import RolloutRequest
    from .prompts import turn1_text, user_message
    from .trainer import SyncProbe

    data = read_descriptor(value)
    if set(data) != {"probes"} or not isinstance(data["probes"], list) or len(data["probes"]) != 2:
        raise ValueError("launch: exactly two bound probe records required")
    declared = {p["id"]: p for p in scope["probes"]}
    result = []
    for row in data["probes"]:
        if set(row) != {"id", "actor", "completion_ids"}:
            raise ValueError("launch: unexpected probe payload")
        actor = from_json(ActorItem, _bytes(row["actor"]).decode())
        expected = {"id": row["id"], "image_count": len(actor.image_paths), "input_sha256": _sha(_bytes(row))}
        if declared.get(row["id"]) != expected or (not row["completion_ids"]
                or any(type(t) is not int or t < 0 for t in row["completion_ids"])):
            raise ValueError("launch: probe content differs from measured GPU gate")
        images = load_images(actor, image_root)
        for image in images:
            stack.callback(image.close)
        request = RolloutRequest(row["id"], None, [user_message(turn1_text(actor, prompts), len(images))],
                                 images, 1, len(row["completion_ids"])+1, [3407])
        result.append(SyncProbe(request, row["completion_ids"]))
    if len({p.request.request_id for p in result}) != 2 or sorted(len(p.request.images) for p in result) != [1, 2]:
        raise ValueError("launch: distinct single/multi-image probes required")
    return result


def execute(job: dict, protocol: dict, validated: dict) -> dict:
    """Production continuation; never pass synthetic approvals here in a real run."""
    if validated.get("status") != "VALIDATED_FOR_TRAINING" or validated.get("training_allowed") is not True:
        raise ValueError("launch: unapproved input cannot reach model allocation")
    # Revalidate instead of trusting an approval object from a prior process/checkout.
    current = preflight(job, protocol)
    if current != validated:
        raise ValueError("launch: approval changed before execution")
    from . import checkpoint, monitor
    root, model_path, output = Path(job["build_root"]), Path(job["model_path"]), Path(job["output"])
    binding = checkpoint_binding(validated, protocol, job["commit"])
    # A summary's numbers alone cannot authorize training, even for <12 steps.
    # Check the closed dev96 input/raw/score ledger before any GPU allocation.
    f0 = monitor.validate_f0_reference(descriptor(job["f0_report"]), root/"dev", root/"images",
        protocol=protocol, input_binding=binding, seed=job["seed"])
    phase = read_json(Path(job["phase_a_report"]))
    if (phase.get("status") != "PHASE_A_COMPLETED" or phase.get("policy_version") != 0
            or _sha(_read(Path(job["phase_a_report"]))) != validated["input_binding"]["phase_a_report_sha256"]
            or f0.get("status") != "VALIDATED_F0_REFERENCE"
            or f0.get("f0_report_sha256") != job["f0_report"]["file_sha256"]):
        raise ValueError("launch: matched actual F0 monitor and train Phase A required")
    # Dev length is separately measured; training health uses the Phase-A train baseline.
    reference = {"len_median": phase["health_reference"]["len_median"],
                 "dev_first_attempt_acc": f0["dev_first_attempt_acc"], "grounding_probe": f0["grounding_probe"]}
    for key, number in reference.items():
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError("launch: incomplete measured F0 health reference")
    if reference["len_median"] <= 0 or reference["dev_first_attempt_acc"] > 1 or reference["grounding_probe"] > 1:
        raise ValueError("launch: invalid F0 health range")
    runtime = runtime_check(job, protocol)
    import numpy as np
    import torch
    from .engines import InputPreparation, VLLMRolloutEngine
    from .gpu_gates import measure_inventory
    from .guards import install_guards, prepare_vllm_schema_dependency
    from .policy import build_policy, merged_named_tensors
    from .prompts import load_prompts
    from .trainer import run_loop
    from .verifier_service import VerifierService

    runtime["schema_dependency_guard"] = prepare_vllm_schema_dependency()
    prompts = load_prompts(HERE/"prompts_v1.yaml")
    if prompts["sha256"] != protocol["prompts_sha256"]:
        raise ValueError("launch: prompts changed")
    _safe_root(output.parent, create=True)
    output.mkdir(mode=0o700, exist_ok=False)
    _write(output/"launch.json", _bytes({"job": job, "binding": binding, "runtime": runtime,
        "approval_binding": validated["input_binding"], "status": "APPROVED_NOT_YET_TRAINED"})+b"\n")
    forbidden = [Path(p) for p in job["forbidden_roots"]]
    install_guards(forbidden_roots=forbidden,
        vault_files=[root/s/name for s in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
        +[descriptor(job[k]) for k in ("index", "evidence")], allow_vault=False, allow_benchmark=False)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    device = f"cuda:{runtime['device']['index']}"
    if job["resume"] is None:
        bundle = build_policy(model_path, job["arm"], protocol)
        bundle.model.to(device)
        optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
        step, cursor = 0, None
    else:
        bundle, optimizer, scheduler, step, cursor = checkpoint.load_resume(model_path, Path(job["resume"]),
            protocol, binding=binding, device=device)
        if bundle.inventory["arm"] != job["arm"]:
            raise ValueError("launch: resumed arm differs")
    # Re-read live tensors even when an earlier GPU report passed. A declaration
    # cannot certify the newly constructed/resumed policy's actual partitions.
    _write(output/"policy_inventory.json", _bytes({"binding": binding,
        "measurements": measure_inventory(bundle, protocol), "training_complete": False})+b"\n")
    saved_rng = checkpoint.capture_rng()
    engine_cfg = protocol["engine"]["vllm"]
    with ExitStack() as stack:
        engine = VLLMRolloutEngine(model_path, protocol["image"], engine_cfg["kv_cache_memory_bytes"],
                                  engine_cfg["max_num_seqs"], protocol["context_limit"])
        stack.callback(engine.llm.llm_engine.engine_core.shutdown)
        preparation = InputPreparation(bundle.processor, bundle.model.config, protocol["image"], protocol["context_limit"])
        if step:
            engine.initialize_weights(merged_named_tensors(bundle), step)
            preparation.policy_version = step
        checkpoint.restore_rng(saved_rng)
        probes = load_probes(job["sync_probes"], job["expected_gpu_scope"], root/"images", prompts, stack)
        verifier = stack.enter_context(VerifierService(root/"train", forbidden_roots=forbidden))
        if verifier.metadata["vault_manifest_sha256"] != read_json(root/"manifest.yaml")["split_manifests"]["train"]:
            raise ValueError("launch: actual verifier split differs from approved build")
        def dev(step):
            return monitor.measure_dev(root/"dev", root/"images", output/"dev"/f"step-{step:06d}",
                engine=engine, protocol=protocol, step=step, deadline_timestamp=runtime["deadline_timestamp"],
                input_binding=binding, f0_report=descriptor(job["f0_report"]), seed=job["seed"], forbidden_roots=forbidden)
        return run_loop(bundle=bundle, optimizer=optimizer, scheduler=scheduler, engine=engine,
            preparation=preparation, verifier=verifier, actors=validated["actors"], structures=validated["structures"],
            image_loader=lambda actor: load_images(actor, root/"images"), prompts=prompts, protocol=protocol,
            algorithm=job["algorithm"], seed=job["seed"], output=output/"train", binding=binding,
            f0_reference=reference, probes=probes, deadline_timestamp=runtime["deadline_timestamp"],
            max_steps=job["max_steps"], cursor=cursor, dev_evaluator=dev)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Private, hash-bound JSON training job")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        job = read_json(args.job)
        protocol = load_protocol(HERE/"protocol_v1.yaml")
        result = preflight(job, protocol)
        if result["status"] != "VALIDATED_FOR_TRAINING" or args.check:
            print(json.dumps({"status": result["status"], "blockers": result["blockers"],
                              "training_executed": False, "gpu_runtime_checked": False}))
            return 0 if result["status"] == "VALIDATED_FOR_TRAINING" else 2
        result = execute(job, protocol, result)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["status"] in ("STEP_LIMIT", "EPOCH_COMPLETED") else 2
    except Exception as exc:
        # Detailed errors stay in the private process log, never serialize private payloads.
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
                          "training_complete": False, "pod_billing_stopped": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
