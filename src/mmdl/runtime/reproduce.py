"""One clean-checkout RunPod job; never provisions or deletes cloud resources."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mmdl.runtime.artifacts import git_commit, read_json, sha256_file, write_json

TEAM_ORIGIN = re.compile(
    r"(?:https://github\.com/jang2296/MMDL|git@github\.com:jang2296/MMDL)(?:\.git)?"
)
HARDWARE_PROFILES = {
    "rtx4090_24gb.yaml": (r"NVIDIA(?: GeForce)? RTX 4090", 23 * 1024**3),
    "rtx3090_24gb.yaml": (r"NVIDIA(?: GeForce)? RTX 3090", 23 * 1024**3),
    "rtx5090_32gb.yaml": (r"NVIDIA(?: GeForce)? RTX 5090", 31 * 1024**3),
}


def arguments(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["evaluate"], default="evaluate")
    parser.add_argument("--target", choices=["runpod"], default="runpod")
    parser.add_argument("--suite", choices=["mmmu-val-two-runs"], default="mmmu-val-two-runs")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--protocol", type=Path, default=root / "configs/eval/mmmu_val_v1.yaml")
    parser.add_argument("--hardware", type=Path, default=root / "configs/hardware/rtx4090_24gb.yaml")
    parser.add_argument("--model-ref", type=Path, default=root / "manifests/models/baseline.json")
    parser.add_argument("--model-path", default=os.environ.get("MMDL_MODEL_PATH", "Qwen/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", args.job_id):
        parser.error("Invalid job ID")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be a full lowercase Git SHA")
    return args


def require_runtime(root):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The locked environment requires Python 3.12")
    names = ("MMDL_VOLUME_ROOT", "HF_HOME", "MMDL_DATA_ROOT", "MMDL_ARTIFACT_ROOT", "MMDL_VENV_ROOT")
    paths = {name: Path(os.environ[name]).expanduser().resolve() for name in names}
    volume = paths["MMDL_VOLUME_ROOT"]
    if volume == Path("/") or not volume.is_mount() or not root.is_relative_to(volume):
        raise ValueError("Checkout and task paths must use a verified persistent volume mount")
    for name, path in paths.items():
        if name != "MMDL_VOLUME_ROOT" and (path == volume or not path.is_relative_to(volume) or path.is_relative_to(root)):
            raise ValueError(f"{name} must be outside the checkout and inside the persistent volume")
    task_paths = [paths[name] for name in names if name != "MMDL_VOLUME_ROOT"]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(task_paths) for b in task_paths[i + 1:]):
        raise ValueError("Cache, data, artifacts and venv roots must be separate")
    reserve = float(os.environ["MMDL_STORAGE_RESERVE_GIB"])
    budget = float(os.environ["MMDL_BUDGET_USD"])
    if not math.isfinite(reserve) or reserve < 1 or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Explicit positive budget and storage reserve are required")
    deadline = datetime.fromisoformat(os.environ["MMDL_STOP_DEADLINE_UTC"].replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        raise ValueError("External watchdog deadline is absent or expired")
    if not os.environ.get("MMDL_WATCHDOG_ID") or not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("An armed external watchdog and actual Pod ID must be recorded")
    if shutil.disk_usage(volume).free < (35 + reserve) * 1024**3:
        raise RuntimeError("Insufficient volume headroom for installation/download/result peak estimate")
    return paths, deadline, budget


def _validate_origin(origin):
    if not TEAM_ORIGIN.fullmatch(origin):
        raise ValueError("RunPod checkout must originate from github.com/jang2296/MMDL")


def _validate_hardware_profile(hardware):
    profile = Path(hardware).name
    if profile not in HARDWARE_PROFILES:
        raise ValueError("Only the approved RTX 3090, RTX 4090, or RTX 5090 hardware profile is allowed")
    return profile


def _validate_doctor_gpu(report_path, hardware="rtx4090_24gb.yaml"):
    profile = _validate_hardware_profile(hardware)
    name_pattern, min_vram = HARDWARE_PROFILES[profile]
    report = read_json(report_path)
    gpus = report.get("environment", {}).get("torch", {}).get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise RuntimeError("RunPod suite requires exactly one visible GPU")
    gpu = gpus[0]
    name = gpu.get("name", "") if isinstance(gpu, dict) else ""
    if not re.fullmatch(name_pattern, str(name).strip(), re.IGNORECASE):
        raise RuntimeError(f"RunPod suite requires the GPU for {profile}")
    total = gpu.get("memory_total_bytes") if isinstance(gpu, dict) else None
    if (not isinstance(total, (int, float)) or isinstance(total, bool)
            or not math.isfinite(total) or total < min_vram):
        raise RuntimeError(f"RunPod {profile} reports insufficient VRAM")


def _open_stage_log(directory, job_id, completed_stages, label):
    index = completed_stages
    while True:
        log = directory / f"{job_id}.{index:03d}-{label}.log"
        try:
            return log, log.open("x", encoding="utf-8")
        except FileExistsError:
            index += 1


def execute(args, root):
    _validate_hardware_profile(args.hardware)
    git_commit(root, args.commit)
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
    _validate_origin(origin)
    paths, deadline, budget = require_runtime(root)
    artifacts, venv = paths["MMDL_ARTIFACT_ROOT"], paths["MMDL_VENV_ROOT"]
    data = args.data_root.resolve() if args.data_root else paths["MMDL_DATA_ROOT"] / "evaluation/mmmu"
    if not data.is_relative_to(paths["MMDL_DATA_ROOT"]):
        raise ValueError("Selected evaluation data must be under MMDL_DATA_ROOT")
    public = artifacts / "public"
    job_file = artifacts / "jobs" / f"{args.job_id}.json"
    binding = {"job_id": args.job_id, "pod_id": os.environ["RUNPOD_POD_ID"], "git_commit": args.commit,
               "lock_sha256": sha256_file(root / "env/requirements-eval.lock"),
               "runs": {role: f"{args.job_id}-{role}" for role in ("assignment", "analysis")}}
    if job_file.exists():
        job = read_json(job_file)
        if not args.resume or any(job.get(key) != value for key, value in binding.items()):
            raise ValueError("Existing job requires --resume and identical Pod/commit/lock/run binding")
    else:
        job = binding | {"stages": [], "started_at": datetime.now(timezone.utc).isoformat()}
    job.update(status="RUNNING", logs_closed=False, origin=origin, budget_usd=budget,
               stop_deadline_utc=deadline.isoformat(), watchdog_id=os.environ["MMDL_WATCHDOG_ID"],
               storage_reserve_gib=float(os.environ["MMDL_STORAGE_RESERVE_GIB"]))
    write_json(job_file, job)
    env = os.environ | {"PIP_NO_CACHE_DIR": "1", "PYTHONPATH": str(root / "src")}

    def stage(label, command):
        if datetime.now(timezone.utc) >= deadline:
            raise RuntimeError("External stop deadline reached; preserve results for recovery")
        log, stream = _open_stage_log(job_file.parent, args.job_id, len(job["stages"]), label)
        begun = time.monotonic()
        print(f"Starting {label}; log={log.name}", flush=True)
        with stream:
            stream.write(json.dumps({"argv": list(map(str, command)), "started_at": datetime.now(timezone.utc).isoformat()}) + "\n")
            stream.flush()
            result = subprocess.run(list(map(str, command)), cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        job["stages"].append({"stage": label, "log": log.name, "seconds": time.monotonic() - begun,
                              "exit_code": result.returncode})
        write_json(job_file, job)
        if result.returncode:
            raise RuntimeError(f"Stage {label} failed; preserved {log.name}")

    python = venv / "bin/python"
    venv_receipt = venv / "mmdl-lock.json"
    try:
        if venv.exists():
            if not venv_receipt.is_file() or read_json(venv_receipt) != {"lock_sha256": binding["lock_sha256"]}:
                raise ValueError("Refusing to modify an unmanaged or different-lock venv")
        else:
            stage("venv", [sys.executable, "-m", "venv", venv])
            stage("install", [python, "-m", "pip", "install", "-r", root / "env/requirements-eval.lock"])
            write_json(venv_receipt, {"lock_sha256": binding["lock_sha256"]})
        stage("package", [python, "-m", "pip", "install", "--no-deps", "-e", root])
        stage("pip-check", [python, "-m", "pip", "check"])
        stage("config", [python, "-c", "import sys; from pathlib import Path; from mmdl.runtime.contracts import load_configs; c,h=load_configs(Path(sys.argv[1]),Path(sys.argv[2])); sys.exit(0 if c['status']=='FROZEN' and h['placement']=='gpu_only' else 'Frozen GPU-only protocol required')", args.protocol, args.hardware])
        env["MMDL_PYTHON"] = str(python)
        stage("doctor", ["bash", "scripts/doctor.sh", "--output", artifacts / "jobs" / f"{args.job_id}.doctor.json",
                          "--storage-path", paths["HF_HOME"], "--storage-path", data,
                          "--storage-path", artifacts, "--required-gib", "28"])
        _validate_doctor_gpu(artifacts / "jobs" / f"{args.job_id}.doctor.json", args.hardware)
        stage("download", [python, "-m", "mmdl.data.download", "--data-root", paths["MMDL_DATA_ROOT"],
                            "--cache", paths["HF_HOME"], "--artifact-root", artifacts, "--skip-pro"])
        common = ["bash", "scripts/eval.sh", "--protocol", args.protocol, "--hardware", args.hardware,
                  "--model-ref", args.model_ref, "--model-path", args.model_path, "--data-root", data,
                  "--artifact-root", artifacts, "--public-root", public, "--require-commit", args.commit,
                  "--job-id", args.job_id, "--no-download"]
        for role in ("smoke", "assignment", "analysis"):
            run_id = f"{args.job_id}-{role}"
            command = common + ["--run-id", run_id, "--run-role", role, "--mode", "smoke" if role == "smoke" else "full"]
            if role == "smoke":
                command += ["--sample-id", "validation_Accounting_1"]
            if (artifacts / "runs" / run_id).exists():
                if not args.resume:
                    raise ValueError("Existing run requires explicit --resume")
                command += ["--resume"]
            stage(role, command)
            if role != "smoke":
                stage(f"{role}-rescore", ["bash", "scripts/rescore.sh", "--artifact-root", artifacts,
                                           "--run-id", run_id, "--verify"])
                stage(f"{role}-audit", [python, "scripts/validate_run.py", "--run-dir", artifacts / "runs" / run_id,
                                         "--public-dir", public / run_id])
        git_commit(root, args.commit)
        job.update(status="REMOTE_VERIFIED", logs_closed=True, ended_at=datetime.now(timezone.utc).isoformat())
        write_json(job_file, job)
        # Packing output must not enter a still-open log selected into this same archive.
        subprocess.run([str(python), "-m", "mmdl.runtime.bundle", "create", "--artifact-root", str(artifacts),
                        "--public-root", str(public), "--job-id", args.job_id, "--output",
                        str(artifacts / "bundles" / f"{args.job_id}.tar.gz")], cwd=root, env=env, check=True)
    except Exception as exc:
        job.update(status="RECOVERY_REQUIRED", logs_closed=True, error=str(exc))
        write_json(job_file, job)
        raise


def main(argv=None):
    args = arguments(argv)
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN", "job_id": args.job_id, "commit": args.commit,
                          "steps": ["clean GitHub checkout", "exact-lock venv", "doctor", "pinned download",
                                    "one smoke", "assignment 900", "analysis 900", "verify and bundle"],
                          "external_gate": "approved budget, persistent volume and independently armed STOP watchdog"}))
        return
    execute(args, Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
