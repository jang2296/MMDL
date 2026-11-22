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
    "a100_sxm_80gb.yaml": (r"NVIDIA A100(?:-SXM4)?[- ]80GB", 79 * 1024**3),
    "a100_pcie_80gb.yaml": (r"NVIDIA A100(?:-PCIE)?[- ]80GB", 79 * 1024**3),
}
_GPU_SAMPLE_INTERVAL_MS = 500
_INFERENCE_STAGES = frozenset({"smoke", "evaluation"})
_PRO_FULL_MIN_SECONDS = 20 * 60 * 60
# Bootstrap Python has no PyYAML, so contracts.SUITE_COUNTS / SUITE_DATA_DIRS are mirrored (unit-tested).
SUITE_COUNTS = {"mmmu-val": 900, "mmmu-pro": 1730, "mmmu-test": 10500}
SUITE_DATA_DIRS = {"mmmu-val": "evaluation/mmmu", "mmmu-pro": "evaluation/mmmu_pro",
                   "mmmu-test": "evaluation/mmmu_test"}


def arguments(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["evaluate"], default="evaluate")
    parser.add_argument("--target", choices=["runpod"], default="runpod")
    parser.add_argument("--suite", choices=list(SUITE_COUNTS), default="mmmu-val")
    parser.add_argument("--model", choices=["base", "finetuned"], default="base")
    parser.add_argument("--score", choices=["v8", "both"], default="v8")
    parser.add_argument("--engine-validation", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--protocol", type=Path, default=root / "configs/eval/mmmu_val_v8.yaml")
    parser.add_argument("--hardware", type=Path, default=root / "configs/hardware/rtx4090_24gb.yaml")
    parser.add_argument("--model-ref", type=Path, default=root / "manifests/models/baseline.json")
    parser.add_argument("--model-path")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Install, probe and download only; no smoke or full inference")
    args = parser.parse_args(argv)
    if args.model_path is None and args.model == "base":
        args.model_path = os.environ.get("MMDL_MODEL_PATH", "Qwen/Qwen3-VL-4B-Instruct")
    if args.suite == "mmmu-test" and args.score != "v8":
        parser.error("MMMU test is scored with V8 only; --score both is not registered")
    if args.suite == "mmmu-test" and args.protocol == root / "configs/eval/mmmu_val_v8.yaml":
        args.protocol = root / "configs/eval/mmmu_test_lowres2k_v8.yaml"
    if args.suite != "mmmu-val" or args.model == "finetuned":
        profile = "4090" if "24gb" in args.hardware.name else "a100"
        if args.protocol == root / "configs/eval/mmmu_val_v8.yaml":
            name = (f"mmmu_pro_standard10_v8_{profile}.yaml" if args.suite == "mmmu-pro"
                    else f"mmmu_val_v18b_ft_{profile}.yaml")
            args.protocol = root / "configs/eval" / name
        if args.model == "finetuned" and args.model_ref == root / "manifests/models/baseline.json":
            args.model_ref = root / "manifests/models/v18b.json"
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
    policy = os.environ.get("MMDL_COST_POLICY", "")
    if policy not in {"", "user_waived"}:
        raise ValueError("MMDL_COST_POLICY must be empty or explicitly user_waived")
    try:
        reserve = float(os.environ["MMDL_STORAGE_RESERVE_GIB"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Explicit positive storage reserve is required") from exc
    if not math.isfinite(reserve) or reserve < 1:
        raise ValueError("Explicit positive storage reserve is required")
    if policy == "user_waived":
        budget = deadline = watchdog = None
    else:
        try:
            budget = float(os.environ["MMDL_BUDGET_USD"])
            deadline = datetime.fromisoformat(os.environ["MMDL_STOP_DEADLINE_UTC"].replace("Z", "+00:00"))
            watchdog = os.environ["MMDL_WATCHDOG_ID"]
        except (KeyError, ValueError) as exc:
            raise ValueError("Explicit budget, deadline and watchdog are required") from exc
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Explicit positive budget is required")
        if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
            raise ValueError("External watchdog deadline is absent or expired")
        if not watchdog:
            raise ValueError("An armed external watchdog is required")
    if not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("Actual Pod ID must be recorded")
    if shutil.disk_usage(volume).free < (35 + reserve) * 1024**3:
        raise RuntimeError("Insufficient volume headroom for installation/download/result peak estimate")
    return paths, deadline, budget


def _validate_origin(origin):
    if not TEAM_ORIGIN.fullmatch(origin):
        raise ValueError("RunPod checkout must originate from github.com/jang2296/MMDL")


def _validate_hardware_profile(hardware):
    profile = Path(hardware).name
    if profile not in HARDWARE_PROFILES:
        raise ValueError("Only registered RTX or A100 hardware profiles are allowed")
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


def _protocol_backend(protocol):
    text = Path(protocol).read_text(encoding="utf-8")
    matches = re.findall(r"^\s*backend:\s*([A-Za-z0-9_.-]+)\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Protocol must declare exactly one backend before installation")
    backend = matches[0].lower()
    if backend not in {"transformers", "vllm"}:
        raise ValueError(f"Unsupported evaluation backend: {backend}")
    return backend


def _lock_path(root, backend, submission=False):
    path = root / "env" / ("requirements-proof.lock" if submission else
                            "requirements-vllm.lock" if backend == "vllm" else "requirements-eval.lock")
    if not path.is_file():
        raise FileNotFoundError(f"Missing exact dependency lock for {backend}: {path}")
    return path


def _lock_receipt(lock):
    receipt: dict[str, object] = {"lock_sha256": sha256_file(lock)}
    if lock.name == "requirements-proof.lock":
        receipt["included_locks_sha256"] = {
            "requirements-vllm.lock": sha256_file(lock.parent / "requirements-vllm.lock")}
    return receipt


def _environment_commands(root, lock, python, submission):
    if submission:
        pip = ["uv", "pip"]
        return {"venv": ["uv", "venv", "--python", sys.executable, python.parent.parent],
                "install": [*pip, "install", "--python", python, "--index-strategy", "unsafe-best-match", "-r", lock],
                "package": [*pip, "install", "--python", python, "--no-deps", "-e", root],
                "pip-check": [*pip, "check", "--python", python]}
    return {"venv": [sys.executable, "-m", "venv", python.parent.parent],
            "install": [python, "-m", "pip", "install", "-r", lock],
            "package": [python, "-m", "pip", "install", "--no-deps", "-e", root],
            "pip-check": [python, "-m", "pip", "check"]}


def _preflight_gpu(hardware):
    """Check host CUDA before installing Torch; never imports torch here."""
    profile = _validate_hardware_profile(hardware)
    pattern, _ = HARDWARE_PROFILES[profile]
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                            check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"CUDA preflight failed: nvidia-smi exited {result.returncode}: {result.stderr.strip()}")
    records = [tuple(part.strip() for part in line.split(",", 1)) for line in result.stdout.splitlines() if line.strip()]
    if len(records) != 1 or not re.fullmatch(pattern, records[0][0], re.IGNORECASE):
        raise RuntimeError(f"CUDA preflight requires exactly one {profile} GPU; detected {records or 'none'}")
    probe = ("import ctypes, os; cuda=ctypes.CDLL('libcuda.so.1'); rc=cuda.cuInit(0); "
             "assert rc == 0, f'cuInit returned {rc}'; fd=os.open('/dev/nvidia-uvm', os.O_RDWR); os.close(fd)")
    result = subprocess.run([sys.executable, "-c", probe], check=False, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"CUDA preflight failed: libcuda/UVM is unavailable ({detail})")
    return {"gpu_name": records[0][0], "driver_version": records[0][1],
            "container_image": os.environ.get("MMDL_CONTAINER_IMAGE")}


def _smoke_sample_ids(backend, protocol=None):
    if backend != "vllm":
        return ["validation_Accounting_1"]
    ids = ["validation_Accounting_1", "validation_Biology_29"]
    if protocol is not None and re.search(r"^\s*scheduling:\s*continuous\s*$",
                                          Path(protocol).read_text(encoding="utf-8"), re.MULTILINE):
        ids.append("validation_Agriculture_1")
    return ids


def _open_stage_log(directory, job_id, completed_stages, label):
    index = completed_stages
    while True:
        log = directory / f"{job_id}.{index:03d}-{label}.log"
        try:
            return log, log.open("x", encoding="utf-8")
        except FileExistsError:
            index += 1


def _gpu_sample_metrics(path, diagnostic=None):
    memories = []
    valid = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            columns = [value.strip() for value in line.split(",")]
            if len(columns) != 3:
                continue
            try:
                memory_mib = int(columns[1])
                int(columns[2])
            except ValueError:
                continue
            if memory_mib < 0:
                continue
            memories.append(memory_mib)
            valid += 1
    return {
        "peak_observed_device_memory_used_bytes": max(memories) * 1024**2 if memories else None,
        "gpu_memory_sample_interval_ms": _GPU_SAMPLE_INTERVAL_MS,
        "memory_scope": "whole_device_sampled_not_allocator_peak",
        "gpu_memory_sample_count": valid,
        "gpu_memory_samples_file": path.name,
        "gpu_memory_monitor_diagnostic": diagnostic or (None if valid else "no valid nvidia-smi samples"),
    }


def _full_run_deadline(args, deadline=None):
    """Reject a short explicit safety deadline; never extend or waive it."""
    if args.suite != "mmmu-pro" or args.prepare_only:
        return deadline
    configured = os.environ.get("MMDL_STOP_DEADLINE_UTC")
    if deadline is None and configured:
        deadline = datetime.fromisoformat(configured.replace("Z", "+00:00"))
    if deadline is not None:
        if deadline.tzinfo is None:
            raise ValueError("MMMU-Pro stop deadline must include a timezone")
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining < _PRO_FULL_MIN_SECONDS:
            raise ValueError("MMMU-Pro FULL requires at least 20 hours before an explicit stop deadline; "
                             "the shorter safety deadline is preserved. Resolve this clash before allocating a Pod")
    return deadline


def _evaluation_commands(args, backend, data, artifacts, public):
    common = ["bash", "scripts/eval.sh", "--protocol", args.protocol, "--hardware", args.hardware,
              "--suite", args.suite, "--model", args.model,
              "--model-ref", args.model_ref, "--data-root", data,
              "--artifact-root", artifacts, "--public-root", public, "--require-commit", args.commit,
              "--job-id", args.job_id]
    if args.engine_validation:
        common += ["--engine-validation", args.engine_validation]
    if args.model_path:
        common += ["--model-path", args.model_path]
    roles = ("smoke", "evaluation") if args.suite == "mmmu-val" and args.model == "base" else ("evaluation",)
    commands = []
    for role in roles:
        score = args.score if role == "evaluation" else "v8"
        command = common + ["--run-id", f"{args.job_id}-{role}", "--run-role", role,
                            "--mode", "smoke" if role == "smoke" else "full", "--score", score]
        if score != "both":
            command += ["--no-download"]
        if role == "smoke":
            for sample_id in _smoke_sample_ids(backend, args.protocol):
                command += ["--sample-id", sample_id]
        commands.append((role, command))
    return commands


def _verification_commands(args, python, artifacts, public):
    run_id = f"{args.job_id}-evaluation"
    return [("evaluation-rescore", ["bash", "scripts/rescore.sh", "--artifact-root", artifacts,
                                   "--run-id", run_id, "--suite", args.suite, "--verify"]),
            ("evaluation-audit", [python, "scripts/validate_run.py", "--run-dir", artifacts / "runs" / run_id,
                                  "--public-dir", public / run_id])]


def _operator_plan(args, root):
    from mmdl.runtime.artifacts import default_path
    from mmdl.runtime.contracts import load_configs, suite_of

    deadline = _full_run_deadline(args)
    cfg, _ = load_configs(args.protocol, args.hardware)
    if args.suite != suite_of(cfg) or cfg["status"] != "FROZEN":
        raise ValueError("Selected suite requires its matching frozen protocol")
    artifacts = default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts")
    cache = default_path("HF_HOME", "mmdl-cache/huggingface")
    data_root = default_path("MMDL_DATA_ROOT", "mmdl-data")
    data = args.data_root or data_root / SUITE_DATA_DIRS[args.suite]
    python = default_path("MMDL_VENV_ROOT", "mmdl-envs/eval") / "bin/python"
    submission = args.suite != "mmmu-val" or args.model == "finetuned" or args.score == "both"
    lock = _lock_path(root, cfg["execution"]["backend"], submission)
    download = [python, "-m", "mmdl.data.download", "--data-root", data_root, "--cache", cache,
                "--artifact-root", artifacts, "--suite", args.suite]
    commands = [("download", download)]
    if not args.prepare_only:
        commands += _evaluation_commands(args, cfg["execution"]["backend"], data, artifacts, artifacts / "public")
        commands += _verification_commands(args, python, artifacts, artifacts / "public")
    return {
        "source": {"required_origin": "https://github.com/jang2296/MMDL", "commit": args.commit,
                   "verification": "clean fixed-commit checkout checked at execution"},
        "model": args.model, "model_ref": str(args.model_ref), "score": args.score,
        "protocol": str(args.protocol), "protocol_id": cfg["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol), "hardware": str(args.hardware),
        "lock": str(lock), "lock_receipt": _lock_receipt(lock),
        "environment": {"manager": "uv" if submission else "pip", "bootstrap_python": sys.executable,
                        "job_python": str(python), "commands": {
                            role: list(map(str, command)) for role, command in
                            _environment_commands(root, lock, python, submission).items()}},
        "enrollment": None if args.prepare_only else {
            "mode": "FULL", "expected_total": cfg["dataset"]["expected_total"],
            "run_id": f"{args.job_id}-evaluation", "one_response_per_sample": True},
        "profile": {"model": cfg["model"], "dataset": cfg["dataset"],
                    "generation": cfg["generation"], "image": cfg["image"], "execution": cfg["execution"]},
        "time_limits": {"chain_timeout_seconds": None, "full_stage_timeout_seconds": None,
                        "subprocess_timeout_seconds": None,
                        "minimum_full_window_seconds": _PRO_FULL_MIN_SECONDS if args.suite == "mmmu-pro" and not args.prepare_only else None,
                        "stop_deadline_utc": deadline.isoformat() if deadline else None,
                        "explicit_safety_deadline_preserved": True,
                        "external_watchdog_required": os.environ.get("MMDL_COST_POLICY") != "user_waived" or deadline is not None},
        "commands": [{"stage": role, "argv": list(map(str, command)), "timeout_seconds": None}
                     for role, command in commands],
        "scoring": {"parser": cfg["parser"], "supplemental": "official pinned CompassVerifier-7B on saved responses" if args.score == "both" else None,
                    "log_pattern": str(artifacts / "jobs" / f"{args.job_id}.*-evaluation.log")},
        "acquisition": "pinned public model and dataset downloaded on this Pod; FT checkpoint restored separately",
    }


def execute(args, root):
    _validate_hardware_profile(args.hardware)
    backend = _protocol_backend(args.protocol)
    submission = args.suite != "mmmu-val" or args.model == "finetuned" or args.score == "both"
    lock = _lock_path(root, backend, submission)
    lock_receipt = _lock_receipt(lock)
    git_commit(root, args.commit)
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
    _validate_origin(origin)
    paths, deadline, budget = require_runtime(root)
    deadline = _full_run_deadline(args, deadline)
    if deadline is not None and not os.environ.get("MMDL_WATCHDOG_ID"):
        raise ValueError("An armed external watchdog is required for an explicit stop deadline")
    artifacts, venv = paths["MMDL_ARTIFACT_ROOT"], paths["MMDL_VENV_ROOT"]
    data = args.data_root.resolve() if args.data_root else paths["MMDL_DATA_ROOT"] / SUITE_DATA_DIRS[args.suite]
    if not data.is_relative_to(paths["MMDL_DATA_ROOT"]):
        raise ValueError("Selected evaluation data must be under MMDL_DATA_ROOT")
    public = artifacts / "public"
    job_file = artifacts / "jobs" / f"{args.job_id}.json"
    binding = {"schema_version": 2, "job_id": args.job_id, "pod_id": os.environ["RUNPOD_POD_ID"], "git_commit": args.commit,
               "suite": args.suite, "backend": backend, **lock_receipt,
               "model": args.model, "score": args.score, "model_ref_sha256": sha256_file(args.model_ref),
               "protocol_sha256": sha256_file(args.protocol),
               "hardware": Path(args.hardware).name, "hardware_sha256": sha256_file(args.hardware),
               "prepare_only": args.prepare_only,
               "runs": {} if args.prepare_only else {"evaluation": f"{args.job_id}-evaluation"}}
    if job_file.exists():
        job = read_json(job_file)
        if not args.resume or any(job.get(key) != value for key, value in binding.items()):
            raise ValueError("Existing job requires --resume and identical Pod/commit/lock/run binding")
    else:
        job = binding | {"stages": [], "started_at": datetime.now(timezone.utc).isoformat()}
    policy = os.environ.get("MMDL_COST_POLICY", "") or "strict"
    job.update(status="RUNNING", logs_closed=False, origin=origin, cost_policy=policy, budget_usd=budget,
               stop_deadline_utc=deadline.isoformat() if deadline else None,
               watchdog_id=os.environ.get("MMDL_WATCHDOG_ID") if deadline else None,
               storage_reserve_gib=float(os.environ["MMDL_STORAGE_RESERVE_GIB"]))
    write_json(job_file, job)
    env = os.environ | {"PIP_NO_CACHE_DIR": "1", "PYTHONPATH": str(root / "src")}

    def stage(label, command):
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            raise RuntimeError("External stop deadline reached; preserve results for recovery")
        log, stream = _open_stage_log(job_file.parent, args.job_id, len(job["stages"]), label)
        begun = time.monotonic()
        print(f"Starting {label}; log={log.name}", flush=True)
        gpu_log = log.with_suffix(log.suffix + ".gpu.csv")
        sampler = sampler_stream = None
        monitor_diagnostic = None
        if label in _INFERENCE_STAGES:
            try:
                sampler_stream = gpu_log.open("x", encoding="utf-8")
                sampler = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits", f"--loop-ms={_GPU_SAMPLE_INTERVAL_MS}"],
                    stdout=sampler_stream, stderr=subprocess.STDOUT, text=True,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                monitor_diagnostic = f"nvidia-smi sampler unavailable: {type(exc).__name__}: {exc}"
                if sampler_stream is not None:
                    sampler_stream.close()
        try:
            with stream:
                stream.write(json.dumps({"argv": list(map(str, command)), "started_at": datetime.now(timezone.utc).isoformat()}) + "\n")
                stream.flush()
                result = subprocess.run(list(map(str, command)), cwd=root, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT)
        finally:
            if sampler is not None:
                try:
                    sampler_returncode = sampler.poll()
                    if sampler_returncode is None:
                        sampler.terminate()
                    elif sampler_returncode != 0:
                        monitor_diagnostic = f"nvidia-smi sampler exited early with code {sampler_returncode}"
                    sampler.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        sampler.kill()
                        sampler.wait(timeout=5)
                        monitor_diagnostic = "nvidia-smi sampler required forced termination"
                    except Exception as exc:  # The sampler may exit between timeout and kill.
                        monitor_diagnostic = f"nvidia-smi sampler cleanup failed: {type(exc).__name__}: {exc}"
                except Exception as exc:  # Monitoring must never change the inference outcome.
                    monitor_diagnostic = f"nvidia-smi sampler cleanup failed: {type(exc).__name__}: {exc}"
            try:
                if sampler_stream is not None and not sampler_stream.closed:
                    sampler_stream.close()
            except Exception as exc:  # The CSV is secondary evidence, not the inference result.
                monitor_diagnostic = f"nvidia-smi sample stream close failed: {type(exc).__name__}: {exc}"
        stage_record = {"stage": label, "log": log.name, "seconds": time.monotonic() - begun,
                        "exit_code": result.returncode}
        if label in _INFERENCE_STAGES:
            stage_record["run_role"] = label
            stage_record.update(_gpu_sample_metrics(gpu_log, monitor_diagnostic))
        job["stages"].append(stage_record)
        write_json(job_file, job)
        if result.returncode:
            raise RuntimeError(f"Stage {label} failed; preserved {log.name}")

    python = venv / "bin/python"
    venv_receipt = venv / "mmdl-lock.json"
    environment_commands = _environment_commands(root, lock, python, submission)
    try:
        job["preflight"] = _preflight_gpu(args.hardware)
        write_json(job_file, job)
        if venv.exists():
            if not venv_receipt.is_file() or read_json(venv_receipt) != lock_receipt:
                raise ValueError("Refusing to modify an unmanaged or different-lock venv")
        else:
            stage("venv", environment_commands["venv"])
            stage("install", environment_commands["install"])
            write_json(venv_receipt, lock_receipt)
        stage("package", environment_commands["package"])
        stage("pip-check", environment_commands["pip-check"])
        stage("config", [python, "-c", "import sys; from pathlib import Path; from mmdl.runtime.contracts import load_configs; c,h=load_configs(Path(sys.argv[1]),Path(sys.argv[2])); sys.exit(0 if c['status']=='FROZEN' and h['placement']=='gpu_only' else 'Frozen GPU-only protocol required')", args.protocol, args.hardware])
        env["MMDL_PYTHON"] = str(python)
        stage("doctor", ["bash", "scripts/doctor.sh", "--output", artifacts / "jobs" / f"{args.job_id}.doctor.json",
                          "--storage-path", paths["HF_HOME"], "--storage-path", data,
                          "--storage-path", artifacts, "--required-gib", "28"])
        _validate_doctor_gpu(artifacts / "jobs" / f"{args.job_id}.doctor.json", args.hardware)
        stage("download", [python, "-m", "mmdl.data.download", "--data-root", paths["MMDL_DATA_ROOT"],
                            "--cache", paths["HF_HOME"], "--artifact-root", artifacts,
                            "--suite", args.suite])
        if args.prepare_only:
            job.update(status="PREPARED", logs_closed=True, ended_at=datetime.now(timezone.utc).isoformat())
            write_json(job_file, job)
            return
        for role, command in _evaluation_commands(args, backend, data, artifacts, public):
            run_id = f"{args.job_id}-{role}"
            if (artifacts / "runs" / run_id).exists():
                if not args.resume:
                    raise ValueError("Existing run requires explicit --resume")
                command += ["--resume"]
            stage(role, command)
        for role, command in _verification_commands(args, python, artifacts, public):
            stage(role, command)
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
        plan = _operator_plan(args, Path(__file__).resolve().parents[3])
        print(json.dumps({"status": "DRY_RUN", "job_id": args.job_id, "commit": args.commit,
                          "suite": args.suite,
                          "prepare_only": args.prepare_only,
                          "steps": ["clean GitHub checkout", "exact-lock venv", "doctor", "pinned download"] +
                                   ([] if args.prepare_only else (
                                       (["one smoke"] if args.suite == "mmmu-val" and args.model == "base" else [])
                                       + [f"one evaluation ({SUITE_COUNTS[args.suite]} samples)", "verify and bundle"])),
                          "external_gate": "persistent volume; strict approved budget and STOP watchdog, or explicit user_waived cost policy",
                          "plan": plan}))
        return
    execute(args, Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
