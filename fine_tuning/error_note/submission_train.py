"""Public local training: restore sources, or run the existing sealed 15-stage v18b chain."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl.pod.common import sha256_bytes

ROOT = Path(__file__).resolve().parents[2]


def _external(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} requires a path")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or not Path(expanded).is_absolute():
        raise ValueError(f"{name} needs an absolute path; set its environment variable in the config")
    path = Path(expanded).resolve()
    if path == Path("/") or path.is_relative_to(ROOT):
        raise ValueError(f"{name} must be outside the repository")
    return str(path)


def _python(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("configure executable python/data_python paths in separate environments")
    expanded = os.path.expandvars(os.path.expanduser(value))
    found = shutil.which(expanded)
    if found is None:
        raise ValueError(f"Python interpreter not executable: {expanded!r}; configure python/data_python or its environment variable")
    # Resolving a venv's python symlink would select the system interpreter instead.
    return os.path.abspath(found)


def parameters(config_path: Path, *, run_id: str | None = None,
               risk_acceptance: str | None = None, prepare_only: bool = False) -> dict:
    import yaml
    from .plan import PARAMS_VERSION_V18B_SPLIT
    from .v18b_execution import validate_settings

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    keys = {"version", "python", "data_python", "model_path", "data_root", "cache_dir", "artifact_root",
            "run_id", "gpu_type", "timeouts", "training_estimate_seconds", "training_estimate_basis",
            "guard_risk_acceptance", "execution_settings"}
    if not isinstance(config, dict) or set(config) != keys or config["version"] != "v18b-public-training-v1":
        raise ValueError("expected the complete v18b-public-training-v1 config")
    identifier = run_id or config["run_id"]
    if prepare_only and isinstance(identifier, str):
        identifier += "-prepare"
    settings = validate_settings(dict(config["execution_settings"], run_id=identifier))
    if settings.get("execution_mode") != "on_prem":
        raise ValueError("public finetune uses on-prem execution only")
    paths = {key: _external(config[key], key) for key in ("model_path", "data_root", "cache_dir", "artifact_root")}
    if Path(paths["model_path"]).name != BASE_REVISION:
        raise ValueError("model_path must be the pinned Qwen snapshot directory")
    artifact = Path(paths.pop("artifact_root"))
    run = artifact / identifier
    for first, second in ((run, Path(paths["data_root"])), (run, Path(paths["cache_dir"])),
                          (run, Path(paths["model_path"]))):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError("run directory must be separate from data, cache and original model")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    result = {"version": PARAMS_VERSION_V18B_SPLIT, "repo_root": str(ROOT), "commit": commit,
              "python": _python(config["python"]), "data_python": _python(config["data_python"]), **paths,
              "run_dir": str(run), "out_root": str(run / "output"),
              "backup_destination": str(artifact / "backups" / identifier),
              "gpu_type": config["gpu_type"], "timeouts": config["timeouts"],
              "training_estimate_seconds": config["training_estimate_seconds"],
              "training_estimate_basis": config["training_estimate_basis"], "execution_settings": settings}
    if result["python"] == result["data_python"]:
        raise ValueError("use separate runtime and data Python environments from their respective locks")
    risk = risk_acceptance if risk_acceptance is not None else config["guard_risk_acceptance"]
    if risk is not None:
        result["guard_risk_acceptance"] = risk
    return result


def _write_once(path: Path, value: dict) -> str:
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(content)
    return sha256_bytes(content)


def data_command(params: dict, command: str) -> list[str]:
    argv = [params["data_python"], "-m", "fine_tuning.error_note.v18b_data", command,
            "--output", params["data_root"], "--tokenizer", params["model_path"]]
    if command == "prepare":
        argv += ["--recipe", str(ROOT / "fine_tuning/error_note/v18_recipe.json"), "--cache-dir", params["cache_dir"]]
    return argv


def main(argv: list[str] | None = None) -> int:
    from .plan import plan, verify_source_v17
    from fine_tuning.notebook_rl.pod.chain import check_spec

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="restore and verify training data; no model inference")
    mode.add_argument("--plan-only", action="store_true", help="write the real chain and params without executing stages")
    mode.add_argument("--train", action="store_true", help="execute preparation, training, fresh reload and export locally")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/v18b.yaml")
    parser.add_argument("--run-id", help="a new directory name for this independent training run")
    parser.add_argument("--guard-risk-acceptance", help="this operator's reason for accepting a measured guard false-alarm risk")
    args = parser.parse_args(argv)
    try:
        params = parameters(args.config, run_id=args.run_id, risk_acceptance=args.guard_risk_acceptance,
                            prepare_only=args.prepare_only)
        spec = check_spec(plan("v18b-train", params))
        run = Path(params["run_dir"])
        run.mkdir(parents=True, exist_ok=False)
        _write_once(run / "params.json", params)
        spec_path = run / "train-spec.json"
        spec_sha = _write_once(spec_path, spec)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))),
                   MMDL_RUN_DIR=str(run), MMDL_JOB_STAGE=spec["meta"]["job_stage"])
        if args.prepare_only:
            verify_source_v17(ROOT, params["commit"], run / "source-prepare.json", repository="MMDL")
            for command in ("prepare", "verify"):
                subprocess.run(data_command(params, command), cwd=ROOT, env=env, check=True)
            print(json.dumps({"status": "DATA_VERIFIED", "data_root": params["data_root"], "run_dir": str(run)}))
            return 0
        if args.plan_only:
            print(json.dumps({"status": "PLANNED_NOT_EXECUTED", "spec": str(spec_path), "spec_sha256": spec_sha,
                              "steps": [step["name"] for step in spec["steps"]]}))
            return 0
        result = subprocess.run([params["python"], "-m", "fine_tuning.notebook_rl.pod.chain", "run",
                                 "--spec", str(spec_path), "--spec-sha256", spec_sha], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
        exported = Path(params["out_root"]) / "export-final"
        print(json.dumps({"status": "TRAINING_EXPORTED", "run_dir": str(run), "export_root": str(exported),
                          "model_ref": str(exported / "model_ref.json")}))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"finetune: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
