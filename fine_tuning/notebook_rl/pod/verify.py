"""Fast Pod pre-flight (no GPU) and the volume manifest it checks.

CLI: python -m fine_tuning.notebook_rl.pod.verify --job JOB.json --commit SHA --volume-root DIR
         --run-dir DIR [--repo-root DIR] [--quick] [--mount-path /workspace] [--disk-path DIR]
         [--workers 4]
Writes <run-dir>/preflight.json (always) and exits 1 on any failed check. `min_free_gib`, the stage and
the training-data pins (`data`) come from the job. Training stages also check `eval_isolation`: the
benchmark root every training-side guard forbids, Path.home()/"mmdl-data/evaluation" (local_run
_install_actor_guards, passrate, local_train, local_eval; MMDL_DATA_ROOT is not consulted), must resolve
to <volume_root>/eval-data, and no other <mount>/*/eval-data may exist. Evaluation stages skip the
training data and check `eval_data` (the MMMU files recorded in the volume manifest, full sha256). --quick compares model files by size plus first/last
64 MiB only; it is recorded as `quick` and never reported as a full hash check (the staged data bundle
is always fully hashed).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import data_bundle
from .job import EVAL_STAGES, SOURCE_REBUILT_STAGES, TRAINING_STAGES, check_repo_root
from .common import (HEX40, QUICK_SPAN, file_digests, quick_digest, read_json, sha256_bytes,
                     sha256_file, utc_now, write_json)

MANIFEST_NAME = "volume_manifest.json"
MANIFEST_SCHEMA = "pod-volume-manifest-v1"
ORIGINS = ("https://github.com/jang2296/MMDL_test.git", "git@github.com:jang2296/MMDL_test.git")
SUBMISSION_ORIGINS = ("https://github.com/jang2296/MMDL.git", "git@github.com:jang2296/MMDL.git")
BASELINE = "manifests/models/baseline.json"
POOL_FREEZE_TIMEOUT_SECONDS = 300
EVAL_DIR = "eval-data"  # <volume_root>/eval-data: every MMMU byte on the volume (cache, data root, .arrow)
EVAL_DATA_ROOT = "eval-data/mmmu"  # the evaluator's --data-root (symlinks + manifest.json)
EVAL_CACHE = "eval-data/hf-cache"  # the dataset's own HF cache, never the training HF_HOME
GUARD_EVAL_ROOT = "mmdl-data/evaluation"  # Path.home()/this is in every training-side guard


def venv_name(lock_sha256: str) -> str:
    return f"venv-{lock_sha256[:16]}"


def lock_closure(lock: str | Path) -> dict[str, str]:
    """sha256 of the lock and of every file it pulls in with -r/-c, recursively.

    pip resolves an include relative to the including file; names are relative to the top
    lock's directory and may not leave it. The top-lock sha alone misses a changed include
    (env/requirements-experiments.lock is `-r requirements-eval.lock` + 2 pins).
    """
    top = Path(lock).resolve()
    found: dict[str, str] = {}
    pending = [top]
    while pending:
        path = pending.pop()
        name = str(path.relative_to(top.parent))  # ValueError when an include escapes env/
        if name in found:
            continue
        found[name] = sha256_file(path)
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("-r", "--requirement", "-c", "--constraint"):
                pending.append((path.parent / parts[1]).resolve())
            elif parts and parts[0].startswith(("--requirement=", "--constraint=")):
                pending.append((path.parent / parts[0].split("=", 1)[1]).resolve())
    return found


def write_volume_manifest(volume_root: str | Path, *, commit: str, environment: str,
                          lock: str | Path, venv: str | Path, freeze: str | Path,
                          snapshot: str | Path, model_files: list[dict[str, Any]],
                          python_version: str | None = None, now: str | None = None,
                          span: int = QUICK_SPAN,
                          eval_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record the prepared volume. Re-hashes every model file once and cross-checks `model_files`.

    `eval_data` ({repo_id, revision, files} from the acquisition) is re-hashed under EVAL_DATA_ROOT the
    same way and recorded with the sha256 of the evaluator's manifest.json; None records no eval data.
    """
    root = Path(volume_root).resolve()
    snapshot = Path(snapshot)
    lock_sha = sha256_file(lock)
    if Path(venv).resolve() != root / venv_name(lock_sha):
        raise ValueError("venv directory name must be venv-<lock sha256[:16]> inside the volume")
    files = []
    for item in sorted(model_files, key=lambda f: f["path"]):
        size, full, quick = file_digests(snapshot / item["path"], span)
        if size != item["bytes"] or full != item["sha256"]:
            raise ValueError(f"model file differs from acquisition record: {item['path']}")
        files.append({"path": item["path"], "bytes": size, "sha256": full, "quick_sha256": quick})
    record = None
    if eval_data is not None:
        check_eval_files(root, eval_data["files"])
        record = {"repo_id": eval_data["repo_id"], "revision": eval_data["revision"],
                  "data_root": EVAL_DATA_ROOT, "cache": EVAL_CACHE, "usage": "evaluation_only",
                  "training_allowed": False,
                  "data_manifest_sha256": sha256_file(root / EVAL_DATA_ROOT / "manifest.json"),
                  "files": [{k: f[k] for k in ("path", "bytes", "sha256")} for f in eval_data["files"]]}
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "VOLUME_READY", "commit": commit,
        "environment": environment, "lock_file": f"env/requirements-{environment}.lock",
        "lock_sha256": lock_sha, "lock_closure_sha256": lock_closure(lock),
        "python_version": python_version or ".".join(map(str, sys.version_info[:3])),
        "venv": str(root / venv_name(lock_sha)), "pip_freeze_file": Path(freeze).name,
        "pip_freeze_sha256": sha256_file(freeze),
        "model_snapshot": str(snapshot.resolve().relative_to(root)),
        "quick_span_bytes": span, "model_files": files, "eval_data": record,
        "created_utc": now or utc_now()}
    write_json(root / MANIFEST_NAME, manifest)
    return manifest


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def _safe_relative(value: str) -> str:
    parts = Path(value).parts
    if Path(value).is_absolute() or not parts or ".." in parts:
        raise ValueError(f"unsafe relative path in manifest: {value!r}")
    return value


def check_eval_files(volume: Path, files: list[dict[str, Any]]) -> None:
    """Each file under EVAL_DATA_ROOT resolves inside EVAL_DIR with its recorded size and sha256."""
    base = volume / EVAL_DIR
    for item in files:
        path = (volume / EVAL_DATA_ROOT / _safe_relative(item["path"])).resolve(strict=True)
        if base not in path.parents:
            raise ValueError(f"eval file resolves outside {base}: {item['path']}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"eval data differs from its record: {item['path']}")


def check_eval_data(volume_root: str | Path, record: Any) -> str:
    """Evaluation stages: the volume holds the recorded MMMU validation files (full sha256)."""
    if not isinstance(record, dict):
        raise ValueError("the volume manifest records no eval data (prep job eval_data: false)")
    volume = Path(volume_root).resolve()
    if record.get("data_root") != EVAL_DATA_ROOT:
        raise ValueError(f"eval data root must be {EVAL_DATA_ROOT}")
    if sha256_file(volume / EVAL_DATA_ROOT / "manifest.json") != record["data_manifest_sha256"]:
        raise ValueError("eval data manifest.json differs from the volume manifest")
    check_eval_files(volume, record["files"])
    return f"{len(record['files'])} {record['repo_id']}@{record['revision']} files, full sha256"


def check_eval_isolation(volume_root: str | Path, mount_path: str | Path) -> str:
    """Training stages: the guards' benchmark root covers <volume_root>/eval-data, the only eval root.

    Evaluated in this process, whose HOME is the one start.sh passes on to the smoke/main commands.
    """
    eval_root = Path(volume_root).resolve() / EVAL_DIR
    guard = Path.home() / GUARD_EVAL_ROOT
    if guard.resolve() != eval_root:
        raise ValueError(f"{guard} resolves to {guard.resolve()}, not {eval_root}: the training guards "
                         f"would not forbid the eval data; run `mkdir -p {guard.parent} && "
                         f"ln -s {eval_root} {guard}` on this Pod")
    others = sorted(str(p) for p in Path(mount_path).glob(f"*/{EVAL_DIR}") if p.resolve() != eval_root)
    if others:
        raise ValueError(f"eval data outside the guarded root: {others}")
    return f"{guard} -> {eval_root}: every training-side guard forbids the eval data"


def check_bundle(data: dict[str, Any]) -> str:
    """The staged bundle matches its pinned manifest and holds the dataset manifest and vault."""
    errors = data_bundle.verify(data["bundle_manifest"], data["bundle_manifest_sha256"],
                                covers=(data["dataset_manifest"], data["vault_root"]))
    if errors:
        raise ValueError(f"{len(errors)} staged data problem(s), first {errors[:3]}")
    return "every bundle file staged at its absolute path with the manifest sha256"


def check_dataset(python: str | Path, repo: str | Path, data: dict[str, Any],
                  timeout: float = POOL_FREEZE_TIMEOUT_SECONDS) -> str:
    """Validate v17 assembled data, or require legacy pool_freeze.verify() == [] for its pinned manifest.

    pool_freeze needs numpy/PIL/yaml, so it runs in the volume venv (already checked by `venv`)
    through its own CLI, which calls verify() and prints the error count; this process stays stdlib.
    """
    if sha256_file(data["dataset_manifest"]) != data["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest sha256 differs from the job's pinned value")
    root = Path(repo).resolve()
    environment = {**os.environ, "PYTHONPATH": f"{root}/src{os.pathsep}{root}"}
    manifest = read_json(data["dataset_manifest"])
    if isinstance(manifest, dict) and manifest.get("version") == "v17-source-bound-data-v1":
        assembled = Path(data["dataset_manifest"]).absolute().parent
        if (Path(data["dataset_manifest"]).name != "dataset-manifest.json"
                or Path(data["vault_root"]).resolve() != (assembled / "vault").resolve()):
            raise ValueError("v17 assembled manifest/vault path mismatch")
        program = ("import hashlib,json,sys; from pathlib import Path; "
                   "from fine_tuning.error_note.v17_data import validate; "
                   "root=Path(sys.argv[1]); report=validate(root); "
                   "report[\'manifest_sha256\']=hashlib.sha256((root/\'dataset-manifest.json\').read_bytes()).hexdigest(); "
                   "print(json.dumps(report,sort_keys=True))")
        done = subprocess.run([str(python), "-B", "-c", program, str(assembled)],
                              capture_output=True, text=True, timeout=timeout, env=environment, cwd=root)
        try:
            report = json.loads(done.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            raise ValueError(f"v17 data validation printed no report (exit {done.returncode}): "
                             f"{done.stderr[-300:]}") from None
        if (done.returncode != 0 or not isinstance(report, dict)
                or report.get("status") != "PASS_CPU_V17_DATA_WITH_LIMITS"
                or report.get("manifest_sha256") != data["dataset_manifest_sha256"]):
            raise ValueError(f"v17 data validation failed (exit {done.returncode}): {done.stderr[-300:]}")
        return "v17_data.validate(assembled_root) passed in the volume venv"
    done = subprocess.run(
        [str(python), "-B", "-m", "fine_tuning.notebook_rl.pool_freeze", "verify",
         "--manifest", data["dataset_manifest"], "--vault", data["vault_root"]],
        capture_output=True, text=True, timeout=timeout, env=environment, cwd=root)
    try:
        report = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise ValueError(f"pool_freeze verify printed no report (exit {done.returncode}): "
                         f"{done.stderr[-300:]}") from None
    if (done.returncode != 0 or not isinstance(report, dict) or report.get("status") != "VERIFIED"
            or report.get("errors") != 0
            or report.get("manifest_sha256") != data["dataset_manifest_sha256"]):
        first = report.get("first_errors", [])[:3] if isinstance(report, dict) else report
        raise ValueError(f"pool_freeze.verify is not []: exit {done.returncode}, first {first}")
    return "pool_freeze.verify(dataset_manifest, vault_root) == [] in the volume venv"


class Preflight:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def run(self, name: str, function: Any) -> Any:
        try:
            detail = function()
            self.checks.append({"name": name, "ok": True, "detail": detail})
            return detail
        except Exception as exc:  # every failure is recorded, then the run is refused
            self.checks.append({"name": name, "ok": False,
                                "detail": f"{type(exc).__name__}: {exc}"})
            return None

    @property
    def ok(self) -> bool:
        return all(check["ok"] for check in self.checks)


def run_checks(*, repo_root: str | Path, commit: str, volume_root: str | Path,
               run_dir: str | Path, data: dict[str, Any] | None, min_free_gib: float,
               quick: bool = False, mount_path: str = "/workspace",
               disk_path: str | Path | None = None, workers: int = 4,
               stage: str | None = None, job: dict[str, Any] | None = None,
               repository: str = "MMDL_test") -> dict[str, Any]:
    """stage None (direct callers) runs the training-data checks only; the CLI passes the job stage."""
    started = time.monotonic()
    repo, volume = Path(repo_root), Path(volume_root)
    state = Preflight()
    context: dict[str, Any] = {}
    if repository not in {"MMDL_test", "MMDL"}:
        raise ValueError("repository must be MMDL_test or MMDL")

    def clone() -> str:
        if job is not None and "repo_root" in job:
            check_repo_root(job, Path(__file__).resolve().parents[3])
            if repo.resolve() != Path(job["repo_root"]) or commit != job["commit"]:
                raise ValueError("verify arguments differ from job repo_root/commit")
        if not HEX40.match(commit):
            raise ValueError("commit must be a 40-char lowercase hex SHA")
        if _git(repo, "rev-parse", "HEAD") != commit:
            raise ValueError("HEAD differs from the job commit")
        if _git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("clone is not clean")
        origin = _git(repo, "remote", "get-url", "origin")
        if origin not in (SUBMISSION_ORIGINS if repository == "MMDL" else ORIGINS):
            raise ValueError(f"origin is not {repository}: {origin}")
        if repository == "MMDL" and (
                Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo.resolve()
                or not _git(repo, "for-each-ref", "--format=%(refname)", "--contains", commit, "refs/remotes/origin")):
            raise ValueError("a fetched exact MMDL clone is required")
        return f"clean {repository} clone at {commit}"

    def manifest() -> str:
        found = read_json(volume / MANIFEST_NAME)
        if found.get("schema") != MANIFEST_SCHEMA or found.get("status") != "VOLUME_READY":
            raise ValueError("volume manifest is not a VOLUME_READY pod-volume-manifest-v1")
        context["manifest"] = found
        return f"volume prepared at commit {found.get('commit')}"

    def lock() -> str:
        found = context["manifest"]
        path = repo / _safe_relative(found["lock_file"])
        current = sha256_file(path)
        if current != found["lock_sha256"]:
            raise ValueError("lock sha256 differs from the volume manifest")
        if lock_closure(path) != found.get("lock_closure_sha256"):
            raise ValueError("a lock or one of its -r/-c includes differs from the volume manifest")
        return current

    def venv() -> str:
        found = context["manifest"]
        path = Path(found["venv"])
        if path != volume.resolve() / venv_name(found["lock_sha256"]):
            raise ValueError("manifest venv path is not venv-<lock sha256[:16]> in the volume")
        config = {key.strip(): value.strip() for key, _, value in
                  (line.partition("=") for line in (path / "pyvenv.cfg").read_text().splitlines())}
        if config.get("version") != found["python_version"]:
            raise ValueError("pyvenv.cfg version differs from the manifest")
        reported = subprocess.run(
            [str(path / "bin/python"), "-I", "-c",
             "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
            check=True, capture_output=True, text=True, timeout=30).stdout.strip()
        if reported != found["python_version"]:
            raise ValueError(f"venv python reports {reported}")
        freeze = volume / _safe_relative(found["pip_freeze_file"])
        if sha256_file(freeze) != found["pip_freeze_sha256"]:
            raise ValueError("pip freeze record changed")
        # The record file proves nothing about what is installed now: freeze the live venv the
        # way volume.sh did (cwd = clone, PYTHONPATH = <clone>/src:<clone>) and compare.
        root = repo.resolve()
        environment = {**os.environ, "PYTHONPATH": f"{root}/src{os.pathsep}{root}"}
        live = subprocess.run([str(path / "bin/python"), "-m", "pip", "freeze", "--all"],
                              capture_output=True, timeout=60, env=environment, cwd=root)
        if live.returncode != 0:
            raise ValueError(f"venv pip freeze failed: {live.stderr[-200:].decode(errors='replace')}")
        if sha256_bytes(live.stdout) != found["pip_freeze_sha256"]:
            raise ValueError("installed packages differ from the recorded pip freeze")
        return f"python {reported}; live pip freeze matches the record"

    def pinned() -> str:
        expected = {f["path"]: (f["bytes"], f["sha256"])
                    for f in read_json(repo / BASELINE)["files"]}
        recorded = {f["path"]: (f["bytes"], f["sha256"])
                    for f in context["manifest"]["model_files"]}
        if expected != recorded:
            raise ValueError("volume model files differ from manifests/models/baseline.json")
        return f"{len(recorded)} files match the pinned baseline manifest"

    def model() -> str:
        found = context["manifest"]
        snapshot = (volume / _safe_relative(found["model_snapshot"]))
        base = volume.resolve()

        def one(item: dict[str, Any]) -> None:
            path = snapshot / _safe_relative(item["path"])
            resolved = path.resolve(strict=True)
            if base not in resolved.parents:
                raise ValueError(f"model file resolves outside the volume: {item['path']}")
            if resolved.stat().st_size != item["bytes"]:
                raise ValueError(f"size mismatch: {item['path']}")
            if quick:
                if quick_digest(resolved, found["quick_span_bytes"])[1] != item["quick_sha256"]:
                    raise ValueError(f"quick hash mismatch: {item['path']}")
            elif sha256_file(resolved) != item["sha256"]:
                raise ValueError(f"sha256 mismatch: {item['path']}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for future in [pool.submit(one, item) for item in found["model_files"]]:
                future.result()
        mode = "quick (size + first/last span hash; NOT a full sha256 check)" if quick else "full"
        return f"{len(found['model_files'])} model files, hash mode {mode}"

    def disk() -> str:
        target = Path(disk_path) if disk_path else Path(run_dir)
        while not target.exists():
            target = target.parent
        free = shutil.disk_usage(target).free / 2**30
        if free < min_free_gib:
            raise ValueError(f"{free:.1f} GiB free at {target} < {min_free_gib} GiB")
        return f"{free:.1f} GiB free at {target}"

    def mount() -> str:
        # Same probe as runpod_setup.sh; os.path.ismount only where the command is missing.
        if shutil.which("mountpoint"):
            mounted = subprocess.run(["mountpoint", "-q", mount_path], timeout=10).returncode == 0
        else:
            mounted = os.path.ismount(mount_path)
        if not mounted:
            raise ValueError(f"{mount_path} is not a mount point")
        return f"{mount_path} is a mount"

    def bundle() -> str:
        if not isinstance(data, dict):
            raise ValueError("the job binds no training data (job.data)")
        return check_bundle(data)

    def dataset() -> str:
        if not isinstance(data, dict):
            raise ValueError("the job binds no training data (job.data)")
        return check_dataset(Path(context["manifest"]["venv"]) / "bin/python", repo, data)

    venv_ok = False
    state.run("clone", clone)
    if state.run("volume_manifest", manifest) is not None:
        state.run("lock_sha256", lock)
        venv_ok = state.run("venv", venv) is not None
        state.run("pinned_baseline", pinned)
        state.run("model_files", model)
    if stage in EVAL_STAGES:
        state.run("eval_data", lambda: check_eval_data(volume, context["manifest"].get("eval_data")))
    elif stage in SOURCE_REBUILT_STAGES:
        state.checks.append({"name": "data_source_rebuilt", "ok": True,
                             "detail": "no staged bundle: the chain's prepare-data step rebuilds the data from the "
                                       "pinned recipe and stops the chain unless every role, order and hash matches"})
    else:
        state.run("data_bundle", bundle)
        if venv_ok:
            state.run("dataset_vault", dataset)
        else:
            state.checks.append({"name": "dataset_vault", "ok": False,
                                 "detail": "not run: the volume venv was not verified"})
    if stage in TRAINING_STAGES:
        state.run("eval_isolation", lambda: check_eval_isolation(volume, mount_path))
    state.run("disk_free", disk)
    state.run("workspace_mount", mount)
    return {"version": "pod-preflight-v1", "ok": state.ok, "commit": commit,
            "volume_root": str(volume), "hash_mode": "quick" if quick else "full",
            "full_sha256_verified": state.ok and not quick, "checks": state.checks,
            "elapsed_seconds": round(time.monotonic() - started, 3), "utc": utc_now()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="frozen job copy: min_free_gib and data pins")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--volume-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mount-path", default="/workspace")
    parser.add_argument("--disk-path")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repository", choices=("MMDL_test", "MMDL"), default="MMDL_test")
    args = parser.parse_args(argv)
    job = read_json(args.job)
    result = run_checks(repo_root=args.repo_root, commit=args.commit,
                        volume_root=args.volume_root, run_dir=args.run_dir, quick=args.quick,
                        data=job.get("data"), min_free_gib=job["min_free_gib"],
                        mount_path=args.mount_path, disk_path=args.disk_path,
                        workers=args.workers, stage=job["stage"], job=job, repository=args.repository)
    write_json(Path(args.run_dir) / "preflight.json", result)
    print(json.dumps({k: result[k] for k in ("ok", "hash_mode", "full_sha256_verified",
                                              "elapsed_seconds")}))
    for check in result["checks"]:
        if not check["ok"]:
            print(f"FAILED {check['name']}: {check['detail']}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
