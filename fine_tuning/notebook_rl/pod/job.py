"""Run contracts `pod-job-v1` (training or evaluation Pod, start.sh) and `pod-prep-v1` (volume prep, volume.sh).

Validation only. Approval is recorded by the user, never generated: `approval.approved_sha256` must equal
the sha256 of the canonical JSON of the job without its `approval` block (`--content-sha256` prints it),
so any later edit of the job, its financial bounds or its data pins voids the approval.
Training stages bind the training data (`data`); evaluation stages (`eval-*`, main = a pod/chain.py
spec from pod/eval_stage.py) carry no `data` key, so training data never needs to be on an eval Pod.
A `pod-prep-v1` job may set `eval_data: true` (default false) to also acquire pinned MMMU validation
into <volume_root>/eval-data (its own approval: the flag is part of the approved content). The financial
bounds live only in the job (`financial`); the shell scripts derive the watchdog flags from it and take
no financial flags of their own.

CLI: python -m fine_tuning.notebook_rl.pod.job JOB.json [--workspace /workspace]
         [--get FIELD | --argv smoke|main | --require-interpreter VOLUME_MANIFEST
          | --financial-argv | --content-sha256]
Without an option it prints the plan (job sha256, job, budget fit) and refuses a job that does not fit
before the financial/GPU deadline. Exit 2 with a message on any refusal.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import posixpath
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fine_tuning.budget import (balance_extension_sha256, balance_extensions, check_balance_extension,
                                check_billing_start, effective_deadline, reserve_deadline)

from .common import HEX40, HEX64, parse_utc, read_json, sha256_bytes, sha256_file

VERSION = "pod-job-v1"
PREP_VERSION = "pod-prep-v1"
TRAINING_STAGES = ("bf16-smoke-throughput", "pilot-300", "main", "note-300", "note-3417", "note-v2",
                   "note-v3-collect", "note-v3-resolve", "note-v3-train", "single-pilot", "single-main",
                   "v18-main")
# v18-main: the chain's first step (v18_data prepare) rebuilds and verifies the data from pinned original sources,
# so the job binds no locally staged bundle (`data`); eval isolation still applies.
SOURCE_REBUILT_STAGES = ("v18-main",)
EVAL_STAGES = ("eval-f0", "eval-ft", "eval-course")
STAGES = TRAINING_STAGES + EVAL_STAGES
ENVIRONMENTS = ("experiments", "vllm", "proof")  # proof = vllm lock + bitsandbytes 0.48.2 (training imports it even under BF16)
KEYS = {"version", "commit", "volume_root", "run_dir", "stage", "smoke", "main",
        "max_main_seconds", "min_free_gib", "financial", "data", "approval"}
PREP_KEYS = {"version", "commit", "volume_root", "environment", "financial", "approval"}
OPTIONAL_KEYS = {"max_approval_age_hours", "repo_root", "financial_mode"}  # an approval older than this is refused
PREP_OPTIONAL_KEYS = {"eval_data"}  # bool, default false
DEFAULT_APPROVAL_AGE_HOURS = 72
MAX_APPROVAL_AGE_HOURS = 168
APPROVAL_KEYS = {"approved_by_user", "approved_utc", "approval_note", "approved_sha256"}
FINANCIAL_KEYS = ("owned_pod_id", "gpu_cutoff_utc", "billing_start_utc", "balance_usd",
                  "reserve_usd", "hourly_usd")  # also the order of the watchdog flags
DATA_KEYS = {"bundle_manifest", "bundle_manifest_sha256", "dataset_manifest",
             "dataset_manifest_sha256", "vault_root"}
POD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SMOKE_TIMEOUT_SECONDS = 300
PREFLIGHT_ALLOWANCE_SECONDS = 600
PREP_MAX_SECONDS = 5400  # = volume.sh max_seconds
RECOVERY_RESERVE_SECONDS = 120  # same reserve budget.py keeps
MAX_MAIN_SECONDS = 7 * 86400
SKEW_SECONDS = 60
GET_FIELDS = ("version", "commit", "volume_root", "run_dir", "stage", "max_main_seconds",
              "min_free_gib", "environment", "owned_pod_id", "eval_data", "repo_root", "financial_mode")


class JobError(ValueError):
    pass


def _workspace_child(value: Any, workspace: str, name: str) -> str:
    if (not isinstance(value, str) or posixpath.normpath(value) != value
            or not value.startswith(workspace + "/") or len(value) <= len(workspace) + 1):
        raise JobError(f"{name} must be a canonical absolute child of {workspace}")
    return value


def _argv(value: Any, name: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or not all(isinstance(item, str) and item and "\0" not in item for item in value)):
        raise JobError(f"{name} must be a non-empty argv list of non-empty strings")
    return value


def content_sha256(job: dict[str, Any]) -> str:
    """sha256 of the canonical JSON (sorted keys, no spaces, ASCII) of the job without `approval`."""
    body = {key: value for key, value in job.items() if key != "approval"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)
    return sha256_bytes(text.encode())


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise JobError(f"{name} must be a finite number")
    return float(value)


def _absolute(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value.startswith("/") or value == "/"
            or posixpath.normpath(value) != value):
        raise JobError(f"{name} must be a canonical absolute path")
    return value


def _financial(value: Any, now: dt.datetime, mode: str = "legacy") -> dt.datetime:
    """Validate the bounds the watchdog will use; return the effective deadline."""
    keys = set(FINANCIAL_KEYS) | ({"balance_snapshot_utc"} if mode in ("balance_reserve_only", "measured_balance_reserve") else set())
    if not isinstance(value, dict) or set(value) != keys:
        raise JobError(f"financial must have exactly the keys {sorted(keys)}")
    if not isinstance(value["owned_pod_id"], str) or not POD_ID.match(value["owned_pod_id"]):
        raise JobError("financial.owned_pod_id must be the RunPod Pod id")
    numbers = [_number(value[key], f"financial.{key}")
               for key in ("balance_usd", "reserve_usd", "hourly_usd")]
    try:
        cutoff = parse_utc(value["gpu_cutoff_utc"])
        start = parse_utc(value["billing_start_utc"])
        check_billing_start(start, now)
        if mode in ("balance_reserve_only", "measured_balance_reserve"):
            return reserve_deadline(parse_utc(value["balance_snapshot_utc"]),
                                    numbers[0], numbers[1], numbers[2], now=now)
        return effective_deadline(cutoff, start, *numbers)
    except (TypeError, ValueError, AttributeError) as exc:
        raise JobError(f"financial: {exc}") from exc


def _data(value: Any, workspace: str) -> None:
    if not isinstance(value, dict) or set(value) != DATA_KEYS:
        raise JobError(f"data must have exactly the keys {sorted(DATA_KEYS)}")
    _workspace_child(value["bundle_manifest"], workspace, "data.bundle_manifest")
    _absolute(value["dataset_manifest"], "data.dataset_manifest")
    _absolute(value["vault_root"], "data.vault_root")
    for key in ("bundle_manifest_sha256", "dataset_manifest_sha256"):
        if not isinstance(value[key], str) or not HEX64.match(value[key]):
            raise JobError(f"data.{key} must be a lowercase sha256")


def _approval(job: dict[str, Any], now: dt.datetime) -> None:
    approval = job["approval"]
    if not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS:
        raise JobError("approval missing or malformed: only the user may record it "
                       f"(keys {sorted(APPROVAL_KEYS)})")
    if approval["approved_by_user"] is not True:
        raise JobError("approval.approved_by_user must be true (set by the user, not generated)")
    note = approval["approval_note"]
    if not isinstance(note, str) or not note.strip():
        raise JobError("approval.approval_note must be a non-empty string")
    age = job.get("max_approval_age_hours", DEFAULT_APPROVAL_AGE_HOURS)
    if (isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age)
            or not 0 < age <= MAX_APPROVAL_AGE_HOURS):
        raise JobError(f"max_approval_age_hours must be a number in (0, {MAX_APPROVAL_AGE_HOURS}]")
    try:
        content = content_sha256(job)
    except ValueError as exc:  # NaN/inf anywhere in the job
        raise JobError(f"job content is not canonical JSON: {exc}") from exc
    if approval["approved_sha256"] != content:
        raise JobError("approval.approved_sha256 does not match this job's content (sha256 of the "
                       "job without `approval`); a changed job needs a new approval")
    try:
        approved = parse_utc(approval["approved_utc"])
    except (TypeError, ValueError, AttributeError) as exc:
        raise JobError(f"approval.approved_utc must be a timezone-aware ISO time: {exc}") from exc
    if approved > now + dt.timedelta(seconds=SKEW_SECONDS):
        raise JobError("approval.approved_utc is in the future")
    if now - approved > dt.timedelta(hours=age):
        raise JobError(f"approval is older than max_approval_age_hours ({age}); ask again")


def validate_job(job: Any, workspace: str = "/workspace",
                 now: dt.datetime | None = None) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise JobError("job must be a JSON object")
    prep = job.get("version") == PREP_VERSION
    required = (PREP_KEYS if prep else KEYS - {"data"}
                if job.get("stage") in EVAL_STAGES + SOURCE_REBUILT_STAGES else KEYS)
    optional = OPTIONAL_KEYS | (PREP_OPTIONAL_KEYS if prep else set())
    if not required <= set(job) or set(job) - required - optional:
        raise JobError(f"job keys must be {sorted(required)} plus optional "
                       f"{sorted(optional)}: missing {sorted(required - set(job))}, "
                       f"unknown {sorted(set(job) - required - optional)}")
    if job["version"] not in (VERSION, PREP_VERSION):
        raise JobError(f"version must be {VERSION} or {PREP_VERSION}")
    if not isinstance(job["commit"], str) or not HEX40.match(job["commit"]):
        raise JobError("commit must be a 40-char lowercase hex SHA")
    _workspace_child(job["volume_root"], workspace, "volume_root")
    now = now or dt.datetime.now(dt.timezone.utc)
    mode = job.get("financial_mode", "legacy")
    if mode not in ("legacy", "balance_reserve_only", "measured_balance_reserve"):
        raise JobError("unknown financial_mode")
    if mode in ("balance_reserve_only", "measured_balance_reserve"):
        measured_prep = prep and mode == "measured_balance_reserve"
        if not measured_prep and (prep or job.get("stage") not in ("single-main", "v18-main", "eval-course")):
            raise JobError("reserve-only mode is for split training/evaluation jobs")
        if "repo_root" not in job:
            raise JobError("split jobs require repo_root")
    if "repo_root" in job:
        _absolute(job["repo_root"], "repo_root")
        if mode in ("balance_reserve_only", "measured_balance_reserve") and job.get("stage") == "eval-course" and job["repo_root"] != "/root/MMDL_test_eval":
            raise JobError("split evaluation repo_root must be /root/MMDL_test_eval")
    _financial(job["financial"], now, mode)
    if job["version"] == PREP_VERSION:
        if job["environment"] not in ENVIRONMENTS:
            raise JobError(f"environment must be one of {ENVIRONMENTS}")
        if type(job.get("eval_data", False)) is not bool:
            raise JobError("eval_data must be true or false")
    else:
        _workspace_child(job["run_dir"], workspace, "run_dir")
        if job["stage"] not in STAGES:
            raise JobError(f"stage must be one of {STAGES}")
        _argv(job["smoke"], "smoke")
        _argv(job["main"], "main")
        seconds = job["max_main_seconds"]
        if (isinstance(seconds, bool) or not isinstance(seconds, int)
                or not (0 < seconds <= MAX_MAIN_SECONDS or seconds == 0 and mode == "measured_balance_reserve")):
            raise JobError(f"max_main_seconds must be an int in 1..{MAX_MAIN_SECONDS}")
        if _number(job["min_free_gib"], "min_free_gib") <= 0:
            raise JobError("min_free_gib must be positive")
        if job["stage"] in TRAINING_STAGES and job["stage"] not in SOURCE_REBUILT_STAGES:
            _data(job["data"], workspace)
    _approval(job, now)
    return job


def load_job(path: str | Path, workspace: str = "/workspace",
             now: dt.datetime | None = None) -> dict[str, Any]:
    try:
        job = read_json(path)
    except (OSError, ValueError) as exc:
        raise JobError(f"cannot read job file: {exc}") from exc
    return validate_job(job, workspace, now)


def check_interpreter(job: dict[str, Any], manifest_path: str | Path) -> str:
    """smoke[0] and main[0] must be the volume manifest's venv python, exactly."""
    try:
        venv = read_json(manifest_path).get("venv")
    except (OSError, ValueError, AttributeError) as exc:
        raise JobError(f"cannot read volume manifest: {exc}") from exc
    if not isinstance(venv, str) or not venv:
        raise JobError("volume manifest has no venv path")
    expected = venv + "/bin/python"
    for name in ("smoke", "main"):
        if job[name][0] != expected:
            raise JobError(f"{name}[0] must be the volume venv interpreter {expected}, "
                           f"got {job[name][0]!r}")
    return expected


def required_seconds(job: dict[str, Any]) -> int:
    if job["version"] == PREP_VERSION:
        return PREP_MAX_SECONDS
    return job["max_main_seconds"] + SMOKE_TIMEOUT_SECONDS + PREFLIGHT_ALLOWANCE_SECONDS


def budget_fit(job: dict[str, Any], now: dt.datetime | None = None) -> dict[str, Any]:
    """Does the job's whole bound fit before the watchdog deadline (budget.py's rule)?"""
    now = now or dt.datetime.now(dt.timezone.utc)
    mode = job.get("financial_mode", "legacy")
    deadline = _financial(job["financial"], now, mode)
    remaining = (deadline - now).total_seconds() - (0 if mode != "legacy" else RECOVERY_RESERVE_SECONDS)
    required = required_seconds(job)
    return {"deadline_utc": deadline.isoformat(), "usable_seconds": math.floor(remaining),
            "required_seconds": required, "fits": required <= remaining}


def financial_argv(job: dict[str, Any]) -> list[str]:
    """The watchdog flags, derived only from the approved job."""
    result = [item for key in FINANCIAL_KEYS
              for item in ("--" + key.replace("_", "-"), str(job["financial"][key]))]
    if job.get("financial_mode") == "balance_reserve_only":
        result += ["--financial-mode", "balance_reserve_only", "--balance-snapshot-utc",
                   job["financial"]["balance_snapshot_utc"],
                   "--extension-receipts", str(Path(job["run_dir"]) / "financial_extensions"),
                   "--extension-root-sha256", content_sha256(job),
                   "--reserve-stop-receipt", job["run_dir"] + ".reserve-stop.json"]
    elif job.get("financial_mode") == "measured_balance_reserve":
        result += ["--financial-mode", "measured_balance_reserve", "--balance-snapshot-utc",
                   job["financial"]["balance_snapshot_utc"],
                   "--balance-observation", str(Path(job["volume_root"]).parent / "jobs" / "balance-latest.json"),
                   "--reserve-stop-receipt", job.get("run_dir", job["volume_root"] + ".prep") + ".reserve-stop.json"]
    return result


def check_repo_root(job: dict[str, Any], executing_root: str | Path) -> None:
    """A split job must execute the start/verify code of its declared pinned clone."""
    if "repo_root" not in job:
        return
    root = Path(executing_root).resolve()
    if Path(job["repo_root"]) != root:
        raise JobError("job.repo_root differs from the executing start.sh/verify clone")
    for arguments, expected in ((("rev-parse", "--show-toplevel"), str(root)),
                                (("rev-parse", "HEAD"), job["commit"])):
        try:
            actual = subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
        except subprocess.CalledProcessError as exc:
            raise JobError("cannot identify the executing clone") from exc
        if actual != expected:
            raise JobError("executing clone root/commit differs from the job")


def extend_financial(job: dict[str, Any], *, balance_snapshot_utc: str, balance_usd: float,
                     topup_reference: str, authorization: str, confirmed_by: str,
                     now: dt.datetime | None = None) -> Path:
    """Append one explicit top-up receipt without altering the approved running job."""
    if job.get("financial_mode") != "balance_reserve_only":
        raise JobError("only reserve-only split jobs support balance extensions")
    now = now or dt.datetime.now(dt.timezone.utc)
    bounds = job["financial"]
    directory = Path(job["run_dir"]) / "financial_extensions"
    if not directory.parent.is_dir():
        raise JobError("balance extensions require the existing owned run directory")
    snapshot = parse_utc(balance_snapshot_utc)
    if now - snapshot > dt.timedelta(minutes=5):
        raise JobError("top-up balance snapshot must be measured within five minutes")
    root = content_sha256(job)
    old_deadline, old_snapshot, hashes = balance_extensions(
        directory, root_sha256=root, owned_pod_id=bounds["owned_pod_id"],
        snapshot=parse_utc(bounds["balance_snapshot_utc"]), balance_usd=bounds["balance_usd"],
        hourly_usd=bounds["hourly_usd"], now=now)
    deadline = reserve_deadline(snapshot, balance_usd, 1.5, bounds["hourly_usd"], now)
    value = {"version": "pod-balance-extension-v1", "previous": hashes[-1] if hashes else root,
             "owned_pod_id": bounds["owned_pod_id"], "balance_snapshot_utc": balance_snapshot_utc,
             "balance_usd": balance_usd, "reserve_usd": 1.5, "hourly_usd": bounds["hourly_usd"],
             "old_deadline_utc": old_deadline.isoformat(), "new_deadline_utc": deadline.isoformat(),
             "topup_reference": topup_reference, "authorization": authorization,
             "confirmed_by": confirmed_by}
    value["sha256"] = balance_extension_sha256(value)
    check_balance_extension(value, previous=value["previous"], deadline=old_deadline,
                            snapshot=old_snapshot, owned_pod_id=bounds["owned_pod_id"],
                            hourly_usd=bounds["hourly_usd"], now=now)
    directory.mkdir(exist_ok=True)
    path = directory / f"{len(hashes) + 1:06d}.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".balance-extension-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--workspace", default="/workspace")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--get", choices=GET_FIELDS)
    choice.add_argument("--argv", choices=("smoke", "main"), help="print NUL-separated argv")
    choice.add_argument("--require-interpreter", metavar="VOLUME_MANIFEST",
                        help="refuse unless smoke[0] and main[0] are that manifest's venv python")
    choice.add_argument("--financial-argv", action="store_true",
                        help="print the NUL-separated watchdog flags from job.financial")
    choice.add_argument("--require-repo-root", metavar="EXECUTING_ROOT")
    choice.add_argument("--extend-financial", action="store_true")
    parser.add_argument("--balance-snapshot-utc")
    parser.add_argument("--balance-usd", type=float)
    parser.add_argument("--topup-reference")
    parser.add_argument("--authorization")
    parser.add_argument("--confirmed-by", choices=("user", "claude"))
    choice.add_argument("--content-sha256", action="store_true",
                        help="print the value the user writes as approval.approved_sha256")
    args = parser.parse_args(argv)
    try:
        if args.content_sha256:
            try:
                value = read_json(args.job)
            except (OSError, ValueError) as exc:
                raise JobError(f"cannot read job file: {exc}") from exc
            if not isinstance(value, dict):
                raise JobError("job must be a JSON object")
            print(content_sha256(value))
            return 0
        job = load_job(args.job, args.workspace)
        if args.require_repo_root:
            check_repo_root(job, args.require_repo_root)
            return 0
        if args.extend_financial:
            if os.environ.get("RUNPOD_POD_ID") != job["financial"]["owned_pod_id"]:
                raise JobError("balance extensions require the exact owned Pod identity")
            if any(getattr(args, name) is None for name in
                   ("balance_snapshot_utc", "balance_usd", "topup_reference", "authorization", "confirmed_by")):
                raise JobError("extension requires a measured balance, top-up reference and explicit authorization")
            path = extend_financial(job, balance_snapshot_utc=args.balance_snapshot_utc,
                                    balance_usd=args.balance_usd, topup_reference=args.topup_reference,
                                    authorization=args.authorization, confirmed_by=args.confirmed_by)
            print(json.dumps({"extension_receipt": str(path), "sha256": read_json(path)["sha256"]}))
            return 0
        if args.get:
            if args.get == "financial_mode":
                print(job.get("financial_mode", "legacy"))
                return 0
            source = job["financial"] if args.get == "owned_pod_id" else job
            if args.get == "eval_data" and job["version"] == PREP_VERSION:
                print("true" if job.get("eval_data", False) else "false")
                return 0
            if args.get not in source:
                raise JobError(f"{job['version']} has no field {args.get}")
            print(source[args.get])
            return 0
        if args.argv:
            if job["version"] != VERSION:
                raise JobError(f"--argv needs a {VERSION} job")
            sys.stdout.write("".join(item + "\0" for item in job[args.argv]))
            return 0
        if args.financial_argv:
            sys.stdout.write("".join(item + "\0" for item in financial_argv(job)))
            return 0
        if args.require_interpreter:
            check_interpreter(job, args.require_interpreter)
            return 0
        plan: dict[str, Any] = {"job_sha256": sha256_file(args.job), **job,
                                "budget": budget_fit(job)}
        if not plan["budget"]["fits"] and job.get("financial_mode") not in ("balance_reserve_only", "measured_balance_reserve"):
            print(json.dumps(plan, indent=2), file=sys.stderr)
            raise JobError("job does not fit before the financial/GPU deadline")
        print(json.dumps(plan, indent=2))
        return 0
    except (JobError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
