"""Recovery and cleanup bookkeeping. Never calls RunPod or any API; the user performs and
confirms every remote action, this module only records evidence.

pack      (on the Pod)  tar of CLOSED run dirs only + sha256 manifest; --unclosed-dead-owner also
                        takes a hard-killed run (dead owner, no process env names the run dir),
                        labelled UNCLOSED in the manifest. Successful chains need their approved spec,
                        complete verified history and expected files; an external spec is included.
pack-eval (on the Pod)  one MMMU evaluation stage (eval-f0|eval-ft; eval-course is not planned): its run dir
                        plus exactly what `proof_eval analyze` reads (the stage's planned run dirs under
                        artifact_root, the ledger, the prereg and, for FT, model_ref.json; never model
                        weights), same manifest/verify/ledger flow, manifest "kind": "eval"
verify    (local)       re-hash the downloaded tar, validate members, safe extraction, receipt; an eval
                        pack and its extraction directory must lie under the private artifact root
                        ($MMDL_ARTIFACT_ROOT, default ~/mmdl-artifacts), never in the repo
state                   append-only, hash-chained ledger with the fixed cleanup order

CLI: collect pack --out-dir DIR --name NAME [--unclosed-dead-owner] RUN_DIR [RUN_DIR ...]
     collect pack-eval --params EVAL_STAGE_PARAMS.json --stage STAGE --run-dir RUN_DIR --out-dir DIR [--name NAME]
     collect verify --archive TAR --manifest JSON --extract-to NEW_DIR --receipt OUT.json
     collect state --ledger FILE show
     collect state --ledger FILE append STATE [--evidence key=value ...] [--receipt-file FILE]
                   [--run-ledger FILE ...]
OWNED_STORAGE_DELETED needs --run-ledger for every run that used the storage (this ledger included);
each must already hold REMOTE_VERIFIED and LOCAL_VERIFIED (one cleanup ledger per pack, training or eval).
LOCAL_VERIFIED needs the typed pod_id of REMOTE_VERIFIED to equal the pod_id the pack recorded on the Pod
(RUNPOD_POD_ID), via the receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import (HEX64, HashingReader, parse_utc, process_startticks, read_json, sha256_bytes,
                     sha256_file, utc_now, write_json)
from .runner import TERMINAL_PREFIXES

PACK_VERSION = "pod-pack-v1"
EVAL_KIND = "eval"
# Exactly what proof_eval.load_run reads from a run directory (no samples/, prompts/, logs, source.patch).
EVAL_RUN_FILES = ("run_manifest.json", "summary.json", "predictions.jsonl", "resolved_eval_config.yaml",
                  "variant_registration.json")
EVAL_REQUIRED = EVAL_RUN_FILES[:4]  # variant_registration.json: required for the statistical (8k) runs
LEDGER_VERSION = "pod-cleanup-ledger-v1"
ORDER = ("REMOTE_VERIFIED", "LOCAL_VERIFIED", "POD_TERMINATED", "OWNED_STORAGE_DELETED",
         "CLEANUP_VERIFIED")
RESULT_FILES = ("result.json", "receipt.json", "throughput_receipt.json")
SKEW = dt.timedelta(seconds=60)
PROC = Path("/proc")


class CollectError(ValueError):
    pass


# ---------------------------------------------------------------- pack

def owner_alive(run_dir: Path) -> bool:
    """True when the recorded owner PID is alive with the recorded start time.

    Fails closed: a PID that exists but whose /proc start time cannot be read counts as alive.
    """
    try:
        owner = read_json(run_dir / "owner.json")
        pid, ticks = int(owner["pid"]), int(owner["startticks"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CollectError(f"{run_dir.name}: owner.json missing or unreadable ({exc})") from exc
    current = process_startticks(pid)
    if current is not None:
        return current == ticks
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_dir_holders(run_dir: Path) -> list[str]:
    """PIDs whose environment still carries MMDL_RUN_DIR=<run_dir> (start.sh exports it).

    A SIGKILLed start.sh leaves its session-leader children running; they keep the variable.
    Fails closed: an environ that cannot be read is reported as a holder, and so is a missing
    /proc. Only a process that vanished mid-scan is ignored.
    """
    if not PROC.is_dir():
        raise CollectError(f"{PROC} unavailable: cannot show that nothing still uses the run dir")
    wanted = {os.fsencode(f"MMDL_RUN_DIR={path}") for path in (os.path.abspath(run_dir),
                                                              run_dir.resolve())}
    holders = []
    for entry in PROC.glob("[0-9]*"):
        try:
            environment = (entry / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            holders.append(f"{entry.name} (environ unreadable)")
            continue
        if wanted & set(environment.split(b"\0")):
            holders.append(entry.name)
    return holders


def closed_run(run_dir: Path, allow_unclosed: bool = False,
               results: tuple[str, ...] = RESULT_FILES) -> str:
    """Raise unless the run is closed: owner gone, terminal state, and a result when it succeeded.

    A failed run (PREFLIGHT_FAILED, SMOKE_FAILED, INTERRUPTED, MAIN_EXIT != 0) is packable
    without a result file: its run_state.json and logs are the evidence that must be recovered.
    With allow_unclosed a non-terminal state (hard kill: no INTERRUPTED was ever written) is
    also accepted, but only when no process env still names the run dir; returns "UNCLOSED"
    for those and "CLOSED" otherwise. A MAIN_EXIT 0 run must hold one of `results` (an evaluation
    stage keeps its results outside its run dir and passes chain.jsonl).
    """
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise CollectError(f"{run_dir}: not a real directory")
    if owner_alive(run_dir):
        raise CollectError(f"{run_dir.name}: owner process is still alive")
    try:
        state = str(read_json(run_dir / "run_state.json")["state"])
    except (OSError, ValueError, KeyError) as exc:
        raise CollectError(f"{run_dir.name}: run_state.json missing or unreadable") from exc
    if not state.startswith(TERMINAL_PREFIXES):
        if not allow_unclosed:
            raise CollectError(f"{run_dir.name}: run is not closed (state {state})")
        holders = run_dir_holders(run_dir)
        if holders:
            raise CollectError(f"{run_dir.name}: processes still name this run dir: {holders}")
        return "UNCLOSED"
    if state == "MAIN_EXIT 0" and not any((run_dir / name).is_file() for name in results):
        raise CollectError(f"{run_dir.name}: successful run lacks result/receipt {results}")
    return "CLOSED"


def _check_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise CollectError("archive name must match [A-Za-z0-9._-]+")


def _completed_chain_spec(run_dir: Path) -> Path:
    """A successful chain is its own completion evidence; never manufacture a training result."""
    from . import chain, job

    try:
        recorded = read_json(run_dir / "job.json")
        argv = recorded["main"]
        if (recorded.get("version") != job.VERSION or recorded.get("run_dir") != str(run_dir.absolute())
                or recorded.get("approval", {}).get("approved_by_user") is not True
                or recorded["approval"].get("approved_sha256") != job.content_sha256(recorded)
                or not isinstance(argv, list) or len(argv) != 8
                or argv[1:5] != ["-m", "fine_tuning.notebook_rl.pod.chain", "run", "--spec"]
                or argv[6] != "--spec-sha256" or not HEX64.fullmatch(argv[7])):
            raise ValueError("recorded job does not bind the chain spec")
        source = Path(argv[5])
        raw = source.read_bytes()
        if not source.is_absolute() or sha256_bytes(raw) != argv[7]:
            raise ValueError("chain spec SHA256 differs from the recorded job")
        spec = chain.check_spec(json.loads(raw))
        if spec.get("meta", {}).get("job_stage") != recorded["stage"]:
            raise ValueError("chain spec stage differs from the recorded job")
        if not any(step["expect_files"] for step in spec["steps"]):
            raise ValueError("chain spec declares no stage outputs")
        log = chain.verify_log(run_dir / chain.LOG_NAME, argv[7])
        if (len(log) != 2 + 2 * len(spec["steps"]) or log[0].get("event") != "CHAIN_START"
                or log[0].get("spec_sha256") != argv[7]
                or log[0].get("steps") != [step["name"] for step in spec["steps"]]
                or (log[-1].get("event"), log[-1].get("status"), log[-1].get("exit_code")) != ("CHAIN_END", "COMPLETE", 0)):
            raise ValueError("successful run lacks its complete chain history")
        for index, step in enumerate(spec["steps"]):
            start, end = log[1 + 2 * index:3 + 2 * index]
            if (start.get("event") != "STEP_START" or start.get("name") != step["name"] or start.get("argv") != step["argv"]
                    or end.get("event") != "STEP_END" or end.get("name") != step["name"]
                    or end.get("status") != "OK" or end.get("exit_code") != 0
                    or end.get("expect_files") != {path: True for path in step["expect_files"]}):
                raise ValueError(f"chain step {step['name']} did not complete as planned")
            for expected in step["expect_files"]:
                _plain_file(Path(expected), Path(expected).parent)
        return _plain_file(source, source.parent)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise CollectError(f"{run_dir.name}: successful chain cannot be recovered ({exc})") from exc


def pack(run_dirs: list[Path], out_dir: Path, name: str,
         unclosed_dead_owner: bool = False) -> dict[str, Any]:
    _check_name(name)
    names = [path.name for path in run_dirs]
    if not run_dirs or len(set(names)) != len(names):
        raise CollectError("one or more run dirs with distinct names required")
    unclosed, specs = [], []
    for path in run_dirs:
        closure = closed_run(path, unclosed_dead_owner, results=(*RESULT_FILES, "chain.jsonl"))
        if closure == "UNCLOSED":
            unclosed.append(path.name)
        if read_json(path / "run_state.json")["state"] == "MAIN_EXIT 0":
            recorded = read_json(path / "job.json") if (path / "job.json").is_file() else {}
            if not any((path / item).is_file() for item in RESULT_FILES) or recorded.get("stage") in ("note-v3-collect", "note-v3-resolve", "note-v3-train"):
                source = _completed_chain_spec(path)
                if not source.is_relative_to(path.absolute()):
                    if (path / "_recovery").exists():
                        raise CollectError(f"{path}: reserved _recovery archive path already exists")
                    specs.append((path, source, sha256_file(source)))

    def fill(tar: tarfile.TarFile, records: list[dict[str, Any]]) -> None:
        for run_dir in run_dirs:
            _tar_run_dir(tar, run_dir, records)
        for run_dir, source, expected_sha in specs:
            if sha256_file(source) != expected_sha:
                raise CollectError(f"{source}: chain spec changed while packing")
            _add_member(tar, source, f"{run_dir.name}/_recovery/chain-spec.json", records)
            if records[-1]["sha256"] != expected_sha:
                raise CollectError(f"{source}: chain spec changed while packing")

    return _seal(out_dir, name, fill, runs=names, closure="UNCLOSED" if unclosed else "CLOSED",
                 unclosed_runs=unclosed)


def _seal(out_dir: Path, name: str, fill: Callable[[tarfile.TarFile, list[dict[str, Any]]], None],
          compression: bool = False, **fields: Any) -> dict[str, Any]:
    """Write <name>.tar (atomic rename, never overwrites) and <name>.manifest.json; fill adds members."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".tar.gz" if compression else ".tar"
    archive, manifest_path = out_dir / f"{name}{suffix}", out_dir / f"{name}.manifest.json"
    if archive.exists() or manifest_path.exists():
        raise CollectError("archive or manifest already exists; refusing to overwrite")
    records: list[dict[str, Any]] = []
    partial = out_dir / f".{name}{suffix}.partial"
    try:
        with tarfile.open(partial, "x:gz" if compression else "x") as tar:
            fill(tar, records)
        os.rename(partial, archive)
    finally:
        partial.unlink(missing_ok=True)  # only the temp file this call created
    manifest = {"version": PACK_VERSION, "archive": archive.name,
                "archive_bytes": archive.stat().st_size, "archive_sha256": sha256_file(archive),
                **fields, "files": records, "created_utc": utc_now(),
                "pod_id": os.environ.get("RUNPOD_POD_ID")}
    write_json(manifest_path, manifest)
    return manifest


def _add_member(tar: tarfile.TarFile, path: Path, arcname: str, records: list[dict[str, Any]]) -> None:
    if path.is_symlink() or not (path.is_dir() or path.is_file()):
        raise CollectError(f"{path}: only regular files and directories are packed")
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if path.is_dir():
        tar.addfile(info)
        return
    with open(path, "rb") as stream:
        reader = HashingReader(stream)
        tar.addfile(info, reader)
    if reader.count != info.size:
        raise CollectError(f"{path}: changed while packing")
    records.append({"path": info.name, "bytes": info.size, "sha256": reader.digest.hexdigest(),
                    "source": os.path.abspath(path)})


def _tar_run_dir(tar: tarfile.TarFile, run_dir: Path, records: list[dict[str, Any]]) -> None:
    for path in sorted([run_dir, *run_dir.rglob("*")]):
        _add_member(tar, path, str(Path(run_dir.name) / path.relative_to(run_dir)), records)


# ---------------------------------------------------------------- pack-eval

def _plain_file(path: Path, root: Path) -> Path:
    """A regular file whose real path is exactly `path` below the real `root`: no symlink anywhere."""
    try:
        wanted = root.resolve() / path.relative_to(root)
    except ValueError as exc:
        raise CollectError(f"{path}: outside {root}") from exc
    if path.is_symlink() or not path.is_file() or path.resolve() != wanted:
        raise CollectError(f"{path}: not a regular file inside {root} (symlinks are refused)")
    return path


def _split_resume_inputs(prereg: dict, registered: dict, model_ref: Path, calibration: Path,
                         preflight: Path, training_seal: Path, *, verify_training: bool) -> dict:
    from fine_tuning.error_note import v17_eval
    from fine_tuning.error_note.v17_artifacts import VERSION, verify_training_artifacts

    try:
        _plain_file(training_seal, training_seal.parent)
        sealed = (verify_training_artifacts(training_seal) if verify_training
                  else v17_eval.checked(read_json(training_seal)))
        validation = read_json(calibration)
        ids = prereg["evaluation_subsets"][registered.get("subset", "all")]["ids"]
        all_ids = sorted(prereg["b2_reference"]["wrong_ids"] + prereg["b2_reference"]["right_ids"]) if "execution_settings" in prereg else ids
        contract = v17_eval.split_execution_contract(validation, all_ids)
        if validation.get("training_protocol_version") == "error-note-v18b":
            from fine_tuning.error_note.v18b_eval import partitions
            settings = prereg["execution_settings"]
            pieces = partitions(all_ids, settings)
            index = registered.get("partition_index")
            if (registered.get("execution_settings") != settings or validation.get("execution_settings") != settings
                    or type(index) is not int or not 0 <= index < len(pieces)
                    or registered.get("subset") != list(pieces)[index]
                    or ids != pieces[registered["subset"]]):
                raise ValueError("v18b first-billing settings or fixed partition changed")
        seal_sha, engine_sha, model_sha = (sha256_file(training_seal), sha256_file(calibration), sha256_file(model_ref))
        if (sealed.get("version") != VERSION or sealed.get("status") not in ("SEALED", "PARTIAL_EVALUATION_CANDIDATE")
                or validation != prereg["engine_validation"] or sha256_file(preflight) != engine_sha
                or any(owner.get("engine_validation_sha256") != engine_sha for owner in (prereg, registered))
                or any(owner.get("training_artifacts_file_sha256") != seal_sha for owner in (prereg, registered))
                or any(owner.get("execution_contract") != contract for owner in (prereg, registered))
                or validation["training_artifact_sha256"] != seal_sha
                or validation["training_artifact_file_sha256"] != seal_sha
                or validation.get("training_artifact_content_sha256") != sealed["sha256"]
                or validation["model_ref_sha256"] != model_sha
                or sealed["export"]["model_ref"]["file_sha256"] != model_sha
                or registered.get("calibration_sha256") != engine_sha or registered.get("preflight_sha256") != engine_sha
                or prereg.get("calibration_sha256") != engine_sha or registered.get("model_ref_sha256") != model_sha
                or registered.get("slots") != contract["slots"] or registered.get("expected_ids") != ids
                or registered.get("observed_engine") != validation["observed_engine"]
                or registered.get("evaluation_commit") != validation["evaluation_commit"]):
            raise ValueError("selected execution/model/training-seal binding mismatch")
        return contract
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CollectError("v17 split resume inputs invalid: " + str(exc)) from exc


def _split_resume_audit(run_dir: Path, registration: dict, *, refresh: bool) -> dict:
    from fine_tuning.error_note import v17_eval

    try:
        audit = v17_eval.submission_audit(run_dir, registration)
        event_path = run_dir / "submission_events.jsonl"
        events = [v17_eval.checked(json.loads(line)) for line in event_path.read_text().splitlines()] if event_path.exists() else []
        path = run_dir / "submission_audit.json"
        if path.exists():
            _plain_file(path, run_dir)
            stored = v17_eval.checked(read_json(path))
            count = stored.get("events")
            if type(count) is not int or not 0 <= count <= len(events):
                raise ValueError("invalid derived submission audit length")
            prefix = events[:count]
            accepted = [e["id"] for e in prefix if e["event"] == "ACCEPTED"]
            intents = [e["id"] for e in prefix if e["event"] == "INTENT"]
            expected = v17_eval.seal({**{k: v for k, v in audit.items() if k != "sha256"},
                "events": count, "last_event_sha256": prefix[-1]["sha256"] if prefix else None,
                "actual_submission_sha256": v17_eval.digest(accepted), "actual_submitted_ids": accepted,
                "submission_intent_sha256": v17_eval.digest(intents), "submission_intent_ids": intents,
                "interrupted_intent_id": prefix[-1]["id"] if prefix and prefix[-1]["event"] == "INTENT" else None})
            if stored != expected or not refresh and stored != audit:
                raise ValueError("derived submission audit differs from durable events")
        elif not refresh:
            raise ValueError("missing derived submission audit")
        accepted_pairs = {(e["invocation_id"], e["id"]) for e in events if e["event"] == "ACCEPTED"}
        for path in (run_dir / "final_raw").glob("*.json"):
            final = v17_eval.checked(read_json(path))
            if (final["invocation_id"], final["record"]["id"]) not in accepted_pairs:
                raise ValueError("FINAL_RAW lacks its accepted submission event")
        backend_path = run_dir / "backend.json"
        if backend_path.exists():
            actual = read_json(backend_path).get("v17_actual_engine", {})
            expected = registration["observed_engine"]
            def execution(value):
                return {key: item for key, item in value.items() if key not in v17_eval.ENGINE_CAPACITY_KEYS}
            if actual != expected and (execution(actual) != execution(expected) or actual.get("safe_slots", 0) < registration["slots"]):
                raise ValueError("actual backend differs from selected engine")
        if refresh:
            write_json(run_dir / "submission_audit.json", audit)
        return audit
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CollectError("v17 split submission audit invalid: " + str(exc)) from exc


def _split_resume_evidence(validation_path: Path, run_dir: Path, *, archive_root: Path | None = None) -> list[tuple[Path, str]]:
    from fine_tuning.error_note import v17_eval

    try:
        validation = read_json(validation_path)
        expected: dict[str, str] = {}
        refs = [*validation["attempts"],
                {"path": validation["preflight_report"], "sha256": validation["preflight_file_sha256"]}]
        if validation.get("training_protocol_version") in ("error-note-v18", "error-note-v18b"):
            ref = validation["artifact_verification"]
            refs.append({"path": ref["path"], "sha256": ref["file_sha256"]})
        for ref in refs:
            original = ref["path"]
            if not Path(original).is_absolute() or ".." in Path(original).parts:
                raise ValueError("engine evidence requires its original absolute path")
            if original in expected and expected[original] != ref["sha256"]:
                raise ValueError("engine evidence path has conflicting hashes")
            expected[original] = ref["sha256"]
        records, sources = [], []
        for index, (original, expected_sha) in enumerate(sorted(expected.items())):
            target = f"eval/inputs/engine-evidence/{index:04}.json"
            path = Path(original) if archive_root is None else archive_root / target
            _plain_file(path, path.parent if archive_root is None else archive_root)
            if sha256_file(path) != expected_sha:
                raise ValueError("measured engine evidence bytes changed")
            records.append({"original_path": original, "path": target, "sha256": expected_sha,
                            "bytes": path.stat().st_size})
            sources.append((path, target))
        mapping = v17_eval.seal({"version": "v17-engine-evidence-v1",
            "engine_validation_sha256": sha256_file(validation_path), "files": records,
            "restore_policy": "restore exact bytes to original paths inside the owned volume before same-START resume"})
        mapping_path = run_dir / "engine_evidence_mapping.json"
        if mapping_path.exists() or mapping_path.is_symlink():
            _plain_file(mapping_path, run_dir)
        if archive_root is None:
            if mapping_path.exists() and v17_eval.checked(read_json(mapping_path)) != mapping:
                raise ValueError("existing engine evidence mapping differs")
            write_json(mapping_path, mapping)
        elif v17_eval.checked(read_json(mapping_path)) != mapping:
            raise ValueError("archived engine evidence mapping differs")
        return sources
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CollectError("v17 split engine evidence invalid: " + str(exc)) from exc


def pack_eval(params: Any, stage: str, run_dir: Path, out_dir: Path,
              name: str | None = None) -> dict[str, Any]:
    """Pack one evaluation stage: its start.sh run dir and the inputs of `proof_eval analyze`.

    Packed (archive paths): <run dir name>/** (logs, chain.jsonl), eval/runs/<run id>/<EVAL_RUN_FILES>
    for each run the params plan (their one arm, requested seeds and subset_tag), eval/<ledger>, eval/<prereg> + .sha256 and, for eval-ft/eval-course,
    eval/model_ref.json. The merged model weights are never packed. Every planned run needs its ledger
    START (arm, variant and subset_tag as planned) and must be finished (summary PARTIAL for the subset runs, COMPLETE for a full run), or be recorded
    FAILED in the ledger (packed as it is, marked FAILED).
    """
    from .. import proof_eval  # lazy: numpy/yaml are not needed by pack, verify or state
    from . import chain, eval_stage

    p = eval_stage._check(stage, params)
    name = name or run_dir.name
    _check_name(name)
    if run_dir.name == "eval":
        raise CollectError("the start.sh run dir may not be named 'eval' (archive prefix)")
    if not os.environ.get("RUNPOD_POD_ID"):
        raise CollectError("RUNPOD_POD_ID is not set: the pack records the Pod it ran on")
    # The run dir must have run exactly the spec these params plan (the bytes `eval_stage plan` writes).
    spec_sha = sha256_bytes((json.dumps(eval_stage.plan(stage, p), indent=2, sort_keys=True) + "\n").encode())
    closed_run(run_dir, results=(chain.LOG_NAME,))
    if not (run_dir / chain.LOG_NAME).is_file():
        raise CollectError(f"{run_dir.name}: no {chain.LOG_NAME}, the chain never started; "
                           "recover its logs with `collect pack`")
    try:
        log = chain.verify_log(run_dir / chain.LOG_NAME, spec_sha)
    except (OSError, ValueError) as exc:
        raise CollectError(f"{run_dir.name}: {chain.LOG_NAME} is not the log of the spec these params plan "
                           f"({exc})") from exc
    if read_json(run_dir / "run_state.json")["state"] == "MAIN_EXIT 0" and (
            not log or (log[-1]["event"], log[-1].get("status")) != ("CHAIN_END", "COMPLETE")):
        raise CollectError(f"{run_dir.name}: MAIN_EXIT 0 without a COMPLETE chain end")

    volume_eval, artifacts = Path(p["volume_root"]) / eval_stage.EVAL_DIR, Path(p["artifact_root"])
    ledger = _plain_file(Path(p["ledger"]), volume_eval)
    events, _ = proof_eval.read_ledger(ledger)  # hash chain and ledger rules
    starts = {e["run_id"]: e for e in events if e["event"] == "START"}
    failed = {e["run_id"] for e in events if e["event"] == "FAIL"}
    sources: list[tuple[Path, str]] = []
    eval_runs: list[dict[str, Any]] = []
    for arm, variant in eval_stage._runs(stage, p):
        rid = eval_stage.run_id(p["run_prefix"], arm, variant, p["subset_tag"])
        start = starts.get(rid)
        if (start is None or (start["arm"], start["variant"], start["subset"]) != (arm, variant, p["subset_tag"])
                or start["preregistration_sha256"] != p["prereg_sha256"]):
            raise CollectError(f"{rid}: no ledger START for {arm}/{variant} under this preregistration")
        base = artifacts / "runs" / rid
        present = [f for f in EVAL_RUN_FILES if (base / f).exists() or (base / f).is_symlink()]
        for file in present:
            _plain_file(base / file, artifacts)
        summary = read_json(base / "summary.json") if "summary.json" in present else None
        role = proof_eval.VARIANTS[variant][3]
        finished = proof_eval.FINISHED[role]  # COMPLETE (full course run) or PARTIAL (the subset runs)
        done = isinstance(summary, dict) and summary.get("status") == finished
        if rid in failed:
            if done:
                raise CollectError(f"{rid}: the ledger records FAILED but summary.json is {finished}")
            status = "FAILED"
        elif done:
            missing = [f for f in (*EVAL_REQUIRED, proof_eval.REGISTRATION_FILE if role == "statistical" else None)
                       if f and f not in present]
            if missing:
                raise CollectError(f"{rid}: {finished} run lacks {missing}")
            status = finished
        else:
            raise CollectError(f"{rid}: summary is not {finished} and the ledger has no FAIL for it "
                               "(record it with proof_eval ledger-fail first)")
        sources += [(base / f, f"eval/runs/{rid}/{f}") for f in present]
        eval_runs.append({"run_id": rid, "arm": arm, "variant": variant, "status": status,
                          "dir": f"eval/runs/{rid}", "files": present})
    prereg = _plain_file(Path(p["prereg"]), volume_eval)
    sidecar = _plain_file(Path(p["prereg"] + ".sha256"), volume_eval)
    if sha256_file(prereg) != p["prereg_sha256"] or sidecar.read_text(encoding="ascii").strip() != p["prereg_sha256"]:
        raise CollectError("prereg or its .sha256 differs from params.prereg_sha256")
    sources += [(ledger, f"eval/{ledger.name}"), (prereg, f"eval/{prereg.name}"),
                (sidecar, f"eval/{sidecar.name}")]
    model_ref_sha = None
    if stage != "eval-f0":
        export = Path(p["ft_export"])
        model_ref = _plain_file(export / "model_ref.json", export)  # the manifest only, never export/model
        model_ref_sha = sha256_file(model_ref)
        if stage == "eval-course" and model_ref_sha != p["ft_model_ref_sha256"]:
            raise CollectError("model_ref.json differs from params.ft_model_ref_sha256")
        sources.append((model_ref, "eval/model_ref.json"))
    if len({arc for _, arc in sources}) != len(sources):
        raise CollectError("two eval inputs would share one archive path")

    def fill(tar: tarfile.TarFile, records: list[dict[str, Any]]) -> None:
        _tar_run_dir(tar, run_dir, records)
        for path, arcname in sources:
            _add_member(tar, path, arcname, records)

    return _seal(out_dir, name, fill, kind=EVAL_KIND, stage=stage, runs=[run_dir.name], closure="CLOSED",
                 unclosed_runs=[], eval_runs=eval_runs, eval_run_ids=[r["run_id"] for r in eval_runs],
                 spec_sha256=spec_sha, ledger_sha256=sha256_file(ledger), prereg_sha256=p["prereg_sha256"],
                 model_ref_sha256=model_ref_sha)


# ---------------------------------------------------------------- verify

def pack_eval_resume(run_dir: Path, prereg: Path, ledger: Path, model_ref: Path,
                     calibration: Path, preflight: Path, out_dir: Path, name: str) -> dict:
    """Compressed, closed v17 evaluation state; never certify a training resume without optimizer state."""
    from mmdl.evaluation.writer import RunWriter
    from fine_tuning.error_note import v17_eval
    from fine_tuning.notebook_rl import proof_eval as pe

    _check_name(name)
    registration, prereg_sha = pe.load_preregistration(prereg)
    if registration["format"] != pe.FORMAT_V12:
        raise CollectError("v17 resume archive requires its original v12 registration")
    events, _ = pe.read_ledger(ledger)
    starts = [e for e in events if e["event"] == "START" and e["run_id"] == run_dir.name]
    if (len(starts) != 1 or starts[0]["preregistration_sha256"] != prereg_sha
            or sha256_file(calibration) != registration["calibration_sha256"]
            or sha256_file(model_ref) != registration["arms"][pe.ARM_SINGLE]["model_ref_sha256"]):
        raise CollectError("v17 resume archive inputs differ from the original START/model/calibration")
    split = "engine_validation" in registration
    training_seal = None
    manifest = read_json(run_dir / "run_manifest.json")
    required = ("variant_registration.json", "environment.json", "code_files.json", "locations.json",
                "resolved_eval_config.yaml", "resolved_hardware_config.yaml")
    if any(not (run_dir / file).is_file() for file in required):
        raise CollectError("v17 resume archive lacks run/backend/config/environment identity")
    if manifest["expected_ids"] != registration["evaluation_subsets"][starts[0].get("subset", "all")]["ids"]:
        raise CollectError("v17 resume archive expected IDs mismatch")
    # A live writer owns this same nonblocking lock; packing it is refused.
    with RunWriter(run_dir, manifest["identity"], manifest["expected_ids"], resume=True) as writer:
        finals = list((run_dir / "final_raw").glob("*.json"))
        variant = read_json(run_dir / pe.REGISTRATION_FILE)
        if variant.get("serving_kind") != pe.LORA_KIND:
            raise CollectError("v17 resume archive requires unmerged serving")
        for path in finals:
            final = v17_eval.checked(read_json(path))
            if final["registration_sha256"] != pe.digest(variant) or final["identity"] != writer.identity:
                raise CollectError("v17 resume archive has conflicting final raw")
            row = v17_eval.record_from_final(final)
            if row["id"] in writer.completed and v17_eval.stable_record(row) != v17_eval.stable_record(writer.completed[row["id"]]):
                raise CollectError("v17 resume archive sample differs from final raw")
        if any(not (run_dir / "final_raw" / (sid + ".json")).is_file() for sid in writer.completed):
            raise CollectError("analysis-only archive has samples without durable raw output receipts")
        if (writer.completed or finals) and not (run_dir / "backend.json").is_file():
            raise CollectError("v17 resume archive lacks backend identity")
        if (writer.completed or finals) and not list((run_dir / "inference_invocations").glob("*.json")):
            raise CollectError("v17 resume archive lacks invocation provenance")
        engine_sources = []
        if split:
            if not isinstance(variant.get("training_artifacts_path"), str):
                raise CollectError("v17 split resume archive lacks its training-artifact seal path")
            training_seal = Path(variant["training_artifacts_path"])
            _split_resume_inputs(registration, variant, model_ref, calibration, preflight, training_seal,
                                 verify_training=True)
            _split_resume_audit(run_dir, variant, refresh=True)
            engine_sources = _split_resume_evidence(calibration, run_dir)
        snapshot = v17_eval.incremental_manifest(run_dir, force=True)
        assert snapshot is not None
        sources = []
        for entry in snapshot["files"]:
            path = run_dir.parents[1] / entry["path"]
            _plain_file(path, run_dir.parents[1])
            if sha256_file(path) != entry["sha256"]:
                raise CollectError("v17 resume source changed after snapshot")
            sources.append((path, "eval/" + entry["path"]))
        # Include all remaining provenance and partial request records, but never model weights.
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                _plain_file(path, run_dir)
                sources.append((path, "eval/runs/" + run_dir.name + "/" + str(path.relative_to(run_dir))))
        for source, target in ((prereg, "prereg.json"), (prereg.with_name(prereg.name + ".sha256"), "prereg.json.sha256"),
                               (ledger, "ledger.jsonl"), (model_ref, "model_ref.json"), (calibration, "calibration.json"),
                               (preflight, "preflight.json")):
            sources.append((source, "eval/inputs/" + target))
        if training_seal is not None:
            sources.append((training_seal, "eval/inputs/training-artifacts.json"))
            sources.extend(engine_sources)
        unique: dict[str, Path] = {}
        for source, target in sources:
            if target in unique and unique[target] != source:
                raise CollectError("duplicate recovery archive target")
            unique[target] = source

        def fill(tar, records):
            for target, source in sorted(unique.items()):
                _add_member(tar, source, target, records)

        status = "FULL900_VERIFIED" if len(writer.completed) == 900 else "INTERRUPTED_RESUMABLE"
        return _seal(out_dir, name, fill, compression=True, kind=EVAL_KIND, stage="v17-eval-resume",
                     runs=[run_dir.name], closure="CLOSED", unclosed_runs=[],
                     eval_runs=[{"run_id": run_dir.name, "status": status}],
                     eval_run_ids=[run_dir.name], prereg_sha256=prereg_sha,
                     resume_capability="EVALUATION_ONLY", training_resume_certified=False,
                     snapshot_sha256=snapshot["sha256"])


def check_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    for member in members:
        parts = Path(member.name).parts
        if (not member.name or Path(member.name).is_absolute() or ".." in parts
                or member.name.startswith(("/", "\\")) or "\\" in member.name):
            raise CollectError(f"unsafe member path: {member.name!r}")
        if not (member.isreg() or member.isdir()):
            raise CollectError(f"member is not a regular file or directory "
                               f"(symlink/hardlink/device refused): {member.name!r}")
    return members


def private_root() -> Path:
    """Where eval packs (raw MMMU responses, never published) may live locally."""
    return Path(os.environ.get("MMDL_ARTIFACT_ROOT") or Path.home() / "mmdl-artifacts").resolve()


def _require_private(path: Path, label: str) -> None:
    if not path.resolve().is_relative_to(private_root()):
        raise CollectError(f"{label} {path} must lie under {private_root()} (private, never the repo)")


def verify_eval_resume_tree(root: Path, manifest: dict) -> dict:
    """An analysis pack cannot acquire evaluation-resume certification by changing a label."""
    from mmdl.evaluation.writer import _load_records
    from fine_tuning.error_note import v17_eval
    from fine_tuning.notebook_rl import proof_eval as pe

    if manifest.get("resume_capability") != "EVALUATION_ONLY" or manifest.get("training_resume_certified") is not False:
        raise CollectError("v17 evaluation archive cannot certify training resume")
    runs = manifest.get("eval_runs", [])
    if len(runs) != 1:
        raise CollectError("v17 resume archive must contain one logical run")
    rid, inputs = runs[0]["run_id"], root / "eval/inputs"
    base = root / "eval/runs" / rid
    required = ("run_manifest.json", "variant_registration.json", "environment.json", "code_files.json",
                "locations.json", "resolved_eval_config.yaml", "resolved_hardware_config.yaml")
    if any(not (base / name).is_file() for name in required):
        raise CollectError("v17 archive lacks identity/config inputs")
    prereg, prereg_sha = pe.load_preregistration(inputs / "prereg.json")
    events, _ = pe.read_ledger(inputs / "ledger.jsonl")
    starts = [e for e in events if e["event"] == "START" and e["run_id"] == rid]
    registered = read_json(base / pe.REGISTRATION_FILE)
    run = read_json(base / "run_manifest.json")
    split = "engine_validation" in prereg
    if split:
        _split_resume_inputs(prereg, registered, inputs / "model_ref.json", inputs / "calibration.json",
                             inputs / "preflight.json", inputs / "training-artifacts.json", verify_training=False)
        _split_resume_audit(base, registered, refresh=False)
        _split_resume_evidence(inputs / "calibration.json", base, archive_root=root)
    if (prereg["format"] != pe.FORMAT_V12 or len(starts) != 1
            or starts[0]["preregistration_sha256"] != prereg_sha
            or registered.get("preregistration_sha256") != prereg_sha
            or registered.get("serving_kind") != pe.LORA_KIND or (not split and registered.get("slots") != 8)
            or run["expected_ids"] != prereg["evaluation_subsets"][starts[0].get("subset", "all")]["ids"]
            or sha256_file(inputs / "model_ref.json") != registered["model_ref_sha256"]
            or sha256_file(inputs / "calibration.json") != registered["calibration_sha256"]
            or sha256_file(inputs / "preflight.json") != registered["preflight_sha256"]):
        raise CollectError("v17 archive registration/START/model/calibration mismatch")
    samples = _load_records(base / "samples")
    invocations = {r["id"]: r for r in (read_json(p) for p in (base / "inference_invocations").glob("*.json"))}
    finals = {}
    for path in (base / "final_raw").glob("*.json"):
        final = v17_eval.checked(read_json(path))
        row = v17_eval.record_from_final(final)
        if (path.stem != row["id"] or row["id"] not in run["expected_ids"]
                or final["registration_sha256"] != pe.digest(registered) or final["identity"] != run["identity"]
                or final["run_id"] != rid or final["invocation_id"] not in invocations
                or row["id"] not in invocations[final["invocation_id"]].get("attempted_ids", [])):
            raise CollectError("v17 archive FINAL_RAW identity mismatch")
        finals[row["id"]] = row
    if set(samples) - set(finals):
        raise CollectError("analysis-only samples lack required FINAL_RAW")
    if (samples or finals) and (not (base / "backend.json").is_file()
                               or not list((base / "inference_invocations").glob("*.json"))):
        raise CollectError("v17 archive lacks backend/invocation provenance")
    for sid, sample in samples.items():
        if v17_eval.stable_record(sample) != v17_eval.stable_record(finals[sid]):
            raise CollectError("v17 archive raw/sample conflict")
    for row in finals.values():
        for image in row.get("messages", [{}])[0].get("content", []):
            if image.get("type") == "image":
                asset = (base / image["image"]).resolve()
                if asset.parent != (root / "eval/assets/images").resolve() or sha256_file(asset) != image["sha256"]:
                    raise CollectError("v17 archive lacks immutable input image")
    return {"resume_capability": "EVALUATION_ONLY", "training_resume_certified": False,
            "completed_samples": len(samples), "durable_final_raw": len(finals)}


def verify_archive(archive: Path, manifest_path: Path, extract_to: Path,
                   receipt: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("version") != PACK_VERSION:
        raise CollectError("not a pod-pack-v1 manifest")
    is_eval = manifest.get("kind") == EVAL_KIND
    if is_eval:
        _require_private(archive, "eval archive")
        _require_private(extract_to, "eval extraction directory")
    if sha256_file(archive) != manifest["archive_sha256"]:
        raise CollectError("archive sha256 differs from the pack manifest")
    if extract_to.exists() or receipt.exists():
        raise CollectError("extraction directory or receipt already exists")
    with tarfile.open(archive, "r:*") as tar:
        members = check_members(tar)
        listed = {m.name for m in members if m.isreg()}
        if listed != {f["path"] for f in manifest["files"]}:
            raise CollectError("archive file set differs from the pack manifest")
        extract_to.mkdir(parents=True, mode=0o700 if is_eval else 0o777)
        tar.extractall(extract_to, members=members, filter="data")
    root = extract_to.resolve()
    for record in manifest["files"]:
        path = (extract_to / record["path"]).resolve()
        if root not in path.parents or path.is_symlink():
            raise CollectError(f"extracted path escapes the target: {record['path']}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise CollectError(f"extracted file differs from manifest: {record['path']}")
    resume_scope = verify_eval_resume_tree(root, manifest) if manifest.get("stage") == "v17-eval-resume" else {}
    result = {**resume_scope, "version": "pod-local-receipt-v1", "status": "LOCAL_VERIFIED",
              "archive_sha256": manifest["archive_sha256"],
              "manifest_sha256": sha256_file(manifest_path),
              "closure": manifest.get("closure", "CLOSED"), "pod_id": manifest.get("pod_id"),
              "files": len(manifest["files"]),
              "bytes": sum(f["bytes"] for f in manifest["files"]), "extracted_to": str(root),
              "verified_utc": utc_now()}
    if is_eval:
        result |= {"kind": EVAL_KIND, "stage": manifest["stage"],
                   "eval_runs": [{k: r[k] for k in ("run_id", "status")} for r in manifest["eval_runs"]]}
    write_json(receipt, result)
    return result


# ---------------------------------------------------------------- ledger

def _confirmed_user(evidence: dict[str, str]) -> None:
    if evidence.get("confirmed_by") != "user":
        raise CollectError("evidence confirmed_by=user required (the user performs and confirms "
                           "remote actions; nothing here calls an API)")


def _require(evidence: dict[str, str], *keys: str) -> None:
    missing = [key for key in keys if not evidence.get(key)]
    if missing:
        raise CollectError(f"evidence missing: {missing}")


def _when(evidence: dict[str, str], key: str) -> dt.datetime:
    try:
        return parse_utc(evidence[key])
    except ValueError as exc:
        raise CollectError(f"{key} must be a timezone-aware ISO time ({exc})") from exc


def check_evidence(state: str, evidence: dict[str, str], records: list[dict[str, Any]],
                   now: dt.datetime, receipt: dict[str, Any] | None = None) -> None:
    first = records[0]["evidence"] if records else {}
    if state == "REMOTE_VERIFIED":
        _require(evidence, "archive_sha256", "manifest_sha256", "pod_id")
        for key in ("archive_sha256", "manifest_sha256"):
            if not HEX64.match(evidence[key]):
                raise CollectError(f"{key} must be a lowercase sha256")
    elif state == "LOCAL_VERIFIED":
        _require(evidence, "archive_sha256", "local_receipt_sha256")
        if not HEX64.match(evidence["local_receipt_sha256"]):
            raise CollectError("local_receipt_sha256 must be a lowercase sha256")
        if evidence["archive_sha256"] != first["archive_sha256"]:
            raise CollectError("archive_sha256 differs from REMOTE_VERIFIED")
        receipt = receipt or {}
        if (receipt.get("version") != "pod-local-receipt-v1"
                or receipt.get("status") != "LOCAL_VERIFIED"
                or receipt.get("archive_sha256") != first["archive_sha256"]
                or receipt.get("manifest_sha256") != first["manifest_sha256"]):
            raise CollectError("LOCAL_VERIFIED needs the receipt written by `collect verify` for "
                               "the REMOTE_VERIFIED archive and manifest (--receipt-file)")
        if receipt.get("pod_id") != first["pod_id"]:
            raise CollectError(f"pod_id typed at REMOTE_VERIFIED ({first['pod_id']}) differs from "
                               f"the pod_id the pack recorded ({receipt.get('pod_id')})")
    elif state == "POD_TERMINATED":
        _require(evidence, "pod_id", "terminated_utc")
        _confirmed_user(evidence)
        if evidence["pod_id"] != first["pod_id"]:
            raise CollectError("pod_id differs from REMOTE_VERIFIED")
        stamp = _when(evidence, "terminated_utc")
        if stamp > now + SKEW or stamp < parse_utc(records[-1]["utc"]) - SKEW:
            raise CollectError("terminated_utc must lie between LOCAL_VERIFIED and now")
    elif state == "OWNED_STORAGE_DELETED":
        _require(evidence, "deleted_resources", "deleted_utc", "run_ledgers")
        _confirmed_user(evidence)
        stamp = _when(evidence, "deleted_utc")
        if stamp > now + SKEW or stamp < parse_utc(records[-1]["evidence"]["terminated_utc"]) - SKEW:
            raise CollectError("deleted_utc must lie between terminated_utc and now")
    elif state == "CLEANUP_VERIFIED":
        _require(evidence, "residual_query_utc", "residual_count")
        _confirmed_user(evidence)
        if evidence["residual_count"] != "0":
            raise CollectError("residual_count must be 0 to verify cleanup")
        stamp = _when(evidence, "residual_query_utc")
        if stamp > now + SKEW or stamp < parse_utc(records[-1]["evidence"]["deleted_utc"]) - SKEW:
            raise CollectError("residual_query_utc must lie between deleted_utc and now")


def _record_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records, previous = [], None
    for number, line in enumerate(path.read_text().splitlines()):
        record = json.loads(line)
        expected = ORDER[number] if number < len(ORDER) else None
        if (record.get("version") != LEDGER_VERSION or record.get("seq") != number
                or record.get("state") != expected or record.get("prev_sha256") != previous
                or record.get("record_sha256") != _record_hash(record)):
            raise CollectError(f"ledger corrupt or out of order at line {number + 1}")
        records.append(record)
        previous = record["record_sha256"]
    return records


def run_ledgers_evidence(current: Path, ledgers: list[Path]) -> str:
    """Every run that used the storage has REMOTE_VERIFIED and LOCAL_VERIFIED; JSON summary."""
    resolved = [Path(ledger).resolve() for ledger in ledgers]
    if not resolved or len(set(resolved)) != len(resolved) or current.resolve() not in resolved:
        raise CollectError("OWNED_STORAGE_DELETED needs --run-ledger for every run that used the "
                           "storage, this ledger included, each once")
    summary = []
    for ledger in resolved:
        records = read_ledger(ledger)
        if len(records) < 2:
            raise CollectError(f"{ledger}: run lacks REMOTE_VERIFIED + LOCAL_VERIFIED")
        summary.append({"ledger": str(ledger), "pod_id": records[0]["evidence"]["pod_id"],
                        "archive_sha256": records[0]["evidence"]["archive_sha256"],
                        "last_record_sha256": records[-1]["record_sha256"]})
    return json.dumps(summary, sort_keys=True)


def append_state(path: Path, state: str, evidence: dict[str, str],
                 now: dt.datetime | None = None, receipt_file: Path | None = None,
                 run_ledgers: list[Path] | None = None) -> dict[str, Any]:
    if state not in ORDER:
        raise CollectError(f"unknown state {state!r}; order is {' -> '.join(ORDER)}")
    receipt = None
    if receipt_file is not None:
        receipt = read_json(receipt_file)
        if not isinstance(receipt, dict):
            raise CollectError("receipt file is not a JSON object")
        evidence = {**evidence, "local_receipt_sha256": sha256_file(receipt_file)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        records = read_ledger(path)
        expected = ORDER[len(records)] if len(records) < len(ORDER) else None
        if state != expected:
            raise CollectError(f"out-of-order transition: next must be {expected}, got {state}")
        if state == "OWNED_STORAGE_DELETED":
            evidence = {**evidence, "run_ledgers": run_ledgers_evidence(path, run_ledgers or [])}
        check_evidence(state, evidence, records, now or dt.datetime.now(dt.timezone.utc), receipt)
        record: dict[str, Any] = {"version": LEDGER_VERSION, "seq": len(records), "state": state,
                                  "utc": utc_now(), "evidence": evidence,
                                  "prev_sha256": records[-1]["record_sha256"] if records else None}
        record["record_sha256"] = _record_hash(record)
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--unclosed-dead-owner", action="store_true",
                   help="also pack a hard-killed run whose owner is dead and that no process "
                        "still names; the manifest is labelled UNCLOSED")
    p.add_argument("run_dirs", nargs="+")
    e = sub.add_parser("pack-eval")
    e.add_argument("--params", required=True, help="the eval-stage-params-v1 file `eval_stage plan` read")
    e.add_argument("--stage", required=True, choices=("eval-f0", "eval-ft", "eval-course"))
    e.add_argument("--run-dir", required=True, help="the stage's start.sh run dir")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--name", help="archive name, default: the run dir name")
    v = sub.add_parser("verify")
    v.add_argument("--archive", required=True)
    v.add_argument("--manifest", required=True)
    v.add_argument("--extract-to", required=True)
    v.add_argument("--receipt", required=True)
    s = sub.add_parser("state")
    s.add_argument("--ledger", required=True)
    action = s.add_subparsers(dest="action", required=True)
    action.add_parser("show")
    a = action.add_parser("append")
    a.add_argument("state")
    a.add_argument("--evidence", action="append", default=[], metavar="KEY=VALUE")
    a.add_argument("--receipt-file", help="sets local_receipt_sha256 from this file's bytes")
    a.add_argument("--run-ledger", action="append", default=[], metavar="LEDGER",
                   help="OWNED_STORAGE_DELETED: ledger of every run that used the storage")
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            result = pack([Path(d) for d in args.run_dirs], Path(args.out_dir), args.name,
                          args.unclosed_dead_owner)
            print(json.dumps({k: result[k] for k in ("archive", "archive_sha256", "runs",
                                                     "closure")}))
        elif args.command == "pack-eval":
            result = pack_eval(read_json(args.params), args.stage, Path(args.run_dir),
                               Path(args.out_dir), args.name)
            print(json.dumps({k: result[k] for k in ("archive", "archive_sha256", "runs", "closure",
                                                     "stage", "eval_runs")}))
        elif args.command == "verify":
            result = verify_archive(Path(args.archive), Path(args.manifest),
                                    Path(args.extract_to), Path(args.receipt))
            print(json.dumps(result, indent=2))
        elif args.action == "show":
            print(json.dumps(read_ledger(Path(args.ledger)), indent=2))
        else:
            evidence = dict(item.partition("=")[::2] for item in args.evidence)
            receipt_file = Path(args.receipt_file) if args.receipt_file else None
            print(json.dumps(append_state(Path(args.ledger), args.state, evidence,
                                          receipt_file=receipt_file,
                                          run_ledgers=[Path(p) for p in args.run_ledger]),
                             indent=2))
        return 0
    except (CollectError, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
