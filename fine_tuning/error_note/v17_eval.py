"""v17-only evaluation durability, capacity checks and measured operating receipts.

All inference still uses VLLMBackend.generate_stream and all commits use RunWriter.
No function here downloads a model, starts a Pod, or chooses between answers.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import sample_seed

SLOTS = 8
CALIBRATION_VERSION = "v17-eval-calibration-v1"
CAPACITY_RULE = {"version": "v17-kv-capacity-v1", "slots": SLOTS, "usable_fraction": .90,
                 "max_model_len": 40960, "vllm_version": "0.11.0"}
TIMING_FIELDS = {"_record_sha256", "total_sample_seconds", "total_sample_timing_policy",
                 "recovery_provenance"}


def source_sha256() -> str:
    """Use the full fine_tuning source inventory shared with training preflight."""
    from .single import source_manifest

    return digest(source_manifest()["actual_files"])


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def seal(value: dict) -> dict:
    body = {k: v for k, v in value.items() if k != "sha256"}
    return body | {"sha256": digest(body)}


def checked(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("sha256") != seal(value)["sha256"]:
        raise ValueError("v17: receipt digest mismatch")
    return value


def exclusive(path: Path, value: dict) -> None:
    from mmdl.evaluation.writer import _write_json_exclusive

    _write_json_exclusive(path, value)


def append_event(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def capacity_receipt(engine: Any, version: str, *, max_model_len: int = 40960,
                     slots: int = SLOTS) -> dict:
    """Read the pinned in-process scheduler's allocated blocks, never a VRAM estimate."""
    if version != CAPACITY_RULE["vllm_version"] or slots != SLOTS or max_model_len != 40960:
        raise ValueError("v17: unsupported KV engine version or execution contract")
    try:
        scheduler = engine.engine_core.engine_core.scheduler
        config = scheduler.kv_cache_config
        blocks = config.num_blocks
        sizes = {group.kv_cache_spec.block_size for group in config.kv_cache_groups}
        if len(sizes) != 1:
            raise ValueError("mixed KV block sizes")
        block_size = sizes.pop()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("v17: actual allocated KV block structure unavailable") from exc
    if type(blocks) is not int or blocks < 1 or type(block_size) is not int or block_size < 1:
        raise ValueError("v17: invalid actual KV capacity")
    reserved = math.ceil(max_model_len / block_size) * block_size
    actual, required = blocks * block_size, slots * reserved
    usable = math.floor(.90 * actual)
    result = seal({"version": "v17-kv-receipt-v1", "rule_sha256": digest(CAPACITY_RULE),
                   "vllm_version": version, "slots": slots, "max_model_len": max_model_len,
                   "actual_blocks": blocks, "block_size": block_size,
                   "actual_kv_token_capacity": actual, "reserved_per_request": reserved,
                   "required_tokens": required, "usable_tokens": usable,
                   "pass": required <= usable})
    if not result["pass"]:
        raise ValueError(f"v17: insufficient actual KV capacity: {required} > {usable}")
    return result


ENGINE_CAPACITY_KEYS = ("actual_blocks", "actual_kv_token_capacity", "usable_tokens", "safe_slots", "sha256")
PROBE_CAP_SECONDS = 2700  # 2026-11-17 A100 eager 8 slots: ~15 tok/s per request at long context, the first wave
# reaches 32,768 tokens after ~2,000 s (900 s observed only 17,423)


def calibration_receipt(probe: dict) -> dict:
    """Conservative measured scenario; LOW_COVERAGE is never a runnable timeout."""
    checked(probe)
    if (probe.get("version") != "v17-eval-probe-v1" or probe.get("slots") != SLOTS
            or not 1 <= probe.get("items", 0) <= 16 or not 0 < probe.get("wall_seconds", 0) <= PROBE_CAP_SECONDS
            or probe.get("dtype") != "bfloat16" or probe.get("backend") != "vllm-0.11.0"
            or probe.get("config_sha256") is None
            or not re.fullmatch(r"[a-f0-9]{64}", str(probe.get("environment_sha256", "")))
            or not re.fullmatch(r"[a-f0-9]{40}", str(probe.get("code_commit", "")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(probe.get("source_sha256", "")))):
        raise ValueError("v17: invalid bounded performance probe")
    wall, tokens = probe["wall_seconds"], probe.get("generated_tokens", 0)
    windows = probe.get("sixty_second_rates", [])
    if any(not isinstance(q, (int, float)) or not math.isfinite(q) or q < 0 for q in windows):
        raise ValueError("v17: invalid throughput windows")
    positive = [q for q in windows if q > 0]
    q_run = tokens / wall
    q_safe = min(q_run, statistics.median(windows)) if positive and len(positive) == len(windows) else 0
    allowances = [probe.get(k) for k in ("load_finalize_allowance_seconds", "long_tail_allowance_seconds")]
    if any(not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 for n in allowances):
        raise ValueError("v17: explicit finite load/finalize and long-tail allowances required")
    adequate = (q_safe > 0 and probe.get("max_observed_context_tokens", 0) >= 32768
                and probe.get("cumulative_preemptions") == 0
                and probe.get("disjoint_receipt_sha256") is not None
                and bool(probe.get("long_tail_basis")))
    allowance = float(probe["load_finalize_allowance_seconds"]) + float(probe["long_tail_allowance_seconds"])
    seconds = math.ceil(1.25 * (1.5 * 4_116_922 / q_safe + allowance)) if q_safe else None
    return seal({"version": CALIBRATION_VERSION,
                 "status": "CALIBRATED" if adequate else "CALIBRATION_LOW_COVERAGE",
                 "probe_sha256": probe["sha256"], "probe": probe, "q_run": q_run,
                 "q_safe": q_safe, "slots": SLOTS, "capacity_rule_sha256": digest(CAPACITY_RULE),
                 "evaluation_watchdog_seconds": seconds,
                 "code_commit": probe.get("code_commit"), "source_sha256": probe.get("source_sha256"),
                 "scenario": {"historical_tokens": 4_116_922, "output_factor": 1.5, "wall_factor": 1.25},
                 "statistical_upper_bound": False})


def load_calibration(path: Path) -> dict:
    value = checked(read_json(path))
    if value.get("version") == "v17-eval-engine-validation-v1":
        from .v17_eval_engine import load_engine_validation

        return load_engine_validation(path)
    if value != calibration_receipt(value.get("probe", {})) or value.get("status") != "CALIBRATED":
        raise ValueError("v17: calibration is incomplete, altered, or has low context coverage")
    return value


def calibration_timeout(receipt: dict) -> int:
    if receipt.get("version") == "v17-eval-engine-validation-v1":
        from .v17_eval_engine import validate_engine_receipt

        return validate_engine_receipt(receipt)["evaluation_watchdog_seconds"]
    checked(receipt)
    if receipt != calibration_receipt(receipt.get("probe", {})) or receipt.get("status") != "CALIBRATED":
        raise ValueError("v17: measured calibrated timeout required")
    seconds = receipt["evaluation_watchdog_seconds"]
    if type(seconds) is not int or seconds < 1:
        raise ValueError("v17: finite positive timeout required")
    return seconds


def doctor_check(value: dict) -> dict:
    environment, storage = value.get("environment", {}), value.get("storage", {})
    torch = environment.get("torch", {})
    if (torch.get("cuda_available") is not True or torch.get("bf16_probe", {}).get("passed") is not True
            or not storage or storage.get("sufficient") is not True
            or not storage.get("filesystems")
            or any(fs.get("sufficient") is not True for fs in storage["filesystems"])):
        raise ValueError("v17: doctor must confirm actual CUDA/BF16 and storage headroom")
    return {"status": "DOCTOR_VERIFIED", "doctor_sha256": digest(value)}


class Heartbeat:
    """A stalled engine gets warnings and a durable heartbeat, never an automatic retry."""

    def __init__(self, root: Path, *, clock=time.monotonic, expected_count: int | None = None):
        self.root, self.clock = Path(root), clock
        self.started = self.last_token = self.last_compute = self.last_emit = clock()
        self.steps = self.tokens = self.prefill = self.completed = 0
        self.completed_output_tokens = 0
        self.preemptions: int | None = None
        self.previous_preemptions: int | None = None
        self.preemption_at_token = 0
        self.active: dict[str, float] = {}
        self.lengths: dict[str, int] = {}
        self.kv_used = self.kv_total = None
        self.waiting = None
        self.expected_count = expected_count
        self.emitted_tokens = self.emitted_prefill = 0
        self.live_context: dict | None = None
        self.lock = threading.RLock()

    def observe(self, *, tokens: int | None = None, prefill: int | None = None,
                preemptions: int | None = None) -> None:
        now = self.clock()
        with self.lock:
            if tokens is not None and tokens > self.tokens:
                self.tokens, self.last_token, self.last_compute = tokens, now, now
                self.preemption_at_token = preemptions if preemptions is not None else 0
            if prefill is not None and prefill > self.prefill:
                self.prefill, self.last_compute = prefill, now
            if preemptions is not None:
                self.preemptions = preemptions

    def emit(self, *, force: bool = False) -> dict | None:
        with self.lock:
            now = self.clock()
            if not force and now - self.last_emit < 10:
                return None
            warnings = []
            if self.active and now - self.last_token >= 180:
                warnings.append("TOKEN_STALL_WARNING")
                if self.preemptions is not None and self.preemptions > self.preemption_at_token:
                    warnings.append("KV_RECOMPUTE_WARNING")
                if now - self.last_compute >= 180:
                    warnings.append("ENGINE_NO_PROGRESS_WARNING")
            row = {"wall_time": utc(), "elapsed_seconds": now - self.started,
                   "engine_steps": self.steps, "requests_running": len(self.active) - (self.waiting or 0),
                   "requests_waiting": self.waiting, "requests_completed": self.completed,
                   "cumulative_output_tokens": self.tokens,
                   "new_output_tokens_since_last": self.tokens - self.emitted_tokens,
                   "max_observed_output_tokens_by_request": dict(self.lengths),
                   "prefill_tokens": self.prefill, "prefill_tokens_since_last": self.prefill - self.emitted_prefill,
                   "kv_blocks_used": self.kv_used,
                   "kv_blocks_total": self.kv_total, "cumulative_preemptions": self.preemptions,
                   "last_new_token_seconds_ago": now - self.last_token,
                   "last_compute_progress_seconds_ago": now - self.last_compute,
                   "oldest_request_seconds": max((now - t for t in self.active.values()), default=0),
                   "active_remaining_output_cap_tokens": sum(max(0, 32768 - self.lengths.get(sid, 0)) for sid in self.active),
                   "pending_unsubmitted_count": (max(0, self.expected_count - self.completed - len(self.active))
                                                   if self.expected_count is not None else None),
                   "warnings": warnings}
            if self.live_context is not None:
                assert self.expected_count is not None
                from .v18b_eval import progress
                settings = self.live_context["settings"]
                row["v18b_live"] = progress(elapsed=now - self.started, tokens=self.tokens,
                    completed=self.completed, expected=self.expected_count,
                    historical_tokens=self.live_context["historical_tokens"],
                    remaining_seconds=self.live_context["planning_deadline"] - time.time(), preemptions=self.preemptions,
                    completed_output_tokens=self.completed_output_tokens,
                    active_tokens=sum(self.lengths.get(sid, 0) for sid in self.active))
                row["v18b_live"].update(first_billed_at=settings["first_billed_at"], run_id=settings["run_id"],
                    execution_settings_sha256=digest(settings), deadline_epoch=self.live_context["deadline"])
                from .v18b_execution import planning_status
                row["v18b_live"]["planning"] = planning_status(settings)
                warnings.extend(row["v18b_live"]["planning"]["warnings"])
                if self.previous_preemptions is not None and self.preemptions is not None and self.preemptions > self.previous_preemptions:
                    warnings.append("KV_PREEMPTION_INCREASE")
                self.previous_preemptions = self.preemptions
            write_json(self.root / "heartbeat.json", row)
            append_event(self.root / "heartbeat.jsonl", row)
            self.last_emit = now
            self.emitted_tokens, self.emitted_prefill = self.tokens, self.prefill
            return row


def stable_record(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in TIMING_FIELDS}


def record_from_final(final: dict) -> dict:
    """Same V8 and narrative policy as engine.run; receipt retains the original request metadata."""
    from mmdl.evaluation.parsers import score_with_parser

    checked(final)
    if final.get("version") != "v17-final-raw-v1":
        raise ValueError("v17: unknown FINAL_RAW contract")
    record, output = final["record"], final["output"]
    if record["id"] != output["sample_id"] or record["seed"] != sample_seed(3407, record["id"]):
        raise ValueError("v17: FINAL_RAW request ID/seed mismatch")
    if len(output["generated_token_ids"]) != output["generated_tokens"]:
        raise ValueError("v17: FINAL_RAW token count mismatch")
    score = score_with_parser(output["raw_response"], record["question_type"], record["options"],
                              record["answer"], "team-final-answer-v8", output.get("finish_reason"),
                              question=record["question"])
    raw = output["raw_response"]
    just_answer = bool(re.fullmatch(
        r"\s*(?:(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?:is|:)\s*)?"
        r"[*\s]*\(?[A-Z]\)?[*\.\s]*", raw, re.IGNORECASE
    )) if record["question_type"] == "multiple-choice" else len(raw.split()) <= 3
    return record | output | score | {
        "model_explanation": None if just_answer else raw or None,
        "inference_invocation_id": final["invocation_id"], "run_id": final["run_id"],
        "explanation_policy": "verbatim narrative, no synthesized explanation; answer-only/short open answer has none; full raw always retained",
        "status": "COMPLETED"}


def recover_finals(writer: Any, registration_sha256: str) -> list[str]:
    recovered = []
    for path in sorted((writer.run_dir / "final_raw").glob("*.json")):
        final = checked(read_json(path))
        if (final.get("registration_sha256") != registration_sha256
                or final.get("identity") != writer.identity or final.get("run_id") != writer.run_dir.name):
            raise ValueError("v17: FINAL_RAW belongs to another logical run")
        row = record_from_final(final)
        sid = row["id"]
        if path.name != sid + ".json" or sid not in writer.expected_ids:
            raise ValueError("v17: unexpected FINAL_RAW ID")
        if sid in writer.completed:
            if stable_record(writer.completed[sid]) != stable_record(row):
                raise ValueError("v17: conflicting immutable sample and FINAL_RAW")
        else:
            writer.save(row | {"recovery_provenance": {"final_raw_sha256": sha256_file(path),
                                                       "recovered_at": utc(), "latency_unavailable": True}})
            recovered.append(sid)
    return recovered


def cpu_resume_finalize(known: argparse.Namespace, cfg: dict, registration: dict) -> bool:
    """Rebuild the existing engine identity on CPU, then recover/finalize without loading a backend."""
    import yaml
    from mmdl.evaluation import engine as evaluation
    from mmdl.evaluation.writer import RunWriter, finalize, render
    from mmdl.runtime.artifacts import file_records
    from mmdl.runtime.environment import collect_environment
    from fine_tuning.notebook_rl import proof_eval as pe

    root, run_dir = pe.ROOT, known.artifact_root / "runs" / known.run_id
    base = read_json(root / "manifests/models/baseline.json")
    model = read_json(known.model_ref)
    hw = yaml.safe_load(known.hardware.read_text())
    files = file_records(root, list((root / "prompts").glob("mmmu_*_v1.txt")) + [
        root / "src/mmdl/evaluation/prompt.py", root / "src/mmdl/evaluation/parsers.py",
        root / "third_party/mmmu/eval_utils.py"])
    files += file_records(root, [root / "src/mmdl/evaluation/final_answer_parser.py",
                                 root / "src/mmdl/evaluation/backends/input_preparation.py"])
    files += file_records(root, [root / "src/mmdl/evaluation/final_answer_parser_v8.py"])
    protocol_hash = digest({"config": cfg, "files": files,
                            "processor": [r for r in base["files"] if not r["path"].endswith(".safetensors")],
                            "dataset": read_json(known.data_root / "manifest.json")["files"]})
    identity = {"model_sha256": digest(model["files"]), "protocol_sha256": protocol_hash,
                "code_sha256": digest(evaluation.code_records(root)),
                "environment_sha256": digest(evaluation.environment_identity(collect_environment(probe=False))),
                "hardware": hw, "ids_sha256": digest(registration["expected_ids"]), "mode": known.mode,
                "git_commit": evaluation.git_commit(root, known.require_commit),
                "job_id": known.job_id, "run_role": known.run_role}
    with pe.registered_variants(slots=registration.get("slots", SLOTS)):
        evaluation.verify_model(known.model_path, model)
        evaluation.verify_model(known.base_path, base)
    with RunWriter(run_dir, identity, registration["expected_ids"], resume=True) as writer:
        pe._record_registration(run_dir, registration)
        recover_finals(writer, digest(registration))
        for sid, row in writer.completed.items():
            if row["seed"] != sample_seed(3407, sid):
                raise ValueError("v17: completed sample seed differs from registration")
            for image in row.get("messages", [{}])[0].get("content", []):
                if image.get("type") == "image" and sha256_file((run_dir / image["image"]).resolve()) != image["sha256"]:
                    raise ValueError("v17: completed sample input asset hash mismatch")
        if set(writer.completed) != set(writer.expected_ids):
            return False
        summary = finalize(run_dir, writer.expected_ids, known.mode)
        write_json(run_dir / "summary.json", summary | {"v17_cpu_finalize": True})
        render(run_dir)
    return True


def deadline_seconds(start: dict, timeout: int, extensions: list[dict], *, now: float | None = None,
                     funding_deadline: float | None = None, recovery_reserve_seconds: float = 0) -> float:
    if (type(timeout) is not int or timeout <= 0 or recovery_reserve_seconds < 0
            or not math.isfinite(recovery_reserve_seconds)
            or (funding_deadline is not None and not math.isfinite(funding_deadline))):
        raise ValueError("v17: finite cumulative watchdog and funding boundary required")
    started = datetime.fromisoformat(start["time"]).timestamp()
    deadline = started + timeout
    previous = "0" * 64
    for extension in extensions:
        checked(extension)
        if (extension.get("version") != "v17-eval-extension-v1" or extension.get("previous") != previous
                or extension.get("run_id") != start["run_id"] or extension.get("confirmed_by") != "user"
                or extension.get("preregistration_sha256") != start.get("preregistration_sha256")
                or not extension.get("authorization") or extension.get("old_deadline") != deadline
                or not isinstance(extension.get("new_deadline"), (int, float))
                or not math.isfinite(extension["new_deadline"]) or extension["new_deadline"] <= deadline):
            raise ValueError("v17: invalid append-only deadline extension")
        deadline, previous = extension["new_deadline"], extension["sha256"]
    if funding_deadline is not None:
        deadline = min(deadline, funding_deadline - recovery_reserve_seconds)
    return deadline - (time.time() if now is None else now)


def paired_report(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    from fine_tuning.error_note.contract import is_loop
    from fine_tuning.notebook_rl.proof_eval import mcnemar_exact

    if not reference or set(reference) != set(candidate):
        raise ValueError("v17: paired comparison needs identical complete unique IDs")
    def bad(row):
        return row.get("finish_reason") == "length" or row.get("is_loop") is True or is_loop(row.get("raw_response", ""))
    def cell(ids):
        f0 = {i: bool(reference[i]["correct"]) for i in ids}
        ft = {i: bool(candidate[i]["correct"]) for i in ids}
        paired = mcnemar_exact(f0, ft)
        return {"n": len(ids), "reference_correct": sum(f0.values()), "candidate_correct": sum(ft.values()),
                "net_change": sum(ft.values()) - sum(f0.values()), **paired}
    normal = [i for i in reference if not bad(reference[i])]
    return {"version": "v17-paired-report-v1", "complete": True, "overall": cell(list(reference)),
            "reference_length_or_loop": cell([i for i in reference if bad(reference[i])]),
            "reference_normal": cell(normal),
            "both_normal": cell([i for i in normal if not bad(candidate[i])]),
            "candidate_only_length_or_loop": cell([i for i in normal if bad(candidate[i])]),
            "content_accuracy_improvement": ("UNPROVEN" if cell(normal)["net_change"] <= 0 else
                                               "POSITIVE_OBSERVED_CHANGE_NOT_CERTIFICATION")}


def evaluation_gate(development: dict, effect: dict | None) -> dict:
    if not development.get("complete"):
        raise ValueError("v17: final development comparison incomplete")
    dev = development["overall"]
    if dev["net_change"] < 0 and dev["p_value"] < .05:
        status = "DEGRADED"
    elif not effect or not effect.get("complete") or effect.get("eligible") is not True:
        status = "EVAL_REVIEW_REQUIRED"
    elif dev["net_change"] <= 0 and effect["overall"]["net_change"] <= 0:
        status = "EVAL_REVIEW_REQUIRED"
    else:
        status = "EVALUATION_CANDIDATE"
    return {"version": "v17-evaluation-gate-v1", "status": status,
            "development": development, "effect": effect, "improvement_certified": False}


def incremental_manifest(run_dir: Path, *, force: bool = False) -> dict | None:
    """Snapshot immutable samples/raw plus resume identity every 32 commits or five minutes."""
    run_dir = Path(run_dir)
    (run_dir / "recovery_manifests").mkdir(parents=True, exist_ok=True)
    # Each caller opens the same inode: thread and recovery-process writers share this lock.
    with (run_dir / "recovery_manifests/.manifest.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _incremental_manifest_locked(run_dir, force=force)


def _incremental_manifest_locked(run_dir: Path, *, force: bool) -> dict | None:
    previous = sorted((run_dir / "recovery_manifests").glob("*.json"))
    old = checked(read_json(previous[-1])) if previous else None
    samples = sorted((run_dir / "samples").glob("*.json"))
    if not force and old and len(samples) - old["completed_count"] < 32 and time.time() - old["created_epoch"] < 300:
        return None
    paths = list(samples) + sorted((run_dir / "final_raw").glob("*.json"))
    paths += sorted((run_dir / "inference_invocations").glob("*.json"))
    paths += [p for p in run_dir.glob("*.json") if p.is_file()]
    paths += [p for p in run_dir.glob("*.yaml") if p.is_file()]
    paths += sorted(run_dir.glob("*.jsonl"))
    assets = set()
    image_records = [read_json(p) for p in samples]
    image_records += [read_json(p)["record"] for p in (run_dir / "final_raw").glob("*.json")]
    for record in image_records:
        for image in record.get("images", []):
            asset = (run_dir / image).resolve()
            if asset.parent != (run_dir.parents[1] / "assets/images").resolve():
                raise ValueError("v17: inspection image outside owned input assets")
            assets.add(asset)
    paths += sorted(assets)
    records = []
    snapshot_dir = run_dir / "recovery_manifests" / ("snapshot-" + uuid.uuid4().hex)
    for path in sorted(set(paths)):
        source = str(path.relative_to(run_dir.parents[1]))
        if path.parent == run_dir:
            # Mutable receipts/logs become immutable bytes; appenders finish a whole locked line first.
            with path.open("rb") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
                content = stream.read()
            if path.suffix == ".jsonl" and content and not content.endswith(b"\n"):
                content = content[:content.rfind(b"\n") + 1]
            snapshot_dir.mkdir(exist_ok=True)
            path = snapshot_dir / path.name
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        records.append({"path": str(path.relative_to(run_dir.parents[1])), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        **({"source_path": source, "closed_bytes": len(content)} if path.parent == snapshot_dir else {})})
    if snapshot_dir.is_dir():
        descriptor = os.open(snapshot_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    result = seal({"version": "v17-eval-recovery-manifest-v1", "run_id": run_dir.name,
                   "created_epoch": time.time(), "completed_count": len(samples),
                   "previous_sha256": old["sha256"] if old else None,
                   "files": records,
                   "transfer_status": "LOCAL_SNAPSHOT_ONLY_NOT_REMOTE_VERIFIED"})
    exclusive(run_dir / "recovery_manifests" / f"{len(previous):06d}.json", result)
    return result


def _metrics(llm: Any) -> dict:
    try:
        values = {m.name: m.value for m in llm.get_metrics() if hasattr(m, "value")}
    except (AttributeError, RuntimeError, TypeError):
        return {}
    names = {"tokens": "vllm:generation_tokens", "prefill": "vllm:prompt_tokens",
             "preemptions": "vllm:num_preemptions"}
    return {key: int(values[name]) for key, name in names.items() if name in values}


ORDER_RULE = "sha256('v17-eval-order:' + id)"


def submission_order(ids: list[str]) -> list[str]:
    if len(ids) != len(set(ids)) or any(not isinstance(sid, str) or not sid for sid in ids):
        raise ValueError("v17: unique nonempty submission IDs required")
    return sorted(ids, key=lambda sid: (hashlib.sha256(("v17-eval-order:" + sid).encode()).hexdigest(), sid))


def split_execution_contract(validation: dict, ids: list[str]) -> dict:
    from .v17_eval_engine import validate_engine_receipt

    validate_engine_receipt(validation)
    rule = "floor(0.9 * actual_kv_token_capacity / 40960)"
    if validation.get("training_protocol_version") in ("error-note-v18", "error-note-v18b"):
        rule = "min(9, " + rule + ")"
    return {"version": "v17-evaluation-execution-v2", "engine_mode": validation["selected_mode"],
            "slots": validation["slots"], "capacity_rule": rule,
            "actual_kv_token_capacity": validation["actual_kv_token_capacity"],
            "submission_order_rule": ORDER_RULE, "submission_order_sha256": digest(submission_order(ids)),
            "serving_kind": "unmerged-lora", "same_start_resume": True, "downtime_counts": True,
            "deadline_extension": "append-only-user-receipt",
            "b2_execution_change": {"baseline": "eager", "selected": validation["selected_mode"],
                                    "bit_identical_claimed": False}}



def split_bindings(engine_validation: Path, training_artifacts: Path, model_ref: Path,
                   *, config: dict | None = None, hardware: dict | None = None,
                   code_commit: str | None = None) -> dict:
    from fine_tuning.notebook_rl import proof_eval as pe
    from .v17_artifacts import evaluation_identity, protected_evaluation, verify_training_artifacts
    from .v17_eval_engine import load_engine_validation

    validation = load_engine_validation(engine_validation)
    artifacts = verify_training_artifacts(training_artifacts)
    if artifacts.get("training_protocol_version") in ("error-note-v18", "error-note-v18b") and validation.get("training_protocol_version") != artifacts["training_protocol_version"]:
        raise ValueError("v18: engine validation cannot reuse a v17 training seal")
    if (validation["training_artifact_sha256"] != sha256_file(training_artifacts)
            or validation["model_ref_sha256"] != sha256_file(model_ref)
            or artifacts["export"]["model_ref"]["file_sha256"] != sha256_file(model_ref)
            or artifacts["protected_evaluation"] != protected_evaluation(pe.ROOT)):
        raise ValueError("v17: final model/training artifacts/protected evaluation binding mismatch")
    identity = evaluation_identity(pe.ROOT, code_commit or validation["evaluation_commit"])
    if (validation["evaluation_commit"] != identity["code_commit"]
            or validation["source_sha256"] != identity["source_sha256"]):
        raise ValueError("v17: evaluation source/commit differs from final-model validation")
    if config is not None and validation["config_sha256"] != digest(config):
        raise ValueError("v17: final-model validation config mismatch")
    if hardware is not None and validation["hardware_sha256"] != digest(hardware):
        raise ValueError("v17: final-model validation hardware mismatch")
    return validation


def reorder_request_stream(requests: Any, expected_pending: list[str]) -> list[str]:
    # The pinned protected engine closes over IDs before loading each image; fail if that contract changes.
    if (not inspect.isgenerator(requests) or requests.gi_code.co_name != "request_stream"
            or inspect.getgeneratorstate(requests) != inspect.GEN_CREATED):
        raise ValueError("v17: unsupported lazy protected-engine request stream")
    frame = requests.gi_frame
    pending = frame.f_locals.get("pending") if frame is not None else None
    if (type(pending) is not list or pending != sorted(expected_pending)
            or len(pending) != len(set(pending))):
        raise ValueError("v17: lazy pending IDs differ from immutable unfinished membership")
    pending[:] = submission_order(pending)
    return list(pending)


def submission_audit(run_dir: Path, registration: dict) -> dict:
    path = run_dir / "submission_events.jsonl"
    if not path.exists() and (any((run_dir / "samples").glob("*.json")) or any((run_dir / "final_raw").glob("*.json"))):
        raise ValueError("v17: persisted answers are missing the durable submission audit")
    events = [checked(json.loads(line)) for line in path.read_text().splitlines()] if path.exists() else []
    previous, current, position, accepted, intents = None, None, 0, [], []
    registration_hash = digest(registration)
    pending: list[str] = []
    awaiting = None
    for event in events:
        if (event.get("registration_sha256") != registration_hash or event.get("previous") != previous):
            raise ValueError("v17: submission audit identity/hash chain mismatch")
        kind = event.get("event")
        if kind == "BEGIN":
            completed = event.get("completed_ids", [])
            pending = event.get("pending_ids", [])
            if (not set(completed) <= set(registration["expected_ids"])
                    or len(completed) != len(set(completed))
                    or pending != submission_order([sid for sid in registration["expected_ids"] if sid not in completed])):
                raise ValueError("v17: submission audit pending order mismatch")
            current, position, awaiting = event["invocation_id"], 0, None
        elif event.get("invocation_id") != current or current is None:
            raise ValueError("v17: submission event lacks its invocation")
        elif kind == "INTENT":
            if awaiting is not None or position >= len(pending) or event.get("id") != pending[position]:
                raise ValueError("v17: actual submission order differs from planned order")
            awaiting = event["id"]
            intents.append(awaiting)
            position += 1
        elif kind == "ACCEPTED":
            if awaiting is None or event.get("id") != awaiting:
                raise ValueError("v17: accepted submission has no matching durable intent")
            accepted.append(awaiting)
            awaiting = None
        else:
            raise ValueError("v17: unknown submission event")
        previous = event["sha256"]
    return seal({"version": "v17-submission-audit-v1", "events": len(events), "last_event_sha256": previous,
                 "submission_order_sha256": digest(submission_order(registration["expected_ids"])),
                 "actual_submission_sha256": digest(accepted), "actual_submitted_ids": accepted,
                 "submission_intent_sha256": digest(intents), "submission_intent_ids": intents,
                 "interrupted_intent_id": awaiting})


def _submission_event(run_dir: Path, registration: dict, *, audit_state: dict | None = None, **fields: Any) -> None:
    if audit_state is None:
        audit_state = {"last_event_sha256": submission_audit(run_dir, registration)["last_event_sha256"]}
    event = seal({"version": "v17-submission-event-v1", "registration_sha256": digest(registration),
                  "previous": audit_state["last_event_sha256"], **fields})
    append_event(run_dir / "submission_events.jsonl", event)
    audit_state["last_event_sha256"] = event["sha256"]


@contextmanager
def evaluation_hooks(run_dir: Path, registration: dict, deadline: float | None) -> Iterator[Any]:
    """Temporary hooks are scoped to this adapter invocation and always restored."""
    from mmdl.evaluation import engine as evaluation
    from mmdl.evaluation.writer import RunWriter, finalize, render
    from fine_tuning.notebook_rl import proof_eval as pe

    original_writer, original_prepare = evaluation.RunWriter, evaluation.prepare_request
    registration_hash = digest(registration)
    records: dict[str, dict] = {}
    state: dict[str, Any] = {}
    split = registration.get("execution_contract", {}).get("version") == "v17-evaluation-execution-v2"
    slots = registration.get("slots", SLOTS)
    restores = []
    heartbeat = Heartbeat(run_dir, expected_count=len(registration["expected_ids"]))
    if "execution_settings" in registration:
        from .v18b_execution import deadline_epoch
        heartbeat.live_context = {"settings": registration["execution_settings"], "deadline": deadline,
                                  "planning_deadline": deadline if deadline is not None else deadline_epoch(registration["execution_settings"]),
                                  "historical_tokens": math.ceil(4_116_922 * len(registration["expected_ids"]) / 900)}
    finished = threading.Event()

    def snapshot_loop():
        while not finished.wait(10):
            if "writer" not in state:
                continue  # the RunWriter creates run_dir (exist_ok=False); a heartbeat must never create it first
            try:
                heartbeat.emit()
                if run_dir.is_dir():
                    incremental_manifest(run_dir)
            except Exception as exc:
                # The final caller also checks the manifest; monitoring failures must be visible.
                print(f"v17 heartbeat persistence failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def prepare(*args, **kwargs):
        request, record = original_prepare(*args, **kwargs)
        records[record["id"]] = record
        return request, record

    class DurableWriter(RunWriter):
        def __init__(self, directory, identity, expected_ids, resume=False):
            super().__init__(directory, identity, expected_ids, resume=resume)
            try:
                pe._record_registration(self.run_dir, registration)
                if self.expected_ids != registration["expected_ids"]:
                    raise ValueError("v17: evaluator ID order differs from preregistration")
                if identity["model_sha256"] != registration["model_sha256"]:
                    raise ValueError("v17: evaluator model differs from preregistration")
                if ("calibration_environment_sha256" in registration
                        and identity["environment_sha256"] != registration["calibration_environment_sha256"]):
                    raise ValueError("v17: evaluation environment differs from the calibrated environment identity")
                state["writer"] = self
                recovered = recover_finals(self, registration_hash)
                heartbeat.completed = len(self.completed)
                heartbeat.completed_output_tokens = sum(row.get("generated_tokens", 0) for row in self.completed.values())
                for sid, row in self.completed.items():
                    if row.get("seed") != sample_seed(3407, sid):
                        raise ValueError("v17: immutable sample seed mismatch")
                    for entry in row.get("messages", [{}])[0].get("content", []):
                        if entry.get("type") == "image":
                            image = (self.run_dir / entry["image"]).resolve()
                            if sha256_file(image) != entry["sha256"]:
                                raise ValueError("v17: persisted input asset changed")
                exclusive(self.run_dir / "v17_invocations" / f"{uuid.uuid4().hex}.json",
                          {"version": "v17-invocation-v1", "registration_sha256": registration_hash,
                           "identity": identity, "started_at": utc(), "recovered_ids": recovered,
                           "completed_ids_at_start": sorted(self.completed),
                           "deadline_epoch": deadline, "remaining_seconds": deadline - time.time() if deadline is not None else None})
                if set(self.completed) == set(self.expected_ids) and not (self.run_dir / "summary.json").exists():
                    # This runs before engine.run's completed-summary shortcut and before any backend load.
                    summary = finalize(self.run_dir, self.expected_ids, identity["mode"])
                    summary["v17_cpu_finalize"] = True
                    write_json(self.run_dir / "summary.json", summary)
                    render(self.run_dir)
            except BaseException:
                self.close()
                raise

        def save(self, row):
            final_path = self.run_dir / "final_raw" / (row["id"] + ".json")
            if final_path.is_file() and stable_record(row) != stable_record(record_from_final(read_json(final_path))):
                raise ValueError("v17: normal and recovery record contracts differ")
            super().save(row)
            incremental_manifest(self.run_dir)

    def backend_ready(backend):
        if backend.batch_size != slots or backend.cfg["execution"].get("scheduling") != "continuous":
            raise ValueError("v17: actual backend must continuously refill its registered slots")
        if split:
            from .v17_eval_engine import observe_engine

            observed = observe_engine(backend, registration["execution_contract"]["engine_mode"], slots)
            # KV blocks depend on the non-torch memory of each process (vLLM 0.11 profiling), so a fresh 900-run
            # process need not match the validation process block-for-block; observe_engine still refuses
            # slots > safe_slots. Every execution setting (mode, graph config, sampler, dtype, slots) must match.
            def execution(value: dict) -> dict:
                return {k: v for k, v in value.items() if k not in ENGINE_CAPACITY_KEYS}
            if execution(observed) != execution(registration["observed_engine"]):
                raise ValueError("v17: actual engine differs from final-model validation")
            capacity = observed | {"actual_blocks": backend.llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config.num_blocks}
            original_describe, original_stream = backend.describe, backend.generate_stream
            restores.extend([(backend, "describe", original_describe), (backend, "generate_stream", original_stream)])

            def describe():
                value = copy.deepcopy(original_describe())
                value["vllm"].update({"enforce_eager": observed["enforce_eager"],
                                       "compilation_config": observed["compilation_config"],
                                       "actual_kv_token_capacity": observed["actual_kv_token_capacity"]})
                value["v17_actual_engine"] = observed
                return value

            def stream(requests):
                writer = state["writer"]
                pending = reorder_request_stream(
                    requests, [sid for sid in registration["expected_ids"] if sid not in writer.completed])
                running = [read_json(path) for path in (run_dir / "inference_invocations").glob("*.json")
                           if path.name not in existing_invocations and read_json(path).get("status") == "RUNNING"]
                if len(running) != 1 or running[0].get("attempted_ids") != []:
                    raise ValueError("v17: submission stream needs its one new untouched inference invocation")
                state["submission_invocation_id"] = running[0]["id"]
                state["submission_pending"], state["submission_position"] = pending, 0
                state["audit"] = {"last_event_sha256": submission_audit(run_dir, registration)["last_event_sha256"]}
                _submission_event(run_dir, registration, audit_state=state["audit"], event="BEGIN",
                                  invocation_id=state["submission_invocation_id"],
                                  completed_ids=sorted(writer.completed), pending_ids=pending)
                yield from original_stream(requests)

            backend.describe, backend.generate_stream = describe, stream
        else:
            capacity = capacity_receipt(backend.llm.llm_engine, backend.vllm_version,
                                        max_model_len=backend.max_model_len, slots=backend.batch_size)
        exclusive(run_dir / "capacity_invocations" / f"{uuid.uuid4().hex}.json",
                  seal(capacity | {"registration_sha256": registration_hash, "measured_at": utc()}))
        heartbeat.kv_total = capacity["actual_blocks"]
        engine = backend.llm.llm_engine
        original_add, original_step, original_row = engine.add_request, engine.step, backend._row
        existing_invocations = {p.name for p in (run_dir / "inference_invocations").glob("*.json")}
        original_deadline = getattr(backend, "_check_deadline", None)
        restores.extend([(engine, "add_request", original_add), (engine, "step", original_step),
                         (backend, "_row", original_row), (backend, "_check_deadline", original_deadline)])

        def check_deadline():
            if deadline is not None and time.time() >= deadline:
                write_json(run_dir / "v17_state.json", {"status": "INTERRUPTED_RESUMABLE",
                           "reason": "CUMULATIVE_APPROVED_DEADLINE", "deadline_epoch": deadline})
                raise TimeoutError("v17: original cumulative evaluation deadline reached")

        def add(request_id, prompt, params, **kwargs):
            check_deadline()
            if request_id not in records or records[request_id]["seed"] != sample_seed(3407, request_id):
                raise ValueError("v17: submitted request is not bound to the original input and seed")
            if split:
                position = state["submission_position"]
                if position >= len(state["submission_pending"]) or request_id != state["submission_pending"][position]:
                    raise ValueError("v17: actual enqueue differs from sealed pending order")
                state["submission_position"] += 1
                _submission_event(run_dir, registration, audit_state=state["audit"], event="INTENT", id=request_id,
                                  invocation_id=state["submission_invocation_id"])
            original_add(request_id, prompt, params, **kwargs)
            if split:
                _submission_event(run_dir, registration, audit_state=state["audit"], event="ACCEPTED", id=request_id,
                                  invocation_id=state["submission_invocation_id"])
            with heartbeat.lock:
                heartbeat.active[request_id] = heartbeat.clock()
            exclusive(run_dir / "started_requests" / f"{request_id}-{uuid.uuid4().hex}.json",
                      seal({"version": "v17-started-v1", "id": request_id,
                            "record_sha256": digest(records[request_id]), "record": records[request_id],
                            "registration_sha256": registration_hash,
                            "started_at": utc(), "kv_rng_restored": False}))

        def step():
            check_deadline()
            outputs = original_step()
            with heartbeat.lock:
                heartbeat.steps += 1
                for output in outputs:
                    lengths = [len(item.token_ids) for item in getattr(output, "outputs", [])]
                    if lengths:
                        heartbeat.lengths[output.request_id] = max(lengths)
                try:
                    scheduler = engine.engine_core.engine_core.scheduler
                    heartbeat.waiting = len(scheduler.waiting)
                    for sid, request in scheduler.requests.items():
                        count = getattr(request, "num_output_tokens", None)
                        if type(count) is int:
                            heartbeat.lengths[sid] = max(heartbeat.lengths.get(sid, 0), count)
                    pool = scheduler.kv_cache_manager.block_pool
                    heartbeat.kv_used = capacity["actual_blocks"] - pool.get_num_free_blocks()
                except AttributeError:
                    heartbeat.waiting = heartbeat.kv_used = None
            metrics = _metrics(backend.llm)
            if "tokens" not in metrics:
                metrics["tokens"] = sum(heartbeat.lengths.values())
            heartbeat.observe(**metrics)
            heartbeat.emit()
            return outputs

        def row(request, item, params, output, **kwargs):
            result = original_row(request, item, params, output, **kwargs)
            sid = result["sample_id"]
            running = [read_json(p) for p in (run_dir / "inference_invocations").glob("*.json")
                       if p.name not in existing_invocations]
            invocation = [p for p in running if p.get("status") == "RUNNING" and sid in p["attempted_ids"]]
            if len(invocation) != 1:
                raise ValueError("v17: final output needs exactly one new active inference invocation")
            active_invocation = invocation[0]
            final = seal({"version": "v17-final-raw-v1", "run_id": run_dir.name,
                          "registration_sha256": registration_hash, "identity": state["writer"].identity,
                          "record": records[sid], "output": result,
                          "invocation_id": active_invocation["id"], "sealed_at": utc()})
            exclusive(run_dir / "final_raw" / (sid + ".json"), final)
            with heartbeat.lock:
                heartbeat.active.pop(sid, None)
                heartbeat.completed += 1
                heartbeat.completed_output_tokens += result["generated_tokens"]
            return result

        engine.add_request, engine.step, backend._row, backend._check_deadline = add, step, row, check_deadline
        heartbeat.emit(force=True)

    thread = threading.Thread(target=snapshot_loop, daemon=True, name="v17-evaluation-heartbeat")
    evaluation.RunWriter, evaluation.prepare_request = DurableWriter, prepare
    thread.start()
    try:
        yield backend_ready
    finally:
        finished.set()
        thread.join(timeout=2)
        evaluation.RunWriter, evaluation.prepare_request = original_writer, original_prepare
        for owner, name, original in reversed(restores):
            if original is None:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        if run_dir.is_dir():
            heartbeat.emit(force=True)
            if split:
                write_json(run_dir / "submission_audit.json", submission_audit(run_dir, registration))
            incremental_manifest(run_dir, force=True)


def run_evaluation(argv: list[str], *, prereg_path: Path, ledger: Path, calibration_path: Path,
                   preflight: Path, resume: bool = False, funding_deadline: float | None = None,
                   recovery_reserve_seconds: float = 0, engine_validation: Path | None = None,
                   training_artifacts: Path | None = None, execution_settings: Path | None = None,
                   partition_index: int = 0) -> int:
    invocation_started, started_at, invocation_id = time.monotonic(), utc(), uuid.uuid4().hex
    import yaml
    from fine_tuning.notebook_rl import proof_eval as pe

    parser = argparse.ArgumentParser(add_help=False)
    for flag in ("protocol", "hardware", "model-ref", "model-path", "artifact-root", "data-root", "base-path"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--mode", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subject")
    parser.add_argument("--require-commit")
    parser.add_argument("--job-id")
    parser.add_argument("--run-role", default="standalone")
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    known = parser.parse_args(argv)
    prereg, prereg_sha = pe.load_preregistration(prereg_path)
    if (engine_validation is None) != (training_artifacts is None):
        raise ValueError("v17: split evaluation requires both engine validation and training artifacts")
    split = engine_validation is not None
    if split != ("engine_validation" in prereg):
        raise ValueError("v17: split receipts differ from the original preregistration")
    calibration = load_calibration(calibration_path)
    if (prereg["format"] != pe.FORMAT_V12 or prereg["calibration_sha256"] != sha256_file(calibration_path)
            or prereg["evaluation_watchdog_seconds"] != calibration_timeout(calibration)):
        raise ValueError("v17: preregistration and measured calibration differ")
    events, _ = pe.read_ledger(ledger)
    starts = [event for event in events if event["event"] == "START" and event["run_id"] == known.run_id]
    settings = None
    subset_tag = "all"
    if prereg.get("training_protocol_version") == "error-note-v18b":
        from .v18b_execution import read_settings, deadline_epoch, enforce_limits, _epoch
        if execution_settings is None:
            raise ValueError("v18b evaluation: sealed execution settings required")
        settings = read_settings(execution_settings)
        if settings != prereg.get("execution_settings"):
            raise ValueError("v18b evaluation: first billing or execution settings changed on resume")
        tags = ("half-a", "half-b") if settings["evaluation_strategy"] == "two_pod" else ("all",)
        if type(partition_index) is not int or not 0 <= partition_index < len(tags):
            raise ValueError("v18b evaluation: partition index differs from the fixed strategy")
        subset_tag = tags[partition_index]
        if time.time() < _epoch(settings["billing"][partition_index]["first_billed_at"]):
            raise ValueError("v18b evaluation: pod began before its declared earliest billing time")
    elif execution_settings is not None or partition_index != 0:
        raise ValueError("v18b evaluation: execution settings and partitions are v18b-only")
    if (len(starts) != 1 or starts[0]["preregistration_sha256"] != prereg_sha
            or (starts[0]["arm"], starts[0]["variant"], starts[0]["subset"]) !=
            (pe.ARM_SINGLE, pe.FINAL_VARIANT_V8, subset_tag)
            or any(e["event"] == "FAIL" and e["run_id"] == known.run_id for e in events)):
        raise ValueError("v17: resume requires the one original, valid ledger START")
    ref, declared = pe._lora_model_ref(known.model_ref)
    version = ref["export"]["training_protocol"]["version"]
    if (version not in ("error-note-v17", "error-note-v18", "error-note-v18b")
            or ref["export"]["campaign"]["stage"] not in (("final", "partial") if version == "error-note-v18b" else ("final",))
            or any(prereg["arms"][pe.ARM_SINGLE].get(k) != v for k, v in declared.items())
            or known.model_path.resolve() != (known.model_ref.parent / "model").resolve()):
        raise ValueError("v17: evaluation requires its one declared final unmerged model and original path")
    cfg = yaml.safe_load(known.protocol.read_text(encoding="utf-8"))
    source_hash = source_sha256()
    if split:
        assert engine_validation is not None and training_artifacts is not None
        validation = split_bindings(engine_validation, training_artifacts, known.model_ref,
                                    config=cfg, hardware=yaml.safe_load(known.hardware.read_text()),
                                    code_commit=known.require_commit)
        if (validation != prereg["engine_validation"] or calibration != validation
                or sha256_file(engine_validation) != prereg["engine_validation_sha256"]
                or sha256_file(training_artifacts) != prereg["training_artifacts_file_sha256"]):
            raise ValueError("v17: split validation changed since preregistration")
        slots = validation["slots"]
    else:
        slots = SLOTS
    if version in ("error-note-v18", "error-note-v18b") and (not split or slots != 9 or prereg.get("training_protocol_version") != version
                                        or validation.get("training_protocol_version") != version):
        raise ValueError("v18: final evaluation requires its sealed protocol and nine validated slots")
    if not split and (calibration["probe"].get("config_sha256") != digest(cfg)
            or calibration.get("source_sha256") != source_hash
            or calibration.get("code_commit") != known.require_commit
            or calibration["probe"].get("hardware") != yaml.safe_load(known.hardware.read_text())):
        raise ValueError("v17: calibrated configuration/source/hardware differs from the approved evaluation")
    expected = prereg["evaluation_subsets"][subset_tag]["ids"]
    if (known.mode != "partial" or known.limit is not None or known.subject is not None
            or known.run_role != "standalone" or sorted(known.sample_id) != expected
            or len(set(known.sample_id)) != len(expected)
            or cfg["protocol_id"] != pe.FINAL_VARIANT_V8
            or digest(cfg) != prereg["protocol_variants"][pe.FINAL_VARIANT_V8]["config_digest"]):
        raise ValueError("v17: expected IDs/order/P0/B2/seed must match the original registration")
    if os.environ.get(pe.SLOTS_ENV) not in (None, str(slots)):
        raise ValueError("v17: slots differ from the immutable evaluation registration")
    if os.environ.get(pe.VLLM_INPROCESS_ENV) != "0":
        raise ValueError("v17: unmerged evaluation requires the pinned in-process engine")
    pf = read_json(preflight)
    if not split and (pf.get("model_ref_sha256") != sha256_file(known.model_ref) or pf.get("protocol_id") != cfg["protocol_id"]
            or pf.get("pass") is not True or pf.get("files", {}).get("pass") is not True
            or pf.get("engine", {}).get("pass") is not True
            or pf.get("engine", {}).get("pf5", {}).get("power") is not True):
        raise ValueError("v17: final adapter HF/vLLM preflight is missing or failed")
    resolved = {"slots": slots, "slots_source": "v17-kv-validated" if split else "v17-fixed", "gpu_name": pe.gpu_info()[0],
                "gpu_total_mib": pe.gpu_info()[1]}
    registration = pe.variant_registration(cfg["protocol_id"], resolved) | {
        "version": "v17-evaluation-registration-v1", "preregistration_sha256": prereg_sha,
        "start": starts[0], "run_id": known.run_id, "expected_ids": expected,
        "model_sha256": declared["model_sha256"], "model_ref_sha256": declared["model_ref_sha256"],
        "model_ref": str(known.model_ref.resolve()), "serving_kind": "unmerged-lora",
        "capacity_rule_sha256": digest(CAPACITY_RULE), "v17_eval_sha256": sha256_file(Path(__file__)),
        "calibration_environment_sha256": (calibration["environment_sha256"] if split else calibration["probe"]["environment_sha256"]),
        "calibration_sha256": sha256_file(calibration_path), "preflight_sha256": sha256_file(preflight)}
    if split:
        assert engine_validation is not None and training_artifacts is not None
        registration.update({"capacity_rule_sha256": digest({"version": "v17-kv-capacity-v2", "slots": slots,
                                                               "usable_fraction": .90, "max_model_len": 40960}),
                             "execution_contract": prereg["execution_contract"],
                             "engine_validation_sha256": sha256_file(engine_validation),
                             "engine_validation": str(engine_validation.resolve()),
                             "training_artifacts_file_sha256": sha256_file(training_artifacts),
                             "training_artifacts": str(training_artifacts.resolve()),
                             "training_artifacts_path": str(training_artifacts.resolve()),
                             "observed_engine": validation["observed_engine"],
                             "evaluation_commit": validation["evaluation_commit"]})
    if settings is not None:
        if validation.get("execution_settings") != settings:
            raise ValueError("v18b evaluation: engine validation settings differ")
        registration.update(execution_settings=settings, partition_index=partition_index, subset=subset_tag)
    run_dir = known.artifact_root / "runs" / known.run_id
    if split:
        submission_audit(run_dir, registration)
    extensions = [read_json(p) for p in sorted((run_dir / "deadline_extensions").glob("*.json"))]
    now = time.time()
    remaining = deadline_seconds(starts[0], calibration_timeout(calibration), extensions, now=now,
                                  funding_deadline=funding_deadline,
                                  recovery_reserve_seconds=recovery_reserve_seconds)
    bounded_deadline = now + remaining
    deadline: float | None = bounded_deadline
    if settings is not None:
        deadline = (min(bounded_deadline, deadline_epoch(settings, reserve_seconds=settings["post_eval_reserve_seconds"]))
                    if enforce_limits(settings) else None)
    if resume:
        if not run_dir.is_dir():
            raise ValueError("v17: cannot resume a missing logical run")
        pe._record_registration(run_dir, registration)
    elif run_dir.exists():
        raise FileExistsError("v17: existing logical run requires the dedicated resume entry point")
    from mmdl.evaluation import cli

    command = [a for a in argv if a != "--resume"] + (["--resume"] if resume else [])
    cwd, saved = Path.cwd(), sys.argv
    cpu_done, recovery_seconds, outcome = False, 0., "INTERRUPTED_RESUMABLE"
    try:
        os.chdir(pe.ROOT)
        sys.argv = ["mmdl.evaluation.cli", *command]
        recovery_started = time.monotonic()
        cpu_done = resume and cpu_resume_finalize(known, cfg, registration)
        recovery_seconds = time.monotonic() - recovery_started if resume else 0.
        if not cpu_done:
            if settings is not None and deadline is not None and time.time() >= deadline:
                raise TimeoutError("v18b evaluation: first-billing deadline reached before model loading")
            with evaluation_hooks(run_dir, registration, deadline) as ready:
                execution: Any = nullcontext()
                if split:
                    from .v17_eval_engine import engine_execution

                    execution = engine_execution(validation["selected_mode"], slots)
                with execution, pe.registered_variants(slots=slots, after_backend=ready, log_stats=True):
                    cli.main()
        run = pe.load_run(run_dir, prereg)
        if len(run["items"]) != len(expected):
            raise ValueError("v17: complete evaluation requires exactly the registered IDs")
        rows = {p.stem: read_json(p) for p in (run_dir / "samples").glob("*.json")}
        comparison = paired_report({sid: prereg["reference_finish_strata"][sid] for sid in expected}, rows)
        report_path = run_dir / "v17_reference_comparison.json"
        if report_path.exists() and read_json(report_path) != comparison:
            raise ValueError("v17: frozen reference comparison changed")
        if not report_path.exists():
            exclusive(report_path, comparison)
        summary = read_json(run_dir / "summary.json")
        if split:
            audit = submission_audit(run_dir, registration)
            write_json(run_dir / "submission_audit.json", audit)
            summary = summary | {"execution_contract": prereg["execution_contract"],
                                 "actual_submission_sha256": audit["actual_submission_sha256"],
                                 "submission_intent_sha256": audit["submission_intent_sha256"]}
        write_json(run_dir / "summary.json", summary | {"reference_finish_strata": comparison})
        write_json(run_dir / "v17_state.json", {"status": "FULL900_VERIFIED" if len(expected) == 900 else "PARTITION_VERIFIED", "preregistration_sha256": prereg_sha,
                   "run_id": known.run_id, "generation_repeated_for_completed_ids": False})
        outcome = "FULL900_VERIFIED" if len(expected) == 900 else "PARTITION_VERIFIED"
    except BaseException as exc:
        if run_dir.is_dir():
            status = ("INVALID_IDENTITY" if isinstance(exc, ValueError) and "identity" in str(exc).lower() else
                      "CORRUPT_ARTIFACT" if isinstance(exc, (ValueError, FileExistsError)) else "INTERRUPTED_RESUMABLE")
            outcome = status
            write_json(run_dir / "v17_state.json", {"status": status, "run_id": known.run_id,
                       "deadline_epoch": deadline, "completed_answers_immutable": True})
        raise
    finally:
        os.chdir(cwd)
        sys.argv = saved
        if run_dir.is_dir():
            exclusive(run_dir / "logical_invocations" / (invocation_id + ".json"), seal({
                "version": "v17-logical-invocation-v1", "invocation_id": invocation_id, "run_id": known.run_id,
                "registration_sha256": digest(registration), "started_at": started_at, "ended_at": utc(),
                "operation": "resume" if resume else "run", "status": outcome,
                "wall_seconds": time.monotonic() - invocation_started, "cpu_recovery_seconds": recovery_seconds,
                "cpu_finalize_only": cpu_done, "model_load_count": 0 if cpu_done else None,
                "timing_scope": "whole invocation wall including identity checks/recovery/load/generation/finalize"}))
            invocations = [checked(read_json(p)) for p in (run_dir / "logical_invocations").glob("*.json")]
            write_json(run_dir / "v17_timing_summary.json", {
                "invocations": len(invocations), "invocation_wall_seconds": sum(r["wall_seconds"] for r in invocations),
                "cpu_recovery_seconds": sum(r["cpu_recovery_seconds"] for r in invocations),
                "logical_elapsed_including_downtime_seconds": time.time() - datetime.fromisoformat(starts[0]["time"]).timestamp(),
                "first_billing_elapsed_seconds": (time.time() - _epoch(settings["first_billed_at"]) if settings is not None else None),
                "sum_of_overlapping_sample_latencies_used": False})
            incremental_manifest(run_dir, force=True)
    return 0


def probe(args: argparse.Namespace) -> dict:
    """One explicitly invoked non-benchmark diagnostic, bounded to 16 inputs and PROBE_CAP_SECONDS.

    The performance-only requests are short-answer sources, so every request is forced long (ignore_eos, the B2
    32,768 cap unchanged): eight concurrent long generations are the slowest regime of the 900 evaluation, and the
    32,768-token context coverage becomes observable. Outputs stay diagnostic only (2026-11-17)."""
    import yaml
    from PIL import Image
    from mmdl.evaluation import engine as evaluation
    from mmdl.evaluation.backends.vllm_backend import VLLMBackend
    from mmdl.evaluation.writer import _safe_id
    from mmdl.runtime.environment import collect_environment
    from fine_tuning.notebook_rl import proof_eval as pe

    manifest = checked(read_json(args.requests))
    items, disjoint = manifest.get("items", []), manifest.get("disjoint_receipt", {})
    hashes = ("main_ids_sha256", "prep_ids_sha256", "development_ids_sha256", "effect_ids_sha256",
              "benchmark_exclusion_sha256")
    if (manifest.get("version") != "v17-performance-only-v1" or not 1 <= len(items) <= 16
            or len({item["sample_id"] for item in items}) != len(items)
            or any("mmmu" in item["sample_id"].lower() or item["sample_id"].startswith("validation_") for item in items)
            or disjoint.get("status") != "VERIFIED" or disjoint.get("overlap_count") != 0
            or any(not re.fullmatch(r"[a-f0-9]{64}", str(disjoint.get(k, ""))) for k in hashes)):
        raise ValueError("v17: fixed disjoint non-benchmark performance-only manifest required")
    for item in items:
        _safe_id(item["sample_id"])
    if args.output.exists() or args.output.with_suffix(".probe.json").exists():
        raise FileExistsError("v17: calibration probe output already exists")
    if os.environ.get(pe.SLOTS_ENV) not in (None, "8"):
        raise ValueError("v17: performance probe uses eight slots")
    if os.environ.get(pe.VLLM_INPROCESS_ENV) != "0":
        raise ValueError("v17: performance probe requires the pinned in-process engine")
    cfg = yaml.safe_load(pe.VARIANTS[pe.FINAL_VARIANT_V8][0].read_text())
    hw = yaml.safe_load(args.hardware.read_text())
    ref = read_json(args.model_ref)
    if ref.get("kind") not in ("base", pe.LORA_KIND):
        raise ValueError("v17: performance probe uses pinned BF16 base or unmerged serving")
    environment = collect_environment()
    if not environment["torch"]["bf16_probe"]["passed"]:
        raise ValueError("v17: probe requires actual BF16 operation")
    diagnostics = args.output.parent / (args.output.stem + "-diagnostics")
    diagnostics.mkdir(parents=True, exist_ok=False)
    heartbeat = Heartbeat(diagnostics, expected_count=len(items))
    observations: list[tuple[float, int]] = []
    prepared_inputs, active_waiting_samples = [], []
    tail_seconds = save_seconds = 0.
    max_context, complete_tokens = 0, 0
    started = time.monotonic()
    with pe.registered_variants(slots=SLOTS, log_stats=True):
        evaluation.verify_model(args.model_path, ref)
        backend = VLLMBackend(str(args.model_path), cfg, hw, ref["kind"], str(args.base_path))
        kv = capacity_receipt(backend.llm.llm_engine, backend.vllm_version)
        load_seconds = time.monotonic() - started
        engine = backend.llm.llm_engine
        original_step, original_prepare = engine.step, backend._prepare_request
        run_started = time.monotonic()
        observations.append((0., 0))

        def prepare(request, **kwargs):
            before = time.monotonic()
            item, prompt, params = original_prepare(request, **kwargs)
            params = copy.copy(params)
            params.ignore_eos = True  # forced-long diagnostic; never scored
            measured = {"sample_id": request["sample_id"], "input_tokens": item["input_tokens"],
                        "image_grid_thw": item["image_grid_thw"], "input_sha256": item["input_sha256"],
                        "prepare_wall_seconds": time.monotonic() - before}
            prepared_inputs.append(measured)
            exclusive(diagnostics / "inputs" / (request["sample_id"] + ".json"), measured)
            return item, prompt, params

        def step():
            nonlocal max_context, tail_seconds
            before = time.monotonic()
            scheduler = engine.engine_core.engine_core.scheduler
            active_before = len(scheduler.running)
            output = original_step()
            if active_before == 1:
                tail_seconds += time.monotonic() - before
            heartbeat.steps += 1
            scheduler = engine.engine_core.engine_core.scheduler
            for request in scheduler.requests.values():
                max_context = max(max_context, request.num_tokens)
            metrics = _metrics(backend.llm)
            heartbeat.observe(**metrics)
            elapsed = time.monotonic() - run_started
            if elapsed - observations[-1][0] >= 10:
                observations.append((elapsed, heartbeat.tokens))
                active_waiting_samples.append({"elapsed_seconds": elapsed, "running": len(scheduler.running),
                                               "waiting": len(scheduler.waiting)})
                heartbeat.emit(force=True)
            return output

        def deadline():
            if time.monotonic() - run_started >= PROBE_CAP_SECONDS:
                for sid, request in engine.engine_core.engine_core.scheduler.requests.items():
                    token_ids = list(request.output_token_ids)
                    exclusive(diagnostics / "partial" / (sid + ".json"), {
                        "sample_id": sid, "generated_token_ids": token_ids,
                        "raw_response": backend.processor.tokenizer.decode(token_ids, skip_special_tokens=True,
                                                                             clean_up_tokenization_spaces=False),
                        "status": "DIAGNOSTIC_PARTIAL_NOT_SCORED", "requests_sha256": sha256_file(args.requests)})
                raise TimeoutError("v17: performance-only probe reached its diagnostic cap")

        def requests():
            for item in items:
                images = []
                for image in item["images"]:
                    path = Path(image["path"])
                    if not path.is_absolute():
                        path = args.requests.parent / path
                    if sha256_file(path) != image["sha256"]:
                        raise ValueError("v17: performance input image changed")
                    with Image.open(path) as loaded:
                        images.append(loaded.copy())
                if len(item["messages"]) != 1 or item["messages"][0].get("role") != "user":
                    raise ValueError("v17: performance-only input must be one P0 user turn")
                yield {"sample_id": item["sample_id"], "messages": item["messages"], "images": images,
                       "seed": sample_seed(3407, item["sample_id"])}

        engine.step, backend._check_deadline, backend._prepare_request = step, deadline, prepare
        status = "COMPLETED"
        try:
            for output in backend.generate_stream(requests()):
                complete_tokens += output["generated_tokens"]
                max_context = max(max_context, output["input_tokens"] + output["generated_tokens"])
                before_save = time.monotonic()
                exclusive(diagnostics / "outputs" / (output["sample_id"] + ".json"), output)
                save_seconds += time.monotonic() - before_save
        except TimeoutError:
            status = "DIAGNOSTIC_CAP_PARTIAL"
        finally:
            engine.step, backend._prepare_request = original_step, original_prepare
        wall = min(float(PROBE_CAP_SECONDS), time.monotonic() - run_started)
        final_metrics = _metrics(backend.llm)
        tokens = final_metrics.get("tokens", complete_tokens)
        observations.append((wall, tokens))
        windows = []
        for begin, count in observations:
            later = next(((end, total) for end, total in observations if end - begin >= 60), None)
            if later:
                windows.append((later[1] - count) / (later[0] - begin))
        result = seal({"version": "v17-eval-probe-v1", "status": status, "slots": SLOTS,
                       "items": len(items), "wall_seconds": wall, "generated_tokens": tokens,
                       "sixty_second_rates": windows, "max_observed_context_tokens": max_context,
                       "cumulative_preemptions": final_metrics.get("preemptions"),
                       "load_finalize_allowance_seconds": load_seconds + save_seconds,
                       "long_tail_allowance_seconds": args.long_tail_allowance_seconds,
                       "long_tail_basis": args.long_tail_basis, "dtype": "bfloat16", "backend": "vllm-0.11.0",
                       "config_sha256": digest(cfg), "hardware": hw,
                       "environment_sha256": digest(evaluation.environment_identity(environment)),
                       "source_sha256": source_sha256(),
                       "code_commit": evaluation.git_commit(pe.ROOT, args.require_commit),
                       "requests_sha256": sha256_file(args.requests), "disjoint_receipt_sha256": digest(disjoint),
                       "capacity": kv, "observations": observations, "prepared_inputs": prepared_inputs,
                       "active_waiting_samples": active_waiting_samples, "single_active_step_seconds": tail_seconds,
                       "save_wall_seconds": save_seconds, "model_load_seconds": load_seconds,
                       "timing_scope": "aggregate wall; single-active step wall is observed tail, not summed request latency",
                       "forced_long_ignore_eos": True, "probe_cap_seconds": PROBE_CAP_SECONDS,
                       "diagnostic_only": True, "used_for_training_or_answer_selection": False})
    exclusive(args.output.with_suffix(".probe.json"), result)
    calibrated = calibration_receipt(result)
    exclusive(args.output, calibrated)
    return calibrated


def extend_deadline(prereg: Path, ledger: Path, run_dir: Path, seconds: int, authorization: str) -> dict:
    from fine_tuning.notebook_rl import proof_eval as pe

    registration, sha = pe.load_preregistration(prereg)
    events, _ = pe.read_ledger(ledger)
    starts = [e for e in events if e["event"] == "START" and e["run_id"] == run_dir.name]
    if (registration["format"] != pe.FORMAT_V12 or len(starts) != 1
            or starts[0]["preregistration_sha256"] != sha or seconds <= 0 or not authorization.strip()):
        raise ValueError("v17: extension needs the original START and explicit user authorization")
    old = [read_json(p) for p in sorted((run_dir / "deadline_extensions").glob("*.json"))]
    deadline = deadline_seconds(starts[0], registration["evaluation_watchdog_seconds"], old, now=0)
    value = seal({"version": "v17-eval-extension-v1", "run_id": run_dir.name,
                   "preregistration_sha256": sha, "old_deadline": deadline, "new_deadline": deadline + seconds,
                   "previous": old[-1]["sha256"] if old else "0" * 64,
                   "confirmed_by": "user", "authorization": authorization, "recorded_at": utc()})
    exclusive(run_dir / "deadline_extensions" / f"{len(old):06d}.json", value)
    return value


def evaluation_decision(campaign_output: Path, authorization: str, output: Path) -> dict:
    """Record the user's standing run instruction as the decision for a reviewed admission (2026-11-17).

    PROCEED only for EVAL_REVIEW_REQUIRED or EVALUATION_CANDIDATE; DEGRADED (paired external degradation) is never
    pre-authorized and raises, so the chain stops before export. Idempotent on a resumed chain: an existing receipt
    must name the same admission and plan."""
    from .prep_v16b import read_sealed, seal as seal_file

    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError("v17 decision: non-empty user authorization text required")
    plan = read_sealed(campaign_output / "campaign-plan.json")
    admission = read_sealed(campaign_output / "evaluation-admission.json")
    if admission.get("plan_sha256") != plan["sha256"]:
        raise ValueError("v17 decision: admission belongs to another plan")
    gate = admission["decision"]
    status = gate["status"]
    if status == "DEGRADED":
        raise ValueError("v17 decision: DEGRADED admission is not pre-authorized; ask the user")
    if status not in ("EVAL_REVIEW_REQUIRED", "EVALUATION_CANDIDATE"):
        raise ValueError(f"v17 decision: unknown admission status {status}")
    if output.exists():
        existing = read_sealed(output)
        if existing.get("admission_sha256") != admission["sha256"] or existing.get("plan_sha256") != plan["sha256"]:
            raise ValueError("v17 decision: an existing receipt names another admission")
        return existing
    development = (gate.get("development") or {}).get("overall")
    effect = gate.get("effect") or {}
    return seal_file(output, {"version": "v17-evaluation-decision-v1", "decision": "PROCEED",
        "authorized_by": "user", "pre_authorized": True, "authorization": authorization,
        "admission_status": status, "admission_sha256": admission["sha256"], "plan_sha256": plan["sha256"],
        "development_overall": development, "effect_eligible": effect.get("eligible") is True,
        "effect_overall": effect.get("overall"), "recorded_at": utc()})


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passed = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        run = sub.add_parser(name)
        for flag in ("prereg", "ledger", "calibration", "preflight"):
            run.add_argument("--" + flag, type=Path, required=True)
        run.add_argument("--engine-validation", type=Path)
        run.add_argument("--training-artifacts", type=Path)
        run.add_argument("--execution-settings", type=Path)
        run.add_argument("--partition-index", type=int, default=0)
        run.add_argument("--funding-deadline-epoch", type=float)
        run.add_argument("--recovery-reserve-seconds", type=float, default=0)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--probe", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    probe_cli = sub.add_parser("probe")
    for flag in ("requests", "model-path", "model-ref", "base-path", "hardware", "output"):
        probe_cli.add_argument("--" + flag, type=Path, required=True)
    probe_cli.add_argument("--long-tail-allowance-seconds", type=float, required=True)
    probe_cli.add_argument("--long-tail-basis", required=True)
    probe_cli.add_argument("--require-commit", required=True)
    doc = sub.add_parser("doctor")
    doc.add_argument("--input", type=Path, required=True)
    extension = sub.add_parser("extend-deadline")
    for flag in ("prereg", "ledger", "run-dir"):
        extension.add_argument("--" + flag, type=Path, required=True)
    extension.add_argument("--seconds", type=int, required=True)
    extension.add_argument("--authorization", required=True)
    comparison = sub.add_parser("paired")
    for flag in ("reference", "candidate", "output"):
        comparison.add_argument("--" + flag, type=Path, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--development", type=Path, required=True)
    gate.add_argument("--effect", type=Path)
    gate.add_argument("--output", type=Path, required=True)
    pack = sub.add_parser("pack")
    for flag in ("run-dir", "prereg", "ledger", "model-ref", "calibration", "preflight", "out-dir"):
        pack.add_argument("--" + flag, type=Path, required=True)
    pack.add_argument("--name", required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--campaign-output", type=Path, required=True)
    decide.add_argument("--authorization", required=True)
    decide.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "decide":
        try:
            value = evaluation_decision(args.campaign_output, args.authorization, args.output)
        except ValueError as exc:
            print(json.dumps({"status": "REFUSED", "reason": str(exc)}))
            return 3
        print(json.dumps({"status": value["decision"], "admission_status": value["admission_status"],
                          "sha256": value["sha256"]}))
        return 0
    if args.command in ("run", "resume"):
        return run_evaluation(passed, prereg_path=args.prereg, ledger=args.ledger,
                              calibration_path=args.calibration, preflight=args.preflight,
                              resume=args.command == "resume", funding_deadline=args.funding_deadline_epoch,
                              recovery_reserve_seconds=args.recovery_reserve_seconds,
                              engine_validation=args.engine_validation, training_artifacts=args.training_artifacts,
                              execution_settings=args.execution_settings, partition_index=args.partition_index)
    if args.command == "doctor":
        print(json.dumps(doctor_check(read_json(args.input))))
    elif args.command in ("calibrate", "probe"):
        if args.command == "calibrate":
            value = calibration_receipt(read_json(args.probe))
            exclusive(args.output, value)
        else:
            value = probe(args)
        print(json.dumps({k: value[k] for k in ("status", "evaluation_watchdog_seconds", "sha256")}))
        return 0 if value["status"] == "CALIBRATED" else 3
    elif args.command == "extend-deadline":
        print(json.dumps(extend_deadline(args.prereg, args.ledger, args.run_dir, args.seconds, args.authorization)))
    elif args.command == "paired":
        def rows(path):
            values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len({r["id"] for r in values}) != len(values):
                raise ValueError("v17: duplicate paired IDs")
            return {r["id"]: r for r in values}
        exclusive(args.output, paired_report(rows(args.reference), rows(args.candidate)))
    elif args.command == "gate":
        value = evaluation_gate(read_json(args.development), read_json(args.effect) if args.effect else None)
        exclusive(args.output, value)
        return 0 if value["status"] == "EVALUATION_CANDIDATE" else 3
    elif args.command == "pack":
        from fine_tuning.notebook_rl.pod.collect import pack_eval_resume

        print(json.dumps(pack_eval_resume(args.run_dir, args.prereg, args.ledger, args.model_ref,
                                          args.calibration, args.preflight, args.out_dir, args.name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
