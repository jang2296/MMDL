"""CPU-only evaluation policy: fixed disjoint shards and live measurements."""
from __future__ import annotations

import math
import hashlib
import argparse
import json
from pathlib import Path

from .v17_eval import seal

PARTITION_RULE = "sort by sha256(v18b-eval-partition:<id>), alternate into half-a/half-b"


def partitions(ids: list[str], settings: dict) -> dict[str, list[str]]:
    from .v18b_execution import validate_settings

    settings = validate_settings(settings)
    if len(ids) != 900 or len(set(ids)) != 900:
        raise ValueError("v18b evaluation: exactly 900 unique registered IDs required")
    if settings["evaluation_strategy"] != "two_pod":
        return {"all": sorted(ids)}
    ordered = sorted(ids, key=lambda sid: hashlib.sha256(("v18b-eval-partition:" + sid).encode()).hexdigest())
    return {name: sorted(ordered[index::2]) for index, name in enumerate(("half-a", "half-b"))}


def live_calibration(observed: dict, settings: dict) -> dict:
    from .v18b_execution import validate_settings

    settings = validate_settings(settings)
    if observed.get("slots") != 9 or observed.get("safe_slots", 0) < 9:
        raise ValueError("v18b evaluation: nine KV-safe slots required")
    return seal({"version": "v18b-eval-live-calibration-v1", "status": "LIVE_MONITORING_REQUIRED",
                 "throughput_source": "evaluation_live_only", "estimated_evaluation_seconds": None,
                 "evaluation_watchdog_seconds": settings["total_timeout_seconds"],
                 "execution_settings": settings, "forced_long_generation": False,
                 "capacity_sha256": observed["sha256"], "statistical_upper_bound": False})


def progress(*, elapsed: float, tokens: int, completed: int, expected: int,
             historical_tokens: int, remaining_seconds: float, preemptions: int | None,
             completed_output_tokens: int = 0, active_tokens: int = 0) -> dict:
    if (not math.isfinite(elapsed) or elapsed < 0 or type(tokens) is not int or tokens < 0
            or type(completed) is not int or type(expected) is not int or expected <= 0
            or type(completed_output_tokens) is not int or completed_output_tokens < 0
            or type(active_tokens) is not int or active_tokens < 0
            or not 0 <= completed <= expected or not math.isfinite(remaining_seconds)):
        raise ValueError("v18b evaluation: invalid live measurement")
    rate = tokens / elapsed if elapsed > 0 and tokens > 0 else None
    measured_lengths = completed > 0 and completed_output_tokens > 0
    mean_length = completed_output_tokens / completed if measured_lengths else historical_tokens / expected
    remaining_tokens = max(0, mean_length * (expected - completed) - active_tokens)
    eta = remaining_tokens / rate if rate else None
    return {"generated_tokens_per_second": rate, "remaining_seconds": remaining_seconds,
            "estimated_remaining_seconds": eta,
            "eta_basis": "actual completed output lengths / actual aggregate rate" if measured_lengths else
                         "historical B2 mean output length / actual aggregate rate",
            "estimate_exceeds_remaining_time": eta is not None and eta > remaining_seconds,
            "preemptions_observed": preemptions, "automatic_retry": False,
            "completed_count": completed, "expected_count": expected}


def merge(preregs: list[Path], ledgers: list[Path], runs: list[Path], output: Path) -> dict:
    """Re-score both original shards; only their exhaustive union is a 900-item result."""
    from fine_tuning.notebook_rl import proof_eval as pe
    from mmdl.runtime.artifacts import sha256_file, write_json
    from .v17_eval import paired_report, submission_audit

    if any(len(group) != 2 for group in (preregs, ledgers, runs)):
        raise ValueError("v18b merge: exactly two original partition runs required")
    loaded = [pe.load_preregistration(path)[0] for path in preregs]
    first = loaded[0]
    if first.get("training_protocol_version") != "error-note-v18b":
        raise ValueError("v18b merge: only v18b partitions may be combined")
    settings = first["execution_settings"]
    all_ids = sorted(first["b2_reference"]["wrong_ids"] + first["b2_reference"]["right_ids"])
    pieces = partitions(all_ids, settings)
    if set(pieces) != {"half-a", "half-b"}:
        raise ValueError("v18b merge: two-pod policy was not authorized")
    rows: dict[str, dict] = {}
    tags = set()
    sources = []
    shared_keys = ("model_sha256", "model_ref_sha256", "checkpoint_metadata_sha256", "train_config_sha256")
    source_identity = None
    for prereg_path, ledger_path, run_dir, prereg in zip(preregs, ledgers, runs, loaded, strict=True):
        if (prereg.get("training_protocol_version") != "error-note-v18b"
                or prereg["execution_settings"] != settings or prereg["evaluation_subsets"] != first["evaluation_subsets"]
                or prereg.get("training_selection") != first.get("training_selection")
                or prereg["reference_finish_strata"] != first["reference_finish_strata"]
                or any(prereg["arms"][pe.ARM_SINGLE][key] != first["arms"][pe.ARM_SINGLE][key] for key in shared_keys)):
            raise ValueError("v18b merge: model, partition, source or billing settings differ")
        events, _ = pe.read_ledger(ledger_path)
        starts = [event for event in events if event["event"] == "START"]
        if (len(starts) != 1 or starts[0]["run_id"] != run_dir.name
                or starts[0]["preregistration_sha256"] != sha256_file(prereg_path)
                or any(event["event"] == "FAIL" for event in events)):
            raise ValueError("v18b merge: one original valid START required per partition")
        run = pe.load_run(run_dir, prereg)
        tag = run["subset"]
        if tag not in pieces or tag in tags or sorted(run["items"]) != pieces[tag] or starts[0]["subset"] != tag:
            raise ValueError("v18b merge: repeated, missing or overlapping partition")
        index = 0 if tag == "half-a" else 1
        if run_dir.name != settings["run_id"] + f"-part-{index}":
            raise ValueError("v18b merge: partition run ID was replaced")
        registration = run["registration"]
        if (registration.get("execution_settings") != settings or registration.get("partition_index") != index
                or registration.get("preregistration_sha256") != sha256_file(prereg_path)
                or registration.get("engine_validation_sha256") != prereg["engine_validation_sha256"]):
            raise ValueError("v18b merge: registered billing/engine/partition identity differs")
        submission_audit(run_dir, registration)
        identity = {key: run["identity"][key] for key in ("model_sha256", "protocol_sha256", "code_sha256")}
        identity["mode"] = prereg["engine_validation"]["selected_mode"]
        if source_identity is not None and identity != source_identity:
            raise ValueError("v18b merge: partition evaluation code, model or execution mode differs")
        source_identity = identity
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError("v18b merge: duplicate answer ID")
            rows[row["id"]] = row
        tags.add(tag)
        sources.append({"prereg": str(prereg_path), "prereg_sha256": sha256_file(prereg_path),
                        "ledger": str(ledger_path), "ledger_sha256": sha256_file(ledger_path),
                        "run_dir": str(run_dir), "predictions_sha256": sha256_file(run_dir / "predictions.jsonl")})
    if tags != set(pieces) or sorted(rows) != all_ids:
        raise ValueError("v18b merge: incomplete 900-item union")
    report = seal({"version": "v18b-evaluation-merge-v1", "status": "FULL900_VERIFIED",
                   "count": 900, "correct": sum(row["correct"] for row in rows.values()),
                   "execution_settings": settings, "training_selection": first.get("training_selection"),
                   "partition_rule": PARTITION_RULE, "sources": sources, "single_answer_per_item": True,
                   "reference_finish_strata": paired_report(first["reference_finish_strata"], rows),
                   "supplementary_scoring": "separate; permitted only after this full-900 receipt"})
    predictions = "".join(json.dumps(rows[sid], ensure_ascii=False, sort_keys=True) + "\n" for sid in all_ids)
    if output.exists():
        if (not output.is_symlink() and (output / "summary.json").is_file() and (output / "predictions.jsonl").is_file()
                and json.loads((output / "summary.json").read_text()) == report
                and (output / "predictions.jsonl").read_text(encoding="utf-8") == predictions):
            return report
        raise ValueError("v18b merge: existing union differs from original partition evidence")
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions.jsonl").write_text(predictions, encoding="utf-8")
    write_json(output / "summary.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("merge")
    for flag in ("prereg", "ledger", "run-dir"):
        command.add_argument("--" + flag, type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge(args.prereg, args.ledger, args.run_dir, args.output)
    print(json.dumps({key: result[key] for key in ("status", "count", "correct", "sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
