"""Read-only saved-response validation, preview, and paired team-score reports."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import SUBJECTS


SUITE_COUNTS = {"mmmu-val": 900, "mmmu-pro": 1730}


def saved_run(source: Path, suite: str, *, preview: bool = False) -> tuple[list[dict], dict]:
    """Read closed records without changing the generation registration or responses."""
    source = Path(source).resolve()
    manifest = read_json(source / "run_manifest.json")
    identity = manifest.get("identity", {})
    expected = manifest["expected_ids"]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Source registration has empty or duplicate IDs")
    config = yaml.safe_load((source / "resolved_eval_config.yaml").read_text(encoding="utf-8"))
    registered_suite = identity.get("suite") or (
        "mmmu-pro" if config.get("dataset", {}).get("id") == "MMMU/MMMU_Pro" else "mmmu-val")
    if registered_suite != suite:
        raise ValueError("Requested suite differs from the source registration")
    rehearsal = bool(identity.get("rehearsal") or config.get("rehearsal"))
    if not rehearsal:
        if len(expected) != SUITE_COUNTS[suite]:
            raise ValueError(f"{suite} requires {SUITE_COUNTS[suite]} registered IDs")
        if suite == "mmmu-val" and set(expected) != {
            f"validation_{subject}_{index}" for subject in SUBJECTS for index in range(1, 31)
        }:
            raise ValueError("MMMU registration differs from the fixed validation IDs")
    mapping = identity.get("id_subjects")
    if suite == "mmmu-pro" and not isinstance(mapping, dict):
        coverage_path = source / "dataset_coverage.json"
        if coverage_path.exists():
            mapping = read_json(coverage_path).get("id_subjects")
        if not isinstance(mapping, dict):
            raise ValueError("Pro source requires its registered ID-to-subject mapping")
    paths = sorted((source / "samples").glob("*.json"))
    rows = [read_json(path) for path in paths]
    if any(path.name != f"{row.get('id')}.json" for path, row in zip(paths, rows, strict=True)):
        raise ValueError("Sample filename and ID differ")
    indexed = {}
    for row in rows:
        sample_id = row.get("id")
        if sample_id in indexed or sample_id not in expected:
            raise ValueError("Duplicate or unregistered source response")
        if row.get("_record_sha256") != digest({k: v for k, v in row.items() if k != "_record_sha256"}):
            raise ValueError(f"Source record integrity mismatch: {sample_id}")
        if row.get("status", "COMPLETED") != "COMPLETED" or type(row.get("correct")) is not bool:
            raise ValueError(f"Source record is not a completed scored response: {sample_id}")
        if mapping is not None and mapping.get(sample_id) != row.get("subject"):
            raise ValueError(f"Registered subject differs: {sample_id}")
        if suite == "mmmu-val" and not rehearsal and not sample_id.startswith(f"validation_{row['subject']}_"):
            raise ValueError(f"MMMU subject differs from its ID: {sample_id}")
        indexed[sample_id] = row
    if not preview:
        summary = read_json(source / "summary.json")
        if (set(indexed) != set(expected) or summary.get("status") not in {"COMPLETE", "PARTIAL", "SMOKE"}
                or any(summary.get(key) != len(expected)
                       for key in ("expected_count", "completed_count", "denominator"))
                or summary.get("missing_count") != 0 or summary.get("unresolved_failure_count") != 0
                or summary.get("correct") != sum(row["correct"] for row in rows)):
            raise ValueError("Final scoring requires the complete registered ID set")
        predictions = [json.loads(line) for line in (source / "predictions.jsonl").read_text(
            encoding="utf-8").splitlines()]
        if len(predictions) != len(indexed) or {r["id"]: r for r in predictions} != indexed:
            raise ValueError("Saved predictions differ from the closed sample records")
    rows = [indexed[sample_id] for sample_id in expected if sample_id in indexed]
    return rows, {"suite": suite, "expected_ids": expected, "id_subjects": mapping,
                  "rehearsal": rehearsal, "reportable": not preview and not rehearsal,
                  "source_run": str(source), "source_manifest_sha256": sha256_file(source / "run_manifest.json"),
                  "source_config_sha256": sha256_file(source / "resolved_eval_config.yaml")}


def score_summary(rows: list[dict], identity: dict, merged: list[dict] | None = None) -> dict:
    from .CompassVerifier_checker.selection import TARGET_CATEGORIES, categorize, replay_v8

    replay_v8(rows)
    indexed = {row["id"]: row for row in rows}
    final = indexed if merged is None else {row["id"]: row for row in merged}
    if len(final) != len(rows) or set(final) != set(indexed):
        raise ValueError("Supplemented results differ from the source IDs")
    subjects: dict[str, dict[str, Any]] = {}
    for row in rows:
        scored = final[row["id"]]
        after = row["correct"] if merged is None else scored["final_correct"]
        if merged is not None and (scored["v8_correct"] != row["correct"]
                                  or scored["subject"] != row["subject"]):
            raise ValueError("Supplemented result differs from its V8 source")
        if type(after) is not bool or (row["correct"] and not after):
            raise ValueError("Supplemented result violates one-directional scoring")
        subject = subjects.setdefault(row["subject"], {"total": 0, "v8_correct": 0, "final_correct": 0})
        subject["total"] += 1
        subject["v8_correct"] += row["correct"]
        subject["final_correct"] += after
    for subject in subjects.values():
        subject["v8_accuracy"] = subject["v8_correct"] / subject["total"]
        subject["final_accuracy"] = subject["final_correct"] / subject["total"]
    n = len(rows)
    v8 = sum(row["correct"] for row in rows)
    final_correct = sum(s["final_correct"] for s in subjects.values())
    return {**identity, "scoring": "team V8 + CompassVerifier" if merged is not None else "team V8",
            "denominator": n, "registered_count": len(identity["expected_ids"]),
            "missing_ids": [sid for sid in identity["expected_ids"] if sid not in indexed],
            "v8_correct": v8, "final_correct": final_correct,
            "v8_accuracy": v8 / n if n else None, "final_accuracy": final_correct / n if n else None,
            "v8_macro_average": math.fsum(s["v8_accuracy"] for s in subjects.values()) / len(subjects) if subjects else None,
            "final_macro_average": math.fsum(s["final_accuracy"] for s in subjects.values()) / len(subjects) if subjects else None,
            "supplement_eligible": sum(categorize(row) in TARGET_CATEGORIES for row in rows),
            "supplement_accepted": final_correct - v8,
            "length_count": sum(row.get("finish_reason") == "length" for row in rows),
            "parse_status_counts": dict(Counter(row["parse_status"] for row in rows)),
            "subject_scores": dict(sorted(subjects.items()))}


def replay_checker(predictions: Path, checker_run: Path) -> tuple[list[dict], dict]:
    """Rebuild and verify saved verdicts; never load judge weights or change artifacts."""
    from argparse import Namespace
    from .CompassVerifier_checker.cli import replay_report

    return replay_report(Namespace(predictions=[Path(predictions)],
                                  targets=Path(checker_run) / "targets.jsonl",
                                  verdicts=Path(checker_run) / "verdicts.jsonl"))


def paired_report(base_run: Path, finetuned_run: Path, suite: str, *,
                  base_checker: Path | None = None, finetuned_checker: Path | None = None) -> dict:
    left, left_identity = saved_run(base_run, suite)
    right, right_identity = saved_run(finetuned_run, suite)
    before, after = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    if set(before) != set(after):
        raise ValueError("Paired runs must have exactly the same registered IDs")
    for sample_id in before:
        if any(before[sample_id].get(key) != after[sample_id].get(key)
               for key in ("subject", "question_type", "question", "options", "answer", "seed")):
            raise ValueError(f"Paired problem/gold/seed mismatch: {sample_id}")
    if bool(base_checker) != bool(finetuned_checker):
        raise ValueError("Provide both saved checker runs for supplemented paired comparison")
    from mmdl.evaluation.parsers import score_with_parser

    def v8_correct(indexed):
        return {sid: score_with_parser(row["raw_response"], row["question_type"], row["options"],
                                       row["answer"], "team-final-answer-v8", row.get("finish_reason"),
                                       question=row.get("question", ""))["correct"]
                for sid, row in indexed.items()}

    scores = {"v8": (v8_correct(before), v8_correct(after))}
    if base_checker is not None:
        assert finetuned_checker is not None
        base_merged, base_summary = replay_checker(Path(base_run) / "predictions.jsonl", base_checker)
        ft_merged, ft_summary = replay_checker(Path(finetuned_run) / "predictions.jsonl", finetuned_checker)
        if base_summary["reportable"] != ft_summary["reportable"] or base_summary["identity"]["judge"] != ft_summary["identity"]["judge"]:
            raise ValueError("Paired checker profiles differ")
        if {r["id"] for r in base_merged} != set(before) or {r["id"] for r in ft_merged} != set(before):
            raise ValueError("Paired checker IDs differ from their source runs")
        scores["v8+checker"] = ({r["id"]: r["final_correct"] for r in base_merged},
                                {r["id"]: r["final_correct"] for r in ft_merged})
    result = {}
    for label, (original, trained) in scores.items():
        counts = Counter((original[sid], trained[sid]) for sid in before)
        gains, losses = counts[(False, True)], counts[(True, False)]
        discordant = gains + losses
        probability = min(1.0, math.fsum(math.comb(discordant, k) / (2 ** discordant)
                                       for k in range(min(gains, losses) + 1)) * 2)
        result[label] = {"both_correct": counts[(True, True)], "finetuned_only": gains,
                         "base_only": losses, "both_wrong": counts[(False, False)],
                         "base_correct": sum(original.values()), "finetuned_correct": sum(trained.values()),
                         "difference": gains - losses, "mcnemar_exact_two_sided_p": probability}
        if sum(counts.values()) != len(before):
            raise ValueError("Paired counts do not sum to the denominator")
    return {"suite": suite, "denominator": len(before), "scores": result,
            "reportable": left_identity["reportable"] and right_identity["reportable"]
            and (base_checker is None or base_summary["reportable"]),
            "base_source": left_identity, "finetuned_source": right_identity}


def write_report(directory: Path, report: dict) -> Path:
    directory = Path(directory).expanduser().resolve()
    root = Path(__file__).resolve().parents[3]
    if directory.is_relative_to(root):
        raise ValueError("Benchmark reports must be outside the repository")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "summary.json", report)
    return directory
