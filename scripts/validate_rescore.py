#!/usr/bin/env python3
"""Read-only integrity and replay check for a V8 rescore artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mmdl.evaluation.final_answer_parser_v8 import COMPARISON_POLICY
from mmdl.evaluation.parsers import score_with_parser, scoring_source_files
from mmdl.runtime.artifacts import digest, read_json, sha256_file
from mmdl.runtime.contracts import SUBJECTS


PARSER = "team-final-answer-v8"
SCORE_FIELDS = {
    "parsed_answer", "parse_status", "correct", "extraction_rule",
    "extraction_evidence", "extraction_status",
}
SUMMARY_COUNTS = ("expected_count", "completed_count", "denominator")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _records(run_dir: Path) -> dict[str, dict[str, Any]]:
    paths = sorted((run_dir / "samples").glob("*.json"))
    _require(len(paths) == 900, f"Expected 900 sample records in {run_dir}")
    result = {}
    for path in paths:
        row = read_json(path)
        _require(isinstance(row, dict), f"Invalid sample record: {path}")
        sample_id = row.get("id")
        _require(isinstance(sample_id, str) and path.name == f"{sample_id}.json",
                 f"Sample filename/ID mismatch: {path}")
        _require(sample_id not in result, f"Duplicate sample ID: {sample_id}")
        subject = row.get("subject")
        mapped_subject = next((name for name in sorted(SUBJECTS, key=len, reverse=True)
                               if sample_id.startswith(f"validation_{name}_")), None)
        _require(mapped_subject == subject, f"Sample ID/subject mismatch: {sample_id}")
        index = sample_id.rsplit("_", 1)[-1]
        _require(index.isdecimal() and 1 <= int(index) <= 30,
                 f"Sample ID is outside the expected 30-item range: {sample_id}")
        _require(row.get("status") == "COMPLETED", f"Sample is not completed: {sample_id}")
        _require(type(row.get("correct")) is bool, f"Sample correctness is not boolean: {sample_id}")
        _require(row.get("_record_sha256") == digest({k: v for k, v in row.items() if k != "_record_sha256"}),
                 f"Sample record hash mismatch: {sample_id}")
        result[sample_id] = row
    _require(Counter(row.get("subject") for row in result.values()) == Counter({s: 30 for s in SUBJECTS}),
             f"Expected 30 records per MMMU subject in {run_dir}")
    return result


def _predictions(path: Path, records: dict[str, dict[str, Any]]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(len(rows) == len(records), f"Prediction row count mismatch: {path}")
    indexed = {row.get("id"): row for row in rows if isinstance(row, dict)}
    _require(len(indexed) == len(rows) and set(indexed) == set(records),
             f"Prediction IDs differ from sample records: {path}")
    _require(indexed == records, f"predictions.jsonl differs from sample records: {path}")


def _summary(run_dir: Path, records: dict[str, dict[str, Any]]) -> int:
    summary = read_json(run_dir / "summary.json")
    _require(summary.get("status") == "COMPLETE" and summary.get("mode") == "full",
             f"Run is not COMPLETE/full: {run_dir}")
    _require(all(summary.get(key) == 900 for key in SUMMARY_COUNTS)
             and summary.get("missing_count") == 0 and summary.get("unresolved_failure_count") == 0,
             f"Incomplete summary counts: {run_dir}")
    correct = sum(row.get("correct") is True for row in records.values())
    _require(summary.get("correct") == correct and summary.get("missing_ids") == [],
             f"Summary correctness differs from sample rows: {run_dir}")
    for key in ("accuracy", "macro_average"):
        _require(math.isclose(float(summary.get(key, -1)), correct / 900, rel_tol=0, abs_tol=1e-12),
                 f"Summary {key} differs from sample rows: {run_dir}")

    subject_scores = summary.get("subject_scores")
    _require(isinstance(subject_scores, list) and len(subject_scores) == 30,
             f"Expected 30 subject summary rows: {run_dir}")
    by_subject = {subject: [row for row in records.values() if row["subject"] == subject]
                  for subject in SUBJECTS}
    scores = {row.get("subject"): row for row in subject_scores}
    _require(len(scores) == 30 and set(scores) == set(SUBJECTS),
             f"Summary subjects differ from MMMU: {run_dir}")
    for subject, rows in by_subject.items():
        score = scores[subject]
        subject_correct = sum(row.get("correct") is True for row in rows)
        generation_seconds = math.fsum(row["generation_seconds"] for row in rows)
        sample_seconds = math.fsum(row.get("total_sample_seconds", row["generation_seconds"])
                                   for row in rows)
        _require(score.get("correct") == subject_correct and score.get("total") == 30
                 and math.isclose(float(score.get("accuracy", -1)), subject_correct / 30,
                                  rel_tol=0, abs_tol=1e-12)
                 and math.isclose(float(score.get("generation_seconds", -1)), generation_seconds,
                                  rel_tol=1e-12, abs_tol=1e-12)
                 and math.isclose(float(score.get("sample_seconds", -1)), sample_seconds,
                                  rel_tol=1e-12, abs_tol=1e-12),
                 f"Subject summary differs from sample rows: {subject}")

    with (run_dir / "subject_scores.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_scores = {row.get("subject"): row for row in csv_rows}
    _require(len(csv_rows) == 30 and len(csv_scores) == 30 and set(csv_scores) == set(SUBJECTS),
             f"Subject CSV rows differ from MMMU: {run_dir}")
    for subject, score in scores.items():
        csv_score = csv_scores[subject]
        _require(int(csv_score["correct"]) == score["correct"] and int(csv_score["total"]) == 30
                 and math.isclose(float(csv_score["accuracy"]), float(score["accuracy"]),
                                  rel_tol=0, abs_tol=1e-12)
                 and math.isclose(float(csv_score["generation_seconds"]),
                                  float(score["generation_seconds"]), rel_tol=1e-12, abs_tol=1e-12)
                 and math.isclose(float(csv_score["sample_seconds"]),
                                  float(score["sample_seconds"]), rel_tol=1e-12, abs_tol=1e-12),
                 f"Subject CSV differs from summary: {subject}")
    return correct


def _verify_scorer(scored_dir: Path, identity: dict[str, Any]) -> str:
    root = Path(__file__).resolve().parents[1]
    files = scoring_source_files(PARSER)
    expected_paths = {str(path.relative_to(root)): path for path in files}
    manifest = read_json(scored_dir / "scorer_source.json")
    _require(manifest.get("parser") == PARSER and manifest.get("comparison_policy") == COMPARISON_POLICY,
             "Scorer snapshot parser or comparison policy mismatch")
    _require(manifest.get("scoring_sha256") == identity.get("scoring_sha256"),
             "Scorer snapshot and run manifest hashes differ")
    entries = manifest.get("files")
    _require(isinstance(entries, list), "Scorer snapshot file manifest is missing")
    by_path = {entry.get("path"): entry for entry in entries if isinstance(entry, dict)}
    _require(len(by_path) == len(entries) and set(by_path) == set(expected_paths),
             "Scorer snapshot file list differs from current scorer dependencies")

    scorer_root = scored_dir / "scorer_source"
    file_hashes = []
    for relative, current in expected_paths.items():
        entry = by_path[relative]
        snapshot = scorer_root / relative
        _require(snapshot.is_file() and not snapshot.is_symlink()
                 and snapshot.resolve().is_relative_to(scorer_root.resolve()),
                 f"Unsafe or missing scorer snapshot file: {relative}")
        _require(entry.get("bytes") == snapshot.stat().st_size == current.stat().st_size
                 and entry.get("sha256") == sha256_file(snapshot) == sha256_file(current),
                 f"Scorer snapshot/current file hash mismatch: {relative}")
        file_hashes.append(entry["sha256"])
    scoring_sha256 = digest([PARSER, *file_hashes])
    _require(scoring_sha256 == manifest["scoring_sha256"], "Scorer fingerprint mismatch")
    return scoring_sha256


def validate(source_run: Path, scored_dir: Path) -> dict[str, Any]:
    """Verify the original 900-row source and its V8-only scored derivative."""
    source_run, scored_dir = Path(source_run).resolve(), Path(scored_dir).resolve()
    source = read_json(scored_dir / "source.json")
    manifest = read_json(scored_dir / "run_manifest.json")
    identity = manifest.get("identity", {})
    source_sha256 = sha256_file(source_run / "predictions.jsonl")
    _require(Path(source.get("source_run", "")).resolve() == source_run,
             "Rescore source_run does not match the supplied original run")
    _require(source.get("source_predictions_sha256") == source_sha256
             and identity.get("source_sha256") == source_sha256,
             "Original predictions hash mismatch")
    _require(source.get("parser") == identity.get("parser") == PARSER,
             "Rescore is not pinned to the V8 parser")
    original_cfg = yaml.safe_load((source_run / "resolved_eval_config.yaml").read_text(encoding="utf-8"))
    rescored_cfg = yaml.safe_load((scored_dir / "resolved_eval_config.yaml").read_text(encoding="utf-8"))
    _require(isinstance(original_cfg, dict) and rescored_cfg == (original_cfg | {"parser": PARSER}),
             "Rescore changed inference configuration beyond selecting V8 scoring")

    original, rescored = _records(source_run), _records(scored_dir)
    _require(set(original) == set(rescored), "Original and rescored sample IDs differ")
    expected_ids = manifest.get("expected_ids")
    _require(isinstance(expected_ids, list) and len(expected_ids) == 900
             and len(set(expected_ids)) == 900 and set(expected_ids) == set(original),
             "Rescore manifest IDs differ from original sample IDs")
    source_ids = read_json(source_run / "run_manifest.json").get("expected_ids")
    _require(isinstance(source_ids, list) and len(source_ids) == 900
             and len(set(source_ids)) == 900 and set(source_ids) == set(original),
             "Original manifest IDs differ from original sample IDs")
    _predictions(source_run / "predictions.jsonl", original)
    _predictions(scored_dir / "predictions.jsonl", rescored)

    scoring_sha256 = _verify_scorer(scored_dir, identity)
    _require(source.get("scoring_sha256") == scoring_sha256,
             "Source metadata scorer hash mismatch")
    allowed = SCORE_FIELDS | {"_record_sha256"}
    for sample_id, before in original.items():
        after = rescored[sample_id]
        expected = score_with_parser(before["raw_response"], before["question_type"], before["options"],
                                     before["answer"], PARSER, before.get("finish_reason"),
                                     question=before.get("question", ""))
        _require(all(after.get(key) == value for key, value in expected.items()),
                 f"V8 replay differs from rescored row: {sample_id}")
        row_allowed = allowed.copy()
        if "comparison_policy" in after:
            _require(after["comparison_policy"] == COMPARISON_POLICY,
                     f"Unexpected V8 comparison policy on row: {sample_id}")
            row_allowed.add("comparison_policy")
        _require({k: v for k, v in before.items() if k not in row_allowed}
                 == {k: v for k, v in after.items() if k not in row_allowed},
                 f"Non-scoring sample fields changed: {sample_id}")

    source_correct = _summary(source_run, original)
    correct = _summary(scored_dir, rescored)
    return {"status": "VERIFIED", "parser": PARSER, "source_run": str(source_run),
            "scored_dir": str(scored_dir), "samples": 900, "source_correct": source_correct,
            "correct": correct, "accuracy": correct / 900,
            "source_predictions_sha256": source_sha256, "scoring_sha256": scoring_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--scored-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.source_run, args.scored_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
