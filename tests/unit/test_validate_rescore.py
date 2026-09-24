from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from mmdl.evaluation.final_answer_parser_v8 import COMPARISON_POLICY
from mmdl.evaluation.parsers import score_with_parser, scoring_source_files
from mmdl.evaluation.writer import finalize
from mmdl.runtime.artifacts import digest, file_records, read_json, sha256_file, write_json
from mmdl.runtime.contracts import SUBJECTS
from scripts.validate_rescore import PARSER, validate


def _save_row(path: Path, row: dict) -> None:
    row = {key: value for key, value in row.items() if key != "_record_sha256"}
    row["_record_sha256"] = digest(row)
    write_json(path, row)


def _sync_predictions(run_dir: Path) -> None:
    rows = [read_json(path) for path in sorted((run_dir / "samples").glob("*.json"))]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class ValidateRescoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / "source"
        self.scored = root / "scored"
        (self.source / "samples").mkdir(parents=True)
        (self.scored / "samples").mkdir(parents=True)
        self.ids = [f"validation_{subject}_{index}" for subject in SUBJECTS for index in range(1, 31)]

        source_rows = []
        scored_rows = []
        for subject in SUBJECTS:
            for index in range(1, 31):
                sample_id = f"validation_{subject}_{index}"
                row = {
                    "id": sample_id, "subject": subject, "question_type": "multiple-choice",
                    "question": "Choose the correct option.", "options": ["alpha", "beta"],
                    "answer": "A", "raw_response": "Final answer: A", "finish_reason": "stop",
                    "generated_tokens": 4, "generation_seconds": 0.25,
                    "total_sample_seconds": 0.5, "status": "COMPLETED",
                }
                row.update(score_with_parser(row["raw_response"], row["question_type"], row["options"],
                                             row["answer"], "team-final-answer-v6"))
                source_rows.append(row)
                v8_score = score_with_parser(row["raw_response"], row["question_type"], row["options"],
                                             row["answer"], PARSER, row["finish_reason"],
                                             question=row["question"])
                scored_rows.append(row | v8_score)

        for row in source_rows:
            _save_row(self.source / "samples" / f"{row['id']}.json", row)
        original_cfg = {
            "protocol_id": "mmmu-val-v8", "parser": "team-final-answer-v6",
            "generation": {"max_new_tokens": 32768}, "execution": {"backend": "vllm"},
        }
        (self.source / "resolved_eval_config.yaml").write_text(yaml.safe_dump(original_cfg), encoding="utf-8")
        write_json(self.source / "run_manifest.json", {"identity": {}, "expected_ids": self.ids})
        finalize(self.source, self.ids, "full")

        scoring_files = scoring_source_files(PARSER)
        repo_root = Path(__file__).resolve().parents[2]
        scorer_dir = self.scored / "scorer_source"
        for path in scoring_files:
            target = scorer_dir / path.relative_to(repo_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        scoring_sha256 = digest([PARSER, *[sha256_file(path) for path in scoring_files]])
        source_sha256 = sha256_file(self.source / "predictions.jsonl")
        (self.scored / "resolved_eval_config.yaml").write_text(
            yaml.safe_dump(original_cfg | {"parser": PARSER}), encoding="utf-8")
        identity = {"parser": PARSER, "scoring_sha256": scoring_sha256,
                    "source_sha256": source_sha256}
        write_json(self.scored / "run_manifest.json", {"identity": identity, "expected_ids": self.ids})
        write_json(self.scored / "source.json", {
            "source_run": str(self.source.resolve()), "source_predictions_sha256": source_sha256,
            "parser": PARSER, "scoring_sha256": scoring_sha256,
        })
        write_json(self.scored / "scorer_source.json", {
            "parser": PARSER, "comparison_policy": COMPARISON_POLICY,
            "scoring_sha256": scoring_sha256, "files": file_records(repo_root, scoring_files),
        })
        for row in scored_rows:
            _save_row(self.scored / "samples" / f"{row['id']}.json", row)
        finalize(self.scored, self.ids, "full")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_v8_artifact(self) -> None:
        result = validate(self.source, self.scored)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["samples"], 900)
        self.assertEqual(result["correct"], 900)

    def test_rejects_changed_non_scoring_field(self) -> None:
        path = self.scored / "samples" / f"{self.ids[0]}.json"
        row = read_json(path)
        row["question"] += " Changed."
        _save_row(path, row)
        _sync_predictions(self.scored)
        with self.assertRaisesRegex(ValueError, "Non-scoring sample fields changed"):
            validate(self.source, self.scored)

    def test_rejects_score_that_does_not_replay(self) -> None:
        path = self.scored / "samples" / f"{self.ids[0]}.json"
        row = read_json(path)
        row["parsed_answer"] = "B"
        _save_row(path, row)
        _sync_predictions(self.scored)
        with self.assertRaisesRegex(ValueError, "V8 replay differs"):
            validate(self.source, self.scored)

    def test_rejects_tampered_record_hash(self) -> None:
        path = self.source / "samples" / f"{self.ids[0]}.json"
        row = read_json(path)
        row["question"] += " Changed."
        write_json(path, row)
        with self.assertRaisesRegex(ValueError, "Sample record hash mismatch"):
            validate(self.source, self.scored)

    def test_rejects_sample_that_is_not_completed(self) -> None:
        path = self.source / "samples" / f"{self.ids[0]}.json"
        row = read_json(path)
        row["status"] = "FAILED"
        _save_row(path, row)
        with self.assertRaisesRegex(ValueError, "Sample is not completed"):
            validate(self.source, self.scored)

    def test_rejects_non_boolean_correctness(self) -> None:
        path = self.source / "samples" / f"{self.ids[0]}.json"
        row = read_json(path)
        row["correct"] = 1
        _save_row(path, row)
        with self.assertRaisesRegex(ValueError, "correctness is not boolean"):
            validate(self.source, self.scored)

    def test_rejects_id_subject_mismatch(self) -> None:
        path = self.source / "samples" / f"{self.ids[0]}.json"
        row = read_json(path)
        row["subject"] = "Biology"
        _save_row(path, row)
        with self.assertRaisesRegex(ValueError, "Sample ID/subject mismatch"):
            validate(self.source, self.scored)

    def test_rejects_source_predictions_hash_mismatch(self) -> None:
        path = self.scored / "source.json"
        source = read_json(path)
        source["source_predictions_sha256"] = "0" * 64
        write_json(path, source)
        with self.assertRaisesRegex(ValueError, "Original predictions hash mismatch"):
            validate(self.source, self.scored)

    def test_rejects_scorer_snapshot_change(self) -> None:
        path = self.scored / "scorer_source/src/mmdl/evaluation/parsers.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Scorer snapshot/current file hash mismatch"):
            validate(self.source, self.scored)

    def test_rejects_summary_mismatch(self) -> None:
        path = self.scored / "summary.json"
        summary = read_json(path)
        summary["correct"] -= 1
        write_json(path, summary)
        with self.assertRaisesRegex(ValueError, "Summary correctness differs"):
            validate(self.source, self.scored)

    def test_rejects_extra_duplicate_csv_row(self) -> None:
        path = self.scored / "subject_scores.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Subject CSV rows differ"):
            validate(self.source, self.scored)

    def test_rejects_rescore_config_change_outside_parser(self) -> None:
        path = self.scored / "resolved_eval_config.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg["generation"]["max_new_tokens"] -= 1
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed inference configuration"):
            validate(self.source, self.scored)


if __name__ == "__main__":
    unittest.main()
