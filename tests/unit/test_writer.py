from __future__ import annotations

import json
import tempfile
import unittest
from math import isclose
from pathlib import Path
from html import escape

from mmdl.evaluation.writer import RunWriter, finalize, render
from mmdl.runtime.contracts import SUBJECTS


def record(sample_id: str, subject: str, correct: bool = True) -> dict:
    return {"id": sample_id, "subject": subject, "correct": correct, "parse_status": "OK",
            "generated_tokens": 3, "generation_seconds": 0.1, "finish_reason": "eos"}


class WriterTests(unittest.TestCase):
    def test_resume_is_identity_bound_and_records_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            writer = RunWriter(path, {"protocol": "one", "model": "base", "environment": "x"}, ["one"])
            writer.save(record("one", "subject"))
            with self.assertRaises(FileExistsError):
                writer.save(record("one", "subject"))
            with self.assertRaises(RuntimeError):
                RunWriter(path, {"protocol": "one", "model": "base", "environment": "x"}, ["one"], resume=True)
            writer.close()
            resumed = RunWriter(path, {"protocol": "one", "model": "base", "environment": "x"}, ["one"], resume=True)
            self.assertEqual(set(resumed.completed), {"one"})
            resumed.close()
            with self.assertRaises(ValueError):
                RunWriter(path, {"protocol": "changed", "model": "base", "environment": "x"}, ["one"], resume=True)

    def test_resume_rejects_tampered_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            identity = {"protocol": "one"}
            with RunWriter(path, identity, ["one"]) as writer:
                writer.save(record("one", "subject"))
            self.assertEqual([item.name for item in (path / "samples").iterdir()], ["one.json"])
            sample_path = path / "samples" / "one.json"
            saved = json.loads(sample_path.read_text(encoding="utf-8"))
            self.assertIn("_record_sha256", saved)
            saved["correct"] = False
            sample_path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash verification"):
                RunWriter(path, identity, ["one"], resume=True)

    def test_full_summary_has_900_denominator_and_30_subject_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            expected = [f"validation_{subject}_{index}" for subject in SUBJECTS for index in range(30)]
            with RunWriter(path, {"protocol": "one"}, expected) as writer:
                for subject in SUBJECTS:
                    for index in range(30):
                        writer.save(record(f"validation_{subject}_{index}", subject, index % 2 == 0))
            summary = finalize(path, expected, "full")
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["denominator"], 900)
            self.assertEqual(summary["correct"], 450)
            self.assertEqual(len(summary["subject_scores"]), 30)
            self.assertTrue(isclose(summary["macro_average"], summary["correct"] / 900, abs_tol=1e-12))
            summary_path = path / "summary.json"
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            saved["engine_telemetry"] = {"peak_vram": 1}
            summary_path.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(finalize(path, expected, "full")["engine_telemetry"], {"peak_vram": 1})

    def test_incomplete_full_does_not_reduce_denominator_and_render_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            expected = [f"validation_{subject}_{index}" for subject in SUBJECTS for index in range(30)]
            with RunWriter(path, {"protocol": "one"}, expected) as writer:
                for subject in SUBJECTS:
                    for index in range(30):
                        if subject == SUBJECTS[0] and index == 0:
                            continue
                        item = record(f"validation_{subject}_{index}", subject)
                        if subject == SUBJECTS[0] and index == 1:
                            item["question"] = "<script>alert(1)</script>"
                        writer.save(item)
                writer.failure(f"validation_{SUBJECTS[0]}_0", RuntimeError("GPU interrupted"))
            summary = finalize(path, expected, "full")
            self.assertEqual(summary["status"], "INCOMPLETE")
            self.assertEqual(summary["denominator"], 900)
            self.assertEqual(summary["subject_scores"][0]["total"], 30)
            html_path = render(path)
            page = html_path.read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>alert", page)

    def test_review_shows_readable_fields_before_full_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            raw = "<raw>&" + "x" * 2000
            item = record("one", "subject") | {
                "question_type": "multiple-choice",
                "question": "<question>",
                "options": ["<first>", "second &"],
                "parsed_answer": "A",
                "answer": "B",
                "raw_response": raw,
                "model_explanation": None,
                "gold_explanation": "gold <explanation>",
                "status": "COMPLETED",
                "generated_token_ids": [1, 2, 3],
            }
            with RunWriter(path, {"protocol": "one"}, ["one"]) as writer:
                writer.save(item)
            page = render(path).read_text(encoding="utf-8")
            self.assertLess(page.index("Question"), page.index("generated_token_ids"))
            self.assertIn(escape(raw), page)
            self.assertIn("<pre>null</pre>", page)
            self.assertIn("&lt;question&gt;", page)
            self.assertIn("&lt;first&gt;", page)
            self.assertIn('<ol type="A">', page)
            self.assertNotIn("<question>", page)

    def test_recovered_failure_is_historical_not_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            identity = {"protocol": "one"}
            writer = RunWriter(path, identity, ["one"])
            writer.failure("one", RuntimeError("interrupted"))
            writer.close()
            with RunWriter(path, identity, ["one"], resume=True) as resumed:
                resumed.save(record("one", "subject"))
            summary = finalize(path, ["one"], "smoke")
            self.assertEqual(summary["status"], "SMOKE")
            self.assertEqual(summary["external_failure_count"], 1)
            self.assertEqual(summary["unresolved_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
