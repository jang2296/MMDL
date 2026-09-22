import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmdl.runtime.contracts import SUBJECTS
from scripts import report_baseline


class ReportBaselineTests(unittest.TestCase):
    def test_final_report_rejects_old_or_non_vllm_protocol(self):
        valid = {"protocol_id": "mmmu-val-fast-vllm-32k-continuous-v1", "generation": {"max_new_tokens": 32768},
                 "execution": {"backend": "vllm", "batch_size": 2, "scheduling": "continuous"}}
        report_baseline._validate_final_protocol(valid)
        for key, value in (("protocol_id", "mmmu-val-fast-vllm-v1"),
                           ("protocol_id", "mmmu-val-fast-vllm-32k-v1"),
                           ("execution", {"backend": "vllm", "batch_size": 2})):
            altered = dict(valid)
            altered[key] = value
            with self.assertRaises(ValueError):
                report_baseline._validate_final_protocol(altered)

    def test_final_metrics_replace_only_the_authored_marker(self):
        rows = [
            {"subject": subject, "correct": index == 0, "parse_status": "PARSED", "finish_reason": "eos"}
            for subject in SUBJECTS
            for index in range(30)
        ]
        cfg = {"execution": {"backend": "vllm", "batch_size": 2}, "generation": {"max_new_tokens": 2048}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = (
                "# Authored provenance stays\n\n<!-- REPORT_STATUS:BEGIN -->\nold status\n<!-- REPORT_STATUS:END -->\n\n"
                "<!-- REPORT_RUNTIME:BEGIN -->\nold runtime\n<!-- REPORT_RUNTIME:END -->\n\n"
                "<!-- REPORT_DYNAMIC:BEGIN -->\nold\n<!-- REPORT_DYNAMIC:END -->\n\n"
                "## 8. Authored limit stays\n"
            )
            (root / "Assignment_1.md").write_text(template, encoding="utf-8")
            output = root / "out.md"
            with patch.object(report_baseline, "ROOT", root), \
                 patch.object(report_baseline, "_load_complete_run", return_value=({}, {}, {}, {}, cfg, rows, list(SUBJECTS))):
                report_baseline.render(root / "run", output)
            rendered = output.read_text(encoding="utf-8")
        self.assertIn("Authored provenance stays", rendered)
        self.assertIn("## 8. Authored limit stays", rendered)
        self.assertIn("| 1 | Accounting | 30 | 3.33% |", rendered)
        self.assertIn("Overall (macro avg)", rendered)
        self.assertNotIn("\nold\n", rendered)
        self.assertNotIn("old status", rendered)
        self.assertNotIn("old runtime", rendered)
        self.assertNotIn("COMPLETE900 검증 전", rendered)

    def test_dynamic_report_has_all_subjects_and_short_gap_diagnosis(self):
        rows = [
            {"subject": subject, "correct": False, "parse_status": "NO_PARSE", "finish_reason": "length"}
            for subject in SUBJECTS
            for _ in range(30)
        ]
        dynamic = report_baseline._dynamic_results(
            rows, list(SUBJECTS), {"execution": {"backend": "vllm"}, "generation": {"max_new_tokens": 2048}}
        )
        self.assertEqual(dynamic.count("| 30 | Sociology | 30 | 0.00% |"), 1)
        self.assertIn("NO_PARSE 900개", dynamic)
        self.assertLess(len(dynamic.split("## 7. 격차 분석\n\n", 1)[1]), 1000)


if __name__ == "__main__":
    unittest.main()
