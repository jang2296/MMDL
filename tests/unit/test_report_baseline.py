import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmdl.runtime.contracts import SUBJECTS
from scripts import report_baseline


class ReportBaselineTests(unittest.TestCase):
    def test_complete_run_accepts_current_and_legacy_labels_without_relabeling(self):
        cfg = {"protocol_id": report_baseline.FINAL_PROTOCOL_ID,
               "generation": {"max_new_tokens": 32768},
               "execution": {"backend": "vllm", "batch_size": 2, "scheduling": "continuous"},
               "dataset": {"subjects": list(SUBJECTS), "expected_total": 900}}
        summary = {"status": "COMPLETE", "mode": "full", "expected_count": 900,
                   "completed_count": 900, "denominator": 900, "missing_count": 0,
                   "unresolved_failure_count": 0, "correct": 0, "macro_average": 0.0}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "runs/job-analysis"
            (run_dir / "samples").mkdir(parents=True)
            for subject in SUBJECTS:
                for index in range(30):
                    row = {"id": f"validation_{subject}_{index}", "subject": subject,
                           "status": "COMPLETED", "correct": False}
                    (run_dir / "samples" / f"{row['id']}.json").write_text(json.dumps(row))
            for name, value in (("summary.json", summary), ("environment.json", {}), ("backend.json", {})):
                (run_dir / name).write_text(json.dumps(value))
            (run_dir / "resolved_eval_config.yaml").write_text(json.dumps(cfg))
            manifest_path = run_dir / "run_manifest.json"
            for role in ("evaluation", "analysis", "assignment"):
                with self.subTest(role=role):
                    manifest_path.write_text(json.dumps({"identity": {"run_role": role}}))
                    original = manifest_path.read_bytes()
                    # Isolate the reporting boundary; the artifact auditor has its own tests.
                    with patch("scripts.validate_run.validate") as audit:
                        loaded = report_baseline._load_complete_run(run_dir)
                    audit.assert_called_once_with(run_dir, Path(temporary) / "public/job-analysis")
                    self.assertEqual(loaded[3]["identity"]["run_role"], role)
                    self.assertEqual(manifest_path.read_bytes(), original)
            for role in ("standalone", "smoke", "unknown", None):
                identity = {"run_role": role} if role is not None else {}
                manifest_path.write_text(json.dumps({"identity": identity}))
                with self.subTest(rejected_role=role), patch("scripts.validate_run.validate") as audit:
                    with self.assertRaisesRegex(ValueError, "fixed-commit inference provenance"):
                        report_baseline._load_complete_run(run_dir)
                    audit.assert_not_called()
            manifest_path.write_text(json.dumps({"identity": {"run_role": "evaluation"}}))
            with patch("scripts.validate_run.validate", side_effect=AssertionError("hash mismatch")):
                with self.assertRaisesRegex(ValueError, "independent recovered-run audit"):
                    report_baseline._load_complete_run(run_dir)
            for changes in ({"status": "INCOMPLETE"}, {"completed_count": 899},
                            {"unresolved_failure_count": 1}, {"macro_average": 0.5}):
                (run_dir / "summary.json").write_text(json.dumps(summary | changes))
                with self.subTest(changes=changes), patch("scripts.validate_run.validate") as audit:
                    with self.assertRaises(ValueError):
                        report_baseline._load_complete_run(run_dir)
                    audit.assert_not_called()

    def test_device_memory_uses_exact_current_or_legacy_evaluation_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "jobs").mkdir()
            for role in ("evaluation", "analysis", "assignment"):
                manifest = {"identity": {"job_id": "job", "run_role": role}}
                (root / "jobs/job.json").write_text(json.dumps({"stages": [
                    {"stage": "smoke", "peak_observed_device_memory_used_bytes": 30 * 1024**3},
                    {"stage": role, "peak_observed_device_memory_used_bytes": 20 * 1024**3,
                     "gpu_memory_sample_interval_ms": 500, "gpu_memory_sample_count": 10},
                ]}))
                with self.subTest(role=role):
                    value = report_baseline._sampled_device_memory(root / f"runs/job-{role}", manifest, "vllm")
                    self.assertIn("20.00 GiB", value)
                    self.assertNotIn("30.00 GiB", value)
                    self.assertIn("500 ms 간격", value)

    def test_rescore_overlay_keeps_original_runtime_and_requires_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scored = root / "scored"
            (scored / "samples").mkdir(parents=True)
            row = {"id": "one", "subject": "Art", "correct": True}
            (scored / "samples/one.json").write_text(json.dumps(row))
            (scored / "scorer_source.json").write_text(json.dumps(
                {"parser": "team-final-answer-v8", "scoring_sha256": "fixture-hash"}))
            (root / "reports").mkdir()
            (root / "reports/mmmu_baseline.md").write_text(
                "<!-- REPORT_DYNAMIC:BEGIN -->\nold\n<!-- REPORT_DYNAMIC:END -->\n"
                "<!-- REPORT_RUNTIME:BEGIN -->\nold\n<!-- REPORT_RUNTIME:END -->\n"
                "<!-- REPORT_STATUS:BEGIN -->\nold\n<!-- REPORT_STATUS:END -->\n")
            summary = {"total_seconds_this_invocation": 12345}
            with (patch.object(report_baseline, "ROOT", root),
                  patch.object(report_baseline, "_load_complete_run", return_value=
                               (summary, {}, {}, {}, {}, [], ["Art"])),
                  patch.object(report_baseline, "_dynamic_results", return_value="new scores") as scores,
                  patch.object(report_baseline, "_runtime_metrics", return_value="original runtime") as timing,
                  patch.object(report_baseline, "_status_metrics", return_value="COMPLETE900"),
                  patch("scripts.validate_rescore.validate") as audit):
                report_baseline.render(root / "original", root / "report.md", scored)
                audit.assert_called_once_with(root / "original", scored)
                self.assertEqual(scores.call_args.args[0], [row])
                self.assertIs(timing.call_args.args[0], summary)
                text = (root / "report.md").read_text()
                self.assertIn("§4의 최종 채점 규칙", text)
                self.assertNotIn("COMPLETE900", text)
                self.assertNotIn("fixture-hash", text)
                self.assertIn("original runtime", text)
                audit.side_effect = ValueError("Source mismatch")
                with self.assertRaisesRegex(ValueError, "Source mismatch"):
                    report_baseline.render(root / "original", root / "bad.md", scored)
                self.assertFalse((root / "bad.md").exists())

    def test_final_report_rejects_old_or_non_vllm_protocol(self):
        valid = {"protocol_id": "mmmu-val-fast-vllm-32k-continuous-v1", "generation": {"max_new_tokens": 32768},
                 "execution": {"backend": "vllm", "batch_size": 2, "scheduling": "continuous"}}
        report_baseline._validate_final_protocol(valid)
        for protocol in ("mmmu-val-official-vllm-b2-v2", "mmmu-val-v8"):
            report_baseline._validate_final_protocol(valid | {"protocol_id": protocol})
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
            (root / "reports").mkdir()
            (root / "reports/mmmu_baseline.md").write_text(template, encoding="utf-8")
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
        self.assertIn("NO_PARSE는 900개", dynamic)
        self.assertIn("겹침은 900개", dynamic)
        self.assertLess(len(dynamic.split("## 7. 격차 분석\n\n", 1)[1]), 1000)

    def test_runtime_metrics_are_only_table_rows_and_fall_back_to_summary_snapshot(self):
        runtime = report_baseline._runtime_metrics(
            {"max_observed_device_memory_used_bytes": 22.87 * 1024**3},
            {"python": "3.12.3", "packages": {"vllm": "0.11"}},
            {"vllm": {"version": "0.11"}},
            {"identity": {}},
            {"execution": {"backend": "vllm", "batch_size": 2}},
            Path("/not-a-run"),
        )
        self.assertTrue(all(line.startswith("|") for line in runtime.splitlines() if line))
        self.assertIn("22.87 GiB", runtime)
        self.assertIn("snapshot 최대치", runtime)
        self.assertIn("Python 3.12.3", runtime)
        self.assertIn("../env/requirements-vllm.lock", runtime)
        self.assertNotIn("###", runtime)

    def test_authored_v8_guard_rejects_mismatched_static_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "configs/eval/mmmu_val_v8.yaml"
            config.parent.mkdir(parents=True)
            expected = {
                "prompt_policy": "P0",
                "parser": "team-final-answer-v8",
                "generation": {"max_new_tokens": 32768},
                "image": {"min_pixels": 1003520},
            }
            config.write_text(json.dumps(expected))
            with patch.object(report_baseline, "ROOT", root):
                report_baseline._validate_authored_v8_config(expected, None)
                with self.assertRaisesRegex(ValueError, "generation"):
                    report_baseline._validate_authored_v8_config(
                        expected | {"generation": {"max_new_tokens": 2048}}, None
                    )
                with self.assertRaisesRegex(ValueError, "scoring"):
                    report_baseline._validate_authored_v8_config(
                        expected, {"parser": "team-final-answer-v6"}
                    )
                with self.assertRaisesRegex(ValueError, "execution"):
                    report_baseline._validate_authored_v8_config(
                        expected | {"execution": {"batch_size": 1}}, None
                    )

    def test_cli_writes_only_canonical_report_or_explicit_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "reports/mmmu_baseline.md"
            canonical.parent.mkdir()
            canonical.write_text("authored report")

            def write_report(run_dir, output, scored_dir):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("verified report")

            with patch.object(report_baseline, "ROOT", root), \
                    patch.object(report_baseline, "render", side_effect=write_report) as render:
                with patch("sys.argv", ["report_baseline", "--run-dir", str(root / "run")]):
                    report_baseline.main()
                render.assert_called_once_with(root / "run", canonical, None)
                self.assertEqual(canonical.read_text(), "verified report")
                canonical.write_text("keep canonical")
                alternate = root / "alternate.md"
                with patch("sys.argv", ["report_baseline", "--run-dir", str(root / "run"),
                                        "--output", str(alternate)]):
                    report_baseline.main()
                self.assertEqual(alternate.read_text(), "verified report")
                self.assertEqual(canonical.read_text(), "keep canonical")
            self.assertFalse((root / "Assignment_1.md").exists())


if __name__ == "__main__":
    unittest.main()
