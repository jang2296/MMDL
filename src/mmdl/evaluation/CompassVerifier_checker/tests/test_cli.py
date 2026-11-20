"""Exercise artifacts and stage boundaries with synthetic V8-scored rows."""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import threading
import io
from pathlib import Path
import tempfile
import unittest

from mmdl.runtime.artifacts import read_json, sha256_file, write_json

from .. import backend, cli, selection
from .test_selection import prediction


class CLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.predictions = self.root / "predictions.jsonl"
        self.rows = [prediction("correct", "Final answer: B."),
                     prediction("open", "Final answer: 7.", question_type="open",
                                options=[], answer=["four", "4"]),
                     prediction("no-parse"),
                     prediction("wrong", "Final answer: A."),
                     prediction("length", "Still working.", finish_reason="length")]
        cli._write_jsonl(self.predictions, self.rows)
        self.fake = backend.FakeJudge(lambda _: {"p_a": 0.95, "p_b": 0.02, "p_c": 0.01,
                                               "top_token_id": 0, "top_token": "A"})

    def args(self, label="test", *extra):
        return cli.parser().parse_args([
            "run", "--predictions", str(self.predictions), "--run-label", label,
            "--output-root", str(self.root / "outputs"), "--mode", "smoke",
            "--model", "opencompass/CompassVerifier-3B", "--revision",
            "27d6d660bdf6c2b8a9f45906d4b76c508af65ab0", "--quantization", "nf4", *extra])

    def stages(self, args):
        directory = cli.select_run(args, tokenizer=self.fake)
        args.targets = directory / "targets.jsonl"
        cli.judge_run(args, judge=self.fake)
        args.verdicts = directory / "verdicts.jsonl"
        return directory

    def test_complete_flow_preserves_v8_and_is_byte_deterministic(self):
        outputs = []
        for label in ("first", "second"):
            args = self.args(label)
            directory = self.stages(args)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli.report_run(args, tokenizer=self.fake)
            self.assertIn("reportable=false", stdout.getvalue())
            self.assertNotIn("%", stdout.getvalue())
            self.assertNotIn("3/5", stdout.getvalue())
            summary = read_json(directory / "summary.json")
            self.assertEqual(summary["denominator"], 5)
            self.assertEqual(summary["transitions"], {"0->0": 2, "0->1": 2, "1->1": 1, "1->0": 0})
            outputs.append([(directory / name).read_bytes()
                            for name in ("verdicts.jsonl", "summary.json")])
        self.assertEqual(*outputs)

    def test_limit_selects_whole_items_and_never_invents_full_dataset_score(self):
        args = self.args("limited", "--limit", "1")
        directory = self.stages(args)
        self.assertEqual(len(cli._jsonl(args.verdicts)), 2)
        cli.report_run(args, tokenizer=self.fake)
        summary = read_json(directory / "summary.json")
        self.assertEqual(summary["denominator"], 1)
        self.assertEqual(summary["source_denominator"], 5)
        self.assertEqual(summary["scope"], "smoke-target-subset")

    def test_no_output_overwrite_at_any_stage(self):
        args = self.args()
        self.stages(args)
        with self.assertRaises(FileExistsError):
            cli.select_run(args, tokenizer=self.fake)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            cli.judge_run(args, judge=self.fake)
        cli.report_run(args, tokenizer=self.fake)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            cli.report_run(args, tokenizer=self.fake)

    def test_concurrent_judge_has_exclusive_stage_ownership(self):
        args = self.args()
        directory = cli.select_run(args, tokenizer=self.fake)
        args.targets = directory / "targets.jsonl"
        entered, release = threading.Event(), threading.Event()

        def probability(_):
            entered.set()
            if not release.wait(5):
                raise AssertionError("concurrent test timed out")
            return {"p_a": 0.95, "p_b": 0.02, "p_c": 0.01, "top_token_id": 0}

        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(cli.judge_run, args, judge=backend.FakeJudge(probability))
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(FileExistsError):
                    cli.judge_run(args, judge=self.fake)
            finally:
                release.set()
            self.assertEqual(first.result(), directory)
        self.assertTrue((directory / ".judge.lock").is_file())

    def test_private_paths_and_run_labels(self):
        for label in ("../escape", "bad/name", ".hidden", "x" * 62):
            with self.subTest(label=label), self.assertRaises(ValueError):
                cli.select_run(self.args(label), tokenizer=self.fake)
        args = self.args()
        args.output_root = cli.ROOT / "analysis_exports"
        with self.assertRaisesRegex(ValueError, "outside"):
            cli.select_run(args, tokenizer=self.fake)

    def test_official_requires_exact_profile_and_disallows_limit(self):
        config = selection.load_config()
        base = self.args()
        base.mode = "official"
        for changes in ({}, {"model": config["judge"]["official"]["model"]},
                        {**config["judge"]["official"], "quantization": "nf4"},
                        {**config["judge"]["official"], "limit": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                args = self.args()
                args.mode = "official"
                vars(args).update(changes)
                cli.select_run(args, tokenizer=self.fake)

    def test_report_rejects_rehashed_target_mutation(self):
        args = self.args()
        self.stages(args)
        targets = cli._jsonl(args.targets)
        targets[0]["response_truncated"] = True
        cli._write_jsonl(args.targets, targets)
        manifest = read_json(args.targets.parent / "selection.json")
        manifest["targets_sha256"] = sha256_file(args.targets)
        write_json(args.targets.parent / "selection.json", manifest)
        with self.assertRaisesRegex(ValueError, "Recomputed targets"):
            cli.report_run(args, tokenizer=self.fake)

    def test_report_rejects_missing_and_extra_verdict_even_when_rehashed(self):
        for label, mutate in (("missing", lambda rows: rows[:-1]),
                              ("extra", lambda rows: rows + [dict(rows[0], id="correct")])):
            args = self.args(label)
            self.stages(args)
            cli._write_jsonl(args.verdicts, mutate(cli._jsonl(args.verdicts)))
            manifest = read_json(args.verdicts.parent / "judge.json")
            manifest["verdicts_sha256"] = sha256_file(args.verdicts)
            write_json(args.verdicts.parent / "judge.json", manifest)
            with self.assertRaises(ValueError):
                cli.report_run(args, tokenizer=self.fake)

    def test_report_rejects_mixed_model_identity(self):
        args = self.args()
        directory = self.stages(args)
        metadata = read_json(directory / "judge.json")
        metadata["judge"]["model"] = "unexpected-model"
        write_json(directory / "judge.json", metadata)
        with self.assertRaisesRegex(ValueError, "Loaded judge identity mismatch"):
            cli.report_run(args, tokenizer=self.fake)

    def test_official_receipt_replays_all_cases_and_rejects_false_accept(self):
        config = selection.load_config()
        profile = {"mode": "official", "reportable": True, **config["judge"]["official"]}
        del profile["dtype"]
        with self.assertRaisesRegex(ValueError, "requires"):
            cli._require_checkset(None, profile, config)
        results = [dict(id=c["id"], expect=c["expect"], p_a=0.01, p_b=0.95, p_c=0.01,
                        top_token_id=1, top_token="B") for c in cli._cases()]
        receipt = {**cli._binding(config), "profile": profile,
                   "checkset_sha256": sha256_file(cli.HERE / "checkset_v1.yaml"),
                   "judge": {**config["judge"]["official"], "label_token_ids": {"A": 0, "B": 1, "C": 2}},
                   "results": results, "false_accept_count": 0}
        path = self.root / "checkset.json"
        write_json(path, receipt)
        self.assertEqual(cli._require_checkset(path, profile, config), sha256_file(path))
        negative = next(row for row in results if row["expect"] == "not_accept")
        negative.update(p_a=0.95, p_b=0.01, top_token_id=0, top_token="A")
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "false accepts"):
            cli._require_checkset(path, profile, config)


if __name__ == "__main__":
    unittest.main()
