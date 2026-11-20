"""Saved-response previews, paired counts, and immutable judge replay."""

import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import yaml

from mmdl.evaluation import cli, reporting
from mmdl.evaluation.parsers import score_with_parser
from mmdl.evaluation.CompassVerifier_checker import backend, cli as checker
from mmdl.evaluation.CompassVerifier_checker.tests.test_selection import prediction
from mmdl.evaluation.writer import RunWriter, finalize
from mmdl.runtime.artifacts import read_json, sha256_file, write_json
from mmdl.runtime.contracts import SUBJECTS


class SubmissionReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_rows(self, name, responses, *, pending=False, parser_id="team-final-answer-v8"):
        directory = self.root / name
        ids = [f"test_Physics_{index}" for index in range(len(responses) + int(pending))]
        mapping = {sample_id: "Physics" for sample_id in ids}
        identity = {"suite": "mmmu-pro", "id_subjects": mapping, "rehearsal": True}
        with RunWriter(directory, identity, ids) as writer:
            for sample_id, response in zip(ids, responses):
                row = prediction(sample_id, response, subject="Physics", seed=3407,
                                 generation_seconds=0.25, status="COMPLETED")
                row.update(score_with_parser(response, row["question_type"], row["options"],
                                             row["answer"], parser_id, "stop", question=row["question"]))
                writer.save(row)
        (directory / "resolved_eval_config.yaml").write_text(yaml.safe_dump({
            "dataset": {"id": "MMMU/MMMU_Pro"}, "parser": parser_id, "rehearsal": True}))
        finalize(directory, ids, "smoke")
        return directory

    def test_preview_reads_closed_records_and_preserves_source(self):
        source = self.run_rows("source", ["Final answer: B."], pending=True)
        before = {p: sha256_file(p) for p in source.rglob("*") if p.is_file()}
        output = self.root / "preview"
        report = cli.report_main(["preview", "--suite", "mmmu-pro", "--source-run", str(source),
                                  "--output-dir", str(output)])
        self.assertFalse(report["reportable"])
        self.assertEqual(report["denominator"], 1)
        self.assertEqual(len(report["missing_ids"]), 1)
        self.assertEqual(before, {p: sha256_file(p) for p in source.rglob("*") if p.is_file()})
        self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 1)
        with self.assertRaisesRegex(ValueError, "complete registered"):
            reporting.saved_run(source, "mmmu-pro")

    def test_exact_paired_counts_and_problem_alignment(self):
        base = self.run_rows("base", ["Final answer: B.", "Final answer: A.", "Final answer: B."])
        ft = self.run_rows("ft", ["Final answer: A.", "Final answer: B.", "Final answer: B."])
        report = reporting.paired_report(base, ft, "mmmu-pro")
        self.assertFalse(report["reportable"])
        counts = report["scores"]["v8"]
        self.assertEqual((counts["both_correct"], counts["finetuned_only"], counts["base_only"], counts["both_wrong"]),
                         (1, 1, 1, 0))
        self.assertEqual(counts["mcnemar_exact_two_sided_p"], 1)
        manifest = read_json(ft / "run_manifest.json")
        manifest["identity"]["id_subjects"]["test_Physics_0"] = "Math"
        write_json(ft / "run_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Registered subject"):
            reporting.paired_report(base, ft, "mmmu-pro")

    def test_paired_v8_replays_historical_scoring_from_raw(self):
        base = self.run_rows("old-parser", ["B. four"], parser_id="mmmu-official-no-random-v1")
        ft = self.run_rows("v8", ["B. four"])
        self.assertEqual(read_json(base / "summary.json")["correct"], 0)
        report = reporting.paired_report(base, ft, "mmmu-pro")
        self.assertEqual(report["scores"]["v8"]["both_correct"], 1)
        self.assertEqual(report["scores"]["v8"]["finetuned_only"], 0)

    def test_checker_replay_is_read_only_and_can_report_to_new_directory(self):
        fake = backend.FakeJudge(lambda _: {"p_a": 0.95, "p_b": 0.02, "p_c": 0.01,
                                           "top_token_id": 0})
        predictions = self.root / "predictions.jsonl"
        predictions.write_text(json.dumps(prediction()) + "\n")
        args = checker.parser().parse_args([
            "run", "--predictions", str(predictions), "--run-label", "synthetic",
            "--output-root", str(self.root / "checker"), "--mode", "smoke",
            "--model", "opencompass/CompassVerifier-3B", "--revision",
            "27d6d660bdf6c2b8a9f45906d4b76c508af65ab0", "--quantization", "nf4"])
        directory = checker.select_run(args, tokenizer=fake)
        args.targets = directory / "targets.jsonl"
        checker.judge_run(args, judge=fake)
        args.verdicts = directory / "verdicts.jsonl"
        checker.report_run(args, tokenizer=fake)
        before = {p: sha256_file(p) for p in directory.rglob("*") if p.is_file()}
        first = checker.replay_report(args, tokenizer=fake)
        second = checker.replay_report(args, tokenizer=fake)
        self.assertEqual(first, second)
        self.assertEqual(before, {p: sha256_file(p) for p in directory.rglob("*") if p.is_file()})
        args.output_dir = self.root / "new-report"
        checker.report_run(args, tokenizer=fake)
        self.assertEqual((args.output_dir / "summary.json").read_bytes(), (directory / "summary.json").read_bytes())

    def test_tokenizer_copy_rejects_extra_or_changed_files(self):
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        with self.assertRaisesRegex(ValueError, "exactly"):
            backend.verify_tokenizer_copy(tokenizer)
        for name in backend.TOKENIZER_GIT_BLOBS:
            (tokenizer / name).write_text("modified")
        with self.assertRaisesRegex(ValueError, "Git blob mismatch"):
            backend.verify_tokenizer_copy(tokenizer)
        (tokenizer / "tokenizer.json").write_text("unverified fast tokenizer")
        with self.assertRaisesRegex(ValueError, "exactly"):
            backend.verify_tokenizer_copy(tokenizer)

    def test_kernel_stage_lock_is_released_after_forced_process_exit(self):
        target = self.root / "targets.jsonl"
        target.touch()
        script = (
            "import sys,time; from argparse import Namespace; from pathlib import Path; "
            "from mmdl.evaluation.CompassVerifier_checker.cli import _exclusive_stage; "
            "locked=_exclusive_stage('judge')(lambda args: (print('ready',flush=True),time.sleep(60))); "
            "locked(Namespace(targets=Path(sys.argv[1])))")
        child = subprocess.Popen([sys.executable, "-c", script, str(target)], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            child.kill()
            child.wait(timeout=10)
            locked = checker._exclusive_stage("judge")(lambda args: "recovered")
            self.assertEqual(locked(type("Args", (), {"targets": target})()), "recovered")
            self.assertTrue((self.root / ".judge.lock").is_file())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            child.stdout.close()
            child.stderr.close()

    def test_partial_judge_resumes_only_unfinished_targets(self):
        predictions = self.root / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(prediction(f"synthetic-{i}")) + "\n" for i in range(3)))
        args = checker.parser().parse_args([
            "run", "--predictions", str(predictions), "--run-label", "restart",
            "--output-root", str(self.root / "outputs"), "--mode", "smoke", "--model",
            "opencompass/CompassVerifier-3B", "--revision", "27d6d660bdf6c2b8a9f45906d4b76c508af65ab0",
            "--quantization", "nf4"])
        calls = []
        def probability(_):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("forced interruption")
            return {"p_a": 0.95, "p_b": 0.02, "p_c": 0.01, "top_token_id": 0}
        fake = backend.FakeJudge(probability)
        directory = checker.select_run(args, tokenizer=fake)
        args.targets = directory / "targets.jsonl"
        with self.assertRaisesRegex(RuntimeError, "forced interruption"):
            checker.judge_run(args, judge=fake)
        self.assertEqual(len(read_json(directory / "judge_progress.json")["results"]), 1)
        checker.judge_run(args, judge=fake)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(checker._jsonl(directory / "verdicts.jsonl")), 3)

    def test_full_pro_requires_registration_and_1730_rows(self):
        source = self.run_rows("source", ["Final answer: B."])
        manifest = read_json(source / "run_manifest.json")
        manifest["identity"].pop("rehearsal")
        write_json(source / "run_manifest.json", manifest)
        cfg = yaml.safe_load((source / "resolved_eval_config.yaml").read_text())
        cfg.pop("rehearsal")
        (source / "resolved_eval_config.yaml").write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, "1730"):
            reporting.saved_run(source, "mmmu-pro")

    def test_full_pro_rescore_uses_1730_and_unequal_subject_denominators(self):
        from scripts.validate_rescore import validate

        source, scored = self.root / "pro-full", self.root / "pro-v8"
        mapping = {f"test_{SUBJECTS[index % 30]}_{index}": SUBJECTS[index % 30] for index in range(1730)}
        ids = list(mapping)
        with RunWriter(source, {"suite": "mmmu-pro", "id_subjects": mapping}, ids) as writer:
            for sample_id, subject in mapping.items():
                response = "Final answer: B." if subject == SUBJECTS[0] else "Final answer: A."
                writer.save(prediction(sample_id, response, subject=subject, generation_seconds=0.25,
                                       status="COMPLETED"))
        (source / "resolved_eval_config.yaml").write_text(yaml.safe_dump({
            "dataset": {"id": "MMMU/MMMU_Pro"}, "parser": "team-final-answer-v8"}))
        finalize(source, ids, "full")
        summary = cli.rescore_main(["--source-run", str(source), "--suite", "mmmu-pro",
                                    "--output-dir", str(scored), "--verify"])
        receipt = validate(source, scored, "mmmu-pro")
        self.assertEqual(receipt["samples"], 1730)
        self.assertEqual(summary["denominator"], 1730)
        self.assertAlmostEqual(summary["macro_average"], 1 / 30)
        self.assertNotEqual(summary["accuracy"], summary["macro_average"])


if __name__ == "__main__":
    unittest.main()
