"""Small CPU checks for the new execution and artifact boundaries."""

import tempfile
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mmdl.evaluation.cli import validate_execution
from mmdl.evaluation.prompt import build_messages
from mmdl.evaluation.writer import RunWriter, finalize, render
from mmdl.runtime.artifacts import git_commit, resolve_image
from scripts import validate_run


class ReproductionContractTests(unittest.TestCase):
    def test_publication_gate_rejects_secrets_private_paths_and_raw_results(self):
        inspect = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/check_submission.py"))["inspect_blob"]
        self.assertTrue(inspect("config.txt", ("ghp_" + "x" * 32).encode()))
        self.assertTrue(inspect("results/run/predictions.jsonl", b'{"raw_response":"fixture"}'))
        self.assertTrue(inspect(".serena/state.json", b"{}"))
        self.assertTrue(inspect("model.safetensors", b"fixture"))
        self.assertFalse(inspect(".env.example", b'export HF_HOME="$HOME/cache"'))

    def test_offload_full_is_rejected_and_evaluation_requires_bound_run(self):
        args = SimpleNamespace(mode="full", job_id="job", run_id="job-evaluation",
                               run_role="evaluation", require_commit="a" * 40)
        with self.assertRaises(ValueError):
            validate_execution(args, {"placement": "cpu_offload"})
        validate_execution(args, {"placement": "gpu_only"})
        args.run_id = "job-other"
        with self.assertRaises(ValueError):
            validate_execution(args, {"placement": "gpu_only"})
        args.run_role = "assignment"
        with self.assertRaises(ValueError):
            validate_execution(args, {"placement": "gpu_only"})

    def test_clean_commit_gate_rejects_dirty_checkout(self):
        sha = "a" * 40
        with patch("mmdl.runtime.artifacts.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=sha)), \
                patch("mmdl.runtime.artifacts.subprocess.check_output", side_effect=[b" M src/x.py", b""]):
            with self.assertRaises(ValueError):
                git_commit(Path("."), sha)
        with self.assertRaises(ValueError):
            git_commit(Path("."), "main")

    def test_evaluation_and_legacy_artifact_audits_require_commit_and_bound_name(self):
        for role in ("evaluation", "assignment", "analysis"):
            for commit, name, message in (("main", f"job-{role}", "Missing fixed commit"),
                                           ("a" * 40, "wrong-name", "Wrong job/run role")):
                manifest = {"identity": {"run_role": role, "job_id": "job", "git_commit": commit}}
                with self.subTest(role=role, commit=commit), \
                        patch.object(Path, "is_file", return_value=True), \
                        patch.object(validate_run, "read_json", side_effect=[{}, manifest, manifest]):
                    with self.assertRaisesRegex(AssertionError, message):
                        validate_run.validate(Path(name), Path("public") / name)

    def test_shared_images_are_safe_in_evaluation_and_legacy_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            relative = "../../assets/images/" + "a" * 64 + ".png"
            image = root / "assets/images" / ("a" * 64 + ".png")
            image.parent.mkdir(parents=True)
            image.write_bytes(b"synthetic-not-a-benchmark")
            for role in ("evaluation", "assignment", "analysis"):
                run_dir = root / "runs" / f"job-{role}"
                with RunWriter(run_dir, {}, ["validation_Accounting_1"]) as writer:
                    writer.save({"id": "validation_Accounting_1", "subject": "Accounting", "correct": True,
                                 "parse_status": "OK", "generated_tokens": 1, "generation_seconds": 0.25,
                                 "total_sample_seconds": 0.5, "finish_reason": "eos", "images": [relative]})
                summary = finalize(run_dir, ["validation_Accounting_1"], "smoke")
                self.assertEqual(summary["subject_scores"][0]["generation_seconds"], 0.25)
                self.assertEqual(resolve_image(run_dir, relative), image)
                self.assertIn(f'src="{relative}"', render(run_dir).read_text())
                for unsafe in ("/etc/passwd", "../../../file.png", "../../assets/images/not-a-hash.png"):
                    with self.assertRaises(ValueError):
                        resolve_image(run_dir, unsafe)
            self.assertEqual(len(list(image.parent.iterdir())), 1)
            image.unlink()
            image.symlink_to(root / "elsewhere.png")
            with self.assertRaises(ValueError):
                resolve_image(run_dir, relative)

    def test_prompt_can_be_built_from_immutable_run_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "mmmu_open_v1.txt").write_text("{hint_prefix}Question: {question}")
            sample = {"question_type": "open", "question": "fixture only", "options": []}
            message = build_messages(sample, [object()], root)
            self.assertEqual(message[0]["content"][-1]["text"], "Question: fixture only")


if __name__ == "__main__":
    unittest.main()
