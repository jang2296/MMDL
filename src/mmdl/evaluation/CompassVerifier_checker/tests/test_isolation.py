"""Keep the optional checker separate from the existing evaluation and trainer."""

import hashlib
import os
import string
import subprocess
import sys
import unittest

from .. import cli, prompt


class IsolationTests(unittest.TestCase):
    def test_cpu_imports_do_not_load_torch(self):
        code = ("import sys; from mmdl.evaluation.CompassVerifier_checker "
                "import selection, scoring, cli; assert 'torch' not in sys.modules")
        environment = dict(os.environ, PYTHONPATH=str(cli.ROOT / "src"))
        subprocess.run([sys.executable, "-c", code], env=environment,
                       cwd=cli.ROOT, check=True, capture_output=True, text=True)

    def test_exact_prompt_hash_and_three_fields(self):
        self.assertEqual(hashlib.sha256(prompt.CV_PROMPT.encode()).hexdigest(), prompt.CV_PROMPT_SHA256)
        fields = [field for _, field, _, _ in string.Formatter().parse(prompt.CV_PROMPT)
                  if field is not None]
        self.assertCountEqual(fields, ["question", "gold_answer", "llm_response"])

    def test_callers_are_limited_to_submission_evaluation_and_tests(self):
        matches = set()
        for directory in ("src", "scripts", "fine_tuning", "tests"):
            for path in (cli.ROOT / directory).rglob("*.py"):
                if path.is_relative_to(cli.HERE):
                    continue
                if "CompassVerifier_checker" in path.read_text(encoding="utf-8"):
                    matches.add(path.relative_to(cli.ROOT).as_posix())
        allowed = {"scripts/sync_submission.py", "tests/unit/test_sync_submission.py",
                   "src/mmdl/evaluation/cli.py", "src/mmdl/evaluation/reporting.py",
                   "tests/unit/test_submission_evaluation_cli.py", "tests/unit/test_submission_reporting.py",
                   "scripts/rehearse_submission.py", "fine_tuning/error_note/v18b_rehearsal_eval.py",
                   "fine_tuning/tests/test_error_note_v18b_rehearsal_eval.py"}
        self.assertFalse(matches - allowed, sorted(matches - allowed))

    def test_package_has_only_publishable_small_source_files(self):
        for path in cli.HERE.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            self.assertIn(path.suffix, {".py", ".md", ".yaml"}, str(path))
            self.assertLess(path.stat().st_size, 5_000_000)


if __name__ == "__main__":
    unittest.main()
