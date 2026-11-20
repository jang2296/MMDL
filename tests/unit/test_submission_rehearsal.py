"""The GPU operator must retain completed records and terminate only its own child."""

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from mmdl.runtime.artifacts import sha256_file
from scripts.rehearse_submission import _kill_owned, amplify_fixture_vision, unchanged_records


class SubmissionRehearsalTests(unittest.TestCase):
    def test_synthetic_positive_control_changes_only_vision_B(self):
        import torch
        names = ("vision.lora_A.default.weight", "vision.lora_B.default.weight",
                 "language.lora_B.default.weight", "merger.weight")
        parameters = {name: torch.nn.Parameter(torch.ones(2, 2)) for name in names}
        bundle = SimpleNamespace(model=SimpleNamespace(named_parameters=lambda: parameters.items()),
                                 inventory={"trainable": [{"name": name, "group": group}
                                            for name, group in zip(names, ("V", "V", "L", "M"), strict=True)]})
        value = amplify_fixture_vision(bundle)
        self.assertEqual(value["changed_parameters"], [names[1]])
        self.assertTrue(torch.equal(parameters[names[1]], torch.full((2, 2), 256)))
        for name in (names[0], names[2], names[3]):
            self.assertTrue(torch.equal(parameters[name], torch.ones(2, 2)))

    def test_resume_keeps_saved_bytes_and_adds_new_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = directory / "first.json"
            first.write_text('{"id":"first"}')
            before = {first.name: sha256_file(first)}
            (directory / "second.json").write_text('{"id":"second"}')
            self.assertEqual(len(unchanged_records(before, directory)), 2)
            first.write_text('{"id":"first","regenerated":true}')
            with self.assertRaisesRegex(ValueError, "changed or disappeared"):
                unchanged_records(before, directory)

    def test_missing_record_and_unreached_boundary_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                unchanged_records({"missing.json": "a" * 64}, Path(tmp))
            with self.assertRaises(ValueError):
                unchanged_records({}, Path(tmp))

    def test_owned_process_group_is_killed_after_identity_check(self):
        process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
        try:
            ticks = (Path("/proc") / str(process.pid) / "stat").read_text().split()[21]
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                _kill_owned(process, str(int(ticks) + 1))
            self.assertIsNone(process.poll())
            _kill_owned(process, ticks)
            self.assertEqual(process.returncode, -9)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
