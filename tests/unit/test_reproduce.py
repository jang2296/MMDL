import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mmdl.runtime.artifacts import read_json
from mmdl.runtime.reproduce import (
    _validate_doctor_gpu,
    _validate_hardware_profile,
    _validate_origin,
    arguments,
    execute,
    main,
)


class ReproduceTests(unittest.TestCase):
    def test_default_is_side_effect_free_dry_run(self):
        output = io.StringIO()
        with patch("mmdl.runtime.reproduce.execute") as execute, contextlib.redirect_stdout(output):
            main(["--job-id", "fixture", "--commit", "a" * 40])
        execute.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["status"], "DRY_RUN")

    def test_refuses_moving_commit_unsafe_job_and_final_suite(self):
        for extra in (["--commit", "main"], ["--job-id", "../escape"], ["--suite", "mmmu-pro"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(["--job-id", "fixture", "--commit", "a" * 40] + extra)

    def test_origin_and_profile_are_currently_team_and_rtx4090_only(self):
        for origin in (
            "https://github.com/jang2296/MMDL",
            "https://github.com/jang2296/MMDL.git",
            "git@github.com:jang2296/MMDL",
            "git@github.com:jang2296/MMDL.git",
        ):
            _validate_origin(origin)
        for origin in ("https://github.com/other/MMDL", "git@github.com:jang2296/other.git"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                _validate_origin(origin)
        hardware_root = Path(__file__).resolve().parents[2] / "configs/hardware"
        _validate_hardware_profile(hardware_root / "rtx4090_24gb.yaml")
        with self.assertRaises(ValueError):
            _validate_hardware_profile(hardware_root / "rtx5060_8gb.yaml")

    def test_doctor_requires_one_rtx4090_with_23_gib(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "doctor.json"
            payload = {"environment": {"torch": {"gpus": [{
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total_bytes": 23 * 1024**3,
            }]}}}
            report.write_text(json.dumps(payload))
            _validate_doctor_gpu(report)
            for gpus in ([], payload["environment"]["torch"]["gpus"] * 2):
                payload["environment"]["torch"]["gpus"] = gpus
                report.write_text(json.dumps(payload))
                with self.subTest(gpus=len(gpus)), self.assertRaises(RuntimeError):
                    _validate_doctor_gpu(report)
            for gpu in (
                {"name": "NVIDIA GeForce RTX 4080", "memory_total_bytes": 24 * 1024**3},
                {"name": "NVIDIA GeForce RTX 4090", "memory_total_bytes": 22 * 1024**3},
                {"name": "NVIDIA GeForce RTX 4090", "memory_total_bytes": float("nan")},
            ):
                payload["environment"]["torch"]["gpus"] = [gpu]
                report.write_text(json.dumps(payload))
                with self.subTest(gpu=gpu), self.assertRaises(RuntimeError):
                    _validate_doctor_gpu(report)

    def test_resume_uses_new_log_after_interrupted_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "env").mkdir()
            (root / "env/requirements-eval.lock").write_text("fixture\n")
            paths = {name: root / name.lower() for name in (
                "MMDL_VOLUME_ROOT", "HF_HOME", "MMDL_DATA_ROOT",
                "MMDL_ARTIFACT_ROOT", "MMDL_VENV_ROOT",
            )}
            for name, path in paths.items():
                if name != "MMDL_VENV_ROOT":
                    path.mkdir()
            paths["MMDL_ARTIFACT_ROOT"].mkdir(exist_ok=True)
            deadline = datetime.now(timezone.utc) + timedelta(hours=1)
            interrupted = True

            def mock_executable(command, **kwargs):
                nonlocal interrupted
                if interrupted:
                    interrupted = False
                    kwargs["stdout"].write("interrupted\n")
                    raise KeyboardInterrupt
                command = [str(value) for value in command]
                if command[1:3] == ["-m", "venv"]:
                    venv = Path(command[3])
                    (venv / "bin").mkdir(parents=True)
                    (venv / "bin/python").touch()
                if "scripts/doctor.sh" in command:
                    output = Path(command[command.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps({"environment": {"torch": {"gpus": [{
                        "name": "NVIDIA GeForce RTX 4090",
                        "memory_total_bytes": 24 * 1024**3,
                    }]}}}))
                return SimpleNamespace(returncode=0)

            common = ["--job-id", "fixture", "--commit", "a" * 40, "--execute"]
            with patch.dict("os.environ", {
                "RUNPOD_POD_ID": "pod-fixture",
                "MMDL_WATCHDOG_ID": "watchdog-fixture",
                "MMDL_STORAGE_RESERVE_GIB": "10",
            }), patch("mmdl.runtime.reproduce.git_commit"), \
                    patch("mmdl.runtime.reproduce.require_runtime",
                          return_value=(paths, deadline, 1.0)), \
                    patch("mmdl.runtime.reproduce.subprocess.check_output",
                          return_value="https://github.com/jang2296/MMDL.git"), \
                    patch("mmdl.runtime.reproduce.subprocess.run", side_effect=mock_executable):
                with self.assertRaises(KeyboardInterrupt):
                    execute(arguments(common), root)
                execute(arguments(common + ["--resume"]), root)

            jobs = paths["MMDL_ARTIFACT_ROOT"] / "jobs"
            self.assertTrue((jobs / "fixture.000-venv.log").is_file())
            job = read_json(jobs / "fixture.json")
            self.assertEqual(job["status"], "REMOTE_VERIFIED")
            self.assertEqual(job["stages"][0]["log"], "fixture.001-venv.log")
            self.assertNotEqual(job["stages"][0]["log"], "fixture.000-venv.log")


if __name__ == "__main__":
    unittest.main()
