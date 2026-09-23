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
    _gpu_sample_metrics,
    _validate_doctor_gpu,
    _validate_hardware_profile,
    _validate_origin,
    _smoke_sample_ids,
    require_runtime,
    arguments,
    execute,
    main,
)


class ReproduceTests(unittest.TestCase):
    def _runtime_env(self, **extra):
        volume = Path(tempfile.mkdtemp())
        root = volume / "checkout"
        root.mkdir()
        for name in ("HF_HOME", "MMDL_DATA_ROOT", "MMDL_ARTIFACT_ROOT", "MMDL_VENV_ROOT"):
            (volume / name.lower()).mkdir()
        values = {
            "MMDL_VOLUME_ROOT": str(volume),
            "HF_HOME": str(volume / "hf_home"),
            "MMDL_DATA_ROOT": str(volume / "mmdl_data_root"),
            "MMDL_ARTIFACT_ROOT": str(volume / "mmdl_artifact_root"),
            "MMDL_VENV_ROOT": str(volume / "mmdl_venv_root"),
            "MMDL_STORAGE_RESERVE_GIB": "10",
            "RUNPOD_POD_ID": "pod-fixture",
        }
        values.update(extra)
        return root, values

    def test_default_cost_policy_rejects_missing_budget(self):
        root, env = self._runtime_env()
        with patch.dict("os.environ", env, clear=True), patch.object(Path, "is_mount", return_value=True), \
                patch("mmdl.runtime.reproduce.shutil.disk_usage", return_value=SimpleNamespace(free=50 * 1024**3)):
            with self.assertRaises(ValueError):
                require_runtime(root)

    def test_user_waived_cost_policy_is_explicit_and_returns_none(self):
        root, env = self._runtime_env(MMDL_COST_POLICY="user_waived")
        with patch.dict("os.environ", env, clear=True), patch.object(Path, "is_mount", return_value=True), \
                patch("mmdl.runtime.reproduce.shutil.disk_usage", return_value=SimpleNamespace(free=50 * 1024**3)):
            _, deadline, budget = require_runtime(root)
        self.assertIsNone(deadline)
        self.assertIsNone(budget)

    def test_invalid_cost_policy_is_rejected(self):
        root, env = self._runtime_env(MMDL_COST_POLICY="free_money")
        with patch.dict("os.environ", env, clear=True), patch.object(Path, "is_mount", return_value=True), \
                patch("mmdl.runtime.reproduce.shutil.disk_usage", return_value=SimpleNamespace(free=50 * 1024**3)):
            with self.assertRaises(ValueError):
                require_runtime(root)
    def test_gpu_samples_record_observed_whole_device_peak(self):
        with tempfile.TemporaryDirectory() as temp:
            samples = Path(temp) / "stage.log.gpu.csv"
            samples.write_text("2026/09/22 10:00:00.000, 1000, 10\n"
                               "2026/09/22 10:00:00.500, 1234, 90\ninvalid\n")
            metrics = _gpu_sample_metrics(samples)
            self.assertEqual(metrics["peak_observed_device_memory_used_bytes"], 1234 * 1024**2)
            self.assertEqual(metrics["gpu_memory_sample_count"], 2)
            self.assertEqual(metrics["gpu_memory_sample_interval_ms"], 500)
            self.assertEqual(metrics["memory_scope"], "whole_device_sampled_not_allocator_peak")

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

    def test_prepare_only_dry_run_excludes_inference(self):
        output = io.StringIO()
        with patch("mmdl.runtime.reproduce.execute") as execute, contextlib.redirect_stdout(output):
            main(["--job-id", "fixture", "--commit", "a" * 40, "--prepare-only"])
        execute.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertTrue(result["prepare_only"])
        self.assertFalse(any("evaluation" in step or "smoke" in step for step in result["steps"]))

    def test_only_one_evaluation_suite_is_accepted(self):
        self.assertEqual(arguments(["--job-id", "fixture", "--commit", "a" * 40]).suite, "mmmu-val")
        for suite in ("mmmu-val-two-runs", "mmmu-val-analysis", "mmmu-val-assignment"):
            with self.subTest(suite=suite), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(["--job-id", "fixture", "--commit", "a" * 40, "--suite", suite])

    def test_smoke_ids_match_backend_batch_policy(self):
        self.assertEqual(_smoke_sample_ids("transformers"), ["validation_Accounting_1"])
        self.assertEqual(_smoke_sample_ids("vllm"), ["validation_Accounting_1", "validation_Biology_29"])
        with tempfile.TemporaryDirectory() as temp:
            continuous = Path(temp) / "mmmu_val_continuous_vllm_v1.yaml"
            continuous.write_text("execution:\n  scheduling: continuous\n")
            self.assertEqual(_smoke_sample_ids("vllm", continuous),
                             ["validation_Accounting_1", "validation_Biology_29", "validation_Agriculture_1"])
            old_fast = Path(temp) / "mmmu_val_fast_vllm_v1.yaml"
            old_fast.write_text("execution:\n  backend: vllm\n")
            self.assertEqual(_smoke_sample_ids("vllm", old_fast),
                             ["validation_Accounting_1", "validation_Biology_29"])

    def test_origin_and_profile_are_currently_team_and_approved_gpu_only(self):
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
        _validate_hardware_profile(hardware_root / "rtx3090_24gb.yaml")
        _validate_hardware_profile(hardware_root / "rtx4090_24gb.yaml")
        _validate_hardware_profile(hardware_root / "rtx5090_32gb.yaml")
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
                {"name": "NVIDIA GeForce RTX 5090", "memory_total_bytes": 32 * 1024**3},
                {"name": "NVIDIA GeForce RTX 4090", "memory_total_bytes": 22 * 1024**3},
                {"name": "NVIDIA GeForce RTX 4090", "memory_total_bytes": float("nan")},
            ):
                payload["environment"]["torch"]["gpus"] = [gpu]
                report.write_text(json.dumps(payload))
                with self.subTest(gpu=gpu), self.assertRaises(RuntimeError):
                    _validate_doctor_gpu(report)

    def test_doctor_requires_selected_rtx5090_with_31_gib(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "doctor.json"
            payload = {"environment": {"torch": {"gpus": [{
                "name": "NVIDIA RTX 5090",
                "memory_total_bytes": 31 * 1024**3,
            }]}}}
            report.write_text(json.dumps(payload))
            hardware = Path(__file__).resolve().parents[2] / "configs/hardware/rtx5090_32gb.yaml"
            _validate_doctor_gpu(report, hardware)
            payload["environment"]["torch"]["gpus"][0]["name"] = "NVIDIA GeForce RTX 4090"
            report.write_text(json.dumps(payload))
            with self.assertRaises(RuntimeError):
                _validate_doctor_gpu(report, hardware)

    def test_doctor_requires_selected_rtx3090_with_23_gib(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "doctor.json"
            payload = {"environment": {"torch": {"gpus": [{
                "name": "NVIDIA GeForce RTX 3090",
                "memory_total_bytes": 23 * 1024**3,
            }]}}}
            report.write_text(json.dumps(payload))
            hardware = Path(__file__).resolve().parents[2] / "configs/hardware/rtx3090_24gb.yaml"
            _validate_doctor_gpu(report, hardware)
            for gpu in (
                {"name": "NVIDIA GeForce RTX 4090", "memory_total_bytes": 24 * 1024**3},
                {"name": "NVIDIA GeForce RTX 3090", "memory_total_bytes": 22 * 1024**3},
            ):
                payload["environment"]["torch"]["gpus"] = [gpu]
                report.write_text(json.dumps(payload))
                with self.subTest(gpu=gpu), self.assertRaises(RuntimeError):
                    _validate_doctor_gpu(report, hardware)

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

            class Monitor:
                def poll(self):
                    return None

                def terminate(self):
                    raise ProcessLookupError("already exited")

                def wait(self, timeout=None):
                    return 0

                def kill(self):
                    pass

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
                    patch("mmdl.runtime.reproduce._preflight_gpu", return_value={"gpu_name": "fixture", "driver_version": "fixture", "container_image": None}), \
                    patch("mmdl.runtime.reproduce.require_runtime",
                          return_value=(paths, deadline, 1.0)), \
                    patch("mmdl.runtime.reproduce.subprocess.check_output",
                          return_value="https://github.com/jang2296/MMDL.git"), \
                    patch("mmdl.runtime.reproduce.subprocess.Popen", return_value=Monitor()), \
                    patch("mmdl.runtime.reproduce.subprocess.run", side_effect=mock_executable):
                with self.assertRaises(KeyboardInterrupt):
                    execute(arguments(common), root)
                execute(arguments(common + ["--resume"]), root)

            jobs = paths["MMDL_ARTIFACT_ROOT"] / "jobs"
            self.assertTrue((jobs / "fixture.000-venv.log").is_file())
            job = read_json(jobs / "fixture.json")
            self.assertEqual(job["status"], "REMOTE_VERIFIED")
            self.assertEqual(job["suite"], "mmmu-val")
            self.assertEqual(job["runs"], {"evaluation": "fixture-evaluation"})
            self.assertEqual([stage["stage"] for stage in job["stages"] if stage["stage"] in {"smoke", "evaluation"}],
                             ["smoke", "evaluation"])
            self.assertEqual([stage["stage"] for stage in job["stages"]][7:],
                             ["smoke", "evaluation", "evaluation-rescore", "evaluation-audit"])
            self.assertEqual(job["stages"][0]["log"], "fixture.001-venv.log")
            self.assertNotEqual(job["stages"][0]["log"], "fixture.000-venv.log")
            smoke = next(stage for stage in job["stages"] if stage["stage"] == "smoke")
            self.assertIsNone(smoke["peak_observed_device_memory_used_bytes"])
            self.assertEqual(smoke["gpu_memory_sample_count"], 0)
            self.assertIn("cleanup failed", smoke["gpu_memory_monitor_diagnostic"])
            self.assertNotIn("gpu_memory_sample_count", job["stages"][0])


if __name__ == "__main__":
    unittest.main()
