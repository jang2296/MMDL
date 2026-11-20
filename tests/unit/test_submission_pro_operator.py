"""CPU checks for the two independent public MMMU-Pro operator jobs."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from mmdl.runtime.reproduce import _full_run_deadline, _lock_receipt, arguments, execute, main


ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40


class SubmissionProOperatorTests(unittest.TestCase):
    def command(self, model="base", *extra):
        return ["--job-id", f"pro-{model}-3407", "--commit", COMMIT,
                "--suite", "mmmu-pro", "--model", model, "--score", "both",
                "--hardware", str(ROOT / "configs/hardware/a100_sxm_80gb.yaml"), *extra]

    def plan(self, command):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("mmdl.runtime.reproduce.execute") as execute:
            main(command)
        execute.assert_not_called()
        return json.loads(output.getvalue())["plan"]

    def test_two_dedicated_jobs_share_profile_but_keep_models_and_full_runs_separate(self):
        with patch.dict(os.environ, {}, clear=True):
            plans = [self.plan(self.command(model)) for model in ("base", "finetuned")]
        for model, plan in zip(("base", "finetuned"), plans):
            with self.subTest(model=model):
                self.assertEqual(plan["source"]["required_origin"], "https://github.com/jang2296/MMDL")
                self.assertEqual(plan["source"]["commit"], COMMIT)
                self.assertEqual(plan["model"], model)
                self.assertEqual(Path(plan["model_ref"]).name,
                                 "baseline.json" if model == "base" else "v18b.json")
                self.assertEqual(plan["enrollment"]["expected_total"], 1730)
                self.assertEqual(plan["enrollment"]["mode"], "FULL")
                self.assertEqual(plan["enrollment"]["run_id"], f"pro-{model}-3407-evaluation")
                self.assertEqual(plan["profile"]["execution"]["batch_size"], 9)
                self.assertEqual(plan["profile"]["execution"]["max_images_per_prompt"], 7)
                self.assertEqual(plan["profile"]["generation"]["max_new_tokens"], 32768)
                self.assertEqual(plan["profile"]["model"]["dtype"], "bfloat16")
                self.assertEqual(Path(plan["lock"]).name, "requirements-proof.lock")
                self.assertEqual(plan["lock_receipt"], _lock_receipt(ROOT / "env/requirements-proof.lock"))
                self.assertEqual(plan["environment"]["manager"], "uv")
                install = plan["environment"]["commands"]["install"]
                self.assertEqual(install[:3], ["uv", "pip", "install"])
                self.assertEqual(install[install.index("--index-strategy") + 1], "unsafe-best-match")
                self.assertEqual(Path(install[-1]).name, "requirements-proof.lock")
                stages = {stage["stage"]: stage for stage in plan["commands"]}
                self.assertEqual(set(stages), {"download", "evaluation", "evaluation-rescore", "evaluation-audit"})
                command = stages["evaluation"]["argv"]
                self.assertEqual(command[command.index("--mode") + 1], "full")
                self.assertEqual(command[command.index("--score") + 1], "both")
                self.assertNotIn("--sample-id", command)
                self.assertNotIn("--limit", command)
                self.assertNotIn("--no-download", command)
                self.assertEqual(stages["download"]["argv"][-2:], ["--suite", "mmmu-pro"])
                self.assertIn("CompassVerifier-7B", plan["scoring"]["supplemental"])
                self.assertIn(f"pro-{model}-3407", plan["scoring"]["log_pattern"])
                for field in ("chain_timeout_seconds", "full_stage_timeout_seconds", "subprocess_timeout_seconds"):
                    self.assertIsNone(plan["time_limits"][field])
                self.assertEqual(plan["time_limits"]["minimum_full_window_seconds"], 72000)
                self.assertTrue(all(stage["timeout_seconds"] is None for stage in stages.values()))
        self.assertEqual(plans[0]["protocol_sha256"], plans[1]["protocol_sha256"])

    def test_absolute_public_wrapper_works_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            for model in ("base", "finetuned"):
                with self.subTest(model=model):
                    env = os.environ | {"MMDL_BOOTSTRAP_PYTHON": sys.executable}
                    command = ["bash", str(ROOT / "scripts/reproduce.sh"), *self.command(model)]
                    completed = subprocess.run(command, cwd=temp, env=env, check=True,
                                               capture_output=True, text=True)
                    result = json.loads(completed.stdout)
                    self.assertEqual(result["status"], "DRY_RUN")
                    self.assertEqual(result["plan"]["model"], model)
                    self.assertEqual(result["plan"]["enrollment"]["expected_total"], 1730)

    def test_short_explicit_deadline_is_preserved_and_rejected_even_when_cost_waived(self):
        stop = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        for policy in ("", "user_waived"):
            with self.subTest(policy=policy), patch.dict(os.environ, {
                    "MMDL_STOP_DEADLINE_UTC": stop, "MMDL_COST_POLICY": policy}, clear=True):
                with self.assertRaisesRegex(ValueError, "before allocating a Pod"):
                    self.plan(self.command())
                self.assertEqual(os.environ["MMDL_STOP_DEADLINE_UTC"], stop)

    def test_twenty_hour_window_or_no_cutoff_passes_without_mutation(self):
        stop = datetime.now(timezone.utc) + timedelta(hours=21)
        with patch.dict(os.environ, {"MMDL_STOP_DEADLINE_UTC": stop.isoformat()}, clear=True):
            plan = self.plan(self.command())
        self.assertEqual(plan["time_limits"]["stop_deadline_utc"], stop.isoformat())
        with patch.dict(os.environ, {"MMDL_COST_POLICY": "user_waived"}, clear=True):
            plan = self.plan(self.command())
        self.assertIsNone(plan["time_limits"]["stop_deadline_utc"])

    def test_explicit_deadline_keeps_watchdog_required_when_cost_is_waived(self):
        stop = datetime.now(timezone.utc) + timedelta(hours=21)
        env = {"MMDL_COST_POLICY": "user_waived", "MMDL_STOP_DEADLINE_UTC": stop.isoformat()}
        with patch.dict(os.environ, env, clear=True), \
                patch("mmdl.runtime.reproduce.git_commit"), \
                patch("mmdl.runtime.reproduce.subprocess.check_output",
                      return_value="https://github.com/jang2296/MMDL.git"), \
                patch("mmdl.runtime.reproduce.require_runtime", return_value=({}, None, None)), \
                patch("mmdl.runtime.reproduce._preflight_gpu") as preflight:
            with self.assertRaisesRegex(ValueError, "armed external watchdog"):
                execute(arguments(self.command()), ROOT)
        preflight.assert_not_called()

    def test_legacy_baseline_and_prepare_only_keep_short_safety_windows(self):
        stop = datetime.now(timezone.utc) + timedelta(hours=1)
        with patch.dict(os.environ, {"MMDL_STOP_DEADLINE_UTC": stop.isoformat()}, clear=True):
            args = arguments(["--job-id", "baseline", "--commit", COMMIT])
            self.assertEqual(_full_run_deadline(args, stop), stop)
            prepared = self.plan(self.command("base", "--prepare-only"))
        self.assertIsNone(prepared["enrollment"])
        self.assertEqual([stage["stage"] for stage in prepared["commands"]], ["download"])

    def test_legacy_base_keeps_vllm_lock_and_pip_installation(self):
        with patch.dict(os.environ, {}, clear=True):
            plan = self.plan(["--job-id", "baseline", "--commit", COMMIT])
        self.assertEqual(Path(plan["lock"]).name, "requirements-vllm.lock")
        self.assertNotIn("included_locks_sha256", plan["lock_receipt"])
        self.assertEqual(plan["environment"]["manager"], "pip")
        self.assertEqual(plan["environment"]["commands"]["install"][1:4], ["-m", "pip", "install"])

    def test_proof_environment_identity_includes_nested_vllm_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            proof = Path(temp) / "requirements-proof.lock"
            nested = Path(temp) / "requirements-vllm.lock"
            proof.write_text("-r requirements-vllm.lock\nbitsandbytes==0.48.2\n")
            nested.write_text("vllm==0.11.0\n")
            original = _lock_receipt(proof)
            nested.write_text("vllm==0.12.0\n")
            changed = _lock_receipt(proof)
        self.assertEqual(original["lock_sha256"], changed["lock_sha256"])
        self.assertNotEqual(original["included_locks_sha256"], changed["included_locks_sha256"])

    def test_existing_unmanaged_job_environment_is_never_adopted(self):
        with tempfile.TemporaryDirectory() as temp:
            volume = Path(temp)
            paths = {name: volume / name.lower() for name in (
                "MMDL_VOLUME_ROOT", "HF_HOME", "MMDL_DATA_ROOT", "MMDL_ARTIFACT_ROOT", "MMDL_VENV_ROOT")}
            paths["MMDL_VENV_ROOT"].mkdir()
            env = {"MMDL_COST_POLICY": "user_waived", "MMDL_STORAGE_RESERVE_GIB": "10",
                   "RUNPOD_POD_ID": "fixture"}
            with patch.dict(os.environ, env, clear=True), \
                    patch("mmdl.runtime.reproduce.git_commit"), \
                    patch("mmdl.runtime.reproduce.subprocess.check_output",
                          return_value="https://github.com/jang2296/MMDL.git"), \
                    patch("mmdl.runtime.reproduce.require_runtime", return_value=(paths, None, None)), \
                    patch("mmdl.runtime.reproduce._preflight_gpu", return_value={}), \
                    patch("mmdl.runtime.reproduce.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "unmanaged or different-lock venv"):
                    execute(arguments(self.command()), ROOT)
            self.assertFalse((paths["MMDL_VENV_ROOT"] / "mmdl-lock.json").exists())
            run.assert_not_called()

    def test_plan_rejects_silent_profile_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = Path(temp) / "modified.yaml"
            original = ROOT / "configs/eval/mmmu_pro_standard10_v8_a100.yaml"
            protocol.write_text(original.read_text().replace("batch_size: 9", "batch_size: 8"))
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                self.plan(self.command("base", "--protocol", str(protocol)))


if __name__ == "__main__":
    unittest.main()
