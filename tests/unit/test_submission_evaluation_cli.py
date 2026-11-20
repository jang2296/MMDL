"""Public submission profiles and CPU-only stage boundaries."""

import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from mmdl.evaluation import cli


ROOT = Path(__file__).resolve().parents[2]


class SubmissionCLITests(unittest.TestCase):
    def test_four_arm_defaults_preserve_baseline_and_share_pro_profile(self):
        cases = [([], "mmmu_val_v8.yaml"),
                 (["--model", "finetuned"], "mmmu_val_v18b_ft_a100.yaml"),
                 (["--suite", "mmmu-pro"], "mmmu_pro_standard10_v8_a100.yaml"),
                 (["--suite", "mmmu-pro", "--model", "finetuned"], "mmmu_pro_standard10_v8_a100.yaml")]
        for extra, filename in cases:
            with self.subTest(filename=filename):
                _, args = cli.arguments(["--run-id", "synthetic", *extra])
                self.assertEqual(args.protocol.name, filename)
                self.assertEqual(args.score, "v8")
                cfg, _ = cli.load_evaluation_configs(args)
                self.assertEqual(cfg["parser"], "team-final-answer-v8")

    def test_rehearsal_is_explicit_and_never_uses_official_judge(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (["--rehearsal"], ["--checker-mode", "smoke"], ["--score-only"]):
                with self.assertRaises(SystemExit):
                    cli.arguments(["--run-id", "synthetic", *extra])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = yaml.safe_load((ROOT / "configs/eval/mmmu_pro_standard10_v8_a100.yaml").read_text())
            hw = yaml.safe_load((ROOT / "configs/hardware/rtx5060_8gb.yaml").read_text())
            hw["placement"] = "gpu_only"
            cfg["generation"]["max_new_tokens"] = 8
            cfg["execution"].update(batch_size=2, gpu_memory_utilization=0.62)
            protocol, hardware = root / "protocol.yaml", root / "hardware.yaml"
            protocol.write_text(yaml.safe_dump(cfg))
            hardware.write_text(yaml.safe_dump(hw))
            _, args = cli.arguments(["--run-id", "synthetic", "--suite", "mmmu-pro", "--rehearsal",
                                     "--mode", "smoke", "--protocol", str(protocol), "--hardware", str(hardware),
                                     "--model-ref", str(root / "model_ref.json"), "--model-path", str(root / "model"),
                                     "--data-root", str(root / "data")])
            actual, _ = cli.load_evaluation_configs(args)
            self.assertTrue(actual["rehearsal"])
            self.assertTrue(args.no_download)
            self.assertEqual(args.checker_mode, "smoke")
            command = cli._checker_command("select", args)
            self.assertIn("opencompass/CompassVerifier-3B", command)
            broken = copy.deepcopy(cfg)
            broken["generation"]["temperature"] = 0.2
            protocol.write_text(yaml.safe_dump(broken))
            with self.assertRaises(ValueError):
                cli.load_evaluation_configs(args)

    def test_local_5060_rejects_7b_before_any_download(self):
        args = SimpleNamespace(checker_run=None, checker_mode="official", no_download=False)
        with patch.object(cli.subprocess, "run", return_value=SimpleNamespace(stdout="NVIDIA RTX 5060 Laptop GPU")):
            with self.assertRaisesRegex(ValueError, "prohibited"):
                cli._provision_checker(args)

    def test_score_only_never_invokes_generation_or_resolves_weights(self):
        rows = [{"id": "synthetic"}]
        argv = ["--run-id", "synthetic", "--score", "both", "--score-only"]
        with patch("mmdl.evaluation.reporting.saved_run", return_value=(rows, {})), \
                patch.object(cli, "_provision_checker"), patch.object(cli, "_checkset"), \
                patch.object(cli, "score_saved_run", return_value={"reportable": True}) as score, \
                patch.object(cli.subprocess, "run") as process:
            self.assertEqual(cli.main(argv), {"reportable": True})
            score.assert_called_once()
            process.assert_not_called()

    def test_shell_wrappers_work_outside_repository(self):
        for name in ("eval.sh", "eval_finetuned.sh", "rescore.sh"):
            with self.subTest(script=name):
                result = subprocess.run(["bash", str(ROOT / "scripts" / name), "--help"], cwd="/tmp",
                                        env=os.environ | {"MMDL_PYTHON": __import__("sys").executable},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--suite", result.stdout)


if __name__ == "__main__":
    unittest.main()
