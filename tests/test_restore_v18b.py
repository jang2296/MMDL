"""Restoration rejects incomplete inputs and protects existing artifacts before loading Qwen."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fine_tuning.error_note.restore_v18b import restore
from mmdl.runtime.artifacts import sha256_file


class RestoreBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        for name in ("adapter.safetensors", "mergers.safetensors", "runtime.pt", "metadata.json", "metadata.sha256"):
            (checkpoint / name).write_text(name)
        reference = root / "model_ref.json"
        reference.write_text(json.dumps({"files": [{"path": "weights", "bytes": 1024}]}))
        manifest = root / "export_manifest.json"
        manifest.write_text("{}")
        selector = root / "selector.json"
        selector.write_text(json.dumps({"reference_sha256": sha256_file(reference),
                                       "export_manifest_sha256": sha256_file(manifest),
                                       "checkpoint_files": {p.name: sha256_file(p) for p in checkpoint.iterdir()}}))
        self.args = {"checkpoint": checkpoint, "base": root / "base", "reference": reference,
                     "export_manifest": manifest, "output": root / "rebuilt", "receipt": root / "receipt.json",
                     "selector": selector}

    def test_existing_output_and_receipt_are_never_overwritten(self):
        for name in ("output", "receipt"):
            with self.subTest(name=name):
                self.args[name].write_text("keep")
                with self.assertRaises(FileExistsError):
                    restore(**self.args)
                self.assertEqual(self.args[name].read_text(), "keep")
                self.args[name].unlink()

    def test_incomplete_checkpoint_is_rejected_without_model_loading(self):
        (self.args["checkpoint"] / "runtime.pt").unlink()
        with self.assertRaisesRegex(ValueError, "runtime.pt"):
            restore(**self.args)
        self.assertFalse(self.args["output"].exists())

    def test_space_reserve_is_checked_before_model_loading(self):
        with patch("fine_tuning.error_note.restore_v18b.shutil.disk_usage", return_value=SimpleNamespace(free=50 * 1024**3)):
            with self.assertRaisesRegex(ValueError, "50 GiB"):
                restore(**self.args)
        self.assertFalse(self.args["output"].exists())

    def test_output_cannot_overlap_checkpoint(self):
        self.args["output"] = self.args["checkpoint"] / "rebuilt"
        with self.assertRaisesRegex(ValueError, "outside source"):
            restore(**self.args)

    def test_cuda_visibility_must_explicitly_disable_gpu(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(ValueError, "CPU only"):
                restore(**self.args)
        self.assertFalse(self.args["output"].exists())

    def test_historical_manifest_order_is_restored_only_after_full_equality(self):
        original = {"arm": "MOD-RV", "binding": {"first": 1, "second": 2},
                    "checkpoint_state_sha256": {"tensor": "d" * 64}}
        manifest = self.args["export_manifest"]
        manifest.write_text(json.dumps(original))
        anchor = json.loads(self.args["selector"].read_text())
        anchor.update(export_manifest_sha256=sha256_file(manifest), slot=206)
        self.args["selector"].write_text(json.dumps(anchor))
        metadata = {"step": 206, "cursor": {"single": {"phase": "final"}},
                    "state_sha256": original["checkpoint_state_sha256"], "binding": original["binding"]}
        self.args["reference"].write_text(json.dumps({"files": [], "export": {"binding": original["binding"]}}))
        anchor["reference_sha256"] = sha256_file(self.args["reference"])
        self.args["selector"].write_text(json.dumps(anchor))

        def exporter(checkpoint, base, output, arm, **kwargs):
            output.mkdir()
            value = dict(reversed(list(original.items())))
            (output / "export_manifest.json").write_text(json.dumps(value))
            return value

        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}), \
                patch("fine_tuning.notebook_rl.checkpoint.read_metadata", return_value=metadata), \
                patch("fine_tuning.notebook_rl.export.export_merged", side_effect=exporter), \
                patch("fine_tuning.error_note.restore_v18b.resolve_export_reference"):
            self.assertEqual(restore(**self.args)["status"], "PASS")
        rebuilt = self.args["output"] / "model/export_manifest.json"
        self.assertEqual(rebuilt.read_bytes(), manifest.read_bytes())

    def test_manifest_with_different_tensor_hash_cannot_be_relabelled(self):
        original = {"arm": "MOD-RV", "binding": {}, "checkpoint_state_sha256": {"tensor": "d" * 64}}
        manifest = self.args["export_manifest"]
        manifest.write_text(json.dumps(original))
        self.args["reference"].write_text(json.dumps({"files": [], "export": {"binding": {}}}))
        anchor = json.loads(self.args["selector"].read_text())
        anchor.update(export_manifest_sha256=sha256_file(manifest),
                      reference_sha256=sha256_file(self.args["reference"]), slot=206)
        self.args["selector"].write_text(json.dumps(anchor))
        metadata = {"step": 206, "cursor": {"single": {"phase": "final"}},
                    "state_sha256": original["checkpoint_state_sha256"], "binding": {}}

        def exporter(checkpoint, base, output, arm, **kwargs):
            output.mkdir()
            value = original | {"checkpoint_state_sha256": {"tensor": "e" * 64}}
            (output / "export_manifest.json").write_text(json.dumps(value))
            return value

        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}), \
                patch("fine_tuning.notebook_rl.checkpoint.read_metadata", return_value=metadata), \
                patch("fine_tuning.notebook_rl.export.export_merged", side_effect=exporter):
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                restore(**self.args)
        self.assertNotEqual((self.args["output"] / "model/export_manifest.json").read_bytes(), manifest.read_bytes())
        self.assertFalse((self.args["output"] / "model_ref.json").exists())


if __name__ == "__main__":
    unittest.main()
