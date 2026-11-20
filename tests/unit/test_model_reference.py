"""Local reference binding: historical identity, fresh runs, and explicit rehearsals."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mmdl.runtime.artifacts import digest, sha256_file
from mmdl.runtime.model_reference import (
    BASE_ID, BASE_MANIFEST_SHA256, BASE_REVISION, read_reference, resolve_export_reference,
)


def fixture(root):
    from fine_tuning.notebook_rl.export import FORMAT_LORA, LORA_FILES, PRECISION_LORA

    model = root / "model"
    model.mkdir()
    for name in (*LORA_FILES, "config.json"):
        path = model / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("authored CPU fixture\n")
    checkpoint = {"metadata.json": "c" * 64}
    manifest = {
        "status": "EXPORTED", "format": FORMAT_LORA, "arm": "MOD-RV",
        "base_id": BASE_ID, "base_revision": BASE_REVISION,
        "precision_policy": PRECISION_LORA, "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "checkpoint_files": checkpoint, "checkpoint_state_sha256": {"tensor": "d" * 64},
        "binding": {"protocol_sha256": "e" * 64}, "resolved_protocol_sha256": "f" * 64,
        "mergers": ["model.visual.merger"],
        "files": {p.relative_to(model).as_posix(): sha256_file(p)
                  for p in model.rglob("*") if p.is_file()},
    }
    (model / "export_manifest.json").write_text(json.dumps(manifest))
    ref = {
        "kind": "unmerged-lora", "base": {"id": BASE_ID, "revision": BASE_REVISION},
        "dtype": "bfloat16", "processor_revision": BASE_REVISION, "usage": "evaluation_only",
        "artifact_revision": sha256_file(model / "export_manifest.json"),
        "train_config_sha256": manifest["resolved_protocol_sha256"],
        "files": [{"path": p.relative_to(model).as_posix(), "bytes": p.stat().st_size,
                   "sha256": sha256_file(p)} for p in sorted(model.rglob("*")) if p.is_file()],
        "export": {key: manifest[key] for key in
                   ("arm", "checkpoint_files", "binding", "base_manifest_sha256", "mergers", "precision_policy")},
    }
    ref["export"].update(export_manifest_sha256=digest(manifest),
                         checkpoint_state_sha256=digest(manifest["checkpoint_state_sha256"]),
                         campaign={"slot": 206, "stage": "final"})
    path = root / "model_ref.json"
    path.write_text(json.dumps(ref))
    selector = {"kind": "export-reference", "slot": 206,
                "reference_sha256": sha256_file(path),
                "export_manifest_sha256": ref["artifact_revision"],
                "checkpoint_files": checkpoint}
    selector_path = root / "selector.json"
    selector_path.write_text(json.dumps(selector))
    return path, selector_path, selector


class ModelReferenceTests(unittest.TestCase):
    def test_historical_env_and_explicit_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, selector, _ = fixture(root)
            with patch.dict("os.environ", {"MMDL_V18B_EXPORT_ROOT": str(root)}):
                self.assertEqual(resolve_export_reference(selector)[:2], (ref, root / "model"))
            self.assertEqual(resolve_export_reference(selector, root / "model")[:2], (ref, root / "model"))

    def test_retrained_reference_uses_own_inventory_and_no_historical_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, _ = fixture(root)
            ref = read_reference(path)
            ref["export"]["campaign"] = {"slot": 42, "stage": "final"}
            path.write_text(json.dumps(ref))
            self.assertEqual(resolve_export_reference(path)[2]["export"]["campaign"]["slot"], 42)

    def test_wrong_reference_reports_expected_actual_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, selector, _ = fixture(Path(tmp))
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "expected SHA256.*actual"):
                resolve_export_reference(selector, Path(tmp))

    def test_missing_adapter_extra_file_and_changed_tensor_rejected(self):
        for operation in ("missing", "extra", "changed"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, _, _ = fixture(root)
                adapter = root / "model/vision_lora/vision_lora.safetensors"
                if operation == "missing":
                    adapter.unlink()
                elif operation == "extra":
                    (root / "model/unexpected.txt").write_text("extra")
                else:
                    adapter.write_text("wrong run")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    resolve_export_reference(path)

    def test_rehearsal_requires_separate_bound_nonreportable_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, selector_path, selector = fixture(root)
            selector.update(kind="derived-rehearsal", reportable=False)
            selector_path.write_text(json.dumps(selector))
            self.assertIs(resolve_export_reference(selector_path, root)[2]["reportable"], False)
            selector["reportable"] = True
            selector_path.write_text(json.dumps(selector))
            with self.assertRaisesRegex(ValueError, "nonreportable"):
                resolve_export_reference(selector_path, root)

    def test_duplicate_json_and_missing_export_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, selector, _ = fixture(Path(tmp))
            with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "MMDL_V18B_EXPORT_ROOT"):
                resolve_export_reference(selector)
            path.write_text('{"kind":"unmerged-lora", "kind":"base"}')
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                read_reference(path)


if __name__ == "__main__":
    unittest.main()
