"""The public split format preserves original bytes and fails closed on tampering."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v18_recipe as recipe
from scripts.submission_publication import encoded, pack_recipe


class RecipePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "fine_tuning/error_note"
        self.source.mkdir(parents=True)
        self.cache = self.root / "external-cache"
        self.original = {}
        self.document("v18_recipe_items_00.json", {"version": "fixture-items", "items": []})
        self.document("v18_recipe_metadata_00.json", {"version": "fixture-metadata",
            "items": [{"identifier": "a", "opaque": "x" * 1100000},
                      {"identifier": "b", "opaque": "y" * 1100000}]})
        self.document("v18_recipe_sidecars.json", {"version": "fixture-sidecars",
            "item_shards": [{"path": "v18_recipe_metadata_00.json"}]})
        self.document("v18_recipe.json", {"version": "fixture-recipe", "sidecars": {
            "path": "v18_recipe_sidecars.json"}, "item_shards": [self.reference("v18_recipe_items_00.json")]})
        self.public = pack_recipe(self.root)
        (self.source / "v18_recipe_metadata_00.json").unlink()
        self.validation = patch.object(recipe, "validate_recipe")
        self.sidecars = patch.object(recipe, "load_sidecars")
        self.validation.start()
        self.sidecars.start()

    def tearDown(self):
        self.sidecars.stop()
        self.validation.stop()
        self.temporary.cleanup()

    def document(self, name, value):
        value["content_sha256"] = recipe.digest(value)
        data = encoded(value)
        (self.source / name).write_bytes(data)
        self.original[name] = data

    def reference(self, name):
        data = self.original[name]
        value = json.loads(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                "content_sha256": value["content_sha256"], "count": len(value["items"])}

    def restore(self):
        return recipe.restore_package(self.source / recipe.PACKAGE_NAME, self.cache)

    def test_restore_exact_originals_outside_source_and_verify_reuse(self):
        path = self.restore()
        self.assertTrue(path.is_relative_to(self.cache))
        for name, data in self.original.items():
            self.assertEqual((path.parent / name).read_bytes(), data)
        self.assertEqual(self.restore(), path)
        self.assertEqual(len([x for x in self.public if "_part_" in x]), 2)
        self.assertTrue(all((self.root / x).stat().st_size < 2 * 1024**2 for x in self.public))

    def test_corrupted_part_rejected_even_after_successful_restoration(self):
        self.restore()
        part = next(self.source.glob("*_part_0.json"))
        part.write_bytes(part.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "RECIPE_PACKAGE_FILE_CHANGED"):
            self.restore()

    def test_corrupted_external_cache_rejected_without_overwrite(self):
        restored = self.restore().parent / "v18_recipe_metadata_00.json"
        restored.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "RECIPE_PACKAGE_FILE_CHANGED"):
            self.restore()
        self.assertEqual(restored.read_bytes(), b"changed")

    def test_resolver_uses_configured_external_cache(self):
        with patch.dict(os.environ, {"MMDL_RECIPE_CACHE": str(self.cache)}):
            resolved = recipe.resolve_recipe_path(self.source / "v18_recipe.json")
        self.assertEqual(resolved, self.restore())

    def test_package_paths_and_private_payload_rejected(self):
        package_path = self.source / recipe.PACKAGE_NAME
        value = json.loads(package_path.read_text())
        value["files"][0]["path"] = "../outside.json"
        value["content_sha256"] = recipe.digest({k: v for k, v in value.items() if k != "content_sha256"})
        package_path.write_bytes(encoded(value))
        with self.assertRaisesRegex(ValueError, "RECIPE_PACKAGE_UNSAFE_PATH"):
            self.restore()
        part = next(self.source.glob("*_part_0.json"))
        self.assertEqual(len(hashlib.sha256(part.read_bytes()).hexdigest()), 64)
        with self.assertRaisesRegex(ValueError, "RECIPE_PRIVATE_PAYLOAD"):
            recipe._public({"raw_response": "private"})


if __name__ == "__main__":
    unittest.main()
