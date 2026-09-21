from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmdl.runtime.bundle import _MANIFEST, _manifest_from_archive, cleanup_targets, create, verify
from mmdl.runtime.artifacts import sha256_file


class BundleTests(unittest.TestCase):
    def _roots(self, base: Path) -> tuple[Path, Path]:
        artifacts, public = base / "artifacts", base / "public"
        for role in ("assignment", "analysis"):
            (artifacts / "runs" / f"job-{role}").mkdir(parents=True)
            (public / f"job-{role}").mkdir(parents=True)
            (artifacts / "runs" / f"job-{role}" / "summary.json").write_text("{}")
            (public / f"job-{role}" / "summary.json").write_text("{}")
        (artifacts / "jobs").mkdir()
        (artifacts / "jobs" / "job.json").write_text("{}")
        return artifacts, public

    def test_create_is_allowlisted_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, public = self._roots(Path(temp))
            archive = Path(temp) / "bundle.tar.gz"
            with patch("mmdl.runtime.bundle._cross_run_checks", return_value={"status": "REMOTE_VERIFIED", "total_inference_records": 1800}):
                manifest = create(artifacts, public, "job", archive)
            self.assertEqual(manifest["job_id"], "job")
            self.assertTrue(archive.is_file())
            self.assertEqual(sha256_file(archive), manifest["archive_sha256"])
            with tarfile.open(archive, "r:gz") as stream:
                names = {member.name for member in stream.getmembers()}
            self.assertIn(_MANIFEST, names)
            self.assertIn("runs/job-assignment/summary.json", names)
            self.assertIn("public/job-analysis/summary.json", names)
            self.assertIn("jobs/job.json", names)
            with self.assertRaises(FileExistsError):
                with patch("mmdl.runtime.bundle._cross_run_checks", return_value={"status": "REMOTE_VERIFIED", "total_inference_records": 1800}):
                    create(artifacts, public, "job", archive)

    def test_forbidden_model_cache_entry_is_not_packable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, public = self._roots(Path(temp))
            forbidden = artifacts / "runs" / "job-assignment" / "models" / "weight.safetensors"
            forbidden.parent.mkdir()
            forbidden.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                create(artifacts, public, "job", Path(temp) / "bundle.tar")

    def test_malformed_or_unlisted_member_is_rejected_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive = base / "unsafe.tar"
            listed = {"path": "runs/job-assignment/summary.json", "bytes": 2,
                      "sha256": hashlib.sha256(b"{}").hexdigest()}
            manifest = {"schema": "mmdl-runpod-bundle-v1", "job_id": "job",
                        "runs": {"assignment": "job-assignment", "analysis": "job-analysis"},
                        "files": [listed]}
            with tarfile.open(archive, "w") as stream:
                payload = json.dumps(manifest).encode()
                info = tarfile.TarInfo(_MANIFEST)
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
                file_info = tarfile.TarInfo(listed["path"])
                file_info.size = 2
                stream.addfile(file_info, io.BytesIO(b"{}"))
                extra = tarfile.TarInfo("unlisted.txt")
                extra.size = 1
                stream.addfile(extra, io.BytesIO(b"x"))
            destination = base / "destination"
            with self.assertRaisesRegex(ValueError, "allowlist"):
                verify(archive, sha256_file(archive), destination, base)
            self.assertFalse(destination.exists())

    def test_archive_digest_mismatch_refuses_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "empty.tar"
            with tarfile.open(archive, "w"):
                pass
            destination = Path(temp) / "destination"
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify(archive, "0" * 64, destination, Path(temp))
            self.assertFalse(destination.exists())

    def test_traversal_and_links_are_rejected_by_manifest_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "traversal.tar"
            with tarfile.open(archive, "w") as stream:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))
            with tarfile.open(archive, "r") as stream:
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    _manifest_from_archive(stream)
            link_archive = Path(temp) / "link.tar"
            with tarfile.open(link_archive, "w") as stream:
                link = tarfile.TarInfo("runs/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/outside"
                stream.addfile(link)
            with tarfile.open(link_archive, "r") as stream:
                with self.assertRaisesRegex(ValueError, "regular files"):
                    _manifest_from_archive(stream)

    def test_cleanup_targets_allow_only_receipted_owned_resources(self) -> None:
        receipt = {"status": "LOCAL_VERIFIED", "job_id": "job-1", "pod_id": "pod-1",
                   "runs": {"total_inference_records": 1800}}
        ledger = {"job_id": "job-1", "pod_id": "pod-1", "created_for_job": True,
                  "preexisting_pod_ids": [], "preexisting_volume_ids": [],
                  "network_volumes": [{"id": "volume-1", "created_for_job": "job-1", "shared": False}]}
        self.assertEqual(cleanup_targets(receipt, ledger, "pod-1")["network_volume_ids"], ["volume-1"])
        ledger["network_volumes"][0]["shared"] = True
        with self.assertRaisesRegex(ValueError, "shared"):
            cleanup_targets(receipt, ledger, "pod-1")


if __name__ == "__main__":
    unittest.main()
