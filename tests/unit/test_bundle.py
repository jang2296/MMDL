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
    def _roots(self, base: Path, *, suite: str | None = None) -> tuple[Path, Path]:
        artifacts, public = base / "artifacts", base / "public"
        roles_by_suite = {
            None: ("assignment", "analysis"),
            "mmmu-val": ("evaluation",),
            "mmmu-val-assignment": ("assignment",),
            "mmmu-val-analysis": ("analysis",),
        }
        roles = roles_by_suite[suite]
        for role in roles:
            (artifacts / "runs" / f"job-{role}").mkdir(parents=True)
            (public / f"job-{role}").mkdir(parents=True)
            (artifacts / "runs" / f"job-{role}" / "summary.json").write_text("{}")
            (public / f"job-{role}" / "summary.json").write_text("{}")
        (artifacts / "jobs").mkdir()
        job = {"runs": {role: f"job-{role}" for role in roles}}
        if suite is not None:
            job |= {"schema_version": 2, "suite": suite}
        (artifacts / "jobs" / "job.json").write_text(json.dumps(job))
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

    def test_single_role_bundle_contains_only_the_explicit_suite_role(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, public = self._roots(Path(temp), suite="mmmu-val-analysis")
            archive = Path(temp) / "analysis.tar.gz"
            evidence = {"status": "REMOTE_VERIFIED", "roles": ["analysis"],
                        "total_inference_records": 900}
            with patch("mmdl.runtime.bundle._cross_run_checks", return_value=evidence):
                result = create(artifacts, public, "job", archive)
            self.assertEqual(result["total_inference_records"], 900)
            with tarfile.open(archive, "r:gz") as stream:
                manifest, _ = _manifest_from_archive(stream)
            self.assertEqual(manifest["schema"], "mmdl-runpod-bundle-v2")
            self.assertEqual(manifest["roles"], ["analysis"])
            self.assertEqual(manifest["runs"], {"analysis": "job-analysis"})

    def test_single_evaluation_bundle_creates_verifies_and_guards_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            artifacts, public = self._roots(base, suite="mmmu-val")
            archive = base / "evaluation.tar.gz"
            evidence = {"status": "REMOTE_VERIFIED", "bundle_schema": "mmdl-runpod-bundle-v2",
                        "suite": "mmmu-val", "roles": ["evaluation"], "pod_id": "pod-1",
                        "total_inference_records": 900,
                        "evaluation": {"run_id": "job-evaluation", "count": 900}}
            with patch("mmdl.runtime.bundle._cross_run_checks", return_value=evidence):
                result = create(artifacts, public, "job", archive)
                receipt = verify(archive, sha256_file(archive), base / "recovered", base)
            with tarfile.open(archive, "r:gz") as stream:
                manifest, _ = _manifest_from_archive(stream)
            self.assertEqual(result["total_inference_records"], 900)
            self.assertEqual(manifest["schema"], "mmdl-runpod-bundle-v2")
            self.assertEqual(manifest["suite"], "mmmu-val")
            self.assertEqual(manifest["roles"], ["evaluation"])
            self.assertEqual(manifest["runs"], {"evaluation": "job-evaluation"})
            self.assertEqual(receipt["roles"], ["evaluation"])

            ledger = {"job_id": "job", "pod_id": "pod-1", "created_for_job": True,
                      "preexisting_pod_ids": [], "preexisting_volume_ids": [],
                      "network_volumes": [{"id": "volume-1", "created_for_job": "job", "shared": False}]}
            self.assertEqual(cleanup_targets(receipt, ledger, "pod-1")["network_volume_ids"], ["volume-1"])
            ledger["network_volumes"][0]["shared"] = True
            with self.assertRaisesRegex(ValueError, "shared"):
                cleanup_targets(receipt, ledger, "pod-1")

    def test_schema_v2_rejects_missing_or_extra_suite_roles(self) -> None:
        for runs in ({}, {"analysis": "job-analysis", "assignment": "job-assignment"}):
            with self.subTest(runs=runs), tempfile.TemporaryDirectory() as temp:
                artifacts, public = self._roots(Path(temp), suite="mmmu-val-analysis")
                (artifacts / "jobs" / "job.json").write_text(json.dumps({
                    "schema_version": 2, "suite": "mmmu-val-analysis", "runs": runs,
                }))
                with self.assertRaisesRegex(ValueError, "exact suite run roles"):
                    create(artifacts, public, "job", Path(temp) / "bundle.tar")

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

    def test_legacy_create_verify_receipt_remains_cleanup_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            artifacts, public = self._roots(base)
            archive = base / "legacy.tar.gz"
            roles = ["assignment", "analysis"]
            evidence = {"status": "REMOTE_VERIFIED", "bundle_schema": "mmdl-runpod-bundle-v1",
                        "suite": "mmmu-val-two-runs", "roles": roles, "pod_id": "pod-1",
                        "total_inference_records": 1800,
                        **{role: {"run_id": f"job-{role}", "count": 900} for role in roles}}
            with patch("mmdl.runtime.bundle._cross_run_checks", return_value=evidence):
                create(artifacts, public, "job", archive)
                receipt = verify(archive, sha256_file(archive), base / "recovered", base)
            ledger = {"job_id": "job", "pod_id": "pod-1", "created_for_job": True,
                      "preexisting_pod_ids": [], "preexisting_volume_ids": [], "network_volumes": []}
            self.assertEqual(receipt["bundle_schema"], "mmdl-runpod-bundle-v1")
            self.assertEqual(cleanup_targets(receipt, ledger, "pod-1")["pod_id"], "pod-1")

    def test_cleanup_accepts_complete_single_role_but_rejects_another_pod(self) -> None:
        evidence = {"roles": ["analysis"], "total_inference_records": 900,
                    "analysis": {"run_id": "job-1-analysis", "count": 900}}
        receipt = {"status": "LOCAL_VERIFIED", "bundle_schema": "mmdl-runpod-bundle-v2",
                   "suite": "mmmu-val-analysis", "roles": ["analysis"],
                   "expected_inference_records": 900, "job_id": "job-1", "pod_id": "pod-1",
                   "runs": evidence}
        ledger = {"job_id": "job-1", "pod_id": "pod-1", "created_for_job": True,
                  "preexisting_pod_ids": [], "preexisting_volume_ids": [], "network_volumes": []}
        self.assertEqual(cleanup_targets(receipt, ledger, "pod-1")["pod_id"], "pod-1")
        with self.assertRaisesRegex(ValueError, "bind the requested Pod"):
            cleanup_targets(receipt, ledger, "old-pod")

    def test_cleanup_rejects_single_role_count_mismatch(self) -> None:
        evidence = {"roles": ["analysis"], "total_inference_records": 899,
                    "analysis": {"run_id": "job-1-analysis", "count": 899}}
        receipt = {"status": "LOCAL_VERIFIED", "bundle_schema": "mmdl-runpod-bundle-v2",
                   "suite": "mmmu-val-analysis", "roles": ["analysis"],
                   "expected_inference_records": 900, "job_id": "job-1", "pod_id": "pod-1",
                   "runs": evidence}
        ledger = {"job_id": "job-1", "pod_id": "pod-1", "created_for_job": True,
                  "preexisting_pod_ids": [], "preexisting_volume_ids": [], "network_volumes": []}
        with self.assertRaisesRegex(ValueError, "exact complete suite roles"):
            cleanup_targets(receipt, ledger, "pod-1")


if __name__ == "__main__":
    unittest.main()
