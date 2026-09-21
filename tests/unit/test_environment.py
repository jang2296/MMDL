from __future__ import annotations

import unittest
import os
from pathlib import Path
from unittest.mock import patch

from mmdl.runtime import environment


class DiskUsage:
    def __init__(self, free: int) -> None:
        self.free = free


class StorageTests(unittest.TestCase):
    def test_wsl_requires_both_filesystem_and_backing_volume_space(self) -> None:
        with (
            patch.object(environment, "_is_wsl", return_value=True),
            patch.object(environment.shutil, "disk_usage", return_value=DiskUsage(environment.RESERVE_BYTES + 9)),
            patch.object(environment, "_windows_backing_volume", return_value={"checked": True, "free_bytes": environment.RESERVE_BYTES + 8}),
        ):
            result = environment.check_storage([Path("/safe")], required_bytes=9)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["required_bytes"], 9)
        self.assertEqual(result["reserve_bytes"], environment.RESERVE_BYTES)

    def test_non_wsl_filesystem_space_is_sufficient(self) -> None:
        required = environment.RESERVE_BYTES + 100
        with (
            patch.object(environment, "_is_wsl", return_value=False),
            patch.object(environment.shutil, "disk_usage", return_value=DiskUsage(required)),
        ):
            result = environment.check_storage([Path("/safe")], required_bytes=100)
        self.assertTrue(result["sufficient"])
        self.assertTrue(result["windows_backing_volume"]["not_applicable"])

    def test_negative_requirement_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            environment.check_storage([Path("/safe")], required_bytes=-1)

    def test_remote_reserve_is_explicit_but_wsl_keeps_local_floor(self) -> None:
        with patch.dict(os.environ, {"MMDL_STORAGE_RESERVE_GIB": "10"}), \
                patch.object(environment, "_is_wsl", return_value=False), \
                patch.object(environment.shutil, "disk_usage", return_value=DiskUsage(20 * environment.GIB)):
            self.assertTrue(environment.check_storage([Path("/safe")])["sufficient"])
        with patch.dict(os.environ, {"MMDL_STORAGE_RESERVE_GIB": "10"}), \
                patch.object(environment, "_is_wsl", return_value=True), \
                patch.object(environment.shutil, "disk_usage", return_value=DiskUsage(20 * environment.GIB)), \
                patch.object(environment, "_windows_backing_volume", return_value={"checked": True, "free_bytes": 100 * environment.GIB}):
            result = environment.check_storage([Path("/safe")])
            self.assertFalse(result["sufficient"])
            self.assertEqual(result["reserve_bytes"], 50 * environment.GIB)
        with patch.dict(os.environ, {"MMDL_STORAGE_RESERVE_GIB": "nan"}):
            with self.assertRaises(ValueError):
                environment.check_storage([Path("/safe")])


if __name__ == "__main__":
    unittest.main()
