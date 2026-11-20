"""Private training-data bundle: pack locally, stage on a Pod at the IDENTICAL absolute paths.

The frozen proof data (vault, contamination/union_receipt.json, the dataset manifest, approval exports,
preparation/rule files) name each other by absolute path strings, so a Pod must see every file at exactly
the path it had locally. Uploading the bundle to a RunPod volume is an external upload of private
(non-benchmark) training data and needs the user's explicit approval; nothing here uploads anything.

pack   (local)  spec -> <name>.tar + <name>.bundle.json (absolute path, bytes, sha256 per file)
stage  (Pod)    archive + bundle manifest -> files at their absolute paths; refuses an existing target,
                traversal, symlink/hardlink/device members, a symlinked parent and any hash mismatch
verify (Pod)    every staged file is a regular unaliased file with the manifest's size and sha256

CLI: data_bundle pack --spec SPEC.json --out-dir DIR --name NAME
     data_bundle stage --archive TAR --manifest BUNDLE.json [--manifest-sha256 SHA]
     data_bundle verify --manifest BUNDLE.json [--manifest-sha256 SHA] [--archive TAR]
Spec `pod-data-bundle-spec-v1`: {"version": ..., "paths": [absolute file or directory, ...]}; a directory
adds every regular file below it. MMMU/benchmark roots and Hugging Face caches are refused by path (the
contents of a listed file are not inspected: deciding what goes in the spec stays a data-governance step).
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import stat
import sys
import tarfile
from pathlib import Path
from typing import Any

from .common import CHUNK, HEX64, HashingReader, read_json, sha256_file, utc_now, write_json

SPEC_VERSION = "pod-data-bundle-spec-v1"
VERSION = "pod-data-bundle-v1"


class BundleError(ValueError):
    pass


def forbidden_roots() -> list[str]:
    home = os.path.expanduser
    roots = [os.path.join(os.environ.get("MMDL_DATA_ROOT") or home("~/mmdl-data"), "evaluation"),
             os.environ.get("HF_HOME") or home("~/.cache/huggingface"), home("~/mmdl-cache")]
    roots += [os.environ[key] for key in ("HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE")
              if os.environ.get(key)]
    return [os.path.realpath(root) for root in roots]


def refuse_path(path: str) -> None:
    """Raise unless `path` is canonical, absolute, symlink-free and outside benchmark/HF roots."""
    if (not isinstance(path, str) or not path.startswith("/") or posixpath.normpath(path) != path
            or "\\" in path or "\0" in path):
        raise BundleError(f"not a canonical absolute path: {path!r}")
    parts = [part.lower() for part in path.split("/")]
    if any("mmmu" in part for part in parts):
        raise BundleError(f"MMMU/benchmark path refused: {path}")
    if any(part in ("huggingface", "hf-cache") or part.startswith(("datasets--", "models--"))
           for part in parts):
        raise BundleError(f"Hugging Face cache path refused: {path}")
    for root in forbidden_roots():
        if path == root or path.startswith(root + "/"):
            raise BundleError(f"path under a benchmark/cache root ({root}) refused: {path}")


def _regular(path: str) -> os.stat_result:
    """lstat of a regular, unaliased file whose absolute path has no symlinked component."""
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise BundleError(f"not a regular unaliased file (symlink/hardlink/device refused): {path}")
    if os.path.realpath(path) != path:
        raise BundleError(f"a parent directory is a symlink: {path}")
    return info


def expand_spec(spec: Any) -> list[str]:
    if not isinstance(spec, dict) or spec.get("version") != SPEC_VERSION or set(spec) != {
            "version", "paths"} or not isinstance(spec["paths"], list) or not spec["paths"]:
        raise BundleError(f"spec must be {{version: {SPEC_VERSION}, paths: [absolute paths]}}")
    files: set[str] = set()
    for entry in spec["paths"]:
        refuse_path(entry)
        info = os.lstat(entry)
        if stat.S_ISDIR(info.st_mode) and os.path.realpath(entry) == entry:
            for directory, dirnames, names in os.walk(entry):
                for name in dirnames:
                    if os.path.islink(os.path.join(directory, name)):
                        raise BundleError(f"symlinked directory refused: {directory}/{name}")
                files.update(os.path.join(directory, name) for name in names)
        else:
            files.add(entry)
    for path in files:
        refuse_path(path)
        _regular(path)
    return sorted(files)


def pack(spec_path: str | Path, out_dir: str | Path, name: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise BundleError("bundle name must match [A-Za-z0-9._-]+")
    files = expand_spec(read_json(spec_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive, manifest_path = out / f"{name}.tar", out / f"{name}.bundle.json"
    if archive.exists() or manifest_path.exists():
        raise BundleError("archive or bundle manifest already exists; refusing to overwrite")
    partial = out / f".{name}.tar.partial"
    records: list[dict[str, Any]] = []
    try:
        with tarfile.open(partial, "x", format=tarfile.PAX_FORMAT) as tar:
            for path in files:
                info = tar.gettarinfo(path, arcname=path.lstrip("/"))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o600
                with open(path, "rb") as stream:
                    reader = HashingReader(stream)
                    tar.addfile(info, reader)
                if reader.count != info.size:
                    raise BundleError(f"{path}: changed while packing")
                records.append({"path": path, "bytes": info.size,
                                "sha256": reader.digest.hexdigest()})
        os.rename(partial, archive)
    finally:
        partial.unlink(missing_ok=True)  # only the temp file this call created
    manifest = {"version": VERSION, "archive": archive.name,
                "archive_bytes": archive.stat().st_size, "archive_sha256": sha256_file(archive),
                "spec_sha256": sha256_file(spec_path), "files": records,
                "bytes": sum(r["bytes"] for r in records), "created_utc": utc_now(),
                "content": "private training data (not benchmark); public_upload_allowed=false",
                "public_upload_allowed": False}
    write_json(manifest_path, manifest)
    return manifest


def load_manifest(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None:
        if not HEX64.match(expected_sha256) or sha256_file(path) != expected_sha256:
            raise BundleError("bundle manifest sha256 differs from the pinned value")
    manifest = read_json(path)
    if (not isinstance(manifest, dict) or manifest.get("version") != VERSION
            or not isinstance(manifest.get("files"), list) or not manifest["files"]):
        raise BundleError(f"not a {VERSION} manifest with files")
    records = manifest["files"]
    paths = []
    for record in records:
        if (not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}
                or not isinstance(record["bytes"], int) or isinstance(record["bytes"], bool)
                or record["bytes"] < 0 or not HEX64.match(str(record["sha256"]))):
            raise BundleError(f"malformed file record: {record!r}")
        refuse_path(record["path"])
        paths.append(record["path"])
    unique = set(paths)
    if len(unique) != len(paths):
        raise BundleError("duplicate path in the bundle manifest")
    for path in paths:  # a file may not also be a parent directory of another file
        parent = posixpath.dirname(path)
        while parent != "/":
            if parent in unique:
                raise BundleError(f"{parent} is both a file and a directory in the bundle")
            parent = posixpath.dirname(parent)
    return manifest


def _make_parent(parent: str) -> None:
    """Create `parent` only below an existing, symlink-free directory."""
    existing = parent
    while not os.path.lexists(existing):
        existing = posixpath.dirname(existing)
    if os.path.realpath(existing) != existing or not os.path.isdir(existing):
        raise BundleError(f"{existing} is a symlink or not a directory")
    os.makedirs(parent, exist_ok=True)


def stage(archive: str | Path, manifest_path: str | Path,
          manifest_sha256: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, manifest_sha256)
    if sha256_file(archive) != manifest["archive_sha256"]:
        raise BundleError("archive sha256 differs from the bundle manifest")
    records = {record["path"].lstrip("/"): record for record in manifest["files"]}
    existing = [r["path"] for r in records.values() if os.path.lexists(r["path"])]
    if existing:
        raise BundleError(f"{len(existing)} target(s) already exist, nothing staged; "
                          f"first {existing[:3]}")
    created: list[str] = []
    try:
        with tarfile.open(archive, "r") as tar:
            members = tar.getmembers()
            for member in members:
                parts = member.name.split("/")
                if (not member.isreg() or member.name.startswith("/") or ".." in parts
                        or "" in parts or "." in parts or "\\" in member.name):
                    raise BundleError(f"unsafe member refused: {member.name!r}")
            if sorted(m.name for m in members) != sorted(records):
                raise BundleError("archive member set differs from the bundle manifest")
            for member in members:
                record = records[member.name]
                target = record["path"]
                _make_parent(posixpath.dirname(target))
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600)
                created.append(target)
                source = tar.extractfile(member)
                if source is None:
                    raise BundleError(f"unreadable member: {member.name!r}")
                reader = HashingReader(source)
                with os.fdopen(descriptor, "wb") as stream:
                    while data := reader.read(CHUNK):
                        stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if reader.count != record["bytes"] or reader.digest.hexdigest() != record["sha256"]:
                    raise BundleError(f"staged bytes differ from the manifest: {target}")
    except BaseException:
        for path in created:  # only files this call created
            Path(path).unlink(missing_ok=True)
        raise
    return {"status": "STAGED", "files": len(created), "bytes": manifest["bytes"],
            "archive_sha256": manifest["archive_sha256"], "manifest": str(manifest_path)}


def verify(manifest_path: str | Path, manifest_sha256: str | None = None,
           archive: str | Path | None = None,
           covers: tuple[str, ...] = ()) -> list[str]:
    """Every problem with the staged files; empty means all are present and exact.

    `covers`: absolute paths that must be a bundle file or a directory holding bundle files.
    """
    try:
        manifest = load_manifest(manifest_path, manifest_sha256)
    except (BundleError, OSError, ValueError) as exc:
        return [f"manifest: {exc}"]
    errors = []
    if archive is not None and sha256_file(archive) != manifest["archive_sha256"]:
        errors.append("archive sha256 differs from the bundle manifest")
    paths = {record["path"] for record in manifest["files"]}
    for wanted in covers:
        if wanted not in paths and not any(p.startswith(wanted.rstrip("/") + "/") for p in paths):
            errors.append(f"not covered by the bundle: {wanted}")
    for record in manifest["files"]:
        path = record["path"]
        try:
            info = _regular(path)
            if info.st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                errors.append(f"size or sha256 differs: {path}")
        except (BundleError, OSError) as exc:
            errors.append(f"{path}: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--spec", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True)
    for name in ("stage", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", required=True)
        s.add_argument("--manifest-sha256")
        s.add_argument("--archive", required=name == "stage")
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            result = pack(args.spec, args.out_dir, args.name)
            print(json.dumps({k: result[k] for k in ("archive", "archive_sha256", "bytes")}
                             | {"files": len(result["files"]),
                                "manifest_sha256": sha256_file(Path(args.out_dir)
                                                               / f"{args.name}.bundle.json")}))
        elif args.command == "stage":
            print(json.dumps(stage(args.archive, args.manifest, args.manifest_sha256)))
        else:
            errors = verify(args.manifest, args.manifest_sha256, args.archive)
            print(json.dumps({"status": "MISMATCH" if errors else "VERIFIED",
                              "errors": len(errors), "first_errors": errors[:50]}))
            return 1 if errors else 0
        return 0
    except (BundleError, OSError, ValueError, tarfile.TarError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
