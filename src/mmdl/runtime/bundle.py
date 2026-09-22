"""Create and locally verify a closed, portable RunPod evaluation bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json
from mmdl.runtime.environment import check_storage


_MANIFEST = "bundle_manifest.json"
_EVALUATION_ROLES = ("evaluation",)
_LEGACY_ROLES = ("assignment", "analysis")
_SUITE_ROLES = {
    "mmmu-val": _EVALUATION_ROLES,
    "mmmu-val-two-runs": _LEGACY_ROLES,
    "mmmu-val-assignment": ("assignment",),
    "mmmu-val-analysis": ("analysis",),
}
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_PARTS = frozenset({"model", "models", "venv", "venvs", "hf", "huggingface", "cache", "caches",
                              "benchmark", "benchmarks", "datasets"})
_FORBIDDEN_SUFFIXES = frozenset({".bin", ".ckpt", ".pt", ".pth", ".safetensors"})


def _sha256_stream(stream: Any) -> str:
    hashed = hashlib.sha256()
    while data := stream.read(1024 * 1024):
        hashed.update(data)
    return hashed.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == ".":
        raise ValueError(f"Unsafe bundle path: {value!r}")
    return path


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Required directory is absent or unsafe: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted in bundles: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"Unsupported bundle entry: {path}")
    return files


def _add_tree(entries: dict[str, Path], source: Path, prefix: PurePosixPath) -> None:
    for path in _regular_files(source):
        relative = prefix / PurePosixPath(path.relative_to(source).as_posix())
        name = str(relative)
        if _FORBIDDEN_PARTS & set(relative.parts) or path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ValueError(f"Forbidden model, environment, or benchmark-cache entry: {name}")
        if name in entries:
            raise ValueError(f"Duplicate bundle path: {name}")
        entries[name] = path


def _image_sources(run_dir: Path) -> dict[str, Path]:
    """Index each referenced image by content hash without touching immutable rows."""
    result: dict[str, Path] = {}
    for sample in sorted((run_dir / "samples").glob("*.json")):
        row = read_json(sample)
        content = row.get("messages", [{}])[0].get("content", [])
        for item in content[:-1]:
            if not isinstance(item, dict) or item.get("type") != "image":
                raise ValueError(f"Invalid image message in {sample.name}")
            relative = item.get("image")
            declared = item.get("sha256")
            if not isinstance(relative, str) or not isinstance(declared, str):
                raise ValueError(f"Image lacks path or hash in {sample.name}")
            path = PurePosixPath(relative)
            if path.parts[:3] == ("..", "..", "assets") and len(path.parts) == 5 and path.parts[3] == "images":
                if path.name != f"{declared}.png":
                    raise ValueError(f"Shared image filename/hash mismatch in {sample.name}")
                local = run_dir.parent.parent / "assets" / "images" / path.name
            else:
                local = run_dir / _safe_relative(relative)
            if not local.is_file() or local.is_symlink() or sha256_file(local) != declared:
                raise ValueError(f"Image integrity failure: {sample.name}")
            prior = result.setdefault(declared, local)
            if sha256_file(prior) != declared:
                raise ValueError(f"Conflicting image source for {declared}")
    return result


def _tar_mode(path: Path, writing: bool) -> Literal["w", "w:gz", "r", "r:gz"]:
    compressed = path.suffix in {".gz", ".tgz"} or path.name.endswith(".tar.gz")
    return ("w:gz" if writing else "r:gz") if compressed else ("w" if writing else "r")


def _write_member(archive: tarfile.TarFile, name: str, path: Path) -> None:
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = 0o600
    with path.open("rb") as stream:
        archive.addfile(info, stream)


def _check_storage(path: Path, required_bytes: int) -> None:
    storage = check_storage([path], required_bytes)
    if not storage["sufficient"]:
        raise RuntimeError("Bundle storage headroom is below the recorded reserve + required bytes")


def _publish_exclusive(path: Path, source: Path) -> None:
    try:
        os.link(source, path)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite output: {path}") from exc
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _job_contract(job_id: str, metadata: dict[str, Any]) -> tuple[str, str, tuple[str, ...], dict[str, str]]:
    """Resolve legacy paired jobs or the explicit role set of schema-v2 jobs."""
    version = metadata.get("schema_version")
    schema: str
    suite: str
    roles: tuple[str, ...]
    if version is None:
        schema, suite, roles = "mmdl-runpod-bundle-v1", "mmmu-val-two-runs", _LEGACY_ROLES
    elif version == 2:
        suite_value = metadata.get("suite")
        if not isinstance(suite_value, str) or suite_value not in _SUITE_ROLES:
            raise ValueError("Schema-v2 job has an unsupported evaluation suite")
        suite = suite_value
        schema, roles = "mmdl-runpod-bundle-v2", _SUITE_ROLES[suite]
    else:
        raise ValueError("Unsupported job metadata schema")
    runs = {role: f"{job_id}-{role}" for role in roles}
    if metadata.get("runs") != runs:
        raise ValueError("Job metadata does not bind the exact suite run roles")
    return schema, suite, roles, runs


def create(artifact_root: Path, public_root: Path, job_id: str, output: Path,
           repository: Path | None = None) -> dict[str, Any]:
    """Build one no-symlink archive for exactly the roles bound to the job."""
    artifact_root, public_root, output = map(Path, (artifact_root, public_root, output))
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("Job ID is unsafe")
    if output.exists() or output.is_symlink() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"Refusing to overwrite bundle output: {output}")
    job_file = artifact_root / "jobs" / f"{job_id}.json"
    if not job_file.is_file() or job_file.is_symlink():
        raise FileNotFoundError(f"Exact job metadata is required: {job_file}")
    job_metadata = read_json(job_file)
    schema, suite, roles, run_names = _job_contract(job_id, job_metadata)
    entries: dict[str, Path] = {}
    for role, run_name in run_names.items():
        run_dir = artifact_root / "runs" / run_name
        _add_tree(entries, run_dir, PurePosixPath("runs") / run_name)
        _add_tree(entries, public_root / run_name, PurePosixPath("public") / run_name)
    entries[f"jobs/{job_id}.json"] = job_file
    for directory, prefix in ((artifact_root / "jobs", "jobs"), (artifact_root / "setup", "setup")):
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.glob(f"{job_id}.*")):
            if path == job_file:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unsafe exact-job evidence: {path}")
            entries[f"{prefix}/{path.name}"] = path
    images: dict[str, Path] = {}
    for run_name in run_names.values():
        for image_hash, image in _image_sources(artifact_root / "runs" / run_name).items():
            if image_hash in images and sha256_file(images[image_hash]) != sha256_file(image):
                raise ValueError(f"Hash collision for image {image_hash}")
            images[image_hash] = image
    for image_hash, image in sorted(images.items()):
        entries[f"assets/images/{image_hash}.png"] = image
    remote_evidence = _cross_run_checks(artifact_root, public_root, job_id, repository or Path.cwd())
    files = [
        {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(entries.items())
    ]
    manifest = {"schema": schema, "job_id": job_id, "suite": suite, "roles": list(roles), "runs": run_names,
                "remote_verification": remote_evidence, "files": files}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    _check_storage(output.parent, sum(item["bytes"] for item in files))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        with tarfile.open(temporary, _tar_mode(output, True), format=tarfile.PAX_FORMAT) as archive:
            info = tarfile.TarInfo(_MANIFEST)
            info.size = len(manifest_bytes)
            info.mode = 0o600
            archive.addfile(info, fileobj=io.BytesIO(manifest_bytes))
            for name, path in sorted(entries.items()):
                _write_member(archive, name, path)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _publish_exclusive(output, temporary)
    finally:
        if temporary.exists():
            temporary.unlink()
    archive_hash = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    descriptor, sidecar_temp_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", dir=sidecar.parent)
    sidecar_temp = Path(sidecar_temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{archive_hash}  {output.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_exclusive(sidecar, sidecar_temp)
    finally:
        if sidecar_temp.exists():
            sidecar_temp.unlink()
    return {"status": "REMOTE_VERIFIED", "job_id": job_id, "archive_sha256": archive_hash,
            "file_count": len(files), "total_inference_records": remote_evidence["total_inference_records"]}


def _manifest_from_archive(archive: tarfile.TarFile) -> tuple[dict[str, Any], dict[str, tarfile.TarInfo]]:
    members = archive.getmembers()
    names: set[str] = set()
    by_name: dict[str, tarfile.TarInfo] = {}
    for member in members:
        _safe_relative(member.name)
        if member.name in names:
            raise ValueError(f"Duplicate archive member: {member.name}")
        names.add(member.name)
        if member.issym() or member.islnk() or not member.isreg():
            raise ValueError(f"Only regular files are permitted: {member.name}")
        by_name[member.name] = member
    if _MANIFEST not in by_name:
        raise ValueError("Bundle manifest is absent")
    stream = archive.extractfile(by_name[_MANIFEST])
    if stream is None:
        raise ValueError("Bundle manifest is unreadable")
    manifest = json.loads(stream.read())
    if not isinstance(manifest, dict) or manifest.get("schema") not in {
        "mmdl-runpod-bundle-v1", "mmdl-runpod-bundle-v2"
    }:
        raise ValueError("Unsupported bundle manifest")
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise ValueError("Bundle file manifest is invalid")
    expected: dict[str, dict[str, Any]] = {}
    for entry in listed:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid file manifest entry")
        name = str(_safe_relative(entry["path"]))
        if name in expected or not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise ValueError("Duplicate or invalid file manifest entry")
        if not isinstance(entry["sha256"], str) or not _SHA256.fullmatch(entry["sha256"]):
            raise ValueError("Invalid file digest")
        expected[name] = entry
    if set(by_name) != set(expected) | {_MANIFEST}:
        raise ValueError("Archive members do not exactly match the allowlist")
    for name, entry in expected.items():
        member = by_name[name]
        if member.size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        stream = archive.extractfile(member)
        if stream is None or _sha256_stream(stream) != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {name}")
    return manifest, by_name


def _verify_checkout(repository: Path, commit: str, records: list[dict[str, Any]]) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Run does not record a full Git commit")
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True)
    if current.returncode != 0 or current.stdout.strip() != commit:
        raise ValueError("Verifier checkout does not match the recorded Git commit")
    for item in records:
        relative = _safe_relative(item["path"])
        path = repository / relative
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]):
            raise ValueError(f"Verifier code differs from recorded run code: {relative}")


def _cross_run_checks(artifact_root: Path, public_root: Path, job_id: str,
                      repository: Path) -> dict[str, Any]:
    validate_path = repository / "scripts" / "validate_run.py"
    if not validate_path.is_file():
        raise FileNotFoundError(f"Validation script is absent: {validate_path}")
    validate = runpy.run_path(str(validate_path))["validate"]
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("Job ID is unsafe")
    job_path = artifact_root / "jobs" / f"{job_id}.json"
    job_metadata = read_json(job_path)
    schema, suite, roles, runs = _job_contract(job_id, job_metadata)
    pod_id = job_metadata.get("pod_id")
    if (job_metadata.get("job_id") != job_id or job_metadata.get("status") != "REMOTE_VERIFIED"
            or job_metadata.get("logs_closed") is not True or not isinstance(pod_id, str) or not pod_id):
        raise ValueError("Exact job metadata is not remotely verified with closed logs")
    evidence: dict[str, Any] = {}
    identities: dict[str, dict[str, Any]] = {}
    id_sets: dict[str, set[str]] = {}
    invocation_sets: dict[str, set[str]] = {}
    inference_sets: dict[str, set[str]] = {}
    for role in roles:
        run_name = runs[role]
        run_dir, public_dir = artifact_root / "runs" / run_name, public_root / run_name
        validation = validate(run_dir, public_dir)
        run_manifest = read_json(run_dir / "run_manifest.json")
        identity = run_manifest.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"{role} run lacks identity")
        run_job = run_manifest.get("job_id", identity.get("job_id"))
        recorded_role = run_manifest.get("run_role", identity.get("run_role"))
        commit = run_manifest.get("git_commit", identity.get("git_commit"))
        if run_job != job_id or recorded_role != role or not isinstance(commit, str) or not commit:
            raise ValueError(f"{role} run does not bind job, role, and commit")
        code_files = read_json(run_dir / "code_files.json")
        if not isinstance(code_files, list) or identity.get("code_sha256") != digest(code_files):
            raise ValueError(f"{role} code file manifest is invalid")
        _verify_checkout(repository, commit, code_files)
        rows = [read_json(path) for path in sorted((run_dir / "samples").glob("*.json"))]
        inference_ids = {row.get("inference_id") for row in rows}
        invocation_ids = {row.get("inference_invocation_id") for row in rows}
        if len(rows) != 900 or None in inference_ids or None in invocation_ids or any(
            not isinstance(row.get("inference_started_at"), str) or not row["inference_started_at"] for row in rows
        ):
            raise ValueError(f"{role} records lack inference evidence")
        if len(inference_ids) != 900:
            raise ValueError(f"{role} inference IDs are not one-per-sample")
        identities[role] = identity
        id_sets[role] = {row["id"] for row in rows}
        inference_sets[role] = inference_ids
        invocation_sets[role] = invocation_ids
        evidence[role] = {"run_id": run_name, "identity": identity, "count": len(rows),
                          "viewer": str((run_dir / "review.html").relative_to(artifact_root)),
                          "validation": validation}
    if roles == _LEGACY_ROLES:
        for key in ("protocol_sha256", "model_sha256", "code_sha256", "environment_sha256", "hardware"):
            if identities["assignment"].get(key) != identities["analysis"].get(key):
                raise ValueError(f"A/B identity differs: {key}")
        if id_sets["assignment"] != id_sets["analysis"]:
            raise ValueError("A/B expected sample IDs differ")
        if inference_sets["assignment"] & inference_sets["analysis"]:
            raise ValueError("A/B inference IDs overlap")
        if invocation_sets["assignment"] & invocation_sets["analysis"]:
            raise ValueError("A/B inference invocation IDs overlap")
        if identities["assignment"].get("git_commit") != identities["analysis"].get("git_commit"):
            raise ValueError("A/B Git commits differ")
    return evidence | {"status": "REMOTE_VERIFIED", "bundle_schema": schema, "suite": suite,
                       "roles": list(roles), "pod_id": pod_id,
                       "total_inference_records": 900 * len(roles)}


def verify(archive_path: Path, expected_sha256: str, destination: Path, repository: Path) -> dict[str, Any]:
    """Validate archive bytes first, then extract once and write a local receipt."""
    archive_path, destination, repository = map(Path, (archive_path, destination, repository))
    if not archive_path.is_file() or archive_path.is_symlink():
        raise FileNotFoundError(f"Archive is absent or unsafe: {archive_path}")
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("Expected archive SHA-256 must be 64 lowercase hexadecimal characters")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite extraction destination: {destination}")
    actual = sha256_file(archive_path)
    if actual != expected_sha256:
        raise ValueError("Archive SHA-256 does not match the supplied value")
    with tarfile.open(archive_path, _tar_mode(archive_path, False)) as archive:
        manifest, members = _manifest_from_archive(archive)
        _check_storage(destination.parent, sum(member.size for name, member in members.items() if name != _MANIFEST))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".mmdl-bundle-", dir=destination.parent))
        try:
            for name, member in members.items():
                target = temporary / _safe_relative(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable archive member: {name}")
                with target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
            job_id = manifest.get("job_id")
            if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
                raise ValueError("Bundle manifest job ID is unsafe")
            job_metadata = read_json(temporary / "jobs" / f"{job_id}.json")
            schema, suite, roles, runs = _job_contract(job_id, job_metadata)
            if (manifest.get("schema") != schema or manifest.get("runs") != runs
                    or (schema == "mmdl-runpod-bundle-v2" and (
                        manifest.get("suite") != suite or manifest.get("roles") != list(roles)))):
                raise ValueError("Bundle manifest roles differ from exact job metadata")
            evidence = _cross_run_checks(temporary, temporary / "public", job_id, repository)
            receipt = {"status": "LOCAL_VERIFIED", "archive": archive_path.name,
                       "archive_sha256": actual, "job_id": job_id,
                       "pod_id": evidence["pod_id"], "bundle_schema": schema,
                       "suite": suite, "roles": list(roles),
                       "expected_inference_records": 900 * len(roles), "runs": evidence}
            write_json(temporary / "local_receipt.json", receipt)
            os.replace(temporary, destination)
            return receipt
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def cleanup_targets(receipt: dict[str, Any], ledger: dict[str, Any], pod_id: str) -> dict[str, Any]:
    """Return only resources proven owned by this locally verified job; never deletes."""
    if not isinstance(receipt, dict) or receipt.get("status") != "LOCAL_VERIFIED":
        raise ValueError("Cleanup requires a LOCAL_VERIFIED receipt")
    job_id = receipt.get("job_id")
    if not _JOB_ID.fullmatch(job_id or "") or receipt.get("pod_id") != pod_id:
        raise ValueError("Receipt does not bind the requested Pod to a safe job")
    runs = receipt.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("Receipt does not prove complete evaluation runs")
    bundle_schema = receipt.get("bundle_schema")
    if bundle_schema is None:
        if runs.get("total_inference_records") != 1800:
            raise ValueError("Legacy receipt does not prove both full A/B runs")
    elif bundle_schema == "mmdl-runpod-bundle-v1":
        if (receipt.get("suite") != "mmmu-val-two-runs"
                or receipt.get("roles") != list(_LEGACY_ROLES)
                or receipt.get("expected_inference_records") != 1800
                or runs.get("roles") != list(_LEGACY_ROLES)
                or runs.get("total_inference_records") != 1800
                or any(not isinstance(runs.get(role), dict)
                       or runs[role].get("run_id") != f"{job_id}-{role}"
                       or runs[role].get("count") != 900 for role in _LEGACY_ROLES)):
            raise ValueError("Legacy receipt does not prove both exact full A/B runs")
    else:
        roles = receipt.get("roles")
        if (receipt.get("bundle_schema") != "mmdl-runpod-bundle-v2"
                or receipt.get("suite") not in _SUITE_ROLES
                or roles != list(_SUITE_ROLES[receipt["suite"]])
                or receipt.get("expected_inference_records") != 900 * len(roles)
                or runs.get("roles") != roles
                or runs.get("total_inference_records") != 900 * len(roles)
                or any(not isinstance(runs.get(role), dict)
                       or runs[role].get("run_id") != f"{job_id}-{role}"
                       or runs[role].get("count") != 900 for role in roles)):
            raise ValueError("Receipt does not prove the exact complete suite roles")
    if (not isinstance(ledger, dict) or ledger.get("job_id") != job_id or ledger.get("pod_id") != pod_id
            or ledger.get("created_for_job") is not True):
        raise ValueError("Resource ledger does not prove Pod ownership for this job")
    if pod_id in set(ledger.get("preexisting_pod_ids", [])):
        raise ValueError("Refusing to target a pre-existing Pod")
    prior_volumes = set(ledger.get("preexisting_volume_ids", []))
    targets: list[str] = []
    for volume in ledger.get("network_volumes", []):
        if not isinstance(volume, dict):
            raise ValueError("Invalid network-volume ledger entry")
        volume_id = volume.get("id")
        if (not isinstance(volume_id, str) or not volume_id or volume_id in prior_volumes
                or volume.get("created_for_job") != job_id or volume.get("shared") is not False):
            raise ValueError("Refusing shared, pre-existing, or unowned network volume")
        if volume_id in targets:
            raise ValueError("Duplicate network-volume target")
        targets.append(volume_id)
    return {"status": "CLEANUP_TARGETS_VERIFIED", "job_id": job_id,
            "pod_id": pod_id, "network_volume_ids": targets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--artifact-root", required=True, type=Path)
    create_parser.add_argument("--public-root", required=True, type=Path)
    create_parser.add_argument("--job-id", required=True)
    create_parser.add_argument("--output", required=True, type=Path)
    create_parser.add_argument("--repository", type=Path, default=Path.cwd())
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--archive", required=True, type=Path)
    verify_parser.add_argument("--sha256", required=True)
    verify_parser.add_argument("--destination", required=True, type=Path)
    verify_parser.add_argument("--repository", required=True, type=Path)
    cleanup_parser = commands.add_parser("cleanup-targets")
    cleanup_parser.add_argument("--receipt", required=True, type=Path)
    cleanup_parser.add_argument("--resources", required=True, type=Path)
    cleanup_parser.add_argument("--pod-id", required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create(args.artifact_root, args.public_root, args.job_id, args.output, args.repository)
    elif args.command == "verify":
        result = verify(args.archive, args.sha256, args.destination, args.repository)
    else:
        result = cleanup_targets(read_json(args.receipt), read_json(args.resources), args.pod_id)
    print(json.dumps({key: result[key] for key in ("status", "job_id", "pod_id", "archive_sha256", "file_count", "total_inference_records", "network_volume_ids") if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
