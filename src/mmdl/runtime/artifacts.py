"""Small atomic artifacts and content identity; paths are supplied by the caller."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_text(path, text):
    atomic_bytes(path, text.encode("utf-8"))


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_path(variable, suffix):
    return Path(os.environ.get(variable, str(Path.home() / suffix))).expanduser().resolve()


def file_records(root, paths):
    return [{"path": str(Path(p).relative_to(root)), "bytes": Path(p).stat().st_size,
             "sha256": sha256_file(p)} for p in sorted(paths)]


def resolve_image(run_dir, relative):
    """Allow legacy run images or exactly the content-addressed shared asset path."""
    run_dir = Path(run_dir).resolve()
    if re.fullmatch(r"images/[A-Za-z0-9_.-]+\.png", relative):
        base = run_dir / "images"
    elif re.fullmatch(r"\.\./\.\./assets/images/[0-9a-f]{64}\.png", relative):
        base = run_dir.parent.parent / "assets/images"
    else:
        raise ValueError("Unsafe inspection image path")
    path = run_dir / relative
    if (base.is_symlink() or base.parent.is_symlink() or path.is_symlink()
            or not base.resolve().is_relative_to(run_dir.parent.parent)
            or not path.resolve().is_relative_to(base.resolve())):
        raise ValueError("Inspection image escapes its asset directory")
    return path.resolve()


def git_commit(root, required=None):
    """Strict clean-checkout gate when a reproducibility commit is supplied."""
    if required is not None and not re.fullmatch(r"[0-9a-f]{40}", required):
        raise ValueError("Expected full lowercase 40-character Git commit SHA")
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    commit = result.stdout.strip() if result.returncode == 0 else None
    if required is not None:
        if commit != required:
            raise ValueError("Checkout does not match required Git commit")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
        extra = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--",
                                         "src", "scripts", "configs", "prompts", "env", "third_party"], cwd=root)
        if dirty.strip() or extra.strip():
            raise ValueError("Reproducible inference requires a clean committed checkout")
    return commit
