"""Pinned resources available in either the original checkout or a folder copy."""
from pathlib import Path

from .vault import _read, _safe_root

HERE = Path(__file__).resolve().parent
RESOURCE_FILES = {
    "manifests/models/baseline.json": "baseline.json",
    "env/requirements-vllm.lock": "requirements-vllm.lock",
    "prompts/mmmu_open_v1.txt": "mmmu_open_v1.txt",
    "prompts/mmmu_mcq_v1.txt": "mmmu_mcq_v1.txt",
}


def resource_path(descriptor: str, *, project_root: Path | None = None) -> Path:
    if not isinstance(descriptor, str) or descriptor not in RESOURCE_FILES:
        raise ValueError("resource: unsupported pinned resource descriptor")
    root = HERE.parents[1] if project_root is None else Path(project_root)
    legacy = root / descriptor
    path = legacy if legacy.exists() or legacy.is_symlink() else HERE / "resources" / RESOURCE_FILES[descriptor]
    _safe_root(path.parent)
    _read(path)
    return path
