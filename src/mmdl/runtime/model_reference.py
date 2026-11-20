"""Resolve a local unmerged export, checking its complete reference before inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mmdl.runtime.artifacts import sha256_file

BASE_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EXPORT_FORMAT = "derived-hf-bf16-unmerged-lora-v1"
BASE_MANIFEST_SHA256 = "ed950145b8b5d6cf895edb12ec8283e866f144b8c0371b7573d020a34d53747e"


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"model reference: duplicate key {key}")
        result[key] = value
    return result


def read_reference(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"model reference: missing regular file {path}")
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if not isinstance(result, dict):
        raise ValueError(f"model reference: JSON object required at {path}")
    return result


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"model reference: missing regular file {path}; expected SHA256 {expected}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"model reference: {path}: expected SHA256 {expected}, actual {actual}")


def resolve_export_reference(
    path: Path, export_root: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """A selector uses MMDL_V18B_EXPORT_ROOT; a fresh run uses its own model_ref.json.

    Small-model rehearsals require a separately hash-bound selector with reportable=false.
    """
    path = Path(path).absolute()
    selector = read_reference(path)
    is_selector = selector.get("kind") in {"export-reference", "derived-rehearsal"}
    rehearsal = selector.get("kind") == "derived-rehearsal"
    if is_selector:
        configured = export_root or os.environ.get(selector.get("export_root_env", "MMDL_V18B_EXPORT_ROOT"))
        if not configured:
            raise ValueError("model reference: set MMDL_V18B_EXPORT_ROOT to the directory containing model_ref.json and model/")
        root = Path(configured).absolute()
        if root.name == "model":
            root = root.parent
        path = root / "model_ref.json"
        require_hash(path, selector["reference_sha256"])
        reference = read_reference(path)
    else:
        reference = selector
        root = Path(export_root).absolute() if export_root else path.parent
        if root.name == "model":
            root = root.parent
    model = root / "model"
    if (reference.get("kind") != "unmerged-lora"
            or reference.get("base") != {"id": BASE_ID, "revision": BASE_REVISION}
            or reference.get("dtype") != "bfloat16"
            or reference.get("processor_revision") != BASE_REVISION
            or reference.get("usage") != "evaluation_only"):
        raise ValueError(f"model reference: pinned unmerged BF16 export required at {path}")
    if rehearsal:
        if selector.get("reportable") is not False:
            raise ValueError("model reference: rehearsal must explicitly be nonreportable")
    elif reference.get("export", {}).get("base_manifest_sha256") != BASE_MANIFEST_SHA256:
        raise ValueError("model reference: export must use the pinned full base manifest")
    if is_selector:
        require_hash(model / "export_manifest.json", selector["export_manifest_sha256"])
        if not rehearsal:
            campaign = reference.get("export", {}).get("campaign", {})
            if (campaign.get("slot") != selector["slot"] or campaign.get("stage") != "final"
                    or reference["export"].get("checkpoint_files") != selector["checkpoint_files"]):
                raise ValueError("model reference: historical checkpoint differs from selector")
    from fine_tuning.notebook_rl.export import validate_export_reference

    manifest = validate_export_reference(model, reference)
    if manifest.get("format") != EXPORT_FORMAT:
        raise ValueError("model reference: unmerged language and vision adapters required")
    if rehearsal:
        reference = reference | {"reportable": False}
    return path, model, reference
