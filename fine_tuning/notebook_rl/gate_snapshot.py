"""File-bound step1 exports for isolated G6 cache controls; no GPU allocation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from . import checkpoint, export, launch
from .vault import _safe_root


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("gate snapshot: " + message)


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_snapshot(descriptor: dict, binding: dict, checkpoint_path: Path) -> dict:
    """Validate the immutable export against its actual saved step1 source."""
    path = launch.descriptor(descriptor)
    root = _safe_root(path.parent)
    _require(path.name == "export_manifest.json", "export completion manifest required")
    manifest = launch.read_descriptor(descriptor)
    ckpt = _safe_root(Path(checkpoint_path))
    metadata = checkpoint.read_metadata(ckpt, binding=binding)
    _require(metadata["step"] == 1 and manifest.get("status") == "EXPORTED"
             and manifest.get("format") == "derived-hf-bf16-v1"
             and manifest.get("base_id") == BASE_ID and manifest.get("base_revision") == BASE_REVISION
             and manifest.get("binding") == binding and manifest.get("arm") == metadata["arm"]
             and manifest.get("resolved_protocol_sha256") == metadata["resolved_protocol_sha256"]
             and manifest.get("checkpoint_state_sha256") == metadata["state_sha256"]
             and manifest.get("checkpoint_files") == {p.name: checkpoint.file_hash(p) for p in ckpt.iterdir()},
             "export differs from bound step1 checkpoint")
    files = manifest.get("files")
    _require(type(files) is dict and bool(files) and all(isinstance(name, str) and bool(name)
             and name not in (".", "..", path.name) and "/" not in name and "\\" not in name
             and _hash(digest) for name, digest in files.items()), "safe flat export files required")
    _require({p.name for p in root.iterdir()} == set(files) | {path.name}, "export file inventory differs")
    for name, digest in files.items():
        file = root/name
        _require(file.is_file() and not file.is_symlink() and checkpoint.file_hash(file) == digest,
                 "export file hash differs: " + name)
    tensors = manifest.get("tensors")
    _require(type(tensors) is list and bool(tensors), "complete tensor inventory required")
    names = set()
    for row in tensors:
        _require(type(row) is dict and set(row) == {"name", "shape", "dtype", "sha256"}
                 and isinstance(row["name"], str) and row["name"] not in names
                 and type(row["shape"]) is list and bool(row["shape"])
                 and all(type(n) is int and n > 0 for n in row["shape"])
                 and row["dtype"] == "torch.bfloat16" and _hash(row["sha256"]),
                 "invalid or duplicate BF16 tensor descriptor")
        names.add(row["name"])
    _require(names == set(metadata["base_sha256"]), "canonical complete tensor names required")
    _require(manifest.get("mergers") == metadata["inventory"]["mergers"]
             and manifest.get("merger_tensor_names") == sorted(name for name in names if any(
                 name.startswith(root + ".") for root in metadata["inventory"]["mergers"])),
             "Merger inventory differs")
    _require(any(name.endswith(".safetensors") for name in files), "weight shards required")
    _require(launch.read_descriptor(descriptor) == manifest
             and checkpoint.read_metadata(ckpt, binding=binding) == metadata,
             "sources changed during validation")
    return manifest


def export_step1(captured: dict, model_path: Path, output: Path, arm: str, binding: dict) -> dict:
    """Reload saved step1 on CPU, never export the caller's live step2 policy."""
    _require(not torch.cuda.is_initialized(), "isolated CPU export process required")
    _require(captured.get("status") == "CAPTURE_COMPLETED" and captured.get("saved_step") == 1
             and captured.get("live_final_step") == 2 and captured.get("binding") == binding
             and isinstance(captured.get("shared_checkpoint"), str)
             and Path(captured["shared_checkpoint"]).is_absolute(), "closed bound step1 capture required")
    ckpt = _safe_root(Path(captured["shared_checkpoint"]))
    metadata = checkpoint.read_metadata(ckpt, binding=binding)
    _require(metadata["step"] == 1 and metadata["arm"] == arm, "saved step1 and matching arm required")
    _safe_root(Path(output).absolute().parent)
    export.export_merged(ckpt, Path(model_path), Path(output), arm, binding=binding)
    path = Path(output).absolute()/"export_manifest.json"
    descriptor = {"path": str(path), "file_sha256": checkpoint.file_hash(path)}
    validate_snapshot(descriptor, binding, ckpt)
    return descriptor


def iter_tensors(descriptor: dict, binding: dict, checkpoint_path: Path
                 ) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream each canonical CPU BF16 tensor once; exhaust before accepting a load."""
    manifest = validate_snapshot(descriptor, binding, checkpoint_path)
    root = launch.descriptor(descriptor).parent
    expected = {row["name"]: row for row in manifest["tensors"]}
    seen = set()
    for filename in sorted(name for name in manifest["files"] if name.endswith(".safetensors")):
        with safe_open(root/filename, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                _require(name in expected and name not in seen, "unexpected or duplicate shard tensor: " + name)
                value = shard.get_tensor(name)
                row = expected[name]
                _require(value.device.type == "cpu" and value.dtype == torch.bfloat16
                         and list(value.shape) == row["shape"] and bool(torch.isfinite(value).all())
                         and tensor_sha256(value) == row["sha256"], "tensor data differs: " + name)
                seen.add(name)
                yield name, value
    _require(seen == set(expected), "incomplete shard tensor inventory")
    _require(validate_snapshot(descriptor, binding, checkpoint_path) == manifest,
             "snapshot changed during tensor streaming")
