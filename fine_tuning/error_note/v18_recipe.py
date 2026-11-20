"""Public, content-bound selection metadata; source text is restored privately."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

VERSION = "v18-portable-selection-v1"
DEFAULT_RECIPE = Path(__file__).with_suffix(".json")
PACKAGE_NAME = "v18_recipe_package.json"
ROLE_COUNTS = {"train": 3000, "prep": 512, "monitor": 200, "effect": 128, "performance": 16}
TRAIN_IDS_SHA256 = "fe4d0e5a796d06972f775cef32cfd00083adefcc83be9aec619baf9f8555fd80"
_PRIVATE_KEYS = {"question", "gold", "raw_response", "lecture", "solution", "raw_annotation",
    "text", "tokens", "raw_question", "raw_answer", "answer", "answers", "rationale",
    "response", "generated_text", "completion", "image_base64", "image_bytes", "pixel_values"}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def portable_payload(value: Any) -> Any:
    """Paths are transport details, while source hashes and pointers remain bound."""
    if isinstance(value, dict):
        return {key: portable_payload(item) for key, item in value.items() if key not in {"path", "origin"}}
    if isinstance(value, list):
        return [portable_payload(item) for item in value]
    return value


def expected_digests(item: dict) -> dict[str, str]:
    direct = {"actor": item["actor"], "verifier": item["verifier"], "lineage": item["lineage"],
        "question": item["question"], "choices": item["choices"], "evaluation": item["evaluation"],
        "annotation": item.get("annotation"),
        "raw_annotation": item.get("references", {}).get("raw_annotation"),
        "native_metadata": item.get("references", {}).get("native_metadata")}
    portable = {key: portable_payload(item.get(key)) for key in ("references", "human_solution")}
    return {f"{key}_sha256": digest(value) for key, value in (direct | portable).items()}


def _public(value: Any) -> None:
    if isinstance(value, dict):
        for key in _PRIVATE_KEYS.intersection(value):
            if key not in {"lecture", "solution"} or not re.fullmatch(r"[0-9a-f]{64}", str(value[key])):
                raise ValueError("RECIPE_PRIVATE_PAYLOAD")
        for child in value.values():
            _public(child)
    elif isinstance(value, list):
        for child in value:
            _public(child)
    elif isinstance(value, str) and (value.startswith(("/home/", "/mnt/", "/tmp/")) or ":\\" in value):
        raise ValueError("RECIPE_HOST_PATH")


def _document(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RECIPE_DOCUMENT_NOT_OBJECT")
    _public(value)
    if value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"}):
        raise ValueError("RECIPE_CONTENT_DIGEST_MISMATCH")
    return value


def _package_file(root: Path, name: str) -> Path:
    if not re.fullmatch(r"v18_recipe(?:_[a-z0-9]+)*\.json", name):
        raise ValueError("RECIPE_PACKAGE_UNSAFE_PATH")
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("RECIPE_PACKAGE_UNSAFE_PATH")
    return path


def _package_bytes(root: Path, ref: dict) -> bytes:
    data = _package_file(root, ref["path"]).read_bytes()
    if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("RECIPE_PACKAGE_FILE_CHANGED")
    return data


def restore_package(package_path: Path, cache_root: Path | None = None) -> Path:
    """Restore the original recipe bytes outside the verified dataset directory."""
    package_path = Path(package_path)
    package = _document(package_path)
    if package["version"] != "v18-recipe-package-v1":
        raise ValueError("RECIPE_PACKAGE_VERSION")
    cache_root = Path(cache_root or os.environ.get("MMDL_RECIPE_CACHE") or
        Path.home() / "mmdl-cache" / "v18-recipe")
    destination = cache_root / package["content_sha256"]
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("RECIPE_PACKAGE_CACHE_SYMLINK")
    destination.mkdir(parents=True, exist_ok=True)
    for ref in package["files"]:
        target = _package_file(destination, ref["path"])
        if "parts" in ref:
            rows = []
            for part in ref["parts"]:
                _package_bytes(package_path.parent, part)
                value = _document(_package_file(package_path.parent, part["path"]))
                if value["version"] != ref["document_version"] or len(value["items"]) != part["count"]:
                    raise ValueError("RECIPE_PACKAGE_PART_CONTRACT")
                rows.extend(value["items"])
            value = {"version": ref["document_version"], "items": rows,
                "content_sha256": ref["document_content_sha256"]}
            data = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                allow_nan=False, separators=(",", ":")) + "\n").encode()
        else:
            data = _package_bytes(package_path.parent, ref)
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("RECIPE_PACKAGE_REASSEMBLY_CHANGED")
        _public(json.loads(data))
        if target.is_file():
            _package_bytes(destination, ref)
            continue
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    recipe_path = _package_file(destination, package["recipe"])
    recipe = _document(recipe_path)
    recipe["items"] = _shards(recipe, destination)
    validate_recipe(recipe)
    load_sidecars(recipe, recipe_path)
    return recipe_path


def resolve_recipe_path(path: Path | None = None) -> Path:
    path = Path(path or DEFAULT_RECIPE)
    package_path = path.with_name(PACKAGE_NAME)
    if path.name == "v18_recipe.json" and package_path.is_file():
        # The packed source is also checked when an old external cache exists.
        return restore_package(package_path)
    return path


def _shards(document: dict, root: Path) -> list[dict]:
    items = []
    for ref in document["item_shards"]:
        path = root / ref["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("RECIPE_SHARD_PATH_ESCAPE")
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError("RECIPE_SHARD_DIGEST_MISMATCH")
        shard = _document(path)
        if len(shard["items"]) != ref["count"] or shard["content_sha256"] != ref["content_sha256"]:
            raise ValueError("RECIPE_SHARD_COUNT_MISMATCH")
        items.extend(shard["items"])
    return items


def validate_recipe(recipe: dict) -> dict:
    if recipe["version"] != VERSION:
        raise ValueError("RECIPE_VERSION_MISMATCH")
    rows = recipe["items"]
    if Counter(row["role"] for row in rows) != ROLE_COUNTS:
        raise ValueError("RECIPE_ROLE_COUNTS_MISMATCH")
    if len({row["problem_id"] for row in rows}) != sum(ROLE_COUNTS.values()):
        raise ValueError("RECIPE_DUPLICATE_ID")
    if len(recipe["sources"]) != 15:
        raise ValueError("RECIPE_SOURCE_COUNT_MISMATCH")
    for role, count in ROLE_COUNTS.items():
        subset = [row for row in rows if row["role"] == role]
        if [row["role_index"] for row in subset] != list(range(count)):
            raise ValueError("RECIPE_ROLE_ORDER_MISMATCH")
        ids = [row["problem_id"] for row in subset]
        if digest(ids) != recipe["roles"][role]["ids_sha256"]:
            raise ValueError("RECIPE_ROLE_IDS_MISMATCH")
    if recipe["roles"]["train"]["ids_sha256"] != TRAIN_IDS_SHA256:
        raise ValueError("RECIPE_TRAIN_ORDER_CHANGED")
    files = recipe["files"]
    if digest(files) != recipe["files_sha256"]:
        raise ValueError("RECIPE_FILES_DIGEST_MISMATCH")
    for spec in files.values():
        if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", ""))) or spec.get("bytes", 0) <= 0:
            raise ValueError("RECIPE_FILE_INTEGRITY_MISSING")
        if spec["provider"] == "huggingface" and not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("RECIPE_UNPINNED_SOURCE")
    for row in rows:
        spec = recipe["sources"][row["source"]]
        if row["source_revision"] != spec["revision"] or row["ordinal"] != int(row["problem_id"].split(":")[-2]):
            raise ValueError("RECIPE_SOURCE_ID_MISMATCH")
        if row["training_allowed"] != (row["role"] in {"train", "prep"}):
            raise ValueError("RECIPE_ROLE_TRAINING_LEAK")
        for ref in [row["locator"], *row.get("auxiliary_refs", [])]:
            if ref.get("file_key") is not None and ref["file_key"] not in files:
                raise ValueError("RECIPE_FILE_REFERENCE_MISSING")
        if len(row["images"]) not in {1, 2} or len(row["expected"]) != 11:
            raise ValueError("RECIPE_EXPECTED_FIELDS_MISSING")
        for image in row["images"]:
            path = PurePosixPath(image["path"])
            if path.is_absolute() or ".." in path.parts or "\\" in image["path"]:
                raise ValueError("RECIPE_IMAGE_PATH_ESCAPE")
    return {"status": "PASS_SELECTION_METADATA_ONLY", "counts": ROLE_COUNTS,
        "sources": len(recipe["sources"]), "files": len(files),
        "source_reconstruction_verified": False, "human_verified": False}


def load_recipe(path: Path | None = None) -> dict:
    path = resolve_recipe_path(path)
    recipe = _document(path)
    recipe["items"] = _shards(recipe, path.parent)
    validate_recipe(recipe)
    return recipe


def load_sidecars(recipe: dict | None = None, recipe_path: Path | None = None) -> dict:
    recipe_path = resolve_recipe_path(recipe_path)
    recipe = recipe or load_recipe(recipe_path)
    ref = recipe["sidecars"]
    path = recipe_path.parent / ref["path"]
    if not path.resolve().is_relative_to(recipe_path.parent.resolve()):
        raise ValueError("RECIPE_SIDECAR_PATH_ESCAPE")
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("RECIPE_SIDECAR_DIGEST_MISMATCH")
    sidecars = _document(path)
    rows = _shards(sidecars, path.parent)
    if len(rows) != len(recipe["items"]) or {row["problem_id"] for row in rows} != {row["problem_id"] for row in recipe["items"]}:
        raise ValueError("RECIPE_SIDECAR_IDS_MISMATCH")
    sidecars["items"] = {row["problem_id"]: row for row in rows}
    for item in recipe["items"]:
        if digest(sidecars["items"][item["problem_id"]]) != item["sidecar_sha256"]:
            raise ValueError("RECIPE_ITEM_SIDECAR_DIGEST_MISMATCH")
    return sidecars
