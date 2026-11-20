"""Build the private v2 native benchmark index on CPU; default is a read-only plan.

Only the shipped, SHA-pinned HF inventory is downloadable. This produces native
text, raw-image cache and perceptual fingerprints, never OCR/embeddings/calibration
or training approval. Completed per-file shards can be reused after interruption.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

from PIL import Image

from fine_tuning import audit

from . import contamination as c
from . import vault

_REPO = Path(__file__).resolve().parents[2]
_IMAGE = re.compile(r"image(?:_[1-7])?\Z")


def _inventory() -> list[dict]:
    raw = vault._read(c._INVENTORY)
    if hashlib.sha256(raw).hexdigest() != c._INVENTORY_SHA256:
        raise ValueError("index builder: pinned inventory changed")
    return json.loads(raw, object_pairs_hook=vault._unique)["files"]


def _private_root(path: Path, *, create: bool = False) -> Path:
    path = Path(path).absolute()
    if path.resolve().is_relative_to(_REPO):
        raise ValueError("index builder: private artifacts must be outside the repository")
    return vault._safe_root(path, create=create)


def _cache_path(entry: dict, cache_root: Path) -> Path | None:
    repo = cache_root / ("datasets--" + entry["dataset_id"].replace("/", "--"))
    for candidate in (repo / "snapshots" / entry["revision"] / entry["native_path"],
                      repo / "blobs" / entry["sha256"]):
        if candidate.exists():
            path = candidate.resolve(strict=True)
            if not path.is_relative_to(cache_root.resolve()) or not path.is_file():
                raise ValueError("index builder: cache file escapes the requested cache root")
            return path
        if candidate.is_symlink():
            raise ValueError("index builder: broken native cache link")
    return None


def _verify_file(path: Path, entry: dict) -> None:
    if path.stat().st_size != entry["bytes"] or audit.sha(path) != entry["sha256"]:
        raise ValueError("index builder: native file bytes/SHA256 differ from pinned inventory")


def plan(cache_root: Path, output_root: Path) -> dict:
    """Report cached presence only; execute always verifies bytes and row counts."""
    if output_root.resolve().is_relative_to(_REPO) or cache_root.resolve().is_relative_to(_REPO):
        raise ValueError("index builder: cache and private output must be outside the repository")
    files = [{**entry, "cached": _cache_path(entry, cache_root) is not None} for entry in _inventory()]
    return {"status": "PLAN_ONLY", "files": files, "record_count": sum(e["records"] for e in files),
            "native_bytes": sum(e["bytes"] for e in files),
            "missing_download_bytes": sum(e["bytes"] for e in files if not e["cached"]),
            "cached_presence_is_sha_verified": False, "output_root": str(output_root.absolute()),
            "training_allowed": False, "ocr_sscd_siglip_calibration_produced": False}


def _write_once(path: Path, raw: bytes) -> None:
    """Publish atomically without replacing a prior payload, including on restart."""
    if path.exists() or path.is_symlink():
        if vault._read(path) != raw:
            raise ValueError("index builder: existing private artifact differs")
        return
    fd, temporary = tempfile.mkstemp(prefix=".index-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if vault._read(path) != raw:
                raise ValueError("index builder: concurrent private artifact differs") from None
    finally:
        os.unlink(temporary)


def _texts(row: dict, vision: bool) -> dict:
    question = row.get("question")
    if vision and question is None:
        question = ""
    if not isinstance(question, str) or (not vision and not question.strip()):
        raise ValueError("index builder: native question missing/invalid")
    options = row.get("options")
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("index builder: native options are not a string list") from exc
    if vision and options is None:
        options = []
    if not isinstance(options, (list, tuple)) or any(not isinstance(v, str) for v in options):
        raise ValueError("index builder: native options are not a string list")
    explanation = row.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise ValueError("index builder: native explanation is not a string/null")
    return {"question": [question], "choices": list(options),
            "explanation": [] if explanation is None else [explanation]}


def _build_shard(entry: dict, path: Path, output: Path, images: dict) -> dict:
    import pyarrow.parquet as pq

    native = pq.ParquetFile(path)
    if native.metadata.num_rows != entry["records"]:
        raise ValueError("index builder: native Parquet row count differs from inventory")
    vision = entry["dataset_id"] == "MMMU/MMMU_Pro" and entry["configuration"] == "vision"
    columns = native.schema_arrow.names
    required = {"id"} if vision else {"id", "question", "options", "explanation"}
    if not required.issubset(columns) or len(set(columns)) != len(columns):
        raise ValueError("index builder: missing/duplicate native columns")
    image_fields = [name for name in columns if _IMAGE.fullmatch(name)]
    if not image_fields:
        raise ValueError("index builder: no native image columns")
    records, owned_images, ids = [], {}, set()
    for batch in native.iter_batches(batch_size=8):
        for row in batch.to_pylist():
            identity = row["id"]
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError("index builder: missing/duplicate native record ID")
            ids.add(identity)
            texts = _texts(row, vision)
            ref = {key: entry[key] for key in ("dataset_id", "configuration", "split")}
            ref.update(id=identity, file_sha256=entry["sha256"])
            record = {"ref": ref, "texts": texts,
                      "native_texts": {key: row.get(key) for key in ("question", "options", "explanation")},
                      "image_sha256": [], "image_fields": [], "native_image_paths": {}}
            for field in image_fields:
                value = row[field]
                if value is None:
                    continue
                if (not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes)
                        or not value["bytes"] or value.get("path") is not None
                        and not isinstance(value["path"], str)):
                    raise ValueError("index builder: native embedded image bytes/path missing/invalid")
                raw = value["bytes"]
                digest = hashlib.sha256(raw).hexdigest()
                _write_once(output / "images" / digest, raw)
                if digest not in images:
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        images[digest] = c.image_fingerprints(image)
                owned_images[digest] = images[digest]
                record["image_sha256"].append(digest)
                record["image_fields"].append(field)
                record["native_image_paths"][field] = value.get("path")
            if not record["image_sha256"]:
                raise ValueError("index builder: native visual record has no image bytes")
            records.append(record)
    if len(records) != entry["records"]:
        raise ValueError("index builder: decoded native row count differs from inventory")
    _verify_file(path, entry)
    return {"records": records, "images": owned_images}


def _build(entries: list[dict], cache_root: Path, output: Path, *, max_download_bytes: int = 0,
           max_files: int | None = None, partial_name: str | None = None) -> dict:
    """Internal partial entrypoint is for explicitly named synthetic fixture checks."""
    if type(max_download_bytes) is not int or max_download_bytes < 0:
        raise ValueError("index builder: nonnegative download byte cap required")
    if max_files is not None and (type(max_files) is not int or max_files <= 0):
        raise ValueError("index builder: positive per-run file limit required")
    cache_root = cache_root.absolute()
    if cache_root.resolve().is_relative_to(_REPO):
        raise ValueError("index builder: native cache must be outside the repository")
    output = _private_root(output, create=True)
    for name in ("shards", "images"):
        vault._safe_root(output / name, create=True)
    producer = audit.sha(c.__file__)
    builder = audit.sha(__file__)
    records, images, coverage = [], {}, []
    download_bytes, built = 0, 0
    for entry in entries:
        shard_path = output / "shards" / (entry["sha256"] + ".json")
        if (max_files is not None and built >= max_files
                and not shard_path.exists() and not shard_path.is_symlink()):
            return {"status": "CHECKPOINTED", "completed_files": len(coverage),
                    "record_count": len(records), "downloaded_bytes": download_bytes,
                    "training_allowed": False}
        path = _cache_path(entry, cache_root)
        if path is None:
            if download_bytes + entry["bytes"] > max_download_bytes:
                raise ValueError("index builder: missing native cache file exceeds download byte cap")
            from huggingface_hub import hf_hub_download

            hf_hub_download(entry["dataset_id"], entry["native_path"], repo_type="dataset",
                            revision=entry["revision"], cache_dir=str(cache_root))
            download_bytes += entry["bytes"]
            path = _cache_path(entry, cache_root)
            if path is None:
                raise ValueError("index builder: downloaded native cache file is missing")
        _verify_file(path, entry)
        identity = {"entry": entry, "producer_sha256": producer, "builder_sha256": builder}
        if shard_path.exists() or shard_path.is_symlink():
            shard = json.loads(vault._read(shard_path), object_pairs_hook=vault._unique)
            if (any(shard.get(k) != v for k, v in identity.items())
                    or shard.get("payload_sha256") != c._digest(shard.get("payload"))):
                raise ValueError("index builder: stale/tampered per-file shard")
            payload = shard["payload"]
            for digest in payload["images"]:
                if hashlib.sha256(vault._read(output / "images" / digest)).hexdigest() != digest:
                    raise ValueError("index builder: cached raw image changed")
        else:
            payload = _build_shard(entry, path, output, images)
            _write_once(shard_path, c._json({**identity, "payload": payload,
                                           "payload_sha256": c._digest(payload)}))
            built += 1
        records.extend(payload["records"])
        images.update(payload["images"])
        coverage.append({**entry, "path": str(path)})
    for entry in coverage:
        _verify_file(Path(entry["path"]), entry)
    count = len(records)
    value = {"schema": "notebook-rl-exclusion-index-v2", "purpose": "exclusion_fingerprints_only",
             "status": "COMPLETE", "training_allowed": False, "public_upload_allowed": False,
             "coverage_verified": True, "producer_sha256": producer, "builder_sha256": builder,
             "normalizer_version": c.VERSION, "fingerprint_version": c.FINGERPRINT_VERSION,
             "coverage": {"files": coverage, "record_count": count}, "record_count": count,
             "per_file_record_counts": {e["path"]: e["records"] for e in coverage},
             "source_files": {e["path"]: e["sha256"] for e in coverage},
             "image_count": sum(len(r["image_sha256"]) for r in records),
             "records": records, "images": images, "raw_image_cache": "images/{image_sha256}",
             "ocr_complete": False, "sscd_complete": False, "siglip_complete": False,
             "calibration_complete": False, "training_approval": False}
    prepared = c._prepare(value, partial_name)
    _write_once(output / "index.json", c._json(value))
    return {"status": "INDEX_READY" if prepared["full_coverage"] else "NAMED_PARTIAL_INDEX_READY",
            "index": str(output / "index.json"), "record_count": count,
            "image_count": value["image_count"], "unique_images": len(images),
            "downloaded_bytes": download_bytes, "training_allowed": False,
            "ocr_sscd_siglip_calibration_produced": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path, help="Existing HF hub cache root")
    parser.add_argument("--output-root", required=True, type=Path, help="Private directory outside repository")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-download-bytes", type=int, default=0,
                        help="Total pinned missing bytes allowed this run; default cache-only")
    parser.add_argument("--max-files", type=int, help="Checkpoint after at most this many newly built shards")
    args = parser.parse_args()
    result = (_build(_inventory(), args.cache_root, args.output_root,
                     max_download_bytes=args.max_download_bytes, max_files=args.max_files)
              if args.execute else plan(args.cache_root, args.output_root))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
