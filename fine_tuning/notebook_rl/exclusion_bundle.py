"""Join complete private native input and real frozen detector caches; never approve training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import contamination as c
from . import vault
from .build_exclusion_index import _private_root, _write_once
from .dataset import SPLITS
from .exclusion_engines import _entry


def read_json(path: Path) -> dict:
    return json.loads(vault._read(path), object_pairs_hook=vault._unique)


def text_entries(data: dict) -> dict:
    entries = {}
    for record in data["records"]:
        ref = record["ref"]
        key = [ref[k] for k in ("dataset_id", "configuration", "split", "id")]
        for field, values in record["texts"].items():
            for n, text in enumerate(values):
                normal = c.normalize_text(text)
                if normal:
                    entries[c._digest([*key, field, n])] = normal
    return entries


def prepare_manifests(index_path: Path, raw_images: Path, build: Path, output: Path) -> dict:
    data = read_json(index_path)
    full, limits = c._coverage(data)
    if not full or limits or data.get("schema") != "notebook-rl-exclusion-index-v2":
        raise ValueError("full native v2 index required")
    items, files, images = c._snapshot({s: build / s for s in SPLITS}, build / "images")
    paths = {sha: str((raw_images / sha).absolute()) for sha in data["images"]}
    paths.update({sha: path for path, sha in files.items() if sha in images})
    for sha, path in paths.items():
        if hashlib.sha256(vault._read(Path(path))).hexdigest() != sha:
            raise ValueError("input/native image bytes changed")
    texts = {hashlib.sha256(text.encode()).hexdigest(): text for text in text_entries(data).values()}
    texts.update({item["text_sha256"]: item["text"] for item in items.values()})
    output = _private_root(output, create=True)
    _write_once(output / "images.json", c._json(paths))
    _write_once(output / "texts.json", c._json(texts))
    summary = {"status": "DETECTOR_INPUTS_ONLY", "training_allowed": False,
               "index_file_sha256": hashlib.sha256(vault._read(index_path)).hexdigest(),
               "input_files_sha256": c._digest(files), "images": len(paths), "unique_texts": len(texts),
               "image_manifest_sha256": c._digest(paths), "text_manifest_sha256": c._digest(texts)}
    _write_once(output / "manifest.json", c._json(summary))
    return summary


def cache_payloads(root: Path, keys: set[str], *, texts=False) -> tuple[dict, dict]:
    identity = read_json(root / "identity.json")
    if (identity.get("producer_sha256") != hashlib.sha256(
            vault._read(Path(__file__).with_name("exclusion_engines.py"))).hexdigest()
            or identity.get("normalizer_sha256") != hashlib.sha256(
                vault._read(Path(c.__file__))).hexdigest()):
        raise ValueError("detector cache producer/normalizer code changed")
    digest = c._digest(identity)
    if root.name != digest:
        raise ValueError("cache directory differs from engine identity")
    result = {}
    for key in sorted(keys):
        payload = _entry(root / ("texts" if texts else "images") / (key + ".json"),
                         digest, key, "text" if texts else "image")
        if payload is None or payload.get("sha256") != key:
            raise ValueError("missing or mismatched real detector cache entry")
        result[key] = payload
    return identity, result


def assemble(index_path: Path, build: Path, calibration_path: Path, cache_roots: dict[str, Path],
             source_registry_path: Path, output: Path) -> dict:
    """Full scan emits all unresolved candidates. This never silently releases them."""
    index = c.load_index(index_path)
    calibration = read_json(calibration_path)
    if calibration.get("index_sha256") != index["sha256"]:
        raise ValueError("calibration belongs to another index")
    items, files, images = c._snapshot({s: build / s for s in SPLITS}, build / "images")
    human = {sha for item in items.values() if item["lineage"].label_provenance == "HUMAN"
             for sha in item["image_sha256"]}
    engine_keys = set(index["images"]) | set(images)
    evidence, identities = {"calibration": calibration}, {}
    for name in ("ocr", "sscd", "siglip"):
        keys = set(index["images"]) | human if name == "ocr" else engine_keys
        identity, payloads = cache_payloads(cache_roots[name], keys)
        if ((name == "ocr" and identity["engine_id"] != "tesseract")
                or (name == "sscd" and identity["engine_id"] != "sscd_disc_mixup")
                or (name == "siglip" and identity["engine_id"] != "google/siglip-base-patch16-224")):
            raise ValueError("wrong detector engine")
        identities[name] = {k: identity[k] for k in ("engine_id", "revision", "artifact_sha256")}
        if name != "ocr" and calibration.get("engines", {}).get(name) != identity:
            raise ValueError("calibration and actual scanning engine identities differ")
        stage = {**identities[name], "index_sha256": index["sha256"],
                 "calibration_sha256": c._digest(calibration), "actual_engine_identity": identity}
        if name == "ocr":
            stage.update(benchmark={k: payloads[k] for k in index["images"]},
                         inputs={k: payloads[k] for k in human})
        else:
            stage.update(benchmark_images={k: payloads[k] for k in index["images"]},
                         input_images={k: payloads[k] for k in images})
        evidence[name] = stage
    texts = text_entries(index["data"])
    text_shas = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in texts.items()}
    own = {pid: item["text_sha256"] for pid, item in items.items()}
    identity, payloads = cache_payloads(cache_roots["siglip"], set(text_shas.values()) | set(own.values()), texts=True)
    if identity != evidence["siglip"]["actual_engine_identity"]:
        raise ValueError("SigLIP image/text engines differ")
    evidence["siglip"].update(benchmark_texts={key: payloads[sha] for key, sha in text_shas.items()},
                               input_texts={key: payloads[sha] for key, sha in own.items()})
    registry = read_json(source_registry_path)
    if "scanner_policy" in registry:
        raise ValueError("source registry already binds another scanner; require fresh copy")
    registry["scanner_policy"] = {"status": "FROZEN_BEFORE_INPUT_SCAN",
        "protocol_sha256": calibration["protocol_sha256"], "calibration_sha256": c._digest(calibration),
        "engines": identities}
    # Validate genuine complete content before serialization, preserving all detected hits.
    c._calibration(evidence, registry, index["sha256"], calibration["protocol_sha256"])
    for name in identities:
        c._engine(evidence[name], name, evidence, registry, index["sha256"])
    receipt = c.scan({s: build / s for s in SPLITS}, build / "images", index, registry, evidence)
    if receipt.get("input_files_sha256") != c._digest(files):
        raise ValueError("inputs changed between detector assembly and scan")
    output = _private_root(output, create=True)
    _write_once(output / "evidence.json", c._json(evidence))
    _write_once(output / "source_registry.json", c._json(registry))
    c.write_receipt(output / "receipt_v2.json", receipt)
    summary = {"status": receipt["status"], "training_allowed": False,
               "candidate_count": len(receipt["candidates"]), "quarantined": len(receipt["quarantined_ids"]),
               "allowlisted": len(receipt["allowlist"]), "blockers": receipt["blockers"],
               "input_files_sha256": receipt["input_files_sha256"],
               "receipt_file_sha256": hashlib.sha256(vault._read(output / "receipt_v2.json")).hexdigest()}
    _write_once(output / "summary.json", c._json(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-images", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_manifests(args.index, args.raw_images, args.build, args.output)))


if __name__ == "__main__":
    main()
