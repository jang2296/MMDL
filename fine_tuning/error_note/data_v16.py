"""Private v16 data admission from frozen images and creator annotations (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .audit import digest
from .content_v16 import (MANIFEST_SCHEMA, PROMPTS_SHA256, link_geometry3k,
                          load_content_manifest, observation_row, parse_gvlqa_structure)
from ..notebook_rl import contamination

SCHEMA = "v16-data-admission-v1"
ALLOWED = {"INCLUDED", "EXCLUDED_QUOTA"}
LAYERS = {"hf-geometry3k": "geometry", "hf-gvlqa": "graph", "hf-clevr": "scene",
          "hf-chartqa": "chart"}


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_pinned(path: str | Path, expected: str) -> dict:
    if file_sha256(path) != expected:
        raise ValueError(f"v16 data: HASH_MISMATCH:{path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source(path: Path, role: str, **kwargs: object) -> dict:
    return {"path": str(path.resolve(strict=True)), "sha256": file_sha256(path),
            "role": role, **kwargs}


def _records(path: Path) -> dict[str, tuple[dict, str]]:
    result = {}
    for raw in path.read_bytes().splitlines():
        row = json.loads(raw)
        pid = row["problem_id"]
        if pid in result:
            raise ValueError("v16 data: DUPLICATE_PARENT_RECORD")
        result[pid] = row, hashlib.sha256(raw).hexdigest()
    return result


def geometry_catalog(train_zip: Path, logic_zip: Path) -> dict[str, dict]:
    """Rebuild the image-keyed catalog; never trust a saved row-number join."""
    with zipfile.ZipFile(logic_zip) as archive:
        logic = json.loads(archive.read("logic_forms/diagram_logic_forms_annot.json"))
    result = {}
    with zipfile.ZipFile(train_zip) as archive:
        for name in sorted(archive.namelist()):
            if not re.fullmatch(r"train/\d+/data\.json", name):
                continue
            raw = json.loads(archive.read(name))
            key = name.split("/")[1]
            result[key] = {
                "image_sha256": hashlib.sha256(archive.read(
                    name.replace("data.json", "img_diagram.png"))).hexdigest(),
                "answer_value": raw["choices"]["ABCD".index(raw["answer"])],
                "texts": [raw.get("annotat_text", ""), raw.get("problem_text", "")],
                "diagram_logic_forms": logic[key]["diagram_logic_forms"],
            }
    return result


def _image_checked(path: Path, entry: dict, receipt: dict) -> None:
    from PIL import Image

    actual = file_sha256(path)
    if actual != entry["image_sha256"][0] or receipt["input_files"].get(str(path)) != actual:
        raise ValueError("v16 data: IMAGE_RECEIPT_MISMATCH")
    with Image.open(path) as image:
        pixels = contamination.image_fingerprints(image)["pixels_sha256"]
    if pixels != entry["pixel_sha256"][0]:
        raise ValueError("v16 data: PIXEL_HASH_MISMATCH")


def _receipt_checked(shard: dict, index_sha256: str) -> tuple[dict, dict]:
    path = Path(shard["root"]) / "contamination/receipt_v2.json"
    receipt = read_pinned(path, shard["exclusion_receipt_file_sha256"])
    if (receipt["index_file_sha256"] != index_sha256 or receipt["blockers"]
            or receipt["status"] != "EXCLUSION_CHECKED_WITH_LIMITS"
            or set(receipt["steps"]) != set(contamination._STEPS)
            or any(step["status"] != "COMPLETE" for step in receipt["steps"].values())
            or receipt.get("adjudication", {}).get("human_verified") is not False):
        raise ValueError("v16 data: INCOMPLETE_HISTORICAL_EXCLUSION")
    return receipt, _source(path, "historical_image_exclusion")


def source_items(parent_path: Path, parent_sha256: str, sources: dict,
                 index_sha256: str) -> tuple[list[dict], list[dict]]:
    """Only historically cleared parent image bytes may enter the new admission."""
    parent = read_pinned(parent_path, parent_sha256)
    parquet = Path(sources["geometry_parquet"])
    import pyarrow.parquet as pq

    local_rows = pq.read_table(parquet).to_pylist()
    train_zip, logic_zip = Path(sources["geometry_train_zip"]), Path(sources["geometry_logic_zip"])
    catalog = geometry_catalog(train_zip, logic_zip)
    by_image: dict[str, dict] = defaultdict(dict)
    for key, value in catalog.items():
        by_image[value["image_sha256"]][key] = value
    scenes_path, questions_path = Path(sources["scene_json"]), Path(sources["scene_questions_json"])
    scenes = {s["image_filename"]: s for s in json.loads(scenes_path.read_text())["scenes"]}
    questions = {q["question_index"]: q for q in json.loads(questions_path.read_text())["questions"]}
    render_path = Path(sources["graph_rendering_review"])
    render = json.loads(render_path.read_text())
    graph_dir = Path(sources["graph_structure_dir"])
    charts = sources.get("chart_items", {})
    shared = {role: _source(path, role) for role, path in (
        ("geometry_problem_zip", train_zip), ("geometry_logic_zip", logic_zip),
        ("geometry_local_parquet", parquet), ("scene_json", scenes_path),
        ("scene_questions_json", questions_path), ("graph_rendering_review", render_path))}
    shard_cache = {}
    items, rejected = [], []
    for entry in parent["items"]:
        if entry["source"] not in LAYERS or entry["status"] not in ALLOWED:
            continue
        pid, layer = entry["problem_id"], LAYERS[entry["source"]]
        try:
            shard_id = entry["shard"]
            root = Path(parent["shards"][shard_id]["root"])
            if shard_id not in shard_cache:
                receipt, receipt_source = _receipt_checked(parent["shards"][shard_id], index_sha256)
                shard_cache[shard_id] = (receipt, receipt_source,
                    _records(root / "train/actor_inputs.jsonl"),
                    _records(root / "train/verifier_private.jsonl"),
                    _records(root / "train/lineage.jsonl"))
            receipt, receipt_source, actors, verifiers, lineages = shard_cache[shard_id]
            actor, verifier, lineage = actors[pid][0], verifiers[pid][0], lineages[pid][0]
            if (entry["split"] != "train" or pid not in receipt["allowlist"]
                    or len(entry["image_paths"]) != 1
                    or lineage["benchmark_derived"] is not False
                    or lineage["model_touched"] is not False
                    or any(rows[pid][1] != entry["record_sha256"][key] for rows, key in
                           ((actors, "actor"), (verifiers, "verifier"), (lineages, "lineage")))):
                raise ValueError("UNADMITTED_PARENT")
            image_path = root / "images" / entry["image_paths"][0]
            _image_checked(image_path, entry, receipt)
            provenance = {"parent_manifest_path": str(parent_path),
                "parent_manifest_sha256": parent_sha256, "parent_record_sha256": digest(entry),
                "parent_status": entry["status"], "actor_path": str(root / "train/actor_inputs.jsonl"),
                "verifier_path": str(root / "train/verifier_private.jsonl"),
                "historical_exclusion": receipt_source, "human_verified": False}
            item = {"problem_id": pid, "parent_id": pid, "group_id": entry["image_sha256"][0],
                "source_repo": lineage["image_provenance"].removeprefix("hf:").split("@")[0],
                "source_revision": entry["source_revision"], "source_split": "train",
                "original_split": "train", "split": "train", "training_allowed": True,
                "benchmark_derived": False, "human_verified": False, "layer": layer,
                "label_source": "HUMAN_ANNOTATION" if layer in {"geometry", "chart"} else "PROGRAM_EXACT",
                "image_path": str(image_path), "image_sha256": entry["image_sha256"][0],
                "pixel_sha256": entry["pixel_sha256"][0], "question": actor["question"],
                "choices": actor["choices"], "gold": verifier["gold"],
                "answer_type": verifier["answer_type"], "accept": verifier["accept"],
                "provenance": provenance}
            item["evaluation"] = {key: item[key] for key in
                ("question", "choices", "gold", "answer_type", "accept")}
            if layer == "geometry":
                index = int(entry["native_id"].split(":")[1])
                raw = local_rows[index]
                local = {"index": index, "problem": raw["problem"], "answer": raw["answer"],
                    "image_sha256": hashlib.sha256(raw["images"][0]["bytes"]).hexdigest()}
                candidates = by_image[local["image_sha256"]]
                joined = link_geometry3k([local], candidates)[index]
                if joined == "LINK_AMBIGUOUS":
                    raise ValueError("LINK_AMBIGUOUS")
                annotation = {"local": local, "candidates": candidates, "matched_id": joined}
                item.update(annotation=annotation, source_image_sha256=local["image_sha256"],
                    group_id=local["image_sha256"], join_method="image-answer-text-v1",
                    join_receipt_sha256=digest(annotation), annotation_process_documented=False)
                provenance["annotation_sources"] = [
                    shared["geometry_problem_zip"], shared["geometry_logic_zip"],
                    {**shared["geometry_local_parquet"], "index": index}]
            elif layer == "graph":
                group = re.sub(r"-\d+_\d+$", "", entry["native_id"])
                task, level, graph_id = group.split("-")
                path = graph_dir / f"{group}.txt"
                struct = parse_gvlqa_structure(path.read_text())
                item.update(group_id=group, visible_nodes=struct["visible_nodes"],
                    annotation={"structure_text": path.read_text(), "directed": task == "flow",
                        "weighted": task == "flow", "task": task, "level": level, "graph_id": graph_id})
                provenance.update(rendering_verified=True,
                    rendering_scope="task_direction_weight_convention_ai_spotcheck_not_all_images",
                    rendering_review_sha256=digest(render),
                    annotation_sources=[_source(path, "graph_structure"),
                                        shared["graph_rendering_review"]])
            elif layer == "scene":
                question = questions[int(entry["native_id"].split(":")[1])]
                name = question["image_filename"]
                item.update(image_filename=name, group_id=name, annotation=scenes[name])
                provenance["annotation_sources"] = [shared["scene_json"], shared["scene_questions_json"]]
            else:
                if pid not in charts:
                    raise ValueError(sources.get("chart_rejections", {}).get(pid, "VISIBILITY_UNVERIFIED"))
                chart = dict(charts[pid])
                provenance["annotation_sources"] = chart.pop("annotation_sources")
                item.update(chart)
            observation_row(item, layer=layer)
            items.append(item)
        except (ValueError, KeyError, IndexError) as exc:
            rejected.append({"problem_id": pid, "layer": layer, "reason": str(exc)})
    return items, rejected


def excluded_holdouts(manifests: list[Path]) -> set[str]:
    """Old dev/sealed and their entire shared image families stay unavailable."""
    excluded = set()
    for path in manifests:
        for item in json.loads(path.read_text())["items"]:
            if item["split"] in {"dev", "sealed", "style_sealed"} or item["status"] in {
                    "HELD_OUT_DEV", "HELD_OUT_SEALED"}:
                excluded.update([item["problem_id"], *item["image_sha256"], *item["pixel_sha256"]])
    return excluded


def _per_block(quota: dict[str, int], block_size: int) -> tuple[int, dict[str, int]]:
    total = sum(quota.values())
    if block_size < 1 or total < 1 or total % block_size:
        raise ValueError("v16 data: QUOTA_NOT_INTERLEAVABLE")
    blocks = total // block_size
    if any(count % blocks for count in quota.values()):
        raise ValueError("v16 data: QUOTA_NOT_INTERLEAVABLE")
    per_block = {layer: count // blocks for layer, count in quota.items()}
    if sum(per_block.values()) != block_size:
        raise ValueError("v16 data: QUOTA_NOT_INTERLEAVABLE")
    return blocks, per_block


def interleave_train(by_layer: dict[str, list[dict]], quota: dict[str, int],
                     block_size: int) -> list[dict]:
    """Every block of block_size parents carries the same per-layer mix, in hash order."""
    blocks, per_block = _per_block(quota, block_size)
    return [x for b in range(blocks) for layer in sorted(quota)
            for x in by_layer[layer][b * per_block[layer]:(b + 1) * per_block[layer]]]


def admit_and_split(items: list[dict], index: dict, holdouts: set[str], *, seed: int = 3407,
                    train_quota: dict[str, int], sentinel_quota: dict[str, int],
                    dev_per_layer: int = 64, block_size: int = 8) -> tuple[list[dict], list[dict]]:
    """Frozen detector thresholds; any fresh text candidate quarantines the whole group."""
    if (set(train_quota) != set(sentinel_quota) or not set(train_quota) <= set(LAYERS.values())
            or min(train_quota.values(), default=0) < 1 or min(sentinel_quota.values()) < 0
            or dev_per_layer < 0):
        raise ValueError("v16 data: INVALID_SPLIT_SIZE")
    _per_block(train_quota, block_size)
    rejected, blocked = [], set()
    for item in items:
        group = (item["layer"], item["group_id"])
        hits = []
        for field, text in (("target", observation_row(item, layer=item["layer"])["target"]),):
            hits += [{"field": field, **hit} for hit in contamination._text_matches(
                text, field, index["text_cache"])]
        if hits or holdouts.intersection((item["problem_id"], item["image_sha256"],
                                         item["pixel_sha256"], item["group_id"])):
            blocked.add(group)
            rejected.append({"problem_id": item["problem_id"], "layer": item["layer"],
                "reason": "NEW_TARGET_TEXT_MATCH" if hits else "EXISTING_HOLDOUT_FAMILY",
                "candidates": hits})
    changed = True
    while changed:
        keys = {item[key] for item in items if (item["layer"], item["group_id"]) in blocked
                for key in ("image_sha256", "pixel_sha256")}
        linked = {(item["layer"], item["group_id"]) for item in items
                  if keys.intersection((item["image_sha256"], item["pixel_sha256"]))}
        changed = not linked <= blocked
        blocked.update(linked)
    representatives = {}
    for item in sorted(items, key=lambda x: x["problem_id"]):
        group = (item["layer"], item["group_id"])
        if group in blocked:
            continue
        if group in representatives:
            rejected.append({"problem_id": item["problem_id"], "layer": item["layer"],
                             "reason": "DUPLICATE_IMAGE_GROUP"})
        else:
            representatives[group] = item
    train: dict[str, list[dict]] = {}
    rest: list[dict] = []
    for layer in sorted(train_quota):
        ordered = sorted((x for (kind, _), x in representatives.items() if kind == layer),
                         key=lambda x: digest({"seed": seed, "group_id": x["group_id"]}))
        head = train_quota[layer] + sentinel_quota[layer]
        if len(ordered) < head:
            raise ValueError(f"v16 data: INSUFFICIENT_ELIGIBLE_DATA:{layer}")
        dev_end = head + min(dev_per_layer, len(ordered) - head)
        train[layer] = [{**x, "split": "train"} for x in ordered[:train_quota[layer]]]
        rest += [{**x, "split": "sentinel" if i < head else "dev" if i < dev_end else "sealed"}
                 for i, x in enumerate(ordered) if i >= train_quota[layer]]
    result = interleave_train(train, train_quota, block_size) + rest
    seen: dict[str, str] = {}
    for item in result:
        for key in ("image_sha256", "pixel_sha256", "group_id"):
            value = f"{key}:{item[key]}"
            if value in seen and seen[value] != item["split"]:
                raise ValueError("v16 data: SPLIT_FAMILY_LEAK")
            seen[value] = item["split"]
    return result, rejected


def prepare(*, parent_path: Path, parent_sha256: str, sources_path: Path,
            index_path: Path, index_sha256: str, holdout_paths: list[Path], output: Path,
            train_quota: dict[str, int], sentinel_quota: dict[str, int],
            dev_per_layer: int = 64, block_size: int = 8) -> dict:
    sources = json.loads(sources_path.read_text())
    items, rejected = source_items(parent_path, parent_sha256, sources, index_sha256)
    rejected += [{"problem_id": x["problem_id"], "layer": x["layer"], "reason": "LAYER_NOT_SELECTED"}
                 for x in items if x["layer"] not in train_quota]
    items = [x for x in items if x["layer"] in train_quota]
    if file_sha256(index_path) != index_sha256:
        raise ValueError("v16 data: INDEX_HASH_MISMATCH")
    index = contamination.load_index(index_path)
    if not index["full_coverage"]:
        raise ValueError("v16 data: INCOMPLETE_BENCHMARK_COVERAGE")
    items, decisions = admit_and_split(items, index, excluded_holdouts(holdout_paths),
        train_quota=train_quota, sentinel_quota=sentinel_quota,
        dev_per_layer=dev_per_layer, block_size=block_size)
    rejected += decisions
    manifest = {"schema": MANIFEST_SCHEMA, "prompts_sha256": PROMPTS_SHA256,
        "human_verified": False, "items": items,
        "rows": [observation_row(item, layer=item["layer"]) for item in items if item["split"] == "train"],
        "admission": {"schema": SCHEMA, "parent_manifest_sha256": parent_sha256,
            "index_sha256": index_sha256, "sources_path": str(sources_path),
            "sources_sha256": file_sha256(sources_path),
            "holdouts": [_source(path, "existing_holdouts") for path in holdout_paths],
            "seed": 3407, "layers": sorted(train_quota), "train_quota": train_quota,
            "sentinel_quota": sentinel_quota, "dev_per_layer": dev_per_layer,
            "block_size": block_size,
            "image_checks": "exact_parent_bytes_reuse_complete_historical_detectors",
            "text_checks": "fresh_unchanged_exact_and_5_7_8gram_detector_including_O_targets",
            "new_images": 0, "limitations": [
                "Historical OCR/SSCD/SigLIP evidence reused, not rerun or independently authenticated",
                "Exclusion does not certify absence of pretraining exposure or every semantic paraphrase",
                "Creator annotation accuracy and rendered pixels are not exhaustively human verified"]}}
    output.mkdir(parents=True, exist_ok=True)
    check_path = output / "exclusion-check.json"
    check = {"schema": "v16-data-exclusion-check-v1", "status": "EXCLUSION_CHECKED_WITH_LIMITS",
        "payload_sha256": digest({"items": manifest["items"], "rows": manifest["rows"]}),
        "index_path": str(index_path), "index_sha256": index_sha256,
        "coverage": {"records": index["data"]["record_count"],
            "image_occurrences": index["data"]["image_count"],
            "source_files": len(index["data"]["source_files"])},
        "detector_sha256": file_sha256(Path(contamination.__file__)),
        "data_builder_sha256": file_sha256(Path(__file__)),
        "retained_target_candidates": 0, "new_images": 0,
        "new_target_candidates": [x for x in decisions if x["reason"] == "NEW_TARGET_TEXT_MATCH"],
        "split_bindings_sha256": digest([(x["problem_id"], x["group_id"], x["split"]) for x in items]),
        "human_verified": False}
    if check_path.exists():
        raise ValueError("v16 data: OUTPUT_ALREADY_EXISTS")
    check_path.write_text(json.dumps(check, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest["admission"]["exclusion_receipt"] = _source(check_path, "v16_target_exclusion")
    path = output / "content-manifest.json"
    if path.exists():
        raise ValueError("v16 data: OUTPUT_ALREADY_EXISTS")
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    pinned = file_sha256(path)
    load_content_manifest(path, pinned)
    validate_admission_binding(manifest)
    summary = {"schema": SCHEMA, "status": "CPU_DATA_ADMITTED_WITH_LIMITS",
        "manifest_path": str(path), "manifest_sha256": pinned,
        "counts": dict(Counter(f"{x['layer']}/{x['split']}" for x in items)),
        "training_rows": len(manifest["rows"]),
        "l0_ready": True,
        "rejections": rejected,
        "rejection_counts": dict(Counter(x["reason"] for x in rejected)),
        "human_verified": False, "gpu_executed": False}
    (output / "admission.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return summary


def validate_admission_binding(manifest: dict) -> None:
    """Verify the CPU scan receipt without loading benchmark text into the trainer."""
    admission = manifest.get("admission", {})
    descriptor = admission.get("exclusion_receipt", {})
    if admission.get("schema") != SCHEMA or not descriptor:
        raise ValueError("v16 data: MISSING_ADMISSION")
    receipt = read_pinned(descriptor["path"], descriptor["sha256"])
    payload = {"items": manifest["items"], "rows": manifest["rows"]}
    if (receipt.get("schema") != "v16-data-exclusion-check-v1"
            or receipt.get("status") != "EXCLUSION_CHECKED_WITH_LIMITS"
            or receipt.get("payload_sha256") != digest(payload)
            or receipt.get("index_sha256") != admission["index_sha256"]
            or receipt.get("detector_sha256") != file_sha256(Path(contamination.__file__))
            or receipt.get("data_builder_sha256") != file_sha256(Path(__file__))
            or receipt.get("new_images") != 0 or receipt.get("retained_target_candidates") != 0):
        raise ValueError("v16 data: ADMISSION_BINDING_MISMATCH")
    if file_sha256(receipt["index_path"]) != receipt["index_sha256"]:
        raise ValueError("v16 data: INDEX_HASH_MISMATCH")
    holdout_paths = []
    for source in admission["holdouts"]:
        read_pinned(source["path"], source["sha256"])
        holdout_paths.append(Path(source["path"]))
    held = excluded_holdouts(holdout_paths)
    checked = {}
    for item in manifest["items"]:
        if held.intersection((item["parent_id"], item["image_sha256"],
                              item["pixel_sha256"], item["group_id"])):
            raise ValueError("v16 data: EXISTING_HOLDOUT_FAMILY")
        source = item["provenance"]["historical_exclusion"]
        if source["path"] not in checked:
            historical = read_pinned(source["path"], source["sha256"])
            if (historical["status"] != "EXCLUSION_CHECKED_WITH_LIMITS"
                    or historical["index_file_sha256"] != admission["index_sha256"]
                    or historical["blockers"] or set(historical["steps"]) != set(contamination._STEPS)
                    or any(step["status"] != "COMPLETE" for step in historical["steps"].values())):
                raise ValueError("v16 data: INCOMPLETE_HISTORICAL_EXCLUSION")
            checked[source["path"]] = historical
        historical = checked[source["path"]]
        if (item["parent_id"] not in historical["allowlist"]
                or historical["input_files"].get(item["image_path"]) != item["image_sha256"]):
            raise ValueError("v16 data: IMAGE_RECEIPT_MISMATCH")


def _quota(text: str) -> dict[str, int]:
    return {key: int(value) for key, value in (part.split("=", 1) for part in text.split(","))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--holdout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-quota", type=_quota, default="geometry=48,graph=32,scene=48")
    parser.add_argument("--sentinel-quota", type=_quota, default="geometry=6,graph=4,scene=6")
    parser.add_argument("--dev-per-layer", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()
    result = prepare(parent_path=args.parent, parent_sha256=args.parent_sha256,
        sources_path=args.sources, index_path=args.index, index_sha256=args.index_sha256,
        holdout_paths=args.holdout, output=args.output, train_quota=args.train_quota,
        sentinel_quota=args.sentinel_quota, dev_per_layer=args.dev_per_layer,
        block_size=args.block_size)
    print(json.dumps({k: v for k, v in result.items() if k != "rejections"}, sort_keys=True))


if __name__ == "__main__":
    main()
