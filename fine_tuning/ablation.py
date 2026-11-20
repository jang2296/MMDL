"""Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

python -m fine_tuning.ablation --output PATH --count 4 --seed 73041 --start-index 1000000
This is not full G7, professional-domain transfer, or authorization to train.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

from fine_tuning import nvw

VERSION = "boolean-finite-state-ablation-v2"
# The v2 repair changes canonical layout, not the frozen episode permutation.
MAPPING_VERSION = "boolean-finite-state-ablation-v1"
SCOPE = "BOOLEAN_FINITE_STATE_CANONICAL_PROTOTYPE"
ARMS = {"A0": (False, False), "A1": (True, False),
        "A2": (False, True), "A3": (True, True)}
VIEWS = ("qa", "structure", "grounded")
FIXED_MAPPING = dict(zip(nvw.RULES, range(4)))
FLAGS = {"training_allowed": False, "public_upload_allowed": False,
         "benchmark_derived": False}


def exclusion_reason(episode: dict[str, Any]) -> str | None:
    """Reject ancestry/approval taint, then filter unsupported semantic roots."""
    if (episode["benchmark_derived"] or episode["contamination"]["benchmark_derived"]
            or episode["parents"] or episode["source_kind"] != "INDEPENDENT_SYNTHETIC"
            or episode["training_allowed"] or episode["public_upload_allowed"]
            or episode["contamination"]["audit_decision"] != "NOT_AUDITED"):
        raise ValueError("only unapproved independent synthetic roots are accepted")
    if episode["family"] != "finite_state":
        return "UNSUPPORTED_NON_BOOLEAN_FAMILY"
    nvw.validate_episode(episode)
    if len(set(episode["world"]["mapping"])) != 2:
        return "DUPLICATE_SEMANTIC_RULES"
    quality = episode["quality"]
    if not (quality["mapping_identifiable"] and quality["rule_identifiable"]
            and quality["answer_identifiable"]
            and all(quality["state_prefix_identifiable"])):
        return "NOT_FULLY_IDENTIFIABLE"
    return None


def condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]:
    reason = exclusion_reason(episode)
    if reason:
        raise ValueError(reason)
    randomized, bridge = ARMS[arm]
    codes = list(FIXED_MAPPING.values())
    if randomized:
        random.Random(nvw.digest([MAPPING_VERSION, seed, episode["root_id"], "rule_glyph"])).shuffle(codes)
    rule_glyph = dict(zip(FIXED_MAPPING, codes))
    result = copy.deepcopy(episode)
    result.update(condition=arm, scope=SCOPE, bridge_enabled=bridge,
                  mapping_policy="EPISODE_PERMUTATION" if randomized else "FIXED_RULE_GLYPH",
                  rule_glyph_mapping=rule_glyph,
                  glyph_codes=[rule_glyph[rule] for rule in episode["world"]["mapping"]])
    return result


def render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Repair only the Boolean query sequence; retain support/initial bit/options."""
    reason = exclusion_reason(episode)
    if reason:
        raise ValueError(reason)
    image, scene = nvw.render(episode, canonical=True)
    draw = ImageDraw.Draw(image)
    draw.rectangle((21, 550, nvw.WIDTH - 21, 660), fill="white")
    sequence = episode["world"]["sequence"]
    centers = [(220 + i * 250, 635) for i in range(len(sequence))]
    labels = []
    for i, (op, center) in enumerate(zip(sequence, centers)):
        if i:
            nvw.arrow(draw, centers[i - 1], center)
            scene["relations"][i - 1]["segment"] = [
                [centers[i - 1][0] + 35, 635], [center[0] - 35, 635]]
        rule = episode["world"]["mapping"][op]
        bbox = nvw.text(draw, (center[0] - 50, 555), rule, small=True)
        if bbox[3] >= center[1] - 20:
            raise ValueError("canonical rule label intersects arrow band")
        scene["objects"][i]["bbox"] = bbox
        labels.append({"id": i, "operator": op, "rule": rule, "bbox": bbox})
    scene["query_labels"] = labels
    scene["canonical_sequence_layout"] = {"label_y": 555, "arrow_y": 635,
                                          "cleared_band": [21, 550, nvw.WIDTH - 21, 660]}
    return image, scene


def paired_record(episode: dict[str, Any], first: str, companion: str,
                  view: str) -> dict[str, Any]:
    record = nvw.view_record(episode, first, view)
    if view == "grounded":
        record["image_paths"] = record["image"] = [first, companion]
        record["user_text"] = "<image>\n" + record["user_text"]
        record["conversations"][0]["value"] = record["user_text"]
    record.update(FLAGS)
    record.update(id=episode["id"] + "-" + episode["condition"] + "-" + view,
                  condition=episode["condition"], scope=SCOPE,
                  mapping_policy=episode["mapping_policy"],
                  provenance_status="UNAPPROVED_RENDER_AND_MODEL_INPUT_REVIEW_PENDING")
    return record


def bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]:
    """Pure deterministic artifact construction; intentionally no data-file argument."""
    if not 1 <= count <= 100 or seed < 0 or start_index < 1_000_000:
        raise ValueError("prototype requires count 1..100, nonnegative seed, index >= 1000000")
    files: dict[str, bytes] = {}

    def emit(name: str, records: list[dict[str, Any]]) -> None:
        files[name] = ("".join(json.dumps(r, sort_keys=True) + "\n" for r in records)).encode()

    source_dir = Path(__file__).parent
    for name in ("ablation.py", "nvw.py", "clean_spec.json"):
        files["sources/" + name] = (source_dir / name).read_bytes()
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    roots: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = start_index
    while len(roots) < count:
        if index - start_index >= count * 1000:
            raise RuntimeError("exhausted bounded Boolean root search")
        episode = nvw.build_episode("finite_state", index, seed)
        reason = exclusion_reason(episode)
        if episode["root_id"] in seen:
            reason = "DUPLICATE_SEMANTIC_ROOT"
        if reason:
            excluded.append({"index": index, "root_id": episode["root_id"],
                             "family_id": episode["family_id"], "reason": reason, **FLAGS})
        else:
            roots.append(episode)
            seen.add(episode["root_id"])
        index += 1
    episodes, pairings, reviews = [], [], []
    arm_records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for root in roots:
        paired: dict[str, Any] = {}
        for arm, (_, bridge) in ARMS.items():
            episode = condition_episode(root, arm, seed)
            novel, observation = nvw.render(episode)
            episode["observed_structure"] = observation
            nvw.validate_episode(episode)
            canonical, canonical_scene = render_canonical(episode)
            stem = "images/" + root["id"] + "-" + arm
            paths = [stem + "-novel.png", stem + "-companion.png"]
            for path, image in zip(paths, (novel, canonical if bridge else novel)):
                encoded = io.BytesIO()
                image.save(encoded, format="PNG")
                files[path] = encoded.getvalue()
            episode.update(FLAGS)
            episode.update(image=paths[0], companion=paths[1],
                           companion_representation="CANONICAL_BOOLEAN" if bridge else "NOVEL_DUPLICATE",
                           companion_scene=canonical_scene if bridge else observation,
                           image_sha256=hashlib.sha256(files[paths[0]]).hexdigest(),
                           companion_sha256=hashlib.sha256(files[paths[1]]).hexdigest(),
                           source_files_sha256=source_hashes)
            episodes.append(episode)
            records = [paired_record(episode, paths[0], paths[1], view) for view in VIEWS]
            arm_records[arm].extend(records)
            paired[arm] = {"record_ids": [r["id"] for r in records], "images": paths,
                           "rule_glyph_mapping": episode["rule_glyph_mapping"]}
            reviews.append({"root_id": root["root_id"], "condition": arm, "images": paths,
                            "status": "PENDING_RENDER_SEMANTICS_AND_MODEL_PREPROCESSING", **FLAGS})
        pairings.append({"root_id": root["root_id"], "family_id": root["family_id"],
                         "split": root["split"], "conditions": paired,
                         "semantic_sha256": nvw.digest([root["world"], root["states"],
                                                         root["answer"], root["options"]]), **FLAGS})
    emit("canonical_private.jsonl", episodes)
    emit("excluded_roots.jsonl", excluded)
    emit("pairing.jsonl", pairings)
    emit("review_queue.jsonl", reviews)
    for arm, records in arm_records.items():
        emit(arm + ".jsonl", records)
    manifest = {"version": VERSION, "mapping_version": MAPPING_VERSION,
                "scope": SCOPE, "count": count, "seed": seed,
                "start_index": start_index, "next_index": index, "root_count": count,
                "records_per_arm": count * 3, "image_files_per_arm": count * 2,
                "image_references_per_arm": count * 4, "excluded_count": len(excluded),
                "fixed_rule_glyph_mapping": FIXED_MAPPING, "source_files_sha256": source_hashes,
                "family_splits": {r["family_id"]: r["split"] for r in roots},
                "render_assets": {"pillow_version": PILLOW_VERSION,
                                  "font_module_sha256": hashlib.sha256(
                                      Path(ImageFont.__file__).read_bytes()).hexdigest()},
                "gate": "UNAPPROVED_NOT_FOR_TRAINING_OR_PUBLICATION",
                "limitations": ["Boolean finite-state only; canonical render retains support glyphs",
                                "No full G7, professional transfer, model preprocessing, or training",
                                "Equal record/image counts do not certify equal visual/target tokens"],
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                          for name, data in files.items()}, **FLAGS}
    files["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    return files


def generate(output: Path, count: int = 4, seed: int = 73041,
             start_index: int = 1_000_000) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("refuse to overwrite existing output")
    files = bundle(count, seed, start_index)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return json.loads(files["manifest.json"])


def verify(output: Path) -> dict[str, Any]:
    """Compare every byte to fresh deterministic generation, not just stored hashes."""
    if output.is_symlink() or any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("symlinks are not allowed")
    manifest = json.loads((output / "manifest.json").read_bytes())
    expected = bundle(manifest["count"], manifest["seed"], manifest["start_index"])
    actual_names = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual_names != set(expected):
        raise ValueError("artifact inventory mismatch")
    for name, data in expected.items():
        if (output / name).read_bytes() != data:
            raise ValueError("artifact differs from deterministic generation: " + name)
    return {"status": "CPU_REPRODUCED_NOT_APPROVED", "scope": SCOPE,
            "root_count": manifest["root_count"], **FLAGS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73041)
    parser.add_argument("--start-index", type=int, default=1_000_000)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (verify(args.output) if args.verify else
              generate(args.output, args.count, args.seed, args.start_index))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
