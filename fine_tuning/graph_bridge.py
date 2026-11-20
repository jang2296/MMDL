"""Quarantined independent Graph/CS representation bridge; partial G7 only.

python -m fine_tuning.graph_bridge --output NEW_PATH --count 4 --seed 73041
No source data, benchmark, model, or cloud input is accepted. This is synthetic
bounded unweighted reachability, not evidence of non-synthetic transfer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

from fine_tuning import nvw
from fine_tuning.ablation import ARMS, FLAGS, VIEWS

VERSION = "graph-cs-reachability-bridge-v2"
# Preserve role, label and layout randomization; v2 changes root eligibility only.
MAPPING_VERSION = "graph-cs-reachability-bridge-v1"
SCOPE = "PARTIAL_INDEPENDENT_SYNTHETIC_GRAPH_CS_REPRESENTATION_BRIDGE"
SOURCE_ID = "independent-graph-cs-bridge-v2"
FIXED_MAPPING = {"SOURCE/TAIL": 0, "TARGET/HEAD": 1}
ROLES = tuple(FIXED_MAPPING)


def font_sha256(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str:
    if not isinstance(font, ImageFont.FreeTypeFont) or not isinstance(font.path, io.BytesIO):
        raise ValueError("Pillow bundled in-memory TrueType font required for provenance")
    return hashlib.sha256(font.path.getvalue()).hexdigest()


def shortcut_counts(world: dict[str, Any]) -> dict[str, int]:
    edges = world["edges"]
    undirected = [*edges, *[[b, a] for a, b in edges]]
    return {
        "directed_final": len(nvw.solve_signal(world)[-1]),
        "one_step_directed": len(nvw.solve_signal({**world, "steps": 1})[-1]),
        "one_step_undirected": len(nvw.solve_signal({**world, "edges": undirected, "steps": 1})[-1]),
        "bounded_undirected": len(nvw.solve_signal({**world, "edges": undirected})[-1]),
        "reversed": len(nvw.solve_signal({**world, "edges": [[b, a] for a, b in edges]})[-1]),
    }


def exclusion_reason(episode: dict[str, Any]) -> str | None:
    if (episode["benchmark_derived"] or episode["contamination"]["benchmark_derived"]
            or episode["parents"] or not episode["all_parents_known"]
            or episode["source_kind"] != "INDEPENDENT_SYNTHETIC"
            or episode["training_allowed"] or episode["public_upload_allowed"]
            or episode["contamination"]["audit_decision"] != "NOT_AUDITED"):
        raise ValueError("only unapproved independent synthetic roots are accepted")
    if episode["family"] != "signal_graph":
        return "UNSUPPORTED_NON_GRAPH_FAMILY"
    nvw.validate_episode(episode)
    world = episode["world"]
    if (not 6 <= world["n"] <= 8 or not 1 <= world["steps"] <= 4
            or world["update"] != "synchronous_monotone"
            or nvw.solve_signal(world) != nvw.verify_signal(world)):
        raise ValueError("unsupported bounded graph/update contract")
    if len(world["edges"]) < 2 or episode["states"][0] == episode["states"][-1]:
        return "EMPTY_OR_UNINFORMATIVE_GRAPH"
    counts = shortcut_counts(world)
    if counts["directed_final"] == counts["one_step_directed"]:
        return "NO_POST_STEP_ONE_GROWTH"
    if counts["directed_final"] in (counts["one_step_undirected"], counts["bounded_undirected"], counts["reversed"]):
        return "DIRECTION_FREE_SHORTCUT_MATCHES_QA_TARGET"
    return None


def condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]:
    reason = exclusion_reason(episode)
    if reason:
        raise ValueError(reason)
    randomized, bridge = ARMS[arm]
    codes = list(FIXED_MAPPING.values())
    if randomized:
        random.Random(nvw.digest([MAPPING_VERSION, seed, episode["root_id"], "endpoint_roles"])).shuffle(codes)
    result = copy.deepcopy(episode)
    labels = list("ABCDEFGH"[:episode["world"]["n"]])
    label_rng = random.Random(nvw.digest([MAPPING_VERSION, seed, episode["root_id"], "node_labels"]))
    edges = episode["world"]["edges"]
    for _ in range(256):
        label_rng.shuffle(labels)
        if (any(labels[a] < labels[b] for a, b in edges)
                and any(labels[a] > labels[b] for a, b in edges)):
            break
    else:
        raise ValueError("NO_NON_TOPOLOGICAL_LABELS_WITHIN_BOUNDED_SEARCH")
    result.update(condition=arm, scope=SCOPE, bridge_enabled=bridge,
                  endpoint_role_mapping=dict(zip(ROLES, codes)),
                  node_labels=labels,
                  shortcut_audit=shortcut_counts(episode["world"]),
                  mapping_policy="EPISODE_ROLE_PERMUTATION" if randomized else "FIXED_ENDPOINT_ROLES",
                  source_id=SOURCE_ID, lineage=[SOURCE_ID, "independent-nvw-v1"],
                  independent_sources=[SOURCE_ID, "nvw-independent-procedural-v1"],
                  generator_parents=["independent-nvw-v1"], generator_version=VERSION)
    return result


def connector(start: list[int], end: list[int]) -> tuple[list[float], ...]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance < 170:
        raise ValueError("nodes too close")
    ux, uy = dx / distance, dy / distance
    return tuple([start[0] + ux * d, start[1] + uy * d]
                 for d in (35, distance - 35, 75, distance - 75))


def segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    scale = max(0., min(1., ((point[0] - start[0]) * dx
                            + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)


def validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None:
    if len(positions) != world["n"] or len({tuple(p) for p in positions}) != world["n"]:
        raise ValueError("layout node inventory mismatch")
    if any(not (80 <= x <= 1120 and 410 <= y <= 820) for x, y in positions):
        raise ValueError("layout outside query panel")
    edges = world["edges"]
    # Both horizontal directions prevent a global left-to-right direction shortcut.
    if not (any(positions[a][0] < positions[b][0] for a, b in edges)
            and any(positions[a][0] > positions[b][0] for a, b in edges)):
        raise ValueError("layout exposes a global direction shortcut")
    segments = [connector(positions[a], positions[b]) for a, b in edges]
    markers = []
    for i, ((a, b), (start, end, tail, head)) in enumerate(zip(edges, segments)):
        for node, node_point in enumerate(positions):
            if node not in (a, b) and segment_distance(node_point, start, end) < 45:
                raise ValueError("connector intersects unrelated node")
        for owner, endpoint_point in ((a, tail), (b, head)):
            if any(math.dist(endpoint_point, center) <= 95 for node, center in enumerate(positions)
                   if node != owner):
                raise ValueError("endpoint glyph ambiguous with another node")
            if any(segment_distance(endpoint_point, s[0], s[1]) < 25
                   for j, s in enumerate(segments) if i != j):
                raise ValueError("another connector crosses an endpoint glyph")
            markers.append(endpoint_point)
    if any(math.dist(a, b) < 45 for i, a in enumerate(markers) for b in markers[i + 1:]):
        raise ValueError("endpoint glyphs overlap")


def layout(episode: dict[str, Any], variant: str) -> list[list[int]]:
    if variant not in ("standard", "standard_relayout"):
        raise ValueError("unknown layout variant")
    rng = random.Random(nvw.digest([MAPPING_VERSION, episode["seeds"]["layout"],
                                   episode["root_id"], variant]))
    for _ in range(256):
        # Independent permutation of node IDs, never topological columns.
        slots = [[x + rng.randrange(-20, 21), y + rng.randrange(-15, 16)]
                 for x in (170, 580, 990) for y in (450, 620, 790)]
        rng.shuffle(slots)
        positions = slots[:episode["world"]["n"]]
        try:
            validate_layout(episode["world"], positions)
        except ValueError:
            continue
        return positions
    raise ValueError("NO_LEGIBLE_LAYOUT_WITHIN_BOUNDED_SEARCH")


def render(episode: dict[str, Any], representation: str,
           positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]:
    reason = exclusion_reason(episode)
    if reason:
        raise ValueError(reason)
    if representation not in ("novel", "standard", "standard_relayout"):
        raise ValueError("unknown representation")
    mapping = episode["endpoint_role_mapping"]
    if set(mapping) != set(ROLES) or sorted(mapping.values()) != [0, 1]:
        raise ValueError("endpoint role mapping must be a bijection")
    world = episode["world"]
    node_labels = episode["node_labels"]
    if (len(node_labels) != world["n"] or len(set(node_labels)) != world["n"]
            or set(node_labels) != set("ABCDEFGH"[:world["n"]])):
        raise ValueError("visible node labels must be a bijection")
    if not (any(node_labels[a] < node_labels[b] for a, b in world["edges"])
            and any(node_labels[a] > node_labels[b] for a, b in world["edges"])):
        raise ValueError("visible node labels expose a topological shortcut")
    validate_layout(world, positions)
    image = Image.new("RGB", (nvw.WIDTH, nvw.HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    printed = []

    def label(xy: tuple[int, int], value: str, small: bool = False) -> None:
        printed.append(value)
        nvw.text(draw, xy, value, small=small)

    label((30, 20), "Graph/CS: bounded unweighted directed reachability")
    draw.rectangle((20, 65, 1180, 330), outline="black", width=2)
    label((35, 78), "LEGEND: node IDs identify the same nodes in all representations.")
    label((35, 112), "Double-ring nodes are initially active; single-ring nodes are initially inactive.", True)
    if representation == "novel":
        for i, role in enumerate(ROLES):
            nvw.glyph(draw, 70 + i * 540, 170, mapping[role], radius=18)
            label((105 + i * 540, 158), "= " + role + " endpoint", True)
        label((35, 205), "Connectors have no arrowheads. Follow from SOURCE/TAIL to TARGET/HEAD.", True)
    else:
        label((35, 158), "A directed arrow runs from SOURCE/TAIL to TARGET/HEAD.")
        label((35, 205), "All edges are unweighted (unit edges).")
    label((35, 244), "At each step, retain active nodes and add their outgoing neighbors simultaneously.", True)
    label((35, 282), f"Exactly {world['steps']} synchronous steps. Crossings have no junction.", True)
    draw.rectangle((20, 350, 1180, 875), outline="black", width=2)
    relations = []
    for a, b in world["edges"]:
        start, end, tail, head = connector(positions[a], positions[b])
        relation = {"type": "DIRECTED_TO", "from": node_labels[a], "to": node_labels[b],
                    "segment": [start, end]}
        if representation == "novel":
            draw.line([tuple(start), tuple(end)], fill="black", width=3)
            relation["endpoints"] = [
                {"role": role, "node_id": node_labels[node], "glyph_code": mapping[role],
                 "center": [round(v) for v in point]}
                for role, node, point in ((ROLES[0], a, tail), (ROLES[1], b, head))]
        else:
            nvw.arrow(draw, tuple(positions[a]), tuple(positions[b]))
        relations.append(relation)
    if representation == "novel":
        for relation in relations:
            for endpoint in relation["endpoints"]:
                cx, cy = endpoint["center"]
                nvw.glyph(draw, int(cx), int(cy), endpoint["glyph_code"], radius=18)
    objects = []
    for node, (x, y) in enumerate(positions):
        bbox = [x - 28, y - 28, x + 28, y + 28]
        draw.ellipse(bbox, fill="white", outline="black", width=2)
        if node in world["initial"]:
            draw.ellipse((x - 23, y - 23, x + 23, y + 23), outline="black", width=2)
        label((x - 6, y - 12), node_labels[node])
        objects.append({"id": node_labels[node], "center": [x, y], "bbox": bbox,
                        "initially_active": node in world["initial"]})
    label((40, 900), episode["question"])
    label((40, 948), "    ".join(f"{chr(65 + i)}. {v}" for i, v in enumerate(episode["options"])))
    scene = {"image_size": [nvw.WIDTH, nvw.HEIGHT], "coordinate_system": "pixel_xy_top_left",
             "objects": objects, "relations": relations,
             "legend": ([{"role": role, "glyph_code": mapping[role]} for role in ROLES]
                        if representation == "novel" else [{"meaning": "directed arrow"}]),
             "representation": representation, "printed_text": printed}
    return image, scene


def paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]:
    record = nvw.view_record(episode, paths[0], view)
    if view == "grounded":
        record["image_paths"] = record["image"] = paths
        record["user_text"] = "<image>\n<image>\n" + record["user_text"]
        record["conversations"][0]["value"] = record["user_text"]
        target = json.loads(record["target_text"])
        for state in target["states"]:
            state["state"] = sorted(episode["node_labels"][node] for node in state["state"])
        record["target_text"] = json.dumps(target, sort_keys=True)
        record["conversations"][1]["value"] = record["target_text"]
    record.update(FLAGS)
    record.update(id=episode["id"] + "-" + episode["condition"] + "-" + view,
                  condition=episode["condition"], scope=SCOPE,
                  mapping_policy=episode["mapping_policy"],
                  provenance_status="UNAPPROVED_RENDER_AND_MODEL_INPUT_REVIEW_PENDING")
    return record


def bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]:
    if not 1 <= count <= 100 or seed < 0 or start_index < 1_000_000:
        raise ValueError("prototype requires count 1..100, nonnegative seed, index >= 1000000")
    files = {"sources/" + name: Path(__file__).with_name(name).read_bytes()
             for name in ("graph_bridge.py", "nvw.py", "ablation.py", "clean_spec.json")}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    episodes, pairings, reviews, exclusions = [], [], [], []
    arm_records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    roots: set[str] = set()
    index = start_index
    while len(roots) < count:
        if index - start_index >= count * 1000:
            raise RuntimeError("exhausted bounded informative graph/layout search")
        root = nvw.build_episode("signal_graph", index, seed)
        reason = exclusion_reason(root)
        if root["root_id"] in roots:
            reason = "DUPLICATE_SEMANTIC_ROOT"
        if reason is None:
            try:
                positions = layout(root, "standard")
                relayout = layout(root, "standard_relayout")
                if positions == relayout:
                    raise ValueError("relayout is unchanged")
            except ValueError as error:
                reason = str(error)
        if reason:
            exclusions.append({"index": index, "root_id": root["root_id"], "reason": reason, **FLAGS})
            index += 1
            continue
        paired = {}
        for arm, (_, bridge) in ARMS.items():
            episode = condition_episode(root, arm, seed)
            representations = ("novel", "standard", "standard_relayout")
            rendered = [render(episode, kind, relayout if kind == "standard_relayout" else positions)
                        for kind in representations]
            paths = ["images/" + root["id"] + "-" + arm + "-" + kind + ".png"
                     for kind in representations]
            scenes = []
            for i, path in enumerate(paths):
                image, scene = rendered[i if bridge else 0]
                data = io.BytesIO()
                image.save(data, format="PNG")
                files[path] = data.getvalue()
                scenes.append(scene)
            episode.update(FLAGS)
            episode.update(image=paths[0], image_paths=paths, observed_structure=scenes[0],
                           representation_scenes=scenes,
                           representations=list(representations) if bridge else ["novel"] * 3,
                           image_sha256=hashlib.sha256(files[paths[0]]).hexdigest(),
                           image_hashes={p: hashlib.sha256(files[p]).hexdigest() for p in paths},
                           source_files_sha256=source_hashes)
            nvw.validate_episode(episode)
            episodes.append(episode)
            records = [paired_record(episode, paths, view) for view in VIEWS]
            arm_records[arm].extend(records)
            paired[arm] = {"record_ids": [r["id"] for r in records], "images": paths,
                           "endpoint_role_mapping": episode["endpoint_role_mapping"]}
            reviews.append({"root_id": root["root_id"], "condition": arm, "images": paths,
                            "status": "PENDING_RENDER_SEMANTICS_AND_MODEL_PREPROCESSING", **FLAGS})
        pairings.append({"root_id": root["root_id"], "family_id": root["family_id"],
                         "split": root["split"], "conditions": paired,
                         "semantic_sha256": nvw.digest([root["world"], root["states"],
                                                         root["answer"], root["options"]]), **FLAGS})
        roots.add(root["root_id"])
        index += 1

    def emit(name: str, rows: list[dict[str, Any]]) -> None:
        files[name] = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()

    emit("canonical_private.jsonl", episodes)
    emit("pairing.jsonl", pairings)
    emit("review_queue.jsonl", reviews)
    emit("excluded_roots.jsonl", exclusions)
    for arm, rows in arm_records.items():
        emit(arm + ".jsonl", rows)
    manifest = {"version": VERSION, "scope": SCOPE, "source_id": SOURCE_ID,
                "count": count, "seed": seed, "start_index": start_index, "next_index": index,
                "root_count": count, "records_per_arm": count * 3,
                "image_files_per_arm": count * 3, "image_references_per_arm": count * 5,
                "fixed_endpoint_role_mapping": FIXED_MAPPING, "source_files_sha256": source_hashes,
                "mapping_randomization_namespace": MAPPING_VERSION,
                "root_selection_policy": "Post-step-one growth; directed target differs from one-hop undirected, bounded undirected and reversed propagation",
                "source_ancestry": {SOURCE_ID: {"parents": ["independent-nvw-v1"],
                                               "rights_approved": False}},
                "family_splits": {p["family_id"]: p["split"] for p in pairings},
                "render_assets": {"pillow_version": PILLOW_VERSION,
                                  "font_module_sha256": hashlib.sha256(
                                      Path(ImageFont.__file__).read_bytes()).hexdigest(),
                                  "font_ttf_sha256": font_sha256(nvw.FONT),
                                  "small_font_ttf_sha256": font_sha256(nvw.SMALL)},
                "gate": "UNAPPROVED_NOT_FOR_TRAINING_OR_PUBLICATION",
                "limitations": ["Independent synthetic unweighted 6-8 node layered DAG subset only",
                                "Inherited topology has at most two-edge paths; no broad depth generalization",
                                "Rejects three explicit shortcuts, not every possible shortcut or selection bias",
                                "Representation bridge is not non-synthetic transfer or full G7",
                                "Render semantics, actual model preprocessing and source rights review pending",
                                "Matched record/image counts do not establish matched processor tokens or GPU time"],
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
    if output.is_symlink() or any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("symlinks are not allowed")
    manifest = json.loads((output / "manifest.json").read_bytes())
    expected = bundle(manifest["count"], manifest["seed"], manifest["start_index"])
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != set(expected):
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
