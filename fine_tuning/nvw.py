"""Private independent synthetic visual worlds. No benchmark or cloud inputs.

Generate: python -m fine_tuning.nvw --output fine_tuning/generated/prototype
The output is quarantined until the separate contamination/rights audit approves it.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

VERSION = "nvw-independent-v1"
FAMILIES = ("signal_graph", "spatial_composition", "finite_state")
RULES = {"IDENTITY": (0, 1), "FLIP": (1, 0), "SET_ZERO": (0, 0), "SET_ONE": (1, 1)}
WIDTH, HEIGHT = 1200, 1000
OUTPUT_SIZE = (960, 800)
FONT = ImageFont.load_default(size=22)
SMALL = ImageFont.load_default(size=18)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seeds(master: int, index: int, family: str) -> dict[str, int]:
    return {key: int(digest([VERSION, master, index, family, key])[:16], 16)
            for key in ("world", "structure", "rule", "support", "glyph", "layout", "style", "options")}


def family_split(family_id: str) -> str:
    bucket = int(family_id[-8:], 16) % 10
    return "train" if bucket < 8 else "development" if bucket == 8 else "sealed_holdout"


def solve_signal(world: dict[str, Any]) -> list[list[int]]:
    states = [sorted(world["initial"])]
    for _ in range(world["steps"]):
        active = set(states[-1])
        states.append(sorted(active | {b for a, b in world["edges"] if a in active}))
    return states


def verify_signal(world: dict[str, Any]) -> list[list[int]]:
    # Independent exhaustive bounded walks, rather than repeated frontier updates.
    walks: list[tuple[int, ...]] = [(source,) for source in world["initial"]]
    result = []
    for depth in range(world["steps"] + 1):
        result.append(sorted({walk[-1] for walk in walks if len(walk) <= depth + 1}))
        walks += [walk + (b,) for walk in list(walks) if len(walk) == depth + 1
                  for a, b in world["edges"] if walk[-1] == a]
    return result


def spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool:
    kind, value = pred
    if kind == "type":
        return obj["type"] == value
    if kind == "color":
        return obj["color"] == value
    if kind == "inside":
        return (obj["x"] < 2) if value == "L" else obj["x"] == 2
    reference = objects[value]
    return obj["x"] < reference["x"] if kind == "left_of" else obj["y"] < reference["y"]


def solve_spatial(world: dict[str, Any]) -> list[list[int]]:
    sets = [{o["id"] for o in world["objects"] if spatial_predicate(o, p, world["objects"])}
            for p in world["filters"]]
    combined = sets[0] & sets[1] if world["operation"] == "INTERSECTION" else sets[0] | sets[1]
    return [sorted(sets[0]), sorted(sets[1]), sorted(combined)]


def verify_spatial(world: dict[str, Any]) -> list[list[int]]:
    # Separately implement predicates and combine integer membership masks.
    masks = []
    for kind, value in world["filters"]:
        mask = 0
        for o in world["objects"]:
            match = {"type": o["type"] == value, "color": o["color"] == value,
                     "inside": (o["x"] < 2 if value == "L" else o["x"] == 2)}
            if kind in ("left_of", "above"):
                ref = next(v for v in world["objects"] if v["id"] == value)
                yes = o["x"] < ref["x"] if kind == "left_of" else o["y"] < ref["y"]
            else:
                yes = match[kind]
            if yes:
                mask |= 1 << o["id"]
        masks.append(mask)
    joined = masks[0] & masks[1] if world["operation"] == "INTERSECTION" else masks[0] | masks[1]
    return [[i for i in range(len(world["objects"])) if value & (1 << i)]
            for value in [*masks, joined]]


def solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]:
    states = [initial]
    for op in sequence:
        states.append(RULES[mapping[op]][states[-1]])
    return states


def verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]:
    value = bool(initial)
    states = [int(value)]
    for op in sequence:
        rule = mapping[op]
        if rule == "FLIP":
            value = not value
        elif rule == "SET_ZERO":
            value = False
        elif rule == "SET_ONE":
            value = True
        elif rule != "IDENTITY":
            raise ValueError("unknown Boolean operator")
        states.append(int(value))
    return states


def fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]:
    candidates = [list(mapping) for mapping in itertools.product(RULES, repeat=2)
                  if all(RULES[mapping[op]][start] == end
                         for op, start, end in world["support"])]
    if not candidates:
        raise ValueError("contradictory support")
    traces = [verify_fsm(m, world["sequence"], world["initial"]) for m in candidates]
    answers = sorted({trace[-1] for trace in traces})
    prefixes = [sorted({trace[i] for trace in traces}) for i in range(len(traces[0]))]
    return {"answer_identifiable": len(answers) == 1,
            "mapping_identifiable": len(candidates) == 1,
            "rule_identifiable": len(candidates) == 1,
            "state_prefix_identifiable": [len(values) == 1 for values in prefixes],
            "evidence_identifiable": True, "candidate_mappings": candidates,
            "possible_answers": answers, "possible_states": prefixes}


def make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]:
    rng = random.Random(streams["structure"])
    rules = random.Random(streams["rule"])
    if family == "signal_graph":
        n = rng.randrange(6, 9)
        edges = [[a, b] for a in range(n) for b in range(a + 1, n)
                 if a // 3 + 1 == b // 3 and rng.random() < .55]
        world = {"n": n, "edges": edges, "initial": [rng.randrange(3)],
                 "steps": rng.randrange(1, 5), "update": "synchronous_monotone"}
        return world, "NVW-K" if index % 2 else "NVW-S"
    if family == "spatial_composition":
        objects = [{"id": i, "x": i % 3, "y": i // 3,
                    "type": rng.randrange(2), "color": rng.randrange(2)} for i in range(6)]
        predicates = [["type", rules.randrange(2)], ["color", rules.randrange(2)],
                      ["inside", rules.choice(["L", "R"])],
                      ["left_of", rules.randrange(6)], ["above", rules.randrange(6)]]
        return {"objects": objects, "filters": rules.sample(predicates, 2),
                "operation": rules.choice(["UNION", "INTERSECTION"])}, (
                    "NVW-K" if index % 2 else "NVW-S")
    if family != "finite_state":
        raise ValueError("unsupported family")
    mapping = [rules.choice(list(RULES)), rules.choice(list(RULES))]
    track = ("NVW-K", "NVW-I", "NVW-S")[index % 3]
    support = [[op, state, RULES[rule][state]] for op, rule in enumerate(mapping)
               for state in (0, 1)]
    if track == "NVW-I" and index % 5 == 1:
        support = [pair for pair in support if pair[1] == 0]
    return {"mapping": mapping, "sequence": [rng.randrange(2) for _ in range(rng.randrange(2, 5))],
            "initial": rng.randrange(2), "support": support}, track


def semantic_family(world: dict[str, Any], family: str) -> str:
    # Root questions, support omissions, and renderings never change split ownership.
    if family == "finite_state":
        key = min([world["mapping"], world["sequence"]],
                  [list(reversed(world["mapping"])), [1-op for op in world["sequence"]]])
    elif family == "signal_graph":
        # Canonicalize unlabeled DAGs: refine structural colors, then enumerate
        # only permutations inside tied color classes. n <= 8 bounds this search.
        n, edges = world["n"], world["edges"]
        colors = [0]*n
        for _ in range(n):
            signatures = [(colors[i], tuple(sorted(colors[a] for a, b in edges if b == i)),
                           tuple(sorted(colors[b] for a, b in edges if a == i))) for i in range(n)]
            ranked = {signature: rank for rank, signature in enumerate(sorted(set(signatures)))}
            updated = [ranked[s] for s in signatures]
            if updated == colors:
                break
            colors = updated
        groups = [[i for i in range(n) if colors[i] == color] for color in sorted(set(colors))]
        if not edges:
            canonical_edges: Any = []
        else:
            candidates = []
            for order in itertools.product(*(itertools.permutations(group) for group in groups)):
                labels = {old: new for new, old in enumerate(itertools.chain.from_iterable(order))}
                candidates.append(sorted([labels[a], labels[b]] for a, b in edges))
            canonical_edges = min(candidates)
        key = [n, canonical_edges]
    else:
        key = world["objects"]
    return family + "-" + digest(key)


def glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int = 24) -> None:
    """Distinct procedural polygons with a four-bit central marker; no external assets."""
    sides = 3 + code % 4
    angle = (code // 4) * math.pi / 8
    points = [(x + int(radius * math.cos(angle + 2 * math.pi * i / sides)),
               y + int(radius * math.sin(angle + 2 * math.pi * i / sides))) for i in range(sides)]
    draw.polygon(points, fill="white", outline="black", width=3)
    for bit in range(4):
        px, py = x - 9 + bit % 2 * 10, y - 9 + bit // 2 * 10
        draw.rectangle((px, py, px + 5, py + 5), fill="black" if code & (1 << bit) else "white",
                       outline="black")


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool = False) -> list[int]:
    font = SMALL if small else FONT
    bbox = [int(v) for v in draw.textbbox(xy, value, font=font)]
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= WIDTH or bbox[3] >= HEIGHT:
        raise ValueError(f"rendered text outside canvas: {value}")
    draw.text(xy, value, fill="black", font=font)
    return bbox


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    ux, uy = dx / distance, dy / distance
    a = (start[0] + ux * 35, start[1] + uy * 35)
    b = (end[0] - ux * 35, end[1] - uy * 35)
    draw.line([a, b], fill="black", width=3)
    draw.polygon([b, (b[0] - ux * 14 - uy * 7, b[1] - uy * 14 + ux * 7),
                  (b[0] - ux * 14 + uy * 7, b[1] - uy * 14 - ux * 7)], fill="black")


def filter_name(pred: list[Any]) -> str:
    kind, value = pred
    return {"type": f"glyph type {value}", "color": f"color {value}",
            "inside": f"inside region {value}", "left_of": f"left of object {value}",
            "above": f"above object {value}"}[kind]


def render(episode: dict[str, Any], canonical: bool = False) -> tuple[Image.Image, dict[str, Any]]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    world, family = episode["world"], episode["family"]
    codes = episode["glyph_codes"]
    jitter = random.Random(episode["seeds"]["layout"])
    objects: list[dict[str, Any]] = []
    visible_relations: list[dict[str, Any]] = []
    legend: list[dict[str, Any]] = []
    text(draw, (30, 20), family.replace("_", " ") + " / " + episode["track"])
    draw.rectangle((20, 65, 1180, 330), outline="black", width=2)
    text(draw, (35, 78), "LEGEND / SUPPORT (same mapping throughout this image)")
    draw.rectangle((20, 350, 1180, 875), outline="black", width=2)
    text(draw, (35, 363), "QUERY")
    if family == "finite_state":
        text(draw, (35, 112), "Allowed unary rules: IDENTITY, FLIP, SET_ZERO, SET_ONE", small=True)
        text(draw, (35, 140), "Rule outputs: IDENTITY (0,1); FLIP (1,0); SET_ZERO (0,0); SET_ONE (1,1)", small=True)
        for op in range(2):
            x = 70 + op * 570
            glyph(draw, x, 218, codes[op])
            pairs = [f"{start}->{end}" for k, start, end in world["support"] if k == op]
            name = world["mapping"][op] if canonical and episode["quality"]["mapping_identifiable"] else f"operator {op}"
            box = text(draw, (x + 45, 193), name)
            text(draw, (x + 45, 231), "Observed: " + "; ".join(pairs))
            legend.append({"operator": op, "glyph_code": codes[op], "bbox": box,
                           "support": [p for p in world["support"] if p[0] == op]})
        text(draw, (45, 426), f"Initial bit: {world['initial']}. Apply operators left to right.")
        centers = [(220 + i * 250, 605) for i in range(len(world["sequence"]))]
        for i, (op, center) in enumerate(zip(world["sequence"], centers)):
            if i:
                arrow(draw, centers[i - 1], center)
            if canonical and episode["quality"]["mapping_identifiable"]:
                text(draw, (center[0] - 50, center[1] - 10), world["mapping"][op], small=True)
            else:
                glyph(draw, *center, codes[op], radius=34)
            objects.append({"id": i, "operator": op, "glyph_code": codes[op],
                            "bbox": [center[0]-40, center[1]-40, center[0]+40, center[1]+40]})
            if i:
                visible_relations.append({"type": "PRECEDES", "from": i-1, "to": i})
        text(draw, (45, 738), "If multiple allowed rules imply different final bits, choose INSUFFICIENT.")
    elif family == "signal_graph":
        glyph(draw, 70, 165, codes[0])
        draw.ellipse((90, 130, 108, 148), fill="black", outline="black", width=2)
        text(draw, (115, 150), "= signal node. Black upper-right badge = initially active.", small=True)
        text(draw, (115, 178), "White upper-right badge = initially inactive.", small=True)
        text(draw, (45, 212), "Arrow = directed link. Crossing lines have no junction.")
        text(draw, (45, 249), "At each step, active nodes stay active and activate all outgoing neighbors.", small=True)
        text(draw, (45, 289), f"Use synchronous updates for exactly {world['steps']} steps.")
        node_centers = {i: (160 + i // 3 * 420 + jitter.randrange(-12, 13),
                       460 + i % 3 * 145 + jitter.randrange(-8, 9)) for i in range(world["n"])}
        for a, b in world["edges"]:
            arrow(draw, node_centers[a], node_centers[b])
            visible_relations.append({"type": "DIRECTED_TO", "from": a, "to": b,
                                      "segment": [list(node_centers[a]), list(node_centers[b])]})
        for i, (x, y) in node_centers.items():
            if canonical:
                draw.ellipse((x-28, y-28, x+28, y+28), fill="white", outline="black", width=3)
            else:
                glyph(draw, x, y, codes[0], radius=32)
            draw.ellipse((x+22, y-36, x+40, y-18),
                         fill="black" if i in world["initial"] else "white", outline="black", width=2)
            text(draw, (x-10, y+40), str(i))
            objects.append({"id": i, "bbox": [x-35, y-35, x+35, y+35],
                            "initially_active": i in world["initial"], "glyph_code": codes[0]})
        legend = [{"glyph_code": codes[0], "meaning": "signal node"}]
    else:
        text(draw, (45, 114), "Coordinates: left/right and above refer to the drawn token centers.", small=True)
        for kind in range(2):
            glyph(draw, 70 + kind * 310, 175, codes[kind])
            text(draw, (110 + kind * 310, 163), f"= glyph type {kind}")
        for color, name in enumerate(("BLUE", "ORANGE")):
            draw.rectangle((710 + color*210, 153, 740 + color*210, 183),
                           fill=("#1673c8", "#c56a00")[color], outline="black")
            text(draw, (752 + color*210, 160), f"= color {color}", small=True)
        text(draw, (45, 220), "Regions L and R are disjoint boxes. UNION means either; INTERSECTION means both.", small=True)
        text(draw, (45, 269), "Count tokens in the resulting set; object labels are unique instance IDs.", small=True)
        text(draw, (45, 410), f"Set A: {filter_name(world['filters'][0])}. Set B: {filter_name(world['filters'][1])}.")
        text(draw, (45, 447), f"Operation: {world['operation']}.")
        draw.rectangle((100, 500, 735, 830), outline="black", width=3)
        draw.rectangle((760, 500, 1100, 830), outline="black", width=3)
        text(draw, (110, 505), "L")
        text(draw, (770, 505), "R")
        column_x = [base + jitter.randrange(-10, 11) for base in (250, 570, 930)]
        row_y = [base + jitter.randrange(-6, 7) for base in (590, 740)]
        for obj in world["objects"]:
            x, y = column_x[obj["x"]], row_y[obj["y"]]
            draw.rectangle((x-49, y-45, x+49, y+45), fill=("#1673c8", "#c56a00")[obj["color"]])
            if canonical:
                text(draw, (x-12, y-12), f"T{obj['type']}")
            else:
                glyph(draw, x, y, codes[obj["type"]], radius=33)
            text(draw, (x-7, y+48), str(obj["id"]))
            objects.append({"id": obj["id"], "type": obj["type"], "color": obj["color"],
                            "grid_position": [obj["x"], obj["y"]], "center": [x, y],
                            "bbox": [x-49, y-45, x+49, y+45],
                            "glyph_code": codes[obj["type"]]})
        legend = [{"type": i, "glyph_code": codes[i]} for i in range(2)]
    text(draw, (40, 900), episode["question"])
    text(draw, (40, 948), "    ".join(f"{chr(65+i)}. {answer}" for i, answer in enumerate(episode["options"])))
    observation = {"image_size": [WIDTH, HEIGHT], "coordinate_system": "pixel_xy_top_left",
                   "objects": objects, "relations": visible_relations, "legend": legend}
    return image, observation


def build_episode(family: str, index: int, master_seed: int = 3407) -> dict[str, Any]:
    streams = seeds(master_seed, index, family)
    world, track = make_world(family, streams, index)
    states: list[Any]
    independent: list[Any]
    if family == "finite_state":
        states = solve_fsm(world["mapping"], world["sequence"], world["initial"])
        independent = verify_fsm(world["mapping"], world["sequence"], world["initial"])
        quality = fsm_identifiability(world)
        answer: Any = quality["possible_answers"][0] if quality["answer_identifiable"] else "INSUFFICIENT"
        options = [0, 1, "INSUFFICIENT"]
        question = "What is the final bit? Choose the uniquely determined answer."
    else:
        solver, verifier = ((solve_signal, verify_signal) if family == "signal_graph"
                            else (solve_spatial, verify_spatial))
        states, independent = solver(world), verifier(world)
        answer = len(states[-1])
        quality = {"answer_identifiable": True, "mapping_identifiable": True,
                   "rule_identifiable": True, "state_prefix_identifiable": [True]*len(states),
                   "evidence_identifiable": True, "candidate_mappings": [],
                   "possible_answers": [answer], "possible_states": [[s] for s in states]}
        limit = world["n"] if family == "signal_graph" else 6
        distractors = random.Random(streams["options"]).sample([v for v in range(limit+1) if v != answer], 3)
        options = [answer, *distractors]
        question = ("How many nodes are active after the specified steps?" if family == "signal_graph"
                    else "How many tokens are in the resulting set?")
    if states != independent:
        raise ValueError("independent solver disagreement")
    random.Random(streams["options"]).shuffle(options)
    family_id = semantic_family(world, family)
    quality.update({"canonical": "PASS", "solver": "PASS", "render": "PENDING",
                    "model_input": "UNKNOWN", "label_status": "DERIVED_VERIFIED"})
    return {"id": f"{VERSION}-{family}-{master_seed}-{index}", "family": family,
            "family_id": family_id, "root_id": digest([family_id, world]), "split": family_split(family_id),
            "track": track, "world": world, "seeds": streams,
            "glyph_codes": random.Random(streams["glyph"]).sample(range(16), 2),
            "question": question, "options": options, "answer": answer,
            "answer_letter": chr(65+options.index(answer)), "states": states, "quality": quality,
            "source_kind": "INDEPENDENT_SYNTHETIC", "parents": [], "all_parents_known": True,
            "source_id": "nvw-independent-procedural-v1", "benchmark_derived": False,
            "lineage": ["independent-nvw-v1"], "independent_sources": ["nvw-independent-procedural-v1"],
            "license": "project_authored_private_pending_rights_review",
            "rights": {"code": "project_authored", "images": "procedural_no_external_assets",
                       "font": "Pillow_bundled_default_Aileron", "review": "PENDING"},
            "contamination": {"benchmark_derived": False, "audit_decision": "NOT_AUDITED"},
            "training_allowed": False, "public_upload_allowed": False,
            "curriculum": "C2" if track == "NVW-I" else "C3",
            "generator_version": VERSION, "clean_spec_sha256": hashlib.sha256(
                Path(__file__).with_name("clean_spec.json").read_bytes()).hexdigest()}


def input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]:
    # Explicit whitelist: never serialize the private world, latent mapping, or gold.
    instructions = {"qa": episode["question"] + " Give only the option letter.",
                    "structure": "Describe visible objects, legend/support, and directed links or token attributes as JSON.",
                    "grounded": episode["question"] + " Report only supported intermediate states and the option letter."}
    if view not in instructions:
        raise ValueError("unknown view")
    return {"image": [image_path], "instruction": "<image>\n" + instructions[view]}


def view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]:
    record = input_record(episode, image_path, view)
    if view == "qa":
        target = episode["answer_letter"]
    elif view == "structure":
        target = json.dumps(episode["observed_structure"], sort_keys=True)
    else:
        supported_states = [{"step": i, "state": values[0]} for i, values in enumerate(
            episode["quality"]["possible_states"]) if episode["quality"]["state_prefix_identifiable"][i]]
        payload = {"states": supported_states, "possible_answers": episode["quality"]["possible_answers"],
                   "answer_identifiable": episode["quality"]["answer_identifiable"],
                   "answer": episode["answer"], "option": episode["answer_letter"]}
        if episode["family"] == "finite_state" and episode["quality"]["rule_identifiable"]:
            payload["rules"] = episode["world"]["mapping"]
        target = json.dumps(payload, sort_keys=True)
    return {"id": episode["id"] + "-" + view, "root_id": episode["root_id"],
            "family_id": episode["family_id"], "split": episode["split"], "view": view,
            "source_kind": episode["source_kind"], "parents": [], "training_allowed": False,
            "source_id": episode["source_id"], "benchmark_derived": False,
            "lineage": episode["lineage"], "quality_status": "DERIVED_VERIFIED",
            "quality_scope": "canonical_and_solver_only; render_semantics_and_model_input_pending",
            "answer_identifiable": episode["quality"]["answer_identifiable"],
            "supervised_target_identifiable": True,
            "state_prefix_identifiable": episode["quality"]["state_prefix_identifiable"],
            "independent_sources": episode["independent_sources"], "license": episode["license"],
            "user_text": record["instruction"], "target_text": target,
            "image_paths": record["image"],
            "image": record["image"], "conversations": [
                {"from": "human", "value": record["instruction"]},
                {"from": "gpt", "value": target}],
            "epoch_view_schedule": {"cycle": 3, "epoch_modulo": ("qa", "structure", "grounded").index(view)}}


def validate_episode(episode: dict[str, Any]) -> None:
    if episode["family_id"] != semantic_family(episode["world"], episode["family"]):
        raise ValueError("family fingerprint mismatch")
    if episode["split"] != family_split(episode["family_id"]):
        raise ValueError("family split mismatch")
    if len(set(episode["options"])) != len(episode["options"]):
        raise ValueError("duplicate choices")
    if episode["options"][ord(episode["answer_letter"])-65] != episode["answer"]:
        raise ValueError("answer option mismatch")
    if episode["parents"] or episode["source_kind"] != "INDEPENDENT_SYNTHETIC":
        raise ValueError("unexpected source ancestry")
    family, world = episode["family"], episode["world"]
    states: list[Any]
    if family == "finite_state":
        expected_quality = fsm_identifiability(world)
        if world["mapping"] not in expected_quality["candidate_mappings"]:
            raise ValueError("selected mapping contradicts observed support")
        for key, expected in expected_quality.items():
            if episode["quality"][key] != expected:
                raise ValueError("identifiability metadata disagrees with exhaustive enumeration")
        expected_answer = (expected_quality["possible_answers"][0]
                           if expected_quality["answer_identifiable"] else "INSUFFICIENT")
        states = verify_fsm(world["mapping"], world["sequence"], world["initial"])
    elif family == "signal_graph":
        if any(not 0 <= a < b < world["n"] for a, b in world["edges"]):
            raise ValueError("invalid acyclic edge")
        states = verify_signal(world)
        expected_answer = len(states[-1])
    else:
        states = verify_spatial(world)
        expected_answer = len(states[-1])
    if episode["answer"] != expected_answer:
        raise ValueError("answer disagrees with independent solver or sufficiency check")
    if states != episode["states"]:
        raise ValueError("stored states failed independent solver")
    observation = episode.get("observed_structure", {})
    width, height = observation.get("image_size", [WIDTH, HEIGHT])
    sx, sy = width/WIDTH, height/HEIGHT
    for obj in observation.get("objects", []):
        x0, y0, x1, y1 = obj["bbox"]
        if not 20*sx <= x0 < x1 < width-20*sx or not 350*sy < y0 < y1 < 875*sy:
            raise ValueError("object outside query panel")


def generate_dataset(output: Path, per_family: int = 100, smoke_roots: int = 120,
                     seed: int = 3407, bridge_every: int = 5) -> dict[str, Any]:
    if per_family < 1 or smoke_roots < 0 or bridge_every < 1:
        raise ValueError("invalid generation count")
    if output.exists():
        raise FileExistsError("refuse to overwrite an existing generated dataset")
    output.mkdir(parents=True)
    source_bytes = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    (output / "generator_source.py").write_bytes(source_bytes)
    spec_bytes = Path(__file__).with_name("clean_spec.json").read_bytes()
    (output / "clean_spec.json").write_bytes(spec_bytes)
    images = output / "images"
    images.mkdir()
    episodes = []
    used_roots: set[str] = set()
    family_groups: dict[str, str] = {}
    for purpose, counts in (("prototype", [per_family]*3),
                            ("smoke", [smoke_roots//3 + (i < smoke_roots % 3) for i in range(3)])):
        for family, count in zip(FAMILIES, counts):
            accepted = 0
            index = 0 if purpose == "prototype" else per_family * 100 + 10000
            attempts = 0
            while accepted < count:
                attempts += 1
                if attempts > count*1000:
                    raise RuntimeError("generation exhausted unique family/root space")
                episode = build_episode(family, index, seed)
                index += 1
                if episode["root_id"] in used_roots or (purpose == "smoke" and episode["split"] != "train"):
                    continue
                episode["purpose"] = purpose
                image, observation = render(episode)
                episode["observed_structure"] = observation
                validate_episode(episode)
                image = image.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
                sx, sy = OUTPUT_SIZE[0]/WIDTH, OUTPUT_SIZE[1]/HEIGHT
                observation["image_size"] = list(OUTPUT_SIZE)
                for obj in observation["objects"]:
                    obj["bbox"] = [round(v * (sx if i % 2 == 0 else sy))
                                   for i, v in enumerate(obj["bbox"])]
                    if "center" in obj:
                        obj["center"] = [round(obj["center"][0]*sx), round(obj["center"][1]*sy)]
                for relation in observation["relations"]:
                    if "segment" in relation:
                        relation["segment"] = [[round(x*sx), round(y*sy)] for x, y in relation["segment"]]
                for item in observation["legend"]:
                    if "bbox" in item:
                        item["bbox"] = [round(v * (sx if i % 2 == 0 else sy))
                                        for i, v in enumerate(item["bbox"])]
                path = images / (episode["id"] + ".png")
                image.save(path)
                episode["image"] = path.relative_to(output).as_posix()
                episode["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                episode["decoded_pixels_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
                episode["quality"]["render"] = "GEOMETRY_PASS_SEMANTIC_REVIEW_PENDING"
                episode["bridge"] = None
                if accepted % bridge_every == 0 and episode["quality"]["mapping_identifiable"]:
                    bridge, _ = render(episode, canonical=True)
                    bridge = bridge.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
                    bridge_path = images / (episode["id"] + "-canonical.png")
                    bridge.save(bridge_path)
                    episode["bridge"] = {"image": bridge_path.relative_to(output).as_posix(),
                                         "image_sha256": hashlib.sha256(bridge_path.read_bytes()).hexdigest(),
                                         "semantic_root_id": episode["root_id"],
                                         "claim": "same synthetic state; no real-domain transfer claim"}
                used_roots.add(episode["root_id"])
                previous = family_groups.setdefault(episode["family_id"], episode["split"])
                if previous != episode["split"]:
                    raise ValueError("family crossed split")
                episodes.append(episode)
                accepted += 1
    files: dict[str, dict[str, Any]] = {}
    def emit(name: str, records: list[dict[str, Any]]) -> None:
        path = output / name
        path.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in records))
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "bytes": path.stat().st_size, "records": len(records)}
    emit("canonical_private.jsonl", episodes)
    for purpose in ("prototype", "smoke"):
        for split in ("train", "development", "sealed_holdout"):
            selected = [e for e in episodes if e["purpose"] == purpose and e["split"] == split]
            records = [view_record(e, e["image"], view) for e in selected
                       for view in ("qa", "structure", "grounded")]
            if records:
                emit(f"{purpose}_{split}.jsonl", records)
    manifest = {"schema_version": VERSION, "generator_sha256": source_hash,
                "generator_source_file": "generator_source.py",
                "clean_spec_sha256": episodes[0]["clean_spec_sha256"], "seed": seed,
                "render_assets": {"pillow_version": PILLOW_VERSION,
                                  "font_module_sha256": hashlib.sha256(Path(ImageFont.__file__).read_bytes()).hexdigest(),
                                  "font": "Pillow bundled Aileron regular default", "external_asset_downloads": 0},
                "episode_count": len(episodes), "root_count": len(used_roots),
                "family_count": len(family_groups), "prototype_roots": per_family*3,
                "smoke_roots": smoke_roots, "files": files,
                "family_splits": family_groups, "source_kind": "INDEPENDENT_SYNTHETIC",
                "training_allowed": False, "public_upload_allowed": False,
                "gate": "PENDING_EXTERNAL_CONTAMINATION_RIGHTS_AND_MODEL_INPUT_AUDIT",
                "quality": {"solver": "PASS", "identifiability": "ENUMERATED_OR_FULL_LEGEND",
                            "render": "GEOMETRY_PASS_SEMANTIC_REVIEW_PENDING", "model_input": "UNKNOWN"},
                "counts": {purpose: {split: sum(e["purpose"] == purpose and e["split"] == split
                                                for e in episodes)
                                     for split in ("train", "development", "sealed_holdout")}
                           for purpose in ("prototype", "smoke")}}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")
    return manifest


def verify_dataset(output: Path) -> dict[str, Any]:
    """Verify closed local artifacts and independently recompute every canonical result."""
    manifest = json.loads((output/"manifest.json").read_text())
    if "generator_source_file" in manifest:
        source = output/manifest["generator_source_file"]
        if source.parent != output or hashlib.sha256(source.read_bytes()).hexdigest() != manifest["generator_sha256"]:
            raise ValueError("generator source archive hash mismatch")
        if hashlib.sha256((output/"clean_spec.json").read_bytes()).hexdigest() != manifest["clean_spec_sha256"]:
            raise ValueError("clean specification hash mismatch")
    episodes = [json.loads(line) for line in (output/"canonical_private.jsonl").read_text().splitlines()]
    if len(episodes) != manifest["episode_count"] or len({e["root_id"] for e in episodes}) != manifest["root_count"]:
        raise ValueError("episode or root counts disagree")
    roots = {e["root_id"]: e for e in episodes}
    checked_images, checked_records = 0, 0
    for name, info in manifest["files"].items():
        path = output/name
        if path.parent != output or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("manifest path or hash mismatch")
        if path.stat().st_size != info["bytes"]:
            raise ValueError("manifest byte count mismatch")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != info["records"]:
            raise ValueError("manifest record count mismatch")
        if name != "canonical_private.jsonl":
            for record in records:
                episode = roots[record["root_id"]]
                if record != view_record(episode, episode["image"], record["view"]):
                    raise ValueError("stored view differs from whitelisted serializer")
                checked_records += 1
    groups: dict[str, str] = {}
    for episode in episodes:
        validate_episode(episode)
        if groups.setdefault(episode["family_id"], episode["split"]) != episode["split"]:
            raise ValueError("family split leakage")
        if episode["purpose"] == "smoke" and episode["split"] != "train":
            raise ValueError("smoke overlaps held-out families")
        if episode["training_allowed"] or episode["contamination"]["audit_decision"] != "NOT_AUDITED":
            raise ValueError("raw generation unexpectedly approved")
        for asset in [episode, *([episode["bridge"]] if episode["bridge"] else [])]:
            path = output / asset["image"]
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("image path escapes dataset")
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset["image_sha256"]:
                raise ValueError("image hash mismatch")
            with Image.open(path) as image:
                image.load()
                if image.size != OUTPUT_SIZE or image.mode != "RGB":
                    raise ValueError("render dimensions or format mismatch")
                if asset is episode and hashlib.sha256(image.tobytes()).hexdigest() != episode["decoded_pixels_sha256"]:
                    raise ValueError("decoded pixel hash mismatch")
            checked_images += 1
    if groups != manifest["family_splits"]:
        raise ValueError("family registry mismatch")
    return {"status": "LOCAL_VERIFIED", "episodes": len(episodes), "images": checked_images,
            "records": checked_records, "independent_solvers": "PASS", "family_splits": "PASS",
            "input_whitelist": "PASS", "training_allowed": False,
            "render_semantic_certification": "PENDING", "model_preprocessing": "UNKNOWN"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--smoke-roots", type=int, default=120)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_dataset(args.output), sort_keys=True))
        return
    manifest = generate_dataset(args.output, args.per_family, args.smoke_roots, args.seed)
    print(json.dumps({key: manifest[key] for key in ("episode_count", "root_count", "counts", "gate")}))


if __name__ == "__main__":
    main()
