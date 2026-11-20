"""Program-authored shortest paths split across topology and edge-weight images."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from ..schema import Mode, Split, to_json
from . import graph
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-graph-pair-v1"
FAMILY = graph.FAMILY
OBSERVATION_INSTRUCTION = graph.OBSERVATION_INSTRUCTION
SOURCE_DEPENDENCIES = (graph,)
_QUESTION = (
    "Image 1 shows a directed graph with edge IDs; image 2 gives each edge ID's weight. "
    "Using both images, what is the shortest path length from A to H? Follow arrowheads "
    "and sum weights. Crossings are not junctions. Give an integer. In Observed facts, "
    "also record dir[<edge ID>] = <source>-><target> and weight[<edge ID>] = <integer>."
)


def _render(world: dict, style: str) -> tuple[tuple[bytes, bytes], tuple[str, ...]]:
    """The topology never prints weights; the table never prints vertex endpoints."""
    graph._validate_world(world)
    if world["kind"] != "shortest" or style not in ("default", "shift"):
        raise ValueError("graph_pair: shortest-path world and known style required")
    shift = style == "shift"
    background, ink = ("#fffaf0", "#17365d") if shift else ("white", "black")
    font = ImageFont.load_default(size=26 if shift else 24)
    topology = Image.new("RGB", (1000, 760), background)
    table = Image.new("RGB", (1000, 760), background)
    topo, weights = ImageDraw.Draw(topology), ImageDraw.Draw(table)
    printed = []

    def label(draw, x, y, value):
        printed.append(value)
        draw.text((x, y), value, fill=ink, font=font, anchor="mm")

    label(topo, 500, 48, "Directed topology")
    label(topo, 500, 98, "Edge IDs refer to image 2; follow arrowheads")
    positions = {name: (110 + (3 - at % 4 if shift else at % 4) * 260, 240 + at // 4 * 300)
                 for name, at in sorted(world["positions"].items())}
    for i, (a, b, _) in enumerate(world["edges"]):
        start, end = positions[a], positions[b]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        head = (end[0] - ux * 32, end[1] - uy * 32)
        topo.line(((start[0] + ux * 32, start[1] + uy * 32), head), fill=ink, width=4)
        topo.polygon((head, (head[0] - ux * 18 - uy * 9, head[1] - uy * 18 + ux * 9),
                      (head[0] - ux * 18 + uy * 9, head[1] - uy * 18 - ux * 9)), fill=ink)
        x, y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        topo.rectangle((x - 29, y - 19, x + 29, y + 19), fill=background)
        label(topo, x, y, f"e{i}")
    for name, (x, y) in positions.items():
        topo.ellipse((x - 28, y - 28, x + 28, y + 28), fill=background, outline=ink, width=4)
        label(topo, x, y, name)

    label(weights, 500, 48, "Edge weights")
    label(weights, 350, 125, "Edge ID")
    label(weights, 650, 125, "Weight")
    rows = list(enumerate(world["edges"]))
    for row, (i, (_, _, weight)) in enumerate(reversed(rows) if shift else rows):
        y = 190 + row * 49
        label(weights, 350, y, f"e{i}")
        label(weights, 650, y, str(weight))
        weights.line((260, y + 24, 740, y + 24), fill=ink, width=1)
    return (png_bytes(topology, dpi=100), png_bytes(table, dpi=100)), tuple(printed)


def _record(world: dict, index: int, seed: int, split: Split, mode: Mode, style: str) -> GeneratedProblem:
    graph._validate_world(world)
    if world["kind"] != "shortest" or graph._shortcut_reason(world):
        raise ValueError("graph_pair: unsupported world or graph shortcut")
    gold, independent = graph._answers(world)
    if gold != independent:
        raise ValueError("graph_pair: independent solver disagreement")
    images, printed = _render(world, style)
    slots = {key: value for i, (a, b, weight) in enumerate(world["edges"])
             for key, value in ((f"dir[e{i}]", f"{a}->{b}"), (f"weight[e{i}]", str(weight)))}
    structure = {k: v for k, v in world.items() if k != "positions"}
    return build(source=GENERATOR_VERSION, family=FAMILY, index=index, split=split, mode=mode,
                 question=_QUESTION, choices=(), answer_type="int", gold=gold,
                 accept={"graph_world": world}, slots=slots,
                 slot_types={key: "relation" if key.startswith("dir[") else "number" for key in slots},
                 required_slots=tuple(slots), oracle_fact_lines=tuple(f"{k} = {v}" for k, v in slots.items()),
                 printed_text=printed, structure=structure, template_id="graph-pair-shortest",
                 image_bytes=images, generator_version=GENERATOR_VERSION, seed=seed, style=style)


def generate(index: int, seed: int, split: Split, mode: Mode, style: str = "default") -> GeneratedProblem:
    if type(index) is not int or index < 0:
        raise ValueError("graph_pair: nonnegative integer index required")
    # The same graph world uses the same split key across single-/two-image sources.
    original = graph.generate(index * 4, seed, split, mode, style)
    return _record(original.verifier.accept["graph_world"], index, seed, split, mode, style)


def self_check(problem: GeneratedProblem) -> None:
    try:
        for record in (problem.actor, problem.verifier, problem.lineage):
            to_json(record)
        lineage = problem.lineage
        index = int(problem.actor.problem_id.split(":")[-2])
        style = "shift" if lineage.split == "style_sealed" else "default"
        world = problem.verifier.accept["graph_world"]
        structure = {k: v for k, v in world.items() if k != "positions"}
        if not matches_split(structure_hash(structure), lineage.split, style):
            raise ValueError("graph_pair: family/split mismatch")
        expected = _record(world, index, lineage.generator_seed, lineage.split, problem.actor.mode, style)
        if problem != expected:
            raise ValueError("graph_pair: source, facts, answer, image order or rendering mismatch")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("graph_pair: self-check failed") from exc
