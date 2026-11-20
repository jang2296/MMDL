"""Exact program-authored DC bridge and 2–3-stage ladders; no AC/transient scope."""
from __future__ import annotations

import hashlib
import math
import random
from fractions import Fraction

from PIL import Image, ImageDraw, ImageFont

from fine_tuning import circuit_bridge

from ..schema import Mode, Split, to_json
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-circuit-v1"
FAMILY = "circuit"
OBSERVATION_INSTRUCTION = (
    "List each component as one line: comp[<name>] = <value with unit or switch state>, "
    "each connection as wire[<name>] = <node>-<node>, and each source polarity as "
    "dir[<name>] = <positive node>-><negative node>.\n"
)


def _validate(world: dict) -> None:
    if world["kind"] == "bridge":
        if set(world) != {"kind", "bridge"}:
            raise ValueError("circuit: invalid bridge structure")
        circuit_bridge.validate_world(world["bridge"])
    elif world["kind"] == "ladder":
        if (set(world) != {"kind", "sources", "series", "shunts"}
                or type(world["sources"]) is not list or not 1 <= len(world["sources"]) <= 2
                or type(world["series"]) is not list or type(world["shunts"]) is not list
                or not 2 <= len(world["series"]) <= 3 or len(world["series"]) != len(world["shunts"])
                or any(type(v) is not int or not 1 <= v <= 48
                       for name in ("sources", "series", "shunts") for v in world[name])):
            raise ValueError("circuit: invalid ladder component/value contract")
    else:
        raise ValueError("circuit: unsupported topology")


def _components(world: dict) -> list[dict]:
    if world["kind"] == "bridge":
        mapping = {"C0": "R1", "C1": "R2", "C2": "R3", "C3": "E1", "C4": "S1"}
        result = []
        for comp in circuit_bridge.canonical_structure(world["bridge"])["components"]:
            result.append({"name": mapping[comp["id"]], "kind": comp["type"],
                           "a": comp["pins"][0], "b": comp["pins"][1],
                           "value": str(comp.get("state", comp.get("value"))),
                           "unit": comp.get("unit", "")})
        return result
    ns = len(world["sources"])
    result = [{"name": f"E{i + 1}", "kind": "DC_SOURCE", "a": f"N{i + 1}", "b": f"N{i}",
               "value": str(v), "unit": "V"} for i, v in enumerate(world["sources"])]
    for i, (series, shunt) in enumerate(zip(world["series"], world["shunts"], strict=True)):
        result.extend([
            {"name": f"R{2 * i + 1}", "kind": "RESISTOR", "a": f"N{ns + i}",
             "b": f"N{ns + i + 1}", "value": str(series), "unit": "ohm"},
            {"name": f"R{2 * i + 2}", "kind": "RESISTOR", "a": f"N{ns + i + 1}",
             "b": "N0", "value": str(shunt), "unit": "ohm"},
        ])
    return result


def _ladder_nodal(world: dict) -> dict:
    ns, n = len(world["sources"]), len(world["series"])
    nodes = {f"N{i}": Fraction(sum(world["sources"][:i])) for i in range(ns + 1)}
    unknown = [f"N{ns + i + 1}" for i in range(n)]
    matrix, rhs = [[Fraction(0) for _ in unknown] for _ in unknown], [Fraction(0) for _ in unknown]
    for comp in _components(world):
        if comp["kind"] != "RESISTOR":
            continue
        a, b, conductance = comp["a"], comp["b"], Fraction(1, int(comp["value"]))
        for node, other in ((a, b), (b, a)):
            if node not in unknown:
                continue
            i = unknown.index(node)
            matrix[i][i] += conductance
            if other in nodes:
                rhs[i] += conductance * nodes[other]
            else:
                matrix[i][unknown.index(other)] -= conductance
    for i in range(n):
        pivot = matrix[i][i]
        if not pivot:
            raise ValueError("circuit: singular ladder")
        matrix[i] = [v / pivot for v in matrix[i]]
        rhs[i] /= pivot
        for j in range(n):
            if i != j:
                factor = matrix[j][i]
                matrix[j] = [a - factor * b for a, b in zip(matrix[j], matrix[i], strict=True)]
                rhs[j] -= factor * rhs[i]
    nodes.update(zip(unknown, rhs, strict=True))
    currents = {comp["name"]: (nodes[comp["a"]] - nodes[comp["b"]]) / int(comp["value"])
                for comp in _components(world) if comp["kind"] == "RESISTOR"}
    current = currents["R1"]
    return {"node_voltages_V": nodes, "resistor_currents_A": currents,
            "source_current_A": current, "equivalent_R_ohm": Fraction(sum(world["sources"])) / current}


def _ladder_reduction(world: dict) -> dict:
    series, shunts = list(map(Fraction, world["series"])), list(map(Fraction, world["shunts"]))
    downstream = shunts[-1]
    for i in range(len(series) - 2, -1, -1):
        branch = series[i + 1] + downstream
        downstream = shunts[i] * branch / (shunts[i] + branch)
    resistance = series[0] + downstream
    current = Fraction(sum(world["sources"])) / resistance
    ns = len(world["sources"])
    nodes = {f"N{i}": Fraction(sum(world["sources"][:i])) for i in range(ns + 1)}
    voltage, flow, currents = nodes[f"N{ns}"], current, {}
    for i, (r, shunt) in enumerate(zip(series, shunts, strict=True)):
        voltage -= flow * r
        nodes[f"N{ns + i + 1}"] = voltage
        currents[f"R{2 * i + 1}"] = flow
        currents[f"R{2 * i + 2}"] = voltage / shunt
        flow -= voltage / shunt
    if flow:
        raise ValueError("circuit: series-parallel KCL disagreement")
    return {"node_voltages_V": nodes, "resistor_currents_A": currents,
            "source_current_A": current, "equivalent_R_ohm": resistance}


def _solutions(world: dict) -> tuple[dict, dict]:
    _validate(world)
    if world["kind"] == "ladder":
        return _ladder_nodal(world), _ladder_reduction(world)
    first, second = circuit_bridge.solve_nodal(world["bridge"]), circuit_bridge.verify_series_parallel(world["bridge"])
    def convert(result):
        current = Fraction(result["source_current_A"])
        return {"node_voltages_V": {k: Fraction(v) for k, v in result["node_voltages_V"].items()},
                "resistor_currents_A": {f"R{int(k[1:]) + 1}": Fraction(v)
                                         for k, v in result["resistor_currents_A"].items()},
                "source_current_A": current,
                "equivalent_R_ohm": Fraction(world["bridge"]["voltage_V"]) / current}
    return convert(first), convert(second)


def _query(world: dict, index: int) -> tuple[str, str]:
    kind = ("current", "voltage", "resistance")[(index // 3) % 3]
    if kind == "current":
        return kind, "R2" if world["kind"] == "bridge" else "R4"
    if kind == "voltage":
        return kind, "N2" if world["kind"] == "bridge" else f"N{len(world['sources']) + len(world['series'])}"
    return kind, "input"


def _answer(solution: dict, query: tuple[str, str]) -> tuple[str, str]:
    kind, target = query
    if kind == "current":
        return f"{solution['resistor_currents_A'][target]} A", "A"
    if kind == "voltage":
        return f"{solution['node_voltages_V'][target]} V", "V"
    return f"{solution['equivalent_R_ohm']} ohm", "ohm"


def _question(query: tuple[str, str]) -> str:
    kind, target = query
    base = ("Ideal DC circuit; N0 is the 0 V reference. Wires and closed switches have zero "
            "resistance; open switches are disconnected. Source + and - signs specify polarity. ")
    if kind == "current":
        return base + f"What is the downward current through {target}? Give an exact integer or reduced fraction with unit A."
    if kind == "voltage":
        return base + f"What is the voltage of {target} relative to N0? Give an exact integer or reduced fraction with unit V."
    return (base + "What equivalent resistance does the resistive load present to the source(s)? "
            "Give an exact integer or reduced fraction with unit ohm.")


def _facts(world: dict) -> tuple[dict[str, str], dict[str, str]]:
    facts, types = {}, {}
    for comp in _components(world):
        name = comp["name"]
        facts[f"comp[{name}]"] = comp["value"] + (" " + comp["unit"] if comp["unit"] else "")
        types[f"comp[{name}]"] = "text" if comp["kind"] == "SWITCH" else "number"
        facts[f"wire[{name}]"] = f"{comp['a']}-{comp['b']}"
        types[f"wire[{name}]"] = "relation"
        if comp["kind"] == "DC_SOURCE":
            facts[f"dir[{name}]"] = f"{comp['a']}->{comp['b']}"
            types[f"dir[{name}]"] = "relation"
    return dict(sorted(facts.items())), dict(sorted(types.items()))


def _render(world: dict, style: str) -> tuple[bytes, tuple[str, ...]]:
    shift = style == "shift"
    image = Image.new("RGB", (1200, 860), "#fffaf0" if shift else "white")
    draw = ImageDraw.Draw(image)
    ink, width = ("#17365d", 4) if shift else ("black", 3)
    font = ImageFont.load_default_imagefont() if shift else ImageFont.load_default(size=22)
    printed = []
    def label(x, y, value):
        printed.append(value)
        if shift:
            box = font.getbbox(value)
            patch = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]))
            ImageDraw.Draw(patch).text((-box[0], -box[1]), value, font=font, fill=ink)
            patch = patch.resize((patch.width * 2, patch.height * 2), Image.Resampling.NEAREST)
            image.paste(patch, (round(x - patch.width / 2), round(y - patch.height / 2)), patch)
        else:
            draw.text((x, y), value, font=font, fill=ink, anchor="mm")
    label(600, 40, "Ideal DC resistive circuit (V, ohm)")
    label(600, 82, "Dots are connected junctions; N0 is the common reference; source polarity is explicit.")
    top, ground = (230, 700) if shift else (250, 680)
    positions: dict[str, tuple[float, float]]
    if world["kind"] == "bridge":
        positions = {"N0": (570, ground), "N1": (140, top), "N2": (570, top), "N3": (1030, top)}
    else:
        ns, n = len(world["sources"]), len(world["series"])
        positions = {f"N{i}": (140, ground - (ground - top) * i / ns) for i in range(ns + 1)}
        positions.update({f"N{ns + i}": (140 + 880 * i / n, top) for i in range(1, n + 1)})
    if shift:
        positions = {name: (1200 - x, y) for name, (x, y) in positions.items()}
    xs = [x for x, _ in positions.values()]
    draw.line((min(xs), ground, max(xs), ground), fill=ink, width=width)
    for comp in _components(world):
        a, b = positions[comp["a"]], positions[comp["b"]]
        if comp["b"] == "N0":
            b = (a[0], ground)
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        length = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        p, q = (cx - ux * 32, cy - uy * 32), (cx + ux * 32, cy + uy * 32)
        draw.line((a, p), fill=ink, width=width)
        draw.line((q, b), fill=ink, width=width)
        if comp["kind"] == "RESISTOR":
            rectangle = (cx - 32, cy - 13, cx + 32, cy + 13) if uy == 0 else (cx - 13, cy - 32, cx + 13, cy + 32)
            draw.rectangle(rectangle, outline=ink, width=width)
        elif comp["kind"] == "DC_SOURCE":
            draw.ellipse((cx - 32, cy - 32, cx + 32, cy + 32), outline=ink, width=width)
            label(cx, cy - 15, "+")
            label(cx, cy + 15, "-")
        else:
            end = q if comp["value"] == "CLOSED" else (q[0], q[1] - 34)
            draw.line((p, end), fill=ink, width=width)
            for x, y in (p, q):
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=ink, width=2)
        value = comp["value"] + (" " + comp["unit"] if comp["unit"] else "")
        label(cx if uy == 0 else cx + (-90 if shift else 90), cy + 45 if uy == 0 else cy,
              f"{comp['name']} {value}")
        if comp["b"] == "N0":
            draw.ellipse((b[0] - 4, ground - 4, b[0] + 4, ground + 4), fill=ink)
    for name, (x, y) in sorted(positions.items()):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=ink)
        label(x + (-25 if shift else 25), y - 27, name)
    return png_bytes(image, dpi=100), tuple(printed)


def generate(index: int, seed: int, split: Split, mode: Mode, style: str = "default") -> GeneratedProblem:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("circuit: nonnegative integer index and seed required")
    matches_split("0" * 16, split, style)
    if mode not in ("plain", "obs"):
        raise ValueError("circuit: invalid mode")
    rng = random.Random(seed * 1_000_003 + index)
    for _ in range(1000):
        world: dict
        if index % 3 == 0:
            world = {"kind": "bridge", "bridge": {
                "voltage_V": rng.randint(2, 30), "resistance_ohm": [rng.randint(2, 24) for _ in range(3)],
                "switch_closed": bool(rng.randrange(2))}}
        else:
            n, ns = 2 + (index % 3 == 2), 1 + (index // 9) % 2
            world = {"kind": "ladder", "sources": [rng.randint(2, 18) for _ in range(ns)],
                     "series": [rng.randint(2, 24) for _ in range(n)],
                     "shunts": [rng.randint(2, 24) for _ in range(n)]}
        if not matches_split(structure_hash(world), split, style):
            continue
        first, second = _solutions(world)
        if first != second:
            raise ValueError("circuit: independent solver disagreement")
        query = _query(world, index)
        gold, unit = _answer(first, query)
        slots, types = _facts(world)
        raw, printed = _render(world, style)
        accept = {"unit_aliases": [unit], "require_unit": True, "circuit_world": world,
                  "circuit_query": list(query)}
        return build(source=GENERATOR_VERSION, family=FAMILY, index=index, seed=seed, split=split,
                     mode=mode, style=style, question=_question(query), choices=(), answer_type="unit_number",
                     gold=gold, accept=accept, slots=slots, slot_types=types, required_slots=tuple(slots),
                     oracle_fact_lines=tuple(f"{k} = {v}" for k, v in slots.items()), printed_text=printed,
                     structure=world, template_id=f"circuit-{query[0]}", image_bytes=(raw,),
                     generator_version=GENERATOR_VERSION)
    raise RuntimeError("circuit: exhausted 1000 candidate structures")


def self_check(p: GeneratedProblem) -> None:
    try:
        for record in (p.actor, p.verifier, p.lineage):
            to_json(record)
        world, query = p.verifier.accept["circuit_world"], tuple(p.verifier.accept["circuit_query"])
        if (set(p.verifier.accept) != {"unit_aliases", "require_unit", "circuit_world", "circuit_query"}
                or len(query) != 2 or query not in {_query(world, i) for i in (0, 3, 6)}):
            raise ValueError("invalid query")
        first, second = _solutions(world)
        if first != second:
            raise ValueError("independent solver disagreement")
        style = "shift" if p.lineage.split == "style_sealed" else "default"
        slots, types = _facts(world)
        raw, printed = _render(world, style)
        gold, unit = _answer(first, query)
        if (p.actor.source != GENERATOR_VERSION or p.actor.family != FAMILY
                or p.lineage.source != GENERATOR_VERSION or p.lineage.generator_version != GENERATOR_VERSION
                or p.lineage.template_id != f"circuit-{query[0]}"
                or p.actor.answer_format != "unit_number" or p.verifier.answer_type != "unit_number"
                or p.actor.problem_id != p.verifier.problem_id or p.actor.problem_id != p.lineage.problem_id
                or p.lineage.label_provenance != "PROGRAM" or p.lineage.model_touched or p.lineage.benchmark_derived
                or structure_hash(world) != p.lineage.structure_hash
                or not matches_split(p.lineage.structure_hash, p.lineage.split, style)
                or p.verifier.gold != gold or p.actor.question != _question(query)
                or p.actor.choices or p.verifier.choices or p.verifier.slots != slots
                or p.verifier.slot_types != types or p.verifier.required_slots != tuple(slots)
                or p.verifier.oracle_fact_lines != tuple(f"{k} = {v}" for k, v in slots.items())
                or p.verifier.accept["unit_aliases"] != [unit] or p.verifier.accept["require_unit"] is not True
                or p.verifier.printed_text != printed or p.image_bytes != (raw,)
                or p.actor.image_sha256 != (hashlib.sha256(raw).hexdigest(),)):
            raise ValueError("answer, structure, or rendering mismatch")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("circuit: self-check failed") from exc
