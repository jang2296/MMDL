"""Pure v16b rewards, source-bound note claims, and unchanged native SFT targets."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import json
import math
import re
from typing import Any

from . import content_v16 as content
from . import self_note_v16 as bounded
from .audit import digest, text_sha256
from .contract import ROUTE_FULL, ROUTE_LM, answer_denial, hint_leak_single_v2, is_loop
from .v4 import target_problem

ROW_SCHEMA = "error-note-v16b-row-v1"
NOTE_VERSION = "v16b-note-v1"
NOTE_INSTRUCTION_V16B = (
    "Inspect your actual first solution, even when its answer was correct, using the image, question, "
    "grading result, given answer and bounded catalog. Return one JSON object with exactly "
    "observed_facts, error, lesson. observed_facts is a list of zero to three {fact_ref,value} objects; "
    "use [] when no catalog is provided. error is null or an object containing exactly earlier_span "
    "(Unicode half-open offsets), earlier_quote, kind (OBSERVATION or ARITHMETIC), operand_fact_refs, "
    "operation, old_value, corrected_value, rounding, unit, rule_id. Quote one complete asserted fact "
    "or arithmetic expression, never a hypothetical, quoted, negated or retracted claim. Without a "
    "catalog only ARITHMETIC with ADD, SUB, MUL or DIV, no operand_fact_refs, empty unit and rounding "
    "to the quoted result's decimal places is supported. lesson is a JSON string: an abstract lesson "
    "of at most 40 words. Do not put problem numbers, options, names or the answer in it. "
    "Do not write a revised solution."
)
RESOLVE_INSTRUCTION_V16B = bounded.RESOLVE_INSTRUCTION_V16
OBS_FORMAT_V16B = {
    "geometry": 'Return JSON only: {"labels":["segment AB = value","angle ABC = value°"],'
                '"marks":["AB ∥ CD","AB ⊥ CD"]}; list only visible labels and marks.',
    "graph": 'Return JSON only: {"nodes":[0,1],"edges":[[0,1],[1,2,"weight"]]}; '
             'preserve direction when shown and include weights only when shown.',
    "scene": 'Return JSON only: {"objects":[{"size":"small","color":"red",'
             '"material":"metal","shape":"cube"}]}; preserve left-to-right order.',
}
_WRONG = {"INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "NO_FINAL", "NO_PARSE",
          "MULTIPLE_FINAL_ANSWERS"}
# A cut-off F0 was not correct (k_i = 0, as in curation); the reward table keeps its own LENGTH rule.
_F0_NOT_CORRECT = _WRONG | {"TRUNCATED"}
# vLLM stops on every generation_config EOS id; a terminal one is the unsupervised end token.
_EOS_TOKENS = frozenset({"<|im_end|>", "<|endoftext|>"})


def reward_v16b(status: str, finish_reason: str, text: str, extracted: Any) -> tuple[float | None, str, bool]:
    if finish_reason in {"length", "max_tokens", "max_new_tokens"}:
        return 0.0, "LENGTH", False
    if finish_reason not in {"stop", "eos"}:
        return None, "NOT_STOPPED", False
    if is_loop(text):
        return 0.0, "LOOP", False
    if status == "CORRECT":
        return (1.0, "CORRECT", True) if extracted is not None and str(extracted).strip() else (0.0, "NO_FINAL", True)
    if status in _WRONG:
        return 0.0, status, True
    return None, "UNVERIFIABLE", False


def _sealed(body: dict) -> dict:
    return body | {"sha256": digest(body)}


def _rate(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("rate", value.get("mean"))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("INVALID_BASELINE_RATE")
    return float(value)


def _history(history: list[dict], cutoff: int, kind: str) -> list[dict]:
    if type(cutoff) is not int or cutoff < -2:
        raise ValueError("EXPLICIT_HISTORY_CUTOFF_REQUIRED")
    selected = []
    for entry in history:
        block = entry.get("block_index", entry.get("block"))
        if type(block) is not int:
            raise ValueError("HISTORY_BLOCK_REQUIRED")
        if (block <= cutoff and (entry.get("committed") is True or entry.get("state") == "COMMITTED")
                and entry.get("kind", "T1") in ({"T1", "RL"} if kind == "T1" else {"OBS"})):
            selected.append(entry)
    return sorted(selected, key=lambda entry: (entry.get("block_index", entry.get("block")),
                                               str(entry.get("problem_id", ""))))


def prep_prior_v16b(records: list[dict]) -> dict:
    """Rates from the frozen F0 preparation outputs, never from current RL results."""
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        output = record["t1"]
        reward, _, _ = reward_v16b(output["status"], output["finish_reason"], output["text"], output.get("extracted"))
        if reward is not None:
            values[record["source"]].append(reward)
    all_values = [value for source in values.values() for value in source]
    if not all_values:
        raise ValueError("PREP_PRIOR_HAS_NO_GRADED_RESULTS")
    return _sealed({"version": "v16b-prep-prior-v1", "overall": sum(all_values) / len(all_values),
                    "source_rates": {source: sum(scores) / len(scores) for source, scores in sorted(values.items())},
                    "records_sha256": digest(records)})


def baseline_receipt_v16b(item: dict, history: list[dict], *, prior: dict,
                          history_cutoff: int | None = None) -> dict:
    cutoff = prior.get("history_cutoff") if history_cutoff is None else history_cutoff
    if type(cutoff) is not int:
        raise ValueError("EXPLICIT_HISTORY_CUTOFF_REQUIRED")
    committed = _history(history, cutoff, "T1")
    source = item["source"]
    rates = prior.get("source_rates", prior.get("sources", {}))
    initial = _rate(rates[source] if source in rates else prior.get("overall", prior.get("global")))
    window = [entry for entry in committed if entry["source"] == source][-64:]
    mean = (8 * initial + sum(_rate(entry["reward"]) for entry in window)) / (8 + len(window))
    previous = item.get("f0_prior")
    if isinstance(previous, dict):
        previous = previous.get("correct", previous.get("status"))
    if isinstance(previous, str):
        previous = int(previous == "CORRECT") if previous == "CORRECT" or previous in _F0_NOT_CORRECT else None
    if previous is None:
        baseline = mean
    else:
        if previous not in (0, 1) or isinstance(previous, str):
            raise ValueError("INVALID_F0_PRIOR")
        baseline = (2 * mean + int(previous)) / 3
    return _sealed({"version": "v16b-baseline-v1", "problem_id": item["problem_id"], "source": source,
                    "baseline": baseline, "source_mean": mean, "history_cutoff": cutoff,
                    "history_sha256": digest(committed), "prior_sha256": digest(prior),
                    "f0_prior": previous, "window_count": len(window)})


def baseline_v16b(item: dict, history: list[dict], *, prior: dict,
                  history_cutoff: int | None = None) -> float:
    return baseline_receipt_v16b(item, history, prior=prior, history_cutoff=history_cutoff)["baseline"]


def obs_baseline_v16b(layer: str, history: list[dict], *, history_cutoff: int,
                      prior: dict | None = None) -> dict:
    committed = _history(history, history_cutoff, "OBS")
    selected = [entry for entry in committed if entry["layer"] == layer]
    rates = (prior or {}).get("layer_rates", {})
    initial = _rate(rates[layer]) if layer in rates else None
    initial_block = None
    if initial is None and selected:
        initial_block = selected[0].get("block_index", selected[0].get("block"))
        first = [entry for entry in selected if entry.get("block_index", entry.get("block")) == initial_block]
        initial = sum(_rate(entry.get("score", entry.get("reward"))) for entry in first) / len(first)
    window = selected[-32:]
    baseline = None if initial is None else (
        8 * initial + sum(_rate(entry.get("score", entry.get("reward"))) for entry in window)) / (8 + len(window))
    return _sealed({"version": "v16b-obs-baseline-v1", "layer": layer, "baseline": baseline,
                    "zero_advantage": initial is None, "initial": initial, "initial_block": initial_block,
                    "history_cutoff": history_cutoff, "history_sha256": digest(committed),
                    "prior_sha256": digest(prior), "window_count": len(window)})


def _observation_item(item: dict) -> dict:
    """View already authenticated native annotations without inventing old v16 O approvals."""
    if isinstance(item.get("annotation"), dict):
        return item | {"parent_id": item.get("parent_id", item.get("problem_id", ""))}
    references = item.get("references", {})
    raw, metadata = references.get("raw_annotation", {}), references.get("native_metadata", {})
    layer = item.get("layer")
    if layer == "graph":
        structure = f"{raw['n']} {raw['m']}\n" + "\n".join(" ".join(map(str, edge)) for edge in raw["edges"])
        annotation = {**metadata, "structure_text": structure, "directed": metadata["directed"]}
    elif layer == "scene":
        annotation = raw["scene"]
    elif layer == "geometry":
        annotation = raw
    else:
        raise ValueError("OBS_UNSUPPORTED_LAYER")
    return item | {"annotation": annotation, "parent_id": item.get("parent_id", item["problem_id"])}


def observation_eligibility_v16b(item: dict) -> dict:
    """Optional OBS has explicit source limits; malformed supported inputs still raise."""
    layer = item.get("layer")
    annotation = item.get("annotation")
    if annotation is None:
        annotation = item.get("references", {}).get("raw_annotation")
    body = {"version": "v16b-observation-eligibility-v1", "problem_id": item["problem_id"],
            "layer": layer, "annotation_sha256": digest(annotation), "target_sha256": None,
            "eligible": False, "reason": None}
    if not annotation:
        return _sealed(body | {"reason": "NO_OBSERVATION_ANNOTATION"})
    if layer not in OBS_FORMAT_V16B:
        return _sealed(body | {"reason": "UNSUPPORTED_OBSERVATION_LAYER"})
    normalized = _observation_item(item)
    annotation = normalized["annotation"]
    if layer == "geometry":
        forms = annotation.get("diagram_logic_forms")
        if forms is None:
            forms = annotation["candidates"][annotation["matched_id"]]["diagram_logic_forms"]
        target, reason = content.geometry_target(forms)
    elif layer == "graph":
        target, reason = content.graph_target(content.parse_gvlqa_structure(annotation["structure_text"]),
                                              directed=annotation["directed"], parent_id=normalized["parent_id"])
    else:
        target, reason = content.scene_target(annotation)
    if reason in {"NO_FACTS", "UNSUPPORTED_FORM", "TOO_MANY_EDGES", "UNSUPPORTED_SCENE"}:
        return _sealed(body | {"reason": reason})
    if reason or target is None:
        raise ValueError("OBS_SOURCE_CONTRACT:" + str(reason))
    _observation_facts(layer, target, directed=annotation.get("directed", False))
    content.content_prompt(normalized, layer)
    return _sealed(body | {"eligible": True, "target_sha256": text_sha256(target)})


def native_note_contract(item: dict) -> dict:
    """Bound a catalog to native annotations; the manifest loader owns source authentication."""
    eligibility = observation_eligibility_v16b(item)
    if item.get("layer") not in OBS_FORMAT_V16B or eligibility["reason"] == "NO_OBSERVATION_ANNOTATION":
        return _sealed({"supported": False, "fact_catalog": [], "rules": {},
                        "annotation_sha256": eligibility["annotation_sha256"], "problem_id": item["problem_id"],
                        "observation_eligibility": eligibility})
    item = _observation_item(item)
    annotation, layer = item["annotation"], item["layer"]
    facts: list[dict[str, Any]] = []

    def fact(ref: str, entity: str, value: str, unit: str) -> None:
        if bounded._ATOM.fullmatch(value):
            facts.append({"fact_id": ref, "kind": "OBSERVATION", "entity": entity, "value": value,
                          "unit": unit, "answer_specific": False,
                          "evidence_origin": f"{layer}:{digest(annotation)}:{ref}"})

    if layer == "geometry":
        forms = annotation.get("diagram_logic_forms")
        if forms is None:
            forms = annotation["candidates"][annotation["matched_id"]]["diagram_logic_forms"]
        for index, form in enumerate(forms):
            length, angle = content._LEN.fullmatch(form), content._ANG.fullmatch(form)
            if length:
                fact(f"f{index}", f"segment {length[1]}{length[2]}", length[3], "length")
            elif angle:
                fact(f"f{index}", f"angle {angle[1]}{angle[2]}{angle[3]}", angle[4], "degree")
    elif layer == "graph":
        graph = content.parse_gvlqa_structure(annotation["structure_text"])
        # Only values that need the image: an edge entity already says "present", and once weighted edges are
        # listed by entity (all edges or none carry a weight) both counts follow from the public catalog too.
        if not any(len(edge) == 3 for edge in graph["edges"]):
            fact("visible_node_count", "number of nodes drawn in the image", str(len(graph["visible_nodes"])), "count")
            fact("edge_count", "number of drawn edges", str(graph["m"]), "count")
        for index, edge in enumerate(graph["edges"]):
            if len(edge) == 3:
                fact(f"edge_weight_{index}", f"weight of edge {edge[0]} to {edge[1]}", edge[2], "weight")
    elif layer == "scene":
        fact("object_count", "number of objects in the scene", str(len(annotation["objects"])), "count")
        target, reason = content.scene_target(annotation)
        if target is not None and reason is None:
            ordered = sorted(annotation["objects"], key=lambda obj: obj["pixel_coords"][0])
            positions = Counter(obj["pixel_coords"][0] for obj in ordered)
            for index, obj in enumerate(ordered, 1):
                if positions[obj["pixel_coords"][0]] != 1:
                    continue
                for key in ("size", "color", "material", "shape"):
                    fact(f"left_object_{index}_{key}", f"object {index} from the left {key}", obj[key], "label")
    rules = {}
    for unit in sorted({entry["unit"] for entry in facts}):
        suffix = text_sha256(unit)[:8]
        rules[f"LOOKUP_{suffix}"] = {"operation": "LOOKUP", "formula": bounded.FORMULAS["LOOKUP"],
                                    "operand_units": [unit], "result_unit": unit}
        if any(entry["unit"] == unit and re.fullmatch(bounded._NUMBER, entry["value"]) for entry in facts):
            for operation in ("ADD", "SUB", "DIV"):
                rules[f"{operation}_{suffix}"] = {
                    "operation": operation, "formula": bounded.FORMULAS[operation], "operand_units": [unit, unit],
                    "result_unit": "dimensionless" if operation == "DIV" else unit}
    bounded._catalog(facts, rules)
    return _sealed({"supported": bool(facts), "fact_catalog": facts, "rules": rules,
                    "annotation_sha256": digest(annotation), "problem_id": item["problem_id"],
                    "observation_eligibility": eligibility})


problem_contract_v16b = native_note_contract

def build_obs_prompt_v16b(item: dict, layer: str | None = None) -> str:
    layer = layer or item.get("layer")
    eligibility = observation_eligibility_v16b(item | {"layer": layer})
    if not eligibility["eligible"]:
        raise ValueError("OBS_INELIGIBLE:" + eligibility["reason"])
    layer = eligibility["layer"]
    item = _observation_item(item | {"layer": layer})
    instruction = content.content_prompt(item, layer)
    suffix = (" For an incident-edge request use nodes: [] and list only incident edges."
              if layer == "graph" and "Requested node:" in instruction else "")
    return instruction + "\n" + OBS_FORMAT_V16B[layer] + suffix


def _geometry_facts(labels: list, marks: list) -> Counter:
    result: Counter = Counter()
    for label in labels:
        if not isinstance(label, str):
            raise ValueError("OBS_LABEL_SCHEMA")
        match = re.fullmatch(r"(segment|angle)\s+([A-Z]{2,3})\s*=\s*([^;\n]+)", label.strip())
        if not match or len(match[2]) != (2 if match[1] == "segment" else 3):
            raise ValueError("OBS_LABEL_SCHEMA")
        entity, value = match[2], match[3].strip().rstrip("°").strip()
        if not content._EXPRESSION.fullmatch(value):
            raise ValueError("OBS_LABEL_VALUE")
        if re.fullmatch(bounded._NUMBER, value):
            value = str(Decimal(value).normalize())
        result[(match[1], min(entity, entity[::-1]), value)] += 1
    for mark in marks:
        match = re.fullmatch(r"([A-Z]{2})\s*([∥⊥])\s*([A-Z]{2})", mark.strip())
        if not match:
            raise ValueError("OBS_MARK_SCHEMA")
        result[(match[2], *sorted((min(match[1], match[1][::-1]), min(match[3], match[3][::-1]))))] += 1
    return result


def _observation_facts(layer: str, text: str, *, directed: bool = False) -> Counter:
    if text.lstrip().startswith("{"):
        obj = json.loads(text, object_pairs_hook=bounded._unique)
        if layer == "geometry" and set(obj) == {"labels", "marks"}:
            if not all(isinstance(obj[key], list) for key in obj):
                raise ValueError("OBS_SCHEMA")
            return _geometry_facts(obj["labels"], obj["marks"])
        if layer == "graph" and set(obj) == {"nodes", "edges"}:
            if not isinstance(obj["nodes"], list) or not isinstance(obj["edges"], list):
                raise ValueError("OBS_SCHEMA")
            result: Counter = Counter()
            for node in obj["nodes"]:
                if type(node) is not int or node < 0:
                    raise ValueError("OBS_NODE_SCHEMA")
                result[("node", node)] += 1
            for edge in obj["edges"]:
                if (not isinstance(edge, list) or len(edge) not in (2, 3)
                        or any(type(node) is not int or node < 0 for node in edge[:2])):
                    raise ValueError("OBS_EDGE_SCHEMA")
                ends = edge[:2] if directed else sorted(edge[:2])
                weight = str(Decimal(str(edge[2])).normalize()) if len(edge) == 3 else ""
                if weight and not Decimal(weight).is_finite():
                    raise ValueError("OBS_EDGE_WEIGHT")
                result[("edge", *ends, weight)] += 1
            return result
        if layer == "scene" and set(obj) == {"objects"} and isinstance(obj["objects"], list):
            result = Counter()
            for index, value in enumerate(obj["objects"]):
                if (not isinstance(value, dict) or set(value) != {"size", "color", "material", "shape"}
                        or any(not isinstance(v, str) or not re.fullmatch(r"[a-z]+", v) for v in value.values())):
                    raise ValueError("OBS_OBJECT_SCHEMA")
                result[("object", index, *(value[key] for key in ("size", "color", "material", "shape")))] += 1
            return result
        raise ValueError("OBS_SCHEMA")
    if layer == "geometry":
        match = re.fullmatch(r"Printed labels: (.*?)\.\s*Marks: (.*?)\.", text.strip(), re.S)
        if not match:
            raise ValueError("OBS_UNREADABLE")
        return _geometry_facts([] if match[1] == "none" else match[1].split("; "),
                               [] if match[2] == "none" else match[2].split("; "))
    if layer == "graph":
        match = re.fullmatch(r"(?:Undirected|Directed) graph\.\s*(?:Drawn nodes: ([\d, ]+)\.\s*|"
                             r"Requested node: \d+\.\s*Incident )Edges: (.*?)\.", text.strip(), re.I | re.S)
        if not match:
            raise ValueError("OBS_UNREADABLE")
        nodes = [] if match[1] is None else [int(node) for node in match[1].split(",")]
        edges = []
        for edge in match[2].split(", "):
            value = re.fullmatch(r"(\d+)\s*(?:->|-)\s*(\d+)(?: \(weight ([^()]+)\))?", edge)
            if not value:
                raise ValueError("OBS_UNREADABLE")
            edges.append([int(value[1]), int(value[2])] + ([value[3]] if value[3] else []))
        return _observation_facts(layer, json.dumps({"nodes": nodes, "edges": edges}), directed=directed)
    if layer == "scene" and text.startswith("Objects (left to right):\n"):
        objects = []
        for index, line in enumerate(text.splitlines()[1:], 1):
            match = re.fullmatch(rf"{index}\. ([a-z]+) ([a-z]+) ([a-z]+) ([a-z]+)\.", line)
            if not match:
                raise ValueError("OBS_UNREADABLE")
            objects.append(dict(zip(("size", "color", "material", "shape"), match.groups(), strict=True)))
        return _observation_facts(layer, json.dumps({"objects": objects}))
    raise ValueError("OBS_UNREADABLE")


def obs_score(item: dict, layer: str, text: str, *, finish_reason: str = "stop") -> dict:
    """F1 over native diagram/graph/scene facts, with scene positions part of each fact."""
    try:
        item = _observation_item(item | {"layer": layer})
        annotation = item["annotation"]
        if layer == "geometry":
            forms = annotation.get("diagram_logic_forms")
            if forms is None:
                forms = annotation["candidates"][annotation["matched_id"]]["diagram_logic_forms"]
            target, reason = content.geometry_target(forms)
        elif layer == "graph":
            target, reason = content.graph_target(content.parse_gvlqa_structure(annotation["structure_text"]),
                                                  directed=annotation["directed"], parent_id=item.get("parent_id", ""))
        elif layer == "scene":
            target, reason = content.scene_target(annotation)
        else:
            raise ValueError("OBS_UNSUPPORTED_LAYER")
        if reason or not target:
            raise ValueError(reason or "OBS_NO_FACTS")
        directed = annotation.get("directed", False)
        expected = _observation_facts(layer, target, directed=directed)
        if finish_reason not in {"stop", "eos"}:
            raise ValueError("OBS_NOT_STOPPED")
        actual = _observation_facts(layer, text, directed=directed)
        tp, fp, fn = sum((actual & expected).values()), sum((actual - expected).values()), sum((expected - actual).values())
        score = 2 * tp / (2 * tp + fp + fn) if actual or expected else 0.0
        return {"score": score, "f1": score, "readable": True, "tp": tp, "fp": fp, "fn": fn,
                "reason": None, "target_sha256": text_sha256(target)}
    except (KeyError, TypeError, ValueError, InvalidOperation, AttributeError):
        return {"score": 0.0, "f1": 0.0, "readable": False, "tp": 0, "fp": 0, "fn": 0,
                "reason": "OBS_UNREADABLE_OR_NOT_STOPPED"}


def json_spans(text: str) -> dict[str, list[int]]:
    """JSON value spans in Unicode offsets; decoded keys cannot impersonate sibling paths."""
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
    spans: dict[str, list[int]] = {}

    def skip(pos: int) -> int:
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1
        return pos

    def value(pos: int, path: str, depth: int = 0) -> int:
        if depth > 32:
            raise ValueError("JSON_TOO_DEEP")
        pos, start = skip(pos), skip(pos)
        if pos >= len(text):
            raise ValueError("INVALID_JSON")
        if text[pos] in "{[":
            opening = text[pos]
            closing = "}" if opening == "{" else "]"
            pos, index, keys = skip(pos + 1), 0, set()
            if pos < len(text) and text[pos] != closing:
                while True:
                    if opening == "{":
                        key, end = decoder.raw_decode(text, pos)
                        if not isinstance(key, str) or key in keys or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
                            raise ValueError("DUPLICATE_OR_UNSAFE_JSON_KEY")
                        keys.add(key)
                        pos = skip(end)
                        if pos >= len(text) or text[pos] != ":":
                            raise ValueError("INVALID_JSON")
                        child = f"{path}.{key}" if path else key
                        pos = value(pos + 1, child, depth + 1)
                    else:
                        pos = value(pos, f"{path}[{index}]", depth + 1)
                        index += 1
                    pos = skip(pos)
                    if pos >= len(text) or text[pos] != ",":
                        break
                    pos = skip(pos + 1)
            if pos >= len(text) or text[pos] != closing:
                raise ValueError("INVALID_JSON")
            pos += 1
        else:
            _, pos = decoder.raw_decode(text, pos)
        spans[path] = [start, pos]
        return pos

    if skip(value(0, "")) != len(text):
        raise ValueError("INVALID_JSON")
    return spans


def _entity_key(entity: str) -> tuple | None:
    if match := re.fullmatch(r"(?:segment|length of) ([A-Z]{2})", entity):
        return "segment", min(match[1], match[1][::-1])
    if match := re.fullmatch(r"angle ([A-Z]{3})", entity):
        return "angle", min(match[1], match[1][::-1])
    if match := re.fullmatch(r"(weight of )?edge ([A-Za-z0-9]+) to ([A-Za-z0-9]+)", entity):
        return "edge_weight" if match[1] else "edge", match[2], match[3]
    if match := re.fullmatch(r"object (\d+) from the left (color|shape|size|material)", entity):
        return "left_object", int(match[1]), match[2]
    return None


def _assertion_status(text: str, span: list[int], quote: str, *, natural: bool = False) -> str:
    """Reuse T2 assertion checks, rejecting explicit denial before any natural-form projection."""
    if (not isinstance(quote, str) or not quote or "\n" in quote or not isinstance(span, list)
            or len(span) != 2 or any(type(pos) is not int for pos in span)
            or not 0 <= span[0] < span[1] <= len(text) or text[span[0]:span[1]] != quote):
        return "UNKNOWN"
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    end = len(text) if end < 0 else end
    lead = text[:start].rstrip()
    lead = lead[lead.rfind("\n") + 1:] if not text[start:span[0]].strip() and lead.endswith(":") else ""
    if re.search(r"\b(?:false|untrue|falsely|falsehood|never|neither|nor|no|cannot)\b|\b\w+n['’]t\b",
                 text[start:end] + "\n" + lead, re.I):
        return "UNKNOWN"
    status = bounded.earlier_assertion_status(text, span, quote)
    if status != "UNKNOWN" or not natural:
        return status
    # The same T2 check sees a canonical fact while all surrounding prose stays intact.
    surrogate = "bound_fact = value"
    changed = text[:span[0]] + surrogate + text[span[1]:]
    return bounded.earlier_assertion_status(changed, [span[0], span[0] + len(surrogate)], surrogate)


def _natural_assertion(text: str, span: list[int], quote: str) -> str:
    return _assertion_status(text, span, quote, natural=True)


def bind_statement(text: str, catalog: list[dict]) -> list[dict]:
    """Bind exact named entities; ambiguous aliases remain unverified."""
    indexed = {fact["fact_id"]: fact for fact in catalog}
    candidates: list[tuple[re.Match, tuple | None, str]] = []
    number = bounded._NUMBER
    pattern = re.compile(rf"(?:(?P<angle>∠|angle\s+)(?P<ang>[A-Z]{{3}})|"
                         rf"(?:(?:segment|length of)\s+)?(?P<seg>[A-Z]{{2}}))\s*=\s*(?P<value>{number})°?")
    for match in pattern.finditer(text):
        entity = match["ang"] or match["seg"]
        candidates.append((match, ("angle" if match["ang"] else "segment", min(entity, entity[::-1])), match["value"]))
    edge = re.compile(rf"(?:edge\s*\(\s*(?P<a>[A-Za-z0-9]+)\s*,\s*(?P<b>[A-Za-z0-9]+)\s*\)|"
                      rf"(?P<c>[A-Za-z0-9]+)\s*[–−]\s*(?P<d>[A-Za-z0-9]+))"
                      rf"(?:\s*(?:=|weight\s+)\s*(?P<value>{number}))?")
    for match in edge.finditer(text):
        candidates.append((match, ("edge_weight" if match["value"] else "edge", match["a"] or match["c"],
                                   match["b"] or match["d"]), match["value"] or "present"))
    scene = re.compile(r"object\s+(\d+)\s+(?:(from the left)\s+)?(color|shape|size|material)"
                       r"\s*(?:=|is)\s*([a-z]+)", re.I)
    for match in scene.finditer(text):
        candidates.append((match, ("left_object" if match[2] else "unqualified_object",
                                   int(match[1]), match[3].lower()), match[4].lower()))
    for match in re.finditer(r"\b([A-Za-z][\w-]*)\s*=\s*([^=\n]{1,64})", text):
        if match[1] in indexed and not any(other.start() <= match.start() < other.end() for other, _, _ in candidates):
            candidates.append((match, None, match[2].strip()))
    result = []
    for match, entity, value in sorted(candidates, key=lambda entry: entry[0].start()):
        refs = ([match[1]] if entity is None else
                [ref for ref, fact in indexed.items() if _entity_key(fact["entity"]) == entity])
        span = [match.start(), match.end()]
        result.append({"fact_ref": refs[0] if len(refs) == 1 else None, "value": value,
                       "char_span": span, "quote": match[0],
                       "assertion": _natural_assertion(text, span, match[0]),
                       "status": "BOUND" if len(refs) == 1 else "UNVERIFIABLE"})
    for index, claim in enumerate(result):
        if claim["fact_ref"] is not None and claim["assertion"] == "ASSERTED" and any(
                later["fact_ref"] == claim["fact_ref"] and later["assertion"] == "ASSERTED"
                and not bounded._same(later["value"], claim["value"]) for later in result[index + 1:]):
            claim["assertion"] = "RETRACTED"
    return result


def _operation(equation: re.Match) -> str:
    if equation["c"] is not None:
        return "TRIANGLE_REMAINDER"
    return {"+": "ADD", "-": "SUB", "*": "MUL", "×": "MUL", "/": "DIV", "÷": "DIV"}[equation["op"]]


def _rounded_equation(equation: re.Match, operation: str) -> tuple[str, str, bool]:
    operands = bounded._equation_operands(equation, operation)
    result = equation["result"]
    digits = len(result.partition(".")[2])
    rounding = f"decimal:{digits}"
    exact = bounded._calculate(operation, operands, "none")
    rounded = bounded._calculate(operation, operands, rounding)
    truncated = exact.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_DOWN)
    original_correct = bounded._same(result, str(rounded)) or bounded._same(result, str(truncated))
    return str(rounded), rounding, original_correct


def t1_error_scan(text: str, catalog: list[dict], rules: dict) -> list[dict]:
    facts = bounded._catalog(catalog, rules)
    errors = []
    for claim in bind_statement(text, catalog):
        ref = claim["fact_ref"]
        if ref is not None and claim["assertion"] == "ASSERTED" and not bounded._same(claim["value"], facts[ref]["value"]):
            errors.append({"kind": "OBSERVATION", "earlier_span": claim["char_span"],
                           "earlier_quote": claim["quote"], "fact_ref": ref,
                           "old_value": claim["value"], "corrected_value": facts[ref]["value"]})
    for equation in bounded._EQUATION_IN.finditer(text):
        span = [equation.start(), equation.end()]
        if _assertion_status(text, span, equation[0]) != "ASSERTED":
            continue
        try:
            corrected, rounding, already_correct = _rounded_equation(equation, _operation(equation))
            if not already_correct:
                errors.append({"kind": "ARITHMETIC", "earlier_span": span, "earlier_quote": equation[0],
                               "old_value": equation["result"], "corrected_value": corrected,
                               "operation": _operation(equation), "rounding": rounding})
        except (ValueError, InvalidOperation):
            continue
    return sorted(errors, key=lambda error: error["earlier_span"])


def build_note_prompt_v16b(record: dict, *, fact_catalog: list[dict], rules: dict, gold: str) -> str:
    output = record["t1"]
    _, _, pg = reward_v16b(output["status"], output["finish_reason"], output["text"], output.get("extracted"))
    if not pg or output["status"] == "CORRECT" and not t1_error_scan(output["text"], fact_catalog, rules):
        raise ValueError("SKIP_NOTE_NO_ERROR_OR_INCOMPLETE")
    return NOTE_INSTRUCTION_V16B + "\n\n" + json.dumps({
        "question": record["p0_text"], "first_solution": output["text"], "grading_result": output["status"],
        "given_answer": gold, "note_catalog": bounded.public_note_catalog(fact_catalog, rules) if fact_catalog else {},
    }, ensure_ascii=False, sort_keys=True)


def build_resolve_prompt_v16b(p0_text: str, lesson: dict) -> str:
    if (not isinstance(lesson, dict) or set(lesson) != {"text"} or not isinstance(lesson["text"], str)
            or not lesson["text"].strip() or len(lesson["text"].split()) > 40):
        raise ValueError("INVALID_ABSTRACT_LESSON")
    return RESOLVE_INSTRUCTION_V16B + "\n\nAbstract lesson: " + lesson["text"] + "\n\n" + p0_text


def _final_contradiction(record: dict, old: str, corrected: str, choices: list | tuple, gold: str) -> bool:
    t1 = record["t1"]
    finals = [match[1] for line in t1["text"].splitlines()
              if (match := bounded._FINAL.fullmatch(line.replace("**", "").replace("__", "")))]
    finals += re.findall(r"\\boxed\{([^{}]+)\}", t1["text"])
    if isinstance(t1.get("extracted"), str):
        finals.append(t1["extracted"])
    key = bounded._answer_key(old, choices)
    is_final = any(bounded._same(old, final) or key is not None and key == bounded._answer_key(final, choices)
                   for final in finals)
    return is_final and bounded._answer_key(corrected, choices) != bounded._answer_key(gold, choices)


def _error_claim(error: dict, record: dict, facts: dict, rules: dict, choices: list | tuple, gold: str) -> tuple[str, str]:
    quote, span = error["earlier_quote"], error["earlier_span"]
    assertion = _assertion_status(record["t1"]["text"], span, quote)
    refs, operation = error["operand_fact_refs"], error["operation"]
    if error["kind"] == "OBSERVATION" and operation == "LOOKUP":
        matches = [claim for claim in bind_statement(record["t1"]["text"], list(facts.values()))
                   if claim["char_span"] == span and claim["quote"] == quote]
        if len(matches) != 1 or matches[0]["fact_ref"] is None or matches[0]["assertion"] != "ASSERTED":
            return "UNVERIFIABLE", "EARLIER_ASSERTION_UNKNOWN"
        claim = matches[0]
        if refs != [claim["fact_ref"]] or not bounded._same(claim["value"], error["old_value"]):
            return "UNVERIFIABLE", "QUOTE_FACT_MISMATCH"
        corrected = facts[refs[0]]["value"]
    elif error["kind"] == "ARITHMETIC" and operation != "LOOKUP":
        if assertion != "ASSERTED":
            return "UNVERIFIABLE", "EARLIER_ASSERTION_" + assertion
        equation = bounded._EQUATION.fullmatch(quote.strip())
        if equation is None or not bounded._same(equation["result"], error["old_value"]):
            return "UNVERIFIABLE", "QUOTE_ARITHMETIC_MISMATCH"
        try:
            operands = bounded._equation_operands(equation, operation)
            if not facts:
                if (operation not in {"ADD", "SUB", "MUL", "DIV"} or refs or error["unit"] != ""
                        or error["rule_id"] != operation):
                    return "UNVERIFIABLE", "UNSUPPORTED_NONCATALOG_ARITHMETIC"
                corrected, rounding, already_correct = _rounded_equation(equation, operation)
                if error["rounding"] not in ({rounding, "none"} if rounding == "decimal:0" else {rounding}):
                    return "UNVERIFIABLE", "ROUNDING_RULE_MISMATCH"
                if already_correct:
                    return "UNVERIFIABLE", "ORIGINAL_CLAIM_ALREADY_CORRECT"
            else:
                if (len(refs) != len(operands) or any(ref not in facts for ref in refs)
                        or any(not bounded._same(value, facts[ref]["value"]) for value, ref in zip(operands, refs))):
                    return "UNVERIFIABLE", "QUOTE_SOURCE_MISMATCH"
                corrected = str(bounded._calculate(operation, operands, error["rounding"]))
        except (ValueError, InvalidOperation):
            return "UNVERIFIABLE", "UNSUPPORTED_ARITHMETIC"
    else:
        return "UNVERIFIABLE", "UNSUPPORTED_ERROR_KIND"
    if facts:
        rule = rules.get(error["rule_id"], {})
        if (not rule or operation != rule["operation"] or error["rounding"] != rule.get("rounding", "none")
                or error["unit"] != rule["result_unit"]
                or any(ref not in facts for ref in refs)
                or [facts[ref]["unit"] for ref in refs] != rule["operand_units"]):
            return "UNVERIFIABLE", "OPERAND_BINDING_OR_UNIT"
    if bounded._same(error["old_value"], corrected):
        return "UNVERIFIABLE", "ORIGINAL_CLAIM_ALREADY_CORRECT"
    if not bounded._same(error["corrected_value"], corrected):
        return "CONTRADICTED", "CORRECTION_MISMATCH"
    if (_final_contradiction(record, error["old_value"], corrected, choices, gold)
            or facts and rules[error["rule_id"]].get("answer_specific", False)
            and bounded._answer_key(corrected, choices) != bounded._answer_key(gold, choices)):
        return "CONTRADICTED", "NOTE_GOLD_CONTRADICTION"
    return "VERIFIED", "SOURCE_BOUND_CORRECTION"


def verify_note_v16b(record: dict, *, fact_catalog: list[dict], rules: dict,
                      choices: list[str] | tuple[str, ...], gold: str) -> dict:
    body: dict[str, Any] = {"version": NOTE_VERSION, "problem_id": record["problem_id"], "claims": [],
            "verified": 0, "contradicted": 0, "accepted": False, "resolve_allowed": False,
            "verified_spans": [], "lesson": None, "reason": None,
            "t1_generation_sha256": digest(record.get("t1", {}).get("generation")),
            "note_generation_sha256": digest(record.get("note", {}).get("generation")),
            "source_contract_sha256": digest({"facts": fact_catalog, "rules": rules}), "lesson_sha256": None}
    try:
        build_note_prompt_v16b(record, fact_catalog=fact_catalog, rules=rules, gold=gold)
        facts = bounded._catalog(fact_catalog, rules)
        output = record["note"]
        if not bounded._stored_output(record["t1"]) or not bounded._stored_output(output):
            raise ValueError("CHANGED_ORIGINAL_GENERATION")
        if problem := target_problem(output["text"], output["generation"]["completion_token_ids"],
                                     output["finish_reason"], "CORRECT", min_tokens=1):
            raise ValueError(problem)
        spans = json_spans(output["text"])
        note = json.loads(output["text"], object_pairs_hook=bounded._unique)
        if not isinstance(note, dict) or set(note) != {"observed_facts", "error", "lesson"}:
            raise ValueError("NOTE_SCHEMA")
        observed, error, lesson = note["observed_facts"], note["error"], note["lesson"]
        if isinstance(lesson, str):
            lesson = {"text": lesson}
        if (not isinstance(lesson, dict) or set(lesson) != {"text"} or not isinstance(lesson["text"], str)
                or not lesson["text"].strip() or len(lesson["text"].split()) > 40):
            raise ValueError("NOTE_SCHEMA")
        claims: list[dict[str, Any]] = []
        # A malformed claim is never trained and never blocks the separately checked lesson.
        if not isinstance(observed, list) or len(observed) > 3:
            claims.append({"path": "observed_facts", "status": "UNVERIFIABLE", "char_span": spans["observed_facts"],
                           "reason": "OBSERVED_FACT_SCHEMA"})
            observed = []
        refs = Counter(fact["fact_ref"] for fact in observed
                       if isinstance(fact, dict) and isinstance(fact.get("fact_ref"), str))
        for index, fact in enumerate(observed):
            path = f"observed_facts[{index}]"
            if (not isinstance(fact, dict) or set(fact) != {"fact_ref", "value"}
                    or not isinstance(fact["fact_ref"], str) or refs[fact["fact_ref"]] > 1
                    or not isinstance(fact["value"], str) or not bounded._ATOM.fullmatch(fact["value"])):
                claims.append({"path": path, "status": "UNVERIFIABLE", "char_span": spans[path],
                               "reason": "OBSERVED_FACT_SCHEMA"})
                continue
            ref = fact["fact_ref"]
            status = ("UNVERIFIABLE" if ref not in facts else
                      "VERIFIED" if bounded._same(fact["value"], facts[ref]["value"]) else "CONTRADICTED")
            if (ref in facts and facts[ref].get("answer_specific", False)
                    and bounded._answer_key(fact["value"], choices) != bounded._answer_key(gold, choices)):
                status = "CONTRADICTED"
            claims.append({"path": path, "status": status, "char_span": spans[path]})
        if error is not None:
            try:
                if (not isinstance(error, dict) or set(error) != bounded._ERROR_FIELDS
                        or any(not isinstance(error[key], str) for key in bounded._ERROR_FIELDS - {"earlier_span", "operand_fact_refs"})
                        or not isinstance(error["operand_fact_refs"], list)
                        or any(not isinstance(ref, str) for ref in error["operand_fact_refs"])
                        or any(not bounded._ATOM.fullmatch(error[key]) for key in ("old_value", "corrected_value"))):
                    raise ValueError("ERROR_SCHEMA")
                status, reason = _error_claim(error, record, facts, rules, choices, gold)
            except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
                status, reason = "UNVERIFIABLE", "ERROR_SCHEMA"
            claims.append({"path": "error", "status": status, "char_span": spans["error"], "reason": reason})
        body["claims"] = claims
        verified = sum(claim["status"] == "VERIFIED" for claim in claims)
        contradicted = sum(claim["status"] == "CONTRADICTED" for claim in claims)
        body["verified"] = verified
        body["contradicted"] = contradicted
        body["lesson"] = lesson
        body["lesson_sha256"] = digest(lesson)
        if leak := hint_leak_single_v2(lesson["text"], gold, choices):
            raise ValueError("LESSON_LEAK:" + leak)
        if body["contradicted"]:
            raise ValueError(next((claim.get("reason", "OBSERVED_FACT_MISMATCH") for claim in claims
                                   if claim["status"] == "CONTRADICTED"), "CONTRADICTED"))
        verified_spans = []
        for claim in claims:
            if claim["status"] != "VERIFIED":
                continue
            start, end = claim["char_span"]
            exclusions = sorted(spans[path] for path in ("error.earlier_quote", "error.old_value")
                                if claim["path"] == "error" and path in spans)
            for left, right in exclusions:
                if start < left:
                    verified_spans.append([start, left])
                start = right
            if start < end:
                verified_spans.append([start, end])
        body.update(accepted=verified > 0, resolve_allowed=True,
                    verified_spans=sorted(verified_spans), reason=None if verified else "NO_VERIFIED_CLAIMS")
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as exc:
        body["reason"] = str(exc) if isinstance(exc, ValueError) else "NOTE_SCHEMA"
        body.update(accepted=False, resolve_allowed=False, verified_spans=[])
    return _sealed(body)


def _record_defaults(record: dict) -> dict:
    return {"subject": record.get("subject", ""), "layer": record.get("layer"),
            "image_paths": record.get("image_paths", []), "image_sha256": record.get("image_sha256", []), **record}


def _self_row(record: dict, output: dict, kind: str, spans: list[list[int]], tokenizer: Any,
              receipt: dict) -> tuple[dict | None, str | None]:
    """(row, None), or (None, reason) when the sampled tokens cannot give a mask (e.g. a generated control token)."""
    from .train import END_TOKEN
    from .train_v16 import token_byte_mask

    record = _record_defaults(record)
    if not bounded._stored_output(output):
        raise ValueError("CHANGED_ORIGINAL_GENERATION")
    target, generation = output["text"], output["generation"]
    actions = generation["completion_token_ids"]
    byte_spans = [[len(target[:start].encode()), len(target[:end].encode())] for start, end in spans]
    end = (actions[-1] if tokenizer.convert_ids_to_tokens(actions[-1:])[0] in _EOS_TOKENS
           else tokenizer.convert_tokens_to_ids(END_TOKEN))
    try:
        mask = token_byte_mask(tokenizer, actions, target, byte_spans, end_token_id=end, supervise_end=False)
    except ValueError as exc:
        return None, "SFT_MASK:" + str(exc)
    return {"schema": ROW_SCHEMA, "row_id": record["problem_id"] + ":" + kind, "kind": kind,
            **{key: record[key] for key in ("problem_id", "source", "subject", "layer", "image_paths", "image_sha256")},
            "parent_id": record.get("parent_id", record["problem_id"]),
            "block_index": record.get("block_index"), "policy_version": generation.get("policy_version"),
            "route": ROUTE_FULL if record["image_paths"] else ROUTE_LM,
            "prompt": output["prompt"] if kind == "N" else record["p0_text"], "target": target,
            "target_source": {"N": "SELF_NOTE", "C": "SELF_RESOLVE", "PREP_SELF": "T1"}[kind],
            "target_origin": "self", "generation": generation, "status": output.get("status", "NOTE"),
            "complete": True, "probability_space": generation.get("probability_space", "processed-full-vocab-v1"),
            "verified_byte_spans": byte_spans, "loss_mask": mask, "verification_receipt": receipt,
            "supervise_end": False}, None


def note_row(record: dict, *, tokenizer: Any, fact_catalog: list[dict], rules: dict,
             choices: list[str] | tuple[str, ...], gold: str) -> dict | None:
    receipt = verify_note_v16b(record, fact_catalog=fact_catalog, rules=rules, choices=choices, gold=gold)
    if not receipt["accepted"]:
        return None
    return _self_row(record, record["note"], "N", receipt["verified_spans"], tokenizer, receipt)[0]


def c_row_with_receipt(record: dict, *, tokenizer: Any, fact_catalog: list[dict], rules: dict,
                       choices: list[str] | tuple[str, ...], gold: str,
                       note_receipt: dict | None = None) -> tuple[dict | None, dict]:
    from fine_tuning.notebook_rl import v8_parser
    from .single import _a2_check

    receipt = verify_note_v16b(record, fact_catalog=fact_catalog, rules=rules, choices=choices, gold=gold)
    if note_receipt is not None and receipt != note_receipt:
        raise ValueError("NOTE_RECEIPT_MISMATCH")
    output = record.get("resolve")
    body = {"version": "v16b-c-v1", "accepted": False, "note_receipt_sha256": receipt["sha256"],
            "generation_sha256": digest(output.get("generation") if isinstance(output, dict) else None)}

    def reject(reason: str, **details: Any) -> tuple[None, dict]:
        return None, _sealed(body | {"reason": reason} | details)

    if not receipt["resolve_allowed"]:
        return reject("RESOLVE_NOT_ALLOWED", note_reason=receipt["reason"])
    if not isinstance(output, dict):
        return reject("MISSING_RESOLVE_OUTPUT")
    if not bounded._stored_output(output):
        return reject("CHANGED_ORIGINAL_GENERATION")
    if any(record.get(stage) is not None for stage in ("resolve2", "resolve3")):
        return reject("MULTIPLE_RESOLVE_ATTEMPTS")
    if len(output["generation"]["completion_token_ids"]) > 4096:
        return reject("RESOLVE_TOO_LONG")
    candidate, reason = _a2_check(_record_defaults(record), "resolve", receipt["lesson"]["text"], image_route=True)
    if candidate is None or reason:
        return reject(reason or "A2_NOT_ADMISSIBLE")
    if t1_error_scan(output["text"], fact_catalog, rules):
        return reject("PROCESS_CONTRADICTION")
    parsed = (v8_parser.extract_mcq(output["text"], choices, output["finish_reason"]) if choices else
              v8_parser.extract_open(output["text"], output["finish_reason"], question=record["p0_text"]))
    final, gold_key = bounded._answer_key(str(output.get("extracted", "")), choices), bounded._answer_key(gold, choices)
    parsed_ok = parsed["status"] == v8_parser.PARSED
    if not choices and gold_key is None:
        # Multi-word or range open gold: no single-token key, and V8 reads a range as several numbers.
        # grade_single already graded this committed answer CORRECT (_a2_check above).
        final = bounded.normalize_final(str(output.get("extracted") or "")).casefold() or None
        gold_key = bounded.normalize_final(gold).casefold()
        parsed_ok = parsed_ok or parsed.get("rule") == "multiple_numeric_values"
    if not parsed_ok or final is None or final != gold_key:
        return reject("PARSE_GOLD_MISMATCH")
    for line in output["text"].splitlines():
        match = bounded._CONCLUSION.fullmatch(line.replace("**", ""))
        if match and (key := bounded._answer_key(match[1], choices)) is not None and key != final:
            return reject("FINAL_CONTRADICTION")
    evidence = _sealed(body | {"accepted": True, "content_status": "PARTIALLY_CHECKED"})
    row, problem = _self_row(record, output, "C", [[0, len(output["text"])]], tokenizer, evidence)
    return (row, evidence) if row is not None else reject(problem or "SFT_MASK")


def c_row(record: dict, *, tokenizer: Any, fact_catalog: list[dict], rules: dict,
          choices: list[str] | tuple[str, ...], gold: str, note_receipt: dict | None = None) -> dict | None:
    return c_row_with_receipt(record, tokenizer=tokenizer, fact_catalog=fact_catalog, rules=rules,
                              choices=choices, gold=gold, note_receipt=note_receipt)[0]


def _prep_human(record: dict, tokenizer: Any) -> dict | None:
    from .references import human_solution
    from .single import human_row

    human = record.get("references", {}).get("human_solution")
    if (not isinstance(human, dict) or human.get("authorship") != "dataset_native_expert_curated"
            or human.get("contamination_status") != "PASS_FRESH_TEXT_SCAN_WITH_LIMITS"
            or human.get("human_verified") is not False):
        return None
    fields = {key: human.get(key, "") for key in ("lecture", "solution")}
    if (any(not isinstance(text, str) for text in fields.values()) or not fields["solution"].strip()
            or human.get("text_sha256") != {key: text_sha256(text) for key, text in fields.items()}):
        return None
    for key in ("source_ref", "provenance_card", "text_check_receipt"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(human.get(key, {}).get("sha256", ""))):
            return None
    # The manifest loader authenticates this native reference and text-check receipt before row construction.
    mapped = human_solution({"source": record["source"], "annotations": [
        {"field": key, "value": text} for key, text in fields.items()]})
    fields = mapped if mapped and mapped.get("solution", "").strip() else fields
    evaluation = record.get("evaluation", record)
    row, _ = human_row(_record_defaults(record), fields, str(evaluation["gold"]), evaluation.get("choices", []), tokenizer)
    if row is None:
        return None
    receipt = _sealed({"version": "v16b-prep-human-v1", "target_sha256": text_sha256(row["target"]),
                       "native_reference_sha256": digest(human), "text_check_receipt": human["text_check_receipt"],
                       "human_verified": False})
    return row | {"schema": ROW_SCHEMA, "row_id": record["problem_id"] + ":PREP_HUMAN", "kind": "PREP_HUMAN",
                  "route": ROUTE_FULL if record.get("image_paths") else ROUTE_LM, "target_origin": "human",
                  "supervise_end": False, "verification_receipt": receipt, "layer": record.get("layer")}


def prep_rows(records: list[dict], *, tokenizer: Any, max_self: int = 160, max_human: int = 128) -> list[dict]:
    from fine_tuning.notebook_rl import v8_parser

    if not 0 <= max_self <= 160 or not 0 <= max_human <= 128:
        raise ValueError("PREP_ROW_LIMIT")
    selected: dict[str, list[dict]] = {"PREP_SELF": [], "PREP_HUMAN": []}
    if len({record["problem_id"] for record in records}) != len(records):
        raise ValueError("DUPLICATE_PREP_PARENT")
    for record in records:
        output = record.get("t1", {})
        actions = output.get("generation", {}).get("completion_token_ids", [])
        evaluation = record.get("evaluation", record)
        choices = evaluation.get("choices", [])
        if (bounded._stored_output(output) and output.get("prompt") == record["p0_text"] and len(actions) <= 4096
                and not target_problem(output["text"], actions, output["finish_reason"], output["status"], min_tokens=256)
                and not answer_denial(output["text"])):
            final = (v8_parser.extract_mcq(output["text"], choices, output["finish_reason"]) if choices else
                     v8_parser.extract_open(output["text"], output["finish_reason"], question=record["p0_text"]))
            if final["status"] == v8_parser.PARSED:
                receipt = _sealed({"version": "v16b-prep-self-v1", "target_sha256": text_sha256(output["text"]),
                                   "generation_sha256": digest(output["generation"]), "content_status": "UNKNOWN"})
                row, _ = _self_row(record, output, "PREP_SELF", [[0, len(output["text"])]], tokenizer, receipt)
                if row is not None:
                    selected["PREP_SELF"].append(row)
        human = _prep_human(record, tokenizer)
        if human is not None:
            selected["PREP_HUMAN"].append(human)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for kind, limit in (("PREP_SELF", max_self), ("PREP_HUMAN", max_human)):
        for row in sorted(selected[kind], key=lambda value: text_sha256(value["row_id"]))[:limit]:
            buckets[(row["source"], kind)].append(row)
    result = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                result.append(buckets[key].pop(0))
    return result
