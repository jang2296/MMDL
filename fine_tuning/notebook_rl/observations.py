"""Parse declared observations only; absent observations are not visual errors."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from .answers import number
from .schema import SlotReport, VerifierItem

_LINE = re.compile(r"^[ \t]*-?[ \t]*([a-z_]+(?:\[[^\]\n]+\]){0,2})[ \t]*=[ \t]*(.+?)[ \t]*$", re.I)
_BEGIN = re.compile(r"^[ \t]*Observed facts:[ \t]*\r?$", re.I | re.M)
_END = re.compile(r"^[ \t]*Solution:[ \t]*\r?$", re.I | re.M)


def slot_id(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"\s+", "", value)
    return re.sub(r"\[([^\]]+)\]", lambda m: "[" + m[1].replace("'", "").replace('"', "") + "]", value)


def parse_observations(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for _, _, key, value in observation_spans(text):
        result.setdefault(key, []).append(value)
    return result


def observation_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Return original assignment offsets only inside accepted observation blocks."""
    starts = list(_BEGIN.finditer(text))
    # Repeated blocks are not silently resolved by choosing the most convenient one.
    if not starts:
        return []
    result = []
    for i, start in enumerate(starts):
        end = _END.search(text, start.end())
        if end is None or (i + 1 < len(starts) and end.start() > starts[i + 1].start()):
            continue
        offset = start.end()
        for line in text[start.end():end.start()].splitlines(keepends=True):
            match = _LINE.fullmatch(line.splitlines()[0])
            if match:
                result.append((offset, offset + len(line), slot_id(match[1]), match[2]))
            offset += len(line)
    return result


def normalize_slot(value: str, slot_type: str) -> str | None:
    if slot_type == "relation":
        value = _relation_value(value)
    value = unicodedata.normalize("NFKC", value).strip()
    if slot_type == "number":
        # Units may be omitted only in observation slots, never in final-answer scoring.
        parsed = number(value)
        if parsed is None:
            unit = r"[A-Za-zµΩ°%]+(?:[0-9]|\^-?[0-9]+)?"
            match = re.fullmatch(rf"(.+?)[ \t]*({unit}(?:[ /·*]{unit})*)", value)
            if match:
                parsed = number(match[1])
        return str(parsed) if parsed is not None else None
    if slot_type == "pitch":
        value = value.replace("♯", "#").replace("♭", "b").replace("♮", "")
        value = re.sub("sharp", "#", value, flags=re.I)
        value = re.sub("flat", "b", value, flags=re.I)
        value = re.sub("natural", "", value, flags=re.I)
        value = re.sub(r"\s+", "", value)
        match = re.fullmatch(r"([A-Ga-g])([#b]{0,2})(-?[0-9])", value)
        return match[1].upper() + match[2] + match[3] if match else None
    if slot_type == "text":
        return " ".join(value.casefold().split()) or None
    if slot_type == "relation":
        value = value.casefold()
        match = re.fullmatch(r"([a-z0-9_]+)[ \t]*(?:<->|↔|,|-)[ \t]*([a-z0-9_]+)", value)
        if match:
            return "-".join(sorted((match[1], match[2])))
        return value if re.fullmatch(r"[a-z0-9_]+", value) else None
    raise ValueError("observation: unsupported slot type")


def _relation_value(value: str) -> str:
    value = value.strip(" \t")
    # Remove exactly one original ASCII wrapper; nested or mismatched quotes stay invalid.
    return value[1:-1] if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0] else value


def _normalize(value: str, kind: str, key: str) -> str | None:
    if kind == "relation" and (key in ("start", "root") or key.startswith(("trans[", "dir[", "child["))):
        value = unicodedata.normalize("NFKC", _relation_value(value)).strip(" \t").casefold().replace("→", "->")
        pattern = (r"[a-z0-9_]+[ \t]*->[ \t]*[a-z0-9_]+" if key.startswith("dir[")
                   else r"[a-z0-9_]+")
        return re.sub(r"[ \t]+", "", value) if re.fullmatch(pattern, value) else None
    return normalize_slot(value, kind)


def native_absent_relations(keys: Iterable[str], item: VerifierItem) -> tuple[str, ...]:
    """Prove unregistered relation keys absent from a complete native graph world."""
    world = item.accept.get("graph_world")
    if (not item.problem_id.startswith("gen-graph-v1:graph:") or not item.slots
            or not isinstance(world, dict) or world.get("kind") not in ("bfs", "shortest", "tree")):
        return ()
    from .generators.graph import _facts, _slot_types, _validate_world

    try:
        _validate_world(world)
        facts = _facts(world)
        if item.slots != facts or item.slot_types != _slot_types(facts):
            return ()
        tree = world["kind"] == "tree"
        nodes = {slot_id(node) for node in (facts.values() if tree else world["nodes"])}
        registered = {slot_id(key) for key in facts}
        relation = "child" if tree else "edge"
        return tuple(sorted({key for key in keys if key not in registered
            and (match := re.fullmatch(rf"{relation}\[([a-z0-9_]+),([a-z0-9_]+)\]", key))
            and match[1] in nodes and match[2] in ({"l", "r"} if tree else nodes)}))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return ()


def verified_wrong_slots(slots: SlotReport) -> tuple[str, ...]:
    return tuple(sorted(set(slots.wrong_required) | set(slots.false_relations)))


def required_observations_verified(slots: SlotReport | None) -> bool:
    return bool(slots and slots.required > 0 and slots.required_correct == slots.required
                and not slots.conflicting and not slots.false_relations and not slots.unverified_relations)


def compare_slots(parsed: dict[str, list[str]], item: VerifierItem) -> SlotReport:
    if set(item.slots) != set(item.slot_types):
        raise ValueError("observation: mismatched reference slot types")
    keys = {slot_id(k): k for k in item.slots}
    if len(keys) != len(item.slots):
        raise ValueError("observation: colliding normalized reference slot IDs")
    required = {slot_id(k) for k in item.required_slots}
    if not required <= keys.keys() or len(required) != len(item.required_slots):
        raise ValueError("observation: invalid required slots")
    observed: dict[str, list[str]] = {}
    extra_relations: dict[str, list[str]] = {}
    for key, values in parsed.items():
        if not isinstance(key, str) or not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("observation: expected raw value lists")
        key = slot_id(key)
        if key in keys and values:
            observed.setdefault(key, []).extend(values)
        elif key.startswith(("edge[", "child[")) and values:
            extra_relations.setdefault(key, []).extend(values)
    correct, wrong, conflicts = set(), set(), set()
    for key, original in keys.items():
        kind = item.slot_types[original]
        expected = _normalize(item.slots[original], kind, key)
        if expected is None:
            raise ValueError("observation: unverifiable reference slot")
        if key not in observed:
            continue
        normalized = [_normalize(v, kind, key) for v in observed[key]]
        if len(set(normalized)) > 1:
            conflicts.add(key)
        if all(v == expected for v in normalized):
            correct.add(key)
        elif any(v is not None and v != expected for v in normalized):
            wrong.add(key)
    absent = set(native_absent_relations(extra_relations, item)) if extra_relations else set()
    false_relations, unverified_relations = set(), set()
    for key, values in extra_relations.items():
        if key.startswith("child["):
            normalized = [normalize_slot(value, "relation") for value in values]
            nodes = {slot_id(value) for value in item.slots.values()}
            positive = any(value in nodes for value in normalized)
            conflicting = len(set(normalized)) > 1
        else:
            numeric = [number(unicodedata.normalize("NFKC", value)) for value in values]
            positive = any(value is not None and value > 0 for value in numeric)
            conflicting = len(set(numeric)) > 1
        if conflicting:
            conflicts.add(key)
        if key in absent and positive:
            # A contradictory duplicate cannot erase an explicit verified false assertion.
            false_relations.add(key)
        else:
            unverified_relations.add(key)
    # Missing slots are absent evidence. Only present wrong slots justify a visual route.
    return SlotReport(len(observed), len(required), len(correct & required),
                      tuple(sorted(wrong & required)), len(correct), len(keys),
                      tuple(sorted(conflicts)), tuple(sorted(false_relations)),
                      tuple(sorted(unverified_relations)))
