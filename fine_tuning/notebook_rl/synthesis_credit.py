"""Host-only credit for independently verified submitted facts and relationships.

Strict synthesis remains unchanged. Receipts contain no replacement facts, lesson,
teacher target, routing authority, or claim of complete diagnosis or improvement.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace
from typing import Any

from . import episode_notebook as episode, notebook_synthesis as synthesis
from .local_eval import digest
from .notebook_learning import _unique
from .observations import slot_id
from .rollout import _hash

VERSION = "verified-synthesis-unit-token-credit-v2"
ASSESSMENT_VERSION = "verified-whole-reference-assessment-credit-v3"
ASSESSMENT_WEIGHT = .0625
ASSESSMENT_NORMALIZATION_CAP = 8192
FACT_WEIGHT = RELATIONSHIP_WEIGHT = .0625
GROUP_SIZE = 4
NATIVE_ERROR_RELATIONSHIP = "NATIVE_ERROR_RELATIONSHIP"
OUTPUT_FINALIZATION = "OUTPUT_FINALIZATION"
_REFERENCE_FIELDS = ("notebook_generation_policy", "public_endpoints", "public_endpoints_sha256",
                     "public_endpoints_binding_sha256")
_PAIRED_FIELDS = ("notebook_generation_policy", "public_pairs", "public_pairs_sha256",
                  "public_pairs_binding_sha256")
_PAIR_ACTION_FIELDS = ("pair_action_binding", "pair_action_binding_sha256")
_WHOLE_REFERENCE_FIELDS = ("notebook_generation_policy", "whole_reference_binding", "whole_reference_binding_sha256")
_WHOLE_REFERENCE_SAMPLE = "whole_reference_sample_binding"
_EXPANSION_FIELDS = ("expansion_status", "expanded_text_sha256", "expanded_raw_text_sha256")
_ASSESSMENT_FIELDS = ("assessment_refs", "assessment_rewards", "assessment_states", "assessment_expected_states",
                      "assessment_spans", "assessment_rows_complete", "native_assessment_count")


def contract(version: str = VERSION) -> dict:
    if version not in (VERSION, ASSESSMENT_VERSION):
        raise ValueError("synthesis credit: unknown objective version")
    if version == ASSESSMENT_VERSION:
        return {**contract(), "objective_version": version, "assessment_weight": ASSESSMENT_WEIGHT,
            "formula": contract()["formula"] + " + 0.0625*correct_native_assessments/all_public_native_refs",
            "assessment_unit": "exact independently checked flags and ordered dependencies; null permits partial classification; nonnull authored decision must be verified",
            "assessment_denominator": "all original public OBSERVATION and TRACE refs, including N and scope refs; no OUTPUT or ABSENCE",
            "assessment_states": "fresh issue I plus supported unverified U; N means no supported claim, never compliance",
            "assessment_decisions": "nonnull native source and destination flags/dependencies must match fresh states; no recursive decision validation",
            "assessment_tokens": "union of original flags, blocked_by_refs and nonnull decision literals; authored JSON null has no assessment PG; exclude ref bookkeeping",
            "assessment_coefficient": "8192*0.0625*(reward-original_n4_mean)/all_public_native_refs/mapped_union_tokens",
            "assessment_normalization_cap": ASSESSMENT_NORMALIZATION_CAP,
            "assessment_normalization": "assessment coefficient alone cancels cap8192 in the shared n4*cap8192 division; original n4, fixed reference and mapped-token denominators and group weights remain",
            "mis_and_kl": "unchanged full-completion mask; strict, facts, relationships and KL retain the original n4*cap8192 division",
            "assessment_policy": "whole-history-reference-assessments-v2 monolithic only"}
    return {"objective_version": VERSION, "strict_reward_version": synthesis.BOUNDED_VERSION,
        "strict_weight": 1.0, "fact_weight": FACT_WEIGHT, "relationship_weight": RELATIONSHIP_WEIGHT,
        "formula": "strict + 0.0625*correct_required_facts/required_facts + 0.0625*correct_relationships/original_native_and_output_issues",
        "zero_denominator": "zero credit, no redistribution", "group_size": GROUP_SIZE,
        "fact_unit": "independently verified submitted required string value; canonical-ID collisions zero the fact component",
        "relationship_unit": "FIRST submitted lesson per original earlier issue, selected before grading",
        "positive_relationships": {NATIVE_ERROR_RELATIONSHIP: ["CORRECTED", "PERSISTED"],
                                   OUTPUT_FINALIZATION: ["FINALIZED"]},
        "relationship_denominator": "all fixed original native issues plus every violated OUTPUT/FINALIZATION/final_answer check, including unrepaired siblings",
        "output_finalization_scope": "whole earlier absence to an actual descendant's single terminal final answer; presence is independent of answer correctness",
        "output_answer_correctness": "separate fresh later_answer_correct boolean; never inferred from FINALIZED credit",
        "output_checks_are_mandatory_native_issues": False,
        "fact_tokens": "V16 coefficient on wholly contained original JSON value literal tokens",
        "relationship_tokens": "coefficient divided by mapped tokens of the original supported lesson object",
        "missing_or_malformed": "unit zero; no invented action; unknown fields receive no partial PG",
        "centering": "original complete n4 per-unit means before MIS, never recentered after masking",
        "strict_credit": "separately centered and clipped whole-completion strict reward",
        "token_boundary": "stable prefix decode only; crossing and special tokens excluded",
        "mis_and_kl": "unchanged full-completion mask and fixed normalization",
        "lesson_semantics_verified": False, "all_reasoning_verified": False, "full_diagnosis": False,
        "module_causality_certified": False, "model_claims_used_for_routing": False}


def _bad_constant(value: str) -> None:
    raise ValueError("synthesis credit: nonfinite JSON")


def _json_units(text: str) -> tuple[dict | None, list[tuple], list[tuple]]:
    """Walk decoder offsets, never search for a repeated textual value or object."""
    decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_float=Decimal, parse_constant=_bad_constant)

    def whitespace(index: int) -> int:
        while index < len(text) and text[index] in " \r\n\t":
            index += 1
        return index

    def members(start: int):
        index = whitespace(start + 1)
        while text[index] != "}":
            key, end = decoder.raw_decode(text, index)
            begin = whitespace(whitespace(end) + 1)
            value, end = decoder.raw_decode(text, begin)
            yield key, value, begin, end
            index = whitespace(end)
            if text[index] == "}":
                break
            index = whitespace(index + 1)

    def elements(start: int):
        index = whitespace(start + 1)
        while text[index] != "]":
            value, end = decoder.raw_decode(text, index)
            yield value, index, end
            index = whitespace(end)
            if text[index] == "]":
                break
            index = whitespace(index + 1)

    try:
        note = decoder.decode(text)
        if type(note) is not dict:
            return None, [], []
        top = {key: (value, begin, end) for key, value, begin, end in members(whitespace(0))}
        facts = list(members(top["corrected_facts"][1])) if type(note.get("corrected_facts")) is dict else []
        lessons = list(elements(top["lessons"][1])) if type(note.get("lessons")) is list else []
        return note, facts, lessons
    except (ValueError, TypeError, IndexError, InvalidOperation, RecursionError):
        return None, [], []


def issue_id(attempt: int, row: dict) -> str:
    return json.dumps([attempt, row["kind"], row["rule"], row["locus"]], separators=(",", ":"))


def _origin(lesson: Any) -> str | None:
    earlier = lesson.get("earlier") if type(lesson) is dict else None
    if (type(earlier) is not dict or type(earlier.get("attempt")) is not int
            or any(type(earlier.get(key)) is not str for key in ("kind", "rule", "locus"))):
        return None
    return issue_id(earlier["attempt"], earlier)


def _selected(lessons: list[tuple], original_ids: list[str], public_index: dict | None = None,
              paired_index: dict | None = None) -> dict[str, int]:
    selected = {}
    for index, (lesson, _, _) in enumerate(lessons):
        origins = [_origin(lesson)]
        if public_index is not None and type(lesson) is dict and type(lesson.get("earlier_ref")) is str:
            origins.append(_origin({"earlier": public_index.get(lesson["earlier_ref"])}))
        if paired_index is not None and type(lesson) is dict and type(lesson.get("pair_id")) is str:
            origins.append(_origin(paired_index.get(lesson["pair_id"])))
        for origin in origins:
            if origin in original_ids and origin not in selected:
                selected[origin] = index
    return selected


def _credit_lesson(value: Any, public_index: dict | None, paired_index: dict | None = None) -> Any:
    if paired_index is not None:
        from .paired_notebook import expand_lesson as paired_expand

        return paired_expand(value, paired_index)
    if public_index is None:
        return value
    from .referenced_notebook import expand_lesson as reference_expand

    return reference_expand(value, public_index)


def _paired_lessons(note: dict | None, lessons: list[tuple]) -> list[tuple]:
    if (note is None or set(note) != {"coverage", "lessons"} or note["coverage"] != "PARTIAL"
            or type(note["lessons"]) is not list):
        return []
    return lessons


def _pair_action_units(text: str, finish_reason: str, target: dict, originals: list[str]) -> tuple[dict, dict, dict]:
    from .notebook_pair_probe import parse_label

    origin = _origin(target)
    if origin not in originals:
        return {}, {}, {}
    label = parse_label(text, finish_reason)
    spans = {origin: [len(text) - len(text.lstrip(" \t\r\n")), len(text.rstrip(" \t\r\n"))]} if label else {}
    return {origin: 0}, spans, {origin: label}


def _pair_action_metadata(tree, sample_id: str, target: dict, solution: str, transcript: str) -> dict:
    from . import pair_action_notebook as action

    sample = next(sample for group in tree.groups if group.kind == "DIAG" for sample in group.samples
                  if sample.sample_id == sample_id)
    note = tree.notebook["rewarded_notebook"]
    binding = {"sample_id": sample_id, "group_id": sample.group_id,
        "completion_token_ids_sha256": digest(sample.completion_token_ids),
        "generation_contract": action.contract(), "target_plan": note["target_plan"],
        "target_plan_sha256": note["target_plan_sha256"], "target_group": note["target_per_group"][sample.group_id]}
    if binding["target_group"]["target"] != target:
        raise ValueError("synthesis credit: pair action target differs from bound group")
    binding = copy.deepcopy(binding)
    return {"notebook_generation_policy": action.VERSION, "pair_action_binding": binding,
            "pair_action_binding_sha256": digest({"binding": binding, "solution_sha256": solution,
                                                  "transcript_sha256": transcript})}


def _validate_pair_action_binding(row: dict, sample) -> dict:
    from . import pair_action_notebook as action

    failure = "synthesis credit: invalid bound pair action"
    binding = row.get("pair_action_binding")
    if (row.get("notebook_generation_policy") != action.VERSION or type(binding) is not dict
            or set(binding) != {"sample_id", "group_id", "completion_token_ids_sha256", "generation_contract",
                                "target_plan", "target_plan_sha256", "target_group"}
            or any(key in row for key in (*_REFERENCE_FIELDS[1:], *_PAIRED_FIELDS[1:]))
            or binding["generation_contract"] != action.contract()
            or binding["sample_id"] != sample.sample_id or binding["group_id"] != sample.group_id
            or sample.kind != "DIAG" or binding["completion_token_ids_sha256"] != digest(sample.completion_token_ids)
            or row.get("pair_action_binding_sha256") != digest({"binding": binding,
                "solution_sha256": row["solution_sha256"], "transcript_sha256": row["transcript_sha256"]})):
        raise ValueError(failure)
    plan, group = binding["target_plan"], binding["target_group"]
    if (type(plan) is not list or not plan or binding["target_plan_sha256"] != _hash(plan)
            or type(group) is not dict or set(group) != {"target_index", "target", "messages", "messages_sha256",
                                                       "prompt_sha256", "image_key", "images", "seeds"}
            or type(group["target_index"]) is not int or not 1 <= group["target_index"] <= len(plan)
            or group["target"] != plan[group["target_index"] - 1]
            or type(group["messages"]) is not list or group["messages_sha256"] != _hash(group["messages"])
            or group["prompt_sha256"] != _hash(sample.prompt_token_ids)
            or type(group["image_key"]) is not str or type(group["images"]) is not dict
            or type(group["seeds"]) is not list or len(group["seeds"]) != GROUP_SIZE
            or any(type(seed) is not int for seed in group["seeds"])):
        raise ValueError(failure)
    if (sample.group_id != f"{sample.problem_id}:v{sample.policy_version}:PAIR_ACTION:target{group['target_index']}"
            or not any(sample.sample_id == f"{sample.group_id}:{index}" and sample.seed == seed
                       for index, seed in enumerate(group["seeds"]))):
        raise ValueError(failure)
    target = group["target"]
    if (type(target) is not dict or set(target) != {"pair_id", "channel", "earlier", "later", "raw_values"}
            or target["channel"] not in ("NATIVE", "OUTPUT") or type(target["pair_id"]) is not str
            or type(target["raw_values"]) is not list or target["raw_values"] != []):
        raise ValueError(failure)
    public_index = {target["pair_id"]: {side: target[side] for side in ("earlier", "later")}}
    _validate_paired_index({**_paired_metadata(public_index, row["solution_sha256"], row["transcript_sha256"]),
                           "solution_sha256": row["solution_sha256"], "transcript_sha256": row["transcript_sha256"]})
    if target["earlier"]["kind"] not in (("OUTPUT",) if target["channel"] == "OUTPUT" else ("OBSERVATION", "TRACE")):
        raise ValueError(failure)
    try:
        body = group["messages"][0]["content"][-1]["text"]
        projected = json.loads(body.rsplit("<target_pair>\n", 1)[1].split("\n</target_pair>", 1)[0])
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        raise ValueError(failure) from None
    if projected != {"earlier_attempt": target["earlier"]["attempt"], "later_attempt": target["later"]["attempt"],
                     "kind": target["earlier"]["kind"], "rule": target["earlier"]["rule"],
                     "earlier_locus": target["earlier"]["locus"], "later_locus": target["later"]["locus"]}:
        raise ValueError(failure)
    return target


def _reference_metadata(public_index: dict, solution_sha256: str, transcript_sha256: str) -> dict:
    from .referenced_notebook import VERSION as reference_version

    metadata = {"notebook_generation_policy": reference_version, "public_endpoints": public_index,
                "public_endpoints_sha256": digest(public_index)}
    return metadata | {"public_endpoints_binding_sha256": digest({
        "notebook_generation_policy": reference_version, "public_endpoints_sha256": metadata["public_endpoints_sha256"],
        "solution_sha256": solution_sha256, "transcript_sha256": transcript_sha256})}


def _paired_metadata(public_index: dict, solution_sha256: str, transcript_sha256: str) -> dict:
    from .paired_notebook import VERSION as paired_version

    metadata = {"notebook_generation_policy": paired_version, "public_pairs": public_index,
                "public_pairs_sha256": digest(public_index)}
    return metadata | {"public_pairs_binding_sha256": digest({
        "notebook_generation_policy": paired_version, "public_pairs_sha256": metadata["public_pairs_sha256"],
        "solution_sha256": solution_sha256, "transcript_sha256": transcript_sha256})}


def _validate_paired_index(row: dict) -> dict:
    """Validate receipt bookkeeping; fresh scoring must still rebuild all public pairs."""
    from .paired_notebook import _pair_id

    failure = "synthesis credit: invalid bound public pair index"
    public_index = row.get("public_pairs")
    if type(public_index) is not dict or any(key in row for key in _REFERENCE_FIELDS[1:]):
        raise ValueError(failure)
    endpoints: dict[tuple, dict] = {}
    for key, pair in public_index.items():
        if type(key) is not str or type(pair) is not dict or set(pair) != {"earlier", "later"}:
            raise ValueError(failure)
        for endpoint in pair.values():
            if (type(endpoint) is not dict or set(endpoint) != synthesis._ENDPOINT
                    or type(endpoint["attempt"]) is not int or endpoint["attempt"] < 1
                    or endpoint["kind"] not in ("OBSERVATION", "TRACE", "OUTPUT")
                    or any(type(endpoint[field]) is not str or not endpoint[field] for field in ("kind", "rule", "locus"))
                    or not _range_shape(endpoint["evidence_ranges"])):
                raise ValueError(failure)
            ranges = endpoint["evidence_ranges"]
            if ranges and (len(ranges) != 1 or ranges[0][0] != 1 or ranges[0][1] < 1):
                raise ValueError(failure)
            identity = tuple(endpoint[field] for field in ("attempt", "kind", "rule", "locus"))
            if identity in endpoints and endpoints[identity] != endpoint:
                raise ValueError(failure)
            endpoints[identity] = endpoint
        earlier, later = pair["earlier"], pair["later"]
        if (earlier["attempt"] >= later["attempt"] or earlier["kind"] != later["kind"]
                or earlier["rule"] != later["rule"] or key != _pair_id(earlier, later)
                or earlier["kind"] == "OBSERVATION" and earlier["locus"] != later["locus"]
                or earlier["kind"] == "OUTPUT" and any((endpoint["rule"], endpoint["locus"])
                    != ("FINALIZATION", "final_answer") for endpoint in (earlier, later))):
            raise ValueError(failure)
    expected = _paired_metadata(public_index, row["solution_sha256"], row["transcript_sha256"])
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(failure)
    return public_index


def _validate_reference_index(row: dict) -> dict | None:
    """Check saved public metadata; optimizer callers still regenerate it from actual solutions."""
    if not any(key in row for key in _REFERENCE_FIELDS):
        return None
    failure = "synthesis credit: invalid bound public reference index"
    public_index = row.get("public_endpoints")
    if type(public_index) is not dict or not public_index or any(type(key) is not str for key in public_index):
        raise ValueError(failure)
    attempts: dict[int, list[tuple[int, dict]]] = {}
    for key, endpoint in public_index.items():
        identity = re.fullmatch(r"a([1-9][0-9]*):e([1-9][0-9]*)", key)
        if (identity is None or type(endpoint) is not dict or set(endpoint) != synthesis._ENDPOINT
                or type(endpoint["attempt"]) is not int or endpoint["attempt"] != int(identity[1])
                or endpoint["kind"] not in ("OBSERVATION", "TRACE", "OUTPUT")
                or any(type(endpoint[field]) is not str or not endpoint[field] for field in ("kind", "rule", "locus"))
                or not _range_shape(endpoint["evidence_ranges"])):
            raise ValueError(failure)
        ranges = endpoint["evidence_ranges"]
        if ranges and (len(ranges) != 1 or ranges[0][0] != 1 or ranges[0][1] < 1):
            raise ValueError(failure)
        attempts.setdefault(endpoint["attempt"], []).append((int(identity[2]), endpoint))
    if sorted(attempts) != list(range(1, len(attempts) + 1)):
        raise ValueError(failure)
    for entries in attempts.values():
        entries.sort(key=lambda pair: pair[0])
        endpoints = [endpoint for _, endpoint in entries]
        if ([ordinal for ordinal, _ in entries] != list(range(1, len(entries) + 1))
                or len({(endpoint["kind"], endpoint["rule"], endpoint["locus"]) for endpoint in endpoints}) != len(entries)
                or any(endpoint["evidence_ranges"] != endpoints[0]["evidence_ranges"] for endpoint in endpoints)
                or [endpoint for endpoint in endpoints if endpoint["kind"] == "OUTPUT"] != [endpoints[-1]]
                or (endpoints[-1]["kind"], endpoints[-1]["rule"], endpoints[-1]["locus"])
                != ("OUTPUT", "FINALIZATION", "final_answer")):
            raise ValueError(failure)
    expected = _reference_metadata(public_index, row["solution_sha256"], row["transcript_sha256"])
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(failure)
    return public_index


def _range_shape(value: Any) -> bool:
    return type(value) is list and all(type(pair) is list and len(pair) == 2
        and all(type(number) is int for number in pair) for pair in value)


def _lesson_shape(value: Any) -> bool:
    if (type(value) is not dict or set(value) != {"rule", "earlier", "later", "relation"}
            or type(value["rule"]) is not str or type(value["relation"]) is not str
            or value["relation"] not in synthesis._RELATIONS | {"UNVERIFIABLE_LATER", "SCOPE_UNVERIFIABLE_LATER", "FINALIZED"}):
        return False
    finalized = value["relation"] == "FINALIZED"
    if finalized and value["rule"] != "FINALIZATION":
        return False
    for key in ("earlier", "later"):
        endpoint = value[key]
        unknown = key == "later" and value["relation"] in {"UNVERIFIABLE_LATER", "SCOPE_UNVERIFIABLE_LATER"}
        absent = key == "later" and value["relation"] == "NOT_REVISITED"
        if (type(endpoint) is not dict or set(endpoint) != synthesis._ENDPOINT | ({"blocked_by"} if unknown else set())
                or type(endpoint["kind"]) is not str or not _range_shape(endpoint["evidence_ranges"])
                or not (type(endpoint["attempt"]) is int or absent and endpoint["attempt"] is None)
                or any(not (type(endpoint[field]) is str or absent and endpoint[field] is None)
                       for field in ("rule", "locus"))):
            return False
        if (finalized and (endpoint["kind"], endpoint["rule"], endpoint["locus"])
                != ("OUTPUT", "FINALIZATION", "final_answer")
                or not finalized and endpoint["kind"] == "OUTPUT"):
            return False
        if unknown and (type(endpoint["blocked_by"]) is not list or any(type(row) is not dict
                or set(row) != {"rule", "locus"} or any(type(item) is not str for item in row.values())
                for row in endpoint["blocked_by"])):
            return False
    return True


def _fact_candidates(note: dict | None, facts: list[tuple]) -> dict[str, tuple[str, list[int]]]:
    if note is None or type(note.get("corrected_facts")) is not dict:
        return {}
    normalized = [slot_id(key) for key, _, _, _ in facts]
    if len(normalized) != len(set(normalized)):
        return {}
    return {slot_id(key): (value, [start, end]) for key, value, start, end in facts
            if synthesis._single_line_scalar(key) and synthesis._single_line_scalar(value)
            and synthesis._native_value_shape(key, value)}


def _seal(row: dict) -> dict:
    return row | {"score_sha256": digest(row)}


def _strict_digest(value: dict) -> str:
    # Rejected submitted claims can retain Decimal values; hash their types without rewriting the report.
    def typed(item):
        if type(item) is dict:
            return ["object", [[key, typed(value)] for key, value in sorted(item.items())]]
        if type(item) is list:
            return ["array", [typed(value) for value in item]]
        return [type(item).__name__, str(item)]
    try:
        return digest(value)
    except (TypeError, ValueError):
        return digest(typed(value))


def _whole_reference_metadata(samples, expanded: dict, solution: str, transcript: str, policy_version: int) -> dict:
    from . import whole_reference_notebook as reference

    binding = {"generation_contract": reference.contract(), "original_samples": [
        {key: getattr(sample, key) for key in ("text", "sample_id", "parent_sample_id", "finish_reason")}
        for sample in samples], "inventory_sha256": _hash(expanded["inventory"]), "source_policy_version": policy_version}
    rendered = expanded["expanded_text"]
    return {"notebook_generation_policy": reference.VERSION, "whole_reference_binding": binding,
        "whole_reference_binding_sha256": digest({"binding": binding, "solution_sha256": solution,
                                                  "transcript_sha256": transcript}),
        "expansion_status": expanded["status"],
        "expanded_text_sha256": digest(rendered) if rendered is not None else None,
        "expanded_raw_text_sha256": hashlib.sha256(rendered.encode()).hexdigest() if rendered is not None else None}


def _whole_reference_units(text: str, expanded: dict) -> tuple[dict | None, list[tuple], list[tuple]]:
    from . import whole_reference_notebook as reference

    if expanded["status"] != reference.CAPTURED:
        return None, [], []
    note, facts, _ = _json_units(text)
    return note, facts, [(lesson, *expanded["decision_spans"][ref])
                         for ref, lesson in expanded["decisions"].items()]


def _assessment_states(inventory: dict, audit: dict) -> dict:
    """Expected claims are host-only; an absent claim never certifies compliance."""
    references = inventory["references"]
    native = [ref for ref in inventory["assessment_refs"] if references[ref]["kind"] in ("OBSERVATION", "TRACE")]
    index = {(row["attempt"], row["kind"], row["rule"], row["locus"]): ref
             for ref, row in references.items() if row["kind"] != "ABSENCE"}
    states = {}
    for ref in native:
        source = references[ref]
        attempt = audit["attempts"][source["attempt"] - 1]
        key = synthesis._key(source)
        issues = [row for row in attempt["issues"] if synthesis._key(row) == key]
        unknowns = [row for row in attempt["unverified"] if synthesis._key(row) == key
                    and synthesis._eligible_unknown(row, attempt["checks"], accounted=True)]
        if len(issues) > 1 or len(unknowns) > 1:
            raise ValueError("synthesis credit: ambiguous fresh assessment state")
        dependencies = []
        for dependency in unknowns[0]["blocked_by"] if unknowns else []:
            target = index.get((source["attempt"], "TRACE", dependency["rule"], dependency["locus"]))
            if target is None or target in dependencies or native.index(target) >= native.index(ref):
                raise ValueError("synthesis credit: fresh assessment dependency has no earlier public ref")
            dependencies.append(target)
        states[ref] = {"flags": ("I" if issues else "") + ("U" if unknowns else "") or "N",
                       "blocked_by": dependencies}
    return states


def _assessment_units(text: str, finish_reason: str, inventory: dict) -> tuple[bool, dict, dict, dict]:
    """Decode original literals independently of header and other row validity."""
    from . import whole_reference_notebook as reference

    if finish_reason != "stop":
        return False, {}, {}, {}
    try:
        note, offsets = reference._decode(text)
        if (type(note) is not dict or set(note) != reference._TOP or type(note["assessments"]) is not list
                or len(note["assessments"]) != len(inventory["assessment_refs"])):
            return False, {}, {}, {}
        rows, states, spans = {}, {}, {}
        for position, (ref, row) in enumerate(zip(inventory["assessment_refs"], note["assessments"], strict=True)):
            if type(row) is not list or not row or type(row[0]) is not str or row[0] != ref:
                return False, {}, {}, {}
            rows[ref] = row
            if inventory["references"][ref]["kind"] not in ("OBSERVATION", "TRACE"):
                continue
            states[ref] = {"flags": row[1] if len(row) > 1 and type(row[1]) is str else None,
                "blocked_by": copy.deepcopy(row[2]) if len(row) > 2 and type(row[2]) is list
                    and all(type(value) is str for value in row[2]) else None}
            spans[ref] = [offsets[("assessments", position, field)] for field in (1, 2, 3)
                          if field < len(row) and (field != 3 or row[field] is not None)]
        return True, rows, states, spans
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, InvalidOperation):
        return False, {}, {}, {}


def _assessment_state_matches(ref: str, rows: dict, expected: dict, inventory: dict) -> bool:
    def matches(key: str) -> bool:
        row = rows.get(key)
        state = expected.get(key)
        if inventory["references"][key]["kind"] == "OUTPUT":
            state = {"flags": "N", "blocked_by": []}
        return bool(type(row) is list and len(row) == 4 and state is not None
                    and type(row[1]) is str and row[1] == state["flags"]
                    and type(row[2]) is list and all(type(value) is str for value in row[2])
                    and row[2] == state["blocked_by"])

    if not matches(ref):
        return False
    source = inventory["references"][ref]
    for dependency in rows[ref][2]:
        target = inventory["references"].get(dependency)
        if (target is None or source["kind"] != "TRACE" or target["kind"] != "TRACE"
                or target["attempt"] != source["attempt"]
                or inventory["assessment_refs"].index(dependency) >= inventory["assessment_refs"].index(ref)
                or not matches(dependency)):
            return False
        if source["rule"].startswith("CIRCUIT_") or target["rule"].startswith("CIRCUIT_"):
            from .circuit_trace import RULES

            if source["rule"] not in RULES or target["rule"] not in RULES or "U" not in rows[dependency][1]:
                return False
        elif "I" not in rows[dependency][1]:
            return False
    return True


def _assessment_decision(ref: str, rows: dict, inventory: dict, expected: dict) -> tuple[bool, dict | None]:
    """Expand one authored decision using existing public grammar, without rewriting it."""
    from . import whole_reference_notebook as reference

    if not _assessment_state_matches(ref, rows, expected, inventory):
        return False, None
    decision = rows[ref][3]
    if decision is None:
        return True, None
    references = inventory["references"]
    source = references[ref]
    if (type(decision) is not list or len(decision) != 2 or type(decision[0]) is not str
            or decision[0] not in reference._RELATIONS or type(decision[1]) is not str
            or decision[1] not in references):
        return False, None
    relation, later_ref = decision
    later = references[later_ref]
    if ((source["kind"] == "OUTPUT") != (relation == "FINALIZED")
            or source["kind"] != "OUTPUT" and "I" not in rows[ref][1]):
        return False, None
    target = reference._endpoint(later)
    if relation == "NOT_REVISITED":
        if later_ref != inventory["absence_refs"][str(source["attempt"])]:
            return False, None
    elif (later["kind"] == "ABSENCE" or later["attempt"] <= source["attempt"]
            or not _assessment_state_matches(later_ref, rows, expected, inventory)):
        return False, None
    elif relation == "FINALIZED":
        if later["kind"] != "OUTPUT":
            return False, None
    elif relation in {"UNVERIFIABLE_LATER", "SCOPE_UNVERIFIABLE_LATER"}:
        if ("U" not in rows[later_ref][1] or relation == "SCOPE_UNVERIFIABLE_LATER"
                and (later["kind"] != "TRACE" or (later["rule"], later["locus"]) not in reference._ROOTS[:3])):
            return False, None
        target["blocked_by"] = [{key: references[dep][key] for key in ("rule", "locus")}
                                for dep in rows[later_ref][2]]
    elif later["kind"] == "OUTPUT":
        return False, None
    return True, {"rule": source["rule"], "earlier": reference._endpoint(source),
                  "later": target, "relation": relation}


def _score_assessments(score: dict, text: str, finish_reason: str, expanded: dict, samples, fresh: dict) -> None:
    inventory = expanded["inventory"]
    expected = _assessment_states(inventory, fresh)
    complete, rows, states, spans = _assessment_units(text, finish_reason, inventory)
    rewards = dict.fromkeys(expected, 0)
    for ref in expected:
        valid, lesson = _assessment_decision(ref, rows, inventory, expected) if complete else (False, None)
        rewards[ref] = int(valid and (lesson is None or synthesis._lesson(lesson, samples, fresh, bounded=True)["verified"]))
    # Relationship credit cannot bypass the independently checked endpoint states.
    origins = {issue_id(source["attempt"], source): ref for ref, source in inventory["references"].items()
               if source["kind"] != "ABSENCE"}
    for origin, reward in score["relationship_rewards"].items():
        if reward:
            valid, _ = _assessment_decision(origins[origin], rows, inventory, expected)
            if not valid:
                score["relationship_rewards"][origin] = 0
                score["relationship_later_answer_correct"].pop(origin, None)
    atomic = FACT_WEIGHT * (sum(score["fact_rewards"].values()) / score["required_fact_count"]
                           if score["required_fact_count"] else 0.)
    atomic += RELATIONSHIP_WEIGHT * (sum(score["relationship_rewards"].values()) / len(score["original_issue_ids"])
                                    if score["original_issue_ids"] else 0.)
    atomic += ASSESSMENT_WEIGHT * (sum(rewards.values()) / len(expected) if expected else 0.)
    score.update(objective_version=ASSESSMENT_VERSION, objective_contract_sha256=digest(contract(ASSESSMENT_VERSION)),
        assessment_refs=list(expected), assessment_rewards=rewards, assessment_states=states,
        assessment_expected_states=expected, assessment_spans=spans, assessment_rows_complete=complete,
        native_assessment_count=len(expected), atomic_reward=atomic, effective_reward=score["strict_reward"] + atomic)


def _whole_reference_sample_binding(sample) -> dict:
    return {"sample_id": sample.sample_id, "group_id": sample.group_id,
            "policy_version": sample.policy_version,
            "prompt_token_ids_sha256": digest(sample.prompt_token_ids),
            "completion_token_ids_sha256": digest(sample.completion_token_ids)}


def _validate_whole_reference_binding(row: dict, sample) -> dict:
    from . import whole_reference_notebook as reference

    failure = "synthesis credit: invalid bound whole reference"
    binding = row.get("whole_reference_binding")
    if (row.get("notebook_generation_policy") != reference.VERSION or type(binding) is not dict
            or set(binding) != {"generation_contract", "original_samples", "inventory_sha256", "source_policy_version"}
            or binding["generation_contract"] != reference.contract()
            or type(binding["source_policy_version"]) is not int or binding["source_policy_version"] != sample.policy_version
            or type(binding["original_samples"]) is not list or not binding["original_samples"]
            or sample.kind != "DIAG" or row.get(_WHOLE_REFERENCE_SAMPLE) != _whole_reference_sample_binding(sample)
            or any(type(source) is not dict or set(source) != {"text", "sample_id", "parent_sample_id", "finish_reason"}
                   for source in binding["original_samples"])
            or any(key in row for key in (*_REFERENCE_FIELDS[1:], *_PAIRED_FIELDS[1:], *_PAIR_ACTION_FIELDS))):
        raise ValueError(failure)
    try:
        samples = [SimpleNamespace(**source) for source in binding["original_samples"]]
        expanded = reference.expand(sample.text, sample.finish_reason, samples)
        expected = _whole_reference_metadata(samples, expanded, row["solution_sha256"], row["transcript_sha256"],
                                             binding["source_policy_version"])
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise ValueError(failure) from exc
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(failure)
    return expanded


def _score_submission(text, finish_reason, tree, fresh, strict, verifier, actor,
                      note, facts, lessons, public_index=None, paired_index=None, action_target=None, *,
                      original=None) -> dict:
    """Shared existing unit scoring on submitted grading text; never creates an action."""
    if original is None:
        original = episode.audit_episode(tree, actor, verifier, routing_rule=episode.STRICT_ROUTE_RULE)
    empty = verifier.grade(tree.problem_id, "Observed facts:\nSolution:\n", "stop", "obs").slots
    required = empty.required if empty else 0
    if type(required) is not int or required < 0:
        raise ValueError("synthesis credit: invalid fixed required-fact count")
    units = [(attempt["attempt"], issue) for attempt in fresh["attempts"] for issue in
             [*attempt["issues"], *(row for row in attempt["output_checks"] if row["status"] == "VIOLATED")]]
    originals = [issue_id(attempt, issue) for attempt, issue in units]
    credit_types = {issue_id(attempt, issue): OUTPUT_FINALIZATION if issue["kind"] == "OUTPUT"
                    else NATIVE_ERROR_RELATIONSHIP for attempt, issue in units}
    if len(originals) != len(set(originals)):
        raise ValueError("synthesis credit: duplicate original known issue")
    samples = episode.solution_samples(tree)
    candidates = _fact_candidates(note, facts)
    fact_rewards, fact_spans = {}, {}
    for key, (value, span) in candidates.items():
        slots = verifier.grade(tree.problem_id, f"Observed facts:\n{key} = {value}\nSolution:\n", "stop", "obs").slots
        if slots is None:
            continue
        if slots.required != required:
            raise ValueError("synthesis credit: required-fact denominator changed")
        if not (slots.required_correct == 1 or key in slots.wrong_required):
            continue
        fact_rewards[key] = int(slots.required_correct == slots.parsed == slots.all_correct == 1
            and not slots.conflicting and not slots.false_relations and not slots.unverified_relations)
        fact_spans[key] = span
    selected = _selected(lessons, originals, public_index, paired_index)
    relationships = dict.fromkeys(originals, 0)
    relationship_spans: dict[str, list[int]] = {}
    relations: dict[str, str | None] = {}
    later_answer_correct: dict[str, bool] = {}
    if action_target is not None:
        selected, relationship_spans, relations = _pair_action_units(text, finish_reason, action_target, originals)
    for origin, index in selected.items():
        if action_target is not None:
            lesson = {"rule": action_target["earlier"]["rule"], "earlier": action_target["earlier"],
                      "later": action_target["later"], "relation": relations[origin]}
        else:
            submitted, start, end = lessons[index]
            relations[origin] = submitted.get("relation") if type(submitted.get("relation")) is str else None
            lesson = _credit_lesson(submitted, public_index, paired_index)
        if not _lesson_shape(lesson):
            continue
        if action_target is None:
            relationship_spans[origin] = [start, end]
        supported = (lesson["relation"] == "FINALIZED" if credit_types[origin] == OUTPUT_FINALIZATION
                     else lesson["relation"] in ("CORRECTED", "PERSISTED"))
        if supported:
            checked = synthesis._lesson(lesson, samples, fresh, bounded=True)
            relationships[origin] = int(checked["verified"])
            if checked["verified"] and credit_types[origin] == OUTPUT_FINALIZATION:
                if type(checked.get("later_answer_correct")) is not bool:
                    raise ValueError("synthesis credit: verified finalization lacks separate answer correctness")
                later_answer_correct[origin] = checked["later_answer_correct"]
    atomic = FACT_WEIGHT * (sum(fact_rewards.values()) / required if required else 0.)
    atomic += RELATIONSHIP_WEIGHT * (sum(relationships.values()) / len(originals) if originals else 0.)
    score = {"objective_version": VERSION, "objective_contract_sha256": digest(contract()),
        "strict_reward_version": synthesis.BOUNDED_VERSION, "strict_reward": strict["reward"],
        "atomic_reward": atomic, "effective_reward": strict["reward"] + atomic,
        "fact_rewards": fact_rewards, "relationship_rewards": relationships,
        "relationship_credit_types": credit_types, "relationship_later_answer_correct": later_answer_correct,
        "required_fact_count": required, "original_issue_ids": originals, "selected_lesson_indices": selected,
        "fact_spans": fact_spans, "relationship_spans": relationship_spans, "relationship_relations": relations,
        "json_complete": note is not None, "objective_independently_verified": True,
        "solution_sha256": fresh["solution_sha256"], "transcript_sha256": fresh["transcript_sha256"],
        "source_sha256": digest({key: fresh[key] for key in ("solution_sha256", "transcript_sha256")}),
        "audit_sha256": _hash(fresh), "synthesis_audit_sha256": _hash(fresh), "episode_audit_sha256": _hash(original),
        "strict_report_sha256": _strict_digest(strict), "text_sha256": digest(text),
        "raw_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "actor_sha256": fresh["actor_sha256"],
        "strict_sample_sha256": digest({"text": text, "finish_reason": finish_reason, "strict_reward": float(strict["reward"])}),
        "finish_reason": finish_reason, "lesson_semantics_verified": False, "all_reasoning_verified": False,
        "full_diagnosis": False, "module_causality_certified": False, "model_claims_used_for_routing": False}
    return score


def score_note(text, finish_reason, tree, audit, verifier, strict_report=None, *, actor,
               objective_version: str = VERSION, replay_relation_v1: bool = False,
               sample_id: str | None = None, observed_sample=None) -> dict:
    """Fresh strict and unit verification; observed policy1 samples remain diagnostic.

    The observed-sample caller must verify tokenizer decode against its raw text.
    """
    if objective_version not in (VERSION, ASSESSMENT_VERSION) or replay_relation_v1 is not False:
        raise ValueError("synthesis credit: explicit current objective and verifier required")
    from . import pair_action_notebook as action, paired_notebook, referenced_notebook, whole_reference_notebook

    generation = tree.notebook.get("rewarded_notebook", {}).get("notebook_generation_policy")
    from .whole_reference_pages import VERSION as pages_version

    if objective_version == ASSESSMENT_VERSION and generation != whole_reference_notebook.VERSION:
        raise ValueError("synthesis credit: assessment objective requires monolithic whole reference policy")
    if generation == pages_version:
        raise ValueError("synthesis credit: complete ordinal page bundles require score_page_bundles")
    if observed_sample is not None and generation != whole_reference_notebook.VERSION:
        raise ValueError("synthesis credit: observed sample requires explicit whole reference policy")
    fresh = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
    if type(audit) is not dict or _hash(audit) != _hash(fresh):
        raise ValueError("synthesis credit: supplied source audit differs from fresh verification")
    strict_grade = (whole_reference_notebook.grade_note if generation == whole_reference_notebook.VERSION
                    else action.grade_note if generation == action.VERSION else synthesis.grade_note)
    strict = strict_grade(text, finish_reason, tree, fresh, verifier, actor=actor, contract=synthesis.BOUNDED_VERSION)
    if strict_report is not None and (type(strict_report) is not dict or _strict_digest(strict_report) != _strict_digest(strict)):
        raise ValueError("synthesis credit: supplied strict report differs from fresh verification")
    samples = episode.solution_samples(tree)
    expanded = None
    reference_sample = None
    if generation == whole_reference_notebook.VERSION:
        if observed_sample is not None:
            from .schema import Sample, to_json

            if (type(observed_sample) is not Sample or tree.policy_version != 0
                    or observed_sample.policy_version != 1 or observed_sample.kind != "DIAG"
                    or observed_sample.problem_id != tree.problem_id
                    or any(sample.sample_id == observed_sample.sample_id for group in tree.groups for sample in group.samples)):
                raise ValueError("synthesis credit: observed policy1 raw sample requires original policy0 history")
            to_json(observed_sample)
            matches = [observed_sample]
        else:
            matches = [sample for group in tree.groups if group.kind == "DIAG" for sample in group.samples
                       if sample.sample_id == sample_id]
        if (type(sample_id) is not str or len(matches) != 1 or matches[0].text != text
                or matches[0].sample_id != sample_id or matches[0].finish_reason != finish_reason):
            raise ValueError("synthesis credit: whole reference requires the actual raw sample")
        reference_sample = matches[0]
        expanded = whole_reference_notebook.expand(text, finish_reason, samples)
        whole_reference_notebook.validate_audit_coverage(expanded["inventory"], fresh)
    public_index = referenced_notebook.public_endpoints(samples) if generation == referenced_notebook.VERSION else None
    paired_index = paired_notebook.public_pairs(samples) if generation == paired_notebook.VERSION else None
    action_target = None
    if generation == action.VERSION:
        if type(sample_id) is not str or not sample_id:
            raise ValueError("synthesis credit: pair action requires the actual sample ID")
        action_target = action.bound_target(tree, sample_id, text, finish_reason)
    note, facts, lessons = _json_units(text) if finish_reason == "stop" else (None, [], [])
    if expanded is not None:
        note, facts, lessons = _whole_reference_units(text, expanded)
    if paired_index is not None:
        facts, lessons = [], _paired_lessons(note, lessons)
    if action_target is not None:
        facts, lessons = [], []
    score = _score_submission(text, finish_reason, tree, fresh, strict, verifier, actor,
                              note, facts, lessons, public_index, paired_index, action_target)
    if public_index is not None:
        score.update(_reference_metadata(public_index, fresh["solution_sha256"], fresh["transcript_sha256"]))
    if paired_index is not None:
        score.update(_paired_metadata(paired_index, fresh["solution_sha256"], fresh["transcript_sha256"]))
    if action_target is not None and sample_id is not None:
        score.update(_pair_action_metadata(tree, sample_id, action_target,
                                          fresh["solution_sha256"], fresh["transcript_sha256"]))
    if expanded is not None:
        score.update(_whole_reference_metadata(samples, expanded, fresh["solution_sha256"], fresh["transcript_sha256"],
                                               tree.policy_version))
        score[_WHOLE_REFERENCE_SAMPLE] = _whole_reference_sample_binding(reference_sample)
        if objective_version == ASSESSMENT_VERSION:
            _score_assessments(score, text, finish_reason, expanded, samples, fresh)
    if observed_sample is not None:
        score["diagnostic_only"] = True
    return _seal(score)


def _score_pair_notes(tree, audit, verifier, strict_reports, *, actor) -> dict[str, dict]:
    """Reuse fresh evidence only within one already binding-checked pair-label batch."""
    from . import pair_action_notebook as action

    fresh = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
    if type(audit) is not dict or _hash(audit) != _hash(fresh):
        raise ValueError("synthesis credit: supplied source audit differs from fresh verification")
    original = episode.audit_episode(tree, actor, verifier, routing_rule=episode.STRICT_ROUTE_RULE)
    scored = {}
    for group in tree.groups:
        if group.kind != "DIAG":
            continue
        for sample in group.samples:
            text, finish = sample.text, sample.finish_reason
            strict = action._grade_note_audited(text, finish, tree, fresh, verifier,
                                                 contract=synthesis.BOUNDED_VERSION)
            if _strict_digest(strict_reports[sample.sample_id]) != _strict_digest(strict):
                raise ValueError("synthesis credit: supplied strict report differs from fresh verification")
            target = action._bound_target_checked(tree, sample.sample_id, text, finish)
            note = _json_units(text)[0] if finish == "stop" else None
            score = _score_submission(text, finish, tree, fresh, strict, verifier, actor,
                                      note, [], [], action_target=target, original=original)
            score.update(_pair_action_metadata(tree, sample.sample_id, target,
                                              fresh["solution_sha256"], fresh["transcript_sha256"]))
            scored[sample.sample_id] = _seal(score)
    return scored


def _raw_page(sample) -> dict:
    from .schema import to_json

    return {key: value for key, value in json.loads(to_json(sample)).items() if key not in ("result", "diag_reward", "reward")}


def _page_layout(tree, observed_groups=None):
    from . import whole_reference_pages as pages

    failure = "synthesis credit: invalid original page membership"
    note = tree.notebook.get("rewarded_notebook", {})
    originals = episode.solution_samples(tree)
    expected_plan = pages.plan(originals)
    calls = note.get("page_calls")
    groups = [group for group in tree.groups if group.kind == "DIAG"]
    if (note.get("notebook_generation_policy") != pages.VERSION or note.get("page_plan") != expected_plan
            or type(calls) is not list or len(calls) != len(expected_plan["pages"])
            or len(groups) != len(calls) or len({group.group_id for group in groups}) != len(groups)):
        raise ValueError(failure)
    selected = groups if observed_groups is None else observed_groups
    if (type(selected) is not list or len(selected) != len(groups)
            or observed_groups is not None and (tree.policy_version != 0
                or any(sample.policy_version != 1 for group in selected for sample in group.samples))):
        raise ValueError(failure)
    all_ids = {sample.sample_id for sample in originals}
    if observed_groups is not None:
        all_ids.update(sample.sample_id for group in groups for sample in group.samples)
        if (len({group.group_id for group in selected}) != len(selected)
                or {group.group_id for group in selected} & {group.group_id for group in groups}):
            raise ValueError(failure)
    for index, (call, group, observed) in enumerate(zip(calls, groups, selected, strict=True)):
        cap = expected_plan["pages"][index]["max_new_tokens"]
        if (type(call) is not dict or type(call.get("page_index")) is not int or call["page_index"] != index
                or call.get("group_id") != group.group_id or type(call.get("messages")) is not list
                or not isinstance(call.get("images"), (dict, list)) or call["images"] != calls[0]["images"]
                or call.get("sample_ids") != [sample.sample_id for sample in group.samples]
                or call.get("seeds") != [sample.seed for sample in group.samples]
                or call.get("max_new_tokens", cap) != cap
                or group.kind != "DIAG" or observed.kind != "DIAG"
                or group.problem_id != tree.problem_id or observed.problem_id != tree.problem_id
                or len(group.samples) != GROUP_SIZE or len(observed.samples) != GROUP_SIZE):
            raise ValueError(failure)
        for ordinal, (saved, sample) in enumerate(zip(group.samples, observed.samples, strict=True)):
            if (sample.sample_id in all_ids or sample.group_id != observed.group_id
                    or sample.problem_id != tree.problem_id or sample.kind != "DIAG" or sample.turn != 1
                    or sample.parent_sample_id != saved.parent_sample_id or sample.seed != call["seeds"][ordinal]
                    or sample.prompt_token_ids != group.samples[0].prompt_token_ids
                    or sample.prompt_token_ids != saved.prompt_token_ids
                    or call.get("prompt_sha256", _hash(sample.prompt_token_ids)) != _hash(sample.prompt_token_ids)
                    or call.get("prompt_token_ids", sample.prompt_token_ids) != sample.prompt_token_ids
                    or not 0 < len(sample.completion_token_ids) <= cap
                    or len(sample.sampler_logprobs) != len(sample.completion_token_ids)
                    or any(not math.isfinite(value) for value in sample.sampler_logprobs)
                    or observed_groups is None and sample.policy_version != tree.policy_version):
                raise ValueError(failure)
            all_ids.add(sample.sample_id)
    return originals, expected_plan, calls, selected


def _project_page_score(common, sample, assembled, page_index, binding):
    facts = assembled["provenance"]["facts"]
    decisions = assembled["provenance"]["decisions"]
    owned_facts = {key: unit for key, unit in facts.items() if unit["sample_id"] == sample.sample_id}
    owned_relations = {key: next(unit for unit in decisions.values()
        if unit["assembled_span"] == span and unit["sample_id"] == sample.sample_id)
        for key, span in common["relationship_spans"].items()
        if any(unit["assembled_span"] == span and unit["sample_id"] == sample.sample_id
               for unit in decisions.values())}
    fact_rewards = {key: reward for key, reward in common["fact_rewards"].items() if key in owned_facts}
    relationships = {key: reward if key in owned_relations else 0
                     for key, reward in common["relationship_rewards"].items()}
    atomic = FACT_WEIGHT * (sum(fact_rewards.values()) / common["required_fact_count"]
                           if common["required_fact_count"] else 0.)
    atomic += RELATIONSHIP_WEIGHT * (sum(relationships.values()) / len(common["original_issue_ids"])
                                    if common["original_issue_ids"] else 0.)
    row = {**common, "notebook_generation_policy": binding["generation_contract"]["version"],
        "page_binding": binding, "page_binding_sha256": digest(binding), "page_index": page_index,
        "bundle_index": binding["bundle_index"], "atomic_reward": atomic,
        "effective_reward": common["strict_reward"] + atomic, "fact_rewards": fact_rewards,
        "relationship_rewards": relationships,
        "fact_spans": {key: owned_facts[key]["source_span"] for key in fact_rewards},
        "relationship_spans": {key: unit["source_span"] for key, unit in owned_relations.items()},
        "selected_lesson_indices": {key: value for key, value in common["selected_lesson_indices"].items()
                                    if key in owned_relations},
        "relationship_relations": {key: value for key, value in common["relationship_relations"].items()
                                   if key in owned_relations},
        "relationship_later_answer_correct": {key: value for key, value in common["relationship_later_answer_correct"].items()
                                             if key in owned_relations},
        "text_sha256": digest(sample.text), "raw_text_sha256": hashlib.sha256(sample.text.encode()).hexdigest(),
        "finish_reason": sample.finish_reason,
        "strict_sample_sha256": digest({"text": sample.text, "finish_reason": sample.finish_reason,
                                       "strict_reward": float(common["strict_reward"])})}
    return _seal(row)


def score_page_bundles(tree, audit, verifier, *, actor, observed_groups=None) -> dict[str, dict]:
    """Regrade four complete ordinal bundles; credit only their actual owned page literals."""
    from . import whole_reference_notebook as reference, whole_reference_pages as pages

    episode.validate_update_binding(tree)
    originals, page_plan, calls, groups = _page_layout(tree, observed_groups)
    fresh = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
    if type(audit) is not dict or _hash(audit) != _hash(fresh):
        raise ValueError("synthesis credit: supplied source audit differs from fresh verification")
    result = {}
    for ordinal in range(GROUP_SIZE):
        members = [group.samples[ordinal] for group in groups]
        assembled = pages.assemble(members, originals)
        text = assembled["reference_text"] if assembled["status"] == pages.CAPTURED else "{}"
        finish = "length" if assembled["status"] == pages.TRUNCATED else "stop"
        strict = reference.grade_note(text, finish, tree, fresh, verifier, actor=actor,
                                      contract=synthesis.BOUNDED_VERSION)
        expanded = assembled["expansion"]
        units = _whole_reference_units(text, expanded) if assembled["status"] == pages.CAPTURED else (None, [], [])
        common = _score_submission(text, finish, tree, fresh, strict, verifier, actor, *units)
        binding = {"generation_contract": pages.contract(), "page_plan": page_plan,
            "page_calls": copy.deepcopy(calls), "bundle_index": ordinal, "source_policy_version": tree.policy_version,
            "original_samples": [{**{key: getattr(sample, key) for key in
                ("text", "sample_id", "parent_sample_id", "finish_reason")}, "problem_id": sample.problem_id}
                for sample in originals], "raw_pages": [_raw_page(sample) for sample in members],
            "assembly_sha256": _hash(assembled), "bundle_score": common}
        for index, sample in enumerate(members):
            row = _project_page_score(common, sample, assembled, index, binding)
            if observed_groups is not None:
                row = _seal({**{key: value for key, value in row.items() if key != "score_sha256"},
                             "diagnostic_only": True, "training_allowed": False})
            result[sample.sample_id] = row
    return result


def _validate_page_score(row, sample) -> None:
    from . import whole_reference_pages as pages
    from .schema import Sample

    failure = "synthesis credit: invalid bound original page score"
    binding = row.get("page_binding")
    if (type(binding) is not dict or set(binding) != {"generation_contract", "page_plan", "page_calls", "bundle_index",
                "original_samples", "raw_pages", "assembly_sha256", "bundle_score", "source_policy_version"}
            or binding["generation_contract"] != pages.contract()
            or row.get("notebook_generation_policy") != pages.VERSION
            or row.get("page_binding_sha256") != digest(binding)
            or type(binding["source_policy_version"]) is not int or binding["source_policy_version"] != sample.policy_version
            or type(row.get("page_index")) is not int or type(binding["bundle_index"]) is not int
            or not 0 <= binding["bundle_index"] < GROUP_SIZE
            or row.get("bundle_index") != binding["bundle_index"]
            or any(key in row for key in (*_WHOLE_REFERENCE_FIELDS[1:], _WHOLE_REFERENCE_SAMPLE, *_EXPANSION_FIELDS,
                                         *_REFERENCE_FIELDS[1:], *_PAIRED_FIELDS[1:], *_PAIR_ACTION_FIELDS))):
        raise ValueError(failure)
    try:
        originals = [SimpleNamespace(**source) for source in binding["original_samples"]]
        # Reconstruct only saved physical actions, never an assembled or replacement action.
        members = [Sample(**raw) for raw in binding["raw_pages"]]
        assembled = pages.assemble(members, originals)
        index, ordinal = row["page_index"], binding["bundle_index"]
        calls = binding["page_calls"]
        if (binding["page_plan"] != pages.plan(originals) or not 0 <= index < len(members)
                or len(calls) != len(members) or _raw_page(sample) != binding["raw_pages"][index]
                or any(type(call["page_index"]) is not int or call["page_index"] != page
                    or raw.group_id != call["group_id"] or len(call["sample_ids"]) != GROUP_SIZE
                    or len(call["seeds"]) != GROUP_SIZE or call["sample_ids"][ordinal] != raw.sample_id
                    or call["seeds"][ordinal] != raw.seed
                    or call.get("prompt_sha256", _hash(raw.prompt_token_ids)) != _hash(raw.prompt_token_ids)
                    or call.get("prompt_token_ids", raw.prompt_token_ids) != raw.prompt_token_ids
                    or not 0 < len(raw.completion_token_ids) <= binding["page_plan"]["pages"][page]["max_new_tokens"]
                    for page, (raw, call) in enumerate(zip(members, calls, strict=True)))
                or binding["assembly_sha256"] != _hash(assembled)):
            raise ValueError(failure)
        common = binding["bundle_score"]
        for key in ("required_fact_count", "original_issue_ids", "relationship_credit_types", "solution_sha256",
                    "synthesis_audit_sha256", "episode_audit_sha256", "actor_sha256", "strict_reward",
                    "strict_report_sha256"):
            if common[key] != row[key]:
                raise ValueError(failure)
        if assembled["status"] == pages.CAPTURED:
            note, facts, lessons = _whole_reference_units(assembled["reference_text"], assembled["expansion"])
        else:
            note, facts, lessons = None, [], []
        if (type(common["fact_rewards"]) is not dict or len(common["fact_rewards"]) > row["required_fact_count"]
                or type(common["relationship_rewards"]) is not dict
                or set(common["relationship_rewards"]) != set(row["original_issue_ids"])
                or any(type(value) is not int or value not in (0, 1)
                       for value in [*common["fact_rewards"].values(), *common["relationship_rewards"].values()])):
            raise ValueError(failure)
        _validate_submission_fields(common, assembled["reference_text"] or "{}", "stop", note, facts, lessons,
                                    row["original_issue_ids"], row["relationship_credit_types"])
        expected = _project_page_score(common, sample, assembled, index, binding)
        if row != expected:
            raise ValueError(failure)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(failure) from exc


def _validate_submission_fields(row, text, finish_reason, note, fact_units, lessons, originals, credit_types,
                                public_index=None, paired_index=None, action_target=None) -> None:
    failure = "synthesis credit: invalid bound unit score"
    facts, relationships, required = row["fact_rewards"], row["relationship_rewards"], row["required_fact_count"]
    candidates = _fact_candidates(note, fact_units)
    selected = _selected(lessons, originals, public_index, paired_index)
    spans = {key: [lessons[index][1], lessons[index][2]] for key, index in selected.items()
             if _lesson_shape(_credit_lesson(lessons[index][0], public_index, paired_index))}
    relations = {key: lessons[index][0].get("relation") if type(lessons[index][0].get("relation")) is str else None
                 for key, index in selected.items()}
    if action_target is not None:
        selected, spans, relations = _pair_action_units(text, finish_reason, action_target, originals)
    if (row.get("json_complete") is not (note is not None) or row.get("selected_lesson_indices") != selected
            or row.get("relationship_spans") != spans or row.get("relationship_relations") != relations
            or any(key not in candidates for key in facts)
            or row.get("fact_spans") != {key: candidates[key][1] for key in facts}
            or any(reward and (key not in spans or (relations[key] != "FINALIZED"
                   if credit_types[key] == OUTPUT_FINALIZATION else relations[key] not in ("CORRECTED", "PERSISTED")))
                   for key, reward in relationships.items())):
        raise ValueError(failure)
    answer_correct = row.get("relationship_later_answer_correct")
    if (type(answer_correct) is not dict or set(answer_correct) != {key for key, reward in relationships.items()
            if reward and credit_types[key] == OUTPUT_FINALIZATION}
            or any(type(value) is not bool for value in answer_correct.values())):
        raise ValueError(failure)
    atomic = FACT_WEIGHT * (sum(facts.values()) / required if required else 0.)
    atomic += RELATIONSHIP_WEIGHT * (sum(relationships.values()) / len(originals) if originals else 0.)
    if row["objective_version"] == ASSESSMENT_VERSION:
        count = row["native_assessment_count"]
        atomic += ASSESSMENT_WEIGHT * (sum(row["assessment_rewards"].values()) / count if count else 0.)
    if (any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) for key in ("atomic_reward", "effective_reward"))
            or row["atomic_reward"] != atomic or row["effective_reward"] != row["strict_reward"] + atomic):
        raise ValueError(failure)


def _validate_assessments(row: dict, sample, expanded: dict) -> None:
    failure = "synthesis credit: invalid bound assessment score"
    inventory = expanded["inventory"]
    refs = [ref for ref in inventory["assessment_refs"]
            if inventory["references"][ref]["kind"] in ("OBSERVATION", "TRACE")]
    expected, rewards = row.get("assessment_expected_states"), row.get("assessment_rewards")
    complete, rows, states, spans = _assessment_units(sample.text, sample.finish_reason, inventory)
    if (type(row.get("native_assessment_count")) is not int or row["native_assessment_count"] != len(refs)
            or row.get("assessment_refs") != refs or type(expected) is not dict or set(expected) != set(refs)
            or type(rewards) is not dict or set(rewards) != set(refs)
            or any(type(value) is not int or value not in (0, 1) for value in rewards.values())
            or row.get("assessment_rows_complete") is not complete or row.get("assessment_states") != states
            or row.get("assessment_spans") != spans or type(row["assessment_spans"]) is not dict
            or any(type(number) is not int for literals in row["assessment_spans"].values()
                   for literal in literals for number in literal)):
        raise ValueError(failure)
    for ref in refs:
        state, source = expected[ref], inventory["references"][ref]
        if (type(state) is not dict or set(state) != {"flags", "blocked_by"}
                or type(state["flags"]) is not str or state["flags"] not in ("N", "I", "U", "IU")
                or ("I" in state["flags"]) != (issue_id(source["attempt"], source) in row["original_issue_ids"])
                or type(state["blocked_by"]) is not list or any(type(dep) is not str for dep in state["blocked_by"])
                or len(state["blocked_by"]) != len(set(state["blocked_by"]))
                or "U" not in state["flags"] and state["blocked_by"]):
            raise ValueError(failure)
        for dep in state["blocked_by"]:
            target = inventory["references"].get(dep)
            if (target is None or dep not in refs[:refs.index(ref)] or source["kind"] != "TRACE"
                    or target["kind"] != "TRACE" or target["attempt"] != source["attempt"]):
                raise ValueError(failure)
            if source["rule"].startswith("CIRCUIT_") or target["rule"].startswith("CIRCUIT_"):
                from .circuit_trace import RULES

                if source["rule"] not in RULES or target["rule"] not in RULES or "U" not in expected[dep]["flags"]:
                    raise ValueError(failure)
            elif "I" not in expected[dep]["flags"]:
                raise ValueError(failure)
        if rewards[ref] and (not complete or not _assessment_decision(ref, rows, inventory, expected)[0]):
            raise ValueError(failure)
    origins = {issue_id(source["attempt"], source): ref for ref, source in inventory["references"].items()
               if source["kind"] != "ABSENCE"}
    if any(reward and not _assessment_decision(origins[origin], rows, inventory, expected)[0]
           for origin, reward in row["relationship_rewards"].items()):
        raise ValueError(failure)


def validate_score(row: dict, sample) -> None:
    """Check receipt math, raw spans and sample binding; callers must also freshly regrade semantics."""
    failure = "synthesis credit: invalid bound unit score"
    if (type(row) is not dict or row.get("objective_version") not in (VERSION, ASSESSMENT_VERSION)
            or any(key in row for key in ("diagnostic_only", "program_rendered_only", "diagnostic_atomic_reward"))
            or row.get("objective_contract_sha256") != digest(contract(row["objective_version"]))
            or row.get("strict_reward_version") != synthesis.BOUNDED_VERSION
            or row.get("objective_independently_verified") is not True
            or row.get("score_sha256") != digest({key: value for key, value in row.items() if key != "score_sha256"})
            or any(row.get(key) is not False for key in ("lesson_semantics_verified", "all_reasoning_verified",
                "full_diagnosis", "module_causality_certified", "model_claims_used_for_routing"))):
        raise ValueError(failure)
    assessment = row["objective_version"] == ASSESSMENT_VERSION
    if not assessment and any(key in row for key in _ASSESSMENT_FIELDS):
        raise ValueError(failure)
    for key in ("solution_sha256", "transcript_sha256", "source_sha256", "audit_sha256", "synthesis_audit_sha256",
                "episode_audit_sha256", "strict_report_sha256", "text_sha256", "raw_text_sha256", "actor_sha256"):
        if type(row.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None:
            raise ValueError(failure)
    if (row["audit_sha256"] != row["synthesis_audit_sha256"]
            or row["source_sha256"] != digest({key: row[key] for key in ("solution_sha256", "transcript_sha256")})
            or type(row.get("strict_reward")) not in (int, float) or row["strict_reward"] not in (0, 1)
            or type(sample.reward) not in (int, float) or sample.reward not in (0, 1)
            or row["strict_reward"] != sample.reward or row.get("finish_reason") != sample.finish_reason
            or row["text_sha256"] != digest(sample.text)
            or row["raw_text_sha256"] != hashlib.sha256(sample.text.encode()).hexdigest()
            or row.get("strict_sample_sha256") != digest({"text": sample.text,
                "finish_reason": sample.finish_reason, "strict_reward": float(sample.reward)})):
        raise ValueError(failure)
    required, originals = row.get("required_fact_count"), row.get("original_issue_ids")
    if (type(required) is not int or required < 0 or type(originals) is not list
            or any(type(key) is not str for key in originals) or len(originals) != len(set(originals))):
        raise ValueError(failure)
    credit_types = {}
    for key in originals:
        try:
            value = json.loads(key)
            valid = (type(value) is list and len(value) == 4 and type(value[0]) is int and value[0] > 0
                and (value[1] in ("TRACE", "OBSERVATION")
                     or value[1:] == ["OUTPUT", "FINALIZATION", "final_answer"])
                and all(type(item) is str for item in value[1:])
                and key == json.dumps(value, separators=(",", ":")))
            if valid:
                credit_types[key] = OUTPUT_FINALIZATION if value[1] == "OUTPUT" else NATIVE_ERROR_RELATIONSHIP
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError(failure)
    if row.get("relationship_credit_types") != credit_types:
        raise ValueError(failure)
    from .whole_reference_pages import VERSION as pages_version

    from .whole_reference_notebook import VERSION as whole_reference_version

    if assessment and row.get("notebook_generation_policy") != whole_reference_version:
        raise ValueError("synthesis credit: assessment objective requires monolithic whole reference policy")
    if row.get("notebook_generation_policy") == pages_version or "page_binding" in row:
        _validate_page_score(row, sample)
        return
    from .pair_action_notebook import VERSION as action_version
    from .paired_notebook import VERSION as paired_version

    whole_reference = (row.get("notebook_generation_policy") == whole_reference_version
                       or any(key in row for key in (*_WHOLE_REFERENCE_FIELDS[1:], _WHOLE_REFERENCE_SAMPLE, *_EXPANSION_FIELDS)))
    expanded = _validate_whole_reference_binding(row, sample) if whole_reference else None
    pair_action = row.get("notebook_generation_policy") == action_version or any(key in row for key in _PAIR_ACTION_FIELDS)
    action_target = _validate_pair_action_binding(row, sample) if pair_action else None
    paired = row.get("notebook_generation_policy") == paired_version or any(key in row for key in _PAIRED_FIELDS[1:])
    paired_index = _validate_paired_index(row) if paired else None
    public_index = None if paired or pair_action or whole_reference else _validate_reference_index(row)
    facts, relationships = row.get("fact_rewards"), row.get("relationship_rewards")
    if (type(facts) is not dict or len(facts) > required or any(type(key) is not str or key != slot_id(key) for key in facts)
            or type(relationships) is not dict or set(relationships) != set(originals)
            or any(type(value) is not int or value not in (0, 1) for value in [*facts.values(), *relationships.values()])):
        raise ValueError(failure)
    note, fact_units, lessons = _json_units(sample.text) if sample.finish_reason == "stop" else (None, [], [])
    if expanded is not None:
        note, fact_units, lessons = _whole_reference_units(sample.text, expanded)
    if paired_index is not None:
        fact_units, lessons = [], _paired_lessons(note, lessons)
    if action_target is not None:
        fact_units, lessons = [], []
    if assessment:
        if expanded is None:
            raise ValueError(failure)
        _validate_assessments(row, sample, expanded)
    _validate_submission_fields(row, sample.text, sample.finish_reason, note, fact_units, lessons,
                                originals, credit_types, public_index, paired_index, action_target)


def _group(samples, scores: dict[str, dict]) -> list[dict]:
    if (len(samples) != GROUP_SIZE or len({sample.sample_id for sample in samples}) != GROUP_SIZE
            or len({sample.group_id for sample in samples}) != 1 or any(sample.kind != "DIAG" for sample in samples)):
        raise ValueError("synthesis credit: original complete unique n4 DIAG group required")
    rows = [scores[sample.sample_id] for sample in samples]
    for sample, row in zip(samples, rows, strict=True):
        validate_score(row, sample)
    for key in ("objective_version", "required_fact_count", "original_issue_ids", "relationship_credit_types", "solution_sha256", "synthesis_audit_sha256",
                "episode_audit_sha256", "actor_sha256"):
        if any(row[key] != rows[0][key] for row in rows[1:]):
            raise ValueError("synthesis credit: complete group source or fixed denominators differ")
    if rows[0]["objective_version"] == ASSESSMENT_VERSION:
        for key in ("assessment_refs", "assessment_expected_states", "native_assessment_count"):
            if any(row[key] != rows[0][key] for row in rows[1:]):
                raise ValueError("synthesis credit: complete group assessment states or fixed denominator differ")
    if any(any(row.get(key) != rows[0].get(key) for row in rows[1:])
           for key in (*_REFERENCE_FIELDS, *_PAIRED_FIELDS[1:], *_WHOLE_REFERENCE_FIELDS[1:])):
        raise ValueError("synthesis credit: complete group public reference index differs")
    if any("page_binding" in row for row in rows):
        common = [{key: value for key, value in row["page_binding"].items()
                   if key not in ("bundle_index", "raw_pages", "assembly_sha256", "bundle_score")} for row in rows]
        if (any(value != common[0] for value in common[1:])
                or [row["bundle_index"] for row in rows] != list(range(GROUP_SIZE))
                or len({row["page_index"] for row in rows}) != 1):
            raise ValueError("synthesis credit: complete group page membership differs")
    if any("pair_action_binding" in row for row in rows):
        common = [{key: value for key, value in row["pair_action_binding"].items()
                   if key not in ("sample_id", "completion_token_ids_sha256")} for row in rows]
        if any(value != common[0] for value in common[1:]):
            raise ValueError("synthesis credit: complete group pair action target differs")
    return rows


def has_signal(samples, scores: dict[str, dict]) -> bool:
    rows = _group(samples, scores)
    facts = {key for row in rows for key in row["fact_rewards"]}
    return (len({row["strict_reward"] for row in rows}) > 1
            or any(len({row["fact_rewards"].get(key, 0) for row in rows}) > 1 for key in facts)
            or any(len({row["relationship_rewards"][key] for row in rows}) > 1 for key in rows[0]["original_issue_ids"])
            or rows[0]["objective_version"] == ASSESSMENT_VERSION and any(
                len({row["assessment_rewards"][ref] for row in rows}) > 1 for ref in rows[0]["assessment_refs"]))


def group_credit(samples, scores: dict[str, dict], tokenizer) -> dict[str, dict]:
    """Center all original samples before MIS; map only original, closed submitted units."""
    from .fact_credit import token_spans

    rows = _group(samples, scores)
    facts = sorted({key for row in rows for key in row["fact_rewards"]})
    means = {key: sum(row["fact_rewards"].get(key, 0) for row in rows) / GROUP_SIZE for key in facts}
    relationship_means = {key: sum(row["relationship_rewards"][key] for row in rows) / GROUP_SIZE
                          for key in rows[0]["original_issue_ids"]}
    assessment = rows[0]["objective_version"] == ASSESSMENT_VERSION
    assessment_means = {ref: sum(row["assessment_rewards"][ref] for row in rows) / GROUP_SIZE
                        for ref in rows[0]["assessment_refs"]} if assessment else {}
    strict_mean = sum(row["strict_reward"] for row in rows) / GROUP_SIZE
    full_group = [{"sample_id": sample.sample_id, "token_ids_sha256": digest(sample.completion_token_ids),
                   "text_sha256": digest(sample.text), "score_sha256": digest(row)}
                  for sample, row in zip(samples, rows, strict=True)]
    output = {}
    for sample, row in zip(samples, rows, strict=True):
        spans = token_spans(tokenizer, sample.completion_token_ids, sample.text)
        fact_tokens, relationship_tokens = ([0.] * len(sample.completion_token_ids) for _ in range(2))
        entries: dict[str, list] = {"facts": [], "relationships": []}
        for component, rewards, locations, averages, count, weight, target in (
            ("facts", row["fact_rewards"], row["fact_spans"], means, row["required_fact_count"], FACT_WEIGHT, fact_tokens),
            ("relationships", row["relationship_rewards"], row["relationship_spans"], relationship_means,
             len(row["original_issue_ids"]), RELATIONSHIP_WEIGHT, relationship_tokens)):
            keys = row["original_issue_ids"] if component == "relationships" else facts
            for key in keys:
                if key not in locations:
                    continue
                literal = locations[key]
                start, end = literal
                indices = [] if spans is None else [index for index, (a, b) in enumerate(spans) if start <= a < b <= end]
                advantage = rewards[key] - averages[key]
                coefficient = weight * advantage / count if count else 0.
                if component == "relationships":
                    coefficient = coefficient / len(indices) if indices else 0.
                for index in indices:
                    target[index] += coefficient
                entry = {"slot" if component == "facts" else "issue_id": key, "reward": rewards[key],
                    "group_mean": averages[key], "advantage": advantage, "coefficient": coefficient,
                    "literal_span": literal, "token_indices": indices}
                if component == "relationships":
                    entry["relation"] = row["relationship_relations"][key]
                    entry["credit_type"] = row["relationship_credit_types"][key]
                    if key in row["relationship_later_answer_correct"]:
                        entry["later_answer_correct"] = row["relationship_later_answer_correct"][key]
                entries[component].append(entry)
        assessment_tokens = [0.] * len(sample.completion_token_ids)
        if assessment:
            entries["assessments"] = []
            for ref in row["assessment_refs"]:
                if ref not in row["assessment_spans"]:
                    continue
                literals = row["assessment_spans"][ref]
                indices = [] if spans is None else [index for index, (a, b) in enumerate(spans)
                    if any(start <= a < b <= end for start, end in literals)]
                advantage = row["assessment_rewards"][ref] - assessment_means[ref]
                coefficient = (ASSESSMENT_WEIGHT * advantage / row["native_assessment_count"] / len(indices)
                               if row["native_assessment_count"] and indices else 0.)
                coefficient *= ASSESSMENT_NORMALIZATION_CAP
                for index in indices:
                    assessment_tokens[index] += coefficient
                entries["assessments"].append({"ref": ref, "reward": row["assessment_rewards"][ref],
                    "group_mean": assessment_means[ref], "advantage": advantage, "coefficient": coefficient,
                    "literal_spans": literals, "token_indices": indices,
                    "expected_flags": row["assessment_expected_states"][ref]["flags"]})
        output[sample.sample_id] = {"strict_advantage": row["strict_reward"] - strict_mean,
            "fact_token_advantages": fact_tokens, "relationship_token_advantages": relationship_tokens,
            "partial_token_advantages": [a+b for a, b in zip(fact_tokens, relationship_tokens, strict=True)],
            **entries, "mapping_status": "UNSTABLE_TOKEN_PREFIX" if spans is None else
                "MAPPED" if any(entry["token_indices"] for values in entries.values() for entry in values) else "NO_MAPPED_UNITS",
            "token_ids_sha256": digest(sample.completion_token_ids), "text_sha256": digest(sample.text),
            "score_sha256": digest(row), "full_group_size": GROUP_SIZE, "full_group_sha256": digest(full_group),
            "full_group": full_group, "required_fact_count": row["required_fact_count"],
            "original_issue_ids": row["original_issue_ids"], "relationship_credit_types": row["relationship_credit_types"],
            "relationship_group_means": relationship_means,
            "fact_group_means": means}
        if assessment:
            output[sample.sample_id]["assessment_token_advantages"] = assessment_tokens
            output[sample.sample_id]["partial_token_advantages"] = [a + b for a, b in zip(
                output[sample.sample_id]["partial_token_advantages"], assessment_tokens, strict=True)]
    return output
