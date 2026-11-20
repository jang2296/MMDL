"""Functional evidence attribution, never a diagnosis of an internal module."""
from __future__ import annotations

import hashlib
import math

from .observations import native_absent_relations, required_observations_verified, slot_id, verified_wrong_slots
from .schema import Cause, DiagTarget, Route, VerifierItem, VerifierResult, to_json


def attempt_cause(result: VerifierResult, item: VerifierItem) -> Cause | None:
    to_json(result)
    to_json(item)
    if result.status == "CORRECT":
        return None
    if result.status in ("INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
        return "FINALIZATION"
    slots = result.slots
    # Empty required sets cannot certify complete perception (native table/chem tasks).
    if result.status != "INCORRECT" or slots is None or not item.required_slots:
        return "UNKNOWN"
    _check_slots(result, item)
    if slots.false_relations:
        return "VISUAL_RELATION"
    parsed_required = slots.required_correct + len(slots.wrong_required)
    if parsed_required * 2 < slots.required:
        return "UNKNOWN"
    if slots.wrong_required:
        kinds = {slot_id(k): v for k, v in item.slot_types.items()}
        return "VISUAL_RELATION" if any(kinds[k] == "relation" for k in slots.wrong_required) else "VISUAL_MISREAD"
    return "NON_VISUAL" if required_observations_verified(slots) else "UNKNOWN"


def _check_slots(result: VerifierResult, item: VerifierItem) -> None:
    slots = result.slots
    if slots is None:
        return
    required = {slot_id(k) for k in item.required_slots}
    if (slots.required != len(required) or slots.all_total != len(item.slots)
            or not set(slots.wrong_required) <= required
            or len(set(slots.wrong_required)) != len(slots.wrong_required)):
        raise ValueError("attribution: report does not match the registered schema")
    if slots.false_relations != native_absent_relations(slots.false_relations, item):
        raise ValueError("attribution: false relations lack a complete native-world proof")
    if set(slots.unverified_relations) & {slot_id(k) for k in item.slots}:
        raise ValueError("attribution: unverified relations overlap registered observations")


def _probe_cause(probe: dict, item: VerifierItem) -> str:
    """Counts from separate diagnostic calls; raw gold facts are not accepted here."""
    if not isinstance(probe, dict) or set(probe) - {"oracle", "self_perception"}:
        raise ValueError("attribution: unknown probe evidence fields")
    oracle, perception = probe.get("oracle"), probe.get("self_perception")
    evidence = []
    if oracle is not None:
        if not isinstance(oracle, dict) or set(oracle) != {"correct", "n", "baseline_correct", "baseline_n"}:
            raise ValueError("attribution: invalid oracle count schema")
        for count, total in (("correct", "n"), ("baseline_correct", "baseline_n")):
            if (type(oracle[count]) is not int or type(oracle[total]) is not int
                    or oracle[total] < 4 or not 0 <= oracle[count] <= oracle[total]):
                raise ValueError("attribution: invalid oracle counts")
        delta = oracle["correct"] / oracle["n"] - oracle["baseline_correct"] / oracle["baseline_n"]
        evidence.append("VISUAL" if delta >= 0.5 else "NON_VISUAL" if delta <= 0 else "UNKNOWN")
    if perception is not None:
        if (item.answer_type == "mcq" or not isinstance(perception, dict)
                or set(perception) != {"correct", "n"} or type(perception["n"]) is not int
                or perception["n"] != 4 or type(perception["correct"]) is not int
                or not 0 <= perception["correct"] <= 4):
            raise ValueError("attribution: self-perception requires an open task and exactly four trials")
        evidence.append("NON_VISUAL" if perception["correct"] >= 3 else "VISUAL"
                        if perception["correct"] == 0 and oracle is not None
                        and oracle["correct"] / oracle["n"] >= 0.75 else "UNKNOWN")
    known = set(evidence) - {"UNKNOWN"}
    return known.pop() if len(known) == 1 else "UNKNOWN"


def problem_route(results: list[VerifierResult], items: VerifierItem, rule: str,
                  probe: dict | None = None) -> tuple[Route, dict]:
    to_json(items)
    if rule not in ("verified-any-visual-v1", "verified-any-visual-v1+probe"):
        raise ValueError("attribution: unsupported routing rule")
    if probe is not None and rule != "verified-any-visual-v1+probe":
        raise ValueError("attribution: probes require a separate explicit protocol")
    evidence, false_relations, comparable = [], [], []
    for index, result in enumerate(results):
        to_json(result)
        _check_slots(result, items)
        if result.status in ("CORRECT", "INCORRECT") and result.slots and result.slots.false_relations:
            false_relations.append({"result_index": index, "wrong_slots": list(result.slots.false_relations)})
        eligible = (result.status in ("CORRECT", "INCORRECT") and result.slots is not None
                    and result.slots.required > 0 and 2 * (result.slots.required_correct
                    + len(result.slots.wrong_required)) >= result.slots.required)
        if eligible and result.slots is not None:
            if result.slots.wrong_required:
                evidence.append({"result_index": index, "wrong_slots": list(result.slots.wrong_required)})
            if required_observations_verified(result.slots):
                comparable.append(index)
    observed = "VISUAL" if evidence or false_relations else "NON_VISUAL" if comparable else "UNKNOWN"
    extra = _probe_cause(probe, items) if probe is not None else "UNKNOWN"
    route: Route = "VISION_ON" if observed == "VISUAL" or (observed == "UNKNOWN" and extra == "VISUAL") else "VISION_OFF"
    return route, {"rule": rule, "observed_evidence": observed, "wrong_required": evidence,
                   "false_relations": false_relations,
                   "complete_required_indices": comparable, "probe_evidence": extra,
                   "probe_used": observed == "UNKNOWN" and extra != "UNKNOWN",
                   "module_causality_certified": False}


def diagnosis_target(first_attempt: VerifierResult, item: VerifierItem) -> DiagTarget | None:
    cause = attempt_cause(first_attempt, item)
    if cause in (None, "UNKNOWN"):
        return None
    if cause in ("VISUAL_MISREAD", "VISUAL_RELATION"):
        assert first_attempt.slots is not None
        return DiagTarget("VISUAL", verified_wrong_slots(first_attempt.slots))
    return DiagTarget("FINALIZATION" if cause == "FINALIZATION" else "NON_VISUAL", ())


def random_route(rate: float, key: str) -> Route:
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1 or not isinstance(key, str):
        raise ValueError("attribution: invalid random-route rate/key")
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return "VISION_ON" if value < rate else "VISION_OFF"
