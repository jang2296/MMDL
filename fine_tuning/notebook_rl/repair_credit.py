"""Fresh bounded native relation audits; no token credit or optimizer wiring."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

from .repair_contract import (
    FINAL_VERSION, GRAMMAR_ERRORS, ORIGIN_VERSION, VERSION, FinalRelation, FinalRepairClaim, JsonValue,
    PublicOriginInventory, RawSpan, RepairClaim, RepairContractError, RetrySections,
    parse_error_check, parse_final_note, project_solution, public_attempt_outcome, public_origin_inventory,
)
from .schema import ActorItem, Group, Sample, VerifierResult, from_json, to_json

AUDIT_VERSION = "repair-online-native-audit-v1"
FINAL_AUDIT_VERSION = "repair-final-native-audit-v1"


@dataclass(frozen=True)
class RepairWitness:
    origin_ref: str
    later_ref: str | None
    semantic_identity: JsonValue
    relation: FinalRelation | None
    claim_verdict: Literal["VERIFIED", "REFUTED", "UNVERIFIABLE", "STRUCTURAL_INVALID"]
    reason: str
    evidence_hashes: tuple[str, ...]


@dataclass(frozen=True)
class UnitScore:
    origin_ref: str
    value: Literal[-1, 0, 1]
    reason: str
    raw_spans: tuple[RawSpan, ...]
    verifier_version: str


@dataclass(frozen=True)
class RepairAudit:
    parent_id: str
    child_id: str
    sections: RetrySections
    origins_digest: str
    witnesses: tuple[RepairWitness, ...]
    unit_scores: tuple[UnitScore, ...]
    source_bindings: dict[str, Any]


@dataclass(frozen=True)
class FinalRepairAudit:
    note_id: str
    origins_digest: str
    metadata_valid: bool
    metadata_errors: tuple[str, ...]
    accurate_submitted_note: bool
    witnesses: tuple[RepairWitness, ...]
    proposed_unit_scores: tuple[UnitScore, ...]
    unit_scores: tuple[UnitScore, ...]
    metrics: dict[str, Any]
    source_bindings: dict[str, Any]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def operation_patch(trace: dict, rule: str, locus: str) -> dict | None:
    """Project one exact authored native piece; this supplies no correctness verdict."""
    from . import notebook_synthesis as synthesis, reasoning_trace

    if type(trace) is not dict or type(rule) is not str or type(locus) is not str:
        return None
    kind = trace.get("kind")
    try:
        if kind == "circuit_ladder":
            from .circuit_trace import schema
            valid = schema(trace)
        else:
            valid = (kind in ("shortest", "bfs", "tree", "dfa")
                     and (kind != "tree" or trace.get("traversal") in ("preorder", "inorder", "postorder"))
                     and reasoning_trace._schema(trace, kind, trace.get("traversal", ""),
                                                 trace.get("edge_ids") is not None))
        if (not valid or (rule, locus) not in reasoning_trace._inventory(trace)
                or synthesis._identity(trace, rule, locus) is None):
            return None
        field: str
        value: Any
        if kind == "shortest":
            if rule in ("PATH_ENDPOINTS", "MINIMALITY"):
                field, value = "nodes", trace["nodes"]
            elif rule == "DIRECTED_EDGE":
                match = re.fullmatch(r"nodes\[([0-9]+):([0-9]+)\]", locus)
                if match is None:
                    return None
                field, value = "nodes", trace["nodes"][int(match[1]):int(match[2])]
            elif rule == "EDGE_WEIGHT":
                match = re.fullmatch(r"weights\[([0-9]+)\]", locus)
                if match is None:
                    return None
                field, value = "weights", trace["weights"][int(match[1])]
            elif rule == "COST_SUM":
                field, value = "cost", trace["cost"]
            else:
                return None
        elif kind in ("bfs", "tree"):
            field = "steps" if kind == "bfs" else "events"
            if rule in ("BFS_COMPLETENESS", "TREE_COMPLETENESS"):
                value = trace[field]
            else:
                match = re.match(rf"{field}\[([0-9]+)\]", locus)
                if match is None:
                    return None
                value = trace[field][int(match[1])]
        elif kind == "dfa":
            field = "decision" if rule == "DFA_ACCEPTANCE" else "states"
            value = trace[field]
            if rule == "DFA_START":
                value = value[0]
            elif rule == "DFA_TRANSITION":
                match = re.fullmatch(r"states\[([0-9]+):([0-9]+)\]", locus)
                if match is None:
                    return None
                value = value[int(match[1]):int(match[2])]
        else:
            match = re.fullmatch(r"(nodes|ohm|kcl|input|query)(?:\[([0-9]+)\])?", locus)
            if match is None:
                return None
            field = match[1]
            value = trace[field]
            if match[2] is not None and field not in ("input", "query"):
                value = value[int(match[2])]
        return {"field": field, "value": copy.deepcopy(value)}
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RecursionError):
        return None


def _solution_history(history: list[Sample], actor: ActorItem) -> tuple[list[Sample], ActorItem, str]:
    if type(history) is not list or not history or any(type(row) is not Sample for row in history):
        raise ValueError("repair audit: nonempty actual Sample history required")
    actor_wire = to_json(actor)
    actor = from_json(ActorItem, actor_wire)
    samples = [from_json(Sample, to_json(replace(row, result=None, reward=0.0, diag_reward=None)))
               for row in history]
    prior: dict[str, Sample] = {}
    last_turn = 0
    for row in samples:
        if (not row.sample_id or row.sample_id in prior or not row.group_id or row.problem_id != actor.problem_id
                or row.policy_version != samples[0].policy_version or row.kind != f"T{row.turn}" or row.turn < last_turn):
            raise ValueError("repair audit: source identity, policy, or order differs")
        if row.turn == 1:
            if row.parent_sample_id is not None or row.prompt_token_ids != samples[0].prompt_token_ids:
                raise ValueError("repair audit: first-attempt input differs")
        else:
            parent = prior.get(row.parent_sample_id) if row.parent_sample_id is not None else None
            prefix = [] if parent is None else parent.prompt_token_ids + parent.completion_token_ids
            if (parent is None or parent.turn + 1 != row.turn or row.prompt_token_ids[:len(prefix)] != prefix):
                raise ValueError("repair audit: actual retry ancestry or token prefix differs")
        prior[row.sample_id] = row
        last_turn = row.turn
    return samples, actor, hashlib.sha256(actor_wire.encode("utf-8")).hexdigest()


def _samples(history: list[Sample], child: Sample, actor: ActorItem) -> tuple[list[Sample], ActorItem, str]:
    if type(history) is not list or type(child) is not Sample:
        raise ValueError("repair audit: actual Sample history and child required")
    samples, actor, actor_sha = _solution_history([*history, child], actor)
    if samples[-1].turn not in (2, 3) or samples[-1].parent_sample_id is None:
        raise ValueError("repair audit: online child must be an actual retry")
    return samples, actor, actor_sha


def _endpoint(index: int, row: dict, samples: list[Sample]) -> dict:
    return {"attempt": index, **{key: row[key] for key in ("kind", "rule", "locus")},
            "evidence_ranges": [[1, len(samples[index - 1].text.splitlines())]]}


def audit_online_repair(history: list[Sample], child: Sample, actor: ActorItem, verifier: Any, *,
                        inventory: PublicOriginInventory, response_contracts: dict[str, str | None],
                        sections: RetrySections, turn1_answer_contract: str | None = None,
                        retry_out_of_choice: str | None = None) -> RepairAudit:
    """Re-audit actual source and child operations; intent alone never earns repair.

    turn1_answer_contract is the bound eval-aligned T1 grading contract (None: strict Final answer);
    retry_out_of_choice its bound retry rule (None: strict comparison)."""
    from . import notebook_synthesis as synthesis
    from .observations import observation_spans

    if type(response_contracts) is not dict:
        raise ValueError("repair audit: explicit response contracts required")
    response_contracts = dict(response_contracts)
    sections = copy.deepcopy(sections)
    samples, actor, actor_sha = _samples(history, child, actor)
    child = samples[-1]
    parent_id = child.parent_sample_id
    assert parent_id is not None
    full = public_origin_inventory(samples, contract_version=ORIGIN_VERSION,
                                   response_contracts=response_contracts)
    if response_contracts[child.sample_id] != VERSION:
        raise ValueError("repair audit: explicit typed child contract required")
    expected = public_origin_inventory(samples[:-1], contract_version=ORIGIN_VERSION,
        response_contracts={row.sample_id: response_contracts[row.sample_id] for row in samples[:-1]},
        parent_sample_id=parent_id)
    if type(inventory) is not PublicOriginInventory or inventory != expected:
        raise ValueError("repair audit: complete frozen inventory binding differs")
    claims = parse_error_check(child.text, sections)
    solution = project_solution(child.text, sections)
    origins = {origin.ref_id: origin for origin in expected.origins}
    positions = {origin.ref_id: index for index, origin in enumerate(expected.origins)}
    uncertain_valid = (all(ref in positions for ref in sections.uncertain_refs)
                       and [positions[ref] for ref in sections.uncertain_refs]
                       == sorted(positions[ref] for ref in sections.uncertain_refs))
    sources = {source.sample_id: source for source in full.sources}
    by_id = {row.sample_id: row for row in samples}
    indices = {row.sample_id: index for index, row in enumerate(samples, 1)}
    fresh: dict[str, dict] = {}

    def native(sample_id: str) -> dict | None:
        source = sources[sample_id]
        if source.projection_error_code is not None:
            return None
        if sample_id not in fresh:
            result = synthesis.audit_native_sample(by_id[sample_id], actor, verifier,
                response_contract=source.response_contract,
                **synthesis.grade_options(by_id[sample_id].turn, turn1_answer_contract, retry_out_of_choice))
            if (result["version"] != synthesis.NATIVE_SAMPLE_AUDIT_VERSION
                    or result["sample_id"] != sample_id or result["problem_id"] != actor.problem_id
                    or result["response_contract"] != source.response_contract
                    or result["raw_sha256"] != source.raw_sha or result["finish_reason"] != source.finish_reason
                    or source.solution_span is None or result["solution_projection"] != {
                        "span": [source.solution_span.start, source.solution_span.end], "sha256": source.solution_sha}):
                raise ValueError("repair audit: fresh native source binding differs")
            fresh[sample_id] = result
        return fresh[sample_id]

    child_audit = native(child.sample_id)
    if child_audit is None:
        raise ValueError("repair audit: current child projection missing")

    def assess(claim: RepairClaim) -> RepairWitness:
        origin = origins.get(claim.origin_ref)
        hashes = [sources[child.sample_id].raw_sha]
        identity: JsonValue = None
        later_ref = None

        def witness(reason: str, *, structural: bool = False, verified: bool = False) -> RepairWitness:
            return RepairWitness(claim.origin_ref, later_ref, identity, "CORRECTED" if verified else None,
                "STRUCTURAL_INVALID" if structural else "VERIFIED" if verified else "UNVERIFIABLE",
                reason, tuple(sorted(set(hashes))))

        if not uncertain_valid:
            return witness("INVALID_UNCERTAIN_REFS", structural=True)
        if origin is None:
            return witness("ORIGIN_NOT_IN_SCOPE", structural=True)
        hashes.append(origin.raw_sha)
        if (claim.rule != origin.rule or claim.intent == "RECHECK_OBSERVATION" and origin.kind != "OBSERVATION"
                or claim.intent == "RECHECK_OPERATION" and origin.kind != "TRACE"):
            return witness("ORIGIN_RULE_OR_KIND_MISMATCH", structural=True)
        source_audit = native(origin.sample_id)
        if source_audit is None:
            return witness("MALFORMED_SOURCE_" + str(sources[origin.sample_id].projection_error_code))
        hashes.extend((_digest(source_audit), _digest(child_audit)))
        if claim.intent == "UNKNOWN":
            return witness("UNKNOWN_INTENT")
        matching = [row for row in source_audit["checks"]
                    if (row["kind"], row["rule"], row["locus"]) == (origin.kind, origin.rule, origin.locus)]
        if len(matching) != 1:
            return witness("SOURCE_CHECK_MISSING_OR_AMBIGUOUS")
        source_row = matching[0]
        identity = source_row["identity"]
        if source_row["status"] != "VIOLATED" or identity is None:
            return witness("SOURCE_NOT_VERIFIED_VIOLATED")
        candidates = [row for row in child_audit["checks"]
                      if row["kind"] == origin.kind and row["rule"] == origin.rule
                      and row["identity"] is not None and _canonical(row["identity"]) == _canonical(identity)]
        if len(candidates) != 1:
            return witness("CHILD_CHECK_MISSING_OR_AMBIGUOUS")
        target = candidates[0]
        destinations = [row for row in full.origins if row.sample_id == child.sample_id
                        and (row.kind, row.rule, row.locus) == (target["kind"], target["rule"], target["locus"])]
        if len(destinations) != 1:
            return witness("CHILD_PUBLIC_REFERENCE_MISSING_OR_AMBIGUOUS")
        later_ref = destinations[0].ref_id
        if target["status"] != "COMPLIANT":
            return witness("CHILD_NOT_VERIFIED_COMPLIANT")
        if not synthesis._descendant(indices[origin.sample_id], len(samples), samples):
            return witness("CHILD_NOT_DESCENDANT", structural=True)
        if origin.kind == "OBSERVATION":
            values = [value for _, _, key, value in observation_spans(solution) if key == target["locus"]]
            if len(values) != 1 or type(claim.proposed_value) is not str:
                return witness("OBSERVATION_PATCH_MISSING_OR_AMBIGUOUS")
            matched = claim.proposed_value.strip(" \t") == values[0].strip(" \t")
        else:
            trace = child_audit["native_trace"]
            patch = operation_patch(trace, target["rule"], target["locus"]) if type(trace) is dict else None
            if patch is None:
                return witness("UNSUPPORTED_OPERATION_PATCH")
            matched = _canonical(claim.proposed_value) == _canonical(patch)
        if not matched:
            return witness("PROPOSED_PATCH_DIFFERS_FROM_EMITTED")
        relation_audit = {"attempts": [{"checks": fresh.get(row.sample_id, {}).get("checks", []),
                                       "issues": fresh.get(row.sample_id, {}).get("issues", [])}
                                      for row in samples]}
        relationship = {"rule": origin.rule, "relation": "CORRECTED",
                        "earlier": _endpoint(indices[origin.sample_id], source_row, samples),
                        "later": _endpoint(len(samples), target, samples)}
        if not synthesis._lesson(relationship, samples, relation_audit, bounded=True)["verified"]:
            return witness("RELATION_NOT_VERIFIED")
        hashes.extend(ref["sha256"] for row in (source_row, target) for ref in row["refs"])
        return witness("VERIFIED_LOCAL_NATIVE_REPAIR", verified=True)

    witnesses = tuple(assess(claim) for claim in claims)
    submitted = {claim.origin_ref: (claim, witness) for claim, witness in zip(claims, witnesses, strict=True)}
    scores = []
    for origin in expected.origins:
        pair = submitted.get(origin.ref_id)
        scores.append(UnitScore(origin.ref_id, 1 if pair and pair[1].claim_verdict == "VERIFIED" else 0,
            pair[1].reason if pair else "NOT_SUBMITTED",
            (pair[0].raw_spans["claim"],) if pair else (), AUDIT_VERSION))
    bindings = {"version": AUDIT_VERSION, "semantic_scope": "native_rule_local_repair",
        "module_causality_certified": False, "all_reasoning_verified": False,
        "actor_sha256": actor_sha, "full_inventory_sha256": full.sha256, "fixed_units": expected.fixed_units,
        "response_contracts": dict(response_contracts), "sources": [asdict(source) for source in full.sources],
        "sections_sha256": _digest(asdict(sections)), "whole_response_truncated": child.finish_reason == "length",
        "uncertain_refs": list(sections.uncertain_refs), "uncertain_refs_valid": uncertain_valid,
        "token_text_correspondence_verified": False,
        "samples": [{"sample_id": row.sample_id, "problem_id": row.problem_id,
            "policy_version": row.policy_version, "turn": row.turn, "kind": row.kind,
            "parent_sample_id": row.parent_sample_id, "prompt_sha256": _digest(row.prompt_token_ids),
            "completion_sha256": _digest(row.completion_token_ids),
            "sampler_logprobs_sha256": _digest(row.sampler_logprobs)} for row in samples],
        "native_audits": {key: {"version": row["version"], "result_verifier_version": row["result"]["verifier_version"],
            "trace_verifier_version": row["trace_report"]["version"], "sha256": _digest(row),
            "result_sha256": _digest(row["result"]), "trace_report_sha256": _digest(row["trace_report"])}
            for key, row in fresh.items()}}
    bindings["sha256"] = _digest(bindings)
    return RepairAudit(parent_id, child.sample_id, sections, expected.sha256, witnesses, tuple(scores), bindings)


def audit_final_note(note: Sample, history: list[Sample], actor: ActorItem, verifier: Any, *,
                     inventory: PublicOriginInventory,
                     response_contracts: dict[str, str | None],
                     turn1_answer_contract: str | None = None,
                     retry_out_of_choice: str | None = None) -> FinalRepairAudit:
    """Audit only the supplied history; tree/prompt/token attachment belongs to P08/P09."""
    from . import notebook_synthesis as synthesis
    from .observations import observation_spans

    if type(note) is not Sample or type(response_contracts) is not dict:
        raise ValueError("repair final: actual DIAG and explicit response contracts required")
    response_contracts = dict(response_contracts)
    note = from_json(Sample, to_json(replace(note, result=None, reward=0.0, diag_reward=None)))
    samples, actor, actor_sha = _solution_history(history, actor)
    by_id = {row.sample_id: row for row in samples}
    if (note.kind != "DIAG" or not note.sample_id or note.sample_id in by_id or not note.group_id
            or note.group_id in {row.group_id for row in samples}
            or note.problem_id != actor.problem_id or note.policy_version != samples[0].policy_version
            or (note.parent_sample_id is None and note.turn != 1)
            or (note.parent_sample_id is not None and (note.parent_sample_id not in by_id
                or note.turn != by_id[note.parent_sample_id].turn))):
        raise ValueError("repair final: DIAG identity, policy, or recorded parent differs")
    parsed = parse_final_note(note.text, contract_version=FINAL_VERSION)
    expected = public_origin_inventory(samples, contract_version=ORIGIN_VERSION,
                                       response_contracts=response_contracts)
    if type(inventory) is not PublicOriginInventory or inventory != expected:
        raise ValueError("repair final: complete supplied-history inventory differs")
    origins = {row.ref_id: row for row in expected.origins}
    positions = {row.ref_id: index for index, row in enumerate(expected.origins)}
    indices = {row.sample_id: index for index, row in enumerate(samples, 1)}
    sources = {row.sample_id: row for row in expected.sources}
    fresh: dict[str, dict | None] = {}
    for row in samples:
        source = sources[row.sample_id]
        try:
            result = synthesis.audit_native_sample(row, actor, verifier, response_contract=source.response_contract,
                **synthesis.grade_options(row.turn, turn1_answer_contract, retry_out_of_choice))
        except RepairContractError as exc:
            if exc.code not in GRAMMAR_ERRORS or source.projection_error_code != exc.code:
                raise ValueError("repair final: fresh projection failure differs") from None
            fresh[row.sample_id] = None
            continue
        if (source.projection_error_code is not None or source.solution_span is None
                or result["version"] != synthesis.NATIVE_SAMPLE_AUDIT_VERSION
                or result["sample_id"] != row.sample_id or result["problem_id"] != actor.problem_id
                or result["response_contract"] != source.response_contract
                or result["raw_sha256"] != source.raw_sha or result["finish_reason"] != source.finish_reason
                or result["solution_projection"] != {"span": [source.solution_span.start, source.solution_span.end],
                                                     "sha256": source.solution_sha}):
            raise ValueError("repair final: fresh native source binding differs")
        fresh[row.sample_id] = result
    available = {key: row for key, row in fresh.items() if row is not None}
    correct = [row for row in samples if row.sample_id in available
               and available[row.sample_id]["result"]["status"] == "CORRECT"]
    if not correct:
        raise ValueError("repair final: no independently correct source for a success note")
    latest = max(row.turn for row in correct)
    success_id = next(row.sample_id for row in correct if row.turn == latest)
    metadata_errors = []
    actual_attempts = []
    for row in samples:
        audited = fresh[row.sample_id]
        grade = from_json(VerifierResult, json.dumps(audited["result"], allow_nan=False)) if audited is not None else None
        outcome, answer = public_attempt_outcome(row.text, response_contract=sources[row.sample_id].response_contract,
                                                 result=grade)
        actual_attempts.append({"sample_id": row.sample_id, "outcome_available": outcome is not None,
                                "outcome": outcome, "answer": answer})
    if any(not row["outcome_available"] for row in actual_attempts):
        metadata_errors.append("SOURCE_OUTCOME_UNAVAILABLE")
    if (len(parsed.attempts) != len(actual_attempts) or any(
            _canonical(submitted) != _canonical({key: actual[key] for key in ("sample_id", "outcome", "answer")})
            for submitted, actual in zip(parsed.attempts, actual_attempts))):
        metadata_errors.append("ATTEMPT_METADATA_DIFFERS")
    if parsed.success_sample_id != success_id:
        metadata_errors.append("SUCCESS_SELECTION_DIFFERS")
    uncertain_valid = (all(ref in positions for ref in parsed.uncertain_refs)
                       and [positions[ref] for ref in parsed.uncertain_refs]
                       == sorted(positions[ref] for ref in parsed.uncertain_refs))
    if not uncertain_valid:
        metadata_errors.append("INVALID_UNCERTAIN_REFS")
    metadata_valid = not metadata_errors
    relation_audit = {"version": synthesis.BOUNDED_VERSION, "attempts": [
        {"checks": available.get(row.sample_id, {}).get("checks", []),
         "issues": available.get(row.sample_id, {}).get("issues", []),
         "unverified": available.get(row.sample_id, {}).get("unverified", []),
         "text_sha256": sources[row.sample_id].raw_sha} for row in samples]}

    def native_row(ref: str) -> dict | None:
        origin = origins[ref]
        audited = fresh[origin.sample_id]
        if audited is None:
            return None
        rows = audited["output_checks"] if origin.kind == "OUTPUT" else audited["checks"]
        matching = [row for row in rows if (row["kind"], row["rule"], row["locus"])
                    == (origin.kind, origin.rule, origin.locus)]
        return matching[0] if len(matching) == 1 else None

    rows_by_ref = {ref: native_row(ref) for ref in origins}

    def ancestors(sample_id: str) -> set[str]:
        result = set()
        parent = by_id[sample_id].parent_sample_id
        while parent is not None:
            result.add(parent)
            parent = by_id[parent].parent_sample_id
        return result

    def assess(claim: FinalRepairClaim) -> RepairWitness:
        identity: JsonValue = None
        hashes = [parsed.raw_sha]

        def verdict(state: Literal["VERIFIED", "REFUTED", "UNVERIFIABLE", "STRUCTURAL_INVALID"], reason: str):
            return RepairWitness(claim.origin_ref, claim.later_ref, identity, claim.relation,
                                 state, reason, tuple(sorted(set(hashes))))

        if claim.origin_ref not in origins or claim.later_ref not in origins:
            return verdict("STRUCTURAL_INVALID", "REFERENCE_NOT_IN_SUPPLIED_HISTORY")
        origin, later = origins[claim.origin_ref], origins[claim.later_ref]
        hashes.extend((origin.raw_sha, later.raw_sha))
        if (origin.kind != later.kind or origin.rule != claim.rule or later.rule != claim.rule
                or indices[origin.sample_id] >= indices[later.sample_id]
                or (claim.relation == "FINALIZED") != (origin.kind == "OUTPUT")):
            return verdict("STRUCTURAL_INVALID", "REFERENCE_KIND_RULE_OR_ORDER_DIFFERS")
        descendant = synthesis._descendant(indices[origin.sample_id], indices[later.sample_id], samples)
        if claim.relation == "OTHER_BRANCH_COMPARISON":
            if (descendant or synthesis._descendant(indices[later.sample_id], indices[origin.sample_id], samples)
                    or not ancestors(origin.sample_id).intersection(ancestors(later.sample_id))):
                return verdict("STRUCTURAL_INVALID", "NO_SHARED_RECORDED_BRANCH_ANCESTOR")
        elif not descendant:
            return verdict("STRUCTURAL_INVALID", "DESTINATION_NOT_DESCENDANT")
        if claim.relation == "UNKNOWN" and (
                any(ref not in origins for ref in claim.blocked_by)
                or any(origins[ref].sample_id != later.sample_id or positions[ref] >= positions[claim.later_ref]
                       for ref in claim.blocked_by)
                or [positions[ref] for ref in claim.blocked_by] != sorted(positions[ref] for ref in claim.blocked_by)):
            return verdict("STRUCTURAL_INVALID", "BLOCKER_REFERENCE_SCOPE_OR_ORDER_DIFFERS")
        source, target = rows_by_ref[claim.origin_ref], rows_by_ref[claim.later_ref]
        if source is None or target is None:
            return verdict("UNVERIFIABLE", "NATIVE_PROOF_MISSING_OR_AMBIGUOUS")
        hashes.extend(ref["sha256"] for row in (source, target) for ref in row["refs"])
        hashes.extend(_digest(fresh[sample_id]) for sample_id in (origin.sample_id, later.sample_id))
        identity = source["identity"]
        if identity is None or target["identity"] is None:
            return verdict("UNVERIFIABLE", "UNSUPPORTED_SEMANTIC_IDENTITY")
        if _canonical(identity) != _canonical(target["identity"]):
            return verdict("STRUCTURAL_INVALID", "SEMANTIC_IDENTITY_DIFFERS")
        if claim.relation == "UNKNOWN" and origin.kind != "TRACE":
            return verdict("UNVERIFIABLE", "UNSUPPORTED_DEPENDENCY_UNKNOWN")
        if source["status"] == "COMPLIANT":
            return verdict("REFUTED", "SOURCE_PROVEN_COMPLIANT")
        if source["status"] != "VIOLATED":
            return verdict("UNVERIFIABLE", "SOURCE_NOT_PROVEN_VIOLATED")
        if claim.relation == "FINALIZED":
            if target["status"] == "VIOLATED":
                return verdict("REFUTED", "DESTINATION_OUTPUT_STILL_MISSING")
            return verdict("VERIFIED" if target["status"] == "COMPLIANT" else "UNVERIFIABLE",
                           "VERIFIED_OUTPUT_FINALIZATION" if target["status"] == "COMPLIANT" else "OUTPUT_UNVERIFIED")
        required = "VIOLATED" if claim.relation == "PERSISTED" else "UNVERIFIABLE" if claim.relation == "UNKNOWN" else "COMPLIANT"
        if target["status"] != required:
            return verdict("REFUTED" if target["status"] in ("COMPLIANT", "VIOLATED") else "UNVERIFIABLE",
                           "DESTINATION_CONTRADICTS_RELATION" if target["status"] in ("COMPLIANT", "VIOLATED")
                           else "DESTINATION_UNVERIFIED")
        if claim.relation == "UNKNOWN":
            if not target["blocked_by"]:
                return verdict("UNVERIFIABLE", "UNSUPPORTED_DEPENDENCY_UNKNOWN")
            blockers = [origins[ref] for ref in claim.blocked_by]
            dependencies = [{"rule": row.rule, "locus": row.locus} for row in blockers]
            if dependencies != target["blocked_by"]:
                return verdict("STRUCTURAL_INVALID", "BLOCKER_MEMBERSHIP_DIFFERS")
            proven = [rows_by_ref[ref] for ref in claim.blocked_by]
            if any(row is None or row["status"] != "VIOLATED" for row in proven):
                return verdict("UNVERIFIABLE", "BLOCKER_NOT_PROVEN_VIOLATED")
            selected = _endpoint(indices[later.sample_id], target, samples)
            del selected["attempt"]
            selected["blocked_by"] = dependencies
            if synthesis._unknown_binding(selected, indices[later.sample_id], samples, relation_audit) is None:
                return verdict("UNVERIFIABLE", "DEPENDENCY_EVIDENCE_NOT_VERIFIED")
            hashes.extend(ref["sha256"] for row in proven if row is not None for ref in row["refs"])
            return verdict("VERIFIED", "VERIFIED_DEPENDENCY_BLOCKED_UNKNOWN")
        if claim.relation == "CORRECTED":
            if origin.kind == "OBSERVATION":
                projection = sources[later.sample_id].solution_span
                assert projection is not None
                text = by_id[later.sample_id].text[projection.start:projection.end]
                values = [value for _, _, key, value in observation_spans(text) if key == later.locus]
                if len(values) != 1:
                    return verdict("UNVERIFIABLE", "OBSERVATION_PATCH_MISSING_OR_AMBIGUOUS")
                if type(claim.repaired_value) is not str:
                    return verdict("STRUCTURAL_INVALID", "REPAIRED_VALUE_KIND_DIFFERS")
                matched = claim.repaired_value.strip(" \t") == values[0].strip(" \t")
            else:
                target_audit = fresh[later.sample_id]
                assert target_audit is not None
                patch = operation_patch(target_audit["native_trace"], target["rule"], target["locus"])
                if patch is None:
                    return verdict("UNVERIFIABLE", "UNSUPPORTED_OPERATION_PATCH")
                if type(claim.repaired_value) is not dict:
                    return verdict("STRUCTURAL_INVALID", "REPAIRED_VALUE_KIND_DIFFERS")
                matched = _canonical(claim.repaired_value) == _canonical(patch)
            if not matched:
                return verdict("REFUTED", "REPAIRED_VALUE_DIFFERS_FROM_EMITTED")
        lesson = {"rule": claim.rule, "relation": claim.relation,
                  "earlier": _endpoint(indices[origin.sample_id], source, samples),
                  "later": _endpoint(indices[later.sample_id], target, samples)}
        if not synthesis._lesson(lesson, samples, relation_audit, bounded=True)["verified"]:
            return verdict("UNVERIFIABLE", "POSITIVE_RELATION_EVIDENCE_NOT_VERIFIED")
        return verdict("VERIFIED", "VERIFIED_NATIVE_RELATION")

    witnesses = tuple(assess(claim) for claim in parsed.claims)
    submitted = {claim.origin_ref: (claim, witness) for claim, witness in zip(parsed.claims, witnesses, strict=True)}
    proposed_scores = []
    for origin in expected.origins:
        pair = submitted.get(origin.ref_id)
        value: Literal[-1, 0, 1] = 0
        if pair and pair[0].relation != "FINALIZED":
            value = 1 if pair[1].claim_verdict == "VERIFIED" else -1 if pair[1].claim_verdict == "REFUTED" else 0
        proposed_scores.append(UnitScore(origin.ref_id, value, pair[1].reason if pair else "NOT_SUBMITTED",
            (pair[0].raw_spans["claim"],) if pair else (), FINAL_AUDIT_VERSION))
    applied_scores = tuple(proposed_scores) if metadata_valid else tuple(
        replace(row, value=0, reason="METADATA_INVALID") for row in proposed_scores)
    known: set[tuple[str, str, str, str]] = set()
    unknown: set[tuple[str, str, str, str]] = set()
    for sample_id, audited in fresh.items():
        if audited is not None:
            known.update((sample_id, row["kind"], row["rule"], row["locus"]) for row in audited["issues"])
            unknown.update((sample_id, row["kind"], row["rule"], row["locus"]) for row in audited["unverified"])
    covered, verified_corrections = set(), set()
    for claim, witness in zip(parsed.claims, witnesses, strict=True):
        if witness.claim_verdict == "VERIFIED" and claim.relation != "FINALIZED":
            origin = origins[claim.origin_ref]
            covered.add((origin.sample_id, origin.kind, origin.rule, origin.locus))
            if claim.relation == "PERSISTED":
                later = origins[claim.later_ref]
                covered.add((later.sample_id, later.kind, later.rule, later.locus))
            if claim.relation == "CORRECTED":
                verified_corrections.add((claim.origin_ref, claim.later_ref))
    opportunities = set()
    for before_ref, before in rows_by_ref.items():
        if (before is None or before["status"] != "VIOLATED" or before["identity"] is None
                or origins[before_ref].kind == "OUTPUT"):
            continue
        for after_ref, after in rows_by_ref.items():
            if (after is not None and after["status"] == "COMPLIANT" and after["identity"] is not None
                    and before["kind"] == after["kind"] and before["rule"] == after["rule"]
                    and _canonical(before["identity"]) == _canonical(after["identity"])
                    and synthesis._descendant(indices[origins[before_ref].sample_id],
                                             indices[origins[after_ref].sample_id], samples)):
                opportunities.add((before_ref, after_ref))
    verified = sum(row.claim_verdict == "VERIFIED" for row in witnesses)
    false = sum(row.claim_verdict == "REFUTED" for row in witnesses)
    unverified = sum(row.claim_verdict == "UNVERIFIABLE" for row in witnesses)
    structural = sum(row.claim_verdict == "STRUCTURAL_INVALID" for row in witnesses)
    output = sum(row.claim_verdict == "VERIFIED" and row.relation == "FINALIZED" for row in witnesses)
    accurate = metadata_valid and verified > 0 and not (false or unverified or structural)
    metrics = {"submitted_claim_count": len(witnesses), "verified_claim_count": verified,
        "native_claim_count": sum(row.relation != "FINALIZED" for row in witnesses),
        "verified_native_count": verified - output, "verified_output_count": output,
        "native_correction_count": len(verified_corrections), "false_claim_count": false,
        "unverifiable_claim_count": unverified, "structural_invalid_count": structural,
        "submitted_claim_precision": verified / len(witnesses) if witnesses else None,
        "known_issue_count": len(known), "covered_known_issue_count": len(known & covered),
        "uncovered_known_issue_count": len(known - covered),
        "known_issue_coverage": len(known & covered) / len(known) if known else None,
        "complete_known_issue_coverage": not (known - covered) if known else None,
        "fresh_unknown_count": len(unknown), "declared_uncertain_count": len(parsed.uncertain_refs),
        "malformed_source_count": sum(row is None for row in fresh.values()),
        "correction_opportunity_count": len(opportunities),
        "covered_correction_opportunity_count": len(opportunities & verified_corrections),
        "native_correction_coverage": len(opportunities & verified_corrections) / len(opportunities) if opportunities else None}
    bindings = {"version": FINAL_AUDIT_VERSION, "semantic_scope": "native_rule_local_repair",
        "history_scope": "supplied_solution_history", "full_tree_attachment_verified": False,
        "token_text_correspondence_verified": False, "module_causality_certified": False,
        "all_reasoning_verified": False, "lesson_semantics_verified": False,
        "actor_sha256": actor_sha, "inventory_sha256": expected.sha256, "fixed_units": expected.fixed_units,
        "response_contracts": response_contracts, "sources": [asdict(row) for row in expected.sources],
        "note_raw_sha256": parsed.raw_sha, "parsed_note_sha256": _digest(asdict(parsed)),
        "whole_response_truncated": note.finish_reason == "length", "actual_attempts": actual_attempts,
        "actual_success_sample_id": success_id, "uncertain_refs_valid": uncertain_valid,
        "samples": [{"sample_id": row.sample_id, "problem_id": row.problem_id, "kind": row.kind,
            "group_id": row.group_id, "seed": row.seed,
            "turn": row.turn, "policy_version": row.policy_version, "parent_sample_id": row.parent_sample_id,
            "finish_reason": row.finish_reason, "prompt_sha256": _digest(row.prompt_token_ids),
            "completion_sha256": _digest(row.completion_token_ids), "sampler_logprobs_sha256": _digest(row.sampler_logprobs)}
            for row in [*samples, note]],
        "native_audits": {key: {"version": row["version"], "result_verifier_version": row["result"]["verifier_version"],
            "trace_verifier_version": row["trace_report"]["version"], "sha256": _digest(row),
            "result_sha256": _digest(row["result"]), "trace_report_sha256": _digest(row["trace_report"])}
            if row is not None else None for key, row in fresh.items()}}
    bindings["sha256"] = _digest(bindings)
    return FinalRepairAudit(note.sample_id, expected.sha256, metadata_valid, tuple(metadata_errors), accurate,
                            witnesses, tuple(proposed_scores), applied_scores, metrics, bindings)


@dataclass(frozen=True)
class ComponentCredit:
    sample_id: str
    component: Literal["online", "final"]
    coefficients: tuple[float, ...]
    original_group_size: int
    fixed_units: int
    completion_cap: int
    ownership_sha256: str
    unit_rows: tuple[dict, ...]
    source_bindings: dict[str, Any]


def _credit_inventory(inventory: PublicOriginInventory) -> PublicOriginInventory:
    from .repair_contract import MAX_RAW_CHARS, PublicOrigin, PublicOriginSource

    if type(inventory) is not PublicOriginInventory:
        raise ValueError("repair group credit: public inventory required")
    inventory = copy.deepcopy(inventory)
    if (inventory.contract_version != ORIGIN_VERSION or type(inventory.origins) is not tuple
            or type(inventory.sources) is not tuple
            or any(type(row) is not PublicOrigin for row in inventory.origins)
            or any(type(row) is not PublicOriginSource for row in inventory.sources)):
        raise ValueError("repair group credit: inventory types differ")
    sources = {row.sample_id: row for row in inventory.sources}
    if (len(sources) != len(inventory.sources)
            or len({row.ref_id for row in inventory.origins}) != inventory.fixed_units
            or inventory.parent_sample_id is not None and inventory.parent_sample_id not in sources):
        raise ValueError("repair group credit: inventory identities differ")
    seen: set[str] = set()
    for source in inventory.sources:
        if (type(source.sample_id) is not str or not source.sample_id or source.sample_id in seen
                or source.parent_sample_id is not None and (type(source.parent_sample_id) is not str
                                                            or source.parent_sample_id not in seen)
                or type(source.raw_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", source.raw_sha) is None
                or type(source.finish_reason) is not str or source.finish_reason not in ("stop", "length")
                or source.response_contract is not None and source.response_contract != VERSION):
            raise ValueError("repair group credit: inventory source shape differs")
        if source.solution_span is None:
            if source.solution_sha is not None or type(source.projection_error_code) is not str:
                raise ValueError("repair group credit: inventory source failure differs")
        elif (type(source.solution_span) is not RawSpan or type(source.solution_span.start) is not int
                or type(source.solution_span.end) is not int
                or not 0 <= source.solution_span.start <= source.solution_span.end <= MAX_RAW_CHARS
                or type(source.solution_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", source.solution_sha) is None
                or source.projection_error_code is not None):
            raise ValueError("repair group credit: inventory source projection differs")
        seen.add(source.sample_id)
    ordinals: dict[str, int] = {}
    source_order = {source.sample_id: index for index, source in enumerate(inventory.sources)}
    last_source = -1
    for origin in inventory.origins:
        if (type(origin.ref_id) is not str or not origin.ref_id or origin.sample_id not in sources
                or origin.kind not in ("OBSERVATION", "TRACE", "OUTPUT")
                or type(origin.rule) is not str or type(origin.locus) is not str
                or origin.raw_sha != sources[origin.sample_id].raw_sha
                or type(origin.occurrence_spans) is not tuple
                or any(type(span) is not RawSpan or type(span.start) is not int or type(span.end) is not int
                       or not 0 <= span.start < span.end for span in origin.occurrence_spans)):
            raise ValueError("repair group credit: inventory origin differs")
        ordinal = ordinals.get(origin.sample_id, 0) + 1
        solution_span = sources[origin.sample_id].solution_span
        if (origin.ref_id != origin.sample_id + f":r{ordinal}" or source_order[origin.sample_id] < last_source
                or solution_span is not None and any(
                    not solution_span.start <= span.start < span.end
                    <= solution_span.end for span in origin.occurrence_spans)):
            raise ValueError("repair group credit: original inventory order or spans differ")
        ordinals[origin.sample_id] = ordinal
        last_source = source_order[origin.sample_id]
    body = {"contract_version": inventory.contract_version, "parent_sample_id": inventory.parent_sample_id,
            "origins": [asdict(row) for row in inventory.origins],
            "sources": [asdict(row) for row in inventory.sources], "fixed_units": inventory.fixed_units}
    if inventory.sha256 != _digest(body):
        raise ValueError("repair group credit: inventory digest differs")
    return inventory


def _credit_audit(sample: Sample, audit: RepairAudit | FinalRepairAudit,
                  inventory: PublicOriginInventory, component: str, parsed) -> tuple[dict, dict]:
    expected_type = RepairAudit if component == "online" else FinalRepairAudit
    version = AUDIT_VERSION if component == "online" else FINAL_AUDIT_VERSION
    if type(audit) is not expected_type or audit.origins_digest != inventory.sha256:
        raise ValueError("repair group credit: audit inventory or type differs")
    bindings = audit.source_bindings
    if (type(bindings) is not dict or bindings.get("sha256") != _digest({
            key: value for key, value in bindings.items() if key != "sha256"})
            or bindings.get("version") != version or type(bindings.get("fixed_units")) is not int
            or bindings["fixed_units"] != inventory.fixed_units):
        raise ValueError("repair group credit: audit seal or version differs")
    hashes = ("actor_sha256", "full_inventory_sha256", "sections_sha256") if component == "online" else (
        "actor_sha256", "inventory_sha256", "note_raw_sha256", "parsed_note_sha256")
    if (any(type(bindings.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", bindings[key]) is None for key in hashes)
            or bindings.get("token_text_correspondence_verified") is not False
            or bindings.get("module_causality_certified") is not False
            or bindings.get("all_reasoning_verified") is not False):
        raise ValueError("repair group credit: audit provenance fields differ")
    if component == "online":
        if (type(audit) is not RepairAudit or audit.child_id != sample.sample_id
                or audit.parent_id != sample.parent_sample_id or inventory.parent_sample_id != audit.parent_id
                or audit.sections != parsed or bindings.get("sections_sha256") != _digest(asdict(parsed))):
            raise ValueError("repair group credit: current retry binding differs")
    elif (type(audit) is not FinalRepairAudit or audit.note_id != sample.sample_id
            or inventory.parent_sample_id is not None
            or type(audit.metadata_valid) is not bool or type(audit.metadata_errors) is not tuple
            or audit.metadata_valid != (not audit.metadata_errors)
            or bindings.get("inventory_sha256") != inventory.sha256
            or bindings.get("note_raw_sha256") != parsed.raw_sha
            or bindings.get("parsed_note_sha256") != _digest(asdict(parsed))):
        raise ValueError("repair group credit: current final binding differs")
    source_rows = bindings.get("sources")
    samples = bindings.get("samples")
    contracts = bindings.get("response_contracts")
    if (type(source_rows) is not list or type(samples) is not list or not samples or type(contracts) is not dict
            or any(type(row) is not dict for row in source_rows + samples)):
        raise ValueError("repair group credit: source binding shape differs")
    source_ids = [row["sample_id"] for row in source_rows]
    if (len(set(source_ids)) != len(source_ids) or set(contracts) != set(source_ids)
            or any(value is not None and (type(value) is not str or value != VERSION) for value in contracts.values())
            or any(row["response_contract"] != contracts[row["sample_id"]] for row in source_rows)):
        raise ValueError("repair group credit: source contract map differs")
    for source in inventory.sources:
        matches = [row for row in source_rows if row["sample_id"] == source.sample_id]
        if len(matches) != 1 or _canonical(matches[0]) != _canonical(asdict(source)):
            raise ValueError("repair group credit: fixed source differs")
    expected = {"sample_id": sample.sample_id, "problem_id": sample.problem_id,
        "policy_version": sample.policy_version, "turn": sample.turn, "kind": sample.kind,
        "parent_sample_id": sample.parent_sample_id, "prompt_sha256": _digest(sample.prompt_token_ids),
        "completion_sha256": _digest(sample.completion_token_ids),
        "sampler_logprobs_sha256": _digest(sample.sampler_logprobs)}
    if component == "final":
        expected.update(group_id=sample.group_id, seed=sample.seed, finish_reason=sample.finish_reason)
    if (_canonical(samples[-1]) != _canonical(expected)
            or [row["sample_id"] for row in samples] != source_ids + ([] if component == "online" else [sample.sample_id])
            or bindings.get("whole_response_truncated") is not (sample.finish_reason == "length")):
        raise ValueError("repair group credit: sample/token/logprob binding differs")
    claims: tuple[RepairClaim | FinalRepairClaim, ...]
    if component == "online":
        solution = project_solution(sample.text, parsed)
        current = source_rows[-1]
        if (current["raw_sha"] != parsed.raw_sha or current["finish_reason"] != sample.finish_reason
                or current["parent_sample_id"] != sample.parent_sample_id or contracts[sample.sample_id] != VERSION
                or current["solution_span"] != asdict(parsed.solution_span)
                or current["solution_sha"] != hashlib.sha256(solution.encode()).hexdigest()
                or current["projection_error_code"] is not None):
            raise ValueError("repair group credit: retry raw projection differs")
        claims = parse_error_check(sample.text, parsed)
    else:
        claims = parsed.claims
    native = bindings.get("native_audits")
    if (type(native) is not dict or not set(native) <= set(source_ids)
            or component == "online" and sample.sample_id not in native
            or component == "final" and set(native) != set(source_ids)):
        raise ValueError("repair group credit: native audit membership differs")
    for proof in native.values():
        if proof is None and component == "final":
            continue
        if (type(proof) is not dict or set(proof) != {"version", "result_verifier_version", "trace_verifier_version",
                "sha256", "result_sha256", "trace_report_sha256"}
                or proof["version"] != "repair-native-sample-audit-v1"
                or any(type(proof[key]) is not str or not proof[key] for key in ("result_verifier_version", "trace_verifier_version"))
                or any(type(proof[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", proof[key]) is None
                       for key in ("sha256", "result_sha256", "trace_report_sha256"))):
            raise ValueError("repair group credit: native audit provenance differs")
    if (type(audit.witnesses) is not tuple or len(audit.witnesses) != len(claims)
            or type(audit.unit_scores) is not tuple or len(audit.unit_scores) != inventory.fixed_units):
        raise ValueError("repair group credit: witness/unit inventory differs")
    witnesses = {}
    origins = {origin.ref_id: origin for origin in inventory.origins}
    for claim, witness in zip(claims, audit.witnesses, strict=True):
        if (type(witness) is not RepairWitness or witness.origin_ref != claim.origin_ref
                or witness.claim_verdict not in ("VERIFIED", "REFUTED", "UNVERIFIABLE", "STRUCTURAL_INVALID")
                or type(witness.reason) is not str or not witness.reason
                or type(witness.evidence_hashes) is not tuple or parsed.raw_sha not in witness.evidence_hashes
                or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in witness.evidence_hashes)
                or witness.evidence_hashes != tuple(sorted(set(witness.evidence_hashes)))):
            raise ValueError("repair group credit: witness binding differs")
        if component == "online":
            if witness.relation != ("CORRECTED" if witness.claim_verdict == "VERIFIED" else None):
                raise ValueError("repair group credit: online witness relation differs")
        elif (witness.later_ref != cast(FinalRepairClaim, claim).later_ref
              or witness.relation != cast(FinalRepairClaim, claim).relation):
            raise ValueError("repair group credit: authored final relation differs")
        origin = origins.get(claim.origin_ref)
        if (origin is not None and witness.claim_verdict in ("VERIFIED", "REFUTED")
                and origin.raw_sha not in witness.evidence_hashes):
            raise ValueError("repair group credit: source witness hash differs")
        if witness.claim_verdict in ("VERIFIED", "REFUTED"):
            if (origin is None or origin.rule != claim.rule or witness.semantic_identity is None
                    or component == "online" and (cast(RepairClaim, claim).intent == "UNKNOWN"
                        or (cast(RepairClaim, claim).intent == "RECHECK_OBSERVATION") != (origin.kind == "OBSERVATION"))):
                raise ValueError("repair group credit: semantic witness and authored claim differ")
            if component == "final":
                later = origins.get(cast(FinalRepairClaim, claim).later_ref)
                if later is None or (later.kind, later.rule) != (origin.kind, claim.rule):
                    raise ValueError("repair group credit: final witness references differ")
        witnesses[claim.origin_ref] = witness
    submitted = {claim.origin_ref: claim for claim in claims}
    proposed = audit.proposed_unit_scores if type(audit) is FinalRepairAudit else audit.unit_scores
    if type(proposed) is not tuple or len(proposed) != inventory.fixed_units:
        raise ValueError("repair group credit: proposed unit inventory differs")
    for origin, unit, before in zip(inventory.origins, audit.unit_scores, proposed, strict=True):
        unit_claim, unit_witness = submitted.get(origin.ref_id), witnesses.get(origin.ref_id)
        semantic = (1 if unit_witness and unit_witness.claim_verdict == "VERIFIED" else
                    -1 if component == "final" and unit_witness and unit_witness.claim_verdict == "REFUTED" else 0)
        if origin.kind == "OUTPUT":
            semantic = 0
        expected_spans = (unit_claim.raw_spans["claim"],) if unit_claim else ()
        reason = unit_witness.reason if unit_witness else "NOT_SUBMITTED"
        for row, value, expected_reason in ((before, semantic, reason), (unit,
                0 if type(audit) is FinalRepairAudit and not audit.metadata_valid else semantic,
                "METADATA_INVALID" if type(audit) is FinalRepairAudit and not audit.metadata_valid else reason)):
            if (type(row) is not UnitScore or type(row.value) is not int or row.value != value
                    or row.origin_ref != origin.ref_id or row.reason != expected_reason
                    or row.verifier_version != version or row.raw_spans != expected_spans
                    or type(row.raw_spans) is not tuple
                    or any(type(span) is not RawSpan or type(span.start) is not int or type(span.end) is not int
                           for span in row.raw_spans)):
                raise ValueError("repair group credit: unit/raw claim consistency differs")
    return submitted, witnesses


def repair_group_credit(group: Group, audits: dict[str, RepairAudit | FinalRepairAudit | None], tokenizer: Any, *,
                        inventory: PublicOriginInventory, completion_cap: int,
                        component: Literal["online", "final"]) -> tuple[ComponentCredit, ...]:
    """Map original units before full-group centering; supplied audits are consistency receipts, not truth."""
    from types import SimpleNamespace
    from .repair_tokens import emitted_patch_spans, span_token_ownership
    from .repair_contract import parse_retry_sections
    from .whole_reference_notebook import public_inventory

    if type(group) is not Group or type(audits) is not dict or component not in ("online", "final"):
        raise ValueError("repair group credit: explicit group/audits/component required")
    group = from_json(Group, to_json(group))
    audits = copy.deepcopy(audits)
    inventory = _credit_inventory(inventory)
    size, cap = (2, 1024) if component == "online" else (4, 8192)
    if (type(completion_cap) is not int or completion_cap != cap or len(group.samples) != size
            or group.kind not in (("T2", "T3") if component == "online" else ("DIAG",))
            or not group.group_id or not group.problem_id or group.state_sha256 != _digest(group.samples[0].prompt_token_ids)
            or set(audits) != {sample.sample_id for sample in group.samples}
            or not group.usable
            or any(not sample.sample_id or not sample.prompt_token_ids or not 0 < len(sample.completion_token_ids) <= cap
                   or component == "online" and (sample.kind != f"T{sample.turn}" or sample.parent_sample_id is None)
                   for sample in group.samples)):
        raise ValueError("repair group credit: original group, cap, or audit membership differs")
    candidates: list[tuple[Sample, RepairAudit | FinalRepairAudit | None, Any, dict, dict, dict]] = []
    empty_ownership = {}
    history_binding = None
    for sample in group.samples:
        empty_ownership[sample.sample_id] = span_token_ownership(sample.text, sample.completion_token_ids, tokenizer, ())
        try:
            parsed = (parse_retry_sections(sample.text, contract_version=VERSION) if component == "online"
                      else parse_final_note(sample.text, contract_version=FINAL_VERSION))
        except RepairContractError as exc:
            if (exc.code not in GRAMMAR_ERRORS or audits[sample.sample_id] is not None
                    or sample.result is not None or sample.reward != 0):
                raise ValueError("repair group credit: malformed audit or integrity failure") from None
            candidates.append((sample, None, None, {}, {}, {}))
            continue
        audit = audits[sample.sample_id]
        if audit is None:
            raise ValueError("repair group credit: parsed candidate requires its actual audit")
        submitted, witnesses = _credit_audit(sample, audit, inventory, component, parsed)
        common = _canonical({"actor_sha256": audit.source_bindings["actor_sha256"],
                             "samples": audit.source_bindings["samples"][:-1],
                             "sources": audit.source_bindings["sources"][:-1] if component == "online"
                             else audit.source_bindings["sources"]})
        if history_binding is not None and common != history_binding:
            raise ValueError("repair group credit: sibling source histories differ")
        history_binding = common
        destinations = {}
        if component == "online":
            solution = project_solution(sample.text, cast(RetrySections, parsed))
            local = public_inventory([SimpleNamespace(text=solution, sample_id=sample.sample_id,
                                       parent_sample_id=None, finish_reason=sample.finish_reason)])
            destinations = {sample.sample_id + ref.removeprefix("a1"): local["references"][ref]
                            for ref in local["assessment_refs"]}
        candidates.append((sample, audit, parsed, submitted, witnesses, destinations))
    mapped = []
    for sample, audit, parsed, submitted, witnesses, destinations in candidates:
        unit_rows: list[dict] = []
        occupied: set[int] = set()
        for index, origin in enumerate(inventory.origins):
            unit = audit.unit_scores[index] if audit is not None else None
            claim, witness = submitted.get(origin.ref_id), witnesses.get(origin.ref_id)
            semantic = unit.value if unit is not None else 0
            spans = unit.raw_spans if unit is not None else ()
            ownership = (span_token_ownership(sample.text, sample.completion_token_ids, tokenizer, spans) if spans
                         else copy.deepcopy(empty_ownership[sample.sample_id]))
            claim_safe = bool(ownership["token_indices"])
            patch_safe, patch_spans = False, None
            if component == "online" and witness is not None and witness.later_ref is not None:
                target = destinations.get(witness.later_ref)
                if (target is None or target["kind"] != origin.kind or target["rule"] != origin.rule):
                    raise ValueError("repair group credit: emitted destination binding differs")
                identity: JsonValue
                if origin.kind == "OBSERVATION":
                    identity = ["OBSERVATION", target["locus"]]
                else:
                    from .notebook_synthesis import _identity
                    from .reasoning_trace import _trace
                    trace = _trace(project_solution(sample.text, parsed))[0]
                    identity = _identity(trace, target["rule"], target["locus"])
                if identity is None or _canonical(identity) != _canonical(witness.semantic_identity):
                    raise ValueError("repair group credit: emitted semantic identity differs")
                patch_spans = emitted_patch_spans(sample.text, parsed, kind=target["kind"],
                                                   rule=target["rule"], locus=target["locus"])
                if patch_spans is not None:
                    if semantic == 1 and claim is not None:
                        actual = (sample.text[patch_spans[0].start:patch_spans[0].end].strip(" \t")
                                  if origin.kind == "OBSERVATION" else operation_patch(trace, target["rule"], target["locus"]))
                        proposed = claim.proposed_value.strip(" \t") if origin.kind == "OBSERVATION" else claim.proposed_value
                        if _canonical(actual) != _canonical(proposed):
                            raise ValueError("repair group credit: proposed and emitted patch differ")
                    patch_safe = bool(span_token_ownership(sample.text, sample.completion_token_ids,
                                                          tokenizer, patch_spans)["token_indices"])
                    ownership = span_token_ownership(sample.text, sample.completion_token_ids,
                                                     tokenizer, spans + patch_spans)
            applied = semantic
            mapping_failed = semantic != 0 and (not claim_safe or component == "online" and not patch_safe)
            if mapping_failed:
                applied = 0
                ownership = copy.deepcopy(empty_ownership[sample.sample_id])
            indices = ownership["token_indices"]
            if occupied.intersection(indices):
                raise ValueError("repair group credit: inter-unit token ownership collision")
            occupied.update(indices)
            relation, verdict = (witness.relation, witness.claim_verdict) if witness is not None else (None, None)
            unit_rows.append({"origin_ref": origin.ref_id, "kind": origin.kind, "semantic_score": semantic,
                "applied_score": applied, "mapping_status": "MAPPING_FAILED" if mapping_failed else ownership["status"],
                "reason": "MAPPING_FAILED" if mapping_failed else unit.reason if unit is not None else "MALFORMED_RESPONSE",
                "semantic_reason": unit.reason if unit is not None else "MALFORMED_RESPONSE",
                "raw_spans": ownership["spans"], "token_indices": indices,
                "excluded_crossing_token_indices": ownership["excluded_crossing_token_indices"],
                "relation": relation, "verdict": verdict,
                "corrected_channel": origin.kind != "OUTPUT" and relation == "CORRECTED",
                "positive_verified_corrected": False})
        mapped.append(unit_rows)
    means = [sum(rows[index]["applied_score"] for rows in mapped) / size for index in range(inventory.fixed_units)]
    output = []
    member_ids = [sample.sample_id for sample in group.samples]
    for sample_index, ((sample, audit, _, _, _, _), rows) in enumerate(zip(candidates, mapped, strict=True)):
        coefficients = [0.0] * len(sample.completion_token_ids)
        for row, mean in zip(rows, means, strict=True):
            advantage = row["applied_score"] - mean
            row.update(mean=mean, advantage=advantage)
            row["positive_verified_corrected"] = bool(row["corrected_channel"] and row["verdict"] == "VERIFIED"
                                                       and row["applied_score"] == 1 and advantage > 0)
            coefficient = completion_cap * .0625 * advantage / (inventory.fixed_units * len(row["token_indices"])) if (
                inventory.fixed_units and row["token_indices"]) else 0.0
            row["coefficient"] = coefficient
            for index in row["token_indices"]:
                coefficients[index] += coefficient
        if any(not math.isfinite(value) for value in coefficients):
            raise ValueError("repair group credit: nonfinite token coefficient")
        bindings = {"version": "repair-group-credit-v1", "group_sha256": _digest(json.loads(to_json(group))),
            "group_id": group.group_id, "problem_id": group.problem_id, "group_kind": group.kind,
            "group_members": list(member_ids), "group_members_sha256": _digest(member_ids), "sample_index": sample_index,
            "sample_sha256": _digest(json.loads(to_json(sample))), "inventory_sha256": inventory.sha256,
            "audit_sha256": _digest(asdict(audit)) if audit is not None else None,
            "token_text_correspondence_verified": True, "semantic_truth_reverified": False,
            "engine_attachment_verified": False, "optimizer_admission": False}
        output.append(ComponentCredit(sample.sample_id, component, tuple(coefficients), size, inventory.fixed_units,
                                      completion_cap, _digest(rows), tuple(rows), bindings))
    return tuple(output)
