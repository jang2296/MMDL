"""Whole-history structured synthesis reward; never a teacher notebook or routing oracle.

Only explicit native traces and declared observations can be certified. The bounded
contract accounts for uncertainty without certifying it; an answer never repairs an issue.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Any

from . import episode_notebook
from .answers import _FINAL, extract_eval_aligned, extract_final
from .notebook_learning import _first_attempt_lines, _unique, _validate
from .observations import _normalize, observation_spans, slot_id, verified_wrong_slots
from .rollout import _hash
from .schema import ActorItem, ProblemTree, Sample, VerifierResult, from_json, to_json

VERSION = "local-whole-history-structured-synthesis-v1"
BOUNDED_VERSION = "local-whole-history-bounded-synthesis-v4"
INPUT_POLICY = "question-images-lossless-all-branches-public-output-relationships-synthesis-v3"
LEGACY_FEEDBACK_POLICY = "whole-history-synthesis-public-format-and-boolean-checks-v2"
FEEDBACK_POLICY = "whole-history-synthesis-submitted-fact-reinspection-and-evidence-v6"
FORMAT_DIAGNOSTICS_VERSION = "synthesis-submitted-json-types-and-empty-attempt-evidence-v4"
REPAIR_PARENT_POLICY = "first-public-shape-valid-v1"
CIRCUIT_EVALUATION_ROUTING = "native-circuit-unit-aware-evaluation-observations-v1"
MAX_FORMAT_ERRORS = 32
FORMAT_ERROR_CODES = ("INVALID_JSON", "MISSING_KEY", "UNEXPECTED_KEY", "EXPECTED_OBJECT", "EXPECTED_LIST",
                      "EXPECTED_INTEGER", "EXPECTED_STRING", "EXPECTED_ANSWER", "EXPECTED_NULL", "EXPECTED_RANGE_PAIR")
LEGACY_RETRY_INSTRUCTION = """Your preceding structured notebook did not pass verification. Rewrite the complete
notebook using ALL original attempts, the original images, and the exact original output contract.
The feedback contains public JSON type/key/range-shape errors, failed indices and boolean checks, never
correct facts, replacement claims, expected trace operations, or teacher prose. Format errors describe
only your submitted JSON; an empty format-error list does not certify content or evidence.
Use outcome, not status; corrected_facts must be an object and evidence_ranges a list of integer pairs.
Correct answers do not certify reasoning or observations.
Keep every branch, separate outcomes from issues, and do not turn unverified later evidence into repair.
Return a new complete JSON notebook. Verifier feedback:
"""
RETRY_INSTRUCTION = """Your preceding structured notebook did not pass verification. Rewrite the complete
notebook using ALL original attempts, original images, and the same five-field output contract, with lessons first.
Compare every public retry-parent pair at the same operation or image slot. A parent without a final answer
still has inspectable observations; a correct child answer alone proves no correction or persistence.
FINALIZED may describe missing-to-present terminal answer output; it never certifies the answer or native reasoning.
Use only relationships you can support with original evidence. Honest abstention is allowed.
submitted_lesson_count counts only your submitted list. Zero means no relationships were submitted.
submitted_lessons_verified is null for an empty list or when lesson checks were not reached.
An empty failed_lessons list then supplies no verified lesson evidence; read all_issues_synthesized separately
for coverage. Empty lists do not prove there were no issues or relationships to inspect.
reinspection_slots identifies only independently failed checks of your submitted corrected_facts values.
These IDs are not per-attempt causes or expected values. Reinspect those facts in the original images.
Missing, malformed, unsupported or ambiguous facts supply no IDs; an empty list certifies no facts.
empty_attempt_evidence_positions lists 1-based positions in your submitted attempts array, not claimed
attempt IDs. For each nonempty original, supply nonempty valid evidence_ranges from its original line IDs.
Use [] only for an actually empty original response; never invent a line. The positions are submission
metadata, not verified failures; a truncated list omits later positions.
The feedback contains only submitted JSON type/key/range-shape errors, counts, failed indices, submitted
fact IDs and boolean checks, never correct facts, expected issues, replacement relationships, or teacher prose.
Preserve every branch and unknown region. Return the complete JSON notebook. Verifier feedback:
"""
REQUIRED_CHECKS = ("format_valid", "all_attempts_verified", "observation_claims_complete",
                   "lessons_verified", "all_issues_synthesized", "answer_verified", "registered_facts_verified")
_TOP = {"corrected_facts", "attempts", "success_attempt", "corrected_answer", "lessons"}
_ATTEMPT = {"attempt", "outcome", "answer", "evidence_ranges", "issues"}
_CLAIM = {"kind", "rule", "locus", "evidence_ranges"}
_ENDPOINT = _CLAIM | {"attempt"}
_FORMAT_PATH_FIELDS = _TOP | _ATTEMPT | _CLAIM | {"unverified", "blocked_by", "earlier", "later", "relation", "status"}
_RELATIONS = {"CORRECTED", "PERSISTED", "NOT_REVISITED", "OTHER_BRANCH_COMPARISON"}
_DEPENDENCIES = {
    "EDGE_WEIGHT": {"DIRECTED_EDGE"},
    "MINIMALITY": {"PATH_ENDPOINTS", "DIRECTED_EDGE", "EDGE_WEIGHT", "COST_SUM"},
    "BFS_FIFO": {"BFS_FIFO", "BFS_NEW_NEIGHBORS"},
    "BFS_NEW_NEIGHBORS": {"BFS_FIFO", "BFS_NEW_NEIGHBORS"},
    "BFS_COMPLETENESS": {"BFS_FIFO", "BFS_NEW_NEIGHBORS"},
    "DFA_ACCEPTANCE": {"DFA_COMPLETE_WORD", "DFA_TRANSITION"},
}
_ROOT_UNKNOWNS = {("TRACE_FORMAT", "solution"), ("TRACE_SCHEMA", "trace"),
                  ("RESPONSE_SCOPE", "response")}
_SCOPE = ("All recorded output must follow the supported native response contract: declared image "
          "observations, one complete structured solution trace, and one final answer. Verified synthesis "
          "covers those actual claims and every sampled branch. Missing, ambiguous or extra prose remains "
          "unverified. It does not certify latent reasoning, universal rules, module causality, future "
          "performance, or missing image observations as correct.")
_BOUNDED_SCOPE = ("Every original branch, proven issue and supported unknown region is accounted for. "
                  "Malformed or unsupported response scope and undecidable observations remain unknown, "
                  "never semantically correct. Native task-contract uncertainty blocks success. This does "
                  "not certify all reasoning, complete diagnosis, latent repair, module causality, universal "
                  "lessons, future performance, or missing image observations as correct.")
PUBLIC_COVERAGE_VERSION = "all-authored-observation-occurrences-public-coverage-v1"
NATIVE_SAMPLE_AUDIT_VERSION = "repair-native-sample-audit-v1"


def public_coverage_contract() -> dict:
    return {"version": PUBLIC_COVERAGE_VERSION,
        "inputs": ["text", "sample_id", "parent_sample_id"],
        "algorithm": "all attempts in input order; all observation_spans occurrences grouped by canonical slot in first-appearance order, without value normalization, deduplication or outcome filtering; exact assignment lines and Unicode-character spans from _first_attempt_lines; unmatched lines inside every Observed facts block, including unterminated blocks, remain raw and ungraded; all public parent/retry edges",
        "attempt_columns": ["attempt", "sample_id", "parent_sample_id", "line_count"],
        "occurrence_columns": ["attempt", "line_id", "start", "end", "value", "source_line"],
        "raw_line_columns": ["attempt", "line_id", "start", "end", "source_line"],
        "retry_pair_columns": ["parent_attempt", "retry_attempt"],
        "semantics": "public index/work items only; no facts, classifications, selected relationships or verification"}


def public_coverage(samples: list[Sample]) -> dict:
    """Index authored claims and lineage using only the three public input fields."""
    ids = {sample.sample_id: index for index, sample in enumerate(samples, 1)}
    if len(ids) != len(samples):
        raise ValueError("public coverage: sample IDs must be unique")
    attempts, raw_lines, pairs = [], [], []
    slots: dict[str, list] = {}
    for attempt, sample in enumerate(samples, 1):
        lines = _first_attempt_lines(sample.text)
        attempts.append([attempt, sample.sample_id, sample.parent_sample_id, len(lines)])
        if sample.parent_sample_id is not None:
            if sample.parent_sample_id not in ids:
                raise ValueError("public coverage: original retry parent missing")
            pairs.append([ids[sample.parent_sample_id], attempt])
        occurrences = {(start, end): (key, value)
                       for start, end, key, value in observation_spans(sample.text)}
        in_observations = False
        for line in lines:
            if re.fullmatch(r"[ \t]*Observed facts:[ \t]*", line["text"].rstrip("\r\n"), re.I):
                in_observations = True
            elif re.fullmatch(r"[ \t]*Solution:[ \t]*", line["text"].rstrip("\r\n"), re.I):
                in_observations = False
            elif (line["start"], line["end"]) in occurrences:
                key, value = occurrences[line["start"], line["end"]]
                slots.setdefault(key, []).append([attempt, line["line_id"], line["start"],
                                                  line["end"], value, line["text"]])
            elif in_observations:
                raw_lines.append([attempt, line["line_id"], line["start"], line["end"], line["text"]])
    return {**public_coverage_contract(), "attempts": attempts, "slots": slots,
            "raw_observation_lines": raw_lines, "retry_pairs": pairs}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _refs(text: str, spans: list[list[int]]) -> list[dict]:
    return [{"span": list(pair), "sha256": _sha(text[pair[0]:pair[1]])} for pair in sorted(set(map(tuple, spans)))]


def _solution_text(sample: Sample, *, response_contract: str | None = None) -> tuple[str, int]:
    if response_contract is None:
        return sample.text, 0
    from .repair_contract import parse_retry_sections, project_solution

    sections = parse_retry_sections(sample.text, contract_version=response_contract)
    return project_solution(sample.text, sections), sections.solution_span.start


def _output_checks(sample: Sample, *, response_contract: str | None = None) -> list[dict]:
    """Inspect final-record presence only, without changing answer extraction or correctness."""
    text, offset = _solution_text(sample, response_contract=response_contract)
    if sample.finish_reason != "stop":
        return []
    extracted = extract_final(text, sample.finish_reason)
    matches = list(_FINAL.finditer(text.rstrip()))
    lines = _first_attempt_lines(text)
    if not matches and extracted.status == "INVALID_NO_FINAL":
        status, evidence = "VIOLATED", lines
    elif (extracted.status == "OK" and len(matches) == 1
            and not text[matches[0].end():].strip()):
        status = "COMPLIANT"
        evidence = [line for line in lines if line["start"] <= matches[0].start()
                    and matches[0].end() <= line["end"]]
        if len(evidence) != 1:
            return []
    else:
        return []
    return [{"kind": "OUTPUT", "rule": "FINALIZATION", "locus": "final_answer", "status": status,
             "identity": ["OUTPUT", "FINALIZATION", "same-task"], "blocked_by": [],
             "refs": _refs(sample.text, [[line["start"] + offset, line["end"] + offset]
                                          for line in evidence])}]


def _identity(trace: dict, rule: str, locus: str) -> Any | None:
    """Bind cross-attempt operations to authored operands, never array index alone."""
    kind = trace.get("kind")
    if kind == "circuit_ladder":
        from .circuit_trace import identity

        return identity(trace, rule, locus)
    if rule in {"PATH_ENDPOINTS", "MINIMALITY", "BFS_COMPLETENESS", "TREE_START",
                "TREE_COMPLETENESS", "DFA_START", "DFA_COMPLETE_WORD"}:
        return [kind, rule, "same-native-task"]
    if kind == "shortest":
        if rule in {"DIRECTED_EDGE", "EDGE_WEIGHT"}:
            pair = re.fullmatch(r"nodes\[([0-9]+):([0-9]+)\]", locus)
            weight = re.fullmatch(r"weights\[([0-9]+)\]", locus)
            if pair or weight:
                match = pair or weight
                assert match is not None
                index = int(match[1])
                nodes, ids = trace.get("nodes", []), trace.get("edge_ids")
                if pair and int(pair[2]) != index + 2 or index + 1 >= len(nodes):
                    return None
                if ids is not None:
                    if index >= len(ids) or not re.fullmatch(r"e[0-9]{1,2}", ids[index]):
                        return None
                    operand = ["edge-id", ids[index]]
                else:
                    operand = ["node-pair", sorted(nodes[index:index + 2])]
                return [kind, rule, operand]
            # Preserve the explicitly versioned v1 trace report's stable edge loci.
            if re.fullmatch(r"(?:edge|weight)\[(?:e[0-9]{1,2}|[A-H],[A-H])\]", locus):
                return [kind, rule, locus]
        if rule == "COST_SUM":
            return [kind, rule, trace.get("nodes"), trace.get("edge_ids"), trace.get("weights")]
    if kind == "bfs":
        match = re.fullmatch(r"steps\[([0-9]+)\]\.(node|new)", locus)
        if match and rule in {"BFS_FIFO", "BFS_NEW_NEIGHBORS"}:
            index = int(match[1])
            steps = trace.get("steps", [])
            if index < len(steps):
                # The preceding declared operations establish the same queue/visited state.
                return [kind, rule, steps[:index], steps[index]["node"] if match[2] == "new" else None]
    if kind == "tree":
        match = re.fullmatch(r"events\[([0-9]+)\]", locus)
        if match:
            index = int(match[1])
            events = trace.get("events", [])
            if index < len(events):
                event = events[index]
                if rule == "TREE_CHILD_EDGE" and event.get("op") == "child":
                    return [kind, rule, event["parent"], event["side"]]
                if rule == "TREE_EVENT_ORDER":
                    return [kind, rule, trace.get("traversal"), events[:index]]
    if kind == "dfa":
        match = re.fullmatch(r"states\[([0-9]+):([0-9]+)\]", locus)
        states = trace.get("states", [])
        if match and rule == "DFA_TRANSITION":
            index = int(match[1])
            if index < len(states) and int(match[2]) == index + 2:
                # Same symbol position in this fixed task AND same source state.
                return [kind, rule, index, states[index]]
        if rule == "DFA_ACCEPTANCE" and states:
            return [kind, rule, states[-1]]
    return None


def _trace_checks(sample: Sample, report: dict, *, response_contract: str | None = None) -> list[dict]:
    from .reasoning_trace import _trace, validate_trace_report

    if response_contract is None:
        validate_trace_report(report, sample.text, sample.finish_reason)
    else:
        validate_trace_report(report, sample.text, sample.finish_reason, response_contract=response_contract)
    if report.get("text_sha256") != _sha(sample.text) or report.get("finish_reason") != sample.finish_reason:
        raise ValueError("synthesis: fresh trace is not bound to actual output")
    evidence = report.get("trace")
    trace: dict = {}
    trace_refs: list[dict] = []
    if evidence is not None:
        text, offset = _solution_text(sample, response_contract=response_contract)
        trace, parsed_evidence = _trace(text)
        parsed_evidence = {**parsed_evidence, "span": [value + offset for value in parsed_evidence["span"]]}
        if parsed_evidence != evidence:
            raise ValueError("synthesis: invalid exact trace reference")
        start, end = evidence["span"]
        trace_refs = _refs(sample.text, [[start, end]])
    checks = []
    for row in report["checks"]:
        spans = [ref["span"] for ref in row["observation_refs"]]
        for ref in row["observation_refs"]:
            start, end = ref["span"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(sample.text)
                    or ref["sha256"] != _sha(sample.text[start:end])):
                raise ValueError("synthesis: invalid exact observation reference")
        checks.append({"kind": "TRACE", "rule": row["rule"], "locus": row["locus"],
                       "status": row["status"], "refs": trace_refs + _refs(sample.text, spans),
                       "identity": _identity(trace, row["rule"], row["locus"]),
                       "blocked_by": row.get("blocked_by", [])})
    return checks


def _observation_checks(sample: Sample, verifier: Any, *, response_contract: str | None = None) -> list[dict]:
    text, offset = _solution_text(sample, response_contract=response_contract)
    grouped: dict[str, list[tuple[str, list[int]]]] = {}
    usable: dict[str, list[tuple[str, list[int]]]] = {}
    for start, end, key, value in observation_spans(text):
        start, end = start + offset, end + offset
        grouped.setdefault(key, []).append((value, [start, end]))
        if (_single_line_scalar(sample.text[start:end].rstrip("\r\n"))
                and _native_value_shape(key, value)):
            usable.setdefault(key, []).append((value, [start, end]))
    checks = []
    for key, occurrences in grouped.items():
        known = usable.get(key, [])
        wrong_units = [(value, pair) for value, pair in occurrences
                       if _single_line_scalar(sample.text[pair[0]:pair[1]].rstrip("\r\n"))
                       and _component_unit_mismatch(key, value)]
        body = "Observed facts:\n" + "\n".join(f"{key} = {value}" for value, _ in known) + "\nSolution:\n"
        # Historical number parsing accepts units; native dimensionless facts cannot hide prose there.
        result = verifier.grade(sample.problem_id, body, "stop", "obs") if known else None
        slots = result.slots if result else None
        status = ("VIOLATED" if slots and key in verified_wrong_slots(slots) else "COMPLIANT"
                  if slots and slots.parsed == 1 and slots.all_correct == 1 and not slots.conflicting
                  and not slots.unverified_relations else "UNVERIFIABLE")
        regions = [(status, known)] if status != "UNVERIFIABLE" else []
        if wrong_units:
            regions.append(("VIOLATED", wrong_units))
        if status == "UNVERIFIABLE" and known or len(known) + len(wrong_units) < len(occurrences):
            regions.append(("UNVERIFIABLE", occurrences))
        for state, evidence in regions:
            checks.append({"kind": "OBSERVATION", "rule": "OBSERVATION_MATCH", "locus": key,
                           "status": state, "refs": _refs(sample.text, [pair for _, pair in evidence]),
                           "identity": ["OBSERVATION", key], "blocked_by": []})
    return checks


def grade_options(turn: int, answer_contract: str | None, retry_out_of_choice: str | None = None) -> dict:
    """Bound eval-aligned grading keywords for one actual attempt: T1 takes the answer contract, a retry the
    out-of-choice rule; each only when bound."""
    if turn == 1:
        return {} if answer_contract is None else {"answer_contract": answer_contract}
    return {} if retry_out_of_choice is None else {"retry_out_of_choice": retry_out_of_choice}


def audit_native_sample(sample: Sample, actor: ActorItem, verifier: Any, *,
                        response_contract: str | None = None, answer_contract: str | None = None,
                        retry_out_of_choice: str | None = None) -> dict:
    """Fresh native evidence for one actual response, independent of episode success.

    answer_contract (eval-aligned T1 only) grades and binds the answer by V8 extraction of the whole
    response; such a response has no Final answer line by design, so no FINALIZATION check is made.
    retry_out_of_choice (typed retries only) grades a committed MCQ answer naming no option INCORRECT."""
    from .reasoning_trace import _trace
    from .repair_contract import MAX_RAW_CHARS
    from .verifier import RETRY_OUT_OF_CHOICE_TAG

    to_json(actor)
    if type(sample) is not Sample:
        raise ValueError("synthesis: actual native sample required")
    sample = from_json(Sample, to_json(replace(sample, result=None, diag_reward=None, reward=0.0)))
    if (sample.problem_id != actor.problem_id or sample.kind != f"T{sample.turn}"
            or len(sample.text) > MAX_RAW_CHARS
            or answer_contract is not None and (sample.turn != 1 or response_contract is not None)
            or retry_out_of_choice is not None and (sample.turn == 1 or response_contract is None)):
        raise ValueError("synthesis: invalid native sample identity or bounds")
    text, offset = _solution_text(sample, response_contract=response_contract)
    options = {} if response_contract is None else {"response_contract": response_contract}
    result = verifier.grade(sample.problem_id, sample.text, sample.finish_reason, actor.mode, **options,
                            **({} if answer_contract is None else {"answer_contract": answer_contract}),
                            **({} if retry_out_of_choice is None else {"retry_out_of_choice": retry_out_of_choice}))
    serialized_result = json.loads(to_json(result))
    expected = (extract_final(text, sample.finish_reason) if answer_contract is None else
                extract_eval_aligned(text, sample.finish_reason, actor.question, actor.choices, actor.answer_format))
    if (result.extracted != expected.answer
            or answer_contract is not None and not result.verifier_version.endswith("+" + answer_contract)
            or result.verifier_version.endswith(RETRY_OUT_OF_CHOICE_TAG) is not (retry_out_of_choice is not None)):
        raise ValueError("synthesis: native answer is not bound to submitted solution")
    report = verifier.trace(sample.problem_id, sample.text, sample.finish_reason, **options)
    trace_checks = _trace_checks(sample, report, response_contract=response_contract)
    if report["trace"] is None and any(row["rule"] == "TRACE_FORMAT" for row in report["checks"]):
        try:
            _trace(text)
        except (ValueError, RecursionError):
            pass
        else:
            raise ValueError("synthesis: native trace format failure differs from submitted solution")
    checks = _merge_checks(_observation_checks(sample, verifier, response_contract=response_contract) + trace_checks)
    return {"version": NATIVE_SAMPLE_AUDIT_VERSION, "sample_id": sample.sample_id,
            "problem_id": sample.problem_id, "response_contract": response_contract,
            "raw_sha256": _sha(sample.text), "finish_reason": sample.finish_reason,
            "solution_projection": {"span": [offset, len(sample.text)], "sha256": _sha(text)},
            "result": serialized_result, "trace_report": report,
            "native_trace": _trace(text)[0] if report["trace"] is not None else None,
            "checks": checks, "issues": [row for row in checks if row["status"] == "VIOLATED"],
            "unverified": [row for row in checks if row["status"] == "UNVERIFIABLE"],
            "output_checks": [] if answer_contract is not None else _output_checks(
                sample, response_contract=response_contract)}


def _native_value_shape(key: str, value: str) -> bool:
    from .reasoning_trace import native_integer_value

    key = slot_id(key)
    if key.startswith(("edge[", "weight[")):
        return native_integer_value(value)
    if re.fullmatch(r"comp\[(?:e[12]|r[1-6])\]", key):
        unit = "V" if key.startswith("comp[e") else "ohm"
        return re.fullmatch(r"[1-9][0-9]?[ \t]+" + unit, value) is not None
    if key in ("start", "root") or key.startswith(("trans[", "dir[", "child[")):
        return _normalize(value, "relation", key) is not None
    return True


def _component_unit_mismatch(key: str, value: str) -> bool:
    """A recognized wrong dimension is an error; missing units or prose stay unknown."""
    key = slot_id(key)
    match = re.fullmatch(r"[1-9][0-9]?[ \t]+(V|A|ohm)", value)
    return bool(re.fullmatch(r"comp\[(?:e[12]|r[1-6])\]", key) and match
                and match[1] != ("V" if key.startswith("comp[e") else "ohm"))


def _single_line_scalar(value: Any) -> bool:
    from .reasoning_trace import native_scalar_line

    return native_scalar_line(value)


def _reinspection_facts(note: Any) -> dict[str, str]:
    """Keep only unambiguous, losslessly parsed native scalar submissions, without private keys."""
    facts = note.get("corrected_facts") if type(note) is dict else None
    if type(facts) is not dict or any(type(key) is not str for key in facts):
        return {}
    canonical = [slot_id(key) for key in facts]
    result = {}
    for key, value in facts.items():
        identity = slot_id(key)
        if (canonical.count(identity) != 1 or not _single_line_scalar(key)
                or not _single_line_scalar(value) or not _native_value_shape(key, value)):
            continue
        kind = "relation"
        if identity == "accepting":
            kind, shape = "text", r"q[0-9]+(?:[ \t]*,[ \t]*q[0-9]+)*"
        elif identity == "start" or identity.startswith("trans["):
            shape = r"q[0-9]+"
        elif identity == "root" or identity.startswith("child["):
            shape = r"[a-z]"
        elif identity.startswith("dir["):
            shape = r"(?:[a-z][ \t]*(?:->|→)[ \t]*[a-z]|n[0-5][ \t]*(?:->|→)[ \t]*n[0-5])"
        elif re.fullmatch(r"comp\[(?:e[12]|r[1-6])\]", identity):
            unit = "V" if identity.startswith("comp[e") else "ohm"
            kind, shape = "number", rf"[1-9][0-9]?[ \t]+{unit}"
        elif re.fullmatch(r"wire\[(?:e[12]|r[1-6])\]", identity):
            shape = r"n[0-5][ \t]*-[ \t]*n[0-5]"
        elif identity.startswith(("edge[", "weight[")):
            kind, shape = "number", r"[0-9]+"
        else:
            continue
        if (_normalize(value, kind, identity) is None
                or not re.fullmatch(rf"[ \t]*{shape}[ \t]*", value, re.I)):
            continue
        parsed = observation_spans("Observed facts:\n" + key + " = " + value + "\nSolution:\n")
        if len(parsed) == 1 and parsed[0][2:] == (identity, value.strip(" \t")):
            result[key] = value
    return result


def _reinspection_projection(facts: dict[str, str]) -> str:
    # No answer is supplied: this independent check cannot inspect or disclose answer correctness.
    return "Observed facts:\n" + "\n".join(f"{key} = {value}" for key, value in sorted(facts.items())) + "\nSolution:\n"


def _fact_reinspection(note: Any, record: dict, problem_id: str, verifier: Any) -> dict:
    facts = _reinspection_facts(note) if record["note_finish_reason"] == "stop" else {}
    projection = _reinspection_projection(facts)
    result = verifier.grade(problem_id, projection, "stop", "obs") if facts else None
    receipt = {"feedback_policy": FEEDBACK_POLICY, "problem_id": problem_id,
               "note_text_sha256": record["note_text_sha256"], "note_finish_reason": record["note_finish_reason"],
               "episode_audit_sha256": record["episode_audit_sha256"], "facts": facts,
               "projection_sha256": _sha(projection), "result": json.loads(to_json(result)) if result else None}
    return receipt | {"sha256": _hash(receipt)}


def _reinspection_feedback(checked: dict, sample: Sample | None) -> list[str]:
    receipt = checked.get("fact_reinspection")
    if (type(receipt) is not dict or set(receipt) != {"feedback_policy", "problem_id", "note_text_sha256",
            "note_finish_reason", "episode_audit_sha256", "facts", "projection_sha256", "result", "sha256"}
            or receipt["sha256"] != _hash({key: value for key, value in receipt.items() if key != "sha256"})
            or receipt["feedback_policy"] != FEEDBACK_POLICY or type(receipt["problem_id"]) is not str
            or any(receipt[key] != checked.get(key) for key in
                   ("note_text_sha256", "note_finish_reason", "episode_audit_sha256"))
            or type(receipt["facts"]) is not dict
            or receipt["facts"] != _reinspection_facts({"corrected_facts": receipt["facts"]})
            or receipt["projection_sha256"] != _sha(_reinspection_projection(receipt["facts"]))):
        raise ValueError("synthesis: submitted-fact reinspection binding differs")
    if sample is not None:
        try:
            note = json.loads(sample.text, object_pairs_hook=_unique, parse_float=Decimal, parse_constant=Decimal)
        except (TypeError, ValueError, InvalidOperation, RecursionError):
            note = None
        facts = _reinspection_facts(note) if sample.finish_reason == "stop" else {}
        if receipt["problem_id"] != sample.problem_id or receipt["facts"] != facts:
            raise ValueError("synthesis: submitted-fact reinspection differs from actual sample")
    if receipt["result"] is None:
        if receipt["facts"]:
            raise ValueError("synthesis: submitted-fact reinspection requires its verifier result")
        return []
    result = from_json(VerifierResult, json.dumps(receipt["result"], allow_nan=False))
    if (not receipt["facts"] or result.status != "INVALID_NO_FINAL" or result.extracted is not None
            or result.reward_answer != 0 or not result.verifier_version):
        raise ValueError("synthesis: reinspection must use an answer-free submitted-fact grade")
    # False optional relations, conflicts and missing required keys are deliberately not hints.
    slots = result.slots
    hints = sorted({slot_id(key) for key in receipt["facts"]} & set(slots.wrong_required)) if slots else []
    if hints and sample is None:
        raise ValueError("synthesis: submitted-fact reinspection requires its actual sample")
    return hints


def _key(row: dict) -> tuple[str, str, str]:
    return row["kind"], row["rule"], row["locus"]


def _merge_checks(rows: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for row in rows:
        key = (*_key(row), row["status"] == "UNVERIFIABLE" if row["kind"] == "OBSERVATION" else False)
        if key not in merged:
            merged[key] = {**row, "refs": list(row["refs"]), "blocked_by": list(row["blocked_by"])}
        else:
            current = merged[key]
            if current["identity"] != row["identity"]:
                current["identity"] = None
            current["refs"] += [ref for ref in row["refs"] if ref not in current["refs"]]
            current["blocked_by"] += [ref for ref in row["blocked_by"] if ref not in current["blocked_by"]]
            if row["status"] == "VIOLATED" or current["status"] != "VIOLATED" and row["status"] == "UNVERIFIABLE":
                current["status"] = row["status"]
    return [merged[key] for key in sorted(merged)]


def _scope_unknown(row: dict) -> bool:
    return (row["kind"] == "TRACE" and row["status"] == "UNVERIFIABLE"
            and (row["rule"], row["locus"]) in _ROOT_UNKNOWNS and row["blocked_by"] == [])


def _eligible_unknown(row: dict, checks: list[dict], *, accounted: bool = False) -> bool:
    if accounted and row["kind"] == "TRACE" and row["rule"].startswith("CIRCUIT_"):
        from .circuit_trace import supported_unknown

        # The original trace report was freshly validated before these checks were projected.
        return supported_unknown(row, checks)
    if accounted and (_scope_unknown(row) or row["kind"] == "OBSERVATION"
            and row["rule"] == "OBSERVATION_MATCH" and row["status"] == "UNVERIFIABLE"
            and row["blocked_by"] == []):
        return True
    allowed = _DEPENDENCIES.get(row["rule"], set())
    if not accounted and row["rule"] == "DFA_ACCEPTANCE":
        allowed = set()
    dependencies = row["blocked_by"]
    return bool(row["kind"] == "TRACE" and row["status"] == "UNVERIFIABLE" and dependencies
                and allowed and all(dependency["rule"] in allowed and any(
                    check["kind"] == "TRACE" and check["status"] == "VIOLATED"
                    and check["rule"] == dependency["rule"] and check["locus"] == dependency["locus"]
                    for check in checks) for dependency in dependencies))


def _unknown_binding(value: Any, attempt: int, samples: list[Sample], audit: dict) -> dict | None:
    if (type(value) is not dict or set(value) != _CLAIM | {"blocked_by"}
            or not _claim({key: item for key, item in value.items() if key != "blocked_by"})):
        return None
    sample = samples[attempt - 1]
    selected = _selected(value["evidence_ranges"], sample, allow_empty=True)
    checks = audit["attempts"][attempt - 1]["checks"]
    matching = [row for row in checks if _key(row) == _key(value) and row["status"] == "UNVERIFIABLE"]
    if len(matching) != 1 or selected is None:
        return None
    row = matching[0]
    if not _eligible_unknown(row, checks, accounted=True) or value["blocked_by"] != row["blocked_by"]:
        return None
    prerequisite_refs = [ref for dependency in row["blocked_by"] for check in checks
                         if check["kind"] == "TRACE" and check["rule"] == dependency["rule"]
                         and check["locus"] == dependency["locus"] for ref in check["refs"]]
    empty_scope = (sample.text == "" and _scope_unknown(row) and selected == [] and row["refs"] == []
                   and audit["attempts"][attempt - 1]["text_sha256"] == _sha(""))
    return row if empty_scope or _covers(selected, row["refs"] + prerequisite_refs) else None


def audit_episode(tree: ProblemTree, actor: ActorItem, verifier: Any, *, contract: str = VERSION,
                  evaluation_only: bool = False) -> dict:
    """Reuse strict raw ancestry/routing checks; audit outcome and issues separately."""
    if contract not in (VERSION, BOUNDED_VERSION):
        raise ValueError("synthesis: unknown explicit contract")
    original = episode_notebook.audit_episode(tree, actor, verifier,
        routing_rule=episode_notebook.STRICT_ROUTE_RULE, evaluation_only=evaluation_only)
    attempts = []
    for index, sample in enumerate(episode_notebook.solution_samples(tree), 1):
        trace = verifier.trace(tree.problem_id, sample.text, sample.finish_reason)
        checks = _merge_checks(_observation_checks(sample, verifier) + _trace_checks(sample, trace))
        if contract == BOUNDED_VERSION:
            for check in checks:
                if _scope_unknown(check):
                    check["refs"] = _refs(sample.text, [[line["start"], line["end"]]
                                                        for line in _first_attempt_lines(sample.text)])
        assert sample.result is not None
        attempts.append({"attempt": index, "sample_id": sample.sample_id,
                         "outcome": sample.result.status, "answer": sample.result.extracted,
                         "checks": checks, "issues": [row for row in checks if row["status"] == "VIOLATED"],
                         "unverified": [row for row in checks if row["status"] == "UNVERIFIABLE"],
                         "trace": trace, "trace_complete": trace["coverage"]["complete"] is True,
                         "text_sha256": _sha(sample.text)})
        if contract == BOUNDED_VERSION:
            attempts[-1]["output_checks"] = _output_checks(sample)
    if evaluation_only and actor.source == "gen-circuit-v1":
        # Legacy slot scores strip units; only the fresh circuit checks certify these observations.
        visual = [row["attempt"] for row in attempts
                  if any(check["kind"] == "OBSERVATION" for check in row["issues"])]
        complete = [row["attempt"] for row in attempts
                    if row["attempt"] in original["complete_observation_attempts"]
                    and not any(check["kind"] == "OBSERVATION" for check in row["issues"] + row["unverified"])]
        verified = bool(attempts) and len(complete) == len(attempts)
        original.update(routing_rule=CIRCUIT_EVALUATION_ROUTING,
            route="VISION_ON" if visual else "VISION_OFF",
            admission="VISION_ON" if visual else "VISION_OFF" if verified else "UNRESOLVED",
            visual_attempts=visual, complete_observation_attempts=complete,
            unobserved_or_partial_attempts=[row["attempt"] for row in attempts
                                          if row["attempt"] not in visual and row["attempt"] not in complete],
            all_required_observations_verified=verified)
    route_keys: tuple[str, ...] = ("route", "routing_rule", "admission", "visual_attempts", "complete_observation_attempts",
                  "unobserved_or_partial_attempts", "all_required_observations_verified",
                  "skipped_solution_branches", "model_claims_used_for_routing", "module_causality_certified")
    if evaluation_only:
        route_keys += ("training_complete_episode", "training_admission", "evaluation_only_single_sample_groups")
    return {"version": contract, "solution_sha256": original["solution_sha256"],
            "transcript_sha256": original["transcript_sha256"], "actor_sha256": _hash(actor.__dict__),
            "policy_version": tree.policy_version, "attempt_count": len(attempts), "attempts": attempts,
            "success_attempt": original["success_attempt"], "complete_episode": original["complete_episode"],
            "solution_trace_complete": bool(attempts) and all(row["trace_complete"] for row in attempts),
            ("unknown_regions_accountable" if contract == BOUNDED_VERSION else "dependency_regions_accountable"):
                bool(attempts) and all(
                all(_eligible_unknown(check, row["checks"], accounted=contract == BOUNDED_VERSION)
                    for check in row["unverified"])
                for row in attempts),
            "observation_claims_complete": all(not any(c["kind"] == "OBSERVATION" for c in row["unverified"])
                                               for row in attempts),
            "semantic_scope": _BOUNDED_SCOPE if contract == BOUNDED_VERSION else _SCOPE,
            **{key: original[key] for key in route_keys}}


BOUNDED_PROMPT_EXTENSION = """
Explicit bounded contract local-whole-history-bounded-synthesis-v4: retain ALL original response text
and all proven issues. Every attempt also has the required key unverified, a list of objects with EXACTLY
kind (TRACE or OBSERVATION), rule, locus, evidence_ranges, blocked_by. Dependency blocked_by lists objects with rule and locus
references to actual earlier violated prerequisites IN THE SAME attempt. Account for EVERY operation
whose check is undecidable because that prerequisite failed; never claim the operation was correct or wrong.
Dependency-bound EDGE_WEIGHT/MINIMALITY, BFS FIFO/new-neighbor/completeness and DFA_ACCEPTANCE checks are eligible.
Root TRACE_FORMAT@solution, TRACE_SCHEMA@trace and RESPONSE_SCOPE@response unknowns require blocked_by=[]
and the WHOLE original response evidence. An actually empty response uses evidence_ranges=[]; invent no line 1.
OBSERVATION_MATCH unknowns require blocked_by=[] and evidence for EVERY original occurrence of that slot.
If a slot also has a proven wrong occurrence, report BOTH its known issue and its unknown region.
These regions remain explicitly unknown, not correct. Unsupported NATIVE_CONTRACT remains blocking.
For a later same-operation or same-slot check that is unverified, use the explicit relation
UNVERIFIABLE_LATER. Its later endpoint adds blocked_by with those actual prerequisite references.
If a later whole-response root unknown may conceal an operation, NOT_REVISITED is forbidden; use
SCOPE_UNVERIFIABLE_LATER with that actual later root unknown endpoint, blocked_by=[] and WHOLE response evidence.
It does not assert same-operation identity. An unrelated slot unknown does not conceal other operations.
These relations mean undecidable later evidence, NEVER correction, persistence or absence. No complete diagnosis,
all-reasoning correctness, latent repair, or universal lesson is certified by this bounded notebook.
Optional OUTPUT lessons use rule FINALIZATION, locus final_answer and relation FINALIZED at both endpoints.
They require an earlier stopped response with no final-answer marker and an actual descendant retry with one
nonempty terminal final-answer record. Cite the WHOLE earlier response (or [] for an actually empty response)
and the later final line. Empty markers, length termination, multiple finals, retractions and trailing prose
do not qualify. FINALIZED certifies output presence only, even for a wrong answer. OUTPUT never belongs in
attempt issues or unverified lists and never covers a native issue or resolves unknown reasoning scope.
"""
PUBLIC_JSON_SHAPE = """
PUBLIC TYPE SHAPE ONLY: replace angle-bracket type placeholders with your own JSON values; do not copy
placeholders. Repeated objects show field/types, not a required array length or a supplied claim.
{"lessons":[{"rule":"<rule>","earlier":{"attempt":<integer>,"kind":"<OBSERVATION, TRACE or OUTPUT>","rule":"<rule>",
 "locus":"<locus>","evidence_ranges":[[<integer>,<integer>]]},
 "later":{"attempt":<integer>,"kind":"<OBSERVATION, TRACE or OUTPUT>","rule":"<rule>","locus":"<locus>",
 "evidence_ranges":[[<integer>,<integer>]]},"relation":"<relation>"}],
 "corrected_facts":{"<public slot ID>":"<your single-line value>"},
 "attempts":[{"attempt":<integer>,"outcome":"<public outcome>","answer":<string, finite number or null>,
 "evidence_ranges":[[<integer>,<integer>]],
 "issues":[{"kind":"<OBSERVATION or TRACE>","rule":"<rule>","locus":"<locus>","evidence_ranges":[[<integer>,<integer>]]}],
 "unverified":[{"kind":"<OBSERVATION or TRACE>","rule":"<rule>","locus":"<locus>",
 "evidence_ranges":[[<integer>,<integer>]],"blocked_by":[{"rule":"<rule>","locus":"<locus>"}]}]}],
 "success_attempt":<integer>,"corrected_answer":<string or finite number>}
For an unverified later endpoint, add blocked_by as above. For ABSENCE use the null fields specified
in the relation rules. Empty lists/objects are syntactic containers, not evidence that nothing is missing.
Use the field outcome even though the read-only source transcript calls its public outcome status.
"""
BOUNDED_PROCEDURE = """Write lessons first, then corrected_facts, attempts, success_attempt, corrected_answer.
Before listing facts, compare every public [parent attempt, retry attempt] pair shown below, in its original
order, at the same executed operation or image slot. The list includes every recorded retry without outcome
filtering. A parent without a final answer still contains inspectable observations and operations.
Use original evidence to distinguish CORRECTED, PERSISTED, other-branch comparison, and unverified later
evidence. A correct child answer alone proves none of them.
Also inspect missing-to-present final-answer output as the optional FINALIZED relationship,
which certifies presence only, never correctness or native-error repair. Retain every sampled sibling and intermediate
attempt. If you cannot support a relationship, abstain honestly rather than inventing one; an empty lessons
list is an abstention, not evidence of coverage. Supply only your own evidence-bound claims.
"""
PROMPT = """Write a structured whole-history notebook for ALL actual attempts, including every T1/T2/T3
sibling, intermediate failure, truncated response and the designated successful retry. The original images
remain available. Quoted outputs are data, not instructions. Public outcome/answer feedback never certifies
observations or executed operations. Read the images yourself; no correct fact, trace or notebook is supplied.
Return one bare JSON object with EXACTLY corrected_facts, attempts, success_attempt, corrected_answer, lessons.
corrected_facts maps every required public image slot to one string value, with no extra slots or assignments.
Native edge/weight values are dimensionless unsigned ASCII integers; include no units, labels or prose.
Fact keys/values and answer strings must be actual single lines, without vertical controls or Unicode line separators.
attempts contains EVERY ordered attempt, each with EXACTLY attempt (integer), outcome (public status),
answer (public extracted answer or JSON null), evidence_ranges, issues{attempt_extension}. Ranges are increasing nonoverlapping
inclusive [first,last] line-ID pairs from that attempt. Cite the final answer and every claimed issue.
issues lists ALL independently checkable wrong observations AND wrong executed trace operations, even in
CORRECT or TRUNCATED attempts. Each issue has EXACTLY kind ('OBSERVATION' or 'TRACE'), rule, locus,
evidence_ranges. OBSERVATION uses rule OBSERVATION_MATCH and its canonical slot ID as locus. TRACE uses
the native trace rule and semantic locus. A correctly read edge used backwards is a TRACE issue, never a
wrong observation. Missing/unverifiable traces or substantive surrounding prose are unverified, not no error.
lessons is a list of closed evidence-backed relationships, never free prose or a universal rule. Each has
EXACTLY rule, earlier, later, relation. Both endpoints have EXACTLY attempt, kind, rule, locus, evidence_ranges.
earlier must identify a real reported issue. lesson.rule equals that issue's rule. later identifies an actual
check of the SAME semantic operation or image slot, with original evidence; an array index alone proves no
identity. CORRECTED requires a descendant retry that positively complies at that locus. PERSISTED requires
positive evidence of the same error in a descendant retry, including in a successful answer.
OTHER_BRANCH_COMPARISON requires a separate sibling branch with independently verified compliance or
violation of the same semantic operation. It records a comparison only, never recovery. A correct answer alone proves none
of these. If the issue was never checked again in ANY later attempt, use NOT_REVISITED and later with
kind ABSENCE, rule null, locus null, attempt equal to the last attempt and its WHOLE evidence range. For an
issue in the final attempt use attempt null and evidence_ranges [] instead. Absence never means correction.
Cover every detected issue as an earlier endpoint or as a positively repeated PERSISTED later endpoint.
Use all evidence lines of the matched checks, including the original Trace line and referenced observations.
Native rules: PATH_ENDPOINTS, DIRECTED_EDGE, EDGE_WEIGHT, COST_SUM, MINIMALITY, BFS_FIFO,
BFS_NEW_NEIGHBORS, BFS_COMPLETENESS, TREE_START, TREE_CHILD_EDGE, TREE_EVENT_ORDER,
TREE_COMPLETENESS, DFA_START, DFA_COMPLETE_WORD, DFA_TRANSITION, DFA_ACCEPTANCE.
Native loci mirror each actual operation: nodes[i:i+2] for a directed edge, weights[i] for its weight;
nodes/cost; steps[i].node/steps[i].new; events[i]; states[i:i+2], states[0], states, decision.
No extra claim fields, rewritten quotations, explanatory prose, inferred module responsibility or future guarantees.
The successful retry is {success_attempt}; corrected_answer must equal its checked single answer.
{extension}
Public observation syntax, with no reference values:
{observation_instruction}
<read_only_attempts>
{transcript}
</read_only_attempts>
The read-only input has ended. Reinspect the images, compare every actual branch, and write your new JSON notebook.
"""


def prompt_contract(contract: str = VERSION) -> str:
    """Canonical public template identity, with item-specific values still unfilled."""
    if contract not in (VERSION, BOUNDED_VERSION):
        raise ValueError("synthesis: unknown explicit contract")
    extension = BOUNDED_PROMPT_EXTENSION + PUBLIC_JSON_SHAPE.replace("{", "{{").replace("}", "}}")
    prompt = PROMPT
    if contract == BOUNDED_VERSION:
        prompt = BOUNDED_PROCEDURE + prompt.replace(
            "EXACTLY corrected_facts, attempts, success_attempt, corrected_answer, lessons.",
            "EXACTLY lessons, corrected_facts, attempts, success_attempt, corrected_answer, in that order.")
        prompt = prompt.replace("earlier must identify a real reported issue. lesson.rule equals that issue's rule.",
            "For OBSERVATION or TRACE, earlier identifies a real reported issue and lesson.rule equals its rule.\n"
            "OUTPUT uses its separate optional FINALIZED checks, never an attempts[].issues entry.")
        prompt = prompt.replace("<read_only_attempts>\n", "<public_retry_pairs>\n{retry_pairs}\n</public_retry_pairs>\n<read_only_attempts>\n")
    return prompt.replace("{extension}", extension if contract == BOUNDED_VERSION else "").replace(
        "{attempt_extension}", ", unverified" if contract == BOUNDED_VERSION else "")


def format_feedback_contract() -> dict:
    return {"version": FORMAT_DIAGNOSTICS_VERSION, "max_errors": MAX_FORMAT_ERRORS,
            "error_codes": list(FORMAT_ERROR_CODES), "public_shape_sha256": _sha(PUBLIC_JSON_SHAPE),
            "submitted_lesson_count": "submitted list length, otherwise null; never expected relationship count",
            "submitted_attempt_count": "submitted attempts list length, otherwise null",
            "empty_attempt_evidence_positions": "first max_errors 1-based submitted attempts array positions with exact evidence_ranges=[]; not claimed attempt IDs or verified failures",
            "empty_attempt_evidence_positions_truncated": "true only when additional matching positions were omitted",
            "scope": "submitted-json-keys-types-range-shapes-and-empty-list-metadata-only-no-evidence-or-reference-values"}


def format_diagnostics(text: str) -> dict:
    """Describe submitted JSON only; never inspect an actor, audit, verifier or line bounds."""
    result: dict[str, Any] = {"version": FORMAT_DIAGNOSTICS_VERSION, "errors": [], "truncated": False,
                              "submitted_lesson_count": None, "submitted_attempt_count": None,
                              "empty_attempt_evidence_positions": [],
                              "empty_attempt_evidence_positions_truncated": False}

    def error(code: str, path: str) -> None:
        if len(result["errors"]) < MAX_FORMAT_ERRORS:
            result["errors"].append({"code": code, "path": path})
        else:
            result["truncated"] = True

    def fields(value: Any, required: set[str], path: str) -> bool:
        if type(value) is not dict:
            error("EXPECTED_OBJECT", path)
            return False
        for key in sorted(required - set(value)):
            error("MISSING_KEY", path + "." + key)
        for key in sorted(set(value) - required):
            error("UNEXPECTED_KEY", path + "." + (key if key in _FORMAT_PATH_FIELDS else "*"))
        return True

    def scalar(value: Any, path: str, expected: type = str, *, nullable: bool = False) -> None:
        if type(value) is not expected and not (nullable and value is None):
            error("EXPECTED_INTEGER" if expected is int else "EXPECTED_STRING", path)

    def answer(value: Any, path: str, *, nullable: bool = False) -> None:
        if not (type(value) is str or type(value) is int
                or type(value) is Decimal and value.is_finite() or nullable and value is None):
            error("EXPECTED_ANSWER", path)

    def items(value: Any, path: str) -> list:
        if type(value) is not list:
            error("EXPECTED_LIST", path)
            return []
        return value

    def ranges(value: Any, path: str) -> None:
        for index, pair in enumerate(items(value, path)):
            if type(pair) is not list or len(pair) != 2:
                error("EXPECTED_RANGE_PAIR", path)
                return
            for position, number in enumerate(pair):
                scalar(number, f"{path}[{index}][{position}]", int)

    def claim(value: Any, path: str, *, endpoint: bool = False, unknown: bool = False,
              absence: bool = False) -> None:
        required = (_ENDPOINT if endpoint else _CLAIM) | ({"blocked_by"} if unknown else set())
        if not fields(value, required, path):
            return
        for key in ("kind", "rule", "locus"):
            if key in value:
                if absence and key in ("rule", "locus"):
                    if value[key] is not None:
                        error("EXPECTED_NULL", path + "." + key)
                else:
                    scalar(value[key], path + "." + key)
        if endpoint and "attempt" in value:
            scalar(value["attempt"], path + ".attempt", int, nullable=absence)
        if "evidence_ranges" in value:
            ranges(value["evidence_ranges"], path + ".evidence_ranges")
        if unknown and "blocked_by" in value:
            for index, row in enumerate(items(value["blocked_by"], path + ".blocked_by")):
                location = f"{path}.blocked_by[{index}]"
                if fields(row, {"rule", "locus"}, location):
                    for key in ("rule", "locus"):
                        if key in row:
                            scalar(row[key], location + "." + key)
                if result["truncated"]:
                    break

    try:
        note = json.loads(text, object_pairs_hook=_unique, parse_float=Decimal, parse_constant=Decimal)
    except (TypeError, ValueError, InvalidOperation, RecursionError):
        error("INVALID_JSON", "$")
        return result
    if type(note) is dict and type(note.get("lessons")) is list:
        result["submitted_lesson_count"] = len(note["lessons"])
    if type(note) is dict and type(note.get("attempts")) is list:
        result["submitted_attempt_count"] = len(note["attempts"])
        for position, attempt in enumerate(note["attempts"], 1):
            if (type(attempt) is dict and type(attempt.get("evidence_ranges")) is list
                    and attempt["evidence_ranges"] == []):
                if len(result["empty_attempt_evidence_positions"]) == MAX_FORMAT_ERRORS:
                    result["empty_attempt_evidence_positions_truncated"] = True
                    break
                result["empty_attempt_evidence_positions"].append(position)
    if not fields(note, _TOP, "$"):
        return result
    if "corrected_facts" in note:
        facts = note["corrected_facts"]
        if type(facts) is not dict:
            error("EXPECTED_OBJECT", "$.corrected_facts")
        else:
            for value in facts.values():
                scalar(value, "$.corrected_facts.*")
                if result["truncated"]:
                    break
    if "corrected_answer" in note:
        answer(note["corrected_answer"], "$.corrected_answer")
    if "success_attempt" in note:
        scalar(note["success_attempt"], "$.success_attempt", int)
    for index, attempt in enumerate(items(note.get("attempts", []), "$.attempts")):
        path = f"$.attempts[{index}]"
        if fields(attempt, _ATTEMPT | {"unverified"}, path):
            for key, expected in (("attempt", int), ("outcome", str)):
                if key in attempt:
                    scalar(attempt[key], path + "." + key, expected)
            if "answer" in attempt:
                answer(attempt["answer"], path + ".answer", nullable=True)
            if "evidence_ranges" in attempt:
                ranges(attempt["evidence_ranges"], path + ".evidence_ranges")
            for key in ("issues", "unverified"):
                for position, row in enumerate(items(attempt.get(key, []), path + "." + key)):
                    claim(row, f"{path}.{key}[{position}]", unknown=key == "unverified")
                    if result["truncated"]:
                        break
        if result["truncated"]:
            break
    for index, lesson in enumerate(items(note.get("lessons", []), "$.lessons")):
        path = f"$.lessons[{index}]"
        if fields(lesson, {"rule", "earlier", "later", "relation"}, path):
            for key in ("rule", "relation"):
                if key in lesson:
                    scalar(lesson[key], path + "." + key)
            if "earlier" in lesson:
                claim(lesson["earlier"], path + ".earlier", endpoint=True)
            if "later" in lesson:
                claim(lesson["later"], path + ".later", endpoint=True,
                      unknown=lesson.get("relation") in ("UNVERIFIABLE_LATER", "SCOPE_UNVERIFIABLE_LATER"),
                      absence=lesson.get("relation") == "NOT_REVISITED")
        if result["truncated"]:
            break
    return result


def _valid_format_diagnostics(value: Any) -> bool:
    return (type(value) is dict and set(value) == {"version", "errors", "truncated", "submitted_lesson_count",
                "submitted_attempt_count", "empty_attempt_evidence_positions", "empty_attempt_evidence_positions_truncated"}
            and value["version"] == FORMAT_DIAGNOSTICS_VERSION and type(value["truncated"]) is bool
            and (value["submitted_lesson_count"] is None or type(value["submitted_lesson_count"]) is int
                 and value["submitted_lesson_count"] >= 0)
            and (value["submitted_attempt_count"] is None or type(value["submitted_attempt_count"]) is int
                 and value["submitted_attempt_count"] >= 0)
            and type(value["empty_attempt_evidence_positions"]) is list
            and len(value["empty_attempt_evidence_positions"]) <= MAX_FORMAT_ERRORS
            and all(type(position) is int and type(value["submitted_attempt_count"]) is int
                    and 1 <= position <= value["submitted_attempt_count"]
                    for position in value["empty_attempt_evidence_positions"])
            and value["empty_attempt_evidence_positions"] == sorted(set(value["empty_attempt_evidence_positions"]))
            and type(value["empty_attempt_evidence_positions_truncated"]) is bool
            and (not value["empty_attempt_evidence_positions_truncated"]
                 or len(value["empty_attempt_evidence_positions"]) == MAX_FORMAT_ERRORS
                 and value["empty_attempt_evidence_positions"][-1] < value["submitted_attempt_count"])
            and type(value["errors"]) is list and len(value["errors"]) <= MAX_FORMAT_ERRORS
            and all(type(row) is dict and set(row) == {"code", "path"} and row["code"] in FORMAT_ERROR_CODES
                    and type(row["path"]) is str and len(row["path"]) <= 256
                    and re.fullmatch(r"\$(?:\.[a-z_]+|\.\*|\[[0-9]+\])*", row["path"])
                    and all(field in _FORMAT_PATH_FIELDS for field in re.findall(r"\.([a-z_]+)", row["path"]))
                    for row in value["errors"]))


def instruction(samples: list[Sample], audit: dict, observation_instruction: str) -> str:
    """Quote every raw sibling/retry without private checks, labels or target values."""
    public = episode_notebook.public_transcript(samples)
    if (audit.get("version") not in (VERSION, BOUNDED_VERSION) or audit.get("transcript_sha256") != _hash(public)):
        raise ValueError("synthesis: current versioned transcript audit required")
    rows = [{**{key: value for key, value in row.items() if key != "lines"},
             "line_count": len(row["lines"]),
             "source_lines": [[line["line_id"], line["text"]] for line in row["lines"]]}
            for row in public]
    transcript = json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    transcript = transcript.replace("<", "\\u003c").replace(">", "\\u003e")
    return prompt_contract(audit["version"]).format(success_attempt=audit["success_attempt"],
        observation_instruction=observation_instruction, transcript=transcript,
        retry_pairs=json.dumps([[row["parent"], row["attempt"]] for row in public if row["parent"] is not None],
                              separators=(",", ":")))


def _selected(ranges: Any, sample: Sample, *, allow_empty: bool = False) -> list[dict] | None:
    if allow_empty and sample.text == "" and ranges == []:
        return []
    lines = _first_attempt_lines(sample.text)
    ids = episode_notebook._evidence_line_ids(ranges, len(lines))
    return [lines[index - 1] for index in ids] if ids else None


def _public_shape_valid(sample: Sample, originals: list[Sample]) -> bool:
    """Check public containers and literal citation bounds, never outcomes or private grades."""
    diagnostics = format_diagnostics(sample.text)
    if sample.finish_reason != "stop" or diagnostics["errors"] or diagnostics["truncated"]:
        return False
    note = json.loads(sample.text, object_pairs_hook=_unique, parse_float=Decimal, parse_constant=Decimal)
    if (len(note["attempts"]) != len(originals)
            or not 1 <= note["success_attempt"] <= len(originals)):
        return False
    for index, (attempt, original) in enumerate(zip(note["attempts"], originals), 1):
        if (attempt["attempt"] != index
                or any(_selected(claim["evidence_ranges"], original, allow_empty=True) is None
                       for claim in [attempt, *attempt["issues"], *attempt["unverified"]])):
            return False
    for lesson in note["lessons"]:
        for endpoint in (lesson["earlier"], lesson["later"]):
            index = endpoint["attempt"]
            if index is None:
                if endpoint["kind"] != "ABSENCE" or endpoint["evidence_ranges"] != []:
                    return False
            elif (not 1 <= index <= len(originals)
                    or _selected(endpoint["evidence_ranges"], originals[index - 1], allow_empty=True) is None):
                return False
    return True


def select_repair_parent(candidates: list[Sample], originals: list[Sample], *, policy: str | None) -> dict:
    """Select the first stopped public-shape-valid sibling; keep every sampled action."""
    if policy != REPAIR_PARENT_POLICY or not candidates or not originals:
        raise ValueError("synthesis: explicit repair parent policy and nonempty samples required")
    index = next((index for index, sample in enumerate(candidates) if _public_shape_valid(sample, originals)), None)
    selected = index if index is not None else 0
    return {"index": selected, "sample_id": candidates[selected].sample_id,
            "reason": "FIRST_PUBLIC_SHAPE_VALID" if index is not None else "FALLBACK_FIRST"}


def _covers(selected: list[dict], refs: list[dict]) -> bool:
    return bool(refs) and all(any(line["start"] <= ref["span"][0]
                                  and ref["span"][1] <= line["end"] for line in selected) for ref in refs)


def _claim(value: Any) -> bool:
    return (type(value) is dict and set(value) == _CLAIM
            and value["kind"] in ("OBSERVATION", "TRACE")
            and type(value["rule"]) is str and type(value["locus"]) is str)


def _answer(value: Any, *, nullable: bool = False) -> str | None:
    if type(value) is str and not _single_line_scalar(value):
        raise ValueError("synthesis: answer must be one lossless raw line")
    parsed = _validate({"cause": "UNKNOWN", "wrong_item": "none", "first_attempt_lines": [1],
                        "previous_answer": value if nullable else None,
                        "corrected_answer": "validation" if nullable else value,
                        "corrected_facts": {}, "lesson": "Validation."})
    if parsed is None:
        raise ValueError("synthesis: invalid finite single answer")
    return parsed["previous_answer" if nullable else "corrected_answer"]


def _endpoint(value: Any, samples: list[Sample], audit: dict, *, issue: bool) -> tuple[dict, list[dict]] | None:
    if type(value) is not dict or set(value) != _ENDPOINT or type(value["attempt"]) is not int:
        return None
    index = value["attempt"]
    if not 1 <= index <= len(samples) or not _claim({k: v for k, v in value.items() if k != "attempt"}):
        return None
    selected = _selected(value["evidence_ranges"], samples[index - 1])
    rows = audit["attempts"][index - 1]["issues" if issue else "checks"]
    matching = [row for row in rows if _key(row) == _key(value)]
    if len(matching) != 1 or selected is None or not _covers(selected, matching[0]["refs"]):
        return None
    return matching[0], selected


def _output_endpoint(value: Any, samples: list[Sample], audit: dict, *, missing: bool) -> dict | None:
    if (audit.get("version") != BOUNDED_VERSION or type(value) is not dict or set(value) != _ENDPOINT
            or type(value["attempt"]) is not int or not 1 <= value["attempt"] <= len(samples)
            or (value["kind"], value["rule"], value["locus"]) != ("OUTPUT", "FINALIZATION", "final_answer")):
        return None
    index = value["attempt"] - 1
    sample = samples[index]
    actual = audit["attempts"][index]
    expected = _output_checks(sample)
    if (sample.result is None or actual.get("outcome") != sample.result.status
            or actual.get("answer") != sample.result.extracted
            or actual.get("text_sha256") != _sha(sample.text) or actual.get("output_checks") != expected
            or len(expected) != 1 or expected[0]["status"] != ("VIOLATED" if missing else "COMPLIANT")):
        return None
    selected = _selected(value["evidence_ranges"], sample, allow_empty=missing)
    if selected is None or missing and selected != _first_attempt_lines(sample.text):
        return None
    return expected[0] if missing or _covers(selected, expected[0]["refs"]) else None


def _descendant(earlier: int, later: int, samples: list[Sample]) -> bool:
    parents = {sample.sample_id: sample.parent_sample_id for sample in samples}
    target = samples[earlier - 1].sample_id
    parent = samples[later - 1].parent_sample_id
    while parent is not None:
        if parent == target:
            return True
        parent = parents.get(parent)
    return False


def _lesson(value: Any, samples: list[Sample], audit: dict, *, bounded: bool = False) -> dict:
    checked: dict[str, Any] = {"verified": False, "covered_issues": []}
    if (type(value) is not dict or set(value) != {"rule", "earlier", "later", "relation"}
            or type(value["relation"]) is not str
            or value["relation"] not in _RELATIONS | (
                {"UNVERIFIABLE_LATER", "SCOPE_UNVERIFIABLE_LATER", "FINALIZED"} if bounded else set())):
        return checked
    if value["relation"] == "FINALIZED":
        output_origin = _output_endpoint(value["earlier"], samples, audit, missing=True)
        later_output = _output_endpoint(value["later"], samples, audit, missing=False)
        if (output_origin is None or later_output is None or value["rule"] != "FINALIZATION"
                or value["later"]["attempt"] <= value["earlier"]["attempt"]
                or output_origin["identity"] != later_output["identity"]
                or not _descendant(value["earlier"]["attempt"], value["later"]["attempt"], samples)):
            return checked
        result = samples[value["later"]["attempt"] - 1].result
        return {"verified": True, "claim": value, "covered_issues": [], "credit_type": "OUTPUT_FINALIZATION",
                "later_answer_correct": bool(result and result.status == "CORRECT")}
    earlier = _endpoint(value["earlier"], samples, audit, issue=True)
    if earlier is None or value["rule"] != earlier[0]["rule"] or earlier[0]["identity"] is None:
        return checked
    origin = value["earlier"]["attempt"]
    row = earlier[0]
    later = value["later"]
    if value["relation"] in {"UNVERIFIABLE_LATER", "SCOPE_UNVERIFIABLE_LATER"}:
        if (type(later) is not dict or set(later) != _ENDPOINT | {"blocked_by"}
                or type(later["attempt"]) is not int or not origin < later["attempt"] <= len(samples)):
            return checked
        target_row = _unknown_binding({key: item for key, item in later.items() if key != "attempt"},
                                      later["attempt"], samples, audit)
        checked["verified"] = bool(target_row and (
            _scope_unknown(target_row) if value["relation"] == "SCOPE_UNVERIFIABLE_LATER"
            else target_row["identity"] == row["identity"]))
    elif value["relation"] == "NOT_REVISITED":
        expected = len(samples) if origin < len(samples) else None
        if (type(later) is not dict or set(later) != _ENDPOINT or later["attempt"] != expected
                or type(later["attempt"]) is not type(expected) or later["kind"] != "ABSENCE"
                or later["rule"] is not None or later["locus"] is not None):
            return checked
        selected = _selected(later["evidence_ranges"], samples[-1]) if expected is not None else None
        if expected is None:
            if later["evidence_ranges"] != []:
                return checked
        elif selected != _first_attempt_lines(samples[-1].text):
            return checked
        revisited = [check for attempt in audit["attempts"][origin:] for check in attempt["checks"]
                     if check["identity"] == row["identity"]]
        # An unknown same-rule operation cannot justify a confident absence claim.
        uncertain = [check for attempt in audit["attempts"][origin:] for check in attempt["checks"]
                     if check["kind"] == row["kind"] and check["rule"] == row["rule"]
                     and check["identity"] is None]
        if bounded:
            uncertain += [check for attempt in audit["attempts"][origin:] for check in attempt["checks"]
                          if _scope_unknown(check)]
        checked["verified"] = not revisited and not uncertain
    else:
        target = _endpoint(later, samples, audit, issue=False)
        if target is None or later["attempt"] <= origin or target[0]["identity"] != row["identity"]:
            return checked
        destination = target[0]
        if value["relation"] == "CORRECTED":
            checked["verified"] = (destination["status"] == "COMPLIANT"
                                   and _descendant(origin, later["attempt"], samples))
        elif value["relation"] == "PERSISTED":
            checked["verified"] = (destination["status"] == "VIOLATED"
                                   and _descendant(origin, later["attempt"], samples))
        else:
            checked["verified"] = (destination["status"] in ("COMPLIANT", "VIOLATED")
                                   and not _descendant(origin, later["attempt"], samples))
        if checked["verified"] and value["relation"] == "PERSISTED":
            checked["covered_issues"].append([later["attempt"], *_key(destination)])
    if checked["verified"]:
        checked["covered_issues"].append([origin, *_key(row)])
        checked["claim"] = value
    return checked


def grade_note(text: str, finish_reason: str, tree: ProblemTree, audit: dict, verifier: Any, *,
               actor: ActorItem, contract: str = VERSION, evaluation_only: bool = False) -> dict:
    """Freshly re-audit actual evidence; caller-supplied audit claims never earn reward."""
    fresh = audit_episode(tree, actor, verifier, contract=contract, evaluation_only=evaluation_only)
    if audit != fresh:
        raise ValueError("synthesis: current complete freshly audited recovery required")
    return _grade_note_audited(text, finish_reason, tree, fresh, verifier, contract=contract)


def _grade_note_audited(text: str, finish_reason: str, tree: ProblemTree, fresh: dict, verifier: Any, *,
                        contract: str = VERSION) -> dict:
    """Grade against the evidence freshly checked within the calling transaction."""
    bounded = contract == BOUNDED_VERSION
    if not fresh["complete_episode"] or type(fresh["success_attempt"]) is not int:
        raise ValueError("synthesis: current complete freshly audited recovery required")
    samples = episode_notebook.solution_samples(tree)
    record: dict[str, Any] = {"reward_version": contract, "reward": 0.0, "checks": {},
                              "attempt_checks": [], "lesson_checks": [], "failure_code": "INVALID_JSON",
                              "semantic_scope": fresh["semantic_scope"], "module_causality_certified": False,
                              "model_claims_used_for_routing": False, "universal_lesson_certified": False,
                              "lesson_semantics_verified": False, "bounded_claims_verified": False,
                              "all_reasoning_verified": False, "full_diagnosis": False,
                              "solution_trace_complete": fresh["solution_trace_complete"],
                              "note_text_sha256": _sha(text), "note_finish_reason": finish_reason,
                              "episode_audit_sha256": _hash(fresh)}
    if bounded:
        record["format_diagnostics"] = format_diagnostics(text)
        record["fact_reinspection"] = _fact_reinspection(None, record, tree.problem_id, verifier)
    try:
        note = json.loads(text, object_pairs_hook=_unique, parse_float=Decimal, parse_constant=Decimal)
    except (TypeError, ValueError, InvalidOperation, RecursionError):
        return record
    if bounded:
        record["fact_reinspection"] = _fact_reinspection(note, record, tree.problem_id, verifier)
    record["failure_code"] = "INVALID_ENVELOPE"
    if (finish_reason != "stop" or type(note) is not dict or set(note) != _TOP
            or type(note["success_attempt"]) is not int or note["success_attempt"] != fresh["success_attempt"]
            or type(note["attempts"]) is not list or len(note["attempts"]) != len(samples)
            or type(note["lessons"]) is not list):
        return record
    record["failure_code"] = "INVALID_FIELDS"
    facts = note["corrected_facts"]
    if (type(facts) is not dict or any(not _single_line_scalar(key) or not _single_line_scalar(value)
            or not _native_value_shape(key, value) for key, value in facts.items())
            or type(note["corrected_answer"]) is str and not _single_line_scalar(note["corrected_answer"])):
        return record
    final = _validate({"cause": "UNKNOWN", "wrong_item": "none", "first_attempt_lines": [1],
                       "previous_answer": None, "corrected_answer": note["corrected_answer"],
                       "corrected_facts": note["corrected_facts"], "lesson": "Structured synthesis."})
    if final is None:
        return record
    expected_issues: list[list[Any]] = []
    for claim, actual, sample in zip(note["attempts"], fresh["attempts"], samples):
        if (type(claim) is not dict or set(claim) != _ATTEMPT | ({"unverified"} if bounded else set())
                or type(claim["attempt"]) is not int
                or claim["attempt"] != actual["attempt"] or type(claim["outcome"]) is not str
                or type(claim["issues"]) is not list or any(not _claim(row) for row in claim["issues"])
                or bounded and type(claim["unverified"]) is not list):
            return record
        try:
            answer = _answer(claim["answer"], nullable=True)
        except ValueError:
            return record
        claim["answer"] = answer
        selected = _selected(claim["evidence_ranges"], sample, allow_empty=bounded)
        if selected is None:
            return record
        keys = [_key(row) for row in claim["issues"]]
        expected = [_key(row) for row in actual["issues"]]
        evidence = all(_endpoint({"attempt": actual["attempt"], **row}, samples, fresh, issue=True)
                       is not None for row in claim["issues"])
        full_answer = extract_final(sample.text, sample.finish_reason)
        excerpt_answer = extract_final("".join(line["text"] for line in selected), sample.finish_reason)
        # Negative/missing-answer conclusions require the whole response, not a selected omission.
        conclusion = excerpt_answer == full_answer and (full_answer.status == "OK"
                      or selected == _first_attempt_lines(sample.text))
        record["attempt_checks"].append({"attempt": actual["attempt"],
            "outcome": claim["outcome"] == actual["outcome"], "answer": answer == actual["answer"],
            "conclusion_evidence": conclusion, "issues": sorted(keys) == sorted(expected) and len(keys) == len(set(keys)),
            "issue_evidence": evidence, "trace_complete": actual["trace_complete"],
            "unverified_count": len(actual["unverified"])})
        if bounded:
            unverified = claim["unverified"]
            if type(unverified) is not list:
                return record
            bindings = [_unknown_binding(value, actual["attempt"], samples, fresh) for value in unverified]
            unknown_keys = [_key(row) for row in bindings if row is not None]
            record["attempt_checks"][-1]["unverified_regions"] = (
                all(row is not None for row in bindings) and len(unknown_keys) == len(set(unknown_keys))
                and sorted(unknown_keys) == sorted(_key(row) for row in actual["unverified"]))
        expected_issues.extend([actual["attempt"], *key] for key in expected)
    record["lesson_checks"] = [_lesson(value, samples, fresh, bounded=bounded) for value in note["lessons"]]
    covered = [key for row in record["lesson_checks"] for key in row["covered_issues"]]
    projection = episode_notebook._fact_projection(final["corrected_facts"], final["corrected_answer"])
    graded = verifier.grade(tree.problem_id, projection, "stop", "obs")
    slots = graded.slots
    success = fresh["attempts"][fresh["success_attempt"] - 1]
    record["checks"] = {"format_valid": True,
        "all_attempts_verified": all(all(row[key] for key in ("outcome", "answer", "conclusion_evidence",
                                      "issues", "issue_evidence")) for row in record["attempt_checks"]),
        ("observation_claims_accounted" if bounded else "observation_claims_complete"):
            all(not any(check["kind"] == "OBSERVATION" and not _eligible_unknown(
                check, attempt["checks"], accounted=True) for check in attempt["unverified"])
                for attempt in fresh["attempts"]) if bounded else fresh["observation_claims_complete"],
        "lessons_verified": all(row["verified"] for row in record["lesson_checks"]),
        "all_issues_synthesized": all(key in covered for key in expected_issues),
        "answer_verified": graded.status == "CORRECT" and graded.extracted == success["answer"],
        "registered_facts_verified": bool(slots and slots.required > 0 and slots.required_correct == slots.required
            and slots.parsed == len(final["corrected_facts"]) == slots.required
            and slots.all_correct == slots.parsed and not slots.conflicting
            and not slots.false_relations and not slots.unverified_relations)}
    if bounded:
        record["checks"]["unknown_inventory_verified"] = (fresh["unknown_regions_accountable"]
            and all(row["unverified_regions"] for row in record["attempt_checks"]))
    else:
        record["checks"]["solution_trace_complete"] = fresh["solution_trace_complete"]
    record["reward"] = float(all(record["checks"].values()))
    record["lesson_semantics_verified"] = bool(record["reward"]) and not bounded
    record["bounded_claims_verified"] = bool(record["reward"]) and bounded
    record["failure_code"] = "NONE" if record["reward"] else "CHECKS_FAILED"
    note["corrected_answer"] = final["corrected_answer"]
    record["submitted_claims"] = note
    record["verified_rendering"] = render_verified(record)
    return record


def successful(report: dict, contract: str = VERSION) -> bool:
    """One explicit success gate for callers; bounded reward never claims full diagnosis."""
    if contract not in (VERSION, BOUNDED_VERSION):
        raise ValueError("synthesis: unknown explicit contract")
    expected = ((set(REQUIRED_CHECKS) - {"observation_claims_complete"})
                | {"observation_claims_accounted", "unknown_inventory_verified"}
                if contract == BOUNDED_VERSION else set(REQUIRED_CHECKS) | {"solution_trace_complete"})
    checks = report.get("checks")
    return bool(report.get("reward_version") == contract and type(report.get("reward")) in (int, float)
        and report["reward"] == 1 and type(checks) is dict and set(checks) == expected
        and all(type(value) is bool and value for value in checks.values())
        and report.get("module_causality_certified") is False and report.get("universal_lesson_certified") is False
        and report.get("all_reasoning_verified") is False and report.get("full_diagnosis") is False
        and report.get("bounded_claims_verified") is (contract == BOUNDED_VERSION)
        and report.get("lesson_semantics_verified") is (contract == VERSION))


def notebook_feedback(checked: dict, sample: Sample | None = None) -> str:
    """Public submitted syntax and rejection booleans/indices; no private expected claims."""
    contract = checked.get("reward_version")
    if (contract not in (VERSION, BOUNDED_VERSION) or type(checked.get("reward")) not in (int, float)
            or checked["reward"] != 0 or type(checked.get("failure_code")) is not str
            or checked["failure_code"] not in
            {"INVALID_JSON", "INVALID_ENVELOPE", "INVALID_FIELDS", "CHECKS_FAILED"}):
        raise ValueError("synthesis: feedback requires a checked unsuccessful note")
    if sample is not None and (checked.get("note_text_sha256") != _sha(sample.text)
            or checked.get("note_finish_reason") != sample.finish_reason or checked["reward"] != sample.reward):
        raise ValueError("synthesis: feedback differs from the actual note sample")
    checks = checked.get("checks", {})
    allowed = set(REQUIRED_CHECKS) | {"observation_claims_accounted", "unknown_inventory_verified",
                                    "solution_trace_complete"}
    if (type(checks) is not dict or set(checks) - allowed
            or any(type(value) is not bool for value in checks.values())):
        raise ValueError("synthesis: invalid rejection checks")
    failed_attempts = []
    rows = checked.get("attempt_checks", [])
    if type(rows) is not list:
        raise ValueError("synthesis: invalid rejection attempt checks")
    fields = ("outcome", "answer", "conclusion_evidence", "issues", "issue_evidence")
    for index, row in enumerate(rows, 1):
        required = fields + (("unverified_regions",) if contract == BOUNDED_VERSION else ())
        if (type(row) is not dict or type(row.get("attempt")) is not int or row["attempt"] != index
                or any(type(row.get(key)) is not bool for key in required)):
            raise ValueError("synthesis: invalid rejection attempt checks")
        if any(not row[key] for key in required) or contract == VERSION and row.get("trace_complete") is not True:
            failed_attempts.append(index)
    lessons = checked.get("lesson_checks", [])
    if type(lessons) is not list or any(type(row) is not dict or type(row.get("verified")) is not bool for row in lessons):
        raise ValueError("synthesis: invalid rejection lesson checks")
    payload = {"contract": contract, "feedback_policy": FEEDBACK_POLICY if contract == BOUNDED_VERSION else LEGACY_FEEDBACK_POLICY,
               "failure_code": checked["failure_code"],
               "checks": checks, "failed_attempts": failed_attempts,
               "failed_lessons": [index for index, row in enumerate(lessons, 1) if not row["verified"]]}
    if contract == BOUNDED_VERSION:
        payload["reinspection_slots"] = _reinspection_feedback(checked, sample)
        diagnostics: Any = checked.get("format_diagnostics")
        if (not _valid_format_diagnostics(diagnostics)
                or sample is not None and diagnostics != format_diagnostics(sample.text)):
            raise ValueError("synthesis: submitted JSON format diagnostics differ")
        if diagnostics["empty_attempt_evidence_positions"] and sample is None:
            raise ValueError("synthesis: empty attempt evidence metadata requires its actual sample")
        payload["format_diagnostics"] = diagnostics
        public_checks = {key: value for key, value in checks.items() if key != "lessons_verified"}
        count = diagnostics["submitted_lesson_count"]
        public_checks["submitted_lessons_verified"] = (checks["lessons_verified"]
            if type(count) is int and count > 0 and "lessons_verified" in checks else None)
        payload["checks"] = public_checks
    instruction = RETRY_INSTRUCTION if contract == BOUNDED_VERSION else LEGACY_RETRY_INSTRUCTION
    return instruction + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def render_verified(report: dict) -> list[str]:
    """Host display from verified model claims; never a policy target or oracle note."""
    rendered = []
    for checked in report.get("lesson_checks", []):
        if checked.get("verified") is not True:
            continue
        claim = checked["claim"]
        earlier, later = claim["earlier"], claim["later"]
        target = "end of recorded history" if later["attempt"] is None else f"attempt {later['attempt']}"
        rendered.append(f"Attempt {earlier['attempt']} {earlier['kind']} {earlier['rule']} at "
                        f"{earlier['locus']}: {claim['relation']} at {target}.")
    return rendered
