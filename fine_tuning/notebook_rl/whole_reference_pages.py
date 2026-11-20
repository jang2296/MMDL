"""Bounded public notebook pages, assembled only for independent verification.

Every action remains its original sampled page. The assembled reference JSON is
bookkeeping, and its literal spans map back to those unchanged page actions.
"""
from __future__ import annotations

import json
from decimal import InvalidOperation
from typing import Any

from . import whole_reference_notebook as reference
from .observations import slot_id
from .rollout import _hash
from .schema import Sample

VERSION = "whole-history-reference-pages-v2"
CAPTURED = "WHOLE_REFERENCE_PAGES_CAPTURED"
INVALID = "INVALID_WHOLE_REFERENCE_PAGES"
TRUNCATED = "TRUNCATED_WHOLE_REFERENCE_PAGES"
_HEADER = ("corrected_facts", "attempts", "success_attempt", "corrected_answer")
_OLD_ENDING = "The read-only input has ended. Return your own complete five-field assessment JSON only.\n"
_ENDING = "The read-only input has ended. Return only this page's owned JSON fields; never another page's fields.\n"
_HEADER_OWNERSHIP = ("Header page 0 owns EXACTLY corrected_facts, attempts, success_attempt, corrected_answer. "
                     "It has no assessments field.")
_ATTEMPT_OWNERSHIP = ("Attempt page {page_index} owns EXACTLY assessments. It has no header fields. "
                      "Include EXACTLY these refs in this order: {refs}.")
_TAIL = ("\nAll original attempts and all "
         "destination refs remain available for your own judgments. Earlier page outputs are unavailable. "
         "Do not invent, omit, select or repair another page's claims.\n")
_REFERENCE_PROMPT = reference.prompt_contract()
_INTRO = _REFERENCE_PROMPT[:_REFERENCE_PROMPT.index("Return one bare JSON object")]
_HEADER_GRAMMAR = _REFERENCE_PROMPT[_REFERENCE_PROMPT.index("corrected_facts:"):
                                    _REFERENCE_PROMPT.index("assessments:")]
_HEADER_EXAMPLE = ('Synthetic JSON type example only (unrelated dummy values; do not copy them as item judgments):\n'
    '{"corrected_facts":{"synthetic_slot":"731"},"attempts":[[17,"INCORRECT","731"]],'
    '"success_attempt":17,"corrected_answer":"731"}\n'
    'Use JSON null for an absent extracted answer, never the string "null".\n')
_ASSESSMENT_GRAMMAR = _REFERENCE_PROMPT[_REFERENCE_PROMPT.index("assessments:"):
                                        _REFERENCE_PROMPT.index("Synthetic JSON type examples")].replace(
    "per assessment_refs entry", "per owned assessment ref", 1)
_ASSESSMENT_EXAMPLES = ("Synthetic JSON type examples only (unrelated dummy values; do not copy them as item judgments):\n"
    '{"assessments":[["<source ref>","N",[],null]]}\n'
    '{"assessments":[["<source ref>","I",[],["CORRECTED","<later ref>"]]]}\n')
_RELATION_GRAMMAR = _REFERENCE_PROMPT[_REFERENCE_PROMPT.index("Use JSON null, never"):
                                      _REFERENCE_PROMPT.index("The host copies chosen refs")].replace(
    "all source issues still need coverage in a strict complete notebook.",
    "all owned source issues still need coverage on this page.", 1)
_READONLY_GRAMMAR = _REFERENCE_PROMPT[_REFERENCE_PROMPT.index("The host copies chosen refs"):
                                      _REFERENCE_PROMPT.index("Public observation schema (format only):")]
_DATA_START = "Public observation schema (format only):\n"
_SUCCESS_METADATA = "Designated successful retry ID (read-only metadata): {success_attempt}"
DRAFT_NOTEBOOK_REQUEST = "Produce the mandatory complete notebook under its unchanged checker; "
DRAFT_PAGE_REQUEST = "Produce only the current page under its declared output contract; "


def prompt_contract() -> dict:
    return {"version": VERSION, "reference": reference.prompt_contract(), "scope": reference.SCOPE_GRAMMAR,
            "intro": _INTRO, "header_grammar": _HEADER_GRAMMAR, "header_example": _HEADER_EXAMPLE,
            "assessment_grammar": _ASSESSMENT_GRAMMAR, "assessment_examples": _ASSESSMENT_EXAMPLES,
            "relation_grammar": _RELATION_GRAMMAR, "readonly_grammar": _READONLY_GRAMMAR,
            "data_start": _DATA_START, "success_metadata": _SUCCESS_METADATA, "old_ending": _OLD_ENDING,
            "ending": _ENDING, "header": _HEADER_OWNERSHIP, "attempt": _ATTEMPT_OWNERSHIP, "tail": _TAIL,
            "draft_wrapper_request": [DRAFT_NOTEBOOK_REQUEST, DRAFT_PAGE_REQUEST]}


def contract() -> dict:
    return {"version": VERSION, "reference_contract": reference.contract(), "group_size": 4,
        "total_tokens": 8192, "header_tokens": 2048, "assessment_tokens": 6144,
        "allocation": "proportional-reference-count; largest-remainders; original-attempt-order-ties",
        "header_fields": list(_HEADER), "attempt_fields": ["assessments"],
        "history": "complete-original-history-and-images-every-call",
        "page_inputs": "no-previous-page-outputs; no-private-judgments",
        "assembly": "all-pages-of-one-original-ordinal; no-selection-or-repair",
        "logical_diag_group_count": 1, "action_normalizer": 4 * 8192,
        "mis": "token-count-weighted-bundle-mean; shared-float32-keep; original-n4-centering",
        "credit": "original-sampled-page-literals-only; assembled-JSON-is-never-an-action"}


def plan(originals: list[Sample], total_tokens: int = 8192) -> dict:
    """Allocate the fixed output budget using only the public reference inventory."""
    if type(total_tokens) is not int or total_tokens != 8192:
        raise ValueError("whole reference pages: fixed 8192-token budget required")
    inventory = reference.public_inventory(originals)
    owned = [[ref for ref in inventory["assessment_refs"]
              if inventory["references"][ref]["attempt"] == attempt]
             for attempt in range(1, len(originals) + 1)]
    counts = [len(refs) for refs in owned]
    count = sum(counts)
    caps = [6144 * value // count for value in counts]
    order = sorted(range(len(counts)), key=lambda index: (-(6144 * counts[index] % count), index))
    for index in order[:6144 - sum(caps)]:
        caps[index] += 1
    if any(cap <= 0 for cap in caps):
        raise ValueError("whole reference pages: proportional allocation has a nonpositive cap")
    pages: list[dict] = [{"page_index": 0, "attempt": None, "max_new_tokens": 2048, "assessment_refs": []}]
    pages.extend({"page_index": index, "attempt": index, "max_new_tokens": cap,
                  "assessment_refs": refs}
                 for index, (cap, refs) in enumerate(zip(caps, owned, strict=True), 1))
    source = [{key: getattr(sample, key) for key in ("text", "sample_id", "parent_sample_id", "finish_reason")}
              for sample in originals]
    result = {"version": VERSION, "total_tokens": total_tokens, "group_size": 4,
              "source_sha256": _hash(source), "inventory_sha256": inventory["sha256"], "pages": pages}
    return {**result, "sha256": _hash(result)}


def instruction(originals: list[Sample], schema: str, success_attempt: int, page_index: int) -> str:
    """Keep every source and destination visible; request only this page's claims."""
    pages = plan(originals)["pages"]
    if type(page_index) is not int or not 0 <= page_index < len(pages) or type(schema) is not str:
        raise ValueError("whole reference pages: invalid page or public schema")
    quoted_schema = json.dumps(schema, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    prompt = reference.instruction(originals, quoted_schema, success_attempt)
    _, marker, readonly = prompt.partition(_DATA_START)
    if not marker or not readonly.endswith(_OLD_ENDING):
        raise ValueError("whole reference pages: original public serialization differs")
    readonly = _DATA_START + readonly.removesuffix(_OLD_ENDING)
    ownership = (_HEADER_OWNERSHIP if page_index == 0 else _ATTEMPT_OWNERSHIP.format(
        page_index=page_index, refs=json.dumps(pages[page_index]["assessment_refs"], separators=(",", ":"))))
    if page_index == 0:
        grammar = _HEADER_GRAMMAR + _HEADER_EXAMPLE
    else:
        grammar = _ASSESSMENT_GRAMMAR + _ASSESSMENT_EXAMPLES + _RELATION_GRAMMAR
        metadata = f"\nDesignated successful retry ID (copy as success_attempt): {success_attempt}\n<public_inventory>\n"
        if readonly.count(metadata) != 1:
            raise ValueError("whole reference pages: designated success metadata differs")
        readonly = readonly.replace(metadata, "\n" + _SUCCESS_METADATA.format(
            success_attempt=success_attempt) + "\n<public_inventory>\n", 1)
    return (reference.SCOPE_GRAMMAR + "\n" + _INTRO + ownership + "\nReturn one bare JSON object.\n"
            + grammar + _READONLY_GRAMMAR + readonly + _ENDING + ownership + _TAIL)


def _bindings(page_samples: list[Sample], originals: list[Sample], page_count: int) -> list[dict]:
    if type(page_samples) is not list or len(page_samples) != page_count:
        raise ValueError("whole reference pages: complete ordered page collection required")
    if (not originals or type(originals[0].problem_id) is not str or not originals[0].problem_id
            or any(sample.problem_id != originals[0].problem_id for sample in originals)):
        raise ValueError("whole reference pages: original problem identity differs")
    ids: set[str] = {sample.sample_id for sample in originals}
    groups: set[str] = set()
    policy: int | None = None
    bindings = []
    for index, sample in enumerate(page_samples):
        if (not isinstance(sample, Sample) or type(sample.sample_id) is not str or not sample.sample_id
                or sample.sample_id in ids or type(sample.group_id) is not str or not sample.group_id
                or sample.group_id in groups or sample.kind != "DIAG"
                or type(sample.turn) is not int or sample.turn != 1
                or sample.problem_id != originals[0].problem_id or type(sample.policy_version) is not int
                or sample.policy_version < 0 or policy is not None and sample.policy_version != policy
                or type(sample.text) is not str or type(sample.finish_reason) is not str):
            raise ValueError("whole reference pages: inconsistent original DIAG page identity")
        ids.add(sample.sample_id)
        groups.add(sample.group_id)
        policy = sample.policy_version
        bindings.append({"page_index": index, "sample_id": sample.sample_id, "group_id": sample.group_id,
            "policy_version": sample.policy_version, "turn": sample.turn, "finish_reason": sample.finish_reason,
            "text_sha256": reference._sha(sample.text)})
    return bindings


def assemble(page_samples: list[Sample], originals: list[Sample]) -> dict:
    """Join one ordinal's original pages, preserving every copied JSON literal."""
    page_plan = plan(originals)
    bindings = _bindings(page_samples, originals, len(page_plan["pages"]))
    result: dict[str, Any] = {"version": VERSION, "status": INVALID, "errors": [],
        "page_plan": page_plan, "pages": bindings, "reference_text": None, "expansion": None,
        "provenance": {"facts": {}, "decisions": {}}}
    nonstop = [row["page_index"] for row in bindings if row["finish_reason"] != "stop"]
    if nonstop:
        result.update(status=TRUNCATED if any(page_samples[index].finish_reason == "length" for index in nonstop)
                      else INVALID, errors=[{"page_index": index, "error": "NONSTOP_RESPONSE"} for index in nonstop])
        return result
    decoded: list[dict] = []
    spans: list[dict[tuple, list[int]]] = []
    for index, sample in enumerate(page_samples):
        try:
            note, positions = reference._decode(sample.text)
            expected = set(_HEADER) if index == 0 else {"assessments"}
            if type(note) is not dict or set(note) != expected:
                raise ValueError("INVALID_PAGE_KEYS")
            if index:
                rows, refs = note["assessments"], page_plan["pages"][index]["assessment_refs"]
                if (type(rows) is not list or len(rows) != len(refs)
                        or any(type(row) is not list or len(row) != 4 or type(row[0]) is not str or row[0] != ref
                               for row, ref in zip(rows, refs, strict=True))):
                    raise ValueError("INVALID_PAGE_REFS")
            decoded.append(note)
            spans.append(positions)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, InvalidOperation) as exc:
            result["errors"].append({"page_index": index, "error": str(exc)})
    if result["errors"]:
        return result

    def literal(index: int, path: tuple) -> str:
        start, end = spans[index][path]
        return page_samples[index].text[start:end]

    # Copy lexical values, including Decimal answers, instead of serializing decoded numbers.
    fields = [json.dumps(key) + ":" + literal(0, (key,)) for key in _HEADER]
    fields.append('"assessments":[' + ",".join(literal(index, ("assessments", ordinal))
        for index, note in enumerate(decoded[1:], 1) for ordinal in range(len(note["assessments"]))) + "]")
    text = "{" + ",".join(fields) + "}"
    expanded = reference.expand(text, "stop", originals)
    result.update(reference_text=text, expansion=expanded)
    if expanded["status"] != reference.CAPTURED:
        result["errors"] = [{"page_index": None, "error": error} for error in expanded["errors"]]
        return result

    def provenance(index: int, source_span: list[int], assembled_span: list[int]) -> dict:
        source = page_samples[index].text[slice(*source_span)]
        if source != text[slice(*assembled_span)]:
            raise ValueError("whole reference pages: mapped literal bytes differ")
        return {"sample_id": page_samples[index].sample_id, "page_index": index,
                "source_span": source_span, "assembled_span": assembled_span}

    facts = {slot_id(key): provenance(0, spans[0][("corrected_facts", key)], expanded["fact_spans"][slot_id(key)])
             for key in decoded[0]["corrected_facts"]}
    decisions = {}
    for index, note in enumerate(decoded[1:], 1):
        for ordinal, row in enumerate(note["assessments"]):
            if row[3] is not None:
                decisions[row[0]] = provenance(index, spans[index][("assessments", ordinal, 3)],
                                                expanded["decision_spans"][row[0]])
    result.update(status=CAPTURED, provenance={"facts": facts, "decisions": decisions})
    return result
