"""Bounded v16 self-note validation; no generation, retries, or effect certification."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any

from .audit import digest, text_sha256
from .contract import ROUTE_FULL, ROUTE_LM, a2_target_problem, answer_denial, hint_leak_single_v2
from .single_grading import normalize_final
from .v4 import target_problem

VERSION = "error-note-self-note-v16"
NOTE_INSTRUCTION_V16 = (
    "Inspect your actual incorrect first solution using the image, question, grading result, "
    "given answer and the bounded catalog. Return JSON only, with exactly observed_facts, error, "
    "lesson. observed_facts contains at most 3 objects {fact_ref,value}. error contains exactly "
    "earlier_span (Unicode half-open offsets), earlier_quote, kind (OBSERVATION or ARITHMETIC), "
    "operand_fact_refs, operation, old_value, corrected_value, rounding, unit, rule_id. "
    "Quote one complete asserted fact or arithmetic line, never a negated, hypothetical, quoted "
    "or retracted claim. lesson contains exactly rule_id and the catalog's abstract formula. "
    "Do not put problem values, options, names, or the answer in the lesson. Do not write a revised solution."
)
RESOLVE_INSTRUCTION_V16 = (
    "Solve the question once from the beginning, using only the image, question and abstract lesson. "
    "End with one explicit conclusion and one consistent Final answer. Do not mention review material."
)
FORMULAS = {
    "LOOKUP": "value=observed_value", "ADD": "result=left+right",
    "SUB": "result=left-right", "MUL": "result=left*right",
    "DIV": "result=numerator/denominator",
    "TRIANGLE_REMAINDER": "x=180deg-a-b", "STRAIGHT_REMAINDER": "x=180deg-a",
}
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_ATOM = re.compile(r"[\w .%°:/+-]{1,64}\Z", re.UNICODE)
_FACT_LINE = re.compile(r"(?P<ref>[A-Za-z][\w-]*)\s*=\s*(?P<value>[^=\n]{1,64})\Z")
_EQUATION_BODY = (rf"(?P<a>{_NUMBER})\s*(?P<op>[+*/×÷-])\s*(?P<b>{_NUMBER})"
                  rf"(?:\s*-\s*(?P<c>{_NUMBER}))?\s*=\s*(?P<result>{_NUMBER})")
_EQUATION = re.compile(_EQUATION_BODY + r"\Z")
_EQUATION_IN = re.compile(_EQUATION_BODY)
_RETRACT = re.compile(r"\b(?:incorrect|wrong|mistake|retract|correction|instead|not true)\b", re.I)
_DOUBT = re.compile(r"\b(?:if|suppose|assume|assuming|hypothetically|would|could|should|might|may|maybe|perhaps"
                    r"|unless|whether|example|consider|say|guess|try|trying|not|wait|check|verify"
                    r"|recheck|recalculate)\b", re.I)
_RECONSIDER = re.compile(r"\b(?:incorrect|wrong|mistake|retract|correction|instead|wait|actually|recalculate"
                         r"|recheck|re-calculate)\b|\blet(?: me|'s|’s) re", re.I)
# A quote touching an operator, digit, LaTeX command, identifier or quote mark is a fragment, not a claim.
# Thousands separators ("1,200"), relations, roots, en-dash minus, implicit products ("-950\cos(x)", "5(2)"),
# '?' and '!' are fragments or non-assertions too.
_FRAGMENT_BEFORE = re.compile(r"(?:[\d.+\-*/x×÷=^({}−–·√≠≈<>≤≥]|\d,|\\[A-Za-z]+)\Z")
_FRAGMENT_AFTER = re.compile(r"[\d.+\-*/×÷=^−–·%?!(\\]|,\d")
_QUOTE_MARKS = ('"', "'", "“", "”", "‘", "’", "`")
_FINAL = re.compile(r"^\s*(?:final answer|answer)\s*:\s*(.+?)\s*$", re.I)
_CONCLUSION = re.compile(
    r"^\s*(?:conclusion\s*:\s*|(?:therefore|thus|hence)[,:]?\s+(?:the answer is\s+)?)(.+?)\s*$", re.I
)
_ERROR_FIELDS = {
    "earlier_span", "earlier_quote", "kind", "operand_fact_refs", "operation", "old_value",
    "corrected_value", "rounding", "unit", "rule_id",
}
_WRONG = frozenset({"INCORRECT", "TRUNCATED", "INVALID_NO_FINAL", "INVALID_MULTIPLE"})
_IDENTITY_HASHES = ("experiment_sha256", "content_manifest_sha256", "protocol_sha256",
                    "annotation_sha256", "actor_sha256")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _decimal(value: str) -> Decimal:
    if not isinstance(value, str) or len(value) > 64 or not re.fullmatch(_NUMBER, value):
        raise ValueError("UNSUPPORTED_NUMBER")
    result = Decimal(value)
    if not result.is_finite() or result.adjusted() > 64:
        raise ValueError("UNSUPPORTED_NUMBER")
    return result


def _same(left: str, right: str) -> bool:
    try:
        return _decimal(left) == _decimal(right)
    except (InvalidOperation, ValueError):
        return left.strip() == right.strip()


def _catalog(facts: list[dict], rules: dict) -> dict[str, dict]:
    if not isinstance(facts, list) or not isinstance(rules, dict):
        raise ValueError("UNSUPPORTED_SOURCE_CONTRACT")
    indexed = {}
    for fact in facts:
        required = {"fact_id", "kind", "entity", "evidence_origin", "value", "unit"}
        if (not isinstance(fact, dict) or not required <= fact.keys()
                or any(not isinstance(fact[k], str) or not fact[k] for k in required)
                or not _ATOM.fullmatch(fact["value"]) or fact["fact_id"] in indexed
                or type(fact.get("answer_specific", False)) is not bool):
            raise ValueError("UNSUPPORTED_SOURCE_FACT")
        indexed[fact["fact_id"]] = fact
    for rule_id, rule in rules.items():
        if (not isinstance(rule_id, str) or not rule_id or not isinstance(rule, dict)
                or rule.get("operation") not in FORMULAS
                or rule.get("formula") != FORMULAS[rule["operation"]]
                or not isinstance(rule.get("operand_units"), list)
                or not all(isinstance(u, str) and u for u in rule["operand_units"])
                or not isinstance(rule.get("result_unit"), str) or not rule["result_unit"]
                or type(rule.get("answer_specific", False)) is not bool):
            raise ValueError("UNSUPPORTED_SOURCE_RULE")
        preconditions = rule.get("preconditions", {})
        if (not isinstance(preconditions, dict)
                or any(ref not in indexed or not isinstance(value, str)
                       or not _same(indexed[ref]["value"], value) for ref, value in preconditions.items())):
            raise ValueError("RULE_PRECONDITION_FAILED")
        if rule["operation"] in {"TRIANGLE_REMAINDER", "STRAIGHT_REMAINDER"} and not preconditions:
            raise ValueError("RULE_PRECONDITION_MISSING")
        operation, units, result_unit = rule["operation"], rule["operand_units"], rule["result_unit"]
        expected_count = 1 if operation in {"LOOKUP", "STRAIGHT_REMAINDER"} else 2
        if len(units) != expected_count:
            raise ValueError("RULE_UNIT_MISMATCH")
        if (operation in {"LOOKUP", "ADD", "SUB"} and any(unit != result_unit for unit in units)
                or operation == "DIV" and (units[0] != units[1] or result_unit != "dimensionless")
                or operation == "MUL" and units + [result_unit] != ["dimensionless"] * 3
                or operation in {"TRIANGLE_REMAINDER", "STRAIGHT_REMAINDER"}
                and any(unit != "degree" for unit in units + [result_unit])):
            raise ValueError("RULE_UNIT_MISMATCH")
    return indexed


def public_note_catalog(fact_catalog: list[dict], rules: dict) -> dict:
    """Project source-fixed identities and rules, never private values or backend paths."""
    _catalog(fact_catalog, rules)
    return {"facts": [{k: fact[k] for k in ("fact_id", "kind", "entity", "evidence_origin")}
                      for fact in fact_catalog],
            "rules": {key: {k: rule[k] for k in ("operation", "formula")}
                      | {"precondition_fact_refs": list(rule.get("preconditions", {}))}
                      for key, rule in rules.items()}}


def build_problem_contract(item: dict) -> dict:
    """Derive bounded facts from an already verified O source, never from model prose."""
    from . import content_v16 as content

    try:
        layer, annotation = item["layer"], item["annotation"]
        observation = content.observation_row(item, layer=layer)
        facts: list[dict[str, Any]] = []

        def fact(ref: str, entity: str, value: str, unit: str) -> None:
            if _ATOM.fullmatch(value):
                facts.append({"fact_id": ref, "kind": "OBSERVATION", "entity": entity,
                              "value": value, "unit": unit, "answer_specific": False,
                              "evidence_origin": f"{layer}:{digest(annotation)}:{ref}"})

        if layer == "graph":
            graph = content.parse_gvlqa_structure(annotation["structure_text"])
            fact("visible_node_count", "number of nodes drawn in the image", str(len(graph["visible_nodes"])), "count")
            fact("edge_count", "number of drawn edges", str(graph["m"]), "count")
            for index, edge in enumerate(graph["edges"]):
                if len(edge) == 3:
                    fact(f"edge_weight_{index}", f"weight of edge {edge[0]} to {edge[1]}", edge[2], "weight")
        elif layer == "geometry":
            forms = annotation["candidates"][annotation["matched_id"]]["diagram_logic_forms"]
            for index, form in enumerate(forms):
                length, angle = content._LEN.fullmatch(form), content._ANG.fullmatch(form)
                if length and re.fullmatch(_NUMBER, length[3].strip()):
                    fact(f"f{index}", f"segment {length[1]}{length[2]}", length[3].strip(), "length")
                elif angle and re.fullmatch(_NUMBER, angle[4].strip()):
                    fact(f"f{index}", f"angle {angle[1]}{angle[2]}{angle[3]}", angle[4].strip(), "degree")
        elif layer == "scene":
            fact("object_count", "number of objects in the scene", str(len(annotation["objects"])), "count")
            for index, obj in enumerate(annotation["objects"]):
                for attribute in ("shape", "size", "color", "material"):
                    value = obj.get(attribute)
                    if isinstance(value, str):
                        fact(f"object_{index}_{attribute}", f"object {index} {attribute}", value, "label")
        elif layer == "chart":
            chart = annotation["chart"]
            model = chart["models"][0]
            for index, (category, value) in enumerate(zip(model["x"], model["y"], strict=True)):
                fact(f"f{index}", f"displayed value for {category}", str(value).strip(), chart["unit"] or "dimensionless")
        else:
            raise ValueError("UNSUPPORTED_SOURCE_CONTRACT")
        if not facts:
            raise ValueError("NO_SUPPORTED_ATOMIC_FACTS")
        units = sorted({entry["unit"] for entry in facts})
        rules = {}
        for unit in units:
            suffix = text_sha256(unit)[:8]
            rules[f"LOOKUP_{suffix}"] = {"operation": "LOOKUP", "formula": FORMULAS["LOOKUP"],
                                         "operand_units": [unit], "result_unit": unit}
            if not any(entry["unit"] == unit and re.fullmatch(_NUMBER, entry["value"]) for entry in facts):
                continue
            for operation in ("ADD", "SUB", "DIV"):
                rules[f"{operation}_{suffix}"] = {
                    "operation": operation, "formula": FORMULAS[operation], "operand_units": [unit, unit],
                    "result_unit": "dimensionless" if operation == "DIV" else unit,
                }
        # Source annotations do not establish that every printed/intermediate value is the answer.
        expected = item.get("gold")
        if expected is None:
            raise ValueError("SOURCE_GOLD_MISSING")
        result = {"supported": True, "reason": None, "fact_catalog": facts, "rules": rules,
                  "expected_final": str(expected), "annotation_sha256": digest(annotation),
                  "problem_id": item["problem_id"], "observation_receipt_sha256": digest(observation["verification_receipt"])}
        _catalog(facts, rules)
        return result | {"sha256": digest(result)}
    except (KeyError, TypeError, ValueError) as exc:
        return {"supported": False, "reason": str(exc), "fact_catalog": [], "rules": {}}


def build_note_prompt(record: dict, *, fact_catalog: list[dict], rules: dict, gold: str) -> str:
    if record["t1"]["status"] == "CORRECT":
        raise ValueError("SKIP_NOTE_NO_ERROR")
    return NOTE_INSTRUCTION_V16 + "\n\n" + json.dumps({
        "question": record["p0_text"], "first_solution": record["t1"]["text"],
        "grading_result": record["t1"]["status"], "given_answer": gold,
        "note_catalog": public_note_catalog(fact_catalog, rules),
    }, ensure_ascii=False, sort_keys=True)


def build_resolve_prompt(p0_text: str, lesson: dict) -> str:
    if (not isinstance(lesson, dict) or set(lesson) != {"rule_id", "formula"}
            or lesson["formula"] not in FORMULAS.values()):
        raise ValueError("UNVERIFIED_LESSON")
    return (RESOLVE_INSTRUCTION_V16 + "\n\nAbstract lesson: "
            + json.dumps(lesson, ensure_ascii=False, sort_keys=True) + "\n\n" + p0_text)


def _round(value: Decimal, rounding: str) -> Decimal:
    if rounding == "none":
        return value
    if not isinstance(rounding, str) or not re.fullmatch(r"decimal:(?:[0-9]|1[0-2])", rounding):
        raise ValueError("UNSUPPORTED_ROUNDING")
    return value.quantize(Decimal(1).scaleb(-int(rounding.split(":")[1])), rounding=ROUND_HALF_UP)


def _calculate(operation: str, operands: list[str], rounding: str) -> Decimal:
    values = [_decimal(value) for value in operands]
    count = 1 if operation == "STRAIGHT_REMAINDER" else 2
    if len(values) != count:
        raise ValueError("OPERAND_COUNT")
    with localcontext() as ctx:
        ctx.prec = 128
        if operation == "ADD":
            value = values[0] + values[1]
        elif operation == "SUB":
            value = values[0] - values[1]
        elif operation == "MUL":
            value = values[0] * values[1]
        elif operation == "DIV":
            if values[1] == 0:
                raise ValueError("DIVISION_BY_ZERO")
            value = values[0] / values[1]
        elif operation == "TRIANGLE_REMAINDER":
            value = Decimal(180) - sum(values)
        elif operation == "STRAIGHT_REMAINDER":
            value = Decimal(180) - values[0]
        else:
            raise ValueError("UNSUPPORTED_OPERATION")
        return _round(value, rounding)


def _same_lhs(left: re.Match, right: re.Match) -> bool:
    def operator(match: re.Match) -> str:
        return match["op"].replace("×", "*").replace("÷", "/")

    return operator(left) == operator(right) and all(
        (x is None) == (y is None) and (x is None or _same(x, y))
        for x, y in ((left["a"], right["a"]), (left["b"], right["b"]), (left["c"], right["c"])))


def earlier_assertion_status(text: str, span: list[int], quote: str) -> str:
    """A whole equation/fact, asserted in place (prose allowed) and never revisited later."""
    if (not isinstance(quote, str) or not quote or "\n" in quote or not isinstance(span, list)
            or len(span) != 2 or any(type(i) is not int for i in span)
            or not 0 <= span[0] < span[1] <= len(text) or text[span[0]:span[1]] != quote):
        return "UNKNOWN"
    equation = _EQUATION.fullmatch(quote.strip())
    if not (equation or _FACT_LINE.fullmatch(quote.strip())):
        return "UNKNOWN"
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    end = len(text) if end < 0 else end
    line, before, after = text[start:end], text[start:span[0]], text[span[1]:end]
    left, right = before.strip(), after.strip()
    if (re.search(r"\w\Z", before) or re.match(r"\w", after)
            or _FRAGMENT_BEFORE.search(left) or _FRAGMENT_AFTER.match(right)
            or left.endswith(_QUOTE_MARKS) or right.startswith(_QUOTE_MARKS)):
        return "UNKNOWN"
    if _RETRACT.search(line):
        return "RETRACTED"
    # A lead-in ending with ':' frames a quote that starts the next line ("Suppose we wrongly compute:").
    lead = text[:start].rstrip()
    lead = lead[lead.rfind("\n") + 1:] if not left and lead.endswith(":") else ""
    if _DOUBT.search(line) or _DOUBT.search(lead) or _RETRACT.search(lead):
        return "UNKNOWN"
    status = "ASSERTED"
    for later in (after, *text[end:].splitlines()):
        if equation and any(_same_lhs(match, equation) and not _same(match["result"], equation["result"])
                            for match in _EQUATION_IN.finditer(later)):
            return "RETRACTED"
        if _RECONSIDER.search(later):
            status = "UNKNOWN"
    return status


def _equation_operands(match: re.Match, operation: str) -> list[str]:
    op = match["op"].replace("×", "*").replace("÷", "/")
    if operation == "TRIANGLE_REMAINDER":
        if op != "-" or not _same(match["a"], "180") or match["c"] is None:
            raise ValueError("QUOTE_OPERATION_MISMATCH")
        return [match["b"], match["c"]]
    if operation == "STRAIGHT_REMAINDER":
        if op != "-" or not _same(match["a"], "180") or match["c"] is not None:
            raise ValueError("QUOTE_OPERATION_MISMATCH")
        return [match["b"]]
    if match["c"] is not None or {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}.get(operation) != op:
        raise ValueError("QUOTE_OPERATION_MISMATCH")
    return [match["a"], match["b"]]


def _answer_key(value: str, choices: list[str] | tuple[str, ...]) -> str | None:
    value = value.strip().rstrip(".").strip()
    letter = re.fullmatch(r"\(?([A-Z])\)?(?:\.\s*(.*))?", value, re.I)
    if choices and letter and ord(letter[1].upper()) - 65 < len(choices):
        index = ord(letter[1].upper()) - 65
        if letter[2] and normalize_final(letter[2]) != normalize_final(choices[index]):
            return None
        return chr(65 + index)
    if choices:
        matched = [chr(65 + i) for i, option in enumerate(choices)
                   if normalize_final(option) == normalize_final(value)]
        return matched[0] if len(matched) == 1 else None
    normalized = normalize_final(value)
    return normalized if re.fullmatch(_NUMBER + r"|[A-Za-z][\w-]{0,63}", normalized) else None


def _receipt(record: dict, accepted: bool, reason: str | None, **extra) -> dict:
    t1_generation = record["t1"].get("generation", {})
    note_generation = record.get("note", {}).get("generation")
    value = {"version": VERSION, "problem_id": record["problem_id"],
             "t1_status": record["t1"]["status"], "accepted": accepted, "reason": reason,
             "identity": {key: record.get(key) for key in _IDENTITY_HASHES}
             | {"parent_id": record.get("parent_id"), "source": record.get("source"),
                "image_sha256": record.get("image_sha256"),
                "policy_sha256": t1_generation.get("policy_sha256")},
             "t1_generation_sha256": digest(t1_generation),
             "note_generation_sha256": digest(note_generation) if note_generation else None,
             "t1_sha256": text_sha256(record["t1"]["text"]),
             "note_sha256": text_sha256(record.get("note", {}).get("text", "")), **extra}
    return value | {"sha256": digest(value)}


def _identity_bound(identity: Any) -> bool:
    hashes = (*_IDENTITY_HASHES, "policy_sha256")
    return (isinstance(identity, dict)
            and all(isinstance(identity.get(key), str) and _SHA.fullmatch(identity[key]) for key in hashes)
            and all(isinstance(identity.get(key), str) and identity[key] for key in ("parent_id", "source"))
            and isinstance(identity.get("image_sha256"), list) and bool(identity["image_sha256"])
            and all(isinstance(value, str) and _SHA.fullmatch(value) for value in identity["image_sha256"]))


def _stored_output(output: dict) -> bool:
    # Stream records carry completion_tokens on the outer output only; any present count must match.
    generation = output.get("generation", {})
    actions = generation.get("completion_token_ids")
    return (isinstance(actions, list) and bool(actions)
            and all(type(value) is int and value >= 0 for value in actions)
            and all(holder.get("completion_tokens", len(actions)) == len(actions) for holder in (output, generation))
            and generation.get("text") == output.get("text")
            and generation.get("finish_reason") == output.get("finish_reason"))


def _note_spans(text: str) -> list[list[int]]:
    excluded = []
    for key in ("earlier_quote", "old_value"):
        match = re.search(r'"' + key + r'"\s*:\s*("(?:[^"\\]|\\.)*")', text)
        if match is None:
            raise ValueError("NOTE_SCHEMA")
        excluded.append(list(match.span(1)))
    spans, cursor = [], 0
    for start, end in sorted(excluded):
        if cursor < start:
            spans.append([cursor, start])
        cursor = end
    if cursor < len(text):
        spans.append([cursor, len(text)])
    return spans


def verify_note(record: dict, *, fact_catalog: list[dict], rules: dict,
                choices: list[str] | tuple[str, ...], gold: str) -> dict:
    """Verify N before C; C success never substitutes for verification of a note."""
    assertion = "UNKNOWN"
    source_sha = digest({"facts": fact_catalog, "rules": rules})
    try:
        if record["t1"]["status"] == "CORRECT":
            raise ValueError("SKIP_NOTE_NO_ERROR")
        if record["t1"]["status"] not in _WRONG:
            raise ValueError("T1_NOT_GRADED_WRONG")
        identity = _receipt(record, False, None)["identity"]
        if (not _identity_bound(identity)
                or record["t1"].get("generation", {}).get("protocol_sha256") != record["protocol_sha256"]):
            raise ValueError("NOTE_IDENTITY_NOT_BOUND")
        facts = _catalog(fact_catalog, rules)
        output = record.get("note")
        if not isinstance(output, dict):
            raise ValueError("NOTE_MISSING")
        if not _stored_output(record["t1"]) or not _stored_output(output):
            raise ValueError("CHANGED_ORIGINAL_GENERATION")
        if (output.get("generation", {}).get("policy_sha256") != identity["policy_sha256"]
                or output.get("generation", {}).get("protocol_sha256") != identity["protocol_sha256"]):
            raise ValueError("NOTE_GENERATION_IDENTITY_MISMATCH")
        actions = output["generation"]["completion_token_ids"]
        if problem := target_problem(output["text"], actions, output["finish_reason"], "CORRECT", min_tokens=1):
            raise ValueError(problem)
        note = json.loads(output["text"], object_pairs_hook=_unique)
        if not isinstance(note, dict) or set(note) != {"observed_facts", "error", "lesson"}:
            raise ValueError("NOTE_SCHEMA")
        observed, error, lesson = note["observed_facts"], note["error"], note["lesson"]
        if (not isinstance(observed, list) or not 1 <= len(observed) <= 3
                or not isinstance(error, dict) or set(error) != _ERROR_FIELDS
                or not isinstance(lesson, dict) or set(lesson) != {"rule_id", "formula"}):
            raise ValueError("NOTE_SCHEMA")
        seen = {}
        answer_values = []
        for fact in observed:
            if (not isinstance(fact, dict) or set(fact) != {"fact_ref", "value"}
                    or not isinstance(fact["fact_ref"], str) or fact["fact_ref"] not in facts
                    or fact["fact_ref"] in seen or not isinstance(fact["value"], str)
                    or not _ATOM.fullmatch(fact["value"])):
                raise ValueError("OBSERVED_FACT_SCHEMA")
            ref = fact["fact_ref"]
            if not _same(fact["value"], facts[ref]["value"]):
                raise ValueError("OBSERVED_FACT_MISMATCH")
            seen[ref] = fact["value"]
            if facts[ref].get("answer_specific", False):
                answer_values.append(fact["value"])
        rule_id = error["rule_id"]
        if not isinstance(rule_id, str) or rule_id not in rules:
            raise ValueError("UNSUPPORTED_RULE")
        rule = rules[rule_id]
        if (lesson != {"rule_id": rule_id, "formula": rule["formula"]}
                or error["operation"] != rule["operation"]):
            raise ValueError("LESSON_RULE_MISMATCH")
        if error["rounding"] != rule.get("rounding", "none"):
            raise ValueError("ROUNDING_RULE_MISMATCH")
        refs = error["operand_fact_refs"]
        if (not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in seen for ref in refs)
                or [facts[ref]["unit"] for ref in refs] != rule["operand_units"]
                or error["unit"] != rule["result_unit"]):
            raise ValueError("OPERAND_BINDING_OR_UNIT")
        if any(not isinstance(error[key], str) or not _ATOM.fullmatch(error[key])
               for key in ("old_value", "corrected_value")):
            raise ValueError("ATOMIC_VALUE_REQUIRED")
        quote = error["earlier_quote"]
        assertion = earlier_assertion_status(record["t1"]["text"], error["earlier_span"], quote)
        if assertion != "ASSERTED":
            raise ValueError("EARLIER_ASSERTION_" + assertion)
        if error["kind"] == "OBSERVATION" and error["operation"] == "LOOKUP":
            match = _FACT_LINE.fullmatch(quote.strip())
            if (len(refs) != 1 or match is None or match["ref"] != refs[0]
                    or not _same(match["value"].strip(), error["old_value"])):
                raise ValueError("QUOTE_FACT_MISMATCH")
            corrected = facts[refs[0]]["value"]
            if error["rounding"] != "none":
                raise ValueError("UNSUPPORTED_ROUNDING")
        elif error["kind"] == "ARITHMETIC" and error["operation"] != "LOOKUP":
            match = _EQUATION.fullmatch(quote.strip())
            if match is None or not _same(match["result"], error["old_value"]):
                raise ValueError("QUOTE_ARITHMETIC_MISMATCH")
            operands = _equation_operands(match, error["operation"])
            if len(refs) != len(operands) or any(not _same(value, seen[ref]) for value, ref in zip(operands, refs)):
                raise ValueError("QUOTE_SOURCE_MISMATCH")
            corrected = str(_calculate(error["operation"], operands, error["rounding"]))
        else:
            raise ValueError("UNSUPPORTED_ERROR_KIND")
        if _same(error["old_value"], corrected):
            raise ValueError("ORIGINAL_CLAIM_ALREADY_CORRECT")
        if not _same(error["corrected_value"], corrected):
            raise ValueError("CORRECTION_MISMATCH")
        # Correcting the step that produced T1's explicit final answer makes the correction an answer claim;
        # an intermediate step is never compared with gold.
        finals = [match[1] for line in record["t1"]["text"].splitlines() if (match := _FINAL.fullmatch(line))]
        # The graded extraction also reads final lines the plain pattern cannot (markdown, boxed).
        stated = [value for value in (finals[-1] if finals else None, record["t1"].get("extracted"))
                  if isinstance(value, str) and value.strip()]
        old_key = _answer_key(error["old_value"], choices)
        is_final = any(_same(error["old_value"], value)
                       or old_key is not None and old_key == _answer_key(value, choices) for value in stated)
        if is_final or rule.get("answer_specific", False):
            answer_values.append(error["corrected_value"])
        if any(_answer_key(value, choices) != _answer_key(gold, choices) for value in answer_values):
            raise ValueError("NOTE_GOLD_CONTRADICTION")
        lesson_text = json.dumps(lesson, ensure_ascii=False, sort_keys=True)
        if problem := hint_leak_single_v2(lesson_text, gold, choices):
            raise ValueError("LESSON_LEAK:" + problem)
        return _receipt(record, True, None, lesson=lesson, answer_specific_values=answer_values,
                        quote_is_final_computation=is_final,
                        earlier_assertion_status=assertion, verified_spans=_note_spans(output["text"]),
                        source_contract_sha256=source_sha)
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "NOTE_SCHEMA"
        return _receipt(record, False, reason, earlier_assertion_status=assertion,
                        source_contract_sha256=source_sha)


def measure_note_yield(records: list[dict]) -> dict:
    """Recompute the verifier from raw bound T1/N and source facts; never trust a PASS label."""
    receipts, evidence = [], []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("note_validation"), dict):
            raise ValueError("RAW_NOTE_YIELD_EVIDENCE_REQUIRED")
        validation = record["note_validation"]
        try:
            receipt = verify_note(record, **validation)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_NOTE_YIELD_EVIDENCE") from exc
        if receipt != record.get("note_receipt"):
            raise ValueError("INVALID_NOTE_YIELD_RECEIPT")
        receipts.append(receipt)
        keys = (*_IDENTITY_HASHES, "problem_id", "parent_id", "source", "image_sha256", "t1", "note")
        evidence.append({key: record[key] for key in keys if key in record}
                        | {"note_validation": validation, "note_receipt": receipt})
    indexed, contexts = {}, set()
    for receipt in receipts:
        payload = {key: value for key, value in receipt.items() if key != "sha256"}
        pid = receipt.get("problem_id")
        if (receipt.get("version") != VERSION or receipt.get("sha256") != digest(payload)
                or not isinstance(pid, str) or pid in indexed or receipt.get("t1_status") not in _WRONG
                or not _identity_bound(receipt.get("identity")) or type(receipt.get("accepted")) is not bool
                or not _SHA.fullmatch(receipt.get("source_contract_sha256", ""))
                or not _SHA.fullmatch(receipt.get("t1_generation_sha256", ""))
                or (receipt["accepted"] and (receipt.get("reason") is not None
                    or receipt.get("earlier_assertion_status") != "ASSERTED"
                    or not receipt.get("verified_spans") or not receipt.get("lesson")
                    or not _SHA.fullmatch(receipt.get("note_generation_sha256", ""))))):
            raise ValueError("INVALID_NOTE_YIELD_RECEIPT")
        contexts.add(tuple(receipt["identity"][key] for key in _IDENTITY_HASHES[:3]))
        indexed[pid] = receipt
    if len(contexts) > 1:
        raise ValueError("MIXED_NOTE_YIELD_EXPERIMENTS")
    accepted = sum(item["accepted"] is True for item in indexed.values())
    count = len(indexed)
    status = "HOLD" if count < 32 else "PASS" if accepted * 10 >= count else "FAIL"
    value = {"version": VERSION, "status": status,
             "wrong_t1_count": count, "verified_note_count": accepted,
             "yield": accepted / count if count else None, "receipts": receipts, "evidence_records": evidence}
    return value | {"sha256": digest(value)}


def create_note_yield_gate(records: list[dict]) -> dict:
    ids = [record["problem_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("INVALID_NOTE_YIELD_RECEIPT")
    # The initial 32 wrong first attempts are fixed; later easy notes cannot replace failures.
    return measure_note_yield(records[:32])


def _yield_pass(gate: dict | None) -> bool:
    if not isinstance(gate, dict):
        return False
    try:
        rebuilt = measure_note_yield(gate.get("evidence_records", []))
    except (TypeError, ValueError):
        return False
    return gate == rebuilt and gate["status"] == "PASS"


def _bound_equation(equation: re.Match, facts: dict[str, dict], rules: dict) -> bool:
    """True when operands bind to unique source facts and the rule recomputes the result; wrong raises."""
    for rule in rules.values():
        if rule["operation"] == "LOOKUP":
            continue
        try:
            operands = _equation_operands(equation, rule["operation"])
        except ValueError:
            continue
        refs = [[ref for ref, value in facts.items() if _same(value["value"], operand)] for operand in operands]
        if any(len(matches) != 1 for matches in refs):
            continue
        if [facts[matches[0]]["unit"] for matches in refs] != rule["operand_units"]:
            continue
        if _decimal(equation["result"]) != _calculate(rule["operation"], operands, rule.get("rounding", "none")):
            raise ValueError("CONTENT_CONTRADICTION")
        return True
    return False


def _arithmetic_wrong(equation: re.Match) -> bool:
    """Plain recomputation of an unbound equation; only used to name the rejection reason."""
    operation = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[equation["op"].replace("×", "*").replace("÷", "/")]
    with localcontext() as ctx:
        ctx.prec = 128
        value = _calculate(operation, [equation["a"], equation["b"]], "none")
        if equation["c"] is not None:
            value -= _decimal(equation["c"])
        return value != _decimal(equation["result"])


def verify_retry(record: dict, *, note_receipt: dict, fact_catalog: list[dict], rules: dict,
                 choices: list[str] | tuple[str, ...], gold: str) -> dict:
    """C may retain UNKNOWN prose, but errors reject the whole row and answer-only is excluded."""
    try:
        expected_note = verify_note(record, fact_catalog=fact_catalog, rules=rules, choices=choices, gold=gold)
        if not note_receipt.get("accepted") or note_receipt != expected_note:
            raise ValueError("NOTE_NOT_VERIFIED")
        facts = _catalog(fact_catalog, rules)
        output = record.get("resolve")
        if not isinstance(output, dict):
            raise ValueError("RESOLVE_MISSING")
        if not _stored_output(output):
            raise ValueError("CHANGED_ORIGINAL_GENERATION")
        if any(record.get(stage) is not None for stage in ("resolve2", "resolve3")):
            raise ValueError("MULTIPLE_RETRIES")
        actions = output["generation"]["completion_token_ids"]
        if problem := target_problem(output["text"], actions, output["finish_reason"], output["status"], min_tokens=1):
            raise ValueError(problem)
        lesson_text = json.dumps(note_receipt["lesson"], ensure_ascii=False, sort_keys=True)
        if problem := a2_target_problem(output["text"], lesson_text):
            raise ValueError(problem)
        lines, cursor = [], 0
        for raw in output["text"].splitlines(keepends=True):
            line = raw.strip()
            left = cursor + len(raw) - len(raw.lstrip())
            lines.append((line, [left, left + len(line)]))
            cursor += len(raw)
        finals = [(match[1], span) for line, span in lines if (match := _FINAL.fullmatch(line))]
        if len(finals) != 1:
            raise ValueError("MULTIPLE_FINAL_ANSWERS" if finals else "NO_EXPLICIT_FINAL")
        final_key = _answer_key(finals[0][0], choices)
        if final_key is None or final_key != _answer_key(gold, choices):
            raise ValueError("FINAL_ANSWER_MISMATCH")
        conclusions = [_answer_key(match[1], choices) for line, _ in lines if (match := _CONCLUSION.fullmatch(line))]
        conclusion_key = conclusions[-1] if conclusions else None
        if any(key is not None and key != final_key for key in conclusions):
            raise ValueError("REASONING_ANSWER_MISMATCH")
        spans, evidence, unknown = [finals[0][1]], 0, 0
        for line, span in lines:
            if not line or _FINAL.fullmatch(line):
                continue
            if conclusion := _CONCLUSION.fullmatch(line):
                if "=" in line:
                    # 'Thus 12 / 3 = 7.' is a content claim whether or not an answer key is also readable;
                    # every '=' of the conclusion must belong to a source-bound, recomputed equation.
                    claims = list(_EQUATION_IN.finditer(conclusion[1]))
                    for claim in claims:
                        if not _bound_equation(claim, facts, rules):
                            raise ValueError("CONTENT_CONTRADICTION" if _arithmetic_wrong(claim)
                                             else "UNSUPPORTED_CONTENT_CLAIM")
                    if not claims or conclusion[1].count("=") != len(claims):
                        raise ValueError("UNSUPPORTED_CONTENT_CLAIM")
                if _answer_key(conclusion[1], choices) is not None:
                    spans.append(span)
                else:
                    unknown += 1
                continue
            fact = _FACT_LINE.fullmatch(line)
            if fact and fact["ref"] in facts:
                if not _same(fact["value"].strip(), facts[fact["ref"]]["value"]):
                    raise ValueError("CONTENT_CONTRADICTION")
                spans.append(span)
                evidence += 1
                continue
            equation = _EQUATION.fullmatch(line)
            if equation and _bound_equation(equation, facts, rules):
                spans.append(span)
                evidence += 1
                continue
            if "=" in line:
                raise ValueError("UNSUPPORTED_CONTENT_CLAIM")
            unknown += 1
        if evidence == 0:
            raise ValueError("ANSWER_ONLY_C")
        if conclusion_key is None:
            raise ValueError("REASONING_ANSWER_UNKNOWN")
        return {"accepted": True, "reason": None, "verified_spans": sorted(spans),
                "unknown_lines": unknown, "fully_verified": unknown == 0,
                "final_answer": final_key, "verified_evidence_count": evidence}
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as exc:
        return {"accepted": False, "reason": str(exc) if isinstance(exc, ValueError) else "RESOLVE_SCHEMA"}


def _row(record: dict, output: dict, kind: str, spans: list[list[int]], tokenizer: Any,
         receipt: dict) -> dict:
    from .content_v16 import ROW_SCHEMA
    from .train import END_TOKEN
    from .train_v16 import token_byte_mask

    target, generation = output["text"], output["generation"]
    if generation.get("text") != target:
        raise ValueError("CHANGED_ORIGINAL_TARGET")
    byte_spans = [[len(target[:start].encode("utf-8")), len(target[:end].encode("utf-8"))]
                  for start, end in spans]
    mask = token_byte_mask(tokenizer, generation["completion_token_ids"], target, byte_spans,
                           end_token_id=tokenizer.convert_tokens_to_ids(END_TOKEN), supervise_end=False)
    if not any(mask):
        raise ValueError("NO_VERIFIED_TOKENS")
    return {"schema": ROW_SCHEMA, "row_id": record["problem_id"] + ":" + kind,
            "parent_id": record.get("parent_id", record["problem_id"]), "kind": kind,
            **{key: record[key] for key in ("problem_id", "source", "subject", "image_paths", "image_sha256")},
            "route": ROUTE_FULL if record["image_paths"] else ROUTE_LM,
            "prompt": output["prompt"] if kind == "N" else record["p0_text"], "target": target,
            "target_source": {"N": "SELF_NOTE", "C": "SELF_RESOLVE", "A1": "T1"}[kind], "generation": generation,
            "target_origin": "self", "layer": record["layer"],
            "status": output.get("status", "NOTE"),
            "complete": True, "probability_space": "stored-support", "loss_mask": mask,
            "verified_byte_spans": byte_spans, "verification_receipt": receipt, "supervise_end": False}


def first_solution_row(record: dict, *, tokenizer: Any,
                       choices: list[str] | tuple[str, ...] = ()) -> tuple[dict | None, str | None]:
    """Retain v15 admission for long correct T1s; this does not verify their reasoning."""
    from fine_tuning.notebook_rl import v8_parser

    try:
        output = record["t1"]
        actions = output["generation"]["completion_token_ids"]
        if output["prompt"] != record["p0_text"]:
            raise ValueError("T1_P0_MISMATCH")
        if problem := target_problem(output["text"], actions, output["finish_reason"], output["status"], min_tokens=256):
            raise ValueError(problem)
        if len(actions) > 4096:
            raise ValueError("TOO_LONG")
        if problem := answer_denial(output["text"]):
            raise ValueError("ANSWER_DENIAL:" + problem)
        final = (v8_parser.extract_mcq(output["text"], choices, output["finish_reason"])
                 if choices else v8_parser.extract_open(output["text"], output["finish_reason"], question=record["p0_text"]))
        if final["status"] != v8_parser.PARSED:
            raise ValueError("T1_FINAL:" + final["rule"])
        receipt = {"version": VERSION, "accepted": True, "verification_scope": "v15-admission-only",
                   "content_status": "UNKNOWN", "target_sha256": text_sha256(output["text"]),
                   "prompt_sha256": text_sha256(record["p0_text"])}
        return _row(record, output, "A1", [[0, len(output["text"])]], tokenizer, receipt), None
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc) if isinstance(exc, ValueError) else "T1_SCHEMA"


def build_self_note_rows(record: dict, *, tokenizer: Any, fact_catalog: list[dict], rules: dict,
                         choices: list[str] | tuple[str, ...], gold: str,
                         note_yield_gate: dict | None = None) -> tuple[list[dict], dict]:
    """Return only accepted C/N targets; the incorrect T1 remains conditioning/audit data."""
    note = verify_note(record, fact_catalog=fact_catalog, rules=rules, choices=choices, gold=gold)
    retry = verify_retry(record, note_receipt=note, fact_catalog=fact_catalog, rules=rules, choices=choices, gold=gold)
    rows = []
    decisions = {"N": note, "C": retry, "note_yield_gate": note_yield_gate or {"status": "NOT_MEASURED"}}
    if note["accepted"] and record.get("resolve") is not None and note["answer_specific_values"]:
        key = _answer_key(record["resolve"].get("extracted", ""), choices)
        if any(_answer_key(value, choices) != key for value in note["answer_specific_values"]):
            decisions["N"] = note | {"accepted": False, "reason": "NOTE_RESOLVE_CONTRADICTION"}
    for kind, stage, receipt in (("C", "resolve", retry), ("N", "note", decisions["N"])):
        if not receipt["accepted"]:
            continue
        if kind == "N" and not _yield_pass(note_yield_gate):
            decisions["N"] = receipt | {"accepted": False, "reason": "NOTE_YIELD_NOT_PASSED"}
            continue
        if kind == "N" and note_yield_gate is not None and any(note_yield_gate["receipts"][0]["identity"][key] != record[key]
                               for key in _IDENTITY_HASHES[:3]):
            decisions["N"] = receipt | {"accepted": False, "reason": "NOTE_YIELD_CONTEXT_MISMATCH"}
            continue
        try:
            rows.append(_row(record, record[stage], kind, receipt["verified_spans"], tokenizer, receipt))
        except ValueError as exc:
            decisions[kind] = receipt | {"accepted": False, "reason": str(exc)}
    return rows, decisions


def human_j_row(record: dict, fields: dict, *, tokenizer: Any,
                baseline_correct_length_median: float) -> tuple[dict | None, str | None]:
    """Dataset-authored solutions only; tokenize authored text once, never append EOS."""
    from .content_v16 import ROW_SCHEMA

    if (fields.get("dataset_authored") is not True or fields.get("ai_generated") is not False
            or not isinstance(fields.get("solution"), str) or not fields["solution"].strip()
            or fields.get("solution_sha256") != text_sha256(fields["solution"])):
        return None, "J_NOT_DATASET_AUTHORED"
    target = fields["solution"]
    if any(token in target for token in tokenizer.all_special_tokens):
        return None, "J_SPECIAL_TOKEN"
    actions = tokenizer.encode(target, add_special_tokens=False)
    if (not isinstance(baseline_correct_length_median, (int, float))
            or not 0 < baseline_correct_length_median < float("inf")):
        return None, "BASELINE_STYLE_NOT_MEASURED"
    if len(actions) < baseline_correct_length_median * 0.5:
        return None, "TARGET_STYLE_SHIFT"
    return {"schema": ROW_SCHEMA, "row_id": record["problem_id"] + ":J", "kind": "J",
            "parent_id": record.get("parent_id", record["problem_id"]),
            **{key: record[key] for key in ("problem_id", "source", "subject", "image_paths", "image_sha256")},
            "route": ROUTE_FULL if record["image_paths"] else ROUTE_LM,
            "prompt": record["p0_text"], "target": target, "target_source": "DATASET_HUMAN",
            "target_origin": "human", "layer": record["layer"],
            "complete": True, "probability_space": "raw-ce", "loss_mask": [True] * len(actions),
            "target_token_ids": actions, "supervise_end": False,
            "verification_receipt": {"dataset_authored": True, "solution_sha256": fields["solution_sha256"]}}, None


def validate_self_note_row(row: dict, record: dict, *, tokenizer: Any, fact_catalog: list[dict],
                           rules: dict, choices: list[str] | tuple[str, ...], gold: str,
                           note_yield_gate: dict | None = None) -> None:
    expected, _ = build_self_note_rows(record, tokenizer=tokenizer, fact_catalog=fact_catalog,
                                      rules=rules, choices=choices, gold=gold, note_yield_gate=note_yield_gate)
    if not any(candidate == row for candidate in expected):
        raise ValueError("SELF_NOTE_ROW_BINDING_MISMATCH")
