"""Source-bound v17 claims, independent repair admission, and sampled-token rows."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from decimal import InvalidOperation
import json
import math
import re
from typing import Any, Sequence

from . import self_note_v16 as bounded
from . import v16b_rows as previous
from .audit import digest, text_sha256
from .contract import (ROUTE_FULL, ROUTE_LM, a2_target_problem, answer_denial,
                       has_repetition, hint_leak_single_v2, is_loop)
from .single_grading import normalize_final
from .v4 import has_block_repetition

ROW_SCHEMA = "error-note-v17-row-v1"
LEDGER_SCHEMA = "v17-claim-ledger-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LENGTH = {"length", "max_tokens", "max_new_tokens"}
_EOS = {"<|im_end|>", "<|endoftext|>"}
_KINDS = {
    "geometry": ("SEGMENT_LABEL", "ANGLE_LABEL", "PARALLEL", "PERPENDICULAR"),
    "graph": ("VISIBLE_NODE", "EDGE_PRESENT", "EDGE_WEIGHT"),
    "scene": ("OBJECT_ATTRIBUTE", "OBJECT_COUNT"),
    "chart": ("CHART_VALUE",),
}


OBS_INSTRUCTION_V17 = "Observe only the image. "
# 2026-11-17: "at most 16" let the policy answer {"claims": []} (26 of 28 pod OBS outputs), so no observation could be
# checked or repaired. The list must now name every in-scope item; an empty list stays reward 0.
OBS_FORMAT_V17 = ("Return one JSON object with exactly claims, a list of 1 to 16 objects containing kind, entity, value, "
                  "and a unit field as specified by the generic types below. List every item the scope asks for, one "
                  "object per item; an empty list is not allowed. Answer with only this shape: "
                  '{"claims": [{"kind": ..., "entity": ..., "value": ..., "unit": ...}, ...]}. '
                  "Write entities and values observed in the image; no claim IDs. Supported kind order: ")
_LINE_DECORATION = re.compile(r"^[^\w]+")
NOTE_INSTRUCTION_V17 = (
    "Inspect your actual solution and image. Return JSON with exactly repairs and lesson. "
    "repairs is [] or one object with exactly claim_id and corrected_value. Correct only a contradicted "
    "current assertion about the same entity, property and unit; write no revised solution. "
    "lesson is an abstract lesson of at most 40 words, without answers, option identities, names or problem values."
)
RESOLVE_INSTRUCTION_V17 = previous.RESOLVE_INSTRUCTION_V16B

def _seal(body: dict) -> dict:
    return deepcopy(body) | {"sha256": digest(body)}


def _check_seal(value: dict) -> None:
    if not isinstance(value, dict) or value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("EVIDENCE_CHANGED")


def reward_v17(status: str, finish_reason: str, text: str, extracted: Any) -> tuple[float | None, str, bool]:
    if finish_reason in _LENGTH | {"stop", "eos"} and is_loop(text):
        return 0.0, "LOOP_CONFIRMED", True
    return previous.reward_v16b(status, finish_reason, text, extracted)


def _token_mask(tokenizer: Any, actions: list[int], text: str, spans: list[list[int]]) -> list[bool]:
    from .train import END_TOKEN
    from .train_v16 import token_byte_mask

    tokens = tokenizer.convert_ids_to_tokens(actions)
    end = actions[-1] if tokens[-1] in _EOS else tokenizer.convert_tokens_to_ids(END_TOKEN)
    return token_byte_mask(tokenizer, actions, text, spans, end_token_id=end, supervise_end=False)


def loop_token_mask_v17(tokenizer: Any, actions: list[int], text: str) -> list[bool]:
    if not is_loop(text):
        raise ValueError("LOOP_NOT_CONFIRMED")
    start = len(text[:-3000].encode("utf-8")) if len(text) > 3000 else 0
    return _token_mask(tokenizer, actions, text, [[start, len(text.encode("utf-8"))]])


def _json(text: str) -> tuple[Any, dict]:
    if re.search(r"<\|[^>]+\|>", text):
        raise ValueError("GENERATED_CONTROL_TOKEN")
    spans = previous.json_spans(text)
    value = json.loads(text, object_pairs_hook=bounded._unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))

    def finite(obj: Any) -> None:
        if isinstance(obj, float) and not math.isfinite(obj):
            raise ValueError("NONFINITE_JSON")
        if isinstance(obj, str):
            if re.fullmatch(r"[+-]?(?:nan|inf(?:inity)?)", obj.strip(), re.I):
                raise ValueError("NONFINITE_JSON")
            if re.search(r"<\|[^>]+\|>|[\ud800-\udfff]", obj):
                raise ValueError("GENERATED_CONTROL_TOKEN")
        if isinstance(obj, dict):
            for child in obj.values():
                finite(child)
        if isinstance(obj, list):
            for child in obj:
                finite(child)

    finite(value)
    return value, spans


def _output_binding(output: dict) -> str:
    return digest({"generation": output.get("generation"), "text": output.get("text"),
                   "prompt_sha256": text_sha256(output.get("prompt", ""))})


def _stored_output_v17(output: dict) -> bool:
    generation = output.get("generation", {})
    if not isinstance(generation, dict):
        return False
    actions = generation.get("completion_token_ids")
    return (isinstance(actions, list) and all(type(value) is int and value >= 0 for value in actions)
            and isinstance(output.get("text"), str)
            and all(holder.get("completion_tokens", len(actions)) == len(actions) for holder in (output, generation))
            and generation.get("text") == output["text"]
            and generation.get("finish_reason") == output.get("finish_reason"))


def _binding(record: dict, output: dict, tokenizer: Any) -> list[str]:
    problems = []
    if not isinstance(output, dict) or not _stored_output_v17(output):
        return ["CHANGED_ORIGINAL_GENERATION"]
    generation = output["generation"]
    for key in ("policy_sha256", "input_sha256", "request_sha256", "protocol_sha256"):
        if not isinstance(generation.get(key), str) or not _SHA.fullmatch(generation[key]):
            problems.append("GENERATION_BINDING:" + key)
    if type(generation.get("policy_version")) is not int:
        problems.append("GENERATION_BINDING:policy_version")
    if not isinstance(output.get("prompt"), str) or not output["prompt"].strip():
        problems.append("GENERATION_PROMPT_MISSING")
    for key in ("problem_id", "parent_id"):
        if key in generation and generation[key] != record["problem_id"]:
            problems.append("GENERATION_PARENT_MISMATCH")
    if "image_sha256" in generation and generation["image_sha256"] != record.get("image_sha256", []):
        problems.append("GENERATION_IMAGE_MISMATCH")
    if len(record.get("image_paths", [])) != len(record.get("image_sha256", [])):
        problems.append("IMAGE_BINDING_MISMATCH")
    if any(not isinstance(x, str) or not _SHA.fullmatch(x) for x in record.get("image_sha256", [])):
        problems.append("IMAGE_HASH_INVALID")
    try:
        from .train_v17 import _end_ids, _sample_mask
        _sample_mask(tokenizer, generation["completion_token_ids"], output["text"], _end_ids(tokenizer))
    except (KeyError, TypeError, ValueError) as exc:
        problems.append("SAMPLED_TOKEN_BINDING:" + str(exc))
    return problems


def _verifier(record: dict, verifier: dict | None) -> dict:
    if verifier is None:
        return {"facts": [], "supported_claim_kinds": [], "closed_world_kinds": [], "sha256": digest(None)}
    _check_seal(verifier)
    if verifier.get("problem_id") != record["problem_id"] or verifier.get("image_sha256") != record.get("image_sha256", []):
        raise ValueError("PRIVATE_VERIFIER_BINDING_MISMATCH")
    return verifier


def _key(claim: dict, verifier: dict) -> tuple:
    kind, entity = claim["kind"], claim["entity"]
    if kind in {"SEGMENT_LABEL", "ANGLE_LABEL"}:
        if isinstance(entity, list) and all(isinstance(part, str) for part in entity):
            entity = min(entity, list(reversed(entity)))
    if kind in {"PARALLEL", "PERPENDICULAR"}:
        if (isinstance(entity, list) and len(entity) == 2
                and all(isinstance(pair, list) and len(pair) == 2
                        and all(isinstance(point, str) for point in pair) for pair in entity)):
            entity = sorted(sorted(pair) for pair in entity)
    if kind in {"EDGE_PRESENT", "EDGE_WEIGHT"} and isinstance(entity, list) and not verifier.get("directed", False):
        entity = sorted(entity, key=str)
    default_unit = {"SEGMENT_LABEL": "length", "ANGLE_LABEL": "degree", "OBJECT_COUNT": "count"}.get(kind, "")
    return kind, json.dumps(entity, sort_keys=True, ensure_ascii=False, separators=(",", ":")), claim.get("unit") or default_unit


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        truth = {"true": True, "present": True, "yes": True, "false": False, "absent": False, "no": False}
        return truth.get(str(left).lower()) is truth.get(str(right).lower()) and str(left).lower() in truth and str(right).lower() in truth
    return bounded._same(str(left), str(right))


def _verdict(claim: dict, verifier: dict) -> tuple[str, str]:
    if claim["kind"] not in verifier.get("supported_claim_kinds", []):
        return "UNKNOWN", "UNSUPPORTED_KIND"
    key = _key(claim, verifier)
    facts = [f for f in verifier.get("facts", []) if _key(f, verifier) == key]
    if facts:
        if any(_same_value(claim["value"], f["value"]) for f in facts):
            return "VERIFIED", "SOURCE_FACT_MATCH"
        return "CONTRADICTED", "SOURCE_VALUE_MISMATCH"
    kinds = verifier.get("closed_world_kinds", [])
    if claim["kind"] in kinds:
        if claim["kind"] == "VISIBLE_NODE":
            return "UNKNOWN", "OUTSIDE_FIXED_SCOPE"
        if claim["kind"] in {"OBJECT_ATTRIBUTE", "OBJECT_COUNT"}:
            return "UNKNOWN", "OUTSIDE_FIXED_SCOPE"
        if claim["kind"] in {"EDGE_PRESENT", "EDGE_WEIGHT"}:
            nodes = [f["entity"] for f in verifier["facts"] if f["kind"] == "VISIBLE_NODE"]
            if (not isinstance(claim["entity"], list) or len(claim["entity"]) != 2
                    or any(type(n) is not int or n not in nodes for n in claim["entity"])):
                return "UNKNOWN", "OUTSIDE_FIXED_SCOPE"
            if claim["kind"] == "EDGE_PRESENT":
                present = any(f["kind"] == "EDGE_WEIGHT" and _key(f, verifier)[1] == key[1]
                              for f in verifier["facts"])
                if _same_value(claim["value"], present):
                    return "VERIFIED", "CERTIFIED_EDGE_PRESENCE" if present else "CERTIFIED_CLOSED_WORLD_ABSENCE"
                if _same_value(claim["value"], not present):
                    return "CONTRADICTED", "CERTIFIED_EDGE_PRESENCE" if present else "CERTIFIED_CLOSED_WORLD_ABSENCE"
                return "UNKNOWN", "INVALID_PRESENCE_VALUE"
        return "CONTRADICTED", "CERTIFIED_CLOSED_WORLD_ABSENCE"
    return "UNKNOWN", "UNANNOTATED_OR_OUTSIDE_SCOPE"


def build_obs_request_v17(item: dict, *, public_schema: dict | None = None) -> dict:
    from .v17_data import public_schema as source_schema

    layer = item.get("layer")
    if layer not in _KINDS:
        raise ValueError("OBS_UNSUPPORTED_LAYER")
    expected = source_schema(layer)
    schema = expected if public_schema is None else public_schema
    if (set(schema) - {"schema", "layer", "supported_claim_kinds", "scope_instruction", "max_claims", "entity_schema", "sha256"}
            or any(schema.get(key) != expected.get(key) for key in expected)):
        raise ValueError("PUBLIC_SCHEMA_PRIVATE_OR_CHANGED_FIELD")
    prompt = (OBS_INSTRUCTION_V17 + schema["scope_instruction"] + " " + OBS_FORMAT_V17
              + ", ".join(schema["supported_claim_kinds"]) + ". Generic entity and value types: "
              + json.dumps(schema["entity_schema"], ensure_ascii=False, sort_keys=True) + ".")
    return {"stage": "OBS", "problem_id": item["problem_id"], "prompt": prompt,
            "image_paths": list(item.get("image_paths", [])), "image_sha256": list(item.get("image_sha256", []))}


def _ledger(record: dict, stage: str, output: dict, verifier: dict, claims: list[dict]) -> dict:
    return _seal({"schema": LEDGER_SCHEMA, "problem_id": record["problem_id"], "stage": stage,
                  "generation_sha256": digest(output["generation"]), "original_text_sha256": text_sha256(output["text"]),
                  "original_token_ids_sha256": digest(output["generation"]["completion_token_ids"]),
                  "image_sha256": list(record.get("image_sha256", [])), "verifier_sha256": verifier["sha256"],
                  "claims": claims})


_FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\n(?P<body>.*?)\n?[ \t]*```\s*\Z", re.S)


def _obs_json(text: str) -> tuple[Any, dict]:
    """The OBS object, also when wrapped in one ```json fence or written as the bare claims list (2026-11-17 local
    F0 check: 3 of 6 graph answers were a fenced bare list). Spans stay offsets into the original text."""
    match = _FENCE.match(text)
    offset, body = (match.start("body"), match["body"]) if match else (0, text)
    obj, spans = _json(body)
    spans = {path: [start + offset, end + offset] for path, (start, end) in spans.items()}
    if isinstance(obj, list):
        obj = {"claims": obj}
        spans = {("claims" + path if path.startswith("[") else path): span for path, span in spans.items()}
    return obj, spans


def audit_observation_v17(record: dict, *, verifier: dict | None, tokenizer: Any) -> dict:
    output = record.get("obs", {})
    source = _verifier(record, verifier)
    reasons, claims = _binding(record, output, tokenizer), []
    if source.get("obs_eligible") is False:
        reasons.append("OBS_INELIGIBLE:" + str(source.get("obs_reason")))
    text = output.get("text", "")
    try:
        obj, spans = _obs_json(text)
        if not isinstance(obj, dict) or set(obj) != {"claims"} or not isinstance(obj["claims"], list) or len(obj["claims"]) > 16:
            raise ValueError("OBS_SCHEMA")
        for index, claim in enumerate(obj["claims"]):
            if (not isinstance(claim, dict) or set(claim) - {"kind", "entity", "value", "unit"}
                    or not {"kind", "entity", "value"} <= set(claim) or not isinstance(claim["kind"], str)
                    or isinstance(claim["value"], (dict, list)) or claim["value"] is None):
                raise ValueError("OBS_CLAIM_SCHEMA")
            verdict, reason = _verdict(claim, source)
            path = f"claims[{index}]"
            start, end = spans[path]
            claims.append({"claim_id": f"o{index}", "source_json_path": path,
                           "original_entity": deepcopy(claim["entity"]), "original_value": deepcopy(claim["value"]),
                           "original_quote": text[start:end], "original_byte_spans": [[len(text[:start].encode()), len(text[:end].encode())]],
                           "canonical_fact_key": list(_key(claim, source)), "kind": claim["kind"], "unit": _key(claim, source)[2],
                           "verdict": verdict, "reason": reason, "assertion": "ASSERTED",
                           "verifier_evidence_sha256": source["sha256"]})
        by_key: dict[tuple, list[dict]] = {}
        for claim in claims:
            by_key.setdefault(tuple(claim["canonical_fact_key"]), []).append(claim)
        for entries in by_key.values():
            if len({str(entry["original_value"]) for entry in entries}) > 1:
                reasons.append("CONFLICTING_ENTITY_CLAIMS")
                for entry in entries:
                    entry.update(verdict="UNKNOWN", reason="CONFLICTING_ENTITY_CLAIMS", assertion="UNKNOWN")
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        reasons.append(str(exc) if isinstance(exc, ValueError) else "OBS_SCHEMA")
        claims = []
    facts = {_key(fact, source) for fact in source.get("facts", [])}
    matched = {tuple(c["canonical_fact_key"]) for c in claims if c["verdict"] == "VERIFIED"} & facts
    tp = len(matched)
    fp = len({(tuple(c["canonical_fact_key"]), str(c["original_value"])) for c in claims if c["verdict"] == "CONTRADICTED"})
    fn = len(facts - matched)
    denominator = 2 * tp + fp + fn
    reward = 2 * tp / denominator if denominator else None
    if not denominator:
        reasons.append("OBS_NO_VERIFIABLE_TARGET")
    if output.get("finish_reason") in _LENGTH:
        reasons.append("LENGTH")
    elif output.get("finish_reason") not in {"stop", "eos"}:
        reasons.append("NOT_STOPPED")
    if is_loop(text):
        reasons.append("LOOP")
    if reasons:
        reward = 0.0 if facts else None
    pg = bool(facts) and not any(r in {"LENGTH", "NOT_STOPPED", "LOOP"} or r.startswith(("GENERATION_", "SAMPLED_", "CHANGED_", "IMAGE_", "OBS_INELIGIBLE")) for r in reasons)
    ledger = _ledger(record, "OBS", output, source, claims) if _stored_output_v17(output) else None
    return _seal({"schema": "v17-observation-receipt-v1", "problem_id": record["problem_id"], "ledger": ledger,
                  "reward": reward, "score": reward, "tp": tp, "fp": fp, "fn": fn,
                  "unknown": sum(c["verdict"] == "UNKNOWN" for c in claims), "pg": pg,
                  "accepted": not reasons, "reasons": list(dict.fromkeys(reasons)), "reason": reasons[0] if reasons else None})


def _source_catalog(verifier: dict) -> list[dict]:
    if verifier.get("repair_catalog_eligible") is False:
        return []
    catalog = []
    for index, fact in enumerate(verifier.get("facts", [])):
        kind, entity = fact["kind"], fact["entity"]
        # 2026-11-17 pod crash at RL block 20: 12 Geometry3K labels are LaTeX ("2\sqrt{3}", "\frac{1}{4}x"); the bounded
        # scanner (_catalog) accepts only atomic values, so such facts are left out of the checkable catalog.
        if not bounded._ATOM.fullmatch(str(fact["value"])):
            continue
        if kind in {"SEGMENT_LABEL", "ANGLE_LABEL"}:
            name = ("segment " if kind == "SEGMENT_LABEL" else "angle ") + "".join(entity)
        elif kind == "EDGE_WEIGHT":
            name = f"weight of edge {entity[0]} to {entity[1]}"
        elif kind == "OBJECT_ATTRIBUTE":
            name = f"object {entity[0]} from the left {entity[1]}"
        elif kind == "EDGE_PRESENT":
            name = f"edge {entity[0]} to {entity[1]}"
        elif kind == "OBJECT_COUNT":
            name = "number of objects in the scene"
        else:
            continue
        catalog.append({"fact_id": f"v17f{index}", "kind": "OBSERVATION", "entity": name,
                        "value": str(fact["value"]), "unit": fact.get("unit", "label"),
                        "evidence_origin": verifier["sha256"], "native_fact": deepcopy(fact)})
    return catalog


def _native_assertions(text: str, verifier: dict, catalog: list[dict]) -> list[dict]:
    indexed = {f["fact_id"]: f for f in catalog}
    claims = []
    for bound in previous.bind_statement(text, catalog):
        fact = indexed.get(bound["fact_ref"])
        native = fact.get("native_fact") if fact else None
        start, end = bound["char_span"]
        if native is not None:
            projected = native | {"value": True if bound["value"] == "present" else bound["value"]}
            verdict, reason = _verdict(projected, verifier)
        else:
            verdict = ("UNKNOWN" if fact is None else "VERIFIED" if bounded._same(bound["value"], fact["value"]) else "CONTRADICTED")
            reason = "BOUNDED_ASSERTION"
        claims.append({"kind": native["kind"] if native else "BOUND_FACT",
                       "original_entity": native["entity"] if native else fact["entity"] if fact else bound["quote"],
                       "original_value": True if native and bound["value"] == "present" else bound["value"], "original_quote": bound["quote"], "unit": native.get("unit", "") if native else fact["unit"] if fact else "",
                       "canonical_fact_key": list(_key(native, verifier)) if native else bound["fact_ref"],
                       "verdict": verdict if bound["assertion"] == "ASSERTED" else "UNKNOWN", "reason": reason,
                       "assertion": bound["assertion"], "original_byte_spans": [[len(text[:start].encode()), len(text[:end].encode())]],
                       "fact_ref": bound["fact_ref"]})
    if verifier.get("repair_catalog_eligible") is False:
        return claims
    for fact in verifier.get("facts", []):
        kind, entity = fact["kind"], fact["entity"]
        if kind == "CHART_VALUE":
            pattern = rf"(?<!\w){re.escape(entity[0])}(?:\s*[/,]\s*|\s+(?:at|in|for)\s+){re.escape(entity[1])}\s*(?:=|is|:)\s*(?P<value>{bounded._NUMBER})"
        elif kind == "OBJECT_COUNT":
            pattern = r"(?:number of (?:visible )?objects|object count)\s*(?:=|is|:)\s*(?P<value>\d+)"
        elif kind == "VISIBLE_NODE":
            pattern = rf"\bnode\s+{re.escape(str(entity))}\s+(?:is\s+)?(?:visible|present|drawn)\b"
        elif kind in {"PARALLEL", "PERPENDICULAR"}:
            first, second = ("".join(part) for part in entity)
            operator = r"(?:∥|parallel to)" if kind == "PARALLEL" else r"(?:⊥|perpendicular to)"
            pattern = rf"\b{re.escape(first)}\s*{operator}\s*{re.escape(second)}\b"
        else:
            continue
        for match in re.finditer(pattern, text, re.I):
            start, end = match.span()
            if any(c["original_byte_spans"] == [[len(text[:start].encode()), len(text[:end].encode())]] for c in claims):
                continue
            value = match.groupdict().get("value", True)
            assertion = previous._natural_assertion(text, [start, end], match[0])
            verdict, reason = _verdict(fact | {"value": value}, verifier)
            claims.append({"kind": kind, "original_entity": deepcopy(entity), "original_value": value,
                           "original_quote": match[0], "unit": fact.get("unit", ""),
                           "canonical_fact_key": list(_key(fact, verifier)), "verdict": verdict if assertion == "ASSERTED" else "UNKNOWN",
                           "reason": reason, "assertion": assertion,
                           "original_byte_spans": [[len(text[:start].encode()), len(text[:end].encode())]]})
    return claims


def audit_t1_v17(record: dict, *, verifier: dict | None, tokenizer: Any,
                 fact_catalog: Sequence[dict] = (), rules: dict | None = None) -> dict:
    output, source = record["t1"], _verifier(record, verifier)
    if problems := _binding(record, output, tokenizer):
        raise ValueError(";".join(problems))
    text, claims = output["text"], []
    catalog = _source_catalog(source) if verifier is not None else list(fact_catalog)
    claims.extend(_native_assertions(text, source, catalog))
    for equation in bounded._EQUATION_IN.finditer(text):
        start, end = equation.span()
        assertion = previous._assertion_status(text, [start, end], equation[0])
        try:
            operation = previous._operation(equation)
            corrected, rounding, correct = previous._rounded_equation(equation, operation)
        except (ValueError, InvalidOperation):
            continue
        claims.append({"kind": "ARITHMETIC", "original_entity": equation[0].split("=")[0].strip(),
                       "original_value": equation["result"], "original_quote": equation[0], "unit": "",
                       "canonical_fact_key": digest([equation[0].split("=")[0], operation]),
                       "verdict": "UNKNOWN" if assertion != "ASSERTED" else "VERIFIED" if correct else "CONTRADICTED",
                       "assertion": assertion, "operation": operation, "rounding": rounding,
                       "computed_value": corrected,
                       "original_byte_spans": [[len(text[:start].encode()), len(text[:end].encode())]]})
    claims.sort(key=lambda c: c["original_byte_spans"])
    for index, claim in enumerate(claims):
        claim.update(claim_id=f"t{index}", source_json_path=None, reason="BOUNDED_ASSERTION",
                     verifier_evidence_sha256=source["sha256"])
    return _ledger(record, "T1", output, source, claims)


def _checked_ledgers(record: dict, ledgers: list[dict], verifier: dict | None, tokenizer: Any,
                     fact_catalog: list[dict], rules: dict) -> list[dict]:
    claims, ids = [], set()
    for ledger in ledgers:
        _check_seal(ledger)
        stage = ledger.get("stage")
        if stage == "OBS" and verifier is None:
            raise ValueError("OBS_PRIVATE_VERIFIER_MISSING")
        expected = (audit_observation_v17(record, verifier=verifier, tokenizer=tokenizer)["ledger"] if stage == "OBS" else
                    audit_t1_v17(record, verifier=verifier, tokenizer=tokenizer, fact_catalog=fact_catalog, rules=rules) if stage == "T1" else None)
        if expected != ledger:
            raise ValueError("CLAIM_LEDGER_BINDING_MISMATCH")
        for claim in ledger["claims"]:
            if claim["claim_id"] in ids:
                raise ValueError("DUPLICATE_CLAIM_ID")
            ids.add(claim["claim_id"])
            claims.append(claim | {"stage": stage})
    return claims


def build_note_request_v17(record: dict, *, ledgers: list[dict], gold: str, tokenizer: Any = None) -> dict:
    claims = []
    for ledger in ledgers:
        _check_seal(ledger)
        output = record.get(str(ledger.get("stage", "")).lower(), {})
        if (ledger.get("problem_id") != record["problem_id"] or ledger.get("generation_sha256") != digest(output.get("generation"))
                or ledger.get("image_sha256") != record.get("image_sha256", [])):
            raise ValueError("CLAIM_LEDGER_BINDING_MISMATCH")
        for claim in ledger["claims"]:
            claims.append({key: claim[key] for key in ("claim_id", "original_entity", "original_value", "original_quote", "verdict")} | {"stage": ledger["stage"]})
    first = record["t1"]
    errors = any(c["verdict"] == "CONTRADICTED" for c in claims)
    if first["finish_reason"] in _LENGTH or is_loop(first["text"]):
        if not errors:
            raise ValueError("SKIP_NOTE_INCOMPLETE_WITHOUT_VERIFIED_ERROR")
    elif first.get("status") == "CORRECT" and not errors:
        raise ValueError("SKIP_NOTE_NO_ERROR")
    prompt = NOTE_INSTRUCTION_V17 + "\n\n" + json.dumps({"question": record["p0_text"], "first_solution": first["text"],
             "grading_result": first.get("status"), "given_answer": gold, "claims": claims}, ensure_ascii=False, sort_keys=True)
    if tokenizer is not None and len(tokenizer.encode(prompt, add_special_tokens=False)) > 16384:
        raise ValueError("NOTE_CONTEXT_BLOCKED")
    return {"stage": "NOTE", "problem_id": record["problem_id"], "prompt": prompt,
            "image_paths": list(record.get("image_paths", [])), "image_sha256": list(record.get("image_sha256", []))}


def _lesson(lesson: Any, record: dict, choices: list | tuple, gold: str, corrected: list[Any]) -> list[str]:
    if not isinstance(lesson, str) or not lesson.strip() or len(lesson.split()) > 40:
        return ["LESSON_SCHEMA"]
    leak = hint_leak_single_v2(lesson, gold, choices)
    if leak and leak.startswith("STEM:") and choices:
        stem = leak.partition(":")[2]
        if all(any(stem in word for word in re.findall(r"[a-z]+", option.lower())) for option in choices):
            # Shared topic words are public; numeric values and option directives remain blocked.
            leak = None
    reasons = [leak] if leak else []
    if re.search(r"\d", lesson):
        reasons.append("PROBLEM_NUMBER")
    for value in corrected:
        if str(value).strip() and re.search(r"(?<!\w)" + re.escape(str(value).strip()) + r"(?!\w)", lesson, re.I):
            reasons.append("CORRECTED_VALUE")
    return list(dict.fromkeys(reasons))


def verify_note_v17(record: dict, *, ledgers: list[dict], verifier: dict | None, choices: list | tuple,
                     gold: str, tokenizer: Any, fact_catalog: Sequence[dict] = (), rules: dict | None = None) -> dict:
    output = record.get("note", {})
    reasons, repairs, byte_spans, lesson, lesson_reasons = [], [], [], None, ["LESSON_UNAVAILABLE"]
    try:
        claims = _checked_ledgers(record, ledgers, verifier, tokenizer, list(fact_catalog), rules or {})
        expected_prompt = build_note_request_v17(record, ledgers=ledgers, gold=gold, tokenizer=tokenizer)["prompt"]
        reasons.extend(_binding(record, output, tokenizer))
        if output.get("prompt") != expected_prompt:
            reasons.append("NOTE_INPUT_CHANGED")
        if output.get("finish_reason") not in {"stop", "eos"}:
            reasons.append("NOTE_NOT_STOPPED")
        if len(output.get("generation", {}).get("completion_token_ids", [])) > 1024:
            reasons.append("NOTE_TOO_LONG")
        if is_loop(output.get("text", "")):
            reasons.append("NOTE_LOOP")
        obj, spans = _json(output["text"])
        if not isinstance(obj, dict) or set(obj) != {"repairs", "lesson"}:
            raise ValueError("NOTE_SCHEMA")
        lesson = obj["lesson"]
        raw_repairs = obj["repairs"]
        if not isinstance(raw_repairs, list) or len(raw_repairs) > 1:
            reasons.append("REPAIR_SCHEMA")
            raw_repairs = []
        indexed = {claim["claim_id"]: claim for claim in claims}
        corrected_values = [r["corrected_value"] for r in raw_repairs if isinstance(r, dict) and "corrected_value" in r]
        lesson_reasons = _lesson(lesson, record, choices, gold, corrected_values)
        for index, repair in enumerate(raw_repairs):
            problem = None
            if (not isinstance(repair, dict) or set(repair) != {"claim_id", "corrected_value"}
                    or not isinstance(repair["claim_id"], str) or not isinstance(repair["corrected_value"], str)):
                problem = "REPAIR_SCHEMA"
            claim = indexed.get(repair.get("claim_id")) if isinstance(repair, dict) else None
            if problem is None:
                if claim is None:
                    problem = "UNKNOWN_CLAIM_ID"
                elif claim["verdict"] != "CONTRADICTED" or claim.get("assertion") != "ASSERTED":
                    problem = "ORIGINAL_NOT_CONTRADICTED_ASSERTION"
                elif claim["kind"] == "ARITHMETIC":
                    if not bounded._same(repair["corrected_value"], claim["computed_value"]):
                        problem = "INCORRECT_ARITHMETIC_REPAIR"
                elif claim["kind"] == "BOUND_FACT":
                    fact = next((f for f in fact_catalog if f["fact_id"] == claim["fact_ref"]), None)
                    if fact is None or not bounded._same(repair["corrected_value"], fact["value"]):
                        problem = "INCORRECT_SOURCE_REPAIR"
                else:
                    repaired = {"kind": claim["kind"], "entity": claim["original_entity"],
                                "value": repair["corrected_value"], "unit": claim.get("unit", "")}
                    if _verdict(repaired, _verifier(record, verifier))[0] != "VERIFIED":
                        problem = "INCORRECT_SOURCE_REPAIR"
            if problem:
                reasons.append(problem)
                repairs.append({"accepted": False, "reason": problem})
                continue
            start, end = spans[f"repairs[{index}].corrected_value"]
            start, end = start + 1, end - 1
            span = [len(output["text"][:start].encode()), len(output["text"][:end].encode())]
            if span[0] == span[1]:
                reasons.append("EMPTY_REPAIR_CONTENT")
                continue
            byte_spans.append(span)
            assert claim is not None
            repairs.append({"accepted": True, "claim_id": repair["claim_id"], "stage": claim["stage"], "byte_span": span})
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        reasons.append(str(exc) if isinstance(exc, ValueError) else "NOTE_SCHEMA")
    mask = []
    if byte_spans:
        try:
            mask = _token_mask(tokenizer, output["generation"]["completion_token_ids"], output["text"], byte_spans)
        except ValueError as exc:
            reasons.append("N_MASK:" + str(exc))
    n_accepted = bool(byte_spans and mask and not reasons)
    lesson_accepted = isinstance(lesson, str) and not lesson_reasons
    binding_bad = any(r.startswith(("CLAIM_LEDGER", "EVIDENCE_", "GENERATION_", "SAMPLED_", "CHANGED_", "NOTE_INPUT", "NOTE_NOT", "NOTE_TOO", "NOTE_LOOP")) for r in reasons)
    return _seal({"schema": "v17-note-receipt-v1", "problem_id": record["problem_id"],
                  "note_output_sha256": digest(output), "output_binding_sha256": _output_binding(output), "ledger_sha256": [x["sha256"] for x in ledgers],
                  "accepted": n_accepted, "n_accepted": n_accepted, "verified_repair_count": sum(r["accepted"] for r in repairs),
                  "repairs": repairs, "lesson": {"text": lesson} if isinstance(lesson, str) else None,
                  "lesson_accepted": lesson_accepted, "lesson_leak_reasons": lesson_reasons,
                  "resolve_allowed": lesson_accepted and not binding_bad, "c_accepted": False,
                  "verified_byte_spans": byte_spans if n_accepted else [], "verified_content_token_mask": mask if n_accepted else [],
                  "reasons": list(dict.fromkeys(reasons)), "reason": reasons[0] if reasons else None})


def _same_answer(left: Any, right: Any, choices: list | tuple) -> bool:
    lkey, rkey = bounded._answer_key(str(left), choices), bounded._answer_key(str(right), choices)
    if lkey is not None and rkey is not None:
        return bounded._same(lkey, rkey)
    return normalize_final(str(left)) == normalize_final(str(right))


def verify_solution_v17(record: dict, *, stage: str = "t1", tokenizer: Any, gold: str,
                        choices: list | tuple = (), fact_catalog: Sequence[dict] = (), rules: dict | None = None,
                        note_receipt: dict | None = None, verifier: dict | None = None) -> dict:
    if stage not in {"t1", "resolve"}:
        raise ValueError("SOLUTION_STAGE")
    output = record.get(stage, {})
    reasons = _binding(record, output, tokenizer)
    text, generation = output.get("text", ""), output.get("generation", {})
    actions = generation.get("completion_token_ids", [])
    if output.get("finish_reason") not in {"stop", "eos"}:
        reasons.append("NOT_STOPPED")
    if not 0 < len(actions) <= 8192:
        reasons.append("OUTPUT_LIMIT")
    if len(generation.get("prompt_token_ids", [])) + len(actions) > 24576:
        reasons.append("CONTEXT_LIMIT")
    if has_repetition(actions) or has_block_repetition(actions) or is_loop(text):
        reasons.append("REPETITION")
    if output.get("status") != "CORRECT":
        reasons.append("NOT_CORRECT")
    finals, remainder = [], []
    for line in text.splitlines():
        clean = line.replace("**", "").replace("__", "")
        # 2026-11-17: a decorated final line ("✅ Final Answer: 4", "- Answer: B") is still the final line.
        match = bounded._FINAL.fullmatch(_LINE_DECORATION.sub("", clean))
        if match:
            finals.append(match[1])
        else:
            remainder.append(clean)
    extracted = output.get("extracted")
    # P0 does not ask for a "Final answer:" line, so the graded extraction (V8, status CORRECT) is the final answer
    # when the model ends with a boxed/bold/sentence answer; any explicit final lines must all agree with it.
    if len(finals) > 1:  # repeated final lines stay refused, even when identical
        reasons.append("MULTIPLE_FINAL_ANSWERS")
    elif finals and (not _same_answer(finals[-1], gold, choices) or not _same_answer(finals[-1], extracted, choices)):
        reasons.append("FINAL_ANSWER_MISMATCH")
    elif not finals and (extracted is None or not str(extracted).strip() or not _same_answer(extracted, gold, choices)):
        reasons.append("NO_EXPLICIT_FINAL")
    for line in remainder:
        if match := bounded._CONCLUSION.fullmatch(line):
            key = bounded._answer_key(match[1], choices)
            if key is not None and not _same_answer(key, gold, choices):
                reasons.append("FINAL_CONTRADICTION")
    if answer_denial(text):
        reasons.append("ANSWER_DENIAL")
    source = _verifier(record, verifier)
    effective_catalog = _source_catalog(source) if verifier is not None else list(fact_catalog)
    native_claims = _native_assertions(text, source, effective_catalog)
    process_errors = previous.t1_error_scan(text, effective_catalog, rules or {})
    process_errors.extend({"kind": c["kind"], "earlier_quote": c["original_quote"]} for c in native_claims if c["verdict"] == "CONTRADICTED")
    if process_errors:
        reasons.append("PROCESS_CONTRADICTION")
    nonfinal = "\n".join(remainder)
    evidence = sum(c["assertion"] == "ASSERTED" and c["verdict"] == "VERIFIED" for c in _native_assertions(nonfinal, source, effective_catalog))
    for equation in bounded._EQUATION_IN.finditer(nonfinal):
        if previous._assertion_status(nonfinal, list(equation.span()), equation[0]) == "ASSERTED":
            try:
                evidence += previous._rounded_equation(equation, previous._operation(equation))[2]
            except (ValueError, InvalidOperation):
                pass
    words = re.findall(r"[a-z]+", nonfinal.lower())
    answer_words = set(re.findall(r"[a-z]+", " ".join(map(str, (*choices, gold))).lower()))
    content_words = set(words) - answer_words - {"the", "a", "an", "is", "answer", "option", "correct", "therefore", "thus", "conclusion", "final", "it"}
    # 2026-11-17: requiring a VERIFIED claim on catalog sources rejected ~97% of correct solutions (claims are almost
    # never verifiable), which disabled A1/PREP_SELF/C. A bare answer is still refused; contradicted facts still
    # reject the row through PROCESS_CONTRADICTION above.
    if not evidence and len(content_words) < 3:
        reasons.append("ANSWER_ONLY")
    if stage == "t1" and output.get("prompt") != record.get("p0_text"):
        reasons.append("A1_INPUT_NOT_P0")
    if stage == "resolve":
        if note_receipt is None:
            reasons.append("SAFE_LESSON_RECEIPT_MISSING")
        else:
            _check_seal(note_receipt)
            if note_receipt.get("note_output_sha256") != digest(record.get("note")) or not note_receipt.get("resolve_allowed"):
                reasons.append("RESOLVE_NOT_ALLOWED")
            lesson = (note_receipt.get("lesson") or {}).get("text", "")
            if problem := a2_target_problem(text, lesson):
                reasons.append(problem)
            expected = previous.build_resolve_prompt_v16b(record["p0_text"], {"text": lesson})
            if output.get("prompt") != expected:
                reasons.append("RESOLVE_INPUT_CHANGED")
        if any(record.get(key) is not None for key in ("resolve2", "resolve3")):
            reasons.append("MULTIPLE_RESOLVE_ATTEMPTS")
    return _seal({"schema": "v17-solution-receipt-v1", "problem_id": record["problem_id"], "stage": stage,
                  "output_sha256": digest(output), "output_binding_sha256": _output_binding(output), "record_binding_sha256": digest({key: record.get(key) for key in ("problem_id", "p0_text", "image_paths", "image_sha256")}),
                  "accepted": not reasons, "reasons": list(dict.fromkeys(reasons)), "reason": reasons[0] if reasons else None,
                  "process_errors": process_errors, "verified_evidence_count": evidence,
                  "knowledge_status": "RESULT_CONDITIONED_PARTIALLY_CHECKED"})


def _sft_row(record: dict, kind: str, tokenizer: Any, receipt: dict) -> dict | None:
    _check_seal(receipt)
    if not receipt.get("n_accepted" if kind == "N" else "accepted"):
        return None
    stage = {"A1": "t1", "C": "resolve", "N": "note", "PREP_SELF": "t1"}[kind]
    output = record[stage]
    if receipt.get("note_output_sha256" if kind == "N" else "output_sha256") != digest(output):
        raise ValueError("ROW_RECEIPT_OUTPUT_CHANGED")
    if problems := _binding(record, output, tokenizer):
        raise ValueError(";".join(problems))
    if receipt.get("problem_id") != record["problem_id"] or kind != "N" and receipt.get("stage") != stage:
        raise ValueError("ROW_RECEIPT_PARENT_OR_STAGE")
    if kind != "N" and receipt["record_binding_sha256"] != digest({key: record.get(key) for key in ("problem_id", "p0_text", "image_paths", "image_sha256")}):
        raise ValueError("ROW_RECEIPT_INPUT_CHANGED")
    text = output["text"]
    spans = receipt["verified_byte_spans"] if kind == "N" else [[0, len(text.encode())]]
    mask = _token_mask(tokenizer, output["generation"]["completion_token_ids"], text, spans)
    stages = ["t1"] if kind in {"A1", "PREP_SELF"} else ["t1", "obs", "note"] if kind == "N" else ["t1", "obs", "note", "resolve"]
    prompt = output["prompt"] if kind == "N" else record["p0_text"]
    return deepcopy({"schema": ROW_SCHEMA, "row_id": record["problem_id"] + ":" + kind, "kind": kind,
        **{key: record.get(key, [] if key.startswith("image_") else "") for key in ("problem_id", "source", "subject", "layer", "image_paths", "image_sha256")},
        "parent_id": record["problem_id"], "block_index": record.get("block_index"),
        "policy_version": output["generation"]["policy_version"], "route": ROUTE_FULL if record.get("image_paths") else ROUTE_LM,
        "prompt": prompt, "target": text, "target_source": {"A1": "T1", "C": "SELF_RESOLVE", "N": "SELF_NOTE", "PREP_SELF": "T1"}[kind],
        "target_origin": "self", "generation": output["generation"],
        "source_generations": [record[s]["generation"] for s in stages if isinstance(record.get(s), dict)],
        "generation_prompt_sha256": text_sha256(output["prompt"]), "training_prompt_sha256": text_sha256(prompt),
        "admission_receipt_sha256": receipt["sha256"], "verifier_receipt_sha256": receipt["sha256"],
        "verification_receipt": receipt, "loss_probability_space": "raw-full-vocab-v1", "probability_space": "raw-full-vocab-v1",
        "loss_token_mask": mask, "loss_mask": mask, "verified_content_token_mask": mask,
        "verified_byte_spans": spans, "mask_mode": "verified-corrected-value-content-only-v1" if kind == "N" else "all-non-eos",
        "knowledge_status": "VERIFIED_REPAIR_CONTENT" if kind == "N" else "RESULT_CONDITIONED_PARTIALLY_CHECKED",
        "supervise_end": False, "status": output.get("status", "NOTE"), "complete": True})


def a1_row_v17(record: dict, *, tokenizer: Any, receipt: dict) -> dict | None:
    return _sft_row(record, "A1", tokenizer, receipt)


def c_row_v17(record: dict, *, tokenizer: Any, receipt: dict) -> dict | None:
    return _sft_row(record, "C", tokenizer, receipt)


def n_row_v17(record: dict, *, tokenizer: Any, receipt: dict) -> dict | None:
    return _sft_row(record, "N", tokenizer, receipt)


def prep_rows_with_receipts_v17(records: list[dict], *, tokenizer: Any,
                                 max_self: int = 160, max_human: int = 128) -> tuple[list[dict], list[dict]]:
    if not 0 <= max_self <= 160 or not 0 <= max_human <= 128:
        raise ValueError("PREP_ROW_LIMIT")
    if len({r["problem_id"] for r in records}) != len(records):
        raise ValueError("DUPLICATE_PREP_PARENT")
    candidates: dict[str, list[dict]] = {"PREP_SELF": [], "PREP_HUMAN": []}
    receipts = []
    for record in records:
        evaluation = record.get("evaluation", record)
        contract = problem_contract_v17(record)
        receipt = verify_solution_v17(record, tokenizer=tokenizer, gold=str(evaluation["gold"]),
            choices=evaluation.get("choices", []), fact_catalog=contract["fact_catalog"],
            rules=contract["rules"], verifier=contract["verifier"])
        receipts.append(_seal({"problem_id": record["problem_id"], "kind": "PREP_SELF", "admission": receipt}))
        if receipt["accepted"]:
            self_row = _sft_row(record, "PREP_SELF", tokenizer, receipt)
            assert self_row is not None
            candidates["PREP_SELF"].append(self_row)
        human = previous._prep_human(record, tokenizer)
        human_receipt = _seal({"problem_id": record["problem_id"], "kind": "PREP_HUMAN",
                              "accepted": human is not None,
                              "reasons": [] if human is not None else ["NATIVE_HUMAN_REFERENCE_NOT_ADMISSIBLE"]})
        receipts.append(human_receipt)
        if human is not None:
            actions = tokenizer.encode(human["target"], add_special_tokens=False)
            mask = [True] * len(actions) + [False]
            verification = _seal({**{k: v for k, v in human["verification_receipt"].items() if k != "sha256"},
                                  "accepted": True, "knowledge_status": "DATASET_AUTHORED"})
            human.update(schema=ROW_SCHEMA, parent_id=record["problem_id"], block_index=record.get("block_index"),
                generation_prompt_sha256=text_sha256(human["prompt"]), training_prompt_sha256=text_sha256(human["prompt"]),
                admission_receipt_sha256=verification["sha256"], verifier_receipt_sha256=verification["sha256"],
                verification_receipt=verification, loss_probability_space="raw-full-vocab-v1", probability_space="raw-full-vocab-v1",
                loss_token_mask=mask, loss_mask=mask, verified_content_token_mask=mask,
                verified_byte_spans=[[0, len(human["target"].encode())]], mask_mode="all-non-eos",
                knowledge_status="DATASET_AUTHORED")
            candidates["PREP_HUMAN"].append(human)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for kind, limit in (("PREP_SELF", max_self), ("PREP_HUMAN", max_human)):
        for row in sorted(candidates[kind], key=lambda r: text_sha256(r["row_id"]))[:limit]:
            buckets[(row["source"], kind)].append(row)
    rows = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                rows.append(buckets[key].pop(0))
    return rows, receipts


def prep_rows_v17(records: list[dict], *, tokenizer: Any, max_self: int = 160, max_human: int = 128) -> list[dict]:
    return prep_rows_with_receipts_v17(records, tokenizer=tokenizer, max_self=max_self, max_human=max_human)[0]


def problem_contract_v17(item: dict) -> dict:
    from .v17_data import private_verifier

    verifier = private_verifier(item)
    return _seal({"supported": verifier.get("repair_catalog_eligible", False), "verifier": verifier,
                  "fact_catalog": _source_catalog(verifier), "rules": {}, "problem_id": item["problem_id"]})


def observation_eligibility_v17(item: dict) -> dict:
    from .v17_data import capability_manifest

    capability = capability_manifest(item)
    return {"eligible": capability["obs_eligible"], "reason": capability["obs_reason"],
            "layer": item.get("layer"), "verifier_sha256": capability["private_verifier_sha256"]}
