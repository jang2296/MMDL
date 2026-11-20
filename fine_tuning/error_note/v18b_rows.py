"""v18b public OBS context, unique source claims and unaltered sampled N tokens."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import partial
import inspect
import json
import re

from . import v17_rows as v17
from . import v18_rows as v18
from . import v18b_claims as checked
from .audit import digest, text_sha256

ROW_SCHEMA = v18.ROW_SCHEMA
LEDGER_SCHEMA = "v18b-claim-ledger-v1"
NOTE_INSTRUCTION_V18B = (
    "Inspect your actual T1 solution, the image, and any supplied OBS question and your original OBS response. "
    'Return JSON with exactly repairs and lesson: {"repairs": [], "lesson": "..."}. '
    'repairs must be an array with zero or one object: {"stage": "T1 or OBS", "quote": "...", '
    '"corrected_value": "...", "unit": "..."}. '
    "Use only an available stage. For OBS, quote your response's value field (the scalar itself; "
    "outer JSON string quotes are optional), correcting that exact public target and unit. "
    "For T1, quote one exact unique passage containing one complete assertion in your actual solution; "
    "correct the same entity and property, or the same path endpoints. Do not quote multiple assertions, "
    "a hypothetical or withdrawn statement, or the final answer as an intermediate correction. "
    "If there is no actual checkable error, use an empty array. Do not write a revised solution. "
    "lesson is an abstract lesson of at most 40 words without answers, option identities, names, "
    "entities or problem values."
)


OBS_INSTRUCTION_V18B = (
    "Answer the visual question below using only the image. Return one JSON object with exactly value "
    "and unit, as shown in the required response shape. For boolean use bare JSON true or false, never 1, 0 or a quoted "
    "boolean; for string use a JSON string. Use null if unreadable. Copy unit exactly as requested, "
    "including the empty string. Do not answer any other question."
)
def render_note_target_last(body: dict) -> str:
    """Same public NOTE context, with T1 background first and the unchanged visual target last."""
    required = {"original_t1", "observation", "observation_prior_value", "observation_question"}
    if not required <= body.keys() or set(body["observation_question"]) != {"target", "unit"}:
        raise ValueError("unexpected public NOTE body")
    if not all(isinstance(body["observation_question"][key], str) for key in ("target", "unit")):
        raise ValueError("visual target/unit must be the exact public strings")
    extra = {key: value for key, value in body.items() if key not in required}
    def dump(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    lines = [
        "Review your own previous work. The original T1 question below is background for a separate visual recheck.",
        "Original T1 background (the answer grade refers only to its final answer):",
        dump(body["original_t1"]),
        "Other unchanged request metadata:",
        dump(extra),
        "Your original OBS response:",
        body["observation"],
        "Your prior OBS scalar: " + dump(body["observation_prior_value"]),
        "Required unit (copy exactly): " + dump(body["observation_question"]["unit"]),
        "Return one JSON object with visual_check and lesson. visual_check must contain read_now, quote, unit, decision, in that order.",
        "In read_now write your current reading of the requested visual target: JSON true or false for a boolean, a string for a name or label, or null if unreadable.",
        "In quote copy only your original OBS scalar as a string; its original JSON string outer quotes are optional. In unit copy the required unit exactly.",
        "In decision write keep if your current reading agrees with your old OBS scalar, correct only if you read a different value, or unsure if you cannot decide. No correction is required.",
        "An optional repairs array may contain zero or one T1 repair with exactly stage (T1), quote (one exact unique complete assertion from first_solution), corrected_value (string), and unit (string).",
        "A T1 repair must correct the same entity/property or path endpoints, never a hypothetical, withdrawn assertion, or final answer as an intermediate correction. Do not write a revised solution.",
        "lesson must be an abstract lesson of at most 40 words without answers, option identities, names, entities or problem values.",
        "For visual_check, read the image for the following visual question, independently of the original T1 question, then compare with your prior OBS scalar even if you agree.",
        "Visual question: " + body["observation_question"]["target"],
    ]
    return "\n".join(lines)


def prompts_sha256() -> str:
    return digest({"version": "v18b", "base": v18.prompts_sha256(), "note": NOTE_INSTRUCTION_V18B,
                   "query_builder": inspect.getsource(v18._query),
                   "balanced_query": inspect.getsource(checked.balanced_query),
                   "note_builder": inspect.getsource(v18.build_note_request_v18),
                   "obs": OBS_INSTRUCTION_V18B, "note_obs_renderer": inspect.getsource(render_note_target_last),
                   "obs_builder": inspect.getsource(v18.build_obs_request_v18),
                   "obs_binding": inspect.getsource(bound_observation_request_v18b)})


def problem_contract_v18b(item: dict) -> dict:
    source = checked.source_for(item, v18._source(item))
    return v17._seal({"supported": source.get("repair_catalog_eligible", False), "verifier": source,
                      "fact_catalog": v17._source_catalog(source), "rules": {}, "problem_id": item["problem_id"]})


def bind_graph_observation_targets_v18b(items: list[dict]) -> list[dict]:
    """Fix equal graph classes by hash rank before generation; no observed answers are used."""
    result = deepcopy(items)
    eligible = []
    for item in result:
        if item.get("layer") != "graph":
            continue
        source = problem_contract_v18b(item)["verifier"]
        try:
            for present in (False, True):
                candidate = item | {"v18b_obs_class": present}
                checked.balanced_query(candidate, source)
            v18._query(item, source, _v18b=True)
        except ValueError:
            continue
        eligible.append((item, source))
    eligible.sort(key=lambda pair: text_sha256(pair[0]["problem_id"] + ":v18b-obs-balance"))
    assignment_sha256 = digest([item["problem_id"] for item, _ in eligible])
    for rank, (item, source) in enumerate(eligible):
        item["v18b_obs_balance"] = v17._seal({"schema": "v18b-balanced-obs-assignment-v1", "problem_id": item["problem_id"],
            "image_sha256": item.get("image_sha256", []), "verifier_sha256": source["sha256"],
            "rank": rank, "total": len(eligible), "assignment_sha256": assignment_sha256, "present": bool(rank % 2)})
    return result


def bound_observation_request_v18b(record: dict) -> dict:
    if record.get("allow_historical_obs_request") is True:
        expected = v18.build_obs_request_v18(record)
        public = json.loads(expected["prompt"].split("\n", 1)[1])
        expected = expected | {"public_query": public, "historical_query": True}
    else:
        expected = v18.build_obs_request_v18(record, _v18b=True)
    output = record.get("obs", {})
    if output.get("prompt") != expected["prompt"]:
        raise ValueError("OBS_INPUT_CHANGED")
    supplied = record.get("obs_request")
    if supplied is not None and (supplied.get("prompt") != expected["prompt"] or
                                  supplied.get("public_query_sha256") != expected["public_query_sha256"]):
        raise ValueError("OBS_REQUEST_CHANGED")
    return expected


def bind_historical_observation_v18b(record: dict) -> dict:
    result = deepcopy(record)
    result["allow_historical_obs_request"] = True
    bound_observation_request_v18b(result)
    return result


def observation_baseline_v18b(records: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    for record in records:
        receipt = record.get("obs_score") or record.get("observation_receipt") or {}
        claims = (receipt.get("ledger") or record.get("obs_ledger") or {}).get("claims", [])
        if receipt.get("pg") is not True or len(claims) != 1 or claims[0].get("kind") != "EDGE_PRESENT":
            continue
        claim = claims[0]
        if type(claim.get("original_value")) is not bool or claim.get("verdict") not in {"VERIFIED", "CONTRADICTED"}:
            continue
        truth = claim["original_value"] if claim["verdict"] == "VERIFIED" else not claim["original_value"]
        counts["graded"] += 1
        counts["correct"] += claim["verdict"] == "VERIFIED"
        counts["always_true_correct" if truth else "always_false_correct"] += 1
    total = counts["graded"]
    return {"schema": "v18b-graph-observation-baseline-v1", **{key: counts[key] for key in
        ("graded", "correct", "always_true_correct", "always_false_correct")},
        "constant_best_correct": max(counts["always_true_correct"], counts["always_false_correct"]),
        "accuracy": counts["correct"] / total if total else None,
        "constant_best_accuracy": max(counts["always_true_correct"], counts["always_false_correct"]) / total if total else None}


def clevr_final_format_receipt(record: dict, gold: str, choices, body: dict) -> dict:
    """Certify only the final answer to the same CLEVR shape question."""
    output = record["t1"]
    body.update(schema="v18b-format-content-receipt-v1", semantic_status="UNKNOWN",
                semantic_certainty="UNVERIFIED")
    if choices or not record.get("native_id") or not record.get("source_revision"):
        return v17._seal(body)
    question = v18._question(record)
    if (output.get("finish_reason") not in {"stop", "eos"}
            or not record.get("p0_text") or output.get("prompt") != record["p0_text"]
            or question != record["p0_text"].removeprefix("Question: ").strip()):
        return v17._seal(body | {"reason": "CLEVR_QUESTION_CONTEXT_UNCERTIFIED"})
    # Only bounded CLEVR descriptions certify the final answer, never arbitrary shape mentions.
    attributes = r"(?:(?:small|large|tiny|big|gray|red|blue|green|brown|purple|cyan|yellow|metal|metallic|shiny|rubber|matte)\s+){1,3}"
    root = attributes + r"(?:object|thing)"
    entity = attributes + r"(?:object|thing|cube|sphere|ball|cylinder|block)"
    relation = r"that is (?:on the (?:left|right) side of|to the (?:left|right) of|behind|in front of) the " + entity
    question_match = re.fullmatch(r"(?:What shape is|What is the shape of) the (?P<root>" + root
        + r")(?P<relations>(?: " + relation + r"){0,2})\?", question, re.I)
    if question_match is None:
        return v17._seal(body | {"reason": "CLEVR_SHAPE_SCOPE_UNCERTIFIED"})
    referent = question_match["root"]
    if len(re.findall(r"\b" + re.escape(referent) + r"\b", question, re.I)) != 1:
        return v17._seal(body | {"reason": "CLEVR_SHAPE_SCOPE_UNCERTIFIED"})
    text = output.get("text", "")
    lines = [m for m in re.finditer(r"[^\n]+", text) if m[0].strip()]
    if not lines:
        return v17._seal(body | {"reason": "NO_FINAL_DECLARATION"})
    last = lines[-1]
    clean = last[0].replace("**", "").replace("__", "").strip()
    declaration_pattern = (r"(?:(?:Therefore|Thus|So),?\s+)?[Tt]he shape of the (?P<referent>"
        + re.escape(referent) + r"(?:" + re.escape(question_match["relations"]) + r")?) is (?:a |an )?"
        + r"(?P<shape>cube|sphere|cylinder)\.")
    declaration = re.fullmatch(declaration_pattern, clean, re.I)
    if declaration is None:
        return v17._seal(body | {"reason": "CLEVR_FINAL_DECLARATION_UNCERTIFIED"})
    if any(re.fullmatch(declaration_pattern, line[0].replace("**", "").replace("__", "").strip(), re.I)
           or re.fullmatch(r"(?:Final answer|Answer)\s*:\s*(?:a )?(?:cube|sphere|cylinder)[.!]?",
                           line[0].replace("**", "").replace("__", "").strip(), re.I) for line in lines[:-1]):
        return v17._seal(body | {"reason": "MULTIPLE_FINAL_DECLARATIONS"})
    candidate = declaration["shape"].casefold()
    equivalent = candidate == str(gold).strip().casefold()
    format_failed = output.get("status") != "CORRECT"
    subject = list(re.finditer(r"\b" + re.escape(candidate) + r"\b", last[0], re.I))[-1]
    start, end = last.start(), last.end()
    spans = [[len(text[:start].encode()), len(text[:start + subject.start()].encode())]]
    if re.search(r"\w", last[0][subject.end():]):
        spans.append([len(text[:start + subject.end()].encode()), len(text[:end].encode())])
    return v17._seal(body | {"accepted": equivalent and format_failed, "content_equivalent": equivalent,
        "semantic_status": "CORRECT" if equivalent else "INCORRECT",
        "semantic_certainty": "SOURCE_FINAL_DECLARATION_ONLY", "extracted_without_gold": candidate,
        "question_scope": {"referent": referent, "relations": question_match["relations"]},
        "declaration_scope": {"referent": declaration["referent"]},
        "byte_spans": spans if equivalent and format_failed else [],
        "reason": "CONTENT_MATCH_FORMAT_FAILURE" if equivalent and format_failed else "ALREADY_CORRECT" if equivalent else "CONTENT_MISMATCH",
        "certification_scope": "SOURCE_FINAL_SHAPE_ONLY", "human_verified": False})


def reconcile_solution_v18b(record: dict, *, stage: str, source: dict, tokenizer, old_body: dict,
                           reasons: list[str], ledger: dict, process_spans: list, used: list,
                           note_receipt: dict | None, gold: str, choices) -> dict:
    text = record[stage]["text"]
    claims = ledger["claims"]
    # Both the old scalar scanner and the new ledger now defer to the same source-bound claims.
    errors = [{"kind": claim["kind"], "earlier_quote": claim["original_quote"],
               "original_byte_spans": claim["original_byte_spans"], "reason": claim.get("reason")}
              for claim in claims if claim.get("assertion") == "ASSERTED" and claim["verdict"] == "CONTRADICTED"]
    reasons = [reason for reason in reasons if reason != "PROCESS_CONTRADICTION"]
    if errors:
        reasons.append("PROCESS_CONTRADICTION")
    if any(claim.get("reason") == "UNIT_MISMATCH" for claim in claims):
        reasons.append("PROCESS_UNIT_UNVERIFIED")
    # The old process-use scan cannot distinguish printed expressions from derived lengths.
    valid_quotes = {claim["original_quote"] for claim in claims if claim["verdict"] == "VERIFIED" and claim.get("assertion") == "ASSERTED"}
    used = [claim for claim in used if claim["original_quote"] in valid_quotes]
    process_spans = [span for claim in used for span in claim["verified_use_byte_spans"]]
    for claim in claims:
        if claim["kind"] == "GRAPH_PATH" and claim["verdict"] == "VERIFIED" and claim.get("assertion") == "ASSERTED":
            spans = claim["original_byte_spans"]
            used.append(claim | {"verified_use_byte_spans": spans})
            process_spans.extend(spans)
    if used and "ANSWER_ONLY" in reasons:
        reasons.remove("ANSWER_ONLY")
    corrected_spans = []
    for repair in (note_receipt or {}).get("repairs", []):
        if not repair.get("accepted"):
            continue
        for claim in used:
            same = claim["kind"] == repair.get("kind") and claim.get("unit", "") == repair.get("unit", "")
            if claim["kind"] == "GRAPH_PATH":
                same = same and re.sub(r"\s", "", repair["corrected_value"]).replace("→", "->") == "->".join(map(str, claim["original_entity"]))
            else:
                same = same and claim["original_entity"] == repair.get("entity") and v17._same_value(claim["original_value"], repair["corrected_value"])
            if same:
                corrected_spans.extend(claim["verified_use_byte_spans"])
    if stage == "resolve" and record["t1"].get("status") == "CORRECT" and not corrected_spans:
        form = record.get("t1_policy_receipt", {}).get("format_receipt", {})
        if not form.get("accepted"):
            reasons.append("CORRECT_T1_WITHOUT_VERIFIED_CORRECTION_USE")
    form = v18.format_receipt_v18(record, gold=gold, choices=choices, _v18b=True) if stage == "t1" else None
    if form and form.get("accepted"):
        reasons.append("FORMAT_CORRECTION_REQUIRED")
    if "MULTIPLE_FINAL_ANSWERS" in reasons:
        finals = [match[1] for line in text.splitlines() if (match := v18.bounded._FINAL.fullmatch(v17._LINE_DECORATION.sub("", line.replace("**", "").replace("__", ""))))]
        if finals and all(v17._same_answer(value, gold, choices) for value in finals):
            reasons.remove("MULTIPLE_FINAL_ANSWERS")
    reasons = list(dict.fromkeys(reasons))
    spans = [list(span) for span in sorted(set(map(tuple, process_spans)))]
    corrected = [list(span) for span in sorted(set(map(tuple, corrected_spans)))]
    return v17._seal(old_body | {"schema": "v18b-solution-receipt-v1", "accepted": not reasons,
        "reasons": reasons, "reason": reasons[0] if reasons else None, "process_errors": errors,
        "claim_ledger_sha256": ledger["sha256"], "verified_evidence_count": len(used),
        "verified_process_byte_spans": spans, "corrected_use_byte_spans": corrected,
        "evidence_grade": "SOURCE_GROUNDED" if spans else "RESULT_ONLY", "verification_version": "v18b",
        "format_receipt": form, "knowledge_status": "RESULT_CONDITIONED_PARTIALLY_CHECKED"})


def t1_policy_receipt_v18b(record: dict, *, tokenizer, gold: str, choices=(), verifier=None,
                          fact_catalog=(), rules=None) -> dict:
    policy = v18.t1_policy_receipt_v18(record, tokenizer=tokenizer, gold=gold, choices=choices,
                                      verifier=verifier, fact_catalog=fact_catalog, rules=rules, _v18b=True)
    ledger = audit_t1_v18b(record, verifier=verifier, tokenizer=tokenizer, fact_catalog=fact_catalog, rules=rules)
    errors = [claim for claim in ledger["claims"] if claim.get("assertion") == "ASSERTED" and claim["verdict"] == "CONTRADICTED"]
    if not errors or policy["reward"] <= .5 or not policy["pg"]:
        return policy
    output = record["t1"]
    text = output["text"]
    finals = [(match.start(), match.end()) for match in re.finditer(r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*[^\n]+$", text)]
    spans = [[len(text[:start].encode()), len(text[:end].encode())] for start, end in finals]
    spans = [span for span in spans if not any(span[0] < b and a < span[1] for error in errors for a, b in error["original_byte_spans"])] if len(spans) == 1 else []
    mask = v17._token_mask(tokenizer, output["generation"]["completion_token_ids"], text, spans) if spans else [False] * len(output["generation"]["completion_token_ids"])
    receipt = v17._seal({"schema": "v18b-policy-mask-v1", "target_sha256": text_sha256(text),
        "output_binding_sha256": v17._output_binding(output), "mask_reason": "VERIFIED_PROCESS_ERROR_FINAL_ONLY",
        "byte_spans": spans, "token_mask": mask, "claim_ledger_sha256": ledger["sha256"],
        "contradiction_byte_spans": [span for claim in errors for span in claim["original_byte_spans"]]})
    return v17._seal({key: value for key, value in policy.items() if key != "sha256"} | {
        "schema": "v18b-t1-policy-receipt-v1", "loss_token_mask": mask, "pg": any(mask),
        "pg_mask_reason": receipt["mask_reason"], "pg_byte_spans": spans, "policy_mask_receipt": receipt,
        "positive_pg_hold_reason": None if any(mask) else "FINAL_ANSWER_SPAN_UNBOUND", "t1_claim_ledger": ledger})


def boolean_note_token_span(output: dict, tokenizer, span: list[int]) -> list[int]:
    """Allow a JSON boolean with its own separator space when they form one sampled token."""
    start, end = span
    raw = output["text"].encode()
    if raw[start:end] not in (b"true", b"false") or raw[max(start - 1, 0):start] != b" ":
        return span
    actions = output["generation"]["completion_token_ids"]
    try:
        v17._token_mask(tokenizer, actions, output["text"], [span])
        return span
    except ValueError as error:
        if str(error) != "v16 train: no wholly verified token":
            raise
    expanded = [start - 1, end]
    mask = v17._token_mask(tokenizer, actions, output["text"], [expanded])
    ids = [token for token, keep in zip(actions, mask, strict=True) if keep]
    if len(ids) == 1 and tokenizer.decode(ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False).encode() == raw[start - 1:end]:
        return expanded
    raise ValueError("N_MASK:UNBOUND_BOOLEAN_TOKEN")


def n_row_v18b(record: dict, *, tokenizer, receipt: dict) -> dict | None:
    row = v18.n_row_v18(record, tokenizer=tokenizer, receipt=receipt)
    if row is not None:
        row["n_source_evidence"] = v17._seal({"schema": "v18b-n-source-evidence-v1", "row_id": row["row_id"],
            "problem_id": row["problem_id"], "target_sha256": text_sha256(row["target"]),
            "generation_sha256": digest(row["generation"]), "receipt_sha256": receipt["sha256"],
            "byte_spans": row["verified_byte_spans"], "token_ids": row["generation"]["completion_token_ids"],
            "token_mask": row["loss_mask"], "supervised_tokens": sum(row["loss_mask"]),
            "note_condition": receipt.get("note_condition"), "provided_location_count": receipt.get("provided_location_count", 0)})
    return row


build_obs_request_v18b = partial(v18.build_obs_request_v18, _v18b=True)
observation_eligibility_v18b = partial(v18.observation_eligibility_v18, _v18b=True)
audit_observation_v18b = partial(v18.audit_observation_v18, _v18b=True)
audit_t1_v18b = partial(v18.audit_t1_v18, _v18b=True)
build_note_request_v18b = partial(v18.build_note_request_v18, _v18b=True)
verify_note_v18b = partial(v18.verify_note_v18, _v18b=True)
verify_solution_v18b = partial(v18.verify_solution_v18, _v18b=True)
build_obs_request_v17 = build_obs_request_v18b
observation_eligibility_v17 = observation_eligibility_v18b
audit_observation_v17 = audit_observation_v18b
audit_t1_v17 = audit_t1_v18b
build_note_request_v17 = build_note_request_v18b
verify_note_v17 = verify_note_v18b
verify_solution_v17 = verify_solution_v18b
a1_row_v17 = v18.a1_row_v18
c_row_v17 = v18.c_row_v18
n_row_v17 = n_row_v18b
problem_contract_v17 = problem_contract_v18b
t1_policy_receipt_v18 = t1_policy_receipt_v18b
loop_token_mask_v17 = v18.loop_token_mask_v18
reward_v17 = v17.reward_v17
prep_rows_with_receipts_v17 = partial(v18.prep_rows_with_receipts_v18, _v18b=True)
def prep_rows_v17(records, **kwargs):
    return prep_rows_with_receipts_v17(records, **kwargs)[0]

note_condition_v18 = v18.note_condition_v18
