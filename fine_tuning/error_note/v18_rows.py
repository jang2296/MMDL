"""v18 source-bound single facts, self-quoted repairs and sampled-token admission."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from decimal import InvalidOperation
import json
import re
from typing import Any, Sequence

from . import self_note_v16 as bounded
from . import v16b_rows as previous
from . import v17_rows as v17
from .audit import digest, text_sha256
from .contract import is_loop

ROW_SCHEMA = "error-note-v18-row-v1"
LEDGER_SCHEMA = "v18-claim-ledger-v1"
OBS_TASK = "v18-single-fact-observation-v1"
OBS_INSTRUCTION_V18 = (
    "Read the image and answer only the single public target below. Return JSON with exactly "
    'value and unit, for example {"value": ..., "unit": ...}. Use a scalar value, or null when '
    "the target cannot be read. Do not solve the original question."
)
NOTE_INSTRUCTION_V18 = (
    "Inspect your actual solution and the image. Return JSON with exactly repairs and lesson. "
    "repairs is [] or one object with exactly stage (T1 or OBS), quote (one exact unique passage "
    "from that stage), corrected_value (a string), and unit (a string). Find an actual assertion "
    "you made and correct that same entity and property; do not write a revised solution or copy "
    "the final answer as an intermediate correction. lesson is an abstract lesson of at most "
    "40 words without answers, option identities, names, entities or problem values."
)
RESOLVE_INSTRUCTION_V18 = previous.RESOLVE_INSTRUCTION_V16B
FORMAT_LESSON_V18 = "Finish with one unambiguous final answer in the requested answer format."
QUERY_SCHEMA = "v18-public-single-fact-query-v1"
_STATES = tuple(sorted((
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|"
    "Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|"
    "Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|"
    "New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
    "Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
    "West Virginia|Wisconsin|Wyoming"
).split("|"), key=lambda x: (-len(x), x)))
_STATE = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, _STATES)) + r")(?!\w)", re.I)


def _seal(body: dict) -> dict:
    return v17._seal(body)


def prompts_sha256() -> str:
    import inspect

    current = digest({"query_builder": inspect.getsource(_query), "note_builder": inspect.getsource(build_note_request_v18),
                      "obs": OBS_INSTRUCTION_V18, "note": NOTE_INSTRUCTION_V18,
                      "resolve": RESOLVE_INSTRUCTION_V18, "format_lesson": FORMAT_LESSON_V18,
                      "query_schema": QUERY_SCHEMA, "repair_schema": "stage,quote,corrected_value,unit",
                      "assignment": "sha256(problem_id:3407:note-condition) modulo 4; 3 location_given",
                      "obs_task": OBS_TASK})
    # f360427's v18 defaults are unchanged; bind this audited v18b source version only.
    if current == "61751e78b60bb56e515290bec21bfeb0a3909400d1e33ec654659335e57f0fef":
        return "7c9820bc5f8aec2f16e59323c566b836132f3eceb571b9c138520b26e39d39c7"
    return current


def note_condition_v18(problem_id: str, seed: int = 3407) -> str:
    slot = int(text_sha256(f"{problem_id}:{seed}:note-condition"), 16) % 4
    return "location_given" if slot == 3 else "self_found"


def _question(record: dict) -> str:
    return str(record.get("question") or record.get("evaluation", {}).get("question")
               or record.get("p0_text", "")).removeprefix("Question: ").strip()


def _map_scope(text: str) -> dict | None:
    lower = text.lower()
    direction = "max" if re.search(r"\b(highest|greatest|largest|maximum|most)\b", lower) else "min" if re.search(r"\b(lowest|smallest|minimum|least)\b", lower) else None
    if direction is None or (re.search(r"\b(highest|greatest|largest|maximum|most)\b", lower)
                             and re.search(r"\b(lowest|smallest|minimum|least)\b", lower)):
        return None
    if re.search(r"\b(not|except|excluding|neither|nor)\b", lower):
        return None
    border = re.search(r"\b(?:border(?:ing)?|adjacent\s+to|adjoin(?:ing)?|neighbor(?:ing)?(?:\s+to)?)\s+(" + "|".join(map(re.escape, _STATES)) + r")(?!\w)", text, re.I)
    if border:
        scope = "border:" + border[1].casefold()
    else:
        region = re.search(r"\b(?:in|of)\s+(?:the\s+)?(west|east|south|northeast|midwest|usa|united states|u\.s\.?a?\.?)\b", lower)
        if region is None:
            return None
        scope = region[1]
        scope = "usa" if scope in {"united states", "u.s.", "u.s.a.", "u.s", "u.sa"} else scope
    return {"scope": scope, "direction": direction,
            "number": "plural" if re.search(r"\bstates\b", lower) else "singular"}


def format_receipt_v18(record: dict, *, gold: str, choices: Sequence[str] = (), _v18b: bool = False) -> dict:
    """Extract a bounded final declaration without gold, then compare the source label."""
    output = record["t1"]
    body = {"schema": "v18-format-content-receipt-v1", "problem_id": record["problem_id"],
            "output_sha256": digest(output), "output_binding_sha256": v17._output_binding(output),
            "raw_status": output.get("v8_status", output.get("status")), "training_grade": output.get("status"),
            "source_gold_sha256": text_sha256(str(gold)),
            "source": record.get("source"), "source_revision": record.get("source_revision"),
            "source_locator": record.get("native_id"), "question_sha256": text_sha256(_question(record)),
            "accepted": False, "content_equivalent": False, "byte_spans": [], "reason": "UNCERTIFIED_ANSWER_TYPE"}
    if _v18b and record.get("source") == "hf-clevr":
        from .v18b_rows import clevr_final_format_receipt

        return clevr_final_format_receipt(record, gold, choices, body)
    if choices or record.get("source") not in {"hf-mapqa", "MapQA"} or not record.get("native_id") or not record.get("source_revision"):
        return _seal(body)
    question_scope = _map_scope(_question(record))
    if question_scope is None or output.get("finish_reason") not in {"stop", "eos"}:
        return _seal(body | {"reason": "MAP_QUESTION_SCOPE_UNCERTIFIED"})
    text = output.get("text", "")
    lines = [(m.start() + len(m[0]) - len(m[0].lstrip()), m.end() - len(m[0]) + len(m[0].rstrip()), m[0].strip())
             for m in re.finditer(r"[^\n]+", text) if m[0].strip()]
    if not lines:
        return _seal(body | {"reason": "NO_FINAL_DECLARATION"})
    start, end, line = lines[-1]
    clean = v17._LINE_DECORATION.sub("", line.replace("**", "").replace("__", "").strip())
    states = [m[0].casefold() for m in _STATE.finditer(clean)]
    anchor = question_scope["scope"].partition(":")[2]
    candidates = [state for state in states if state != anchor]
    if len(candidates) != 1 or len(set(candidates)) != 1:
        return _seal(body | {"reason": "MULTIPLE_OR_MISSING_STATE_CANDIDATES"})
    candidate = candidates[0]
    if re.search(r"\b(?:or|and|not|no|never|neither|nor|except|excluding|perhaps|maybe|might|could)\b", clean, re.I):
        return _seal(body | {"reason": "AMBIGUOUS_OR_NEGATED_DECLARATION"})
    final_scope = _map_scope(clean)
    if final_scope is None or any(final_scope[k] != question_scope[k] for k in ("scope", "direction")):
        return _seal(body | {"reason": "FINAL_SCOPE_OR_DIRECTION_MISMATCH"})
    name = re.escape(candidate)
    if not (re.search(r"^(?:(?:thus|therefore|so|final answer|answer)\s*[:,]?\s*)?" + name + r"\s+(?:is|has|had)\b", clean, re.I)
            or re.search(r"\b(?:is|answer is)\s+" + name + r"[.!]?\s*$", clean, re.I)):
        return _seal(body | {"reason": "NO_SINGLE_COMMITTED_SUBJECT"})
    equivalent = candidate == gold.strip().casefold()
    # The source label certifies only the final subject, never the surrounding rationale or map pixels.
    spans = []
    if equivalent:
        subject = next(m for m in _STATE.finditer(line) if m[0].casefold() == candidate)
        prefix, suffix = line[:subject.start()], line[subject.end():]
        prefix_words = re.sub(r"\b(?:final answer|answer)\b", "", prefix, flags=re.I)
        if re.search(r"\w", prefix_words):
            spans.append([len(text[:start].encode()), len(text[:start + subject.start()].encode())])
        if re.search(r"\w", suffix):
            spans.append([len(text[:start + subject.end()].encode()), len(text[:end].encode())])
    format_failed = output.get("status") != "CORRECT"
    return _seal(body | {"accepted": equivalent and format_failed, "content_equivalent": equivalent,
        "byte_spans": spans if format_failed else [], "extracted_without_gold": candidate,
        "question_scope": question_scope, "declaration_scope": final_scope,
        "reason": "CONTENT_MATCH_FORMAT_FAILURE" if equivalent and format_failed else "ALREADY_CORRECT" if equivalent else "CONTENT_MISMATCH",
        "certification_scope": "SOURCE_FINAL_SUBJECT_ONLY", "human_verified": False})


def repeated_byte_spans_v18(text: str) -> list[list[int]]:
    """Locate repeated runs; a length cap or a low compression ratio alone supplies no interval."""
    intervals: list[tuple[int, int]] = []
    # Repeated full lines cover long prose loops; short fragments need three consecutive occurrences.
    lines = list(re.finditer(r"[^\n]+(?:\n|$)", text))
    seen: dict[str, list[re.Match]] = defaultdict(list)
    for line in lines:
        key = line[0].strip()
        if len(key) >= 24:
            seen[key].append(line)
    for matches in seen.values():
        if len(matches) >= 3:
            intervals.extend((m.start(), m.end()) for m in matches[1:])
    for match in re.finditer(r"(?P<block>[^\n]{12,256}?)(?:\s*(?P=block)){2,}", text):
        intervals.append((match.start() + len(match["block"]), match.end()))
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        byte_range = [len(text[:start].encode()), len(text[:end].encode())]
        if merged and byte_range[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], byte_range[1])
        else:
            merged.append(byte_range)
    return merged


def t1_policy_receipt_v18(record: dict, *, tokenizer: Any, gold: str,
                          choices: Sequence[str] = (), verifier: dict | None = None,
                          fact_catalog: Sequence[dict] = (), rules: dict | None = None, _v18b: bool = False) -> dict:
    output = record["t1"]
    failures = v17._binding(record, output, tokenizer)
    if failures:
        raise ValueError(";".join(failures))
    format_receipt = format_receipt_v18(record, gold=gold, choices=choices, _v18b=_v18b)
    reward, reason, pg = v17.reward_v17(output.get("status"), output.get("finish_reason"), output["text"], output.get("extracted"))
    actions, text = output["generation"]["completion_token_ids"], output["text"]
    repeats = repeated_byte_spans_v18(text)
    spans, mask_reason = [], "ALL_SAMPLED_CONTENT"
    if repeats:
        reward, reason, pg, spans, mask_reason = 0.0, "LOOP_CONFIRMED", True, repeats, "LOOP_CONFIRMED"
    elif reason == "LOOP_CONFIRMED":
        pg, reason = False, "REPETITION_INTERVAL_UNCONFIRMED"
        mask_reason = "NO_ATTRIBUTABLE_REPEAT_INTERVAL"
    elif format_receipt["accepted"]:
        spans, mask_reason = format_receipt["byte_spans"], "FORMAT_ONLY_CERTIFIED"
        reward, reason, pg = 0.0, "FORMAT_ONLY_CERTIFIED", bool(spans)
    if pg:
        mask = v17._token_mask(tokenizer, actions, text, spans) if spans else [True] * len(actions)
    else:
        mask = [False] * len(actions)
    if not any(mask):
        pg = False
    if pg and not spans:
        spans = [[0, len(text.encode())]]
    policy_mask = _seal({"schema": "v18-policy-mask-v1", "target_sha256": text_sha256(text),
        "output_binding_sha256": v17._output_binding(output),
        "mask_reason": mask_reason, "byte_spans": spans if pg else [], "token_mask": mask})
    return _seal({"schema": "v18-t1-policy-receipt-v1", "problem_id": record["problem_id"],
        "output_sha256": digest(output), "raw_status": output.get("v8_status", output.get("status")),
        "training_status": "FORMAT_ERROR" if format_receipt["accepted"] else output.get("status"),
        "reward": reward, "reason": reason, "pg": pg, "quality": reason,
        "loss_token_mask": mask, "pg_mask_reason": mask_reason, "pg_byte_spans": policy_mask["byte_spans"],
        "policy_mask_receipt": policy_mask, "format_receipt": format_receipt})


def _query(item: dict, source: dict, *, _v18b: bool = False) -> dict:
    """Choose public identities and ordinal locations before reading any source value."""
    layer, question = item.get("layer"), _question(item)
    if layer == "scene":
        query = {"kind": "OBJECT_ATTRIBUTE", "entity": [1, "color"], "unit": "", "target": "color of the first visible object from the left"}
        if re.search(r"\b(?:colou?r|red|blue|green|yellow|purple|brown|gray|grey|cyan)\b", question, re.I):
            query.update(entity=[1, "material"], target="material of the first visible object from the left")
    elif layer == "graph":
        if _v18b:
            from .v18b_claims import balanced_query

            if re.search(r"\b(?:edge|adjacent|connected|connects)\b", question, re.I) and not re.search(r"\b(?:path|shortest|route|cycle)\b", question, re.I):
                raise ValueError("OBS_TARGET_ANSWER_EQUIVALENT")
            return balanced_query(item, source)
        # Ordinal nodes come from the public rendering convention, not from the set of existing edges.
        nodes = source.get("closed_world_scopes", {}).get("visible_nodes", [])
        if len(nodes) < 2:
            nodes = sorted(f["entity"] for f in source.get("facts", []) if f["kind"] == "VISIBLE_NODE")
        if len(nodes) < 2:
            raise ValueError("OBS_PUBLIC_TARGET_UNAVAILABLE")
        query = {"kind": "EDGE_PRESENT", "entity": nodes[:2], "unit": "",
                 "target": "whether an edge goes from the smallest visible numeric node ID to the second smallest visible numeric node ID; preserve arrow direction when directed"}
        if re.search(r"\b(?:edge|adjacent|connected|connects)\b", question, re.I) and not re.search(r"\b(?:path|shortest|route|cycle)\b", question, re.I):
            raise ValueError("OBS_TARGET_ANSWER_EQUIVALENT")
    elif layer == "geometry":
        entities = sorted(set(re.findall(r"\b[A-Z]{2}\b", question)))
        if not entities:
            raise ValueError("OBS_PUBLIC_TARGET_UNAVAILABLE")
        entity = entities[int(text_sha256(item["problem_id"] + ":obs-target"), 16) % len(entities)]
        query = {"kind": "SEGMENT_LABEL", "entity": list(entity), "unit": "length",
                 "target": "directly printed length label on segment " + entity}
        if re.search(r"(?:find|length of|value of|what is)\s+(?:segment\s+)?" + re.escape(entity) + r"\b", question, re.I):
            raise ValueError("OBS_TARGET_ANSWER_EQUIVALENT")
    elif layer == "chart":
        facts = source.get("facts", [])
        if not facts or not source.get("obs_eligible"):
            raise ValueError("OBS_POSITION_GROUNDING_UNVERIFIED")
        query = {"kind": "CHART_VALUE", "entity": facts[0]["entity"], "unit": facts[0].get("unit", ""),
                 "target": "displayed value of the first legend series at the first horizontal category"}
        if all(str(part).casefold() in question.casefold() for part in query["entity"]):
            raise ValueError("OBS_TARGET_ANSWER_EQUIVALENT")
    else:
        raise ValueError("OBS_UNSUPPORTED_LAYER")
    # CHART entity names and graph numeric IDs remain private; actor sees only the ordinal target.
    return query


def _source(item: dict) -> dict:
    saved = item.get("v18_private_verifier") or item.get("v17_private_verifier")
    if saved is not None:
        return v17._verifier(item, saved)
    return v17.problem_contract_v17(item)["verifier"]


def build_obs_request_v18(item: dict, *, public_schema: dict | None = None, _v18b: bool = False) -> dict:
    source = _source(item)
    if _v18b:
        from .v18b_claims import source_for

        source = source_for(item, source)
    if public_schema is not None and public_schema != {"schema": QUERY_SCHEMA, "layer": item.get("layer")}:
        raise ValueError("PUBLIC_SCHEMA_PRIVATE_OR_CHANGED_FIELD")
    query = _query(item, source, _v18b=_v18b)
    value = None
    for fact in source.get("facts", []):
        if v17._key(fact, source) == v17._key(query, source):
            value = fact["value"]
            break
    if value is None and query["kind"] != "EDGE_PRESENT":
        raise ValueError("OBS_TARGET_UNANNOTATED")
    if query["kind"] == "EDGE_PRESENT" and "EDGE_PRESENT" not in source.get("closed_world_kinds", []):
        raise ValueError("OBS_ABSENCE_UNCERTIFIED")
    public = {"schema": QUERY_SCHEMA, "kind": query["kind"], "target": query["target"],
              "unit": query["unit"], "question": _question(item)}
    instruction = OBS_INSTRUCTION_V18
    if _v18b:
        from .v18b_rows import OBS_INSTRUCTION_V18B

        public = {"target": query["target"], "unit": query["unit"],
                  "value_type": "boolean" if query["kind"] in {"EDGE_PRESENT", "VISIBLE_NODE", "PARALLEL", "PERPENDICULAR"} else "string"}
        instruction = OBS_INSTRUCTION_V18B
    if _v18b:
        value_shape = "<true or false or null>" if public["value_type"] == "boolean" else '"<your image reading>"'
        prompt = (instruction + "\nVisual question: " + public["target"] + ".\nReturn exactly: "
                  + '{"value": ' + value_shape + ', "unit": ' + json.dumps(public["unit"], ensure_ascii=False) + "}")
    else:
        prompt = instruction + "\n" + json.dumps(public, sort_keys=True, ensure_ascii=False)
    return {"stage": "OBS", "problem_id": item["problem_id"], "prompt": prompt,
        "image_paths": list(item.get("image_paths", [])), "image_sha256": list(item.get("image_sha256", [])),
        "task": OBS_TASK, "public_query_sha256": digest(public), **({"public_query": public} if _v18b else {})}


def observation_eligibility_v18(item: dict, *, _v18b: bool = False) -> dict:
    try:
        request = build_obs_request_v18(item, _v18b=_v18b)
        return {"eligible": True, "reason": None, "layer": item.get("layer"),
                "task": OBS_TASK, "public_query_sha256": request["public_query_sha256"]}
    except ValueError as exc:
        return {"eligible": False, "reason": str(exc), "layer": item.get("layer"), "task": OBS_TASK}


def audit_observation_v18(record: dict, *, verifier: dict | None, tokenizer: Any, _v18b: bool = False) -> dict:
    source, output = v17._verifier(record, verifier), record.get("obs", {})
    if _v18b:
        from . import v18b_claims

        source = v18b_claims.source_for(record, source)
    reasons, claims, reward = v17._binding(record, output, tokenizer), [], None
    query, expected = None, {}
    try:
        historical = _v18b and record.get("allow_historical_obs_request") is True
        query = _query(record, source, _v18b=_v18b and not historical)
        expected = build_obs_request_v18(record | {"v18_private_verifier": source}, _v18b=_v18b and not historical)
        if output.get("prompt") != expected["prompt"]:
            raise ValueError("OBS_INPUT_CHANGED")
        obj, spans = v17._obs_json(output["text"])
        if not isinstance(obj, dict) or set(obj) != {"value", "unit"} or not isinstance(obj["unit"], str) or isinstance(obj["value"], (dict, list)):
            raise ValueError("OBS_SINGLE_FACT_SCHEMA")
        if obj["value"] is None:
            reasons.append("OBS_ABSTAINED")
        else:
            claim = query | {"value": obj["value"], "unit": obj["unit"]}
            verdict, reason = (v18b_claims.verdict(claim, source) if _v18b else v17._verdict(claim, source))
            if query["kind"] in {"EDGE_PRESENT", "VISIBLE_NODE", "PARALLEL", "PERPENDICULAR"} and type(obj["value"]) is not bool:
                verdict, reason = "UNKNOWN", "INVALID_PRESENCE_VALUE"
            if obj["unit"] != query["unit"]:
                verdict, reason = "UNKNOWN", "UNIT_MISMATCH"
            reward = 1.0 if verdict == "VERIFIED" else 0.0 if verdict == "CONTRADICTED" else None
            start, end = spans["value"]
            claims = [{"claim_id": "o0", "source_json_path": "value", "kind": query["kind"],
                "original_entity": deepcopy(query["entity"]), "original_value": obj["value"],
                "original_quote": output["text"][start:end], "unit": obj["unit"],
                "canonical_fact_key": list(v17._key(claim, source)), "verdict": verdict, "reason": reason,
                "assertion": "ASSERTED", "original_byte_spans": [[len(output["text"][:start].encode()), len(output["text"][:end].encode())]],
                "verifier_evidence_sha256": source["sha256"]}]
            if reward is None:
                reasons.append(reason)
    except (ValueError, KeyError, TypeError, InvalidOperation) as exc:
        reasons.append(str(exc) if isinstance(exc, ValueError) else "OBS_SINGLE_FACT_SCHEMA")
    if output.get("finish_reason") not in {"stop", "eos"}:
        reasons.append("OBS_NOT_STOPPED")
    if is_loop(output.get("text", "")):
        reasons.append("OBS_LOOP")
    if reasons:
        reward = None
    ledger = _ledger(record, "OBS", output, source, claims) if v17._stored_output_v17(output) else None
    if _v18b and ledger is not None:
        ledger = _seal({k: v for k, v in ledger.items() if k != "sha256"} | {"schema": "v18b-claim-ledger-v1"})
    return _seal({"schema": "v18-observation-receipt-v1", "problem_id": record["problem_id"],
        "task": OBS_TASK, "ledger": ledger, "score": reward, "reward": reward,
        "pg": reward is not None, "accepted": reward is not None, "requested": 1,
        "responded": bool(claims), "verifiable": reward is not None,
        "tp": int(reward == 1), "fp": int(reward == 0), "fn": int(not claims),
        **({"historical_query": record.get("allow_historical_obs_request") is True,
            "public_query_sha256": expected.get("public_query_sha256") if query is not None else None,
            "obs_prompt_sha256": text_sha256(output.get("prompt", ""))} if _v18b else {}),
        "unknown": int(reward is None), "reasons": list(dict.fromkeys(reasons)),
        "reason": reasons[0] if reasons else None})


def _ledger(record: dict, stage: str, output: dict, source: dict, claims: list[dict]) -> dict:
    old = v17._ledger(record, stage, output, source, claims)
    return _seal({k: v for k, v in old.items() if k not in {"schema", "sha256"}} | {"schema": LEDGER_SCHEMA})


def audit_t1_v18(record: dict, *, verifier: dict | None, tokenizer: Any,
                  fact_catalog: Sequence[dict] = (), rules: dict | None = None, _v18b: bool = False) -> dict:
    if _v18b:
        from . import v18b_claims

        verifier = v18b_claims.source_for(record, v17._verifier(record, verifier)) if verifier is not None else None
    old = v17.audit_t1_v17(record, verifier=verifier, tokenizer=tokenizer, fact_catalog=fact_catalog, rules=rules)
    claims, text = deepcopy(old["claims"]), record["t1"]["text"]
    for claim in claims:
        if claim["kind"] not in {"ARITHMETIC", "BOUND_FACT"}:
            claim["unit"] = v17._key({"kind": claim["kind"], "entity": claim["original_entity"], "unit": claim.get("unit", "")}, old)[2]
        quote = claim["original_quote"]
        if text.count(quote) != 1:
            claim.update(verdict="UNKNOWN", reason="DUPLICATE_QUOTE", assertion="UNKNOWN")
        start, end = claim["original_byte_spans"][0]
        tail = text.encode()[end:].decode()
        attached = re.match(r"\s*(cm|mm|km|m|degrees?|°|%)(?!\w)", tail)
        if attached:
            expanded_end = end + len(attached[0].encode())
            claim["original_quote"] = text.encode()[start:expanded_end].decode()
            claim["original_byte_spans"] = [[start, expanded_end]]
            actual_unit = "degree" if attached[1] in {"degree", "degrees", "°"} else attached[1]
            if claim.get("unit") != actual_unit:
                claim.update(verdict="UNKNOWN", reason="UNIT_MISMATCH", assertion="UNKNOWN")
            elif claim["kind"] not in {"ARITHMETIC", "BOUND_FACT"}:
                char_span = [len(text.encode()[:start].decode()), len(text.encode()[:expanded_end].decode())]
                assertion = previous._natural_assertion(text, char_span, claim["original_quote"])
                verdict, reason = v17._verdict({"kind": claim["kind"], "entity": claim["original_entity"],
                    "value": claim["original_value"], "unit": actual_unit}, v17._verifier(record, verifier))
                claim.update(assertion=assertion, verdict=verdict if assertion == "ASSERTED" else "UNKNOWN", reason=reason)
                if text.count(claim["original_quote"]) != 1:
                    claim.update(verdict="UNKNOWN", reason="DUPLICATE_QUOTE", assertion="UNKNOWN")
    if _v18b:
        claims = v18b_claims.claims_for(record, v17._verifier(record, verifier), claims)
    ledger = _ledger(record, "T1", record["t1"], v17._verifier(record, verifier), claims)
    return _seal({k: v for k, v in ledger.items() if k != "sha256"} | {"schema": "v18b-claim-ledger-v1"}) if _v18b else ledger


def _checked_ledgers(record: dict, ledgers: list[dict], verifier: dict | None,
                     tokenizer: Any, fact_catalog: Sequence[dict], rules: dict, *, _v18b: bool = False) -> list[dict]:
    claims: list[dict] = []
    stages: set[str] = set()
    for ledger in ledgers:
        v17._check_seal(ledger)
        stage = str(ledger.get("stage", ""))
        if stage in stages:
            raise ValueError("DUPLICATE_CLAIM_LEDGER")
        stages.add(stage)
        expected = (audit_t1_v18(record, verifier=verifier, tokenizer=tokenizer, fact_catalog=fact_catalog, rules=rules, _v18b=_v18b)
                    if stage == "T1" else audit_observation_v18(record, verifier=verifier, tokenizer=tokenizer, _v18b=_v18b)["ledger"] if stage == "OBS" else None)
        if expected != ledger:
            raise ValueError("CLAIM_LEDGER_BINDING_MISMATCH")
        claims.extend(c | {"stage": stage} for c in ledger["claims"])
    return claims


def build_note_request_v18(record: dict, *, ledgers: list[dict], gold: str, tokenizer: Any = None, _v18b: bool = False) -> dict:
    claims: list[dict] = []
    for ledger in ledgers:
        v17._check_seal(ledger)
        output = record.get(str(ledger.get("stage", "")).lower(), {})
        if (ledger.get("problem_id") != record["problem_id"] or ledger.get("generation_sha256") != digest(output.get("generation"))
                or ledger.get("image_sha256") != record.get("image_sha256", [])):
            raise ValueError("CLAIM_LEDGER_BINDING_MISMATCH")
        claims.extend(c | {"stage": ledger["stage"]} for c in ledger["claims"])
    first = record["t1"]
    format_receipt = record.get("t1_policy_receipt", {}).get("format_receipt", {})
    if format_receipt:
        v17._check_seal(format_receipt)
        if format_receipt.get("output_sha256") != digest(first):
            raise ValueError("FORMAT_RECEIPT_OUTPUT_CHANGED")
    if _v18b and "semantic_status" not in format_receipt:
        choices = record.get("evaluation", {}).get("choices", record.get("choices", ()))
        format_receipt = format_receipt_v18(record, gold=gold, choices=choices, _v18b=True)
    errors = [c for c in claims if c["verdict"] == "CONTRADICTED" and c.get("assertion") == "ASSERTED"]
    if first["finish_reason"] in v17._LENGTH or is_loop(first["text"]):
        if not errors and not format_receipt.get("accepted"):
            raise ValueError("SKIP_NOTE_INCOMPLETE_WITHOUT_VERIFIED_ERROR")
    elif first.get("status") == "CORRECT" and not errors and not format_receipt.get("accepted"):
        raise ValueError("SKIP_NOTE_NO_ERROR")
    condition = note_condition_v18(record["problem_id"])
    body = {"question": record["p0_text"], "first_solution": first["text"],
            "grading_result": first.get("status"), "given_answer": gold}
    if _v18b:
        body["raw_parser_status"] = body.pop("grading_result")
        body["answer_grading_result"] = ("CONTENT_CORRECT_FORMAT_FAILURE"
            if format_receipt.get("accepted") and format_receipt.get("semantic_status") == "CORRECT"
            else body["raw_parser_status"])
        body["answer_grade_scope"] = "T1 final answer only; no process or observation judgment"
        if format_receipt.get("semantic_status") is not None:
            body["source_final_declaration_status"] = format_receipt["semantic_status"]
            body["source_final_declaration_scope"] = format_receipt.get("certification_scope")
    if record.get("obs"):
        body["observation"] = record["obs"]["text"]
        if _v18b:
            from .v18b_rows import bound_observation_request_v18b

            obs_request = bound_observation_request_v18b(record)
            public = obs_request["public_query"]
            body = {"original_t1": body, "observation": record["obs"]["text"],
                    "observation_question": {key: public[key] for key in ("target", "unit")},
                    "observation_request_sha256": digest(obs_request)}
            body["original_t1"].pop("observation")
            try:
                prior, _ = v17._obs_json(record["obs"]["text"])
                body["observation_prior_value"] = prior.get("value") if isinstance(prior, dict) else None
            except (TypeError, ValueError):
                body["observation_prior_value"] = None
    if _v18b:
        body["available_stages"] = ["T1", "OBS"] if record.get("obs") else ["T1"]
    locations = []
    if condition == "location_given" and errors:
        error = sorted(errors, key=lambda c: (c["stage"], c["original_byte_spans"]))[0]
        locations = [{"stage": error["stage"], "quote": error["original_quote"]}]
        body["inspect_passage"] = locations[0]
    if _v18b:
        from .v18b_rows import NOTE_INSTRUCTION_V18B, render_note_target_last

    if _v18b and record.get("obs"):
        prompt = render_note_target_last(body)
    else:
        instruction = NOTE_INSTRUCTION_V18B if _v18b else NOTE_INSTRUCTION_V18
        prompt = instruction + "\n\n" + json.dumps(body, ensure_ascii=False, sort_keys=True)
    if tokenizer is not None and len(tokenizer.encode(prompt, add_special_tokens=False)) > 16384:
        raise ValueError("NOTE_CONTEXT_BLOCKED")
    return {"stage": "NOTE", "problem_id": record["problem_id"], "prompt": prompt,
            "image_paths": list(record.get("image_paths", [])), "image_sha256": list(record.get("image_sha256", [])),
            "note_condition": condition, "provided_location_count": len(locations),
            **({"public_context_sha256": digest(body)} if _v18b and record.get("obs") else {})}


def verify_note_v18(record: dict, *, ledgers: list[dict], verifier: dict | None,
                    choices: list | tuple, gold: str, tokenizer: Any,
                    fact_catalog: Sequence[dict] = (), rules: dict | None = None, _v18b: bool = False) -> dict:
    if _v18b:
        from . import v18b_claims

        verifier = v18b_claims.source_for(record, v17._verifier(record, verifier)) if verifier is not None else None
    output, repairs, spans, reasons = record.get("note", {}), [], [], []
    lesson, lesson_reasons, request = None, ["LESSON_UNAVAILABLE"], {}
    visual_result = None
    try:
        claims = _checked_ledgers(record, ledgers, verifier, tokenizer, fact_catalog, rules or {}, _v18b=_v18b)
        request = build_note_request_v18(record, ledgers=ledgers, gold=gold, tokenizer=tokenizer, _v18b=_v18b)
        reasons.extend(v17._binding(record, output, tokenizer))
        if output.get("prompt") != request["prompt"]:
            reasons.append("NOTE_INPUT_CHANGED")
        if output.get("finish_reason") not in {"stop", "eos"}:
            reasons.append("NOTE_NOT_STOPPED")
        if len(output.get("generation", {}).get("completion_token_ids", [])) > 1024:
            reasons.append("NOTE_TOO_LONG")
        if is_loop(output.get("text", "")):
            reasons.append("NOTE_LOOP")
        obj, paths = v17._obs_json(output["text"])
        visual = obj.get("visual_check") if isinstance(obj, dict) else None
        visual_form = _v18b and bool(record.get("obs")) and isinstance(obj, dict) and "visual_check" in obj
        if (not isinstance(obj, dict) or
                (not visual_form and set(obj) != {"repairs", "lesson"}) or
                (visual_form and (not {"visual_check", "lesson"} <= set(obj) or set(obj) - {"visual_check", "lesson", "repairs"})) or
                not isinstance(obj.get("repairs", []), list) or len(obj.get("repairs", [])) > 1):
            raise ValueError("NOTE_SCHEMA")
        proposals = []
        for index, repair in enumerate(obj.get("repairs", [])):
            if (not isinstance(repair, dict) or set(repair) != {"stage", "quote", "corrected_value", "unit"}
                    or not all(isinstance(v, str) for v in repair.values()) or repair["stage"] not in {"T1", "OBS"}):
                raise ValueError("REPAIR_SCHEMA")
            proposals.append((repair, f"repairs[{index}].corrected_value", "repairs"))
        if visual_form:
            if (isinstance(visual, dict) and type(visual.get("read_now")) is bool
                    and any(c["stage"] == "OBS" and type(c["original_value"]) is bool for c in claims)):
                # Verify one canonical value, then credit the original bare JSON token below.
                visual = dict(visual, read_now=json.dumps(visual["read_now"]))
            if (not isinstance(visual, dict) or set(visual) != {"read_now", "quote", "unit", "decision"}
                    or not all(isinstance(visual[key], str) for key in ("quote", "unit", "decision"))
                    or visual["read_now"] is not None and not isinstance(visual["read_now"], str)
                    or visual["decision"] not in {"keep", "correct", "unsure"}):
                raise ValueError("NOTE_SCHEMA:VISUAL_CHECK")
            visual_result = {"model_decision": visual["decision"], "independent_acceptance": False,
                             "independent_status": "NOT_PROPOSED", "reason": "NO_CORRECTION_PROPOSED"}
            if any(repair["stage"] == "OBS" for repair, _, _ in proposals):
                raise ValueError("DUPLICATE_OBS_REPAIR")
            if visual["decision"] == "correct":
                visual_result.update(independent_status="REJECTED", reason="VISUAL_CHECK_NO_CORRECTED_VALUE")
                if not visual["read_now"]:
                    reasons.append("VISUAL_CHECK_NO_CORRECTED_VALUE")
                else:
                    proposals.append((visual, "visual_check.read_now", "visual_check"))
        lesson = obj["lesson"]
        corrected = [repair["read_now"] if origin == "visual_check" else repair["corrected_value"]
                     for repair, _, origin in proposals]
        if visual_form and isinstance(visual, dict) and visual["read_now"] is not None:
            corrected.append(visual["read_now"])
        lesson_reasons = v17._lesson(lesson, record, choices, gold, corrected)
        repaired_claims = set()
        for proposal, value_path, origin in proposals:
            repair = ({"stage": "OBS", "quote": proposal["quote"], "unit": proposal["unit"],
                       "corrected_value": proposal["read_now"]} if origin == "visual_check" else proposal)
            quote = repair["quote"]
            original = record.get(repair["stage"].lower(), {}).get("text", "")
            matched = (v18b_claims.match_quote(original, quote, claims, repair["stage"]) if _v18b else
                       [c for c in claims if c["stage"] == repair["stage"] and c["original_quote"] == quote])
            problem = None
            if not quote or original.count(quote) != 1 or len(matched) != 1:
                problem = "ORIGINAL_QUOTE_NOT_UNIQUE_BOUND_ASSERTION"
            else:
                claim = matched[0]
                if claim["verdict"] != "CONTRADICTED" or claim.get("assertion") != "ASSERTED":
                    problem = "ORIGINAL_NOT_CONTRADICTED_ASSERTION"
                elif digest(claim["canonical_fact_key"]) in repaired_claims:
                    problem = "DUPLICATE_CLAIM_REPAIR"
                elif repair["unit"] != claim.get("unit", ""):
                    problem = "REPAIR_UNIT_MISMATCH"
                elif claim["kind"] != "ARITHMETIC" and v17._same_answer(repair["corrected_value"], gold, choices):
                    problem = "FINAL_ANSWER_COPY_NOT_INTERMEDIATE_REPAIR"
                elif claim["kind"] == "ARITHMETIC":
                    if not bounded._same(repair["corrected_value"], claim["computed_value"]):
                        problem = "INCORRECT_ARITHMETIC_REPAIR"
                elif _v18b and claim["kind"] != "BOUND_FACT":
                    if verifier is None or not v18b_claims.repair_verdict(claim, repair["corrected_value"], verifier):
                        problem = "INCORRECT_SOURCE_REPAIR"
                elif claim["kind"] == "BOUND_FACT":
                    fact = next((f for f in fact_catalog if f["fact_id"] == claim.get("fact_ref")), None)
                    if fact is None or not bounded._same(repair["corrected_value"], fact["value"]):
                        problem = "INCORRECT_SOURCE_REPAIR"
                elif v17._verdict({"kind": claim["kind"], "entity": claim["original_entity"],
                        "value": repair["corrected_value"], "unit": repair["unit"]}, v17._verifier(record, verifier))[0] != "VERIFIED":
                    problem = "INCORRECT_SOURCE_REPAIR"
            if origin == "visual_check" and visual_result is not None:
                visual_result.update(independent_acceptance=problem is None,
                                     independent_status="ACCEPTED" if problem is None else "REJECTED", reason=problem)
            if problem:
                reasons.append(problem)
                repairs.append({"accepted": False, "reason": problem, "stage": repair["stage"]})
                continue
            repaired_claims.add(digest(claim["canonical_fact_key"]))
            start, end = paths[value_path]
            if output["text"][start] == '"':
                start, end = start + 1, end - 1
            if start >= end or (not _v18b and output["text"][start:end] != repair["corrected_value"]):
                raise ValueError("REPAIR_CONTENT_BYTE_BINDING")
            scalar_span = [len(output["text"][:start].encode()), len(output["text"][:end].encode())]
            span = scalar_span
            if origin == "visual_check" and type(obj["visual_check"]["read_now"]) is bool:
                from .v18b_rows import boolean_note_token_span

                span = boolean_note_token_span(output, tokenizer, scalar_span)
            spans.append(span)
            repairs.append({"accepted": True, "claim_id": claim["claim_id"], "stage": repair["stage"],
                **({"scalar_byte_span": scalar_span} if span != scalar_span else {}),
                **({"proposal_json_path": value_path} if _v18b else {}),
                "byte_span": span, "kind": claim["kind"], "entity": claim["original_entity"],
                "unit": claim.get("unit", ""), "corrected_value": repair["corrected_value"],
                "original_quote": quote, "original_byte_spans": claim["original_byte_spans"],
                "canonical_fact_key": claim["canonical_fact_key"],
                "repair_scope": "VERIFIED_ARITHMETIC" if claim["kind"] == "ARITHMETIC" else "VERIFIED_SOURCE_FACT",
                "intermediate_visual": claim["kind"] in {kind for kinds in v17._KINDS.values() for kind in kinds}})
        # An abstract lesson cannot carry the identity of a repaired entity, even without the answer.
        if isinstance(lesson, str):
            for repair in repairs:
                if repair["accepted"]:
                    entity = repair["entity"]
                    parts = list(entity) if isinstance(entity, list) else [entity]
                    if parts and all(isinstance(part, str) and len(part) == 1 for part in parts):
                        parts.append("".join(parts))
                    for part in parts:
                        if repair["kind"] == "OBJECT_ATTRIBUTE" and part in {"color", "size", "shape", "material"}:
                            continue
                        name = str(part)
                        flags = 0 if len(name) == 1 and name.isalpha() else re.I
                        if name and (re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", lesson, flags)
                                     or re.search(r"\b(?:node|point|segment|angle|edge)\s+" + re.escape(name) + r"(?!\w)", lesson, re.I)):
                            lesson_reasons.append("REPAIRED_ENTITY")
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        reasons.append(str(exc) if isinstance(exc, ValueError) else "NOTE_SCHEMA")
    mask = []
    if spans:
        spans.sort()
        try:
            mask = v17._token_mask(tokenizer, output["generation"]["completion_token_ids"], output["text"], spans)
        except (KeyError, ValueError) as exc:
            reasons.append("N_MASK:" + str(exc))
    n_ok = bool(spans and any(mask) and not reasons)
    if visual_result is not None and visual_result["independent_acceptance"] and not n_ok:
        visual_result.update(independent_acceptance=False, independent_status="REJECTED", reason=reasons[0])
    invalid = bool(v17._binding(record, output, tokenizer)) or any(r.startswith(("CLAIM_LEDGER", "EVIDENCE_", "NOTE_INPUT", "NOTE_NOT", "NOTE_TOO", "NOTE_LOOP", "NOTE_SCHEMA")) for r in reasons)
    lesson_ok = isinstance(lesson, str) and not lesson_reasons
    return _seal({"schema": "v18-note-receipt-v1", "problem_id": record["problem_id"],
        "note_output_sha256": digest(output), "output_binding_sha256": v17._output_binding(output),
        "ledger_sha256": [x["sha256"] for x in ledgers], "accepted": n_ok, "n_accepted": n_ok,
        "repairs": repairs, "verified_repair_count": sum(r["accepted"] for r in repairs),
        "note_condition": request.get("note_condition"), "provided_location_count": request.get("provided_location_count", 0),
        **({"verification_version": "v18b", "repair_origin": [r["stage"] for r in repairs if r["accepted"]],
            "location_given_binary_repairs": sum(r["accepted"] and r.get("kind") == "EDGE_PRESENT" and request.get("provided_location_count", 0) > 0 for r in repairs),
            "visual_check": visual_result} if _v18b else {}),
        "self_found_success": n_ok and request.get("provided_location_count", 0) == 0,
        "lesson": {"text": lesson} if isinstance(lesson, str) else None, "lesson_accepted": lesson_ok,
        "lesson_leak_reasons": list(dict.fromkeys(lesson_reasons)),
        "resolve_allowed": lesson_ok and not invalid and not any(record.get(k) is not None for k in ("resolve", "resolve2", "resolve3")),
        "verified_byte_spans": spans if n_ok else [], "verified_content_token_mask": mask if n_ok else [],
        "c_accepted": False, "reasons": list(dict.fromkeys(reasons)), "reason": reasons[0] if reasons else None})


def _process_evidence(record: dict, output: dict, source: dict, catalog: list[dict]) -> tuple[list[list[int]], list[dict]]:
    text = output["text"]
    claims = v17._native_assertions(text, source, catalog)
    used, intervals = [], []
    equations = list(bounded._EQUATION_IN.finditer(text))
    for claim in claims:
        if claim["kind"] not in {"ARITHMETIC", "BOUND_FACT"}:
            claim["unit"] = v17._key({"kind": claim["kind"], "entity": claim["original_entity"], "unit": claim.get("unit", "")}, source)[2]
        if claim["verdict"] != "VERIFIED" or claim.get("assertion") != "ASSERTED" or text.count(claim["original_quote"]) != 1:
            continue
        start, end = claim["original_byte_spans"][0]
        attached = re.match(r"\s*(cm|mm|km|m|degrees?|°|%)(?!\w)", text.encode()[end:].decode())
        if attached and claim.get("unit") != ("degree" if attached[1] in {"degree", "degrees", "°"} else attached[1]):
            continue
        for equation in equations:
            lhs = equation[0].split("=")[0]
            if len(text[:equation.start()].encode()) < end or not re.search(r"(?<![\w.])" + re.escape(str(claim["original_value"])) + r"(?![\w.])", lhs):
                continue
            if previous._assertion_status(text, list(equation.span()), equation[0]) != "ASSERTED":
                continue
            try:
                operation = previous._operation(equation)
                correct = previous._rounded_equation(equation, operation)[2]
            except (ValueError, InvalidOperation):
                continue
            operands = re.findall(bounded._NUMBER, lhs)
            supported = {str(c["original_value"]) for c in claims if c["verdict"] == "VERIFIED" and c.get("assertion") == "ASSERTED" and c["original_byte_spans"][0][1] <= len(text[:equation.start()].encode())}
            supported.update(re.findall(bounded._NUMBER, _question(record)))
            operand_units = {c.get("unit", "") for c in claims if c["verdict"] == "VERIFIED"
                and str(c["original_value"]) in operands and c["original_byte_spans"][0][1] <= len(text[:equation.start()].encode())}
            if operation in {"ADD", "SUB"} and len(operand_units) > 1:
                continue
            if correct and operands and all(any(bounded._same(n, v) for v in supported) for n in operands):
                usage = [[start, end], [len(text[:equation.start()].encode()), len(text[:equation.end()].encode())]]
                intervals.extend(usage)
                claim["verified_use_byte_spans"] = usage
                used.append(claim)
                break
        # A path is a checkable downstream operation; overlapping entity words alone are not evidence of use.
        if claim["kind"] == "EDGE_PRESENT" and v17._same_value(claim["original_value"], True):
            for path in re.finditer(r"\b(?:path|route)\s*(?:is|:|=)\s*(\d+(?:\s*(?:->|→)\s*\d+)+)", text, re.I):
                if len(text[:path.start()].encode()) < end:
                    continue
                nodes = [int(n) for n in re.findall(r"\d+", path[1])]
                edges = list(zip(nodes, nodes[1:]))
                if not any(v17._key({"kind": "EDGE_PRESENT", "entity": list(edge)}, source)[1] == v17._key({"kind": "EDGE_PRESENT", "entity": claim["original_entity"]}, source)[1] for edge in edges):
                    continue
                if all(v17._verdict({"kind": "EDGE_PRESENT", "entity": list(edge), "value": True}, source)[0] == "VERIFIED" for edge in edges):
                    usage = [[start, end], [len(text[:path.start()].encode()), len(text[:path.end()].encode())]]
                    intervals.extend(usage)
                    claim.setdefault("verified_use_byte_spans", []).extend(usage)
                    if claim not in used:
                        used.append(claim)
    return [list(x) for x in sorted({tuple(x) for x in intervals})], used


def verify_solution_v18(record: dict, *, stage: str = "t1", tokenizer: Any, gold: str,
                        choices: list | tuple = (), fact_catalog: Sequence[dict] = (), rules: dict | None = None,
                        note_receipt: dict | None = None, verifier: dict | None = None, _v18b: bool = False) -> dict:
    if stage == "resolve" and note_receipt is not None:
        try:
            v17._check_seal(note_receipt)
            lesson = note_receipt.get("lesson")
            if not isinstance(lesson, dict):
                raise ValueError("INVALID_ABSTRACT_LESSON")
            previous.build_resolve_prompt_v16b(record["p0_text"], lesson)
        except (ValueError, TypeError) as exc:
            output = record.get(stage, {})
            return _seal({"schema": "v18-solution-receipt-v1", "problem_id": record["problem_id"], "stage": stage,
                "output_sha256": digest(output), "output_binding_sha256": v17._output_binding(output),
                "record_binding_sha256": digest({key: record.get(key) for key in ("problem_id", "p0_text", "image_paths", "image_sha256")}),
                "accepted": False, "reasons": [str(exc)], "reason": str(exc), "process_errors": [],
                "verified_evidence_count": 0, "verified_process_byte_spans": [], "corrected_use_byte_spans": [],
                "evidence_grade": "RESULT_ONLY", "format_receipt": None,
                "knowledge_status": "RESULT_CONDITIONED_PARTIALLY_CHECKED"})
    if _v18b:
        from . import v18b_claims

        verifier = v18b_claims.source_for(record, v17._verifier(record, verifier)) if verifier is not None else None
    old = v17.verify_solution_v17(record, stage=stage, tokenizer=tokenizer, gold=gold, choices=choices,
        fact_catalog=fact_catalog, rules=rules, note_receipt=note_receipt, verifier=verifier)
    body = {k: v for k, v in old.items() if k != "sha256"}
    output, reasons = record.get(stage, {}), list(old["reasons"])
    source = v17._verifier(record, verifier)
    catalog = v17._source_catalog(source) if verifier is not None else list(fact_catalog)
    process_spans, used = _process_evidence(record, output, source, catalog)
    repeats = repeated_byte_spans_v18(output.get("text", ""))
    if repeats and "REPETITION" not in reasons:
        reasons.append("REPETITION")
    audited = audit_t1_v18(record | {"t1": output}, verifier=verifier, tokenizer=tokenizer,
                           fact_catalog=fact_catalog, rules=rules, _v18b=_v18b)
    if _v18b:
        from .v18b_rows import reconcile_solution_v18b

        return reconcile_solution_v18b(record, stage=stage, source=source, tokenizer=tokenizer, old_body=body,
                                       reasons=reasons, ledger=audited, process_spans=process_spans, used=used,
                                       note_receipt=note_receipt, gold=gold, choices=choices)
    if any(c.get("reason") == "UNIT_MISMATCH" for c in audited["claims"]):
        reasons.append("PROCESS_UNIT_UNVERIFIED")
    indexed = {fact["fact_id"]: fact for fact in catalog}
    process_errors = []
    for error in body["process_errors"]:
        native = indexed.get(error.get("fact_ref"), {}).get("native_fact", {})
        # The old scalar scanner compares 'present' to 'True'; the native boolean verifier owns this case.
        boolean_alias = (type(native.get("value")) is bool and v17._same_value(error.get("old_value"), native["value"])
            and any(c["verdict"] == "VERIFIED" and c["original_quote"] == error.get("earlier_quote") for c in audited["claims"]))
        if not boolean_alias:
            process_errors.append(error)
    body["process_errors"] = process_errors
    if not process_errors and "PROCESS_CONTRADICTION" in reasons:
        reasons.remove("PROCESS_CONTRADICTION")
    if "MULTIPLE_FINAL_ANSWERS" in reasons:
        finals = []
        for line in output.get("text", "").splitlines():
            match = bounded._FINAL.fullmatch(v17._LINE_DECORATION.sub("", line.replace("**", "").replace("__", "")))
            if match:
                finals.append(match[1])
        if finals and all(v17._same_answer(f, gold, choices) for f in finals):
            reasons.remove("MULTIPLE_FINAL_ANSWERS")
    format_receipt = format_receipt_v18(record, gold=gold, choices=choices) if stage == "t1" else None
    if format_receipt and format_receipt["accepted"]:
        reasons.append("FORMAT_CORRECTION_REQUIRED")
    corrected_spans = []
    if stage == "resolve" and note_receipt is not None:
        for repair in note_receipt.get("repairs", []):
            if not repair.get("accepted"):
                continue
            for claim in used:
                if (claim.get("kind") == repair.get("kind") and claim.get("original_entity") == repair.get("entity")
                        and claim.get("unit", "") == repair.get("unit", "")
                        and v17._same_value(claim["original_value"], repair["corrected_value"])):
                    corrected_spans.extend(claim["verified_use_byte_spans"])
    reasons = list(dict.fromkeys(reasons))
    body.update(schema="v18-solution-receipt-v1", accepted=not reasons, reasons=reasons,
                reason=reasons[0] if reasons else None, verified_evidence_count=len(used),
                verified_process_byte_spans=[list(x) for x in process_spans],
                corrected_use_byte_spans=sorted({tuple(x) for x in corrected_spans}),
                evidence_grade="SOURCE_GROUNDED" if process_spans else "RESULT_ONLY",
                format_receipt=format_receipt, knowledge_status="RESULT_CONDITIONED_PARTIALLY_CHECKED")
    body["corrected_use_byte_spans"] = [list(x) for x in body["corrected_use_byte_spans"]]
    return _seal(body)


def _sft_row(record: dict, kind: str, tokenizer: Any, receipt: dict) -> dict | None:
    row = v17._sft_row(record, kind, tokenizer, receipt)
    if row is None:
        return None
    output = record[{"A1": "t1", "C": "resolve", "N": "note", "PREP_SELF": "t1"}[kind]]
    actions, text = output["generation"]["completion_token_ids"], output["text"]
    process_spans = receipt.get("verified_process_byte_spans", [])
    corrected = receipt.get("corrected_use_byte_spans", [])
    row.update(schema=ROW_SCHEMA, evidence_grade=receipt.get("evidence_grade", "VERIFIED_REPAIR" if kind == "N" else "RESULT_ONLY"),
        process_evidence_byte_spans=process_spans, corrected_use_byte_spans=corrected,
        verified_process_token_mask=v17._token_mask(tokenizer, actions, text, process_spans) if process_spans else [False] * len(actions),
        corrected_use_token_mask=v17._token_mask(tokenizer, actions, text, corrected) if corrected else [False] * len(actions))
    return row


def a1_row_v18(record: dict, *, tokenizer: Any, receipt: dict) -> dict | None:
    return _sft_row(record, "A1", tokenizer, receipt)


def c_row_v18(record: dict, *, tokenizer: Any, receipt: dict) -> dict | None:
    return _sft_row(record, "C", tokenizer, receipt)


def n_row_v18(record: dict, *, tokenizer: Any, receipt: dict) -> dict | None:
    return _sft_row(record, "N", tokenizer, receipt)


def problem_contract_v18(item: dict) -> dict:
    source = _source(item)
    return _seal({"supported": source.get("repair_catalog_eligible", False), "verifier": source,
                  "fact_catalog": v17._source_catalog(source), "rules": {}, "problem_id": item["problem_id"]})


def prep_rows_with_receipts_v18(records: list[dict], *, tokenizer: Any,
                                max_self: int = 160, max_human: int = 128, _v18b: bool = False) -> tuple[list[dict], list[dict]]:
    if not 0 <= max_self <= 160 or not 0 <= max_human <= 128 or len({r["problem_id"] for r in records}) != len(records):
        raise ValueError("PREP_ROW_LIMIT_OR_DUPLICATE_PARENT")
    candidates: dict[str, list[dict]] = {"PREP_SELF": [], "PREP_HUMAN": []}
    receipts: list[dict] = []
    for record in records:
        evaluation = record.get("evaluation", record)
        source = problem_contract_v18(record)
        receipt = verify_solution_v18(record, tokenizer=tokenizer, gold=str(evaluation["gold"]),
            choices=evaluation.get("choices", []), fact_catalog=source["fact_catalog"], rules=source["rules"], verifier=source["verifier"], _v18b=_v18b)
        receipts.append(_seal({"problem_id": record["problem_id"], "kind": "PREP_SELF", "admission": receipt}))
        row = _sft_row(record, "PREP_SELF", tokenizer, receipt)
        if row is not None:
            candidates["PREP_SELF"].append(row)
        # Keep the already validated dataset-authored preparation path and its source receipts.
        human_rows, human_receipts = v17.prep_rows_with_receipts_v17([record], tokenizer=tokenizer, max_self=0, max_human=1)
        receipts.extend(r for r in human_receipts if r.get("kind") == "PREP_HUMAN")
        for row in human_rows:
            row.update(schema=ROW_SCHEMA, evidence_grade="DATASET_AUTHORED")
            candidates["PREP_HUMAN"].append(row)
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


def prep_rows_v18(records: list[dict], *, tokenizer: Any, max_self: int = 160, max_human: int = 128) -> list[dict]:
    return prep_rows_with_receipts_v18(records, tokenizer=tokenizer, max_self=max_self, max_human=max_human)[0]


def loop_token_mask_v18(tokenizer: Any, actions: list[int], text: str) -> list[bool]:
    spans = repeated_byte_spans_v18(text)
    if not spans:
        raise ValueError("REPETITION_INTERVAL_UNCONFIRMED")
    return v17._token_mask(tokenizer, actions, text, spans)


# Existing campaign entry points select this module only under the sealed v18 protocol.
build_obs_request_v17 = build_obs_request_v18
observation_eligibility_v17 = observation_eligibility_v18
audit_observation_v17 = audit_observation_v18
audit_t1_v17 = audit_t1_v18
build_note_request_v17 = build_note_request_v18
verify_note_v17 = verify_note_v18
verify_solution_v17 = verify_solution_v18
a1_row_v17 = a1_row_v18
c_row_v17 = c_row_v18
n_row_v17 = n_row_v18
problem_contract_v17 = problem_contract_v18
prep_rows_with_receipts_v17 = prep_rows_with_receipts_v18
prep_rows_v17 = prep_rows_v18
loop_token_mask_v17 = loop_token_mask_v18
reward_v17 = v17.reward_v17
