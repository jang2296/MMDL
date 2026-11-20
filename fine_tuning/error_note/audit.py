"""Private, source-bound note review receipts; no model, network or universal prose verifier.

Only an explicit full source review can admit a whole note or clear a lesson. Program/format checks
remain bounded evidence. Review evidence is metadata, never a replacement training target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contract import _has_words, hint_leak, hint_leak_single, parse_note_v2, resolve_prompt

VERSION = "note-audit-v1"
FILE_VERSION = "note-audit-file-v1"
REVIEW_VERSION = "note-review-v1"
SOLUTION_FILE_VERSION = "solution-audit-file-v1"
SOLUTION_REVIEW_VERSION = "solution-review-v1"
SINGLE_NOTE_AUDIT_VERSION = "single-response-note-audit-v1"
SINGLE_SOLUTION_REVIEW_VERSION = "single-response-solution-review-v1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def review_bindings(record: dict, provenance: dict) -> dict:
    """Hash exact generated text plus all training-input metadata, not merely file sizes."""
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("audit: nonempty source provenance required")
    note = record.get("note")
    text = note["text"] if note else ""
    parsed = parse_note_v2(text)
    return {"note_sha256": text_sha256(text), "note_record_sha256": digest(note),
            "t1_sha256": digest(record["t1"]), "source_sha256": digest(provenance),
            "input_sha256": digest({key: record[key] for key in ("p0_text", "image_paths", "subject", "source")}),
            "p0_sha256": text_sha256(record["p0_text"]), "t1_text_sha256": text_sha256(record["t1"]["text"]),
            "note_target_sha256": text_sha256(parsed.note_text) if parsed else None,
            "note_prompt_sha256": text_sha256(note["prompt"]) if note else None,
            "lesson_sha256": text_sha256(parsed.lesson) if parsed else None}


def _review(review: dict | None, record: dict, bindings: dict) -> None:
    if review is None:
        return
    if (not isinstance(review, dict) or review.get("version") != REVIEW_VERSION or
            review.get("problem_id") != record["problem_id"] or review.get("bindings") != bindings):
        raise ValueError("audit: review identity/bindings mismatch")
    kind = review.get("reviewer_kind")
    if (kind not in ("human", "ai_review", "program") or type(review.get("human_verified")) is not bool or
            (review["human_verified"] and kind != "human")):
        raise ValueError("audit: reviewer kind/human_verified mismatch")
    if review.get("method") not in ("source_review", "bounded_check", "format_only", "no_contradiction", "regex"):
        raise ValueError("audit: unsupported review method")
    for key, results, scopes in (("content", ("PASS", "FAIL", "UNKNOWN"), ("entire_note", "partial")),
                                 ("leak", ("CLEAR", "LEAK", "UNCERTAIN"), ("entire_lesson", "partial"))):
        check = review.get(key)
        if not isinstance(check, dict) or check.get("result") not in results or check.get("scope") not in scopes:
            raise ValueError(f"audit: invalid {key} review result/scope")
        evidence = check.get("evidence")
        if (not isinstance(evidence, list) or any(not isinstance(ref, str) or not ref.strip() for ref in evidence) or
                (check["result"] in ("PASS", "FAIL", "CLEAR", "LEAK") and not evidence)):
            raise ValueError(f"audit: {key} review requires explicit evidence references")


def _strings(values: tuple[str, ...] | list[str]) -> list[str]:
    if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("audit: aliases/acronym candidates must be nonempty strings")
    return list(values)


def audit_note(record: dict, *, gold: str, choices: tuple[str, ...] | list[str], provenance: dict,
               review: dict | None = None, aliases: tuple[str, ...] | list[str] = (),
               acronym_candidates: tuple[str, ...] | list[str] = ()) -> dict:
    """Return a deterministic private audit. Missing/partial reviews never certify whole free prose.

    aliases are source-confirmed answer/option synonyms. acronym_candidates are unconfirmed potential
    synonyms: a matching candidate blocks CLEAR, including when a reviewer overlooked it.
    """
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("audit: nonblank gold required for private leak checks")
    choices, aliases, acronym_candidates = _strings(choices), _strings(aliases), _strings(acronym_candidates)
    bindings = review_bindings(record, provenance)
    _review(review, record, bindings)
    note = record.get("note")
    parsed = parse_note_v2(note["text"]) if note else None
    format_ok = note is not None and parsed is not None and note["finish_reason"] == "stop"
    content = {"result": "UNKNOWN", "scope": "partial", "reason": "MISSING_REVIEW"}
    leak = {"result": "UNCERTAIN", "scope": "partial", "reason": "MISSING_REVIEW"}
    if review is not None:
        full_review = review["method"] == "source_review" and review["reviewer_kind"] in ("human", "ai_review")
        for key, decision, positive in (("content", content, "PASS"), ("leak", leak, "CLEAR")):
            check = review[key]
            required_scope = "entire_note" if key == "content" else "entire_lesson"
            if check["result"] == positive and (not full_review or check["scope"] != required_scope):
                decision.update(scope=check["scope"], reason="BOUNDED_REVIEW_ONLY")
            else:
                decision.update(result=check["result"], scope=check["scope"], reason="EXPLICIT_SOURCE_REVIEW")
    checks: dict[str, str | None] = {
        "hint_leak": None, "alias_leak": None, "acronym_candidate": None, "explicit_contradiction": None}
    if parsed is not None:
        checks["hint_leak"] = hint_leak(parsed.lesson, gold, choices)
        checks["alias_leak"] = next((v for v in aliases if _has_words(parsed.lesson, v)), None)
        checks["acronym_candidate"] = next((v for v in acronym_candidates if _has_words(parsed.lesson, v)), None)
        if parsed.image_reading == "MISREAD" and parsed.cause == "NONE":
            checks["explicit_contradiction"] = "MISREAD_WITH_CAUSE_NONE"
            content.update(result="FAIL", scope="partial", reason="MISREAD_WITH_CAUSE_NONE")
        if checks["hint_leak"] or checks["alias_leak"]:
            leak.update(result="LEAK", scope="partial", reason=checks["hint_leak"] or "SOURCE_ALIAS")
        elif checks["acronym_candidate"]:
            leak.update(result="UNCERTAIN", scope="partial", reason="ACRONYM_CANDIDATE")
    if not format_ok:
        content.update(result="UNKNOWN", scope="partial", reason="FORMAT_INVALID")
        leak.update(result="UNCERTAIN", scope="partial", reason="FORMAT_INVALID")
    body = {"version": VERSION, "problem_id": record["problem_id"], "bindings": bindings,
            "format": {"result": "PASS" if format_ok else "FAIL"}, "content": content, "leak": leak,
            "checks": checks, "review": review, "review_sha256": digest(review) if review is not None else None,
            "leak_inputs": {"gold": gold, "choices": choices, "aliases": aliases,
                            "acronym_candidates": acronym_candidates}}
    return body | {"sha256": digest(body)}


def validate_audit(audit: dict, record: dict, provenance: dict) -> dict:
    """Recompute decisions and all bindings; reject stale text, source metadata or self-digest."""
    if not isinstance(audit, dict) or audit.get("version") != VERSION:
        raise ValueError("audit: unsupported audit version")
    if audit.get("sha256") != digest({k: v for k, v in audit.items() if k != "sha256"}):
        raise ValueError("audit: sha256 mismatch")
    try:
        expected = audit_note(record, provenance=provenance, review=audit["review"], **audit["leak_inputs"])
    except (KeyError, TypeError) as error:
        raise ValueError("audit: malformed audit") from error
    if audit != expected:
        raise ValueError("audit: record/source/decision binding mismatch")
    return audit


def _single_source_bindings(record: dict, provenance: dict) -> dict:
    """Original source-file hashes are separate from a generation's decoded/processor input hashes."""
    try:
        paths, hashes = record["image_paths"], record["image_sha256"]
        if (not isinstance(provenance, dict) or not provenance or type(paths) is not list or not paths
                or any(not isinstance(path, str) or not path.strip() for path in paths)
                or type(hashes) is not list or len(hashes) != len(paths)
                or any(not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                       for sha in hashes)
                or any(not isinstance(record[key], str) or not record[key].strip()
                       for key in ("problem_id", "p0_text", "subject", "source"))):
            raise ValueError("single audit: source provenance and original image-file hashes required")
        return {"input_sha256": digest({key: record[key] for key in
                                       ("p0_text", "image_paths", "image_sha256", "subject", "source")}),
                "source_sha256": digest(provenance), "image_files_sha256": digest(list(zip(paths, hashes, strict=True))),
                "p0_sha256": text_sha256(record["p0_text"]), "t1_sha256": digest(record["t1"])}
    except (KeyError, TypeError) as error:
        raise ValueError("single audit: malformed source/input record") from error


def single_note_review_bindings(record: dict, provenance: dict) -> dict:
    """Bind the reviewer's opinion to original image files and complete generated T1/note records."""
    return _single_source_bindings(record, provenance) | {"note_record_sha256": digest(record.get("note"))}


def audit_single_note(record: dict, *, gold: str, choices: tuple[str, ...] | list[str], provenance: dict,
                      review: dict | None = None, aliases: tuple[str, ...] | list[str] = (),
                      acronym_candidates: tuple[str, ...] | list[str] = ()) -> dict:
    """Keep the old note audit intact; add the single-response leak rule and strict whole-note admission."""
    bindings = single_note_review_bindings(record, provenance)
    if review is not None and (not isinstance(review, dict) or review.get("single_bindings") != bindings):
        raise ValueError("single audit: reviewer source/image/generation bindings mismatch")
    base = audit_note(record, gold=gold, choices=choices, provenance=provenance, review=review,
                      aliases=aliases, acronym_candidates=acronym_candidates)
    note = record.get("note")
    parsed = parse_note_v2(note["text"]) if note else None
    leak = hint_leak_single(parsed.lesson, gold, choices, aliases=aliases) if parsed else None
    accepted = (base["format"]["result"] == "PASS"
                and base["content"]["result"] == "PASS" and base["content"]["scope"] == "entire_note"
                and base["leak"]["result"] == "CLEAR" and base["leak"]["scope"] == "entire_lesson"
                and all(value is None for value in base["checks"].values()) and leak is None)
    body = {"version": SINGLE_NOTE_AUDIT_VERSION, "problem_id": record["problem_id"], "bindings": bindings,
            "base_audit": base, "single_hint_leak": leak, "accepted": accepted}
    return body | {"sha256": digest(body)}


def validate_single_note_audit(audit: dict, record: dict, provenance: dict) -> dict:
    """Recompute both audit contracts; a re-hashed stale/forged single receipt is still rejected."""
    if (not isinstance(audit, dict) or audit.get("version") != SINGLE_NOTE_AUDIT_VERSION
            or audit.get("sha256") != digest({k: v for k, v in audit.items() if k != "sha256"})):
        raise ValueError("single audit: unsupported note audit or sha256 mismatch")
    try:
        base = validate_audit(audit["base_audit"], record, provenance)
        expected = audit_single_note(record, provenance=provenance, review=base["review"], **base["leak_inputs"])
    except (KeyError, TypeError) as error:
        raise ValueError("single audit: malformed note audit") from error
    if audit != expected:
        raise ValueError("single audit: note record/source/decision binding mismatch")
    return audit


def single_solution_review_bindings(row: dict, record: dict, provenance: dict,
                                    note_audit: dict | None = None) -> dict:
    """Bind A1/A2 to actual generated prose and hint-free P0; A1 does not depend on a note."""
    try:
        kind = row["kind"]
        if kind not in ("A1", "A2") or row["target_source"] != {"A1": "T1", "A2": "RESOLVE"}[kind]:
            raise ValueError("single solution audit: only matching A1/T1 or A2/RESOLVE rows are allowed")
        if (row["problem_id"] != record["problem_id"] or not isinstance(row["row_id"], str) or not row["row_id"].strip()
                or row["prompt"] != record["p0_text"]
                or any(row[key] != record[key] for key in ("image_paths", "subject", "source"))
                or ("image_sha256" in row and row["image_sha256"] != record["image_sha256"])):
            raise ValueError("single solution audit: row must use the exact hint-free P0/source/image input")
        bindings = _single_source_bindings(record, provenance)
        generation = record["t1" if kind == "A1" else "resolve"]
        if (generation["finish_reason"] != "stop" or generation["status"] != "CORRECT"
                or not isinstance(generation["text"], str) or not generation["text"].strip()
                or row["target"] != generation["text"]):
            raise ValueError("single solution audit: target must equal the completed correct generation")
        lesson_sha = note_sha = None
        if kind == "A2":
            if note_audit is None or not validate_single_note_audit(note_audit, record, provenance)["accepted"]:
                raise ValueError("single solution audit: A2 requires an accepted single note audit")
            note_sha = note_audit["sha256"]
            parsed = parse_note_v2(record["note"]["text"])
            if parsed is None:
                raise ValueError("single solution audit: A2 requires a parsed note")
            lesson_sha = text_sha256(parsed.lesson)
            if (generation.get("prompt") != resolve_prompt(record["p0_text"], parsed.lesson)
                    or generation.get("lesson_sha256") != lesson_sha
                    or ("lesson_sha256" in row and row["lesson_sha256"] != lesson_sha)):
                raise ValueError("single solution audit: A2 generation prompt/lesson binding mismatch")
        elif generation.get("prompt", record["p0_text"]) != record["p0_text"]:
            raise ValueError("single solution audit: T1 generation must use hint-free P0")
        return bindings | {"row_sha256": digest(row), "target_sha256": text_sha256(row["target"]),
                           "prompt_sha256": text_sha256(row["prompt"]), "generation_sha256": digest(generation),
                           "note_audit_sha256": note_sha,
                           "lesson_sha256": lesson_sha}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("single solution audit: malformed row/generation bindings") from error


def validate_single_solution_review(review: dict | None, row: dict, record: dict, provenance: dict,
                                    note_audit: dict | None = None) -> dict:
    """Missing/partial/negative reviews reject prose; malformed or stale receipts raise instead of passing."""
    bindings = single_solution_review_bindings(row, record, provenance, note_audit)
    reason = None
    if review is None:
        reason = "SOLUTION_CONTENT_MISSING_REVIEW"
    else:
        fields = {"version", "row_id", "problem_id", "target_source", "bindings", "reviewer_kind",
                  "human_verified", "method", "content"}
        if (not isinstance(review, dict) or set(review) != fields
                or review["version"] != SINGLE_SOLUTION_REVIEW_VERSION or review["row_id"] != row["row_id"]
                or review["problem_id"] != row["problem_id"] or review["target_source"] != row["target_source"]
                or review["bindings"] != bindings):
            raise ValueError("single solution audit: review identity/bindings mismatch")
        kind, method = review["reviewer_kind"], review["method"]
        if (kind not in ("human", "ai_review", "program") or type(review["human_verified"]) is not bool
                or review["human_verified"] and kind != "human"):
            raise ValueError("single solution audit: reviewer kind/human_verified mismatch")
        if method not in ("source_review", "bounded_check", "format_only", "no_contradiction", "regex"):
            raise ValueError("single solution audit: unsupported review method")
        check = review["content"]
        if (not isinstance(check, dict) or set(check) != {"result", "scope", "evidence"}
                or check["result"] not in ("PASS", "FAIL", "UNKNOWN")
                or check["scope"] not in ("entire_solution", "partial")):
            raise ValueError("single solution audit: invalid content result/scope")
        evidence = check["evidence"]
        if (not isinstance(evidence, list) or any(not isinstance(ref, str) or not ref.strip() for ref in evidence)
                or check["result"] in ("PASS", "FAIL") and not evidence):
            raise ValueError("single solution audit: content review requires explicit evidence references")
        if check["result"] != "PASS":
            reason = f"SOLUTION_CONTENT_{check['result']}"
        elif kind not in ("human", "ai_review") or method != "source_review" or check["scope"] != "entire_solution":
            reason = "SOLUTION_CONTENT_BOUNDED_REVIEW_ONLY"
    return {"accepted": reason is None, "reason": reason, "bindings": bindings,
            "review_sha256": digest(review) if review is not None else None}


def _read_items(path: Path, expected_sha256: str | None = None) -> dict[str, dict]:
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("audit: file sha256 mismatch")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"version", "items"} or data["version"] != FILE_VERSION:
        raise ValueError("audit: invalid file wrapper")
    if not isinstance(data["items"], dict):
        raise ValueError("audit: file items must be an object")
    for pid, audit in data["items"].items():
        if (not isinstance(audit, dict) or audit.get("version") != VERSION or audit.get("problem_id") != pid or
                audit.get("sha256") != digest({k: v for k, v in audit.items() if k != "sha256"})):
            raise ValueError("audit: file item identity/sha256 mismatch")
        if audit.get("review_sha256") != (digest(audit["review"]) if audit.get("review") is not None else None):
            raise ValueError("audit: review sha256 mismatch")
        bindings = audit.get("bindings")
        if not isinstance(bindings, dict):
            raise ValueError("audit: file item bindings must be an object")
        _review(audit.get("review"), {"problem_id": pid}, bindings)
    return data["items"]


def read_audits(path: Path, records: list[dict] | dict[str, dict], provenances: dict[str, dict], *,
                expected_sha256: str | None = None) -> dict[str, dict]:
    """Read a private note-audit-file-v1 wrapper and validate every item against current inputs."""
    items = _read_items(path, expected_sha256)
    if isinstance(records, list):
        by_id = {r["problem_id"]: r for r in records}
        if len(by_id) != len(records):
            raise ValueError("audit: duplicate record problem_id")
    else:
        by_id = records
    if set(items) != set(by_id) or not set(items) <= set(provenances):
        raise ValueError("audit: file/record/source ids mismatch")
    for pid, audit in items.items():
        if not isinstance(audit, dict) or audit.get("problem_id") != pid or by_id[pid]["problem_id"] != pid:
            raise ValueError("audit: file item problem_id mismatch")
        validate_audit(audit, by_id[pid], provenances[pid])
    return items


def validate_row_audits(rows: list[dict], audit_path: Path) -> None:
    """Verify immutable row links before admission/loading; the caller also pins the whole audit file.

    NOTE/T1 targets are matched to pre-generation audit bindings. RESOLVE's output is collected later:
    its self-hash and audited lesson are checked here; correctness remains the collector's grading evidence.
    """
    audits = _read_items(audit_path)
    for row in rows:
        try:
            audit = audits[row["problem_id"]]
            bindings = audit["bindings"]
            if (row["audit_sha256"] != audit["sha256"] or row["source_sha256"] != bindings["source_sha256"] or
                    row["prompt_sha256"] != text_sha256(row["prompt"]) or
                    row["target_sha256"] != text_sha256(row["target"])):
                raise ValueError("audit: row hash/crosslink mismatch")
            review = audit["review"]
            content_ok = (audit["format"]["result"] == "PASS" and audit["content"]["result"] == "PASS" and
                          audit["content"]["scope"] == "entire_note" and review is not None and
                          review["method"] == "source_review" and review["reviewer_kind"] in ("human", "ai_review") and
                          review["content"]["result"] == "PASS" and review["content"]["scope"] == "entire_note" and
                          audit["checks"]["explicit_contradiction"] is None)
            if row["kind"] == "B" and row["target_source"] == "NOTE":
                if (not content_ok or row["target_sha256"] != bindings["note_target_sha256"] or
                        row["prompt_sha256"] != bindings["note_prompt_sha256"]):
                    raise ValueError("audit: note row target/prompt/content mismatch")
            elif row["kind"] == "A" and row["target_source"] == "T1":
                if (row["target_sha256"] != bindings["t1_text_sha256"] or
                        row["prompt_sha256"] != bindings["p0_sha256"]):
                    raise ValueError("audit: T1 row target/prompt mismatch")
            elif row["kind"] == "A" and row["target_source"] == "RESOLVE":
                clear = (content_ok and audit["leak"]["result"] == "CLEAR" and
                         audit["leak"]["scope"] == "entire_lesson" and review["leak"]["result"] == "CLEAR" and
                         review["leak"]["scope"] == "entire_lesson" and
                         all(audit["checks"][key] is None for key in ("hint_leak", "alias_leak", "acronym_candidate")))
                if (not clear or row["lesson_sha256"] != bindings["lesson_sha256"] or
                        row["prompt_sha256"] != bindings["p0_sha256"]):
                    raise ValueError("audit: resolve row lesson/content/prompt mismatch")
            else:
                raise ValueError("audit: invalid row kind/target_source")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("audit: missing or malformed row/audit linkage") from error


def solution_review_bindings(row: dict, note_audit: dict) -> dict:
    """Bind an independent solution review to the exact row and its existing source/image audit."""
    try:
        if row["kind"] != "A" or row["target_source"] not in ("T1", "RESOLVE"):
            raise ValueError("solution audit: reviews are only for A solution rows")
        return {"row_sha256": digest(row), "target_sha256": text_sha256(row["target"]),
                "prompt_sha256": text_sha256(row["prompt"]),
                "source_sha256": note_audit["bindings"]["source_sha256"],
                "input_sha256": note_audit["bindings"]["input_sha256"],
                "note_audit_sha256": note_audit["sha256"]}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("solution audit: malformed row/note bindings") from error


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("solution audit: duplicate JSON key")
        result[key] = value
    return result


def solution_quality_reasons(rows: list[dict], note_audit_path: Path, solution_audit_path: Path, *,
                             rows_sha256: str) -> dict[str, str]:
    """Return only rejected A rows; note approval never certifies a solution's prose.

    Missing reviews are explicit drops. Malformed or stale receipts are errors, including when no A rows remain.
    """
    if (not isinstance(rows_sha256, str) or len(rows_sha256) != 64 or
            any(character not in "0123456789abcdef" for character in rows_sha256)):
        raise ValueError("solution audit: explicit rows SHA256 required")
    note_audit_path, solution_audit_path = Path(note_audit_path), Path(solution_audit_path)
    note_raw = note_audit_path.read_bytes()
    json.loads(note_raw, object_pairs_hook=_unique_json_pairs)
    note_sha256 = hashlib.sha256(note_raw).hexdigest()
    validate_row_audits(rows, note_audit_path)
    audits = _read_items(note_audit_path, expected_sha256=note_sha256)
    data = json.loads(solution_audit_path.read_bytes(), object_pairs_hook=_unique_json_pairs)
    if (not isinstance(data, dict) or set(data) != {"version", "bindings", "items"} or
            data["version"] != SOLUTION_FILE_VERSION or
            data["bindings"] != {"rows_sha256": rows_sha256, "note_audit_sha256": note_sha256} or
            not isinstance(data["items"], dict)):
        raise ValueError("solution audit: invalid file wrapper or file bindings")
    by_id: dict[str, dict] = {}
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip() or row_id in by_id:
            raise ValueError("solution audit: missing or duplicate row_id")
        by_id[row_id] = row
    solutions = {row_id: row for row_id, row in by_id.items() if row["kind"] == "A"}
    if not set(data["items"]) <= set(solutions):
        raise ValueError("solution audit: unknown or non-A review row_id")
    reasons = {}
    fields = {"version", "row_id", "problem_id", "target_source", "bindings", "reviewer_kind",
              "human_verified", "method", "content"}
    for row_id, row in solutions.items():
        audit = audits[row["problem_id"]]
        try:
            input_sha256 = digest({"p0_text": row["prompt"],
                                   **{key: row[key] for key in ("image_paths", "subject", "source")}})
        except (KeyError, TypeError) as error:
            raise ValueError("solution audit: missing or malformed solution input") from error
        if input_sha256 != audit["bindings"]["input_sha256"]:
            raise ValueError("solution audit: source/image input binding mismatch")
        review = data["items"].get(row_id)
        if row_id not in data["items"]:
            reasons[row_id] = "SOLUTION_CONTENT_MISSING_REVIEW"
            continue
        if (not isinstance(review, dict) or set(review) != fields or
                review["version"] != SOLUTION_REVIEW_VERSION or review["row_id"] != row_id or
                review["problem_id"] != row["problem_id"] or review["target_source"] != row["target_source"] or
                review["bindings"] != solution_review_bindings(row, audit)):
            raise ValueError("solution audit: review identity/bindings mismatch")
        kind, method = review["reviewer_kind"], review["method"]
        if (kind not in ("human", "ai_review", "program") or type(review["human_verified"]) is not bool or
                review["human_verified"] and kind != "human"):
            raise ValueError("solution audit: reviewer kind/human_verified mismatch")
        if method not in ("source_review", "bounded_check", "format_only", "no_contradiction", "regex"):
            raise ValueError("solution audit: unsupported review method")
        check = review["content"]
        if (not isinstance(check, dict) or set(check) != {"result", "scope", "evidence"} or
                check["result"] not in ("PASS", "FAIL", "UNKNOWN") or
                check["scope"] not in ("entire_solution", "partial")):
            raise ValueError("solution audit: invalid content result/scope")
        evidence = check["evidence"]
        if (not isinstance(evidence, list) or any(not isinstance(ref, str) or not ref.strip() for ref in evidence) or
                check["result"] in ("PASS", "FAIL") and not evidence):
            raise ValueError("solution audit: content review requires explicit evidence references")
        if check["result"] != "PASS":
            reasons[row_id] = f"SOLUTION_CONTENT_{check['result']}"
        elif kind not in ("human", "ai_review") or method != "source_review" or check["scope"] != "entire_solution":
            reasons[row_id] = "SOLUTION_CONTENT_BOUNDED_REVIEW_ONLY"
    return reasons
