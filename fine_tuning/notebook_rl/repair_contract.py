"""Bounded repair-check-v1 codec; spans refer to original Python characters."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypeGuard, cast

if TYPE_CHECKING:
    from .schema import Sample, VerifierResult

VERSION = "repair-check-v1"
BOUNDARY_POLICY = "ascii-blank-lines-before-native-header-v2"
FINAL_VERSION = "repair-note-v2"
ORIGIN_VERSION = "repair-origin-inventory-v1"
# Opt-in gold-free finalization channel; kept here so the profile never imports training helpers.
FINALIZATION_VERSION = "repair-finalization-credit-v1"
FINALIZATION_WEIGHT = .25
MAX_RAW_CHARS = 131072
MAX_CHECK_CHARS = 16384
MAX_CLAIM_CHARS = 4096
JsonValue: TypeAlias = str | int | float | None | list["JsonValue"] | dict[str, "JsonValue"]
Intent: TypeAlias = Literal["RECHECK_OBSERVATION", "RECHECK_OPERATION", "UNKNOWN"]
FinalRelation: TypeAlias = Literal["CORRECTED", "PERSISTED", "UNKNOWN", "OTHER_BRANCH_COMPARISON", "FINALIZED"]
_FINAL_OUTCOMES = ("CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED",
                   "UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR", "INVALID_RESPONSE_CONTRACT")
GRAMMAR_ERRORS = frozenset({"RAW_TYPE", "RAW_TOO_LARGE", "INVALID_UTF8", "SECTION_HEADER", "INVALID_JSON",
    "DUPLICATE_JSON_KEY", "NONFINITE_JSON", "SCHEMA_FIELDS", "INVALID_FIELD", "CLAIM_LIMIT",
    "DUPLICATE_ORIGIN", "UNCERTAIN_REFS"})
_ERROR_CODES = frozenset({
    "CONTRACT_VERSION", "RAW_TYPE", "RAW_TOO_LARGE", "INVALID_UTF8", "SECTION_HEADER",
    "INVALID_JSON", "DUPLICATE_JSON_KEY", "NONFINITE_JSON", "SCHEMA_FIELDS", "INVALID_FIELD",
    "CLAIM_LIMIT", "DUPLICATE_ORIGIN", "UNCERTAIN_REFS", "SECTION_MISMATCH", "TOKEN_TEXT_MISMATCH",
    "TOKEN_IDS", "ORIGIN_INPUT", "ORIGIN_CONTRACTS", "ORIGIN_LINEAGE", "ORIGIN_REFERENCE",
})
_HEADER = re.compile(r"(?:^|(?<=[\r\v\f\x1c-\x1e\x85\u2028\u2029]))"
                     r"[ \t]*(?:ErrorCheck|RevisedSolution):", re.M)


class RepairContractError(ValueError):
    """A bounded reason and field/offset context, without submitted text."""

    def __init__(self, code: str, *, at: str = "raw", offset: int | None = None):
        if code not in _ERROR_CODES:
            raise ValueError("repair contract: unknown error code")
        self.code, self.at, self.offset = code, at, offset
        super().__init__(f"repair contract: {code} at {at}" +
                         (f" (char {offset})" if offset is not None else ""))


@dataclass(frozen=True)
class RawSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end:
            raise RepairContractError("INVALID_FIELD", at="span")


@dataclass(frozen=True)
class RetrySections:
    raw_sha: str
    check_span: RawSpan
    solution_span: RawSpan
    claim_spans: tuple[RawSpan, ...]
    contract_version: str
    uncertain_refs: tuple[str, ...]


@dataclass(frozen=True)
class RepairClaim:
    origin_ref: str
    intent: Intent
    rule: str
    proposed_value: JsonValue
    raw_spans: dict[str, RawSpan]


@dataclass(frozen=True)
class FinalRepairClaim:
    origin_ref: str
    later_ref: str
    relation: FinalRelation
    rule: str
    repaired_value: JsonValue
    unknown_reason: Literal["DEPENDENCY_BLOCKED"] | None
    blocked_by: tuple[str, ...]
    raw_spans: dict[str, RawSpan]


@dataclass(frozen=True)
class ParsedRepairNote:
    version: str
    scope: Literal["BOUNDED_CLAIMS_FULL_HISTORY"]
    attempts: tuple[dict[str, JsonValue], ...]
    success_sample_id: str
    claims: tuple[FinalRepairClaim, ...]
    uncertain_refs: tuple[str, ...]
    lesson_text: str
    raw_sha: str
    raw_spans: dict[str, RawSpan]


@dataclass(frozen=True)
class PublicOrigin:
    """A source ref; audits must rederive the complete inventory/source binding."""

    ref_id: str
    sample_id: str
    kind: str
    rule: str
    locus: str
    occurrence_spans: tuple[RawSpan, ...]
    raw_sha: str


@dataclass(frozen=True)
class PublicOriginSource:
    sample_id: str
    parent_sample_id: str | None
    finish_reason: str
    response_contract: str | None
    raw_sha: str
    solution_span: RawSpan | None
    solution_sha: str | None
    projection_error_code: str | None


@dataclass(frozen=True)
class PublicOriginInventory:
    contract_version: str
    parent_sample_id: str | None
    origins: tuple[PublicOrigin, ...]
    sources: tuple[PublicOriginSource, ...]
    sha256: str

    @property
    def fixed_units(self) -> int:
        return len(self.origins)


def _unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise RepairContractError("DUPLICATE_JSON_KEY", at="check")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise RepairContractError("NONFINITE_JSON", at="check")


def _integer(value: str) -> int:
    if len(value.lstrip("-")) > 19 or not -(2**63) <= (result := int(value)) <= 2**63 - 1:
        raise RepairContractError("INVALID_FIELD", at="check.number")
    return result


def _float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RepairContractError("NONFINITE_JSON", at="check")
    return result


_DECODER = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_constant,
                            parse_int=_integer, parse_float=_float)


def _space(raw: str, start: int) -> int:
    while start < len(raw) and raw[start] in " \t\r\n":
        start += 1
    return start


def _members(raw: str, start: int) -> dict[str, tuple[JsonValue, RawSpan]]:
    """Walk a previously decoded object with the same strict decoder."""
    index = _space(raw, start + 1)
    result = {}
    while raw[index] != "}":
        key, end = _DECODER.raw_decode(raw, index)
        begin = _space(raw, _space(raw, end) + 1)
        value, end = _DECODER.raw_decode(raw, begin)
        result[key] = (value, RawSpan(begin, end))
        index = _space(raw, end)
        if raw[index] == "}":
            break
        index = _space(raw, index + 1)
    return result


def _elements(raw: str, start: int) -> list[tuple[JsonValue, RawSpan]]:
    index = _space(raw, start + 1)
    result = []
    while raw[index] != "]":
        value, end = _DECODER.raw_decode(raw, index)
        result.append((value, RawSpan(index, end)))
        index = _space(raw, end)
        if raw[index] == "]":
            break
        index = _space(raw, index + 1)
    return result


def _text(value: object, limit: int) -> TypeGuard[str]:
    return (type(value) is str and 0 < len(value) <= limit and bool(value.strip(" \t"))
            and value.splitlines() == [value] and all(ch == "\t" or ch.isprintable() for ch in value))


def _operation(value: JsonValue, depth: int = 0) -> bool:
    if depth > 4 or type(value) is bool:
        return False
    if value is None or type(value) in (int, float):
        return True
    if type(value) is str:
        return _text(value, 2048)
    if isinstance(value, list):
        return len(value) <= 64 and all(_operation(item, depth + 1) for item in value)
    return (isinstance(value, dict) and len(value) <= 32
            and all(_text(key, 128) and _operation(item, depth + 1) for key, item in value.items()))


def _value_spans(raw: str, value: JsonValue, span: RawSpan, path: str) -> dict[str, RawSpan]:
    result = {path: span}
    if isinstance(value, dict):
        for key, (item, child) in _members(raw, span.start).items():
            pointer = key.replace("~", "~0").replace("/", "~1")
            result.update(_value_spans(raw, item, child, f"{path}/{pointer}"))
    elif isinstance(value, list):
        for index, (item, child) in enumerate(_elements(raw, span.start)):
            result.update(_value_spans(raw, item, child, f"{path}/{index}"))
    return result


def _parse(raw: str, contract_version: str) -> tuple[RetrySections, tuple[RepairClaim, ...]]:
    if type(contract_version) is not str or contract_version != VERSION:
        raise RepairContractError("CONTRACT_VERSION")
    if type(raw) is not str:
        raise RepairContractError("RAW_TYPE")
    if len(raw) > MAX_RAW_CHARS:
        raise RepairContractError("RAW_TOO_LARGE")
    try:
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        raise RepairContractError("INVALID_UTF8") from None
    prefix = re.match(r"ErrorCheck:[ \t]*", raw)
    if prefix is None or not raw.startswith("{", prefix.end()):
        raise RepairContractError("SECTION_HEADER", at="check", offset=0)
    start = prefix.end()
    # The object must close within the bound; a huge solution is not decoded as JSON.
    try:
        payload, relative_end = _DECODER.raw_decode(raw[start:start + MAX_CHECK_CHARS + 1])
    except json.JSONDecodeError as exc:
        raise RepairContractError("INVALID_JSON", at="check", offset=start + exc.pos) from None
    except (RecursionError, OverflowError, ValueError) as exc:
        if isinstance(exc, RepairContractError):
            raise
        raise RepairContractError("INVALID_JSON", at="check") from None
    end = start + relative_end
    if end - start > MAX_CHECK_CHARS:
        raise RepairContractError("RAW_TOO_LARGE", at="check", offset=start)
    boundary = re.match(r"[ \t]*\r?\n(?:[ \t]*\r?\n)*RevisedSolution:[ \t]*\r?\n", raw[end:])
    if boundary is None:
        raise RepairContractError("SECTION_HEADER", at="solution", offset=end)
    solution_start = end + boundary.end()
    if not raw[solution_start:].strip(" \t\r\n") or _HEADER.search(raw, solution_start):
        raise RepairContractError("SECTION_HEADER", at="solution", offset=solution_start)
    if type(payload) is not dict or set(payload) != {"version", "claims", "uncertain_refs"}:
        raise RepairContractError("SCHEMA_FIELDS", at="check")
    if payload["version"] != VERSION or type(payload["version"]) is not str:
        raise RepairContractError("INVALID_FIELD", at="check.version")
    if type(payload["claims"]) is not list or len(payload["claims"]) > 3:
        raise RepairContractError("CLAIM_LIMIT", at="check.claims")
    fields = _members(raw, start)
    claims = []
    origins = set()
    for index, (value, span) in enumerate(_elements(raw, fields["claims"][1].start)):
        at = f"check.claims[{index}]"
        if span.end - span.start > MAX_CLAIM_CHARS:
            raise RepairContractError("CLAIM_LIMIT", at=at, offset=span.start)
        if type(value) is not dict or set(value) != {"origin_ref", "intent", "rule", "proposed_value"}:
            raise RepairContractError("SCHEMA_FIELDS", at=at, offset=span.start)
        origin, intent, rule, proposed = (value[key] for key in
                                        ("origin_ref", "intent", "rule", "proposed_value"))
        if (not _text(origin, 256) or not _text(rule, 128) or type(intent) is not str
                or intent not in ("RECHECK_OBSERVATION", "RECHECK_OPERATION", "UNKNOWN")):
            raise RepairContractError("INVALID_FIELD", at=at, offset=span.start)
        if origin in origins:
            raise RepairContractError("DUPLICATE_ORIGIN", at=at, offset=span.start)
        if ((intent == "UNKNOWN" and proposed is not None)
                or (intent == "RECHECK_OBSERVATION" and not _text(proposed, 2048))
                or (intent == "RECHECK_OPERATION" and
                    (type(proposed) is not dict or not proposed or not _operation(proposed)))):
            raise RepairContractError("INVALID_FIELD", at=at + ".proposed_value", offset=span.start)
        origins.add(origin)
        spans = {"claim": span}
        for key, (item, literal) in _members(raw, span.start).items():
            spans.update(_value_spans(raw, item, literal, key))
        claims.append(RepairClaim(origin, cast(Intent, intent), rule, proposed, spans))
    uncertain = payload["uncertain_refs"]
    if (type(uncertain) is not list or len(uncertain) > 3
            or any(not _text(ref, 256) for ref in uncertain)
            or len(set(uncertain)) != len(uncertain) or origins.intersection(uncertain)):
        raise RepairContractError("UNCERTAIN_REFS", at="check.uncertain_refs")
    sections = RetrySections(raw_sha, RawSpan(start, end), RawSpan(solution_start, len(raw)),
                             tuple(claim.raw_spans["claim"] for claim in claims), VERSION,
                             tuple(uncertain))
    return sections, tuple(claims)


def parse_retry_sections(raw: str, *, contract_version: str) -> RetrySections:
    """Parse exactly one leading typed check and one native-solution boundary."""
    return _parse(raw, contract_version)[0]


def public_attempt_outcome(raw: str, *, response_contract: str | None,
                           result: VerifierResult | None) -> tuple[str | None, str | None]:
    """Expose fresh grades or independently proven response syntax failure only."""
    from .schema import to_json

    if response_contract is not None:
        if type(response_contract) is not str or response_contract != VERSION:
            raise RepairContractError("CONTRACT_VERSION")
        try:
            parse_retry_sections(raw, contract_version=response_contract)
        except RepairContractError as exc:
            if exc.code not in GRAMMAR_ERRORS:
                raise
            if result is not None:
                raise ValueError("repair outcome: malformed response acquired a native result") from None
            return "INVALID_RESPONSE_CONTRACT", None
    if result is None:
        return None, None
    to_json(result)
    return result.status, result.extracted


def parse_final_note(raw: str, *, contract_version: str) -> ParsedRepairNote:
    """Parse model-authored syntax and original spans, without certifying claims."""
    if type(contract_version) is not str or contract_version != FINAL_VERSION:
        raise RepairContractError("CONTRACT_VERSION")
    if type(raw) is not str:
        raise RepairContractError("RAW_TYPE")
    if len(raw) > MAX_RAW_CHARS:
        raise RepairContractError("RAW_TOO_LARGE")
    try:
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        raise RepairContractError("INVALID_UTF8") from None
    start = _space(raw, 0)
    try:
        payload, end = _DECODER.raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise RepairContractError("INVALID_JSON", at="note", offset=exc.pos) from None
    except (RecursionError, OverflowError, ValueError) as exc:
        if isinstance(exc, RepairContractError):
            raise
        raise RepairContractError("INVALID_JSON", at="note") from None
    if _space(raw, end) != len(raw):
        raise RepairContractError("INVALID_JSON", at="note", offset=end)
    if type(payload) is not dict or set(payload) != {
            "version", "scope", "attempts", "success_sample_id", "claims", "uncertain_refs", "lesson_text"}:
        raise RepairContractError("SCHEMA_FIELDS", at="note")
    if type(payload["version"]) is not str or payload["version"] != FINAL_VERSION:
        raise RepairContractError("INVALID_FIELD", at="note.version")
    if type(payload["scope"]) is not str or payload["scope"] != "BOUNDED_CLAIMS_FULL_HISTORY":
        raise RepairContractError("INVALID_FIELD", at="note.scope")
    attempts = payload["attempts"]
    if type(attempts) is not list or not 1 <= len(attempts) <= 8:
        raise RepairContractError("INVALID_FIELD", at="note.attempts")
    sample_ids: set[str] = set()
    for index, attempt in enumerate(attempts):
        at = f"note.attempts[{index}]"
        if type(attempt) is not dict or set(attempt) != {"sample_id", "outcome", "answer"}:
            raise RepairContractError("SCHEMA_FIELDS", at=at)
        sample_id, outcome, answer = (attempt[key] for key in ("sample_id", "outcome", "answer"))
        if (not _text(sample_id, 256) or sample_id in sample_ids
                or type(outcome) is not str or outcome not in _FINAL_OUTCOMES
                or (answer is not None and not _text(answer, 2048))):
            raise RepairContractError("INVALID_FIELD", at=at)
        sample_ids.add(sample_id)
    success = payload["success_sample_id"]
    if not _text(success, 256) or success not in sample_ids:
        raise RepairContractError("INVALID_FIELD", at="note.success_sample_id")
    if type(payload["claims"]) is not list or len(payload["claims"]) > 3:
        raise RepairContractError("CLAIM_LIMIT", at="note.claims")
    fields = _members(raw, start)
    claims: list[FinalRepairClaim] = []
    origins: set[str] = set()
    for index, (value, span) in enumerate(_elements(raw, fields["claims"][1].start)):
        at = f"note.claims[{index}]"
        if type(value) is not dict:
            raise RepairContractError("SCHEMA_FIELDS", at=at, offset=span.start)
        relation = value.get("relation")
        keys = {"origin_ref", "later_ref", "relation", "rule", "repaired_value"}
        if relation == "UNKNOWN":
            keys.update(("unknown_reason", "blocked_by"))
        if set(value) != keys:
            raise RepairContractError("SCHEMA_FIELDS", at=at, offset=span.start)
        origin, later, rule, repaired = (value[key] for key in
                                        ("origin_ref", "later_ref", "rule", "repaired_value"))
        if (not _text(origin, 256) or not _text(later, 256) or not _text(rule, 128)
                or type(relation) is not str or relation not in
                ("CORRECTED", "PERSISTED", "UNKNOWN", "OTHER_BRANCH_COMPARISON", "FINALIZED")):
            raise RepairContractError("INVALID_FIELD", at=at, offset=span.start)
        if origin in origins:
            raise RepairContractError("DUPLICATE_ORIGIN", at=at, offset=span.start)
        if ((relation == "CORRECTED" and not (_text(repaired, 2048) or
                (type(repaired) is dict and bool(repaired) and _operation(repaired))))
                or (relation != "CORRECTED" and repaired is not None)
                or (relation == "FINALIZED" and rule != "FINALIZATION")):
            raise RepairContractError("INVALID_FIELD", at=at + ".repaired_value", offset=span.start)
        blockers = value.get("blocked_by", [])
        if relation == "UNKNOWN" and (
                value["unknown_reason"] != "DEPENDENCY_BLOCKED" or type(blockers) is not list
                or not 1 <= len(blockers) <= 64 or any(not _text(ref, 256) for ref in blockers)
                or len(set(blockers)) != len(blockers)):
            raise RepairContractError("INVALID_FIELD", at=at + ".blocked_by", offset=span.start)
        origins.add(origin)
        spans = {"claim": span}
        for key, (item, literal) in _members(raw, span.start).items():
            spans.update(_value_spans(raw, item, literal, key))
        claims.append(FinalRepairClaim(origin, later, cast(FinalRelation, relation), rule, repaired,
                                      "DEPENDENCY_BLOCKED" if relation == "UNKNOWN" else None,
                                      tuple(cast(list[str], blockers)), spans))
    uncertain = payload["uncertain_refs"]
    if (type(uncertain) is not list or len(uncertain) > 3
            or any(not _text(ref, 256) for ref in uncertain)
            or len(set(uncertain)) != len(uncertain) or origins.intersection(uncertain)):
        raise RepairContractError("UNCERTAIN_REFS", at="note.uncertain_refs")
    lesson = payload["lesson_text"]
    if type(lesson) is not str or len(lesson) > 16384:
        raise RepairContractError("INVALID_FIELD", at="note.lesson_text")
    try:
        lesson.encode("utf-8")
    except UnicodeEncodeError:
        raise RepairContractError("INVALID_UTF8", at="note.lesson_text") from None
    raw_spans = {"note": RawSpan(start, end)}
    for key, (item, literal) in fields.items():
        raw_spans.update(_value_spans(raw, item, literal, key))
    return ParsedRepairNote(FINAL_VERSION, "BOUNDED_CLAIMS_FULL_HISTORY",
                            tuple(cast(list[dict[str, JsonValue]], attempts)), success, tuple(claims),
                            tuple(cast(list[str], uncertain)), lesson, raw_sha, raw_spans)


def _bound(raw: str, sections: RetrySections) -> tuple[RepairClaim, ...]:
    if type(sections) is not RetrySections:
        raise RepairContractError("SECTION_MISMATCH")
    actual, claims = _parse(raw, sections.contract_version)
    if actual != sections:
        raise RepairContractError("SECTION_MISMATCH")
    return claims


def parse_error_check(raw: str, sections: RetrySections) -> tuple[RepairClaim, ...]:
    """Return submitted claims after rechecking all raw/offset bindings."""
    return _bound(raw, sections)


def project_solution(raw: str, sections: RetrySections) -> str:
    """Return the exact native body; do not strip or rewrite the model's action."""
    _bound(raw, sections)
    return raw[sections.solution_span.start:sections.solution_span.end]


class _Tokenizer(Protocol):
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool,
               clean_up_tokenization_spaces: bool) -> str: ...


def claim_token_ownership(raw: str, sections: RetrySections, tokenizer: _Tokenizer,
                          completion_token_ids: list[int]) -> tuple[dict, ...]:
    """Map wholly contained original indices; unstable prefixes abstain entirely."""
    _bound(raw, sections)
    if (type(completion_token_ids) is not list
            or any(type(token) is not int or token < 0 for token in completion_token_ids)):
        raise RepairContractError("TOKEN_IDS")
    from .fact_credit import token_spans

    try:
        tokens = token_spans(tokenizer, completion_token_ids, raw)
    except ValueError:
        raise RepairContractError("TOKEN_TEXT_MISMATCH") from None
    result = []
    for index, span in enumerate(sections.claim_spans):
        status = "UNSTABLE_TOKEN_PREFIX" if tokens is None else "MAPPED"
        indices = []
        crossing = []
        if tokens is not None:
            indices = [i for i, (a, b) in enumerate(tokens) if span.start <= a < b <= span.end]
            crossing = [i for i, (a, b) in enumerate(tokens)
                        if a < b and a < span.end and b > span.start
                        and not span.start <= a < b <= span.end]
            status = ("NO_OWNED_TOKENS" if not indices else
                      "MAPPED_WITH_EXCLUDED_BOUNDARIES" if crossing else "MAPPED")
        result.append({"claim_index": index, "status": status, "token_indices": indices,
                       "excluded_crossing_token_indices": crossing})
    return tuple(result)


def public_origin_inventory(samples: list[Sample], *, contract_version: str,
                            response_contracts: dict[str, str | None],
                            parent_sample_id: str | None = None) -> PublicOriginInventory:
    """Freeze public units before generation; source syntax errors remain explicit."""
    if type(contract_version) is not str or contract_version != ORIGIN_VERSION:
        raise RepairContractError("CONTRACT_VERSION", at="origins")
    if type(samples) is not list or type(response_contracts) is not dict:
        raise RepairContractError("ORIGIN_INPUT")
    public: list[tuple[str, str | None, str, str]] = []
    parents: dict[str, str | None] = {}
    for index, sample in enumerate(samples):
        at = f"sources[{index}]"
        try:
            sample_id, parent, finish, raw = (sample.sample_id, sample.parent_sample_id,
                                               sample.finish_reason, sample.text)
        except AttributeError:
            raise RepairContractError("ORIGIN_INPUT", at=at) from None
        if (not _text(sample_id, 256) or sample_id in parents
                or type(finish) is not str or finish not in ("stop", "length")):
            raise RepairContractError("ORIGIN_INPUT", at=at)
        if parent is not None and (type(parent) is not str or parent not in parents):
            raise RepairContractError("ORIGIN_LINEAGE", at=at)
        if type(raw) is not str:
            raise RepairContractError("RAW_TYPE", at=at)
        if len(raw) > MAX_RAW_CHARS:
            raise RepairContractError("RAW_TOO_LARGE", at=at)
        try:
            raw.encode("utf-8")
        except UnicodeEncodeError:
            raise RepairContractError("INVALID_UTF8", at=at) from None
        public.append((sample_id, parent, finish, raw))
        parents[sample_id] = parent
    if (any(type(key) is not str for key in response_contracts) or set(response_contracts) != set(parents)
            or any(value is not None and (type(value) is not str or value != VERSION)
                   for value in response_contracts.values())):
        raise RepairContractError("ORIGIN_CONTRACTS")
    if parent_sample_id is not None and (type(parent_sample_id) is not str or parent_sample_id not in parents):
        raise RepairContractError("ORIGIN_LINEAGE", at="parent_sample_id")
    selected = set(parents) if parent_sample_id is None else set()
    current = parent_sample_id
    while current is not None:
        selected.add(current)
        current = parents[current]

    # Legacy bookkeeping imports the rollout stack; keep verifier startup independent.
    from .whole_reference_notebook import public_inventory

    origins, sources = [], []
    seen: set[str] = set()
    for sample_id, parent, finish, raw in public:
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        response_contract = response_contracts[sample_id]
        scope: RawSpan | None = RawSpan(0, len(raw))
        error = None
        if response_contract is not None:
            try:
                scope = parse_retry_sections(raw, contract_version=response_contract).solution_span
            except RepairContractError as exc:
                scope, error = None, exc.code
        solution = raw[scope.start:scope.end] if scope is not None else ""
        source = PublicOriginSource(sample_id, parent, finish, response_contract, raw_sha, scope,
                                    hashlib.sha256(solution.encode("utf-8")).hexdigest()
                                    if scope is not None else None, error)
        view = SimpleNamespace(text=solution, sample_id=sample_id, parent_sample_id=None, finish_reason=finish)
        inventory = public_inventory([view])
        for local_ref in inventory["assessment_refs"]:
            if re.fullmatch(r"a1:r[1-9][0-9]*", local_ref) is None:
                raise RepairContractError("ORIGIN_REFERENCE")
            ref_id = sample_id + local_ref.removeprefix("a1")
            if len(ref_id) > 256 or ref_id in seen:
                raise RepairContractError("ORIGIN_REFERENCE")
            seen.add(ref_id)
            row = inventory["references"][local_ref]
            spans = []
            for occurrence in row["occurrences"]:
                start, end = occurrence[2:4]
                if (scope is None or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(solution)
                        or solution[start:end] != occurrence[-1]):
                    raise RepairContractError("ORIGIN_REFERENCE")
                spans.append(RawSpan(start + scope.start, end + scope.start))
            if not spans:
                # A malformed declared retry has raw-scope refs only, coupled to source.error.
                region = scope if scope is not None else RawSpan(0, len(raw))
                if region.start < region.end:
                    spans.append(region)
            if sample_id in selected:
                origins.append(PublicOrigin(ref_id, sample_id, row["kind"], row["rule"], row["locus"],
                                            tuple(spans), raw_sha))
        if sample_id in selected:
            sources.append(source)
    binding = {"contract_version": ORIGIN_VERSION, "parent_sample_id": parent_sample_id,
               "origins": [asdict(origin) for origin in origins],
               "sources": [asdict(source) for source in sources], "fixed_units": len(origins)}
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()
    return PublicOriginInventory(ORIGIN_VERSION, parent_sample_id, tuple(origins), tuple(sources), digest)


def public_origins(samples: list[Sample], *, contract_version: str,
                   response_contracts: dict[str, str | None],
                   parent_sample_id: str | None = None) -> tuple[PublicOrigin, ...]:
    """Tuple view for consumers that already retain the complete inventory binding."""
    return public_origin_inventory(samples, contract_version=contract_version,
        response_contracts=response_contracts, parent_sample_id=parent_sample_id).origins
