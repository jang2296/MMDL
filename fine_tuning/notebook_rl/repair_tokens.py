"""Original revised-patch spans and safe token ownership; no scores or loss wiring."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .repair_contract import (
    MAX_RAW_CHARS, RawSpan, RepairContractError, RetrySections, _elements, _members, _value_spans,
    project_solution,
)


def emitted_patch_spans(raw: str, sections: RetrySections, *, kind: str, rule: str,
                        locus: str) -> tuple[RawSpan, ...] | None:
    """Locate authored values only; the caller must independently verify their semantics."""
    solution = project_solution(raw, sections)
    offset = sections.solution_span.start
    if kind == "OBSERVATION":
        from .notebook_synthesis import _native_value_shape, _single_line_scalar
        from .observations import _LINE, observation_spans, slot_id

        if type(locus) is not str or rule != "OBSERVATION_MATCH" or slot_id(locus) != locus:
            return None
        occurrences = [(start, end, value) for start, end, key, value in observation_spans(solution) if key == locus]
        if len(occurrences) != 1:
            return None
        start, end, value = occurrences[0]
        line = solution[start:end].rstrip("\r\n")
        match = _LINE.fullmatch(line)
        if (match is None or not _single_line_scalar(line) or not _native_value_shape(locus, value)
                or match[2] != value):
            return None
        return (RawSpan(offset + start + match.start(2), offset + start + match.end(2)),)
    if kind != "TRACE":
        return None
    from .reasoning_trace import _trace
    from .repair_credit import operation_patch

    try:
        trace, evidence = _trace(solution)
        patch = operation_patch(trace, rule, locus)
        if patch is None:
            return None
        match = re.fullmatch(r"[ \t]*Trace:[ \t]*(?P<payload>\{[^\r\n]*\})[ \t]*", evidence["line"])
        if match is None:
            return None
        start = offset + evidence["span"][0] + match.start("payload")
        end = offset + evidence["span"][0] + match.end("payload")
        members = _members(raw, start)
        paths = _value_spans(raw, trace, RawSpan(start, end), "")
        field = patch["field"]
        spans: tuple[RawSpan, ...]
        if locus == field or trace["kind"] == "circuit_ladder" and field in ("input", "query"):
            values, spans = members[field][0], (paths["/" + field],)
        else:
            pair = re.fullmatch(re.escape(field) + r"\[([0-9]+):([0-9]+)\]", locus)
            item = re.fullmatch(re.escape(field) + r"\[([0-9]+)\](?:\.(?:node|new))?", locus)
            elements = _elements(raw, members[field][1].start)
            if pair is not None:
                selected = elements[int(pair[1]):int(pair[2])]
                values = [value for value, _ in selected]
                spans = tuple(span for _, span in selected)
            elif item is not None:
                values, span = elements[int(item[1])]
                spans = (span,)
            else:
                return None
        if not spans or json.dumps(values, sort_keys=True, allow_nan=False) != json.dumps(
                patch["value"], sort_keys=True, allow_nan=False):
            return None
        return spans
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RecursionError):
        return None


def span_token_ownership(raw: str, ids: list[int], tokenizer: Any,
                         spans: tuple[RawSpan, ...]) -> dict:
    """Union one unit's original spans; crossing and zero-width tokens remain unowned."""
    if type(raw) is not str:
        raise RepairContractError("RAW_TYPE")
    if len(raw) > MAX_RAW_CHARS:
        raise RepairContractError("RAW_TOO_LARGE")
    try:
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        raise RepairContractError("INVALID_UTF8") from None
    if type(ids) is not list or any(type(token) is not int or token < 0 for token in ids):
        raise RepairContractError("TOKEN_IDS")
    if type(spans) is not tuple or any(type(span) is not RawSpan or type(span.start) is not int
            or type(span.end) is not int or not 0 <= span.start <= span.end <= len(raw) for span in spans):
        raise RepairContractError("INVALID_FIELD", at="spans")
    regions: list[list[int]] = []
    for start, end in sorted((span.start, span.end) for span in spans if span.start < span.end):
        if regions and start <= regions[-1][1]:
            regions[-1][1] = max(end, regions[-1][1])
        else:
            regions.append([start, end])
    from .fact_credit import token_spans

    original_ids = list(ids)
    token_ids = list(ids)

    class BoundDecoder:
        def decode(self, prefix: list[int], **kwargs) -> str:
            if type(prefix) is not list or any(type(token) is not int for token in prefix):
                raise RepairContractError("TOKEN_IDS")
            before = tuple(prefix)
            decoded = tokenizer.decode(prefix, **kwargs)
            if any(type(token) is not int for token in prefix) or tuple(prefix) != before:
                raise RepairContractError("TOKEN_IDS")
            if type(decoded) is not str or before == tuple(original_ids) and decoded != raw:
                raise RepairContractError("TOKEN_TEXT_MISMATCH")
            return decoded

    decoder = BoundDecoder()
    try:
        tokens = token_spans(decoder, token_ids, raw)
        decoder.decode(list(original_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except RepairContractError:
        raise
    except ValueError:
        raise RepairContractError("TOKEN_TEXT_MISMATCH") from None
    if token_ids != original_ids:
        raise RepairContractError("TOKEN_IDS")
    if tokens is not None and (tokens[-1][1] if tokens else 0) != len(raw):
        raise RepairContractError("TOKEN_TEXT_MISMATCH")
    indices, crossing = [], []
    if tokens is not None:
        for index, (start, end) in enumerate(tokens):
            if start == end:
                continue
            if any(a <= start < end <= b for a, b in regions):
                indices.append(index)
            elif any(start < b and end > a for a, b in regions):
                crossing.append(index)
    status = ("NO_SPANS" if not regions else "UNSTABLE_TOKEN_PREFIX" if tokens is None else
              "NO_OWNED_TOKENS" if not indices else "MAPPED_WITH_EXCLUDED_BOUNDARIES" if crossing else "MAPPED")
    return {"status": status, "raw_sha256": raw_sha,
            "token_ids_sha256": hashlib.sha256(json.dumps(original_ids, separators=(",", ":")).encode()).hexdigest(),
            "spans": regions, "token_indices": indices, "excluded_crossing_token_indices": crossing}
