"""Gold-blind, deterministic extraction of explicit final answers.

Open answer spans are normalized with the pinned MMMU parser/evaluator. Currency
markers, thousands separators, decimal/scientific notation, rational numbers,
and simple LaTeX fractions are supported before official numeric normalization.
No team-specific tolerance, unit conversion, modulo arithmetic or expression evaluation.
"""

from __future__ import annotations

import re
import math
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


VERSION = "team-final-answer-v4"
EMPTY = "EMPTY"
NO_PARSE = "NO_PARSE"
PARSED = "PARSED"

_MCQ_TYPES = {"mcq", "multiple-choice", "multiple choice"}
_OPEN_TYPES = {"open", "open-ended", "open ended", "free-response", "free response"}
_FINAL_MARKER = re.compile(r"\bfinal\s+(?:answer|decision)\b", re.IGNORECASE)
_ANSWER_MARKER = re.compile(
    r"\b(?:(?:correct|best)\s+(?:answer|option|choice)|answer)\b"
    r"(?=\s*(?:(?:is|should\s+be|would\s+be)\b|[:=]|\n))",
    re.IGNORECASE | re.MULTILINE,
)
_CORRECT_MARKER = re.compile(
    r"\b(?:correct|best)\s+(?:answer|option|choice)\b"
    r"(?=\s*(?:(?:is|should\s+be|would\s+be)\b|[:=]|\n))", re.IGNORECASE)
_HYPOTHETICAL_LEAD = re.compile(
    r"\b(?:if|whether|suppose|assuming|maybe|perhaps)\b[^.\n]*$", re.IGNORECASE)
_BOX = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.IGNORECASE)
_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_RATIONAL = re.compile(
    rf"(?P<numerator>{_DECIMAL.pattern})\s*/\s*(?P<denominator>{_DECIMAL.pattern})"
)
_LATEX_FRACTION = re.compile(
    rf"\\(?:d?frac)\s*\{{\s*(?P<numerator>{_DECIMAL.pattern})\s*\}}"
    rf"\s*\{{\s*(?P<denominator>{_DECIMAL.pattern})\s*\}}",
    re.IGNORECASE,
)
_NONFINITE = re.compile(r"[+-]?(?:nan|inf(?:inity)?)", re.IGNORECASE)


def _result(answer: Any, status: str, rule: str, evidence: str) -> dict[str, Any]:
    return {"answer": answer, "status": status, "rule": rule, "evidence": evidence}


def _plain(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("**", "").replace("__", "").replace("`", "")
    for _ in range(4):
        cleaned = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\s*\{([^{}]*)\}", r"\1", text)
        if cleaned == text:
            break
        text = cleaned
    return text.strip()


def _snippet(value: str, limit: int = 240) -> str:
    return " ".join(value.strip().split())[:limit]


def _question_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in _MCQ_TYPES:
        return "mcq"
    if normalized in _OPEN_TYPES:
        return "open"
    raise ValueError(f"Unsupported question_type: {value!r}")


def _choices(options: Sequence[str] | Mapping[str, str]) -> dict[str, str]:
    if isinstance(options, Mapping):
        normalized = [(str(label).strip().upper(), str(text)) for label, text in options.items()]
        choices = dict(normalized)
        if len(choices) != len(normalized):
            raise ValueError("MCQ option labels must be unique letters A-Z")
    else:
        if isinstance(options, (str, bytes)):
            raise TypeError("options must be a list or mapping")
        if len(options) > 26:
            raise ValueError("MCQ supports at most 26 options")
        choices = {chr(65 + index): str(text) for index, text in enumerate(options)}
    if not choices or any(not re.fullmatch(r"[A-Z]", label) for label in choices):
        raise ValueError("MCQ option labels must be unique letters A-Z")
    return choices


def _answer_declarations(text: str, parser_id: str) -> list[re.Match[str]]:
    # V4 keeps its historical phrase priority. V5 uses response order: a later
    # explicit Answer must not be hidden by an earlier correct/best declaration.
    declarations = list(_ANSWER_MARKER.finditer(text))
    if parser_id == VERSION:
        return list(_CORRECT_MARKER.finditer(text)) or declarations
    # A hypothetical mention is not a declaration; remove it before selecting
    # the last answer, including declarations inside a dedicated final section.
    qualified = []
    for marker in declarations:
        prefix = text[:marker.start()]
        if parser_id == "team-final-answer-v5":
            prefix = prefix[-35:]  # Preserve the saved V5 candidate's context window.
        else:
            prefix = prefix.rsplit("\n", 1)[-1]
        # V6 checks the full current line; the regex stops at the previous period.
        # Long conditional clauses must not turn into asserted answers by truncation.
        if not _HYPOTHETICAL_LEAD.search(prefix):
            qualified.append(marker)
    return qualified


def _asserted_spans(text: str, parser_id: str = VERSION) -> list[tuple[str | None, str]]:
    """Explicit final sections outrank earlier answer assertions and rationale mentions."""
    spans: list[tuple[str | None, str]] = []
    finals = list(_FINAL_MARKER.finditer(text))
    declarations = _answer_declarations(text, parser_id)
    # Without a dedicated final section, use the last explicit declaration, not all
    # intermediate assertions. An invalid last declaration still cannot fall back.
    markers = finals or declarations[-1:]
    for index, marker in enumerate(markers):
        lead = text[max(0, marker.start() - 35) : marker.start()]
        if parser_id == VERSION and not finals and _HYPOTHETICAL_LEAD.search(lead):
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        tail = text[marker.end():end]
        if finals:
            # A final section can state a value, then explicitly map it to an option.
            # Only a syntactic declaration inside this section may supersede its first line.
            declarations = _answer_declarations(tail, parser_id)
            if declarations:
                declaration = declarations[-1]
                tail = tail[declaration.end():]
        tail = re.sub(
            r"^\s*(?:(?:is|should\s+be|would\s+be)\b)?\s*(?::|=|[-\u2013\u2014])?\s*",
            "",
            tail,
            flags=re.I,
        )
        if re.match(r"(?i)^not\b", tail):
            continue
        tail = re.sub(r"^(?:(?:\s+)|[>#✅✔]|[$]|\\[\[(])+", "", tail)
        line = tail.splitlines()[0].strip() if tail.splitlines() else ""
        evidence = _snippet(text[marker.start() : marker.end()] + " " + line)
        spans.append((line or None, evidence))
    return spans


def _boxed_values(text: str) -> list[tuple[str, str]]:
    return [(match.group(1).strip(), _snippet(match.group(0))) for match in _BOX.finditer(text)]


def _strip_tex_text(value: str) -> str:
    match = re.fullmatch(r"\\(?:text|mathrm)\s*\{\s*(.*?)\s*\}", value, re.I | re.S)
    return match.group(1).strip() if match else value.strip()


def _text_key(value: str) -> str:
    value = _plain(value)
    if len(value) >= 2 and value[0] == value[-1] == "$":
        value = value[1:-1]
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(".?!;:").strip().casefold()


def _mcq_candidate(value: str, choices: Mapping[str, str]) -> tuple[str | None, str]:
    value = _plain(value)
    box = _BOX.match(value)
    if box:
        value = _strip_tex_text(box.group(1))
    match = re.match(r"^(?:option|choice)?\s*[\[(]?\s*([A-Z])\s*[\])]?"
                     r"(?=$|[.\s:;,-])", value, re.I)
    if match:
        label = match.group(1).upper()
        if label == "A":
            suffix = value[match.end(1) :]
            punctuated = bool(re.match(r"^\s*[\]).:;,–—-]", suffix))
            exact_description = _text_key(suffix) == _text_key(choices.get(label, ""))
            if suffix.strip() and not punctuated and not exact_description:
                return None, "article_a"
        return (label, "label") if label in choices else (None, "invalid_label")
    matches = [label for label, option in choices.items() if _text_key(value) == _text_key(option)]
    if len(matches) == 1:
        return matches[0], "option_text_exact"
    return None, "ambiguous_option_text" if matches else "no_candidate"


def _same_answers(values: Sequence[str | None]) -> bool:
    return bool(values) and len(set(values)) == 1


def extract_mcq(
    raw: Any,
    options: Sequence[str] | Mapping[str, str],
    finish_reason: str | None = None,
    *,
    parser_id: str = VERSION,
) -> dict[str, Any]:
    """Extract one explicit MCQ answer without seeing the question or gold answer."""
    if parser_id not in {VERSION, "team-final-answer-v5", "team-final-answer-v6"}:
        raise ValueError(f"Unknown final-answer scoring version: {parser_id}")
    if raw is None or not str(raw).strip():
        return _result(None, EMPTY, "empty", "")
    choices = _choices(options)
    text = _plain(raw)

    assertions = _asserted_spans(text, parser_id)
    if assertions:
        candidates = [_mcq_candidate(span, choices)[0] if span is not None else None
                      for span, _ in assertions]
        evidence = " | ".join(item[1] for item in assertions)
        if candidates[-1] is None:
            rule = ("length_without_complete_final"
                    if str(finish_reason).lower() == "length" else "invalid_final")
            return _result(None, NO_PARSE, rule, evidence)
        if not _same_answers([candidate for candidate in candidates if candidate is not None]):
            return _result(None, NO_PARSE, "conflicting_final", evidence)
        return _result(candidates[-1], PARSED, "final_assertion", assertions[-1][1])

    boxes = _boxed_values(text)
    if boxes:
        candidates = [_mcq_candidate(value, choices)[0] for value, _ in boxes]
        evidence = " | ".join(item[1] for item in boxes)
        if any(candidate is None for candidate in candidates):
            return _result(None, NO_PARSE, "invalid_box", evidence)
        if not _same_answers(candidates):
            return _result(None, NO_PARSE, "conflicting_boxed_answers", evidence)
        return _result(candidates[-1], PARSED, "boxed_answer", boxes[-1][1])

    # A terminal answer-only line is explicit; do not search earlier rationale lines.
    text = text.splitlines()[-1].lstrip("# >").strip()
    direct = re.fullmatch(r"\s*[\[(]?\s*([A-Z])\s*[\])]?[.\s]*", text, re.I)
    if direct:
        label = direct.group(1).upper()
        if label in choices:
            return _result(label, PARSED, "answer_only_label", _snippet(text))
        return _result(None, NO_PARSE, "invalid_answer_only_label", _snippet(text))

    labelled = re.fullmatch(r"\s*([A-Z])\s*[.):;-]\s*(.+?)\s*", text, re.I | re.S)
    if labelled:
        label = labelled.group(1).upper()
        if label not in choices:
            return _result(None, NO_PARSE, "invalid_label_option", _snippet(text))
        if _text_key(labelled.group(2)) != _text_key(choices[label]):
            return _result(None, NO_PARSE, "label_option_mismatch", _snippet(text))
        return _result(label, PARSED, "label_option_exact", _snippet(text))

    exact = [label for label, option in choices.items() if _text_key(text) == _text_key(option)]
    if len(exact) == 1:
        return _result(exact[0], PARSED, "answer_only_option_text", _snippet(text))
    rule = ("length_without_complete_final"
            if str(finish_reason).lower() == "length" else "no_explicit_final")
    return _result(None, NO_PARSE, rule, "")


def _unwrap_open(value: str) -> str:
    value = _plain(value).strip()
    box = _BOX.fullmatch(value)
    if box:
        value = box.group(1).strip()
    value = _strip_tex_text(value)
    if len(value) >= 2 and value[0] == value[-1] == "$":
        value = value[1:-1].strip()
    return value.rstrip(".?!;:").strip()


def _fraction_from_decimal(value: str) -> Fraction:
    return Fraction(Decimal(value))


def _normalize_open(value: Any) -> tuple[str | None, Fraction | None, bool]:
    text = _unwrap_open(str(value)).replace("\u2212", "-")
    numeric = re.sub(r"^[\$\u00a3\u20ac\u00a5]\s*", "", text).replace(",", "").strip()
    if _NONFINITE.fullmatch(numeric):
        return None, None, False
    match = _LATEX_FRACTION.fullmatch(numeric) or _RATIONAL.fullmatch(numeric)
    if match:
        try:
            denominator = _fraction_from_decimal(match.group("denominator"))
            if denominator == 0:
                return None, None, False
            number = _fraction_from_decimal(match.group("numerator")) / denominator
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None, None, False
        return f"{number.numerator}/{number.denominator}", number, True
    if _DECIMAL.fullmatch(numeric):
        try:
            number = _fraction_from_decimal(numeric)
        except (InvalidOperation, ValueError):
            return None, None, False
        display = str(number.numerator) if number.denominator == 1 else numeric.lower()
        return display, number, True
    return _text_key(text), None, bool(_text_key(text))


def _open_equal(left: Any, right: Any) -> bool:
    left_text, left_number, left_valid = _normalize_open(left)
    right_text, right_number, right_valid = _normalize_open(right)
    if not left_valid or not right_valid:
        return False
    if left_number is not None or right_number is not None:
        return left_number is not None and right_number is not None and left_number == right_number
    return left_text == right_text


def extract_open(raw: Any, finish_reason: str | None = None, *,
                 parser_id: str = VERSION) -> dict[str, Any]:
    """Extract one terminal open answer with no semantic or tolerance-based guessing."""
    if parser_id not in {VERSION, "team-final-answer-v5", "team-final-answer-v6"}:
        raise ValueError(f"Unknown final-answer scoring version: {parser_id}")
    if raw is None or not str(raw).strip():
        return _result(None, EMPTY, "empty", "")
    text = _plain(raw)
    assertions = _asserted_spans(text, parser_id)
    if assertions:
        normalized = [_normalize_open(span) if span is not None else (None, None, False)
                      for span, _ in assertions]
        evidence = " | ".join(item[1] for item in assertions)
        if any(not item[2] for item in normalized):
            rule = ("length_without_complete_final"
                    if str(finish_reason).lower() == "length" else "invalid_final")
            return _result(None, NO_PARSE, rule, evidence)
        answers = [item[0] for item in normalized]
        if any(not _open_equal(answers[0], answer) for answer in answers[1:]):
            return _result(None, NO_PARSE, "conflicting_final", evidence)
        return _result(answers[-1], PARSED, "final_assertion", assertions[-1][1])

    boxes = _boxed_values(text)
    if boxes:
        normalized = [_normalize_open(value) for value, _ in boxes]
        evidence = " | ".join(item[1] for item in boxes)
        if any(not item[2] for item in normalized):
            return _result(None, NO_PARSE, "invalid_box", evidence)
        answers = [item[0] for item in normalized]
        if any(not _open_equal(answers[0], answer) for answer in answers[1:]):
            return _result(None, NO_PARSE, "conflicting_boxed_answers", evidence)
        return _result(answers[-1], PARSED, "boxed_answer", boxes[-1][1])

    if len(text.splitlines()) == 1 and len(text.split()) <= 12:
        answer, _, valid = _normalize_open(text)
        if valid:
            return _result(answer, PARSED, "answer_only", _snippet(text))
        return _result(None, NO_PARSE, "invalid_answer_only", _snippet(text))
    rule = ("length_without_complete_final"
            if str(finish_reason).lower() == "length" else "no_explicit_final")
    return _result(None, NO_PARSE, rule, "")


def _mcq_correct(gold: Any, parsed: str) -> bool:
    values = gold if isinstance(gold, (list, tuple, set)) else [gold]
    return any(str(value).strip().upper() == parsed for value in values)


def _open_correct(gold: Any, parsed: str) -> bool:
    from mmdl.evaluation.parsers import _official_eval_utils

    official = _official_eval_utils()
    _, number, valid = _normalize_open(parsed)
    if not valid:
        return False
    if number is not None:
        try:
            numeric = float(number)
        except OverflowError:
            return False
        if not math.isfinite(numeric):
            return False
        candidates = official.normalize_str(parsed) + official.normalize_str(str(numeric))
    else:
        candidates = official.parse_open_response(parsed)
    # Only the extracted final span is visible to the official open scorer.
    return bool(official.eval_open(gold, candidates))


def score_response(
    raw: Any,
    question_type: str,
    options: Sequence[str] | Mapping[str, str],
    answer: Any,
    finish_reason: str | None = None,
    *,
    parser_id: str = VERSION,
) -> dict[str, Any]:
    """Extract first, then compare with gold; gold never influences extraction."""
    kind = _question_type(question_type)
    extracted = (extract_mcq(raw, options, finish_reason, parser_id=parser_id) if kind == "mcq"
                 else extract_open(raw, finish_reason, parser_id=parser_id))
    parsed = extracted["answer"]
    correct = extracted["status"] == PARSED and (
        _mcq_correct(answer, parsed) if kind == "mcq" else _open_correct(answer, parsed)
    )
    return {
        "parsed_answer": parsed,
        "parse_status": extracted["status"],
        "correct": bool(correct),
        "extraction_rule": extracted["rule"],
        "extraction_evidence": extracted["evidence"],
        "extraction_status": extracted["status"],
    }
