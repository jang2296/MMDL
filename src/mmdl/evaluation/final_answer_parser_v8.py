"""V8: gold-blind extraction of one positive, unrevoked final answer.

This is a team scoring policy, not an LLM judge or the official Qwen scorer.
Only the pinned MMMU numeric ``normalize_str`` policy is reused at comparison
time.  Extraction never sees gold and never falls back to an older parser.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from mmdl.evaluation.final_answer_parser import (
    EMPTY,
    NO_PARSE,
    PARSED,
    _BOX,
    _LATEX_FRACTION,
    _choices,
    _plain,
    _question_type,
    _result,
    _snippet,
    _text_key,
)


VERSION = "team-final-answer-v8"
COMPARISON_POLICY = "single-final-mmmu-round2-v1"

_DECLARATION = re.compile(
    r"\b(?:(?:final|correct|best)\s+(?:answer|option|choice|decision)|answer)\b"
    r"(?=\s*(?:(?:is|should\s+be|would\s+be)\b|[:=]|\n|$))"
    r"\s*(?:(?:is|should\s+be|would\s+be)\b\s*)?(?:[:=]\s*)?",
    re.I,
)
_CONDITIONAL = re.compile(
    r"\b(?:if|whether|suppose|assuming|maybe|perhaps|possibly|hypothetically)\b",
    re.I,
)
_ATTRIBUTION = re.compile(
    r"\b(?:example|textbook|source|quotation|quoted?|according\s+to|"
    r"(?:someone|they|he|she|it)\s+(?:says?|states?|claims?))\b",
    re.I,
)
_CORRECTION = re.compile(
    r"\b(?:correction|corrected|revised?\s+answer|revise|instead|actually)\b",
    re.I,
)
_SELF_CORRECTION = re.compile(
    r"\b(?:made\s+an?\s+error|misread|contradicts?\s+(?:my|our)\s+reasoning|"
    r"I\s+was\s+wrong|correct\s+(?:myself|ourselves)|"
    r"previous\s+answer\s+(?:was|is)\s+(?:incorrect|wrong)|"
    r"answer\s+should\s+be)\b",
    re.I,
)
_WITHDRAW = re.compile(
    r"\b(?:I\s+)?(?:withdraw|retract|take\s+back|abandon)(?:ed|s|ing)?\b",
    re.I,
)
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RATIO = re.compile(rf"(?P<numerator>{_NUMBER})\s*/\s*(?P<denominator>{_NUMBER})")
_LABEL = re.compile(r"^(?:(?:option|choice)\s+)?\(?([A-Z])\)?(?=$|[\s.):;,\-])", re.I)
_NEGATED_THEN_CORRECT = re.compile(
    r"^(?:option\s+)?([A-Z])\s+is\s+(?:not\s+(?:the\s+)?(?:correct|right)"
    r"(?:\s+answer)?|incorrect|wrong)[.;,]\s*(?:option\s+)?([A-Z])\s+is\s+"
    r"(?:the\s+)?(?:correct|right)(?:\s+answer)?[.!]?\s*$",
    re.I,
)
_UNRESOLVED_REPLACEMENT = re.compile(
    r"\b(?:correct|right|final|revised)\s+(?:answer|choice|option)\s+"
    r"(?:may|might|could|would)\s+be\s*(?:\.{3})?\s*$",
    re.I,
)
_KNOWN_UNITS = {
    "v": "V", "volt": "V", "volts": "V",
    "a": "A", "amp": "A", "amps": "A", "ampere": "A", "amperes": "A",
    "w": "W", "watt": "W", "watts": "W",
    "n": "N", "newton": "N", "newtons": "N",
    "pa": "Pa", "pascal": "Pa", "pascals": "Pa",
    "j": "J", "joule": "J", "joules": "J",
    "hz": "Hz", "hertz": "Hz", "ohm": "ohm", "ohms": "ohm", "ω": "ohm",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "mm": "mm", "km": "km",
    "s": "s", "sec": "s", "second": "s", "seconds": "s", "ms": "ms",
    "g": "g", "kg": "kg", "mol": "mol", "k": "K",
    "°c": "°C", "celsius": "°C", "%": "%", "percent": "%",
    "unit": "unit", "units": "unit",
    "per share": "per share",
}


def _expression(value: str) -> str:
    value = _plain(value).strip().lstrip("#> ✅✔").rstrip(" ✅✔").strip()
    for _ in range(4):
        value = re.sub(r"\\[,;:!]\s*", " ", value)
        changed = value
        for opening, closing in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
            if value.startswith(opening) and value.endswith(closing) and len(value) > len(opening + closing):
                value = value[len(opening):-len(closing)].strip()
                break
        box = _BOX.match(value.rstrip(".! "))
        if box:
            suffix = value[box.end():].strip(" .!")
            if not suffix or suffix.casefold() in _KNOWN_UNITS:
                value = (box.group(1) + (" " + suffix if suffix else "")).strip()
        if value == changed:
            break
    value = re.sub(r"\\[,;:!]\s*", " ", value)
    return value.rstrip(" ✅✔").strip()


def _numeric(value: str) -> Fraction | None:
    value = value.strip().replace("−", "-")
    value = re.sub(r"^[£$€¥]\s*", "", value)
    ratio = _LATEX_FRACTION.fullmatch(value) or _RATIO.fullmatch(value)
    try:
        if ratio:
            numerator = Fraction(Decimal(ratio.group("numerator").replace(",", "")))
            denominator = Fraction(Decimal(ratio.group("denominator").replace(",", "")))
            return numerator / denominator
        if re.fullmatch(_NUMBER, value):
            return Fraction(Decimal(value.replace(",", "")))
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def _display_number(number: Fraction) -> str:
    return str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"


def _number_text(value: str, number: Fraction) -> str:
    value = value.strip()
    for unit in sorted(_KNOWN_UNITS, key=len, reverse=True):
        value = re.sub(rf"\s*{re.escape(unit)}\s*$", "", value, flags=re.I)
    value = re.sub(r"^[£$€¥]\s*", "", value).replace(",", "")
    if re.fullmatch(_NUMBER, value):
        return value.lstrip("+").lower()
    return _display_number(number)


def _quantity_parts(value: str) -> tuple[Fraction | None, str | None]:
    value = value.strip()
    if (number := _numeric(value)) is not None:
        return number, None
    units = sorted(_KNOWN_UNITS, key=len, reverse=True)
    unit_pattern = "|".join(re.escape(unit) for unit in units)
    match = re.fullmatch(rf"(.+?)\s*({unit_pattern})", value, re.I)
    if not match or (number := _numeric(match.group(1))) is None:
        return None, None
    return number, _KNOWN_UNITS[match.group(2).casefold()]


def _question_schema(question: str) -> tuple[str, str | None]:
    text = _plain(question)
    if re.search(r"\b(?:which|what)\b[^?\n]{0,50}\bstep\b|\bwhich\s+step\b", text, re.I):
        return "step", None
    if re.search(r"\b(?:which|what)\b[^?\n]{0,80}\bpoints?\b|\bwhich\s+points?\b", text, re.I):
        return "point", None
    if re.search(r"\bper\s+share\b", text, re.I):
        return "quantity", "per share"
    matches = list(re.finditer(
        r"(?:\bin\s+|\b(?:expressed|measured|given)\s+in\s+|\()"
        r"([A-Za-z°%Ω]+(?:\s+per\s+share)?)\b",
        text,
    ))
    for match in reversed(matches):
        token = match.group(1)
        if token == "a":  # An incidental article is not the ampere symbol A.
            continue
        if token.casefold() in _KNOWN_UNITS:
            return "quantity", _KNOWN_UNITS[token.casefold()]
    if re.search(r"\b(?:what|which)\b[^?\n]{0,180}\bvoltage\b|\bvoltage\b[^?\n]{0,80}\?", text, re.I):
        return "quantity", "V"
    if re.search(r"\b(?:what|which)\b[^?\n]{0,180}\bcurrent\b|\bcurrent\b[^?\n]{0,80}\?", text, re.I):
        return "quantity", "A"
    return "literal", None


def _candidate_block(tail: str) -> str:
    tail = tail.lstrip(" \t:=\r\n")
    block = re.split(r"\n\s*\n", tail, maxsplit=1)[0]
    return block.strip().lstrip("#> ").strip()


def _candidate_value(block: str) -> str:
    lines = [line.strip().lstrip("#> ✅✔") for line in block.splitlines() if line.strip()]
    if not lines:
        return ""
    pairs = {"$$": "$$", r"\[": r"\]", r"\(": r"\)"}
    if lines[0] in pairs:
        closing = pairs[lines[0]]
        try:
            end = lines[1:].index(closing) + 1
        except ValueError:
            return lines[0]
        return " ".join(lines[:end + 1])
    return lines[0]


def _is_attributed(text: str, marker: re.Match[str], tail: str) -> bool:
    line_start = text.rfind("\n", 0, marker.start()) + 1
    prefix = text[line_start:marker.start()]
    prefix = re.split(r"(?<=[.!?])\s+", prefix)[-1]
    if text[:marker.start()].count("```") % 2:
        return True
    if _ATTRIBUTION.search(prefix):
        return True
    excerpt = tail[:240]
    return bool(re.search(r"\b(?:only\s+quoting|merely\s+quoting|have\s+not\s+selected|"
                          r"do\s+not\s+(?:select|choose|endorse))\b", excerpt, re.I))


def _prefix_is_conditional(text: str, marker: re.Match[str]) -> bool:
    prefix = text[:marker.start()].rsplit("\n", 1)[-1]
    prefix = re.split(r"(?<=[.!?])\s+", prefix)[-1]
    return bool(_CONDITIONAL.search(prefix))


def _explicit_correction(prefix: str) -> bool:
    if re.search(r"^\s*#{0,6}\s*Correction\s*:?.*$", prefix, re.I | re.M):
        return True
    return bool(_SELF_CORRECTION.search(prefix) or _CORRECTION.search(prefix.rsplit("\n\n", 1)[-1]))


def _mcq_value(value: str, choices: Mapping[str, str]) -> tuple[str | None, str]:
    value = _expression(value).rstrip(".! ").strip()
    mapped = re.fullmatch(r"(.+?)\s*(?:->|→|⇒)\s*(?:option|choice)\s+([A-Z])[.!]?", value, re.I)
    if mapped:
        label = mapped.group(2).upper()
        if label not in choices:
            return None, "invalid_label"
        left = _expression(mapped.group(1))
        left_number, _ = _quantity_parts(left)
        option_number, _ = _quantity_parts(_expression(choices[label]))
        same = (_text_key(left) == _text_key(choices[label])
                or (left_number is not None and option_number is not None and left_number == option_number))
        return (label, "mapped_label_option_exact") if same else (None, "label_option_conflict")
    bare_label = _LABEL.fullmatch(value)
    if bare_label and bare_label.group(1).upper() in choices:
        return bare_label.group(1).upper(), "label"
    exact = [label for label, option in choices.items() if _text_key(value) == _text_key(option)]
    if len(exact) == 1:
        return exact[0], "option_text_exact"
    if len(exact) > 1:
        return None, "ambiguous_option_text"

    correction = re.search(
        r"\b(?:actually|correction|instead)[,:]?\s*(?:(?:option|choice)\s+)?\(?([A-Z])\)?[.!]?\s*$",
        value,
        re.I,
    )
    if correction and correction.group(1).upper() in choices:
        return correction.group(1).upper(), "explicit_correction"
    corrected = _NEGATED_THEN_CORRECT.fullmatch(value)
    if corrected and corrected.group(2).upper() in choices:
        return corrected.group(2).upper(), "explicit_correction"

    label_match = _LABEL.match(value)
    if not label_match:
        return None, "unsupported_mcq_expression"
    label = label_match.group(1).upper()
    if label not in choices:
        return None, "invalid_label"
    raw_suffix = value[label_match.end():]
    suffix = raw_suffix.strip().lstrip(".):;,–—- ").strip()

    described = [key for key, option in choices.items() if _text_key(suffix) == _text_key(option)]
    if described:
        return ((label, "label_option_exact") if described == [label]
                else (None, "label_option_conflict"))
    if re.match(r"^\s*[,/]\s*(?:(?:option|choice)\s+)?\(?[A-Z]\)?\b", raw_suffix, re.I):
        return None, "multiple_choice_alternatives"
    if re.match(r"^(?:or|and|nor|versus|vs\.?|/)\s*(?:(?:option|choice)\s+)?\(?[A-Z]\)?\b", suffix, re.I):
        return None, "multiple_choice_alternatives"
    if re.search(r"(?:^|[.!?]\s*)(?:(?:option|choice)\s+)?[A-Z]\s+is\s+"
                 r"(?:also\s+)?(?:correct|right)(?:\s+too)?\b", suffix, re.I):
        return None, "multiple_choice_alternatives"
    denial = re.search(r"\?\s*(?:no|nope)\b", suffix, re.I)
    if denial:
        questioned = suffix[:denial.start()].strip()
        if not questioned or _text_key(questioned) == _text_key(choices[label]):
            return None, "negated_answer"
    if re.match(r"^(?:is\s+)?(?:not\s+(?:the\s+)?(?:correct|right)(?:\s+answer)?|incorrect|wrong)\b", suffix, re.I):
        return None, "negated_answer"
    if re.match(r"^(?:isn['’]t|cannot\s+be|can['’]t\s+be)\s+(?:correct|right)\b", suffix, re.I):
        return None, "negated_answer"
    if re.search(rf"\b{re.escape(label)}\s+is\s+(?:not\s+(?:correct|right)|incorrect|wrong)\b", value, re.I):
        return None, "negated_answer"
    if label == "A" and suffix and not re.match(r"^[.):;,–—-]", raw_suffix):
        if not re.fullmatch(r"is\s+(?:the\s+)?(?:correct|right)(?:\s+answer)?", suffix, re.I):
            return None, "article_a"
    return label, "label"


def _conclusion(value: str) -> str | None:
    value = _expression(value)
    stage = re.fullmatch(
        r"(?:therefore|thus|hence|so)[,:]?\s+[^.!?\n]+?\s+belongs\s+to\s+(?:the\s+)?"
        r"([A-Za-z][A-Za-z -]*?)\s+(?:stage|period)[.!]?",
        value,
        re.I,
    )
    if stage:
        return stage.group(1)
    direct = re.fullmatch(r"(?:therefore|thus|hence|so)[,:]?\s+(.+)", value, re.I)
    return direct.group(1) if direct else None


def _open_value(value: str, question: str, _depth: int = 0) -> tuple[str | None, str]:
    value = _expression(value)
    if _depth >= 8:
        return None, "unsupported_nested_answer"
    conclusion = _conclusion(value)
    if conclusion is not None:
        value = conclusion
    value = value.rstrip(".! ").strip()

    corrected = re.search(r"\b(actually|correction|instead)[,:]?\s*(.+?)\s*$", value, re.I)
    if corrected:
        cue, replacement = corrected.group(1).casefold(), corrected.group(2)
        if (cue == "correction"
                or re.match(rf"(?:[£$€¥]?\s*{_NUMBER}|step\b|point\b|\\boxed)", replacement, re.I)):
            answer, rule = _open_value(replacement, question, _depth + 1)
            return answer, "explicit_correction:" + rule
    if re.search(r"\b(?:or|versus|alternatively)\b", value, re.I):
        return None, "multiple_answer_alternatives"
    if re.search(r"\b(?:(?:is|was)\s+(?:not\s+correct|incorrect|wrong)|"
                 r"isn['’]t\s+(?:correct|right)|cannot\s+be\s+(?:correct|right))\b", value, re.I):
        return None, "negated_answer"
    if re.search(r"\b(?:may|might|could)\s+be\s+(?:correct|right)\b", value, re.I):
        return None, "uncertain_answer"
    if re.search(r"\b(?:cannot|can't|unable\s+to)\s+(?:determine|identify|choose)|"
                 r"\bno\s+(?:unique|definite)\s+answer\b|\b(?:do\s+not|don't)\s+know\b", value, re.I):
        return None, "no_committed_answer"
    if re.fullmatch(r"[\[(]\s*[^,;]+\s*[,;]\s*[^,;]+\s*[\])]", value):
        return None, "unsupported_structured_answer"

    stated = re.fullmatch(r".{1,100}?\s+(?:is|equals|=)\s+(.+)", value, re.I)
    if stated:
        replacement = stated.group(1).strip()
        if replacement == value:
            return None, "unsupported_nested_answer"
        stated_answer, stated_rule = _open_value(replacement, question, _depth + 1)
        if stated_answer is not None and stated_rule in {"numeric", "quantity", "step_index", "named_point"}:
            return stated_answer, "stated_" + stated_rule
        if stated_rule == "unsupported_nested_answer":
            return None, stated_rule

    schema, expected_unit = _question_schema(question)
    if schema == "step":
        match = re.fullmatch(rf"(?:the\s+)?step\s+({_NUMBER})", value, re.I)
        if match and (number := _numeric(match.group(1))) is not None:
            return _display_number(number), "step_index"
    if schema == "point":
        match = re.match(r"(?:the\s+)?point\s+([A-Za-z0-9]+)\b", value, re.I)
        if match:
            return match.group(1), "named_point"
        if re.fullmatch(r"[A-Za-z0-9]+", value):
            return value, "named_point"

    number, unit = _quantity_parts(value)
    if number is not None:
        display = _number_text(value, number)
        if unit:
            return f"{display} {unit}", "quantity"
        return display, "numeric"

    numbers = re.findall(_NUMBER, value)
    if len(numbers) > 1:
        return None, "multiple_numeric_values"
    key = _text_key(value)
    return (key, "conclusion_literal" if conclusion is not None else "literal") if key else (None, "empty_answer")


def _later_action(tail: str, answer: str, kind: str, choices: Mapping[str, str], question: str) -> tuple[str | None, str | None]:
    """Return a supported correction, or a reason revoking the current answer."""
    corrections = list(re.finditer(
        r"(?:^|\n|[.!?]\s+)(actually|correction|instead|revised\s+answer)\s*[,!:]\s*([^\n]+)",
        tail,
        re.I,
    ))
    current, action_end = answer, 0
    if corrections:
        for correction in corrections:
            cue, value = correction.group(1).casefold(), correction.group(2)
            if (kind == "open" and cue in {"actually", "instead"}
                    and not re.match(rf"(?:[£$€¥]?\s*{_NUMBER}|step\b|point\b|\\boxed)", value, re.I)):
                continue
            parsed, _ = (_mcq_value(value, choices) if kind == "mcq" else _open_value(value, question))
            if parsed is not None:
                if _WITHDRAW.search(value):
                    return None, "withdrawn_answer"
                if _UNRESOLVED_REPLACEMENT.search(value):
                    return None, "unresolved_replacement"
                current, action_end = parsed, correction.end()
            elif cue in {"correction", "revised answer"}:
                return None, "unresolved_replacement"
    scope = tail[action_end:]
    if _WITHDRAW.search(scope):
        return None, "withdrawn_answer"
    if _UNRESOLVED_REPLACEMENT.search(scope):
        return None, "unresolved_replacement"
    if re.search(r"\b(?:correction|revised\s+answer)\s*:\s*$", scope, re.I):
        return None, "unresolved_replacement"
    subject = rf"(?:point\s+|step\s+)?{re.escape(current)}"
    if re.search(
        rf"\b{subject}\s+(?:(?:is|was)\s+(?:not\s+(?:correct|right)|incorrect|wrong)|"
        rf"isn['’]t\s+(?:correct|right)|cannot\s+be\s+(?:correct|right))\b",
        scope,
        re.I,
    ):
        return None, "negated_answer"
    if re.search(rf"\b{subject}\s+(?:may|might|could)\s+be\s+(?:correct|right)\b", scope, re.I):
        return None, "uncertain_answer"
    if kind == "mcq":
        affirmations = re.findall(r"\b([A-Z])\s+is\s+(?:also\s+)?(?:correct|right)\b", scope, re.I)
        if any(label.upper() in choices and label.upper() != current for label in affirmations):
            return None, "multiple_choice_alternatives"
    return current, None


def _same_extracted(left: str, right: str, kind: str, question: str) -> bool:
    if kind == "mcq":
        return left == right
    left_number, left_unit = _split_quantity(left)
    right_number, right_unit = _split_quantity(right)
    if left_number is not None or right_number is not None:
        _, expected_unit = _question_schema(question)
        left_unit = left_unit or expected_unit
        right_unit = right_unit or expected_unit
        return (left_number is not None and right_number is not None
                and left_unit == right_unit and left_number == right_number)
    return _text_key(left) == _text_key(right)


def _span_attributed(text: str, start: int) -> bool:
    prefix = text[:start]
    paragraph = prefix.rsplit("\n\n", 1)[-1]
    paragraph = re.split(r"(?<=[.!?])\s+", paragraph)[-1]
    return bool(prefix.count("```") % 2 or _ATTRIBUTION.search(paragraph))


def _extract(raw: Any, kind: str, choices: Mapping[str, str], finish_reason: str | None,
             question: str) -> dict[str, Any]:
    del finish_reason  # A completed visible answer survives a truncated explanation.
    if raw is None or not str(raw).strip():
        return _result(None, EMPTY, "empty", "")
    text = _plain(raw)
    markers = list(_DECLARATION.finditer(text))
    accepted: str | None = None
    conflict = False
    latest: tuple[str | None, str, str, str] | None = None

    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        tail = text[marker.end():end]
        if _prefix_is_conditional(text, marker) or _is_attributed(text, marker, tail):
            continue
        block = _candidate_block(tail)
        value = _candidate_value(block)
        parsed, rule = (_mcq_value(value, choices) if kind == "mcq" else _open_value(value, question))
        prefix = text[(markers[index - 1].end() if index else 0):marker.start()]
        correction = _explicit_correction(prefix) or rule.startswith("explicit_correction")
        if parsed is not None and accepted is not None and not _same_extracted(parsed, accepted, kind, question):
            conflict = not correction
        if correction and parsed is not None:
            conflict = False
        if parsed is not None:
            accepted = parsed
        latest = (parsed, rule, value, tail)

    if markers:
        if latest is None:
            return _result(None, NO_PARSE, "no_asserted_declaration", _snippet(text))
        parsed, rule, value, tail = latest
        if parsed is None:
            return _result(None, NO_PARSE, rule, _snippet(value))
        if conflict:
            return _result(None, NO_PARSE, "conflicting_declarations", _snippet(value))
        replacement, reason = _later_action(tail, parsed, kind, choices, question)
        if reason:
            return _result(None, NO_PARSE, reason, _snippet(tail))
        if replacement is None:
            return _result(None, NO_PARSE, "unresolved_replacement", _snippet(tail))
        if replacement != parsed:
            parsed, rule = replacement, "explicit_correction"
        return _result(parsed, PARSED, "final_declaration:" + rule, _snippet(value))

    box_matches = [match for match in _BOX.finditer(text) if not _span_attributed(text, match.start())]
    boxes = [(match.group(1), match.group(0)) for match in box_matches]
    if boxes:
        parsed_boxes = [(_mcq_value(value, choices) if kind == "mcq" else _open_value(value, question))[0]
                        for value, _ in boxes]
        valid_boxes = [value for value in parsed_boxes if value is not None]
        same = (len(valid_boxes) == len(parsed_boxes)
                and all(_same_extracted(valid_boxes[0], value, kind, question)
                        for value in valid_boxes[1:]))
        if same:
            parsed = valid_boxes[-1]
            replacement, reason = _later_action(
                text[box_matches[-1].end():], parsed, kind, choices, question)
            if reason:
                return _result(None, NO_PARSE, reason, _snippet(text[box_matches[-1].start():]))
            return _result(replacement, PARSED, "boxed_answer", _snippet(boxes[-1][1]))
        return _result(None, NO_PARSE, "conflicting_or_invalid_boxes", " | ".join(_snippet(item[1]) for item in boxes))

    lines = [line.strip().lstrip("#> ") for line in text.splitlines() if line.strip()]
    if not lines:
        return _result(None, NO_PARSE, "no_explicit_final", "")
    terminal = lines[-1]
    conclusion = _conclusion(terminal)
    value = conclusion or terminal
    parsed, rule = (_mcq_value(value, choices) if kind == "mcq" else _open_value(value, question))
    explicit = len(lines) == 1 or conclusion is not None
    if kind == "mcq":
        explicit = explicit or bool(re.fullmatch(r"\(?[A-Z]\)?[.!]?", _expression(terminal), re.I))
        explicit = explicit or rule in {"option_text_exact", "label_option_exact"}
    if not explicit or parsed is None:
        return _result(None, NO_PARSE, rule if parsed is None else "no_explicit_final", _snippet(terminal))
    return _result(parsed, PARSED, "terminal:" + rule, _snippet(terminal))


def extract_mcq(raw: Any, options: Sequence[str] | Mapping[str, str],
                finish_reason: str | None = None, *, question: str = "") -> dict[str, Any]:
    return _extract(raw, "mcq", _choices(options), finish_reason, question)


def extract_open(raw: Any, finish_reason: str | None = None, *, question: str = "") -> dict[str, Any]:
    return _extract(raw, "open", {}, finish_reason, question)


def _split_quantity(value: Any) -> tuple[Fraction | None, str | None]:
    text = _expression(str(value)).rstrip(".! ").strip()
    return _quantity_parts(text)


def _numeric_equal(left: Fraction, right: Fraction) -> bool:
    try:
        values = [float(left), float(right)]
    except OverflowError:
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    from mmdl.evaluation.parsers import _official_eval_utils
    normalize = _official_eval_utils().normalize_str
    return normalize(str(values[0])) == normalize(str(values[1]))


def _open_equal(parsed: str, gold: Any, question: str, rule: str) -> bool:
    schema, expected_unit = _question_schema(question)
    left_number, left_unit = _split_quantity(parsed)
    right_number, right_unit = _split_quantity(gold)
    if left_number is not None or right_number is not None:
        if left_number is None or right_number is None:
            return False
        if expected_unit:
            if left_unit is not None and left_unit != expected_unit:
                return False
            if right_unit is not None and right_unit != expected_unit:
                return False
        elif left_unit != right_unit:
            return False
        return _numeric_equal(left_number, right_number)
    if schema == "point" and rule.endswith("named_point"):
        return _text_key(parsed) == _text_key(str(gold))
    return _text_key(parsed) == _text_key(str(gold))


def score_response(raw: Any, question_type: str, options: Sequence[str] | Mapping[str, str],
                   answer: Any, finish_reason: str | None = None, *, question: str = "") -> dict[str, Any]:
    kind = _question_type(question_type)
    extracted = (extract_mcq(raw, options, finish_reason, question=question) if kind == "mcq"
                 else extract_open(raw, finish_reason, question=question))
    parsed = extracted["answer"]
    aliases = answer if isinstance(answer, (list, tuple, set)) else [answer]
    correct = extracted["status"] == PARSED and any(
        str(gold).strip().upper() == parsed if kind == "mcq"
        else _open_equal(parsed, gold, question, extracted["rule"])
        for gold in aliases
    )
    return {
        "parsed_answer": parsed,
        "parse_status": extracted["status"],
        "correct": bool(correct),
        "extraction_rule": extracted["rule"],
        "extraction_evidence": extracted["evidence"],
        "extraction_status": extracted["status"],
    }
