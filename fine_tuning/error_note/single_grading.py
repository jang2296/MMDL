"""SINGLE v8 answer grading: V8 extraction, one deterministic notation normalization, the training comparison.

MCQ is graded exactly as before (V8 contract p0-eval-aligned-v1). An open answer is graded in three steps:

1. Read one committed answer from the whole response (``_committed``):
   - the V8 open extractor (notebook_rl.v8_parser.extract_open, used read-only); PARSED gives the answer;
   - V8 "A or B" alternatives are INVALID_MULTIPLE; a V8 revoked/negated/uncertain declaration is INVALID_NO_FINAL;
   - when V8 rejects declarations (or boxes) only because they disagree or hold several numbers, its own declared
     values are normalized: exactly one distinct value is one answer in several notations ("$\\boxed{61^\\circ}$" and
     "\\boxed{61^\\circ}", "y = \u221a3" and "\\boxed{\\sqrt{3}}"); several distinct values are INVALID_MULTIPLE;
   - otherwise the training extractor (training-answer-v2) on the whole response.
   Nothing is chosen among several answers: disagreement is INVALID_MULTIPLE.
2. Normalize it (NORMALIZATION_RULES; ``normalize_final`` is the bare-number result of those rules).
3. Compare through the verifier's existing typed training comparison by grading "Final answer: <form>" with
   training-answer-v2: first the value with its suffix, then the bare number only when that is not CORRECT, then
   a radical's decimal value only when both were uncomparable; an answer still undecided keeps V8's INCORRECT
   (the v7/MMMU-evaluation result) when V8 grades the whole response INCORRECT. A suffixed answer can match only a gold with the
   same suffix and the bare number only a plain-number gold, so a scale word is dropped only against a gold without
   a scale ("31.00 thousand" matches 31, never "31 million"; "31 thousand" against 31000 stays INCORRECT: no
   multiplication). Ambiguous scale/format ("75" vs 0.75, "X-Ray" vs "chest X-ray") is never loosened.

``extracted`` is the committed value the grader read (None when no single committed answer exists), so a
committed answer is never treated as missing (reward gate v5).

Scorer v2 (``units=True``, SCORER_V2) is scorer v1 plus NORMALIZATION_RULES_V2: a boxed number followed by a length/area
unit ("\\boxed{855} yd^2", "8.6 m") also yields the bare number, still only compared against a plain-number gold.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from fractions import Fraction
from typing import Any

SCORER = "v8-extract+normalize+training-compare-v1"
# Per-source material synonyms (CLEVR's own synonym table); applied to the whole normalized answer only.
ALIASES = {"hf-clevr": {"metallic": "metal", "matte": "rubber"}}
ALIASES_SHA256 = hashlib.sha256(json.dumps(ALIASES, sort_keys=True).encode()).hexdigest()
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")
# Words that change the value or join several values: never stripped as a unit.
SUFFIX_STOP = frozenset((
    "and", "or", "to", "than", "more", "less", "fewer", "greater", "larger", "smaller", "higher", "lower", "times",
    "plus", "minus", "over", "divided", "squared", "cubed", "root", "roots", "sqrt", "pi", "of", "in", "on", "at",
    "the", "an", "by", "not", "no", "dozen", "dozens", "half", "halves", "third", "thirds", "quarter", "quarters",
    "fourth", "fourths", "fifth", "fifths", "tenth", "tenths", "hundredth", "hundredths", "thousandth",
    "thousandths", "percentile", "percentiles"))
NORMALIZATION_RULES = (
    "NFKC; drop markdown ** __ `; \\text/\\mathrm/\\textbf/\\mathbf{X} -> X; LaTeX spaces -> space; \\% -> %",
    "strip whole wrappers $$..$$ $..$ \\[..\\] \\(..\\) \\boxed{..}, leading #/>/check marks, trailing . !",
    "^\\circ, ^{\\circ}, \\circ, a degree sign or 'degree(s)' after a number -> one degree sign",
    "\\sqrt{N} -> \u221aN; thousands separators removed; en/em dash or spaced hyphen between numbers -> -",
    "'therefore/thus/hence/so' prefix dropped; 'X is|equals|= Y' -> Y when Y is a number or radical (V8 stated rule)",
    "space before % removed; 'percent' after a number -> %",
    "spelled-out zero..twenty -> digits when it is the whole answer or followed only by unit words",
    "bare number: a number followed only by %, a degree sign and/or 1-3 unit words (letters, length >= 2, none in "
    "SUFFIX_STOP), e.g. unit/units/parts/years/euros/thousand/million/billion -> the number",
    "per-source alias table (ALIASES) on the whole lowercased answer",
    "compare order: suffixed form, then bare number if not CORRECT, then radical decimal (12 significant digits) "
    "if both were UNVERIFIABLE; the first CORRECT wins, else the most decisive status",
    "a committed answer the training comparison cannot decide is INCORRECT when the V8 contract on the whole "
    "response grades it INCORRECT (as MMMU evaluation scores it), else keeps its status",
)
# Scorer v2 (SINGLE v14, ``units=True``): scorer v1 plus two rules for a number that carries a unit. v1 graded
# "\boxed{855} yd^2", "\boxed{25.1} in", "8.6 m" and "\boxed{1.5} m^2" as not correct against the plain-number gold.
# Like v1's unit words, a unit is only ever dropped against a plain-number gold; nothing is multiplied or converted.
SCORER_V2 = "v8-extract+normalize+training-compare-v2"
UNIT_ABBREVIATIONS = ("mm", "cm", "dm", "m", "km", "in", "ft", "yd", "mi")
NORMALIZATION_RULES_V2 = NORMALIZATION_RULES + (
    "v2: a boxed number followed by unit text, optionally inside $ $ ('\\boxed{855} yd^2', '$ \\boxed{21} $ square "
    "feet') -> the number followed by that text ('855 yd^2', '21 square feet'); the box content must be one number",
    "v2 bare number: a unit word may carry an exponent 2 or 3 ('^2', '^{3}', or the digit left by NFKC from a "
    "superscript: 'm2'), and a suffix that is exactly one UNIT_ABBREVIATIONS token (with or without exponent) is a "
    "unit even when it has one letter or is in SUFFIX_STOP ('8.6 m', '25.1 in', '855 yd^2' -> the number)",
)
_NUM = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
# Scorer v2: "$ \boxed{N} $ unit text" with N one number (thousands separators allowed); the unit text has no further box.
_BOXED_UNIT = re.compile(r"[\s$]*\\boxed\{\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)\s*\}[\s$]*"
                         r"((?![\s\S]*\\boxed)[^\s$][^$]*?)[\s$]*", re.S)
# Scorer v2 unit token: letters, optionally with an exponent 2 or 3 ("^2", "^{3}", or the digit NFKC leaves from "m²").
_UNIT_TOKEN = r"[A-Za-z]+(?:\s*\^\s*\{?\s*[23]\s*\}?|[23])?"
_V8_ALTERNATIVES = frozenset(("multiple_answer_alternatives", "multiple_choice_alternatives"))
_V8_REVOKED = frozenset(("negated_answer", "withdrawn_answer", "uncertain_answer", "no_committed_answer",
                         "unresolved_replacement", "no_asserted_declaration"))
_V8_DISAGREE = frozenset(("conflicting_declarations", "conflicting_or_invalid_boxes", "multiple_numeric_values"))
_RANK = {"CORRECT": 0, "INCORRECT": 1, "UNVERIFIABLE": 2}


def _canonical(value: str, units: bool = False) -> str:
    """NORMALIZATION_RULES except the bare-number step: the answer with its unit/scale/percent/degree suffix.
    ``units`` (scorer v2): also the first NORMALIZATION_RULES_V2 rule (a boxed number followed by unit text)."""
    from fine_tuning.notebook_rl.v8_parser import _BOX

    text = unicodedata.normalize("NFKC", value).replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\\[,;:! ]", " ", text).replace("\\%", "%")
    for _ in range(4):
        text = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\s*\{([^{}]*)\}", r"\1", text)
    boxed = _BOXED_UNIT.fullmatch(text) if units else None
    if boxed:
        text = f"{boxed[1]} {boxed[2]}"
    for _ in range(8):
        before = text
        text = re.sub(r"^[\s#>\u2705\u2714\u2611]+|[\s.!\u2705\u2714]+$", "", text)
        for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
            if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
                text = text[len(left):-len(right)]
                break
        box = _BOX.fullmatch(text.strip())
        if box:
            text = box.group(1)
        if text == before:
            break
    text = re.sub(r"\s*\^\s*\{?\s*\\circ\s*\}?|\s*\\circ\b|\s*\u00b0", "\u00b0", text)
    text = re.sub(r"(?<=\d)\s*degrees?\b", "\u00b0", text, flags=re.I)
    text = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}|\\sqrt\s*(\d+(?:\.\d+)?)", lambda m: "\u221a" + (m[1] or m[2]), text)
    text = re.sub(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?![\d,])", lambda m: m[1].replace(",", ""), text)
    text = re.sub(r"(?<=[\d%])\s*[\u2013\u2014]\s*(?=[+-]?\d)|(?<=[\d%])\s+-\s+(?=\d)", "-", text)
    text = text.replace("\u2212", "-")
    text = " ".join(text.split())
    text = re.sub(r"^(?:therefore|thus|hence|so)[,:]?\s+", "", text, flags=re.I)
    stated = re.fullmatch(r".{1,100}?\s+(?:is|equals|=)\s+(.+)", text, re.I) or re.fullmatch(r"[A-Za-z]\s*=\s*(.+)", text)
    if stated and (_bare(stated[1]) is not None or _radical(stated[1]) is not None):
        text = stated[1]
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    text = re.sub(r"(?<=\d)\s*per\s*cent\b|(?<=\d)\s*percent\b", "%", text, flags=re.I)
    word = re.fullmatch(rf"({'|'.join(NUMBER_WORDS)})\b(.*)", text, re.I)
    if word and (not word[2].strip() or _bare("0 " + word[2].strip()) is not None):
        text = str(NUMBER_WORDS.index(word[1].casefold())) + word[2]
    return text.strip()


def _bare(text: str, units: bool = False) -> str | None:
    """The number of 'number [%|degree] [1-3 unit words]', else None (ranges, expressions, comparisons stay whole).
    ``units`` (scorer v2): also the second NORMALIZATION_RULES_V2 rule (exponents and UNIT_ABBREVIATIONS)."""
    if units:
        match = re.fullmatch(rf"[£$€¥]?\s*({_NUM})\s*([%°]?)\s*((?:{_UNIT_TOKEN})(?:\s+{_UNIT_TOKEN}){{0,2}})?",
                             text.strip())
        if match is None:
            return None
        words = re.findall(r"[A-Za-z]+", match[3] or "")  # one letter group per token (exponents hold no letters)
        if len(words) == 1 and words[0].casefold() in UNIT_ABBREVIATIONS:
            return match[1]
        return None if any(len(word) < 2 or word.casefold() in SUFFIX_STOP for word in words) else match[1]
    match = re.fullmatch(rf"[\u00a3$\u20ac\u00a5]?\s*({_NUM})\s*([%\u00b0]?)\s*([A-Za-z]{{2,}}(?:\s+[A-Za-z]{{2,}}){{0,2}})?",
                         text.strip())
    if match is None or any(word.casefold() in SUFFIX_STOP for word in (match[3] or "").split()):
        return None
    return match[1]


def _radical(text: str) -> str | None:
    """'a\u221ab' as a decimal string (12 significant digits), else None."""
    match = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)?\s*[*\u00d7\u00b7]?\s*\u221a\s*\(?(\d+(?:\.\d+)?)\)?", text.strip())
    if match is None:
        return None
    value = (-1 if match[1] == "-" else 1) * float(match[2] or 1) * math.sqrt(float(match[3]))
    return f"{value:.12g}"


def normalize_final(value: str) -> str:
    """Pure, deterministic: the NORMALIZATION_RULES result, the bare number when the answer is a number with a
    unit/scale/percent/degree suffix ("2,399,999 units" -> "2399999", "28.5^\\circ" -> "28.5", "three" -> "3")."""
    text = _canonical(value)
    bare = _bare(text)
    return text if bare is None else bare


def _declared(text: str, rule: str) -> tuple[list[str], str]:
    """V8's own unconditional, unattributed declared values (or boxes) and the text after the last one."""
    from fine_tuning.notebook_rl import v8_parser as v8

    plain = v8._plain(text)
    if rule == "conflicting_or_invalid_boxes":
        boxes = [m for m in v8._BOX.finditer(plain) if not v8._span_attributed(plain, m.start())]
        return [m.group(1) for m in boxes], plain[boxes[-1].end():] if boxes else ""
    markers = list(v8._DECLARATION.finditer(plain))
    values, tail = [], ""
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(plain)
        span = plain[marker.end():end]
        if not (v8._prefix_is_conditional(plain, marker) or v8._is_attributed(plain, marker, span)):
            values.append(v8._candidate_value(v8._candidate_block(span)))
            tail = span
    return values, tail


def _committed(text: str, finish_reason: str, question: str) -> tuple[str | None, str]:
    """(value, "OK") for one committed answer, else (None, "INVALID_MULTIPLE" | "INVALID_NO_FINAL")."""
    from fine_tuning.notebook_rl import v8_parser as v8
    from fine_tuning.notebook_rl.answers import extract_training

    found = v8.extract_open(text, finish_reason, question=question)
    if found["status"] == v8.PARSED:
        return found["answer"], "OK"
    if found["rule"] in _V8_ALTERNATIVES:
        return None, "INVALID_MULTIPLE"
    if found["rule"] in _V8_REVOKED:
        return None, "INVALID_NO_FINAL"
    if found["rule"] in _V8_DISAGREE:
        values, tail = _declared(text, found["rule"])
        keys = {_canonical(value).casefold() for value in values}
        if len(keys) == 1 and "" not in keys:
            current, reason = v8._later_action(tail, values[-1], "open", {}, question)
            if reason is not None:
                return None, "INVALID_NO_FINAL"
            return (values[-1], "OK") if current == values[-1] else (None, "INVALID_MULTIPLE")
        if len(keys) > 1 and found["rule"] != "multiple_numeric_values":
            return None, "INVALID_MULTIPLE"
    fallback = extract_training(text, finish_reason, question, (), "open")
    if fallback.status == "OK":
        return fallback.answer, "OK"
    return None, "INVALID_MULTIPLE" if fallback.status == "INVALID_MULTIPLE" else "INVALID_NO_FINAL"


def _scale_ambiguous(verifier: Any, pid: str, mode: str, bare: str) -> bool:
    """A bare number that matches the gold only after x100 or /100 (percent vs fraction) is ambiguous, not wrong."""
    from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT

    try:
        number = Fraction(bare)
    except (ValueError, ZeroDivisionError):
        return False
    return any(verifier.grade(pid, "Final answer: " + format(float(number * factor), "g"), "stop", mode,
                              answer_contract=TRAINING_ANSWER_CONTRACT).status == "CORRECT"
               for factor in (Fraction(100), Fraction(1, 100)))


def grade_single(verifier: Any, actor: Any, pid: str, text: str, finish_reason: str, mode: str, *,
                 units: bool = False) -> tuple[str, Any]:
    """One response, one committed answer, one status; see the module docstring. ``units``: scorer v2 (SCORER_V2)."""
    from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT, TURN1_EVAL_CONTRACT

    if actor.choices:
        result = verifier.grade(pid, text, finish_reason, mode, answer_contract=TURN1_EVAL_CONTRACT)
        return result.status, result.extracted
    if finish_reason == "length":
        return "TRUNCATED", None
    value, status = _committed(text, finish_reason, actor.question)
    if value is None:
        return status, None
    canonical = _canonical(value, units)
    canonical = ALIASES.get(actor.source, {}).get(canonical.casefold(), canonical)
    if not canonical:
        return "INVALID_NO_FINAL", None
    forms = [canonical] + [form for form in (_bare(canonical, units),) if form is not None and form != canonical]
    results = []
    for form in forms:
        results.append(verifier.grade(pid, "Final answer: " + form, "stop", mode, answer_contract=TRAINING_ANSWER_CONTRACT))
        if results[-1].status == "CORRECT":
            break
    radical = _radical(forms[-1])
    if radical is not None and all(result.status == "UNVERIFIABLE" for result in results):
        results.append(verifier.grade(pid, "Final answer: " + radical, "stop", mode,
                                      answer_contract=TRAINING_ANSWER_CONTRACT))
    chosen = min(results, key=lambda result: _RANK.get(result.status, 3))
    if chosen.status == "INCORRECT" and _scale_ambiguous(verifier, pid, mode, forms[-1]):
        return "UNVERIFIABLE", canonical  # e.g. '75' for gold 0.75: excluded, never made CORRECT
    if chosen.status not in ("CORRECT", "INCORRECT"):
        evaluation = verifier.grade(pid, text, finish_reason, mode, answer_contract=TURN1_EVAL_CONTRACT)
        if evaluation.status == "INCORRECT":
            return "INCORRECT", canonical
    return chosen.status, chosen.extracted if chosen.extracted is not None else canonical
