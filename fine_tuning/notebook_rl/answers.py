"""Strict RL answer contract; never searches reasoning for the reference answer."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Literal

from .schema import VerifierItem

VERSION = "strict-final-v1"
# Opt-in first-attempt contract: MMMU P0 prompt bytes and team-final-answer-v8 extraction (v8_parser.py).
TURN1_EVAL_CONTRACT = "p0-eval-aligned-v1"
TRAINING_ANSWER_CONTRACT = "training-answer-v2"
# Opt-in retry rule bound only with that T1 contract: a committed MCQ retry answer naming no option is INCORRECT.
RETRY_OUT_OF_CHOICE = "incorrect-v1"
_FINAL = re.compile(r"^[ \t]*final answer[ \t]*:[ \t]*(.*?)[ \t]*$", re.I | re.M)
_INT = r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_NUM = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_RATIONAL = rf"(?:{_INT}\s*/\s*{_INT}|{_NUM})"
_RETRACT = re.compile(r"\b(?:actually|instead|correction|retract(?:ed)?|wrong|not correct)\b|\bthe answer is\b", re.I)


@dataclass(frozen=True)
class Extraction:
    status: Literal["OK", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"]
    answer: str | None


def _clean(value: str) -> str:
    value = value.strip()
    # Whole-value wrappers only; never delete symbols inside the submitted answer.
    removed_period = value.endswith(".")
    if removed_period:
        value = value[:-1].rstrip()
    for _ in range(4):
        before = value
        if value.startswith("**") and value.endswith("**") and len(value) > 4:
            value = value[2:-2].strip()
        elif value.startswith("$") and value.endswith("$") and len(value) > 2:
            value = value[1:-1].strip()
        elif re.fullmatch(r"\\boxed\{[^{}]*\}", value):
            value = value[7:-1].strip()
        if value == before:
            break
    if not removed_period and value.endswith("."):
        value = value[:-1].rstrip()
    return value


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).removesuffix(".")


def extract_final(text: str, finish_reason: str) -> Extraction:
    if not isinstance(text, str) or finish_reason not in ("stop", "length"):
        raise ValueError("answer: text and stop/length finish reason required")
    if finish_reason == "length":
        return Extraction("TRUNCATED", None)
    text = text.rstrip()
    matches = list(_FINAL.finditer(text))
    if not matches:
        return Extraction("INVALID_NO_FINAL", None)
    if len(matches) != 1:
        return Extraction("INVALID_MULTIPLE", None)
    tail = text[matches[0].end():]
    # The 200-character rule must not accept a short explicit retraction.
    if sum(not ch.isspace() for ch in tail) > 200 or _RETRACT.search(tail):
        return Extraction("INVALID_MULTIPLE", None)
    answer = _clean(matches[0].group(1))
    if not answer:
        return Extraction("INVALID_NO_FINAL", None)
    if re.search(r"\bor\b", answer, re.I):
        return Extraction("INVALID_MULTIPLE", None)
    return Extraction("OK", answer)


def extract_eval_aligned(text: str, finish_reason: str, question: str, choices: tuple[str, ...],
                         answer_type: str) -> Extraction:
    """Gold-blind V8 extraction of the whole response, as MMMU evaluation scores it.

    Length stops stay TRUNCATED (training-only cap). V8 itself rejects conflicting, revoked or
    alternative answers (NO_PARSE), so nothing here chooses among several answers."""
    if not isinstance(text, str) or finish_reason not in ("stop", "length") or not isinstance(question, str):
        raise ValueError("answer: text, question and stop/length finish reason required")
    if finish_reason == "length":
        return Extraction("TRUNCATED", None)
    from . import v8_parser

    found = (v8_parser.extract_mcq(text, choices, finish_reason, question=question) if answer_type == "mcq"
             else v8_parser.extract_open(text, finish_reason, question=question))
    if found["status"] != v8_parser.PARSED:
        return Extraction("INVALID_NO_FINAL", None)
    return Extraction("OK", found["answer"])


def extract_training(text: str, finish_reason: str, question: str, choices: tuple[str, ...],
                     answer_type: str) -> Extraction:
    """One committed answer, with no reference/alias/rule access during extraction."""
    if not isinstance(text, str) or finish_reason not in ("stop", "length") or not isinstance(question, str):
        raise ValueError("answer: text, question and stop/length finish reason required")
    if finish_reason == "length":
        return Extraction("TRUNCATED", None)
    header = r"(?:(?:the\s+)?(?:final|correct)\s+answer|(?:the\s+)?answer)"
    prefix = r"^[ \t]*(?:(?:therefore|thus|hence)[,:]?\s+)?"
    text = re.sub(r"(" + prefix + r")\*\*(" + header + r"\s*(?::|=|\bis\b)?)\*\*",
                  r"\1\2", text, flags=re.I | re.M)
    text = re.sub(r"^[ \t]*\*\*((?:(?!\*\*).)+)\*\*[ \t]*$", r"\1", text, flags=re.M)
    terminal_heading = re.search(r"^[ \t]*final[ \t]+answer[ \t]*:?[ \t]*\r?\n([^\r\n]+)[ \t\r\n]*\Z",
                                 text, re.I | re.M)
    if terminal_heading:
        from . import v8_parser

        boxed = terminal_heading[1].strip()
        if (v8_parser._BOX.fullmatch(boxed) and len(v8_parser._BOX.findall(text)) == 1 and
                not v8_parser._span_attributed(text, terminal_heading.start())):
            # A terminal heading and its boxed next line are one declaration for the existing conflict checks.
            text = text[:terminal_heading.start()] + "Final answer: " + boxed
    declaration_pattern = r"\b(?:final\s+answer|answer)\s*(?::|=|is\b)"
    declarations = list(re.finditer(declaration_pattern, text, re.I))
    if len(declarations) > 1:
        from . import v8_parser

        if re.search(r"\b(?:ignore|disregard|discard|reject|replace|revise|change)\s+"
                     r"(?:(?:my|our|the|this|that|previous|earlier|last|final)\s+)*answer\b|"
                     r"\banswer\s+(?:(?:may|might|could|would|should)\s+be|"
                     r"(?:is|was)\s+(?:not\b|incorrect\b|wrong\b))", text, re.I):
            return Extraction("INVALID_MULTIPLE", None)
        negation = (r"(?:(?:is|was)\s+(?:not\s+(?:correct|right)|incorrect|wrong)|"
                    r"isn['’]t\s+(?:correct|right)|cannot\s+be\s+(?:correct|right))\b")
        keys: set[tuple[Fraction | None, str]] = set()
        for index, declaration in enumerate(declarations):
            start = text.rfind("\n", 0, declaration.start()) + 1
            end = text.find("\n", declaration.end())
            end = len(text) if end < 0 else end
            marker = re.fullmatch(r"[ \t]*(?:(?:so|therefore|thus|hence)[,:]?\s+)?" + header
                                  + r"\s*(?::|=|\bis\b)[ \t]*(.*?)[ \t]*", text[start:end], re.I)
            if (marker is None or index + 1 < len(declarations) and declarations[index + 1].start() < end
                    or index == len(declarations) - 1 and text[end:].strip()
                    or v8_parser._span_attributed(text, declaration.start())):
                return Extraction("INVALID_MULTIPLE", None)
            value = marker[1].strip()
            if (not value or value.startswith(('"', "'", "“", "‘", "`"))
                    or re.search(r"\.{2,}|…|\?\s*$|\baccording\s+to\b", value, re.I)
                    or v8_parser._CONDITIONAL.search(value) or v8_parser._WITHDRAW.search(text)):
                return Extraction("INVALID_MULTIPLE", None)
            candidate = extract_training("Answer: " + value, finish_reason, question, choices, answer_type)
            if candidate.status != "OK" or candidate.answer is None:
                return Extraction("INVALID_MULTIPLE", None)
            normalized = " ".join(training_value(candidate.answer).split())
            if (normalized.startswith(('"', "'", "“", "‘"))
                    or re.search(rf"(?<!\w){re.escape(candidate.answer)}\s+{negation}", text, re.I)):
                return Extraction("INVALID_MULTIPLE", None)
            keys.add(_training_answer_key(candidate.answer))
            if len(keys) > 1:
                return Extraction("INVALID_MULTIPLE", None)
        for replacement in re.finditer(
                r"(?:^|\n|[.!?]\s+)(?:actually|instead|correction|revised\s+answer)\s*[,!:]\s*([^\n]+)", text, re.I):
            value = replacement[1]
            if (re.match(rf"{_NUM}|\\(?:boxed|\(|\[)|\$", value)
                    and _training_answer_key(value) not in keys):
                return Extraction("INVALID_MULTIPLE", None)
        for negated in re.finditer(
                rf"(?<!\w)({_RATIONAL}\s*(?:[A-Za-z°%µΩ][A-Za-z0-9°%µΩ^ /.*-]*)?)\s+{negation}", text, re.I):
            if _training_answer_key(negated[1]) in keys:
                return Extraction("INVALID_MULTIPLE", None)
        return candidate
    if answer_type == "mcq":
        return extract_eval_aligned(text, finish_reason, question, choices, answer_type)
    marker = re.search(prefix + header + r"\s*(?::|=|\bis\b)\s*(.*)$", text, re.I | re.M)
    if marker:
        value = marker[1].strip()
        tail = text[marker.end():].strip()
        if tail or not value:
            return Extraction("INVALID_MULTIPLE" if tail else "INVALID_NO_FINAL", None)
    else:
        from . import v8_parser
        boxes = list(v8_parser._BOX.finditer(text))
        if boxes:
            if len(boxes) != 1 or text[boxes[0].end():].strip(" \t\n.$"):
                return Extraction("INVALID_MULTIPLE", None)
            if v8_parser._span_attributed(text, boxes[0].start()):
                return Extraction("INVALID_NO_FINAL", None)
            value = boxes[0].group(0)
        else:
            # A standalone short answer is itself a declaration; never search a derivation for numbers.
            value = text.strip()
            if "\n" in value or re.search(r"\b(?:answer|actually|instead|correction)\b", value, re.I):
                return Extraction("INVALID_NO_FINAL", None)
    if not value or len(value) > 200 or len(value.split()) > 20:
        return Extraction("INVALID_NO_FINAL", None)
    value = _clean(value)
    normalized = training_value(value)
    if (re.search(r"\bor\b|;|\d\s+and\s+[+-]?\d", normalized, re.I)
            or "," in normalized and number(normalized) is None):
        return Extraction("INVALID_MULTIPLE", None)
    return Extraction("OK", value)


def training_value(value: str, *, variable: str | None = None) -> str:
    """Normalize whole mathematical wrappers, preserving signs, ranges and every unit."""
    value = unicodedata.normalize("NFKC", value).strip().removesuffix(".").strip()
    for _ in range(8):
        before = value
        value = _clean(value)
        for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("`", "`")):
            if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
                value = value[len(left):-len(right)].strip()
        if variable is not None:
            value = re.sub(rf"^{re.escape(variable)}\s*=\s*", "", value)
        match = re.fullmatch(r"\\(?:boxed|text|mathrm)\s*\{(.*)\}", value)
        if match:
            value = match[1].strip()
        if value == before:
            break
    value = re.sub(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ)", "°", value)
    value = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\^\{([23])\}", r"^\1", value)
    fraction = re.fullmatch(r"\\(?:d?frac)\s*\{\s*(" + _INT + r")\s*\}\s*\{\s*(" + _INT
                            + r")\s*\}(\s*(?:[A-Za-z°%µΩ][A-Za-z0-9°%µΩ^ /.*-]*)?)", value)
    if fraction:
        value = f"{fraction[1]}/{fraction[2]}{fraction[3]}"
    return value.replace("−", "-").strip()


def _training_answer_key(value: str) -> tuple[Fraction | None, str]:
    """Exact declaration identity; no missing-unit inference, conversion, tolerance or source aliases."""
    normalized = " ".join(training_value(value).split())
    quantity = re.fullmatch(rf"({_RATIONAL})\s*([A-Za-z°%µΩ][A-Za-z0-9°%µΩ^ /.*-]*)?", normalized)
    if quantity is None or (numeric := number(quantity[1])) is None:
        return None, normalized
    unit = quantity[2] or ""
    return numeric, "°" if unit.casefold() in ("degree", "degrees") else unit


def compare_training(answer: str, item: VerifierItem, rule: dict | None = None) -> bool | None:
    """Use only a declared typed rule; unsupported math/units are never guessed."""
    kind, accept = (item.answer_type, {}) if rule is None else (rule["answer_type"], rule["accept"])
    variable = accept.get("variable")
    answer, gold = (training_value(value, variable=variable) for value in (answer, item.gold))
    if kind == "range":
        def endpoints(value: str):
            match = re.fullmatch(rf"({_RATIONAL})\s*(?:-|–|—|to)\s*({_RATIONAL})", value)
            if match is None:
                return None
            left, right = number(match[1]), number(match[2])
            return None if left is None or right is None or left > right else (left, right)
        a, g = endpoints(answer), endpoints(gold)
        return None if a is None or g is None else a == g
    return compare(answer, replace(item, answer_type=kind, gold=gold, accept=accept))


def number(value: str) -> Fraction | None:
    """Parse one exact decimal/rational; reject malformed commas and non-finite values."""
    value = value.strip()
    if len(value) > 2048 or not re.fullmatch(_RATIONAL, value):
        return None
    value = value.replace(",", "")
    try:
        if "/" in value:
            left, right = value.split("/")
            return Fraction(int(left.strip()), int(right.strip()))
        decimal = Decimal(value)
        exponent = decimal.as_tuple().exponent
        if not decimal.is_finite() or not isinstance(exponent, int) or abs(exponent) > 1000:
            return None
        return Fraction(decimal)
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def mcq_choice(answer: str, choices: tuple[str, ...]) -> tuple[str | None, bool]:
    """Return label and content-conflict flag without access to the reference label."""
    match = re.fullmatch(r"(?:\(([A-J])\)|([A-J]))(?:[.:)]\s+(.+)|\.)?", answer.strip())
    if match is None:
        return None, False
    label = match[1] or match[2]
    index = ord(label) - ord("A")
    if index >= len(choices):
        return None, False
    conflict = match[3] is not None and normalize_text(match[3]) != normalize_text(choices[index])
    return label, conflict


def mcq_out_of_choice(answer: str, choices: tuple[str, ...]) -> bool:
    """True only for an answer that names no option: no standalone in-range label letter (any case) and no
    option text inside it or containing it. It cannot select the gold option and V8 evaluation scores it
    wrong; anything that may refer to an option stays undecided (False)."""
    key = normalize_text(answer)
    labels = "abcdefghijklmnopqrstuvwxyz"[:len(choices)]
    if not key or not labels or re.search(rf"(?<![a-z])[{labels}](?![a-z])", key):
        return False
    return not any(option and (option in key or key in option) for option in map(normalize_text, choices))


def is_ambiguous(answer: str, item: VerifierItem) -> bool:
    """Type-aware delimiters: commas in 1,234 and slash in 3/4 are not alternatives."""
    if re.search(r"\bor\b", answer, re.I):
        return True
    if item.answer_type == "list":
        return False
    if item.answer_type == "mcq":
        label, conflict = mcq_choice(answer, item.choices)
        if label is not None:
            return conflict
    if "/" not in answer and "," not in answer:
        return False
    if number(answer) is not None:
        return False
    if item.answer_type == "unit_number":
        match = re.fullmatch(rf"({_RATIONAL})\s*([^0-9\s].*)?", answer.strip())
        if match and number(match[1]) is not None:
            return False
    return True


def _numeric_equal(a: Fraction, g: Fraction, accept: dict) -> bool | None:
    policies = [k for k in ("round_to", "abs_tol", "rel_tol") if k in accept]
    if not policies:
        return a == g
    if len(policies) != 1:
        return None
    key = policies[0]
    value = accept[key]
    if key == "round_to":
        if type(value) is not int or not -1000 <= value <= 1000:
            return None
        scale = Fraction(10) ** value
        return round(a * scale) == round(g * scale)  # exact ties-to-even
    if type(value) not in (str, int):
        return None
    tolerance = number(str(value))
    if tolerance is None or tolerance < 0:
        return None
    limit = tolerance if key == "abs_tol" else tolerance * abs(g)
    return abs(a - g) <= limit


def _compare_typed(answer: str, gold: str, kind: str, accept: dict) -> bool | None:
    if kind in ("int", "decimal", "fraction"):
        a, g = number(answer), number(gold)
        if a is None or g is None:
            return None
        if kind == "int":
            if not re.fullmatch(_INT, gold):
                return None
            return bool(re.fullmatch(_INT, answer)) and a == g
        if kind == "fraction":
            flag = accept.get("accept_decimal", False)
            if type(flag) is not bool:
                return None
            if not flag and ("." in answer or "e" in answer.casefold()):
                return False
            return a == g
        return _numeric_equal(a, g, accept)
    if kind == "unit_number":
        aliases = accept.get("unit_aliases")
        require = accept.get("require_unit", True)
        if (not isinstance(aliases, list) or not aliases
                or any(type(v) is not str or not v.strip() for v in aliases)
                or type(require) is not bool):
            return None
        values = []
        for text in (answer, gold):
            match = re.fullmatch(rf"({_RATIONAL})\s*([^0-9\s].*)?", text)
            if match is None or (n := number(match[1])) is None:
                return None
            unit = " ".join(unicodedata.normalize("NFKC", match[2] or "").split())
            values.append((n, unit))
        units = {" ".join(unicodedata.normalize("NFKC", x).split()) for x in aliases}
        (a, au), (g, gu) = values
        if gu and gu not in units:
            return None
        if (require and not au) or (au and au not in units):
            return False
        return _numeric_equal(a, g, accept)
    if kind == "short_text":
        aliases = accept.get("aliases", [])
        case_sensitive = accept.get("case_sensitive", False)
        if (not isinstance(aliases, list) or any(type(v) is not str for v in aliases)
                or type(case_sensitive) is not bool):
            return None
        if case_sensitive:
            def exact_text(value: str) -> str:
                return " ".join(unicodedata.normalize("NFKC", value).split()).removesuffix(".")
            return exact_text(answer) in {exact_text(x) for x in [gold, *aliases]}
        return normalize_text(answer) in {normalize_text(x) for x in [gold, *aliases]}
    if kind == "list":
        ordered = accept.get("ordered", True)
        element_type = accept.get("element_type", "short_text")
        element_accept = accept.get("element_accept", {})
        if (type(ordered) is not bool or element_type not in ("int", "decimal", "fraction", "short_text")
                or type(element_accept) is not dict):
            return None
        def elements(s: str) -> list[str]:
            s = s.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            if re.search(r",\s*,", s) or s.strip().startswith(",") or s.strip().endswith(","):
                return []
            return [v for v in re.split(r"[,\s]+", s.strip()) if v]
        aa, gg = elements(answer), elements(gold)
        if not gg or not aa:
            return None
        if len(aa) != len(gg):
            return False
        if ordered:
            tests = [_compare_typed(a, g, element_type, element_accept) for a, g in zip(aa, gg, strict=True)]
            return None if None in tests else all(tests)
        # No fuzzy/tolerance matching for unordered lists: it can be non-transitive.
        if set(element_accept) - {"accept_decimal"}:
            return None
        if element_type == "short_text":
            return Counter(map(normalize_text, aa)) == Counter(map(normalize_text, gg))
        nums_a, nums_g = [number(x) for x in aa], [number(x) for x in gg]
        if None in nums_a or None in nums_g:
            return None
        if element_type == "int" and any(not re.fullmatch(_INT, x) for x in aa + gg):
            return None
        if element_type == "fraction":
            valid = [_compare_typed(x, x, element_type, element_accept) for x in aa + gg]
            if None in valid:
                return None
            if not all(valid):
                return False
        return Counter(nums_a) == Counter(nums_g)
    return None


def compare(answer: str, item: VerifierItem) -> bool | None:
    if not isinstance(answer, str) or not isinstance(item, VerifierItem):
        raise ValueError("answer: string and VerifierItem required")
    if is_ambiguous(answer, item):
        return None
    if item.answer_type == "mcq":
        label, conflict = mcq_choice(answer, item.choices)
        gold, gold_conflict = mcq_choice(item.gold, item.choices)
        if label is None or gold is None or conflict or gold_conflict:
            return None
        return label == gold
    return _compare_typed(answer.strip(), item.gold.strip(), item.answer_type, item.accept)
