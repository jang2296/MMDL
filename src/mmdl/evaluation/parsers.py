"""MMMU response parsing with the official open evaluator and no MCQ guessing."""

from __future__ import annotations

import importlib.util
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


EMPTY = "EMPTY"
NO_PARSE = "NO_PARSE"
PARSED = "PARSED"
_MCQ_TYPES = {"mcq", "multiple-choice", "multiple choice"}
_OPEN_TYPES = {"open", "open-ended", "open ended", "free-response", "free response"}
_OFFICIAL: ModuleType | None = None


def score_with_parser(raw: Any, question_type: str, options: Sequence[str] | Mapping[str, str],
                      answer: Any, parser_id: str = "mmmu-official-no-random-v1",
                      finish_reason: str | None = None) -> dict[str, Any]:
    """Select an explicit scoring version; keep historical results reproducible."""
    if parser_id == "mmmu-official-no-random-v1":
        return score_response(raw, question_type, options, answer)
    if parser_id == "team-final-answer-v4":
        from mmdl.evaluation.final_answer_parser import score_response as final_score

        return final_score(raw, question_type, options, answer, finish_reason)
    raise ValueError(f"Unknown scoring version: {parser_id}")


def _question_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in _MCQ_TYPES:
        return "mcq"
    if normalized in _OPEN_TYPES:
        return "open"
    raise ValueError(f"Unsupported question_type: {value!r}")


def _official_eval_utils() -> ModuleType:
    """Load the pinned evaluator without leaving its import-time random seed behind."""
    global _OFFICIAL
    if _OFFICIAL is not None:
        return _OFFICIAL
    path = Path(__file__).resolve().parents[3] / "third_party/mmmu/eval_utils.py"
    spec = importlib.util.spec_from_file_location("mmdl_official_mmmu_eval_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official evaluator from {path}")
    state = random.getstate()
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        random.setstate(state)
    _OFFICIAL = module
    return module


def _choices(options: Sequence[str] | Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
    if isinstance(options, Mapping):
        labels = [str(label) for label in options]
        return labels, {str(label): str(text) for label, text in options.items()}
    if isinstance(options, (str, bytes)):
        raise TypeError("options must be a list of option text")
    if len(options) > 26:
        raise ValueError("MCQ supports at most 26 options")
    labels = [chr(65 + index) for index in range(len(options))]
    return labels, dict(zip(labels, (str(option) for option in options)))


def parse_mcq_response(response: str, options: Sequence[str] | Mapping[str, str]) -> str | None:
    """Adapt official parse_multi_choice_response, omitting its random fallback."""
    all_choices, index2ans = _choices(options)
    response = str(response)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for index, answer in index2ans.items():
            if answer.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    positions = []
    if index_ans:
        needle = ("({})" if ans_with_brack else " {} ")
        positions = [response.rfind(needle.format(choice)) for choice in candidates]
    else:
        positions = [response.lower().rfind(index2ans[choice].lower()) for choice in candidates]
    return candidates[max(range(len(candidates)), key=positions.__getitem__)]


def _mcq_correct(answer: Any, parsed: str) -> bool:
    if isinstance(answer, (list, tuple, set)):
        return parsed in answer
    return answer == parsed


def score_response(
    raw: Any,
    question_type: str,
    options: Sequence[str] | Mapping[str, str],
    answer: Any,
) -> dict[str, Any]:
    """Return parsed answer, parser status, and correctness for one response."""
    if raw is None or not str(raw).strip():
        return {"parsed_answer": None, "parse_status": EMPTY, "correct": False}

    kind = _question_type(question_type)
    if kind == "mcq":
        parsed = parse_mcq_response(str(raw), options)
        if parsed is None:
            return {"parsed_answer": None, "parse_status": NO_PARSE, "correct": False}
        return {
            "parsed_answer": parsed,
            "parse_status": PARSED,
            "correct": _mcq_correct(answer, parsed),
        }

    official = _official_eval_utils()
    open_answers = sorted(
        official.parse_open_response(str(raw)),
        key=lambda value: (type(value).__name__, repr(value)),
    )
    return {
        "parsed_answer": open_answers,
        "parse_status": PARSED,
        "correct": bool(official.eval_open(answer, open_answers)),
    }
