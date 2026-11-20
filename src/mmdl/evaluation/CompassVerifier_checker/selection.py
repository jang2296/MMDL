"""Validate saved V8 records and select fixed checker inputs without model imports."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from mmdl.evaluation.parsers import score_with_parser
from mmdl.runtime.artifacts import digest, sha256_file

from . import CHECKER_VERSION, SCORING_ID
from .prompt import CV_PROMPT, CV_PROMPT_SHA256

CATEGORIES = (
    "V8_CORRECT", "LENGTH", "EMPTY", "NO_PARSE_STOP", "OPEN_MISMATCH_STOP",
    "MCQ_EXPLICIT_WRONG",
)
TARGET_CATEGORIES = ("NO_PARSE_STOP", "OPEN_MISMATCH_STOP")

_PARSER_FILES = {
    "src/mmdl/evaluation/final_answer_parser.py":
        "fc2c6a757ce7361efae13be9f22f8bedcc9c35d66373124b8815b8528b2e31a0",
    "src/mmdl/evaluation/final_answer_parser_v8.py":
        "2beb1230283cb882f47d6e8af72b01180847fd7aa4b7ac640a5173e391f3f891",
    "src/mmdl/evaluation/parsers.py":
        "1bfdabe251d5cbf14b6554f2c979492ec18207de5fb826a0045dd2d93ca3e221",
    "third_party/mmmu/eval_utils.py":
        "cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367",
}


def load_config(path: Path | None = None) -> dict:
    """Reject changes to v1 instead of silently reusing its scoring identity."""
    config = yaml.safe_load((path or Path(__file__).with_name("checker_v1.yaml"))
                            .read_text(encoding="utf-8"))
    expected = {
        "checker_version": CHECKER_VERSION,
        "scoring_id": SCORING_ID,
        "base_parser": "team-final-answer-v8",
        "direction": "v8_wrong_to_correct_only",
        "base_parser_files": _PARSER_FILES,
        "judge": {
            "official": {
                "model": "opencompass/CompassVerifier-7B",
                "revision": "8fee654b23ec971ad4a5c6d69bd99cf0f0188197",
                "dtype": "bfloat16", "quantization": "none",
            },
            "smoke_allowed": [
                {"model": "opencompass/CompassVerifier-3B",
                 "revision": "27d6d660bdf6c2b8a9f45906d4b76c508af65ab0"},
                {"model": "opencompass/CompassVerifier-7B",
                 "revision": "8fee654b23ec971ad4a5c6d69bd99cf0f0188197"},
            ],
            "prompt": {
                "repo": "open-compass/CompassVerifier",
                "commit": "2d7cba6df0b21f9c6121786ac1e5770c68473598",
                "path": "src/prompts.py", "constant": "CV_PROMPT",
                "text_sha256": CV_PROMPT_SHA256,
            },
            "labels": {"A": "CORRECT", "B": "INCORRECT", "C": "INVALID"},
        },
        "selection": {"finish_reason": "stop", "categories": list(TARGET_CATEGORIES)},
        "input": {"question_field": "prompt", "mcq_gold_format": "{letter}. {option_text}",
                  "response_tail_tokens": 3072},
        "decision": {"accept_threshold": 0.90},
    }
    if hashlib.sha256(CV_PROMPT.encode("utf-8")).hexdigest() != CV_PROMPT_SHA256:
        raise ValueError("Checker prompt SHA256 differs from the frozen prompt")
    try:
        matches = isinstance(config, dict) and digest(config) == digest(expected)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError("Checker v1 configuration changed; use a new version")
    return config


def verify_base_parser(config: dict, root: Path) -> None:
    if config.get("base_parser_files") != _PARSER_FILES:
        raise ValueError("V8 parser file manifest differs from the frozen contract")
    for relative, expected in _PARSER_FILES.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"V8 parser SHA256 mismatch: {relative}")


def _validate_row(row: object) -> None:
    if not isinstance(row, dict):
        raise ValueError("Prediction record must be a JSON object")
    for key in ("id", "subject", "question_type", "prompt", "question", "raw_response",
                "finish_reason", "parse_status", "extraction_rule", "_record_sha256"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"Prediction field {key} must be a string")
    if not row["id"].strip() or not row["subject"].strip():
        raise ValueError("Prediction id and subject must be nonempty")
    if row["question_type"] not in ("multiple-choice", "open"):
        raise ValueError("Unsupported prediction question_type")
    if row["finish_reason"] not in ("stop", "length"):
        raise ValueError("Unsupported prediction finish_reason")
    if row["parse_status"] not in ("EMPTY", "NO_PARSE", "PARSED"):
        raise ValueError("Unsupported prediction parse_status")
    if type(row.get("correct")) is not bool:
        raise ValueError("Prediction correct must be a boolean")
    if type(row.get("generated_tokens")) is not int or row["generated_tokens"] < 0:
        raise ValueError("Prediction generated_tokens must be a nonnegative integer")
    if "parsed_answer" not in row or (row["parsed_answer"] is not None
                                         and not isinstance(row["parsed_answer"], str)):
        raise ValueError("Prediction parsed_answer must be a string or null")
    options = row.get("options")
    if not isinstance(options, list) or any(not isinstance(value, str) for value in options):
        raise ValueError("Prediction options must be a list of strings")
    if len(options) > 26 or (row["question_type"] == "multiple-choice" and not options):
        raise ValueError("Prediction MCQ requires 1 to 26 options")
    gold_texts(row)
    if not re.fullmatch(r"[0-9a-f]{64}", row["_record_sha256"]):
        raise ValueError("Invalid prediction record SHA256")
    if row["_record_sha256"] != digest({k: v for k, v in row.items()
                                        if k != "_record_sha256"}):
        raise ValueError(f"Prediction record integrity failure: {row['id']}")


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_rows(paths: Sequence[Path]) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=_json_object)
                    _validate_row(row)
                    if row["id"] in seen:
                        raise ValueError(f"Duplicate prediction id: {row['id']}")
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                seen.add(row["id"])
                rows.append(row)
    if not rows:
        raise ValueError("No prediction records supplied")
    return rows


def replay_v8(rows: Sequence[dict]) -> None:
    for row in rows:
        score = score_with_parser(row["raw_response"], row["question_type"], row["options"],
                                  row["answer"], "team-final-answer-v8", row["finish_reason"],
                                  question=row["question"])
        if any(score[key] != row[key] for key in ("correct", "parse_status", "parsed_answer")):
            raise ValueError(f"Stored prediction differs from V8 replay: {row['id']}")


def categorize(row: Mapping) -> str:
    if row["correct"]:
        return "V8_CORRECT"
    if row["finish_reason"] != "stop":
        return "LENGTH"
    if row["parse_status"] == "EMPTY" or not row["raw_response"].strip():
        return "EMPTY"
    if row["parse_status"] == "NO_PARSE":
        return "NO_PARSE_STOP"
    if row["question_type"] == "open" and row["parse_status"] == "PARSED":
        return "OPEN_MISMATCH_STOP"
    return "MCQ_EXPLICIT_WRONG"


def gold_texts(row: Mapping) -> list[str]:
    answer = row.get("answer")
    aliases = [answer] if isinstance(answer, str) else answer
    if not isinstance(aliases, list) or not aliases or any(
            not isinstance(value, str) or not value.strip() for value in aliases):
        raise ValueError("Prediction answer must be a nonempty string or list of strings")
    if row["question_type"] == "open":
        return list(aliases)
    options = row["options"]
    if any(len(letter) != 1 or not "A" <= letter <= "Z"
           or ord(letter) - 65 >= len(options) for letter in aliases):
        raise ValueError("Prediction gold letter is outside its MCQ options")
    return [f"{letter}. {options[ord(letter) - 65]}" for letter in aliases]


def tail(text: str, max_tokens: int, encode: Callable[[str], Sequence[int]],
         decode: Callable[[Sequence[int]], str]) -> tuple[str, bool]:
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("Response token limit must be a positive integer")
    ids = encode(text)
    if len(ids) <= max_tokens:
        return text, False
    # Byte-level token cuts can decode to replacement characters; keep only a true source suffix.
    for start in range(len(ids) - max_tokens, len(ids)):
        try:
            candidate = decode(ids[start:])
        except UnicodeDecodeError:
            continue
        if candidate and text.endswith(candidate):
            offset = len(text) - len(candidate)
            if offset and text[offset - 1] != "\n" and "\n" in candidate:
                aligned = candidate.split("\n", 1)[1]
                if aligned.strip():
                    candidate = aligned
            if candidate.strip() and len(encode(candidate)) <= max_tokens:
                return candidate, True
    raise ValueError("Tokenizer cannot produce a valid response suffix within the token limit")


def build_targets(rows: Sequence[dict], config: dict, encode, decode) -> list[dict]:
    targets = []
    for row in rows:
        category = categorize(row)
        if category not in TARGET_CATEGORIES:
            continue
        response, truncated = tail(row["raw_response"], config["input"]["response_tail_tokens"],
                                   encode, decode)
        for alias_index, gold in enumerate(gold_texts(row)):
            question = row["prompt"]
            targets.append({
                "id": row["id"], "subject": row["subject"],
                "question_type": row["question_type"], "category": category,
                "alias_index": alias_index, "question": question, "gold_answer": gold,
                "llm_response": response, "response_truncated": truncated,
                "response_tokens": len(encode(response)),
                "input_sha256": digest([CHECKER_VERSION, CV_PROMPT_SHA256, question, gold, response]),
            })
    return targets
