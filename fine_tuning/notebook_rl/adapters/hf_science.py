"""Pinned HF science, diagram, and geometry train rows -> verifier-only Candidate records.

Gold comes from upstream human-annotated answer indices (ScienceQA, TQA) or from a numeric
answer string (Geometry3K). Only train parquet files are read. Rows with a missing image or
failing a data-quality check are skipped silently; programming errors propagate. No upstream
lecture, solution, or rationale text is copied into a Candidate.

ScienceQA rows are additionally dropped under FILTER_VERSION: the gold choice text appears verbatim in
the hint, the category is a scientific-names lesson, or the hint is longer than 200 characters.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .hf_common import Candidate, SourceSpec, mcq_from_answer

SPECS: dict[str, SourceSpec] = {
    "hf-scienceqa": SourceSpec(
        "hf-scienceqa", "derek-thomas/ScienceQA", "f18b0a70359ebfb41f658fd564208d0355b013f4",
        "CC-BY-NC-SA-4.0", "HUMAN", "IXL curriculum items (ScienceQA)", "science_diagram",
    ),
    "hf-tqa": SourceSpec(
        "hf-tqa", "yyyyifan/TQA", "1e9a4d2a5900cb3c97d1e8d4d78ce927584895bb",
        "CC-BY-NC-3.0", "HUMAN", "CK-12 textbook diagram questions (TQA)", "science_diagram",
    ),
    "hf-geometry3k": SourceSpec(
        "hf-geometry3k", "hiyouga/geometry3k", "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7",
        "MIT", "HUMAN", "Geometry3K annotators via InterGPS", "geometry",
    ),
}

FILTER_VERSION = "scienceqa-filter-v1"   # part of this file's hash, so pool manifests pin it
_SCIENTIFIC_NAME_CATEGORIES = frozenset(("Scientific names", "Classification and scientific names"))
_MAX_HINT_CHARS = 200
_SCIENCEQA_COLUMNS = ["image", "question", "choices", "answer", "hint", "topic", "category"]
_SCIENCEQA_SUBJECTS = {
    "biology": "Biology",
    "chemistry": "Chemistry",
    "physics": "Physics",
    "geography": "Geography",
    "earth-science": "Geography",
    "us-history": "History",
    "world-history": "History",
    "economics": "Economics",
    "civics": "Sociology",
    "global-studies": "Sociology",
    "writing-strategies": "Literature",
    "reading-comprehension": "Literature",
    "literacy-in-science": "Literature",
    "vocabulary": "Literature",
    "word-study": "Literature",
}
_TQA_COLUMNS = ["image", "question", "options", "answer"]
# Checked in order; the first subject with a keyword at a word start in question+options wins
# (word start, so 'ion' does not fire on 'formation'/'ablation'; plurals like 'atoms' still match).
_TQA_RULES = (
    ("Geography", ("earth", "moon", "sun", "planet", "season", "eclipse", "volcano", "rock", "layer",
                   "water cycle", "tide", "crust", "mantle", "phase", "orbit", "soil", "erosion",
                   "glacier", "waterfall", "river", "ocean", "weather", "cloud", "wind", "mountain", "delta",
                   "tsunami", "earthquake", "climate", "desert")),
    ("Physics", ("circuit", "magnet", "force", "lever", "pulley", "light", "refraction", "electric",
                 "energy", "wave")),
    ("Chemistry", ("atom", "molecule", "element", "compound", "reaction", "acid", "ion", "bond")),
)
_GEOMETRY_COLUMNS = ["images", "problem", "answer"]
_NUMBER = re.compile(r"-?\d+(\.\d+)?")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _image_bytes(image: dict | None) -> bytes | None:
    raw = image.get("bytes") if image else None
    return raw if isinstance(raw, bytes) and raw else None


def _train_rows(repo_dir: Path, columns: list[str]) -> Iterator[tuple[Path, int, dict]]:
    files = sorted(repo_dir.glob("data/train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet files under {repo_dir / 'data'}")
    for path in files:
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=columns):
            for row in batch.to_pylist():
                yield path, index, row
                index += 1


def _mcq(source: str, native_id: str, subject: str, question: str, options: list, answer: Any,
         raw: bytes) -> Candidate | None:
    if not question.strip():
        return None
    try:
        text, choices, gold = mcq_from_answer(question, options, answer)
    except ValueError:
        return None
    return Candidate(source=source, native_id=native_id, subject=subject, question=text, choices=choices,
                     answer_type="mcq", gold=gold, accept={}, images=(raw,), family_key=_sha256(raw))


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


def _drop_rules(hint: str, category: Any, gold_text: str) -> tuple[str, ...]:
    """Names of the FILTER_VERSION rules that drop an item (empty tuple = keep)."""
    gold = _norm(gold_text)
    rules = []
    if gold and f" {gold} " in f" {_norm(hint)} ":
        rules.append("gold_in_hint")
    if category in _SCIENTIFIC_NAME_CATEGORIES:
        rules.append("scientific_names")
    if len(hint) > _MAX_HINT_CHARS:
        rules.append("long_hint")
    return tuple(rules)


def _scienceqa_candidate(row: dict, native_id: str, dropped: Counter | None = None) -> Candidate | None:
    raw = _image_bytes(row["image"])
    subject = _SCIENCEQA_SUBJECTS.get(row["topic"])
    question, choices = row["question"], row["choices"]
    if raw is None or subject is None or not isinstance(question, str) or not isinstance(choices, list):
        return None
    hint = row["hint"] or ""
    text = (f"Context: {hint}\n" if hint.strip() else "") + question
    candidate = _mcq("hf-scienceqa", native_id, subject, text, choices, row["answer"], raw)
    if candidate is None:
        return None
    rules = _drop_rules(hint, row["category"], candidate.choices["ABCDEFGHIJ".index(candidate.gold)])
    if dropped is not None:
        dropped["valid"] += 1
        if rules:
            dropped.update(rules)
            dropped["any"] += 1
    return None if rules else candidate


def _tqa_candidate(row: dict, native_id: str) -> Candidate | None:
    raw = _image_bytes(row["image"])
    question, options = row["question"], row["options"]
    if raw is None or not isinstance(question, str) or not isinstance(options, list):
        return None
    text = " ".join([question, *(x for x in options if isinstance(x, str))]).lower()
    subject = next((name for name, words in _TQA_RULES
                    if any(re.search(rf"\b{re.escape(w)}", text) for w in words)), "Biology")
    return _mcq("hf-tqa", native_id, subject, question, options, row["answer"], raw)


def _geometry_candidate(row: dict, native_id: str) -> Candidate | None:
    images, problem, answer = row["images"], row["problem"], row["answer"]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(problem, str) \
            or not isinstance(answer, str):
        return None
    raw = _image_bytes(images[0])
    value = answer.strip()
    text = problem.strip().removeprefix("<image>").strip()
    if raw is None or not text or _NUMBER.fullmatch(value) is None:
        return None
    accept = {"round_to": len(value.partition(".")[2])} if "." in value else {"abs_tol": "0.000001"}
    return Candidate(source="hf-geometry3k", native_id=native_id, subject="Math", question=text, choices=(),
                     answer_type="decimal", gold=value, accept=accept,
                     images=(raw,), family_key=_sha256(raw))


def iter_scienceqa(repo_dir: Path, dropped: Counter | None = None) -> Iterator[Candidate]:
    """`dropped`, when given, counts valid rows ("valid"), rows dropped per rule and by any rule ("any")."""
    for _, index, row in _train_rows(repo_dir, _SCIENCEQA_COLUMNS):
        candidate = _scienceqa_candidate(row, f"train:{index}", dropped)
        if candidate is not None:
            yield candidate


def iter_tqa(repo_dir: Path) -> Iterator[Candidate]:
    for path, index, row in _train_rows(repo_dir, _TQA_COLUMNS):
        candidate = _tqa_candidate(row, f"{path.stem}:{index}")
        if candidate is not None:
            yield candidate


def iter_geometry3k(repo_dir: Path) -> Iterator[Candidate]:
    for _, index, row in _train_rows(repo_dir, _GEOMETRY_COLUMNS):
        candidate = _geometry_candidate(row, f"train:{index}")
        if candidate is not None:
            yield candidate


ITERATORS = {
    "hf-scienceqa": iter_scienceqa,
    "hf-tqa": iter_tqa,
    "hf-geometry3k": iter_geometry3k,
}
