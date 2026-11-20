"""Pinned HF medical train rows -> verifier-only Candidate records.

Gold comes from upstream clinician/physician-written answers: VQA-RAD (radiology QA) and SLAKE
(English physician annotations). Yes/no answers become a two-option MCQ; other short answers become
a short-text item. Invalid rows are skipped silently; programming errors propagate. No upstream
rationale text is copied into a Candidate.

SLAKE rows are additionally dropped under FILTER_VERSION: empty answers, and answers that are a truncated
form (>= 5 chars, next character a letter) of another SLAKE answer, e.g. "Cardiomegal" of "Cardiomegaly".
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..schema import AnswerType
from .hf_common import Candidate, SourceSpec

SPECS: dict[str, SourceSpec] = {
    "hf-vqarad": SourceSpec(
        "hf-vqarad", "flaviagiammarino/vqa-rad", "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9",
        "CC0-1.0", "HUMAN", "VQA-RAD clinician-written QA (Lau et al. 2018)", "radiology",
    ),
    "hf-slake": SourceSpec(
        "hf-slake", "BoKelvin/SLAKE", "a9083ce6c34ac3ffb17671a605962924d8a8f9e9",
        "CC-BY-4.0", "HUMAN", "SLAKE physician annotations (Liu et al. 2021)", "radiology",
    ),
}
FILTER_VERSION = "slake-filter-v1"   # part of this file's hash, so pool manifests pin it
_PREFIX_MIN_CHARS = 5
_VQARAD_COLUMNS = ["image", "question", "answer"]
_SHORT_MAX_WORDS = 3
_SHORT_MAX_CHARS = 40


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _qa_answer(question: Any, answer: Any) -> tuple[str, tuple[str, ...], AnswerType, str] | None:
    """Answer rule shared by both sources: yes/no -> 2-option MCQ, else a <=3-word short answer."""
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    text = question.strip()
    value = answer.strip()
    if not text or not value:
        return None
    text = text[:1].upper() + text[1:]
    if value.casefold() in ("yes", "no"):
        return text, ("Yes", "No"), "mcq", "A" if value.casefold() == "yes" else "B"
    if len(value.split()) <= _SHORT_MAX_WORDS and "," not in value and len(value) <= _SHORT_MAX_CHARS:
        return text, (), "short_text", value
    return None


def _train_rows(repo_dir: Path, columns: list[str]) -> Iterator[tuple[int, dict]]:
    """Yields (row index across all train shards, row). Skipped rows still consume an index."""
    files = sorted(repo_dir.glob("data/train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet files under {repo_dir / 'data'}")
    index = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns):
            for row in batch.to_pylist():
                yield index, row
                index += 1


def _vqarad_candidate(row: dict, native_id: str) -> Candidate | None:
    image = row["image"]
    raw = image.get("bytes") if image else None
    parsed = _qa_answer(row["question"], row["answer"])
    if not isinstance(raw, bytes) or not raw or parsed is None:
        return None
    question, choices, answer_type, gold = parsed
    return Candidate(source="hf-vqarad", native_id=native_id, subject="Clinical_Medicine",
                     question=question, choices=choices, answer_type=answer_type, gold=gold,
                     images=(raw,), family_key=_sha256(raw))


def iter_vqarad(repo_dir: Path) -> Iterator[Candidate]:
    for index, row in _train_rows(repo_dir, _VQARAD_COLUMNS):
        candidate = _vqarad_candidate(row, f"train:{index}")
        if candidate is not None:
            yield candidate


def _answer_key(answer: Any) -> str:
    return answer.strip().casefold() if isinstance(answer, str) else ""


def slake_prefix_answers(answers: Iterable[Any]) -> dict[str, tuple[str, ...]]:
    """Normalised answer -> the longer answers it is a strict prefix of (>= 5 chars, next char alphabetic)."""
    keys = {_answer_key(a) for a in answers} - {""}
    found: defaultdict[str, set[str]] = defaultdict(set)
    for longer in keys:
        for end in range(_PREFIX_MIN_CHARS, len(longer)):
            if longer[end].isalpha() and longer[:end] in keys:
                found[longer[:end]].add(longer)
    return {prefix: tuple(sorted(found[prefix])) for prefix in sorted(found)}


def iter_slake(repo_dir: Path, dropped: Counter | None = None) -> Iterator[Candidate]:
    """`dropped`, when given, counts English rows dropped as "empty_answer" and (valid rows) "prefix_answer"."""
    dropped = Counter() if dropped is None else dropped
    rows = json.loads((repo_dir / "train.json").read_text(encoding="utf-8"))
    prefixes = slake_prefix_answers(r["answer"] for r in rows if r["q_lang"] == "en")
    with zipfile.ZipFile(repo_dir / "imgs.zip") as archive:
        members = set(archive.namelist())
        for row in rows:
            img_name = row["img_name"]
            if row["q_lang"] != "en" or not isinstance(img_name, str):
                continue
            key = _answer_key(row["answer"])
            if not key:
                dropped["empty_answer"] += 1
                continue
            member = f"imgs/{img_name}"
            parsed = _qa_answer(row["question"], row["answer"])
            if member not in members or parsed is None:
                continue
            if key in prefixes:
                dropped["prefix_answer"] += 1
                continue
            raw = archive.read(member)
            if not raw:
                continue
            question, choices, answer_type, gold = parsed
            subject = ("Clinical_Medicine" if row["content_type"] == "Abnormality"
                       else "Basic_Medical_Science")
            yield Candidate(source="hf-slake", native_id=f"train:{row['qid']}", subject=subject,
                            question=question, choices=choices, answer_type=answer_type, gold=gold,
                            images=(raw,), family_key=img_name.split("/")[0])


ITERATORS = {
    "hf-vqarad": iter_vqarad,
    "hf-slake": iter_slake,
}
