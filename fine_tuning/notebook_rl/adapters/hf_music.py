"""Pinned SSMR-Bench visual train rows -> verifier-only Candidate records.

Gold is a program over upstream answer fields (rule-based sheet-music QA, no LLM). Only the
Train_visual parquet shard is read. Rows without a sheet-music image, with ABC notation context,
or with missing, empty, or duplicate answers are skipped silently; programming errors propagate.
No ABC text is copied into a Candidate.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from .hf_common import Candidate, SourceSpec, program_mcq

SPECS: dict[str, SourceSpec] = {
    "hf-ssmr": SourceSpec(
        "hf-ssmr", "Sylence/SSMR-Bench", "5f6f53d1de73d26d050ca27d10ee595f8018e3bd",
        "MIT", "PROGRAM", "SSMR-Bench programmatic sheet-music QA (rule-based, no LLM)", "music",
    ),
}

_COLUMNS = [
    "question", "abc_context", "sheet_music_img", "correct_answer",
    "incorrect_answer1", "incorrect_answer2", "incorrect_answer3",
]
_ANSWER_COLUMNS = ["correct_answer", "incorrect_answer1", "incorrect_answer2", "incorrect_answer3"]


def _candidate(row: dict, row_index: int) -> Candidate | None:
    image = row["sheet_music_img"]
    raw = image.get("bytes") if image else None
    question, abc = row["question"], row["abc_context"]
    answers = [row[name] for name in _ANSWER_COLUMNS]
    if not isinstance(raw, bytes) or not raw:
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    if abc is not None and abc.strip():
        return None
    if not all(type(x) is str for x in answers):
        return None
    texts = [x.strip() for x in answers]
    if not all(texts) or len(set(texts)) != len(texts):
        return None
    choices, letter = program_mcq(texts[0], texts, key=f"ssmr:{row_index}", n_options=4)
    return Candidate(source="hf-ssmr", native_id=f"Train_visual:{row_index}", subject="Music",
                     question=question.strip(), choices=choices, answer_type="mcq", gold=letter,
                     accept={}, images=(raw,), family_key=hashlib.sha256(raw).hexdigest())


def iter_ssmr(repo_dir: Path) -> Iterator[Candidate]:
    files = sorted(repo_dir.glob("data/Train_visual-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Train_visual parquet files under {repo_dir / 'data'}")
    row_index = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=_COLUMNS):
            for row in batch.to_pylist():
                candidate = _candidate(row, row_index)
                row_index += 1
                if candidate is not None:
                    yield candidate


ITERATORS = {"hf-ssmr": iter_ssmr}
