"""HuggingFaceM4/the_cauldron train rows -> hf_common.Candidate (ai2d, mapqa, finqa, tat_qa, plotqa).

One row = one image (or two) plus several QA turns; at most ONE turn per row becomes a Candidate. Only
the final answer of the assistant text is used, as verifier-only gold; any computation/rationale text
is dropped. Rows or turns that do not fit a rule are skipped, never repaired.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

from ..schema import AnswerType
from .hf_common import SUBJECTS, Candidate, SourceSpec, mcq_from_answer, parse_letter_choices, strip_instruction_suffix

REVISION = "847a98a779b1652d65111daf20c972dfcd333605"
_REPO = "HuggingFaceM4/the_cauldron"

SPECS: dict[str, SourceSpec] = {
    "hf-ai2d": SourceSpec("hf-ai2d", _REPO, REVISION, "CC-BY-SA-4.0", "HUMAN",
                          "AI2D annotators (Allen AI) via the_cauldron", "diagram"),
    "hf-mapqa": SourceSpec("hf-mapqa", _REPO, REVISION, "CC-BY-SA-4.0", "PROGRAM",
                           "MapQA templates over KFF/US data via the_cauldron", "map"),
    "hf-finqa": SourceSpec("hf-finqa", _REPO, REVISION, "MIT", "HUMAN",
                           "FinQA financial experts via the_cauldron", "table"),
    "hf-tatqa": SourceSpec("hf-tatqa", _REPO, REVISION, "CC-BY-4.0", "HUMAN",
                           "TAT-QA financial annotators via the_cauldron", "table"),
    "hf-plotqa": SourceSpec("hf-plotqa", _REPO, REVISION, "CC-BY-4.0", "PROGRAM",
                            "PlotQA templates over real data via the_cauldron", "chart_line"),
}

_Made = tuple[str, str, tuple[str, ...], AnswerType, str, dict]   # subject, question, choices, type, gold, accept
_NUMBER = re.compile(r"-?\d+(\.\d+)?")
_GEOGRAPHY = ("earth", "moon", "sun", "planet", "season", "eclipse", "volcano", "rock", "layer", "water cycle",
              "tide", "crust", "mantle", "phase", "orbit", "soil", "erosion")
_PHYSICS = ("circuit", "magnet", "force", "lever", "pulley", "light ray", "refraction", "electric")
_PLOT_FINANCE = re.compile(r"(?i)(\bus\$|\b(usd|revenue|earned|expenditure|cost|tax|debt|trade|exports?|imports?)\b)")
_FIN_ACCOUNTING = re.compile(r"(?i)(depreciation|amortization|tax|lease|liabilit|asset|inventory|expense|goodwill|receivable|payable)")


def _answer(assistant: str) -> str:
    text = assistant.strip()
    return text[:-1] if text.endswith(".") else text


def _yes_no(subject: str, question: str, a: str) -> _Made:
    return subject, question, ("Yes", "No"), "mcq", "A" if a == "Yes" else "B", {}


def _decimal(subject: str, question: str, a: str, accept: dict) -> _Made:
    return subject, question, (), "decimal", a, accept


def _word_start(words: tuple[str, ...], text: str) -> bool:
    return any(re.search(r"\b" + re.escape(w), text) for w in words)


def _ai2d(user: str, assistant: str, key: str) -> _Made | None:
    parsed = parse_letter_choices(user)
    found = re.fullmatch(r"Answer: ([A-J])", assistant.strip())
    if parsed is None or found is None:
        return None
    try:
        question, choices, gold = mcq_from_answer(parsed[0], list(parsed[1]), "ABCDEFGHIJ".index(found[1]))
    except ValueError:                    # duplicate options or letter beyond the choices: bad row
        return None
    text = f"{question} {' '.join(choices)}".lower()
    subject = ("Geography" if _word_start(_GEOGRAPHY, text)
               else "Physics" if _word_start(_PHYSICS, text) else "Biology")
    return subject, question, choices, "mcq", gold, {}


def _mapqa(user: str, assistant: str, key: str) -> _Made | None:
    a, question = _answer(assistant), strip_instruction_suffix(user)
    subject = "Sociology" if int(key[:8], 16) % 2 == 0 else "Geography"
    if a in ("Yes", "No"):
        return _yes_no(subject, question, a)
    if a and a.casefold() != "n/a" and "," not in a and len(a) <= 20:
        return subject, question, (), "short_text", a, {}
    return None


def _plotqa(user: str, assistant: str, key: str) -> _Made | None:
    a, question = _answer(assistant), strip_instruction_suffix(user)
    subject = "Finance" if _PLOT_FINANCE.search(question) else "Economics"
    if a in ("Yes", "No"):
        return _yes_no(subject, question, a)
    if _NUMBER.fullmatch(a):
        accept = {"round_to": len(a.partition(".")[2])} if "." in a else {"abs_tol": "0.000001"}
        return _decimal(subject, question, a, accept)
    return None


def _finqa(user: str, assistant: str, key: str) -> _Made | None:
    a, question = _answer(assistant.rpartition("Answer:")[2]), strip_instruction_suffix(user)   # drops "Computations:"
    if not _NUMBER.fullmatch(a):
        return None
    subject = "Accounting" if _FIN_ACCOUNTING.search(question.rpartition("\n")[2]) else "Finance"   # question line only
    return _decimal(subject, question, a, {"rel_tol": "0.01"} if "." in a else {"abs_tol": "0.000001"})


def _tatqa(user: str, assistant: str, key: str) -> _Made | None:
    a = _answer(assistant)
    if not _NUMBER.fullmatch(a):
        return None
    return _decimal("Accounting", strip_instruction_suffix(user), a, {"rel_tol": "0.01"})


def _iter_config(repo: Path, config: str, source: str, parse: Callable[[str, str, str], _Made | None]) -> Iterator[Candidate]:
    for path in sorted((repo / config).glob("train-*.parquet")):
        index = -1
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["images", "texts"]):
            for row in batch.to_pylist():
                index += 1
                images = tuple(img["bytes"] for img in row["images"] or ())
                if not 1 <= len(images) <= 2 or not all(images):
                    continue
                key = hashlib.sha256(images[0]).hexdigest()
                best: tuple[str, str, _Made] | None = None      # label-blind pick: smallest digest of the native id
                for turn_index, turn in enumerate(row["texts"] or ()):
                    made = parse(turn["user"] or "", turn["assistant"] or "", key)
                    if made is None or not made[1]:
                        continue
                    subject, _, choices, answer_type, gold, _ = made
                    if subject not in SUBJECTS or (answer_type == "mcq" and gold not in "ABCDEFGHIJ"[:len(choices)]):
                        raise ValueError(f"{source}: parser produced an invalid candidate")
                    native_id = f"{config}/{path.stem}:{index}:{turn_index}"
                    digest = hashlib.sha256(native_id.encode()).hexdigest()
                    if best is None or digest < best[0]:
                        best = (digest, native_id, made)
                if best is not None:
                    subject, question, choices, answer_type, gold, accept = best[2]
                    yield Candidate(source, best[1], subject, question, choices, answer_type, gold, accept, images, key)


def iter_ai2d(repo: Path) -> Iterator[Candidate]:
    return _iter_config(repo, "ai2d", "hf-ai2d", _ai2d)


def iter_mapqa(repo: Path) -> Iterator[Candidate]:
    return _iter_config(repo, "mapqa", "hf-mapqa", _mapqa)


def iter_finqa(repo: Path) -> Iterator[Candidate]:
    return _iter_config(repo, "finqa", "hf-finqa", _finqa)


def iter_tatqa(repo: Path) -> Iterator[Candidate]:
    return _iter_config(repo, "tat_qa", "hf-tatqa", _tatqa)


def iter_plotqa(repo: Path) -> Iterator[Candidate]:
    return _iter_config(repo, "plotqa", "hf-plotqa", _plotqa)


ITERATORS: dict[str, Callable[[Path], Iterator[Candidate]]] = {
    "hf-ai2d": iter_ai2d, "hf-mapqa": iter_mapqa, "hf-finqa": iter_finqa, "hf-tatqa": iter_tatqa, "hf-plotqa": iter_plotqa,
}
