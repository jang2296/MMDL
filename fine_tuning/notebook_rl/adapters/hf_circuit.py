"""Original CircuitVQA train questions and images -> verifier-only program gold.

The pinned HF master_bbox.json is byte-identical to the authors' question release. Read
only count/count-complex/junction answers computed from upstream annotations; omit
optional ChatGPT descriptions and all value/position targets. Question templates
were paraphrased with ChatGPT upstream; no generated explanation is a target.
Images are read directly from the authors' 384a.zip (no extraction or redrawing).
Original val/test circuit families, including alternate drawing variants, are held
out before the shared split_for assignment. Nothing here reads benchmark data.
Simple counts must also equal the count in the supplied box annotation; unsupported
source answers are excluded, never corrected.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

from ..schema import AnswerType
from .hf_common import Candidate, SourceSpec

SPECS = {
    "hf-circuitvqa": SourceSpec(
        "hf-circuitvqa", "MirandaAbhilash/circuitvqa-dataset",
        "bae273394c787aa7bd67d3869d291c8793fccc97", "CC-BY-SA-3.0", "PROGRAM",
        "CircuitVQA authors' annotated-box programs (Mehta et al. 2024); "
        "ChatGPT question templates only, no descriptions or teacher targets",
        "circuit",
    ),
}
_FILE = re.compile(r"[A-Za-z0-9_.-]+\Z")
_COUNT = re.compile(r"\d+(?:\.0+)?\Z")


def circuit_family(filename: str) -> str:
    """Group source augmentations and the D/P redrawings of one original circuit."""
    if not isinstance(filename, str) or _FILE.fullmatch(filename) is None or filename in (".", ".."):
        raise ValueError("CircuitVQA: invalid source image name")
    base = re.split(r"_(?:jpg|png)\.rf\.", filename, maxsplit=1)[0]
    return re.sub(r"_D\d+_P\d+$", "", base)


def iter_circuitvqa(repo_dir: Path) -> Iterator[Candidate]:
    root = Path(repo_dir)
    rows = json.loads((root / "questions/master_bbox.json").read_text(encoding="utf-8"))
    heldout = {circuit_family(row["file"]) for row in rows if row["splittype"] in ("val", "test")}
    with zipfile.ZipFile(root / "384a.zip") as archive:
        for index, row in enumerate(rows):
            if row["splittype"] != "train" or row["qtype"] not in ("count", "count-complex", "junction"):
                continue
            family = circuit_family(row["file"])
            if family in heldout:
                continue
            question, answer = row["question"], row["answer"]
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
                continue
            choices: tuple[str, ...]
            answer_type: AnswerType
            if row["qtype"] == "junction":
                if answer not in ("yes", "no"):
                    continue
                choices, answer_type, gold = ("Yes", "No"), "mcq", "A" if answer == "yes" else "B"
                directory = "model-inputs-jn"
            else:
                if _COUNT.fullmatch(answer) is None:
                    continue
                choices, answer_type, gold = (), "int", str(int(answer.partition(".")[0]))
                if row["qtype"] == "count":
                    bbox, symbol = row.get("bbox"), row.get("symbol")
                    if not isinstance(bbox, str) or not isinstance(symbol, str):
                        continue
                    boxes = [entry.strip().split() for entry in bbox.split(" . ") if entry.strip()]
                    if not boxes or any(len(box) != 5 for box in boxes):
                        continue
                    count = sum(box[0].replace(".", "-") == symbol.replace(".", "-") for box in boxes)
                    if str(count) != gold:
                        continue
                directory = "model-inputs"
            image = archive.read(f"384a/{directory}/train/{row['file']}.jpg")
            if image:
                yield Candidate(
                    source="hf-circuitvqa", native_id=f"train:{index}", subject="Electronics",
                    question=question.strip(), choices=choices, answer_type=answer_type, gold=gold,
                    images=(image,), family_key=family,
                )


ITERATORS = {"hf-circuitvqa": iter_circuitvqa}
