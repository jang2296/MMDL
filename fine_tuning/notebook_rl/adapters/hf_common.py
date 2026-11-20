"""Pinned Hugging Face train-split rows -> verifier-only gold items.

Labels come from the upstream human annotation or from a program over upstream metadata.
No upstream solution/rationale text is ever copied into an actor item. Gold stays in the
VerifierItem only. Nothing here reads MMMU/MMMU-Pro.
"""
from __future__ import annotations

import hashlib
import io
import random
import re
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image

from ..generators.common import GeneratedProblem, matches_split, png_chunks, structure_hash
from ..schema import ActorItem, AnswerType, Lineage, Mode, Split, VerifierItem, to_json

MAX_STORED_PIXELS = 2_000_000          # storage cap; the processor still applies the protocol budget
SUBJECTS = (
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials", "Math",
    "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
)


@dataclass(frozen=True)
class SourceSpec:
    source: str                     # registry id, e.g. "hf-scienceqa"
    repo: str                       # HF dataset repo id
    revision: str                   # full 40-hex commit sha (never "main")
    license: str
    label_provenance: Literal["HUMAN", "PROGRAM"]
    upstream_origin: str            # who made the labels, e.g. "IXL/ScienceQA annotators"
    family: str                     # ActorItem.family string, [a-z0-9_]+


@dataclass(frozen=True)
class Candidate:
    source: str
    native_id: str                  # stable upstream id (row index, sub-question index, file name ...)
    subject: str                    # one of SUBJECTS; used only for quota planning, never shown
    question: str
    choices: tuple[str, ...]        # empty unless answer_type == "mcq"
    answer_type: AnswerType
    gold: str                       # mcq: single letter "A".."J"
    accept: dict = field(default_factory=dict)
    images: tuple[bytes, ...] = ()  # upstream encoded image bytes, 1 or 2
    family_key: str = ""            # everything sharing this key lands in one split


def normalize_image(raw: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if getattr(image, "n_frames", 1) > 1:
                raise ValueError("image: animated or multi-frame images are not accepted")
            rgba = image.convert("RGBA")
    except (OSError, Image.DecompressionBombError) as exc:  # corrupt/undecodable upstream image
        raise ValueError(f"image: undecodable upstream image ({type(exc).__name__})") from exc
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    width, height = rgb.size
    if width * height > MAX_STORED_PIXELS:
        scale = (MAX_STORED_PIXELS / (width * height)) ** 0.5
        rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    out = io.BytesIO()
    rgb.save(out, format="PNG", optimize=False)
    png = out.getvalue()
    png_chunks(png)
    return png


def pixel_sha256(png: bytes) -> str:
    with Image.open(io.BytesIO(png)) as image:
        rgb = image.convert("RGB")
    return hashlib.sha256(b"%dx%d:" % rgb.size + rgb.tobytes()).hexdigest()


def family_digest(c: Candidate) -> str:
    return structure_hash({"source": c.source, "family_key": c.family_key or c.native_id})


def split_for(c: Candidate) -> Split:
    digest = family_digest(c)
    splits: tuple[Split, ...] = ("train", "dev", "sealed", "style_sealed")
    hits = [s for s in splits if matches_split(digest, s, "shift" if s == "style_sealed" else "default")]
    if len(hits) != 1:
        raise ValueError("split: expected exactly one matching split")
    return hits[0]


def mcq_from_answer(question: str, options: list[str], answer_index: int) -> tuple[str, tuple[str, ...], str]:
    if not 2 <= len(options) <= 10 or any(type(x) is not str or not x.strip() for x in options):
        raise ValueError("mcq: 2..10 nonempty string options required")
    stripped = tuple(x.strip() for x in options)
    if len(set(stripped)) != len(stripped):
        raise ValueError("mcq: duplicate options")
    if type(answer_index) is not int or not 0 <= answer_index < len(stripped):
        raise ValueError("mcq: answer index out of range")
    return question.strip(), stripped, "ABCDEFGHIJ"[answer_index]


def program_mcq(gold_label: str, pool: list[str], *, key: str, n_options: int = 4) -> tuple[tuple[str, ...], str]:
    if not 2 <= n_options <= 10:
        raise ValueError("program_mcq: n_options must be 2..10")
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))
    available = sorted(set(pool) - {gold_label})
    count = min(n_options - 1, len(available))
    if count < 1:
        raise ValueError("program_mcq: at least one distractor is required")
    choices = rng.sample(available, count)
    position = rng.randrange(count + 1)
    choices.insert(position, gold_label)
    return tuple(choices), "ABCDEFGHIJ"[position]


# The exact Cauldron instruction lines (all 18 seen in ai2d/mapqa/finqa/tat_qa/plotqa train).
# An allowlist, so a real question line containing "short"/"answer" is never stripped.
_CAULDRON_SUFFIXES = frozenset((
    "Give a very brief answer.", "Your answer should be very brief.", "Keep it short and to the point.",
    "Offer a very short reply.", "Your answer should be compact.", "Provide a short and direct response.",
    "Your response must be concise.", "Provide a succinct answer.", "Make the answer very short.",
    "Ensure brevity in your answer.", "Offer a terse response.", "Quick response, please.",
    "Concise answer only.", "Keep it brief.", "Answer briefly.", "Write a very short answer.",
    "Short answer required.", "Be succinct.", "Answer with the letter.",
))


def strip_instruction_suffix(text: str) -> str:
    head, sep, last = text.strip().rpartition("\n")
    return head.strip() if sep and last.strip() in _CAULDRON_SUFFIXES else text.strip()


def parse_letter_choices(user: str) -> tuple[str, tuple[str, ...]] | None:
    head, sep, rest = user.strip().partition("\nChoices:\n")
    if not sep or not head.startswith("Question: ") or not head[len("Question: "):].strip():
        return None
    lines = rest.split("\n")
    choices: list[str] = []
    while lines:
        found = re.fullmatch(r"([A-J])\. (.+)", lines[0].strip())
        if found is None or found.group(1) != "ABCDEFGHIJ"[len(choices)]:
            break
        choices.append(found.group(2).strip())
        lines.pop(0)
    if len(choices) < 2 or len(lines) != 1 or not lines[0].strip():
        return None
    return head[len("Question: "):].strip(), tuple(choices)


def to_problem(c: Candidate, spec: SourceSpec, *, index: int, mode: Mode = "plain") -> GeneratedProblem:
    if c.source != spec.source:
        raise ValueError("to_problem: candidate source differs from spec")
    if c.subject not in SUBJECTS:
        raise ValueError("to_problem: unknown subject")
    if not 1 <= len(c.images) <= 2:
        raise ValueError("to_problem: 1 or 2 images required")
    if type(index) is not int or index < 0:
        raise ValueError("to_problem: nonnegative integer index required")
    if c.answer_type == "mcq":
        if not 2 <= len(c.choices) <= 10 or len(c.gold) != 1 or c.gold not in "ABCDEFGHIJ"[:len(c.choices)]:
            raise ValueError("to_problem: mcq needs 2..10 choices and a gold letter among them")
    elif c.choices:
        raise ValueError("to_problem: choices require mcq answer type")
    images = tuple(normalize_image(x) for x in c.images)
    digest = family_digest(c)
    split = split_for(c)
    pid = f"{spec.source}:{spec.family}:{split}:{index:06d}:{digest}"
    paths = tuple(f"{spec.source}/{split}/{index:06d}-{digest}-{i}.png" for i in range(len(images)))
    actor = ActorItem(pid, spec.source, spec.family, mode, c.question, c.choices, c.answer_type,
                      paths, tuple(hashlib.sha256(x).hexdigest() for x in images))
    verifier = VerifierItem(pid, c.answer_type, c.gold, dict(c.accept), c.choices, {}, {}, (), (), ())
    lineage = Lineage(pid, spec.source, spec.revision, spec.license, spec.label_provenance,
                      f"hf:{spec.repo}@{spec.revision}", spec.upstream_origin, None, None,
                      f"{spec.source}-row-v1", digest, split, ())
    for record in (actor, verifier, lineage):
        to_json(record)
    return GeneratedProblem(actor, verifier, lineage, images)


def candidate_order(c: Candidate, seed: int) -> str:
    """Deterministic, label-blind selection order."""
    return hashlib.sha256(f"{seed}:{c.source}:{c.native_id}".encode()).hexdigest()
