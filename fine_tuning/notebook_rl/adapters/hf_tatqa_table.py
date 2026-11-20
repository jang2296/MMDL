"""Original human table-only arithmetic/count labels over pinned Cauldron table images.

Native author JSON is a sibling NExTplusplus__TAT-QA/dataset_raw directory under the
explicit raw root. Compound paragraph/question/scale identity binds the mirror to
native table/question IDs; this is provenance, not a visual table-quality verdict.
No paragraph or derivation is copied into a Candidate. Campaign guards and fixed
hash selection are applied separately, before materialization.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow.parquet as pq

from ..schema import AnswerType
from .hf_common import Candidate, SourceSpec, family_digest, split_for, strip_instruction_suffix

SOURCE = "hf-tatqa-table"
AUTHOR_REPO = "NExTplusplus/TAT-QA"
AUTHOR_REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
AUTHOR_SHA256 = "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69"
MIRROR_SHARD = "tat_qa/train-00000-of-00001-af106c1f4db5a01b.parquet"
MIRROR_SHA256 = "2e840743d2a9c8d2d4dc6f26c6f7741210bee2f673596dfe99f3b2b0ec05470c"
SPECS = {SOURCE: SourceSpec(
    SOURCE, "HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605", "CC-BY-4.0", "HUMAN",
    "TAT-QA original human table-only labels; author train JSON@" + AUTHOR_REVISION + "; pinned Cauldron images",
    "table",
)}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?\Z")
_SCALES = frozenset(("", "percent", "thousand", "million", "billion"))


def _checked(path: Path, expected: str) -> None:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("TAT-QA table: regular pinned source files required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("TAT-QA table: pinned source SHA256 mismatch")


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def question_text(question: dict) -> str:
    text, scale = question["question"].strip(), question["scale"]
    return f"{text}\nAnswer scale should be: {scale}." if scale else text


def _gold(question: dict) -> str | None:
    value = question["answer"]
    if type(value) is bool or not isinstance(value, (str, int, float)):
        return None
    if type(value) is float and not math.isfinite(value):
        return None
    text = str(value).strip()
    if _NUMBER.fullmatch(text) is None:
        return None
    if question["answer_type"] == "count":
        return text if re.fullmatch(r"\d+", text) else None
    return format(Decimal(text), "f")


def iter_table_records(repo_dir: Path, stats: Counter | None = None) -> Iterator[tuple[Candidate, dict]]:
    """Yield eligible native QAs and private ancestry; never select using model results."""
    stats = Counter() if stats is None else stats
    root = Path(repo_dir).absolute()
    author = root.parent / "NExTplusplus__TAT-QA/dataset_raw/tatqa_dataset_train.json"
    shard = root / MIRROR_SHARD
    _checked(author, AUTHOR_SHA256)
    _checked(shard, MIRROR_SHA256)
    entries = json.loads(author.read_text())
    tails: dict[tuple[str, ...], list[tuple[int, dict]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        tails[tuple(_norm(question_text(q)) for q in entry["questions"][1:])].append((index, entry))
    row_index = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=["images", "texts"]):
        for row in batch.to_pylist():
            index = row_index
            row_index += 1
            if len(row["images"] or ()) != 1 or not row["texts"]:
                stats["invalid_image_or_dialogue"] += 1
                continue
            raw = row["images"][0].get("bytes")
            if not isinstance(raw, bytes) or not raw:
                continue
            users = [strip_instruction_suffix(turn["user"] or "") for turn in row["texts"]]
            matches = []
            for original_index, entry in tails.get(tuple(_norm(q) for q in users[1:]), ()):
                questions = entry["questions"]
                if len(questions) != len(users):
                    continue
                paragraphs = "\n".join(p["text"] for p in sorted(entry["paragraphs"], key=lambda p: p["order"]))
                if _norm(users[0]) == _norm(paragraphs + "\n" + question_text(questions[0])):
                    matches.append((original_index, entry))
            if len(matches) != 1:
                stats["unmatched_or_ambiguous_native_context"] += 1
                continue
            original_index, entry = matches[0]
            raw_sha = hashlib.sha256(raw).hexdigest()
            table_uid = entry["table"]["uid"]
            for turn_index, (turn, question) in enumerate(zip(row["texts"], entry["questions"], strict=True)):
                if (question["answer_from"] != "table" or question["answer_type"] not in ("arithmetic", "count")
                        or question["rel_paragraphs"] not in ([], None) or question["scale"] not in _SCALES):
                    continue
                gold = _gold(question)
                label = (turn["assistant"].rpartition("Answer:")[-1] if "Answer:" in turn["assistant"]
                         else turn["assistant"]).strip().removesuffix(".").strip()
                try:
                    agrees = gold is not None and _NUMBER.fullmatch(label) and Decimal(gold) == Decimal(label)
                except InvalidOperation:
                    agrees = False
                if not agrees:
                    stats["native_mirror_label_mismatch"] += 1
                    continue
                assert gold is not None
                native_id = f"tat_qa/{shard.stem}:{index}:{turn_index}"
                kind: AnswerType = "int" if question["answer_type"] == "count" else "decimal"
                candidate = Candidate(SOURCE, native_id, "Accounting", question_text(question), (), kind, gold,
                                      {} if kind == "int" else {"rel_tol": "0.01"}, (raw,), table_uid)
                legacy = replace(candidate, source="hf-tatqa", family_key=raw_sha)
                metadata = {"original_table_uid": table_uid, "original_question_uid": question["uid"],
                    "original_entry_index": original_index, "mirror_row_index": index, "mirror_turn_index": turn_index,
                    "canonical_source": "hf-tatqa", "canonical_native_id": native_id,
                    "canonical_family_digest": family_digest(legacy), "canonical_split": split_for(legacy),
                    "image_raw_sha256": raw_sha, "native_answer_type": question["answer_type"],
                    "native_answer_from": question["answer_from"], "native_scale": question["scale"],
                    "author_revision": AUTHOR_REVISION, "author_file_sha256": AUTHOR_SHA256,
                    "mirror_file_sha256": MIRROR_SHA256, "report_family_status": "NOT_PROVIDED_BY_AUTHOR_TRAIN_JSON"}
                yield candidate, metadata


def admission_block_reason(candidate: Candidate, origin: dict, *, blocked_table_uids: set[str],
                           blocked_families: set[tuple[str, str]], old_native_ids: set[tuple[str, str]],
                           blocked_raw_images: set[str], blocked_png_images: set[str], blocked_pixels: set[str],
                           png_sha256: str, pixel_sha256: str) -> str | None:
    """Campaign source/exposure/image guards; a new namespace never erases its ancestor."""
    raw_sha = hashlib.sha256(candidate.images[0]).hexdigest()
    legacy = replace(candidate, source="hf-tatqa", family_key=raw_sha)
    if (candidate.source != SOURCE or candidate.family_key != origin["original_table_uid"]
            or origin["canonical_source"] != "hf-tatqa"
            or origin["canonical_native_id"] != candidate.native_id
            or origin["canonical_family_digest"] != family_digest(legacy)
            or origin["canonical_split"] != split_for(legacy) or origin["image_raw_sha256"] != raw_sha):
        raise ValueError("TAT-QA table: inconsistent canonical ancestry")
    if split_for(candidate) != origin["canonical_split"]:
        return "ANCESTRAL_SPLIT_DISAGREEMENT"
    if origin["original_table_uid"] in blocked_table_uids:
        return "PROTECTED_CANONICAL_TABLE_UID"
    if ("hf-tatqa", origin["canonical_family_digest"]) in blocked_families:
        return "PROTECTED_ANCESTRAL_FAMILY"
    if ("hf-tatqa", origin["canonical_native_id"]) in old_native_ids:
        return "OLD_CANONICAL_NATIVE_ID"
    if raw_sha in blocked_raw_images or png_sha256 in blocked_png_images or pixel_sha256 in blocked_pixels:
        return "PROTECTED_SOURCE_IMAGE_OR_PIXEL"
    return None


def iter_tatqa_table(repo_dir: Path) -> Iterator[Candidate]:
    for candidate, origin in iter_table_records(repo_dir):
        if split_for(candidate) == origin["canonical_split"]:
            yield candidate


ITERATORS = {SOURCE: iter_tatqa_table}
