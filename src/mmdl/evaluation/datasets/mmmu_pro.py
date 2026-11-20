"""Pinned MMMU-Pro Standard 10-way test; answer-side images never enter inference."""

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Image, load_dataset

from mmdl.evaluation.datasets.mmmu import load_image, parse_options
from mmdl.evaluation.prompt import _referenced_image_numbers, build_prompt
from mmdl.runtime.artifacts import digest, read_json, sha256_file
from mmdl.runtime.contracts import PRO_ID, PRO_REVISION, SUBJECTS

CONFIG = "standard (10 options)"
TOTAL = 1730
SHARDS = [f"{CONFIG}/test-0000{i}-of-00002.parquet" for i in range(2)]


def subject_from_id(sample_id):
    for subject in sorted(SUBJECTS, key=len, reverse=True):
        if re.fullmatch(rf"(?:test|validation)_{re.escape(subject)}_\d+", sample_id):
            return subject
    raise ValueError(f"Unknown MMMU-Pro sample identity: {sample_id!r}")


def inference_fields(row, subject):
    """Validate the source mapping without inspecting gold or explanations."""
    sid = row["id"]
    if subject_from_id(sid) != subject:
        raise ValueError(f"MMMU-Pro ID/subject mismatch: {sid}")
    options = parse_options(row["options"])
    if not 2 <= len(options) <= 12 or not isinstance(row["question"], str):
        raise ValueError(f"Invalid MMMU-Pro question/options: {sid}")
    sample = dict(id=sid, subject=subject, question_type="multiple-choice",
                  question=row["question"], options=options)
    slots = sorted(_referenced_image_numbers(sample))
    if not slots or any(index < 1 or index > 7 for index in slots):
        raise ValueError(f"Invalid MMMU-Pro image references: {sid}")
    for index in slots:
        entry = row.get(f"image_{index}")
        if not isinstance(entry, dict) or not (entry.get("bytes") or entry.get("path")):
            raise ValueError(f"Question or choices reference an absent image: {sid}, image_{index}")
    return sample, slots


def separate_sample(row, subject):
    sample, slots = inference_fields(row, subject)
    answer = row["answer"]
    if answer not in [chr(65 + index) for index in range(len(sample["options"]))]:
        raise ValueError(f"Gold is not a valid original option: {sample['id']}")
    images, references = [], []
    for index in slots:
        image = load_image(row[f"image_{index}"])
        references.append(dict(field=f"image_{index}", marker=f"<image {index}>",
                               content_index=len(images), original_size=list(image.size),
                               pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        images.append(image)
    sample["image_references"] = references
    sample["original_image_slots"] = [index for index in range(1, 8)
                                       if row.get(f"image_{index}") is not None]
    sample["used_image_slots"] = slots
    return sample, images, dict(answer=answer, gold_explanation=row.get("explanation") or None)


def build_messages(sample, images, root):
    """Preserve original slot labels, including sparse references, in unchanged P0 text."""
    refs = sample["image_references"]
    numbers = _referenced_image_numbers(sample)
    if len(refs) != len(images) or not images or len(images) > 7:
        raise ValueError("Pro image payload does not match its slot mapping")
    slots = []
    for index, ref in enumerate(refs):
        match = re.fullmatch(r"image_([1-7])", ref["field"])
        if match is None:
            raise ValueError("Invalid Pro image slot")
        slot = int(match[1])
        if ref["marker"] != f"<image {slot}>" or ref["content_index"] != index:
            raise ValueError("Invalid Pro image content mapping")
        slots.append(slot)
    if slots != sorted(numbers) or len(set(slots)) != len(slots):
        raise ValueError("Pro images must follow referenced unique source slot order")
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": build_prompt(sample, root)})
    return [{"role": "user", "content": content}]


def load_test(data_root):
    data_root = Path(data_root)
    manifest = read_json(data_root / "manifest.json")
    pinned = read_json(Path(__file__).resolve().parents[4] / "manifests/datasets/mmmu_pro.json")
    if manifest.get("repo_id") != PRO_ID or manifest.get("revision") != PRO_REVISION:
        raise ValueError("Dataset is not the fixed MMMU-Pro Standard 10-way snapshot")
    if manifest["files"] != pinned["files"]:
        raise ValueError("MMMU-Pro file contract differs from the pinned manifest")
    records = {item["path"]: item for item in pinned["files"]}
    for filename in ["README.md"] + SHARDS:
        path, record = data_root / filename, records[filename]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"MMMU-Pro source integrity failure: {filename}")
    dataset = load_dataset("parquet", name=CONFIG, split="test",
                           data_files={"test": [str(data_root / name) for name in SHARDS]},
                           cache_dir=str(data_root / ".arrow"))
    for key in dataset.column_names:
        if re.fullmatch(r"image_[1-7]", key):
            dataset = dataset.cast_column(key, Image(decode=False))
    if len(dataset) != TOTAL:
        raise ValueError(f"MMMU-Pro requires {TOTAL} rows, got {len(dataset)}")
    indices, id_subjects, option_counts = defaultdict(list), {}, Counter()
    original_counts, used_counts, excluded, source_slots = Counter(), Counter(), {}, {}
    for index, row in enumerate(dataset):
        sid = row["id"]
        subject = subject_from_id(sid)
        if sid in id_subjects:
            raise ValueError(f"Duplicate MMMU-Pro ID: {sid}")
        try:
            sample, slots = inference_fields(row, subject)
            if row["answer"] not in [chr(65 + n) for n in range(len(sample["options"]))]:
                raise ValueError("Gold is outside original choices")
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid MMMU-Pro row {sid}: {exc}") from exc
        original = [n for n in range(1, 8) if row.get(f"image_{n}") is not None]
        extra = sorted(set(original) - set(slots))
        if extra:
            excluded[sid] = extra
        source_slots[sid] = slots
        indices[subject].append(index)
        id_subjects[sid] = subject
        option_counts[len(sample["options"])] += 1
        original_counts[len(original)] += 1
        used_counts[len(slots)] += 1
    if set(indices) != set(SUBJECTS) or any(not 50 <= len(rows) <= 60 for rows in indices.values()):
        raise ValueError("Unexpected MMMU-Pro subject coverage")
    ordered_ids = sorted(id_subjects)
    coverage = dict(total=TOTAL, subjects=len(indices), question_types={"multiple-choice": TOTAL},
                    loader="verified MMMU-Pro Standard 10-way two-shard test",
                    ordered_ids=ordered_ids, id_subjects=id_subjects,
                    subject_counts={subject: len(rows) for subject, rows in indices.items()},
                    option_counts=dict(option_counts), original_image_counts=dict(original_counts),
                    used_image_counts=dict(used_counts), excluded_image_slots=excluded,
                    used_image_slots=source_slots, source_order_sha256=digest(list(dataset["id"])))
    return {subject: dataset.select(indices[subject]) for subject in SUBJECTS}, coverage
