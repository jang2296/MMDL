"""Load all subject configurations from the verified local validation snapshot."""

import ast
import io
import re
from collections import Counter
from pathlib import Path

from datasets import Image, load_dataset
from PIL import Image as PILImage

from mmdl.runtime.artifacts import read_json, sha256_file
from mmdl.runtime.contracts import DATA_ID, DATA_REVISION, SUBJECTS


def load_validation(data_root):
    data_root = Path(data_root)
    manifest = read_json(data_root / "manifest.json")
    if manifest["repo_id"] != DATA_ID or manifest["revision"] != DATA_REVISION:
        raise ValueError("Dataset manifest is not the required MMMU validation snapshot")
    records = {item["path"]: item for item in manifest["files"]}
    datasets, ids, types = {}, [], Counter()
    for subject in SUBJECTS:
        filename = f"{subject}/validation-00000-of-00001.parquet"
        path = data_root / filename
        record = records[filename]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Dataset integrity failure: {subject}")
        # Each original config is loaded separately. Explicit files prevent dev/test downloads.
        dataset = load_dataset("parquet", name=subject, split="validation",
                               data_files={"validation": str(path)},
                               cache_dir=str(data_root / ".arrow"))
        if len(dataset) != 30:
            raise ValueError(f"Expected 30 validation samples for {subject}, got {len(dataset)}")
        for key in dataset.column_names:
            if re.fullmatch(r"image_[1-7]", key):
                dataset = dataset.cast_column(key, Image(decode=False))
        subject_ids = list(dataset["id"])
        if any(not item.startswith(f"validation_{subject}_") for item in subject_ids):
            raise ValueError(f"Unexpected sample identity: {subject}")
        ids.extend(subject_ids)
        types.update(dataset["question_type"])
        datasets[subject] = dataset
    if len(ids) != 900 or len(set(ids)) != 900:
        raise ValueError("Coverage must be 900 unique IDs")
    if not set(types) <= {"multiple-choice", "open"}:
        raise ValueError(f"Unknown question types: {sorted(types)}")
    return datasets, {"total": 900, "subjects": 30, "question_types": dict(types),
                       "loader": "per-subject named Parquet config from verified original snapshot",
                       "ordered_ids": sorted(ids)}


def separate_sample(row, subject):
    """Allowlist inference fields. Gold stays with the scoring caller."""
    options = row["options"]
    if isinstance(options, str):
        options = ast.literal_eval(options)
    if not isinstance(options, list) or any(not isinstance(x, str) for x in options):
        raise ValueError("Options must be the original ordered list of strings")
    kind = row["question_type"]
    if kind == "multiple-choice" and not 2 <= len(options) <= 26:
        raise ValueError("Invalid multiple-choice option count")
    if kind not in {"multiple-choice", "open"}:
        raise ValueError("Unknown question type")
    sid = row["id"]
    if not re.fullmatch(r"validation_[A-Za-z_]+_\d+", sid):
        raise ValueError("Unsafe sample ID")
    sample = {"id": sid, "subject": subject, "question_type": kind,
              "question": row["question"], "options": options}
    hint = row.get("hint")
    if isinstance(hint, str) and hint.strip():
        sample["hint"] = hint
    gold = {"answer": row["answer"], "gold_explanation": row.get("explanation") or None}
    images, references = [], []
    for index in range(1, 8):
        entry = row.get(f"image_{index}")
        if entry is None:
            continue
        if entry.get("bytes") is not None:
            image = PILImage.open(io.BytesIO(entry["bytes"]))
        elif entry.get("path"):
            image = PILImage.open(entry["path"])
        else:
            raise ValueError("Empty image data")
        image.load()
        images.append(image.convert("RGB"))
        references.append({"field": f"image_{index}", "marker": f"<image {index}>",
                           "content_index": len(images) - 1, "original_size": list(image.size)})
    if not images:
        raise ValueError("MMMU sample has no actual image")
    text = sample["question"] + "\n" + "\n".join(options)
    mentioned = {int(x) for x in re.findall(r"<image\s+(\d+)>", text)}
    available = {int(x["field"].split("_")[1]) for x in references}
    if not mentioned <= available:
        raise ValueError("Question or options reference an absent image")
    # Do not silently renumber <image n> or change original image order.
    if sorted(available) != list(range(1, len(images) + 1)):
        raise ValueError("Noncontiguous image slots require explicit source review")
    sample["image_references"] = references
    return sample, images, gold
