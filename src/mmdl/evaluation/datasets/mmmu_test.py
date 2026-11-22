"""Pinned MMMU test split (10,500); every parquet gold must equal the pinned official answer file."""

import ast
import json
import re
from collections import Counter
from pathlib import Path

from datasets import Image, load_dataset

from mmdl.evaluation.datasets import mmmu
from mmdl.runtime.artifacts import read_json, sha256_file
from mmdl.runtime.contracts import DATA_ID, DATA_REVISION, SUBJECTS, TEST_ANSWER_SOURCE

TOTAL = 10500
ANSWER_FILE = "answer_dict_test.json"
PINNED = Path(__file__).resolve().parents[4] / "manifests/datasets/mmmu_test.json"


def subject_from_id(sample_id):
    for subject in sorted(SUBJECTS, key=len, reverse=True):
        if re.fullmatch(rf"test_{re.escape(subject)}_\d+", sample_id):
            return subject
    raise ValueError(f"Unknown MMMU test sample identity: {sample_id!r}")


def separate_sample(row, subject):
    """The validation allowlist, applied to a checked test ID.

    mmmu.py stays byte-identical because it is hashed into the frozen Pro/FT protocol identity.
    """
    sid = row["id"]
    if subject_from_id(sid) != subject:
        raise ValueError(f"MMMU test ID/subject mismatch: {sid}")
    sample, images, gold = mmmu.separate_sample(dict(row, id="validation" + sid[len("test"):]), subject)
    sample["id"] = sid
    answer = gold["answer"]
    if sample["question_type"] == "open" and isinstance(answer, str) and answer.lstrip().startswith("["):
        # The pinned official answer file stores these 32 open golds as lists of accepted answers; V8 accepts any.
        value = ast.literal_eval(answer)
        if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
            raise ValueError(f"Unexpected list gold form: {sid}")
        gold["answer"] = value
    return sample, images, gold


def official_gold_matches(parquet_answer, official):
    """Lists in the official file are stored as their Python repr in the parquet."""
    truth = official["ground_truth"]
    if isinstance(truth, list):
        try:
            return ast.literal_eval(parquet_answer) == truth
        except (ValueError, SyntaxError):
            return False
    return parquet_answer == truth


def _verified(path, record):
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"MMMU test source integrity failure: {record['path']}")


def load_test(data_root, pinned_path=PINNED):
    data_root = Path(data_root)
    manifest = read_json(data_root / "manifest.json")
    pinned = read_json(pinned_path)
    if manifest.get("repo_id") != DATA_ID or manifest.get("revision") != DATA_REVISION:
        raise ValueError("Dataset manifest is not the pinned MMMU snapshot")
    answer = pinned["answer_file"]
    if (manifest["files"] != pinned["files"] or manifest.get("answer_file") != answer
            or answer["path"] != ANSWER_FILE
            or any(answer[key] != TEST_ANSWER_SOURCE[key] for key in ("url", "sha256", "commit"))):
        raise ValueError("MMMU test file contract differs from the pinned manifest or answer source")
    for record in pinned["files"] + [answer]:
        _verified(data_root / record["path"], record)
    official = json.loads((data_root / ANSWER_FILE).read_text(encoding="utf-8"))
    datasets, id_subjects, types, image_counts = {}, {}, Counter(), Counter()
    for subject in SUBJECTS:
        shards = sorted(r["path"] for r in pinned["files"]
                        if re.fullmatch(rf"{re.escape(subject)}/test-\d{{5}}-of-\d{{5}}\.parquet", r["path"]))
        if not shards:
            raise ValueError(f"No pinned test shard for {subject}")
        dataset = load_dataset("parquet", name=subject, split="test",
                               data_files={"test": [str(data_root / name) for name in shards]},
                               cache_dir=str(data_root / ".arrow"))
        for key in dataset.column_names:
            if re.fullmatch(r"image_[1-7]", key):
                dataset = dataset.cast_column(key, Image(decode=False))
        slots = [key for key in dataset.column_names if re.fullmatch(r"image_[1-7]", key)]
        for index, (sid, kind, gold) in enumerate(zip(dataset["id"], dataset["question_type"], dataset["answer"],
                                                      strict=True)):
            if subject_from_id(sid) != subject or sid in id_subjects:
                raise ValueError(f"Unexpected or duplicate MMMU test ID: {sid}")
            expected = official.get(sid)
            if expected is None or expected.get("question_type") != kind or not official_gold_matches(gold, expected):
                raise ValueError(f"Parquet gold differs from the pinned official answer file: {sid}")
            id_subjects[sid] = subject
        for row in dataset.select_columns(slots):
            image_counts[sum(row[key] is not None for key in slots)] += 1
        types.update(dataset["question_type"])
        datasets[subject] = dataset
    if len(id_subjects) != TOTAL or set(official) != set(id_subjects):
        raise ValueError(f"MMMU test requires exactly the {TOTAL} IDs of the official answer file")
    if not set(types) <= {"multiple-choice", "open"}:
        raise ValueError(f"Unknown question types: {sorted(types)}")
    ordered_ids = sorted(id_subjects)
    return datasets, {"total": TOTAL, "subjects": len(datasets), "question_types": dict(types),
                      "loader": "per-subject pinned MMMU test shards; golds equal the pinned official answer file",
                      "ordered_ids": ordered_ids, "id_subjects": id_subjects,
                      "subject_counts": {subject: len(datasets[subject]) for subject in SUBJECTS},
                      "image_counts": dict(sorted(image_counts.items())),
                      "answer_source": TEST_ANSWER_SOURCE}
