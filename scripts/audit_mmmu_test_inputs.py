"""CPU audit: run the frozen low-res MMMU test input path over all items and check the context length."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mmdl.runtime.artifacts import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/eval/mmmu_test_lowres2k_v8.yaml"
_STATE: dict[str, Any] = {}


def _measure(keys: list[tuple[str, int]]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoProcessor

    from mmdl.evaluation.backends.input_preparation import prepare_protocol_inputs
    from mmdl.evaluation.datasets.mmmu_test import separate_sample
    from mmdl.evaluation.prompt import build_messages

    torch.set_num_threads(1)
    model_path, image_cfg = _STATE["model_path"], _STATE["cfg"]["image"]
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    processor.image_processor.size = {"shortest_edge": image_cfg["min_pixels"], "longest_edge": image_cfg["max_pixels"]}
    rows = []
    for subject, index in keys:
        sample, images, _ = separate_sample(_STATE["datasets"][subject][index], subject)
        item, _ = prepare_protocol_inputs(processor, config, build_messages(sample, images, ROOT), images, image_cfg)
        rows.append({"id": sample["id"], "images": len(images), "input_tokens": item["input_tokens"]})
    return rows


def summarize(rows: list[dict[str, Any]], max_new_tokens: int, max_model_len: int) -> dict[str, Any]:
    longest = max(rows, key=lambda row: (row["input_tokens"], row["id"]))
    tokens = sorted(row["input_tokens"] for row in rows)
    over = sorted(row["id"] for row in rows if row["input_tokens"] + max_new_tokens > max_model_len)
    return {"items": len(rows), "max_input_tokens": longest["input_tokens"], "max_input_id": longest["id"],
            "mean_input_tokens": sum(tokens) / len(tokens), "p99_input_tokens": tokens[math.ceil(.99 * len(tokens)) - 1],
            "max_images": max(row["images"] for row in rows),
            "image_count_distribution": dict(sorted(Counter(row["images"] for row in rows).items())),
            "max_new_tokens": max_new_tokens, "max_model_len": max_model_len,
            "max_input_plus_output": longest["input_tokens"] + max_new_tokens,
            "over_context_ids": over, "fits": not over}


def audit(data_root: Path, model_path: Path, output: Path, workers: int, limit: int | None) -> dict[str, Any]:
    from mmdl.evaluation.datasets.mmmu_test import load_test

    if output.exists():
        raise FileExistsError(f"Receipt already exists: {output}")
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    datasets, coverage = load_test(data_root)
    keys = [(subject, index) for subject, dataset in datasets.items() for index in range(len(dataset))]
    keys = keys[:limit] if limit else keys
    _STATE.update(datasets=datasets, cfg=cfg, model_path=str(model_path))
    chunks = [keys[start::workers] for start in range(workers)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        rows = [row for part in pool.map(_measure, chunks) for row in part]
    receipt = {"reportable": False, "scope": "CPU processor reference only; no model weights or generation",
               "protocol_id": cfg["protocol_id"], "protocol_sha256": sha256_file(PROTOCOL),
               "image": cfg["image"], "data_manifest_sha256": sha256_file(data_root / "manifest.json"),
               "processor_files": {p.name: sha256_file(p) for p in sorted(model_path.glob("*.json"))},
               "registered_items": coverage["total"], "complete": len(rows) == coverage["total"],
               **summarize(rows, cfg["generation"]["max_new_tokens"], cfg["execution"]["max_model_len"])}
    write_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Pinned base snapshot (processor files)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    receipt = audit(args.data_root, args.model_path, args.output, args.workers, args.limit)
    print(json.dumps({key: value for key, value in receipt.items() if key != "processor_files"}))


if __name__ == "__main__":
    main()
