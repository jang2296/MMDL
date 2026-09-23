"""CPU-only comparison of the three Qwen3-VL image preprocessing protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from mmdl.evaluation.backends.input_preparation import prepare_protocol_inputs
from mmdl.evaluation.prompt import build_messages
from mmdl.runtime.artifacts import read_json, resolve_image

PROTOCOLS = {
    "A": {"min_pixels": 262144, "max_pixels": 1310720, "preprocessing": "hf_processor"},
    "B": {"min_pixels": 262144, "max_pixels": 1310720, "preprocessing": "qwen_vl_utils_0_0_14"},
    "C": {"min_pixels": 1003520, "max_pixels": 4014080, "preprocessing": "qwen_vl_utils_0_0_14"},
}


def _sha(values: Any) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sample_paths(run_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted((run_dir / "samples").glob("*.json"))
    return paths if limit is None else paths[:limit]


def _prepare_sample(sample: dict[str, Any], run_dir: Path, processor: Any, config: Any,
                    image_cfg: dict[str, Any]) -> dict[str, Any]:
    processor.image_processor.size = {"shortest_edge": image_cfg["min_pixels"],
                                      "longest_edge": image_cfg["max_pixels"]}
    images = [Image.open(resolve_image(run_dir, ref)).convert("RGB") for ref in sample["images"]]
    raw_hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    messages = build_messages(sample, images, Path(__file__).resolve().parents[3])
    prepared, processed = prepare_protocol_inputs(processor, config, messages, images, image_cfg)
    return {
        "id": sample["id"],
        "raw_image_hashes": raw_hashes,
        "processed_image_hashes": [hashlib.sha256(image.tobytes()).hexdigest() for image in processed],
        "image_sizes": [list(image.size) for image in processed],
        "input_tokens": prepared["input_tokens"],
        "prompt_token_ids_sha256": _sha(prepared["prompt_token_ids"]),
        "tensor_identities": prepared["input_tensors"],
        "image_grid_thw": prepared["image_grid_thw"],
        "input_sha256": prepared["input_sha256"],
        "reference_input_tokens": sample.get("input_tokens"),
        "reference_input_sha256": sample.get("input_sha256"),
        "reference_image_grid_thw": sample.get("image_grid_thw"),
    }


def audit_inputs(model_path: str | Path, run_dir: str | Path, output: str | Path,
                 limit: int | None = None) -> dict[str, Any]:
    """Write append-only JSONL records and a compact comparison summary."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    run_dir = Path(run_dir).resolve()
    paths = _sample_paths(run_dir, limit)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not paths or (limit is None and len(paths) != 900):
        raise ValueError("Expected 900 sample records, or an explicit positive smoke limit")
    import torch
    from transformers import AutoConfig, AutoProcessor

    torch.set_num_threads(4)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    output.mkdir(parents=True)
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in PROTOCOLS}
    for index, path in enumerate(paths, 1):
        sample = read_json(path)
        for name, image_cfg in PROTOCOLS.items():
            records[name].append(_prepare_sample(sample, run_dir, processor, config, image_cfg))
        if index % 30 == 0:
            print(f"CPU input audit: {index}/{len(paths)}", flush=True)

    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(len(records["A"])):
            stream.write(json.dumps({name: records[name][index] for name in records}, ensure_ascii=False) + "\n")
    differences = {}
    for name in ("B", "C"):
        differences[name] = sum(records["A"][i]["input_sha256"] != records[name][i]["input_sha256"]
                                for i in range(len(records["A"])))
    max_input = max((row["input_tokens"] for row in records["C"]), default=0)
    recommended_context = ((max_input + 32768 + 4095) // 4096) * 4096
    summary = {"samples": len(records["A"]), "protocols": PROTOCOLS,
               "differences_from_A": differences, "max_input_tokens_C": max_input,
               "A_reference_mismatches": {
                   key: sum(row[key] != row["reference_" + key] for row in records["A"])
                   for key in ("input_tokens", "input_sha256", "image_grid_thw")},
               "recommended_context": recommended_context}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(audit_inputs(args.model_path, args.run_dir, args.output, args.limit)))


if __name__ == "__main__":
    main()
