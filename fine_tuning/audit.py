"""Fail-closed provenance, exact/near duplicate and family-split training gate.

Benchmark records enter this module ONLY as exclusion fingerprints. They are
never returned to a generator or serializer. A PASS reports its exact coverage;
it is not proof that pretraining contamination is absent.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from PIL import Image


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def question_text(text):
    # These wrappers are loader syntax, not independent question content.
    value = re.sub(r"<image>", " ", unicodedata.normalize("NFKC", text), flags=re.I)
    value = re.sub(r"^\s*(?:Question\s*:\s*)+", "", value, flags=re.I)
    value = re.split(r"\n\s*(?:Options|Choices)\s*:", value, maxsplit=1, flags=re.I)[0]
    return normalized(value)


def text_hash(text):
    return hashlib.sha256(question_text(text).encode()).hexdigest()


def shingles(text, n=7):
    words = re.findall(r"\w+", question_text(text))
    return {" ".join(words[i:i+n]) for i in range(max(0, len(words)-n+1))}


def image_fingerprint(image):
    rgb = image.convert("RGB")
    pixel = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
    small = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(small.getdata())
    bits = [values[y*9+x] > values[y*9+x+1] for y in range(8) for x in range(8)]
    dhash = sum(int(bit) << i for i, bit in enumerate(bits))
    return {"pixels_sha256": pixel, "dhash": f"{dhash:016x}", "size": list(rgb.size)}


def check_lineage(row, sources):
    if row.get("benchmark_derived") is not False or row.get("source_kind") == "BENCHMARK_DERIVED":
        raise ValueError(f"Benchmark taint or unknown derivation: {row.get('id')}")
    lineage = row.get("lineage")
    if not isinstance(lineage, list) or not lineage:
        raise ValueError("All source parents must be explicitly registered")
    if any(not isinstance(parent, str) or not parent for parent in lineage):
        raise ValueError("Malformed source lineage")
    seen, visiting = set(), set()
    def visit(parent):
        if parent in visiting:
            raise ValueError("Source lineage cycle")
        if parent in seen:
            return
        visiting.add(parent)
        source = sources.get(parent)
        if not source or source.get("benchmark_derived") is not False:
            raise ValueError(f"Unknown or benchmark-tainted ancestor: {parent}")
        if source.get("rights_approved") is not True or source.get("source_known") is not True:
            raise ValueError(f"Unapproved rights/source: {parent}")
        parents = source.get("parents")
        if not isinstance(parents, list):
            raise ValueError("Unknown source ancestry")
        if any(not isinstance(item, str) or not item for item in parents):
            raise ValueError("Malformed source ancestry")
        for item in parents:
            visit(item)
        visiting.remove(parent)
        seen.add(parent)
    for parent in lineage:
        visit(parent)


def check_splits(rows):
    families = defaultdict(set)
    roots = defaultdict(set)
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate record ID")
        ids.add(row["id"])
        families[row["family_id"]].add(row["split"])
        roots[row["root_id"]].add(row["split"])
    if any(len(s) != 1 for s in list(families.values())+list(roots.values())):
        raise ValueError("Root/family leaks across splits")


def image_path(image_root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Image path escapes allowed root")
    root = Path(image_root).resolve(strict=True)
    current = root
    for part in Path(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink image path is forbidden")
    path = current.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Image path escapes allowed root")
    return path


def model_input_record(row):
    """Remove only counted loader placeholders; retain the original source prompt."""
    paths, text = row.get("image_paths"), row.get("user_text")
    if not isinstance(paths, list) or not paths or not isinstance(text, str) or not text.strip():
        raise ValueError("Missing image/input contract")
    original = row.get("original_user_text", text)
    count = original.count("<image>")
    if count not in (0, len(paths)) or text.count("<image>") not in (0, len(paths)):
        raise ValueError("Image placeholder count differs from image count")
    if "<image>" in row.get("target_text", ""):
        raise ValueError("Assistant target contains image placeholders")
    stripped = original.replace("<image>", "").strip()
    if "original_user_text" in row and text != stripped:
        raise ValueError("Normalized input differs from preserved original")
    return dict(row, user_text=stripped, original_user_text=original,
                original_user_text_sha256=hashlib.sha256(original.encode()).hexdigest())


def input_fingerprint(row, image_root):
    prepared = model_input_record(row)
    value = {k: prepared[k] for k in ("id", "user_text", "original_user_text", "target_text", "image_paths")}
    value["image_sha256"] = {name: sha(image_path(image_root, name)) for name in row["image_paths"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def training_input_contract(protocol_path=None):
    """Only the frozen parent or its explicit canonical-bridge child is supported."""
    from .modules import BASE_REVISION
    parent = Path(__file__).with_name("protocol.json")
    parent_sha = "e6c175e1a7d0cf67a2aa7fa7b64c8f85c4a8d491c02579d35e5f3594d0157e4d"
    if sha(parent) != parent_sha:
        raise ValueError("Frozen parent training protocol changed")
    contract = {"processor_revision": BASE_REVISION, "sequence_budget": 4096,
                "min_pixels": 262144, "max_pixels": 1048576, "episode_pixels": 1048576,
                "protocol_sha256": parent_sha, "protocol_id": "nvw-private-bf16-pilot-v1"}
    if protocol_path is None:
        return contract
    path = Path(protocol_path)
    child = json.loads(path.read_text())
    required = {"kind": "FROZEN_CHILD_TRAINING_INPUT_CONTRACT",
                "protocol_id": "nvw-private-canonical-bridge-training-input-v1",
                "parent_protocol_sha256": parent_sha, "processor_revision": BASE_REVISION,
                "dtype": "bfloat16", "quantization": None}
    budgets = {"sequence_budget": 4096, "min_pixels": 262144,
               "max_pixels": 1048576, "episode_pixels": 3145728}
    if (any(k not in child or child[k] != v for k, v in required.items())
            or any(type(child.get(k)) is not int or child[k] != v for k, v in budgets.items())
            or any(child.get(k) is not False for k in (
                "automatic_resolution_precision_or_context_fallback", "benchmark_training_allowed",
                "benchmark_derived_training_allowed", "public_upload_allowed", "data_approval_granted"))):
        raise ValueError("Explicit frozen child training input contract required; CPU-only protocols are not training")
    if sha(path) != "a1b96396f159ad6f761e460c6517172309ea00ee6ecc6fca0168818e09b58ad1":
        raise ValueError("Frozen child training input contract hash changed")
    return {**contract, **budgets, "protocol_id": child["protocol_id"],
            "protocol_sha256": sha(path), "parent_protocol_sha256": parent_sha}


def processor_pixel_budgets(image_processor):
    """Pinned Qwen processing gives the pair of pixel overrides priority over size."""
    overrides = (getattr(image_processor, "min_pixels", None), getattr(image_processor, "max_pixels", None))
    if all(value is not None for value in overrides):
        return overrides
    return (image_processor.size["shortest_edge"], image_processor.size["longest_edge"])


def audit_model_inputs(records_path, image_root, processor, output, sequence_budget=4096,
                       episode_pixels=1048576, processor_revision=None, training_input_protocol=None):
    """Run the actual installed Qwen processor/collator; no model inference/download."""
    import torch
    from transformers import Qwen3VLProcessor
    from .modules import BASE_REVISION
    from .train import AssistantCollator
    if not isinstance(processor, Qwen3VLProcessor):
        raise ValueError("Actual Qwen3VLProcessor required")
    revision = processor_revision or BASE_REVISION
    if revision != BASE_REVISION or Path(processor.tokenizer.name_or_path).name != revision:
        raise ValueError("Processor tokenizer must use the exact pinned local snapshot")
    contract = training_input_contract(training_input_protocol) if training_input_protocol is not None else None
    if contract is not None and (sequence_budget, episode_pixels) != (
            contract["sequence_budget"], contract["episode_pixels"]):
        raise ValueError("Processor budgets differ from explicit child input contract")
    image_processor = processor.image_processor
    min_pixels, max_pixels = processor_pixel_budgets(image_processor)
    if contract is not None and (min_pixels, max_pixels) != (contract["min_pixels"], contract["max_pixels"]):
        raise ValueError("Actual processor per-image budgets differ from child input contract")
    template = processor.chat_template or processor.tokenizer.chat_template
    if not template:
        raise ValueError("Missing actual processor chat template")
    collator = AssistantCollator(processor, Path(image_root), sequence_budget, episode_pixels)
    receipts = {}
    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    for row in read_rows(records_path):
        if row["id"] in receipts:
            raise ValueError("Duplicate input preflight record")
        prepared = model_input_record(row)
        for name in prepared["image_paths"]:
            image_path(image_root, name)
        inputs = collator(prepared)
        if "pixel_values" not in inputs or inputs["image_grid_thw"].shape[0] != len(prepared["image_paths"]):
            raise ValueError("Processor did not produce all image inputs")
        receipts[row["id"]] = {"status": "PASS", "input_sha256": input_fingerprint(row, image_root),
            "sequence_tokens": int(inputs["input_ids"].shape[1]),
            "assistant_tokens": int((inputs["labels"] != -100).sum()),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "tensor_sha256": {key: tensor_hash(inputs[key]) for key in
                              ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}}
    receipt = {"status": "PASS", "kind": "ACTUAL_PROCESSOR_PREFLIGHT",
        "processor_revision": revision, "processor_class": type(processor).__name__,
        "chat_template_sha256": hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest(),
        "sequence_budget": sequence_budget, "episode_pixels": episode_pixels,
        "min_pixels": min_pixels, "max_pixels": max_pixels,
        "producer_sha256": sha(__file__), "records_sha256": sha(records_path), "records": receipts,
        "truncation": False, "training_allowed": False}
    if contract is not None:
        receipt["training_input_contract"] = contract
    dump(output, receipt)
    return receipt


def preflight_records(records_path, image_root, model_path, output, sequence_budget=4096,
                      episode_pixels=1048576, min_pixels=262144, max_pixels=1048576,
                      training_input_protocol=None):
    """Pod-callable: local pinned processor only, sanitized records plus receipt."""
    from transformers import Qwen3VLProcessor
    from .modules import BASE_REVISION
    model_path = Path(model_path)
    if not model_path.is_dir() or model_path.name != BASE_REVISION:
        raise ValueError("Exact existing local model snapshot required; downloads are disabled")
    contract = training_input_contract(training_input_protocol)
    if (sequence_budget, episode_pixels, min_pixels, max_pixels) != tuple(
            contract[k] for k in ("sequence_budget", "episode_pixels", "min_pixels", "max_pixels")):
        raise ValueError("Do not alter the frozen training preprocessing budgets")
    processor = Qwen3VLProcessor.from_pretrained(str(model_path), local_files_only=True,
                                                min_pixels=min_pixels, max_pixels=max_pixels)
    receipt = audit_model_inputs(records_path, image_root, processor, output, sequence_budget,
                                 episode_pixels, BASE_REVISION, training_input_protocol)
    return [model_input_record(row) for row in read_rows(records_path)], receipt


def check_target_identifiability(row, image_root):
    if row.get("supervised_target_identifiable") is not True:
        raise ValueError("Supervised target identifiability must be explicitly verified")
    if row.get("answer_identifiable") is not False:
        return
    if row.get("view") == "structure":
        return
    try:
        target = json.loads(row["target_text"])
    except (ValueError, TypeError):
        target = None
    if isinstance(target, dict) and target.get("answer") == "INSUFFICIENT" and target.get("answer_identifiable") is False:
        return
    # A bare letter needs the independently verified canonical episode, whose
    # image hash binds the printed INSUFFICIENT choice to the input image.
    canonical = Path(image_root) / "canonical_private.jsonl"
    if not canonical.is_file():
        raise ValueError("Ambiguous QA lacks a trusted printed insufficiency label")
    from .nvw import OUTPUT_SIZE, render, validate_episode, view_record
    for episode in read_rows(canonical):
        if episode["root_id"] == row["root_id"]:
            validate_episode(episode)
            path = image_path(image_root, episode["image"])
            if sha(path) != episode["image_sha256"] or row != view_record(episode, episode["image"], row["view"]):
                raise ValueError("Ambiguous QA differs from trusted canonical image/label")
            expected_image, _ = render(episode)
            expected_image = expected_image.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
            with Image.open(path) as actual_image:
                if expected_image.tobytes() != actual_image.convert("RGB").tobytes():
                    raise ValueError("Printed options differ from trusted insufficiency mapping")
            if episode["answer"] == "INSUFFICIENT":
                return
    raise ValueError("Ambiguous QA lacks a trusted printed insufficiency label")


def contamination_candidates(row, image_root, index):
    text = row["user_text"]
    reasons = []
    text_hashes = index["_text_hashes"] if "_text_hashes" in index else set(index.get("text_sha256", []))
    if text_hash(text) in text_hashes:
        reasons.append("EXACT_BENCHMARK_TEXT")
    tokens = shingles(text)
    matches = set()
    for token in tokens:
        matches.update(index.get("_inverted_shingles", {}).get(token, ()))
    candidate_sets = index.get("_shingle_sets", [])
    for candidate in matches:
        other = candidate_sets[candidate]
        if tokens and other and len(tokens & other)/min(len(tokens), len(other)) >= 0.8:
            reasons.append("NEAR_BENCHMARK_TEXT")
            break
    pixels = index["_pixels"] if "_pixels" in index else set(index.get("pixels_sha256", []))
    bytes_hashes = index["_bytes"] if "_bytes" in index else set(index.get("image_bytes_sha256", []))
    # dHash is a quarantine candidate signal, never an automatic similarity proof.
    dhashes = index["_dhashes"] if "_dhashes" in index else [int(h, 16) for h in index.get("dhash", [])]
    for name in row["image_paths"]:
        path = image_path(image_root, name)
        with Image.open(path) as image:
            fp = image_fingerprint(image)
        if sha(path) in bytes_hashes or fp["pixels_sha256"] in pixels:
            reasons.append("EXACT_BENCHMARK_IMAGE")
        if any((int(fp["dhash"],16) ^ b).bit_count() <= 2 for b in dhashes):
            reasons.append("NEAR_IMAGE_REVIEW_REQUIRED")
    return sorted(set(reasons))


def approve(records_path, image_root, sources, benchmark_index, output, input_preflight_receipt=None,
            training_input_protocol=None):
    rows = read_rows(records_path)
    check_splits(rows)
    index = json.loads(Path(benchmark_index).read_text())
    if index.get("status") != "COMPLETE" or not index.get("coverage"):
        raise ValueError("Missing completed benchmark exclusion index")
    if index.get("coverage_verified") is not True:
        raise ValueError("Benchmark requested file/record coverage was not verified")
    requested = index["coverage"].get("files")
    if not isinstance(requested, list) or not requested:
        raise ValueError("Missing exact benchmark coverage file inventory")
    expected_counts = {str(Path(entry["path"]).resolve()): entry["records"] for entry in requested}
    if (len(expected_counts) != len(requested) or index.get("per_file_record_counts") != expected_counts
            or index.get("record_count") != sum(expected_counts.values())
            or index["coverage"].get("record_count") != index.get("record_count")
            or set(index.get("source_files", {})) != set(expected_counts)
            or index.get("image_count", 0) <= 0):
        raise ValueError("Benchmark exact requested coverage receipt is inconsistent")
    if input_preflight_receipt is None:
        raise ValueError("Actual per-record processor/model-input preflight receipt required")
    receipt_bytes = Path(input_preflight_receipt).read_bytes()
    preflight = json.loads(receipt_bytes.decode("utf-8"))
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    contract = training_input_contract(training_input_protocol)
    from .modules import BASE_REVISION
    if (preflight.get("status") != "PASS" or preflight.get("kind") != "ACTUAL_PROCESSOR_PREFLIGHT"
            or preflight.get("producer_sha256") != sha(__file__)
            or preflight.get("processor_revision") != BASE_REVISION
            or preflight.get("records_sha256") != sha(records_path)):
        raise ValueError("Invalid, stale or wrong-record processor receipt")
    if (preflight.get("processor_class") != "Qwen3VLProcessor"
            or preflight.get("truncation") is not False
            or not preflight.get("chat_template_sha256")
            or preflight.get("sequence_budget") != contract["sequence_budget"]
            or preflight.get("episode_pixels") != contract["episode_pixels"]
            or any(k in preflight and preflight[k] != contract[k] for k in ("min_pixels", "max_pixels"))
            or training_input_protocol is not None and preflight.get("training_input_contract") != contract
            or training_input_protocol is not None and any(preflight.get(k) != contract[k]
                for k in ("min_pixels", "max_pixels"))
            or training_input_protocol is None and "training_input_contract" in preflight):
        raise ValueError("Processor receipt does not use the frozen training input contract")
    index["_text_hashes"] = set(index.get("text_sha256", []))
    index["_pixels"] = set(index.get("pixels_sha256", []))
    index["_bytes"] = set(index.get("image_bytes_sha256", []))
    index["_dhashes"] = [int(h,16) for h in index.get("dhash", [])]
    index["_shingle_sets"] = [set(s) for s in index.get("long_text_shingles", [])]
    inverted = defaultdict(set)
    for i, values in enumerate(index["_shingle_sets"]):
        for value in values:
            inverted[value].add(i)
    index["_inverted_shingles"] = inverted
    train, quarantine, files = [], [], {}
    for row in rows:
        check_lineage(row, sources)
        if row.get("quality_status") not in {"DERIVED_VERIFIED", "SOURCE_GOLD"}:
            raise ValueError("Unverified labels cannot enter training")
        check_target_identifiability(row, image_root)
        if row["split"] != "train":
            continue
        proof = preflight.get("records", {}).get(row["id"], {})
        if (proof.get("status") != "PASS" or proof.get("input_sha256") != input_fingerprint(row, image_root)
                or proof.get("assistant_tokens", 0) <= 0
                or not 0 < proof.get("sequence_tokens", 0) <= preflight["sequence_budget"]
                or set(proof.get("tensor_sha256", {})) != {
                    "input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
                or len(proof.get("image_grid_thw", [])) != len(row["image_paths"])):
            raise ValueError("Missing/mismatched actual per-record model-input proof")
        reasons = contamination_candidates(row, image_root, index)
        if reasons:
            quarantine.append({"id": row["id"], "reasons": reasons})
            continue
        item = dict(model_input_record(row), training_allowed=True,
                    processor_receipt_sha256=receipt_sha256)
        train.append(item)
        for name in row["image_paths"]:
            files[name] = sha(Path(image_root)/name)
    if sha(input_preflight_receipt) != receipt_sha256:
        raise ValueError("Processor receipt changed during approval")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out/"records.jsonl", train)
    write_rows(out/"quarantine.jsonl", quarantine)
    module_root = Path(__file__).parent
    manifest = {
        "dataset_id": "nvw-private-common-training-v1", "status": "APPROVED",
        "training_allowed": True, "benchmark_derived": False,
        "records_sha256": sha(out/"records.jsonl"), "record_count": len(train),
        "image_root": str(Path(image_root).resolve()), "files": files,
        "known_parent_ids": sorted(sources), "sources": sources,
        "source_sha256": {n: sha(module_root/n) for n in ("train.py", "modules.py")},
        "contamination_audit": {"status": "PASS", "index_sha256": sha(benchmark_index),
            "coverage": index["coverage"], "quarantined": len(quarantine),
            "limitations": "Exact/near fingerprints and registered ancestry; not pretraining-contamination certification."},
        "model_input_audit": {"status": "PASS", "receipt_sha256": receipt_sha256,
            "processor_revision": preflight["processor_revision"], "sequence_budget": preflight["sequence_budget"],
            "episode_pixels": preflight["episode_pixels"], "record_count": len(train)},
    }
    if training_input_protocol is not None:
        manifest["model_input_audit"]["training_input_contract"] = contract
        manifest["model_input_audit"].update({k: preflight[k] for k in ("min_pixels", "max_pixels")})
    dump(out/"manifest.json", manifest)
    return manifest
