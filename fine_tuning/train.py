"""Private BF16 selective SFT. No remote downloads or implicit GPU fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from fine_tuning.modules import (ARMS, BASE_REVISION, configure_arm,
                                 load_checkpoint, parameter_group, save_checkpoint, tensor_sha256)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def approved_records(records_path: Path, manifest_path: Path, image_root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "APPROVED" or manifest.get("training_allowed") is not True
            or manifest.get("benchmark_derived") is not False
            or manifest.get("contamination_audit", {}).get("status") != "PASS"):
        raise ValueError("Training requires an approved clean contamination manifest")
    if manifest.get("records_sha256") != file_hash(records_path):
        raise ValueError("Records differ from immutable manifest")
    for name in ("train.py", "modules.py"):
        if manifest.get("source_sha256", {}).get(name) != file_hash(Path(__file__).with_name(name)):
            raise ValueError(f"Training source differs from manifest: {name}")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or manifest.get("record_count") != len(records):
        raise ValueError("Empty or mismatched training record count")
    known_parents = set(manifest.get("known_parent_ids", []))
    ids = set()
    verified_images = set()
    for record in records:
        if (record.get("split") != "train" or record.get("training_allowed") is not True
                or record.get("benchmark_derived") is not False
                or not record.get("lineage") or not set(record["lineage"]).issubset(known_parents)
                or record.get("id") in ids):
            raise ValueError("Invalid split, lineage, permission, or repeated record")
        ids.add(record["id"])
        if not all(isinstance(record.get(k), str) and record[k].strip() for k in ("id", "user_text", "target_text")):
            raise ValueError("Missing nonempty record text")
        if not isinstance(record.get("image_paths"), list) or not record["image_paths"]:
            raise ValueError("All module arms require image-based records")
        for relative in record["image_paths"]:
            image = (image_root / relative).resolve()
            if Path(relative).is_absolute() or not image.is_relative_to(image_root.resolve()):
                raise ValueError("Image escapes approved root")
            if relative not in verified_images:
                if manifest.get("files", {}).get(relative) != file_hash(image):
                    raise ValueError(f"Unapproved image bytes: {relative}")
                verified_images.add(relative)
    return records, manifest


def validate_training_input_contract(manifest: dict, min_pixels: int, max_pixels: int,
                                     episode_pixels: int, sequence_budget: int,
                                     training_input_protocol: Path | None = None) -> dict:
    """Bind CLI budgets to approved processor evidence before any model/GPU load."""
    from .audit import training_input_contract
    contract = training_input_contract(training_input_protocol)
    actual = (min_pixels, max_pixels, episode_pixels, sequence_budget)
    expected = tuple(contract[k] for k in ("min_pixels", "max_pixels", "episode_pixels", "sequence_budget"))
    proof = manifest.get("model_input_audit", {})
    if (actual != expected or proof.get("status") != "PASS"
            or proof.get("processor_revision") != BASE_REVISION
            or proof.get("record_count") != manifest.get("record_count")
            or type(manifest.get("record_count")) is not int or manifest["record_count"] <= 0
            or proof.get("sequence_budget") != sequence_budget
            or proof.get("episode_pixels") != episode_pixels
            or any(k in proof and proof[k] != contract[k] for k in ("min_pixels", "max_pixels"))
            or training_input_protocol is not None and proof.get("training_input_contract") != contract
            or training_input_protocol is not None and any(proof.get(k) != contract[k]
                for k in ("min_pixels", "max_pixels"))
            or training_input_protocol is None and "training_input_contract" in proof):
        raise ValueError("Training CLI and approved processor input contract differ")
    return contract


class AssistantCollator:
    """Microbatch one, full assistant response and EOS, no silent truncation."""

    def __init__(self, processor, image_root: Path, sequence_budget: int, episode_pixels: int):
        self.processor = processor
        self.image_root = image_root
        self.sequence_budget = sequence_budget
        self.episode_pixels = episode_pixels

    def __call__(self, record: dict) -> dict[str, torch.Tensor]:
        images = []
        try:
            for relative in record["image_paths"]:
                with Image.open(self.image_root / relative) as original:
                    images.append(original.convert("RGB"))
            content = [{"type": "image", "image": image} for image in images]
            for key in ("user_text", "target_text"):
                if any(token in record[key] for token in self.processor.tokenizer.all_special_tokens):
                    raise ValueError("Record text injects chat/image control tokens")
            content.append({"type": "text", "text": record["user_text"]})
            user = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            full = self.processor.apply_chat_template(
                user + [{"role": "assistant", "content": [{"type": "text", "text": record["target_text"]}]}],
                tokenize=False, add_generation_prompt=False,
            )
            if not full.startswith(prompt):
                raise ValueError("Chat template cannot establish assistant-only boundary")
            inputs = self.processor(text=[full], images=images, padding=True, truncation=False, return_tensors="pt")
            prefix = self.processor(text=[prompt], images=images, padding=True, truncation=False, return_tensors="pt")
            ids = inputs["input_ids"]
            prefix_ids = prefix["input_ids"]
            n = prefix_ids.shape[1]
            if ids.shape[1] > self.sequence_budget or not torch.equal(ids[:, :n], prefix_ids):
                raise ValueError("Sequence overflow or unstable assistant boundary; do not truncate")
            grid = inputs["image_grid_thw"]
            patch_size = self.processor.image_processor.patch_size
            if int(grid.prod(-1).sum()) * patch_size**2 > self.episode_pixels:
                raise ValueError("Episode image budget exceeded; do not auto-resize")
            labels = ids.clone()
            labels[:, :n] = -100
            labels[inputs["attention_mask"] == 0] = -100
            tokenizer = self.processor.tokenizer
            image_tokens = {tokenizer.convert_tokens_to_ids(token) for token in
                            ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")}
            for token_id in image_tokens:
                labels[ids == token_id] = -100
            if tokenizer.eos_token_id not in labels[labels != -100].tolist():
                raise ValueError("Assistant EOS is not supervised")
            if not int((labels[:, 1:] != -100).sum()):
                raise ValueError("No assistant loss tokens")
            inputs["labels"] = labels
            return dict(inputs)
        finally:
            for image in images:
                image.close()


def probe_indices(numel: int, device: torch.device | str = "cpu") -> torch.Tensor:
    if numel <= 0:
        raise ValueError("Cannot probe an empty parameter")
    count = min(1024, numel)
    # Integer arithmetic: float32 linspace can round a large last index to numel.
    return torch.tensor([i * (numel - 1) // max(1, count - 1) for i in range(count)],
                        dtype=torch.long, device=device)


def parameter_probes(model, inventory: dict) -> dict:
    """Bounded frozen sampling is explicit; trainable hashes are complete in export."""
    selected = {}
    for group in ("V", "M", "L"):
        frozen = [(n, p) for n, p in model.named_parameters()
                  if not p.requires_grad and parameter_group(n, inventory) == group]
        for name, parameter in frozen[::max(1, len(frozen) // 3)][:3]:
            flat = parameter.detach().reshape(-1)
            indices = probe_indices(flat.numel(), flat.device)
            selected[name] = {"indices": indices, "sha256": tensor_sha256(flat[indices])}
    return selected


def optimizer_update(model, optimizer, tokens: int, inventory: dict, probes: dict) -> dict:
    if tokens <= 0:
        raise ValueError("Empty update")
    parameters = dict(model.named_parameters())
    norms = {}
    before = {}
    for group in optimizer.param_groups:
        squared = 0.0
        before[group["group"]] = [(p, p.detach().flatten()[:1024].float().clone()) for p in group["params"]]
        for parameter in group["params"]:
            if parameter.grad is None:
                raise ValueError(f"Disconnected gradient in group {group['group']}")
            parameter.grad.div_(tokens)
            squared += float(parameter.grad.float().square().sum())
        norms[group["group"]] = math.sqrt(squared)
        if not math.isfinite(norms[group["group"]]) or norms[group["group"]] <= 0:
            raise ValueError(f"Absent/nonfinite group gradient: {group['group']}")
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()
    deltas = {group: sum(float((p.detach().flatten()[:1024].float() - old).abs().sum()) for p, old in values)
              for group, values in before.items()}
    if any(not math.isfinite(value) or value <= 0 for value in deltas.values()):
        raise ValueError("Optimizer failed to change a trainable group")
    for name, probe in probes.items():
        if tensor_sha256(parameters[name].detach().flatten()[probe["indices"]]) != probe["sha256"]:
            raise ValueError(f"Frozen sampled parameter changed: {name}")
    optimizer.zero_grad(set_to_none=True)
    return {"gradient_l2": norms, "sampled_update_l1": deltas,
            "frozen_sampled_checks": len(probes), "loss_tokens": tokens}


def order_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def stop_for_budget(elapsed: float, budget_seconds: float | None,
                    mean_update_seconds: float, checkpoint_reserve_seconds: float = 120) -> bool:
    """At update boundaries, leave time for another update and checkpoint export."""
    return budget_seconds is not None and elapsed + max(30, mean_update_seconds * 1.5) + checkpoint_reserve_seconds >= budget_seconds


def load_base(model_path: Path):
    from transformers import Qwen3VLForConditionalGeneration

    if not model_path.is_dir() or model_path.name != BASE_REVISION:
        raise ValueError("Use the local pinned HF snapshot directory; downloads are disabled")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", device_map=None,
    )
    if model.config.model_type != "qwen3_vl":
        raise ValueError("Wrong base model architecture")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--verify-checkpoint", type=Path)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--episode-pixels", type=int)
    parser.add_argument("--sequence-budget", type=int, default=4096)
    parser.add_argument("--input-protocol", type=Path,
                        help="Explicit frozen child input contract; omission keeps original1M episode cap")
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    import peft
    import transformers

    if (torch.__version__, transformers.__version__, peft.__version__) != ("2.8.0+cu128", "4.57.1", "0.17.1"):
        raise ValueError("Training requires exact torch/transformers/peft environment")
    if args.verify_checkpoint:
        model = load_checkpoint(load_base(args.model_path), args.verify_checkpoint)
        print(json.dumps({"status": "RELOAD_VERIFIED", "checkpoint": str(args.verify_checkpoint),
                          "parameters": sum(p.numel() for p in model.parameters())}))
        return
    required = (args.records, args.manifest, args.image_root, args.output, args.arm,
                args.min_pixels, args.max_pixels, args.episode_pixels)
    if not all(required) or not 0 < args.min_pixels <= args.max_pixels <= args.episode_pixels:
        parser.error("Training needs approved data, arm, output and explicit positive pixel budgets")
    if (args.accumulation <= 0 or args.max_steps < 0 or args.sequence_budget <= 0
            or args.budget_seconds is not None and (not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0)):
        parser.error("Invalid accumulation, steps or sequence budget")
    if args.output.exists():
        raise ValueError("Output already exists; checkpoints are immutable, resume is unsupported")
    records, manifest = approved_records(args.records, args.manifest, args.image_root)
    input_contract = validate_training_input_contract(manifest, args.min_pixels, args.max_pixels,
        args.episode_pixels, args.sequence_budget, args.input_protocol)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 CUDA training required; CPU/offload/quantization fallback disabled")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, inventory, groups = configure_arm(load_base(args.model_path), args.arm)
    model.to("cuda")
    pretrained_model: Any = model
    pretrained_model.config.use_cache = False
    pretrained_model.config.text_config.use_cache = False
    pretrained_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pretrained_model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Optimizer parameter identity mismatch")
    processor = transformers.AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    collator = AssistantCollator(processor, args.image_root, args.sequence_budget, args.episode_pixels)
    probes = parameter_probes(model, inventory)
    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    args.output.mkdir(parents=True)
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    totals = {"records": 0, "images": 0, "input_tokens": 0, "visual_tokens": 0, "target_tokens": 0, "updates": 0}
    loss_tokens = 0
    weighted_loss = 0.0
    consumed_ids: list[str] = []
    termination = "EPOCH_COMPLETE"
    training_started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    with (args.output / "updates.jsonl").open("x", encoding="utf-8") as log, \
            (args.output / "consumed_ids.jsonl").open("x", encoding="utf-8") as exposure_log:
        for position, index in enumerate(order):
            mean_update = ((time.monotonic() - training_started) / totals["updates"]
                           if totals["updates"] else 0.0)
            if loss_tokens == 0 and stop_for_budget(time.monotonic() - started, args.budget_seconds, mean_update):
                termination = "BUDGET_STOP"
                break
            record = records[index]
            batch = {name: value.to("cuda") for name, value in collator(record).items()}
            tokens = int((batch["labels"][:, 1:] != -100).sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch, use_cache=False).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite loss")
            (loss * tokens).backward()
            weighted_loss += float(loss.detach()) * tokens
            loss_tokens += tokens
            totals["records"] += 1
            totals["images"] += len(record["image_paths"])
            totals["input_tokens"] += int(batch["attention_mask"].sum())
            totals["target_tokens"] += tokens
            totals["visual_tokens"] += int(batch["image_grid_thw"].prod(-1).sum()) // processor.image_processor.merge_size**2
            consumed_ids.append(record["id"])
            exposure_log.write(json.dumps({"id": record["id"], "position": position,
                "target_tokens": tokens, "input_tokens": int(batch["attention_mask"].sum())}) + "\n")
            exposure_log.flush()
            del batch, loss
            if (position + 1) % args.accumulation == 0 or position + 1 == len(order):
                metrics = optimizer_update(model, optimizer, loss_tokens, inventory, probes)
                totals["updates"] += 1
                metrics.update(step=totals["updates"], mean_token_loss=weighted_loss / loss_tokens,
                               elapsed_seconds=time.monotonic() - started,
                               mean_update_seconds=(time.monotonic() - training_started) / totals["updates"],
                               consumed_records=len(consumed_ids), consumed_order_sha256=order_hash(consumed_ids),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                loss_tokens = 0
                weighted_loss = 0.0
                if args.max_steps and totals["updates"] >= args.max_steps:
                    if len(consumed_ids) != len(order):
                        termination = "MAX_STEPS_STOP"
                    break
    metadata = {"status": "BUDGET_STOP" if termination == "BUDGET_STOP" else "TRAINED_RELOAD_PENDING",
                "termination": termination, "epoch_exposure_complete": len(consumed_ids) == len(order),
                "arm": args.arm, "seed": args.seed,
                "manifest_sha256": file_hash(args.manifest), "records_sha256": manifest["records_sha256"],
                "record_order_sha256": order_hash([records[i]["id"] for i in order]),
                "planned_records": len(order), "consumed_order_sha256": order_hash(consumed_ids),
                "consumed_ids_sha256": file_hash(args.output / "consumed_ids.jsonl"),
                "mean_update_seconds": ((time.monotonic() - training_started) / totals["updates"]
                                        if totals["updates"] else None),
                "exposure": totals, "seconds": time.monotonic() - started,
                "budgets": {key: getattr(args, key) for key in ("min_pixels", "max_pixels", "episode_pixels", "sequence_budget", "accumulation", "max_steps", "budget_seconds")},
                "base_precision": "BF16", "trainable_master_precision": "FP32", "epochs": 1,
                "training_input_contract": input_contract,
                "source_sha256": {n: file_hash(Path(__file__).with_name(n)) for n in ("train.py", "modules.py")}}
    save_checkpoint(model, args.output / "checkpoint", metadata, inventory)
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
