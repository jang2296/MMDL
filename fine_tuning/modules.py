"""Inspect and restore the actual Qwen3-VL selective training partitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

BASE_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ARMS = {"FL": ("L",), "FM": ("M",), "FV": ("V",), "FML": ("M", "L")}
LEARNING_RATES = {"L": 5e-5, "M": 1e-5, "V": 1e-5}


def inspect_modules(model: nn.Module) -> dict:
    """Derive roots from instantiated module types, never from advertised counts."""
    named = dict(model.named_modules())
    visual = [n for n, m in named.items() if type(m).__name__ == "Qwen3VLVisionModel"]
    language = [n for n, m in named.items() if type(m).__name__ == "Qwen3VLTextModel"]
    mergers = [n for n, m in named.items() if type(m).__name__ == "Qwen3VLVisionPatchMerger"]
    if len(visual) != 1 or len(language) != 1 or len(mergers) != 4:
        raise ValueError("Expected pinned Qwen3-VL: one vision/text root and all four mergers")
    if any(not n.startswith(visual[0] + ".") for n in mergers):
        raise ValueError("Merger outside vision root")
    targets: dict[str, list[str]] = {"V": [], "L": []}
    for name, module in named.items():
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent_type = type(named[parent_name]).__name__
        if parent_type == "Qwen3VLVisionAttention" and leaf in {"qkv", "proj"}:
            targets["V"].append(name)
        elif parent_type in {"Qwen3VLTextAttention", "Qwen3VLTextMLP"}:
            targets["L"].append(name)
    if not all(targets.values()):
        raise ValueError("Missing actual attention/MLP LoRA targets")
    inventory: dict[str, Any] = {"visual": visual[0], "language": language[0], "mergers": mergers, "targets": targets}
    inventory["base_parameters"] = [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
         "group": parameter_group(name, inventory), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
    ]
    inventory["base_group_counts"] = {
        group: sum(row["numel"] for row in inventory["base_parameters"] if row["group"] == group)
        for group in ("V", "M", "L")
    }
    return inventory


def parameter_group(name: str, inventory: dict) -> str:
    # PEFT adds base_model.model prefixes; original roots remain exact suffix segments.
    name = name.removeprefix("base_model.model.")
    if any(name.startswith(root + ".") for root in inventory["mergers"]):
        return "M"
    return "V" if name.startswith(inventory["visual"] + ".") else "L"


def configure_arm(model: nn.Module, arm: str) -> tuple[nn.Module, dict, list[dict]]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}")
    if hasattr(model, "peft_config"):
        raise ValueError("Every arm must start from fresh F0, not an existing adapter")
    inventory = inspect_modules(model)
    model.requires_grad_(False)
    targets = [n for group in ARMS[arm] if group != "M" for n in inventory["targets"][group]]
    if targets:
        from peft import LoraConfig, get_peft_model
        from transformers import PreTrainedModel

        model = get_peft_model(cast(PreTrainedModel, model), LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets,
            bias="none", task_type="CAUSAL_LM",
        ))
    if "M" in ARMS[arm]:
        for name, module in model.named_modules():
            if name.removeprefix("base_model.model.") in inventory["mergers"]:
                # FP32 master parameters avoid rounding 1e-5 updates into BF16 zeros.
                # Forward remains BF16 under autocast; checkpoint preserves master weights.
                module.float().requires_grad_(True)
    groups = []
    rows = []
    expected_ids = set()
    for group in ARMS[arm]:
        selected = []
        for name, param in model.named_parameters():
            if param.requires_grad and parameter_group(name, inventory) == group:
                if group != "M" and "lora_" not in name:
                    raise ValueError(f"Unintended full training parameter: {name}")
                if id(param) in expected_ids:
                    raise ValueError("Overlapping optimizer partitions")
                expected_ids.add(id(param))
                selected.append(param)
                rows.append({"name": name, "shape": list(param.shape), "dtype": str(param.dtype),
                             "group": group, "numel": param.numel()})
        if not selected:
            raise ValueError(f"Empty training partition {group}")
        groups.append({"params": selected, "lr": LEARNING_RATES[group], "group": group})
    if expected_ids != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Optimizer misses trainable parameters")
    inventory.update(arm=arm, trainable=rows, trainable_count=sum(r["numel"] for r in rows))
    return model, inventory, groups


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def save_checkpoint(model: nn.Module, destination: Path, metadata: dict, inventory: dict) -> None:
    destination.mkdir(exist_ok=False, parents=True)
    files = {}
    if hasattr(model, "peft_config"):
        cast(Any, model).save_pretrained(destination / "adapter", safe_serialization=True)
    merger_state = {name: p.detach().cpu() for name, p in model.named_parameters()
                    if p.requires_grad and parameter_group(name, inventory) == "M"}
    if merger_state:
        torch.save(merger_state, destination / "mergers.pt")
    state_hashes = {name: tensor_sha256(p) for name, p in model.named_parameters() if p.requires_grad}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {**metadata, "base_id": BASE_ID, "base_revision": BASE_REVISION,
                "inventory": inventory, "state_sha256": state_hashes, "files": files}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(base_model: nn.Module, checkpoint: str | Path) -> nn.Module:
    """Load complete adapter + all merger masters into a freshly loaded F0."""
    path = Path(checkpoint)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["base_revision"] != BASE_REVISION or metadata["base_id"] != BASE_ID:
        raise ValueError("Checkpoint base identity mismatch")
    for name, digest in metadata["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checkpoint file mismatch: {name}")
    if (path / "adapter").exists():
        from peft import PeftModel

        base_model = PeftModel.from_pretrained(base_model, str(path / "adapter"), is_trainable=False)
    if (path / "mergers.pt").exists():
        state = torch.load(path / "mergers.pt", map_location="cpu", weights_only=True)
        parameters = dict(base_model.named_parameters())
        expected = {r["name"] for r in metadata["inventory"]["trainable"] if r["group"] == "M"}
        if set(state) != expected:
            raise ValueError("Incomplete merger checkpoint")
        with torch.no_grad():
            for name, value in state.items():
                parameter = parameters[name]
                if parameter.shape != value.shape:
                    raise ValueError(f"Merger shape mismatch: {name}")
                parameter.data = value.to(parameter.device, dtype=value.dtype)
    parameters = dict(base_model.named_parameters())
    for name, digest in metadata["state_sha256"].items():
        if tensor_sha256(parameters[name]) != digest:
            raise ValueError(f"Reloaded trainable state differs: {name}")
    return base_model
