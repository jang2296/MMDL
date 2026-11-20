"""Live inventory and device numerical probes, not a complete G0--G13 approval.

Inventory reads the instantiated policy, not declared counts in a report. The
standalone numerical command loads no model or dataset and changes no weights.
Actual pretrained-model, data, sync and budget gates remain separate obligations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .dataset import _bytes, _sha, _write
from .vault import _read, _safe_root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_COUNTS = {"L": 33030144, "M": 109105152, "V": 2359296}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("gate probe: " + message)


def measure_inventory(bundle, protocol: dict) -> dict:
    """G1 live evidence; never substitutes for checkpoint-file or GPU provenance."""
    _require(protocol["expected_trainable"] == _COUNTS, "pinned parameter contract required")
    return _inventory(bundle, protocol, expected_targets={"L": 252, "V": 48})


def _inventory(bundle, protocol: dict, *, expected_targets: dict) -> dict:
    """Internal numeric-size seam permits tiny real Qwen CPU regression tests."""
    import torch
    from peft.tuners.lora.layer import Linear as LoraLinear
    from peft.utils.other import ModulesToSaveWrapper
    from torch import nn

    from .policy import ARMS, assert_base_unchanged

    base = bundle.model.get_base_model()
    named = dict(base.named_modules())
    visual = [n for n, m in named.items() if type(m).__name__ == "Qwen3VLVisionModel"]
    text = [n for n, m in named.items() if type(m).__name__ == "Qwen3VLTextModel"]
    _require(len(visual) == len(text) == 1, "one actual vision/text root required")
    arm = bundle.inventory["arm"]
    _require(arm in ARMS and type(bundle.route.vision_on) is bool, "known arm and explicit route required")
    wrappers = {n: m for n, m in named.items() if isinstance(m, ModulesToSaveWrapper)}
    _require(set(wrappers) == set(protocol["model"]["modules_to_save"]) and len(wrappers) == 4
             and all(n.startswith(visual[0] + ".") for n in wrappers), "all four actual Merger wrappers required")
    mergers = []
    for name, wrapper in sorted(wrappers.items()):
        _require(set(wrapper.modules_to_save) == {"default"} and wrapper.active_adapter == "default"
                 and not wrapper.disable_adapters, "active default Merger copy required")
        original, master = wrapper.original_module, wrapper.modules_to_save["default"]
        _require(type(original).__name__ == type(master).__name__ == "Qwen3VLVisionPatchMerger",
                 "actual Qwen Merger types required")
        originals, masters = dict(original.named_parameters()), dict(master.named_parameters())
        _require(set(originals) == set(masters) and len(masters) == 6,
                 "complete original and master Merger tensor sets required")
        for key, p in masters.items():
            q = originals[key]
            _require(p.shape == q.shape and p.dtype == torch.float32 and p.requires_grad
                     and q.dtype == torch.bfloat16 and not q.requires_grad
                     and p.data_ptr() != q.data_ptr(), "FP32 master / distinct frozen BF16 original required")
        mergers.append({"name": name, "master": [{"name": k, "shape": list(p.shape), "numel": p.numel()}
                                                 for k, p in masters.items()],
                        "original_dtype": "bfloat16", "master_dtype": "float32"})

    targets: dict[str, list[str]] = {"L": [], "V": []}
    installed: dict[str, list[str]] = {"L": [], "V": []}
    for name, module in named.items():
        parent, _, leaf = name.rpartition(".")
        parent_type = type(named.get(parent)).__name__
        group = ("V" if parent_type == "Qwen3VLVisionAttention" and leaf in ("qkv", "proj")
                 else "L" if parent_type in ("Qwen3VLTextAttention", "Qwen3VLTextMLP") else None)
        if group is not None and isinstance(module, (nn.Linear, LoraLinear)):
            targets[group].append(name)
        if isinstance(module, LoraLinear):
            if group is None:
                raise ValueError("gate probe: unexpected LoRA target")
            _require(isinstance(module.base_layer, nn.Linear), "unexpected LoRA base layer")
            cfg = protocol["model"]["lora"]
            _require(set(module.lora_A) == set(module.lora_B) == {"default"}
                     and module.active_adapters == ["default"] and not module.disable_adapters
                     and not module.merged and module.r["default"] == cfg["r"]
                     and module.lora_alpha["default"] == cfg["alpha"]
                     and module.scaling["default"] == cfg["alpha"] / cfg["r"]
                     and not module.use_dora.get("default", False), "LoRA state differs from frozen contract")
            dropout = module.lora_dropout["default"]
            _require(isinstance(dropout, nn.Identity) or isinstance(dropout, nn.Dropout) and dropout.p == 0,
                     "nonzero adapter dropout")
            installed[group].append(name)
    _require({g: len(v) for g, v in targets.items()} == expected_targets, "actual target count mismatch")
    _require(set(installed["L"]) == set(targets["L"])
             and set(installed["V"]) == (set() if arm == "MOD-FML" else set(targets["V"])),
             "actual installed targets differ from arm")
    _require(all(set(bundle.inventory["targets"][g]) == set(targets[g]) for g in targets)
             and bundle.inventory["visual"] == visual[0] and bundle.inventory["language"] == text[0]
             and set(bundle.inventory["mergers"]) == set(wrappers), "cached module inventory differs from live modules")

    partition: dict[str, list] = {"L": [], "M": [], "V": []}
    rows, ids, frozen = [], set(), []
    for name, p in base.named_parameters():
        group = ("M" if any(name.startswith(n + ".") for n in wrappers)
                 else "V" if name.startswith(visual[0] + ".") else "L")
        adapter = name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
        master = group == "M" and ".modules_to_save.default." in name
        permitted = master or adapter
        expected_grad = permitted and (group != "V" or bundle.route.vision_on)
        _require(p.requires_grad == expected_grad, "live requires_grad differs from partition/route: " + name)
        if permitted:
            _require(p.dtype == torch.float32 and id(p) not in ids and bool(torch.isfinite(p).all()),
                     "trainable master dtype, finite value or identity mismatch")
            ids.add(id(p))
            partition[group].append(p)
            rows.append({"name": name, "shape": list(p.shape), "numel": p.numel(), "group": group,
                         "dtype": str(p.dtype), "requires_grad": p.requires_grad, "device": str(p.device)})
        else:
            _require(p.dtype == torch.bfloat16, "frozen base is not BF16: " + name)
            frozen.append(name)
    # named_parameters normally de-duplicates aliases; trainable aliases are not safe partitions.
    aliases: dict[int, list[str]] = {}
    for name, p in base.named_parameters(remove_duplicate=False):
        if id(p) in ids:
            aliases.setdefault(id(p), []).append(name)
    _require(all(len(names) == 1 for names in aliases.values()), "trainable tensor has aliased names")
    _require(set(bundle.params) == {"L", "M", "V"}
             and all(len({id(p) for p in bundle.params[g]}) == len(bundle.params[g])
                     and {id(p) for p in bundle.params[g]} == {id(p) for p in partition[g]} for g in partition),
             "optimizer partitions differ from live parameters")
    _require({id(p) for p in bundle.route.params} == {id(p) for p in partition["V"]},
             "route does not cover exactly Vision adapters")
    expected = dict(protocol["expected_trainable"])
    if arm == "MOD-FML":
        expected["V"] = 0
        _require(not bundle.route.vision_on, "Vision cannot be enabled in FML")
    counts = {g: sum(p.numel() for p in ps) for g, ps in partition.items()}
    _require(counts == expected and len(partition["M"]) == 24, "actual trainable parameter counts differ")
    _require(bundle.inventory["trainable_counts"] == counts, "cached parameter counts differ")
    assert_base_unchanged(bundle)
    return {"trainable_counts": counts, "mergers": len(wrappers), "master_dtype": "float32",
            "base_dtype": "bfloat16", "roots": {"vision": visual[0], "text": text[0]},
            "targets": targets, "installed_targets": installed, "merger_tensors": mergers,
            "parameters": rows, "vision_route_on": bundle.route.vision_on,
            "currently_enabled_counts": {g: sum(p.numel() for p in ps if p.requires_grad) for g, ps in partition.items()},
            "frozen_tensor_count": len(frozen), "base_sha256": dict(bundle.base_sha256),
            "frozen_sha_unchanged": True, "partitions_disjoint": True,
            "pinned_architecture_counts": expected_targets == {"L": 252, "V": 48}
                and counts == (_COUNTS | {"V": 0} if arm == "MOD-FML" else _COUNTS)}


def _sources() -> dict:
    return {name: _sha(_read(HERE / name)) for name in
            ("gpu_gates.py", "gate_numerics.py", "losses.py", "policy.py", "protocol_v1.yaml")}


def run_numerical_probe(device: str, output: Path) -> dict:
    """Write an actual-device, model-free check; it grants no training permission."""
    output = Path(output)
    _require(output.is_absolute() and output.suffix == ".json" and not output.exists()
             and not output.is_symlink() and not output.resolve().is_relative_to(ROOT),
             "new absolute JSON output outside checkout required")
    _safe_root(output.parent, create=True)
    source = _sources()
    began = time.monotonic()
    result: dict = {"format": "notebook-rl-numerical-probe-v1", "status": "INCOMPLETE",
        "training_allowed": False, "gpu_gates_certified": False, "model_loaded": False,
        "weights_updated": False, "source_sha256": source,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "requested_device": device,
        "checks": ["G10_NUMERICAL_COMPONENT"],
        "not_measured": ["G0-G9", "G11-G13", "pretrained4B", "data_eligibility", "pinned_runtime", "cloud_budget"],
        "limitations": ["Synthetic tensors test the objective, not learned self-correction or GPU memory fitness",
                        "This partial probe is not a complete GPU gate report or a training approval"]}
    try:
        from .gate_numerics import measure_numerics
        result["measurements"] = measure_numerics(device)
        _require(_sources() == source, "source changed during numerical measurement")
        result["status"] = "CHECK_COMPLETED_NOT_GPU_APPROVAL"
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - began
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write(output, _bytes(result) + b"\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-check", action="store_true", required=True)
    parser.add_argument("--device", required=True, help="cpu or an explicitly selected cuda:N; no fallback")
    parser.add_argument("--out", type=Path, required=True, help="New private JSON outside the checkout")
    args = parser.parse_args(argv)
    try:
        result = run_numerical_probe(args.device, args.out)
        print(json.dumps({"status": result["status"], "output": str(args.out),
                          "training_allowed": False, "gpu_gates_certified": False}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED_NOT_GPU_APPROVAL", "error_type": type(exc).__name__,
                          "training_allowed": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
