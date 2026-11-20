"""G5 complete live tensor readback for pinned, unquantized TP1 Qwen3-VL.

No loading, updates, artifact writes or approval creation occur here. Canonical
HF hashes are streamed; workers return hashes of the corresponding packed-tensor
slices, not a second full model. Private CPU fixtures cannot claim native execution.
"""
from __future__ import annotations

import re
from importlib import import_module
from importlib.metadata import version
from typing import Callable, Iterable

import torch

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256

_EMBED = "language_model.model.embed_tokens.weight"
_HEAD = "language_model.lm_head.weight"
_MERGERS = ["model.visual.merger", *[f"model.visual.deepstack_merger_list.{i}" for i in range(3)]]
_MERGER_SUFFIXES = {"norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
                    "linear_fc2.weight", "linear_fc2.bias"}
SOURCE_REFERENCE: dict = {
    "version": "0.11.0", "commit": "b8b302cde434df8c9289a2b465406b47ebab1c2d",
    "url": "https://github.com/vllm-project/vllm/tree/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm",
    "files": {
        "model_executor/models/qwen3_vl.py": {"sha256": "1ad8854100c4c39d664eb23793ebecd3a65deff9e3733b30a5848c08e76a472f",
            "lines": [[1053, 1074], [1087, 1109], [534, 561]]},
        "model_executor/models/qwen2.py": {"sha256": "ae57d84ad2a2cbc986538ce221d082f4662e169413988ddaaf4a48539dee1bff",
            "lines": [[379, 442]]},
        "model_executor/layers/linear.py": {"sha256": "b2cc3cb18d161976a957f0b70af302291baa66aaed99229497674f266e1aac4d",
            "lines": [[722, 753], [849, 944]]},
        "executor/uniproc_executor.py": {"sha256": "f8d2b673b5112c5c27a18c29af112fac8ac419f461aabf2abe9d4a44c8043a5a",
            "lines": [[71, 83]]},
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G5: " + message)


def require_idle_engine(engine, *, require_push: bool = True) -> dict:
    """Shared live-engine guard; fresh F0 is allowed only without a push claim."""
    from .engines import VLLMRolloutEngine

    _require(type(require_push) is bool, "explicit push requirement must be boolean")
    _require(type(engine) is VLLMRolloutEngine and version("vllm") == "0.11.0",
             "actual pinned VLLMRolloutEngine required")
    client = engine.llm.llm_engine.engine_core
    _require(isinstance(client, import_module("vllm.v1.engine.core_client").InprocClient),
             "native in-process V1 client required")
    executor = client.engine_core.model_executor
    _require(isinstance(executor, import_module("vllm.executor.uniproc_executor").UniProcExecutor),
             "native single-process executor required")
    device = torch.device(executor.driver_worker.device)
    _require(device.type == "cuda" and device.index is not None, "explicit actual CUDA worker device required")
    cfg = engine.llm.llm_engine.vllm_config
    topology = {k: getattr(cfg.parallel_config, k) for k in
                ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "world_size")}
    _require(all(type(n) is int and n == 1 for n in topology.values()), "TP1/PP1/DP1 required")
    _require(cfg.quant_config is None and cfg.lora_config is None
             and cfg.model_config.quantization is None and cfg.model_config.dtype == torch.bfloat16,
             "unquantized BF16 merged rollout model required")
    policy_version, loaded = engine.policy_version, getattr(engine, "_last_loaded_version", None)
    _require(type(policy_version) is int and policy_version >= 0 and engine._push_complete is True,
             "nonnegative version and completed/nonpartial engine state required")
    _require(loaded is None and policy_version == 0 or type(loaded) is int and loaded == policy_version,
             "load marker differs from current policy version")
    if require_push:
        _require(type(loaded) is int and loaded == policy_version and engine._initialization_open is False,
                 "explicit completed weight push required")
    scheduler = client.engine_core.scheduler
    _require(not scheduler.has_unfinished_requests() and not any(scheduler.encoder_cache_manager.cached.values()),
             "idle scheduler/encoder required")
    return {"vllm_version": "0.11.0", "in_process": True, "topology": topology,
            "device": str(device), "dtype": "bfloat16", "quantization": None, "policy_version": policy_version,
            "last_loaded_version": loaded, "completed_push_required": require_push}


def _layout(shapes: dict[str, tuple[int, ...]]) -> dict:
    """Independent HF-prefix and dim-0 packing map; not the native weight loader."""
    groups: dict[str, list[dict]] = {}
    for name, shape in sorted(shapes.items()):
        _require(bool(shape) and all(type(n) is int and n > 0 for n in shape), "invalid canonical shape")
        if name.startswith("model.visual."):
            worker_name = name.removeprefix("model.")
        elif name.startswith("model.language_model."):
            worker_name = "language_model.model." + name.removeprefix("model.language_model.")
        else:
            raise ValueError("G5: unexpected canonical prefix or duplicate tied head: " + name)
        match = re.fullmatch(r"(language_model\.model\.layers\.\d+\.)(self_attn\.([qkv])_proj|mlp\.(gate|up)_proj)\.weight",
                             worker_name)
        order, packed = 0, False
        if match:
            root, _, qkv, gate_up = match.groups()
            worker_name = root + ("self_attn.qkv_proj.weight" if qkv else "mlp.gate_up_proj.weight")
            order = "qkv".index(qkv) if qkv else (0 if gate_up == "gate" else 1)
            packed = True
        groups.setdefault(worker_name, []).append({"canonical_name": name, "shape": list(shape),
                                                    "order": order, "packed": packed})
    result = {}
    for name, parts in sorted(groups.items()):
        parts.sort(key=lambda part: part["order"])
        if parts[0]["packed"]:
            count = 3 if ".qkv_proj." in name else 2
            _require(len(parts) == count and [p["order"] for p in parts] == list(range(count))
                     and all(len(p["shape"]) == 2 and p["shape"][1:] == parts[0]["shape"][1:]
                             and p["packed"] for p in parts), "incomplete/incompatible packed shards")
        else:
            _require(len(parts) == 1, "colliding canonical mapping")
        offset = 0
        for part in parts:
            part["start"], part["stop"] = offset, offset + part["shape"][0]
            offset = part["stop"]
        result[name] = {"shape": [offset, *parts[0]["shape"][1:]], "parts": parts}
    _require(_EMBED in result, "canonical tied embedding required")
    return result


def _read_worker(model, layout: dict, *, native: bool) -> dict:
    if native:
        _require(type(model).__name__ == "Qwen3VLForConditionalGeneration", "actual native Qwen3-VL model required")
    parameters = dict(model.named_parameters())
    all_parameters = dict(model.named_parameters(remove_duplicate=False))
    _require(set(parameters) == set(layout), "missing/extra worker parameters")
    _require(set(all_parameters) == set(layout) | {_HEAD}
             and all_parameters[_HEAD] is all_parameters[_EMBED], "expected tied embedding/head alias required")
    aliases: dict[int, list[str]] = {}
    for name, parameter in all_parameters.items():
        aliases.setdefault(id(parameter), []).append(name)
    alias_groups = sorted(sorted(names) for names in aliases.values() if len(names) > 1)
    _require(alias_groups == [sorted([_EMBED, _HEAD])], "unexpected worker parameter alias")
    rows, devices, storage = [], set(), set()
    for name, expected in layout.items():
        parameter = parameters[name]
        _require(tuple(parameter.shape) == tuple(expected["shape"]) and parameter.dtype == torch.bfloat16
                 and bool(torch.isfinite(parameter).all()), "worker tensor shape/dtype/nonfinite: " + name)
        _require(not native or parameter.device.type == "cuda", "native readback must observe CUDA tensors")
        identity = (str(parameter.device), parameter.untyped_storage().data_ptr())
        _require(identity not in storage, "unexpected shared worker tensor storage")
        storage.add(identity)
        devices.add(str(parameter.device))
        parts = []
        for part in expected["parts"]:
            view = parameter[part["start"]:part["stop"]] if part["packed"] else parameter
            parts.append({"canonical_name": part["canonical_name"], "shape": list(view.shape),
                          "sha256": tensor_sha256(view), "start": part["start"], "stop": part["stop"]})
        rows.append({"worker_name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                     "device": str(parameter.device), "sha256": tensor_sha256(parameter), "parts": parts})
    _require(len(devices) == 1, "worker parameters span multiple devices")
    return {"parameters": rows, "aliases": alias_groups, "device": next(iter(devices))}


def _measure_parity(named: Iterable[tuple[str, torch.Tensor]], shapes: dict[str, tuple[int, ...]],
                    apply_model: Callable, *, native: bool = False) -> dict:
    """Private authored-worker seam; its default never certifies native execution."""
    layout, expected = _layout(shapes), {}
    for name, tensor in named:
        _require(name in shapes and name not in expected and tuple(tensor.shape) == shapes[name]
                 and tensor.dtype == torch.bfloat16 and bool(torch.isfinite(tensor).all()),
                 "invalid/incomplete expected canonical tensor: " + name)
        expected[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_sha256(tensor)}
    _require(set(expected) == set(shapes), "incomplete expected canonical tensor inventory")
    reports = apply_model(lambda model: _read_worker(model, layout, native=native))
    _require(isinstance(reports, list) and len(reports) == 1 and type(reports[0]) is dict,
             "exactly one worker readback required")
    report, ledger = reports[0], []
    for worker in report["parameters"]:
        for part in worker["parts"]:
            name = part["canonical_name"]
            item = expected[name]
            _require(part["shape"] == item["shape"] and part["sha256"] == item["sha256"],
                     "live tensor readback mismatch: " + name)
            ledger.append({"canonical_name": name, "worker_name": worker["worker_name"],
                           "shape": item["shape"], "dtype": item["dtype"],
                           "expected_sha256": item["sha256"], "actual_sha256": part["sha256"],
                           "start": part["start"], "stop": part["stop"]})
    _require(len(ledger) == len(expected) and {r["canonical_name"] for r in ledger} == set(expected),
             "readback did not cover every canonical tensor exactly once")
    merger_names = {f"{root}.{suffix}" for root in _MERGERS for suffix in _MERGER_SUFFIXES}
    _require(merger_names <= expected.keys(), "all four complete Merger tensor sets required")
    return {"all_tensors_match": all(r["expected_sha256"] == r["actual_sha256"] for r in ledger),
        "all_four_mergers": len(merger_names) == 24 and merger_names <= {r["canonical_name"] for r in ledger},
        "native_execution": native, "training_allowed": False, "gpu_gates_certified": False,
        "canonical_tensor_count": len(expected), "worker_tensor_count": len(layout),
        "merger_tensor_count": len(merger_names), "device": report["device"],
        "ledger": sorted(ledger, key=lambda row: row["canonical_name"]),
        "worker_parameters": report["parameters"], "aliases": report["aliases"]}


def _pinned_config(config) -> None:
    text, vision = config.text_config, config.vision_config
    _require(config.model_type == "qwen3_vl" and config.tie_word_embeddings is True
             and text.tie_word_embeddings is True and text.attention_bias is False,
             "pinned tied Qwen3-VL architecture required")
    _require({k: getattr(text, k) for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
             "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")} ==
             dict(num_hidden_layers=36, hidden_size=2560, intermediate_size=9728, num_attention_heads=32,
                  num_key_value_heads=8, head_dim=128, vocab_size=151936), "pinned text dimensions required")
    _require({k: getattr(vision, k) for k in ("depth", "hidden_size", "num_heads", "out_hidden_size",
             "intermediate_size", "patch_size", "spatial_merge_size", "temporal_patch_size", "num_position_embeddings")} ==
             dict(depth=24, hidden_size=1024, num_heads=16, out_hidden_size=2560, intermediate_size=4096,
                  patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304)
             and list(vision.deepstack_visual_indexes) == [5, 11, 17], "pinned vision dimensions required")


def measure_weight_parity(bundle, engine) -> dict:
    """Read actual post-push worker tensors; no report grants training permission."""
    from .policy import assert_base_unchanged, merged_named_tensors

    runtime = require_idle_engine(engine)
    _require(bundle.inventory["base_id"] == BASE_ID and bundle.inventory["base_revision"] == BASE_REVISION,
             "pinned source model identity required")
    _pinned_config(bundle.model.config)
    _pinned_config(engine.model_config)
    shapes = engine._weight_shapes
    _require(len(shapes) == 713 and set(shapes) == set(bundle.base_sha256), "pinned complete canonical inventory required")
    base_sha = dict(bundle.base_sha256)
    assert_base_unchanged(bundle)
    try:
        report = _measure_parity(merged_named_tensors(bundle), shapes, engine.llm.apply_model, native=True)
        _require(report["worker_tensor_count"] == 605, "pinned complete packed worker inventory required")
        _require(report["device"] == runtime["device"], "readback device differs from actual worker device")
        _require(require_idle_engine(engine) == runtime, "engine state changed during readback")
        report.update(hf_base_unchanged=True, policy_version=runtime["policy_version"], runtime=runtime,
            base_sha256=base_sha, source_reference=SOURCE_REFERENCE,
            limitations=["Tensor parity only; no grid, cache-behavior or logprob equivalence claim",
                         "Native source references identify the mapping; they are not installed-file attestations",
                         "This component does not create a complete G0-G13 or data/training approval"])
        return report
    finally:
        assert_base_unchanged(bundle)
        _require(bundle.base_sha256 == base_sha, "reference hash binding changed during readback")
