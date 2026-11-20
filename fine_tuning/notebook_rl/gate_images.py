"""G2 observations from the actual worker visual forward, never CPU expectations alone.

The pinned native path is vLLM v0.11.0 qwen3_vl.py:1288-1314: BF16 pixels and
grid_thw enter visual; returned rows concatenate the final and DeepStack features.
Temporary hooks observe that call without replacing inputs, outputs or weights.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json

import torch

from fine_tuning.modules import tensor_sha256
from .engines import Prepared, RolloutRequest


def _image_rows(pixels: torch.Tensor, grid: list[list[int]], merge: int) -> list[dict]:
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 2
            or pixels.dtype != torch.bfloat16 or not bool(torch.isfinite(pixels).all())
            or type(merge) is not int or merge != 2 or type(grid) is not list or not grid
            or any(type(row) is not list or len(row) != 3
                   or any(type(x) is not int or x <= 0 for x in row)
                   or row[0] != 1 or row[1] % merge or row[2] % merge for row in grid)):
        raise ValueError("G2: finite BF16 image patches and pinned still-image grid required")
    rows, offset = [], 0
    for t, h, w in grid:
        count = t * h * w
        image = pixels[offset:offset + count]
        if image.shape[0] != count:
            raise ValueError("G2: patch rows do not cover image grid")
        rows.append({"grid_thw": [t, h, w], "pixel_shape": list(image.shape),
                     "pixel_bf16_sha256": tensor_sha256(image), "feature_rows": count // merge**2})
        offset += count
    if offset != pixels.shape[0]:
        raise ValueError("G2: extra image patch rows")
    return rows


def _observe_visual(model, stack: ExitStack, records: list[dict]) -> dict:
    """Private seam tests hook semantics with authored modules, not a native GPU."""
    visual = model.visual
    merge = visual.spatial_merge_size
    width = model.config.vision_config.out_hidden_size * (1 + len(visual.deepstack_merger_list))
    if merge != 2 or len(visual.deepstack_merger_list) != 3:
        raise ValueError("G2: all three DeepStack mergers and pinned merge size required")

    def observed(module, args, kwargs, output):
        if module is not visual or len(args) != 1 or set(kwargs) != {"grid_thw"}:
            raise RuntimeError("G2: unsupported native visual call signature")
        pixels, grid = args[0], kwargs["grid_thw"]
        rows = _image_rows(pixels, grid, merge)
        if (not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16
                or output.device != pixels.device
                or list(output.shape) != [sum(r["feature_rows"] for r in rows), width]
                or not bool(torch.isfinite(output).all())):
            raise RuntimeError("G2: actual final/DeepStack feature tensor differs from image token budget")
        records.append({"images": rows, "feature_shape": list(output.shape),
                        "dtype": str(output.dtype), "device": str(output.device)})

    handle = visual.register_forward_hook(observed, with_kwargs=True)
    stack.callback(handle.remove)
    return {"merge_size": merge, "feature_width": width}


def _compare(prepared: Prepared, observed: list[dict], image_token_id: int) -> dict:
    pixels = prepared.pixels
    expected = _image_rows(pixels["pixel_values"].to(torch.bfloat16),
                           pixels["image_grid_thw"].tolist(), 2)
    actual = [image for call in observed for image in call["images"]]
    # Native consecutive batching preserves image order across call boundaries.
    if expected != actual:
        raise RuntimeError("G2: actual engine image order/grid/pixels/feature coverage differs from HF")
    token_count = prepared.token_ids.count(image_token_id)
    if token_count != sum(row["feature_rows"] for row in expected):
        raise RuntimeError("G2: literal image tokens differ from measured feature rows")
    return {"expected_images": expected, "observed_calls": observed,
            "encoder_order_matches_input": expected == actual, "image_tokens": token_count,
            "prompt_sha256": hashlib.sha256(json.dumps(prepared.token_ids, separators=(",", ":")).encode()).hexdigest()}


def measure_image_parity(engine, requests: list[RolloutRequest]) -> dict:
    """Observe fixed single-/multi-image probes in an already-owned idle engine.

    Does not allocate a model, approve data, spend money, update weights or assert
    complete GPU approval. Caller must preserve any failed-probe evidence.
    """
    from .gate_weights import SOURCE_REFERENCE, require_idle_engine

    runtime = require_idle_engine(engine, require_push=False)
    if (len(requests) != 2 or sorted(len(r.images) for r in requests) != [1, 2]
            or len({r.request_id for r in requests}) != 2 or any(r.n != 1 for r in requests)):
        raise ValueError("G2: distinct single-image and two-image probes with n=1 required")
    config = engine.llm.llm_engine.vllm_config
    if config.cache_config.enable_prefix_caching or config.model_config.multimodal_config.mm_processor_cache_gb != 0:
        raise ValueError("G2: pinned disabled prefix/processor caches required")
    records: list[dict] = []
    with ExitStack() as stack:
        def attach(model):
            if (type(model).__name__ != "Qwen3VLForConditionalGeneration"
                    or type(model.visual).__name__ != "Qwen3_VisionTransformer"
                    or model.use_data_parallel):
                raise RuntimeError("G2: pinned unsharded native Qwen3-VL vision path required")
            return _observe_visual(model, stack, records)

        if len(engine.llm.apply_model(attach)) != 1:
            raise RuntimeError("G2: exactly one synchronous in-process worker required")
        probes = []
        for original in requests:
            # One diagnostic output token only; this is not a training rollout.
            request = replace(original, max_new_tokens=1)
            prepared = engine.prepare(request)
            start = len(records)
            # The existing probe checks actual returned prompt_token_ids exactly.
            probabilities = engine.probe_logprobs(request, [prepared.token_ids[-1]])
            if len(probabilities) != 1:
                raise RuntimeError("G2: actual token-identity probe did not complete")
            observed = records[start:]
            if any(call["device"] != runtime["device"] for call in observed):
                raise RuntimeError("G2: visual forward executed on a different device")
            probes.append({"request_id": original.request_id, "seed": original.seeds[0],
                           **_compare(prepared, observed, engine.model_config.image_token_id)})
    if require_idle_engine(engine, require_push=False) != runtime:
        raise RuntimeError("G2: runtime changed during image probe")
    return {"images_checked": sum(len(p["expected_images"]) for p in probes),
            "grid_mismatches": 0, "token_feature_mismatches": 0, "pixel_bf16_mismatches": 0,
            "runtime": runtime, "probes": probes, "temporary_hooks_removed": True,
            "source_reference": {"commit": SOURCE_REFERENCE["commit"],
                "file": "vllm/model_executor/models/qwen3_vl.py",
                "sha256": SOURCE_REFERENCE["files"]["model_executor/models/qwen3_vl.py"]["sha256"],
                "lines": [[1288, 1314], [1361, 1388]]},
            "training_allowed": False, "gpu_gates_certified": False,
            "limitations": ["Image order is checked across encoder calls, not downstream feature placement",
                            "Checks token/feature counts, not HF/native feature-value equivalence",
                            "Not a complete GPU gate report or a training approval"]}
