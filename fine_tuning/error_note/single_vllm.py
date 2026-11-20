"""SINGLE v3 generation on an in-process vLLM 0.11 engine: continuous batching, policy pushed after every update.

The HF/PEFT bundle stays the training authority. Before a block generates, its LoRA-merged BF16 weights (Merger
masters rounded to BF16, as in the eval export) are pushed with the existing complete-inventory loader of
notebook_rl.engines.VLLMRolloutEngine. vLLM's processed log-probabilities (presence penalty -> temperature ->
top-k/top-p, the order rl.processed_scores uses) give each sampled action's log q and the support it was drawn
from; every block's training replay checks them against the HF policy (protocol replay bounds). Sleep mode
level 2 frees vLLM weights and KV cache while the HF policy trains; the next push restores complete weights.

v15 LoRA mode (`lora=True`, llm_lora): the original BF16 snapshot stays in vLLM; the LLM adapter is a fresh vLLM LoRA
per policy, the vision adapter runs through exact hooks, only the Mergers are pushed, and sleep is level 1 (the pool
also holds the LoRA slots). The first LoRA sync runs the PF1-PF5 preflight before any generation; PF5 probes an
amplified copy of the adapter, after which the real adapter is restored under a fresh id and re-checked bit-exactly.
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import warnings
from typing import Any, Iterable, Iterator

from . import llm_lora
from .collect import SINGLE_SAMPLING, Stream, StreamStats, continuous_stream, single_request_binding, single_request_sha256
from .contract import CONTEXT_LIMIT, IMAGE

ENGINE = "vllm-0.11.0-processed-logprobs"
TOP_LOGPROBS = 20  # == SINGLE_SAMPLING top_k: every token the sampler could draw is returned
TAIL_LOGPROBS = 20  # logging only: top-20 plus the sampled token decide B2 membership; open ties give None
_B2_CUT = 1.0 - 0.800000011920928955078125  # fp32 1 - fp32(0.8): vLLM keeps an action when its ascending cumsum exceeds it


@contextmanager
def _nondeterministic():
    """vLLM sampling is batch-dependent anyway; HF replay alone needs deterministic kernels."""
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def support_from_logprobs(token: int, row: dict) -> tuple[float, list[int]]:
    """Finite processed log-probs are exactly the top-k/top-p support; the sampled token must be inside it."""
    support = sorted(t for t, value in row.items() if math.isfinite(value.logprob))
    if token not in support or len(support) > SINGLE_SAMPLING["top_k"]:
        raise RuntimeError("SINGLE vLLM: sampled action outside its returned processed support")
    return float(row[token].logprob), support


def sampled_logprob(token: int, row: dict) -> float:
    """Selected processed probability; returned alternatives never renormalize a full-vocabulary draw."""
    if token not in row:
        raise RuntimeError("v16b vLLM: selected action missing from processed log-probabilities")
    value = float(row[token].logprob)
    if not math.isfinite(value) or value > 1e-6:
        raise RuntimeError("v16b vLLM: invalid selected processed log-probability")
    return value


def b2_tail_flag(token: int, row: dict, *, vocab_size: int) -> bool | None:
    """True when the sampled action lies outside B2's support: native top-k 20 (threshold ties kept), then ascending
    top-p 0.8. `row` is a processed full-vocabulary row: top-TAIL_LOGPROBS plus the sampled action. Unreturned tokens
    may tie the 20th value; their mass is bounded by the row's missing probability. None (diagnostic only) when tie
    order or that unseen mass decides membership; raises only for an incomplete row."""
    if type(vocab_size) is not int or vocab_size < 1:
        raise RuntimeError("v16b vLLM: vocabulary size required for the tail diagnostic")
    values = {int(t): float(v.logprob) for t, v in row.items() if math.isfinite(v.logprob)}
    ranked = sorted(values.values(), reverse=True)
    if token not in values or len(ranked) < min(TAIL_LOGPROBS, vocab_size):
        raise RuntimeError("v16b vLLM: incomplete top-20 tail diagnostic")
    threshold, value = ranked[min(20, vocab_size) - 1], values[token]
    if value < threshold:
        return True
    weight = math.exp(value)
    total = sum(math.exp(v) for v in ranked if v >= threshold)
    below = sum(math.exp(v) for v in ranked if threshold <= v < value)
    tied = sum(math.exp(v) for v in ranked if v == value)
    unseen = 0.0
    if ranked[-1] >= threshold:  # every returned value reaches the threshold: unreturned tokens may tie it
        missing = max(0.0, 1.0 - sum(math.exp(v) for v in ranked)) + 1e-6
        unseen = min(max(0, vocab_size - len(ranked)) * math.exp(threshold), missing)
    retained = set()
    for extra in (0.0, unseen):  # cumsum fractions are monotone in the unseen tied mass and in tie position
        low = below + (extra if value > threshold else 0.0)
        for cumulative in (low + weight, low + tied + (extra if value == threshold else 0.0)):
            retained.add(cumulative / (total + extra) > _B2_CUT)
    return None if len(retained) > 1 else not retained.pop()


class _LoraEngine:
    """continuous_stream calls add_request(rid, prompt, params); attach the policy's LoRA and confirm it on the live
    scheduler request (vLLM 0.11 V1 RequestOutput never carries lora_request, output_processor.py:251)."""

    def __init__(self, inner: Any, lora: Any):
        self.inner, self.lora = inner, lora

    def add_request(self, request_id: str, prompt: Any, params: Any) -> None:
        self.inner.add_request(request_id, prompt, params, lora_request=self.lora)
        live = self.inner.engine_core.engine_core.scheduler.requests[request_id].lora_request
        if live is None or live.lora_int_id != self.lora.lora_int_id:
            raise RuntimeError("SINGLE vLLM: request is not bound to the current policy LoRA")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


PROBE_TEXT = "Describe every visible detail of the image, then explain what it could represent."
PROBE_TOKENS = 64


def probe_requests(image: dict) -> list:
    """PF5 inputs: fixed text with 1 and 2 deterministic noise images of at least min_pixels (no task data)."""
    import numpy as np
    from PIL import Image

    from fine_tuning.notebook_rl.engines import RolloutRequest

    side = -(-(math.isqrt(image["min_pixels"] - 1) + 1) // 32) * 32  # ceil(sqrt(min_pixels)) up to a multiple of 32
    if not image["min_pixels"] <= side * side <= image["max_pixels"]:
        raise ValueError("SINGLE vLLM: probe image size outside the image budget")
    rng = np.random.RandomState(3407)
    pictures = [Image.fromarray(rng.randint(0, 256, (side, side, 3), dtype=np.uint8)) for _ in range(3)]
    out = []
    for index, images in enumerate((pictures[:1], pictures[1:])):
        messages = [{"role": "user", "content": [*({"type": "image"} for _ in images),
                                                 {"type": "text", "text": PROBE_TEXT}]}]
        out.append(RolloutRequest(f"lora-preflight-{index}", None, messages, images, 1, PROBE_TOKENS, [3407]))
    return out


class SingleVllmSampler:
    """One engine for the whole campaign; `stream_builder` matches collect.hf_single_stream_builder's signature. The image config comes from the plan's protocol."""

    lora = False  # class defaults keep __new__-built samplers on the merged-push path
    lora_request: Any = None
    vision_hooks: Any = None

    def __init__(self, model_path: Path, *, kv_cache_gib: float, max_num_seqs: int, gpu_memory_utilization: float,
                 sleep: bool, eos_token_ids: Iterable[int], image: dict | None = None, lora: bool = False,
                 sampling: dict | None = None, version_registry: bool = False,
                 protocol_version: str = "error-note-v16b", preflight_stage: str = "trained",
                 generation_context_limit: int | None = None, rehearsal_profile: str | None = None):
        if rehearsal_profile not in (None, "tiny-bf16", "a100-bf16"):
            raise ValueError("unknown vLLM rehearsal profile")
        self.rehearsal_profile = rehearsal_profile
        self.protocol_version = protocol_version
        self.context_limit = CONTEXT_LIMIT
        if generation_context_limit is not None:
            if protocol_version != "error-note-v18b" or generation_context_limit not in (CONTEXT_LIMIT, 40960):
                raise ValueError("only the fixed v18b prep and monitor context lengths are supported")
            self.context_limit = generation_context_limit
        self.preflight_stage = preflight_stage
        if protocol_version in ("error-note-v17", "error-note-v18", "error-note-v18b"):
            if type(max_num_seqs) is not int or not 1 <= max_num_seqs <= 16:
                raise ValueError("v17 training concurrency ceiling is sixteen")
            if preflight_stage not in ("f0", "trained"):
                raise ValueError("v17 PF5 stage must be f0 or trained")
        self.admission_wait_seconds = 0.0
        self.preemption_count = 0
        self._last_preemption_report = 0
        self.version_registry = bool(version_registry)
        if self.version_registry and (not lora or sleep):
            raise ValueError("v16b vLLM: version registry requires LoRA and sleep off")
        self.sampling = dict(SINGLE_SAMPLING if sampling is None else sampling)
        self.versions: dict[int, dict] = {}
        self.pending: dict[str, dict] = {}
        self._ready_outputs: list[tuple[dict, dict]] = []
        self.coexistence_reports: list[dict] = []
        self.engine_steps = 0
        self.visual_version: int | None = None
        self._input_memo: dict[str, tuple[str, dict, Any]] = {}
        self._image_memo: dict[str, dict] = {}
        self.image = dict(IMAGE if image is None else image)
        low, high = self.image.get("min_pixels"), self.image.get("max_pixels")
        if not (isinstance(low, int) and isinstance(high, int) and 0 < low <= high):
            raise ValueError("SINGLE vLLM: image config needs positive integer min_pixels <= max_pixels")
        from importlib import import_module
        from importlib.metadata import version

        from fine_tuning.notebook_rl.engines import InputPreparation, VLLMRolloutEngine, _weight_inventory, vllm_kwargs

        if version("vllm") != "0.11.0":
            raise RuntimeError("SINGLE vLLM: vllm==0.11.0 required")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine: weight push and add_request/step
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                                  min_pixels=self.image["min_pixels"], max_pixels=self.image["max_pixels"])
        processor.image_processor.size = {"shortest_edge": self.image["min_pixels"], "longest_edge": self.image["max_pixels"]}
        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)  # reuse prepare/_prompt/_load_weights, own LLM kwargs
        InputPreparation.__init__(engine, processor, AutoConfig.from_pretrained(str(model_path), local_files_only=True),
                                  self.image, self.context_limit)
        if self.rehearsal_profile == "tiny-bf16" and (
                engine.model_config.text_config.num_hidden_layers != 2 or engine.model_config.vision_config.depth != 3
                or engine.model_config.text_config.hidden_size != 256 or engine.model_config.vision_config.hidden_size != 128):
            raise ValueError("tiny rehearsal numerical admission requires the fixed synthetic model")
        engine._weight_shapes = _weight_inventory(engine.model_config)
        engine.model_dir = Path(model_path).resolve()
        kwargs = vllm_kwargs(Path(model_path), self.image, int(kv_cache_gib * 2**30), max_num_seqs, self.context_limit,
                             gpu_memory_utilization=float(gpu_memory_utilization))
        extra: dict = {}
        if lora:
            self.pf0 = llm_lora.host_memory(bool(sleep))
            if not self.pf0["pass"]:
                self._report({"pf0": self.pf0, "pass": False})
                raise RuntimeError("SINGLE vLLM: PF0 host memory below the level-1 sleep requirement")
            llm_lora.scope_qwen3vl_lora()
            extra = dict(llm_lora.ENGINE_ARGS_V16B if version_registry else llm_lora.ENGINE_ARGS)
            if extra.keys() & kwargs.keys():
                raise ValueError("SINGLE vLLM: LoRA engine arguments overlap the pinned vLLM arguments")
        if version_registry:
            kwargs.update(enable_prefix_caching=False, mm_processor_cache_gb=0, disable_log_stats=False)
        with _nondeterministic():
            engine.llm = import_module("vllm").LLM(**kwargs, logprobs_mode="processed_logprobs",
                                                   max_logprobs=TAIL_LOGPROBS if version_registry else TOP_LOGPROBS,
                                                   enable_sleep_mode=bool(sleep), **extra)
        engine._call, engine._synchronized, engine._push_complete, engine._last_loaded_version = 0, True, True, None
        self.engine, self.max_num_seqs, self.sleep = engine, max_num_seqs, bool(sleep)
        if self.protocol_version in ("error-note-v17", "error-note-v18", "error-note-v18b"):
            cache = self._scheduler().kv_cache_config
            if len(cache.kv_cache_groups) != 1:
                raise RuntimeError("v17 capacity: one measured full-attention KV group required")
            self.kv_block_size = cache.kv_cache_groups[0].kv_cache_spec.block_size
            self.actual_kv_tokens = cache.num_blocks * self.kv_block_size
            if type(self.actual_kv_tokens) is not int or self.actual_kv_tokens < 1:
                raise RuntimeError("v17 capacity: actual engine KV token capacity unavailable")
        self.eos_token_ids = sorted(set(eos_token_ids))
        self.loaded_policy: str | None = None  # the disk snapshot is F0, but a push always precedes generation
        self.asleep = False
        if lora:
            self.lora = True
            self.frozen_fp = llm_lora.base_fingerprint(engine.llm)  # before any push
            self.vision_hooks = engine.llm.apply_model(llm_lora.install_vision_hooks)[0]
            pf1 = self._scope()
            if not pf1["pass"]:
                self._report({"pf0": self.pf0, "pf1": pf1, "pass": False})
                raise RuntimeError("SINGLE vLLM: PF1 LoRA scope check failed")
            self.lora_dir = Path(tempfile.mkdtemp(prefix="single-vllm-lora-"))
            self._lora_id, self._preflight_done = 0, False
            self.preflight_report: dict | None = None

    def _scheduler(self) -> Any:
        return self.engine.llm.llm_engine.engine_core.engine_core.scheduler

    def has_unfinished_requests(self) -> bool:
        return bool(self.pending) or bool(self._ready_outputs) or self.engine.llm.llm_engine.has_unfinished_requests()

    def prefill_barrier(self) -> bool:
        scheduler = self._scheduler()
        return not scheduler.waiting and all(r.num_computed_tokens >= len(r.prompt_token_ids)
                                             for r in scheduler.running)

    def live_registry(self) -> dict:
        return {str(v): {"lora_int_id": e["lora_int_id"], "policy_sha256": e["policy_sha256"],
                         "refcount": e["refcount"]} for v, e in self.versions.items()}

    def release_version(self, version: int) -> None:
        entry = self.versions[version]
        if entry["refcount"]:
            raise RuntimeError("v16b vLLM: cannot remove a referenced policy")
        if self.engine.llm.llm_engine.remove_lora(entry["lora_int_id"]) is not True:
            raise RuntimeError("v16b vLLM: policy removal not acknowledged")
        if self.lora_request is not None and self.lora_request.lora_int_id == entry["lora_int_id"]:
            self.lora_request, self.loaded_policy = None, None
        shutil.rmtree(entry["request"].lora_path)
        del self.versions[version]

    def release_idle_versions(self, *, keep: int | None = None) -> None:
        for version, entry in list(self.versions.items()):
            if entry["refcount"] == 0 and version != keep:
                self.release_version(version)

    def add_version(self, bundle: Any, version: int) -> str:
        """Push only between engine steps; retain immutable slots until the last request completes."""
        from fine_tuning.notebook_rl.policy import merged_named_tensors
        from .rl import policy_sha256
        from vllm.lora.request import LoRARequest

        if not self.version_registry or version in self.versions or not self.prefill_barrier():
            raise RuntimeError("v16b vLLM: registry, fresh version and visual-prefill barrier required")
        policy = policy_sha256(bundle)
        previous = self.visual_version if version in (1, 2) else None
        if previous in self.versions:
            self.versions[previous]["refcount"] += 1
            buffered = list(self._ready_outputs)
            self._ready_outputs.clear()
            while self.pending:
                buffered.extend(self.step())
            self._ready_outputs = buffered
        self.release_idle_versions()
        if len(self.versions) >= 2 or llm_lora.free_lora_slots(self.engine) < 1:
            raise RuntimeError("v16b vLLM: no safe LoRA slot; add_lora was not called")
        if not self._preflight_done:
            if self.pending:
                raise RuntimeError("v16b vLLM: preflight must precede requests")
            self.sync(bundle)
            request = self.lora_request
        else:
            self._lora_id += 1
            path = self.lora_dir / str(self._lora_id)
            llm_lora.write_llm_adapter(bundle, path)
            request = LoRARequest(f"v16b-{version}-{policy[:16]}", self._lora_id, str(path))
            if self.engine.llm.llm_engine.add_lora(request) is not True:
                raise RuntimeError("v16b vLLM: policy registration not acknowledged")
        entry = {"request": request, "lora_int_id": request.lora_int_id, "refcount": 0, "policy_sha256": policy,
                 "slots": {name: [(a.detach().cpu().clone(), b.detach().cpu().clone()) for a, b in pairs]
                           for name, pairs in llm_lora.llm_slots(bundle).items()}}
        self.versions[version] = entry
        for state in self.pending.values():
            live = state["live"]
            if not live.output_token_ids:
                state["mixed_version"] = True
                state["visual_version"] = version
        with _nondeterministic():
            self.vision_hooks.load(llm_lora.vision_lora(bundle))
            llm_lora.push_mergers(self.engine, merged_named_tensors(bundle), version, inflight=True)
        report = llm_lora.check_all_live_identity(self.engine, self.versions, bundle=bundle, hooks=self.vision_hooks)
        if report["pass"] is not True:
            raise RuntimeError(f"v16b vLLM: live identity mismatch: {report}")
        self.visual_version, self.loaded_policy, self.lora_request = version, policy, request
        if previous in self.versions:
            try:
                self.coexistence_reports.append(self.verify_coexistence(previous, version))
            finally:
                self.versions[previous]["refcount"] -= 1
            self.release_idle_versions(keep=version)
        return policy

    def verify_coexistence(self, old_version: int, new_version: int) -> dict:
        """GPU gate: old/new LLM LoRA alone and mixed, with the same current global visual weights. The old adapter's
        greedy probe completion is teacher-forced (prompt_logprobs, as _probe) in [old], [new] and [old, new] batches;
        a version passes when its mean |alone - mixed| log-prob is <= 0.05. Greedy token identity is logged only:
        LoRA split-K atomics and batch-1/batch-2 kernels are not bit-reproducible, so BF16 argmax ties may flip."""
        import time
        from vllm import SamplingParams

        if self.pending or self.engine.llm.llm_engine.has_unfinished_requests():
            raise RuntimeError("v16b vLLM: coexistence probes require the bounded startup drain")
        started = time.monotonic()
        probe = probe_requests(self.image)[0]
        prepared = self.engine.prepare(probe)
        greedy = SamplingParams(n=1, temperature=0.0, max_tokens=PROBE_TOKENS, ignore_eos=True,
                                detokenize=False, seed=3407)
        forced = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=0, presence_penalty=0.0, repetition_penalty=1.0,
                                max_tokens=1, seed=3407, prompt_logprobs=0, detokenize=False)
        old = self.versions[old_version]["request"]
        new = self.versions[new_version]["request"]

        def run(adapters: list, ids: list[int], params: Any) -> list:
            prompts = [self.engine._prompt(prepared, ids) for _ in adapters]
            with _nondeterministic():
                outputs = self.engine.llm.generate(prompts, params, lora_request=adapters, use_tqdm=False)
            if len(outputs) != len(adapters):
                raise RuntimeError("v16b vLLM: missing coexistence probe output")
            return outputs

        def tokens(adapters: list) -> list[list[int]]:
            return [list(output.outputs[0].token_ids) for output in run(adapters, prepared.token_ids, greedy)]

        greedy_alone, greedy_mixed = tokens([old]) + tokens([new]), tokens([old, new])
        completion = greedy_alone[0]
        if len(completion) != PROBE_TOKENS:
            raise RuntimeError("v16b vLLM: coexistence greedy probe length differs")
        forced_ids = list(prepared.token_ids) + completion

        def scores(adapters: list) -> list[list[float]]:
            result = []
            for output in run(adapters, forced_ids, forced):
                if list(output.prompt_token_ids) != forced_ids or output.prompt_logprobs is None:
                    raise RuntimeError("v16b vLLM: coexistence probe input mismatch or missing teacher-forced log-probs")
                rows = output.prompt_logprobs[len(prepared.token_ids):]
                result.append([float(row[t].logprob) for row, t in zip(rows, completion, strict=True)])
            return result

        singles, mixed = scores([old]) + scores([new]), scores([old, new])
        results = []
        for index, version in enumerate((old_version, new_version)):
            deltas = [abs(a - b) for a, b in zip(singles[index], mixed[index], strict=True)]
            mean = sum(deltas) / len(deltas)
            results.append({"version": version, "same_token_ids": greedy_alone[index] == greedy_mixed[index],
                            "mean_logprob_delta": mean, "max_logprob_delta": max(deltas),
                            "pass": math.isfinite(mean) and mean <= .05})
        report = {"old_version": old_version, "new_version": new_version, "visual_version": self.visual_version,
                  "synthetic": True, "training_use": False, "buffered_completions": len(self._ready_outputs),
                  "method": "teacher-forced-old-greedy-v1", "tolerance_mean_abs_logprob": .05,
                  "limitation": ("one synthetic noise-image probe; teacher-forced raw log-probs of one fixed old-adapter "
                                 "greedy completion, mean tolerance 0.05; same_token_ids is a diagnostic, not a gate; "
                                 "does not certify sampled task outputs"),
                  "probe_seconds": time.monotonic() - started, "versions": results,
                  "pass": all(result["pass"] for result in results)}
        if report["pass"] is not True:
            self._report({"coexistence": report, "pass": False})
            raise RuntimeError("v16b vLLM: old/new policy mixed-batch coexistence check failed")
        return report

    def forget_item(self, problem_id: str) -> None:
        self._image_memo.pop(problem_id, None)
        for key in list(self._input_memo):
            if key.startswith(problem_id + ":"):
                del self._input_memo[key]

    def _prepared_input(self, request: dict, image_identity: list[dict]) -> tuple[dict, list]:
        from fine_tuning.notebook_rl.input_preparation import prepare_inputs, prepare_protocol_inputs
        from .rl import digest

        pid = request["problem_id"]
        image_sha = digest(image_identity)
        memo = self._image_memo.setdefault(pid, {"image_sha256": image_sha})
        if memo["image_sha256"] != image_sha:
            raise RuntimeError("v16b vLLM: a problem's image bytes changed")
        processor = self.engine.processor
        original = processor.image_processor

        class CachedImageProcessor:
            def __getattr__(self, name: str) -> Any:
                return getattr(original, name)

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                if "image_inputs" not in memo:
                    memo["image_inputs"] = original(*args, **kwargs)
                return memo["image_inputs"]

        processor.image_processor = CachedImageProcessor()
        try:
            if "processed" not in memo:
                item, processed = prepare_protocol_inputs(processor, self.engine.model_config,
                    request["messages"], request["images"], self.image)
                memo["processed"] = processed
            else:
                item = prepare_inputs(processor, self.engine.model_config, request["messages"],
                                      memo["processed"], do_resize=False)
            pixel_sha = digest({key: value for key, value in item["input_tensors"].items()
                                if key in ("pixel_values", "image_grid_thw")})
            if "pixel_sha256" in memo and memo["pixel_sha256"] != pixel_sha:
                raise RuntimeError("v16b vLLM: memoized image tensors changed")
            memo["pixel_sha256"] = pixel_sha
            return item, memo["processed"]
        finally:
            processor.image_processor = original

    def _prepare_request(self, request: dict) -> dict:
        if request.get("evaluation_monitor") and getattr(self, "protocol_version", None) != "error-note-v18b":
            raise ValueError("held-out B2 monitor requires its v18b generation engine")
        from fine_tuning.notebook_rl.engines import RolloutRequest
        from .rl import digest

        version, rid = request["policy_version"], request["request_id"]
        image_identity = [{"size": list(img.size), "sha256": hashlib.sha256(img.tobytes()).hexdigest()}
                          for img in request["images"]]
        cache_binding = {"messages": request["messages"], "images": image_identity}
        if getattr(self, "protocol_version", None) in ("error-note-v17", "error-note-v18", "error-note-v18b"):
            cache_binding.update(image_policy=self.image, processor_revision=str(self.engine.model_dir),
                                 context_limit=getattr(self, "context_limit", CONTEXT_LIMIT))
        key = request["problem_id"] + ":" + digest(cache_binding)
        memo = self._input_memo.get(key)
        if memo is None:
            item, processed = self._prepared_input(request, image_identity)
            image_key = hashlib.sha256(json.dumps([(tuple(img["size"]), img["sha256"]) for img in image_identity],
                                                  separators=(",", ":")).encode()).hexdigest()
            self.engine._image_key, self.engine._processed_images = image_key, processed
            self.engine._pixels = {k: item["inputs"][k] for k in ("pixel_values", "image_grid_thw")}
            prepared = self.engine.prepare(RolloutRequest(rid, list(item["prompt_token_ids"]), None, request["images"],
                                                         1, request["max_tokens"], [request["seed"]]))
            if prepared.token_ids != list(item["prompt_token_ids"]):
                raise RuntimeError("v16b vLLM: HF and engine prompt IDs differ")
            prompt = self.engine._prompt(prepared, prepared.token_ids)
            self._input_memo[key] = (item["input_sha256"], item, prompt)
        else:
            old_input_sha, item, prompt = memo
            if old_input_sha != item["input_sha256"]:
                raise RuntimeError("v16b vLLM: memo input hash changed")
        prompt = dict(prompt)
        if request["images"]:
            prompt["multi_modal_uuids"] = {"image": [f"v{version}:{rid}:{i}" for i in range(len(request["images"]))]}
        if item["input_tokens"] > 16384:
            raise RuntimeError("v16b vLLM: HOLD prompt exceeds the measured 16384-token capacity bound")
        if item["input_tokens"] + request["max_tokens"] > getattr(self, "context_limit", CONTEXT_LIMIT):
            raise ValueError("v16b vLLM: prompt plus completion exceeds context limit")
        return {"request_sha256": single_request_sha256(request), "item": item, "prompt": prompt,
                "prompt_tokens": item["input_tokens"], "max_tokens": request["max_tokens"]}

    def prepare_request_v17(self, request: dict) -> dict:
        if getattr(self, "protocol_version", None) not in ("error-note-v17", "error-note-v18", "error-note-v18b"):
            raise RuntimeError("v17 preparation requires its explicit protocol")
        single_request_binding(request)
        prepared = self._prepare_request(request)
        self._reservation_v17(prepared)
        return prepared

    def _reservation_v17(self, prepared: dict) -> dict:
        from .v16b_runtime import reserve_request_v17

        if prepared["prompt_tokens"] != len(prepared["item"]["prompt_token_ids"]):
            raise RuntimeError("v17 capacity: reservation differs from actual prepared prompt tokens")
        return reserve_request_v17(prepared, self.actual_kv_tokens, self.kv_block_size,
                                  [s["kv_reserved_tokens"] for s in self.pending.values()],
                                  active=len(self.pending), ceiling=self.max_num_seqs)

    def can_submit_v17(self, prepared: dict) -> bool:
        return self._reservation_v17(prepared)["status"] == "ADMIT"

    def heartbeat_v17(self) -> dict:
        scheduler = self._scheduler()
        preemptions = self.preemption_count - self._last_preemption_report
        self._last_preemption_report = self.preemption_count
        return {"train_KV_capacity": self.actual_kv_tokens,
                "reserved_tokens": sum(s["kv_reserved_tokens"] for s in self.pending.values()),
                "active": len(scheduler.running), "waiting": len(scheduler.waiting),
                "admission_wait_seconds": self.admission_wait_seconds, "preemption_delta": preemptions,
                "warning": "TRAIN_PREEMPTION" if preemptions else None}

    def abort_request_v17(self, rid: str) -> None:
        if rid not in self.pending:
            raise RuntimeError("v17 capacity: cannot release an unknown or completed request")
        self.engine.llm.llm_engine.abort_request([rid])
        state = self.pending.pop(rid)
        if "original_event" in state:
            state["live"].record_event = state["original_event"]
        self.versions[state["request"]["policy_version"]]["refcount"] -= 1

    def set_preflight_stage_v17(self, stage: str) -> None:
        if getattr(self, "protocol_version", None) not in ("error-note-v17", "error-note-v18", "error-note-v18b") or stage not in ("f0", "trained"):
            raise ValueError("v17 PF5 requires an explicit f0 or trained stage")
        if self.pending:
            raise RuntimeError("v17 PF5 stage changes require a drained engine")
        if stage != self.preflight_stage:
            self.preflight_stage = stage
            self._preflight_done, self.loaded_policy = False, None

    def submit(self, request: dict, *, prepared: dict | None = None) -> None:
        """Admit exactly one request against its immutable policy and full-vocabulary sampling."""
        from vllm import SamplingParams
        single_request_binding(request)
        version, rid = request["policy_version"], request["request_id"]
        if rid in self.pending or version not in self.versions:
            raise ValueError("v16b vLLM: duplicate request or unknown policy version")
        entry = self.versions[version]
        if request["policy_sha256"] != entry["policy_sha256"]:
            raise ValueError("v16b vLLM: admission policy hash changed")
        if prepared is None:
            prepared = self._prepare_request(request)
        if prepared["request_sha256"] != single_request_sha256(request):
            raise RuntimeError("v17 capacity: prepared input belongs to a different request")
        item, prompt = prepared["item"], prepared["prompt"]
        if prepared["max_tokens"] != request["max_tokens"]:
            raise RuntimeError("v17 capacity: prepared completion cap changed")
        # The engine transport collapses each registered image span (a run of image tokens) to one token that vLLM
        # re-expands (engines.VLLMRolloutEngine._prompt); the reserved count stays the expanded prompt length.
        ids, image_token = list(item["prompt_token_ids"]), self.engine.model_config.image_token_id
        collapsed = [t for i, t in enumerate(ids) if not (t == image_token and i and ids[i - 1] == image_token)]
        if list(prompt.get("prompt_token_ids", collapsed)) != collapsed:
            raise RuntimeError("v17 capacity: native prompt differs from reserved prepared prompt")
        reservation = None
        if getattr(self, "protocol_version", None) in ("error-note-v17", "error-note-v18", "error-note-v18b"):
            reservation = self._reservation_v17(prepared)
            if reservation["status"] != "ADMIT":
                raise RuntimeError("v17 capacity: scheduler must pump existing requests before admission")
        sampling = dict(request.get("sampling", self.sampling))
        params = SamplingParams(n=1, **sampling, repetition_penalty=1.0, max_tokens=request["max_tokens"],
                                seed=request["seed"], stop_token_ids=self.eos_token_ids,
                                logprobs=TAIL_LOGPROBS, detokenize=False)
        core = self.engine.llm.llm_engine
        try:
            with _nondeterministic():
                core.add_request(rid, prompt, params, lora_request=entry["request"])
            live = self._scheduler().requests[rid]
            if live.lora_request is None or live.lora_request.lora_int_id != entry["lora_int_id"]:
                raise RuntimeError("v16b vLLM: request LoRA differs from the admission version")
        except BaseException:
            if getattr(self, "protocol_version", None) in ("error-note-v17", "error-note-v18", "error-note-v18b"):
                core.abort_request([rid])
            raise
        self.pending[rid] = {"request": request, "item": item, "live": live, "admit_step": self.engine_steps,
                             "visual_version": self.visual_version, "mixed_version": version != self.visual_version,
                             "preempted": False, "last_computed": 0}
        if reservation is not None:
            self.pending[rid]["kv_reserved_tokens"] = reservation["request_reserved_tokens"]
        state = self.pending[rid]
        original_event = live.record_event
        state["original_event"] = original_event
        def record_event(event_type: Any, timestamp: Any = None) -> None:
            if event_type.name == "PREEMPTED":
                if getattr(self, "protocol_version", None) in ("error-note-v17", "error-note-v18", "error-note-v18b"):
                    self.preemption_count += 1
                    warnings.warn("v17 training KV preemption; policy mixing is checked separately", RuntimeWarning)
                state["preempted"] = True
                if state["visual_version"] != self.visual_version:
                    state["mixed_version"] = True
                    state["visual_version"] = self.visual_version
            original_event(event_type, timestamp)
        live.record_event = record_event
        entry["refcount"] += 1

    def step(self) -> list[tuple[dict, dict]]:
        from .rl import sampling_sha256

        if self._ready_outputs:
            ready, self._ready_outputs = self._ready_outputs, []
            return ready
        with _nondeterministic():
            outputs = self.engine.llm.llm_engine.step()
        self.engine_steps += 1
        finished = []
        for state in self.pending.values():
            live = state["live"]
            status = getattr(getattr(live, "status", None), "name", "")
            preempted = status == "PREEMPTED" or live.num_computed_tokens < state["last_computed"]
            if preempted:
                if (getattr(self, "protocol_version", None) in ("error-note-v17", "error-note-v18", "error-note-v18b")
                        and not state["preempted"] and "original_event" not in state):
                    self.preemption_count += 1
                    warnings.warn("v17 training KV preemption; policy mixing is checked separately", RuntimeWarning)
                state["preempted"] = True
                if state["visual_version"] != self.visual_version:
                    state["mixed_version"] = True
                    state["visual_version"] = self.visual_version
            state["last_computed"] = live.num_computed_tokens
        for output in outputs:
            if not output.finished:
                continue
            state = self.pending.pop(output.request_id)
            if "original_event" in state:
                state["live"].record_event = state["original_event"]
            request, item = state["request"], state["item"]
            if getattr(self, "protocol_version", None) in ("error-note-v17", "error-note-v18", "error-note-v18b"):
                self.versions[request["policy_version"]]["refcount"] -= 1
            if len(output.outputs) != 1 or list(output.prompt_token_ids) != list(item["prompt_token_ids"]):
                raise RuntimeError("v16b vLLM: exact prompt and one completion required")
            result = output.outputs[0]
            ids = list(result.token_ids)
            if result.logprobs is None or len(result.logprobs) != len(ids):
                raise RuntimeError("v16b vLLM: one selected processed probability required per token")
            if result.finish_reason not in ("stop", "length") or (result.finish_reason == "stop"
                    and (not ids or ids[-1] not in self.eos_token_ids)):
                raise RuntimeError("v16b vLLM: completion must end at EOS or the token cap")
            sampling = dict(request.get("sampling", self.sampling))
            full = sampling["top_p"] == 1.0 and sampling["top_k"] == 0
            text = self.engine.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            entry = self.versions[request["policy_version"]]
            generation = single_request_binding(request) | {
                "kind": request["kind"], "prompt_token_ids": list(item["prompt_token_ids"]),
                "completion_token_ids": ids, "sampling_logprobs":
                    [sampled_logprob(t, row) for t, row in zip(ids, result.logprobs, strict=True)],
                "sampling": sampling, "sampling_sha256": sampling_sha256(sampling),
                "probability_space": "processed-full-vocab-v1" if full else "processed-truncated-b2",
                "policy_sha256": entry["policy_sha256"], "policy_version": request["policy_version"],
                "lora_int_id": entry["lora_int_id"], "visual_version": state["visual_version"],
                "mixed_version": state["mixed_version"], "preempted": state["preempted"],
                "admit_step": state["admit_step"], "finish_step": self.engine_steps,
                "protocol_sha256": request["protocol_sha256"], "input_sha256": item["input_sha256"],
                "request_sha256": request["request_sha256"], "text": text, "finish_reason": result.finish_reason,
                "eos_token_ids": self.eos_token_ids, "completion_count": 1,
                "tail_diagnostic_logprobs": TAIL_LOGPROBS if full else 20,
                "b2_tail_flags": [b2_tail_flag(t, row, vocab_size=self.engine.model_config.text_config.vocab_size)
                                  if full else False for t, row in zip(ids, result.logprobs, strict=True)]}
            for key in ("baseline_receipt", "obs_baseline_receipt"):
                if key in request:
                    generation[key] = request[key]
            if getattr(self, "protocol_version", None) not in ("error-note-v17", "error-note-v18", "error-note-v18b"):
                entry["refcount"] -= 1
            if request["kind"] == "RESOLVE":
                self.forget_item(request["problem_id"])
            finished.append((request, {"text": text, "finish_reason": result.finish_reason, "completion_tokens": len(ids),
                                      "prompt_tokens": len(item["prompt_token_ids"]), "completion_count": 1,
                                      "generation": generation}))
        self.release_idle_versions(keep=self.visual_version)
        return finished

    def release(self) -> None:
        """Free vLLM weights and KV cache before an HF update (sleep level 2: weights are re-pushed anyway; level 1 in
        LoRA mode: the weight pool also holds the LoRA slots and is restored bit for bit)."""
        if self.sleep and not self.asleep:
            import torch

            self.engine.llm.sleep(level=1 if self.lora else 2)
            self.asleep, self.loaded_policy = True, None
            torch.cuda.empty_cache()

    def sync(self, bundle: Any) -> str:
        from fine_tuning.notebook_rl.policy import merged_named_tensors

        from .rl import policy_sha256

        policy = policy_sha256(bundle)
        if policy != self.loaded_policy:
            if self.asleep:
                import torch

                torch.cuda.empty_cache()
                self.engine.llm.wake_up()
                self.asleep = False
            if self.lora:
                with _nondeterministic():
                    self.vision_hooks.load(llm_lora.vision_lora(bundle))
                    llm_lora.push_mergers(self.engine, merged_named_tensors(bundle), self.engine.policy_version + 1)
                    self._swap_adapter(bundle, policy)
                if not self._preflight_done:
                    self.verify_lora(bundle)
                    self._preflight_done = True
            else:
                with _nondeterministic():
                    self.engine._load_weights(merged_named_tensors(bundle), self.engine.policy_version + 1)
            self.loaded_policy = policy
        return policy

    def _swap_adapter(self, bundle: Any, policy: str, factor: float = 1) -> Any:
        """A fresh, never reused id per policy: vLLM's LRU manager only touches an id it already knows. `factor` != 1
        loads the PF5 amplified probe adapter (B x factor) under its own fresh id."""
        from vllm.lora.request import LoRARequest

        new_id = self._lora_id + 1
        self._lora_id = new_id
        path = self.lora_dir / str(new_id)
        llm_lora.write_llm_adapter(bundle, path, factor)
        core, old = self.engine.llm.llm_engine, self.lora_request
        self.lora_request = None
        if old is not None:
            if core.remove_lora(old.lora_int_id) is not True:
                raise RuntimeError("SINGLE vLLM: previous LoRA not removed")
            shutil.rmtree(old.lora_path)
        name = f"policy-{policy[:16]}" if factor == 1 else f"probe-x{factor:g}-{policy[:16]}"
        request = LoRARequest(name, new_id, str(path))
        if core.add_lora(request) is not True or core.list_loras() != {new_id}:
            raise RuntimeError("SINGLE vLLM: policy LoRA not loaded as the only adapter")
        self.lora_request = request
        return request

    def _report(self, report: dict) -> None:
        self.preflight_report = report
        print("SINGLE_LORA_PREFLIGHT " + json.dumps(report, sort_keys=True, default=str), file=sys.stderr, flush=True)

    def _scope(self) -> dict:
        config = self.engine.model_config
        return llm_lora.check_scope(self.engine.llm, self.vision_hooks, config.text_config.num_hidden_layers,
                                    config.vision_config.depth)

    def _probe(self, request: Any, completion: list[int], lora: Any) -> list[float]:
        """Teacher-forced raw log-probs (engines.probe_logprobs recipe) with or without the policy LoRA."""
        from vllm import SamplingParams

        prepared = self.engine.prepare(request)
        ids = prepared.token_ids + completion
        params = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=0, presence_penalty=0.0, repetition_penalty=1.0,
                                max_tokens=1, seed=request.seeds[0], prompt_logprobs=0, detokenize=False)
        output = self.engine.llm.generate([self.engine._prompt(prepared, ids)], params, use_tqdm=False,
                                          lora_request=lora)[0]
        if list(output.prompt_token_ids) != ids or output.prompt_logprobs is None:
            raise RuntimeError("SINGLE vLLM: probe input mismatch or missing teacher-forced log-probs")
        rows = output.prompt_logprobs[len(prepared.token_ids):]
        return [float(row[t].logprob) for row, t in zip(rows, completion, strict=True)]

    def _pf5(self, bundle: Any) -> tuple[dict, Any]:
        """llm_lora.slope_check on greedy probe completions of the engine as it generates (real and amplified adapter,
        llm_lora.vllm_columns); returns the report and the request of the real adapter restored under a fresh id."""
        from vllm import SamplingParams

        from fine_tuning.notebook_rl.engines import VLLMRolloutEngine

        from .rl import policy_sha256

        pairs = []
        for request in probe_requests(self.image):
            prepared = self.engine.prepare(request)
            greedy = SamplingParams(n=1, temperature=0.0, max_tokens=PROBE_TOKENS, ignore_eos=True, detokenize=False)
            with _nondeterministic():
                output = self.engine.llm.generate([self.engine._prompt(prepared, prepared.token_ids)], greedy,
                                                  use_tqdm=False, lora_request=self.lora_request)[0]
            completion = list(output.outputs[0].token_ids)
            if len(completion) != PROBE_TOKENS:
                raise RuntimeError("SINGLE vLLM: PF5 greedy probe length differs")
            pairs.append((request, completion))
        policy = policy_sha256(bundle)
        with _nondeterministic():
            columns, restored = llm_lora.vllm_columns(bundle, self.vision_hooks, self._probe, pairs, self.lora_request,
                                                      lambda factor: self._swap_adapter(bundle, policy, factor))
        reference: Any = llm_lora.hf_columns(bundle, self.engine, pairs)
        if getattr(self, "protocol_version", None) in ("error-note-v17", "error-note-v18", "error-note-v18b"):
            contract = llm_lora.pf5_contract_v17(bundle, self.preflight_stage)
            if getattr(self, "rehearsal_profile", None) == "tiny-bf16":
                contract = contract | {"reportable": False,
                    "numerical_admission": "TINY_REHEARSAL_ONLY"}
            tower = None
            if self.protocol_version in ("error-note-v18", "error-note-v18b") and self.preflight_stage == "trained":
                from types import SimpleNamespace
                with _nondeterministic():
                    vision_reference = llm_lora.hf_vision_columns(bundle, self.engine, pairs)
                    vision_actual = llm_lora.vllm_vision_columns(
                        SimpleNamespace(llm=self.engine.llm, vision_hooks=self.vision_hooks), bundle, self.engine, pairs)
                tower = llm_lora.vision_tower_check(vision_reference, vision_actual)
            report = llm_lora.slope_check(**columns, **reference, cap=VLLMRolloutEngine.sync_threshold_cap,
                                         require_power=contract["require_power"], vision_tower=tower)
            report["stage_contract"] = contract
            report["transport_pass"] = report["pass"]
            report["effect_status"] = ("PASS" if report["power"] and report["pass"] else
                                       "UNMEASURED" if contract["require_power"] else "NOT_APPLICABLE_F0")
            if getattr(self, "rehearsal_profile", None) == "tiny-bf16" and (not report["power"] or not report["pass"]):
                report["effect_status"] = "NUMERICAL_INCONCLUSIVE_TINY_REHEARSAL"
            return report, restored
        return llm_lora.slope_check(**columns, **reference, cap=VLLMRolloutEngine.sync_threshold_cap,
                                    require_power=not self.version_registry), restored

    def verify_lora(self, bundle: Any) -> dict:
        """PF1-PF5 (PF0 ran at start-up); prints SINGLE_LORA_PREFLIGHT {json} and raises on any failure. PF5 tests the
        application mechanism on an amplified adapter (llm_lora.slope_check; the real adapter's slope is a diagnostic),
        then the real adapter returns under a fresh id and PF2 is repeated bit-exactly (pf2_restored)."""
        import torch

        llm = self.engine.llm
        report: dict = {"pf0": self.pf0, "pf1": self._scope(), "lora_int_id": self.lora_request.lora_int_id}
        if report["pf1"]["pass"]:
            lora_id = self.lora_request.lora_int_id
            report["pf2"] = llm_lora.check_identity(llm, bundle, lora_id, self.vision_hooks)
            report["pf3"] = llm_lora.check_base(llm, bundle, self.frozen_fp)
            if self.sleep:
                with _nondeterministic():
                    llm.sleep(level=1)
                    torch.cuda.empty_cache()
                    llm.wake_up()
                identity = llm_lora.check_identity(llm, bundle, lora_id, self.vision_hooks)
                base = llm_lora.check_base(llm, bundle, self.frozen_fp)
                report["pf4"] = {"identity": identity["pass"], "base": base["pass"],
                                 "pass": identity["pass"] and base["pass"]}
            else:
                report["pf4"] = {"pass": None, "reason": "sleep mode off"}
            report["pf5"], restored = self._pf5(bundle)
            report["restored_lora_int_id"] = restored.lora_int_id
            report["pf2_restored"] = llm_lora.check_identity(llm, bundle, restored.lora_int_id, self.vision_hooks)
        report["pass"] = bool(report["pf1"]["pass"]) and all(
            report[k]["pass"] is True for k in ("pf2", "pf3", "pf5", "pf2_restored")) and report["pf4"]["pass"] is not False
        if getattr(self, "rehearsal_profile", None) == "tiny-bf16":
            from .v18b_rehearsal import tiny_rehearsal_transport
            tiny_rehearsal_transport(report)
        self._report(report)
        if not report["pass"]:
            raise RuntimeError("SINGLE vLLM: LoRA preflight failed")
        return report

    def stream_builder(self, bundle: Any, args: argparse.Namespace, stats: StreamStats) -> Stream:
        from vllm import SamplingParams

        from fine_tuning.notebook_rl.engines import RolloutRequest
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

        from .rl import digest, sampling_sha256

        policy, protocol = args.policy_sha256, args.protocol_sha256
        if self.sync(bundle) != policy:
            raise ValueError("SINGLE vLLM: stale live policy")
        engine, sampling_hash = self.engine, sampling_sha256(SINGLE_SAMPLING)
        label = llm_lora.ENGINE_LORA if self.lora else ENGINE
        stats.receipt_extra.update(single_sampler=label, single_batch_rows=self.max_num_seqs)
        eos = set(self.eos_token_ids)

        def prepare(request: dict) -> tuple[dict, Any, dict]:
            binding = single_request_binding(request)
            item, _ = prepare_protocol_inputs(engine.processor, engine.model_config, request["messages"],
                                              request["images"], self.image)
            if item["input_tokens"] + request["max_tokens"] > getattr(self, "context_limit", CONTEXT_LIMIT):
                raise ValueError("SINGLE vLLM: prompt plus completion exceeds the context limit")
            prepared = engine.prepare(RolloutRequest(binding["request_id"], None, request["messages"], request["images"],
                                                     1, request["max_tokens"], [request["seed"]]))
            if prepared.token_ids != list(item["prompt_token_ids"]):
                raise RuntimeError("SINGLE vLLM: engine prompt IDs differ from the HF training input")
            params = SamplingParams(n=1, temperature=SINGLE_SAMPLING["temperature"], top_p=SINGLE_SAMPLING["top_p"],
                                    top_k=SINGLE_SAMPLING["top_k"], presence_penalty=SINGLE_SAMPLING["presence_penalty"],
                                    repetition_penalty=1.0, max_tokens=request["max_tokens"], seed=request["seed"],
                                    stop_token_ids=self.eos_token_ids, logprobs=TOP_LOGPROBS,
                                    detokenize=False)  # text is decoded from the action IDs in finish()
            return engine._prompt(prepared, prepared.token_ids), params, item

        def finish(output: Any, request: dict, item: dict) -> dict:
            if len(output.outputs) != 1 or list(output.prompt_token_ids) != list(item["prompt_token_ids"]):
                raise RuntimeError("SINGLE vLLM: one completion on the exact HF prompt IDs required")
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            if completion.logprobs is None or len(completion.logprobs) != len(ids):
                raise RuntimeError("SINGLE vLLM: one processed log-probability row per action required")
            reason = completion.finish_reason
            if reason not in ("stop", "length") or (reason == "stop" and (not ids or ids[-1] not in eos)):
                raise RuntimeError(f"SINGLE vLLM: unsupported finish {reason!r} or stop without an EOS action")
            logprobs, supports = [], []
            for token, row in zip(ids, completion.logprobs, strict=True):
                value, support = support_from_logprobs(token, row)
                logprobs.append(value)
                supports.append(support)
            text = engine.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generation = single_request_binding(request) | {
                "prompt_token_ids": list(item["prompt_token_ids"]), "completion_token_ids": ids,
                "sampling_logprobs": logprobs, "support_sha256": [digest(s) for s in supports],
                "support_token_ids": supports, "sampling": dict(SINGLE_SAMPLING), "policy_sha256": policy,
                "protocol_sha256": protocol, "sampling_sha256": sampling_hash, "input_sha256": item["input_sha256"],
                "finish_reason": reason, "request_sha256": single_request_sha256(request), "text": text,
                "eos_token_ids": self.eos_token_ids, "completion_count": 1, "engine": label}
            return {"text": text, "finish_reason": reason, "prompt_tokens": len(item["prompt_token_ids"]),
                    "completion_tokens": len(ids), "completion_count": 1, "generation": generation}

        def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
            if self.sync(bundle) != policy:
                raise ValueError("SINGLE vLLM: policy changed before generation")
            core = engine.llm.llm_engine
            if self.lora:
                if self.lora_request is None:
                    raise RuntimeError("SINGLE vLLM: LoRA mode without a loaded policy adapter")
                core = _LoraEngine(core, self.lora_request)
            with _nondeterministic():
                yield from continuous_stream(core, requests, self.max_num_seqs, prepare, finish,
                                             stats_, stop_after=stop_after)

        setattr(stream, "continuous", True)
        return stream
