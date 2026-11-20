"""Final-model engine validation, isolated from the historical v17 eager probe.

Only an explicit ``validate`` command runs a model. CPU receipt helpers never load one.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import math
import os
import multiprocessing
from pathlib import Path
import re
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mmdl.runtime.artifacts import digest, read_json, sha256_file
from mmdl.runtime.contracts import sample_seed

from .v17_eval import PROBE_CAP_SECONDS, _metrics, checked, exclusive, seal, source_sha256

VERSION = "v17-eval-engine-validation-v1"
CONTEXT = 40960
OUTPUT = 32768
HISTORICAL_TOKENS = 4_116_922
DECODE_TOKENS = 32
MODES = ("cuda_graph", "eager")
V18_SLOTS = 9


def capacity_observation(engine: Any, version: str) -> dict:
    """Allocated scheduler blocks, including graph memory costs; no VRAM estimate."""
    if version != "0.11.0":
        raise ValueError("v17 engine: unsupported vLLM version")
    cache = engine.engine_core.engine_core.scheduler.kv_cache_config
    sizes = {group.kv_cache_spec.block_size for group in cache.kv_cache_groups}
    if len(sizes) != 1:
        raise ValueError("v17 engine: mixed KV block sizes")
    block_size = sizes.pop()
    blocks = cache.num_blocks
    if any(type(n) is not int or n < 1 for n in (blocks, block_size)) or CONTEXT % block_size:
        raise ValueError("v17 engine: invalid or unaligned allocated KV blocks")
    tokens = blocks * block_size
    slots = math.floor(.90 * tokens / CONTEXT)
    if slots < 1:
        raise ValueError("v17 engine: no safe 40,960-token request fits")
    return {"actual_blocks": blocks, "block_size": block_size,
            "actual_kv_token_capacity": tokens, "usable_tokens": math.floor(.90 * tokens),
            "reserved_per_request": CONTEXT, "safe_slots": slots}


def observe_engine(backend: Any, mode: str, slots: int) -> dict:
    """Read live engine config; VLLMBackend.describe() has a historical eager literal."""
    if mode not in MODES or type(slots) is not int or slots < 1:
        raise ValueError("v17 engine: invalid mode or slots")
    engine = backend.llm.llm_engine
    cfg = engine.vllm_config
    model, scheduler, compilation = cfg.model_config, cfg.scheduler_config, cfg.compilation_config
    graph_mode = getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode))
    sampler = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.sampler
    if sampler.logprobs_mode != "raw_logprobs":
        raise ValueError("v17 engine: actual running sampler must return raw full-vocabulary logprobs")
    value = {"version": "v17-engine-observed-v1", "engine_mode": mode,
             "vllm_version": backend.vllm_version, "enforce_eager": model.enforce_eager,
             "dtype": str(model.dtype), "quantization": model.quantization,
             "logprobs_mode": model.logprobs_mode, "sampler_logprobs_mode": sampler.logprobs_mode,
             "max_model_len": model.max_model_len,
             "slots": scheduler.max_num_seqs,
             "compilation_config": {"level": compilation.level, "cudagraph_mode": graph_mode,
                                    "cudagraph_capture_sizes": compilation.cudagraph_capture_sizes},
             **capacity_observation(engine, backend.vllm_version)}
    if (type(model.enforce_eager) is not bool or model.enforce_eager != (mode == "eager")
            or str(model.dtype) not in ("bfloat16", "torch.bfloat16") or model.quantization is not None
            or model.logprobs_mode != "raw_logprobs" or model.max_model_len != CONTEXT
            or scheduler.max_num_seqs != slots or slots > value["safe_slots"]
            or (mode == "cuda_graph" and graph_mode in ("NONE", "None", "0"))):
        raise ValueError("v17 engine: effective engine config differs from requested execution")
    return seal(value)


@contextmanager
def engine_execution(mode: str, slots: int, *, gpu_memory_utilization: float | None = None) -> Iterator[dict]:
    """Wrap only LLM construction, and restore it even across registered_variants nesting."""
    from mmdl.evaluation.backends import vllm_backend
    from fine_tuning.notebook_rl import proof_eval as pe

    if mode not in MODES or type(slots) is not int or slots < 1:
        raise ValueError("v17 engine: invalid engine execution")
    if gpu_memory_utilization is not None and (type(gpu_memory_utilization) is not float
            or not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1):
        raise ValueError("v17 engine: invalid explicit GPU memory fraction")
    previous = {key: os.environ.get(key) for key in (pe.SLOTS_ENV, pe.VLLM_INPROCESS_ENV)}
    original = vllm_backend._vllm
    state: dict[str, Any] = {"engine_mode": mode, "slots": slots, "constructor_calls": []}

    def factory():
        llm_cls, sampling_cls, version = original()

        def construct(**kwargs):
            kwargs = kwargs | {"enforce_eager": mode == "eager", "max_num_seqs": slots}
            if gpu_memory_utilization is not None:
                kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            state["constructor_calls"].append({"enforce_eager": kwargs["enforce_eager"],
                                                "max_num_seqs": kwargs["max_num_seqs"]})
            return llm_cls(**kwargs)

        return construct, sampling_cls, version

    try:
        os.environ[pe.SLOTS_ENV], os.environ[pe.VLLM_INPROCESS_ENV] = str(slots), "0"
        vllm_backend._vllm = factory
        yield state
    finally:
        vllm_backend._vllm = original
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def baseline_lengths(path: Path, expected_sha256: str) -> list[int]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("v17 engine: B2 length source changed")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        checked(value)
        if (value.get("version") != "v17-b2-length-distribution-v1"
                or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("source_sha256", "")))):
            raise ValueError("v17 engine: unsupported B2 length-only sidecar")
        lengths = value.get("generated_tokens", [])
    elif isinstance(value, list):
        lengths = [row["generated_tokens"] for row in value]
    else:
        raise ValueError("v17 engine: B2 raw records or sealed length-only sidecar required")
    if (len(lengths) != 900 or any(type(n) is not int or not 0 <= n <= OUTPUT for n in lengths)
            or sum(lengths) != HISTORICAL_TOKENS):
        raise ValueError("v17 engine: B2 length count, range or total differs")
    return lengths


def quantile_lengths(lengths: list[int], sample_ids: list[str]) -> dict:
    if len(sample_ids) != 16 or len(set(sample_ids)) != 16:
        raise ValueError("v17 engine: exactly sixteen external performance IDs required")
    ordered = sorted(lengths)
    rows = []
    for i, sid in enumerate(sorted(sample_ids)):
        rank = math.ceil((i + .5) * len(ordered) / 16) - 1
        rows.append({"sample_id": sid, "quantile": (i + .5) / 16, "source_rank": rank,
                     "max_tokens": max(1, ordered[rank])})
    return seal({"version": "v17-fixed-b2-quantiles-v1", "method": "nearest rank at (i+0.5)/16",
                 "source_count": len(lengths), "source_sum_tokens": sum(lengths), "items": rows})


def calibration_receipt(actual: dict, long: dict, lengths: list[int], *, load_seconds: float,
                        long_tail_seconds: float, long_tail_basis: str) -> dict:
    """Both estimates use the actual-length probe; the long probe only establishes safety."""
    allowances = (load_seconds, long_tail_seconds)
    if (any(not isinstance(n, (float, int)) or isinstance(n, bool) or not math.isfinite(n) or n < 0
            for n in allowances) or not long_tail_basis):
        raise ValueError("v17 engine: explicit finite allowances and tail basis required")
    wall, tokens = actual.get("wall_seconds", 0), actual.get("generated_tokens", 0)
    windows = actual.get("sixty_second_rates", [])
    if (not isinstance(wall, (float, int)) or not math.isfinite(wall) or wall <= 0
            or type(tokens) is not int or tokens < 1
            or any(not isinstance(q, (float, int)) or not math.isfinite(q) or q <= 0 for q in windows)):
        raise ValueError("v17 engine: missing actual-length throughput")
    q_run = tokens / wall
    q_safe = min(q_run, statistics.median(windows)) if windows else q_run
    adequate = (actual.get("status") == "COMPLETED" and actual.get("completed_count") == 16
                and long.get("max_generated_tokens", 0) >= OUTPUT
                and long.get("max_observed_context_tokens", 0) >= OUTPUT
                and long.get("forced_ignore_eos") is True
                and actual.get("forced_ignore_eos") is True
                and actual.get("requested_count") == 16 and len(actual.get("fixed_lengths", {})) == 16
                and tokens == sum(actual.get("fixed_lengths", {}).values())
                and actual.get("cumulative_preemptions") == long.get("cumulative_preemptions") == 0)
    allowance = sum(allowances)
    return seal({"version": "v17-eval-calibration-v2",
                 "status": "CALIBRATED" if adequate else "CALIBRATION_LOW_COVERAGE",
                 "q_run": q_run, "q_safe": q_safe, "throughput_source": "fixed_b2_quantile_lengths",
                 "actual_length_probe_sha256": actual["sha256"], "long_context_probe_sha256": long["sha256"],
                 "estimated_evaluation_seconds": math.ceil(sum(lengths) / q_run + allowance),
                 "evaluation_watchdog_seconds": math.ceil(1.25 * (1.5 * sum(lengths) / q_safe + allowance)),
                 "scenario": {"historical_tokens": sum(lengths), "historical_count": len(lengths),
                              "output_factor": 1.5, "wall_factor": 1.25,
                              "load_finalize_allowance_seconds": load_seconds,
                              "long_tail_allowance_seconds": long_tail_seconds,
                              "long_tail_basis": long_tail_basis},
                 "estimate_method": "historical output sum / measured actual-length aggregate token rate + allowances",
                 "statistical_upper_bound": False})


@contextmanager
def diagnostic_timing() -> Iterator[dict]:
    """CPU dispatch wall only, without CUDA synchronization or changing sampler algorithms."""
    import torch

    sampler: Any = importlib.import_module("vllm.v1.sample.sampler")
    stats: dict[str, Any] = {"sampling": {"calls": 0, "cpu_wall_seconds": 0.},
                             "penalties": {"calls": 0, "cpu_wall_seconds": 0.}, "graph_replay_calls": 0,
                             "scope": "host wall, asynchronous CUDA; sampling includes penalties; do not sum"}
    forward, penalties, replay = sampler.Sampler.forward, sampler.apply_all_penalties, torch.cuda.CUDAGraph.replay

    def timed(name, function):
        def call(*args, **kwargs):
            before = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stats[name]["calls"] += 1
                stats[name]["cpu_wall_seconds"] += time.perf_counter() - before
        return call

    def replayed(*args, **kwargs):
        result = replay(*args, **kwargs)
        stats["graph_replay_calls"] += 1
        return result

    try:
        sampler.Sampler.forward = timed("sampling", forward)
        sampler.apply_all_penalties = timed("penalties", penalties)
        setattr(torch.cuda.CUDAGraph, "replay", replayed)
        yield stats
    finally:
        sampler.Sampler.forward, sampler.apply_all_penalties = forward, penalties
        setattr(torch.cuda.CUDAGraph, "replay", replay)


def _raw_decoded(output: Any, token: int, count: int) -> list[float]:
    completion = output.outputs[0]
    if list(completion.token_ids) != [token] * count or completion.logprobs is None:
        raise ValueError("v17 engine: fixed native decode continuation was not preserved")
    values = [float(row[token].logprob) for row in completion.logprobs]
    if len(values) != count or any(not math.isfinite(v) or v >= -1e-8 for v in values):
        raise ValueError("v17 engine: finite raw full-vocabulary decode probabilities required (masked zero rejected)")
    return values


def decode_pf5(backend: Any, bundle: Any, prep: Any, pairs: list, reference: dict,
               timing: dict, mode: str, root: Path, vision_exception: str | None = None,
               vision_tower: dict | None = None) -> dict:
    """Supplement the historical prefill PF5 with real amplified autoregressive decode."""
    from vllm import SamplingParams
    from fine_tuning.notebook_rl.engines import VLLMRolloutEngine
    from . import llm_lora, lora_preflight

    prep.llm = backend.llm
    replay_deltas = []

    def probe(request, completion, lora):
        token = completion[0]
        if completion != [token] * DECODE_TOKENS:
            raise ValueError("v17 engine: decode probe requires one repeated native token")
        prepared = prep.prepare(request)
        params = SamplingParams(temperature=0., max_tokens=DECODE_TOKENS, min_tokens=DECODE_TOKENS,
                                ignore_eos=True, allowed_token_ids=[token], logprobs=0,
                                seed=request.seeds[0], detokenize=False)
        before = timing["graph_replay_calls"]
        output = backend.llm.generate([prep._prompt(prepared, prepared.token_ids)], params,
                                      lora_request=lora, use_tqdm=False)[0]
        delta = timing["graph_replay_calls"] - before
        replay_deltas.append(delta)
        if mode == "cuda_graph" and delta < DECODE_TOKENS - 1:
            raise ValueError("v17 engine: PF5 decode did not actually replay a CUDA graph")
        return _raw_decoded(output, token, DECODE_TOKENS)

    swap = lora_preflight.adapter_swap(backend.llm.llm_engine, bundle, backend.lora_request, root)
    columns, restored = llm_lora.vllm_columns(bundle, backend.vision_hooks, probe, pairs, backend.lora_request, swap)
    report = llm_lora.slope_check(**columns, **reference, cap=VLLMRolloutEngine.sync_threshold_cap,
                                  require_power=True, vision_exception=vision_exception, vision_tower=vision_tower)
    report["restored"] = llm_lora.check_identity(backend.llm, bundle, restored.lora_int_id, backend.vision_hooks)
    report |= {"graph_replay_deltas": replay_deltas, "generated_decode_tokens_per_probe": DECODE_TOKENS,
               "raw_logprobs": True, "native_allowed_token_ids": True, "diagnostic_only": True}
    report["pass"] = report["pass"] and report["restored"]["pass"]
    return report


def _requests(manifest: dict, root: Path) -> Iterator[dict]:
    from PIL import Image

    for item in manifest["items"]:
        images = []
        for entry in item["images"]:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = root / path
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("v17 engine: performance image changed")
            with Image.open(path) as loaded:
                images.append(loaded.copy())
        yield {"sample_id": item["sample_id"], "messages": item["messages"], "images": images,
               "seed": sample_seed(3407, item["sample_id"])}


def probability_columns(backend: Any, manifest: dict, root: Path, timing: dict, mode: str) -> list[dict]:
    """All sixteen external prefixes, same native continuation and raw decode probabilities."""
    token = backend.processor.tokenizer.encode(" The", add_special_tokens=False)[0]
    result = []
    for request in _requests(manifest, root):
        item, prompt, params = backend._prepare_request(request, final_only=False)
        params = copy.copy(params)
        params.max_tokens = params.min_tokens = DECODE_TOKENS
        params.ignore_eos, params.allowed_token_ids, params.logprobs = True, [token], 0
        before = timing["graph_replay_calls"]
        output = backend.llm.generate([prompt], params, lora_request=backend.lora_request, use_tqdm=False)[0]
        delta = timing["graph_replay_calls"] - before
        if mode == "cuda_graph" and delta < DECODE_TOKENS - 1:
            raise ValueError("v17 engine: external decode did not replay a CUDA graph")
        result.append({"sample_id": request["sample_id"], "seed": request["seed"],
                       "input_sha256": item["input_sha256"], "token_ids": [token] * DECODE_TOKENS,
                       "logprobs": _raw_decoded(output, token, DECODE_TOKENS), "graph_replay_calls": delta})
    return result


def compare_probabilities(candidate: list[dict], eager: list[dict]) -> dict:
    from fine_tuning.notebook_rl.scale import VLLM_SYNC_THRESHOLD

    left, right = ({row["sample_id"]: row for row in rows} for rows in (candidate, eager))
    if len(candidate) != 16 or len(eager) != 16 or len(left) != 16 or len(right) != 16 or left.keys() != right.keys():
        raise ValueError("v17 engine: all sixteen eager comparison rows required")
    differences = []
    for sid in sorted(left):
        a, b = left[sid], right[sid]
        if any(a[key] != b[key] for key in ("seed", "input_sha256", "token_ids")):
            raise ValueError("v17 engine: eager comparison input or continuation changed")
        if any(not math.isfinite(x) or x >= -1e-8 for x in a["logprobs"] + b["logprobs"]):
            raise ValueError("v17 engine: raw full-vocabulary probabilities required")
        delta = [abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"], strict=True)]
        if len(delta) != DECODE_TOKENS or any(not math.isfinite(x) for x in delta):
            raise ValueError("v17 engine: nonfinite or incomplete eager comparison")
        differences.append({"sample_id": sid, "mean_abs_logprob_difference": statistics.mean(delta),
                            "max_abs_logprob_difference": max(delta)})
    return seal({"items": differences, "threshold": VLLM_SYNC_THRESHOLD,
                 "criterion": "each external prefix mean absolute raw logprob difference <= existing sync threshold",
                 "pass": all(row["mean_abs_logprob_difference"] <= VLLM_SYNC_THRESHOLD for row in differences)})


def probe_generation(backend: Any, manifest: dict, root: Path, output: Path, *, lengths: dict[str, int],
                     max_seconds: float, timing: dict) -> dict:
    """Real generated histories, native B2 sampling/penalties, diagnostic-only forced lengths."""
    engine = backend.llm.llm_engine
    original_step, original_prepare = engine.step, backend._prepare_request
    previous_deadline = getattr(backend, "_check_deadline", None)
    initial = _metrics(backend.llm)
    observations = [(0., 0)]
    max_context = max_generated = complete = 0
    completed_tokens = 0
    started = time.monotonic()
    timing_before = copy.deepcopy(timing)

    def prepare(request, **kwargs):
        item, prompt, params = original_prepare(request, **kwargs)
        params = copy.copy(params)
        params.max_tokens = params.min_tokens = lengths[request["sample_id"]]
        params.ignore_eos = True
        return item, prompt, params

    def sample_state():
        nonlocal max_context, max_generated
        scheduler = engine.engine_core.engine_core.scheduler
        for request in scheduler.requests.values():
            max_context = max(max_context, request.num_tokens)
            max_generated = max(max_generated, len(request.output_token_ids))
        current = _metrics(backend.llm)
        elapsed = time.monotonic() - started
        count = current["tokens"] - initial.get("tokens", 0) if "tokens" in current else completed_tokens
        if elapsed - observations[-1][0] >= 10:
            observations.append((elapsed, count))
        return current, elapsed, count

    def step():
        value = original_step()
        sample_state()
        return value

    def deadline():
        if time.monotonic() - started >= max_seconds:
            sample_state()
            for sid, request in engine.engine_core.engine_core.scheduler.requests.items():
                tokens = list(request.output_token_ids)
                exclusive(output.parent / (output.name + "-partial") / (sid + ".json"), {
                    "sample_id": sid, "generated_token_ids": tokens,
                    "raw_response": backend.processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                                                       clean_up_tokenization_spaces=False),
                    "status": "DIAGNOSTIC_PARTIAL_NOT_SCORED"})
            raise TimeoutError("v17 engine: bounded diagnostic generation cap")

    status = "COMPLETED"
    try:
        engine.step, backend._prepare_request, backend._check_deadline = step, prepare, deadline
        selected = (r for r in _requests(manifest, root) if r["sample_id"] in lengths)
        for row in backend.generate_stream(selected):
            count = row["generated_tokens"]
            if count != lengths[row["sample_id"]]:
                raise ValueError("v17 engine: diagnostic output did not reach its fixed length")
            complete += 1
            completed_tokens += count
            max_context = max(max_context, row["input_tokens"] + count)
            max_generated = max(max_generated, count)
            exclusive(output / (row["sample_id"] + ".json"), row)
    except TimeoutError:
        status = "DIAGNOSTIC_CAP_PARTIAL"
    finally:
        engine.step, backend._prepare_request = original_step, original_prepare
        if previous_deadline is None:
            del backend._check_deadline
        else:
            backend._check_deadline = previous_deadline
    current, elapsed, tokens = sample_state()
    observations.append((elapsed, tokens))
    windows = []
    for begin, count in observations:
        later = next(((end, total) for end, total in observations if end - begin >= 60), None)
        if later:
            windows.append((later[1] - count) / (later[0] - begin))
    measured_timing = {key: {field: timing[key][field] - timing_before[key][field]
                             for field in ("calls", "cpu_wall_seconds")} for key in ("sampling", "penalties")}
    return seal({"version": "v17-engine-generation-probe-v1", "status": status,
                 "wall_seconds": elapsed, "generated_tokens": tokens, "completed_count": complete,
                 "requested_count": len(lengths), "fixed_lengths": lengths, "sixty_second_rates": windows,
                 "max_observed_context_tokens": max_context, "max_generated_tokens": max_generated,
                 "cumulative_preemptions": current.get("preemptions"), "observations": observations,
                 "forced_ignore_eos": True, "diagnostic_only": True,
                 "used_for_training_or_answer_selection": False,
                 "sampler_cpu_timing": measured_timing, "timing_scope": timing["scope"],
                 "graph_replay_calls": timing["graph_replay_calls"] - timing_before["graph_replay_calls"]})


def _manifest(path: Path, expected: str) -> dict:
    from mmdl.evaluation.writer import _safe_id

    if sha256_file(path) != expected:
        raise ValueError("v17 engine: performance requests changed")
    value = checked(read_json(path))
    items, disjoint = value.get("items", []), value.get("disjoint_receipt", {})
    hashes = ("main_ids_sha256", "prep_ids_sha256", "development_ids_sha256", "effect_ids_sha256",
              "benchmark_exclusion_sha256")
    if (value.get("version") != "v17-performance-only-v1" or len(items) != 16
            or len({item["sample_id"] for item in items}) != 16
            or disjoint.get("status") != "VERIFIED" or disjoint.get("overlap_count") != 0
            or value.get("diagnostic_only") is not True or value.get("used_for_training_or_answer_selection") is not False
            or any(not re.fullmatch(r"[a-f0-9]{64}", str(disjoint.get(key, ""))) for key in hashes)):
        raise ValueError("v17 engine: fixed disjoint sixteen external performance requests required")
    for item in items:
        _safe_id(item["sample_id"])
        if ("mmmu" in item["sample_id"].lower() or item["sample_id"].startswith("validation_")
                or len(item["messages"]) != 1 or item["messages"][0].get("role") != "user"
                or item.get("seed") != sample_seed(3407, item["sample_id"])):
            raise ValueError("v17 engine: invalid external P0 performance request")
    return value


DISCOVERY_SECONDS = 1200  # one fresh process: model load, LoRA, compile/graph capture, KV profiling
DISCOVERY_ROUNDS = 3


def stable_slots(measure: Any, *, maximum: int | None = None) -> tuple[int, list[dict]]:
    """Largest slot count whose own fresh engine can hold it: start from the one-slot engine's safe count and
    lower it while an engine built at that count (larger graph pool, less KV) reports fewer safe slots."""
    history: list[dict] = []
    capacity = measure(1)
    history.append({"constructed_slots": 1, **capacity})
    candidate = min(capacity["safe_slots"], maximum) if maximum is not None else capacity["safe_slots"]
    for _ in range(DISCOVERY_ROUNDS):
        if candidate <= 1:
            return 1, history
        capacity = measure(candidate)
        history.append({"constructed_slots": candidate, **capacity})
        if capacity["safe_slots"] >= candidate:
            return candidate, history
        candidate = capacity["safe_slots"]
    raise ValueError("v17 engine: safe slot capacity did not stabilize after graph allocation")


def _discovery_worker(args: Any, mode: str, slots: int, path: Path, context: dict) -> None:
    from mmdl.evaluation import engine as evaluation
    from mmdl.evaluation.backends.vllm_backend import VLLMBackend
    from fine_tuning.notebook_rl import proof_eval as pe

    model_path = args.model_ref.parent / "model"
    with engine_execution(mode, slots), pe.registered_variants(slots=slots, log_stats=True):
        evaluation.verify_model(model_path, context["model_ref"])
        backend = VLLMBackend(str(model_path), context["config"], context["hardware"], pe.LORA_KIND,
                              str(args.base_path))
    capacity = capacity_observation(backend.llm.llm_engine, backend.vllm_version)
    backend.llm.llm_engine.engine_core.shutdown()
    exclusive(path, seal({"version": "v17-capacity-discovery-v1", "engine_mode": mode, "constructed_slots": slots,
                          **capacity}))


def discover_slots(args: Any, mode: str, root: Path, context: dict) -> tuple[int, list[dict]]:
    """Each capacity measurement is a fresh spawned process, like the later 900-question evaluation process."""
    root.mkdir(parents=True, exist_ok=False)

    def measure(slots: int) -> dict:
        path = root / f"capacity-{len(list(root.glob('capacity-*.json'))) + 1}-{slots}.json"
        process = multiprocessing.get_context("spawn").Process(
            target=_discovery_worker, args=(args, mode, slots, path, context))
        process.start()
        process.join(timeout=DISCOVERY_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join()
            raise ValueError("v17 engine: capacity discovery exceeded its technical deadline")
        if process.exitcode != 0 or not path.is_file():
            raise ValueError(f"v17 engine: capacity discovery process exit {process.exitcode}")
        value = checked(read_json(path))
        return {k: value[k] for k in ("actual_blocks", "block_size", "actual_kv_token_capacity", "usable_tokens",
                                      "reserved_per_request", "safe_slots")}

    return stable_slots(measure, maximum=V18_SLOTS if context.get("training_protocol_version") == "error-note-v18" else None)


def _attempt(args: Any, mode: str, root: Path, context: dict, *, reference_only: bool = False,
             slots: int = 1, discovery: list | None = None) -> dict:
    """One concrete final-model engine; every observation and PF5 belongs to this instance."""
    import torch
    from mmdl.evaluation import engine as evaluation
    from mmdl.evaluation.backends.vllm_backend import VLLMBackend
    from fine_tuning.notebook_rl import proof_eval as pe
    from fine_tuning.notebook_rl.checkpoint import load_eval
    from . import llm_lora, lora_preflight

    root.mkdir(parents=True, exist_ok=False)
    backend = bundle = prep = None
    report: dict[str, Any] = {"version": "v17-engine-attempt-v1", "engine_mode": mode,
                               "reference_only": reference_only, "status": "FAIL", "reasons": []}
    started = time.monotonic()
    try:
        cfg, hw = context["config"], context["hardware"]
        bundle = load_eval(args.base_path, Path(context["seal"]["checkpoint"]["root"]),
                           binding=pe._binding_file(Path(context["seal"]["campaign_plan"]["path"]))[0])
        model_path = args.model_ref.parent / "model"
        files = lora_preflight.check_files(model_path, args.base_path, bundle)
        prep = lora_preflight.probe_preparation(args.base_path, model_path, cfg["image"], CONTEXT)
        pairs = lora_preflight.probe_pairs(prep, cfg["image"])
        token = prep.tokenizer.encode(" The", add_special_tokens=False)[0]
        decode_pairs = [(request, [token] * DECODE_TOKENS) for request, _ in pairs]
        bundle.model.to("cuda")
        reference = llm_lora.hf_columns(bundle, prep, pairs)
        decode_reference = llm_lora.hf_columns(bundle, prep, decode_pairs)
        version = context.get("training_protocol_version", "error-note-v17")
        tower_reference = llm_lora.hf_vision_columns(bundle, prep, pairs) if version in ("error-note-v18", "error-note-v18b") else None
        bundle.model.to("cpu")
        torch.cuda.empty_cache()
        # Slots come from fresh-process discovery (discover_slots), so this process builds exactly one engine:
        # an in-process rebuild may not release weights/KV/graph pools (vLLM 0.11 needs 0.9 of total free).
        report["capacity_discovery"] = discovery or []
        with engine_execution(mode, slots), pe.registered_variants(slots=slots, log_stats=True):
            evaluation.verify_model(model_path, context["model_ref"])
            backend = VLLMBackend(str(model_path), cfg, hw, pe.LORA_KIND, str(args.base_path))
        observed = observe_engine(backend, mode, slots)
        if version in ("error-note-v18", "error-note-v18b") and (slots != V18_SLOTS or observed["safe_slots"] < V18_SLOTS):
            raise ValueError("v18 engine: nine safe slots required; changing slot count requires a new admission")
        tower = (llm_lora.vision_tower_check(tower_reference, llm_lora.vllm_vision_columns(backend, bundle, prep, pairs))
                 if tower_reference is not None else None)
        report["observed_engine"] = observed
        with diagnostic_timing() as timing:
            pf5 = {"format": "lora-preflight-v1", "model_ref_sha256": sha256_file(args.model_ref),
                   "protocol_id": cfg["protocol_id"], "files": files,
                   "engine": lora_preflight.engine_checks(backend, bundle, prep, pairs, reference,
                                                          protocol_version=("error-note-v18" if version == "error-note-v18b" else version), stage="trained", vision_tower=tower,
                                                          vision_exception=getattr(args, "pf5_vision_exception", None))}
            pf5["decode_engine"] = decode_pf5(backend, bundle, prep, decode_pairs, decode_reference,
                                                timing, mode, root / "amplified-adapters",
                                                vision_exception=getattr(args, "pf5_vision_exception", None), vision_tower=tower)
            pf5["pass"] = files["pass"] and pf5["engine"]["pass"] and pf5["decode_engine"]["pass"]
            exclusive(root / "pf5.json", pf5)
            report["preflight"] = {"path": str((root / "pf5.json").resolve()),
                                     "sha256": sha256_file(root / "pf5.json")}
            report["preflight_evidence"] = seal({"files_pass": files["pass"],
                "historical_engine_pass": pf5["engine"]["pass"], "historical_pf5": pf5["engine"]["pf5"],
                "decode_pf5": pf5["decode_engine"]})
            if pf5["pass"] is not True:
                raise ValueError("v17 engine: final-model PF5 or amplified graph decode failed")
            report["probabilities"] = probability_columns(backend, context["requests"], args.requests.parent,
                                                            timing, mode)
            if not reference_only and version == "error-note-v18b":
                from .v18b_eval import live_calibration
                report["calibration"] = live_calibration(observed, context["execution_settings"])
            elif not reference_only:
                load_seconds = time.monotonic() - started
                long_ids = [item["sample_id"] for item in context["requests"]["items"][:min(slots, 16)]]
                long = probe_generation(backend, context["requests"], args.requests.parent, root / "long-outputs",
                                        lengths=dict.fromkeys(long_ids, OUTPUT), max_seconds=args.max_seconds, timing=timing)
                quantiles = context["quantiles"]
                actual = probe_generation(backend, context["requests"], args.requests.parent, root / "actual-outputs",
                    lengths={row["sample_id"]: row["max_tokens"] for row in quantiles["items"]},
                    max_seconds=args.max_seconds, timing=timing)
                report["long_context_probe"], report["actual_length_probe"] = long, actual
                report["quantiles"] = quantiles
                report["calibration"] = calibration_receipt(actual, long, context["lengths"],
                    load_seconds=load_seconds, long_tail_seconds=args.long_tail_allowance_seconds,
                    long_tail_basis=args.long_tail_basis)
                if report["calibration"]["status"] != "CALIBRATED":
                    raise ValueError("v17 engine: incomplete long-context or actual-length calibration")
            report["status"] = "MEASURED"
    except Exception as exc:
        report["reasons"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        if backend is not None:
            try:
                backend.llm.llm_engine.engine_core.shutdown()
            except Exception as exc:
                report["status"] = "FAIL"
                report["reasons"].append({"type": type(exc).__name__, "message": "engine shutdown: " + str(exc)})
        del backend, bundle, prep
        gc.collect()
        torch.cuda.empty_cache()
    return report


def _attempt_worker(args: Any, mode: str, root: Path, context: dict, reference_only: bool,
                    slots: int = 1, discovery: list | None = None) -> None:
    report = _attempt(args, mode, root, context, reference_only=reference_only, slots=slots, discovery=discovery)
    exclusive(root / "measurement.json", seal(report))


def validation_timeout_seconds(max_seconds: float, mode: str = "auto", *, training_protocol_version: str = "error-note-v17") -> int:
    if (mode not in ("auto", *MODES) or isinstance(max_seconds, bool)
            or not isinstance(max_seconds, (float, int)) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= PROBE_CAP_SECONDS):
        raise ValueError("v17 engine: invalid bounded validation timeout inputs")
    if training_protocol_version == "error-note-v18b":
        return math.ceil((2 if mode == "eager" else 4) * max_seconds + 60)
    full_attempts = 1 if mode == "eager" else 2
    reference_attempts = 1 if mode == "eager" else 2
    discovery = (1 + DISCOVERY_ROUNDS) * (DISCOVERY_SECONDS + 60)
    return math.ceil(full_attempts * (2 * max_seconds + 1830 + discovery) + reference_attempts * 1830 + 60)


def _isolated_attempt(args: Any, mode: str, root: Path, context: dict, *, reference_only: bool = False) -> dict:
    """Fresh process per mode releases CUDA graphs, workers and HF models before the next engine."""
    v18b = context.get("training_protocol_version") == "error-note-v18b"
    slots = V18_SLOTS if v18b or (reference_only and context.get("training_protocol_version") == "error-note-v18") else 1
    discovery: list[dict] = []
    if not reference_only and not v18b:
        try:
            slots, discovery = discover_slots(args, mode, root.parent / (root.name + "-capacity"), context)
        except Exception as exc:
            return {"version": "v17-engine-attempt-v1", "engine_mode": mode, "reference_only": reference_only,
                    "status": "FAIL", "reasons": [{"type": type(exc).__name__, "message": str(exc)}]}
    timeout = (0 if reference_only else 2 * args.max_seconds) + 1800
    if v18b:
        from .v18b_execution import deadline_epoch, enforce_limits
        settings = context["execution_settings"]
        timeout = args.max_seconds
        if enforce_limits(settings):
            timeout = min(timeout, deadline_epoch(settings, reserve_seconds=
                settings["post_eval_reserve_seconds"]) - time.time())
        if timeout <= 0:
            return {"version": "v17-engine-attempt-v1", "engine_mode": mode, "reference_only": reference_only,
                    "status": "FAIL", "reasons": [{"type": "DeadlineReached", "message": "first-billing deadline reached"}]}
    process = multiprocessing.get_context("spawn").Process(
        target=_attempt_worker, args=(args, mode, root, context, reference_only, slots, discovery))
    process.start()
    process.join(timeout=max(0, timeout))
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join()
        reason = "isolated engine validation exceeded its diagnostic technical deadline"
    elif process.exitcode != 0:
        reason = f"isolated engine validation process exit {process.exitcode}"
    elif (root / "measurement.json").is_file():
        return checked(read_json(root / "measurement.json"))
    else:
        reason = "isolated engine produced no measurement receipt"
    return {"version": "v17-engine-attempt-v1", "engine_mode": mode, "reference_only": reference_only,
            "status": "FAIL", "reasons": [{"type": "EngineProcessFailed", "message": reason}]}


def select_attempts(run: Any, mode: str, root: Path) -> tuple[dict | None, list[dict]]:
    """Automatic graph-first selection, before any benchmark run or preregistration."""
    attempts: list[dict] = []

    def measured(which, role, reference_only=False):
        path = root / (f"attempt-{len(attempts) + 1}-" + which + "-" + role)
        result = run(which, path, reference_only=reference_only)
        result["role"] = role
        return path, result

    def persist(path, result):
        value = seal(result)
        path.mkdir(parents=True, exist_ok=True)
        exclusive(path / "receipt.json", value)
        attempts.append({"path": str((path / "receipt.json").resolve()),
                         "sha256": sha256_file(path / "receipt.json"), "engine_mode": value["engine_mode"],
                         "status": value["status"], "reasons": value.get("reasons", [])})
        return value

    reference = None
    if mode in ("auto", "cuda_graph"):
        graph_path, graph = measured("cuda_graph", "candidate")
        if graph["status"] == "MEASURED":
            eager_path, eager = measured("eager", "reference", True)
            if eager["status"] == "MEASURED":
                reference = eager["probabilities"]
                try:
                    comparison = compare_probabilities(graph["probabilities"], reference)
                except (ValueError, KeyError, TypeError) as exc:
                    comparison = seal({"pass": False, "reason": str(exc)})
                graph["eager_comparison"] = comparison
                graph["eager_reference_probabilities"] = reference
                graph["status"] = "PASS" if comparison["pass"] else "FAIL"
                if not comparison["pass"]:
                    graph["reasons"].append({"type": "ProbabilityMismatch", "message": "eager raw decode threshold exceeded"})
            else:
                graph["status"] = "FAIL"
                graph["reasons"].append({"type": "EagerReferenceFailed", "message": "no valid eager comparison"})
            persist(eager_path, eager)
        graph = persist(graph_path, graph)
        if graph["status"] == "PASS":
            return graph, attempts
    eager_path, eager = measured("eager", "fallback" if attempts else "candidate")
    if eager["status"] == "MEASURED":
        if reference is None:
            ref_path, ref = measured("eager", "reference", True)
            persist(ref_path, ref)
            if ref["status"] == "MEASURED":
                reference = ref["probabilities"]
        if reference is None:
            eager["status"] = "FAIL"
            eager["reasons"].append({"type": "EagerReferenceFailed", "message": "no valid independent eager comparison"})
        else:
            try:
                eager["eager_comparison"] = compare_probabilities(eager["probabilities"], reference)
            except (ValueError, KeyError, TypeError) as exc:
                eager["eager_comparison"] = seal({"pass": False, "reason": str(exc)})
            eager["eager_reference_probabilities"] = reference
            eager["status"] = "PASS" if eager["eager_comparison"]["pass"] else "FAIL"
            if eager["status"] == "FAIL":
                eager["reasons"].append({"type": "ProbabilityMismatch", "message": "eager replay threshold exceeded"})
    eager = persist(eager_path, eager)
    return (eager if eager["status"] == "PASS" else None), attempts


def validate_engine_receipt(value: dict) -> dict:
    checked(value)
    if (value.get("version") != VERSION or value.get("status") != "PASS"
            or value.get("engine_mode") not in MODES or value.get("selected_mode") != value["engine_mode"]
            or type(value.get("slots")) is not int or value["slots"] < 1
            or value.get("safe_slots") != value["slots"]):
        raise ValueError("v17 engine: passing selected final-model validation required")
    for key in ("training_artifact_sha256", "training_artifact_file_sha256", "model_ref_sha256", "source_sha256",
                "config_sha256", "hardware_sha256", "environment_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(value.get(key, ""))):
            raise ValueError("v17 engine: missing hash binding " + key)
    if not re.fullmatch(r"[a-f0-9]{40}", str(value.get("code_commit", ""))):
        raise ValueError("v17 engine: missing evaluation commit binding")
    observed = checked(value["observed_engine"])
    v18 = value.get("training_protocol_version") in ("error-note-v18", "error-note-v18b")
    if value.get("training_protocol_version", "error-note-v17") not in ("error-note-v17", "error-note-v18", "error-note-v18b"):
        raise ValueError("v17 engine: unsupported training protocol binding")
    if v18 and value["slots"] != V18_SLOTS:
        raise ValueError("v18 engine: nine registered slots required")
    calibration = checked(value["calibration"])
    selected = checked(value["selected_attempt"])
    evidence = checked(selected["preflight_evidence"])
    if v18:
        for name in ("historical_pf5", "decode_pf5"):
            pf5 = evidence[name]
            tower = pf5.get("vision_tower", {})
            # Mirror llm_lora.vision_tower_check: an amplified x16 effect below 4x the cross-engine noise is recorded as
            # UNPOWERED_NOT_GATING (pass None) and does not gate; a powered mismatch, or an effect that survives hook
            # removal (hook_removed pass True), still fails. The v18b pod (2026-11-20) stopped here on a final model
            # whose x16 tower effect had correlation 0.886 / slope 0.82 and a zero hook-removed effect.
            amplified, removed = tower.get("amplified", {}), tower.get("hook_removed", {})
            if ("vision_exception" in pf5 or pf5.get("vision_gate") != "direct_tower"
                    or tower.get("version") != "v18-direct-vision-pf5-v1" or tower.get("pass") is not True
                    or "pass" not in amplified or amplified["pass"] is False
                    or "pass" not in removed or removed["pass"] is True
                    or tower.get("application_path_status") != ("CONFIRMED" if amplified["pass"] else "UNPOWERED_NOT_GATING")
                    or tower.get("real_effect_status") not in ("CONFIRMED", "UNCONFIRMED")
                    or not tower.get("inputs") or tower.get("coordinates", 0) < 1):
                raise ValueError("v18 engine: direct vision evidence or hook-removal negative control missing")
        authorization = checked(value.get("evaluation_authorization", {}))
        decision = checked(authorization.get("evaluation_decision", {}))
        identity = {"code_commit": value["code_commit"], "source_sha256": value["source_sha256"]}
        if (authorization.get("status") != "PASS" or authorization.get("evaluation") != identity
                or authorization.get("training_artifact", {}).get("file_sha256") != value["training_artifact_sha256"]
                or decision.get("version") != "v18-evaluation-decision-v1" or decision.get("decision") != "PROCEED"
                or decision.get("authorized_by") != "user" or decision.get("pre_authorized") is not False
                or not isinstance(decision.get("authorization"), str) or not decision["authorization"].strip()
                or decision.get("admission_status") not in ("EVAL_REVIEW_REQUIRED", "EVALUATION_CANDIDATE")
                or decision.get("training_artifact") != authorization.get("training_artifact")
                or decision.get("evaluation") != identity
                or decision.get("model_ref", {}).get("file_sha256") != value["model_ref_sha256"]
                or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("artifact_verification", {}).get("file_sha256", "")))):
            raise ValueError("v18 engine: separate user authorization is missing or bound to another model/evaluator")
    v18b = value.get("training_protocol_version") == "error-note-v18b"
    lengths = value["baseline_generated_tokens"]
    if (len(lengths) != 900 or sum(lengths) != HISTORICAL_TOKENS
            or any(type(n) is not int or not 0 <= n <= OUTPUT for n in lengths)):
        raise ValueError("v17 engine: invalid historical B2 length evidence")
    if v18b:
        from .v18b_eval import live_calibration
        reconstructed = live_calibration(observed, value["execution_settings"])
        length_evidence_ok = not any(key in selected for key in ("actual_length_probe", "long_context_probe", "quantiles"))
    else:
        actual, long = checked(selected["actual_length_probe"]), checked(selected["long_context_probe"])
        scenario = calibration["scenario"]
        reconstructed = calibration_receipt(actual, long, lengths,
            load_seconds=scenario["load_finalize_allowance_seconds"],
            long_tail_seconds=scenario["long_tail_allowance_seconds"], long_tail_basis=scenario["long_tail_basis"])
        quantiles = quantile_lengths(lengths, [row["sample_id"] for row in selected["probabilities"]])
        length_evidence_ok = (checked(selected["quantiles"]) == quantiles and actual.get("fixed_lengths") ==
                              {row["sample_id"]: row["max_tokens"] for row in quantiles["items"]})
    comparison = compare_probabilities(selected["probabilities"], selected["eager_reference_probabilities"])
    if (calibration != reconstructed or not length_evidence_ok
            or checked(selected["eager_comparison"]) != comparison
            or evidence.get("files_pass") is not True or evidence.get("historical_engine_pass") is not True
            or evidence["historical_pf5"].get("pass") is not True or evidence["historical_pf5"].get("power") is not True
            or evidence["decode_pf5"].get("pass") is not True or evidence["decode_pf5"].get("power") is not True
            or evidence["decode_pf5"].get("raw_logprobs") is not True
            or observed.get("sampler_logprobs_mode") != "raw_logprobs"
            or observed.get("enforce_eager") != (value["engine_mode"] == "eager")
            or observed.get("quantization") is not None or observed.get("max_model_len") != CONTEXT
            or observed.get("safe_slots") != math.floor(.90 * observed["actual_kv_token_capacity"] / CONTEXT)
            or (value["engine_mode"] == "cuda_graph" and (
                not evidence["decode_pf5"].get("graph_replay_deltas")
                or any(n < DECODE_TOKENS - 1 for n in evidence["decode_pf5"]["graph_replay_deltas"])
                or any(row.get("graph_replay_calls", 0) < DECODE_TOKENS - 1 for row in selected["probabilities"])))):
        raise ValueError("v17 engine: final PF5, actual decode or calibrated throughput evidence differs")
    if (observed["engine_mode"] != value["engine_mode"] or observed["slots"] != value["slots"]
            or (observed["safe_slots"] < value["slots"] if v18 else observed["safe_slots"] != value["slots"])
            or observed["actual_kv_token_capacity"] != value["actual_kv_token_capacity"]
            or calibration["status"] != ("LIVE_MONITORING_REQUIRED" if v18b else "CALIBRATED")
            or calibration["throughput_source"] != ("evaluation_live_only" if v18b else "fixed_b2_quantile_lengths")
            or selected["status"] != "PASS" or selected["observed_engine"] != observed
            or selected["calibration"] != calibration or checked(selected["eager_comparison"])["pass"] is not True
            or value["evaluation_watchdog_seconds"] != calibration["evaluation_watchdog_seconds"]
            or value["estimated_evaluation_seconds"] != calibration["estimated_evaluation_seconds"]
            or type(value["evaluation_watchdog_seconds"]) is not int or value["evaluation_watchdog_seconds"] < 1):
        raise ValueError("v17 engine: inconsistent selection, capacity, probability or calibration evidence")
    return value


def load_engine_validation(path: Path) -> dict:
    value = validate_engine_receipt(read_json(path))
    if value.get("training_protocol_version") in ("error-note-v18", "error-note-v18b"):
        from .v17_artifacts import _check_ref, verify_evaluation_authorization
        authorization_path = _check_ref(value["artifact_verification"])
        authorized = verify_evaluation_authorization(authorization_path, Path(value["training_artifact_path"]),
            {"code_commit": value["code_commit"], "source_sha256": value["source_sha256"]})
        if authorized != value["evaluation_authorization"]:
            raise ValueError("v18 engine: original evaluation authorization changed")
    for attempt in value["attempts"]:
        if sha256_file(Path(attempt["path"])) != attempt["sha256"]:
            raise ValueError("v17 engine: validation attempt file changed")
    if sha256_file(Path(value["preflight_report"])) != value["preflight_file_sha256"]:
        raise ValueError("v17 engine: selected final-model PF5 file changed")
    pf5 = read_json(Path(value["preflight_report"]))
    evidence = value["selected_attempt"]["preflight_evidence"]
    if (pf5.get("pass") is not True or pf5.get("model_ref_sha256") != value["model_ref_sha256"]
            or evidence != seal({"files_pass": pf5["files"]["pass"],
                "historical_engine_pass": pf5["engine"]["pass"], "historical_pf5": pf5["engine"]["pf5"],
                "decode_pf5": pf5["decode_engine"]})):
        raise ValueError("v17 engine: selected PF5 evidence does not match its preserved report")
    preserved = [checked(read_json(Path(row["path"]))) for row in value["attempts"]]
    if value["selected_attempt"] not in preserved:
        raise ValueError("v17 engine: selected attempt is not present in preserved attempts")
    return value


validate_receipt = load_engine_validation


def calibration_timeout(value: dict) -> int:
    return int(validate_engine_receipt(value)["evaluation_watchdog_seconds"])


def validate(args: Any) -> dict:
    import yaml
    from mmdl.evaluation import engine as evaluation
    from mmdl.runtime.environment import collect_environment
    from fine_tuning.notebook_rl import proof_eval as pe
    from .v17_artifacts import protected_evaluation, verify_training_artifacts

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("v17 engine: validation output already exists")
    if not 0 < args.max_seconds <= PROBE_CAP_SECONDS:
        raise ValueError("v17 engine: diagnostic cap must remain within the existing probe cap")
    seal_value = verify_training_artifacts(args.training_seal)
    if protected_evaluation(pe.ROOT) != seal_value["protected_evaluation"]:
        raise ValueError("v17 engine: protected evaluation semantics differ from training seal")
    sealed_ref = seal_value["export"]["model_ref"]
    if args.model_ref.resolve() != Path(sealed_ref["path"]).resolve() or sha256_file(args.model_ref) != sealed_ref["file_sha256"]:
        raise ValueError("v17 engine: final model differs from training artifact seal")
    plan = read_json(Path(seal_value["campaign_plan"]["path"]))
    version = plan["protocol"].get("version")
    if version not in ("error-note-v17", "error-note-v18", "error-note-v18b"):
        raise ValueError("v17 engine: unsupported sealed training protocol")
    if version in ("error-note-v18", "error-note-v18b") and getattr(args, "pf5_vision_exception", None) is not None:
        raise ValueError("v18 engine: PF5 vision exception is forbidden")
    if Path(plan["model_path"]).resolve() != args.base_path.resolve():
        raise ValueError("v17 engine: base differs from sealed training plan")
    ref, _ = pe._lora_model_ref(args.model_ref)
    if ref.get("kind") != pe.LORA_KIND or ref.get("export", {}).get("training_protocol", {}).get("version") != version:
        raise ValueError("v17 engine: final v17 unmerged model required; F0 is not valid")
    cfg = yaml.safe_load(args.eval_protocol.read_text())
    pinned = yaml.safe_load(pe.VARIANTS[pe.FINAL_VARIANT_V8][0].read_text())
    if cfg != pinned:
        raise ValueError("v17 engine: pinned B2 V8 evaluation protocol required")
    hw = yaml.safe_load(args.hardware.read_text())
    commit = evaluation.git_commit(pe.ROOT, args.code_commit)
    source = source_sha256()
    if args.source_hash is not None and source != args.source_hash:
        raise ValueError("v17 engine: evaluation source changed")
    authorization = None
    if version in ("error-note-v18", "error-note-v18b"):
        from .v17_artifacts import verify_evaluation_authorization
        if getattr(args, "artifact_verification", None) is None:
            raise ValueError("v18 engine: --artifact-verification with separate user authorization required")
        authorization = verify_evaluation_authorization(args.artifact_verification, args.training_seal,
            {"code_commit": commit, "source_sha256": source})
    requests = _manifest(args.requests, args.requests_sha256)
    lengths = baseline_lengths(args.baseline_raw, args.baseline_raw_sha256)
    environment = collect_environment()
    if environment["torch"]["bf16_probe"]["passed"] is not True:
        raise ValueError("v17 engine: actual BF16 operation required")
    settings = None
    if version == "error-note-v18b":
        from .v18b_execution import read_settings, deadline_epoch, enforce_limits
        if getattr(args, "execution_settings", None) is None:
            raise ValueError("v18b engine: sealed execution settings required")
        settings = read_settings(args.execution_settings)
        if plan.get("execution_settings") != settings:
            raise ValueError("v18b engine: execution settings differ from the sealed campaign")
        if enforce_limits(settings) and deadline_epoch(settings, reserve_seconds=settings["post_eval_reserve_seconds"]) <= time.time():
            raise TimeoutError("v18b engine: first-billing deadline already reached")
    context = {"seal": seal_value, "model_ref": ref, "config": cfg, "hardware": hw, "requests": requests,
               "lengths": lengths, "quantiles": quantile_lengths(lengths, [row["sample_id"] for row in requests["items"]]),
               "training_protocol_version": version, "execution_settings": settings}
    root = args.output.parent / (args.output.stem + "-validation")
    root.mkdir(parents=True, exist_ok=False)
    selected, attempts = select_attempts(lambda mode, path, **kw: _isolated_attempt(args, mode, path, context, **kw), args.mode, root)
    if verify_training_artifacts(args.training_seal) != seal_value or source_sha256() != source:
        raise ValueError("v17 engine: training artifacts or evaluation source changed during validation")
    body = {"version": VERSION, "status": "PASS" if selected else "FAIL", "attempts": attempts,
            "training_artifact_sha256": sha256_file(args.training_seal),
            "training_artifact_content_sha256": seal_value["sha256"],
            "training_artifact_file_sha256": sha256_file(args.training_seal),
            "training_artifact_path": str(args.training_seal.resolve()), "model_ref_sha256": sha256_file(args.model_ref),
            "code_commit": commit, "evaluation_commit": commit, "source_sha256": source,
            "evaluation_source_sha256": source, "environment_sha256": digest(evaluation.environment_identity(environment)),
            "config_sha256": digest(cfg), "hardware_sha256": digest(hw), "hardware": hw,
            "protocol_id": cfg["protocol_id"], "requests_sha256": sha256_file(args.requests),
            "baseline_lengths_sha256": sha256_file(args.baseline_raw), "baseline_generated_tokens": lengths,
            "diagnostic_only": True,
            "used_for_training_or_answer_selection": False,
            "execution_change": "B2 execution eager to graph, if validated; no bitwise B2 equivalence claim"}
    if version in ("error-note-v18", "error-note-v18b"):
        from .v17_artifacts import _ref
        body["training_protocol_version"] = version
        body["artifact_verification"] = _ref(args.artifact_verification)
        body["evaluation_authorization"] = authorization
    if version == "error-note-v18b":
        body["execution_settings"] = settings
    if selected:
        observed, calibration = selected["observed_engine"], selected["calibration"]
        body |= {"selected_attempt": selected, "engine_mode": selected["engine_mode"], "selected_mode": selected["engine_mode"],
                 "slots": observed["slots"], "safe_slots": observed["slots"], "observed_engine": observed,
                 "actual_kv_token_capacity": observed["actual_kv_token_capacity"], "calibration": calibration,
                 "estimated_evaluation_seconds": calibration["estimated_evaluation_seconds"],
                 "evaluation_watchdog_seconds": calibration["evaluation_watchdog_seconds"],
                 "watchdog_seconds": calibration["evaluation_watchdog_seconds"],
                 "preflight_report": selected["preflight"]["path"], "preflight_file_sha256": selected["preflight"]["sha256"],
                 "pass": True, "model_ref_file_sha256": sha256_file(args.model_ref)}
    result = seal(body)
    if selected:
        validate_engine_receipt(result)
    exclusive(args.output, result)
    if selected is None:
        raise ValueError("v17 engine: no mode passed validation; preserved attempt receipts")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("validate")
    command.add_argument("--training-seal", "--training-artifact-seal", "--training-artifacts", type=Path, required=True)
    for name in ("model-ref", "base-path", "requests", "hardware", "eval-protocol", "output", "baseline-raw"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--requests-sha256", required=True)
    command.add_argument("--baseline-raw-sha256", required=True)
    command.add_argument("--code-commit", "--commit", required=True)
    command.add_argument("--source-hash")
    command.add_argument("--artifact-verification", type=Path)
    command.add_argument("--execution-settings", type=Path)
    command.add_argument("--mode", choices=("auto", *MODES), default="auto")
    command.add_argument("--max-seconds", type=float, default=PROBE_CAP_SECONDS)
    command.add_argument("--long-tail-allowance-seconds", type=float, default=1800.)
    command.add_argument("--long-tail-basis", default="explicit conservative 30-minute load/tail scenario")
    command.add_argument("--pf5-vision-exception", default=None,
                         help="recorded user authorization: evaluate with the vision LoRA application unconfirmed")
    args = parser.parse_args(argv)
    result = validate(args)
    print(json.dumps({key: result[key] for key in ("status", "selected_mode", "safe_slots",
                                                   "estimated_evaluation_seconds", "evaluation_watchdog_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
