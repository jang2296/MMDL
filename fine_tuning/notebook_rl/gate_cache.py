"""G6 cache measurements across caller-owned, sequential native processes.

The default-cache control deliberately leaves stale caches after a weight load.
It owns a disposable LLM, never exposes that LLM to training, and shuts it down.
The model-free coordinator must wait for each process to exit before the next.
"""
from __future__ import annotations

import math
import os
import subprocess
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable

import torch

from .engines import InputPreparation, VLLMRolloutEngine, _weight_inventory, vllm_kwargs
from .gate_images import _observe_visual
from .gate_tokens import _pixels
from .gate_weights import _layout, _pinned_config, _read_worker, require_idle_engine
from .trainer import SyncProbe


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("G6: " + message)


def _process() -> dict:
    import psutil
    process = psutil.Process()
    return {"pid": process.pid, "parent_pid": process.ppid(), "create_time": process.create_time(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def _closed_previous(report: dict) -> dict:
    import psutil
    previous = report.get("process", {})
    _require(set(previous) == {"pid", "parent_pid", "create_time", "boot_id"} and type(previous["pid"]) is int
             and previous["pid"] > 0 and previous["pid"] != os.getpid()
             and type(previous["create_time"]) in (int, float)
             and math.isfinite(previous["create_time"]) and previous["create_time"] > 0
             and previous["boot_id"] == _process()["boot_id"],
             "separate actual predecessor identity required")
    try:
        process = psutil.Process(previous["pid"])
        _require(process.create_time() != previous["create_time"] or not process.is_running(),
                 "previous GPU process is still live; wait for exit before allocation")
    except psutil.NoSuchProcess:
        pass
    return previous


def _dedicated_process() -> None:
    import psutil
    _require(torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             and torch.cuda.memory_allocated() == 0, "model-free dedicated CUDA process required")
    ancestors = {p.pid for p in psutil.Process().parents()}
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    _require(not ancestors.intersection(int(s.strip()) for s in active.splitlines() if s.strip().isdigit()),
             "coordinator/ancestors must not retain a CUDA model")


def _probes(probes: list[SyncProbe]) -> None:
    _require(len(probes) == 2 and {len(p.request.images) for p in probes} == {1, 2}
             and len({p.request.request_id for p in probes}) == 2
             and all(p.request.n == 1 and 1 <= len(p.completion_ids) <= 32
                     and all(type(t) is int and t >= 0 for t in p.completion_ids) for p in probes),
             "distinct fixed single-/two-image probes required")


def _state(llm, config) -> dict:
    layout = _layout(_weight_inventory(config))
    results = llm.apply_model(lambda model: _read_worker(model, layout, native=True))
    _require(isinstance(results, list) and len(results) == 1, "one native worker required")
    result = results[0]
    _require(len(result["parameters"]) == 605, "pinned complete worker inventory required")
    return {"parameters": {row["worker_name"]: {"shape": row["shape"], "dtype": row["dtype"],
                       "sha256": row["sha256"]} for row in result["parameters"]},
            "device": result["device"]}


def _input(preparation, probe: SyncProbe) -> dict:
    prepared = preparation.prepare(probe.request)
    return {"request_id": probe.request.request_id, "prompt_token_ids": list(prepared.token_ids),
            "completion_ids": list(probe.completion_ids), "seed": probe.request.seeds[0],
            "image": _pixels(prepared)}


def _probabilities(values: list[float], count: int) -> None:
    _require(len(values) == count and count > 0 and all(type(p) in (int, float)
             and math.isfinite(p) and p <= 1e-5 for p in values), "finite aligned probabilities required")


def _capture(engine, probes: list[SyncProbe], stage: str, record: Callable[[dict], None],
             prefix_actions: list[list[int]] | None = None) -> dict:
    _probes(probes)
    rows = []
    state = _state(engine.llm, engine.model_config)
    for i, probe in enumerate(probes):
        item = _input(engine, probe)
        probabilities = engine.probe_logprobs(probe.request, probe.completion_ids)
        row = {"input": item, "logprobs": probabilities}
        record({"kind": "G6_FIXED_PROBE", "stage": stage, **row})
        _probabilities(probabilities, len(probe.completion_ids))
        if prefix_actions is not None:
            actions = prefix_actions[i]
            _require(len(actions) == 1 and type(actions[0]) is int and actions[0] >= 0,
                     "one actual generated default-control action required")
            values = engine.probe_logprobs(probe.request, actions)
            row["control_action"] = {"token_ids": actions, "logprobs": values}
            record({"kind": "G6_CONTROL_ACTION_REFERENCE", "stage": stage, **row})
            _probabilities(values, 1)
        _require(_input(engine, probe) == item, "probe input changed")
        rows.append(row)
    _require(_state(engine.llm, engine.model_config) == state, "measurement mutated worker weights")
    return {"stage": stage, "pid": os.getpid(), "process": _process(), "state": state, "probes": rows,
            "training_allowed": False, "gpu_gates_certified": False}


def measure_current(engine, probes: list[SyncProbe], *, stage: str, record: Callable[[dict], None]) -> dict:
    """Capture the same actual engine BEFORE and AFTER the caller's real update/push."""
    _require(stage in ("BEFORE", "AFTER") and callable(record), "stage and durable journal required")
    runtime = require_idle_engine(engine, require_push=stage == "AFTER")
    _pinned_config(engine.model_config)
    cfg = engine.llm.llm_engine.vllm_config
    _require(engine._synchronized is True and not cfg.cache_config.enable_prefix_caching
             and cfg.model_config.multimodal_config.mm_processor_cache_gb == 0,
             "normal synchronized cache-disabled training engine required")
    result = _capture(engine, probes, stage, record)
    _require(require_idle_engine(engine, require_push=stage == "AFTER") == runtime, "runtime changed")
    result.update(runtime=runtime, native_execution=True)
    return result


def measure_fresh(model_path: Path, image_cfg: dict, *, kv_cache_memory_bytes: int,
        max_num_seqs: int, max_model_len: int, probes: list[SyncProbe], control: dict,
        record: Callable[[dict], None]) -> dict:
    """Allocate only after the old process exits; always close this isolated engine."""
    _require(callable(record) and control.get("stage") == "DEFAULT_CONTROL"
             and control.get("pid") != os.getpid() and len(control.get("probes", [])) == 2,
             "closed separate control measurement required")
    predecessor = _closed_previous(control)
    _probes(probes)
    _dedicated_process()
    engine = None
    try:
        engine = VLLMRolloutEngine(model_path, image_cfg, kv_cache_memory_bytes, max_num_seqs, max_model_len)
        runtime = require_idle_engine(engine, require_push=False)
        _pinned_config(engine.model_config)
        _require(engine._initialization_open is True and engine._last_loaded_version is None
                 and engine.policy_version == 0, "fresh engine, not an in-process weight replacement, required")
        result = _capture(engine, probes, "FRESH", record,
                          [r["prefix_after"]["token_ids"] for r in control["probes"]])
        _require(require_idle_engine(engine, require_push=False) == runtime, "fresh runtime changed")
        result.update(runtime=runtime, native_execution=True, snapshot_path=str(engine.model_dir),
                      closed_predecessor=predecessor)
        return result
    except BaseException as exc:
        record({"kind": "G6_FRESH_ERROR", "pid": os.getpid(), "error_type": type(exc).__name__})
        raise
    finally:
        if engine is not None:
            engine.llm.llm_engine.engine_core.shutdown()


def _cache_trace(manager, runner, model, stack: ExitStack, hits: list[dict], visual: list[dict]) -> None:
    original = manager.check_and_update_cache
    had_local = "check_and_update_cache" in vars(manager)
    previous = vars(manager).get("check_and_update_cache")

    def check(request, input_id):
        answer = original(request, input_id)
        key = request.mm_features[input_id].identifier
        hits.append({"identifier": key, "hit": answer,
                     "worker_entry_present": key in runner.encoder_cache})
        return answer

    def restore():
        if had_local:
            manager.check_and_update_cache = previous
        else:
            del manager.check_and_update_cache
    stack.callback(restore)
    manager.check_and_update_cache = check
    _observe_visual(model, stack, visual)


def _control_call(llm, preparation, probe, *, prefix: bool, hits, visual, record) -> dict:
    item = preparation.prepare(probe.request)
    ids = item.token_ids if prefix else item.token_ids + probe.completion_ids
    _require(len(ids) + 1 <= preparation.context_limit, "control probe overflows context")
    # Reuse production transport, but explicitly retain identifiers in this
    # disposable control. Training continues to use version/call-unique UUIDs.
    prompt = VLLMRolloutEngine._prompt(preparation, item, ids)
    prompt["multi_modal_uuids"] = {"image": [f"G6:{item.image_key}:{i}" for i in range(len(item.images))]}
    params = import_module("vllm").SamplingParams(n=1, temperature=1, top_p=1, top_k=0,
        presence_penalty=0, repetition_penalty=1, max_tokens=1, seed=probe.request.seeds[0],
        logprobs=0, prompt_logprobs=None if prefix else 0)
    start, visuals = len(hits), len(visual)
    result = llm.generate([prompt], params, use_tqdm=False)
    _require(len(result) == 1, "one native control return required")
    output = result[0]
    raw = {"kind": "G6_DEFAULT_RETURN", "request_id": probe.request.request_id, "prefix": prefix,
        "actual_prompt_token_ids": list(output.prompt_token_ids), "finished": output.finished,
        "num_cached_tokens": output.num_cached_tokens,
        "outputs": [{"token_ids": list(c.token_ids), "finish_reason": c.finish_reason,
                     "logprobs": [{str(t): float(p.logprob) for t, p in row.items()}
                                  for row in (c.logprobs or [])]} for c in output.outputs],
        "prompt_logprobs": None if output.prompt_logprobs is None else [None if row is None else
            {str(t): float(p.logprob) for t, p in row.items()} for row in output.prompt_logprobs],
        "encoder_checks": hits[start:], "visual_calls": visual[visuals:]}
    record(raw)
    _require(output.finished and list(output.prompt_token_ids) == ids and len(output.outputs) == 1,
             "native control prompt/coverage mismatch")
    completion = output.outputs[0]
    _require(len(completion.token_ids) == 1 and completion.logprobs is not None,
             "one actual generated diagnostic action required")
    if prefix:
        action = list(completion.token_ids)
        probabilities = [float(completion.logprobs[0][action[0]].logprob)]
        _require(output.prompt_logprobs is None, "prompt logprobs would bypass prefix reuse")
    else:
        action = list(probe.completion_ids)
        _require(output.prompt_logprobs is not None, "missing fixed-action probabilities")
        probabilities = [float(row[t].logprob) for row, t in
                         zip(output.prompt_logprobs[len(item.token_ids):], action, strict=True)]
    _probabilities(probabilities, len(action))
    return {"token_ids": action, "logprobs": probabilities,
            "num_cached_tokens": output.num_cached_tokens,
            "encoder_checks": hits[start:], "visual_call_count": len(visual)-visuals}


def _uncleared_load(llm, named: Iterable[tuple[str, torch.Tensor]], shapes: dict) -> None:
    """Deliberate negative control; never mark a training engine synchronized."""
    seen = set()
    def checked():
        for name, tensor in named:
            _require(name in shapes and name not in seen and tensor.device.type == "cpu"
                     and tuple(tensor.shape) == shapes[name] and tensor.dtype == torch.bfloat16
                     and bool(torch.isfinite(tensor).all()), "complete finite CPU BF16 snapshot required")
            seen.add(name)
            yield name, tensor
        _require(seen == shapes.keys(), "incomplete default-control load")
    def load(model):
        return set(model.load_weights(checked())) == {n for n, _ in model.named_parameters()}
    _require(llm.apply_model(load) == [True] and seen == shapes.keys(), "native control load incomplete")


def measure_default_control(model_path: Path, image_cfg: dict, *, kv_cache_memory_bytes: int,
        max_num_seqs: int, max_model_len: int, post_weights: Iterable[tuple[str, torch.Tensor]],
        previous: dict, probes: list[SyncProbe], record: Callable[[dict], None]) -> dict:
    """Allocate one disposable native LLM in its OWN process, never beside HF.

    The caller must supply file-verified pre/post snapshots and an outer watchdog.
    Snapshot provenance/G4 and spending permission remain the runner's responsibility.
    """
    from importlib.metadata import version
    from transformers import AutoConfig, AutoProcessor

    _probes(probes)
    _require(previous.get("stage") == "AFTER" and previous.get("native_execution") is True,
             "actual post-push measurement required before the default control")
    predecessor = _closed_previous(previous)
    _require(callable(record) and version("vllm") == "0.11.0", "pinned native CUDA required")
    _dedicated_process()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _pinned_config(config)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=image_cfg["min_pixels"], max_pixels=image_cfg["max_pixels"])
    processor.image_processor.size = {"shortest_edge": image_cfg["min_pixels"], "longest_edge": image_cfg["max_pixels"]}
    preparation = InputPreparation(processor, config, image_cfg, max_model_len)
    preparation._call = 0
    kwargs = vllm_kwargs(model_path, image_cfg, kv_cache_memory_bytes, max_num_seqs, max_model_len)
    del kwargs["enable_prefix_caching"], kwargs["mm_processor_cache_gb"]
    llm = import_module("vllm").LLM(**kwargs)
    core = llm.llm_engine.engine_core
    try:
        _require(isinstance(core, import_module("vllm.v1.engine.core_client").InprocClient), "in-process V1 required")
        executor = core.engine_core.model_executor
        _require(isinstance(executor, import_module("vllm.executor.uniproc_executor").UniProcExecutor),
                 "native TP1 executor required")
        cfg, scheduler = llm.llm_engine.vllm_config, core.engine_core.scheduler
        _require(cfg.cache_config.enable_prefix_caching is True
                 and cfg.model_config.multimodal_config.mm_processor_cache_gb == 4,
                 "actual pinned V1 default caches required; environment override not accepted")
        rows: list[dict] = []
        hits: list[dict] = []
        visual: list[dict] = []
        with ExitStack() as stack:
            worker = executor.driver_worker
            llm.apply_model(lambda model: _cache_trace(scheduler.encoder_cache_manager,
                worker.model_runner, model, stack, hits, visual))
            for probe in probes:
                item = _input(preparation, probe)
                fixed = _control_call(llm, preparation, probe, prefix=False, hits=hits, visual=visual, record=record)
                prefix = _control_call(llm, preparation, probe, prefix=True, hits=hits, visual=visual, record=record)
                rows.append({"input": item, "fixed_before": fixed, "prefix_before": prefix})
            before = _state(llm, config)
            _require(not scheduler.has_unfinished_requests()
                     and not any(scheduler.encoder_cache_manager.cached.values()), "idle control required")
            _uncleared_load(llm, post_weights, _weight_inventory(config))
            after = _state(llm, config)
            record({"kind": "G6_UNCLEARED_CONTROL_LOAD", "before": before, "after": after})
            _require(after == previous["state"], "control did not load the actually measured post-update tensors")
            for row, probe in zip(rows, probes, strict=True):
                row["prefix_after"] = _control_call(llm, preparation, probe, prefix=True,
                                                  hits=hits, visual=visual, record=record)
                row["fixed_after"] = _control_call(llm, preparation, probe, prefix=False,
                                                 hits=hits, visual=visual, record=record)
                _require(_input(preparation, probe) == row["input"], "control input changed")
            _require(_state(llm, config) == after, "control measurement mutated weights")
        return {"stage": "DEFAULT_CONTROL", "pid": os.getpid(), "process": _process(),
                "closed_predecessor": predecessor, "state_before": before, "state_after": after,
                "probes": rows, "cache_settings": {"enable_prefix_caching": True, "mm_processor_cache_gb": 4},
                "native_execution": True, "unsafe_control": True, "training_allowed": False,
                "gpu_gates_certified": False, "weights_updated_by_optimizer": False}
    except BaseException as exc:
        record({"kind": "G6_CONTROL_ERROR", "pid": os.getpid(), "error_type": type(exc).__name__})
        raise
    finally:
        core.shutdown()


def _difference(a: list[float], b: list[float]) -> dict:
    _probabilities(a, len(b))
    _probabilities(b, len(a))
    values = [abs(x-y) for x, y in zip(a, b, strict=True)]
    return {"mean_abs": sum(values)/len(values), "max_abs": max(values)}


def _compare(before: dict, after: dict, control: dict, fresh: dict, threshold: float) -> dict:
    """Arithmetic seam for authored CPU tests; no native execution certification."""
    _require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 < threshold <= .02,
             "predeclared bounded comparison threshold required")
    _require([v.get("stage") for v in (before, after, control, fresh)] ==
             ["BEFORE", "AFTER", "DEFAULT_CONTROL", "FRESH"], "all distinct measured stages required")
    _require(before["state"] == control["state_before"] and after["state"] == control["state_after"] == fresh["state"],
             "pre/post/fresh worker tensors do not match")
    pre, post = before["state"]["parameters"], after["state"]["parameters"]
    roots = ["visual.merger.", *[f"visual.deepstack_merger_list.{i}." for i in range(3)]]
    mergers = {k for k in pre if any(k.startswith(root) for root in roots)}
    _require(set(pre) == set(post) and len(mergers) == 24
             and all(pre[k]["shape"] == post[k]["shape"] and
                     pre[k]["dtype"] == post[k]["dtype"] == "torch.bfloat16" for k in pre)
             and any(pre[k]["sha256"] != post[k]["sha256"] for k in mergers),
             "effective merged BF16 Merger change required; FP32-only changes do not exercise G6")
    _require(all(len(v["probes"]) == 2 for v in (before, after, control, fresh)), "two probe coverage required")
    _require({len(r["input"]["image"]["image_spans"]) for r in before["probes"]} == {1, 2}
             and len({r["input"]["request_id"] for r in before["probes"]}) == 2,
             "distinct single-/two-image probe coverage required")
    rows = []
    for b, a, c, f in zip(before["probes"], after["probes"], control["probes"], fresh["probes"], strict=True):
        _require(b["input"] == a["input"] == c["input"] == f["input"], "probe identity/history/images differ")
        fixed, prefix = c["fixed_after"], c["prefix_after"]
        _require(type(prefix["num_cached_tokens"]) is int and prefix["num_cached_tokens"] > 0,
                 "default prefix reuse was not observed")
        image_count = len(c["input"]["image"]["image_spans"])
        hit_keys = {r["identifier"] for r in fixed["encoder_checks"] if r["hit"] is True and r["worker_entry_present"] is True}
        expected_keys = {f"G6:{c['input']['image']['image_key']}:{i}" for i in range(image_count)}
        _require(image_count in (1, 2) and hit_keys == expected_keys and fixed["visual_call_count"] == 0,
                 "default encoder reuse was not observed independently of prefix reuse")
        _require(fixed["token_ids"] == c["input"]["completion_ids"]
                 and prefix["token_ids"] == f["control_action"]["token_ids"], "comparison actions differ")
        normal = _difference(a["logprobs"], f["logprobs"])
        _require(normal["max_abs"] <= threshold, "normal cache-invalidating engine differs from fresh engine")
        rows.append({"request_id": a["input"]["request_id"], "normal": normal,
            "default_encoder": _difference(fixed["logprobs"], f["logprobs"]),
            "default_prefix": _difference(prefix["logprobs"], f["control_action"]["logprobs"]),
            "prefix_cached_tokens": prefix["num_cached_tokens"], "encoder_hit_count": len(hit_keys)})
    return {"fresh_engine_match": True, "default_cache_comparison_measured": True,
        "default_mismatch_observed": any(r[k]["max_abs"] > threshold for r in rows for k in
                                         ("default_encoder", "default_prefix")),
        "threshold": threshold, "threshold_metric": "max_abs_token_logprob_delta",
        "probes": rows, "native_execution": False,
        "training_allowed": False, "gpu_gates_certified": False}


def compare_cache(before: dict, after: dict, control: dict, fresh: dict, *, threshold: float,
                  current_process, control_process, fresh_process) -> dict:
    """Compare only completed sequential children; never accept a success boolean."""
    _require(not torch.cuda.is_initialized(), "model-free coordinator required")
    processes = (current_process, control_process, fresh_process)
    _require(all(isinstance(p, subprocess.Popen) and p.poll() == 0 for p in processes)
             and len({p.pid for p in processes}) == 3 and os.getpid() not in {p.pid for p in processes},
             "three completed separate native processes required")
    _require(before["pid"] == after["pid"] == current_process.pid
             and control["pid"] == control_process.pid and fresh["pid"] == fresh_process.pid
             and all(v.get("native_execution") is True for v in (before, after, control, fresh)),
             "measurement/process identity mismatch")
    _require(before["process"] == after["process"] and control["closed_predecessor"] == after["process"]
             and fresh["closed_predecessor"] == control["process"]
             and all(r["process"]["parent_pid"] == os.getpid() for r in (before, control, fresh)),
             "sequential owned process closure was not measured")
    _require(after["runtime"]["policy_version"] > before["runtime"]["policy_version"]
             and after["runtime"]["last_loaded_version"] == after["runtime"]["policy_version"],
             "actual completed advancing training push required")
    result = _compare(before, after, control, fresh, threshold)
    result.update(native_execution=True,
        limitations=["Checks fixed probe actions, not every response or image",
                     "A default-cache mismatch is observed, not required or manufactured",
                     "G4 optimizer evidence, snapshot provenance and complete runtime/data approval are separate"])
    return result
