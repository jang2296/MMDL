"""G11: one measured production rollout/update/synchronization, not training approval.

The longest initial input is selected from a frozen caller-provided gate cohort
before seeing rewards. Actual retry coverage is required, never fabricated by
padding a successful answer. Missing coverage is INCOMPLETE, not a memory pass.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict

import torch

from fine_tuning.modules import tensor_sha256
from . import batches, gate_update, prompts as prompt_builders, trainer
from .engines import RolloutRequest
from .rollout import RolloutConfig, seeds
from .schema import to_json
from .updates import _stage_time


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G11: " + message)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _input_row(prepared, images) -> dict:
    _require(set(prepared.pixels) == {"pixel_values", "image_grid_thw"},
             "exact actual image tensor fields required")
    grid = prepared.pixels["image_grid_thw"]
    _require(isinstance(grid, torch.Tensor) and grid.dtype == torch.long
             and grid.ndim == 2 and list(grid.shape) == [len(images), 3]
             and bool((grid > 0).all()) and bool((grid[:, 0] == 1).all()),
             "actual still-image processor grid required")
    _require(bool(prepared.token_ids), "actual initial prompt tokens required")
    return {"image_count": len(images), "image_grid_thw": grid.tolist(),
        "patch_cells": int(grid.prod(-1).sum()), "prompt_tokens": len(prepared.token_ids),
        "prompt_sha256": _hash(prepared.token_ids),
        "pixel_tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                "sha256": tensor_sha256(value)}
                          for key, value in prepared.pixels.items()},
        "images": [{"size": list(image.size), "mode": image.mode,
                    "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()} for image in images]}


def _check_input(prepared, images, expected, *, initial):
    observed = _input_row(prepared, images)
    if not initial:
        for key in ("prompt_tokens", "prompt_sha256"):
            observed.pop(key)
    _require(observed == {key: expected[key] for key in observed},
             "input changed after longest-cohort selection")


def _native_budgets(engine, protocol) -> dict:
    """Read resolved native configuration, not wrapper constructor arguments."""
    cfg = engine.llm.llm_engine.vllm_config
    model, cache, sched = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    mm = model.multimodal_config
    actual = {"kv_cache_memory_bytes": cache.kv_cache_memory_bytes,
        "max_num_seqs": sched.max_num_seqs, "max_model_len": model.max_model_len,
        "scheduler_max_model_len": sched.max_model_len, "enforce_eager": model.enforce_eager,
        "enable_prefix_caching": cache.enable_prefix_caching,
        "mm_processor_cache_gb": mm.mm_processor_cache_gb,
        "limit_per_prompt": dict(mm.limit_per_prompt), "mm_processor_kwargs": mm.mm_processor_kwargs,
        "generation_config": model.generation_config,
        "override_generation_config": model.override_generation_config}
    fixed = protocol["engine"]["vllm"]
    expected = {key: fixed[key] for key in ("kv_cache_memory_bytes", "max_num_seqs",
        "enforce_eager", "enable_prefix_caching", "mm_processor_cache_gb")}
    expected.update(max_model_len=protocol["context_limit"], scheduler_max_model_len=protocol["context_limit"],
        limit_per_prompt={"image": protocol["image"]["max_images_per_problem"], "video": 0},
        mm_processor_kwargs={"size": {"shortest_edge": protocol["image"]["min_pixels"],
            "longest_edge": protocol["image"]["max_pixels"]}, "do_resize": False},
        generation_config="vllm", override_generation_config={})
    _require(actual == expected and all(type(actual[key]) is int for key in
             ("kv_cache_memory_bytes", "max_num_seqs", "max_model_len", "scheduler_max_model_len"))
             and all(type(actual[key]) is bool for key in ("enforce_eager", "enable_prefix_caching")),
             "actual native cache/scheduler/input configuration differs from fixed protocol")
    return actual


class _Incomplete(Exception):
    pass


def _cohort(actors, structures, loader, engine, prompts, cfg, should_stop, record):
    _require(bool(actors) and set(actors) == set(structures)
             and all(key == actor.problem_id for key, actor in actors.items()),
             "exact nonempty actor/structure cohort required")
    rows = {}
    for pid in sorted(actors):
        if should_stop():
            raise _Incomplete("BUDGET_STOP_BEFORE_COHORT_CLOSED")
        actor, images = actors[pid], []
        try:
            images = loader(actor)
            _require(len(images) == len(actor.image_paths) and 1 <= len(images) <= 2
                     and all(image.mode == "RGB" for image in images), "original RGB image coverage required")
            request = RolloutRequest(pid + ":G11_INITIAL", None,
                [prompt_builders.user_message(prompt_builders.turn1_text(actor, prompts), len(images))],
                images, cfg.n[1], cfg.max_new_tokens[1],
                seeds(pid, engine.policy_version, 1, "T1:root", cfg.n[1], master_seed=cfg.master_seed))
            row = _input_row(engine.prepare(request), images)
            rows[pid] = row | {"actor_sha256": hashlib.sha256(to_json(actor).encode()).hexdigest(),
                              "structure_hash": structures[pid]}
            record({"kind": "G11_COHORT_INPUT", "problem_id": pid, **rows[pid]})
        finally:
            for image in images:
                image.close()
    # Frozen workload choice is independent of answers/reward. No resampling to pass.
    order = sorted(rows, key=lambda pid: (-rows[pid]["image_count"],
        -rows[pid]["prompt_tokens"], -rows[pid]["patch_cells"], pid))
    return rows, order


class _Memory:
    """Device-wide samples include non-Torch pools; allocator peaks are separate."""
    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.thread = None
        self.peak = self.count = 0
        self.total = None
        self.error = None

    def _sample(self):
        free, total = torch.cuda.mem_get_info(self.device)
        _require(0 <= free <= total and total > 0, "invalid actual device memory reading")
        if self.total is not None:
            _require(self.total == total, "device memory capacity changed")
        self.total = total
        self.peak = max(self.peak, total - free)
        self.count += 1

    def _poll(self):
        while not self.stop.wait(.1):
            try:
                self._sample()
            except Exception as exc:
                self.error = type(exc).__name__
                return

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._sample()
            self.thread = threading.Thread(target=self._poll, name="G11-memory", daemon=True)
            self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            _require(not self.thread.is_alive(), "device memory observer did not terminate")
        if self.device.type != "cuda":
            return {"device_sampled_peak_bytes": None, "torch_peak_allocated_bytes": None,
                    "torch_peak_reserved_bytes": None, "device_samples": 0,
                    "device_total_bytes": None, "memory_observed": False}
        self._sample()
        _require(self.error is None, "device memory observer failed: " + str(self.error))
        return {"device_sampled_peak_bytes": self.peak, "device_samples": self.count,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "device_total_bytes": self.total, "memory_observed": True}


class _TimedSampler:
    def __init__(self, engine, rows, device, record):
        self.engine, self.rows, self.device, self.record = engine, rows, device, record
        self.seconds = 0.
        self.output_tokens = self.calls = 0

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def prepare(self, request):
        prepared = self.engine.prepare(request)
        pid = request.request_id.rsplit(":v", 1)[0]
        _require(pid in self.rows, "rollout outside frozen cohort")
        _check_input(prepared, request.images, self.rows[pid], initial=request.messages is not None)
        return prepared

    def sample(self, requests):
        for request in requests:
            self.prepare(request)
        start = _stage_time(self.device)
        results = self.engine.sample(requests)
        elapsed = _stage_time(self.device) - start
        # Journal engine returns before grading; invalid backend returns remain failures.
        self.record({"kind": "G11_SAMPLER_RETURN", "seconds": elapsed,
                     "requests": [r.request_id for r in requests],
                     "returns": [asdict(row) for row in results]})
        self.seconds += elapsed
        self.calls += 1
        self.output_tokens += sum(len(c.token_ids) for row in results for c in row.completions)
        for request in requests:
            self.prepare(request)
        return results


def _measure_step(bundle, optimizer, scheduler, engine, preparation, actors, structures,
                  loader, verifier, prompts, protocol, probes, *, algorithm, step, seed,
                  should_stop, record, _native=False) -> dict:
    """Private CPU seam uses real tiny gradients but is never native/GPU evidence."""
    cfg = RolloutConfig.from_protocol(protocol, algorithm, master_seed=seed)
    cfg.validate()
    _require(engine.policy_version == preparation.policy_version == scheduler.last_epoch == step,
             "one synchronized actual policy cursor required")
    device = next(bundle.model.parameters()).device
    _require(all(p.device == device for p in bundle.model.parameters()), "single policy device required")
    result = {"status": "INCOMPLETE", "native_execution": _native,
        "training_allowed": False, "gpu_gates_certified": False, "algorithm": algorithm,
        "step_before": step, "step_after": step, "protocol_sha256": protocol["sha256"],
        "longest_sample_step_complete": False, "no_oom": None, "weights_updated": False,
        "engine_sync_required": False, "peak_memory_bytes": None, "step_seconds": None,
        "memory_scope": "sampled whole device + separate exact Torch allocator peaks; not additive",
        "workload_scope": "frozen measured cohort; longest initial prompt at maximum image count",
        "theoretical_full_output_cap_measured": False,
        "limitations": ["Observed rollout lengths are not synthetic full-cap padding or a worst-case OOM guarantee",
            "A failed post-update check cannot roll back weights/AdamW; caller must preserve its checkpoint",
            "CPU fixtures do not measure CUDA memory, native generation or learned self-correction",
            "No data, full-GPU-gate or financial authority is granted by this component"]}
    memory, batch, began = _Memory(device), None, None
    started_memory = False
    try:
        rows, order = _cohort(actors, structures, loader, engine, prompts, cfg, should_stop, record)
        stress = order[0]
        result.update(cohort_sha256=_hash(rows), cohort=rows, order=order, stress_problem_id=stress,
                      required_images=protocol["image"]["max_images_per_problem"],
                      required_attempts=cfg.max_attempts)
        if rows[stress]["image_count"] != result["required_images"]:
            raise _Incomplete("MAXIMUM_IMAGE_COUNT_NOT_IN_COHORT")
        timed = _TimedSampler(engine, rows, device, record)
        memory.start()
        started_memory = True
        began = _stage_time(device)
        batch = batches.collect_batch(actors, structures, order, 0, loader, timed, verifier,
            prompts, protocol, algorithm, should_stop=should_stop, seed=seed,
            on_group=lambda g: record({"kind": "G11_GROUP", "group": to_json(g)}),
            on_tree=lambda t: record({"kind": "G11_TREE", "tree": to_json(t)}))
        result.update(collection_status=batch.status, attempted=batch.attempted,
            discarded=batch.discarded, rollout_output_tokens=timed.output_tokens,
            rollout_calls=timed.calls, rollout_engine_seconds=timed.seconds,
            rollout_tokens_per_second=timed.output_tokens/timed.seconds if timed.seconds else None,
            rollout_collection_seconds=batch.seconds)
        if batch.status != "COLLECTED" or should_stop():
            raise _Incomplete("FULL_BATCH_NOT_COLLECTED_OR_BUDGET_STOP")
        witnesses = [s for t in batch.trees if t.problem_id == stress
                     for g in t.groups if g.usable and g.kind != "DIAG" for s in g.samples
                     if s.turn == cfg.max_attempts]
        if not witnesses:
            raise _Incomplete("LONGEST_INPUT_ACTUAL_MAXIMUM_RETRY_NOT_OBSERVED")
        witness = max(witnesses, key=lambda s: len(s.prompt_token_ids)+len(s.completion_token_ids))
        result["witness"] = {"sample_id": witness.sample_id, "turn": witness.turn,
            "prompt_tokens": len(witness.prompt_token_ids), "output_tokens": len(witness.completion_token_ids),
            "actual_total_tokens": len(witness.prompt_token_ids)+len(witness.completion_token_ids),
            "finish_reason": witness.finish_reason, "sample_sha256": hashlib.sha256(to_json(witness).encode()).hexdigest()}
        # Grading the final turn can mutate caller-owned images too. Bind both
        # sampler and trainer preprocessing again before any optimizer update.
        for tree in batch.trees:
            first = next(s for g in tree.groups if g.kind == "T1" for s in g.samples)
            request = RolloutRequest(tree.problem_id + f":v{step}:G11_UPDATE",
                first.prompt_token_ids, None, batch.images[tree.problem_id], 1, 1, [seed])
            timed.prepare(request)
            _check_input(preparation.prepare(request), request.images, rows[tree.problem_id], initial=False)
        # One shared G4/G11 update, not another optimizer step for each gate.
        result["engine_sync_required"] = True
        result["weights_updated"] = None  # Until checked, a failed update may have changed weights.
        update = (gate_update.measure_update if _native else gate_update._measure_update)(
            bundle, optimizer, scheduler, batch.trees, batch.images, preparation, protocol,
            algorithm, step=step, seed=seed)
        result.update(weights_updated=True, step_after=step+1, G4=update)
        record({"kind": "G11_UPDATE_UNSYNCHRONIZED", "G4": update})
        start = _stage_time(device)
        result["sync_probes"] = trainer.synchronize(bundle, preparation, engine, probes,
                                                   protocol, step+1, push=True)
        push_seconds = _stage_time(device)-start
        result.update(engine_sync_required=False,
            status="STEP_MEASURED_NOT_GPU_APPROVAL", no_oom=True,
            seconds={"rollout": batch.seconds, **update["update"]["seconds"], "push": push_seconds})
        delta = update["update"]["mean_log_ratio_by_sample"].get(witness.sample_id)
        if delta is None or abs(delta) > protocol["loss"]["mismatch"]["delta_seq"]:
            result["status"] = "INCOMPLETE"
            raise _Incomplete("LONGEST_ACTION_MISMATCH_MASKED_NOT_TRAINED")
        result["longest_sample_step_complete"] = True
    except _Incomplete as exc:
        result["incomplete_reason"] = str(exc)
    except BaseException as exc:
        result.update(status="FAILED", error_type=type(exc).__name__,
                      no_oom=False if isinstance(exc, torch.cuda.OutOfMemoryError) else None)
        raise
    finally:
        result["weights_updated"] = True if scheduler.last_epoch > step else result["weights_updated"]
        result["step_after"] = scheduler.last_epoch
        try:
            if began is not None:
                result["step_seconds"] = _stage_time(device)-began
            if started_memory:
                result.update(memory.finish())
                result["peak_memory_bytes"] = result["device_sampled_peak_bytes"]
        except BaseException as exc:
            result.update(status="FAILED", longest_sample_step_complete=False,
                          measurement_error_type=type(exc).__name__)
            raise
        finally:
            # Always stop an observer, including a failing CUDA synchronization.
            memory.stop.set()
            if memory.thread is not None:
                memory.thread.join(timeout=5)
            if batch is not None:
                for values in batch.images.values():
                    for image in values:
                        image.close()
            record({"kind": "G11_RESULT", **result})
    return result


def measure_step(bundle, optimizer, scheduler, engine, preparation, actors, structures,
                 loader, verifier, prompts, protocol, probes, *, algorithm, step, seed,
                 should_stop, record) -> dict:
    """Live native G11 stress: two images + actual third attempt, one update + sync.

Caller must first establish separate data eligibility and bounded spend authority.
An approved two-image cohort is required; ordinary one-image pilot data cannot pass.
"""
    from .engines import InputPreparation
    from .gate_weights import _pinned_config, require_idle_engine
    from .gpu_gates import measure_inventory

    device = next(bundle.model.parameters()).device
    _require(device.type == "cuda", "actual CUDA policy required; no CPU fallback")
    _require(algorithm == "ALG-NB3", "maximum three-attempt stress requires ALG-NB3")
    _require(type(preparation) is InputPreparation and preparation.image_cfg == protocol["image"]
             and preparation.context_limit == protocol["context_limit"], "pinned actual input preparation required")
    _pinned_config(bundle.model.config)
    measure_inventory(bundle, protocol)
    before = require_idle_engine(engine, require_push=step > 0)
    budgets = _native_budgets(engine, protocol)
    _require(str(device) == before["device"] and engine.image_cfg == preparation.image_cfg
             and engine.context_limit == preparation.context_limit, "same native device and input budgets required")
    record({"kind": "G11_NATIVE_RUNTIME_BEFORE", "runtime": before, "budgets": budgets})
    result = _measure_step(bundle, optimizer, scheduler, engine, preparation, actors, structures,
        loader, verifier, prompts, protocol, probes, algorithm=algorithm, step=step, seed=seed,
        should_stop=should_stop, record=record, _native=True)
    after = require_idle_engine(engine, require_push=result["weights_updated"] or step > 0)
    _require(after["device"] == before["device"] and after["policy_version"] == result["step_after"],
             "native runtime/cursor changed outside measured update")
    _require(_native_budgets(engine, protocol) == budgets, "native budgets changed during measurement")
    result.update(runtime_before=before, runtime_after=after, native_budgets=budgets)
    record({"kind": "G11_NATIVE_RUNTIME_AFTER", "runtime": after, "budgets": budgets})
    return result
