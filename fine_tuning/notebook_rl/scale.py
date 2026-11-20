"""Opt-in speed/scale options for the repair-notebook protocol.

The group is bound into the protocol (and therefore its sha256) only when at least one option
is given; an absent group means the historical defaults below, so every default, compact and
flagged protocol hash and every historical capture replay is unchanged.
Nothing here certifies GPU behaviour: BF16 and vLLM wiring is CPU/mock tested only.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

VERSION = "scale-options-v1"
RESIDENT_POLICY = "adamw-moments-resident-v1"
DEFAULTS: dict[str, Any] = {"rollout_batch_rows": None, "problems_per_update": 2, "checkpoint_every_updates": 1,
    "train_problems": 10, "max_updates": 5, "evidence_audit_rule": None, "policy_precision": "nf4",
    "rollout_engine": "hf", "vllm": None, "park_adam_moments": True}
_VLLM_KEYS = {"gpu_memory_utilization", "max_num_seqs", "enable_prefix_caching", "kv_cache_memory_bytes"}
_RULE_KEYS = {"first", "last", "every"}
RECEIPT_NAME = "throughput_receipt.json"
# vLLM sync probes: a continuation the current HF policy itself sampled, not forced tail tokens.
VLLM_SYNC_PROBE = "hf-policy-sampled-32-v1"
VLLM_SYNC_PROBE_TOKENS = 32
# User decision 2026-10-31, verbatim: "vLLM 전용 기준 0.04로 보정". The HF<->HF threshold
# engine.probe_max_mean_abs_logprob_diff stays 0.02; only the separate vLLM-sampler<->HF-policy report uses 0.04.
# Evidence (A40, vLLM 0.11.0 vs HF BF16, identical F0 weights, no push): hf-gvlqa:000541 32 tokens mean |d| 0.0066;
# ScienceQA 001890 (1,120 image tokens) 32 tokens 0.0245, 4x128-token continuations pooled 0.0208 (per continuation
# 0.0191-0.0228, median 0.0002, 13% of tokens > 0.05): a consistent cross-engine numeric gap, not a weight error.
# Max measured 0.0228 -> 0.04 (about 1.75x headroom). Bound into vLLM protocols only (HF/NF4 hashes unchanged).
VLLM_SYNC_THRESHOLD = 0.04
VLLM_SYNC_THRESHOLD_LABEL = "vllm-hf-bf16-calibrated-0.04-v1"
VLLM_SYNC_THRESHOLD_KEY = "vllm_probe_max_mean_abs_logprob_diff"
_LABEL_KEYS = ("vllm_sync_probe", "vllm_sync_threshold")


def _count(value: Any, low: int, high: int | None = None) -> bool:
    return type(value) is int and value >= low and (high is None or value <= high)


def validate(explicit: dict) -> dict:
    """Validate an explicit option group; returns a sorted deep copy of the given keys only."""
    if (type(explicit) is not dict or not explicit or not set(explicit) <= set(DEFAULTS)
            or any(value is None for value in explicit.values())):
        raise ValueError("scale: explicit nonempty group of known options required")
    opts = DEFAULTS | explicit
    rule, vllm = opts["evidence_audit_rule"], opts["vllm"]
    if (opts["rollout_batch_rows"] is not None and not _count(opts["rollout_batch_rows"], 1, 64)
            or not all(_count(opts[key], 1) for key in ("problems_per_update", "checkpoint_every_updates",
                                                         "train_problems", "max_updates"))
            or opts["train_problems"] != opts["problems_per_update"] * opts["max_updates"]):
        raise ValueError("scale: positive batch rows/sizes with train_problems == problems_per_update * max_updates")
    if rule is not None and (type(rule) is not dict or set(rule) != _RULE_KEYS
            or type(rule["first"]) is not bool or type(rule["last"]) is not bool or not _count(rule["every"], 0)
            or not (rule["first"] or rule["last"] or rule["every"])):
        raise ValueError("scale: evidence audit rule {first: bool, last: bool, every: int>=0} with an audited step")
    if (opts["policy_precision"] not in ("nf4", "bf16") or opts["rollout_engine"] not in ("hf", "vllm")
            or type(opts["park_adam_moments"]) is not bool):
        raise ValueError("scale: precision nf4|bf16, engine hf|vllm and a boolean parking flag required")
    if opts["rollout_engine"] == "vllm" and (opts["policy_precision"] != "bf16"
            or opts["rollout_batch_rows"] is not None or type(vllm) is not dict or set(vllm) != _VLLM_KEYS
            or type(vllm["gpu_memory_utilization"]) is not float
            or not 0 < vllm["gpu_memory_utilization"] <= 1 or not _count(vllm["max_num_seqs"], 1)
            or not _count(vllm["kv_cache_memory_bytes"], 1) or type(vllm["enable_prefix_caching"]) is not bool):
        raise ValueError("scale: vLLM sampling requires bf16, no HF batch rows and a complete vllm budget")
    if opts["rollout_engine"] == "hf" and vllm is not None:
        raise ValueError("scale: vllm budget without the vllm rollout engine")
    return {key: copy.deepcopy(explicit[key]) for key in sorted(explicit)}


def bind(protocol: dict, explicit: dict) -> None:
    """Bind the group into an unhashed repair protocol and keep its batch/budget fields consistent."""
    group = validate(explicit)
    protocol["scale"] = {"version": VERSION, **group, **_probe_label(group)}
    if _probe_label(group):
        protocol["engine"][VLLM_SYNC_THRESHOLD_KEY] = VLLM_SYNC_THRESHOLD
    opts = options(protocol)
    protocol["batch"].update(problems_per_step=opts["problems_per_update"],
                             max_attempted_problems_per_step=opts["problems_per_update"])
    protocol["local_budget"].update(train_problems=opts["train_problems"], train_updates=opts["max_updates"])
    if opts["policy_precision"] == "bf16":
        # A BF16 policy never inherits the NF4 identity contract.
        protocol.pop("quantization", None)


def explicit(protocol: dict) -> dict | None:
    """The bound explicit group without its version, or None (the historical default)."""
    bound = protocol.get("scale")
    if bound is None:
        return None
    if type(bound) is not dict or bound.get("version") != VERSION:
        raise ValueError("scale: unknown bound option group version")
    group = validate({key: value for key, value in bound.items() if key != "version" and key not in _LABEL_KEYS})
    labels, engine = _probe_label(group), protocol.get("engine")
    if ({key: value for key, value in bound.items() if key in _LABEL_KEYS} != labels or type(engine) is not dict
            or (VLLM_SYNC_THRESHOLD_KEY in engine) != bool(labels)
            or engine.get(VLLM_SYNC_THRESHOLD_KEY, VLLM_SYNC_THRESHOLD) != VLLM_SYNC_THRESHOLD):
        raise ValueError("scale: a vLLM group binds exactly its sampler sync probe source and calibrated threshold")
    return group


def _probe_label(group: dict) -> dict:
    """Bound only for vLLM sampling, so every HF/NF4 protocol hash is unchanged."""
    return ({"vllm_sync_probe": VLLM_SYNC_PROBE, "vllm_sync_threshold": VLLM_SYNC_THRESHOLD_LABEL}
            if (DEFAULTS | group)["rollout_engine"] == "vllm" else {})


def sampler_sync_threshold(protocol: dict) -> float:
    """Threshold for the separate vLLM-sampler<->HF-policy report (the bound 0.04); never for HF<->HF reports."""
    if options(protocol)["rollout_engine"] != "vllm":
        raise ValueError("scale: only a vLLM-sampling protocol binds a sampler sync threshold")
    return protocol["engine"][VLLM_SYNC_THRESHOLD_KEY]


def options(protocol: dict) -> dict:
    group = explicit(protocol)
    return dict(DEFAULTS) if group is None else DEFAULTS | group


def scaled(protocol: dict) -> bool:
    return "scale" in protocol


def bf16(protocol: dict) -> bool:
    return options(protocol)["policy_precision"] == "bf16"


def audited(protocol: dict, step: int, *, final_batch: bool) -> bool:
    """Fixed-before-run VJP audit rule; no rule means every update is measured (the default)."""
    opts = options(protocol)
    rule = opts["evidence_audit_rule"]
    if rule is None:
        return True
    return bool(rule["first"] and step == 0
                or rule["last"] and (final_batch or step == opts["max_updates"] - 1)
                or rule["every"] and (step + 1) % rule["every"] == 0)


def fold_s(current: bool | None, value: bool | None) -> bool | None:
    """Three-state S over committed updates: True if any audited True, False if audited and none True."""
    if current is True or value is True:
        return True
    return False if current is False or value is False else None


def initial_checkpoint(protocol: dict) -> str:
    return (f"standalone {'BF16' if bf16(protocol) else 'NF4'} F0 with newly seeded adapters, "
            "not minimal-trained weights")


def batches(actors: list, protocol: dict) -> list[list]:
    opts = options(protocol)
    size, count = opts["problems_per_update"], opts["train_problems"]
    if len(actors) != count or len({a.problem_id for a in actors}) != count:
        raise ValueError(f"local train: exactly {count} unique fixed actors required")
    return [actors[i:i + size] for i in range(0, count, size)]


# ---------------------------------------------------------------- policy, guards, parking

def build_bf16_policy(model_path: Path, arm: str, protocol: dict, *, device_index: int = 0):
    """Historical BF16 loader on one explicit CUDA device; no NF4 cache or quantization contract."""
    import torch

    from .policy import build_policy

    if not bf16(protocol) or type(device_index) is not int or device_index < 0:
        raise ValueError("scale: BF16 loader requires the bound bf16 protocol and a CUDA index")
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise ValueError("scale: actual CUDA required for the BF16 policy; no CPU fallback")
    bundle = build_policy(Path(model_path), arm, protocol)
    bundle.model.to(torch.device("cuda", device_index))
    return bundle


_CHECKSUM_CHUNK = 1 << 22  # elements per float64 temporary (32 MiB), never a full-size float64 copy


def _checksum(bundle) -> list:
    """(name, float64 sum, float64 L2 norm) per frozen tensor, converted chunk by chunk; deterministic."""
    import torch

    from .policy import _canonical_frozen

    def digest(tensor) -> tuple[float, float]:
        flat, total, squares = tensor.detach().reshape(-1), 0., 0.
        for start in range(0, flat.numel(), _CHECKSUM_CHUNK):
            chunk = flat[start:start + _CHECKSUM_CHUNK].to(torch.float64)
            total += float(chunk.sum())
            squares += float(torch.linalg.vector_norm(chunk)) ** 2
        return total, math.sqrt(squares)

    with torch.no_grad():
        return [(name, *digest(p)) for name, p in _canonical_frozen(bundle)]


def base_guard(bundle, protocol: dict) -> Callable[[Any], None]:
    """NF4: the full quantization guard. BF16: a cheap device checksum per update; the full
    SHA check (assert_base_unchanged) runs inside every checkpoint.save."""
    if not bf16(protocol):
        from . import quantized_policy

        return lambda value: quantized_policy.assert_quantized_base_unchanged(value)
    from .policy import assert_base_unchanged

    assert_base_unchanged(bundle)
    expected = _checksum(bundle)

    def guard(value) -> None:
        if value is not bundle or _checksum(value) != expected:
            raise ValueError("scale: frozen BF16 base checksum changed")
    return guard


def parks(protocol: dict) -> bool:
    opts = options(protocol)
    return opts["park_adam_moments"] and opts["policy_precision"] == "nf4"


def parking_policy(protocol: dict) -> str:
    from .checkpoint import OPTIMIZER_PARKING_POLICY

    return OPTIMIZER_PARKING_POLICY if parks(protocol) else RESIDENT_POLICY


@contextmanager
def _resident() -> Iterator[Callable[[], None]]:
    yield lambda: None


def park(bundle, optimizer, scheduler, protocol: dict, step: int):
    from . import checkpoint

    return checkpoint.park_adam_moments(bundle, optimizer, scheduler, protocol, step) if parks(protocol) else _resident()


# ---------------------------------------------------------------- engines

def shared_engine(bundle, protocol: dict, *, policy_version: int = 0, context_limit: int | None = None):
    """The live HF shared-policy engine; row-seeded batching only when the protocol binds it."""
    from . import shared_policy_engine

    rows = options(protocol)["rollout_batch_rows"]
    return shared_policy_engine.SharedPolicyRolloutEngine(bundle, protocol["image"],
        protocol["context_limit"] if context_limit is None else context_limit,
        logprob_chunk=protocol["model"]["logprob_chunk"], policy_version=policy_version,
        **({"batch_rows": rows} if rows is not None else {}))


def make_engines(bundle, protocol: dict, model_path: Path, *, policy_version: int = 0,
                 context_limit: int | None = None) -> tuple[Any, Any]:
    """(sampler, preparation). HF: one live shared-policy engine for both. vLLM: a merged-weight
    vLLM sampler plus the live HF engine for inputs, update forwards and saved probes."""
    opts = options(protocol)
    limit = protocol["context_limit"] if context_limit is None else context_limit
    shared = shared_engine(bundle, protocol, policy_version=policy_version, context_limit=limit)
    if opts["rollout_engine"] == "hf":
        return shared, shared
    from .engines import VLLMRolloutEngine
    from .policy import merged_named_tensors

    budget = opts["vllm"]
    sampler = VLLMRolloutEngine(Path(model_path), protocol["image"], budget["kv_cache_memory_bytes"],
        budget["max_num_seqs"], limit, enable_prefix_caching=budget["enable_prefix_caching"],
        gpu_memory_utilization=budget["gpu_memory_utilization"])
    if policy_version:
        sampler.initialize_weights(merged_named_tensors(bundle), policy_version)
    return sampler, shared


def synchronize(bundle, sampler, preparation, probes, protocol: dict, step: int, *, push: bool,
                image_counts: tuple[int, ...] = (1, 2)) -> tuple[list, list | None]:
    """HF reports always; a separate vLLM report (merged-weight push, then probes) when it samples.

    `image_counts`: the preparation's bound probe coverage, forwarded only when it is not the default pair."""
    from . import trainer

    counts = {} if image_counts == (1, 2) else {"image_counts": image_counts}
    reports = trainer.synchronize(bundle, preparation, preparation, probes, protocol, step, push=push, **counts)
    if sampler is preparation:
        return reports, None
    # After the HF report (0.02) certifies the live policy: score the policy's own continuation against the
    # bound vLLM-only calibrated threshold (0.04, user decision 2026-10-31).
    sampled = policy_sampled_probes(preparation, probes)
    # Evidence only: the sampler is still on the previous version here; the post-push gate is unchanged by it.
    before = pre_push_discrimination(bundle, sampler, preparation, sampled, protocol) if push else {}
    rows = trainer.synchronize(bundle, preparation, sampler, sampled, protocol, step, push=push,
                               threshold=sampler_sync_threshold(protocol), **counts)
    ids = {probe.request.request_id: probe.completion_ids for probe in sampled}
    return reports, [row | {"probe_source": VLLM_SYNC_PROBE, "completion_ids": ids[row["request_id"]]}
                     | before.get(row["request_id"], {}) for row in rows]


def pre_push_discrimination(bundle, sampler, preparation, sampled: list, protocol: dict) -> dict[str, dict]:
    """Per request: mean |old sampler - current HF policy| logprob over the sampled probe tokens, measured BEFORE the push.

    Proves the post-push probe can tell old from new weights. Never raises and never gates: a failed
    measurement is recorded as `pre_push_measured: false` with its reason."""
    import torch

    from .policy import completion_logprobs

    try:
        out = {}
        device = next(bundle.model.parameters()).device
        with torch.no_grad():  # same reference as trainer.synchronize: completion_logprobs on model_inputs
            for probe in sampled:
                old = sampler.probe_logprobs(probe.request, probe.completion_ids)
                inputs = preparation.model_inputs(probe.request, probe.completion_ids, device)
                ref = completion_logprobs(bundle, inputs, len(probe.completion_ids),
                                          chunk=protocol["model"]["logprob_chunk"])[0].cpu().tolist()
                if len(old) != len(ref) or not ref:
                    raise RuntimeError("old sampler and HF probe lengths differ")
                delta = sum(abs(a - b) for a, b in zip(old, ref, strict=True)) / len(ref)
                if not math.isfinite(delta):
                    raise RuntimeError("nonfinite pre-push delta")
                out[probe.request.request_id] = {"pre_push_measured": True, "pre_push_mean_abs_delta": delta}
        return out
    except Exception as error:  # evidence must not change pass/fail
        reason = f"{type(error).__name__}: {error}"[:200]
        return {probe.request.request_id: {"pre_push_measured": False, "pre_push_reason": reason} for probe in sampled}


def policy_sampled_probes(preparation, probes: list) -> list:
    """Each probe request's 32-token continuation sampled by the current HF policy with the probe's own seed.

    Global torch CPU/CUDA RNG states are restored, so training sampling is unaffected. An immediate EOS
    leaves nothing to compare and is refused rather than passed."""
    import dataclasses

    import torch

    from .trainer import SyncProbe

    sampled = []
    with torch.random.fork_rng(devices=range(torch.cuda.device_count()) if torch.cuda.is_initialized() else []):
        for probe in probes:
            request = dataclasses.replace(probe.request, max_new_tokens=VLLM_SYNC_PROBE_TOKENS)
            [result] = preparation.sample([request])
            [completion] = result.completions
            ids = list(completion.token_ids)
            if len(ids) - (completion.finish_reason == "stop") < 1:
                raise RuntimeError("scale: HF policy ended a vLLM sync probe at once; no continuation to compare")
            sampled.append(SyncProbe(request, ids))
    return sampled


def sampler_label(protocol: dict) -> dict:
    from .engines import ROW_SEEDED_BATCH_SAMPLER

    opts = options(protocol)
    if opts["rollout_engine"] == "vllm":
        return {"engine": "vllm", "sampler_version": "vllm-0.11.0-per-seed-v1", "batch_rows": None,
                "sync_probe": VLLM_SYNC_PROBE}
    rows = opts["rollout_batch_rows"]
    return {"engine": "hf", "sampler_version": ROW_SEEDED_BATCH_SAMPLER if rows else "hf-serial-v1", "batch_rows": rows}


# ---------------------------------------------------------------- measured throughput receipts

class TokenMeter:
    """Counts every completion an engine returns (rollouts, notes, probes); measurement only."""

    def __init__(self, *engines: Any):
        self.tokens = self.completions = self.calls = 0
        seen: set[int] = set()
        for engine in engines:
            original = getattr(engine, "sample_with_settings", None)
            if original is None or id(engine) in seen:
                continue
            seen.add(id(engine))
            engine.sample_with_settings = self._wrap(original)

    def _wrap(self, original: Callable) -> Callable:
        def sample_with_settings(*args, **kwargs):
            results = original(*args, **kwargs)
            self.calls += 1
            self.completions += sum(len(result.completions) for result in results)
            self.tokens += sum(len(c.token_ids) for result in results for c in result.completions)
            return results
        return sample_with_settings


def peak_memory() -> tuple[float, str]:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # A separate vLLM engine process holds memory this process's allocator never sees: take the larger of the
        # process peak and the whole-device use at receipt time (pods have a dedicated GPU).
        free, total = torch.cuda.mem_get_info()
        process, device = torch.cuda.max_memory_reserved() / 2**30, (total - free) / 2**30
        return (process, "cuda_max_memory_reserved") if process >= device else (device, "cuda_device_used_at_receipt")
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "process_max_rss"


def write_receipt(output: Path, *, items: int, completion_tokens: int, wall_seconds: float, updates: int,
                  update_seconds: float, n_samples: int, retries_per_item: float, startup_seconds: float,
                  extra: dict | None = None) -> dict:
    """`throughput-receipt-v1` (fine_tuning/notebook_rl/pod/cost.py) from measured values only."""
    from .pod.cost import RECEIPT_VERSION

    values = (items, completion_tokens, wall_seconds, updates, update_seconds, n_samples, retries_per_item,
              startup_seconds)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("throughput receipt: finite nonnegative measured values required")
    peak, source = peak_memory()
    receipt = {"version": RECEIPT_VERSION, "label": "measured", "items": int(items),
        "completion_tokens": int(completion_tokens), "wall_seconds": round(float(wall_seconds), 6),
        "updates": int(updates), "update_seconds": round(float(update_seconds), 6),
        "startup_seconds": round(float(startup_seconds), 6), "peak_memory_gib": round(peak, 6),
        "peak_memory_source": source, "n_samples": n_samples, "retries_per_item": round(float(retries_per_item), 6),
        "measured_utc_epoch": round(time.time(), 3), **(extra or {})}
    path = Path(output) / RECEIPT_NAME
    _atomic_json(path, receipt)
    # pod/start.sh runs one command per job in MMDL_RUN_DIR; pod.collect and pod.cost read the run-dir root.
    run_dir = os.environ.get("MMDL_RUN_DIR")
    if run_dir and path.parent.resolve() != Path(run_dir).resolve() and path.resolve().is_relative_to(Path(run_dir).resolve()):
        _atomic_json(Path(run_dir) / RECEIPT_NAME, receipt | {"mirrored_from": str(path.resolve())})
    return receipt


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# ---------------------------------------------------------------- CLI

_SIMPLE = ("rollout_batch_rows", "problems_per_update", "checkpoint_every_updates", "train_problems",
           "max_updates", "evidence_audit_rule", "policy_precision", "rollout_engine")


def add_arguments(parser) -> None:
    group = parser.add_argument_group("opt-in scale options (bound into the protocol hash only when given)")
    for name in ("rollout-batch-rows", "problems-per-update", "checkpoint-every-updates", "train-problems",
                 "max-updates", "vllm-max-num-seqs", "vllm-kv-cache-memory-bytes"):
        group.add_argument("--" + name, type=int)
    group.add_argument("--evidence-audit-rule", type=json.loads,
                       help='JSON, e.g. {"first": true, "last": true, "every": 5}')
    group.add_argument("--policy-precision", choices=("nf4", "bf16"))
    group.add_argument("--rollout-engine", choices=("hf", "vllm"))
    group.add_argument("--vllm-gpu-memory-utilization", type=float)
    group.add_argument("--vllm-enable-prefix-caching", action="store_true")
    group.add_argument("--no-park-adam-moments", action="store_true")


def from_args(args) -> dict[str, Any]:
    """build_protocol keywords: {} unless an option was given, so default calls stay byte-identical."""
    chosen = {key: getattr(args, key) for key in _SIMPLE if getattr(args, key, None) is not None}
    vllm = {key: getattr(args, "vllm_" + key) for key in ("gpu_memory_utilization", "max_num_seqs",
                                                         "kv_cache_memory_bytes")
            if getattr(args, "vllm_" + key, None) is not None}
    if vllm or getattr(args, "vllm_enable_prefix_caching", False):
        chosen["vllm"] = vllm | {"enable_prefix_caching": bool(getattr(args, "vllm_enable_prefix_caching", False))}
    if getattr(args, "no_park_adam_moments", False):
        chosen["park_adam_moments"] = False
    return {"scale": chosen} if chosen else {}
