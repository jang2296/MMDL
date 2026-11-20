"""Opt-in admission from pinned CUDA measurements; estimates never become measured capacity.

Capture wraps existing generation/backward, changes no parameters, and creates no extra sampled answers.
Only observed generation lengths are covered; a short completion does not calibrate its unused output cap.
Training entries use the actual packed group's geometry and a common resident-state baseline. Running peaks
can include earlier replay/work in that transaction; they are conservative bounds, not isolated phase costs.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from uuid import UUID

from fine_tuning.notebook_rl.input_preparation import digest

VERSION = "single-hf-measured-capacity-v1"
PHASES = {"generation": "LM", "rl_kl": "LM", "sft_lm": "LM", "sft_full": "FULL"}
DIMENSIONS = ("rows", "prompt_tokens", "total_tokens", "target_tokens", "action_tokens", "images", "image_cells", "max_image_cells")


class CapacityHold(ValueError):
    """No measured admission; never retry a started generation at a smaller capacity."""


def shape(rows: list[dict], *, generation: bool = False) -> dict:
    if not rows or len(rows) > 16 or len({row["problem_id"] for row in rows}) != len(rows):
        raise CapacityHold("capacity HOLD: 1..16 different questions required")
    if any(type(row["prompt_tokens"]) is not int or row["prompt_tokens"] < 1
           or type(row["target_tokens"]) is not int or row["target_tokens"] < 1
           or not isinstance(row["image_grids"], list)
           or any(not isinstance(grid, list) or len(grid) != 3 or any(type(v) is not int or v < 1 for v in grid)
                  for grid in row["image_grids"]) for row in rows):
        raise CapacityHold("capacity HOLD: actual integer token lengths and image grids required")
    prompts = [row["prompt_tokens"] for row in rows]
    targets = [row["target_tokens"] for row in rows]
    cells = [math.prod(grid) for row in rows for grid in row["image_grids"]]
    return {"rows": len(rows), "prompt_tokens": max(prompts), "target_tokens": max(targets),
            "total_tokens": max(prompts) + max(targets) if generation else max(p + t for p, t in zip(prompts, targets, strict=True)),
            "action_tokens": sum(targets), "images": len(cells), "image_cells": sum(cells), "max_image_cells": max(cells, default=0),
            "row_layout": [{key: row[key] for key in ("prompt_tokens", "target_tokens", "image_grids")} for row in rows]}


def encoded_shapes(rows: list[dict]) -> list[dict]:
    return [{"problem_id": row["problem_id"], "prompt_tokens": row["input_ids"].numel() - row["target_len"],
             "target_tokens": row["target_len"], "image_grids": row["image_grid_thw"].tolist()} for row in rows]


def _layout_covers(actual: list[dict], measured: list[dict]) -> bool:
    """Inject each real row into a measured row; maxima cannot bound retained per-row projection graphs."""
    matched: dict[int, int] = {}

    def fit(row: dict, limit: dict) -> bool:
        return (row["prompt_tokens"] <= limit["prompt_tokens"] and row["target_tokens"] <= limit["target_tokens"]
                and len(row["image_grids"]) <= len(limit["image_grids"])
                and all(all(a <= b for a, b in zip(grid, bound, strict=True))
                        for grid, bound in zip(row["image_grids"], limit["image_grids"])))

    def assign(index: int, seen: set[int]) -> bool:
        for candidate, limit in enumerate(measured):
            if candidate in seen or not fit(actual[index], limit):
                continue
            seen.add(candidate)
            if candidate not in matched or assign(matched[candidate], seen):
                matched[candidate] = index
                return True
        return False

    return all(assign(index, set()) for index in range(len(actual)))


def load_profile(path: Path, expected_sha256: str, *, allow_partial: bool = False) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise CapacityHold("capacity HOLD: profile SHA256 changed")
    profile = json.loads(raw)
    validate_profile(profile, allow_partial=allow_partial)
    return profile


def _publish(path: Path, profile: dict) -> None:
    from .collect import write_json
    from .single import _runtime_lock

    validate_profile(profile, allow_partial=True)
    with _runtime_lock(path.parent, "." + path.name + ".publish.lock"):
        if path.exists():
            raise CapacityHold("capacity HOLD: profile output already exists")
        write_json(path, profile)


def assemble(paths: list[Path], expected_sha256: list[str], output: Path) -> dict:
    """Join externally pinned, disjoint actual captures; missing evidence stays uncalibrated."""
    if (not paths or len(paths) != len(expected_sha256) or len(set(expected_sha256)) != len(paths)
            or len({path.resolve() for path in paths}) != len(paths)):
        raise CapacityHold("capacity HOLD: distinct partial paths with one external SHA256 each required")
    partials = [load_profile(path, pin, allow_partial=True) for path, pin in zip(paths, expected_sha256, strict=True)]
    if any(partial["status"] == "UNCALIBRATED_FAILED" for partial in partials):
        raise CapacityHold("capacity HOLD: failed capture cannot certify or assemble capacity")
    profile = partials[0] | {"entries": [entry for partial in partials for entry in partial["entries"]], "status": "UNCALIBRATED"}
    if any(any(partial[key] != profile[key] for key in ("binding", "reserve_bytes", "calibration_policy_sha256")) for partial in partials):
        raise CapacityHold("capacity HOLD: partial runtime/source/model/policy/reserve bindings differ")
    validate_profile(profile, allow_partial=True)  # Overlap or invalid facts must fail, not become a partial profile.
    profile["status"] = "MEASURED"
    try:
        validate_profile(profile)
    except CapacityHold:
        profile["status"] = "UNCALIBRATED"
    _publish(output, profile)
    return profile


def validate_profile(profile: dict, *, allow_partial: bool = False) -> None:
    if (not isinstance(profile, dict) or set(profile) != {"version", "status", "binding", "reserve_bytes", "entries", "calibration_policy_sha256"}
            or profile["version"] != VERSION or profile["status"] not in (("MEASURED", "UNCALIBRATED", "UNCALIBRATED_FAILED") if allow_partial else ("MEASURED",))
            or re.fullmatch(r"[0-9a-f]{64}", str(profile.get("calibration_policy_sha256"))) is None
            or type(profile["reserve_bytes"]) is not int or profile["reserve_bytes"] <= 0
            or not isinstance(profile["binding"], dict) or not isinstance(profile["entries"], list)):
        raise CapacityHold("capacity HOLD: a completed measured profile is required")
    fixed = profile["binding"]
    if (set(fixed) != {"source_sha256", "model_sha256", "precision", "layout_sha256", "config_sha256", "device",
                       "total_bytes", "versions", "attention", "compute_dtype", "cache", "checkpointing", "state_bytes", "execution"}
            or any(re.fullmatch(r"[0-9a-f]{64}", str(fixed.get(key))) is None
                   for key in ("source_sha256", "model_sha256", "layout_sha256", "config_sha256"))
            or fixed["precision"] not in ("nf4", "bf16") or not isinstance(fixed["device"], str) or not fixed["device"]
            or type(fixed["total_bytes"]) is not int or fixed["total_bytes"] <= 0 or not isinstance(fixed["versions"], dict)
            or not isinstance(fixed["attention"], str) or fixed["compute_dtype"] != "bfloat16" or fixed["cache"] != "dynamic-bfloat16"
            or type(fixed["checkpointing"]) is not bool or not isinstance(fixed["state_bytes"], dict)
            or not isinstance(fixed["execution"], dict)
            or set(fixed["execution"]) != set(_execution())
            or set(fixed["state_bytes"]) != set(PHASES)
            or any(type(value) is not int or value <= 0 for value in fixed["state_bytes"].values())
            or any(fixed["state_bytes"][key] * 2 != fixed["state_bytes"]["generation"] * 3 for key in PHASES if key != "generation")):
        raise CapacityHold("capacity HOLD: incomplete fixed runtime/state binding")
    phases, kinds, work_seen = set(), set(), set()
    pg = False
    for entry in profile["entries"]:
        if (not isinstance(entry, dict) or entry.get("phase") not in PHASES
                or entry.get("route") != PHASES[entry["phase"]]
                or set(entry.get("shape", {})) != {*DIMENSIONS, "row_layout"}
                or any(type(entry["shape"][key]) is not int or entry["shape"][key] < 0 for key in DIMENSIONS)
                or not 1 <= entry["shape"]["rows"] <= 16
                or entry["shape"]["prompt_tokens"] < 1 or entry["shape"]["target_tokens"] < 1
                or entry["shape"]["total_tokens"] < entry["shape"]["prompt_tokens"]
                or type(entry.get("extra_bytes")) is not int or entry["extra_bytes"] < 0
                or entry.get("resident_state_bytes") != profile["binding"].get("state_bytes", {}).get(entry["phase"])
                or type(entry.get("resident_state_bytes")) is not int or entry["resident_state_bytes"] < 0
                or not isinstance(entry.get("seconds"), (int, float)) or not math.isfinite(entry["seconds"]) or entry["seconds"] <= 0
                or entry.get("measurement") != "cuda_allocator_actual_work"
                or entry.get("kind") not in (("GENERATION",) if entry["phase"] == "generation" else ("RL",) if entry["phase"] == "rl_kl" else ("A1", "A2", "B"))
                or type(entry.get("pg_rows")) is not int or not 0 <= entry["pg_rows"] <= entry["shape"]["rows"]
                or not isinstance(entry.get("work_ids"), list) or len(entry["work_ids"]) != entry["shape"]["rows"]
                or any(not isinstance(value, str) or not value for value in entry["work_ids"])
                or len(set(entry["work_ids"])) != len(entry["work_ids"])
                or not isinstance(entry.get("baseline"), dict)
                or any(type(entry["baseline"].get(key)) is not int or entry["baseline"][key] < 0
                       for key in ("free", "total", "allocated", "reserved"))
                or entry["baseline"]["total"] != profile["binding"].get("total_bytes")
                or entry["baseline"]["allocated"] < entry["resident_state_bytes"]
                or not 0 <= entry["baseline"]["free"] <= entry["baseline"]["total"]
                or entry["baseline"]["allocated"] > entry["baseline"]["reserved"]
                or entry["shape"]["total_tokens"] < entry["shape"]["target_tokens"]
                or type(entry.get("peak_allocated_bytes")) is not int or type(entry.get("peak_reserved_bytes")) is not int
                or entry["peak_allocated_bytes"] < entry["baseline"]["allocated"]
                or entry["peak_reserved_bytes"] < entry["baseline"]["reserved"]
                or entry["peak_reserved_bytes"] > entry["baseline"]["total"]
                or entry["peak_allocated_bytes"] > entry["peak_reserved_bytes"]
                or type(entry.get("minimum_moment_bytes")) is not int
                or entry["minimum_moment_bytes"] not in (0, fixed["state_bytes"]["generation"])
                or type(entry.get("state_baseline_allocated_bytes")) is not int
                or not 0 <= entry["state_baseline_allocated_bytes"] <= entry["baseline"]["allocated"]
                or entry.get("state_overhead_bytes") != max(0, entry["baseline"]["allocated"] - entry["state_baseline_allocated_bytes"]
                    - (entry["resident_state_bytes"] - entry["minimum_moment_bytes"]))
                or entry["extra_bytes"] != max(entry["peak_allocated_bytes"] - entry["baseline"]["allocated"],
                    entry["peak_reserved_bytes"] - entry["baseline"]["reserved"], 0) + entry["state_overhead_bytes"]
                or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("workload_sha256"))) is None
                or entry["phase"] == "rl_kl" and entry.get("full_kl") is not True):
            raise CapacityHold("capacity HOLD: invalid measurement or missing full-KL/state coverage")
        layout = entry["shape"]["row_layout"]
        if (not isinstance(layout, list) or any(not isinstance(row, dict) or set(row) != {"prompt_tokens", "target_tokens", "image_grids"}
                                              for row in layout)
                or shape([row | {"problem_id": str(index)} for index, row in enumerate(layout)],
                         generation=entry["phase"] == "generation") != entry["shape"]):
            raise CapacityHold("capacity HOLD: measured per-row token/image layout is inconsistent")
        phases.add(entry["phase"])
        kinds.add(entry["kind"])
        pg |= entry["phase"] == "rl_kl" and entry["pg_rows"] > 0
        for work_id in entry["work_ids"]:
            key = (entry["phase"], entry["kind"], work_id)
            if key in work_seen:
                raise CapacityHold("capacity HOLD: overlapping measurement work cannot expand calibration evidence")
            work_seen.add(key)
    if (not allow_partial or profile["status"] == "MEASURED") and (phases != set(PHASES) or not {"A1", "A2", "B"} <= kinds or not pg):
        raise CapacityHold("capacity HOLD: uncalibrated generation/full-KL/PG/A1/A2/B/LM/FULL coverage")


def _execution() -> dict:
    import torch

    return {"deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "cuda_runtime": torch.version.cuda,
            "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(), "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
            "sdpa_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(), "sdpa_cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "sdpa_math_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
            "matmul_precision": torch.get_float32_matmul_precision(), "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_accumulation": getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None)}


def _layout(bundle: Any) -> dict:
    owned = {id(p): group for group, parameters in bundle.params.items() for p in parameters}
    return {"parameters": {name: {"shape": list(p.shape), "dtype": str(p.dtype), "group": owned.get(id(p), "FROZEN")}
                           for name, p in bundle.model.named_parameters()},
            "buffers": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
                        for name, value in bundle.model.named_buffers()}}


def _physical_device(device: Any) -> tuple[str, int]:
    import torch

    properties = torch.cuda.get_device_properties(device)
    try:
        # PyTorch 2.8 exposes _CUuuid; its string is bare UUID, while nvidia-smi expects GPU-UUID.
        actual_uuid = str(UUID(str(properties.uuid).removeprefix("GPU-")))
        canonical = "GPU-" + actual_uuid
        physical = subprocess.check_output(["nvidia-smi", f"--id={canonical}",
            "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        observed, driver = (value.strip() for value in physical.split(",", 1))
        if str(UUID(observed.removeprefix("GPU-"))) != actual_uuid or not driver:
            raise ValueError("UUID/driver mismatch")
    except (AttributeError, ValueError, subprocess.SubprocessError) as error:
        raise CapacityHold("capacity HOLD: actual CUDA/NVML UUID identity unavailable or mismatched") from error
    return f"{canonical}, {driver}", properties.total_memory


def binding(bundle: Any, identity: dict) -> dict:
    """Actual hardware/backend and state layout, independent of changing adapter values or routing flags."""
    import torch

    if (set(identity) != {"source_sha256", "model_sha256", "precision"} or identity["precision"] not in ("nf4", "bf16")
            or any(re.fullmatch(r"[0-9a-f]{64}", str(identity[key])) is None for key in ("source_sha256", "model_sha256"))):
        raise CapacityHold("capacity HOLD: fixed source/model/precision identity required")
    device = next(bundle.model.parameters()).device
    if device.type != "cuda":
        raise CapacityHold("capacity HOLD: no CUDA calibration for this runtime")
    base = bundle.model.get_base_model()
    owned = {id(p): group for group, parameters in bundle.params.items() for p in parameters}
    found_owned = {id(p) for p in bundle.model.parameters() if id(p) in owned}
    if (len(found_owned) != len(owned) or any(p.dtype != torch.float32 for ps in bundle.params.values() for p in ps)
            or any(p.device != device or p.requires_grad and id(p) not in owned for p in bundle.model.parameters())):
        raise CapacityHold("capacity HOLD: unsupported adapter/gradient/AdamW state layout")
    parameter_bytes = sum(p.numel() * p.element_size() for ps in bundle.params.values() for p in ps)
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    physical, total = _physical_device(device)
    return identity | {"layout_sha256": digest(_layout(bundle)), "config_sha256": digest(base.config.to_dict()),
                       "device": physical, "total_bytes": total,
                       "versions": versions, "attention": base.config._attn_implementation,
                       "compute_dtype": "bfloat16", "cache": "dynamic-bfloat16", "checkpointing": bool(base.is_gradient_checkpointing),
                       "execution": _execution(),
                       "state_bytes": {phase: parameter_bytes * (2 if phase == "generation" else 3) for phase in PHASES}}


def memory(bundle: Any) -> dict:
    import torch

    device = next(bundle.model.parameters()).device
    if device.type != "cuda":
        raise CapacityHold("capacity HOLD: actual CUDA memory readings required")
    free, total = torch.cuda.mem_get_info(device)
    result = {"free": free, "total": total, "allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device)}
    if not 0 <= free <= total or not 0 <= result["allocated"] <= result["reserved"] <= total:
        raise CapacityHold("capacity HOLD: invalid or physically oversubscribed memory readings")
    return result


def _live_state_bytes(bundle: Any, optimizer: Any = None, *, gradients: bool) -> int:
    total = 0
    for parameters in bundle.params.values():
        for parameter in parameters:
            if gradients and parameter.grad is not None:
                total += parameter.grad.numel() * parameter.grad.element_size()
            if optimizer is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    value = optimizer.state.get(parameter, {}).get(key)
                    if value is not None and value.device == parameter.device:
                        if value.shape != parameter.shape or value.dtype != parameter.dtype:
                            raise CapacityHold("capacity HOLD: unsupported live Adam moment layout")
                        total += value.numel() * value.element_size()
    return total


class Capacity:
    def __init__(self, bundle: Any, profile: dict, identity: dict, *, optimizer: Any = None):
        validate_profile(profile)
        self.bundle, self.profile, self.identity, self.optimizer = bundle, json.loads(json.dumps(profile)), identity, optimizer
        self._check_binding()

    def _check_binding(self) -> None:
        if binding(self.bundle, self.identity) != self.profile["binding"]:
            raise CapacityHold("capacity HOLD: measured hardware/model/source/precision/backend/layout changed")

    def check(self, rows: list[dict], phase: str) -> dict:
        self._check_binding()
        actual = shape(rows, generation=phase == "generation")
        moments = _live_state_bytes(self.bundle, self.optimizer, gradients=False)
        candidates = [entry for entry in self.profile["entries"] if entry["phase"] == phase and entry["minimum_moment_bytes"] <= moments
                      and all(actual[key] <= entry["shape"][key] for key in DIMENSIONS)
                      and _layout_covers(actual["row_layout"], entry["shape"]["row_layout"])]
        if not candidates:
            raise CapacityHold("capacity HOLD: unsupported measured shape/route (including a long singleton)")
        current = memory(self.bundle)
        if current["total"] != self.profile["binding"]["total_bytes"]:
            raise CapacityHold("capacity HOLD: device capacity changed")
        live = _live_state_bytes(self.bundle, self.optimizer, gradients=phase != "generation")
        available = current["free"] + max(0, current["reserved"] - current["allocated"])
        entry = min(candidates, key=lambda item: item["extra_bytes"])
        required = entry["extra_bytes"] + max(0, entry["resident_state_bytes"] - live) + self.profile["reserve_bytes"]
        if required > available:
            raise CapacityHold("capacity HOLD: measured work and resident state exceed available memory")
        return {"phase": phase, "shape": actual, "required_bytes": required, "available_bytes": available}

    def partition(self, rows: list[dict], plan: list[list[int]], phase: str) -> list[list[int]]:
        """Split before execution; preserve the supplied FIFO/route order and every original row."""
        result = []
        for indices in plan:
            current: list[int] = []
            for index in indices:
                try:
                    self.check([rows[i] for i in [*current, index]], phase)
                except CapacityHold:
                    if not current:
                        raise
                    result.append(current)
                    current = []
                    self.check([rows[index]], phase)
                current.append(index)
            if current:
                result.append(current)
        return result


class Capture:
    """Explicit work capture; no retries/updates. Optimizer-step workspace is outside this profile's scope.

    Attach to stream_args.capacity_capture to record existing, once-only generation. Run capture_backward on
    accepted rows for the full-KL and both SFT routes, then write a fresh profile path. CPU/uncovered work cannot
    produce an admitted measurement. Short generations cover their observed length, never an unused token cap.
    """
    def __init__(self, bundle: Any, identity: dict, reserve_bytes: int, *, optimizer: Any = None):
        if type(reserve_bytes) is not int or reserve_bytes <= 0:
            raise CapacityHold("capacity HOLD: explicit positive reserve required")
        self.bundle, self.identity, self.reserve_bytes, self.optimizer = bundle, identity, reserve_bytes, optimizer
        self.binding = binding(bundle, identity)  # CPU invocations cannot manufacture a CUDA measurement.
        from .rl import policy_sha256

        self.policy_sha256 = policy_sha256(bundle)
        self.entries: list[dict] = []
        self.failed = False
        self._training_state: dict | None = None

    @contextmanager
    def _resident_state(self, phase: str):
        import torch
        from .rl import policy_sha256

        if phase not in PHASES or any(p.grad is not None for ps in self.bundle.params.values() for p in ps):
            raise CapacityHold("capacity HOLD: known measurement phase and cleared incoming gradients required")
        if binding(self.bundle, self.identity) != self.binding:
            raise CapacityHold("capacity HOLD: calibration binding changed")
        before = policy_sha256(self.bundle)
        if before != self.policy_sha256:
            raise CapacityHold("capacity HOLD: calibration policy changed between measurements")
        from fine_tuning.notebook_rl.policy import assert_base_unchanged
        from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged

        check_base = assert_quantized_base_unchanged if self.identity["precision"] == "nf4" else assert_base_unchanged
        check_base(self.bundle)
        device = next(self.bundle.model.parameters()).device
        moments = _live_state_bytes(self.bundle, self.optimizer, gradients=False)
        if moments not in (0, self.binding["state_bytes"]["generation"]):
            raise CapacityHold("capacity HOLD: capture requires cold or fully resident Adam moments; partial layout is unmeasured")
        reservation = None
        try:
            state_baseline = memory(self.bundle)["allocated"]
            reservation = [torch.empty_like(parameter) for parameters in self.bundle.params.values() for parameter in parameters
                           for key in ("exp_avg", "exp_avg_sq") if self.optimizer is None
                           or self.optimizer.state.get(parameter, {}).get(key) is None
                           or self.optimizer.state[parameter][key].device != parameter.device]
            if phase != "generation":
                for parameters in self.bundle.params.values():
                    for parameter in parameters:
                        parameter.grad = torch.zeros_like(parameter)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            baseline = memory(self.bundle)
            overhead = max(0, baseline["allocated"] - state_baseline - (self.binding["state_bytes"][phase] - moments))
            torch.cuda.reset_peak_memory_stats(device)
            yield {"baseline": baseline, "overhead": overhead, "moments": moments, "state_baseline": state_baseline,
                   "check_base": check_base}
        finally:
            self.bundle.model.zero_grad(set_to_none=True)
            del reservation
            check_base(self.bundle)

    @contextmanager
    def training_state(self):
        """Reserve once; native backward accumulates into the same full gradient buffers across batches."""
        if self._training_state is not None:
            raise CapacityHold("capacity HOLD: nested training capture")
        try:
            with self._resident_state("rl_kl") as state:
                self._training_state = state
                yield
        except BaseException:
            self.failed = True
            raise
        finally:
            self._training_state = None

    @contextmanager
    def measure(self, phase: str, dimensions: dict, workload_sha256: str):
        import torch
        from .rl import policy_sha256

        if phase not in PHASES or phase != "generation" and self._training_state is None:
            raise CapacityHold("capacity HOLD: known phase and one resident training transaction required")
        context = self._resident_state(phase) if phase == "generation" else nullcontext(self._training_state)
        sample = {"complete": False, "full_kl": False, "shape": dimensions, "pg_rows": 0,
                  "kind": "GENERATION" if phase == "generation" else "RL" if phase == "rl_kl" else None, "work_ids": []}
        started = time.monotonic()
        try:
            with context as state:
                if not isinstance(state, dict):
                    raise CapacityHold("capacity HOLD: missing resident-state baseline")
                if binding(self.bundle, self.identity) != self.binding or policy_sha256(self.bundle) != self.policy_sha256:
                    raise CapacityHold("capacity HOLD: calibration binding/policy changed")
                device = next(self.bundle.model.parameters()).device
                yield sample
                torch.cuda.synchronize(device)
                if not sample["complete"] or phase == "rl_kl" and sample["full_kl"] is not True:
                    raise CapacityHold("capacity HOLD: incomplete measurement or full KL not exercised")
                if policy_sha256(self.bundle) != self.policy_sha256:
                    raise CapacityHold("capacity HOLD: parameters changed during measurement")
                state["check_base"](self.bundle)
                baseline, overhead = state["baseline"], state["overhead"]
                # Running transaction peaks include earlier replay/work. This is a conservative envelope, not isolated phase cost.
                extra = max(torch.cuda.max_memory_allocated(device) - baseline["allocated"],
                            torch.cuda.max_memory_reserved(device) - baseline["reserved"], 0) + overhead
                self.entries.append({"phase": phase, "route": PHASES[phase], "shape": sample["shape"],
                    "extra_bytes": extra, "resident_state_bytes": self.binding["state_bytes"][phase],
                    "measurement": "cuda_allocator_actual_work", "workload_sha256": workload_sha256,
                    "kind": sample["kind"], "work_ids": sample["work_ids"], "pg_rows": sample["pg_rows"],
                    "minimum_moment_bytes": state["moments"], "state_baseline_allocated_bytes": state["state_baseline"], "state_overhead_bytes": overhead,
                    "seconds": time.monotonic() - started, "full_kl": sample["full_kl"],
                    "baseline": baseline, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        except BaseException:
            self.failed = True
            raise

    @contextmanager
    def batch(self, rows: list[dict], phase: str, kind: str, *, full_kl: bool = False):
        workload = [{"row_id": row["row_id"], "input_ids": row["input_ids"].tolist(),
                     "image_grids": row["image_grid_thw"].tolist()} for row in rows]
        with self.measure(phase, shape(encoded_shapes(rows)), digest(workload)) as sample:
            sample.update(kind=kind, work_ids=[row["row_id"] for row in rows],
                          pg_rows=sum(row["reward"] != 0 for row in rows) if kind == "RL" else 0)
            yield
            sample.update(complete=True, full_kl=full_kl)

    def write(self, path: Path) -> dict:
        profile = {"version": VERSION, "status": "UNCALIBRATED_FAILED" if getattr(self, "failed", False) else "MEASURED", "binding": self.binding,
                   "reserve_bytes": self.reserve_bytes, "entries": self.entries, "calibration_policy_sha256": self.policy_sha256}
        if profile["status"] == "MEASURED":
            try:
                validate_profile(profile)
            except CapacityHold:
                profile["status"] = "UNCALIBRATED"
        _publish(path, profile)
        return profile


def capture_backward(capture: Capture, encoded: list[dict], contract: dict, *, component: str = "RL",
                     token_budget: int = 0) -> dict:
    """Measure native packed backward with the original block/kind denominators; never subset the contract."""
    from . import train_v4 as training

    try:
        if component not in ("RL", *training.SFT_KINDS) or component == "RL" and contract["rl"]["beta"] <= 0:
            raise CapacityHold("capacity HOLD: explicit RL/full-KL or SFT component required")
        components = ("PG", "KL") if component == "RL" else (component,)
        with capture.training_state():
            return training.backward_single_block(capture.bundle, encoded, contract, token_budget,
                components=components, force_full_kl=True, capture=capture)
    except BaseException:
        capture.failed = True
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Assemble pinned actual SINGLE CUDA captures; never runs a model.")
    parser.add_argument("command", choices=("assemble",))
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--input-sha256", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = assemble(args.input, args.input_sha256, args.output)
    print(json.dumps({"status": profile["status"], "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "entries": len(profile["entries"]), "scope": "observed shapes only; no optimizer-step workspace calibration"}))
    return 0 if profile["status"] == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
