"""Write-once, fully bound RL checkpoints; restoration always starts from fresh F0."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from fine_tuning.notebook_rl.policy import PolicyBundle, assert_base_unchanged, build_policy

FORMAT = "notebook-rl-checkpoint-v2"
OPTIMIZER_PARKING_POLICY = "adamw-moments-cpu-during-collection-v1"
_FILES = {"adapter.safetensors", "mergers.safetensors", "runtime.pt"}
_BINDING = {"protocol_sha256", "prompts_sha256", "data_sha256", "code_sha256"}
PolicyFactory = Callable[[Path, str, dict], PolicyBundle]


def digest(value: Any) -> str:
    value = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _binding(protocol: dict, binding: dict) -> None:
    if (set(binding) != _BINDING or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                                      for v in binding.values())
            or binding["protocol_sha256"] != protocol["sha256"]
            or binding["prompts_sha256"] != protocol["prompts_sha256"]):
        raise ValueError("checkpoint: explicit protocol/prompts/data/code binding required")


def capture_rng() -> dict:
    """Safe-loadable representation; CUDA is never initialized merely to save CPU state."""
    state: Any = np.random.get_state()
    algorithm, keys, position, has_gauss, cached = state
    return {"python": random.getstate(), "numpy": {"algorithm": algorithm,
            "keys": torch.tensor(keys.astype(np.int64)), "position": position,
            "has_gauss": has_gauss, "cached_gaussian": cached},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(rng: dict) -> None:
    if set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("checkpoint: incomplete RNG state")
    numpy = rng["numpy"]
    if (set(numpy) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
            or numpy["algorithm"] != "MT19937" or numpy["keys"].shape != (624,)
            or numpy["keys"].dtype != torch.int64
            or not bool(((numpy["keys"] >= 0) & (numpy["keys"] < 2**32)).all())
            or not 0 <= numpy["position"] <= 624 or numpy["has_gauss"] not in (0, 1)
            or not math.isfinite(numpy["cached_gaussian"])
            or not isinstance(rng["cuda"], list)
            or any(not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or t.ndim != 1
                   for t in [rng["torch"], *rng["cuda"]])):
        raise ValueError("checkpoint: invalid RNG state")
    if rng["cuda"] and (not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count()):
        raise ValueError("checkpoint: CUDA RNG topology differs; no CPU fallback")
    if not rng["cuda"] and torch.cuda.is_initialized():
        raise ValueError("checkpoint: initialized CUDA state missing from CPU checkpoint")
    # Validate on private generators before changing any process-global state.
    random.Random().setstate(rng["python"])
    torch.Generator().set_state(rng["torch"])
    state = (numpy["algorithm"], numpy["keys"].numpy().astype(np.uint32), numpy["position"],
             numpy["has_gauss"], numpy["cached_gaussian"])
    np.random.RandomState().set_state(state)
    random.setstate(rng["python"])
    np.random.set_state(state)
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def make_optimizer(bundle: PolicyBundle, protocol: dict) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Stable L/M/V order, including route-OFF parameters, with no implicit LR scaling."""
    cfg = protocol["optimizer"]
    if cfg["type"] != "adamw":
        raise ValueError("checkpoint: unsupported optimizer/scheduler")
    multiplier = _schedule(cfg)
    groups = [{"params": bundle.params[g], "lr": cfg["lr"][g], "name": g}
              for g in ("L", "M", "V") if bundle.params[g]]
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg["betas"]), eps=cfg["eps"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def _schedule(cfg: dict) -> partial:
    """The LR multiplier of the protocol: constant_after_warmup, or linear_decay_after_warmup (error-note v2)."""
    warmup = cfg["warmup_steps"]
    if type(warmup) is not int or warmup < 0:
        raise ValueError("checkpoint: invalid warmup")
    if cfg["schedule"] == "constant_after_warmup":
        return partial(_warmup_multiplier, warmup=warmup)
    if cfg["schedule"] == "linear_decay_after_warmup":
        total = cfg["total_steps"]
        if type(total) is not int or total < 1:
            raise ValueError("checkpoint: invalid total_steps")
        return partial(_linear_decay_multiplier, warmup=warmup, total=total)
    raise ValueError("checkpoint: unsupported optimizer/scheduler")


def _warmup_multiplier(step: int, *, warmup: int) -> float:
    return min(1.0, (step + 1) / warmup) if warmup else 1.0


def _linear_decay_multiplier(step: int, *, warmup: int, total: int) -> float:
    """(step + 1) / warmup during the warmup, then linear from 1 at step `warmup` to 0 at step `total`, one past the
    last update (the last update keeps 1 / (total - warmup), so no update runs at LR 0). A run of total <= warmup
    steps ends inside the warmup."""
    fall = max(0.0, (total - step) / (total - warmup)) if total > warmup else 1.0
    return min(_warmup_multiplier(step, warmup=warmup), fall)


def _parameters(bundle: PolicyBundle) -> dict[str, torch.nn.Parameter]:
    actual = dict(bundle.model.named_parameters())
    rows = bundle.inventory["trainable"]
    names = [row["name"] for row in rows]
    installed = {n for n in actual if ".lora_" in n or ".modules_to_save.default." in n}
    if len(names) != len(set(names)) or set(names) != installed:
        raise ValueError("checkpoint: duplicate/incomplete installed parameter inventory")
    expected = {name: actual[name] for name in names}
    if ({id(p) for p in expected.values()} != {id(p) for ps in bundle.params.values() for p in ps}
            or len([r for r in rows if r["group"] == "M"]) != 24):
        raise ValueError("checkpoint: incomplete trainable partitions/four Merger masters")
    for row in rows:
        p = expected[row["name"]]
        if (list(p.shape) != row["shape"] or str(p.dtype) != row["dtype"]
                or not bool(torch.isfinite(p).all()) or (row["group"] == "M" and p.dtype != torch.float32)):
            raise ValueError("checkpoint: invalid trainable shape/dtype/value")
    return expected


def _optimizer_names(bundle: PolicyBundle, optimizer: torch.optim.Optimizer) -> list[list[str]]:
    parameters = _parameters(bundle)
    names = {id(p): name for name, p in parameters.items()}
    try:
        groups = [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]
    except KeyError as exc:
        raise ValueError("checkpoint: optimizer includes frozen/unknown parameters") from exc
    flattened = [name for group in groups for name in group]
    if len(flattened) != len(parameters) or set(flattened) != set(parameters):
        raise ValueError("checkpoint: optimizer must include every installed trainable exactly once")
    return groups


def _cursor(cursor: dict) -> None:
    if (not {"epoch", "position", "discarded_problem_ids"} <= cursor.keys()
            or any(type(cursor[k]) is not int or cursor[k] < 0 for k in ("epoch", "position"))
            or not isinstance(cursor["discarded_problem_ids"], list)
            or any(not isinstance(s, str) for s in cursor["discarded_problem_ids"])):
        raise ValueError("checkpoint: complete nonnegative data cursor required")
    digest(cursor)


def _validate_optimizer(bundle: PolicyBundle, optimizer: torch.optim.AdamW,
                        scheduler: torch.optim.lr_scheduler.LambdaLR, protocol: dict, step: int) -> None:
    expected_opt, expected_sched = make_optimizer(bundle, protocol)
    if (type(optimizer) is not torch.optim.AdamW or type(scheduler) is not type(expected_sched)
            or scheduler.optimizer is not optimizer
            or _optimizer_names(bundle, optimizer) != _optimizer_names(bundle, expected_opt)):
        raise ValueError("checkpoint: use make_optimizer for exact resumable groups/scheduler")
    for actual, expected in zip(optimizer.param_groups, expected_opt.param_groups, strict=True):
        if {k: v for k, v in actual.items() if k not in ("params", "lr")} != {
                k: v for k, v in expected.items() if k not in ("params", "lr")}:
            raise ValueError("checkpoint: optimizer hyperparameters differ from protocol")
    state, initial = scheduler.state_dict(), expected_sched.state_dict()
    variable = {"last_epoch", "_step_count", "_last_lr"}
    multiplier = _schedule(protocol["optimizer"])
    if any(not isinstance(fn, partial) or fn.func is not multiplier.func
           or fn.args or fn.keywords != multiplier.keywords for fn in scheduler.lr_lambdas):
        raise ValueError("checkpoint: scheduler must be constructed by make_optimizer")
    expected_lr = [g["initial_lr"] * multiplier(step) for g in optimizer.param_groups]
    if (set(state) != set(initial) or state["last_epoch"] != step or state["_step_count"] != step + 1
            or {k: v for k, v in state.items() if k not in variable} != {
                k: v for k, v in initial.items() if k not in variable}
            or state["_last_lr"] != expected_lr or [g["lr"] for g in optimizer.param_groups] != expected_lr):
        raise ValueError("checkpoint: scheduler state/step/LR differs from bound warmup")
    installed = {p for ps in bundle.params.values() for p in ps}
    for parameter, values in optimizer.state.items():
        if (parameter not in installed or set(values) != {"step", "exp_avg", "exp_avg_sq"}
                or any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in values.values())
                or values["step"].numel() != 1 or not 1 <= float(values["step"]) <= step
                or float(values["step"]) != int(float(values["step"]))
                or any(values[k].shape != parameter.shape or values[k].dtype != parameter.dtype
                       for k in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("checkpoint: invalid AdamW moment state")


@contextmanager
def park_adam_moments(bundle: PolicyBundle, optimizer: torch.optim.AdamW,
                      scheduler: torch.optim.lr_scheduler.LambdaLR,
                      protocol: dict, step: int) -> Iterator[Callable[[], None]]:
    """Park only CUDA AdamW moments during collection; yielded restore() may move them back early."""
    _validate_optimizer(bundle, optimizer, scheduler, protocol, step)
    storages = {(p.device, p.untyped_storage().data_ptr())
                for p in bundle.model.parameters() if p.numel()}
    for parameter, state in optimizer.state.items():
        if (state["step"].device.type != "cpu" or state["step"].ndim != 0
                or any(state[key].device != parameter.device for key in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("checkpoint: parking requires CPU scalar steps and colocated moments")
        for value in state.values():
            if value.layout != torch.strided or value.device.type not in ("cpu", "cuda"):
                raise ValueError("checkpoint: unsupported AdamW parking tensor")
            storage = (value.device, value.untyped_storage().data_ptr())
            if value.numel() and storage in storages:
                raise ValueError("checkpoint: aliased AdamW state cannot be parked")
            if value.numel():
                storages.add(storage)
        del value  # Retaining this reference would keep a CUDA moment alive while parked.
    parked: list[tuple[dict, str, torch.device]] = []
    restored_early = False

    def unpark() -> None:
        for state, key, device in parked:
            state[key] = state[key].to(device)
        parked.clear()

    def restore() -> None:
        nonlocal restored_early
        restored_early = True
        unpark()

    try:
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                device = state[key].device
                if device.type == "cuda":
                    parked.append((state, key, device))
                    state[key] = state[key].to("cpu")
        yield restore
    finally:
        unpark()
        committed = restored_early and scheduler.last_epoch == step + 1
        _validate_optimizer(bundle, optimizer, scheduler, protocol, step + 1 if committed else step)


def save(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR,
         step: int, cursor: dict, rng: dict, out: Path, *, protocol: dict, binding: dict) -> dict:
    _binding(protocol, binding)
    _cursor(cursor)
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint: invalid step")
    assert_base_unchanged(bundle)
    parameters = _parameters(bundle)
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("checkpoint: save only after optimizer.zero_grad(set_to_none=True)")
    _validate_optimizer(bundle, optimizer, scheduler, protocol, step)
    before = capture_rng()
    try:
        restore_rng(rng)
    finally:
        restore_rng(before)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    groups = {row["name"]: row["group"] for row in bundle.inventory["trainable"]}
    for filename, merger in (("adapter.safetensors", False), ("mergers.safetensors", True)):
        save_file({name: p.detach().cpu().contiguous().clone() for name, p in parameters.items()
                   if (groups[name] == "M") == merger}, str(out / filename), metadata={"format": "pt"})
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng}, out / "runtime.pt")
    metadata = {"format": FORMAT, "base_id": BASE_ID, "base_revision": BASE_REVISION,
        "arm": bundle.inventory["arm"], "protocol": protocol, "binding": binding,
        "resolved_protocol_sha256": digest(protocol),
        "inventory": bundle.inventory, "base_sha256": bundle.base_sha256,
        "state_sha256": {n: tensor_sha256(p) for n, p in parameters.items()},
        "optimizer_names": _optimizer_names(bundle, optimizer), "step": step, "cursor": cursor,
        "vision_on": bundle.route.vision_on, "files": {n: file_hash(out / n) for n in sorted(_FILES)}}
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    wire = json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    (out / "metadata.json").write_text(wire, encoding="utf-8")
    (out / "metadata.sha256").write_text(file_hash(out / "metadata.json") + "\n", encoding="ascii")
    assert_base_unchanged(bundle)
    return metadata


def read_metadata(ckpt: Path, *, binding: dict) -> dict:
    ckpt = Path(ckpt)
    expected = _FILES | {"metadata.json", "metadata.sha256"}
    if (not ckpt.is_dir() or ckpt.is_symlink() or {p.name for p in ckpt.iterdir()} != expected
            or any(not p.is_file() or p.is_symlink() for p in ckpt.iterdir())):
        raise ValueError("checkpoint: missing/unexpected files or symlinks")
    if (ckpt / "metadata.sha256").read_text(encoding="ascii").strip() != file_hash(ckpt / "metadata.json"):
        raise ValueError("checkpoint: metadata checksum mismatch")
    metadata = json.loads((ckpt / "metadata.json").read_text(encoding="utf-8"))
    keys = {"format", "base_id", "base_revision", "arm", "protocol", "binding", "resolved_protocol_sha256", "inventory", "base_sha256",
            "state_sha256", "optimizer_names", "step", "cursor", "vision_on", "files"}
    if (set(metadata) != keys or metadata["format"] != FORMAT or metadata["base_id"] != BASE_ID
            or metadata["base_revision"] != BASE_REVISION or metadata["binding"] != binding
            or metadata["resolved_protocol_sha256"] != digest(metadata["protocol"])
            or set(metadata["files"]) != _FILES
            or any(file_hash(ckpt / name) != value for name, value in metadata["files"].items())
            or type(metadata["step"]) is not int or metadata["step"] < 0
            or type(metadata["vision_on"]) is not bool):
        raise ValueError("checkpoint: binding, identity, runtime, or file checksum mismatch")
    _binding(metadata["protocol"], binding)
    _cursor(metadata["cursor"])
    return metadata


def _load(model_path: Path, ckpt: Path, binding: dict,
          factory: PolicyFactory | None) -> tuple[PolicyBundle, dict]:
    metadata = read_metadata(ckpt, binding=binding)
    bundle = (factory or build_policy)(Path(model_path), metadata["arm"], metadata["protocol"])
    if bundle.inventory != metadata["inventory"] or bundle.base_sha256 != metadata["base_sha256"]:
        raise ValueError("checkpoint: fresh F0 inventory/reference differs")
    parameters = _parameters(bundle)
    adapters = load_file(str(ckpt / "adapter.safetensors"), device="cpu")
    mergers = load_file(str(ckpt / "mergers.safetensors"), device="cpu")
    rows = {r["name"]: r for r in bundle.inventory["trainable"]}
    if (set(adapters) != {n for n, r in rows.items() if r["group"] != "M"}
            or set(mergers) != {n for n, r in rows.items() if r["group"] == "M"}
            or set(metadata["state_sha256"]) != set(parameters)):
        raise ValueError("checkpoint: missing/unexpected adapter or Merger tensor keys")
    for name, value in (adapters | mergers).items():
        parameter = parameters[name]
        if (value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())
                or tensor_sha256(value) != metadata["state_sha256"][name]):
            raise ValueError(f"checkpoint: invalid tensor {name}")
    with torch.no_grad():
        for name, value in (adapters | mergers).items():
            parameters[name].copy_(value)
    bundle.route.apply(metadata["vision_on"])
    assert_base_unchanged(bundle)
    if read_metadata(ckpt, binding=binding) != metadata:
        raise ValueError("checkpoint: source changed while loading")
    return bundle, metadata


def load_eval(model_path: Path, ckpt: Path, *, binding: dict,
              policy_factory: PolicyFactory | None = None,
              _test_policy_factory: PolicyFactory | None = None) -> PolicyBundle:
    if policy_factory is not None and _test_policy_factory is not None:
        raise ValueError("checkpoint: choose exactly one policy factory")
    bundle, _ = _load(model_path, Path(ckpt), binding, policy_factory or _test_policy_factory)
    bundle.model.eval()
    return bundle


def load_resume(model_path: Path, ckpt: Path, protocol: dict, *, binding: dict,
                device: str | torch.device = "cpu",
                policy_factory: PolicyFactory | None = None,
                _test_policy_factory: PolicyFactory | None = None
                ) -> tuple[PolicyBundle, torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR, int, dict]:
    _binding(protocol, binding)
    metadata = read_metadata(ckpt, binding=binding)
    if metadata["protocol"] != json.loads(json.dumps(protocol)):
        raise ValueError("checkpoint: protocol differs")
    if policy_factory is not None and _test_policy_factory is not None:
        raise ValueError("checkpoint: choose exactly one policy factory")
    bundle, metadata = _load(model_path, Path(ckpt), binding, policy_factory or _test_policy_factory)
    # Move before reconstructing/loading AdamW: load_state_dict casts moments to
    # parameter devices. Moving only the model afterward leaves stale CPU moments.
    bundle.model.to(torch.device(device))
    optimizer, scheduler = make_optimizer(bundle, protocol)
    if metadata["optimizer_names"] != _optimizer_names(bundle, optimizer):
        raise ValueError("checkpoint: optimizer parameter order differs")
    runtime = torch.load(Path(ckpt) / "runtime.pt", map_location="cpu", weights_only=True)
    if set(runtime) != {"optimizer", "scheduler", "rng"}:
        raise ValueError("checkpoint: unexpected runtime state")
    optimizer.load_state_dict(runtime["optimizer"])
    scheduler.load_state_dict(runtime["scheduler"])
    _validate_optimizer(bundle, optimizer, scheduler, protocol, metadata["step"])
    if read_metadata(ckpt, binding=binding) != metadata:
        raise ValueError("checkpoint: source changed while loading runtime")
    # Fresh construction may consume random numbers; restore only after every load.
    restore_rng(runtime["rng"])
    bundle.model.train()
    return bundle, optimizer, scheduler, metadata["step"], metadata["cursor"]
