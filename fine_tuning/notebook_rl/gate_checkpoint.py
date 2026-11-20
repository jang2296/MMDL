"""G12 capture/replay stages for sequential, coordinator-owned processes.

Capture saves a caller-owned policy, observes logits and advances its real RL
optimizer once. Replay loads only after capture exits and repeats that operation.
The model-free coordinator must retain both Popen handles and their raw logs.
These stages do not rent resources, launch a runner, synchronize vLLM or approve
training. Failed updates and partial artifacts are deliberately never rolled back.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import load_file, save_file

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from . import checkpoint, policy, updates
from .engines import InputPreparation
from .gate_reference import _input_fingerprint, _observed_logits
from .schema import to_json
from .trainer import _write

ROOT = Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G12: " + message)


def _process() -> dict:
    process = psutil.Process()
    return {"pid": process.pid, "parent_pid": process.ppid(), "birth": process.create_time(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def _exited(identity: dict) -> bool:
    _require(identity["boot_id"] == _process()["boot_id"], "same live host/boot required")
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["birth"]
    except psutil.NoSuchProcess:
        return True


def _state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), "nonfinite measured state")
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha256(value.reshape(-1))}
    if isinstance(value, dict):
        return {str(k): _state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_state(v) for v in value]
    return value


def _runtime(bundle, optimizer, scheduler) -> dict:
    policy.assert_base_unchanged(bundle)
    return {"parameters": {n: _state(p) for n, p in bundle.model.named_parameters()},
            "optimizer": _state(optimizer.state_dict()), "scheduler": _state(scheduler.state_dict()),
            "rng": _state(checkpoint.capture_rng()), "vision_on": bundle.route.vision_on,
            "requires_grad": {n: p.requires_grad for n, p in bundle.model.named_parameters()},
            "training": bundle.model.training, "base_sha256": bundle.base_sha256}


def _sources() -> dict:
    names = ["modules.py", "train.py", *["notebook_rl/" + n + ".py" for n in
        ("gate_checkpoint", "gate_inputs", "gate_session", "gate_sequence", "gate_snapshot", "gate_reference", "checkpoint", "policy", "updates", "losses",
         "engines", "answers", "schema", "trainer", "dataset", "launch", "vault")]]
    return {name: checkpoint.file_hash(ROOT / "fine_tuning" / name) for name in names}


def _model_files(model_path: Path, protocol: dict, test_manifest: Path | None) -> dict:
    manifest_path = test_manifest or ROOT / protocol["initial_checkpoint"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["repo_id"] == BASE_ID and manifest["revision"] == BASE_REVISION
             and model_path.name == BASE_REVISION, "pinned source identity required")
    names = [row["path"] for row in manifest["files"]]
    _require(bool(names) and len(names) == len(set(names)) and all(Path(n).name == n for n in names),
             "unique flat model-file manifest required")
    for row in manifest["files"]:
        path = model_path / row["path"]
        _require(path.stat().st_size == row["bytes"] and checkpoint.file_hash(path) == row["sha256"],
                 "model/processor source bytes changed: " + row["path"])
    return {"manifest_sha256": checkpoint.file_hash(manifest_path), "files": manifest["files"]}


def _native(bundle, preparation, protocol: dict) -> None:
    from .gate_weights import _pinned_config
    from .gpu_gates import measure_inventory

    _require(type(bundle) is policy.PolicyBundle, "actual PolicyBundle required")
    devices = {p.device for p in bundle.model.parameters()}
    _require(len(devices) == 1 and next(iter(devices)).type == "cuda"
             and next(iter(devices)).index is not None, "explicit actual CUDA policy required")
    _require(bundle.inventory["base_id"] == BASE_ID and bundle.inventory["base_revision"] == BASE_REVISION,
             "pinned model identity required")
    _pinned_config(bundle.model.config)
    _require(measure_inventory(bundle, protocol)["pinned_architecture_counts"], "pinned architecture required")
    _require(type(preparation) is InputPreparation and preparation.processor is bundle.processor
             and preparation.model_config is bundle.model.config,
             "actual policy InputPreparation required, never an injected forward")
    ancestors = {p.pid for p in psutil.Process().parents()}
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                     text=True, timeout=15)
    _require(not ancestors.intersection(int(s.strip()) for s in active.splitlines() if s.strip().isdigit()),
             "coordinator/ancestors must not retain a CUDA model during child measurement")


class _ObservedPreparation:
    def __init__(self, preparation):
        self.preparation = preparation
        self.tokenizer, self.policy_version = preparation.tokenizer, preparation.policy_version
        self.calls: list[dict] = []

    def model_inputs(self, request, completion, device):
        inputs = self.preparation.model_inputs(request, completion, device)
        _require(inputs["input_ids"].tolist() == [request.prompt_token_ids + completion],
                 "actual RL token inputs differ from the recorded action")
        self.calls.append({"sample_id": request.request_id,
                           "input": _input_fingerprint(inputs, len(completion), device)})
        return inputs


def _batch(trees, images, algorithm: str, seed: int, vision_disabled: bool) -> dict:
    _require(type(vision_disabled) is bool and type(seed) is int, "explicit route/seed required")
    return {"trees": [to_json(t) for t in trees], "images": {pid: [{"mode": im.mode,
        "size": list(im.size), "pixels_sha256": hashlib.sha256(im.tobytes()).hexdigest()}
        for im in values] for pid, values in images.items()}, "algorithm": algorithm,
        "seed": seed, "vision_disabled": vision_disabled}


def _observe_and_step(bundle, optimizer, scheduler, trees, images, preparation, probes,
                      protocol: dict, algorithm: str, step: int, seed: int, vision_disabled: bool,
                      output: Path) -> dict:
    _require(bundle.model.training and all(p.grad is None for p in bundle.model.parameters()),
             "training-mode checkpoint with cleared gradients required")
    _require(1 <= len(trees) <= protocol["batch"]["problems_per_step"],
             "bounded next-step problem batch required")
    _require(len(probes) == 2 and len({p.request.request_id for p in probes}) == 2
             and sorted(len(p.request.images) for p in probes) == [1, 2]
             and all(1 <= len(p.completion_ids) <= 32 for p in probes),
             "two distinct single/multi-image probes with at most 32 completion tokens required")
    checkpoint._validate_optimizer(bundle, optimizer, scheduler, protocol, step)
    before = _runtime(bundle, optimizer, scheduler)
    device = next(bundle.model.parameters()).device
    rng, route = checkpoint.capture_rng(), bundle.route.vision_on
    measured, logits = [], {}
    try:
        for index, probe in enumerate(probes):
            inputs = preparation.model_inputs(probe.request, probe.completion_ids, device)
            length = len(probe.completion_ids)
            _require(inputs["input_ids"].tolist() == [probe.request.prompt_token_ids + probe.completion_ids],
                     "probe tokens differ from actual inputs")
            _require(inputs["image_grid_thw"].shape[0] == len(probe.request.images),
                     "probe image count differs from actual grid")
            measured.append({"id": probe.request.request_id, "input": _input_fingerprint(inputs, length, device)})
            for reference in (False, True):
                full, lp = _observed_logits(bundle, inputs, length, reference=reference)
                logits[f"{index}.{'reference' if reference else 'policy'}.logits"] = full
                logits[f"{index}.{'reference' if reference else 'policy'}.logprobs"] = lp
    finally:
        bundle.route.apply(route)
        checkpoint.restore_rng(rng)
    _require(_runtime(bundle, optimizer, scheduler) == before, "logit observation mutated policy/runtime")
    save_file(logits, str(output / "logits.safetensors"))
    batch = _batch(trees, images, algorithm, seed, vision_disabled)
    observed = _ObservedPreparation(preparation)
    update = updates.optimizer_step(bundle, optimizer, scheduler, trees, images, observed,
        protocol, algorithm, step=step, seed=seed, vision_disabled=vision_disabled)
    _require(update["status"] == "UPDATED_UNSYNCHRONIZED" and update["step_after"] == step + 1,
             "actual next production RL optimizer step required; no-op cannot pass")
    _require(all(p.grad is None for p in bundle.model.parameters()), "update left gradients")
    checkpoint._validate_optimizer(bundle, optimizer, scheduler, protocol, step + 1)
    _require(batch == _batch(trees, images, algorithm, seed, vision_disabled), "bound RL batch mutated")
    after = _runtime(bundle, optimizer, scheduler)
    _require(before["parameters"] != after["parameters"], "actual optimizer step must change parameters")
    return {"before": before, "after": after, "batch": batch, "probes": measured,
            "input_calls": observed.calls, "update": {k: v for k, v in update.items() if k != "seconds"},
            "logits_sha256": checkpoint.file_hash(output / "logits.safetensors")}


def _stage(output: Path, operation) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "started.json", {"process": _process()})
    try:
        record = operation()
        _write(output / "result.json", record)
        return record
    except BaseException as exc:
        _write(output / "error.json", {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc(), "process": _process()})
        raise


def _capture_checkpoint(bundle, optimizer, scheduler, trees, images, preparation, probes,
                        protocol: dict, algorithm: str, *, step: int, seed: int, cursor: dict,
                        binding: dict, model_path: Path, output: Path, vision_disabled: bool = False,
                        _test_manifest: Path | None = None) -> dict:
    """Private CPU seam; supplying a tiny source manifest never yields native evidence."""
    def operation():
        native = _test_manifest is None
        if native:
            _native(bundle, preparation, protocol)
        files, sources = _model_files(model_path, protocol, _test_manifest), _sources()
        metadata = checkpoint.save(bundle, optimizer, scheduler, step, cursor, checkpoint.capture_rng(),
            output / "checkpoint", protocol=protocol, binding=binding)
        measured = _observe_and_step(bundle, optimizer, scheduler, trees, images, preparation, probes,
            protocol, algorithm, step, seed, vision_disabled, output)
        _require(files == _model_files(model_path, protocol, _test_manifest) and sources == _sources(),
                 "model/source changed during capture")
        return {"stage": "CAPTURE", "process": _process(), "native_execution": native,
            "model_files": files, "sources": sources, "binding": binding, "protocol": protocol,
            "checkpoint_sha256": checkpoint.file_hash(output / "checkpoint/metadata.json"),
            "step": step, "cursor": metadata["cursor"], "measured": measured}
    return _stage(output, operation)


def capture_checkpoint(bundle, optimizer, scheduler, trees, images, preparation, probes,
                       protocol: dict, algorithm: str, *, step: int, seed: int, cursor: dict,
                       binding: dict, model_path: Path, output: Path, vision_disabled: bool = False) -> dict:
    """Save and measure once in a dedicated pinned-CUDA child, which must then exit."""
    return _capture_checkpoint(bundle, optimizer, scheduler, trees, images, preparation, probes,
        protocol, algorithm, step=step, seed=seed, cursor=cursor, binding=binding, model_path=model_path,
        output=output, vision_disabled=vision_disabled)


def _replay_checkpoint(model_path: Path, capture: Path, output: Path, trees, images, probes,
                       preparation_factory, *, protocol: dict, binding: dict, device: str,
                       _test_manifest: Path | None = None, _test_policy_factory=None) -> dict:
    def operation():
        previous = json.loads((capture / "result.json").read_text())
        _require(previous["stage"] == "CAPTURE" and previous["process"]["pid"] != os.getpid()
                 and _exited(previous["process"]), "capture process must exit before fresh reload")
        native = _test_manifest is None and _test_policy_factory is None
        _require(previous["native_execution"] == native, "cannot promote a private CPU measurement")
        _require(not native or torch.device(device).type == "cuda", "actual CUDA reload required")
        _require(checkpoint.digest(protocol) == checkpoint.digest(previous["protocol"])
                 and binding == previous["binding"] and _sources() == previous["sources"]
                 and _model_files(model_path, protocol, _test_manifest) == previous["model_files"],
                 "capture/replay source or input binding differs")
        path = capture / "checkpoint"
        _require(checkpoint.file_hash(path / "metadata.json") == previous["checkpoint_sha256"],
                 "capture checkpoint changed")
        bundle, optimizer, scheduler, step, cursor = checkpoint.load_resume(model_path, path, protocol,
            binding=binding, device=device, _test_policy_factory=_test_policy_factory)
        saved_rng = checkpoint.capture_rng()
        preparation = preparation_factory(bundle, step)
        checkpoint.restore_rng(saved_rng)
        if native:
            _native(bundle, preparation, protocol)
        _require((step, cursor) == (previous["step"], previous["cursor"]), "resumed cursor differs")
        batch = previous["measured"]["batch"]
        measured = _observe_and_step(bundle, optimizer, scheduler, trees, images, preparation, probes,
            protocol, batch["algorithm"], step, batch["seed"], batch["vision_disabled"], output)
        _require(_sources() == previous["sources"]
                 and _model_files(model_path, protocol, _test_manifest) == previous["model_files"]
                 and checkpoint.file_hash(path / "metadata.json") == previous["checkpoint_sha256"],
                 "source/checkpoint changed during replay")
        checkpoint.read_metadata(path, binding=binding)
        return {**{k: v for k, v in previous.items() if k not in ("stage", "process", "measured")},
            "stage": "REPLAY", "process": _process(), "capture_result_sha256": checkpoint.file_hash(capture / "result.json"),
            "measured": measured}
    return _stage(output, operation)


def replay_checkpoint(model_path: Path, capture: Path, output: Path, trees, images, probes,
                      preparation_factory, *, protocol: dict, binding: dict, device: str) -> dict:
    """Load fresh F0 plus saved state; construct InputPreparation from the new bundle."""
    return _replay_checkpoint(model_path, capture, output, trees, images, probes, preparation_factory,
                              protocol=protocol, binding=binding, device=device)


def compare_checkpoint(capture: Path, replay: Path, capture_process: subprocess.Popen,
                       replay_process: subprocess.Popen, output: Path) -> dict:
    """Require actual owned exited children, then compare every measured value exactly."""
    def operation():
        _require(not torch.cuda.is_initialized(), "comparison coordinator must remain model-free on CPU")
        a, b = [json.loads((p / "result.json").read_text()) for p in (capture, replay)]
        for child, record in ((capture_process, a), (replay_process, b)):
            _require(type(child) is subprocess.Popen and child.pid == record["process"]["pid"]
                     and child.poll() == 0 and record["process"]["parent_pid"] == os.getpid()
                     and _exited(record["process"]), "actual owned child PID and successful process exit required")
        _require(a["stage"] == "CAPTURE" and b["stage"] == "REPLAY"
                 and a["process"]["pid"] != b["process"]["pid"], "separate capture/replay processes required")
        _require(b["capture_result_sha256"] == checkpoint.file_hash(capture / "result.json"),
                 "replay did not bind this exact capture")
        common = ("native_execution", "model_files", "sources", "binding", "protocol", "checkpoint_sha256", "step", "cursor")
        _require(all(a[k] == b[k] for k in common) and _sources() == a["sources"], "stage bindings differ")
        metadata = checkpoint.read_metadata(capture / "checkpoint", binding=a["binding"])
        _require(checkpoint.file_hash(capture / "checkpoint/metadata.json") == a["checkpoint_sha256"]
                 and metadata["cursor"] == a["cursor"], "checkpoint changed after measurement")
        measurements = [r["measured"] for r in (a, b)]
        for path, measured in zip((capture, replay), measurements, strict=True):
            _require(checkpoint.file_hash(path / "logits.safetensors") == measured["logits_sha256"],
                     "measured raw logits changed")
        left, right = [load_file(str(path / "logits.safetensors")) for path in (capture, replay)]
        _require(set(left) == set(right) and len(left) == 8, "complete policy/reference logits and logprobs required")
        errors = {}
        for key in left:
            x, y = left[key], right[key]
            _require(x.shape == y.shape and x.dtype == y.dtype and bool(torch.isfinite(x).all())
                     and bool(torch.isfinite(y).all()), "logit shape/dtype/finite mismatch")
            errors[key] = {"max_abs": float((x - y).abs().max()), "bit_equal": torch.equal(x, y),
                           "expected_sha256": tensor_sha256(x), "actual_sha256": tensor_sha256(y)}
        differences = [k for k in measurements[0] if k != "logits_sha256" and measurements[0][k] != measurements[1][k]]
        result = {"fresh_process": True, "native_execution": a["native_execution"],
            "capture_process": a["process"], "replay_process": b["process"],
            "checkpoint_sha256": a["checkpoint_sha256"], "binding": a["binding"],
            "logit_errors": errors, "different_fields": differences,
            "next_loss_equal": measurements[0]["update"]["loss"] == measurements[1]["update"]["loss"],
            "training_allowed": False, "gpu_gates_certified": False,
            "limitations": ["Two sequential owned children; no vLLM synchronization or full gate runner",
                "Full vocabulary at two completion probes, not every prompt position",
                "Private tiny-CPU measurements cannot certify pinned 4B CUDA behavior"]}
        passed = not differences and all(row["bit_equal"] for row in errors.values())
        _write(output / "comparison.json", result | {"passed": passed})
        _require(passed, "checkpoint roundtrip differs; see raw comparison and preserved stage artifacts")
        return result | {"passed": True, "logits_equal": True, "next_loss_equal": True}
    return _stage(output, operation)
