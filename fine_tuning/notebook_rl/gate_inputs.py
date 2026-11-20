"""Private, write-once G12 input handoff; not data/GPU/training approval.

Preserve actual versioned action trees and literal probe tokens. Images stay in
the caller's private dataset; their encoded bytes and decoded pixels are checked
again in the fresh process. Load this packet BEFORE checkpoint.load_resume so
input reconstruction cannot consume the restored optimizer-step RNG state.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

from . import checkpoint, launch
from .dataset import _bytes, _sha, _write, load_images
from .engines import RolloutRequest
from .schema import ActorItem, ProblemTree, from_json, to_json
from .trainer import SyncProbe
from .vault import _read, _safe_root

FORMAT = "notebook-rl-g12-inputs-v1"


def _require(value, message):
    if not value:
        raise ValueError("G12 inputs: " + message)


def _pixels(images):
    return [{"mode": im.mode, "size": list(im.size),
             "pixels_sha256": hashlib.sha256(im.tobytes()).hexdigest()} for im in images]


def _load_images(actor, image_root, stack):
    # Register each image immediately, including when a later image is corrupt.
    result = []
    for rel, digest in zip(actor.image_paths, actor.image_sha256, strict=True):
        image = load_images(replace(actor, image_paths=(rel,), image_sha256=(digest,)), image_root)[0]
        stack.callback(image.close)
        result.append(image)
    return result


def _record(cls, value):
    return from_json(cls, json.dumps(value, allow_nan=False))


def _validate(packet, protocol, binding):
    _require(type(packet) is dict and set(packet) == {
        "format", "binding", "step", "algorithm", "seed", "cursor", "trees", "actors", "images",
        "probes", "training_allowed", "gpu_gates_certified"}, "exact packet schema required")
    checkpoint._binding(protocol, binding)
    _require(packet["format"] == FORMAT and packet["binding"] == binding
             and packet["training_allowed"] is False and packet["gpu_gates_certified"] is False,
             "packet identity differs; a handoff is never approval")
    step = packet["step"]
    _require(type(step) is int and step > 0 and type(packet["seed"]) is int
             and packet["seed"] in protocol["seeds"] and packet["algorithm"] in ("ALG-T1", "ALG-NB", "ALG-NB3"),
             "positive policy version and supported fixed RL protocol required")
    checkpoint._cursor(packet["cursor"])
    _require(type(packet["trees"]) is list and 1 <= len(packet["trees"]) <= protocol["batch"]["problems_per_step"],
             "bounded actual next-step trees required")
    trees = [_record(ProblemTree, value) for value in packet["trees"]]
    ids = [tree.problem_id for tree in trees]
    _require(len(ids) == len(set(ids)) and type(packet["actors"]) is dict
             and type(packet["images"]) is dict and set(ids) == set(packet["actors"]) == set(packet["images"]),
             "exact tree/actor/image coverage required")
    actors = {pid: _record(ActorItem, value) for pid, value in packet["actors"].items()}
    for tree in trees:
        _require(tree.policy_version == step and actors[tree.problem_id].problem_id == tree.problem_id,
                 "stale tree or actor identity")
        _require(bool(tree.groups), "empty action tree")
        for group in tree.groups:
            _require(group.problem_id == tree.problem_id, "cross-problem group")
            for sample in group.samples:
                _require(sample.problem_id == tree.problem_id and sample.group_id == group.group_id
                         and sample.policy_version == step, "cross-problem or stale sampled action")
    _require(type(packet["probes"]) is list and len(packet["probes"]) == 2, "two actual probes required")
    probe_ids, counts = [], []
    for row in packet["probes"]:
        _require(type(row) is dict and set(row) == {"actor", "request", "completion_ids", "images"},
                 "exact probe schema required")
        actor, req = _record(ActorItem, row["actor"]), row["request"]
        _require(type(req) is dict and set(req) == {"request_id", "prompt_token_ids", "messages", "n",
                 "max_new_tokens", "seeds"} and req["messages"] is None, "literal probe tokens required")
        _require(isinstance(req["request_id"], str) and bool(req["request_id"])
                 and type(req["n"]) is int and req["n"] == 1
                 and type(req["max_new_tokens"]) is int and 1 <= req["max_new_tokens"] <= 33
                 and type(req["seeds"]) is list and len(req["seeds"]) == 1
                 and type(req["seeds"][0]) is int and 0 <= req["seeds"][0] < 2**31,
                 "fixed one-action probe request required")
        for key, values in (("prompt", req["prompt_token_ids"]), ("completion", row["completion_ids"])):
            _require(type(values) is list and bool(values)
                     and all(type(t) is int and t >= 0 for t in values), "invalid " + key + " token IDs")
        _require(len(row["completion_ids"]) <= 32
                 and len(req["prompt_token_ids"]) + max(len(row["completion_ids"]), req["max_new_tokens"])
                 <= protocol["context_limit"],
                 "probe exceeds fixed context/completion bound")
        probe_ids.append(req["request_id"])
        counts.append(len(actor.image_paths))
    _require(len(set(probe_ids)) == 2 and sorted(counts) == [1, 2], "distinct single/two-image probes required")
    return trees, actors


def save_inputs(path: Path, *, trees, images, actors, probes, probe_actors, preparation,
                image_root: Path, protocol: dict, binding: dict, step: int, algorithm: str,
                seed: int, cursor: dict) -> dict:
    """Freeze actual caller-owned next-step inputs without creating/changing images.

The caller must establish data eligibility first. No checkpoint/model is loaded,
no response is generated and no reward is recomputed by this serializer.
"""
    _safe_root(image_root)
    _safe_root(path.parent)
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents))
             and not path.resolve().is_relative_to(launch.ROOT)
             and not path.resolve().is_relative_to(image_root.resolve()), "new private output outside input/checkout required")
    _require(preparation.policy_version == step, "preparation version differs from sampled batch")
    ids = [tree.problem_id for tree in trees]
    _require(set(images) == set(ids) and set(ids) <= set(actors), "actual batch images/actors required")
    packet: dict = {"format": FORMAT, "binding": binding, "step": step, "algorithm": algorithm,
        "seed": seed, "cursor": cursor, "trees": [json.loads(to_json(t)) for t in trees],
        "actors": {pid: json.loads(to_json(actors[pid])) for pid in ids},
        "images": {pid: _pixels(images[pid]) for pid in ids}, "probes": [],
        "training_allowed": False, "gpu_gates_certified": False}
    with ExitStack() as stack:
        for pid in ids:
            _require(_pixels(_load_images(actors[pid], image_root, stack)) == packet["images"][pid],
                     "batch pixels differ from source bytes")
        _require(set(probe_actors) == {p.request.request_id for p in probes}, "exact probe-source mapping required")
        for probe in probes:
            req, actor = probe.request, probe_actors[probe.request.request_id]
            _require(_pixels(_load_images(actor, image_root, stack)) == _pixels(req.images),
                     "probe pixels differ from source bytes")
            prepared = preparation.prepare(req)
            literal = {"request_id": req.request_id, "prompt_token_ids": prepared.token_ids, "messages": None,
                       "n": req.n, "max_new_tokens": req.max_new_tokens, "seeds": req.seeds}
            packet["probes"].append({"actor": json.loads(to_json(actor)), "request": literal,
                                     "completion_ids": probe.completion_ids, "images": _pixels(req.images)})
    _validate(packet, protocol, binding)
    raw = _bytes(packet) + b"\n"
    _write(path, raw)
    return {"path": str(path), "file_sha256": _sha(raw)}


@dataclass(frozen=True)
class ReplayInputs:
    trees: list[ProblemTree]
    images: dict
    probes: list[SyncProbe]
    step: int
    algorithm: str
    seed: int
    cursor: dict


def load_inputs(descriptor: dict, *, image_root: Path, protocol: dict, binding: dict,
                stack: ExitStack) -> ReplayInputs:
    """Reconstruct private inputs; caller's stack owns every opened image.

Call before loading checkpoint RNG. The parent's immutable descriptor, not a
replacement file's newly computed hash, must be passed to the fresh process.
"""
    path = launch.descriptor(descriptor)
    _safe_root(path.parent)
    _require(not path.is_symlink(), "packet symlink forbidden")
    raw = _read(path)
    _require(_sha(raw) == descriptor["file_sha256"], "packet bytes changed")
    packet = launch.read_json(path)
    _require(_read(path) == raw, "packet changed while reading")
    trees, actors = _validate(packet, protocol, binding)
    # A local stack closes partial reconstructions even if the caller catches
    # an error and keeps its own context alive.
    with ExitStack() as local:
        images = {pid: _load_images(actor, image_root, local) for pid, actor in actors.items()}
        _require({pid: _pixels(values) for pid, values in images.items()} == packet["images"],
                 "batch pixel fingerprints differ")
        probes = []
        for row in packet["probes"]:
            values = _load_images(_record(ActorItem, row["actor"]), image_root, local)
            _require(_pixels(values) == row["images"], "probe pixel fingerprints differ")
            probes.append(SyncProbe(RolloutRequest(images=values, **row["request"]), row["completion_ids"]))
        result = ReplayInputs(trees, images, probes, packet["step"], packet["algorithm"], packet["seed"], packet["cursor"])
        stack.callback(local.pop_all().close)
        return result


def capture_from_inputs(bundle, optimizer, scheduler, preparation, descriptor: dict, *,
                        image_root: Path, protocol: dict, binding: dict, model_path: Path,
                        output: Path) -> dict:
    """Consume the frozen packet for the real G12 capture/update, then bind it.

Caller owns the already-approved native policy and financial watchdog. This is
not a launch/approval bypass; the native capture's device/source guards remain.
"""
    from .gate_checkpoint import capture_checkpoint

    with ExitStack() as stack:
        inputs = load_inputs(descriptor, image_root=image_root, protocol=protocol, binding=binding, stack=stack)
        _require(preparation.policy_version == inputs.step, "capture preparation version differs")
        result = capture_checkpoint(bundle, optimizer, scheduler, inputs.trees, inputs.images,
            preparation, inputs.probes, protocol, inputs.algorithm, step=inputs.step, seed=inputs.seed,
            cursor=inputs.cursor, binding=binding, model_path=model_path, output=output)
    receipt = {"input_packet": descriptor, "capture_result_sha256": checkpoint.file_hash(output / "result.json")}
    _write(output / "inputs.json", _bytes(receipt) + b"\n")
    return result


def replay_from_inputs(model_path: Path, capture: Path, output: Path, descriptor: dict, *,
                       image_root: Path, protocol: dict, binding: dict, device: str) -> dict:
    """Fresh owned child replay, preserving loaded RNG and exact action inputs.

The coordinator must retain both actual Popen handles for compare_checkpoint.
This wrapper does not certify the run or start a cloud resource.
"""
    from .engines import InputPreparation
    from .gate_checkpoint import _batch, replay_checkpoint

    receipt = launch.read_json(capture / "inputs.json")
    _require(receipt == {"input_packet": descriptor,
                        "capture_result_sha256": checkpoint.file_hash(capture / "result.json")},
             "packet is not bound to this completed capture")
    previous = launch.read_json(capture / "result.json")
    with ExitStack() as stack:
        inputs = load_inputs(descriptor, image_root=image_root, protocol=protocol, binding=binding, stack=stack)
        _require(previous["step"] == inputs.step and previous["cursor"] == inputs.cursor
                 and previous["binding"] == binding
                 and previous["measured"]["batch"] == _batch(inputs.trees, inputs.images,
                    inputs.algorithm, inputs.seed, False), "packet does not reproduce captured RL batch")
        def preparation_factory(bundle, step):
            _require(step == inputs.step, "restored checkpoint version differs")
            value = InputPreparation(bundle.processor, bundle.model.config, protocol["image"], protocol["context_limit"])
            value.policy_version = step
            return value
        return replay_checkpoint(model_path, capture, output, inputs.trees, inputs.images, inputs.probes,
            preparation_factory, protocol=protocol, binding=binding, device=device)
