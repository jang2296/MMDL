"""G13 observations around ONE existing G4/G11 production reward update.

Data eligibility is rechecked by the isolated approval worker, never inferred
from a caller's PASS. This is a gate measurement, not training/spending approval.
Python audit guards remain installed after exit; use an owned fresh process.
Weights/AdamW are not rolled back, including on a failed post-update check.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from fine_tuning import train
from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from . import approval, checkpoint, guards, policy, updates
from .dataset import SPLITS, load_images
from .engines import InputPreparation, RolloutRequest
from .prompts import load_prompts, turn1_text, user_message
from .schema import ActorItem, from_json, load_protocol, to_json

_HERE = Path(__file__).resolve().parent
_DATA_KEYS = {"index", "source_registry", "evidence", "data_report", "expected_measurement_scope"}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("G13: " + message)


def _native(bundle, preparation, protocol: dict) -> None:
    from .gate_weights import _pinned_config
    from .gpu_gates import measure_inventory

    _require(type(bundle) is policy.PolicyBundle, "actual PolicyBundle required")
    devices = {p.device for p in bundle.model.parameters()}
    _require(len(devices) == 1 and next(iter(devices)).type == "cuda"
             and next(iter(devices)).index is not None, "actual explicit CUDA policy required")
    _require(bundle.inventory["base_id"] == BASE_ID and bundle.inventory["base_revision"] == BASE_REVISION,
             "pinned model identity required")
    _pinned_config(bundle.model.config)
    _require(measure_inventory(bundle, protocol)["pinned_architecture_counts"], "pinned 4B architecture required")
    _require(type(preparation) is InputPreparation and preparation.processor is bundle.processor
             and preparation.model_config is bundle.model.config, "actual policy InputPreparation required")
    _require(protocol == load_protocol(_HERE / "protocol_v1.yaml"), "resolved protocol differs from bound source")


def _eligibility(root: Path, request: dict, protocol: dict, validator=None) -> dict:
    _require(set(request) == _DATA_KEYS, "complete data descriptors/scope required; never inline PASS")
    for key in _DATA_KEYS - {"expected_measurement_scope"}:
        value = request[key]
        _require(type(value) is dict and set(value) == {"path", "file_sha256"}
                 and Path(value["path"]).is_absolute(), "path/SHA data descriptors required")
    result = (validator or approval.validate_for_gate_data)(root, **request)
    _require(result.get("status") == "VALIDATED_FOR_MEASUREMENT"
             and result.get("measurement_allowed") is True and result.get("training_allowed") is False
             and result.get("approval_created") is False and result.get("blockers") == [],
             "data prerequisites rejected: " + ",".join(result.get("blockers", [])))
    _require(result["input_binding"]["protocol_sha256"] == protocol["sha256"], "data protocol differs")
    actors = [a if type(a) is ActorItem else from_json(ActorItem, json.dumps(a)) for a in result["actors"]]
    _require(bool(actors) and [a.problem_id for a in actors] == result["selected_ids"]
             and len(set(result["selected_ids"])) == len(actors), "complete actual actor allowlist required")
    if validator is None:
        _require(result["code_sha256"] == approval.code_hashes(), "data approval code is stale")
    return result | {"actors": actors}


def _image(image) -> dict:
    return {"size": list(image.size), "mode": image.mode,
            "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest()}


def _bind_batch(trees, images, eligible: dict, root: Path, preparation, protocol: dict,
                algorithm: str, native: bool) -> dict:
    actors = {a.problem_id: a for a in eligible["actors"]}
    ids = [t.problem_id for t in trees]
    _require(1 <= len(ids) <= protocol["batch"]["problems_per_step"] and len(set(ids)) == len(ids)
             and set(images) == set(ids) and set(ids) <= actors.keys(), "actual batch is not exclusion allowlisted")
    bound = []
    prompts = load_prompts(_HERE / "prompts_v1.yaml")
    _require(prompts["sha256"] == protocol["prompts_sha256"], "prompts differ from protocol")
    for tree in trees:
        actor = actors[tree.problem_id]
        originals = load_images(actor, root / "images")
        try:
            actual = images[tree.problem_id]
            if algorithm == "CTL-BLIND":
                _require(len(actual) == len(originals) and all(a.size == b.size and a.mode == "RGB"
                         and a.getextrema() == ((128, 128),) * 3 for a, b in zip(actual, originals, strict=True)),
                         "blind input differs from the declared same-size gray control")
            else:
                _require([_image(im) for im in originals] == [_image(im) for im in actual],
                         "actual image bytes/pixels/order differ from approved actor")
            if native:
                request = RolloutRequest("G13:" + actor.problem_id, None,
                    [user_message(turn1_text(actor, prompts), len(actual))], actual, 1, 1, [3407])
                expected = preparation.prepare(request).token_ids
                first = [s for group in tree.groups if group.kind == "T1" for s in group.samples]
                _require(bool(first) and all(s.prompt_token_ids == expected for s in first),
                         "actual first-turn token prompt differs from approved actor")
            bound.append({"problem_id": actor.problem_id, "actor_sha256": checkpoint.digest(to_json(actor)),
                "tree_sha256": checkpoint.digest(to_json(tree)), "source_images_sha256": list(actor.image_sha256),
                "actual_images": [_image(im) for im in actual]})
        finally:
            for original in originals:
                original.close()
    return {"rows": bound, "input_binding": eligible["input_binding"], "native_prompt_checked": native}


def _denied(action, name: str, record) -> dict:
    try:
        value = action()
    except PermissionError as exc:
        result = {"path": name, "error_type": type(exc).__name__, "message": str(exc)}
        record({"kind": "G13_DENIED_PROBE", **result})
        return result
    else:
        if hasattr(value, "close"):
            value.close()  # A failed open probe never reads private bytes.
        raise ValueError("G13: forbidden path was not denied: " + name)


def _actor_guards(root: Path, request: dict, forbidden_roots: list[Path], record) -> list[dict]:
    _require(bool(forbidden_roots) and all(type(p) is Path or isinstance(p, Path) for p in forbidden_roots)
             and all(p.is_absolute() for p in forbidden_roots), "explicit absolute benchmark roots required")
    vaults = [root / split / name for split in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
    vaults += [Path(request[key]["path"]) for key in ("index", "evidence")]
    guards.assert_no_teacher_imports()
    if not guards._installed:
        guards.install_guards(forbidden_roots=forbidden_roots, vault_files=vaults,
                              allow_vault=False, allow_benchmark=False)
    results = [_denied(lambda p=p: p.open("rb"), "private:" + str(p), record) for p in vaults]
    results += [_denied(lambda p=p: (p / "__g13_denied_probe__").open("rb"),
                        "benchmark:" + str(p), record) for p in forbidden_roots]
    names = [n for n in guards._TEACHERS if not (n == "openai" and guards._vllm_sdk_methods)]
    results += [_denied(lambda n=n: __import__(n), "import:" + n, record) for n in names]
    # A dotted importlib call may fail at an absent parent before consulting its
    # child finder. Top-level names exercise the installed finder unambiguously.
    results += [_denied(lambda n=n: importlib.import_module(n), "importlib:" + n, record)
                for n in names if "." not in n]
    for cls, method in guards._vllm_sdk_methods:
        # No client exists: the blocked method is invoked before provider/network work.
        action = (lambda cls=cls: cls()) if method == "__init__" else (lambda cls=cls: cls.request(object()))
        results.append(_denied(action, "schema-client:" + cls.__name__ + "." + method, record))
    guards.assert_no_teacher_imports()
    return results


@contextmanager
def _measure_guards(bundle, optimizer, scheduler, preparation, protocol: dict, *, build_root: Path,
                    data_request: dict, forbidden_roots: list[Path], algorithm: str, step: int,
                    seed: int, record, _test_data_validator=None):
    """Private authored-data/tiny-CPU seam; never certifies full data or CUDA gates."""
    native = _test_data_validator is None
    result = {"status": "OBSERVING", "native_execution": native, "training_allowed": False,
              "gpu_gates_certified": False, "data_eligibility_certified": False}
    try:
        _require(callable(record), "durable raw journal callback required")
        if native:
            _native(bundle, preparation, protocol)
        checkpoint._validate_optimizer(bundle, optimizer, scheduler, protocol, step)
        _require(all(p.grad is None for p in bundle.model.parameters()), "cleared gradient boundary required")
        root = Path(build_root).absolute()
        eligible = _eligibility(root, data_request, protocol, _test_data_validator)
        # Import trusted data readers before installing the one-shot actor checks.
        from PIL import Image
        Image.init()
        probes = _actor_guards(root, data_request, forbidden_roots, record)
        parameters = checkpoint._parameters(bundle)
        before = {n: tensor_sha256(p) for n, p in parameters.items()}
        adam_before = {n: float(optimizer.state[p]["step"]) if p in optimizer.state else 0.
                       for n, p in parameters.items()}
        protocol_sha = checkpoint.digest(protocol)
        calls: list[dict] = []
        optimizer_calls: list[bool] = []
        blocked: list[str] = []
        teacher_attempts: list[str] = []
        original = updates.optimizer_step
        _require(original.__module__ == updates.__name__ and original.__name__ == "optimizer_step",
                 "unmodified production optimizer_step required")
        signature = inspect.signature(original)

        def deny(name):
            def reject(*args, **kwargs):
                blocked.append(name)
                raise PermissionError("G13: CE/SFT path blocked: " + name)
            return reject

        def observed(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            values = arguments.arguments
            _require(not calls and values["bundle"] is bundle and values["optimizer"] is optimizer
                     and values["scheduler"] is scheduler and values["preparation"] is preparation
                     and values["protocol"] is protocol and values["algorithm"] == algorithm
                     and values["step"] == step and values["seed"] == seed,
                     "exactly one bound existing production update required")
            bound = _bind_batch(values["trees"], values["images"], eligible, root, preparation,
                                protocol, algorithm, native)
            calls.append({"batch": bound})
            record({"kind": "G13_UPDATE_BEGIN", **bound})
            update = original(*args, **kwargs)
            _require(update["status"] == "UPDATED_UNSYNCHRONIZED" and update["step_after"] == step + 1,
                     "actual production RL update required; no-op cannot pass")
            calls[0]["update"] = update
            record({"kind": "G13_UPDATE_RETURN", "update": update})
            return update

        with ExitStack() as stack:
            surfaces = [(torch.nn.functional, "cross_entropy"), (torch._C._nn, "cross_entropy_loss"),
                        (train, "AssistantCollator"), (train, "optimizer_update"), (train, "main")]
            for owner, name in surfaces:
                stack.enter_context(patch.object(owner, name, deny(name)))
            logits, target = torch.zeros((2, 2)), torch.tensor([0, 1])
            probes.append(_denied(lambda: torch.nn.functional.cross_entropy(logits, target), "CE", record))
            probes.append(_denied(lambda: torch.nn.CrossEntropyLoss()(logits, target), "CE-module", record))
            for owner, name in surfaces[1:]:
                probes.append(_denied(lambda owner=owner, name=name: getattr(owner, name)(), name, record))
            self_test_count = len(blocked)
            original_teacher, original_sdk = guards._teacher, guards._deny_sdk_call

            def teacher(name):
                denied = original_teacher(name)
                if denied:
                    teacher_attempts.append(name)
                    record({"kind": "G13_BLOCKED_TEACHER_ATTEMPT", "module": name})
                return denied

            def sdk_call(*args, **kwargs):
                teacher_attempts.append("schema-client")
                record({"kind": "G13_BLOCKED_TEACHER_ATTEMPT", "module": "schema-client"})
                return original_sdk(*args, **kwargs)

            stack.enter_context(patch.object(guards, "_teacher", teacher))
            if guards._vllm_sdk_methods:
                stack.enter_context(patch.object(guards, "_deny_sdk_call", sdk_call))
                for cls, method in guards._vllm_sdk_methods:
                    stack.enter_context(patch.object(cls, method, sdk_call))
            handle = optimizer.register_step_post_hook(lambda *args: optimizer_calls.append(True))
            stack.callback(handle.remove)
            stack.enter_context(patch.object(updates, "optimizer_step", observed))
            yield result
            _require(len(calls) == len(optimizer_calls) == 1 and "update" in calls[0],
                     "one observed production call and actual optimizer step required")
            _require(len(blocked) == self_test_count, "forbidden CE/SFT attempted during guarded work")
            _require(not teacher_attempts, "forbidden teacher/API use attempted during guarded work")
            _require(checkpoint.digest(protocol) == protocol_sha, "protocol changed during guard measurement")
            checkpoint._validate_optimizer(bundle, optimizer, scheduler, protocol, step + 1)
            policy.assert_base_unchanged(bundle)
            after = {n: tensor_sha256(p) for n, p in parameters.items()}
            advanced = {n: float(optimizer.state[p]["step"]) - adam_before[n]
                        if p in optimizer.state else 0. for n, p in parameters.items()}
            _require(before != after and all(value in (0., 1.) for value in advanced.values())
                     and any(value == 1 for value in advanced.values()), "parameters/AdamW did not advance once")
            _require(all(p.grad is None for p in bundle.model.parameters()), "update left gradients")
            guards.assert_no_teacher_imports()
            _require(_eligibility(root, data_request, protocol, _test_data_validator) == eligible,
                     "data eligibility changed during measurement")
            result.update(status="MEASURED_ONE_RL_UPDATE", exclusion_allowlist_only=True, ce_patch_raises=True,
                teacher_import_denied=True, data_eligibility_certified=native, input_binding=eligible["input_binding"],
                code_sha256=eligible["code_sha256"], actor_ids=[r["problem_id"] for r in calls[0]["batch"]["rows"]],
                batch=calls[0]["batch"], denied_probes=probes, adam_step_deltas=advanced,
                changed_parameters=[n for n in before if before[n] != after[n]], update=calls[0]["update"],
                update_return_requires_sampler_sync=True, sampler_sync_observed=False,
                schema_only_exception=bool(guards._vllm_sdk_methods),
                process_id=os.getpid(), limitations=[
                    "Python audit/patch checks, not an OS/native-code or network sandbox",
                    "Data worker checks existing full-exclusion/human declarations, not their independent authenticity",
                    "One caller-owned update; no sampler synchronization, training/spending approval, or rollback",
                    "Private authored-data/tiny-CPU seam cannot certify native data or pinned 4B CUDA"])
            record({"kind": "G13_MEASURED", "measurement": result})
    except BaseException as exc:
        result.update(status="FAILED", error_type=type(exc).__name__)
        if callable(record):
            record({"kind": "G13_ERROR", "error_type": type(exc).__name__, "message": str(exc)})
        raise


def measure_guards(bundle, optimizer, scheduler, preparation, protocol: dict, *, build_root: Path,
                   data_request: dict, forbidden_roots: list[Path], algorithm: str, step: int,
                   seed: int, record):
    """Wrap the caller's shared G4/G11 call; no model load or extra optimizer step."""
    return _measure_guards(bundle, optimizer, scheduler, preparation, protocol, build_root=build_root,
        data_request=data_request, forbidden_roots=forbidden_roots, algorithm=algorithm, step=step,
        seed=seed, record=record)
