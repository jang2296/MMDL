"""Measured CUDA/NF4 readiness; this command does not perform RL or claim improvement."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .checkpoint import file_hash
from .dataset import _bytes, _write
from .local_profile import build_protocol

LONG_HISTORY_VERSION = "local-nf4-long-history-backward-calibration-v1"
MAX_SECONDS = 3600


def history_parent(parent: Path, proof: Path, proof_sha256: str, actors: list,
                   model_path: Path, problem_id: str):
    """Consume the pinned terminal auditor receipt, never a live/prefix capture."""
    from . import checkpoint
    from .local_eval import read, wire
    from .local_run import owner_alive, read_actor_descriptor
    from .local_train import training_binding
    from .schema import ProblemTree, from_json, to_json

    if file_hash(proof) != proof_sha256:
        raise ValueError("calibration: parent audit hash differs")
    audit = read(proof)
    launch, result = read(parent / "launch.json"), read(parent / "result.json")
    ids = [actor.problem_id for actor in actors]
    protocol = launch["protocol"]
    if (audit.get("status") != "PASS_TERMINAL_TEN_CPU_SAVED_RESPONSE_REPLAY"
            or Path(audit.get("capture", "")).resolve() != parent.resolve()
            or audit.get("capture_pid_absent") is not True
            or type(launch.get("pid")) is not int or launch["pid"] <= 1
            or audit.get("capture_pid") != launch["pid"] or owner_alive(launch)
            or audit.get("capture_result") != result or result.get("attempted_ids") != ids
            or result.get("status") != "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"
            or audit.get("fixed_ten_training_complete") is not True
            or audit.get("exact_replayed_trees") != 10 or audit.get("unclosed_groups") != 0
            or audit.get("protocol_sha256") != protocol["sha256"]
            or checkpoint.digest({k: v for k, v in protocol.items() if k != "sha256"}) != protocol["sha256"]
            or len(ids) != 10 or len(set(ids)) != 10 or problem_id not in ids
            or launch.get("problem_ids") != ids or launch.get("standalone_base") is not True
            or launch.get("execution_contract", {}).get("stage") != "FIXED_TEN_FROM_FRESH_F0"
            or Path(launch.get("model_path", "")).resolve() != model_path.resolve()):
        raise ValueError("calibration: closed audited fixed-ten parent with absent PID required")
    preparation = Path(launch["preparation"])
    prepared = read(preparation)
    contract = launch["execution_contract"]
    binding = training_binding(prepared, protocol, contract)
    binding["code_sha256"] = launch["binding"]["code_sha256"]
    if (prepared.get("local_protocol_sha256") != protocol["sha256"]
            or launch.get("preparation_sha256") != checkpoint.digest(prepared)
            or launch.get("execution_contract_sha256") != checkpoint.digest(contract)
            or launch["binding"] != binding
            or read_actor_descriptor(prepared["execution_actors"]) != actors):
        raise ValueError("calibration: parent actor/input binding differs")
    raw_paths = sorted(parent.glob("raw-*.json"))
    if not raw_paths or [p.name for p in raw_paths] != [f"raw-{i:04d}.json" for i in range(len(raw_paths))]:
        raise ValueError("calibration: contiguous complete parent journal required")
    paths = [parent / "launch.json", parent / "result.json", preparation,
             Path(prepared["execution_actors"]["path"]), *raw_paths]
    hashes = audit["read_metadata_sha256"]
    if any(hashes.get(str(path)) != file_hash(path) for path in paths):
        raise ValueError("calibration: audited parent bytes changed or unbound")
    trees, groups, observations = [], {}, {}
    for path in raw_paths:
        record = read(path)
        raw = record["raw"]
        if record["kind"] == "TREE":
            trees.append(from_json(ProblemTree, wire(raw).decode()))
        elif record["kind"] == "GROUP":
            if raw["group_id"] in groups:
                raise ValueError("calibration: duplicate parent group")
            groups[raw["group_id"]] = raw
        elif record["kind"] == "OBSERVATION" and raw.get("phase") == "COMPLETED":
            if raw["request_id"] in observations:
                raise ValueError("calibration: duplicate parent observation")
            observations[raw["request_id"]] = {k: v for k, v in raw.items() if k != "phase"}
        elif record["kind"] != "OBSERVATION" or raw.get("phase") != "STARTED":
            raise ValueError("calibration: unknown parent journal event")
    expected_groups = {g.group_id: json.loads(to_json(g)) for tree in trees for g in tree.groups}
    expected_observations = {value["request_id"]: value for tree in trees
        if (value := tree.notebook.get("rewarded_notebook", {}).get("fresh_observation")) is not None}
    if ([tree.problem_id for tree in trees] != ids or len(expected_groups) != len(groups)
            or groups != expected_groups or observations != expected_observations):
        raise ValueError("calibration: exact complete parent TREE/GROUP/OBSERVATION coverage required")
    return trees[ids.index(problem_id)], {"files": {str(p): hashes[str(p)] for p in paths}
        | {str(proof): proof_sha256}, "protocol_sha256": protocol["sha256"],
        "history_policy_version": trees[ids.index(problem_id)].policy_version}


def history_request(actor, tree, images, prompts, engine, verifier):
    from . import episode_notebook as notebook, fresh_observation
    from .engines import RolloutRequest
    from .prompts import for_actor, p0_body, retry_text, turn1_text, user_message
    from .rollout import continuation_ids, seeds

    audit = notebook.audit_episode(tree, actor, verifier, routing_rule=notebook.STRICT_ROUTE_RULE)
    if not audit["complete_episode"] or audit["success_attempt"] is None:
        raise ValueError("calibration: complete actual history including a successful retry required")
    schema_prompts = for_actor(actor, prompts)
    schema = schema_prompts["observation_instruction"][actor.family]
    initial = engine.prepare(RolloutRequest("calibration-original-input", None,
        [user_message(turn1_text(actor, prompts), len(images))], images, 1, 1, [3407]))
    prior = {}
    for sample in notebook.solution_samples(tree):
        if engine.tokenizer.decode(sample.completion_token_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False) != sample.text:
            raise ValueError("calibration: original history token/text differs")
        if sample.turn == 1:
            expected = initial.token_ids
        else:
            parent = prior[sample.parent_sample_id]
            expected = continuation_ids(parent, retry_text(parent.result.status, parent.result.extracted,
                actor.mode, actor.family, schema_prompts), engine.tokenizer)
        if sample.prompt_token_ids != expected:
            raise ValueError("calibration: original history image/prompt/branch differs")
        prior[sample.sample_id] = sample
    reread = tree.notebook["rewarded_notebook"]["fresh_observation"]
    fresh_observation.validate(reread, policy_version=tree.policy_version, tokenizer=engine.tokenizer,
                               actor=actor, schema=schema, master_seed=3407)
    observed = engine.prepare(fresh_observation.request_for(actor, schema, images, tree.policy_version, 3407))
    if observed.token_ids != reread["prompt_token_ids"] or observed.image_key != reread["image_key"]:
        raise ValueError("calibration: original reread image/prompt differs")
    body = fresh_observation.quoted_context(reread) + notebook.instruction(
        notebook.solution_samples(tree), audit, schema)
    request = RolloutRequest(f"{actor.problem_id}:{LONG_HISTORY_VERSION}", None,
        [user_message(p0_body(actor.question, actor.choices) + "\n\n" + body, len(images))],
        images, 1, 2048, seeds(actor.problem_id, 0, 1, LONG_HISTORY_VERSION, 1))
    return request, audit


def history_backward(bundle, inputs: dict, completion_len: int) -> dict:
    """One signed generated-action objective; no labels, rewards, or optimizer."""
    import torch

    from .gate_gradients import _capture
    from .policy import completion_logprobs

    if (set(bundle.params) != {"L", "M", "V"} or any(not ps for ps in bundle.params.values())
            or any(p.grad is not None for p in bundle.model.parameters())):
        raise ValueError("calibration: complete clean L/M/V trainable groups required")
    bundle.route.apply(True)
    names = {id(p): name for name, p in bundle.model.named_parameters()}
    try:
        bundle.model.train()
        with torch.enable_grad():
            values = completion_logprobs(bundle, inputs, completion_len, chunk=64)
            coefficients = torch.tensor([1. if i % 2 == 0 else -1. for i in range(completion_len)],
                                        device=values.device)
            objective = (values[0] * coefficients).sum() / completion_len
            if not bool(torch.isfinite(objective)):
                raise ValueError("calibration: nonfinite signed generated-action objective")
            objective.backward()
        summary, _ = _capture(bundle, names, set())
        if any(not row["all_present"] or row["norm"] <= 0 for row in summary.values()):
            raise ValueError("calibration: finite present nonzero L/M/V gradients required")
        return {"objective": float(objective.detach()), "gradients": summary,
                "frozen_gradients_none": True, "temporary_gradients_cleared": True}
    finally:
        bundle.model.zero_grad(set_to_none=True)


def long_history_probe(preparation: Path, model_path: Path, cache_check: Path, parent: Path,
                       parent_audit: Path, parent_audit_sha256: str, problem_id: str, output: Path) -> dict:
    import torch

    from . import approval, checkpoint
    from .dataset import load_images
    from .fact_context_probe import output_path
    from .input_preparation import tensor_identity
    from .local_eval import read, save
    from .local_profile import local_prompts
    from .local_run import (_install_actor_guards, cache_readiness, host_identity, make_probes,
                            parameter_hashes, read_preparation)
    from .policy import enable_deterministic_execution
    from .quantized_cache import block_original_access
    from .quantized_policy import assert_quantized_base_unchanged, load_quantized_policy
    from .shared_policy_engine import SharedPolicyRolloutEngine
    from .trainer import synchronize
    from .verifier_service import VerifierService

    if int(os.environ.get("MMDL_CALIBRATION_SUPERVISOR", "0")) != os.getppid():
        raise ValueError("calibration: use the supervised CLI")
    began = time.monotonic()
    output = output_path(output, model_path, cache_check, parent, parent_audit, preparation,
                         Path(read(preparation)["build_root"]))
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    result = {"status": "INCOMPLETE", "pid": os.getpid(), "host": host_identity(), "optimizer_updates": 0,
        "training_allowed": False, "full_24k_sequence_verified": False, "optimizer_fit_verified": False,
        "notebook_quality_verified": False, "self_improvement_verified": False}
    bundle = None
    try:
        protocol = build_protocol(verified_note_fact_credit=True, context_limit=24576)
        prepared, data_request, actors = read_preparation(preparation, protocol)
        readiness = cache_readiness(cache_check, model_path, protocol)
        tree, parent_evidence = history_parent(parent, parent_audit, parent_audit_sha256,
                                               actors, model_path, problem_id)
        actor = next(a for a in actors if a.problem_id == problem_id)
        sources = approval.code_hashes()
        diagnostic = {"protocol_id": LONG_HISTORY_VERSION, "runtime_protocol_sha256": protocol["sha256"],
            "parent_protocol_sha256": parent_evidence["protocol_sha256"], "context_limit": 24576,
            "n": 1, "max_new_tokens": 2048, "max_seconds": MAX_SECONDS, "arm": "MOD-RV",
            "history_policy_version": tree.policy_version, "generation_policy_version": 0,
            "objective": "alternating signed logprob of the actual generated suffix; no reward or labels",
            "optimizer_updates": 0, "repairs": False}
        save(output / "launch.json", {"pid": os.getpid(), "host": host_identity(), "supervisor_pid": os.getppid(),
            "protocol": protocol, "diagnostic": diagnostic, "diagnostic_sha256": checkpoint.digest(diagnostic),
            "sources": sources, "parent": parent_evidence, "problem_id": problem_id,
            "preparation_sha256": checkpoint.digest(prepared), "cache_readiness_sha256": checkpoint.digest(readiness),
            "deterministic_execution": enable_deterministic_execution()})
        forbidden = _install_actor_guards(prepared, data_request)
        access = block_original_access(Path(readiness["original_model_root"]))
        result["original_access_guard"] = access
        torch.manual_seed(3407)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with ExitStack() as stack:
            root = Path(prepared["build_root"])
            verifier = stack.enter_context(VerifierService(root / "train", forbidden_roots=forbidden, timeout=60))
            if verifier.metadata["vault_manifest_sha256"] != read(root / "manifest.yaml")["split_manifests"]["train"]:
                raise ValueError("calibration: verifier vault changed")
            bundle = load_quantized_policy(model_path, "MOD-RV", protocol)
            before = parameter_hashes(bundle)
            engine = SharedPolicyRolloutEngine(bundle, protocol["image"], 24576, logprob_chunk=64)
            images = {a.problem_id: load_images(a, root / "images") for a in actors}
            for values in images.values():
                for image in values:
                    stack.callback(image.close)
            prompts = local_prompts(protocol)
            save(output / "initial_sync.json", synchronize(bundle, engine, engine,
                make_probes(actors, images, engine, prompts), protocol, 0, push=False))
            request, audit = history_request(actor, tree, images[problem_id], prompts, engine, verifier)
            actual = engine.prepare(request)
            save(output / "request.json", {"request_id": request.request_id, "messages": request.messages,
                "seeds": request.seeds, "n": request.n, "max_new_tokens": request.max_new_tokens,
                "prompt_token_ids": actual.token_ids, "image_paths": actor.image_paths,
                "image_sha256": actor.image_sha256, "image_key": actual.image_key,
                "pixel_tensors": {k: tensor_identity(v) for k, v in actual.pixels.items()},
                "image_spans": actual.image_spans, "episode_audit": audit})
            torch.cuda.synchronize()
            started = time.monotonic()
            generated = engine.sample([request])
            torch.cuda.synchronize()
            result["generation_seconds"] = time.monotonic() - started
            save(output / "generation.json", [asdict(row) for row in generated])
            if (len(generated) != 1 or generated[0].request_id != request.request_id
                    or generated[0].prompt_token_ids != actual.token_ids or len(generated[0].completions) != 1):
                raise ValueError("calibration: exact single notebook generation required")
            completion = generated[0].completions[0]
            if not completion.token_ids or engine.tokenizer.decode(completion.token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) != completion.text:
                raise ValueError("calibration: generated suffix token/text differs")
            result.update(prompt_tokens=len(actual.token_ids), completion_tokens=len(completion.token_ids),
                actual_sequence_tokens=len(actual.token_ids) + len(completion.token_ids), finish_reason=completion.finish_reason)
            inputs = engine.model_inputs(request, completion.token_ids, next(bundle.model.parameters()).device)
            if inputs["input_ids"].tolist() != [actual.token_ids + completion.token_ids]:
                raise ValueError("calibration: backward must use the exact generated sequence")
            torch.cuda.synchronize()
            started = time.monotonic()
            result.update(history_backward(bundle, inputs, len(completion.token_ids)))
            torch.cuda.synchronize()
            result["backward_seconds"] = time.monotonic() - started
            assert_quantized_base_unchanged(bundle)
            if (before != parameter_hashes(bundle) or sources != approval.code_hashes()
                    or any(file_hash(Path(p)) != h for p, h in parent_evidence["files"].items())
                    or access["denied_original_access_attempts"] != 0):
                raise ValueError("calibration: source, parent, base or trainables changed")
            result.update(status="ACTUAL_LONG_HISTORY_GENERATION_BACKWARD_VERIFIED", trainables_unchanged=True,
                base_unchanged=True, source_unchanged=True, trainable_sha256=before,
                frozen_base_sha256=bundle.base_sha256)
    except BaseException as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if bundle is not None:
            bundle.model.zero_grad(set_to_none=True)
        result.update(seconds=time.monotonic() - began)
        if torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(output / "result.json", result)
    return result


def supervised_history(args) -> int:
    from fine_tuning.budget import terminate_group

    previous, process = {}, None
    try:
        def interrupted(signum, _):
            for pending in (signal.SIGINT, signal.SIGTERM):
                signal.signal(pending, signal.SIG_IGN)
            raise SystemExit(128 + signum)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        names = ("preparation", "model_path", "cache_check", "parent", "parent_audit",
                 "parent_audit_sha256", "problem_id", "output")
        if args.worker:
            long_history_probe(**{name: getattr(args, name) for name in names})
            return 0
        command = [sys.executable, "-B", "-m", "fine_tuning.notebook_rl.local_calibrate", "long-history", "--worker"]
        for name in names:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        process = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
            env=dict(os.environ, MMDL_CALIBRATION_SUPERVISOR=str(os.getpid())))
        return process.wait(timeout=MAX_SECONDS - 30)
    except subprocess.TimeoutExpired:
        return 124
    finally:
        if process is not None:
            terminate_group(process, grace_seconds=10)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def verify_original(model_path: Path) -> dict:
    manifest = json.loads((Path(__file__).parent / "resources/baseline.json").read_text())
    if model_path.name != manifest["revision"]:
        raise ValueError("calibration: pinned original snapshot required")
    checked = {}
    for row in manifest["files"]:
        path = model_path / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError(f"calibration: original model mismatch: {row['path']}")
        checked[row["path"]] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    return checked


def kernel_probe() -> dict:
    import bitsandbytes as bnb
    import torch

    torch.manual_seed(3407)
    layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16,
                            compress_statistics=True, quant_type="nf4").bfloat16().cuda()
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    if (x.grad is None or not bool(torch.isfinite(x.grad).all()) or not bool(x.grad.abs().sum() > 0)
            or layer.weight.grad is not None):
        raise RuntimeError("NF4 CUDA forward/backward probe failed")
    return {"status": "KERNEL_FORWARD_BACKWARD_VERIFIED", "shape": list(y.shape),
            "input_grad_norm": float(x.grad.float().norm()),
            "weight_grad": None, "pretrained_model_test": False, "optimizer_updates": 0}


def policy_probe(model_path: Path, output: Path) -> dict:
    import torch

    from .quantized_policy import assert_quantized_base_unchanged, build_quantized_policy

    _write(output / "original_files.json", _bytes(verify_original(model_path)) + b"\n")
    protocol = build_protocol()
    _write(output / "protocol.json", _bytes(protocol) + b"\n")
    torch.manual_seed(protocol["local_budget"]["seed"])
    bundle = build_quantized_policy(model_path, "MOD-RV", protocol)
    assert_quantized_base_unchanged(bundle)
    _write(output / "inventory.json", _bytes(bundle.inventory) + b"\n")
    return {"status": "PRETRAINED_NF4_POLICY_LOADED", "optimizer_updates": 0,
            "trainable_parameters": {k: sum(p.numel() for p in ps) for k, ps in bundle.params.items()},
            "quantized_modules": len(bundle.inventory["quantization"]["modules"]),
            "generation_verified": False, "training_verified": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("kernel", "policy", "long-history"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    for name in ("preparation", "cache-check", "parent", "parent-audit"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--parent-audit-sha256")
    parser.add_argument("--problem-id")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stage == "long-history":
        if any(getattr(args, name) is None for name in ("model_path", "preparation", "cache_check",
                "parent", "parent_audit", "parent_audit_sha256", "problem_id")):
            parser.error("long-history requires preparation/model/cache and a pinned terminal parent audit/problem ID")
        return supervised_history(args)
    if args.stage == "policy" and args.model_path is None:
        parser.error("policy requires --model-path")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    import torch

    from .local_run import host_identity

    began = time.monotonic()
    result = {"status": "INCOMPLETE", "stage": args.stage, "pid": os.getpid(), "host": host_identity(),
              "versions": {k: version(k) for k in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
              "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
    torch.cuda.reset_peak_memory_stats()
    code = 0
    try:
        result.update(kernel_probe() if args.stage == "kernel" else policy_probe(args.model_path, args.output))
    except Exception as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        code = 1
    finally:
        result.update(seconds=time.monotonic()-began,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        _write(args.output / "result.json", _bytes(result) + b"\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
