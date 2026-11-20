"""Bounded native gate sequence and fresh-process controls, not training approval.

Two sequential owned children execute 0->1 (shared gates), then 1->2 (capture),
and replay 1->2 from saved step1. Separate export/cache children consume saved
step1, never live step2. The parent never initializes CUDA or grants approval.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from . import approval, launch, measure
from .dataset import SPLITS, _bytes, _sha, _write
from .schema import ActorItem, from_json, load_protocol, to_json
from .vault import _read, _safe_root

FORMAT = "notebook-rl-gate-session-job-v2"
PURPOSE = "BOUNDED_RL_GATE_MEASUREMENT_ONLY"
KEYS = measure.KEYS | {"arm", "sync_probes", "probe_scope"}
COMPLETE = "GPU_COMPONENTS_MEASURED_NOT_TRAINING_APPROVAL"


def _require(value, message):
    if not value:
        raise ValueError("gate session: " + message)


def _actors(validated):
    return [a if type(a) is ActorItem else from_json(ActorItem, _bytes(a).decode()) for a in validated["actors"]]


def _validation_hash(validated):
    return _sha(_bytes(validated | {"actors": [json.loads(to_json(a)) for a in _actors(validated)]}))


def validate_job(job: dict, protocol: dict) -> None:
    _require(type(job) is dict and set(job) == KEYS and job["format"] == FORMAT,
             "exact bounded gate job required")
    _require(job["arm"] in protocol["arms"] and job["arm"] != "MOD-RS",
             "supported fixed module arm required")
    _require(type(job["runtime_scope"]) is dict and job["runtime_scope"].get("purpose") == PURPOSE,
             "explicit bounded RL update purpose required; frozen-F0 authority is insufficient")
    # Reuse path/device syntax checks only, never its frozen-weight execution.
    syntax = {k: job[k] for k in measure.KEYS}
    syntax.update(format=measure.FORMAT, runtime_scope=job["runtime_scope"] |
                  {"purpose": "FROZEN_WEIGHT_MEASUREMENT_ONLY"})
    measure.validate_job(syntax, protocol)
    source = launch.descriptor(job["sync_probes"]).resolve()
    output = Path(job["output"]).resolve()
    _require(not (source.is_relative_to(output) or output.is_relative_to(source)),
             "output overlaps probe input")
    _require(type(job["probe_scope"]) is dict and set(job["probe_scope"]) == {"probes"},
             "explicit single/two-image probe bindings required")


def _probe_actors(job, validated):
    """Probe images are eligible actors too; a caller cannot add arbitrary data."""
    rows = launch.read_descriptor(job["sync_probes"])
    expected = job["probe_scope"]["probes"]
    _require(set(rows) == {"probes"} and type(rows["probes"]) is list and len(rows["probes"]) == 2
             and type(expected) is list and len(expected) == 2, "two bound probe records required")
    eligible = {a.problem_id: json.loads(to_json(a)) for a in _actors(validated)}
    result, identities = {}, []
    for row in rows["probes"]:
        _require(type(row) is dict and set(row) == {"id", "actor", "completion_ids"}
                 and isinstance(row["id"], str) and bool(row["id"]), "malformed probe record")
        actor = from_json(ActorItem, _bytes(row["actor"]).decode())
        _require(eligible.get(actor.problem_id) == row["actor"], "probe actor is outside eligible pool")
        tokens = row["completion_ids"]
        _require(type(tokens) is list and 2 <= len(tokens) <= 32
                 and all(type(t) is int and t >= 0 for t in tokens), "bounded literal probe tokens required")
        identities.append({"id": row["id"], "image_count": len(actor.image_paths), "input_sha256": _sha(_bytes(row))})
        result[row["id"]] = actor
    _require(len(result) == 2 and sorted(p["image_count"] for p in identities) == [1, 2]
             and expected == identities, "distinct bound single/two-image probes required")
    return result


def preflight(job: dict, protocol: dict) -> dict:
    validate_job(job, protocol)
    result = approval.validate_for_gate_data(Path(job["build_root"]),
        **{k: job[k] for k in (*measure.DESCRIPTORS, "expected_measurement_scope")})
    if result["status"] == "VALIDATED_FOR_MEASUREMENT":
        _probe_actors(job, result)
    return result


def _eligible(result):
    return (result.get("status") == "VALIDATED_FOR_MEASUREMENT"
            and result.get("measurement_allowed") is True and result.get("training_allowed") is False)


def _shared_step(bundle, optimizer, scheduler, engine, preparation, actors, structures, loader,
                 verifier, prompts, protocol, probes, *, job, record, should_stop):
    """Observe one and the same production update; no extra G13 optimizer call."""
    from .gate_guards import measure_guards
    from .gate_throughput import measure_step

    with measure_guards(bundle, optimizer, scheduler, preparation, protocol,
            build_root=Path(job["build_root"]),
            data_request={k: job[k] for k in (*measure.DESCRIPTORS, "expected_measurement_scope")},
            forbidden_roots=[Path(p) for p in job["forbidden_roots"]], algorithm="ALG-NB3",
            step=0, seed=job["seed"], record=record) as guard:
        result = measure_step(bundle, optimizer, scheduler, engine, preparation, actors, structures,
            loader, verifier, prompts, protocol, probes, algorithm="ALG-NB3", step=0, seed=job["seed"],
            should_stop=should_stop, record=record)
        _require(result["status"] == "STEP_MEASURED_NOT_GPU_APPROVAL"
                 and result["longest_sample_step_complete"] is True and result["no_oom"] is True
                 and result["step_after"] == 1 and result["engine_sync_required"] is False,
                 "G11 incomplete; preserve raw records, do not continue to G12")
    _require(guard["status"] == "MEASURED_ONE_RL_UPDATE" and guard["native_execution"] is True
             and result["native_execution"] is True, "native finalized shared measurement required")
    record({"kind": "SHARED_G4_G11_G13_COMPLETED", "G11": result, "G13": guard})
    return {"G11": result, "G13": guard}


def _capture(job, protocol, validated, runtime, output, binding, record):
    import random

    import numpy as np
    import torch

    from . import batches, checkpoint, gate_inputs, gate_sequence
    from .dataset import load_images
    from .engines import InputPreparation, VLLMRolloutEngine
    from .guards import prepare_vllm_schema_dependency
    from .policy import build_policy
    from .prompts import load_prompts
    from .verifier_service import VerifierService

    root, model = Path(job["build_root"]), Path(job["model_path"])
    prompts = load_prompts(launch.HERE/"prompts_v1.yaml")
    _require(prompts["sha256"] == protocol["prompts_sha256"], "prompt bytes changed")
    probe_actors = _probe_actors(job, validated)
    actors = {a.problem_id: a for a in _actors(validated)}
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    record({"kind": "SCHEMA_DEPENDENCY", "measurement": prepare_vllm_schema_dependency()})
    bundle = build_policy(model, job["arm"], protocol)
    bundle.model.to(f"cuda:{runtime['device']['index']}")
    optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
    cfg = protocol["engine"]["vllm"]
    def should_stop():
        return time.time() >= runtime["deadline_timestamp"]
    def loader(actor):
        return load_images(actor, root/"images")
    with ExitStack() as stack:
        preparation = InputPreparation(bundle.processor, bundle.model.config, protocol["image"], protocol["context_limit"])
        probes = launch.load_probes(job["sync_probes"], job["probe_scope"], root/"images", prompts, stack)
        measured = gate_sequence.before_engine(bundle, preparation, probes, model, protocol,
            record=record, should_stop=should_stop)
        engine = VLLMRolloutEngine(model, protocol["image"], cfg["kv_cache_memory_bytes"],
                                  cfg["max_num_seqs"], protocol["context_limit"])
        stack.callback(engine.llm.llm_engine.engine_core.shutdown)
        measured.update(gate_sequence.before_update(bundle, engine, probes, protocol, runtime,
            seed=job["seed"], record=record, should_stop=should_stop))
        verifier = stack.enter_context(VerifierService(root/"train", forbidden_roots=[Path(p) for p in job["forbidden_roots"]]))
        _require(verifier.metadata["vault_manifest_sha256"] == launch.read_json(root/"manifest.yaml")["split_manifests"]["train"],
                 "actual verifier differs from eligible train split")
        shared = _shared_step(bundle, optimizer, scheduler, engine, preparation, actors, validated["structures"],
            loader, verifier, prompts, protocol, probes, job=job, record=record, should_stop=should_stop)
        _write(output/"shared.json", _bytes(shared)+b"\n")
        # Preserve the actual first update even if the later rollout budget is
        # exhausted. This gate cursor is not a full training-resume cursor.
        checkpoint.save(bundle, optimizer, scheduler, 1,
            {"epoch": 0, "position": 0, "discarded_problem_ids": [],
             "scope": "SHARED_GATE_ONLY_NOT_TRAINING_RESUME"}, checkpoint.capture_rng(),
            output/"shared_checkpoint", protocol=protocol, binding=binding)
        _require(not should_stop(), "deadline after shared update")
        measured.update(gate_sequence.after_update(bundle, engine, preparation, probes, model, protocol,
            seed=job["seed"], record=record, should_stop=should_stop))
        _write(output/"components.json", _bytes(measured)+b"\n")
        # G11 has closed its images. Sample a genuinely new version1 batch with
        # freshly loaded originals; never reuse version0 actions after the push.
        order = batches.family_order(list(actors.values()), job["seed"])
        batch = batches.collect_batch(actors, validated["structures"], order, 0, loader, engine, verifier,
            prompts, protocol, "ALG-NB3", should_stop=should_stop, seed=job["seed"],
            on_tree=lambda t: record({"kind": "G12_NEXT_TREE", "tree": to_json(t)}),
            on_group=lambda g: record({"kind": "G12_NEXT_GROUP", "group": to_json(g)}))
        try:
            _require(batch.status == "COLLECTED" and not should_stop(), "G12 next-step batch incomplete")
            packet = gate_inputs.save_inputs(output/"inputs.json", trees=batch.trees, images=batch.images,
                actors=actors, probes=probes, probe_actors=probe_actors, preparation=preparation,
                image_root=root/"images", protocol=protocol, binding=binding, step=1, algorithm="ALG-NB3",
                seed=job["seed"], cursor={"epoch": 0, "position": batch.cursor,
                    "discarded_problem_ids": [r["problem_id"] for r in batch.discarded],
                    "scope": "G12_GATE_BATCH_ONLY_NOT_TRAINING_RESUME", "order_sha256": _sha(_bytes(order))})
            captured = gate_inputs.capture_from_inputs(bundle, optimizer, scheduler, preparation, packet,
                image_root=root/"images", protocol=protocol, binding=binding, model_path=model, output=output/"capture")
            _require(captured["native_execution"] is True and captured["step"] == 1 and scheduler.last_epoch == 2,
                     "capture must preserve step1 before actual second gate update")
        finally:
            for images in batch.images.values():
                for image in images:
                    image.close()
    return {"status": "CAPTURE_COMPLETED", "input_packet": packet, "capture": str(output/"capture"),
            "shared_sha256": _sha(_read(output/"shared.json")), "saved_step": 1,
            "shared_checkpoint": str(output/"shared_checkpoint"),
            "components": {"path": str(output/"components.json"),
                           "file_sha256": _sha(_read(output/"components.json"))},
            "optimizer_steps_executed": 2, "live_final_step": 2, "binding": binding}


def worker(task_descriptor: dict, protocol: dict) -> dict:
    task = launch.read_descriptor(task_descriptor)
    _require(set(task) == {"stage", "job", "validated_sha256", "capture_handoff", "snapshot_handoff", "control_handoff"}
             and task["stage"] in ("capture", "replay", "export", "control", "fresh"), "exact private stage task required")
    job = task["job"]
    checked = preflight(job, protocol)
    _require(_eligible(checked) and _validation_hash(checked) == task["validated_sha256"],
             "data/code changed before child execution")
    binding = launch.checkpoint_binding(checked, protocol, job["commit"])
    previous = None
    if task["stage"] != "capture":
        previous = launch.read_descriptor(task["capture_handoff"])
        _require(previous.get("status") == "CAPTURE_COMPLETED" and previous.get("binding") == binding
                 and previous.get("saved_step") == 1 and previous.get("live_final_step") == 2,
                 "closed bound step1 capture required")
    else:
        _require(task["capture_handoff"] is None, "fresh capture cannot accept prior state")
    _require((task["snapshot_handoff"] is not None) == (task["stage"] in ("control", "fresh"))
             and (task["control_handoff"] is not None) == (task["stage"] == "fresh"),
             "stage-specific snapshot/control handoffs required")
    # The model-free parent never calls this CUDA-initializing check. The new
    # job format deliberately retains training's two-copy + fixed KV floor.
    runtime = (None if task["stage"] == "export" else
               launch.runtime_check(job | {"expected_gpu_scope": job["runtime_scope"]}, protocol))
    output = Path(job["output"])
    _safe_root(output.parent, create=True)
    output.mkdir(mode=0o700, exist_ok=False)
    _write(output/"launch.json", _bytes({"task": task_descriptor, "runtime": runtime, "binding": binding})+b"\n")
    result = {"status": "INCOMPLETE", "binding": binding, "gpu_gates_certified": False,
              "training_allowed": False, "pod_billing_stopped": False}
    try:
        if task["stage"] == "capture":
            assert runtime is not None
            with (output/"journal.jsonl").open("xb") as stream:
                def record(row):
                    stream.write(_bytes(row)+b"\n")
                    stream.flush()
                result.update(_capture(job, protocol, checked, runtime, output, binding, record))
        elif task["stage"] == "replay":
            from .gate_inputs import replay_from_inputs
            from .guards import install_guards, prepare_vllm_schema_dependency

            prepare_vllm_schema_dependency()
            root = Path(job["build_root"])
            install_guards(forbidden_roots=[Path(p) for p in job["forbidden_roots"]],
                vault_files=[root/s/n for s in SPLITS for n in ("verifier_private.jsonl", "lineage.jsonl")]
                    +[launch.descriptor(job[k]) for k in measure.DESCRIPTORS],
                allow_vault=False, allow_benchmark=False)
            assert previous is not None
            assert runtime is not None
            replayed = replay_from_inputs(Path(job["model_path"]), Path(previous["capture"]), output/"replay",
                previous["input_packet"], image_root=root/"images", protocol=protocol, binding=binding,
                device=f"cuda:{runtime['device']['index']}")
            _require(replayed["native_execution"] is True and replayed["step"] == 1, "native step1 reload required")
            result.update(status="REPLAY_COMPLETED", replay=str(output/"replay"), optimizer_steps_executed=1)
        elif task["stage"] == "export":
            from .gate_snapshot import export_step1
            assert previous is not None
            snapshot = export_step1(previous, Path(job["model_path"]), output/"snapshot", job["arm"], binding)
            result.update(status="EXPORT_COMPLETED", snapshot=snapshot)
        else:
            from . import gate_cache, gate_snapshot
            from .guards import install_guards, prepare_vllm_schema_dependency
            from .prompts import load_prompts

            assert previous is not None
            snapshot = launch.read_descriptor(task["snapshot_handoff"])
            _require(snapshot.get("status") == "EXPORT_COMPLETED" and snapshot.get("binding") == binding,
                     "closed bound step1 export required")
            descriptor = snapshot["snapshot"]
            ckpt = Path(previous["shared_checkpoint"])
            gate_snapshot.validate_snapshot(descriptor, binding, ckpt)
            components = launch.read_descriptor(previous["components"])
            prepare_vllm_schema_dependency()
            root = Path(job["build_root"])
            install_guards(forbidden_roots=[Path(p) for p in job["forbidden_roots"]],
                vault_files=[root/s/n for s in SPLITS for n in ("verifier_private.jsonl", "lineage.jsonl")]
                    +[launch.descriptor(job[k]) for k in measure.DESCRIPTORS],
                allow_vault=False, allow_benchmark=False)
            with ExitStack() as stack, (output/"journal.jsonl").open("xb") as stream:
                def record(row):
                    stream.write(_bytes(row)+b"\n")
                    stream.flush()
                probes = launch.load_probes(job["sync_probes"], job["probe_scope"], root/"images",
                    load_prompts(launch.HERE/"prompts_v1.yaml"), stack)
                cfg = protocol["engine"]["vllm"]
                common = dict(kv_cache_memory_bytes=cfg["kv_cache_memory_bytes"],
                    max_num_seqs=cfg["max_num_seqs"], max_model_len=protocol["context_limit"],
                    probes=probes, record=record)
                if task["stage"] == "control":
                    observed = gate_cache.measure_default_control(Path(job["model_path"]), protocol["image"],
                        post_weights=gate_snapshot.iter_tensors(descriptor, binding, ckpt),
                        previous=components["G6_AFTER"], **common)
                else:
                    control = launch.read_descriptor(task["control_handoff"])
                    _require(control.get("status") == "CONTROL_COMPLETED" and control.get("binding") == binding,
                             "closed bound default control required")
                    observed = gate_cache.measure_fresh(launch.descriptor(descriptor).parent, protocol["image"],
                        control=control["measurement"], **common)
                gate_snapshot.validate_snapshot(descriptor, binding, ckpt)
                result.update(status=task["stage"].upper()+"_COMPLETED", measurement=observed)
    except BaseException as exc:
        result.update(status="INCOMPLETE", error_type=type(exc).__name__)
        raise
    finally:
        _write(output/"result.json", _bytes(result)+b"\n")
    return result


def _run_child(task: dict, path: Path, deadline: float) -> subprocess.Popen:
    from fine_tuning.budget import terminate_group

    raw = _bytes(task)+b"\n"
    _write(path, raw)
    child = None
    try:
        _require(time.time() < deadline, "deadline before child launch")
        with path.with_suffix(".log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-m", "fine_tuning.notebook_rl.gate_session", "--worker", "--task", str(path),
                "--task-sha256", _sha(raw)], cwd=launch.ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child.wait(timeout=max(0.01, deadline-time.time()))
        _require(child.returncode == 0, "child failed; inspect preserved private log")
        return child
    finally:
        if child is not None:
            # Also remove residual workers after a leader's successful exit.
            terminate_group(child)


def execute(job: dict, protocol: dict, validated: dict) -> dict:
    from fine_tuning.budget import utc

    import torch

    _require(not torch.cuda.is_initialized(), "coordinator must not own CUDA")
    _require(_eligible(validated) and preflight(job, protocol) == validated, "data eligibility changed or missing")
    output = Path(job["output"])
    _safe_root(output.parent, create=True)
    output.mkdir(mode=0o700, exist_ok=False)
    result = {"status": "INCOMPLETE", "gpu_gates_certified": False, "training_allowed": False,
              "phase_a_completed": False, "model_improvement_verified": False, "pod_billing_stopped": False}
    try:
        common = {"validated_sha256": _validation_hash(validated), "capture_handoff": None,
                  "snapshot_handoff": None, "control_handoff": None}
        capture_job = job | {"output": str(output/"capture-stage")}
        capture_process = _run_child(common | {"stage": "capture", "job": capture_job},
            output/"capture-task.json", utc(job["deadline_utc"]).timestamp())
        capture_result = output/"capture-stage/result.json"
        handoff = {"path": str(capture_result), "file_sha256": _sha(_read(capture_result))}
        captured = launch.read_descriptor(handoff)
        _require(captured.get("status") == "CAPTURE_COMPLETED", "capture did not finish")
        replay_process = _run_child(common | {"stage": "replay", "job": job | {"output": str(output/"replay-stage")},
            "capture_handoff": handoff}, output/"replay-task.json", utc(job["deadline_utc"]).timestamp())
        from .gate_checkpoint import compare_checkpoint
        compared = compare_checkpoint(Path(captured["capture"]), output/"replay-stage/replay",
                                      capture_process, replay_process, output/"comparison")
        _require(compared["native_execution"] is True and compared["passed"] is True, "native roundtrip required")

        def cache_child(stage, **handoffs):
            process = _run_child(common | {"stage": stage, "job": job | {"output": str(output/(stage+"-stage"))},
                "capture_handoff": handoff, **handoffs}, output/(stage+"-task.json"),
                utc(job["deadline_utc"]).timestamp())
            path = output/(stage+"-stage/result.json")
            descriptor = {"path": str(path), "file_sha256": _sha(_read(path))}
            value = launch.read_descriptor(descriptor)
            _require(value.get("status") == stage.upper()+"_COMPLETED", stage+" did not finish")
            return process, descriptor, value

        _, snapshot_handoff, _ = cache_child("export")
        control_process, control_handoff, control = cache_child("control", snapshot_handoff=snapshot_handoff)
        fresh_process, fresh_handoff, fresh = cache_child("fresh", snapshot_handoff=snapshot_handoff,
                                                        control_handoff=control_handoff)
        from . import gate_cache, gate_sequence
        components = launch.read_descriptor(captured["components"])
        shared = launch.read_descriptor({"path": str(output/"capture-stage/shared.json"),
                                         "file_sha256": captured["shared_sha256"]})
        cache = gate_cache.compare_cache(components["G6_BEFORE"], components["G6_AFTER"],
            control["measurement"], fresh["measurement"], threshold=protocol["engine"]["probe_max_mean_abs_logprob_diff"],
            current_process=capture_process, control_process=control_process, fresh_process=fresh_process)
        measured = gate_sequence.combine(components, shared, cache, compared, protocol,
                                         {"arm": job["arm"], "device": job["runtime_scope"]["device"]})
        _write(output/"measurements.json", _bytes(measured)+b"\n")
        result.update(status=COMPLETE, G12=compared, capture_handoff=handoff,
                      snapshot_handoff=snapshot_handoff, control_handoff=control_handoff, fresh_handoff=fresh_handoff,
                      measurements_sha256=_sha(_read(output/"measurements.json")),
                      shared_sha256=captured["shared_sha256"], saved_checkpoint_step=1,
                      capture_optimizer_steps=2, replay_optimizer_steps=1)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__)
        raise
    finally:
        _write(output/"result.json", _bytes(result)+b"\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--task-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        protocol = load_protocol(launch.HERE/"protocol_v1.yaml")
        if args.worker:
            _require(args.job is None and args.task is not None and args.task_sha256 is not None, "private task required")
            result = worker({"path": str(args.task), "file_sha256": args.task_sha256}, protocol)
        else:
            _require(args.job is not None and args.task is None and args.task_sha256 is None, "job required")
            job = launch.read_json(args.job)
            checked = preflight(job, protocol)
            if args.check or not _eligible(checked):
                print(json.dumps({"status": checked["status"], "blockers": checked["blockers"],
                    "gpu_runtime_checked": False, "gpu_gates_certified": False, "training_allowed": False}))
                return 0 if _eligible(checked) else 2
            result = execute(job, protocol, checked)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
            "gpu_gates_certified": False, "training_allowed": False, "pod_billing_stopped": False}))
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
