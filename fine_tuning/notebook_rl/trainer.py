"""Bounded RL execution core. Call only after separate data/environment approval.

This module never reads an answer key. The injected verifier must be the isolated
service in production. Tiny-model tests exercise orchestration, not 4B learning.
An outer owned-process watchdog is still required: a budget check cannot interrupt
a CUDA kernel or an in-flight rollout. Stop files/checkpoints do not stop Pod billing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from . import checkpoint, metrics
from .batches import CollectedBatch, _repair_protocol, collect_batch, family_order
from .engines import RolloutRequest
from .policy import PolicyBundle, assert_base_unchanged, completion_logprobs, merged_named_tensors
from .schema import ActorItem, ProblemTree, from_json, to_json
from .updates import effective_rewards, optimizer_step


def _write(path: Path, value: dict) -> None:
    """Write-once evidence, never overwrite an earlier failed/partial attempt."""
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class SyncProbe:
    request: RolloutRequest
    completion_ids: list[int]


def synchronize(bundle: PolicyBundle, preparation: Any, engine: Any, probes: list[SyncProbe],
                protocol: dict, step: int, *, push: bool, image_counts: tuple[int, ...] = (1, 2),
                threshold: float | None = None) -> list[dict]:
    """`image_counts` (1,) only for a preparation that binds single-image-only probe coverage.

    `threshold`: None = the protocol's HF threshold; scale.synchronize passes the bound vLLM-only one for its
    sampler report (the engine's own cap still refuses anything above 0.02 for HF engines)."""
    if (image_counts not in ((1,), (1, 2)) or sorted(len(p.request.images) for p in probes) != list(image_counts)
            or len({p.request.request_id for p in probes}) != len(probes)):
        raise ValueError("trainer: fixed single/multi-image synchronization probes required")
    if push:
        from .shared_policy_engine import SharedPolicyRolloutEngine
        if isinstance(engine, SharedPolicyRolloutEngine):
            engine.mark_updated(bundle, step)
        else:
            engine.push_weights(merged_named_tensors(bundle), step)
    if engine.policy_version != step:
        raise RuntimeError("trainer: sampler policy version not synchronized")
    reference = []
    device = next(bundle.model.parameters()).device
    with torch.no_grad():
        for probe in probes:
            inputs = preparation.model_inputs(probe.request, probe.completion_ids, device)
            values = completion_logprobs(bundle, inputs, len(probe.completion_ids),
                                        chunk=protocol["model"]["logprob_chunk"])[0].cpu().tolist()
            reference.append((probe.request, probe.completion_ids, values))
    limit = protocol["engine"]["probe_max_mean_abs_logprob_diff"] if threshold is None else threshold
    reports = engine.validate_sync(reference, limit,
                                   **({} if image_counts == (1, 2) else {"image_counts": image_counts}))
    # Only after actual engine probe validation may the separate preparer advance.
    preparation.policy_version = step
    return reports


def grounding_disabled(history: list[dict], f0: dict) -> bool:
    values = [r["grounding_probe"] for r in history if r.get("grounding_probe") is not None]
    if not values:
        return False
    baseline = f0.get("grounding_probe")
    if (not isinstance(baseline, (int, float)) or isinstance(baseline, bool)
            or not math.isfinite(baseline) or not 0 <= baseline <= 1
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)):
        raise ValueError("trainer: actual matched F0/current grounding probe required")
    return len(values) >= 2 and all(v < baseline for v in values[-2:])


def batch_statistics(batch: CollectedBatch) -> dict:
    trees = batch.trees + [from_json(ProblemTree, r["tree"]) for r in batch.discarded if "tree" in r]
    solutions = [s for t in trees for g in t.groups if g.kind != "DIAG" for s in g.samples]
    repair = ({"missing_result_fraction": sum(s.result is None for s in solutions)/len(solutions) if solutions else None}
              if any("repair_collector" in tree.notebook for tree in trees) else {})
    return {"n_observed_solutions": len(solutions),
        "len_median": statistics.median(len(s.completion_token_ids) for s in solutions) if solutions else None,
        "truncation_rate": sum(s.finish_reason == "length" for s in solutions)/len(solutions) if solutions else None,
        "short_fraction_le32": sum(len(s.completion_token_ids) <= 32 for s in solutions)/len(solutions) if solutions else None,
        "unverifiable_fraction": sum(s.result is not None and s.result.status == "UNVERIFIABLE"
                                    for s in solutions)/len(solutions) if solutions else None, **repair}


def _record(batch: CollectedBatch, update: dict, algorithm: str, seed: int, step: int,
            push_seconds: float, dev: dict | None) -> dict:
    repair = update.get("repair_preparation")
    groups = [g for t in batch.trees for g in t.groups if g.usable or repair is not None]
    rewards = update["effective_rewards"] if repair is not None else effective_rewards(groups, algorithm, seed, step-1)
    by_kind = {kind: [v for g in groups if g.kind == kind for v in rewards[g.group_id]]
               for kind in ("T1", "T2", "T3", "DIAG")}
    fields = ({"reward_basis": "answer_only_DIAG_zero", "degenerate_fraction_scope": "answer_rewards_only",
        "answer_degeneracy_does_not_imply_zero_repair_signal": True,
        "repair_signal_components": {name: any(row["signal_components"][name] for row in repair.values())
                                     for name in ("answer", "online", "final") + (("finalization",) if any(
                                         "finalization" in row["signal_components"] for row in repair.values()) else ())},
        "retained_repair_signal": update["retained_repair_signal"],
        "repair_preparation_sha256": checkpoint.digest(repair),
        "repair_aggregate_gradient_sha256": checkpoint.digest(update["repair_aggregate_gradient"])}
        if repair is not None else {})
    return metrics.step_record(step=step, policy_version=step, n_problems=len(batch.trees),
        n_discarded=len(batch.discarded), n_groups_by_kind=update["n_groups_by_kind"],
        reward_mean_by_kind={k: statistics.mean(v) if v else None for k, v in by_kind.items()},
        degenerate_fraction=sum(len(set(rewards[g.group_id])) <= 1 for g in groups)/len(groups),
        kl_mean=update["kl_mean"], masked_fraction=update["masked_fraction"],
        mean_abs_lp_diff=update["mean_abs_lp_diff"], grad_norm=update["grad_norm"],
        vision_on_fraction=sum(update["routes"].values())/len(batch.trees),
        **batch_statistics(batch),
        dev_first_attempt_acc=None if dev is None else dev["dev_first_attempt_acc"],
        grounding_probe=None if dev is None else dev.get("grounding_probe"),
        seconds={"rollout": batch.seconds, **update["seconds"], "push": push_seconds}, **fields)


def run_loop(*, bundle: PolicyBundle, optimizer: Any, scheduler: Any, engine: Any,
             preparation: Any, verifier: Any, actors: list[ActorItem], structures: dict[str, str],
             image_loader: Callable[[ActorItem], list], prompts: dict, protocol: dict,
             algorithm: str, seed: int, output: Path, binding: dict, f0_reference: dict,
             probes: list[SyncProbe], deadline_timestamp: float, max_steps: int,
             cursor: dict | None = None, dev_evaluator: Callable[[int], dict] | None = None,
             clock: Callable[[], float] = time.time) -> dict:
    """Train one frozen epoch; dependency-injected core, not an approval bypass CLI.

    Every completed batch is saved before updating. A failed sync saves a diagnostic
    checkpoint but blocks automatic resume. Valid checkpoints carry the full health
    history and consumed-ID cursor; resumes use a NEW output directory.
    """
    repair = _repair_protocol(protocol, prompts, algorithm, seed)
    if repair is not None:
        protocol = repair
        if (bundle.inventory["arm"] != "MOD-RV" or verifier is None
                or type(max_steps) is not int or not 1 <= max_steps <= 5):
            raise ValueError("trainer: repair requires its verifier, MOD-RV and at most five steps")
    if (algorithm not in protocol["algorithms"] or type(seed) is not int or seed not in protocol["seeds"]
            or type(max_steps) is not int or max_steps < 1 or protocol["batch"]["epochs"] != 1
            or not math.isfinite(deadline_timestamp) or not actors):
        raise ValueError("trainer: invalid frozen run limits")
    if max_steps >= 12 and dev_evaluator is None:
        raise ValueError("trainer: matched dev/grounding evaluation every 12 steps required")
    order = family_order(actors, seed)
    actor_by_id = {a.problem_id: a for a in actors}
    if set(structures) != set(order):
        raise ValueError("trainer: missing frozen structures")
    run_contract = {"algorithm": algorithm, "arm": bundle.inventory["arm"], "seed": seed,
        "order_sha256": checkpoint.digest(order), "actors_sha256": checkpoint.digest([to_json(actor_by_id[p]) for p in order]),
        "structures_sha256": checkpoint.digest(structures), "f0_sha256": checkpoint.digest(f0_reference),
        "binding": binding, "resolved_protocol_sha256": checkpoint.digest(protocol),
        "probes": [{"request_id": p.request.request_id, "prompt": p.request.prompt_token_ids,
                    "messages": p.request.messages, "completion": p.completion_ids,
                    "images": [{"size": im.size, "mode": im.mode,
                                "pixels_sha256": hashlib.sha256(im.tobytes()).hexdigest()}
                               for im in p.request.images]} for p in probes]}
    contract_sha = checkpoint.digest(run_contract)
    step = scheduler.last_epoch
    if cursor is None:
        if step != 0:
            raise ValueError("trainer: nonzero optimizer requires a bound resume cursor")
        state: dict[str, Any] = {"epoch": 0, "position": 0, "discarded_problem_ids": [], "history": [],
                 "vision_health_disabled": False, "sync_pending": False,
                 "contract_sha256": contract_sha, "batch_sequence": 0, "untrained_problem_ids": [],
                 "stop_reasons": []}
    else:
        state = json.loads(json.dumps(cursor))
        if (state.get("contract_sha256") != contract_sha or state.get("sync_pending") is not False
                or state.get("stop_reasons") != [] or state.get("epoch") != 0
                or type(state.get("position")) is not int or not 0 <= state["position"] <= len(order)
                or not isinstance(state.get("history"), list) or len(state["history"]) != step
                or type(state.get("batch_sequence")) is not int or state["batch_sequence"] < 0
                or type(state.get("vision_health_disabled")) is not bool
                or any(not isinstance(state.get(key), list)
                       or any(pid not in actor_by_id for pid in state[key])
                       for key in ("discarded_problem_ids", "untrained_problem_ids"))):
            raise ValueError("trainer: uncommitted, stale or incompatible resume cursor")
    if engine.policy_version != step or preparation.policy_version != step:
        raise ValueError("trainer: reconstruct and verify the resumed engine before continuation")
    if repair is not None and (state["vision_health_disabled"] or grounding_disabled(state["history"], f0_reference)):
        raise ValueError("trainer: repair cannot resume with disabled Vision or a Vision health override")
    # Validate actual Phase-A health references before allocating any output artifacts.
    reference_length = f0_reference.get("len_median")
    if (not isinstance(reference_length, (float, int)) or isinstance(reference_length, bool)
            or not math.isfinite(reference_length) or reference_length <= 0):
        raise ValueError("trainer: measured positive F0 length required")
    if step >= max_steps:
        raise ValueError("trainer: run limit already reached")
    output = Path(output)
    checkout = Path(__file__).resolve().parents[2]
    if output.resolve().is_relative_to(checkout) or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("trainer: use an external, non-symlink artifact directory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "batches").mkdir()
    (output / "closed_trees").mkdir()
    (output / "closed_groups").mkdir()
    (output / "checkpoints").mkdir()
    if repair is not None:
        (output / "raw").mkdir()
    _write(output / "run.json", run_contract | {"contract_sha256": contract_sha,
        "start_step": step, "start_cursor": state, "deadline_timestamp": deadline_timestamp,
        "max_steps": max_steps, "status": "STARTED_NOT_COMPLETED"})
    status, stop_reasons, failure = "STEP_LIMIT", [], None
    latest_checkpoint = None
    raw_sequence = 0
    def preserve_raw(event):
        nonlocal raw_sequence
        _write(output / "raw" / f"{raw_sequence:06d}.json", {"event": "REPAIR_RAW", "step": step,
            "batch_sequence": state["batch_sequence"] + 1, "record": event})
        raw_sequence += 1
    def should_stop():
        return clock() >= deadline_timestamp-120
    def persist(label):
        nonlocal latest_checkpoint
        destination = output / "checkpoints" / f"step-{step:06d}-batch-{state['batch_sequence']:06d}-{label}"
        checkpoint.save(bundle, optimizer, scheduler, step, state, checkpoint.capture_rng(),
                        destination, protocol=protocol, binding=binding)
        latest_checkpoint = str(destination)
    try:
        if should_stop():
            status = "BUDGET_STOP"
        else:
            initial = synchronize(bundle, preparation, engine, probes, protocol, step, push=False)
            _write(output / "initial_sync.json", {"step": step, "probes": initial})
        while status != "BUDGET_STOP" and step < max_steps and state["position"] < len(order):
            if should_stop():
                status = "BUDGET_STOP"
                break
            def preserve_tree(tree):
                _write(output / "closed_trees" / f"{step:06d}-{hashlib.sha256(tree.problem_id.encode()).hexdigest()}.json",
                       json.loads(to_json(tree)))
            def preserve_group(group):
                _write(output / "closed_groups" / f"{hashlib.sha256(group.group_id.encode()).hexdigest()}.json",
                       json.loads(to_json(group)))
            batch = collect_batch(actor_by_id, structures, order, state["position"], image_loader,
                engine, verifier, prompts, protocol, algorithm, should_stop=should_stop, seed=seed,
                on_tree=preserve_tree, on_group=preserve_group, **({"on_raw": preserve_raw} if repair is not None else {}))
            state["batch_sequence"] += 1
            batch_id = f"{state['batch_sequence']:06d}"
            try:
                # This closed record includes both selected and discarded raw trees.
                _write(output / "batches" / f"{batch_id}.json", {"status": batch.status,
                    "step_before": step, "cursor_before": state["position"], "cursor_after": batch.cursor,
                    "attempted": batch.attempted, "discarded": batch.discarded, "diagnosis": batch.diagnosis,
                    "trees": [json.loads(to_json(t)) for t in batch.trees]})
                if batch.status == "BUDGET_STOP" or should_stop():
                    # No optimizer update committed: resume this whole batch at the
                    # old cursor, not after its untrained (possibly incomplete) trees.
                    state["untrained_problem_ids"] = sorted(set(state["untrained_problem_ids"] + batch.attempted))
                    status = "BUDGET_STOP"
                    break
                state["position"] = batch.cursor
                state["discarded_problem_ids"].extend(r["problem_id"] for r in batch.discarded)
                stats = batch_statistics(batch)
                if (stats["unverifiable_fraction"] is not None
                        and stats["unverifiable_fraction"] > protocol["health"]["unverifiable_max"]):
                    _write(output / "batches" / f"{batch_id}.health.json", stats)
                    state["untrained_problem_ids"].extend(t.problem_id for t in batch.trees)
                    status, stop_reasons = "HEALTH_STOP", ["VERIFIER"]
                    break
                if not batch.trees:
                    continue
                update = optimizer_step(bundle, optimizer, scheduler, batch.trees, batch.images,
                    preparation, protocol, algorithm, step=step, seed=seed,
                    vision_disabled=state["vision_health_disabled"], **({"objective_verifier": verifier,
                        "observation_actors": {tree.problem_id: actor_by_id[tree.problem_id] for tree in batch.trees}}
                        if repair is not None else {}))
                _write(output / "batches" / f"{batch_id}.update.json", update)
                if update["status"] == "NO_MATCHED_ACTIONS":
                    state["untrained_problem_ids"].extend(t.problem_id for t in batch.trees)
                    status, stop_reasons = "HEALTH_STOP", ["NO_MATCHED_ACTIONS"]
                    break
                if update["status"] == "NO_ACTIONS" or repair is not None and update["status"] in (
                        "NO_ADMITTED_REPAIR_TREES", "NO_LEARNING_SIGNAL"):
                    state["untrained_problem_ids"].extend(t.problem_id for t in batch.trees)
                    continue
                if update["status"] != "UPDATED_UNSYNCHRONIZED":
                    raise RuntimeError("trainer: unexpected optimizer result")
                step = update["step_after"]
                trained = {t.problem_id for t in batch.trees}
                state["untrained_problem_ids"] = [pid for pid in state["untrained_problem_ids"] if pid not in trained]
                state["sync_pending"] = True
                began = time.monotonic()
                sync = synchronize(bundle, preparation, engine, probes, protocol, step, push=True)
                push_seconds = time.monotonic()-began
                _write(output / "batches" / f"{batch_id}.sync.json", {"step": step, "probes": sync})
                dev = None
                if step % 12 == 0:
                    if should_stop():
                        # No invented dev score at a budget boundary; next run must check it first.
                        status, stop_reasons = "BUDGET_STOP", ["DEV_CHECK_PENDING"]
                    else:
                        if dev_evaluator is None:
                            raise ValueError("trainer: missing scheduled dev evaluator")
                        dev = dev_evaluator(step)
                        if (dev.get("step") != step or dev.get("status") != "COMPLETED"
                                or dev.get("n") != 96 or dev.get("samples_per_problem") != 1
                                or bundle.params["V"] and dev.get("grounding_probe") is None):
                            raise ValueError("trainer: incomplete scheduled dev/grounding measurements")
                        _write(output / "batches" / f"{batch_id}.dev.json", dev)
                record = _record(batch, update, algorithm, seed, step, push_seconds, dev)
                metrics.append_record(output / "metrics.jsonl", record)
                state["history"].append(record)
                state["sync_pending"] = bool(stop_reasons == ["DEV_CHECK_PENDING"])
                if grounding_disabled(state["history"], f0_reference):
                    if repair is not None:
                        status, stop_reasons = "HEALTH_STOP", ["VISION_HEALTH_REVIEW_REQUIRED"]
                        break
                    state["vision_health_disabled"] = True
                health = metrics.health_check(state["history"], protocol, f0_reference)
                if health:
                    status, stop_reasons = "HEALTH_STOP", health
                    break
                if status == "BUDGET_STOP":
                    break
                if step % 12 == 0:
                    persist("synchronized")
            finally:
                for values in batch.images.values():
                    for image in values:
                        image.close()
        if status == "STEP_LIMIT" and state["position"] == len(order):
            status = "EPOCH_COMPLETED" if step else "NO_LEARNING_SIGNAL"
        # Health stops require a new reviewed protocol, never an automatic resume.
        state["stop_reasons"] = stop_reasons
        assert_base_unchanged(bundle)
        persist("closed")
    except BaseException as exc:
        # Preserve an updated-but-unsynchronized state for diagnosis, never report success.
        status = "FAILED"
        failure = {"type": type(exc).__name__, "message": str(exc), "step": step}
        state["sync_pending"] = True  # No failed boundary is eligible for automatic continuation.
        optimizer.zero_grad(set_to_none=True)
        if scheduler.last_epoch != step:
            state["sync_pending"] = True
            step = scheduler.last_epoch
        try:
            persist("failed")
        except Exception as save_exc:
            failure["checkpoint_error"] = f"{type(save_exc).__name__}: {save_exc}"
    summary = {"status": status, "stop_reasons": stop_reasons, "failure": failure, "step": step,
        "consumed": state["position"], "selected": len(order), "last_checkpoint": latest_checkpoint,
        "contract_sha256": contract_sha, "sync_pending": state["sync_pending"],
        "vision_health_disabled": state["vision_health_disabled"],
        "full_epoch_completed": status == "EPOCH_COMPLETED",
        "model_improvement_verified": False, "pod_billing_stopped": False}
    _write(output / "summary.json", summary)
    return summary
