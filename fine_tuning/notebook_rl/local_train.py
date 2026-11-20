"""Fixed-ten local RL and matched notebook-reward control after the minimal gate."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

from . import approval, checkpoint, scale as scale_options
from .cli import artifact_path
from .dataset import load_images
from .local_profile import DEFAULT_NOTE_CONTRACT, NOTE_CONTRACT_CHOICES, PROFILE, local_prompts, restore_protocol
from .repair_replay import raw_journal_paths, read_snapshot as _repair_read
from .local_run import (_binding, _install_actor_guards, cache_readiness, make_probes,
                        parameter_hashes, probe_evidence, read_preparation, verify_changes, write_json,
                        _positive_error_relationship, relationship_learning_summary)
from .local_run import (VERIFIED_UPDATE_EXPLORATION, minimum_execution_contract, minimum_binding,
                        minimal_actors, retained_update_signal, native_relationship_update_evidence)
from .local_run import (covering_actors, minimal_training_actors, pool_v2, probe_counts, probe_set, probe_source,
                        sync_options, with_probe_actors)
from .local_run import add_frozen_approval_arguments, frozen_approval_option, read_frozen_preparation
from .local_run import disk_low, host_identity, owner_alive, same_process
from .local_run import read_recorded_preparation
from .schema import to_json

STRICT_ADMISSION = "strict-notebook-learning-v1"
EXPLORATORY_ADMISSION = "verified-fact-credit-exploration-v1"
SYNTHESIS_CREDIT_ADMISSION = "verified-synthesis-credit-exploration-v2"
ASSESSMENT_CREDIT_ADMISSION = "verified-assessment-credit-exploration-v1"
REPAIR_ADMISSION = "verified-native-repair-learning-v1"
REPAIR_FUNCTIONAL_EXPLORATION = "verified-functional-repair-exploration-v1"
REPAIR_ADMISSIONS = (REPAIR_ADMISSION, REPAIR_FUNCTIONAL_EXPLORATION)
FIXED_TEN_NATIVE_EXPLORATION = "fixed-ten-native-exploration-v1"
TRAINING_BUDGET_VERSION = "fixed-ten-wall-time-v1"


def execution_max_seconds(contract: dict, protocol: dict) -> int:
    """Only the explicitly bound fixed-ten training budget may override the protocol default."""
    budget = contract.get("training_budget", {"version": TRAINING_BUDGET_VERSION,
        "max_seconds": protocol["local_budget"]["max_seconds_per_run"]})
    if (type(budget) is not dict or set(budget) != {"version", "max_seconds"}
            or budget["version"] != TRAINING_BUDGET_VERSION
            or type(budget["max_seconds"]) is not int or not 1 <= budget["max_seconds"] <= 86400):
        raise ValueError("local train: explicit versioned training budget requires integer seconds in 1..86400")
    return budget["max_seconds"]


def execution_admission(contract: dict, protocol: dict) -> str:
    """Historical contracts remain strict; exploration requires the exact V16 objective."""
    from .notebook_objective import FACT_CREDIT_VERSION, contract as objective_contract
    from .synthesis_credit import ASSESSMENT_VERSION, contract as synthesis_contract
    from .local_profile import ASSESSMENT_PROTOCOL_ID
    from .updates import ASSESSMENT_REQUIREMENT

    admission = contract.get("admission_policy", STRICT_ADMISSION)
    from .local_profile import REPAIR_PROTOCOL_ID, repair_objective_contract, repair_options
    from .repair_notebook import note_contract

    repair = (protocol.get("protocol_id") == REPAIR_PROTOCOL_ID
              or any(key.startswith("repair_") for key in protocol)
              or protocol.get("rollout", {}).get("retry_contract") is not None)
    if admission in REPAIR_ADMISSIONS:
        flags = {key: value for key, value in repair_options(protocol).items() if key != "slotless_sources"}
        keys = {"stage", "arm", "admission_policy", "training_budget", "notebook_reward_enabled",
                "max_optimizer_updates", "pair_ids", "no_signal_policy", "same_generated_notebook_checks_and_kl",
                "initial_checkpoint", "authority", "parent_gate", "deterministic_execution", "matched_notebook_arm",
                "repair_protocol_sha256", "repair_objective_sha256"}
        # Segment fields exist only under a bound scale group, so historical contracts keep their exact key set.
        allowed = keys | ({"problem_range", "resume"} if scale_options.scaled(protocol) else set())
        window = contract.get("problem_range", [0, 1])
        if (protocol.get("protocol_id") != REPAIR_PROTOCOL_ID
                or protocol.get("repair_objective") != repair_objective_contract(note_contract(protocol), **flags)
                or protocol.get("repair_objective_sha256") != checkpoint.digest(repair_objective_contract(
                    note_contract(protocol), **flags))
                or not (set(contract) == {"admission_policy"} or keys <= set(contract) <= allowed)
                or type(window) is not list or len(window) != 2 or any(type(value) is not int for value in window)
                or not 0 <= window[0] < window[1]
                or "resume" in contract and type(contract["resume"]) is not dict
                or "stage" in contract and (contract.get("stage") != "FIXED_TEN_FROM_FRESH_F0"
                    or contract.get("repair_protocol_sha256") != protocol.get("sha256")
                    or contract.get("repair_objective_sha256") != protocol.get("repair_objective_sha256")
                    or contract.get("arm") not in ("notebook", "answer-only")
                    or contract.get("notebook_reward_enabled") is not (contract["arm"] == "notebook")
                    or type(contract.get("max_optimizer_updates")) is not int
                    or contract["max_optimizer_updates"] != scale_options.options(protocol)["max_updates"]
                    or contract.get("same_generated_notebook_checks_and_kl") is not True
                    or contract.get("no_signal_policy") != "record; consume the fixed pair; no replacement or forced update"
                    or contract.get("initial_checkpoint") != scale_options.initial_checkpoint(protocol)
                    or contract.get("authority") != protocol.get("local_authorization")
                    or type(contract.get("parent_gate")) is not dict
                    or contract["arm"] == "answer-only" and type(contract.get("matched_notebook_arm")) is not dict
                    or contract["arm"] == "notebook" and contract.get("matched_notebook_arm") is not None)):
            raise ValueError("local train: repair admission requires its exact canonical protocol/objective")
        restore_protocol(protocol)
        return admission
    if repair:
        raise ValueError("local train: repair requires its explicit repair admission")
    if admission == ASSESSMENT_CREDIT_ADMISSION:
        if (protocol.get("protocol_id") != ASSESSMENT_PROTOCOL_ID
                or protocol.get("notebook_objective") != synthesis_contract(ASSESSMENT_VERSION)
                or protocol.get("notebook_update_requirement") != ASSESSMENT_REQUIREMENT):
            raise ValueError("local train: assessment-credit admission requires its exact assessment protocol")
        restore_protocol(protocol)
        return admission
    if protocol.get("notebook_objective", {}).get("objective_version") == ASSESSMENT_VERSION:
        raise ValueError("local train: assessment-credit fixed-ten admission is unsupported; partial classification is not full notebook success")

    if admission == FIXED_TEN_NATIVE_EXPLORATION:
        from .local_profile import PAIR_ACTION_NOTEBOOK_PROTOCOL_ID
        from .updates import NATIVE_RELATIONSHIP_REQUIREMENT

        if (protocol.get("protocol_id") != PAIR_ACTION_NOTEBOOK_PROTOCOL_ID
                or protocol.get("notebook_generation_policy") != "verified-pair-action-notebook-v2"
                or protocol.get("notebook_update_requirement") != NATIVE_RELATIONSHIP_REQUIREMENT):
            raise ValueError("local train: fixed-ten native exploration requires the exact native pair-action v2 protocol")
        restore_protocol(protocol)
        return admission
    if admission == VERIFIED_UPDATE_EXPLORATION:
        minimum_execution_contract(admission, protocol)
        return admission
    if admission == SYNTHESIS_CREDIT_ADMISSION:
        from .synthesis_credit import contract as synthesis_contract
        from .local_profile import (PAIR_ACTION_NOTEBOOK_PROTOCOL_ID, PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID,
                                    PAIRED_NOTEBOOK_PROTOCOL_ID, PARTIAL_NOTEBOOK_PROTOCOL_ID,
                                    REFERENCED_NOTEBOOK_PROTOCOL_ID, SYNTHESIS_CREDIT_PROTOCOL_ID,
                                    WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID)

        if (protocol.get("protocol_id") not in (
                SYNTHESIS_CREDIT_PROTOCOL_ID, PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID,
                PAIRED_NOTEBOOK_PROTOCOL_ID, PAIR_ACTION_NOTEBOOK_PROTOCOL_ID, PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID,
                WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID)
                or restore_protocol(protocol).get("notebook_objective") != synthesis_contract()):
            raise ValueError("local train: synthesis credit requires its exact protocol")
        return admission
    if (admission not in (STRICT_ADMISSION, EXPLORATORY_ADMISSION)
            or admission == EXPLORATORY_ADMISSION
            and protocol.get("notebook_objective") != objective_contract(FACT_CREDIT_VERSION)):
        raise ValueError("local train: unknown or incompatible admission policy")
    return admission


def _launch_preparation(launch: dict, protocol: dict):
    """A launch's preparation in the mode it was verified with (this module's read_preparation unless frozen)."""
    if "data_verification" in launch:
        return read_recorded_preparation(launch, protocol)
    return read_preparation(Path(launch["preparation"]), protocol)


def _read(path: Path):
    from .vault import _read as guarded_read, _unique

    return json.loads(guarded_read(path), object_pairs_hook=_unique)


def _repair_probe_binding(forward: dict, sync: list, step: int, actors: list, *, batch_rows: int | None = None,
                          image_counts=(1, 2)) -> None:
    """`image_counts`: the preparation's bound probe coverage (local_run.probe_counts)."""
    from .repair_replay import validate_probe_evidence

    selected = covering_actors(actors, image_counts)
    validate_probe_evidence(forward, [actor.problem_id for actor in selected], [len(actor.image_paths) for actor in selected],
                            batch_rows=batch_rows)
    _sync_rows(sync, step, actors, image_counts=image_counts)


def _sync_rows(sync: list, step: int, actors: list, *, image_counts=(1, 2), threshold: float = .02) -> None:
    """`threshold`: 0.02 for HF reports; a vLLM sampler report passes scale.sampler_sync_threshold (0.04)."""
    selected = covering_actors(actors, image_counts)
    if (type(sync) is not list or len(sync) != len(selected)
            or any(type(row) is not dict or type(row.get("policy_version")) is not int or row["policy_version"] != step
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0
                       for key in ("mean_abs_delta", "max_abs_delta"))
                or row["mean_abs_delta"] > threshold or row["mean_abs_delta"] > row["max_abs_delta"] for row in sync)
            or [row.get("request_id") for row in sync] != ["local-sync:" + actor.problem_id for actor in selected]):
        raise ValueError("local train: exact single/multi-image repair forward and synchronized probes required")


def notebook_learning_gate(capture: Path, result: dict, update: dict, protocol: dict) -> list[Path]:
    """Bind a rewarded note to its isolated backward pass, not merely answer learning."""
    from .episode_notebook import STRICT_ROUTE_RULE, validate_update_binding
    from .updates import has_learning_signal, notebook_objective_scores, notebook_signal_groups
    from .notebook_objective import FACT_CREDIT_VERSION, SYNTHESIS_CREDIT_VERSION
    from .rollout import _hash
    from .schema import ProblemTree, from_json
    from . import notebook_synthesis as synthesis
    from .synthesis_credit import ASSESSMENT_VERSION

    if protocol.get("notebook_objective", {}).get("objective_version") == ASSESSMENT_VERSION:
        raise ValueError("local train: assessment-credit strict notebook admission is unsupported")

    failure = "local train: verified positive notebook reward-gradient contribution required"
    if "notebook_update_requirement" in protocol:
        native_relationship_update_evidence(update, protocol)
    if (type(result.get("notebook_signal_groups")) is not int or result["notebook_signal_groups"] <= 0
            or result.get("notebook_gradient_contribution_verified") is not True
            or update.get("notebook_reward_enabled") is not True):
        raise ValueError(failure)
    backward = {row["sample_id"]: row for row in update["notebook_backward"]}
    if len(backward) != len(update["notebook_backward"]):
        raise ValueError("local train: duplicate notebook backward sample")
    objective = protocol.get("notebook_objective")
    synthesis_contract = protocol.get("notebook_synthesis_contract")
    synthesis_selected = synthesis_contract is not None or protocol.get("notebook_reward") == synthesis.BOUNDED_VERSION
    if synthesis_selected and (synthesis_contract != synthesis.BOUNDED_VERSION
            or protocol.get("notebook_reward") != synthesis_contract
            or objective is not None and objective.get("objective_version") != SYNTHESIS_CREDIT_VERSION
            or type(update.get("synthesis_regrade_receipts")) is not dict):
        raise ValueError("local train: bounded synthesis requires fresh regraded synthesis evidence")
    if objective is not None and update.get("notebook_objective") != objective:
        raise ValueError("local train: notebook objective contract differs from update")
    paths, seen_trees, seen_samples = [], set(), set()
    signal_groups, contributing = 0, 0
    for path in raw_journal_paths(capture):
        record = _read(path)
        if record.get("kind") != "TREE":
            continue
        tree = from_json(ProblemTree, json.dumps(record["raw"], allow_nan=False))
        if tree.problem_id not in update["routes"]:
            continue
        if tree.problem_id in seen_trees:
            raise ValueError("local train: duplicate notebook tree")
        seen_trees.add(tree.problem_id)
        paths.append(path)
        validate_update_binding(tree, feedback_policy=protocol.get("notebook_feedback_policy"),
                                observation_policy=protocol.get("notebook_observation_policy"),
                                generation_policy=protocol.get("notebook_generation_policy"),
                                draft_policy=protocol.get("notebook_draft_policy"))
        if protocol.get("notebook_route_rule") == STRICT_ROUTE_RULE:
            audit = tree.notebook.get("rewarded_notebook", {}).get("episode_audit", {})
            if audit.get("routing_rule") != STRICT_ROUTE_RULE or not has_learning_signal(
                    tree, include_notebooks=True, notebook_objective=objective):
                raise ValueError("local train: strict notebook routing admission missing or unresolved")
        scores = notebook_objective_scores(tree, objective)
        signal_groups += notebook_signal_groups(tree, scores)
        if synthesis_selected:
            receipt = update["synthesis_regrade_receipts"].get(tree.problem_id)
            stored = tree.notebook["rewarded_notebook"]
            grades = {note["sample_id"]: {key: value for key, value in note.items()
                       if key not in ("sample_id", "text", "eligible_for_rl_loss")}
                      for note in tree.notebook.get("model_notes", [])}
            expected = {"contract": synthesis_contract, "actor_sha256": stored["synthesis_audit"]["actor_sha256"],
                "synthesis_audit_sha256": stored["synthesis_audit_sha256"],
                "notebook_groups_sha256": stored["notebook_groups_sha256"], "samples": grades}
            from .whole_reference_pages import VERSION as pages_version

            if protocol.get("notebook_generation_policy") == pages_version:
                expected["bundles"] = stored["bundles"]
            if (stored.get("notebook_synthesis_contract") != synthesis_contract
                    or receipt != expected | {"sha256": _hash(expected)}):
                raise ValueError("local train: freshly regraded synthesis evidence differs")
        if objective is not None and update.get("notebook_objective_receipts", {}).get(tree.problem_id) != tree.notebook["learning_objective"]:
            raise ValueError("local train: notebook objective receipt differs from update")
        if (tree.policy_version != update["step_before"]
                or update["routes"][tree.problem_id] is not (tree.route == "VISION_ON")):
            raise ValueError("local train: notebook policy or route differs from update")
        notes = tree.notebook.get("model_notes", [])
        by_id = {note["sample_id"]: note for note in notes}
        if len(by_id) != len(notes):
            raise ValueError("local train: duplicate checked notebook sample")
        for group in tree.groups:
            if group.kind != "DIAG" or not group.usable:
                continue
            rewards = [scores[s.sample_id]["effective_reward"] if scores else s.reward for s in group.samples]
            if update["effective_rewards"].get(group.group_id) != rewards:
                raise ValueError("local train: notebook effective rewards differ")
            scoped = objective and objective["objective_version"] in (FACT_CREDIT_VERSION, SYNTHESIS_CREDIT_VERSION)
            policy_rewards = [s.reward for s in group.samples] if scoped else rewards
            mean = sum(policy_rewards) / len(policy_rewards)
            for sample in group.samples:
                if sample.sample_id in seen_samples:
                    raise ValueError("local train: duplicate notebook sample")
                seen_samples.add(sample.sample_id)
                note, row = by_id.get(sample.sample_id), backward.get(sample.sample_id)
                effective = scores[sample.sample_id]["effective_reward"] if scores else sample.reward
                policy_reward = sample.reward if scoped else effective
                if sample.reward != 1 or policy_reward - mean <= 0 or row is None:
                    continue
                if scores and (row.get("strict_reward") != sample.reward
                        or row.get("effective_reward") != effective
                        or row.get("atomic_reward") != scores[sample.sample_id]["atomic_reward"]
                        or row.get("objective_score_sha256") != _hash(scores[sample.sample_id])):
                    raise ValueError("local train: notebook backward objective evidence differs")
                checks = {key: True for key in ("format_valid", "all_attempts_verified",
                                               "answer_verified", "registered_facts_verified")}
                verified = (synthesis.successful(note, contract=synthesis.BOUNDED_VERSION)
                            if synthesis_selected and note is not None
                            else note is not None and set(note.get("checks", {})) == set(checks)
                            and all(note["checks"][key] is True for key in checks))
                if (note is None or type(note.get("reward")) not in (int, float) or note["reward"] != 1
                        or not verified
                        or note.get("eligible_for_rl_loss") is not True
                        or note.get("reward_version") != protocol["notebook_reward"]
                        or note.get("text") != sample.text or sample.diag_reward != 1
                        or sample.finish_reason != "stop" or row.get("group_id") != group.group_id
                        or row.get("vision_on") is not update["routes"][tree.problem_id]
                        or type(row.get("advantage")) not in (int, float)
                        or not math.isclose(row["advantage"], policy_reward - mean, rel_tol=0, abs_tol=1e-7)
                        or type(row.get("action_tokens")) is not int
                        or row["action_tokens"] != len(sample.completion_token_ids)):
                    raise ValueError(failure)
                before, after = row["gradient_before"], row["gradient_after"]
                if (set(before) != {"L", "M", "V"} or set(after) != set(before)
                        or any(type(before[g]) is not list or type(after[g]) is not list
                               or not before[g] or len(before[g]) != len(after[g]) for g in before)
                        or any(value is not None and (type(value) is not str
                               or re.fullmatch(r"[0-9a-f]{64}", value) is None)
                               for state in (before, after) for values in state.values() for value in values)):
                    raise ValueError("local train: incomplete notebook gradient evidence")
                changed = [g for g in before if before[g] != after[g]]
                if row.get("changed_groups") != changed:
                    raise ValueError(failure)
                contributing += int(bool(changed) and row.get("reward_gradient_isolated") is True)
    if (seen_trees != set(update["routes"]) or signal_groups != result["notebook_signal_groups"]
            or not contributing):
        raise ValueError(failure)
    return paths


def parent_gate(capture: Path, reload: Path, *, require_notebook_learning: bool = False) -> dict:
    """Recheck closed real-update evidence; keep its historical code binding intact."""
    capture, reload = Path(capture).absolute(), Path(reload).absolute()
    launch, result = _read(capture / "launch.json"), _read(capture / "result.json")
    restored, expected = _read(reload / "result.json"), _read(capture / "reload_expected.json")
    update, changes = _read(capture / "update.json"), _read(capture / "parameter_changes.json")
    protocol = json.loads(json.dumps(restore_protocol(launch.get("protocol"))))
    # A BF16 minimum loads the verified original snapshot (no NF4 cache, no original-access guard).
    bf16 = scale_options.bf16(protocol)
    if (launch.get("protocol") != protocol or launch.get("standalone_base") is not (not bf16)
            or type(launch.get("pid")) is not int or launch["pid"] <= 0 or owner_alive(launch)
            or launch.get("purpose") != "MINIMAL_TWO_PROBLEM_ONE_UPDATE"
            or len(launch.get("problem_ids", [])) != 2
            or result.get("status") != "UPDATED_SAVED_PENDING_FRESH_RELOAD"
            or result.get("optimizer_updates") != 1 or expected.get("step") != 1
            or restored.get("status") != "FRESH_PROCESS_RELOAD_VERIFIED"
            or restored.get("capture_pid") != launch.get("pid")
            or same_process(restored, launch) or restored.get("step") != 1
            or any(restored.get(k) is not True for k in ("all_trainable_hashes_match",
                "optimizer_restored", "actual_forward_and_generation_match", "frozen_quantized_base_unchanged"))
            or not bf16 and any(v.get("original_access_guard", {}).get("denied_original_access_attempts") != 0
                                for v in (result, restored))):
        raise ValueError("local train: completed real minimal update and fresh reload required")
    verified = verify_changes(changes["before"], changes["after"], update)
    if any(result.get(k) != value for k, value in verified.items()):
        raise ValueError("local train: parent parameter evidence differs")
    metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
    flattened = {name: sha for values in expected["trainable_sha256"].values() for name, sha in values.items()}
    if (metadata["step"] != 1 or metadata["state_sha256"] != flattened
            or changes["after"] != expected["trainable_sha256"]
            or metadata["base_sha256"] != expected["frozen_base_sha256"]
            or metadata["inventory"].get("quantization") != expected["quantization"]
            or _read(capture / "updated_forward_probes.json") != expected["forward_probes"]
            or _read(reload / "forward_probes.json") != expected["forward_probes"]):
        raise ValueError("local train: parent checkpoint/forward evidence differs")
    from .updates import ASSESSMENT_REQUIREMENT

    if protocol.get("notebook_update_requirement") == ASSESSMENT_REQUIREMENT:
        from .local_run import assessment_update_evidence

        assessment = assessment_update_evidence(update, protocol, parameter_inventory=metadata["inventory"]["trainable"])
        if any(result.get(key) != value for key, value in assessment.items()):
            raise ValueError("local train: parent assessment evidence differs")
    note_paths = notebook_learning_gate(capture, result, update, protocol) if require_notebook_learning else []
    paths = [capture / name for name in ("launch.json", "result.json", "update.json",
             "parameter_changes.json", "reload_expected.json", "updated_forward_probes.json")]
    paths += [capture / "checkpoint" / name for name in
              ("metadata.json", "metadata.sha256", "adapter.safetensors", "mergers.safetensors", "runtime.pt")]
    paths += [reload / "result.json", reload / "forward_probes.json", reload / "sync.json"]
    paths += note_paths
    return {"status": "REAL_MINIMAL_UPDATE_AND_FRESH_RELOAD_VERIFIED", "launch": launch,
            "notebook_reward_learning_verified": require_notebook_learning,
            "files": [{"path": str(p), "sha256": checkpoint.file_hash(p)} for p in paths]}


def fact_credit_learning_gate(capture: Path, result: dict, update: dict, protocol: dict, *,
                             actors: dict, verifier, tokenizer, replay_relation_v1: bool = False,
                             admission: str | None = None, tree_ids: set[str] | None = None,
                             control: bool = False, parameter_inventory=None) -> list[Path]:
    """Regrade complete groups before MIS; require retained positive fact-token learning."""
    import torch

    from .episode_notebook import STRICT_ROUTE_RULE, audit_episode, validate_update_binding
    from . import fact_credit, synthesis_credit
    from .rollout import _hash
    from .schema import ProblemTree, from_json
    from .updates import (_score_notebook_objective, effective_mis_deltas,
                          effective_rewards, notebook_objective_scores, notebook_signal_groups)

    assessment_selected = protocol.get("notebook_objective", {}).get("objective_version") == synthesis_credit.ASSESSMENT_VERSION
    if assessment_selected and admission != ASSESSMENT_CREDIT_ADMISSION:
        raise ValueError("local train: assessment-credit exploratory notebook admission is unsupported")
    if control and (not assessment_selected or update.get("notebook_reward_enabled") is not False):
        raise ValueError("local train: assessment control replay requires its explicit reward-disabled arm")
    synthesis_selected = assessment_selected or protocol.get("notebook_objective", {}).get("objective_version") == synthesis_credit.VERSION
    from .pair_action_notebook import VERSION as pair_action_version

    if protocol.get("notebook_generation_policy") == pair_action_version:
        retained_update_signal(update, protocol)
    native_contributing = False
    if "notebook_update_requirement" in protocol and not assessment_selected:
        native_contributing = native_relationship_update_evidence(update, protocol)["native_relationship_only_gradient_verified"]
    execution_admission({"admission_policy": ASSESSMENT_CREDIT_ADMISSION if assessment_selected else
                         SYNTHESIS_CREDIT_ADMISSION if synthesis_selected else EXPLORATORY_ADMISSION}, protocol)
    group_credit = synthesis_credit.group_credit if synthesis_selected else fact_credit.group_credit
    credit_key = "synthesis_token_credit" if synthesis_selected else "fact_token_credit"
    credit_hash_key = "synthesis_credit_sha256" if synthesis_selected else "fact_credit_sha256"
    failure = ("local train: retained positive verified relationship-token gradient required" if synthesis_selected else
               "local train: retained positive fact-token isolated notebook gradient required")
    if (not control and update.get("notebook_reward_enabled") is not True
            or update.get("notebook_objective") != protocol["notebook_objective"]
            or not synthesis_selected and result.get("notebook_gradient_contribution_verified") is not True):
        raise ValueError(failure)
    deltas = effective_mis_deltas(update, protocol)
    backward = {row["sample_id"]: row for row in update["notebook_backward"]}
    if len(backward) != len(update["notebook_backward"]):
        raise ValueError("local train: duplicate notebook backward sample")
    paths, seen_trees, audited_trees, credits, retained = [], set(), set(), {}, set()
    regraded_trees, regraded_scores = [], {}
    original_actions: set[str] = set()
    signal_groups, contributing = 0, native_contributing
    for path in raw_journal_paths(capture):
        record = _read(path)
        if record.get("kind") != "TREE":
            continue
        tree = from_json(ProblemTree, json.dumps(record["raw"], allow_nan=False))
        if tree_ids is not None and tree.problem_id not in tree_ids:
            continue
        if tree.problem_id in audited_trees:
            raise ValueError("local train: duplicate notebook tree")
        audited_trees.add(tree.problem_id)
        paths.append(path)
        validate_update_binding(tree, tokenizer=tokenizer,
            feedback_policy=protocol.get("notebook_feedback_policy"),
            observation_policy=protocol.get("notebook_observation_policy"),
            generation_policy=protocol.get("notebook_generation_policy"), draft_policy=protocol.get("notebook_draft_policy"))
        audit = tree.notebook["rewarded_notebook"]["episode_audit"]
        if audit_episode(tree, actors[tree.problem_id], verifier,
                routing_rule=audit.get("routing_rule"), replay_relation_v1=replay_relation_v1) != audit:
            raise ValueError("local train: freshly regraded episode audit differs")
        if synthesis_selected:
            from .episode_notebook import regrade_synthesis

            receipt = regrade_synthesis(tree, actors[tree.problem_id], verifier, tokenizer)
            if tree.problem_id in update["routes"] and update.get("synthesis_regrade_receipts", {}).get(tree.problem_id) != receipt:
                raise ValueError("local train: freshly regraded synthesis receipt differs")
        objective = protocol["notebook_objective"]
        fresh = _score_notebook_objective(tree, actors[tree.problem_id], verifier, objective,
                                           replay_relation_v1=replay_relation_v1)
        if fresh != tree.notebook["learning_objective"]:
            raise ValueError("local train: freshly regraded fact objective differs")
        if tree.problem_id not in update["routes"]:
            continue
        seen_trees.add(tree.problem_id)
        regraded_trees.append(tree)
        if (protocol.get("notebook_route_rule") != STRICT_ROUTE_RULE
                or audit.get("routing_rule") != STRICT_ROUTE_RULE or audit.get("admission") == "UNRESOLVED"
                or tree.policy_version != update["step_before"]
                or update["routes"][tree.problem_id] is not (tree.route == "VISION_ON")):
            raise ValueError("local train: fact-credit policy or routing admission differs")
        scores = notebook_objective_scores(tree, objective)
        regraded_scores.update(scores)
        signal_groups += notebook_signal_groups(tree, scores)
        if fresh != update.get("notebook_objective_receipts", {}).get(tree.problem_id):
            raise ValueError("local train: freshly regraded fact objective differs")
        for group in tree.groups:
            if group.usable:
                original_actions.update(sample.sample_id for sample in group.samples)
            if group.kind != "DIAG" or not group.usable:
                continue
            if update["effective_rewards"].get(group.group_id) != (
                    [0.] * len(group.samples) if control else [scores[s.sample_id]["effective_reward"] for s in group.samples]):
                raise ValueError("local train: notebook effective rewards differ")
            if control:
                continue
            full_credit = group_credit(group.samples, scores, tokenizer)
            for sample in group.samples:
                if sample.sample_id in credits:
                    raise ValueError("local train: duplicate notebook sample")
                credit = credits[sample.sample_id] = full_credit[sample.sample_id]
                delta = deltas.get(sample.sample_id)
                if delta is None or type(delta) not in (int, float) or not math.isfinite(delta):
                    raise ValueError("local train: missing finite MIS evidence")
                row = backward.get(sample.sample_id)
                keep = bool(torch.tensor(delta, dtype=torch.float32).abs() <= protocol["loss"]["mismatch"]["delta_seq"])
                if not keep:
                    if row is not None:
                        raise ValueError("local train: MIS-rejected notebook has backward evidence")
                    continue
                retained.add(sample.sample_id)
                if (row is None or row.get("group_id") != group.group_id
                        or row.get("vision_on") is not update["routes"][tree.problem_id]
                        or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                               for key in ("advantage", "strict_reward", "effective_reward", "atomic_reward"))
                        or type(row.get("action_tokens")) is not int
                        or row.get("action_tokens") != len(sample.completion_token_ids)
                        or row.get("advantage") != credit["strict_advantage"]
                        or row.get("strict_reward") != sample.reward
                        or row.get("effective_reward") != scores[sample.sample_id]["effective_reward"]
                        or row.get("atomic_reward") != scores[sample.sample_id]["atomic_reward"]
                        or row.get("objective_score_sha256") != _hash(scores[sample.sample_id])
                        or row.get(credit_hash_key) != _hash(credit)
                        or row.get("nonzero_partial_tokens") != sum(v != 0 for v in credit["partial_token_advantages"])):
                    raise ValueError("local train: fact-credit backward evidence differs")
                before, after = row["gradient_before"], row["gradient_after"]
                if (set(before) != {"L", "M", "V"} or set(after) != set(before)
                        or any(type(before[g]) is not list or type(after[g]) is not list
                               or not before[g] or len(before[g]) != len(after[g]) for g in before)
                        or any(v is not None and (type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None)
                               for state in (before, after) for values in state.values() for v in values)):
                    raise ValueError("local train: incomplete notebook gradient evidence")
                changed = [g for g in before if before[g] != after[g]]
                if (row.get("changed_groups") != changed
                        or not update["routes"][tree.problem_id] and "V" in changed):
                    raise ValueError(failure)
                if synthesis_selected and not assessment_selected:
                    relation = row.get("relationship_gradient", {})
                    if (type(relation) is not dict or relation.get("scope") != "relationship_policy_loss_only"
                            or type(relation.get("measured")) is not bool
                            or relation["measured"] is False and set(relation) != {"scope", "measured"}):
                        raise ValueError("local train: explicit relationship-only gradient measurement required")
                    if relation.get("measured") is True:
                        norms = relation.get("group_norms", {})
                        positive = sum(v > 0 for v in credit["relationship_token_advantages"])
                        types = list(dict.fromkeys(entry["credit_type"] for entry in credit["relationships"]
                                                   if entry["coefficient"] != 0 and entry["token_indices"]))
                        positive_types = list(dict.fromkeys(entry["credit_type"] for entry in credit["relationships"]
                                                            if _positive_error_relationship(entry)))
                        effect = ("OUTPUT_FINALIZATION_ONLY" if set(types) == {"OUTPUT_FINALIZATION"} else
                                  "NATIVE_ERROR_RELATIONSHIP_ONLY" if set(types) == {"NATIVE_ERROR_RELATIONSHIP"} else
                                  "COMBINED_RELATIONSHIP_TYPES")
                        if (relation.get("scope") != "relationship_policy_loss_only"
                                or relation.get("finite") is not True or type(norms) is not dict or set(norms) != {"L", "M", "V"}
                                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in norms.values())
                                or relation.get("nonzero_groups") != [g for g in ("L", "M", "V") if norms[g] > 0]
                                or type(relation.get("positive_tokens")) is not int or relation["positive_tokens"] != positive
                                or type(relation.get("token_count")) is not int
                                or relation["token_count"] != len(sample.completion_token_ids)
                                or relation.get("credit_sha256") != _hash(credit)
                                or relation.get("credit_types") != types or relation.get("positive_credit_types") != positive_types
                                or relation.get("effect_scope") != effect
                                or relation.get("basis") != "nonzero_mapped_coefficients_not_per_type_gradient_decomposition"
                                or not update["routes"][tree.problem_id] and norms["V"] != 0):
                            raise ValueError("local train: measured relationship-only gradient evidence differs")
                        contributing |= (bool(relation["nonzero_groups"]) and positive > 0
                            and any(_positive_error_relationship(entry) for entry in credit["relationships"]))
                elif not assessment_selected:
                    contributing |= (bool(changed) and row.get("reward_gradient_isolated") is True
                        and credit["strict_advantage"] == 0
                        and any(v > 0 for v in credit["partial_token_advantages"]))
    effective_mis_deltas(update, protocol, trees=regraded_trees)
    if assessment_selected and (set(deltas) != original_actions
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in deltas.values())
            or set(update.get("notebook_objective_receipts", {})) != seen_trees
            or set(update.get("synthesis_regrade_receipts", {})) != seen_trees
            or update.get("n_groups_by_kind") != {kind: sum(group.kind == kind and group.usable
                for tree in regraded_trees for group in tree.groups) for kind in ("T1", "T2", "T3", "DIAG")}
            or update.get("effective_rewards") != effective_rewards(
                [group for tree in regraded_trees for group in tree.groups if group.usable], "ALG-NB3",
                protocol["local_budget"]["seed"], update["step_before"],
                notebook_reward_enabled=not control, notebook_scores=regraded_scores)):
        raise ValueError("local train: complete finite original assessment action MIS evidence required")
    if control:
        assessment_control_update_evidence(update, protocol, regraded_trees)
        if (seen_trees != set(update["routes"]) or tree_ids is not None and audited_trees != tree_ids
                or signal_groups != result.get("notebook_signal_groups")):
            raise ValueError("local train: original assessment control pair replay differs")
        return paths
    if assessment_selected:
        from .local_run import assessment_update_evidence

        contributing = assessment_update_evidence(update, protocol, trees=regraded_trees,
            fresh_credits=credits, parameter_inventory=parameter_inventory)["assessment_only_gradient_verified"]
    if (seen_trees != set(update["routes"]) or set(backward) != retained
            or tree_ids is not None and audited_trees != tree_ids
            or update.get(credit_key) != credits or not contributing
            or type(result.get("notebook_signal_groups")) is not int
            or signal_groups != result["notebook_signal_groups"]):
        raise ValueError(failure)
    return paths


def training_parent_gate(capture: Path, reload: Path, *, admission: str = STRICT_ADMISSION) -> dict:
    """Exploratory functional admission never certifies strict notebook correctness."""
    if admission in REPAIR_ADMISSIONS:
        from .repair_replay import replay_capture

        require_s = admission == REPAIR_ADMISSION
        capture, reload = Path(capture).absolute(), Path(reload).absolute()
        snapshots = {path: _repair_read(path) for path in (
            capture / "launch.json", capture / "result.json", capture / "reload_expected.json",
            reload / "result.json", reload / "forward_probes.json", reload / "sync.json")}
        gate = parent_gate(capture, reload, require_notebook_learning=False)
        launch = gate["launch"]
        if (checkpoint.digest(launch) != checkpoint.digest(snapshots[capture / "launch.json"][0])
                or any(checkpoint.file_hash(path) != sha for path, (_, sha) in snapshots.items())):
            raise ValueError("local train: minimum source changed during functional validation")
        protocol = restore_protocol(launch["protocol"])
        execution_admission({"admission_policy": admission}, protocol)
        result, restored = snapshots[capture / "result.json"][0], snapshots[reload / "result.json"][0]
        expected = snapshots[capture / "reload_expected.json"][0]
        if (type(result.get("optimizer_updates")) is not int or result["optimizer_updates"] != 1
                or type(expected.get("step")) is not int or expected["step"] != 1
                or type(restored.get("step")) is not int or restored["step"] != 1
                or type(restored.get("pid")) is not int or restored["pid"] <= 0
                or same_process(restored, launch) or owner_alive(restored)
                or type(restored.get("capture_pid")) is not int or restored["capture_pid"] != launch["pid"]
                or type(launch.get("problem_ids")) is not list or len(set(launch["problem_ids"])) != 2
                or restored.get("repair_functional_verified") is not True
                or type(restored.get("S_gradient_verified")) is not bool
                or restored.get("repair_learning_verified") is not restored["S_gradient_verified"]
                or require_s and restored["S_gradient_verified"] is not True
                or restored.get("benefit_verified") is not False):
            raise ValueError("local train: original closed minimum R1 functional and repair evidence required")
        prepared, _, actors = _launch_preparation(launch, protocol)
        _repair_probe_binding(expected.get("forward_probes"), _read(reload / "sync.json"), 1,
                              probe_source(prepared, actors), batch_rows=scale_options.options(protocol)["rollout_batch_rows"],
                              image_counts=probe_counts(prepared))
        metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
        # v2: the trained minimal pair and its separate probe set; otherwise the whole selection as before.
        replay = replay_capture(capture, protocol=protocol, prepared=prepared,
                                parameter_inventory=metadata["inventory"]["trainable"], require_s=require_s,
                                **({"actors": minimal_training_actors(prepared, actors),
                                    "probe_actors": probe_source(prepared, actors)} if pool_v2(prepared) else {"actors": actors}))
        evidence_sha = checkpoint.file_hash(capture / "repair_update_evidence.json")
        if (type(replay.get("S_gradient_verified")) is not bool
                or require_s and replay["S_gradient_verified"] is not True
                or restored["S_gradient_verified"] is not replay["S_gradient_verified"]
                or restored.get("repair_replay_sha256") != replay.get("sha256")
                or restored.get("repair_update_evidence_sha256") != evidence_sha
                or restored.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or restored.get("repair_protocol_sha256") != protocol["sha256"]):
            raise ValueError("local train: fresh minimum repair replay differs from actual reload evidence")
        if (any(replay["paths"].get(name) != snapshots[capture / name][1]
                for name in ("launch.json", "result.json", "reload_expected.json"))
                or any(checkpoint.file_hash(path) != sha for path, (_, sha) in snapshots.items())):
            raise ValueError("local train: cached minimum/reload source versions differ from fresh replay")
        paths = gate["files"] + [{"path": str(capture / name), "sha256": sha} for name, sha in replay["paths"].items()]
        files = {row["path"]: row for row in paths}
        return gate | {"admission_policy": admission, "repair_functional_verified": True,
            "S_gradient_verified": replay["S_gradient_verified"],
            "repair_learning_verified": replay["S_gradient_verified"], "benefit_verified": False,
            "repair_replay": replay, "files": list(files.values()), "strict_notebook_correctness_verified": False}
    if admission == STRICT_ADMISSION:
        return parent_gate(capture, reload, require_notebook_learning=True)
    if admission not in (EXPLORATORY_ADMISSION, SYNTHESIS_CREDIT_ADMISSION, ASSESSMENT_CREDIT_ADMISSION,
                         VERIFIED_UPDATE_EXPLORATION):
        raise ValueError("local train: unknown admission policy")
    capture = Path(capture).absolute()
    gate = (parent_gate(capture, reload, require_notebook_learning=False)
            if admission == VERIFIED_UPDATE_EXPLORATION else parent_gate(capture, reload))
    launch = gate["launch"]
    protocol = restore_protocol(launch["protocol"])
    execution_admission({"admission_policy": admission}, protocol)
    from transformers import AutoTokenizer
    from .answers import VERSION as ANSWER_VERSION
    from .launch import descriptor, read_descriptor
    from .local_run import read_actor_descriptor, read_execution_actors
    from .schema import ProblemTree, from_json
    from .verifier import VERSION as VERIFIER_VERSION
    from .verifier_service import VerifierService

    preparation = Path(launch["preparation"])
    prepared = _read(preparation)
    expected_binding = None
    if admission in (VERIFIED_UPDATE_EXPLORATION, ASSESSMENT_CREDIT_ADMISSION):
        contract = launch.get("execution_contract")
        if (admission == VERIFIED_UPDATE_EXPLORATION and (
                type(contract) is not dict or launch.get("execution_contract_sha256") != checkpoint.digest(contract)
                or contract != minimum_execution_contract(admission, protocol))):
            raise ValueError("local train: explicit minimum execution contract required")
        if admission == ASSESSMENT_CREDIT_ADMISSION and any(
                key in launch for key in ("execution_contract", "execution_contract_sha256")):
            raise ValueError("local train: assessment minimum uses the original implicit execution contract")
        historical_sources = read_descriptor(prepared["validated_data"]).get("code_sha256")
        if (type(historical_sources) is not dict or not historical_sources
                or any(type(name) is not str or not name or type(sha) is not str
                       or re.fullmatch(r"[0-9a-f]{64}", sha) is None for name, sha in historical_sources.items())):
            raise ValueError("local train: validated minimum source mapping required")
        expected_binding = minimum_binding(prepared, protocol, contract)
        expected_binding["code_sha256"] = checkpoint.digest(historical_sources)
        if expected_binding != launch["binding"]:
            raise ValueError("local train: minimum source/preparation/execution binding differs")
    if (launch["binding"]["data_sha256"] != (expected_binding["data_sha256"] if expected_binding else checkpoint.digest(prepared))
            or launch["binding"]["protocol_sha256"] != protocol["sha256"]
            or launch["binding"]["prompts_sha256"] != protocol["prompts_sha256"]
            or prepared.get("format") != "notebook-rl-local-data-v1"
            or prepared.get("status") != "LOCAL_DATA_REVALIDATED"
            or prepared.get("local_protocol_sha256") != protocol["sha256"]):
        raise ValueError("local train: minimal preparation binding differs")
    request = read_descriptor(prepared["data_request"])
    checked = read_descriptor(prepared["validated_data"])
    originals = read_actor_descriptor(prepared["actors"])
    actors = read_execution_actors(prepared, originals, protocol)
    if (prepared["build_root"] != request["build_root"] or checked.get("blockers") != []
            or checked.get("status") != "VALIDATED_FOR_MEASUREMENT"
            or checked.get("measurement_allowed") is not True
            or checked.get("training_allowed") is not False
            or checked["input_binding"]["build_manifest_sha256"] != checkpoint.file_hash(Path(prepared["build_root"]) / "manifest.yaml")
            or any(json.loads(to_json(a)) not in checked["actors"] for a in originals)
            or not set(launch["problem_ids"]) <= {a.problem_id for a in actors}):
        raise ValueError("local train: minimal input validation differs")
    if admission == VERIFIED_UPDATE_EXPLORATION:
        restored = _read(Path(reload) / "result.json")
        update, result = _read(capture / "update.json"), _read(capture / "result.json")
        if (launch["problem_ids"] != [actor.problem_id for actor in minimal_actors(actors)]
                or len(set(launch["problem_ids"])) != 2
                or type(restored.get("pid")) is not int or restored["pid"] <= 0 or owner_alive(restored)
                or restored.get("execution_contract") != contract
                or restored.get("execution_contract_sha256") != checkpoint.digest(contract)
                or restored.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or update.get("step_before") != 0 or update.get("step_after") != 1
                or result.get("retained_learning_signal") != retained_update_signal(update, protocol)):
            raise ValueError("local train: exact two-input update and matching explicit-policy reload required")
        paths = [preparation, descriptor(prepared["data_request"]), descriptor(prepared["validated_data"]),
                 descriptor(prepared["actors"]), descriptor(prepared["execution_actors"]), Path(prepared["build_root"]) / "manifest.yaml"]
        return gate | {"admission_policy": admission, "verified_update_exploration_admitted": True,
            "retained_policy_coefficient_signal_verified": True, "strict_notebook_correctness_verified": False,
            "files": gate["files"] + [{"path": str(p), "sha256": checkpoint.file_hash(p)} for p in paths]}
    tokenizer = AutoTokenizer.from_pretrained(launch["model_path"], local_files_only=True, trust_remote_code=False)
    root = Path(prepared["build_root"])
    forbidden = [Path(request["index"]["path"]).parent, Path.home() / "mmdl-data/evaluation"]
    trees = [from_json(ProblemTree, json.dumps(row["raw"], allow_nan=False))
             for path in raw_journal_paths(capture) if (row := _read(path)).get("kind") == "TREE"]
    if admission == ASSESSMENT_CREDIT_ADMISSION:
        restored = _read(Path(reload) / "result.json")
        update, result = _read(capture / "update.json"), _read(capture / "result.json")
        if (launch["problem_ids"] != [actor.problem_id for actor in minimal_actors(actors)]
                or len(set(launch["problem_ids"])) != 2
                or [tree.problem_id for tree in trees] != launch["problem_ids"]
                or any(tree.policy_version != 0 for tree in trees)
                or update.get("step_before") != 0 or update.get("step_after") != 1
                or result.get("attempted_ids") != launch["problem_ids"]
                or type(restored.get("pid")) is not int or restored["pid"] <= 0 or owner_alive(restored)):
            raise ValueError("local train: assessment requires exact two-input original TREEs and independent closed reload")
    versions = {sample.result.verifier_version for tree in trees for group in tree.groups
                for sample in group.samples if sample.result is not None}
    old_version = ANSWER_VERSION + "+native-graph-relations-v1"
    replay_relation_v1 = old_version in versions
    if replay_relation_v1 and (versions != {old_version}
            or len(trees) != len(launch["problem_ids"])
            or {tree.problem_id for tree in trees} != set(launch["problem_ids"])
            or re.fullmatch(r"[0-9a-f]{64}", launch["binding"].get("code_sha256", "")) is None
            or launch["binding"]["code_sha256"] == checkpoint.digest(approval.code_hashes())):
        raise ValueError("local train: homogeneous historical relation-v1 parent required")
    compatibility = {}
    with VerifierService(root / "train", forbidden_roots=forbidden, timeout=60) as verifier:
        if verifier.metadata["vault_manifest_sha256"] != _read(root / "manifest.yaml")["split_manifests"]["train"]:
            raise ValueError("local train: minimal verifier identity differs")
        paths = fact_credit_learning_gate(Path(capture), _read(Path(capture) / "result.json"),
            _read(Path(capture) / "update.json"), protocol,
            actors={a.problem_id: a for a in actors}, verifier=verifier, tokenizer=tokenizer,
            replay_relation_v1=replay_relation_v1, admission=admission if admission == ASSESSMENT_CREDIT_ADMISSION else None)
        if replay_relation_v1:
            compatibility = {"historical_verifier_replay": {
                "version": "native-graph-relations-v1-to-v3-semantic-replay-v1",
                "from_verifier_version": old_version, "to_verifier_version": VERIFIER_VERSION,
                "current_verifier_code_sha256": verifier.metadata["code_sha256"],
                "comparison": "all-result-fields-except-the-exact-allowed-version-transition",
                "semantic_regrade_scope": "every-TREE-solution-and-submitted-DIAG-projection; full-audit-strict-objective-credit-MIS-gradient-equality",
                "tree_ids": [tree.problem_id for tree in trees]}}
    paths += [preparation, descriptor(prepared["data_request"]), descriptor(prepared["validated_data"]),
              descriptor(prepared["actors"]), descriptor(prepared["execution_actors"]), root / "manifest.yaml"]
    verified_key = ("positive_relationship_credit_and_isolated_relationship_gradient_verified"
                    if admission == SYNTHESIS_CREDIT_ADMISSION else "positive_fact_credit_and_isolated_note_gradient_verified")
    summary = (relationship_learning_summary(_read(capture / "update.json"))
               if admission == SYNTHESIS_CREDIT_ADMISSION else {})
    if admission == ASSESSMENT_CREDIT_ADMISSION:
        from .local_run import assessment_update_evidence

        summary = assessment_update_evidence(_read(capture / "update.json"), protocol) | {
            "accurate_whole_notebook_verified": False, "self_improvement_verified": False}
        verified_key = "assessment_learning_verified"
    return gate | compatibility | summary | {"admission_policy": admission, verified_key: True,
        "strict_notebook_correctness_verified": False,
        "files": gate["files"] + [{"path": str(p), "sha256": checkpoint.file_hash(p)} for p in paths]}


def fixed_pairs(actors: list) -> list[list]:
    if len(actors) != 10 or len({a.problem_id for a in actors}) != 10:
        raise ValueError("local train: exactly ten unique fixed actors required")
    return [actors[i:i+2] for i in range(0, 10, 2)]


def eligible_trees(trees: list, arm: str, *, notebook_objective: dict | None = None) -> list:
    from .updates import partition_learning_trees

    if arm not in ("notebook", "answer-only"):
        raise ValueError("local train: explicit notebook reward arm required")
    return partition_learning_trees(trees, include_notebooks=arm == "notebook",
                                    notebook_objective=notebook_objective)[0]


def training_binding(prepared: dict, protocol: dict, contract: dict) -> dict:
    execution_max_seconds(contract, protocol)
    if contract.get("admission_policy") in REPAIR_ADMISSIONS:
        execution_admission(contract, protocol)
    return _binding(prepared, protocol) | {"data_sha256": checkpoint.digest({
        "preparation_sha256": checkpoint.digest(prepared),
        "execution_contract_sha256": checkpoint.digest(contract)})}


def native_collection_gate(prepared: dict, protocol: dict, readiness: dict, actors: list) -> dict:
    """Permission to collect the original cohort is separate from any learning evidence."""
    from .launch import read_descriptor

    execution_admission({"admission_policy": FIXED_TEN_NATIVE_EXPLORATION}, protocol)
    selected = read_descriptor(prepared["selection"])
    original = read_descriptor(prepared["previous_selection"])
    pairs = [[a.problem_id for a in pair] for pair in fixed_pairs(actors)]
    if ([a.problem_id for a in actors] != original.get("problem_ids")
            or selected.get("problem_ids") != original.get("problem_ids")
            or selected.get("previous_selection") != prepared["previous_selection"]
            or selected.get("local_protocol_sha256") != protocol["sha256"]):
        raise ValueError("local train: original fixed ten collection and order must be preserved")
    return {"status": "FIXED_TEN_NATIVE_COLLECTION_ADMITTED", "collection_allowed": True,
        "admission_policy": FIXED_TEN_NATIVE_EXPLORATION,
        "preparation_sha256": checkpoint.digest(prepared), "cache_readiness_sha256": checkpoint.digest(readiness),
        "code_sha256": checkpoint.digest(approval.code_hashes()), "protocol_sha256": protocol["sha256"],
        "pair_ids": pairs,
        "native_relationship_learning_verified": False, "strict_notebook_correctness_verified": False,
        "self_improvement_verified": False}


def validate_native_collection(contract: dict, prepared: dict, protocol: dict,
                               readiness: dict, actors: list) -> dict:
    gate = native_collection_gate(prepared, protocol, readiness, actors)
    if (contract.get("admission_policy") != FIXED_TEN_NATIVE_EXPLORATION
            or contract.get("stage") != "FIXED_TEN_FROM_FRESH_F0" or contract.get("arm") != "notebook"
            or contract.get("notebook_reward_enabled") is not True
            or type(contract.get("max_optimizer_updates")) is not int or contract["max_optimizer_updates"] != 5
            or contract.get("pair_ids") != gate["pair_ids"]
            or contract.get("parent_gate") is not None or contract.get("matched_notebook_arm") is not None
            or contract.get("authority") != protocol["local_authorization"]
            or contract.get("initial_checkpoint") != "standalone NF4 F0 with newly seeded adapters, not minimal-trained weights"
            or contract.get("no_signal_policy") != "record; consume the fixed pair; no replacement or forced update"
            or contract.get("same_generated_notebook_checks_and_kl") is not True
            or contract.get("collection_gate") != gate):
        raise ValueError("local train: exact source-bound native collection execution contract required")
    execution_max_seconds(contract, protocol)
    return gate


def assessment_control_update_evidence(update: dict, protocol: dict, trees: list) -> None:
    """The matched control keeps DIAG actions for KL and removes their reward credit."""
    from .updates import ASSESSMENT_REQUIREMENT

    if (protocol.get("notebook_update_requirement") != ASSESSMENT_REQUIREMENT
            or update.get("notebook_update_requirement") != ASSESSMENT_REQUIREMENT
            or update.get("notebook_objective") != protocol.get("notebook_objective")
            or update.get("notebook_reward_enabled") is not False
            or update.get("assessment_commit_required") is not False
            or update.get("assessment_learning_verified") is not False
            or update.get("notebook_backward") != []
            or "assessment_aggregate_gradient" in update
            or any(update.get(key) for key in ("synthesis_token_credit", "fact_token_credit", "retained_assessment_signal"))
            or any(update.get("effective_rewards", {}).get(group.group_id) != [0.] * len(group.samples)
                   for tree in trees if tree.problem_id in update.get("routes", {})
                   for group in tree.groups if group.kind == "DIAG" and group.usable)):
        raise ValueError("local train: assessment answer-only control must have zero DIAG reward and no assessment credit")


def _repair_training_replay(capture: Path, result: dict, contract: dict, protocol: dict, *, parameter_inventory) -> dict:
    if scale_options.scaled(protocol):
        return _scaled_training_replay(capture, result, contract, protocol, parameter_inventory=parameter_inventory)
    from .repair_replay import checkpoint_snapshot, replay_trees, validate_journal
    from .schema import ProblemTree, from_json
    from .repair_evidence import repair_update_evidence

    execution_admission(contract, protocol)
    capture = Path(capture)
    snapshots = {name: _repair_read(capture / name) for name in (
        "launch.json", "result.json", "repair_preparation_index.json", "repair_update_evidence.json",
        "reload_expected.json", "initialization.json", "inventory.json", "updated_forward_probes.json")}
    paths = {name: sha for name, (_, sha) in snapshots.items()}
    launch, index = snapshots["launch.json"][0], snapshots["repair_preparation_index.json"][0]
    if checkpoint.digest(result) != checkpoint.digest(snapshots["result.json"][0]):
        raise ValueError("local train: cached repair result differs from its replayed file version")
    pairs = contract.get("pair_ids")
    if (type(pairs) is not list or len(pairs) != 5 or any(type(pair) is not list or len(pair) != 2 for pair in pairs)
            or any(type(pid) is not str or not pid for pair in pairs for pid in pair)
            or len({pid for pair in pairs for pid in pair}) != 10
            or contract.get("arm") not in ("notebook", "answer-only")
            or contract.get("notebook_reward_enabled") is not (contract["arm"] == "notebook")
            or type(result.get("optimizer_updates")) is not int or not 1 <= result["optimizer_updates"] <= 5
            or result.get("attempted_ids") != [pid for pair in pairs for pid in pair]
            or launch.get("execution_contract") != contract or launch.get("execution_contract_sha256") != checkpoint.digest(contract)
            or index.get("version") != "repair-local-training-preparation-v1"
            or index.get("protocol_sha256") != protocol["sha256"]
            or index.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
            or index.get("control") is not (contract["arm"] == "answer-only")
            or result.get("repair_preparation_index_sha256") != checkpoint.file_hash(capture / "repair_preparation_index.json")
            or result.get("arm") != contract["arm"] or launch.get("protocol") != json.loads(json.dumps(protocol))
            or type(index.get("trees")) is not list or any(type(row) is not dict for row in index["trees"])
            or [row.get("problem_id") for row in index["trees"]] != result["attempted_ids"]
            or [path.name for path in sorted(capture.glob("pair-*.json"))] != [f"pair-{i:02d}.json" for i in range(5)]):
        raise ValueError("local train: complete original repair five-pair execution binding required")
    prepared, _, actors = _launch_preparation(launch, protocol)
    if ([actor.problem_id for actor in actors] != result["attempted_ids"]
            or launch.get("binding") != training_binding(prepared, protocol, contract)
            or launch.get("preparation_sha256") != checkpoint.digest(prepared)):
        raise ValueError("local train: repair original fixed-ten actor order differs")
    trees, preparations = [], {}
    expected = snapshots["reload_expected.json"][0]
    if (snapshots["inventory.json"][0].get("trainable") != parameter_inventory
            or snapshots["updated_forward_probes.json"][0] != expected.get("forward_probes")):
        raise ValueError("local train: actual repair inventory or saved forward probes differ")
    envelope = {"version": "repair-local-training-update-v1", "protocol_sha256": protocol["sha256"],
        "capture_binding_sha256": checkpoint.digest(launch["binding"]),
        "preparation_index_sha256": paths["repair_preparation_index.json"],
        "control": index["control"], "commits": result.get("repair_commits")}
    if (snapshots["repair_update_evidence.json"][0] != envelope
            or result.get("repair_update_evidence_sha256") != paths["repair_update_evidence.json"]):
        raise ValueError("local train: repair aggregate evidence envelope differs")
    for ordinal, link in enumerate(index["trees"]):
        if (set(link) != {"problem_id", "tree_file", "tree_file_sha256", "preparation_file",
                          "preparation_file_sha256", "preparation_receipt_sha256"}
                or type(link["tree_file"]) is not str or re.fullmatch(r"raw-[0-9]{4}\.json", link["tree_file"]) is None
                or link["preparation_file"] != f"repair-preparation-{ordinal:04d}.json"):
            raise ValueError("local train: repair original TREE/preparation link differs")
        for field in ("tree_file", "preparation_file"):
            name = link[field]
            value, paths[name] = _repair_read(capture / name)
            if paths[name] != link[field + "_sha256"]:
                raise ValueError("local train: repair original file changed")
            if field == "preparation_file":
                preparations[link["problem_id"]] = value
            else:
                row = value
        if row.get("kind") != "TREE":
            raise ValueError("local train: repair original TREE journal required")
        trees.append(from_json(ProblemTree, json.dumps(row["raw"], allow_nan=False)))
    paths.update(validate_journal(capture, index["raw_records"], trees))
    fresh = replay_trees(trees, actors, protocol, prepared, Path(launch["model_path"]), control=contract["arm"] == "answer-only")
    for tree, link in zip(trees, index["trees"], strict=True):
        receipt = preparations[tree.problem_id]
        if (tree.problem_id != link["problem_id"] or receipt != fresh[tree.problem_id]
                or receipt.get("sha256") != link["preparation_receipt_sha256"]):
            raise ValueError("local train: fresh original repair preparation differs")
    step, summaries, commits, raw_links, tree_links = 0, [], [], [], []
    discarded, unresolved, signal_free = [], [], []
    answer_route, answer_ids = "repair_route_policy" in protocol, []
    incomplete: list[str] = []
    previous = snapshots["initialization.json"][0]["trainable_sha256"]
    for pair_index, ids in enumerate(pairs):
        pair_trees = trees[pair_index * 2:pair_index * 2 + 2]
        pair_file, update_file = f"repair-pair-{pair_index:02d}.json", f"pair-{pair_index:02d}.json"
        pair, paths[pair_file] = _repair_read(capture / pair_file)
        update, paths[update_file] = _repair_read(capture / update_file)
        if (pair.get("version") != "repair-local-training-pair-v1" or type(pair.get("pair_index")) is not int
                or pair["pair_index"] != pair_index or type(pair.get("step")) is not int or pair["step"] != step
                or pair.get("protocol_sha256") != protocol["sha256"] or pair.get("control") is not index["control"]
                or pair.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or pair.get("trees") != index["trees"][pair_index * 2:pair_index * 2 + 2]
                or type(pair.get("raw_records")) is not list or update.get("attempted_ids") != ids
                or [tree.problem_id for tree in pair_trees] != ids or any(tree.policy_version != step for tree in pair_trees)):
            raise ValueError("local train: repair pair policy, original order or source binding differs")
        raw_links.extend(pair["raw_records"])
        tree_links.extend(pair["trees"])
        eligible = [tree for tree in pair_trees if fresh[tree.problem_id]["admitted"] and fresh[tree.problem_id]["has_signal"]]
        bad = [tree.problem_id for tree in pair_trees if tree not in eligible]
        unknown = [pid for pid in ids if fresh[pid]["admission"] == "UNRESOLVED"]
        zero = [pid for pid in ids if fresh[pid]["admitted"] and not fresh[pid]["has_signal"]]
        answered = [pid for pid in ids if fresh[pid]["admission"] == "ANSWER_VERIFIED"]
        incomplete.extend(pid for pid in ids if any(reason != "UNRESOLVED_ROUTING" for reason in fresh[pid]["admission_reasons"]))
        if (update.get("discarded_ids") != bad or update.get("unresolved_ids") != unknown or update.get("signal_free_ids") != zero
                or (update.get("answer_verified_ids") != answered if answer_route else "answer_verified_ids" in update)):
            raise ValueError("local train: repair fresh pair admission or original discarded members differ")
        answer_ids.extend(answered)
        discarded.extend(bad)
        unresolved.extend(unknown)
        signal_free.extend(zero)
        if update.get("status") != "UPDATED_UNSYNCHRONIZED":
            if (update.get("status") not in ("UNRESOLVED_ROUTING", "NO_LEARNING_SIGNAL", "NO_ADMITTED_REPAIR_TREES", "NO_ACTIONS", "NO_MATCHED_ACTIONS")
                    or type(update.get("step", update.get("step_before"))) is not int
                    or update.get("step", update.get("step_before")) != step
                    or type(update.get("step_after", step)) is not int or update.get("step_after", step) != step):
                raise ValueError("local train: uncommitted repair pair changed the update sequence")
            continue
        if (not eligible or type(update.get("step_before")) is not int or update["step_before"] != step
                or type(update.get("step_after")) is not int or update["step_after"] != step + 1):
            raise ValueError("local train: repair committed pair sequence differs")
        summary = repair_update_evidence(update, protocol, trees=eligible,
            preparations={tree.problem_id: fresh[tree.problem_id] for tree in eligible},
            parameter_inventory=parameter_inventory, require_s=False, control=index["control"])
        evidence_file = f"repair-evidence-{pair_index:02d}.json"
        recorded, paths[evidence_file] = _repair_read(capture / evidence_file)
        if recorded != summary:
            raise ValueError("local train: recorded repair evidence differs from original pair replay")
        commits.append({"pair_index": pair_index, "file": evidence_file, "sha256": paths[evidence_file],
                        "S_gradient_verified": summary["S_gradient_verified"]})
        summaries.append(summary)
        changes_file, sync_file = f"changes-{pair_index:02d}.json", f"sync-{pair_index:02d}.json"
        changes, paths[changes_file] = _repair_read(capture / changes_file)
        sync, paths[sync_file] = _repair_read(capture / sync_file)
        verified = verify_changes(changes["before"], changes["after"], update)
        if changes["before"] != previous or any(changes.get(key) != value for key, value in verified.items()):
            raise ValueError("local train: actual repair parameter chain differs")
        previous = changes["after"]
        if (type(sync) is not list or len(sync) != len(probe_counts(prepared))
                or any(type(row.get("policy_version")) is not int or row["policy_version"] != step + 1 for row in sync)):
            raise ValueError("local train: actual repair synchronized probe sequence differs")
        _repair_probe_binding(expected.get("forward_probes"), sync, step + 1, probe_source(prepared, actors),
                              image_counts=probe_counts(prepared))
        committed, checkpoint_paths = checkpoint_snapshot(capture, f"checkpoint-pair-{pair_index:02d}", binding=launch["binding"])
        paths.update(checkpoint_paths)
        cursor = committed["cursor"]
        if (type(committed["step"]) is not int or committed["step"] != step + 1
                or committed["protocol"] != launch["protocol"] or type(cursor.get("position")) is not int
                or cursor["position"] != (pair_index + 1) * 2 or cursor.get("sync_pending") is not False
                or cursor.get("problem_ids") != result["attempted_ids"]
                or cursor.get("consumed_problem_ids") != result["attempted_ids"][:(pair_index + 1) * 2]
                or cursor.get("discarded_problem_ids") != discarded
                or committed["state_sha256"] != {name: sha for group in changes["after"].values() for name, sha in group.items()}
                or committed["inventory"] != snapshots["inventory.json"][0]
                or committed["base_sha256"] != snapshots["initialization.json"][0]["base_sha256"]
                or committed["inventory"].get("quantization") != snapshots["initialization.json"][0]["quantization"]):
            raise ValueError("local train: actual repair checkpoint cursor differs")
        step += 1
    if (step != result["optimizer_updates"] or commits != result.get("repair_commits")
            or result.get("S_gradient_verified") is not any(row["S_gradient_verified"] for row in summaries)
            or any(result.get(key) is not False for key in ("repair_functional_verified", "repair_learning_verified", "benefit_verified"))
            or raw_links != index["raw_records"] or tree_links != index["trees"]
            or discarded != result.get("discarded_ids") or unresolved != result.get("unresolved_ids")
            or signal_free != result.get("signal_free_ids") or incomplete != result.get("incomplete_ids")
            or (result.get("answer_verified_ids") != answer_ids
                or result.get("S_text_gradient_verified") is not any(row["S_text_gradient_verified"] for row in summaries)
                if answer_route else "answer_verified_ids" in result)):
        raise ValueError("local train: repair counted commits or full fixed-ten history differs")
    if (type(expected.get("step")) is not int or expected["step"] != step
            or expected.get("trainable_sha256") != previous
            or expected.get("repair_protocol_sha256") != protocol["sha256"]
            or expected.get("repair_preparation_index_sha256") != paths["repair_preparation_index.json"]
            or expected.get("repair_update_evidence_sha256") != paths["repair_update_evidence.json"]
            or any(checkpoint.file_hash(capture / name) != sha for name, sha in paths.items())):
        raise ValueError("local train: repair final snapshot or files changed during replay")
    final, checkpoint_paths = checkpoint_snapshot(capture, "checkpoint", binding=launch["binding"])
    paths.update(checkpoint_paths)
    cursor = {"epoch": 0, "position": 10, "discarded_problem_ids": discarded,
              "problem_ids": result["attempted_ids"], "consumed_problem_ids": result["attempted_ids"], "sync_pending": False}
    if (type(final["step"]) is not int or final["step"] != step or final["protocol"] != launch["protocol"]
            or checkpoint.digest(final["cursor"]) != checkpoint.digest(cursor)
            or checkpoint.digest(expected.get("cursor")) != checkpoint.digest(cursor)
            or final["state_sha256"] != {name: sha for group in expected["trainable_sha256"].values() for name, sha in group.items()}
            or final["inventory"] != snapshots["inventory.json"][0]
            or final["base_sha256"] != expected["frozen_base_sha256"]
            or final["base_sha256"] != snapshots["initialization.json"][0]["base_sha256"]
            or final["inventory"].get("quantization") != expected["quantization"]
            or final["inventory"].get("quantization") != snapshots["initialization.json"][0]["quantization"]
            or any(checkpoint.file_hash(capture / name) != sha for name, sha in paths.items())):
        raise ValueError("local train: actual final repair checkpoint payload or state differs")
    proof = {"version": "repair-training-replay-v1", "protocol_sha256": protocol["sha256"],
        "capture_binding_sha256": checkpoint.digest(launch["binding"]), "optimizer_updates": step,
        "S_gradient_verified": result["S_gradient_verified"], "control": index["control"],
        "semantic_replay_verified": True, "fresh_cpu_inputs_verified": True, "vjp_remeasured": False,
        "updates": summaries, "paths": paths}
    return proof | {"sha256": checkpoint.digest(proof)}


def contract_actors(actors: list, contract: dict) -> list:
    """The bound `problem_range` chunk of a longer fixed selection; the whole selection otherwise."""
    window = contract.get("problem_range")
    if window is None:
        return actors
    if (type(window) is not list or len(window) != 2 or any(type(value) is not int for value in window)
            or not 0 <= window[0] < window[1] <= len(actors)):
        raise ValueError("local train: problem range must be [a, b] with 0 <= a < b <= selection size")
    return actors[window[0]:window[1]]


_RESUME_KEYS = {"from", "launch_sha256", "checkpoint", "batch_index", "step", "prefix_replay_sha256"}


def _resume_parent(parent: Path, protocol: dict, launch: dict, contract: dict) -> tuple[dict, str]:
    """A closed (dead-owner) earlier segment with the same protocol, code, preparation and contract."""
    old, sha = _repair_read(parent / "launch.json")
    old_contract = old.get("execution_contract")
    if (not parent.is_absolute() or type(old.get("pid")) is not int or old["pid"] <= 0
            or same_process(old, launch) or owner_alive(old)
            or type(old_contract) is not dict or old.get("protocol") != launch.get("protocol")
            or checkpoint.digest({k: v for k, v in old_contract.items() if k != "resume"})
            != checkpoint.digest({k: v for k, v in contract.items() if k != "resume"})
            or any(old.get("binding", {}).get(key) != launch.get("binding", {}).get(key)
                   for key in ("protocol_sha256", "prompts_sha256", "code_sha256"))
            or old.get("preparation_sha256") != launch.get("preparation_sha256")):
        raise ValueError("local train: resume parent must be a closed segment of the same code/protocol/contract")
    return old, sha


def _resume_prefix(record: dict, protocol: dict, launch: dict, contract: dict, parameter_inventory) -> dict:
    """Re-verify the closed parent segment up to its resume checkpoint; returns the verified chain state."""
    if (type(record) is not dict or set(record) != _RESUME_KEYS or type(record["from"]) is not str
            or type(record["batch_index"]) is not int or type(record["step"]) is not int
            or record["checkpoint"] != f"checkpoint-pair-{record['batch_index']:02d}"):
        raise ValueError("local train: explicit resume record required")
    parent = Path(record["from"])
    old, sha = _resume_parent(parent, protocol, launch, contract)
    proof = _scaled_training_replay(parent, None, old["execution_contract"], protocol,
                                    parameter_inventory=parameter_inventory, until_batch=record["batch_index"])
    if (sha != record["launch_sha256"] or proof["sha256"] != record["prefix_replay_sha256"]
            or proof["state"]["step"] != record["step"]):
        raise ValueError("local train: resume parent prefix changed since the segment started")
    return proof["state"]


def _scaled_training_replay(capture: Path, result: dict | None, contract: dict, protocol: dict, *,
                            parameter_inventory, until_batch: int | None = None) -> dict:
    """Replay of a scale-option capture or resume segment chain (CPU; VJPs are not remeasured).

    K-problem batches, intermediate checkpoints every k updates plus the final one, and the fixed
    audit rule (audited updates measured, others `measured: false`, three-state S). `until_batch`
    replays only a parent segment's prefix up to its resume checkpoint. Historical captures never
    reach this function; they keep `_repair_training_replay` unchanged.
    """
    from .repair_evidence import repair_update_evidence
    from .repair_replay import checkpoint_snapshot, replay_trees, validate_journal, validate_probe_evidence
    from .schema import ProblemTree, from_json

    execution_admission(contract, protocol)
    opts = scale_options.options(protocol)
    capture, prefix = Path(capture), until_batch is not None
    names = ["launch.json", "initialization.json", "inventory.json"]
    if not prefix:
        names += ["result.json", "repair_preparation_index.json", "repair_update_evidence.json",
                  "reload_expected.json", "updated_forward_probes.json"]
    snapshots = {name: _repair_read(capture / name) for name in names}
    paths = {name: sha for name, (_, sha) in snapshots.items()}
    launch, initial = snapshots["launch.json"][0], snapshots["initialization.json"][0]
    batches: list = contract["pair_ids"] if type(contract.get("pair_ids")) is list else []
    size = opts["problems_per_update"]
    flat = [pid for batch in batches for pid in batch] if all(type(batch) is list for batch in batches) else None
    control, audit_rule = contract.get("arm") == "answer-only", opts["evidence_audit_rule"] is not None
    if (flat is None or len(batches) != opts["max_updates"] or any(len(batch) != size for batch in batches)
            or any(type(pid) is not str or not pid for pid in flat) or len(set(flat)) != opts["train_problems"]
            or contract.get("arm") not in ("notebook", "answer-only")
            or contract.get("notebook_reward_enabled") is not (not control)
            or launch.get("execution_contract") != contract
            or launch.get("execution_contract_sha256") != checkpoint.digest(contract)
            or launch.get("protocol") != json.loads(json.dumps(protocol)) or launch.get("problem_ids") != flat
            or launch.get("sampler") != scale_options.sampler_label(protocol)
            or type(launch.get("pid")) is not int or launch["pid"] <= 0 or owner_alive(launch)
            or until_batch is not None and (type(until_batch) is not int or not 0 <= until_batch < len(batches) - 1)):
        raise ValueError("local train: complete closed scaled repair execution binding required")
    if result is not None and (checkpoint.digest(result) != checkpoint.digest(snapshots["result.json"][0])
            or result.get("arm") != contract["arm"] or result.get("attempted_ids") != flat
            or type(result.get("optimizer_updates")) is not int
            or not 1 <= result["optimizer_updates"] <= opts["max_updates"]) or result is None and not prefix:
        raise ValueError("local train: closed scaled repair result differs")
    prepared, _, actors = _launch_preparation(launch, protocol)
    actors = contract_actors(actors, contract)
    if ([actor.problem_id for actor in actors] != flat
            or launch.get("binding") != training_binding(prepared, protocol, contract)
            or launch.get("preparation_sha256") != checkpoint.digest(prepared)
            or snapshots["inventory.json"][0].get("trainable") != parameter_inventory):
        raise ValueError("local train: scaled repair actor order, binding or inventory differs")
    if "resume" in contract:
        state = _resume_prefix(contract["resume"], protocol, launch, contract, parameter_inventory)
        if state["initialization"] != initial:
            raise ValueError("local train: resumed segment must carry the chain's original F0 initialization")
    else:
        state = {"step": 0, "batch": 0, "previous": initial.get("trainable_sha256"), "attempted": [],
                 "discarded": [], "unresolved": [], "signal_free": [], "incomplete": [], "answer_ids": [],
                 "S": None if audit_rule else False, "S_text": None if audit_rule else False, "routes": [],
                 "initialization": initial, "chain": []}
    start, stop = state["batch"], (len(batches) if until_batch is None else until_batch + 1)
    if start >= stop:
        raise ValueError("local train: a segment must replay at least one batch after its resume point")
    records, tree_links, raw_links, trees, saved = {}, [], [], [], {}
    for index in range(start, stop):
        name = f"repair-pair-{index:02d}.json"
        records[index], paths[name] = _repair_read(capture / name)
        pair = records[index]
        if (pair.get("version") != "repair-local-training-pair-v1" or pair.get("pair_index") != index
                or type(pair.get("pair_index")) is not int or pair.get("protocol_sha256") != protocol["sha256"]
                or pair.get("control") is not control or pair.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or type(pair.get("trees")) is not list or len(pair["trees"]) != size
                or type(pair.get("raw_records")) is not list):
            raise ValueError("local train: scaled repair batch record differs")
        tree_links.extend(pair["trees"])
        raw_links.extend(pair["raw_records"])
    for ordinal, link in enumerate(tree_links):
        if (type(link) is not dict or set(link) != {"problem_id", "tree_file", "tree_file_sha256", "preparation_file",
                "preparation_file_sha256", "preparation_receipt_sha256"}
                or type(link["tree_file"]) is not str or re.fullmatch(r"raw-[0-9]{4,}\.json", link["tree_file"]) is None
                or link["preparation_file"] != f"repair-preparation-{ordinal:04d}.json"
                or link["problem_id"] != flat[start * size + ordinal]):
            raise ValueError("local train: scaled repair TREE/preparation link differs")
        (row, tree_sha), (receipt, prep_sha) = _repair_read(capture / link["tree_file"]), _repair_read(
            capture / link["preparation_file"])
        if (tree_sha != link["tree_file_sha256"] or prep_sha != link["preparation_file_sha256"]
                or row.get("kind") != "TREE" or receipt.get("sha256") != link["preparation_receipt_sha256"]):
            raise ValueError("local train: scaled repair original file changed")
        paths[link["tree_file"]], paths[link["preparation_file"]] = tree_sha, prep_sha
        trees.append(from_json(ProblemTree, json.dumps(row["raw"], allow_nan=False)))
        saved[link["problem_id"]] = receipt
    if result is not None:
        index_file = snapshots["repair_preparation_index.json"][0]
        envelope = {"version": "repair-local-training-update-v1", "protocol_sha256": protocol["sha256"],
            "capture_binding_sha256": checkpoint.digest(launch["binding"]),
            "preparation_index_sha256": paths["repair_preparation_index.json"],
            "control": control, "commits": result.get("repair_commits")}
        if (index_file.get("version") != "repair-local-training-preparation-v1"
                or index_file.get("protocol_sha256") != protocol["sha256"] or index_file.get("control") is not control
                or index_file.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or index_file.get("trees") != tree_links or index_file.get("raw_records") != raw_links
                or result.get("repair_preparation_index_sha256") != paths["repair_preparation_index.json"]
                or snapshots["repair_update_evidence.json"][0] != envelope
                or result.get("repair_update_evidence_sha256") != paths["repair_update_evidence.json"]):
            raise ValueError("local train: scaled repair index or aggregate evidence envelope differs")
    paths.update(validate_journal(capture, raw_links, trees, complete=not prefix))
    fresh = replay_trees(trees, actors, protocol, prepared, Path(launch["model_path"]), control=control)
    if any(fresh[tree.problem_id] != saved[tree.problem_id] for tree in trees):
        raise ValueError("local train: fresh scaled repair preparation differs")
    step, previous = state["step"], state["previous"]
    lists = {key: list(state[key]) for key in ("attempted", "discarded", "unresolved", "signal_free", "incomplete", "answer_ids")}
    s_flag, s_text, routes = state["S"], state["S_text"], list(state["routes"])
    answer_route = "repair_route_policy" in protocol
    every = opts["checkpoint_every_updates"]
    commits, summaries, checkpointed = [], [], None
    for index in range(start, stop):
        ids = batches[index]
        batch_trees = trees[(index - start) * size:(index - start + 1) * size]
        update_file = f"pair-{index:02d}.json"
        update, paths[update_file] = _repair_read(capture / update_file)
        if (records[index].get("step") != step or type(records[index].get("step")) is not int
                or update.get("attempted_ids") != ids or [tree.problem_id for tree in batch_trees] != ids
                or any(tree.policy_version != step for tree in batch_trees)):
            raise ValueError("local train: scaled repair batch policy or original order differs")
        eligible = [tree for tree in batch_trees if fresh[tree.problem_id]["admitted"] and fresh[tree.problem_id]["has_signal"]]
        bad = [tree.problem_id for tree in batch_trees if tree not in eligible]
        unknown = [pid for pid in ids if fresh[pid]["admission"] == "UNRESOLVED"]
        zero = [pid for pid in ids if fresh[pid]["admitted"] and not fresh[pid]["has_signal"]]
        answered = [pid for pid in ids if fresh[pid]["admission"] == "ANSWER_VERIFIED"]
        if (update.get("discarded_ids") != bad or update.get("unresolved_ids") != unknown
                or update.get("signal_free_ids") != zero
                or (update.get("answer_verified_ids") != answered if answer_route else "answer_verified_ids" in update)):
            raise ValueError("local train: scaled repair fresh batch admission differs")
        lists["attempted"].extend(ids)
        lists["discarded"].extend(bad)
        lists["unresolved"].extend(unknown)
        lists["signal_free"].extend(zero)
        lists["answer_ids"].extend(answered)
        lists["incomplete"].extend(pid for pid in ids if any(reason != "UNRESOLVED_ROUTING"
                                                               for reason in fresh[pid]["admission_reasons"]))
        if update.get("status") != "UPDATED_UNSYNCHRONIZED":
            if (update.get("status") not in ("UNRESOLVED_ROUTING", "NO_LEARNING_SIGNAL", "NO_ADMITTED_REPAIR_TREES",
                                             "NO_ACTIONS", "NO_MATCHED_ACTIONS")
                    or update.get("step", update.get("step_before")) != step
                    or update.get("step_after", step) != step):
                raise ValueError("local train: uncommitted scaled batch changed the update sequence")
            continue
        audit = scale_options.audited(protocol, step, final_batch=index == len(batches) - 1)
        aggregate = update.get("repair_aggregate_gradient")
        if (not eligible or update.get("step_before") != step or update.get("step_after") != step + 1
                or type(aggregate) is not dict or (aggregate.get("measured") is not False) is not audit
                or update.get("group_weight_scale") != _group_weight_scale(update, protocol)):
            raise ValueError("local train: scaled committed sequence, fixed audit rule or group weights differ")
        summary = repair_update_evidence(update, protocol, trees=eligible,
            preparations={tree.problem_id: fresh[tree.problem_id] for tree in eligible},
            parameter_inventory=parameter_inventory, require_s=False, control=control)
        evidence_file = f"repair-evidence-{index:02d}.json"
        recorded, paths[evidence_file] = _repair_read(capture / evidence_file)
        if recorded != summary:
            raise ValueError("local train: recorded scaled repair evidence differs from replay")
        commits.append({"pair_index": index, "file": evidence_file, "sha256": paths[evidence_file],
                        "S_gradient_verified": summary["S_gradient_verified"]})
        summaries.append(summary)
        s_flag = scale_options.fold_s(s_flag, summary["S_gradient_verified"]) if audit_rule else s_flag or summary["S_gradient_verified"]
        if answer_route:
            s_text = (scale_options.fold_s(s_text, summary["S_text_gradient_verified"]) if audit_rule
                      else s_text or summary["S_text_gradient_verified"])
        changes, paths[f"changes-{index:02d}.json"] = _repair_read(capture / f"changes-{index:02d}.json")
        verified = verify_changes(changes["before"], changes["after"], update)
        if changes["before"] != previous or any(changes.get(key) != value for key, value in verified.items()):
            raise ValueError("local train: scaled repair parameter chain differs")
        routes.append([verified["vision_on_observed"], verified["vision_off_only_observed"]])
        previous = changes["after"]
        sync_names = [f"sync-{index:02d}.json"] + ([f"sampler-sync-{index:02d}.json"]
                                                 if opts["rollout_engine"] == "vllm" else [])
        for name in sync_names:
            sync, paths[name] = _repair_read(capture / name)
            _sync_rows(sync, step + 1, probe_source(prepared, actors), image_counts=probe_counts(prepared),
                       threshold=scale_options.sampler_sync_threshold(protocol) if name.startswith("sampler-") else .02)
        name = f"checkpoint-pair-{index:02d}"
        if (step + 1) % every == 0:
            committed, checkpoint_paths = checkpoint_snapshot(capture, name, binding=launch["binding"])
            paths.update(checkpoint_paths)
            cursor = committed["cursor"]
            if (committed["step"] != step + 1 or committed["protocol"] != launch["protocol"]
                    or cursor.get("position") != (index + 1) * size or cursor.get("sync_pending") is not False
                    or cursor.get("problem_ids") != flat or cursor.get("consumed_problem_ids") != lists["attempted"]
                    or cursor.get("discarded_problem_ids") != lists["discarded"]
                    or committed["state_sha256"] != {n: sha for group in previous.values() for n, sha in group.items()}
                    or committed["inventory"] != snapshots["inventory.json"][0]
                    or committed["base_sha256"] != initial["base_sha256"]
                    or committed["inventory"].get("quantization") != initial["quantization"]):
                raise ValueError("local train: scaled repair checkpoint cursor or state differs")
            checkpointed = index
        elif (capture / name).exists():
            raise ValueError("local train: unexpected checkpoint off the bound cadence")
        step += 1
    chain = state["chain"] + [{"capture": str(capture.absolute()), "until_batch": until_batch, "paths": paths}]
    if until_batch is not None:
        if checkpointed != until_batch:
            raise ValueError("local train: resume point must be a committed cadence checkpoint")
        state = {"step": step, "batch": until_batch + 1, "previous": previous, **lists, "S": s_flag, "S_text": s_text,
                 "routes": routes, "initialization": initial, "chain": chain}
        proof = {"version": "repair-scaled-prefix-replay-v1", "protocol_sha256": protocol["sha256"],
                 "capture_binding_sha256": checkpoint.digest(launch["binding"]), "state": state,
                 "updates": summaries, "commits": commits}
        return proof | {"sha256": checkpoint.digest(proof)}
    assert result is not None
    expected = snapshots["reload_expected.json"][0]
    probed = probe_set(prepared, actors)
    validate_probe_evidence(expected.get("forward_probes"), [a.problem_id for a in probed],
                            [len(a.image_paths) for a in probed], batch_rows=opts["rollout_batch_rows"])
    if (step != result["optimizer_updates"] or commits != result.get("repair_commits")
            or result.get("S_gradient_verified") is not s_flag
            or any(result.get(key) is not False for key in ("repair_functional_verified", "repair_learning_verified",
                                                           "benefit_verified"))
            or lists["attempted"] != flat or lists["discarded"] != result.get("discarded_ids")
            or lists["unresolved"] != result.get("unresolved_ids") or lists["signal_free"] != result.get("signal_free_ids")
            or lists["incomplete"] != result.get("incomplete_ids")
            or (result.get("answer_verified_ids") != lists["answer_ids"] or result.get("S_text_gradient_verified") is not s_text
                if answer_route else "answer_verified_ids" in result)
            or result.get("both_routes_verified") is not (any(on for on, _ in routes) and any(off for _, off in routes))):
        raise ValueError("local train: scaled repair counted commits or full history differs")
    if (expected.get("step") != step or expected.get("trainable_sha256") != previous
            or snapshots["updated_forward_probes.json"][0] != expected.get("forward_probes")
            or expected.get("repair_protocol_sha256") != protocol["sha256"]
            or expected.get("repair_preparation_index_sha256") != paths["repair_preparation_index.json"]
            or expected.get("repair_update_evidence_sha256") != paths["repair_update_evidence.json"]):
        raise ValueError("local train: scaled repair final snapshot differs")
    final, checkpoint_paths = checkpoint_snapshot(capture, "checkpoint", binding=launch["binding"])
    paths.update(checkpoint_paths)
    cursor = {"epoch": 0, "position": len(flat), "discarded_problem_ids": lists["discarded"],
              "problem_ids": flat, "consumed_problem_ids": flat, "sync_pending": False}
    if (final["step"] != step or final["protocol"] != launch["protocol"]
            or checkpoint.digest(final["cursor"]) != checkpoint.digest(cursor)
            or checkpoint.digest(expected.get("cursor")) != checkpoint.digest(cursor)
            or final["state_sha256"] != {n: sha for group in previous.values() for n, sha in group.items()}
            or final["inventory"] != snapshots["inventory.json"][0] or final["base_sha256"] != expected["frozen_base_sha256"]
            or final["base_sha256"] != initial["base_sha256"]
            or final["inventory"].get("quantization") != expected.get("quantization")
            or final["inventory"].get("quantization") != initial["quantization"]
            or any(checkpoint.file_hash(capture / name) != sha for name, sha in paths.items())):
        raise ValueError("local train: scaled repair final checkpoint payload or state differs")
    proof = {"version": "repair-scaled-training-replay-v1", "protocol_sha256": protocol["sha256"],
        "capture_binding_sha256": checkpoint.digest(launch["binding"]), "optimizer_updates": step,
        "S_gradient_verified": s_flag, "control": control, "semantic_replay_verified": True,
        "fresh_cpu_inputs_verified": True, "vjp_remeasured": False, "updates": summaries, "paths": paths,
        "chain": chain[:-1]}
    return proof | {"sha256": checkpoint.digest(proof)}


def _group_weight_scale(update: dict, protocol: dict) -> dict | None:
    """Per-group weight lambda/sum(lambda) of one update; it falls about 1/K with K problems per update."""
    counts = update.get("n_groups_by_kind")
    if type(counts) is not dict:
        return None
    weights = protocol["loss"]["group_weights"]
    denominator = sum(weights[kind] * count for kind, count in counts.items())
    return {"denominator": denominator,
            "per_group_weight": {kind: weights[kind] / denominator for kind in sorted(counts)} if denominator else None}


def validate_native_updates(capture: Path, result: dict, contract: dict, protocol: dict, *, parameter_inventory=None) -> None:
    """Every counted commit retains its own original pair's token/VJP evidence."""
    if (contract.get("admission_policy") in REPAIR_ADMISSIONS or "repair_objective" in protocol
            or protocol.get("rollout", {}).get("retry_contract") is not None):
        execution_admission(contract, protocol)
        _repair_training_replay(capture, result, contract, protocol, parameter_inventory=parameter_inventory)
        return
    from .episode_notebook import validate_update_binding
    from .schema import ProblemTree, from_json
    from .updates import ASSESSMENT_REQUIREMENT, effective_mis_deltas, notebook_objective_scores, notebook_signal_groups

    assessment = protocol.get("notebook_update_requirement") == ASSESSMENT_REQUIREMENT
    if assessment:
        if (execution_admission(contract, protocol) != ASSESSMENT_CREDIT_ADMISSION
                or contract.get("arm") not in ("notebook", "answer-only")
                or contract.get("notebook_reward_enabled") is not (contract["arm"] == "notebook")):
            raise ValueError("local train: explicit matched assessment arm required")
    step = 0
    paths = sorted(capture.glob("pair-*.json"))
    trees = [from_json(ProblemTree, json.dumps(record["raw"], allow_nan=False))
             for path in raw_journal_paths(capture) if (record := _read(path)).get("kind") == "TREE"]
    if (type(result.get("optimizer_updates")) is not int or not 1 <= result["optimizer_updates"] <= 5
            or len(contract["pair_ids"]) != 5 or any(len(pair) != 2 for pair in contract["pair_ids"])
            or len({pid for pair in contract["pair_ids"] for pid in pair}) != 10
            or [path.name for path in paths] != [f"pair-{i:02d}.json" for i in range(5)]
            or [tree.problem_id for tree in trees] != [pid for pair in contract["pair_ids"] for pid in pair]):
        raise ValueError("local train: complete original five pair records required")
    with ExitStack() as stack:
        actors, tokenizer, verifier = {}, None, None
        if assessment:
            from transformers import AutoTokenizer
            from .launch import read_descriptor
            from .local_run import read_actor_descriptor, read_execution_actors
            from .verifier_service import VerifierService

            launch = _read(capture / "launch.json")
            prepared = _read(Path(launch["preparation"]))
            request = read_descriptor(prepared["data_request"])
            actors = {a.problem_id: a for a in read_execution_actors(
                prepared, read_actor_descriptor(prepared["actors"]), protocol)}
            tokenizer = AutoTokenizer.from_pretrained(launch["model_path"], local_files_only=True, trust_remote_code=False)
            root = Path(prepared["build_root"])
            verifier = stack.enter_context(VerifierService(root / "train", forbidden_roots=[
                Path(request["index"]["path"]).parent, Path.home() / "mmdl-data/evaluation"], timeout=60))
            if verifier.metadata["vault_manifest_sha256"] != _read(root / "manifest.yaml")["split_manifests"]["train"]:
                raise ValueError("local train: assessment pair verifier identity differs")
        for index, (path, ids) in enumerate(zip(paths, contract["pair_ids"])):
            update = _read(path)
            pair_trees = trees[index * 2:index * 2 + 2]
            if update.get("attempted_ids") != ids or any(tree.policy_version != step for tree in pair_trees):
                raise ValueError("local train: native pair record order differs")
            if update.get("status") != "UPDATED_UNSYNCHRONIZED":
                if assessment and (update.get("step", update.get("step_before")) != step
                        or update.get("step_after", step) != step
                        or update.get("assessment_learning_verified", False) is not False):
                    raise ValueError("local train: uncommitted assessment pair changed the update sequence")
                continue
            if (type(update.get("step_before")) is not int or type(update.get("step_after")) is not int
                    or update["step_before"] != step or update["step_after"] != step + 1):
                raise ValueError("local train: native commit sequence differs")
            routes = update.get("routes", {})
            if not routes or not set(routes) <= set(ids):
                raise ValueError("local train: native commit routes differ from original pair")
            samples, group_samples, scores = {}, {}, {}
            for tree in pair_trees:
                if tree.problem_id not in routes:
                    continue
                if (routes[tree.problem_id] is not (tree.route == "VISION_ON")
                        or update.get("notebook_objective_receipts", {}).get(tree.problem_id)
                        != tree.notebook.get("learning_objective")):
                    raise ValueError("local train: native commit policy, route or objective differs from TREE")
                validate_update_binding(tree, generation_policy=protocol.get("notebook_generation_policy"),
                    feedback_policy=protocol.get("notebook_feedback_policy"),
                    observation_policy=protocol.get("notebook_observation_policy"),
                    draft_policy=protocol.get("notebook_draft_policy"))
                scores.update(notebook_objective_scores(tree, protocol["notebook_objective"]))
                for group in tree.groups:
                    if not group.usable:
                        continue
                    group_samples[group.group_id] = group.samples
                    for sample in group.samples:
                        if (sample.sample_id in samples or sample.problem_id != tree.problem_id
                                or sample.group_id != group.group_id or group.problem_id != tree.problem_id
                                or sample.policy_version != step or sample.kind != group.kind):
                            raise ValueError("local train: native original sample identity differs")
                        samples[sample.sample_id] = sample
            credits = update.get("synthesis_token_credit", {})
            backward = update.get("notebook_backward", [])
            referenced = set(effective_mis_deltas(update, protocol)) | set(credits)
            referenced.update(update.get("retained_native_relationship_signal", {}).get("signal_sample_ids", []))
            referenced.update(update.get("retained_learning_signal", {}).get("signal_sample_ids", []))
            referenced.update(update.get("retained_assessment_signal", {}).get("signal_sample_ids", []))
            referenced.update(row["sample_id"] for row in backward)
            if not referenced <= set(samples) or len({row["sample_id"] for row in backward}) != len(backward):
                raise ValueError("local train: native proof sample belongs to another original TREE/pair")
            for sample_id, credit in credits.items():
                sample = samples[sample_id]
                members = group_samples[sample.group_id]
                if (sample.kind != "DIAG" or len(members) != 4
                        or any(member.sample_id not in scores for member in members)):
                    raise ValueError("local train: native credit requires original complete scored DIAG group")
                full_group = [{"sample_id": member.sample_id,
                    "token_ids_sha256": checkpoint.digest(member.completion_token_ids),
                    "text_sha256": checkpoint.digest(member.text), "score_sha256": checkpoint.digest(scores[member.sample_id])}
                    for member in members]
                if (credit.get("token_ids_sha256") != checkpoint.digest(sample.completion_token_ids)
                        or credit.get("text_sha256") != checkpoint.digest(sample.text)
                        or credit.get("score_sha256") != checkpoint.digest(scores[sample_id])
                        or credit.get("full_group") != full_group or credit.get("full_group_sha256") != checkpoint.digest(full_group)
                        or len(credit.get("partial_token_advantages", [])) != len(sample.completion_token_ids)):
                    raise ValueError("local train: native credit differs from original group tokens/scores")
            for row in backward:
                sample = samples[row["sample_id"]]
                if (sample.kind != "DIAG" or row.get("group_id") != sample.group_id
                        or row.get("action_tokens") != len(sample.completion_token_ids)
                        or row.get("vision_on") is not routes[sample.problem_id]):
                    raise ValueError("local train: native backward differs from original group/token/route")
            if assessment:
                fact_credit_learning_gate(capture, {"notebook_signal_groups": sum(
                    notebook_signal_groups(tree, notebook_objective_scores(tree, protocol["notebook_objective"]))
                    for tree in pair_trees if tree.problem_id in routes)}, update, protocol,
                    actors=actors, verifier=verifier, tokenizer=tokenizer,
                    admission=ASSESSMENT_CREDIT_ADMISSION, tree_ids=set(ids), control=contract["arm"] == "answer-only",
                    parameter_inventory=parameter_inventory)
            else:
                native_relationship_update_evidence(update, protocol)
            step += 1
        if step != result["optimizer_updates"]:
            raise ValueError("local train: native commit count differs")
    if assessment and (any(result.get(key) is not (contract["arm"] == "notebook") for key in
            ("assessment_learning_verified", "assessment_only_gradient_verified"))
            or any(result.get(key) is not False for key in ("strict_notebook_correctness_verified",
                "accurate_whole_notebook_verified", "visual_improvement_verified", "self_improvement_verified"))):
        raise ValueError("local train: assessment learning and whole-notebook success must remain separate")


def matched_notebook_arm(path: Path, prepared: dict, protocol: dict, model_path: Path,
                         cache_check: Path | None, pairs: list[list], *,
                         historical_code_sha256: str | None = None,
                         admission: str = STRICT_ADMISSION, parent: dict | None = None,
                         max_seconds: int | None = None, matched_reload: Path | None = None) -> dict:
    """Training requires current code; read-only evaluation may pin its verified training source."""
    if admission in REPAIR_ADMISSIONS and matched_reload is None:
        raise ValueError("local train: repair control requires its explicit matched fresh reload")
    if path.name != "initialization.json":
        raise ValueError("local train: use the notebook run's actual initialization record")
    if (historical_code_sha256 is not None and (type(historical_code_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", historical_code_sha256) is None)):
        raise ValueError("local train: explicit historical training code SHA required")
    versions: dict[Path, str] = {}
    if admission in REPAIR_ADMISSIONS:
        launch, versions[path.parent / "launch.json"] = _repair_read(path.parent / "launch.json")
        result, versions[path.parent / "result.json"] = _repair_read(path.parent / "result.json")
    else:
        launch, result = _read(path.parent / "launch.json"), _read(path.parent / "result.json")
    if admission in REPAIR_ADMISSIONS:
        if (historical_code_sha256 is not None or type(launch.get("pid")) is not int or launch["pid"] <= 0
                or owner_alive(launch) or type(result.get("optimizer_updates")) is not int
                or not 1 <= result["optimizer_updates"] <= scale_options.options(protocol)["max_updates"]):
            raise ValueError("local train: closed current-source matched repair arm required")
    contract = launch.get("execution_contract", {})
    if max_seconds is not None and execution_max_seconds(contract, protocol) != execution_max_seconds(
            {"training_budget": {"version": TRAINING_BUDGET_VERSION, "max_seconds": max_seconds}}, protocol):
        raise ValueError("local train: control must match notebook training time budget")
    if (execution_admission(contract, protocol) != admission
            or parent is not None and contract.get("parent_gate") != parent):
        raise ValueError("local train: control must match notebook admission and parent proof")
    expected_binding = training_binding(prepared, protocol, contract)
    if historical_code_sha256 is not None:
        expected_binding["code_sha256"] = historical_code_sha256
    if (result.get("status") != "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"
            or result.get("arm") != "notebook" or contract.get("arm") != "notebook"
            or result.get("initialization_sha256") != checkpoint.file_hash(path)
            or contract.get("notebook_reward_enabled") is not True
            or launch.get("protocol") != json.loads(json.dumps(protocol))
            or launch.get("preparation_sha256") != checkpoint.digest(prepared)
            or launch.get("model_path") != str(model_path)
            or launch.get("cache_check") != (None if cache_check is None else str(cache_check))
            or contract.get("pair_ids") != [[a.problem_id for a in pair] for pair in pairs]
            or launch.get("execution_contract_sha256") != checkpoint.digest(contract)
            or launch.get("binding") != expected_binding):
        raise ValueError("local train: control must match the closed notebook arm's data/order/protocol")
    metadata = checkpoint.read_metadata(path.parent / "checkpoint", binding=launch["binding"])
    if metadata["step"] != result["optimizer_updates"] or metadata["protocol"] != launch["protocol"]:
        raise ValueError("local train: notebook control source has inconsistent update evidence")
    repair_links = {}
    if admission in REPAIR_ADMISSIONS:
        assert matched_reload is not None
        reload = Path(matched_reload).absolute()
        restored, versions[reload / "result.json"] = _repair_read(reload / "result.json")
        expected, versions[path.parent / "reload_expected.json"] = _repair_read(path.parent / "reload_expected.json")
        forward, versions[reload / "forward_probes.json"] = _repair_read(reload / "forward_probes.json")
        sync, versions[reload / "sync.json"] = _repair_read(reload / "sync.json")
        proof = _repair_training_replay(path.parent, result, contract, protocol,
                                        parameter_inventory=metadata["inventory"]["trainable"])
        _repair_probe_binding(expected.get("forward_probes"), sync, result["optimizer_updates"],
                              probe_source(prepared, [actor for pair in pairs for actor in pair]),
                              batch_rows=scale_options.options(protocol)["rollout_batch_rows"],
                              image_counts=probe_counts(prepared))
        if (any(proof["paths"].get(name) != versions[path.parent / name]
                for name in ("launch.json", "result.json", "reload_expected.json"))
                or any(checkpoint.file_hash(name) != sha for name, sha in versions.items())):
            raise ValueError("local train: matched cached source/reload versions differ from fresh replay")
        if (type(launch.get("pid")) is not int or launch["pid"] <= 0 or owner_alive(launch)
                or type(result.get("optimizer_updates")) is not int
                or not 1 <= result["optimizer_updates"] <= scale_options.options(protocol)["max_updates"]
                or type(restored.get("pid")) is not int or restored["pid"] <= 0 or owner_alive(restored)
                or same_process(restored, launch) or type(restored.get("capture_pid")) is not int
                or restored["capture_pid"] != launch["pid"] or type(restored.get("step")) is not int
                or restored["step"] != result["optimizer_updates"]
                or restored.get("status") != "LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED"
                or any(restored.get(key) is not True for key in ("fresh_reload_verified", "all_trainable_hashes_match",
                    "optimizer_restored", "actual_forward_and_generation_match", "frozen_quantized_base_unchanged", "repair_functional_verified"))
                or restored.get("repair_replay_sha256") != proof["sha256"]
                or restored.get("repair_update_evidence_sha256") != checkpoint.file_hash(path.parent / "repair_update_evidence.json")
                or restored.get("repair_protocol_sha256") != protocol["sha256"]
                or restored.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or restored.get("S_gradient_verified") is not proof["S_gradient_verified"]
                or restored.get("repair_learning_verified") is not proof["S_gradient_verified"]
                or restored.get("benefit_verified") is not False
                or forward != expected["forward_probes"]):
            raise ValueError("local train: matched repair arm requires its original closed fresh reload and replay")
        runtime_files = {str(path.parent / name): sha for name, sha in proof["paths"].items()}
        runtime_files.update({str(name): sha for name, sha in versions.items()})
        if any(checkpoint.file_hash(Path(name)) != sha for name, sha in runtime_files.items()):
            raise ValueError("local train: matched repair runtime files changed during validation")
        repair_links = {"repair_replay_sha256": proof["sha256"], "repair_runtime_files": runtime_files,
            "fresh_reload": {"path": str(reload),
            "files": {name: checkpoint.file_hash(reload / name) for name in ("result.json", "forward_probes.json", "sync.json")}}}
    elif admission == ASSESSMENT_CREDIT_ADMISSION:
        validate_native_updates(path.parent, result, contract, protocol,
            parameter_inventory=metadata["inventory"]["trainable"])
    return {"initialization": {"path": str(path), "sha256": checkpoint.file_hash(path)},
        "launch_sha256": checkpoint.file_hash(path.parent / "launch.json"),
        "result_sha256": checkpoint.file_hash(path.parent / "result.json"), **repair_links}


def train(preparation: Path, model_path: Path, cache_check: Path | None, minimal: Path | None,
          minimal_reload: Path | None, output: Path, *, arm: str,
          match_initialization: Path | None = None, admission: str = STRICT_ADMISSION,
          max_seconds: int = 14400, matched_reload: Path | None = None,
          note_contract: str = DEFAULT_NOTE_CONTRACT, answer_verified_route: bool = False,
          finalization_reward: bool = False, slotless_sources: tuple[str, ...] = (),
          resume_from: Path | None = None, problem_range: tuple[int, int] | None = None,
          frozen_approval: dict | None = None, min_free_gib: float | None = None) -> dict:
    """`resume_from`/`problem_range` (and every scale option) exist only under a bound scale group.

    `min_free_gib` (operational, not bound into the protocol) stops with DISK_LOW before any checkpoint write."""
    import numpy as np
    import torch

    from .notebook_learning import append_rewarded_notebooks
    from .policy import enable_deterministic_execution
    from .quantized_cache import block_original_access
    from .quantized_policy import load_quantized_policy
    from .rollout import RolloutConfig, build_tree
    from .updates import (bind_notebook_objective, notebook_objective_scores,
                          optimizer_step, partition_learning_trees)
    from .verifier_service import VerifierService

    if arm not in ("notebook", "answer-only") or (arm == "answer-only") != (match_initialization is not None):
        raise ValueError("local train: matched initialization required for the answer-only control")
    if (admission in REPAIR_ADMISSIONS and (arm == "answer-only") != (matched_reload is not None)
            or admission not in REPAIR_ADMISSIONS and matched_reload is not None):
        raise ValueError("local train: explicit matched reload is required only for repair answer-only control")
    collection = admission == FIXED_TEN_NATIVE_EXPLORATION
    if collection:
        from .local_profile import build_protocol

        if arm != "notebook" or minimal is not None or minimal_reload is not None:
            raise ValueError("local train: native collection requires no minimum parent; answer-only control is unsupported")
        parent = None
        protocol = build_protocol(verified_note_synthesis_credit=True, pair_action_notebook=True, context_limit=24576)
    else:
        if minimal is None or minimal_reload is None:
            raise ValueError("local train: completed minimum and reload paths required for this admission")
        parent = training_parent_gate(minimal, minimal_reload, admission=admission)
        protocol = restore_protocol(parent["launch"]["protocol"])
    execution_admission({"admission_policy": admission}, protocol)
    from .repair_notebook import note_contract as bound_note_contract

    if bound_note_contract(protocol) != note_contract:
        raise ValueError("local train: --note-contract must match the restored minimum protocol")
    from .local_profile import repair_options

    bound_options = repair_options(protocol)
    if (bound_options.get("answer_verified_route", False) is not answer_verified_route
            or bound_options.get("finalization_reward", False) is not finalization_reward
            or bound_options.get("slotless_sources", ()) != tuple(slotless_sources)):
        raise ValueError("local train: --answer-verified-route/--finalization-reward must match the restored minimum protocol")
    answer_route = "repair_route_policy" in protocol
    repair_selected = admission in REPAIR_ADMISSIONS
    scaled, opts = scale_options.scaled(protocol), scale_options.options(protocol)
    bf16 = opts["policy_precision"] == "bf16"
    if (resume_from is not None or problem_range is not None) and not (scaled and repair_selected):
        raise ValueError("local train: --resume-from/--problem-range require the repair protocol's bound scale group")
    if not bf16 and cache_check is None:
        raise ValueError("local train: the NF4 policy requires its completed cache check")
    if arm == "answer-only" and "notebook_update_requirement" in protocol and admission != ASSESSMENT_CREDIT_ADMISSION:
        raise ValueError("local train: native-only objective has no matched answer-only control contract")
    training_budget = {"version": TRAINING_BUDGET_VERSION, "max_seconds": max_seconds}
    max_seconds = execution_max_seconds({"training_budget": training_budget}, protocol)
    objective = protocol.get("notebook_objective")
    from .whole_reference_pages import VERSION as pages_version

    paged = protocol.get("notebook_generation_policy") == pages_version
    if (protocol["profile"] != PROFILE or protocol["local_budget"]["train_problems"] != opts["train_problems"]
            or protocol["local_budget"]["train_updates"] != opts["max_updates"]
            or parent is not None and (str(model_path) != parent["launch"]["model_path"]
                or (None if cache_check is None else str(cache_check)) != parent["launch"]["cache_check"])):
        raise ValueError("local train: fixed local ten/five budget and parent NF4 base required")
    if bf16:
        from .local_calibrate import verify_original

        # The verified original snapshot is the BF16 base itself; there is no NF4 cache readiness.
        readiness, original_files = {}, verify_original(Path(model_path))
    else:
        assert cache_check is not None
        readiness = cache_readiness(cache_check, model_path, protocol)
    verification = None
    if frozen_approval is None:
        prepared, request, actors = read_preparation(preparation, protocol)
    else:
        prepared, request, actors, verification = read_frozen_preparation(preparation, protocol, frozen_approval)
    window = None if problem_range is None else [*problem_range]
    actors = contract_actors(actors, {} if window is None else {"problem_range": window})
    pairs = scale_options.batches(actors, protocol) if scaled else fixed_pairs(actors)
    prompts = local_prompts(protocol)
    collection_gate = native_collection_gate(prepared, protocol, readiness, actors) if collection else None
    root = Path(prepared["build_root"])
    vault_identity = _read(root / "manifest.yaml")["split_manifests"]["train"]
    initialization_sha = None
    if repair_selected and match_initialization is not None:
        expected_initialization, initialization_sha = _repair_read(match_initialization)
    else:
        expected_initialization = None if match_initialization is None else _read(match_initialization)
    matched_options: dict = {"matched_reload": matched_reload} if repair_selected else {}
    matched = (None if match_initialization is None else matched_notebook_arm(
        match_initialization, prepared, protocol, model_path, cache_check, pairs,
        admission=admission, parent=parent, max_seconds=max_seconds,
        **matched_options))
    if repair_selected:
        assert parent is not None
        if (matched is not None and matched["initialization"]["sha256"] != initialization_sha
                or any(checkpoint.file_hash(Path(row["path"])) != row["sha256"] for row in parent["files"])):
            raise ValueError("local train: cached matched initialization or minimum parent files changed")
    output = artifact_path(output)
    protected: tuple[Path | None, ...] = (model_path, root, minimal, minimal_reload)
    if collection:
        protected += (preparation.parent, cache_check)
    if resume_from is not None:
        protected += (Path(resume_from),)
    if any(output.resolve().is_relative_to(p.resolve()) for p in protected if p is not None):
        raise ValueError("local train: preserve immutable source and gate artifacts")
    if match_initialization is not None and output.resolve().is_relative_to(match_initialization.parent.resolve()):
        raise ValueError("local train: control output must preserve the notebook arm")
    runtime = enable_deterministic_execution()
    contract = {"stage": "FIXED_TEN_FROM_FRESH_F0", "arm": arm,
        "admission_policy": admission,
        "training_budget": training_budget,
        "notebook_reward_enabled": arm == "notebook", "max_optimizer_updates": opts["max_updates"],
        "pair_ids": [[a.problem_id for a in pair] for pair in pairs],
        "no_signal_policy": "record; consume the fixed pair; no replacement or forced update",
        "same_generated_notebook_checks_and_kl": True,
        "initial_checkpoint": scale_options.initial_checkpoint(protocol),
        "authority": protocol["local_authorization"], "parent_gate": parent,
        "deterministic_execution": runtime,
        "matched_notebook_arm": matched}
    if collection:
        contract["collection_gate"] = collection_gate
    if repair_selected:
        contract.update(repair_protocol_sha256=protocol["sha256"],
                        repair_objective_sha256=protocol["repair_objective_sha256"])
    if window is not None:
        contract["problem_range"] = window
    resumed = None
    if resume_from is not None:
        # Verify the closed earlier segment and its committed prefix before any output or model load.
        resumed = _resume_point(Path(resume_from).absolute(), protocol, prepared, contract)
        contract["resume"] = resumed["record"]
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    binding = training_binding(prepared, protocol, contract)
    launch: dict = {"profile": PROFILE, "protocol": protocol, "binding": binding, "pid": os.getpid(), "host": host_identity(),
        "preparation": str(preparation), "model_path": str(model_path), "standalone_base": not bf16,
        "cache_check": None if cache_check is None else str(cache_check), "problem_ids": [a.problem_id for a in actors],
        "optimizer_state_collection_policy": scale_options.parking_policy(protocol),
        "execution_contract": contract, "execution_contract_sha256": checkpoint.digest(contract)}
    launch["preparation_sha256"] = checkpoint.digest(prepared)
    if scaled:
        launch["sampler"] = scale_options.sampler_label(protocol)
    if verification is not None:
        launch["data_verification"] = verification
    write_json(output / "launch.json", launch)
    if bf16:
        write_json(output / "original_files.json", original_files)
    schema = {} if opts["rollout_engine"] == "vllm" else None
    forbidden = _install_actor_guards(prepared, request, vllm_schema=schema)
    original_access = None if bf16 else block_original_access(Path(readiness["original_model_root"]))
    seed = protocol["local_budget"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    deadline = began + max_seconds
    result: dict = {"status": "INCOMPLETE", "optimizer_updates": 0, "attempted_ids": [],
        "signal_free_ids": [], "unresolved_ids": [], "discarded_ids": [],
        "failed_update_ids": [], "arm": arm, "fresh_reload_verified": False,
        "self_improvement_verified": False, "both_routes_verified": False,
        "notebook_generated_groups": 0, "notebook_signal_groups": 0,
        "notebook_gradient_contribution_verified": False,
        "notebook_combined_objective_gradient_observed": False, **(schema or {})}
    if paged:
        result["notebook_physical_signal_groups"] = 0
    if collection:
        result.update(collection_allowed=True, native_relationship_learning_verified=False,
                      strict_notebook_correctness_verified=False, accurate_whole_notebook_verified=False)
    if admission == ASSESSMENT_CREDIT_ADMISSION:
        result.update(assessment_learning_verified=False, assessment_only_gradient_verified=False,
                      strict_notebook_correctness_verified=False, accurate_whole_notebook_verified=False,
                      visual_improvement_verified=False)
    audit_rule = opts["evidence_audit_rule"] is not None
    if repair_selected:
        # Under a fixed audit rule S is three-state: None until an audited update measures it.
        result.update(repair_functional_verified=False, S_gradient_verified=None if audit_rule else False,
                      repair_learning_verified=False, benefit_verified=False, incomplete_ids=[], repair_commits=[])
    if answer_route:
        # Answer-verified (slotless) trees are tracked apart and never count toward both_routes_verified.
        result.update(answer_verified_ids=[], S_text_gradient_verified=None if audit_rule else False)
    notebook_enabled = arm == "notebook"
    journal, changes_seen = 0, []
    repair_raw_links: list[dict] = []
    repair_tree_links: list[dict] = []
    start_batch, prefix_items = 0, 0
    if resumed is not None:
        state = resumed["state"]
        start_batch, prefix_items = state["batch"], len(state["attempted"])
        result.update(optimizer_updates=state["step"], attempted_ids=list(state["attempted"]),
            discarded_ids=list(state["discarded"]), unresolved_ids=list(state["unresolved"]),
            signal_free_ids=list(state["signal_free"]), incomplete_ids=list(state["incomplete"]),
            S_gradient_verified=state["S"], resumed_from=resumed["record"])
        if answer_route:
            result.update(answer_verified_ids=list(state["answer_ids"]), S_text_gradient_verified=state["S_text"])
        changes_seen = [{"vision_on_observed": on, "vision_off_only_observed": off} for on, off in state["routes"]]
    start_step = result["optimizer_updates"]
    meter: scale_options.TokenMeter | None = None
    measured: dict[str, Any] = {"startup_seconds": 0., "update_seconds": 0., "retry_groups": 0, "group_weight_denominators": []}

    def record(kind, value):
        nonlocal journal
        raw = json.loads(to_json(value)) if kind in ("TREE", "GROUP") else value
        filename = f"raw-{journal:04d}.json"
        write_json(output / filename, {"kind": kind, "raw": raw})
        if repair_selected:
            repair_raw_links.append({"file": filename, "kind": kind, "sha256": checkpoint.file_hash(output / filename)})
        journal += 1
        print(json.dumps({"event": kind, "index": journal, "seconds": time.monotonic()-began}), flush=True)

    def preserve_repair_index():
        if repair_selected and "repair_preparation_index_sha256" not in result:
            write_json(output / "repair_preparation_index.json", {"version": "repair-local-training-preparation-v1",
                "protocol_sha256": protocol["sha256"], "capture_binding_sha256": checkpoint.digest(binding),
                "control": not notebook_enabled, "trees": repair_tree_links, "raw_records": repair_raw_links})
            result["repair_preparation_index_sha256"] = checkpoint.file_hash(output / "repair_preparation_index.json")
            write_json(output / "repair_update_evidence.json", {"version": "repair-local-training-update-v1",
                "protocol_sha256": protocol["sha256"], "capture_binding_sha256": checkpoint.digest(binding),
                "preparation_index_sha256": result["repair_preparation_index_sha256"],
                "control": not notebook_enabled, "commits": result["repair_commits"]})
            result["repair_update_evidence_sha256"] = checkpoint.file_hash(output / "repair_update_evidence.json")

    try:
        loader = scale_options.build_bf16_policy if bf16 else load_quantized_policy
        if resumed is None:
            bundle = loader(model_path, "MOD-RV", protocol)
            optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
        else:
            bundle, optimizer, scheduler, step, cursor = checkpoint.load_resume(model_path, resumed["checkpoint"],
                protocol, binding=resumed["binding"], device="cuda:0", policy_factory=partial(loader, device_index=0))
            state = resumed["state"]
            if (parameter_hashes(bundle) != state["previous"] or step != state["step"]
                    or bundle.inventory != resumed["inventory"] or cursor != {"epoch": 0,
                        "position": len(state["attempted"]), "discarded_problem_ids": state["discarded"],
                        "problem_ids": launch["problem_ids"], "consumed_problem_ids": state["attempted"],
                        "sync_pending": False}):
                raise ValueError("local train: resumed weights/step/cursor differ from the verified parent prefix")
        write_json(output / "inventory.json", bundle.inventory)
        if bf16:
            initial = {"trainable_sha256": parameter_hashes(bundle), "base_sha256": bundle.base_sha256,
                "quantization": None, "seed": seed, "original_files_sha256": checkpoint.digest(original_files)}
        else:
            initial = {"trainable_sha256": parameter_hashes(bundle), "base_sha256": bundle.base_sha256,
                "quantization": bundle.inventory["quantization"], "seed": seed,
                "model_cache_metadata_sha256": checkpoint.file_hash(model_path / "nf4_base.json")}
        if resumed is not None:
            # The chain keeps its original fresh-F0 record; the resumed state is bound by the prefix replay.
            if initial["base_sha256"] != resumed["state"]["initialization"]["base_sha256"]:
                raise ValueError("local train: resumed frozen base differs from the chain's F0")
            initial = resumed["state"]["initialization"]
        if expected_initialization is not None and initial != expected_initialization:
            raise ValueError("local train: control does not start from identical fresh F0 initialization")
        write_json(output / "initialization.json", initial)
        result["initialization_sha256"] = checkpoint.file_hash(output / "initialization.json")
        guard = scale_options.base_guard(bundle, protocol)
        sampler, engine = scale_options.make_engines(bundle, protocol, model_path, policy_version=start_step)
        meter = scale_options.TokenMeter(sampler, engine)
        with ExitStack() as stack:
            probe_inputs = probe_source(prepared, actors)
            images = {a.problem_id: load_images(a, root / "images") for a in with_probe_actors(actors, probe_inputs)}
            for values in images.values():
                for image in values:
                    stack.callback(image.close)
            verifier = stack.enter_context(VerifierService(root / "train", forbidden_roots=forbidden, timeout=60))
            if verifier.metadata["vault_manifest_sha256"] != vault_identity:
                raise ValueError("local train: verifier identity differs")
            probes = make_probes(probe_inputs, images, engine, prompts, image_counts=probe_counts(prepared))
            initial_sync, sampler_sync = scale_options.synchronize(bundle, sampler, engine, probes, protocol,
                                                                   start_step, push=False, **sync_options(prepared))
            write_json(output / "initial_sync.json", initial_sync)
            if sampler_sync is not None:
                write_json(output / "initial_sampler_sync.json", sampler_sync)
            measured["startup_seconds"] = time.monotonic() - began
            cfg = RolloutConfig.from_protocol(protocol, "ALG-NB3", master_seed=seed)
            for index, pair in enumerate(pairs):
                if index < start_batch:
                    continue
                if time.monotonic() >= deadline:
                    result["status"] = "BUDGET_STOP"
                    return result
                trees = []
                repair_preparations: dict[str, dict] = {}
                pair_raw_start, pair_tree_start = len(repair_raw_links), len(repair_tree_links)
                for actor in pair:
                    with scale_options.park(bundle, optimizer, scheduler, protocol, result["optimizer_updates"]):
                        tree = build_tree(actor, images[actor.problem_id], sampler, verifier, prompts, cfg,
                            result["optimizer_updates"], protocol["route_rule"],
                            on_group=lambda g: record("GROUP", g), should_stop=lambda: time.monotonic() >= deadline,
                            **({"on_raw": lambda event: record("REPAIR_RAW", event)} if repair_selected else {}))
                        append_rewarded_notebooks([tree], {actor.problem_id: images[actor.problem_id]},
                            sampler, verifier, prompts, cfg, actors={actor.problem_id: actor},
                            on_group=lambda g: record("GROUP", g),
                            **({"on_raw": lambda event: record("REPAIR_RAW", event)} if repair_selected else {
                                "on_observation": lambda event: record("OBSERVATION", event),
                                **({"on_draft": lambda event: record("DRAFT", event)} if protocol.get("notebook_draft_policy") else {}),
                                **({"on_page": lambda event: record("PAGE", event)} if paged else {})}),
                            should_stop=lambda: time.monotonic() >= deadline)
                    if not repair_selected and tree.notebook.get("rewarded_notebook", {}).get("notebook_generation_policy") != protocol.get("notebook_generation_policy"):
                        raise ValueError("local train: notebook generation policy differs from protocol")
                    if not repair_selected and tree.notebook.get("rewarded_notebook", {}).get("notebook_draft_policy") != protocol.get("notebook_draft_policy"):
                        raise ValueError("local train: notebook draft policy differs from protocol")
                    if objective is not None:
                        bind_notebook_objective(tree, actor, verifier, objective)
                    if repair_selected:
                        from .repair_training import prepare_tree_credit

                        receipt = prepare_tree_credit(tree, actor, verifier, engine.tokenizer,
                            preparation=engine, images=images[actor.problem_id], master_seed=seed,
                            notebook_reward_enabled=notebook_enabled)
                        repair_preparations[actor.problem_id] = receipt
                        receipt_file = f"repair-preparation-{len(repair_tree_links):04d}.json"
                        write_json(output / receipt_file, receipt)
                        tree_file = f"raw-{journal:04d}.json"
                    record("TREE", tree)
                    if repair_selected:
                        repair_tree_links.append({"problem_id": actor.problem_id, "tree_file": tree_file,
                            "tree_file_sha256": checkpoint.file_hash(output / tree_file), "preparation_file": receipt_file,
                            "preparation_file_sha256": checkpoint.file_hash(output / receipt_file),
                            "preparation_receipt_sha256": receipt["sha256"]})
                    note_groups = [g for g in tree.groups if g.kind == "DIAG"]
                    result["notebook_generated_groups"] += len(note_groups)
                    measured["retry_groups"] += sum(g.kind in ("T2", "T3") for g in tree.groups)
                    from .updates import group_has_signal, notebook_signal_groups

                    if not repair_selected:
                        scores = notebook_objective_scores(tree, objective)
                        result["notebook_signal_groups"] += notebook_signal_groups(tree, scores)
                        if paged:
                            result["notebook_physical_signal_groups"] += sum(
                                g.usable and group_has_signal(g, scores) for g in note_groups)
                    trees.append(tree)
                    result["attempted_ids"].append(actor.problem_id)
                if repair_selected:
                    eligible = [tree for tree in trees if repair_preparations[tree.problem_id]["admitted"]
                                and repair_preparations[tree.problem_id]["has_signal"]]
                    unresolved = [tree.problem_id for tree in trees
                                  if repair_preparations[tree.problem_id]["admission"] == "UNRESOLVED"]
                    signal_free = [tree.problem_id for tree in trees if repair_preparations[tree.problem_id]["admitted"]
                                   and not repair_preparations[tree.problem_id]["has_signal"]]
                    result["incomplete_ids"].extend(tree.problem_id for tree in trees if any(
                        reason != "UNRESOLVED_ROUTING" for reason in repair_preparations[tree.problem_id]["admission_reasons"]))
                    write_json(output / f"repair-pair-{index:02d}.json", {"version": "repair-local-training-pair-v1",
                        "protocol_sha256": protocol["sha256"], "capture_binding_sha256": checkpoint.digest(binding),
                        "control": not notebook_enabled, "pair_index": index, "step": result["optimizer_updates"],
                        "trees": repair_tree_links[pair_tree_start:], "raw_records": repair_raw_links[pair_raw_start:]})
                else:
                    eligible, unresolved, signal_free = partition_learning_trees(trees, include_notebooks=notebook_enabled,
                                                                                 notebook_objective=objective)
                discarded = [t.problem_id for t in trees if t not in eligible]
                answer_ids = [tree.problem_id for tree in trees if answer_route
                              and repair_preparations[tree.problem_id]["admission"] == "ANSWER_VERIFIED"]
                if answer_route:
                    result["answer_verified_ids"].extend(answer_ids)
                result["signal_free_ids"].extend(signal_free)
                result["unresolved_ids"].extend(unresolved)
                result["discarded_ids"].extend(discarded)
                if not eligible:
                    write_json(output / f"pair-{index:02d}.json", {
                        "status": "UNRESOLVED_ROUTING" if unresolved else "NO_LEARNING_SIGNAL",
                        "attempted_ids": [t.problem_id for t in trees], "signal_free_ids": signal_free,
                        "unresolved_ids": unresolved, "discarded_ids": discarded,
                        "step": result["optimizer_updates"], **({"answer_verified_ids": answer_ids} if answer_route else {})})
                    continue
                if time.monotonic() >= deadline:
                    result["status"] = "BUDGET_STOP"
                    return result
                update_began = time.monotonic()
                before = parameter_hashes(bundle)
                # The audit rule is fixed in the protocol before the run; non-audit updates skip only evidence VJPs.
                audit = ({"measure_evidence_vjp": scale_options.audited(protocol, result["optimizer_updates"],
                                                                        final_batch=index == len(pairs) - 1)}
                         if audit_rule else {})
                # Moments stay parked through forward/backward and return just before optimizer.step().
                with scale_options.park(bundle, optimizer, scheduler, protocol,
                                        result["optimizer_updates"]) as restore_moments:
                    update = optimizer_step(bundle, optimizer, scheduler, eligible,
                        {t.problem_id: images[t.problem_id] for t in eligible}, engine, protocol, "ALG-NB3",
                        step=result["optimizer_updates"], seed=seed, notebook_reward_enabled=notebook_enabled,
                        observation_actors={a.problem_id: a for a in pair if a.problem_id in {t.problem_id for t in eligible}},
                        objective_verifier=verifier,
                        **({"require_retained_learning_signal": True}
                           if repair_selected or admission in (VERIFIED_UPDATE_EXPLORATION, FIXED_TEN_NATIVE_EXPLORATION) else {}),
                        on_progress=lambda event: print(json.dumps(event, allow_nan=False), flush=True),
                        before_optimizer_step=restore_moments, **audit)
                update.update(attempted_ids=[t.problem_id for t in trees], unresolved_ids=unresolved,
                              signal_free_ids=signal_free, discarded_ids=discarded,
                              **({"answer_verified_ids": answer_ids} if answer_route else {}))
                if scaled and "n_groups_by_kind" in update:
                    update["group_weight_scale"] = _group_weight_scale(update, protocol)
                    measured["group_weight_denominators"].append(update["group_weight_scale"]["denominator"])
                write_json(output / f"pair-{index:02d}.json", update)
                if update["status"] != "UPDATED_UNSYNCHRONIZED":
                    result["failed_update_ids"].extend(t.problem_id for t in eligible)
                    measured["update_seconds"] += time.monotonic() - update_began
                    continue
                if repair_selected:
                    from .repair_evidence import repair_update_evidence

                    fresh = {tree.problem_id: prepare_tree_credit(tree, next(actor for actor in pair
                        if actor.problem_id == tree.problem_id), verifier, engine.tokenizer, preparation=engine,
                        images=images[tree.problem_id], master_seed=seed, notebook_reward_enabled=notebook_enabled)
                        for tree in eligible}
                    evidence = repair_update_evidence(update, protocol, trees=eligible, preparations=fresh,
                        parameter_inventory=bundle.inventory["trainable"], require_s=False, control=not notebook_enabled)
                    evidence_file = f"repair-evidence-{index:02d}.json"
                    write_json(output / evidence_file, evidence)
                    result["repair_commits"].append({"pair_index": index, "file": evidence_file,
                        "sha256": checkpoint.file_hash(output / evidence_file), "S_gradient_verified": evidence["S_gradient_verified"]})
                    if audit_rule:
                        result["S_gradient_verified"] = scale_options.fold_s(result["S_gradient_verified"],
                                                                             evidence["S_gradient_verified"])
                    else:
                        result["S_gradient_verified"] |= evidence["S_gradient_verified"]
                    if answer_route and audit_rule:
                        result["S_text_gradient_verified"] = scale_options.fold_s(result["S_text_gradient_verified"],
                                                                                  evidence["S_text_gradient_verified"])
                    elif answer_route:
                        result["S_text_gradient_verified"] |= evidence["S_text_gradient_verified"]
                elif admission == ASSESSMENT_CREDIT_ADMISSION:
                    from .local_run import assessment_update_evidence

                    if notebook_enabled:
                        evidence = assessment_update_evidence(update, protocol, parameter_inventory=bundle.inventory["trainable"])
                        result.update(assessment_learning_verified=evidence["assessment_learning_verified"],
                                      assessment_only_gradient_verified=evidence["assessment_only_gradient_verified"])
                    else:
                        assessment_control_update_evidence(update, protocol, eligible)
                elif notebook_enabled and "notebook_update_requirement" in protocol:
                    native_relationship_update_evidence(update, protocol)
                    if collection:
                        result["native_relationship_learning_verified"] = True
                from .pair_action_notebook import VERSION as pair_action_version

                if admission == VERIFIED_UPDATE_EXPLORATION or protocol.get("notebook_generation_policy") == pair_action_version:
                    retained_update_signal(update, protocol)
                result["optimizer_updates"] = update["step_after"]
                result["notebook_gradient_contribution_verified"] |= not repair_selected and any(
                    row["reward_gradient_isolated"] and (row["advantage"] != 0 or row.get("nonzero_partial_tokens", 0) > 0)
                    and row["changed_groups"]
                    for row in update["notebook_backward"])
                result["notebook_combined_objective_gradient_observed"] |= not repair_selected and any(
                    row["changed_groups"] for row in update["notebook_backward"])
                after = parameter_hashes(bundle)
                changes = verify_changes(before, after, update)
                changes_seen.append(changes)
                write_json(output / f"changes-{index:02d}.json", {"before": before, "after": after, **changes})
                guard(bundle)
                # Synchronized after every update; vLLM additionally receives the merged-weight push.
                synced, sampler_synced = scale_options.synchronize(bundle, sampler, engine, probes, protocol,
                                                                   result["optimizer_updates"], push=True,
                                                                   **sync_options(prepared))
                write_json(output / f"sync-{index:02d}.json", synced)
                if sampler_synced is not None:
                    write_json(output / f"sampler-sync-{index:02d}.json", sampler_synced)
                if not scaled or result["optimizer_updates"] % opts["checkpoint_every_updates"] == 0:
                    cursor = {"epoch": 0, "position": len(result["attempted_ids"]),
                        "discarded_problem_ids": list(result["discarded_ids"]),
                        "problem_ids": list(launch["problem_ids"]),
                        "consumed_problem_ids": list(result["attempted_ids"]), "sync_pending": False}
                    intermediate = output / f"checkpoint-pair-{index:02d}"
                    if (low := disk_low(output, min_free_gib)) is not None:
                        result.update(status="DISK_LOW", disk_low=low)  # earlier committed checkpoints stay resumable
                        return result
                    checkpoint.save(bundle, optimizer, scheduler, result["optimizer_updates"], cursor,
                        checkpoint.capture_rng(), intermediate, protocol=protocol, binding=binding)
                    guard(bundle)
                    committed = {"status": "INTERMEDIATE_SYNCHRONIZED_UPDATE_SAVED",
                        "path": str(intermediate), "metadata_sha256": checkpoint.file_hash(intermediate / "metadata.json"),
                        "step": result["optimizer_updates"], "pair_index": index, "cursor": cursor,
                        "fixed_ten_training_complete": False, "resume_supported": scaled}
                    write_json(output / f"checkpoint-pair-{index:02d}.json", committed)
                    result["last_committed_checkpoint"] = committed
                measured["update_seconds"] += time.monotonic() - update_began
            if result["attempted_ids"] != launch["problem_ids"]:
                raise ValueError("local train: fixed ten coverage differs")
            preserve_repair_index()
            if result["optimizer_updates"] == 0:
                result["status"] = "UNRESOLVED_ROUTING" if result["unresolved_ids"] else "NO_OPTIMIZER_UPDATES"
                return result
            forward = probe_evidence(engine, probes)
            guard(bundle)
            write_json(output / "updated_forward_probes.json", forward)
            cursor = {"epoch": 0, "position": len(launch["problem_ids"]), "discarded_problem_ids": result["discarded_ids"],
                "problem_ids": launch["problem_ids"], "sync_pending": False}
            if repair_selected:
                cursor["consumed_problem_ids"] = list(result["attempted_ids"])
            if (low := disk_low(output, min_free_gib)) is not None:
                result.update(status="DISK_LOW", disk_low=low)  # no partial final checkpoint is written
                return result
            checkpoint.save(bundle, optimizer, scheduler, result["optimizer_updates"], cursor,
                checkpoint.capture_rng(), output / "checkpoint", protocol=protocol, binding=binding)
            guard(bundle)
            write_json(output / "reload_expected.json", {"trainable_sha256": parameter_hashes(bundle),
                "step": result["optimizer_updates"], "cursor": cursor, "forward_probes": forward,
                "frozen_base_sha256": bundle.base_sha256, "quantization": bundle.inventory.get("quantization"),
                **({"repair_preparation_index_sha256": result["repair_preparation_index_sha256"],
                    "repair_update_evidence_sha256": result["repair_update_evidence_sha256"],
                    "repair_protocol_sha256": protocol["sha256"]} if repair_selected else {})})
        if original_access is not None and original_access["denied_original_access_attempts"]:
            raise ValueError("local train: original BF16 model read attempted")
        if repair_selected and matched is not None and any(checkpoint.file_hash(Path(name)) != sha
                for name, sha in matched["repair_runtime_files"].items()):
            raise ValueError("local train: original matched repair runtime files changed during training")
        result.update(status="LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD",
            both_routes_verified=any(c["vision_on_observed"] for c in changes_seen)
                and any(c["vision_off_only_observed"] for c in changes_seen))
        return result
    except BaseException as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        preserve_repair_index()
        result.update(seconds=time.monotonic()-began, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), original_access_guard=original_access)
        try:
            items = len(result["attempted_ids"]) - prefix_items
            n_samples = protocol["rollout"]["n"]
            scale_options.write_receipt(output, items=items, completion_tokens=meter.tokens if meter else 0,
                wall_seconds=result["seconds"], updates=result["optimizer_updates"] - start_step,
                update_seconds=measured["update_seconds"], n_samples=n_samples.get(1, n_samples.get("1")),
                retries_per_item=measured["retry_groups"] / items if items else 0.,
                startup_seconds=measured["startup_seconds"], extra={"run": "local_train", "status": result["status"],
                    "sampler": scale_options.sampler_label(protocol), "completion_scope": "every_engine_sample_call",
                    "problems_per_update": opts["problems_per_update"],
                    "checkpoint_every_updates": opts["checkpoint_every_updates"],
                    "group_weight_denominators": measured["group_weight_denominators"],
                    "resumed": resumed is not None})
        except Exception as exc:  # measurement never masks the run's own outcome
            result["throughput_receipt_error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "result.json", result)
        print(json.dumps(result), flush=True)


def _resume_point(parent: Path, protocol: dict, prepared: dict, contract: dict) -> dict:
    """Last committed cadence checkpoint of a closed earlier segment, with its prefix re-verified on CPU."""
    provisional = {"pid": os.getpid(), "host": host_identity(), "protocol": json.loads(json.dumps(protocol)),
                   "binding": _binding(prepared, protocol), "preparation_sha256": checkpoint.digest(prepared)}
    old, sha = _resume_parent(parent, protocol, provisional, contract)
    # reload_expected.json follows the final save, so a bare/partial `checkpoint` dir is an interrupted save.
    if (parent / "reload_expected.json").exists():
        raise ValueError("local train: the parent segment completed; nothing to resume")
    # Only saves whose commit record was written (after checkpoint.save returned) and that leave a batch
    # to replay; a save interrupted by a crash or the outer kill falls back to the previous one.
    indices = [int(match.group(1)) for path in parent.iterdir()
               if path.is_dir() and (match := re.fullmatch(r"checkpoint-pair-([0-9]{2,})", path.name))
               and (parent / f"{path.name}.json").is_file()
               and int(match.group(1)) < len(old["execution_contract"]["pair_ids"]) - 1]
    if not indices:
        raise ValueError("local train: the parent segment has no committed checkpoint to resume from")
    index = max(indices)
    inventory = _repair_read(parent / "inventory.json")[0]
    proof = _scaled_training_replay(parent, None, old["execution_contract"], protocol,
                                    parameter_inventory=inventory["trainable"], until_batch=index)
    record = {"from": str(parent), "launch_sha256": sha, "checkpoint": f"checkpoint-pair-{index:02d}",
              "batch_index": index, "step": proof["state"]["step"], "prefix_replay_sha256": proof["sha256"]}
    return {"record": record, "state": proof["state"], "binding": old["binding"], "inventory": inventory,
            "checkpoint": parent / record["checkpoint"]}


def reload_training(capture: Path, output: Path) -> dict:
    import torch

    from .quantized_cache import block_original_access
    from .policy import enable_deterministic_execution
    from .quantized_policy import load_quantized_policy
    from .trainer import synchronize

    launch, result = _read(capture / "launch.json"), _read(capture / "result.json")
    protocol = restore_protocol(launch["protocol"])
    contract = launch["execution_contract"]
    admission = execution_admission(contract, protocol)
    opts = scale_options.options(protocol)
    bf16, batch_rows = opts["policy_precision"] == "bf16", opts["rollout_batch_rows"]
    repair_selected = admission in REPAIR_ADMISSIONS
    source_versions: dict[str, str] = {}
    source_values = {}
    if repair_selected:
        for name in ("launch.json", "result.json", "reload_expected.json"):
            source_values[name], source_versions[name] = _repair_read(capture / name)
        if (checkpoint.digest(launch) != checkpoint.digest(source_values["launch.json"])
                or checkpoint.digest(result) != checkpoint.digest(source_values["result.json"])):
            raise ValueError("local train reload: cached repair launch/result versions changed")
    if checkpoint.digest(contract) != launch["execution_contract_sha256"]:
        raise ValueError("local train reload: execution contract changed")
    if launch.get("optimizer_state_collection_policy") != scale_options.parking_policy(protocol):
        raise ValueError("local train reload: optimizer state collection policy differs")
    if repair_selected and (type(launch.get("pid")) is not int or launch["pid"] <= 0
            or type(result.get("optimizer_updates")) is not int
            or not 1 <= result["optimizer_updates"] <= opts["max_updates"]):
        raise ValueError("local train reload: strict repair PID and actual update count required")
    if (same_process(launch, {"pid": os.getpid(), "host": host_identity()}) or type(launch.get("pid")) is not int
            or owner_alive(launch)
            or launch["protocol"] != json.loads(json.dumps(protocol))
            or launch["binding"]["code_sha256"] != checkpoint.digest(approval.code_hashes())
            or result.get("status") != "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"
            or not 1 <= result.get("optimizer_updates", 0) <= opts["max_updates"]
            or result["attempted_ids"] != launch["problem_ids"] or len(launch["problem_ids"]) != opts["train_problems"]):
        raise ValueError("local train reload: closed current-code fixed-ten updated run required")
    expected = source_values["reload_expected.json"] if repair_selected else _read(capture / "reload_expected.json")
    readiness = {} if bf16 else cache_readiness(Path(launch["cache_check"]), Path(launch["model_path"]), protocol)
    prepared, request, actors = _launch_preparation(launch, protocol)
    actors = contract_actors(actors, contract)
    pairs = scale_options.batches(actors, protocol) if scale_options.scaled(protocol) else fixed_pairs(actors)
    if admission == FIXED_TEN_NATIVE_EXPLORATION:
        validate_native_collection(contract, prepared, protocol, readiness, actors)
        validate_native_updates(capture, result, contract, protocol)
    elif "admission_policy" in contract:
        parent = contract["parent_gate"]
        parent_launch = next(row["path"] for row in parent["files"] if Path(row["path"]).name == "launch.json")
        parent_reload = next(row["path"] for row in parent["files"]
                             if Path(row["path"]).name == "sync.json")
        fresh = training_parent_gate(Path(parent_launch).parent, Path(parent_reload).parent, admission=admission)
        if fresh != parent:
            raise ValueError("local train reload: admission parent proof changed")
    repair_proof = None
    if repair_selected:
        metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
        repair_proof = _repair_training_replay(capture, result, contract, protocol,
                                              parameter_inventory=metadata["inventory"]["trainable"])
        if any(repair_proof["paths"].get(name) != sha for name, sha in source_versions.items()):
            raise ValueError("local train reload: cached repair source versions differ from fresh replay")
        expected_cursor = expected.get("cursor")
        if (type(expected_cursor) is not dict or type(expected_cursor.get("position")) is not int
                or expected_cursor["position"] != opts["train_problems"]
                or expected_cursor.get("consumed_problem_ids") != launch["problem_ids"]
                or expected_cursor.get("sync_pending") is not False):
            raise ValueError("local train reload: strict repair original fixed-ten cursor required")
        if contract["arm"] == "answer-only":
            original = contract["matched_notebook_arm"]
            matched = matched_notebook_arm(Path(original["initialization"]["path"]), prepared, protocol,
                Path(launch["model_path"]), launch["cache_check"] and Path(launch["cache_check"]), pairs, admission=admission,
                parent=contract["parent_gate"], max_seconds=execution_max_seconds(contract, protocol),
                matched_reload=Path(original["fresh_reload"]["path"]))
            if matched != original:
                raise ValueError("local train reload: matched original repair arm or fresh reload changed")
    elif admission == ASSESSMENT_CREDIT_ADMISSION:
        metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
        validate_native_updates(capture, result, contract, protocol,
            parameter_inventory=metadata["inventory"]["trainable"])
    if (training_binding(prepared, protocol, contract) != launch["binding"]
            or checkpoint.digest(contract) != launch["execution_contract_sha256"]
            or contract["arm"] != result["arm"]
            or contract["notebook_reward_enabled"] != (result["arm"] == "notebook")
            or contract["pair_ids"] != [[a.problem_id for a in pair] for pair in pairs]
            or result.get("initialization_sha256") != checkpoint.file_hash(capture / "initialization.json")
            or [a.problem_id for a in actors] != launch["problem_ids"]):
        raise ValueError("local train reload: input binding changed")
    output = artifact_path(output)
    if output.resolve().is_relative_to(capture.resolve()):
        raise ValueError("local train reload: preserve closed capture")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    _install_actor_guards(prepared, request)
    original_access = None if bf16 else block_original_access(Path(readiness["original_model_root"]))
    runtime = enable_deterministic_execution()
    loader = scale_options.build_bf16_policy if bf16 else load_quantized_policy
    bundle, optimizer, scheduler, step, cursor = checkpoint.load_resume(Path(launch["model_path"]),
        capture / "checkpoint", protocol, binding=launch["binding"], device="cuda:0",
        policy_factory=partial(loader, device_index=0))
    if (parameter_hashes(bundle) != expected["trainable_sha256"] or step != expected["step"]
            or step != result["optimizer_updates"] or cursor != expected["cursor"]
            or cursor["position"] != opts["train_problems"] or cursor["problem_ids"] != launch["problem_ids"]
            or cursor["discarded_problem_ids"] != result["discarded_ids"]
            or cursor["sync_pending"] is not False or not optimizer.state or scheduler.last_epoch != step
            or bundle.base_sha256 != expected["frozen_base_sha256"]
            or bundle.inventory.get("quantization") != expected["quantization"]):
        raise ValueError("local train reload: restored weights/optimizer/cursor differ")
    guard = scale_options.base_guard(bundle, protocol)
    guard(bundle)
    # Saved probes come from the live HF policy engine in every engine mode.
    engine = scale_options.shared_engine(bundle, protocol, policy_version=step)
    with ExitStack() as stack:
        probe_inputs = probe_source(prepared, actors)
        images = {a.problem_id: load_images(a, Path(prepared["image_root"])) for a in with_probe_actors(actors, probe_inputs)}
        for values in images.values():
            for image in values:
                stack.callback(image.close)
        probes = make_probes(probe_inputs, images, engine, local_prompts(protocol), image_counts=probe_counts(prepared))
        write_json(output / "sync.json", synchronize(bundle, engine, engine, probes, protocol, step, push=False,
                                                     **sync_options(prepared)))
        actual = probe_evidence(engine, probes)
        write_json(output / "forward_probes.json", actual)
        if actual != expected["forward_probes"]:
            raise ValueError("local train reload: actual forward/generation differs")
        if repair_selected:
            _repair_probe_binding(actual, _read(output / "sync.json"), step, probe_inputs, batch_rows=batch_rows,
                                  image_counts=probe_counts(prepared))
        guard(bundle)
    if original_access is not None and original_access["denied_original_access_attempts"]:
        raise ValueError("local train reload: original model read attempted")
    if repair_selected:
        assert repair_proof is not None
        if any(checkpoint.file_hash(capture / name) != sha for name, sha in repair_proof["paths"].items()):
            raise ValueError("local train reload: original repair evidence changed during actual reload")
        if contract["arm"] == "answer-only" and any(checkpoint.file_hash(Path(name)) != sha
                for name, sha in contract["matched_notebook_arm"]["repair_runtime_files"].items()):
            raise ValueError("local train reload: original matched repair evidence changed during actual reload")
    restored = {"status": "LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED", "fresh_reload_verified": True,
        "pid": os.getpid(), "host": host_identity(), "capture_pid": launch["pid"], "step": step,
        "all_trainable_hashes_match": True, "optimizer_restored": True,
        "actual_forward_and_generation_match": True, "frozen_quantized_base_unchanged": True,
        "self_improvement_verified": False, "original_access_guard": original_access,
        "deterministic_execution": runtime,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    if scale_options.scaled(protocol):
        # For BF16 the frozen-base key above means the unquantized base passed assert_base_unchanged.
        restored["policy_precision"] = opts["policy_precision"]
    if admission in (FIXED_TEN_NATIVE_EXPLORATION, ASSESSMENT_CREDIT_ADMISSION, *REPAIR_ADMISSIONS):
        restored.update(execution_contract=contract, execution_contract_sha256=checkpoint.digest(contract),
                        capture_binding_sha256=checkpoint.digest(launch["binding"]))
    if repair_selected:
        assert repair_proof is not None
        restored.update(repair_functional_verified=True, S_gradient_verified=repair_proof["S_gradient_verified"],
            repair_learning_verified=repair_proof["S_gradient_verified"], benefit_verified=False,
            repair_replay_sha256=repair_proof["sha256"], repair_protocol_sha256=protocol["sha256"],
            repair_update_evidence_sha256=checkpoint.file_hash(capture / "repair_update_evidence.json"))
    if admission == ASSESSMENT_CREDIT_ADMISSION:
        restored.update({key: result[key] for key in ("assessment_learning_verified", "assessment_only_gradient_verified",
            "strict_notebook_correctness_verified", "accurate_whole_notebook_verified", "visual_improvement_verified")})
    write_json(output / "result.json", restored)
    return restored


def _problem_range(value: str) -> tuple[int, int]:
    first, _, last = value.partition(":")
    if not first.isdigit() or not last.isdigit() or int(first) >= int(last):
        raise argparse.ArgumentTypeError("problem range must be a:b with integers 0 <= a < b")
    return int(first), int(last)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    run = sub.add_parser("train")
    for name in ("preparation", "model-path", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    for name in ("minimal", "minimal-reload"):
        run.add_argument("--" + name, type=Path)
    run.add_argument("--cache-check", type=Path, help="required for the NF4 policy; absent for a bound bf16 policy")
    run.add_argument("--resume-from", type=Path,
                     help="closed earlier segment (scale group only); writes a NEW segment directory")
    run.add_argument("--problem-range", type=_problem_range,
                     help="a:b chunk of the fixed selection (scale group only; b-a must equal train_problems)")
    run.add_argument("--arm", choices=("notebook", "answer-only"), required=True)
    run.add_argument("--match-initialization", type=Path)
    run.add_argument("--matched-reload", type=Path)
    run.add_argument("--admission", choices=(STRICT_ADMISSION, EXPLORATORY_ADMISSION, SYNTHESIS_CREDIT_ADMISSION,
                                           ASSESSMENT_CREDIT_ADMISSION, VERIFIED_UPDATE_EXPLORATION,
                                           FIXED_TEN_NATIVE_EXPLORATION, *REPAIR_ADMISSIONS), default=STRICT_ADMISSION)
    run.add_argument("--max-seconds", type=int, default=14400)
    run.add_argument("--min-free-gib", type=float, help="stop with DISK_LOW instead of writing a checkpoint when "
                     "the output filesystem has less free space (default: no check)")
    run.add_argument("--note-contract", choices=NOTE_CONTRACT_CHOICES, default=DEFAULT_NOTE_CONTRACT,
                     help="must equal the final-note contract restored from the minimum parent protocol")
    run.add_argument("--answer-verified-route", action="store_true",
                     help="must equal the answer-verified route bound in the minimum parent protocol")
    run.add_argument("--finalization-reward", action="store_true",
                     help="must equal the finalization reward bound in the minimum parent protocol")
    run.add_argument("--slotless-source", action="append", default=[],
                     help="repeatable; must equal the slotless sources bound in the minimum parent protocol")
    add_frozen_approval_arguments(run)
    reload = sub.add_parser("reload")
    reload.add_argument("--capture", type=Path, required=True)
    reload.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "train":
        flags: dict[str, Any] = {**({"answer_verified_route": True} if args.answer_verified_route else {}),
                                 **({"finalization_reward": True} if args.finalization_reward else {}),
                                 **({"slotless_sources": tuple(args.slotless_source)} if args.slotless_source else {})}
        flags.update({**({"resume_from": args.resume_from} if args.resume_from is not None else {}),
                      **({"problem_range": args.problem_range} if args.problem_range is not None else {})})
        frozen = frozen_approval_option(parser, args)
        if frozen is not None:
            flags["frozen_approval"] = frozen
        if args.min_free_gib is not None:
            flags["min_free_gib"] = args.min_free_gib
        result = train(args.preparation, args.model_path, args.cache_check, args.minimal,
            args.minimal_reload, args.output, arm=args.arm, match_initialization=args.match_initialization,
            admission=args.admission, max_seconds=args.max_seconds,
            **({"matched_reload": args.matched_reload} if args.matched_reload is not None else {}),
            **({"note_contract": args.note_contract} if args.note_contract != DEFAULT_NOTE_CONTRACT else {}), **flags)
    else:
        result = reload_training(args.capture, args.output)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] in ("LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD",
        "LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
