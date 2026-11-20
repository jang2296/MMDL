"""Fresh original-token repair preparation; no optimizer or execution authority."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Literal, cast

from .repair_contract import FINALIZATION_VERSION, FINALIZATION_WEIGHT
from .schema import ActorItem, ProblemTree, Sample, from_json, to_json

VERSION = "repair-tree-credit-v1"
_FINALIZED = ("CORRECT", "INCORRECT")


def finalization_reward(sample: Sample, response_contract: str | None) -> float:
    """1.0 iff one parseable Final answer line ended normally (CORRECT or INCORRECT), else 0.0.

    Gold-free and record-only: TRUNCATED, INVALID_NO_FINAL, INVALID_MULTIPLE and a malformed
    retry (INVALID_RESPONSE_CONTRACT, result=None) all earn 0. Correctness itself is not rewarded here.
    """
    from .repair_contract import public_attempt_outcome

    if type(sample) is not Sample or sample.kind not in ("T1", "T2", "T3"):
        raise ValueError("repair finalization: original solution sample required")
    outcome, _ = public_attempt_outcome(sample.text, response_contract=response_contract, result=sample.result)
    if outcome is None:
        raise ValueError("repair finalization: graded or proven-malformed original outcome required")
    return 1.0 if outcome in _FINALIZED and sample.finish_reason == "stop" else 0.0


def finalization_vector(rewards: list[float], index: int, tokens: int, *, weight: float, n: int) -> list[float]:
    """(r_i - mean over the original n) * weight on every generated token; equal rewards give exact zeros."""
    if (type(rewards) is not list or len(rewards) != n or any(value not in (0., 1.) for value in rewards)
            or type(index) is not int or not 0 <= index < n or type(tokens) is not int or tokens < 1
            or weight != FINALIZATION_WEIGHT):
        raise ValueError("repair finalization: original group rewards, member and fixed weight required")
    advantage = rewards[index] - sum(rewards) / n
    return [weight * advantage] * tokens


def prepare_tree_credit(tree: ProblemTree, actor: ActorItem, verifier: Any, tokenizer: Any, *,
                        preparation: Any, images: list, master_seed: int,
                        notebook_reward_enabled: bool = True) -> dict:
    """Reproduce physical inputs and fresh semantics before preparing detached vectors."""
    from PIL.Image import Image

    from . import compact_note, episode_notebook as episode, repair_notebook as notes
    from .repair_contract import (FINAL_VERSION, ORIGIN_VERSION, VERSION as RETRY_VERSION,
                                  PublicOriginInventory, RepairContractError, RetrySections, parse_final_note, parse_retry_sections,
                                  public_origin_inventory)
    from .repair_credit import (FinalRepairAudit, RepairAudit, _digest, audit_final_note,
                                audit_online_repair, repair_group_credit)
    from .repair_tokens import span_token_ownership

    if (type(tree) is not ProblemTree or type(actor) is not ActorItem or verifier is None or tokenizer is None
            or preparation is None or not callable(getattr(preparation, "prepare", None))
            or getattr(preparation, "tokenizer", None) is not tokenizer
            or type(images) is not list or len(images) != len(actor.image_paths)
            or type(master_seed) is not int or not 0 <= master_seed < 2**31
            or type(notebook_reward_enabled) is not bool):
        raise ValueError("repair preparation: actual physical inputs and explicit control required")
    original_tree, original_actor = to_json(tree), to_json(actor)
    current = from_json(ProblemTree, original_tree)
    actual_actor = from_json(ActorItem, original_actor)
    original_images = list(images)

    def image_state() -> str:
        return _digest([(id(image), image.mode, image.size, hashlib.sha256(image.tobytes()).hexdigest())
                        if isinstance(image, Image) else (id(image),) for image in images])

    image_sha = image_state()

    def unchanged() -> None:
        if (to_json(tree) != original_tree or to_json(actor) != original_actor
                or to_json(current) != original_tree or to_json(actual_actor) != original_actor
                or images != original_images or image_state() != image_sha
                or preparation.policy_version != current.policy_version
                or preparation.tokenizer is not tokenizer):
            raise ValueError("repair preparation: original source or physical input mutated")

    record = current.notebook.get("rewarded_notebook", {})
    capabilities = notes.validate_update_binding(current, tokenizer=tokenizer, preparation=preparation,
        images=images, observation_actor=actual_actor, observation_master_seed=master_seed)
    unchanged()
    source = notes.audit_tree(current, actual_actor, verifier, prompts=record["source_prompts"],
        master_seed=master_seed, tokenizer=tokenizer, preparation=preparation, images=images)
    unchanged()
    if (_digest(source) != _digest(record["source_audit"])
            or source["capabilities"] != {"fresh_native_audit": True, "token_text_decoded": True,
                                           "source_inputs_reprocessed": True}):
        raise ValueError("repair preparation: fresh physical source audit differs")
    history_all = episode.solution_samples(current)
    contracts = dict(source["response_contracts"])
    full_inventory = public_origin_inventory(history_all, contract_version=ORIGIN_VERSION,
                                             response_contracts=contracts)
    if _digest(asdict(full_inventory)) != _digest(source["inventory"]):
        raise ValueError("repair preparation: full source inventory differs")
    source_rows = {row.sample_id: row for row in full_inventory.sources}
    compact = notes.note_contract(record["source_prompts"]) == compact_note.VERSION
    routing = source["routing"]
    bound = record["source_prompts"]
    route_policy = bound.get("repair_route_policy")
    finalization = "repair_finalization_contract" in bound
    text_route = routing["admission"] == episode.ANSWER_VERIFIED
    if (finalization and (bound["repair_finalization_contract"] != FINALIZATION_VERSION or not compact
                          or type(bound.get("repair_finalization_weight")) is not float
                          or bound["repair_finalization_weight"] != FINALIZATION_WEIGHT)
            or text_route and (route_policy != episode.ANSWER_ROUTE_RULE or not compact)):
        raise ValueError("repair preparation: opt-in route or finalization channel requires its bound protocol")
    reasons = []
    if routing["admission"] == "UNRESOLVED":
        reasons.append("UNRESOLVED_ROUTING")
    if episode.solution_skips(current):
        reasons.append("SKIPPED_SOLUTION_BRANCH")
    if any(not group.usable for group in current.groups if group.kind != "DIAG"):
        reasons.append("UNUSABLE_SOLUTION_GROUP")
    if record["status"] not in ("GENERATED", "NO_CORRECT_ANSWER"):
        reasons.append("INCOMPLETE_FINAL_COLLECTION")
    if not routing["complete_episode"]:
        reasons.append("INCOMPLETE_SOLUTION_COLLECTION")
    admitted = not reasons
    groups = []
    history: list[Sample] = []
    signal = dict.fromkeys(("answer", "online", "final") + (("finalization",) if finalization else ()), False)
    from .prompts import turn1_contract

    turn1 = turn1_contract(actual_actor, bound) if "repair_turn1_contract" in bound else None
    retry_rule = bound.get("repair_retry_out_of_choice") if turn1 is not None else None
    for group in current.groups:
        size, cap, weight = (4, 8192, .5) if group.kind == "DIAG" else (
            (4, 1024 if turn1 is None else turn1["max_new_tokens"], 1.) if group.kind == "T1" else (2, 1024, 1.))
        if (len(group.samples) != size or group.kind not in ("T1", "T2", "T3", "DIAG")
                or any(not 0 < len(row.completion_token_ids) <= cap for row in group.samples)):
            raise ValueError("repair preparation: original fixed group size or cap differs")
        component: Literal["online", "final"] | None = (
            "final" if group.kind == "DIAG" else "online" if group.kind in ("T2", "T3") else None)
        inventory = (full_inventory if component == "final" else public_origin_inventory(history,
            contract_version=ORIGIN_VERSION, response_contracts={row.sample_id: contracts[row.sample_id] for row in history},
            parent_sample_id=group.samples[0].parent_sample_id) if component == "online" else None)
        audits: dict[str, RepairAudit | FinalRepairAudit | None] = {}
        sections: dict[str, RetrySections] = {}
        failures = {}
        compact_audits: dict[str, dict | None] | None = None
        grades: list[dict] = []
        if component == "final" and compact:
            # Same _grade the generator sealed, so the fresh recomputation is byte-identical.
            grades = [notes._grade(row, current, actual_actor, verifier, source, compact_note.VERSION, route_policy)
                      for row in group.samples]
            if grades != record["grades"]:
                raise ValueError("repair preparation: fresh final semantics differ")
            compact_audits = {grade["sample_id"]: grade["audit"] for grade in grades}
            failures = {grade["sample_id"]: grade["syntax_failure"] for grade in grades}
        for row in (group.samples if compact_audits is None else ()):
            failure = None
            audit: RepairAudit | FinalRepairAudit | None = None
            if component is not None:
                try:
                    parsed = (parse_retry_sections(row.text, contract_version=RETRY_VERSION) if component == "online"
                              else parse_final_note(row.text, contract_version=FINAL_VERSION))
                except RepairContractError as exc:
                    if exc.code not in notes._GRAMMAR_ERRORS:
                        raise
                    failure = {"code": exc.code, "at": exc.at, "offset": exc.offset}
                    if component == "online" and (source_rows[row.sample_id].projection_error_code != exc.code
                                                   or row.result is not None or row.reward != 0):
                        raise ValueError("repair preparation: malformed source is not freshly bound") from None
                else:
                    assert inventory is not None
                    if component == "online":
                        retry = cast(RetrySections, parsed)
                        sections[row.sample_id] = retry
                        audit = audit_online_repair(history, row, actual_actor, verifier, inventory=inventory,
                            response_contracts={**{prior.sample_id: contracts[prior.sample_id] for prior in history},
                                                row.sample_id: contracts[row.sample_id]}, sections=retry,
                            turn1_answer_contract=None if turn1 is None else turn1["version"],
                            retry_out_of_choice=retry_rule)
                    else:
                        audit = audit_final_note(row, history_all, actual_actor, verifier,
                                                 inventory=inventory, response_contracts=contracts,
                                                 turn1_answer_contract=None if turn1 is None else turn1["version"],
                                                 retry_out_of_choice=retry_rule)
                audits[row.sample_id] = audit
                failures[row.sample_id] = failure
            if component == "final":
                grade = {"sample_id": row.sample_id, "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                    "finish_reason": row.finish_reason, "syntax_failure": failure,
                    "audit": json.loads(json.dumps(asdict(audit))) if audit is not None else None,
                    "scalar_reward": 0., "component_credit_applied": False, "eligible_for_rl_loss": False}
                ordinal = len(audits) - 1
                if grade != record["grades"][ordinal]:
                    raise ValueError("repair preparation: fresh final semantics differ")
        # An unusable solution group (BLOCKING verifier status) already denies admission via
        # UNUSABLE_SOLUTION_GROUP; it gets no component credit, and the strict credit check stays unchanged.
        excluded = not group.usable and group.kind != "DIAG"
        if excluded and admitted:
            raise ValueError("repair preparation: an unusable solution group cannot be admitted")
        credits = (() if excluded
                   else compact_note.group_credit(group, grades, tokenizer, completion_cap=cap,
                                                  route_admission=routing["admission"] if text_route else None)
                   if compact_audits is not None
            else repair_group_credit(group, audits, tokenizer, inventory=cast(PublicOriginInventory, inventory),
                                     completion_cap=cap, component=component) if component is not None else ())
        answers = [row.reward if group.kind != "DIAG" else 0. for row in group.samples]
        if any(type(value) not in (int, float) or value not in (0., 1.) for value in answers):
            raise ValueError("repair preparation: binary original answer reward required")
        mean = sum(answers) / size
        finalized = ([finalization_reward(row, contracts[row.sample_id]) for row in group.samples]
                     if finalization and group.kind != "DIAG" else None)
        samples = []
        for index, row in enumerate(group.samples):
            sample_audit = audits.get(row.sample_id)
            sample_failure = failures.get(row.sample_id)
            spans = (sections[row.sample_id].solution_span,) if row.sample_id in sections else ()
            ownership = span_token_ownership(row.text, row.completion_token_ids, tokenizer, spans)
            if group.kind == "T1":
                ownership.update(status="FULL_ORIGINAL_ACTION", spans=[[0, len(row.text)]],
                                 token_indices=list(range(len(row.completion_token_ids))))
            owned = set(ownership["token_indices"])
            answer = [answers[index] - mean if token in owned else 0. for token in range(len(row.completion_token_ids))]
            vectors = {"answer": answer, "online": [0.] * len(answer), "final": [0.] * len(answer)}
            if component is not None and credits:
                vectors[component] = list(credits[index].coefficients)
            if finalization:
                vectors["finalization"] = ([0.] * len(answer) if finalized is None else finalization_vector(
                    finalized, index, len(answer), weight=FINALIZATION_WEIGHT, n=size))
            # Finalization is a gold-free solution channel like answer (same in both control arms);
            # online repair credit is recorded but never applied on the text-only answer route.
            applied = {name: values if admitted and (name in ("answer", "finalization") or notebook_reward_enabled)
                       and not (text_route and name == "online") else [0.] * len(answer)
                       for name, values in vectors.items()}
            for name, values in applied.items():
                signal[name] |= any(value != 0 for value in values)
            samples.append({"sample_id": row.sample_id, "group_id": group.group_id,
                "raw_sha256": ownership["raw_sha256"], "token_ids_sha256": ownership["token_ids_sha256"],
                "sampler_logprobs_sha256": _digest(row.sampler_logprobs), "answer_reward": answers[index],
                "answer_advantage": answers[index] - mean, "answer_ownership": ownership,
                "action_mask": [True] * len(answer),
                **{name + "_coefficients": values for name, values in vectors.items()},
                **{"applied_" + name + "_coefficients": values for name, values in applied.items()},
                "component_credit": asdict(credits[index]) if credits else None,
                "semantic_audit": (compact_audits[row.sample_id] if compact_audits is not None
                                   else asdict(sample_audit) if sample_audit is not None else None),
                "parser_error_code": sample_failure["code"] if sample_failure is not None else None,
                **({"finalization_reward": None if finalized is None else finalized[index]} if finalization else {})})
        groups.append({"group_id": group.group_id, "kind": group.kind, "group_weight": weight,
            "original_group_size": size, "completion_cap": cap, "members": [row.sample_id for row in group.samples],
            "answer_mean": mean, "inventory": asdict(inventory) if inventory is not None else None,
            "source_group_sha256": _digest(json.loads(to_json(group))), "samples": samples})
        if group.kind != "DIAG":
            history.extend(group.samples)
        unchanged()
    bindings = {"tree_sha256": _digest(json.loads(original_tree)), "actor_sha256": _digest(json.loads(original_actor)),
        "source_audit": source, "response_contracts": contracts, "master_seed": master_seed,
        "group_members_sha256": _digest([group["members"] for group in groups]),
        "physical_capabilities": capabilities, "fresh_semantics_verified": True,
        "semantic_scope": "native_rule_local_repair", "module_causality_certified": False,
        **({"final_note_contract": compact_note.VERSION} if compact else {}),
        "all_reasoning_verified": False, "optimizer_admission": False}
    receipt = {"version": VERSION, "problem_id": current.problem_id, "policy_version": current.policy_version,
        "route": routing["route"], "admission": routing["admission"], "admitted": admitted,
        "admission_reasons": reasons, "notebook_reward_enabled": notebook_reward_enabled,
        "has_signal": any(signal.values()), "signal_components": signal, "groups": groups, "source_bindings": bindings}
    unchanged()
    detached = json.loads(json.dumps(receipt, allow_nan=False))
    return detached | {"sha256": _digest(detached)}
