"""One on-policy optimizer update, not a deployment or data-approval entry point.

Inputs are actual sampled actions and bounded verifier results. No answer key,
teacher completion, labels, or cross-entropy target enters this module. The caller
must synchronize the rollout engine after UPDATED before collecting another batch.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .answers import extract_final
from .engines import RolloutRequest
from .losses import group_advantages, group_loss, k3, mis_mask
from .notebook_learning import NOTEBOOK_REWARD_VERSION
from .episode_notebook import STRICT_ROUTE_RULE, VERSION as EPISODE_REWARD_VERSION, validate_update_binding
from .policy import PolicyBundle, completion_logprobs, reference_logprobs
from .rollout import _hash
from .schema import ActorItem, Group, ProblemTree, Sample, to_json


NATIVE_RELATIONSHIP_REQUIREMENT = "positive-native-relationship-post-mis-v1"
ASSESSMENT_REQUIREMENT = "verified-native-assessment-aggregate-gradient-v2"


def _objective_identity(tree: ProblemTree, objective: dict) -> dict:
    from .notebook_objective import SYNTHESIS_CREDIT_VERSION, contract
    from .notebook_synthesis import BOUNDED_VERSION
    from .synthesis_credit import ASSESSMENT_VERSION

    version = objective.get("objective_version")
    if not isinstance(version, str) or objective != contract(version):
        raise ValueError("update: unknown notebook objective contract")
    validate_update_binding(tree)
    note = tree.notebook.get("rewarded_notebook", {})
    synthesis_credit = version in (SYNTHESIS_CREDIT_VERSION, ASSESSMENT_VERSION)
    if version == ASSESSMENT_VERSION:
        from .whole_reference_notebook import VERSION as whole_reference_version

        if note.get("notebook_generation_policy") != whole_reference_version:
            raise ValueError("update: assessment credit requires monolithic whole-reference notebooks")
    if synthesis_credit and (note.get("reward_version") != BOUNDED_VERSION
            or note.get("notebook_synthesis_contract") != BOUNDED_VERSION):
        raise ValueError("update: synthesis unit credit requires the explicit bounded synthesis contract")
    if not synthesis_credit and note.get("reward_version") != EPISODE_REWARD_VERSION:
        raise ValueError("update: atomic objective requires full-episode notebooks")
    identity = {"contract": objective, "problem_id": tree.problem_id,
            "policy_version": tree.policy_version,
            "solution_sha256": note["episode_audit"]["solution_sha256"],
            "episode_audit_sha256": note["episode_audit_sha256"],
            "notebook_groups_sha256": note["notebook_groups_sha256"]}
    if synthesis_credit:
        identity["synthesis_audit_sha256"] = note["synthesis_audit_sha256"]
    return identity


def _score_notebook_objective(tree: ProblemTree, actor: ActorItem, verifier: Any, objective: dict, *,
                              replay_relation_v1: bool = False) -> dict:
    from .episode_notebook import _fact_projection, audit_episode, verify_replayed_outcome
    from . import notebook_objective
    from .notebook_objective import SYNTHESIS_CREDIT_VERSION
    from .synthesis_credit import ASSESSMENT_VERSION

    receipt = _objective_identity(tree, objective)
    note = tree.notebook["rewarded_notebook"]
    synthesis_credit = objective["objective_version"] in (SYNTHESIS_CREDIT_VERSION, ASSESSMENT_VERSION)
    score_note = notebook_objective.score_note
    if synthesis_credit:
        from . import synthesis_credit as credit_objective

        score_note = credit_objective.score_note
    from .pair_action_notebook import VERSION as pair_action_version
    from .whole_reference_notebook import VERSION as whole_reference_version

    from .whole_reference_pages import VERSION as pages_version

    if note.get("notebook_generation_policy") == pages_version:
        if not synthesis_credit or replay_relation_v1:
            raise ValueError("update: pages require the current synthesis unit objective")
        receipt["samples"] = (credit_objective.score_page_bundles(tree, note["synthesis_audit"], verifier, actor=actor)
                              if any(group.kind == "DIAG" for group in tree.groups) else {})
        return receipt | {"sha256": _hash(receipt)}
    sample_bound_generation = note.get("notebook_generation_policy") in (pair_action_version, whole_reference_version)
    if not synthesis_credit and replay_relation_v1 and audit_episode(tree, actor, verifier,
            routing_rule=note["episode_audit"].get("routing_rule"), replay_relation_v1=True) != note["episode_audit"]:
        raise ValueError("update: historical source audit differs from fresh verification")
    notes = {row["sample_id"]: row for row in tree.notebook.get("model_notes", [])}
    if len(notes) != len(tree.notebook.get("model_notes", [])):
        raise ValueError("update: duplicate strict notebook grades")
    if (synthesis_credit and note.get("notebook_generation_policy") == pair_action_version
            and any(group.kind == "DIAG" for group in tree.groups)):
        if replay_relation_v1 is not False:
            raise ValueError("synthesis credit: explicit current objective and verifier required")
        strict_reports = {sample_id: {key: value for key, value in stored.items()
                          if key not in ("sample_id", "text", "eligible_for_rl_loss")}
                          for sample_id, stored in notes.items()}
        receipt["samples"] = credit_objective._score_pair_notes(tree, note["synthesis_audit"], verifier,
                                                               strict_reports, actor=actor)
        return receipt | {"sha256": _hash(receipt)}
    scored = {}
    for group in tree.groups:
        if group.kind != "DIAG":
            continue
        for sample in group.samples:
            stored = notes[sample.sample_id]
            strict = {key: value for key, value in stored.items()
                      if key not in ("sample_id", "text", "eligible_for_rl_loss")}
            scored[sample.sample_id] = score_note(sample.text, sample.finish_reason, tree,
                note["synthesis_audit" if synthesis_credit else "episode_audit"], verifier,
                strict_report=strict, actor=actor,
                objective_version=objective["objective_version"], replay_relation_v1=replay_relation_v1,
                **({"sample_id": sample.sample_id} if sample_bound_generation else {}))
            if replay_relation_v1 and not synthesis_credit:
                claims = strict.get("submitted_claims")
                if sample.result is None:
                    if claims is not None:
                        raise ValueError("update: historical submitted claims lack their verifier result")
                elif claims is None:
                    raise ValueError("update: historical verifier result lacks submitted claims")
                else:
                    fresh = verifier.grade(tree.problem_id,
                        _fact_projection(claims["corrected_facts"], claims["corrected_answer"]), "stop", "obs")
                    verify_replayed_outcome(sample.result, fresh, replay_relation_v1=True)
    receipt["samples"] = scored
    return receipt | {"sha256": _hash(receipt)}


def bind_notebook_objective(tree: ProblemTree, actor: ActorItem, verifier: Any, objective: dict) -> dict:
    """Attach host-only rewards once; original sampled outcomes remain unchanged."""
    if "learning_objective" in tree.notebook:
        raise ValueError("update: notebook objective already bound")
    receipt = _score_notebook_objective(tree, actor, verifier, objective)
    tree.notebook["learning_objective"] = receipt
    return receipt


def notebook_objective_scores(tree: ProblemTree, objective: dict | None) -> dict[str, dict]:
    """Check recorded scores for selection; optimizer must also regrade them fresh."""
    from .notebook_objective import SYNTHESIS_CREDIT_VERSION, effective_reward
    from .synthesis_credit import ASSESSMENT_VERSION
    from .local_eval import digest

    receipt = tree.notebook.get("learning_objective")
    if objective is None:
        if receipt is not None:
            raise ValueError("update: notebook objective must be explicitly selected")
        return {}
    identity = _objective_identity(tree, objective)
    if (type(receipt) is not dict or set(receipt) != {*identity, "samples", "sha256"}
            or any(receipt[key] != value for key, value in identity.items())
            or receipt["sha256"] != _hash({k: v for k, v in receipt.items() if k != "sha256"})
            or type(receipt["samples"]) is not dict):
        raise ValueError("update: stale notebook objective binding")
    samples = {s.sample_id: s for g in tree.groups if g.kind == "DIAG" for s in g.samples}
    if set(receipt["samples"]) != set(samples):
        raise ValueError("update: notebook objective sample coverage differs")
    for sample_id, row in receipt["samples"].items():
        sample = samples[sample_id]
        if objective["objective_version"] in (SYNTHESIS_CREDIT_VERSION, ASSESSMENT_VERSION):
            from .synthesis_credit import validate_score

            if row.get("objective_version") != objective["objective_version"]:
                raise ValueError("update: synthesis objective score version differs")
            validate_score(row, sample)
            if (row.get("solution_sha256") != identity["solution_sha256"]
                    or row.get("audit_sha256") != identity["synthesis_audit_sha256"]
                    or row.get("synthesis_audit_sha256") != identity["synthesis_audit_sha256"]
                    or row.get("episode_audit_sha256") != identity["episode_audit_sha256"]):
                raise ValueError("update: synthesis objective source binding differs")
            continue
        if (type(row) is not dict or type(row.get("numerator")) is not int
                or type(row.get("denominator")) is not int):
            raise ValueError("update: notebook objective arithmetic differs")
        if (row.get("objective_version") != objective["objective_version"]
                or row.get("objective_independently_verified") is not True
                or row.get("solution_sha256") != identity["solution_sha256"]
                or row.get("audit_sha256") != digest(tree.notebook["rewarded_notebook"]["episode_audit"])
                or row.get("text_sha256") != digest(sample.text)
                or row.get("finish_reason") != sample.finish_reason
                or row.get("strict_reward") != sample.reward
                or row.get("effective_reward") != effective_reward(
                    sample.reward, row["numerator"], row["denominator"])
                or row.get("atomic_reward") != effective_reward(0, row["numerator"], row["denominator"])
                or row.get("atom_fraction") != (row["numerator"] / row["denominator"] if row["denominator"] else 0)):
            raise ValueError("update: notebook objective arithmetic differs")
    return receipt["samples"]


def has_learning_signal(tree: ProblemTree, *, include_notebooks: bool = False,
                        notebook_objective: dict | None = None) -> bool:
    from .notebook_synthesis import BOUNDED_VERSION

    to_json(tree)
    note = tree.notebook.get("rewarded_notebook", {})
    if note.get("reward_version") in (EPISODE_REWARD_VERSION, BOUNDED_VERSION):
        validate_update_binding(tree)
        if (not note["episode_audit"]["complete_episode"]
                or note["episode_audit"].get("admission") == "UNRESOLVED"):
            return False
    notebook_groups: list[str | None] = []
    if (include_notebooks and note.get("status") == "GENERATED"
            and note.get("reward_version") in (NOTEBOOK_REWARD_VERSION, EPISODE_REWARD_VERSION, BOUNDED_VERSION)):
        notebook_groups = (note["group_ids"] if note["reward_version"] in (EPISODE_REWARD_VERSION, BOUNDED_VERSION)
                           else [note.get("group_id")])
    scores = notebook_objective_scores(tree, notebook_objective)
    return any(g.usable and (g.kind != "DIAG" or g.group_id in notebook_groups)
               and group_has_signal(g, scores)
               for g in tree.groups)


def group_has_signal(group: Group, scores: dict[str, dict]) -> bool:
    from .notebook_objective import FACT_CREDIT_VERSION, SYNTHESIS_CREDIT_VERSION
    from .synthesis_credit import ASSESSMENT_VERSION

    if group.kind == "DIAG" and scores:
        rows = [scores[s.sample_id] for s in group.samples]
        if all(row["objective_version"] in (SYNTHESIS_CREDIT_VERSION, ASSESSMENT_VERSION) for row in rows):
            from .synthesis_credit import has_signal

            return has_signal(group.samples, scores)
        if all(row["objective_version"] == FACT_CREDIT_VERSION for row in rows):
            keys = sorted({key for row in rows for key in row["fact_rewards"]})
            return len({(s.reward, *(row["fact_rewards"].get(k, 0) for k in keys))
                        for s, row in zip(group.samples, rows, strict=True)}) > 1
        return len({row["effective_reward"] for row in rows}) > 1
    return len({s.reward for s in group.samples}) > 1


def notebook_signal_groups(tree: ProblemTree, scores: dict[str, dict]) -> int:
    """Count one complete paged notebook set as one logical DIAG signal group."""
    from .whole_reference_pages import VERSION as pages_version

    count = sum(group.usable and group.kind == "DIAG" and group_has_signal(group, scores) for group in tree.groups)
    return int(bool(count)) if tree.notebook.get("rewarded_notebook", {}).get("notebook_generation_policy") == pages_version else count


def _page_memberships(receipts: dict) -> list[dict]:
    from . import synthesis_credit, whole_reference_pages as pages

    result = []
    for problem_id, receipt in receipts.items():
        rows = receipt.get("samples")
        if type(rows) is not dict:
            raise ValueError("update: missing original page score coverage")
        if not rows:
            continue
        by_bundle: dict[int, list[tuple[int, str, dict]]] = {}
        for sample_id, row in rows.items():
            binding = row.get("page_binding", {})
            if row.get("notebook_generation_policy") != pages.VERSION:
                raise ValueError("update: mixed page score membership")
            try:
                raw = binding["raw_pages"][row["page_index"]]
                sample = Sample(**raw, reward=row["strict_reward"])
                synthesis_credit.validate_score(row, sample)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError("update: invalid original page score membership") from exc
            if sample.sample_id != sample_id or sample.problem_id != problem_id:
                raise ValueError("update: original page score identity differs")
            by_bundle.setdefault(row["bundle_index"], []).append((row["page_index"], sample_id, binding))
        if sorted(by_bundle) != list(range(4)):
            raise ValueError("update: four original ordinal bundles required")
        for ordinal, entries in sorted(by_bundle.items()):
            entries.sort(key=lambda entry: entry[0])
            common = entries[0][2]
            ids = [entry[1] for entry in entries]
            if ([entry[0] for entry in entries] != list(range(len(common["page_plan"]["pages"])))
                    or any(entry[2] != common for entry in entries)
                    or ids != [raw["sample_id"] for raw in common["raw_pages"]]):
                raise ValueError("update: missing or mixed original ordinal page")
            result.append({"problem_id": problem_id, "bundle_index": ordinal, "sample_ids": ids})
    if len({sample_id for row in result for sample_id in row["sample_ids"]}) != sum(len(row["sample_ids"]) for row in result):
        raise ValueError("update: duplicate original page action")
    return result


def _page_mis_evidence(memberships, physical, counts, threshold, sums) -> tuple[dict, dict]:
    from .whole_reference_pages import VERSION as pages_version

    effective, bundles = dict(physical), []
    for membership in memberships:
        ids = membership["sample_ids"]
        if (any(sample_id not in physical or sample_id not in counts
                or type(counts[sample_id]) is not int or counts[sample_id] <= 0
                or type(physical[sample_id]) not in (int, float) or not math.isfinite(physical[sample_id])
                for sample_id in ids)):
            raise ValueError("update: missing original page action or MIS tokens")
        tokens = [counts[sample_id] for sample_id in ids]
        delta = float(torch.tensor(math.fsum(sums[sample_id] for sample_id in ids) / sum(tokens), dtype=torch.float32))
        for sample_id in ids:
            effective[sample_id] = delta
        bundles.append({**membership, "action_tokens": tokens, "mean_log_ratio": delta,
                        "keep": bool(torch.tensor(delta, dtype=torch.float32).abs() <= threshold)})
    evidence = {"version": pages_version, "delta_seq": threshold, "bundles": bundles}
    return effective, evidence | {"sha256": _hash(evidence)}


def validate_page_mis_evidence(update: dict, protocol: dict, trees=None) -> dict[str, float]:
    """Recompute shared ordinal MIS using all original masked page tokens."""
    from . import whole_reference_pages as pages

    physical = update.get("mean_log_ratio_by_sample", {})
    if protocol.get("notebook_generation_policy") != pages.VERSION:
        if "bundle_mis" in update or "effective_mean_log_ratio_by_sample" in update:
            raise ValueError("update: page MIS evidence requires its explicit policy")
        return physical
    if protocol.get("notebook_page_contract") != pages.contract():
        raise ValueError("update: page MIS contract differs")
    receipts = update.get("notebook_objective_receipts")
    if type(receipts) is not dict or not receipts:
        raise ValueError("update: original page objective receipts missing")
    if trees is not None:
        selected = {tree.problem_id: tree for tree in trees if tree.problem_id in update.get("routes", {})}
        if (set(selected) != set(receipts) or any(tree.notebook.get("learning_objective") != receipts[problem_id]
                for problem_id, tree in selected.items())):
            raise ValueError("update: page MIS source tree binding differs")
    memberships = _page_memberships(receipts)
    counts = update.get("action_token_counts_by_sample")
    sums = update.get("masked_log_ratio_sum_by_sample")
    if (type(physical) is not dict or type(counts) is not dict or type(sums) is not dict
            or set(physical) != set(counts) or set(sums) != set(counts)
            or any(type(count) is not int or count <= 0 for count in counts.values())
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in [*physical.values(), *sums.values()])
            or any(float(torch.tensor(sums[sample_id] / count, dtype=torch.float32))
                   != float(torch.tensor(physical[sample_id], dtype=torch.float32)) for sample_id, count in counts.items())):
        raise ValueError("update: original action MIS evidence differs")
    if any(counts.get(sample_id) != len(row["page_binding"]["raw_pages"][row["page_index"]]["completion_token_ids"])
           for receipt in receipts.values() for sample_id, row in receipt["samples"].items()):
        raise ValueError("update: original page MIS token count differs from raw action")
    effective, evidence = _page_mis_evidence(memberships, physical, counts, protocol["loss"]["mismatch"]["delta_seq"], sums)
    if update.get("bundle_mis") != evidence or update.get("effective_mean_log_ratio_by_sample") != effective:
        raise ValueError("update: ordinal bundle MIS evidence differs")
    return effective


def effective_mis_deltas(update: dict, protocol: dict, *, trees=None) -> dict[str, float]:
    return validate_page_mis_evidence(update, protocol, trees)


def partition_learning_trees(trees: list[ProblemTree], *, include_notebooks: bool = False,
                             notebook_objective: dict | None = None
                             ) -> tuple[list[ProblemTree], list[str], list[str]]:
    eligible, unresolved, signal_free = [], [], []
    for tree in trees:
        signal = has_learning_signal(tree, include_notebooks=include_notebooks,
                                     notebook_objective=notebook_objective)
        audit = tree.notebook.get("rewarded_notebook", {}).get("episode_audit", {})
        if audit.get("routing_rule") == STRICT_ROUTE_RULE and audit["admission"] == "UNRESOLVED":
            unresolved.append(tree.problem_id)
        elif signal:
            eligible.append(tree)
        else:
            signal_free.append(tree.problem_id)
    return eligible, unresolved, signal_free


def action_mask(sample: Sample, algorithm: str, tokenizer: Any) -> torch.Tensor:
    """Only this completion's tokens; notebook-only spans use actual token decoding."""
    mask = torch.ones((1, len(sample.completion_token_ids)), dtype=torch.bool)
    if algorithm != "ALG-NB-noteonly" or sample.turn == 1 or sample.kind == "DIAG":
        return mask
    mask.zero_()
    kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    decoded = tokenizer.decode(sample.completion_token_ids, **kwargs)
    if decoded != sample.text:
        raise ValueError("update: tokenizer and recorded completion text differ")
    starts = list(re.finditer(r"^Error check:[ \t]*\r?$", decoded, re.M))
    if len(starts) != 1:
        return mask
    start = starts[0].start()
    end = re.search(r"^(?:Observed facts|Solution):", decoded[starts[0].end():], re.M)
    if end is None:
        return mask  # An unclosed note is not a verified note/solution boundary.
    stop = starts[0].end() + end.start()
    previous = 0
    for index in range(len(sample.completion_token_ids)):
        prefix = tokenizer.decode(sample.completion_token_ids[:index+1], **kwargs)
        current = len(prefix)
        if current < previous:
            raise ValueError("update: nonmonotone decoding cannot identify note token spans")
        mask[0, index] = current > start and previous < stop
        previous = current
    return mask


def route_flags(trees: list[ProblemTree], arm: str, seed: int, step: int) -> dict[str, bool]:
    ids = [t.problem_id for t in trees]
    if len(ids) != len(set(ids)):
        raise ValueError("update: duplicate problems")
    for tree in trees:
        audit = tree.notebook.get("rewarded_notebook", {}).get("episode_audit", {})
        if audit.get("routing_rule") == STRICT_ROUTE_RULE or "admission" in audit:
            validate_update_binding(tree)
            if arm != "MOD-RV" or audit["admission"] == "UNRESOLVED":
                raise ValueError("update: strict admission requires resolved MOD-RV routing")
    if arm == "MOD-FML":
        return dict.fromkeys(ids, False)
    if arm == "MOD-FVML":
        return dict.fromkeys(ids, True)
    if arm == "MOD-RV":
        return {t.problem_id: t.route == "VISION_ON" for t in trees}
    if arm == "MOD-RR":
        count = sum(t.route == "VISION_ON" for t in trees)
        ranked = sorted(ids, key=lambda x: hashlib.sha256(f"route:{seed}:{step}:{x}".encode()).digest())
        selected = set(ranked[:count])
        return {x: x in selected for x in ids}
    raise ValueError("update: MOD-RS needs a separately frozen self-diagnosis selection policy")


def effective_rewards(groups: list[Group], algorithm: str, seed: int, step: int, *,
                      notebook_reward_enabled: bool = True,
                      notebook_scores: dict[str, dict] | None = None) -> dict[str, list[float]]:
    """Control rewards do not overwrite observed outcomes or raw trajectories."""
    if type(notebook_reward_enabled) is not bool:
        raise ValueError("update: explicit notebook reward control required")
    solution = [s for g in groups if g.kind != "DIAG" for s in g.samples]
    if not solution or any(s.result is None for s in solution):
        raise ValueError("update: solution samples require verifier results")
    actual_rate = sum(s.result.status == "CORRECT" for s in solution if s.result) / len(solution)
    result = {}
    for group in groups:
        values = []
        for sample in group.samples:
            if group.kind == "DIAG" and not notebook_reward_enabled:
                value = 0.0
            elif group.kind == "DIAG" and notebook_scores is not None:
                value = notebook_scores[sample.sample_id]["effective_reward"]
            elif algorithm == "CTL-RAND" and group.kind != "DIAG":
                key = hashlib.sha256(f"reward:{seed}:{step}:{sample.sample_id}".encode()).digest()
                value = float(random.Random(int.from_bytes(key, "big")).random() < actual_rate)
            elif algorithm == "CTL-FORMAT" and group.kind != "DIAG":
                value = float(extract_final(sample.text, sample.finish_reason).status == "OK")
            else:
                value = sample.reward
            values.append(value)
        result[group.group_id] = values
    return result


@dataclass
class _PreparedAction:
    sample: Sample
    request: RolloutRequest
    mask: torch.Tensor
    advantage: float
    weight: float
    norm: int
    vision_on: bool
    partial_credit: dict | None = None
    old: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    keep: bool = False


def _retained_learning_signal(actions: list[_PreparedAction], clip: dict) -> dict:
    """Policy coefficients at current=old; this is not measured model-gradient evidence."""
    signed = (clip.get("eps_low"), clip.get("eps_high")) == (.2, .28)
    retained, tokens, sample_ids = 0, 0, []
    for action in actions:
        if not action.keep:
            continue
        retained += 1
        assert action.old is not None
        dtype = torch.float32 if action.old.dtype in (torch.float16, torch.bfloat16) else action.old.dtype
        scale = torch.tensor(action.weight, dtype=dtype, device=action.old.device) / action.norm
        strict = torch.tensor(action.advantage, device=action.old.device).to(dtype)
        coefficient = strict * scale
        active = coefficient != 0
        if action.partial_credit is not None:
            partial = torch.tensor([action.partial_credit["partial_token_advantages"]],
                                   dtype=action.old.dtype, device=action.old.device).to(dtype)
            if partial.shape != action.old.shape:
                raise ValueError("update: retained partial policy coefficient shape differs")
            partial = partial * scale
            coefficient = coefficient + partial
            active = (coefficient != 0) if signed else active | (partial != 0)
        if not bool(torch.isfinite(coefficient).all()):
            raise ValueError("update: nonfinite retained policy coefficient")
        count = int((active & action.mask.to(action.old.device)).sum())
        if count:
            tokens += count
            sample_ids.append(action.sample.sample_id)
    basis = ("scaled_policy_coefficients_at_current_equals_old_on_retained_action_tokens" if signed else
             "potential_independent_surrogate_policy_coefficients_on_retained_action_tokens")
    return {"required": True, "basis": basis,
        "policy_gradient_present": bool(tokens), "retained_action_count": retained,
        "nonzero_policy_gradient_tokens": tokens, "signal_sample_ids": sample_ids}


def _stage_time(device: torch.device) -> float:
    # GPU launches are asynchronous; wall-time evidence must include finished work.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.monotonic()


def _gradient_identity(parameters) -> dict:
    """Distinguish an actual gradient contribution from allocating zero buffers."""
    from fine_tuning.modules import tensor_sha256
    return {group: [None if p.grad is None or not bool(torch.count_nonzero(p.grad))
                    else tensor_sha256(p.grad) for p in values]
            for group, values in parameters.items()}


def _relationship_gradient_norms(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor,
                                 old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor,
                                 loss_config: dict, partial: torch.Tensor, *,
                                 channel: str = "RELATIONSHIP", accumulator=None) -> dict[str, float]:
    """Measure one partial-credit VJP on the same forward without touching .grad."""
    loss = action.weight * group_loss(current, old, reference,
        torch.zeros(1, device=current.device, dtype=current.dtype), mask,
        torch.tensor([True], device=current.device), eps_low=loss_config["clip"]["eps_low"],
        eps_high=loss_config["clip"]["eps_high"], beta=0, norm_const=action.norm, partial_adv=partial)
    active = [(group, parameter) for group, values in bundle.params.items()
              for parameter in values if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in active],
                                    retain_graph=True, allow_unused=True)
    norms = dict.fromkeys(("L", "M", "V"), 0.0)
    for (group, parameter), gradient in zip(active, gradients, strict=True):
        if gradient is not None:
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"HEALTH_STOP:NONFINITE_{channel}_GRADIENT")
            norms[group] = math.hypot(norms[group], float(torch.linalg.vector_norm(
                gradient.detach(), dtype=torch.float64)))
            if accumulator is not None:
                accumulator.add(parameter, gradient)
    if any(not math.isfinite(value) for value in norms.values()):
        raise RuntimeError(f"HEALTH_STOP:NONFINITE_{channel}_GRADIENT")
    return norms


def _relationship_gradient(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor,
                           old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor,
                           loss_config: dict) -> dict:
    assert action.partial_credit is not None
    partial = torch.tensor([action.partial_credit["relationship_token_advantages"]],
                           device=current.device, dtype=current.dtype)
    norms = _relationship_gradient_norms(bundle, action, current, old, reference, mask, loss_config, partial)
    entries = action.partial_credit["relationships"]
    credit_types = list(dict.fromkeys(row["credit_type"] for row in entries
                                     if row["coefficient"] != 0 and row["token_indices"]))
    positive_types = list(dict.fromkeys(row["credit_type"] for row in entries
        if row["reward"] == 1 and row["advantage"] > 0 and row["token_indices"]))
    effect_scope = ("OUTPUT_FINALIZATION_ONLY" if credit_types == ["OUTPUT_FINALIZATION"] else
                    "NATIVE_ERROR_RELATIONSHIP_ONLY" if credit_types == ["NATIVE_ERROR_RELATIONSHIP"] else
                    "COMBINED_RELATIONSHIP_TYPES")
    return {"scope": "relationship_policy_loss_only", "measured": True, "finite": True,
        "credit_types": credit_types, "positive_credit_types": positive_types, "effect_scope": effect_scope,
        "basis": "nonzero_mapped_coefficients_not_per_type_gradient_decomposition",
        "nonzero_groups": [group for group, value in norms.items() if value > 0],
        "group_norms": norms, "positive_tokens": int(((partial > 0) & mask).sum()),
        "token_count": int(mask.sum()), "credit_sha256": _hash(action.partial_credit)}


def _native_relationship_vector(action: _PreparedAction, like: torch.Tensor,
                                mask: torch.Tensor) -> tuple[list[float], torch.Tensor, int]:
    from .synthesis_credit import NATIVE_ERROR_RELATIONSHIP

    assert action.partial_credit is not None
    coefficients = [0.] * len(action.sample.completion_token_ids)
    positive = [False] * len(coefficients)
    for entry in action.partial_credit["relationships"]:
        if entry["credit_type"] != NATIVE_ERROR_RELATIONSHIP:
            continue
        for index in entry["token_indices"]:
            coefficients[index] += entry["coefficient"]
            if entry["reward"] == 1 and entry["advantage"] > 0 and entry["coefficient"] > 0:
                positive[index] = True
    partial = torch.tensor([coefficients], device=like.device, dtype=like.dtype)
    positive_tokens = int(((partial > 0) & torch.tensor([positive], device=like.device) & mask).sum())
    return coefficients, partial, positive_tokens


def _retained_native_relationship_signal(actions: list[_PreparedAction]) -> dict:
    tokens, sample_ids = 0, []
    for action in actions:
        if not action.keep or action.sample.kind != "DIAG" or action.partial_credit is None:
            continue
        assert action.old is not None
        _, _, count = _native_relationship_vector(action, action.old, action.mask.to(action.old.device))
        if count:
            tokens += count
            sample_ids.append(action.sample.sample_id)
    return {"required": True, "basis": "positive_native_coefficients_after_dtype_conversion_and_mis",
        "policy_gradient_present": bool(tokens), "positive_native_relationship_tokens": tokens,
        "signal_sample_ids": sample_ids}


def _native_relationship_gradient_present(gradient: dict, credit: dict) -> bool:
    norms = gradient.get("group_norms", {})
    return (gradient.get("scope") == "native_error_relationship_policy_loss_only"
        and gradient.get("credit_type") == "NATIVE_ERROR_RELATIONSHIP"
        and gradient.get("measured") is True and gradient.get("finite") is True
        and gradient.get("positive_tokens", 0) > 0 and gradient.get("credit_sha256") == _hash(credit)
        and set(norms) == {"L", "M", "V"} and all(math.isfinite(value) and value >= 0 for value in norms.values())
        and any(value > 0 for value in norms.values()))


def _native_relationship_gradient(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor,
                                  old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor,
                                  loss_config: dict, relationship_gradient: dict | None = None) -> dict:
    """Separate native credit from OUTPUT, facts, strict reward and KL after MIS."""
    from .synthesis_credit import NATIVE_ERROR_RELATIONSHIP

    receipt = {"scope": "native_error_relationship_policy_loss_only", "measured": False}
    assert action.partial_credit is not None
    coefficients, partial, positive_tokens = _native_relationship_vector(action, current, mask)
    if not positive_tokens:
        return receipt
    if (relationship_gradient is not None and relationship_gradient.get("measured") is True
            and relationship_gradient.get("finite") is True
            and relationship_gradient.get("credit_sha256") == _hash(action.partial_credit)
            and coefficients == action.partial_credit["relationship_token_advantages"]):
        norms = relationship_gradient["group_norms"]
        basis = "same_forward_identical_native_only_vector"
    else:
        norms = _relationship_gradient_norms(bundle, action, current, old, reference, mask, loss_config, partial)
        basis = "same_forward_native_only_vector_vjp"
    return receipt | {"measured": True, "finite": True, "credit_type": NATIVE_ERROR_RELATIONSHIP,
        "basis": basis,
        "nonzero_groups": [group for group, value in norms.items() if value > 0], "group_norms": norms,
        "positive_tokens": positive_tokens, "token_count": int(((partial != 0) & mask).sum()),
        "credit_sha256": _hash(action.partial_credit), "token_advantages_sha256": _hash(coefficients)}


def _assessment_vector(action: _PreparedAction, like: torch.Tensor,
                       mask: torch.Tensor) -> tuple[list[float], torch.Tensor, dict[str, int]]:
    assert action.partial_credit is not None
    coefficients = [0.] * len(action.sample.completion_token_ids)
    positive = {flags: [False] * len(coefficients) for flags in ("N", "I", "U", "IU")}
    for entry in action.partial_credit["assessments"]:
        for index in entry["token_indices"]:
            coefficients[index] += entry["coefficient"]
            if entry["reward"] == 1 and entry["advantage"] > 0 and entry["coefficient"] > 0:
                positive[entry["expected_flags"]][index] = True
    partial = torch.tensor([coefficients], device=like.device, dtype=like.dtype)
    if partial.shape != like.shape or not bool(torch.isfinite(partial).all()):
        raise ValueError("update: assessment policy coefficients differ or are nonfinite")
    counts = {flags: int(((partial > 0) & mask & torch.tensor([tokens], device=like.device)).sum())
              for flags, tokens in positive.items()}
    return coefficients, partial, counts


def _retained_assessment_signal(actions: list[_PreparedAction]) -> dict:
    tokens, sample_ids = 0, []
    by_flags: dict[str, dict[str, Any]] = {flags: {"positive_assessment_tokens": 0, "signal_sample_ids": []}
                                        for flags in ("N", "I", "U", "IU")}
    for action in actions:
        if not action.keep or action.sample.kind != "DIAG" or action.partial_credit is None:
            continue
        assert action.old is not None
        _, _, counts = _assessment_vector(action, action.old, action.mask.to(action.old.device))
        if sum(counts.values()):
            tokens += sum(counts.values())
            sample_ids.append(action.sample.sample_id)
        for flags, count in counts.items():
            if count:
                by_flags[flags]["positive_assessment_tokens"] += count
                by_flags[flags]["signal_sample_ids"].append(action.sample.sample_id)
    return {"required": True, "basis": "positive_assessment_coefficients_after_dtype_conversion_and_mis",
        "policy_gradient_present": bool(tokens), "positive_assessment_tokens": tokens,
        "signal_sample_ids": sample_ids, "by_expected_flags": by_flags}


def _assessment_gradient_present(gradient: dict, credit: dict) -> bool:
    norms = gradient.get("group_norms", {})
    return (gradient.get("scope") == "native_assessment_policy_loss_only"
        and gradient.get("credit_type") == "NATIVE_ASSESSMENT"
        and gradient.get("measured") is True and gradient.get("finite") is True
        and gradient.get("positive_tokens", 0) > 0 and gradient.get("credit_sha256") == _hash(credit)
        and set(norms) == {"L", "M", "V"} and all(math.isfinite(value) and value >= 0 for value in norms.values())
        and any(value > 0 for value in norms.values()))


def _assessment_gradient(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor,
                         old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor,
                         loss_config: dict, accumulator=None) -> dict:
    """Exclude facts, relationships, strict reward and KL from assessment evidence."""
    receipt = {"scope": "native_assessment_policy_loss_only", "measured": False}
    if accumulator is not None and [id(parameter) for group in ("L", "M", "V")
            for parameter in bundle.params[group]] != [id(parameter) for _, _, parameter in accumulator.parameters]:
        raise ValueError("update: assessment parameter inventory changed")
    assert action.partial_credit is not None
    coefficients, partial, counts = _assessment_vector(action, current, mask)
    positive_tokens = sum(counts.values())
    if not bool(((partial != 0) & mask).any()):
        return receipt
    norms = _relationship_gradient_norms(bundle, action, current, old, reference, mask,
                                         loss_config, partial, channel="ASSESSMENT", accumulator=accumulator)
    return receipt | {"measured": True, "finite": True, "credit_type": "NATIVE_ASSESSMENT",
        "basis": "same_forward_assessment_only_vector_vjp",
        "nonzero_groups": [group for group, value in norms.items() if value > 0], "group_norms": norms,
        "positive_tokens": positive_tokens, "token_count": int(((partial != 0) & mask).sum()),
        "credit_sha256": _hash(action.partial_credit), "token_advantages_sha256": _hash(coefficients)}


class _AssessmentGradientAccumulator:
    """Sum signed VJPs by fixed parameter identity across changing Vision routes."""
    def __init__(self, bundle: PolicyBundle):
        names = {id(parameter): name for name, parameter in bundle.model.named_parameters()}
        self.parameters = [(group, index, parameter) for group in ("L", "M", "V")
                           for index, parameter in enumerate(bundle.params[group])]
        if len({id(parameter) for _, _, parameter in self.parameters}) != len(self.parameters):
            raise ValueError("update: duplicate assessment parameter identity")
        self.inventory = [{"group": group, "index": index, "name": names[id(parameter)],
            "shape": list(parameter.shape), "dtype": str(parameter.dtype), "numel": parameter.numel()}
            for group, index, parameter in self.parameters]
        self.sums: dict[int, torch.Tensor] = {}
        self.identities = {id(parameter) for _, _, parameter in self.parameters}

    def validate(self, bundle: PolicyBundle) -> None:
        current = _AssessmentGradientAccumulator(bundle)
        if (current.inventory != self.inventory
                or [id(p) for _, _, p in current.parameters] != [id(p) for _, _, p in self.parameters]):
            raise ValueError("update: assessment parameter inventory changed before optimizer")

    def add(self, parameter: torch.Tensor, gradient: torch.Tensor) -> None:
        identity = id(parameter)
        if identity not in self.identities:
            raise ValueError("update: foreign assessment parameter identity")
        if identity in self.sums:
            self.sums[identity].add_(gradient.detach().to(device="cpu", dtype=torch.float64))
        else:
            self.sums[identity] = gradient.detach().to(device="cpu", dtype=torch.float64, copy=True)

    def receipt(self, members: list[dict]) -> dict:
        norms = dict.fromkeys(("L", "M", "V"), 0.)
        for group, _, parameter in self.parameters:
            value = self.sums.get(id(parameter))
            if value is not None:
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError("HEALTH_STOP:NONFINITE_ASSESSMENT_GRADIENT")
                norms[group] = math.hypot(norms[group], float(torch.linalg.vector_norm(value)))
        if any(not math.isfinite(value) for value in norms.values()):
            raise RuntimeError("HEALTH_STOP:NONFINITE_ASSESSMENT_GRADIENT")
        return {"scope": "signed_all_retained_native_assessment_policy_loss_only",
            "basis": "same_forward_vjp_sum_by_parameter_identity_cpu_float64",
            "requirement": ASSESSMENT_REQUIREMENT, "finite": True, "group_norms": norms,
            "nonzero_groups": [group for group, value in norms.items() if value > 0],
            "positive_only": False, "benefit_verified": False,
            "parameter_inventory": self.inventory, "parameter_inventory_sha256": _hash(self.inventory),
            "members": members, "members_sha256": _hash(members)}


def _assessment_member(action: _PreparedAction) -> dict:
    assert action.old is not None and action.partial_credit is not None
    coefficients, partial, counts = _assessment_vector(action, action.old, action.mask.to(action.old.device))
    count = int(((partial != 0) & action.mask.to(action.old.device)).sum())
    credit = action.partial_credit
    return {"sample_id": action.sample.sample_id, "problem_id": action.sample.problem_id,
        "group_id": action.sample.group_id, "policy_version": action.sample.policy_version,
        "vision_on": action.vision_on, "retained": action.keep,
        "disposition": "MIS_EXCLUDED" if not action.keep else "VJP" if count else "ZERO_VECTOR",
        "full_group": credit["full_group"], "full_group_sha256": credit["full_group_sha256"],
        "credit_sha256": _hash(credit), "token_advantages_sha256": _hash(coefficients),
        "mask_sha256": _hash(action.mask.tolist()), "logprob_dtype": str(action.old.dtype),
        "token_count": count, "positive_tokens": sum(counts.values()),
        "weight": action.weight, "norm": action.norm}


def _repair_vectors(action: _PreparedAction, credit, like: torch.Tensor,
                    mask: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
    """Check a credit receipt's consistency; callers must freshly verify its semantics."""
    from dataclasses import asdict
    from . import compact_note
    from .repair_credit import ComponentCredit

    if type(credit) is not ComponentCredit or credit.component not in ("online", "final"):
        raise ValueError("repair gradient: explicit ComponentCredit required")
    size, cap = (2, 1024) if credit.component == "online" else (4, 8192)
    sample, binding = action.sample, credit.source_bindings
    if (credit.sample_id != sample.sample_id or type(credit.original_group_size) is not int
            or credit.original_group_size != size or type(credit.completion_cap) is not int
            or credit.completion_cap != cap or type(credit.fixed_units) is not int or credit.fixed_units < 0
            or sample.kind not in (("T2", "T3") if credit.component == "online" else ("DIAG",))
            or type(action.norm) is not int or action.norm != size * cap
            or type(action.weight) not in (int, float) or not math.isfinite(action.weight) or action.weight < 0
            or type(action.keep) is not bool or type(action.vision_on) is not bool
            or type(credit.coefficients) is not tuple or len(credit.coefficients) != len(sample.completion_token_ids)
            or not 0 < len(credit.coefficients) <= cap
            or like.shape != (1, len(credit.coefficients)) or not like.is_floating_point()
            or mask.dtype != torch.bool or mask.shape != like.shape or mask.device != like.device
            or not torch.equal(mask, action.mask.to(like.device))
            or action.old is None or action.old.dtype != like.dtype or action.old.shape != like.shape
            or type(credit.unit_rows) is not tuple or len(credit.unit_rows) != credit.fixed_units
            or credit.ownership_sha256 != _hash(credit.unit_rows) or type(binding) is not dict):
        raise ValueError("repair gradient: action, shape, cap, or ownership differs")
    members = binding.get("group_members")
    compact = binding.get("version") == compact_note.CREDIT_VERSION
    # A text-only route binds its positive notes to S_text; they never enter the S_note positive channel.
    text_route = binding.get("route_admission")
    if (binding.get("version") not in ("repair-group-credit-v1", compact_note.CREDIT_VERSION)
            or "route_admission" in binding and (not compact or text_route not in compact_note.TEXT_ONLY_ROUTES)
            or compact and (credit.component != "final" or credit.fixed_units != 1
                            or binding.get("note_contract") != compact_note.VERSION
                            or type(binding.get("audit_sha256")) is not str)
            or binding.get("sample_sha256") != _hash(json.loads(to_json(sample)))
            or binding.get("group_id") != sample.group_id or binding.get("problem_id") != sample.problem_id
            or binding.get("group_kind") != sample.kind or type(members) is not list or len(members) != size
            or any(type(value) is not str or not value for value in members) or len(set(members)) != size
            or type(binding.get("sample_index")) is not int or not 0 <= binding["sample_index"] < size
            or members[binding["sample_index"]] != sample.sample_id
            or binding.get("group_members_sha256") != _hash(members)
            or any(type(binding.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None
                   for key in ("group_sha256", "grades_sha256" if compact else "inventory_sha256", "sample_sha256"))
            or binding.get("audit_sha256") is not None and (type(binding["audit_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", binding["audit_sha256"]) is None)
            or binding.get("token_text_correspondence_verified") is not True
            or any(binding.get(key) is not False for key in
                   ("semantic_truth_reverified", "engine_attachment_verified", "optimizer_admission"))):
        raise ValueError("repair gradient: source or original group membership differs")
    vectors = {key: [0.] * len(credit.coefficients) for key in ("all", "signed", "positive", "text")}
    refs: set[str] = set()
    occupied: set[int] = set()
    if compact:
        # One whole-note reward row; every credited token is the final channel, verified notes are positive.
        row = credit.unit_rows[0]
        indices = row.get("token_indices") if type(row) is dict else None
        counts_ok = type(row) is dict and all(type(row.get(key)) is int for key in
                                              ("verified_entries", "refuted_entries", "vision_score"))
        if (not counts_ok or set(row) != compact_note.ROW_KEYS or row["origin_ref"] != sample.sample_id + ":note"
                or row["kind"] != "NOTE" or type(indices) is not list
                or any(type(i) is not int or not 0 <= i < len(credit.coefficients) for i in indices)
                or indices != sorted(set(indices))
                or any(type(row[key]) not in (int, float) or not math.isfinite(row[key])
                       for key in ("reward", "mean", "advantage", "coefficient"))
                or row["verified_entries"] < 0 or row["refuted_entries"] < 0
                or row["verified_entries"] + row["refuted_entries"] > compact_note.MAX_ERRORS
                or row["vision_score"] not in (-1, 0, 1)
                or row["reward"] != compact_note.note_reward(row["verified_entries"], row["refuted_entries"],
                                                             row["vision_score"])
                or not -1 <= row["mean"] <= 1 or row["advantage"] != row["reward"] - row["mean"]
                or row["coefficient"] != compact_note.coefficient(row["advantage"], len(indices), cap)
                or row["positive_verified_note"] is not (row["advantage"] > 0 and (
                    row["verified_entries"] > 0 or row["vision_score"] == 1))
                or text_route is not None and row["vision_score"] != 0):
            raise ValueError("repair gradient: compact note provenance or arithmetic differs")
        positive_key = "positive" if text_route is None else "text"
        for index in indices:
            for key in ("all", "signed", positive_key) if row["positive_verified_note"] else ("all", "signed"):
                vectors[key][index] += row["coefficient"]
    for row in credit.unit_rows if not compact else ():
        if type(row) is not dict:
            raise ValueError("repair gradient: unit provenance shape differs")
        indices = row.get("token_indices")
        if (type(row.get("origin_ref")) is not str or not row["origin_ref"] or row["origin_ref"] in refs
                or row.get("kind") not in ("OBSERVATION", "TRACE", "OUTPUT")
                or row.get("relation") not in (None, "CORRECTED", "PERSISTED", "OTHER_BRANCH_COMPARISON", "UNKNOWN", "FINALIZED")
                or row.get("verdict") not in (None, "VERIFIED", "REFUTED", "UNVERIFIABLE", "STRUCTURAL_INVALID")
                or type(indices) is not list or any(type(i) is not int or not 0 <= i < len(credit.coefficients) for i in indices)
                or indices != sorted(set(indices)) or occupied.intersection(indices)
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                       for key in ("mean", "advantage", "coefficient"))
                or any(type(row.get(key)) is not int or row[key] not in (-1, 0, 1)
                       for key in ("semantic_score", "applied_score"))
                or not -1 <= row["mean"] <= 1 or row["advantage"] != row["applied_score"] - row["mean"]):
            raise ValueError("repair gradient: unit provenance or arithmetic differs")
        native = row["kind"] in ("OBSERVATION", "TRACE") and row["relation"] == "CORRECTED"
        positive = native and row["verdict"] == "VERIFIED" and row["applied_score"] == 1 and row["advantage"] > 0
        coefficient = cap * .0625 * row["advantage"] / (credit.fixed_units * len(indices)) if indices else 0.
        if (row.get("corrected_channel") is not native or row.get("positive_verified_corrected") is not positive
                or row["coefficient"] != coefficient
                or row["kind"] == "OUTPUT" and row["applied_score"] != 0
                or binding.get("audit_sha256") is None and (row["relation"] is not None or row["applied_score"] != 0)):
            raise ValueError("repair gradient: corrected semantic provenance differs")
        refs.add(row["origin_ref"])
        occupied.update(indices)
        for index in indices:
            vectors["all"][index] += coefficient
            if native:
                vectors["signed"][index] += coefficient
            if positive:
                vectors["positive"][index] += coefficient
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in credit.coefficients)
            or tuple(vectors["all"]) != credit.coefficients):
        raise ValueError("repair gradient: original token coefficients differ")
    partials = {key: torch.tensor([vectors[key]], device=like.device, dtype=like.dtype)
                for key in ("signed", "positive") + (("text",) if text_route is not None else ())}
    if any(not bool(torch.isfinite(value).all()) for value in partials.values()):
        raise ValueError("repair gradient: nonfinite converted policy coefficient")
    counts = {key: int(((value != 0) & mask).sum()) for key, value in partials.items()}
    member = {"sample_id": sample.sample_id, "problem_id": sample.problem_id, "group_id": sample.group_id,
        "policy_version": sample.policy_version, "component": credit.component, "vision_on": action.vision_on,
        "retained": action.keep, "disposition": "MIS_EXCLUDED" if not action.keep else
            "VJP" if counts["signed"] else "ZERO_VECTOR",
        "credit_sha256": _hash(asdict(credit)), "source_bindings": dict(binding),
        "signed_token_advantages_sha256": _hash(vectors["signed"]),
        "positive_token_advantages_sha256": _hash(vectors["positive"]),
        "mask_sha256": _hash(mask.tolist()), "old_logprobs_sha256": _hash(action.old.tolist()),
        "logprob_dtype": str(like.dtype),
        "signed_tokens": counts["signed"] if action.keep else 0,
        "positive_verified_tokens": counts["positive"] if action.keep else 0,
        "weight": action.weight, "norm": action.norm}
    if text_route is not None:
        member.update(text_token_advantages_sha256=_hash(vectors["text"]),
                      text_verified_tokens=counts["text"] if action.keep else 0)
    return partials, member


class _RepairGradientAccumulator:
    """Four signed tensor sums; combined norms use one temporary parameter sum.

    An opt-in text-only route adds one separate S_text sum; it never contributes to S."""
    def __init__(self, bundle: PolicyBundle, s_definition: str = "native-corrected-v1",
                 text_definition: str | None = None, *, measured: bool = True):
        if type(measured) is not bool:
            raise ValueError("repair gradient: explicit measured/unmeasured mode required")
        self.measured = measured
        self.s_definition = s_definition
        self.text_definition = text_definition
        self.text = _AssessmentGradientAccumulator(bundle) if text_definition is not None else None
        self.channels = {component: {kind: _AssessmentGradientAccumulator(bundle)
            for kind in ("signed", "positive")} for component in ("online", "final")}
        self.members: list[dict] = []
        self.sources: list[tuple[_PreparedAction, Any, str]] = []
        self.failed = False

    def validate(self, bundle: PolicyBundle) -> None:
        if self.failed:
            raise RuntimeError("repair gradient: accumulator aborted")
        for channel in self.channels.values():
            for accumulator in channel.values():
                accumulator.validate(bundle)
        if self.text is not None:
            self.text.validate(bundle)

    def clear(self) -> None:
        for accumulator in [*(item for channel in self.channels.values() for item in channel.values()),
                            *((self.text,) if self.text is not None else ())]:
            accumulator.sums.clear()

    def receipt(self, bundle: PolicyBundle) -> dict:
        self.validate(bundle)
        for action, credit, seal in self.sources:
            assert action.old is not None
            _, member = _repair_vectors(action, credit, action.old, action.mask.to(action.old.device))
            if _hash(member) != seal:
                raise ValueError("repair gradient: measured source or provenance changed")
        if [_hash(member) for member in self.members] != [seal for _, _, seal in self.sources]:
            raise ValueError("repair gradient: measured membership changed")
        if not self.measured:
            return self._unmeasured()
        channels = {component: {kind: accumulator.receipt([])["group_norms"]
            for kind, accumulator in channel.items()} for component, channel in self.channels.items()}
        combined = {kind: dict.fromkeys(("L", "M", "V"), 0.) for kind in ("signed", "positive")}
        for kind, norms in combined.items():
            online, final = self.channels["online"][kind], self.channels["final"][kind]
            for group, _, parameter in online.parameters:
                left, right = online.sums.get(id(parameter)), final.sums.get(id(parameter))
                value = right if left is None else left if right is None else left + right
                if value is not None:
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError("HEALTH_STOP:NONFINITE_REPAIR_GRADIENT")
                    norms[group] = math.hypot(norms[group], float(torch.linalg.vector_norm(value)))
        if any(not math.isfinite(value) for norms in combined.values() for value in norms.values()):
            raise RuntimeError("HEALTH_STOP:NONFINITE_REPAIR_GRADIENT")
        positive_tokens = sum(member["positive_verified_tokens"] for member in self.members)
        result = {"scope": "native_corrected_repair_policy_loss_only",
            "basis": "same_forward_vjp_sum_by_parameter_identity_cpu_float64",
            "finite": True, "benefit_verified": False,
            "channels": channels, "combined": combined, "retained_positive_verified_tokens": positive_tokens,
            "S": bool(positive_tokens and all(any(value > 0 for value in norms.values()) for norms in combined.values())),
            "s_definition": self.s_definition,
            "parameter_inventory": self.channels["online"]["signed"].inventory,
            "members": self.members}
        if self.text is not None:
            text = self.text.receipt([])["group_norms"]
            tokens = sum(member.get("text_verified_tokens", 0) for member in self.members)
            result["S_text"] = {"s_definition": self.text_definition, "scope": "text_only_route_note_policy_loss_only",
                "group_norms": text, "retained_text_verified_tokens": tokens,
                "S": bool(tokens and any(value > 0 for value in text.values())), "counts_as_S": False}
        return result | {"parameter_inventory_sha256": _hash(result["parameter_inventory"]),
                         "members_sha256": _hash(self.members)}


    def _unmeasured(self) -> dict:
        """Non-audit update under a fixed audit rule: original members only, no VJP, S unknown (None)."""
        result: dict[str, Any] = {"scope": "native_corrected_repair_policy_loss_only",
            "basis": "not_measured_fixed_audit_rule", "measured": False, "benefit_verified": False,
            "retained_positive_verified_tokens": sum(member["positive_verified_tokens"] for member in self.members),
            "S": None, "s_definition": self.s_definition,
            "parameter_inventory": self.channels["online"]["signed"].inventory, "members": self.members}
        if self.text is not None:
            result["S_text"] = {"s_definition": self.text_definition, "scope": "text_only_route_note_policy_loss_only",
                "measured": False, "retained_text_verified_tokens": sum(member.get("text_verified_tokens", 0)
                                                                         for member in self.members),
                "S": None, "counts_as_S": False}
        return result | {"parameter_inventory_sha256": _hash(result["parameter_inventory"]),
                         "members_sha256": _hash(self.members)}

    def record(self, action: _PreparedAction, credit) -> dict:
        """Unmeasured mode: bind the original member exactly as _repair_gradient would, without a VJP."""
        if self.measured or self.failed or action.old is None:
            raise ValueError("repair gradient: member-only recording requires the unmeasured mode and original logprobs")
        _, member = _repair_vectors(action, credit, action.old, action.mask.to(action.old.device))
        if any(row["component"] == credit.component and row["sample_id"] == credit.sample_id for row in self.members):
            raise ValueError("repair gradient: original logprobs, clipping, or member differs")
        self.members.append(member)
        self.sources.append((action, credit, _hash(member)))
        return member


def _repair_gradient(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor,
                     old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor,
                     loss_config: dict, credit, accumulator: _RepairGradientAccumulator) -> dict:
    """Measure signed CORRECTED and verified-positive CORRECTED on the caller's forward."""
    accumulator.validate(bundle)
    partials, member = _repair_vectors(action, credit, current, mask)
    assert action.old is not None
    if (old.dtype != current.dtype or not torch.equal(old, action.old.to(old.device)) or current.shape != old.shape
            or current.shape != reference.shape or loss_config["clip"] != {"eps_low": .2, "eps_high": .28}
            or any(row["component"] == credit.component and row["sample_id"] == credit.sample_id
                   for row in accumulator.members)):
        raise ValueError("repair gradient: original logprobs, clipping, or member differs")
    if action.keep and (getattr(bundle.route, "vision_on", None) is not action.vision_on
            or any(parameter.requires_grad != action.vision_on for parameter in bundle.params["V"])):
        raise ValueError("repair gradient: active Vision route differs")
    try:
        if action.keep:
            for kind, partial in partials.items():
                if kind == "text" and accumulator.text is None:
                    raise ValueError("repair gradient: text-only route credit without its bound S_text definition")
                if bool(((partial != 0) & mask).any()):
                    _relationship_gradient_norms(bundle, action, current, old, reference, mask,
                        loss_config, partial, channel="REPAIR", accumulator=accumulator.text if kind == "text"
                        else accumulator.channels[credit.component][kind])
        accumulator.validate(bundle)
        if _hash(_repair_vectors(action, credit, current, mask)[1]) != _hash(member):
            raise ValueError("repair gradient: source or provenance changed during VJP")
    except Exception:
        accumulator.failed = True
        raise
    accumulator.members.append(member)
    accumulator.sources.append((action, credit, _hash(member)))
    return member


def _repair_update_snapshot(trees, actors, protocol, images, preparation, receipts, actions) -> str:
    from PIL.Image import Image

    return _hash({"trees": [json.loads(to_json(tree)) for tree in trees],
        "actors": {key: json.loads(to_json(actor)) for key, actor in actors.items()},
        "protocol": protocol, "receipts": receipts,
        "preparation": [id(preparation), preparation.policy_version, id(preparation.tokenizer),
            *[id(getattr(getattr(preparation, name), "__func__", getattr(preparation, name)))
              for name in ("prepare", "model_inputs")]],
        "images": {key: [(id(value), value.mode, value.size, hashlib.sha256(value.tobytes()).hexdigest())
                          if isinstance(value, Image) else (id(value),) for value in values]
                   for key, values in images.items()},
        "actions": [{"sample_id": row.sample.sample_id, "mask": row.mask.tolist(),
            "advantage": row.advantage, "weight": row.weight, "norm": row.norm, "vision_on": row.vision_on,
            "request": {key: value for key, value in vars(row.request).items() if key != "images"},
            "images": [id(value) for value in row.request.images]} for row in actions]})


def _prepare_repair_update(trees, actors, verifier, protocol, images, preparation, *,
                           seed: int, notebook_reward_enabled: bool) -> dict:
    from .episode_notebook import ANSWER_ROUTE_RULE, ANSWER_VERIFIED, wire_route
    from .local_profile import REPAIR_PROTOCOL_ID, local_prompts, restore_protocol
    from .repair_credit import ComponentCredit
    from .repair_training import prepare_tree_credit

    restore_protocol(protocol)
    if (protocol["protocol_id"] != REPAIR_PROTOCOL_ID or actors is None or set(actors) != set(images)
            or verifier is None or type(notebook_reward_enabled) is not bool):
        raise ValueError("repair update: explicit canonical profile, verifier, and actors required")
    if any(_hash(tree.notebook.get("rewarded_notebook", {}).get("source_prompts")) != _hash(local_prompts(protocol))
           for tree in trees):
        raise ValueError("repair update: original source prompts differ from canonical protocol")
    before = _repair_update_snapshot(trees, actors, protocol, images, preparation, {}, [])
    receipts = {tree.problem_id: prepare_tree_credit(tree, actors[tree.problem_id], verifier,
        preparation.tokenizer, preparation=preparation, images=images[tree.problem_id], master_seed=seed,
        notebook_reward_enabled=notebook_reward_enabled) for tree in trees}
    if before != _repair_update_snapshot(trees, actors, protocol, images, preparation, {}, []):
        raise ValueError("repair update: original source mutated during preparation")
    rows, groups, credits = {}, {}, {}
    for tree in trees:
        receipt = receipts[tree.problem_id]
        if (wire_route(receipt["route"]) != tree.route or receipt["policy_version"] != tree.policy_version
                or receipt["route"] == ANSWER_VERIFIED and protocol.get("repair_route_policy") != ANSWER_ROUTE_RULE):
            raise ValueError("repair update: independent routing or policy differs")
        for group in receipt["groups"]:
            if group["group_id"] in groups or group["group_weight"] != protocol["loss"]["group_weights"][group["kind"]]:
                raise ValueError("repair update: duplicate group or fixed group weight differs")
            groups[group["group_id"]] = group
            for row in group["samples"]:
                if row["sample_id"] in rows:
                    raise ValueError("repair update: duplicate original sample identity")
                rows[row["sample_id"]] = row
                wire = row["component_credit"]
                if wire is not None:
                    credits[row["sample_id"]] = ComponentCredit(**(wire | {
                        "coefficients": tuple(wire["coefficients"]), "unit_rows": tuple(wire["unit_rows"])}))
    return {"receipts": receipts, "rows": rows, "groups": groups, "credits": credits}


def _repair_channels(row: dict) -> tuple[str, ...]:
    """Loss channels of one prepared row; finalization exists only when its opt-in protocol bound it."""
    return ("answer", "online", "final") + (("finalization",) if "applied_finalization_coefficients" in row else ())


def _retained_repair_signal(actions: list[_PreparedAction], rows: dict) -> dict:
    count, sample_ids = 0, []
    finalization = any("applied_finalization_coefficients" in row for row in rows.values())
    channel_tokens = dict.fromkeys(("answer", "online", "final", "finalization"), 0)
    for action in actions:
        if not action.keep:
            continue
        assert action.old is not None
        row, old = rows[action.sample.sample_id], action.old
        channels = _repair_channels(row)
        if ("finalization" in channels) is not finalization:
            raise ValueError("repair update: mixed finalization channel membership")
        values = torch.tensor([row[f"applied_{key}_coefficients"] for key in channels],
                              dtype=old.dtype, device=old.device)
        values = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
        combined = values.sum(dim=0) * (action.weight / action.norm)
        if not bool(torch.isfinite(combined).all()):
            raise ValueError("repair update: nonfinite retained policy coefficient")
        mask = action.mask.to(old.device)[0]
        tokens = int(((combined != 0) & mask).sum())
        for key, vector in zip(channels, values, strict=True):
            channel_tokens[key] += int(((vector != 0) & mask).sum())
        if tokens:
            count += tokens
            sample_ids.append(action.sample.sample_id)
    result = {"required": True, "basis": "combined_answer_online_final_policy_coefficients_at_current_equals_old_after_dtype_mask_mis",
        "policy_gradient_present": bool(count), "nonzero_policy_gradient_tokens": count,
        "signal_sample_ids": sample_ids, "parameter_gradient_cancellation_verified": False}
    if finalization:
        # F evidence names its channels, so a finalization-only gradient is never read as note learning.
        others = any(channel_tokens[key] for key in ("answer", "online", "final"))
        result.update(basis="combined_answer_online_final_finalization_policy_coefficients_at_current_equals_old_after_dtype_mask_mis",
                      channel_tokens=channel_tokens,
                      channel_signal={key: bool(value) for key, value in channel_tokens.items()},
                      finalization_only_signal=bool(channel_tokens["finalization"]) and not others)
    return result


def _repair_action_journal(actions: list[_PreparedAction], rows: dict) -> list[dict]:
    result = []
    for action in actions:
        assert action.old is not None
        sample, credit = action.sample, rows[action.sample.sample_id]["component_credit"]
        result.append({"sample_id": sample.sample_id, "problem_id": sample.problem_id,
            "group_id": sample.group_id, "kind": sample.kind, "policy_version": sample.policy_version,
            "retained": action.keep, "vision_on": action.vision_on, "logprob_dtype": str(action.old.dtype),
            "action_tokens": len(sample.completion_token_ids), "mask_sha256": _hash(action.mask.tolist()),
            "weight": action.weight, "norm": action.norm, "old_logprobs": action.old[0].tolist(),
            "old_logprobs_sha256": _hash(action.old.tolist()),
            "sample_sha256": _hash(json.loads(to_json(sample))),
            "raw_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest(),
            "token_ids_sha256": _hash(sample.completion_token_ids),
            "sampler_logprobs_sha256": _hash(sample.sampler_logprobs),
            "component_credit_sha256": _hash(credit) if credit is not None else None})
    return result


def optimizer_step(bundle: PolicyBundle, optimizer: Any, scheduler: Any,
                   trees: list[ProblemTree], images: dict[str, list], preparation: Any,
                   protocol: dict, algorithm: str, *, step: int, seed: int,
                   vision_disabled: bool = False, notebook_reward_enabled: bool = True,
                   require_retained_learning_signal: bool = False,
                   observation_actors: dict[str, ActorItem] | None = None,
                   objective_verifier: Any | None = None,
                   on_progress: Callable[[dict[str, Any]], None] | None = None,
                   before_optimizer_step: Callable[[], None] | None = None,
                   measure_evidence_vjp: bool = True) -> dict:
    """Compute all old/reference LPs first, then microbatch backward and ONE update.

    `measure_evidence_vjp=False` (opt-in fixed audit rule, repair only) records the original
    VJP members but skips the evidence VJPs; the aggregate is then marked measured: false.
    The policy loss, backward pass and optimizer step are identical either way.

    Returns unsynchronized evidence. It is deliberately not a metrics.step_record:
    only the controller can attest a subsequent weight push and probe comparison.
    A forward/backward failure clears partial gradients and never calls optimizer.step.
    Progress observes host calls, not GPU completion; callbacks run only before commit.
    """
    if on_progress is not None and not callable(on_progress):
        raise ValueError("update: progress observer must be callable")
    if before_optimizer_step is not None and not callable(before_optimizer_step):
        raise ValueError("update: pre-step hook must be callable")
    if type(require_retained_learning_signal) is not bool:
        raise ValueError("update: retained learning-signal requirement must be boolean")
    if type(measure_evidence_vjp) is not bool:
        raise ValueError("update: explicit evidence VJP audit flag required")
    from .local_profile import REPAIR_PROTOCOL_ID
    from .repair_notebook import VERSIONS as repair_versions

    repair_selected = (protocol.get("protocol_id") == REPAIR_PROTOCOL_ID or "repair_objective" in protocol
                       or protocol.get("rollout", {}).get("retry_contract") is not None)
    if not measure_evidence_vjp and not repair_selected:
        raise ValueError("update: skipping evidence VJPs is defined only for the repair audit rule")
    if not repair_selected and any(tree.notebook.get("rewarded_notebook", {}).get("notebook_generation_policy")
                                   in repair_versions for tree in trees):
        raise ValueError("repair update: repair trees require the explicit repair protocol")
    if repair_selected and (algorithm != "ALG-NB3" or bundle.inventory["arm"] != "MOD-RV"
            or type(seed) is not int or seed != 3407 or vision_disabled is not False):
        raise ValueError("repair update: exact ALG-NB3 MOD-RV seed3407 route contract required")
    requirement = protocol.get("notebook_update_requirement")
    if "notebook_update_requirement" in protocol and (type(requirement) is not str
            or requirement not in (NATIVE_RELATIONSHIP_REQUIREMENT, ASSESSMENT_REQUIREMENT)):
        raise ValueError("update: unknown notebook update requirement")
    native_requirement_selected = requirement == NATIVE_RELATIONSHIP_REQUIREMENT
    assessment_requirement_selected = requirement == ASSESSMENT_REQUIREMENT
    from .pair_action_notebook import VERSION as pair_action_version

    require_retained_learning_signal |= protocol.get("notebook_generation_policy") == pair_action_version
    if (algorithm not in protocol["algorithms"] or protocol["loss"]["updates_per_batch"] != 1
            or not trees or type(step) is not int or step < 0 or scheduler.last_epoch != step
            or preparation.policy_version != step or any(t.policy_version != step for t in trees)):
        raise ValueError("update: fresh same-policy batch and one-update contract required")
    if set(images) != {t.problem_id for t in trees}:
        raise ValueError("update: exact image coverage required")
    repair = (_prepare_repair_update(trees, observation_actors, objective_verifier, protocol, images, preparation,
        seed=seed, notebook_reward_enabled=notebook_reward_enabled) if repair_selected else None)
    if repair is not None and any(not row["admitted"] for row in repair["receipts"].values()):
        return {"status": "NO_ADMITTED_REPAIR_TREES", "step_before": step, "step_after": step,
            "engine_sync_required": False, "repair_preparation": repair["receipts"], "repair_learning_verified": False}
    if repair is not None and not any(row["has_signal"] for row in repair["receipts"].values()):
        return {"status": "NO_LEARNING_SIGNAL", "step_before": step, "step_after": step,
            "engine_sync_required": False, "repair_preparation": repair["receipts"], "repair_learning_verified": False}
    from . import notebook_synthesis as synthesis
    from .episode_notebook import regrade_synthesis
    from .notebook_objective import SYNTHESIS_CREDIT_VERSION
    from .synthesis_credit import ASSESSMENT_VERSION
    from .whole_reference_notebook import VERSION as whole_reference_version

    from . import whole_reference_pages as pages

    paged = protocol.get("notebook_generation_policy") == pages.VERSION
    if paged and protocol.get("notebook_page_contract") != pages.contract():
        raise ValueError("update: explicit original page contract required")
    objective = protocol.get("notebook_objective")
    synthesis_credit_selected = (isinstance(objective, dict)
        and objective.get("objective_version") in (SYNTHESIS_CREDIT_VERSION, ASSESSMENT_VERSION))
    assessment_credit_selected = (isinstance(objective, dict)
                                 and objective.get("objective_version") == ASSESSMENT_VERSION)
    synthesis_contract = protocol.get("notebook_synthesis_contract")
    synthesis_selected = synthesis_contract is not None or protocol.get("notebook_reward") == synthesis.BOUNDED_VERSION
    if synthesis_selected and (synthesis_contract != synthesis.BOUNDED_VERSION
            or protocol.get("notebook_reward") != synthesis_contract
            or objective is not None and not synthesis_credit_selected
            or objective_verifier is None or observation_actors is None
            or set(observation_actors) != set(images)):
        raise ValueError("update: bounded synthesis requires its explicit contract and fresh verifier/actors")
    if synthesis_credit_selected and not synthesis_selected:
        raise ValueError("update: synthesis unit credit requires the explicit bounded synthesis contract")
    strict = protocol.get("notebook_route_rule") == STRICT_ROUTE_RULE
    if native_requirement_selected and (not strict or not synthesis_credit_selected or not synthesis_selected
            or assessment_credit_selected or protocol.get("profile") != "local-nf4-notebook-rl-v1"):
        raise ValueError("update: native relationship requirement needs strict local synthesis unit credit")
    if assessment_credit_selected != assessment_requirement_selected or assessment_requirement_selected and (
            not strict or not synthesis_selected or protocol.get("profile") != "local-nf4-notebook-rl-v1"
            or protocol.get("notebook_generation_policy") != whole_reference_version):
        raise ValueError("update: assessment requirement needs explicit strict local whole-reference assessment credit")
    if assessment_requirement_selected and (protocol["rollout"]["diag"]["max_new_tokens"] != 8192
            or any(len(group.samples) != 4 for tree in trees for group in tree.groups
                   if group.kind == "DIAG" and group.usable)):
        raise ValueError("update: assessment optimizer requires original n4 and cap8192")
    audits = [t.notebook.get("rewarded_notebook", {}).get("episode_audit", {}) for t in trees]
    strict_trees = any(audit.get("routing_rule") == STRICT_ROUTE_RULE or "admission" in audit for audit in audits)
    if strict_trees and not strict:
        raise ValueError("update: strict tree/protocol routing admission differs")
    if strict:
        if (bundle.inventory["arm"] != "MOD-RV" or vision_disabled is not False
                or protocol.get("notebook_reward") not in (EPISODE_REWARD_VERSION, synthesis.BOUNDED_VERSION)):
            raise ValueError("update: strict admission requires MOD-RV without vision health override")
        for tree in trees:
            audit = tree.notebook.get("rewarded_notebook", {}).get("episode_audit", {})
            if audit.get("routing_rule") != STRICT_ROUTE_RULE:
                raise ValueError("update: strict admission binding missing")
    if (type(notebook_reward_enabled) is not bool or not notebook_reward_enabled
            and protocol.get("profile") != "local-nf4-notebook-rl-v1"):
        raise ValueError("update: notebook reward control requires an explicit local profile")
    include_notebooks = (notebook_reward_enabled
                        and protocol.get("profile") == "local-nf4-notebook-rl-v1"
                        and protocol.get("notebook_reward") in (NOTEBOOK_REWARD_VERSION, EPISODE_REWARD_VERSION,
                                                                 synthesis.BOUNDED_VERSION))
    require_native_relationship = native_requirement_selected and notebook_reward_enabled
    require_assessment = assessment_requirement_selected and notebook_reward_enabled
    if protocol.get("notebook_reward") in (EPISODE_REWARD_VERSION, synthesis.BOUNDED_VERSION):
        observation_prompts = None
        if protocol.get("notebook_observation_policy") is not None or synthesis_selected:
            from .local_profile import local_prompts
            from .prompts import for_actor
            if observation_actors is None or set(observation_actors) != set(images):
                raise ValueError("update: approved observation actors required")
            observation_prompts = local_prompts(protocol)
        for tree in trees:
            if tree.notebook.get("rewarded_notebook", {}).get("reward_version") != protocol["notebook_reward"]:
                raise ValueError("update: full-episode notebook contract missing")
            actor = observation_actors[tree.problem_id] if observation_prompts is not None and observation_actors else None
            schema = (for_actor(actor, observation_prompts)["observation_instruction"].get(actor.family)
                      if actor and observation_prompts is not None else None)
            validate_update_binding(tree, tokenizer=preparation.tokenizer,
                                    generation_policy=protocol.get("notebook_generation_policy"),
                                    draft_policy=protocol.get("notebook_draft_policy"),
                                    feedback_policy=protocol.get("notebook_feedback_policy"),
                                    observation_policy=protocol.get("notebook_observation_policy"),
                                    preparation=preparation, images=images[tree.problem_id],
                                    observation_actor=actor, observation_schema=schema, observation_master_seed=seed)
    def synthesis_receipts() -> dict:
        if not synthesis_selected:
            return {}
        assert observation_actors is not None
        return {tree.problem_id: regrade_synthesis(tree, observation_actors[tree.problem_id],
                    objective_verifier, preparation.tokenizer) for tree in trees}

    regraded_synthesis = synthesis_receipts()
    notebook_scores: dict[str, dict] | None = None
    if objective is not None:
        if (not strict or not include_notebooks and notebook_reward_enabled
                or objective_verifier is None or observation_actors is None
                or set(observation_actors) != set(images)):
            raise ValueError("update: atomic notebook objective requires fresh verifier and approved actors")
        notebook_scores = {}
        for tree in trees:
            for group in tree.groups:
                if group.kind == "DIAG":
                    for sample in group.samples:
                        decoded = preparation.tokenizer.decode(sample.completion_token_ids,
                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        if decoded != sample.text:
                            raise ValueError("update: notebook sampled tokens and rewarded text differ")
            recorded = notebook_objective_scores(tree, objective)
            fresh = _score_notebook_objective(tree, observation_actors[tree.problem_id], objective_verifier, objective)
            if fresh != tree.notebook["learning_objective"]:
                raise ValueError("update: freshly regraded notebook objective differs")
            if notebook_scores.keys() & recorded.keys():
                raise ValueError("update: duplicate objective sample IDs")
            notebook_scores.update(recorded)
    if not repair_selected and any(not has_learning_signal(t, include_notebooks=include_notebooks,
                                   notebook_objective=objective) for t in trees):
        raise ValueError("update: dynamic sampling must exclude signal-free solution trees")
    parameters = [p for ps in bundle.params.values() for p in ps]
    expected = [p for g in ("L", "M", "V") for p in bundle.params[g]]
    if ([id(p) for g in optimizer.param_groups for p in g["params"]] != [id(p) for p in expected]
            or any(p.grad is not None for p in bundle.model.parameters())):
        raise ValueError("update: optimizer partitions or cleared gradient boundary differ")
    groups = [g for t in trees for g in t.groups if g.usable or repair_selected]
    flags = ({tree.problem_id: repair["receipts"][tree.problem_id]["route"] == "VISION_ON" for tree in trees}
             if repair is not None else route_flags(trees, bundle.inventory["arm"], seed, step))
    if type(vision_disabled) is not bool:
        raise ValueError("update: explicit grounding-health route flag required")
    if vision_disabled:
        flags = dict.fromkeys(flags, False)
    if algorithm == "CTL-BLIND":
        # Runtime must replace before sampling, not merely before the gradient pass.
        if any(t.notebook.get("input_condition") != "BLIND_SAMPLED" for t in trees):
            raise ValueError("update: blind rollout input attestation missing")
        for values in images.values():
            for image in values:
                if image.mode != "RGB" or image.getextrema() != ((128, 128),)*3:
                    raise ValueError("update: CTL-BLIND requires the sampled same-size gray image")
    rewards = ({group.group_id: [repair["rows"][row.sample_id]["answer_reward"] for row in group.samples]
                for group in groups} if repair is not None else effective_rewards(groups, algorithm, seed, step,
                                notebook_reward_enabled=notebook_reward_enabled,
                                notebook_scores=notebook_scores))
    lambdas = protocol["loss"]["group_weights"]
    memberships = _page_memberships({tree.problem_id: tree.notebook["learning_objective"] for tree in trees}) if paged else []
    page_caps = {sample_id: row["page_binding"]["page_plan"]["pages"][row["page_index"]]["max_new_tokens"]
                 for sample_id, row in (notebook_scores or {}).items()} if paged else {}
    if paged and any(not group.usable for tree in trees for group in tree.groups if group.kind == "DIAG"):
        raise ValueError("update: every original page must remain usable before optimizer")
    denominator = sum(lambdas[group.kind] for group in groups if not paged or group.kind != "DIAG")
    if paged:
        denominator += len({row["problem_id"] for row in memberships}) * lambdas["DIAG"]
    if denominator <= 0:
        raise ValueError("update: positive fixed group-weight denominator required")
    actions = []
    missing_actions = 0
    for tree in trees:
        for group in tree.groups:
            if not group.usable and not repair_selected:
                continue
            cap = (repair["groups"][group.group_id]["completion_cap"] if repair is not None else
                   protocol["rollout"]["diag"]["max_new_tokens"] if group.kind == "DIAG" else
                   protocol["rollout"]["max_new_tokens"][group.samples[0].turn])
            if paged and group.kind == "DIAG":
                cap = page_caps[group.samples[0].sample_id]
                if any(page_caps[sample.sample_id] != cap for sample in group.samples):
                    raise ValueError("update: original page cap differs within group")
            advantages = group_advantages(torch.tensor(rewards[group.group_id], dtype=torch.float32))
            credits = {}
            from .notebook_objective import FACT_CREDIT_VERSION
            if (group.kind == "DIAG" and include_notebooks and objective
                    and objective["objective_version"] in (FACT_CREDIT_VERSION, SYNTHESIS_CREDIT_VERSION, ASSESSMENT_VERSION)):
                from . import fact_credit, synthesis_credit

                assert notebook_scores is not None
                if synthesis_credit_selected:
                    credits = synthesis_credit.group_credit(group.samples, notebook_scores, preparation.tokenizer)
                else:
                    credits = fact_credit.group_credit(group.samples, notebook_scores, preparation.tokenizer)
            for index, sample in enumerate(group.samples):
                if not 0 < len(sample.completion_token_ids) <= cap:
                    raise ValueError("update: empty or over-cap actual completion")
                mask = (torch.tensor([repair["rows"][sample.sample_id]["action_mask"]], dtype=torch.bool)
                        if repair is not None else action_mask(sample, algorithm, preparation.tokenizer))
                if not mask.any():
                    missing_actions += 1
                    continue
                request = RolloutRequest(sample.sample_id, sample.prompt_token_ids, None,
                    images[tree.problem_id], 1, cap, [sample.seed])
                credit = credits.get(sample.sample_id)
                advantage = credit["strict_advantage"] if credit is not None else float(advantages[index])
                actions.append(_PreparedAction(sample, request, mask, advantage,
                    lambdas[group.kind]/denominator,
                    len(group.samples) * (8192 if paged and group.kind == "DIAG" else cap),
                    flags[tree.problem_id], credit))
    if paged and any(sample_id not in {action.sample.sample_id for action in actions}
                     for membership in memberships for sample_id in membership["sample_ids"]):
        raise ValueError("update: missing original page action mask")
    base: dict[str, Any] = {"step_before": step, "step_after": step, "engine_sync_required": False,
            "n_problems": len(trees), "n_groups_by_kind": {k: sum(g.kind == k for g in groups)
            for k in ("T1", "T2", "T3", "DIAG")}, "no_action_samples": missing_actions,
            "routes": flags, "vision_health_disabled": vision_disabled, "effective_rewards": rewards,
            "notebook_reward_enabled": notebook_reward_enabled}
    if repair is not None:
        labels = {tree.problem_id: repair["receipts"][tree.problem_id]["route"] for tree in trees}
        if any(label not in ("VISION_ON", "VISION_OFF") for label in labels.values()):
            base["route_labels"] = labels
    if native_requirement_selected:
        base["notebook_update_requirement"] = protocol["notebook_update_requirement"]
        base["native_relationship_commit_required"] = require_native_relationship
    if assessment_requirement_selected:
        base.update(notebook_update_requirement=requirement, assessment_commit_required=require_assessment,
                    assessment_learning_verified=False)
    if synthesis_selected:
        base["synthesis_regrade_receipts"] = regraded_synthesis
    if objective is not None:
        base["notebook_objective"] = objective
        base["notebook_objective_receipts"] = {t.problem_id: json.loads(json.dumps(t.notebook["learning_objective"], allow_nan=False))
                                                for t in trees}
    if any(a.partial_credit is not None for a in actions):
        base["synthesis_token_credit" if synthesis_credit_selected else "fact_token_credit"] = {
            a.sample.sample_id: a.partial_credit for a in actions if a.partial_credit is not None}
    if require_retained_learning_signal:
        base["retained_learning_signal"] = (_retained_repair_signal([], repair["rows"]) if repair is not None
            else _retained_learning_signal([], protocol["loss"]["clip"]))
        if any(action.sample.policy_version != step for action in actions):
            raise ValueError("update: retained learning signal requires same-policy samples")
    if require_native_relationship:
        base["retained_native_relationship_signal"] = _retained_native_relationship_signal([])
        if any(action.sample.policy_version != step for action in actions):
            raise ValueError("update: native relationship signal requires same-policy samples")
    if require_assessment:
        base["retained_assessment_signal"] = _retained_assessment_signal([])
        if any(action.sample.policy_version != step for action in actions):
            raise ValueError("update: assessment signal requires same-policy samples")
    if not actions:
        return base | {"status": "NO_ACTIONS"}
    if repair_selected:
        from .episode_notebook import ANSWER_ROUTE_RULE
        from .repair_notebook import S_DEFINITIONS, TEXT_S_DEFINITION, note_contract
    repair_accumulator = (_RepairGradientAccumulator(bundle, S_DEFINITIONS[note_contract(protocol)],
                          TEXT_S_DEFINITION if protocol.get("repair_route_policy") == ANSWER_ROUTE_RULE else None,
                          measured=measure_evidence_vjp)
                          if repair_selected else None)
    repair_state = (_repair_update_snapshot(trees, observation_actors, protocol, images, preparation,
                    repair["receipts"], actions) if repair is not None else None)
    if repair is not None:
        base.update(repair_preparation=repair["receipts"], repair_learning_verified=False)
    repair_journal_sha = None

    def repair_unchanged() -> None:
        if repair is not None and repair_state != _repair_update_snapshot(trees, observation_actors,
                protocol, images, preparation, repair["receipts"], actions):
            raise ValueError("repair update: source, credit, protocol, or images mutated")
        if repair is not None and repair_journal_sha is not None and (
                base.get("repair_actions_sha256") != _hash(base.get("repair_actions"))
                or _hash([base.get("repair_execution"), base.get("repair_actions")]) != repair_journal_sha
                or _hash(_repair_action_journal(actions, repair["rows"])) != base["repair_actions_sha256"]):
            raise ValueError("repair update: original old-logprob action journal mutated")

    assessment_accumulator = _AssessmentGradientAccumulator(bundle) if require_assessment else None
    progress_started = time.monotonic()

    def progress(stage: str, state: str, index: int = 0, action: _PreparedAction | None = None) -> None:
        if on_progress is not None:
            event = {"event": "UPDATE_PROGRESS", "step": step, "stage": stage, "state": state,
                     "action_index": index, "action_count": len(actions),
                     "host_elapsed_seconds": time.monotonic()-progress_started}
            if action is not None:
                event.update(sample_id=action.sample.sample_id, kind=action.sample.kind,
                             prompt_tokens=len(action.sample.prompt_token_ids),
                             completion_tokens=len(action.sample.completion_token_ids),
                             vision_on=action.vision_on)
            try:
                on_progress(event)
                repair_unchanged()
            except BaseException:
                if repair_selected:
                    optimizer.zero_grad(set_to_none=True)
                raise

    device = next(bundle.model.parameters()).device
    bundle.model.train()
    timings, abs_diff, delta_values = {}, [], {}
    log_ratio_sums = {}
    began = _stage_time(device)
    with torch.no_grad():
        for index, action in enumerate(actions, 1):
            progress("old", "START", index, action)
            bundle.route.apply(action.vision_on)
            inputs = preparation.model_inputs(action.request, action.sample.completion_token_ids, device)
            old = completion_logprobs(bundle, inputs, len(action.sample.completion_token_ids),
                                     chunk=protocol["model"]["logprob_chunk"])
            sampler = torch.tensor([action.sample.sampler_logprobs], device=device, dtype=old.dtype)
            mask = action.mask.to(device)
            keep, delta = mis_mask(old, sampler, mask, protocol["loss"]["mismatch"]["delta_seq"])
            action.old, action.keep = old.detach().cpu(), bool(keep.item())
            delta_values[action.sample.sample_id] = float(delta.item())
            if paged:
                work_old = old.float() if old.dtype in (torch.float16, torch.bfloat16) else old
                work_sampler = sampler.float() if sampler.dtype in (torch.float16, torch.bfloat16) else sampler
                log_ratio_sums[action.sample.sample_id] = float(torch.where(mask, work_old - work_sampler, 0.).sum())
            abs_diff.extend((old-sampler).abs()[mask].cpu().tolist())
            del inputs, old, sampler
            progress("old", "RETURN", index, action)
    timings["old"] = _stage_time(device)-began
    if paged:
        counts = {action.sample.sample_id: int(action.mask.sum()) for action in actions}
        effective, mis_evidence = _page_mis_evidence(memberships, delta_values, counts,
                                                   protocol["loss"]["mismatch"]["delta_seq"], log_ratio_sums)
        for action in actions:
            if action.sample.kind == "DIAG":
                action.keep = bool(torch.tensor(effective[action.sample.sample_id], dtype=torch.float32).abs()
                                   <= protocol["loss"]["mismatch"]["delta_seq"])
        base.update(action_token_counts_by_sample=counts, masked_log_ratio_sum_by_sample=log_ratio_sums,
                    effective_mean_log_ratio_by_sample=effective, bundle_mis=mis_evidence)
    masked = sum(not a.keep for a in actions)/len(actions)
    base.update(masked_fraction=masked, mean_abs_lp_diff=statistics.mean(abs_diff),
                mean_log_ratio_by_sample=delta_values)
    if repair is not None:
        repair_unchanged()
        base["repair_execution"] = {"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol["sha256"],
            "algorithm": algorithm, "arm": bundle.inventory["arm"], "seed": seed}
        base["repair_actions"] = _repair_action_journal(actions, repair["rows"])
        base["repair_actions_sha256"] = _hash(base["repair_actions"])
        repair_journal_sha = _hash([base["repair_execution"], base["repair_actions"]])
        base["retained_repair_signal"] = _retained_repair_signal(actions, repair["rows"])
        if notebook_reward_enabled and repair_accumulator is not None:
            for action in actions:
                credit = repair["credits"].get(action.sample.sample_id)
                if not action.keep and credit is not None:
                    assert action.old is not None
                    if measure_evidence_vjp:
                        _repair_gradient(bundle, action, action.old, action.old, action.old, action.mask,
                                         protocol["loss"], credit, repair_accumulator)
                    else:
                        repair_accumulator.record(action, credit)
        if repair_accumulator is not None:
            base["repair_aggregate_gradient"] = repair_accumulator.receipt(bundle)
    if require_retained_learning_signal:
        if any(parameter.grad is not None for parameter in parameters):
            raise ValueError("update: retained learning signal requires cleared gradients")
        base["retained_learning_signal"] = (_retained_repair_signal(actions, repair["rows"]) if repair is not None
            else _retained_learning_signal(actions, protocol["loss"]["clip"]))
    if require_native_relationship:
        if any(parameter.grad is not None for parameter in parameters):
            raise ValueError("update: native relationship signal requires cleared gradients")
        base["retained_native_relationship_signal"] = _retained_native_relationship_signal(actions)
        if not base["retained_native_relationship_signal"]["policy_gradient_present"]:
            return base | {"status": "NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL", "seconds": timings}
    if require_assessment:
        if any(parameter.grad is not None for parameter in parameters):
            raise ValueError("update: assessment signal requires cleared gradients")
        base["retained_assessment_signal"] = _retained_assessment_signal(actions)
        assessment_members = [_assessment_member(action) for action in actions
                              if action.sample.kind == "DIAG" and action.partial_credit is not None]
        if not base["retained_assessment_signal"]["policy_gradient_present"]:
            return base | {"status": "NO_RETAINED_ASSESSMENT_SIGNAL", "seconds": timings}
    if all(not a.keep for a in actions):
        return base | {"status": "NO_MATCHED_ACTIONS", "seconds": timings}
    if repair is not None and not base["retained_repair_signal"]["policy_gradient_present"]:
        return base | {"status": "NO_LEARNING_SIGNAL", "seconds": timings}
    if require_retained_learning_signal and not base["retained_learning_signal"]["policy_gradient_present"]:
        return base | {"status": "NO_RETAINED_LEARNING_SIGNAL", "seconds": timings}
    began = _stage_time(device)
    for index, action in enumerate(actions, 1):
        if action.keep:
            progress("reference", "START", index, action)
            bundle.route.apply(action.vision_on)
            inputs = preparation.model_inputs(action.request, action.sample.completion_token_ids, device)
            action.reference = reference_logprobs(bundle, inputs, len(action.sample.completion_token_ids),
                chunk=protocol["model"]["logprob_chunk"]).cpu()
            del inputs
            progress("reference", "RETURN", index, action)
    timings["ref"] = _stage_time(device)-began
    began = _stage_time(device)
    total_loss, kl_sum, token_count, max_old_diff = 0.0, 0.0, 0, 0.0
    notebook_backward = []
    relationship_measured = False
    native_relationship_measured = False
    try:
        for index, action in enumerate(actions, 1):
            if not action.keep:
                continue
            progress("backward", "START", index, action)
            assert action.old is not None and action.reference is not None
            bundle.route.apply(action.vision_on)
            inputs = preparation.model_inputs(action.request, action.sample.completion_token_ids, device)
            current = completion_logprobs(bundle, inputs, len(action.sample.completion_token_ids),
                                         chunk=protocol["model"]["logprob_chunk"])
            old, reference, mask = action.old.to(device), action.reference.to(device), action.mask.to(device)
            difference = float((current.detach()-old).abs().max())
            max_old_diff = max(max_old_diff, difference)
            if difference >= 1e-3:
                raise RuntimeError("update: pre-update current/old log probabilities differ")
            if repair is not None:
                from .losses import repair_component_losses

                repair_unchanged()
                row, group = repair["rows"][action.sample.sample_id], repair["groups"][action.sample.group_id]
                components = {key: torch.tensor([row[f"applied_{key}_coefficients"]],
                    device=device, dtype=current.dtype) for key in _repair_channels(row)}
                loss = action.weight * repair_component_losses(current, old, reference, mask,
                    torch.tensor([True], device=device), **components, kind=action.sample.kind,
                    original_group_size=group["original_group_size"], completion_cap=group["completion_cap"])["total"]
                credit = repair["credits"].get(action.sample.sample_id)
                if notebook_reward_enabled and credit is not None and repair_accumulator is not None:
                    if measure_evidence_vjp:
                        _repair_gradient(bundle, action, current, old, reference, mask, protocol["loss"],
                                         credit, repair_accumulator)
                    else:
                        repair_accumulator.record(action, credit)
            else:
                loss = action.weight * group_loss(current, old, reference,
                torch.tensor([action.advantage], device=device), mask,
                torch.tensor([True], device=device), eps_low=protocol["loss"]["clip"]["eps_low"],
                eps_high=protocol["loss"]["clip"]["eps_high"], beta=protocol["loss"]["kl"]["beta"],
                norm_const=action.norm,
                partial_adv=(None if action.partial_credit is None else torch.tensor(
                    [action.partial_credit["partial_token_advantages"]], device=device, dtype=current.dtype)))
            kl_sum += float(k3(reference, current.detach())[mask].sum())
            token_count += int(mask.sum())
            total_loss += float(loss.detach())
            before_note = (_gradient_identity(bundle.params) if include_notebooks
                           and action.sample.kind == "DIAG" else None)
            relationship_gradient = {"scope": "relationship_policy_loss_only", "measured": False}
            native_relationship_gradient = {"scope": "native_error_relationship_policy_loss_only", "measured": False}
            assessment_gradient = {"scope": "native_assessment_policy_loss_only", "measured": False}
            if (synthesis_credit_selected and before_note is not None and action.partial_credit is not None
                    and not relationship_measured and any(row["reward"] == 1 and row["advantage"] > 0
                        and row["token_indices"] for row in action.partial_credit["relationships"])):
                relationship_gradient = _relationship_gradient(bundle, action, current, old, reference,
                                                                 mask, protocol["loss"])
                relationship_measured = True
            if (synthesis_credit_selected and before_note is not None and action.partial_credit is not None
                    and not native_relationship_measured):
                native_relationship_gradient = _native_relationship_gradient(bundle, action, current, old,
                    reference, mask, protocol["loss"], relationship_gradient)
                native_relationship_measured = (_native_relationship_gradient_present(
                    native_relationship_gradient, action.partial_credit) if require_native_relationship
                    else native_relationship_gradient["measured"] is True)
            if (assessment_credit_selected and before_note is not None and action.partial_credit is not None
                    and assessment_accumulator is not None):
                assessment_gradient = _assessment_gradient(bundle, action, current, old, reference,
                                                           mask, protocol["loss"], assessment_accumulator)
            loss.backward()
            if before_note is not None:
                after_note = _gradient_identity(bundle.params)
                evidence = {"sample_id": action.sample.sample_id,
                    "group_id": action.sample.group_id, "advantage": action.advantage,
                    "loss": float(loss.detach()), "action_tokens": int(mask.sum()),
                    "vision_on": action.vision_on,
                    "gradient_before": before_note, "gradient_after": after_note,
                    # Only a zero KL derivative separates reward from regularization here.
                    "reward_gradient_isolated": protocol["loss"]["kl"]["beta"] == 0
                        or torch.equal(current.detach()[mask], reference[mask]),
                    "changed_groups": [g for g in before_note if before_note[g] != after_note[g]]}
                if notebook_scores is not None:
                    score = notebook_scores[action.sample.sample_id]
                    evidence.update(strict_reward=action.sample.reward,
                                    atomic_reward=score["atomic_reward"],
                                    effective_reward=score["effective_reward"],
                                    objective_score_sha256=_hash(score),
                                    strict_and_atomic_gradients_separately_measured=False)
                if action.partial_credit is not None:
                    partial = action.partial_credit["partial_token_advantages"]
                    evidence["synthesis_credit_sha256" if synthesis_credit_selected else "fact_credit_sha256"] = _hash(
                        action.partial_credit)
                    evidence["nonzero_partial_tokens"] = sum(value != 0 for value in partial)
                if synthesis_credit_selected:
                    evidence["relationship_gradient"] = relationship_gradient
                    evidence["native_relationship_gradient"] = native_relationship_gradient
                if assessment_credit_selected:
                    evidence["assessment_gradient"] = assessment_gradient
                notebook_backward.append(evidence)
            del loss, current, inputs
            progress("backward", "RETURN", index, action)
        if assessment_accumulator is not None:
            base["assessment_aggregate_gradient"] = assessment_accumulator.receipt(assessment_members)
            assessment_accumulator.sums.clear()
        if repair_accumulator is not None:
            base["repair_aggregate_gradient"] = repair_accumulator.receipt(bundle)
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise RuntimeError("HEALTH_STOP:NONFINITE_GRADIENT")
        trainable_ids = {id(p) for p in parameters}
        if any(p.grad is not None for p in bundle.model.parameters() if id(p) not in trainable_ids):
            raise RuntimeError("update: frozen base acquired a gradient")
        grad_norm = {}
        for partition, ps in bundle.params.items():
            active = [p for p in ps if p.grad is not None]
            grad_norm[partition] = float(torch.nn.utils.clip_grad_norm_(active,
                protocol["optimizer"]["grad_clip_per_group"], error_if_nonfinite=True)) if active else None
        # No fallible observer runs after parameters or optimizer state may change.
        progress("optimizer", "START")
        if synthesis_selected:
            # Recheck after observers/forwards/backwards, directly before parameter mutation.
            for tree in trees:
                actor = observation_actors[tree.problem_id]  # type: ignore[index]
                validate_update_binding(tree, tokenizer=preparation.tokenizer,
                    generation_policy=protocol.get("notebook_generation_policy"),
                    draft_policy=protocol.get("notebook_draft_policy"),
                    feedback_policy=protocol.get("notebook_feedback_policy"),
                    observation_policy=protocol.get("notebook_observation_policy"),
                    preparation=preparation, images=images[tree.problem_id], observation_actor=actor,
                    observation_schema=tree.notebook["rewarded_notebook"]["notebook_schema"],
                    observation_master_seed=seed)
            if synthesis_receipts() != regraded_synthesis:
                raise ValueError("update: freshly regraded synthesis changed before optimizer")
        if synthesis_credit_selected:
            from .synthesis_credit import group_credit

            assert observation_actors is not None and objective is not None
            fresh_scores = {}
            for tree in trees:
                fresh_scores.update(notebook_objective_scores(tree, objective))
                fresh = _score_notebook_objective(tree, observation_actors[tree.problem_id],
                                                 objective_verifier, objective)
                if (fresh != tree.notebook["learning_objective"]
                        or fresh != base["notebook_objective_receipts"][tree.problem_id]):
                    raise ValueError("update: freshly regraded synthesis objective changed before optimizer")
            if include_notebooks:
                fresh_credits = {sample_id: credit for group in groups if group.kind == "DIAG"
                    for sample_id, credit in group_credit(group.samples, fresh_scores,
                                                          preparation.tokenizer).items()}
                if fresh_credits != base.get("synthesis_token_credit", {}):
                    raise ValueError("update: synthesis token credit changed before optimizer")
                if require_native_relationship and not any(_native_relationship_gradient_present(
                        row["native_relationship_gradient"], fresh_credits[row["sample_id"]])
                        for row in notebook_backward):
                    timings["train"] = _stage_time(device)-began
                    return base | {"status": "NO_NATIVE_RELATIONSHIP_GRADIENT", "seconds": timings,
                                   "notebook_backward": notebook_backward}
                if require_assessment and not base["assessment_aggregate_gradient"]["nonzero_groups"]:
                    timings["train"] = _stage_time(device)-began
                    return base | {"status": "NO_ASSESSMENT_GRADIENT", "seconds": timings,
                                   "notebook_backward": notebook_backward}
        if paged:
            validate_page_mis_evidence(base, protocol, trees)
        if assessment_accumulator is not None:
            assessment_accumulator.validate(bundle)
        if repair is not None and repair_accumulator is not None:
            repair_unchanged()
            fresh_repair = _prepare_repair_update(trees, observation_actors, objective_verifier, protocol,
                images, preparation, seed=seed, notebook_reward_enabled=notebook_reward_enabled)
            repair_unchanged()
            if _hash(fresh_repair["receipts"]) != _hash(repair["receipts"]):
                raise ValueError("repair update: fresh physical or semantic credit changed before optimizer")
            repair_accumulator.validate(bundle)
            base["repair_aggregate_gradient"] = repair_accumulator.receipt(bundle)
        if before_optimizer_step is not None:
            before_optimizer_step()
        optimizer.step()
        scheduler.step()
        if require_assessment:
            base["assessment_learning_verified"] = True
    finally:
        if assessment_accumulator is not None:
            assessment_accumulator.sums.clear()
        if repair_accumulator is not None:
            repair_accumulator.clear()
        optimizer.zero_grad(set_to_none=True)
    timings["train"] = _stage_time(device)-began
    return base | {"status": "UPDATED_UNSYNCHRONIZED", "step_after": step+1,
        "engine_sync_required": True, "loss": total_loss, "kl_mean": kl_sum/token_count,
        "grad_norm": grad_norm, "max_current_old_difference": max_old_diff, "seconds": timings,
        "notebook_backward": notebook_backward}
