"""Host-only verified note atoms; the original strict reward remains unchanged."""
from __future__ import annotations

import json

from . import episode_notebook as notebook
from .local_eval import digest
from .observations import observation_spans, slot_id
from .schema import to_json

VERSION = "verified-note-atoms-v1"
COMPONENT_VERSION = "verified-note-components-v1"
FACT_CREDIT_VERSION = "verified-note-fact-token-credit-v2"
SYNTHESIS_CREDIT_VERSION = "verified-synthesis-unit-token-credit-v2"
ASSESSMENT_VERSION = "verified-whole-reference-assessment-credit-v3"
ATOMIC_WEIGHT = 0.25


def contract(version: str = VERSION) -> dict:
    if version in (SYNTHESIS_CREDIT_VERSION, ASSESSMENT_VERSION):
        from .synthesis_credit import contract as synthesis_contract

        return synthesis_contract(version)
    if version not in (VERSION, COMPONENT_VERSION, FACT_CREDIT_VERSION):
        raise ValueError("notebook objective: unknown version")
    base = {"objective_version": version, "strict_reward_version": notebook.VERSION,
        "wrong_slot_ids": "observations.slot_id normalization; submitted order and multiplicity preserved",
        "strict_weight": 1.0, "atomic_weight": ATOMIC_WEIGHT,
        "formula": "strict_reward + 0.25 * atom_fraction",
        "denominator": "actual_all_attempt_count * private_required_slot_count",
        "atom": "independently correct required final slot, cited original observation, correct wrong_slots membership",
        "attempt_gate": ["history", "cause", "conclusion_evidence"],
        "note_gate": "complete strict checks/submitted_claims and verified answer",
        "fact_guard": "canonical IDs; no unknown keys, conflicts, false or unverified relations",
        "singleton_queries": "one per submitted slot within this note; model-provided values only",
        "missing_or_bad_atoms": "zero numerator credit; fixed denominator",
        "selection_or_forced_variance": False, "model_claims_used_for_routing": False,
        "lesson_semantics_verified": False, "module_causality_certified": False}
    if version == COMPONENT_VERSION:
        base.update(denominator="4 * actual_all_attempt_count * private_required_slot_count",
            atom="equal-weight verified fact fraction and three independently checked attempt fractions",
            attempt_gate=[], components={
                "required_facts": "independently correct submitted required facts / all required facts",
                "history_with_conclusion": "history AND conclusion_evidence / all attempts",
                "cause_with_history": "history AND cause / all attempts; taxonomy match, not module causality",
                "error_locations_with_evidence": "exact wrong_slots AND visual_evidence / all attempts"},
            component_weight=0.25, vacuous_visual_evidence_credit=False,
            empty_error_list_credit="only independently matching empty error sets; not visual evidence or healthy Vision")
    elif version == FACT_CREDIT_VERSION:
        base.update(denominator="4 * actual_all_attempt_count * private_required_slot_count",
            formula="strict_reward + 0.0625 * verified_required_fact_fraction",
            atom="independently checked submitted required fact value", attempt_gate=[],
            fact_guard="observations.slot_id normalized IDs; no unknown keys, collisions, conflicts, false or unverified relations",
            corrected_fact_ids="observations.slot_id normalization; original values and JSON value spans preserved",
            fact_id_collisions="reject normalized duplicate IDs even for equal values",
            partial_credit="per-slot full-group centered rewards on original JSON value tokens only",
            strict_credit="separately centered and clipped whole-completion strict reward",
            partial_weight_per_slot="0.0625 / private_required_slot_count",
            other_component_partial_weights=0, invalid_or_missing_slot_reward=0,
            missing_value_action="none; never synthesize a target",
            token_boundary="wholly contained original value literal; stable prefix decoding only",
            mis_and_kl="unchanged full-completion mask", normalization="unchanged n_g * L_cap")
    return base


def effective_reward(strict_reward: float, numerator: int, denominator: int) -> float:
    if (type(strict_reward) not in (int, float) or strict_reward not in (0, 1)
            or type(numerator) is not int or type(denominator) is not int
            or not 0 <= numerator <= denominator):
        raise ValueError("notebook objective: binary strict reward and fixed valid atom counts required")
    return float(strict_reward + ATOMIC_WEIGHT * (numerator / denominator if denominator else 0.0))


def score_note(text, finish_reason, tree, audit, verifier, strict_report=None, *, actor,
               objective_version: str = VERSION, replay_relation_v1: bool = False) -> dict:
    """Recompute the bound strict report; never trust or mutate a supplied record."""
    if objective_version == ASSESSMENT_VERSION:
        raise ValueError("notebook objective: assessment credit requires synthesis_credit.score_note with the actual raw sample ID")
    contract(objective_version)
    fresh_audit = notebook.audit_episode(tree, actor, verifier, routing_rule=audit.get("routing_rule"),
                                         replay_relation_v1=replay_relation_v1)
    if digest(fresh_audit) != digest(audit):
        raise ValueError("notebook objective: independently bound source audit differs")
    policy = tree.notebook.get("rewarded_notebook", {}).get("feedback_policy")
    strict = notebook.grade_note(text, finish_reason, tree, fresh_audit, verifier, feedback_policy=policy)
    if strict_report is not None and digest(strict_report) != digest(strict):
        raise ValueError("notebook objective: supplied strict report differs from fresh verification")
    empty = verifier.grade(tree.problem_id, "Observed facts:\nSolution:\n", "stop", "obs").slots
    required = empty.required if empty else 0
    samples = notebook.solution_samples(tree)
    denominator = len(samples) * required * (4 if objective_version != VERSION else 1)
    receipt = {"objective_version": objective_version, "strict_reward": strict["reward"],
        "numerator": 0, "denominator": denominator, "atom_fraction": 0.0, "atomic_reward": 0.0,
        "effective_reward": effective_reward(strict["reward"], 0, denominator),
        "objective_independently_verified": True, "atomic_gate": "INVALID_STRICT_NOTE",
        "solution_sha256": fresh_audit["solution_sha256"], "audit_sha256": digest(fresh_audit),
        "strict_report_sha256": digest(strict), "text_sha256": digest(text),
        "actor_sha256": digest(json.loads(to_json(actor))), "finish_reason": finish_reason,
        "lesson_semantics_verified": False, "module_causality_certified": False}
    if objective_version == FACT_CREDIT_VERSION:
        receipt.update(required_slot_count=required, fact_rewards={})
    claims = strict.get("submitted_claims")
    if (set(strict["checks"]) != set(notebook._SUMMARY_CHECKS)
            or strict["checks"].get("format_valid") is not True or type(claims) is not dict
            or len(strict["attempt_checks"]) != len(samples)):
        return receipt
    if not strict["checks"]["answer_verified"]:
        return receipt | {"atomic_gate": "ANSWER_UNVERIFIED"}
    facts = claims["corrected_facts"]
    if objective_version == FACT_CREDIT_VERSION:
        normalized = {slot_id(key): value for key, value in facts.items()}
        if len(normalized) != len(facts):
            return receipt | {"atomic_gate": "INVALID_SLOT_IDS"}
        facts = normalized
    wrong_lists = [claim["wrong_slots"] for claim in claims["attempts"]]
    if (any(key != slot_id(key) for key in facts)
            or any(wrong != sorted(set(wrong)) or any(key != slot_id(key) for key in wrong)
                   for wrong in wrong_lists)):
        return receipt | {"atomic_gate": "INVALID_SLOT_IDS"}
    if not denominator or not facts:
        return receipt | {"atomic_gate": "NO_REQUIRED_OR_SUBMITTED_FACTS"}
    final = verifier.grade(tree.problem_id,
        notebook._fact_projection(facts, claims["corrected_answer"]), "stop", "obs").slots
    if (final is None or final.required != required or final.parsed != len(facts)
            or final.conflicting or final.false_relations or final.unverified_relations):
        return receipt | {"atomic_gate": "FACT_CONFLICT_OR_UNKNOWN"}
    # These private singleton checks reveal no reference values or unsubmitted slot names.
    verified, submitted_required = set(), set()
    for key, value in facts.items():
        singleton = verifier.grade(tree.problem_id,
            f"Observed facts:\n{key} = {value}\nSolution:\n", "stop", "obs").slots
        if singleton and (singleton.required_correct == 1 or key in singleton.wrong_required):
            submitted_required.add(key)
        if (singleton and singleton.required == required and singleton.required_correct == 1
                and singleton.parsed == singleton.all_correct == 1 and not singleton.conflicting
                and not singleton.false_relations and not singleton.unverified_relations):
            verified.add(key)
    if objective_version == FACT_CREDIT_VERSION:
        numerator = len(verified) * len(samples)
        fraction = numerator / denominator
        return receipt | {"numerator": numerator, "atom_fraction": fraction,
            "atomic_reward": ATOMIC_WEIGHT * fraction,
            "effective_reward": effective_reward(strict["reward"], numerator, denominator),
            "atomic_gate": "VERIFIED_FACT_TOKEN_CREDIT",
            "fact_rewards": {key: int(key in verified) for key in sorted(submitted_required)}}
    if objective_version == COMPONENT_VERSION:
        # A failed cause must not erase a separately verified fact or source quotation.
        checks = strict["attempt_checks"]
        counts = {
            "required_facts": len(verified),
            "history_with_conclusion": sum(row["history"] and row["conclusion_evidence"] for row in checks),
            "cause_with_history": sum(row["history"] and row["cause"] for row in checks),
            "error_locations_with_evidence": sum(row["wrong_slots"] and row["visual_evidence"] for row in checks),
        }
        numerator = counts["required_facts"] * len(samples) + required * sum(
            value for name, value in counts.items() if name != "required_facts")
        fraction = numerator / denominator
        return receipt | {"numerator": numerator, "atom_fraction": fraction,
            "atomic_reward": ATOMIC_WEIGHT * fraction,
            "effective_reward": effective_reward(strict["reward"], numerator, denominator),
            "atomic_gate": "VERIFIED_COMPONENTS", "component_counts": counts,
            "component_denominators": {name: required if name == "required_facts" else len(samples)
                                       for name in counts}}
    numerator = 0
    for sample, claim, checked, expected in zip(samples, claims["attempts"], strict["attempt_checks"],
                                               fresh_audit["attempts"], strict=True):
        assert sample.result is not None  # audit_episode already reverified every actual solution.
        source = sample.result.slots
        if not source or not all(checked[name] is True for name in ("history", "cause", "conclusion_evidence")):
            continue
        if source.required != required:
            raise ValueError("notebook objective: private required-slot count changed")
        cited = checked["resolved_lines"]
        observed = {key for start, end, key, _ in observation_spans(sample.text)
                    if any(line["start"] <= start and end <= line["end"] for line in cited)}
        for key in verified & observed - set(source.conflicting):
            numerator += (key in claim["wrong_slots"]) == (key in expected["wrong_slots"])
    fraction = numerator / denominator
    return receipt | {"numerator": numerator, "atom_fraction": fraction,
        "atomic_reward": ATOMIC_WEIGHT * fraction,
        "effective_reward": effective_reward(strict["reward"], numerator, denominator),
        "atomic_gate": "VERIFIED_ATOMS"}
