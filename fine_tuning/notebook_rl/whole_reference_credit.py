"""Diagnostic unit checks on rendered references, with credit on original raw tokens.

The expanded JSON is a program rendering, never a model-generated action. Nothing
in this module is an admission gate, training objective, or optimizer integration.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from . import episode_notebook as episode, notebook_synthesis as synthesis
from . import synthesis_credit as existing
from . import whole_reference_notebook as reference
from .fact_credit import token_spans
from .local_eval import digest
from .rollout import _hash

VERSION = "whole-reference-raw-unit-diagnostic-v1"
GROUP_SIZE = 4
FACT_WEIGHT = RELATIONSHIP_WEIGHT = .0625
NATIVE_ERROR_RELATIONSHIP = existing.NATIVE_ERROR_RELATIONSHIP
OUTPUT_FINALIZATION = existing.OUTPUT_FINALIZATION


def _seal(row: dict) -> dict:
    return row | {"score_sha256": digest(row)}


def _binding(sample) -> dict:
    if (type(sample.sample_id) is not str or not sample.sample_id
            or type(sample.group_id) is not str or not sample.group_id
            or type(sample.text) is not str or type(sample.finish_reason) is not str
            or type(sample.completion_token_ids) is not list
            or any(type(value) is not int or value < 0 for value in sample.completion_token_ids)):
        raise ValueError("whole reference credit: original raw sample required")
    return {"sample_id": sample.sample_id, "group_id": sample.group_id,
            "text_sha256": digest(sample.text), "raw_text_sha256": hashlib.sha256(sample.text.encode()).hexdigest(),
            "token_ids_sha256": digest(sample.completion_token_ids), "finish_reason": sample.finish_reason}


def check(sample, tree, audit: dict, actor, verifier) -> dict:
    """Freshly grade the rendering and independent units; retain only raw spans."""
    binding = _binding(sample)
    fresh = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
    if type(audit) is not dict or _hash(audit) != _hash(fresh):
        raise ValueError("whole reference credit: source audit differs from fresh verification")
    samples = episode.solution_samples(tree)
    expanded = reference.expand(sample.text, sample.finish_reason, samples)
    reference.validate_audit_coverage(expanded["inventory"], fresh)
    rendered = expanded["expanded_text"]
    captured = type(rendered) is str
    # Rejected raw outputs have no rendered action; '{}' obtains a zero strict report.
    strict = synthesis.grade_note(rendered if captured else "{}", sample.finish_reason,
        tree, fresh, verifier, actor=actor, contract=synthesis.BOUNDED_VERSION)
    empty = verifier.grade(tree.problem_id, "Observed facts:\nSolution:\n", "stop", "obs").slots
    required = empty.required if empty else 0
    if type(required) is not int or required < 0:
        raise ValueError("whole reference credit: invalid fixed fact denominator")
    units = [(attempt["attempt"], issue) for attempt in fresh["attempts"] for issue in
             [*attempt["issues"], *(row for row in attempt["output_checks"] if row["status"] == "VIOLATED")]]
    originals = [existing.issue_id(attempt, issue) for attempt, issue in units]
    if len(originals) != len(set(originals)):
        raise ValueError("whole reference credit: duplicate original issue")
    types = {existing.issue_id(attempt, issue): OUTPUT_FINALIZATION if issue["kind"] == "OUTPUT"
             else NATIVE_ERROR_RELATIONSHIP for attempt, issue in units}
    facts, fact_spans = {}, {}
    if captured:
        note, values, _ = existing._json_units(rendered)
        for key, (value, _) in existing._fact_candidates(note, values).items():
            slots = verifier.grade(tree.problem_id, f"Observed facts:\n{key} = {value}\nSolution:\n", "stop", "obs").slots
            if slots is None:
                continue
            if slots.required != required:
                raise ValueError("whole reference credit: fact denominator changed")
            if slots.required_correct == 1 or key in slots.wrong_required:
                facts[key] = int(slots.required_correct == slots.parsed == slots.all_correct == 1
                    and not slots.conflicting and not slots.false_relations and not slots.unverified_relations)
                fact_spans[key] = expanded["fact_spans"][key]
    rewards = dict.fromkeys(originals, 0)
    spans, relations, selected, answer_correct = {}, {}, {}, {}
    for ref, lesson in expanded["decisions"].items():
        origin = existing._origin(lesson)
        if origin not in rewards or origin in selected:
            continue
        selected[origin] = ref
        spans[origin] = expanded["decision_spans"][ref]
        relations[origin] = lesson["relation"]
        positive = (lesson["relation"] == "FINALIZED" if types[origin] == OUTPUT_FINALIZATION
                    else lesson["relation"] in ("CORRECTED", "PERSISTED"))
        if positive:
            checked = synthesis._lesson(lesson, samples, fresh, bounded=True)
            rewards[origin] = int(checked["verified"])
            if checked["verified"] and types[origin] == OUTPUT_FINALIZATION:
                if type(checked.get("later_answer_correct")) is not bool:
                    raise ValueError("whole reference credit: separate output correctness missing")
                answer_correct[origin] = checked["later_answer_correct"]
    atomic = FACT_WEIGHT * (sum(facts.values()) / required if required else 0.)
    atomic += RELATIONSHIP_WEIGHT * (sum(rewards.values()) / len(originals) if originals else 0.)
    return _seal({"version": VERSION, "diagnostic_only": True, "generation_policy": reference.VERSION,
        "expansion_status": expanded["status"], "expansion_captured": captured,
        "expanded_text_sha256": digest(rendered) if captured else None,
        "program_rendered_only": True, "strict_report": strict,
        "strict_report_sha256": existing._strict_digest(strict), "strict_reward": strict["reward"],
        "diagnostic_atomic_reward": atomic, "fact_rewards": facts, "fact_spans": fact_spans,
        "relationship_rewards": rewards, "relationship_spans": spans, "relationship_relations": relations,
        "relationship_credit_types": types, "relationship_later_answer_correct": answer_correct,
        "selected_refs": selected, "required_fact_count": required, "original_issue_ids": originals,
        "native_verified_count": sum(reward for key, reward in rewards.items() if types[key] == NATIVE_ERROR_RELATIONSHIP),
        "output_verified_count": sum(reward for key, reward in rewards.items() if types[key] == OUTPUT_FINALIZATION),
        "inventory": expanded["inventory"], "inventory_sha256": digest(expanded["inventory"]),
        "solution_sha256": fresh["solution_sha256"], "transcript_sha256": fresh["transcript_sha256"],
        "audit_sha256": _hash(fresh), "actor_sha256": fresh["actor_sha256"],
        "model_claims_used_for_routing": False, "module_causality_certified": False,
        "full_diagnosis": False, "all_reasoning_verified": False, **binding})


def _raw_decisions(text: str) -> dict[str, tuple[Any, list[int]]]:
    """Walk JSON offsets, so identical decision literals retain their own position."""
    note, spans = reference._decode(text)
    refs = [row[0] for row in note["assessments"]]
    if len(refs) != len(set(refs)):
        raise ValueError("whole reference credit: duplicate raw source reference")
    return {row[0]: (row[3], spans[("assessments", index, 3)])
            for index, row in enumerate(note["assessments"]) if row[3] is not None}


def validate_score(row: dict, sample) -> None:
    """Validate raw bookkeeping; semantic claims still require a fresh check()."""
    failure = "whole reference credit: invalid bound diagnostic score"
    if (type(row) is not dict or row.get("version") != VERSION or row.get("diagnostic_only") is not True
            or row.get("generation_policy") != reference.VERSION or row.get("program_rendered_only") is not True
            or type(row.get("inventory")) is not dict or type(row.get("strict_report")) is not dict
            or row.get("score_sha256") != digest({key: value for key, value in row.items() if key != "score_sha256"})
            or any(row.get(key) != value for key, value in _binding(sample).items())
            or row.get("inventory_sha256") != digest(row.get("inventory"))
            or row.get("strict_report_sha256") != existing._strict_digest(row.get("strict_report"))
            or type(row.get("strict_reward")) not in (float, int) or row["strict_reward"] not in (0, 1)
            or row["strict_report"].get("reward") != row["strict_reward"]
            or type(row.get("expansion_captured")) is not bool
            or any(row.get(key) is not False for key in ("model_claims_used_for_routing",
                "module_causality_certified", "full_diagnosis", "all_reasoning_verified"))):
        raise ValueError(failure)
    if (any(type(row.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None
            for key in ("solution_sha256", "transcript_sha256", "audit_sha256", "actor_sha256"))
            or row["inventory"].get("sha256") != _hash({
                key: value for key, value in row["inventory"].items() if key != "sha256"})
            or row["expansion_captured"] and (sample.finish_reason != "stop"
                or row.get("expansion_status") != reference.CAPTURED
                or type(row.get("expanded_text_sha256")) is not str)
            or not row["expansion_captured"] and row.get("expanded_text_sha256") is not None):
        raise ValueError(failure)
    required, originals = row.get("required_fact_count"), row.get("original_issue_ids")
    facts, rewards = row.get("fact_rewards"), row.get("relationship_rewards")
    if (type(required) is not int or required < 0 or type(originals) is not list
            or any(type(key) is not str for key in originals) or len(originals) != len(set(originals))
            or type(facts) is not dict or len(facts) > required or type(rewards) is not dict
            or set(rewards) != set(originals)
            or any(type(value) is not int or value not in (0, 1) for value in [*facts.values(), *rewards.values()])):
        raise ValueError(failure)
    try:
        types = {}
        for key in originals:
            origin = json.loads(key)
            if (type(origin) is not list or len(origin) != 4 or type(origin[0]) is not int or origin[0] < 1
                    or not all(type(value) is str for value in origin[1:])
                    or origin[1] not in ("OBSERVATION", "TRACE", "OUTPUT")
                    or origin[1] == "OUTPUT" and origin[2:] != ["FINALIZATION", "final_answer"]
                    or key != existing.issue_id(origin[0], dict(zip(("kind", "rule", "locus"), origin[1:], strict=True)))):
                raise ValueError(failure)
            types[key] = OUTPUT_FINALIZATION if origin[1] == "OUTPUT" else NATIVE_ERROR_RELATIONSHIP
        note, values, _ = existing._json_units(sample.text)
        candidates = existing._fact_candidates(note, values) if row["expansion_captured"] else {}
        decisions = _raw_decisions(sample.text) if row["expansion_captured"] else {}
        selected, spans, relations = {}, {}, {}
        references = row["inventory"]["references"]
        for ref, (decision, span) in decisions.items():
            endpoint = references[ref]
            origin = existing.issue_id(endpoint["attempt"], endpoint)
            if origin in rewards and origin not in selected:
                selected[origin], spans[origin], relations[origin] = ref, span, decision[0]
        if (row.get("relationship_credit_types") != types or row.get("selected_refs") != selected
                or row.get("relationship_spans") != spans or row.get("relationship_relations") != relations
                or any(key not in candidates for key in facts)
                or row.get("fact_spans") != {key: candidates[key][1] for key in facts}
                or any(reward and (key not in spans or (relations[key] != "FINALIZED"
                    if types[key] == OUTPUT_FINALIZATION else relations[key] not in ("CORRECTED", "PERSISTED")))
                    for key, reward in rewards.items())):
            raise ValueError(failure)
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise ValueError(failure) from exc
    answer_correct = row.get("relationship_later_answer_correct")
    atomic = FACT_WEIGHT * (sum(facts.values()) / required if required else 0.)
    atomic += RELATIONSHIP_WEIGHT * (sum(rewards.values()) / len(originals) if originals else 0.)
    if (type(answer_correct) is not dict or set(answer_correct) != {key for key, reward in rewards.items()
            if reward and types[key] == OUTPUT_FINALIZATION}
            or any(type(value) is not bool for value in answer_correct.values())
            or type(row.get("diagnostic_atomic_reward")) not in (int, float)
            or not math.isfinite(row["diagnostic_atomic_reward"]) or row["diagnostic_atomic_reward"] != atomic
            or type(row.get("native_verified_count")) is not int
            or row["native_verified_count"] != sum(value for key, value in rewards.items() if types[key] == NATIVE_ERROR_RELATIONSHIP)
            or type(row.get("output_verified_count")) is not int
            or row["output_verified_count"] != sum(value for key, value in rewards.items() if types[key] == OUTPUT_FINALIZATION)
            or not row["expansion_captured"] and (facts or selected or row["strict_reward"])):
        raise ValueError(failure)


def group_credit(samples, rows: dict[str, dict], tokenizer) -> dict[str, dict]:
    """Center the original n4 before MIS; absent decisions never create actions."""
    if (len(samples) != GROUP_SIZE or len({sample.sample_id for sample in samples}) != GROUP_SIZE
            or len({sample.group_id for sample in samples}) != 1 or any(sample.kind != "DIAG" for sample in samples)
            or any(sample.sample_id not in rows for sample in samples)):
        raise ValueError("whole reference credit: original complete unique n4 DIAG group required")
    scores = [rows[sample.sample_id] for sample in samples]
    for sample, row in zip(samples, scores, strict=True):
        validate_score(row, sample)
    for key in ("required_fact_count", "original_issue_ids", "relationship_credit_types", "inventory_sha256",
                "solution_sha256", "transcript_sha256", "audit_sha256", "actor_sha256"):
        if any(row[key] != scores[0][key] for row in scores[1:]):
            raise ValueError("whole reference credit: original group source or denominators differ")
    facts = sorted({key for row in scores for key in row["fact_rewards"]})
    means = {key: sum(row["fact_rewards"].get(key, 0) for row in scores) / GROUP_SIZE for key in facts}
    relation_means = {key: sum(row["relationship_rewards"][key] for row in scores) / GROUP_SIZE
                      for key in scores[0]["original_issue_ids"]}
    signal_counts = {channel: sum(0 < relation_means[key] < 1 for key in relation_means
        if scores[0]["relationship_credit_types"][key] == credit_type)
        for channel, credit_type in (("native", NATIVE_ERROR_RELATIONSHIP), ("output", OUTPUT_FINALIZATION))}
    full_group = [_binding(sample) | {"score_sha256": row["score_sha256"]}
                  for sample, row in zip(samples, scores, strict=True)]
    output = {}
    for sample, row in zip(samples, scores, strict=True):
        positions = token_spans(tokenizer, sample.completion_token_ids, sample.text)
        vectors = {key: [0.] * len(sample.completion_token_ids) for key in ("fact", "native", "output")}
        entries: dict[str, list] = {"facts": [], "relationships": []}
        for component, rewards, spans, averages, count, weight in (
            ("facts", row["fact_rewards"], row["fact_spans"], means, row["required_fact_count"], FACT_WEIGHT),
            ("relationships", row["relationship_rewards"], row["relationship_spans"], relation_means,
             len(row["original_issue_ids"]), RELATIONSHIP_WEIGHT)):
            for key in sorted(spans):
                start, end = spans[key]
                indices = [] if positions is None else [index for index, (a, b) in enumerate(positions)
                                                       if start <= a < b <= end]
                advantage = rewards[key] - averages[key]
                coefficient = weight * advantage / count if count else 0.
                relationship = component == "relationships"
                if relationship:
                    coefficient = coefficient / len(indices) if indices else 0.
                target = "fact" if not relationship else "native" if row["relationship_credit_types"][key] == NATIVE_ERROR_RELATIONSHIP else "output"
                for index in indices:
                    vectors[target][index] += coefficient
                entry = {"issue_id" if relationship else "slot": key, "reward": rewards[key],
                    "group_mean": averages[key], "advantage": advantage, "coefficient": coefficient,
                    "literal_span": [start, end], "token_indices": indices}
                if relationship:
                    entry.update(ref=row["selected_refs"][key], relation=row["relationship_relations"][key],
                                 credit_type=row["relationship_credit_types"][key])
                    if key in row["relationship_later_answer_correct"]:
                        entry["later_answer_correct"] = row["relationship_later_answer_correct"][key]
                entries[component].append(entry)
        combined = [sum(values) for values in zip(*vectors.values(), strict=True)]
        output[sample.sample_id] = {"version": VERSION, "diagnostic_only": True,
            "strict_credit_applied": False, "pre_mis": True, "full_group_size": GROUP_SIZE,
            "gradient_measured": False, "optimizer_updates": 0,
            "full_group": full_group, "full_group_sha256": digest(full_group),
            "fact_token_advantages": vectors["fact"], "native_token_advantages": vectors["native"],
            "output_token_advantages": vectors["output"], "partial_token_advantages": combined,
            "native_signal": any(vectors["native"]), "output_signal": any(vectors["output"]),
            "native_reward_signal_count": signal_counts["native"], "output_reward_signal_count": signal_counts["output"],
            "native_mapped_signal_count": sum(bool(entry["token_indices"]) and entry["coefficient"] != 0
                for entry in entries["relationships"] if entry["credit_type"] == NATIVE_ERROR_RELATIONSHIP),
            "output_mapped_signal_count": sum(bool(entry["token_indices"]) and entry["coefficient"] != 0
                for entry in entries["relationships"] if entry["credit_type"] == OUTPUT_FINALIZATION),
            "fact_signal": any(vectors["fact"]), "fact_group_means": means,
            "relationship_group_means": relation_means, "original_issue_ids": row["original_issue_ids"],
            "required_fact_count": row["required_fact_count"], **entries,
            "mapping_status": "UNSTABLE_TOKEN_PREFIX" if positions is None else "MAPPED" if any(
                entry["token_indices"] for values in entries.values() for entry in values) else "NO_MAPPED_UNITS",
            "score_sha256": row["score_sha256"], **_binding(sample)}
    return output
