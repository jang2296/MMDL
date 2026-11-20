"""Paired external B2 monitoring and length-based planning; never training targets."""
from __future__ import annotations

from collections import Counter
import math
import random

from .audit import digest
from .guard_v16b import length_distribution_v17

MONITOR_BLOCKS = (8, 16)
REASONING_SOURCES = ("hf-geometry3k", "hf-gvlqa", "hf-ssmr", "hf-chebi-role")


def monitor_due(block: int) -> bool:
    return block in MONITOR_BLOCKS


def _sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n) if n else 1.0


def _indexed(records: list[dict]) -> dict[str, dict]:
    result = {row["problem_id"]: row for row in records}
    if len(result) != len(records) or not result:
        raise ValueError("V18B_MONITOR_DUPLICATE_OR_EMPTY")
    for row in records:
        output = row["output"]
        tokens = output["generation"]["completion_token_ids"]
        if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("V18B_MONITOR_TOKEN_IDS")
        if output.get("status") not in {"CORRECT", "WRONG", "NO_PARSE", "INCORRECT", "UNPARSED",
                "INVALID_NO_FINAL", "INVALID_MULTIPLE", "UNVERIFIABLE", "TRUNCATED"}:
            raise ValueError("V18B_MONITOR_UNGRADED")
    return result


def _pair(reference: dict[str, dict], candidate: dict[str, dict], ids: list[str]) -> dict:
    a = [reference[pid]["output"] for pid in ids]
    b = [candidate[pid]["output"] for pid in ids]
    lengths_a = [len(x["generation"]["completion_token_ids"]) for x in a]
    lengths_b = [len(x["generation"]["completion_token_ids"]) for x in b]
    correct_a, correct_b = [x.get("status") == "CORRECT" for x in a], [x.get("status") == "CORRECT" for x in b]
    gain = sum(not x and y for x, y in zip(correct_a, correct_b, strict=True))
    loss = sum(x and not y for x, y in zip(correct_a, correct_b, strict=True))
    short_a = [length < 32 and row["finish_reason"] == "stop" for length, row in zip(lengths_a, a, strict=True)]
    short_b = [length < 32 and row["finish_reason"] == "stop" for length, row in zip(lengths_b, b, strict=True)]
    added = sum(not x and y for x, y in zip(short_a, short_b, strict=True))
    removed = sum(x and not y for x, y in zip(short_a, short_b, strict=True))
    n = len(ids)
    return {"n": n, "reference_correct": sum(correct_a), "candidate_correct": sum(correct_b),
        "recovered": gain, "lost": loss, "net_change": gain - loss, "p_value": _sign_p(gain, loss),
        "reference_lengths": length_distribution_v17(lengths_a),
        "candidate_lengths": length_distribution_v17(lengths_b),
        "short_added": added, "short_removed": removed,
        "short_delta": (added - removed) / n if n else 0, "short_p_value": _sign_p(added, removed),
        "reference_natural_short": sum(short_a), "candidate_natural_short": sum(short_b),
        "reference_truncated": sum(x["finish_reason"] == "length" for x in a),
        "candidate_truncated": sum(x["finish_reason"] == "length" for x in b)}


def monitor_comparison(f0_records: list[dict], post_prep_records: list[dict],
                       candidate_records: list[dict], items: list[dict], *, warning_only: bool = False) -> dict:
    """Compare matched draws; bounded short diagnostics can only produce warnings."""
    baseline, post, current = map(_indexed, (f0_records, post_prep_records, candidate_records))
    by_id = {x["problem_id"]: x for x in items}
    if (len(by_id) != len(items) or set(baseline) != set(post) or set(baseline) != set(current)
            or set(baseline) != set(by_id)):
        raise ValueError("V18B_MONITOR_IDS_CHANGED")
    for pid, item in by_id.items():
        if item.get("training_allowed") is not False or item.get("benchmark_derived"):
            raise ValueError("V18B_MONITOR_TRAINING_LEAK")
        # Sampling contracts and actual input tokens must match, while policy versions intentionally differ.
        reference = baseline[pid]
        for row in (post[pid], current[pid]):
            for field in ("seed", "sampling", "max_new_tokens", "prompt"):
                if row.get(field) != reference.get(field):
                    raise ValueError("V18B_MONITOR_REQUEST_CHANGED:" + field)
            for field in ("prompt_token_ids", "sampling", "max_tokens", "seed", "image_sha256", "input_sha256"):
                if row["output"]["generation"].get(field) != reference["output"]["generation"].get(field):
                    raise ValueError("V18B_MONITOR_GENERATION_CHANGED:" + field)
    ids = sorted(by_id)
    decision_ids = ids
    if not decision_ids:
        raise ValueError("V18B_MONITOR_NO_DECISION_ITEMS")
    against_f0 = _pair(baseline, current, decision_ids)
    against_prep = _pair(post, current, decision_ids)
    sources = {s: {"against_f0": _pair(baseline, current, [p for p in ids if by_id[p]["source"] == s]),
                   "against_post_prep": _pair(post, current, [p for p in ids if by_id[p]["source"] == s])}
               for s in sorted({x["source"] for x in items})}
    reasons = []
    warnings = []
    if not warning_only and against_f0["net_change"] < 0 and against_f0["p_value"] < .05:
        reasons.append("MONITOR_DEGRADED")
    drift_sources = [source for source, values in sources.items()
                     if all(x["short_delta"] >= .15 and x["short_p_value"] < .05 for x in values.values())]
    if (all(x["short_delta"] >= .15 and x["short_p_value"] < .05 for x in (against_f0, against_prep))
            or warning_only and drift_sources):
        warnings.append("B2_SHORT_DRIFT")
    return {"version": "v18b-short-monitor-v1" if warning_only else "v18b-accuracy-monitor-v1",
        "ids_sha256": digest(ids), "decision_ids_sha256": digest(decision_ids),
        "status": "STOP_REQUESTED" if reasons else "CONTINUE", "reasons": reasons, "warnings": warnings,
        "against_f0": against_f0, "against_post_prep": against_prep,
        "all_items": _pair(baseline, current, ids),
        "sources": sources, "short_drift_sources": drift_sources,
        "generation_n": len(ids), "decision_n": len(decision_ids),
        "excluded_ceiling_source": None, "excluded_ceiling_n": 0,
        "ceiling_generation_policy": "ALL_FROZEN_MONITOR_SOURCES_INCLUDED",
        "thresholds": {"short_delta": .15, "paired_p_lt": .05, "references_required": ["F0", "post-prep"],
            "short_drift_action": "WARNING", "stop_rule": "NONE" if warning_only else "F0_PAIRED_NET_NEGATIVE_AND_P_LT_0_05"},
        "false_alarm_calibration": "UNMEASURED_SINGLE_F0", "training_allowed": False,
        "effect_measured": False, "auto_evaluate": False}


def short_monitor_comparison(f0_records: list[dict], post_prep_records: list[dict],
                             candidate_records: list[dict], items: list[dict]) -> dict:
    for row in f0_records + post_prep_records + candidate_records:
        cap = row["output"]["generation"].get("max_tokens")
        if cap is not None and cap != 64:
            raise ValueError("V18B_SHORT_MONITOR_CAP_CHANGED")
    return monitor_comparison(f0_records, post_prep_records, candidate_records, items, warning_only=True)


def monitor_stop_decision(report: dict, action: str, current_checkpoint: dict,
                          previous_normal_checkpoint: dict | None) -> dict:
    """Select only the previously registered normal boundary, never the best score seen."""
    if action not in {"evaluate_checkpoint", "stop"}:
        raise ValueError("V18B_MONITOR_STOP_ACTION_REQUIRED")
    stopped = report.get("status") == "STOP_REQUESTED"
    if stopped and action == "evaluate_checkpoint" and not previous_normal_checkpoint:
        raise ValueError("V18B_PREVIOUS_NORMAL_CHECKPOINT_REQUIRED")
    return {"status": "STOP_REQUESTED" if stopped else "CONTINUE", "action": action,
        "checkpoint": previous_normal_checkpoint if stopped and action == "evaluate_checkpoint" else None,
        "current_checkpoint": current_checkpoint, "previous_normal_checkpoint": previous_normal_checkpoint,
        "evaluate_requested": bool(stopped and action == "evaluate_checkpoint"), "auto_evaluate": False,
        "selection_rule": "PREVIOUS_NORMAL_BOUNDARY"}


def project_mmmu_lengths(f0_records: list[dict], candidate_records: list[dict], items: list[dict], *,
                         reference_mmmu_tokens: int, throughput_tokens_per_second: float,
                         hourly_usd: float, elapsed_seconds: float = 0, remaining_train_seconds: float = 0,
                         post_eval_seconds: float = 0, time_limit_seconds: float = 14 * 3600,
                         budget_usd: float = 22, evaluation_pods: int = 1,
                         second_pod_overhead_seconds: float = 0, second_pod_hourly_usd: float | None = None,
                         bootstrap_trials: int = 2000,
                         seed: int = 3407) -> dict:
    """Bootstrap source-paired length ratios; historical MMMU total is a planning scalar only."""
    second_rate = hourly_usd if second_pod_hourly_usd is None else second_pod_hourly_usd
    numbers = (throughput_tokens_per_second, hourly_usd, second_rate, elapsed_seconds, remaining_train_seconds,
               post_eval_seconds, time_limit_seconds, budget_usd, second_pod_overhead_seconds)
    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in numbers)
            or throughput_tokens_per_second <= 0 or hourly_usd <= 0 or second_rate <= 0 or type(reference_mmmu_tokens) is not int
            or reference_mmmu_tokens <= 0 or evaluation_pods not in (1, 2)
            or type(bootstrap_trials) is not int or bootstrap_trials < 100):
        raise ValueError("V18B_PROJECTION_INVALID_INPUT")
    f0, candidate = _indexed(f0_records), _indexed(candidate_records)
    by_id = {x["problem_id"]: x for x in items}
    if set(f0) != set(candidate) or set(f0) != set(by_id):
        raise ValueError("V18B_PROJECTION_IDS_CHANGED")
    groups: dict[str, list[tuple[int, int]]] = {}
    for pid in sorted(f0):
        item = by_id[pid]
        if item.get("training_allowed") is not False or item.get("benchmark_derived"):
            raise ValueError("V18B_PROJECTION_TRAINING_LEAK")
        if item["source"] not in REASONING_SOURCES:
            continue
        groups.setdefault(item["source"], []).append((len(f0[pid]["output"]["generation"]["completion_token_ids"]),
            len(candidate[pid]["output"]["generation"]["completion_token_ids"])))
    if not groups or any(sum(a for a, _ in values) == 0 for values in groups.values()):
        raise ValueError("V18B_PROJECTION_NO_REASONING_LENGTHS")
    rng = random.Random(seed)
    def ratio(sample):
        return sum(b for values in sample.values() for _, b in values) / sum(a for values in sample.values() for a, _ in values)
    point = ratio(groups)
    draws = sorted(ratio({s: rng.choices(v, k=len(v)) for s, v in groups.items()}) for _ in range(bootstrap_trials))
    low, high = draws[int(.025 * bootstrap_trials)], draws[min(bootstrap_trials - 1, int(.975 * bootstrap_trials))]
    # shortcut: heldout length ratios do not certify MMMU transfer; retain the full B2-length scenario.
    ratios = {"lower": low, "point": point, "upper": max(1.0, high)}
    scenarios = {}
    for name, value in ratios.items():
        tokens = math.ceil(reference_mmmu_tokens * value)
        gpu_seconds = tokens / throughput_tokens_per_second
        evaluation_wall = gpu_seconds / evaluation_pods + (second_pod_overhead_seconds if evaluation_pods == 2 else 0)
        wall = elapsed_seconds + remaining_train_seconds + evaluation_wall + post_eval_seconds
        cost = hourly_usd * (elapsed_seconds + remaining_train_seconds + gpu_seconds / evaluation_pods + post_eval_seconds) / 3600
        if evaluation_pods == 2:
            cost += second_rate * (gpu_seconds / 2 + second_pod_overhead_seconds) / 3600
        scenarios[name] = {"tokens": tokens, "evaluation_seconds": evaluation_wall,
            "total_seconds": wall, "estimated_usd": cost,
            "within_time": wall <= time_limit_seconds, "within_budget": cost <= budget_usd}
    return {"version": "v18b-block16-mmmu-projection-v1", "length_ratio": point,
        "ratio_bootstrap_95": [low, high], "scenarios": scenarios, "source_n": dict(Counter({s: len(v) for s, v in groups.items()})),
        "reference_mmmu_tokens": reference_mmmu_tokens, "throughput_tokens_per_second": throughput_tokens_per_second,
        "evaluation_pods": evaluation_pods, "second_pod_hourly_usd": second_rate if evaluation_pods == 2 else None,
        "cost_basis": "FUTURE_HANDOFF_SCENARIO; actual lease-clock billing is enforced separately", "seed": seed, "bootstrap_trials": bootstrap_trials,
        "uncertainty": "DESCRIPTIVE_SOURCE_STRATIFIED_BOOTSTRAP; MMMU_TRANSFER_AND_FUTURE_SPEED_UNMEASURED",
        "training_allowed": False, "auto_launch": False}


def supply_report(records: dict[str, dict], ordered_items: list[dict], update_results: list[dict],
                  consumed_row_ids: list[str], *, slot: int, block_size: int = 16) -> dict:
    """Ground supply by actual row evidence, distinguishing opportunity, closure and saved updates."""
    from .v18b_rows import observation_baseline_v18b
    consumed = set(consumed_row_ids)
    saved = {}
    for result in update_results:
        stats = result["result"]["stats"]
        if not result["result"].get("optimizer_steps"):
            continue
        for row in stats.get("n_parent_evidence", []):
            if (row.get("gradient_finite_nonzero") is True and row.get("note_gradient_retained") is True
                    and row.get("supervised_tokens", 0) > 0 and row["row_id"] in consumed):
                saved[row["row_id"]] = row | {"global_slot": result["global_slot"]}
    parents = []
    for position, item in enumerate(ordered_items):
        pid = item["problem_id"]
        record = records.get(pid, {})
        note = record.get("note_receipt") or {}
        request = record.get("note_request_receipt") or {}
        stages = [stage for stage, field in (("T1", "t1_ledger"), ("OBS", "obs_ledger"))
            if any(c.get("verdict") == "CONTRADICTED" for c in (record.get(field) or {}).get("claims", []))]
        due = item.get("block_index", position // block_size)
        mature = type(due) is int and slot >= due + 3 and record.get("closed") is True
        rid = pid + ":N"
        evidence = saved.get(rid)
        parents.append({"problem_id": pid, "source": item["source"], "opportunity_stages": stages,
            "opportunity_mature": bool(stages and mature), "note_generated": "note" in record,
            "note_condition": note.get("note_condition", request.get("note_condition")),
            "actual_hint_exposed": note.get("actual_hint_exposed", request.get("actual_hint_exposed", False)),
            "n_accepted": note.get("n_accepted", note.get("accepted")) is True,
            "n_consumed": rid in consumed, "n_saved_update_linked": evidence is not None,
            "n_supervised_tokens": evidence["supervised_tokens"] if evidence else 0,
            "n_row_id": rid if rid in consumed else None, "n_update_slot": evidence["global_slot"] if evidence else None,
            "status": "NOT_GENERATED" if "t1" not in record else "PENDING" if stages and not mature else "NO_OPPORTUNITY" if not stages else
                "SAVED_UPDATE_LINKED" if evidence else "UPDATE_UNLINKED" if note.get("n_accepted", note.get("accepted"))
                else "REJECTED", "note_reasons": note.get("reasons", [])})
    def counts(rows):
        return {"parents": len(rows), "opportunity_t1": sum("T1" in r["opportunity_stages"] for r in rows),
            "opportunity_obs": sum("OBS" in r["opportunity_stages"] for r in rows),
            "mature_opportunities": sum(r["opportunity_mature"] for r in rows),
            "n_accepted": sum(r["n_accepted"] for r in rows), "n_consumed": sum(r["n_consumed"] for r in rows),
            "n_saved_update_linked": sum(r["n_saved_update_linked"] for r in rows),
            "n_supervised_tokens": sum(r["n_supervised_tokens"] for r in rows),
            "states": dict(Counter(r["status"] for r in rows))}
    return {"version": "v18b-parent-supply-v1", "slot": slot, "parents": parents, "overall": counts(parents),
        "graph_observation_baseline": observation_baseline_v18b([
            rec for pid, rec in records.items() if pid + ":OBS" in consumed]),
        "sources": {source: counts([r for r in parents if r["source"] == source]) for source in sorted({r["source"] for r in parents})},
        "gradient_attribution": "PER_PARENT_EVIDENCE_ONLY; joint update alignment is not source attribution",
        "effect_measured": False}
