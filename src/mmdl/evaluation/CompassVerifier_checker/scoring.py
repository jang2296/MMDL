"""Validate saved judge probabilities and preserve every V8-correct result."""

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from numbers import Real
import re

from mmdl.runtime.artifacts import digest

from . import CHECKER_VERSION, SCORING_ID
from .prompt import CV_PROMPT_SHA256
from .selection import CATEGORIES, TARGET_CATEGORIES, categorize, gold_texts

VERDICTS = ("ACCEPT", "UNSURE", "REJECT", "INVALID")


def _probability(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return float(value)


def verdict(probabilities: Mapping, label_token_ids: Mapping[str, int], threshold: float) -> str:
    """Derive a verdict from full-vocabulary probabilities, without renormalizing."""
    threshold = _probability(threshold, "threshold")
    if set(label_token_ids) != {"A", "B", "C"}:
        raise ValueError("label_token_ids must contain exactly A, B, and C")
    ids = list(label_token_ids.values())
    if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != 3:
        raise ValueError("label token IDs must be distinct nonnegative integers")
    try:
        values = {label: _probability(probabilities[f"p_{label.lower()}"], f"p_{label.lower()}")
                  for label in "ABC"}
        top_id, top_token = probabilities["top_token_id"], probabilities["top_token"]
    except KeyError as error:
        raise ValueError(f"missing judge probability field: {error.args[0]}") from error
    # Three separately rounded float32 softmax entries may sum slightly above one.
    if sum(values.values()) > 1 + 3e-7:
        raise ValueError("A/B/C probabilities exceed full-vocabulary probability mass")
    if type(top_id) is not int or top_id < 0 or not isinstance(top_token, str):
        raise ValueError("invalid top token ID or text")
    label = next((name for name, token_id in label_token_ids.items() if token_id == top_id), None)
    if (label is not None and top_token != label) or (label is None and top_token in "ABC"
                                                    and len(top_token) == 1):
        raise ValueError("top token text and label token ID disagree")
    if label is not None and values[label] < max(values.values()):
        raise ValueError("top label probability is lower than another label")
    if label == "A":
        return "ACCEPT" if values["A"] >= threshold else "UNSURE"
    if label is None:
        return "UNSURE"
    return {"B": "REJECT", "C": "INVALID"}.get(label, "UNSURE")


def item_verdict(alias_results: Sequence[Mapping]) -> dict:
    """Accept any accepted alias; resolve equal probabilities by alias index."""
    if not alias_results:
        raise ValueError("an item needs at least one alias result")
    indices = set()
    for result in alias_results:
        index = result.get("alias_index")
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError("invalid or duplicate alias_index")
        indices.add(index)
        _probability(result.get("p_a"), "p_a")
        if result.get("verdict") not in VERDICTS:
            raise ValueError("invalid alias verdict")
    accepted = [result for result in alias_results if result["verdict"] == "ACCEPT"]
    return dict(min(accepted or alias_results,
                    key=lambda result: (-result["p_a"], result["alias_index"])))


def _indexed(records: Sequence[Mapping], name: str) -> dict:
    indexed = {}
    for record in records:
        item_id, alias_index = record.get("id"), record.get("alias_index")
        if not isinstance(item_id, str) or not item_id or type(alias_index) is not int \
                or alias_index < 0:
            raise ValueError(f"invalid {name} ID or alias_index")
        key = item_id, alias_index
        if key in indexed:
            raise ValueError(f"duplicate {name}: {key}")
        indexed[key] = record
    return indexed


def merge(rows: Sequence[dict], targets: Sequence[dict], verdicts: Sequence[dict],
          config: dict) -> list[dict]:
    """Require a complete, exact alias match and rederive every saved verdict."""
    row_ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in row_ids) \
            or len(set(row_ids)) != len(row_ids):
        raise ValueError("prediction IDs must be nonempty and unique")
    target_map, verdict_map = _indexed(targets, "target"), _indexed(verdicts, "verdict")
    expected = {(row["id"], index) for row in rows if categorize(row) in TARGET_CATEGORIES
                for index in range(len(gold_texts(row)))}
    if set(target_map) != expected:
        raise ValueError("targets do not match the complete eligible prediction aliases")
    if set(verdict_map) != expected:
        raise ValueError("verdicts do not match the complete eligible prediction aliases")
    alias_results: dict[str, list[dict]] = {}
    for row in rows:
        if type(row.get("correct")) is not bool:
            raise ValueError("V8 correctness must be boolean")
        category = categorize(row)
        if category not in TARGET_CATEGORIES:
            continue
        for index, gold in enumerate(gold_texts(row)):
            key = row["id"], index
            target, result = target_map[key], verdict_map[key]
            if target.get("question") != row["prompt"] or target.get("gold_answer") != gold \
                    or target.get("category") != category:
                raise ValueError(f"target disagrees with prediction: {key}")
            expected_sha = digest([CHECKER_VERSION, CV_PROMPT_SHA256, target["question"],
                                   target["gold_answer"], target.get("llm_response")])
            if target.get("input_sha256") != expected_sha \
                    or result.get("input_sha256") != expected_sha:
                raise ValueError(f"input_sha256 mismatch: {key}")
            checked = dict(result)
            checked["verdict"] = verdict(result, config["label_token_ids"],
                                         config["decision"]["accept_threshold"])
            alias_results.setdefault(row["id"], []).append(checked)
    merged = []
    for row in rows:
        aliases = alias_results.get(row["id"], [])
        decision = item_verdict(aliases) if aliases else None
        final_correct = row["correct"] or (decision is not None and decision["verdict"] == "ACCEPT")
        if row["correct"] and not final_correct:
            raise ValueError("V8-correct results cannot become incorrect")
        merged.append({
            "id": row["id"], "subject": row["subject"], "question_type": row["question_type"],
            "finish_reason": row["finish_reason"], "category": categorize(row),
            "v8_parse_status": row["parse_status"], "v8_correct": row["correct"],
            "checker_verdict": decision["verdict"] if decision else None,
            "p_a": decision["p_a"] if decision else None, "final_correct": final_correct,
            "response_truncated": any(target_map[(row["id"], result["alias_index"])].get(
                "response_truncated", False) for result in aliases),
        })
    return merged


def summarize(merged: Sequence[dict], identity: Mapping) -> dict:
    """Keep count diagnostics for smoke mode, explicitly marked unreportable."""
    mode, reportable = identity["mode"], identity["reportable"]
    if mode not in {"smoke", "official"} or type(reportable) is not bool \
            or reportable != (mode == "official"):
        raise ValueError("only official mode can be reportable")
    transitions = {key: 0 for key in ("0->0", "0->1", "1->1", "1->0")}
    categories: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    subjects: dict[str, dict[str, int]] = {}
    seen = set()
    for row in merged:
        if row["id"] in seen:
            raise ValueError("duplicate merged prediction ID")
        seen.add(row["id"])
        before, after = row["v8_correct"], row["final_correct"]
        if type(before) is not bool or type(after) is not bool or (before and not after):
            raise ValueError("merged correctness violates one-directional scoring")
        if row["category"] not in CATEGORIES:
            raise ValueError("unknown merged category")
        decision = row["checker_verdict"]
        if (row["category"] in TARGET_CATEGORIES and decision not in VERDICTS) \
                or (row["category"] not in TARGET_CATEGORIES and decision is not None):
            raise ValueError("merged verdict does not match target category")
        if after != (before or decision == "ACCEPT"):
            raise ValueError("final correctness does not match checker verdict")
        transitions[f"{int(before)}->{int(after)}"] += 1
        categories[row["category"]] += 1
        if decision is not None:
            decisions[decision] += 1
        subject = subjects.setdefault(row["subject"], {"n": 0, "v8_correct": 0, "final_correct": 0})
        subject["n"] += 1
        subject["v8_correct"] += int(before)
        subject["final_correct"] += int(after)
    notes = ["one-directional: only V8-wrong items can change", "judge sees the gold answer",
             "V8-correct items are not independently reviewed",
             "V8+checker is supplementary; the assignment score remains V8"]
    if not reportable:
        notes.append("smoke outputs are diagnostics only, not reportable scores")
    return {
        "scoring_id": SCORING_ID, "checker_version": CHECKER_VERSION,
        "mode": mode, "reportable": reportable, "denominator": len(merged),
        "v8_correct": transitions["1->1"],
        "final_correct": transitions["1->1"] + transitions["0->1"],
        "transitions": transitions, "categories": {name: categories[name] for name in CATEGORIES},
        "verdicts": {name: decisions[name] for name in VERDICTS},
        "truncated_inputs": sum(bool(row["response_truncated"]) for row in merged),
        "by_subject": dict(sorted(subjects.items())),
        "identity": {key: identity[key] for key in (
            "source_predictions_sha256", "v8_scoring_sha256", "config_sha256", "prompt_sha256", "judge")},
        "notes": notes,
        **{key: identity[key] for key in ("scope", "source_denominator") if key in identity},
    }


def _fenced(text: str) -> str:
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", text)), default=0))
    return f"{fence}\n{text}\n{fence}\n"


def review_markdown(merged: Sequence[dict], rows: Sequence[dict], verdicts: Sequence[dict]) -> str:
    """Write gold answers and response tails only into the private review artifact."""
    original = {row["id"]: row for row in rows}
    sections = ["# Checker review — private\n",
                "Contains gold answers and prediction text. Keep this artifact private.\n",
                "Review every ACCEPT and UNSURE item; UNSURE remains incorrect.\n"]
    for item in merged:
        if item["checker_verdict"] not in {"ACCEPT", "UNSURE"}:
            continue
        row = original[item["id"]]
        sections.append("## Item\n" + _fenced(json.dumps({key: item[key] for key in (
            "id", "subject", "category", "checker_verdict", "p_a")}, ensure_ascii=False)))
        sections.append("Gold answer\n" + _fenced(json.dumps(row["answer"], ensure_ascii=False)))
        aliases = sorted((result for result in verdicts if result["id"] == item["id"]),
                         key=lambda result: result["alias_index"])
        sections.append("Alias probabilities\n" + _fenced(json.dumps(aliases, ensure_ascii=False,
                                                                    sort_keys=True)))
        sections.append("Response tail (up to 1,200 characters)\n" + _fenced(row["raw_response"][-1200:]))
    return "\n".join(sections)
