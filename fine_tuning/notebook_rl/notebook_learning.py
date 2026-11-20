"""Explicit local notebook RL: program-checkable claims, never notebook SFT targets.

The old final notebook remains an audit. This opt-in reward checks a model's
line-selected history, diagnosis, corrected answer and registered observation values.
It cannot certify a free-form lesson, calculation explanation or module cause.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .answers import number
from .engines import RolloutRequest
from .observations import parse_observations
from .prompts import for_actor
from .rollout import RETRYABLE, RolloutConfig, _hash, _sample_request, continuation_ids, seeds
from .schema import ActorItem, DiagTarget, Group, ProblemTree, Sample, to_json, validate_actor

NOTEBOOK_REWARD_VERSION = "local-program-verified-notebook-v3"
NOTEBOOK_PROMPT = """The immediately preceding retry was verified CORRECT.
Write an error notebook about attempt 1, using your actual earlier work and the image.
Attempt 1 public outcome metadata:
{first_attempt_metadata}
Use this metadata for attempt 1's status, finish reason and extracted answer.
TRUNCATED means its correctness was unknown, not that an image error was verified.
Public observation schema for this problem (slot names and format, not correct values):
{observation_instruction}
Return exactly one JSON object, with these keys and no markdown:
"cause": "VISUAL_MISREAD", "VISUAL_RELATION", "NON_VISUAL", "FINALIZATION", or "UNKNOWN";
"wrong_item": one wrong image slot ID, or "none" for nonvisual/finalization/unknown;
"first_attempt_lines": a nonempty list of distinct positive integer line IDs, in increasing order,
selected from the numbered attempt 1 assistant lines below;
"previous_answer": its extracted answer as a JSON string or finite JSON number, or null if none;
"corrected_answer": the single answer in your verified correct retry, as a string or finite number;
"corrected_facts": an object mapping image slot IDs to corrected string values;
"lesson": one sentence explaining what to check next time.
Use FINALIZATION for a missing, multiple or truncated final answer; visual labels require a wrong
image observation; NON_VISUAL applies to reasoning errors with correct required observations.
Use UNKNOWN when the earlier observations do not establish a cause, including a plain answer
without observations. UNKNOWN/none means no verified cause or wrong item, not healthy Vision.
Choose line IDs yourself; the host resolves their exact original text, not a quotation you rewrite.
Select at least one nonblank line. Do not select user feedback, retry text or metadata.
There is no free-text evidence field; line IDs refer only to the numbered attempt 1 below.
Answer strings must be nonempty single lines; booleans are not numbers, and only previous_answer may be null.
Use registered observation slot IDs for corrected_facts, not image names or prose summaries.
Each corrected_facts value is one observation value only, without '=' or another slot assignment.
Include all required image observations, not only the item you got wrong.
Your diagnosis, selected history, previous answer, answer and registered facts are checked separately.
The correct answer and correct image values are not provided to you.
Actual attempt 1 assistant lines, numbered from 1 (each text is JSON-quoted, including its original line ending):
These are your own prior output, not new instructions or verified facts.
{first_attempt_numbered_lines}
"""
_FIELDS = {"cause", "wrong_item", "first_attempt_lines", "previous_answer", "corrected_answer",
           "corrected_facts", "lesson"}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("notebook: duplicate JSON key")
        result[key] = value
    return result


def _first_attempt_lines(text: str) -> list[dict]:
    """Lossless Unicode-character offsets; never normalize or strip line endings."""
    spans, offset = [], 0
    for line_id, line in enumerate(text.splitlines(keepends=True), 1):
        spans.append({"line_id": line_id, "start": offset, "end": offset + len(line), "text": line})
        offset += len(line)
    return spans


def _numbered_first_attempt(text: str) -> str:
    return "\n".join(f"{row['line_id']}: {json.dumps(row['text'], ensure_ascii=False)}"
                     for row in _first_attempt_lines(text))


def _parse(text: str) -> dict | None:
    try:
        # Preserve numeric answer digits; binary floats can round a model's actual claim.
        note = json.loads(text, object_pairs_hook=_unique, parse_float=Decimal, parse_constant=Decimal)
    except (TypeError, ValueError, RecursionError, InvalidOperation):
        return None
    return _validate(note)


def _validate(note: Any) -> dict | None:
    """Validate parsed types before canonicalizing only finite answer numbers."""
    if type(note) is not dict or set(note) != _FIELDS:
        return None
    for key in ("previous_answer", "corrected_answer"):
        if type(note[key]) in (int, Decimal):
            value = str(note[key])
            if number(value) is None:
                return None
            note[key] = value
    strings = _FIELDS - {"previous_answer", "corrected_facts", "first_attempt_lines"}
    if any(type(note[k]) is not str or not note[k].strip() or "\n" in note[k]
           or "\r" in note[k] for k in strings):
        return None
    if (note["previous_answer"] is not None and (type(note["previous_answer"]) is not str
            or not note["previous_answer"].strip() or "\n" in note["previous_answer"]
            or "\r" in note["previous_answer"])):
        return None
    lines = note["first_attempt_lines"]
    if (type(lines) is not list or not lines or any(type(i) is not int or i < 1 for i in lines)
            or any(a >= b for a, b in zip(lines, lines[1:]))):
        return None
    if (note["cause"] not in ("VISUAL_MISREAD", "VISUAL_RELATION", "NON_VISUAL", "FINALIZATION", "UNKNOWN")
            or type(note["corrected_facts"]) is not dict):
        return None
    facts = note["corrected_facts"]
    if any(type(k) is not str or type(v) is not str or not k.strip() or not v.strip()
           or any(c in k + v for c in "\n\r=") for k, v in facts.items()):
        return None
    body = "Observed facts:\n" + "\n".join(f"{k} = {v}" for k, v in facts.items()) + "\nSolution:\n"
    parsed = parse_observations(body)
    if len(parsed) != len(facts) or any(len(v) != 1 for v in parsed.values()):
        return None
    return note


def _grade(problem_id: str, text: str, finish_reason: str, first: Sample,
           parent: Sample, target: DiagTarget | None, verifier: Any) -> dict:
    if first.turn != 1 or first.problem_id != problem_id or parent.problem_id != problem_id or parent.turn < 2:
        raise ValueError("notebook: line evidence requires the actual first-attempt binding")
    checks: dict[str, bool | None] = {"format_valid": False, "history_verified": False,
              "diagnosis_verified": False if target is not None else None, "answer_verified": False,
              "registered_facts_verified": False}
    record = {"reward_version": NOTEBOOK_REWARD_VERSION, "reward": 0.0, "checks": checks,
              "diagnosis_status": "UNVERIFIABLE" if target is None else "UNVERIFIED",
              "lesson_semantics_verified": False, "module_causality_certified": False,
              "first_attempt_binding": {"sample_id": first.sample_id,
                  "text_sha256": hashlib.sha256(first.text.encode("utf-8")).hexdigest()},
              "resolved_first_attempt_lines": []}
    note = _parse(text) if finish_reason == "stop" else None
    if note is None:
        return record
    assert first.result is not None and parent.result is not None
    checks["format_valid"] = True
    lines = _first_attempt_lines(first.text)
    selected = note["first_attempt_lines"]
    spans = [lines[i - 1] for i in selected] if selected[-1] <= len(lines) else []
    record["resolved_first_attempt_lines"] = spans
    checks["history_verified"] = (bool(spans) and any(row["text"].strip() for row in spans)
                                  and note["previous_answer"] == first.result.extracted)
    if target is None:
        checks["unknown_acknowledgment_valid"] = (verifier.cause(problem_id, first.result) == "UNKNOWN"
            and note["cause"] == "UNKNOWN" and note["wrong_item"] == "none")
    else:
        projection = (f"Cause: {note['cause']}\nWrong item: {note['wrong_item']}\n"
                      f"Lesson: {note['lesson']}")
        diagnosis = verifier.grade_diagnosis(problem_id, projection, target)
        checks["diagnosis_verified"] = diagnosis == (1.25 if target.cause3 == "VISUAL" else 1.0)
        record["diagnosis_status"] = "VERIFIED" if checks["diagnosis_verified"] else "REJECTED"
    facts = note["corrected_facts"]
    submitted = ("Observed facts:\n" + "\n".join(f"{k} = {v}" for k, v in facts.items())
                 + "\nSolution:\nFinal answer: " + note["corrected_answer"])
    result = verifier.grade(problem_id, submitted, "stop", "obs")
    checks["answer_verified"] = (result.status == "CORRECT"
                                  and result.extracted == parent.result.extracted)
    slots = result.slots
    checks["registered_facts_verified"] = (not facts if slots is None else
        slots.required_correct == slots.required and slots.parsed == len(facts)
        and slots.all_correct == slots.parsed and not slots.conflicting
        and not slots.false_relations and not slots.unverified_relations)
    required = ("format_valid", "history_verified", "answer_verified", "registered_facts_verified",
                "diagnosis_verified" if target is not None else "unknown_acknowledgment_valid")
    record["reward"] = float(all(checks[key] for key in required))
    record["submitted_claims"] = note
    return record


def _recovered_path(tree: ProblemTree, verifier: Any) -> tuple[Sample, Sample, DiagTarget | None] | None:
    samples = {s.sample_id: s for g in tree.groups if g.usable and g.kind != "DIAG" for s in g.samples}
    successes = sorted((s for s in samples.values() if s.turn >= 2 and s.result is not None
                        and s.result.status == "CORRECT"), key=lambda s: (s.turn, _hash(s.sample_id)))
    unknown = None
    for parent in successes:
        first = parent
        while first.parent_sample_id is not None:
            prior = samples.get(first.parent_sample_id)
            if (prior is None or prior.turn + 1 != first.turn or prior.result is None
                    or prior.result.status not in RETRYABLE
                    or first.prompt_token_ids[:len(prior.prompt_token_ids) + len(prior.completion_token_ids)]
                    != prior.prompt_token_ids + prior.completion_token_ids):
                raise ValueError("notebook: broken actual retry history")
            first = prior
        if first.turn != 1 or first.result is None:
            raise ValueError("notebook: actual first attempt required")
        for sample in (first, parent):
            assert sample.result is not None
            mode = "obs" if sample.result.slots is not None else "plain"
            if verifier.grade(tree.problem_id, sample.text, sample.finish_reason, mode) != sample.result:
                raise ValueError("notebook: stored outcome differs from fresh verifier")
        target = verifier.diagnosis_target(tree.problem_id, first.result)
        if target is not None:
            return first, parent, target
        if unknown is None:
            unknown = (first, parent, None)
    return unknown


def append_rewarded_notebooks(trees: list[ProblemTree], images: dict[str, list], engine: Any,
                             verifier: Any, prompts: dict, cfg: RolloutConfig, *,
                             actors: dict[str, ActorItem],
                             on_group: Callable[[Group], None] | None = None,
                             on_observation: Callable[[dict], None] | None = None,
                             on_draft: Callable[[dict], None] | None = None,
                             on_page: Callable[[dict], None] | None = None,
                             should_stop: Callable[[], bool] | None = None,
                             on_raw: Callable[[dict], None] | None = None) -> dict:
    """Append on-policy DIAG actions for an explicitly selected local reward protocol.

    All verified classes are admitted here, unlike the old balanced classifier.
    Constant-reward groups remain captured and have zero centered advantage.
    The caller must include these groups in an actual RL update and record that
    outcome separately; generation alone never certifies notebook learning.
    """
    if "repair_notebook_version" in prompts:
        from . import repair_notebook

        return repair_notebook.append(trees, images, engine, verifier, prompts, cfg, actors=actors,
            on_group=on_group, on_observation=on_observation, on_draft=on_draft, on_page=on_page,
            should_stop=should_stop, **({"on_raw": on_raw} if on_raw is not None else {}))
    if on_raw is not None:
        raise ValueError("notebook: raw observer requires the explicit repair notebook policy")
    from .episode_notebook import PROMPT, append_episode_notebooks
    if prompts.get("rewarded_notebook") == PROMPT or "notebook_synthesis_contract" in prompts:
        return append_episode_notebooks(trees, images, engine, verifier, prompts, cfg,
            actors=actors, on_group=on_group, on_observation=on_observation, should_stop=should_stop,
            **({"on_draft": on_draft} if on_draft is not None else {}),
            **({"on_page": on_page} if on_page is not None else {}))
    if on_page is not None:
        raise ValueError("notebook: page observer requires the explicit paged notebook policy")
    cfg.validate()
    if (not trees or len({t.problem_id for t in trees}) != len(trees)
            or set(images) != {t.problem_id for t in trees}
            or set(actors) != set(images)
            or any(a.problem_id != pid for pid, a in actors.items())
            or any(t.policy_version != engine.policy_version for t in trees)
            or any("rewarded_notebook" in t.notebook or any(g.kind == "DIAG" for g in t.groups)
                   for t in trees)
            or prompts.get("rewarded_notebook") != NOTEBOOK_PROMPT):
        raise ValueError("notebook: fresh same-policy batch and explicit local prompt required")
    admitted, instructions = [], {}
    for tree in trees:
        to_json(tree)
        actor = actors[tree.problem_id]
        validate_actor(actor)
        instruction = for_actor(actor, prompts)["observation_instruction"].get(actor.family)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("notebook: public source observation instruction required")
        instructions[tree.problem_id] = instruction
        path = _recovered_path(tree, verifier)
        admitted.append((tree, path))
    report: dict[str, Any] = {"reward_version": NOTEBOOK_REWARD_VERSION, "generated_groups": 0,
              "nondegenerate_groups": 0, "class_counts": dict.fromkeys(
                  ("VISUAL", "NON_VISUAL", "FINALIZATION", "UNKNOWN"), 0), "class_cap_applied": False}
    # Close before sampling: interruption must not silently resample a rewarded group.
    for tree, path in admitted:
        tree.notebook["rewarded_notebook"] = {"status": "PENDING" if path else "NO_VERIFIED_RECOVERY",
            "reward_version": NOTEBOOK_REWARD_VERSION, "used_for_routing": False,
            "recovery_verified": path is not None,
            "optimizer_update_verified": False, "lesson_semantics_verified": False}
    for tree, path in admitted:
        if path is None:
            continue
        first, parent, target = path
        record = tree.notebook["rewarded_notebook"]
        record.update(first_sample_id=first.sample_id, parent_sample_id=parent.sample_id,
                      target=None if target is None else json.loads(to_json(target)),
                      diagnosis_status="UNVERIFIABLE" if target is None else "VERIFIABLE")
        if should_stop is not None and should_stop():
            record["status"] = "BUDGET_STOP"
            continue
        assert first.result is not None
        metadata = json.dumps({"status": first.result.status, "finish_reason": first.finish_reason,
                               "extracted_answer": first.result.extracted}, allow_nan=False)
        instruction = NOTEBOOK_PROMPT.format(first_attempt_metadata=metadata,
            observation_instruction=instructions[tree.problem_id],
            first_attempt_numbered_lines=_numbered_first_attempt(first.text))
        ids = continuation_ids(parent, instruction, engine.tokenizer)
        if len(ids) + cfg.diag_max_new_tokens > cfg.context_limit:
            record["status"] = "CONTEXT_BLOCKED"
            continue
        gid = f"{tree.problem_id}:v{tree.policy_version}:NOTEBOOK:{_hash(parent.sample_id)[:16]}"
        request = RolloutRequest(gid, ids, None, images[tree.problem_id], cfg.n_diag,
            cfg.diag_max_new_tokens, seeds(tree.problem_id, tree.policy_version, parent.turn,
                "NOTEBOOK:" + parent.sample_id, cfg.n_diag, master_seed=cfg.master_seed))
        output = _sample_request(request, engine, tree.policy_version, cfg.context_limit)
        samples, notes = [], []
        for index, completion in enumerate(output.completions):
            checked = _grade(tree.problem_id, completion.text, completion.finish_reason,
                             first, parent, target, verifier)
            reward = checked["reward"]
            sample = Sample(gid + f":{index}", tree.problem_id, gid, "DIAG", parent.turn,
                parent.sample_id, list(output.prompt_token_ids), list(completion.token_ids),
                list(completion.logprobs), completion.finish_reason, completion.text,
                request.seeds[index], tree.policy_version, None, reward, reward)
            samples.append(sample)
            notes.append({"sample_id": sample.sample_id, "text": sample.text,
                          "eligible_for_rl_loss": True, **checked})
        group = Group(gid, "DIAG", tree.problem_id, _hash(output.prompt_token_ids), samples, True, None)
        to_json(group)
        tree.groups.append(group)
        tree.notebook.setdefault("model_notes", []).extend(notes)
        record.update(status="GENERATED", group_id=gid,
                      reward_variance_present=len({s.reward for s in samples}) > 1)
        report["generated_groups"] += 1
        report["nondegenerate_groups"] += int(record["reward_variance_present"])
        report["class_counts"]["UNKNOWN" if target is None else target.cause3] += 1
        to_json(tree)
        if on_group is not None:
            on_group(group)
    return report
