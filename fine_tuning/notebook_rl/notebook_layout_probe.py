"""Paired notebook-input layout diagnosis; no training, repairs, or reward changes."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from . import episode_notebook as notebook, fresh_observation
from .engines import RolloutRequest, _completion
from .notebook_learning import _numbered_first_attempt
from .prompts import for_actor, p0_body, user_message
from .rollout import _hash, seeds
from .schema import ActorItem, ProblemTree

VERSION = "local-nf4-notebook-layout-probe-v1"
TRAINED_VERSION = "local-nf4-postupdate-same-history-notebook-v1"
CONDITIONS = ("CURRENT_JSON", "NUMBERED_LINES")
MAX_SECONDS = 3600
MAX_NEW_TOKENS = 2048
CONTEXT_LIMIT = 16384


def contract(protocol: dict) -> dict:
    return {"protocol_id": VERSION, "runtime_protocol_sha256": protocol["sha256"],
            "conditions": list(CONDITIONS), "n": 4, "max_new_tokens": MAX_NEW_TOKENS,
            "context_limit": CONTEXT_LIMIT, "max_seconds": MAX_SECONDS,
            "prompt": notebook.PROMPT, "input_intervention": "transcript-layout-only",
            "order": "reverse-conditions-for-odd-actor-index",
            "seeds": "original-first-DIAG-four-seeds", "optimizer_updates": 0,
            "retries": False, "feedback": False, "training_allowed": False,
            "answer_accuracy_measured": False, "self_improvement_verified": False,
            "module_causality_certified": False}


def trained_contract(protocol: dict) -> dict:
    return {**contract(protocol), "protocol_id": TRAINED_VERSION,
            "conditions": ["CURRENT_JSON"], "order": "original-fixed-actor-order",
            "input_intervention": "restored post-update weights on fixed pre-update context",
            "history_policy_version": 0, "generalization_verified": False,
            "fresh_model_self_improvement_verified": False}


@dataclass(frozen=True)
class ProbeRequest:
    actor: ActorItem
    tree: ProblemTree
    condition: str
    request: RolloutRequest
    schema: str
    control_messages: list[dict]


def _escaped(text: str) -> str:
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def numbered_transcript(samples) -> str:
    """Each physical source line remains a reversible, separately quoted string."""
    rows = []
    for row, sample in zip(notebook.public_transcript(samples), samples, strict=True):
        header = {key: value for key, value in row.items() if key != "lines"}
        header["line_count"] = len(row["lines"])
        lines = _numbered_first_attempt(sample.text)
        # JSON permits these Unicode separators literally; escape them to keep physical lines distinct.
        for separator in ("\x85", "\u2028", "\u2029"):
            lines = lines.replace(separator, f"\\u{ord(separator):04x}")
        rows.append("ATTEMPT " + json.dumps(header, ensure_ascii=False, allow_nan=False,
                                            separators=(",", ":")) + "\n" + lines)
    return _escaped("\n".join(rows))


def numbered_instruction(samples, audit: dict, schema: str) -> str:
    # Change template sentences before inserting any original model text.
    template = notebook.PROMPT.replace(
        'Each input attempt\'s "source_lines" contains [line_ID, original_string] pairs, including blank lines and line endings.',
        'Each input attempt has an ATTEMPT metadata header followed by line_ID: JSON-quoted original_string lines, including blank lines and line endings.')
    template = template.replace("Complete recorded transcript (JSON, line endings preserved):",
                                "Complete recorded transcript (numbered quoted lines, line endings preserved):")
    differences = _escaped(json.dumps(notebook.public_observation_differences(samples),
                                     ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    return template.format(observation_instruction=schema, success_attempt=audit["success_attempt"],
                           transcript=numbered_transcript(samples), observation_differences=differences)


def _messages(actor, tree, schema, audit, condition):
    samples = notebook.solution_samples(tree)
    body = (notebook.instruction(samples, audit, schema) if condition == CONDITIONS[0]
            else numbered_instruction(samples, audit, schema))
    reread = tree.notebook["rewarded_notebook"]["fresh_observation"]
    return [user_message(p0_body(actor.question, actor.choices) + "\n\n"
            + fresh_observation.quoted_context(reread) + body, len(actor.image_paths))]


def paired_requests(actors, trees, images, prompts) -> dict:
    if (len(actors) != 2 or [a.problem_id for a in actors] != [t.problem_id for t in trees]
            or len({a.problem_id for a in actors}) != 2):
        raise ValueError("notebook layout: fixed matching two actors and trees required")
    requests, skipped = [], []
    for index, (actor, tree) in enumerate(zip(actors, trees, strict=True)):
        samples = notebook.solution_samples(tree)
        complete = bool(samples) and not notebook.solution_skips(tree) and all(
            g.usable for g in tree.groups if g.kind != "DIAG")
        successes = [i for i, sample in enumerate(samples, 1)
                     if sample.turn >= 2 and sample.result and sample.result.status == "CORRECT"]
        groups = [g for g in tree.groups if g.kind == "DIAG"]
        reason = ("INCOMPLETE_EPISODE" if not complete else "NO_VERIFIED_RECOVERY" if not successes
                  else "NO_FIRST_DIAG" if not groups else None)
        if reason:
            skipped.append({"problem_id": actor.problem_id, "reason": reason})
            continue
        schema = for_actor(actor, prompts)["observation_instruction"][actor.family]
        note = tree.notebook["rewarded_notebook"]
        if note.get("observation_policy") != fresh_observation.POLICY:
            raise ValueError("notebook layout: original fresh observation required")
        fresh_observation.validate(note["fresh_observation"], policy_version=tree.policy_version,
                                   actor=actor, schema=schema, master_seed=3407)
        audit = {"success_attempt": successes[-1]}
        control = _messages(actor, tree, schema, audit, CONDITIONS[0])
        group = groups[0]
        expected_seeds = seeds(actor.problem_id, tree.policy_version, samples[successes[-1]-1].turn,
                               "EPISODE_NOTEBOOK", 4, master_seed=3407)
        if (note.get("notebook_messages") != control or note.get("messages_sha256") != _hash(control)
                or len(group.samples) != 4 or not group.usable
                or [s.seed for s in group.samples] != expected_seeds
                or any(s.prompt_token_ids != group.samples[0].prompt_token_ids for s in group.samples)
                or group.state_sha256 != _hash(group.samples[0].prompt_token_ids)
                or note.get("prompt_sha256") != group.state_sha256):
            raise ValueError("notebook layout: original first-DIAG control binding differs")
        for condition in CONDITIONS[::(-1 if index % 2 else 1)]:
            request = RolloutRequest(f"{actor.problem_id}:{VERSION}:{condition}", None,
                control if condition == CONDITIONS[0] else _messages(actor, tree, schema, audit, condition),
                images[actor.problem_id], 4, MAX_NEW_TOKENS, list(expected_seeds))
            requests.append(ProbeRequest(actor, tree, condition, request, schema, control))
    return {"requests": requests, "skipped": skipped}


def _image_identity(actor, prepared) -> dict:
    from .input_preparation import tensor_identity

    return {"image_paths": list(actor.image_paths), "image_sha256": list(actor.image_sha256),
            "image_key": prepared.image_key, "processed_sizes": [list(im.size) for im in prepared.images],
            "pixel_tensors": {k: tensor_identity(v) for k, v in prepared.pixels.items()},
            "image_spans": prepared.image_spans}


def collect(requests, engine, verifier, record, deadline: float, *,
            generation_policy_version: int | None = None) -> list[dict]:
    cross_policy = generation_policy_version is not None
    if cross_policy and (type(generation_policy_version) is not int or generation_policy_version <= 0):
        raise ValueError("notebook layout: positive integer generation policy required")
    policy_version = engine.policy_version

    def check_policy():
        if type(engine.policy_version) is not int or engine.policy_version != policy_version:
            raise ValueError("notebook layout: generation policy changed")

    rows, prepared_requests, audits = [], [], {}
    identities: dict[str, dict] = {}
    for item in requests:
        if time.monotonic() >= deadline:
            raise TimeoutError("notebook layout: frozen one-hour budget exhausted")
        actor, tree, request = item.actor, item.tree, item.request
        check_policy()
        if cross_policy:
            if (type(tree.policy_version) is not int or tree.policy_version != 0
                    or policy_version != generation_policy_version or item.condition != "CURRENT_JSON"):
                raise ValueError("notebook layout: policy-zero CURRENT_JSON history and requested trained policy required")
        elif policy_version != tree.policy_version:
            raise ValueError("notebook layout: same original policy required")
        audit = notebook.audit_episode(tree, actor, verifier, routing_rule=notebook.STRICT_ROUTE_RULE)
        if not audit["complete_episode"] or audit["success_attempt"] is None:
            raise ValueError("notebook layout: complete audited recovery required")
        control = _messages(actor, tree, item.schema, audit, CONDITIONS[0])
        note = tree.notebook["rewarded_notebook"]
        group = next(g for g in tree.groups if g.kind == "DIAG")
        if (item.condition not in CONDITIONS or item.control_messages != control
                or note.get("notebook_messages") != control or note.get("messages_sha256") != _hash(control)
                or request.messages != _messages(actor, tree, item.schema, audit, item.condition)
                or request.prompt_token_ids is not None or request.n != 4
                or request.max_new_tokens != MAX_NEW_TOKENS or request.seeds != [s.seed for s in group.samples]):
            raise ValueError("notebook layout: original control/request changed")
        fresh_observation.validate(note["fresh_observation"], policy_version=tree.policy_version,
            tokenizer=engine.tokenizer, preparation=None if cross_policy else engine, images=request.images, actor=actor,
            schema=item.schema, master_seed=3407)
        if cross_policy:
            reread = note["fresh_observation"]
            prepared_reread = engine.prepare(fresh_observation.request_for(
                actor, item.schema, request.images, tree.policy_version, 3407))
            if (prepared_reread.token_ids != reread["prompt_token_ids"]
                    or prepared_reread.image_key != reread["image_key"]):
                raise ValueError("notebook layout: original fresh observation image/prompt differs")
        check_policy()
        original = engine.prepare(RolloutRequest(group.group_id, None, control, request.images,
                                                4, MAX_NEW_TOKENS, request.seeds))
        check_policy()
        if any(s.prompt_token_ids != original.token_ids for s in group.samples):
            raise ValueError("notebook layout: original first-DIAG token input differs")
        for sample in notebook.solution_samples(tree) + group.samples:
            _completion(sample.completion_token_ids, sample.sampler_logprobs, sample.finish_reason, sample.text)
            if engine.tokenizer.decode(sample.completion_token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) != sample.text:
                raise ValueError("notebook layout: original completion token decode differs")
        prepared = engine.prepare(request)
        check_policy()
        if len(prepared.token_ids) + MAX_NEW_TOKENS > CONTEXT_LIMIT:
            raise ValueError("CONTEXT_BLOCKED: no truncation permitted")
        identity = _image_identity(actor, prepared)
        if (_image_identity(actor, original) != identity
                or identities.setdefault(actor.problem_id, identity) != identity):
            raise ValueError("notebook layout: paired original image tensors changed")
        audits[actor.problem_id] = audit
        prepared_requests.append((item, prepared, identity))
    for item, prepared, identity in prepared_requests:
        if time.monotonic() >= deadline:
            raise TimeoutError("notebook layout: frozen one-hour budget exhausted")
        began, request = time.monotonic(), item.request
        check_policy()
        versions = ({"history_policy_version": item.tree.policy_version,
                     "generation_policy_version": policy_version} if cross_policy else {})
        record("CALL_STARTED", {"request_id": request.request_id, "condition": item.condition,
            "messages": request.messages, "messages_sha256": _hash(request.messages), "seeds": request.seeds,
            "n": request.n, "max_new_tokens": request.max_new_tokens,
            "prompt_token_ids": prepared.token_ids, "images": identity,
            "episode_audit": audits[item.actor.problem_id], **versions})
        check_policy()
        generated = engine.sample([request])
        check_policy()
        if (len(generated) != 1 or generated[0].request_id != request.request_id
                or generated[0].prompt_token_ids != prepared.token_ids or len(generated[0].completions) != 4):
            raise ValueError("notebook layout: incomplete or mismatched generation")
        samples = []
        for seed, completion in zip(request.seeds, generated[0].completions, strict=True):
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if (len(completion.token_ids) > MAX_NEW_TOKENS or engine.tokenizer.decode(completion.token_ids,
                    skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("notebook layout: invalid completion budget/token decode")
            checked = notebook.grade_note(completion.text, completion.finish_reason, item.tree,
                audits[item.actor.problem_id], verifier, feedback_policy=notebook.FEEDBACK_POLICY)
            baseline = {}
            if cross_policy:
                group = next(g for g in item.tree.groups if g.kind == "DIAG")
                original_sample = next(s for s in group.samples if s.seed == seed)
                baseline["baseline_notebook_check"] = notebook.grade_note(
                    original_sample.text, original_sample.finish_reason, item.tree,
                    audits[item.actor.problem_id], verifier, feedback_policy=notebook.FEEDBACK_POLICY)
            samples.append({"seed": seed, **asdict(completion), "notebook_check": checked,
                            "eligible_for_rl_loss": False, **baseline})
        check_policy()
        row = {"request_id": request.request_id, "condition": item.condition,
               "problem_id": item.actor.problem_id, "seconds": time.monotonic() - began,
               "samples": samples, **versions}
        record("CALL_COMPLETED", row)
        check_policy()
        rows.append(row)
    return rows
