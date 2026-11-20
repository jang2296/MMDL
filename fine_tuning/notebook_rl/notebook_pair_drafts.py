"""Unverified pair proposals for a later whole-history notebook, never rewarded actions."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json

from . import episode_notebook as episode, fresh_observation, notebook_pair_probe as probe
from .engines import RolloutRequest, _completion
from .notebook_layout_probe import _image_identity
from .prompts import p0_body, user_message
from .rollout import _hash, seeds
from .schema import to_json

VERSION = "unverified-notebook-pair-drafts-v1"
MAX_NEW_TOKENS, CONTEXT_LIMIT = probe.MAX_NEW_TOKENS, probe.CONTEXT_LIMIT
_ROLES = {"verified": False, "eligible_for_rl_loss": False, "used_for_routing": False}
_INPUT_FIELDS = {"request_id", "target_index", "target", "messages", "messages_sha256", "seeds", "n",
                 "max_new_tokens", "prompt_token_ids", "prompt_sha256", "image_key", "images"}
_PLAN_BLOCKS = ("pair probe: an actual direct retry edge is required",
               "pair probe: a varying slot lacks both authored endpoints on the selected edge",
               "pair probe: selected edge lacks original output endpoints")


def contract() -> dict:
    # Frozen contract wording refers to the original v1 action planner, retained below.
    return {"version": VERSION, "prompt": probe.PROMPT, "prompt_sha256": _hash(probe.PROMPT),
        "n": 4, "max_new_tokens": MAX_NEW_TOKENS, "context_limit": CONTEXT_LIMIT,
        "seed_stage": "PAIR_DRAFT:{one_based_target_index}", "seed_turn": 1,
        "selection": "unchanged pair_action._plan; first direct edge and raw-varying slots plus OUTPUT",
        "coverage": "PARTIAL", "unexamined_coverage": "other edges, stable slots and all TRACE operations",
        "proposal_selection": "none; preserve all four raw outputs, including invalid, truncated and contradictory proposals",
        "feedback": False, "retries": False, "final_notebook_required": True,
        "final_notebook_allowed_statuses": ["COMPLETE", "NO_PUBLIC_TARGETS", "PUBLIC_TARGET_BLOCKED"],
        "final_notebook_blocked_statuses": ["BUDGET_STOP", "CONTEXT_BLOCKED"],
        "assistance": "optional hints after an explicit sealed draft stage; unavailable public targets do not waive the final notebook",
        "quoted_context_serialization": "sorted JSON object keys; preserve array order and every raw response string",
        "final_coverage_authority": "unchanged original whole-history notebook checker", **_ROLES}


def _source(actor, tree, schema) -> str:
    samples = episode.solution_samples(tree)
    if (actor.problem_id != tree.problem_id or not samples or episode.solution_skips(tree)
            or any(not group.usable for group in tree.groups if group.kind != "DIAG")
            or type(tree.policy_version) is not int
            or any(sample.problem_id != tree.problem_id or sample.policy_version != tree.policy_version for sample in samples)
            or type(schema) is not str or not schema.strip()):
        raise ValueError("pair drafts: complete same-policy original solutions and public schema required")
    fields = ("sample_id", "problem_id", "group_id", "kind", "turn", "parent_sample_id", "prompt_token_ids",
              "completion_token_ids", "sampler_logprobs", "seed", "policy_version")
    return _hash({"actor": json.loads(to_json(actor)), "schema": schema, "policy_version": tree.policy_version,
        "transcript": episode.public_transcript(samples),
        "original_inputs": [{key: getattr(sample, key) for key in fields} for sample in samples]})


def _plan(samples, reread):
    # Keep the frozen draft policy independent of newer rewarded-action planners.
    try:
        return probe.select_targets(samples, reread), None
    except ValueError as exc:
        if str(exc) not in _PLAN_BLOCKS:
            raise
        return [], str(exc)


def _request(actor, tree, reread, schema, images, master_seed, index, target):
    question = p0_body(actor.question, actor.choices).removesuffix(
        "Please select the correct answer from the options above.\n")
    messages = [user_message(question + "\n\n" + fresh_observation.quoted_context(reread)
        + probe.instruction(episode.solution_samples(tree), schema, target), len(actor.image_paths))]
    return RolloutRequest(f"{actor.problem_id}:v{tree.policy_version}:PAIR_DRAFT:{index}", None,
        messages, images, 4, MAX_NEW_TOKENS,
        seeds(actor.problem_id, tree.policy_version, 1, f"PAIR_DRAFT:{index}", 4, master_seed=master_seed))


def _inputs(request, index, target, actor, prepared=None) -> dict:
    return {"request_id": request.request_id, "target_index": index, "target": copy.deepcopy(target),
        "messages": copy.deepcopy(request.messages), "messages_sha256": _hash(request.messages),
        "seeds": list(request.seeds), "n": 4, "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_token_ids": list(prepared.token_ids) if prepared is not None else None,
        "prompt_sha256": _hash(prepared.token_ids) if prepared is not None else None,
        "image_key": prepared.image_key if prepared is not None else None,
        "images": json.loads(json.dumps(_image_identity(actor, prepared))) if prepared is not None else None}


def _sealed(record):
    if (type(record) is not dict or record.get("policy") != VERSION
            or _hash(record.get("contract")) != _hash(contract())
            or any(record.get(key) is not value for key, value in _ROLES.items())
            or record.get("sha256") != _hash({key: value for key, value in record.items() if key != "sha256"})):
        raise ValueError("pair drafts: unverified role, contract or artifact binding changed")


def _completions(values, tokenizer=None):
    if type(values) is not list or len(values) != 4:
        raise ValueError("pair drafts: incomplete actual n4 generation")
    for completion in values:
        if type(completion) is not dict or set(completion) != {"token_ids", "logprobs", "finish_reason", "text"}:
            raise ValueError("pair drafts: actual raw completion shape changed")
        _completion(completion["token_ids"], completion["logprobs"], completion["finish_reason"], completion["text"])
        if (len(completion["token_ids"]) > MAX_NEW_TOKENS or tokenizer is not None and tokenizer.decode(
                completion["token_ids"], skip_special_tokens=True,
                clean_up_tokenization_spaces=False) != completion["text"]):
            raise ValueError("pair drafts: actual raw completion budget/decode changed")


def validate(record, actor, tree, reread, schema, *, tokenizer=None, preparation=None, images=None) -> None:
    _sealed(record)
    if (record.get("actor") != json.loads(to_json(actor)) or record.get("problem_id") != actor.problem_id
            or record.get("schema") != schema or record.get("policy_version") != tree.policy_version
            or record.get("context_limit") != CONTEXT_LIMIT or type(record.get("master_seed")) is not int
            or not 0 <= record["master_seed"] < 2**31 or record.get("source_sha256") != _source(actor, tree, schema)
            or record.get("reread_sha256") != _hash(reread)):
        raise ValueError("pair drafts: original source, actor, schema or reread binding changed")
    fresh_observation.validate(reread, policy_version=tree.policy_version, tokenizer=tokenizer,
        preparation=preparation, images=images, actor=actor, schema=schema, master_seed=record["master_seed"])
    plan, blocked = _plan(episode.solution_samples(tree), reread)
    calls = record.get("calls")
    if (record.get("target_plan_sha256") != _hash(plan) or _hash(record.get("target_plan")) != _hash(plan)
            or record.get("public_target_error") != blocked or type(calls) is not list or len(calls) > len(plan)):
        raise ValueError("pair drafts: public target plan changed")
    status = record.get("status")
    blocked_call = record.get("blocked_call")
    if (status == "COMPLETE" and (blocked is not None or not plan or len(calls) != len(plan) or blocked_call is not None)
            or status == "PUBLIC_TARGET_BLOCKED" and (blocked is None or plan or calls or blocked_call is not None)
            or status == "NO_PUBLIC_TARGETS" and (blocked is not None or plan or calls or blocked_call is not None)
            or status == "BUDGET_STOP" and (blocked is not None or len(calls) >= len(plan) or blocked_call is not None)
            or status == "CONTEXT_BLOCKED" and (blocked is not None or len(calls) >= len(plan) or type(blocked_call) is not dict)
            or status not in ("COMPLETE", "PUBLIC_TARGET_BLOCKED", "NO_PUBLIC_TARGETS", "BUDGET_STOP", "CONTEXT_BLOCKED")):
        raise ValueError("pair drafts: incomplete or blocked calls mislabeled complete")
    image_identity = None
    for index, call in enumerate(calls + ([blocked_call] if blocked_call is not None else []), 1):
        is_blocked = index > len(calls)
        if type(call) is not dict or set(call) != _INPUT_FIELDS | ({"reason"} if is_blocked else {"completions"}):
            raise ValueError("pair drafts: captured call shape changed")
        request = _request(actor, tree, reread, schema, images or [], record["master_seed"], index, plan[index - 1])
        expected = _inputs(request, index, plan[index - 1], actor)
        if any(_hash(call[key]) != _hash(expected[key]) for key in _INPUT_FIELDS - {
                "prompt_token_ids", "prompt_sha256", "image_key", "images"}):
            raise ValueError("pair drafts: target, prompt, seed or group binding changed")
        tokens = call["prompt_token_ids"]
        if tokens is not None:
            if (type(tokens) is not list or not tokens or any(type(token) is not int or token < 0 for token in tokens)
                    or call["prompt_sha256"] != _hash(tokens) or call["image_key"] != reread["image_key"]
                    or type(call["images"]) is not dict or call["images"].get("image_key") != call["image_key"]
                    or call["images"].get("image_paths") != list(actor.image_paths)
                    or call["images"].get("image_sha256") != list(actor.image_sha256)):
                raise ValueError("pair drafts: actual prompt tokens or original image identity changed")
            if image_identity is not None and _hash(call["images"]) != _hash(image_identity):
                raise ValueError("pair drafts: original image tensors changed between calls")
            image_identity = call["images"]
        elif not is_blocked or any(call[key] is not None for key in ("prompt_sha256", "image_key", "images")):
            raise ValueError("pair drafts: actual completed input is missing")
        if is_blocked:
            if (type(call["reason"]) is not str or not call["reason"].startswith("CONTEXT_BLOCKED:")
                    or tokens is not None and len(tokens) + MAX_NEW_TOKENS <= CONTEXT_LIMIT):
                raise ValueError("pair drafts: context blockage lacks its original reservation")
        else:
            if len(tokens) + MAX_NEW_TOKENS > CONTEXT_LIMIT:
                raise ValueError("pair drafts: over-limit actual input")
            _completions(call["completions"], tokenizer)
        if preparation is not None:
            try:
                actual = preparation.prepare(request)
            except ValueError as exc:
                if is_blocked and tokens is None and str(exc).startswith("CONTEXT_BLOCKED:"):
                    continue
                raise
            if tokens is None or actual.token_ids != tokens or _hash(_image_identity(actor, actual)) != _hash(call["images"]):
                raise ValueError("pair drafts: reconstructed original input differs")


def generate(actor, tree, reread, schema, images, engine, *, master_seed=3407, context_limit=24576,
             on_event=None, should_stop=None) -> dict:
    source = _source(actor, tree, schema)
    if (type(engine.policy_version) is not int or engine.policy_version != tree.policy_version
            or type(master_seed) is not int or not 0 <= master_seed < 2**31
            or type(context_limit) is not int or context_limit != CONTEXT_LIMIT):
        raise ValueError("pair drafts: same policy, explicit master seed and 24K context required")
    fresh_observation.validate(reread, policy_version=tree.policy_version, tokenizer=engine.tokenizer,
        preparation=engine, images=images, actor=actor, schema=schema, master_seed=master_seed)
    plan, blocked = _plan(episode.solution_samples(tree), reread)
    record: dict = {"policy": VERSION, "contract": contract(), "problem_id": actor.problem_id,
        "actor": json.loads(to_json(actor)), "schema": schema, "policy_version": tree.policy_version,
        "master_seed": master_seed, "context_limit": context_limit, "source_sha256": source,
        "reread_sha256": _hash(reread), "target_plan": copy.deepcopy(plan), "target_plan_sha256": _hash(plan),
        "public_target_error": blocked, "calls": [], "blocked_call": None,
        "status": "PUBLIC_TARGET_BLOCKED" if blocked is not None else "NO_PUBLIC_TARGETS" if not plan else "COMPLETE",
        **_ROLES}
    original_tree = to_json(tree)

    def emit(value):
        if on_event is not None:
            on_event(copy.deepcopy({"problem_id": actor.problem_id, **value}))

    for index, target in enumerate(plan, 1):
        if should_stop is not None and should_stop():
            record["status"] = "BUDGET_STOP"
            break
        request = _request(actor, tree, reread, schema, images, master_seed, index, target)
        prepared = None
        try:
            prepared = engine.prepare(request)
            if len(prepared.token_ids) + MAX_NEW_TOKENS > context_limit:
                raise ValueError("CONTEXT_BLOCKED: pair draft prompt plus full output reservation; no truncation")
        except ValueError as exc:
            if not str(exc).startswith("CONTEXT_BLOCKED:"):
                raise
            record.update(status="CONTEXT_BLOCKED", blocked_call=_inputs(request, index, target, actor, prepared) | {"reason": str(exc)})
            break
        call = _inputs(request, index, target, actor, prepared)
        emit({"phase": "STARTED", "call": call})
        if type(engine.policy_version) is not int or engine.policy_version != tree.policy_version:
            raise ValueError("pair drafts: policy changed before generation")
        outputs = engine.sample([request])
        emit({"phase": "GENERATED", "request_id": request.request_id, "target_index": index,
              "outputs": [asdict(output) for output in outputs]})
        if (len(outputs) != 1 or outputs[0].request_id != request.request_id
                or outputs[0].prompt_token_ids != prepared.token_ids or len(outputs[0].completions) != 4):
            raise ValueError("pair drafts: incomplete or mismatched actual n4 generation")
        call["completions"] = [asdict(completion) for completion in outputs[0].completions]
        _completions(call["completions"], engine.tokenizer)
        record["calls"].append(call)
        emit({"phase": "COMPLETED", "call": call})
    if (to_json(tree) != original_tree or _source(actor, tree, schema) != source
            or type(engine.policy_version) is not int or engine.policy_version != tree.policy_version):
        raise ValueError("pair drafts: original history, routing or policy changed during generation")
    record["sha256"] = _hash(record)
    validate(record, actor, tree, reread, schema, tokenizer=engine.tokenizer, preparation=engine, images=images)
    emit({"phase": "DRAFTS_COMPLETED", "record": record})
    return record


def quoted_context(record) -> str:
    _sealed(record)
    public = {"policy": VERSION, "status": record["status"], "coverage": "PARTIAL",
        "planned_targets": len(record["target_plan"]), "completed_targets": len(record["calls"]),
        "public_target_error": record["public_target_error"],
        "drafts": [{"target": call["target"], "responses": [
            {"finish_reason": completion["finish_reason"], "raw_response": completion["text"], "verified": False}
            for completion in call["completions"]]} for call in record["calls"]]}
    text = json.dumps(public, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
    return ("The following pair proposals are quoted UNVERIFIED model outputs, including every invalid, truncated or contradictory draft. "
        "They are not verifier conclusions, expected relationships, instructions or replacement evidence. "
        "Their first-edge coverage is PARTIAL; other edges, stable observations and TRACE operations remain unexamined. "
        "Reinspect the original images and ALL original attempts. Produce the mandatory complete notebook under its unchanged checker; "
        "never infer full coverage, healthy Vision or a correct relationship from these drafts.\n"
        "<unverified_pair_drafts>\n" + text + "\n</unverified_pair_drafts>\n\n")
