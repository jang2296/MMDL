"""Post-update whole notes on the exact saved pre-update context; never training."""
from __future__ import annotations

import copy
import time
from dataclasses import asdict

from . import episode_notebook as episode, notebook_pair_drafts as drafts
from . import notebook_synthesis as synthesis, synthesis_credit as credit
from . import whole_reference_notebook as reference
from . import whole_reference_pages as pages
from .engines import RolloutRequest, _completion
from .local_profile import (ASSESSMENT_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID,
                            WHOLE_REFERENCE_PAGES_PROTOCOL_ID, restore_protocol)
from .notebook_layout_probe import _image_identity
from .notebook_whole_synthesis_probe import _totals
from .prompts import for_actor
from .rollout import _hash, seeds
from .schema import Group, Sample, to_json

VERSION = "local-nf4-postupdate-same-history-whole-notebook-v1"
REFERENCE_VERSION = "local-nf4-postupdate-same-history-whole-reference-notebook-v1"
PAGES_VERSION = "local-nf4-postupdate-same-history-whole-reference-pages-v1"
ASSESSMENT_VERSION = "local-nf4-postupdate-same-history-whole-reference-assessment-v1"


def contract(protocol: dict) -> dict:
    restore_protocol(protocol)
    whole_reference = protocol.get("notebook_generation_policy") == reference.VERSION
    paged = protocol.get("notebook_generation_policy") == pages.VERSION
    assessment = protocol.get("protocol_id") == ASSESSMENT_PROTOCOL_ID
    objective = credit.contract(credit.ASSESSMENT_VERSION if assessment else credit.VERSION)
    if (protocol.get("notebook_synthesis_contract") != synthesis.BOUNDED_VERSION
            or "notebook_generation_policy" in protocol and not (whole_reference or paged)
            or whole_reference and protocol.get("protocol_id") not in (
                WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID)
            or paged and protocol.get("protocol_id") != WHOLE_REFERENCE_PAGES_PROTOCOL_ID
            or protocol.get("notebook_objective") != objective
            or protocol.get("notebook_draft_policy") != drafts.VERSION
            or protocol.get("notebook_draft_contract") != drafts.contract()
            or protocol["rollout"]["diag"]["n"] != 4
            or protocol["rollout"]["diag"]["max_new_tokens"] != 8192):
        raise ValueError("trained whole notebook: canonical complete draft-assisted synthesis-credit protocol required")
    result = {"protocol_id": VERSION, "runtime_protocol_sha256": protocol["sha256"],
        "reward_version": synthesis.BOUNDED_VERSION, "objective": objective,
        "n": 4, "max_new_tokens": 8192, "context_limit": protocol["context_limit"],
        "max_seconds": 3600, "history_policy_version": 0, "generation_policy_version": 1,
        "input_intervention": "none; exact saved first-DIAG messages, images and seeds",
        "optimizer_updates": 0, "training_allowed": False, "draft_regeneration": False,
        "feedback": False, "retries": False, "heldout": False,
        "notebook_reward_effect_isolated": False, "self_improvement_verified": False,
        "module_causality_certified": False, "all_reasoning_verified": False}
    if whole_reference or paged:
        result.update(protocol_id=PAGES_VERSION if paged else REFERENCE_VERSION, diagnostic_only=True,
            notebook_generation_policy=pages.VERSION if paged else reference.VERSION,
            notebook_reference_contract=reference.contract(),
            notebook_reference_contract_sha256=_hash(reference.contract()),
            scoring_representation="program-rendered bounded notebook; action remains original raw reference JSON")
    if paged:
        result.update(notebook_page_contract=pages.contract(),
            notebook_page_contract_sha256=_hash(pages.contract()),
            input_intervention="none; exact saved page messages, images, caps and seeds",
            completion_capture="per-seed-raw-journal; incomplete bundles never scored or resumed",
            scoring_representation="four ordinal bundles; only original sampled pages are actions")
    if assessment:
        result["protocol_id"] = ASSESSMENT_VERSION
    return result


def _checked(sample, tree, audit, actor, verifier, *, objective_version=credit.VERSION) -> dict:
    whole_reference = tree.notebook.get("rewarded_notebook", {}).get("notebook_generation_policy") == reference.VERSION
    grade = reference.grade_note if whole_reference else synthesis.grade_note
    strict = grade(sample.text, sample.finish_reason, tree, audit, verifier,
                   actor=actor, contract=synthesis.BOUNDED_VERSION)
    binding = {"sample_id": sample.sample_id} if whole_reference else {}
    if whole_reference and sample.policy_version != tree.policy_version:
        binding["observed_sample"] = sample
    return {"sample_id": sample.sample_id, "seed": sample.seed, "text": sample.text,
        "finish_reason": sample.finish_reason, "completion_token_ids": sample.completion_token_ids,
        "sampler_logprobs": sample.sampler_logprobs, "strict_check": strict,
        "score": credit.score_note(sample.text, sample.finish_reason, tree, audit, verifier, strict,
            actor=actor, objective_version=objective_version, **binding),
        "eligible_for_rl_loss": False, "used_for_routing": False}


def _page_summaries(groups, tree, audit, actor, verifier, scores) -> list[dict]:
    """Logical reports contain no invented sample or assembled action tokens."""
    originals = episode.solution_samples(tree)
    rows = []
    for ordinal in range(4):
        members = [group.samples[ordinal] for group in groups]
        assembled = pages.assemble(members, originals)
        finish = "length" if any(sample.finish_reason == "length" for sample in members) else "stop"
        strict = reference.grade_note(assembled["reference_text"] or "{}", finish,
            tree, audit, verifier, actor=actor, contract=synthesis.BOUNDED_VERSION)
        common = scores[members[0].sample_id]["page_binding"]["bundle_score"]
        summary = {key: common[key] for key in ("atomic_reward", "fact_rewards", "relationship_rewards",
                                                "relationship_credit_types")}
        summary.update(program_rendered_only=True, diagnostic_only=True, training_allowed=False)
        rows.append({"ordinal": ordinal, "member_sample_ids": [sample.sample_id for sample in members],
            "assembly": assembled, "strict_check": strict, "score": summary,
            "eligible_for_rl_loss": False, "used_for_routing": False})
    return rows


def _collect_pages(actor, tree, images, engine, verifier, record, unchanged, audit, policy) -> dict:
    originals = episode.solution_samples(tree)
    note = tree.notebook["rewarded_notebook"]
    page_plan = pages.plan(originals)
    groups = [group for group in tree.groups if group.kind == "DIAG"]
    calls = note.get("page_calls")
    if (_hash(note.get("page_plan")) != _hash(page_plan) or type(calls) is not list
            or len(calls) != len(page_plan["pages"]) or len(groups) != len(calls)):
        raise ValueError("trained whole notebook pages: complete saved page plan and groups required")
    requests = []
    for index, (page, call, group) in enumerate(zip(page_plan["pages"], calls, groups, strict=True)):
        expected_seeds = seeds(actor.problem_id, 0, 1, f"EPISODE_NOTEBOOK:PAGE:{index}", 4, master_seed=3407)
        request = RolloutRequest(f"{actor.problem_id}:{PAGES_VERSION}:PAGE:{index}", None,
            copy.deepcopy(call["messages"]), images, 4, page["max_new_tokens"], list(expected_seeds))
        prepared = engine.prepare(request)
        identity = _image_identity(actor, prepared)
        if (type(call.get("page_index")) is not int or call["page_index"] != index
                or call.get("group_id") != group.group_id
                or group.group_id != f"{actor.problem_id}:v0:EPISODE_NOTEBOOK:PAGE:{index}"
                or call.get("max_new_tokens") != page["max_new_tokens"]
                or type(call.get("max_new_tokens")) is not int or not group.usable or len(group.samples) != 4
                or call.get("seeds") != expected_seeds
                or call.get("sample_ids") != [sample.sample_id for sample in group.samples]
                or group.state_sha256 != _hash(prepared.token_ids)
                or call.get("prompt_sha256") != group.state_sha256
                or call.get("prompt_token_ids") != prepared.token_ids
                or call.get("image_key") != prepared.image_key or _hash(call.get("images")) != _hash(identity)
                or any(sample.sample_id != f"{group.group_id}:{ordinal}" or sample.group_id != group.group_id
                    or sample.problem_id != actor.problem_id or sample.kind != "DIAG" or sample.turn != 1
                    or type(sample.policy_version) is not int or sample.policy_version != 0
                    or sample.seed != expected_seeds[ordinal] or sample.prompt_token_ids != prepared.token_ids
                    for ordinal, sample in enumerate(group.samples))):
            raise ValueError("trained whole notebook pages: saved page input, owner, images or seeds differ")
        if len(prepared.token_ids) + request.max_new_tokens > policy["context_limit"]:
            raise ValueError("CONTEXT_BLOCKED: no page history or output-budget truncation permitted")
        for sample in group.samples:
            _completion(sample.completion_token_ids, sample.sampler_logprobs, sample.finish_reason, sample.text)
            if (len(sample.completion_token_ids) > request.max_new_tokens or engine.tokenizer.decode(
                    sample.completion_token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) != sample.text):
                raise ValueError("trained whole notebook pages: original token/text binding differs")
        requests.append((request, prepared, identity))
    baseline_scores = credit.score_page_bundles(tree, audit, verifier, actor=actor)
    baseline = _page_summaries(groups, tree, audit, actor, verifier, baseline_scores)
    observed: list[Group] = []
    page_calls = []
    began = time.monotonic()
    for index, (request, prepared, identity) in enumerate(requests):
        unchanged()
        record("CALL_STARTED", {"request_id": request.request_id, "page_index": index,
            "messages": request.messages, "messages_sha256": _hash(request.messages),
            "prompt_token_ids": prepared.token_ids, "images": identity, "seeds": request.seeds,
            "n": 4, "max_new_tokens": request.max_new_tokens, "context_limit": policy["context_limit"],
            "history_policy_version": 0, "generation_policy_version": 1})
        captured: list[dict] = []

        def completed(request_id, ordinal, seed, completion):
            if (request_id != request.request_id or type(ordinal) is not int
                    or ordinal != len(captured) or ordinal >= 4
                    or type(seed) is not int or seed != request.seeds[ordinal]):
                raise ValueError("trained whole notebook pages: completion order or identity differs")
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if (len(completion.token_ids) > request.max_new_tokens or engine.tokenizer.decode(
                    completion.token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("trained whole notebook pages: actual token/text binding or cap differs")
            value = asdict(completion)
            record("COMPLETION_CAPTURED", {"request_id": request_id, "page_index": index,
                "ordinal": ordinal, "seed": seed, "prompt_sha256": _hash(prepared.token_ids),
                "messages_sha256": _hash(request.messages), "completion": copy.deepcopy(value),
                "bundle_complete": False, "eligible_for_rl_loss": False, "used_for_routing": False})
            captured.append(value)

        outputs = engine.sample([request], on_completion=completed)
        record("CALL_GENERATED", {"page_index": index, "outputs": [asdict(output) for output in outputs]})
        unchanged()
        if (len(outputs) != 1 or outputs[0].request_id != request.request_id
                or outputs[0].prompt_token_ids != prepared.token_ids or len(outputs[0].completions) != 4
                or captured != [asdict(value) for value in outputs[0].completions]):
            raise ValueError("trained whole notebook pages: incomplete or mismatched captured page")
        samples = [Sample(f"{request.request_id}:{ordinal}", actor.problem_id, request.request_id, "DIAG", 1,
            None, list(prepared.token_ids), list(completion.token_ids), list(completion.logprobs),
            completion.finish_reason, completion.text, request.seeds[ordinal], 1)
            for ordinal, completion in enumerate(outputs[0].completions)]
        after = engine.prepare(request)
        unchanged()
        if (after.token_ids != prepared.token_ids or _hash(_image_identity(actor, after)) != _hash(identity)
                or request.messages != calls[index]["messages"] or request.seeds != calls[index]["seeds"]):
            raise ValueError("trained whole notebook pages: input, images or seeds changed during generation")
        observed.append(Group(request.request_id, "DIAG", actor.problem_id, _hash(prepared.token_ids), samples, True, None))
        page_calls.append({"page_index": index, "request_id": request.request_id,
            "baseline_group_id": groups[index].group_id, "max_new_tokens": request.max_new_tokens,
            "seeds": list(request.seeds), "sample_ids": [sample.sample_id for sample in samples]})
    scores = credit.score_page_bundles(tree, audit, verifier, actor=actor, observed_groups=observed)
    checked = _page_summaries(observed, tree, audit, actor, verifier, scores)
    for ordinal, value in enumerate(checked):
        value.update(baseline_notebook_check=baseline[ordinal]["strict_check"],
                     notebook_check=value["strict_check"])
    row = {"problem_id": actor.problem_id, "seconds": time.monotonic() - began,
        "notebook_generation_policy": pages.VERSION, "page_plan": page_plan, "page_calls": page_calls,
        "physical_call_count": len(observed), "physical_sample_count": sum(len(group.samples) for group in observed),
        "physical_samples": [{**asdict(sample), "score": scores[sample.sample_id],
            "eligible_for_rl_loss": False, "used_for_routing": False} for group in observed for sample in group.samples],
        "samples": checked, "baseline": baseline, "before_totals": _totals(baseline), "after_totals": _totals(checked),
        "history_policy_version": 0, "generation_policy_version": 1, "optimizer_updates": 0,
        "heldout": False, "notebook_reward_effect_isolated": False, "self_improvement_verified": False}
    unchanged()
    record("CALL_COMPLETED", row)
    return row


def collect(actors, trees, images, prompts, engine, verifier, record, deadline, *, protocol):
    policy = contract(protocol)
    if (len(actors) != 2 or len({actor.problem_id for actor in actors}) != 2
            or [actor.problem_id for actor in actors] != [tree.problem_id for tree in trees]
            or type(engine.policy_version) is not int or engine.policy_version != 1
            or engine.context_limit != policy["context_limit"]):
        raise ValueError("trained whole notebook: fixed two actors and independently reloaded policy one required")
    original_trees = [to_json(tree) for tree in trees]
    rows, skipped = [], []

    def unchanged():
        if ([to_json(tree) for tree in trees] != original_trees
                or type(engine.policy_version) is not int or engine.policy_version != 1):
            raise ValueError("trained whole notebook: original history or generation policy changed")
        if time.monotonic() >= deadline:
            raise TimeoutError("trained whole notebook: frozen one-hour budget exhausted")

    for actor, tree in zip(actors, trees, strict=True):
        unchanged()
        if type(tree.policy_version) is not int or tree.policy_version != 0:
            raise ValueError("trained whole notebook: pre-update policy-zero history required")
        audit = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
        if not audit["complete_episode"] or audit["success_attempt"] is None:
            skipped.append({"problem_id": actor.problem_id, "reason": "INCOMPLETE_EPISODE"
                            if not audit["complete_episode"] else "NO_VERIFIED_RECOVERY"})
            continue
        schema = for_actor(actor, prompts)["observation_instruction"][actor.family]
        episode.validate_update_binding(tree, tokenizer=engine.tokenizer, observation_actor=actor,
            observation_schema=schema, feedback_policy=protocol.get("notebook_feedback_policy"),
            observation_policy=protocol.get("notebook_observation_policy"),
            generation_policy=protocol.get("notebook_generation_policy"),
            draft_policy=protocol.get("notebook_draft_policy"))
        if protocol.get("notebook_generation_policy") == pages.VERSION:
            rows.append(_collect_pages(actor, tree, images[actor.problem_id], engine, verifier,
                record, unchanged, audit, policy))
            continue
        replay = episode.regrade_synthesis(tree, actor, verifier, engine.tokenizer)
        groups = [group for group in tree.groups if group.kind == "DIAG"]
        if not groups:
            raise ValueError("trained whole notebook: saved first-DIAG group required")
        group = groups[0]
        parent = episode.solution_samples(tree)[audit["success_attempt"] - 1]
        expected_seeds = seeds(actor.problem_id, 0, parent.turn, "EPISODE_NOTEBOOK", 4, master_seed=3407)
        note = tree.notebook["rewarded_notebook"]
        saved_drafts = note["pair_drafts"]
        draft_images = [call["images"] for call in saved_drafts["calls"]]
        if saved_drafts["status"] != "COMPLETE" or not draft_images:
            raise ValueError("trained whole notebook: historical draft pixel identities required")
        request = RolloutRequest(f"{actor.problem_id}:{policy['protocol_id']}", None,
            copy.deepcopy(note["notebook_messages"]), images[actor.problem_id], 4,
            policy["max_new_tokens"], list(expected_seeds))
        prepared = engine.prepare(request)
        identity = _image_identity(actor, prepared)
        if (not group.usable or len(group.samples) != 4
                or [sample.seed for sample in group.samples] != expected_seeds
                or any(sample.prompt_token_ids != prepared.token_ids for sample in group.samples)
                or group.state_sha256 != _hash(prepared.token_ids)
                or note["prompt_sha256"] != group.state_sha256
                or note["notebook_image_key"] != prepared.image_key
                or any(_hash(value) != _hash(identity) for value in draft_images)):
            raise ValueError("trained whole notebook: saved first-DIAG input, images or seeds differ")
        if len(prepared.token_ids) + request.max_new_tokens > policy["context_limit"]:
            raise ValueError("CONTEXT_BLOCKED: no history, draft or output-budget truncation permitted")
        objective_version = policy["objective"]["objective_version"]
        baseline = [_checked(sample, tree, audit, actor, verifier, objective_version=objective_version)
                    for sample in group.samples]
        record("CALL_STARTED", {"request_id": request.request_id, "messages": request.messages,
            "messages_sha256": _hash(request.messages), "prompt_token_ids": prepared.token_ids,
            "images": identity, "seeds": expected_seeds, "n": 4,
            "max_new_tokens": request.max_new_tokens, "context_limit": policy["context_limit"],
            "history_policy_version": 0, "generation_policy_version": 1, "baseline_replay": replay})
        unchanged()
        began = time.monotonic()
        outputs = engine.sample([request])
        record("CALL_GENERATED", {"outputs": [asdict(output) for output in outputs]})
        unchanged()
        if (len(outputs) != 1 or outputs[0].request_id != request.request_id
                or outputs[0].prompt_token_ids != prepared.token_ids or len(outputs[0].completions) != 4):
            raise ValueError("trained whole notebook: incomplete or mismatched generation")
        checked = []
        for index, completion in enumerate(outputs[0].completions):
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if (len(completion.token_ids) > request.max_new_tokens or engine.tokenizer.decode(
                    completion.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("trained whole notebook: completion budget or token/text binding differs")
            sample = Sample(f"{request.request_id}:{index}", actor.problem_id, request.request_id, "DIAG",
                parent.turn, parent.sample_id, list(prepared.token_ids), completion.token_ids,
                completion.logprobs, completion.finish_reason, completion.text, expected_seeds[index], 1)
            value = _checked(sample, tree, audit, actor, verifier, objective_version=objective_version)
            checked.append(value | {"baseline_notebook_check": baseline[index]["strict_check"],
                                    "notebook_check": value["strict_check"]})
        after = engine.prepare(request)
        unchanged()
        if after.token_ids != prepared.token_ids or _image_identity(actor, after) != identity:
            raise ValueError("trained whole notebook: input or image tensors changed during generation")
        row = {"request_id": request.request_id, "problem_id": actor.problem_id,
            "seconds": time.monotonic() - began, "samples": checked, "baseline": baseline,
            "before_totals": _totals(baseline), "after_totals": _totals(checked),
            "history_policy_version": 0, "generation_policy_version": 1,
            "optimizer_updates": 0, "heldout": False, "notebook_reward_effect_isolated": False,
            "self_improvement_verified": False}
        record("CALL_COMPLETED", row)
        rows.append(row)
    unchanged()
    return rows, skipped
