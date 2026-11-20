"""Collect real n4 page actions; assemble four ordinal notebooks only for grading."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from typing import Any

from . import episode_notebook as episode, fresh_observation, notebook_pair_drafts as drafts
from . import notebook_synthesis as synthesis, whole_reference_notebook as reference, whole_reference_pages as pages
from .engines import Completion, RolloutRequest, _completion
from .notebook_layout_probe import _image_identity
from .prompts import for_actor, user_message
from .rollout import _hash, _sample_request, seeds
from .schema import ActorItem, Group, ProblemTree, Sample, from_json, to_json


def generation_messages(actor, originals, schema, success_attempt, page_index, reread, draft_record) -> list[dict]:
    original = reference.generation_text(actor, originals, schema, success_attempt, reread, draft_record)
    suffix = reference.SCOPE_GRAMMAR + reference.instruction(originals, schema, success_attempt)
    if not original.endswith(suffix):
        raise ValueError("whole reference pages: original public input contract differs")
    draft_context = drafts.quoted_context(draft_record)
    wrapper, marker, payload = draft_context.partition("<unverified_pair_drafts>\n")
    context = original[:-len(suffix)]
    if (not marker or wrapper.count(pages.DRAFT_NOTEBOOK_REQUEST) != 1
            or context.count(draft_context) != 1):
        raise ValueError("whole reference pages: trusted draft wrapper differs")
    page_context = wrapper.replace(pages.DRAFT_NOTEBOOK_REQUEST, pages.DRAFT_PAGE_REQUEST, 1) + marker + payload
    body = context.replace(draft_context, page_context, 1) + pages.instruction(originals, schema, success_attempt, page_index)
    return [user_message(body, len(actor.image_paths))]


def _request(tree, actor, schema, page, reread, draft_record, images) -> RolloutRequest:
    index = page["page_index"]
    branch = f"EPISODE_NOTEBOOK:PAGE:{index}"
    return RolloutRequest(f"{tree.problem_id}:v{tree.policy_version}:{branch}", None,
        generation_messages(actor, episode.solution_samples(tree), schema,
            tree.notebook["rewarded_notebook"]["episode_audit"]["success_attempt"], index, reread, draft_record),
        images, 4, page["max_new_tokens"],
        seeds(tree.problem_id, tree.policy_version, 1, branch, 4, master_seed=3407))


def _group_hash(tree) -> str:
    return _hash([json.loads(to_json(group)) for group in tree.groups if group.kind == "DIAG"])


def _call(request, page, actor, prepared) -> dict:
    return {"page_index": page["page_index"], "group_id": request.request_id,
        "messages": copy.deepcopy(request.messages), "messages_sha256": _hash(request.messages),
        "images": json.loads(json.dumps(_image_identity(actor, prepared))), "image_key": prepared.image_key,
        "seeds": list(request.seeds), "sample_ids": [f"{request.request_id}:{ordinal}" for ordinal in range(4)],
        "n": 4, "max_new_tokens": page["max_new_tokens"], "prompt_token_ids": list(prepared.token_ids),
        "prompt_sha256": _hash(prepared.token_ids), "completions": []}


def reconstruct_bundles(tree, audit, verifier, *, actor) -> list[dict]:
    """Fresh strict grades use the assembled text without constructing sampled actions."""
    groups = [group for group in tree.groups if group.kind == "DIAG"]
    originals = episode.solution_samples(tree)
    if len(groups) != len(pages.plan(originals)["pages"]) or any(len(group.samples) != 4 for group in groups):
        raise ValueError("whole reference pages: complete physical collection required")
    bundles = []
    for ordinal in range(4):
        members = [group.samples[ordinal] for group in groups]
        assembly = pages.assemble(members, originals)
        text = assembly["reference_text"] if assembly["status"] == pages.CAPTURED else "{}"
        finish = "length" if assembly["status"] == pages.TRUNCATED else "stop"
        grade = reference.grade_note(text, finish, tree, audit, verifier, actor=actor,
                                     contract=synthesis.BOUNDED_VERSION)
        bundle = {"ordinal": ordinal, "member_sample_ids": [sample.sample_id for sample in members],
            "members_sha256": _hash([json.loads(to_json(sample)) | {"reward": None, "diag_reward": None}
                                      for sample in members]), "assembly": assembly, "grade": grade}
        bundles.append(bundle | {"sha256": _hash(bundle)})
    return bundles


def _physical_grade(sample, page_index, bundle) -> dict:
    return bundle["grade"] | {"sample_id": sample.sample_id, "text": sample.text,
        "eligible_for_rl_loss": True, "notebook_generation_policy": pages.VERSION,
        "page_index": page_index, "bundle_ordinal": bundle["ordinal"], "bundle_sha256": bundle["sha256"],
        "note_text_sha256": reference._sha(sample.text), "note_finish_reason": sample.finish_reason,
        "assembled_note_text_sha256": bundle["grade"]["note_text_sha256"]}


def _collect(tree, actor, images, engine, verifier, cfg, *, on_page, should_stop) -> None:
    record = tree.notebook["rewarded_notebook"]
    pending: list[Group] = []
    for page in record["page_plan"]["pages"]:
        if should_stop is not None and should_stop():
            record.update(status="FINAL_NOTE_BLOCKED", stop_reason="BUDGET_STOP")
            return
        request = _request(tree, actor, record["notebook_schema"], page,
                           record["fresh_observation"], record["pair_drafts"], images)
        try:
            prepared = engine.prepare(request)
            if len(prepared.token_ids) + request.max_new_tokens > cfg.context_limit:
                raise ValueError("CONTEXT_BLOCKED: full page reservation; no truncation")
        except ValueError as exc:
            if not str(exc).startswith("CONTEXT_BLOCKED:"):
                raise
            record.update(status="FINAL_NOTE_BLOCKED", stop_reason="CONTEXT_BLOCKED")
            return
        call = _call(request, page, actor, prepared)
        observed: list[dict] = []

        def completed(request_id: str, index: int, seed: int, completion: Completion) -> None:
            if (request_id != request.request_id or type(index) is not int or index != len(observed)
                    or index >= 4 or seed != request.seeds[index]):
                raise ValueError("whole reference pages: completed response identity changed")
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if len(completion.token_ids) > request.max_new_tokens or engine.tokenizer.decode(
                    completion.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion.text:
                raise ValueError("whole reference pages: completed raw page budget or decode changed")
            observed.append(asdict(completion))
            if on_page is not None:
                on_page(copy.deepcopy({"version": pages.VERSION, "purpose": "WHOLE_REFERENCE_PAGE_COMPLETION",
                    "problem_id": tree.problem_id, "policy_version": tree.policy_version,
                    "page_index": page["page_index"], "group_id": request_id, "index": index, "seed": seed,
                    "max_new_tokens": request.max_new_tokens, "messages": request.messages,
                    "images": call["images"], "image_key": prepared.image_key,
                    "prompt_token_ids": prepared.token_ids, "completion": observed[-1]}))

        output = _sample_request(request, engine, tree.policy_version, cfg.context_limit,
                                 on_completion=completed if on_page is not None else None)
        call["completions"] = [asdict(completion) for completion in output.completions]
        if on_page is not None and observed != call["completions"]:
            raise ValueError("whole reference pages: streamed and final raw responses differ")
        record["page_calls"].append(call)
        generated = []
        for ordinal, completion in enumerate(output.completions):
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if (len(completion.token_ids) > request.max_new_tokens or engine.tokenizer.decode(
                    completion.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("whole reference pages: actual page budget or decode changed")
            generated.append(Sample(call["sample_ids"][ordinal], tree.problem_id, request.request_id, "DIAG", 1,
                None, list(output.prompt_token_ids), list(completion.token_ids), list(completion.logprobs),
                completion.finish_reason, completion.text, request.seeds[ordinal], tree.policy_version))
        pending.append(Group(request.request_id, "DIAG", tree.problem_id, _hash(output.prompt_token_ids), generated, True, None))
    # No page enters the tree or immutable GROUP ledger until all four bundles can be graded.
    candidate = copy.deepcopy(tree)
    candidate.groups.extend(pending)
    bundles = reconstruct_bundles(candidate, record["synthesis_audit"], verifier, actor=actor)
    notes = []
    for index, group in enumerate(pending):
        for ordinal, sample in enumerate(group.samples):
            sample.reward = sample.diag_reward = bundles[ordinal]["grade"]["reward"]
            notes.append(_physical_grade(sample, index, bundles[ordinal]))
    tree.groups.extend(pending)
    tree.notebook.setdefault("model_notes", []).extend(notes)
    record.update(status="GENERATED", bundles=bundles, group_ids=[group.group_id for group in pending],
        reward_variance_present=len({bundle["grade"]["reward"] for bundle in bundles}) > 1,
        stop_reason="VALID_NOTE" if any(bundle["grade"]["reward"] == 1 for bundle in bundles) else "NOTEBOOK_PAGE_LIMIT")


def append(trees, images, engine, verifier, prompts, cfg, *, actors, on_group=None,
           on_observation=None, on_draft=None, on_page=None, should_stop=None) -> dict:
    cfg.validate()
    required = {"notebook_generation_policy": pages.VERSION, "notebook_page_contract": pages.contract(),
        "notebook_page_contract_sha256": _hash(pages.contract()), "notebook_reference_contract": reference.contract(),
        "notebook_reference_contract_sha256": _hash(reference.contract()), "rewarded_notebook": pages.prompt_contract(),
        "notebook_synthesis_contract": synthesis.BOUNDED_VERSION, "notebook_input_policy": synthesis.INPUT_POLICY,
        "notebook_draft_policy": drafts.VERSION, "notebook_draft_contract": drafts.contract(),
        "notebook_draft_contract_sha256": _hash(drafts.contract()), "notebook_max_rounds": 1,
        "notebook_strict_feedback_repairs": False, "notebook_route_rule": episode.STRICT_ROUTE_RULE,
        "notebook_feedback_policy": synthesis.FEEDBACK_POLICY, "notebook_observation_policy": fresh_observation.POLICY,
        "notebook_repair_parent_policy": synthesis.REPAIR_PARENT_POLICY}
    if (not trees or len({tree.problem_id for tree in trees}) != len(trees)
            or set(images) != {tree.problem_id for tree in trees} or set(actors) != set(images)
            or _hash({key: prompts.get(key) for key in required}) != _hash(required)
            or (cfg.n_diag, cfg.diag_max_new_tokens, cfg.context_limit, cfg.master_seed) != (4, 8192, 24576, 3407)
            or any(tree.policy_version != engine.policy_version or "rewarded_notebook" in tree.notebook
                   or any(group.kind == "DIAG" for group in tree.groups) for tree in trees)):
        raise ValueError("whole reference pages: fresh same-policy explicit page protocol required")
    audits = {tree.problem_id: episode.audit_episode(tree, actors[tree.problem_id], verifier,
              routing_rule=episode.STRICT_ROUTE_RULE) for tree in trees}
    synthesis_audits = {tree.problem_id: synthesis.audit_episode(tree, actors[tree.problem_id], verifier,
                        contract=synthesis.BOUNDED_VERSION) for tree in trees}
    report = {"reward_version": synthesis.BOUNDED_VERSION, "generated_groups": 0, "nondegenerate_groups": 0}
    for tree in trees:
        actor, audit = actors[tree.problem_id], audits[tree.problem_id]
        schema = for_actor(actor, prompts)["observation_instruction"].get(actor.family)
        if type(schema) is not str or not schema.strip():
            raise ValueError("whole reference pages: public schema required")
        originals = episode.solution_samples(tree)
        record = {key: copy.deepcopy(value) for key, value in required.items()
                  if key not in ("rewarded_notebook", "notebook_route_rule", "notebook_input_policy",
                                 "notebook_feedback_policy", "notebook_observation_policy")}
        inventory = reference.public_inventory(originals) if originals else None
        record.update(status="PENDING", reward_version=synthesis.BOUNDED_VERSION, input_policy=synthesis.INPUT_POLICY,
            feedback_policy=synthesis.FEEDBACK_POLICY, observation_policy=fresh_observation.POLICY,
            episode_audit=audit, synthesis_audit=synthesis_audits[tree.problem_id],
            synthesis_audit_sha256=_hash(synthesis_audits[tree.problem_id]),
            notebook_actor=json.loads(to_json(actor)), notebook_schema=schema, coverage="COMPLETE",
            reference_inventory=inventory, reference_inventory_sha256=_hash(inventory) if inventory else None,
            reference_audit_coverage=reference.validate_audit_coverage(inventory, synthesis_audits[tree.problem_id]) if inventory else None,
            page_plan=pages.plan(originals) if originals else None, page_calls=[], bundles=[], group_ids=[], rounds=[],
            used_for_routing=False, routing_finalized=False, reward_variance_present=False,
            optimizer_update_verified=False, lesson_semantics_verified=False)
        tree.notebook["rewarded_notebook"] = record
        tree.notebook["pre_notebook_route"] = tree.route
        if not audit["complete_episode"] or audit["success_attempt"] is None:
            record["status"] = "INCOMPLETE_EPISODE" if not audit["complete_episode"] else "NO_VERIFIED_RECOVERY"
        elif should_stop is not None and should_stop():
            record["status"] = "BUDGET_STOP"
        else:
            try:
                record["fresh_observation"] = fresh_observation.generate(actor, schema, images[tree.problem_id], engine,
                    master_seed=3407, context_limit=24576, on_event=on_observation)
            except ValueError as exc:
                if not str(exc).startswith("CONTEXT_BLOCKED:"):
                    raise
                record.update(status="CONTEXT_BLOCKED", stop_reason="CONTEXT_BLOCKED")
            if "fresh_observation" in record:
                draft_record = drafts.generate(actor, tree, record["fresh_observation"], schema, images[tree.problem_id],
                    engine, master_seed=3407, context_limit=24576, on_event=on_draft, should_stop=should_stop)
                record.update(pair_drafts=draft_record, pair_drafts_sha256=_hash(draft_record))
                if draft_record["status"] in ("BUDGET_STOP", "CONTEXT_BLOCKED"):
                    record.update(status="DRAFT_BLOCKED", stop_reason=draft_record["status"])
                else:
                    _collect(tree, actor, images[tree.problem_id], engine, verifier, cfg,
                             on_page=on_page, should_stop=should_stop)
        if (episode.audit_episode(tree, actor, verifier, routing_rule=episode.STRICT_ROUTE_RULE) != audit
                or synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION) != synthesis_audits[tree.problem_id]):
            raise ValueError("whole reference pages: original independent evidence changed")
        tree.route = audit["route"]
        record.update(episode_audit_sha256=_hash(audit), notebook_groups_sha256=_group_hash(tree),
                      rounds_sha256=_hash([]), routing_finalized=True)
        validate(tree, tokenizer=engine.tokenizer, preparation=engine, images=images[tree.problem_id],
                 observation_actor=actor, observation_schema=schema, observation_master_seed=3407)
        if on_group is not None:
            for group in tree.groups:
                if group.kind == "DIAG":
                    on_group(group)
        count = len(record["group_ids"])
        report["generated_groups"] += count
        report["nondegenerate_groups"] += count * int(record["reward_variance_present"])
    return report


def validate(tree, *, tokenizer=None, preparation=None, images=None, observation_actor=None,
             observation_schema=None, observation_master_seed=None, **policies) -> None:
    note = tree.notebook.get("rewarded_notebook", {})
    expected_policies = {"generation_policy": pages.VERSION, "feedback_policy": synthesis.FEEDBACK_POLICY,
        "observation_policy": fresh_observation.POLICY, "draft_policy": drafts.VERSION}
    if any(key not in expected_policies or value is not episode._UNSPECIFIED_FEEDBACK_POLICY
           and value != expected_policies[key] for key, value in policies.items()):
        raise ValueError("whole reference pages: execution policy differs")
    required = {"notebook_generation_policy": pages.VERSION, "notebook_page_contract": pages.contract(),
        "notebook_page_contract_sha256": _hash(pages.contract()), "notebook_synthesis_contract": synthesis.BOUNDED_VERSION,
        "notebook_reference_contract": reference.contract(), "notebook_reference_contract_sha256": _hash(reference.contract()),
        "notebook_draft_policy": drafts.VERSION, "notebook_draft_contract": drafts.contract(),
        "notebook_draft_contract_sha256": _hash(drafts.contract()), "notebook_max_rounds": 1,
        "notebook_strict_feedback_repairs": False, "notebook_repair_parent_policy": synthesis.REPAIR_PARENT_POLICY,
        "reward_version": synthesis.BOUNDED_VERSION, "input_policy": synthesis.INPUT_POLICY,
        "feedback_policy": synthesis.FEEDBACK_POLICY, "observation_policy": fresh_observation.POLICY,
        "coverage": "COMPLETE", "used_for_routing": False, "routing_finalized": True, "rounds": [], "rounds_sha256": _hash([])}
    if _hash({key: note.get(key) for key in required}) != _hash(required) or observation_master_seed not in (None, 3407):
        raise ValueError("whole reference pages: explicit page contract changed")
    actor = from_json(ActorItem, json.dumps(note.get("notebook_actor"), allow_nan=False))
    originals, schema = episode.solution_samples(tree), note.get("notebook_schema")
    audit, synthesis_audit = note.get("episode_audit", {}), note.get("synthesis_audit", {})
    routing = episode.strict_routing(tree)
    if (actor.problem_id != tree.problem_id or observation_actor is not None and actor != observation_actor
            or type(schema) is not str or not schema.strip() or observation_schema is not None and schema != observation_schema
            or note.get("episode_audit_sha256") != _hash(audit) or audit.get("solution_sha256") != episode.solution_hash(tree)
            or audit.get("transcript_sha256") != _hash(episode.public_transcript(originals))
            or audit.get("policy_version") != tree.policy_version or tree.route != audit.get("route")
            or any(_hash(audit.get(key)) != _hash(value) for key, value in routing.items())
            or note.get("synthesis_audit_sha256") != _hash(synthesis_audit)
            or synthesis_audit.get("version") != synthesis.BOUNDED_VERSION
            or synthesis_audit.get("solution_sha256") != audit.get("solution_sha256")
            or synthesis_audit.get("transcript_sha256") != audit.get("transcript_sha256")
            or synthesis_audit.get("actor_sha256") != _hash(actor.__dict__)
            or synthesis_audit.get("success_attempt") != audit.get("success_attempt")):
        raise ValueError("whole reference pages: original history, actor or independent routing changed")
    inventory = reference.public_inventory(originals) if originals else None
    page_plan = pages.plan(originals) if originals else None
    if (note.get("page_plan") != page_plan or _hash(note.get("reference_inventory")) != _hash(inventory)
            or note.get("reference_inventory_sha256") != (_hash(inventory) if inventory else None)
            or note.get("reference_audit_coverage") != (reference.validate_audit_coverage(inventory, synthesis_audit) if inventory else None)):
        raise ValueError("whole reference pages: complete public inventory or page plan changed")
    groups = [group for group in tree.groups if group.kind == "DIAG"]
    calls, bundles = note.get("page_calls"), note.get("bundles")
    if (type(calls) is not list or type(bundles) is not list or note.get("group_ids") != [group.group_id for group in groups]
            or note.get("notebook_groups_sha256") != _group_hash(tree)
            or page_plan is not None and len(calls) > len(page_plan["pages"])
            or groups and (len(groups) != len(page_plan["pages"]) or len(calls) != len(groups)
                           or len(bundles) != 4 or note.get("status") != "GENERATED")
            or not groups and (bundles or page_plan is not None and len(calls) == len(page_plan["pages"])
                or note.get("status") not in ("NO_VERIFIED_RECOVERY", "INCOMPLETE_EPISODE",
                "BUDGET_STOP", "CONTEXT_BLOCKED", "DRAFT_BLOCKED", "FINAL_NOTE_BLOCKED"))):
        raise ValueError("whole reference pages: incomplete collection admitted or group binding changed")
    reread, draft_record = note.get("fresh_observation"), note.get("pair_drafts")
    if reread is not None:
        fresh_observation.validate(reread, policy_version=tree.policy_version, tokenizer=tokenizer,
            preparation=preparation, images=images, actor=actor, schema=schema, master_seed=3407)
    if draft_record is not None:
        if reread is None or note.get("pair_drafts_sha256") != _hash(draft_record):
            raise ValueError("whole reference pages: draft evidence changed")
        drafts.validate(draft_record, actor, tree, reread, schema, tokenizer=tokenizer, preparation=preparation, images=images)
        if calls and draft_record["status"] not in ("COMPLETE", "NO_PUBLIC_TARGETS", "PUBLIC_TARGET_BLOCKED"):
            raise ValueError("whole reference pages: blocked drafts admitted pages")
    if (calls or groups) and (reread is None or draft_record is None):
        raise ValueError("whole reference pages: every page needs complete reread and draft evidence")
    for index, call in enumerate(calls):
        page = page_plan["pages"][index]
        request = _request(tree, actor, schema, page, reread, draft_record, images or [])
        identity = call.get("images")
        if (type(call.get("page_index")) is not int or call["page_index"] != index or call.get("group_id") != request.request_id
                or call.get("messages") != request.messages or call.get("messages_sha256") != _hash(request.messages)
                or call.get("seeds") != request.seeds or type(call.get("n")) is not int or call["n"] != 4
                or type(call.get("max_new_tokens")) is not int or call["max_new_tokens"] != page["max_new_tokens"]
                or call.get("sample_ids") != [f"{request.request_id}:{ordinal}" for ordinal in range(4)]
                or call.get("prompt_sha256") != _hash(call.get("prompt_token_ids"))
                or type(call.get("prompt_token_ids")) is not list or not call["prompt_token_ids"]
                or len(call["prompt_token_ids"]) + page["max_new_tokens"] > 24576
                or type(identity) is not dict or identity.get("image_paths") != list(actor.image_paths)
                or identity.get("image_sha256") != list(actor.image_sha256)
                or identity.get("image_key") != call.get("image_key") or call.get("image_key") != reread["image_key"]
                or type(call.get("completions")) is not list or len(call["completions"]) != 4):
            raise ValueError("whole reference pages: page messages, seeds, ownership, budget or pixels changed")
        if groups and len(groups[index].samples) != 4:
            raise ValueError("whole reference pages: complete physical n4 page required")
        if draft_record["calls"] and _hash(identity) != _hash(draft_record["calls"][0]["images"]):
            raise ValueError("whole reference pages: original prepared tensors differ from drafts")
        for ordinal, completion in enumerate(call["completions"]):
            _completion(completion["token_ids"], completion["logprobs"], completion["finish_reason"], completion["text"])
            if len(completion["token_ids"]) > page["max_new_tokens"] or tokenizer is not None and tokenizer.decode(
                    completion["token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion["text"]:
                raise ValueError("whole reference pages: raw page completion differs")
            if groups:
                group, sample = groups[index], groups[index].samples[ordinal]
                expected = Sample(call["sample_ids"][ordinal], tree.problem_id, request.request_id, "DIAG", 1,
                    None, call["prompt_token_ids"], completion["token_ids"], completion["logprobs"],
                    completion["finish_reason"], completion["text"], request.seeds[ordinal], tree.policy_version,
                    None, bundles[ordinal]["grade"]["reward"], bundles[ordinal]["grade"]["reward"])
                if (sample != expected or group.group_id != request.request_id or group.problem_id != tree.problem_id
                        or group.kind != "DIAG" or group.usable is not True or group.reason is not None
                        or len(group.samples) != 4 or group.state_sha256 != call["prompt_sha256"]):
                    raise ValueError("whole reference pages: actual physical action changed")
        if preparation is not None:
            if images is None or len(images) != len(actor.image_paths) or preparation.policy_version != tree.policy_version:
                raise ValueError("whole reference pages: original same-policy images required")
            actual = preparation.prepare(request)
            if actual.token_ids != call["prompt_token_ids"] or _hash(_image_identity(actor, actual)) != _hash(identity):
                raise ValueError("whole reference pages: reconstructed original page input differs")
    notes = tree.notebook.get("model_notes", [])
    expected_notes: list[dict] = []
    for ordinal, bundle in enumerate(bundles):
        members = [group.samples[ordinal] for group in groups]
        expected = {"ordinal": ordinal, "member_sample_ids": [sample.sample_id for sample in members],
            "members_sha256": _hash([json.loads(to_json(sample)) | {"reward": None, "diag_reward": None} for sample in members]),
            "assembly": pages.assemble(members, originals), "grade": bundle.get("grade")}
        if bundle != expected | {"sha256": _hash(expected)}:
            raise ValueError("whole reference pages: ordinal bundle, provenance or raw membership changed")
        if bundle["assembly"]["status"] != pages.CAPTURED and bundle["grade"].get("reward") != 0:
            raise ValueError("whole reference pages: invalid whole bundle has nonzero strict reward")
    for index, group in enumerate(groups):
        expected_notes.extend(_physical_grade(sample, index, bundles[ordinal]) for ordinal, sample in enumerate(group.samples))
    if notes != expected_notes or note.get("reward_variance_present") != (len({bundle["grade"]["reward"] for bundle in bundles}) > 1):
        raise ValueError("whole reference pages: physical grade or variance binding changed")
    if groups and note.get("stop_reason") != ("VALID_NOTE" if any(bundle["grade"]["reward"] == 1 for bundle in bundles) else "NOTEBOOK_PAGE_LIMIT"):
        raise ValueError("whole reference pages: collection stop reason changed")
    for sample in originals:
        if tokenizer is not None and tokenizer.decode(sample.completion_token_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False) != sample.text:
            raise ValueError("whole reference pages: original solution tokens differ")


def regrade(tree: ProblemTree, actor: ActorItem, verifier: Any, tokenizer: Any) -> dict:
    if verifier is None or tokenizer is None:
        raise ValueError("whole reference pages: fresh verifier and tokenizer required")
    validate(tree, tokenizer=tokenizer, observation_actor=actor)
    record = tree.notebook["rewarded_notebook"]
    audit = episode.audit_episode(tree, actor, verifier, routing_rule=episode.STRICT_ROUTE_RULE)
    synthesis_audit = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
    if audit != record["episode_audit"] or synthesis_audit != record["synthesis_audit"]:
        raise ValueError("whole reference pages: fresh original evidence differs")
    bundles = reconstruct_bundles(tree, synthesis_audit, verifier, actor=actor) if record["group_ids"] else []
    if bundles != record["bundles"]:
        raise ValueError("whole reference pages: fresh whole-bundle strict grade differs")
    receipt = {"contract": synthesis.BOUNDED_VERSION, "actor_sha256": _hash(actor.__dict__),
        "synthesis_audit_sha256": _hash(synthesis_audit), "notebook_groups_sha256": record["notebook_groups_sha256"],
        "bundles": bundles, "samples": {row["sample_id"]: {key: value for key, value in row.items()
            if key not in ("sample_id", "text", "eligible_for_rl_loss")}
            for row in tree.notebook.get("model_notes", [])}}
    return receipt | {"sha256": _hash(receipt)}
