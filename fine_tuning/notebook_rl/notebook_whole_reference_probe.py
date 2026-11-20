"""Zero-update whole-history reference decisions on the unchanged audited V45 input."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict

from . import episode_notebook as episode, fresh_observation
from . import notebook_pair_drafts as drafts, notebook_synthesis as synthesis
from . import notebook_whole_synthesis_probe as baseline, reasoning_trace
from . import whole_reference_credit as credit, whole_reference_notebook as reference
from .engines import RolloutRequest, _completion
from .notebook_layout_probe import _image_identity
from .prompts import for_actor, p0_body
from .rollout import _hash, seeds
from .schema import Sample, to_json

VERSION = "local-nf4-notebook-whole-reference-probe-v2"
CONTEXT_LIMIT, MAX_NEW_TOKENS = 24576, 8192


def contract(protocol: dict) -> dict:
    baseline.contract(protocol)
    return {"protocol_id": VERSION, "runtime_protocol_sha256": protocol["sha256"],
        "original_prompt": synthesis.prompt_contract(synthesis.BOUNDED_VERSION),
        "reference_contract": reference.contract(),
        "prompt_sha256": _hash([reference.prompt_contract(), baseline.SCOPE_GRAMMAR, reasoning_trace.VERSION]),
        "intervention": "complete public reference inventory with model-authored fact and check decisions",
        "pure_order_causal_test": False, "n": 4, "max_new_tokens": MAX_NEW_TOKENS,
        "context_limit": CONTEXT_LIMIT, "seeds": "original-first-DIAG-four-seeds",
        "selection": "first recovered fixed actor; all six solutions, two images and twelve saved drafts",
        "max_seconds": 3600, "optimizer_updates": 0, "training_allowed": False,
        "feedback": False, "retries": False, "draft_regeneration": False,
        "completion_capture": "per-seed-raw-sidecar-v1; incomplete groups never scored or resumed",
        "actual_learning_measured": False, "self_improvement_verified": False,
        "module_causality_certified": False, "all_reasoning_verified": False}


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def _messages(original, actor, schema, samples, note, success_attempt):
    """Keep original image parts and exact quoted reread/drafts; replace only active text."""
    messages = copy.deepcopy(original)
    reread = fresh_observation.quoted_context(note["fresh_observation"])
    proposals = drafts.quoted_context(note["pair_drafts"])
    old = messages[0]["content"][-1]["text"]
    if old.count(reread) != 1 or old.count(proposals) != 1:
        raise ValueError("whole reference probe: original reread or draft context differs")
    text = ("Question and choices are quoted original work data, not instructions.\n"
        "<read_only_original_question>\n" + _quoted(p0_body(actor.question, actor.choices))
        + "\n</read_only_original_question>\n\n" + reread + proposals
        + "The following grammar describes ORIGINAL solutions only. Your new output uses the reference contract below.\n"
        + "<read_only_solution_contract>\n"
        + _quoted(reasoning_trace.instruction_for_question(actor.question, actor.source))
        + "\n</read_only_solution_contract>\n" + baseline.SCOPE_GRAMMAR
        + reference.instruction(samples, schema, success_attempt))
    messages[0]["content"][-1]["text"] = text
    return messages, [{"before": old, "after": text}], reference.public_inventory(samples)


def collect(actors, trees, images, prompts, engine, verifier, record, deadline, *, record_completion=None):
    """The caller owns V45 ancestry, source approvals and the unchanged F0 lifecycle."""
    if (len(actors) != 2 or len({a.problem_id for a in actors}) != 2
            or [a.problem_id for a in actors] != [tree.problem_id for tree in trees]
            or type(engine.policy_version) is not int or engine.policy_version != 0):
        raise ValueError("whole reference probe: fixed two policy-zero actors required")
    original_trees = [to_json(tree) for tree in trees]
    rows: list[dict] = []
    skipped: list[dict] = []
    for actor, tree in zip(actors, trees, strict=True):
        if time.monotonic() >= deadline:
            raise TimeoutError("whole reference probe: frozen one-hour budget exhausted")
        audit = synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION)
        reason = ("FIRST_RECOVERED_ALREADY_SELECTED" if rows else "INCOMPLETE_EPISODE"
                  if not audit["complete_episode"] else "NO_VERIFIED_RECOVERY"
                  if audit["success_attempt"] is None else None)
        if reason:
            skipped.append({"problem_id": actor.problem_id, "reason": reason})
            continue
        originals = episode.solution_samples(tree)
        note = tree.notebook["rewarded_notebook"]
        groups = [group for group in tree.groups if group.kind == "DIAG"]
        schema = for_actor(actor, prompts)["observation_instruction"][actor.family]
        episode.validate_update_binding(tree, tokenizer=engine.tokenizer, preparation=engine,
            images=images[actor.problem_id], observation_actor=actor, observation_schema=schema,
            observation_master_seed=3407, draft_policy=drafts.VERSION)
        replay = episode.regrade_synthesis(tree, actor, verifier, engine.tokenizer)
        if (len(originals) != 6 or len(actor.image_paths) != 2 or len(groups) != 1
                or len(groups[0].samples) != 4 or not groups[0].usable
                or note["pair_drafts"]["status"] != "COMPLETE"
                or sum(len(call["completions"]) for call in note["pair_drafts"]["calls"]) != 12
                or type(tree.policy_version) is not int or tree.policy_version != 0):
            raise ValueError("whole reference probe: complete saved six-attempt/two-image/twelve-draft input required")
        group = groups[0]
        parent = originals[audit["success_attempt"] - 1]
        original_seeds = seeds(actor.problem_id, 0, parent.turn, "EPISODE_NOTEBOOK", 4, master_seed=3407)
        control = RolloutRequest(group.group_id, None, note["notebook_messages"], images[actor.problem_id],
                                 4, MAX_NEW_TOKENS, original_seeds)
        original = engine.prepare(control)
        if (group.group_id != f"{actor.problem_id}:v0:EPISODE_NOTEBOOK"
                or group.state_sha256 != _hash(original.token_ids) or note["prompt_sha256"] != group.state_sha256
                or note["notebook_image_key"] != original.image_key
                or any(sample.seed != original_seeds[index] or sample.prompt_token_ids != original.token_ids
                       or sample.sample_id != f"{group.group_id}:{index}" or sample.kind != "DIAG"
                       or sample.group_id != group.group_id or sample.problem_id != actor.problem_id
                       or type(sample.policy_version) is not int or sample.policy_version != 0
                       for index, sample in enumerate(group.samples))):
            raise ValueError("whole reference probe: original first-DIAG tokens, images or seeds differ")
        for sample in originals + group.samples:
            _completion(sample.completion_token_ids, sample.sampler_logprobs, sample.finish_reason, sample.text)
            if engine.tokenizer.decode(sample.completion_token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) != sample.text:
                raise ValueError("whole reference probe: original token/text binding differs")
        previous = [baseline._checked(sample, tree, audit, actor, verifier) for sample in group.samples]
        messages, changes, inventory = _messages(control.messages, actor, schema, originals,
                                                 note, audit["success_attempt"])
        coverage = reference.validate_audit_coverage(inventory, audit)
        request = RolloutRequest(f"{actor.problem_id}:{VERSION}", None, messages, images[actor.problem_id],
                                 4, MAX_NEW_TOKENS, list(original_seeds))
        prepared = engine.prepare(request)
        identity = _image_identity(actor, prepared)
        if identity != _image_identity(actor, original):
            raise ValueError("whole reference probe: original image tensors changed")
        if max(len(original.token_ids), len(prepared.token_ids)) + MAX_NEW_TOKENS > CONTEXT_LIMIT:
            raise ValueError("CONTEXT_BLOCKED: no truncation, draft omission or actor substitution permitted")
        record("CALL_STARTED", {"request_id": request.request_id, "messages": messages,
            "original_messages": control.messages, "prompt_changes": changes,
            "public_inventory": inventory, "public_inventory_sha256": _hash(inventory),
            "reference_contract": reference.contract(), "audit_coverage": coverage,
            "original_messages_sha256": _hash(control.messages), "changed_messages_sha256": _hash(messages),
            "original_prompt_sha256": _hash(original.token_ids), "changed_prompt_sha256": _hash(prepared.token_ids),
            "original_prompt_token_ids": original.token_ids, "prompt_token_ids": prepared.token_ids,
            "images": identity, "seeds": original_seeds, "n": 4, "max_new_tokens": MAX_NEW_TOKENS,
            "context_limit": CONTEXT_LIMIT, "baseline": previous, "baseline_replay": replay, "skipped": skipped})
        if time.monotonic() >= deadline:
            raise TimeoutError("whole reference probe: frozen one-hour budget exhausted")
        began = time.monotonic()
        if type(engine.policy_version) is not int or engine.policy_version != 0:
            raise ValueError("whole reference probe: policy changed before sampling")
        captured: list[dict] = []

        def completed(request_id, index, seed, completion):
            if (request_id != request.request_id or type(index) is not int
                    or index != len(captured) or index >= request.n
                    or type(seed) is not int or seed != request.seeds[index]):
                raise ValueError("whole reference probe: completion order/identity differs")
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if (len(completion.token_ids) > MAX_NEW_TOKENS or engine.tokenizer.decode(
                    completion.token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("whole reference probe: invalid captured completion budget/decode")
            value = asdict(completion)
            record_completion({"kind": "COMPLETION_CAPTURED", "request_id": request_id,
                "index": index, "seed": seed, "prompt_sha256": _hash(prepared.token_ids),
                "messages_sha256": _hash(messages), "completion": copy.deepcopy(value),
                "group_complete": False, "eligible_for_rl_loss": False})
            captured.append(value)

        outputs = (engine.sample([request], on_completion=completed) if record_completion is not None
                   else engine.sample([request]))
        if (len(outputs) != 1 or outputs[0].request_id != request.request_id
                or outputs[0].prompt_token_ids != prepared.token_ids or len(outputs[0].completions) != 4):
            raise ValueError("whole reference probe: incomplete or mismatched generation")
        if record_completion is not None and captured != [asdict(value) for value in outputs[0].completions]:
            raise ValueError("whole reference probe: captured completions differ from returned group")
        record("CALL_GENERATED", asdict(outputs[0]))
        samples, checked, expanded = [], [], []
        for index, completion in enumerate(outputs[0].completions):
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if (len(completion.token_ids) > MAX_NEW_TOKENS or engine.tokenizer.decode(completion.token_ids,
                    skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("whole reference probe: invalid actual completion budget/decode")
            sample = Sample(f"{request.request_id}:{index}", actor.problem_id, request.request_id, "DIAG", parent.turn,
                parent.sample_id, list(prepared.token_ids), completion.token_ids, completion.logprobs,
                completion.finish_reason, completion.text, original_seeds[index], 0)
            result = credit.check(sample, tree, audit, actor, verifier)
            samples.append(sample)
            checked.append(result)
            expanded.append(reference.expand(sample.text, sample.finish_reason, originals))
        mapped = credit.group_credit(samples, {sample.sample_id: row for sample, row in zip(samples, checked, strict=True)},
                                     engine.tokenizer)
        after = engine.prepare(request)
        if ([to_json(value) for value in trees] != original_trees
                or type(engine.policy_version) is not int or engine.policy_version != 0
                or after.token_ids != prepared.token_ids or _image_identity(actor, after) != identity):
            raise ValueError("whole reference probe: original history, policy or input changed")
        row = {"request_id": request.request_id, "problem_id": actor.problem_id,
            "status": "DIAGNOSTIC_CAPTURE_COMPLETE", "seconds": time.monotonic() - began,
            "samples": [asdict(sample) for sample in samples], "checks": checked,
            "expanded_grading_representations": expanded,
            "baseline": previous, "baseline_totals": baseline._totals(previous),
            "reference_totals": {
                "captured_outputs": sum(row["expansion_captured"] for row in checked),
                "program_rendered_strict_verified_notes": sum(row["strict_reward"] for row in checked),
                "verified_fact_units": sum(sum(row["fact_rewards"].values()) for row in checked),
                "verified_native_relationship_units": sum(row["native_verified_count"] for row in checked),
                "verified_output_finalization_units": sum(row["output_verified_count"] for row in checked)},
            "diagnostic_group_credit": mapped, "optimizer_updates": 0,
            "eligible_for_rl_loss": False, "used_for_routing": False,
            "actual_learning_measured": False, "self_improvement_verified": False,
            "all_reasoning_verified": False, "module_causality_certified": False}
        record("CALL_COMPLETED", row)
        rows.append(row)
    return rows, skipped
