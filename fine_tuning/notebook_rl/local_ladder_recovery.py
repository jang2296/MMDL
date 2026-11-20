"""Native n=1 recovery and strict whole-history notes; no training or model loading.

Callers own execution authority. Literal records and CPU replay do not certify
model provenance, tokenizer equivalence, lesson truth, or module causality.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

from . import episode_notebook as notebook, fresh_observation as fresh, local_ladder_generation as generation, prompts
from .approval import code_hashes
from .cli import artifact_path
from .dataset import _bytes, _write, load_images
from .engines import RolloutRequest
from .local_ladder import _descriptor, _json, load_ladder_actors
from .rollout import (BLOCKING, RETRYABLE, REPAIR_SELECTION_POLICY, _hash, _sample_request, continuation_ids,
                      malformed_retry_receipt, repair_parent_feedback, seeds, select_repair_parents)
from .schema import ActorItem, Group, ProblemTree, Sample, from_json, to_json
from .vault import _safe_root
from .verifier_service import VerifierService

VERSION = "local-native-ladder-recovery-notebook-v1"
REPAIR_EVALUATOR_VERSION = "local-native-ladder-verified-repair-evaluation-v3"
_SINGLE = "EVALUATION_SINGLE_SAMPLE"
_METRICS = ("first_attempt_accuracy", "final_accuracy", "recovery_accuracy", "recovered_count",
            "initially_failed", "initially_incorrect", "recovered_fraction", "note_count",
            "note_verified_count", "note_verified_fraction")
_IMAGE_FIELDS = ("actor", "image_key", "image_token_id", "image_tokens", "image_spans",
                 "pixel_tensors", "image_grid_thw", "original_image_sizes", "processed_image_sizes")


def _reference(templates: dict) -> bool:
    return templates.get("evaluation_prompt_policy") == generation.CIRCUIT_REFERENCE_PROMPT_POLICY


def _repair(templates: dict) -> bool:
    return templates.get("evaluation_prompt_policy") == generation.CIRCUIT_REPAIR_PROMPT_POLICY


def _note_contract(contract: dict) -> str:
    """Final-note contract recorded in a recovery protocol; anything unknown rebuilds as v2 and fails closed."""
    from .compact_note import VERSION as compact_version
    from .repair_contract import FINAL_VERSION

    notebook = contract.get("notebook") if type(contract) is dict else None
    value = notebook.get("response_contract") if type(notebook) is dict else None
    return compact_version if value == compact_version else FINAL_VERSION


def protocol(prompt_templates: dict, local_protocol_sha256: str, *, context_limit: int = generation.CONTEXT_LIMIT,
             note_contract: str | None = None) -> dict:
    first = generation.protocol(prompt_templates, local_protocol_sha256, context_limit=context_limit)
    if note_contract is not None and not _repair(prompt_templates):
        raise ValueError("ladder recovery: a final-note contract requires the repair evaluation policy")
    if _repair(prompt_templates):
        from . import compact_note, repair_contract, repair_credit, repair_notebook

        if note_contract not in (None, *repair_notebook.NOTE_CONTRACTS):
            raise ValueError("ladder recovery: explicit known final-note contract required")
        contract = {**first, "protocol_id": VERSION, "evaluator_version": REPAIR_EVALUATOR_VERSION,
            "max_attempts": 3, "first_attempt_protocol_id": generation.VERSION,
            "seed_rule": "T1 unchanged; retries/NOTE rollout.seeds(policy_version=0,branch=protocol_id:stage)",
            "request_order": "all-ten-T1; one-fresh-before-each-eligible-T2; ordered-T2; ordered-T3; ordered-recovered-NOTE",
            "retryable_statuses": sorted(RETRYABLE), "retry_correct_native_issues": True,
            "retry_selection_policy": REPAIR_SELECTION_POLICY,
            "response_contracts": {"T1": None, "T2": repair_contract.VERSION, "T3": repair_contract.VERSION},
            "retry_prompt_sha256": generation._digest(prompts.repair_retry_prompt_contract()),
            "notebook": {"n": 1, "max_new_tokens": 8192, "max_calls_per_recovery": 1,
                "repair_calls": 0, "fresh_observation_calls": "one-before-first-eligible-retry-otherwise-zero",
                "fresh_observation_policy": fresh.POLICY, "fresh_observation_max_new_tokens": fresh.MAX_NEW_TOKENS,
                "fresh_observation_prompt_sha256": _hash(fresh.PROMPT),
                "fresh_observation_context_sha256": _hash(fresh.RETRY_CONTEXT), "history_scope": "every-actual-attempt-including-success",
                "response_contract": repair_contract.FINAL_VERSION,
                "prompt_sha256": generation._digest(repair_notebook.PROMPT),
                "metric_identity": {"online": repair_credit.AUDIT_VERSION, "final": repair_credit.FINAL_AUDIT_VERSION},
                "grader": "repair_credit.audit_final_note", "strict_notebook_equivalent": False,
                "trigger": "actual-retry-and-independent-correct-source",
                "success_selection": "latest-independently-correct-source"}}
        if note_contract == compact_note.VERSION:
            contract["notebook"].update(response_contract=compact_note.VERSION,
                prompt_sha256=generation._digest(compact_note.PROMPT), note_renderer="compact-history-v1",
                metric_identity={"online": repair_credit.AUDIT_VERSION, "final": compact_note.VERSION},
                grader="compact_note.audit")
        return contract
    metric = ({"metric_identity": notebook.VERSION,
               "grader": "episode_notebook.grade_note", "bounded_synthesis_verified": False}
              if "evaluation_prompt_policy" in first else {})
    if _reference(prompt_templates):
        from . import circuit_trace, notebook_synthesis as synthesis, whole_reference_notebook as reference

        public = reference.evaluation_generation_contract()
        metric = {"metric_identity": {"reference": reference.VERSION, "bounded": synthesis.BOUNDED_VERSION,
                                      "circuit": circuit_trace.VERSION},
                  "grader": "whole_reference_notebook.grade_note", "bounded_synthesis_verified": True,
                  "generation_contract": public, "generation_contract_sha256": generation._digest(public),
                  "prompt_sha256": generation._digest(public), "reward_version": synthesis.BOUNDED_VERSION,
                  "routing_rule": synthesis.CIRCUIT_EVALUATION_ROUTING,
                  "unverified_draft_calls": 0, "candidate_selection": False,
                  "training_assistance_equivalent": False}
    return {**first, "protocol_id": VERSION, "max_attempts": 3,
        "first_attempt_protocol_id": generation.VERSION,
        "seed_rule": "T1 unchanged; retries/NOTE rollout.seeds(policy_version=0,branch=protocol_id:stage)",
        "request_order": "all-ten-T1; ordered-T2; ordered-T3; ordered-recovered-NOTE",
        "retryable_statuses": sorted(RETRYABLE), "correct_first_attempts_retained": True,
        "notebook": {"n": 1, "max_new_tokens": 8192 if _reference(prompt_templates) else 2048, "max_calls_per_recovery": 1,
                     "repair_calls": 0, "fresh_observation_calls": 0,
                     "history_scope": "every-actual-attempt-including-success",
                     "prompt_sha256": generation._digest(notebook.PROMPT),
                     "reward_version": notebook.VERSION, "routing_rule": notebook.STRICT_ROUTE_RULE,
                     "reward": "strict-whole-history-only", **metric}}


def _seed(pid: str, stage: str) -> int:
    return seeds(pid, 0, 2 if stage == "T2" else 3, VERSION + ":" + stage, 1, master_seed=3407)[0]


def _wire(value) -> dict:
    return json.loads(to_json(value))


def _group(sample: Sample) -> Group:
    # Training groups still require n>=2; this evaluation sample is never usable for loss.
    return Group(sample.group_id, sample.kind, sample.problem_id, _hash(sample.prompt_token_ids),
                 [sample], False, _SINGLE)


def _audit(tree, actor, verifier, templates: dict | None = None) -> dict:
    if templates is not None and _reference(templates):
        from . import notebook_synthesis as synthesis
        return synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION, evaluation_only=True)
    return notebook.audit_episode(tree, actor, verifier,
        routing_rule=notebook.STRICT_ROUTE_RULE, evaluation_only=True)


def _note_messages(actor, templates, tree, audit, n_images):
    observation = prompts.for_actor(actor, templates)["observation_instruction"].get(actor.family)
    if type(observation) is not str or not observation.strip():
        raise ValueError("ladder recovery: public family observation schema required")
    if _reference(templates):
        from . import whole_reference_notebook as reference
        originals = notebook.solution_samples(tree)
        reference.validate_audit_coverage(reference.public_inventory(originals), audit)
        body = reference.evaluation_generation_text(actor, originals, observation,
                                                   audit["success_attempt"])
        return [prompts.user_message(body, n_images)]
    body = notebook.instruction(notebook.solution_samples(tree), audit, observation)
    return [prompts.user_message(prompts.p0_body(actor.question, actor.choices) + "\n\n" + body, n_images)]


def _grade_note(row, tree, audit, verifier, actor, templates):
    if _reference(templates):
        from . import notebook_synthesis as synthesis, whole_reference_notebook as reference
        return reference.grade_note(row["text"], row["finish_reason"], tree, audit, verifier,
            evaluation_only=True, actor=actor, contract=synthesis.BOUNDED_VERSION)
    return notebook.grade_note(row["text"], row["finish_reason"], tree, audit, verifier,
                               evaluation_only=True, actor=actor)


def _check_verifier(verifier, preparation):
    if (verifier.metadata["item_count"] != 10 or verifier.metadata["vault_manifest_sha256"]
            != preparation["records_manifest"]["file_sha256"]):
        raise ValueError("ladder recovery: fresh verifier cohort identity differs")


def _verifier(root):
    return VerifierService(root / "records", forbidden_roots=[Path.home() / "mmdl-data/evaluation"], timeout=60)


def _sample(row, kind, turn, parent, verifier, actor):
    sample = Sample(row["request_id"] + ":0", actor.problem_id, row["request_id"], kind, turn,
        parent.sample_id if parent else None, list(row["input"]["prompt_token_ids"]),
        list(row["completion_token_ids"]), list(row["logprobs"]), row["finish_reason"],
        row["text"], row["seed"], row["policy_version"])
    sample.result = verifier.grade(actor.problem_id, sample.text, sample.finish_reason, actor.mode)
    return sample


def _validate_call(row, pid, version, first, parent=None, *, context_limit, note_tokens=2048, repair_policy=False):
    stage = row.get("stage")
    cap = note_tokens if stage == "NOTE" else 1024
    ids, logs, inp = row.get("completion_token_ids"), row.get("logprobs"), row.get("input")
    if (stage not in ("T2", "T3", "NOTE") or row.get("problem_id") != pid
            or row.get("request_id") != pid + ":" + stage or row.get("policy_version") != version
            or type(row.get("policy_version")) is not int or row.get("seed") != _seed(pid, stage)
            or type(row.get("seed")) is not int or row.get("n") != 1 or type(row.get("n")) is not int
            or type(row.get("max_new_tokens")) is not int or row.get("max_new_tokens") != cap
            or type(row.get("context_limit")) is not int or row.get("context_limit") != context_limit
            or type(inp) is not dict or row.get("input_sha256") != generation._digest(inp)
            or row.get("record_sha256") != generation._digest({k: v for k, v in row.items() if k != "record_sha256"})
            or type(row.get("text")) is not str or row.get("finish_reason") not in ("stop", "length")
            or type(ids) is not list or not 1 <= len(ids) <= cap
            or any(type(token) is not int or token < 0 for token in ids)
            or type(logs) is not list or len(logs) != len(ids)
            or any(type(log) not in (int, float) or not math.isfinite(log) or log > 1e-5 for log in logs)
            or row.get("completion_tokens") != len(ids)
            or type(row.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0):
        raise ValueError("ladder recovery: invalid literal response binding")
    tokens = inp.get("prompt_token_ids")
    if (type(tokens) is not list or not tokens or len(tokens) + cap > context_limit
            or any(type(token) is not int or token < 0 for token in tokens)
            or inp.get("input_tokens") != len(tokens)
            or any(inp.get(key) != first["input"].get(key) for key in _IMAGE_FIELDS)):
        raise ValueError("ladder recovery: changed context or native image inputs")
    if stage != "NOTE":
        prefix = parent.prompt_token_ids + parent.completion_token_ids if parent else []
        suffix = row.get("continuation_suffix_token_ids")
        malformed = False
        if repair_policy and parent is not None and parent.result is None:
            from .repair_contract import VERSION as retry_version
            malformed_retry_receipt(parent, retry_version)
            malformed = True
        if (parent is None or row.get("parent_sample_id") != parent.sample_id
                or parent.turn + 1 != int(stage[1])
                or not malformed and (parent.result is None
                    or parent.result.status not in (RETRYABLE | {"CORRECT"} if repair_policy else RETRYABLE))
                or inp.get("messages") is not None
                or type(suffix) is not list or not suffix or tokens != prefix + suffix
                or any(type(token) is not int or token < 0 for token in suffix)):
            raise ValueError("ladder recovery: broken literal retry ancestry")
    elif (parent is None or parent.result is None or parent.result.status != "CORRECT"
          or row.get("parent_sample_id") != parent.sample_id or row.get("feedback") is not None
          or row.get("continuation_suffix_token_ids") is not None):
        raise ValueError("ladder recovery: note requires the actual recovered success")


def _metrics(trees, rows, *, reference_policy=False):
    first, final, verified = {}, {}, []
    for entry in trees:
        tree = from_json(ProblemTree, json.dumps(entry["tree"]))
        samples = notebook.solution_samples(tree)
        first[tree.problem_id], final[tree.problem_id] = samples[0].result.status, samples[-1].result.status
        if entry.get("note") is not None:
            checked = entry["note"]["checked"]
            if type(checked.get("reward")) not in (int, float) or checked["reward"] not in (0.0, 1.0):
                raise ValueError("ladder recovery: strict binary note grade required")
            if reference_policy:
                from . import notebook_synthesis as synthesis, whole_reference_notebook as reference
                strict = checked.get("rendered_strict_report")
                if (type(strict) is not dict or checked.get("notebook_generation_policy") != reference.VERSION
                        or checked.get("program_rendered_only") is not True
                        or any(checked.get(key) != strict.get(key) for key in ("reward", "checks", "failure_code"))):
                    raise ValueError("ladder recovery: bound strict reference notebook report required")
                passed = (checked.get("expansion_status") == reference.CAPTURED
                          and synthesis.successful(strict, synthesis.BOUNDED_VERSION))
            else:
                passed = (checked.get("failure_code") == "NONE" and _bytes(checked.get("checks")) == _bytes({
                    key: True for key in ("format_valid", "all_attempts_verified", "answer_verified", "registered_facts_verified")}))
            if (checked["reward"] == 1.0) != passed:
                raise ValueError("ladder recovery: inconsistent strict note checks")
            verified.append(checked["reward"] == 1.0)
    failed = sum(status != "CORRECT" for status in first.values())
    recovered = sum(first[pid] != "CORRECT" and final[pid] == "CORRECT" for pid in first)
    accuracy = sum(status == "CORRECT" for status in final.values()) / 10
    return {"first_statuses": first, "final_statuses": final,
        "first_attempt_accuracy": sum(status == "CORRECT" for status in first.values()) / 10,
        "final_accuracy": accuracy, "recovery_accuracy": accuracy,
        "initially_failed": failed, "initially_incorrect": sum(status == "INCORRECT" for status in first.values()),
        "recovered_count": recovered, "first_status_counts": dict(Counter(first.values())),
        "final_status_counts": dict(Counter(final.values())),
        "recovered_fraction": recovered / failed if failed else None,
        "note_count": len(verified), "note_verified_count": sum(verified),
        "note_verified_fraction": sum(verified) / len(verified) if verified else None,
        "usage": {"input_tokens": sum(r["input"]["input_tokens"] for r in rows),
                  "image_tokens": sum(r["input"]["image_tokens"] for r in rows),
                  "completion_tokens": sum(r["completion_tokens"] for r in rows)},
        "generation_seconds": sum(r["elapsed_seconds"] for r in rows),
        "generation_calls": len(rows), "length_terminated_calls": sum(r["finish_reason"] == "length" for r in rows)}


def _repair_sample(row, turn, parent, verifier, actor):
    from . import notebook_synthesis as synthesis, repair_contract, repair_notebook
    from .schema import VerifierResult

    sample = Sample(row["request_id"] + ":0", actor.problem_id, row["request_id"], f"T{turn}", turn,
        parent.sample_id if parent else None, list(row["input"]["prompt_token_ids"]),
        list(row["completion_token_ids"]), list(row["logprobs"]), row["finish_reason"],
        row["text"], row["seed"], row["policy_version"])
    response_contract = None if turn == 1 else repair_contract.VERSION
    try:
        audited = synthesis.audit_native_sample(sample, actor, verifier, response_contract=response_contract)
    except repair_contract.RepairContractError as exc:
        if exc.code not in repair_notebook._GRAMMAR_ERRORS:
            raise
        return sample, {"syntax_failure": {"code": exc.code, "at": exc.at, "offset": exc.offset,
            "raw_sha256": _raw_sha(sample.text)}, "native": None}
    sample.result = from_json(VerifierResult, json.dumps(audited["result"]))
    if sample.result.status in BLOCKING:
        raise ValueError("ladder repair: blocking verifier outcome")
    return sample, {"syntax_failure": None, "native": audited}


def _repair_contracts(samples):
    from .repair_contract import VERSION as retry_version
    return {sample.sample_id: None if sample.turn == 1 else retry_version for sample in samples}


def _repair_inventory(samples, *, parent=None):
    from .repair_contract import ORIGIN_VERSION, public_origin_inventory
    return public_origin_inventory(samples, contract_version=ORIGIN_VERSION,
        response_contracts=_repair_contracts(samples), parent_sample_id=parent.sample_id if parent else None)


def _repair_feedback(actor, templates, samples, audits, observation=None):
    parent = samples[-1]
    native = {sid: row["native"] for sid, row in audits.items() if row["native"] is not None}
    invalid = [{"sample_id": sid, "code": row["syntax_failure"]["code"],
                "raw_sha256": row["syntax_failure"]["raw_sha256"]}
               for sid, row in audits.items() if row["syntax_failure"] is not None]
    contracts = _repair_contracts(samples)
    key = f"{actor.problem_id}:{parent.policy_version}" + (f":{parent.turn}" if parent.turn > 1 else "")
    selected, _ = select_repair_parents([parent], 1, key, native_audits=native,
        invalid_responses=invalid, response_contracts=contracts)
    if not selected:
        return None
    feedback = repair_parent_feedback(parent, actor, templates, _repair_inventory(samples, parent=parent),
        native_audits=native, invalid_responses=invalid, response_contracts=contracts)
    return feedback + ("\n" + fresh.quoted_retry_context(observation) if observation is not None else "")


def _repair_episode(tree, native):
    from .repair_contract import public_attempt_outcome

    samples = notebook.solution_samples(tree)
    contracts = _repair_contracts(samples)
    outcomes = [public_attempt_outcome(row.text, response_contract=contracts[row.sample_id], result=row.result)
                for row in samples]
    if any(outcome is None for outcome, _ in outcomes):
        raise ValueError("ladder repair: source outcome unavailable")
    correct = [row for row, (outcome, _) in zip(samples, outcomes, strict=True) if outcome == "CORRECT"]
    success = correct[-1].sample_id if len(samples) > 1 and correct else None
    return {"evaluator_version": REPAIR_EVALUATOR_VERSION, "solution_sha256": notebook.solution_hash(tree),
        "transcript_sha256": _hash(notebook.public_transcript(samples)),
        "complete_episode": True, "training_complete_episode": False, "training_admission": False,
        "evaluation_only_single_sample_groups": True, "route": notebook.strict_routing(tree)["route"],
        "routing": notebook.strict_routing(tree),
        "success_sample_id": success, "response_contracts": _repair_contracts(samples),
        "inventory": asdict(_repair_inventory(samples)), "native_audits": native}


def _repair_note_messages(actor, templates, tree, audit, feedback, note_contract="repair-note-v2"):
    from .compact_note import VERSION as compact_version
    from .repair_notebook import compact_final_text, public_final_text
    observation = tree.notebook.get("fresh_observation", tree.notebook.get("repair_collector", {}).get("fresh_observation"))
    if note_contract == compact_version:
        body = compact_final_text(actor, notebook.solution_samples(tree), audit["response_contracts"], observation)
    else:
        body = public_final_text(actor, templates, notebook.solution_samples(tree), feedback,
                                 audit["inventory"], audit["success_sample_id"], observation)
    return [prompts.user_message(body, len(actor.image_paths))]


def _repair_original_feedback(pid, calls):
    return [{"group_id": pid + ":T1", "turn": 1, "parent_sample_id": None,
             "feedback": None, "feedback_sha256": None}] + [
        {"group_id": row["request_id"], "turn": int(row["stage"][1]),
         "parent_sample_id": row["parent_sample_id"], "feedback": row["feedback"],
         "feedback_sha256": _hash(row["feedback"])}
        for row in calls if row["problem_id"] == pid and row["stage"] in ("T2", "T3")]


def _repair_note_grade(row, tree, audit, verifier, actor, note_contract="repair-note-v2"):
    from . import repair_notebook
    parent = next(sample for sample in notebook.solution_samples(tree) if sample.sample_id == audit["success_sample_id"])
    sample = Sample(row["request_id"] + ":0", actor.problem_id, row["request_id"], "DIAG", parent.turn,
        parent.sample_id, row["input"]["prompt_token_ids"], row["completion_token_ids"], row["logprobs"],
        row["finish_reason"], row["text"], row["seed"], row["policy_version"])
    return repair_notebook._grade(sample, tree, actor, verifier, audit, note_contract)


def _repair_online(samples, actor, verifier):
    from .repair_contract import VERSION as retry_version, parse_retry_sections
    from .repair_credit import audit_online_repair
    return asdict(audit_online_repair(samples[:-1], samples[-1], actor, verifier,
        inventory=_repair_inventory(samples[:-1], parent=samples[-2]),
        response_contracts=_repair_contracts(samples),
        sections=parse_retry_sections(samples[-1].text, contract_version=retry_version)))


def _compact_note_metrics(entries) -> dict:
    """Per-recovery compact note verification; refuted entries are the note's false accusations."""
    from .compact_note import VERSION as compact_version
    notes = []
    for entry in entries:
        if entry["note"] is None:
            continue
        audit = entry["note"]["checked"]["audit"]
        failure = entry["note"]["checked"]["syntax_failure"]
        notes.append({"problem_id": entry["problem_id"], "syntax_failure_code": failure["code"] if failure else None,
            "reward": audit["reward"] if audit else 0.0, "verified_entry_count": audit["verified"] if audit else 0,
            "refuted_entry_count": audit["refuted"] if audit else 0,
            "vision_claim": audit["vision"]["claim"] if audit else None,
            "vision_route": audit["vision"]["route"] if audit else None,
            "vision_agreement": audit["vision"]["score"] if audit else None})
    return {"note_contract": compact_version, "per_recovery": notes, "note_count": len(notes),
        "parsed_note_count": sum(row["syntax_failure_code"] is None for row in notes),
        "verified_entry_count": sum(row["verified_entry_count"] for row in notes),
        "refuted_entry_count": sum(row["refuted_entry_count"] for row in notes),
        "vision_agreement_count": sum(row["vision_agreement"] == 1 for row in notes),
        "vision_contradiction_count": sum(row["vision_agreement"] == -1 for row in notes),
        "mean_reward": sum(row["reward"] for row in notes) / len(notes) if notes else None,
        "lesson_rewarded": False}


def _repair_metrics(entries, rows, note_contract="repair-note-v2"):
    from .compact_note import VERSION as compact_version
    compact = note_contract == compact_version
    # shortcut: resource sums cover solutions/notes; include fresh receipts when reporting total resources.
    from .repair_contract import public_attempt_outcome

    first, final = {}, {}
    online: list[dict] = []
    final_audits: list[dict] = []
    native: list[dict] = []
    known: set[tuple[str, str, str, str]] = set()
    reviewed: set[tuple[str, str, str, str]] = set()
    eligible_known: set[tuple[str, str, str, str]] = set()
    new_errors, eligible_histories, correction_opportunities = 0, 0, 0
    invalid_responses, output_limits, no_final = 0, 0, 0
    for entry in entries:
        samples = notebook.solution_samples(from_json(ProblemTree, json.dumps(entry["tree"])))
        contracts = _repair_contracts(samples)
        outcomes = [public_attempt_outcome(row.text, response_contract=contracts[row.sample_id], result=row.result)
                    for row in samples]
        if any(outcome is None for outcome, _ in outcomes):
            raise ValueError("ladder repair: source outcome unavailable")
        first[entry["problem_id"]], final[entry["problem_id"]] = outcomes[0][0], outcomes[-1][0]
        invalid_responses += sum(outcome == "INVALID_RESPONSE_CONTRACT" for outcome, _ in outcomes)
        output_limits += sum(sample.finish_reason == "length" for sample in samples)
        no_final += sum(outcome == "INVALID_NO_FINAL" for outcome, _ in outcomes)
        audited = entry["episode_audit"]["native_audits"]
        native.extend(row["native"] for row in audited.values() if row["native"] is not None)
        origins = {row["ref_id"]: row for row in entry["episode_audit"]["inventory"]["origins"]}
        entry_known = {(sample_id, issue["kind"], issue["rule"], issue["locus"])
                       for sample_id, row in audited.items() if row["native"] is not None for issue in row["native"]["issues"]}
        known.update(entry_known)
        if len(samples) > 1 and any(outcome == "CORRECT" for outcome, _ in outcomes):
            eligible_histories += 1
            eligible_known.update(entry_known)
            positions = {sample.sample_id: index for index, sample in enumerate(samples)}
            checks = []
            for origin in origins.values():
                source = audited[origin["sample_id"]]["native"]
                matching = [] if source is None else [check for check in source["checks"]
                    if (check["kind"], check["rule"], check["locus"]) == (origin["kind"], origin["rule"], origin["locus"])]
                if len(matching) == 1 and matching[0]["identity"] is not None:
                    checks.append((positions[origin["sample_id"]], matching[0]))
            correction_opportunities += sum(before["status"] == "VIOLATED" and after["status"] == "COMPLIANT"
                and before["kind"] != "OUTPUT" and before["kind"] == after["kind"] and before["rule"] == after["rule"]
                and _hash(before["identity"]) == _hash(after["identity"])
                for i, before in checks for j, after in checks if i < j)
        submitted_refs = {ref for audit in entry["online_audits"] for witness in audit["witnesses"]
                          for ref in (witness["origin_ref"], witness["later_ref"]) if ref is not None}
        if not compact and entry["note"] is not None and entry["note"]["checked"]["audit"] is not None:
            submitted_refs.update(ref for witness in entry["note"]["checked"]["audit"]["witnesses"]
                                  for ref in (witness["origin_ref"], witness["later_ref"]) if ref is not None)
        reviewed.update((origins[ref]["sample_id"], origins[ref]["kind"], origins[ref]["rule"], origins[ref]["locus"])
                        for ref in submitted_refs if ref in origins)
        for parent, child in zip(samples, samples[1:]):
            before, after = audited[parent.sample_id]["native"], audited[child.sample_id]["native"]
            if before is not None and after is not None:
                previous = {_hash([issue["kind"], issue["rule"], issue["identity"]]) for issue in before["issues"]}
                new_errors += sum(_hash([issue["kind"], issue["rule"], issue["identity"]]) not in previous
                                  for issue in after["issues"] if issue["identity"] is not None)
        online.extend(entry["online_audits"])
        if not compact and entry["note"] is not None and entry["note"]["checked"]["audit"] is not None:
            final_audits.append(entry["note"]["checked"]["audit"])
    witnesses = [witness for audit in online for witness in audit["witnesses"]]
    counts: dict[str, int] = {key: sum(audit["metrics"][key] for audit in final_audits) for key in (
        "submitted_claim_count", "verified_claim_count", "false_claim_count", "unverifiable_claim_count",
        "structural_invalid_count", "known_issue_count", "covered_known_issue_count", "uncovered_known_issue_count",
        "correction_opportunity_count", "covered_correction_opportunity_count")}
    counts.update(known_issue_count=len(eligible_known), correction_opportunity_count=correction_opportunities)
    counts["uncovered_known_issue_count"] = counts["known_issue_count"] - counts["covered_known_issue_count"]
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None
    repair_metrics = {"evaluator_version": REPAIR_EVALUATOR_VERSION,
        "online_claim_count": len(witnesses),
        "verified_native_corrected_count": sum(w["claim_verdict"] == "VERIFIED" and w["relation"] == "CORRECTED" for w in witnesses),
        "false_accusation_count": counts["false_claim_count"] + sum(w["claim_verdict"] == "REFUTED" for w in witnesses),
        "online_false_accusation_count": sum(w["claim_verdict"] == "REFUTED" for w in witnesses),
        "final_false_accusation_count": counts["false_claim_count"],
        "new_error_count": new_errors, "new_error_scope": "identity-bound-native-child-issues-absent-from-parent",
        "known_native_issue_count": len(known),
        "unknown_count": sum(len(audit["unverified"]) for audit in native),
        "unreviewed_count": len(known - reviewed),
        "malformed_response_count": invalid_responses,
        "invalid_response_contract_count": invalid_responses,
        "output_limit_count": output_limits, "no_final_count": no_final,
        "final_note_count": sum(entry["note"] is not None for entry in entries),
        "final_syntax_failure_count": sum(entry["note"] is not None and entry["note"]["checked"]["syntax_failure"] is not None for entry in entries),
        "final_claims": {**counts, "eligible_history_count": eligible_histories,
            "submitted_claim_precision": ratio("verified_claim_count", "submitted_claim_count"),
            "known_issue_coverage": ratio("covered_known_issue_count", "known_issue_count"),
            "native_correction_coverage": ratio("covered_correction_opportunity_count", "correction_opportunity_count")}}
    if compact:
        # v2 bounded-claim metrics do not exist for compact notes; report their own verification instead.
        compact_notes = _compact_note_metrics(entries)
        repair_metrics.update(final_claims=None, compact_notes=compact_notes,
            final_false_accusation_count=compact_notes["refuted_entry_count"],
            false_accusation_count=repair_metrics["online_false_accusation_count"] + compact_notes["refuted_entry_count"])
    failed = sum(value != "CORRECT" for value in first.values())
    recovered = sum(first[pid] != "CORRECT" and final[pid] == "CORRECT" for pid in first)
    accuracy = sum(value == "CORRECT" for value in final.values()) / 10
    return {"first_statuses": first, "final_statuses": final,
        "first_attempt_accuracy": sum(value == "CORRECT" for value in first.values()) / 10,
        "final_accuracy": accuracy, "recovery_accuracy": accuracy, "recovered_count": recovered,
        "initially_failed": failed, "initially_incorrect": sum(value == "INCORRECT" for value in first.values()),
        "recovered_fraction": recovered / failed if failed else None,
        "first_status_counts": dict(Counter(first.values())), "final_status_counts": dict(Counter(final.values())),
        "note_count": None, "note_verified_count": None, "note_verified_fraction": None,
        "usage": {"input_tokens": sum(r["input"]["input_tokens"] for r in rows),
            "image_tokens": sum(r["input"]["image_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows)},
        "generation_seconds": sum(r["elapsed_seconds"] for r in rows), "generation_calls": len(rows),
        "length_terminated_calls": sum(r["finish_reason"] == "length" for r in rows), "repair_metrics": repair_metrics}


def diagnose(cohort: Path, output: Path, engine: Any, prompt_templates: dict, *,
             expected_preparation_sha256: str, local_protocol_sha256: str, deadline: float,
             context_limit: int = generation.CONTEXT_LIMIT, note_contract: str | None = None) -> dict:
    """All T1 calls precede n=1 retries, followed by one strict note per recovery."""
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("ladder recovery: finite monotonic deadline required")
    root, output = _safe_root(cohort), Path(output).absolute()
    if (output.exists() or output.is_relative_to(root) or ".." in output.parts
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("ladder recovery: preserve output and input directories")
    identity = _descriptor(root / "preparation.json")
    if not generation._sha256(expected_preparation_sha256) or identity["file_sha256"] != expected_preparation_sha256:
        raise ValueError("ladder recovery: pinned preparation SHA required")
    manifest, actors, sources = _json(root / "preparation.json"), load_ladder_actors(root), code_hashes()
    contract = protocol(prompt_templates, local_protocol_sha256, context_limit=context_limit, note_contract=note_contract)
    note = _note_contract(contract)
    if (manifest.get("status") != "CPU_COHORT_PREPARED_PENDING_GATES" or manifest.get("count") != 10
            or manifest.get("gpu_evaluation_completed") is not False
            or any(a.mode != "obs" or a.answer_format != "unit_number" or a.choices for a in actors)
            or type(engine.policy_version) is not int or engine.policy_version < 0
            or type(engine.context_limit) is not int or engine.context_limit != context_limit):
        raise ValueError("ladder recovery: native ten, explicit policy and context required")
    output = artifact_path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    result = {"format": VERSION, "status": "INCOMPLETE", "protocol": contract,
        "protocol_sha256": generation._digest(contract), "preparation": identity, "source_sha256": sources,
        "problem_ids": [a.problem_id for a in actors], "policy_version": engine.policy_version,
        "prompt_templates": copy.deepcopy(prompt_templates), "generation": None, "calls": [], "trees": [],
        **{key: None for key in _METRICS}, "first_statuses": {}, "final_statuses": {},
        "training_allowed": False, "evaluation_execution_authorized": False,
        "model_generation_provenance_verified": False, "module_causality_certified": False,
        "lesson_semantics_verified": False, "notebook_history_scope": "every-actual-attempt-including-success"}
    trees: dict[str, ProblemTree] = {}
    repair_native: dict[str, dict] = {}
    repair_online: dict[str, list] = {}
    if _repair(prompt_templates):
        result["fresh_observation_events"] = []

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("ladder recovery: evaluation deadline reached")

    def call(actor, stage, parent, *, feedback=None, messages=None):
        check_deadline()
        images = load_images(actor, root / "images")
        try:
            ids = continuation_ids(parent, feedback, engine.tokenizer) if feedback is not None else None
            request = RolloutRequest(actor.problem_id + ":" + stage, ids, messages, images, 1,
                                     contract["notebook"]["max_new_tokens"] if stage == "NOTE" else 1024,
                                     [_seed(actor.problem_id, stage)])
            inp = generation._input(actor, request, engine, context_limit=context_limit)
            check_deadline()
            begin = time.monotonic()
            sampled = _sample_request(request, engine, result["policy_version"], context_limit)
            elapsed = time.monotonic() - begin
            completion = sampled.completions[0]
            row = {"problem_id": actor.problem_id, "stage": stage, "request_id": request.request_id,
                "policy_version": result["policy_version"], "seed": request.seeds[0], "input": inp,
                "input_sha256": generation._digest(inp), "parent_sample_id": parent.sample_id,
                "feedback": feedback, "continuation_suffix_token_ids": None if ids is None else
                    ids[len(parent.prompt_token_ids) + len(parent.completion_token_ids):],
                "n": 1, "max_new_tokens": request.max_new_tokens, "context_limit": context_limit,
                "text": completion.text, "completion_token_ids": list(completion.token_ids),
                "logprobs": list(completion.logprobs), "completion_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason, "elapsed_seconds": elapsed}
            if _repair(prompt_templates):
                from .repair_contract import VERSION as retry_version
                row["response_contract"] = note if stage == "NOTE" else retry_version
            try:
                row["record_sha256"] = generation._digest(row)
            except (TypeError, ValueError):
                _write(output / f"invalid-call-{len(result['calls']):04d}.json", _bytes({
                    "request_id": request.request_id, "text": str(completion.text),
                    "completion_token_ids_repr": repr(completion.token_ids),
                    "logprobs_repr": repr(completion.logprobs), "finish_reason": str(completion.finish_reason)}) + b"\n")
                raise
            # Persist the literal completion before grading or post-generation checks.
            _write(output / f"call-{len(result['calls']):04d}.json", _bytes(row) + b"\n")
            result["calls"].append(row)
            check_deadline()
            first = result["generation"]["rows"][result["problem_ids"].index(actor.problem_id)]
            _validate_call(row, actor.problem_id, result["policy_version"], first, parent,
                           context_limit=context_limit, note_tokens=contract["notebook"]["max_new_tokens"],
                           repair_policy=_repair(prompt_templates))
            if (generation._input(actor, request, engine, context_limit=context_limit) != inp
                    or sampled.prompt_token_ids != inp["prompt_token_ids"]
                    or engine.tokenizer.decode(completion.token_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("ladder recovery: actual input or tokenizer decode changed")
            return row
        finally:
            for image in images:
                image.close()

    try:
        check_deadline()
        result["generation"] = generation.generate_ladder(root, output / "first", engine, prompt_templates,
            expected_preparation_sha256=expected_preparation_sha256,
            local_protocol_sha256=local_protocol_sha256, deadline=deadline, context_limit=context_limit)
        with _verifier(root) as verifier:
            _check_verifier(verifier, manifest)
            result["verifier_process"] = verifier.metadata
            for actor, row in zip(actors, result["generation"]["rows"], strict=True):
                if _repair(prompt_templates):
                    sample, native = _repair_sample(row, 1, None, verifier, actor)
                    repair_native[actor.problem_id] = {sample.sample_id: native}
                    repair_online[actor.problem_id] = []
                else:
                    sample = _sample(row, "T1", 1, None, verifier, actor)
                trees[actor.problem_id] = ProblemTree(actor.problem_id, result["policy_version"],
                    [_group(sample)], "VISION_OFF", {}, {"skipped_branches": []})
            for turn in (2, 3):
                for actor in actors:
                    tree = trees[actor.problem_id]
                    parent = notebook.solution_samples(tree)[-1]
                    if _repair(prompt_templates):
                        if parent.turn + 1 != turn:
                            continue
                        history = notebook.solution_samples(tree)
                        feedback = _repair_feedback(actor, prompt_templates, history, repair_native[actor.problem_id])
                        if feedback is None:
                            continue
                        if "fresh_observation" not in tree.notebook:
                            check_deadline()
                            original_images = load_images(actor, root / "images")
                            try:
                                def observation_event(event):
                                    payload = {"problem_id": actor.problem_id,
                                        "before_call_index": len(result["calls"]), "event": copy.deepcopy(event)}
                                    _write(output / f"fresh-{len(result['fresh_observation_events']):04d}.json",
                                           _bytes(payload) + b"\n")
                                    result["fresh_observation_events"].append(payload)

                                tree.notebook["fresh_observation"] = fresh.generate(actor,
                                    prompts.for_actor(actor, prompt_templates)["observation_instruction"][actor.family],
                                    original_images, engine, master_seed=3407, context_limit=context_limit,
                                    on_event=observation_event)
                            finally:
                                for image in original_images:
                                    image.close()
                        feedback = _repair_feedback(actor, prompt_templates, history,
                            repair_native[actor.problem_id], tree.notebook["fresh_observation"])
                        call_count = len(result["calls"])
                        try:
                            row = call(actor, f"T{turn}", parent, feedback=feedback)
                        except (TimeoutError, ValueError) as exc:
                            if len(result["calls"]) == call_count and (
                                    isinstance(exc, TimeoutError) or str(exc).startswith("CONTEXT_BLOCKED:")):
                                tree.notebook["skipped_branches"].append({"kind": f"T{turn}",
                                    "parent_sample_id": parent.sample_id,
                                    "status": "BUDGET_STOP" if isinstance(exc, TimeoutError) else "CONTEXT_BLOCKED"})
                            raise
                        sample, native = _repair_sample(row, turn, parent, verifier, actor)
                        tree.groups.append(_group(sample))
                        repair_native[actor.problem_id][sample.sample_id] = native
                        if sample.result is not None:
                            repair_online[actor.problem_id].append(_repair_online([*history, sample], actor, verifier))
                        continue
                    if parent.result.status in BLOCKING:
                        raise ValueError("ladder recovery: blocking verifier outcome")
                    if parent.result.status not in RETRYABLE:
                        continue
                    feedback = prompts.retry_text(parent.result.status, parent.result.extracted,
                        actor.mode, actor.family, prompts.for_actor(actor, prompt_templates))
                    row = call(actor, f"T{turn}", parent, feedback=feedback)
                    tree.groups.append(_group(_sample(row, f"T{turn}", turn, parent, verifier, actor)))
            for actor in actors:
                tree = trees[actor.problem_id]
                if _repair(prompt_templates):
                    audit = _repair_episode(tree, repair_native[actor.problem_id])
                    tree.route = audit["route"]
                    entry = {"problem_id": actor.problem_id, "tree": _wire(tree), "episode_audit": audit,
                        "online_audits": repair_online[actor.problem_id], "note": None}
                    result["trees"].append(entry)
                    if audit["success_sample_id"] is not None:
                        history = notebook.solution_samples(tree)
                        parent = next(sample for sample in history if sample.sample_id == audit["success_sample_id"])
                        feedback = _repair_original_feedback(actor.problem_id, result["calls"])
                        messages = _repair_note_messages(actor, prompt_templates, tree, audit, feedback, note)
                        row = call(actor, "NOTE", parent, messages=messages)
                        entry["note"] = {"request_id": row["request_id"],
                            "checked": _repair_note_grade(row, tree, audit, verifier, actor, note)}
                    continue
                audit = _audit(tree, actor, verifier, prompt_templates)
                tree.route = audit["route"]
                entry = {"problem_id": actor.problem_id, "tree": _wire(tree), "episode_audit": audit, "note": None}
                result["trees"].append(entry)
                if audit["success_attempt"] is not None:
                    parent = notebook.solution_samples(tree)[-1]
                    messages = _note_messages(actor, prompt_templates, tree, audit, len(actor.image_paths))
                    row = call(actor, "NOTE", parent, messages=messages)
                    checked = _grade_note(row, tree, audit, verifier, actor, prompt_templates)
                    entry["note"] = {"request_id": row["request_id"], "checked": checked}
            if (_descriptor(root / "preparation.json") != identity or load_ladder_actors(root) != actors
                    or code_hashes() != sources
                    or protocol(prompt_templates, local_protocol_sha256, context_limit=context_limit,
                                note_contract=note_contract) != contract
                    or result["generation"]["source_sha256"] != sources):
                raise ValueError("ladder recovery: source/input/protocol changed during evaluation")
            result.update(_repair_metrics(result["trees"], result["generation"]["rows"] + result["calls"], note)
                if _repair(prompt_templates) else _metrics(result["trees"], result["generation"]["rows"] + result["calls"],
                    reference_policy=_reference(prompt_templates)))
            check_deadline()
            result["status"] = "COMPLETE_NATIVE_LADDER_RECOVERY"
            result["record_sha256"] = generation._digest(result)
            _validate_complete(result)
    except Exception as exc:
        result.update(status="INCOMPLETE", reason=str(exc), error_type=type(exc).__name__)
        result.update({key: None for key in _METRICS})
        result["partial_trees"] = [_wire(tree) for tree in trees.values()]
    first_rows = result["generation"]["rows"] if result["generation"] else [
        _json(path) for path in sorted((output / "first").glob("response-*.json"))]
    if result["status"] == "INCOMPLETE":
        partial = _metrics([], first_rows + result["calls"])
        result.update({key: value for key, value in partial.items() if key not in _METRICS
                       and key not in ("first_statuses", "final_statuses")})
    result["partial_first_attempt_count"] = len(first_rows)
    result["record_sha256"] = generation._digest({k: v for k, v in result.items() if k != "record_sha256"})
    _write(output / "result.json", _bytes(result) + b"\n")
    return result


def _validate_complete(record):
    if (type(record) is not dict or record.get("format") != VERSION
            or record.get("status") != "COMPLETE_NATIVE_LADDER_RECOVERY"
            or type(record.get("protocol")) is not dict
            or type(record.get("prompt_templates")) is not dict
            or record.get("record_sha256") != generation._digest({k: v for k, v in record.items() if k != "record_sha256"})):
        raise ValueError("ladder recovery: complete bound native recovery record required")
    if any(record.get(key) is not False for key in ("training_allowed", "evaluation_execution_authorized",
            "model_generation_provenance_verified", "module_causality_certified", "lesson_semantics_verified")):
        raise ValueError("ladder recovery: evaluation evidence limits changed")
    note = _note_contract(record["protocol"])
    contract = protocol(record["prompt_templates"], record["protocol"].get("local_protocol_sha256"),
                        context_limit=record["protocol"].get("context_limit"),
                        **({"note_contract": note} if _repair(record["prompt_templates"]) else {}))
    if _bytes(record["protocol"]) != _bytes(contract) or record["protocol_sha256"] != generation._digest(contract):
        raise ValueError("ladder recovery: recovery protocol changed")
    first = record["generation"]
    generation.validate_record(first)
    if (any(record[key] != first[key] for key in ("preparation", "source_sha256", "problem_ids", "policy_version"))
            or _bytes(first["protocol"]) != _bytes(generation.protocol(record["prompt_templates"],
                contract["local_protocol_sha256"], context_limit=contract["context_limit"]))
            or len(record["trees"]) != 10 or [entry["problem_id"] for entry in record["trees"]] != record["problem_ids"]):
        raise ValueError("ladder recovery: nested first attempts or coverage changed")
    calls = {row["request_id"]: row for row in record["calls"]}
    if len(calls) != len(record["calls"]):
        raise ValueError("ladder recovery: duplicate literal calls")
    if _repair(record["prompt_templates"]):
        _validate_repair_complete(record, calls, first, contract)
        return
    used, expected_order = set(), []
    for actor_row, entry in zip(first["rows"], record["trees"], strict=True):
        actor = from_json(ActorItem, json.dumps(actor_row["input"]["actor"]))
        tree = from_json(ProblemTree, json.dumps(entry["tree"]))
        samples = notebook.solution_samples(tree)
        if (tree.problem_id != actor.problem_id or tree.policy_version != record["policy_version"]
                or not 1 <= len(samples) <= 3 or tree.notebook != {"skipped_branches": []}):
            raise ValueError("ladder recovery: invalid recorded solution coverage")
        for i, (sample, group) in enumerate(zip(samples, tree.groups, strict=True), 1):
            row = actor_row if i == 1 else calls.get(actor.problem_id + f":T{i}")
            if (row is None or group.usable or group.reason != _SINGLE or len(group.samples) != 1
                    or group.state_sha256 != _hash(sample.prompt_token_ids)):
                raise ValueError("ladder recovery: missing n=1 evaluation attempt")
            expected = Sample(row["request_id"] + ":0", actor.problem_id, row["request_id"], f"T{i}", i,
                samples[i-2].sample_id if i > 1 else None, row["input"]["prompt_token_ids"],
                row["completion_token_ids"], row["logprobs"], row["finish_reason"], row["text"],
                row["seed"], row["policy_version"], sample.result)
            if sample != expected or sample.result is None or sample.result.status in BLOCKING:
                raise ValueError("ladder recovery: stored attempt differs from literal response")
            if i > 1:
                parent = samples[i-2]
                _validate_call(row, actor.problem_id, record["policy_version"], actor_row, parent,
                               context_limit=contract["context_limit"], note_tokens=contract["notebook"]["max_new_tokens"])
                feedback = prompts.retry_text(parent.result.status, parent.result.extracted,
                    actor.mode, actor.family, prompts.for_actor(actor, record["prompt_templates"]))
                if row["feedback"] != feedback:
                    raise ValueError("ladder recovery: retry feedback changed")
                used.add(row["request_id"])
                expected_order.append((i, record["problem_ids"].index(actor.problem_id), row["request_id"]))
        if samples[-1].result.status in RETRYABLE and len(samples) != 3:
            raise ValueError("ladder recovery: retry coverage incomplete")
        audit = entry["episode_audit"]
        if (audit.get("solution_sha256") != notebook.solution_hash(tree)
                or audit.get("transcript_sha256") != _hash(notebook.public_transcript(samples))
                or audit.get("complete_episode") is not True
                or audit.get("training_complete_episode") is not False or audit.get("training_admission") is not False
                or audit.get("evaluation_only_single_sample_groups") is not True
                or tree.route != audit.get("route")
                or audit.get("success_attempt") != (len(samples) if len(samples) > 1
                    and samples[-1].result.status == "CORRECT" else None)):
            raise ValueError("ladder recovery: episode audit changed")
        note = entry["note"]
        if (note is not None) != (audit["success_attempt"] is not None):
            raise ValueError("ladder recovery: recovered-note coverage incomplete")
        if note is not None:
            row = calls.get(note["request_id"])
            if row is None or row["stage"] != "NOTE":
                raise ValueError("ladder recovery: missing actual note")
            _validate_call(row, actor.problem_id, record["policy_version"], actor_row, samples[-1],
                           context_limit=contract["context_limit"], note_tokens=contract["notebook"]["max_new_tokens"])
            if row["input"]["messages"] != _note_messages(actor, record["prompt_templates"], tree, audit, len(actor.image_paths)):
                raise ValueError("ladder recovery: whole-history notebook input changed")
            used.add(row["request_id"])
            expected_order.append((4, record["problem_ids"].index(actor.problem_id), row["request_id"]))
    if (used != set(calls) or [value[2] for value in sorted(expected_order)] != [row["request_id"] for row in record["calls"]]):
        raise ValueError("ladder recovery: extra, missing, or reordered evaluation calls")
    expected_metrics = _metrics(record["trees"], first["rows"] + record["calls"],
                                reference_policy=_reference(record["prompt_templates"]))
    if any(_bytes(record.get(key)) != _bytes(value) for key, value in expected_metrics.items()):
        raise ValueError("ladder recovery: metrics differ from complete actual history")


def _validate_repair_complete(record, calls, first, contract):
    from . import repair_contract
    used, order = set(), []
    expected_events: list[dict] = []
    for actor_row, entry in zip(first["rows"], record["trees"], strict=True):
        actor = from_json(ActorItem, json.dumps(actor_row["input"]["actor"]))
        tree = from_json(ProblemTree, json.dumps(entry["tree"]))
        samples = notebook.solution_samples(tree)
        native = entry["episode_audit"]["native_audits"]
        observation = tree.notebook.get("fresh_observation")
        if (observation is not None) != (len(samples) > 1):
            raise ValueError("ladder repair: one reread for each eligible retry history required")
        if observation is not None:
            fresh.validate(observation, policy_version=tree.policy_version, actor=actor,
                schema=prompts.for_actor(actor, record["prompt_templates"])["observation_instruction"][actor.family],
                master_seed=3407)
            if observation["image_key"] != actor_row["input"]["image_key"]:
                raise ValueError("ladder repair: fresh original images differ")
            before_index = next(index for index, row in enumerate(record["calls"])
                                if row["request_id"] == actor.problem_id + ":T2")
            expected_events.extend({"problem_id": actor.problem_id, "before_call_index": before_index, "event": event}
                                   for event in fresh.events_for(observation))
        if (tree.problem_id != actor.problem_id or tree.policy_version != record["policy_version"]
                or tree.notebook != {"skipped_branches": [], **({"fresh_observation": observation} if observation is not None else {})}
                or tree.route != notebook.strict_routing(tree)["route"]
                or not 1 <= len(samples) <= 3 or len(tree.groups) != len(samples)
                or list(native) != [sample.sample_id for sample in samples]):
            raise ValueError("ladder repair: actual solution coverage differs")
        for index, (sample, group) in enumerate(zip(samples, tree.groups, strict=True), 1):
            row = actor_row if index == 1 else calls.get(actor.problem_id + f":T{index}")
            if row is None or group != _group(sample):
                raise ValueError("ladder repair: n1 evaluation group changed")
            expected = Sample(row["request_id"] + ":0", actor.problem_id, row["request_id"], f"T{index}", index,
                samples[index-2].sample_id if index > 1 else None, row["input"]["prompt_token_ids"],
                row["completion_token_ids"], row["logprobs"], row["finish_reason"], row["text"],
                row["seed"], row["policy_version"], sample.result)
            checked = native[sample.sample_id]
            if sample != expected or checked["native"] is None and sample.result is not None:
                raise ValueError("ladder repair: literal response or unavailable result changed")
            if checked["native"] is not None:
                audit = checked["native"]
                if (checked["syntax_failure"] is not None or sample.result is None
                        or audit["sample_id"] != sample.sample_id or audit["raw_sha256"] != _raw_sha(sample.text)
                        or audit["response_contract"] != (None if index == 1 else repair_contract.VERSION)
                        or _bytes(audit["result"]) != _bytes(_wire(sample.result))):
                    raise ValueError("ladder repair: native audit binding changed")
            if index > 1:
                parent = samples[index-2]
                feedback = _repair_feedback(actor, record["prompt_templates"], samples[:index-1], native, observation)
                _validate_call(row, actor.problem_id, record["policy_version"], actor_row, parent,
                    context_limit=24576, note_tokens=8192, repair_policy=True)
                if (feedback is None or row["feedback"] != feedback
                        or row.get("response_contract") != repair_contract.VERSION):
                    raise ValueError("ladder repair: public typed retry differs")
                used.add(row["request_id"])
                order.append((index, record["problem_ids"].index(actor.problem_id), row["request_id"]))
        if len(samples) < 3 and _repair_feedback(actor, record["prompt_templates"], samples, native) is not None:
            raise ValueError("ladder repair: eligible retry omitted")
        expected_audit = _repair_episode(tree, native)
        if _bytes(entry["episode_audit"]) != _bytes(expected_audit):
            raise ValueError("ladder repair: public full history audit changed")
        online = entry["online_audits"]
        expected_children = [sample.sample_id for sample in samples[1:] if sample.result is not None]
        expected_contracts = [_repair_contracts(samples[:i+1]) for i, sample in enumerate(samples)
                              if i > 0 and sample.result is not None]
        if ([audit["child_id"] for audit in online] != expected_children or any(
                audit["source_bindings"]["response_contracts"] != contracts
                for audit, contracts in zip(online, expected_contracts, strict=True))):
            raise ValueError("ladder repair: online audit coverage changed")
        note = entry["note"]
        if (note is not None) != (expected_audit["success_sample_id"] is not None):
            raise ValueError("ladder repair: success note coverage changed")
        if note is not None:
            row = calls.get(note["request_id"])
            parent = next(sample for sample in samples if sample.sample_id == expected_audit["success_sample_id"])
            if row is None or row["stage"] != "NOTE" or row.get("response_contract") != _note_contract(contract):
                raise ValueError("ladder repair: typed final call missing")
            _validate_call(row, actor.problem_id, record["policy_version"], actor_row, parent,
                context_limit=24576, note_tokens=8192, repair_policy=True)
            messages = _repair_note_messages(actor, record["prompt_templates"], tree, expected_audit,
                                            _repair_original_feedback(actor.problem_id, record["calls"]),
                                            _note_contract(contract))
            if row["input"]["messages"] != messages or note["checked"]["text_sha256"] != _raw_sha(row["text"]):
                raise ValueError("ladder repair: final raw history binding changed")
            used.add(row["request_id"])
            order.append((4, record["problem_ids"].index(actor.problem_id), row["request_id"]))
    if _bytes(record.get("fresh_observation_events")) != _bytes(expected_events):
        raise ValueError("ladder repair: missing, duplicate or reordered fresh events")
    if used != set(calls) or [item[2] for item in sorted(order)] != [row["request_id"] for row in record["calls"]]:
        raise ValueError("ladder repair: extra or reordered calls")
    metrics = _repair_metrics(record["trees"], first["rows"] + record["calls"], _note_contract(contract))
    if any(_bytes(record.get(key)) != _bytes(value) for key, value in metrics.items()):
        raise ValueError("ladder repair: actual history metrics changed")


def _raw_sha(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replay(cohort: Path, record: dict, *, expected_preparation_sha256: str) -> dict:
    """Fresh private CPU regrading, without a model or tokenizer load."""
    _validate_complete(record)
    root = _safe_root(cohort)
    before = generation._digest(record)
    identity = _descriptor(root / "preparation.json")
    actors, sources = load_ladder_actors(root), code_hashes()
    if (identity["file_sha256"] != expected_preparation_sha256 or record["preparation"] != identity
            or record["problem_ids"] != [a.problem_id for a in actors] or record["source_sha256"] != sources):
        raise ValueError("ladder recovery: pinned source/preparation differs from replay")
    calls = {row["request_id"]: row for row in record["calls"]}
    with _verifier(root) as verifier:
        _check_verifier(verifier, _json(root / "preparation.json"))
        for actor, entry, first in zip(actors, record["trees"], record["generation"]["rows"], strict=True):
            if first["input"]["actor"] != _wire(actor):
                raise ValueError("ladder recovery: actual public actor changed")
            tree = from_json(ProblemTree, json.dumps(entry["tree"]))
            if _repair(record["prompt_templates"]):
                samples = notebook.solution_samples(tree)
                native, online = {}, []
                actual: list[Sample] = []
                for index, stored in enumerate(samples, 1):
                    row = first if index == 1 else calls[actor.problem_id + f":T{index}"]
                    sample, audited = _repair_sample(row, index, actual[-1] if actual else None, verifier, actor)
                    if sample != stored:
                        raise ValueError("ladder repair: fresh native result differs")
                    actual.append(sample)
                    native[sample.sample_id] = audited
                    if index > 1 and sample.result is not None:
                        online.append(_repair_online(actual, actor, verifier))
                audit = _repair_episode(tree, native)
                if _bytes(audit) != _bytes(entry["episode_audit"]) or _bytes(online) != _bytes(entry["online_audits"]):
                    raise ValueError("ladder repair: fresh repair audits differ")
                if entry["note"] is not None:
                    checked = _repair_note_grade(calls[entry["note"]["request_id"]], tree, audit, verifier, actor,
                                                 _note_contract(record["protocol"]))
                    if _bytes(checked) != _bytes(entry["note"]["checked"]):
                        raise ValueError("ladder repair: fresh final audit differs")
                continue
            audit = _audit(tree, actor, verifier, record["prompt_templates"])
            if _bytes(audit) != _bytes(entry["episode_audit"]):
                raise ValueError("ladder recovery: fresh episode grading differs")
            if entry["note"] is not None:
                row = calls[entry["note"]["request_id"]]
                checked = _grade_note(row, tree, audit, verifier, actor, record["prompt_templates"])
                if _bytes(checked) != _bytes(entry["note"]["checked"]):
                    raise ValueError("ladder recovery: fresh strict notebook grading differs")
        metadata = verifier.metadata
    if (_descriptor(root / "preparation.json") != identity or load_ladder_actors(root) != actors
            or code_hashes() != sources or generation._digest(record) != before):
        raise ValueError("ladder recovery: replay source/input/record changed")
    return {"status": "CPU_NATIVE_LADDER_RECOVERY_REPLAYED", "record_sha256": before,
            "verifier_process": metadata, **{key: record[key] for key in _METRICS},
            **({"metric_identity": record["protocol"]["notebook"]["metric_identity"]}
               if _reference(record["prompt_templates"]) else {}),
            **({"repair_metrics": record["repair_metrics"], "evaluator_version": REPAIR_EVALUATOR_VERSION}
               if _repair(record["prompt_templates"]) else {}),
            "model_generation_provenance_verified": False, "literal_feedback_tokenization_verified": False,
            "cpu_replay_only": True, "training_allowed": False, "module_causality_certified": False}


def validate_paired_records(left: dict, right: dict) -> dict:
    """Recorded conditions and arithmetic, not model provenance or learned causality."""
    for record in (left, right):
        _validate_complete(record)
        preparation = record["preparation"]
        if (type(preparation) is not dict or set(preparation) != {"path", "file_sha256"}
                or type(preparation["path"]) is not str
                or not generation._sha256(preparation["file_sha256"])):
            raise ValueError("ladder recovery: exact pinned preparation descriptor required")
        replay(Path(preparation["path"]).parent, record,
               expected_preparation_sha256=preparation["file_sha256"])
    generation.validate_paired_records(left["generation"], right["generation"])
    if any(left[key] != right[key] for key in ("protocol", "protocol_sha256", "preparation",
            "source_sha256", "problem_ids", "prompt_templates")):
        raise ValueError("ladder recovery: paired recovery conditions differ")
    gains = [pid for pid in left["problem_ids"] if left["first_statuses"][pid] != "CORRECT"
             and right["first_statuses"][pid] == "CORRECT"]
    losses = [pid for pid in left["problem_ids"] if left["first_statuses"][pid] == "CORRECT"
              and right["first_statuses"][pid] != "CORRECT"]
    return {"status": "RECORDED_PAIRED_CONDITIONS_MATCH", "n_pairs": 10,
        "first_gains": gains, "first_losses": losses,
        "first_attempt_accuracy_delta": right["first_attempt_accuracy"] - left["first_attempt_accuracy"],
        "left": {key: left[key] for key in _METRICS}, "right": {key: right[key] for key in _METRICS},
        **({"repair_metrics": {"left": left["repair_metrics"], "right": right["repair_metrics"]}}
           if _repair(left["prompt_templates"]) else {}),
        "model_generation_provenance_verified": False, "module_causality_certified": False,
        "training_allowed": False}
