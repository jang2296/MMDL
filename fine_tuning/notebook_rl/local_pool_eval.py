"""Generic n=1 recovery evaluation over a prepared pool's held-out evaluation IDs; no training.

Same request order, retry selection, typed retry contract, unrewarded fresh reread and compact
final note as the circuit ladder evaluator, reusing its helpers; the route policy, slotless
sources and prompts are the training protocol's. Callers own model loading, the capture/reload
binding and execution authority (local_eval evaluate-pool). Literal records and CPU replay do not
certify model provenance, lesson truth or module causality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from . import episode_notebook as notebook, fresh_observation as fresh, local_ladder_generation as generation
from . import local_ladder_recovery as recovery, prompts
from .approval import code_hashes
from .cli import artifact_path
from .dataset import _bytes, _write, load_images
from .engines import RolloutRequest
from .local_ladder import _descriptor, _json
from .rollout import RETRYABLE, REPAIR_SELECTION_POLICY, _hash, _sample_request, continuation_ids
from .schema import ProblemTree, Sample, from_json
from .vault import _safe_root
from .verifier_service import VerifierService

VERSION = "local-pool-recovery-evaluation-v1"
COMPLETE = "COMPLETE_POOL_RECOVERY"
COMPARISON_VERSION = "local-pool-paired-comparison-v1"
_LIMITS = {"training_allowed": False, "evaluation_execution_authorized": False,
           "model_generation_provenance_verified": False, "module_causality_certified": False,
           "lesson_semantics_verified": False}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol(templates: dict, local_protocol_sha256: str, *, context_limit: int, note_contract: str) -> dict:
    from . import compact_note, repair_contract, repair_credit

    if note_contract != compact_note.VERSION or "evaluation_prompt_policy" in templates:
        raise ValueError("pool evaluation: training-protocol prompts with the compact final note required")
    first = generation.protocol(templates, local_protocol_sha256, context_limit=context_limit)
    return {**first, "protocol_id": VERSION, "first_attempt_protocol_id": generation.VERSION, "max_attempts": 3,
        "route_policy": templates.get("repair_route_policy"),
        "seed_rule": "T1 first-attempt seed; retries/NOTE rollout.seeds(policy_version=0,branch="
                     + recovery.VERSION + ":stage)",
        "request_order": "all-T1; one-fresh-before-each-eligible-T2; ordered-T2; ordered-T3; ordered-recovered-NOTE",
        "retryable_statuses": sorted(RETRYABLE), "retry_correct_native_issues": True,
        "retry_selection_policy": REPAIR_SELECTION_POLICY,
        "response_contracts": {"T1": None, "T2": repair_contract.VERSION, "T3": repair_contract.VERSION},
        "retry_prompt_sha256": generation._digest(prompts.repair_retry_prompt_contract()),
        "notebook": {"n": 1, "max_new_tokens": 8192, "trigger": "actual-retry-and-independent-correct-source",
            "fresh_observation_policy": fresh.POLICY, "fresh_observation_prompt_sha256": _hash(fresh.PROMPT),
            "response_contract": compact_note.VERSION, "prompt_sha256": generation._digest(compact_note.PROMPT),
            "metric_identity": {"online": repair_credit.AUDIT_VERSION, "final": compact_note.VERSION},
            "grader": "compact_note.audit", "lesson_rewarded": False,
            "note_verified": "parsed note with >=1 verified entry and 0 refuted entries"},
        "metric_definitions": {
            "finalization_rate": "solution attempts whose public outcome is CORRECT or INCORRECT / all attempts",
            "length_terminated_fraction": "solution attempts with finish_reason length / all attempts",
            "recovered_fraction": "final CORRECT among first-attempt non-CORRECT items"}}


def _cohort(preparation: Path, expected_sha256: str) -> tuple[dict, dict, Path, list, dict]:
    from .local_data import POOL_FORMATS, POOL_STATUS
    from .local_run import read_actor_descriptor
    from .launch import read_descriptor
    from .dataset import SPLITS
    from .vault import read_actors

    identity = _descriptor(Path(preparation).absolute())
    if not generation._sha256(expected_sha256) or identity["file_sha256"] != expected_sha256:
        raise ValueError("pool evaluation: pinned preparation SHA required")
    prepared = _json(Path(identity["path"]))
    if prepared.get("format") not in POOL_FORMATS or prepared.get("status") != POOL_STATUS:
        raise ValueError("pool evaluation: validated pool preparation required")
    selection = read_descriptor(prepared["selection"])
    actors = read_actor_descriptor(prepared["evaluation_actors"])
    root = _safe_root(Path(prepared["build_root"]))
    # Actors only: this runs inside the guarded actor process, where lineage.jsonl is a blocked vault file.
    public = {a.problem_id: a for split in SPLITS for a in read_actors(root / split)}
    if (not actors or [a.problem_id for a in actors] != selection.get("evaluation_ids")
            or any(public.get(a.problem_id) != a for a in actors)):
        raise ValueError("pool evaluation: bound evaluation actors differ from the selection or the build")
    return identity, prepared, root, actors, selection


def _verifiers(stack: ExitStack, root: Path, splits: set[str]) -> dict[str, Any]:
    manifest = _json(root / "manifest.yaml")["split_manifests"]
    services = {}
    for split in sorted(splits):
        service = stack.enter_context(VerifierService(root / split, forbidden_roots=[Path.home() / "mmdl-data/evaluation"],
                                                      timeout=60))
        if service.metadata["vault_manifest_sha256"] != manifest[split]:
            raise ValueError("pool evaluation: fresh verifier identity differs")
        services[split] = service
    return services


def _episode(tree: ProblemTree, native: dict, policy: str | None) -> dict:
    routing = notebook.routing_for(tree, policy)
    return recovery._repair_episode(tree, native) | {"route": routing["route"], "routing": routing}


def _note_grade(row: dict, tree: ProblemTree, audit: dict, verifier: Any, actor, policy: str | None) -> dict:
    from . import compact_note, repair_notebook

    parent = next(s for s in notebook.solution_samples(tree) if s.sample_id == audit["success_sample_id"])
    sample = Sample(row["request_id"] + ":0", actor.problem_id, row["request_id"], "DIAG", parent.turn,
        parent.sample_id, row["input"]["prompt_token_ids"], row["completion_token_ids"], row["logprobs"],
        row["finish_reason"], row["text"], row["seed"], row["policy_version"])
    return repair_notebook._grade(sample, tree, actor, verifier, audit, compact_note.VERSION, policy)


def _legacy_names(value: Any) -> Any:
    """Records saved before the rename call the verifier-feedback recovery rate `final_accuracy`; read it as
    `recovery_rate_diagnostic` (never emitted under the old name: first-attempt accuracy is the only accuracy)."""
    if type(value) is not dict:
        return value
    if "final_accuracy" in value and "recovery_rate_diagnostic" in value:
        raise ValueError("pool evaluation: ambiguous recovery metric name")
    return {("recovery_rate_diagnostic" if key == "final_accuracy" else key): _legacy_names(item)
            for key, item in value.items()}


def _summary(items: list[dict]) -> dict:
    """`recovery_rate_diagnostic` = share CORRECT after verifier feedback; a diagnostic, not an accuracy."""
    n = len(items)
    first = sum(item["first_status"] == "CORRECT" for item in items)
    final = sum(item["final_status"] == "CORRECT" for item in items)
    recovered = sum(item["first_status"] != "CORRECT" and item["final_status"] == "CORRECT" for item in items)
    return {"n": n, "first_correct": first, "final_correct": final,
            "first_attempt_accuracy": first / n if n else None, "recovery_rate_diagnostic": final / n if n else None,
            "initially_failed": n - first, "recovered_count": recovered,
            "recovered_fraction": recovered / (n - first) if n - first else None}


def metrics(entries: list[dict], rows: list[dict], meta: dict) -> dict:
    """Recomputable from the literal record only; resources such as GPU memory are recorded separately."""
    from .repair_contract import public_attempt_outcome

    items = []
    for entry in entries:
        tree = from_json(ProblemTree, json.dumps(entry["tree"]))
        samples = notebook.solution_samples(tree)
        contracts = recovery._repair_contracts(samples)
        outcomes = [public_attempt_outcome(s.text, response_contract=contracts[s.sample_id], result=s.result)[0]
                    for s in samples]
        if any(outcome is None for outcome in outcomes):
            raise ValueError("pool evaluation: source outcome unavailable")
        note = None
        if entry["note"] is not None:
            checked = entry["note"]["checked"]
            audit = checked["audit"]
            parsed = checked["syntax_failure"] is None and audit is not None
            verified, refuted = (audit["verified"], audit["refuted"]) if parsed else (0, 0)
            note = {"text_sha256": checked["text_sha256"], "finish_reason": checked["finish_reason"],
                    "parsed": parsed, "reward": audit["reward"] if parsed else 0.0, "verified_entry_count": verified,
                    "refuted_entry_count": refuted, "verified": parsed and verified > 0 and refuted == 0}
        pid = entry["problem_id"]
        items.append({"problem_id": pid, **meta[pid], "first_status": outcomes[0], "final_status": outcomes[-1],
            "attempt_statuses": outcomes, "attempt_finish_reasons": [s.finish_reason for s in samples],
            "attempt_text_sha256": [_sha(s.text) for s in samples],
            "route": entry["episode_audit"]["routing"]["route"],
            "admission": entry["episode_audit"]["routing"]["admission"], "note": note})
    attempts = [(status, finish) for item in items
                for status, finish in zip(item["attempt_statuses"], item["attempt_finish_reasons"], strict=True)]
    notes = [item["note"] for item in items if item["note"] is not None]

    def grouped(key: str) -> dict:
        def name(item: dict) -> str:
            return "UNKNOWN" if item[key] is None else str(item[key])
        return {value: _summary([item for item in items if name(item) == value])
                for value in sorted({name(item) for item in items})}

    result = _summary(items) | {
        "first_status_counts": dict(Counter(item["first_status"] for item in items)),
        "final_status_counts": dict(Counter(item["final_status"] for item in items)),
        "attempt_count": len(attempts),
        "finalized_attempt_count": sum(status in ("CORRECT", "INCORRECT") for status, _ in attempts),
        "finalization_rate": sum(status in ("CORRECT", "INCORRECT") for status, _ in attempts) / len(attempts),
        "length_terminated_attempts": sum(finish == "length" for _, finish in attempts),
        "length_terminated_fraction": sum(finish == "length" for _, finish in attempts) / len(attempts),
        "note_count": len(notes), "parsed_note_count": sum(note["parsed"] for note in notes),
        "verified_entry_count": sum(note["verified_entry_count"] for note in notes),
        "refuted_entry_count": sum(note["refuted_entry_count"] for note in notes),
        "note_verified_count": sum(note["verified"] for note in notes),
        "note_verified_fraction": sum(note["verified"] for note in notes) / len(notes) if notes else None,
        "generation_calls": len(rows), "length_terminated_calls": sum(r["finish_reason"] == "length" for r in rows),
        "generation_seconds": sum(r["elapsed_seconds"] for r in rows),
        "usage": {"input_tokens": sum(r["input"]["input_tokens"] for r in rows),
                  "image_tokens": sum(r["input"]["image_tokens"] for r in rows),
                  "completion_tokens": sum(r["completion_tokens"] for r in rows)},
        "by_source": grouped("source"), "by_subject": grouped("subject")}
    return {"metrics": result, "items": items}


def _row(actor, request: RolloutRequest, engine, version: int, context_limit: int, stage: str,
         parent: Sample | None, feedback: str | None, ids: list[int] | None) -> dict:
    inp = generation._input(actor, request, engine, context_limit=context_limit)
    begin = time.monotonic()
    sampled = _sample_request(request, engine, version, context_limit)
    completion = sampled.completions[0]
    row = {"problem_id": actor.problem_id, "request_id": request.request_id, "policy_version": version,
        "seed": request.seeds[0], "input": inp, "input_sha256": generation._digest(inp), "n": 1,
        "max_new_tokens": request.max_new_tokens, "context_limit": context_limit, "text": completion.text,
        "completion_token_ids": list(completion.token_ids), "logprobs": list(completion.logprobs),
        "completion_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason,
        "elapsed_seconds": time.monotonic() - begin}
    if stage != "T1":
        row.update(stage=stage, parent_sample_id=parent.sample_id if parent else None, feedback=feedback,
                   continuation_suffix_token_ids=None if ids is None or parent is None else
                   ids[len(parent.prompt_token_ids) + len(parent.completion_token_ids):])
    row["record_sha256"] = generation._digest(row)
    if (sampled.prompt_token_ids != inp["prompt_token_ids"] or engine.tokenizer.decode(
            completion.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion.text):
        raise ValueError("pool evaluation: actual input or tokenizer decode changed")
    return row


def diagnose(preparation: Path, output: Path, engine: Any, templates: dict, *, expected_preparation_sha256: str,
             local_protocol_sha256: str, deadline: float, context_limit: int, note_contract: str,
             memory_probe: Callable[[], int] | None = None) -> dict:
    """All T1 calls precede n=1 retries, then one compact note per recovery; every literal call is kept."""
    from .repair_contract import VERSION as retry_version

    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("pool evaluation: finite monotonic deadline required")
    identity, prepared, root, actors, selection = _cohort(preparation, expected_preparation_sha256)
    contract = protocol(templates, local_protocol_sha256, context_limit=context_limit, note_contract=note_contract)
    policy = contract["route_policy"]
    if (type(engine.policy_version) is not int or engine.policy_version < 0
            or type(engine.context_limit) is not int or engine.context_limit != context_limit):
        raise ValueError("pool evaluation: explicit policy and context required")
    output = artifact_path(Path(output).absolute())
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(Path(identity["path"]).parent):
        raise ValueError("pool evaluation: preserve output and input directories")
    output.mkdir(mode=0o700, parents=True)
    version, sources = engine.policy_version, code_hashes()
    meta = {a.problem_id: {key: selection[name][a.problem_id] for key, name in
            (("source", "sources"), ("subject", "subjects"), ("split", "splits"))} for a in actors}
    result: dict = {"format": VERSION, "status": "INCOMPLETE", "protocol": contract,
        "protocol_sha256": generation._digest(contract), "preparation": identity, "source_sha256": sources,
        "problem_ids": [a.problem_id for a in actors], "item_meta": meta, "policy_version": version,
        "prompt_templates": copy.deepcopy(templates), "first_rows": [], "calls": [],
        "fresh_observation_events": [], "trees": [], "metrics": None, "items": None, **_LIMITS}

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("pool evaluation: deadline reached")

    def generate(actor, stage, parent=None, *, feedback=None, messages=None):
        check_deadline()
        images = load_images(actor, root / "images")
        try:
            if stage == "T1":
                request = RolloutRequest(actor.problem_id + ":T1", None, [prompts.user_message(
                    prompts.turn1_text(actor, templates), len(images))], images, 1, generation.OUTPUT_TOKENS,
                    [generation._seed(actor.problem_id)])
                ids = None
            else:
                ids = continuation_ids(parent, feedback, engine.tokenizer) if feedback is not None else None
                request = RolloutRequest(actor.problem_id + ":" + stage, ids, messages, images, 1,
                                         8192 if stage == "NOTE" else 1024, [recovery._seed(actor.problem_id, stage)])
            row = _row(actor, request, engine, version, context_limit, stage, parent, feedback, ids)
            if stage != "T1":
                row["response_contract"] = note_contract if stage == "NOTE" else retry_version
                row["record_sha256"] = generation._digest({k: v for k, v in row.items() if k != "record_sha256"})
            count = len(result["first_rows"] if stage == "T1" else result["calls"])
            _write(output / f"{'first' if stage == 'T1' else 'call'}-{count:04d}.json", _bytes(row) + b"\n")
            return row
        finally:
            for image in images:
                image.close()

    try:
        trees: dict[str, ProblemTree] = {}
        natives: dict[str, dict] = {}
        online: dict[str, list] = {}
        with ExitStack() as stack:
            verifiers = _verifiers(stack, root, {meta[a.problem_id]["split"] for a in actors})
            for actor in actors:
                row = generate(actor, "T1")
                generation._validate_row(row, context_limit=context_limit)
                result["first_rows"].append(row)
            for actor, row in zip(actors, result["first_rows"], strict=True):
                verifier = verifiers[meta[actor.problem_id]["split"]]
                sample, native = recovery._repair_sample(row, 1, None, verifier, actor)
                natives[actor.problem_id], online[actor.problem_id] = {sample.sample_id: native}, []
                trees[actor.problem_id] = ProblemTree(actor.problem_id, version, [recovery._group(sample)],
                                                      "VISION_OFF", {}, {"skipped_branches": []})
            for turn in (2, 3):
                for actor, first in zip(actors, result["first_rows"], strict=True):
                    pid, tree = actor.problem_id, trees[actor.problem_id]
                    verifier = verifiers[meta[pid]["split"]]
                    history = notebook.solution_samples(tree)
                    if history[-1].turn + 1 != turn or recovery._repair_feedback(
                            actor, templates, history, natives[pid]) is None:
                        continue
                    if "fresh_observation" not in tree.notebook:
                        check_deadline()
                        images = load_images(actor, root / "images")
                        try:
                            def observed(event, pid=pid):
                                payload = {"problem_id": pid, "before_call_index": len(result["calls"]),
                                           "event": copy.deepcopy(event)}
                                _write(output / f"fresh-{len(result['fresh_observation_events']):04d}.json",
                                       _bytes(payload) + b"\n")
                                result["fresh_observation_events"].append(payload)
                            tree.notebook["fresh_observation"] = fresh.generate(actor,
                                prompts.for_actor(actor, templates)["observation_instruction"][actor.family],
                                images, engine, master_seed=3407, context_limit=context_limit, on_event=observed)
                        finally:
                            for image in images:
                                image.close()
                    feedback = recovery._repair_feedback(actor, templates, history, natives[pid],
                                                         tree.notebook["fresh_observation"])
                    row = generate(actor, f"T{turn}", history[-1], feedback=feedback)
                    result["calls"].append(row)
                    recovery._validate_call(row, pid, version, first, history[-1], context_limit=context_limit,
                                            note_tokens=8192, repair_policy=True)
                    sample, native = recovery._repair_sample(row, turn, history[-1], verifier, actor)
                    tree.groups.append(recovery._group(sample))
                    natives[pid][sample.sample_id] = native
                    if sample.result is not None:
                        online[pid].append(recovery._repair_online([*history, sample], actor, verifier))
            for actor, first in zip(actors, result["first_rows"], strict=True):
                pid, tree = actor.problem_id, trees[actor.problem_id]
                audit = _episode(tree, natives[pid], policy)
                tree.route = notebook.wire_route(audit["route"])
                entry = {"problem_id": pid, "tree": recovery._wire(tree), "episode_audit": audit,
                         "online_audits": online[pid], "note": None}
                result["trees"].append(entry)
                if audit["success_sample_id"] is not None:
                    parent = next(s for s in notebook.solution_samples(tree) if s.sample_id == audit["success_sample_id"])
                    messages = recovery._repair_note_messages(actor, templates, tree, audit, None, note_contract)
                    row = generate(actor, "NOTE", parent, messages=messages)
                    result["calls"].append(row)
                    recovery._validate_call(row, pid, version, first, parent, context_limit=context_limit,
                                            note_tokens=8192, repair_policy=True)
                    entry["note"] = {"request_id": row["request_id"], "checked": _note_grade(
                        row, tree, audit, verifiers[meta[pid]["split"]], actor, policy)}
        if (_descriptor(Path(identity["path"])) != identity or code_hashes() != sources
                or protocol(templates, local_protocol_sha256, context_limit=context_limit,
                            note_contract=note_contract) != contract):
            raise ValueError("pool evaluation: source/input/protocol changed during evaluation")
        result.update(metrics(result["trees"], result["first_rows"] + result["calls"], meta))
        result["status"] = COMPLETE
    except Exception as exc:
        result.update(status="INCOMPLETE", reason=str(exc), error_type=type(exc).__name__, metrics=None, items=None)
    result["resources"] = {"peak_gpu_memory_bytes": memory_probe() if memory_probe is not None else None}
    result["record_sha256"] = generation._digest({k: v for k, v in result.items() if k != "record_sha256"})
    _write(output / "result.json", _bytes(result) + b"\n")
    return json.loads(_bytes(result))  # the exact record a later CPU replay reads


def _complete(record: dict) -> None:
    if (type(record) is not dict or record.get("format") != VERSION or record.get("status") != COMPLETE
            or any(record.get(key) is not value for key, value in _LIMITS.items())
            or record.get("record_sha256") != generation._digest({k: v for k, v in record.items() if k != "record_sha256"})
            or record.get("protocol_sha256") != generation._digest(record.get("protocol"))
            or record.get("protocol") != protocol(record["prompt_templates"], record["protocol"]["local_protocol_sha256"],
                context_limit=record["protocol"]["context_limit"],
                note_contract=record["protocol"]["notebook"]["response_contract"])):
        raise ValueError("pool evaluation: complete bound pool recovery record required")


def replay(record: dict, *, expected_preparation_sha256: str) -> dict:
    """Fresh private CPU regrading of the saved literal responses; no model or tokenizer load."""
    _complete(record)
    before = generation._digest(record)
    identity, _, root, actors, _ = _cohort(Path(record["preparation"]["path"]), expected_preparation_sha256)
    templates, policy = record["prompt_templates"], record["protocol"]["route_policy"]
    note_contract, context = record["protocol"]["notebook"]["response_contract"], record["protocol"]["context_limit"]
    meta, version = record["item_meta"], record["policy_version"]
    if (identity != record["preparation"] or record["problem_ids"] != [a.problem_id for a in actors]
            or record["source_sha256"] != code_hashes() or len(record["trees"]) != len(actors)):
        raise ValueError("pool evaluation: pinned source/preparation differs from replay")
    calls = {row["request_id"]: row for row in record["calls"]}
    if len(calls) != len(record["calls"]):
        raise ValueError("pool evaluation: duplicate literal calls")
    used: set[str] = set()
    order: list[tuple] = []
    events: list[dict] = []
    with ExitStack() as stack:
        verifiers = _verifiers(stack, root, {meta[a.problem_id]["split"] for a in actors})
        for actor, first, entry in zip(actors, record["first_rows"], record["trees"], strict=True):
            pid, verifier = actor.problem_id, verifiers[meta[actor.problem_id]["split"]]
            generation._validate_row(first, context_limit=context)
            if (first["problem_id"] != pid or first["input"]["actor"] != recovery._wire(actor)
                    or first["input"]["messages"] != [prompts.user_message(prompts.turn1_text(actor, templates),
                                                                           len(actor.image_paths))]
                    or first["record_sha256"] != generation._digest({k: v for k, v in first.items() if k != "record_sha256"})):
                raise ValueError("pool evaluation: literal first attempt changed")
            tree = from_json(ProblemTree, json.dumps(entry["tree"]))
            stored = notebook.solution_samples(tree)
            observation = tree.notebook.get("fresh_observation")
            if (observation is not None) != (len(stored) > 1) or tree.notebook != {
                    "skipped_branches": [], **({"fresh_observation": observation} if observation is not None else {})}:
                raise ValueError("pool evaluation: one reread for each eligible retry history required")
            if observation is not None:
                fresh.validate(observation, policy_version=tree.policy_version, actor=actor, master_seed=3407,
                               schema=prompts.for_actor(actor, templates)["observation_instruction"][actor.family])
                if observation["image_key"] != first["input"]["image_key"]:
                    raise ValueError("pool evaluation: fresh original images differ")
                before_index = next(index for index, row in enumerate(record["calls"]) if row["request_id"] == pid + ":T2")
                events.extend({"problem_id": pid, "before_call_index": before_index, "event": event}
                              for event in fresh.events_for(observation))
            actual: list[Sample] = []
            native: dict = {}
            online: list = []
            for index in range(1, len(stored) + 1):
                row = first if index == 1 else calls.get(f"{pid}:T{index}")
                if row is None:
                    raise ValueError("pool evaluation: missing literal retry")
                if index > 1:
                    feedback = recovery._repair_feedback(actor, templates, actual, native, observation)
                    recovery._validate_call(row, pid, version, first, actual[-1], context_limit=context,
                                            note_tokens=8192, repair_policy=True)
                    if feedback is None or row["feedback"] != feedback:
                        raise ValueError("pool evaluation: typed retry feedback differs")
                    used.add(row["request_id"])
                    order.append((index, record["problem_ids"].index(pid), row["request_id"]))
                sample, audited = recovery._repair_sample(row, index, actual[-1] if actual else None, verifier, actor)
                if sample != stored[index - 1]:
                    raise ValueError("pool evaluation: fresh native result differs")
                actual.append(sample)
                native[sample.sample_id] = audited
                if index > 1 and sample.result is not None:
                    online.append(recovery._repair_online(actual, actor, verifier))
            if len(actual) < 3 and recovery._repair_feedback(actor, templates, actual, native) is not None:
                raise ValueError("pool evaluation: eligible retry omitted")
            audit = _episode(tree, native, policy)
            if (_bytes(audit) != _bytes(entry["episode_audit"]) or _bytes(online) != _bytes(entry["online_audits"])
                    or tree.route != notebook.wire_route(audit["route"])
                    or (entry["note"] is not None) != (audit["success_sample_id"] is not None)):
                raise ValueError("pool evaluation: fresh repair audits differ")
            if entry["note"] is not None:
                row = calls.get(entry["note"]["request_id"])
                parent = next(s for s in actual if s.sample_id == audit["success_sample_id"])
                if row is None or row["stage"] != "NOTE" or row["response_contract"] != note_contract:
                    raise ValueError("pool evaluation: typed final call missing")
                recovery._validate_call(row, pid, version, first, parent, context_limit=context, note_tokens=8192,
                                        repair_policy=True)
                if (row["input"]["messages"] != recovery._repair_note_messages(actor, templates, tree, audit, None,
                                                                              note_contract)
                        or _bytes(_note_grade(row, tree, audit, verifier, actor, policy)) != _bytes(entry["note"]["checked"])):
                    raise ValueError("pool evaluation: fresh final note grade differs")
                used.add(row["request_id"])
                order.append((4, record["problem_ids"].index(pid), row["request_id"]))
    if (used != set(calls) or [item[2] for item in sorted(order)] != [row["request_id"] for row in record["calls"]]
            or _bytes(events) != _bytes(record["fresh_observation_events"])):
        raise ValueError("pool evaluation: extra, reordered or unbound calls/reread events")
    fresh_metrics = metrics(record["trees"], record["first_rows"] + record["calls"], meta)
    if _bytes(fresh_metrics) != _bytes({"metrics": _legacy_names(record["metrics"]), "items": record["items"]}):
        raise ValueError("pool evaluation: replayed metrics differ from the record")
    if generation._digest(record) != before or code_hashes() != record["source_sha256"]:
        raise ValueError("pool evaluation: replay source/record changed")
    return {"status": "CPU_POOL_RECOVERY_REPLAYED", "record_sha256": record["record_sha256"],
            "metrics": fresh_metrics["metrics"], "cpu_replay_only": True, **_LIMITS}


def compare(records: dict[str, dict], *, expected_preparation_sha256: str) -> dict:
    """Paired table over identical items; records over different item sets are never pooled."""
    if len(records) < 2:
        raise ValueError("pool comparison: at least two labelled records required")
    replays = {label: replay(record, expected_preparation_sha256=expected_preparation_sha256)
               for label, record in records.items()}
    base = next(iter(records.values()))
    for record in records.values():
        if any(record[key] != base[key] for key in ("problem_ids", "item_meta", "preparation", "protocol",
                                                    "protocol_sha256", "prompt_templates", "source_sha256")):
            raise ValueError("pool comparison: records differ in items, preparation or protocol; not pooled")
    items = {label: {row["problem_id"]: row for row in record["items"]} for label, record in records.items()}
    labels = list(records)
    rows = [{"problem_id": pid, **base["item_meta"][pid],
             **{label: {"first": items[label][pid]["first_status"], "final": items[label][pid]["final_status"],
                        "note_verified": None if items[label][pid]["note"] is None else items[label][pid]["note"]["verified"]}
                for label in labels}} for pid in base["problem_ids"]]
    keys = ("first_attempt_accuracy", "recovery_rate_diagnostic", "recovered_fraction", "note_count", "parsed_note_count",
            "verified_entry_count", "note_verified_fraction", "finalization_rate", "length_terminated_fraction",
            "generation_seconds")
    reference = labels[0]
    paired = {label: {"first_gains": [pid for pid in base["problem_ids"]
                                      if items[reference][pid]["first_status"] != "CORRECT"
                                      and items[label][pid]["first_status"] == "CORRECT"],
                      "first_losses": [pid for pid in base["problem_ids"]
                                       if items[reference][pid]["first_status"] == "CORRECT"
                                       and items[label][pid]["first_status"] != "CORRECT"]}
              for label in labels[1:]}
    return {"format": COMPARISON_VERSION, "n_items": len(base["problem_ids"]), "labels": labels,
            "reference": reference, "table": rows,
            "summary": {label: {key: replays[label]["metrics"][key] for key in keys} | {
                "peak_gpu_memory_bytes": record["resources"]["peak_gpu_memory_bytes"]} for label, record in records.items()},
            "by_source": {label: replayed["metrics"]["by_source"] for label, replayed in replays.items()},
            "by_subject": {label: replayed["metrics"]["by_subject"] for label, replayed in replays.items()},
            "paired_vs_reference": paired, "replayed_record_sha256": {k: v["record_sha256"] for k, v in replays.items()},
            "improvement_claimed": False, **_LIMITS}
