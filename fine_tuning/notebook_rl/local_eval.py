"""Fixed-ten NF4 circuit diagnosis; no training, teacher, or official-score claim.

The actor never opens gold. A fresh trusted subprocess grades each complete
ten-answer frontier and each post-recovery note. Python guards are not an OS
sandbox. Install this file only after the live capture and fresh reload close.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VERSION = "local-nf4-circuit-recovery-notebook-v3"
CONTEXT_LIMIT = 16384
NOTE_TOKENS = 2048
MAX_WIRE = 8 * 1024 * 1024
EVIDENCE_LIMITS = {"actual_token_input_verified": False, "model_provenance_verified": False,
                   "actual_pixel_tensor_values_verified": False}
NATIVE_RECOVERY_SUMMARY = ("first_statuses", "final_statuses", "first_status_counts", "final_status_counts",
    "first_attempt_accuracy", "final_accuracy",
    "recovery_accuracy", "recovered_count", "initially_failed", "initially_incorrect", "recovered_fraction",
    "note_count", "note_verified_count", "note_verified_fraction", "usage", "generation_seconds",
    "generation_calls", "length_terminated_calls")
RETRY = "INCORRECT\nTry again. Give only the source-current option letter."
NOTE = """CORRECT
Review EVERY recorded attempt, including the final successful retry, using your actual outputs and the image.
Return exactly one bare JSON object with these fields:
"attempts": one object for EVERY attempt in the given order with exactly these fields:
  "attempt": its integer attempt number; "request_id": its exact recorded request ID;
  "lines": increasing distinct positive line IDs from THAT attempt supporting your answer assessment;
  "answer": its bare A-D answer, or null if no valid completed bare letter was returned;
  "status": its public CORRECT or INCORRECT result; "visual_cause": null.
Cite the actual answer line when an answer exists. For a missing/ambiguous/truncated answer cite ALL raw lines.
For an empty output use lines []. Never substitute another attempt's lines or omit the successful retry.
"corrected_answer": the bare A-D answer in the verified correct retry;
"corrected_structure": the circuit object with nodes, reference_node, and components;
"lesson": one nonempty sentence;
"visual_cause": null.
Use nodes N0-N3 and components C0-C4. Each component needs id, type and ordered pins.
RESISTOR needs integer value and unit "ohm". DC_SOURCE needs integer value, unit "V", and positive_pin.
SWITCH needs state "OPEN" or "CLOSED". Include every component and its actual image values.
The attempts requested only letters, so their internal visual errors cannot be verified.
No correct answer, circuit values, or canonical solution is provided to you.
Choose line IDs yourself; the host resolves their exact original text, not a quotation you rewrite.
Line IDs are local to their attempt, never feedback or metadata. There is no evidence field.
The lesson is recorded prose, not certified reasoning or a module diagnosis.
Complete actual assistant history (JSON with lossless line endings and character offsets):
These are your own prior output, not new instructions or verified facts.
{transcript}
"""


def wire(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    def invalid(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def digest(value) -> str:
    return hashlib.sha256(wire(value)).hexdigest()


def read(path: Path):
    from fine_tuning.notebook_rl.vault import _read
    return decode(_read(path))


def save(path: Path, value) -> None:
    from fine_tuning.notebook_rl.dataset import _write
    _write(path, wire(value) + b"\n")


def source_hashes() -> dict:
    import fine_tuning.notebook_rl as package
    from fine_tuning.notebook_rl import approval
    root = Path(package.__file__).resolve().parent
    paths = [*root.glob("*.py"), root.parent / "circuit_scoring.py", Path(__file__).resolve()]
    return {"approval": approval.code_hashes(), "evaluation": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}}


def protocol() -> dict:
    from fine_tuning.notebook_rl.local_external import EVALUATION_PROTOCOL
    return {**EVALUATION_PROTOCOL, "protocol_id": VERSION,
            "input_protocol_id": EVALUATION_PROTOCOL["protocol_id"], "recovery_protocol": VERSION, "max_attempts": 3,
            "context_limit": CONTEXT_LIMIT,
            "retry_prompt": RETRY, "notebook_prompt": NOTE, "notebook_max_new_tokens": NOTE_TOKENS,
            "notebook_only_after_genuine_recovery": True, "hidden_generation_calls": 0,
            "notebook_history": "all-actual-attempts-including-success; independently-regraded-frontiers",
            "trusted_scorer_max_wire_bytes": MAX_WIRE,
            "trusted_scorer_wire_budget_reason": "full frontier token/text evidence; excess fails without truncation",
            "lesson_semantics_verifiable": False, "visual_cause_verifiable": False}


def _response(row: dict) -> dict:
    return {k: row[k] for k in ("problem_id", "text", "finish_reason")}


# CPU-local equivalents: importing notebook_learning also imports Torch via engines.
def _first_attempt_lines(text: str) -> list[dict]:
    """Lossless Unicode-character offsets; never normalize or strip line endings."""
    spans, offset = [], 0
    for line_id, line in enumerate(text.splitlines(keepends=True), 1):
        spans.append({"line_id": line_id, "start": offset, "end": offset + len(line), "text": line})
        offset += len(line)
    return spans


def _answer(row: dict) -> str | None:
    from fine_tuning.circuit_scoring import extract_response
    try:
        return extract_response(row["text"], "qa") if row["finish_reason"] in ("stop", "eos") else None
    except ValueError:
        return None


def note_feedback(history: list[dict], statuses: list[str]) -> str:
    if len(history) != len(statuses) or not 2 <= len(history) <= 3:
        raise ValueError("complete recovered attempt history required")
    transcript = [{"attempt": i, "request_id": row["request_id"],
        "parent_request_id": row["parent_request_id"], "status": status,
        "finish_reason": row["finish_reason"], "lines": _first_attempt_lines(row["text"])}
        for i, (row, status) in enumerate(zip(history, statuses, strict=True), 1)]
    # Reversible JSON escapes keep model-produced role delimiters inside quoted data.
    raw = wire(transcript).decode().replace("<", "\\u003c").replace(">", "\\u003e")
    return NOTE.format(transcript=raw)


def score_note(note: dict, history: list[dict], statuses: list[str], structure_target: str) -> dict:
    """CPU-only per-attempt citations/answers and registered circuit; no cause claim."""
    from fine_tuning.circuit_scoring import score_circuit
    if (not 2 <= len(history) <= 3 or len(history) != len(statuses)
            or statuses != ["INCORRECT"] * (len(history)-1) + ["CORRECT"]
            or any(row.get("request_id") != history[0]["problem_id"] + f":T{i}"
                   or row.get("problem_id") != history[0]["problem_id"]
                   for i, row in enumerate(history, 1))):
        raise ValueError("complete actual recovered attempt identities required")
    checks: dict[str, Any] = {"format_valid": False, "history_verified": False, "previous_answer_verified": False,
              "corrected_answer_verified": False, "corrected_structure_verified": False,
              "visual_cause_verified": None, "visual_cause_status": "UNSUPPORTED_BARE_LETTER_ATTEMPTS",
              "lesson_semantics_verified": False, "module_causality_certified": False,
              "attempt_checks": [], "history_sha256": digest(history), **EVIDENCE_LIMITS}
    if note["finish_reason"] not in ("stop", "eos"):
        return checks | {"verified": False}
    try:
        obj = decode(note["text"])
        fields = {"attempts", "corrected_answer", "corrected_structure", "lesson", "visual_cause"}
        if (type(obj) is not dict or set(obj) != fields
                or type(obj["lesson"]) is not str or not obj["lesson"].strip()
                or obj["visual_cause"] is not None or type(obj["attempts"]) is not list
                or len(obj["attempts"]) != len(history)):
            return checks | {"verified": False}
        for i, (claim, row, status) in enumerate(zip(obj["attempts"], history, statuses, strict=True), 1):
            if (type(claim) is not dict or set(claim) != {
                    "attempt", "request_id", "lines", "answer", "status", "visual_cause"}
                    or type(claim["attempt"]) is not int or claim["attempt"] != i
                    or claim["request_id"] != row["request_id"] or claim["visual_cause"] is not None
                    or (claim["answer"] is not None and type(claim["answer"]) is not str)):
                return checks | {"verified": False}
            selected, lines = claim["lines"], _first_attempt_lines(row["text"])
            if (type(selected) is not list or any(type(k) is not int or k < 1 or k > len(lines) for k in selected)
                    or any(a >= b for a, b in zip(selected, selected[1:]))):
                return checks | {"verified": False}
            spans = [lines[k-1] for k in selected]
            answer = _answer(row)
            # A header cannot prove an answer; absence requires the entire raw output.
            evidence = (_answer({"text": "".join(s["text"] for s in spans), "finish_reason": "stop"}) == answer
                        if answer is not None else selected == list(range(1, len(lines)+1)))
            checks["attempt_checks"].append({"attempt": i, "request_id": row["request_id"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "resolved_lines": spans, "history_verified": evidence,
                "answer_verified": claim["answer"] == answer, "status_verified": claim["status"] == status})
        checks["format_valid"] = True
        checks["history_verified"] = all(r["history_verified"] and r["status_verified"] for r in checks["attempt_checks"])
        checks["previous_answer_verified"] = all(r["answer_verified"] for r in checks["attempt_checks"])
        checks["corrected_answer_verified"] = obj["corrected_answer"] == _answer(history[-1])
        checks["corrected_structure_verified"] = score_circuit(
            wire(obj["corrected_structure"]).decode(), structure_target, "structure")["correct"]
    except (ValueError, TypeError, KeyError, RecursionError):
        return checks | {"verified": False}
    return checks | {"verified": all(checks[k] for k in (
        "format_valid", "history_verified", "previous_answer_verified", "corrected_answer_verified",
        "corrected_structure_verified"))}


def _validate_call(row: dict, pid: str, stage: str) -> None:
    if (type(row) is not dict or row.get("problem_id") != pid or row.get("stage") != stage
            or row.get("request_id") != f"{pid}:{stage}" or type(row.get("text")) is not str
            or row.get("finish_reason") not in ("stop", "eos", "length")
            or row.get("seed") != seed(pid, stage) or type(row.get("n")) is not int or row["n"] != 1
            or type(row.get("policy_version")) is not int or row["policy_version"] < 0
            or row.get("max_new_tokens") != (NOTE_TOKENS if stage == "NOTE" else 1024)):
        raise ValueError("recorded generated-call identity required")
    for name in ("prompt", "completion"):
        tokens = row.get(name + "_token_ids")
        if (type(tokens) is not list or (name == "prompt" and not tokens)
                or any(type(k) is not int or k < 0 for k in tokens)
                or row.get(name + "_sha256") != digest(tokens)):
            raise ValueError("recorded token binding changed")
    if row.get("text_sha256") != hashlib.sha256(row["text"].encode()).hexdigest():
        raise ValueError("recorded text binding changed")
    if (len(row["prompt_token_ids"]) + row["max_new_tokens"] > CONTEXT_LIMIT
            or len(row["completion_token_ids"]) > row["max_new_tokens"]):
        raise ValueError("actual generated call exceeds the frozen token budget")
    pixels = row.get("pixel_tensors")
    if type(pixels) is not dict or set(pixels) != {"pixel_values", "image_grid_thw"}:
        raise ValueError("complete recorded pixel tensor summaries required")
    for name, summary in pixels.items():
        if (type(summary) is not dict or set(summary) != {"shape", "dtype", "sha256"}
                or type(summary["shape"]) is not list or len(summary["shape"]) != 2
                or any(type(n) is not int or n <= 0 for n in summary["shape"])
                or type(summary["sha256"]) is not str or len(summary["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in summary["sha256"])
                or (name == "image_grid_thw" and (summary["shape"] != [1, 3]
                    or summary["dtype"] != "torch.int64"))
                or (name == "pixel_values" and summary["dtype"] not in (
                    "torch.float32", "torch.float16", "torch.bfloat16"))):
            raise ValueError("invalid recorded pixel tensor summary")


def _note_messages(actor, feedback: str | None) -> list[dict]:
    from fine_tuning.notebook_rl.local_external import external_messages
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("notebook request requires actual feedback")
    messages = external_messages(actor)
    messages[0]["content"][-1]["text"] += "\n\n" + feedback
    return messages


def _audit_frontiers(cohort: Path, journals: list, responses: list, actors: list) -> tuple[dict, dict]:
    """Derive histories from every freshly regraded full frontier, never a supplied first answer."""
    from fine_tuning.notebook_rl.local_external import external_messages, score_external_responses
    ids = [a.problem_id for a in actors]
    if type(journals) is not list or not 2 <= len(journals) <= 3:
        raise ValueError("full actual frontier journal required")
    histories: dict[str, list] = {pid: [] for pid in ids}
    outcomes: dict[str, list] = {pid: [] for pid in ids}
    previous: dict[str, dict] = {}
    previous_status: dict[str, str] = {}
    for turn, rows in enumerate(journals, 1):
        if (type(rows) is not list or len(rows) != len(ids)
                or any(type(r) is not dict for r in rows) or [r.get("problem_id") for r in rows] != ids):
            raise ValueError("ordered complete frontier required")
        for actor, row in zip(actors, rows, strict=True):
            pid = actor.problem_id
            if turn > 1 and previous_status[pid] == "CORRECT":
                if row != previous[pid]:
                    raise ValueError("correct frontier entry was resampled or changed")
                continue
            _validate_call(row, pid, f"T{turn}")
            if row["policy_version"] != journals[0][0]["policy_version"]:
                raise ValueError("evaluation history mixes policies")
            if row.get("image_sha256") != list(actor.image_sha256):
                raise ValueError("actual original image binding changed")
            if turn == 1:
                if (row.get("parent_request_id") is not None or row.get("feedback") is not None
                        or row.get("messages") != external_messages(actor)):
                    raise ValueError("actual original question binding changed")
            else:
                parent = previous[pid]
                prefix = parent["prompt_token_ids"] + parent["completion_token_ids"]
                if (row.get("parent_request_id") != parent["request_id"] or row.get("feedback") != RETRY
                        or row.get("messages") is not None
                        or len(row["prompt_token_ids"]) <= len(prefix)
                        or row["prompt_token_ids"][:len(prefix)] != prefix
                        or row.get("pixel_tensors") != parent.get("pixel_tensors")):
                    raise ValueError("actual retry parent chain changed")
            histories[pid].append(row)
        scored = score_external_responses(cohort, [_response(r) for r in rows])
        if scored.get("status") != "COMPLETE":
            raise ValueError("complete independently regraded frontier required")
        current_status = {r["problem_id"]: "CORRECT" if r["correct"] else "INCORRECT" for r in scored["rows"]}
        for row in rows:
            pid = row["problem_id"]
            if turn == 1 or previous_status[pid] == "INCORRECT":
                outcomes[pid].append(current_status[pid])
        previous, previous_status = dict(zip(ids, rows, strict=True)), current_status
    if [_response(previous[pid]) for pid in ids] != responses:
        raise ValueError("journal differs from the current actual frontier")
    return histories, outcomes


def worker(payload: dict) -> dict:
    """Only this fresh process reads target bytes; responses contain bounded booleans."""
    from fine_tuning.notebook_rl.local_external import (
        FROZEN_MANIFEST_SHA256, _actor_inputs, _rows, score_external_responses)
    from fine_tuning.notebook_rl.guards import install_guards
    if (type(payload) is not dict or set(payload) != {"kind", "cohort", "manifest_sha256", "responses", "note"}
            or payload["kind"] not in ("frontier", "note")
            or payload["manifest_sha256"] != FROZEN_MANIFEST_SHA256):
        raise ValueError("unregistered scorer request")
    cohort = Path(payload["cohort"])
    install_guards(forbidden_roots=[Path.home() / "mmdl-data/evaluation"],
                   vault_files=[], allow_vault=True, allow_benchmark=False)
    scored = score_external_responses(cohort, payload["responses"])
    if scored["status"] != "COMPLETE":
        return {"status": "INCOMPLETE", "n_received": scored["n_received"], "rows": [], **EVIDENCE_LIMITS}
    rows = [{"problem_id": row["problem_id"], "status": "CORRECT" if row["correct"] else "INCORRECT"}
            for row in scored["rows"]]
    if payload["kind"] == "frontier":
        if payload["note"] is not None:
            raise ValueError("frontier has no notebook payload")
        return {"status": "COMPLETE", "n_correct": scored["n_correct"], "rows": rows,
                "responses_sha256": scored["responses_sha256"], **EVIDENCE_LIMITS}
    request = payload["note"]
    if type(request) is not dict or set(request) != {"problem_id", "frontiers", "note"}:
        raise ValueError("bounded notebook request required")
    pid = request["problem_id"]
    manifest, actors, selected = _actor_inputs(cohort, FROZEN_MANIFEST_SHA256)
    histories, outcomes = _audit_frontiers(cohort, request["frontiers"], payload["responses"], actors)
    if (pid not in histories or len(histories[pid]) < 2
            or outcomes[pid] != ["INCORRECT"] * (len(histories[pid])-1) + ["CORRECT"]):
        raise ValueError("notebook requires an actual incorrect-to-correct history")
    history, statuses = histories[pid], outcomes[pid]
    note = request["note"]
    _validate_call(note, pid, "NOTE")
    actor = next(a for a in actors if a.problem_id == pid)
    feedback = note_feedback(history, statuses)
    if (note.get("parent_request_id") != history[-1]["request_id"] or note.get("feedback") != feedback
            or note["policy_version"] != history[-1]["policy_version"]
            or note.get("messages") != _note_messages(actor, feedback)
            or note.get("image_sha256") != list(actor.image_sha256)
            or note.get("pixel_tensors") != history[-1].get("pixel_tensors")):
        raise ValueError("notebook prompt is not bound to the complete actual history")
    entry = next(r for r in selected if r["id"] == pid)
    gold = _rows(cohort / "gold.jsonl", manifest["gold_sha256"])
    target = [r for r in gold if r["root_id"] == entry["root_id"]
              and r["condition"] == "A0" and r["view"] == "structure"]
    if len(target) != 1 or target[0].get("metric_kind") != "circuit_structure":
        raise ValueError("unique registered corrected-structure target required")
    return {"status": "COMPLETE", "problem_id": pid,
            "frontiers_sha256": digest(request["frontiers"]),
            "checks": score_note(note, history, statuses, target[0]["target"]), **EVIDENCE_LIMITS}


def trusted(cohort: Path, responses: list[dict], note: dict | None = None) -> dict:
    import fine_tuning
    from fine_tuning.notebook_rl.local_external import FROZEN_MANIFEST_SHA256
    payload = {"kind": "note" if note is not None else "frontier", "cohort": str(cohort),
               "manifest_sha256": FROZEN_MANIFEST_SHA256, "responses": responses, "note": note}
    raw = wire(payload)
    if len(raw) > MAX_WIRE:
        raise ValueError("scorer request exceeds bound")
    environment = dict(os.environ)
    root = str(Path(fine_tuning.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker"], input=raw,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False, env=environment)
    if result.returncode or len(result.stdout) > MAX_WIRE or result.stderr:
        raise RuntimeError("trusted scorer failed; no in-process fallback")
    value = decode(result.stdout)
    if type(value) is not dict or value.get("status") not in ("COMPLETE", "INCOMPLETE"):
        raise RuntimeError("invalid bounded scorer result")
    return value | {"scorer_request_bytes": len(raw), "scorer_response_bytes": len(result.stdout)}


def frontier(actors, latest: dict) -> list[dict]:
    if set(latest) != {a.problem_id for a in actors}:
        raise ValueError("full ten frontier required, retaining previous correct answers")
    return [_response(latest[a.problem_id]) for a in actors]


def continuation(row: dict, feedback: str | None, tokenizer) -> list[int]:
    from fine_tuning.notebook_rl.rollout import continuation_ids
    if not isinstance(feedback, str) or not feedback:
        raise ValueError("continuation requires actual feedback")
    history: Any = SimpleNamespace(prompt_token_ids=row["prompt_token_ids"],
                                   completion_token_ids=row["completion_token_ids"])
    return continuation_ids(history, feedback, tokenizer)


def seed(pid: str, stage: str) -> int:
    # Policy-independent seeds permit paired F0/FT requests without changing conditions.
    return int.from_bytes(hashlib.sha256(f"3407:{VERSION}:{pid}:{stage}".encode()).digest()[:8], "big") % 2**31


def generate(actor, images, engine, stage: str, output: Path, *, parent: dict | None = None,
             feedback: str | None = None) -> dict:
    from fine_tuning.notebook_rl.engines import RolloutRequest
    from fine_tuning.notebook_rl.input_preparation import tensor_identity
    from fine_tuning.notebook_rl.local_external import external_messages
    from fine_tuning.notebook_rl.rollout import _sample_request
    is_note = stage == "NOTE"
    literal = None if parent is None or is_note else continuation(parent, feedback, engine.tokenizer)
    messages = (_note_messages(actor, feedback) if is_note else external_messages(actor) if parent is None else None)
    request = RolloutRequest(f"{actor.problem_id}:{stage}", literal,
        messages, images, 1, NOTE_TOKENS if is_note else 1024, [seed(actor.problem_id, stage)])
    prepared = engine.prepare(request)
    began = time.monotonic()
    actual = _sample_request(request, engine, engine.policy_version, CONTEXT_LIMIT)
    completion = actual.completions[0]
    row = {"problem_id": actor.problem_id, "stage": stage, "request_id": request.request_id,
        "parent_request_id": None if parent is None else parent["request_id"], "feedback": feedback,
        "messages": request.messages, "prompt_token_ids": actual.prompt_token_ids,
        "prompt_sha256": digest(actual.prompt_token_ids), "completion_token_ids": completion.token_ids,
        "completion_sha256": digest(completion.token_ids), "text": completion.text,
        "text_sha256": hashlib.sha256(completion.text.encode()).hexdigest(),
        "finish_reason": completion.finish_reason, "seed": request.seeds[0],
        "n": request.n, "max_new_tokens": request.max_new_tokens,
        "sampler_logprobs": completion.logprobs, "input_tokens": len(actual.prompt_token_ids),
        "output_tokens": len(completion.token_ids), "seconds": time.monotonic() - began,
        "image_sha256": list(actor.image_sha256),
        "processed_image_sizes": [list(im.size) for im in prepared.images],
        "image_grid_thw": prepared.pixels["image_grid_thw"].tolist(),
        "image_tokens": actual.prompt_token_ids.count(engine.model_config.image_token_id),
        "pixel_tensors": {k: tensor_identity(v) for k, v in prepared.pixels.items()},
        "policy_version": engine.policy_version}
    save(output / f"call-{request.request_id}.json", row)
    return row


def diagnose(actors, image_loader, engine, cohort: Path, output: Path, *, deadline: float) -> dict:
    first, latest, attempts, notes = {}, {}, {}, {}
    calls, journals = [], []
    histories: dict[str, list] = {}
    outcomes: dict[str, list] = {}
    result: dict = {"status": "INCOMPLETE", "first_attempt_accuracy": None, "recovery_accuracy": None,
              "note_verified_fraction": None, "visual_cause_verified": None,
              "visual_cause_status": "UNSUPPORTED_BARE_LETTER_ATTEMPTS",
              "notebook_history_scope": "every-actual-attempt-including-success",
              "lesson_semantics_verified": False, "module_causality_certified": False,
              "generation_calls": 0, "hidden_generation_calls": 0, "attempt_limit_includes_first": 3,
              **EVIDENCE_LIMITS}
    def call(actor, stage, parent=None, feedback=None):
        if time.monotonic() >= deadline:
            raise TimeoutError("local evaluation budget exhausted")
        images = image_loader(actor)
        try:
            row = generate(actor, images, engine, stage, output, parent=parent, feedback=feedback)
            calls.append(row)
            result["generation_calls"] += 1
            return row
        finally:
            for image in images:
                image.close()
    try:
        for actor in actors:
            row = call(actor, "T1")
            first[actor.problem_id] = latest[actor.problem_id] = row
            attempts[actor.problem_id] = 1
            histories[actor.problem_id] = [row]
        journals.append([latest[a.problem_id] for a in actors])
        initial = trusted(cohort, frontier(actors, first))
        save(output / "frontier-T1.json", initial)
        if initial["status"] != "COMPLETE":
            raise ValueError("first full ten scorer incomplete")
        statuses = {r["problem_id"]: r["status"] for r in initial["rows"]}
        first_statuses = dict(statuses)
        outcomes = {pid: [status] for pid, status in statuses.items()}
        result["first_attempt_accuracy"] = initial["n_correct"] / 10
        for turn in (2, 3):
            pending = [a for a in actors if statuses[a.problem_id] == "INCORRECT"]
            if not pending:
                break
            for actor in pending:
                pid = actor.problem_id
                latest[pid] = call(actor, f"T{turn}", latest[pid], RETRY)
                histories[pid].append(latest[pid])
                attempts[pid] += 1
            journals.append([latest[a.problem_id] for a in actors])
            judged = trusted(cohort, frontier(actors, latest))
            save(output / f"frontier-T{turn}.json", judged)
            if judged["status"] != "COMPLETE":
                raise ValueError("retry full ten scorer incomplete")
            statuses = {r["problem_id"]: r["status"] for r in judged["rows"]}
            for actor in pending:
                outcomes[actor.problem_id].append(statuses[actor.problem_id])
        recovered = [a for a in actors if first_statuses[a.problem_id] == "INCORRECT"
                     and statuses[a.problem_id] == "CORRECT"]
        save(output / "frontier-history.json", {"protocol": VERSION, "frontiers": journals})
        result["frontier_history_sha256"] = digest(journals)
        for actor in recovered:
            pid = actor.problem_id
            note = call(actor, "NOTE", latest[pid], note_feedback(histories[pid], outcomes[pid]))
            checked = trusted(cohort, frontier(actors, latest), {"problem_id": pid,
                "frontiers": journals, "note": note})
            save(output / f"note-check-{pid}.json", checked)
            if checked["status"] != "COMPLETE":
                raise ValueError("notebook scorer incomplete")
            notes[pid] = checked["checks"]
        result.update(status="COMPLETE", recovery_accuracy=sum(v == "CORRECT" for v in statuses.values()) / 10,
            recovered_count=len(recovered), initially_incorrect=10 - initial["n_correct"],
            recovered_fraction=None if initial["n_correct"] == 10 else len(recovered)/(10-initial["n_correct"]),
            note_verified_count=sum(n["verified"] for n in notes.values()),
            note_verified_fraction=None if not notes else sum(n["verified"] for n in notes.values())/len(notes),
            final_statuses=statuses, first_statuses=first_statuses)
    except Exception as exc:
        result.update(status="INCOMPLETE", error_type=type(exc).__name__, error=str(exc))
        # Preserve completed first-attempt arithmetic, but never credit partial recovery/note coverage.
        result["recovery_accuracy"] = result["note_verified_fraction"] = None
    finally:
        result.update(attempts=attempts, completed_first_ids=list(first), completed_note_ids=list(notes),
            input_tokens=sum(r["input_tokens"] for r in calls),
            output_tokens=sum(r["output_tokens"] for r in calls),
            generation_seconds=sum(r["seconds"] for r in calls),
            length_terminated_calls=sum(r["finish_reason"] == "length" for r in calls),
            first_request_conditions={pid: {k: r[k] for k in (
                "prompt_sha256", "seed", "image_sha256", "pixel_tensors")} for pid, r in first.items()})
        save(output / "diagnosis.json", result)
    return result


def evaluate(capture: Path, external_preparation: Path, output: Path, *, condition: str,
             reload_result: Path, minimal_capture: Path, minimal_reload: Path) -> dict:
    return _evaluate(capture, external_preparation, output, condition=condition,
        reload_result=reload_result, minimal_capture=minimal_capture, minimal_reload=minimal_reload)


def evaluate_ladder(capture: Path, cohort: Path, data_gate: Path, output: Path, *, condition: str,
                    arm: str, reload_result: Path, minimal_capture: Path | None = None, minimal_reload: Path | None = None,
                    with_recovery: bool = False, evaluation_prompt_policy: str | None = None) -> dict:
    """Native evaluation after explicit data/learning/reload gates.

    Invoke in a fresh process under the external watchdog recorded in conditions.
    Recovery uses fresh trusted verifiers; default first attempts are scored later.
    """
    return _evaluate(capture, data_gate, output, condition=condition, arm=arm, ladder_cohort=cohort,
        reload_result=reload_result, minimal_capture=minimal_capture, minimal_reload=minimal_reload,
        with_recovery=with_recovery, evaluation_prompt_policy=evaluation_prompt_policy)


def evaluate_pool(capture: Path, preparation: Path, output: Path, *, condition: str, arm: str, reload_result: Path,
                  minimal_capture: Path | None = None, minimal_reload: Path | None = None) -> dict:
    """Held-out pool items of the capture's own pool preparation; same capture/reload binding as evaluate-ladder."""
    return _evaluate(capture, preparation, output, condition=condition, arm=arm, reload_result=reload_result,
        minimal_capture=minimal_capture, minimal_reload=minimal_reload, with_recovery=True, pool=True)


def _evaluate(capture: Path, external_preparation: Path, output: Path, *, condition: str,
              reload_result: Path, minimal_capture: Path | None, minimal_reload: Path | None,
              ladder_cohort: Path | None = None, arm: str | None = None,
              with_recovery: bool = False, evaluation_prompt_policy: str | None = None, pool: bool = False) -> dict:
    if pool and (ladder_cohort is not None or with_recovery is not True or evaluation_prompt_policy is not None):
        raise ValueError("pool evaluation: the bound training pool preparation and recovery protocol only")
    if type(with_recovery) is not bool or with_recovery and ladder_cohort is None and not pool:
        raise ValueError("native ladder: recovery requires the explicit native cohort")
    if evaluation_prompt_policy is not None:
        from fine_tuning.notebook_rl.local_ladder_generation import (
            CIRCUIT_PROMPT_POLICY, CIRCUIT_REFERENCE_PROMPT_POLICY, CIRCUIT_REPAIR_PROMPT_POLICY)
        if (ladder_cohort is None
                or evaluation_prompt_policy not in (CIRCUIT_PROMPT_POLICY, CIRCUIT_REFERENCE_PROMPT_POLICY, CIRCUIT_REPAIR_PROMPT_POLICY)
                or evaluation_prompt_policy in (CIRCUIT_REFERENCE_PROMPT_POLICY, CIRCUIT_REPAIR_PROMPT_POLICY) and not with_recovery):
            raise ValueError("native ladder: explicit supported evaluation prompt policy required")
    evaluation_started = time.monotonic()
    import random
    import numpy as np
    import torch
    from fine_tuning.notebook_rl import checkpoint, scale as scale_options
    from fine_tuning.notebook_rl.cli import artifact_path
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.launch import read_descriptor
    from fine_tuning.notebook_rl.local_external import EVALUATION_PROTOCOL, load_external_actors
    from fine_tuning.notebook_rl.local_profile import PROFILE, REPAIR_PROTOCOL_ID, local_prompts, restore_protocol
    from fine_tuning.notebook_rl.local_run import (
        _install_actor_guards, cache_readiness, host_identity, make_probes, owner_alive, parameter_hashes,
        read_actor_descriptor, read_execution_actors, minimum_binding, same_process)
    from fine_tuning.notebook_rl.local_train import (
        ASSESSMENT_CREDIT_ADMISSION, FIXED_TEN_NATIVE_EXPLORATION, REPAIR_ADMISSIONS, execution_admission, execution_max_seconds, parent_gate,
        training_binding, training_parent_gate, validate_native_collection, validate_native_updates)
    from fine_tuning.notebook_rl.policy import enable_deterministic_execution
    from fine_tuning.notebook_rl.quantized_cache import block_original_access
    from fine_tuning.notebook_rl.quantized_policy import load_quantized_policy

    if condition not in ("F0", "FT"):
        raise ValueError("explicit F0 or FT condition required")
    native = ladder_cohort is not None or pool
    if native and (arm not in ("notebook", "answer-only") or condition == "F0" and arm != "notebook"):
        raise ValueError("native ladder: explicit requested arm; F0 must anchor notebook initialization")
    launch, captured = read(capture / "launch.json"), read(capture / "result.json")
    training_protocol = restore_protocol(launch["protocol"])
    scale = scale_options.options(training_protocol)
    bf16 = scale["policy_precision"] == "bf16"
    repair = training_protocol["protocol_id"] == REPAIR_PROTOCOL_ID
    repair_snapshots: dict[Path, tuple[Any, str]] = {}
    repair_proof = None
    from fine_tuning.notebook_rl.local_ladder_recovery import REPAIR_EVALUATOR_VERSION as repair_version
    if repair or evaluation_prompt_policy == "circuit-ladder-verified-repair-notebook-v1":
        from fine_tuning.notebook_rl.repair_replay import read_snapshot

        if (not repair or not native or not with_recovery or training_protocol["context_limit"] != 24576
                or (evaluation_prompt_policy is not None if pool
                    else evaluation_prompt_policy != "circuit-ladder-verified-repair-notebook-v1")):
            raise ValueError("native ladder: repair requires its exact native recovery evaluation policy")
        repair_snapshots = {path: read_snapshot(path) for path in (
            capture / "launch.json", capture / "result.json", capture / "reload_expected.json",
            capture / "initialization.json", capture / "checkpoint/metadata.json", Path(launch["preparation"]),
            reload_result, reload_result.parent / "forward_probes.json", reload_result.parent / "sync.json")}
        if (digest(launch) != digest(repair_snapshots[capture / "launch.json"][0])
                or digest(captured) != digest(repair_snapshots[capture / "result.json"][0])):
            raise ValueError("native ladder: cached repair capture versions changed")
    if (arm == "answer-only" and "notebook_update_requirement" in training_protocol
            and not (native and launch.get("execution_contract", {}).get("admission_policy") == ASSESSMENT_CREDIT_ADMISSION)):
        raise ValueError("native ladder: native-only objective control is unsupported without a matched experiment contract")
    context_limit = training_protocol["context_limit"] if native else CONTEXT_LIMIT
    collection = False
    assessment = False
    if native:
        contract = launch.get("execution_contract", {})
        if launch.get("execution_contract_sha256") != checkpoint.digest(contract):
            raise ValueError("native ladder: training admission contract hash differs")
        execution_max_seconds(contract, training_protocol)
        admission = execution_admission(contract, training_protocol)
        collection = admission == FIXED_TEN_NATIVE_EXPLORATION
        assessment = admission == ASSESSMENT_CREDIT_ADMISSION
        if repair and admission not in REPAIR_ADMISSIONS:
            raise ValueError("native ladder: explicit repair training admission required")
        if collection:
            if arm != "notebook" or minimal_capture is not None or minimal_reload is not None:
                raise ValueError("native ladder: native collection has no minimum parent; answer-only control is unsupported")
            gate = contract.get("collection_gate", {})
        else:
            if minimal_capture is None or minimal_reload is None:
                raise ValueError("native ladder: minimum capture and reload required for this admission")
            gate = training_parent_gate(minimal_capture, minimal_reload, admission=admission)
        if not collection and "admission_policy" in contract and contract.get("parent_gate") != gate:
            raise ValueError("native ladder: training admission parent evidence differs")
    else:
        if minimal_capture is None or minimal_reload is None:
            raise ValueError("external diagnosis: minimum capture and reload required")
        gate = parent_gate(minimal_capture, minimal_reload)
    if (launch["protocol"] != json.loads(json.dumps(training_protocol))
            or launch.get("standalone_base") is not (not bf16)
            or not collection and gate["launch"]["protocol"] != launch["protocol"]
            or captured.get("status") not in ("UPDATED_SAVED_PENDING_FRESH_RELOAD",
                "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD")
            or type(captured.get("optimizer_updates")) is not int
            or not 1 <= captured["optimizer_updates"] <= scale["max_updates"]):
        raise ValueError("closed standalone NF4 update required before external diagnosis")
    reloaded = repair_snapshots[reload_result][0] if repair else read(reload_result)
    if (reloaded.get("status") not in ("FRESH_PROCESS_RELOAD_VERIFIED", "LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED")
            or reloaded.get("capture_pid") != launch["pid"]
            or reloaded.get("step") != captured["optimizer_updates"]
            or reloaded.get("all_trainable_hashes_match") is not True
            or reloaded.get("actual_forward_and_generation_match") is not True
            or reloaded.get("frozen_quantized_base_unchanged") is not True):
        raise ValueError("fresh reload not bound to this capture")
    if native:
        contract = launch.get("execution_contract", {})
        if (captured.get("status") != "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"
                or captured.get("arm") != arm or contract.get("arm") != arm
                or contract.get("notebook_reward_enabled") is not (arm == "notebook")
                or launch.get("execution_contract_sha256") != checkpoint.digest(contract)
                or type(launch.get("pid")) is not int or launch["pid"] <= 0
                or owner_alive(launch)
                or reloaded.get("status") != "LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED"
                or type(reloaded.get("pid")) is not int or reloaded["pid"] <= 0
                or same_process(reloaded, launch) or owner_alive(reloaded)
                or reloaded.get("optimizer_restored") is not True
                or not bf16 and any(record.get("original_access_guard", {}).get("denied_original_access_attempts") != 0
                                    for record in (captured, reloaded))):
            raise ValueError("native ladder: closed requested ten-arm capture and independent reload required")
    prepared = repair_snapshots[Path(launch["preparation"])][0] if repair else read(Path(launch["preparation"]))
    request = read_descriptor(prepared["data_request"])
    minimum_contract = launch.get("execution_contract") if captured["status"] != "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD" else None
    if (captured["status"] != "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"
            and any(key in launch for key in ("execution_contract", "execution_contract_sha256"))
            and (minimum_contract is None or launch.get("execution_contract_sha256") != checkpoint.digest(minimum_contract))):
        raise ValueError("minimum execution contract changed before evaluation")
    expected_binding = (training_binding(prepared, training_protocol, launch["execution_contract"])
        if captured["status"] == "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD" else
        minimum_binding(prepared, training_protocol, minimum_contract))
    if captured["status"] != "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD" and minimum_contract is None:
        expected_binding = {**launch["binding"], "data_sha256": expected_binding["data_sha256"]}
    if not repair:
        expected_binding["code_sha256"] = launch["binding"]["code_sha256"]
    from fine_tuning.notebook_rl.local_data import POOL_FORMATS, POOL_STATUS
    if (prepared.get("status") != (POOL_STATUS if pool else "LOCAL_DATA_REVALIDATED")
            or pool != (prepared.get("format") in POOL_FORMATS) or prepared.get("profile") != PROFILE
            or expected_binding != launch["binding"]
            or launch["binding"]["protocol_sha256"] != training_protocol["sha256"]):
        raise ValueError("capture training data/protocol binding changed")
    model_path = Path(launch["model_path"])
    if bf16:
        from fine_tuning.notebook_rl.local_calibrate import verify_original

        readiness, base_metadata_sha = {}, checkpoint.digest(verify_original(model_path))
    else:
        readiness = cache_readiness(Path(launch["cache_check"]), model_path, training_protocol)
        base_metadata_sha = hashlib.sha256((model_path / "nf4_base.json").read_bytes()).hexdigest()
    metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
    if repair:
        from fine_tuning.notebook_rl.local_train import _repair_training_replay

        if digest(metadata) != digest(repair_snapshots[capture / "checkpoint/metadata.json"][0]):
            raise ValueError("native ladder: cached repair checkpoint metadata version changed")
        repair_proof = _repair_training_replay(capture, captured, contract, training_protocol,
                                              parameter_inventory=metadata["inventory"]["trainable"])
        if (any(repair_proof["paths"].get(name) != repair_snapshots[capture / name][1]
                for name in ("launch.json", "result.json", "reload_expected.json", "initialization.json"))
                or reloaded.get("execution_contract") != contract
                or reloaded.get("execution_contract_sha256") != checkpoint.digest(contract)
                or reloaded.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or reloaded.get("repair_protocol_sha256") != training_protocol["sha256"]
                or reloaded.get("repair_replay_sha256") != repair_proof.get("sha256")
                or reloaded.get("repair_update_evidence_sha256") != repair_proof["paths"].get("repair_update_evidence.json")
                or reloaded.get("fresh_reload_verified") is not True
                or reloaded.get("repair_functional_verified") is not True
                or type(reloaded.get("step")) is not int or type(reloaded.get("capture_pid")) is not int
                or type(repair_proof.get("S_gradient_verified")) is not bool
                and not (scale["evidence_audit_rule"] is not None and repair_proof.get("S_gradient_verified") is None)
                or reloaded.get("S_gradient_verified") is not repair_proof["S_gradient_verified"]
                or reloaded.get("repair_learning_verified") is not repair_proof["S_gradient_verified"]
                or reloaded.get("benefit_verified") is not False):
            raise ValueError("native ladder: current repair source replay and actual fresh reload binding required")
    if assessment:
        validate_native_updates(capture, captured, contract, training_protocol,
            parameter_inventory=metadata["inventory"]["trainable"])
        if (reloaded.get("execution_contract") != contract
                or reloaded.get("execution_contract_sha256") != checkpoint.digest(contract)
                or reloaded.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])
                or any(reloaded.get(key) != captured.get(key) for key in ("assessment_learning_verified",
                    "assessment_only_gradient_verified", "strict_notebook_correctness_verified",
                    "accurate_whole_notebook_verified", "visual_improvement_verified"))):
            raise ValueError("native ladder: assessment reload execution or learning evidence differs")
    if collection:
        from fine_tuning.notebook_rl.local_run import read_launch_preparation

        fresh_prepared, _, fresh_actors = read_launch_preparation(launch, training_protocol)
        if fresh_prepared != prepared:
            raise ValueError("native ladder: native collection preparation changed")
        gate = validate_native_collection(contract, prepared, training_protocol, readiness, fresh_actors)
        validate_native_updates(capture, captured, contract, training_protocol)
        if (gate["code_sha256"] != launch["binding"]["code_sha256"]
                or reloaded.get("execution_contract") != contract
                or reloaded.get("execution_contract_sha256") != checkpoint.digest(contract)
                or reloaded.get("capture_binding_sha256") != checkpoint.digest(launch["binding"])):
            raise ValueError("native ladder: native collection reload execution binding differs")
    if pool:
        from fine_tuning.notebook_rl import local_pool_eval
        from fine_tuning.notebook_rl.local_ladder import _descriptor
        from fine_tuning.notebook_rl.repair_notebook import note_contract as bound_note_contract
        if Path(external_preparation).absolute() != Path(launch["preparation"]).absolute():
            raise ValueError("pool evaluation: the capture's own pool preparation is required")
        cohort = Path(prepared["build_root"])
        evaluation_prompts = local_prompts(training_protocol)
        pool_note_contract = bound_note_contract(training_protocol)
        evaluation_protocol = local_pool_eval.protocol(evaluation_prompts, training_protocol["sha256"],
                                                       context_limit=context_limit, note_contract=pool_note_contract)
        actors = read_actor_descriptor(prepared["evaluation_actors"])
        data_gate = {"status": "LOCAL_POOL_EVALUATION_INPUTS_BOUND",
                     "preparation": _descriptor(Path(launch["preparation"]).absolute()),
                     "problem_ids": [a.problem_id for a in actors],
                     "protocol_sha256": digest(evaluation_protocol)}
    elif ladder_cohort is not None:
        from fine_tuning.notebook_rl.local_ladder import _descriptor, load_ladder_actors
        from fine_tuning.notebook_rl.local_ladder_gate import validate_gate
        from fine_tuning.notebook_rl import local_ladder_generation as ladder
        from fine_tuning.notebook_rl import local_ladder_recovery
        cohort = Path(ladder_cohort)
        native_core = local_ladder_recovery if with_recovery else ladder
        evaluation_prompts = local_prompts(training_protocol)
        if evaluation_prompt_policy is not None:
            evaluation_prompts = ladder.circuit_evaluation_prompts(evaluation_prompt_policy)
        from fine_tuning.notebook_rl.compact_note import VERSION as compact_note_version
        from fine_tuning.notebook_rl.repair_notebook import note_contract
        # A compact-trained capture is evaluated with the final-note contract it was trained with.
        note_options = ({"note_contract": compact_note_version}
                        if repair and native_core is local_ladder_recovery
                        and note_contract(training_protocol) == compact_note_version else {})
        evaluation_protocol = native_core.protocol(evaluation_prompts, training_protocol["sha256"],
                                                   context_limit=context_limit, **note_options)
        if repair and evaluation_protocol.get("evaluator_version") != repair_version:
            raise ValueError("native ladder: exact repair evaluator version required")
        data_gate = validate_gate(cohort, Path(launch["preparation"]), external_preparation, evaluation_protocol)
        actors = load_ladder_actors(cohort)
        if (data_gate.get("status") != "LOCAL_LADDER_DIAGNOSTIC_DATA_VERIFIED"
                or data_gate.get("preparation") != _descriptor(cohort / "preparation.json")
                or data_gate.get("protocol_sha256") != ladder._digest(evaluation_protocol)
                or len(actors) != 10 or data_gate.get("problem_ids") != [a.problem_id for a in actors]
                or any(a.mode != "obs" or a.answer_format != "unit_number" or a.choices for a in actors)):
            raise ValueError("native ladder: current trusted diagnostic data gate required")
    else:
        external = read(external_preparation)
        actors = load_external_actors(Path(external["cohort"]))
        if (external.get("status") != "CPU_EXTERNAL_INPUTS_PREPARED_NOT_EVALUATED"
                or external.get("n_problems") != 10 or external.get("training_allowed") is not False
                or read_descriptor(external["evaluation_protocol"]) != EVALUATION_PROTOCOL
                or read_actor_descriptor(external["actors"]) != actors
                or external["problem_ids"] != [a.problem_id for a in actors]):
            raise ValueError("fixed evaluation preparation changed")
        cohort = Path(external["cohort"])
        evaluation_protocol = protocol()
    train_actors = read_actor_descriptor(prepared["actors"])
    probe_actors = read_execution_actors(prepared, train_actors, training_protocol)
    if native and "problem_range" in launch.get("execution_contract", {}):
        from fine_tuning.notebook_rl.local_train import contract_actors

        # A bound chunk of a longer selection trains (and probes) only its own actors.
        probe_actors = contract_actors(probe_actors, launch["execution_contract"])
    from fine_tuning.notebook_rl.local_run import pool_v2, probe_counts, probe_source, sync_options

    # v2 pools probe their bound coverage set (single-image, or single and two-image) and train N (the scale group).
    probe_actors = probe_source(prepared, probe_actors)
    count = scale["train_problems"] if pool_v2(prepared) else 10
    from fine_tuning.notebook_rl.local_data import POOL_FORMAT_PROOF

    # Proof sentinels share sources with training by design; their structure-family disjointness is rule-checked.
    shared_sources = prepared.get("format") == POOL_FORMAT_PROOF
    if (len(train_actors) != count or len({a.problem_id for a in train_actors}) != count
            or not shared_sources and {a.source for a in train_actors} & {a.source for a in actors}
            or {h for a in train_actors for h in a.image_sha256} & {h for a in actors for h in a.image_sha256}):
        raise ValueError("actual training and evaluation sources/images overlap")
    if native:
        from fine_tuning.notebook_rl.local_train import contract_actors, fixed_pairs, matched_notebook_arm
        trained = contract_actors(train_actors, contract)
        pairs = scale_options.batches(trained, training_protocol) if scale_options.scaled(training_protocol) else fixed_pairs(trained)
        if (launch.get("problem_ids") != [a.problem_id for a in trained]
                or captured.get("attempted_ids") != launch["problem_ids"]
                or contract.get("pair_ids") != [[a.problem_id for a in pair] for pair in pairs]):
            raise ValueError("native ladder: actual fixed ten training coverage/order changed")
        if arm == "answer-only":
            matched = contract.get("matched_notebook_arm", {})
            admission_options: dict[str, Any] = (
                {"admission": execution_admission(contract, training_protocol), "parent": gate}
                if "admission_policy" in contract else {})
            if repair:
                if (type(matched) is not dict or type(matched.get("fresh_reload")) is not dict
                        or type(matched["fresh_reload"].get("path")) is not str):
                    raise ValueError("native ladder: explicit matched repair fresh reload path required")
                admission_options["matched_reload"] = Path(matched["fresh_reload"]["path"])
            if (type(matched) is not dict or type(matched.get("initialization")) is not dict
                    or matched_notebook_arm(Path(matched["initialization"]["path"]), prepared,
                        training_protocol, model_path, launch["cache_check"] and Path(launch["cache_check"]), pairs,
                        historical_code_sha256=None if repair else launch["binding"]["code_sha256"],
                        max_seconds=execution_max_seconds(contract, training_protocol), **admission_options) != matched):
                raise ValueError("native ladder: answer-only arm must bind the matched notebook run")
    if repair:
        from fine_tuning.notebook_rl.local_train import _repair_probe_binding

        assert repair_proof is not None
        expected = repair_snapshots[capture / "reload_expected.json"][0]
        _repair_probe_binding(expected.get("forward_probes"), repair_snapshots[reload_result.parent / "sync.json"][0],
                              captured["optimizer_updates"], probe_actors, batch_rows=scale["rollout_batch_rows"],
                              image_counts=probe_counts(prepared))
        if (repair_snapshots[reload_result.parent / "forward_probes.json"][0] != expected.get("forward_probes")
                or type(metadata.get("step")) is not int or metadata["step"] != captured["optimizer_updates"]
                or captured.get("initialization_sha256") != repair_snapshots[capture / "initialization.json"][1]
                or any(checkpoint.file_hash(capture / name) != sha for name, sha in repair_proof["paths"].items())
                or any(checkpoint.file_hash(Path(row["path"])) != row["sha256"] for row in gate["files"])
                or any(checkpoint.file_hash(path) != sha for path, (_, sha) in repair_snapshots.items())):
            raise ValueError("native ladder: actual repair reload/probe/source snapshot changed before evaluation")
    output = artifact_path(output)
    protected: tuple[Path, ...] = (capture, model_path, cohort)
    if native:
        protected += tuple(p for p in (external_preparation.parent, minimal_capture, minimal_reload, reload_result.parent,
                      Path(launch["preparation"]).parent, Path(prepared["build_root"])) if p is not None)
    if any(output.resolve().is_relative_to(p.resolve()) for p in protected if p is not None):
        raise ValueError("external evaluation output must preserve source artifacts")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    sources = source_hashes()
    expected = repair_snapshots[capture / "reload_expected.json"][0] if repair else read(capture / "reload_expected.json")
    if metadata["protocol"] != launch["protocol"] or metadata["step"] != captured["optimizer_updates"]:
        raise ValueError("checkpoint and capture protocol/step differ")
    if native and (expected.get("step") != metadata["step"]
            or metadata.get("state_sha256") != {name: sha for group in expected["trainable_sha256"].values()
                                                for name, sha in group.items()}
            or metadata.get("base_sha256") != expected.get("frozen_base_sha256")
            or metadata.get("inventory", {}).get("quantization") != expected.get("quantization")):
        raise ValueError("native ladder: checkpoint differs from the verified final tensors/base")
    training_initial = (read(capture / "initialization.json")["trainable_sha256"]
        if (capture / "initialization.json").exists() else read(capture / "parameter_changes.json")["before"])
    if (captured["status"] == "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD"
            and captured.get("initialization_sha256") != checkpoint.file_hash(capture / "initialization.json")):
        raise ValueError("closed training initialization record changed")
    if native and arm == "answer-only" and read(capture / "initialization.json") != read(
            Path(matched["initialization"]["path"])):
        raise ValueError("native ladder: answer-only initialization differs from matched notebook F0")
    if native:
        runtime_bindings = {str(path): checkpoint.file_hash(path) for path in (
            capture / "launch.json", capture / "result.json", capture / "reload_expected.json",
            capture / "initialization.json", reload_result, Path(launch["preparation"]))}
        if repair:
            assert repair_proof is not None
            runtime_bindings.update({str(path): sha for path, (_, sha) in repair_snapshots.items()})
            runtime_bindings.update({str(capture / name): sha for name, sha in repair_proof["paths"].items()})
            runtime_bindings.update({row["path"]: row["sha256"] for row in gate["files"]})
            if arm == "answer-only":
                runtime_bindings.update(matched["repair_runtime_files"])
                runtime_bindings.update({str(Path(matched["fresh_reload"]["path"]) / name): sha
                                        for name, sha in matched["fresh_reload"]["files"].items()})
    runtime = enable_deterministic_execution()
    conditions = {"protocol": evaluation_protocol, "training_protocol_sha256": training_protocol["sha256"],
        "source_sha256": sources, "external_preparation_sha256": hashlib.sha256(external_preparation.read_bytes()).hexdigest(),
        "base_metadata_sha256": base_metadata_sha,
        "device": str(torch.cuda.get_device_properties(0)), "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "sdpa": {"math": torch.backends.cuda.math_sdp_enabled(),
                 "flash": torch.backends.cuda.flash_sdp_enabled(),
                 "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled()},
        "concurrency": 1, "seed": 3407, "output_tokens": 1024,
        "notebook_output_tokens": NOTE_TOKENS, "context_limit": context_limit}
    conditions["deterministic_execution"] = runtime
    if native:
        if not with_recovery:
            conditions.pop("notebook_output_tokens")
        else:
            conditions["notebook_output_tokens"] = evaluation_protocol["notebook"]["max_new_tokens"]
        conditions["ladder_data_gate_sha256"] = conditions.pop("external_preparation_sha256")
        conditions.update(ladder_preparation=data_gate["preparation"], max_attempts=3 if with_recovery else 1, n=1,
            external_watchdog_required=True,
            max_seconds_per_run=training_protocol["local_budget"]["max_seconds_per_run"])
    save(output / "launch.json", {"status": "STARTED_NOT_COMPLETED", "condition": condition,
        "pid": os.getpid(), "host": host_identity(), "capture": str(capture), "conditions": conditions,
        **({"native_collection_gate": gate} if collection else {"minimal_parent_gate": gate}),
        "capture_binding": launch["binding"],
        "reload_result_sha256": hashlib.sha256(reload_result.read_bytes()).hexdigest(),
        "checkpoint_metadata_sha256": hashlib.sha256((capture / "checkpoint/metadata.json").read_bytes()).hexdigest(),
        "current_source_binding_is_separate_from_training_binding": True,
        "fresh_trusted_worker": not native or with_recovery, "python_guards_are_not_os_sandbox": True,
        # The capture's frozen-approval-verify-v1 record (absent: its preparation was read with a fresh approval).
        **({"data_verification": launch["data_verification"]} if "data_verification" in launch else {}),
        **({"requested_arm": arm, "ladder_data_gate": data_gate, "runtime_input_files": runtime_bindings}
           if native else {})})
    # A pool's held-out vault files are already in the build-wide guard list.
    vault_files = ([] if pool else
                   [cohort / "records/verifier_private.jsonl", cohort / "records/lineage.jsonl"] if native else
                   [cohort / "gold.jsonl", cohort / "canonical.jsonl"])
    schema = {} if scale["rollout_engine"] == "vllm" else None
    _install_actor_guards(prepared, request, extra_vault_files=vault_files, vllm_schema=schema)
    original_guard = None if bf16 else block_original_access(Path(readiness["original_model_root"]))
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    result = {"status": "INCOMPLETE", "condition": condition, "improvement_claimed": False,
              "execution_conditions_verified": False, "training_updates": 0, **EVIDENCE_LIMITS, **(schema or {})}
    if native:
        result.update(evaluation_protocol=(local_pool_eval.VERSION if pool else repair_version if repair
                                           else native_core.VERSION), arm=arm, accuracy=None,
                      cpu_scoring_required_after_actor_exit=not with_recovery, training_allowed=False,
                      **({"cpu_replay_required_after_actor_exit": True} if with_recovery else {}),
                      ladder_data_gate=data_gate)
        if repair:
            assert repair_proof is not None
            result.update(repair_metrics=None, repair_training_replay_sha256=repair_proof["sha256"],
                          repair_source_S_gradient_verified=repair_proof["S_gradient_verified"])
    meter, startup_seconds = None, 0.
    try:
        loader = scale_options.build_bf16_policy if bf16 else load_quantized_policy
        bundle = (loader(model_path, "MOD-RV", training_protocol) if condition == "F0" else
            checkpoint.load_eval(model_path, capture / "checkpoint", binding=launch["binding"],
                                 policy_factory=partial(loader, device_index=0)))
        guard = scale_options.base_guard(bundle, training_protocol)
        if condition == "FT" and parameter_hashes(bundle) != expected["trainable_sha256"]:
            raise ValueError("loaded FT tensors differ from verified capture")
        initial_parameters = parameter_hashes(bundle)
        if native and condition == "F0" and initial_parameters != training_initial:
            raise ValueError("native ladder: fresh F0 differs from actual notebook training initialization")
        save(output / "initial_parameters.json", initial_parameters)
        # Bound sampler (HF serial/row-batched or vLLM with merged weights) plus the live HF policy engine.
        engine, policy_engine = scale_options.make_engines(bundle, training_protocol, model_path,
            policy_version=0 if condition == "F0" else metadata["step"], context_limit=context_limit)
        meter = scale_options.TokenMeter(engine)
        with ExitStack() as stack:
            probe_images = {a.problem_id: load_images(a, Path(prepared["image_root"])) for a in probe_actors}
            for values in probe_images.values():
                for image in values:
                    stack.callback(image.close)
            probes = make_probes(probe_actors, probe_images, engine, local_prompts(training_protocol),
                                 image_counts=probe_counts(prepared))
            synchronized, sampler_synchronized = scale_options.synchronize(bundle, engine, policy_engine, probes,
                training_protocol, engine.policy_version, push=False, **sync_options(prepared))
            save(output / "synchronization-forward-only.json", synchronized)
            if sampler_synchronized is not None:
                save(output / "sampler-synchronization-forward-only.json", sampler_synchronized)
            startup_seconds = time.monotonic() - evaluation_started
            if pool:
                deadline = evaluation_started + training_protocol["local_budget"]["max_seconds_per_run"]
                recovered = local_pool_eval.diagnose(Path(launch["preparation"]), output / "recovery", engine,
                    evaluation_prompts, expected_preparation_sha256=data_gate["preparation"]["file_sha256"],
                    local_protocol_sha256=training_protocol["sha256"], deadline=deadline,
                    context_limit=context_limit, note_contract=pool_note_contract,
                    memory_probe=torch.cuda.max_memory_allocated)
                result["recovery"] = recovered
                if (recovered.get("status") != local_pool_eval.COMPLETE
                        or recovered.get("problem_ids") != data_gate["problem_ids"]
                        or recovered.get("protocol") != evaluation_protocol
                        or recovered.get("preparation") != data_gate["preparation"]
                        or recovered.get("policy_version") != engine.policy_version
                        or recovered.get("source_sha256") != sources["approval"]):
                    raise ValueError("pool evaluation: complete bound pool recovery record required")
                save(output / "responses.json", [_response(row) for row in recovered["first_rows"]])
                result.update(status=local_pool_eval.COMPLETE, metrics=recovered["metrics"], items=recovered["items"])
            elif native:
                deadline = evaluation_started + training_protocol["local_budget"]["max_seconds_per_run"]
                if with_recovery:
                    recovered = local_ladder_recovery.diagnose(cohort, output / "recovery", engine,
                        evaluation_prompts,
                        expected_preparation_sha256=data_gate["preparation"]["file_sha256"],
                        local_protocol_sha256=training_protocol["sha256"], deadline=deadline, context_limit=context_limit,
                        **note_options)
                    result["recovery"] = recovered
                    if (recovered.get("status") != "COMPLETE_NATIVE_LADDER_RECOVERY"
                            or recovered.get("problem_ids") != [a.problem_id for a in actors]
                            or recovered.get("protocol") != evaluation_protocol
                            or recovered.get("protocol_sha256") != digest(evaluation_protocol)
                            or recovered.get("preparation") != data_gate["preparation"]
                            or recovered.get("policy_version") != engine.policy_version
                            or recovered.get("source_sha256") != sources["approval"]):
                        raise ValueError("native ladder: complete bound recovery record required")
                    generated = recovered["generation"]
                else:
                    generated = ladder.generate_ladder(cohort, output / "generation", engine,
                        evaluation_prompts,
                        expected_preparation_sha256=data_gate["preparation"]["file_sha256"],
                        local_protocol_sha256=training_protocol["sha256"], deadline=deadline, context_limit=context_limit)
                if (generated.get("status") != "TEN_FIRST_ATTEMPTS_RECORDED"
                        or generated.get("problem_ids") != [a.problem_id for a in actors]
                        or type(generated.get("rows")) is not list or len(generated["rows"]) != 10
                        or [row.get("problem_id") for row in generated["rows"]] != generated["problem_ids"]
                        or any(type(row.get("context_limit")) is not int or row["context_limit"] != context_limit
                               for row in generated["rows"])
                        or generated.get("protocol") != ladder.protocol(
                            evaluation_prompts, training_protocol["sha256"], context_limit=context_limit)
                        or with_recovery and generated.get("source_sha256") != recovered["source_sha256"]
                        or generated.get("policy_version") != engine.policy_version):
                    raise ValueError("native ladder: complete original first-attempt generation required")
                save(output / "responses.json", [_response(row) for row in generated["rows"]])
                if with_recovery:
                    result.update(status="COMPLETE_NATIVE_LADDER_RECOVERY", generation=generated,
                                  **{key: recovered[key] for key in NATIVE_RECOVERY_SUMMARY})
                    if repair:
                        if type(recovered.get("repair_metrics")) is not dict or any(recovered.get(key) is not None
                                for key in ("note_count", "note_verified_count", "note_verified_fraction")):
                            raise ValueError("native ladder: distinct repair metrics and null historical note metrics required")
                        result["repair_metrics"] = recovered["repair_metrics"]
                else:
                    result.update(status="COMPLETE_NATIVE_LADDER_GENERATION", generation=generated,
                                  usage=generated["usage"], generation_seconds=generated["elapsed_seconds"])
            else:
                result.update(diagnose(actors, lambda a: load_images(a, cohort), engine, cohort, output,
                    deadline=began + training_protocol["local_budget"]["max_seconds_per_run"]))
        guard(bundle)
        if parameter_hashes(bundle) != initial_parameters or source_hashes() != sources:
            raise ValueError("parameters or current evaluation sources changed during diagnosis")
        if checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"]) != metadata:
            raise ValueError("bound checkpoint changed during diagnosis")
        if original_guard is not None and original_guard["denied_original_access_attempts"]:
            raise ValueError("original BF16 access attempted during standalone evaluation")
        if native and not pool and validate_gate(cohort, Path(launch["preparation"]), external_preparation,
                                                 evaluation_protocol) != data_gate:
            raise ValueError("native ladder: trusted data gate changed during inference")
        if pool and _descriptor(Path(launch["preparation"]).absolute()) != data_gate["preparation"]:
            raise ValueError("pool evaluation: bound pool preparation changed during inference")
        if native and any(checkpoint.file_hash(Path(path)) != sha for path, sha in runtime_bindings.items()):
            raise ValueError("native ladder: capture/reload/preparation evidence changed during inference")
        if native and time.monotonic() > deadline:
            raise TimeoutError("native ladder: evaluation deadline exceeded")
        result["unchanged_parameters_verified"] = result["unchanged_current_code_verified"] = True
        result["initial_trainable_sha256"] = initial_parameters
        result["training_initial_trainable_sha256"] = training_initial
        result["checkpoint_training_updates"] = 0 if condition == "F0" else metadata["step"]
        result["actual_source_and_image_separation_verified"] = True
        result["conditions"] = conditions
        if native:
            result.update(runtime_input_files=runtime_bindings, unchanged_quantized_base_verified=True)
    except BaseException as exc:
        result.update(status="INCOMPLETE", error_type=type(exc).__name__, error=str(exc))
        result["recovery_accuracy"] = result["note_verified_fraction"] = None
        if repair:
            result["repair_metrics"] = None
        if with_recovery:
            result["first_attempt_accuracy"] = result["recovered_fraction"] = None
            result["recovery_rate_diagnostic" if pool else "final_accuracy"] = None
        raise
    finally:
        result.update(seconds=time.monotonic()-began, original_access_guard=original_guard,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if native:
            result["total_seconds"] = time.monotonic() - evaluation_started
        try:
            calls = meter.calls if meter else 0
            scale_options.write_receipt(output, items=len(actors), completion_tokens=meter.tokens if meter else 0,
                wall_seconds=time.monotonic() - evaluation_started, updates=0, update_seconds=0., n_samples=1,
                retries_per_item=max(0., calls / len(actors) - 1) if actors else 0.,
                startup_seconds=startup_seconds,
                extra={"run": "local_eval", "status": result["status"], "condition": condition,
                       "sampler": scale_options.sampler_label(training_protocol),
                       "completion_scope": "every_sampler_call", "sampler_calls": calls,
                       "retries_basis": "sampler_calls_per_item_minus_one_including_note_calls"})
        except Exception as exc:  # measurement never masks the evaluation's own outcome
            result["throughput_receipt_error"] = f"{type(exc).__name__}: {exc}"
        save(output / "result.json", result)
    return result


def _compare_native_recovery(before: dict, after: dict) -> dict:
    from fine_tuning.notebook_rl.local_ladder_generation import CIRCUIT_REFERENCE_PROMPT_POLICY, CIRCUIT_REPAIR_PROMPT_POLICY
    from fine_tuning.notebook_rl import local_ladder_recovery as recovery
    if (before.get("status") != "COMPLETE_NATIVE_LADDER_RECOVERY"
            or after.get("status") != "COMPLETE_NATIVE_LADDER_RECOVERY"
            or before.get("condition") != "F0" or after.get("condition") != "FT"
            or not before.get("conditions") or before["conditions"] != after.get("conditions")
            or not before.get("initial_trainable_sha256")
            or not after.get("initial_trainable_sha256")
            or before["initial_trainable_sha256"] != before.get("training_initial_trainable_sha256")
            or before["initial_trainable_sha256"] != after.get("training_initial_trainable_sha256")
            or before.get("arm") != "notebook" or after.get("arm") not in ("notebook", "answer-only")):
        raise ValueError("native recovery: complete paired F0/FT and matching initialization/conditions required")
    context_limit = before["conditions"].get("protocol", {}).get("context_limit")
    repair = before["conditions"].get("protocol", {}).get("evaluation_prompt_policy") == CIRCUIT_REPAIR_PROMPT_POLICY
    expected_version = recovery.REPAIR_EVALUATOR_VERSION if repair else recovery.VERSION
    if repair and (context_limit != 24576 or before["conditions"]["protocol"].get("evaluator_version") != expected_version):
        raise ValueError("native recovery: exact repair evaluator policy/version required")
    if type(context_limit) is not int or context_limit not in (16384, 24576):
        raise ValueError("native recovery: canonical recorded execution context required")
    required_conditions = {"max_attempts": 3, "n": 1, "output_tokens": 1024,
        "notebook_output_tokens": 8192 if before["conditions"]["protocol"].get("evaluation_prompt_policy")
            in (CIRCUIT_REFERENCE_PROMPT_POLICY, CIRCUIT_REPAIR_PROMPT_POLICY) else NOTE_TOKENS, "context_limit": context_limit,
        "concurrency": 1, "seed": 3407, "external_watchdog_required": True, "max_seconds_per_run": 14400}
    if any(type(before["conditions"].get(key)) is not type(value)
           or before["conditions"][key] != value for key, value in required_conditions.items()):
        raise ValueError("native recovery: canonical recorded execution conditions required")
    for result in (before, after):
        record = result.get("recovery", {})
        if (result.get("evaluation_protocol") != expected_version
                or type(result.get("training_updates")) is not int or result["training_updates"] != 0
                or type(result.get("checkpoint_training_updates")) is not int
                or not result.get("runtime_input_files")
                or result.get("original_access_guard", {}).get("denied_original_access_attempts") != 0
                or any(result.get(key) is not True for key in ("unchanged_parameters_verified",
                    "unchanged_current_code_verified", "actual_source_and_image_separation_verified",
                    "unchanged_quantized_base_verified"))
                or record.get("protocol") != result["conditions"].get("protocol")
                or record.get("preparation") != result["conditions"].get("ladder_preparation")
                or record.get("source_sha256") != result["conditions"].get("source_sha256", {}).get("approval")
                or record.get("policy_version") != result.get("checkpoint_training_updates")
                or record.get("generation") != result.get("generation")
                or any(key not in record or key not in result or result[key] != record[key]
                       for key in NATIVE_RECOVERY_SUMMARY)):
            raise ValueError("native recovery: bound runtime, source, metrics and unchanged parameters required")
        if repair and (type(result.get("repair_metrics")) is not dict or result["repair_metrics"] != record.get("repair_metrics")
                or not result.get("repair_training_replay_sha256")
                or any(result.get(key) is not None for key in ("note_count", "note_verified_count", "note_verified_fraction"))):
            raise ValueError("native recovery: original distinct repair metrics required")
    paired = recovery.validate_paired_records(before["recovery"], after["recovery"])
    return {**paired, "status": "COMPLETE_PAIRED_NATIVE_LADDER_RECOVERY", "n_paired": 10,
        "before_sha256": digest(before), "after_sha256": digest(after),
        "execution_conditions_verified": True, "general_performance_improvement_verified": False,
        "module_causality_certified": False, "lesson_semantics_verified": False,
        "model_generation_provenance_verified": False, "visual_cause_verified": None}


def compare(before: dict, after: dict) -> dict:
    if "recovery" in before or "recovery" in after:
        return _compare_native_recovery(before, after)
    if (before.get("status") != "COMPLETE" or after.get("status") != "COMPLETE"
            or before.get("condition") != "F0" or after.get("condition") != "FT"
            or not before.get("conditions") or before["conditions"] != after.get("conditions")
            or len(before.get("first_statuses", {})) != 10
            or set(before["first_statuses"]) != set(after.get("first_statuses", {}))
            or set(before.get("first_request_conditions", {})) != set(before["first_statuses"])
            or before.get("first_request_conditions") != after.get("first_request_conditions")
            or not before.get("initial_trainable_sha256")
            or before["initial_trainable_sha256"] != after.get("training_initial_trainable_sha256")
            or any(r.get(k) is not True for r in (before, after) for k in (
                "unchanged_parameters_verified", "unchanged_current_code_verified",
                "actual_source_and_image_separation_verified"))):
        raise ValueError("complete F0/FT and actual identical execution/input conditions required")
    old, new = before["first_statuses"], after["first_statuses"]
    if any(v not in ("CORRECT", "INCORRECT") for v in (*old.values(), *new.values())):
        raise ValueError("unverified first-attempt status")
    return {"status": "COMPLETE_PAIRED_DIAGNOSTIC", "n_paired": 10,
        "before_sha256": digest(before), "after_sha256": digest(after),
        "first_attempt_net_correct": sum(v == "CORRECT" for v in new.values()) - sum(v == "CORRECT" for v in old.values()),
        "wrong_to_correct": sum(old[k] == "INCORRECT" and new[k] == "CORRECT" for k in old),
        "correct_to_wrong": sum(old[k] == "CORRECT" and new[k] == "INCORRECT" for k in old),
        "before_recovery_accuracy": before["recovery_accuracy"], "after_recovery_accuracy": after["recovery_accuracy"],
        "execution_conditions_verified": True, "general_performance_improvement_verified": False,
        "visual_cause_verified": None}


def _pool_cpu(args) -> int:
    """CPU-only regrading of saved pool records; never loads a model."""
    from fine_tuning.notebook_rl import local_pool_eval
    from fine_tuning.notebook_rl.cli import artifact_path
    from fine_tuning.notebook_rl.local_ladder import _path

    output = artifact_path(_path(Path(args.output).absolute()))
    if output.exists():
        raise ValueError("pool replay: new output outside immutable inputs required")
    if args.stage == "replay-pool":
        record_path = _path(Path(args.record).absolute())
        raw = record_path.read_bytes()
        result = local_pool_eval.replay(decode(raw), expected_preparation_sha256=args.preparation_sha256)
        if record_path.read_bytes() != raw:
            raise ValueError("pool replay: saved record changed during replay")
        result["record_file"] = {"path": str(record_path), "file_sha256": hashlib.sha256(raw).hexdigest()}
    else:
        labelled = [value.split("=", 1) for value in args.record]
        if any(len(pair) != 2 or not pair[0] for pair in labelled) or len({pair[0] for pair in labelled}) != len(labelled):
            raise ValueError("pool comparison: unique LABEL=PATH records required")
        result = local_pool_eval.compare({label: decode(_path(Path(path).absolute()).read_bytes())
                                          for label, path in labelled},
                                         expected_preparation_sha256=args.preparation_sha256)
        result["status"] = "POOL_COMPARISON_REPLAYED"
    save(output, result)
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    return 0


def main(argv=None) -> int:
    if (sys.argv[1:] if argv is None else argv) == ["_worker"]:
        try:
            raw = sys.stdin.buffer.read(MAX_WIRE + 1)
            if len(raw) > MAX_WIRE:
                return 2
            value = wire(worker(decode(raw)))
            if len(value) > MAX_WIRE:
                return 2
            sys.stdout.buffer.write(value)
            return 0
        except Exception:
            sys.stdout.buffer.write(wire({"status": "SCORER_ERROR"}))
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    run = sub.add_parser("evaluate")
    for name in ("capture", "external-preparation", "reload-result", "minimal-capture", "minimal-reload", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--condition", choices=("F0", "FT"), required=True)
    native = sub.add_parser("evaluate-ladder")
    for name in ("capture", "cohort", "data-gate", "reload-result", "output"):
        native.add_argument("--" + name, type=Path, required=True)
    for name in ("minimal-capture", "minimal-reload"):
        native.add_argument("--" + name, type=Path)
    native.add_argument("--condition", choices=("F0", "FT"), required=True)
    native.add_argument("--arm", choices=("notebook", "answer-only"), required=True)
    native.add_argument("--with-recovery", action="store_true",
                        help="Use the separately gated native three-attempt/notebook protocol")
    native.add_argument("--evaluation-prompt-policy",
                        choices=("generic-circuit-observations-v1", "circuit-ladder-reference-notebook-v1",
                                 "circuit-ladder-verified-repair-notebook-v1"),
                        help="Bind circuit evaluation prompts separately from graph training prompts")
    replay = sub.add_parser("replay-ladder")
    for name in ("cohort", "record", "output"):
        replay.add_argument("--" + name, type=Path, required=True)
    replay.add_argument("--cohort-preparation-sha256", required=True)
    paired = sub.add_parser("compare")
    for name in ("before", "after", "output"):
        paired.add_argument("--" + name, type=Path, required=True)
    pool = sub.add_parser("evaluate-pool")
    for name in ("capture", "preparation", "reload-result", "output"):
        pool.add_argument("--" + name, type=Path, required=True)
    for name in ("minimal-capture", "minimal-reload"):
        pool.add_argument("--" + name, type=Path)
    pool.add_argument("--condition", choices=("F0", "FT"), required=True)
    pool.add_argument("--arm", choices=("notebook", "answer-only"), required=True)
    pool_replay = sub.add_parser("replay-pool")
    for name in ("record", "output"):
        pool_replay.add_argument("--" + name, type=Path, required=True)
    pool_replay.add_argument("--preparation-sha256", required=True)
    pool_compare = sub.add_parser("compare-pool")
    pool_compare.add_argument("--record", action="append", required=True, metavar="LABEL=PATH",
                              help="repeat in table order, e.g. F0=... FT-notebook=... FT-control=...")
    pool_compare.add_argument("--preparation-sha256", required=True)
    pool_compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage in ("replay-pool", "compare-pool"):
        return _pool_cpu(args)
    if args.stage == "evaluate":
        result = evaluate(args.capture, args.external_preparation, args.output,
            condition=args.condition, reload_result=args.reload_result,
            minimal_capture=args.minimal_capture, minimal_reload=args.minimal_reload)
    elif args.stage == "evaluate-ladder":
        result = evaluate_ladder(args.capture, args.cohort, args.data_gate, args.output,
            condition=args.condition, arm=args.arm, reload_result=args.reload_result,
            minimal_capture=args.minimal_capture, minimal_reload=args.minimal_reload,
            with_recovery=args.with_recovery, evaluation_prompt_policy=args.evaluation_prompt_policy)
    elif args.stage == "evaluate-pool":
        result = evaluate_pool(args.capture, args.preparation, args.output, condition=args.condition, arm=args.arm,
            reload_result=args.reload_result, minimal_capture=args.minimal_capture, minimal_reload=args.minimal_reload)
    elif args.stage == "replay-ladder":
        from fine_tuning.notebook_rl import local_ladder_recovery
        from fine_tuning.notebook_rl.cli import artifact_path
        from fine_tuning.notebook_rl.local_ladder import _path
        cohort, record, output = map(_path, (args.cohort, args.record, args.output))
        if output.exists() or output.is_relative_to(cohort) or output == record:
            raise ValueError("native recovery replay: new output outside immutable inputs required")
        raw = record.read_bytes()
        result = local_ladder_recovery.replay(cohort, decode(raw),
            expected_preparation_sha256=args.cohort_preparation_sha256)
        if record.read_bytes() != raw:
            raise ValueError("native recovery replay: saved record changed during replay")
        result["record_file"] = {"path": str(record), "file_sha256": hashlib.sha256(raw).hexdigest()}
        save(artifact_path(output), result)
    else:
        result = compare(read(args.before), read(args.after))
        save(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if (result["status"].startswith("COMPLETE")
                 or result["status"] == "CPU_NATIVE_LADDER_RECOVERY_REPLAYED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
