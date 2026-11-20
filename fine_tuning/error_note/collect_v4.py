"""Version-4 collect stages of one round (claudedocs/plan_rl_sft_hybrid_v4_20261104.md sections 3.2-3.3, 10.3).

CLI: python -m fine_tuning.error_note.collect_v4 {rollout,note,selfcheck,resolve,rows} --round R --build-root V
         --order O --start S --limit N --output DIR [--answer-contract C --training-rules F --training-rules-sha256 H]
         generation stages: --model-path M --engine {vllm,hf} [--kv-cache-gib auto] [--max-num-seqs auto]
                            [--prefix-caching] [--loop-stop] [--stop-after-seconds T]
         rollout: [--reuse-t1 T1.jsonl --reuse-t1-sha256 H] (round 1 only: the F0 first solve becomes rollout 0)
         note: [--leak-terms TERMS.json] (private {problem_id: {aliases: [...], acronyms: [...]}})
         resolve: [--lesson-control] (the same seeds without the lesson, for the lesson effect)
         rows: --tokenizer-path M

Stages and files (all append-only; a killed stage resumes by skipping sample ids already written):
  rollout   gold-blind P0 samples j=0,1, then j=2,3 unless both were correct, in one engine pass (rollouts.jsonl)
  note      gold-aware v2 note for each ALL_WRONG group (collect.run_note), then the program gate in the same process:
            gates-v4.json holds only passing lessons, their checkable image claims and reasons without gold text
  selfcheck gold-blind TRUE/FALSE probes of those claims on the image (selfcheck.jsonl, selfcheck-v4.json)
  resolve   gold-blind: lesson + P0, RESOLVE_M samples; --lesson-control adds P0-only samples at the same seeds,
            in the same pass and file (resolve.jsonl, field "lesson")
  rows      rows.jsonl / drops.jsonl / rows_summary.json from contract v4.build_rows_v4
Sample ids are "<problem_id>#<tag>" so collect.read_jsonl_done keeps one row per sample. No stage edits model text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT
from fine_tuning.notebook_rl.prompts import p0_eval_text, user_message

from .collect import (LOOP_STOP_FINISH, PREFETCH_WORKERS, WAIT, ResourceLog, SingleGenerationDeadline, Stream, StreamStats, _blocked_fields,
                      _grader, _hf_stream_builder, _kv_arg, _seqs_arg, _vllm_stream_builder, append_jsonl, auto_seqs,
                      bounded_map, install_training_guards, read_jsonl_done, resolve_kv, run_generation, run_note,
                      single_request_binding, single_request_sha256, write_atomic, write_json, SINGLE_SAMPLING)
from .contract import MAX_NEW_TOKENS, TURN1_ANSWER_CONTRACT, VERDICT_LABELS, resolve_prompt
from .order import read_order
from .v4 import (ADAPTIVE_FIRST, ALL_WRONG, LOOP_STOPPED, PROBE_MAX_TOKENS, PROBE_SEEDS, RESOLVE_M, ROLLOUT_N,
                 UNCHECKED, VERSION, build_rows_v4, classify_group, decide_selfcheck, extract_claims, needs_more,
                 parse_probe, probe_prompt, program_gate, summarize_v4, v4_seed)

COLLECTION_FILE = "collection-v4.json"
GATES_FILE = "gates-v4.json"
SELFCHECK_FILE = "selfcheck-v4.json"
ROLLOUT_FILE = "rollouts"
SINGLE_LEDGER = "single-requests.json"
SINGLE_LEDGER_LOCK = ".single-requests.lock"


def validate_single_output(request: dict, output: dict, *, policy_sha256: str, protocol_sha256: str) -> None:
    """Reject incomplete/actionless, multiple or mismatched SINGLE completions before persistence."""
    _validate_single_trace(request, output, policy_sha256=policy_sha256, protocol_sha256=protocol_sha256, partial=False)


def validate_single_partial(request: dict, output: dict, *, policy_sha256: str, protocol_sha256: str) -> None:
    """Validate deadline evidence only; even an EOS-finished row remains inadmissible in an interrupted batch."""
    _validate_single_trace(request, output, policy_sha256=policy_sha256, protocol_sha256=protocol_sha256, partial=True)


def _validate_single_trace(request: dict, output: dict, *, policy_sha256: str, protocol_sha256: str, partial: bool) -> None:
    from .rl import sampling_sha256

    generation = output.get("generation") if isinstance(output, dict) else None
    if (not isinstance(generation, dict) or type(output.get("completion_count")) is not int
            or output["completion_count"] != 1 or type(generation.get("completion_count")) is not int
            or generation["completion_count"] != 1 or output.get("blocked")):
        raise ValueError("SINGLE: exactly one complete action-bearing output required")
    if (output.get("incomplete", False) is not partial or generation.get("incomplete", False) is not partial):
        raise ValueError("SINGLE: incomplete deadline evidence cannot be a completed output")
    expected = single_request_binding(request) | {"policy_sha256": policy_sha256, "protocol_sha256": protocol_sha256,
        "sampling_sha256": sampling_sha256(SINGLE_SAMPLING), "sampling": SINGLE_SAMPLING,
        "request_sha256": single_request_sha256(request)}
    if any(generation.get(k) != v for k, v in expected.items()):
        raise ValueError("SINGLE: request/policy/protocol/sampling binding mismatch")
    if sampling_sha256(generation["sampling"]) != generation["sampling_sha256"]:
        raise ValueError("SINGLE: altered sampling transform")
    prompt, actions = generation.get("prompt_token_ids"), generation.get("completion_token_ids")
    if (not isinstance(prompt, list) or not isinstance(actions, list) or not prompt or not actions and not partial
            or any(any(type(token) is not int or token < 0 for token in ids) for ids in (prompt, actions))):
        raise ValueError("SINGLE: original nonempty prompt and action IDs required")
    if (output.get("prompt_tokens") != len(prompt) or output.get("completion_tokens") != len(actions)
            or len(actions) > request["max_tokens"] or not isinstance(output.get("text"), str)
            or output["text"] != generation.get("text")):
        raise ValueError("SINGLE: original text or token counts mismatch")
    probabilities, supports = generation.get("sampling_logprobs"), generation.get("support_sha256")
    if (not isinstance(probabilities, list) or len(probabilities) != len(actions)
            or any(type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in probabilities)
            or not isinstance(supports, list) or len(supports) != len(actions)):
        raise ValueError("SINGLE: one finite processed probability/support per action required")
    hashes = [generation.get("input_sha256"), policy_sha256, protocol_sha256, *supports]
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
        raise ValueError("SINGLE: invalid input/policy/protocol/support digest")
    if partial:
        import torch
        from fine_tuning.notebook_rl.input_preparation import digest, tensor_identity

        identities = generation.get("input_tensors")
        if (not isinstance(identities, dict) or digest(identities) != generation["input_sha256"]
                or identities.get("input_ids") != tensor_identity(torch.tensor([prompt], dtype=torch.long))):
            raise ValueError("SINGLE: deadline prompt/input tensor binding mismatch")
    eos = generation.get("eos_token_ids")
    if not isinstance(eos, list) or any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("SINGLE: actual EOS identity required")
    finish = generation.get("finish_reason")
    allowed_finishes = ("stop", "length", LOOP_STOP_FINISH) + (("deadline",) if partial else ())
    if (output.get("finish_reason") != finish or finish not in allowed_finishes
            or any(token in eos for token in actions[:-1]) or ((bool(actions) and actions[-1] in eos) != (finish == "stop"))
            or (finish == "length" and len(actions) != request["max_tokens"])
            or (not actions and finish != "deadline") or (finish == "deadline" and len(actions) >= request["max_tokens"])):
        raise ValueError("SINGLE: termination differs from original actions; EOS cannot be manufactured")


def run_single_requests(requests: list[dict], stream: Stream, stats: StreamStats, out: Path, *,
                        policy_sha256: str, protocol_sha256: str, stop_after: float | None = None,
                        follow_up: Callable[[dict, dict], list[dict]] | None = None) -> dict[str, dict]:
    """Exclusively own the per-policy ledger through generation and persistence; refuse overlapping collectors.

    `out` belongs to one frozen-policy block. Additional stages may append distinct request IDs in the same directory.
    A deadline or interrupted STARTED request is INCOMPLETE and must be preserved for the parent to resolve.
    follow_up(request, output) -> next requests (a later stage of the same question), called once per completed
    request, including reused ones on resume; it needs a continuous stream (WAIT) so later stages start while
    other questions are still generating. The result then also holds every follow-up's output.
    """
    import fcntl

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep the inode: unlinking permits overlapping collectors to lock different files.
    with (out / SINGLE_LEDGER_LOCK).open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("SINGLE: BUSY: collection ledger is busy; another collector owns this output directory") from error
        return _run_single_requests_locked(requests, stream, stats, out, policy_sha256=policy_sha256,
                                            protocol_sha256=protocol_sha256, stop_after=stop_after, follow_up=follow_up)


def _run_single_requests_locked(requests: list[dict], stream: Stream, stats: StreamStats, out: Path, *,
                                policy_sha256: str, protocol_sha256: str, stop_after: float | None = None,
                                follow_up: Callable[[dict, dict], list[dict]] | None = None) -> dict[str, dict]:
    """Reuse only verified completions while the caller owns SINGLE_LEDGER_LOCK."""
    from collections import deque

    from .rl import sampling_sha256

    if follow_up is not None and not getattr(stream, "continuous", False):
        raise ValueError("SINGLE: follow-up requests need a continuous (WAIT-capable) stream")
    bindings = [single_request_binding(r) for r in requests]
    ids = [r["request_id"] for r in bindings]
    if (len(ids) != len(set(ids)) or len({(r["block_id"], r["stage"], r["problem_id"]) for r in bindings}) != len(ids)):
        raise ValueError("SINGLE: duplicate request or second candidate")
    expected = dict(zip(ids, requests, strict=True))
    fingerprints = {key: single_request_sha256(r) for key, r in expected.items()}
    header = {"version": "single-request-ledger-v1", "policy_sha256": policy_sha256,
              "protocol_sha256": protocol_sha256, "sampling_sha256": sampling_sha256(SINGLE_SAMPLING)}
    path = Path(out) / SINGLE_LEDGER
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in saved.items() if key != "sha256"}
        if (saved.get("sha256") != _digest(body) or any(saved.get(k) != v for k, v in header.items())
                or saved.get("error")):
            raise ValueError("SINGLE: corrupt or stale collection ledger")
        records = saved["requests"]
    else:
        records = {}
    for key, record in records.items():
        if (record.get("state") != "COMPLETED" or record.get("output_sha256") != _digest(record.get("output"))):
            raise ValueError(f"SINGLE: INCOMPLETE or altered request {key}; resampling is forbidden")
        if key in expected:
            if record.get("request_sha256") != fingerprints[key]:
                raise ValueError("SINGLE: request changed during resume")
            validate_single_output(expected[key], record["output"], policy_sha256=policy_sha256,
                                   protocol_sha256=protocol_sha256)
    existing = {tuple(record["output"]["generation"][k] for k in ("block_id", "stage", "problem_id")): key
                for key, record in records.items()}
    if len(existing) != len(records) or any(
            (r["block_id"], r["stage"], r["problem_id"]) in existing
            and existing[(r["block_id"], r["stage"], r["problem_id"])] != r["request_id"] for r in bindings):
        raise ValueError("SINGLE: second candidate or changed request identity during resume")
    all_bindings = bindings + [r["output"]["generation"] for r in records.values()]
    blocks = {r["block_id"] for r in all_bindings}
    if len(blocks) > 1 or len({r["problem_id"] for r in all_bindings}) > 16:
        raise ValueError("SINGLE: ledger belongs to one frozen block of at most 16 different questions")
    Path(out).mkdir(parents=True, exist_ok=True)
    failure = None

    def persist() -> None:
        body = header | {"requests": records, **({"error": failure} if failure else {})}
        write_json(path, body | {"sha256": _digest(body)})

    def register(request: dict) -> None:
        """A follow-up joins the same block ledger under the same identity/duplicate rules."""
        binding = single_request_binding(request)
        key = binding["request_id"]
        if key in expected or any(single_request_binding(r)["stage"] == binding["stage"]
                                  and single_request_binding(r)["problem_id"] == binding["problem_id"]
                                  for r in expected.values()):
            raise ValueError("SINGLE: duplicate follow-up request or second candidate")
        if binding["block_id"] != bindings[0]["block_id"]:
            raise ValueError("SINGLE: a follow-up must stay in its frozen block")
        expected[key] = request
        fingerprints[key] = single_request_sha256(request)
        ids.append(key)

    queue: deque[dict] = deque()

    def advance(request: dict, output: dict) -> None:
        for later in follow_up(request, output) if follow_up is not None else ():
            register(later)
            done = records.get(later["request_id"])
            if done is None:
                queue.append(later)
                continue
            if done.get("request_sha256") != fingerprints[later["request_id"]]:
                raise ValueError("SINGLE: request changed during resume")
            validate_single_output(later, done["output"], policy_sha256=policy_sha256, protocol_sha256=protocol_sha256)
            advance(later, done["output"])  # A reused completion re-derives its own next stage.

    for request in requests:
        if request["request_id"] in records:
            advance(request, records[request["request_id"]]["output"])
    todo = [r for r in requests if r["request_id"] not in records] + list(queue)
    queue.clear()
    if todo and (admit := getattr(stream, "admit", None)) is not None:
        admit(todo)  # Before pending() writes STARTED: rejected capacity has consumed no answer or RNG draw.
    set_start = getattr(stream, "set_start_callback", None)

    def start_batch(batch: list[dict]) -> None:
        if len({request["request_id"] for request in batch}) != len(batch):
            raise ValueError("SINGLE: duplicate request in a batch before STARTED")
        for request in batch:
            key = request["request_id"]
            if key not in expected or key in records or single_request_sha256(request) != fingerprints[key]:
                raise ValueError("SINGLE: unexpected, duplicate or changed batch before STARTED")
        for request in batch:
            key = request["request_id"]
            records[key] = {"state": "STARTED", "request_sha256": fingerprints[key]}
        persist()

    in_flight = 0

    def pending() -> Iterator[dict]:
        nonlocal in_flight
        queue.extend(todo)
        while queue or (follow_up is not None and in_flight):
            if not queue:
                yield WAIT  # later stages appear only when an active question completes
                continue
            request = queue.popleft()
            if set_start is None:
                start_batch([request])
            in_flight += 1
            yield request

    completed = set()
    if todo:
        try:
            if set_start is not None:
                set_start(start_batch)
            for request, output in stream(pending(), stats, stop_after):
                key = request.get("request_id")
                if (key not in expected or key in completed or key not in records
                        or records[key].get("state") != "STARTED"
                        or single_request_sha256(request) != fingerprints[key]):
                    raise ValueError("SINGLE: duplicate, unexpected or unstarted completion")
                validate_single_output(request, output, policy_sha256=policy_sha256, protocol_sha256=protocol_sha256)
                # JSON roundtrip detaches caller-owned mutable lists from the authoritative persisted evidence.
                output = json.loads(json.dumps(output, allow_nan=False))
                records[key] = {"state": "COMPLETED", "request_sha256": fingerprints[key], "output": output,
                                "output_sha256": _digest(output)}
                persist()
                completed.add(key)
                in_flight -= 1
                advance(request, output)
        except BaseException as exc:
            from .capacity import CapacityHold

            if not (set_start is not None and isinstance(exc, CapacityHold)
                    and all(record.get("state") == "COMPLETED" for record in records.values())):
                failure = type(exc).__name__ + ": " + str(exc)[:500]
                if isinstance(exc, SingleGenerationDeadline):
                    try:
                        partials = exc.partials
                        keys = [partial["generation"]["request_id"] for partial in partials]
                        started = {key for key, record in records.items() if record["state"] == "STARTED"}
                        if not keys or keys != exc.request_ids or len(keys) != len(set(keys)) or not set(keys) <= started:
                            raise ValueError("deadline trace must cover its started batch exactly once")
                        for key, partial in zip(keys, partials, strict=True):
                            validate_single_partial(expected[key], partial, policy_sha256=policy_sha256,
                                                    protocol_sha256=protocol_sha256)
                        detached = json.loads(json.dumps(partials, allow_nan=False))
                        for key, partial in zip(keys, detached, strict=True):
                            records[key].update(partial_output=partial, partial_output_sha256=_digest(partial))
                    except (KeyError, TypeError, ValueError) as invalid:
                        failure = "INVALID_DEADLINE_TRACE: " + str(invalid)[:500]
                        persist()
                        raise ValueError("SINGLE: invalid deadline trace; collection remains INCOMPLETE") from invalid
                persist()
            raise
        finally:
            if set_start is not None:
                set_start(None)
    if any(key not in records or records[key].get("state") != "COMPLETED" for key in ids):
        raise ValueError("SINGLE: INCOMPLETE collection; missing completions cannot be declared complete")
    return {key: records[key]["output"] for key in ids}


def sample_id(problem_id: str, tag: str) -> str:
    return f"{problem_id}#{tag}"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def by_item(rows: dict[str, dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows.values():
        grouped.setdefault(row["item"], []).append(row)
    return {pid: sorted(values, key=lambda r: r["j"]) for pid, values in grouped.items()}


def rollouts_of(out: Path) -> dict[str, list[dict]]:
    return by_item(read_jsonl_done(Path(out) / f"{ROLLOUT_FILE}.jsonl", repair=False))


def _graded(grade: Callable, pid: str, output: dict, mode: str) -> tuple[str, str | None]:
    if output["finish_reason"] == LOOP_STOP_FINISH:
        return LOOP_STOPPED, None  # stopped by the repetition detector: reward 0, never sent to the grader
    return grade(pid, output["text"], output["finish_reason"], mode)


def _sample_row(request: dict, output: dict, grade: Callable, engine: str, stage: str) -> dict:
    row = {"problem_id": request["request_id"], "item": request["item"], "j": request["j"], "stage": stage,
           "seed": request["seed"], **({"prompt": request["prompt"]} if "prompt" in request else {})}
    if output.get("blocked"):
        return row | _blocked_fields(output) | {"prompt_tokens": None, "completion_tokens": 0, "extracted": None,
                                                "finish_reason": "blocked", "text": "", "seconds": 0.0, "engine": engine}
    status, extracted = _graded(grade, request["item"], output, request["mode"])
    return row | {"prompt_tokens": output["prompt_tokens"], "completion_tokens": output["completion_tokens"],
                  "finish_reason": output["finish_reason"], "text": output["text"], "status": status,
                  "extracted": extracted, "seconds": output["seconds"], "engine": engine}


def _requests(todo: list[tuple[str, int]], actors: dict, load: Callable, S: int,
              make: Callable[[Any, int], tuple[str, int, str]]) -> Iterator[dict]:
    """todo: (problem_id, index); make(actor, index) -> (prompt text, seed, tag). Images are prefetched on CPU."""
    for (actor, images), (pid, index) in zip(bounded_map(lambda pair: (actors[pair[0]], load(actors[pair[0]])), todo,
                                                         workers=PREFETCH_WORKERS, buffer=2 * S), todo, strict=True):
        prompt, seed, tag = make(actor, index)
        yield {"request_id": sample_id(pid, tag), "item": pid, "j": index, "seed": seed, "prompt": prompt,
               "messages": [user_message(prompt, len(images))], "images": images, "max_tokens": MAX_NEW_TOKENS,
               "mode": actor.mode}


def run_rollout(items: list[dict], actors: dict, load: Callable, stream: Stream, stats: StreamStats, grade: Callable,
                out: Path, *, round_index: int, S: int, B: int, engine: str, stop_after: float | None = None,
                gpu_probe: Callable | None = None, reuse_t1: dict[str, dict] | None = None) -> dict:
    """Adaptive gold-blind rollouts in one engine pass: j=0,1 for every item, then j=2,3 for an item as soon as its
    first two are graded and not both correct (WAIT keeps the engine busy meanwhile). reuse_t1: the F0 first solves
    (round 1 only) become rollout 0 and are regraded here with the current contract."""
    out = Path(out)
    path = out / f"{ROLLOUT_FILE}.jsonl"
    ids = [item["problem_id"] for item in items]
    if reuse_t1 is not None:
        if round_index != 1:
            raise ValueError("rollout: a reused F0 T1 is valid only for round 1 (the F0 policy)")
        done = read_jsonl_done(path)
        for pid in ids:
            key = sample_id(pid, "r0")
            if key in done or pid not in reuse_t1:
                continue
            first = reuse_t1[pid]
            if "text" not in first or first.get("stage") != "t1":
                raise ValueError(f"rollout: reused T1 row of {pid} has no generated text")
            status, extracted = _graded(grade, pid, first, actors[pid].mode)
            append_jsonl(path, {"problem_id": key, "item": pid, "j": 0, "stage": "rollout", "seed": first["seed"],
                                "reused_t1": True, "prompt_tokens": first["prompt_tokens"],
                                "completion_tokens": first["completion_tokens"], "finish_reason": first["finish_reason"],
                                "text": first["text"], "status": status, "extracted": extracted,
                                "seconds": first.get("seconds", 0.0), "engine": first.get("engine")})
    done = read_jsonl_done(path)
    statuses: dict[str, dict[int, str]] = {}
    for row in done.values():
        statuses.setdefault(row["item"], {})[row["j"]] = row["status"]

    def request(actor: Any, images: list, j: int) -> dict:
        prompt = p0_eval_text(actor.question, actor.choices)
        return {"request_id": sample_id(actor.problem_id, f"r{j}"), "item": actor.problem_id, "j": j,
                "seed": v4_seed(actor.problem_id, round_index, "roll", j), "prompt": prompt,
                "messages": [user_message(prompt, len(images))], "images": images, "max_tokens": MAX_NEW_TOKENS,
                "mode": actor.mode}

    def second(pending: dict[str, tuple[Any, list]]) -> Iterator[dict]:
        for pid in [pid for pid in pending if len(statuses.get(pid, {}).keys() & set(range(ADAPTIVE_FIRST))) == ADAPTIVE_FIRST]:
            actor, images = pending.pop(pid)
            if needs_more([statuses[pid][j] for j in range(ADAPTIVE_FIRST)]):
                yield from (request(actor, images, j) for j in range(ADAPTIVE_FIRST, ROLLOUT_N)
                            if sample_id(pid, f"r{j}") not in done)

    def requests() -> Iterator[Any]:
        pending: dict[str, tuple[Any, list]] = {}
        for actor, images in bounded_map(lambda pid: (actors[pid], load(actors[pid])), ids,
                                         workers=PREFETCH_WORKERS, buffer=2 * S):
            yield from (request(actor, images, j) for j in range(ADAPTIVE_FIRST)
                        if sample_id(actor.problem_id, f"r{j}") not in done)
            pending[actor.problem_id] = (actor, images)
            yield from second(pending)
        while pending:
            before = len(pending)
            yield from second(pending)
            if pending and len(pending) == before:
                yield WAIT

    def handle(req: dict, output: dict) -> dict:
        row = _sample_row(req, output, grade, engine, "rollout")
        statuses.setdefault(req["item"], {})[req["j"]] = row["status"]
        return row

    receipt = run_generation(ROLLOUT_FILE, out, requests(), stream, stats, handle, total=len(ids) * ROLLOUT_N, S=S,
                             B=B, engine=engine, stop_after=stop_after, gpu_probe=gpu_probe)
    groups = rollouts_of(out)
    receipt["groups"] = dict(Counter(classify_group([r["status"] for r in groups.get(pid, [])])[0] for pid in ids))
    return receipt


def note_inputs(groups: dict[str, list[dict]], ids: list[str]) -> dict[str, dict]:
    """The first wrong rollout of each ALL_WRONG group as the note's earlier solution (a loop stop shows as CUT OFF)."""
    chosen = {}
    for pid in ids:
        samples = groups.get(pid, [])
        if classify_group([r["status"] for r in samples])[0] != ALL_WRONG:
            continue
        for sample in samples:
            status = "TRUNCATED" if sample["status"] == LOOP_STOPPED else sample["status"]
            if status in VERDICT_LABELS and status != "CORRECT":
                chosen[pid] = {"text": sample["text"], "status": status, "note_j": sample["j"]}
                break
    return chosen


def write_gates(out: Path, ids: list[str], gold: dict[str, str], actors: dict, round_index: int,
                leak_terms: dict[str, dict] | None = None) -> dict:
    """Program gate in the gold-aware process; the written file carries no gold and no failing lesson."""
    out = Path(out)
    notes = read_jsonl_done(out / "note.jsonl", repair=False)
    items = {}
    for pid in ids:
        if pid not in notes:
            continue
        terms = (leak_terms or {}).get(pid, {})
        gate = program_gate(notes[pid], gold[pid], actors[pid].choices, aliases=terms.get("aliases", ()),
                            acronyms=terms.get("acronyms", ()))
        if gate["result"] == "PASS":
            gate["claims"] = extract_claims(notes[pid]["text"], gold[pid], actors[pid].choices)
        else:
            gate["lesson"] = None
        items[pid] = gate
    note_file = out / "note.jsonl"
    body = {"version": VERSION, "round": round_index,
            "note_sha256": _file_sha(note_file) if note_file.exists() else None, "items": items}
    value = body | {"sha256": _digest(body)}
    write_json(out / GATES_FILE, value)
    return value


def run_note_v4(items: list[dict], actors: dict, gold: dict[str, str], load: Callable, stream: Stream,
                stats: StreamStats, grade: Callable, out: Path, *, round_index: int, answer_contract: str,
                leak_terms: dict[str, dict] | None = None, **options: Any) -> dict:
    """Gold-aware notes for the ALL_WRONG groups only, then the program gate in this same process. No ALL_WRONG
    group (or none with a gradable first rollout) means no generation and an empty gates file."""
    out = Path(out)
    chosen = note_inputs(rollouts_of(out), [item["problem_id"] for item in items])
    wrong = [item for item in items if item["problem_id"] in chosen]
    receipt = (run_note(wrong, actors, chosen, gold, load, stream, stats, grade, out, **options, version="v2",
                        answer_contract=answer_contract) if wrong else None)
    gates = write_gates(out, [item["problem_id"] for item in wrong], gold, actors, round_index, leak_terms)
    return {"note": receipt, "all_wrong_groups": len(wrong), "gates_sha256": gates["sha256"],
            "passed": sum(g["result"] == "PASS" for g in gates["items"].values())}


def read_gates(out: Path) -> dict[str, dict]:
    value = json.loads((Path(out) / GATES_FILE).read_text(encoding="utf-8"))
    if value.get("version") != VERSION or value.get("sha256") != _digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("gates-v4.json: version or digest mismatch")
    return value["items"]


def run_selfcheck(items: list[dict], actors: dict, gates: dict[str, dict], load: Callable, stream: Stream,
                  stats: StreamStats, out: Path, *, round_index: int, S: int, B: int, engine: str,
                  stop_after: float | None = None, gpu_probe: Callable | None = None) -> dict:
    """Gold-blind probes: each claim of a passing note, PROBE_SEEDS times, image + one-sentence question only."""
    out = Path(out)
    done = read_jsonl_done(out / "selfcheck.jsonl")
    plan = {pid: gates[pid]["claims"] for pid in (item["problem_id"] for item in items)
            if pid in gates and gates[pid]["result"] == "PASS"}
    todo = [(pid, k * PROBE_SEEDS + s) for pid, claims in plan.items() for k in range(len(claims))
            for s in range(PROBE_SEEDS) if sample_id(pid, f"c{k}s{s}") not in done]

    def make(actor: Any, index: int) -> tuple[str, int, str]:
        k, s = divmod(index, PROBE_SEEDS)
        return probe_prompt(plan[actor.problem_id][k]), v4_seed(actor.problem_id, round_index, "probe", index), f"c{k}s{s}"

    def handle(request: dict, output: dict) -> dict:
        row = {"problem_id": request["request_id"], "item": request["item"], "j": request["j"], "stage": "selfcheck",
               "seed": request["seed"], "prompt": request["prompt"]}
        if output.get("blocked"):
            return row | _blocked_fields(output) | {"verdict": "INVALID", "engine": engine}
        return row | {"text": output["text"], "finish_reason": output["finish_reason"],
                      "verdict": parse_probe(output["text"]), "completion_tokens": output["completion_tokens"],
                      "seconds": output["seconds"], "engine": engine}

    def requests() -> Iterator[dict]:
        for request in _requests(todo, actors, load, S, make):
            yield request | {"max_tokens": PROBE_MAX_TOKENS}

    receipt = run_generation("selfcheck", out, requests(), stream, stats, handle, total=len(todo), S=S, B=B,
                             engine=engine, stop_after=stop_after, gpu_probe=gpu_probe)
    rows = by_item(read_jsonl_done(out / "selfcheck.jsonl", repair=False))
    decisions = {}
    for pid, claims in plan.items():
        verdicts = [[r["verdict"] for r in rows.get(pid, []) if r["j"] // PROBE_SEEDS == k] for k in range(len(claims))]
        complete = all(len(v) == PROBE_SEEDS for v in verdicts)
        decisions[pid] = {"result": decide_selfcheck(verdicts) if complete and claims else UNCHECKED,
                          "claims": len(claims), "complete": complete}
    body = {"version": VERSION, "round": round_index, "gates_sha256": _file_sha(out / GATES_FILE), "items": decisions}
    write_json(out / SELFCHECK_FILE, body | {"sha256": _digest(body)})
    return receipt


def read_selfcheck(out: Path) -> dict[str, dict]:
    path = Path(out) / SELFCHECK_FILE
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("version") != VERSION or value.get("sha256") != _digest({k: v for k, v in value.items() if k != "sha256"})
            or value["gates_sha256"] != _file_sha(Path(out) / GATES_FILE)):
        raise ValueError("selfcheck-v4.json: version, digest or gates binding mismatch")
    return value["items"]


def run_resolve_v4(items: list[dict], actors: dict, gates: dict[str, dict], checks: dict[str, dict], load: Callable,
                   stream: Stream, stats: StreamStats, grade: Callable, out: Path, *, round_index: int, S: int, B: int,
                   engine: str, control: bool = False, stop_after: float | None = None,
                   gpu_probe: Callable | None = None) -> dict:
    """RESOLVE_M gold-blind re-solves with the lesson (tag l<j>); with control, P0-only samples at the same seeds
    (tag c<j>), all in one engine pass and one file. Only the lesson block differs between a pair."""
    out = Path(out)
    eligible = [item["problem_id"] for item in items if gates.get(item["problem_id"], {}).get("result") == "PASS"
                and checks.get(item["problem_id"], {}).get("result") != "SELF_CONTRADICTED"]
    done = read_jsonl_done(out / "resolve.jsonl")
    kinds = ("l", "c") if control else ("l",)
    todo = [(pid, index) for pid in eligible for index in range(len(kinds) * RESOLVE_M)
            if sample_id(pid, f"{kinds[index // RESOLVE_M]}{index % RESOLVE_M}") not in done]

    def make(actor: Any, index: int) -> tuple[str, int, str]:
        kind, j = kinds[index // RESOLVE_M], index % RESOLVE_M
        p0 = p0_eval_text(actor.question, actor.choices)
        prompt = resolve_prompt(p0, gates[actor.problem_id]["lesson"]) if kind == "l" else p0
        return prompt, v4_seed(actor.problem_id, round_index, "resolve", j), f"{kind}{j}"

    def requests() -> Iterator[dict]:
        for req in _requests(todo, actors, load, S, make):
            yield req | {"j": req["j"] % RESOLVE_M, "lesson": req["request_id"].endswith(tuple(f"#l{j}" for j in range(RESOLVE_M)))}

    def handle(req: dict, output: dict) -> dict:
        return _sample_row(req, output, grade, engine, "resolve") | {"lesson": req["lesson"]}

    receipt = run_generation("resolve", out, requests(), stream, stats, handle, total=len(todo), S=S, B=B,
                             engine=engine, stop_after=stop_after, gpu_probe=gpu_probe)
    receipt |= {"eligible": len(eligible), "control": control}
    return receipt


def make_records_v4(items: list[dict], actors: dict, out: Path) -> list[dict]:
    out = Path(out)
    groups = rollouts_of(out)
    notes = read_jsonl_done(out / "note.jsonl", repair=False) if (out / "note.jsonl").exists() else {}
    gates = read_gates(out) if (out / GATES_FILE).exists() else {}
    checks = read_selfcheck(out) if (out / GATES_FILE).exists() else {}
    resolved = read_jsonl_done(out / "resolve.jsonl", repair=False)
    resolves = by_item({k: v for k, v in resolved.items() if v["lesson"]})
    controls = by_item({k: v for k, v in resolved.items() if not v["lesson"]})
    records = []
    for item in items:
        pid, actor = item["problem_id"], actors[item["problem_id"]]
        note = notes.get(pid)
        records.append({"problem_id": pid, "subject": item["subject"], "source": item["source"],
                        "p0_text": p0_eval_text(actor.question, actor.choices), "image_paths": list(actor.image_paths),
                        "rollouts": groups.get(pid, []),
                        "note": {k: note[k] for k in ("prompt", "text", "finish_reason")} if note and "status" not in note
                        else None, "gate": gates.get(pid), "selfcheck": checks.get(pid),
                        "resolves": resolves.get(pid, []), "controls": controls.get(pid, [])})
    return records


def collect_rows_v4(records: list[dict], tokenize: Callable[[str], list[int]]) -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    drops: list[dict] = []
    for record in records:
        built, dropped = build_rows_v4(record, tokenize)
        rows += built
        drops += dropped
    return rows, drops, summarize_v4(records, rows, drops)


# ------------------------------------------------------------------ CLI

def _items(args: argparse.Namespace) -> list[dict]:
    items = read_order(Path(args.order))["items"]
    if args.start < 0 or args.limit < 1 or args.start + args.limit > len(items):
        raise ValueError(f"--start/--limit must select inside the order of {len(items)} items")
    return items[args.start:args.start + args.limit]


def bind_collection_v4(args: argparse.Namespace) -> dict:
    """The round's identity, written by its first stage and required unchanged by every later stage."""
    out = Path(args.output)
    body = {"version": VERSION, "round": args.round, "start": args.start, "limit": args.limit,
            "order_sha256": _file_sha(args.order), "answer_contract": args.answer_contract,
            "training_rules_sha256": args.training_rules_sha256 if args.training_rules else None}
    path = out / COLLECTION_FILE
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != body:
            raise ValueError("collect v4: this output directory belongs to another round, slice, order or contract")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(path, body)
    return body


def _generate_v4(args: argparse.Namespace, stage: str) -> int:
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    from fine_tuning.notebook_rl.verifier_service import VerifierService

    build_root, out, vllm = Path(args.build_root), Path(args.output), args.engine == "vllm"
    items = _items(args)
    bind_collection_v4(args)
    ids = [item["problem_id"] for item in items]
    training = args.answer_contract == TRAINING_ANSWER_CONTRACT
    if training and (args.training_rules is None or args.training_rules_sha256 is None):
        raise ValueError(f"{TRAINING_ANSWER_CONTRACT} needs --training-rules and --training-rules-sha256")
    service_options = ({"training_rules_path": args.training_rules, "training_rules_sha256": args.training_rules_sha256}
                       if training else {})
    gold = leak_terms = reuse = None
    if stage == "note":  # the gold is read before any guard, in this process only
        verifier = read_verifier(build_root / "train")
        gold = {pid: verifier[pid].gold for pid in ids}
        del verifier
        if args.leak_terms is not None:
            leak_terms = json.loads(Path(args.leak_terms).read_text(encoding="utf-8"))
    if stage == "rollout" and args.reuse_t1 is not None:
        if args.reuse_t1_sha256 is None or _file_sha(args.reuse_t1) != args.reuse_t1_sha256:
            raise ValueError("rollout: --reuse-t1 needs a matching --reuse-t1-sha256")
        reuse = read_jsonl_done(Path(args.reuse_t1), repair=False)
    private = [p for p in (args.training_rules, getattr(args, "leak_terms", None)) if p is not None]
    if stage != "note":  # gold-blind stages may not open the gold-aware files
        private += [out / "note.jsonl", out / "lessons.json"]
    forbidden = install_training_guards(build_root, vllm_schema={} if vllm else None, private_files=private)
    actors = {actor.problem_id: actor for actor in read_actors(build_root / "train")}
    unknown = [pid for pid in ids if pid not in actors]
    if unknown:
        raise ValueError(f"{len(unknown)} ordered items are not in the train split, first {unknown[:3]}")
    seqs = args.max_num_seqs
    if seqs == "auto":
        candidates = auto_seqs(int(resolve_kv(args.kv_cache_gib)["kv_cache_gib_resolved"]))
        seqs = candidates[len(candidates) // 2]
    args.max_num_seqs = seqs
    stats = StreamStats()
    stream = (_vllm_stream_builder if vllm else _hf_stream_builder)(args, stats)
    options = dict(S=seqs, B=args.max_num_batched_tokens, engine=args.engine, stop_after=args.stop_after_seconds)
    load = lambda actor: load_images(actor, build_root / "images")  # noqa: E731
    with VerifierService(build_root / "train", forbidden_roots=forbidden, **service_options) as service, \
            ResourceLog(out / "resource.jsonl"):
        grade = _grader(service, args.answer_contract)
        if stage == "rollout":
            receipt: Any = run_rollout(items, actors, load, stream, stats, grade, out, round_index=args.round,
                                       reuse_t1=reuse, **options)
        elif stage == "note":
            assert gold is not None
            receipt = run_note_v4(items, actors, gold, load, stream, stats, grade, out, round_index=args.round,
                                  answer_contract=args.answer_contract, leak_terms=leak_terms, **options)
        elif stage == "selfcheck":
            receipt = run_selfcheck(items, actors, read_gates(out), load, stream, stats, out, round_index=args.round,
                                    **options)
        else:
            receipt = run_resolve_v4(items, actors, read_gates(out), read_selfcheck(out), load, stream, stats, grade,
                                     out, round_index=args.round, control=args.lesson_control, **options)
    print(json.dumps(receipt))
    return 0


def _rows_v4(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from fine_tuning.notebook_rl.vault import read_actors

    out = Path(args.output)
    bind_collection_v4(args)
    items = _items(args)
    actors = {actor.problem_id: actor for actor in read_actors(Path(args.build_root) / "train")}
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True)
    rows, drops, summary = collect_rows_v4(make_records_v4(items, actors, out),
                                           lambda text: tokenizer.encode(text, add_special_tokens=False))
    write_atomic(out / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(out / "drops.jsonl", "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in drops))
    write_json(out / "rows_summary.json", summary)
    print(json.dumps(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rollout", "note", "selfcheck", "resolve", "rows"):
        p = sub.add_parser(name)
        p.add_argument("--round", type=int, required=True)
        for key in ("build-root", "order", "output"):
            p.add_argument(f"--{key}", type=Path, required=True)
        p.add_argument("--start", type=int, required=True)
        p.add_argument("--limit", type=int, required=True)
        p.add_argument("--answer-contract", choices=(TURN1_ANSWER_CONTRACT, TRAINING_ANSWER_CONTRACT),
                       default=TRAINING_ANSWER_CONTRACT)
        p.add_argument("--training-rules", type=Path)
        p.add_argument("--training-rules-sha256")
        if name == "rows":
            p.add_argument("--tokenizer-path", type=Path, required=True)
            continue
        p.add_argument("--model-path", type=Path, required=True)
        p.add_argument("--engine", choices=("vllm", "hf"), required=True)
        p.add_argument("--kv-cache-gib", type=_kv_arg, default="auto")
        p.add_argument("--max-num-batched-tokens", type=int, default=16384)
        p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        p.add_argument("--max-num-seqs", type=_seqs_arg, default="auto")
        p.add_argument("--stop-after-seconds", type=float, default=None)
        p.add_argument("--prefix-caching", action="store_true", help="vLLM prefix caching (version-4 rollouts)")
        p.add_argument("--loop-stop", action="store_true", help="stop a training sample that repeats one 64-token block")
    sub.choices["rollout"].add_argument("--reuse-t1", type=Path)
    sub.choices["rollout"].add_argument("--reuse-t1-sha256")
    sub.choices["note"].add_argument("--leak-terms", type=Path)
    sub.choices["resolve"].add_argument("--lesson-control", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.round < 1:
        raise ValueError("--round must be at least 1")
    if args.command == "rows":
        return _rows_v4(args)
    for name in ("leak_terms", "reuse_t1", "reuse_t1_sha256"):
        if not hasattr(args, name):
            setattr(args, name, None)
    if not hasattr(args, "lesson_control"):
        args.lesson_control = False
    return _generate_v4(args, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
