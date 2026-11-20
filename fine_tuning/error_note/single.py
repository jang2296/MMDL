"""Same-policy, single-response RL + self-note SFT with program-only row admission (protocol SINGLE v8).

The CLI collects a preregistered local prefix and proves backward without an optimizer.
Rows are admitted by program checks only (graded correct, normal stop, no loop, no lesson leak);
note/solution content is unverified. Paid execution is absent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from .audit import audit_single_note, digest, single_note_review_bindings, text_sha256, validate_single_note_audit
from .collect import ResourceLog, StreamStats, hf_single_stream_builder, install_training_guards, write_json
from .collect_v3 import file_sha, generation_identity
from .collect_v4 import run_single_requests
from . import references
from .contract import (CONTEXT_LIMIT, HUMAN_TARGET_SOURCE, KIND_HUMAN, MASTER_SEED, MAX_NEW_TOKENS, MIN_TARGET_TOKENS,
                       RESOLVE_INSTRUCTION_SINGLE, ROUTE_FULL, ROUTE_LM, a2_target_problem, answer_denial,
                       b_target_problem, contained_answer,
                       hint_leak_single_v2, human_solution_target, is_loop, note_prompt_single, parse_note_v2,
                       resolve_prompt_single, sample_seed)
from .protocol import (ARM, SINGLE_ADVANTAGE_V11, SINGLE_ADVANTAGE_V12, SINGLE_ADVANTAGE_VERSION_V11,
                       SINGLE_ADVANTAGE_VERSION_V12, SINGLE_ANSWER_UNION, SINGLE_NOTE_CAUSE_RULE,
                       SINGLE_NOTE_MAX_TOKENS, SINGLE_OPEN_SCORER, SINGLE_REWARD_GATE, SINGLE_REWARD_GATE_V5,
                       SINGLE_ADVANTAGE_V13, SINGLE_ADVANTAGE_VERSION_V13, SINGLE_PRAISE_V13, SINGLE_REWARD_GATE_V7,
                       SINGLE_REWARD_GATE_V8, SINGLE_SPIRAL_V13, SINGLE_TOKEN_SCALE_V11, VERSION_SINGLE, VERSION_SINGLE_V12,
                       VERSION_SINGLE_V13, SINGLE_A2_MIN_TOKENS_V14, SINGLE_ADVANTAGE_V14, SINGLE_ADVANTAGE_VERSION_V14,
                       SINGLE_REWARD_GATE_V9, SINGLE_TAIL_PRAISE_V14, VERSION_SINGLE_V14, plain_protocol_single,
                       ROLLOUT_ADAPTER_V15, SINGLE_RESOLVE_STAGES_V15, SINGLE_RESOLVE_V15, SINGLE_ROUTES_V15,
                       VERSION_SINGLE_V15,
                       single_blocks, single_prompts_sha256, single_source_baseline, spiral_window, tail_window)
from .v4 import target_problem

VERSION = "single-response-runtime-v3"
COMPONENTS = ("PG", "KL", "A1", "A2", "B")
GRADED = frozenset(("CORRECT", "INCORRECT", "TRUNCATED", "INVALID_NO_FINAL", "INVALID_MULTIPLE"))
# A correct solution earns reward/A1/A2 only with some written reasoning: bare answers collapsed pilot v3's outputs.
SOLUTION_MIN_TOKENS = MIN_TARGET_TOKENS
# The verifier could not compare (e.g. an unbound unit): reward 0, no note (no truthful wrong/right feedback exists).
UNVERIFIABLE = "UNVERIFIABLE"
NO_COMMITTED = frozenset(("INVALID_NO_FINAL", "INVALID_MULTIPLE"))
# Protocol version -> its T1 reward gate (v11 and v12 are the runnable SINGLE protocols).
PROTOCOL_GATES = {VERSION_SINGLE: SINGLE_REWARD_GATE, VERSION_SINGLE_V12: SINGLE_REWARD_GATE_V7,
                  VERSION_SINGLE_V13: SINGLE_REWARD_GATE_V8, VERSION_SINGLE_V14: SINGLE_REWARD_GATE_V9,
                  VERSION_SINGLE_V15: SINGLE_REWARD_GATE_V9}
# Gates whose decision carries the item's option count and the v7 order (gate v8 = v7 + praise and spiral window;
# gate v9 = gate v8 + the tail window).
GATES_V7 = (SINGLE_REWARD_GATE_V7, SINGLE_REWARD_GATE_V8, SINGLE_REWARD_GATE_V9)
GATES_V8 = (SINGLE_REWARD_GATE_V8, SINGLE_REWARD_GATE_V9)  # decision carries praise and window
# rows carry a penalty-table advantage
TABLE_GATES = (SINGLE_REWARD_GATE, SINGLE_REWARD_GATE_V7, SINGLE_REWARD_GATE_V8, SINGLE_REWARD_GATE_V9)
CONTAINED_OPEN = "T1_CONTAINED_OPEN"
SINGLE_CLI_VERSIONS = {"v11": VERSION_SINGLE, "v12": VERSION_SINGLE_V12, "v13": VERSION_SINGLE_V13,
                       "v14": VERSION_SINGLE_V14, "v15": VERSION_SINGLE_V15}
NO_LESSON_CONTROL_VERSION = "no-lesson-control-diagnostic-v1"
NO_LESSON_CONTROL_FILE = "diagnostic-no-lesson-control.json"
_MCQ_P0_END = "\nPlease select the correct answer from the options above."  # prompts.p0_body, MCQ only


def grade_answer(verifier: Any, contract: str, pid: str, text: str, finish_reason: str, mode: str, *,
                 actor: Any = None) -> tuple[str, Any]:
    """One response, one grade. v8 open answers (and v14's scorer v2, with unit aliases) delegate to
    single_grading.grade_single (needs the actor). The v7
    open union keeps the V8 result unless it is not CORRECT and the unit-aware training contract grades the same
    text CORRECT; a plain contract name grades with that contract only."""
    from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT, TURN1_EVAL_CONTRACT
    from .single_grading import SCORER_V2, grade_single

    if contract in (SINGLE_OPEN_SCORER, SCORER_V2):
        if actor is None:
            raise ValueError("single grading: the v8 open scorer needs the item actor")
        return grade_single(verifier, actor, pid, text, finish_reason, mode, units=contract == SCORER_V2)
    union = contract == SINGLE_ANSWER_UNION
    value = verifier.grade(pid, text, finish_reason, mode, answer_contract=TURN1_EVAL_CONTRACT if union else contract)
    if union and value.status != "CORRECT":
        second = verifier.grade(pid, text, finish_reason, mode, answer_contract=TRAINING_ANSWER_CONTRACT)
        if second.status == "CORRECT":
            value = second
    return value.status, value.extracted


@contextmanager
def _runtime_lock(directory: Path, name: str):
    """An overlapping writer must stop before generation, phase mutation, or backward starts."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("single runtime: HOLD concurrent writer already owns this run") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("single runtime: expected a JSON object")
    return value


def _sealed(path: Path, value: dict) -> None:
    """Append a new immutable artifact, or verify its exact existing value."""
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"single runtime: immutable binding changed: {path.name}")
    else:
        write_json(path, value)


def _state(path: Path, value: dict | None = None) -> dict:
    if value is None:
        data = _read(path)
        if data.get("sha256") != digest({k: v for k, v in data.items() if k != "sha256"}):
            raise ValueError("single runtime: corrupt phase state")
        return data
    body = {k: v for k, v in value.items() if k != "sha256"}
    result = body | {"sha256": digest(body)}
    write_json(path, result)
    return result


def source_manifest() -> dict:
    """Describe actual source bytes, including an uncommitted working tree."""
    root = Path(__file__).resolve().parents[1]
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*.py"))
             if not {"tests", "runs", "generated", "__pycache__"}.intersection(p.relative_to(root).parts)}
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {"base_commit": base, "actual_files": files,
            "identity": "actual source manifest; base_commit does not assert a clean committed tree"}


def _protocol_check(protocol: dict) -> None:
    from .protocol import (VERSION_CONTENT_V16, VERSION_V16B, _digest,
                           validate_content_protocol_v16, validate_protocol_v16b)

    if protocol.get("version") == "error-note-v18b":
        from .protocol import validate_protocol_v18b

        validate_protocol_v18b(protocol)
        return
    if protocol.get("version") == "error-note-v18":
        from .protocol import validate_protocol_v18

        validate_protocol_v18(protocol)
        return
    if protocol.get("version") == "error-note-v17":
        from .protocol import validate_protocol_v17

        validate_protocol_v17(protocol)
        return
    if protocol.get("version") == VERSION_V16B:
        validate_protocol_v16b(protocol)
        return
    if protocol.get("version") == VERSION_CONTENT_V16:
        validate_content_protocol_v16(protocol)
        return

    if (protocol.get("version") not in PROTOCOL_GATES
            or protocol.get("sha256") != _digest({k: v for k, v in protocol.items() if k != "sha256"})
            or protocol.get("prompts_sha256") != single_prompts_sha256()
            or protocol["training_contract"].get("version") != protocol["version"]
            or protocol["training_contract"]["rl"].get("reward") != PROTOCOL_GATES[protocol["version"]]
            or protocol["training_contract"]["generation_counts"] != {
                "t1": 1, "note": 1, "resolve": 3 if protocol["version"] == VERSION_SINGLE_V15 else 1}
            or (protocol["version"] == VERSION_SINGLE_V15) != ("rollout_adapter" in protocol)):
        raise ValueError("single runtime: invalid single-response protocol")
    if protocol["version"] == VERSION_SINGLE_V15 and (
            protocol["rollout_adapter"] != ROLLOUT_ADAPTER_V15
            or protocol["training_contract"].get("routes") != SINGLE_ROUTES_V15
            or protocol["training_contract"].get("resolve_attempts") != SINGLE_RESOLVE_V15):
        raise ValueError("single runtime: invalid single-response protocol")


def _candidate(record: dict, kind: str, *, stage: str = "resolve", image_route: bool = False) -> dict | None:
    """``stage`` is the A2 re-solve stage. ``image_route`` (v15): every row of an item with an image is FULL and every
    row of a text-only item LM; otherwise (v11-v14) only an A2/B row whose note says MISREAD is FULL."""
    output = record.get({"RL": "t1", "A1": "t1", "A2": stage, "B": "note"}[kind])
    if output is None:
        return None
    actions = output["generation"]["completion_token_ids"]
    minimum = SOLUTION_MIN_TOKENS if kind in ("A1", "A2") else 1
    if kind != "RL" and (not output["text"].strip()
                         or target_problem(output["text"], actions, output["finish_reason"], "CORRECT", min_tokens=minimum)):
        return None
    if kind in ("A1", "A2") and output["status"] != "CORRECT":
        return None
    parsed = parse_note_v2(record["note"]["text"]) if record.get("note") else None
    full = bool(record["image_paths"]) if image_route else kind in ("A2", "B") and parsed and parsed.image_reading == "MISREAD"
    route = ROUTE_FULL if full else ROUTE_LM
    return {"row_id": record["problem_id"] + ":" + kind, "kind": kind, "route": route,
            **{k: record[k] for k in ("problem_id", "image_paths", "image_sha256", "subject", "source")},
            "prompt": output["prompt"] if kind == "B" else record["p0_text"], "target": output["text"],
            "target_source": {"RL": "T1", "A1": "T1", "A2": "RESOLVE", "B": "NOTE"}[kind],
            "generation": output["generation"], "complete": True,
            "status": output.get("status", "NOTE"), "reward": 0 if kind == "RL" else int(record["t1"]["status"] == "CORRECT")}


def _a2_check(record: dict, stage: str, lesson: str, *, image_route: bool) -> tuple[dict | None, str | None]:
    """v15: the v14 A2 checks on one re-solve attempt (admissible, no lesson echo/hedge, >= 256 sampled tokens).
    (row, None) when the attempt passes, else (None, reason)."""
    row = _candidate(record, "A2", stage=stage, image_route=image_route)
    if row is None:
        return None, "A2_NOT_ADMISSIBLE"
    if problem := a2_target_problem(row["target"], lesson):
        return None, "A2_" + problem
    if len(row["generation"]["completion_token_ids"]) < SINGLE_A2_MIN_TOKENS_V14:
        return None, "A2_UNDER_MIN_TOKENS"
    return row, None


def _single_reward_decision(row: dict, evidence: dict) -> dict:
    """Rebuild the program gate from the original T1: R=1 iff graded CORRECT and an admissible A1 target,
    R=-1 iff a not-CORRECT T1 carries a penalty, else 0. Gate v5 penalizes a bare wrong answer, an uncommitted stop
    and a loop; gate v6 (v11) also every other wrong T1: cut off without a loop, or reasoned and wrong. Gate v7 (v12)
    first gives an open INCORRECT answer that contains the gold's words reason CONTAINED_OPEN and no penalty; its
    evidence carries the item's option count, and the gold is the source-bound provenance label. Gate v8 (v13)
    is gate v7 plus ``praise`` (an accepted row of at least 256 tokens) and ``window`` (the spiral window of a LOOP or
    TRUNCATED_NO_LOOP row, protocol.spiral_window on its sampled token IDs). Gate v9 (v14) adds ``tail``: the
    protocol.tail_window of a CORRECT, normally stopped T1 of at least 256 tokens whose reason is one of the tail-praise
    reasons (a correct answer that is no A1 target), else None."""
    try:
        v7 = isinstance(evidence, dict) and evidence.get("version") in GATES_V7
        # The per-version route rule (v11-v14 LM; v15 FULL iff the item has an image) is enforced by train_v4.
        if (row["kind"] != "RL" or row["target_source"] != "T1" or row["route"] not in (ROUTE_LM, ROUTE_FULL)
                or row["status"] not in GRADED | {"LOOP_STOPPED", UNVERIFIABLE}
                or not isinstance(evidence, dict)
                or set(evidence) - {"baseline", "advantage"} != {"version", "t1_metadata", "provenance", "decision",
                                                                 *(("item",) if v7 else ())}
                or evidence["version"] not in (SINGLE_REWARD_GATE_V5, *TABLE_GATES)
                or not isinstance(evidence["t1_metadata"], dict)
                or {"text", "generation", "prompt", "status"} & set(evidence["t1_metadata"])
                or row["target"] != row["generation"]["text"]):
            raise ValueError("single reward: malformed or unsupported original-T1 evidence")
        t1 = evidence["t1_metadata"] | {"text": row["target"], "generation": row["generation"],
                                        "prompt": row["prompt"], "status": row["status"]}
        record = {key: row[key] for key in ("problem_id", "image_paths", "image_sha256", "subject", "source")}
        record.update(p0_text=row["prompt"], t1=t1)
        bindings = single_note_review_bindings(record, evidence["provenance"])
        if row["status"] != "CORRECT":
            reason: str | None = "T1_UNVERIFIABLE" if row["status"] == UNVERIFIABLE else "T1_NOT_CORRECT"
        else:
            problem = target_problem(row["target"], row["generation"]["completion_token_ids"], t1["finish_reason"],
                                     "CORRECT", min_tokens=SOLUTION_MIN_TOKENS)
            reason = None if _candidate(record, "A1") is not None else "T1_INELIGIBLE_" + (problem or "EMPTY_TARGET")
            if reason is None and (phrase := answer_denial(row["target"])) is not None:
                reason = "T1_ANSWER_DENIAL:" + phrase  # v5: a correct answer that calls the question wrong earns 0
        # v7: a correct answer is never pushed down (v5's -1 on correct bare answers lowered the answer tokens);
        # nor is an UNVERIFIABLE one, which may be correct. Truncation without a loop stays 0.
        penalty, stopped = None, t1["finish_reason"] == "stop"
        all_wrong = evidence["version"] == SINGLE_REWARD_GATE
        open_item = v7 and _option_count(row, evidence["item"]) == 0
        if v7 and row["status"] not in ("CORRECT", UNVERIFIABLE):
            # v7 order: loop and cut-off penalties first (the one signal with a measured benefit), then an
            # uncommitted stop; CONTAINED_OPEN only before BARE_WRONG / REASONED_WRONG.
            if is_loop(row["target"]):
                penalty = "LOOP"
            elif row["status"] == "TRUNCATED" or not stopped:
                penalty = "TRUNCATED_NO_LOOP"
            elif row["status"] in NO_COMMITTED and t1.get("extracted") is None:
                penalty = "NO_COMMITTED_ANSWER"
            elif open_item and row["status"] == "INCORRECT" and _contains_gold(evidence["provenance"], t1.get("extracted")):
                reason = CONTAINED_OPEN
            elif len(row["generation"]["completion_token_ids"]) < SOLUTION_MIN_TOKENS:
                penalty = "BARE_WRONG"
            else:
                penalty = "REASONED_WRONG"
        elif row["status"] not in ("CORRECT", UNVERIFIABLE):
            if stopped and len(row["generation"]["completion_token_ids"]) < SOLUTION_MIN_TOKENS:
                penalty = "BARE_WRONG"
            elif stopped and row["status"] in NO_COMMITTED and t1.get("extracted") is None:
                penalty = "NO_COMMITTED_ANSWER"  # v5: never when the grader read one committed answer
            elif is_loop(row["target"]):
                penalty = "LOOP"
            elif all_wrong and (row["status"] == "TRUNCATED" or not stopped):
                penalty = "TRUNCATED_NO_LOOP"
            elif all_wrong:
                penalty = "REASONED_WRONG"  # stopped, >= 32 tokens, no loop, a committed or read wrong answer
        decision = {"accepted": reason is None, "reason": reason, "penalty": penalty, "bindings": bindings,
                    "method": "program"}
        if evidence["version"] in GATES_V8:
            actions = row["generation"]["completion_token_ids"]
            decision["praise"] = reason is None and len(actions) >= SINGLE_PRAISE_V13["min_tokens"]
            decision["window"] = spiral_window(actions) if penalty in SINGLE_SPIRAL_V13["penalties"] else None
            if evidence["version"] == SINGLE_REWARD_GATE_V9:
                # A repeating row never reaches the A1 answer-denial check above, so it is applied here as well.
                decision["tail"] = tail_window(actions) if (
                    row["status"] == "CORRECT" and stopped and reason in SINGLE_TAIL_PRAISE_V14["reasons"]
                    and len(actions) >= SINGLE_TAIL_PRAISE_V14["min_tokens"]
                    and answer_denial(row["target"]) is None) else None
        return decision
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("single reward: malformed original-T1 evidence") from error


def _option_count(row: dict, item: dict) -> int:
    """Gate v7 evidence: the item's option count (0 = open), which must agree with the original P0 text."""
    count = item.get("option_count") if isinstance(item, dict) and set(item) == {"option_count"} else None
    if type(count) is not int or count < 0 or (count > 0) != row["prompt"].endswith(_MCQ_P0_END):
        raise ValueError("single reward: gate v7 needs the item's option count matching its P0 text")
    return count


def _contains_gold(provenance: dict, extracted: Any) -> bool:
    """Gate v7 CONTAINED_OPEN test: the committed answer contains the gold's answer tokens
    (contract.contained_answer); the gold is the source-bound provenance label."""
    if extracted is None or not str(extracted).strip():
        return False
    label = provenance.get("label")
    if not isinstance(label, dict) or not isinstance(label.get("value"), str) or label.get("sha256") != text_sha256(label["value"]):
        raise ValueError("single reward: gate v7 needs the source-bound gold label")
    return contained_answer(label["value"], str(extracted))


def _reward_value(decision: dict) -> int:
    return 1 if decision["accepted"] else -1 if decision["penalty"] else 0


def single_advantage(decision: dict, gate: str = SINGLE_REWARD_GATE) -> float:
    """v11 penalty table: accepted +1, bare/uncommitted/loop -1, reasoned-wrong/cut-off -0.5, anything else 0.
    v12 (gate v7): reasoned-wrong 0 and CONTAINED_OPEN 0; the rest as v11.
    v13 (gate v8): accepted +1 only when praised (>= 256 tokens), else 0; cut-off -1 like a loop; the rest as v12.
    v14 (gate v9): as v13, and an otherwise-0 row with a ``tail`` window is praised +1 (``tail_praised``)."""
    if gate in GATES_V8:
        if decision["accepted"]:
            key = "accepted_praised" if decision["praise"] else "accepted"
        elif decision["penalty"]:
            key = decision["penalty"]
        elif decision["reason"] == CONTAINED_OPEN:
            key = "CONTAINED_OPEN"
        elif gate == SINGLE_REWARD_GATE_V9 and decision["tail"] is not None:
            key = "tail_praised"
        else:
            key = "other"
        return (SINGLE_ADVANTAGE_V14 if gate == SINGLE_REWARD_GATE_V9 else SINGLE_ADVANTAGE_V13)[key]
    if gate == SINGLE_REWARD_GATE_V7:
        key = ("accepted" if decision["accepted"] else decision["penalty"]
               or ("CONTAINED_OPEN" if decision["reason"] == CONTAINED_OPEN else "other"))
        return SINGLE_ADVANTAGE_V12[key]
    return SINGLE_ADVANTAGE_V11["accepted" if decision["accepted"] else decision["penalty"] or "other"]


def _advantage_version(gate: str) -> str:
    return {SINGLE_REWARD_GATE_V9: SINGLE_ADVANTAGE_VERSION_V14, SINGLE_REWARD_GATE_V8: SINGLE_ADVANTAGE_VERSION_V13,
            SINGLE_REWARD_GATE_V7: SINGLE_ADVANTAGE_VERSION_V12}.get(gate, SINGLE_ADVANTAGE_VERSION_V11)


def bind_single_reward(row: dict, t1: dict, provenance: dict, baseline: dict | None = None, *,
                       gate: str = SINGLE_REWARD_GATE, option_count: int | None = None) -> dict:
    """Attach evidence without duplicating response text or sampled actions in the target.

    gate v6 (v11) and v7 (v12): the row carries advantage = the penalty-table value; gate v7 also binds the item's
    ``option_count`` (0 = open). baseline (v10, gate v5 only): the protocol's baseline pin; the row then carries
    b(source) and advantage = reward - b."""
    if (gate not in (SINGLE_REWARD_GATE_V5, *TABLE_GATES)
            or (baseline is not None and gate != SINGLE_REWARD_GATE_V5)):
        raise ValueError("single reward: the v10 source baseline belongs to gate v5 only")
    if (option_count is None) != (gate not in GATES_V7):
        raise ValueError("single reward: the item option count belongs to gates v7 and v8 only")
    if any(t1[key] != row[field] for key, field in (("text", "target"), ("prompt", "prompt"),
                                                  ("generation", "generation"), ("status", "status"))):
        raise ValueError("single reward: original T1 differs from the RL row")
    evidence: dict = {"version": gate, "t1_metadata": {k: v for k, v in t1.items()
                if k not in {"text", "generation", "prompt", "status"}}, "provenance": provenance, "decision": None}
    if gate in GATES_V7:
        evidence["item"] = {"option_count": option_count}
    evidence["decision"] = _single_reward_decision(row, evidence)
    row.update(reward=_reward_value(evidence["decision"]), reward_evidence=evidence)
    if gate in TABLE_GATES:
        value = single_advantage(evidence["decision"], gate)
        evidence["advantage"] = {"version": _advantage_version(gate), "value": value}
        row["advantage"] = value
    if baseline is not None:
        evidence["baseline"] = single_source_baseline(row["problem_id"], baseline)
        row.update(baseline=evidence["baseline"]["value"], advantage=row["reward"] - evidence["baseline"]["value"])
    return row


def validate_single_reward(row: dict) -> dict:
    evidence = row.get("reward_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("single reward: original-T1 program-gate evidence required")
    decision = _single_reward_decision(row, evidence)
    if (evidence["decision"] != decision or type(row.get("reward")) is not int
            or row["reward"] != _reward_value(decision)):
        raise ValueError("single reward: reward/decision differs from the program gate on the original solution")
    if evidence["version"] in TABLE_GATES:
        value = single_advantage(decision, evidence["version"])
        if ("baseline" in evidence or "baseline" in row
                or evidence.get("advantage") != {"version": _advantage_version(evidence["version"]), "value": value}
                or type(row.get("advantage")) is not float or row["advantage"] != value):
            table = {SINGLE_REWARD_GATE_V9: "v14", SINGLE_REWARD_GATE_V8: "v13",
                     SINGLE_REWARD_GATE_V7: "v12"}.get(evidence["version"], "v11")
            raise ValueError(f"single reward: advantage differs from the {table} penalty table")
    elif "advantage" in evidence:
        raise ValueError("single reward: gate v5 evidence carries no penalty-table advantage")
    elif "baseline" in evidence or "baseline" in row or "advantage" in row:
        bound = evidence.get("baseline")
        pin = {key: bound.get(key) for key in ("method", "table_sha256")} if isinstance(bound, dict) else {}
        expected = single_source_baseline(row["problem_id"], pin | {"unknown_source": "global_mean"})
        if (bound != expected or type(row.get("baseline")) is not float or row["baseline"] != expected["value"]
                or type(row.get("advantage")) is not float or row["advantage"] != row["reward"] - expected["value"]):
            raise ValueError("single reward: RL baseline/advantage differs from the shipped source table")
    return decision


def note_cause_problem(note_text: str, t1_status: str) -> str | None:
    """v8: why a note may not be a B row because of its Cause (notes exist only for wrong answers, so NONE is never
    right; an unaccepted or cut-off answer is NO_FINAL_ANSWER). The note still drives its one re-solve."""
    parsed = parse_note_v2(note_text)
    if parsed is None:
        return "NOTE_FORMAT_INVALID"
    if parsed.cause in SINGLE_NOTE_CAUSE_RULE["reject_cause"]:
        return "NOTE_CAUSE_NONE"
    if (t1_status in SINGLE_NOTE_CAUSE_RULE["unaccepted_statuses"]
            and parsed.cause != SINGLE_NOTE_CAUSE_RULE["unaccepted_cause"]):
        return "NOTE_CAUSE_MISMATCH"
    return None


def note_program_decision(audit: dict, record: dict, *, leak_v2: bool = False) -> dict:
    """Program-only note admission for the re-solve: parsed note that stopped normally and passed every bounded leak
    check. Its Cause is checked only when it would become a B row (note_cause_problem), so a NONE note's lesson is
    still tried once (cycle-2 audit: rejecting it here removed almost every A2).

    ``leak_v2`` (v12): the lesson check is contract.hint_leak_single_v2 on the audit's own leak inputs instead of the
    audit's stored v1 result. The base hint_leak (NOTE_HINT_LEAK, MCQ only) needs no narrowing: it flags option-letter
    references, answer phrases and the gold letter/option text, never a non-gold option."""
    base = audit["base_audit"]
    parsed = parse_note_v2(record["note"]["text"]) if base["format"]["result"] == "PASS" else None
    if base["format"]["result"] != "PASS" or parsed is None:
        reason = "NOTE_FORMAT_INVALID"
    # Open items only: hint_leak_single re-runs the base hint_leak there without option letters. MCQ keeps the base
    # check, whose raw lesson/option comparison catches e.g. a gold option "x²" that the NFKC single check misses.
    elif (failed := next((key for key, value in base["checks"].items()
                          if value is not None and (key != "hint_leak" or base["leak_inputs"]["choices"])),
                         None)) is not None:
        reason = "NOTE_" + failed.upper()
    elif (leak := audit["single_hint_leak"] if not leak_v2 else hint_leak_single_v2(
            parsed.lesson, base["leak_inputs"]["gold"], base["leak_inputs"]["choices"],
            aliases=base["leak_inputs"]["aliases"])) is not None:
        reason = "LESSON_LEAK_" + leak
    else:
        reason = None
    return {"accepted": reason is None, "reason": reason, "method": "program", "audit_sha256": audit["sha256"]}


def human_row(record: dict, fields: dict[str, str], gold: str, choices: tuple[str, ...] | list[str],
              tokenizer: Any) -> tuple[dict | None, str | None]:
    """v12 H row: the P0 prompt and the dataset's human-written solution (contract.human_solution_target); admitted
    when the solution is nonblank, the target has >= 32 tokens and no special token. (row, None) or (None, reason)."""
    target = human_solution_target(fields["lecture"], fields["solution"], gold, choices)
    if not fields["solution"].strip():
        return None, "H_SOLUTION_BLANK"
    if target is None:
        return None, "H_GOLD_NOT_AN_OPTION"
    if any(token in target for token in tokenizer.all_special_tokens):
        return None, "H_SPECIAL_TOKEN"
    if len(tokenizer.encode(target, add_special_tokens=False)) < MIN_TARGET_TOKENS:
        return None, "H_TOO_SHORT"
    return {"row_id": record["problem_id"] + ":" + KIND_HUMAN, "kind": KIND_HUMAN, "route": ROUTE_LM,
            **{k: record[k] for k in ("problem_id", "image_paths", "image_sha256", "subject", "source")},
            "prompt": record["p0_text"], "target": target, "target_source": HUMAN_TARGET_SOURCE, "complete": True}, None


def run_single_block(bundle: Any, **kwargs) -> dict:
    """Serialize the complete generation/review/row transaction for one output block."""
    with _runtime_lock(Path(kwargs["output"]), ".runtime.lock"):
        from .protocol import VERSION_CONTENT_V16

        if kwargs["protocol"].get("version") == VERSION_CONTENT_V16:
            return _run_content_block_v16(bundle, **kwargs)
        return _run_single_block(bundle, **kwargs)


def _run_content_block_v16(bundle: Any, *, actors: dict, items: list[dict], provenance: dict, gold: Callable,
                           grade: Callable, load_images: Callable, protocol: dict, output: Path,
                           content_rows: list[dict], note_yield_records: list[dict] | None = None,
                           effect_gate: dict | None = None,
                           block_id: int = 0, limit: int | None = None,
                           stream: Callable | None = None, stream_args: argparse.Namespace | None = None,
                           stop_after: float | None = None, reference: Callable | None = None) -> dict:
    """Reuse the durable single-request ledger; authored O/J targets never enter a P0 generation."""
    from fine_tuning.notebook_rl.prompts import p0_eval_text, user_message
    from .rl import policy_sha256
    from .self_note_v16 import (build_note_prompt, build_problem_contract, build_resolve_prompt,
                               build_self_note_rows, create_note_yield_gate, first_solution_row, human_j_row, verify_note,
                               _WRONG as wrong_statuses)

    _protocol_check(protocol)
    ids = [item["problem_id"] for item in items]
    if not ids or len(ids) > 16 or len(set(ids)) != len(ids) or any(pid not in actors for pid in ids):
        raise ValueError("v16 runtime: a distinct source-bound parent block is required")
    limit = len(ids) if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= len(ids):
        raise ValueError("v16 runtime: prefix outside fixed parent block")
    active = ids[:limit]
    policy = policy_sha256(bundle)
    binding = {"version": "v16-content-block-v1", "block_id": block_id, "items": items,
               "protocol_sha256": protocol["sha256"], "policy_sha256": policy,
               "content_rows_sha256": digest(content_rows), "provenance_sha256": digest(provenance)}
    _sealed(output / "binding.json", binding)
    phase_path = output / "phase.json"
    state = _state(phase_path) if phase_path.exists() else {
        "version": "v16-content-block-v1", "binding_sha256": digest(binding), "records": {}, "limit": 0}
    if state["binding_sha256"] != digest(binding) or limit < state["limit"]:
        raise ValueError("v16 runtime: stale content policy, source or prefix")
    state["limit"] = limit
    from .protocol import content_contract_for_block

    active_contract = content_contract_for_block(protocol, block_id)
    weights = active_contract["weights"]
    generated = any(weights[kind] > 0 for kind in ("C", "N", "A1"))
    rows = [row for row in content_rows if row["problem_id"] in active and row["kind"] in ("O", "J")]
    diagnostics = {}
    yield_records = list(note_yield_records or [])

    def commit(status: str, **extra) -> dict:
        state.update(status=status, **extra)
        return _state(phase_path, state)

    if generated or weights["J"] > 0:
        from .run_v16 import validate_bf16_effect_receipt

        effect_result = None if effect_gate is None else validate_bf16_effect_receipt(effect_gate, expected_binding={
                "source_sha256": protocol["training_contract"]["hashes"]["source"],
                "dataset_sha256": protocol["training_contract"]["hashes"]["dataset"]})
        if effect_result is None or effect_result.get("passed") is not True:
            raise ValueError("v16 runtime: EFFECT_NOTRUN_OR_STOP; later content targets remain gated")
    if weights["J"] > 0:
        assert effect_gate is not None and effect_result is not None
        baseline = effect_result.get("baseline_correct_length_median")
        for item in items[:limit]:
            pid, actor = item["problem_id"], actors[item["problem_id"]]
            layer = next(row["layer"] for row in content_rows if row["problem_id"] == pid and row["kind"] == "O")
            record = item | {"problem_id": pid, "p0_text": p0_eval_text(actor.question, actor.choices),
                "layer": layer, "source": actor.source, "subject": item.get("subject", layer),
                "image_paths": list(actor.image_paths), "image_sha256": list(actor.image_sha256)}
            authored, reason = human_j_row(record, provenance[pid].get("human_solution", {}),
                tokenizer=bundle.processor.tokenizer, baseline_correct_length_median=baseline)
            diagnostics[pid + ":J"] = {"accepted": authored is not None, "reason": reason}
            if authored is not None:
                rows.append(authored)
    if generated:
        stats = StreamStats()
        if stream is None:
            args = argparse.Namespace(**vars(stream_args)) if stream_args is not None else argparse.Namespace()
            args.policy_sha256, args.protocol_sha256 = policy, protocol["sha256"]
            stream = (getattr(args, "stream_builder", None) or hf_single_stream_builder)(bundle, args, stats)
        for item in items[:limit]:
            pid, actor = item["problem_id"], actors[item["problem_id"]]
            metadata = {key: value for key, value in item.items() if key not in ("annotation", "gold")}
            layer = next(row["layer"] for row in content_rows if row["problem_id"] == pid and row["kind"] == "O")
            metadata |= {"problem_id": pid, "layer": layer,
                         "experiment_sha256": digest({"protocol": protocol["sha256"],
                             "dataset": protocol["training_contract"]["hashes"]["dataset"]}),
                         "content_manifest_sha256": protocol["training_contract"]["hashes"]["dataset"],
                         "protocol_sha256": protocol["sha256"], "actor_sha256": digest(vars(actor)),
                         "annotation_sha256": digest(provenance[pid]["annotation"]),
                         "p0_text": p0_eval_text(actor.question, actor.choices),
                         "image_paths": list(actor.image_paths), "image_sha256": list(actor.image_sha256),
                         "source": actor.source, "subject": item.get("subject", item.get("layer", "content"))}
            if pid not in state["records"]:
                state["records"][pid] = metadata
            elif any(state["records"][pid].get(key) != value for key, value in metadata.items()):
                raise ValueError("v16 runtime: original actor metadata changed")

        def generate(stage: str, prompts: dict[str, str]) -> bool:
            if not prompts:
                return True
            requests = [{"request_id": f"{pid}:{stage}", "problem_id": pid, "block_id": block_id, "stage": stage,
                "seed": sample_seed(pid, stage), "max_tokens": protocol["training_contract"]["max_tokens"][stage],
                "messages": [user_message(prompt, len(actors[pid].image_paths))],
                "images": load_images(actors[pid])} for pid, prompt in prompts.items()]
            results = run_single_requests(requests, stream, stats, output, policy_sha256=policy,
                                          protocol_sha256=protocol["sha256"], stop_after=stop_after)
            for pid, prompt in prompts.items():
                result = results.get(f"{pid}:{stage}")
                if result is None:
                    continue
                result = dict(result) | {"prompt": prompt}
                if stage != "note":
                    status, extracted = grade(pid, result["text"], result["finish_reason"], actors[pid].mode)
                    if status not in GRADED | {UNVERIFIABLE}:
                        raise ValueError("v16 runtime: a system grading failure blocks collection")
                    result |= {"status": status, "extracted": extracted}
                previous = state["records"][pid].get(stage)
                if previous is not None and previous != result:
                    raise ValueError("v16 runtime: completed action changed")
                state["records"][pid][stage] = result
                commit("COLLECTING_" + stage.upper())
            return all(stage in state["records"][pid] for pid in prompts)

        if not generate("t1", {pid: state["records"][pid]["p0_text"] for pid in active}):
            return commit("INCOMPLETE_T1")
        for pid in active if weights["A1"] > 0 else ():
            accepted, reason = first_solution_row(state["records"][pid], tokenizer=bundle.processor.tokenizer,
                                                   choices=actors[pid].choices)
            diagnostics[pid + ":A1"] = {"accepted": accepted is not None, "reason": reason}
            if accepted is not None:
                rows.append(accepted)
        wrong = [pid for pid in active if state["records"][pid]["t1"]["status"] in wrong_statuses
                 and (weights["C"] > 0 or weights["N"] > 0)]
        contracts = {pid: build_problem_contract(provenance[pid] | {"layer": state["records"][pid]["layer"]})
                     for pid in wrong}
        supported = [pid for pid in wrong if contracts[pid].get("supported") is True]
        for pid in set(wrong) - set(supported):
            diagnostics[pid] = {"reason": "UNSUPPORTED_SOURCE_CONTRACT", "accepted": False}
        prompts = {pid: build_note_prompt(state["records"][pid], fact_catalog=contracts[pid]["fact_catalog"],
                    rules=contracts[pid]["rules"], gold=gold(pid)) for pid in supported}
        if not generate("note", prompts):
            return commit("INCOMPLETE_NOTE")
        audits = {pid: verify_note(state["records"][pid], fact_catalog=contracts[pid].get("fact_catalog", []),
                    rules=contracts[pid].get("rules", {}), choices=actors[pid].choices, gold=gold(pid)) for pid in wrong}
        approved = [pid for pid in supported if audits[pid]["accepted"]]
        if not generate("resolve", {pid: build_resolve_prompt(state["records"][pid]["p0_text"], audits[pid]["lesson"])
                                     for pid in approved}):
            return commit("INCOMPLETE_RESOLVE")
        for pid in wrong[:max(0, 32 - len(yield_records))]:
            yield_records.append(state["records"][pid] | {"note_receipt": audits[pid],
                "note_validation": {"fact_catalog": contracts[pid].get("fact_catalog", []),
                    "rules": contracts[pid].get("rules", {}), "choices": list(actors[pid].choices), "gold": gold(pid)}})
        note_gate = create_note_yield_gate(yield_records)
        for pid in supported:
            accepted, decisions = build_self_note_rows(state["records"][pid], tokenizer=bundle.processor.tokenizer,
                fact_catalog=contracts[pid]["fact_catalog"], rules=contracts[pid]["rules"],
                choices=actors[pid].choices, gold=gold(pid), note_yield_gate=note_gate)
            rows.extend(accepted)
            diagnostics[pid] = decisions
        state["note_yield_gate"] = note_gate
    if policy_sha256(bundle) != policy:
        raise ValueError("v16 runtime: policy changed during collection")
    contract = active_contract | {"expected_problem_ids": active, "policy_sha256": policy}
    result = {"version": "v16-content-block-v1", "status": "ROWS_READY", "rows": rows,
              "rows_sha256": digest(rows), "contract": contract, "row_decisions": diagnostics,
              "note_yield_records": yield_records}
    path = output / f"rows-{limit}.json"
    _sealed(path, result)
    return commit("ROWS_READY", rows_path=path.name, row_decisions=diagnostics,
                  note_yield_records=yield_records)


def _run_single_block(bundle: Any, *, actors: dict, items: list[dict], provenance: dict, gold: Callable[[str], str],
                     grade: Callable, load_images: Callable, protocol: dict, output: Path, block_id: int = 0,
                     limit: int | None = None, stream: Callable | None = None,
                     stream_args: argparse.Namespace | None = None, stop_after: float | None = None,
                     reference: Callable[[str], str | None] | None = None) -> dict:
    """Collect at most 16 distinct items; resume completed actions; admit rows by program checks only.

    ``items`` is the immutable whole block, even when limit first requests its prefix of two.
    Gold is requested only while constructing the note/audit stage and (v12) the H targets after all generation.
    It never reaches T1/retry requests.
    ``reference`` (v11) gives the dataset reference text for a problem's note prompt only (None: no reference).
    v14: an A2 row needs at least 256 sampled tokens (reason A2_UNDER_MIN_TOKENS), and a note is a B candidate only
    when its own re-solve passed the v11 A2 checks (decided before the 256-token rule).
    v15: up to three re-solves with the same prompt (records ``resolve``, ``resolve2``, ``resolve3``); attempt k+1 runs
    only when attempt k failed _a2_check; the first passing attempt is the only A2 row, and only it brings the B row.
    v12 H rows read the human-solution fields from ``provenance`` (references.human_solution) and count tokens with
    ``bundle.processor.tokenizer``.
    """
    from fine_tuning.notebook_rl.prompts import p0_eval_text, user_message
    from .rl import policy_sha256

    _protocol_check(protocol)
    v15 = protocol["version"] == VERSION_SINGLE_V15  # every row of an image item takes the FULL route
    v14 = v15 or protocol["version"] == VERSION_SINGLE_V14  # v15 keeps every v14 rule
    v13 = v14 or protocol["version"] == VERSION_SINGLE_V13  # v14 keeps every v13 rule
    v12 = v13 or protocol["version"] == VERSION_SINGLE_V12  # v13 keeps v12's note filter, B rule and item evidence
    # v15: attempt k+1 (same prompt, own seed) exists only when attempts 1..k failed _a2_check; v11-v14: one re-solve.
    stages = SINGLE_RESOLVE_STAGES_V15 if v15 else ("resolve",)
    ids =[item["problem_id"] for item in items]
    if not 1 <= len(ids) <= 16 or len(set(ids)) != len(ids) or any(pid not in actors or pid not in provenance for pid in ids):
        raise ValueError("single runtime: unique, source-bound block of 1..16 actors required")
    limit = len(ids) if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= len(ids):
        raise ValueError("single runtime: requested prefix outside the fixed block")
    if v13 and [bundle.processor.tokenizer.decode([token]).strip() for token in SINGLE_SPIRAL_V13["marker_token_ids"]] != (
            [SINGLE_SPIRAL_V13["marker"]] * len(SINGLE_SPIRAL_V13["marker_token_ids"])):
        raise ValueError("single runtime: the tokenizer's spiral marker tokens differ from the sealed v13 IDs")
    policy = policy_sha256(bundle)
    binding = {"version": VERSION, "block_id": block_id, "items": items, "protocol_sha256": protocol["sha256"],
               "policy_sha256": policy, "actors_sha256": digest({pid: vars(actors[pid]) for pid in ids}),
               "provenance_sha256": digest({pid: provenance[pid] for pid in ids})}
    output.mkdir(parents=True, exist_ok=True)
    _sealed(output / "binding.json", binding)
    state_path = output / "phase.json"
    state = _state(state_path) if state_path.exists() else {"version": VERSION, "binding_sha256": digest(binding),
                                                          "records": {}, "limit": 0}
    if state["binding_sha256"] != digest(binding) or limit < state["limit"]:
        raise ValueError("single runtime: stale policy/binding or prefix moved backwards")
    state["limit"] = limit
    active = ids[:limit]
    stats = StreamStats()
    if stream is None:
        args = argparse.Namespace(**vars(stream_args)) if stream_args is not None else argparse.Namespace()
        args.policy_sha256, args.protocol_sha256 = policy, protocol["sha256"]
        stream = (getattr(args, "stream_builder", None) or hf_single_stream_builder)(bundle, args, stats)

    def checkpoint(status: str, **extra) -> dict:
        # Hashing all 624 policy tensors per completion cost minutes per block; the policy is re-checked once
        # after generation (below) and again by the update's sealed contract.
        state.update(status=status, **extra)
        return _state(state_path, state)

    def lesson(pid: str) -> str:
        parsed = parse_note_v2(state["records"][pid]["note"]["text"])
        if parsed is None:
            raise ValueError("single runtime: accepted note is no longer parseable")
        return parsed.lesson

    def request(pid: str, stage: str, prompt: str) -> dict:
        return {"request_id": f"{pid}:{stage}", "problem_id": pid, "block_id": block_id, "stage": stage,
                "seed": sample_seed(pid, "t1" if stage == "t1" else stage),
                "max_tokens": SINGLE_NOTE_MAX_TOKENS if stage == "note" else MAX_NEW_TOKENS,
                "messages": [user_message(prompt, len(actors[pid].image_paths))], "images": load_images(actors[pid]),
                "prompt": prompt}

    def accept(stage: str, pid: str, output_: dict, prompt: str) -> None:
        result = dict(output_) | {"prompt": prompt}
        if stage != "note":
            status, extracted = grade(pid, result["text"], result["finish_reason"], actors[pid].mode)
            if status not in GRADED | {UNVERIFIABLE}:
                raise ValueError(f"single runtime: ungraded/system outcome blocks the block: {status}")
            result.update(status=status, extracted=extracted)
        if stage in SINGLE_RESOLVE_STAGES_V15:
            result["lesson_sha256"] = text_sha256(lesson(pid))
        previous = state["records"][pid].get(stage)
        if previous is not None and previous != result:
            raise ValueError("single runtime: completed generation/grade changed")
        state["records"][pid][stage] = result
        checkpoint("COLLECTING_" + stage.upper())

    def note_prompt(pid: str) -> str:
        t1 = state["records"][pid]["t1"]
        return note_prompt_single(state["records"][pid]["p0_text"], t1["text"], t1["status"], gold(pid),
                                  read=t1.get("extracted"), reference=reference(pid) if reference is not None else None)

    def note_decision(pid: str) -> tuple[dict, dict]:
        record, source = state["records"][pid], provenance[pid]
        audit = audit_single_note(record, gold=gold(pid), choices=actors[pid].choices, provenance=source,
                                  aliases=source.get("leak_aliases", ()),
                                  acronym_candidates=source.get("acronym_candidates", ()))
        validate_single_note_audit(audit, record, source)
        return audit, note_program_decision(audit, record, leak_v2=v12)

    def is_wrong(pid: str) -> bool:
        return state["records"][pid]["t1"]["status"] not in ("CORRECT", UNVERIFIABLE)

    def generate(stage: str, prompts: dict[str, str]) -> bool:
        requests = [request(pid, stage, prompt) for pid, prompt in prompts.items()]
        if not requests:
            return True
        results = run_single_requests([{k: v for k, v in r.items() if k != "prompt"} for r in requests], stream, stats,
                                      output, policy_sha256=policy, protocol_sha256=protocol["sha256"], stop_after=stop_after)
        for item in requests:
            if item["request_id"] in results:
                accept(stage, item["problem_id"], results[item["request_id"]], item["prompt"])
        return all(stage in state["records"][pid] for pid in prompts)

    def pipeline() -> None:
        """Continuous engines: a question's note/re-solve starts as soon as its own previous stage ends."""
        prompts: dict[str, str] = {}

        def submit(item: dict) -> dict:
            prompts[item["request_id"]] = item.pop("prompt")
            return item

        def follow(item: dict, result: dict) -> list[dict]:
            pid, stage = item["problem_id"], item["stage"]
            accept(stage, pid, result, prompts[item["request_id"]])
            if stage == "t1" and is_wrong(pid):
                return [submit(request(pid, "note", note_prompt(pid)))]
            if stage == "note" and note_decision(pid)[1]["accepted"]:
                return [submit(request(pid, "resolve", resolve_prompt_single(state["records"][pid]["p0_text"], lesson(pid))))]
            if stage in stages[:-1] and _a2_check(state["records"][pid], stage, lesson(pid), image_route=v15)[0] is None:
                return [submit(request(pid, stages[stages.index(stage) + 1],
                                       resolve_prompt_single(state["records"][pid]["p0_text"], lesson(pid))))]
            return []

        run_single_requests([submit(request(pid, "t1", state["records"][pid]["p0_text"])) for pid in active], stream,
                            stats, output, policy_sha256=policy, protocol_sha256=protocol["sha256"],
                            stop_after=stop_after, follow_up=follow)

    for item in items[:limit]:
        pid, actor = item["problem_id"], actors[item["problem_id"]]
        metadata = {"problem_id": pid, "p0_text": p0_eval_text(actor.question, actor.choices),
                    "image_paths": list(actor.image_paths), "image_sha256": list(actor.image_sha256),
                    "subject": item["subject"], "source": actor.source}
        if pid not in state["records"]:
            state["records"][pid] = metadata
        elif any(state["records"][pid][k] != value for k, value in metadata.items()):
            raise ValueError("single runtime: actor metadata changed")
    if getattr(stream, "continuous", False):
        pipeline()  # The staged pass below then reuses every completion from the ledger without generating.
    if not generate("t1", {pid: state["records"][pid]["p0_text"] for pid in active}):
        return checkpoint("INCOMPLETE_T1")
    wrong = [pid for pid in active if is_wrong(pid)]
    if not generate("note", {pid: note_prompt(pid) for pid in wrong}):
        return checkpoint("INCOMPLETE_NOTE")
    audits, note_decisions = {}, {}
    for pid in wrong:
        audits[pid], note_decisions[pid] = note_decision(pid)
    approved = [pid for pid in wrong if note_decisions[pid]["accepted"]]
    pending = approved
    for stage in stages:
        if not generate(stage, {pid: resolve_prompt_single(state["records"][pid]["p0_text"], lesson(pid))
                                for pid in pending}):
            return checkpoint("INCOMPLETE_RESOLVE", note_audits=audits, note_decisions=note_decisions)
        pending = [pid for pid in pending
                   if _a2_check(state["records"][pid], stage, lesson(pid), image_route=v15)[0] is None]
    if policy_sha256(bundle) != policy:
        raise ValueError("single runtime: policy changed during collection")
    rows, decisions = [], {}
    for pid in active:
        record = state["records"][pid]
        rl_row = _candidate(record, "RL", image_route=v15)
        if rl_row is None:
            raise ValueError("single runtime: original T1 is missing")
        gate = {"gate": PROTOCOL_GATES[protocol["version"]], "option_count": len(actors[pid].choices)} if v12 else {}
        rows.append(bind_single_reward(rl_row, record["t1"], provenance[pid], **gate))
        praised = not v13 or rows[-1]["reward_evidence"]["decision"]["praise"]
        kind = "A1" if record["t1"]["status"] == "CORRECT" else "A2"
        if kind == "A2" and pid not in approved:
            continue
        if v15 and kind == "A2":  # the first attempt that passes every v14 A2 check is the only A2 row
            tried = [stage for stage in stages if stage in record]
            row = reason = attempt = None
            for number, stage in enumerate(tried, 1):
                row, reason = _a2_check(record, stage, lesson(pid), image_route=v15)
                if row is not None:
                    attempt = number
                    break
            decisions[f"{pid}:A2"] = {"accepted": row is not None, "method": "program", "reason": reason,
                                      "attempt": attempt, "attempts": len(tried)}
            if row is None:
                continue  # v15: B follows a passing A2 attempt only; failed attempts are never rows
            rows.append(row)
        else:
            row = _candidate(record, kind, image_route=v15)
            reason = None if row is not None else f"{kind}_NOT_ADMISSIBLE"
            if row is not None and kind == "A2" and (problem := a2_target_problem(row["target"], lesson(pid))):
                row, reason = None, "A2_" + problem
            elif row is not None and kind == "A1" and (phrase := answer_denial(row["target"])) is not None:
                row, reason = None, "A1_ANSWER_DENIAL:" + phrase
            elif row is not None and kind == "A1" and not praised:
                row, reason = None, "A1_NOT_PRAISED"  # v13: a correct answer under 256 tokens is not copied either
            follows = row is not None and kind == "A2"  # v14: the re-solve passed every v11 A2 check
            if v14 and follows and row is not None and len(row["generation"]["completion_token_ids"]) < SINGLE_A2_MIN_TOKENS_V14:
                row, reason = None, "A2_UNDER_MIN_TOKENS"
            decisions[f"{pid}:{kind}"] = {"accepted": row is not None, "method": "program", "reason": reason}
            if row is not None:
                rows.append(row)
            elif kind == "A1" or not v12 or (v14 and not follows):
                continue  # v11: B follows its A2; v12/v13: every accepted note is a B candidate; v14: B follows a passing A2
        if kind == "A2" and (note_row := _candidate(record, "B", image_route=v15)) is not None:
            problem = (b_target_problem(note_row["target"], v12=v12)
                       or note_cause_problem(note_row["target"], record["t1"]["status"]))
            decisions[f"{pid}:B"] = {"accepted": problem is None, "method": "program",
                                     "reason": None if problem is None else "B_" + problem}
            if problem is None:
                rows.append(note_row)
    for pid in active if v12 and not v13 else ():
        fields = references.human_solution(provenance[pid])
        if fields is not None:
            row, reason = human_row(state["records"][pid], fields, gold(pid), actors[pid].choices,
                                    bundle.processor.tokenizer)
            decisions[f"{pid}:{KIND_HUMAN}"] = {"accepted": row is not None, "method": "program", "reason": reason}
            if row is not None:
                rows.append(row)
    contract = protocol["training_contract"] | {"expected_problem_ids": active, "policy_sha256": policy,
        "reward_provenance_sha256": {pid: digest(provenance[pid]) for pid in active},
        "reward_evidence_sha256": {row["problem_id"]: digest(row["reward_evidence"]) for row in rows if row["kind"] == "RL"}}
    result = {"version": VERSION, "status": "ROWS_READY", "binding_sha256": digest(binding), "limit": limit,
              "rows": rows, "rows_sha256": digest(rows), "contract": contract, "note_audits": audits,
              "note_decisions": note_decisions, "row_decisions": decisions}
    write_json(output / f"rows-{limit}.json", result)
    return checkpoint("ROWS_READY", note_audits=audits, note_decisions=note_decisions, row_decisions=decisions,
                      rows_path=f"rows-{limit}.json")


def run_no_lesson_control(bundle: Any, *, block: Path, actors: dict, grade: Callable, load_images: Callable,
                          protocol: dict, output: Path, stream: Callable | None = None,
                          stream_args: argparse.Namespace | None = None, stop_after: float | None = None) -> dict:
    """Diagnostic only (user 2026-11-11; local CLI ``--no-lesson-control``, never the campaign): after a block's rows
    are sealed, one lesson-free re-solve (RESOLVE_INSTRUCTION_SINGLE + P0, seed tag "control") per item that got a
    lesson re-solve, graded like it. Results go to ``output/NO_LESSON_CONTROL_FILE``; the generations have their own
    ledger directory and never become rows, rewards or updates."""
    from fine_tuning.notebook_rl.prompts import user_message
    from .rl import policy_sha256

    _protocol_check(protocol)
    state, binding = _state(block / "phase.json"), _read(block / "binding.json")
    policy = policy_sha256(bundle)
    if (state.get("status") != "ROWS_READY" or state["binding_sha256"] != digest(binding)
            or binding["protocol_sha256"] != protocol["sha256"] or binding["policy_sha256"] != policy):
        raise ValueError("single control: needs the sealed rows of this block under the same policy and protocol")
    records = state["records"]
    ids = [item["problem_id"] for item in binding["items"][:state["limit"]] if "resolve" in records[item["problem_id"]]]
    requests = []
    for pid in ids:
        prompt = f"{RESOLVE_INSTRUCTION_SINGLE}\n\n{records[pid]['p0_text']}"
        requests.append({"request_id": f"{pid}:control", "problem_id": pid, "block_id": binding["block_id"],
                         "stage": "control", "seed": sample_seed(pid, "control"), "max_tokens": MAX_NEW_TOKENS,
                         "messages": [user_message(prompt, len(actors[pid].image_paths))],
                         "images": load_images(actors[pid])})
    stats = StreamStats()
    if stream is None:
        args = argparse.Namespace(**vars(stream_args)) if stream_args is not None else argparse.Namespace()
        args.policy_sha256, args.protocol_sha256 = policy, protocol["sha256"]
        stream = (getattr(args, "stream_builder", None) or hf_single_stream_builder)(bundle, args, stats)
    results = run_single_requests(requests, stream, stats, output / "no-lesson-control", policy_sha256=policy,
                                  protocol_sha256=protocol["sha256"], stop_after=stop_after) if requests else {}
    entries = []
    for pid in ids:
        record, result = records[pid], results.get(f"{pid}:control")
        entry = {"problem_id": pid, "source": record["source"], "t1_status": record["t1"]["status"],
                 "lesson_resolve_status": record["resolve"]["status"], "control_status": None, "tokens": None}
        if result is not None:
            status, _ = grade(pid, result["text"], result["finish_reason"], actors[pid].mode)
            entry.update(control_status=status, tokens=len(result["generation"]["completion_token_ids"]),
                         control_finish_reason=result["finish_reason"])
        entries.append(entry)
    body = {"version": NO_LESSON_CONTROL_VERSION, "diagnostic_only": True, "training_rows": 0,
            "status": "COMPLETE" if all(e["control_status"] is not None for e in entries) else "INCOMPLETE",
            "block_binding_sha256": state["binding_sha256"], "rows_path": state["rows_path"],
            "prompt": "RESOLVE_INSTRUCTION_SINGLE + blank line + P0 (no lesson header, no lesson)",
            "seed_stage": "control", "items": entries}
    write_json(output / NO_LESSON_CONTROL_FILE, body)
    return body


def prove_single_backward(bundle: Any, encoded: list[dict], contract: dict, *, precision: str,
                          token_budget: int = 0, capacity: Any = None) -> dict:
    """Independent component gradients; no optimizer, scheduler, or model checkpoint exists here."""
    from .train import _all_parameter_hashes, _gradient_norms
    from .train_v4 import backward_single_block
    from fine_tuning.notebook_rl.policy import assert_base_unchanged
    from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged

    check_base = assert_quantized_base_unchanged if precision == "nf4" else assert_base_unchanged
    before = _all_parameter_hashes(bundle)
    report: dict[str, Any] = {"status": "HOLD", "optimizer_steps": 0, "scheduler_steps": 0, "weights_saved": False,
              "rows_by_kind": dict(Counter(row["kind"] for row in encoded)), "components": {}, "missing": []}
    try:
        check_base(bundle)
        from .protocol import VERSION_CONTENT_V16

        content = contract.get("version") == VERSION_CONTENT_V16
        components = tuple(kind for kind in ("O", "J", "C", "N", "A1") if any(r["kind"] == kind for r in encoded)) if content else (
            COMPONENTS + ((KIND_HUMAN,) if any(r["kind"] == KIND_HUMAN for r in encoded) else ()))
        for name in components:
            started = time.monotonic()
            bundle.model.zero_grad(set_to_none=True)
            stats = backward_single_block(bundle, encoded, contract, token_budget, components=(name,), capacity=capacity)
            norms = _gradient_norms(bundle)
            rows = [r for r in encoded if r["kind"] == ("RL" if name in ("PG", "KL") else name)]
            signal = bool(stats["has_gradient"])
            full = any(row["route"] == ROUTE_FULL for row in rows)  # v11-v14: only A2/B rows can be FULL
            image = any(row.get("image_paths") or row.get("has_image") or row.get("pixel_values") is not None
                        or "pixel_values" in row.get("inputs", {})
                        for row in rows) if content else True
            expected = {"L": signal, "M": signal and image, "V": signal and full and image}
            positive = {group: value is not None and value > 0 for group, value in norms.items()}
            valid = all(positive.get(group, False) == need for group, need in expected.items())
            report["components"][name] = {"stats": stats, "gradient_norms": norms, "route_valid": valid,
                                           "zero_F0_KL_allowed": name == "KL" and not signal,
                                           "seconds": time.monotonic() - started}
            if name != "KL" and (not rows or not signal):
                report["missing"].append(name)
            if not valid:
                report["missing"].append(name + ":GRADIENT_ROUTE")
        a2 = {r["problem_id"] for r in encoded if r["kind"] == "A2"}
        notes = {r["problem_id"] for r in encoded if r["kind"] == "B"}
        incorrect = {r["problem_id"] for r in encoded if r["kind"] == "RL" and r["status"] == "INCORRECT"}
        report["paired_incorrect_a2_b"] = sorted(a2 & notes & incorrect)
        if not content and not report["paired_incorrect_a2_b"]:
            report["missing"].append("SAME_PROBLEM_INCORRECT_A2_B")
        if content and not components:
            report["missing"].append("NO_CONTENT_ROWS")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["missing"].append("BACKWARD_FAILED")
    finally:
        bundle.model.zero_grad(set_to_none=True)
        check_base(bundle)
        report["all_parameters_unchanged"] = before == _all_parameter_hashes(bundle)
        report["base_unchanged"] = True
        report["gradients_cleared"] = all(p.grad is None for p in bundle.model.parameters())
    report["status"] = ("FAIL" if report.get("error") or not report["all_parameters_unchanged"] else
                        "HOLD" if report["missing"] else "PASS")
    return report


def run_joint_single_update(bundle: Any, optimizer: Any, scheduler: Any, encoded: list[dict], contract: dict, *,
                            cursor: dict, block_ids: list[str], allow_update: bool = False, token_budget: int = 0,
                            capacity: Any = None) -> dict:
    """Consume one sealed schedule slot, retaining the optimizer across preparation and RL."""
    import torch
    from .rl import policy_sha256
    from .train_v4 import backward_single_block
    from .protocol import VERSION_CONTENT_V16, VERSION_V16B

    if allow_update is not True:
        raise ValueError("single runtime: parameter updates require explicit caller opt-in")
    slot = cursor["slot"]
    is_v16b = contract.get("version") in (VERSION_V16B, "error-note-v17", "error-note-v18", "error-note-v18b")
    prep_rows: list[str] = []
    if contract.get("version") == VERSION_CONTENT_V16:
        stage_a = slot < contract["stage_a_blocks"]
        expected_weights = contract["stage_a_weights"] if stage_a else contract["stage_b_weights"]
        if (contract.get("block_id") != slot or contract.get("active_stage") != ("A" if stage_a else "B")
                or contract["weights"] != expected_weights):
            raise ValueError("v16 runtime: block weights differ from its sealed stage schedule")
    if (type(slot) is not int or not 0 <= slot < contract["total_slots"]
            or cursor["policy_sha256"] != contract["policy_sha256"]
            or cursor["policy_sha256"] != policy_sha256(bundle)
            or contract["expected_problem_ids"] != block_ids):
        raise ValueError("single runtime: stale or invalid continuation cursor")
    if is_v16b:
        prep = slot < contract["prep_updates"]
        block = slot if prep else slot - contract["prep_updates"]
        if (contract.get("global_slot") != slot or contract.get("block_id") != block
                or contract.get("phase") != ("prep" if prep else "rl")
                or len(block_ids) != len(set(block_ids))):
            raise ValueError("v16b runtime: phase or global schedule differs")
        if prep:
            prep_rows = [f'{row["kind"]}:{row["problem_id"]}' for row in encoded]
            if (not prep_rows or len(prep_rows) != len(set(prep_rows))
                    or set(prep_rows) & set(cursor.get("consumed_prep_rows", []))):
                raise ValueError("v16b runtime: repeated preparation row")
        elif (set(block_ids) & set(cursor["consumed_problem_ids"])
                or (block >= contract["rl_blocks"]) != (not block_ids)):
            raise ValueError("v16b runtime: duplicate RL parent or invalid flush")
    elif (cursor["consumed_problem_ids"][-len(block_ids):] == block_ids
            or set(block_ids) & set(cursor["consumed_problem_ids"])):
        raise ValueError("single runtime: duplicate continuation cursor")
    optimizer.zero_grad(set_to_none=True)
    note_gradients: dict = {}
    is_v18 = contract.get("version") in ("error-note-v18", "error-note-v18b")
    if is_v18 and set(row["row_id"] for row in encoded) & set(cursor.get("consumed_row_ids", [])):
        raise ValueError("v18 runtime: repeated consumed supervised or policy row")
    try:
        if capacity is not None:
            capacity.optimizer = optimizer
        if is_v16b and contract.get("external_hold"):
            stats = {"has_gradient": False, "task_signal": False, "hold": True,
                     "hold_reason": contract["external_hold"]}
            if is_v18:
                stats["consumed_row_ids"] = []
        else:
            if is_v18:
                stats = backward_single_block(bundle, encoded, contract, token_budget, capacity=capacity,
                                              note_gradient_capture=note_gradients)
            else:
                stats = backward_single_block(bundle, encoded, contract, token_budget, capacity=capacity)
        step = bool(stats["has_gradient"])
        if contract.get("version") in (VERSION_CONTENT_V16, VERSION_V16B, "error-note-v17", "error-note-v18", "error-note-v18b"):
            step = step and stats.get("task_signal") is True
        if is_v16b:
            step = step and not bool(stats.get("hold") or stats.get("HOLD"))
            stats["update_gradient_norms"] = {group: None for group in bundle.params}
        if is_v18:
            from .train_v18 import note_step_after_v18, note_step_before_v18

            snapshots, alignment = note_step_before_v18(bundle, note_gradients, stats)
        if step:
            for group, parameters in bundle.params.items():
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                if not bool(torch.isfinite(norm)):
                    raise ValueError("single runtime: nonfinite gradient")
                if is_v16b:
                    stats["update_gradient_norms"][group] = float(norm.detach())
            optimizer.step()
        if is_v18:
            stats["note_update_alignment"] = note_step_after_v18(bundle, note_gradients, snapshots, alignment, step=step)
            stats["optimizer_step"] = step
        scheduler.step()
    finally:
        optimizer.zero_grad(set_to_none=True)
    consumed = cursor["consumed_problem_ids"] + ([] if prep_rows else block_ids)
    cursor.update(slot=slot + 1, updates=cursor["updates"] + int(step),
                  consumed_problem_ids=consumed, policy_sha256=policy_sha256(bundle))
    if is_v18:
        cursor["consumed_row_ids"] = cursor.get("consumed_row_ids", []) + stats["consumed_row_ids"]
    if is_v16b:
        cursor["consumed_prep_rows"] = cursor.get("consumed_prep_rows", []) + prep_rows
        next_rl = max(0, slot + 1 - contract["prep_updates"])
        cursor.update(phase="prep" if slot + 1 < contract["prep_updates"] else "rl",
                      rl_block=next_rl, policy_version=next_rl,
                      flush=next_rl >= contract["rl_blocks"])
    return {"stats": stats, "cursor": cursor, "optimizer_steps": int(step), "scheduled_slots": 1}


def save_single_continuation(bundle: Any, optimizer: Any, scheduler: Any, *, cursor: dict,
                             path: Path, protocol: dict, binding: dict, allow_save: bool = False) -> None:
    """Use the existing atomic checkpoint format; the caller must separately authorize real weight saving."""
    import os
    from fine_tuning.notebook_rl import checkpoint

    _protocol_check(protocol)
    if allow_save is not True or path.exists() or path.with_name(path.name + ".tmp").exists():
        raise ValueError("single runtime: explicit save opt-in and unused checkpoint paths required")
    bundle.route.apply(True)
    temporary = path.with_name(path.name + ".tmp")
    checkpoint.save(bundle, optimizer, scheduler, cursor["slot"],
                    {"epoch": 0, "position": cursor["slot"], "discarded_problem_ids": [], "single": cursor},
                    checkpoint.capture_rng(), temporary, protocol=protocol, binding=binding)
    os.replace(temporary, path)


def load_single_continuation(model_path: Path, path: Path, protocol: dict, *, binding: dict,
                             device: str = "cpu", policy_factory: Callable | None = None) -> tuple:
    """Restore the existing BF16/CPU checkpoint format, including moments, LR, RNG and single-use cursor."""
    from fine_tuning.notebook_rl import checkpoint
    from .rl import policy_sha256

    _protocol_check(protocol)
    from .protocol import VERSION_V16B

    if protocol["precision"] != "bf16" and not (
            protocol["version"] in (VERSION_V16B, "error-note-v17", "error-note-v18", "error-note-v18b")
            and protocol["training_contract"]["local_smoke"] is True
            and policy_factory is not None):
        raise ValueError("single runtime: NF4 reload requires the bounded v16b smoke and an explicit cache factory")
    bundle, optimizer, scheduler, slot, saved = checkpoint.load_resume(
        model_path, path, protocol, binding=binding, device=device, policy_factory=policy_factory)
    cursor = saved.get("single", {})
    if (cursor.get("slot") != slot or saved["position"] != slot
            or not 0 <= slot <= protocol["training_contract"]["total_slots"]
            or cursor.get("policy_sha256") != policy_sha256(bundle)):
        raise ValueError("single runtime: continuation policy/slot mismatch")
    return bundle, optimizer, scheduler, cursor


def _validate_data(args: argparse.Namespace) -> tuple[dict, dict]:
    from fine_tuning.notebook_rl.local_run import frozen_approval_output
    from .frozen_data import verify_frozen_data
    from .order import read_order

    approval = _read(args.frozen_approval)
    order = read_order(args.order, expected_sha256=args.order_sha256)
    if args.mode == "frozen3500":
        checked = verify_frozen_data(args.build_root, approval, args.order, order_sha256=args.order_sha256)
    else:
        selected, _, checked = frozen_approval_output(args.build_root, approval)
        if len(order["items"]) < 10 or any(i["problem_id"] not in selected["selected_ids"] for i in order["items"]):
            raise ValueError("single runtime: local smoke order is not an approved training selection")
        checked = checked | {"scope": "old approved pool local fixed10 only; not 3500-ready"}
    return order, checked


def prepare(args: argparse.Namespace) -> dict:
    """Serialize CPU preparation, including immutable input/source/capacity binding."""
    with _runtime_lock(args.output, ".prepare.lock"):
        return _prepare(args)


def _prepare(args: argparse.Namespace) -> dict:
    """CPU-only freeze. Revalidate source, data and model-file identity on every invocation."""
    import math

    measured = getattr(args, "capacity_profile", None)
    measured_sha = getattr(args, "capacity_profile_sha256", None)
    capture_path = getattr(args, "capture_capacity", None)
    reserve = getattr(args, "capacity_reserve_bytes", None)
    if (measured is None) != (measured_sha is None):
        raise ValueError("single runtime: capacity profile and externally pinned SHA256 are required together")
    if (capture_path is None) != (reserve is None) or capture_path is not None and (
            measured is not None or type(reserve) is not int or reserve <= 0 or capture_path.exists()):
        raise ValueError("single runtime: capture requires a fresh output, explicit positive reserve, and no admission profile")
    for name, explicit_default, measured_default in (("max_num_seqs", 1, 16),
            ("max_num_batched_tokens", CONTEXT_LIMIT, 16 * CONTEXT_LIMIT), ("token_budget", 0, 16 * CONTEXT_LIMIT)):
        if getattr(args, name, None) is None:
            setattr(args, name, measured_default if measured is not None else explicit_default)
    precision = getattr(args, "precision", "nf4")
    if precision not in ("nf4", "bf16"):
        raise ValueError("single runtime: explicit nf4 or bf16 precision required")

    if (type(args.max_num_seqs) is not int or not 1 <= args.max_num_seqs <= 16
            or type(args.max_num_batched_tokens) is not int or args.max_num_batched_tokens < 1
            or type(args.token_budget) is not int or args.token_budget < 0
            or args.stop_after_seconds is not None and (not math.isfinite(args.stop_after_seconds) or args.stop_after_seconds <= 0)):
        raise ValueError("single runtime: explicit positive generation capacity/deadline and nonnegative training budget required")
    order, admission = _validate_data(args)
    from fine_tuning.notebook_rl.schema import to_json
    from fine_tuning.notebook_rl.vault import read_actors
    from .answer_contract import VERSION as RULES_VERSION
    from .source_evidence import VERSION as SOURCE_VERSION

    document = _read(args.source_provenance)
    sources = document["items"]
    actors = {actor.problem_id: actor for actor in read_actors(args.build_root / "train")}
    train_sha = file_sha(args.build_root / "train" / "manifest.yaml")
    rules = _read(args.training_rules)
    if (document.get("version") != SOURCE_VERSION or rules.get("version") != RULES_VERSION
            or rules.get("vault_manifest_sha256") != train_sha):
        raise ValueError("single runtime: source evidence/rules are not bound to the approved training vault")
    # --limit 16 (v12 local check): the whole first order block, so v3-order ScienceQA items (positions 11-15) give H rows.
    fixed = order["items"][:16 if getattr(args, "limit", None) == 16 else 10]
    for item in fixed:
        pid = item["problem_id"]
        actor, evidence = actors.get(pid), sources.get(pid, {})
        if (actor is None or item["source"] != actor.source or evidence.get("source") != actor.source
                or evidence.get("actor_sha256") != text_sha256(to_json(actor))
                or evidence.get("image_sha256") != list(actor.image_sha256)
                or evidence.get("vault_manifest_sha256") != train_sha or evidence.get("split") != "train"):
            raise ValueError("single runtime: fixed10 source/actor/original image binding mismatch")
    source = source_manifest()
    ids = [i["problem_id"] for i in order["items"]]
    blocks = single_blocks(ids)
    model_config = _read(args.model_path / "config.json")
    if (args.model_path / "nf4_base.json").exists() or model_config.get("quantization_config") is not None:
        raise ValueError("single runtime: --model-path requires the raw BF16 snapshot; NF4 cache needs its separate verified loader")
    generation = generation_identity(argparse.Namespace(engine="hf", model_path=args.model_path, code_commit=source["base_commit"])) | {"precision": precision}
    contracts = {"source": digest(source), "dataset": file_sha(args.build_root / "manifest.yaml"),
                 "order": args.order_sha256, "data_admission": digest(admission)}
    version = SINGLE_CLI_VERSIONS[getattr(args, "single_version", "v12")]
    supplement_path = getattr(args, "note_reference_supplement", None)
    supplement_sha = getattr(args, "note_reference_supplement_sha256", None)
    if (supplement_path is None) != (supplement_sha is None):
        raise ValueError("single runtime: note-reference supplement and its external SHA256 are required together")
    note_reference = None
    if supplement_path is not None:
        supplement = references.read_supplement(supplement_path, supplement_sha)
        if not supplement.keys() <= set(ids):
            raise ValueError("single runtime: reference supplement names an item outside the order")
        note_reference = references.note_reference_pin(sources, supplement, supplement_sha, ids)
    protocol = plain_protocol_single(precision, len(blocks), contracts=contracts, token_scale=args.token_scale,
                                     note_reference=note_reference, version=version,
                                     **({"schedule_steps": len(blocks)} if version in (VERSION_SINGLE_V14, VERSION_SINGLE_V15) else {}))
    body = {"version": VERSION, "mode": args.mode, "source": source, "generation": generation,
            "protocol": protocol, "data_admission": admission, "fixed10": order["items"][:10],
            **({"fixed16": fixed} if getattr(args, "limit", None) == 16 else {}),
            "blocks": blocks, "source_provenance_sha256": digest(sources), "rules_sha256": file_sha(args.training_rules),
            "capacity": {"max_num_seqs": args.max_num_seqs, "padded_generation_token_budget": args.max_num_batched_tokens,
                         "training_token_budget": args.token_budget, "context_limit": CONTEXT_LIMIT,
                         "stop_after_seconds": args.stop_after_seconds},
            "runtime_implemented": True, "actual_model_execution": False, "cloud_ready": False}
    if measured is not None:
        from .capacity import load_profile

        if not isinstance(measured_sha, str):
            raise ValueError("single runtime: expected capacity profile SHA256 required")
        profile = load_profile(measured, measured_sha)
        expected = {"source_sha256": contracts["source"], "model_sha256": generation["model_files_sha256"], "precision": precision}
        if any(profile["binding"].get(key) != value for key, value in expected.items()):
            raise ValueError("single runtime: measured capacity source/model/precision differs from this plan")
        body["capacity"]["measured_profile"] = {"path": str(Path(measured).absolute()), "file_sha256": measured_sha}
    if capture_path is not None:
        # The fresh partial filename may change across review/resume invocations; capture behavior stays sealed.
        body["capacity"]["capture"] = {"enabled": True, "reserve_bytes": reserve}
    _sealed(args.output / "plan.json", body)
    return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "dry-run"))
    for name in ("build-root", "order", "frozen-approval", "source-provenance", "training-rules", "model-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--order-sha256", required=True)
    parser.add_argument("--mode", choices=("local-smoke", "frozen3500"), default="local-smoke")
    parser.add_argument("--token-scale", type=float, default=SINGLE_TOKEN_SCALE_V11)
    parser.add_argument("--limit", type=int, choices=(2, 10, 16), default=2,
                        help="items of the first order block; 16 seals the whole block (fixed16)")
    parser.add_argument("--single-version", choices=tuple(SINGLE_CLI_VERSIONS), default="v12")
    parser.add_argument("--note-reference-supplement", type=Path)
    parser.add_argument("--note-reference-supplement-sha256")
    parser.add_argument("--no-lesson-control", action="store_true",
                        help="diagnostic only (user 2026-11-11): after the rows are sealed, one lesson-free re-solve per "
                             "lesson re-solve, written to " + NO_LESSON_CONTROL_FILE + "; never a row or a reward")
    parser.add_argument("--precision", choices=("nf4", "bf16"), default="nf4")
    parser.add_argument("--capacity-profile", type=Path)
    parser.add_argument("--capacity-profile-sha256")
    parser.add_argument("--capture-capacity", type=Path, help="fresh actual-measurement partial; never generates extra answers")
    parser.add_argument("--capacity-reserve-bytes", type=int, help="explicit safety reserve, paired with --capture-capacity")
    parser.add_argument("--max-num-seqs", type=int, help="default 1; measured-profile mode ceiling 16")
    parser.add_argument("--max-num-batched-tokens", type=int, help="default one context; profile mode ceiling 16 contexts")
    parser.add_argument("--token-budget", type=int, help="default singleton rows; profile mode ceiling 16 contexts")
    parser.add_argument("--stop-after-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with _runtime_lock(args.output, ".cli.lock"):
        return _main(args)


def _main(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    plan = prepare(args)
    args.image_cfg = plan["protocol"]["image"]  # collect.hf_single_stream_builder reads getattr(args, "image_cfg", IMAGE)
    if args.command == "prepare":
        print(json.dumps({"status": "PREPARED", "model_loaded": False, "cloud_ready": False}))
        return 0
    import random
    import numpy as np
    import torch
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.policy import enable_deterministic_execution
    from fine_tuning.notebook_rl.policy import build_policy
    from fine_tuning.notebook_rl.quantized_policy import build_quantized_policy
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    from fine_tuning.notebook_rl.verifier_service import VerifierService
    from .train_v4 import encode_single_row

    enable_deterministic_execution()
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)
    actors = {a.problem_id: a for a in read_actors(args.build_root / "train")}
    sources = _read(args.source_provenance)["items"]
    items = plan.get("fixed16", plan["fixed10"])
    # The trusted controller keeps only the fixed items' gold strings; model requests never receive this mapping.
    private = read_verifier(args.build_root / "train")
    gold_values = {item["problem_id"]: private[item["problem_id"]].gold for item in items}
    del private
    note_references = {}
    if getattr(args, "note_reference_supplement", None) is not None:
        supplement = references.read_supplement(args.note_reference_supplement, args.note_reference_supplement_sha256)
        note_references = {item["problem_id"]: references.note_reference(sources[item["problem_id"]],
                                                                          supplement.get(item["problem_id"]))
                           for item in items}
    private_files = [args.training_rules, args.source_provenance,
                     *([args.note_reference_supplement] if note_references else [])]
    forbidden = install_training_guards(args.build_root, private_files=private_files)
    profile = None
    if "measured_profile" in plan["capacity"]:
        from .capacity import load_profile

        descriptor = plan["capacity"]["measured_profile"]
        profile = load_profile(Path(descriptor["path"]), descriptor["file_sha256"])
    precision = getattr(args, "precision", "nf4")
    bundle = (build_quantized_policy if precision == "nf4" else build_policy)(args.model_path, ARM, plan["protocol"])
    if precision == "bf16":
        bundle.model.to("cuda:0")
    capacity, capture = None, None
    identity = {"source_sha256": digest(plan["source"]), "model_sha256": plan["generation"]["model_files_sha256"], "precision": precision}
    if profile is not None:
        from .capacity import Capacity

        capacity = Capacity(bundle, profile, identity)
        args.capacity_session = capacity
    if getattr(args, "capture_capacity", None) is not None:
        from .capacity import Capture

        capture = Capture(bundle, identity, args.capacity_reserve_bytes)
        args.capacity_capture = capture

    def gold(pid):
        return gold_values[pid]

    try:
        with VerifierService(args.build_root / "train", forbidden_roots=forbidden, training_rules_path=args.training_rules,
                             training_rules_sha256=plan["rules_sha256"]) as verifier, ResourceLog(args.output / "resources.jsonl"):
            contracts = plan["protocol"]["training_contract"]["answer_contracts"]

            def grade(pid, text, finish_reason, mode):
                return grade_answer(verifier, contracts["mcq" if actors[pid].choices else "open"], pid, text,
                                    finish_reason, mode, actor=actors[pid])

            state = run_single_block(bundle, actors=actors, items=items, provenance=sources, gold=gold, grade=grade,
                                     load_images=lambda actor: load_images(actor, args.build_root / "images"),
                                     protocol=plan["protocol"], output=args.output / "block-0", limit=args.limit,
                                     stream_args=args, stop_after=args.stop_after_seconds,
                                     reference=note_references.get if note_references else None)
            if getattr(args, "no_lesson_control", False) and state["status"] == "ROWS_READY":
                control = run_no_lesson_control(
                    bundle, block=args.output / "block-0", actors=actors, grade=grade,
                    load_images=lambda actor: load_images(actor, args.build_root / "images"), protocol=plan["protocol"],
                    output=args.output, stream_args=args, stop_after=args.stop_after_seconds)
                print(json.dumps({"no_lesson_control": control["status"], "items": len(control["items"])}))
        if args.command == "dry-run" and state["status"] == "ROWS_READY":
            closed = _read(args.output / "block-0" / state["rows_path"])
            encoded = [encode_single_row(bundle.processor, bundle.model.config, row,
                                         load_images(actors[row["problem_id"]], args.build_root / "images"),
                                         image=plan["protocol"]["image"]) for row in closed["rows"]]
            with ResourceLog(args.output / "backward-resources.jsonl"):
                report = prove_single_backward(bundle, encoded, closed["contract"], precision=precision,
                                               token_budget=args.token_budget, capacity=capacity)
            _sealed(args.output / f"dry-run-{args.limit}.json", report)
            if capture is not None:
                if report.get("error") or report["status"] == "FAIL" or report.get("all_parameters_unchanged") is False:
                    capture.failed = True
                else:
                    from .capacity import capture_backward

                    try:
                        capture_backward(capture, encoded, closed["contract"], component="RL", token_budget=args.token_budget)
                        for kind in ("A1", "A2", "B"):
                            if any(row["kind"] == kind for row in encoded):
                                capture_backward(capture, encoded, closed["contract"], component=kind, token_budget=args.token_budget)
                    except BaseException:
                        capture.failed = True
                        raise
            print(json.dumps(report))
            return 0 if report["status"] == "PASS" else 2
        print(json.dumps({"status": state["status"], "waiting": state.get("waiting", []), "weights_saved": False}))
        return 0 if state["status"] == "ROWS_READY" else 2
    finally:
        if capture is not None:
            pending = sys.exception()
            try:
                partial = capture.write(args.capture_capacity)
                print(json.dumps({"capacity_capture": str(args.capture_capacity), "status": partial["status"],
                                  "file_sha256": file_sha(args.capture_capacity), "entries": len(partial["entries"])}))
            except Exception as error:
                if pending is None:
                    raise
                pending.add_note(f"capacity partial publication failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
