"""Error-note version 4: group rollouts with an RL reward plus self-note SFT for items the model never solves.

Plan: claudedocs/plan_rl_sft_hybrid_v4_20261104.md sections 3-4 (budget variant of section 10). CPU only, no model imports.
Nothing here grades: statuses come from the separate verifier service. Nothing here writes or edits model text: every
target is a generated completion as produced. The v1-v3 names in contract.py are unchanged.

Per item and round: up to ROLLOUT_N gold-blind P0 rollouts (adaptive: ADAPTIVE_FIRST first, the rest only when those
are not all correct). A MIXED group gives RL rows (reward - group mean). An ALL_CORRECT group gives at most one A1
retention row. An ALL_WRONG group gets one gold-aware note; a lesson that passes the program gate (and is not
contradicted by the model's own gold-blind re-reading of its image claims) is used for RESOLVE_M gold-blind
re-solves; a correct one becomes an A2 row on the hint-free P0 prompt, and the note becomes a B row.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Callable, Iterable

from .contract import (MASTER_SEED, MAX_TARGET_TOKENS, MIN_TARGET_TOKENS, ROUTE_FULL,
                       ROUTE_LM, _has_words, has_repetition, hint_leak, is_loop, parse_note_v2, resolve_leaked_phrase)

VERSION = "error-note-v4"
ROLLOUT_N = 4
ADAPTIVE_FIRST = 2
RESOLVE_M = 2
PROBE_CLAIMS = 3
PROBE_SEEDS = 2
PROBE_MAX_TOKENS = 256
LOOP_BLOCK = 64
LOOP_REPEATS = 8
KIND_RL, KIND_A2, KIND_B, KIND_A1 = "RL", "A2", "B", "A1"
KINDS = (KIND_RL, KIND_A2, KIND_B, KIND_A1)
SFT_KINDS = (KIND_A2, KIND_B, KIND_A1)
CORRECT = "CORRECT"
LOOP_STOPPED = "LOOP_STOPPED"
UNUSABLE_STATUSES = ("UNVERIFIABLE", "CONTEXT_BLOCKED")
MIXED, ALL_CORRECT, ALL_WRONG, UNUSABLE = "MIXED", "ALL_CORRECT", "ALL_WRONG", "UNUSABLE"
SELF_CONSISTENT, SELF_CONTRADICTED, UNCHECKED = "SELF_CONSISTENT", "SELF_CONTRADICTED", "UNCHECKED"
PURPOSES = ("roll", "resolve", "probe")
# Predeclared candidates (plan section 3.5); never tuned on MMMU. A changed value is a new protocol.
SFT_WEIGHTS = {KIND_A2: 0.5, KIND_B: 0.05, KIND_A1: 0.05}
RL_SETTINGS = {"coefficient": 1.0, "eps_low": 0.2, "eps_high": 0.28, "beta": 0.02,
               "advantage": "reward_minus_group_mean", "normalization": "group_action_tokens",
               "old_logprobs": "current_policy_detached"}
LR_V4 = {"L": 5e-5, "M": 5e-6, "V": 1e-5}
ARMS = ("hybrid", "rl-only")  # rl-only: the same RL and A1 rows, no note-derived A2/B (the no-note control)
CLAIM_PRIORITY = re.compile(r"\d|\b(?:left|right|top|bottom|above|below|upper|lower|arrow|label(?:ed|s)?|line|"
                            r"curve|axis|count|counts?|number|each|two|three|four|five|six|seven|eight|nine|ten)\b",
                            re.IGNORECASE)
CLAIM_SKIP = re.compile(r"\b(?:cannot|can't|unclear|not visible|not possible|unable)\b", re.IGNORECASE)
PROBE_TEMPLATE = ('Look at the image. Is this statement about the image true?\nStatement: "{claim}"\n'
                  "Answer with exactly one word first: TRUE, FALSE or CANNOT_TELL. Then give one short sentence why.")
_PROBE_WORD = re.compile(r"\W*(TRUE|FALSE|CANNOT[ _]TELL)\b", re.IGNORECASE)
_IMAGE_SECTION = re.compile(r"What the image shows\s*:?(.*?)(?=\n\s*Image reading\b)", re.IGNORECASE | re.DOTALL)


def v4_seed(problem_id: str, round_index: int, purpose: str, index: int, master_seed: int = MASTER_SEED) -> int:
    """SHA256 seed of one v4 sample; disjoint from the evaluation seed "3407:{id}" and the v1-v3 stage seeds."""
    if (not isinstance(problem_id, str) or not problem_id or type(round_index) is not int or round_index < 1
            or purpose not in PURPOSES or type(index) is not int or index < 0):
        raise ValueError("v4_seed: problem id, round >= 1, purpose and index >= 0 required")
    key = f"{master_seed}:{problem_id}:v4r{round_index}:{purpose}:{index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31)


def reward(status: str) -> int | None:
    """1 for CORRECT, 0 for any other graded outcome (including a loop stop), None when the sample cannot be scored."""
    if not isinstance(status, str) or not status:
        raise ValueError("reward: a status string is required")
    if status in UNUSABLE_STATUSES:
        return None
    return int(status == CORRECT)


def needs_more(first_statuses: list[str]) -> bool:
    """Adaptive rollouts: stop after ADAPTIVE_FIRST samples only when all of them are correct."""
    if len(first_statuses) != ADAPTIVE_FIRST:
        raise ValueError(f"needs_more: exactly {ADAPTIVE_FIRST} first statuses required")
    return not all(status == CORRECT for status in first_statuses)


def classify_group(statuses: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """(label, [(rollout index, reward)]) over the scorable samples; fewer than two of them is UNUSABLE."""
    usable = [(j, value) for j, status in enumerate(statuses) if (value := reward(status)) is not None]
    if len(usable) < 2:
        return UNUSABLE, usable
    total = sum(value for _, value in usable)
    label = ALL_CORRECT if total == len(usable) else ALL_WRONG if total == 0 else MIXED
    return label, usable


def advantages(rewards: list[int]) -> list[float]:
    """Reward minus the group mean, no standard-deviation scaling (notebook_rl.losses.group_advantages)."""
    if len(rewards) < 2:
        raise ValueError("advantages: a group of at least two samples is required")
    mean = sum(rewards) / len(rewards)
    return [value - mean for value in rewards]


class BlockRepetition:
    """Incremental detector: one LOOP_BLOCK-token window seen min_repeats times without overlap.

    Stores 64-bit window hashes, not windows. Used to stop a training rollout that only repeats itself (the
    reward is 0 either way); it never shortens an evaluation run and never raises the 8192 cap.
    """

    def __init__(self, block: int = LOOP_BLOCK, min_repeats: int = LOOP_REPEATS):
        if block < 1 or min_repeats < 2:
            raise ValueError("BlockRepetition: block >= 1 and min_repeats >= 2 required")
        self.block, self.min_repeats = block, min_repeats
        self.ids: list[int] = []
        self.seen: dict[int, tuple[int, int]] = {}
        self.found = False

    def feed(self, token_ids: Iterable[int]) -> bool:
        for token in token_ids:
            self.ids.append(int(token))
            if self.found or len(self.ids) < self.block:
                continue
            start = len(self.ids) - self.block
            key = hash(tuple(self.ids[start:]))
            count, last = self.seen.get(key, (0, -self.block))
            if start >= last + self.block:
                count, last = count + 1, start
                self.seen[key] = (count, last)
                if count >= self.min_repeats:
                    self.found = True
        return self.found


def has_block_repetition(token_ids: list[int], block: int = LOOP_BLOCK, min_repeats: int = LOOP_REPEATS) -> bool:
    return BlockRepetition(block, min_repeats).feed(token_ids)


def program_gate(note_row: dict | None, gold: str, choices: tuple[str, ...] | list[str], *,
                 aliases: Iterable[str] = (), acronyms: Iterable[str] = ()) -> dict:
    """Program-only acceptance of a gold-aware v2 note. No human or AI review is read or required here.

    PASS needs: a stopped, parsable note; no Image reading MISREAD with Cause NONE; no option letter, answer phrase,
    gold text, source alias or unconfirmed acronym in the lesson; no compressed loop. Reasons carry no gold text."""
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("program_gate: nonblank gold required")
    result = {"result": "FAIL", "reason": None, "lesson": None, "image_reading": None, "cause": None}
    if note_row is None or "status" in note_row:
        return result | {"reason": "NO_NOTE"}
    if note_row.get("finish_reason") != "stop":
        return result | {"reason": "NOTE_NOT_STOPPED"}
    parsed = parse_note_v2(note_row.get("text", ""))
    if parsed is None:
        return result | {"reason": "NOTE_UNPARSED"}
    result |= {"image_reading": parsed.image_reading, "cause": parsed.cause}
    if parsed.image_reading == "MISREAD" and parsed.cause == "NONE":
        return result | {"reason": "MISREAD_WITH_CAUSE_NONE"}
    leak = hint_leak(parsed.lesson, gold, choices)
    if leak is not None:
        return result | {"reason": "HINT_LEAK:" + leak.split(":")[0]}
    if any(_has_words(parsed.lesson, value) for value in aliases):
        return result | {"reason": "ALIAS_LEAK"}
    if any(_has_words(parsed.lesson, value) for value in acronyms):
        return result | {"reason": "ACRONYM_CANDIDATE"}
    if is_loop(parsed.note_text):
        return result | {"reason": "NOTE_LOOP"}
    return result | {"result": "PASS", "lesson": parsed.lesson}


def extract_claims(note_text: str, gold: str, choices: tuple[str, ...] | list[str], limit: int = PROBE_CLAIMS) -> list[str]:
    """Up to `limit` checkable sentences of "What the image shows", by rule (no model call).

    Sentences with a number or a position/count word come first, then document order. A sentence that would give the
    answer away (contract.hint_leak) or that only states uncertainty is skipped, so the gold-blind probe stays gold-blind."""
    match = _IMAGE_SECTION.search(note_text or "")
    if match is None or limit < 1:
        return []
    sentences = [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+|\n+", match[1]) if s.strip()]
    usable = [(index, s) for index, s in enumerate(sentences)
              if 4 <= len(s.split()) <= 60 and '"' not in s and not CLAIM_SKIP.search(s)
              and hint_leak(s, gold, choices) is None]
    ranked = sorted(usable, key=lambda pair: (CLAIM_PRIORITY.search(pair[1]) is None, pair[0]))
    return [sentence for _, sentence in ranked[:limit]]


def probe_prompt(claim: str) -> str:
    if not isinstance(claim, str) or not claim.strip() or '"' in claim:
        raise ValueError("probe_prompt: a nonblank claim without double quotes is required")
    return PROBE_TEMPLATE.format(claim=claim.strip())


def parse_probe(text: str) -> str:
    match = _PROBE_WORD.match(text or "")
    if match is None:
        return "INVALID"
    return match[1].upper().replace(" ", "_")


def decide_selfcheck(verdicts: list[list[str]]) -> str:
    """verdicts[claim][seed]. Any FALSE contradicts; every answer TRUE is consistent; anything else is unchecked."""
    if not verdicts or any(not answers for answers in verdicts):
        return UNCHECKED
    flat = [answer for answers in verdicts for answer in answers]
    if "FALSE" in flat:
        return SELF_CONTRADICTED
    return SELF_CONSISTENT if all(answer == "TRUE" for answer in flat) else UNCHECKED


def target_problem(text: str, token_ids: list[int], finish_reason: str, status: str, *,
                   min_tokens: int = MIN_TARGET_TOKENS) -> str | None:
    """Why a correct completion may not be an SFT target (A1/A2); None when it may. The v2 length rules plus loops."""
    if finish_reason != "stop":
        return "NOT_STOPPED"
    if status != CORRECT:
        return "NOT_CORRECT"
    if type(min_tokens) is not int or min_tokens < 1:
        raise ValueError("target_problem: positive integer token minimum required")
    if len(token_ids) < min_tokens:
        return "TOO_SHORT"
    if len(token_ids) > MAX_TARGET_TOKENS:
        return "TOO_LONG"
    if has_repetition(token_ids):
        return "REPETITION"
    if has_block_repetition(token_ids):
        return "BLOCK_REPETITION"
    if is_loop(text):
        return "LOOP"
    return None


def build_rows_v4(record: dict, tokenize: Callable[[str], list[int]], arm: str = "hybrid") -> tuple[list[dict], list[dict]]:
    """One item of one round -> (rows, drop records). Texts are used exactly as generated.

    record: problem_id, subject, source, p0_text, image_paths, rollouts [{j, seed, text, finish_reason, status}],
    optional note {prompt, text, finish_reason}, gate (program_gate), selfcheck {result}, resolves [{j, seed, text,
    finish_reason, status}]. RL rows keep truncated or loop-stopped samples (reward 0) without an end token."""
    if arm not in ARMS:
        raise ValueError(f"unknown v4 arm {arm!r}")
    pid = record["problem_id"]
    rows: list[dict] = []
    drops: list[dict] = []
    rollouts = sorted(record.get("rollouts", []), key=lambda r: r["j"])
    if [r["j"] for r in rollouts] != list(range(len(rollouts))):
        raise ValueError(f"{pid}: rollout indices must be 0..n-1")

    def drop(kind: str, reason: str) -> None:
        drops.append({"problem_id": pid, "kind": kind, "reason": reason})

    def make(kind: str, suffix: str, prompt: str, target: str, route: str, tokens: int, **extra) -> dict:
        return {"row_id": f"{pid}:{kind}:{suffix}", "problem_id": pid, "kind": kind, "prompt": prompt, "target": target,
                "route": route, "image_paths": list(record["image_paths"]), "target_tokens": tokens,
                "subject": record["subject"], "source": record["source"], "append_end": True, **extra}

    label, usable = classify_group([r["status"] for r in rollouts])
    if label == UNUSABLE:
        drop(KIND_RL, "GROUP_UNUSABLE")
        return rows, drops
    if label == MIXED:
        values = advantages([value for _, value in usable])
        for (j, value), adv in zip(usable, values, strict=True):
            sample = rollouts[j]
            stopped = sample["finish_reason"] == "stop"
            rows.append(make(KIND_RL, str(j), record["p0_text"], sample["text"], ROUTE_LM, len(tokenize(sample["text"])),
                             reward=value, advantage=adv, group_size=len(usable), rollout_seed=sample["seed"],
                             finish_reason=sample["finish_reason"], append_end=stopped, target_source="ROLLOUT"))
        return rows, drops
    if label == ALL_CORRECT:
        for sample in rollouts:
            problem = target_problem(sample["text"], tokenize(sample["text"]), sample["finish_reason"], sample["status"])
            if problem is None:
                rows.append(make(KIND_A1, str(sample["j"]), record["p0_text"], sample["text"], ROUTE_LM,
                                 len(tokenize(sample["text"])), rollout_seed=sample["seed"], target_source="ROLLOUT"))
                return rows, drops
        drop(KIND_A1, "NO_USABLE_CORRECT_ROLLOUT")
        return rows, drops
    # ALL_WRONG: only the note path can supervise this item.
    if arm == "rl-only":
        drop(KIND_A2, "ARM_RL_ONLY")
        return rows, drops
    gate, check = record.get("gate"), (record.get("selfcheck") or {}).get("result", UNCHECKED)
    if gate is None or gate["result"] != "PASS":
        drop(KIND_B, "GATE_" + (gate["reason"] if gate else "MISSING"))
        drop(KIND_A2, "NO_AUTHORIZED_LESSON")
        return rows, drops
    if check == SELF_CONTRADICTED:
        drop(KIND_B, "SELF_CONTRADICTED")
        drop(KIND_A2, "SELF_CONTRADICTED")
        return rows, drops
    parsed = parse_note_v2(record["note"]["text"])
    if parsed is None or parsed.lesson != gate["lesson"]:
        raise ValueError(f"{pid}: the gate lesson differs from the note")
    route = ROUTE_FULL if parsed.image_reading == "MISREAD" and check == SELF_CONSISTENT else ROUTE_LM
    a2 = None
    for sample in sorted(record.get("resolves", []), key=lambda r: r["j"]):
        ids = tokenize(sample["text"])
        problem = target_problem(sample["text"], ids, sample["finish_reason"], sample["status"])
        if problem is None and (phrase := resolve_leaked_phrase(sample["text"])):
            problem = f"LEAK:{phrase}"
        if problem is None:
            a2 = make(KIND_A2, str(sample["j"]), record["p0_text"], sample["text"], route, len(ids),
                      resolve_seed=sample["seed"], target_source="RESOLVE")
            break
        drop(KIND_A2, f"RESOLVE_{sample['j']}:{problem}")
    if a2 is not None:
        rows.append(a2)
    elif not record.get("resolves"):
        drop(KIND_A2, "NO_RESOLVE")
    if a2 is None and check != SELF_CONSISTENT:
        drop(KIND_B, f"NOTE_{check}_WITHOUT_A2")
        return rows, drops
    note_ids = tokenize(parsed.note_text)
    if len(note_ids) < MIN_TARGET_TOKENS:
        drop(KIND_B, "NOTE_EMPTY")
    else:
        rows.append(make(KIND_B, "note", record["note"]["prompt"], parsed.note_text, route, len(note_ids),
                         selfcheck=check, target_source="NOTE"))
    return rows, drops


def summarize_v4(records: list[dict], rows: list[dict], drops: list[dict]) -> dict:
    """Counts only, no text: group labels, rows, drops and the lesson effect against the no-lesson control."""
    labels = Counter(classify_group([r["status"] for r in rec.get("rollouts", [])])[0] for rec in records)
    wrong = [rec for rec in records if classify_group([r["status"] for r in rec.get("rollouts", [])])[0] == ALL_WRONG]

    def rate(key: str) -> dict:
        samples = [s for rec in wrong for s in rec.get(key, []) if reward(s["status"]) is not None]
        correct = sum(s["status"] == CORRECT for s in samples)
        return {"samples": len(samples), "correct": correct, "rate": correct / len(samples) if samples else None}

    return {"version": VERSION, "items": len(records), "groups": dict(labels), "rows": len(rows),
            "rows_by_kind": dict(Counter(row["kind"] for row in rows)),
            "rows_by_route": dict(Counter(row["route"] for row in rows)),
            "drops": len(drops), "drops_by_reason": dict(Counter(f"{d['kind']}:{d['reason']}" for d in drops)),
            "gates": dict(Counter((rec.get("gate") or {}).get("reason") or (rec.get("gate") or {}).get("result", "NONE")
                                  for rec in wrong)),
            "selfcheck": dict(Counter((rec.get("selfcheck") or {}).get("result", "NONE") for rec in wrong)),
            "lesson_resolve": rate("resolves"), "no_lesson_control": rate("controls"),
            "loop_stopped_rollouts": sum(r["status"] == LOOP_STOPPED for rec in records for r in rec.get("rollouts", []))}
