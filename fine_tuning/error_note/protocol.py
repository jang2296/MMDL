"""The plain training protocol dict for fine_tuning.notebook_rl.policy and checkpoint.

Learning rates are the recipe G6 actually ran; only the training target changed (plan section 3.4).
Version 2 (claudedocs/plan_self_note_v2_20261102.md section 4): the same model, LoRA, learning rates and clip;
16 items per update and a linear decay to 0 after the 10-step warmup.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from fine_tuning.modules import BASE_ID, BASE_REVISION
from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT, TURN1_EVAL_CONTRACT

from .contract import (CONTEXT_LIMIT, IMAGE, LOOP_RATIO, LOOP_WINDOW, MASTER_SEED, MAX_NEW_TOKENS, MAX_TARGET_TOKENS,
                       MIN_TARGET_TOKENS, NOTE_INSTRUCTION, NOTE_INSTRUCTION_SINGLE, NOTE_INSTRUCTION_V2, NOTE_INSTRUCTION_V3,
                       REFERENCE_HEADER, RESOLVE_HEADER, RESOLVE_INSTRUCTION_SINGLE, SINGLE_VERDICT_LABELS, DENIAL_PHRASES,
                       GRADER_READ_LINE, GRADER_READ_NONE, LESSON_STEM_RULE, _REVIEW_REFERENCES)
from .single_grading import (ALIASES, ALIASES_SHA256, NORMALIZATION_RULES, NORMALIZATION_RULES_V2, NUMBER_WORDS, SCORER,
                             SCORER_V2, SUFFIX_STOP, UNIT_ABBREVIATIONS)

ARM = "MOD-RV"  # installs Vision, LLM and Merger trainables; the per-item route decides who gets gradients
MERGERS = ["model.visual.merger", "model.visual.deepstack_merger_list.0",
           "model.visual.deepstack_merger_list.1", "model.visual.deepstack_merger_list.2"]
FORBIDDEN = {"additional_sft": False, "teacher_or_upper_model_labels": True, "benchmark_training": True}
GRAD_CLIP = 1.0
VERSION_V2 = "error-note-protocol-v2"
PER_UPDATE_ITEMS_V2 = 16
VERSION_V3 = "error-note-protocol-v3"
ADMISSION_VERSION_V3 = "error-note-admission-v4"
V3_CONTRACT_HASHES = frozenset(("rows", "order", "audit", "sidecar", "solution_audit", "admission"))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def plain_protocol(precision: str) -> dict:
    """precision: "bf16" on the rented GPU, "nf4" for local functional checks only."""
    if precision not in ("bf16", "nf4"):
        raise ValueError("protocol: precision must be bf16 or nf4")
    body: dict = {
        "version": "error-note-protocol-v1", "precision": precision, "forbidden": FORBIDDEN,
        "initial_checkpoint": {"model_id": BASE_ID, "revision": BASE_REVISION},
        "image": dict(IMAGE), "max_new_tokens": MAX_NEW_TOKENS, "context_limit": CONTEXT_LIMIT,
        "model": {"lora": {"r": 16, "alpha": 32, "dropout": 0.0, "bias": "none"}, "modules_to_save": MERGERS},
        "expected_trainable": {"L": 33030144, "M": 109105152, "V": 2359296},
        "optimizer": {"type": "adamw", "schedule": "constant_after_warmup", "warmup_steps": 10,
                      "lr": {"L": 5e-5, "M": 1e-5, "V": 1e-5}, "betas": [0.9, 0.95], "eps": 1e-8,
                      "weight_decay": 0.0, "grad_clip_per_group": GRAD_CLIP},
        "prompts_sha256": hashlib.sha256(NOTE_INSTRUCTION.encode()).hexdigest(),
    }
    if precision == "nf4":
        from fine_tuning.notebook_rl.quantized_policy import quantization_contract

        body["quantization"] = quantization_contract()
    return body | {"sha256": _digest(body)}


def plain_protocol_v2(precision: str, total_steps: int) -> dict:
    """total_steps: the optimizer steps of the run (its non-empty update groups); the multiplier reaches 0 there."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("protocol: total_steps must be a positive integer")
    body = {key: value for key, value in plain_protocol(precision).items() if key != "sha256"}
    body |= {"version": VERSION_V2, "per_update_items": PER_UPDATE_ITEMS_V2,
             "optimizer": body["optimizer"] | {"schedule": "linear_decay_after_warmup", "total_steps": total_steps},
             "prompts_sha256": _digest({"note": NOTE_INSTRUCTION_V2, "resolve_header": RESOLVE_HEADER})}
    return body | {"sha256": _digest(body)}


VERSION_V4 = "error-note-protocol-v4"
V4_CONTRACT_HASHES = frozenset(("rows", "order", "init"))


def plain_protocol_v4(precision: str, total_steps: int, *, arm: str, round_index: int, start: int, limit: int,
                      contracts: dict[str, str | None]) -> dict:
    """RL + note SFT (claudedocs/plan_rl_sft_hybrid_v4_20261104.md section 3.5). contracts: rows/order SHA256 and the
    init checkpoint metadata SHA256 (None for a round that starts from F0). Every loss value is fixed here."""
    from .v4 import ARMS, LR_V4, RL_SETTINGS, ROLLOUT_N, SFT_WEIGHTS, VERSION as COLLECTION_VERSION

    if arm not in ARMS:
        raise ValueError(f"protocol v4: arm must be one of {ARMS}")
    if any(type(v) is not int for v in (round_index, start, limit)) or round_index < 1 or start < 0 or limit < 1:
        raise ValueError("protocol v4: round >= 1, start >= 0 and limit >= 1 required")
    if set(contracts) != V4_CONTRACT_HASHES or any(
            value is not None and (not isinstance(value, str) or len(value) != 64
                                   or any(c not in "0123456789abcdef" for c in value))
            for value in contracts.values()) or contracts["rows"] is None or contracts["order"] is None:
        raise ValueError("protocol v4: rows/order SHA256 and an init SHA256 or null required")
    if (round_index == 1) != (contracts["init"] is None):
        raise ValueError("protocol v4: round 1 starts from F0 and later rounds from an init checkpoint")
    body = {key: value for key, value in plain_protocol_v2(precision, total_steps).items() if key != "sha256"}
    weights = dict(SFT_WEIGHTS) if arm == "hybrid" else {**SFT_WEIGHTS, "A2": 0.0, "B": 0.0}
    body |= {"version": VERSION_V4, "optimizer": body["optimizer"] | {"lr": dict(LR_V4)},
             "prompts_sha256": _digest({"note": NOTE_INSTRUCTION_V3, "resolve_header": RESOLVE_HEADER,
                                        "collection": COLLECTION_VERSION}),
             "training_contract": {"arm": arm, "round": round_index, "start": start, "limit": limit,
                                   "hashes": dict(contracts), "master_seed": MASTER_SEED, "rollout_n": ROLLOUT_N,
                                   "rl": dict(RL_SETTINGS), "sft_weights": weights,
                                   "sft_row": "mean_target_and_eos_nll", "renormalize_missing_kind": False,
                                   "length_policy": {"min_target_tokens": MIN_TARGET_TOKENS,
                                                     "max_target_tokens": MAX_TARGET_TOKENS,
                                                     "loop_window": LOOP_WINDOW, "loop_ratio": LOOP_RATIO}}}
    return body | {"sha256": _digest(body)}


def plain_protocol_v3(precision: str, total_steps: int, *, arm: str, contracts: dict[str, str]) -> dict:
    """The common admitted update schedule, with arm-specific loss and immutable input receipts."""
    from .audit import SOLUTION_REVIEW_VERSION

    if arm not in ("note", "no-note"):
        raise ValueError("protocol v3: arm must be note or no-note")
    if set(contracts) != V3_CONTRACT_HASHES or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in contracts.values()):
        raise ValueError("protocol v3: rows/order/audit/sidecar/solution-audit/admission SHA256 bindings required")
    body = {key: value for key, value in plain_protocol_v2(precision, total_steps).items() if key != "sha256"}
    body |= {"version": VERSION_V3,
             "prompts_sha256": _digest({"note": NOTE_INSTRUCTION_V3, "resolve_header": RESOLVE_HEADER}),
             "training_contract": {"arm": arm, "hashes": dict(contracts), "master_seed": MASTER_SEED,
                                   "answer_contract": TRAINING_ANSWER_CONTRACT, "audit_contract": "note-audit-v1",
                                   "solution_content_contract": SOLUTION_REVIEW_VERSION,
                                   "admission": ADMISSION_VERSION_V3, "active_group": "at_least_one_encoded_A",
                                   "route": "preserve_same_A_in_both_arms", "unexpected_drop": "STOP",
                                   "length_policy": {"min_target_tokens": MIN_TARGET_TOKENS,
                                                     "max_target_tokens": MAX_TARGET_TOKENS,
                                                     "loop_window": LOOP_WINDOW, "loop_ratio": LOOP_RATIO},
                                   "loss": {"row": "mean_target_and_eos_nll", "A": 0.9,
                                            "B": 0.1 if arm == "note" else 0.0, "renormalize_missing_kind": False}}}
    return body | {"sha256": _digest(body)}


# v3 (user 2026-11-05): program-only row admission; RL replay over the stored generation-time support.
# v4: MCQ graded by the V8-aligned contract, >=32-token solutions for reward/A1/A2 (pilot v3 collapsed to bare
# answers), replay drift bounded per block and on long rows (one-token rows measure engine noise, not binding).
# v5 (2026-11-06, main v4 run stopped at slot 75): V2 note instruction (V3 taught image-only hedging) and reward -1
# for a normally finished T1 under 32 tokens (bare answers rose 28% -> 44% with no penalty), see
# claudedocs/analysis_single_rlsft_failure_20261106.md.
# v6 (2026-11-06): open answers also use the V8-aligned contract MMMU scores with. training-answer-v2 rejected
# common final-answer notation (emoji/bold/heading, a repeated boxed value): on the stored v4 T1, 262 of 514 open
# answers were INVALID, so their notes reviewed parser output; V8 grading gives +33 correct open answers
# (-24 bare range lines) and A2-eligible re-solves 50 -> 76.
# v7 (2026-11-06, cycle-1 64-item review): open answers V8 first, then training-answer-v2 only to turn a non-CORRECT
# V8 result CORRECT (unit/word false negatives); reward -1 only for a wrong bare answer, a stopped answer the parser
# could not commit to, or a looping wrong answer (correct bare answers 0); notes capped at 1024 tokens; note and
# re-solve instructions; program filters on A2 (lesson echo, image-only hedging) and B (answer denial).
# v8 (2026-11-06, cycle-2 64-item review, claudedocs/analysis_single_v7_cycle2_20261106.md section 5): open answers
# graded by single_grading (V8 extraction, recorded notation normalization incl. a per-source alias table, the
# training comparison); a committed answer never gets NO_COMMITTED_ANSWER; the note input shows the grader-read
# value; notes with Cause NONE, or a non-NO_FINAL_ANSWER cause for an unaccepted/cut-off answer, are rejected;
# lesson stem leak check; answer-denial phrases block T1 reward/A1 and A2; review phrases on word boundaries.
# v9 (2026-11-07, v8 main run stopped at block 27 of 219 on one row's vLLM-vs-HF replay mismatch, and the T1
# answer-first share rose within sources as in v4): a block may drop up to two replay-failing RL rows from PG, KL and
# the anchor (block mean still covers every row; fixed denominators), and the first T1 positions carry an
# always-on raw F0 KL anchor so the reason-first/answer-first decision stays near the original model.
# v10 (2026-11-07, claudedocs/analysis_why_rlsft_not_improving_20261107.md: over v9's 55 updates the weighted loss was
# A2 9.12, anchor 1.58, B 0.62, A1 0.57, PG 0.013, KL 0.004 and 51% of rewards were 0): T1 PG uses a fixed per-source
# baseline (mean v9 T1 reward of blocks 0-54, shipped as source_baseline_v10.json and pinned by SHA256), so every
# non-excluded RL row carries A = reward - b(source); token scale 256 instead of 1024; A2 SFT weight 0.35 per row
# instead of 0.5 shared by the block's A2 rows. Rewards, gates, prompts, anchor, KL and replay are v9's.
# v11 (2026-11-08, claudedocs/plan_v11_20261108.md): no baseline; reward gate v6 adds REASONED_WRONG and
# TRUNCATED_NO_LOOP penalties and a fixed advantage table (+1 / -1 / -0.5 / 0); negative rows use the raw
# full-vocabulary log p (a one-token stored support gives log q = 0 and no gradient); per-row cap min(1, 2048/n);
# token scale 1024; the note prompt may show a dataset reference after the correct answer. LOOP rows are exempt from
# the row cap (weight 1.0, as in v9) so a long repetition loop keeps its full penalty mass.
# v12 (2026-11-11, claudedocs/plan_v12_20261111.md items 1-4, 6): reward gate v7 (REASONED_WRONG 0, new CONTAINED_OPEN
# 0 for an open answer that contains the gold's words), lesson leak rule v2 (wrong-option names allowed unless they
# eliminate), B rows from every accepted note whatever the re-solve gave, and H rows: a dataset's human-written
# solution (ScienceQA lecture + solution + one final-answer line) as LM-only SFT under P0, weight 0.02 per kind (0.10 gave an H gradient 7x the PG gradient in the local NF4 block, 2026-11-11).
# Everything else is v11b. plain_protocol_single(version=VERSION_SINGLE) still builds the byte-identical v11 protocol.
# v13 (2026-11-12, claudedocs/research_v12_drop_and_67_path_20261112.md section 7.3): the answers MMMU loses most are
# self-doubt spirals ("Wait ..." after a finished first derivation until the token cap), so v13 trains finishing only.
# Reward gate v8 = gate v7 plus two program fields: a LOOP / TRUNCATED_NO_LOOP row carries the spiral window (from the
# third marker token, or token 1024 when there are fewer than three, for at most 2048 tokens) and its PG term reads the
# raw log p of that window alone at advantage -1 with no row cap, so the first derivation is never pushed down; an
# accepted row is praised (+1, and an A1 row) only when it has at least 256 tokens, shorter correct answers carry 0.
# No H rows. Lesson leak rule v2, B rows, KL, anchor, replay, SFT weights and everything else are v12's.
# SINGLE v14 (user 2026-11-13, claudedocs/plan_v14_20261113.md): v13 with its reviewed defects fixed, on the same
# 1,020-item head and order. Reward gate v9 = gate v8 + ``tail``: a CORRECT, stopped T1 of at least 256 tokens that is
# no A1 target only because it repeats or is too long earns +1 on its conclusion alone (tail_window: the tokens after
# its last "Wait" marker, at most the last 1,024, when that window has at least 32 tokens and does not repeat; PG on
# the stored-support log q of that window at full row weight; never an A1 row). Training images use the B2 evaluation
# pixel budget. The LR reaches 0 at ``schedule_steps`` (the blocks the plan may execute), not at all 219 slots. An A2
# row needs at least 256 tokens. A note is a B row only when its own re-solve passes v11's A2 checks ("B follows its
# A2", decided before the 256-token rule). Open answers are graded by scorer v2. Everything else is v13's.
VERSION_SINGLE_V8 = "error-note-single-response-v8"
VERSION_SINGLE_V9 = "error-note-single-response-v9"
VERSION_SINGLE_V10 = "error-note-single-response-v10"
VERSION_SINGLE = "error-note-single-response-v11"
VERSION_SINGLE_V12 = "error-note-single-response-v12"
VERSION_SINGLE_V13 = "error-note-single-response-v13"
VERSION_SINGLE_V14 = "error-note-single-response-v14"
# SINGLE v15 (user 2026-11-14, claudedocs/plan_v15_20261114.md): v14 plus up to three lesson re-solves, the vision route
# on every training row of an image item, and training rollouts that apply the LLM LoRA unmerged in vLLM with exact
# vision-LoRA hooks. The protocol is v14's with exactly the keys named below added or changed.
VERSION_SINGLE_V15 = "error-note-single-response-v15"
SINGLE_RESOLVE_STAGES_V15 = ("resolve", "resolve2", "resolve3")
SINGLE_RESOLVE_V15 = {
    "version": "lesson-resolve-attempts-v1", "max_attempts": 3, "stages": list(SINGLE_RESOLVE_STAGES_V15),
    "prompt": "resolve_prompt_single(P0, lesson), identical for every attempt",
    "seeds": {"resolve": "<master>:<pid>:resolve", "resolve2": "<master>:<pid>:resolve:2",
              "resolve3": "<master>:<pid>:resolve:3"},
    "next_attempt": "only while every earlier attempt fails the v14 A2 checks (CORRECT, v11 A2 filters, >=256 tokens)",
    "chosen": "first passing attempt; only it may be an A2 row and only then is the note a B candidate",
    "failed_attempts": "never training rows"}
SINGLE_ROUTES_V15 = {"RL": "FULL_iff_item_has_image", "A1": "FULL_iff_item_has_image",
                     "A2_B": "FULL_iff_item_has_image"}
ROLLOUT_ADAPTER_V15 = "vllm-lora-unmerged+vision-hooks"
SINGLE_TOKEN_SCALE_V10 = 256.0
SINGLE_TOKEN_SCALE_V11 = 1024.0
SINGLE_ROW_TOKEN_CAP_V11 = 2048
SINGLE_ROW_TOKEN_CAP_EXEMPT_V11 = ("LOOP",)
SINGLE_NEGATIVE_LOGPROB_V11 = "raw_full_vocab_log_softmax_T1_no_presence"
SINGLE_ADVANTAGE_VERSION_V11 = "penalty-table-v1"
SINGLE_ADVANTAGE_V11 = {"accepted": 1.0, "BARE_WRONG": -1.0, "NO_COMMITTED_ANSWER": -1.0, "LOOP": -1.0,
                        "REASONED_WRONG": -0.5, "TRUNCATED_NO_LOOP": -0.5, "other": 0.0}
SINGLE_PLAN_EXCEPTIONS_V11 = [
    "2026-11-05 plan: PG used only the stored sampling distribution q; v11 negative-advantage rows use the raw "
    "full-vocabulary log p (T=1, no presence penalty) so one-token supports still carry gradient",
    "2026-11-05 plan: the PG denominator was length-independent and fixed; v11 scales each row by min(1, 2048/n) "
    "so long rows do not dominate the block"]
SINGLE_SFT_WEIGHTS_V10 = {"A1": 0.05, "A2": 0.35, "B": 0.05}
SINGLE_SFT_WEIGHTING_V10 = {"A1": "per_kind_total", "A2": "per_row", "B": "per_kind_total"}
SINGLE_ADVANTAGE_VERSION_V12 = "penalty-table-v2"
SINGLE_ADVANTAGE_V12 = {"accepted": 1.0, "BARE_WRONG": -1.0, "NO_COMMITTED_ANSWER": -1.0, "LOOP": -1.0,
                        "REASONED_WRONG": 0.0, "TRUNCATED_NO_LOOP": -0.5, "CONTAINED_OPEN": 0.0, "other": 0.0}
SINGLE_CONTAINED_OPEN_RULE = ("open item (no options), T1 INCORRECT and stopped with a committed (extracted) answer whose "
                              "tokens contain the gold's answer tokens contiguously while the sequences differ; gate v7 "
                              "order: accepted, LOOP, TRUNCATED_NO_LOOP, NO_COMMITTED_ANSWER, CONTAINED_OPEN, BARE_WRONG, "
                              "REASONED_WRONG; gold = the source-bound provenance label")
SINGLE_SFT_WEIGHTS_V12 = SINGLE_SFT_WEIGHTS_V10 | {"H": 0.02}
SINGLE_SFT_WEIGHTING_V12 = SINGLE_SFT_WEIGHTING_V10 | {"H": "per_kind_total"}
SINGLE_ADVANTAGE_VERSION_V13 = "penalty-table-v3"
SINGLE_ADVANTAGE_V13 = {"accepted_praised": 1.0, "accepted": 0.0, "BARE_WRONG": -1.0, "NO_COMMITTED_ANSWER": -1.0,
                        "LOOP": -1.0, "TRUNCATED_NO_LOOP": -1.0, "REASONED_WRONG": 0.0, "CONTAINED_OPEN": 0.0, "other": 0.0}
# The pinned Qwen3-VL tokenizer's two single tokens that decode to "Wait" ("Wait", " Wait"); prepare re-checks them.
SINGLE_SPIRAL_V13: dict[str, Any] = {"version": "spiral-window-v1", "marker": "Wait", "marker_token_ids": [13824, 14190],
                     "marker_count": 3, "fallback_start": 1024, "window_tokens": 2048,
                     "penalties": ["LOOP", "TRUNCATED_NO_LOOP"], "logprob": "raw", "row_weight": 1.0}
SINGLE_PRAISE_V13 = {"version": "praise-v1", "min_tokens": 256, "applies_to": ["PG", "A1"]}


def spiral_window(token_ids: list[int]) -> list[int]:
    """v13: [start, end) of a cut-off or looping T1's penalized tokens. start = the index of the third marker token,
    or min(1024, n // 2) with fewer than three markers; end = min(n, start + 2048)."""
    if not isinstance(token_ids, list) or not token_ids or any(type(token) is not int for token in token_ids):
        raise ValueError("single protocol: spiral window needs the sampled token IDs")
    rule, count = SINGLE_SPIRAL_V13, len(token_ids)
    marks = [index for index, token in enumerate(token_ids) if token in rule["marker_token_ids"]]
    start = marks[rule["marker_count"] - 1] if len(marks) >= rule["marker_count"] else min(rule["fallback_start"], count // 2)
    return [start, min(count, start + rule["window_tokens"])]


SINGLE_ADVANTAGE_VERSION_V14 = "penalty-table-v4"
SINGLE_ADVANTAGE_V14 = SINGLE_ADVANTAGE_V13 | {"tail_praised": 1.0}
# "reasons" are _single_reward_decision reasons: "T1_INELIGIBLE_" + a v4.target_problem code.
SINGLE_TAIL_PRAISE_V14: dict[str, Any] = {"version": "tail-praise-v1",
                          "reasons": ["T1_INELIGIBLE_REPETITION", "T1_INELIGIBLE_BLOCK_REPETITION", "T1_INELIGIBLE_LOOP",
                                      "T1_INELIGIBLE_TOO_LONG"],
                          "min_tokens": 256, "marker_token_ids": [13824, 14190], "window_tokens": 1024,
                          "min_window_tokens": 32, "start": "max(last_marker_index + 1, n - window_tokens)",
                          "window_filter": ["has_repetition", "has_block_repetition"],
                          "answer_denial": "no_tail_when_the_whole_answer_denies_the_question", "logprob": "stored_support",
                          "row_weight": 1.0, "applies_to": ["PG"]}
SINGLE_A2_MIN_TOKENS_V14 = 256
SINGLE_B_ROWS_V14 = "program_accepted_note_whose_resolve_passes_the_v11_a2_checks"
SINGLE_IMAGE_V14 = {"min_pixels": 1003520, "max_pixels": 4014080, "preprocessing": IMAGE["preprocessing"]}  # B2's budget
SINGLE_ANSWER_CONTRACTS_V14 = {"mcq": TURN1_EVAL_CONTRACT, "open": SCORER_V2}


def tail_window(token_ids: list[int]) -> list[int] | None:
    """v14: [start, n) of the praised conclusion of a correct T1 that is no A1 target, or None.

    start = max(index of the last marker token + 1, n - 1024). None when there is no marker token, when the window
    has fewer than 32 tokens, or when the window itself repeats (contract.has_repetition or v4.has_block_repetition
    on the window's token IDs)."""
    if not isinstance(token_ids, list) or not token_ids or any(type(token) is not int for token in token_ids):
        raise ValueError("single protocol: tail window needs the sampled token IDs")
    from .contract import has_repetition
    from .v4 import has_block_repetition

    rule = SINGLE_TAIL_PRAISE_V14
    count = len(token_ids)
    markers = [index for index, token in enumerate(token_ids) if token in rule["marker_token_ids"]]
    if not markers:
        return None
    start = max(markers[-1] + 1, count - rule["window_tokens"])
    window = token_ids[start:]
    if len(window) < rule["min_window_tokens"] or has_repetition(window) or has_block_repetition(window):
        return None
    return [start, count]


SINGLE_BASELINE_METHOD = "fixed-source-mean-v1"
SINGLE_BASELINE_TABLE = Path(__file__).with_name("source_baseline_v10.json")
SINGLE_REWARD_GATE_V5 = "single-t1-program-gate-v5"
SINGLE_REWARD_GATE = "single-t1-program-gate-v6"
SINGLE_REWARD_GATE_V7 = "single-t1-program-gate-v7"
SINGLE_REWARD_GATE_V8 = "single-t1-program-gate-v8"
SINGLE_REWARD_GATE_V9 = "single-t1-program-gate-v9"
SINGLE_NOTE_MAX_TOKENS = 1024
SINGLE_ANSWER_UNION = f"{TURN1_EVAL_CONTRACT}|{TRAINING_ANSWER_CONTRACT}"  # v7 open contract, kept for old callers
SINGLE_OPEN_SCORER = SCORER
SINGLE_REWARD_VALUES = {"accepted": 1, "BARE_WRONG": -1, "NO_COMMITTED_ANSWER": -1, "LOOP": -1,
                        "REASONED_WRONG": -1, "TRUNCATED_NO_LOOP": -1, "other": 0}
SINGLE_REPLAY = "stored-generation-support-v1"
SINGLE_REPLAY_BLOCK_MEAN_ATOL = 0.05
SINGLE_REPLAY_ROW_MEAN_ATOL = 0.1
SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS = 64
SINGLE_REPLAY_MAX_ATOL = 3.0
SINGLE_REPLAY_ROW_FAILURE = "exclude_row_from_rl_kl"
SINGLE_REPLAY_MAX_EXCLUDED_ROWS = 2
SINGLE_FORMAT_ANCHOR_POSITIONS = 4
SINGLE_FORMAT_ANCHOR_BETA = 0.5
SINGLE_ANSWER_CONTRACTS = {"mcq": TURN1_EVAL_CONTRACT, "open": SINGLE_OPEN_SCORER}
SINGLE_NOTE_CAUSE_RULE = {
    "reject_cause": ["NONE"], "unaccepted_statuses": ["INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"],
    "unaccepted_cause": "NO_FINAL_ANSWER", "reasons": ["NOTE_CAUSE_NONE", "NOTE_CAUSE_MISMATCH"],
    "applies_to": "B rows only; the note's lesson still drives its one re-solve"}
SINGLE_CONTRACT_HASHES = frozenset(("source", "dataset", "order", "data_admission"))
SINGLE_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


def single_prompts_sha256() -> str:
    return _digest({"note": NOTE_INSTRUCTION_SINGLE, "note_feedback": SINGLE_VERDICT_LABELS, "resolve_header": RESOLVE_HEADER,
                    "resolve_instruction": RESOLVE_INSTRUCTION_SINGLE,
                    "note_grader_read": [GRADER_READ_LINE, GRADER_READ_NONE], "note_reference": REFERENCE_HEADER})


def single_baseline_contract() -> dict:
    """The v10 RL baseline pin: method plus the SHA256 of the shipped per-source table bytes."""
    return {"method": SINGLE_BASELINE_METHOD, "table_sha256": hashlib.sha256(SINGLE_BASELINE_TABLE.read_bytes()).hexdigest(),
            "unknown_source": "global_mean"}


def single_source_baseline(problem_id: str, pin: dict) -> dict:
    """b(source) for one RL row from the shipped table; the table bytes must still match the sealed pin."""
    raw = SINGLE_BASELINE_TABLE.read_bytes()
    if pin != {"method": SINGLE_BASELINE_METHOD, "table_sha256": hashlib.sha256(raw).hexdigest(),
               "unknown_source": "global_mean"}:
        raise ValueError("single baseline: shipped source table differs from the sealed baseline pin")
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("single baseline: problem ID required")
    table, source = json.loads(raw), problem_id.split(":")[0]
    value = table["sources"][source]["mean"] if source in table["sources"] else table["global_mean"]
    return {"method": SINGLE_BASELINE_METHOD, "table_sha256": pin["table_sha256"], "source": source, "value": float(value)}


def single_blocks(problem_ids: list[str], *, pilot_count: int = 300) -> list[list[str]]:
    """Pilot and continuation close separate 16-item blocks; every original ID occurs once."""
    if (not problem_ids or any(not isinstance(pid, str) or not pid for pid in problem_ids)
            or len(set(problem_ids)) != len(problem_ids) or type(pilot_count) is not int or pilot_count < 1):
        raise ValueError("single protocol: unique nonempty problem IDs and positive pilot count required")
    boundary = min(pilot_count, len(problem_ids))
    return [part[start:start + PER_UPDATE_ITEMS_V2]
            for part in (problem_ids[:boundary], problem_ids[boundary:])
            for start in range(0, len(part), PER_UPDATE_ITEMS_V2)]


def plain_protocol_single(precision: str, total_slots: int, *, contracts: dict[str, str],
                          token_scale: float = SINGLE_TOKEN_SCALE_V11, beta: float = 0.02,
                          sft_weights: dict[str, float] | None = None, note_reference: dict | None = None,
                          version: str = VERSION_SINGLE, schedule_steps: int | None = None) -> dict:
    """Explicit local-data/source pins and an on-policy, one-response objective; no execution permission (v11, or
    v12 / v13 / v14 / v15 with ``version=VERSION_SINGLE_V12`` / ``..._V13`` / ``..._V14`` / ``..._V15``).

    ``schedule_steps`` (v14 and v15 only, required there): the optimizer steps the LR decays over, 1..total_slots; it replaces
    ``optimizer.total_steps`` (v11-v13 keep total_slots there) and is recorded as ``training_contract.schedule_steps``."""
    import math

    if version not in (VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13, VERSION_SINGLE_V14, VERSION_SINGLE_V15):
        raise ValueError("single protocol: version must be v11, v12, v13, v14 or v15")
    v15 = version == VERSION_SINGLE_V15
    v14 = version in (VERSION_SINGLE_V14, VERSION_SINGLE_V15)  # v15 keeps every v14 branch below
    v13 = version in (VERSION_SINGLE_V13, VERSION_SINGLE_V14, VERSION_SINGLE_V15)  # v14/v15 keep every v13 branch
    if (schedule_steps is None) == v14 or (v14 and (
            type(schedule_steps) is not int or not 1 <= schedule_steps <= total_slots)):
        raise ValueError("single protocol: schedule_steps belongs to v14/v15 and must be an integer in 1..total_slots")
    v12 = version == VERSION_SINGLE_V12  # H rows are v12 only; v13-v15 share v12's note filters and B rule below
    weights = dict(SINGLE_SFT_WEIGHTS_V12 if v12 else SINGLE_SFT_WEIGHTS_V10) if sft_weights is None else dict(sft_weights)
    reference = {"version": "none"} if note_reference is None else note_reference
    if token_scale != SINGLE_TOKEN_SCALE_V11:
        raise ValueError(f"single protocol: v11 fixes the PG token scale at {SINGLE_TOKEN_SCALE_V11}")
    if not isinstance(reference, dict) or not isinstance(reference.get("version"), str):
        raise ValueError("single protocol: note_reference must be a dict with a string version")
    if (set(contracts) != SINGLE_CONTRACT_HASHES or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in contracts.values())):
        raise ValueError("single protocol: explicit source, dataset, order and data-admission SHA256 required")
    if (isinstance(token_scale, bool) or not isinstance(token_scale, (int, float))
            or not math.isfinite(token_scale) or token_scale <= 0
            or isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0
            or set(weights) != set(SINGLE_SFT_WEIGHTS_V12 if v12 else SINGLE_SFT_WEIGHTS_V10) or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in weights.values())):
        raise ValueError("single protocol: finite fixed normalization and nonnegative weights required")
    body = {key: value for key, value in plain_protocol_v2(precision, total_slots).items() if key != "sha256"}
    body |= {"version": VERSION_SINGLE, "sampling": dict(SINGLE_SAMPLING),
             "prompts_sha256": single_prompts_sha256(),
             "training_contract": {
                 "version": VERSION_SINGLE, "hashes": dict(contracts), "master_seed": MASTER_SEED, "per_update_items": PER_UPDATE_ITEMS_V2,
                 "generation_counts": {"t1": 1, "note": 1, "resolve": 1}, "sampling": dict(SINGLE_SAMPLING),
                 "max_tokens": {"t1": MAX_NEW_TOKENS, "resolve": MAX_NEW_TOKENS}, "note_max_tokens": SINGLE_NOTE_MAX_TOKENS,
                 "rl": {"method": "reinforce", "baseline": "none",
                        "advantage": {"version": SINGLE_ADVANTAGE_VERSION_V11, "values": dict(SINGLE_ADVANTAGE_V11)},
                        "token_scale": float(token_scale), "row_token_cap": SINGLE_ROW_TOKEN_CAP_V11,
                        "row_token_cap_exempt": list(SINGLE_ROW_TOKEN_CAP_EXEMPT_V11),
                        "negative_logprob": SINGLE_NEGATIVE_LOGPROB_V11, "beta": float(beta), "reward": SINGLE_REWARD_GATE,
                        "reward_values": dict(SINGLE_REWARD_VALUES),
                        "kl": "raw_F0_to_current_on_all_completed_T1_prefixes",
                        "format_anchor": {"positions": SINGLE_FORMAT_ANCHOR_POSITIONS, "beta": SINGLE_FORMAT_ANCHOR_BETA},
                        "plan_exceptions": list(SINGLE_PLAN_EXCEPTIONS_V11)},
                 "note_reference": dict(reference),
                 "sft_weights": weights, "sft_weighting": dict(SINGLE_SFT_WEIGHTING_V10),
                 "sft_row": "mean_original_target_token_nll",
                 "length_policy": {"min_target_tokens": MIN_TARGET_TOKENS, "note_min_target_tokens": 1,
                                   "max_target_tokens": MAX_TARGET_TOKENS,
                                   "content_acceptance": "program_checks_only_content_unverified"},
                 "answer_contracts": dict(SINGLE_ANSWER_CONTRACTS),
                 "answer_scorer": {"id": SINGLE_OPEN_SCORER, "rules": list(NORMALIZATION_RULES),
                                   "suffix_stop": sorted(SUFFIX_STOP), "number_words": list(NUMBER_WORDS),
                                   "aliases": ALIASES, "aliases_sha256": ALIASES_SHA256,
                                   "committed_answer_never_no_committed_penalty": True},
                 "program_filters": {"note_cause": SINGLE_NOTE_CAUSE_RULE, "lesson_stem": LESSON_STEM_RULE,
                                     "answer_denial": {"phrases": list(DENIAL_PHRASES), "match": "word_start_after_markdown_emphasis_removed",
                                                       "blocks": ["T1_reward", "A1", "A2"]},
                                     "a2_review_references": {"phrases": list(_REVIEW_REFERENCES),
                                                              "match": "word_boundary"}},
                 "renormalize_missing_kind": False, "replay": "original_actions_same_live_policy_once",
                 "replay_support": {"method": SINGLE_REPLAY, "block_mean_atol": SINGLE_REPLAY_BLOCK_MEAN_ATOL,
                                    "row_mean_atol": SINGLE_REPLAY_ROW_MEAN_ATOL,
                                    "row_mean_min_tokens": SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS,
                                    "row_max_atol": SINGLE_REPLAY_MAX_ATOL,
                                    "row_failure": SINGLE_REPLAY_ROW_FAILURE,
                                    "max_excluded_rows": SINGLE_REPLAY_MAX_EXCLUDED_ROWS},
                 "zero_signal": "consume_slot_without_optimizer_or_decay", "total_slots": total_slots,
                 "pilot_boundary": 300, "pilot_main_optimizer": "continuous",
                 "routes": {"RL": "LM", "A1": "LM", "A2_B": "accepted_note_self_report"}}}
    if v12 or v13:
        from .contract import (_ANSWER_DENIAL_V12, HUMAN_FINAL_LINE, LESSON_LEAK_RULE_V2, LESSON_LEAK_VERSION_V2,
                               ANSWER_TOKEN_RULE)
        from .references import HUMAN_SOLUTION_FIELDS, HUMAN_SOLUTION_VERSION, human_solution_sha256

        contract = body["training_contract"]
        body["version"] = contract["version"] = version
        contract["rl"] |= {"reward": SINGLE_REWARD_GATE_V7, "reward_values": dict(SINGLE_REWARD_VALUES, CONTAINED_OPEN=0),
                           "advantage": {"version": SINGLE_ADVANTAGE_VERSION_V12, "values": dict(SINGLE_ADVANTAGE_V12)},
                           "contained_open": {"rule": SINGLE_CONTAINED_OPEN_RULE, "tokens": ANSWER_TOKEN_RULE}}
        contract["program_filters"] |= {
            "lesson_stem": LESSON_LEAK_RULE_V2, "lesson_leak": LESSON_LEAK_VERSION_V2,
            "b_answer_denial": {"pattern": _ANSWER_DENIAL_V12.pattern, "match": "lowercase_whitespace_collapsed"}}
        contract["b_rows"] = "every_program_accepted_note_whatever_the_resolve_result"
    if v13:
        contract["rl"] |= {"reward": SINGLE_REWARD_GATE_V8,
                           "advantage": {"version": SINGLE_ADVANTAGE_VERSION_V13, "values": dict(SINGLE_ADVANTAGE_V13)},
                           "spiral_window": copy.deepcopy(SINGLE_SPIRAL_V13), "praise": copy.deepcopy(SINGLE_PRAISE_V13)}
    if v14:
        body["image"] = dict(SINGLE_IMAGE_V14)
        body["optimizer"] = body["optimizer"] | {"total_steps": schedule_steps}
        contract["schedule_steps"] = schedule_steps
        contract["rl"] |= {"reward": SINGLE_REWARD_GATE_V9,
                           "advantage": {"version": SINGLE_ADVANTAGE_VERSION_V14, "values": dict(SINGLE_ADVANTAGE_V14)},
                           "tail_praise": copy.deepcopy(SINGLE_TAIL_PRAISE_V14)}
        contract["b_rows"] = SINGLE_B_ROWS_V14
        contract["a2_min_tokens"] = SINGLE_A2_MIN_TOKENS_V14
        contract["answer_contracts"] = dict(SINGLE_ANSWER_CONTRACTS_V14)
        contract["answer_scorer"] = contract["answer_scorer"] | {"id": SCORER_V2, "rules": list(NORMALIZATION_RULES_V2),
                                                                 "unit_abbreviations": list(UNIT_ABBREVIATIONS)}
    if v15:
        contract["generation_counts"] = {"t1": 1, "note": 1, "resolve": len(SINGLE_RESOLVE_STAGES_V15)}
        contract["resolve_attempts"] = copy.deepcopy(SINGLE_RESOLVE_V15)
        contract["routes"] = dict(SINGLE_ROUTES_V15)
        body["rollout_adapter"] = ROLLOUT_ADAPTER_V15
    # training_contract.total_slots stays total_slots; spiral_window and praise stay v13's.
    if v12:
        contract["sft_weighting"] = dict(SINGLE_SFT_WEIGHTING_V12)
        contract["human_solution"] = {
            "version": HUMAN_SOLUTION_VERSION, "fields": {k: list(v) for k, v in HUMAN_SOLUTION_FIELDS.items()},
            "fields_sha256": human_solution_sha256(), "target": "lecture?\\n\\nsolution\\n\\n" + HUMAN_FINAL_LINE,
            "mcq_answer": "{gold letter}. {gold option text}", "min_target_tokens": MIN_TARGET_TOKENS,
            "end_token": "appended_like_sampled_rows", "items": "every_block_item_of_a_mapped_source",
            "excluded_from": ["PG", "KL", "format_anchor", "replay"]}
        contract["routes"] = contract["routes"] | {"H": "LM"}
    return body | {"sha256": _digest(body)}


VERSION_CONTENT_V16 = "error-note-content-v16"
CONTENT_WEIGHTS_V16 = {"O": 1.0, "J": 0.0, "C": 0.0, "N": 0.0, "A1": 0.0}
CONTENT_GENERATIONS_V16 = {"t1": 1, "note": 1, "resolve": 1}
CONTENT_VERIFIER_V16 = "verified-content-v16"


def content_prompts_sha256() -> str:
    from .content_v16 import GRAPH_NEIGHBOR_INSTRUCTION, O_INSTRUCTION
    from .self_note_v16 import NOTE_INSTRUCTION_V16, RESOLVE_INSTRUCTION_V16

    return _digest({"observation": O_INSTRUCTION, "neighbors": GRAPH_NEIGHBOR_INSTRUCTION,
                    "note": NOTE_INSTRUCTION_V16,
                    "resolve": RESOLVE_INSTRUCTION_V16})


def plain_protocol_content_v16(precision: str, total_blocks: int, *, contracts: dict[str, str],
                               source_weights: dict[str, float], weights: dict[str, float] | None = None,
                               stage_a_blocks: int = 0) -> dict:
    """A fresh-F0, content-only protocol; a sealed protocol grants no permission to run a model."""
    import math

    chosen = dict(CONTENT_WEIGHTS_V16 if weights is None else weights)
    if (set(contracts) != SINGLE_CONTRACT_HASHES or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in contracts.values())):
        raise ValueError("v16 protocol: source/dataset/order/admission SHA256 pins required")
    if (type(stage_a_blocks) is not int or type(total_blocks) is not int
            or not 0 <= stage_a_blocks <= total_blocks or total_blocks < 1
            or set(chosen) != set(CONTENT_WEIGHTS_V16) or not source_weights
            or any(not isinstance(key, str) or not key for key in source_weights)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0
                   for value in (*chosen.values(), *source_weights.values()))
            or sum(source_weights.values()) <= 0 or sum(chosen.values()) <= 0):
        raise ValueError("v16 protocol: positive block count and finite nonnegative fixed weights required")
    body = {key: value for key, value in plain_protocol_v2(precision, total_blocks).items() if key != "sha256"}
    prompts = content_prompts_sha256()
    body |= {"version": VERSION_CONTENT_V16, "image": dict(SINGLE_IMAGE_V14),
             "sampling": dict(SINGLE_SAMPLING), "rollout_adapter": ROLLOUT_ADAPTER_V15,
             "prompts_sha256": prompts,
             "optimizer": body["optimizer"] | {"warmup_steps": 4, "total_steps": total_blocks,
                                                  "lr": {"L": 5e-5, "M": 1e-5, "V": 1e-5}},
             "training_contract": {
                 "version": VERSION_CONTENT_V16, "hashes": dict(contracts), "master_seed": MASTER_SEED,
                 "initial_policy": "fresh_F0", "method": "verified_content_sft", "pg_weight": 0.0,
                 "generation_counts": dict(CONTENT_GENERATIONS_V16), "sampling": dict(SINGLE_SAMPLING),
                 "max_tokens": {"t1": MAX_NEW_TOKENS, "note": SINGLE_NOTE_MAX_TOKENS, "resolve": MAX_NEW_TOKENS},
                 "weights": chosen, "source_weights": dict(source_weights), "stage_a_blocks": stage_a_blocks,
                 "stage_a_weights": dict(CONTENT_WEIGHTS_V16),
                 "verifier_version": CONTENT_VERIFIER_V16, "prompts_sha256": prompts,
                 "total_slots": total_blocks, "schedule_steps": total_blocks,
                 "r_enabled": False, "renormalize_missing_kind": False,
                 "self_target_logprob": "stored_support_logq", "authored_target_logprob": "raw_full_vocabulary_ce",
                 "zero_signal": "consume_slot_and_schedule_without_optimizer",
                 "routes": {"image": "FULL", "text": "LM"},
                 "runtime_gates": {"cloud": "sealed_local_functional_ready_and_explicit_launch",
                                   "phase2": "sealed_BF16_O_effect_PASS",
                                   "N": "sealed_32_wrong_T1_yield_ge_0.10"}}}
    return body | {"sha256": _digest(body)}


def validate_content_protocol_v16(protocol: dict) -> None:
    """Rebuild the complete contract so resealing a relaxed/tampered field never admits it."""
    try:
        contract = protocol["training_contract"]
        expected = plain_protocol_content_v16(protocol["precision"], contract["total_slots"],
            contracts=contract["hashes"], source_weights=contract["source_weights"], weights=contract["weights"],
            stage_a_blocks=contract["stage_a_blocks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v16 protocol: invalid content protocol") from exc
    if protocol != expected:
        raise ValueError("v16 protocol: invalid content protocol")


def content_contract_for_block(protocol: dict, slot: int) -> dict:
    """Derive the admitted block's fixed coefficients without changing the sealed run protocol."""
    validate_content_protocol_v16(protocol)
    contract = protocol["training_contract"]
    if type(slot) is not int or not 0 <= slot < contract["total_slots"]:
        raise ValueError("v16 protocol: block outside the sealed schedule")
    stage = "A" if slot < contract["stage_a_blocks"] else "B"
    return contract | {"weights": dict(contract["stage_a_weights"] if stage == "A" else contract["weights"]),
                       "stage_b_weights": dict(contract["weights"]), "active_stage": stage, "block_id": slot,
                       "protocol_sha256": protocol["sha256"]}


# v16b is a separate contract; previous builders above remain byte-for-byte unchanged.
VERSION_V16B = "error-note-v16b"
TRAIN_SAMPLING_V16B = {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "presence_penalty": 1.5}
EVAL_SAMPLING_V16B = dict(SINGLE_SAMPLING)
PROBABILITY_SPACE_V16B = "processed-full-vocab-v1"
MAX_TOKENS_V16B = {"t1": 8192, "obs": 1024, "note": 1024, "resolve": 8192, "prep": 8192, "monitor": 8192}
REWARD_V16B = {"CORRECT": 1.0, "INCORRECT": 0.0, "NO_FINAL": 0.0, "MULTIPLE": 0.0,
              "CAP_OR_LOOP": {"reward": 0.0, "pg": False}, "UNVERIFIABLE": None}
BASELINE_V16B = {"rule": "shrunk-source-mean-v1", "a": 2, "pseudo_count": 8, "window_rows": 64,
                "obs_window": 32, "obs_init": "measured_prep_or_first_committed_layer_block",
                "snapshot": "request_admission_latest_committed_history", "eta1_history_lag": 2,
                "missing_obs_init": "frozen_zero_advantage"}
PG_V16B = {"token_scale": 1024, "items_per_block": 16, "row_cap": 2048, "obs_weight": 0.25,
          "ntf_tau_neg": 0.5, "is_cap": 2.0, "neg_row_logratio_mask": 0.3,
          "hold_block_logratio": 0.5, "hold_clip_frac": 0.10, "hold_mixed_fraction": 0.05,
          "lag_zero_replay_mean_atol": SINGLE_REPLAY_BLOCK_MEAN_ATOL}
SCHEDULE_V16B = {"eta": 1, "note_carry": 1, "resolve_carry": 3, "rl_blocks": 188,
                "last_block_items": 8, "finish_updates_max": 3, "time_cut": None}
PREP_V16B = {"items": 512, "select": "disjoint_sealed_prep_manifest", "self_max": 160,
            "human_max": 128, "self_tokens": [256, 4096], "human_min_tokens": 32,
            "overlap_with_rl": False, "overlap_with_monitor": False}
GUARD_V16B = {"start_block": 36, "ref_blocks": 24, "recent_blocks": 12, "acc_drop": 0.15,
             "count_rule": "recent >= 2*ref and recent >= ref + 4*sqrt(ref)",
             "collapse_len": {"window": 6, "ratio": 0.5, "consecutive_windows": 2, "start_block": 12},
             "collapse_short": {"tokens": 32, "count": 12, "blocks": 2},
             "tail_rate": {"reference_blocks": 4, "multiplier": 2.0, "consecutive": 2},
             "is_hold_consecutive": 2, "false_alarm_max": 0.02}
FUNDING_V16B = {"reserve_usd": 1.5, "post_train_hours": 5.65, "recharge_wait_min": 30}
SFT_V16B = {"PREP_SELF": 1.0, "PREP_HUMAN": 1.0, "C": 0.35, "N": 0.05}
ANCHOR_V16B = {"positions": 4, "beta": 0.5, "rl_denominator": 16,
              "prep_denominator": "actual_rows", "reference": "fresh_F0"}
CHECKPOINT_V16B = {"every_updates": 8, "always_on": ["prep_end", "guard_stop", "funding_stop", "final"],
                  "drain_before_save": True, "keep": ["latest", "prep_end"]}
V16B_CONTRACT_HASHES = frozenset(("source", "dataset", "order", "prep_rows", "monitor"))
SMOKE_IMAGE_V16B = IMAGE | {"max_pixels": 262144}


def v16b_prompts_sha256() -> str:
    from .v16b_rows import NOTE_INSTRUCTION_V16B, OBS_FORMAT_V16B, RESOLVE_INSTRUCTION_V16B
    from .content_v16 import GRAPH_NEIGHBOR_INSTRUCTION, O_INSTRUCTION

    return _digest({"note": NOTE_INSTRUCTION_V16B, "resolve": RESOLVE_INSTRUCTION_V16B,
                    "obs_format": OBS_FORMAT_V16B, "obs": dict(O_INSTRUCTION),
                    "graph_neighbors": GRAPH_NEIGHBOR_INSTRUCTION})


def plain_protocol_v16b(precision: str, *, rl_blocks: int, prep_updates: int,
                        contracts: dict[str, str], image: dict | None = None,
                        local_smoke: bool = False) -> dict:
    """Seal the whole schedule after prep admission; smoke is never a cloud protocol."""
    if (type(local_smoke) is not bool or type(rl_blocks) is not int or type(prep_updates) is not int
            or not 0 <= prep_updates <= 18 or rl_blocks != (4 if local_smoke else 188)
            or precision != ("nf4" if local_smoke else "bf16")):
        raise ValueError("v16b protocol: full BF16 188 blocks or explicit bounded NF4 smoke required")
    if (set(contracts) != V16B_CONTRACT_HASHES or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in contracts.values())):
        raise ValueError("v16b protocol: source/dataset/order/prep_rows/monitor SHA256 pins required")
    chosen_image = dict(SMOKE_IMAGE_V16B if local_smoke else SINGLE_IMAGE_V14)
    if image is not None and image != chosen_image:
        raise ValueError("v16b protocol: image policy differs from the sealed execution mode")
    if local_smoke and prep_updates != 1:
        raise ValueError("v16b protocol: local smoke requires exactly one preparation update")
    total = prep_updates + rl_blocks + 3
    body = copy.deepcopy({key: value for key, value in plain_protocol_v2(precision, total).items() if key != "sha256"})
    prompts = v16b_prompts_sha256()
    schedule = SCHEDULE_V16B | {"rl_blocks": rl_blocks, "last_block_items": 2 if local_smoke else 8}
    limits = {kind: 256 for kind in MAX_TOKENS_V16B} if local_smoke else dict(MAX_TOKENS_V16B)
    contract = {
        "version": VERSION_V16B, "hashes": dict(contracts), "master_seed": MASTER_SEED,
        "initial_policy": "fresh_F0", "method": "single_response_rl_verified_self_sft",
        "generation_counts": {"t1": 1, "obs": 1, "note": 1, "resolve": 1},
        "sampling": dict(TRAIN_SAMPLING_V16B), "eval_sampling": dict(EVAL_SAMPLING_V16B),
        "probability_space": PROBABILITY_SPACE_V16B, "max_tokens": limits,
        "reward": REWARD_V16B, "baseline": BASELINE_V16B, "pg": PG_V16B,
        "schedule": schedule, "prep": PREP_V16B, "guard": GUARD_V16B,
        "funding": FUNDING_V16B, "sft_weights": SFT_V16B, "anchor": ANCHOR_V16B,
        "checkpoint": CHECKPOINT_V16B, "prep_updates": prep_updates, "rl_blocks": rl_blocks,
        "total_slots": total, "schedule_steps": total, "finish_updates": 3,
        "items_per_block": 2 if local_smoke else 16, "rl_items": 8 if local_smoke else 3000,
        "answer_contracts": dict(SINGLE_ANSWER_CONTRACTS_V14),
        "routes": {"image": "FULL", "text": "LM"}, "full_kl_beta": 0.0,
        "local_smoke": local_smoke, "cloud_allowed": not local_smoke,
        "prompts_sha256": prompts, "zero_signal": "consume_schedule_without_optimizer",
        "hold": "discard_all_gradients_consume_schedule_without_optimizer",
        "tail_observation": "top20_processed_logprobs_to_b2_flags_unresolved_ties_null",
        "stop_evaluation": "explicit_user_decision_required",
    }
    body |= {"version": VERSION_V16B, "image": chosen_image, "max_new_tokens": limits["t1"],
             "sampling": dict(TRAIN_SAMPLING_V16B), "eval_sampling": dict(EVAL_SAMPLING_V16B),
             "rollout_adapter": ROLLOUT_ADAPTER_V15, "prompts_sha256": prompts,
             "optimizer": body["optimizer"] | {"warmup_steps": 4, "total_steps": total,
                                                "lr": {"L": 5e-5, "M": 1e-5, "V": 1e-5}},
             "training_contract": copy.deepcopy(contract)}
    return body | {"sha256": _digest(body)}


def validate_protocol_v16b(protocol: dict) -> None:
    """Rebuild all fixed values, rejecting even a tampered protocol with a new digest."""
    try:
        contract = protocol["training_contract"]
        expected = plain_protocol_v16b(protocol["precision"], rl_blocks=contract["rl_blocks"],
            prep_updates=contract["prep_updates"], contracts=contract["hashes"], image=protocol["image"],
            local_smoke=contract["local_smoke"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v16b protocol: invalid sealed protocol") from exc
    if protocol != expected:
        raise ValueError("v16b protocol: invalid sealed protocol")


def v16b_contract_for_update(protocol: dict, global_slot: int, *, phase: str,
                            policy_sha256: str, expected_problem_ids: list[str],
                            expected_obs_problem_ids: list[str] | None = None,
                            note_problem_ids: list[str] | None = None,
                            resolve_problem_ids: list[str] | None = None,
                            baseline_receipts: dict | None = None,
                            reward_values: dict | None = None) -> dict:
    """Bind phase-local RL versions separately from the global LR/checkpoint slot."""
    validate_protocol_v16b(protocol)
    base = protocol["training_contract"]
    prep = base["prep_updates"]
    if (type(global_slot) is not int or not 0 <= global_slot < base["total_slots"]
            or phase not in ("prep", "rl") or (global_slot < prep) != (phase == "prep")
            or not isinstance(policy_sha256, str) or len(policy_sha256) != 64
            or any(char not in "0123456789abcdef" for char in policy_sha256)):
        raise ValueError("v16b protocol: invalid phase, slot or current policy hash")
    block = global_slot if phase == "prep" else global_slot - prep
    return copy.deepcopy(base) | {
        "phase": phase, "global_slot": global_slot, "block_id": block,
        "policy_version": block if phase == "rl" else 0, "policy_sha256": policy_sha256,
        "expected_problem_ids": list(expected_problem_ids),
        "expected_obs_problem_ids": list(expected_obs_problem_ids or []),
        "note_problem_ids": list(note_problem_ids or []),
        "resolve_problem_ids": list(resolve_problem_ids or []),
        "baseline_receipts": copy.deepcopy(baseline_receipts or {}),
        "reward_values": copy.deepcopy(reward_values or {}), "protocol_sha256": protocol["sha256"],
    }


# v17 is additive: the builders and golden outputs above retain their original contracts.
VERSION_V17 = "error-note-v17"
V17_CONTRACT_HASHES = V16B_CONTRACT_HASHES
V17_SPECIFICATION_SHA256 = "0899453deb1b5dd3f351977589feb6a55d4d6a5cdfde0a754a73ee607c78cd27"
V17_IDENTIFIERS = {
    "row_schema": "error-note-v17-row-v1", "claim_schema": "v17-claim-ledger-v1",
    "capability_schema": "v17-capability-v1", "probability_space_rl": "processed-full-vocab-v1",
    "probability_space_sft": "raw-full-vocab-v1", "note_mask": "verified-corrected-value-content-only-v1",
    "obs_reward": "bounded-scope-f1-v1", "negative_t1_mask": "probability-le-0.5",
    "negative_obs_mask": "all-sampled-actions", "loop_signal": "confirmed-loop-negative-pg-v1",
    "loop_detector": "contract.is_loop(window=3000, ratio=0.12)",
    "loop_token_scope": "detected-window-no-probability-mask", "eval_adapter_serving": "unmerged-lora",
    "comparison_strata": "f0-termination-state-v1", "mixed_policy_action": "hold-any-mixed-source",
    "main_parent_denominator": 16, "prep_denominator": "actual-rows",
    "f0_main_positions": "stratified-64-bin-weighted-v1", "f0_prep_positions": "first-4",
    "f0_position_salt": "v17-f0-stratified-64-v1", "signal_gate_supported_parents": 32,
    "signal_gate_verified_repair_parents": 3, "training_concurrency_ceiling": 16,
    "training_kv_admission": "full-request-reservation-v1", "short_drift_warning_delta": 0.05,
    "short_drift_stop_min_delta": 0.15, "short_drift_stop_se_multiplier": 4,
    "eval_concurrency": 8, "eval_kv_capacity_margin": 0.10, "eval_refill": "continuous",
    "eval_resume": "immutable-completed-same-registration-v1",
    "eval_watchdog_source": "sealed-throughput-calibration", "total_timeout_seconds": None,
    "training_elapsed_cutoff": None, "target_total_hours": [14, 17],
}


def v17_prompts_sha256() -> str:
    from .v17_data import public_schema
    from .v17_rows import NOTE_INSTRUCTION_V17, OBS_FORMAT_V17, OBS_INSTRUCTION_V17, RESOLVE_INSTRUCTION_V17

    return _digest({"note": NOTE_INSTRUCTION_V17, "obs": OBS_INSTRUCTION_V17, "obs_format": OBS_FORMAT_V17,
                    "resolve": RESOLVE_INSTRUCTION_V17,
                    "observation_scopes": [public_schema(layer) for layer in ("geometry", "graph", "scene", "chart")]})


def plain_protocol_v17(precision: str, *, rl_blocks: int, prep_updates: int,
                       contracts: dict[str, str], image: dict | None = None,
                       local_smoke: bool = False) -> dict:
    """Seal v17 separately while sharing the unchanged F0, optimizer and schedule checks."""
    body = plain_protocol_v16b(precision, rl_blocks=rl_blocks, prep_updates=prep_updates,
                              contracts=contracts, image=image, local_smoke=local_smoke)
    del body["sha256"]
    prompts = v17_prompts_sha256()
    contract = body["training_contract"]
    contract.update(copy.deepcopy(V17_IDENTIFIERS))
    contract.update({
        "version": VERSION_V17, "specification_sha256": V17_SPECIFICATION_SHA256,
        "prompts_sha256": prompts, "monitor_items": 200, "note_input_max_tokens": 16384,
        "measure_component_gradients": False, "eval_max_new_tokens": 32768, "eval_max_model_len": 40960,
        "sampling_controls": {"do_sample": True, "repetition_penalty": 1.0, "num_beams": 1},
        "sft_weights": {"PREP_SELF": 1.0, "PREP_HUMAN": 1.0, "A1": 0.20, "C": 0.35, "N": 0.05},
        "anchor": {"positions": 64, "beta": 0.02, "rl_denominator": 16,
                   "prep_denominator": "actual_rows", "reference": "fresh_F0",
                   "selection": "stratified-64-bin-weighted-v1", "prep_positions": 4,
                   "prep_beta": 0.5, "salt": "v17-f0-stratified-64-v1"},
        "funding": {"reserve_source": "sealed-throughput-calibration",
                    "recharge_wait_min": 30, "time_target_is_stop": False},
        "signal_gate": {"supported_parents": 32, "verified_repair_parents": 3,
                        "require_verified_content_tokens": True,
                        "require_nonzero_finite_n_gradient_and_saved_update": True,
                        "close_pending_note_resolve_before_check": True},
    })
    contract["pg"].update({"hold_mixed_fraction": 0.0, "negative_obs_mask": "all-sampled-actions"})
    contract["prep"].update({"self_tokens": [1, 8192], "self_acceptance": "verified-solution-v17",
                             "probability_space_sft": "raw-full-vocab-v1"})
    contract["guard"].update({
        "count_rule": "recent >= max(4, ceil(2*scaled_ref), ceil(scaled_ref + 4*sqrt(scaled_ref)))",
        "short_drift": {"reference_blocks": 24, "recent_blocks": 12, "start_block": 36,
                        "tokens": 32, "warning_delta": 0.05, "stop_min_delta": 0.15,
                        "stop_se_multiplier": 4, "denominator": "actual_parents"},
    })
    contract["reward"].update({
        "LOOP_CONFIRMED": {"reward": 0.0, "pg": True, "scope": "last-3000-chars",
                           "probability_mask": False, "row_cap": 2048},
        "LENGTH": {"reward": 0.0, "pg": False},
    })
    body.update({"version": VERSION_V17, "prompts_sha256": prompts,
                 "specification_sha256": V17_SPECIFICATION_SHA256})
    return body | {"sha256": _digest(body)}


def validate_protocol_v17(protocol: dict) -> None:
    """Reject resealed changes to v17's fixed values as well as digest tampering."""
    try:
        contract = protocol["training_contract"]
        expected = plain_protocol_v17(protocol["precision"], rl_blocks=contract["rl_blocks"],
            prep_updates=contract["prep_updates"], contracts=contract["hashes"], image=protocol["image"],
            local_smoke=contract["local_smoke"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v17 protocol: invalid sealed protocol") from exc
    if protocol != expected:
        raise ValueError("v17 protocol: invalid sealed protocol")


def _sealed_contract_for_slot(protocol: dict, global_slot: int, *, phase: str,
                           policy_sha256: str, expected_problem_ids: list[str],
                           expected_obs_problem_ids: list[str] | None = None,
                           note_problem_ids: list[str] | None = None,
                           resolve_problem_ids: list[str] | None = None,
                           baseline_receipts: dict | None = None,
                           reward_values: dict | None = None) -> dict:
    """Keep global slots and the two-policy delayed schedule identical to v16b."""
    base = protocol["training_contract"]
    prep = base["prep_updates"]
    if (type(global_slot) is not int or not 0 <= global_slot < base["total_slots"]
            or phase not in ("prep", "rl") or (global_slot < prep) != (phase == "prep")
            or not isinstance(policy_sha256, str) or len(policy_sha256) != 64
            or any(char not in "0123456789abcdef" for char in policy_sha256)):
        raise ValueError("v17 protocol: invalid phase, slot or current policy hash")
    block = global_slot if phase == "prep" else global_slot - prep
    return copy.deepcopy(base) | {
        "phase": phase, "global_slot": global_slot, "block_id": block,
        "policy_version": block if phase == "rl" else 0, "policy_sha256": policy_sha256,
        "expected_problem_ids": list(expected_problem_ids),
        "expected_obs_problem_ids": list(expected_obs_problem_ids or []),
        "note_problem_ids": list(note_problem_ids or []),
        "resolve_problem_ids": list(resolve_problem_ids or []),
        "baseline_receipts": copy.deepcopy(baseline_receipts or {}),
        "reward_values": copy.deepcopy(reward_values or {}), "protocol_sha256": protocol["sha256"],
    }

def v17_contract_for_update(protocol: dict, global_slot: int, **kwargs) -> dict:
    validate_protocol_v17(protocol)
    return _sealed_contract_for_slot(protocol, global_slot, **kwargs)


VERSION_V18 = "error-note-v18"
V18_CONTRACT_HASHES = V17_CONTRACT_HASHES
V18_IDENTIFIERS = {
    **V17_IDENTIFIERS,
    "row_schema": "error-note-v18-row-v1",
    "claim_schema": "v18-verified-self-quote-v1",
    "obs_reward": "binary-grounded-fact-v1",
    "obs_initial_baseline": 0.5,
    "obs_history_task": "v18-single-fact-observation-v1",
    "loop_detector": "confirmed-repeated-lines-or-contiguous-run-v18",
    "loop_token_scope": "repeated-byte-spans-only",
    "note_location_given_buckets": [3],
    "note_assignment_modulus": 4,
    "adapter_strength": 1.0,
    "eval_concurrency": 9,
    "training_data": "pinned-source-selection-v18",
}
V18_SFT_WEIGHTS = {"PREP_SELF": 1.0, "PREP_HUMAN": 1.0, "A1": 0.09, "C": 0.30, "N": 0.05}
V18_PROCESS_WEIGHTS = {"A1_verified": 0.18, "A1_span": 0.02, "C_span": 0.05}
V18_SPECIFICATION_SHA256 = _digest({"version": VERSION_V18, "identifiers": V18_IDENTIFIERS,
    "sft": V18_SFT_WEIGHTS, "process": V18_PROCESS_WEIGHTS, "parent_spec": V17_SPECIFICATION_SHA256})


def plain_protocol_v18(precision: str, *, rl_blocks: int, prep_updates: int, contracts: dict[str, str],
                       image: dict | None = None, local_smoke: bool = False) -> dict:
    from .v18_rows import prompts_sha256

    body = plain_protocol_v17(precision, rl_blocks=rl_blocks, prep_updates=prep_updates,
                              contracts=contracts, image=image, local_smoke=local_smoke)
    del body["sha256"]
    common = {"version": VERSION_V18, "specification_sha256": V18_SPECIFICATION_SHA256,
              "prompts_sha256": prompts_sha256()}
    c = body["training_contract"]
    c.update(copy.deepcopy(V18_IDENTIFIERS))
    c.update(common, sft_weights=dict(V18_SFT_WEIGHTS), process_weights=dict(V18_PROCESS_WEIGHTS),
             note_gradient_measurement="accepted-N-and-saved-update",
             signal_window_parents=512)
    c["baseline"] = c["baseline"] | {"obs_prior": 0.5, "obs_history_kind": c["obs_history_task"]}
    c["prep"]["self_acceptance"] = "verified-solution-v18"
    c["signal_gate"]["report_only"] = True
    c["reward"]["LOOP_CONFIRMED"] = {"reward": 0.0, "pg": True, "scope": c["loop_token_scope"]}
    c["reward"]["FORMAT_ONLY"] = {"reward": 0.0, "pg": "certified-span-only",
                                  "empty_span_action": "omit-pg"}
    body.update(common)
    return body | {"sha256": _digest(body)}


def validate_protocol_v18(protocol: dict) -> None:
    try:
        c = protocol["training_contract"]
        expected = plain_protocol_v18(protocol["precision"], rl_blocks=c["rl_blocks"],
            prep_updates=c["prep_updates"], contracts=c["hashes"], image=protocol["image"],
            local_smoke=c["local_smoke"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v18 protocol: invalid sealed protocol") from exc
    if protocol != expected:
        raise ValueError("v18 protocol: invalid sealed protocol")


def v18_contract_for_update(protocol: dict, global_slot: int, **kwargs) -> dict:
    validate_protocol_v18(protocol)
    return _sealed_contract_for_slot(protocol, global_slot, **kwargs)


VERSION_V18B = "error-note-v18b"
V18B_IDENTIFIERS = {
    **V18_IDENTIFIERS,
    "claim_schema": "v18b-single-grounded-correction",
    "observation_selection": "seeded-balanced-edge-label",
    "note_context": "public-observation-question-unit-and-actual-response",
    "external_monitor_blocks": [8, 16],
    "external_short_monitor_tokens": 64,
    "external_short_monitor_roles": ["monitor", "reasoning_monitor"],
    "external_accuracy_monitor_tokens": 8192,
    "checkpoint_copy_interval": 16,
}
V18B_SPECIFICATION_SHA256 = _digest({"version": VERSION_V18B,
    "identifiers": V18B_IDENTIFIERS, "parent_spec": V18_SPECIFICATION_SHA256})


def plain_protocol_v18b(precision: str, *, rl_blocks: int, prep_updates: int, contracts: dict[str, str],
                        image: dict | None = None, local_smoke: bool = False,
                        rehearsal_profile: str | None = None) -> dict:
    from .v18b_rows import prompts_sha256

    if rehearsal_profile is not None:
        from .v18b_rehearsal import rehearsal_protocol
        return rehearsal_protocol(precision, rl_blocks=rl_blocks, prep_updates=prep_updates,
            contracts=contracts, image=image, local_smoke=local_smoke, profile=rehearsal_profile)
    body = plain_protocol_v18(precision, rl_blocks=rl_blocks, prep_updates=prep_updates,
                              contracts=contracts, image=image, local_smoke=local_smoke)
    del body["sha256"]
    identity = {"version": VERSION_V18B, "specification_sha256": V18B_SPECIFICATION_SHA256,
                "prompts_sha256": prompts_sha256()}
    body.update(identity)
    body["training_contract"].update(copy.deepcopy(V18B_IDENTIFIERS), **identity)
    return body | {"sha256": _digest(body)}


def validate_protocol_v18b(protocol: dict) -> None:
    try:
        c = protocol["training_contract"]
        expected = plain_protocol_v18b(protocol["precision"], rl_blocks=c["rl_blocks"],
            prep_updates=c["prep_updates"], contracts=c["hashes"], image=protocol["image"],
            local_smoke=c["local_smoke"], rehearsal_profile=c.get("rehearsal_profile"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v18b: invalid sealed protocol") from exc
    if protocol != expected:
        raise ValueError("v18b: invalid sealed protocol")


def v18b_contract_for_update(protocol: dict, global_slot: int, **kwargs) -> dict:
    validate_protocol_v18b(protocol)
    return _sealed_contract_for_slot(protocol, global_slot, **kwargs)
