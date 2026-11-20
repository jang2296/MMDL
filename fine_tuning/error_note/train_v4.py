"""Version-4 training of one round: an RL term on group rollouts plus weighted SFT on A2/B/A1 rows, then save/reload.

Plan: claudedocs/plan_rl_sft_hybrid_v4_20261104.md sections 3.5-3.6 and 4.2(5). One optimizer step per order group of
PER_UPDATE_ITEMS_V2 items. The loss of a step is

    L = coefficient / n_groups * sum_g rl.group_term(g) + sum_k w_k * mean(NLL of the kind-k rows)

with the values fixed in protocol.plain_protocol_v4. A missing kind adds 0 (no renormalization). Round 1 starts from
F0; a later round loads the previous round's checkpoint-final (--init-run) and starts a fresh optimizer.

CLI: python -m fine_tuning.error_note.train_v4 train --rows R --build-root V --order O --round K --start S --limit N
         --arm {hybrid,rl-only} --precision {bf16,nf4} --model-path M --output DIR --code-commit 40HEX
         [--init-run PREVIOUS_RUN] [--token-budget T] [--gradient-checkpointing {on,off}] [--save-every 50]
         [--max-row-tokens K] [--resume]
     python -m fine_tuning.error_note.train_v4 dry-run (the train arguments, at most 10 items, no optimizer or save)
     python -m fine_tuning.error_note.train_v4 reload --run DIR --model-path M --precision {bf16,nf4}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import nullcontext
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.checkpoint import checkpoint as recompute

from fine_tuning.notebook_rl import checkpoint as ckpt
from fine_tuning.notebook_rl.input_preparation import digest, prepare_protocol_inputs, tensor_identity
from fine_tuning.notebook_rl.policy import PolicyBundle, assert_base_unchanged, build_policy
from fine_tuning.notebook_rl.prompts import user_message
from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged

from . import rl
from .contract import (CONTEXT_LIMIT, HUMAN_TARGET_SOURCE, IMAGE, KIND_HUMAN, MASTER_SEED, MIN_TARGET_TOKENS, ROUTE_FULL,
                       ROUTE_LM, VERDICT_LABELS, update_groups)
from .order import read_order
from .protocol import (ARM, PER_UPDATE_ITEMS_V2, SINGLE_FORMAT_ANCHOR_BETA, SINGLE_FORMAT_ANCHOR_POSITIONS,
                       SINGLE_REPLAY_BLOCK_MEAN_ATOL, SINGLE_REPLAY_MAX_ATOL, SINGLE_REPLAY_MAX_EXCLUDED_ROWS,
                       SINGLE_REPLAY_ROW_FAILURE, SINGLE_REPLAY_ROW_MEAN_ATOL, SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS,
                       SINGLE_ADVANTAGE_V11, SINGLE_ADVANTAGE_V12, SINGLE_ADVANTAGE_VERSION_V11,
                       SINGLE_ADVANTAGE_VERSION_V12, SINGLE_NEGATIVE_LOGPROB_V11,
                       SINGLE_REWARD_GATE, SINGLE_REWARD_GATE_V5, SINGLE_REWARD_GATE_V7, SINGLE_ROW_TOKEN_CAP_EXEMPT_V11,
                       SINGLE_ROW_TOKEN_CAP_V11, SINGLE_SFT_WEIGHTING_V10, SINGLE_SFT_WEIGHTING_V12,
                       SINGLE_SFT_WEIGHTS_V10, SINGLE_SFT_WEIGHTS_V12, SINGLE_TOKEN_SCALE_V10, SINGLE_TOKEN_SCALE_V11,
                       VERSION_SINGLE, VERSION_SINGLE_V8, VERSION_SINGLE_V9, VERSION_SINGLE_V10, VERSION_SINGLE_V12,
                       VERSION_SINGLE_V13, SINGLE_ADVANTAGE_V13, SINGLE_ADVANTAGE_VERSION_V13, SINGLE_PRAISE_V13,
                       SINGLE_REWARD_GATE_V8, SINGLE_SPIRAL_V13,
                       SINGLE_ADVANTAGE_V14, SINGLE_ADVANTAGE_VERSION_V14, SINGLE_A2_MIN_TOKENS_V14, SINGLE_B_ROWS_V14,
                       SINGLE_REWARD_GATE_V9, SINGLE_TAIL_PRAISE_V14, VERSION_SINGLE_V14,
                       SINGLE_RESOLVE_V15, SINGLE_ROUTES_V15, VERSION_SINGLE_V15,
                       VERSION_V4, plain_protocol_v4, single_baseline_contract)
from .train import (END_TOKEN, ENCODE_BUFFER, ENCODE_WORKERS, PROBE_ATOL, PROBE_ROWS, DropRow, NonFiniteStop,
                    _all_parameter_hashes, _append_jsonl, _backward_rows, _changed_counts, _check_output, _clip_groups,
                    _commit, _encode_or_drop, _factory, _gradient_norms, _keep_before, _newest_checkpoint, _peak_gib,
                    _autocast, _image_inputs, _pad_id,
                    _read_jsonl, _read_probe_rows, _save, _summary, _tensor_counts, _trainable_hashes, _write_json,
                    _write_probe, batch_logprobs, compute_probe, encoded_groups, plan_microbatches)
from .v4 import ARMS, KIND_A2, KIND_B, KIND_RL, KINDS, SFT_KINDS

STATUS_COMPLETE, STATUS_NONFINITE, STATUS_NO_UPDATES = "COMPLETE", "NONFINITE", "NO_UPDATES"


def encode_row_v4(processor: Any, model_config: Any, row: dict, images: list[Any]) -> dict:
    """train.encode_row, except that a truncated or loop-stopped RL rollout gets no end token (append_end false)."""
    tokenizer = processor.tokenizer
    if any(token in text for token in tokenizer.all_special_tokens for text in (row["prompt"], row["target"])):
        raise DropRow("CONTROL_TOKEN")
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(row["prompt"], len(images))], images, IMAGE)
    inputs = prepared["inputs"]
    target = tokenizer.encode(row["target"], add_special_tokens=False)
    if row.get("append_end", True):
        target.append(tokenizer.convert_tokens_to_ids(END_TOKEN))
    if not target:
        raise DropRow("EMPTY_TARGET")
    return _with_row_fields({"row_id": row["row_id"], "kind": row["kind"], "route": row["route"],
                             "input_ids": torch.cat([inputs["input_ids"][0], torch.tensor(target, dtype=torch.long)]),
                             "target_len": len(target), "pixel_values": inputs["pixel_values"],
                             "image_grid_thw": inputs["image_grid_thw"]}, row)


def _with_row_fields(item: dict, row: dict) -> dict:
    return item | {"problem_id": row["problem_id"], **({"advantage": float(row["advantage"])} if row["kind"] == KIND_RL else {})}


def _actor_encoder_v4(processor: Any, config: Any, build_root: Path) -> Callable[[dict], dict]:
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors

    actors = {actor.problem_id: actor for actor in read_actors(Path(build_root) / "train")}

    def encode(row: dict) -> dict:
        actor = actors[row["problem_id"]]
        if list(actor.image_paths) != row["image_paths"]:
            raise ValueError(f"train v4: row {row['row_id']} images differ from the vault")
        images = load_images(actor, Path(build_root) / "images")
        try:
            return encode_row_v4(processor, config, row, images)
        finally:
            for image in images:
                image.close()
    return encode


def _vault_encoder_v4(bundle: PolicyBundle, build_root: Path) -> Callable[[dict], dict]:
    return _actor_encoder_v4(bundle.processor, bundle.model.get_base_model().config, build_root)


def read_groups_v4(rows_path: Path, order_path: Path, start: int, limit: int, arm: str) -> list[list[dict]]:
    """Order groups of the round's slice; the rl-only arm keeps the RL and A1 rows only (the same RL rows as hybrid)."""
    if arm not in ARMS:
        raise ValueError(f"train v4: arm must be one of {ARMS}")
    items = read_order(order_path)["items"]
    if start < 0 or limit < 1 or start + limit > len(items):
        raise ValueError(f"train v4: start/limit outside the order of {len(items)} items")
    ids = [item["problem_id"] for item in items[start:start + limit]]
    rows = _read_jsonl(rows_path)
    known = set(ids)
    for row in rows:
        if (row.get("kind") not in KINDS or row.get("route") not in (ROUTE_FULL, ROUTE_LM)
                or row.get("problem_id") not in known or not isinstance(row.get("target"), str)):
            raise ValueError(f"train v4: invalid row {row.get('row_id')!r} (kind, route or problem_id)")
        if row["kind"] == KIND_RL and (not isinstance(row.get("advantage"), (int, float)) or not math.isfinite(row["advantage"])
                                       or type(row.get("group_size")) is not int or row["group_size"] < 2):
            raise ValueError(f"train v4: RL row {row['row_id']} needs a finite advantage and group_size >= 2")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("train v4: duplicate row_id")
    if arm == "rl-only":
        rows = [row for row in rows if row["kind"] not in (KIND_A2, KIND_B)]
    by_id: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda row: (KINDS.index(row["kind"]), row["row_id"])):
        by_id.setdefault(row["problem_id"], []).append(row)
    return [[row for pid in group for row in by_id.get(pid, [])] for group in update_groups(ids, PER_UPDATE_ITEMS_V2)]


def schedule_v4(rows_path: Path, order_path: Path, start: int, limit: int) -> tuple[int, list[bool]]:
    """The update schedule of both arms: one scheduler step per order group that has a hybrid row. The rl-only arm
    advances the scheduler without a weight change where only note-derived rows were, so LR and step count match."""
    scheduled = [bool(group) for group in read_groups_v4(rows_path, order_path, start, limit, "hybrid")]
    return sum(scheduled), scheduled


def _reference(bundle: PolicyBundle, rows: list[dict]) -> list[torch.Tensor]:
    try:
        with torch.no_grad(), bundle.model.disable_adapter():
            return [value.detach() for value in batch_logprobs(bundle, rows)]
    finally:
        bundle.route.apply(bundle.route.vision_on)  # PEFT re-enables every adapter on exit (policy.reference_logprobs)


def backward_v4(bundle: PolicyBundle, encoded: list[dict], contract: dict, token_budget: int = 0,
                kinds: Iterable[str] = KINDS) -> dict:
    """Backward of the selected loss terms of one update group; never builds an optimizer or changes weights."""
    settings, weights = contract["rl"], contract["sft_weights"]
    selected = set(kinds)
    stats: dict[str, Any] = {"rl": {"groups": 0, "rows": 0, "action_tokens": 0, "surrogate": 0.0,
                                    "mean_abs_advantage": None}, "sft": {}}
    rl_groups: dict[str, list[dict]] = defaultdict(list)
    for row in encoded:
        if row["kind"] == KIND_RL:
            rl_groups[row["problem_id"]].append(row)
    if KIND_RL in selected and rl_groups:
        scale = settings["coefficient"] / len(rl_groups)
        advantages_seen = []
        for rows in rl_groups.values():
            group_tokens = sum(row["target_len"] for row in rows)
            plan = plan_microbatches([row["input_ids"].numel() for row in rows], [row["route"] for row in rows], token_budget)
            for indices in plan:
                batch = [rows[i] for i in indices]
                bundle.route.apply(batch[0]["route"] == ROUTE_FULL)
                current = batch_logprobs(bundle, batch)
                reference = _reference(bundle, batch) if settings["beta"] else None
                term = rl.group_term(current, reference, [row["advantage"] for row in batch], group_tokens=group_tokens,
                                     eps_low=settings["eps_low"], eps_high=settings["eps_high"], beta=settings["beta"])
                if not bool(torch.isfinite(term)):
                    raise NonFiniteStop("nonfinite RL term")
                (term * scale).backward()
                stats["rl"]["surrogate"] += float(term.detach()) * scale
            advantages_seen += [abs(row["advantage"]) for row in rows]
            stats["rl"]["rows"] += len(rows)
            stats["rl"]["action_tokens"] += group_tokens
        stats["rl"]["groups"] = len(rl_groups)
        stats["rl"]["mean_abs_advantage"] = sum(advantages_seen) / len(advantages_seen)
    for kind in SFT_KINDS:
        rows = [row for row in encoded if row["kind"] == kind]
        if kind not in selected or not rows or not weights[kind]:
            continue
        plan = plan_microbatches([row["input_ids"].numel() for row in rows], [row["route"] for row in rows], token_budget)
        losses = _backward_rows(bundle, rows, plan, [weights[kind] / len(rows)] * len(rows))
        mean = losses[kind] / len(rows)
        if not math.isfinite(mean):
            raise NonFiniteStop(f"nonfinite {kind} loss")
        stats["sft"][kind] = {"rows": len(rows), "row_nll_mean": mean, "weight": weights[kind],
                              "weighted_loss": weights[kind] * mean}
    bundle.route.apply(True)
    return stats


def run_update_v4(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: Any, encoded: list[dict],
                  contract: dict, token_budget: int = 0) -> dict:
    """One optimizer step over every row of one order group. NonFiniteStop is raised before any parameter changes."""
    import time

    started = time.perf_counter()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    stats = backward_v4(bundle, encoded, contract, token_budget)
    grad_norms = _clip_groups(bundle)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    bundle.route.apply(True)
    kinds = Counter(row["kind"] for row in encoded)
    loss = stats["rl"]["surrogate"] + sum(value["weighted_loss"] for value in stats["sft"].values())
    return {"rows": len(encoded), "rows_by_kind": dict(kinds), "routes": dict(Counter(row["route"] for row in encoded)),
            "tokens": sum(row["input_ids"].numel() for row in encoded),
            "target_tokens": sum(row["target_len"] for row in encoded), "loss": loss, **stats,
            "grad_norms": grad_norms, "seconds": time.perf_counter() - started, "peak_allocated_gib": _peak_gib(),
            "skipped": False}


def _single_ids(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(v) is not int or v < 0 for v in value):
        raise ValueError(f"single train: {name} must contain the original nonempty integer token IDs")
    return value


def _single_encoded_input_sha256(row: dict) -> str:
    prompt = row["input_ids"][:-row["target_len"]][None]
    values = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt),
              "pixel_values": row["pixel_values"], "image_grid_thw": row["image_grid_thw"]}
    return digest({key: tensor_identity(value) for key, value in values.items()})


def _encode_human_row(processor: Any, model_config: Any, row: dict, images: list[Any]) -> dict:
    """v12 H: an authored human-solution target has no sampled actions. Tokenize it and append the end token that
    sampled rows end with (END_TOKEN, as encode_row_v4); the P0 prompt is encoded as for every other row."""
    tokenizer = processor.tokenizer
    if (row.get("route") != ROUTE_LM or row.get("target_source") != HUMAN_TARGET_SOURCE or "generation" in row
            or not isinstance(row.get("target"), str) or not row["target"].strip()
            or any(token in text for token in tokenizer.all_special_tokens for text in (row["prompt"], row["target"]))):
        raise ValueError("single train: an H row is an LM-routed human target without generation or control tokens")
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(row["prompt"], len(images))], images, IMAGE)
    inputs = prepared["inputs"]
    target = tokenizer.encode(row["target"], add_special_tokens=False)
    if len(target) < MIN_TARGET_TOKENS:
        raise ValueError("single train: H target below the admitted token minimum")
    target.append(tokenizer.convert_tokens_to_ids(END_TOKEN))
    result = dict(row)
    result.update(input_ids=torch.cat([inputs["input_ids"][0], torch.tensor(target, dtype=torch.long)]),
                  target_len=len(target), pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                  input_sha256=prepared["input_sha256"])
    if _single_encoded_input_sha256(result) != prepared["input_sha256"]:
        raise ValueError("single train: prepared inputs cannot be reproduced by the training forward")
    return result


def encode_single_row(processor: Any, model_config: Any, row: dict, images: list[Any], *,
                      image: dict | None = None) -> dict:
    """Preserve sampled actions, including actual EOS; targets never include the prompt or review material.
    v12 H rows (no sampled actions) take _encode_human_row. `image` is the plan's image config (v14); the default
    is the module constant IMAGE."""
    from .content_v16 import ROW_SCHEMA

    if row.get("schema") == "error-note-v18-row-v1":
        from .train_v18 import encode_v18_row

        return encode_v18_row(processor, model_config, row, images, image=image or IMAGE)
    if row.get("schema") == "error-note-v17-row-v1":
        from .train_v17 import encode_v17_row

        return encode_v17_row(processor, model_config, row, images, image=image or IMAGE)
    if row.get("schema") == "error-note-v16b-row-v1":
        from .train_v16b import encode_v16b_row

        return encode_v16b_row(processor, model_config, row, images, image=image or IMAGE)
    if row.get("schema") == ROW_SCHEMA:
        from .train_v16 import encode_content_row

        return encode_content_row(processor, model_config, row, images, image=image or IMAGE, max_target_tokens=8192)
    if row.get("kind") == KIND_HUMAN:
        return _encode_human_row(processor, model_config, row, images)
    generation = row["generation"]
    actions = _single_ids(generation.get("completion_token_ids"), "completion_token_ids")
    if any(getattr(model_config, name, None) in actions for name in ("image_token_id", "video_token_id")):
        raise ValueError("single train: generated image/video placeholder cannot be replayed; HOLD")
    if (row.get("kind") not in KINDS or row.get("route") not in (ROUTE_FULL, ROUTE_LM)
            or row.get("target") != generation.get("text")
            or any(token in row["prompt"] for token in processor.tokenizer.all_special_tokens)):
        raise ValueError("single train: invalid row kind, route, control token or changed original target")
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(row["prompt"], len(images))], images,
                                          IMAGE if image is None else image)
    inputs = prepared["inputs"]
    if row["kind"] == KIND_RL:
        from .single import validate_single_reward

        validate_single_reward(row)
        if (generation.get("prompt_token_ids") != prepared["prompt_token_ids"]
                or generation.get("input_sha256") != prepared["input_sha256"]):
            raise ValueError("single train: RL original prompt IDs or image/input binding changed")
    if generation.get("completion_tokens", len(actions)) != len(actions):
        raise ValueError("single train: completion token count changed")
    result = dict(row)
    result.update(input_ids=torch.cat([inputs["input_ids"][0], torch.tensor(actions, dtype=torch.long)]),
                  target_len=len(actions), pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                  input_sha256=prepared["input_sha256"])
    if _single_encoded_input_sha256(result) != prepared["input_sha256"]:
        raise ValueError("single train: prepared inputs cannot be reproduced by the training forward")
    return result


def _single_hidden(bundle: PolicyBundle, rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    base = bundle.model.get_base_model()
    device = next(base.parameters()).device
    lengths = [row["input_ids"].numel() for row in rows]
    ids = torch.full((len(rows), max(lengths)), _pad_id(bundle), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        if not 0 < row["target_len"] < length:
            raise ValueError("single train: nonempty prompt and actions required")
        ids[index, :length] = row["input_ids"].to(device)
        mask[index, :length] = 1
    with _autocast(device):
        hidden = base.model(input_ids=ids, attention_mask=mask, **_image_inputs(rows, device),
                            use_cache=False, return_dict=True).last_hidden_state
    return hidden, ids, lengths


def _single_reference_hidden(bundle: PolicyBundle, rows: list[dict]) -> torch.Tensor:
    from fine_tuning.modules import tensor_sha256

    base = bundle.model.get_base_model()
    for root in bundle.inventory["mergers"]:
        original = base.get_submodule(root).original_module
        for name, parameter in original.named_parameters():
            if parameter.requires_grad or tensor_sha256(parameter) != bundle.base_sha256[f"{root}.{name}"]:
                raise ValueError("single train: F0 original Merger changed")
    vision_on = bundle.route.vision_on
    try:
        with torch.no_grad(), bundle.model.disable_adapter():
            return _single_hidden(bundle, rows)[0].detach()
    finally:
        bundle.route.apply(vision_on)


def _single_project(head: Any, states: torch.Tensor, actions: torch.Tensor, offset: int,
                    sampling: dict, support: list[list[int]]) -> torch.Tensor:
    """[2, n]: stored-support log q (row 0) and raw full-vocabulary log p at T=1 without presence (row 1, v11
    negative rows), both from the same float32 logits."""
    logits = head(states).float()
    values = []
    for index, scores in enumerate(logits):
        step = offset + index
        values.append(rl.stored_support_logprob(scores[None], actions[:step][None], int(actions[step]),
                                                support[step], **sampling))
    taken = actions[offset:offset + len(logits)]
    raw = logits.log_softmax(-1).gather(1, taken[:, None])[:, 0]
    return torch.stack([torch.stack(values), raw])


def _single_raw_kl(head: Any, states: torch.Tensor, reference: torch.Tensor,
                   vocab_chunk: int, *, force_full: bool = False) -> torch.Tensor:
    """Exact raw KL(F0||theta), streaming vocabulary reductions with recomputed projections."""
    if not isinstance(head, torch.nn.Linear) or any(p.requires_grad for p in head.parameters()):
        raise ValueError("single train: raw F0 KL requires the frozen original linear language head")
    reference = reference.detach()
    if not force_full and torch.equal(states, reference):
        return states.float().sum(-1) * 0  # Exact F0 equality must not create an update from reduction roundoff.

    def log_z(hidden: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        bias = head.bias[start:stop] if head.bias is not None else None
        return torch.nn.functional.linear(hidden, head.weight[start:stop], bias).float().logsumexp(-1)

    current_z: torch.Tensor | None = None
    with torch.no_grad():
        reference_z: torch.Tensor | None = None
        for start in range(0, head.weight.shape[0], vocab_chunk):
            value = log_z(reference, start, start + vocab_chunk)
            reference_z = value if reference_z is None else torch.logaddexp(reference_z, value)
    for start in range(0, head.weight.shape[0], vocab_chunk):
        project = partial(log_z, start=start, stop=start + vocab_chunk)
        value = recompute(project, states, use_reentrant=False) if torch.is_grad_enabled() else project(states)
        current_z = value if current_z is None else torch.logaddexp(current_z, value)
    assert current_z is not None and reference_z is not None
    reference_log_z = reference_z

    def expectation(hidden: torch.Tensor, original: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        bias = head.bias[start:stop] if head.bias is not None else None
        scores = torch.nn.functional.linear(hidden, head.weight[start:stop], bias).float()
        with torch.no_grad():
            scores0 = torch.nn.functional.linear(original, head.weight[start:stop], bias).float()
            probability = (scores0 - reference_log_z[:, None]).exp()
        return (probability * (scores0 - scores)).sum(-1)

    result = current_z - reference_z
    for start in range(0, head.weight.shape[0], vocab_chunk):
        project = partial(expectation, start=start, stop=start + vocab_chunk)
        value = (recompute(project, states, reference, use_reentrant=False) if torch.is_grad_enabled()
                 else project(states, reference))
        result = result + value
    return result


def single_batch_statistics(bundle: PolicyBundle, rows: list[dict], sampling: dict, *, reference: bool = False,
                            chunk: int = 32, vocab_chunk: int = 4096, force_full_kl: bool = False) -> list[dict]:
    """Replay original T1 prefixes; checkpoint projections retain no whole token-by-vocabulary graph."""
    if not rows or chunk < 1 or vocab_chunk < 1:
        raise ValueError("single train: rows and positive projection chunks required")
    original = _single_reference_hidden(bundle, rows) if reference else None
    hidden, ids, lengths = _single_hidden(bundle, rows)
    head = bundle.model.get_base_model().lm_head
    result = []
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        start = length - row["target_len"]
        states, actions = hidden[index, start - 1:length - 1], ids[index, start:length]
        logprobs, kl = [], []
        support = row["generation"]["support_token_ids"]
        for offset in range(0, row["target_len"], chunk):
            h = states[offset:offset + chunk]
            project = partial(_single_project, head, actions=actions, offset=offset, sampling=sampling, support=support)
            logprobs.append(recompute(project, h, use_reentrant=False) if torch.is_grad_enabled() else project(h))
            if original is not None:
                h0 = original[index, start - 1 + offset:min(length - 1, start - 1 + offset + chunk)]
                kl.append(_single_raw_kl(head, h, h0, vocab_chunk, force_full=force_full_kl))
        value, raw = torch.cat(logprobs, dim=1)
        divergence = torch.cat(kl) if kl else torch.zeros_like(value)
        if not all(bool(torch.isfinite(v).all()) for v in (value, raw, divergence)):
            raise NonFiniteStop("single train: nonfinite action log probability or raw KL")
        result.append({"logprobs": value, "raw_logprobs": raw, "kl": divergence})
    return result


def _validate_single_block(bundle: PolicyBundle, encoded: list[dict], contract: dict) -> list[dict]:
    from .single import single_advantage, validate_single_reward

    version = contract.get("version")
    gate = {VERSION_SINGLE: SINGLE_REWARD_GATE, VERSION_SINGLE_V12: SINGLE_REWARD_GATE_V7,
            VERSION_SINGLE_V13: SINGLE_REWARD_GATE_V8,
            VERSION_SINGLE_V14: SINGLE_REWARD_GATE_V9,
            VERSION_SINGLE_V15: SINGLE_REWARD_GATE_V9}.get(version, SINGLE_REWARD_GATE_V5)
    if (version not in (VERSION_SINGLE_V8, VERSION_SINGLE_V9, VERSION_SINGLE_V10, VERSION_SINGLE, VERSION_SINGLE_V12,
                        VERSION_SINGLE_V13, VERSION_SINGLE_V14, VERSION_SINGLE_V15)
            or contract["rl"].get("reward") != gate):
        raise ValueError("single train: SINGLE v4 program-gate reward contract required; v1-v3 cannot resume")
    v10, v11, v12 = version == VERSION_SINGLE_V10, version == VERSION_SINGLE, version == VERSION_SINGLE_V12
    v13, v14, v15 = version == VERSION_SINGLE_V13, version in (VERSION_SINGLE_V14, VERSION_SINGLE_V15), version == VERSION_SINGLE_V15
    if v10 and (contract["rl"].get("baseline") != single_baseline_contract()
                or contract["rl"].get("token_scale") != SINGLE_TOKEN_SCALE_V10
                or contract.get("sft_weights") != SINGLE_SFT_WEIGHTS_V10
                or contract.get("sft_weighting") != SINGLE_SFT_WEIGHTING_V10):
        raise ValueError("single train: v10 source-baseline table, token scale and A2 per-row weighting required")
    if v11 and (contract["rl"].get("baseline") != "none"
                or contract["rl"].get("advantage") != {"version": SINGLE_ADVANTAGE_VERSION_V11,
                                                       "values": SINGLE_ADVANTAGE_V11}
                or contract["rl"].get("token_scale") != SINGLE_TOKEN_SCALE_V11
                or contract["rl"].get("row_token_cap") != SINGLE_ROW_TOKEN_CAP_V11
                or contract["rl"].get("row_token_cap_exempt") != list(SINGLE_ROW_TOKEN_CAP_EXEMPT_V11)
                or contract["rl"].get("negative_logprob") != SINGLE_NEGATIVE_LOGPROB_V11
                or contract.get("sft_weights") != SINGLE_SFT_WEIGHTS_V10
                or contract.get("sft_weighting") != SINGLE_SFT_WEIGHTING_V10):
        raise ValueError("single train: v11 penalty table, token scale, row cap with LOOP exemption, raw negative "
                         "log p and A2 per-row weighting required")
    if v12:
        from .references import human_solution_sha256

        if (contract["rl"].get("baseline") != "none"
                or contract["rl"].get("advantage") != {"version": SINGLE_ADVANTAGE_VERSION_V12,
                                                       "values": SINGLE_ADVANTAGE_V12}
                or contract["rl"].get("token_scale") != SINGLE_TOKEN_SCALE_V11
                or contract["rl"].get("row_token_cap") != SINGLE_ROW_TOKEN_CAP_V11
                or contract["rl"].get("row_token_cap_exempt") != list(SINGLE_ROW_TOKEN_CAP_EXEMPT_V11)
                or contract["rl"].get("negative_logprob") != SINGLE_NEGATIVE_LOGPROB_V11
                or contract.get("sft_weights") != SINGLE_SFT_WEIGHTS_V12
                or contract.get("sft_weighting") != SINGLE_SFT_WEIGHTING_V12
                or not isinstance(contract.get("human_solution"), dict)
                or contract["human_solution"].get("fields_sha256") != human_solution_sha256()):
            raise ValueError("single train: v12 penalty table, v11 PG settings, H weight 0.10 per kind and the "
                             "sealed human-solution fields required")
    if v13 and (contract["rl"].get("baseline") != "none"
                or contract["rl"].get("advantage") != {"version": SINGLE_ADVANTAGE_VERSION_V13,
                                                       "values": SINGLE_ADVANTAGE_V13}
                or contract["rl"].get("spiral_window") != SINGLE_SPIRAL_V13
                or contract["rl"].get("praise") != SINGLE_PRAISE_V13
                or contract["rl"].get("token_scale") != SINGLE_TOKEN_SCALE_V11
                or contract["rl"].get("row_token_cap") != SINGLE_ROW_TOKEN_CAP_V11
                or contract["rl"].get("row_token_cap_exempt") != list(SINGLE_ROW_TOKEN_CAP_EXEMPT_V11)
                or contract["rl"].get("negative_logprob") != SINGLE_NEGATIVE_LOGPROB_V11
                or contract.get("sft_weights") != SINGLE_SFT_WEIGHTS_V10
                or contract.get("sft_weighting") != SINGLE_SFT_WEIGHTING_V10
                or "human_solution" in contract):
        raise ValueError("single train: v13 penalty table, spiral window, praise rule, v11 PG settings and v10 SFT "
                         "weights without human-solution rows required")
    if v14 and (contract["rl"].get("baseline") != "none"
                or contract["rl"].get("advantage") != {"version": SINGLE_ADVANTAGE_VERSION_V14,
                                                       "values": SINGLE_ADVANTAGE_V14}
                or contract["rl"].get("spiral_window") != SINGLE_SPIRAL_V13
                or contract["rl"].get("praise") != SINGLE_PRAISE_V13
                or contract["rl"].get("tail_praise") != SINGLE_TAIL_PRAISE_V14
                or contract["rl"].get("token_scale") != SINGLE_TOKEN_SCALE_V11
                or contract["rl"].get("row_token_cap") != SINGLE_ROW_TOKEN_CAP_V11
                or contract["rl"].get("row_token_cap_exempt") != list(SINGLE_ROW_TOKEN_CAP_EXEMPT_V11)
                or contract["rl"].get("negative_logprob") != SINGLE_NEGATIVE_LOGPROB_V11
                or contract.get("sft_weights") != SINGLE_SFT_WEIGHTS_V10
                or contract.get("sft_weighting") != SINGLE_SFT_WEIGHTING_V10
                or "human_solution" in contract
                or contract.get("a2_min_tokens") != SINGLE_A2_MIN_TOKENS_V14
                or contract.get("b_rows") != SINGLE_B_ROWS_V14
                or type(contract.get("schedule_steps")) is not int
                or not 1 <= contract["schedule_steps"] <= contract["total_slots"]):
        raise ValueError("single train: v14 penalty table, spiral window, praise and tail-praise rules, A2 minimum, "
                         "B rule, schedule steps, v11 PG settings and v10 SFT weights without human-solution rows "
                         "required")
    if v15 and (contract.get("routes") != SINGLE_ROUTES_V15 or contract.get("resolve_attempts") != SINGLE_RESOLVE_V15
                or contract.get("generation_counts") != {"t1": 1, "note": 1, "resolve": 3}):
        raise ValueError("single train: v15 routes and lesson re-solve attempts contract required")
    if version in (VERSION_SINGLE_V8, VERSION_SINGLE_V9) and (
            contract["rl"].get("baseline", 0.0) != 0.0 or "sft_weighting" in contract):
        raise ValueError("single train: v8/v9 contracts keep b=0 and per-kind SFT weighting")
    if version != VERSION_SINGLE_V8 and (
            contract["rl"].get("format_anchor") != {"positions": SINGLE_FORMAT_ANCHOR_POSITIONS, "beta": SINGLE_FORMAT_ANCHOR_BETA}
            or not isinstance(contract.get("replay_support"), dict)
            or contract["replay_support"].get("row_failure") != SINGLE_REPLAY_ROW_FAILURE
            or contract["replay_support"].get("max_excluded_rows") != SINGLE_REPLAY_MAX_EXCLUDED_ROWS):
        raise ValueError("single train: v9 format anchor and replay row-failure contract required")
    expected = contract["expected_problem_ids"]
    if (not isinstance(expected, list) or not expected or len(set(expected)) != len(expected)
            or any(not isinstance(pid, str) or not pid for pid in expected)):
        raise ValueError("single train: unique preregistered expected problem IDs required")
    sampling = contract["sampling"]
    sampling_hash = rl.sampling_sha256(sampling)
    if rl.policy_sha256(bundle) != contract["policy_sha256"]:
        raise ValueError("single train: current policy differs from collection policy")
    if len({row["row_id"] for row in encoded}) != len(encoded):
        raise ValueError("single train: duplicate row ID")
    t1 = [row for row in encoded if row["kind"] == KIND_RL]
    if len(t1) != len(expected) or {row["problem_id"] for row in t1} != set(expected):
        raise ValueError("single train: incomplete or duplicate T1 membership")
    if any(not isinstance(contract.get(key), dict) or set(contract[key]) != set(expected)
           for key in ("reward_provenance_sha256", "reward_evidence_sha256")):
        raise ValueError("single train: fixed per-ID reward provenance/evidence pins required")
    pairs = [(row["problem_id"], row["kind"]) for row in encoded]
    if len(set(pairs)) != len(pairs):
        raise ValueError("single train: multiple responses for a problem/kind")
    for row in encoded:
        if (row["problem_id"] not in expected or row["kind"] not in KINDS + ((KIND_HUMAN,) if v12 else ())
                or row["route"] not in (ROUTE_FULL, ROUTE_LM)
                # v15: every row of an image item is FULL, every row of a text-only item LM; v11-v14: RL/A1/H are LM.
                or v15 and row["route"] != (ROUTE_FULL if row.get("image_sha256") else ROUTE_LM)
                or not v15 and row["kind"] in (KIND_RL, "A1", KIND_HUMAN) and row["route"] != ROUTE_LM
                or row["kind"] == KIND_HUMAN and (row.get("target_source") != HUMAN_TARGET_SOURCE or "generation" in row)):
            raise ValueError("single train: invalid kind, problem or route")
    if v14 and any(row["kind"] == "A2" and row["target_len"] < SINGLE_A2_MIN_TOKENS_V14 for row in encoded):
        raise ValueError("single train: v14 A2 row below the sealed token minimum")
    for row in t1:
        generation = row["generation"]
        actions = _single_ids(generation.get("completion_token_ids"), "completion_token_ids")
        prompt = _single_ids(generation.get("prompt_token_ids"), "prompt_token_ids")
        if (generation.get("policy_sha256") != contract["policy_sha256"]
                or generation.get("sampling_sha256") != sampling_hash
                or generation.get("input_sha256") != row.get("input_sha256")
                or generation.get("input_sha256") != _single_encoded_input_sha256(row)
                or row["input_ids"].tolist() != prompt + actions or row["target_len"] != len(actions)
                or row.get("complete") is not True or type(row.get("reward")) is not int or row["reward"] not in (-1, 0, 1)
                or row.get("status") not in {*VERDICT_LABELS, "LOOP_STOPPED", "UNVERIFIABLE"}
                or len(generation.get("sampling_logprobs", [])) != len(actions)
                or len(generation.get("support_sha256", [])) != len(actions)
                or len(generation.get("support_token_ids", [])) != len(actions)):
            raise ValueError("single train: incomplete, stale or changed original RL generation")
        for action, support, sha in zip(actions, generation["support_token_ids"], generation["support_sha256"], strict=True):
            if (not isinstance(support, list) or action not in support
                    or any(type(token) is not int for token in support) or digest(support) != sha):
                raise ValueError("single train: stored generation support changed or excludes its sampled action")
        decision = validate_single_reward(row)
        if (row["reward_evidence"]["version"] != gate or ("baseline" in row["reward_evidence"]) != v10
                or ("advantage" in row) != (v10 or v11 or v12 or v13 or v14)
                or (v11 or v12 or v13 or v14) and row["advantage"] != single_advantage(decision, gate)):
            raise ValueError("single train: RL baseline/advantage binding differs from the contract version")
        if (digest(row["reward_evidence"]["provenance"]) != contract["reward_provenance_sha256"][row["problem_id"]]
                or digest(row["reward_evidence"]) != contract["reward_evidence_sha256"][row["problem_id"]]):
            raise ValueError("single train: original reward source/evidence differs from the sealed contract")
        expected_lp = torch.tensor(generation["sampling_logprobs"], dtype=torch.float32)
        if not bool(torch.isfinite(expected_lp).all()):
            raise ValueError("single train: nonfinite stored sampling log probabilities")
    return t1


def _require_single_zero_dropout(bundle: PolicyBundle) -> None:
    config = bundle.model.get_base_model().config
    for value in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if value is not None and getattr(value, "attention_dropout", 0) != 0:
            raise ValueError("single train: nonzero configured attention dropout changes sampling replay; HOLD")
    if any(value.lora_dropout != 0 for value in bundle.model.peft_config.values()):
        raise ValueError("single train: nonzero configured LoRA dropout changes sampling replay; HOLD")
    for name, module in bundle.model.named_modules():
        if ((isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p != 0)
                or getattr(module, "attention_dropout", 0) != 0):
            raise ValueError(f"single train: nonzero actual dropout at {name} changes sampling replay; HOLD")


def backward_single_block(bundle: PolicyBundle, encoded: list[dict], contract: dict, token_budget: int = 0,
                          components: Iterable[str] = ("PG", "KL", "A1", "A2", "B", KIND_HUMAN), *,
                          capacity: Any = None, force_full_kl: bool = False, capture: Any = None,
                          note_gradient_capture: dict | None = None) -> dict:
    """One frozen-policy block, fixed B*C PG/raw KL and per-kind row-mean SFT; no optimizer or scheduler.

    v10: PG = -(reward - b(source)) * sum(logprobs) / (B * 256) on every non-excluded RL row, and A2 rows each carry
    the full A2 weight (per_row) instead of sharing it.
    v11: PG = -A * min(1, 2048/n) * sum(l) / (B * 1024) with A from the penalty table; l = stored-support log q when
    A > 0, raw full-vocabulary log p when A < 0, no PG term when A == 0. SFT weighting is v10's.
    v12: v11's PG/KL/anchor with the v12 table; H rows (human solutions) are one more SFT kind (0.10 per kind
    total) and never enter PG, KL, the anchor or replay.
    v13: v11's PG/KL/anchor with the v13 table, except that a LOOP or TRUNCATED_NO_LOOP row's PG term is
    -A * sum(raw log p over its sealed spiral window) / (B * 1024) with no row cap; no H rows.
    v14: v13, plus a tail-praised row (A = +1) pushes up only its sealed tail window, sum(stored-support log q) at full
    mass (no row cap); the LR schedule and the A2 minimum are fixed by the contract."""
    from .protocol import VERSION_CONTENT_V16, VERSION_V16B

    if contract.get("version") in ("error-note-v18", "error-note-v18b"):
        from .train_v18 import backward_v18_block

        selected_v18 = tuple(components) if set(components) <= {"PG", "RL", "OBS", "KL", "A1", "C", "N", "PREP_SELF", "PREP_HUMAN"} else None
        return backward_v18_block(bundle, encoded, contract, token_budget, components=selected_v18,
                                  capacity=capacity, note_gradient_capture=note_gradient_capture)
    if contract.get("version") == "error-note-v17":
        from .train_v17 import backward_v17_block

        selected_v17 = tuple(components) if set(components) <= {"PG", "RL", "OBS", "KL", "A1", "C", "N", "PREP_SELF", "PREP_HUMAN"} else None
        return backward_v17_block(bundle, encoded, contract, token_budget, components=selected_v17, capacity=capacity)
    if contract.get("version") == VERSION_V16B:
        from .train_v16b import backward_v16b_block

        return backward_v16b_block(bundle, encoded, contract, token_budget, components=components, capacity=capacity)
    if contract.get("version") == VERSION_CONTENT_V16:
        from .train_v16 import backward_content_block

        selected_content = tuple(components) if set(components) <= {"O", "J", "C", "N", "A1"} else None
        return backward_content_block(bundle, encoded, contract, token_budget,
                                      components=selected_content, capacity=capacity)
    selected = set(components)
    if capacity is not None and capture is not None:
        raise ValueError("single train: measured admission and calibration are separate transactions")
    if selected - {"PG", "KL", "A1", "A2", "B", KIND_HUMAN}:
        raise ValueError("single train: unknown loss component")
    v12 = contract.get("version") == VERSION_SINGLE_V12
    sft_kinds = SFT_KINDS + ((KIND_HUMAN,) if v12 else ())
    settings, weights = contract["rl"], contract["sft_weights"]
    scale, beta = settings["token_scale"], settings["beta"]
    if (not math.isfinite(scale) or scale <= 0 or not math.isfinite(beta) or beta < 0
            or set(weights) != set(sft_kinds) or any(not math.isfinite(w) or w < 0 for w in weights.values())):
        raise ValueError("single train: explicit finite nonnegative weights and positive token scale required")
    _require_single_zero_dropout(bundle)
    t1 = _validate_single_block(bundle, encoded, contract)
    v9 = contract["version"] != VERSION_SINGLE_V8  # v8 keeps the first-row replay HOLD and has no anchor
    # v12 keeps v11's PG, KL, anchor and per-row/per-kind SFT weighting.
    v10 = contract["version"] == VERSION_SINGLE_V10
    v14 = contract["version"] in (VERSION_SINGLE_V14, VERSION_SINGLE_V15)  # v15 keeps every v14 rule
    v13 = contract["version"] in (VERSION_SINGLE_V13, VERSION_SINGLE_V14, VERSION_SINGLE_V15)  # v14 keeps the spiral window
    v11 = contract["version"] in (VERSION_SINGLE, VERSION_SINGLE_V12, VERSION_SINGLE_V13, VERSION_SINGLE_V14,
                                  VERSION_SINGLE_V15)
    per_row = contract.get("sft_weighting", {}) if v10 or v11 else {}
    anchor = v9 and "KL" in selected
    reference = "KL" in selected and (beta > 0 or anchor)  # the anchor reads the same exact raw F0 KL
    initial_route = bundle.route.vision_on
    was_training = bundle.model.training
    plan = plan_microbatches([row["input_ids"].numel() for row in t1], [row["route"] for row in t1], token_budget)
    from .capacity import CapacityHold, encoded_shapes

    if capacity is not None and any(row["route"] == ROUTE_FULL for row in t1):
        raise CapacityHold("capacity HOLD: the measured profile has no FULL RL phase")  # capacity.PHASES["rl_kl"] is LM
    if capacity is not None or capture is not None:
        if type(token_budget) is not int or token_budget < 0:
            raise CapacityHold("capacity HOLD: nonnegative integer training token budget required")
        if token_budget and any(row["input_ids"].numel() > token_budget for row in encoded):
            raise CapacityHold("capacity HOLD: a training singleton exceeds the explicit token budget")
    if capacity is not None:
        plan = capacity.partition(encoded_shapes(t1), plan, "rl_kl")
    sft_plans = {}
    for kind in sft_kinds:
        rows = [row for row in encoded if row["kind"] == kind]
        if kind not in selected or not rows or not weights[kind]:
            continue
        batches = plan_microbatches([row["input_ids"].numel() for row in rows], [row["route"] for row in rows], token_budget)
        if capacity is not None:
            bounded = []
            geometry = encoded_shapes(rows)
            for indices in batches:
                phase = "sft_full" if rows[indices[0]]["route"] == ROUTE_FULL else "sft_lm"
                bounded.extend(capacity.partition(geometry, [indices], phase))
            batches = bounded
        sft_plans[kind] = (rows, batches)
    stats: dict[str, Any] = {"rl": {"rows": len(t1), "problem_count": len(t1), "token_scale": scale,
                                     "action_tokens": sum(row["target_len"] for row in t1), "pg": 0., "kl": 0.},
                             "sft": {}, "replay_verified": False}
    if v9:
        stats["rl"]["anchor"] = 0.
    if v11:
        stats["rl"].update(pg_positive=0., pg_negative=0., pg_rows_positive=0, pg_rows_negative=0, capped_rows=0,
                           cap_exempt_rows=0)
    if v13:
        stats["rl"].update(window_rows=0, window_tokens=0)
    if v14:
        stats["rl"].update(tail_rows=0, tail_tokens=0)
    advantages: list[float] = []
    excluded: list[dict] = []
    try:
        bundle.model.train()  # Qwen activation checkpointing requires training mode; checked zero dropout keeps replay exact.
        # Validate every sampled distribution before any component can leave a partial gradient.
        bundle.route.apply(False)
        replay_stats = {"tokens": 0, "abs_sum": 0., "max_abs": 0., "over_0.1": 0, "row_mean_max": 0.}
        with torch.no_grad():
            for indices in plan:
                batch = [t1[i] for i in indices]
                if capacity is not None:
                    capacity.check(encoded_shapes(batch), "rl_kl")
                replay = single_batch_statistics(bundle, batch, contract["sampling"])
                for row, actual in zip(batch, replay, strict=True):
                    expected = torch.tensor(row["generation"]["sampling_logprobs"], device=actual["logprobs"].device)
                    gap = (actual["logprobs"].float() - expected).abs()
                    row_mean, row_max = float(gap.mean()), float(gap.max())
                    replay_stats.update(tokens=replay_stats["tokens"] + gap.numel(),
                                        abs_sum=replay_stats["abs_sum"] + float(gap.sum()),
                                        max_abs=max(replay_stats["max_abs"], row_max),
                                        row_mean_max=max(replay_stats["row_mean_max"], row_mean))
                    replay_stats["over_0.1"] += int((gap > .1).sum())
                    long_row = gap.numel() >= SINGLE_REPLAY_ROW_MEAN_MIN_TOKENS
                    if long_row and row_mean > SINGLE_REPLAY_ROW_MEAN_ATOL or row_max > SINGLE_REPLAY_MAX_ATOL:
                        if not v9 or len(excluded) == SINGLE_REPLAY_MAX_EXCLUDED_ROWS:
                            raise ValueError(f"single train: sampling logprob replay changed (row mean {row_mean:.4f}, "
                                             f"max {row_max:.4f}); RL HOLD")
                        # v9: a rare per-row engine mismatch drops only that row; denominators stay fixed.
                        excluded.append({"problem_id": row["problem_id"], "row_mean": row_mean, "row_max": row_max,
                                         "tokens": gap.numel()})
        block_mean = replay_stats["abs_sum"] / max(1, replay_stats["tokens"])
        if block_mean > SINGLE_REPLAY_BLOCK_MEAN_ATOL:
            raise ValueError(f"single train: sampling logprob replay changed (block mean {block_mean:.4f}); RL HOLD")
        stats["replay"] = replay_stats | {"mean_abs": block_mean} | ({"excluded": excluded} if v9 else {})
        stats["replay_verified"] = True
        dropped = {value["problem_id"] for value in excluded}
        if "PG" in selected or reference:
            for indices in plan:
                batch = [t1[i] for i in indices]
                # Micro-batches are single-route; v11-v14 RL rows are always LM (checked in _validate_single_block).
                bundle.route.apply(batch[0]["route"] == ROUTE_FULL)
                if capacity is not None:
                    capacity.check(encoded_shapes(batch), "rl_kl")
                with capture.batch(batch, "rl_kl", "RL", full_kl=force_full_kl and reference) if capture is not None else nullcontext():
                    values = single_batch_statistics(bundle, batch, contract["sampling"], reference=reference,
                                                      force_full_kl=force_full_kl)
                    terms = []
                    for row, value in zip(batch, values, strict=True):
                        if row["problem_id"] in dropped:
                            continue  # the replay-validated forward keeps its batch; only this row's terms are dropped
                        if "PG" in selected and v11:
                            advantage = row["advantage"]
                            advantages.append(advantage)
                            if advantage:
                                # Negative rows read the raw distribution: a one-token stored support has log q = 0.
                                sign = "positive" if advantage > 0 else "negative"
                                capped = min(1.0, settings["row_token_cap"] / row["target_len"])
                                # LOOP rows keep v9's full per-row mass; the cap would cut an 8192-token loop to 1/4.
                                exempt = row["reward_evidence"]["decision"]["penalty"] in settings["row_token_cap_exempt"]
                                weight = 1.0 if exempt else capped
                                taken = value["logprobs"] if advantage > 0 else value["raw_logprobs"]
                                window = row["reward_evidence"]["decision"]["window"] if v13 else None
                                tail = row["reward_evidence"]["decision"]["tail"] if v14 else None
                                if window is not None:
                                    # v13: only the spiral window is pushed down, at full mass (no row cap).
                                    start, end = window
                                    if advantage >= 0 or not 0 <= start < end <= taken.numel():
                                        raise ValueError("single train: invalid sealed spiral window")
                                    taken, weight, capped = taken[start:end], 1.0, 1.0  # not a capped or cap-exempt row
                                    stats["rl"]["window_rows"] += 1
                                    stats["rl"]["window_tokens"] += end - start
                                elif tail is not None:
                                    # v14: only the sealed tail is pulled up (stored-support log q), at full mass.
                                    start, end = tail
                                    if not advantage > 0 or not 0 <= start < end == taken.numel():
                                        raise ValueError("single train: invalid sealed tail window")
                                    taken, weight, capped = taken[start:end], 1.0, 1.0
                                    stats["rl"]["tail_rows"] += 1
                                    stats["rl"]["tail_tokens"] += end - start
                                term = -advantage * weight * taken.sum() / (len(t1) * scale)
                                stats["rl"]["pg"] += float(term.detach())
                                stats["rl"]["pg_" + sign] += float(term.detach())
                                stats["rl"]["pg_rows_" + sign] += 1
                                stats["rl"]["capped_rows"] += int(weight < 1)
                                stats["rl"]["cap_exempt_rows"] += int(exempt and capped < 1)
                                terms.append(term)
                        elif "PG" in selected and v10:
                            # Fixed source baseline: a zero reward on an easy source now pushes its actions down.
                            term = -row["advantage"] * value["logprobs"].sum() / (len(t1) * scale)
                            stats["rl"]["pg"] += float(term.detach())
                            advantages.append(row["advantage"])
                            terms.append(term)
                        elif "PG" in selected:
                            term = rl.single_pg(value["logprobs"], row["reward"], problem_count=len(t1), token_scale=scale)
                            stats["rl"]["pg"] += float(term.detach())
                            if row["reward"]:
                                terms.append(term)
                        if "KL" in selected and beta:
                            term = beta * value["kl"].sum() / (len(t1) * scale)
                            stats["rl"]["kl"] += float(term.detach())
                            terms.append(term)
                        if anchor:
                            term = SINGLE_FORMAT_ANCHOR_BETA * value["kl"][:SINGLE_FORMAT_ANCHOR_POSITIONS].sum() / len(t1)
                            stats["rl"]["anchor"] += float(term.detach())
                            terms.append(term)
                    if terms:
                        torch.stack(terms).sum().backward()
        for kind, (rows, batches) in sft_plans.items():
            losses: dict[str, float] = defaultdict(float)
            for indices in batches:
                if capacity is not None:
                    phase = "sft_full" if rows[indices[0]]["route"] == ROUTE_FULL else "sft_lm"
                    capacity.check(encoded_shapes([rows[i] for i in indices]), phase)
                batch = [rows[i] for i in indices]
                phase = "sft_full" if batch[0]["route"] == ROUTE_FULL else "sft_lm"
                with capture.batch(batch, phase, kind) if capture is not None else nullcontext():
                    coefficient = weights[kind] if per_row.get(kind) == "per_row" else weights[kind] / len(rows)
                    values = _backward_rows(bundle, rows, [indices], [coefficient] * len(rows))
                losses[kind] += values[kind]
            mean = losses[kind] / len(rows)
            if not math.isfinite(mean):
                raise NonFiniteStop(f"single train: nonfinite {kind} loss")
            stats["sft"][kind] = {"rows": len(rows), "row_nll_mean": mean, "weight": weights[kind],
                                   "weighted_loss": weights[kind] * (losses[kind] if per_row.get(kind) == "per_row" else mean)}
            if v10 or v11:
                stats["sft"][kind]["weighting"] = per_row[kind]
    finally:
        bundle.model.train(was_training)
        bundle.route.apply(initial_route)
    if v10 or v11:
        stats["rl"].update(pg_rows=len(advantages), advantage_mean=sum(advantages) / len(advantages) if advantages else 0.,
                           advantage_abs_mean=sum(map(abs, advantages)) / len(advantages) if advantages else 0.)
    stats["loss"] = stats["rl"]["pg"] + stats["rl"]["kl"] + stats["rl"].get("anchor", 0.) + sum(v["weighted_loss"] for v in stats["sft"].values())
    stats["has_gradient"] = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for ps in bundle.params.values() for p in ps)
    return stats


def _contracts(args: argparse.Namespace) -> tuple[dict, dict | None]:
    init = None
    if args.round == 1:
        if args.init_run is not None:
            raise ValueError("train v4: round 1 starts from F0 (no --init-run)")
    else:
        if args.init_run is None:
            raise ValueError("train v4: a later round needs --init-run (the previous round's run directory)")
        final = Path(args.init_run) / "checkpoint-final"
        init = {"checkpoint": final, "binding": json.loads((Path(args.init_run) / "binding.json").read_text(encoding="utf-8"))}
        ckpt.read_metadata(final, binding=init["binding"])
    return {"rows": ckpt.file_hash(args.rows), "order": ckpt.file_hash(args.order),
            "init": ckpt.file_hash(init["checkpoint"] / "metadata.json") if init else None}, init


def _protocol(args: argparse.Namespace, total_steps: int, contracts: dict) -> dict:
    return plain_protocol_v4(args.precision, max(1, total_steps), arm=args.arm, round_index=args.round,
                             start=args.start, limit=args.limit, contracts=contracts)


def _initial_bundle(args: argparse.Namespace, protocol: dict, init: dict | None, policy_factory: Callable | None,
                    device: str) -> PolicyBundle:
    if init is not None:
        bundle = ckpt.load_eval(args.model_path, init["checkpoint"], binding=init["binding"], policy_factory=policy_factory)
        bundle.model.train()
    else:
        bundle = (policy_factory or build_policy)(args.model_path, ARM, protocol)
    if args.precision == "bf16":
        bundle.model.to(device)
    return bundle


def run_train_v4(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0",
                 make_encoder: Callable[[PolicyBundle, Path], Callable[[dict], dict]] = _vault_encoder_v4,
                 map_fn: Callable[..., Iterable] = map) -> int:
    if args.gradient_checkpointing == "off" and args.precision == "nf4":
        raise SystemExit("train v4: --gradient-checkpointing off is bf16 only")
    out: Path = args.output
    groups = read_groups_v4(args.rows, args.order, args.start, args.limit, args.arm)
    total_steps, scheduled = schedule_v4(args.rows, args.order, args.start, args.limit)
    contracts, init = _contracts(args)
    protocol = _protocol(args, total_steps, contracts)
    contract = protocol["training_contract"]
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": contracts["rows"], "code_sha256": hashlib.sha256(args.code_commit.encode()).hexdigest()}
    run = {"arm": args.arm, "precision": args.precision, "round": args.round, "start": args.start, "limit": args.limit,
           "max_row_tokens": args.max_row_tokens, "order_sha256": contracts["order"], "init": contracts["init"]}
    _check_output(args, binding, run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)
    assert_base = assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged
    torch.manual_seed(MASTER_SEED)
    newest = _newest_checkpoint(out) if args.resume else None
    step, position = 0, 0
    if newest is None:
        bundle = _initial_bundle(args, protocol, init, policy_factory, device)
        optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
        _write_json(out / "binding.json", binding)
        _write_json(out / "initial_state.json", {"run": run, "sha256": _trainable_hashes(bundle)})
    else:
        bundle, optimizer, scheduler, step, cursor = ckpt.load_resume(
            args.model_path, newest, protocol, binding=binding, device=device, policy_factory=policy_factory)
        position = cursor["position"]
    if args.gradient_checkpointing == "off":
        bundle.model.gradient_checkpointing_disable()
    updates_path, drops_path = out / "updates.jsonl", out / "drops_local.jsonl"
    for path in (updates_path, drops_path):
        _keep_before(path, position)
        path.touch()
    encode = make_encoder(bundle, args.build_root)
    for index, (encoded, drops) in enumerate(encoded_groups(groups[position:], encode, args.max_row_tokens or CONTEXT_LIMIT,
                                                            map_fn), start=position):
        for drop in drops:
            _append_jsonl(drops_path, {"group": index, **drop})
        if not encoded:
            if scheduled[index]:  # the hybrid arm steps here: keep both arms on one LR schedule (no weight change)
                optimizer.zero_grad(set_to_none=True)
                optimizer.step()  # every gradient is None: AdamW leaves every parameter and moment unchanged
                scheduler.step()
                step += 1
            _append_jsonl(updates_path, {"step": step, "group": index, "rows": 0, "skipped": True,
                                         "scheduler_only": scheduled[index]})
            continue
        lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
        try:
            stats = run_update_v4(bundle, optimizer, scheduler, encoded, contract, args.token_budget)
        except NonFiniteStop as exc:
            _write_json(out / "result.json", {"status": STATUS_NONFINITE, "group": index, "step": step, "detail": str(exc)})
            return 1
        step += 1
        _append_jsonl(updates_path, {"step": step, "group": index, **stats, "lr": lrs})
        if args.save_every and step % args.save_every == 0:
            _save(bundle, optimizer, scheduler, step, index + 1, out / f"checkpoint-{step}", protocol, binding)
    summary = _summary(_read_jsonl(updates_path))
    if not summary["updates"]:
        _write_json(out / "result.json", {"status": STATUS_NO_UPDATES, "arm": args.arm, **summary})
        return 1
    if step != total_steps:
        raise ValueError(f"train v4: {step} scheduler steps differ from the scheduled {total_steps}")
    assert_base(bundle)
    initial = json.loads((out / "initial_state.json").read_text(encoding="utf-8"))["sha256"]
    changed = _changed_counts(bundle, initial)
    _save(bundle, optimizer, scheduler, step, len(groups), out / "checkpoint-final", protocol, binding)
    probe_rows: list[dict] = []
    for row in (row for group in groups for row in group):
        item = _encode_or_drop(row, encode, args.max_row_tokens or CONTEXT_LIMIT)
        probe_rows += [item] if "input_ids" in item else []
        if len(probe_rows) == PROBE_ROWS:
            break
    _write_probe(bundle, probe_rows, out)
    _write_json(out / "result.json", {
        "status": STATUS_COMPLETE, "arm": args.arm, "precision": args.precision, "round": args.round, **summary,
        "changed_tensors": changed, "total_tensors": _tensor_counts(bundle), "protocol_version": VERSION_V4,
        "total_steps": protocol["optimizer"]["total_steps"], "init": contracts["init"],
        "gradient_checkpointing": args.gradient_checkpointing, "token_budget": args.token_budget,
        "checkpoint": str(out / "checkpoint-final")})
    return 0


def run_dry_run_v4(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0",
                   make_encoder: Callable[[PolicyBundle, Path], Callable[[dict], dict]] = _vault_encoder_v4,
                   map_fn: Callable[..., Iterable] = map) -> int:
    """At most 10 predeclared items: each loss kind backpropagated alone, gradients checked and cleared, weights unchanged."""
    if not 1 <= args.limit <= 10:
        raise ValueError("dry-run v4: limit must be 1..10 items")
    out = args.output
    if (out / "dry_run.json").exists():
        raise ValueError("dry-run v4: refusing to overwrite an existing diagnostic")
    out.mkdir(parents=True, exist_ok=True)
    groups = read_groups_v4(args.rows, args.order, args.start, args.limit, args.arm)
    contracts, init = _contracts(args)
    protocol = _protocol(args, schedule_v4(args.rows, args.order, args.start, args.limit)[0], contracts)
    report: dict[str, Any] = {"status": "FAIL", "optimizer_steps": 0, "scheduler_steps": 0, "weights_saved": False,
                              "scope": "diagnostic only: finite per-kind backward with unchanged weights; no learning claim",
                              "protocol_sha256": protocol["sha256"], "components": {}, "weights_unchanged": None,
                              "gradients_cleared": None, "rows_by_kind": dict(Counter(r["kind"] for g in groups for r in g))}
    assert_base = assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged
    bundle = None
    before = None
    try:
        torch.manual_seed(MASTER_SEED)
        bundle = _initial_bundle(args, protocol, init, policy_factory, device)
        assert_base(bundle)
        before = _all_parameter_hashes(bundle)
        encode = make_encoder(bundle, args.build_root)
        encoded = [row for kept, _ in encoded_groups(groups, encode, args.max_row_tokens or CONTEXT_LIMIT, map_fn)
                   for row in kept]
        report["encoded_rows_by_kind"] = dict(Counter(row["kind"] for row in encoded))
        report["paired_a2_b_problem_ids"] = sorted({r["problem_id"] for r in encoded if r["kind"] == KIND_A2}
                                                   & {r["problem_id"] for r in encoded if r["kind"] == KIND_B})
        for kind in KINDS:
            rows = [row for row in encoded if row["kind"] == kind]
            if not rows:
                continue
            bundle.model.zero_grad(set_to_none=True)
            stats = backward_v4(bundle, rows, protocol["training_contract"], args.token_budget, kinds=[kind])
            norms = _gradient_norms(bundle)
            full = any(row["route"] == ROUTE_FULL for row in rows)
            weight_zero = kind != KIND_RL and not protocol["training_contract"]["sft_weights"][kind]
            expected = ({"L": "none", "M": "none", "V": "none"} if weight_zero else
                        {"L": "finite_positive", "M": "finite_positive", "V": "finite_positive" if full else "none"})
            def absent(group: str) -> bool:
                value = norms.get(group)
                return value is None or not value > 0

            errors = [f"{'UNEXPECTED' if need == 'none' else 'MISSING'}_GRADIENT:{group}"
                      for group, need in expected.items() if (need == "none") != absent(group)]
            report["components"][kind] = {"rows": len(rows), "stats": stats, "grad_norms": norms,
                                          "expected": expected, "route_check_passed": not errors,
                                          "route_check_reason": ";".join(errors) or None}
            bundle.model.zero_grad(set_to_none=True)
        report["status"] = ("PASS" if report["components"] and all(c["route_check_passed"] for c in report["components"].values())
                            else "NO_ROWS" if not report["components"] else "GRADIENT_ROUTE_MISMATCH")
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if bundle is not None:
            bundle.model.zero_grad(set_to_none=True)
            report["gradients_cleared"] = all(p.grad is None for p in bundle.model.parameters())
            bundle.route.apply(True)
            report["weights_unchanged"] = before == _all_parameter_hashes(bundle) if before is not None else None
            if report["weights_unchanged"] is False:
                report["status"] = "FAIL"
        _write_json(out / "dry_run.json", report)
    return 0 if report["status"] == "PASS" else 1


def run_reload_v4(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0") -> int:
    """Fresh process: rebuild F0, load checkpoint-final, recompute the probe. Any disagreement is a FAIL."""
    report: dict[str, Any] = {"status": "FAIL", "step": None, "tensors": None, "max_abs_delta": None, "greedy_equal": None}
    try:
        final = args.run / "checkpoint-final"
        binding = json.loads((args.run / "binding.json").read_text(encoding="utf-8"))
        metadata = ckpt.read_metadata(final, binding=binding)
        recorded = metadata["protocol"]
        contract = recorded.get("training_contract", {})
        protocol = plain_protocol_v4(args.precision, recorded.get("optimizer", {}).get("total_steps"),
                                     arm=contract.get("arm"), round_index=contract.get("round"),
                                     start=contract.get("start"), limit=contract.get("limit"),
                                     contracts=contract.get("hashes", {}))
        if (recorded != json.loads(json.dumps(protocol)) or binding["data_sha256"] != contract["hashes"]["rows"]
                or metadata["step"] != protocol["optimizer"]["total_steps"]):
            raise ValueError("reload v4: the checkpoint protocol, rows binding or final step differs")
        bundle, _, _, step, _ = ckpt.load_resume(args.model_path, final, protocol, binding=binding, device=device,
                                                 policy_factory=policy_factory)
        (assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged)(bundle)
        report.update(step=step, tensors=_tensor_counts(bundle))
        probe = json.loads((args.run / "probe.json").read_text(encoding="utf-8"))
        actual = compute_probe(bundle, _read_probe_rows(args.run, probe))
        if [len(v) for v in actual["logprobs"]] != [len(v) for v in probe["logprobs"]]:
            raise ValueError("reload v4: probe token counts differ")
        delta = max((abs(a - b) for new, old in zip(actual["logprobs"], probe["logprobs"], strict=True)
                     for a, b in zip(new, old, strict=True)), default=0.0)
        report.update(max_abs_delta=delta, greedy_equal=actual["greedy_ids"] == probe["greedy_ids"])
        if delta <= PROBE_ATOL and report["greedy_equal"]:
            report["status"] = "PASS"
    except (ValueError, OSError, KeyError, TypeError, NonFiniteStop) as exc:
        report["error"] = str(exc)
    _write_json(args.run / "reload.json", report)
    return 0 if report["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "dry-run"):
        p = sub.add_parser(name)
        for key in ("rows", "build-root", "order", "model-path", "output"):
            p.add_argument(f"--{key}", type=Path, required=True)
        for key in ("round", "start", "limit"):
            p.add_argument(f"--{key}", type=int, required=True)
        p.add_argument("--arm", choices=ARMS, required=True)
        p.add_argument("--precision", choices=("bf16", "nf4"), required=True)
        p.add_argument("--code-commit", type=_commit, required=True)
        p.add_argument("--init-run", type=Path, default=None)
        p.add_argument("--token-budget", type=int, default=0)
        p.add_argument("--max-row-tokens", type=int, default=None, help="local 8GB only; never set on the rented GPU")
        if name == "train":
            p.add_argument("--gradient-checkpointing", choices=("on", "off"), default="on")
            p.add_argument("--save-every", type=int, default=50)
            p.add_argument("--resume", action="store_true")
    reload = sub.add_parser("reload")
    reload.add_argument("--run", type=Path, required=True)
    reload.add_argument("--model-path", type=Path, required=True)
    reload.add_argument("--precision", choices=("bf16", "nf4"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reload":
        return run_reload_v4(args, policy_factory=_factory(args.precision))
    from .collect import ResourceLog, bounded_map, install_training_guards

    install_training_guards(args.build_root)
    args.output.mkdir(parents=True, exist_ok=True)
    map_fn = partial(bounded_map, workers=ENCODE_WORKERS, buffer=ENCODE_BUFFER)
    with ResourceLog(args.output / "resource.jsonl"):
        if args.command == "dry-run":
            return run_dry_run_v4(args, policy_factory=_factory(args.precision), map_fn=map_fn)
        return run_train_v4(args, policy_factory=_factory(args.precision), map_fn=map_fn)


if __name__ == "__main__":
    raise SystemExit(main())
