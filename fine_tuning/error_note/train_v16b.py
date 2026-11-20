"""v16b joint delayed-policy RL, verified self SFT, and first-four-token F0 anchor."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Iterable

import torch
from torch.utils.checkpoint import checkpoint as recompute

from fine_tuning.notebook_rl.input_preparation import digest, prepare_protocol_inputs
from fine_tuning.notebook_rl.prompts import user_message

from . import rl
from .contract import IMAGE, ROUTE_FULL, ROUTE_LM
from .train import END_TOKEN, NonFiniteStop, _gradient_norms, plan_microbatches

ROW_SCHEMA = "error-note-v16b-row-v1"
PG_KINDS = {"RL", "OBS"}
SFT_KINDS = {"N", "C", "PREP_SELF", "PREP_HUMAN"}
KINDS = PG_KINDS | SFT_KINDS
DENOMINATOR = 16 * 1024


def encode_v16b_row(processor: Any, model_config: Any, row: dict, images: list, *,
                     image: dict = IMAGE) -> dict:
    """Preserve sampled actions; only human targets are tokenized, and SFT never teaches EOS."""
    from .train_v16 import token_byte_mask
    from .train_v4 import _single_encoded_input_sha256

    tokenizer = processor.tokenizer
    kind, prompt, target = row.get("kind"), row.get("prompt"), row.get("target")
    if (row.get("schema") != ROW_SCHEMA or kind not in KINDS
            or row.get("route") != (ROUTE_FULL if images else ROUTE_LM)
            or not isinstance(prompt, str) or not prompt.strip() or not isinstance(target, str)
            or kind in SFT_KINDS and not target.strip()
            or any(token in text for token in tokenizer.all_special_tokens for text in (prompt, target))):
        raise ValueError("v16b: invalid row, image route, prompt, target or control token")
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(prompt, len(images))], images, image)
    inputs = prepared["inputs"]
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if type(end_id) is not int or end_id == tokenizer.unk_token_id:
        raise ValueError("v16b: EOS token unavailable")
    # vLLM stops on every generation_config EOS id, so a sampled terminal <|endoftext|> is an end token, never a control error.
    stop_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = {end_id} | ({stop_id} if type(stop_id) is int and stop_id != tokenizer.unk_token_id else set())
    if kind == "PREP_HUMAN":
        if "generation" in row or row.get("target_origin", "human") != "human":
            raise ValueError("v16b: PREP_HUMAN requires a dataset-authored target")
        actions = tokenizer.encode(target, add_special_tokens=False) + [end_id]
        mask = [True] * (len(actions) - 1) + [False]
    else:
        generation = row.get("generation", {})
        actions = generation.get("completion_token_ids")
        if (not isinstance(actions, list) or not actions or any(type(x) is not int or x < 0 for x in actions)
                or generation.get("text") != target or generation.get("completion_tokens", len(actions)) != len(actions)):
            raise ValueError("v16b: sampled actions, text or token count changed")
        if kind in PG_KINDS and (generation.get("prompt_token_ids") != prepared["prompt_token_ids"]
                                or generation.get("input_sha256") != prepared["input_sha256"]):
            raise ValueError("v16b: original PG input binding changed")
        mask = [x not in eos_ids for x in actions]
        if kind == "N":
            mask = token_byte_mask(tokenizer, actions, target, row.get("verified_byte_spans", []),
                                   end_token_id=actions[-1] if actions[-1] in eos_ids else end_id)
        if kind in SFT_KINDS and row.get("loss_mask") is not None and row["loss_mask"] != mask:
            raise ValueError("v16b: supervised token mask changed")
    if (len(actions) > (1024 if kind in {"N", "OBS"} else 8192)
            or inputs["input_ids"].numel() + len(actions) > 24576
            or any(getattr(model_config, name, None) in actions for name in ("image_token_id", "video_token_id"))):
        raise ValueError("v16b: target/context limit or generated image/video placeholder")
    result = row | {"input_ids": torch.cat([inputs["input_ids"][0], torch.tensor(actions, dtype=torch.long)]),
                    "target_len": len(actions), "loss_mask": torch.tensor(mask, dtype=torch.bool),
                    "end_token_id": end_id, "has_image": bool(images),
                    "pixel_values": inputs.get("pixel_values"), "image_grid_thw": inputs.get("image_grid_thw"),
                    "input_sha256": prepared["input_sha256"]}
    if _single_encoded_input_sha256(result) != prepared["input_sha256"]:
        raise ValueError("v16b: prepared image/input cannot be replayed")
    return result


def _project(head: Any, states: torch.Tensor, actions: torch.Tensor, offset: int, *, human: bool) -> torch.Tensor:
    logits = head(states).float()
    if human:
        return logits.log_softmax(-1).gather(1, actions[offset:offset + len(states), None])[:, 0]
    return rl.processed_full_logprobs(logits, actions, offset, temperature=.7, presence_penalty=1.5)


def v16b_row_logprobs(bundle: Any, rows: list[dict], *, chunk: int = 32) -> list[torch.Tensor]:
    """Project actual PolicyBundle hidden states in bounded, recomputed vocabulary chunks."""
    from .train_v4 import _single_hidden

    if type(chunk) is not int or chunk < 1 or not rows or len({r["route"] for r in rows}) != 1:
        raise ValueError("v16b: nonempty single-route rows and positive projection chunk required")
    hidden, ids, lengths = _single_hidden(bundle, rows)
    head = bundle.model.get_base_model().lm_head
    result = []
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        n = row["target_len"]
        actions = ids[index, length - n:length]
        states = hidden[index, length - n - 1:length - 1]
        pieces = []
        for offset in range(0, n, chunk):
            project = partial(_project, head, actions=actions, offset=offset, human=row["kind"] == "PREP_HUMAN")
            states_chunk = states[offset:offset + chunk]
            pieces.append(recompute(project, states_chunk, use_reentrant=False) if torch.is_grad_enabled()
                          else project(states_chunk))
        result.append(torch.cat(pieces))
    return result


def _validate_v16b_block(bundle: Any, encoded: list[dict], contract: dict) -> None:
    from .protocol import VERSION_V16B

    if (contract.get("version") != VERSION_V16B or contract.get("phase", "rl") not in {"rl", "prep"}
            or contract.get("sampling") != {"temperature": .7, "presence_penalty": 1.5, "top_p": 1., "top_k": 0}
            or contract.get("policy_sha256") != rl.policy_sha256(bundle)):
        raise ValueError("v16b: pinned version, full-vocabulary sampling and current policy binding required")
    phase, version = contract.get("phase", "rl"), contract.get("block_id")
    if type(version) is not int or version < 0:
        raise ValueError("v16b: nonnegative schedule block required")
    expected = contract.get("expected_problem_ids")
    if not isinstance(expected, list) or len(expected) != len(set(expected)):
        raise ValueError("v16b: unique expected problem IDs required")
    pg_seen: dict[str, set] = {kind: set() for kind in PG_KINDS}
    seen = set()
    for row in encoded:
        kind, pid, n = row.get("kind"), row.get("problem_id"), row.get("target_len")
        key = (kind, pid)
        if (row.get("schema") != ROW_SCHEMA or kind not in KINDS or not isinstance(pid, str) or not pid or key in seen
                or type(n) is not int or n < 1 or not isinstance(row.get("input_ids"), torch.Tensor)
                or row["input_ids"].dtype != torch.long or row["input_ids"].ndim != 1 or n >= len(row["input_ids"])
                or row.get("route") not in {ROUTE_FULL, ROUTE_LM}
                or (row.get("pixel_values") is not None) != (row["route"] == ROUTE_FULL)):
            raise ValueError("v16b: duplicate or invalid encoded row/route")
        if n > (1024 if kind in {"N", "OBS"} else 8192):
            raise ValueError("v16b: encoded target exceeds its unchanged limit")
        if row["route"] == ROUTE_FULL:
            grids, pixels = row.get("image_grid_thw"), row["pixel_values"]
            if (not isinstance(grids, torch.Tensor) or grids.dtype != torch.long or grids.ndim != 2
                    or grids.shape[1] != 3 or not len(grids) or bool((grids < 1).any())
                    or not isinstance(pixels, torch.Tensor) or int(grids.prod(-1).sum()) != len(pixels)):
                raise ValueError("v16b: image grids and pixel patch count differ")
        seen.add(key)
        if phase == "prep" and kind not in {"PREP_SELF", "PREP_HUMAN"} or phase == "rl" and kind.startswith("PREP_"):
            raise ValueError("v16b: row belongs to another training phase")
        actions = row["input_ids"][-n:].tolist()
        if kind in SFT_KINDS:
            mask = row.get("loss_mask")
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (n,)
                    or not bool(mask.any()) or row.get("end_token_id") is None
                    or bool(mask[row["input_ids"][-n:] == row["end_token_id"]].any())):
                raise ValueError("v16b: nonempty SFT mask excluding end tokens required")
        if kind != "PREP_HUMAN":
            generation = row.get("generation", {})
            expected_space = "processed-truncated-b2" if kind == "PREP_SELF" else "processed-full-vocab-v1"
            if (generation.get("completion_token_ids") != actions
                    or kind in SFT_KINDS and generation.get("finish_reason") not in {"stop", "eos"}
                    or generation.get("probability_space", row.get("probability_space")) != expected_space
                    or kind != "PREP_SELF" and generation.get("sampling_sha256") != rl.sampling_sha256(contract["sampling"])):
                raise ValueError("v16b: sampled actions or probability definition changed")
            if phase == "rl":
                policy_version = version if kind == "N" else max(0, version - 1)
                block = version - ({"N": 1, "C": 3}.get(kind, 0))
                allowed = contract.get({"N": "note_problem_ids", "C": "resolve_problem_ids",
                                        "OBS": "expected_obs_problem_ids"}.get(kind, "expected_problem_ids"), [])
                if (generation.get("policy_version") != policy_version or row.get("block_index") != block
                        or pid not in allowed):
                    raise ValueError("v16b: row differs from the sealed delayed-policy schedule")
        if kind in PG_KINDS:
            pg_seen[kind].add(pid)
            generation = row["generation"]
            mu = generation.get("sampling_logprobs")
            receipt = row.get("baseline_receipt", {})
            sealed = contract.get("baseline_receipts", {}).get(f"{kind}:{pid}")
            reward, baseline, advantage = row.get("reward"), row.get("baseline"), row.get("advantage")
            cold = kind == "OBS" and receipt.get("zero_advantage") is True and baseline is None
            sealed_reward = contract.get("reward_values", {}).get(f"{kind}:{pid}")
            if (not isinstance(mu, list) or len(mu) != n
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in mu)
                    or type(row.get("pg")) is not bool or advantage is None or type(advantage) not in (int, float) or not math.isfinite(advantage)
                    or not receipt or receipt.get("sha256") != digest({k: v for k, v in receipt.items() if k != "sha256"})
                    or sealed != receipt or receipt.get("baseline") != baseline
                    or receipt.get("history_cutoff") != row["block_index"] - 2
                    or sealed_reward != {"reward": reward, "pg": row["pg"]}
                    or row["pg"] and generation.get("finish_reason") not in {"stop", "eos"}
                    or not cold and (baseline is None or type(baseline) not in (int, float) or not math.isfinite(baseline) or not 0 <= baseline <= 1)
                    or reward is not None and (type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1)
                    or advantage != (float(reward - baseline) if row["pg"] and reward is not None and not cold else 0.)):
                raise ValueError("v16b: reward, sealed baseline, advantage or sampled probabilities changed")
    if phase == "rl" and (pg_seen["RL"] != set(expected)
                          or pg_seen["OBS"] != set(contract.get("expected_obs_problem_ids", []))):
        raise ValueError("v16b: incomplete first-solution or observation block")
    if phase == "prep" and {r["problem_id"] for r in encoded} != set(expected):
        raise ValueError("v16b: incomplete preparation block")


def _anchor(bundle: Any, rows: list[dict], denominator: int) -> torch.Tensor:
    from .train_v4 import _single_hidden, _single_raw_kl, _single_reference_hidden

    prefixes = []
    for row in rows:
        n = min(4, row["target_len"])
        prefix = row["input_ids"][:-row["target_len"]]
        prefixes.append(row | {"input_ids": torch.cat([prefix, row["input_ids"][-row["target_len"]:][:n]]), "target_len": n})
    reference = _single_reference_hidden(bundle, prefixes)
    current, _, lengths = _single_hidden(bundle, prefixes)
    head = bundle.model.get_base_model().lm_head
    terms = []
    for i, (row, length) in enumerate(zip(prefixes, lengths, strict=True)):
        n = row["target_len"]
        terms.append(_single_raw_kl(head, current[i, length - n - 1:length - 1],
                                   reference[i, length - n - 1:length - 1], 8192).mean())
    return .5 * torch.stack(terms).sum() / denominator


def backward_v16b_block(bundle: Any, encoded: list[dict], contract: dict, token_budget: int = 12000, *,
                        components: Iterable[str] | None = None, capacity: Any = None) -> dict:
    """Accumulate one joint update; HOLD clears every gradient, including earlier SFT contributions."""
    from .train_v4 import _require_single_zero_dropout

    _validate_v16b_block(bundle, encoded, contract)
    _require_single_zero_dropout(bundle)
    if type(token_budget) is not int or token_budget < 0:
        raise ValueError("v16b: nonnegative token budget required")
    selected = set(components) if components is not None else KINDS | {"ANCHOR"}
    if selected == {"PG", "KL", "A1", "A2", "B", "H"}:
        selected = KINDS | {"ANCHOR"}
    selected |= ({"RL", "OBS"} if "PG" in selected else set()) | ({"ANCHOR"} if "KL" in selected else set())
    selected -= {"PG", "KL"}
    if selected - KINDS - {"ANCHOR"}:
        raise ValueError("v16b: unknown loss component")
    plan = plan_microbatches([len(r["input_ids"]) for r in encoded], [r["route"] for r in encoded], token_budget)
    memory_bounds = contract.get("memory_shape_bounds")
    if memory_bounds is not None:
        if (not isinstance(memory_bounds, dict)
                or any(type(memory_bounds.get(key)) is not int or memory_bounds[key] < 1
                       for key in ("max_images", "max_vision_patches", "max_prompt_tokens"))):
            raise ValueError("v16b: invalid measured input-shape bounds")
        for row in encoded:
            grids = row.get("image_grid_thw")
            if (len(row["input_ids"]) - row["target_len"] > memory_bounds["max_prompt_tokens"]
                    or len(row["input_ids"]) > 24576
                    or row["route"] == ROUTE_FULL and (not isinstance(grids, torch.Tensor)
                        or len(grids) > memory_bounds["max_images"]
                        or int(grids.prod(-1).sum()) > memory_bounds["max_vision_patches"])):
                raise ValueError("v16b: training row exceeds the measured shape; no truncation or resize")
        # The CUDA probe measures one FULL row, so several image graphs cannot share that memory certificate.
        plan = [part for batch in plan for part in
                ([[i] for i in batch] if encoded[batch[0]]["route"] == ROUTE_FULL else [batch])]
    if capacity is not None:
        from .capacity import encoded_shapes

        bounded = []
        for indices in plan:
            phase = "sft_full" if encoded[indices[0]]["route"] == ROUTE_FULL else "sft_lm"
            bounded.extend(capacity.partition(encoded_shapes(encoded), [indices], phase))
        plan = bounded
    # N runs first, so its accumulated gradient norm can be measured without copying every trainable tensor.
    if any(p.grad is not None and bool(p.grad.abs().sum() > 0) for ps in bundle.params.values() for p in ps):
        raise ValueError("v16b: clear incoming gradients before a joint block")
    plan = ([[i for i in batch if encoded[i]["kind"] == "N"] for batch in plan]
            + [[i for i in batch if encoded[i]["kind"] != "N"] for batch in plan])
    plan = [batch for batch in plan if batch]
    counts = Counter(row["kind"] for row in encoded)
    prep_count = counts["PREP_SELF"] + counts["PREP_HUMAN"]
    stats: dict[str, Any] = {"rl": {"rows": counts["RL"], "pg": 0., "anchor": 0., "kl": 0.},
                             "obs": {"rows": counts["OBS"], "pg": 0.}, "sft": {}, "hold": False,
                             "HOLD": False, "task_signal": False, "has_gradient": False, "microbatches": plan}
    drift_sum, clipped, drift_n, masked, negative_n = 0., 0, 0, 0, 0
    row_excluded = 0
    anchored: set[int] = set()  # RL rows that entered the PG loss; spec 7.5 anchors only these
    losses: dict[str, float] = defaultdict(float)
    tokens: Counter = Counter()
    was_training, initial_route = bundle.model.training, bundle.route.vision_on
    try:
        bundle.model.train()
        for batch_index, indices in enumerate(plan):
            rows = [encoded[i] for i in indices]
            bundle.route.apply(rows[0]["route"] == ROUTE_FULL)
            values = v16b_row_logprobs(bundle, rows)
            terms = []
            for row, value in zip(rows, values, strict=True):
                kind, n = row["kind"], row["target_len"]
                if not bool(torch.isfinite(value).all()):
                    raise NonFiniteStop("v16b: nonfinite current log probability")
                if kind in PG_KINDS:
                    if not row["pg"]:
                        continue
                    mu = value.new_tensor(row["generation"]["sampling_logprobs"])
                    difference = (value.detach() - mu).abs()
                    rho = (value.detach() - mu).exp()
                    drift_sum += float(difference.sum())
                    clipped += int((rho > 2).sum())
                    drift_n += n
                    advantage = row["advantage"]
                    mask = torch.ones_like(value, dtype=torch.bool)
                    if advantage < 0:
                        mask = rl.negative_token_mask(value)
                        negative_n += n
                        masked += int((~mask).sum())
                    excluded = advantage < 0 and float(difference.mean()) > .3
                    row_excluded += int(excluded)
                    if excluded or advantage == 0:
                        continue
                    if kind == "RL":
                        anchored.add(id(row))
                    if kind not in selected:
                        continue
                    coefficient = (1. if kind == "RL" else .25) * advantage * min(1., 2048 / n) / DENOMINATOR
                    term = -coefficient * (rho.clamp(max=2) * value * mask).sum()
                    terms.append(term)
                    stats["rl" if kind == "RL" else "obs"]["pg"] += float(term.detach())
                    stats["task_signal"] |= bool(mask.any())
                    tokens[kind] += int(mask.sum())
                elif kind in selected:
                    mask = row["loss_mask"].to(value.device)
                    coefficient = {"N": .05, "C": .35, "PREP_SELF": 1 / max(1, prep_count),
                                   "PREP_HUMAN": 1 / max(1, prep_count)}[kind]
                    term = -coefficient * value[mask].mean()
                    terms.append(term)
                    losses[kind] += float(term.detach())
                    tokens[kind] += int(mask.sum())
                    stats["task_signal"] = True
            if terms:
                torch.stack(terms).sum().backward()
            if rows[0]["kind"] == "N" and (batch_index + 1 == len(plan) or encoded[plan[batch_index + 1][0]]["kind"] != "N"):
                stats["note_gradient_norms"] = _gradient_norms(bundle)
            anchor_rows = [r for r in rows if id(r) in anchored or r["kind"].startswith("PREP_")]
            if anchor_rows and "ANCHOR" in selected:
                term = _anchor(bundle, anchor_rows, prep_count or 16)
                if not bool(torch.isfinite(term)):
                    raise NonFiniteStop("v16b: nonfinite F0 anchor")
                term.backward()
                stats["rl"]["anchor"] += float(term.detach())
        mean = drift_sum / drift_n if drift_n else 0.
        fraction = clipped / drift_n if drift_n else 0.
        pg_rows = [r for r in encoded if r["kind"] in PG_KINDS]
        mixed = sum(bool(r["generation"].get("mixed_version")) for r in pg_rows) / len(pg_rows) if pg_rows else 0.
        reason = ("IS_MEAN" if mean > .5 else "IS_CLIP" if fraction > .1 else
                  "REPLAY" if contract.get("phase", "rl") == "rl" and contract["block_id"] == 0 and mean > .05 else
                  "MIXED_VERSION" if mixed > .05 else None)
        stats.update(hold=reason is not None, HOLD=reason is not None, hold_reason=reason,
                     gradients_discarded=reason is not None,
                     importance={"mean_abs_log_ratio": mean, "clipped_fraction": fraction, "tokens": drift_n,
                                 "row_excluded": row_excluded, "mixed_fraction": mixed},
                     ntf={"negative_tokens": negative_n, "masked_tokens": masked,
                          "masked_fraction": masked / negative_n if negative_n else 0.})
        if reason:
            bundle.model.zero_grad(set_to_none=True)
            stats["task_signal"] = False
        norms = _gradient_norms(bundle)
        if any(value is not None and not math.isfinite(value) for value in norms.values()):
            raise NonFiniteStop("v16b: nonfinite gradient")
        stats.update(gradient_norms=norms, has_gradient=not reason and any(v is not None and v > 0 for v in norms.values()),
                     note_gradient_retained=not reason and any(v is not None and v > 0
                                                             for v in stats.get("note_gradient_norms", {}).values()))
    except BaseException:
        bundle.model.zero_grad(set_to_none=True)
        raise
    finally:
        bundle.model.train(was_training)
        bundle.route.apply(initial_route)
    stats["sft"] = {kind: {"rows": counts[kind], "tokens": tokens[kind], "weighted_loss": value}
                    for kind, value in losses.items()}
    stats["loss"] = stats["rl"]["pg"] + stats["obs"]["pg"] + stats["rl"]["anchor"] + sum(losses.values())
    advantages = [r["advantage"] for r in encoded if r["kind"] in PG_KINDS and r["pg"]]
    stats["rl"].update(tokens=tokens["RL"], denominator=DENOMINATOR,
                       advantage_positive=sum(a > 0 for a in advantages),
                       advantage_negative=sum(a < 0 for a in advantages),
                       advantage_mean=sum(advantages) / len(advantages) if advantages else 0.)
    stats["ntf"]["outside_expected_range"] = bool(negative_n and not .8 <= masked / negative_n <= .95)
    stats["obs"]["tokens"] = tokens["OBS"]
    return stats


# Fixed production bounds; CPU tests patch these constants together with the CUDA readings.
MEMORY_PROBE_ACTIONS = 8192
MEMORY_PROBE_PROMPT = 4000


def measure_update_memory_v16b(bundle: Any, encoded_row: dict, *, shape_bounds: dict,
                               max_target_tokens: int = MEMORY_PROBE_ACTIONS,
                               max_input_tokens: int = MEMORY_PROBE_PROMPT) -> dict:
    """Measure one bounded BF16 FULL backward, preserving weights, RNG, incoming gradients and route."""
    from fine_tuning.notebook_rl.checkpoint import capture_rng, restore_rng
    from fine_tuning.notebook_rl.policy import assert_base_unchanged
    from .capacity import CapacityHold, memory
    from .train import _all_parameter_hashes
    from .v16b_runtime import kv_budget_v16b

    base = bundle.model.get_base_model()
    pixels, grids = encoded_row.get("pixel_values"), encoded_row.get("image_grid_thw")
    ids, n = encoded_row.get("input_ids"), encoded_row.get("target_len")
    if (base.get_input_embeddings().weight.dtype != torch.bfloat16
            or getattr(base, "is_loaded_in_4bit", False) or getattr(base, "is_loaded_in_8bit", False)
            or encoded_row.get("route") != ROUTE_FULL or not isinstance(pixels, torch.Tensor)
            or not isinstance(grids, torch.Tensor) or grids.ndim != 2 or grids.shape[1] != 3
            or grids.dtype != torch.long or bool((grids < 1).any())
            or not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 1
            or type(n) is not int or not 0 < n < len(ids)
            or type(max_target_tokens) is not int or max_target_tokens < MEMORY_PROBE_ACTIONS
            or type(max_input_tokens) is not int or max_input_tokens < MEMORY_PROBE_PROMPT):
        raise ValueError("v16b memory: BF16 FULL image specimen and production-length bounds required")
    if (any(type(shape_bounds.get(key)) is not int or shape_bounds[key] < 1
            for key in ("max_images", "max_vision_patches", "max_prompt_tokens"))
            or not isinstance(shape_bounds.get("dataset_sha256"), str)
            or len(shape_bounds["dataset_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in shape_bounds["dataset_sha256"])
            or len(grids) < shape_bounds["max_images"]
            or int(grids.prod(-1).sum()) < shape_bounds["max_vision_patches"]
            or int(grids.prod(-1).sum()) != len(pixels)):
        raise ValueError("v16b memory: specimen must cover the sealed image-count and visual-patch bounds")
    prompt_tokens = max(max_input_tokens, shape_bounds["max_prompt_tokens"], len(ids) - n)
    if prompt_tokens + max_target_tokens > 24576:
        raise ValueError("v16b memory: bounded prompt and actions exceed the unchanged context limit")
    banned = set(getattr(getattr(bundle.processor, "tokenizer", None), "all_special_ids", []))
    for config in (base.config, base.config.text_config):
        for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
                    "eos_token_id", "pad_token_id"):
            value = getattr(config, key, None)
            banned.update(value if isinstance(value, (list, tuple)) else [value])
    safe = next((int(value) for value in ids[-n:] if int(value) not in banned), None)
    if safe is None:
        raise ValueError("v16b memory: a non-control synthetic action token is required")
    probe = encoded_row | {
        "kind": "RL", "target_len": max_target_tokens,
        "input_ids": torch.cat([ids[:-n], ids.new_full((prompt_tokens - len(ids) + n + max_target_tokens,), safe)]),
    }
    device = next(base.parameters()).device
    memory(bundle)  # Reject CPU and invalid/oversubscribed physical readings before changing working state.
    assert_base_unchanged(bundle)
    before = _all_parameter_hashes(bundle)
    parameters = list(bundle.model.parameters())
    gradients = [parameter.grad for parameter in parameters]
    flags = [parameter.requires_grad for parameter in parameters]
    modes = [(module, module.training) for module in bundle.model.modules()]
    route = bundle.route.vision_on
    rng = capture_rng()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    reservation = []
    try:
        for parameter in parameters:
            parameter.grad = None
        bundle.route.apply(True)
        bundle.model.train()
        torch.use_deterministic_algorithms(True)
        # The later AdamW moments are resident during updates; reserve their exact tensor layout without an optimizer.
        reservation = [torch.empty_like(parameter) for group in bundle.params.values()
                       for parameter in group for _ in range(2)]
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        baseline = memory(bundle)
        if baseline["reserved"] > 10.5 * 2**30:
            raise CapacityHold("v16b memory: HF residency exceeds the fixed 10.5 GiB budget assumption")
        torch.cuda.reset_peak_memory_stats(device)
        value = v16b_row_logprobs(bundle, [probe])[0]
        (-value.sum() / DENOMINATOR).backward()
        del value
        anchor = _anchor(bundle, [probe], 16)
        anchor.backward()
        del anchor
        norms = _gradient_norms(bundle)
        torch.cuda.synchronize(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        if not baseline["reserved"] <= peak_reserved <= baseline["total"]:
            raise CapacityHold("v16b memory: invalid measured allocator peak")
        extra = peak_reserved - baseline["reserved"]
        try:
            kv = kv_budget_v16b(extra / 2**30)
        except RuntimeError as exc:
            raise CapacityHold(f"v16b memory: measured X={extra / 2**30:.6f} GiB; {exc}") from exc
        report = {
            "version": "v16b-update-memory-v1", "measurement": "actual_cuda_bf16_backward_only",
            "shape_bounds": dict(shape_bounds), "shape_bounds_sha256": digest(shape_bounds),
            "prompt_tokens": prompt_tokens, "action_tokens": max_target_tokens, "rows": 1,
            "image_grids": grids.tolist(), "vision_patches": int(grids.prod(-1).sum()),
            "baseline": baseline, "peak_reserved_bytes": peak_reserved, "peak_allocated_bytes": peak_allocated,
            "extra_reserved_bytes": extra, "extra_reserved_gib": extra / 2**30, "x_gib": extra / 2**30,
            "resident_moment_reservation_bytes": sum(t.numel() * t.element_size() for t in reservation),
            "kv_cache_gib": kv, "gradient_norms": norms, "optimizer_steps": 0,
            "scope": "one FULL row within the sealed input bounds; larger packed shapes are unmeasured",
        }
        return report | {"sha256": digest(report)}
    finally:
        bundle.model.zero_grad(set_to_none=True)
        reservation.clear()
        for parameter, gradient, flag in zip(parameters, gradients, flags, strict=True):
            parameter.grad = gradient
            parameter.requires_grad_(flag)
        bundle.route.apply(route)
        for module, training in modes:
            module.training = training
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        restore_rng(rng)
        assert_base_unchanged(bundle)
        if _all_parameter_hashes(bundle) != before:
            raise ValueError("v16b memory: parameters changed during the backward-only measurement")
