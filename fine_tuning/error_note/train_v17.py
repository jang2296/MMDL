"""v17 fixed-parent RL, raw self-SFT, and stratified full-vocabulary F0 KL."""
from __future__ import annotations

import hashlib
import math
import statistics
import time
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
from .train_v16 import _restore_state, token_byte_mask

ROW_SCHEMA = "error-note-v17-row-v1"
PG_KINDS = {"RL", "OBS"}
SFT_KINDS = {"A1", "C", "N", "PREP_SELF", "PREP_HUMAN"}
KINDS = PG_KINDS | SFT_KINDS
SAMPLING = {"temperature": .7, "presence_penalty": 1.5, "top_p": 1., "top_k": 0}
F0_POSITION_SALT = "v17-f0-stratified-64-v1"
DENOMINATOR = 16 * 1024
SFT_WEIGHTS = {"A1": .20, "C": .35, "N": .05, "PREP_SELF": 1., "PREP_HUMAN": 1.}


class RowEncodingError(ValueError):
    """An individual mask or replay-size failure, with provenance checks still enforced."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _end_ids(tokenizer: Any) -> set[int]:
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if type(end) is not int or end == tokenizer.unk_token_id:
        raise ValueError("v17: terminal token unavailable")
    other = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return {end} | ({other} if type(other) is int and other != tokenizer.unk_token_id else set())


def _sample_mask(tokenizer: Any, actions: list[int], text: str, end_ids: set[int],
                 spans: list[list[int]] | None = None) -> list[bool]:
    terminal = actions[-1] if actions and actions[-1] in end_ids else min(end_ids)
    if not text:
        if actions not in ([], [terminal]):
            raise ValueError("v17: empty text has nonterminal sampled actions")
        return [False] * len(actions)
    try:
        return token_byte_mask(tokenizer, actions, text,
                               spans if spans is not None else [[0, len(text.encode("utf-8"))]],
                               end_token_id=terminal)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("v17: invalid sampled token or unavailable byte identity") from error


def encoded_input_sha256_v17(row: dict) -> str:
    from fine_tuning.notebook_rl.input_preparation import tensor_identity

    n = row["target_len"]
    prompt = (row["input_ids"][:-n] if n else row["input_ids"])[None]
    values = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
    values.update({key: row[key] for key in ("pixel_values", "image_grid_thw") if row.get(key) is not None})
    return digest({key: tensor_identity(value) for key, value in values.items()})


def _prepare_row_inputs(processor: Any, model_config: Any, prompt: str, images: list, image: dict) -> dict:
    if images:
        return prepare_protocol_inputs(processor, model_config, [user_message(prompt, len(images))], images, image)[0]
    from fine_tuning.notebook_rl.input_preparation import tensor_identity

    chat = processor.apply_chat_template([user_message(prompt, 0)], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], return_tensors="pt")
    return {"inputs": inputs, "prompt_token_ids": inputs["input_ids"][0].tolist(),
            "input_sha256": digest({key: tensor_identity(value) for key, value in inputs.items()})}


def _validate_sft_receipt(row: dict) -> None:
    kind = row["kind"]
    if kind not in SFT_KINDS:
        return
    receipt = row.get("verification_receipt", {})
    sha = receipt.get("sha256")
    if (not receipt or sha != digest({k: v for k, v in receipt.items() if k != "sha256"})
            or row.get("admission_receipt_sha256") != sha or row.get("verifier_receipt_sha256") != sha
            or receipt.get("accepted") is not True):
        raise ValueError("v17: supervised admission/verifier receipt changed")
    if kind == "PREP_HUMAN":
        if (receipt.get("target_sha256") != hashlib.sha256(row["target"].encode()).hexdigest()
                or receipt.get("human_verified") is not False or not receipt.get("native_reference_sha256")
                or not receipt.get("text_check_receipt")):
            raise ValueError("v17: native human reference provenance changed")
        return
    binding = {"generation": row["generation"], "text": row["target"],
               "prompt_sha256": row.get("generation_prompt_sha256")}
    if receipt.get("problem_id") != row.get("problem_id") or receipt.get("output_binding_sha256") != digest(binding):
        raise ValueError("v17: sampled output differs from its admission receipt")
    if kind == "N":
        if (receipt.get("n_accepted") is not True or receipt.get("verified_repair_count", 0) < 1
                or receipt.get("verified_byte_spans") != row.get("verified_byte_spans")
                or receipt.get("verified_content_token_mask") != row.get("verified_content_token_mask")):
            raise ValueError("v17: note receipt has no bound verified corrected content")
    else:
        expected_stage = "resolve" if kind == "C" else "t1"
        binding = {"problem_id": row["problem_id"], "p0_text": row["prompt"],
                   "image_paths": row.get("image_paths", []), "image_sha256": row.get("image_sha256", [])}
        if receipt.get("stage") != expected_stage or receipt.get("record_binding_sha256") != digest(binding):
            raise ValueError("v17: supervised parent or P0 input differs from its receipt")


def encode_v17_row(processor: Any, model_config: Any, row: dict, images: list, *,
                   image: dict = IMAGE) -> dict:
    """Validate original token bytes and kind-specific masks without retokenizing sampled actions."""
    tokenizer = processor.tokenizer
    kind, prompt, target = row.get("kind"), row.get("prompt"), row.get("target")
    if (row.get("schema") != ROW_SCHEMA or kind not in KINDS
            or row.get("route") != (ROUTE_FULL if images else ROUTE_LM)
            or not isinstance(prompt, str) or not prompt.strip() or not isinstance(target, str)):
        raise ValueError("v17: invalid row, route, prompt, target or control token")
    _validate_sft_receipt(row)
    if row.get("training_prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError("v17: training prompt binding changed")
    if (kind in SFT_KINDS and not target.strip()
            or any(token in text for token in tokenizer.all_special_tokens for text in (prompt, target))):
        raise RowEncodingError("UNREPLAYABLE_CONTENT", "v17: empty SFT target or literal control token")
    prepared = _prepare_row_inputs(processor, model_config, prompt, images, image)
    inputs = prepared["inputs"]
    ends = _end_ids(tokenizer)
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if kind == "PREP_HUMAN":
        if "generation" in row or row.get("target_origin") != "human":
            raise ValueError("v17: human target must have dataset provenance and no generated actions")
        actions = tokenizer.encode(target, add_special_tokens=False) + [end_id]
        mask = [True] * (len(actions) - 1) + [False]
    else:
        generation = row.get("generation", {})
        actions = generation.get("completion_token_ids")
        if (not isinstance(actions, list) or any(type(x) is not int or x < 0 for x in actions)
                or not actions and kind != "RL" or generation.get("text") != target
                or generation.get("completion_tokens", len(actions)) != len(actions)):
            raise ValueError("v17: sampled token identity or text changed")
        mask = _sample_mask(tokenizer, actions, target, ends,
                            row.get("verified_byte_spans", []) if kind == "N" else None)
        if kind in PG_KINDS and (generation.get("prompt_token_ids") != prepared["prompt_token_ids"]
                                or generation.get("input_sha256") != prepared["input_sha256"]):
            raise ValueError("v17: original PG input binding changed")
        if kind == "RL" and row.get("reason") == "LOOP_CONFIRMED":
            start = len(target[:-3000].encode("utf-8"))
            mask = _sample_mask(tokenizer, actions, target, ends, [[start, len(target.encode("utf-8"))]])
        elif kind in PG_KINDS:
            mask = [True] * len(actions)
    for name in ("loss_mask", "loss_token_mask"):
        supplied = row.get(name)
        if isinstance(supplied, torch.Tensor):
            supplied = supplied.tolist()
        if supplied is not None and supplied != mask:
            raise RowEncodingError("TOKEN_MASK", "v17: independently recomputed token mask changed")
    if kind == "N" and row.get("verified_content_token_mask") != mask:
        raise RowEncodingError("NOTE_CONTENT_MASK", "v17: note content mask changed")
    if kind in {"A1", "C", "PREP_SELF"} and row.get("mask_mode") != "all-non-eos":
        raise RowEncodingError("SFT_MASK_MODE", "v17: self SFT must use all nonterminal actions")
    if kind == "N" and row.get("mask_mode") != "verified-corrected-value-content-only-v1":
        raise RowEncodingError("NOTE_MASK_MODE", "v17: note SFT requires verified corrected content")
    if (len(actions) > (1024 if kind in {"N", "OBS"} else 8192)
            or inputs["input_ids"].numel() + len(actions) > 24576
            or any(getattr(model_config, name, None) in actions for name in ("image_token_id", "video_token_id"))):
        raise RowEncodingError("REPLAY_LIMIT", "v17: target/context limit or generated image/video token")
    result = row | {"input_ids": torch.cat([inputs["input_ids"][0], torch.tensor(actions, dtype=torch.long)]),
                    "target_len": len(actions), "loss_mask": torch.tensor(mask, dtype=torch.bool),
                    "loss_token_mask": torch.tensor(mask, dtype=torch.bool),
                    "end_token_id": end_id, "end_token_ids": sorted(ends), "has_image": bool(images),
                    "pixel_values": inputs.get("pixel_values"), "image_grid_thw": inputs.get("image_grid_thw"),
                    "input_sha256": prepared["input_sha256"]}
    if encoded_input_sha256_v17(result) != prepared["input_sha256"]:
        raise ValueError("v17: prepared input cannot be replayed")
    return result


def _project(head: Any, states: torch.Tensor, actions: torch.Tensor, offset: int) -> torch.Tensor:
    logits = head(states).float()
    raw = logits.log_softmax(-1).gather(1, actions[offset:offset + len(states), None])[:, 0]
    return torch.stack([raw, rl.processed_full_logprobs(logits, actions, offset, **{
        "temperature": .7, "presence_penalty": 1.5})])


def _row_values(bundle: Any, rows: list[dict], *, chunk: int = 32):
    from .train_v4 import _single_hidden

    hidden, ids, lengths = _single_hidden(bundle, rows)
    head = bundle.model.get_base_model().lm_head
    result = []
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        n = row["target_len"]
        actions = ids[index, length - n:length]
        states = hidden[index, length - n - 1:length - 1]
        pieces = []
        for offset in range(0, n, chunk):
            project = partial(_project, head, actions=actions, offset=offset)
            part = states[offset:offset + chunk]
            pieces.append(recompute(project, part, use_reentrant=False) if torch.is_grad_enabled() else project(part))
        result.append((torch.cat(pieces, dim=1), states))
    return result


def v17_row_logprobs(bundle: Any, rows: list[dict], *, chunk: int = 32) -> list[torch.Tensor]:
    """Return raw log p for SFT and processed log q for RL, from the same chunked projection."""
    if type(chunk) is not int or chunk < 1 or not rows or len({r["route"] for r in rows}) != 1:
        raise ValueError("v17: nonempty single-route rows and positive projection chunk required")
    return [values[1 if row["kind"] in PG_KINDS else 0]
            for row, (values, _) in zip(rows, _row_values(bundle, rows, chunk=chunk), strict=True)]


def f0_positions_v17(n: int, problem_id: str, *, master_seed: int = 3407,
                     salt: str = F0_POSITION_SALT) -> list[dict]:
    """One probability-independent hash sample per contiguous bin, weighted by bin length."""
    if type(n) is not int or n < 0 or not isinstance(problem_id, str) or not problem_id:
        raise ValueError("v17: valid action count and parent ID required for F0 positions")
    bins = min(64, n)
    result = []
    for index in range(bins):
        start, stop = index * n // bins, (index + 1) * n // bins
        value = hashlib.sha256(f"{master_seed}:{problem_id}:{salt}:{index}".encode()).digest()
        result.append({"position": start + int.from_bytes(value[:8], "big") % (stop - start),
                       "start": start, "stop": stop, "weight": (stop - start) / n})
    return result


def _mixed(row: dict) -> bool:
    def mixed(value: Any) -> bool:
        if isinstance(value, dict):
            return bool(value.get("mixed_version")) or any(mixed(v) for v in value.values())
        return isinstance(value, (list, tuple)) and any(mixed(v) for v in value)
    return bool(row.get("mixed_version")) or mixed({
        key: row[key] for key in ("generation", "source_generation", "source_generations") if key in row})


def validate_v17_block(bundle: Any, encoded: list[dict], contract: dict) -> None:
    """Reject altered sampling, delayed schedule, parent identities, masks and sealed rewards."""
    if (contract.get("version") != "error-note-v17" or contract.get("phase", "rl") not in {"rl", "prep"}
            or contract.get("sampling") != SAMPLING or contract.get("policy_sha256") != rl.policy_sha256(bundle)
            or contract.get("f0_position_salt") != F0_POSITION_SALT or contract.get("master_seed") != 3407):
        raise ValueError("v17: sealed version, sampling, policy and F0 position contract required")
    phase, version = contract.get("phase", "rl"), contract.get("block_id")
    expected = contract.get("expected_problem_ids")
    if (type(version) is not int or version < 0 or not isinstance(expected, list)
            or len(expected) != len(set(expected)) or len(expected) > 16):
        raise ValueError("v17: valid slot and at most sixteen unique expected parents required")
    seen: set[tuple[str, str]] = set()
    pg_seen: dict[str, set[str]] = {kind: set() for kind in PG_KINDS}
    for row in encoded:
        kind, pid, n = row.get("kind"), row.get("problem_id"), row.get("target_len")
        ids, mask = row.get("input_ids"), row.get("loss_mask")
        if (row.get("schema") != ROW_SCHEMA or kind not in KINDS or not isinstance(pid, str) or not pid
                or row.get("row_id") != f"{pid}:{kind}" or row.get("parent_id", pid) != pid
                or (kind, pid) in seen or type(n) is not int or n < 0 or not n and kind != "RL"
                or not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 1
                or n >= len(ids) or len(ids) > 24576
                or row.get("route") not in {ROUTE_FULL, ROUTE_LM}
                or (row.get("pixel_values") is not None) != (row["route"] == ROUTE_FULL)
                or n > (1024 if kind in {"N", "OBS"} else 8192)
                or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (n,)):
            raise ValueError("v17: duplicate or invalid encoded row, route, action count or mask")
        seen.add((kind, pid))
        _validate_sft_receipt(row)
        if row.get("input_sha256") != encoded_input_sha256_v17(row):
            raise ValueError("v17: encoded prompt/image identity changed")
        if row["route"] == ROUTE_FULL:
            grids, pixels = row.get("image_grid_thw"), row["pixel_values"]
            if (not isinstance(grids, torch.Tensor) or grids.dtype != torch.long or grids.ndim != 2
                    or grids.shape[1] != 3 or not len(grids) or bool((grids < 1).any())
                    or not isinstance(pixels, torch.Tensor) or int(grids.prod(-1).sum()) != len(pixels)):
                raise ValueError("v17: image grids and pixel patches differ")
        if (phase == "prep") != kind.startswith("PREP_"):
            raise ValueError("v17: row belongs to another phase")
        if row.get("loss_probability_space") != ("processed-full-vocab-v1" if kind in PG_KINDS else "raw-full-vocab-v1"):
            raise ValueError("v17: wrong loss probability space")
        actions = ids[-n:].tolist() if n else []
        if kind in SFT_KINDS:
            ends = row.get("end_token_ids", [row.get("end_token_id")])
            if (not bool(mask.any()) or any(bool(mask[i]) for i, action in enumerate(actions) if action in ends)
                    or any(name in row for name in ("reward", "baseline", "advantage", "baseline_receipt", "pg"))):
                raise ValueError("v17: SFT requires nonempty content, no terminal loss and no PG fields")
            if kind == "N":
                content = row.get("verified_content_token_mask")
                content = content.tolist() if isinstance(content, torch.Tensor) else content
                if content != mask.tolist() or row.get("mask_mode") != "verified-corrected-value-content-only-v1":
                    raise ValueError("v17: note content mask changed")
            elif kind != "PREP_HUMAN" and mask.tolist() != [action not in ends for action in actions]:
                raise ValueError("v17: self SFT must include every nonterminal sampled token")
        if kind == "PREP_HUMAN":
            if "generation" in row or row.get("target_origin") != "human":
                raise ValueError("v17: invalid dataset-authored target")
            continue
        generation = row.get("generation", {})
        space = "processed-truncated-b2" if kind == "PREP_SELF" else "processed-full-vocab-v1"
        loop = kind == "RL" and row.get("reason") == "LOOP_CONFIRMED"
        if (generation.get("completion_token_ids") != actions or generation.get("probability_space") != space
                or kind != "PREP_SELF" and generation.get("sampling_sha256") != rl.sampling_sha256(SAMPLING)
                or kind in SFT_KINDS and generation.get("finish_reason") not in {"stop", "eos"}):
            raise ValueError("v17: original generation identity, sampling or completion changed")
        if phase == "rl":
            source_block = version - {"N": 1, "C": 3}.get(kind, 0)
            allowed = contract.get({"N": "note_problem_ids", "C": "resolve_problem_ids", "OBS": "expected_obs_problem_ids"}.get(kind, "expected_problem_ids"), [])
            if (row.get("block_index") != source_block or pid not in allowed
                    or generation.get("policy_version") != (version if kind == "N" else max(0, version - 1))):
                raise ValueError("v17: row differs from delayed-policy slot")
        if kind in PG_KINDS:
            pg_seen[kind].add(pid)
            receipt = row.get("baseline_receipt", {})
            reward, baseline, advantage = row.get("reward"), row.get("baseline"), row.get("advantage")
            cold = kind == "OBS" and receipt.get("zero_advantage") is True and baseline is None
            mu = generation.get("sampling_logprobs")
            if (not isinstance(mu, list) or len(mu) != n
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in mu)
                    or type(row.get("pg")) is not bool or advantage is None
                    or type(advantage) not in (int, float) or not math.isfinite(advantage)
                    or receipt.get("sha256") != digest({k: v for k, v in receipt.items() if k != "sha256"})
                    or contract.get("baseline_receipts", {}).get(f"{kind}:{pid}") != receipt
                    or receipt.get("baseline") != baseline or receipt.get("history_cutoff") != row["block_index"] - 2
                    or contract.get("reward_values", {}).get(f"{kind}:{pid}") != {"reward": reward, "pg": row["pg"]}
                    or row["pg"] and (not n or generation.get("finish_reason") not in {"stop", "eos"} and not loop)
                    or not cold and (type(baseline) not in (int, float) or not math.isfinite(baseline) or not 0 <= baseline <= 1)
                    or reward is not None and (type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1)
                    or advantage != (float(reward - baseline) if row["pg"] and reward is not None and not cold else 0.)):
                raise ValueError("v17: reward, sealed baseline, advantage or behavior probabilities changed")
            if loop:
                from .contract import is_loop
                if reward != 0 or not row["pg"] or not is_loop(row.get("target", "")) or not bool(mask.any()):
                    raise ValueError("v17: confirmed-loop negative PG requires actual detector evidence")
            elif mask.tolist() != [True] * n:
                raise ValueError("v17: ordinary PG action mask includes sampled terminals")
    if phase == "rl" and (pg_seen["RL"] != set(expected)
                          or pg_seen["OBS"] != set(contract.get("expected_obs_problem_ids", []))):
        raise ValueError("v17: incomplete T1 or observation block")
    if phase == "prep" and {r["problem_id"] for r in encoded} != set(expected):
        raise ValueError("v17: incomplete preparation block")
    first = {r["problem_id"]: r for r in encoded if r["kind"] == "RL"}
    for row in encoded:
        if row["kind"] == "A1":
            original = first.get(row["problem_id"])
            if original is None or original.get("reason") == "LOOP_CONFIRMED" or _share_key(original) != _share_key(row):
                raise ValueError("v17: A1 must bind the same T1 and P0 training input")


_validate_v17_block = validate_v17_block


def _share_key(row: dict) -> tuple:
    generation = row.get("generation", {})
    return (generation.get("policy_sha256"), row.get("input_sha256"), row["route"],
            digest(row["input_ids"].tolist()), digest(generation.get("completion_token_ids")))


def _measurement_clock(device: torch.device, timings: dict | None) -> float:
    if timings is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _kl(bundle: Any, row: dict, states: torch.Tensor, contract: dict, *, prep: bool,
        timings: dict | None = None, force_full: bool = False) -> tuple[torch.Tensor, list[dict]]:
    from .train_v4 import _single_raw_kl, _single_reference_hidden

    n = row["target_len"]
    positions = ([{"position": i, "weight": 1 / min(4, n)} for i in range(min(4, n))] if prep
                 else f0_positions_v17(n, row["problem_id"], master_seed=contract["master_seed"], salt=contract["f0_position_salt"]))
    stop = max(p["position"] for p in positions) + 1
    prefix = row["input_ids"][:-n]
    bounded = row | {"input_ids": torch.cat([prefix, row["input_ids"][-n:][:stop]]), "target_len": stop}
    started = _measurement_clock(states.device, timings)
    reference = _single_reference_hidden(bundle, [bounded])
    elapsed = _measurement_clock(states.device, timings) - started
    if timings is not None:
        timings["f0_prefix_forward_seconds"] = timings.get("f0_prefix_forward_seconds", 0.) + elapsed
    indices = torch.tensor([p["position"] for p in positions], device=states.device)
    ref_states = reference[0, len(prefix) - 1:len(prefix) + stop - 1].index_select(0, indices)
    current = states.index_select(0, indices)
    weights = states.new_tensor([p["weight"] for p in positions], dtype=torch.float32)
    started = _measurement_clock(states.device, timings)
    values = _single_raw_kl(bundle.model.get_base_model().lm_head, current, ref_states, 8192, force_full=force_full)
    elapsed = _measurement_clock(states.device, timings) - started
    if timings is not None:
        timings["f0_vocabulary_projection_seconds"] = timings.get("f0_vocabulary_projection_seconds", 0.) + elapsed
    return (values * weights).sum(), positions


def _component_gradients(bundle: Any, terms: dict[str, torch.Tensor], accumulated: dict) -> dict:
    active = [(group, index, parameter) for group, params in bundle.params.items()
              for index, parameter in enumerate(params) if parameter.requires_grad]
    norms: dict[str, dict[str, float]] = {}
    for kind, term in terms.items():
        gradients = torch.autograd.grad(term, [p for _, _, p in active], retain_graph=True, allow_unused=True)
        squares = dict.fromkeys(bundle.params, 0.)
        for (group, index, _), gradient in zip(active, gradients, strict=True):
            if gradient is not None:
                key = (kind, group, index)
                value = gradient.detach().float().cpu()
                accumulated[key] = accumulated.get(key, 0) + value
                squares[group] += float(value.square().sum())
        norms[kind] = {group: math.sqrt(value) for group, value in squares.items()}
    return norms


def backward_v17_block(bundle: Any, encoded: list[dict], contract: dict, token_budget: int = 12000, *,
                       components: Iterable[str] | None = None, capacity: Any = None,
                       measurement_timings: dict | None = None, measure_component_gradients: bool | None = None,
                       note_gradient_capture: dict | None = None) -> dict:
    """Accumulate one joint update, sharing identical RL/A1 graphs and clearing all gradients on HOLD."""
    from .train_v4 import _require_single_zero_dropout

    is_v18 = contract.get("version") in ("error-note-v18", "error-note-v18b")
    if is_v18:
        from .train_v18 import sft_term_v18, validate_v18_block

        validate_v18_block(bundle, encoded, contract)
    else:
        validate_v17_block(bundle, encoded, contract)
    _require_single_zero_dropout(bundle)
    if type(token_budget) is not int or token_budget < 0:
        raise ValueError("v17: nonnegative microbatch token budget required")
    if any(p.grad is not None and bool(p.grad.abs().sum() > 0) for ps in bundle.params.values() for p in ps):
        raise ValueError("v17: clear incoming gradients before joint backward")
    selected = set(components) if components is not None else KINDS | {"KL"}
    if selected == {"PG", "KL", "A1", "A2", "B", "H"}:
        selected = KINDS | {"KL"}
    if "PG" in selected:
        selected = selected - {"PG"} | PG_KINDS
    if "ANCHOR" in selected:
        selected = selected - {"ANCHOR"} | {"KL"}
    if selected - KINDS - {"KL"}:
        raise ValueError("v17: unknown loss component")
    stats: dict[str, Any] = {"hold": False, "HOLD": False, "hold_reason": None, "task_signal": False,
                             "has_gradient": False, "rl": {"pg": 0., "anchor": 0., "kl": 0.},
                             "obs": {"pg": 0.}, "sft": {}, "f0_positions": {}, "consumed_row_ids": [],
                             "t1_negative_candidate_tokens": 0, "t1_negative_retained_tokens": 0,
                             "t1_negative_retained_ratio": 0., "loop_confirmed": 0,
                             "loop_tokens_penalized": 0, "length_nonloop_excluded": 0}
    mixed = [r["row_id"] for r in encoded if _mixed(r)]
    if mixed:
        bundle.model.zero_grad(set_to_none=True)
        return stats | {"hold": True, "HOLD": True, "hold_reason": "MIXED_VERSION", "gradients_discarded": True,
                        "mixed_row_ids": mixed, "gradient_norms": _gradient_norms(bundle), "loss": 0.,
                        "importance": {"mixed_fraction": len(mixed) / len(encoded)}}
    counts = Counter(r["kind"] for r in encoded)
    prep = contract.get("phase", "rl") == "prep"
    denominator = len(encoded) if prep else 16
    # Share only identical T1/A1 inputs, policy and actions within this joint update.
    groups: list[list[dict]] = []
    shared: dict[tuple, list[dict]] = {}
    for row in encoded:
        if not row["target_len"]:
            stats["f0_positions"][row["problem_id"]] = {"status": "KL_EMPTY_RESPONSE", "positions": []}
            continue
        key = _share_key(row) if row["kind"] in {"RL", "A1"} else None
        if key is not None and key in shared:
            shared[key].append(row)
        else:
            group = [row]
            groups.append(group)
            if key is not None:
                shared[key] = group
    representatives = [group[0] for group in groups]
    plan = plan_microbatches([len(r["input_ids"]) for r in representatives], [r["route"] for r in representatives], token_budget)
    bounds = contract.get("memory_shape_bounds")
    if bounds is not None:
        if (not isinstance(bounds, dict) or any(type(bounds.get(k)) is not int or bounds[k] < 1
                                               for k in ("max_images", "max_vision_patches", "max_prompt_tokens"))):
            raise ValueError("v17: invalid measured shape bounds")
        for row in representatives:
            grids = row.get("image_grid_thw")
            if (len(row["input_ids"]) - row["target_len"] > bounds["max_prompt_tokens"]
                    or len(row["input_ids"]) > 24576 or row["route"] == ROUTE_FULL and
                    (not isinstance(grids, torch.Tensor) or len(grids) > bounds["max_images"]
                     or int(grids.prod(-1).sum()) > bounds["max_vision_patches"])):
                raise ValueError("v17: row exceeds measured shape; no truncation or resize")
        plan = [part for batch in plan for part in
                ([[i] for i in batch] if representatives[batch[0]]["route"] == ROUTE_FULL else [batch])]
    if capacity is not None:
        from .capacity import encoded_shapes
        plan = [part for batch in plan for part in capacity.partition(encoded_shapes(representatives), [batch],
                "sft_full" if representatives[batch[0]]["route"] == ROUTE_FULL else "sft_lm")]
    indices_by_identity = {id(row): i for i, row in enumerate(encoded)}
    stats["microbatches"] = [[indices_by_identity[id(groups[i][0])] for i in part] for part in plan]
    stats["shared_t1_a1_pairs"] = sum(len(group) == 2 for group in groups)
    losses: dict[str, float] = defaultdict(float)
    tokens: Counter[str] = Counter()
    gradients: dict[tuple, torch.Tensor] = {}
    diagnosed_channels: set[str] = set()
    note_parent_gradients: dict[str, dict[str, float]] = {}
    note_row_gradients: dict[str, dict[str, float]] = {}
    drift_sum, drift_n, clipped, excluded_n, negative_n, masked_n, loop_n = 0., 0, 0, 0, 0, 0, 0
    policy_mask_counts: dict[str, Counter] = defaultdict(Counter)
    process_tokens: Counter[str] = Counter()
    try:
        with _restore_state(bundle):
            bundle.model.train()
            for indices in plan:
                rows = [representatives[i] for i in indices]
                bundle.route.apply(rows[0]["route"] == ROUTE_FULL)
                device = next(bundle.model.parameters()).device
                started = _measurement_clock(device, measurement_timings)
                values = _row_values(bundle, rows)
                elapsed = _measurement_clock(device, measurement_timings) - started
                if measurement_timings is not None:
                    measurement_timings["current_forward_projection_seconds"] = measurement_timings.get("current_forward_projection_seconds", 0.) + elapsed
                terms: dict[str, list[torch.Tensor]] = defaultdict(list)
                note_terms: dict[str, torch.Tensor] = {}
                for group_index, (both, states) in zip(indices, values, strict=True):
                    if not bool(torch.isfinite(both).all()):
                        raise NonFiniteStop("v17: nonfinite current log probability")
                    for row in groups[group_index]:
                        kind, n = row["kind"], row["target_len"]
                        if kind in PG_KINDS:
                            value = both[1]
                            if row["pg"]:
                                mu = value.new_tensor(row["generation"]["sampling_logprobs"])
                                difference = (value.detach() - mu).abs()
                                rho = (value.detach() - mu).exp()
                                drift_sum += float(difference.sum())
                                drift_n += n
                                clipped += int((rho > 2).sum())
                                advantage = row["advantage"]
                                mask = row["loss_mask"].to(value.device).clone()
                                loop = kind == "RL" and row.get("reason") == "LOOP_CONFIRMED"
                                if kind == "RL" and advantage < 0 and not loop:
                                    negative_n += n
                                    mask &= rl.negative_token_mask(value)
                                    masked_n += n - int(mask.sum())
                                excluded = advantage < 0 and float(difference.mean()) > .3
                                excluded_n += int(excluded)
                                if is_v18:
                                    # ordinary T1 rows carry pg_mask_reason=None; a None key cannot be sealed beside
                                    # the string keys (v18 pod 2026-11-19 stopped after the first RL update).
                                    reason_key = row.get("pg_mask_reason") or row.get("reason") or kind
                                    detail = policy_mask_counts[reason_key]
                                    detail["candidate_tokens"] += int(row["loss_mask"].sum())
                                    retained_tokens = int(mask.sum()) if kind in selected and not excluded and advantage else 0
                                    detail["retained_tokens"] += retained_tokens
                                    detail["positive_tokens" if advantage > 0 else "negative_tokens"] += retained_tokens
                                if kind in selected and not excluded and advantage and bool(mask.any()):
                                    size = int(mask.sum()) if loop else n
                                    coefficient = (1. if kind == "RL" else .25) * advantage * min(1., 2048 / size) / DENOMINATOR
                                    terms[kind].append(-coefficient * (rho.clamp(max=2) * value * mask).sum())
                                    stats["task_signal"] = True
                                    tokens[kind] += int(mask.sum())
                                    loop_n += int(mask.sum()) if loop else 0
                        elif kind in selected:
                            mask = row["loss_mask"].to(both.device)
                            if is_v18:
                                term, verified_tokens = sft_term_v18(row, both[0], contract, denominator)
                                process_tokens[kind] += verified_tokens
                            else:
                                term = -SFT_WEIGHTS[kind] * both[0][mask].mean() / denominator
                            terms[kind].append(term)
                            if kind == "N":
                                note_terms[row["row_id"]] = term
                            tokens[kind] += int(mask.sum())
                            stats["task_signal"] = True
                        if "KL" in selected and (kind == "RL" or prep):
                            value, positions = _kl(bundle, row, states, contract, prep=prep, timings=measurement_timings)
                            terms["KL"].append((.5 if prep else .02) * value / denominator)
                            stats["f0_positions"][row["row_id"]] = {"positions": positions, "action_tokens": n}
                combined = {kind: torch.stack(parts).sum() for kind, parts in terms.items()}
                if combined:
                    if any(not bool(torch.isfinite(value)) for value in combined.values()):
                        raise NonFiniteStop("v17: nonfinite component loss")
                    started = _measurement_clock(device, measurement_timings)
                    diagnose_all = (contract.get("measure_component_gradients", False) if measure_component_gradients is None
                                    else measure_component_gradients)
                    diagnosed = {kind: value for kind, value in combined.items() if diagnose_all and kind != "N"}
                    diagnosed_channels.update(diagnosed)
                    _component_gradients(bundle, diagnosed, gradients)
                    for row_id, term in note_terms.items():
                        diagnosed_channels.add("N")
                        measured = _component_gradients(bundle, {"N": term}, gradients)["N"]
                        note_row_gradients[row_id] = measured
                        parent_id = next(row["problem_id"] for row in rows if row["row_id"] == row_id)
                        note_parent_gradients[parent_id] = measured
                    torch.stack(list(combined.values())).sum().backward()
                    elapsed = _measurement_clock(device, measurement_timings) - started
                    if measurement_timings is not None:
                        measurement_timings["backward_with_diagnostics_seconds"] = measurement_timings.get("backward_with_diagnostics_seconds", 0.) + elapsed
                    for kind, value in combined.items():
                        losses[kind] += float(value.detach())
            mean = drift_sum / drift_n if drift_n else 0.
            fraction = clipped / drift_n if drift_n else 0.
            reason = ("IS_MEAN" if mean > .5 else "IS_CLIP" if fraction > .10 else
                      "REPLAY" if not prep and contract["block_id"] == 0 and mean > .05 else None)
            if reason:
                bundle.model.zero_grad(set_to_none=True)
                stats.update(hold=True, HOLD=True, hold_reason=reason, task_signal=False)
            norms = _gradient_norms(bundle)
            stats.update(gradient_norms=norms, has_gradient=not reason and any(v is not None and v > 0 for v in norms.values()),
                         gradients_discarded=reason is not None,
                         importance={"mean_abs_log_ratio": mean, "clipped_fraction": fraction, "tokens": drift_n,
                                     "row_excluded": excluded_n, "mixed_fraction": 0.},
                         ntf={"negative_tokens": negative_n, "masked_tokens": masked_n,
                              "retained_tokens": negative_n - masked_n,
                              "masked_fraction": masked_n / negative_n if negative_n else 0.},
                         loop={"rows": sum(r.get("reason") == "LOOP_CONFIRMED" for r in encoded), "tokens": loop_n})
    except BaseException:
        bundle.model.zero_grad(set_to_none=True)
        raise
    component_norms = {kind: {group: math.sqrt(sum(float(value.square().sum())
                       for (channel, name, _), value in gradients.items() if channel == kind and name == group))
                       for group in bundle.params} for kind in sorted(diagnosed_channels)}
    if any(not math.isfinite(v) for norms in component_norms.values() for v in norms.values()):
        bundle.model.zero_grad(set_to_none=True)
        raise NonFiniteStop("v17: nonfinite component gradient")
    stats["component_gradient_norms"] = component_norms
    stats["component_gradient_channels_measured"] = sorted(diagnosed_channels)
    stats["note_gradient_norms"] = component_norms.get("N", {})
    retained = not stats["hold"] and any(v > 0 for v in stats["note_gradient_norms"].values())
    stats["note_parent_gradient_norms"] = note_parent_gradients
    stats["note_row_gradient_norms"] = note_row_gradients
    stats["n_parent_evidence"] = []
    for row in encoded:
        if row["kind"] != "N" or "N" not in selected:
            continue
        norms = note_row_gradients.get(row["row_id"], {})
        measured = bool(norms) and all(math.isfinite(value) for value in norms.values()) and any(value > 0 for value in norms.values())
        evidence = row.get("n_source_evidence", {})
        receipt = row.get("verification_receipt", {})
        stats["n_parent_evidence"].append({"problem_id": row["problem_id"], "row_id": row["row_id"],
            "supervised_tokens": int(row["loss_mask"].sum()), "gradient_finite_nonzero": measured,
            "note_gradient_retained": measured and not stats["hold"], "gradient_norms": norms,
            "source_evidence_sha256": evidence.get("sha256"), "target_sha256": hashlib.sha256(row["target"].encode()).hexdigest(),
            "generation_sha256": digest(row["generation"]), "receipt_sha256": receipt.get("sha256"),
            "byte_spans": row.get("verified_byte_spans", []), "token_mask_sha256": digest(row["loss_mask"].tolist()),
            "note_condition": receipt.get("note_condition"), "actual_hint_exposed": receipt.get("provided_location_count", 0) > 0,
            "claim_stages": [repair.get("stage") for repair in receipt.get("repairs", []) if repair.get("accepted")]})
    notes = [r for r in encoded if r["kind"] == "N" and "N" in selected
             and any(value > 0 for value in note_parent_gradients.get(r["problem_id"], {}).values())]
    stats.update(note_gradient_retained=retained, n_content_gradient_finite_nonzero=retained,
                 verified_repair_parent_ids=[r["problem_id"] for r in notes] if retained else [],
                 n_verified_content_tokens=sum(int(r["loss_mask"].sum()) for r in notes) if retained else 0,
                 consumed_row_ids=[] if stats["hold"] else [r["row_id"] for r in encoded],
                 loss=sum(losses.values()))
    stats["rl"].update(rows=counts["RL"], pg=losses["RL"], kl=losses["KL"], anchor=losses["KL"],
                       tokens=tokens["RL"], denominator=DENOMINATOR)
    stats["obs"].update(rows=counts["OBS"], pg=losses["OBS"], tokens=tokens["OBS"])
    stats.update(t1_negative_candidate_tokens=negative_n, t1_negative_retained_tokens=negative_n - masked_n,
                 t1_negative_retained_ratio=(negative_n - masked_n) / negative_n if negative_n else 0.,
                 loop_confirmed=stats["loop"]["rows"], loop_tokens_penalized=loop_n,
                 length_nonloop_excluded=sum(r["kind"] == "RL" and r.get("reason") == "LENGTH" for r in encoded))
    if is_v18:
        stats["policy_mask_counts"] = {key: dict(value) for key, value in policy_mask_counts.items()}
        stats["verified_process_tokens"] = dict(process_tokens)
        if note_gradient_capture is not None:
            note_gradient_capture.clear()
            if not stats["hold"]:
                note_gradient_capture.update({(group, index): value for (kind, group, index), value in gradients.items()
                                              if kind == "N"})
    stats["sft"] = {kind: {"rows": counts[kind], "tokens": tokens[kind], "weighted_loss": losses[kind],
                           "parent_denominator": denominator} for kind in SFT_KINDS if counts[kind]}
    return stats


MEMORY_PROBE_ACTIONS = 8192
MEMORY_PROBE_PROMPT = 4000


def measure_update_memory_v17(bundle: Any, encoded_row: dict, *, shape_bounds: dict,
                              max_target_tokens: int = MEMORY_PROBE_ACTIONS,
                              max_input_tokens: int = MEMORY_PROBE_PROMPT,
                              live_engine: bool = False) -> dict:
    """Measure the shared RL+A1+64-position KL graph on a bounded BF16 FULL capacity fixture."""
    from fine_tuning.notebook_rl.checkpoint import capture_rng, restore_rng
    from fine_tuning.notebook_rl.policy import assert_base_unchanged
    from .capacity import CapacityHold, memory
    from .train import _all_parameter_hashes
    from .v16b_runtime import kv_budget_v16b

    base = bundle.model.get_base_model()
    ids, n = encoded_row.get("input_ids"), encoded_row.get("target_len")
    pixels, grids = encoded_row.get("pixel_values"), encoded_row.get("image_grid_thw")
    if (base.get_input_embeddings().weight.dtype != torch.bfloat16
            or getattr(base, "is_loaded_in_4bit", False) or getattr(base, "is_loaded_in_8bit", False)
            or encoded_row.get("route") != ROUTE_FULL or not isinstance(pixels, torch.Tensor)
            or not isinstance(grids, torch.Tensor) or grids.ndim != 2 or grids.shape[1] != 3
            or grids.dtype != torch.long or not len(grids) or bool((grids < 1).any())
            or not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 1
            or type(n) is not int or not 0 < n < len(ids)
            or type(max_target_tokens) is not int or not MEMORY_PROBE_ACTIONS <= max_target_tokens <= 8192
            or type(max_input_tokens) is not int or max_input_tokens < MEMORY_PROBE_PROMPT):
        raise ValueError("v17 memory: BF16 FULL image specimen and production-length bounds required")
    if (any(type(shape_bounds.get(k)) is not int or shape_bounds[k] < 1
            for k in ("max_images", "max_vision_patches", "max_prompt_tokens"))
            or not isinstance(shape_bounds.get("dataset_sha256"), str)
            or len(shape_bounds["dataset_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in shape_bounds["dataset_sha256"])
            or len(grids) < shape_bounds["max_images"]
            or int(grids.prod(-1).sum()) < shape_bounds["max_vision_patches"]
            or int(grids.prod(-1).sum()) != len(pixels)):
        raise ValueError("v17 memory: specimen does not cover sealed image and patch bounds")
    prompt_n = max(max_input_tokens, shape_bounds["max_prompt_tokens"], len(ids) - n)
    if prompt_n + max_target_tokens > 24576:
        raise ValueError("v17 memory: full prompt/actions exceed unchanged context limit")
    banned = set(getattr(getattr(bundle.processor, "tokenizer", None), "all_special_ids", []))
    for config in (base.config, base.config.text_config):
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
                     "eos_token_id", "pad_token_id"):
            value = getattr(config, name, None)
            banned.update(value if isinstance(value, (list, tuple)) else [value])
    safe = next((int(value) for value in ids[-n:] if int(value) not in banned), None)
    if safe is None:
        raise ValueError("v17 memory: non-control fixture action required")
    fixture = encoded_row | {"problem_id": "v17-capacity-fixture", "kind": "RL", "target_len": max_target_tokens,
                             "input_ids": torch.cat([ids[:-n], ids.new_full((prompt_n - len(ids) + n + max_target_tokens,), safe)])}
    device = next(base.parameters()).device
    memory(bundle)
    assert_base_unchanged(bundle)
    before, rng = _all_parameter_hashes(bundle), capture_rng()
    parameters = list(bundle.model.parameters())
    incoming = [p.grad for p in parameters]
    reservation = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with _restore_state(bundle):
            bundle.model.zero_grad(set_to_none=True)
            bundle.route.apply(True)
            bundle.model.train()
            torch.use_deterministic_algorithms(True)
            reservation = [torch.empty_like(p) for group in bundle.params.values() for p in group for _ in range(2)]
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            baseline = memory(bundle)
            if not live_engine and baseline["reserved"] > 10.5 * 2**30:
                raise CapacityHold("v17 memory: HF residency exceeds the fixed 10.5 GiB assumption")
            torch.cuda.reset_peak_memory_stats(device)
            timings: dict[str, float] = {}
            started = _measurement_clock(device, timings)
            ((values, states),) = _row_values(bundle, [fixture])
            timings["current_forward_projection_seconds"] = _measurement_clock(device, timings) - started
            pg = -min(1., 2048 / max_target_tokens) * values[1].sum() / DENOMINATOR
            a1 = -.20 * values[0].mean() / 16
            kl, positions = _kl(bundle, fixture, states,
                                {"master_seed": 3407, "f0_position_salt": F0_POSITION_SALT},
                                prep=False, timings=timings, force_full=True)
            started = _measurement_clock(device, timings)
            (pg + a1 + .02 * kl / 16).backward()
            timings["backward_seconds"] = _measurement_clock(device, timings) - started
            norms = _gradient_norms(bundle)
            torch.cuda.synchronize(device)
            peak_reserved, peak_allocated = torch.cuda.max_memory_reserved(device), torch.cuda.max_memory_allocated(device)
            if not baseline["reserved"] <= peak_reserved <= baseline["total"]:
                raise CapacityHold("v17 memory: invalid measured allocator peak")
            extra = peak_reserved - baseline["reserved"]
            after = memory(bundle)
            if live_engine:
                # The engine already owns its KV pool; measure remaining physical headroom in that live process.
                if after["free"] < 2 * 2**30:
                    raise CapacityHold("v18b memory: live maximum-shape backward leaves less than 2 GiB headroom")
                kv = None
            else:
                try:
                    kv = kv_budget_v16b(extra / 2**30)
                except RuntimeError as error:
                    raise CapacityHold(f"v17 memory: X={extra / 2**30:.6f} GiB; {error}") from error
            report = {"version": "v17-update-memory-v1", "measurement": "actual_cuda_bf16_shared_backward_only",
                      "fixture_only": True, "optimizer_steps": 0, "rows": 1, "shared_t1_a1_pairs": 1,
                      "shape_bounds": dict(shape_bounds), "shape_bounds_sha256": digest(shape_bounds),
                      "prompt_tokens": prompt_n, "action_tokens": max_target_tokens,
                      "image_grids": grids.tolist(), "vision_patches": int(grids.prod(-1).sum()),
                      "f0_positions": positions, "timings": timings, "gradient_norms": norms,
                      "baseline": baseline, "peak_reserved_bytes": peak_reserved, "peak_allocated_bytes": peak_allocated,
                      "extra_reserved_bytes": extra, "extra_reserved_gib": extra / 2**30, "x_gib": extra / 2**30,
                      "resident_moment_reservation_bytes": sum(p.numel() * p.element_size() for p in reservation),
                      "kv_cache_gib": kv,
                      **({"live_engine": True, "post_backward_memory": after,
                          "minimum_free_bytes": 2 * 2**30} if live_engine else {}),
                      "scope": "one FULL capacity fixture; actual solution quality and packed shapes unmeasured"}
            return report | {"sha256": digest(report)}
    finally:
        bundle.model.zero_grad(set_to_none=True)
        reservation.clear()
        for parameter, gradient in zip(parameters, incoming, strict=True):
            parameter.grad = gradient
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        restore_rng(rng)
        assert_base_unchanged(bundle)
        if _all_parameter_hashes(bundle) != before:
            raise ValueError("v17 memory: parameter changed during backward-only measurement")


def calibrate_update_v17(bundle: Any, encoded: list[dict], contract: dict, *, token_budget: int = 12000,
                         capacity: Any = None) -> dict:
    """Warm up once, measure A/B/C three times, and restore all parameters, gradients, modes and RNG."""
    from fine_tuning.notebook_rl.checkpoint import capture_rng, restore_rng
    from .train import _all_parameter_hashes

    parameters = list(bundle.model.parameters())
    incoming = [p.grad for p in parameters]
    rng, before = capture_rng(), _all_parameter_hashes(bundle)
    device = next(bundle.model.parameters()).device
    cuda = device.type == "cuda"
    phases = {"A": {"RL", "OBS"}, "B": {"RL", "OBS", "A1"}, "C": {"RL", "OBS", "A1", "KL"}}
    report: dict[str, Any] = {"version": "v17-update-calibration-v1", "optimizer_steps": 0,
                              "device": str(device), "measurement": "cuda-bf16" if cuda else "cpu-functional",
                              "contract_sha256": digest(contract), "shapes": [], "measurements": {}}
    for row in encoded:
        grids = row.get("image_grid_thw")
        report["shapes"].append({"row_id": row["row_id"], "route": row["route"],
                                 "prompt_tokens": len(row["input_ids"]) - row["target_len"],
                                 "action_tokens": row["target_len"], "images": len(grids) if grids is not None else 0,
                                 "vision_patches": int(grids.prod(-1).sum()) if grids is not None else 0})
    try:
        with _restore_state(bundle):
            for name, components in phases.items():
                measurements = []
                for repeat in range(4):
                    restore_rng(rng)
                    bundle.model.zero_grad(set_to_none=True)
                    if cuda:
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    timings: dict[str, float] = {}
                    result = backward_v17_block(bundle, encoded, contract, token_budget,
                                                components=components, capacity=capacity,
                                                measurement_timings=timings, measure_component_gradients=True)
                    if cuda:
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    if result["hold"]:
                        raise ValueError("v17 calibration: HOLD is not a valid timing measurement")
                    if repeat:
                        measurements.append({"seconds": elapsed, "timings": timings,
                                             "component_gradient_norms": result["component_gradient_norms"],
                                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
                                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
                                             "gradient_norms": result["gradient_norms"]})
                times = [value["seconds"] for value in measurements]
                report["measurements"][name] = {"samples": measurements, "median_seconds": statistics.median(times),
                                                 "min_seconds": min(times), "max_seconds": max(times)}
        values = report["measurements"]
        report.update(delta_a1_seconds=values["B"]["median_seconds"] - values["A"]["median_seconds"],
                      delta_f0_seconds=values["C"]["median_seconds"] - values["B"]["median_seconds"])
        return report | {"sha256": digest(report)}
    finally:
        bundle.model.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, incoming, strict=True):
            parameter.grad = gradient
        restore_rng(rng)
        if _all_parameter_hashes(bundle) != before:
            raise ValueError("v17 calibration: parameter changed without an optimizer step")
