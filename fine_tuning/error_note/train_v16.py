"""v16 content SFT and diagnostics; callers own admission, optimizer, and checkpointing."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterable

import torch

from fine_tuning.notebook_rl.input_preparation import digest, prepare_protocol_inputs
from fine_tuning.notebook_rl.prompts import user_message

from . import rl
from .contract import ROUTE_FULL, ROUTE_LM
from .llm_lora import scaled_adapters
from .train import END_TOKEN, NonFiniteStop, _gradient_norms, batch_logprobs, plan_microbatches

SELF_KINDS = ("A1", "C", "N")
KINDS = ("O", "J", *SELF_KINDS)


def token_byte_mask(tokenizer: Any, actions: list[int], text: str, spans: list[list[int]], *,
                    end_token_id: int, supervise_end: bool = False) -> list[bool]:
    """Select only tokens wholly inside verified UTF-8 spans, without retokenizing generated text."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    raw = text.encode("utf-8")
    if (not spans or any(not isinstance(span, (list, tuple)) or len(span) != 2
                         or any(type(value) is not int for value in span)
                         or not 0 <= span[0] < span[1] <= len(raw) for span in spans)
            or any(left[1] > right[0] for left, right in zip(spans, spans[1:]))):
        raise ValueError("v16 train: invalid verified byte spans")
    decoder = {value: key for key, value in bytes_to_unicode().items()}
    offset, chunks, mask = 0, [], []
    for index, (action, token) in enumerate(zip(actions, tokenizer.convert_ids_to_tokens(actions), strict=True)):
        if action == end_token_id:
            if index != len(actions) - 1:
                raise ValueError("v16 train: nonterminal end token")
            mask.append(supervise_end)
            continue
        if token in tokenizer.all_special_tokens:
            raise ValueError("v16 train: generated control token")
        try:
            chunk = bytes(decoder[character] for character in token)
        except (KeyError, TypeError) as error:
            raise ValueError("v16 train: tokenizer byte history unavailable; HOLD") from error
        stop = offset + len(chunk)
        mask.append(bool(chunk) and any(start <= offset and stop <= end for start, end in spans))
        chunks.append(chunk)
        offset = stop
    if b"".join(chunks) != raw:
        raise ValueError("v16 train: token bytes differ from original target; HOLD")
    if not any(mask):
        raise ValueError("v16 train: no wholly verified token")
    return mask


def encode_content_row(processor: Any, model_config: Any, row: dict, images: list, *, image: dict,
                       max_target_tokens: int = 4096, max_context_tokens: int = 24576) -> dict:
    """Encode authored O/J using raw CE, and native A1/C/N using their unchanged sampled actions."""
    from .content_v16 import GRAPH_NEIGHBOR_INSTRUCTION, O_INSTRUCTION, ROW_SCHEMA
    from .audit import text_sha256

    tokenizer = processor.tokenizer
    kind, prompt, target = row.get("kind"), row.get("prompt"), row.get("target")
    if (row.get("schema") != ROW_SCHEMA or kind not in KINDS or row.get("route") not in (ROUTE_FULL, ROUTE_LM)
            or not isinstance(prompt, str) or not prompt.strip() or not isinstance(target, str) or not target.strip()
            or any(token in text for token in tokenizer.all_special_tokens for text in (prompt, target))
            or type(max_target_tokens) is not int or max_target_tokens < 1
            or type(max_context_tokens) is not int or max_context_tokens < 2):
        raise ValueError("v16 train: invalid content row, control token, or token limits")
    if kind == "O":
        instruction = O_INSTRUCTION.get(row.get("layer", ""))
        graph_prompt = (isinstance(instruction, str) and row.get("layer") == "graph"
                        and re.fullmatch(re.escape(GRAPH_NEIGHBOR_INSTRUCTION) + r"\nRequested node: \d+\.", prompt))
        if (not images or row["route"] != ROUTE_FULL or "generation" in row
                or not instruction or prompt != instruction and not graph_prompt
                or row.get("target_origin") not in ("program", "human")
                or row.get("verification_receipt", {}).get("training_prompt_sha256") != text_sha256(prompt)
                or row.get("verification_receipt", {}).get("target_sha256") != text_sha256(target)):
            raise ValueError("v16 train: O requires an authored observation instruction and FULL image route")
    elif kind == "J" and ("generation" in row or row.get("target_origin") != "human"):
        raise ValueError("v16 train: J requires a dataset-authored human solution")
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(prompt, len(images))], images, image)
    inputs = prepared["inputs"]
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if type(end_id) is not int or end_id == tokenizer.unk_token_id:
        raise ValueError("v16 train: end token is unavailable")
    if kind in ("O", "J"):
        actions = tokenizer.encode(target, add_special_tokens=False)
        if not actions:
            raise ValueError("v16 train: empty target")
        actions.append(end_id)
        mask = [True] * len(actions)
        mask[-1] = kind == "O"
    else:
        generation = row.get("generation", {})
        actions = generation.get("completion_token_ids")
        if (not isinstance(actions, list) or not actions or any(type(value) is not int or value < 0 for value in actions)
                or generation.get("text") != target
                or generation.get("completion_tokens", len(actions)) != len(actions)
                or generation.get("finish_reason") not in ("stop", "eos")):
            raise ValueError("v16 train: self target must preserve a complete original generation")
        mask = token_byte_mask(tokenizer, actions, target, row.get("verified_byte_spans", []), end_token_id=end_id)
        if row.get("loss_mask") != mask:
            raise ValueError("v16 train: self mask differs from verified token bytes")
        if kind == "A1":
            from .contract import answer_denial
            from .v4 import target_problem

            reason = target_problem(target, actions, generation["finish_reason"], row.get("status", ""), min_tokens=256)
            if row.get("target_source") != "T1" or reason or answer_denial(target):
                raise ValueError(f"v16 train: A1 requires an admitted correct long first solution ({reason})")
            if (generation.get("prompt_token_ids") != prepared["prompt_token_ids"]
                    or generation.get("input_sha256") != prepared["input_sha256"]):
                raise ValueError("v16 train: A1 original P0 IDs or image/input binding changed")
    if len(actions) > min(max_target_tokens, 1024 if kind == "N" else max_target_tokens):
        raise ValueError("TARGET_TOO_LONG")
    if inputs["input_ids"].numel() + len(actions) > max_context_tokens:
        raise ValueError("CONTEXT_BLOCKED")
    if any(getattr(model_config, name, None) in actions for name in ("image_token_id", "video_token_id")):
        raise ValueError("v16 train: target contains an image/video placeholder")
    return row | {"input_ids": torch.cat([inputs["input_ids"][0], torch.tensor(actions, dtype=torch.long)]),
                  "target_len": len(actions), "loss_mask": torch.tensor(mask, dtype=torch.bool), "has_image": bool(images),
                  "pixel_values": inputs.get("pixel_values"), "image_grid_thw": inputs.get("image_grid_thw"),
                  "input_sha256": prepared["input_sha256"]}


def block_coefficients(rows: list[dict], alpha: dict[str, float],
                       weights: dict[str, float] | None = None) -> list[float]:
    """Fix source/kind parent denominators over the closed block, never over a microbatch."""
    weights = {kind: 1.0 for kind in KINDS} if weights is None else weights
    if (any(not math.isfinite(value) or value < 0 for value in (*alpha.values(), *weights.values()))
            or any(row.get("layer") not in alpha or row.get("kind") not in weights for row in rows)):
        raise ValueError("v16 train: explicit finite nonnegative source and kind weights required")
    parents: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen = set()
    for row in rows:
        parent = row.get("parent_id")
        key = (parent, row["kind"])
        if not isinstance(parent, str) or not parent or key in seen:
            raise ValueError("v16 train: one row per parent and kind")
        seen.add(key)
        parents[row["layer"], row["kind"]].add(parent)
    return [alpha[row["layer"]] * weights[row["kind"]] / len(parents[row["layer"], row["kind"]]) for row in rows]


@contextmanager
def _restore_state(bundle: Any):
    modules = [(module, module.training) for module in bundle.model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in bundle.model.parameters()]
    initial_route = bundle.route.vision_on
    try:
        yield
    finally:
        bundle.route.apply(initial_route)
        for parameter, value in parameters:
            parameter.requires_grad_(value)
        for module, value in modules:
            module.training = value


def _validate_encoded(row: dict) -> torch.Tensor:
    ids, count = row.get("input_ids"), row.get("target_len")
    mask = row.get("loss_mask")
    if (row.get("kind") not in KINDS or row.get("route") not in (ROUTE_FULL, ROUTE_LM)
            or not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long
            or type(count) is not int or not 0 < count < ids.numel()
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (count,)
            or not bool(mask.any())):
        raise ValueError("v16 train: mask/target mismatch or no supervised token")
    if row["kind"] == "O" and (row["route"] != ROUTE_FULL or row.get("pixel_values") is None):
        raise ValueError("v16 train: O requires FULL image route")
    return mask


def masked_backward(bundle: Any, encoded: list[dict], plan: list[list[int]], coefficients: list[float], *,
                    sampling: dict | None = None) -> dict:
    """Accumulate exact masked row-mean losses; never construct an optimizer or change a parameter."""
    from .train_v4 import _require_single_zero_dropout, single_batch_statistics

    if (len(coefficients) != len(encoded) or any(not math.isfinite(value) or value < 0 for value in coefficients)
            or sorted(index for indices in plan for index in indices) != list(range(len(encoded)))
            or any(not indices or len({encoded[i]["route"] for i in indices}) != 1 for indices in plan)):
        raise ValueError("v16 train: plan must cover the block once with one route per microbatch")
    for row in encoded:
        _validate_encoded(row)
        if row["kind"] in SELF_KINDS and sampling is None:
            raise ValueError("v16 train: explicit self-generation sampling required")
    _require_single_zero_dropout(bundle)
    losses_by_kind: dict[str, float] = defaultdict(float)
    weighted_by_kind: dict[str, float] = defaultdict(float)
    with _restore_state(bundle):
        bundle.model.train()
        for indices in plan:
            active = [i for i in indices if coefficients[i] > 0]
            if not active:
                continue
            # Self rows and authored rows use different probability definitions, with the same closed-block factors.
            for self_generated in (False, True):
                selected = [i for i in active if (encoded[i]["kind"] in SELF_KINDS) == self_generated]
                if not selected:
                    continue
                rows = [encoded[index] for index in selected]
                bundle.route.apply(rows[0]["route"] == ROUTE_FULL)
                if self_generated:
                    assert sampling is not None  # validated for every self row before backward
                    values = [value["logprobs"] for value in single_batch_statistics(bundle, rows, sampling)]
                else:
                    values = batch_logprobs(bundle, rows)
                terms = []
                for index, row, value in zip(selected, rows, values, strict=True):
                    mask = row["loss_mask"].to(value.device)
                    if mask.shape != value.shape or not bool(torch.isfinite(value[mask]).all()):
                        raise NonFiniteStop("v16 train: nonfinite or mismatched supervised log probability")
                    loss = -value[mask].mean()
                    term = loss * coefficients[index]
                    terms.append(term)
                    losses_by_kind[row["kind"]] += float(loss.detach())
                    weighted_by_kind[row["kind"]] += float(term.detach())
                torch.stack(terms).sum().backward()
        norms = _gradient_norms(bundle)
    return {"loss_by_kind": dict(losses_by_kind), "weighted_by_kind": dict(weighted_by_kind),
            "weighted_loss": sum(weighted_by_kind.values()), "loss": sum(weighted_by_kind.values()),
            "gradient_norms": norms, "task_signal": any(value > 0 for value in coefficients),
            "has_gradient": any(value is not None and value > 0 for value in norms.values()), "microbatches": plan}


def backward_content_block(bundle: Any, encoded: list[dict], contract: dict, token_budget: int = 0, *,
                           components: Iterable[str] | None = None, capacity: Any = None) -> dict:
    """Contract entry point for single/campaign; all source denominators include the complete admitted block."""
    selected = set(KINDS if components is None else components)
    if selected - set(KINDS) or type(token_budget) is not int or token_budget < 0 or contract.get("pg_weight", 0) != 0:
        raise ValueError("v16 train: content-only components, zero PG, and nonnegative token budget required")
    if (not isinstance(contract.get("source_weights"), dict) or not contract["source_weights"]
            or not isinstance(contract.get("weights"), dict) or not contract["weights"]
            or set(contract["weights"]) - set(KINDS)):
        raise ValueError("v16 train: explicit source and kind weights required")
    coefficients = block_coefficients(encoded, contract["source_weights"], contract["weights"])
    for row in encoded:
        _validate_encoded(row)
    accepted = {row["problem_id"] for row in encoded}
    excluded = contract.get("excluded_problem_ids", {})
    expected = contract.get("expected_problem_ids")
    if (not isinstance(excluded, dict)
            or any(not isinstance(pid, str) or not pid or not isinstance(reason, str) or not reason.strip()
                   for pid, reason in excluded.items())
            or accepted.intersection(excluded)
            or excluded and expected is None
            or expected is not None and (len(expected) != len(set(expected)) or set(expected) != accepted.union(excluded))):
        raise ValueError("v16 train: closed block problem IDs or declared exclusions changed")
    if not encoded:
        return {"loss_by_kind": {}, "weighted_by_kind": {}, "weighted_loss": 0.0, "loss": 0.0,
                "gradient_norms": {group: None for group in bundle.params}, "task_signal": False,
                "has_gradient": False, "microbatches": []}
    sampling = contract.get("sampling")
    policy_hash = contract.get("policy_sha256")
    has_self = any(row["kind"] in SELF_KINDS for row in encoded)
    current_policy = rl.policy_sha256(bundle) if has_self else None
    vocabulary = bundle.model.get_base_model().config.text_config.vocab_size if has_self else 0
    for row in encoded:
        if row["kind"] not in SELF_KINDS:
            continue
        generation = row["generation"]
        actions = row["input_ids"][-row["target_len"]:].tolist()
        supports = generation.get("support_token_ids", [])
        if (sampling is None or generation.get("sampling_sha256") != rl.sampling_sha256(sampling)
                or generation.get("completion_token_ids") != actions or len(supports) != len(actions)
                or generation.get("support_sha256") != [digest(support) for support in supports]
                or any(not isinstance(support, list) or not support
                       or any(type(token) is not int or not 0 <= token < vocabulary for token in support)
                       or action not in support or len(support) != len(set(support))
                       for action, support in zip(actions, supports, strict=True))
                or policy_hash is None or generation.get("policy_sha256") != policy_hash
                or policy_hash != current_policy):
            raise ValueError("v16 train: self-generation support, sampling, or policy binding changed")
    coefficients = [value if row["kind"] in selected else 0.0 for row, value in zip(encoded, coefficients, strict=True)]
    plan = []
    for kind in KINDS:
        indices = [index for index, row in enumerate(encoded) if row["kind"] == kind]
        local = plan_microbatches([encoded[index]["input_ids"].numel() for index in indices],
                                  [encoded[index]["route"] for index in indices], token_budget)
        plan.extend([[indices[index] for index in batch] for batch in local])
    if capacity is not None:
        from .capacity import encoded_shapes

        bounded = []
        for indices in plan:
            phase = "sft_full" if encoded[indices[0]]["route"] == ROUTE_FULL else "sft_lm"
            bounded.extend(capacity.partition(encoded_shapes(encoded), [indices], phase))
        plan = bounded
        for indices in plan:
            phase = "sft_full" if encoded[indices[0]]["route"] == ROUTE_FULL else "sft_lm"
            capacity.check(encoded_shapes([encoded[index] for index in indices]), phase)
    return masked_backward(bundle, encoded, plan, coefficients, sampling=sampling)


def bf16_survival(masters: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compare BF16 forward values, separating genuine updates from base FP32-to-BF16 rounding noise."""
    if not masters or set(masters) != set(base):
        raise ValueError("v16 train: nonempty matching BF16 diagnostic tensors required")
    total = changed = 0
    dot = true_sq = seen_sq = 0.0
    for name, master in masters.items():
        origin = base[name]
        if master.shape != origin.shape or not master.numel() or not all(bool(torch.isfinite(tensor).all())
                                                                       for tensor in (master, origin)):
            raise ValueError("v16 train: finite matching nonempty BF16 diagnostic tensors required")
        current, original = master.detach().cpu().double(), origin.detach().cpu().double()
        delta = current - original
        seen = current.to(torch.bfloat16).double() - original.to(torch.bfloat16).double()
        if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(seen).all()):
            raise ValueError("v16 train: BF16 diagnostic cast overflow")
        total += delta.numel()
        changed += int((seen != 0).sum())
        dot += float((delta * seen).sum())
        true_sq += float(delta.square().sum())
        seen_sq += float(seen.square().sum())
    return {"changed_fraction": changed / total, "cosine": dot / math.sqrt(true_sq * seen_sq) if true_sq and seen_sq else 0.0,
            "projection": dot / true_sq if true_sq else 0.0, "true_norm": math.sqrt(true_sq), "visible_norm": math.sqrt(seen_sq)}


@contextmanager
def mergers_original(bundle: Any, roots: list[str]):
    """Temporarily select original Merger modules, retaining hooks and every incoming trainable flag."""
    if not roots or len(roots) != len(set(roots)):
        raise ValueError("v16 train: nonempty distinct Merger roots required")
    wrappers = [bundle.model.get_base_model().get_submodule(root) for root in roots]
    if any(not hasattr(wrapper, "original_module") or not hasattr(wrapper, "enable_adapters") for wrapper in wrappers):
        raise ValueError("v16 train: expected preserved original Merger wrappers")
    states = [(wrapper, wrapper.disable_adapters) for wrapper in wrappers]
    parameters = [(parameter, parameter.requires_grad) for wrapper in wrappers for parameter in wrapper.parameters()]
    try:
        for wrapper in wrappers:
            wrapper.enable_adapters(False)
        yield
    finally:
        for wrapper, disabled in reversed(states):
            wrapper.enable_adapters(not disabled)
        for parameter, trainable in parameters:
            parameter.requires_grad_(trainable)


@torch.no_grad()
def gold_logp(bundle: Any, encoded: list[dict]) -> list[float]:
    """Raw mean log p of authored gold observations, one diagnostic value per held-out parent."""
    for row in encoded:
        _validate_encoded(row)
        if row["kind"] != "O":
            raise ValueError("v16 train: visual diagnostic requires authored observations")
    result = []
    for row in encoded:
        value = batch_logprobs(bundle, [row])[0]
        result.append(float(value[row["loss_mask"].to(value.device)].mean()))
    return result


@torch.no_grad()
def visual_contribution(bundle: Any, encoded: list[dict], merger_roots: list[str]) -> dict[str, list[float]]:
    """Four teacher-forced ablations; a diagnostic only, never generated answer selection."""
    if not encoded:
        raise ValueError("v16 train: visual diagnostic requires held-out observations")
    with _restore_state(bundle):
        bundle.model.eval()
        bundle.route.apply(True)
        out = {"on": gold_logp(bundle, encoded)}
        with scaled_adapters(bundle, V=0.0):
            out["vision_off"] = gold_logp(bundle, encoded)
        with scaled_adapters(bundle, L=0.0):
            out["language_off"] = gold_logp(bundle, encoded)
        with mergers_original(bundle, merger_roots):
            out["merger_f0"] = gold_logp(bundle, encoded)
    return out


def main(argv: list[str] | None = None) -> int:
    """Keep the plan entry point: python -m fine_tuning.error_note.train_v16 {smoke,l0,...}."""
    from .run_v16 import main as run_main

    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
