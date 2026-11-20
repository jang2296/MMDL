"""Version-4 RL term on stored rollouts: notebook_rl.losses.group_loss with the current policy as the old policy.

Each rollout is used in exactly one optimizer step (one pass over the round), so old = current.detach(): the ratio is 1
and the gradient is advantage * grad log pi (the clip range never binds). The k3 KL term is against the adapter-disabled
F0 reference (policy.reference_logprobs' context). The normalization is the group's total action tokens, so a long
sample is not weighted more per token than a short one, and the loss decomposes over micro-batches of one group.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.notebook_rl import losses


def processed_scores(scores: torch.Tensor, generated_ids: torch.Tensor, *, temperature: float = .7,
                     top_p: float = .8, top_k: int = 20, presence_penalty: float = 1.5) -> torch.Tensor:
    """The sampled distribution: generated-only presence, temperature, then the pinned HF top-k/top-p warpers."""
    from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

    if (scores.ndim != 2 or not scores.is_floating_point() or not bool(torch.isfinite(scores).all())
            or generated_ids.ndim != 2 or generated_ids.shape[0] != scores.shape[0]
            or generated_ids.dtype != torch.long or generated_ids.device != scores.device
            or (generated_ids.numel() and (int(generated_ids.min()) < 0 or int(generated_ids.max()) >= scores.shape[1]))):
        raise ValueError("single RL: finite [B,V] scores and generated-only int64 [B,T] IDs required")
    if (not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(top_p) or not 0 < top_p <= 1
            or type(top_k) is not int or top_k < 0 or not math.isfinite(presence_penalty)):
        raise ValueError("single RL: invalid sampling settings")
    seen = torch.zeros_like(scores, dtype=torch.bool)
    if generated_ids.numel():
        seen.scatter_(1, generated_ids, True)
    result = (scores.float() - seen * presence_penalty) / temperature
    if top_k:
        result = TopKLogitsWarper(top_k)(generated_ids, result)
    if top_p < 1:
        result = TopPLogitsWarper(top_p)(generated_ids, result)
    return result


def support_ids(scores: torch.Tensor) -> list[list[int]]:
    """The ordered supported vocabulary IDs per processed-score row."""
    if scores.ndim != 2 or bool(torch.isnan(scores).any()) or bool(torch.isposinf(scores).any()):
        raise ValueError("single RL: invalid processed scores")
    return [torch.isfinite(row).nonzero().flatten().tolist() for row in scores]


def support_sha256(scores: torch.Tensor) -> list[str]:
    """One digest of the ordered supported vocabulary IDs per processed-score row."""
    return [digest(ids) for ids in support_ids(scores)]


def stored_support_logprob(scores: torch.Tensor, generated_ids: torch.Tensor, action: int, support: list[int], *,
                           temperature: float = .7, presence_penalty: float = 1.5, **_: Any) -> torch.Tensor:
    """log q(action) on the generation-time top-k/top-p support, renormalized with replayed scores.

    Re-deriving the support from replayed logits can drop a boundary token that was actually sampled
    (cached decode and full forward round differently), giving -inf. The stored support is the set the
    sampler really drew from; inside it this equals processed_scores(...).log_softmax(-1)[action].
    """
    if action not in support or len(set(support)) != len(support):
        raise ValueError("single RL: sampled action outside its stored generation support")
    seen = torch.zeros_like(scores, dtype=torch.bool)
    if generated_ids.numel():
        seen.scatter_(1, generated_ids, True)
    values = ((scores.float() - seen * presence_penalty) / temperature)[0]
    index = torch.tensor(support, dtype=torch.long, device=values.device)
    return values[action] - values.index_select(0, index).logsumexp(0)


def sampling_sha256(sampling: dict) -> str:
    if set(sampling) != {"temperature", "top_p", "top_k", "presence_penalty"}:
        raise ValueError("single RL: sampling transform must be explicit")
    processed_scores(torch.zeros(1, 1), torch.empty(1, 0, dtype=torch.long), **sampling)
    return digest(sampling)


def policy_sha256(bundle) -> str:
    """Hash every installed L/M/V tensor, including Vision tensors currently frozen by routing."""
    from fine_tuning.modules import tensor_sha256

    owned = {id(parameter) for group in bundle.params.values() for parameter in group}
    values = {name: tensor_sha256(parameter) for name, parameter in bundle.model.named_parameters()
              if id(parameter) in owned}
    if len(values) != len(owned):
        raise ValueError("single RL: incomplete policy tensor inventory")
    return digest(values)


def single_pg(logprobs: torch.Tensor, reward: int, *, problem_count: int, token_scale: float) -> torch.Tensor:
    """b=0 REINFORCE (reward -1, 0 or 1); wrong answers remain in the fixed B*C denominator."""
    if (logprobs.ndim != 1 or not logprobs.numel() or not bool(torch.isfinite(logprobs).all())
            or type(reward) is not int or reward not in (-1, 0, 1) or type(problem_count) is not int or problem_count < 1
            or not math.isfinite(token_scale) or token_scale <= 0):
        raise ValueError("single RL: invalid reward, action log probabilities or fixed denominator")
    return -reward * logprobs.sum() / (problem_count * token_scale)


def pad(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D per-row log probabilities -> [B, T] right-padded with zeros and the binary action mask."""
    if not values or any(v.ndim != 1 or v.numel() == 0 for v in values):
        raise ValueError("rl: nonempty 1-D log probabilities required")
    width = max(v.numel() for v in values)
    mask = torch.zeros((len(values), width), dtype=torch.long, device=values[0].device)
    for index, value in enumerate(values):
        mask[index, :value.numel()] = 1
    return torch.stack([torch.nn.functional.pad(v, (0, width - v.numel())) for v in values]), mask


def group_term(current: list[torch.Tensor], reference: list[torch.Tensor] | None, advantages: list[float],
               *, group_tokens: int, eps_low: float, eps_high: float, beta: float) -> torch.Tensor:
    """The loss of one micro-batch of one rollout group, already divided by the whole group's action tokens."""
    if len(current) != len(advantages) or (reference is not None and len(reference) != len(current)):
        raise ValueError("rl: one advantage (and reference) per row required")
    if type(group_tokens) is not int or group_tokens < sum(v.numel() for v in current):
        raise ValueError("rl: group_tokens must cover every row of the micro-batch")
    if beta and reference is None:
        raise ValueError("rl: a reference is required when beta > 0")
    cur, mask = pad(current)
    ref = pad(reference)[0].to(cur.device) if reference is not None else cur.detach()
    adv = torch.tensor(advantages, dtype=torch.float32, device=cur.device)
    keep = torch.ones(len(current), dtype=torch.bool, device=cur.device)
    return losses.group_loss(cur, cur.detach(), ref, adv, mask, keep, eps_low=eps_low, eps_high=eps_high, beta=beta,
                             norm_const=float(group_tokens))


def processed_full_logprobs(logits: torch.Tensor, actions: torch.Tensor, offset: int = 0, *,
                            temperature: float = .7, presence_penalty: float = 1.5) -> torch.Tensor:
    """Full-vocabulary sampled log q; presence sees generated actions before each absolute position."""
    if (logits.ndim != 2 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
            or actions.ndim != 1 or actions.dtype != torch.long or actions.device != logits.device
            or type(offset) is not int or offset < 0 or offset + len(logits) > len(actions)
            or not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(presence_penalty)
            or actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= logits.shape[1])):
        raise ValueError("v16b: finite [T,V] logits, complete int64 actions and a valid chunk offset required")
    # First occurrence is enough: presence is binary, including occurrences in earlier projection chunks.
    first = torch.full((logits.shape[1],), len(actions), dtype=torch.long, device=actions.device)
    positions = torch.arange(len(actions), device=actions.device)
    first.scatter_reduce_(0, actions, positions, reduce="amin", include_self=True)
    seen = first[None, :] < positions[offset:offset + len(logits), None]
    scores = (logits.float() - presence_penalty * seen) / temperature
    return scores.log_softmax(-1).gather(1, actions[offset:offset + len(logits), None])[:, 0]


def negative_token_mask(logprobs: torch.Tensor, tau: float = .5) -> torch.Tensor:
    """Keep negative-advantage tokens whose current probability is at most tau."""
    if (not logprobs.is_floating_point() or not bool(torch.isfinite(logprobs).all())
            or bool((logprobs > 0).any()) or not math.isfinite(tau) or not 0 < tau <= 1):
        raise ValueError("v16b: finite log probabilities and tau in (0,1] required")
    return logprobs.detach() <= math.log(tau)
