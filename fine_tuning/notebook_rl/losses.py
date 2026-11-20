"""Pure Torch v2 group-relative token losses; no model or gold-label forward."""
from __future__ import annotations

import math
from numbers import Real

import torch


def _finite_tensor(value: torch.Tensor, name: str, *, floating: bool = True) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name}: expected tensor")
    if floating and not value.is_floating_point():
        raise ValueError(f"{name}: expected floating tensor")
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name}: expected finite real values")


def _scalar(value: float, name: str, *, minimum: float = 0.0, positive: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or value < minimum or (positive and value == minimum)):
        raise ValueError(f"{name}: invalid finite scalar")


def _matching(left: torch.Tensor, right: torch.Tensor, name: str) -> None:
    _finite_tensor(right, name)
    if right.shape != left.shape or right.device != left.device:
        raise ValueError(f"{name}: shape/device mismatch")


def _actions(action_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    _finite_tensor(action_mask, "action_mask", floating=False)
    if action_mask.shape != like.shape or action_mask.device != like.device:
        raise ValueError("action_mask: shape/device mismatch")
    if not bool(((action_mask == 0) | (action_mask == 1)).all()):
        raise ValueError("action_mask: expected binary values")
    mask = action_mask.detach().bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("action_mask: each sample needs an action token")
    return mask


def _matrix(value: torch.Tensor, name: str) -> None:
    _finite_tensor(value, name)
    if value.ndim != 2 or any(n == 0 for n in value.shape):
        raise ValueError(f"{name}: expected nonempty [B, T]")


def _work(value: torch.Tensor) -> torch.Tensor:
    # Avoid reduced-precision exp/cancellation while preserving FP64 numeric checks.
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Center a complete group's rewards, with no standard-deviation scaling."""
    _finite_tensor(rewards, "rewards")
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards: expected [n] with n >= 2")
    rewards = _work(rewards.detach())
    result = rewards - rewards.mean()
    _finite_tensor(result, "advantages")
    return result


def mis_mask(old_lp: torch.Tensor, samp_lp: torch.Tensor, action_mask: torch.Tensor,
             delta_seq: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep sequences whose action-token mean log ratio is within delta_seq."""
    _matrix(old_lp, "old_lp")
    _matching(old_lp, samp_lp, "samp_lp")
    _scalar(delta_seq, "delta_seq")
    mask = _actions(action_mask, old_lp)
    delta = _work(old_lp.detach()) - _work(samp_lp.detach())
    mean_delta = torch.where(mask, delta, 0.0).sum(dim=1) / mask.sum(dim=1)
    _finite_tensor(mean_delta, "mean log ratio")
    return mean_delta.abs() <= delta_seq, mean_delta


def k3(ref_lp: torch.Tensor, cur_lp: torch.Tensor) -> torch.Tensor:
    """Elementwise k3 KL estimate; only the current policy receives gradients."""
    _finite_tensor(cur_lp, "cur_lp")
    _matching(cur_lp, ref_lp, "ref_lp")
    delta = _work(ref_lp.detach()) - _work(cur_lp)
    result = (torch.expm1(delta) - delta).clamp_min(0.0)
    _finite_tensor(result, "k3")
    return result


def group_loss(cur_lp: torch.Tensor, old_lp: torch.Tensor, ref_lp: torch.Tensor,
               adv: torch.Tensor, action_mask: torch.Tensor, seq_keep: torch.Tensor, *,
               eps_low: float, eps_high: float, beta: float,
               norm_const: float, partial_adv: torch.Tensor | None = None) -> torch.Tensor:
    """Return a group (or sample-slice) sum divided by the full n_g * L_cap."""
    _matrix(cur_lp, "cur_lp")
    _matching(cur_lp, old_lp, "old_lp")
    _matching(cur_lp, ref_lp, "ref_lp")
    _finite_tensor(adv, "adv")
    if adv.shape != (cur_lp.shape[0],) or adv.device != cur_lp.device:
        raise ValueError("adv: expected [B] on the logprob device")
    _finite_tensor(seq_keep, "seq_keep", floating=False)
    if (seq_keep.dtype != torch.bool or seq_keep.shape != adv.shape
            or seq_keep.device != cur_lp.device):
        raise ValueError("seq_keep: expected bool [B] on the logprob device")
    for name, value in (("eps_low", eps_low), ("eps_high", eps_high), ("beta", beta)):
        _scalar(value, name)
    if eps_low >= 1.0:
        raise ValueError("eps_low: must be less than 1")
    _scalar(norm_const, "norm_const", positive=True)
    if partial_adv is not None:
        if not isinstance(partial_adv, torch.Tensor) or partial_adv.shape != cur_lp.shape or partial_adv.device != cur_lp.device:
            raise ValueError("partial_adv: shape/device mismatch")
        _matching(cur_lp, partial_adv, "partial_adv")
    active = _actions(action_mask, cur_lp) & seq_keep.detach().unsqueeze(1)
    cur = _work(cur_lp)
    # Mask before exp: excluded padding/sequences cannot overflow or contribute gradients.
    log_ratio = torch.where(active, cur - _work(old_lp.detach()), 0.0)
    ratio = log_ratio.exp()
    _finite_tensor(ratio, "policy ratio")
    advantage = _work(adv.detach()).unsqueeze(1)
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * advantage)
    token_loss = -surrogate
    if partial_adv is not None:
        # Clip the two objectives separately; opposing advantages must not cancel first.
        partial = _work(partial_adv.detach())
        token_loss = token_loss - torch.minimum(ratio * partial,
            ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * partial)
    if beta:
        ref = torch.where(active, _work(ref_lp.detach()), 0.0)
        current = torch.where(active, cur, 0.0)
        token_loss = token_loss + beta * k3(ref, current)
    result = torch.where(active, token_loss, 0.0).sum() / norm_const
    _finite_tensor(result, "group loss")
    return result


def combine(group_losses: list[torch.Tensor], lambdas: list[float]) -> torch.Tensor:
    """Combine complete usable groups once, using the step's fixed weight sum."""
    if not group_losses or len(group_losses) != len(lambdas):
        raise ValueError("combine: expected matching nonempty losses and lambdas")
    for loss in group_losses:
        _finite_tensor(loss, "group loss")
        if loss.ndim != 0 or loss.device != group_losses[0].device:
            raise ValueError("group loss: expected scalars on the same device")
    for weight in lambdas:
        _scalar(weight, "lambda")
    denominator = sum(lambdas)
    _scalar(denominator, "lambda sum", positive=True)
    result = torch.stack([loss * weight for loss, weight in zip(group_losses, lambdas, strict=True)]).sum()
    result = result / denominator
    _finite_tensor(result, "combined loss")
    return result


def repair_component_losses(cur_lp: torch.Tensor, old_lp: torch.Tensor, ref_lp: torch.Tensor,
                            action_mask: torch.Tensor, seq_keep: torch.Tensor, *,
                            answer: torch.Tensor, online: torch.Tensor, final: torch.Tensor,
                            kind: str, original_group_size: int, completion_cap: int,
                            finalization: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Separate fixed repair objectives on one forward; caller applies group weight once.

    The opt-in finalization channel exists only when passed and never carries a DIAG coefficient."""
    expected = {"T1": (4, 1024), "T2": (2, 1024), "T3": (2, 1024), "DIAG": (4, 8192)}
    if kind == "T1" and type(completion_cap) is int and 0 < completion_cap < 24576:
        # An opt-in eval-aligned T1 binds its own cap; this function has no protocol, so the bound value
        # is checked where the receipt is built (repair_training) and re-checked by repair_evidence.
        expected["T1"] = (4, completion_cap)
    if (type(kind) is not str or kind not in expected or type(original_group_size) is not int
            or type(completion_cap) is not int
            or (original_group_size, completion_cap) != expected[kind]):
        raise ValueError("repair loss: fixed original group size/cap differs")
    _matrix(cur_lp, "cur_lp")
    if cur_lp.shape[0] > original_group_size or cur_lp.shape[1] > completion_cap:
        raise ValueError("repair loss: action slice exceeds original group/cap")
    mask = _actions(action_mask, cur_lp)
    components = {"answer": answer, "online": online, "final": final,
                  **({"finalization": finalization} if finalization is not None else {})}
    for name, coefficients in components.items():
        _matching(cur_lp, coefficients, name)
        if bool((coefficients.detach()[~mask] != 0).any()):
            raise ValueError("repair loss: coefficient outside original action mask")
    forbidden: tuple[str, ...] = (("online", "final") if kind == "T1" else ("answer", "online") if kind == "DIAG"
                                  else ("final",))
    if kind == "DIAG" and finalization is not None:
        forbidden += ("finalization",)
    if any(bool((components[name].detach() != 0).any()) for name in forbidden):
        raise ValueError("repair loss: coefficient on an inapplicable channel")
    zeros = torch.zeros(cur_lp.shape[0], device=cur_lp.device, dtype=cur_lp.dtype)
    values = {name: group_loss(cur_lp, old_lp, ref_lp, zeros, mask, seq_keep,
        eps_low=.2, eps_high=.28, beta=0, norm_const=original_group_size * completion_cap,
        partial_adv=coefficients) for name, coefficients in components.items()}
    values["kl"] = group_loss(cur_lp, old_lp, ref_lp, zeros, mask, seq_keep,
        eps_low=.2, eps_high=.28, beta=.01, norm_const=original_group_size * completion_cap)
    values["total"] = sum(values.values(), torch.zeros((), device=cur_lp.device, dtype=values["kl"].dtype))
    _finite_tensor(values["total"], "repair loss")
    return values
