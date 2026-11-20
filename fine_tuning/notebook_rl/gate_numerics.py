"""G10 synthetic numerical/autograd measurements on the explicitly requested device.

The independent TRL comparison covers only token-level DR-GRPO with reward scaling
disabled, no extra importance correction, no KL ratio weighting and accumulation=1.
Its mask is supplied externally: our sequence MIS and group weighting are separate
checks, not claims of equivalence to the whole TRL trainer. No model, file or cloud IO.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from . import losses

TRL_SOURCE = {
    "tag": "v1.2.0",
    "tag_object": "36e40b6035734e0a4d79abe6975e2ef8a8be0cf7",
    "commit": "aca45154c8b6b1d9b0c06b674c0fa6bcfd9df051",
    "url": "https://raw.githubusercontent.com/huggingface/trl/"
           "aca45154c8b6b1d9b0c06b674c0fa6bcfd9df051/trl/trainer/grpo_trainer.py",
    "file_sha256": "4ac442a6d117838bf26fa3b786feb092c3953b2f9212081646db7fad9b356b41",
    "license": "Apache-2.0",
    "lines": {"no_reward_scaling": [2133, 2155], "ratio_k3": [2483, 2506],
              "clipping": [2513, 2521], "kl_dr_grpo": [2550, 2565]},
}


def _scalar(cur, old, ref, adv, mask, keep, norm: float, beta: float):
    """Scalar arithmetic and analytic derivative, away from clipping boundaries."""
    total = 0.0
    gradient = [[0.0] * len(row) for row in cur]
    for i, row in enumerate(cur):
        for j, value in enumerate(row):
            if not mask[i][j] or not keep[i]:
                continue
            ratio = math.exp(value - old[i][j])
            clipped = min(1.28, max(.8, ratio))
            delta = ref[i][j] - value
            total += (-min(ratio * adv[i], clipped * adv[i])
                      + beta * (math.expm1(delta) - delta)) / norm
            saturated = (adv[i] > 0 and ratio > 1.28) or (adv[i] < 0 and ratio < .8)
            derivative = 0.0 if saturated else -adv[i] * ratio
            gradient[i][j] = (derivative + beta * (1 - math.exp(delta))) / norm
    return total, gradient


def _trl_dr_grpo(cur, old, ref, adv, mask, cap: int, beta: float):
    """Independent transcription of the scoped, pinned TRL expressions above."""
    current, previous = cur.float(), old.detach().float()
    reference, advantage = ref.detach().float(), adv.detach().float().unsqueeze(1)
    coefficient = torch.exp(current - previous)
    clipped_coefficient = torch.clamp(coefficient, min=.8, max=1.28)
    per_token = -torch.minimum(coefficient * advantage, clipped_coefficient * advantage)
    delta = reference - current
    per_token = per_token + beta * (torch.exp(delta) - delta - 1)
    return (per_token * mask).sum() / (per_token.shape[0] * cap)


def measure_numerics(device: str | torch.device) -> dict:
    """Return observed checks, or raise; CPU results never imply CUDA execution."""
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("G10: only an explicit CPU or CUDA device is supported")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("G10: requested CUDA is unavailable; no CPU fallback")
        with torch.cuda.device(target):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("G10: requested CUDA device lacks BF16 support")
    target = torch.empty(0, device=target).device
    cases: list[dict[str, Any]] = []

    def check(name: str, actual: torch.Tensor, expected, *, atol=2e-6, rtol=2e-6):
        expected = torch.as_tensor(expected, dtype=actual.dtype, device=target)
        if actual.device != target or not bool(torch.isfinite(actual).all()):
            raise RuntimeError(f"G10 {name}: nonfinite result or wrong device")
        if expected.shape != actual.shape or not bool(torch.isfinite(expected).all()):
            raise RuntimeError(f"G10 {name}: invalid independent expectation")
        difference = float((actual.detach().double() - expected.double()).abs().max())
        matches = bool(torch.allclose(actual.detach(), expected, atol=atol, rtol=rtol))
        if not matches:
            raise RuntimeError(f"G10 {name}: numerical mismatch (max_abs_error={difference})")
        cases.append({"name": name, "match": matches, "device": str(actual.device),
            "dtype": str(actual.dtype), "actual": actual.detach().cpu().tolist(),
            "expected": expected.detach().cpu().tolist(), "max_abs_error": difference,
            "atol": atol, "rtol": rtol})

    def condition(name: str, passed: bool):
        check(name, torch.tensor(passed, device=target), True, atol=0, rtol=0)

    with torch.enable_grad():
        for dtype in (torch.float32, torch.bfloat16):
            prefix = str(dtype).removeprefix("torch.")

            def tensor(values, *, grad=False):
                return torch.tensor(values, device=target, dtype=dtype, requires_grad=grad)

            def objective(c, o, r, a, m, k, *, norm=12., beta=.07):
                return losses.group_loss(c, o, r, a, m, k, eps_low=.2, eps_high=.28,
                                         beta=beta, norm_const=norm)

            def scalar(c, o, r, a, m, k, *, norm=12., beta=.07):
                return _scalar(*(v.detach().float().cpu().tolist() for v in (c, o, r, a, m, k)),
                               norm, beta)

            rewards = tensor([0, 0, 3], grad=True)
            advantages = losses.group_advantages(rewards)
            check(f"{prefix}/no_std", advantages, [-1, -1, 2])
            condition(f"{prefix}/reward_detached", not advantages.requires_grad)
            equal = losses.group_advantages(tensor([7, 7], grad=True))
            check(f"{prefix}/equal_rewards", equal, [0, 0], atol=0, rtol=0)

            cur = tensor([[-1.4, -2., -7.], [-1., -1.8, -2.4], [-2., -1., -3.]], grad=True)
            old = tensor([[-1.8, -1.8, -7.], [-.6, -2., -2.4], [-2., -1., -3.]], grad=True)
            ref = tensor([[-1.1, -2.2, -7.], [-1.1, -1.4, -2.3], [-2., -1., -3.]], grad=True)
            adv = tensor([1.5, -1.5, .75], grad=True)
            mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1]], device=target, dtype=torch.bool)
            keep = torch.tensor([True, True, False], device=target)
            actual = objective(cur, old, ref, adv, mask, keep)
            expected, gradient = scalar(cur, old, ref, adv, mask, keep)
            check(f"{prefix}/scalar_clipped_loss", actual, expected)
            derivatives = torch.autograd.grad(actual, (cur, old, ref, adv), allow_unused=True,
                                              retain_graph=True)
            grad_tolerance = {"atol": 2e-4, "rtol": .01} if dtype == torch.bfloat16 else {}
            check(f"{prefix}/scalar_clipped_gradient", derivatives[0], gradient, **grad_tolerance)
            condition(f"{prefix}/old_ref_adv_detached", all(v is None for v in derivatives[1:]))
            condition(f"{prefix}/fp32_work", actual.dtype == advantages.dtype == torch.float32)

            trl = _trl_dr_grpo(cur, old, ref, adv, mask & keep[:, None], 4, .07)
            check(f"trl/{prefix}/loss", actual, trl.detach())
            check(f"trl/{prefix}/gradient", derivatives[0], torch.autograd.grad(trl, cur)[0],
                  **grad_tolerance)

            slices = [objective(cur[i:i+1], old[i:i+1], ref[i:i+1], adv[i:i+1],
                                mask[i:i+1], keep[i:i+1]) for i in range(3)]
            sliced = torch.stack(slices).sum()
            check(f"{prefix}/microbatch_loss", sliced, actual.detach())
            check(f"{prefix}/microbatch_gradient", torch.autograd.grad(sliced, cur, retain_graph=True)[0],
                  derivatives[0], **grad_tolerance)
            second = objective(cur[:2], old[:2], ref[:2], adv[:2], mask[:2], keep[:2], norm=10.)
            second_expected, second_gradient = scalar(cur[:2], old[:2], ref[:2], adv[:2],
                                                     mask[:2], keep[:2], norm=10.)
            combined = losses.combine([actual, second], [1., .5])
            check(f"{prefix}/group_weights_loss", combined, (expected + .5*second_expected)/1.5)
            weighted_grad = (torch.tensor(gradient, device=target)
                             + .5*torch.tensor(second_gradient + [[0., 0., 0.]], device=target))/1.5
            combined_gradient = torch.autograd.grad(combined, cur)[0]
            check(f"{prefix}/group_weights_gradient", combined_gradient,
                  weighted_grad, **grad_tolerance)
            second_slices = [objective(cur[i:i+1], old[i:i+1], ref[i:i+1], adv[i:i+1],
                                      mask[i:i+1], keep[i:i+1], norm=10.) for i in range(2)]
            sliced_step = (sliced + .5*torch.stack(second_slices).sum())/1.5
            check(f"{prefix}/weighted_microbatch_loss", sliced_step, combined.detach())
            check(f"{prefix}/weighted_microbatch_gradient", torch.autograd.grad(sliced_step, cur)[0],
                  combined_gradient, **grad_tolerance)

            ratio_one = cur[:2].detach().clone().requires_grad_()
            ratio_adv = tensor([.5, -.5])
            ones = torch.ones_like(ratio_one, dtype=torch.bool)
            all_keep = torch.ones(2, device=target, dtype=torch.bool)
            same = objective(ratio_one, ratio_one.detach(), ratio_one.detach(), ratio_adv,
                             ones, all_keep, beta=0.)
            check(f"{prefix}/ratio_one_loss", same, 0., atol=0, rtol=0)
            check(f"{prefix}/ratio_one_gradient", torch.autograd.grad(same, ratio_one)[0],
                  [[-.5/12]*3, [.5/12]*3], **grad_tolerance)
            kl_only = objective(ratio_one, ratio_one.detach()-.4, ref[:2], equal, ones, all_keep)
            kl_value, kl_gradient = scalar(ratio_one, ratio_one.detach()-.4, ref[:2], equal,
                                          ones, all_keep)
            check(f"{prefix}/equal_rewards_kl_loss", kl_only, kl_value)
            check(f"{prefix}/equal_rewards_kl_gradient", torch.autograd.grad(kl_only, ratio_one)[0],
                  kl_gradient, **grad_tolerance)

            current = tensor([-1., -2., -3., -4.], grad=True)
            reference = tensor([-1., -1.9999, -3.25, -3.6], grad=True)
            delta = (reference.detach().float() - current.detach().float()).cpu().tolist()
            kl = losses.k3(reference, current)
            check(f"{prefix}/k3_values", kl, [math.expm1(d)-d for d in delta])
            grad_cur, grad_ref = torch.autograd.grad(kl.sum(), (current, reference), allow_unused=True)
            check(f"{prefix}/k3_gradient", grad_cur, [1-math.exp(d) for d in delta], **grad_tolerance)
            condition(f"{prefix}/k3_nonnegative_detached_fp32",
                      bool((kl >= 0).all()) and grad_ref is None and kl.dtype == torch.float32)

            old_mis = tensor([[-1., -1., -1.]]*3, grad=True)
            sampled = tensor([[-1.1, -1.1, -500.], [-1.01, -1.01, -500.],
                              [-1.1, -.9, -500.]], grad=True)
            actions = torch.tensor([[1, 1, 0]]*3, device=target, dtype=torch.bool)
            retained, mean = losses.mis_mask(old_mis, sampled, actions, .05)
            means = [(sum(o[:2])-sum(s[:2]))/2 for o, s in
                     zip(old_mis.detach().float().cpu().tolist(), sampled.detach().float().cpu().tolist(), strict=True)]
            check(f"{prefix}/mis_0_1_reject_0_01_keep", retained, [False, True, True], atol=0, rtol=0)
            check(f"{prefix}/mis_signed_action_mean", mean, means)
            condition(f"{prefix}/mis_detached_fp32", not mean.requires_grad and mean.dtype == torch.float32)

            extreme = tensor([[-1., -1000.], [-1000., -1000.]], grad=True)
            previous = tensor([[-1., 1000.], [1000., 1000.]])
            reference_extreme = tensor([[-.8, 1000.], [1000., 1000.]])
            extreme_mask = torch.tensor([[1, 0], [1, 1]], device=target, dtype=torch.bool)
            extreme_keep = torch.tensor([True, False], device=target)
            extreme_adv = tensor([1., -1.])
            masked = objective(extreme, previous, reference_extreme, extreme_adv, extreme_mask, extreme_keep)
            masked_value, masked_gradient = scalar(extreme, previous, reference_extreme,
                                                   extreme_adv, extreme_mask, extreme_keep)
            check(f"{prefix}/padding_excluded_loss", masked, masked_value)
            measured_gradient = torch.autograd.grad(masked, extreme)[0]
            check(f"{prefix}/padding_excluded_gradient", measured_gradient, masked_gradient, **grad_tolerance)
            check(f"{prefix}/excluded_exact_zero", measured_gradient[~(extreme_mask & extreme_keep[:, None])],
                  [0., 0., 0.], atol=0, rtol=0)
            dropped = objective(extreme, previous, reference_extreme, extreme_adv, extreme_mask,
                                torch.zeros_like(extreme_keep))
            check(f"{prefix}/all_excluded_loss", dropped, 0., atol=0, rtol=0)
            check(f"{prefix}/all_excluded_gradient", torch.autograd.grad(dropped, extreme)[0],
                  [[0., 0.], [0., 0.]], atol=0, rtol=0)

    return {"numeric_tests_match": bool(cases) and all(c["match"] for c in cases),
        "trl_formula_match": all(c["match"] for c in cases if c["name"].startswith("trl/")),
        "device": str(target), "input_dtypes": ["float32", "bfloat16"], "work_dtype": "float32",
        "cases": cases, "max_abs_error": max(c["max_abs_error"] for c in cases),
        "trl_source": dict(TRL_SOURCE),
        "trl_scope": {"loss_type": "dr_grpo", "importance_sampling_level": "token",
            "scale_rewards": "none", "epsilon_low": .2, "epsilon_high": .28,
            "use_bias_correction_kl": False, "gradient_accumulation_steps": 1,
            "extra_importance_correction": False, "mask": "externally supplied action/sequence mask"},
        "limitations": ["Synthetic objective/autograd checks only; no model or rollout evidence",
            "Independent TRL formula transcription; TRL trainer is not installed or executed",
            "Sequence MIS, group weights and microbatch normalization are project-specific checks",
            "CPU results do not certify CUDA, BF16 model execution or G0-G13 completion"]}
