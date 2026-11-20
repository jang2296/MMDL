"""v18 source-bound token weighting and signed note evidence for the shared v17 trainer."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from fine_tuning.notebook_rl.input_preparation import digest

from . import train_v17 as previous
from .contract import IMAGE

ROW_SCHEMA = "error-note-v18-row-v1"
SFT_WEIGHTS = {"PREP_SELF": 1., "PREP_HUMAN": 1., "A1": .09, "C": .30, "N": .05}
PROCESS_WEIGHTS = {"A1_verified": .18, "A1_span": .02, "C_span": .05}
LOCALIZED_REASONS = {"FORMAT_ONLY_CERTIFIED", "LOOP_CONFIRMED"}


def _list(value: Any) -> Any:
    return value.tolist() if isinstance(value, torch.Tensor) else value


def _legacy_row(row: dict) -> dict:
    value = row | {"schema": previous.ROW_SCHEMA}
    if row["kind"] in previous.PG_KINDS:
        n = row.get("target_len", len(row["generation"]["completion_token_ids"]))
        value.update(loss_mask=torch.ones(n, dtype=torch.bool), loss_token_mask=torch.ones(n, dtype=torch.bool))
        if row.get("reason") == "LOOP_CONFIRMED":
            value["reason"] = "INCORRECT"
            value["generation"] = row["generation"] | {"finish_reason": "stop"}
    return value


def _mask(tokenizer: Any, row: dict, spans: list) -> list[bool]:
    actions = row["generation"]["completion_token_ids"]
    if not spans:
        return [False] * len(actions)
    return previous._sample_mask(tokenizer, actions, row["target"], previous._end_ids(tokenizer), spans)


def _validate_masks(row: dict, tokenizer: Any) -> None:
    kind, target = row["kind"], row["target"]
    if kind == "PREP_HUMAN":
        return
    if row["generation"].get("text") != target:
        raise ValueError("v18: original sampled text changed")
    actions = row["generation"]["completion_token_ids"]
    full = previous._sample_mask(tokenizer, actions, target, previous._end_ids(tokenizer))
    supplied = _list(row["loss_mask"])
    if kind in previous.PG_KINDS:
        reason = row.get("pg_mask_reason", row.get("reason"))
        if kind == "RL" and reason == "VERIFIED_PROCESS_ERROR_FINAL_ONLY":
            policy, receipt = row.get("t1_policy_receipt", {}), row.get("policy_mask_receipt", {})
            ledger = policy.get("t1_claim_ledger", {})
            spans = row.get("pg_byte_spans", [])
            expected = _mask(tokenizer, row, spans)
            errors = [claim for claim in ledger.get("claims", []) if claim.get("verdict") == "CONTRADICTED" and claim.get("assertion") == "ASSERTED"]
            if (any(value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}) for value in (policy, receipt, ledger))
                    or (row.get("pg") and row.get("advantage", 0) <= 0) or (not row.get("pg") and row.get("advantage") != 0)
                    or not errors or ledger.get("problem_id") != row["problem_id"]
                    or ledger.get("generation_sha256") != digest(row["generation"])
                    or ledger.get("image_sha256") != row.get("image_sha256", [])
                    or receipt.get("claim_ledger_sha256") != ledger.get("sha256")
                    or receipt.get("mask_reason") != reason or receipt.get("byte_spans") != spans
                    or receipt.get("target_sha256") != hashlib.sha256(target.encode()).hexdigest()
                    or receipt.get("token_mask") != expected or supplied != expected
                    or receipt.get("contradiction_byte_spans") != [span for error in errors for span in error["original_byte_spans"]]
                    or any(a < end and start < b for start, end in spans for error in errors for a, b in error["original_byte_spans"])):
                raise ValueError("v18b: positive policy mask lacks verified contradiction binding")
            import re
            finals = [[len(target[:match.start()].encode()), len(target[:match.end()].encode())]
                      for match in re.finditer(r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*[^\n]+$", target)]
            if spans and (len(finals) != 1 or spans != finals):
                raise ValueError("v18b: positive policy mask is not a unique final answer")
            if row.get("pg") is not any(expected):
                raise ValueError("v18b: positive policy mask/hold mismatch")
            return
        # A localized reason whose spans mapped to no action leaves pg False and an all-False mask: an excluded row.
        if kind == "RL" and reason in LOCALIZED_REASONS and (row.get("pg") or any(supplied)):
            spans, receipt = row.get("pg_byte_spans", []), row.get("policy_mask_receipt", {})
            expected = _mask(tokenizer, row, spans)
            if (not any(expected) or receipt.get("sha256") != digest({k: v for k, v in receipt.items() if k != "sha256"})
                    or receipt.get("target_sha256") != hashlib.sha256(target.encode()).hexdigest()
                    or receipt.get("mask_reason") != reason or receipt.get("byte_spans") != spans
                    or receipt.get("token_mask") != expected or supplied != expected
                    or row.get("advantage", 0) > 0):
                raise ValueError("v18: localized policy mask lacks bound raw-token evidence")
            if reason == "LOOP_CONFIRMED":
                from .v18_rows import repeated_byte_spans_v18

                if (spans != repeated_byte_spans_v18(target) or row.get("reason") != "LOOP_CONFIRMED"
                        or row.get("reward") != 0 or row.get("pg") is not True
                        or row["generation"].get("finish_reason") not in {"stop", "eos", "length"}):
                    raise ValueError("v18: repeated-interval penalty lacks actual detector evidence")
            if reason == "FORMAT_ONLY_CERTIFIED":
                form = row.get("format_receipt", {})
                if (form.get("sha256") != digest({k: v for k, v in form.items() if k != "sha256"})
                        or form.get("accepted") is not True or form.get("content_equivalent") is not True
                        or form.get("output_binding_sha256") != digest({"generation": row["generation"], "text": target,
                            "prompt_sha256": row.get("generation_prompt_sha256")})
                        or receipt.get("output_binding_sha256") != form.get("output_binding_sha256")
                        or form.get("byte_spans") != spans):
                    raise ValueError("v18: format-only policy mask lacks content certification")
        elif supplied != ([True] * len(actions) if row.get("pg") else [False] * len(actions)):
            raise ValueError("v18: ordinary policy mask must retain all sampled actions or exclude an ungraded row")
        return
    if kind == "N":
        evidence = row.get("n_source_evidence")
        if evidence is not None:
            generation = row["generation"]
            prefix = row["input_ids"][:-row["target_len"]].tolist()
            if (row["training_prompt_sha256"] != row.get("generation_prompt_sha256")
                    or generation.get("prompt_token_ids") != prefix
                    or generation.get("input_sha256") != row.get("input_sha256")):
                raise ValueError("v18b: note training context differs from the generated note")
            expected_evidence = {"schema": "v18b-n-source-evidence-v1", "row_id": row["row_id"],
                "problem_id": row["problem_id"], "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                "generation_sha256": digest(row["generation"]), "receipt_sha256": row["verification_receipt"]["sha256"],
                "byte_spans": row["verified_byte_spans"], "token_ids": actions, "token_mask": supplied,
                "supervised_tokens": sum(supplied), "note_condition": row["verification_receipt"].get("note_condition"),
                "provided_location_count": row["verification_receipt"].get("provided_location_count", 0)}
            if evidence != expected_evidence | {"sha256": digest(expected_evidence)}:
                raise ValueError("v18b: note source evidence changed")
        if supplied != _mask(tokenizer, row, row.get("verified_byte_spans", [])):
            raise ValueError("v18: note mask differs from verified sampled bytes")
        return
    if supplied != full:
        raise previous.RowEncodingError("SFT_CONTENT_MASK", "v18: self SFT must use every nonterminal sampled action")
    if kind not in {"A1", "C"}:
        return
    receipt = row["verification_receipt"]
    spans = row.get("process_evidence_byte_spans", [])
    expected = _mask(tokenizer, row, spans)
    grade = receipt.get("evidence_grade")
    if (grade not in {"RESULT_ONLY", "SOURCE_GROUNDED"}
            or receipt.get("verified_process_byte_spans", []) != spans
            or (grade == "SOURCE_GROUNDED") != any(expected)
            or any(expected) and receipt.get("verified_evidence_count", 0) < 1):
        raise ValueError("v18: source-grounded process mask or admission grade changed")
    if _list(row.get("verified_process_token_mask")) != expected:
        raise previous.RowEncodingError("PROCESS_MASK", "v18: source-grounded process mask differs from verified bytes")
    use_spans = row.get("corrected_use_byte_spans", [])
    used = _mask(tokenizer, row, use_spans)
    if (receipt.get("corrected_use_byte_spans", []) != use_spans
            or kind != "C" and any(used)
            or any(value and not verified for value, verified in zip(used, expected, strict=True))):
        raise ValueError("v18: corrected-use mask lacks verified source use")
    if _list(row.get("corrected_use_token_mask", [False] * len(actions))) != used:
        raise previous.RowEncodingError("CORRECTED_USE_MASK", "v18: corrected-use mask differs from verified bytes")


def encode_v18_row(processor: Any, model_config: Any, row: dict, images: list, *, image: dict = IMAGE) -> dict:
    """Reuse original action/input checks, then recompute every added mask from sampled UTF-8 bytes."""
    if row.get("schema") != ROW_SCHEMA:
        raise ValueError("v18: wrong row schema")
    legacy = _legacy_row(row)
    if row["kind"] in previous.PG_KINDS:
        legacy.pop("loss_mask", None)
        legacy.pop("loss_token_mask", None)
    result = previous.encode_v17_row(processor, model_config, legacy, images, image=image)
    result.update({key: row[key] for key in ("schema", "reason", "generation") if key in row})
    supplied = [_list(row[key]) for key in ("loss_mask", "loss_token_mask") if key in row]
    if any(not isinstance(mask, list) or len(mask) != result["target_len"]
           or any(type(bit) is not bool for bit in mask) or mask != supplied[0] for mask in supplied):
        raise previous.RowEncodingError("TOKEN_MASK", "v18: supplied raw-token masks differ or contain non-booleans")
    if supplied:
        result["loss_mask"] = torch.tensor(supplied[0], dtype=torch.bool)
        result["loss_token_mask"] = result["loss_mask"].clone()
    _validate_masks(result, processor.tokenizer)
    return result


def validate_v18_block(bundle: Any, encoded: list[dict], contract: dict) -> None:
    if (contract.get("version") not in {"error-note-v18", "error-note-v18b"} or contract.get("row_schema") != ROW_SCHEMA
            or contract.get("sft_weights") != SFT_WEIGHTS or contract.get("process_weights") != PROCESS_WEIGHTS
            or any(row.get("schema") != ROW_SCHEMA for row in encoded)):
        raise ValueError("v18: sealed row schema and source-specific loss weights required")
    previous.validate_v17_block(bundle, [_legacy_row(row) for row in encoded], contract | {"version": "error-note-v17"})
    for row in encoded:
        if contract.get("version") == "error-note-v18b" and row["kind"] == "N" and not row.get("n_source_evidence"):
            raise ValueError("v18b: N row requires source-to-token evidence")
        mask = row.get("loss_mask")
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != (row["target_len"],)):
            raise ValueError("v18: encoded raw-token policy/SFT mask shape changed")
        _validate_masks(row, bundle.processor.tokenizer)


def sft_term_v18(row: dict, raw: torch.Tensor, contract: dict, denominator: int) -> tuple[torch.Tensor, int]:
    """Average a complete row and the independently verified source-use span separately."""
    kind = row["kind"]
    mask = row["loss_mask"].to(raw.device)
    coefficient = contract["sft_weights"][kind]
    if kind == "A1" and row["verification_receipt"]["evidence_grade"] == "SOURCE_GROUNDED":
        coefficient = contract["process_weights"]["A1_verified"]
    term = -coefficient * raw[mask].mean()
    selected = row.get("verified_process_token_mask") if kind == "A1" else row.get("corrected_use_token_mask") if kind == "C" else None
    count = 0
    if selected is not None:
        verified = torch.tensor(_list(selected), device=raw.device, dtype=torch.bool)
        count = int(verified.sum())
        if count:
            term -= contract["process_weights"][kind + "_span"] * raw[verified].mean()
    return term / denominator, count


def backward_v18_block(bundle: Any, encoded: list[dict], contract: dict, token_budget: int = 12000, **kwargs: Any) -> dict:
    return previous.backward_v17_block(bundle, encoded, contract, token_budget, **kwargs)


def note_step_before_v18(bundle: Any, note_gradients: dict, stats: dict) -> tuple[dict, dict]:
    """Keep CPU snapshots only for parameters touched by the actual signed N channel."""
    snapshots, dot, total_sq, note_sq = {}, 0., 0., 0.
    for (group, index), note in note_gradients.items():
        parameter = bundle.params[group][index]
        if parameter.grad is None:
            continue
        total = parameter.grad.detach().float().cpu()
        snapshots[group, index] = parameter.detach().float().cpu().clone()
        dot += float((note.double() * total.double()).sum())
        total_sq += float(total.double().square().sum())
        note_sq += float(note.double().square().sum())
    report = {"optimizer_step": False, "gradient_measured": bool(note_gradients),
              "parent_ids": stats.get("verified_repair_parent_ids", []),
              "row_ids": [row["row_id"] for row in stats.get("n_parent_evidence", []) if row.get("note_gradient_retained")],
              "n_parent_evidence": stats.get("n_parent_evidence", []),
              "norms": stats.get("note_gradient_norms", {}), "signed_n_gradient_norm": math.sqrt(note_sq),
              "combined_gradient_norm_on_note_parameters": math.sqrt(total_sq),
              "combined_gradient_dot": dot if note_gradients else None,
              "parameter_delta_dot": None, "locally_descends_note_loss": None}
    return snapshots, report


def note_step_after_v18(bundle: Any, note_gradients: dict, snapshots: dict, report: dict, *, step: bool) -> dict:
    report["optimizer_step"] = step
    if note_gradients:
        dot = sum(float(((bundle.params[group][index].detach().float().cpu() - before).double()
                         * note_gradients[group, index].double()).sum())
                  for (group, index), before in snapshots.items())
        if not math.isfinite(dot):
            raise ValueError("v18: nonfinite actual note/update inner product")
        report.update(parameter_delta_dot=dot, locally_descends_note_loss=bool(step and dot < 0))
    return report


# Capacity fixtures measure the same unmodified image/model/optimizer residency.
measure_update_memory_v18 = previous.measure_update_memory_v17


def calibrate_update_v18(bundle: Any, encoded: list[dict], contract: dict, *, token_budget: int = 12000,
                         capacity: Any = None) -> dict:
    report = previous.calibrate_update_v17(bundle, encoded, contract, token_budget=token_budget, capacity=capacity)
    body = {key: value for key, value in report.items() if key != "sha256"}
    body.update(objective_version="error-note-v18", row_schema=ROW_SCHEMA)
    return body | {"sha256": digest(body)}
