"""Next-token training on the model's own solutions and notes, with save and fresh-process reload.

Plan: claudedocs/research_self_note_finetuning_plan_20261101.md sections 3.4, 5.5. Rows come from the collect stage.
Loss covers target tokens and the end token only: a wrong solution inside a note prompt and all image tokens are input.

CLI: python -m fine_tuning.error_note.train train --rows ROWS.jsonl --build-root V --order ORDER.json --limit N
         --arm {note,solution,no-note} --precision {bf16,nf4} --model-path M --output DIR --code-commit 40HEX
         [--token-budget T] [--gradient-checkpointing {on,off}] [--save-every 50] [--max-row-tokens K] [--resume]
         [--protocol-version {v1,v2,v3}] [--admission ADMISSION.json --audit AUDIT.json --sidecar SIDECAR.json --solution-audit SOLUTIONS.json]
     python -m fine_tuning.error_note.train admit --rows ROWS.jsonl --build-root V --order ORDER.json --limit N
         --precision {bf16,nf4} --model-path M --output ADMISSION.json --audit AUDIT.json --sidecar SIDECAR.json
         --solution-audit SOLUTIONS.json --code-commit 40HEX
     python -m fine_tuning.error_note.train reload --run DIR --model-path M --precision {bf16,nf4} [--protocol-version {v1,v2,v3}]
     python -m fine_tuning.error_note.train dry-run --rows R --build-root V --order O --limit 1 --model-path M
         --output DIR --precision nf4 --code-commit 40HEX --audit A --sidecar S --solution-audit Q --admission P [--token-budget T]
Protocol v2 (claudedocs/plan_self_note_v2_20261102.md section 4): 16 items per update and a linear LR decay whose end is
the number of non-empty update groups of the rows file, so the protocol (and its binding) follows the rows.
Version 3 requires source-reviewed whole solutions, admits both arms before training, preserves the common A routes,
and uses .9 mean(A) + .1 mean(B).
Its no-note arm omits B without renormalizing A; solution remains a legacy-only arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint as recompute

from fine_tuning.modules import tensor_sha256
from fine_tuning.notebook_rl import checkpoint as ckpt
from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
from fine_tuning.notebook_rl.policy import PolicyBundle, assert_base_unchanged, build_policy
from fine_tuning.notebook_rl.prompts import user_message
from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged, build_quantized_policy

from .contract import (CONTEXT_LIMIT, IMAGE, KIND_NOTE, KIND_SOLUTION, MASTER_SEED, MAX_NEW_TOKENS,
                       PER_UPDATE_ITEMS, ROUTE_FULL, ROUTE_LM, select_rows, update_groups)
from .order import read_order
from .protocol import (ADMISSION_VERSION_V3, ARM, GRAD_CLIP, PER_UPDATE_ITEMS_V2, VERSION_V2, VERSION_V3,
                       _digest, plain_protocol, plain_protocol_v2, plain_protocol_v3)

END_TOKEN = "<|im_end|>"
PROJECTION_CHUNK = 1024
PROBE_ROWS = 2
PROBE_NEW_TOKENS = 16
PROBE_ATOL = 1e-3
PROBE_TENSORS = ("input_ids", "pixel_values", "image_grid_thw")
ENCODE_WORKERS = 8
ENCODE_BUFFER = 32  # rows ready ahead of the GPU; each holds its pixels in RAM
SOLUTION_DROP_FIELD = "_solution_quality_drop"
_SOLUTION_DROP_MARKER = object()
STATUS_COMPLETE, STATUS_NONFINITE, STATUS_NO_UPDATES = "COMPLETE", "NONFINITE", "NO_UPDATES"


class NonFiniteStop(RuntimeError):
    """A loss or gradient is not finite. The run stops; nothing is skipped."""


class DropRow(Exception):
    """A row that must not be trained on; the argument is the recorded reason."""


class AdmissionChanged(RuntimeError):
    """An encoded group differs from the preflight; no row may silently leave a v3 run."""


def plan_microbatches(lengths: list[int], routes: list[str], token_budget: int = 0) -> list[list[int]]:
    """Index lists: same route, longest first, max(length) * count <= budget. An over-budget row stays alone; 0 = one row each."""
    if len(lengths) != len(routes):
        raise ValueError("lengths and routes differ in size")
    batches: list[list[int]] = []
    for route in sorted(set(routes)):
        current: list[int] = []
        for index in sorted((i for i, r in enumerate(routes) if r == route), key=lambda i: (-lengths[i], i)):
            if current and token_budget and lengths[current[0]] * (len(current) + 1) <= token_budget:
                current.append(index)
            else:
                if current:
                    batches.append(current)
                current = [index]
        batches.append(current)
    return batches


def encode_row(processor: Any, model_config: Any, row: dict, images: list[Any]) -> dict:
    """CPU only. input_ids = the evaluation-style prompt + target ids + end token; target_len counts the last two."""
    tokenizer = processor.tokenizer
    if any(token in text for token in tokenizer.all_special_tokens for text in (row["prompt"], row["target"])):
        raise DropRow("CONTROL_TOKEN")  # model text would otherwise become image placeholders or end tokens
    prepared, _ = prepare_protocol_inputs(processor, model_config, [user_message(row["prompt"], len(images))], images, IMAGE)
    inputs = prepared["inputs"]
    target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(END_TOKEN)]
    return {"row_id": row["row_id"], "kind": row["kind"], "route": row["route"],
            "input_ids": torch.cat([inputs["input_ids"][0], torch.tensor(target, dtype=torch.long)]), "target_len": len(target),
            "pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"]}


def _pad_id(bundle: PolicyBundle) -> int:
    tokenizer = getattr(bundle.processor, "tokenizer", None)
    pad = tokenizer.pad_token_id if tokenizer is not None else bundle.model.get_base_model().config.pad_token_id
    if type(pad) is not int:
        raise ValueError("train: an explicit pad token id is required")
    return pad


def _image_inputs(rows: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    with_images = [row for row in rows if row.get("pixel_values") is not None]
    if not with_images:
        return {}
    return {"pixel_values": torch.cat([row["pixel_values"] for row in with_images]).to(device),
            "image_grid_thw": torch.cat([row["image_grid_thw"] for row in with_images]).to(device)}


def _autocast(device: torch.device) -> Any:
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def batch_logprobs(bundle: PolicyBundle, encoded_rows: list[dict], *, chunk: int = PROJECTION_CHUNK) -> list[torch.Tensor]:
    """Per-row 1-D log probabilities of the target tokens from one right-padded forward (policy._logprobs for many rows)."""
    if not encoded_rows:
        raise ValueError("train: an empty micro-batch")
    base = bundle.model.get_base_model()
    device = next(base.parameters()).device
    lengths = [row["input_ids"].numel() for row in encoded_rows]
    if any(not 0 < row["target_len"] < length for row, length in zip(encoded_rows, lengths, strict=True)):
        raise ValueError("train: a nonempty prompt and target are required")
    ids = torch.full((len(encoded_rows), max(lengths)), _pad_id(bundle), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, (row, length) in enumerate(zip(encoded_rows, lengths, strict=True)):
        ids[index, :length] = row["input_ids"]
        mask[index, :length] = 1
    ids, mask = ids.to(device), mask.to(device)

    def project(hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = base.lm_head(hidden).float()
        return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    results = []
    with _autocast(device):
        # The body directly, so no full-vocabulary logits are allocated at prompt positions.
        hidden = base.model(input_ids=ids, attention_mask=mask, **_image_inputs(encoded_rows, device),
                            use_cache=False, return_dict=True).last_hidden_state
        for index, (row, length) in enumerate(zip(encoded_rows, lengths, strict=True)):
            start = length - row["target_len"]
            states, actions = hidden[index, start - 1:length - 1], ids[index, start:length]
            pieces = []
            for offset in range(0, row["target_len"], chunk):
                h, tokens = states[offset:offset + chunk], actions[offset:offset + chunk]
                # Without the recompute, backward would retain all vocabulary logits of every chunk.
                pieces.append(recompute(project, h, tokens, use_reentrant=False) if torch.is_grad_enabled()
                              else project(h, tokens))
            values = torch.cat(pieces)
            if not bool(torch.isfinite(values).all()):
                raise NonFiniteStop(f"nonfinite log probability in row {row['row_id']}")
            results.append(values)
    return results


def _clip_groups(bundle: PolicyBundle) -> dict[str, float | None]:
    """Norm before clipping per group; None for a group with no gradient, which AdamW then skips."""
    norms: dict[str, float | None] = {}
    for group in ("L", "M", "V"):
        active = [p for p in bundle.params[group] if p.grad is not None]
        norm = float(torch.nn.utils.clip_grad_norm_(active, GRAD_CLIP)) if active else None
        if norm is not None and not math.isfinite(norm):
            raise NonFiniteStop(f"nonfinite gradient in group {group}")
        norms[group] = norm
    return norms


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_initialized() else 0.0


def _backward_rows(bundle: PolicyBundle, encoded: list[dict], plan: list[list[int]],
                   coefficients: list[float] | None = None) -> dict[str, float]:
    """Shared target/EOS NLL backward only; never constructs an optimizer or changes parameters."""
    loss_by_kind: dict[str, float] = defaultdict(float)
    for indices in plan:
        rows = [encoded[index] for index in indices]
        bundle.route.apply(rows[0]["route"] == ROUTE_FULL)
        row_losses = [-values.mean() for values in batch_logprobs(bundle, rows)]
        if coefficients is None:
            (torch.stack(row_losses).sum() / len(encoded)).backward()
        else:
            torch.stack([loss * coefficients[index] for index, loss in zip(indices, row_losses, strict=True)]).sum().backward()
        for row, value in zip(rows, row_losses, strict=True):
            loss_by_kind[row["kind"]] += float(value.detach())
    return loss_by_kind


def run_update(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: Any, encoded: list[dict],
               token_budget: int = 0, *, kind_weights: dict[str, float] | None = None) -> dict:
    """One optimizer step over every row of one order group. Raises NonFiniteStop before any parameter changes."""
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    kinds = Counter(row["kind"] for row in encoded)
    if kind_weights is not None and (not kinds[KIND_SOLUTION] or set(kinds) - {KIND_SOLUTION, KIND_NOTE}
                                    or set(kind_weights) != {KIND_SOLUTION, KIND_NOTE}
                                    or any(not math.isfinite(w) or w < 0 for w in kind_weights.values())):
        raise ValueError("train v3: a group needs A rows and finite nonnegative A/B weights")
    coefficients = [kind_weights[row["kind"]] / kinds[row["kind"]] if kind_weights is not None else 1 / len(encoded)
                    for row in encoded]
    plan = plan_microbatches([row["input_ids"].numel() for row in encoded], [row["route"] for row in encoded], token_budget)
    loss_by_kind = _backward_rows(bundle, encoded, plan, coefficients if kind_weights is not None else None)
    grad_norms = _clip_groups(bundle)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    bundle.route.apply(True)
    means = {k: loss_by_kind[k] / n for k, n in kinds.items()}
    weighted = {k: kind_weights[k] * means.get(k, 0.0) for k in kind_weights} if kind_weights is not None else {}
    return {"rows": len(encoded), "rows_by_kind": dict(kinds), "routes": dict(Counter(row["route"] for row in encoded)),
            "tokens": sum(row["input_ids"].numel() for row in encoded), "target_tokens": sum(row["target_len"] for row in encoded),
            "loss": sum(weighted.values()) if kind_weights is not None else sum(loss_by_kind.values()) / len(encoded),
            "loss_by_kind": means,
            "grad_norms": grad_norms, "seconds": time.perf_counter() - started, "peak_allocated_gib": _peak_gib(),
            "microbatches": len(plan), "skipped": False,
            **({"denominators": {k: kinds[k] for k in kind_weights}, "weighted_loss_by_kind": weighted,
                "row_coefficients": {row["row_id"]: coefficients[i] for i, row in enumerate(encoded)},
                "tokens_by_kind": {k: sum(row["input_ids"].numel() for row in encoded if row["kind"] == k)
                                   for k in kind_weights},
                "routes_by_kind": {k: dict(Counter(row["route"] for row in encoded if row["kind"] == k))
                                   for k in kind_weights}} if kind_weights is not None else {})}


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    # Line iteration, not str.splitlines: raw U+2028 inside JSON text is valid and must not split a row.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _keep_before(path: Path, position: int) -> None:
    """Drop this run's records past the resume point, including a line cut by a crash."""
    if not path.exists():
        return
    kept = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["group"] < position:
                kept.append(line)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(kept), encoding="utf-8")
    os.replace(tmp, path)


def _read_groups(rows_path: Path, order_path: Path, limit: int, arm: str,
                 per_update: int = PER_UPDATE_ITEMS, *, v3: bool = False, audit_path: Path | None = None,
                 solution_audit_path: Path | None = None) -> list[list[dict]]:
    """Groups come from the order, so both arms get the same updates; a group may be empty."""
    ids = [item["problem_id"] for item in read_order(order_path)["items"]]
    if not 1 <= limit <= len(ids):
        raise ValueError(f"train: limit {limit} is outside the order of {len(ids)} items")
    if v3:
        from .audit import solution_quality_reasons, validate_row_audits
        from .contract import select_rows_v3

        rows = _read_jsonl(rows_path)
        if audit_path is None or solution_audit_path is None:
            raise ValueError("train v3: --audit and --solution-audit are required")
        if any(SOLUTION_DROP_FIELD in row for row in rows):
            raise ValueError("train v3: reserved solution-quality row field supplied")
        validate_row_audits(rows, audit_path)
        reasons = solution_quality_reasons(rows, audit_path, solution_audit_path,
                                           rows_sha256=ckpt.file_hash(rows_path))
        rows = [row | {_SOLUTION_DROP_MARKER: reasons[row["row_id"]]} if row["row_id"] in reasons else row for row in rows]
        rows = select_rows_v3(rows, arm)
        if any(row["problem_id"] not in ids or row["kind"] not in (KIND_SOLUTION, KIND_NOTE) for row in rows):
            raise ValueError("train v3: an unknown problem_id or row kind in the rows file")
    else:
        rows = select_rows(_read_jsonl(rows_path), arm)
    if len({row["row_id"] for row in rows}) != len(rows) or any(row["route"] not in (ROUTE_FULL, ROUTE_LM) for row in rows):
        raise ValueError("train: duplicate row_id or unknown route in the rows file")
    by_id: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda row: row["kind"]):
        by_id.setdefault(row["problem_id"], []).append(row)
    return [[row for problem_id in group for row in by_id.get(problem_id, [])]
            for group in update_groups(ids[:limit], per_update)]


def _encode_or_drop(row: dict, encode: Callable[[dict], dict], max_row_tokens: int | None) -> dict:
    if _SOLUTION_DROP_MARKER in row:
        return {"row_id": row["row_id"], "reason": row[_SOLUTION_DROP_MARKER]}
    try:
        item = encode(row)
    except DropRow as exc:
        return {"row_id": row["row_id"], "reason": str(exc)}
    tokens = item["input_ids"].numel()
    if max_row_tokens and tokens > max_row_tokens:
        return {"row_id": row["row_id"], "reason": "OVER_MAX_ROW_TOKENS", "tokens": tokens}
    return item


def encoded_groups(groups: list[list[dict]], encode: Callable[[dict], dict], max_row_tokens: int | None,
                   map_fn: Callable[..., Iterable]) -> Iterator[tuple[list[dict], list[dict]]]:
    """(kept encoded rows, drop records) per group. map_fn keeps input order and may run ahead of the GPU."""
    fn = partial(_encode_or_drop, encode=encode, max_row_tokens=max_row_tokens)
    results = iter(map_fn(fn, (row for group in groups for row in group)))
    for group in groups:
        items = [next(results) for _ in group]
        yield [item for item in items if "input_ids" in item], [item for item in items if "input_ids" not in item]


def _vault_encoder(bundle: PolicyBundle, build_root: Path) -> Callable[[dict], dict]:
    return _actor_encoder(bundle.processor, bundle.model.get_base_model().config, build_root)


def _actor_encoder(processor: Any, config: Any, build_root: Path) -> Callable[[dict], dict]:
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors

    actors = {actor.problem_id: actor for actor in read_actors(Path(build_root) / "train")}

    def encode(row: dict) -> dict:
        actor = actors[row["problem_id"]]
        if list(actor.image_paths) != row["image_paths"]:
            raise ValueError(f"train: row {row['row_id']} images differ from the vault")
        images = load_images(actor, Path(build_root) / "images")
        try:
            return encode_row(processor, config, row, images)
        finally:
            for image in images:
                image.close()
    return encode


def _admission_inputs(args: argparse.Namespace) -> dict:
    if args.max_row_tokens is not None and (args.precision != "nf4" or type(args.max_row_tokens) is not int
                                           or not 0 < args.max_row_tokens <= CONTEXT_LIMIT):
        raise ValueError("train v3: --max-row-tokens is a predeclared NF4 functional-check limit only")
    if args.audit is None or args.sidecar is None or getattr(args, "solution_audit", None) is None:
        raise ValueError("train v3: --audit, --sidecar and --solution-audit are required")
    return {"hashes": {name: ckpt.file_hash(getattr(args, name))
                       for name in ("rows", "order", "audit", "sidecar", "solution_audit")},
            "code_sha256": hashlib.sha256(args.code_commit.encode()).hexdigest(),
            "precision": args.precision, "limit": args.limit,
            "encoding": {"version": ADMISSION_VERSION_V3, "image": dict(IMAGE), "context_limit": CONTEXT_LIMIT,
                         "max_new_tokens": MAX_NEW_TOKENS, "end_token": END_TOKEN,
                         "max_row_tokens": args.max_row_tokens}}


def _encoded_record(row: dict) -> dict:
    length, target = row["input_ids"].numel(), row["target_len"]
    if type(target) is not int or not 0 < target < length:
        raise ValueError("train v3: a nonempty prompt and target are required")
    return {"row_id": row["row_id"], "kind": row["kind"], "route": row["route"], "tokens": length,
            "target_tokens": target, "encoded_sha256": _digest({key: {"sha256": tensor_sha256(row[key]),
                "shape": list(row[key].shape), "dtype": str(row[key].dtype)}
                for key in PROBE_TENSORS if row.get(key) is not None})}


def run_admit(args: argparse.Namespace, *, encode: Callable[[dict], dict] | None = None,
              map_fn: Callable[..., Iterable] = map) -> int:
    """CPU-only processor preflight for both arms. No model weights or CUDA initialization are needed."""
    if args.output.exists():
        raise ValueError(f"admit: refusing to overwrite {args.output}")
    inputs = _admission_inputs(args)
    groups = _read_groups(args.rows, args.order, args.limit, "note", PER_UPDATE_ITEMS_V2, v3=True,
                          audit_path=args.audit, solution_audit_path=args.solution_audit)
    if encode is None:
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True,
            min_pixels=IMAGE["min_pixels"], max_pixels=IMAGE["max_pixels"])
        processor.image_processor.size = {"shortest_edge": IMAGE["min_pixels"], "longest_edge": IMAGE["max_pixels"]}
        encode = _actor_encoder(processor, AutoConfig.from_pretrained(str(args.model_path), local_files_only=True),
                                args.build_root)
    admitted: list[dict] = []
    for index, (encoded, drops) in enumerate(encoded_groups(groups, encode, args.max_row_tokens or CONTEXT_LIMIT, map_fn)):
        records = [_encoded_record(row) for row in encoded]
        kinds = Counter(row["kind"] for row in records)
        active = kinds[KIND_SOLUTION] > 0
        admitted.append({"group": index, "active": active, "nA": kinds[KIND_SOLUTION], "nB": kinds[KIND_NOTE],
                         "rows": records, "drops": drops, "inactive_reason": None if active else "NO_ENCODED_A"})
    body = {"version": ADMISSION_VERSION_V3, "inputs": inputs, "groups": admitted,
            "total_steps": sum(group["active"] for group in admitted)}
    # Recheck source bytes after encoding, so a concurrent edit cannot receive the earlier identity.
    if _admission_inputs(args) != inputs:
        raise ValueError("admit: inputs changed during encoding")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, body | {"sha256": _digest(body)})
    return 0 if body["total_steps"] else 1


def _read_admission(args: argparse.Namespace, groups: list[list[dict]]) -> dict:
    if args.admission is None:
        raise ValueError("train v3: --admission is required")
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    if (admission.get("version") != ADMISSION_VERSION_V3
            or admission.get("sha256") != _digest({k: v for k, v in admission.items() if k != "sha256"})
            or admission.get("inputs") != _admission_inputs(args) or len(admission.get("groups", [])) != len(groups)):
        raise ValueError("train v3: admission identity differs from rows/order/audit/sidecar/solution-audit/code/precision/limit")
    for index, (raw, group) in enumerate(zip(groups, admission["groups"], strict=True)):
        kept = [row["row_id"] for row in group["rows"]]
        dropped = [row["row_id"] for row in group["drops"]]
        raw_ids = [row["row_id"] for row in raw]
        kinds = Counter(row["kind"] for row in group["rows"])
        active = kinds[KIND_SOLUTION] > 0
        if (group["group"] != index or len(kept + dropped) != len(set(kept + dropped))
                or set(kept + dropped) != set(raw_ids) or kept != [rid for rid in raw_ids if rid in kept]
                or group["active"] is not active or group["nA"] != kinds[KIND_SOLUTION]
                or group["nB"] != kinds[KIND_NOTE]
                or group["inactive_reason"] != (None if active else "NO_ENCODED_A")):
            raise ValueError("train v3: admission group membership, order or A/B counts differ")
    if type(admission["total_steps"]) is not int or admission["total_steps"] != sum(g["active"] for g in admission["groups"]):
        raise ValueError("train v3: admission total_steps differs from active groups")
    if admission["total_steps"] < 1:
        raise ValueError("train v3: admission has no A-present update groups")
    return admission


def _verify_admitted_group(admitted: dict, encoded: list[dict], drops: list[dict]) -> None:
    if [_encoded_record(row) for row in encoded] != admitted["rows"] or drops != admitted["drops"]:
        raise AdmissionChanged(f"train v3: group {admitted['group']} changed after admission (drop or encoded input)")


def _trainable_hashes(bundle: PolicyBundle) -> dict[str, str]:
    parameters = dict(bundle.model.named_parameters())
    return {row["name"]: tensor_sha256(parameters[row["name"]]) for row in bundle.inventory["trainable"]}


def _all_parameter_hashes(bundle: PolicyBundle) -> dict:
    return {name: {"sha256": tensor_sha256(parameter), "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in bundle.model.named_parameters()}


def _gradient_norms(bundle: PolicyBundle) -> dict[str, float | None]:
    norms = {}
    for group, parameters in bundle.params.items():
        values = [float(torch.linalg.vector_norm(parameter.grad.detach().float()))
                  for parameter in parameters if parameter.grad is not None]
        if any(not math.isfinite(value) for value in values):
            raise NonFiniteStop(f"nonfinite dry-run gradient in group {group}")
        norms[group] = math.sqrt(sum(value * value for value in values)) if values else None
    return norms


def run_dry_run(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0",
                make_encoder: Callable[[PolicyBundle, Path], Callable[[dict], dict]] = _vault_encoder,
                map_fn: Callable[..., Iterable] = map) -> int:
    """Small predeclared prefix: encode and backpropagate A/B separately, with no optimizer or weight save."""
    from .collect_v3 import source_identity

    if not 1 <= args.limit <= 10:
        raise ValueError("dry-run: limit must be a predeclared prefix of 1..10 items")
    out = args.output
    if (out / "dry_run.json").exists():
        raise ValueError("dry-run: refusing to overwrite an existing diagnostic")
    out.mkdir(parents=True, exist_ok=True)
    inputs = _admission_inputs(args)
    report: dict[str, Any] = {"status": "FAIL", "optimizer_steps": 0, "scheduler_steps": 0, "weights_saved": False,
        "scope": "diagnostic only: finite A/B backward with unchanged weights; no training or learning claim",
        "weights_unchanged": None, "precision": args.precision, "limit": args.limit,
        "parameter_hashes_before": None, "parameter_hashes_after": None,
        "frozen_base_verified": None, "gradients_cleared": None,
        "rows_by_target_source": {}, "paired_resolve_note_problem_ids": [], "model_provenance": None,
        "source_sha256": source_identity(), "inputs": inputs, "components": {}}
    groups = _read_groups(args.rows, args.order, args.limit, "note", PER_UPDATE_ITEMS_V2,
                          v3=True, audit_path=args.audit, solution_audit_path=args.solution_audit)
    report["rows_by_kind"] = dict(Counter(row["kind"] for group in groups for row in group))
    if not any(row["kind"] == KIND_SOLUTION for group in groups for row in group):
        report.update(status="NO_USABLE_SUPERVISION", reason="NO_A_ROWS")
        _write_json(out / "dry_run.json", report)
        return 1
    try:
        admission = _read_admission(args, groups)
    except ValueError as exc:
        if "no A-present update groups" not in str(exc):
            raise
        report.update(status="NO_USABLE_SUPERVISION", reason="NO_ENCODED_A")
        _write_json(out / "dry_run.json", report)
        return 1
    if not any(group["nB"] for group in admission["groups"] if group["active"]):
        report.update(status="NO_USABLE_SUPERVISION", reason="NO_ADMITTED_B")
        _write_json(out / "dry_run.json", report)
        return 1
    contracts = admission["inputs"]["hashes"] | {"admission": ckpt.file_hash(args.admission)}
    protocol = plain_protocol_v3(args.precision, admission["total_steps"], arm="note", contracts=contracts)
    report.update(protocol_sha256=protocol["sha256"], admission_sha256=contracts["admission"])
    assert_base = assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged
    bundle: PolicyBundle | None = None
    before: dict | None = None
    initial_base_verified = False
    phase = "policy_initialization"

    def fail(stage: str, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        report.update(status="FAIL")
        report.setdefault("error", message)
        report.setdefault("errors", []).append({"phase": stage, "error": message})

    try:
        torch.manual_seed(MASTER_SEED)
        bundle = (policy_factory or build_policy)(args.model_path, ARM, protocol)
        if args.precision == "bf16":  # CPU test fixtures; the public dry-run CLI accepts NF4 only.
            bundle.model.to(device)
        quantization = bundle.inventory.get("quantization")
        report["model_provenance"] = {key: bundle.inventory[key] for key in
            ("base_id", "base_revision", "arm", "trainable_counts") if key in bundle.inventory}
        if quantization is not None:
            report["model_provenance"]["quantization"] = {
                key: quantization[key] for key in ("contract", "versions", "reference")}
            report["model_provenance"]["quantization"].update(
                module_count=len(quantization["modules"]), modules_sha256=_digest(quantization["modules"]),
                state_sha256=_digest(quantization["state_sha256"]))
        phase = "initial_frozen_base"
        assert_base(bundle)
        initial_base_verified = True
        phase = "initial_parameter_hashes"
        before = _all_parameter_hashes(bundle)
        report["parameter_hashes_before"] = before
        phase = "encoding"
        encoded = []
        encode = make_encoder(bundle, args.build_root)
        for admitted, (kept, drops) in zip(admission["groups"], encoded_groups(
                groups, encode, args.max_row_tokens or CONTEXT_LIMIT, map_fn), strict=True):
            _verify_admitted_group(admitted, kept, drops)
            if admitted["active"]:
                encoded.extend(kept)
        report["encoded_rows_by_kind"] = dict(Counter(row["kind"] for row in encoded))
        raw_by_id = {row["row_id"]: row for group in groups for row in group}
        admitted_rows = [raw_by_id[row["row_id"]] for row in encoded]
        report["rows_by_target_source"] = dict(Counter(row["target_source"] for row in admitted_rows))
        report["paired_resolve_note_problem_ids"] = sorted(
            {row["problem_id"] for row in admitted_rows if row["kind"] == KIND_SOLUTION and row["target_source"] == "RESOLVE"}
            & {row["problem_id"] for row in admitted_rows if row["kind"] == KIND_NOTE and row["target_source"] == "NOTE"})
        report["loss_mask"] = {row["row_id"]: {"input_tokens": row["input_ids"].numel(),
            "prompt_tokens_excluded": row["input_ids"].numel() - row["target_len"],
            "target_and_eos_tokens": row["target_len"]} for row in encoded}
        for kind, weight in ((KIND_SOLUTION, .9), (KIND_NOTE, .1)):
            phase = f"backward_{kind}"
            bundle.model.zero_grad(set_to_none=True)
            rows = [row for row in encoded if row["kind"] == kind]
            plan = plan_microbatches([row["input_ids"].numel() for row in rows],
                                     [row["route"] for row in rows], args.token_budget)
            losses = _backward_rows(bundle, rows, plan, [weight / len(rows)] * len(rows))
            mean_loss = losses[kind] / len(rows)
            if not math.isfinite(mean_loss):
                raise NonFiniteStop(f"nonfinite dry-run {kind} loss")
            norms = _gradient_norms(bundle)
            full = any(row["route"] == ROUTE_FULL for row in rows)
            expected = {"L": "finite_positive", "M": "finite_positive",
                        "V": "finite_positive" if full else "none"}
            route_errors = []
            for group, requirement in expected.items():
                value = norms.get(group)
                if requirement == "none":
                    if value is not None:
                        route_errors.append(f"UNEXPECTED_GRADIENT:{group}")
                elif value is None or not math.isfinite(value) or value <= 0:
                    route_errors.append(f"MISSING_OR_NONPOSITIVE_GRADIENT:{group}")
            report["components"][kind] = {"rows": len(rows), "row_nll_mean": mean_loss,
                "coefficient": weight, "weighted_loss": weight * mean_loss, "grad_norms": norms,
                "nonzero_gradient": any(value is not None and value > 0 for value in norms.values()),
                "expected_gradient_groups": expected, "route_check_passed": not route_errors,
                "route_check_reason": ";".join(route_errors) if route_errors else None,
                "routes": dict(Counter(row["route"] for row in rows)), "microbatches": len(plan)}
            bundle.model.zero_grad(set_to_none=True)
        report["status"] = "PASS" if all(row["route_check_passed"] for row in report["components"].values()) else "GRADIENT_ROUTE_MISMATCH"
    except Exception as exc:
        fail(phase, exc)
    finally:
        if bundle is not None:
            try:
                phase = "clear_gradients"
                bundle.model.zero_grad(set_to_none=True)
                report["gradients_cleared"] = all(parameter.grad is None for parameter in bundle.model.parameters())
                phase = "restore_route"
                bundle.route.apply(True)
                phase = "final_parameter_hashes"
                after = _all_parameter_hashes(bundle)
                report.update(parameter_hashes_after=after, weights_unchanged=before == after if before is not None else None)
                phase = "final_frozen_base"
                assert_base(bundle)
                report["frozen_base_verified"] = True if initial_base_verified else None
            except Exception as exc:
                fail(phase, exc)
        try:
            if (report["weights_unchanged"] is False or report["gradients_cleared"] is False
                    or source_identity() != report["source_sha256"]
                    or _admission_inputs(args) != inputs):
                raise ValueError("dry-run parameters, gradients, inputs or source changed")
        except Exception as exc:
            fail("final_input_binding", exc)
        _write_json(out / "dry_run.json", report)
    return 0 if report["status"] == "PASS" else 1


def _tensor_counts(bundle: PolicyBundle) -> dict[str, int]:
    counts = Counter(row["group"] for row in bundle.inventory["trainable"])
    return {group: counts[group] for group in ("L", "M", "V")}


def _changed_counts(bundle: PolicyBundle, initial: dict[str, str]) -> dict[str, int]:
    current = _trainable_hashes(bundle)
    changed: Counter[str] = Counter()
    for row in bundle.inventory["trainable"]:
        changed[row["group"]] += current[row["name"]] != initial[row["name"]]
    return {group: changed[group] for group in ("L", "M", "V")}


def _save(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: Any, step: int, position: int, out: Path,
          protocol: dict, binding: dict) -> None:
    """Whole directory or nothing: checkpoint.save refuses an existing path, and a crash must not leave a half one to resume from."""
    bundle.route.apply(True)
    tmp = out.with_name(out.name + ".tmp")
    for stale in (tmp, out):  # a retried resume recomputes checkpoint-final; numbered ones never repeat
        if stale.exists():
            shutil.rmtree(stale)
    ckpt.save(bundle, optimizer, scheduler, step, {"epoch": 0, "position": position, "discarded_problem_ids": []},
              ckpt.capture_rng(), tmp, protocol=protocol, binding=binding)
    os.replace(tmp, out)


def _newest_checkpoint(out: Path) -> Path | None:
    numbered = [(int(match[1]), path) for path in out.iterdir() if (match := re.fullmatch(r"checkpoint-(\d+)", path.name))]
    return max(numbered)[1] if numbered else None


def _greedy_ids(bundle: PolicyBundle, row: dict) -> list[int]:
    from transformers import GenerationConfig

    base = bundle.model.get_base_model()
    device = next(base.parameters()).device
    prompt = row["input_ids"][:-row["target_len"]].unsqueeze(0).to(device)
    defaults = base.generation_config
    config = GenerationConfig(do_sample=False, max_new_tokens=PROBE_NEW_TOKENS, use_cache=True,
                              eos_token_id=defaults.eos_token_id, pad_token_id=defaults.pad_token_id)
    with _autocast(device):
        output = base.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), **_image_inputs([row], device),
                               generation_config=config, use_model_defaults=False)
    return output[0, prompt.shape[1]:].tolist()


def compute_probe(bundle: PolicyBundle, rows: list[dict]) -> dict:
    """Per-token log probabilities one row at a time in eval mode, plus greedy ids of the first row."""
    bundle.route.apply(True)
    bundle.model.eval()
    with torch.no_grad():
        return {"logprobs": [batch_logprobs(bundle, [row])[0].cpu().tolist() for row in rows],
                "greedy_ids": _greedy_ids(bundle, rows[0])}


def _write_probe(bundle: PolicyBundle, rows: list[dict], out: Path) -> None:
    # The reload command takes no rows or images, so the exact probe inputs stay beside the probe.
    save_file({f"{key}.{index}": row[key].contiguous().clone() for index, row in enumerate(rows)
               for key in PROBE_TENSORS if row.get(key) is not None}, str(out / "probe_inputs.safetensors"))
    _write_json(out / "probe.json", {"row_ids": [row["row_id"] for row in rows],
                                     "target_lens": [row["target_len"] for row in rows], **compute_probe(bundle, rows)})


def _read_probe_rows(run: Path, probe: dict) -> list[dict]:
    tensors = load_file(str(run / "probe_inputs.safetensors"))
    return [{"row_id": row_id, "target_len": target_len, **{key: tensors.get(f"{key}.{index}") for key in PROBE_TENSORS}}
            for index, (row_id, target_len) in enumerate(zip(probe["row_ids"], probe["target_lens"], strict=True))]


def _summary(updates: list[dict]) -> dict:
    real = [row for row in updates if not row["skipped"]]
    seconds, tokens = sum(row["seconds"] for row in real), sum(row["tokens"] for row in real)
    totals: dict[str, Counter] = {"rows_by_kind": Counter(), "route_counts": Counter()}
    for row in real:
        totals["rows_by_kind"].update(row["rows_by_kind"])
        totals["route_counts"].update(row["routes"])
    return {"updates": len(real), "skipped_updates": len(updates) - len(real), "rows_used": sum(row["rows"] for row in real),
            "rows_by_kind": dict(totals["rows_by_kind"]), "route_counts": dict(totals["route_counts"]),
            "mean_loss_first_update": real[0]["loss"] if real else None, "mean_loss_last_update": real[-1]["loss"] if real else None,
            "seconds": seconds, "tokens_per_second": tokens / seconds if seconds else 0.0,
            "peak_allocated_gib": max((row["peak_allocated_gib"] for row in real), default=0.0)}


def _check_output(args: argparse.Namespace, binding: dict, run: dict) -> None:
    out = args.output
    if ((out / "result.json").exists() or (out / "binding.json").exists()) and not args.resume:
        raise SystemExit(f"train: {out} already holds a run; pass --resume or choose a new directory")
    if not args.resume:
        return
    if (out / "result.json").exists() and json.loads((out / "result.json").read_text())["status"] == STATUS_COMPLETE:
        raise SystemExit(f"train: {out} is already complete")
    if (out / "binding.json").exists() and (json.loads((out / "binding.json").read_text()) != binding
                                            or json.loads((out / "initial_state.json").read_text())["run"] != run):
        raise SystemExit(f"train: --resume with different rows, commit, arm, precision, limit or order than {out}")


def run_train(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0",
              make_encoder: Callable[[PolicyBundle, Path], Callable[[dict], dict]] = _vault_encoder,
              map_fn: Callable[..., Iterable] = map) -> int:
    if args.gradient_checkpointing == "off" and args.precision == "nf4":
        raise SystemExit("train: --gradient-checkpointing off is bf16 only")
    out: Path = args.output
    version = getattr(args, "protocol_version", "v1")
    v2, v3 = version == "v2", version == "v3"
    if args.arm == "no-note" and not v3 or args.arm == "solution" and v3:
        raise ValueError("train: v3 uses note/no-note; solution is a legacy route-changing arm")
    groups = _read_groups(args.rows, args.order, args.limit, "note" if v3 else args.arm,
                          PER_UPDATE_ITEMS_V2 if v2 or v3 else PER_UPDATE_ITEMS, v3=v3,
                          audit_path=getattr(args, "audit", None),
                          solution_audit_path=getattr(args, "solution_audit", None))
    admission = _read_admission(args, groups) if v3 else None
    if admission is not None:
        contracts = admission["inputs"]["hashes"] | {"admission": ckpt.file_hash(args.admission)}
        protocol = plain_protocol_v3(args.precision, admission["total_steps"], arm=args.arm, contracts=contracts)
    else:
        protocol = plain_protocol_v2(args.precision, max(1, sum(1 for group in groups if group))) if v2 \
            else plain_protocol(args.precision)
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": ckpt.file_hash(args.rows), "code_sha256": hashlib.sha256(args.code_commit.encode()).hexdigest()}
    # token_budget and checkpointing are left out on purpose: they change memory use, not the gradient, so an OOM can resume smaller.
    run = {"arm": args.arm, "precision": args.precision, "limit": args.limit, "max_row_tokens": args.max_row_tokens,
           "order_sha256": ckpt.file_hash(args.order)}
    _check_output(args, binding, run)
    out.mkdir(parents=True, exist_ok=True)
    if admission is not None:
        saved_admission = out / "admission.json"
        bound_run = any((out / name).exists() for name in ("binding.json", "initial_state.json", "checkpoint-final")) \
            or _newest_checkpoint(out) is not None
        if args.resume and bound_run:
            if not saved_admission.exists() or ckpt.file_hash(saved_admission) != contracts["admission"]:
                raise ValueError("train v3: the run's saved admission is missing or changed")
        else:
            shutil.copyfile(args.admission, saved_admission)
    (out / "result.json").unlink(missing_ok=True)
    assert_base = assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged
    torch.manual_seed(MASTER_SEED)  # before building, so both arms start from the same adapter init
    newest = _newest_checkpoint(out) if args.resume else None
    step, position = 0, 0
    if newest is None:
        bundle = (policy_factory or build_policy)(args.model_path, ARM, protocol)
        if args.precision == "bf16":
            bundle.model.to(device)
        optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
        _write_json(out / "binding.json", binding)
        _write_json(out / "initial_state.json", {"run": run, "sha256": _trainable_hashes(bundle)})
    else:
        bundle, optimizer, scheduler, step, cursor = ckpt.load_resume(
            args.model_path, newest, protocol, binding=binding, device=device, policy_factory=policy_factory)
        position = cursor["position"]
        if admission is not None and (not 0 <= position <= len(groups)
                or step != sum(group["active"] for group in admission["groups"][:position])):
            raise ValueError("train v3: checkpoint cursor/step differs from the admitted update schedule")
    if args.gradient_checkpointing == "off":
        bundle.model.gradient_checkpointing_disable()
    updates_path, drops_path = out / "updates.jsonl", out / "drops_local.jsonl"
    for path in (updates_path, drops_path):
        _keep_before(path, position)
        path.touch()
    encode = make_encoder(bundle, args.build_root)
    stream = encoded_groups(groups[position:], encode, (args.max_row_tokens or CONTEXT_LIMIT) if v3 else args.max_row_tokens, map_fn)
    for index, (encoded, drops) in enumerate(stream, start=position):
        if admission is not None:
            try:
                _verify_admitted_group(admission["groups"][index], encoded, drops)
            except AdmissionChanged as exc:
                _write_json(out / "result.json", {"status": "ADMISSION_CHANGED", "group": index, "step": step,
                                                  "detail": str(exc)})
                return 1
            if not admission["groups"][index]["active"]:
                encoded = []
            elif args.arm == "no-note":
                encoded = [row for row in encoded if row["kind"] == KIND_SOLUTION]
        for drop in drops:
            _append_jsonl(drops_path, {"group": index, **drop})
        if not encoded:
            _append_jsonl(updates_path, {"step": step, "group": index, "rows": 0, "skipped": True,
                **({"reason": "NO_ENCODED_A", "nA": 0, "nB": admission["groups"][index]["nB"],
                    "inactive_row_ids": [row["row_id"] for row in admission["groups"][index]["rows"]]}
                   if admission is not None else {})})
            continue
        lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
        try:
            if v3:
                stats = run_update(bundle, optimizer, scheduler, encoded, args.token_budget,
                                   kind_weights={KIND_SOLUTION: 0.9, KIND_NOTE: 0.1 if args.arm == "note" else 0.0})
            else:
                stats = run_update(bundle, optimizer, scheduler, encoded, args.token_budget)
        except NonFiniteStop as exc:
            _write_json(out / "result.json", {"status": STATUS_NONFINITE, "arm": args.arm, "precision": args.precision,
                                              "group": index, "step": step, "detail": str(exc)})
            return 1
        step += 1
        _append_jsonl(updates_path, {"step": step, "group": index, **stats, **({"lr": lrs} if v2 or v3 else {}),
            **({"nA": admission["groups"][index]["nA"], "nB": admission["groups"][index]["nB"]}
               if admission is not None else {})})
        if args.save_every and step % args.save_every == 0:
            _save(bundle, optimizer, scheduler, step, index + 1, out / f"checkpoint-{step}", protocol, binding)
    summary = _summary(_read_jsonl(updates_path))
    if not step:
        _write_json(out / "result.json", {"status": STATUS_NO_UPDATES, "arm": args.arm, "precision": args.precision, **summary})
        return 1
    assert_base(bundle)
    if admission is not None and step != admission["total_steps"]:
        raise ValueError("train v3: optimizer steps differ from the admitted update schedule")
    initial = json.loads((out / "initial_state.json").read_text(encoding="utf-8"))["sha256"]
    changed = _changed_counts(bundle, initial)
    _save(bundle, optimizer, scheduler, step, len(groups), out / "checkpoint-final", protocol, binding)
    probe_rows: list[dict] = []
    probe_ids = ({row["row_id"]: row for group in admission["groups"] if group["active"] for row in group["rows"]
                  if args.arm == "note" or row["kind"] == KIND_SOLUTION} if admission is not None else None)
    for row in (row for group in groups for row in group if probe_ids is None or row["row_id"] in probe_ids):
        item = _encode_or_drop(row, encode, (args.max_row_tokens or CONTEXT_LIMIT) if v3 else args.max_row_tokens)
        if probe_ids is not None and ("input_ids" not in item or _encoded_record(item) != probe_ids[row["row_id"]]):
            raise AdmissionChanged("train v3: probe input changed after admission")
        probe_rows += [item] if "input_ids" in item else []
        if len(probe_rows) == PROBE_ROWS:
            break
    _write_probe(bundle, probe_rows, out)
    _write_json(out / "result.json", {
        "status": STATUS_COMPLETE, "arm": args.arm, "precision": args.precision, **summary, "changed_tensors": changed,
        "total_tensors": _tensor_counts(bundle), "gradient_checkpointing": args.gradient_checkpointing,
        "token_budget": args.token_budget, "checkpoint": str(out / "checkpoint-final"),
        **({"protocol_version": VERSION_V3 if v3 else VERSION_V2, "total_steps": protocol["optimizer"]["total_steps"]}
           if v2 or v3 else {})})
    return 0


def run_reload(args: argparse.Namespace, *, policy_factory: Callable | None = None, device: str = "cuda:0") -> int:
    """Fresh process: rebuild F0, load checkpoint-final and recompute the probe. Any disagreement is a FAIL with a nonzero exit."""
    report: dict[str, Any] = {"status": "FAIL", "step": None, "tensors": None, "max_abs_delta": None, "greedy_equal": None}
    try:
        final = args.run / "checkpoint-final"
        binding = json.loads((args.run / "binding.json").read_text(encoding="utf-8"))
        metadata = ckpt.read_metadata(final, binding=binding)
        recorded = metadata["protocol"]
        if getattr(args, "protocol_version", "v1") == "v3":
            contract = recorded.get("training_contract", {})
            protocol = plain_protocol_v3(args.precision, recorded.get("optimizer", {}).get("total_steps"),
                                         arm=contract.get("arm"), contracts=contract.get("hashes", {}))
            admission = json.loads((args.run / "admission.json").read_text(encoding="utf-8"))
            state = json.loads((args.run / "initial_state.json").read_text(encoding="utf-8"))["run"]
            hashes = contract["hashes"]
            if (ckpt.file_hash(args.run / "admission.json") != hashes["admission"]
                    or admission.get("version") != ADMISSION_VERSION_V3
                    or admission.get("sha256") != _digest({k: v for k, v in admission.items() if k != "sha256"})
                    or admission["inputs"]["hashes"] != {k: v for k, v in hashes.items() if k != "admission"}
                    or admission["inputs"]["code_sha256"] != binding["code_sha256"]
                    or admission["inputs"]["precision"] != args.precision
                    or binding["data_sha256"] != hashes["rows"] or state["arm"] != contract["arm"]
                    or state["precision"] != args.precision
                    or state["max_row_tokens"] != admission["inputs"]["encoding"]["max_row_tokens"]
                    or state["order_sha256"] != hashes["order"] or state["limit"] != admission["inputs"]["limit"]
                    or metadata["step"] != admission["total_steps"]
                    or metadata["cursor"]["position"] != len(admission["groups"])):
                raise ValueError("reload v3: admission, arm or common schedule binding differs")
        elif getattr(args, "protocol_version", "v1") == "v2":  # only the step count comes from the run; the rest is code
            protocol = plain_protocol_v2(args.precision, recorded.get("optimizer", {}).get("total_steps"))
        else:
            protocol = plain_protocol(args.precision)
        if recorded != json.loads(json.dumps(protocol)):
            raise ValueError(f"reload: the checkpoint protocol differs from the {args.precision} protocol")
        bundle, _, _, step, _ = ckpt.load_resume(args.model_path, final, protocol, binding=binding, device=device,
                                                 policy_factory=policy_factory)
        (assert_quantized_base_unchanged if args.precision == "nf4" else assert_base_unchanged)(bundle)
        report.update(step=step, tensors=_tensor_counts(bundle))
        probe = json.loads((args.run / "probe.json").read_text(encoding="utf-8"))
        actual = compute_probe(bundle, _read_probe_rows(args.run, probe))
        if [len(values) for values in actual["logprobs"]] != [len(values) for values in probe["logprobs"]]:
            raise ValueError("reload: probe token counts differ")
        delta = max((abs(a - b) for new, old in zip(actual["logprobs"], probe["logprobs"], strict=True)
                     for a, b in zip(new, old, strict=True)), default=0.0)
        report.update(max_abs_delta=delta, greedy_equal=actual["greedy_ids"] == probe["greedy_ids"])
        if delta <= PROBE_ATOL and report["greedy_equal"]:
            report["status"] = "PASS"
    except (ValueError, OSError, KeyError, NonFiniteStop) as exc:
        report["error"] = str(exc)
    _write_json(args.run / "reload.json", report)
    return 0 if report["status"] == "PASS" else 1


def _commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("a full 40-hex commit is required")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    for name in ("rows", "build-root", "order", "model-path", "output"):
        train.add_argument(f"--{name}", type=Path, required=True)
    train.add_argument("--limit", type=int, required=True)
    train.add_argument("--arm", choices=("note", "solution", "no-note"), required=True)
    train.add_argument("--precision", choices=("bf16", "nf4"), required=True)
    train.add_argument("--code-commit", type=_commit, required=True)
    train.add_argument("--token-budget", type=int, default=0)
    train.add_argument("--gradient-checkpointing", choices=("on", "off"), default="on")
    train.add_argument("--save-every", type=int, default=50)
    train.add_argument("--max-row-tokens", type=int, default=None, help="local 8GB only; never set on the rented GPU")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--protocol-version", choices=("v1", "v2", "v3"), default="v1")
    for name in ("admission", "audit", "sidecar", "solution-audit"):
        train.add_argument(f"--{name}", type=Path)
    admit = sub.add_parser("admit", help="CPU-only common v3 row encoding and update admission")
    for name in ("rows", "build-root", "order", "model-path", "output", "audit", "sidecar", "solution-audit"):
        admit.add_argument(f"--{name}", type=Path, required=True)
    admit.add_argument("--limit", type=int, required=True)
    admit.add_argument("--precision", choices=("bf16", "nf4"), required=True)
    admit.add_argument("--code-commit", type=_commit, required=True)
    admit.add_argument("--max-row-tokens", type=int, default=None, help="predeclared NF4 functional-check admission only")
    dry = sub.add_parser("dry-run", help="v3 forward/backward diagnostics only; no optimizer or weight saving")
    for name in ("rows", "build-root", "order", "model-path", "output", "audit", "sidecar", "solution-audit", "admission"):
        dry.add_argument(f"--{name}", type=Path, required=True)
    dry.add_argument("--limit", type=int, required=True, help="predeclared first 1..10 items")
    dry.add_argument("--precision", choices=("nf4",), required=True)
    dry.add_argument("--code-commit", type=_commit, required=True)
    dry.add_argument("--token-budget", type=int, default=0)
    dry.add_argument("--max-row-tokens", type=int, default=None, help="must equal the predeclared NF4 admission limit")
    reload = sub.add_parser("reload")
    reload.add_argument("--run", type=Path, required=True)
    reload.add_argument("--model-path", type=Path, required=True)
    reload.add_argument("--precision", choices=("bf16", "nf4"), required=True)
    reload.add_argument("--protocol-version", choices=("v1", "v2", "v3"), default="v1")
    return parser


def _factory(precision: str) -> Callable | None:
    return partial(build_quantized_policy, device_index=0) if precision == "nf4" else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reload":
        return run_reload(args, policy_factory=_factory(args.precision))
    from .collect import ResourceLog, bounded_map, install_training_guards

    install_training_guards(args.build_root)
    if args.command == "admit":
        return run_admit(args, map_fn=partial(bounded_map, workers=ENCODE_WORKERS, buffer=ENCODE_BUFFER))
    args.output.mkdir(parents=True, exist_ok=True)
    with ResourceLog(args.output / "resource.jsonl"):
        if args.command == "dry-run":
            return run_dry_run(args, policy_factory=_factory(args.precision),
                               map_fn=partial(bounded_map, workers=ENCODE_WORKERS, buffer=ENCODE_BUFFER))
        return run_train(args, policy_factory=_factory(args.precision),
                         map_fn=partial(bounded_map, workers=ENCODE_WORKERS, buffer=ENCODE_BUFFER))


if __name__ == "__main__":
    raise SystemExit(main())
