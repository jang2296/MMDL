"""Immutable preparation → delayed single-response RL campaign, with drained checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from . import single
from .audit import digest
from .collect import write_json
from .contract import MASTER_SEED
from .prep_v16b import (collect, eta_report, file_sha, grader, images, make_request, p0,
                         preflight_items, read_sealed, seal, summary, data_api)

CAMPAIGN_V16B = "single-campaign-v16b"
CAMPAIGN_V17 = "single-campaign-v17"
CAMPAIGN_V18 = "single-campaign-v18"
CAMPAIGN_V18B = "single-campaign-v18b"
MODERN_VERSIONS = {"error-note-v17", "error-note-v18", "error-note-v18b"}
CAMPAIGN_VERSIONS = {"error-note-v18b": CAMPAIGN_V18B, "error-note-v18": CAMPAIGN_V18,
                     "error-note-v17": CAMPAIGN_V17, "error-note-v16b": CAMPAIGN_V16B}
# single_vllm.submit raises above this measured prompt capacity bound; NOTE admission skips instead.
NOTE_PROMPT_BOUND = 16384


def prompt_tokens(bundle: Any, request: dict, image: dict) -> int:
    """Prompt length measured as the sampler measures it at admission."""
    from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

    # ponytail: re-processes the image once per NOTE (~0.1-0.5 s); a size-only bound would avoid it.
    return prepare_protocol_inputs(bundle.processor, bundle.model.config, request["messages"],
                                   request["images"], image)[0]["input_tokens"]


def _stop_markers(output: Path) -> list[str]:
    return [marker for marker in ("STOP_REQUESTED", "WATCHDOG_STOP")
            if (output / marker).exists() or (output / "runtime" / marker).exists()]


def descriptor(path: Path) -> dict:
    from .v18b_recovery import original_recovery_path, resolve_recovery_path
    return {"path": str(original_recovery_path(path.absolute())), "file_sha256": file_sha(resolve_recovery_path(path))}


def checked(record: dict) -> Path:
    from .v18b_recovery import resolve_recovery_path
    path = resolve_recovery_path(record["path"])
    if path.is_symlink() or file_sha(path) != record["file_sha256"]:
        raise ValueError("v16b campaign: changed bound artifact")
    return path


def _binding(plan: dict) -> dict:
    return {"protocol_sha256": plan["protocol"]["sha256"],
        "prompts_sha256": plan["protocol"]["prompts_sha256"], "data_sha256": plan["sha256"],
        "code_sha256": digest(plan["source"]["actual_files"])}


def _plan_check(plan: dict) -> None:
    from .protocol import validate_protocol_v16b

    if plan.get("protocol", {}).get("training_contract", {}).get("rehearsal_profile") is not None:
        from .v18b_rehearsal import validate_plan
        validate_plan(plan)
        return
    is_v17 = plan["protocol"].get("version") in MODERN_VERSIONS
    if is_v17:
        from .protocol import validate_protocol_v17
        if plan["protocol"]["version"] == "error-note-v18b":
            from .protocol import validate_protocol_v18b
            validate_protocol_v18b(plan["protocol"])
        elif plan["protocol"]["version"] == "error-note-v18":
            from .protocol import validate_protocol_v18
            validate_protocol_v18(plan["protocol"])
        else:
            validate_protocol_v17(plan["protocol"])
    else:
        validate_protocol_v16b(plan["protocol"])
    cfg = plan["protocol"]["training_contract"]
    smoke = cfg["local_smoke"]
    if plan["protocol"]["version"] == "error-note-v18b":
        from .v18b_execution import validate_settings
        if plan.get("execution_settings") is not None:
            validate_settings(plan["execution_settings"])
        elif not smoke:
            raise ValueError("v18b production needs immutable execution decisions")
        recovery = plan.get("recovery")
        on_prem = (plan.get("execution_settings") or {}).get("execution_mode") == "on_prem"
        if not smoke and not on_prem and isinstance(recovery, dict) and recovery.get("transfer_command") is None:
            raise ValueError("v18b production requires a verified off-pod receiver argv")
        if (not isinstance(recovery, dict) or recovery.get("every_main_slots") != 16
                or not isinstance(recovery.get("destination"), str) or not Path(recovery["destination"]).is_absolute()
                or recovery.get("transfer_command") is not None and (not isinstance(recovery["transfer_command"], list)
                    or not recovery["transfer_command"] or any(not isinstance(part, str) or not part
                                                               for part in recovery["transfer_command"]))):
            raise ValueError("v18b campaign needs an immutable boundary-copy contract")
    ids = [pid for block in plan["blocks"] for pid in block]
    expected = [item["problem_id"] for item in plan["items"]]
    if (plan.get("version") != CAMPAIGN_VERSIONS.get(plan["protocol"]["version"])
            or plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"})
            or ids != expected or len(ids) != len(set(ids)) or len(ids) != cfg["rl_items"]
            or len(plan["blocks"]) != cfg["rl_blocks"]
            or any(len(block) != (2 if smoke else 16) for block in plan["blocks"][:-1])
            or len(plan["blocks"][-1]) != (2 if smoke else 8)
            or len(plan["prep_batches"]) != cfg["prep_updates"]
            or plan["total_slots"] != cfg["total_slots"]
            or plan.get("time_cut") is not None or plan["local_sequential"] != smoke
            or plan["capacity"]["engine"] != ("hf" if smoke else "vllm")
            or plan["capacity"]["max_num_seqs"] != (1 if smoke else 16 if is_v17 else 32)):
        raise ValueError("v16b campaign: invalid immutable preparation/RL/flush plan")


def prepare(args: argparse.Namespace) -> dict:
    version = getattr(args, "version", "error-note-v16b")
    from .v18b_recovery import original_recovery_path
    from .v18b_execution import read_settings
    settings_path = getattr(args, "execution_settings", None)
    settings = read_settings(settings_path) if settings_path is not None else None
    backup_destination = getattr(args, "backup_destination", None)
    transfer = getattr(args, "backup_transfer_command", None)
    transfer = json.loads(transfer) if isinstance(transfer, str) else transfer
    if transfer is not None and (not isinstance(transfer, list) or not transfer
                                or any(not isinstance(part, str) or not part for part in transfer)):
        raise ValueError("v18b recovery: transfer command must be a nonempty JSON argv list")
    data = data_api(version)
    load_items, validate = data.load_items, data.validate
    from .protocol import SINGLE_IMAGE_V14, SMOKE_IMAGE_V16B, plain_protocol_v16b
    from .prep_v16b import select

    is_v17 = version in MODERN_VERSIONS
    if is_v17:
        from .protocol import plain_protocol_v17
        plain_protocol_v16b = plain_protocol_v17
        if version == "error-note-v18b":
            from .protocol import plain_protocol_v18b as plain_protocol_v16b
        elif version == "error-note-v18":
            from .protocol import plain_protocol_v18
            plain_protocol_v16b = plain_protocol_v18
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "campaign-plan.json"
    if path.exists():
        plan = read_sealed(path)
        _plan_check(plan)
        if (plan["data_root"] != str(original_recovery_path(args.data_root.absolute()))
                or plan["prep_root"] != str(original_recovery_path(args.prep_root.absolute()))
                or plan["model_path"] != str(original_recovery_path(args.model_path.absolute()))
                or plan["protocol"]["version"] != version
                or plan["local_sequential"] != args.local_sequential
                or plan["protocol"]["precision"] != args.precision or plan["capacity"]["engine"] != args.engine):
            raise ValueError("v16b campaign: immutable execution arguments changed")
        if version == "error-note-v18b" and (
                settings is not None and settings != plan.get("execution_settings")
                or backup_destination is not None and str(original_recovery_path(backup_destination.absolute()))
                    != plan.get("recovery", {}).get("destination")
                or transfer is not None and transfer != plan.get("recovery", {}).get("transfer_command")):
            raise ValueError("v18b campaign: immutable execution decisions or backup contract changed")
        return plan
    on_prem = (settings or {}).get("execution_mode") == "on_prem"
    if version == "error-note-v18b" and not args.local_sequential and (settings is None or backup_destination is None or transfer is None and not on_prem):
        raise ValueError("v18b production requires --execution-settings, --backup-destination and verified off-pod --backup-transfer-command")
    validate(args.data_root)
    selection = select(args.data_root, args.prep_root, local_sequential=args.local_sequential, version=version)
    prep = read_sealed(args.prep_root / "prep-rows.json")
    prior = read_sealed(args.prep_root / "baseline-init.json")
    generated = read_sealed(args.prep_root / "f0-generations.json")
    if (prep["selection_sha256"] != selection["sha256"]
            or prior["prep_rows_sha256"] != prep["sha256"]
            or prep["f0_generations_sha256"] != generated["sha256"]
            or prep["local_sequential"] != args.local_sequential):
        raise ValueError("v16b campaign: preparation was not sealed under this data selection")
    source = single.source_manifest()
    items = load_items(args.data_root, "train")
    if version == "error-note-v18b":
        from .v18b_rows import bind_graph_observation_targets_v18b
        items = bind_graph_observation_targets_v18b(items)
    preflight = seal(args.output / "data-contract-preflight.json",
        preflight_items(items, version=version) if is_v17 else preflight_items(items))
    observation_eligibility = {row["problem_id"]: row["observation_eligibility"] for row in preflight["items"]}
    if args.local_sequential and version == "error-note-v18b":
        # Fixed before inference. Archived diagnostic parents are disjoint from the live smoke.
        diagnostic_suffixes = (":00c977656045ae4d", ":7b15b6cffe093171", ":362321e203f6d8f3", ":786c3412cf22e7f0")
        targets = []
        for layer, count in (("graph", 6), ("scene", 2)):
            matching = [i["problem_id"] for i in items if i.get("layer") == layer
                        and not i["problem_id"].endswith(diagnostic_suffixes)
                        and observation_eligibility[i["problem_id"]]["eligible"]]
            if len(matching) < count:
                raise ValueError(f"v18b smoke requires {count} eligible {layer} parents")
            targets.extend(matching[:count])
        items = [i for i in items if i["problem_id"] in targets]
    elif args.local_sequential:
        selected = []
        repair = {r["problem_id"] for r in preflight["items"] if r.get("repair_supported")}
        for layer in (("geometry", "graph", "scene", "chart") if is_v17 else ("geometry", "graph", "scene")):
            member = next((item for item in items if item.get("layer") == layer
                           and (observation_eligibility[item["problem_id"]]["eligible"]
                                or is_v17 and item["problem_id"] in repair)), None)
            if member is None:
                if is_v17:
                    continue
                raise ValueError("v16b smoke: a supported parent from each observation layer is required")
            selected.append(member["problem_id"])
        selected += [item["problem_id"] for item in items if item["problem_id"] not in selected][:8 - len(selected)]
        items = [item for item in items if item["problem_id"] in selected]
    size = 2 if args.local_sequential else 16
    blocks = [[item["problem_id"] for item in items[i:i + size]] for i in range(0, len(items), size)]
    inputs = {name: descriptor(path) for name, path in {
        "payload": args.data_root / "payload-receipt.json", "order": args.data_root / "order.json",
        "prep_rows": args.prep_root / "prep-rows.json", "baseline": args.prep_root / "baseline-init.json",
        "f0": args.prep_root / "f0-generations.json", "selection": args.prep_root / "selection.json",
        "data_contract_preflight": args.output / "data-contract-preflight.json"}.items()}
    contracts = {"source": digest(source["actual_files"]), "dataset": inputs["payload"]["file_sha256"],
        "order": inputs["order"]["file_sha256"], "prep_rows": inputs["prep_rows"]["file_sha256"],
        "monitor": digest(generated["monitor"])}
    protocol = plain_protocol_v16b(args.precision, rl_blocks=len(blocks), prep_updates=prep["prep_updates"],
        contracts=contracts, image=SMOKE_IMAGE_V16B if args.local_sequential else SINGLE_IMAGE_V14,
        local_smoke=args.local_sequential)
    body = {"version": CAMPAIGN_VERSIONS[version], "protocol": protocol, "source": source, "inputs": inputs,
        "data_root": str(args.data_root.absolute()), "prep_root": str(args.prep_root.absolute()),
        "model_path": str(args.model_path.absolute()), "nf4_cache": str(args.nf4_cache.absolute()) if args.nf4_cache else None,
        "local_sequential": args.local_sequential, "items": items, "blocks": blocks, "block_size": size,
        "observation_eligibility": {item["problem_id"]: observation_eligibility[item["problem_id"]] for item in items},
        "prep_batches": prep["batches"], "total_slots": protocol["training_contract"]["total_slots"],
        "time_cut": None, "capacity": {"engine": args.engine, "max_num_seqs": 1 if args.local_sequential else 16 if is_v17 else 32,
            "kv_cache_gib": args.kv_cache_gib, "gpu_memory_utilization": args.gpu_memory_utilization,
            "token_budget": args.token_budget}, "auto_evaluate": False}
    if version == "error-note-v18b":
        backup_destination = backup_destination or args.output.parent / ("recovery-copies-" + args.output.name)
        body.update(execution_settings=settings, recovery={
            "destination": str(backup_destination.absolute()), "transfer_command": transfer, "every_main_slots": 16})
        if settings_path is not None:
            inputs["execution_settings"] = descriptor(settings_path)
    if is_v17:
        from .guard_v16b import SignalGateV17, SignalGateV18B
        eligible = {r["problem_id"] for r in preflight["items"] if r["repair_supported"]}
        supported = [item["problem_id"] for item in items if item["problem_id"] in eligible]
        gate = (SignalGateV18B if version == "error-note-v18b" else SignalGateV17)(
            blocks, supported, local_smoke=args.local_sequential)
        body["signal_gate"] = {"supported_parent_ids": supported, "parents": gate.parents, "due_slot": gate.due_slot}
    plan = body | {"sha256": digest(body)}
    _plan_check(plan)
    single._sealed(path, plan)
    return plan


def _factory(args: argparse.Namespace):
    from fine_tuning.notebook_rl.policy import build_policy
    from fine_tuning.notebook_rl.quantized_policy import load_quantized_policy

    if getattr(args, "rehearsal_profile", None) is not None:
        from .v18b_rehearsal import policy_factory
        if args.precision != "bf16" or args.engine != "vllm" or args.local_sequential:
            raise ValueError("rehearsal requires BF16 and the actual vLLM engine")
        return policy_factory
    if args.precision == "nf4":
        if not args.local_sequential or args.nf4_cache is None or args.engine != "hf":
            raise ValueError("v16b: NF4 requires the explicit local sequential path and an existing cache")
        return lambda model, arm, protocol: load_quantized_policy(args.nf4_cache, arm, protocol)
    if args.local_sequential or args.engine != "vllm":
        raise ValueError("v16b: production BF16 requires the persistent vLLM path")
    return build_policy


def load_f0(args: argparse.Namespace, *, image: dict) -> tuple[Any, dict]:
    from .protocol import ARM, V16B_CONTRACT_HASHES, plain_protocol_v16b

    if getattr(args, "version", "error-note-v16b") in MODERN_VERSIONS:
        from .protocol import plain_protocol_v17
        plain_protocol_v16b = plain_protocol_v17
        if args.version == "error-note-v18b":
            from .protocol import plain_protocol_v18b as plain_protocol_v16b
        elif args.version == "error-note-v18":
            from .protocol import plain_protocol_v18
            plain_protocol_v16b = plain_protocol_v18
    cfg = plain_protocol_v16b(args.precision, rl_blocks=4 if args.local_sequential else 188,
        prep_updates=1 if args.local_sequential else 0,
        contracts={name: "0" * 64 for name in V16B_CONTRACT_HASHES}, image=image,
        local_smoke=args.local_sequential)
    bundle = _factory(args)(args.model_path, ARM, cfg)
    if args.precision == "bf16":
        bundle.model.to("cuda:0")
    return bundle, cfg


def calibrate_capacity(bundle: Any, args: argparse.Namespace, protocol: dict, *, live_engine: bool = False) -> dict:
    """Backpropagate a worst-context synthetic specimen without taking an optimizer step."""
    import torch
    from PIL import Image
    from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
    from fine_tuning.notebook_rl.prompts import user_message
    load_items = data_api(protocol["version"]).load_items
    from .train_v16b import measure_update_memory_v16b
    if protocol["version"] in MODERN_VERSIONS:
        from .train_v17 import measure_update_memory_v17
        measure_update_memory_v16b = measure_update_memory_v17
        if protocol["version"] == "error-note-v18":
            from .train_v18 import measure_update_memory_v18
            measure_update_memory_v16b = measure_update_memory_v18

    items = [item for split in ("train", "prep", "monitor") for item in load_items(args.data_root, split)]
    image_count = max(len(item["image_paths"]) for item in items)
    if image_count < 1:
        raise ValueError("v16b capacity: the full vision path needs an image specimen")
    # This exact B2 maximum covers every admitted image; these pixels are used only for memory calibration.
    pictures = [Image.new("RGB", (1792, 2240)) for _ in range(image_count)]
    prepared, _ = prepare_protocol_inputs(bundle.processor, bundle.model.config,
        [user_message("Question: Inspect the supplied calibration image.", image_count)], pictures, protocol["image"])
    token = bundle.processor.tokenizer.encode(" observation", add_special_tokens=False)[0]
    inputs = prepared["inputs"]
    row = {"input_ids": torch.cat([inputs["input_ids"][0], torch.tensor([token], dtype=torch.long)]),
        "target_len": 1, "pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"],
        "route": "FULL", "kind": "RL", "row_id": "synthetic-capacity-only"}
    shape_bounds: dict[str, Any] = {"max_images": image_count,
        "max_vision_patches": int(inputs["image_grid_thw"].prod(dim=1).sum().item()),
        "max_prompt_tokens": 24576 - 8192, "dataset_sha256": file_sha(args.data_root / "payload-receipt.json")}
    is_v18b = protocol["version"] == "error-note-v18b"
    if is_v18b != live_engine:
        raise ValueError("v18b maximum-shape capacity must be measured with the actual resident engine")
    report = measure_update_memory_v16b(bundle, row, shape_bounds=shape_bounds,
                                      max_target_tokens=8192, max_input_tokens=shape_bounds["max_prompt_tokens"],
                                      **({"live_engine": True} if live_engine else {}))
    path = args.output / (f"capacity-live-{os.getpid()}.json" if live_engine else "capacity.json")
    return seal(path, {"version": protocol["version"].removeprefix("error-note-") + "-capacity-v1", "report": report,
        "image_count": image_count, "synthetic_image_size": [1792, 2240], "shape_bounds": shape_bounds,
        "synthetic_bound": True, "model_parameter_update": False, "source_sha256": digest(single.source_manifest()["actual_files"])})


def make_sampler(bundle: Any, args: argparse.Namespace, protocol: dict, *, image: dict, sampling: dict):
    if args.local_sequential:
        from .v16b_runtime import LocalSequentialSampler
        return LocalSequentialSampler(bundle, protocol_sha256=protocol["sha256"], image=image, sampling=sampling)
    import contextlib
    import torch
    from transformers import GenerationConfig
    from fine_tuning.notebook_rl.checkpoint import park_adam_moments
    from .single_vllm import SingleVllmSampler

    if bundle is None:
        raise ValueError("v16b capacity: the HF policy must be measured before engine creation")
    receipt_path = args.output / "capacity.json"
    if protocol["version"] == "error-note-v18b":
        kv_cache_gib = args.kv_cache_gib
    else:
        receipt = read_sealed(receipt_path) if receipt_path.exists() else calibrate_capacity(bundle, args, protocol)
        if (receipt["source_sha256"] != digest(single.source_manifest()["actual_files"])
                or receipt["shape_bounds"]["dataset_sha256"] != file_sha(args.data_root / "payload-receipt.json")):
            raise ValueError("v16b capacity: stale source/data measurement")
        kv_cache_gib = receipt["report"]["kv_cache_gib"]
    eos = GenerationConfig.from_pretrained(str(args.model_path), local_files_only=True).eos_token_id
    optimizer = getattr(args, "_optimizer", None)
    parked = (park_adam_moments(bundle, optimizer, args._scheduler, protocol, args._slot)
              if optimizer is not None else contextlib.nullcontext())
    with parked:
        bundle.model.to("cpu")
        torch.cuda.empty_cache()
        try:
            v17_options = ({"protocol_version": protocol["version"],
                "preflight_stage": getattr(args, "_preflight_stage", "f0")} if protocol["version"] in MODERN_VERSIONS else {})
            if protocol["version"] == "error-note-v18b" and getattr(args, "_f0_legacy_generation", False):
                v17_options["generation_context_limit"] = 24576
            max_sequences = 16 if protocol["version"] in MODERN_VERSIONS else 32
            rehearsal_profile = protocol["training_contract"].get("rehearsal_profile")
            if rehearsal_profile is not None:
                from .v18b_rehearsal import profile_settings
                max_sequences = profile_settings(rehearsal_profile)["max_num_seqs"]
                v17_options["rehearsal_profile"] = rehearsal_profile
            sampler = SingleVllmSampler(args.model_path, kv_cache_gib=kv_cache_gib,
                max_num_seqs=max_sequences,
                gpu_memory_utilization=args.gpu_memory_utilization, sleep=False,
                eos_token_ids=eos if isinstance(eos, list) else [eos], image=image,
                lora=True, sampling=sampling, version_registry=True, **v17_options)
        finally:
            bundle.model.to("cuda:0")
    return sampler


def _encode(bundle: Any, rows: list[dict], items: dict[str, dict], image: dict) -> list[dict]:
    from .train_v4 import encode_single_row
    return [encode_single_row(bundle.processor, bundle.model.config, row, images(items[row["problem_id"]]), image=image)
            for row in rows]


def _encode_update_rows(bundle: Any, rows: list[dict], items: dict[str, dict], image: dict) -> tuple[list[dict], dict]:
    """Quarantine row-local auxiliary failures before any optimizer or cursor mutation."""
    from .train_v17 import RowEncodingError

    auxiliary = {"A1", "C", "N"}
    identities = [row["row_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("v18b: duplicate attempted update row")
    encoded, rejected = [], []
    for row in rows:
        try:
            encoded.extend(_encode(bundle, [row], items, image))
        except RowEncodingError as error:
            if row["kind"] not in auxiliary:
                raise
            rejected.append({"row_id": row["row_id"], "problem_id": row["problem_id"], "kind": row["kind"],
                "source_row_sha256": digest(row), "reason_code": error.code, "reason": str(error)})
    optional = sum(row["kind"] in auxiliary for row in rows)
    report = {"schema": "v18b-row-quarantine-v1", "attempted_rows": len(rows), "auxiliary_rows": optional,
        "admitted_row_ids": [row["row_id"] for row in encoded], "rejected_rows": rejected,
        "status": "SYSTEMATIC_AUXILIARY_FAILURE" if len(rejected) * 2 > optional else "CONTINUE"}
    return encoded, report


def v17_stage_report(records: dict[str, dict], contract: dict, rows: list[dict], stats: dict,
                     items: dict[str, dict] | None = None) -> dict:
    """Keep stage-specific denominators; a note generation is not a verified repair."""
    from collections import Counter
    from .guard_v16b import length_distribution_v17

    memberships = {"T1": contract["expected_problem_ids"], "OBS": contract["expected_obs_problem_ids"],
                   "NOTE": contract["note_problem_ids"], "RESOLVE": contract["resolve_problem_ids"]}
    report: dict[str, Any] = {"parents_planned": len(memberships["T1"]), "row_counts": dict(Counter(r["kind"] for r in rows)),
              "stages": {}, "rejection_reasons_all": {}, "claims": {}}
    rejected: Counter[str] = Counter()
    claims: Counter[str] = Counter()
    for stage, ids in memberships.items():
        selected = [records[pid] for pid in ids]
        outputs = [r[stage.lower()] for r in selected if stage.lower() in r]
        values: dict[str, Any] = {"parents": len(ids), "requested": sum(stage.lower() + "_prompt" in r for r in selected),
                  "completed": len(outputs), "tokens": sum(len(o["generation"]["completion_token_ids"]) for o in outputs)}
        if stage == "T1":
            rewards = [r["reward_receipt"] for r in selected]
            advantages = [r["reward_receipt"]["reward"] - r["t1"]["generation"]["baseline_receipt"]["baseline"]
                          for r in selected if r["reward_receipt"]["pg"]]
            values.update(graded=sum(o.get("status") in {
                              "CORRECT", "INCORRECT", "TRUNCATED", "INVALID_NO_FINAL", "INVALID_MULTIPLE",
                              "NO_FINAL", "NO_PARSE", "MULTIPLE_FINAL_ANSWERS"} for o in outputs),
                          pg_positive=sum(x > 0 for x in advantages), pg_negative=sum(x < 0 for x in advantages),
                          pg_excluded=sum(not r["pg"] for r in rewards),
                          pg_excluded_tokens=sum(len(r["t1"]["generation"]["completion_token_ids"])
                                                for r in selected if not r["reward_receipt"]["pg"]))
            values["statuses"] = dict(Counter(o.get("status") or "UNVERIFIABLE" for o in outputs))
            values["length"] = length_distribution_v17(
                [len(o["generation"]["completion_token_ids"]) for o in outputs])
            values["truncated"] = sum(o["generation"].get("finish_reason") == "length" for o in outputs)
        receipt_key = {"T1": "a1_receipt", "OBS": "obs_score", "NOTE": "note_receipt",
                       "RESOLVE": "resolve_receipt"}[stage]
        receipts = [r[receipt_key] for r in selected if receipt_key in r]
        values["accepted"] = sum(r.get("accepted") is True for r in receipts)
        for receipt in receipts:
            rejected.update(str(reason) for reason in receipt.get("reasons") or [])
        if stage == "NOTE":
            values.update(verified_repairs=sum(r.get("verified_repair_count", 0) for r in receipts),
                          lesson_safe=sum(r.get("lesson_accepted") is True for r in receipts),
                          lesson_rejected=sum(r.get("lesson_accepted") is False for r in receipts),
                          lesson_rejection_reasons=dict(Counter(reason for r in receipts
                              for reason in r.get("lesson_leak_reasons", []))))
        elif stage == "RESOLVE":
            values["correct"] = sum(o.get("status") == "CORRECT" for o in outputs)
        elif stage == "OBS":
            values["schema_valid"] = sum(not any("SCHEMA" in reason for reason in r.get("reasons", [])) for r in receipts)
        for record in selected:
            ledger = record.get("t1_ledger" if stage == "T1" else "obs_ledger") if stage in {"T1", "OBS"} else None
            if ledger:
                claims.update(str(c["verdict"]) for c in ledger["claims"])  # sealed keys must stay strings
        report["stages"][stage] = values
    report["rejection_reasons_all"], report["claims"] = dict(rejected), dict(claims)
    report["answer_only"] = rejected["ANSWER_ONLY"]
    report["n_verified_content_tokens"] = sum(sum(row.get("verified_content_token_mask", []))
                                               for row in rows if row["kind"] == "N")
    report["training_signal"] = {key: stats.get(key) for key in (
        "t1_negative_candidate_tokens", "t1_negative_retained_tokens", "t1_negative_retained_ratio",
        "loop_confirmed", "loop_tokens_penalized", "length_nonloop_excluded",
        "n_content_gradient_finite_nonzero", "note_gradient_retained", "verified_repair_parent_ids",
        "consumed_row_ids", "note_parent_gradient_norms", "component_gradient_norms",
        "note_update_alignment", "policy_mask_counts")}
    if "n_parent_evidence" in stats:
        report["training_signal"]["n_parent_evidence"] = stats["n_parent_evidence"]
    if items is not None:
        keys = ("expected_problem_ids", "expected_obs_problem_ids", "note_problem_ids", "resolve_problem_ids")
        sources = sorted({str(items[pid]["source"]) for ids in memberships.values() for pid in ids})
        report["sources"] = {}
        for source in sources:
            subset = {key: [pid for pid in contract[key] if str(items[pid]["source"]) == source] for key in keys}
            source_rows = [row for row in rows if str(items[row["problem_id"]]["source"]) == source]
            detail = v17_stage_report(records, subset, source_rows, {})
            del detail["training_signal"]  # Joint gradients cannot be attributed by dividing source counts.
            report["sources"][source] = detail
    return report



def v18_supply_report(records: dict[str, dict], ordered_items: list[dict], update_results: list[dict],
                      consumed_row_ids: list[str]) -> dict:
    """Fixed parent cohorts; joint update alignment is reported without source attribution."""
    consumed = set(consumed_row_ids)
    measured: set[str] = set()
    nonzero: set[str] = set()
    linked: set[str] = set()
    alignments = []
    for result in update_results:
        stats = result["result"]["stats"]
        norms = stats.get("note_parent_gradient_norms") or {}
        measured.update(norms)
        nonzero.update(pid for pid, groups in norms.items() if any(v > 0 for v in groups.values()))
        if result["result"].get("optimizer_steps", 0):
            linked.update(stats.get("verified_repair_parent_ids") or [])
        if stats.get("note_update_alignment") is not None:
            alignments.append({"global_slot": result["global_slot"], "measurement": stats["note_update_alignment"]})

    def metrics(items: list[dict]) -> dict:
        total: dict[str, Any] = {"parents": len(items), "t1_completed": 0, "supported_parents": 0, "claimed_parents": 0,
                 "contradicted_parents": 0, "note_generated": 0, "verified_repair_parents": 0,
                 "n_accepted": 0, "n_gradient_measured": 0, "n_gradient_nonzero": 0,
                 "n_saved_update_linked": 0, "c_accepted": 0, "c_repair_used_and_consumed": 0,
                 "note_conditions": {"self_found": 0, "location_given": 0}}
        for item in items:
            pid, record = item["problem_id"], records.get(item["problem_id"], {})
            claims = [claim for key in ("t1_ledger", "obs_ledger") for claim in (record.get(key) or {}).get("claims", [])]
            note, resolve = record.get("note_receipt") or {}, record.get("resolve_receipt") or {}
            condition = note.get("note_condition") or (record.get("note_request_receipt") or {}).get("note_condition")
            total["t1_completed"] += int("t1" in record)
            total["supported_parents"] += int(bool((item.get("v18_private_verifier") or item.get("v17_private_verifier") or {}).get("repair_catalog_eligible")))
            total["claimed_parents"] += int(bool(claims))
            total["contradicted_parents"] += int(any(c.get("verdict") == "CONTRADICTED" for c in claims))
            total["note_generated"] += int("note" in record)
            total["verified_repair_parents"] += int(note.get("verified_repair_count", 0) > 0)
            total["n_accepted"] += int(note.get("n_accepted", note.get("accepted")) is True)
            total["n_gradient_measured"] += int(pid in measured)
            total["n_gradient_nonzero"] += int(pid in nonzero)
            total["n_saved_update_linked"] += int(pid in linked and pid + ":N" in consumed)
            total["c_accepted"] += int(resolve.get("accepted") is True)
            total["c_repair_used_and_consumed"] += int(bool(resolve.get("corrected_use_byte_spans")) and pid + ":C" in consumed)
            if condition in total["note_conditions"]:
                total["note_conditions"][condition] += 1
        return total

    cohorts = []
    for start in range(0, len(ordered_items), 512):
        members = ordered_items[start:start + 512]
        sources = sorted({item["source"] for item in members})
        cohorts.append({"start_position": start, "end_position_exclusive": start + len(members),
            "counts": metrics(members), "sources": {source: metrics([i for i in members if i["source"] == source]) for source in sources}})
    return {"cohorts": cohorts, "overall": metrics(ordered_items), "joint_note_update_alignment": alignments,
            "gradient_attribution": "Parent norms and consumed N rows are counted; joint alignment is not divided by source.",
            "effect_measured": False}


def _checkpoint_path(state: dict) -> Path:
    return checked(state["checkpoint"]).parent


def _rollback(output: Path, slot: int, *, committed_checkpoint: Path | None = None,
              committed_probe: Path | None = None, committed_runtime: Path | None = None) -> None:
    protected = {p.resolve() for p in (committed_checkpoint, committed_probe, committed_runtime) if p is not None}
    marker = output / "prep-checkpoint.json"
    if marker.exists():
        protected.add(checked(json.loads(marker.read_text())).parent.resolve())
    candidates = [path for directory in ("transactions", "update-results", "row-admissions")
                  for path in (output / directory).glob("slot-*.json")
                  if int(path.stem.split("-")[-1]) >= slot]
    inclusive = committed_checkpoint is not None
    for base, pattern in ((output, "checkpoint-*"), (output / "probes", "slot-*"),
                          (output / "runtime-snapshots", "slot-*.json")):
        for path in base.glob(pattern):
            number = int(path.name.split("-")[1].split(".")[0])
            if path.resolve() not in protected and (number >= slot if inclusive else number > slot):
                candidates.append(path)
    if candidates:
        destination = output / "rolled-back" / f"recovery-{time.time_ns()}"
        destination.mkdir(parents=True)
        for path in candidates:
            target = destination / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))


def run_campaign(plan: dict, *, output: Path, args: argparse.Namespace, reload_only: bool = False) -> dict:
    from .v18b_recovery import RecoveryCopyHook, recovery_paths, resolve_recovery_path
    with recovery_paths(output):
        config = plan.get("recovery")
        hook = (RecoveryCopyHook(output, resolve_recovery_path(config["destination"]),
                transfer_command=config["transfer_command"]) if config is not None and not reload_only else None)
        try:
            return _run_campaign(plan, output=output, args=args, reload_only=reload_only, recovery_hook=hook)
        finally:
            if hook is not None:
                hook.close()


def _run_campaign(plan: dict, *, output: Path, args: argparse.Namespace, reload_only: bool = False,
                  recovery_hook=None) -> dict:
    from fine_tuning.notebook_rl import checkpoint
    algorithm_version = plan["protocol"]["version"]
    is_v18b = algorithm_version == "error-note-v18b"
    execution_settings = plan.get("execution_settings")
    args.monitor_stop_action = (execution_settings or {}).get("monitor_stop_action", "stop")
    is_v17 = algorithm_version in MODERN_VERSIONS
    is_v18 = is_v18b or algorithm_version == "error-note-v18"
    data = data_api(algorithm_version)
    load_items, validate = data.load_items, data.validate
    from .guard_v16b import GuardV16B
    guard_class: Any = GuardV16B
    from .protocol import ARM, EVAL_SAMPLING_V16B, TRAIN_SAMPLING_V16B, v16b_contract_for_update
    from .rl import policy_sha256
    from .run_v16 import _setup_execution, _state_equal
    from .train import _read_probe_rows, _write_probe, compute_probe
    from .v16b_runtime import RuntimeCallbacks, V16bScheduler, run_v16b
    from . import v16b_rows as row_tools

    if is_v17:
        from . import v17_rows as rows_v17
        v17_rows: Any = rows_v17
        from .protocol import v17_contract_for_update
        from .guard_v16b import GuardV17
        v16b_contract_for_update = v17_contract_for_update
        guard_class = GuardV17
        if is_v18:
            from . import v18_rows as rows_v18
            v17_rows = rows_v18
            from .protocol import v18_contract_for_update
            v16b_contract_for_update = v18_contract_for_update
        if is_v18b:
            from . import v18b_rows as rows_v18b
            v17_rows = rows_v18b
            from .protocol import v18b_contract_for_update as v16b_contract_for_update
    _plan_check(plan)
    rehearsal = plan["protocol"]["training_contract"].get("rehearsal_profile") is not None
    if rehearsal:
        from .v18b_rehearsal import validate_execution
        validate_execution(args, plan)
    elif is_v17:
        from .v17_preflight import bind_execution, bind_training
        (bind_training if getattr(args, "execution_phase", "legacy") == "training" else bind_execution)(args, plan)
    for value in plan["inputs"].values():
        checked(value)
    execution_lineage = {}
    if is_v18b:
        from .v18b_hotfix import execution_source
        execution_lineage = execution_source(args, plan, single.source_manifest())
    elif digest(single.source_manifest()["actual_files"]) != digest(plan["source"]["actual_files"]):
        raise ValueError("v16b campaign: execution source differs from the prepared source")
    if not reload_only and not (args.allow_model_run and args.allow_update and args.allow_save):
        raise ValueError("v16b campaign: actual execution requires model/update/save opt-ins")
    if rehearsal:
        from .v18b_rehearsal import validate_inputs
        data_validation = validate_inputs(plan)
    else:
        data_validation = validate(args.data_root)
    _setup_execution(MASTER_SEED)
    protocol, image = plan["protocol"], plan["protocol"]["image"]
    cfg = protocol["training_contract"]
    all_items = {item["problem_id"]: item for split in (("prep", "train", "monitor", "effect", "reasoning_monitor") if is_v18b else ("prep", "train", "monitor", "effect") if is_v17 else ("prep", "train", "monitor"))
                 for item in load_items(args.data_root, split)}
    state_path = output / "campaign-state.json"
    selected_state_path = output / "evaluation-checkpoint-state.json"
    if reload_only and selected_state_path.exists():
        state_path = selected_state_path
    loaded_state = read_sealed(state_path) if state_path.exists() else None
    if loaded_state is not None and loaded_state["plan_sha256"] != plan["sha256"]:
        raise ValueError("v16b campaign: stale state belongs to another plan")
    sampler: Any = None
    optimizer: Any
    scheduler: Any
    factory = _factory(args)
    if loaded_state is None:
        _rollback(output, -1)
        if reload_only:
            raise ValueError("v16b campaign: no committed checkpoint")
        bundle = factory(args.model_path, ARM, protocol)
        if args.precision == "bf16":
            bundle.model.to("cuda:0")
        optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
        cursor: dict[str, Any] = {"slot": 0, "updates": 0, "consumed_problem_ids": [], "consumed_row_ids": [],
            "policy_sha256": policy_sha256(bundle), "phase": "prep", "policy_version": 0, "rl_block": 0,
            "producer_pid": os.getpid(), "last_transaction_sha256": None, "parent_checkpoint": None}
        state: dict = {"version": plan["version"], "plan_sha256": plan["sha256"], "cursor": cursor,
                 "status": "READY", "checkpoint": None, "runtime": None, "guard": None,
                 "stop_reasons": [], "block_seconds": []}
    else:
        state = loaded_state
        if reload_only and state["cursor"].get("producer_pid") == os.getpid():
            raise ValueError("v16b campaign: reload must run in a fresh process")
        bundle, optimizer, scheduler, cursor = single.load_single_continuation(args.model_path,
            _checkpoint_path(state), protocol, binding=_binding(plan), device="cuda:0", policy_factory=factory)
        if cursor != state["cursor"]:
            raise ValueError("v16b campaign: checkpoint cursor differs from durable state")
        if not reload_only:
            _rollback(output, cursor["slot"], committed_checkpoint=_checkpoint_path(state),
                committed_probe=checked(state["probe"]["json"]).parent,
                committed_runtime=checked(state["runtime"]) if state.get("runtime") is not None else None)
    if execution_lineage:
        state.update(execution_lineage)
    if reload_only:
        import torch
        path = _checkpoint_path(state)
        runtime = torch.load(path / "runtime.pt", map_location="cpu", weights_only=True)
        equal = all(_state_equal(actual, runtime[key]) for key, actual in (
            ("optimizer", optimizer.state_dict()), ("scheduler", scheduler.state_dict()), ("rng", checkpoint.capture_rng())))
        probe_path = checked(state["probe"]["json"])
        checked(state["probe"]["tensors"])
        expected = json.loads(probe_path.read_text())
        actual = compute_probe(bundle, _read_probe_rows(probe_path.parent, expected))
        differences = [abs(a - b) for aa, bb in zip(actual["logprobs"], expected["logprobs"], strict=True)
                       for a, b in zip(aa, bb, strict=True)]
        if not equal or not differences or max(differences) != 0 or actual["greedy_ids"] != expected["greedy_ids"]:
            raise ValueError("v16b campaign: fresh reload optimizer/RNG/probe mismatch")
        receipt = seal(output / f"reload-{cursor['slot']}.json", {"version": "v16b-reload-v1",
            "plan_sha256": plan["sha256"], "checkpoint": state["checkpoint"], "cursor": cursor,
            "producer_pid": cursor["producer_pid"], "reload_pid": os.getpid(), "runtime_equal": True,
            "probe_max_abs_difference": max(differences), "greedy_equal": True})
        return state | {"status": "RELOADED", "reload": receipt}
    if getattr(args, "resume_funding_stop", False):
        if (state["status"] != "STOPPED" or state.get("stop_reasons") != ["STOP_REQUESTED"]
                or (state.get("guard") or {}).get("reasons") or cursor["slot"] >= plan["total_slots"]):
            raise ValueError("v16b campaign: --resume-funding-stop needs a non-final STOPPED state whose only "
                             f"reason is STOP_REQUESTED (status {state['status']}, reasons {state.get('stop_reasons')})")
        if _stop_markers(output):
            raise ValueError("v16b campaign: archive the STOP_REQUESTED/WATCHDOG_STOP markers before resuming")
        # In memory only: the next regular commit persists READY; a crash before it keeps the STOPPED checkpoint.
        cursor["phase"] = "rl" if cursor["slot"] >= cfg["prep_updates"] else "prep"
        state = dict(state, status="READY", stop_reasons=[], cursor=cursor)
    if (is_v18b and execution_lineage and state["status"] == "STOPPED" and state.get("stop_reasons") == ["TRUNC_UP"]
            and cursor["slot"] < plan["total_slots"] and not _stop_markers(output)):
        # v18b operator fix (user 2026-11-19 KST): TRUNC_UP is a recorded warning (GuardV17.trunc_stop), so a campaign
        # stopped only on it continues from its stopped boundary under an explicit hotfix authorization.
        # In memory only: the next regular commit persists READY; a crash before it keeps the STOPPED checkpoint.
        cursor["phase"] = "rl" if cursor["slot"] >= cfg["prep_updates"] else "prep"
        state = dict(state, status="READY", stop_reasons=[], cursor=cursor)
    partial_pending = partial_evaluation_state(plan, state, output) if is_v18b else None
    if state["status"] in {"STOPPED", "WATCHDOG_STOP", "PREP_LENGTH_STOP", "COMPLETE"} and partial_pending is None:
        if state["status"] == "COMPLETE":
            require_complete(output, check_evaluation=False)
        return state
    if sampler is None:
        args._optimizer, args._scheduler, args._slot = optimizer, scheduler, cursor["slot"]
        args._preflight_stage = "f0" if cursor["slot"] == 0 else "trained"
        sampler = make_sampler(bundle, args, protocol, image=image, sampling=TRAIN_SAMPLING_V16B)
    live_capacity = None
    if is_v18b and not args.local_sequential:
        try:
            if rehearsal:
                from .v18b_rehearsal import live_preflight
                live = live_preflight(args, plan, bundle, sampler)
            else:
                from .v17_preflight import live_training_preflight
                live = live_training_preflight(args, plan, bundle, sampler,
                    capacity=lambda: calibrate_capacity(bundle, args, protocol, live_engine=True))
            live_capacity = live["capacity"]
        finally:
            sampler.release_idle_versions()
    prior = read_sealed(checked(plan["inputs"]["baseline"]))
    generated = read_sealed(checked(plan["inputs"]["f0"]))
    prep_rows = [row for batch in plan["prep_batches"] for row in batch]
    probe_rows = _encode(bundle, prep_rows[:1], all_items, image)
    guard = (guard_class(plan["blocks"], state.get("guard"), collapse_stop=False) if is_v18
             else guard_class(plan["blocks"], state.get("guard")))
    signal_gate: Any = None
    if is_v17:
        from .guard_v16b import SignalGateV17, SignalGateV18B
        signal_gate = (SignalGateV18B if is_v18b else SignalGateV17)(plan["blocks"], plan["signal_gate"]["supported_parent_ids"],
            state.get("signal_gate"), local_smoke=args.local_sequential or rehearsal)
    runtime = None

    def commit(label: str, envelope: dict | None = None) -> dict:
        nonlocal state
        if sampler.has_unfinished_requests():
            raise RuntimeError("v16b campaign: checkpoint attempted before engine drain")
        cursor["producer_pid"] = os.getpid()
        path = output / f"checkpoint-{cursor['slot']}-{label}"
        runtime_descriptor = None
        if envelope is not None:
            snapshot = output / "runtime-snapshots" / f"slot-{cursor['slot']}.json"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            single._sealed(snapshot, envelope)
            runtime_descriptor = descriptor(snapshot)
        single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=path,
            protocol=protocol, binding=_binding(plan), allow_save=True)
        rng = checkpoint.capture_rng()
        try:
            probe_dir = output / "probes" / f"slot-{cursor['slot']}-{label}"
            probe_dir.mkdir(parents=True)
            _write_probe(bundle, probe_rows, probe_dir)
        finally:
            checkpoint.restore_rng(rng)
        if signal_gate is not None:
            signal_gate.mark_checkpoint(descriptor(path / "metadata.json")["file_sha256"])
            state["signal_gate"] = copy.deepcopy(signal_gate.state)
        body = {k: v for k, v in state.items() if k != "sha256"}
        body.update(cursor=copy.deepcopy(cursor), checkpoint=descriptor(path / "metadata.json"),
                    runtime=runtime_descriptor, guard=copy.deepcopy(guard.state), status=state["status"],
                    probe={"json": descriptor(probe_dir / "probe.json"),
                           "tensors": descriptor(probe_dir / "probe_inputs.safetensors")})
        state = body | {"sha256": digest(body)}
        write_json(state_path, state)
        if label == "prep-end":
            write_json(output / "prep-checkpoint.json", state["checkpoint"])
        if recovery_hook is not None:
            recovery_hook.poll()
            if label == "prep-end":
                recovery_hook.checkpoint(state, force=True)
        keep = {path.resolve()}
        normal = state.get("v18b_normal_state")
        if normal is not None:
            keep.add(checked(normal["checkpoint"]).parent.resolve())
        prep_marker = output / "prep-checkpoint.json"
        if prep_marker.exists():
            keep.add(checked(json.loads(prep_marker.read_text())).parent.resolve())
        for old in output.glob("checkpoint-*"):
            if old.is_dir() and old.resolve() not in keep:
                shutil.rmtree(old)
        return state["checkpoint"]

    def update(rows: list[dict], contract: dict) -> dict:
        slot = cursor["slot"]
        if contract["global_slot"] != slot:
            raise ValueError("v16b campaign: update slot differs from immutable LR schedule")
        if not args.local_sequential:
            capacity = live_capacity if is_v18b else read_sealed(output / "capacity.json")
            if capacity is None:
                raise ValueError("v18b campaign: live memory-shape receipt is missing")
            contract["memory_shape_bounds"] = capacity["shape_bounds"]
        import torch
        cuda_initialized = torch.cuda.is_initialized()
        if cuda_initialized:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        attempted_rows = rows
        quarantine = None
        if is_v18b:
            encoded, quarantine = _encode_update_rows(bundle, rows, all_items, image)
            admission = seal(output / "row-admissions" / f"slot-{slot}.json", {
                "plan_sha256": plan["sha256"], "slot": slot, "parent_checkpoint": state["checkpoint"],
                "report": quarantine})
            if quarantine["status"] != "CONTINUE":
                raise RuntimeError("v18b: more than half the auxiliary rows failed replay; "
                                   "the last closed checkpoint is preserved")
            admitted_ids = set(quarantine["admitted_row_ids"])
            rows = [row for row in rows if row["row_id"] in admitted_ids]
        else:
            encoded = _encode(bundle, rows, all_items, image)
        transaction = seal(output / "transactions" / f"slot-{slot}.json", {
            "plan_sha256": plan["sha256"], "slot": slot, "rows_sha256": digest(attempted_rows),
            **({"admitted_rows_sha256": digest(rows), "row_admission_sha256": admission["sha256"]} if is_v18b else {}),
            "contract": contract, "parent_checkpoint": state["checkpoint"]})
        cursor["last_transaction_sha256"] = transaction["sha256"]
        result = single.run_joint_single_update(bundle, optimizer, scheduler, encoded, contract,
            cursor=cursor, block_ids=contract["expected_problem_ids"], allow_update=True,
            token_budget=plan["capacity"]["token_budget"])
        if quarantine is not None:
            result["stats"]["row_quarantine"] = quarantine
        if cuda_initialized:
            torch.cuda.synchronize()
        generation_counts = {}
        if runtime is not None:
            for kind, key in (("NOTE", "note_problem_ids"), ("RESOLVE", "resolve_problem_ids")):
                records = [runtime.records[pid] for pid in contract[key]]
                receipts = [record[kind.lower() + "_receipt"] for record in records
                            if kind.lower() + "_receipt" in record]
                generation_counts[kind] = {"generated": sum(kind.lower() in record for record in records),
                    "accepted": sum(receipt["accepted"] is True for receipt in receipts),
                    "rejected": sum(receipt["accepted"] is False for receipt in receipts)}
        if is_v17 and runtime is not None:
            seal(output / "signal-reports" / f"slot-{slot}.json", {
                "version": "v17-stage-signal-report-v1", "plan_sha256": plan["sha256"],
                "global_slot": slot, "report": v17_stage_report(runtime.records, contract, rows, result["stats"], all_items)})
        seal(output / "update-results" / f"slot-{slot}.json", {
            "version": "v18b-update-result-v1" if is_v18b else "v17-update-result-v1" if is_v17 else "v16b-update-result-v1",
            **({"producer_pid": os.getpid(), "timing_scope": "encode_backward_optimizer"} if is_v18b else {}),
            "plan_sha256": plan["sha256"],
            "transaction_sha256": transaction["sha256"], "phase": contract["phase"], "global_slot": slot,
            "result": result, "row_counts": {kind: sum(row["kind"] == kind for row in rows)
                                            for kind in sorted({row["kind"] for row in rows})},
            **({"attempted_row_counts": {kind: sum(row["kind"] == kind for row in attempted_rows)
                 for kind in sorted({row["kind"] for row in attempted_rows})}} if is_v18b else {}),
            "generation_counts": generation_counts, "wall_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda_initialized else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda_initialized else None})
        result["stats"]["optimizer_step"] = bool(result["optimizer_steps"])
        if signal_gate is not None:
            signal_gate.consume(result["stats"])
        write_json(output / "progress.json", {"plan_sha256": plan["sha256"], "cursor": cursor,
                   "transaction_sha256": transaction["sha256"]})
        return result["stats"]

    if state["checkpoint"] is None:
        commit("initial")
    while cursor["slot"] < cfg["prep_updates"]:
        slot = cursor["slot"]
        rows = plan["prep_batches"][slot]
        contract = v16b_contract_for_update(protocol, slot, phase="prep", policy_sha256=policy_sha256(bundle),
            expected_problem_ids=list(dict.fromkeys(row["problem_id"] for row in rows)))
        update(rows, contract)
        if cursor["slot"] % 8 == 0 and cursor["slot"] < cfg["prep_updates"]:
            commit("prep")
        markers = _stop_markers(output)
        if markers:
            state.update(status="STOPPED", stop_reasons=markers)
            cursor["phase"] = "stopped"
            commit("stopped")
            sampler.release()
            return state
    if not (output / "prep-checkpoint.json").exists():
        if (state["checkpoint"] is not None and _checkpoint_path(state).name.endswith("-prep-end")
                and cursor["slot"] == cfg["prep_updates"]):
            write_json(output / "prep-checkpoint.json", state["checkpoint"])
        else:
            cursor.update(phase="rl", policy_version=0, rl_block=0)
            state["status"] = "READY"
            commit("prep-end")
    if is_v17 and hasattr(sampler, "set_preflight_stage_v17"):
        sampler.set_preflight_stage_v17("trained")
    selection = read_sealed(checked(plan["inputs"]["selection"]))
    monitors = [all_items[pid] for pid in selection["monitor_ids"]]
    def monitor(label: str, version: int, split: str = "monitor") -> dict:
        path = output / (label + ".json")
        if path.exists():
            stored = read_sealed(path)
            if stored.get("checkpoint") != state["checkpoint"] or stored.get("plan_sha256") != plan["sha256"]:
                raise ValueError("campaign: monitor belongs to another checkpoint or plan")
            return stored
        if cfg.get("rehearsal_profile"):
            from .v18b_rehearsal import before_final_monitor
            before_final_monitor(args, plan, label)
        sampler.release_idle_versions()
        saved_rng = checkpoint.capture_rng()
        short = is_v18b and split in {"short_monitor", "reasoning_monitor"}
        targets = monitors if split in {"monitor", "short_monitor"} else [all_items[row["problem_id"]] for row in generated[split]]
        grade_split = "monitor" if split == "short_monitor" else split
        try:
            options = {"algorithm_version": algorithm_version} if is_v17 else {}
            records = collect(bundle, sampler, targets, data_root=args.data_root, split=grade_split, out=output / label,
                stage=("v18b-b2-" + split.replace("_", "-")) if short
                    else ("v18-" if is_v18 else "v17-" if is_v17 else "v16b-") + ("monitor" if split == "monitor" else "effect"),
                protocol_sha256=protocol["sha256"], sampling=EVAL_SAMPLING_V16B,
                cap=64 if short else cfg["max_tokens"]["monitor"], version=version,
                **(options | {"resume_incomplete": True} if is_v18b else options))
        finally:
            checkpoint.restore_rng(saved_rng)
            # collect() registers `version`; the signal-gate monitor uses the next policy version, which run_v16b
            # registers again right after (add_version refuses an existing version), so release it here.
            sampler.release_idle_versions()
        body = {"version": "v17-monitor-v1" if is_v17 else "v16b-monitor-v1", "records": records,
                "summary": summary(records), "checkpoint": state["checkpoint"], "plan_sha256": plan["sha256"]}
        if is_v17 and not short:
            from .v17_eval import paired_report
            from .guard_v16b import length_distribution_v17
            def paired_input(values):
                return {r["problem_id"]: {"correct": r["output"].get("status") == "CORRECT",
                    "finish_reason": r["output"]["finish_reason"], "raw_response": r["output"]["text"]}
                    for r in values}
            body["comparison"] = paired_report(paired_input(generated[split]), paired_input(records))
            if split == "effect":
                body["comparison"]["eligible"] = data_validation.get("effect_eligible") is True
                body["comparison"]["exposure_status"] = data_validation.get("effect_status", "UNVERIFIED")
            body["length_distribution"] = length_distribution_v17(
                [len(r["output"]["generation"]["completion_token_ids"]) for r in records])
            body["f0_length_distribution"] = length_distribution_v17(
                [len(r["output"]["generation"]["completion_token_ids"]) for r in generated[split]])
        return seal(output / (label + ".json"), body)
    def finish_partial() -> dict | None:
        nonlocal state, bundle, optimizer, scheduler, cursor, sampler
        selected = partial_evaluation_state(plan, state, output) if is_v18b else None
        if selected is None:
            return None
        selected_path = output / "evaluation-checkpoint-state.json"
        if selected_path.exists() and read_sealed(selected_path) != selected:
            raise ValueError("v18b partial selection changed after publication")
        single._sealed(selected_path, selected)
        if selected["checkpoint"] != state["checkpoint"]:
            sampler.release()
            # Reuse the allocated frozen base; load the selected adapters/moments/RNG into it.
            existing = bundle
            optimizer, scheduler = None, None
            args._optimizer, args._scheduler = None, None
            bundle, optimizer, scheduler, cursor = single.load_single_continuation(args.model_path,
                _checkpoint_path(selected), protocol, binding=_binding(plan), device="cuda:0",
                policy_factory=lambda *unused: existing)
            args._optimizer, args._scheduler, args._slot = optimizer, scheduler, cursor["slot"]
            args._preflight_stage = "trained"
            sampler = make_sampler(bundle, args, protocol, image=image, sampling=TRAIN_SAMPLING_V16B)
        state = selected
        def selected_monitor(label: str, split: str = "monitor") -> dict:
            path = output / (label + ".json")
            record = read_sealed(path) if path.exists() else monitor(label, state["cursor"]["policy_version"], split=split)
            if record.get("checkpoint") != state["checkpoint"] or record.get("plan_sha256") != plan["sha256"]:
                raise ValueError("v18b partial monitor belongs to another checkpoint")
            return record
        final_monitor = selected_monitor("monitor-final")
        selected_monitor("short-monitor-final", "short_monitor")
        selected_monitor("reasoning-monitor-final", "reasoning_monitor")
        effect = selected_monitor("effect-final", "effect")
        from .v17_eval import evaluation_gate
        seal(output / "evaluation-admission.json", {"version": "v17-evaluation-admission-v1",
            "plan_sha256": plan["sha256"], "checkpoint": state["checkpoint"],
            "monitor_sha256": final_monitor["sha256"], "effect_sha256": effect["sha256"],
            "decision": evaluation_gate(final_monitor["comparison"], effect["comparison"]),
            "evaluation_selection": state["evaluation_selection"]})
        require_complete(output, check_evaluation=False)
        sampler.release()
        return state

    if partial_pending is not None:
        selected = finish_partial()
        if selected is None:
            raise ValueError("v18b preauthorized partial boundary disappeared")
        return selected
    post_path = output / "monitor-post-prep.json"
    post = read_sealed(post_path) if post_path.exists() else monitor("monitor-post-prep", 0)
    prep_checkpoint = json.loads((output / "prep-checkpoint.json").read_text())
    if post.get("plan_sha256") != plan["sha256"] or post.get("checkpoint") != prep_checkpoint:
        raise ValueError("v16b campaign: post-preparation monitor binding changed")
    if is_v18b:
        short_post_path = output / "short-monitor-post-prep.json"
        short_post = (read_sealed(short_post_path) if short_post_path.exists() else
                      monitor("short-monitor-post-prep", 0, split="short_monitor"))
        if short_post.get("checkpoint") != prep_checkpoint or short_post.get("plan_sha256") != plan["sha256"]:
            raise ValueError("V18B_SHORT_POST_PREP_BINDING")
        reasoning_post_path = output / "reasoning-monitor-post-prep.json"
        reasoning_post = (read_sealed(reasoning_post_path) if reasoning_post_path.exists() else
                          monitor("reasoning-monitor-post-prep", 0, split="reasoning_monitor"))
        if reasoning_post.get("checkpoint") != prep_checkpoint or reasoning_post.get("plan_sha256") != plan["sha256"]:
            raise ValueError("V18B_REASONING_POST_PREP_BINDING")
        if "v18b_normal_state" not in state:
            state["v18b_normal_checkpoint"] = copy.deepcopy(state["checkpoint"])
            state["v18b_normal_state"] = {key: copy.deepcopy(state[key]) for key in
                ("cursor", "checkpoint", "runtime", "probe", "guard")}
    before, after = generated["monitor_summary"]["median_tokens"], post["summary"]["median_tokens"]
    if before is None or after is None or (not is_v18b and after < .5 * before):
        state["status"] = "PREP_LENGTH_STOP"
        state["stop_reasons"] = ["PREP_LENGTH_COLLAPSE"]
        cursor["phase"] = "stopped"
        commit("stopped")
        sampler.release()
        return state
    sampler.release_idle_versions()
    grade = grader(args.data_root, "train")
    grade_v8 = grader(args.data_root, "train", raw_v8=True) if is_v18 else None
    contracts: dict[str, dict] = {}
    for item in plan["items"]:
        contracts[item["problem_id"]] = (v17_rows.problem_contract_v17(item) if is_v17
                                              else row_tools.problem_contract_v16b(item))
    starts = [time.monotonic()]

    def record_view(item: dict, record: dict) -> dict:
        block = next(i for i, ids in enumerate(plan["blocks"]) if item["problem_id"] in ids)
        return dict(item, **record, p0_text=p0(item), block_index=block, subject=item.get("subject", item["source"]))

    def request(item: dict, kind: str, version: int, record: dict) -> dict | None:
        view, source = record_view(item, record), contracts[item["problem_id"]]
        if kind == "OBS":
            eligibility = (v17_rows.observation_eligibility_v17(item) if is_v17
                           else row_tools.observation_eligibility_v16b(item))
            if eligibility != plan["observation_eligibility"][item["problem_id"]]:
                raise ValueError("campaign: source observation eligibility changed")
            if not eligibility["eligible"]:
                record["obs_skip"] = eligibility
                return None
            prompt = (v17_rows.build_obs_request_v17(item)["prompt"] if is_v17
                      else row_tools.build_obs_prompt_v16b(item))
        elif kind == "NOTE":
            try:
                if is_v17:
                    ledgers = [record[key] for key in ("t1_ledger", "obs_ledger") if record.get(key) is not None]
                    note_request = v17_rows.build_note_request_v17(view, ledgers=ledgers,
                        gold=item["evaluation"]["gold"], tokenizer=bundle.processor.tokenizer)
                    if note_request is None:
                        record["note_skip"] = {"reason": "NO_VERIFIED_ERROR"}
                        return None
                    prompt = note_request["prompt"]
                    if is_v18:
                        record["note_request_receipt"] = {key: value for key, value in note_request.items() if key != "prompt"}
                else:
                    prompt = row_tools.build_note_prompt_v16b(view, fact_catalog=source["fact_catalog"],
                        rules=source["rules"], gold=item["evaluation"]["gold"])
            except ValueError as exc:
                allowed = {"SKIP_NOTE_NO_ERROR_OR_INCOMPLETE"}
                if is_v17:
                    allowed |= {"SKIP_NOTE_INCOMPLETE_WITHOUT_VERIFIED_ERROR", "SKIP_NOTE_NO_ERROR", "NOTE_CONTEXT_BLOCKED"}
                if str(exc) not in allowed:
                    raise
                record["note_skip"] = {"reason": str(exc)}
                return None
        elif kind == "RESOLVE":
            receipt = record.get("note_receipt", {})
            if not receipt.get("resolve_allowed"):
                return None
            prompt = row_tools.build_resolve_prompt_v16b(p0(item), receipt["lesson"])
        else:
            prompt = p0(item)
        request = make_request(item, prompt, ("v18-" if is_v18 else "v17-" if is_v17 else "v16b-") + kind.lower(),
            cap=cfg["max_tokens"][kind.lower()], sampling=TRAIN_SAMPLING_V16B, version=version,
            protocol_sha256=protocol["sha256"],
            policy_sha256=sampler.live_registry()[str(version)]["policy_sha256"], engine_step=sampler.engine_steps)
        if kind == "NOTE":
            tokens = prompt_tokens(bundle, request, image)
            if tokens > NOTE_PROMPT_BOUND:
                record["note_skip"] = {"reason": "NOTE_PROMPT_OVER_BOUND", "input_tokens": tokens,
                                       "bound": NOTE_PROMPT_BOUND}
                return None
        record[kind.lower() + "_prompt"] = prompt
        return request

    def complete(item: dict, kind: str, out: dict, record: dict) -> dict:
        out["prompt"] = record[kind.lower() + "_prompt"]
        if kind in {"T1", "RESOLVE"}:
            status, extracted = grade(item, out)
            out.update(status=status, extracted=extracted)
            if grade_v8 is not None:
                raw_status, raw_extracted = grade_v8(item, out)
                out.update(v8_status=raw_status, v8_extracted=raw_extracted)
        if is_v17:
            source = contracts[item["problem_id"]]
            view = record_view(item, record)
            kwargs = {"verifier": source["verifier"], "tokenizer": bundle.processor.tokenizer,
                      "fact_catalog": source["fact_catalog"], "rules": source["rules"]}
            if kind == "T1":
                if is_v18:
                    policy_receipt = v17_rows.t1_policy_receipt_v18(view, gold=item["evaluation"]["gold"],
                        choices=item["evaluation"]["choices"], **kwargs)
                    record["t1_policy_receipt"] = policy_receipt
                    view["t1_policy_receipt"] = policy_receipt
                    reward, reason, pg = (policy_receipt[key] for key in ("reward", "reason", "pg"))
                else:
                    reward, reason, pg = v17_rows.reward_v17(out["status"], out["finish_reason"], out["text"], out["extracted"])
                if not out["generation"]["completion_token_ids"]:
                    reward, reason, pg = 0.0, "EMPTY_RESPONSE", False
                receipt = {"reward": reward, "pg": pg, "reason": reason, "output_sha256": digest(out)}
                if is_v18:
                    receipt["policy_receipt"] = policy_receipt
                receipt["sha256"] = digest(receipt)
                ledger = v17_rows.audit_t1_v17(view, **kwargs)
                admission = v17_rows.verify_solution_v17(view, gold=item["evaluation"]["gold"],
                    choices=item["evaluation"]["choices"], **kwargs)
                return {"reward": reward, "reward_receipt": receipt, "t1_ledger": ledger,
                        "a1_receipt": admission}
            if kind == "OBS":
                receipt = v17_rows.audit_observation_v17(view, verifier=source["verifier"],
                    tokenizer=bundle.processor.tokenizer)
                return {"obs_score": receipt, "obs_reward_receipt": receipt, "obs_ledger": receipt["ledger"]}
            if kind == "NOTE":
                ledgers = [record[key] for key in ("t1_ledger", "obs_ledger") if record.get(key) is not None]
                receipt = v17_rows.verify_note_v17(view, ledgers=ledgers, choices=item["evaluation"]["choices"],
                    gold=item["evaluation"]["gold"], **kwargs)
                return {"note_receipt": receipt}
            if kind == "RESOLVE":
                receipt = v17_rows.verify_solution_v17(view, stage="resolve", choices=item["evaluation"]["choices"],
                    gold=item["evaluation"]["gold"], note_receipt=record.get("note_receipt"), **kwargs)
                return {"resolve_receipt": receipt}
        if kind == "T1":
            reward, reason, pg = row_tools.reward_v16b(out["status"], out["finish_reason"], out["text"], out["extracted"])
            receipt = {"reward": reward, "pg": pg, "reason": reason, "output_sha256": digest(out)}
            return {"reward": reward, "reward_receipt": receipt}
        if kind == "OBS":
            score = row_tools.obs_score(item, item["layer"], out["text"], finish_reason=out["finish_reason"])
            return {"obs_score": score, "obs_reward_receipt": {"reward": score["score"],
                    "pg": out["finish_reason"] in {"stop", "eos"}, "output_sha256": digest(out)}}
        if kind == "NOTE":
            source = contracts[item["problem_id"]]
            receipt = row_tools.verify_note_v16b(record_view(item, record), fact_catalog=source["fact_catalog"],
                rules=source["rules"], choices=item["evaluation"]["choices"], gold=item["evaluation"]["gold"])
            return {"note_receipt": receipt}
        if kind == "RESOLVE":
            source = contracts[item["problem_id"]]
            _, receipt = row_tools.c_row_with_receipt(record_view(item, record), tokenizer=bundle.processor.tokenizer,
                fact_catalog=source["fact_catalog"], rules=source["rules"], choices=item["evaluation"]["choices"],
                gold=item["evaluation"]["gold"], note_receipt=record.get("note_receipt"))
            return {"resolve_receipt": receipt}
        return {}

    def build_rows(kind: str, item: dict, record: dict) -> list[dict]:
        if kind.lower() not in record:
            return []
        view = record_view(item, record)
        out, source = record[kind.lower()], contracts[item["problem_id"]]
        if kind in {"T1", "OBS"}:
            receipt = out["generation"]["baseline_receipt" if kind == "T1" else "obs_baseline_receipt"]
            reward = record["reward_receipt" if kind == "T1" else "obs_reward_receipt"]
            base = receipt["baseline"]
            advantage = reward["reward"] - base if reward["pg"] and reward["reward"] is not None and base is not None else 0.
            rows = [{"schema": v17_rows.ROW_SCHEMA if is_v17 else row_tools.ROW_SCHEMA,
                "row_id": item["problem_id"] + ":" + kind,
                "kind": "RL" if kind == "T1" else "OBS", "problem_id": item["problem_id"], "source": item["source"],
                "subject": view["subject"], "layer": item.get("layer"), "block_index": view["block_index"],
                "image_paths": item["image_paths"], "image_sha256": item["image_sha256"],
                "route": "FULL" if item["image_paths"] else "LM", "prompt": out["prompt"], "target": out["text"],
                "generation": out["generation"], "reward": reward["reward"], "baseline": base,
                "advantage": advantage, "pg": reward["pg"], "baseline_receipt": receipt, "complete": True,
                "policy_version": out["generation"]["policy_version"], "status": out.get("status", "OBS"),
                "mixed_version": out["generation"].get("mixed_version", False)}]
            if is_v17:
                from .audit import text_sha256
                row = rows[0]
                row.update(row_id=item["problem_id"] + ":" + row["kind"],
                    parent_id=item["problem_id"], reason=reward.get("reason"), reward_receipt=reward,
                    generation_prompt_sha256=text_sha256(out["prompt"]),
                    training_prompt_sha256=text_sha256(out["prompt"]),
                    admission_receipt_sha256=reward["sha256"],
                    verifier_receipt_sha256=record["t1_ledger"]["sha256"] if kind == "T1" else reward["sha256"],
                    loss_probability_space="processed-full-vocab-v1",
                    loss_token_mask=[True] * len(out["generation"]["completion_token_ids"]),
                    verified_content_token_mask=[], knowledge_status="ANSWER_REWARD" if kind == "T1" else "BOUNDED_OBSERVATION")
                if is_v18 and kind == "T1":
                    policy = record["t1_policy_receipt"]
                    row.update(loss_token_mask=policy["loss_token_mask"],
                               t1_policy_receipt=policy, format_receipt=policy.get("format_receipt"))
                    row.update({key: policy[key] for key in ("pg_mask_reason", "pg_byte_spans", "policy_mask_receipt")
                                if key in policy})
                elif is_v18 and not row["pg"]:
                    # An unscored v18 observation (abstained, unit mismatch, not stopped) has no policy term; its
                    # mask must exclude every action (v18 pod 2026-11-19 stopped at the first RL block on one).
                    row["loss_token_mask"] = [False] * len(out["generation"]["completion_token_ids"])
                elif reward.get("reason") == "LOOP_CONFIRMED":
                    row["loss_token_mask"] = v17_rows.loop_token_mask_v17(bundle.processor.tokenizer,
                        out["generation"]["completion_token_ids"], out["text"])
                if kind == "T1":
                    admitted = v17_rows.a1_row_v17(view, tokenizer=bundle.processor.tokenizer,
                        receipt=record["a1_receipt"])
                    if admitted is not None:
                        rows.append(admitted)
            return rows
        if is_v17:
            if kind == "NOTE":
                sft_row = v17_rows.n_row_v17(view, tokenizer=bundle.processor.tokenizer, receipt=record["note_receipt"])
            else:
                sft_row = v17_rows.c_row_v17(view, tokenizer=bundle.processor.tokenizer, receipt=record["resolve_receipt"])
            return [sft_row] if sft_row is not None else []
        kwargs = {"tokenizer": bundle.processor.tokenizer, "fact_catalog": source["fact_catalog"],
            "rules": source["rules"], "choices": item["evaluation"]["choices"], "gold": item["evaluation"]["gold"]}
        if kind == "NOTE":
            legacy_row = row_tools.note_row(view, **kwargs)
        else:
            legacy_row, receipt = row_tools.c_row_with_receipt(view, **kwargs)
            if "resolve_receipt" in record and receipt != record["resolve_receipt"]:
                raise ValueError("RESOLVE_RECEIPT_MISMATCH")
        return [legacy_row] if legacy_row is not None else []

    def baseline(item: dict, kind: str, history: list[dict], cutoff: int) -> dict:
        if is_v18 and kind == "OBS":
            from .v16b_runtime import observation_baseline_v18
            return observation_baseline_v18(item["layer"], history, history_cutoff=cutoff, prior=prior)
        return (row_tools.baseline_receipt_v16b(item, history, prior=prior, history_cutoff=cutoff) if kind == "T1" else
                row_tools.obs_baseline_v16b(item["layer"], history, history_cutoff=cutoff, prior=prior))

    def rl_update(v: int, rows: list[dict], supplied: dict) -> dict:
        contract = v16b_contract_for_update(protocol, cfg["prep_updates"] + v, phase="rl",
            policy_sha256=policy_sha256(bundle), expected_problem_ids=supplied["expected_problem_ids"],
            expected_obs_problem_ids=supplied["expected_obs_problem_ids"], note_problem_ids=supplied["note_problem_ids"],
            resolve_problem_ids=supplied["resolve_problem_ids"], baseline_receipts=supplied["baseline_receipts"])
        contract["reward_values"] = {}
        for kind, ids in (("RL", contract["expected_problem_ids"]), ("OBS", contract["expected_obs_problem_ids"])):
            for pid in ids:
                receipt = runtime.records[pid]["reward_receipt" if kind == "RL" else "obs_reward_receipt"]
                contract["reward_values"][kind + ":" + pid] = {"reward": receipt["reward"], "pg": receipt["pg"]}
        if supplied.get("external_hold"):
            contract["external_hold"] = supplied["external_hold"]
        return update(rows, contract)

    def check_guard(v: int, stats: dict) -> list[str]:
        state["block_seconds"].append(time.monotonic() - starts[0])
        starts[0] = time.monotonic()
        cursor.update(policy_version=v + 1, rl_block=min(v + 1, len(plan["blocks"])),
                      phase="rl", flush=v >= len(plan["blocks"]))
        reasons = []
        if v < len(plan["blocks"]):
            items = []
            flags = []
            for pid in plan["blocks"][v]:
                out, item = runtime.records[pid]["t1"], all_items[pid]
                items.append({"id": pid, "source": item["source"], "correct": out["status"] == "CORRECT", "truncated": out["finish_reason"] == "length",
                    "no_final": out["extracted"] is None and out["finish_reason"] != "length", "tokens": len(out["generation"]["completion_token_ids"]),
                    "f0_tokens": (item.get("f0_prior") or {}).get("completion_tokens")})
                flags.extend(out["generation"].get("b2_tail_flags", []))
            resolved = [flag for flag in flags if flag is not None]  # None = unresolved BF16 boundary tie
            report = guard.update(v + 1, items, tail_rate=sum(resolved) / len(resolved) if resolved else None,
                                 is_hold=bool(stats.get("HOLD", stats.get("hold", False))))
            reasons.extend(report["reasons"])
            write_json(output / "guard.json", report | {"tail_flags": {
                "resolved": len(resolved), "unresolved": len(flags) - len(resolved)}})
        reasons.extend(_stop_markers(output))
        if recovery_hook is not None:
            recovery_hook.poll()
        if is_v18b and execution_settings is not None:
            from .v18b_execution import _epoch, remaining, enforce_limits, planning_status
            write_json(output / "execution-planning.json", planning_status(execution_settings))
            cutoff = execution_settings["training_cutoff_seconds"]
            if cutoff is not None and time.time() >= _epoch(execution_settings["first_billed_at"]) + cutoff:
                reasons.append("TRAINING_CUTOFF")
            if enforce_limits(execution_settings) and remaining(execution_settings, reserve_seconds=execution_settings["post_eval_reserve_seconds"]) <= 0:
                reasons.append("RUN_DEADLINE")
        if is_v18b:
            eta = eta_v18b(state["block_seconds"], done=v + 1, total=len(plan["blocks"]) + 3)
            eta.update(pending_requests=len(set(runtime.requests) - set(runtime.closed)),
                       delayed_closed_outputs=len(runtime.pending_descriptors()))
            projection_path = output / "mmmu-projection-block-16.json"
            if projection_path.exists():
                eta["evaluation_projection"] = read_sealed(projection_path)
            write_json(output / "eta.json", eta)
        elif v == 3:
            write_json(output / "eta.json", eta_report(state["block_seconds"][:4], done=4, total=len(plan["blocks"]) + 3))
        if reasons:
            state.update(status="STOPPED", stop_reasons=sorted(set(reasons)))
            cursor["phase"] = "stopped"
        elif v == len(plan["blocks"]) + 2:
            cursor["phase"] = "final"
        return reasons

    def finish_signal_gate(v: int, envelope: dict, saved: dict) -> dict:
        if is_v18b:
            from .v18b_monitor import monitor_comparison, short_monitor_comparison, monitor_stop_decision, supply_report
            supported = signal_gate.supported
            closed = [pid for pid in supported if runtime._closed_stage(signal_gate.positions[pid], "NOTE")
                      and runtime._closed_stage(signal_gate.positions[pid], "RESOLVE")]
            stage_opportunities = {stage: [pid for pid in supported if any(c.get("verdict") == "CONTRADICTED"
                for c in (runtime.records.get(pid, {}).get(field) or {}).get("claims", []))]
                for stage, field in (("T1", "t1_ledger"), ("OBS", "obs_ledger"))}
            gate = signal_gate.evaluate(v, closed, stage_opportunities)
            state["signal_gate"] = copy.deepcopy(signal_gate.state)
            gate_path = output / "signal-reports" / f"v18b-gate-slot-{v}.json"
            seal(gate_path, {"version": "v18b-signal-gate-result-v1", "slot": v,
                "plan_sha256": plan["sha256"], "checkpoint": state["checkpoint"], "gate": gate,
                "gate_stops_training": gate["stop_requested"]})
            write_json(output / "signal-gate.json", {"version": "v18b-signal-gate-index-v1",
                "latest": descriptor(gate_path)})
            for pid in closed:
                runtime.records[pid]["closed"] = True
            updates = [read_sealed(path) for path in sorted((output / "update-results").glob("slot-*.json"))]
            seal(output / "supply-reports" / f"v18b-slot-{v}.json", {"version": "v18b-supply-v1",
                "plan_sha256": plan["sha256"], "checkpoint": saved,
                "report": supply_report(runtime.records, plan["items"], updates, cursor["consumed_row_ids"], slot=v, block_size=plan["block_size"])})
            done = min(v + 1, len(plan["blocks"]))
            if done in cfg.get("external_monitor_blocks", (8, 16)) and str(done) not in state.get("v18b_monitor_blocks", []):
                short = monitor(f"short-monitor-block-{done}", v + 1, split="short_monitor")
                reasoning = monitor(f"reasoning-monitor-block-{done}", v + 1, split="reasoning_monitor")
                panel_items = monitors + [all_items[r["problem_id"]] for r in generated["reasoning_monitor"]]
                comparison = short_monitor_comparison(generated["short_monitor"] + generated["reasoning_monitor"],
                    short_post["records"] + reasoning_post["records"], short["records"] + reasoning["records"], panel_items)
                seal(output / f"b2-monitor-block-{done}.json", comparison | {
                    "plan_sha256": plan["sha256"], "checkpoint": saved})
                state.setdefault("v18b_monitor_blocks", []).append(str(done))
            if v >= signal_gate.due_slot and not state.get("v18b_accuracy_monitor_done"):
                normal = monitor("monitor-signal-gate", v + 1)
                comparison = monitor_comparison(generated["monitor"], post["records"], normal["records"], monitors)
                decision = monitor_stop_decision(comparison, args.monitor_stop_action, saved,
                    state.get("v18b_normal_checkpoint"))
                state["v18b_accuracy_monitor_done"] = True
                seal(output / "accuracy-monitor-signal-gate.json", comparison | {
                    "plan_sha256": plan["sha256"], "checkpoint": saved})
                if comparison["status"] == "STOP_REQUESTED":
                    state["v18b_monitor_decision"] = decision
                    state.update(status="STOPPED", stop_reasons=["MONITOR_STOP"] + comparison["reasons"])
                    cursor["phase"] = "stopped"
                    saved = commit("stopped", envelope)
                else:
                    state["v18b_normal_checkpoint"] = copy.deepcopy(saved)
                    state["v18b_normal_state"] = {key: copy.deepcopy(state[key]) for key in
                        ("cursor", "checkpoint", "runtime", "probe", "guard")}
            body = {key: value for key, value in state.items() if key != "sha256"}
            state["sha256"] = digest(body)
            write_json(state_path, state)
            return saved | {"stop_reasons": state.get("stop_reasons", [])}
        if signal_gate is None or v < signal_gate.due_slot or signal_gate.state["local_smoke"]:
            return saved
        if signal_gate.state["status"] == "WARMUP":
            closed = [pid for pid in signal_gate.parents if
                      runtime._closed_stage(next(i for i, ids in enumerate(plan["blocks"]) if pid in ids), "NOTE")
                      and runtime._closed_stage(next(i for i, ids in enumerate(plan["blocks"]) if pid in ids), "RESOLVE")]
            opportunities = [pid for pid in signal_gate.parents if any(
                claim.get("verdict") == "CONTRADICTED"
                for name in ("t1_ledger", "obs_ledger")
                for claim in (runtime.records[pid].get(name) or {}).get("claims", []))]
            gate = signal_gate.evaluate(v, closed, opportunities)
            if gate["status"] == "CHECKPOINT_REQUIRED":
                raise RuntimeError("v17 signal gate has uncommitted content gradients")
            state["signal_gate"] = copy.deepcopy(signal_gate.state)
            state["signal_gate_monitor_pending"] = True
            state["signal_gate_monitor_version"] = v + 1
            # [사용자 2026-11-17] "관문은 기록만, 품질 하락 시만 정지": FAIL/INCONCLUSIVE_SIGNAL is recorded, not a stop.
            body = {k: val for k, val in state.items() if k != "sha256"}
            state["sha256"] = digest(body)
            write_json(state_path, state)
        if state.get("signal_gate_monitor_pending"):
            result = monitor("monitor-signal-gate", state["signal_gate_monitor_version"])
            state["signal_gate_monitor_pending"] = False
            overall = (result.get("comparison") or {}).get("overall") or {}
            degraded = monitor_degraded_v17(overall)
            if degraded:
                # The external development monitor is significantly below F0 (2026-11-17 run: 146 -> 123, p 3e-5).
                state.update(status="STOPPED", stop_reasons=["MONITOR_DEGRADED"])
                cursor["phase"] = "stopped"
                saved = commit("stopped", envelope)
            else:
                body = {k: val for k, val in state.items() if k != "sha256"}
                state["sha256"] = digest(body)
                write_json(state_path, state)
            seal(output / "signal-gate.json", {"version": "v17-signal-gate-result-v2",
                "plan_sha256": plan["sha256"], "checkpoint": state["checkpoint"],
                "gate": signal_gate.state, "gate_stops_training": False,
                "monitor_overall": overall, "monitor_degraded": degraded, "length": guard.state})
        return saved | {"stop_reasons": state.get("stop_reasons", [])}

    def save_runtime(v: int, envelope: dict, pending: list) -> dict:
        if sampler.has_unfinished_requests() or envelope["state"]["pending"] != pending:
            raise ValueError("campaign: runtime drain/preserved output manifest differs")
        saved = commit("stopped" if state["status"] == "STOPPED" else "rl", envelope)
        if is_v18:
            updates = [read_sealed(path) for path in sorted((output / "update-results").glob("slot-*.json"))]
            seal(output / "supply-reports" / f"slot-{v}.json", {"version": "v18-fixed-cohort-supply-v1",
                "plan_sha256": plan["sha256"], "checkpoint": saved,
                "report": v18_supply_report(runtime.records, plan["items"], updates, cursor["consumed_row_ids"])})
        result = finish_signal_gate(v, envelope, saved) if is_v17 else saved
        if recovery_hook is not None and ((v + 1) % 16 == 0 or state["status"] == "STOPPED" or cursor["phase"] == "final"
                or (args.local_sequential or rehearsal) and getattr(args, "checkpoint_after_main_blocks", None) == v + 1):
            recovery_hook.checkpoint(state, force=True)
        if rehearsal:
            from .v18b_rehearsal import after_checkpoint
            after_checkpoint(args, state)
        return result

    callbacks = RuntimeCallbacks(make_request=request, complete=complete, rows=build_rows, baseline=baseline,
                                update=rl_update, checkpoint=save_runtime, guard=check_guard,
        **({"checkpoint_due": lambda v: signal_gate is not None and signal_gate.state["status"] == "WARMUP"
            and v >= signal_gate.due_slot or is_v18b and (v + 1 in cfg.get("external_monitor_blocks", (8, 16)) or
            (args.local_sequential or rehearsal) and getattr(args, "checkpoint_after_main_blocks", None) == v + 1)} if is_v17 else {}))
    if state["runtime"] is None and (output / "runtime").exists():
        destination = output / "rolled-back" / f"runtime-{time.time_ns()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output / "runtime"), str(destination))
    runtime = V16bScheduler(sampler, plan=plan, protocol=cfg | {"protocol_sha256": protocol["sha256"]},
        items=plan["items"], grade=None, verify=None, out=output / "runtime", local_sequential=args.local_sequential,
        callbacks=callbacks, restore=state["runtime"] is not None)
    if state["runtime"] is not None:
        runtime.restore_drain_state(checked(state["runtime"]))
        if is_v17:
            saved = finish_signal_gate(runtime.version - 1, json.loads(checked(state["runtime"]).read_text()), state["checkpoint"])
            if saved.get("stop_reasons"):
                selected = finish_partial() if is_v18b else None
                if selected is not None:
                    return selected
                sampler.release()
                return state
    result = run_v16b(bundle, optimizer, scheduler, sampler, plan, out=output / "runtime", protocol=cfg,
        items=plan["items"], callbacks=callbacks, runtime=runtime, local_sequential=args.local_sequential)
    if result["status"] != "COMPLETE" and is_v18b:
        selected = finish_partial()
        if selected is not None:
            return selected
    if result["status"] == "COMPLETE":
        if is_v18b:
            state["status"] = "TRAINED"
            state["sha256"] = digest({k: value for k, value in state.items() if k != "sha256"})
            write_json(state_path, state)
        final_monitor = monitor("monitor-final", result["version"])
        if is_v18b:
            monitor("short-monitor-final", result["version"], split="short_monitor")
            monitor("reasoning-monitor-final", result["version"], split="reasoning_monitor")
        if is_v17:
            from .v17_eval import evaluation_gate
            final_effect = monitor("effect-final", result["version"], split="effect")
            seal(output / "evaluation-admission.json", {"version": "v17-evaluation-admission-v1",
                "plan_sha256": plan["sha256"], "checkpoint": state["checkpoint"],
                "monitor_sha256": final_monitor["sha256"], "effect_sha256": final_effect["sha256"],
                "decision": evaluation_gate(final_monitor["comparison"], final_effect["comparison"])})
        if is_v18:
            closure = runtime.terminal_closure(cursor.get("consumed_row_ids", []))
            seal(output / "terminal-closure.json", {"version": "v18-terminal-closure-v1",
                "plan_sha256": plan["sha256"], "checkpoint": state["checkpoint"], **closure})
        state["status"] = "COMPLETE"
        body = {k: v for k, v in state.items() if k != "sha256"}
        state = body | {"sha256": digest(body)}
        write_json(state_path, state)
        seal(output / "export-ready.json", {"version": plan["version"], "plan_sha256": plan["sha256"],
            "checkpoint": str(_checkpoint_path(state)), "metadata_descriptor": state["checkpoint"],
            "status": "COMPLETE", "auto_evaluate": False})
        if recovery_hook is not None:
            recovery_hook.checkpoint(state, force=True)
    sampler.release()
    return state


def eta_v18b(block_seconds: list[float], *, done: int, total: int) -> dict:
    report = eta_report(block_seconds[-16:], done=done, total=total)
    return report | {"version": "v18b-eta-v1", "recent_slots": min(16, len(block_seconds)),
        "elapsed_main_seconds": sum(block_seconds), "estimate_scope": "remaining main and delayed settlement slots",
        "timing_basis": "guard-to-guard wall time; prior checkpoint and monitor costs already included",
        "current_boundary_accounting": "current boundary save/monitor enters next slot timing",
        "separate_checkpoint_or_monitor_charge_seconds": 0, "evaluation_added_to_training_seconds": False,
        "initial_estimate": done < 16}


def monitor_degraded_v17(overall: dict) -> bool:
    """Quality stop at the signal-gate monitor: fewer correct than F0 on the external development set, paired sign
    test p < 0.05. A missing comparison is not treated as degradation."""
    change, p_value = overall.get("net_change"), overall.get("p_value")
    return isinstance(change, (int, float)) and isinstance(p_value, (int, float)) and change < 0 and p_value < .05


def partial_evaluation_state(plan: dict, state: dict, output: Path) -> dict | None:
    """Select a previously authorized durable boundary without relabeling interrupted training complete."""
    if plan["protocol"]["version"] != "error-note-v18b" or state["status"] != "STOPPED":
        return None
    settings, reasons = plan.get("execution_settings") or {}, state.get("stop_reasons", [])
    if "RUN_DEADLINE" in reasons or any(reason in reasons for reason in ("STOP_REQUESTED", "WATCHDOG_STOP")):
        return None
    if "MONITOR_STOP" in reasons and settings.get("monitor_stop_action") == "evaluate_checkpoint":
        chosen, reason = state.get("v18b_normal_state"), "MONITOR_STOP"
    elif "TRAINING_CUTOFF" in reasons and settings.get("evaluation_strategy") == "training_cutoff":
        chosen, reason = state, "TRAINING_CUTOFF"
    else:
        return None
    if chosen is None:
        return None
    cursor = chosen["cursor"]
    consumed = cursor.get("consumed_problem_ids", [])
    expected = [pid for block in plan["blocks"] for pid in block]
    preparation_only = (not consumed and reason == "MONITOR_STOP" and cursor.get("slot") == len(plan["prep_batches"])
                        and cursor.get("policy_version") == 0 and cursor.get("updates", 0) > 0)
    all_parents_cutoff = (len(consumed) == len(expected) and reason == "TRAINING_CUTOFF"
        and type(cursor.get("slot")) is int
        and len(plan["prep_batches"]) + len(plan["blocks"]) <= cursor["slot"] <= plan["total_slots"])
    if (not 0 <= len(consumed) < len(expected) and not all_parents_cutoff or not consumed and not preparation_only
            or len(set(consumed)) != len(consumed) or consumed != expected[:len(consumed)]):
        return None
    if preparation_only:
        marker = output / "prep-checkpoint.json"
        if not marker.exists() or json.loads(marker.read_text()) != chosen["checkpoint"]:
            raise ValueError("v18b preparation-only evaluation does not bind the completed prep checkpoint")
    metadata = json.loads(checked(chosen["checkpoint"]).read_text())
    transaction = read_sealed(output / "transactions" / ("slot-{}.json".format(cursor["slot"] - 1)))
    if (metadata.get("cursor", {}).get("single") != cursor or metadata.get("step") != cursor["slot"]
            or preparation_only and transaction.get("contract", {}).get("phase") != "prep"
            or transaction["sha256"] != cursor.get("last_transaction_sha256")
            or metadata.get("protocol") != plan["protocol"] or transaction.get("plan_sha256") != plan["sha256"]
            or not preparation_only and chosen.get("runtime") is None):
        raise ValueError("v18b partial selection has no valid checkpoint/update/runtime trace")
    if chosen.get("runtime") is not None:
        checked(chosen["runtime"])
    selection = {"reason": reason, "authorized_action": "evaluate_checkpoint",
        "source_checkpoint_sha256": chosen["checkpoint"]["file_sha256"],
        "consumed_items": len(consumed), "total_items": len(expected), "preparation_only": preparation_only}
    body = {key: value for key, value in state.items() if key != "sha256"}
    body.update({key: copy.deepcopy(chosen[key]) for key in ("cursor", "checkpoint", "runtime", "probe", "guard")})
    body.update(status="PARTIAL_EVALUATION_CANDIDATE", evaluation_selection=selection,
                stopped_training_state_sha256=state["sha256"], stopped_training_checkpoint=state["checkpoint"])
    return body | {"sha256": digest(body)}


def evaluation_state(output: Path) -> dict:
    selected = output / "evaluation-checkpoint-state.json"
    return read_sealed(selected if selected.exists() else output / "campaign-state.json")


def require_complete(output: Path, *, evaluation_decision: Path | None = None,
                     check_evaluation: bool = True) -> dict:
    state = evaluation_state(output)
    partial = state["status"] == "PARTIAL_EVALUATION_CANDIDATE"
    if not partial and (state["status"] != "COMPLETE" or state.get("stop_reasons")):
        raise ValueError("v16b campaign: stopped/watchdog checkpoint requires an explicit evaluation decision")
    plan = read_sealed(output / "campaign-plan.json")
    _plan_check(plan)
    if partial:
        original = read_sealed(output / "campaign-state.json")
        expected = partial_evaluation_state(plan, original, output)
        if expected != state:
            raise ValueError("v18b partial selection changed or lacks preauthorized stopping policy")
        effect = read_sealed(output / "effect-final.json")
        admission = read_sealed(output / "evaluation-admission.json")
        if any(record.get("checkpoint") != state["checkpoint"] or record.get("plan_sha256") != plan["sha256"]
                for record in (effect, admission)) or admission.get("evaluation_selection") != state["evaluation_selection"]:
            raise ValueError("v18b partial monitor/effect/admission binding changed")
    final = read_sealed(output / "monitor-final.json")
    if (state["plan_sha256"] != plan["sha256"] or final.get("plan_sha256") != plan["sha256"]
            or final.get("checkpoint") != state["checkpoint"]
            or not partial and (state["cursor"]["slot"] != plan["total_slots"] or state["cursor"].get("phase") != "final")):
        raise ValueError("v16b campaign: final monitor/checkpoint binding changed")
    if not partial and plan["protocol"]["version"] in ("error-note-v18", "error-note-v18b"):
        closure = read_sealed(output / "terminal-closure.json")
        if (closure.get("plan_sha256") != plan["sha256"] or closure.get("checkpoint") != state["checkpoint"]
                or closure.get("status") != "DRAINED_AND_SETTLED" or closure.get("pending_requests") != 0
                or closure.get("unconsumed_rows") != 0):
            raise ValueError("v18 campaign: terminal closure does not bind the final checkpoint")
    if check_evaluation and plan["protocol"]["version"] == "error-note-v17" and not plan["local_sequential"]:
        admission = read_sealed(output / "evaluation-admission.json")
        if admission["plan_sha256"] != plan["sha256"] or admission["checkpoint"] != state["checkpoint"]:
            raise ValueError("v17 evaluation admission belongs to another checkpoint")
        status = admission["decision"]["status"]
        if status == "DEGRADED":
            raise ValueError("v17 evaluation blocked: external paired degradation")
        if status == "EVAL_REVIEW_REQUIRED":
            if evaluation_decision is None:
                raise ValueError("EVAL_REVIEW_REQUIRED: explicit user decision receipt required")
            decision = read_sealed(evaluation_decision)
            if (decision.get("decision") != "PROCEED" or decision.get("authorized_by") != "user"
                    or decision.get("admission_sha256") != admission["sha256"]
                    or decision.get("plan_sha256") != plan["sha256"]):
                raise ValueError("v17 evaluation decision does not authorize this admission")
        elif status != "EVALUATION_CANDIDATE":
            raise ValueError("v17 evaluation admission is incomplete")
    ready = seal(output / "export-ready.json", {"version": plan["version"], "plan_sha256": plan["sha256"],
        "checkpoint": str(_checkpoint_path(state)), "metadata_descriptor": state["checkpoint"],
        "status": state["status"], "auto_evaluate": False,
        **({"evaluation_selection": state["evaluation_selection"]} if partial else {})})
    checked(ready["metadata_descriptor"])
    return ready


def export_metadata(metadata: dict, plan: dict, checkpoint: Path) -> dict:
    _plan_check(plan)
    if plan["protocol"].get("training_contract", {}).get("rehearsal_profile") is not None:
        raise ValueError("a non-reportable rehearsal cannot be exported for benchmark evaluation")
    if plan["local_sequential"]:
        raise ValueError("v16b campaign: local NF4 smoke cannot be exported for benchmark evaluation")
    cursor = metadata.get("cursor", {}).get("single", {})
    state = evaluation_state(checkpoint.parent)
    partial = state["status"] == "PARTIAL_EVALUATION_CANDIDATE"
    if partial:
        expected = partial_evaluation_state(plan, read_sealed(checkpoint.parent / "campaign-state.json"), checkpoint.parent)
        if state != expected:
            raise ValueError("v18b export selection was not authorized")
    if (state.get("plan_sha256") != plan["sha256"] or state.get("cursor") != cursor
            or state.get("pending") is not None or metadata.get("step") != cursor.get("slot")
            or state.get("checkpoint") != descriptor(checkpoint / "metadata.json")):
        raise ValueError("v16b campaign: export requires the latest committed checkpoint")
    if (metadata.get("protocol") != plan["protocol"] or cursor.get("phase") not in ({"final", "stopped", "rl"} if partial else {"final", "stopped"})
            or not 0 < cursor.get("slot", 0) <= plan["total_slots"]):
        raise ValueError("v16b campaign: export requires a committed final/stopped checkpoint")
    if cursor["phase"] == "final" and cursor["slot"] != plan["total_slots"]:
        raise ValueError("v16b campaign: incomplete final checkpoint")
    return {"version": plan["version"], "plan_sha256": plan["sha256"], "slot": cursor["slot"],
        "updates": cursor["updates"], "total_slots": plan["total_slots"], "stage": "partial" if partial else cursor["phase"],
        **({"hotfix_resumes": state["hotfix_resumes"], "execution_code_commit": state["execution_code_commit"],
            "hotfix_lineage_sha256": state["hotfix_lineage_sha256"]} if state.get("hotfix_resumes") else {}),
        **({"evaluation_selection": state["evaluation_selection"]} if partial else {}),
        "checkpoint": descriptor(checkpoint / "metadata.json"), "prep_updates": len(plan["prep_batches"]),
        "rl_blocks": len(plan["blocks"]), "finish_updates": 3, "local_sequential": plan["local_sequential"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "advance", "reload", "require-complete", "export", "calibrate", "preflight", "training-artifact-seal"))
    for name in ("data-root", "prep-root", "model-path", "output"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    parser.add_argument("--version", choices=("error-note-v16b", "error-note-v17", "error-note-v18", "error-note-v18b"), default="error-note-v16b")
    parser.add_argument("--precision", choices=("nf4", "bf16"), default="bf16")
    parser.add_argument("--engine", choices=("hf", "vllm"), default="vllm")
    parser.add_argument("--local-sequential", action="store_true")
    parser.add_argument("--nf4-cache", type=Path)
    parser.add_argument("--kv-cache-gib", type=float, default=16.)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    parser.add_argument("--token-budget", type=int, default=12000)
    for name in ("allow-model-run", "allow-update", "allow-save", "resume-funding-stop"):
        parser.add_argument("--" + name, action="store_true")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--export-output", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--hotfix-resume", type=Path, help="explicit verified operational hotfix authorization receipt")
    parser.add_argument("--execution-settings", type=Path)
    parser.add_argument("--backup-destination", type=Path)
    parser.add_argument("--backup-transfer-command", help="JSON argv; {archive}/{receipt}/{slot}/{bundle_sha256} placeholders")
    parser.add_argument("--checkpoint-after-main-blocks", type=int, help="bounded local crash-recovery smoke only")
    parser.add_argument("--execution-phase", choices=("legacy", "training", "evaluation"), default="legacy")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument("--training-artifact-seal", "--training-seal", type=Path)
    parser.add_argument("--artifact-verification", type=Path)
    parser.add_argument("--engine-validation-receipt", type=Path)
    parser.add_argument("--engine-validation-receipt-sha256")
    for name in ("calibration", "functional-receipt", "receipt-output", "preflight-output",
                 "hardware", "eval-protocol", "evaluation-decision"):
        parser.add_argument("--" + name, type=Path)
    for name in ("calibration-sha256", "functional-receipt-sha256"):
        parser.add_argument("--" + name)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .v18b_recovery import recovery_paths
    args = build_parser().parse_args(argv)
    with recovery_paths(args.output):
        return _main(args)


def _main(args: argparse.Namespace) -> int:
    if args.checkpoint_after_main_blocks is not None and (not args.local_sequential
            or args.checkpoint_after_main_blocks != 2):
        raise ValueError("only the local two-block crash-recovery checkpoint is supported")
    if args.resume_funding_stop and args.command != "advance":
        raise ValueError("v16b campaign: --resume-funding-stop applies to advance only")
    if args.command == "training-artifact-seal":
        if (args.version not in MODERN_VERSIONS or args.execution_phase != "training"
                or args.export_output is None or (args.version == "error-note-v17" and args.evaluation_decision is None)):
            raise ValueError("training-artifact-seal requires v17 training, export and evaluation decision")
        from .v17_artifacts import create_training_artifacts
        result = create_training_artifacts(args, read_sealed(args.output / "campaign-plan.json"))
        print(json.dumps({"status": result["status"], "sha256": result["sha256"]}))
        return 0
    if args.command in {"require-complete", "export"}:
        if args.version in MODERN_VERSIONS:
            from .v17_preflight import bind_execution, bind_training
            bind = bind_training if args.execution_phase == "training" else bind_execution
            bind(args, read_sealed(args.output / "campaign-plan.json"))
        ready = require_complete(args.output, evaluation_decision=args.evaluation_decision)
        if args.command == "export":
            from fine_tuning.notebook_rl.proof_eval import export_for_eval
            if args.base is None or args.export_output is None or not args.code_commit:
                raise ValueError("v16b export: --base --export-output --code-commit required")
            export_for_eval(Path(ready["checkpoint"]), args.output / "campaign-plan.json", args.base,
                            args.export_output, args.code_commit)
        print(json.dumps({"status": ready.get("status", "COMPLETE"), "checkpoint": ready["checkpoint"]}))
        return 0
    if args.command == "preflight" and args.execution_phase == "evaluation":
        from .v17_preflight import trained_export_preflight
        result = trained_export_preflight(args, read_sealed(args.output / "campaign-plan.json"))
        print(json.dumps({"status": result["status"]}))
        return 0
    if any(getattr(args, name) is None for name in ("data_root", "prep_root", "model_path")):
        raise ValueError("v16b campaign: --data-root --prep-root --model-path required")
    with single._runtime_lock(args.output, ".v16b-campaign.lock"):
        plan = prepare(args)
        if args.command == "calibrate":
            if args.version not in MODERN_VERSIONS:
                raise ValueError("calibrate requires explicit v17")
            from .v17_preflight import calibrate
            result = calibrate(args, plan)
        elif args.command == "preflight":
            if args.version not in MODERN_VERSIONS:
                raise ValueError("preflight requires explicit v17")
            from .v17_preflight import trained_export_preflight
            result = trained_export_preflight(args, plan)
        else:
            result = {"status": "PREPARED"} if args.command == "prepare" else run_campaign(
                plan, output=args.output, args=args, reload_only=args.command == "reload")
    print(json.dumps({"status": result["status"], "cursor": result.get("cursor")}))
    return 0 if result["status"] in {"PREPARED", "COMPLETE", "PARTIAL_EVALUATION_CANDIDATE", "RELOADED", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
