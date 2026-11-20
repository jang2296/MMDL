"""Explicit non-reportable BF16/vLLM campaigns through the production callbacks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from typing import Any

from .audit import digest
from .prep_v16b import read_sealed, seal

PROFILES = {
    "tiny-bf16": {"blocks": 8, "block_size": 1, "max_tokens": 32,
                  "max_num_seqs": 2, "kv_cache_gib": .25, "monitor_blocks": [8]},
    "a100-bf16": {"blocks": 2, "block_size": 2, "max_tokens": 128,
                  "max_num_seqs": 4, "kv_cache_gib": 2., "monitor_blocks": [1, 2]},
}


def tiny_rehearsal_transport(report: dict) -> None:
    """Retain raw PF5 numerics while admitting only fixed-fixture lifecycle checks."""
    import math
    from . import llm_lora

    numeric = report["pf5"]
    values = [numeric[key] for key in ("mean_abs_on", "mean_abs_off_vs_on", "amplified_mean_abs_on")]
    values += [value for family in ("real", "amplified") for fit in numeric.get(family, {}).values()
               for key, value in fit.items() if key in {"rms_effect", "slope", "correlation"}]
    failures = []
    if numeric["mean_abs_on"] > numeric["cap"]:
        failures.append({"criterion": "on_error_at_most_cap", "actual": numeric["mean_abs_on"], "maximum": numeric["cap"]})
    if numeric.get("stage_contract", {}).get("require_power") and not numeric["power"]:
        failures.append({"criterion": "trained_total_effect_power", "actual": False, "minimum_rms": llm_lora.MIN_EFFECT})
    if numeric.get("real", {}).get("total", {}).get("rms_effect", 0) >= llm_lora.MIN_EFFECT and numeric["mean_abs_on"] >= numeric["mean_abs_off_vs_on"]:
        failures.append({"criterion": "on_error_below_off_error", "on": numeric["mean_abs_on"], "off": numeric["mean_abs_off_vs_on"]})
    for component, fit in numeric.get("amplified", {}).items():
        if fit.get("gating") and fit.get("pass") is False:
            failures.append({"criterion": "amplified_" + component, "actual": fit,
                             "slope_range": llm_lora.SLOPE_RANGE, "minimum_correlation": llm_lora.MIN_CORRELATION})
    tower = numeric.get("vision_tower") or {}
    if tower.get("pass") is False:
        failures.append({"criterion": "direct_vision_numeric", "actual": tower})
    report.update(validation_scope="TINY_REHEARSAL_TRANSPORT", production_admitted=False,
                  numerical_status="UNCONFIRMED", failed_numeric_checks=failures)
    report["pass"] = (all(report[key]["pass"] is True for key in ("pf1", "pf2", "pf3", "pf2_restored"))
        and report["pf4"]["pass"] is not False and all(math.isfinite(value) for value in values)
        and (tower.get("hook_removed") or {}).get("pass") is not True)


def profile_settings(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError("rehearsal needs an explicit tiny-bf16 or a100-bf16 profile")
    return deepcopy(PROFILES[name])


def rehearsal_protocol(precision: str, *, rl_blocks: int, prep_updates: int,
                       contracts: dict, image: dict | None, local_smoke: bool, profile: str) -> dict:
    from .protocol import plain_protocol_v18b, SINGLE_IMAGE_V14

    limits = profile_settings(profile)
    if precision != "bf16" or local_smoke or rl_blocks != limits["blocks"] or prep_updates != 1:
        raise ValueError("rehearsal profile requires its exact BF16 schedule and one preparation update")
    chosen = (dict(SINGLE_IMAGE_V14) if profile == "a100-bf16" else
              {"min_pixels": 3136, "max_pixels": 12544, "preprocessing": SINGLE_IMAGE_V14["preprocessing"]})
    if image is not None and image != chosen:
        raise ValueError("rehearsal image policy changed")
    body = plain_protocol_v18b("bf16", rl_blocks=188, prep_updates=1, contracts=contracts)
    body.pop("sha256")
    total = prep_updates + rl_blocks + 3
    contract = body["training_contract"]
    contract.update(rehearsal_profile=profile, reportable=False, cloud_allowed=False,
        rl_blocks=rl_blocks, rl_items=rl_blocks * limits["block_size"], items_per_block=limits["block_size"],
        prep_updates=prep_updates, total_slots=total, schedule_steps=total,
        external_monitor_blocks=limits["monitor_blocks"], monitor_items=2)
    contract["schedule"].update(rl_blocks=rl_blocks, last_block_items=limits["block_size"])
    contract["max_tokens"] = {key: limits["max_tokens"] for key in contract["max_tokens"]}
    contract["max_tokens"]["monitor"] = 64
    body.update(image=chosen, max_new_tokens=limits["max_tokens"], reportable=False)
    body["optimizer"].update(total_steps=total, warmup_steps=1)
    return body | {"sha256": digest(body)}


def validate_plan(plan: dict) -> None:
    from .protocol import validate_protocol_v18b

    protocol = plan["protocol"]
    validate_protocol_v18b(protocol)
    profile = protocol["training_contract"]["rehearsal_profile"]
    settings = profile_settings(profile)
    ids = [pid for block in plan["blocks"] for pid in block]
    cfg = protocol["training_contract"]
    if (plan.get("reportable") is not False or plan.get("auto_evaluate") is not False
            or plan.get("version") != "single-campaign-v18b-rehearsal-v1"
            or plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"})
            or plan.get("local_sequential") is not False or plan.get("execution_settings") is not None
            or plan.get("time_cut") is not None or len(plan["blocks"]) != settings["blocks"]
            or any(len(block) != settings["block_size"] for block in plan["blocks"])
            or ids != [item["problem_id"] for item in plan["items"]] or len(ids) != len(set(ids))
            or len(plan["prep_batches"]) != 1 or plan["total_slots"] != cfg["total_slots"]
            or plan["capacity"]["engine"] != "vllm"
            or plan["capacity"]["max_num_seqs"] != settings["max_num_seqs"]):
        raise ValueError("invalid non-reportable rehearsal campaign")


def validate_execution(args: Any, plan: dict) -> dict:
    from .single import source_manifest

    validate_plan(plan)
    current = source_manifest()
    if (getattr(args, "rehearsal_profile", None) != plan["protocol"]["training_contract"]["rehearsal_profile"]
            or args.engine != "vllm" or args.precision != "bf16" or args.local_sequential):
        raise ValueError("rehearsal source, profile or engine changed")
    for name in ("model_path", "data_root", "prep_root"):
        if Path(getattr(args, name)).resolve() != Path(plan[name]).resolve():
            raise ValueError("rehearsal bound input path changed: " + name)
    if current["actual_files"] != plan["source"]["actual_files"]:
        from .v18b_hotfix import execution_source
        execution_source(args, plan, current)
    return {"status": "REHEARSAL_ONLY", "reportable": False, "source_sha256": digest(current["actual_files"])}


def validate_inputs(plan: dict) -> dict:
    """Recheck the immutable, fully validated fixture inputs without rescanning unused images."""
    from .campaign_v16b import checked

    value = read_sealed(checked(plan["inputs"]["rehearsal_data_validation"]))
    if value["reportable"] is not False:
        raise ValueError("rehearsal validation lacks fixture provenance")
    for ref in value["files"]:
        checked(ref)
    return value["validation"]


def after_checkpoint(args: Any, state: dict) -> None:
    if not getattr(args, "exercise_recovery", False) or state["cursor"].get("policy_version") != 2:
        return
    marker = Path(args.output) / "rehearsal-killed-checkpoint.json"
    if marker.exists():
        return
    seal(marker, {"reportable": False, "producer_pid": os.getpid(), "checkpoint": state["checkpoint"],
                  "cursor": state["cursor"], "action": "SELF_SIGKILL_AFTER_DURABLE_BOUNDARY"})
    os.kill(os.getpid(), signal.SIGKILL)


def before_final_monitor(args: Any, plan: dict, label: str) -> None:
    marker = Path(args.output) / "rehearsal-final-interruption.json"
    if (plan["protocol"]["training_contract"].get("rehearsal_profile") == "tiny-bf16"
            and getattr(args, "exercise_recovery", False) and label == "monitor-final"
            and not marker.exists()):
        state = read_sealed(Path(args.output) / "campaign-state.json")
        seal(marker, {"reportable": False, "checkpoint": state["checkpoint"], "cursor": state["cursor"],
                      "status": state["status"], "reason": "DELIBERATE_FINAL_MONITOR_FAILURE"})
        raise RuntimeError("rehearsal deliberate final-monitor interruption")


def policy_factory(model: Path, arm: str, protocol: dict):
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    import torch
    from fine_tuning.notebook_rl.policy import configure_policy, build_policy

    profile = protocol["training_contract"].get("rehearsal_profile")
    if profile == "a100-bf16":
        return build_policy(model, arm, protocol)
    if profile != "tiny-bf16":
        raise ValueError("fixture policy requires an explicit tiny rehearsal protocol")
    config = AutoConfig.from_pretrained(str(model), local_files_only=True)
    if (config.text_config.num_hidden_layers != 2 or config.vision_config.depth != 3
            or config.text_config.hidden_size != 256 or config.vision_config.hidden_size != 128):
        raise ValueError("tiny rehearsal requires the fixed two-layer synthetic model")
    base = Qwen3VLForConditionalGeneration.from_pretrained(str(model), dtype=torch.bfloat16,
                                                         local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True,
        min_pixels=protocol["image"]["min_pixels"], max_pixels=protocol["image"]["max_pixels"])
    return configure_policy(base, processor, arm, protocol, enforce_counts=False)


def live_preflight(args: Any, plan: dict, bundle: Any, sampler: Any) -> dict:
    """Exercise both real engine registrations; full-model maximum-shape admission is separate."""
    validate_execution(args, plan)
    sampler.add_version(bundle, 0)
    sampler.add_version(bundle, 1)
    transport = sampler.preflight_report
    coexistence = sampler.coexistence_reports
    if (not transport or not coexistence or transport.get("pass") is not True
            or any(row.get("pass") is not True for row in coexistence)):
        raise ValueError("rehearsal transport did not pass")
    bounds = {"max_images": 4, "max_vision_patches": 60000, "max_prompt_tokens": 16384,
              "dataset_sha256": plan["inputs"]["payload"]["file_sha256"]}
    capacity = {"shape_bounds": bounds, "reportable": False, "maximum_shape_verified": False}
    pid = os.getpid()
    return seal(Path(args.output) / "training-bindings" / f"rehearsal-live-{pid}.json", {
        "status": "REHEARSAL_ONLY", "reportable": False, "producer_pid": pid, "capacity": capacity,
        "transport": transport, "coexistence": coexistence,
        "full_model_memory_admission": "NOT_MEASURED_BY_THIS_BOUNDED_REHEARSAL"})


def make_plan(args: Any) -> dict:
    """Use pinned source rows and production collection/admission with a bounded schedule."""
    from . import campaign_v16b as campaign, prep_v16b as prep, v18b_rows as rows
    from .protocol import ARM, plain_protocol_v18b, V16B_CONTRACT_HASHES, EVAL_SAMPLING_V16B
    from .v16b_rows import prep_prior_v16b
    from .v18b_data import load_items, validate, data_files

    path = args.output / "campaign-plan.json"
    if path.exists():
        plan = read_sealed(path)
        validate_execution(args, plan)
        return plan
    args.output.mkdir(parents=True, exist_ok=True)
    args.prep_root.mkdir(parents=True, exist_ok=True)
    validation = validate(args.data_root)
    settings = profile_settings(args.rehearsal_profile)
    all_train = rows.bind_graph_observation_targets_v18b(load_items(args.data_root, "train"))
    supported = [item for item in all_train if rows.observation_eligibility_v17(item)["eligible"]]
    count = settings["blocks"] * settings["block_size"]
    items = supported[:count]
    if len(items) != count:
        raise ValueError("rehearsal requires enough pinned observation parents")
    prep_candidates = [item for item in load_items(args.data_root, "prep") if item.get("human_solution")][:4]
    if not prep_candidates:
        raise ValueError("rehearsal preparation requires dataset-native human candidates")
    roles = {role: load_items(args.data_root, role)[:2]
             for role in ("monitor", "effect", "reasoning_monitor")}
    metadata = [args.data_root / key for key in data_files(args.data_root) if not key.endswith(".png")]
    used = items + prep_candidates + [item for group in roles.values() for item in group]
    input_paths = set(metadata + [Path(path) for item in used for path in item["image_paths"]])
    input_paths.add(args.data_root / "payload-receipt.json")
    input_paths.update(path for path in args.model_path.iterdir() if path.is_file())
    seal(args.output / "rehearsal-data-validation.json", {"reportable": False, "validation": validation,
        "files": [campaign.descriptor(path) for path in sorted(input_paths)]})
    selection = seal(args.prep_root / "selection.json", {
        "version": "v18b-rehearsal-selection-v1", "reportable": False,
        "prep_ids": [item["problem_id"] for item in prep_candidates],
        "monitor_ids": [item["problem_id"] for item in roles["monitor"]],
        "reasoning_monitor_ids": [item["problem_id"] for item in roles["reasoning_monitor"]],
        "excluded_ceiling_monitor_ids": [], "frozen_role_ids": {},
        "effect_ids": [item["problem_id"] for item in roles["effect"]]})
    protocol = plain_protocol_v18b("bf16", rl_blocks=settings["blocks"], prep_updates=1,
        contracts={key: "0" * 64 for key in V16B_CONTRACT_HASHES}, rehearsal_profile=args.rehearsal_profile)
    bundle = policy_factory(args.model_path, ARM, protocol)
    bundle.model.to("cuda:0")
    sampler = campaign.make_sampler(bundle, args, protocol, image=protocol["image"], sampling=EVAL_SAMPLING_V16B)
    try:
        def collect(members, split, label, *, cap):
            return prep.collect(bundle, sampler, members, split=split, out=args.prep_root / label,
                stage=prep.preparation_stage("error-note-v18b", label), sampling=EVAL_SAMPLING_V16B,
                data_root=args.data_root, protocol_sha256=protocol["sha256"],
                cap=cap, algorithm_version="error-note-v18b")

        prep_records = collect(prep_candidates, "prep", "prep", cap=settings["max_tokens"])
        records = [dict(item, p0_text=record["prompt"], t1=record["output"])
                   for item, record in zip(prep_candidates, prep_records, strict=True)]
        accepted, receipts = rows.prep_rows_with_receipts_v17(records, tokenizer=bundle.processor.tokenizer)
        human = [row for row in accepted if row["kind"] == "PREP_HUMAN"]
        if not human:
            raise ValueError("rehearsal has no admitted dataset-native preparation row")
        generated = {"version": "v18b-rehearsal-f0-v1", "reportable": False,
            "selection_sha256": selection["sha256"], "prep": prep_records, "obs": []}
        for role, cap in (("monitor", 64), ("effect", 64), ("reasoning_monitor", 64)):
            generated[role] = collect(roles[role], role, role, cap=cap)
            generated[role + "_summary"] = prep.summary(generated[role])
        generated["short_monitor"] = collect(roles["monitor"], "monitor", "short_monitor", cap=64)
        generated["short_monitor_summary"] = prep.summary(generated["short_monitor"])
        generated = seal(args.prep_root / "f0-generations.json", generated)
        admitted = seal(args.prep_root / "prep-rows.json", {"version": "v18b-rehearsal-prep-v1",
            "rows": human[:1], "batches": [human[:1]], "prep_updates": 1,
            "selection_sha256": selection["sha256"], "f0_generations_sha256": generated["sha256"],
            "local_sequential": False, "reportable": False, "receipts": receipts})
        prior = prep_prior_v16b(records)
        seal(args.prep_root / "baseline-init.json", {"version": "v18b-rehearsal-baseline-v1",
            "prep_rows_sha256": admitted["sha256"], "source_rates": prior["source_rates"],
            "overall": prior["overall"], "layer_rates": {}, "prep_prior": prior,
            "missing_obs_policy": "binary-task-beta-prior-0.5", "obs_task": "v18-single-fact-observation-v1"})
    finally:
        sampler.release_idle_versions()
    preflight = seal(args.output / "data-contract-preflight.json", prep.preflight_items(items, version="error-note-v18b"))
    paths = {"payload": args.data_root / "payload-receipt.json", "order": args.data_root / "order.json",
             "prep_rows": args.prep_root / "prep-rows.json", "baseline": args.prep_root / "baseline-init.json",
             "f0": args.prep_root / "f0-generations.json", "selection": args.prep_root / "selection.json",
             "data_contract_preflight": args.output / "data-contract-preflight.json",
             "rehearsal_data_validation": args.output / "rehearsal-data-validation.json"}
    inputs = {key: campaign.descriptor(value) for key, value in paths.items()}
    source = campaign.single.source_manifest()
    contracts = {"source": digest(source["actual_files"]), "dataset": inputs["payload"]["file_sha256"],
                 "order": inputs["order"]["file_sha256"], "prep_rows": inputs["prep_rows"]["file_sha256"],
                 "monitor": digest(generated["monitor"])}
    protocol = plain_protocol_v18b("bf16", rl_blocks=settings["blocks"], prep_updates=1,
        contracts=contracts, rehearsal_profile=args.rehearsal_profile)
    size = settings["block_size"]
    blocks = [[item["problem_id"] for item in items[index:index + size]] for index in range(0, len(items), size)]
    from .guard_v16b import SignalGateV18B
    supported_ids = [row["problem_id"] for row in preflight["items"] if row.get("repair_supported")]
    gate = SignalGateV18B(blocks, supported_ids, local_smoke=True)
    plan = seal(path, {"version": "single-campaign-v18b-rehearsal-v1", "reportable": False,
        "protocol": protocol, "source": source, "inputs": inputs, "data_root": str(args.data_root.absolute()),
        "prep_root": str(args.prep_root.absolute()), "model_path": str(args.model_path.absolute()), "nf4_cache": None,
        "local_sequential": False, "items": items, "blocks": blocks, "block_size": size,
        "observation_eligibility": {row["problem_id"]: row["observation_eligibility"] for row in preflight["items"]},
        "prep_batches": [human[:1]], "total_slots": protocol["training_contract"]["total_slots"],
        "time_cut": None, "execution_settings": None, "capacity": {"engine": "vllm",
            "max_num_seqs": settings["max_num_seqs"], "kv_cache_gib": args.kv_cache_gib,
            "gpu_memory_utilization": args.gpu_memory_utilization, "token_budget": args.token_budget},
        "recovery": {"destination": str((args.output.parent / "rehearsal-backups").absolute()),
                     "transfer_command": None, "every_main_slots": 16}, "auto_evaluate": False,
        "signal_gate": {"supported_parent_ids": supported_ids, "parents": gate.parents, "due_slot": gate.due_slot}})
    validate_plan(plan)
    return plan


def mechanical_note_update(args: Any) -> dict:
    """A declared synthetic vLLM-shaped note tests the actual GPU loss and durable reload."""
    from . import campaign_v16b as campaign, single, rl
    from .protocol import ARM, v18b_contract_for_update
    from fine_tuning.notebook_rl import checkpoint
    from fine_tuning.tests.test_error_note_v18b_row_quarantine import mechanical_note_row

    plan = read_sealed(args.output / "campaign-plan.json")
    validate_execution(args, plan)
    protocol = plan["protocol"]
    bundle = policy_factory(args.model_path, ARM, protocol)
    bundle.model.to("cuda:0")
    block = protocol["training_contract"]["rl_blocks"]
    row, item = mechanical_note_row(bundle, "mechanical-note-fixture", block_index=block - 1)
    encoded, quarantine = campaign._encode_update_rows(bundle, [row], {item["problem_id"]: item}, protocol["image"])
    if len(encoded) != 1 or quarantine["rejected_rows"]:
        raise RuntimeError("mechanical note failed real row admission")
    optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
    slot = protocol["training_contract"]["prep_updates"] + block
    # A mechanical fixture starts at the delayed-note flush, with no preceding gradient updates.
    for _ in range(slot):
        optimizer.step()
        scheduler.step()
    cursor = {"slot": slot, "updates": 0, "consumed_problem_ids": [], "consumed_row_ids": [],
              "policy_sha256": rl.policy_sha256(bundle), "phase": "rl", "policy_version": block, "rl_block": block}
    definition = v18b_contract_for_update(protocol, slot, phase="rl", policy_sha256=rl.policy_sha256(bundle),
        expected_problem_ids=[], note_problem_ids=[item["problem_id"]])
    result = single.run_joint_single_update(bundle, optimizer, scheduler, encoded, definition,
        cursor=cursor, block_ids=[], allow_update=True, token_budget=12000)
    stats = result["stats"]
    if (result["optimizer_steps"] != 1 or stats.get("n_content_gradient_finite_nonzero") is not True
            or row["row_id"] not in stats.get("consumed_row_ids", [])):
        raise RuntimeError("mechanical note did not contribute actual GPU update")
    destination = args.output / "mechanical-note-checkpoint"
    single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=destination,
        protocol=protocol, binding=campaign._binding(plan), allow_save=True)
    return seal(args.output / "mechanical-note-receipt.json", {"status": "PASS", "reportable": False,
        "origin": "SYNTHETIC_VLLM_SHAPED_MECHANICAL_FIXTURE", "natural_correction_count": 0,
        "empty_scheduler_offset_slots": slot, "actual_gradient_updates": result["optimizer_steps"],
        "source_generation_sha256": digest(row["generation"]), "note_gradient": stats.get("n_parent_evidence"),
        "checkpoint": campaign.descriptor(destination / "metadata.json"), "cursor": cursor})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "advance", "reload", "mechanical-note", "run"))
    p.add_argument("--profile", dest="rehearsal_profile", choices=tuple(PROFILES), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-seconds", type=int, default=1500)
    p.add_argument("--exercise-recovery", action="store_true")
    p.add_argument("--hotfix-resume", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--code-commit")
    return p


def arguments(ns: Any) -> Any:
    profile = profile_settings(ns.rehearsal_profile)
    maximum = 1800 if ns.rehearsal_profile == "tiny-bf16" else 1500
    if type(ns.deadline_seconds) is not int or not 0 < ns.deadline_seconds <= maximum:
        raise ValueError("rehearsal deadline exceeds its bounded local/A100 profile")
    return argparse.Namespace(**vars(ns), version="error-note-v18b", precision="bf16", engine="vllm",
        local_sequential=False, nf4_cache=None, kv_cache_gib=profile["kv_cache_gib"],
        gpu_memory_utilization=.35 if ns.rehearsal_profile == "tiny-bf16" else .65,
        token_budget=12000, allow_model_run=True, allow_update=True, allow_save=True,
        prep_root=ns.output / "preparation", execution_phase="rehearsal",
        checkpoint_after_main_blocks=2, resume_funding_stop=False)


def run_child(command: list[str], *, env: dict, stream: Any, deadline: float) -> subprocess.CompletedProcess:
    from .v18b_smoke import process_identity, signal_owned

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("rehearsal exhausted its single deadline")
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    identity = process_identity(process.pid)
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=remaining))
    finally:
        if process.poll() is None:
            signal_owned(process, identity, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                signal_owned(process, identity, signal.SIGKILL)
                process.wait(timeout=2)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = arguments(parser().parse_args(argv))
    if args.command == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "fine_tuning.error_note.v18b_rehearsal", "prepare", "--profile", args.rehearsal_profile,
            "--data-root", str(args.data_root), "--model-path", str(args.model_path), "--output", str(args.output)]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        for phase in ("prepare", "mechanical-note", "advance", "reload"):
            command[3] = phase
            remaining = args.deadline_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("rehearsal exhausted its single deadline")
            if args.exercise_recovery and "--exercise-recovery" not in command:
                command.append("--exercise-recovery")
            with (args.output / (phase + ".log")).open("a") as stream:
                result = run_child(command, stream=stream, env=env, deadline=started + args.deadline_seconds)
                if phase == "advance" and args.exercise_recovery:
                    if result.returncode != -signal.SIGKILL or not (args.output / "rehearsal-killed-checkpoint.json").exists():
                        raise RuntimeError("rehearsal did not reach its durable crash boundary")
                    result = run_child(command, stream=stream, env=env, deadline=started + args.deadline_seconds)
                    if result.returncode != 1 or not (args.output / "rehearsal-final-interruption.json").exists():
                        raise RuntimeError("rehearsal did not reach its final-monitor interruption")
                    result = run_child(command, stream=stream, env=env, deadline=started + args.deadline_seconds)
                    state = read_sealed(args.output / "campaign-state.json")
                    before = read_sealed(args.output / "rehearsal-killed-checkpoint.json")
                    ids = state["cursor"]["consumed_row_ids"]
                    if (len(ids) != len(set(ids)) or not set(before["cursor"]["consumed_row_ids"]) <= set(ids)
                            or state["cursor"]["producer_pid"] == before["producer_pid"]):
                        raise RuntimeError("rehearsal crash replay duplicated or lost committed learning")
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, command)
        from .v18b_rehearsal_eval import run_rehearsal_evaluation
        evaluation = run_rehearsal_evaluation(campaign_output=args.output, model_path=args.model_path,
            data_root=args.data_root, output=args.output.parent / "rehearsal-evaluation",
            mode=args.rehearsal_profile, deadline_monotonic=started + args.deadline_seconds)
        receipt = seal(args.output / "rehearsal-receipt.json", {"status": "PASS", "reportable": False,
            "profile": args.rehearsal_profile, "wall_seconds": time.monotonic() - started,
            "mechanical_note": read_sealed(args.output / "mechanical-note-receipt.json"), "evaluation": evaluation})
    else:
        from . import campaign_v16b as campaign
        from .run_v16 import _setup_execution
        import torch

        torch.set_num_threads(4)
        _setup_execution(3407)
        if args.command == "mechanical-note":
            receipt = mechanical_note_update(args)
        elif args.command == "prepare":
            receipt = make_plan(args)
        else:
            plan = read_sealed(args.output / "campaign-plan.json")
            receipt = campaign.run_campaign(plan, output=args.output, args=args, reload_only=args.command == "reload")
    print(json.dumps({key: receipt.get(key) for key in ("status", "reportable", "wall_seconds")}))
    return 0 if args.command == "prepare" or receipt.get("status") in {"PASS", "COMPLETE", "RELOADED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
