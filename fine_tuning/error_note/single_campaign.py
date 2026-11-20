"""Bounded SINGLE campaign execution; prepare never loads a model.

The fixed 300-item pilot and remaining 3,200 items share one BF16 optimizer.
Local NF4 supports no-update proofs only. Rows are admitted by program checks (SINGLE v3).
Only checkpoint-0, the pilot checkpoint and the newest two checkpoints are kept on disk.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable

from . import references, single
from .audit import digest, text_sha256
from .collect_v3 import file_sha, generation_identity
from .contract import CONTEXT_LIMIT, MASTER_SEED
from .protocol import (ARM, ROLLOUT_ADAPTER_V15, SINGLE_TOKEN_SCALE_V11, VERSION_SINGLE_V12, VERSION_SINGLE_V14,
                       VERSION_SINGLE_V15, plain_protocol_single, single_blocks)

VERSION = "single-campaign-v2"
GATE_VERSION = "single-pilot-main-gate-v1"
GATE_RULE = "program-functional-all;missing=0;failures=0;dev>F0;retention>=F0;no-final<=F0;cap<=F0;cost<=budget"
# --direct-main (v5, 2026-11-06): the user's decision replaces the external-dev pilot gate; the pilot checkpoint is
# still committed and reloaded in a fresh process before the main blocks run.
DIRECT_GATE_VERSION = "single-direct-user-gate-v1"
CAMPAIGN_INPUTS = ("order", "frozen_approval", "source_provenance", "training_rules", "note_reference_supplement")
DIRECT_DECISION = ("user 2026-11-06: after the stopped v4 main run, train all 3,500 items without the external-dev "
                   "pilot comparison, then evaluate MMMU-val 900 once")
# Warm start (v11c, 2026-11-10): a new plan that differs from a stopped campaign plan only in the code-source hash seals
# one checkpoint of it as its starting point; the record replaces that slot's predecessor transaction/metadata checks.
WARM_START_VERSION = "single-warm-start-v1"
WARM_START_DECISION = ("user 2026-11-10: continue the stopped SINGLE v11b campaign from its slot-156 checkpoint (stopped "
                       "by the preregistered FORMAT_DRIFT rule) on the remaining blocks with the unchanged v11b training "
                       "method; the slot-155 predecessor metadata was not retrieved before the source pod was deleted")
WARM_START_SAME = ("items", "blocks", "data_admission", "inputs", "build_root", "model_path", "capacity", "gate_rule",
                   "pilot_ids_sha256", "dev_ids_sha256", "retention_ids_sha256", "execution_max_blocks", "mode")
WARM_START_ARGS = ("warm_start_plan", "warm_start_plan_sha256", "warm_start_checkpoint", "warm_start_checkpoint_sha256",
                   "warm_start_reload", "warm_start_reload_sha256")
WARM_START_KEYS = {"version", "decision", "source_plan", "source_plan_sha256", "source_binding", "checkpoint", "slot",
                   "updates", "cursor_sha256", "reload_receipt"}
# Warm start v2 (v15, 2026-11-14): the one sealed protocol transition. Only these protocol paths may differ from the
# stopped v14 plan; WARM_START_SAME (incl. execution_max_blocks, so the 64-step LR schedule) stays identical.
WARM_START_VERSION_V2 = "single-warm-start-v2"
WARM_START_DECISION_V2 = ("user 2026-11-14: v15 continues the stopped SINGLE v14 campaign from its slot-41 checkpoint "
    "(fresh-process reload receipt slot 41) on blocks 41-63 of the same 1,020-item order with the v15 changes (vision "
    "route on every image row, up to 3 lesson re-solves, unmerged vLLM LoRA rollouts); the LR keeps v14's 64-step schedule")
WARM_START_TRANSITION_V2 = {"from": VERSION_SINGLE_V14, "to": VERSION_SINGLE_V15, "protocol_changes": [
    "rollout_adapter", "training_contract.generation_counts", "training_contract.hashes.source",
    "training_contract.resolve_attempts", "training_contract.routes", "training_contract.version", "version"]}


def _descriptor(path: Path) -> dict:
    return {"path": str(path.absolute()), "file_sha256": file_sha(path)}


def _checked(descriptor: dict) -> Path:
    path = Path(descriptor["path"])
    if path.is_symlink() or not path.is_file() or file_sha(path) != descriptor["file_sha256"]:
        raise ValueError("single campaign: artifact identity changed")
    return path


def _plan_check(plan: dict) -> None:
    if plan.get("version") == CONTENT_CAMPAIGN_VERSION:
        _content_plan_check(plan)
        return
    if plan.get("version") in ("single-campaign-v16b", "single-campaign-v17"):
        from .campaign_v16b import _plan_check as _plan_check_v16b
        _plan_check_v16b(plan)
        return
    single._protocol_check(plan["protocol"])
    ids = [item["problem_id"] for item in plan["items"]]
    if (plan.get("version") != VERSION or plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"})
            or plan["mode"] not in ("proof", "train") or plan["blocks"] != single_blocks(ids)
            or plan["protocol"]["training_contract"]["total_slots"] != len(plan["blocks"])
            or plan["gate_rule"] != GATE_RULE):
        raise ValueError("single campaign: invalid immutable campaign plan")
    if plan["protocol"]["version"] in (VERSION_SINGLE_V14, VERSION_SINGLE_V15) and not (
            plan["protocol"]["optimizer"]["total_steps"] == plan["protocol"]["training_contract"]["schedule_steps"]
            == plan.get("execution_max_blocks")):
        # v14/v15: the LR decays over exactly the blocks the plan may run (a bool or None never equals the int steps).
        raise ValueError("single campaign: invalid immutable campaign plan")
    if plan["mode"] == "train" and plan["protocol"]["precision"] != "bf16":
        raise ValueError("single campaign: HOLD NF4 training/checkpoint continuation is not validated")
    warm = plan.get("warm_start")
    v2 = warm is not None and warm.get("version") == WARM_START_VERSION_V2
    if warm is not None and (
            set(warm) != WARM_START_KEYS | ({"transition"} if v2 else set())
            or warm["version"] not in (WARM_START_VERSION, WARM_START_VERSION_V2)
            or warm["decision"] != (WARM_START_DECISION_V2 if v2 else WARM_START_DECISION)
            or (v2 and (warm["transition"] != WARM_START_TRANSITION_V2
                        or plan["protocol"]["version"] != WARM_START_TRANSITION_V2["to"]))
            or plan["mode"] != "train"
            or type(warm["slot"]) is not int or not 1 <= warm["slot"] < len(plan["blocks"])
            or type(warm["updates"]) is not int or warm["updates"] != warm["slot"]
            or set(warm["source_binding"]) != {"protocol_sha256", "prompts_sha256", "data_sha256", "code_sha256"}
            or not all(_hex64(value) for value in (warm["source_plan"]["file_sha256"], warm["source_plan_sha256"],
                       warm["checkpoint"]["file_sha256"], warm["cursor_sha256"], warm["reload_receipt"]["file_sha256"],
                       *warm["source_binding"].values()))):
        raise ValueError("single campaign: invalid warm-start record")


def _hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


CONTENT_CAMPAIGN_VERSION = "single-campaign-content-v16"
CONTENT_GATE_RULE = "verified_local_functional_before_cloud;measured_BF16_O_effect_before_phase2"


def _content_blocks(ids: list[str], block_size: int) -> list[list[str]]:
    if (type(block_size) is not int or not 1 <= block_size <= 16 or not ids
            or len(set(ids)) != len(ids) or any(not isinstance(pid, str) or not pid for pid in ids)):
        raise ValueError("v16 campaign: unique parent IDs and a block size in 1..16 required")
    return [ids[start:start + block_size] for start in range(0, len(ids), block_size)]


def _content_runtime_gate(plan: dict, descriptor: dict | None, *, checkpoint_sha256: str) -> dict:
    from .run_v16 import validate_bf16_effect_receipt

    if descriptor is None:
        raise ValueError("v16 campaign: EFFECT_NOTRUN; a sealed BF16 O-effect receipt is required")
    receipt = single._read(_checked(descriptor))
    contract = plan["protocol"]["training_contract"]
    result = validate_bf16_effect_receipt(receipt, expected_binding={
        "source_sha256": contract["hashes"]["source"], "dataset_sha256": contract["hashes"]["dataset"],
        "f0_sha256": plan["generation"]["model_files_sha256"], "checkpoint_sha256": checkpoint_sha256})
    if result.get("passed") is not True:
        raise ValueError("v16 campaign: EFFECT_STOP; the measured O-effect gate did not pass")
    return result


def _content_functional_gate(plan: dict) -> dict:
    from .run_v16 import validate_local_functional_receipt

    descriptor = plan.get("functional_gate")
    if descriptor is None:
        raise ValueError("v16 campaign: FUNCTIONAL_NOTRUN; a verified local training/reload receipt is required")
    contract = plan["protocol"]["training_contract"]
    result = validate_local_functional_receipt(single._read(_checked(descriptor)), expected_binding={
        "source_sha256": contract["hashes"]["source"], "dataset_sha256": contract["hashes"]["dataset"]})
    if result.get("ready") is not True:
        raise ValueError("v16 campaign: local functional training/reload proof did not pass")
    return result


def _content_grade(item: dict, actor: Any, text: str, finish: str) -> tuple[str, str | None]:
    from fine_tuning.notebook_rl.answers import compare, extract_eval_aligned
    from fine_tuning.notebook_rl.schema import VerifierItem

    kind = item.get("answer_type", "mcq" if actor.choices else "short_text")
    extracted = extract_eval_aligned(text, finish, actor.question, actor.choices, kind)
    if extracted.status != "OK":
        return extracted.status, extracted.answer
    target = VerifierItem(problem_id=item["problem_id"], answer_type=kind, gold=item["gold"],
        accept=item.get("accept", {}), choices=actor.choices, slots={}, slot_types={}, required_slots=(),
        oracle_fact_lines=(), printed_text=())
    correct = compare(extracted.answer, target)
    return ("CORRECT" if correct else "INCORRECT") if correct is not None else "UNVERIFIABLE", extracted.answer


def _content_plan_check(plan: dict) -> None:
    single._protocol_check(plan["protocol"])
    ids = [item["problem_id"] for item in plan["items"]]
    contract = plan["protocol"]["training_contract"]
    if (plan.get("version") != CONTENT_CAMPAIGN_VERSION
            or plan.get("sha256") != digest({key: value for key, value in plan.items() if key != "sha256"})
            or plan.get("mode") not in ("proof", "train") or plan.get("gate_rule") != CONTENT_GATE_RULE
            or plan.get("blocks") != _content_blocks(ids, plan["block_size"])
            or contract["total_slots"] != len(plan["blocks"])
            or plan.get("execution_max_blocks") != len(plan["blocks"]) or plan.get("warm_start") is not None
            or contract["hashes"]["source"] != digest(plan["source"]["actual_files"])
            or contract["hashes"]["dataset"] != plan["inputs"]["content_manifest"]["file_sha256"]
            or contract["hashes"]["order"] != digest(ids)
            or contract["hashes"]["data_admission"] != digest(plan["data_admission"])):
        raise ValueError("v16 campaign: invalid immutable content campaign")
    if (contract["stage_a_blocks"] < contract["total_slots"]
            and any(contract["weights"][kind] > 0 for kind in ("J", "A1", "C", "N"))
            and contract["stage_a_blocks"] != 16):
        raise ValueError("v16 campaign: phase2 requires exactly sixteen O-only blocks for its checkpoint-bound effect gate")
    if plan["mode"] == "train" and plan["protocol"]["precision"] != "bf16":
        raise ValueError("single campaign: HOLD NF4 training/checkpoint continuation is not validated")


def _prepare_content_v16(args: argparse.Namespace) -> dict:
    from .content_v16 import load_content_manifest, validate_human_j_capability
    from .protocol import plain_protocol_content_v16

    manifest = load_content_manifest(args.content_manifest, args.content_manifest_sha256)
    items = [item for item in manifest["items"] if item["split"] == "train"]
    ids = [item["problem_id"] for item in items]
    blocks = _content_blocks(ids, args.block_size)
    if getattr(args, "content_total_blocks", None) is not None and args.content_total_blocks != len(blocks):
        raise ValueError("v16 campaign: actual parent manifest differs from declared block count")
    if (args.functional_gate is None) != (args.functional_gate_sha256 is None):
        raise ValueError("v16 campaign: a functional receipt requires its external SHA256")
    source = single.source_manifest()
    admission = {"schema": manifest["schema"], "manifest_sha256": args.content_manifest_sha256,
                 "train_count": len(items), "parent_ids_sha256": digest(ids)}
    contracts = {"source": digest(source["actual_files"]), "dataset": args.content_manifest_sha256,
                 "order": digest(ids), "data_admission": digest(admission)}
    layers = sorted({row["layer"] for row in manifest["rows"]})
    source_weights = {layer: 1 / len(layers) for layer in layers}
    weights = json.loads(args.content_weights) if getattr(args, "content_weights", None) else None
    protocol = plain_protocol_content_v16(args.precision, len(blocks), contracts=contracts,
                                         source_weights=source_weights, weights=weights,
                                         stage_a_blocks=getattr(args, "stage_a_blocks", 0))
    validate_human_j_capability(manifest, requested_weight=protocol["training_contract"]["weights"]["J"])
    config = single._read(args.model_path / "config.json")
    if (args.model_path / "nf4_base.json").exists() or config.get("quantization_config") is not None:
        raise ValueError("v16 campaign: the original BF16 F0 snapshot is required")
    generation = generation_identity(SimpleNamespace(engine="hf", model_path=args.model_path,
                                    code_commit=source["base_commit"])) | {"precision": args.precision}
    capacity = {"max_num_seqs": args.max_num_seqs or 1,
                "max_num_batched_tokens": args.max_num_batched_tokens or CONTEXT_LIMIT,
                "token_budget": args.token_budget if args.token_budget is not None else 0,
                "context_limit": CONTEXT_LIMIT, "engine": args.engine}
    if (type(capacity["max_num_seqs"]) is not int or not 1 <= capacity["max_num_seqs"] <= 16
            or type(capacity["max_num_batched_tokens"]) is not int or capacity["max_num_batched_tokens"] < 1
            or type(capacity["token_budget"]) is not int or capacity["token_budget"] < 0):
        raise ValueError("v16 campaign: invalid generation/training capacity")
    if args.engine == "vllm":
        if (args.precision != "bf16" or isinstance(args.kv_cache_gib, bool)
                or not isinstance(args.kv_cache_gib, (int, float)) or not math.isfinite(args.kv_cache_gib)
                or args.kv_cache_gib <= 0 or not 0 < args.gpu_memory_utilization < 1):
            raise ValueError("v16 campaign: invalid BF16 vLLM capacity")
        capacity |= {"kv_cache_gib": args.kv_cache_gib, "gpu_memory_utilization": args.gpu_memory_utilization,
                     "vllm_sleep": args.vllm_sleep}
    body = {"version": CONTENT_CAMPAIGN_VERSION, "mode": args.campaign_mode, "items": items, "blocks": blocks,
            "block_size": args.block_size, "protocol": protocol, "source": source, "generation": generation,
            "data_admission": admission, "inputs": {"content_manifest": _descriptor(args.content_manifest)},
            "content_rows": manifest["rows"], "capacity": capacity, "gate_rule": CONTENT_GATE_RULE,
            "build_root": str(args.content_manifest.parent.absolute()), "model_path": str(args.model_path.absolute()),
            "execution_max_blocks": len(blocks),
            "functional_gate": _descriptor(args.functional_gate) if args.functional_gate else None,
            "runtime_status": {"effect": "NOTRUN", "note_yield": "NOTRUN"}}
    if args.functional_gate is not None:
        _checked({"path": str(args.functional_gate), "file_sha256": args.functional_gate_sha256})
    plan = body | {"sha256": digest(body)}
    _content_plan_check(plan)
    single._sealed(args.output / "campaign-plan.json", plan)
    return plan


def _strip(protocol: dict, paths: list[str]) -> dict:
    """The protocol without its sha256 and without each dotted path (when present)."""
    stripped = copy.deepcopy(protocol)
    del stripped["sha256"]
    for path in paths:
        *parents, leaf = path.split(".")
        node = stripped
        for key in parents:
            node = node.get(key, {})
        node.pop(leaf, None)
    return stripped


def warm_start_record(body: dict, *, plan_path: Path, plan_sha256: str, checkpoint_path: Path, checkpoint_sha256: str,
                      reload_path: Path, reload_sha256: str) -> dict:
    """Seal a stopped campaign's committed checkpoint (``checkpoint_path`` is its metadata.json) as the plan's start.

    Same protocol version: v1 (only the code-source hash may differ). v14 -> v15 only: v2, where exactly the
    WARM_START_TRANSITION_V2 protocol paths may differ; any other version pair is refused.
    """
    from fine_tuning.notebook_rl import checkpoint

    old = single._read(_checked({"path": str(plan_path), "file_sha256": plan_sha256}))
    _plan_check(old)
    versions = (old["protocol"]["version"], body["protocol"]["version"])
    transition = None
    if versions[0] != versions[1]:
        if versions != (WARM_START_TRANSITION_V2["from"], WARM_START_TRANSITION_V2["to"]):
            raise ValueError(f"single campaign: no sealed protocol transition {versions[0]} -> {versions[1]}")
        transition = copy.deepcopy(WARM_START_TRANSITION_V2)
    paths = transition["protocol_changes"] if transition else ["training_contract.hashes.source"]
    missing = object()
    if (old["mode"] != "train" or any(old.get(key, missing) != body.get(key, missing) for key in WARM_START_SAME)
            or old["generation"]["model_files_sha256"] != body["generation"]["model_files_sha256"]
            or _strip(old["protocol"], paths) != _strip(body["protocol"], paths)):
        raise ValueError("single campaign: warm start needs a stopped train plan with the same items, blocks and protocol")
    metadata = checkpoint.read_metadata(_checked({"path": str(checkpoint_path), "file_sha256": checkpoint_sha256}).parent,
                                        binding=_binding(old))
    cursor = metadata["cursor"].get("single", {})
    slot = cursor.get("slot")
    if (metadata["protocol"] != old["protocol"] or type(slot) is not int or not 1 <= slot < len(body["blocks"])
            or type(cursor.get("updates")) is not int or cursor["updates"] != slot
            or cursor.get("consumed_problem_ids") != [pid for block in body["blocks"][:slot] for pid in block]
            or not _hex64(cursor.get("last_transaction_sha256"))):
        raise ValueError("single campaign: warm-start checkpoint is not a committed prefix of this plan")
    if single._read(_checked({"path": str(reload_path), "file_sha256": reload_sha256})) != {
            "status": "RELOADED", "slot": slot, "updates": slot}:
        raise ValueError("single campaign: warm-start checkpoint lacks its fresh-process reload receipt")
    record = {"version": WARM_START_VERSION, "decision": WARM_START_DECISION, "source_plan": _descriptor(plan_path),
              "source_plan_sha256": old["sha256"], "source_binding": _binding(old), "checkpoint": _descriptor(checkpoint_path),
              "slot": slot, "updates": slot, "cursor_sha256": digest(cursor), "reload_receipt": _descriptor(reload_path)}
    if transition is not None:
        record |= {"version": WARM_START_VERSION_V2, "decision": WARM_START_DECISION_V2, "transition": transition}
    return record


def prepare(args: argparse.Namespace) -> dict:
    if getattr(args, "version", None) in ("error-note-v16b", "error-note-v17", "error-note-v18", "error-note-v18b"):
        from .campaign_v16b import prepare as _prepare_v16b
        return _prepare_v16b(args)
    with single._runtime_lock(args.output, ".campaign.lock"):
        return _prepare(args)


def _prepare(args: argparse.Namespace) -> dict:
    """Validate all actual selected 3,500 items and seal the immutable campaign."""
    if getattr(args, "content_manifest", None) is not None:
        return _prepare_content_v16(args)
    warm = [getattr(args, name, None) for name in WARM_START_ARGS]
    if None in warm and any(value is not None for value in warm):
        raise ValueError("single campaign: warm start needs its plan, checkpoint and reload receipt with external SHA256s")
    required = (*CAMPAIGN_INPUTS, "build_root", "pilot_ids_sha256", "dev_ids_sha256", "retention_ids_sha256")
    if any(getattr(args, key, None) is None for key in required):
        raise ValueError("single campaign: all legacy source/order/admission inputs are required")
    from fine_tuning.notebook_rl.schema import to_json
    from fine_tuning.notebook_rl.vault import read_actors, read_lineage, read_verifier
    from .answer_contract import VERSION as RULES_VERSION, load_training_rules
    from .capacity import load_profile
    from .frozen_data import verify_frozen_data
    from .order import read_order
    from .source_evidence import VERSION as SOURCE_VERSION, _rule

    if args.campaign_mode == "train" and args.precision != "bf16":
        raise ValueError("single campaign: HOLD NF4 training/checkpoint continuation is not validated")
    if type(args.execution_max_blocks) is not int or not 1 <= args.execution_max_blocks <= 219:
        raise ValueError("single campaign: preregistered execution block limit must be in1..219")
    for name in ("pilot_ids_sha256", "dev_ids_sha256", "retention_ids_sha256"):
        value = getattr(args, name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("single campaign: fixed pilot/dev/retention SHA256 registrations required")
    for key in CAMPAIGN_INPUTS:
        _checked({"path": str(getattr(args, key)), "file_sha256": getattr(args, key + "_sha256")})
    approval = single._read(args.frozen_approval)
    order = read_order(args.order, expected_sha256=args.order_sha256)
    admission = verify_frozen_data(args.build_root, approval, args.order, order_sha256=args.order_sha256,
                                   expected_pilot_sha256=args.pilot_ids_sha256)
    actors = {actor.problem_id: actor for actor in read_actors(args.build_root / "train")}
    private = read_verifier(args.build_root / "train")
    lineages = read_lineage(args.build_root / "train")
    provenance = single._read(args.source_provenance)
    sources = provenance["items"]
    # The supplement is verified by its file sha only: its annotations hash values with text_sha256, not digest.
    supplement = references.read_supplement(args.note_reference_supplement, args.note_reference_supplement_sha256)
    if not supplement.keys() <= {item["problem_id"] for item in order["items"]}:
        raise ValueError("single campaign: reference supplement names an item outside the order")
    rules = single._read(args.training_rules)
    train_sha = file_sha(args.build_root / "train" / "manifest.yaml")
    if (len(order["items"]) != 3500 or provenance.get("version") != SOURCE_VERSION
            or rules.get("version") != RULES_VERSION or rules.get("vault_manifest_sha256") != train_sha):
        raise ValueError("single campaign: actual frozen3500 source/rules binding required")
    checked_rules = load_training_rules(args.training_rules, args.training_rules_sha256,
        vault_root=args.build_root / "train", actors=actors, items=private, lineages=lineages)
    for item in order["items"]:
        pid = item["problem_id"]
        actor, source = actors.get(pid), sources.get(pid, {})
        if (actor is None or item["source"] != actor.source or source.get("source") != actor.source
                or source.get("actor_sha256") != text_sha256(to_json(actor))
                or source.get("image_sha256") != list(actor.image_sha256)
                or source.get("vault_manifest_sha256") != train_sha or source.get("split") != "train"
                or source.get("source_revision") != lineages[pid].source_revision
                or source.get("label", {}).get("value") != private[pid].gold
                or source.get("label", {}).get("sha256") != text_sha256(private[pid].gold)):
            raise ValueError("single campaign: source/actor/image binding differs in the full3500 order")
        annotations = source.get("annotations", [])
        if any(reference.get("sha256") != digest(reference.get("value")) for reference in annotations):
            raise ValueError("single campaign: native annotation/unit evidence hash changed")
        expected_rule, unit_policy = _rule(actor, private[pid], lineages[pid], source, annotations, source)
        if checked_rules.get(pid) != expected_rule or source.get("unit_policy") != unit_policy:
            raise ValueError("single campaign: original gold/tolerance/native unit policy changed")
        for relative, expected in zip(actor.image_paths, actor.image_sha256, strict=True):
            _checked({"path": str(args.build_root / "images" / relative), "file_sha256": expected})
    model_config = single._read(args.model_path / "config.json")
    if (args.model_path / "nf4_base.json").exists() or model_config.get("quantization_config") is not None:
        raise ValueError("single campaign: raw BF16 source snapshot required; no precision fallback")
    source = single.source_manifest()
    generation = generation_identity(SimpleNamespace(engine="hf", model_path=args.model_path,
                                                     code_commit=source["base_commit"])) | {"precision": args.precision}
    blocks = single_blocks([item["problem_id"] for item in order["items"]])
    contracts = {"source": digest(source), "dataset": file_sha(args.build_root / "manifest.yaml"),
                 "order": args.order_sha256, "data_admission": digest(admission)}
    note_reference = references.note_reference_pin(sources, supplement, args.note_reference_supplement_sha256,
                                                   [item["problem_id"] for item in order["items"]])
    # New plans are SINGLE v15 (2026-11-14), as v14 replaced v13; sealed v11-v14 plans still pass _plan_check.
    # The sealed order still holds all 219 blocks; the LR decays over the execution_max_blocks the plan may run.
    protocol = plain_protocol_single(args.precision, len(blocks), contracts=contracts, token_scale=args.token_scale,
                                     note_reference=note_reference, version=VERSION_SINGLE_V15,
                                     schedule_steps=args.execution_max_blocks)
    measured = args.capacity_profile is not None
    if measured != (args.capacity_profile_sha256 is not None):
        raise ValueError("single campaign: profile requires an external SHA256")
    capacity = {"max_num_seqs": args.max_num_seqs if args.max_num_seqs is not None else (16 if measured else 1),
                "max_num_batched_tokens": args.max_num_batched_tokens if args.max_num_batched_tokens is not None else
                    (16 * CONTEXT_LIMIT if measured else CONTEXT_LIMIT),
                "token_budget": args.token_budget if args.token_budget is not None else (16 * CONTEXT_LIMIT if measured else 0),
                "context_limit": CONTEXT_LIMIT, "engine": getattr(args, "engine", "hf")}
    if capacity["engine"] == "vllm":
        capacity |= {"kv_cache_gib": args.kv_cache_gib, "gpu_memory_utilization": args.gpu_memory_utilization,
                     "vllm_sleep": args.vllm_sleep}
        if (args.precision != "bf16" or not isinstance(args.kv_cache_gib, (int, float)) or not args.kv_cache_gib > 0
                or not isinstance(args.gpu_memory_utilization, float) or not 0 < args.gpu_memory_utilization < 1
                or type(args.vllm_sleep) is not bool):
            raise ValueError("single campaign: vLLM generation needs BF16, positive KV GiB and a utilization in (0,1)")
    elif capacity["engine"] != "hf":
        raise ValueError("single campaign: engine must be hf or vllm")
    if (type(capacity["max_num_seqs"]) is not int or not 1 <= capacity["max_num_seqs"] <= 16
            or type(capacity["max_num_batched_tokens"]) is not int or capacity["max_num_batched_tokens"] < 1
            or type(capacity["token_budget"]) is not int or capacity["token_budget"] < 0):
        raise ValueError("single campaign: invalid generation/training capacity")
    if measured:
        profile = load_profile(args.capacity_profile, args.capacity_profile_sha256)
        identity = {"source_sha256": contracts["source"], "model_sha256": generation["model_files_sha256"],
                    "precision": args.precision}
        if any(profile["binding"].get(key) != value for key, value in identity.items()):
            raise ValueError("single campaign: stale measured capacity identity")
        capacity["profile"] = _descriptor(args.capacity_profile)
    body = {"version": VERSION, "mode": args.campaign_mode, "items": order["items"], "blocks": blocks,
            "protocol": protocol, "source": source, "generation": generation, "data_admission": admission,
            "inputs": {key: _descriptor(getattr(args, key)) for key in CAMPAIGN_INPUTS},
            "build_root": str(args.build_root.absolute()), "model_path": str(args.model_path.absolute()),
            "capacity": capacity, "gate_rule": GATE_RULE, "pilot_ids_sha256": args.pilot_ids_sha256,
            "dev_ids_sha256": args.dev_ids_sha256, "retention_ids_sha256": args.retention_ids_sha256,
            "execution_max_blocks": args.execution_max_blocks}
    if warm[0] is not None:
        body["warm_start"] = warm_start_record(body, plan_path=args.warm_start_plan,
            plan_sha256=args.warm_start_plan_sha256, checkpoint_path=args.warm_start_checkpoint,
            checkpoint_sha256=args.warm_start_checkpoint_sha256, reload_path=args.warm_start_reload,
            reload_sha256=args.warm_start_reload_sha256)
    plan = body | {"sha256": digest(body)}
    _plan_check(plan)
    single._sealed(args.output / "campaign-plan.json", plan)
    return plan


def _binding(plan: dict) -> dict:
    return {"protocol_sha256": plan["protocol"]["sha256"], "prompts_sha256": plan["protocol"]["prompts_sha256"],
            "data_sha256": plan["sha256"], "code_sha256": (digest(plan["source"]["actual_files"]) if plan.get("version") == CONTENT_CAMPAIGN_VERSION
                               else digest(plan["source"]))}


def _cursor(plan: dict, cursor: dict, policy: str) -> None:
    slot = cursor.get("slot")
    if (type(slot) is not int or not 0 <= slot <= len(plan["blocks"])
            or cursor.get("consumed_problem_ids") != [pid for block in plan["blocks"][:slot] for pid in block]
            or cursor.get("policy_sha256") != policy or type(cursor.get("updates")) is not int
            or not 0 <= cursor["updates"] <= slot):
        raise ValueError("single campaign: stale policy, skipped/duplicate/reordered continuation prefix")


def _commit_check(plan: dict, output: Path, cursor: dict, descriptor: dict) -> None:
    """The successor must bind the exact reviewed rows and its immutable predecessor."""
    from fine_tuning.notebook_rl import checkpoint

    metadata = checkpoint.read_metadata(_checked(descriptor).parent, binding=_binding(plan))
    if metadata["cursor"].get("single") != cursor:
        raise ValueError("single campaign: checkpoint cursor changed")
    warm = plan.get("warm_start")
    if warm is not None and cursor["slot"] == warm["slot"]:
        # The sealed warm-start record stands in for the source campaign's transaction and parent checkpoint.
        if digest(cursor) != warm["cursor_sha256"]:
            raise ValueError("single campaign: warm-start checkpoint cursor changed")
        return
    if cursor["slot"]:
        transaction = single._read(output / f"transaction-{cursor['slot'] - 1}.json")
        parent = transaction["parent_checkpoint"]
        if (cursor.get("last_transaction_sha256") != digest(transaction)
                or cursor.get("parent_checkpoint") != parent or transaction["plan_sha256"] != plan["sha256"]
                or transaction["ids"] != plan["blocks"][cursor["slot"] - 1]
                or transaction["slot"] != cursor["slot"] - 1):
            raise ValueError("single campaign: checkpoint transaction/parent mismatch")
        previous = checkpoint.read_metadata(_checked(parent).parent, binding=_binding(plan))
        if (previous["cursor"]["single"]["slot"] != cursor["slot"] - 1
                or previous["cursor"]["single"]["policy_sha256"] != transaction["policy_sha256"]):
            raise ValueError("single campaign: checkpoint predecessor mismatch")
        closed = single._read(_checked(transaction["rows"]))
        if digest(closed["contract"]) != transaction["contract_sha256"]:
            raise ValueError("single campaign: committed rows/contract changed")


def _rng_save(output: Path, state: dict, phase: dict | None) -> None:
    import torch
    from fine_tuning.notebook_rl import checkpoint

    data = io.BytesIO()
    checkpoint_rng = checkpoint.capture_rng()
    torch.save(checkpoint_rng, data)
    raw = data.getvalue()
    import hashlib

    sha = hashlib.sha256(raw).hexdigest()
    path = output / ("rng-" + sha + ".pt")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(path)
    state["rng"] = {"file": _descriptor(path), "binding_sha256": digest({"cursor": state["cursor"],
                    "checkpoint": state["checkpoint"], "phase_sha256": phase.get("sha256") if phase else None}),
                    "phase_sha256": phase.get("sha256") if phase else None}


def _rng_restore(state: dict, phase: dict | None) -> None:
    import torch
    from fine_tuning.notebook_rl import checkpoint

    if state.get("rng") is None:
        return
    rng = state["rng"]
    if rng["binding_sha256"] != digest({"cursor": state["cursor"], "checkpoint": state["checkpoint"],
                                       "phase_sha256": phase.get("sha256") if phase else None}):
        raise ValueError("single campaign: WAIT phase/policy/checkpoint RNG binding changed")
    checkpoint.restore_rng(torch.load(_checked(rng["file"]), map_location="cpu", weights_only=True))


def _gate(plan: dict, state: dict, descriptor: dict | None, authorized: bool) -> None:
    if authorized is not True or descriptor is None:
        raise ValueError("single campaign: HOLD explicit main continuation and pinned pilot/dev/budget gate required")
    if "raw" in descriptor:
        import hashlib
        import json

        if hashlib.sha256(descriptor["raw"]).hexdigest() != descriptor["file_sha256"]:
            raise ValueError("single campaign: preloaded private gate bytes changed")
        gate = json.loads(descriptor["raw"])
    else:
        gate = single._read(_checked(descriptor))
    metrics = gate.get("metrics", {})
    f0, hyb = metrics.get("f0", {}), metrics.get("hybrid", {})
    cost, budget = gate.get("remaining_cost"), gate.get("remaining_budget")
    names = ("correct", "retention_correct", "no_final", "token_cap")
    checks = gate.get("functional_checks", {})
    if (gate.get("version") != GATE_VERSION or gate.get("campaign_sha256") != plan["sha256"]
            or gate.get("rule") != GATE_RULE or gate.get("pilot_checkpoint") != state.get("pilot_checkpoint", state["checkpoint"])
            or gate.get("dev_ids_sha256") != plan["dev_ids_sha256"]
            or gate.get("retention_ids_sha256") != plan["retention_ids_sha256"]
            or set(checks) != {"data", "probability", "gradients", "function"}
            or any(value is not True for value in checks.values())
            or type(gate.get("system_failures")) is not int or gate["system_failures"] != 0
            or type(gate.get("missing")) is not int or gate["missing"] != 0
            or any(type(values.get(name)) is not int or values[name] < 0 for values in (f0, hyb) for name in names)
            or hyb["correct"] <= f0["correct"] or hyb["retention_correct"] < f0["retention_correct"]
            or hyb["no_final"] > f0["no_final"] or hyb["token_cap"] > f0["token_cap"]
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                   for value in (cost, budget)) or cost > budget):
        raise ValueError("single campaign: HOLD pilot/dev/budget expansion gate failed or stale")
    accepted = {"file_sha256": descriptor["file_sha256"], "receipt_sha256": digest(gate)}
    if state.get("main_gate") is not None and state["main_gate"] != accepted:
        raise ValueError("single campaign: accepted main-continuation receipt cannot be replaced")
    state["main_gate"] = accepted


def _direct_gate(plan: dict, state: dict, authorized: bool, descriptor: dict | None) -> None:
    if authorized is not True or descriptor is not None:
        raise ValueError("single campaign: HOLD direct main requires --continue-main and no pilot/dev gate file")
    accepted = {"version": DIRECT_GATE_VERSION, "decision": DIRECT_DECISION, "campaign_sha256": plan["sha256"],
                "pilot_checkpoint": state.get("pilot_checkpoint", state["checkpoint"])}
    if state.get("main_gate") is not None and state["main_gate"] != accepted:
        raise ValueError("single campaign: accepted main-continuation receipt cannot be replaced")
    state["main_gate"] = accepted


def _proof_summary(state: dict) -> dict:
    present, pairs = set(), set()
    for entry in state["proofs"]:
        proof = single._read(_checked(entry["proof"]))["report"]
        for name, component in proof.get("components", {}).items():
            if component.get("route_valid") and component.get("stats", {}).get("has_gradient"):
                present.add(name)
        pairs.update(proof.get("paired_incorrect_a2_b", []))
    missing = sorted({"PG", "A1", "A2", "B"} - present)
    if not pairs:
        missing.append("SAME_PROBLEM_INCORRECT_A2_B")
    return {"verified_slots": len(state["proofs"]), "optimizer_steps": 0, "scheduler_steps": 0,
            "weights_saved": False, "required_component_status": "HOLD" if missing else "PASS", "missing": missing}


def run_campaign(plan: dict, *, output: Path, model_path: Path, actors: dict, provenance: dict,
                 gold: Callable, grade: Callable, load_images: Callable, policy_factory: Callable | None = None,
                 stream: Callable | None = None, encoder: Callable | None = None, device: str = "cpu",
                 max_blocks: int = 1, allow_update: bool = False, allow_save: bool = False,
                 continue_main: bool = False, pilot_gate: dict | None = None, stop_after: float | None = None,
                 reload_only: bool = False, direct_main: bool = False,
                 reference: Callable[[str], str | None] | None = None,
                 effect_gate: dict | None = None, effect_prereg: dict | None = None,
                 effect_evaluator: Callable | None = None) -> dict:
    """Execute an explicit bounded number of blocks, restoring durable state before any work.

    Factory/stream/encoder seams support authored CPU fixtures; defaults are the real production functions.
    A valid checkpoint is the commit authority. Uncertain generated requests are never retried.
    ``reference`` (v11) maps a problem id to its dataset reference text; it reaches the note prompt only.
    """
    if plan.get("version") in ("single-campaign-v16b", "single-campaign-v17"):
        from .campaign_v16b import run_campaign as run_v16b_campaign
        args = argparse.Namespace(data_root=Path(plan["data_root"]), prep_root=Path(plan["prep_root"]),
            model_path=model_path, output=output, precision=plan["protocol"]["precision"],
            local_sequential=plan["local_sequential"], nf4_cache=Path(plan["nf4_cache"]) if plan["nf4_cache"] else None,
            allow_model_run=allow_update or reload_only, allow_update=allow_update, allow_save=allow_save,
            **plan["capacity"])
        return run_v16b_campaign(plan, output=output, args=args, reload_only=reload_only)
    from fine_tuning.notebook_rl import checkpoint
    from fine_tuning.notebook_rl.policy import build_policy, enable_deterministic_execution
    from fine_tuning.notebook_rl.quantized_policy import build_quantized_policy
    from .rl import policy_sha256
    from .train_v4 import encode_single_row
    import random
    import numpy as np
    import torch

    _plan_check(plan)
    content = plan.get("version") == CONTENT_CAMPAIGN_VERSION
    if content:
        from .content_v16 import load_content_manifest

        descriptor = plan["inputs"]["content_manifest"]
        admitted = load_content_manifest(_checked(descriptor), descriptor["file_sha256"])
        if (admitted["rows"] != plan["content_rows"]
                or [item for item in admitted["items"] if item["split"] == "train"] != plan["items"]):
            raise ValueError("v16 campaign: admitted content manifest differs from sealed rows or order")
    if not content and policy_factory is None and (len(plan["items"]) != 3500 or len(plan["blocks"]) != 219
            or plan.get("data_admission", {}).get("train_count") != 3500
            or plan.get("data_admission", {}).get("pilot_count") != 300
            or "execution_max_blocks" not in plan):
        raise ValueError("single campaign: production model requires the actual prepared frozen3500 campaign")
    if type(max_blocks) is not int or not 1 <= max_blocks <= min(len(plan["blocks"]), plan.get("execution_max_blocks", len(plan["blocks"]))):
        raise ValueError("single campaign: explicit bounded block count required")
    if stop_after is not None and (isinstance(stop_after, bool) or not isinstance(stop_after, (int, float))
                                   or not math.isfinite(stop_after) or stop_after <= 0):
        raise ValueError("single campaign: finite positive execution deadline required")
    train = plan["mode"] == "train"
    if content and train and not reload_only:
        _content_functional_gate(plan)
    if train and not reload_only and (allow_update is not True or allow_save is not True):
        raise ValueError("single campaign: explicit update and checkpoint-save opt-in required")
    if not train and (allow_update or allow_save):
        raise ValueError("single campaign: immutable proof mode cannot be upgraded to training")
    if reload_only and not train:
        raise ValueError("single campaign: proof mode has no model checkpoint to reload")
    with single._runtime_lock(output, ".campaign.lock"):
        single._sealed(output / "campaign-plan.json", plan)
        state_path = output / "campaign-state.json"
        state = single._state(state_path) if state_path.exists() else None
        if content:
            if effect_prereg is not None:
                effect_prereg = _descriptor(_checked(effect_prereg))
            saved_prereg = state.get("effect_prereg") if state is not None else None
            if saved_prereg is not None and effect_prereg != saved_prereg and not reload_only:
                raise ValueError("v16 campaign: an effect trial must resume with the same preregistration")
            if (state is not None and saved_prereg is None and effect_prereg is not None
                    and state["cursor"]["slot"] != 0):
                raise ValueError("v16 campaign: effect registration must precede every O-trial update")
        if state is not None and state.get("plan_sha256") != plan["sha256"]:
            raise ValueError("single campaign: mode/source/order/model/profile changed")
        selected, reconciled = None, False
        warm = plan.get("warm_start")  # train-only (_plan_check); its checkpoint takes checkpoint-0's place
        first = output / f"checkpoint-{warm['slot'] if warm else 0}"
        if state is not None:
            _cursor(plan, state["cursor"], state["cursor"]["policy_sha256"])
            if train:
                _commit_check(plan, output, state["cursor"], state["checkpoint"])
                selected = state["checkpoint"]
                if state["pending"] is not None:
                    transaction = single._read(_checked(state["pending"]))
                    slot = state["cursor"]["slot"]
                    if (transaction["slot"] != slot or transaction["ids"] != plan["blocks"][slot]
                            or transaction["parent_checkpoint"] != selected
                            or transaction["policy_sha256"] != state["cursor"]["policy_sha256"]
                            or transaction["plan_sha256"] != plan["sha256"]):
                        raise ValueError("single campaign: stale update transaction")
                    _checked(transaction["rows"])
                    next_path = output / f"checkpoint-{slot + 1}"
                    if next_path.exists():
                        descriptor = _descriptor(next_path / "metadata.json")
                        committed = checkpoint.read_metadata(next_path, binding=_binding(plan))["cursor"]["single"]
                        if (committed.get("last_transaction_sha256") != digest(transaction)
                                or committed.get("parent_checkpoint") != selected or committed["slot"] != slot + 1):
                            raise ValueError("single campaign: next checkpoint belongs to another transaction")
                        _cursor(plan, committed, committed["policy_sha256"])
                        _commit_check(plan, output, committed, descriptor)
                        selected, reconciled = descriptor, True
                        state.update(cursor=committed, checkpoint=descriptor, pending=None, rng=None, status="READY")
                        if content:
                            state["note_yield_records"] = single._read(_checked(transaction["rows"])).get("note_yield_records", [])
                        if not content and committed["slot"] == 19 and len(committed["consumed_problem_ids"]) == 300:
                            state.update(status="PILOT_COMPLETE", pilot_checkpoint=descriptor)
                    elif next_path.with_name(next_path.name + ".tmp").exists():
                        state["status"] = "RECOVERY_REQUIRED"
                        return single._state(state_path, state)
        elif train:
            if first.exists():
                selected = _descriptor(first / "metadata.json")
                saved = checkpoint.read_metadata(first, binding=_binding(plan))["cursor"]["single"]
                if warm is not None:
                    if digest(saved) != warm["cursor_sha256"]:
                        raise ValueError("single campaign: warm-start checkpoint cursor changed")
                elif (saved.get("slot") != 0 or saved.get("updates") != 0 or saved.get("consumed_problem_ids") != []
                        or saved.get("last_transaction_sha256") is not None or saved.get("parent_checkpoint") is not None):
                    raise ValueError("single campaign: initial checkpoint is not fresh F0")
            elif first.with_name(first.name + ".tmp").exists():
                raise ValueError("single campaign: RECOVERY_REQUIRED partial initial checkpoint")
        enable_deterministic_execution()
        random.seed(MASTER_SEED)
        np.random.seed(MASTER_SEED)
        torch.manual_seed(MASTER_SEED)
        factory = policy_factory or (build_quantized_policy if plan["protocol"]["precision"] == "nf4" else build_policy)
        optimizer = scheduler = None
        sampler = None
        if plan["capacity"]["engine"] == "vllm" and not reload_only and stream is None:
            from transformers import GenerationConfig
            from .single_vllm import SingleVllmSampler

            # Before the HF policy reaches the GPU: vLLM checks its memory fraction against free memory at start.
            eos = GenerationConfig.from_pretrained(str(model_path), local_files_only=True).eos_token_id
            sampler = SingleVllmSampler(model_path, kv_cache_gib=plan["capacity"]["kv_cache_gib"],
                max_num_seqs=plan["capacity"]["max_num_seqs"],
                gpu_memory_utilization=plan["capacity"]["gpu_memory_utilization"],
                sleep=plan["capacity"]["vllm_sleep"], eos_token_ids=eos if isinstance(eos, list) else [eos],
                image=plan["protocol"]["image"], lora=plan["protocol"].get("rollout_adapter") == ROLLOUT_ADAPTER_V15)
        if selected is not None:
            def checked_factory(*values):
                fresh = factory(*values)
                if state is None and warm is None and policy_sha256(fresh) != saved["policy_sha256"]:
                    raise ValueError("single campaign: initial checkpoint differs from fresh F0")
                return fresh

            bundle, optimizer, scheduler, cursor = single.load_single_continuation(
                model_path, _checked(selected).parent, plan["protocol"], binding=_binding(plan), device=device,
                policy_factory=checked_factory)
            if state is not None and state["cursor"] != cursor:
                raise ValueError("single campaign: state pointer differs from committed checkpoint")
        elif warm is not None and state is None and not reload_only:
            # The source checkpoint loads under its own protocol/binding; it is re-saved below under this plan's.
            old = single._read(_checked(warm["source_plan"]))
            if old.get("sha256") != warm["source_plan_sha256"] or _binding(old) != warm["source_binding"]:
                raise ValueError("single campaign: warm-start source plan changed")
            bundle, optimizer, scheduler, cursor = single.load_single_continuation(
                model_path, _checked(warm["checkpoint"]).parent, old["protocol"], binding=warm["source_binding"],
                device=device, policy_factory=factory)
            if digest(cursor) != warm["cursor_sha256"] or cursor["slot"] != warm["slot"] or cursor["updates"] != warm["slot"]:
                raise ValueError("single campaign: warm-start checkpoint cursor changed")
        else:
            bundle = factory(model_path, ARM, plan["protocol"])
            if plan["protocol"]["precision"] == "bf16":
                bundle.model.to(device)
            cursor = ({"slot": 0, "updates": 0, "consumed_problem_ids": [], "policy_sha256": policy_sha256(bundle),
                       "last_transaction_sha256": None, "parent_checkpoint": None} if state is None else state["cursor"])
        _cursor(plan, cursor, policy_sha256(bundle))
        if content and effect_prereg is not None:
            cursor["content_effect_prereg"] = effect_prereg
        if state is None:
            if reload_only and selected is None:
                raise ValueError("single campaign: no committed campaign to reload")
            state = {"version": VERSION, "plan_sha256": plan["sha256"], "cursor": cursor, "checkpoint": None,
                     "pending": None, "rng": None, "proofs": [], "status": "READY"}
            if train:
                path = first
                if selected is None:
                    if warm is None:
                        optimizer, scheduler = checkpoint.make_optimizer(bundle, plan["protocol"])
                    single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=path,
                        protocol=plan["protocol"], binding=_binding(plan), allow_save=True)
                state["checkpoint"] = _descriptor(path / "metadata.json")
                if warm is not None:
                    # A fixed main-gate anchor: the state["checkpoint"] fallback moves with every committed block.
                    state |= {"warm_start": warm["checkpoint"], "pilot_checkpoint": state["checkpoint"]}
            if content and effect_prereg is not None:
                state["effect_prereg"] = effect_prereg
            single._state(state_path, state)
        elif reconciled:
            single._state(state_path, state)
        if content and effect_prereg is not None and state.get("effect_prereg") is None:
            state["effect_prereg"] = effect_prereg
            single._state(state_path, state)
        capacity = None
        args = argparse.Namespace(**plan["capacity"])
        args.image_cfg = plan["protocol"]["image"]  # the plan's image budget reaches the HF stream builder (v14: B2's)
        if sampler is not None:
            args.stream_builder = sampler.stream_builder
        if "profile" in plan["capacity"]:
            from .capacity import Capacity, load_profile

            descriptor = plan["capacity"]["profile"]
            profile = load_profile(_checked(descriptor), descriptor["file_sha256"])
            identity = {"source_sha256": digest(plan["source"]), "model_sha256": plan["generation"]["model_files_sha256"],
                        "precision": plan["protocol"]["precision"]}
            capacity = Capacity(bundle, profile, identity, optimizer=optimizer)
            args.capacity_session = capacity
        slot = cursor["slot"]
        phase_path = output / f"block-{slot}" / "phase.json"
        phase = single._state(phase_path) if phase_path.exists() else None
        _rng_restore(state, phase if (state.get("rng") or {}).get("phase_sha256") is not None else None)
        if reload_only:
            return state | {"status": "RELOADED"}
        if reconciled and state["status"] == "PILOT_COMPLETE":
            return state
        def observe_effect() -> None:
            if not content or effect_prereg is None or cursor["slot"] not in (0, 8, 16):
                return
            stage = str(cursor["slot"])
            saved = state.get("effect_phases", {}).get(stage)
            if saved is not None:
                previous = single._read(_checked(saved))
                if previous.get("stop_reasons"):
                    state.update(status="CONTENT_EFFECT_STOP", stop_reasons=previous["stop_reasons"])
                    single._state(state_path, state)
                return
            from .run_v16 import evaluate_effect_bundle

            registration = single._read(_checked(effect_prereg))
            descriptor = plan["inputs"]["content_manifest"]
            checkpoint_record = ({"path": state["checkpoint"]["path"],
                                  "file_sha256": state["checkpoint"]["file_sha256"]} if state["checkpoint"] else None)
            effect_output = output / "effect"
            effect_output.mkdir(exist_ok=True)
            rng = checkpoint.capture_rng()
            try:
                receipt = (effect_evaluator or evaluate_effect_bundle)(bundle, registration, admitted,
                    _checked(descriptor), stage=stage, output=effect_output, checkpoint=checkpoint_record)
            finally:
                checkpoint.restore_rng(rng)
            path = effect_output / f"phase{stage}.json"
            single._sealed(path, receipt)
            state.setdefault("effect_phases", {})[stage] = _descriptor(path)
            if receipt.get("stop_reasons"):
                state.update(status="CONTENT_EFFECT_STOP", stop_reasons=receipt["stop_reasons"])
            single._state(state_path, state)

        observe_effect()
        if state["status"] == "CONTENT_EFFECT_STOP":
            return state
        advanced = int(reconciled)
        by_id = {item["problem_id"]: item for item in plan["items"]}
        # v14/v15's scheduler has exactly execution_max_blocks steps: no block at or past that slot is generated or updated.
        last = (plan["execution_max_blocks"] if plan["protocol"]["version"] in (VERSION_SINGLE_V14, VERSION_SINGLE_V15)
                else len(plan["blocks"]))
        while cursor["slot"] < last and advanced < max_blocks:
            slot = cursor["slot"]
            if content:
                from .protocol import content_contract_for_block

                active = content_contract_for_block(plan["protocol"], slot)
                if any(active["weights"][kind] > 0 for kind in ("J", "A1", "C", "N")):
                    anchor = cursor.get("content_effect_checkpoint_sha256") or (
                        state["checkpoint"]["file_sha256"] if state["checkpoint"] else None)
                    if effect_gate is None:
                        state.update(status="CONTENT_GATE_HOLD", gate_reason="EFFECT_NOTRUN",
                                     runtime_status={"effect": "NOTRUN", "note_yield": "NOTRUN"})
                        return single._state(state_path, state)
                    if anchor is None:
                        raise ValueError("v16 campaign: phase2 requires its actual O-trial checkpoint")
                    try:
                        _content_runtime_gate(plan, effect_gate, checkpoint_sha256=anchor)
                    except ValueError as exc:
                        if not str(exc).startswith("v16 campaign: EFFECT_STOP"):
                            raise
                        state.update(status="CONTENT_GATE_HOLD", gate_reason=str(exc))
                        return single._state(state_path, state)
                    cursor["content_effect_checkpoint_sha256"] = anchor
            if not content and slot >= 19 and len(cursor["consumed_problem_ids"]) >= 300:
                if train:
                    if direct_main:
                        _direct_gate(plan, state, continue_main, pilot_gate)
                    else:
                        _gate(plan, state, pilot_gate, continue_main)
                    single._state(state_path, state)
                elif not continue_main:
                    state["status"] = "PILOT_COMPLETE"
                    return single._state(state_path, state)
            ids = plan["blocks"][slot]
            block_output = output / f"block-{slot}"
            transaction_path = output / f"transaction-{slot}.json"
            next_path = output / f"checkpoint-{slot + 1}"
            # Timings go to stderr only: update-stats.json is sealed and a crash-restart must reproduce it exactly.
            seconds: dict = {"generation": None}  # None: rows were generated by an earlier process
            if state["pending"] is not None:
                transaction = single._read(_checked(state["pending"]))
                if (transaction["slot"] != slot or transaction["ids"] != ids
                        or transaction["parent_checkpoint"] != state["checkpoint"]
                        or transaction["policy_sha256"] != policy_sha256(bundle)
                        or transaction["plan_sha256"] != plan["sha256"]):
                    raise ValueError("single campaign: stale update transaction")
                _checked(transaction["rows"])
                if next_path.exists():
                    raise ValueError("single campaign: checkpoint appeared after initialization; restart recovery")
                if next_path.with_name(next_path.name + ".tmp").exists():
                    state["status"] = "RECOVERY_REQUIRED"
                    return single._state(state_path, state)
                closed = single._read(_checked(transaction["rows"]))
                if digest(closed["contract"]) != transaction["contract_sha256"]:
                    raise ValueError("single campaign: saved transaction contract changed")
            else:
                started = time.monotonic()
                state_block = single.run_single_block(bundle, actors=actors, items=[by_id[pid] for pid in ids],
                    provenance=provenance, gold=gold, grade=grade, load_images=load_images, protocol=plan["protocol"],
                    output=block_output, block_id=slot, stream=stream, stream_args=args, stop_after=stop_after,
                    reference=reference, **({"content_rows": plan["content_rows"],
                        "note_yield_records": state.get("note_yield_records", []),
                        "effect_gate": single._read(_checked(effect_gate)) if effect_gate else None}
                        if content else {}))
                if state_block["status"] != "ROWS_READY":
                    state.update(status=state_block["status"], waiting=state_block.get("waiting", []))
                    _rng_save(output, state, state_block)
                    return single._state(state_path, state)
                closed_path = block_output / state_block["rows_path"]
                closed = single._read(closed_path)
                transaction = {"version": VERSION, "plan_sha256": plan["sha256"], "slot": slot, "ids": ids,
                    "policy_sha256": policy_sha256(bundle), "parent_checkpoint": state["checkpoint"],
                    "rows": _descriptor(closed_path), "contract_sha256": digest(closed["contract"])}
                single._sealed(transaction_path, transaction)
                seconds["generation"] = round(time.monotonic() - started, 1)
                state["pending"] = _descriptor(transaction_path)
                _rng_save(output, state, state_block)
                single._state(state_path, state)
            if (closed["status"] != "ROWS_READY" or closed["rows_sha256"] != digest(closed["rows"])
                    or closed["contract"]["expected_problem_ids"] != ids
                    or closed["contract"]["policy_sha256"] != policy_sha256(bundle)):
                raise ValueError("single campaign: closed rows differ from frozen block policy/order")
            encoded, exclusions = [], []
            for row in closed["rows"]:
                try:
                    encoded.append(encoder(bundle, row) if encoder is not None else encode_single_row(
                        bundle.processor, bundle.model.config, row, load_images(actors[row["problem_id"]]),
                        image=plan["protocol"]["image"]))
                except ValueError as exc:
                    if not content or str(exc) not in ("TARGET_TOO_LONG", "CONTEXT_BLOCKED", "NO_VERIFIED_TOKENS"):
                        raise
                    exclusions.append({"problem_id": row["problem_id"], "row_id": row["row_id"], "reason": str(exc)})
            update_contract = closed["contract"]
            if content:
                accepted_ids = {row["problem_id"] for row in encoded}
                excluded = {pid: next((entry["reason"] for entry in exclusions if entry["problem_id"] == pid),
                                     "NO_TASK_SIGNAL") for pid in ids if pid not in accepted_ids}
                update_contract = update_contract | {"excluded_problem_ids": excluded}
                single._sealed(block_output / "encoded-exclusions.json",
                               {"rows": exclusions, "excluded_problem_ids": excluded})
            if train:
                cursor.update(last_transaction_sha256=digest(transaction), parent_checkpoint=state["checkpoint"])
                if sampler is not None:
                    sampler.release()  # vLLM weights/KV leave the GPU before the HF backward
                started = time.monotonic()
                update = single.run_joint_single_update(bundle, optimizer, scheduler, encoded, update_contract, cursor=cursor,
                    block_ids=ids, allow_update=True, token_budget=args.token_budget, capacity=capacity)
                seconds["update"] = round(time.monotonic() - started, 1)
                kinds = ("O", "J", "C", "N", "A1") if content else (
                    ("RL", "A1", "A2", "B") + (("H",) if plan["protocol"]["version"] == VERSION_SINGLE_V12 else ()))
                single._sealed(block_output / "update-stats.json", {"transaction_sha256": digest(transaction),
                    "optimizer_steps": update["optimizer_steps"], "stats": update["stats"],
                    "rows_by_kind": {kind: sum(row["kind"] == kind for row in closed["rows"]) for kind in kinds}})
                started = time.monotonic()
                single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=next_path,
                    protocol=plan["protocol"], binding=_binding(plan), allow_save=True)
                print(f"SINGLE_BLOCK_SECONDS slot={slot} {seconds} save={time.monotonic() - started:.1f}",
                      file=sys.stderr, flush=True)
                state["checkpoint"] = _descriptor(next_path / "metadata.json")
            else:
                proof_path = block_output / "backward-proof.json"
                if proof_path.exists():
                    saved = single._read(proof_path)
                    if saved["transaction_sha256"] != digest(transaction):
                        raise ValueError("single campaign: proof belongs to another row transaction")
                    proof = saved["report"]
                else:
                    proof = single.prove_single_backward(bundle, encoded, update_contract, precision=plan["protocol"]["precision"],
                                                        token_budget=args.token_budget, capacity=capacity)
                    single._sealed(proof_path, {"transaction_sha256": digest(transaction), "report": proof})
                if proof["status"] == "FAIL" or proof.get("error") or not proof["all_parameters_unchanged"]:
                    state["status"] = "PROOF_FAILED"
                    return single._state(state_path, state)
                state["proofs"].append({"slot": slot, "proof": _descriptor(proof_path)})
                cursor.update(slot=slot + 1, consumed_problem_ids=cursor["consumed_problem_ids"] + ids)
            state.update(cursor=cursor, pending=None, rng=None, status="READY", waiting=[])
            if content:
                state["note_yield_records"] = closed.get("note_yield_records", [])
            _cursor(plan, cursor, policy_sha256(bundle))
            _rng_save(output, state, None)
            single._state(state_path, state)
            observe_effect()
            if state["status"] == "CONTENT_EFFECT_STOP":
                return state
            retained = {0, 8, 16} if content else {19}
            if train and slot >= 2 and slot - 1 not in retained:
                # The committed successor only needs its immediate parent; older full checkpoints are dropped.
                shutil.rmtree(output / f"checkpoint-{slot - 1}", ignore_errors=True)
            advanced += 1
            if not content and cursor["slot"] == 19 and len(cursor["consumed_problem_ids"]) == 300:
                state.update(status="PILOT_COMPLETE", pilot_checkpoint=state["checkpoint"])
                if not train:
                    state["proof_summary"] = _proof_summary(state)
                return single._state(state_path, state)
        state["status"] = "COMPLETE" if cursor["slot"] == last else "BOUNDED_STOP"
        if not train:
            state["proof_summary"] = _proof_summary(state)
        return single._state(state_path, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "advance", "reload"), nargs="?", default="prepare")
    for name in ("build-root", "order", "frozen-approval", "source-provenance", "training-rules",
                 "note-reference-supplement", "model-path", "output"):
        parser.add_argument("--" + name, type=Path, required=name in ("model-path", "output"))
    for name in ("order", "frozen-approval", "source-provenance", "training-rules", "note-reference-supplement",
                 "pilot-ids", "dev-ids", "retention-ids"):
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--content-manifest", type=Path)
    parser.add_argument("--content-manifest-sha256")
    parser.add_argument("--content-weights", help="fixed O/J/C/N weights as a JSON object")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--content-total-blocks", type=int)
    parser.add_argument("--stage-a-blocks", type=int, default=0)
    parser.add_argument("--functional-gate", type=Path)
    parser.add_argument("--functional-gate-sha256")
    parser.add_argument("--effect-gate", type=Path)
    parser.add_argument("--effect-gate-sha256")
    parser.add_argument("--effect-prereg", type=Path)
    parser.add_argument("--effect-prereg-sha256")
    parser.add_argument("--campaign-mode", choices=("proof", "train"), default="proof")
    parser.add_argument("--precision", choices=("nf4", "bf16"), default="nf4")
    parser.add_argument("--token-scale", type=float, default=SINGLE_TOKEN_SCALE_V11)
    for name in ("capacity-profile", "pilot-gate", "warm-start-plan", "warm-start-checkpoint", "warm-start-reload"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--" + name + "-sha256")
    for name in ("max-num-seqs", "max-num-batched-tokens", "token-budget"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--engine", choices=("hf", "vllm"), default="hf", help="generation engine (vllm: BF16 pods)")
    parser.add_argument("--kv-cache-gib", type=float, default=16.)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    parser.add_argument("--vllm-sleep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-blocks", type=int, default=1, help="blocks to run in this call")
    parser.add_argument("--execution-max-blocks", type=int, default=219, help="sealed campaign-wide ceiling")
    parser.add_argument("--allow-model-run", action="store_true")
    parser.add_argument("--allow-update", action="store_true")
    parser.add_argument("--allow-save", action="store_true")
    parser.add_argument("--continue-main", action="store_true")
    parser.add_argument("--direct-main", action="store_true", help="continue after the pilot on the user's decision, "
                        "without a pilot/dev gate file")
    parser.add_argument("--stop-after-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    import json
    import os
    import sys
    values = list(sys.argv[1:] if argv is None else argv)
    if ("--data-root" in values or "--version=error-note-v16b" in values
            or "error-note-v16b" in values or "--version=error-note-v17" in values
            or "error-note-v17" in values or "--version=error-note-v18" in values or "error-note-v18" in values
            or "--version=error-note-v18b" in values or "error-note-v18b" in values):
        from .campaign_v16b import main as main_v16b
        return main_v16b(values)
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    from fine_tuning.notebook_rl.verifier_service import VerifierService
    from .collect import install_training_guards

    args = build_parser().parse_args(argv)
    if args.command != "prepare":
        from fine_tuning.notebook_rl.policy import enable_deterministic_execution

        enable_deterministic_execution()  # must precede the guards/verifier imports that initialize CUDA
    try:
        with single._runtime_lock(args.output, ".campaign-cli.lock"):
            plan = prepare(args)
            if args.command == "prepare":
                print(json.dumps({"status": "PREPARED", "slots": len(plan["blocks"]), "model_loaded": False}))
                return 0
            if not args.allow_model_run:
                raise ValueError("single campaign: actual model execution requires explicit --allow-model-run")
            if (args.pilot_gate is None) != (args.pilot_gate_sha256 is None):
                raise ValueError("single campaign: gate path and external SHA256 required together")
            if args.pilot_gate is not None:
                _checked({"path": str(args.pilot_gate), "file_sha256": args.pilot_gate_sha256})
            if plan.get("version") == CONTENT_CAMPAIGN_VERSION:
                from PIL import Image

                actors = {}
                for item in plan["items"]:
                    image_path = Path(item["image_path"])
                    if not image_path.is_absolute():
                        image_path = _checked(plan["inputs"]["content_manifest"]).parent / image_path
                    actors[item["problem_id"]] = SimpleNamespace(
                        problem_id=item["problem_id"], question=item.get("question", item.get("p0_text", "")),
                        choices=tuple(item.get("choices", ())), image_paths=(str(image_path),),
                        image_sha256=(item["image_sha256"],), mode=item.get("mode", "plain"),
                        source=item["source_repo"])
                def images(actor):
                    with Image.open(actor.image_paths[0]) as image:
                        return [image.convert("RGB")]

                if (args.effect_gate is None) != (args.effect_gate_sha256 is None) or (
                        args.effect_prereg is None) != (args.effect_prereg_sha256 is None):
                    raise ValueError("v16 campaign: effect artifacts require external SHA256 pins")
                result = run_campaign(plan, output=args.output, model_path=args.model_path, actors=actors,
                    effect_gate={"path": str(args.effect_gate), "file_sha256": args.effect_gate_sha256}
                        if args.effect_gate else None,
                    effect_prereg={"path": str(args.effect_prereg), "file_sha256": args.effect_prereg_sha256}
                        if args.effect_prereg else None,
                    provenance={item["problem_id"]: item for item in plan["items"]},
                    gold=lambda pid: next(item["gold"] for item in plan["items"] if item["problem_id"] == pid),
                    grade=lambda pid, text, finish, mode: _content_grade(
                        next(item for item in plan["items"] if item["problem_id"] == pid), actors[pid], text, finish),
                    load_images=images, device="cuda:0", max_blocks=args.max_blocks, allow_update=args.allow_update,
                    allow_save=args.allow_save, stop_after=args.stop_after_seconds, reload_only=args.command == "reload")
                print(json.dumps({"status": result["status"], "slot": result["cursor"]["slot"],
                                  "updates": result["cursor"]["updates"]}))
                code = 0 if result["status"] in ("BOUNDED_STOP", "COMPLETE", "RELOADED") else 2
                if args.engine == "vllm":
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(code)
                return code
            # The trusted controller preloads private bytes before actor file/import guards.
            gate_descriptor = (_descriptor(args.pilot_gate) | {"raw": args.pilot_gate.read_bytes()}) if args.pilot_gate else None
            actors = {actor.problem_id: actor for actor in read_actors(args.build_root / "train")}
            sources = single._read(args.source_provenance)["items"]
            supplement = references.read_supplement(args.note_reference_supplement, args.note_reference_supplement_sha256)
            note_references = {item["problem_id"]: references.note_reference(sources[item["problem_id"]],
                                                                             supplement.get(item["problem_id"]))
                               for item in plan["items"]}
            private = read_verifier(args.build_root / "train")
            gold = {item["problem_id"]: private[item["problem_id"]].gold for item in plan["items"]}
            del private
            # vLLM needs the reviewed OpenAI schema types (API clients stay denied) and is imported before the file guards.
            forbidden = install_training_guards(args.build_root, vllm_schema={} if args.engine == "vllm" else None,
                private_files=[args.source_provenance, args.training_rules, args.note_reference_supplement,
                               *[path for path in (args.pilot_gate,) if path is not None]])
            with VerifierService(args.build_root / "train", forbidden_roots=forbidden, training_rules_path=args.training_rules,
                                 training_rules_sha256=args.training_rules_sha256) as verifier:
                # MCQ: the V8-aligned evaluation contract only; open answers: the protocol's scorer (v8:
                # single_grading, V8 extraction + recorded normalization + training comparison).
                contracts = plan["protocol"]["training_contract"]["answer_contracts"]

                def grade(pid, text, finish, mode):
                    return single.grade_answer(verifier, contracts["mcq" if actors[pid].choices else "open"], pid, text,
                                               finish, mode, actor=actors[pid])

                result = run_campaign(plan, output=args.output, model_path=args.model_path, actors=actors, provenance=sources,
                    gold=gold.__getitem__, grade=grade, load_images=lambda actor: load_images(actor, args.build_root / "images"),
                    device="cuda:0", max_blocks=args.max_blocks, allow_update=args.allow_update, allow_save=args.allow_save,
                    continue_main=args.continue_main, pilot_gate=gate_descriptor,
                    stop_after=args.stop_after_seconds, reload_only=args.command == "reload", direct_main=args.direct_main,
                    reference=note_references.get)
            print(json.dumps({"status": result["status"], "slot": result["cursor"]["slot"], "updates": result["cursor"]["updates"]}))
            code = 0 if result["status"] in ("BOUNDED_STOP", "PILOT_COMPLETE", "COMPLETE", "RELOADED") else 2
    except BaseException:
        if args.engine != "vllm":
            raise
        import traceback

        traceback.print_exc()  # the direct exit below must still report the failure
        code = 1
    if args.engine == "vllm":
        # ponytail: vLLM 0.11 sleep-mode MemPool teardown aborts at interpreter exit after all state is durable;
        # exit directly once outputs are flushed. Drop this when a vLLM upgrade fixes the destructor.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
