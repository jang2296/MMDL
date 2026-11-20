"""Local real-model minimal RL and fresh-process reload, using the existing trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import socket
import time
import traceback
from contextlib import ExitStack
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any

from . import approval, checkpoint
from .dataset import SPLITS, _bytes, _sha, _write, load_images
from .launch import descriptor, read_descriptor
from .local_data import (ACTOR_MODE_POLICY, LOCAL_PREFLIGHT_SECONDS, POOL_FORMAT_PROOF, POOL_FORMAT_V2, POOL_FORMATS,
                         POOL_MODE_POLICY, POOL_STATUS, PROBE_COVERAGE_RULE, project_observation_actors)
from .local_profile import (DEFAULT_NOTE_CONTRACT, NOTE_CONTRACT_CHOICES, PROFILE, REPAIR_PROTOCOL_ID,
                            add_repair_cli_arguments, build_protocol, local_prompts, repair_cli_keywords,
                            repair_flag_keywords, restore_protocol)
from .schema import ActorItem, from_json, to_json
from .vault import _read, _unique

VERIFIED_UPDATE_EXPLORATION = "verified-update-exploration-v1"
FROZEN_APPROVAL_VERIFY = "frozen-approval-verify-v1"


def write_json(path: Path, value) -> None:
    _write(path, _bytes(value) + b"\n")


def host_identity() -> dict:
    """Where a recorded PID lives: a PID is meaningful only on the host/boot/PID namespace that issued it."""
    def read(source) -> str:
        try:
            return source()
        except OSError:
            return ""
    return {"hostname": socket.gethostname(),
            "boot_id": read(lambda: Path("/proc/sys/kernel/random/boot_id").read_text().strip()),
            "pid_ns": read(lambda: os.readlink("/proc/self/ns/pid"))}


def owner_alive(record: dict) -> bool:
    """Is the process that wrote `record` still running? False for a record from another host/boot/PID namespace
    (its PID says nothing here); a legacy record without `host` keeps the plain local PID check."""
    import psutil

    if "host" in record and record["host"] != host_identity():
        return False
    return psutil.pid_exists(record["pid"])


def same_process(a: dict, b: dict) -> bool:
    """Same PID on the same host (a record without `host` compares by PID alone, as before)."""
    return a.get("pid") == b.get("pid") and ("host" not in a or "host" not in b or a["host"] == b["host"])


def disk_low(path: Path, min_free_gib: float | None) -> dict | None:
    """Free space of the output filesystem against an optional bound minimum; checked BEFORE a checkpoint write."""
    if min_free_gib is None:
        return None
    free, need = shutil.disk_usage(path).free, int(min_free_gib * 2**30)
    return None if free >= need else {"path": str(path), "free_bytes": free, "min_free_bytes": need}


def cache_readiness(path: Path, model_path: Path, protocol: dict) -> dict:
    from .quantized_cache import _sources
    from .local_profile import REPAIR_PROTOCOL_ID

    result = json.loads((path / "result.json").read_text())
    if (result.get("status") != "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED"
            or result.get("denied_original_access_attempts") != 0
            or checkpoint.file_hash(model_path / "nf4_base.json") != result.get("cache_metadata_sha256")
            or checkpoint.digest(json.loads((path / "protocol.json").read_text())) != checkpoint.digest(protocol)
            or json.loads((path / "sources.json").read_text()) != _sources(
                repair=protocol.get("protocol_id") == REPAIR_PROTOCOL_ID)):
        raise ValueError("local run: verified current independent NF4 cache required")
    return result


def read_actor_descriptor(value: dict) -> list[ActorItem]:
    raw = _read(descriptor(value))
    if _sha(raw) != value["file_sha256"]:
        raise ValueError("local run: actor bytes differ from descriptor")
    rows = json.loads(raw, object_pairs_hook=_unique)
    if type(rows) is not list:
        raise ValueError("local run: actor list required")
    return [from_json(ActorItem, json.dumps(row, allow_nan=False)) for row in rows]


def read_execution_actors(prepared: dict, originals: list[ActorItem], protocol: dict) -> list[ActorItem]:
    if prepared.get("format") in POOL_FORMATS:
        # Pool preparations keep each built mode: plain stays plain, obs stays obs.
        if (prepared.get("actor_projection") != {"policy": POOL_MODE_POLICY, "changed_ids": []}
                or "execution_actors" not in prepared):
            raise ValueError("local run: bound pool mode-preserving actors required")
        actors = read_actor_descriptor(prepared["execution_actors"])
        if actors != originals:
            raise ValueError("local run: pool execution actors must be the unchanged built actors")
        return actors
    expected = project_observation_actors(originals, protocol)
    if (prepared.get("actor_projection") != {"policy": ACTOR_MODE_POLICY,
            "changed_ids": [a.problem_id for a in originals if a.mode != "obs"]}
            or "execution_actors" not in prepared):
        raise ValueError("local run: bound observation projection required")
    actors = read_actor_descriptor(prepared["execution_actors"])
    if actors != expected:
        raise ValueError("local run: execution actors must be the exact mode-only projection")
    return actors


def read_pool_preparation(prepared: dict, protocol: dict) -> tuple[dict, dict, list[ActorItem]]:
    """Pool format: same build/approval/protocol binding as the fixed cohort, sizes from the bound rule.

    The rule is re-checked against the fresh approval's eligible IDs. v2 also binds N/K to the protocol's
    scale group and the separate single/two-image probe actors."""
    prepared, request, actors, _ = _read_pool_preparation(prepared, protocol, None)
    return prepared, request, actors


def _frozen_approval(prepared: dict, frozen: dict) -> dict:
    """The saved approval output: bytes equal to the pin, the pin equal to the preparation's validated_data."""
    descriptor(prepared["validated_data"])  # shape only; the pod may hold the file at another path
    descriptor(frozen)
    if frozen["file_sha256"] != prepared["validated_data"]["file_sha256"]:
        raise ValueError("local run: the frozen approval is not this preparation's bound validated_data")
    return read_descriptor(frozen)


def _approval_output_ready(checked: dict) -> None:
    if (checked.get("code_sha256") != approval.code_hashes() or checked.get("status") != "VALIDATED_FOR_MEASUREMENT"
            or checked.get("blockers") != [] or checked.get("measurement_allowed") is not True
            or checked.get("training_allowed") is not False or checked.get("stage") != approval.AI_LOCAL_STAGE
            or checked.get("human_verified") is not False):
        raise ValueError("local run: stale source or incomplete input validation")


def frozen_approval_output(root: Path, frozen: dict) -> tuple[dict, list[str], dict]:
    """frozen-approval-verify-v1 before a preparation exists (export round trip, propose-proof, prepare-pool): the
    pinned saved approval output, its approval checks and the union vault re-hash; no approval is re-run.
    -> (approval output, eligible IDs, data_verification record)."""
    checked = read_descriptor(frozen)
    _approval_output_ready(checked)
    verification = _frozen_vault(root, checked, frozen)
    eligible = (checked.get("eligibility") or {}).get("eligible_ids")
    if type(eligible) is not list or not set(checked.get("selected_ids") or ()) <= set(eligible):
        raise ValueError("local run: the frozen approval carries no usable eligibility")
    return checked, eligible, verification


def _frozen_vault(root: Path, checked: dict, frozen: dict) -> dict:
    """Hash-only check of a union vault: its union receipt and build manifest are the ones the approval bound,
    and pool_freeze.verify re-hashes every record/image against the dataset manifest the union receipt binds.
    No shard root, registry, evidence, benchmark index or exclusion engine is read or re-run."""
    from . import pool_freeze

    root = Path(root).absolute()
    union_path = root / "contamination" / approval.UNION_RECEIPT
    if not union_path.is_file():
        raise ValueError(f"local run: {FROZEN_APPROVAL_VERIFY} requires a union vault (contamination/{approval.UNION_RECEIPT})")
    union, read = pool_freeze.read_union(root)
    bound = checked.get("input_binding") or {}
    if (read.get(str(union_path)) != bound.get("exclusion_receipt_sha256")
            or read.get(str(root / "manifest.yaml")) != bound.get("build_manifest_sha256")):
        raise ValueError("local run: the vault's union receipt or build manifest is not the one the approval bound")
    manifest = union["dataset_manifest"]
    errors = pool_freeze.verify(Path(manifest["path"]), root)
    if errors:
        raise ValueError(f"local run: frozen vault differs from its dataset manifest: {errors[:5]}")
    return {"mode": FROZEN_APPROVAL_VERIFY, "approval_path": frozen["path"], "approval_sha256": frozen["file_sha256"],
            "manifest_sha256": manifest["file_sha256"],
            "vault": {"root": str(root), "build_manifest_sha256": bound["build_manifest_sha256"],
                      "union_receipt_file_sha256": bound["exclusion_receipt_sha256"],
                      "union_receipt_sha256": union["union_receipt_sha256"],
                      "vault_manifest_sha256": union["vault_manifest_sha256"],
                      "freeze_binding_sha256": union["freeze_binding_sha256"],
                      "kept_items": len(union["items"]), "verify_errors": 0}}


def _read_pool_preparation(prepared: dict, protocol: dict,
                           frozen: dict | None) -> tuple[dict, dict, list[ActorItem], dict | None]:
    """`frozen` None: a fresh ai_reviewed_local approval as always. Otherwise frozen-approval-verify-v1: the saved
    approval output stands in for the fresh one, and the 4th value is the run's data_verification record."""
    from .local_data import (POOL_DATA_POLICY, check_pool_rule, check_pool_rule_v2, check_proof_rule, pool_sizes,
                             proof_subjects, public_pool)

    proof = prepared.get("format") == POOL_FORMAT_PROOF
    v2 = proof or prepared.get("format") == POOL_FORMAT_V2
    if (prepared.get("status") != POOL_STATUS or prepared.get("profile") != PROFILE
            or v2 and prepared.get("data_policy") != POOL_DATA_POLICY
            or prepared.get("local_protocol_sha256") != protocol["sha256"]
            or prepared.get("cloud_allowed") is not False or prepared.get("training_allowed") is not False
            or prepared.get("approval_stage") != approval.AI_LOCAL_STAGE or prepared.get("human_verified") is not False):
        raise ValueError("local run: current explicitly authorized local pool preparation required")
    checked = read_descriptor(prepared["validated_data"]) if frozen is None else _frozen_approval(prepared, frozen)
    request = read_descriptor(prepared["data_request"])
    root = Path(request.get("build_root", ""))
    if prepared.get("build_root") != str(root) or prepared.get("image_root") != str(root / "images"):
        raise ValueError("local run: approved build/image root changed")
    _approval_output_ready(checked)
    verification = None
    if frozen is None:
        fresh = approval.validate_for_ai_reviewed_local(root, timeout=LOCAL_PREFLIGHT_SECONDS,
            **{k: v for k, v in request.items() if k != "build_root"})
        eligible = (fresh.get("eligibility") or {}).get("eligible_ids")
        if (fresh.get("status") != "VALIDATED_FOR_MEASUREMENT" or fresh.get("blockers") != []
                or fresh.get("measurement_allowed") is not True or type(eligible) is not list
                or any(fresh.get(k) != checked.get(k) for k in
                       ("input_binding", "code_sha256", "selected_ids", "structures", "stage", "ai_review_file_sha256",
                        "human_verified"))
                or [json.loads(to_json(a)) for a in fresh.get("actors", [])] != checked["actors"]):
            raise ValueError("local run: current input/vault validation differs from preparation")
    else:
        verification = _frozen_vault(root, checked, frozen)
        # The eligibility the approval derived locally; the rule/actor checks below run against it unchanged.
        eligible = (checked.get("eligibility") or {}).get("eligible_ids")
        if type(eligible) is not list or not set(checked.get("selected_ids") or ()) <= set(eligible):
            raise ValueError("local run: the frozen approval carries no usable eligibility")
    selection, rule = read_descriptor(prepared["selection"]), read_descriptor(prepared["rule"])
    public = public_pool(root)
    build_sha = checked["input_binding"]["build_manifest_sha256"]
    summary = (check_proof_rule(rule, public, local_prompts(protocol), build_sha, sizes=pool_sizes(protocol),
                                subjects=proof_subjects(root, public), eligible=eligible) if proof else
               check_pool_rule_v2(rule, public, local_prompts(protocol), build_sha, sizes=pool_sizes(protocol),
                                  eligible=eligible) if v2 else
               check_pool_rule(rule, public, local_prompts(protocol), build_sha, eligible=eligible))
    actors = read_actor_descriptor(prepared["actors"])
    approved = [from_json(ActorItem, json.dumps(x)) for x in checked["actors"]]
    if (selection.get("rule") != rule or selection.get("rule_sha256") != prepared["rule"]["file_sha256"]
            or any(selection.get(key) != value for key, value in summary.items())
            or selection.get("local_protocol_sha256") != protocol["sha256"]
            or [a.problem_id for a in actors] != rule["fixed"] or any(a not in approved for a in actors)
            or read_actor_descriptor(prepared["evaluation_actors"]) != [public[pid][0] for pid in summary["evaluation_ids"]]):
        raise ValueError("local run: bound pool selection changed")
    if v2:
        probes = read_actor_descriptor(prepared["probe_actors"])
        if (prepared.get("probe_coverage") != summary["probe_coverage"]
                or probes != [public[pid][0] for pid in summary["probe_ids"]] or any(a not in approved for a in probes)
                or probe_set(prepared, actors) != probes):
            raise ValueError("local run: bound pool probe coverage or probe actors changed")
    return prepared, request, read_execution_actors(prepared, actors, protocol), verification


def read_frozen_preparation(path: Path, protocol: dict,
                            frozen: dict) -> tuple[dict, dict, list[ActorItem], dict]:
    """Explicit opt-in frozen-approval-verify-v1 (never a default): a union-vault pool preparation whose saved
    ai_reviewed_local approval output `frozen` ({path, file_sha256}) is verified by hashes instead of re-running
    the approval. Returns read_preparation's values plus the data_verification record the run must save."""
    prepared = json.loads(Path(path).read_text())
    if prepared.get("format") not in POOL_FORMATS:
        raise ValueError(f"local run: {FROZEN_APPROVAL_VERIFY} accepts only pool preparations")
    prepared, request, actors, verification = _read_pool_preparation(prepared, protocol, frozen)
    assert verification is not None
    return prepared, request, actors, verification


def read_recorded_preparation(launch: dict, protocol: dict) -> tuple[dict, dict, list[ActorItem]]:
    """A run's own preparation re-read in the frozen mode its launch recorded; the verification must repeat."""
    recorded = launch["data_verification"]
    if type(recorded) is not dict or recorded.get("mode") != FROZEN_APPROVAL_VERIFY:
        raise ValueError("local run: unknown recorded data verification mode")
    frozen = {"path": recorded.get("approval_path"), "file_sha256": recorded.get("approval_sha256")}
    prepared, request, actors, verification = read_frozen_preparation(Path(launch["preparation"]), protocol, frozen)
    if verification != recorded:
        raise ValueError("local run: frozen data verification differs from the run's launch")
    return prepared, request, actors


def read_launch_preparation(launch: dict, protocol: dict) -> tuple[dict, dict, list[ActorItem]]:
    """A launch's preparation in the mode it was verified with: fresh approval unless it recorded a frozen one."""
    if "data_verification" in launch:
        return read_recorded_preparation(launch, protocol)
    return read_preparation(Path(launch["preparation"]), protocol)


def add_frozen_approval_arguments(parser) -> None:
    parser.add_argument("--frozen-approval", type=Path,
                        help=f"{FROZEN_APPROVAL_VERIFY}: saved ai_reviewed_local approval output (the preparation's "
                             "validated_data) of a union vault; verifies hashes instead of re-running the approval")
    parser.add_argument("--frozen-approval-sha256", help="pinned SHA-256 of --frozen-approval (required with it)")


def frozen_approval_option(parser, args) -> dict | None:
    """The explicit CLI opt-in as a descriptor; None (fresh approval) unless both flags are given."""
    path, sha = getattr(args, "frozen_approval", None), getattr(args, "frozen_approval_sha256", None)
    if path is None and sha is None:
        return None
    value = {"path": str(Path(path).absolute()) if path is not None else "", "file_sha256": sha}
    try:
        descriptor(value)
    except ValueError:
        parser.error("--frozen-approval PATH and --frozen-approval-sha256 HEX64 are required together")
    return value


def read_preparation(path: Path, protocol: dict) -> tuple[dict, dict, list[ActorItem]]:
    prepared = json.loads(path.read_text())
    if prepared.get("format") in POOL_FORMATS:
        return read_pool_preparation(prepared, protocol)
    if (prepared.get("format") != "notebook-rl-local-data-v1"
            or prepared.get("status") != "LOCAL_DATA_REVALIDATED"
            or prepared.get("profile") != PROFILE
            or prepared.get("local_protocol_sha256") != protocol["sha256"]
            or prepared.get("cloud_allowed") is not False):
        raise ValueError("local run: current explicitly authorized local input preparation required")
    checked = read_descriptor(prepared["validated_data"])
    request = read_descriptor(prepared["data_request"])
    if (prepared.get("build_root") != request.get("build_root")
            or prepared.get("image_root") != str(Path(request["build_root"]) / "images")):
        raise ValueError("local run: approved build/image root changed")
    if (checked.get("code_sha256") != approval.code_hashes()
            or checked.get("status") != "VALIDATED_FOR_MEASUREMENT"
            or checked.get("blockers") != []
            or checked.get("measurement_allowed") is not True
            or checked.get("training_allowed") is not False):
        raise ValueError("local run: stale source or incomplete input validation")
    fresh = approval.validate_for_gate_data(
        Path(request["build_root"]), timeout=LOCAL_PREFLIGHT_SECONDS,
        **{k: v for k, v in request.items() if k != "build_root"})
    if (fresh.get("status") != "VALIDATED_FOR_MEASUREMENT" or fresh.get("blockers") != []
            or fresh.get("measurement_allowed") is not True
            or any(fresh.get(k) != checked.get(k) for k in
                   ("input_binding", "code_sha256", "selected_ids", "structures"))
            or [json.loads(to_json(a)) for a in fresh.get("actors", [])] != checked["actors"]):
        raise ValueError("local run: current input/vault validation differs from preparation")
    selection = read_descriptor(prepared["selection"])
    actors = read_actor_descriptor(prepared["actors"])
    if ([a.problem_id for a in actors] != selection["problem_ids"] or len(actors) != 10
            or len({selection["structures"][a.problem_id] for a in actors}) != 10
            or any(a not in [from_json(ActorItem, json.dumps(x)) for x in checked["actors"]] for a in actors)):
        raise ValueError("local run: fixed eligible ten changed")
    return prepared, request, read_execution_actors(prepared, actors, protocol)


def covering_actors(actors: list[ActorItem], image_counts=(1, 2)) -> list[ActorItem]:
    """The first actor with each image count, in count order; every count must be present."""
    chosen = [next((a for a in actors if len(a.image_paths) == n), None) for n in image_counts]
    if not chosen or None in chosen:
        raise ValueError("local run: single/multiple-image coverage required")
    return chosen


def minimal_actors(actors: list[ActorItem]) -> list[ActorItem]:
    """Fix one single-image and one multi-image problem before observing answers."""
    return covering_actors(actors, (1, 2))


def pool_v2(prepared: dict | None) -> bool:
    """v2 layout (N/K scale group, bound minimal pair, separate probe actors): a v2 or a proof pool preparation."""
    return type(prepared) is dict and prepared.get("format") in (POOL_FORMAT_V2, POOL_FORMAT_PROOF)


def probe_counts(prepared: dict | None) -> tuple[int, ...]:
    """Image counts the sync/reload probes cover: a v2 pool's bound probe-coverage-trained-image-counts-v1 counts
    ((1,) or (1, 2)); every other preparation the fixed single- and two-image pair."""
    if not pool_v2(prepared):
        return (1, 2)
    coverage = prepared.get("probe_coverage")
    if (type(coverage) is not dict or set(coverage) != {"rule", "image_counts"}
            or coverage["rule"] != PROBE_COVERAGE_RULE or coverage["image_counts"] not in ([1], [1, 2])):
        raise ValueError("local run: bound v2 probe coverage required")
    return tuple(coverage["image_counts"])


def sync_options(prepared: dict | None) -> dict:
    """synchronize() keywords for the bound probe coverage; none for the default single- and two-image pair."""
    counts = probe_counts(prepared)
    return {} if counts == (1, 2) else {"image_counts": counts}


def minimal_training_actors(prepared: dict, actors: list[ActorItem]) -> list[ActorItem]:
    """The minimal stage's trained pair: v2 the rule's bound pair, otherwise the first single/two-image actors."""
    if not pool_v2(prepared):
        return minimal_actors(actors)
    ids = read_descriptor(prepared["selection"]).get("minimal_ids")
    by_id = {a.problem_id: a for a in actors}
    if type(ids) is not list or len(ids) != 2 or not set(ids) <= by_id.keys():
        raise ValueError("local run: bound v2 minimal pair required")
    return [by_id[pid] for pid in ids]


def probe_source(prepared: dict, actors: list[ActorItem]) -> list[ActorItem]:
    """Actors whose `probe_counts` covering set are the sync/reload probes (`make_probes` selects them).

    v2 binds a separate set from minimal ∪ probe_only that is never trained or rolled out: exactly one actor per
    bound covered image count, in count order; every other format probes the given actors exactly as before."""
    if not pool_v2(prepared):
        return actors
    probes = read_actor_descriptor(prepared["probe_actors"])
    if [len(a.image_paths) for a in probes] != list(probe_counts(prepared)):
        raise ValueError("local run: bound v2 probe actors must cover exactly the bound image counts")
    return probes


def probe_set(prepared: dict, actors: list[ActorItem]) -> list[ActorItem]:
    """The exact ordered sync/reload probe actors (one or two) the preparation binds."""
    return covering_actors(probe_source(prepared, actors), probe_counts(prepared))


def with_probe_actors(actors: list, probes: list) -> list:
    """Actors whose images are loaded: the given ones, then any probe-only actor."""
    known = {a.problem_id for a in actors}
    return [*actors, *(a for a in probes if a.problem_id not in known)]


def parameter_hashes(bundle) -> dict:
    from fine_tuning.modules import tensor_sha256
    actual = dict(bundle.model.named_parameters())
    return {group: {r["name"]: tensor_sha256(actual[r["name"]])
                    for r in bundle.inventory["trainable"] if r["group"] == group}
            for group in ("L", "M", "V")}


def verify_changes(before: dict, after: dict, update: dict) -> dict:
    if update["status"] != "UPDATED_UNSYNCHRONIZED":
        raise ValueError("local run: actual completed optimizer update required")
    changed = {g: sum(before[g][n] != after[g][n] for n in before[g]) for g in before}
    if any(not changed[g] or not update["grad_norm"][g] or update["grad_norm"][g] <= 0 for g in ("L", "M")):
        raise ValueError("local run: LLM and Merger parameter changes/gradients not observed")
    vision_on = any(update["routes"].values())
    if (vision_on and (not changed["V"] or not update["grad_norm"]["V"] or update["grad_norm"]["V"] <= 0)
            or not vision_on and (changed["V"] or update["grad_norm"]["V"] is not None)):
        raise ValueError("local run: Vision update does not match verified routing")
    result = {"changed_tensor_counts": changed, "vision_on_observed": vision_on,
              "vision_off_only_observed": not vision_on, "both_routes_verified": False}
    labels = update.get("route_labels")
    if labels is None:
        return result
    # An answer-verified (slotless) tree freezes Vision but is never VISION_OFF route evidence.
    if (type(labels) is not dict or set(labels) != set(update["routes"])
            or any(label not in ("VISION_ON", "VISION_OFF", "ANSWER_VERIFIED")
                   or (label == "VISION_ON") is not update["routes"][pid] for pid, label in labels.items())):
        raise ValueError("local run: route labels differ from the update routes")
    return result | {"vision_off_only_observed": not vision_on and all(label == "VISION_OFF" for label in labels.values()),
                     "answer_verified_observed": any(label == "ANSWER_VERIFIED" for label in labels.values())}


def repair_flag_options(answer_verified_route: bool, finalization_reward: bool,
                        slotless_sources: tuple[str, ...]) -> dict:
    """build_protocol keywords for the opt-in flags only when set, so default calls stay byte-identical."""
    return repair_flag_keywords(answer_verified_route, finalization_reward, slotless_sources)


def _binding(prepared: dict, protocol: dict) -> dict:
    return {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
            "data_sha256": checkpoint.digest(prepared), "code_sha256": checkpoint.digest(approval.code_hashes())}


def minimum_execution_contract(policy: str | None, protocol: dict) -> dict | None:
    if policy is None:
        return None
    from .local_profile import PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID, SYNTHESIS_CREDIT_PROTOCOL_ID

    if policy != VERIFIED_UPDATE_EXPLORATION or restore_protocol(protocol)["protocol_id"] not in (
            SYNTHESIS_CREDIT_PROTOCOL_ID, PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID):
        raise ValueError("local run: explicit verified-update policy requires the current synthesis-credit protocol")
    return {"version": "minimum-execution-contract-v1", "policy": policy,
            "stage": "MINIMAL_TWO_PROBLEM_ONE_UPDATE", "max_optimizer_updates": 1,
            "require_retained_learning_signal": True}


def minimum_binding(prepared: dict, protocol: dict, execution_contract: dict | None = None) -> dict:
    if execution_contract is None:
        return _binding(prepared, protocol)
    if (type(execution_contract) is not dict
            or execution_contract != minimum_execution_contract(execution_contract.get("policy"), protocol)):
        raise ValueError("local run: minimum execution contract changed")
    return _binding(prepared, protocol) | {"data_sha256": checkpoint.digest({
        "preparation_sha256": checkpoint.digest(prepared),
        "execution_contract_sha256": checkpoint.digest(execution_contract)})}


def retained_update_signal(update: dict, protocol: dict) -> dict:
    """Check recorded retained coefficients; actual update/reload evidence is separate."""
    import torch
    from .updates import effective_mis_deltas

    signal = update.get("retained_learning_signal", {})
    deltas = effective_mis_deltas(update, protocol)
    expected_basis = ("scaled_policy_coefficients_at_current_equals_old_on_retained_action_tokens"
        if (protocol["loss"]["clip"]["eps_low"], protocol["loss"]["clip"]["eps_high"]) == (.2, .28) else
        "potential_independent_surrogate_policy_coefficients_on_retained_action_tokens")
    if (type(signal) is not dict or set(signal) != {"required", "basis", "policy_gradient_present",
            "retained_action_count", "nonzero_policy_gradient_tokens", "signal_sample_ids"}
            or signal.get("required") is not True or signal.get("policy_gradient_present") is not True
            or signal.get("basis") != expected_basis or type(deltas) is not dict or not deltas
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in deltas.values())
            or any(type(signal.get(key)) is not int or signal[key] <= 0
                   for key in ("retained_action_count", "nonzero_policy_gradient_tokens"))
            or type(signal.get("signal_sample_ids")) is not list or not signal["signal_sample_ids"]
            or any(type(s) is not str or s not in deltas for s in signal["signal_sample_ids"])
            or len(set(signal["signal_sample_ids"])) != len(signal["signal_sample_ids"])):
        raise ValueError("local run: retained policy coefficient evidence required")
    kept = {s for s, delta in deltas.items() if bool(torch.tensor(delta, dtype=torch.float32).abs()
                                                   <= protocol["loss"]["mismatch"]["delta_seq"])}
    if (signal["retained_action_count"] != len(kept) or not set(signal["signal_sample_ids"]) <= kept
            or len(signal["signal_sample_ids"]) > signal["nonzero_policy_gradient_tokens"]):
        raise ValueError("local run: retained policy coefficient MIS evidence differs")
    return signal


def _install_actor_guards(prepared: dict, request: dict, *, extra_vault_files=(),
                          vllm_schema: dict | None = None) -> list[Path]:
    """`vllm_schema`: the run record of a bound vLLM sampler; it receives `schema_dependency_guard` from
    guards.prepare_vllm_schema_dependency(), run before any guard/model import. None (HF): openai stays unimported."""
    from . import pool_freeze
    from .guards import install_guards, prepare_vllm_schema_dependency
    from .quantized_policy import _versions
    from .shared_smoke import _prepare_hf_dependencies

    _versions()
    if vllm_schema is not None:
        vllm_schema["schema_dependency_guard"] = prepare_vllm_schema_dependency()
    _prepare_hf_dependencies()
    # bnb imports Torch compiler/stdlib sources lazily; prepare trusted code before the file guard.
    import_module("bitsandbytes")
    root = Path(prepared["build_root"])
    # Deny MMMU caches in the project cache and in whatever hub HF_HUB_CACHE/HF_HOME currently select.
    hubs = {Path.home() / "mmdl-cache/huggingface/hub",
            Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")}
    forbidden = [Path(request["index"]["path"]).parent, Path.home() / "mmdl-data/evaluation",
                 *(hub / name for hub in sorted(hubs) for name in ("datasets--MMMU--MMMU", "datasets--MMMU--MMMU_Pro"))]
    vaults = [root / split / name for split in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
    vaults += [Path(request[k]["path"]) for k in ("index", "data_report")]
    for key in ("source_registry", "evidence"):
        # A frozen union vault names one registry/evidence per shard ({"shards": [...]}).
        vaults += [Path(d["path"]) for d in (request[key]["shards"] if "shards" in request[key] else [request[key]])]
    if (root / "contamination" / pool_freeze.UNION_FILE).exists():
        # The shards' own build roots hold the same items' private gold/lineage: deny them too.
        for shard in pool_freeze.read_union(root)[0]["shards"]:
            shard_root = Path(shard["receipt"]["path"]).parent.parent
            vaults += [shard_root / split / name for split in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
    vaults += [Path(path) for path in extra_vault_files]
    install_guards(forbidden_roots=forbidden, vault_files=vaults, allow_vault=False, allow_benchmark=False)
    return forbidden


def make_probes(actors, images, engine, prompts, *, image_counts=(1, 2)):
    from .engines import RolloutRequest
    from .prompts import turn1_text, user_message
    from .trainer import SyncProbe

    return [SyncProbe(RolloutRequest("local-sync:" + a.problem_id, None,
            [user_message(turn1_text(a, prompts), len(images[a.problem_id]))],
            images[a.problem_id], 1, 8, [3407]), engine.tokenizer.encode("Final answer", add_special_tokens=False))
            for a in covering_actors(actors, image_counts)]


def probe_evidence(engine, probes) -> dict:
    """Actual single/multi-image forward and short generation, not task accuracy."""
    from dataclasses import asdict
    from .input_preparation import tensor_identity

    inputs, logprobs = [], []
    for probe in probes:
        item = engine.prepare(probe.request)
        inputs.append({"request_id": probe.request.request_id, "input_tokens": len(item.token_ids),
                       "image_tokens": item.token_ids.count(engine.model_config.image_token_id),
                       "image_grid_thw": item.pixels["image_grid_thw"].tolist(),
                       "processed_image_sizes": [list(im.size) for im in item.images],
                       "pixel_tensors": {k: tensor_identity(v) for k, v in item.pixels.items()}})
        logprobs.append(engine.probe_logprobs(probe.request, probe.completion_ids))
    outputs = engine.sample([p.request for p in probes])
    return {"inputs": inputs, "reference_logprobs": logprobs,
            "generated": [asdict(result) for result in outputs], "task_accuracy_measured": False}


def _positive_error_relationship(entry: dict, *, native_only: bool = False) -> bool:
    return bool(entry["reward"] == 1 and entry["advantage"] > 0 and entry["token_indices"]
        and ((entry.get("credit_type") == "NATIVE_ERROR_RELATIONSHIP" and entry["relation"] in ("CORRECTED", "PERSISTED"))
             or (not native_only and entry.get("credit_type") == "OUTPUT_FINALIZATION" and entry["relation"] == "FINALIZED")))


def relationship_learning_summary(update: dict) -> dict:
    """Keep aggregate relationship effects distinct from measured native-only VJPs."""
    learned, positive = set(), set()
    native_samples = []
    scopes = {"OUTPUT_FINALIZATION_ONLY": "OUTPUT_FINALIZATION",
              "NATIVE_ERROR_RELATIONSHIP_ONLY": "NATIVE_ERROR_RELATIONSHIP",
              "COMBINED_RELATIONSHIP_TYPES": "COMBINED_RELATIONSHIP_POLICY_LOSS"}
    for row in update["notebook_backward"]:
        gradient = row.get("relationship_gradient", {})
        if (gradient.get("measured") is True and gradient.get("finite") is True
                and gradient.get("nonzero_groups") and gradient.get("positive_credit_types")):
            learned.add(scopes[gradient["effect_scope"]])
            positive.update(gradient["positive_credit_types"])
        native = row.get("native_relationship_gradient", {})
        norms = native.get("group_norms", {})
        if (native.get("scope") == "native_error_relationship_policy_loss_only"
                and native.get("measured") is True and native.get("finite") is True
                and type(native.get("positive_tokens")) is int and native["positive_tokens"] > 0
                and type(norms) is dict and set(norms) == {"L", "M", "V"}
                and all(type(value) in (float, int) and math.isfinite(value) and value >= 0
                        for value in norms.values())
                and native.get("nonzero_groups") == [group for group in ("L", "M", "V") if norms[group] > 0]
                and native["nonzero_groups"]
                and type(native.get("credit_sha256")) is str
                and native["credit_sha256"] == row.get("synthesis_credit_sha256")):
            native_samples.append(row["sample_id"])
    return {"learned_signal_types": sorted(learned), "positive_verified_relationship_credit_types": sorted(positive),
            "output_finalization_only_gradient_verified": "OUTPUT_FINALIZATION" in learned,
            "native_relationship_only_gradient_verified": bool(native_samples),
            "native_relationship_gradient_sample_ids": native_samples,
            "strict_notebook_correctness_verified": False, "visual_improvement_verified": False}


def native_relationship_update_evidence(update: dict, protocol: dict) -> dict:
    """Bind native-only commit evidence to retained raw-token credits and actual VJPs."""
    import torch

    from .updates import NATIVE_RELATIONSHIP_REQUIREMENT, effective_mis_deltas
    from .rollout import _hash

    signal = update.get("retained_native_relationship_signal", {})
    summary = relationship_learning_summary(update)
    ids = signal.get("signal_sample_ids", [])
    credits = update.get("synthesis_token_credit", {})
    deltas = effective_mis_deltas(update, protocol)
    if (protocol.get("notebook_update_requirement") != NATIVE_RELATIONSHIP_REQUIREMENT
            or update.get("notebook_update_requirement") != NATIVE_RELATIONSHIP_REQUIREMENT
            or update.get("native_relationship_commit_required") is not True
            or update.get("notebook_reward_enabled") is not True
            or set(signal) != {"required", "basis", "policy_gradient_present",
                               "positive_native_relationship_tokens", "signal_sample_ids"}
            or signal["required"] is not True or signal["policy_gradient_present"] is not True
            or signal["basis"] != "positive_native_coefficients_after_dtype_conversion_and_mis"
            or type(signal["positive_native_relationship_tokens"]) is not int
            or signal["positive_native_relationship_tokens"] <= 0
            or type(ids) is not list or not ids or any(type(value) is not str for value in ids)
            or len(set(ids)) != len(ids)
            or not summary["native_relationship_only_gradient_verified"]
            or not set(summary["native_relationship_gradient_sample_ids"]) <= set(ids)):
        raise ValueError("local run: retained native-only relationship gradient evidence required")
    possible_tokens = 0
    for sample_id in ids:
        delta, credit = deltas.get(sample_id), credits.get(sample_id)
        if (delta is None or type(delta) not in (int, float) or not math.isfinite(delta)
                or not bool(torch.tensor(delta, dtype=torch.float32).abs() <= protocol["loss"]["mismatch"]["delta_seq"])
                or type(credit) is not dict):
            raise ValueError("local run: native signal lacks retained sample credit")
        indices = {index for entry in credit["relationships"]
                   if _positive_error_relationship(entry, native_only=True) and entry["coefficient"] > 0
                   for index in entry["token_indices"]}
        if not indices:
            raise ValueError("local run: positive native relationship tokens required")
        possible_tokens += len(indices)
    if signal["positive_native_relationship_tokens"] > possible_tokens:
        raise ValueError("local run: native token count exceeds mapped source tokens")
    for row in update["notebook_backward"]:
        if row["sample_id"] not in summary["native_relationship_gradient_sample_ids"]:
            continue
        native = row["native_relationship_gradient"]
        if (native.get("credit_type") != "NATIVE_ERROR_RELATIONSHIP"
                or native["credit_sha256"] != _hash(credits[row["sample_id"]])
                or native["positive_tokens"] > signal["positive_native_relationship_tokens"]
                or not row["vision_on"] and native["group_norms"]["V"] != 0):
            raise ValueError("local run: native VJP differs from retained raw-token credit")
    return summary


def error_relationship_signal(trees: list, objective: dict | None, tokenizer, *, native_only: bool = False) -> bool:
    """Require original-group variance on mapped, verified error-relationship units."""
    from . import synthesis_credit
    from .updates import notebook_objective_scores

    if objective != synthesis_credit.contract():
        raise ValueError("local run: explicit synthesis token-credit objective required")
    for tree in trees:
        scores = notebook_objective_scores(tree, objective)
        for group in tree.groups:
            if group.kind != "DIAG" or not group.usable:
                continue
            credits = synthesis_credit.group_credit(group.samples, scores, tokenizer)
            if any(_positive_error_relationship(entry, native_only=native_only)
                   for credit in credits.values() for entry in credit["relationships"]):
                return True
    return False


def _positive_assessment(entry: dict) -> bool:
    return bool(type(entry) is dict and type(entry.get("reward")) is int and entry["reward"] == 1
                and all(type(entry.get(key)) in (int, float) and math.isfinite(entry[key]) and entry[key] > 0
                        for key in ("advantage", "coefficient"))
                and type(entry.get("token_indices")) is list and entry["token_indices"])


def assessment_signal(trees: list, objective: dict | None, tokenizer) -> bool:
    """Use the original n4 assessment literals; relationships cannot satisfy this gate."""
    from . import synthesis_credit
    from .updates import notebook_objective_scores

    if objective != synthesis_credit.contract(synthesis_credit.ASSESSMENT_VERSION):
        raise ValueError("local run: explicit assessment token-credit objective required")
    for tree in trees:
        scores = notebook_objective_scores(tree, objective)
        for group in tree.groups:
            if group.kind == "DIAG" and group.usable:
                credits = synthesis_credit.group_credit(group.samples, scores, tokenizer)
                if any(_positive_assessment(entry) for credit in credits.values() for entry in credit["assessments"]):
                    return True
    return False


def assessment_update_evidence(update: dict, protocol: dict, *, trees=None, fresh_credits=None,
                               parameter_inventory=None) -> dict:
    """Validate exact signed aggregate membership; fresh replay supplies original trees/credits."""
    import torch

    from . import synthesis_credit
    from .rollout import _hash
    from .updates import ASSESSMENT_REQUIREMENT, effective_mis_deltas

    failure = "local run: retained signed aggregate assessment-only gradient evidence required"
    signal = update.get("retained_assessment_signal")
    credits, rows = update.get("synthesis_token_credit"), update.get("notebook_backward")
    aggregate = update.get("assessment_aggregate_gradient")
    if (protocol.get("notebook_update_requirement") != ASSESSMENT_REQUIREMENT
            or protocol.get("notebook_objective") != synthesis_credit.contract(synthesis_credit.ASSESSMENT_VERSION)
            or update.get("notebook_objective") != protocol["notebook_objective"]
            or update.get("notebook_update_requirement") != ASSESSMENT_REQUIREMENT
            or update.get("assessment_commit_required") is not True or update.get("assessment_learning_verified") is not True
            or update.get("notebook_reward_enabled") is not True or update.get("status") != "UPDATED_UNSYNCHRONIZED"
            or protocol["rollout"]["diag"]["max_new_tokens"] != 8192
            or type(signal) is not dict or type(credits) is not dict or not credits or type(rows) is not list
            or type(aggregate) is not dict or fresh_credits is not None and credits != fresh_credits
            or any(type(row) is not dict or type(row.get("sample_id")) is not str for row in rows)
            or len({row["sample_id"] for row in rows}) != len(rows)):
        raise ValueError(failure)
    inventory, members = aggregate.get("parameter_inventory"), aggregate.get("members")
    if (aggregate.get("scope") != "signed_all_retained_native_assessment_policy_loss_only"
            or aggregate.get("basis") != "same_forward_vjp_sum_by_parameter_identity_cpu_float64"
            or aggregate.get("requirement") != ASSESSMENT_REQUIREMENT or aggregate.get("finite") is not True
            or aggregate.get("positive_only") is not False or aggregate.get("benefit_verified") is not False
            or type(inventory) is not list or not inventory or type(members) is not list
            or aggregate.get("parameter_inventory_sha256") != _hash(inventory)
            or aggregate.get("members_sha256") != _hash(members)
            or any(type(member) is not dict or type(member.get("sample_id")) is not str for member in members)
            or len({member["sample_id"] for member in members}) != len(members)
            or {member["sample_id"] for member in members} != set(credits)):
        raise ValueError(failure)
    partition_counts = dict.fromkeys(("L", "M", "V"), 0)
    names = set()
    for parameter in inventory:
        if (type(parameter) is not dict or set(parameter) != {"group", "index", "name", "shape", "dtype", "numel"}
                or parameter["group"] not in partition_counts or type(parameter["index"]) is not int
                or parameter["index"] != partition_counts[parameter["group"]]
                or type(parameter["name"]) is not str or not parameter["name"] or parameter["name"] in names
                or type(parameter["shape"]) is not list
                or any(type(size) is not int or size < 0 for size in parameter["shape"])
                or type(parameter["numel"]) is not int or parameter["numel"] != math.prod(parameter["shape"])
                or parameter["dtype"] not in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64")):
            raise ValueError(failure)
        partition_counts[parameter["group"]] += 1
        names.add(parameter["name"])
    if parameter_inventory is not None:
        keys = ("name", "group", "shape", "dtype", "numel")
        if (type(parameter_inventory) is not list
                or any(type(row) is not dict or not set(keys) <= set(row) for row in parameter_inventory)
                or len({row["name"] for row in parameter_inventory}) != len(parameter_inventory)
                or {row["name"]: {key: row[key] for key in keys} for row in parameter_inventory}
                != {row["name"]: {key: row[key] for key in keys} for row in inventory}):
            raise ValueError(failure)
    norms = aggregate.get("group_norms")
    if (type(norms) is not dict or set(norms) != set(partition_counts)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in norms.values())
            or aggregate.get("nonzero_groups") != [group for group in ("L", "M", "V") if norms[group] > 0]
            or not aggregate["nonzero_groups"]):
        raise ValueError(failure)
    deltas = effective_mis_deltas(update, protocol)
    backward = {row["sample_id"]: row for row in rows}
    scores = {sid: (pid, score) for pid, receipt in update.get("notebook_objective_receipts", {}).items()
              for sid, score in receipt.get("samples", {}).items()}
    if set(scores) != set(credits):
        raise ValueError(failure)
    if trees is not None:
        original = {sample.sample_id: (tree, group, sample) for tree in trees
                    for group in tree.groups if group.kind == "DIAG" and group.usable for sample in group.samples}
        if set(original) != set(credits) or any(len(group.samples) != 4 for _, group, _ in original.values()):
            raise ValueError(failure)
    counts = update.get("n_groups_by_kind", {})
    if set(counts) != {"T1", "T2", "T3", "DIAG"} or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError(failure)
    denominator = sum(protocol["loss"]["group_weights"][kind] * count for kind, count in counts.items())
    if denominator <= 0:
        raise ValueError(failure)
    by_flags: dict[str, dict] = {flag: {"positive_assessment_tokens": 0, "signal_sample_ids": []}
                                for flag in ("N", "I", "U", "IU")}
    positive_ids, retained, measured, group_ids, diagnostic_nonzero = [], [], [], set(), set()
    dtypes = {str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
    for member in members:
        sid = member["sample_id"]
        credit, (pid, score) = credits[sid], scores[sid]
        binding = score.get("whole_reference_sample_binding", {})
        full_group, vector = credit.get("full_group"), credit.get("assessment_token_advantages")
        dtype = dtypes.get(member.get("logprob_dtype"))
        delta = deltas.get(sid)
        if (type(vector) is not list or not 0 < len(vector) <= 8192
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                or type(full_group) is not list or len(full_group) != 4 or credit.get("full_group_size") != 4
                or any(type(row) is not dict or row.get("sample_id") not in credits for row in full_group)
                or len({row["sample_id"] for row in full_group}) != 4
                or any(credits[row["sample_id"]].get("full_group") != full_group for row in full_group)
                or credit.get("full_group_sha256") != _hash(full_group)
                or dtype is None or delta is None or type(delta) not in (int, float) or not math.isfinite(delta)
                or type(credit.get("assessments")) is not list
                or binding.get("sample_id") != sid or binding.get("policy_version") != update.get("step_before")
                or binding.get("completion_token_ids_sha256") != credit.get("token_ids_sha256")
                or pid not in update.get("routes", {}) or type(update["routes"][pid]) is not bool):
            raise ValueError(failure)
        coefficients = [0.] * len(vector)
        flags: dict[str, set[int]] = {flag: set() for flag in by_flags}
        for entry in credit["assessments"]:
            if (type(entry) is not dict or type(entry.get("token_indices")) is not list
                    or type(entry.get("coefficient")) not in (int, float) or not math.isfinite(entry["coefficient"])
                    or entry.get("expected_flags") not in flags
                    or any(type(index) is not int or not 0 <= index < len(vector) for index in entry["token_indices"])):
                raise ValueError(failure)
            for index in entry["token_indices"]:
                coefficients[index] += entry["coefficient"]
                if _positive_assessment(entry):
                    flags[entry["expected_flags"]].add(index)
        if coefficients != vector:
            raise ValueError(failure)
        converted = torch.tensor(vector, dtype=dtype)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(failure)
        keep = bool(torch.tensor(delta, dtype=torch.float32).abs() <= protocol["loss"]["mismatch"]["delta_seq"])
        nonzero = int((converted != 0).sum())
        positives = {flag: sum(bool(converted[index] > 0) for index in indices) for flag, indices in flags.items()}
        expected = {"sample_id": sid, "problem_id": pid, "group_id": binding.get("group_id"),
            "policy_version": update.get("step_before"), "vision_on": update["routes"][pid], "retained": keep,
            "disposition": "MIS_EXCLUDED" if not keep else "VJP" if nonzero else "ZERO_VECTOR",
            "full_group": full_group, "full_group_sha256": _hash(full_group), "credit_sha256": _hash(credit),
            "token_advantages_sha256": _hash(vector), "mask_sha256": _hash([[True] * len(vector)]),
            "logprob_dtype": str(dtype), "token_count": nonzero, "positive_tokens": sum(positives.values()),
            "weight": protocol["loss"]["group_weights"]["DIAG"] / denominator, "norm": 4 * 8192}
        if (member != expected or type(member.get("retained")) is not bool
                or type(member.get("vision_on")) is not bool or type(member.get("policy_version")) is not int
                or type(member.get("norm")) is not int or type(member.get("token_count")) is not int
                or type(member.get("positive_tokens")) is not int
                or type(member.get("weight")) not in (int, float)):
            raise ValueError(failure)
        if trees is not None:
            tree, group, sample = original[sid]
            if (pid != tree.problem_id or binding.get("group_id") != group.group_id
                    or [row["sample_id"] for row in full_group] != [s.sample_id for s in group.samples]
                    or credit["token_ids_sha256"] != _hash(sample.completion_token_ids)):
                raise ValueError(failure)
        group_ids.add(member["group_id"])
        if not keep:
            if sid in backward:
                raise ValueError(failure)
            continue
        retained.append(sid)
        row = backward.get(sid, {})
        gradient = row.get("assessment_gradient", {})
        if (row.get("group_id") != member["group_id"] or row.get("vision_on") is not member["vision_on"]
                or row.get("action_tokens") != len(vector) or row.get("synthesis_credit_sha256") != _hash(credit)
                or any(len(row.get("gradient_before", {}).get(group, [])) != count
                       for group, count in partition_counts.items())):
            raise ValueError(failure)
        if not nonzero:
            if gradient != {"scope": "native_assessment_policy_loss_only", "measured": False}:
                raise ValueError(failure)
        else:
            local_norms = gradient.get("group_norms", {})
            if (gradient.get("scope") != "native_assessment_policy_loss_only" or gradient.get("measured") is not True
                    or gradient.get("basis") != "same_forward_assessment_only_vector_vjp"
                    or gradient.get("credit_type") != "NATIVE_ASSESSMENT" or gradient.get("finite") is not True
                    or set(local_norms) != set(partition_counts)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in local_norms.values())
                    or gradient.get("nonzero_groups") != [g for g in ("L", "M", "V") if local_norms[g] > 0]
                    or gradient.get("token_count") != nonzero or gradient.get("positive_tokens") != sum(positives.values())
                    or gradient.get("credit_sha256") != _hash(credit)
                    or gradient.get("token_advantages_sha256") != _hash(vector)
                    or not member["vision_on"] and local_norms["V"] != 0):
                raise ValueError(failure)
            measured.append(sid)
            diagnostic_nonzero.update(gradient["nonzero_groups"])
        if sum(positives.values()):
            positive_ids.append(sid)
        for flag, count in positives.items():
            if count:
                by_flags[flag]["positive_assessment_tokens"] += count
                by_flags[flag]["signal_sample_ids"].append(sid)
    expected_signal = {"required": True, "basis": "positive_assessment_coefficients_after_dtype_conversion_and_mis",
        "policy_gradient_present": True, "positive_assessment_tokens": sum(v["positive_assessment_tokens"] for v in by_flags.values()),
        "signal_sample_ids": positive_ids, "by_expected_flags": by_flags}
    if (signal != expected_signal or not positive_ids or set(backward) != set(retained)
            or len(group_ids) != counts["DIAG"] or not measured
            or not set(aggregate["nonzero_groups"]) <= diagnostic_nonzero
            or not any(update["routes"].values()) and norms["V"] != 0):
        raise ValueError(failure)
    return {"assessment_learning_verified": True, "assessment_only_gradient_verified": True,
            "assessment_gradient_sample_ids": measured, "retained_assessment_signal": signal,
            "strict_notebook_correctness_verified": False, "visual_improvement_verified": False}


def capture(preparation: Path, model_path: Path, output: Path, *, standalone: bool,
            cache_check: Path | None = None, verified_note_atoms: bool = False,
            verified_note_components: bool = False, verified_note_fact_credit: bool = False,
            verified_note_synthesis: bool = False, verified_note_synthesis_credit: bool = False,
            verified_note_assessment_credit: bool = False,
            partial_notebook: bool = False, referenced_notebook: bool = False, paired_notebook: bool = False,
            pair_action_notebook: bool = False, pair_draft_synthesis: bool = False,
            whole_reference_notebook: bool = False, whole_reference_pages: bool = False,
            repair_notebook: bool = False,
            context_limit: int = 16384, execution_policy: str | None = None,
            note_contract: str = DEFAULT_NOTE_CONTRACT, answer_verified_route: bool = False,
            finalization_reward: bool = False, slotless_sources: tuple[str, ...] = (),
            scale: dict | None = None, turn1_contract: str | None = None,
            turn1_max_new_tokens: int | None = None, frozen_approval: dict | None = None,
            retry_out_of_choice: str | None = None, min_free_gib: float | None = None) -> dict:
    import numpy as np
    import torch

    from . import scale as scale_options
    from .local_calibrate import verify_original

    scale_keywords: dict[str, Any] = {"scale": scale} if scale is not None else {}
    # Only given options reach build_protocol, so default calls stay byte-identical.
    scale_keywords.update({key: value for key, value in (("turn1_contract", turn1_contract),
                                                          ("turn1_max_new_tokens", turn1_max_new_tokens),
                                                          ("retry_out_of_choice", retry_out_of_choice))
                           if value is not None})
    from .notebook_learning import append_rewarded_notebooks
    from .quantized_policy import build_quantized_policy, load_quantized_policy
    from .rollout import RolloutConfig, build_tree
    from .updates import (ASSESSMENT_REQUIREMENT, NATIVE_RELATIONSHIP_REQUIREMENT,
                          bind_notebook_objective, notebook_objective_scores,
                          optimizer_step, partition_learning_trees)
    from .verifier_service import VerifierService
    from .quantized_cache import block_original_access

    protocol = build_protocol(verified_note_atoms=verified_note_atoms, verified_note_components=verified_note_components,
                              verified_note_fact_credit=verified_note_fact_credit,
                              verified_note_synthesis=verified_note_synthesis,
                              verified_note_synthesis_credit=verified_note_synthesis_credit,
                              verified_note_assessment_credit=verified_note_assessment_credit,
                              partial_notebook=partial_notebook, referenced_notebook=referenced_notebook,
                              paired_notebook=paired_notebook,
                              pair_action_notebook=pair_action_notebook,
                              pair_draft_synthesis=pair_draft_synthesis,
                              whole_reference_notebook=whole_reference_notebook,
                              whole_reference_pages=whole_reference_pages,
                              context_limit=context_limit,
                              **({"repair_notebook": repair_notebook} if repair_notebook is not False else {}),
                              **({"note_contract": note_contract} if note_contract != DEFAULT_NOTE_CONTRACT else {}),
                              **repair_flag_options(answer_verified_route, finalization_reward, slotless_sources),
                              **scale_keywords)
    repair_selected = protocol["protocol_id"] == REPAIR_PROTOCOL_ID
    bf16 = scale_options.bf16(protocol)
    if bf16 and standalone:
        raise ValueError("local run: a BF16 policy loads the verified original snapshot, not an NF4 cache")
    answer_route = "repair_route_policy" in protocol
    objective = protocol.get("notebook_objective")
    execution_contract = minimum_execution_contract(execution_policy, protocol)
    if standalone != (cache_check is not None):
        raise ValueError("local run: standalone base requires its completed cache check")
    readiness = cache_readiness(cache_check, model_path, protocol) if cache_check is not None else None
    verification = None
    if frozen_approval is None:
        prepared, request, all_actors = read_preparation(preparation, protocol)
    else:
        prepared, request, all_actors, verification = read_frozen_preparation(preparation, protocol, frozen_approval)
    actors = minimal_training_actors(prepared, all_actors)
    probe_inputs = probe_source(prepared, actors)
    root, prompts = Path(prepared["build_root"]), local_prompts(protocol)
    binding = minimum_binding(prepared, protocol, execution_contract)
    vault_identity = json.loads((root / "manifest.yaml").read_text())["split_manifests"]["train"]
    from .cli import artifact_path
    output = artifact_path(output)
    if any(output.resolve().is_relative_to(p.resolve()) for p in (model_path, root)):
        raise ValueError("local run: output cannot be inside model or data")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if not standalone:
        write_json(output / "original_files.json", verify_original(model_path))
    write_json(output / "launch.json", {
        "profile": PROFILE, "protocol": protocol, "binding": binding,
        "preparation": str(preparation), "model_path": str(model_path), "standalone_base": standalone,
        "cache_check": None if cache_check is None else str(cache_check),
        "problem_ids": [a.problem_id for a in actors], "pid": os.getpid(), "host": host_identity(),
        "execution_authority": protocol["local_authorization"], "purpose": "MINIMAL_TWO_PROBLEM_ONE_UPDATE",
        **({"execution_contract": execution_contract, "execution_contract_sha256": checkpoint.digest(execution_contract)}
           if execution_contract is not None else {}),
        **({"sampler": scale_options.sampler_label(protocol)} if scale_options.scaled(protocol) else {}),
        **({"data_verification": verification} if verification is not None else {}),
    })
    schema = {} if scale_options.options(protocol)["rollout_engine"] == "vllm" else None
    forbidden = _install_actor_guards(prepared, request, vllm_schema=schema)
    original_access = (block_original_access(Path(readiness["original_model_root"]))
                       if readiness else None)
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    deadline = began + protocol["local_budget"]["max_seconds_per_run"]
    result: dict = {"status": "INCOMPLETE", "optimizer_updates": 0, "fresh_reload_verified": False,
              "self_improvement_verified": False, "both_routes_verified": False,
              "notebook_signal_groups": 0, "notebook_gradient_contribution_verified": False, **(schema or {})}
    if repair_selected:
        result.update(repair_functional_verified=False, repair_learning_verified=False, S_gradient_verified=False)
    if answer_route:
        result.update(answer_verified_ids=[], S_text_gradient_verified=False)
    if verified_note_assessment_credit:
        result.update(assessment_learning_verified=False, assessment_only_gradient_verified=False,
                      strict_notebook_correctness_verified=False)
    if whole_reference_pages:
        result["notebook_physical_signal_groups"] = 0
    journal = 0
    repair_raw_links: list[dict] = []

    def record(kind, value):
        nonlocal journal
        raw = json.loads(to_json(value)) if kind in ("GROUP", "TREE") else value
        raw_file = output / f"raw-{journal:04d}.json"
        write_json(raw_file, {"kind": kind, "raw": raw})
        if repair_selected:
            repair_raw_links.append({"file": raw_file.name, "kind": kind, "sha256": checkpoint.file_hash(raw_file)})
        journal += 1
        print(json.dumps({"event": kind, "index": journal, "seconds": time.monotonic()-began}), flush=True)

    try:
        loader = (scale_options.build_bf16_policy if bf16 else
                  load_quantized_policy if standalone else build_quantized_policy)
        bundle = loader(model_path, "MOD-RV", protocol)
        write_json(output / "inventory.json", bundle.inventory)
        guard = scale_options.base_guard(bundle, protocol)
        # HF: one live engine; vLLM: merged-weight sampler + live HF preparation/probe engine.
        sampler, engine = scale_options.make_engines(bundle, protocol, model_path)
        optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
        with ExitStack() as stack:
            images = {a.problem_id: load_images(a, root / "images") for a in with_probe_actors(actors, probe_inputs)}
            for values in images.values():
                for image in values:
                    stack.callback(image.close)
            verifier = stack.enter_context(VerifierService(root / "train", forbidden_roots=forbidden, timeout=60))
            if verifier.metadata["vault_manifest_sha256"] != vault_identity:
                raise ValueError("local run: verifier dataset changed")
            probes = make_probes(probe_inputs, images, engine, prompts, image_counts=probe_counts(prepared))
            initial, sampler_sync = scale_options.synchronize(bundle, sampler, engine, probes, protocol, 0, push=False,
                                                              **sync_options(prepared))
            write_json(output / "initial_sync.json", initial)
            if sampler_sync is not None:
                write_json(output / "initial_sampler_sync.json", sampler_sync)
            cfg = RolloutConfig.from_protocol(protocol, "ALG-NB3", master_seed=3407)
            trees = []
            repair_preparations: dict[str, dict] = {}
            repair_tree_links: list[dict] = []
            for actor in actors:
                tree = build_tree(actor, images[actor.problem_id], sampler, verifier, prompts, cfg, 0,
                    protocol["route_rule"], on_group=lambda g: record("GROUP", g),
                    **({"on_raw": lambda event: record("REPAIR_RAW", event)} if repair_selected else {}),
                    should_stop=lambda: time.monotonic() >= deadline)
                append_rewarded_notebooks([tree], {actor.problem_id: images[actor.problem_id]},
                    sampler, verifier, prompts, cfg, actors={actor.problem_id: actor},
                    on_group=lambda g: record("GROUP", g),
                    **({"on_raw": lambda event: record("REPAIR_RAW", event)} if repair_selected else {
                        "on_observation": lambda event: record("OBSERVATION", event),
                        **({"on_draft": lambda event: record("DRAFT", event)} if pair_draft_synthesis else {}),
                        **({"on_page": lambda event: record("PAGE", event)} if whole_reference_pages else {})}),
                    should_stop=lambda: time.monotonic() >= deadline)
                if not repair_selected and tree.notebook.get("rewarded_notebook", {}).get("notebook_generation_policy") != protocol.get("notebook_generation_policy"):
                    raise ValueError("local run: notebook generation policy differs from protocol")
                if not repair_selected and tree.notebook.get("rewarded_notebook", {}).get("notebook_draft_policy") != protocol.get("notebook_draft_policy"):
                    raise ValueError("local run: notebook draft policy differs from protocol")
                if objective is not None:
                    bind_notebook_objective(tree, actor, verifier, objective)
                if repair_selected:
                    from .repair_training import prepare_tree_credit

                    receipt = prepare_tree_credit(tree, actor, verifier, engine.tokenizer,
                        preparation=engine, images=images[actor.problem_id], master_seed=3407)
                    repair_preparations[actor.problem_id] = receipt
                    receipt_file = f"repair-preparation-{len(repair_tree_links):04d}.json"
                    write_json(output / receipt_file, receipt)
                    tree_file = f"raw-{journal:04d}.json"
                record("TREE", tree)
                if repair_selected:
                    repair_tree_links.append({"problem_id": actor.problem_id, "tree_file": tree_file,
                        "tree_file_sha256": checkpoint.file_hash(output / tree_file),
                        "preparation_file": receipt_file, "preparation_file_sha256": checkpoint.file_hash(output / receipt_file),
                        "preparation_receipt_sha256": receipt["sha256"]})
                trees.append(tree)
            if repair_selected:
                eligible = [tree for tree in trees if repair_preparations[tree.problem_id]["admitted"]
                            and repair_preparations[tree.problem_id]["has_signal"]]
                unresolved = [tree.problem_id for tree in trees
                              if repair_preparations[tree.problem_id]["admission"] == "UNRESOLVED"]
                signal_free = [tree.problem_id for tree in trees if repair_preparations[tree.problem_id]["admitted"]
                               and not repair_preparations[tree.problem_id]["has_signal"]]
                result["incomplete_ids"] = [tree.problem_id for tree in trees if any(
                    reason != "UNRESOLVED_ROUTING" for reason in repair_preparations[tree.problem_id]["admission_reasons"])]
                write_json(output / "repair_preparation_index.json", {"version": "repair-local-minimum-preparation-v1",
                    "protocol_sha256": protocol["sha256"], "capture_binding_sha256": checkpoint.digest(binding),
                    "trees": repair_tree_links, "raw_records": repair_raw_links})
                result["repair_preparation_index_sha256"] = checkpoint.file_hash(output / "repair_preparation_index.json")
            else:
                eligible, unresolved, signal_free = partition_learning_trees(trees, include_notebooks=True,
                                                                             notebook_objective=objective)
            result["attempted_ids"] = [t.problem_id for t in trees]
            result["unresolved_ids"], result["signal_free_ids"] = unresolved, signal_free
            if answer_route:
                result["answer_verified_ids"] = [tree.problem_id for tree in trees
                                                 if repair_preparations[tree.problem_id]["admission"] == "ANSWER_VERIFIED"]
            result["discarded_ids"] = [t.problem_id for t in trees if t not in eligible]
            if not eligible:
                result["status"] = ("UNRESOLVED_ROUTING" if repair_selected and unresolved else
                                    "INCOMPLETE_REPAIR_COLLECTION" if repair_selected and result["incomplete_ids"] else
                                    "NO_LEARNING_SIGNAL" if repair_selected else
                                    "NO_ASSESSMENT_LEARNING_SIGNAL" if verified_note_assessment_credit else
                                    "NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL" if verified_note_synthesis_credit and execution_contract is None else
                                    "UNRESOLVED_ROUTING" if unresolved else "NO_LEARNING_SIGNAL")
                return result
            for tree in ([] if repair_selected else eligible):
                from .updates import group_has_signal, notebook_signal_groups

                scores = notebook_objective_scores(tree, objective)
                result["notebook_signal_groups"] += notebook_signal_groups(tree, scores)
                if whole_reference_pages:
                    result["notebook_physical_signal_groups"] += sum(
                        g.kind == "DIAG" and g.usable and group_has_signal(g, scores) for g in tree.groups)
            if verified_note_assessment_credit and not assessment_signal(eligible, objective, engine.tokenizer):
                result["status"] = "NO_ASSESSMENT_LEARNING_SIGNAL"
                return result
            if (not verified_note_assessment_credit and execution_contract is None and verified_note_synthesis_credit
                    and not error_relationship_signal(eligible, objective, engine.tokenizer,
                                                       native_only=protocol.get("notebook_update_requirement") == NATIVE_RELATIONSHIP_REQUIREMENT)):
                result["status"] = "NO_ERROR_RELATIONSHIP_LEARNING_SIGNAL"
                return result
            # Answer-only updates cannot validate the requested notebook-learning minimum.
            if not repair_selected and execution_contract is None and not result["notebook_signal_groups"]:
                result["status"] = "NO_NOTEBOOK_LEARNING_SIGNAL"
                return result
            if time.monotonic() >= deadline:
                result["status"] = "BUDGET_STOP"
                return result
            before = parameter_hashes(bundle)
            required: dict[str, Any] = ({"require_retained_learning_signal": True}
                                        if execution_contract is not None or repair_selected else {})
            update = optimizer_step(bundle, optimizer, scheduler, eligible,
                {t.problem_id: images[t.problem_id] for t in eligible}, engine, protocol, "ALG-NB3", step=0, seed=3407,
                observation_actors={a.problem_id: a for a in actors if a.problem_id in {t.problem_id for t in eligible}},
                objective_verifier=verifier, **required,
                on_progress=lambda event: print(json.dumps(event, allow_nan=False), flush=True))
            write_json(output / "update.json", update)
            if repair_selected:
                from .repair_evidence import repair_update_evidence
                from .repair_replay import validate_journal

                validation = None
                gradient = update.get("repair_aggregate_gradient")
                if gradient is not None and (type(gradient) is not dict or type(gradient.get("S")) is not bool):
                    raise ValueError("local run: actual repair gradient receipt required")
                if update["status"] == "UPDATED_UNSYNCHRONIZED":
                    validate_journal(output, repair_raw_links, trees)
                    fresh = {tree.problem_id: prepare_tree_credit(tree,
                        next(actor for actor in actors if actor.problem_id == tree.problem_id), verifier, engine.tokenizer,
                        preparation=engine, images=images[tree.problem_id], master_seed=3407) for tree in eligible}
                    if checkpoint.digest(fresh) != checkpoint.digest(
                            {tree.problem_id: repair_preparations[tree.problem_id] for tree in eligible}):
                        raise ValueError("local run: repair source preparation changed during update")
                    validation = repair_update_evidence(update, protocol, preparations=fresh, trees=eligible,
                        parameter_inventory=bundle.inventory["trainable"], require_s=False, control=False)
                write_json(output / "repair_update_evidence.json", {"version": "repair-local-minimum-update-v1",
                    "protocol_sha256": protocol["sha256"], "capture_binding_sha256": checkpoint.digest(binding),
                    "preparation_index_sha256": result["repair_preparation_index_sha256"],
                    "update_file_sha256": checkpoint.file_hash(output / "update.json"),
                    "retained_repair_signal_sha256": checkpoint.digest(update.get("retained_repair_signal")),
                    "repair_aggregate_gradient_sha256": checkpoint.digest(gradient), "validation": validation})
                result["repair_update_evidence_sha256"] = checkpoint.file_hash(output / "repair_update_evidence.json")
            if update["status"] != "UPDATED_UNSYNCHRONIZED":
                result["status"] = update["status"]
                return result
            if repair_selected:
                if validation is None:
                    raise ValueError("local run: completed canonical repair validation required")
                result["S_gradient_verified"] = validation["S_gradient_verified"]
                if answer_route:
                    result["S_text_gradient_verified"] = validation["S_text_gradient_verified"]
            if protocol.get("notebook_update_requirement") == NATIVE_RELATIONSHIP_REQUIREMENT:
                result.update(native_relationship_update_evidence(update, protocol))
            if protocol.get("notebook_update_requirement") == ASSESSMENT_REQUIREMENT:
                result.update(assessment_update_evidence(update, protocol, parameter_inventory=bundle.inventory["trainable"]))
            if execution_contract is not None or pair_action_notebook:
                result["retained_learning_signal"] = retained_update_signal(update, protocol)
            result["optimizer_updates"] = update["step_after"]
            after = parameter_hashes(bundle)
            changes = verify_changes(before, after, update)
            write_json(output / "parameter_changes.json", {"before": before, "after": after, **changes})
            guard(bundle)
            updated, sampler_sync = scale_options.synchronize(bundle, sampler, engine, probes, protocol, 1, push=True,
                                                              **sync_options(prepared))
            write_json(output / "updated_sync.json", updated)
            if sampler_sync is not None:
                write_json(output / "updated_sampler_sync.json", sampler_sync)
            forward_probes = probe_evidence(engine, probes)
            write_json(output / "updated_forward_probes.json", forward_probes)
            guard(bundle)
            if (low := disk_low(output, min_free_gib)) is not None:
                result.update(status="DISK_LOW", disk_low=low)  # stop before writing any checkpoint file
                return result
            checkpoint.save(bundle, optimizer, scheduler, 1,
                {"epoch": 0, "position": len(actors), "discarded_problem_ids": result["discarded_ids"],
                 "problem_ids": [a.problem_id for a in actors], "sync_pending": False},
                checkpoint.capture_rng(), output / "checkpoint", protocol=protocol, binding=binding)
            guard(bundle)
            if original_access and original_access["denied_original_access_attempts"]:
                raise ValueError("local run: original BF16 model access attempted")
            write_json(output / "reload_expected.json", {"trainable_sha256": after, "step": 1,
                "frozen_base_sha256": bundle.base_sha256, "quantization": bundle.inventory.get("quantization"),
                "forward_probes": forward_probes,
                **({"repair_update_evidence_sha256": result["repair_update_evidence_sha256"],
                    "repair_protocol_sha256": protocol["sha256"]} if repair_selected else {})})
            result.update(status="UPDATED_SAVED_PENDING_FRESH_RELOAD", **changes)
            if not repair_selected:
                result["notebook_gradient_contribution_verified"] = any(
                    row["reward_gradient_isolated"] and (row["advantage"] != 0 or row.get("nonzero_partial_tokens", 0) > 0)
                    and row["changed_groups"]
                    for row in update["notebook_backward"])
                result["notebook_combined_objective_gradient_observed"] = any(
                    row["changed_groups"] for row in update["notebook_backward"])
            if verified_note_synthesis_credit:
                result.update(relationship_learning_summary(update))
            return result
    except BaseException as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(seconds=time.monotonic()-began, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result["original_access_guard"] = original_access
        write_json(output / "result.json", result)
        print(json.dumps(result), flush=True)


def reload_capture(capture_root: Path, output: Path) -> dict:
    import torch
    from . import scale as scale_options
    from .quantized_policy import build_quantized_policy, load_quantized_policy

    cached_records = {name: (capture_root / name).read_bytes()
                      for name in ("launch.json", "result.json", "reload_expected.json")}
    launch, original, expected = (json.loads(cached_records[name].decode("utf-8"))
                                 for name in ("launch.json", "result.json", "reload_expected.json"))
    if (type(launch.get("pid")) is not int or launch["pid"] <= 0
            or same_process(launch, {"pid": os.getpid(), "host": host_identity()}) or owner_alive(launch)):
        raise ValueError("local reload: closed successful update in a fresh process required")
    protocol = restore_protocol(launch.get("protocol"))
    repair_selected = protocol["protocol_id"] == REPAIR_PROTOCOL_ID
    execution_contract = launch.get("execution_contract")
    if (any(key in launch for key in ("execution_contract", "execution_contract_sha256"))
            and (execution_contract is None
                 or launch.get("execution_contract_sha256") != checkpoint.digest(execution_contract))):
        raise ValueError("local reload: minimum execution contract changed")
    if execution_contract is not None:
        if (type(execution_contract) is not dict
                or execution_contract != minimum_execution_contract(execution_contract.get("policy"), protocol)):
            raise ValueError("local reload: minimum execution contract changed")
    if (original["status"] != "UPDATED_SAVED_PENDING_FRESH_RELOAD"
            or original["optimizer_updates"] != 1
            or launch["binding"]["code_sha256"] != checkpoint.digest(approval.code_hashes())):
        raise ValueError("local reload: closed successful update, fresh process and same code/protocol required")
    from .updates import ASSESSMENT_REQUIREMENT

    if protocol.get("notebook_update_requirement") == ASSESSMENT_REQUIREMENT:
        metadata = checkpoint.read_metadata(capture_root / "checkpoint", binding=launch["binding"])
        assessment = assessment_update_evidence(json.loads((capture_root / "update.json").read_text()), protocol,
            parameter_inventory=metadata["inventory"]["trainable"])
        if any(original.get(key) != value for key, value in assessment.items()):
            raise ValueError("local reload: capture assessment evidence differs")
    readiness = (cache_readiness(Path(launch["cache_check"]), Path(launch["model_path"]), protocol)
                 if launch["standalone_base"] else None)
    prepared, request, all_actors = read_launch_preparation(launch, protocol)
    actors = minimal_training_actors(prepared, all_actors)
    probe_inputs = probe_source(prepared, actors)
    if minimum_binding(prepared, protocol, execution_contract) != launch["binding"] or [a.problem_id for a in actors] != launch["problem_ids"]:
        raise ValueError("local reload: capture input binding changed")
    replay = None
    if repair_selected:
        from .quantized_cache import block_original_access
        from .repair_replay import replay_capture

        _install_actor_guards(prepared, request)
        original_access = (block_original_access(Path(readiness["original_model_root"])) if readiness else None)
        metadata = checkpoint.read_metadata(capture_root / "checkpoint", binding=launch["binding"])
        # v2: the trained minimal pair and its separate probe set; otherwise the whole selection as before.
        replay = replay_capture(capture_root, protocol=protocol, prepared=prepared,
            parameter_inventory=metadata["inventory"]["trainable"], require_s=False,
            **({"actors": actors, "probe_actors": probe_inputs} if pool_v2(prepared) else {"actors": all_actors}))
        if any(_sha(raw) != replay["paths"].get(name) for name, raw in cached_records.items()):
            raise ValueError("local reload: cached repair records differ from independently replayed file versions")
    from .cli import artifact_path
    output = artifact_path(output)
    if output.resolve().is_relative_to(capture_root.resolve()):
        raise ValueError("local reload: preserve closed capture")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if not repair_selected:
        _install_actor_guards(prepared, request)
    from .quantized_cache import block_original_access
    if not repair_selected:
        original_access = (block_original_access(Path(readiness["original_model_root"])) if readiness else None)
    if replay is not None:
        write_json(output / "repair_replay.json", replay)
    factory = (scale_options.build_bf16_policy if scale_options.bf16(protocol) else
               load_quantized_policy if launch["standalone_base"] else build_quantized_policy)
    bundle, optimizer, scheduler, step, cursor = checkpoint.load_resume(
        Path(launch["model_path"]), capture_root / "checkpoint", protocol, binding=launch["binding"],
        device="cuda:0", policy_factory=partial(factory, device_index=0))
    guard = scale_options.base_guard(bundle, protocol)
    guard(bundle)
    if repair_selected and (type(step) is not int or type(cursor) is not dict
            or checkpoint.digest(cursor) != checkpoint.digest({"epoch": 0, "position": 2,
                "discarded_problem_ids": original["discarded_ids"], "problem_ids": launch["problem_ids"], "sync_pending": False})):
        raise ValueError("local reload: original repair step/cursor types differ")
    if (parameter_hashes(bundle) != expected["trainable_sha256"] or step != expected["step"]
            or bundle.base_sha256 != expected["frozen_base_sha256"]
            or bundle.inventory.get("quantization") != expected["quantization"] or not optimizer.state
            or scheduler.last_epoch != step
            or cursor != {"epoch": 0, "position": 2, "discarded_problem_ids": original["discarded_ids"],
                          "problem_ids": launch["problem_ids"], "sync_pending": False}
            or original["attempted_ids"] != launch["problem_ids"] or len(launch["problem_ids"]) != 2):
        raise ValueError("local reload: restored model/optimizer/reference differs")
    from .trainer import synchronize
    # Saved probes come from the live HF policy engine in every engine mode.
    engine = scale_options.shared_engine(bundle, protocol, policy_version=step)
    with ExitStack() as stack:
        images = {a.problem_id: load_images(a, Path(prepared["image_root"])) for a in with_probe_actors(actors, probe_inputs)}
        for values in images.values():
            for image in values:
                stack.callback(image.close)
        probes = make_probes(probe_inputs, images, engine, local_prompts(protocol), image_counts=probe_counts(prepared))
        write_json(output / "sync.json", synchronize(bundle, engine, engine, probes, protocol, step, push=False,
                                                     **sync_options(prepared)))
        actual = probe_evidence(engine, probes)
        write_json(output / "forward_probes.json", actual)
        if actual != expected["forward_probes"]:
            raise ValueError("local reload: actual forward/generation differs from saved policy")
        guard(bundle)
    if original_access and original_access["denied_original_access_attempts"]:
        raise ValueError("local reload: original BF16 model access attempted")
    if replay is not None and (checkpoint.digest(restore_protocol(protocol)) != checkpoint.digest(launch["protocol"])
            or launch["binding"]["code_sha256"] != checkpoint.digest(approval.code_hashes())
            or any(checkpoint.file_hash(capture_root / name) != sha for name, sha in replay["paths"].items())
            or sorted(path.name for path in capture_root.glob("raw-*.json")) != sorted(
                name for name in replay["paths"] if name.startswith("raw-"))):
        raise ValueError("local reload: original repair evidence changed during model reload")
    result = {"status": "FRESH_PROCESS_RELOAD_VERIFIED", "pid": os.getpid(), "host": host_identity(),
              "capture_pid": launch["pid"],
              "step": step, "all_trainable_hashes_match": True, "optimizer_restored": True,
              "actual_forward_and_generation_match": True,
              "frozen_quantized_base_unchanged": True, "self_improvement_verified": False,
              "original_access_guard": original_access,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    if scale_options.scaled(protocol):
        # For BF16 the frozen-base key above means the unquantized base passed assert_base_unchanged.
        result["policy_precision"] = scale_options.options(protocol)["policy_precision"]
    if execution_contract is not None:
        result.update(execution_contract=execution_contract, execution_contract_sha256=checkpoint.digest(execution_contract),
                      capture_binding_sha256=checkpoint.digest(launch["binding"]))
    if replay is not None:
        result.update(fresh_reload_verified=True, repair_functional_verified=True,
            S_gradient_verified=replay["S_gradient_verified"], repair_learning_verified=replay["S_gradient_verified"],
            repair_replay_sha256=replay["sha256"], repair_replay_file_sha256=checkpoint.file_hash(output / "repair_replay.json"),
            repair_update_evidence_sha256=replay["paths"]["repair_update_evidence.json"],
            capture_binding_sha256=checkpoint.digest(launch["binding"]), repair_protocol_sha256=protocol["sha256"],
            benefit_verified=False)
    write_json(output / "result.json", result)
    return result


def _pool_main(parser, args) -> int:
    """Repair-notebook pool commands; the protocol is rebuilt from explicit flags exactly as for minimal."""
    if ((args.answer_verified_route or args.finalization_reward) and args.note_contract == DEFAULT_NOTE_CONTRACT
            or args.answer_verified_route is not bool(args.slotless_source)):
        parser.error("--answer-verified-route/--finalization-reward require the compact --note-contract; "
                     "--slotless-source is required exactly with --answer-verified-route")
    v2 = args.stage == "propose-pool" and args.rule_version == "v2"
    if args.stage == "propose-pool" and (args.exclude_source and not v2 or v2 and args.approval is None):
        parser.error("--exclude-source requires --rule-version v2; a v2 proposal requires --approval")
    frozen = frozen_approval_option(parser, args)
    if args.stage == "propose-proof":
        if (args.approval is None) == (frozen is None):
            parser.error("propose-proof takes exactly one of --approval or --frozen-approval/--frozen-approval-sha256")
        if args.train_problems is None or args.problems_per_update is None:
            parser.error("propose-proof requires the training run's explicit --train-problems N --problems-per-update K")
        if (args.passrate_selection is None) != (args.passrate_report is None):
            parser.error("--passrate-selection and --passrate-report are required together")
    from .cli import artifact_path
    from .local_data import (approved_eligible_ids, pool_sizes, prepare_pool_data, propose_pool_rule,
                             propose_proof_rule)

    try:
        protocol = build_protocol(repair_notebook=True, context_limit=24576, **repair_cli_keywords(args))
        if args.stage == "propose-proof":
            output = artifact_path(Path(args.output).absolute())
            # The approval's eligible IDs: a fresh validation, or the pinned saved output verified by hashes (pod).
            proof_eligible = (approved_eligible_ids(args.build, args.approval) if frozen is None
                              else frozen_approval_output(Path(args.build).absolute(), frozen)[1])
            rule = propose_proof_rule(args.build, local_prompts(protocol), eligible=proof_eligible, sizes=pool_sizes(protocol),
                sentinels_per_subject=args.sentinels_per_subject, seed=args.seed,
                excluded_sources=tuple(args.exclude_source),
                passrate=None if args.passrate_selection is None else (args.passrate_selection, args.passrate_report))
            write_json(output, rule)
            print(json.dumps({"status": "PROOF_RULE_PROPOSED_LABEL_BLIND", "output": str(output),
                              "fixed": len(rule["fixed"]), "dropped": len(rule["selection_source"]["dropped_ids"]),
                              "sentinels": {subject: len(ids) for subject, ids in rule["sentinels"].items()}}), flush=True)
            return 0
        if args.stage == "propose-pool":
            output = artifact_path(Path(args.output).absolute())
            # The approval's eligible IDs (allowlisted, outside AI-review-blocked sources) bound the proposal.
            eligible = None if args.approval is None else approved_eligible_ids(args.build, args.approval)
            options: dict[str, Any] = ({"version": "v2", "excluded_sources": tuple(args.exclude_source),
                                        "sizes": pool_sizes(protocol)} if v2 else {})
            rule = propose_pool_rule(args.build, local_prompts(protocol), seed=args.seed,
                                     evaluation_count=args.evaluation_count, eligible=eligible, **options)
            write_json(output, rule)
            print(json.dumps({"status": "POOL_RULE_PROPOSED_LABEL_BLIND", "output": str(output)}), flush=True)
            return 0
        result = prepare_pool_data(args.build, args.approval, args.rule, protocol, args.output, frozen_approval=frozen)
        print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
        return 0 if result["status"] == POOL_STATUS else 2
    except Exception:
        traceback.print_exc()
        return 1


def main(argv=None) -> int:
    from . import scale as scale_options

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--previous-request", type=Path, required=True)
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--context-limit", type=int, choices=(16384, 24576), default=16384)
    prepare_generation = prepare.add_mutually_exclusive_group()
    prepare_generation.add_argument("--partial-notebook", action="store_true")
    prepare_generation.add_argument("--referenced-notebook", action="store_true")
    prepare_generation.add_argument("--paired-notebook", action="store_true")
    prepare_generation.add_argument("--pair-action-notebook", action="store_true")
    prepare_generation.add_argument("--pair-draft-synthesis", action="store_true")
    prepare_reference = prepare.add_mutually_exclusive_group()
    prepare_reference.add_argument("--whole-reference-notebook", action="store_true")
    prepare_reference.add_argument("--whole-reference-pages", action="store_true")
    prepare.add_argument("--verified-note-assessment-credit", action="store_true")
    prepare_objective = prepare.add_mutually_exclusive_group()
    prepare_objective.add_argument("--verified-note-atoms", action="store_true")
    prepare_objective.add_argument("--verified-note-components", action="store_true")
    prepare_objective.add_argument("--verified-note-fact-credit", action="store_true")
    prepare_objective.add_argument("--verified-note-synthesis", action="store_true")
    prepare_objective.add_argument("--verified-note-synthesis-credit", action="store_true")
    prepare_objective.add_argument("--repair-notebook", action="store_true")
    prepare.add_argument("--note-contract", choices=NOTE_CONTRACT_CHOICES, default=DEFAULT_NOTE_CONTRACT)
    add_repair_cli_arguments(prepare)
    run = sub.add_parser("minimal")
    run.add_argument("--preparation", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--standalone-base", action="store_true")
    run.add_argument("--cache-check", type=Path)
    run.add_argument("--execution-policy", choices=(VERIFIED_UPDATE_EXPLORATION,))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--context-limit", type=int, choices=(16384, 24576), default=16384)
    run.add_argument("--min-free-gib", type=float, help="stop with DISK_LOW instead of writing the checkpoint "
                     "when the output filesystem has less free space (default: no check)")
    run_generation = run.add_mutually_exclusive_group()
    run_generation.add_argument("--partial-notebook", action="store_true")
    run_generation.add_argument("--referenced-notebook", action="store_true")
    run_generation.add_argument("--paired-notebook", action="store_true")
    run_generation.add_argument("--pair-action-notebook", action="store_true")
    run_generation.add_argument("--pair-draft-synthesis", action="store_true")
    run_reference = run.add_mutually_exclusive_group()
    run_reference.add_argument("--whole-reference-notebook", action="store_true")
    run_reference.add_argument("--whole-reference-pages", action="store_true")
    run.add_argument("--verified-note-assessment-credit", action="store_true")
    run_objective = run.add_mutually_exclusive_group()
    run_objective.add_argument("--verified-note-atoms", action="store_true")
    run_objective.add_argument("--verified-note-components", action="store_true")
    run_objective.add_argument("--verified-note-fact-credit", action="store_true")
    run_objective.add_argument("--verified-note-synthesis", action="store_true")
    run_objective.add_argument("--verified-note-synthesis-credit", action="store_true")
    run_objective.add_argument("--repair-notebook", action="store_true")
    run.add_argument("--note-contract", choices=NOTE_CONTRACT_CHOICES, default=DEFAULT_NOTE_CONTRACT)
    add_repair_cli_arguments(run)
    add_frozen_approval_arguments(run)
    reload = sub.add_parser("reload")
    reload.add_argument("--capture", type=Path, required=True)
    reload.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export-approval", help="save a fresh ai_reviewed_local approval output of a union vault "
                            f"for {FROZEN_APPROVAL_VERIFY} (prints its SHA-256 to pin)")
    for name in ("build", "approval", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    pool_parsers = {name: sub.add_parser(name) for name in ("propose-pool", "prepare-pool", "propose-proof")}
    for name, pool in pool_parsers.items():
        pool.add_argument("--build", type=Path, required=True)
        pool.add_argument("--note-contract", choices=NOTE_CONTRACT_CHOICES, default=DEFAULT_NOTE_CONTRACT)
        add_repair_cli_arguments(pool)
        pool.add_argument("--output", type=Path, required=True)
    pool_parsers["propose-pool"].add_argument("--seed", type=int, default=3407)
    pool_parsers["propose-pool"].add_argument("--evaluation-count", type=int, default=12)
    pool_parsers["propose-pool"].add_argument("--rule-version", choices=("v1", "v2"), default="v1",
        help="v2: pool-local-selection-v2 (existing datasets only; N/K from the scale options; probe-only actors)")
    pool_parsers["propose-pool"].add_argument("--exclude-source", action="append", default=[],
        help="v2 only; repeatable source name or glob, added to the default exclusions")
    pool_parsers["propose-pool"].add_argument("--approval", type=Path,
        help="ai_reviewed_local data request; its eligible IDs bound the proposal (required for v2)")
    pool_parsers["prepare-pool"].add_argument("--approval", type=Path, required=True,
                                              help="ai_reviewed_local data request (pool_data_evidence produce)")
    pool_parsers["prepare-pool"].add_argument("--rule", type=Path, required=True)
    add_frozen_approval_arguments(pool_parsers["prepare-pool"])
    proof = pool_parsers["propose-proof"]
    proof.add_argument("--seed", type=int, default=3407)
    proof.add_argument("--sentinels-per-subject", type=int, required=True)
    proof.add_argument("--exclude-source", action="append", default=[],
                       help="repeatable source name or glob, added to the default exclusions")
    proof.add_argument("--approval", type=Path, help="ai_reviewed_local data request (fresh approval; local only)")
    add_frozen_approval_arguments(proof)
    proof.add_argument("--passrate-selection", type=Path, help="passrate select output (absent: all eligible items)")
    proof.add_argument("--passrate-report", type=Path, help="the measurement report the selection was derived from")
    args = parser.parse_args(argv)
    if args.stage == "export-approval":
        from .local_data import export_approval
        try:
            exported = export_approval(args.build, args.approval, args.output)
        except Exception:
            traceback.print_exc()
            return 1
        print(json.dumps({"status": "APPROVAL_OUTPUT_EXPORTED", **exported}), flush=True)
        return 0
    if args.stage in pool_parsers:
        return _pool_main(parser, args)
    if args.stage != "reload" and args.repair_notebook and (
            args.context_limit != 24576 or any(getattr(args, key) for key in (
                "verified_note_atoms", "verified_note_components", "verified_note_fact_credit", "verified_note_synthesis",
                "verified_note_synthesis_credit", "verified_note_assessment_credit", "partial_notebook", "referenced_notebook",
                "paired_notebook", "pair_action_notebook", "pair_draft_synthesis", "whole_reference_notebook", "whole_reference_pages"))
            or getattr(args, "execution_policy", None) is not None):
        parser.error("--repair-notebook requires exclusive --context-limit 24576 without a legacy execution policy")
    if args.stage != "reload" and args.note_contract != DEFAULT_NOTE_CONTRACT and not args.repair_notebook:
        parser.error("--note-contract other than the default requires --repair-notebook")
    scale_kw: dict[str, Any] = scale_options.from_args(args) if args.stage != "reload" else {}
    if scale_kw and not args.repair_notebook:
        parser.error("scale options (--rollout-batch-rows, --policy-precision, ...) require --repair-notebook")
    if args.stage != "reload" and ((args.answer_verified_route or args.finalization_reward)
            and (not args.repair_notebook or args.note_contract == DEFAULT_NOTE_CONTRACT)
            or args.answer_verified_route is not bool(args.slotless_source)):
        parser.error("--answer-verified-route/--finalization-reward require --repair-notebook with the compact "
                     "--note-contract; --slotless-source is required exactly with --answer-verified-route")
    if args.stage != "reload" and (args.turn1_contract is not None and not args.answer_verified_route
                                   or args.turn1_max_new_tokens is not None and args.turn1_contract is None):
        parser.error("--turn1-contract requires --answer-verified-route; --turn1-max-new-tokens requires --turn1-contract")
    if (args.stage != "reload" and args.context_limit == 24576
            and not (args.verified_note_fact_credit or args.verified_note_synthesis or args.verified_note_synthesis_credit or args.repair_notebook)):
        parser.error("--context-limit 24576 requires --verified-note-fact-credit or --verified-note-synthesis")
    if args.stage != "reload" and args.verified_note_synthesis_credit and args.context_limit != 24576:
        parser.error("--verified-note-synthesis-credit requires --context-limit 24576")
    if args.stage != "reload" and args.partial_notebook and not args.verified_note_synthesis_credit:
        parser.error("--partial-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.stage != "reload" and args.referenced_notebook and not args.verified_note_synthesis_credit:
        parser.error("--referenced-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.stage != "reload" and args.paired_notebook and not args.verified_note_synthesis_credit:
        parser.error("--paired-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.stage != "reload" and args.pair_action_notebook and not args.verified_note_synthesis_credit:
        parser.error("--pair-action-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.stage != "reload" and args.pair_draft_synthesis and not args.verified_note_synthesis_credit:
        parser.error("--pair-draft-synthesis requires --verified-note-synthesis-credit --context-limit 24576")
    if args.stage != "reload" and args.whole_reference_notebook and not args.pair_draft_synthesis:
        parser.error("--whole-reference-notebook requires --pair-draft-synthesis")
    if args.stage != "reload" and args.whole_reference_pages and not args.pair_draft_synthesis:
        parser.error("--whole-reference-pages requires --pair-draft-synthesis")
    if (args.stage != "reload" and args.verified_note_assessment_credit
            and (not args.verified_note_synthesis_credit or not args.pair_draft_synthesis
                 or not args.whole_reference_notebook or args.whole_reference_pages or args.context_limit != 24576
                 or getattr(args, 'execution_policy', None) is not None)):
        parser.error("--verified-note-assessment-credit requires monolithic --whole-reference-notebook --pair-draft-synthesis --verified-note-synthesis-credit --context-limit 24576 without exploration override")
    try:
        if args.stage == "prepare":
            from .local_data import prepare_local_data
            result = prepare_local_data(args.previous_request, args.selection,
                                        build_protocol(verified_note_atoms=args.verified_note_atoms,
                                            verified_note_components=args.verified_note_components,
                                            verified_note_fact_credit=args.verified_note_fact_credit,
                                            verified_note_synthesis=args.verified_note_synthesis,
                                            verified_note_synthesis_credit=args.verified_note_synthesis_credit,
                                            verified_note_assessment_credit=args.verified_note_assessment_credit,
                                            partial_notebook=args.partial_notebook,
                                            referenced_notebook=args.referenced_notebook,
                                            paired_notebook=args.paired_notebook,
                                            pair_action_notebook=args.pair_action_notebook,
                                            pair_draft_synthesis=args.pair_draft_synthesis,
                                            whole_reference_notebook=args.whole_reference_notebook,
                                            whole_reference_pages=args.whole_reference_pages,
                                            context_limit=args.context_limit,
                                            **({"repair_notebook": True} if args.repair_notebook else {}),
                                            **repair_cli_keywords(args)), args.output)
            print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
            return 0 if result["status"] == "LOCAL_DATA_REVALIDATED" else 2
        elif args.stage == "minimal":
            capture_options: dict[str, Any] = {}
            if args.repair_notebook:
                capture_options["repair_notebook"] = True
            if args.execution_policy is not None:
                capture_options["execution_policy"] = args.execution_policy
            frozen = frozen_approval_option(parser, args)
            if frozen is not None:
                capture_options["frozen_approval"] = frozen
            # The same shared flag -> keyword mapping quantized_cache uses for its readiness protocol.
            capture_options.update(repair_cli_keywords(args))
            if args.min_free_gib is not None:
                capture_options["min_free_gib"] = args.min_free_gib
            captured = capture(args.preparation, args.model_path, args.output, standalone=args.standalone_base,
                    cache_check=args.cache_check, verified_note_atoms=args.verified_note_atoms,
                    verified_note_components=args.verified_note_components,
                    verified_note_fact_credit=args.verified_note_fact_credit,
                    verified_note_synthesis=args.verified_note_synthesis,
                    verified_note_synthesis_credit=args.verified_note_synthesis_credit,
                    verified_note_assessment_credit=args.verified_note_assessment_credit,
                    partial_notebook=args.partial_notebook, referenced_notebook=args.referenced_notebook,
                    paired_notebook=args.paired_notebook,
                    pair_action_notebook=args.pair_action_notebook,
                    pair_draft_synthesis=args.pair_draft_synthesis,
                    whole_reference_notebook=args.whole_reference_notebook,
                    whole_reference_pages=args.whole_reference_pages,
                    context_limit=args.context_limit,
                    **capture_options)
            if (captured or {}).get("status") == "DISK_LOW":
                return 2
        else:
            print(json.dumps(reload_capture(args.capture, args.output)), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
