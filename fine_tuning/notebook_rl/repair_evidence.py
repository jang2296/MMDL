"""Bind recorded repair VJPs to fresh CPU semantics; never certify a new VJP or F/E."""
from __future__ import annotations

import json
import math

import torch

from .engines import RolloutRequest
from .local_profile import REPAIR_PROTOCOL_ID, restore_protocol
from .losses import mis_mask
from .repair_credit import ComponentCredit, _digest
from . import compact_note
from .episode_notebook import ANSWER_ROUTE_RULE, ANSWER_VERIFIED
from .repair_notebook import S_DEFINITIONS, TEXT_S_DEFINITION, note_contract
from .rollout import _hash
from .schema import ProblemTree, to_json
from .updates import _PreparedAction, _repair_action_journal, _repair_vectors, _retained_repair_signal

VERSION = "repair-update-evidence-v1"
_DTYPES = {str(value): value for value in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
_GROUPS = ("L", "M", "V")


def _same(actual, expected, name: str) -> None:
    # Canonical JSON distinguishes True/1 and rejects NaN at persisted trust boundaries.
    if _hash(actual) != _hash(expected):
        raise ValueError(f"repair evidence: {name} differs")


def _number(value, *, minimum: float | None = None) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or minimum is not None and value < minimum):
        raise ValueError("repair evidence: finite numeric value required")
    return float(value)


def _inventory(recorded: list | None, expected: list) -> None:
    fields = ("group", "name", "shape", "dtype", "numel")
    if type(recorded) is not list or not recorded or type(expected) is not list or not expected:
        raise ValueError("repair evidence: actual trainable parameter inventory required")
    counters = dict.fromkeys(_GROUPS, 0)
    names = set()
    for row in recorded:
        if (type(row) is not dict or set(row) != {*fields, "index"} or row["group"] not in counters
                or type(row["index"]) is not int or row["index"] != counters[row["group"]]
                or type(row["name"]) is not str or not row["name"] or row["name"] in names
                or type(row["shape"]) is not list or any(type(size) is not int or size <= 0 for size in row["shape"])
                or type(row["numel"]) is not int or row["numel"] != math.prod(row["shape"])
                or row["dtype"] not in _DTYPES):
            raise ValueError("repair evidence: invalid parameter partition")
        counters[row["group"]] += 1
        names.add(row["name"])
    if (not all(counters.values()) or any(type(row) is not dict or not set(fields) <= set(row) for row in expected)
            or any(type(row["name"]) is not str for row in expected)
            or len({row["name"] for row in expected}) != len(expected)):
        raise ValueError("repair evidence: incomplete parameter inventory")
    _same({row["name"]: {key: row[key] for key in fields} for row in recorded},
          {row["name"]: {key: row[key] for key in fields} for row in expected}, "physical parameter inventory")


def _text_aggregate(aggregate: dict, members: list[dict], text_definition: str | None) -> bool | None:
    """S_text is separate evidence for text-only routes; it never satisfies or alters S."""
    if text_definition is None:
        if "S_text" in aggregate or any("text_verified_tokens" in row for row in members):
            raise ValueError("repair evidence: S_text without its bound answer-verified route")
        return None
    text = aggregate.get("S_text")
    if aggregate.get("measured") is False:
        if (type(text) is not dict or text != {"s_definition": text_definition,
                "scope": "text_only_route_note_policy_loss_only", "measured": False,
                "retained_text_verified_tokens": sum(row.get("text_verified_tokens", 0) for row in members),
                "S": None, "counts_as_S": False}):
            raise ValueError("repair evidence: unmeasured S_text aggregate differs")
        return None
    if (type(text) is not dict or set(text) != {"s_definition", "scope", "group_norms", "retained_text_verified_tokens",
                                                "S", "counts_as_S"}
            or text["s_definition"] != text_definition or text["scope"] != "text_only_route_note_policy_loss_only"
            or text["counts_as_S"] is not False or type(text["group_norms"]) is not dict
            or set(text["group_norms"]) != set(_GROUPS)):
        raise ValueError("repair evidence: separate S_text aggregate required")
    active = [row for row in members if row.get("text_verified_tokens", 0) > 0]
    tokens = sum(row.get("text_verified_tokens", 0) for row in members)
    _same(text["retained_text_verified_tokens"], tokens, "text-only verified token count")
    for group, value in text["group_norms"].items():
        _number(value, minimum=0)
        if (not active or group == "V" and not any(row["vision_on"] for row in active)) and value != 0:
            raise ValueError("repair evidence: S_text gradient without active text tokens or Vision route")
    status = bool(tokens and any(value > 0 for value in text["group_norms"].values()))
    _same(text["S"], status, "S_text assertion")
    return status


def _unmeasured(aggregate: dict, members: list[dict], parameter_inventory: list, s_definition: str) -> None:
    """A non-audit update under a fixed audit rule: original membership only, never an S claim."""
    keys = {"scope", "basis", "measured", "benefit_verified", "retained_positive_verified_tokens", "S",
            "s_definition", "parameter_inventory", "members", "parameter_inventory_sha256", "members_sha256"}
    if (set(aggregate) - {"S_text"} != keys or aggregate["scope"] != "native_corrected_repair_policy_loss_only"
            or aggregate["basis"] != "not_measured_fixed_audit_rule" or aggregate["S"] is not None
            or aggregate["benefit_verified"] is not False or aggregate["s_definition"] != s_definition):
        raise ValueError("repair evidence: unmeasured aggregate differs")
    _inventory(aggregate.get("parameter_inventory"), parameter_inventory)
    _same(aggregate["parameter_inventory_sha256"], _hash(aggregate["parameter_inventory"]), "inventory hash")
    _same(aggregate["members"], members, "VJP original token membership")
    _same(aggregate["members_sha256"], _hash(members), "VJP membership hash")
    _same(aggregate["retained_positive_verified_tokens"], sum(row["positive_verified_tokens"] for row in members),
          "positive native token count")


def _aggregate(aggregate: dict | None, members: list[dict], parameter_inventory: list,
               s_definition: str = "native-corrected-v1", *, allow_unmeasured: bool = False) -> bool | None:
    if type(aggregate) is dict and aggregate.get("measured") is False:
        if not allow_unmeasured:
            raise ValueError("repair evidence: unmeasured aggregate requires the bound audit rule")
        _unmeasured(aggregate, members, parameter_inventory, s_definition)
        return None
    if (type(aggregate) is not dict
            or aggregate.get("s_definition") != s_definition
            or aggregate.get("scope") != "native_corrected_repair_policy_loss_only"
            or aggregate.get("basis") != "same_forward_vjp_sum_by_parameter_identity_cpu_float64"
            or aggregate.get("finite") is not True or aggregate.get("benefit_verified") is not False):
        raise ValueError("repair evidence: native signed and positive aggregate required")
    _inventory(aggregate.get("parameter_inventory"), parameter_inventory)
    _same(aggregate.get("parameter_inventory_sha256"), _hash(aggregate["parameter_inventory"]), "inventory hash")
    _same(aggregate.get("members"), members, "VJP original token membership")
    _same(aggregate.get("members_sha256"), _hash(members), "VJP membership hash")
    positive = sum(row["positive_verified_tokens"] for row in members)
    _same(aggregate.get("retained_positive_verified_tokens"), positive, "positive native token count")
    channels, combined = aggregate.get("channels"), aggregate.get("combined")
    if (type(channels) is not dict or set(channels) != {"online", "final"}
            or type(combined) is not dict or set(combined) != {"signed", "positive"}):
        raise ValueError("repair evidence: aggregate channels differ")
    for component in ("online", "final"):
        if type(channels[component]) is not dict or set(channels[component]) != {"signed", "positive"}:
            raise ValueError("repair evidence: channel kinds differ")
        for kind, count_key in (("signed", "signed_tokens"), ("positive", "positive_verified_tokens")):
            norms = channels[component][kind]
            if type(norms) is not dict or set(norms) != set(_GROUPS):
                raise ValueError("repair evidence: channel parameter groups differ")
            active = [row for row in members if row["component"] == component and row[count_key] > 0]
            for group, value in norms.items():
                _number(value, minimum=0)
                if (not active or group == "V" and not any(row["vision_on"] for row in active)) and value != 0:
                    raise ValueError("repair evidence: gradient without active native tokens or Vision route")
    for kind, norms in combined.items():
        if type(norms) is not dict or set(norms) != set(_GROUPS):
            raise ValueError("repair evidence: combined parameter groups differ")
        for group, value in norms.items():
            value = _number(value, minimum=0)
            left, right = (channels[component][kind][group] for component in ("online", "final"))
            # Only FP64 summation roundoff; this does not substitute for remeasuring VJPs.
            slack = 64 * torch.finfo(torch.float64).eps * max(left, right, value)
            if value > left + right + slack or value < abs(left - right) - slack:
                raise ValueError("repair evidence: impossible combined gradient norm")
    status = bool(positive and all(any(value > 0 for value in norms.values()) for norms in combined.values()))
    _same(aggregate.get("S"), status, "S assertion")
    return status


def repair_update_evidence(update: dict, protocol: dict, *, trees: list[ProblemTree], preparations: dict,
                           parameter_inventory: list, require_s: bool = False, control: bool = False) -> dict:
    """Caller must freshly prepare original trees; hashes alone are not semantic approval."""
    canonical = restore_protocol(protocol)
    _same(protocol, canonical, "canonical protocol")
    if (protocol["protocol_id"] != REPAIR_PROTOCOL_ID or type(update) is not dict
            or type(require_s) is not bool or type(control) is not bool
            or type(trees) is not list or not trees or any(type(tree) is not ProblemTree for tree in trees)
            or len({tree.problem_id for tree in trees}) != len(trees) or type(preparations) is not dict
            or set(preparations) != {tree.problem_id for tree in trees}
            or update.get("status") != "UPDATED_UNSYNCHRONIZED" or update.get("engine_sync_required") is not True
            or type(update.get("step_before")) is not int or update["step_before"] < 0
            or type(update.get("step_after")) is not int or update["step_after"] != update["step_before"] + 1
            or update.get("notebook_reward_enabled") is not (not control)
            or update.get("repair_learning_verified") is not False or update.get("vision_health_disabled") is not False):
        raise ValueError("repair evidence: explicit committed same-policy update required")
    _same(update.get("repair_execution"), {"protocol_id": REPAIR_PROTOCOL_ID, "protocol_sha256": protocol["sha256"],
        "algorithm": "ALG-NB3", "arm": "MOD-RV", "seed": 3407}, "execution")
    _same(update.get("repair_preparation"), preparations, "fresh preparation")
    s_definition = S_DEFINITIONS[note_contract(protocol)]
    text_definition = TEXT_S_DEFINITION if protocol.get("repair_route_policy") == ANSWER_ROUTE_RULE else None
    allowed_routes = ("VISION_ON", "VISION_OFF") + ((ANSWER_VERIFIED,) if text_definition is not None else ())
    labels = {}
    final_credit = (compact_note.CREDIT_VERSION if note_contract(protocol) == compact_note.VERSION
                    else "repair-group-credit-v1")
    rows, groups, routes = {}, {}, {}
    samples = []
    for tree in trees:
        receipt = preparations[tree.problem_id]
        if (type(tree.policy_version) is not int or tree.policy_version != update["step_before"]
                or type(receipt) is not dict or receipt.get("version") != "repair-tree-credit-v1"
                or receipt.get("admitted") is not True or receipt.get("has_signal") is not True
                or receipt.get("notebook_reward_enabled") is not (not control)
                or receipt.get("route") not in allowed_routes
                or receipt.get("sha256") != _digest({key: value for key, value in receipt.items() if key != "sha256"})):
            raise ValueError("repair evidence: fresh admitted tree receipt required")
        _same(receipt["problem_id"], tree.problem_id, "tree ID")
        _same(receipt["policy_version"], tree.policy_version, "tree policy")
        _same(receipt["source_bindings"]["tree_sha256"], _digest(json.loads(to_json(tree))), "original tree")
        if receipt["source_bindings"].get("fresh_semantics_verified") is not True:
            raise ValueError("repair evidence: fresh semantics required")
        routes[tree.problem_id] = receipt["route"] == "VISION_ON"
        labels[tree.problem_id] = receipt["route"]
        if len(tree.groups) != len(receipt["groups"]):
            raise ValueError("repair evidence: full original groups required")
        for group, source in zip(tree.groups, receipt["groups"], strict=True):
            if group.group_id in groups:
                raise ValueError("repair evidence: duplicate group")
            size, cap = (4, 8192) if group.kind == "DIAG" else (4 if group.kind == "T1" else 2, 1024)
            if group.kind == "T1" and "repair_turn1_contract" in protocol:
                cap = protocol["repair_turn1_contract"]["max_new_tokens"]  # opt-in eval-aligned T1 cap
            if group.kind not in ("T1", "T2", "T3", "DIAG") or len(group.samples) != size:
                raise ValueError("repair evidence: fixed group size or kind differs")
            _same([source["group_id"], source["kind"], source["original_group_size"], source["completion_cap"],
                   source["members"], source["source_group_sha256"], source["group_weight"]],
                  [group.group_id, group.kind, size, cap, [sample.sample_id for sample in group.samples],
                   _digest(json.loads(to_json(group))), protocol["loss"]["group_weights"][group.kind]], "original group")
            if len(source["samples"]) != size:
                raise ValueError("repair evidence: original member removed")
            groups[group.group_id] = source
            for sample, row in zip(group.samples, source["samples"], strict=True):
                if sample.sample_id in rows or sample.sample_id != row["sample_id"]:
                    raise ValueError("repair evidence: duplicate or reordered sample")
                _same([sample.policy_version, sample.problem_id, sample.group_id, sample.kind],
                      [tree.policy_version, tree.problem_id, group.group_id, group.kind], "sample identity")
                rows[sample.sample_id] = row
                samples.append(sample)
    counts = {kind: sum(group["kind"] == kind for group in groups.values()) for kind in ("T1", "T2", "T3", "DIAG")}
    for key, value in (("n_groups_by_kind", counts), ("n_problems", len(trees)), ("routes", routes),
                       ("no_action_samples", 0), ("notebook_backward", [])):
        _same(update.get(key), value, key)
    if any(label not in ("VISION_ON", "VISION_OFF") for label in labels.values()):
        _same(update.get("route_labels"), labels, "route labels")
    elif "route_labels" in update:
        raise ValueError("repair evidence: route labels without a non-binary admitted route")
    _same(update.get("effective_rewards"), {key: [row["answer_reward"] for row in group["samples"]]
                                           for key, group in groups.items()}, "effective answer rewards")
    journal = update.get("repair_actions")
    if type(journal) is not list or len(journal) != len(samples):
        raise ValueError("repair evidence: complete original old-logprob journal required")
    _same(update.get("repair_actions_sha256"), _hash(journal), "action journal hash")
    denominator = sum(protocol["loss"]["group_weights"][kind] * count for kind, count in counts.items())
    actions, credits, deltas = [], {}, {}
    for sample, saved in zip(samples, journal, strict=True):
        row, group = rows[sample.sample_id], groups[sample.group_id]
        if (type(saved) is not dict or saved.get("logprob_dtype") not in _DTYPES
                or type(saved.get("old_logprobs")) is not list
                or len(saved["old_logprobs"]) != len(sample.completion_token_ids)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in saved["old_logprobs"])):
            raise ValueError("repair evidence: original finite logprobs and dtype required")
        old = torch.tensor([saved["old_logprobs"]], dtype=_DTYPES[saved["logprob_dtype"]])
        _same(saved["old_logprobs"], old[0].tolist(), "representable old logprobs")
        _same(row["action_mask"], [True] * len(sample.completion_token_ids), "full original action mask")
        mask = torch.tensor([row["action_mask"]], dtype=torch.bool)
        sampler = torch.tensor([sample.sampler_logprobs], dtype=old.dtype)
        keep, delta = mis_mask(old, sampler, mask, protocol["loss"]["mismatch"]["delta_seq"])
        action = _PreparedAction(sample, RolloutRequest(sample.sample_id, sample.prompt_token_ids, None, [],
            1, group["completion_cap"], [sample.seed]), mask, row["answer_advantage"],
            group["group_weight"] / denominator, group["original_group_size"] * group["completion_cap"],
            routes[sample.problem_id], old=old, keep=bool(keep.item()))
        actions.append(action)
        stored_delta = _number(update.get("mean_log_ratio_by_sample", {}).get(sample.sample_id))
        work_dtype = torch.float32 if old.dtype in (torch.float16, torch.bfloat16) else old.dtype
        producer_delta = torch.tensor(stored_delta, dtype=work_dtype)
        if (float(producer_delta.item()) != stored_delta
                or bool((producer_delta.abs() <= protocol["loss"]["mismatch"]["delta_seq"]).item()) != action.keep):
            raise ValueError("repair evidence: stored MIS mean and retained decision differ")
        # Device reduction roundoff only; the retained/excluded decision must still match exactly.
        tolerance = 8 * torch.finfo(work_dtype).eps * max(1., float((old.double() - sampler.double()).abs().mean()))
        if abs(stored_delta - float(delta.item())) > tolerance:
            raise ValueError("repair evidence: old/sampler log ratio differs")
        deltas[sample.sample_id] = stored_delta
        wire = row["component_credit"]
        if wire is not None and (type(wire) is not dict or type(wire.get("source_bindings")) is not dict
                                 or wire["source_bindings"].get("version") != (
                                     final_credit if group["kind"] == "DIAG" else "repair-group-credit-v1")):
            raise ValueError("repair evidence: credit version differs from the protocol's final-note contract")
        if wire is not None:
            credits[sample.sample_id] = ComponentCredit(**(wire | {"coefficients": tuple(wire["coefficients"]),
                                                                 "unit_rows": tuple(wire["unit_rows"])}))
    _same(update.get("mean_log_ratio_by_sample"), deltas, "log ratio membership")
    _same(journal, _repair_action_journal(actions, rows), "replayed action journal")
    _same(update.get("masked_fraction"), sum(not action.keep for action in actions) / len(actions), "MIS fraction")
    signal = _retained_repair_signal(actions, rows)
    if signal["policy_gradient_present"] is not True:
        raise ValueError("repair evidence: committed update has no retained policy coefficient")
    _same(update.get("retained_repair_signal"), signal, "retained policy signal")
    if "retained_learning_signal" in update:
        _same(update["retained_learning_signal"], signal, "required retained signal")
    members = []
    if not control:
        for retained in (False, True):
            for action in actions:
                if action.keep is retained and action.sample.sample_id in credits:
                    assert action.old is not None
                    members.append(_repair_vectors(action, credits[action.sample.sample_id], action.old, action.mask)[1])
    from .scale import options as scale_options

    audit_rule = scale_options(protocol)["evidence_audit_rule"] is not None
    status = _aggregate(update.get("repair_aggregate_gradient"), members, parameter_inventory, s_definition,
                        allow_unmeasured=audit_rule)
    text_status = _text_aggregate(update["repair_aggregate_gradient"], members, text_definition)
    if require_s and status is not True:
        raise ValueError(f"repair evidence: positive aggregate S ({s_definition}) required")
    for key in ("loss", "kl_mean", "max_current_old_difference"):
        _number(update.get(key), minimum=None if key == "loss" else 0)
    if update["max_current_old_difference"] >= 1e-3:
        raise ValueError("repair evidence: current/old policy mismatch")
    result = {"version": VERSION, "S_gradient_verified": status, "repair_learning_verified": False,
        "functional_verified": False, "benefit_verified": False, "step_before": update["step_before"],
        "step_after": update["step_after"], "problem_ids": [tree.problem_id for tree in trees],
        "preparation_sha256": _hash(preparations), "parameter_inventory_sha256": _hash(parameter_inventory),
        "update_sha256": _hash(update), "scope": {"fresh_receipt_binding": True, "oldLP_MIS_replayed": True,
                                                 "VJP_remeasured": False}}
    if text_definition is not None:
        result.update(S_text_gradient_verified=text_status, S_text_definition=text_definition)
    if status is None:
        result["scope"]["vjp_audit"] = "NOT_AUDITED_FIXED_RULE_S_UNKNOWN"
    elif audit_rule:
        result["scope"]["vjp_audit"] = "AUDITED_FIXED_RULE"
    return json.loads(json.dumps(result, allow_nan=False))
