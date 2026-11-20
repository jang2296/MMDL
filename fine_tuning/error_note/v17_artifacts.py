"""Immutable bridge from the original training checkout to a frozen evaluator."""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess

from .audit import digest
from .prep_v16b import file_sha, read_sealed

VERSION = "v17-training-artifacts-v1"
TRAINING_VERSIONS = ("error-note-v17", "error-note-v18", "error-note-v18b")
ROOT = Path(__file__).resolve().parents[2]


def _safe(path: Path) -> Path:
    from .v18b_recovery import resolve_recovery_path

    path = resolve_recovery_path(Path(path).absolute())
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("v17 artifacts: symlink or path escape refused")
    return path


def _ref(path: Path) -> dict:
    path = _safe(path)
    if not path.is_file():
        raise ValueError("v17 artifacts: bound file is missing")
    return {"path": str(path), "file_sha256": file_sha(path)}


def _check_ref(ref: dict) -> Path:
    original = Path(ref["path"])
    if not original.is_absolute() or str(original) != ref["path"] or ".." in original.parts:
        raise ValueError("v17 artifacts: canonical absolute reference required")
    path = _safe(original)
    if set(ref) != {"path", "file_sha256"} or _ref(path)["file_sha256"] != ref["file_sha256"]:
        raise ValueError("v17 artifacts: bound file bytes changed")
    return path


def _same_ref(left: object, right: object) -> bool:
    return isinstance(left, dict) and isinstance(right, dict) and _check_ref(left) == _check_ref(right)


def _artifact_paths(value: dict):
    if value.get("training_protocol_version") != "error-note-v18b":
        return nullcontext()
    from .v18b_recovery import recovery_paths

    return recovery_paths(_safe(Path(value["campaign_plan"]["path"])).parent)


def _files(root: Path) -> list[dict]:
    root = _safe(root)
    if not root.is_dir():
        raise ValueError("v17 artifacts: artifact directory is missing")
    records = []
    for path in sorted(root.rglob("*")):
        _safe(path)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("v17 artifacts: nonregular artifact file refused")
        records.append({"path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size, "sha256": file_sha(path)})
    if not records:
        raise ValueError("v17 artifacts: empty artifact directory")
    return records


def _tree(root: Path) -> dict:
    records = _files(root)
    return {"root": str(_safe(root)), "files": records, "files_sha256": digest(records)}


def protected_evaluation(repo: Path) -> dict:
    """Hash protected evaluation bytes and the preregistration/ledger format, not engine glue."""
    repo = _safe(repo)
    paths = sorted((repo / "src/mmdl/evaluation").rglob("*.py"))
    paths += sorted((repo / "prompts").glob("mmmu_*_v1.txt"))
    paths += [repo / "configs/eval/mmmu_val_v8.yaml",
              repo / "fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml",
              repo / "third_party/mmmu/eval_utils.py", repo / "src/mmdl/runtime/contracts.py"]
    if not paths or not (repo / "src/mmdl/evaluation/engine.py").is_file():
        raise ValueError("v17 artifacts: protected evaluation source missing")
    records = [{"path": p.relative_to(repo).as_posix(), **{k: v for k, v in _ref(p).items() if k != "path"}}
               for p in paths]
    path = _safe(repo / "fine_tuning/notebook_rl/proof_eval.py")
    module = ast.parse(path.read_text(encoding="utf-8"))
    names = {"load_preregistration", "read_ledger", "_check_event", "_parse", "_append", "ledger_start", "ledger_fail",
             "_single_definition", "_paired_run_rules", "_variant_entry", "_scorer", "variant_registration"}
    contract = [ast.dump(node, include_attributes=False) for node in module.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (node.name.startswith("build_preregistration") or node.name in names)]
    if not contract or not any(isinstance(n, ast.FunctionDef) and n.name == "load_preregistration" for n in module.body):
        raise ValueError("v17 artifacts: preregistration contract missing")
    body = {"files": records, "preregistration_contract_sha256": digest(contract)}
    return body | {"sha256": digest(body)}


def evaluation_identity(repo: Path, commit: str) -> dict:
    from . import single

    repo = _safe(repo)
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != commit or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("v17 artifacts: frozen evaluation commit differs")
    if repo.resolve() != ROOT.resolve():
        raise ValueError("v17 artifacts: evaluator must run from its declared checkout")
    return {"code_commit": commit, "source_sha256": digest(single.source_manifest()["actual_files"])}


def _v18b_training_evidence(output: Path, plan: dict, step: int, selection: dict | None = None) -> tuple[dict, dict]:
    from .v17_preflight import _validate_live_capacity
    from .v18b_hotfix import lineage
    from .v18b_recovery import check_row_admission

    identity = {"code_commit": plan["source"]["base_commit"], "source_sha256": digest(plan["source"]["actual_files"]),
                "campaign_plan_sha256": plan["sha256"], "protocol_sha256": plan["protocol"]["sha256"],
                "training_protocol_version": "error-note-v18b"}
    hotfixes = lineage(output, plan)
    preparation_only = bool(selection and selection.get("preparation_only") is True)
    evidence_end = step
    stopped_timing: dict = {}
    if preparation_only:
        stopped_path, selected_path = output / "campaign-state.json", output / "evaluation-checkpoint-state.json"
        stopped, selected = read_sealed(stopped_path), read_sealed(selected_path)
        prep_marker = output / "prep-checkpoint.json"
        if (selection is None or selection.get("reason") != "MONITOR_STOP" or selection.get("consumed_items") != 0
                or selection.get("authorized_action") != "evaluate_checkpoint" or step != len(plan["prep_batches"])
                or plan.get("execution_settings", {}).get("monitor_stop_action") != "evaluate_checkpoint"
                or stopped.get("plan_sha256") != plan["sha256"] or stopped.get("status") != "STOPPED"
                or "MONITOR_STOP" not in stopped.get("stop_reasons", [])
                or selected.get("plan_sha256") != plan["sha256"]
                or selected.get("status") != "PARTIAL_EVALUATION_CANDIDATE" or selected.get("evaluation_selection") != selection
                or selected.get("stopped_training_state_sha256") != stopped["sha256"]
                or not _same_ref(selected.get("stopped_training_checkpoint"), stopped.get("checkpoint"))
                or selected.get("cursor", {}).get("slot") != step
                or selected.get("cursor", {}).get("consumed_problem_ids") != []
                or not _same_ref(json.loads(prep_marker.read_text()), selected.get("checkpoint"))
                or selected.get("checkpoint", {}).get("file_sha256") != selection.get("source_checkpoint_sha256")):
            raise ValueError("v18b artifacts: preparation-only checkpoint lacks its stopped-run timing provenance")
        evidence_end = stopped.get("cursor", {}).get("slot")
        if type(evidence_end) is not int or not step < evidence_end <= plan["total_slots"]:
            raise ValueError("v18b artifacts: preparation-only selection needs a later stopped production run")
        stopped_timing = {"campaign_state": _ref(stopped_path), "evaluation_selection_state": _ref(selected_path),
                          "preparation_checkpoint": _ref(prep_marker), "included_in_selected_checkpoint": False}
    updates = []
    for path in (output / "update-results").glob("slot-*.json"):
        filename_slot = int(path.stem.removeprefix("slot-"))
        if filename_slot >= evidence_end:
            continue
        row = read_sealed(_safe(path))
        slot = row.get("global_slot")
        if type(slot) is not int or slot < 0 or slot != filename_slot:
            raise ValueError("v18b artifacts: invalid production slot")
        if (row.get("version") != "v18b-update-result-v1" or row.get("plan_sha256") != plan["sha256"]
                or type(row.get("producer_pid")) is not int or row["producer_pid"] < 1
                or row.get("timing_scope") != "encode_backward_optimizer"
                or type(row.get("result", {}).get("optimizer_steps")) is not int
                or type(row.get("wall_seconds")) not in (int, float)
                or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0):
            raise ValueError("v18b artifacts: actual production update timing required")
        transaction_path = output / "transactions" / f"slot-{slot}.json"
        transaction = read_sealed(_safe(transaction_path))
        if (row.get("transaction_sha256") != transaction["sha256"]
                or transaction.get("plan_sha256") != plan["sha256"] or transaction.get("slot") != slot):
            raise ValueError("v18b artifacts: timing is not bound to its actual transaction")
        check_row_admission(output, transaction, row)
        updates.append((slot, path, row, transaction_path))
    updates.sort()
    if [row[0] for row in updates] != list(range(evidence_end)):
        raise ValueError("v18b artifacts: selected checkpoint lacks its exact production slot history")
    main = [row for row in updates if row[2].get("phase") == "rl" and row[2].get("result", {}).get("optimizer_steps", 0) > 0]
    if not main:
        raise ValueError("v18b artifacts: no measured actual main optimizer update")
    if preparation_only and any(row[2].get("phase") != "prep" for row in updates[:step]):
        raise ValueError("v18b artifacts: a preparation-only checkpoint contains a main update")
    preflights = []
    for pid in sorted({row[2]["producer_pid"] for row in updates}):
        path = output / "training-bindings" / f"live-{pid}.json"
        value = read_sealed(_safe(path))
        first_slot = min(row[0] for row in updates if row[2]["producer_pid"] == pid)
        active_hotfixes = [row for row in hotfixes if row["slot"] <= first_slot]
        execution = ({"execution_code_commit": active_hotfixes[-1]["new_commit"],
                      "hotfix_lineage_sha256": digest(active_hotfixes)} if active_hotfixes else {})
        if (any(value.get(key) != item for key, item in identity.items())
                or any(value.get(key) != item for key, item in execution.items())
                or value.get("version") != "v18b-live-training-preflight-v1" or value.get("status") != "BOUND"
                or value.get("production_admitted") is not True or value.get("producer_pid") != pid
                or value.get("maximum_shape_measurements") != 1 or value.get("optimizer_steps") != 0
                or value.get("production_parents_consumed") != 0
                or value.get("actual_abc_status") != "NOT_RUN_USE_PRODUCTION_SLOT_TIMINGS"
                or value.get("f0_transport", {}).get("pass") is not True or not value.get("coexistence")
                or any(row.get("pass") is not True for row in value["coexistence"])):
            raise ValueError("v18b artifacts: actual producer admission differs")
        _validate_live_capacity(value["capacity"], plan)
        preflights.append(_ref(path))
    first = main[0]
    evidence: dict = {
        "live_training_preflights": preflights, "actual_update_results": [_ref(row[1]) for row in updates],
        "first_main_update": _ref(first[1]), "first_main_transaction": _ref(first[3]),
        "actual_update_timing_scope": "encode_backward_optimizer"}
    if hotfixes:
        state = read_sealed(output / "campaign-state.json")
        evidence["hotfix_resumes"] = state["hotfix_resumes"]
        identity.update(execution_code_commit=hotfixes[-1]["new_commit"], hotfix_lineage_sha256=digest(hotfixes))
    if preparation_only:
        stopped_timing.update(first_main_update=evidence.pop("first_main_update"),
                              first_main_transaction=evidence.pop("first_main_transaction"),
                              actual_update_results=[_ref(row[1]) for row in updates[step:]])
        evidence.update(actual_update_results=[_ref(row[1]) for row in updates[:step]], stopped_run_timing=stopped_timing)
    return identity | {"production_admitted": True}, evidence


def create_training_artifacts(args, plan: dict) -> dict:
    from . import campaign_v16b as campaign
    from .v17_preflight import _write, bind_training
    from fine_tuning.notebook_rl.checkpoint import read_metadata

    protocol_version = plan["protocol"].get("version")
    if protocol_version not in TRAINING_VERSIONS:
        raise ValueError("v17 artifacts: unsupported training protocol")
    output = _safe(Path(args.output))
    ready = campaign.require_complete(output, evaluation_decision=getattr(args, "evaluation_decision", None))
    checkpoint = _safe(Path(ready["checkpoint"]))
    metadata = read_metadata(checkpoint, binding=campaign._binding(plan))
    partial = protocol_version == "error-note-v18b" and ready.get("status") == "PARTIAL_EVALUATION_CANDIDATE"
    step = metadata["step"]
    live_evidence: dict = {}
    if protocol_version == "error-note-v18b":
        binding, live_evidence = _v18b_training_evidence(output, plan, step, ready.get("evaluation_selection"))
    else:
        binding = bind_training(args, plan)
    if not binding.get("production_admitted"):
        raise ValueError("v17 artifacts: local smoke cannot seal production artifacts")
    if (metadata["protocol"] != plan["protocol"] or (not partial and step != plan["total_slots"])
            or partial and not 0 < step <= plan["total_slots"]):
        raise ValueError("v17 artifacts: final checkpoint identity changed")
    reload_path = output / f"reload-{step}.json"
    reload = read_sealed(reload_path)
    if (reload.get("plan_sha256") != plan["sha256"] or reload.get("checkpoint") != ready["metadata_descriptor"]
            or reload.get("runtime_equal") is not True or reload.get("greedy_equal") is not True
            or reload.get("probe_max_abs_difference") != 0
            or not isinstance(reload.get("producer_pid"), int) or not isinstance(reload.get("reload_pid"), int)
            or reload["producer_pid"] == reload["reload_pid"]):
        raise ValueError("v17 artifacts: exact fresh-process final reload required")
    admission_path = output / "evaluation-admission.json"
    admission = read_sealed(admission_path)
    if protocol_version == "error-note-v17":
        decision = read_sealed(args.evaluation_decision)
        if (decision.get("decision") != "PROCEED" or decision.get("authorized_by") != "user"
                or decision.get("plan_sha256") != plan["sha256"] or decision.get("admission_sha256") != admission["sha256"]):
            raise ValueError("v17 artifacts: evaluation decision differs")
    export = _tree(Path(args.export_output))
    model_ref = _safe(Path(args.export_output) / "model_ref.json")
    model = json.loads(model_ref.read_text(encoding="utf-8"))
    if (model.get("code_commit") != binding.get("execution_code_commit", binding["code_commit"]) or model.get("kind") != "unmerged-lora"
            or model.get("dtype") != "bfloat16" or model.get("export", {}).get("binding") != campaign._binding(plan)
            or model.get("export", {}).get("campaign", {}).get("plan_sha256") != plan["sha256"]):
        raise ValueError("v17 artifacts: export is not the bound final unmerged model")
    if protocol_version in ("error-note-v18", "error-note-v18b") and model["export"].get("training_protocol") != plan["protocol"]:
        raise ValueError("v18 artifacts: exported training protocol differs from sealed plan")
    for record in model["files"]:
        path = _safe(Path(args.export_output) / "model" / record["path"])
        if (not path.is_relative_to(_safe(Path(args.export_output) / "model"))
                or path.stat().st_size != record["bytes"] or file_sha(path) != record["sha256"]):
            raise ValueError("v17 artifacts: model-ref file inventory differs")
    code_root = ROOT / "fine_tuning"
    code_identity = plan["source"]
    execution: dict = {}
    if binding.get("execution_code_commit"):
        from types import SimpleNamespace
        from . import single
        from .v18b_hotfix import execution_source, lineage
        execution = execution_source(SimpleNamespace(output=output, repo=ROOT), plan, single.source_manifest())
        code_identity = lineage(output, plan)[-1]["new_source"]
    code_files = [{"path": name, "file_sha256": _ref(code_root / name)["file_sha256"]}
                  for name in sorted(code_identity["actual_files"])]
    if {r["path"]: r["file_sha256"] for r in code_files} != code_identity["actual_files"]:
        raise ValueError("v17 artifacts: training source files changed")
    body: dict = {"version": VERSION, "status": "SEALED", "training": {k: binding[k] for k in
            ("code_commit", "source_sha256", "campaign_plan_sha256", "protocol_sha256")},
        **execution,
        "training_code": {"root": str(code_root), "files": code_files, "files_sha256": digest(code_files)},
        "campaign_plan": _ref(output / "campaign-plan.json"),
        "checkpoint": _tree(checkpoint) | {"metadata": _ref(checkpoint / "metadata.json")},
        "export": export | {"model_ref": _ref(model_ref)}, "reload": _ref(reload_path),
        "evaluation_admission": _ref(admission_path),
        "export_ready": _ref(output / "export-ready.json"), "protected_evaluation": protected_evaluation(ROOT)}
    if protocol_version == "error-note-v18b":
        body.update(live_evidence)
    else:
        body.update(functional_receipt=binding["functional_receipt"], update_cost_receipt=binding["update_cost_receipt"])
    if partial:
        body["status"] = "PARTIAL_EVALUATION_CANDIDATE"
        body["evaluation_selection"] = model["export"]["campaign"].get("evaluation_selection")
        _partial_selection(body, model)
    if protocol_version in ("error-note-v18", "error-note-v18b"):
        body.update(training_protocol_version=protocol_version, monitor=_ref(output / "monitor-final.json"),
                    effect=_ref(output / "effect-final.json"))
        _v18_admission(body)
    else:
        body["evaluation_decision"] = _ref(args.evaluation_decision)
    destination = Path(getattr(args, "artifact_output", None) or output / "training-artifacts.json")
    if any(_safe(destination).is_relative_to(Path(t["root"])) for t in (body["checkpoint"], body["export"])):
        raise ValueError("v17 artifacts: receipt cannot be inside its bound artifact tree")
    return _write(destination, body)


def verify_training_artifacts(path: Path) -> dict:
    value = read_sealed(_safe(path))
    with _artifact_paths(value):
        return _verify_training_artifacts(path)


def _verify_training_artifacts(path: Path) -> dict:
    value = read_sealed(_safe(path))
    if value.get("version") != VERSION or value.get("status") not in ("SEALED", "PARTIAL_EVALUATION_CANDIDATE"):
        raise ValueError("v17 artifacts: sealed production training artifacts required")
    for name in ("campaign_plan", "reload", "evaluation_admission", "export_ready"):
        _check_ref(value[name])
    if value.get("training_protocol_version") != "error-note-v18b":
        for name in ("functional_receipt", "update_cost_receipt"):
            _check_ref(value[name])
    if value.get("training_protocol_version") not in ("error-note-v18", "error-note-v18b"):
        _check_ref(value["evaluation_decision"])
    elif "evaluation_decision" in value:
        raise ValueError("v18 artifacts: training seal cannot authorize evaluation")
    for name in ("checkpoint", "export"):
        tree = value[name]
        if _tree(Path(tree["root"])) != {k: tree[k] for k in ("root", "files", "files_sha256")}:
            raise ValueError("v17 artifacts: checkpoint/export inventory changed")
    _check_ref(value["checkpoint"]["metadata"])
    _check_ref(value["export"]["model_ref"])
    for name, key, filename in (("checkpoint", "metadata", "metadata.json"),
                                 ("export", "model_ref", "model_ref.json")):
        if value[name][key] != _ref(Path(value[name]["root"]) / filename):
            raise ValueError("v17 artifacts: identity file escapes its artifact tree")
    code = value["training_code"]
    if digest(code["files"]) != code["files_sha256"]:
        raise ValueError("v17 artifacts: training code inventory changed")
    for record in code["files"]:
        source = _safe(Path(code["root"]) / record["path"])
        if not source.is_relative_to(_safe(Path(code["root"]))) or file_sha(source) != record["file_sha256"]:
            raise ValueError("v17 artifacts: original training source changed")
    plan = read_sealed(Path(value["campaign_plan"]["path"]))
    protocol_version = plan["protocol"].get("version")
    if protocol_version == "error-note-v18b":
        metadata = json.loads(Path(value["checkpoint"]["metadata"]["path"]).read_text())
        binding, evidence = _v18b_training_evidence(Path(value["campaign_plan"]["path"]).parent, plan, metadata["step"],
                                                  value.get("evaluation_selection"))
        if "stopped_run_timing" in evidence and any(key in value for key in ("first_main_update", "first_main_transaction")):
            raise ValueError("v18b artifacts: stopped-run timing cannot be attributed to the preparation-only checkpoint")
        if any(value.get(key) != item for key, item in evidence.items()) or any(value["training"][key] != binding[key] for key in value["training"]):
            raise ValueError("v18b artifacts: actual producer or first-update evidence changed")
    model = json.loads(Path(value["export"]["model_ref"]["path"]).read_text(encoding="utf-8"))
    code_identity = plan["source"]
    if value.get("hotfix_resumes"):
        from types import SimpleNamespace
        from .v18b_hotfix import execution_source, lineage
        output = Path(value["campaign_plan"]["path"]).parent
        # Bind the lineage to the sealed training checkout, not to the evaluator's own source: evaluation may run a
        # newer commit (v18b pod 2026-11-20 stopped here: eval 89d64fd vs. last authorized training commit 6dbc649).
        training_root = Path(code["root"])
        training_source = {
            "base_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=training_root, check=True,
                                          capture_output=True, text=True).stdout.strip(),
            "actual_files": {p.relative_to(training_root).as_posix(): file_sha(_safe(p))
                             for p in sorted(training_root.rglob("*.py"))
                             if not {"tests", "runs", "generated", "__pycache__"}.intersection(p.relative_to(training_root).parts)}}
        execution = execution_source(SimpleNamespace(output=output, repo=training_root.parent), plan, training_source)
        if any(value.get(key) != item for key, item in execution.items()) or model.get("code_commit") != execution["execution_code_commit"]:
            raise ValueError("v18b artifacts: final execution source lineage changed")
        code_identity = lineage(output, plan)[-1]["new_source"]
    if (protocol_version not in TRAINING_VERSIONS
            or (protocol_version in ("error-note-v18", "error-note-v18b") and (
                value.get("training_protocol_version") != protocol_version
                or model.get("export", {}).get("training_protocol") != plan["protocol"]))
            or (protocol_version == "error-note-v17" and value.get("training_protocol_version") not in (None, protocol_version))):
        raise ValueError("v17 artifacts: training protocol version was relabeled")
    if (plan["sha256"] != value["training"]["campaign_plan_sha256"]
            or plan["source"]["base_commit"] != value["training"]["code_commit"]
            or digest(plan["source"]["actual_files"]) != value["training"]["source_sha256"]
            or plan["protocol"]["sha256"] != value["training"]["protocol_sha256"]
            or {r["path"]: r["file_sha256"] for r in code["files"]} != code_identity["actual_files"]):
        raise ValueError("v17 artifacts: training identity was relabeled")
    observed = {p.relative_to(Path(code["root"])).as_posix(): file_sha(_safe(p))
                for p in Path(code["root"]).rglob("*.py")
                if not {"tests", "runs", "generated", "__pycache__"}.intersection(p.relative_to(Path(code["root"])).parts)}
    actual_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code["root"], check=True,
                                   capture_output=True, text=True).stdout.strip()
    if observed != code_identity["actual_files"] or actual_commit != code_identity["base_commit"]:
        raise ValueError("v17 artifacts: original training checkout changed")
    if value.get("status") == "PARTIAL_EVALUATION_CANDIDATE":
        if protocol_version != "error-note-v18b":
            raise ValueError("v18b artifacts: only explicit v18b selection can be partial")
        _partial_selection(value, model)
    elif model.get("export", {}).get("campaign", {}).get("stage") == "partial":
        raise ValueError("v18b artifacts: partial checkpoint mislabeled as complete")
    if protocol_version in ("error-note-v18", "error-note-v18b"):
        _v18_admission(value)
    return value


def _partial_selection(artifacts: dict, model: dict) -> dict:
    selection = artifacts.get("evaluation_selection")
    campaign = model.get("export", {}).get("campaign", {})
    if (not isinstance(selection, dict) or campaign.get("stage") != "partial"
            or campaign.get("evaluation_selection") != selection
            or selection.get("reason") not in ("TRAINING_CUTOFF", "MONITOR_STOP")
            or selection.get("authorized_action") != "evaluate_checkpoint"
            or selection.get("source_checkpoint_sha256") != artifacts["checkpoint"]["metadata"]["file_sha256"]
            or type(selection.get("consumed_items")) is not int
            or not 0 <= selection["consumed_items"] <= 3000 or selection.get("total_items") != 3000):
        raise ValueError("v18b artifacts: explicit partial checkpoint selection required")
    contract = model.get("export", {}).get("training_protocol", {}).get("training_contract", {})
    slot, prep, blocks, total = (campaign.get("slot"), contract.get("prep_updates"),
                                contract.get("rl_blocks"), contract.get("total_slots"))
    if selection["consumed_items"] == 3000 or type(slot) is int and type(total) is int and slot == total:
        if (selection["consumed_items"] != 3000 or selection["reason"] != "TRAINING_CUTOFF"
                or any(type(value) is not int for value in (slot, prep, blocks, total))
                or not prep + blocks <= slot <= total):
            raise ValueError("v18b artifacts: all parents at cutoff require a saved main or settlement boundary")
    preparation_only = selection.get("preparation_only", False)
    if type(preparation_only) is not bool or (selection["consumed_items"] == 0) != preparation_only:
        raise ValueError("v18b artifacts: zero main-training items require explicit preparation-only selection")
    if preparation_only:
        contract = model.get("export", {}).get("training_protocol", {}).get("training_contract", {})
        if (selection["reason"] != "MONITOR_STOP" or campaign.get("slot") != contract.get("prep_updates")
                or type(campaign.get("slot")) is not int or campaign["slot"] <= 0):
            raise ValueError("v18b artifacts: only a monitor stop may select the preparation checkpoint with zero main-training items")
    return selection


def _v18_admission(artifacts: dict) -> dict:
    with _artifact_paths(artifacts):
        return _bound_admission(artifacts)


def _bound_admission(artifacts: dict) -> dict:
    """A training seal preserves the final admission, including failures, without approving an evaluation."""
    from .v17_eval import evaluation_gate

    plan = read_sealed(_check_ref(artifacts["campaign_plan"]))
    admission = read_sealed(_check_ref(artifacts["evaluation_admission"]))
    monitor = read_sealed(_check_ref(artifacts["monitor"]))
    effect = read_sealed(_check_ref(artifacts["effect"]))
    ready = read_sealed(_check_ref(artifacts["export_ready"]))
    checkpoint = artifacts["checkpoint"]["metadata"]
    if (plan["protocol"]["version"] not in ("error-note-v18", "error-note-v18b")
            or artifacts["training"]["campaign_plan_sha256"] != plan["sha256"]
            or any(row.get("plan_sha256") != plan["sha256"] or not _same_ref(row.get("checkpoint"), checkpoint)
                   for row in (admission, monitor, effect))
            or ready.get("plan_sha256") != plan["sha256"] or not _same_ref(ready.get("metadata_descriptor"), checkpoint)
            or admission.get("monitor_sha256") != monitor["sha256"] or admission.get("effect_sha256") != effect["sha256"]
            or admission.get("decision") != evaluation_gate(monitor["comparison"], effect["comparison"])):
        raise ValueError("v18 artifacts: final admission/monitor/effect/checkpoint binding differs")
    return admission


def _v18_evaluation_decision(artifacts: dict, artifact_ref: dict, identity: dict, authorization: str) -> dict:
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError("v18 evaluation: separate non-empty user authorization required")
    admission = _v18_admission(artifacts)
    status = admission["decision"]["status"]
    if status not in ("EVAL_REVIEW_REQUIRED", "EVALUATION_CANDIDATE"):
        raise ValueError("v18 evaluation: DEGRADED or incomplete admission cannot be authorized")
    body = {"version": "v18-evaluation-decision-v1", "decision": "PROCEED", "authorized_by": "user",
            "pre_authorized": False, "authorization": authorization,
            "training_artifact": artifact_ref, "plan_sha256": artifacts["training"]["campaign_plan_sha256"],
            "admission": artifacts["evaluation_admission"], "admission_sha256": admission["sha256"],
            "admission_status": status, "checkpoint": artifacts["checkpoint"]["metadata"],
            "model_ref": artifacts["export"]["model_ref"], "evaluation": identity}
    if artifacts.get("status") == "PARTIAL_EVALUATION_CANDIDATE":
        body["evaluation_selection"] = artifacts["evaluation_selection"]
    return body | {"sha256": digest(body)}


def verify_evaluation_authorization(path: Path, training_artifacts: Path, identity: dict) -> dict:
    """Recheck consent against live sealed evidence before engine validation and every final/resumed run."""
    verified = read_sealed(_safe(path))
    artifacts = verify_training_artifacts(training_artifacts)
    ref = _ref(training_artifacts)
    decision = verified.get("evaluation_decision", {})
    if (artifacts.get("training_protocol_version") not in ("error-note-v18", "error-note-v18b")
            or verified.get("version") != "v17-training-artifact-verification-v1" or verified.get("status") != "PASS"
            or verified.get("training_artifact") != ref or verified.get("training") != artifacts["training"]
            or verified.get("evaluation") != identity
            or verified.get("protected_evaluation_sha256") != artifacts["protected_evaluation"]["sha256"]
            or decision != _v18_evaluation_decision(artifacts, ref, identity, decision.get("authorization"))):
        raise ValueError("v18 evaluation: authorization belongs to another artifact, admission or evaluator")
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-protocol", choices=TRAINING_VERSIONS)
    parser.add_argument("--authorization")
    args = parser.parse_args(argv)
    value = verify_training_artifacts(args.receipt)
    version = read_sealed(Path(value["campaign_plan"]["path"]))["protocol"]["version"]
    if args.expected_protocol is not None and version != args.expected_protocol:
        raise ValueError("v17 artifacts: training seal does not match expected protocol")
    identity = evaluation_identity(args.repo, args.commit)
    if protected_evaluation(args.repo) != value["protected_evaluation"]:
        raise ValueError("v17 artifacts: protected evaluation/preregistration contract changed")
    from .v17_preflight import _write
    body = {"version": "v17-training-artifact-verification-v1", "status": "PASS",
        "training_artifact": _ref(args.receipt), "training": value["training"], "evaluation": identity,
        "protected_evaluation_sha256": value["protected_evaluation"]["sha256"]}
    if version in ("error-note-v18", "error-note-v18b"):
        if args.expected_protocol != version:
            raise ValueError("v18 evaluation: explicit --expected-protocol error-note-v18 required")
        body["evaluation_decision"] = _v18_evaluation_decision(value, _ref(args.receipt), identity, args.authorization)
    _write(args.output, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
