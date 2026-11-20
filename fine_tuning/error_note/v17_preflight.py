"""Actual-device v17 admission and backward-only cost measurements; never train here."""
from __future__ import annotations

from copy import copy
import math
from pathlib import Path
import re
from typing import Any

from .audit import digest, text_sha256
from .prep_v16b import file_sha, read_sealed, seal

VERSION = "v17-bf16-preflight-v1"


def _write(path: Path, body: dict) -> dict:
    result = body | {"sha256": digest(body)}
    if path.is_symlink():
        raise ValueError("v17 preflight: symlink receipt refused")
    if path.exists():
        if read_sealed(path) != result:
            raise ValueError("v17 preflight: immutable receipt differs")
        return result
    return seal(path, body)


def _identity(args: Any, plan: dict) -> dict:
    from . import campaign_v16b as campaign

    campaign._plan_check(plan)
    current = campaign.single.source_manifest()
    commit = getattr(args, "code_commit", None)
    if not re.fullmatch(r"[a-f0-9]{40}", str(commit or "")):
        raise ValueError("v17 preflight: exact --code-commit required")
    if (commit != current["base_commit"]
            or plan["protocol"]["version"] not in ("error-note-v17", "error-note-v18", "error-note-v18b")):
        raise ValueError("v17 preflight: code commit, source bytes or protocol changed")
    source_changed = (commit != plan["source"]["base_commit"]
                      or current["actual_files"] != plan["source"]["actual_files"])
    execution = {}
    if source_changed or getattr(args, "hotfix_resume", None) is not None:
        if plan["protocol"]["version"] != "error-note-v18b":
            raise ValueError("v17 preflight: code commit, source bytes or protocol changed")
        from .v18b_hotfix import execution_source
        execution = execution_source(args, plan, current)
    return {"code_commit": plan["source"]["base_commit"], "source_sha256": digest(plan["source"]["actual_files"]),
            "campaign_plan_sha256": plan["sha256"], "protocol_sha256": plan["protocol"]["sha256"],
            **execution,
            **({"training_protocol_version": plan["protocol"]["version"]}
               if plan["protocol"]["version"] in ("error-note-v18", "error-note-v18b") else {})}


def _file_receipt(path: Path | None, expected: str | None) -> tuple[dict, dict]:
    if path is None or not re.fullmatch(r"[a-f0-9]{64}", str(expected or "")):
        raise ValueError("v17 preflight: explicit receipt path and SHA256 required")
    path = Path(path)
    if path.is_symlink() or file_sha(path) != expected:
        raise ValueError("v17 preflight: bound receipt bytes changed")
    return read_sealed(path), {"path": str(path.absolute()), "file_sha256": expected}


def _validate_cost(cost: dict) -> None:
    measurement = cost.get("measurement")
    coverage = cost.get("coverage") or {}
    if cost.get("status") not in {"PASS", "CALIBRATION_LOW_COVERAGE"}:
        raise ValueError("v17 preflight: invalid cost calibration status")
    if measurement is None:
        if (cost["status"] != "CALIBRATION_LOW_COVERAGE" or coverage.get("actual_a1_count") != 0
                or cost.get("actual_abc_status") != "UNMEASURED_NO_ADMITTED_A1"):
            raise ValueError("v17 preflight: unmeasured A/B/C cannot be called measured")
        return
    if (measurement.get("version") != "v17-update-calibration-v1"
            or measurement.get("measurement") != "cuda-bf16" or measurement.get("optimizer_steps") != 0
            or not coverage.get("actual_a1_count")):
        raise ValueError("v17 preflight: actual admitted A1 and CUDA cost evidence required")
    if cost.get("training_protocol_version") == "error-note-v18" and (
            measurement.get("objective_version") != "error-note-v18"
            or measurement.get("row_schema") != "error-note-v18-row-v1"):
        raise ValueError("v18 preflight: measured objective and row schema must be v18")
    for phase in ("A", "B", "C"):
        measured = measurement.get("measurements", {}).get(phase, {})
        samples = measured.get("samples", [])
        if (len(samples) != 3 or any(type(s.get("seconds")) not in (int, float)
                or not math.isfinite(s["seconds"]) or s["seconds"] <= 0 for s in samples)
                or measured.get("median_seconds") != sorted(s["seconds"] for s in samples)[1]):
            raise ValueError("v17 preflight: actual warmup plus three A/B/C measurements required")
    if cost["status"] == "PASS" and (coverage.get("missing_actual_a1_lengths") or not coverage.get("actual_full_rows")):
        raise ValueError("v17 preflight: missing timing strata must remain low coverage")


def _validate_functional(functional: dict) -> None:
    capacity = functional.get("capacity", {})
    report = capacity.get("report", {})
    # campaign_v16b.capacity seals "v18-capacity-v1" under the v18 protocol (v18 pod 2026-11-18 failed here).
    expected = "v18-capacity-v1" if functional.get("training_protocol_version") == "error-note-v18" else "v17-capacity-v1"
    if (functional.get("optimizer_steps") != 0 or functional.get("production_parents_consumed") != 0
            or capacity.get("version") != expected or capacity.get("model_parameter_update") is not False
            or report.get("version") != "v17-update-memory-v1" or report.get("optimizer_steps") != 0
            or report.get("fixture_only") is not True or report.get("action_tokens") != 8192
            or report.get("measurement") != "actual_cuda_bf16_shared_backward_only"
            or functional.get("f0_transport", {}).get("pass") is not True
            or not functional.get("coexistence")
            or any(r.get("pass") is not True for r in functional["coexistence"])):
        raise ValueError("v17 preflight: actual maximum FULL capacity and F0 transport evidence required")


def bind_execution(args: Any, plan: dict) -> dict:
    """Bind the measured evaluation and training receipts before loading a production model."""
    if plan.get("local_sequential"):
        return _write(Path(args.output) / "execution-binding.json", {
            "version": "v17-execution-binding-v1", "status": "LOCAL_SMOKE_EXEMPT",
            "campaign_plan_sha256": plan["sha256"], "production_admitted": False,
            "reason": "Local NF4 smoke does not certify the BF16 engine or measured evaluation timeout."})
    from .v17_eval import load_calibration

    identity = _identity(args, plan)
    calibration, calibrated_file = _file_receipt(getattr(args, "calibration", None), getattr(args, "calibration_sha256", None))
    if load_calibration(Path(args.calibration)) != calibration:
        raise ValueError("v17 preflight: evaluation calibration changed")
    functional, functional_file = _file_receipt(getattr(args, "functional_receipt", None), getattr(args, "functional_receipt_sha256", None))
    if (functional.get("version") != VERSION or functional.get("status") != "PASS"
            or functional.get("optimizer_steps") != 0 or functional.get("local_smoke") is not False
            or any(functional.get(key) != value for key, value in identity.items())
            or any(calibration.get(key) != identity[key] for key in ("code_commit", "source_sha256"))):
        raise ValueError("v17 preflight: incomplete or stale BF16 functional/calibration receipt")
    _validate_functional(functional)
    cost_ref = functional.get("update_cost_receipt", {})
    cost, _ = _file_receipt(Path(cost_ref["path"]) if cost_ref.get("path") else None, cost_ref.get("file_sha256"))
    if (cost.get("sha256") != functional.get("update_cost_sha256") or cost.get("status") not in {"PASS", "CALIBRATION_LOW_COVERAGE"}
            or any(cost.get(key) != value for key, value in identity.items()) or cost.get("optimizer_steps") != 0):
        raise ValueError("v17 preflight: cost receipt differs from the validated functional run")
    _validate_cost(cost)
    return _write(Path(args.output) / "execution-binding.json", {
        "version": "v17-execution-binding-v1", "status": "BOUND", "production_admitted": True,
        **identity, "calibration": calibrated_file, "functional_receipt": functional_file,
        "update_cost_receipt": cost_ref, "update_cost_status": cost["status"],
        "cost_operator_review_required": cost["status"] != "PASS",
        "unmeasured_timing_strata": cost.get("coverage", {}).get("missing_actual_a1_lengths", []),
        "evaluation_watchdog_seconds": calibration["evaluation_watchdog_seconds"]})


def bind_training(args: Any, plan: dict) -> dict:
    """Bind training evidence without making training depend on an evaluation engine."""
    if plan.get("local_sequential"):
        return _write(Path(args.output) / "training-binding.json", {
            "version": "v17-training-binding-v1", "status": "LOCAL_SMOKE_EXEMPT",
            "campaign_plan_sha256": plan["sha256"], "production_admitted": False})
    identity = _identity(args, plan)
    if plan["protocol"]["version"] == "error-note-v18b":
        return {"version": "v18b-training-binding-v1", "status": "LIVE_CHECK_REQUIRED",
                "production_admitted": False, **identity,
                "reason": "Maximum-shape backward and current-engine checks run inside campaign-advance."}
    functional, functional_file = _file_receipt(
        getattr(args, "functional_receipt", None), getattr(args, "functional_receipt_sha256", None))
    if (functional.get("version") != VERSION or functional.get("status") != "PASS"
            or functional.get("local_smoke") is not False
            or any(functional.get(key) != value for key, value in identity.items())):
        raise ValueError("v17 training: incomplete or stale BF16 functional receipt")
    _validate_functional(functional)
    cost_ref = functional.get("update_cost_receipt", {})
    cost, cost_file = _file_receipt(Path(cost_ref["path"]) if cost_ref.get("path") else None,
                                    cost_ref.get("file_sha256"))
    if (cost.get("sha256") != functional.get("update_cost_sha256") or cost.get("optimizer_steps") != 0
            or any(cost.get(key) != value for key, value in identity.items())):
        raise ValueError("v17 training: cost receipt differs from validated functional run")
    _validate_cost(cost)
    return _write(Path(args.output) / "training-binding.json", {
        "version": "v17-training-binding-v1", "status": "BOUND", "production_admitted": True,
        **identity, "functional_receipt": functional_file, "update_cost_receipt": cost_file,
        "update_cost_status": cost["status"], "cost_operator_review_required": cost["status"] != "PASS",
        "unmeasured_timing_strata": cost.get("coverage", {}).get("missing_actual_a1_lengths", [])})


def live_training_preflight(args: Any, plan: dict, bundle: Any, sampler: Any, *, capacity: Any) -> dict:
    """Admit this live campaign process after one maximum-shape backward, without A/B/C timing jobs."""
    import os

    if plan.get("local_sequential"):
        return {"version": "v18b-live-training-preflight-v1", "status": "LOCAL_SMOKE_EXEMPT",
                "production_admitted": False, "campaign_plan_sha256": plan["sha256"]}
    identity = _identity(args, plan)
    if (plan["protocol"]["version"] != "error-note-v18b" or args.precision != "bf16"
            or args.engine != "vllm" or getattr(args, "allow_model_run", False) is not True
            or plan["protocol"]["precision"] != "bf16" or not callable(capacity)):
        raise ValueError("v18b live preflight: authorized BF16 vLLM and capacity callable required")
    if sampler.has_unfinished_requests():
        raise ValueError("v18b live preflight: engine must be drained before maximum-shape backward")
    path = Path(args.output) / "training-bindings" / f"live-{os.getpid()}.json"
    if path.exists():
        raise FileExistsError("v18b live preflight: do not repeat a measured process admission")
    sampler.add_version(bundle, 0)
    sampler.add_version(bundle, 1)
    transport, coexistence = sampler.preflight_report, sampler.coexistence_reports
    if (not isinstance(transport, dict) or transport.get("pass") is not True or not coexistence
            or any(r.get("pass") is not True for r in coexistence)):
        raise ValueError("v18b live preflight: actual policy transport/coexistence failed")
    measured = capacity()
    _validate_live_capacity(measured, plan)
    return _write(path, {"version": "v18b-live-training-preflight-v1", "status": "BOUND",
        **identity, "production_admitted": True, "producer_pid": os.getpid(), "capacity": measured,
        "f0_transport": transport, "coexistence": coexistence, "optimizer_steps": 0,
        "production_parents_consumed": 0, "maximum_shape_measurements": 1,
        "actual_abc_status": "NOT_RUN_USE_PRODUCTION_SLOT_TIMINGS",
        "solution_acceptance": "UNCHANGED_ACTUAL_ROW_VALIDATION_REQUIRED"})


def _validate_live_capacity(capacity: dict, plan: dict) -> None:
    report = capacity.get("report", {})
    bounds = capacity.get("shape_bounds", {})
    after = report.get("post_backward_memory", {})
    if (capacity.get("version") != "v18b-capacity-v1" or capacity.get("model_parameter_update") is not False
            or report.get("version") != "v17-update-memory-v1" or report.get("optimizer_steps") != 0
            or report.get("measurement") != "actual_cuda_bf16_shared_backward_only"
            or report.get("live_engine") is not True or report.get("fixture_only") is not True
            or report.get("action_tokens") != 8192 or report.get("prompt_tokens") != 16384
            or report.get("shape_bounds") != bounds or report.get("shape_bounds_sha256") != digest(bounds)
            or bounds.get("dataset_sha256") != plan.get("inputs", {}).get("payload", {}).get("file_sha256")
            or type(after.get("free")) is not int or after["free"] < 2 * 2**30):
        raise ValueError("v18b live preflight: current maximum FULL memory evidence required")


def bind_evaluation(args: Any, plan: dict, *, require_engine_validation: bool = True) -> dict:
    """Keep the original training identity while binding a separately frozen evaluator."""
    from .v17_artifacts import (ROOT, _ref, evaluation_identity, protected_evaluation,
                                verify_training_artifacts)

    seal_path = getattr(args, "training_artifact_seal", None)
    if seal_path is None:
        raise ValueError("v17 evaluation: --training-artifact-seal required")
    artifacts = verify_training_artifacts(Path(seal_path))
    if artifacts["training"]["campaign_plan_sha256"] != plan["sha256"]:
        raise ValueError("v17 evaluation: training seal belongs to another plan")
    version = plan["protocol"]["version"]
    if version in {"error-note-v18", "error-note-v18b"} and artifacts.get("training_protocol_version") != version:
        raise ValueError("v18 evaluation: matching training-artifact seal required")
    identity = evaluation_identity(Path(getattr(args, "repo", None) or ROOT), args.code_commit)
    if version in {"error-note-v18", "error-note-v18b"}:
        from .v17_artifacts import verify_evaluation_authorization
        authorization_path = getattr(args, "artifact_verification", None)
        if authorization_path is None:
            raise ValueError("v18 evaluation: --artifact-verification required")
        verify_evaluation_authorization(Path(authorization_path), Path(seal_path), identity)
    protected = protected_evaluation(ROOT)
    if protected != artifacts["protected_evaluation"]:
        raise ValueError("v17 evaluation: protected evaluation/preregistration contract changed")
    validation_ref = None
    if require_engine_validation:
        validation, validation_ref = _file_receipt(
            getattr(args, "engine_validation_receipt", None),
            getattr(args, "engine_validation_receipt_sha256", None))
        if (validation.get("version") != "v17-eval-engine-validation-v1" or validation.get("status") != "PASS"
                or validation.get("training_artifact_sha256") != file_sha(Path(seal_path))
                or validation.get("model_ref_sha256") != artifacts["export"]["model_ref"]["file_sha256"]
                or validation.get("evaluation_commit") != identity["code_commit"]
                or validation.get("source_sha256") != identity["source_sha256"]):
            raise ValueError("v17 evaluation: engine validation is for another model or evaluator")
        if version in {"error-note-v18", "error-note-v18b"}:
            from .v17_eval_engine import validate_engine_receipt
            validate_engine_receipt(validation)
            if validation.get("training_protocol_version") != version or validation["slots"] != 9:
                raise ValueError("v18 evaluation: v18 engine evidence and nine slots required")
        from mmdl.runtime.environment import collect_environment
        from mmdl.evaluation import engine as evaluation
        if validation.get("environment_sha256") != digest(evaluation.environment_identity(collect_environment(probe=False))):
            raise ValueError("v17 evaluation: engine validation environment changed")
    body = {"version": "v17-evaluation-binding-v1", "status": "BOUND" if require_engine_validation else "PREVALIDATION",
        "training": artifacts["training"], "evaluation": identity, "training_artifact": _ref(Path(seal_path)),
        "model_ref": artifacts["export"]["model_ref"], "engine_validation_receipt": validation_ref,
        "protected_evaluation_sha256": protected["sha256"]}
    name = "evaluation-binding.json" if require_engine_validation else "evaluation-prevalidation-binding.json"
    return _write(Path(args.output) / name, body)


def _rl_row(item: dict, output: dict, baseline: dict, *, policy: dict | None = None) -> dict:
    from .v17_rows import ROW_SCHEMA, reward_v17

    reward, reason, pg = reward_v17(output["status"], output["finish_reason"], output["text"], output.get("extracted"))
    if policy is not None:
        reward, reason, pg = (policy[key] for key in ("reward", "reason", "pg"))
    receipt: dict[str, Any] = {"reward": reward, "reason": reason, "pg": pg}
    if policy is not None:
        receipt["policy_receipt"] = policy
    receipt["sha256"] = digest(receipt)
    row = {"schema": ROW_SCHEMA, "row_id": item["problem_id"] + ":RL", "kind": "RL",
        "problem_id": item["problem_id"], "parent_id": item["problem_id"], "source": item["source"],
        "subject": item.get("subject", item["source"]), "layer": item.get("layer"), "block_index": 0,
        "image_paths": item["image_paths"], "image_sha256": item["image_sha256"],
        "route": "FULL" if item["image_paths"] else "LM", "prompt": output["prompt"], "target": output["text"],
        "generation": output["generation"], "reward": reward, "reason": reason, "pg": pg,
        "baseline": baseline["baseline"], "baseline_receipt": baseline,
        "advantage": reward - baseline["baseline"] if pg and reward is not None else 0.,
        "generation_prompt_sha256": text_sha256(output["prompt"]), "training_prompt_sha256": text_sha256(output["prompt"]),
        "admission_receipt_sha256": receipt["sha256"], "verifier_receipt_sha256": receipt["sha256"],
        "reward_receipt": receipt, "loss_probability_space": "processed-full-vocab-v1",
        "knowledge_status": "ANSWER_REWARD", "complete": True}
    if policy is not None:
        row.update(schema="error-note-v18-row-v1", t1_policy_receipt=policy,
                   format_receipt=policy.get("format_receipt"))
        row.update({key: policy[key] for key in ("loss_token_mask", "pg_mask_reason", "pg_byte_spans", "policy_mask_receipt")})
    return row


def _coverage(encoded: list[dict]) -> dict:
    groups: dict[str, list[str]] = {"short": [], "medium": [], "long": []}
    for row in encoded:
        if row["kind"] != "A1":
            continue
        n = row["target_len"]
        groups["short" if n <= 256 else "medium" if n <= 2048 else "long"].append(row["row_id"])
    return {"actual_a1": groups, "actual_a1_count": sum(map(len, groups.values())),
            "missing_actual_a1_lengths": [name for name, members in groups.items() if not members],
            "actual_full_rows": sum(row["kind"] == "RL" and row["route"] == "FULL" for row in encoded)}


def calibrate(args: Any, plan: dict) -> dict:
    """Run actual F0 transport/capacity and A/B/C backward probes on disjoint performance parents."""
    from . import campaign_v16b as campaign
    from . import prep_v16b as prep
    from . import rl, train_v17, v17_rows
    from .protocol import TRAIN_SAMPLING_V16B, v17_contract_for_update
    from .run_v16 import _setup_execution
    from .v16b_rows import baseline_receipt_v16b

    output = getattr(args, "receipt_output", None)
    if output is None:
        raise ValueError("v17 preflight: --receipt-output required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("v17 preflight: refusing to repeat a measured receipt")
    if plan.get("local_sequential"):
        return _write(output, {"version": VERSION, "status": "SKIPPED_LOCAL_SMOKE", "local_smoke": True,
            "campaign_plan_sha256": plan["sha256"], "optimizer_steps": 0,
            "reason": "A local NF4 smoke cannot certify actual BF16 vLLM or maximum FULL capacity."})
    identity = _identity(args, plan)
    version = plan["protocol"]["version"]
    if version == "error-note-v18b":
        raise ValueError("v18b preflight: use the single in-process campaign-advance memory check; A/B/C jobs are removed")
    is_v18 = version == "error-note-v18"
    row_api: Any = v17_rows
    if is_v18:
        from . import v18_rows
        row_api = v18_rows
        from .protocol import v18_contract_for_update as v17_contract_for_update
        from .train_v18 import calibrate_update_v18
    if (getattr(args, "allow_model_run", False) is not True or args.precision != "bf16"
            or args.engine != "vllm" or plan["protocol"]["precision"] != "bf16"):
        raise ValueError("v17 preflight: explicit model-run authorization and BF16 vLLM required")
    data = campaign.data_api(version)
    validation = data.validate(args.data_root)
    items = data.load_items(args.data_root, "performance")
    if not 1 <= len(items) <= 16:
        raise ValueError("v17 preflight: one disjoint performance block of at most sixteen parents required")
    prior = read_sealed(campaign.checked(plan["inputs"]["baseline"]))
    # The current outputs cannot influence these admission baselines.
    baselines = {item["problem_id"]: baseline_receipt_v16b(item, [], prior=prior, history_cutoff=-2) for item in items}
    _setup_execution(3407)
    run_args = copy(args)
    run_args._preflight_stage = "f0"
    image = plan["protocol"]["image"]
    bundle, _ = campaign.load_f0(run_args, image=image)
    capacity = campaign.calibrate_capacity(bundle, run_args, plan["protocol"])
    sampler = campaign.make_sampler(bundle, run_args, plan["protocol"], image=image, sampling=TRAIN_SAMPLING_V16B)
    try:
        sampler.add_version(bundle, 0)
        sampler.add_version(bundle, 1)
        collected = prep.collect(bundle, sampler, items, data_root=args.data_root, split="performance",
            out=Path(args.output) / "preflight-generations", stage=version.removeprefix("error-note-") + "-bf16-preflight-t1",
            protocol_sha256=plan["protocol"]["sha256"], sampling=TRAIN_SAMPLING_V16B, cap=8192,
            algorithm_version=version, baseline_receipts=baselines)
        if len(collected) != len(items) or sampler.has_unfinished_requests():
            raise ValueError("v17 preflight: performance collection did not drain completely")
        lookup = {item["problem_id"]: item for item in items}
        rows, admissions = [], []
        for generated in collected:
            item = lookup[generated["problem_id"]]
            candidate = generated["output"]
            baseline = baselines[item["problem_id"]]
            if candidate["generation"].get("baseline_receipt") != baseline:
                raise ValueError("v17 preflight: actual generation lost its pre-admission baseline")
            view = item | {"t1": candidate, "p0_text": prep.p0(item), "block_index": 0}
            source = row_api.problem_contract_v17(item)
            policy = (row_api.t1_policy_receipt_v18(view, tokenizer=bundle.processor.tokenizer,
                gold=item["evaluation"]["gold"], choices=item["evaluation"].get("choices", []),
                verifier=source["verifier"], fact_catalog=source["fact_catalog"], rules=source["rules"]) if is_v18 else None)
            if policy is not None:
                view["t1_policy_receipt"] = policy
            receipt = row_api.verify_solution_v17(view, tokenizer=bundle.processor.tokenizer,
                gold=item["evaluation"]["gold"], choices=item["evaluation"].get("choices", []),
                verifier=source["verifier"], fact_catalog=source["fact_catalog"], rules=source["rules"])
            admissions.append(receipt)
            rows.append(_rl_row(item, candidate, baseline, policy=policy))
            admitted = row_api.a1_row_v17(view, tokenizer=bundle.processor.tokenizer, receipt=receipt)
            if admitted is not None:
                rows.append(admitted)
        encoded = campaign._encode(bundle, rows, lookup, image)
        coverage = _coverage(encoded)
        contract = v17_contract_for_update(plan["protocol"], len(plan["prep_batches"]), phase="rl",
            policy_sha256=rl.policy_sha256(bundle), expected_problem_ids=[i["problem_id"] for i in items],
            baseline_receipts={f"RL:{pid}": value for pid, value in baselines.items()},
            reward_values={f"RL:{r['problem_id']}": {"reward": r["reward"], "pg": r["pg"]} for r in rows if r["kind"] == "RL"})
        calibrator = calibrate_update_v18 if is_v18 else train_v17.calibrate_update_v17
        measurement = calibrator(bundle, encoded, contract,
            token_budget=args.token_budget) if coverage["actual_a1_count"] else None
        status = "PASS" if measurement is not None and not coverage["missing_actual_a1_lengths"] and coverage["actual_full_rows"] else "CALIBRATION_LOW_COVERAGE"
        transport = sampler.preflight_report
        coexistence = sampler.coexistence_reports
        if (not isinstance(transport, dict) or transport.get("pass") is not True or not coexistence
                or any(report.get("pass") is not True for report in coexistence)):
            raise ValueError("v17 preflight: actual F0 transport/coexistence checks did not pass")
        if measurement is not None and measurement.get("measurement") != "cuda-bf16":
            raise ValueError("v17 preflight: CPU timing is not BF16 hardware evidence")
        cost = _write(output.with_name("update-cost-calibration.json"), {"version": "v17-update-cost-receipt-v1",
            "status": status, **identity, "optimizer_steps": 0, "coverage": coverage, "measurement": measurement,
            "actual_abc_status": "MEASURED" if measurement is not None else "UNMEASURED_NO_ADMITTED_A1",
            "operator_review_required": status != "PASS", "conservative_eta_required": status != "PASS",
            "admission_receipts": admissions, "performance_parent_ids": [i["problem_id"] for i in items],
            "generated_records_sha256": digest(collected), "production_parents_consumed": 0,
            "capacity_fixture_is_solution_evidence": False})
        _validate_cost(cost)
        functional = {"version": VERSION, "status": "PASS", **identity, "local_smoke": False,
            "scope": "Actual F0 transport and synthetic maximum FULL backward capacity; timing coverage is separate.",
            "update_cost_status": status, "cost_operator_review_required": status != "PASS",
            "optimizer_steps": 0, "production_parents_consumed": 0,
            "data_validation_sha256": digest(validation), "capacity": capacity,
            "f0_transport": transport, "f0_effect_requirement": "NOT_APPLICABLE_ZERO_ADAPTER",
            "coexistence": coexistence, "update_cost_sha256": cost["sha256"],
            "update_cost_receipt": {"path": str(output.with_name("update-cost-calibration.json").absolute()),
                                    "file_sha256": file_sha(output.with_name("update-cost-calibration.json"))}}
        _validate_functional(functional)
        return _write(output, functional)
    finally:
        sampler.release()


def trained_export_preflight(args: Any, plan: dict) -> dict:
    """Check the committed final unmerged checkpoint through the existing real evaluation engine."""
    from . import campaign_v16b as campaign
    from .lora_preflight import check

    if plan.get("local_sequential"):
        raise ValueError("v17 preflight: local NF4 is not a trained BF16 evaluation export")
    split = getattr(args, "execution_phase", "legacy") == "evaluation"
    binding = bind_evaluation(args, plan, require_engine_validation=False) if split else bind_execution(args, plan)
    if getattr(args, "allow_model_run", False) is not True:
        raise ValueError("v17 preflight: --allow-model-run required")
    for field in ("export_output", "preflight_output", "hardware", "eval_protocol"):
        if getattr(args, field, None) is None:
            raise ValueError("v17 preflight: --" + field.replace("_", "-") + " required")
    ready = campaign.require_complete(Path(args.output), evaluation_decision=getattr(args, "evaluation_decision", None))
    ready_file = Path(args.output) / "export-ready.json"
    if ready.get("plan_sha256") != plan["sha256"]:
        raise ValueError("v17 preflight: export-ready checkpoint belongs to another plan")
    base = Path(getattr(args, "base", None) or plan["model_path"])
    if base.resolve() != Path(plan["model_path"]).resolve():
        raise ValueError("v17 preflight: --base differs from the pinned model path")
    model_ref = Path(args.export_output) / "model_ref.json"
    if model_ref.is_symlink() or not model_ref.is_file():
        raise ValueError("v17 preflight: exported model_ref.json required")
    if split and (str(model_ref.absolute()) != binding["model_ref"]["path"]
                  or file_sha(model_ref) != binding["model_ref"]["file_sha256"]):
        raise ValueError("v17 preflight: export differs from sealed final model")
    binding_file = Path(args.output) / ("checkpoint-evaluation-binding.json" if split else "evaluation-binding.json")
    _write(binding_file, {"binding": campaign._binding(plan)})
    report = check(model_ref, Path(ready["checkpoint"]), binding_file,
        base, Path(args.eval_protocol), Path(args.hardware), Path(args.preflight_output))
    if report.get("pass") is not True:
        raise ValueError("v17 preflight: trained adapter application failed")
    return _write(Path(args.preflight_output).with_name("trained-export-receipt.json"), {
        "version": "v17-trained-export-preflight-v1", "status": "PASS",
        **(binding["evaluation"] if split else _identity(args, plan)),
        **({"training": binding["training"]} if split else {}),
        "execution_binding_sha256": binding["sha256"], "export_ready_file_sha256": file_sha(ready_file),
        "model_ref_file_sha256": file_sha(model_ref),
        "preflight_file_sha256": file_sha(Path(args.preflight_output)), "trained_effect_required": True})
