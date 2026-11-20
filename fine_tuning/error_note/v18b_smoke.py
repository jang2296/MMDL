"""One offline NF4 acceptance run, supervised by a single 1,800-second clock."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import unittest

from .audit import digest
from .prep_v16b import file_sha, read_sealed, seal

VERSION = "error-note-v18b"
DIAGNOSTICS = (
    ("graph-error", "005290:00c977656045ae4d", False),
    ("scene-error", "003539:7b15b6cffe093171", False),
    ("graph-negative", "005255:362321e203f6d8f3", True),
    ("geometry-negative", "004545:786c3412cf22e7f0", True),
)


def cpu(output: Path) -> dict:
    """Run focused regressions outside the GPU clock; preserve actual unittest totals."""
    output.mkdir(parents=True, exist_ok=True)
    modules = [
        "fine_tuning.tests.test_error_note_v18_rows",
        "fine_tuning.tests.test_error_note_v18_prompt_identity",
        "fine_tuning.tests.test_error_note_v18_loss",
        "fine_tuning.tests.test_error_note_v18_campaign_runtime",
        "fine_tuning.tests.test_error_note_v16b_resume",
        "fine_tuning.tests.test_error_note_v17_preflight",
    ]
    modules += ["fine_tuning.tests." + path.stem for path in sorted(
        (Path(__file__).parents[1] / "tests").glob("test_error_note_v18b*.py"))]
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    with (output / "cpu.log").open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return seal(output / "cpu-receipt.json", {
        "version": VERSION, "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "modules": modules,
        "wall_seconds": time.monotonic() - started,
    })


def fixed_inputs(data_root: Path, native_root: Path) -> list[dict]:
    from .v18_data import load_items

    items = {item["problem_id"]: item for item in load_items(data_root, "train")}
    native = json.loads((native_root / "native-records.json").read_text())
    plan = json.loads((native_root / "native-plan.json").read_text())
    old_items = {item["problem_id"]: item for item in plan["items"]}
    selected = []
    for role, suffix, negative in DIAGNOSTICS:
        matches = [pid for pid in native if pid.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("smoke: diagnostic parent is missing or ambiguous: " + suffix)
        pid = matches[0]
        item = items[pid]
        if item["image_sha256"] != old_items[pid]["image_sha256"]:
            raise ValueError("smoke: diagnostic image lineage changed")
        for path, expected in zip(item["image_paths"], item["image_sha256"], strict=True):
            if file_sha(Path(path)) != expected:
                raise ValueError("smoke: diagnostic image bytes changed")
        selected.append({"role": role, "negative": negative, "item": item,
                         "native": native[pid], "native_sha256": digest(native[pid])})
    return selected


def diagnostic(args: argparse.Namespace) -> dict:
    """Real preparation, then one NOTE per fixed historical parent under the prepared policy."""
    import torch
    from . import campaign_v16b as campaign, prep_v16b as prep, rl, single, train_v18
    from . import v18b_rows as rows
    from .protocol import SMOKE_IMAGE_V16B, TRAIN_SAMPLING_V16B, v18b_contract_for_update
    from .run_v16 import _setup_execution
    from .train import _write_probe
    from fine_tuning.notebook_rl import checkpoint

    torch.set_num_threads(4)
    _setup_execution(3407)
    cases = fixed_inputs(args.data_root, args.native_root)
    bundle, protocol = campaign.load_f0(args, image=SMOKE_IMAGE_V16B)
    sampler = campaign.make_sampler(bundle, args, protocol, image=protocol["image"], sampling=TRAIN_SAMPLING_V16B)
    prep.generate(args.data_root, args.prep_root, bundle, sampler, image=protocol["image"],
                  local_sequential=True, version=VERSION)
    prepared = prep.rows(args.data_root, args.prep_root, bundle.processor.tokenizer,
                         local_sequential=True, version=VERSION)
    sampler.release()
    if prepared["prep_updates"] != 1:
        raise ValueError("smoke: exactly one real preparation update is required")
    optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
    cursor: dict = {"slot": 0, "updates": 0, "consumed_problem_ids": [], "consumed_row_ids": [],
              "policy_sha256": rl.policy_sha256(bundle)}
    all_items = {item["problem_id"]: item for split in ("prep", "train")
                 for item in prep.data_api(VERSION).load_items(args.data_root, split)}
    batch = prepared["batches"][0]
    contract = v18b_contract_for_update(protocol, 0, phase="prep", policy_sha256=rl.policy_sha256(bundle),
        expected_problem_ids=list(dict.fromkeys(row["problem_id"] for row in batch)))
    encoded = campaign._encode(bundle, batch, all_items, protocol["image"])
    preparation = single.run_joint_single_update(bundle, optimizer, scheduler, encoded, contract,
        cursor=cursor, block_ids=contract["expected_problem_ids"], allow_update=True, token_budget=12000)
    seal(args.output / "diagnostic-preparation.json", {"result": preparation, "cursor": cursor})
    if cursor["slot"] != 1:
        raise ValueError("smoke: preparation did not advance the actual cursor")

    accepted, results = [], []
    diagnostic_sampler = campaign.make_sampler(bundle, args, protocol, image=protocol["image"], sampling=TRAIN_SAMPLING_V16B)
    diagnostic_sampler.add_version(bundle, 1)
    for case in cases:
        item, native = case["item"], case["native"]
        record = deepcopy(item)
        record.update(p0_text=prep.p0(item), block_index=0,
                      **{key: deepcopy(native[key]) for key in ("t1", "obs") if key in native})
        if "obs" in record:
            record = rows.bind_historical_observation_v18b(record)
        contract = rows.problem_contract_v18b(item)
        kwargs = {"verifier": contract["verifier"], "tokenizer": bundle.processor.tokenizer,
                  "fact_catalog": contract["fact_catalog"], "rules": contract["rules"]}
        ledgers = [rows.audit_t1_v18b(record, **kwargs)]
        record["t1_ledger"] = ledgers[0]
        if "obs" in record:
            observed = rows.audit_observation_v18b(record, verifier=contract["verifier"],
                                                   tokenizer=bundle.processor.tokenizer)
            record.update(obs_score=observed, obs_ledger=observed["ledger"])
            ledgers.append(observed["ledger"])
        detail = {"role": case["role"], "problem_id": item["problem_id"], "negative": case["negative"],
                  "historical_native_sha256": case["native_sha256"], "ledgers": ledgers,
                  "policy_origin": "F0 plus one actual preparation update", "policy_version": 1}
        try:
            request = rows.build_note_request_v18b(record, ledgers=ledgers,
                gold=item["evaluation"]["gold"], tokenizer=bundle.processor.tokenizer)
        except ValueError as error:
            if not str(error).startswith("SKIP_NOTE_"):
                raise
            detail.update(status="NO_REQUEST", reason=str(error), accepted=False)
            results.append(detail)
            seal(args.output / (case["role"] + ".json"), detail)
            continue
        policy = rl.policy_sha256(bundle)
        generated_request = prep.make_request(item, request["prompt"], prep.preparation_stage(VERSION, "note"), cap=1024,
            sampling=TRAIN_SAMPLING_V16B, version=1, protocol_sha256=protocol["sha256"], policy_sha256=policy)
        remaining()
        diagnostic_sampler.submit(generated_request)
        completed = diagnostic_sampler.step()
        if len(completed) != 1:
            raise RuntimeError("smoke: one original versioned diagnostic completion required")
        _, output = completed[0]
        output["prompt"] = request["prompt"]
        record.update(note=output, note_request_receipt=request)
        receipt = rows.verify_note_v18b(record, ledgers=ledgers, gold=item["evaluation"]["gold"],
                                      choices=item["evaluation"]["choices"], **kwargs)
        nrow = rows.n_row_v18b(record, tokenizer=bundle.processor.tokenizer, receipt=receipt)
        detail.update(status="GENERATED", request=request, output=output, receipt=receipt,
                      accepted=nrow is not None, row_id=nrow["row_id"] if nrow else None)
        if case["negative"] and nrow is not None:
            detail["status"] = "FALSE_ACCEPTANCE"
        elif nrow is not None:
            accepted.append(nrow)
        results.append(detail)
        seal(args.output / (case["role"] + ".json"), detail)

    diagnostic_sampler.release()
    result = {"version": VERSION, "pid": os.getpid(), "cases": results,
              "accepted_n": len(accepted), "false_acceptance": sum(
                  r["negative"] and r["accepted"] for r in results), "origin": "historical-error diagnosis"}
    if accepted and not result["false_acceptance"]:
        contract = v18b_contract_for_update(protocol, 2, phase="rl",
            policy_sha256=rl.policy_sha256(bundle), expected_problem_ids=[],
            note_problem_ids=[row["problem_id"] for row in accepted])
        encoded = campaign._encode(bundle, accepted, all_items, protocol["image"])
        # This isolated N objective has no online RL block; never pass it off as campaign consumption.
        optimizer.zero_grad(set_to_none=True)
        capture: dict = {}
        stats = train_v18.backward_v18_block(bundle, encoded, contract, 12000, note_gradient_capture=capture)
        snapshots, alignment = train_v18.note_step_before_v18(bundle, capture, stats)
        step = bool(stats["has_gradient"] and stats["task_signal"] and not stats.get("hold"))
        if step:
            for parameters in bundle.params.values():
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        stats["note_update_alignment"] = train_v18.note_step_after_v18(
            bundle, capture, snapshots, alignment, step=step)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        cursor.update(slot=cursor["slot"] + 1, updates=cursor["updates"] + int(step),
            policy_sha256=rl.policy_sha256(bundle), phase="isolated_diagnostic",
            normal_campaign_cursor=False,
            consumed_row_ids=cursor["consumed_row_ids"] + stats["consumed_row_ids"])
        update = {"stats": stats, "optimizer_steps": int(step), "normal_campaign_slot": False,
                  "objective_slot_binding": 2, "actual_optimizer_updates": cursor["updates"]}
        binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
                   "data_sha256": file_sha(args.data_root / "payload-receipt.json"),
                   "code_sha256": digest(single.source_manifest()["actual_files"])}
        checkpoint_path = args.output / "checkpoint"
        single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor, path=checkpoint_path,
            protocol=protocol, binding=binding, allow_save=True)
        probe = args.output / "probe"
        probe.mkdir()
        _write_probe(bundle, encoded[:1], probe)
        result.update(update=update, cursor=cursor, n_row_ids=[row["row_id"] for row in accepted],
                      checkpoint_metadata_sha256=file_sha(checkpoint_path / "metadata.json"))
        seal(args.output / "diagnostic-reload-input.json", {"protocol": protocol, "binding": binding})
        seal(args.output / "accepted-rows.json", {"rows": accepted})
    return seal(args.output / "diagnostic-receipt.json", result)


def continuation_human(data_root: Path, excluded: set[str], tokenizer) -> tuple[dict, dict]:
    from . import prep_v16b as prep, v17_rows
    from .train_v18 import ROW_SCHEMA

    for item in prep.data_api(VERSION).load_items(data_root, "prep"):
        if item["problem_id"] in excluded or not item.get("references", {}).get("human_solution"):
            continue
        rows, _ = v17_rows.prep_rows_with_receipts_v17(
            [item | {"p0_text": prep.p0(item)}], tokenizer=tokenizer, max_self=0, max_human=1)
        if rows:
            return item, rows[0] | {"schema": ROW_SCHEMA, "evidence_grade": "DATASET_AUTHORED"}
    raise ValueError("smoke: no distinct admissible human preparation row for diagnostic continuation")


def reload_diagnostic(args: argparse.Namespace) -> dict:
    import torch
    from . import campaign_v16b as campaign, single, train_v18, rl
    from .protocol import v18b_contract_for_update
    from .run_v16 import _state_equal
    from .train import _read_probe_rows, compute_probe
    from fine_tuning.notebook_rl import checkpoint

    source = read_sealed(args.output / "diagnostic-reload-input.json")
    original = read_sealed(args.output / "diagnostic-receipt.json")
    bundle, optimizer, scheduler, cursor = single.load_single_continuation(args.model_path,
        args.output / "checkpoint", source["protocol"], binding=source["binding"],
        device="cuda:0", policy_factory=campaign._factory(args))
    saved = torch.load(args.output / "checkpoint" / "runtime.pt", map_location="cpu", weights_only=True)
    loaded_cursor = deepcopy(cursor)
    state_equal = all(_state_equal(actual, saved[key]) for key, actual in (
        ("optimizer", optimizer.state_dict()), ("scheduler", scheduler.state_dict()),
        ("rng", checkpoint.capture_rng())))
    path = args.output / "probe" / "probe.json"
    expected = json.loads(path.read_text())
    actual = compute_probe(bundle, _read_probe_rows(path.parent, expected))
    differences = [abs(a - b) for aa, bb in zip(actual["logprobs"], expected["logprobs"], strict=True)
                   for a, b in zip(aa, bb, strict=True)]
    equal = bool(differences) and max(differences) == 0 and actual["greedy_ids"] == expected["greedy_ids"]
    passed = (state_equal and equal and original["pid"] != os.getpid() and cursor == original["cursor"]
              and set(original["n_row_ids"]) <= set(cursor["consumed_row_ids"]))
    continuation = {}
    if passed:
        from .train import _trainable_hashes
        before = _trainable_hashes(bundle)
        selection = read_sealed(args.prep_root / "selection.json")
        item, row = continuation_human(args.data_root, set(selection["prep_ids"]), bundle.processor.tokenizer)
        if row["row_id"] in cursor["consumed_row_ids"]:
            raise ValueError("smoke: diagnostic continuation would consume a row twice")
        encoded = campaign._encode(bundle, [row], {item["problem_id"]: item}, source["protocol"]["image"])
        contract = v18b_contract_for_update(source["protocol"], 0, phase="prep",
            policy_sha256=rl.policy_sha256(bundle), expected_problem_ids=[item["problem_id"]])
        checkpoint.restore_rng(saved["rng"])
        optimizer.zero_grad(set_to_none=True)
        stats = train_v18.backward_v18_block(bundle, encoded, contract, 12000)
        step = bool(stats["has_gradient"] and stats["task_signal"] and not stats.get("hold"))
        if step:
            for parameters in bundle.params.values():
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        changed = before != _trainable_hashes(bundle)
        cursor.update(slot=cursor["slot"] + 1, updates=cursor["updates"] + int(step),
            policy_sha256=rl.policy_sha256(bundle), phase="isolated_diagnostic_resumed",
            consumed_row_ids=cursor["consumed_row_ids"] + stats["consumed_row_ids"])
        single.save_single_continuation(bundle, optimizer, scheduler, cursor=cursor,
            path=args.output / "checkpoint-resumed", protocol=source["protocol"], binding=source["binding"], allow_save=True)
        continuation = {"problem_id": item["problem_id"], "row_id": row["row_id"],
            "target_origin": "dataset_native_human", "optimizer_steps": int(step), "parameters_changed": changed,
            "n_row_ids_preserved": set(original["n_row_ids"]) <= set(cursor["consumed_row_ids"]),
            "metadata_sha256": file_sha(args.output / "checkpoint-resumed" / "metadata.json")}
        passed = passed and step and changed and continuation["n_row_ids_preserved"]
    return seal(args.output / "diagnostic-reload.json", {"version": VERSION,
        "status": "PASS" if passed else "FAIL", "resumed_learning": continuation, "producer_pid": original["pid"], "reload_pid": os.getpid(),
        "optimizer_scheduler_rng_equal": state_equal, "cursor_equal": loaded_cursor == original["cursor"],
        "probe_equal": equal, "probe_max_abs_difference": max(differences) if differences else None,
        "n_row_ids": original["n_row_ids"], "checkpoint_metadata_sha256": original["checkpoint_metadata_sha256"]})


def remaining() -> float:
    value = float(os.environ["MMDL_V18B_SMOKE_DEADLINE"]) - time.monotonic()
    if value <= 0:
        raise TimeoutError("smoke: shared 1,800-second deadline reached")
    return value


def process_identity(pid: int) -> dict:
    root = Path("/proc") / str(pid)
    return {"pid": pid, "startticks": root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19],
            "cwd": str(root.joinpath("cwd").resolve()),
            "command_sha256": digest(root.joinpath("cmdline").read_bytes().hex())}


def signal_owned(process: subprocess.Popen, identity: dict, number: int) -> bool:
    if process.poll() is not None:
        return False
    try:
        if process_identity(process.pid) != identity:
            raise RuntimeError("smoke: process identity changed; refusing signal")
    except FileNotFoundError:
        return False
    os.killpg(process.pid, number)
    return True


def inflight_after_checkpoint(campaign: Path, state: dict) -> dict | None:
    snapshot = json.loads(Path(state["runtime"]["path"]).read_text())["state"]
    journal = campaign / "runtime" / "requests.jsonl"
    tail = journal.read_bytes()[snapshot["journal_bytes"]:]
    started, closed = {}, set()
    for raw in tail.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            continue
        event = json.loads(raw)
        if event["status"] == "STARTED":
            started[event["request_id"]] = event
        elif event["status"] == "COMPLETED":
            closed.add(event["request_id"])
    return next((event for rid, event in started.items() if rid not in closed), None)


def run_child(command: list[str], *, env: dict, log: Path, launch: dict, output: Path,
              campaign: Path | None = None) -> tuple[int, dict | None]:
    """Kill only the owned process after block two is durable and a later request is active."""
    killed = None
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        identity = process_identity(process.pid)
        try:
            launch["workers"].append(identity | {"command": command})
            from .single import _state
            seal(output / f"worker-{len(launch['workers'])}.json", identity | {"command": command})
            _state(output / "launch-current.json", launch)
            while process.poll() is None:
                if "advance" in command and remaining() <= 180:
                    index = command.index("--output")
                    marker = Path(command[index + 1]) / "STOP_REQUESTED"
                    if marker.parent.exists() and not marker.exists():
                        marker.write_text("v18b smoke: reserve final three minutes for saved boundary and reload\\n")
                if campaign is not None and (campaign / "campaign-state.json").is_file():
                    state = read_sealed(campaign / "campaign-state.json")
                    cursor = state["cursor"]
                    if cursor.get("rl_block") == 2 and cursor.get("phase") == "rl":
                        inflight = inflight_after_checkpoint(campaign, state)
                        if inflight is not None:
                            signal_owned(process, identity, signal.SIGKILL)
                            killed = {"identity": identity, "state": state,
                                      "checkpoint_metadata_sha256": file_sha(Path(state["checkpoint"]["path"])),
                                      "after_main_blocks": 2, "next_consumption_block": 3,
                                      "actual_inflight_request": inflight}
                            seal(output / "sigkill-receipt.json", killed)
                            break
                if remaining() <= 5:
                    raise TimeoutError("smoke: shared deadline exhausted")
                time.sleep(.2)
            return process.wait(timeout=min(5, remaining())), killed
        finally:
            if process.poll() is None:
                signal_owned(process, identity, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    signal_owned(process, identity, signal.SIGKILL)
                    process.wait(timeout=2)


def gates(*, cpu_receipt: dict, diagnostic_receipt: dict, diagnostic_reload: dict,
          resume_receipt: dict, monitoring_receipt: dict, campaign_n_linked: int = 0,
          local_acceptance: dict | None = None) -> dict:
    cases = diagnostic_receipt.get("cases", [])
    controls = [case for case in cases if case.get("negative") is True]
    expected = {role: suffix for role, suffix, negative in DIAGNOSTICS if negative}
    audited = [case for case in controls if case.get("role") in expected
        and case.get("problem_id", "").endswith(expected[case["role"]])
        and case.get("accepted") is False
        and any(ledger.get("stage") == "T1" and ledger.get("problem_id") == case["problem_id"]
                and ledger.get("sha256") for ledger in case.get("ledgers", []))
        and (case.get("status") == "NO_REQUEST" and case.get("reason") == "SKIP_NOTE_NO_ERROR"
             or case.get("status") == "GENERATED" and bool(case.get("output"))
             and case.get("receipt", {}).get("problem_id") == case["problem_id"]
             and case.get("receipt", {}).get("n_accepted") is False
             and case.get("receipt", {}).get("sha256"))]
    negatives_valid = (len(controls) == len(audited) == 2
        and {case["role"] for case in audited} == set(expected)
        and type(diagnostic_receipt.get("false_acceptance")) is int
        and diagnostic_receipt["false_acceptance"] == 0)
    linked = bool(diagnostic_receipt.get("accepted_n", 0) and diagnostic_reload.get("status") == "PASS"
                  and diagnostic_receipt.get("update", {}).get("optimizer_steps", 0)
                  and diagnostic_reload.get("resumed_learning", {}).get("optimizer_steps", 0)
                  and any(evidence.get("gradient_finite_nonzero") and evidence.get("note_gradient_retained")
                          and evidence.get("supervised_tokens", 0) > 0
                          and evidence.get("row_id") in diagnostic_receipt.get("n_row_ids", [])
                          and evidence.get("source_evidence_sha256")
                          for evidence in diagnostic_receipt.get("update", {}).get("stats", {}).get("n_parent_evidence", [])))
    result = {
        "cpu_regression": {"status": "PASS" if cpu_receipt.get("status") == "PASS" else "FAIL",
                           "tests": cpu_receipt.get("tests", 0)},
        "actual_qwen_n_saved_and_resumed": {"status": "PASS" if linked or campaign_n_linked else "FAIL",
            "diagnostic_n": diagnostic_receipt.get("accepted_n", 0), "campaign_n_linked": campaign_n_linked},
        "negative_false_acceptance": {"status": "PASS" if negatives_valid else "FAIL",
            "controls": len(controls), "audited_controls": len(audited),
            "coverage_limited": any(case.get("status") == "NO_REQUEST" for case in audited),
            "generated_notes": sum(r.get("negative", False) and r.get("status") in {"GENERATED", "FALSE_ACCEPTANCE"} for r in cases),
            "skipped_no_error": sum(r.get("negative", False) and r.get("status") == "NO_REQUEST" for r in cases),
            "false_acceptance": diagnostic_receipt.get("false_acceptance")},
        "sigkill_resume": {"status": resume_receipt.get("status", "FAIL"),
                           "resumed_slots": resume_receipt.get("resumed_slots", 0)},
        "b2_monitor": {"status": monitoring_receipt.get("status", "FAIL")},
    }
    local_acceptance = local_acceptance or {"status": "FAIL", "reason": "LOCAL_RUN_NOT_CHECKED"}
    return {"gates": result, "local_acceptance": local_acceptance,
            "paid_run_ready": all(r["status"] == "PASS" for r in result.values())
                              and local_acceptance.get("status") == "PASS"}


def exact_campaign_reload(state: dict, receipt: dict, row_id: str) -> bool:
    cursor = state.get("cursor") or {}
    return bool(
        receipt.get("runtime_equal") is True and receipt.get("greedy_equal") is True
        and receipt.get("probe_max_abs_difference") == 0
        and receipt.get("plan_sha256") == state.get("plan_sha256")
        and receipt.get("checkpoint") == state.get("checkpoint")
        and receipt.get("cursor") == cursor and row_id in cursor.get("consumed_row_ids", [])
        and type(receipt.get("reload_pid")) is int and type(receipt.get("producer_pid")) is int
        and receipt["reload_pid"] != receipt["producer_pid"]
        and receipt["producer_pid"] == cursor.get("producer_pid"))


def summarize(args: argparse.Namespace) -> dict:
    def optional(path: Path | None) -> dict:
        return read_sealed(path) if path is not None and path.is_file() else {}

    diagnostic_root, campaign_root = args.output / "diagnostic", args.output / "campaign"
    state = optional(campaign_root / "campaign-state.json")
    reloaded = [optional(path) for path in campaign_root.glob("reload-*.json")]
    consumed = set(state.get("cursor", {}).get("consumed_row_ids", []))
    linked = set()
    updates = []
    for path in sorted((campaign_root / "update-results").glob("slot-*.json")):
        update = read_sealed(path)
        updates.append(update)
        stats = update["result"]["stats"]
        for evidence in stats.get("n_parent_evidence", []):
            if (update["result"].get("optimizer_steps") and evidence.get("gradient_finite_nonzero")
                    and evidence.get("note_gradient_retained") and evidence.get("supervised_tokens", 0) > 0
                    and evidence.get("source_evidence_sha256") and evidence["row_id"] in consumed
                    and any(exact_campaign_reload(state, receipt, evidence["row_id"]) for receipt in reloaded)):
                linked.add(evidence["problem_id"])
    recoveries = sorted(args.output.glob("recovery-run-receipt-*.json"),
                        key=lambda path: int(path.stem.rsplit("-", 1)[1]))
    run = optional(recoveries[-1] if recoveries else args.output / "run-receipt.json")
    plan = optional(campaign_root / "campaign-plan.json")
    resume = optional(args.output / "resume-receipt.json")
    killed = optional(args.output / "sigkill-receipt.json")
    cursor = state.get("cursor") or {}
    before, after = resume.get("before_cursor") or {}, resume.get("after_cursor") or {}
    checkpoint = state.get("checkpoint") or {}
    checkpoint_path = Path(checkpoint.get("path", ""))
    checkpoint_valid = bool(checkpoint.get("file_sha256") and checkpoint_path.is_file()
        and file_sha(checkpoint_path) == checkpoint["file_sha256"])
    exact_reload = any(exact_campaign_reload(state, receipt, row_id)
                       for receipt in reloaded for row_id in consumed)
    resumed = (resume.get("status") == "PASS" and after == cursor
        and resume.get("duplicates") == 0 and len(consumed) == len(cursor.get("consumed_row_ids", []))
        and set(before.get("consumed_row_ids", [])) <= consumed
        and type(before.get("producer_pid")) is int and type(after.get("producer_pid")) is int
        and before["producer_pid"] != after["producer_pid"]
        and cursor.get("slot", 0) - before.get("slot", 0) == resume.get("resumed_slots", 0) >= 1
        and before.get("rl_block") == 2 and killed.get("state", {}).get("cursor") == before
        and killed.get("identity", {}).get("pid") == before["producer_pid"]
        and killed.get("after_main_blocks") == 2
        and killed.get("actual_inflight_request", {}).get("status") == "STARTED")
    total, completed = plan.get("total_slots", 0), cursor.get("slot", 0)
    complete = (run.get("status") == state.get("status") == "COMPLETE"
                and total > 0 and completed == total and not state.get("stop_reasons"))
    partial = (run.get("status") == "PARTIAL" and state.get("status") == "STOPPED"
        and state.get("stop_reasons") == ["STOP_REQUESTED"]
        and (campaign_root / "STOP_REQUESTED").is_file() and 0 < completed < total)
    bounded = run.get("deadline_seconds") == 1800 and 0 < run.get("wall_seconds", 0) <= 1800
    valid = bool(bounded and run.get("error", "MISSING") is None
        and (complete or partial) and resumed and checkpoint_valid and exact_reload
        and plan.get("local_sequential") is True and plan.get("sha256") == state.get("plan_sha256"))
    local_acceptance = {"status": "PASS" if valid else "FAIL",
        "completion": "COMPLETE" if complete else "BOUNDED_PARTIAL" if partial else "INCOMPLETE",
        "completed_slots": completed, "planned_slots": total, "remaining_slots": max(0, total - completed),
        "supervisor_error": run.get("error"), "within_original_time_limit": bounded,
        "checkpoint_valid": checkpoint_valid, "exact_final_reload": exact_reload,
        "resumed_after_sigkill": resumed}
    result = gates(cpu_receipt=optional(args.cpu_receipt),
        diagnostic_receipt=optional(diagnostic_root / "diagnostic-receipt.json"),
        diagnostic_reload=optional(diagnostic_root / "diagnostic-reload.json"),
        resume_receipt=resume, monitoring_receipt=optional(args.monitoring_receipt),
        campaign_n_linked=len(linked), local_acceptance=local_acceptance)
    result.update(version=VERSION, run=run,
        natural_campaign_n_parent_ids=sorted(linked),
        campaign_status=state.get("status", "NOT_STARTED"), cursor=state.get("cursor"),
        consumed_rows=len(consumed), update_receipts=len(updates),
        exact_n_reload_audit={row_id: any(exact_campaign_reload(state, receipt, row_id)
            for receipt in reloaded) for row_id in sorted(consumed) if row_id.endswith(":N")},
        limitations=["NF4/HF functionality only", "no BF16/vLLM or accuracy claim",
                     "historical diagnosis is separate from current-policy natural repair"])
    from .single import _state
    _state(args.output / "acceptance-current.json", result)
    index = len(list(args.output.glob("recovery-event-*.json")))
    return seal(args.output / f"acceptance-summary-{index}.json", result)


def supervise(args: argparse.Namespace, *, recover: bool = False) -> dict:
    if args.cpu_receipt is None or read_sealed(args.cpu_receipt).get("status") != "PASS":
        raise ValueError("smoke: a passing focused CPU receipt is required before GPU execution")
    if recover:
        launch = read_sealed(args.output / "launch-current.json" if (args.output / "launch-current.json").exists()
                             else args.output / "launch.json")
        started, deadline = launch["started_monotonic"], launch["deadline_monotonic"]
        if time.monotonic() >= deadline:
            raise TimeoutError("smoke: original registered trial deadline expired")
        if (args.output / "campaign" / "campaign-plan.json").exists():
            raise ValueError("smoke: recovery command is only for this pre-generation controller failure")
        for worker in launch["workers"]:
            try:
                actual = process_identity(worker["pid"])
            except FileNotFoundError:
                continue
            expected = {key: worker[key] for key in ("pid", "startticks", "cwd", "command_sha256")}
            if actual == expected:
                raise RuntimeError("smoke: an owned worker is still live; do not duplicate its phase")
        recovery_index = len(list(args.output.glob("recovery-event-*.json"))) + 1
        seal(args.output / f"recovery-event-{recovery_index}.json", {
            "status": "CONTINUE_ORIGINAL_TRIAL", "original_deadline": deadline, "controller_pid": os.getpid(),
            "previous_failure_sha256": file_sha(args.output / "run-receipt.json"),
            "diagnostic_sha256": file_sha(args.output / "diagnostic" / "diagnostic-receipt.json"),
            "prep_sha256": file_sha(args.output / "prep" / "prep-rows.json"),
            "f0_generations_sha256": file_sha(args.output / "prep" / "f0-generations.json"),
            "diagnostic_resampled": False, "deadline_reset": False})
    else:
        started = time.monotonic()
        deadline = started + 1800
        args.output.mkdir(parents=True, exist_ok=False)
        launch = {"version": VERSION, "supervisor_pid": os.getpid(), "started_monotonic": started,
                  "deadline_monotonic": deadline, "max_seconds": 1800, "workers": [],
                  "scope": "offline NF4 functionality; no BF16/vLLM or accuracy claim"}
        seal(args.output / "launch.json", launch)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
               MMDL_V18B_SMOKE_DEADLINE=str(deadline))
    os.environ["MMDL_V18B_SMOKE_DEADLINE"] = str(deadline)
    options = ["--data-root", str(args.data_root), "--prep-root", str(args.output / "prep"),
               "--model-path", str(args.model_path), "--nf4-cache", str(args.nf4_cache),
               "--native-root", str(args.native_root)]
    module = [sys.executable, "-B", "-m", "fine_tuning.error_note.v18b_smoke"]
    status, error = "PARTIAL", None
    try:
        if not recover:
            from fine_tuning.notebook_rl.quantized_policy import read_quantized_base_metadata
            metadata = read_quantized_base_metadata(args.nf4_cache)
            cases = fixed_inputs(args.data_root, args.native_root)
            if shutil.disk_usage(args.output).free < 50 * 1024**3:
                raise ValueError("smoke: less than 50 GiB disk headroom")
            seal(args.output / "preflight.json", {"status": "PASS", "nf4_metadata_sha256": digest(metadata),
                "diagnostic_ids": [case["item"]["problem_id"] for case in cases],
                "image_sha256": [case["item"]["image_sha256"] for case in cases]})
            diagnostic_root = args.output / "diagnostic"
            diagnostic_root.mkdir()
            code, _ = run_child(module + ["diagnose", *options, "--output", str(diagnostic_root)], env=env,
                log=args.output / "diagnose.log", launch=launch, output=args.output)
            if code:
                raise RuntimeError(f"diagnostic worker failed ({code}); no retry")
            if (diagnostic_root / "diagnostic-reload-input.json").exists():
                code, _ = run_child(module + ["reload-diagnostic", *options, "--output", str(diagnostic_root)], env=env,
                    log=args.output / "diagnostic-reload.log", launch=launch, output=args.output)
                if code:
                    raise RuntimeError(f"diagnostic reload failed ({code})")
        campaign_root = args.output / "campaign"
        command = [sys.executable, "-B", "-m", "fine_tuning.error_note.campaign_v16b", "advance",
            "--version", VERSION, "--data-root", str(args.data_root), "--prep-root", str(args.output / "prep"),
            "--model-path", str(args.model_path), "--nf4-cache", str(args.nf4_cache), "--output", str(campaign_root),
            "--precision", "nf4", "--engine", "hf", "--local-sequential",
            "--checkpoint-after-main-blocks", "2", "--allow-model-run", "--allow-update", "--allow-save"]
        code, killed = run_child(command, env=env, log=args.output / ("campaign-first-recovery.log" if recover else "campaign-first.log"), launch=launch,
                                  output=args.output, campaign=campaign_root)
        if killed is None or code != -signal.SIGKILL:
            raise RuntimeError("smoke: campaign did not cross the required SIGKILL boundary")
        code, _ = run_child(command, env=env, log=args.output / "campaign-resume.log", launch=launch, output=args.output)
        state = read_sealed(campaign_root / "campaign-state.json")
        before, after = killed["state"]["cursor"], state["cursor"]
        ids = after["consumed_row_ids"]
        resumed = after["slot"] - before["slot"]
        valid = (code in (0, 2) and resumed >= 1 and len(ids) == len(set(ids))
                 and set(before["consumed_row_ids"]) <= set(ids)
                 and before["producer_pid"] != after["producer_pid"])
        seal(args.output / "resume-receipt.json", {"status": "PASS" if valid else "FAIL",
            "resumed_slots": resumed, "duplicates": len(ids) - len(set(ids)), "before_cursor": before,
            "after_cursor": after, "campaign_status": state["status"]})
        if valid and remaining() > 60:
            reload_command = list(command)
            reload_command[reload_command.index("advance")] = "reload"
            reload_code, _ = run_child(reload_command, env=env, log=args.output / "campaign-reload.log",
                launch=launch, output=args.output)
            if reload_code:
                raise RuntimeError(f"campaign final reload failed ({reload_code})")
        status = "COMPLETE" if state["status"] == "COMPLETE" and valid else "PARTIAL"
    except Exception as exc:
        error = repr(exc)
    result = {"version": VERSION, "status": status, "error": error,
              "wall_seconds": time.monotonic() - started, "deadline_seconds": 1800}
    seal(args.output / (f"recovery-run-receipt-{recovery_index}.json" if recover else "run-receipt.json"), result)
    summarize(args)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cpu", "run", "diagnose", "reload-diagnostic", "summary", "recover"))
    parser.add_argument("--output", required=True, type=Path)
    for name in ("data-root", "prep-root", "model-path", "nf4-cache", "native-root",
                 "cpu-receipt", "monitoring-receipt"):
        parser.add_argument("--" + name, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cpu":
        result = cpu(args.output)
    elif args.command == "summary":
        result = summarize(args)
    else:
        if any(getattr(args, name) is None for name in ("data_root", "model_path", "nf4_cache", "native_root")):
            raise ValueError("smoke: explicit local data, model, cache and native inputs required")
        args.version, args.precision, args.engine, args.local_sequential = VERSION, "nf4", "hf", True
        if args.command in {"run", "recover"}:
            result = supervise(args, recover=args.command == "recover")
        else:
            if args.prep_root is None:
                raise ValueError("smoke: --prep-root required")
            if not all(os.environ.get(name) == "1" for name in (
                    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")):
                raise ValueError("smoke: offline environment is required")
            result = diagnostic(args) if args.command == "diagnose" else reload_diagnostic(args)
    print(json.dumps({key: result.get(key) for key in ("status", "tests", "accepted_n", "wall_seconds", "error")}))
    return 0 if result.get("status") in {None, "PASS", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
