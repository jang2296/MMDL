"""Plan the error-note stages as pod/chain.py specs (`pod-chain-spec-v1`).

CLI: python -m fine_tuning.error_note.plan plan --stage {note-300,note-3417,note-v2,note-v3-collect,note-v3-resolve,note-v3-train}
         --params PARAMS.json --output STEPS.json
Writes STEPS.json once (an existing file is refused) and prints its sha256. Same params -> same bytes.
Executes nothing; the job's main argv runs the steps with `pod.chain run --spec STEPS.json`.

params (`error-note-plan-params-v1`): exactly the keys below; the paths are canonical absolute paths.
pilot_limit 0 omits the pilot steps. kv_cache_gib is a positive number or "auto"; seqs is a non-empty list of positive
integers or "auto". Both reach collect as given, and collect sizes the KV cache and the sweep from the GPU's memory.
Step timeouts by kind: pin-order and the collect steps -> collect;
train-* and pilot-train -> train; reload-* and pilot-reload -> reload; pilot-export -> export.

Stage note-v2 (claudedocs/plan_self_note_v2_20261102.md) takes `error-note-plan-params-v2`: the v1 keys without
pilot_limit and arms (no pilot; arm note only) plus volume_root and reuse_t1_sha256. Steps: pin-order (item_order_v2.json)
-> sweep (T1), or with reuse_t1_sha256 a pin of the copied <out_root>/collect/t1.jsonl instead -> note (v2) ->
resolve -> rows (v2) -> train-note (protocol v2) -> reload-note -> export to <volume_root>/eval-data/proof/export-note-v2.
Without the sweep, note and resolve take --max-num-seqs auto (the middle sweep candidate for the KV size).

Stages note-v3-collect, note-v3-resolve and note-v3-train take `error-note-plan-params-v3`. All use external order_path,
training_rules_path and source_provenance_path with their SHA256 pins, volume_root, auto KV/concurrency and
token_budget 24000. The collect stage stops after fresh T1 and v2 notes; the resolve stage additionally requires
reviews_path/reviews_sha256 and t1_sha256/note_sha256. It pins those collected files before audit -> resolve ->
candidate rows v3, then stops. No private file is read while planning. Train requires closed rows_path/audit_path/solution_audit_path
and their SHA256 pins, then performs admission and both training arms without regenerating targets. Shared generation
settings remain schema fields for all stages. Note and solution reviews are created outside these executable specs.

Stages single-pilot and single-main (single-response RL + SFT campaign, error_note.single_campaign) take
`error-note-plan-params-single-v1`: SINGLE_KEYS (+ the dev-evaluation inputs for single-pilot, + pilot_gate_path/sha256
for single-main). The campaign directory is <out_root>/campaign; every campaign call carries the same pinned inputs,
including the v11 note-reference supplement (note_reference_supplement_path/_sha256).
single-pilot: pins -> advance --max-blocks 1 -> fresh-process reload receipt -> advance --max-blocks 18 (checkpoint-19 =
PILOT_COMPLETE) -> fresh-process reload receipt -> export checkpoint-19 -> external-dev preregister-hybrid -> F0 and HYB
run/grade -> analyze-hybrid. The pilot-gate JSON is built afterwards by the operator (evaluate pilot-gate).
single-main: pins (incl. the gate) -> advance --continue-main --max-blocks 200 -> reload receipt -> export the final
checkpoint to <volume_root>/eval-data/proof/export-single-final. The one MMMU pass is a separate eval-ft job (arm FT-SINGLE,
prereg v7): a training job never carries an MMMU evaluation.
single-direct (v5, user 2026-11-06; runs under job stage single-main): pins -> advance 1 -> reload -> advance 18 ->
reload checkpoint-19 -> advance --continue-main --direct-main 200 (no external-dev gate) -> reload -> export the final checkpoint.
Optional `execution_max_blocks` (integer 20..219, this stage only) replaces the 219-block total: every campaign call carries it,
advance-main runs total-19 blocks instead of 200, and the final checkpoint is checkpoint-<total>; a shorter campaign's checkpoint is exported by the evaluation stage, not by this plan.
Optional single-direct `warm_start` = {plan,checkpoint,reload}_{path,sha256} + slot (v15 warm start v2; needs
execution_max_blocks, 19 < slot < total): pins -> pin the three warm files -> advance-main --continue-main --direct-main
total-slot -> reload checkpoint-<total>; every campaign call carries the six --warm-start-* arguments, and there is no export.

Stages v17-calibration and v17 take error-note-plan-params-v17. Stage the private bundle before approved start.sh;
the calibration chain verifies it, then runs source/doctor/preparation and separate BF16 functional/throughput checks.
Seal the main spec only after the two measured receipt files exist, adding their raw-file SHA256 pins. Both phases
share the recovery parent params.run_dir and out_root; use different child job.run_dir values for their immutable chain
logs. The main funding gate checks the declared balance snapshot; execution uses the approved job.financial watchdog.
The 14..17-hour target supplies neither a training cutoff nor job.max_main_seconds.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
import math
import posixpath
import subprocess
import sys
from pathlib import Path
from typing import Any

from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT
from fine_tuning.notebook_rl.pod.common import HEX40, HEX64, sha256_bytes

PARAMS_VERSION = "error-note-plan-params-v1"
SPEC_VERSION = "pod-chain-spec-v1"
PARAMS_VERSION_V2 = "error-note-plan-params-v2"
PARAMS_VERSION_V3 = "error-note-plan-params-v3"
V3_STAGES = ("note-v3-collect", "note-v3-resolve", "note-v3-train")
SINGLE_STAGES = ("single-pilot", "single-main", "single-direct")
STAGES = ("note-300", "note-3417", "note-v2", *V3_STAGES, *SINGLE_STAGES, "content-v16", "v16b", "v17-calibration", "v17", "v17-train", "v17-eval", "v18-train", "v18-eval", "v18b-train", "v18b-eval")
ARMS = ("note", "solution")
TIMEOUT_KINDS = ("collect", "train", "reload", "export")
PATH_KEYS = ("python", "repo_root", "build_root", "model_path", "out_root")
PARAM_KEYS = {"version", "commit", "order_sha256", "limit", "pilot_limit", "seqs", "sweep_seconds",
              "kv_cache_gib", "max_num_batched_tokens", "token_budget", "gradient_checkpointing", "arms",
              "timeouts", *PATH_KEYS}
PARAM_KEYS_V2 = (PARAM_KEYS - {"pilot_limit", "arms"}) | {"volume_root", "reuse_t1_sha256"}
V3_PRIVATE_PATHS = ("order_path", "training_rules_path", "source_provenance_path")
V3_HASH_KEYS = ("training_rules_sha256", "source_provenance_sha256")
PARAM_KEYS_V3 = (PARAM_KEYS_V2 - {"reuse_t1_sha256"}) | {*V3_PRIVATE_PATHS, *V3_HASH_KEYS}
PARAM_KEYS_V3_RESOLVE = PARAM_KEYS_V3 | {"reviews_path", "reviews_sha256", "t1_sha256", "note_sha256"}
PARAM_KEYS_V3_TRAIN = PARAM_KEYS_V3 | {"rows_path", "rows_sha256", "audit_path", "audit_sha256",
                                     "solution_audit_path", "solution_audit_sha256"}
COLLECT = "fine_tuning.error_note.collect"
TRAIN = "fine_tuning.error_note.train"
CHAIN = "fine_tuning.notebook_rl.pod.chain"
PROOF = "fine_tuning.notebook_rl.proof_eval"
CAMPAIGN = "fine_tuning.error_note.single_campaign"
EVALUATE = "fine_tuning.error_note.evaluate"
PARAMS_VERSION_SINGLE = "error-note-plan-params-single-v1"
# <name>_path + <name>_sha256; the last one is the v11 dataset note-reference supplement (references.py)
SINGLE_INPUTS = ("order", "frozen_approval", "source_provenance", "training_rules", "note_reference_supplement")
SINGLE_PATHS = ("python", "repo_root", "build_root", "model_path", "out_root", "volume_root",
                *(f"{name}_path" for name in SINGLE_INPUTS))
SINGLE_HASHES = (*(f"{name}_sha256" for name in SINGLE_INPUTS), "pilot_ids_sha256", "dev_ids_sha256",
                 "retention_ids_sha256")
SINGLE_CAPACITY = ("max_num_seqs", "max_num_batched_tokens", "token_budget")
SINGLE_KEYS = {"version", "commit", "timeouts", "engine", *SINGLE_PATHS, *SINGLE_HASHES, *SINGLE_CAPACITY}
SINGLE_ENGINE_KEYS = {"name", "kv_cache_gib", "gpu_memory_utilization", "sleep"}  # campaign --engine/... flags
# stage -> (extra paths, extra hashes, timeout kinds)
SINGLE_EXTRA = {"single-pilot": (("dev_manifest_path", "dev_vault_root", "dev_rules_path"),
                                 ("dev_manifest_sha256", "dev_rules_sha256"),
                                 ("cpu", "advance_first", "advance_pilot", "reload", "export", "evaluate")),
                "single-main": (("pilot_gate_path",), ("pilot_gate_sha256",), ("cpu", "advance_main", "reload", "export")),
                "single-direct": ((), (), ("cpu", "advance_first", "advance_pilot", "advance_main", "reload", "export"))}
PILOT_BLOCKS, MAIN_BLOCKS = 19, 200  # single_campaign: 219 slots of 16, PILOT_COMPLETE after slot 19 (300 items)
WARM_FILES = ("plan", "checkpoint", "reload")  # single-direct warm_start: <name>_path/_sha256 -> --warm-start-<name>[-sha256]


def _positive_int(params: dict[str, Any], key: str, low: int) -> None:
    value = params[key]
    if type(value) is not int or value < low:
        raise ValueError(f"{key} must be an integer >= {low}")


def _check(stage: str, params: Any) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    if stage in ("v17-train", "v17-eval", "v18-train", "v18-eval", "v18b-train", "v18b-eval"):
        return _check_v17_split(stage, params)
    if stage in ("v17-calibration", "v17"):
        return _check_v17(params, calibration=stage == "v17-calibration")
    if stage == "v16b":
        return _check_v16b(params)
    if stage == "content-v16":
        return _check_content_v16(params)
    if stage in SINGLE_STAGES:
        return _check_single(stage, params)
    v2, v3 = stage == "note-v2", stage in V3_STAGES
    extra_paths: tuple[str, ...] = ()
    extra_hashes: tuple[str, ...] = ()
    if v3:
        keys, version = PARAM_KEYS_V3, PARAMS_VERSION_V3
        if stage == "note-v3-resolve":
            keys = PARAM_KEYS_V3_RESOLVE
            extra_paths, extra_hashes = ("reviews_path",), ("reviews_sha256", "t1_sha256", "note_sha256")
        elif stage == "note-v3-train":
            keys = PARAM_KEYS_V3_TRAIN
            extra_paths = ("rows_path", "audit_path", "solution_audit_path")
            extra_hashes = ("rows_sha256", "audit_sha256", "solution_audit_sha256")
    else:
        keys, version = (PARAM_KEYS_V2, PARAMS_VERSION_V2) if v2 else (PARAM_KEYS, PARAMS_VERSION)
    missing, unknown = sorted(keys - set(params)), sorted(set(params) - keys)
    if missing or unknown:
        raise ValueError(f"params keys: missing {missing}, unknown {unknown}")
    if params["version"] != version:
        raise ValueError(f"version must be {version}")
    if not isinstance(params["commit"], str) or not HEX40.fullmatch(params["commit"]):
        raise ValueError("commit must be 40 lowercase hex characters")
    if not isinstance(params["order_sha256"], str) or not HEX64.fullmatch(params["order_sha256"]):
        raise ValueError("order_sha256 must be 64 lowercase hex characters")
    path_keys: tuple[str, ...] = (*PATH_KEYS, "volume_root") if v2 or v3 else PATH_KEYS
    if v3:
        path_keys += (*V3_PRIVATE_PATHS, *extra_paths)
    for key in path_keys:
        value = params[key]
        if (not isinstance(value, str) or not value.startswith("/") or value == "/"
                or posixpath.normpath(value) != value):
            raise ValueError(f"{key} must be a canonical absolute path: {value!r}")
    if posixpath.basename(params["model_path"]) != BASE_REVISION:  # load_base and export refuse anything else, but only after GPU time
        raise ValueError(f"model_path must be the pinned snapshot directory {BASE_REVISION}")
    _positive_int(params, "limit", 1)
    if v3:
        hash_keys = (*V3_HASH_KEYS, *extra_hashes)
        for key in hash_keys:
            if not isinstance(params[key], str) or not HEX64.fullmatch(params[key]):
                raise ValueError(f"{key} must be 64 lowercase hex characters")
        for key in (*V3_PRIVATE_PATHS, "build_root", "out_root", *extra_paths):
            if posixpath.commonpath((params["repo_root"], params[key])) == params["repo_root"]:
                raise ValueError(f"{key} must be outside repo_root (private evidence is not repository content)")
        if params["seqs"] != "auto" or params["kv_cache_gib"] != "auto" or params["token_budget"] != 24000:
            raise ValueError("v3 requires seqs=auto, kv_cache_gib=auto and token_budget=24000")
    elif v2:
        reuse = params["reuse_t1_sha256"]
        if reuse is not None and (not isinstance(reuse, str) or not HEX64.fullmatch(reuse)):
            raise ValueError("reuse_t1_sha256 must be null or 64 lowercase hex characters")
    else:
        _positive_int(params, "pilot_limit", 0)
    _positive_int(params, "sweep_seconds", 1)
    _positive_int(params, "max_num_batched_tokens", 1)
    _positive_int(params, "token_budget", 1)
    kv = params["kv_cache_gib"]
    if kv != "auto" and (isinstance(kv, bool) or not isinstance(kv, (int, float)) or not math.isfinite(kv) or kv <= 0):
        raise ValueError('kv_cache_gib must be a positive finite number or "auto"')
    seqs = params["seqs"]
    if seqs != "auto" and (not isinstance(seqs, list) or not seqs or any(type(s) is not int or s < 1 for s in seqs)):
        raise ValueError('seqs must be a non-empty list of positive integers or "auto"')
    if params["gradient_checkpointing"] not in ("on", "off"):
        raise ValueError('gradient_checkpointing must be "on" or "off"')
    arms = params.get("arms", ["note"])
    if not isinstance(arms, list) or not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"arms must be a non-empty list of unique values from {ARMS}")
    timeouts = params["timeouts"]
    if not isinstance(timeouts, dict) or set(timeouts) != set(TIMEOUT_KINDS):
        raise ValueError(f"timeouts must have exactly the keys {list(TIMEOUT_KINDS)}")
    for kind in TIMEOUT_KINDS:
        if type(timeouts[kind]) is not int or timeouts[kind] < 1:
            raise ValueError(f"timeouts.{kind} must be an integer >= 1")
    return params


def _step(name: str, argv: list[str], expect: list[str], kind: str, timeouts: dict[str, int]) -> dict[str, Any]:
    return {"name": name, "argv": argv, "expect_files": expect, "timeout_seconds": timeouts[kind]}


def _plan_v2(p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out, model, build, commit = p["python"], p["out_root"], p["model_path"], p["build_root"], p["commit"]
    order = f"{p['repo_root']}/fine_tuning/error_note/item_order_v2.json"
    collect_cmd, train_cmd, chain_cmd = [py, "-m", COLLECT], [py, "-m", TRAIN], [py, "-m", CHAIN]
    collect_dir, limit, timeouts = f"{out}/collect", str(p["limit"]), p["timeouts"]
    run, export = f"{out}/train-note", f"{p['volume_root']}/eval-data/proof/export-note-v2"
    common = ["--build-root", build, "--order", order, "--model-path", model, "--kv-cache-gib", str(p["kv_cache_gib"]),
              "--max-num-batched-tokens", str(p["max_num_batched_tokens"]), "--engine", "vllm",
              "--limit", limit, "--output", collect_dir]
    steps = [_step("pin-order", [*chain_cmd, "pin", "--file", order, "--sha256", p["order_sha256"]], [], "collect", timeouts)]
    if p["reuse_t1_sha256"] is None:
        seqs = p["seqs"] if p["seqs"] == "auto" else ",".join(str(s) for s in p["seqs"])
        steps.append(_step("sweep", [*collect_cmd, "sweep", *common, "--seqs", seqs, "--seconds", str(p["sweep_seconds"])],
                           [f"{collect_dir}/sweep.json"], "collect", timeouts))
        concurrency = ["--max-num-seqs-from", f"{collect_dir}/sweep.json"]
    else:  # T1 copied from the v1 run (same ids, seeds and image budget): bound by its sha256, no sweep
        steps.append(_step("pin-t1", [*chain_cmd, "pin", "--file", f"{collect_dir}/t1.jsonl", "--sha256",
                                      p["reuse_t1_sha256"]], [], "collect", timeouts))
        concurrency = ["--max-num-seqs", "auto"]
    return steps + [
        _step("note", [*collect_cmd, "note", *common, *concurrency, "--note-version", "v2"],
              [f"{collect_dir}/note.jsonl", f"{collect_dir}/lessons.json"], "collect", timeouts),
        _step("resolve", [*collect_cmd, "resolve", *common, *concurrency],
              [f"{collect_dir}/receipt-resolve.json"], "collect", timeouts),
        _step("rows", [*collect_cmd, "rows", "--build-root", build, "--order", order, "--limit", limit,
                       "--tokenizer-path", model, "--output", collect_dir, "--rows-version", "v2"],
              [f"{collect_dir}/rows.jsonl"], "collect", timeouts),
        _step("train-note", [*train_cmd, "train", "--rows", f"{collect_dir}/rows.jsonl", "--build-root", build,
                             "--order", order, "--limit", limit, "--arm", "note", "--precision", "bf16",
                             "--model-path", model, "--output", run, "--code-commit", commit,
                             "--token-budget", str(p["token_budget"]), "--gradient-checkpointing",
                             p["gradient_checkpointing"], "--resume", "--protocol-version", "v2"],
              [f"{run}/checkpoint-final/metadata.json", f"{run}/result.json"], "train", timeouts),
        _step("reload-note", [*train_cmd, "reload", "--run", run, "--model-path", model, "--precision", "bf16",
                              "--protocol-version", "v2"], [f"{run}/reload.json"], "reload", timeouts),
        _step("export", [py, "-m", PROOF, "export", "--checkpoint", f"{run}/checkpoint-final",
                         "--binding", f"{run}/binding.json", "--base", model, "--output", export, "--code-commit", commit],
              [f"{export}/model_ref.json", f"{export}/model/export_manifest.json"], "export", timeouts),
    ]


def _plan_v3(stage: str, p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out, model, build, commit = p["python"], p["out_root"], p["model_path"], p["build_root"], p["commit"]
    order, timeouts, limit = p["order_path"], p["timeouts"], str(p["limit"])
    collect_dir = f"{out}/collect"
    collect_cmd, train_cmd, chain_cmd = [py, "-m", COLLECT], [py, "-m", TRAIN], [py, "-m", CHAIN]
    common = ["--build-root", build, "--order", order, "--limit", limit, "--output", collect_dir,
              "--answer-contract", TRAINING_ANSWER_CONTRACT, "--training-rules", p["training_rules_path"],
              "--training-rules-sha256", p["training_rules_sha256"], "--code-commit", commit]
    engine = ["--model-path", model, "--kv-cache-gib", "auto", "--max-num-batched-tokens",
              str(p["max_num_batched_tokens"]), "--engine", "vllm"]
    pins = [("order", order, p["order_sha256"]),
            ("training-rules", p["training_rules_path"], p["training_rules_sha256"]),
            ("source-provenance", p["source_provenance_path"], p["source_provenance_sha256"])]
    if stage == "note-v3-resolve":
        pins += [("t1", f"{collect_dir}/t1.jsonl", p["t1_sha256"]),
                 ("note", f"{collect_dir}/note.jsonl", p["note_sha256"]),
                 ("reviews", p["reviews_path"], p["reviews_sha256"])]
    elif stage == "note-v3-train":
        pins += [("rows", p["rows_path"], p["rows_sha256"]),
                 ("audit", p["audit_path"], p["audit_sha256"]),
                 ("solution-audit", p["solution_audit_path"], p["solution_audit_sha256"])]
    steps = [_step(f"pin-{name}", [*chain_cmd, "pin", "--file", path, "--sha256", digest], [], "collect", timeouts)
             for name, path, digest in pins]
    if stage == "note-v3-collect":
        return steps + [
            _step("sweep", [*collect_cmd, "sweep", *common, *engine, "--seqs", "auto",
                            "--seconds", str(p["sweep_seconds"])], [f"{collect_dir}/sweep.json"], "collect", timeouts),
            _step("note", [*collect_cmd, "note", *common, *engine, "--max-num-seqs-from", f"{collect_dir}/sweep.json",
                           "--note-version", "v2"], [f"{collect_dir}/note.jsonl", f"{collect_dir}/lessons.json"], "collect", timeouts),
        ]
    if stage == "note-v3-resolve":
        audit, rows = f"{collect_dir}/audit.json", f"{collect_dir}/rows.jsonl"
        source = ["--source-provenance", p["source_provenance_path"]]
        return steps + [
            _step("audit", [*collect_cmd, "audit", *common, *source, "--reviews", p["reviews_path"]],
                  [audit, f"{collect_dir}/lessons-v3.json"], "collect", timeouts),
            _step("resolve", [*collect_cmd, "resolve", *common, *engine, "--max-num-seqs", "auto", *source, "--audit", audit],
                  [f"{collect_dir}/receipt-resolve.json"], "collect", timeouts),
            _step("rows", [*collect_cmd, "rows", *common, "--tokenizer-path", model, "--rows-version", "v3", *source, "--audit", audit],
                  [rows, f"{collect_dir}/rows_summary.json"], "collect", timeouts),
        ]
    audit, rows, admission = p["audit_path"], p["rows_path"], f"{out}/admission.json"
    contracts = ["--audit", audit, "--sidecar", p["training_rules_path"], "--solution-audit", p["solution_audit_path"]]
    train_common = ["--rows", rows, "--build-root", build, "--order", order, "--limit", limit,
                    "--precision", "bf16", "--model-path", model, "--code-commit", commit, *contracts]
    steps.append(_step("admit", [*train_cmd, "admit", *train_common, "--output", admission], [admission], "train", timeouts))
    for arm in ("note", "no-note"):
        run, export = f"{out}/train-{arm}", f"{p['volume_root']}/eval-data/proof/export-{arm}-v3"
        steps += [
            _step(f"train-{arm}", [*train_cmd, "train", *train_common, "--arm", arm, "--output", run,
                                  "--protocol-version", "v3", "--admission", admission, "--token-budget", "24000",
                                  "--gradient-checkpointing", p["gradient_checkpointing"], "--resume"],
                  [f"{run}/checkpoint-final/metadata.json", f"{run}/result.json"], "train", timeouts),
            _step(f"reload-{arm}", [*train_cmd, "reload", "--run", run, "--model-path", model,
                                   "--precision", "bf16", "--protocol-version", "v3"], [f"{run}/reload.json"], "reload", timeouts),
            _step(f"export-{arm}", [py, "-m", PROOF, "export", "--checkpoint", f"{run}/checkpoint-final",
                                   "--binding", f"{run}/binding.json", "--base", model, "--output", export, "--code-commit", commit],
                  [f"{export}/model_ref.json", f"{export}/model/export_manifest.json"], "export", timeouts),
        ]
    return steps


def _check_single(stage: str, params: dict[str, Any]) -> dict[str, Any]:
    paths, hashes, kinds = SINGLE_EXTRA[stage]
    keys = SINGLE_KEYS | set(paths) | set(hashes)
    optional = {"execution_max_blocks", "warm_start"} if stage == "single-direct" else set()
    missing, unknown = sorted(keys - set(params)), sorted(set(params) - keys - optional)
    if missing or unknown:
        raise ValueError(f"params keys: missing {missing}, unknown {unknown}")
    if params["version"] != PARAMS_VERSION_SINGLE:
        raise ValueError(f"version must be {PARAMS_VERSION_SINGLE}")
    if not isinstance(params["commit"], str) or not HEX40.fullmatch(params["commit"]):
        raise ValueError("commit must be 40 lowercase hex characters")
    for key in (*SINGLE_PATHS, *paths):
        value = params[key]
        if (not isinstance(value, str) or not value.startswith("/") or value == "/"
                or posixpath.normpath(value) != value):
            raise ValueError(f"{key} must be a canonical absolute path: {value!r}")
    for key in (*SINGLE_HASHES, *hashes):
        if not isinstance(params[key], str) or not HEX64.fullmatch(params[key]):
            raise ValueError(f"{key} must be 64 lowercase hex characters")
    if posixpath.basename(params["model_path"]) != BASE_REVISION:
        raise ValueError(f"model_path must be the pinned snapshot directory {BASE_REVISION}")
    private = ("build_root", "out_root", *(f"{name}_path" for name in SINGLE_INPUTS), *paths)
    for key in private:
        if posixpath.commonpath((params["repo_root"], params[key])) == params["repo_root"]:
            raise ValueError(f"{key} must be outside repo_root (private evidence is not repository content)")
    # The campaign's own bounds (single_campaign.prepare): 1..16 sequences, positive batch tokens, budget >= 0.
    _positive_int(params, "max_num_seqs", 1)
    _positive_int(params, "max_num_batched_tokens", 1)
    _positive_int(params, "token_budget", 0)
    if params["max_num_seqs"] > 16:
        raise ValueError("max_num_seqs must be <= 16")
    if "execution_max_blocks" in params:  # reaches here only for single-direct (see the unknown-key check)
        blocks = params["execution_max_blocks"]
        if type(blocks) is not int or not PILOT_BLOCKS < blocks <= PILOT_BLOCKS + MAIN_BLOCKS:
            raise ValueError("execution_max_blocks must be an integer in 20..219")
    if "warm_start" in params:  # single-direct only; the campaign re-verifies all three files on every call
        warm = params["warm_start"]
        if not isinstance(warm, dict) or set(warm) != {*(f"{name}_{kind}" for name in WARM_FILES
                                                        for kind in ("path", "sha256")), "slot"}:
            raise ValueError("warm_start must map exactly plan/checkpoint/reload _path and _sha256 plus slot")
        if "execution_max_blocks" not in params:
            raise ValueError("warm_start requires execution_max_blocks")
        if type(warm["slot"]) is not int or not PILOT_BLOCKS < warm["slot"] < params["execution_max_blocks"]:
            raise ValueError("warm_start slot must be an integer with 19 < slot < execution_max_blocks")
        for name in WARM_FILES:
            path, sha = warm[f"{name}_path"], warm[f"{name}_sha256"]
            if (not isinstance(path, str) or not path.startswith("/") or path == "/" or posixpath.normpath(path) != path
                    or posixpath.commonpath((params["repo_root"], path)) == params["repo_root"]):
                raise ValueError(f"warm_start {name}_path must be a canonical absolute path outside repo_root: {path!r}")
            if not isinstance(sha, str) or not HEX64.fullmatch(sha):
                raise ValueError(f"warm_start {name}_sha256 must be 64 lowercase hex characters")
    engine = params["engine"]
    if (not isinstance(engine, dict) or set(engine) != SINGLE_ENGINE_KEYS or engine["name"] not in ("hf", "vllm")
            or isinstance(engine["kv_cache_gib"], bool) or not isinstance(engine["kv_cache_gib"], (int, float))
            or not engine["kv_cache_gib"] > 0 or not isinstance(engine["gpu_memory_utilization"], float)
            or not 0 < engine["gpu_memory_utilization"] < 1 or type(engine["sleep"]) is not bool):
        raise ValueError(f"engine must map exactly {sorted(SINGLE_ENGINE_KEYS)} (name hf|vllm, KV GiB > 0, "
                         "utilization in (0,1), sleep bool)")
    timeouts = params["timeouts"]
    if not isinstance(timeouts, dict) or set(timeouts) != set(kinds) or any(
            type(timeouts[k]) is not int or timeouts[k] < 1 for k in kinds):
        raise ValueError(f"timeouts must map exactly {list(kinds)} to integers >= 1")
    return params


def _plan_single(stage: str, p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out, model, commit, t = p["python"], p["out_root"], p["model_path"], p["commit"], p["timeouts"]
    total = p.get("execution_max_blocks", PILOT_BLOCKS + MAIN_BLOCKS)  # campaign length; only single-direct may set it
    campaign_dir, chain_cmd = f"{out}/campaign", [py, "-m", CHAIN]
    plan_file = f"{campaign_dir}/campaign-plan.json"
    warm = p.get("warm_start")  # single-direct only: every campaign call re-prepares and must reseal the same plan
    warm_args = [item for name in WARM_FILES for item in (f"--warm-start-{name}", warm[f"{name}_path"],
                                                          f"--warm-start-{name}-sha256", warm[f"{name}_sha256"])] if warm else []

    def step(name: str, argv: list[str], expect: list[str], kind: str) -> dict[str, Any]:
        return {"name": name, "argv": argv, "expect_files": expect, "timeout_seconds": t[kind]}

    def campaign(command: str, *extra: str) -> list[str]:
        return [py, "-m", CAMPAIGN, command, "--build-root", p["build_root"], "--model-path", model,
                "--output", campaign_dir,
                *(item for name in SINGLE_INPUTS for item in (f"--{name.replace('_', '-')}", p[f"{name}_path"],
                                                               f"--{name.replace('_', '-')}-sha256", p[f"{name}_sha256"])),
                *(item for name in ("pilot", "dev", "retention") for item in (f"--{name}-ids-sha256", p[f"{name}_ids_sha256"])),
                "--campaign-mode", "train", "--precision", "bf16", "--max-num-seqs", str(p["max_num_seqs"]),
                "--max-num-batched-tokens", str(p["max_num_batched_tokens"]), "--token-budget", str(p["token_budget"]),
                "--engine", p["engine"]["name"], "--kv-cache-gib", str(p["engine"]["kv_cache_gib"]),
                "--gpu-memory-utilization", str(p["engine"]["gpu_memory_utilization"]),
                "--vllm-sleep" if p["engine"]["sleep"] else "--no-vllm-sleep",
                "--execution-max-blocks", str(total), "--allow-model-run", *warm_args, *extra]

    def advance(blocks: int, *extra: str) -> list[str]:
        return campaign("advance", "--allow-update", "--allow-save", "--max-blocks", str(blocks), *extra)

    def reload(slot: int) -> dict[str, Any]:  # a new process restores weights, moments, LR, RNG and cursor
        receipt = f"{out}/reload-{slot}.json"
        return step(f"reload-{slot}", [*chain_cmd, "capture", "--output", receipt, "--", *campaign("reload", "--max-blocks", "1")],
                    [receipt], "reload")

    def export(slot: int, target: str) -> dict[str, Any]:
        return step(f"export-{slot}", [py, "-m", PROOF, "export", "--checkpoint", f"{campaign_dir}/checkpoint-{slot}",
                                       "--binding", plan_file, "--base", model, "--output", target, "--code-commit", commit],
                    [f"{target}/model_ref.json", f"{target}/model/export_manifest.json"], "export")

    paths, hashes, _ = SINGLE_EXTRA[stage]
    pins = [(name.replace("_", "-"), p[f"{name}_path"], p[f"{name}_sha256"]) for name in SINGLE_INPUTS]
    pins += [(key.removesuffix("_path").replace("_", "-"), p[key], p[key.removesuffix("_path") + "_sha256"])
             for key in paths if key.endswith("_path")]
    pins += [(f"warm-start-{name}", warm[f"{name}_path"], warm[f"{name}_sha256"]) for name in WARM_FILES] if warm else []
    steps = [step(f"pin-{name}", [*chain_cmd, "pin", "--file", path, "--sha256", sha], [], "cpu") for name, path, sha in pins]
    if stage == "single-direct" and warm:
        # v15 warm start v2: the sealed source checkpoint replaces checkpoint-0, the pilot and its reloads.
        return steps + [
            step("advance-main", advance(total - warm["slot"], "--continue-main", "--direct-main"),
                 [f"{campaign_dir}/checkpoint-{total}/metadata.json"], "advance_main"),
            reload(total),
        ]
    if stage == "single-direct":
        last = total
        # proof_eval exports only slot 19 or 219 at the training commit; a shorter campaign is exported elsewhere
        exported = [export(last, f"{p['volume_root']}/eval-data/proof/export-single-final")] if total == PILOT_BLOCKS + MAIN_BLOCKS else []
        return steps + [
            step("advance-1", advance(1), [f"{campaign_dir}/checkpoint-1/metadata.json"], "advance_first"),
            reload(1),
            step("advance-pilot", advance(PILOT_BLOCKS - 1), [f"{campaign_dir}/checkpoint-{PILOT_BLOCKS}/metadata.json"],
                 "advance_pilot"),
            reload(PILOT_BLOCKS),
            step("advance-main", advance(total - PILOT_BLOCKS, "--continue-main", "--direct-main"),
                 [f"{campaign_dir}/checkpoint-{last}/metadata.json"], "advance_main"),
            reload(last),
            *exported,
        ]
    if stage == "single-main":
        last = PILOT_BLOCKS + MAIN_BLOCKS
        return steps + [
            step("advance-main", advance(MAIN_BLOCKS, "--continue-main", "--pilot-gate", p["pilot_gate_path"],
                                         "--pilot-gate-sha256", p["pilot_gate_sha256"]),
                 [f"{campaign_dir}/checkpoint-{last}/metadata.json"], "advance_main"),
            reload(last),
            export(last, f"{p['volume_root']}/eval-data/proof/export-single-final"),
        ]
    dev, pilot_export = f"{out}/dev-eval", f"{out}/export-pilot"
    prereg, runs = f"{dev}/prereg.json", f"{dev}/runs"
    evaluate = [py, "-m", EVALUATE]
    steps += [
        step("advance-1", advance(1), [f"{campaign_dir}/checkpoint-1/metadata.json"], "advance_first"),
        reload(1),
        step("advance-pilot", advance(PILOT_BLOCKS - 1), [f"{campaign_dir}/checkpoint-{PILOT_BLOCKS}/metadata.json"],
             "advance_pilot"),
        reload(PILOT_BLOCKS),
        export(PILOT_BLOCKS, pilot_export),
        step("dev-prereg", [*evaluate, "preregister-hybrid", "--manifest", p["dev_manifest_path"],
                            "--vault-root", p["dev_vault_root"], "--campaign-plan", plan_file, "--f0-model-path", model,
                            "--hyb-export", pilot_export, "--dev-rules", p["dev_rules_path"], "--run-root", runs,
                            "--code-commit", commit, "--output", prereg], [prereg], "cpu"),
    ]
    for arm in ("F0", "HYB"):
        run = f"{runs}/{arm}"
        steps += [step(f"dev-{arm.lower()}", [*evaluate, "run", "--prereg", prereg, "--arm", arm, "--output", run],
                       [f"{run}/generation.json"], "evaluate"),
                  step(f"dev-{arm.lower()}-grade", [*evaluate, "grade", "--prereg", prereg, "--run", run],
                       [f"{run}/summary.json"], "cpu")]
    analysis = f"{dev}/analysis.json"
    return steps + [step("dev-analyze", [*evaluate, "analyze-hybrid", "--prereg", prereg, "--f0", f"{runs}/F0",
                                         "--hyb", f"{runs}/HYB", "--output", analysis], [analysis], "cpu")]


PARAMS_VERSION_CONTENT_V16 = "error-note-plan-params-content-v16"


def _check_content_v16(params: dict[str, Any]) -> dict[str, Any]:
    paths = ("python", "repo_root", "model_path", "out_root", "content_manifest_path", "functional_gate_path")
    keys = {"version", "commit", "content_manifest_sha256", "functional_gate_sha256", "total_blocks", "block_size",
            "timeouts", *paths}
    if set(params) - {"stage_a_blocks", "weights"} != keys or params["version"] != PARAMS_VERSION_CONTENT_V16:
        raise ValueError("v16 plan: exact versioned content campaign parameters required")
    if not isinstance(params["commit"], str) or not HEX40.fullmatch(params["commit"]):
        raise ValueError("v16 plan: full training commit required")
    for name in paths:
        value = params[name]
        if not isinstance(value, str) or not value.startswith("/") or value == "/" or posixpath.normpath(value) != value:
            raise ValueError(f"v16 plan: canonical absolute {name} required")
    for name in ("content_manifest_sha256", "functional_gate_sha256"):
        if not isinstance(params[name], str) or not HEX64.fullmatch(params[name]):
            raise ValueError(f"v16 plan: SHA256 required for {name}")
    for name in ("out_root", "content_manifest_path", "functional_gate_path"):
        if posixpath.commonpath((params["repo_root"], params[name])) == params["repo_root"]:
            raise ValueError("v16 plan: private content and output must be outside the repository")
    if posixpath.basename(params["model_path"]) != BASE_REVISION:
        raise ValueError("v16 plan: pinned F0 snapshot required")
    if (type(params["total_blocks"]) is not int or params["total_blocks"] < 1
            or type(params["block_size"]) is not int or not 1 <= params["block_size"] <= 16
            or not isinstance(params["timeouts"], dict) or set(params["timeouts"]) != {"cpu", "advance", "reload", "export"}
            or any(type(value) is not int or value < 1 for value in params["timeouts"].values())):
        raise ValueError("v16 plan: positive actual block count, bounded block size and timeouts required")
    stage_a = params.get("stage_a_blocks", 0)
    if type(stage_a) is not int or not 0 <= stage_a <= params["total_blocks"]:
        raise ValueError("v16 plan: stage A must be a prefix of the actual block schedule")
    if "weights" in params:
        from .protocol import CONTENT_WEIGHTS_V16

        weights = params["weights"]
        if (not isinstance(weights, dict) or set(weights) != set(CONTENT_WEIGHTS_V16)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in weights.values()) or sum(weights.values()) <= 0):
            raise ValueError("v16 plan: explicit finite nonnegative content weights required")
    return params


def _plan_content_v16(p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out, total = p["python"], p["out_root"], p["total_blocks"]
    campaign = f"{out}/campaign"
    common = [py, "-m", CAMPAIGN]
    args = ["--content-manifest", p["content_manifest_path"], "--content-manifest-sha256", p["content_manifest_sha256"],
            "--functional-gate", p["functional_gate_path"], "--functional-gate-sha256", p["functional_gate_sha256"],
            "--model-path", p["model_path"], "--output", campaign, "--campaign-mode", "train", "--precision", "bf16",
            "--block-size", str(p["block_size"]), "--content-total-blocks", str(total),
            "--stage-a-blocks", str(p.get("stage_a_blocks", 0)), "--allow-model-run"]
    if "weights" in p:
        args += ["--content-weights", json.dumps(p["weights"], sort_keys=True, separators=(",", ":"))]
    reload_file = f"{out}/reload-{total}.json"
    exported = f"{out}/export-final"

    def step(name: str, argv: list[str], files: list[str], kind: str) -> dict[str, Any]:
        return {"name": name, "argv": argv, "expect_files": files, "timeout_seconds": p["timeouts"][kind]}

    steps = [step("pin-" + name, [py, "-m", CHAIN, "pin", "--file", p[name + "_path"],
                                 "--sha256", p[name + "_sha256"]], [], "cpu")
             for name in ("content_manifest", "functional_gate")]
    return steps + [
        step("advance-content", [*common, "advance", *args, "--max-blocks", str(total), "--allow-update", "--allow-save"],
             [f"{campaign}/checkpoint-{total}/metadata.json"], "advance"),
        step("reload-content", [py, "-m", CHAIN, "capture", "--output", reload_file, "--",
                                *common, "reload", *args], [reload_file], "reload"),
        step("export-content", [py, "-m", PROOF, "export", "--checkpoint", f"{campaign}/checkpoint-{total}",
            "--binding", f"{campaign}/campaign-plan.json", "--base", p["model_path"], "--output", exported,
            "--code-commit", p["commit"]], [f"{exported}/model_ref.json", f"{exported}/model/export_manifest.json"], "export"),
    ]


PARAMS_VERSION_V16B = "error-note-plan-params-v16b"
V16B_PATHS = ("python", "repo_root", "model_path", "out_root", "data_root", "bundle_manifest_path",
               "bundle_archive_path", "guard_falsealarm_path", "b2_predictions_path", "hardware_path",
               "eval_data_root", "cv_script_path", "handoff_root", "run_dir")
V16B_HASHED = ("bundle_manifest", "guard_falsealarm", "b2_predictions", "cv_script")


def _check_v16b(params: dict[str, Any]) -> dict[str, Any]:
    keys = {"version", "commit", "timeouts", "hourly_usd", "gpu_type", *V16B_PATHS,
            *(name + "_sha256" for name in V16B_HASHED)}
    if set(params) != keys or params.get("version") != PARAMS_VERSION_V16B:
        raise ValueError("v16b plan: exact versioned parameters required")
    if not isinstance(params["commit"], str) or not HEX40.fullmatch(params["commit"]):
        raise ValueError("v16b plan: full training commit required")
    for name in V16B_PATHS:
        value = params[name]
        if not isinstance(value, str) or not value.startswith("/") or value == "/" or posixpath.normpath(value) != value:
            raise ValueError(f"v16b plan: canonical absolute {name} required")
    for name in V16B_HASHED:
        value = params[name + "_sha256"]
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ValueError(f"v16b plan: SHA256 required for {name}")
    for name in ("out_root", "data_root", "bundle_manifest_path", "bundle_archive_path", "guard_falsealarm_path",
                 "b2_predictions_path", "eval_data_root", "cv_script_path", "handoff_root", "run_dir"):
        if posixpath.commonpath((params["repo_root"], params[name])) == params["repo_root"]:
            raise ValueError(f"v16b plan: private {name} must be outside repository")
    if posixpath.commonpath((params["run_dir"], params["out_root"])) != params["run_dir"]:
        raise ValueError("v16b plan: out_root must be inside run_dir so recovery packs every result and checkpoint")
    if posixpath.basename(params["model_path"]) != BASE_REVISION:
        raise ValueError("v16b plan: pinned fresh-F0 snapshot required")
    if params["gpu_type"] not in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"):  # user 2026-11-15: cheapest A100
        raise ValueError("v16b plan: community A100 80GB (SXM or PCIe) only")
    rate = params["hourly_usd"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("v16b plan: explicit finite hourly price required")
    timeouts = params["timeouts"]
    if (not isinstance(timeouts, dict) or set(timeouts) != {"cpu", "reload", "export", "evaluate", "compass"}
            or any(type(t) is not int or t <= 0 for t in timeouts.values()) or timeouts["evaluate"] != 23400):
        raise ValueError("v16b plan: 6.5h evaluation watchdog, separate finite non-training timeouts required")
    return params


def _plan_v16b(p: dict[str, Any]) -> dict[str, Any]:
    from mmdl.runtime.contracts import SUBJECTS

    eval_ids = sorted(f"validation_{subject}_{i}" for subject in SUBJECTS for i in range(1, 31))
    py, out = p["python"], p["out_root"]
    prep, campaign, proof = f"{out}/prep", f"{out}/campaign", f"{out}/proof"
    export, prereg, ledger = f"{out}/export-final", f"{proof}/prereg.json", f"{proof}/ledger.jsonl"
    artifact, run_id = f"{out}/eval-artifacts", "v16b-ft-single"
    run = f"{artifact}/runs/{run_id}"
    campaign_cmd = [py, "-m", "fine_tuning.error_note.campaign_v16b"]
    common = ["--data-root", p["data_root"], "--prep-root", prep, "--model-path", p["model_path"],
              "--output", campaign, "--precision", "bf16", "--engine", "vllm"]
    steps = []

    def step(name: str, argv: list[str], files: list[str], kind: str | None = "cpu") -> None:
        item = {"name": name, "argv": argv, "expect_files": files}
        if kind is not None:
            item["timeout_seconds"] = p["timeouts"][kind]
        steps.append(item)

    for name in V16B_HASHED:
        step(f"pin-{name.replace('_', '-')}", [py, "-m", CHAIN, "pin", "--file", p[name + "_path"],
                                              "--sha256", p[name + "_sha256"]], [])
    bundle = [py, "-m", "fine_tuning.notebook_rl.pod.data_bundle"]
    step("stage-data", [*bundle, "stage", "--manifest", p["bundle_manifest_path"], "--manifest-sha256",
                       p["bundle_manifest_sha256"], "--archive", p["bundle_archive_path"]], [])
    step("verify-data", [*bundle, "verify", "--manifest", p["bundle_manifest_path"],
                        "--manifest-sha256", p["bundle_manifest_sha256"]], [])
    step("guard-falsealarm-gate", [py, f"{p['handoff_root']}/guard_pod_v16b.py", "--check-falsealarm",
                                  p["guard_falsealarm_path"]], [])
    for command in ("select", "generate", "rows"):
        argv = [py, "-m", "fine_tuning.error_note.prep_v16b", command, "--data-root", p["data_root"], "--output", prep]
        if command in ("generate", "rows"):  # rows loads the processor from the same snapshot
            argv += ["--model-path", p["model_path"]]
        if command == "generate":
            argv += ["--precision", "bf16", "--engine", "vllm", "--allow-model-run"]
        step(f"prep-{command}", argv, [], None if command == "generate" else "cpu")
    step("campaign-prepare", [*campaign_cmd, "prepare", *common], [f"{campaign}/campaign-plan.json"])
    step("campaign-advance", [*campaign_cmd, "advance", *common, "--allow-model-run", "--allow-update", "--allow-save"],
         [], None)
    step("require-complete", [*campaign_cmd, "require-complete", *common], [f"{campaign}/export-ready.json"])
    step("reload", [py, "-m", CHAIN, "capture", "--output", f"{out}/reload-final.json", "--",
                    *campaign_cmd, "reload", *common, "--allow-model-run"], [f"{out}/reload-final.json"], "reload")
    step("export", [*campaign_cmd, "export", *common, "--base", p["model_path"], "--export-output", export,
                    "--code-commit", p["commit"]], [f"{export}/model_ref.json"], "export")
    for kind in ("wrong", "rest", "all"):
        step(f"subset-{kind}", [py, "-m", PROOF, "make-subset", "--kind", kind, "--output", f"{proof}/{kind}.json",
                                "--b2-predictions", p["b2_predictions_path"]], [f"{proof}/{kind}.json"])
    step("preregister-v11", [py, "-m", PROOF, "preregister", "--format", "v11", "--output", prereg,
                             "--subset", f"{proof}/wrong.json", "--rest-subset", f"{proof}/rest.json",
                             "--all-subset", f"{proof}/all.json", "--b2-predictions", p["b2_predictions_path"],
                             "--ft-single-model-ref", f"{export}/model_ref.json"], [prereg])
    step("ledger-start", [py, "-m", PROOF, "ledger-start", "--prereg", prereg, "--arm", "FT-SINGLE",
                          "--variant", "mmmu-val-v8-b2-seed3407-v1", "--subset", "all", "--run-id", run_id,
                          "--ledger", ledger], [ledger])
    step("evaluate-900", ["/usr/bin/env", "MMDL_EVAL_SLOTS=auto", "VLLM_ENABLE_V1_MULTIPROCESSING=0", py, "-m", PROOF, "run-eval",
                          "--protocol", f"{p['repo_root']}/fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml",
                          "--hardware", p["hardware_path"], "--model-ref", f"{export}/model_ref.json",
                          "--model-path", f"{export}/model", "--data-root", p["eval_data_root"], "--artifact-root", artifact,
                          "--run-id", run_id, "--mode", "partial", "--run-role", "standalone",
                          "--no-download", "--require-commit", p["commit"],
                          *(arg for sid in eval_ids for arg in ("--sample-id", sid))],
         [f"{run}/summary.json"], "evaluate")
    step("v8-analyze", [py, "-m", PROOF, "analyze", "--prereg", prereg, "--ledger", ledger, "--runs", run,
                        "--ft-single-model-ref", f"{export}/model_ref.json", "--output", f"{proof}/v8"],
         [f"{proof}/v8.json"])
    step("compass-separate", [py, p["cv_script_path"], "--python", py, "--predictions", f"{run}/predictions.jsonl",
                              "--b2-predictions", p["b2_predictions_path"], "--output", f"{proof}/compass"],
         [f"{proof}/compass/dual-summary.json"], "compass")
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {"version": SPEC_VERSION, "meta": {"stage": "v16b", "job_stage": "single-main",
            "params_sha256": sha256_bytes(canonical.encode()), "generator": "fine_tuning.error_note.plan",
            "total_timeout_seconds": None, "training_elapsed_cutoff": None, "evaluation_watchdog_seconds": 23400,
            "funding": {"reserve_usd": 1.5, "post_train_hours": 5.65, "next_train_hours": 1,
                        "recharge_wait_seconds": 1800, "poll_seconds": 600, "hourly_usd": p["hourly_usd"]},
            "guard_stop_auto_evaluate": False, "gpu_type": p["gpu_type"], "cloud_execution": False,
            "after_run_closed": [py, "-m", "fine_tuning.notebook_rl.pod.collect", "pack", "--out-dir",
                                 f"{out}/recovery", "--name", "v16b-final", p["run_dir"]]},
            "steps": steps}



PARAMS_VERSION_V17 = "error-note-plan-params-v17"
V17_STANDING_EVALUATION_AUTHORIZATION = (
    "User 2026-11-17 standing run instruction: run v17 on RunPod and report the MMMU 900 V8 and V8+CompassVerifier "
    "scores when all results are in. Pre-authorizes EVAL_REVIEW_REQUIRED only; DEGRADED is never pre-authorized.")
V17_PATHS = tuple(name for name in V16B_PATHS if name != "handoff_root") + (
    "guard_manifest_path", "performance_requests_path", "performance_model_ref_path",
    "calibration_path", "functional_receipt_path",
)
V17_HASHED = V16B_HASHED + ("hardware", "guard_manifest", "performance_requests", "performance_model_ref")
V17_FUNDING_KEYS = {
    "balance_usd", "balance_observed_at", "first_bill_at", "storage_usd_per_hour",
    "recovery_reserve_usd", "recovery_seconds", "other_post_train_seconds",
    "next_train_seconds", "funding_deadline_utc",
}


def _check_v17(params: dict[str, Any], *, calibration: bool) -> dict[str, Any]:
    keys = {"version", "commit", "timeouts", "hourly_usd", "gpu_type", "funding",
            "long_tail_allowance_seconds", "long_tail_basis", *V17_PATHS,
            *(name + "_sha256" for name in V17_HASHED)}
    if not calibration:
        keys |= {"calibration_sha256", "functional_receipt_sha256"}
    if set(params) != keys or params.get("version") != PARAMS_VERSION_V17:
        raise ValueError("v17 plan: exact versioned parameters required")
    legacy = {name: params[name] for name in V16B_PATHS if name != "handoff_root"}
    legacy |= {name + "_sha256": params[name + "_sha256"] for name in V16B_HASHED}
    legacy |= {"version": PARAMS_VERSION_V16B, "commit": params["commit"],
               "gpu_type": params["gpu_type"], "hourly_usd": params["hourly_usd"],
               "handoff_root": params["run_dir"], "timeouts": {
                   "cpu": 1, "reload": 1, "export": 1, "compass": 1, "evaluate": 23400}}
    _check_v16b(legacy)
    for name in V17_PATHS:
        value = params[name]
        if (not isinstance(value, str) or not value.startswith("/") or value == "/"
                or posixpath.normpath(value) != value):
            raise ValueError(f"v17 plan: canonical absolute {name} required")
    for name in (*V17_HASHED, *(("calibration", "functional_receipt") if not calibration else ())):
        if not isinstance(params[name + "_sha256"], str) or not HEX64.fullmatch(params[name + "_sha256"]):
            raise ValueError(f"v17 plan: SHA256 required for {name}")
    for name in ("guard_manifest_path", "performance_requests_path", "performance_model_ref_path",
                 "calibration_path", "functional_receipt_path"):
        if posixpath.commonpath((params["repo_root"], params[name])) == params["repo_root"]:
            raise ValueError(f"v17 plan: private {name} must be outside repository")
    for name in ("calibration_path", "functional_receipt_path"):
        if posixpath.commonpath((params["run_dir"], params[name])) != params["run_dir"]:
            raise ValueError(f"v17 plan: {name} must be inside recovered run_dir")
    timeouts = params["timeouts"]
    if (not isinstance(timeouts, dict) or set(timeouts) != {"cpu", "reload", "export", "compass", "calibrate"}
            or any(type(value) is not int or value <= 0 for value in timeouts.values())):
        raise ValueError("v17 plan: finite non-training timeouts required; evaluation is calibrated")
    tail = params["long_tail_allowance_seconds"]
    if isinstance(tail, bool) or not isinstance(tail, (int, float)) or not math.isfinite(tail) or tail <= 0:
        raise ValueError("v17 plan: explicit positive long-tail allowance required")
    if not isinstance(params["long_tail_basis"], str) or not params["long_tail_basis"].strip():
        raise ValueError("v17 plan: explicit long-tail scenario basis required")
    funding = params["funding"]
    if not isinstance(funding, dict) or set(funding) != V17_FUNDING_KEYS:
        raise ValueError("v17 plan: actual balance, billing, storage and recovery fields required")
    from fine_tuning.notebook_rl.pod.common import parse_utc

    for name in ("balance_observed_at", "first_bill_at"):
        if not isinstance(funding[name], str):
            raise ValueError(f"v17 plan: UTC {name} required")
        parse_utc(funding[name])
    if funding["funding_deadline_utc"] is not None:
        if not isinstance(funding["funding_deadline_utc"], str):
            raise ValueError("v17 plan: optional funding deadline must be UTC text")
        parse_utc(funding["funding_deadline_utc"])
    for name in ("balance_usd", "storage_usd_per_hour", "recovery_reserve_usd", "recovery_seconds",
                 "other_post_train_seconds", "next_train_seconds"):
        value = funding[name]
        zero_allowed = name in ("storage_usd_per_hour", "recovery_reserve_usd", "other_post_train_seconds")
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (value == 0 and not zero_allowed)):
            raise ValueError(f"v17 plan: finite actual funding.{name} required")
    if not calibration:
        _v17_calibration(params)
    return params


def _v17_calibration(p: dict[str, Any]) -> dict[str, Any]:
    from .v17_eval import calibration_timeout

    loaded = {}
    for name in ("calibration", "functional_receipt"):
        raw = Path(p[name + "_path"]).read_bytes()
        if sha256_bytes(raw) != p[name + "_sha256"]:
            raise ValueError(f"v17 plan: {name} file SHA256 differs")
        loaded[name] = json.loads(raw)
    receipt = loaded["calibration"]
    calibration_timeout(receipt)
    if receipt.get("code_commit") != p["commit"]:
        raise ValueError("v17 plan: calibration source commit differs from the published main source")
    probe = receipt.get("probe", {})
    if (probe.get("long_tail_allowance_seconds") != p["long_tail_allowance_seconds"]
            or probe.get("long_tail_basis") != p["long_tail_basis"]
            or probe.get("requests_sha256") != p["performance_requests_sha256"]):
        raise ValueError("v17 plan: calibrated performance inputs differ from the sealed main parameters")
    functional = loaded["functional_receipt"]
    if (not isinstance(functional, dict) or functional.get("version") != "v17-bf16-preflight-v1"
            or functional.get("status") != "PASS" or functional.get("code_commit") != p["commit"]):
        raise ValueError("v17 plan: same-source BF16 functional receipt required before main sealing")
    return receipt


def verify_source_v17(repo_root: Path, commit: str, output: Path, *, repository: str = "MMDL_test") -> dict[str, Any]:
    """Check an already fetched publication locally; never fetch or publish code."""
    if not HEX40.fullmatch(commit):
        raise ValueError("v17 source: full published commit required")
    repo = repo_root.resolve()
    if output.resolve().is_relative_to(repo):
        raise ValueError("v17 source: receipt must be outside the clean clone")

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                              capture_output=True).stdout.strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("v17 source: exact clone root required")
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("v17 source: exact clean clone commit required")
    from fine_tuning.notebook_rl.pod.verify import ORIGINS, SUBMISSION_ORIGINS

    if repository not in {"MMDL_test", "MMDL"}:
        raise ValueError("v17 source: repository must be MMDL_test or MMDL")

    origin = git("remote", "get-url", "origin")
    refs = git("for-each-ref", "--format=%(refname)", "--contains", commit, "refs/remotes/origin")
    if origin not in (SUBMISSION_ORIGINS if repository == "MMDL" else ORIGINS) or not refs:
        raise ValueError(f"v17 source: fetched {repository} publication reference required")
    body = {"version": "v17-clean-published-source-v1", "status": "PASS", "code_commit": commit,
            "origin": origin, "publication_refs": sorted(refs.splitlines()), "network_checked": False}
    body["sha256"] = sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return body


def _plan_v17(p: dict[str, Any], *, calibration: bool) -> dict[str, Any]:
    from fine_tuning.notebook_rl.pod.common import parse_utc
    from .protocol import VERSION_V17, V17_IDENTIFIERS, V17_SPECIFICATION_SHA256
    from .v17_eval import calibration_timeout

    py, out = p["python"], p["out_root"]
    template = _plan_v16b(p | {"handoff_root": p["run_dir"],
                              "timeouts": p["timeouts"] | {"evaluate": 23400}})
    steps = template["steps"]
    prep_end = next(index for index, item in enumerate(steps) if item["name"] == "campaign-advance")
    selected = steps[:prep_end] if calibration else steps[prep_end:]
    source = f"{out}/source-{'calibration' if calibration else 'main'}.json"
    doctor = f"{out}/doctor.json"
    prefix = [{"name": "verify-published-source", "argv": [
        py, "-m", "fine_tuning.error_note.plan", "verify-source", "--repo-root", p["repo_root"],
        "--commit", p["commit"], "--output", source],
        "expect_files": [source], "timeout_seconds": p["timeouts"]["cpu"]}]
    if calibration:
        prefix += [
            {"name": "doctor", "argv": [py, "-m", "mmdl.runtime.environment", "--output", doctor,
                "--storage-path", p["run_dir"], "--required-gib", "50"],
             "expect_files": [doctor], "timeout_seconds": p["timeouts"]["cpu"]},
            {"name": "doctor-content-gate", "argv": [
                py, "-m", "fine_tuning.error_note.v17_eval", "doctor", "--input", doctor],
             "expect_files": [], "timeout_seconds": p["timeouts"]["cpu"]},
        ]
    hashes = V17_HASHED if calibration else (*V17_HASHED, "calibration", "functional_receipt")
    hash_steps = []
    for name in hashes:
        hash_steps.append({"name": f"pin-v17-{name.replace('_', '-')}",
            "argv": [py, "-m", CHAIN, "pin", "--file", p[name + "_path"], "--sha256", p[name + "_sha256"]],
            "expect_files": [], "timeout_seconds": p["timeouts"]["cpu"]})
    if calibration:
        prefix += [item for item in hash_steps if item["name"] == "pin-v17-bundle-manifest"]
        staging = [item for item in selected if item["name"] == "verify-data"]
        selected = (staging + [item for item in hash_steps if item["name"] != "pin-v17-bundle-manifest"]
                    + [item for item in selected if not item["name"].startswith("pin-")
                       and item["name"] not in ("stage-data", "verify-data")])
    else:
        prefix += hash_steps
    for item in selected:
        argv = item["argv"]
        if "fine_tuning.error_note.prep_v16b" in argv or "fine_tuning.error_note.campaign_v16b" in argv:
            argv += ["--version", VERSION_V17]
        if item["name"] == "guard-falsealarm-gate":
            item["argv"] = [py, "-m", "fine_tuning.error_note.guard_v16b",
                "--version", VERSION_V17, "--manifest", p["guard_manifest_path"],
                "--check-receipt", p["guard_falsealarm_path"]]
    campaign = f"{out}/campaign"
    campaign_cmd = [py, "-m", "fine_tuning.error_note.campaign_v16b"]
    common = ["--data-root", p["data_root"], "--prep-root", f"{out}/prep", "--model-path", p["model_path"],
              "--output", campaign, "--precision", "bf16", "--engine", "vllm", "--version", VERSION_V17,
              "--code-commit", p["commit"]]
    if calibration:
        selected += [
            {"name": "bf16-functional-calibration", "argv": [
                *campaign_cmd, "calibrate", *common, "--allow-model-run",
                "--receipt-output", p["functional_receipt_path"]],
             "expect_files": [p["functional_receipt_path"]], "timeout_seconds": p["timeouts"]["calibrate"]},
            {"name": "eval-throughput-calibration", "argv": ["/usr/bin/env", "MMDL_EVAL_SLOTS=8",
                "VLLM_ENABLE_V1_MULTIPROCESSING=0", py, "-m", "fine_tuning.error_note.v17_eval", "probe",
                "--requests", p["performance_requests_path"], "--model-path", p["model_path"],
                "--model-ref", p["performance_model_ref_path"], "--base-path", p["model_path"],
                "--hardware", p["hardware_path"], "--output", p["calibration_path"],
                "--long-tail-allowance-seconds", str(p["long_tail_allowance_seconds"]),
                "--long-tail-basis", p["long_tail_basis"], "--require-commit", p["commit"]],
             "expect_files": [p["calibration_path"]], "timeout_seconds": p["timeouts"]["calibrate"]},
        ]
        watchdog = None
    else:
        watchdog = calibration_timeout(_v17_calibration(p))
        prereg = f"{out}/proof/prereg.json"
        final_preflight = f"{out}/final-preflight.json"
        for item in selected:
            name, argv = item["name"], item["argv"]
            if "fine_tuning.error_note.campaign_v16b" in argv:
                if "--code-commit" not in argv:
                    argv += ["--code-commit", p["commit"]]
                argv += ["--calibration", p["calibration_path"], "--calibration-sha256", p["calibration_sha256"],
                         "--functional-receipt", p["functional_receipt_path"],
                         "--functional-receipt-sha256", p["functional_receipt_sha256"]]
            if name == "preregister-v11":
                item["name"] = "preregister-v17"
                argv[argv.index("--format") + 1] = "v12"
                argv += ["--calibration", p["calibration_path"]]
            if name == "ledger-start":
                argv[argv.index("--variant") + 1] = "mmmu-val-v8-b2-seed3407-v1"
                argv[argv.index("--run-id") + 1] = "v17-ft-single"
            if name == "evaluate-900":
                argv[argv.index("MMDL_EVAL_SLOTS=auto")] = "MMDL_EVAL_SLOTS=8"
                proof_start = argv.index(PROOF)
                normal = argv[proof_start + 2:]
                normal += ["--base-path", p["model_path"]]
                normal[normal.index("--run-id") + 1] = "v17-ft-single"
                funding_args = ["--recovery-reserve-seconds", str(p["funding"]["recovery_seconds"])]
                if p["funding"]["funding_deadline_utc"] is not None:
                    funding_args += ["--funding-deadline-epoch",
                        str(parse_utc(p["funding"]["funding_deadline_utc"]).timestamp())]
                item["argv"] = [*argv[:proof_start], "fine_tuning.error_note.v17_eval", "run",
                    "--prereg", prereg, "--ledger", f"{out}/proof/ledger.jsonl",
                    "--calibration", p["calibration_path"], "--preflight", final_preflight,
                    *funding_args, "--", *normal]
                item["timeout_seconds"] = watchdog
                item["expect_files"] = [f"{out}/eval-artifacts/runs/v17-ft-single/summary.json"]
            if name in ("v8-analyze", "compass-separate"):
                item["argv"] = [value.replace("/v16b-ft-single/", "/v17-ft-single/") for value in argv]
            if name == "v8-analyze":
                index = item["argv"].index("--runs") + 1
                item["argv"][index] = item["argv"][index].replace("/v16b-ft-single", "/v17-ft-single")
        # The external effect set is EXPOSURE_HISTORY_INCOMPLETE, so the admission is EVAL_REVIEW_REQUIRED unless
        # DEGRADED. The user's 2026-11-17 run instruction is recorded as the decision; DEGRADED still stops here.
        decision = f"{out}/evaluation-decision.json"
        for item in selected:
            if item["name"] in ("require-complete", "export"):
                item["argv"] += ["--evaluation-decision", decision]
        selected.insert(next(i for i, item in enumerate(selected) if item["name"] == "require-complete"), {
            "name": "evaluation-decision", "argv": [py, "-m", "fine_tuning.error_note.v17_eval", "decide",
                "--campaign-output", campaign, "--authorization", V17_STANDING_EVALUATION_AUTHORIZATION,
                "--output", decision],
            "expect_files": [decision], "timeout_seconds": p["timeouts"]["cpu"]})
        export_index = next(index for index, item in enumerate(selected) if item["name"] == "export")
        selected.insert(export_index + 1, {"name": "final-adapter-preflight",
            "argv": ["/usr/bin/env", "MMDL_EVAL_SLOTS=8", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
                     *campaign_cmd, "preflight", *common, "--allow-model-run",
                     "--export-output", f"{out}/export-final", "--preflight-output", final_preflight,
                     "--hardware", p["hardware_path"], "--eval-protocol",
                     f"{p['repo_root']}/fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml",
                     "--calibration", p["calibration_path"], "--calibration-sha256", p["calibration_sha256"],
                     "--functional-receipt", p["functional_receipt_path"],
                     "--functional-receipt-sha256", p["functional_receipt_sha256"],
                     "--evaluation-decision", decision],
            "expect_files": [final_preflight], "timeout_seconds": p["timeouts"]["calibrate"]})
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    funding = dict(p["funding"]) | {"hourly_usd": p["hourly_usd"], "recharge_wait_seconds": 1800,
        "poll_seconds": 600, "reserve_usd": p["funding"]["recovery_reserve_usd"],
        "evaluation_watchdog_seconds": watchdog, "calibration_sha256": p.get("calibration_sha256")}
    funding["post_train_hours"] = (None if watchdog is None else
        (watchdog + funding["other_post_train_seconds"] + funding["recovery_seconds"]) / 3600)
    funding["snapshot_only"] = True
    funding["execution_authority"] = "approved-job.financial-and-start.sh-budget-watchdog"
    if watchdog is not None:
        from .guard_v16b import funding_decision_v17

        decision = funding_decision_v17(funding["balance_usd"], p["hourly_usd"],
            parse_utc(funding["balance_observed_at"]).timestamp(), evaluation_seconds=watchdog,
            other_post_train_seconds=funding["other_post_train_seconds"], recovery_seconds=funding["recovery_seconds"],
            storage_usd_per_hour=funding["storage_usd_per_hour"], reserve_usd=funding["recovery_reserve_usd"],
            next_train_seconds=funding["next_train_seconds"])
        if decision["status"] != "CONTINUE":
            raise ValueError("v17 plan: actual funding snapshot cannot protect calibrated evaluation and recovery; recharge required")
        funding["initial_snapshot_decision"] = decision
    return {"version": SPEC_VERSION, "meta": {
        "stage": "v17-calibration" if calibration else "v17", "job_stage": "single-main",
        "params_sha256": sha256_bytes(canonical.encode()), "generator": "fine_tuning.error_note.plan",
        "protocol_version": VERSION_V17, "protocol_contract": deepcopy(V17_IDENTIFIERS),
        "specification_sha256": V17_SPECIFICATION_SHA256, "total_timeout_seconds": None,
        "training_elapsed_cutoff": None, "target_total_hours": [14, 17],
        "evaluation_watchdog_seconds": watchdog, "calibration_sha256": p.get("calibration_sha256"),
        "calibration_must_precede_main_seal": True, "initial_policy": "fresh_F0",
        "data_staging": "private-bundle-stage-before-approved-start.sh;chain-verifies-only",
        "training_counts": {"main": 3000, "prep_candidates": 512, "development": 200},
        "funding": funding, "guard_stop_auto_evaluate": False, "gpu_type": p["gpu_type"],
        "cloud_execution": False,
        "after_run_closed": [py, "-m", "fine_tuning.notebook_rl.pod.collect", "pack", "--out-dir",
                            f"{out}/recovery", "--name", "v17-final", p["run_dir"]]},
        "steps": prefix + selected}

PARAMS_VERSION_V18B_SPLIT = "error-note-v18b-split-params-v1"
PARAMS_VERSION_V18_SPLIT = "error-note-plan-params-v18-split-v1"
PARAMS_VERSION_V17_SPLIT = "error-note-plan-params-v17-split-v1"
V17_SPLIT_COMMON_PATHS = ("python", "repo_root", "model_path", "out_root", "run_dir")
V17_TRAIN_PATHS = ("data_root", "bundle_manifest_path", "bundle_archive_path",
                   "guard_falsealarm_path", "guard_manifest_path", "functional_receipt_path")
V17_TRAIN_HASHED = ("bundle_manifest", "guard_falsealarm", "guard_manifest")
V17_EVAL_PATHS = ("training_artifacts_path", "export_root", "hardware_path", "eval_data_root",
                  "b2_predictions_path", "b2_lengths_path", "performance_requests_path", "cv_script_path")
V17_EVAL_HASHED = ("training_artifacts", "hardware", "b2_predictions", "b2_lengths",
                   "performance_requests", "cv_script")
V17_SPLIT_FUNDING_KEYS = {"balance_usd", "balance_observed_at", "first_bill_at",
                         "storage_usd_per_hour", "recovery_reserve_usd", "recovery_seconds"}


def _check_v17_split(stage: str, p: dict[str, Any]) -> dict[str, Any]:
    v18b = stage.startswith("v18b-")
    v18 = v18b or stage in ("v18-train", "v18-eval")
    training = stage.endswith("-train")
    paths: tuple[str, ...] = V17_TRAIN_PATHS if training else V17_EVAL_PATHS
    hashed: tuple[str, ...] = V17_TRAIN_HASHED if training else V17_EVAL_HASHED
    extras = ({"training_estimate_seconds", "training_estimate_basis"} if training else
              {"long_tail_allowance_seconds", "long_tail_basis", "eval_engine_mode"})
    if v18 and training:
        paths = ("data_root", "cache_dir", "functional_receipt_path")
        hashed = ()
    if v18b:
        from .v18b_execution import validate_settings
        settings = validate_settings(p.get("execution_settings"))
        extras |= {"execution_settings"}
        if training:
            paths = ("data_root", "cache_dir", "backup_destination")
            if settings["reuse_f0_raw"]:
                paths += ("f0_reuse_source_path", "f0_reuse_evidence_path")
                hashed = ("f0_reuse_source", "f0_reuse_evidence")
        else:
            check_eval_v18b(p)
    on_prem = v18b and settings.get("execution_mode") == "on_prem"
    if on_prem and not training:
        raise ValueError("on-prem settings apply to public training only")
    if on_prem:
        paths += ("data_python",)
    if v18 and not training:
        extras = extras | {"evaluation_authorization"}
    keys = {"version", "commit", "timeouts", "hourly_usd", "gpu_type", "funding",
            *V17_SPLIT_COMMON_PATHS, *paths, *(name + "_sha256" for name in hashed), *extras}
    if on_prem:
        keys -= {"hourly_usd", "funding"}
    optional = {"guard_risk_acceptance"} if v18 and training else (set() if training or v18 else {"pf5_vision_exception"})
    if v18 and not training and (not isinstance(p.get("evaluation_authorization"), str) or not p["evaluation_authorization"].strip()):
        raise ValueError("v18 evaluation plan requires the operator's actual evaluation decision")
    if v18b:
        optional.add("generation_timing")
        if training and not on_prem:
            keys.add("backup_transfer_command")
        elif not training:
            optional.add("evaluation_partition_index")
        command = p.get("backup_transfer_command")
        if training and not on_prem and (not isinstance(command, list) or not command
                or not all(isinstance(arg, str) and arg for arg in command)):
            raise ValueError("v18b paid training requires an off-pod backup receiver command")
    expected_version = PARAMS_VERSION_V18B_SPLIT if v18b else PARAMS_VERSION_V18_SPLIT if v18 else PARAMS_VERSION_V17_SPLIT
    if not keys <= set(p) <= keys | optional or p.get("version") != expected_version:
        raise ValueError("v17 split plan: exact versioned stage parameters required")
    if "guard_risk_acceptance" in p and (not isinstance(p["guard_risk_acceptance"], str) or not p["guard_risk_acceptance"].strip()):
        raise ValueError("guard risk acceptance requires an explicit reason")
    if "pf5_vision_exception" in p and (not isinstance(p["pf5_vision_exception"], str)
                                         or not p["pf5_vision_exception"].strip()):
        raise ValueError("v17 split plan: pf5_vision_exception must be the recorded authorization text")
    if not isinstance(p["commit"], str) or not HEX40.fullmatch(p["commit"]):
        raise ValueError("v17 split plan: full published commit required")
    for name in (*V17_SPLIT_COMMON_PATHS, *paths):
        value = p[name]
        if (not isinstance(value, str) or not value.startswith("/") or value == "/"
                or posixpath.normpath(value) != value):
            raise ValueError(f"v17 split plan: canonical absolute {name} required")
    for name in hashed:
        if not isinstance(p[name + "_sha256"], str) or not HEX64.fullmatch(p[name + "_sha256"]):
            raise ValueError(f"v17 split plan: SHA256 required for {name}")
    for name in ("out_root", "run_dir", "model_path", *paths):
        if name not in ("cv_script_path", "data_python") and posixpath.commonpath((p["repo_root"], p[name])) == p["repo_root"]:
            raise ValueError(f"v17 split plan: private {name} must be outside repository")
    if posixpath.commonpath((p["run_dir"], p["out_root"])) != p["run_dir"]:
        raise ValueError("v17 split plan: out_root must be recovered inside run_dir")
    if training and not v18b and posixpath.commonpath((p["run_dir"], p["functional_receipt_path"])) != p["run_dir"]:
        raise ValueError("v17 split plan: functional receipt must be recovered inside run_dir")
    if posixpath.basename(p["model_path"]) != BASE_REVISION:
        raise ValueError("v17 split plan: pinned fresh-F0 snapshot required")
    if p["gpu_type"] not in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"):
        raise ValueError("v17 split plan: community A100 80GB required")
    if not on_prem and (isinstance(p["hourly_usd"], bool) or not isinstance(p["hourly_usd"], (int, float))
            or not math.isfinite(p["hourly_usd"]) or p["hourly_usd"] <= 0):
        raise ValueError("v17 split plan: finite positive hourly price required")
    timeouts = {"cpu", "calibrate", "reload", "export"} if training else {"cpu", "calibrate", "validate", "compass"}
    if (not isinstance(p["timeouts"], dict) or set(p["timeouts"]) != timeouts
            or any(type(value) is not int or value <= 0 for value in p["timeouts"].values())):
        raise ValueError("v17 split plan: finite technical stage timeouts required")
    if not training:
        from .v17_eval_engine import validation_timeout_seconds
        if p["eval_engine_mode"] not in ("cuda_graph", "eager"):
            raise ValueError("v17 split plan: eval_engine_mode must be cuda_graph or eager")
        mode = "auto" if p["eval_engine_mode"] == "cuda_graph" else "eager"
        min_time = validation_timeout_seconds(p["timeouts"]["calibrate"], mode,
            **({"training_protocol_version": "error-note-v18b"} if v18b else {}))
        if p["timeouts"]["validate"] < min_time:
            raise ValueError("v17 split plan: validate timeout must cover all engine attempts and fallback")
    duration = "training_estimate_seconds" if training else "long_tail_allowance_seconds"
    basis = "training_estimate_basis" if training else "long_tail_basis"
    if (isinstance(p[duration], bool) or not isinstance(p[duration], (int, float))
            or not math.isfinite(p[duration]) or p[duration] <= 0
            or not isinstance(p[basis], str) or not p[basis].strip()):
        raise ValueError(f"v17 split plan: positive {duration} and explicit {basis} required")
    if on_prem:
        if posixpath.commonpath((p["run_dir"], p["backup_destination"])) == p["run_dir"]:
            raise ValueError("on-prem backup destination must be outside run_dir")
        return p
    funding = p["funding"]
    if not isinstance(funding, dict) or set(funding) != V17_SPLIT_FUNDING_KEYS:
        raise ValueError("v17 split plan: actual balance snapshot and recovery fields required")
    from fine_tuning.notebook_rl.pod.common import parse_utc
    for name in ("balance_observed_at", "first_bill_at"):
        if not isinstance(funding[name], str):
            raise ValueError(f"v17 split plan: UTC {name} required")
        parse_utc(funding[name])
    for name in ("balance_usd", "storage_usd_per_hour", "recovery_reserve_usd", "recovery_seconds"):
        value = funding[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"v17 split plan: finite nonnegative funding.{name} required")
    if funding["recovery_reserve_usd"] != 1.5 or funding["recovery_seconds"] <= 0:
        raise ValueError("v17 split plan: actual-balance recovery reserve must be 1.5 USD")
    if v18b:
        lease = settings["billing"][p.get("evaluation_partition_index", 0)]
        if (parse_utc(settings["first_billed_at"]) != parse_utc(funding["first_bill_at"])
                or settings["recovery_reserve_seconds"] != funding["recovery_seconds"]
                or not math.isclose(lease["usd_per_hour"], p["hourly_usd"] + funding["storage_usd_per_hour"])):
            raise ValueError("v18b first billing time, recovery reserve or complete pod price differs between chains")
    return p


def _split_meta(stage: str, p: dict[str, Any], estimate: float | None) -> dict[str, Any]:
    from .v18b_generation_plan import generation_plan
    from .protocol import VERSION_V17, V17_IDENTIFIERS, V17_SPECIFICATION_SHA256
    if stage.startswith("v18b-"):
        from .protocol import VERSION_V18B, V18B_IDENTIFIERS, V18B_SPECIFICATION_SHA256
        version, contract, specification = VERSION_V18B, V18B_IDENTIFIERS, V18B_SPECIFICATION_SHA256
    elif stage.startswith("v18-"):
        from .protocol import VERSION_V18, V18_IDENTIFIERS, V18_SPECIFICATION_SHA256
        version, contract, specification = VERSION_V18, V18_IDENTIFIERS, V18_SPECIFICATION_SHA256
    else:
        version, contract, specification = VERSION_V17, V17_IDENTIFIERS, V17_SPECIFICATION_SHA256
    on_prem = p.get("execution_settings", {}).get("execution_mode") == "on_prem"
    funding = dict(p.get("funding", {}))
    estimated_cost = None if on_prem or estimate is None else (
        (estimate + funding["recovery_seconds"]) * (p["hourly_usd"] + funding["storage_usd_per_hour"]) / 3600)
    from .v18b_execution import enforce_limits
    completion = stage.startswith("v18b-") and not enforce_limits(p["execution_settings"])
    financial_mode = "on_prem" if on_prem else "measured_balance_reserve" if completion else "first_bill_22usd_and_recovery" if stage.startswith("v18b-") else "balance_reserve_only"
    if not on_prem:
        funding |= {"financial_mode": financial_mode, "hourly_usd": p["hourly_usd"],
                "estimated_cost_usd": estimated_cost, "estimated_cost_stops_execution": False,
                "initial_snapshot_decision": ("PENDING_ENGINE_VALIDATION" if estimated_cost is None else
                    "RECHARGE_REQUESTED" if funding["balance_usd"] < estimated_cost + 1.5 else "CONTINUE")}
    job_stage = ("v18-main" if stage in ("v18-train", "v18b-train") else "single-main") if stage.endswith("-train") else "eval-course"
    return {"stage": stage, "job_stage": job_stage,
            **({"generation_workload": generation_plan(stage, p)} if stage.startswith("v18b-") else {}),
            "params_sha256": sha256_bytes(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()),
            "generator": "fine_tuning.error_note.plan", "repo_root": p["repo_root"], "code_commit": p["commit"],
            "protocol_version": version, "protocol_contract": deepcopy(contract),
            "specification_sha256": specification, "total_timeout_seconds": None,
            "training_elapsed_cutoff": None, "evaluation_watchdog_seconds": None,
            "financial_mode": financial_mode, **({} if on_prem else {"funding": funding}), "gpu_type": p["gpu_type"],
            "cloud_execution": False, "guard_stop_auto_evaluate": False,
            **({"execution_settings": p["execution_settings"], "total_timeout_seconds": None if completion else p["execution_settings"]["total_timeout_seconds"],
                "planning_target_seconds": p["execution_settings"].get("total_timeout_seconds"),
                "training_elapsed_cutoff": p["execution_settings"]["training_cutoff_seconds"]} if stage.startswith("v18b-") else {})}


def _split_prefix(stage: str, p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out = p["python"], p["out_root"]
    source, doctor = f"{out}/source-{stage}.json", f"{out}/doctor.json"
    steps = [
        _step("verify-published-source", [py, "-m", "fine_tuning.error_note.plan", "verify-source",
              "--repo-root", p["repo_root"], "--commit", p["commit"], "--output", source,
              *(["--repository", "MMDL"] if p.get("execution_settings", {}).get("execution_mode") == "on_prem" else [])],
              [source], "cpu", p["timeouts"]),
        _step("doctor", [py, "-m", "mmdl.runtime.environment", "--output", doctor,
              "--storage-path", p["run_dir"], "--required-gib", "50"], [doctor], "cpu", p["timeouts"]),
        _step("doctor-content-gate", [py, "-m", "fine_tuning.error_note.v17_eval", "doctor", "--input", doctor],
              [], "cpu", p["timeouts"]),
    ]
    if stage.startswith("v18b-"):
        settings_path = f"{out}/execution-settings.json"
        steps.append(_step("seal-execution-settings", [py, "-m", "fine_tuning.error_note.v18b_execution",
            "--settings-json", json.dumps(p["execution_settings"], sort_keys=True), "--output", settings_path],
            [settings_path], "cpu", p["timeouts"]))
    return steps


def _plan_v17_train(p: dict[str, Any]) -> dict[str, Any]:
    from .protocol import VERSION_V17, VERSION_V18, VERSION_V18B
    v18b = p["version"] == PARAMS_VERSION_V18B_SPLIT
    v18 = v18b or p["version"] == PARAMS_VERSION_V18_SPLIT
    stage = "v18b-train" if v18b else "v18-train" if v18 else "v17-train"
    version = VERSION_V18B if v18b else VERSION_V18 if v18 else VERSION_V17
    py, out = p["python"], p["out_root"]
    prep, campaign, exported = f"{out}/prep", f"{out}/campaign", f"{out}/export-final"
    decision, artifact = f"{out}/evaluation-decision.json", f"{out}/training-artifacts.json"
    steps = _split_prefix(stage, p)

    def add(name: str, argv: list[str], files: list[str], kind: str | None = "cpu") -> None:
        item: dict[str, Any] = {"name": name, "argv": argv, "expect_files": files}
        if kind is not None:
            item["timeout_seconds"] = p["timeouts"][kind]
        steps.append(item)

    if v18:
        recipe = f"{p['repo_root']}/fine_tuning/error_note/v18_recipe.json"
        data_module = "fine_tuning.error_note.v18b_data" if v18b else "fine_tuning.error_note.v18_data"
        add("prepare-data", [p.get("data_python", py), "-m", data_module, "prepare",
            "--recipe", recipe, "--cache-dir", p["cache_dir"], "--output", p["data_root"],
            "--tokenizer", p["model_path"]],
            [f"{p['data_root']}/validation.json"], None)
        guard = f"{out}/guard/guard-falsealarm.json"
        add("guard-falsealarm-gate", [py, "-m", "fine_tuning.error_note.guard_v16b", "--version", version,
            "--materialize-data-root", p["data_root"], "--output", guard,
            *(["--risk-acceptance", p["guard_risk_acceptance"]] if p.get("guard_risk_acceptance") else [])],
            [guard])
    else:
        for name in V17_TRAIN_HASHED:
            add("pin-" + name.replace("_", "-"), [py, "-m", CHAIN, "pin", "--file", p[name + "_path"],
                "--sha256", p[name + "_sha256"]], [])
        add("verify-data", [py, "-m", "fine_tuning.notebook_rl.pod.data_bundle", "verify",
            "--manifest", p["bundle_manifest_path"], "--manifest-sha256", p["bundle_manifest_sha256"]], [])
        add("guard-falsealarm-gate", [py, "-m", "fine_tuning.error_note.guard_v16b", "--version", VERSION_V17,
            "--manifest", p["guard_manifest_path"], "--check-receipt", p["guard_falsealarm_path"]], [])
    for command in ("select", "generate", "rows"):
        argv = [py, "-m", "fine_tuning.error_note.prep_v16b", command, "--data-root", p["data_root"],
                "--output", prep, "--version", version]
        if command != "select":
            argv += ["--model-path", p["model_path"]]
        if command == "generate":
            argv += ["--precision", "bf16", "--engine", "vllm", "--allow-model-run"]
            if v18b and p["execution_settings"]["reuse_f0_raw"]:
                argv += ["--allow-f0-raw-reuse", "--f0-reuse-source", p["f0_reuse_source_path"],
                         "--f0-reuse-sha256", p["f0_reuse_source_sha256"], "--f0-reuse-evidence",
                         p["f0_reuse_evidence_path"], "--f0-reuse-evidence-sha256", p["f0_reuse_evidence_sha256"]]
        add("prep-" + command, argv, [], None if command == "generate" else "cpu")
    cmd = [py, "-m", "fine_tuning.error_note.campaign_v16b"]
    common = ["--data-root", p["data_root"], "--prep-root", prep, "--model-path", p["model_path"],
              "--output", campaign, "--precision", "bf16", "--engine", "vllm",
              "--version", version, "--code-commit", p["commit"], "--execution-phase", "training"]
    if v18b:
        common.extend(["--execution-settings", f"{out}/execution-settings.json",
                       "--backup-destination", p["backup_destination"]])
        if p.get("backup_transfer_command"):
            common.extend(["--backup-transfer-command", json.dumps(p["backup_transfer_command"])])
    add("campaign-prepare", [*cmd, "prepare", *common], [f"{campaign}/campaign-plan.json"])
    if not v18b:
        add("bf16-functional-calibration", [*cmd, "calibrate", *common, "--allow-model-run",
            "--receipt-output", p["functional_receipt_path"]], [p["functional_receipt_path"]], "calibrate")

    def bound(command: str, *extra: str) -> list[str]:
        if v18b:
            return [*cmd, command, *common, *extra]
        return [py, "-m", CHAIN, "with-sha256", "--file", p["functional_receipt_path"],
                "--token", "V17_FUNCTIONAL_SHA", "--", *cmd, command, *common,
                "--functional-receipt", p["functional_receipt_path"],
                "--functional-receipt-sha256", "V17_FUNCTIONAL_SHA", *extra]

    add("campaign-advance", bound("advance", "--allow-model-run", "--allow-update", "--allow-save"), [], None)
    decision_args = [] if v18 else ["--evaluation-decision", decision]
    if not v18:
        add("evaluation-decision", [py, "-m", "fine_tuning.error_note.v17_eval", "decide",
            "--campaign-output", campaign, "--authorization", V17_STANDING_EVALUATION_AUTHORIZATION,
            "--output", decision], [decision])
    add("require-complete", bound("require-complete", *decision_args), [f"{campaign}/export-ready.json"])
    reload = f"{out}/reload-final.json"
    add("reload", [py, "-m", CHAIN, "capture", "--output", reload, "--",
        *bound("reload", "--allow-model-run")], [reload], "reload")
    add("export", bound("export", "--base", p["model_path"], "--export-output", exported,
        *decision_args), [f"{exported}/model_ref.json"], "export")
    add("training-artifact-seal", bound("training-artifact-seal", "--export-output", exported,
        *decision_args, "--artifact-output", artifact), [artifact])
    meta = _split_meta(stage, p, p["training_estimate_seconds"])
    meta |= {"training_estimate_seconds": p["training_estimate_seconds"],
             "training_estimate_basis": p["training_estimate_basis"], "initial_policy": "fresh_F0",
             "training_counts": {"main": 3000, "prep_candidates": 512, "development": 200},
             "training_artifacts_path": artifact,
             "after_run_closed": [py, "-m", "fine_tuning.notebook_rl.pod.collect", "pack", "--out-dir",
                                   f"{out}/recovery", "--name", f"{stage}-final", p["run_dir"]]}
    return {"version": SPEC_VERSION, "meta": meta, "steps": steps}


def check_eval_v18b(p: dict[str, Any]) -> dict:
    from .v18b_execution import validate_settings
    settings = validate_settings(p["execution_settings"])
    index = p.get("evaluation_partition_index", 0)
    count = 2 if settings["evaluation_strategy"] == "two_pod" else 1
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("v18b evaluation: partition index must match the authorized strategy")
    return settings


def _plan_v17_eval(p: dict[str, Any]) -> dict[str, Any]:
    v18b = p["version"] == PARAMS_VERSION_V18B_SPLIT
    v18 = p["version"] == PARAMS_VERSION_V18_SPLIT or v18b
    stage = "v18b-eval" if v18b else "v18-eval" if v18 else "v17-eval"
    run_id = (p["execution_settings"]["run_id"] + (f"-part-{p.get('evaluation_partition_index', 0)}" if p["execution_settings"]["evaluation_strategy"] == "two_pod" else "")) if v18b else "v18-ft-single" if v18 else "v17-ft-single"
    py, out = p["python"], p["out_root"]
    verified, validation = f"{out}/training-artifacts-verified.json", f"{out}/engine-validation.json"
    steps = _split_prefix(stage, p)
    for name in V17_EVAL_HASHED:
        steps.append(_step("pin-" + name.replace("_", "-"), [py, "-m", CHAIN, "pin",
            "--file", p[name + "_path"], "--sha256", p[name + "_sha256"]], [], "cpu", p["timeouts"]))
    steps += [
        _step("verify-training-artifacts", [py, "-m", "fine_tuning.error_note.v17_artifacts", "verify",
            "--receipt", p["training_artifacts_path"], "--repo", p["repo_root"], "--commit", p["commit"],
            "--output", verified, *(["--expected-protocol", "error-note-v18b" if v18b else "error-note-v18", "--authorization", p["evaluation_authorization"]] if v18 else [])], [verified], "cpu", p["timeouts"]),
        _step("eval-engine-validation", ["/usr/bin/env", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
            py, "-m", "fine_tuning.error_note.v17_eval_engine", "validate", "--training-seal",
            p["training_artifacts_path"], *(["--artifact-verification", verified] if v18 else []), "--model-ref", f"{p['export_root']}/model_ref.json",
            "--base-path", p["model_path"], "--requests", p["performance_requests_path"],
            "--requests-sha256", p["performance_requests_sha256"], "--hardware", p["hardware_path"],
            "--eval-protocol", f"{p['repo_root']}/fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml",
            "--output", validation, "--code-commit", p["commit"], "--baseline-raw", p["b2_lengths_path"],
            "--baseline-raw-sha256", p["b2_lengths_sha256"], "--mode",
            "auto" if p["eval_engine_mode"] == "cuda_graph" else "eager",
            "--max-seconds", str(p["timeouts"]["calibrate"]),
            *(["--execution-settings", f"{out}/execution-settings.json"] if v18b else []),
            "--long-tail-allowance-seconds", str(p["long_tail_allowance_seconds"]),
            "--long-tail-basis", p["long_tail_basis"],
            *(["--pf5-vision-exception", p["pf5_vision_exception"]] if "pf5_vision_exception" in p else [])],
            [validation], "validate", p["timeouts"]),
        {"name": "finalize-and-run-evaluation", "argv": [py, "-m", "fine_tuning.error_note.plan",
            "finalize-evaluation", "--params-json", json.dumps(p, sort_keys=True, separators=(",", ":")),
            "--engine-validation", validation, "--output-dir", f"{out}/continuation", "--run"],
         "expect_files": [f"{out}/continuation/continuation-seal.json",
                          f"{out}/eval-artifacts/runs/{run_id}/summary.json",
                          *([] if v18b and p["execution_settings"]["evaluation_strategy"] == "two_pod" else [f"{out}/proof/compass/dual-summary.json"])]},
    ]
    meta = _split_meta(stage, p, None)
    meta |= {"engine_selection": "cuda_graph_then_eager_before_preregistration",
             "engine_validation_path": validation, "continuation_sealed_after_validation": True}
    return {"version": SPEC_VERSION, "meta": meta, "steps": steps}


def _evaluation_continuation(p: dict[str, Any], validation_path: Path) -> dict[str, Any]:
    from mmdl.runtime.contracts import SUBJECTS
    from fine_tuning.notebook_rl.pod.common import sha256_file
    from .v17_eval_engine import load_engine_validation
    checked = load_engine_validation(validation_path)
    watchdog, slots = checked["evaluation_watchdog_seconds"], checked["slots"]
    py, out, exported = p["python"], p["out_root"], p["export_root"]
    v18b = p["version"] == PARAMS_VERSION_V18B_SPLIT
    v18 = p["version"] == PARAMS_VERSION_V18_SPLIT or v18b
    stage = "v18b-eval" if v18b else "v18-eval" if v18 else "v17-eval"
    if v18 and (checked.get("evaluation_authorization", {}).get("evaluation_decision", {}).get("authorization")
                != p["evaluation_authorization"] or checked.get("artifact_verification", {}).get("path")
                != f"{out}/training-artifacts-verified.json"
                or checked.get("training_artifact_path") != p["training_artifacts_path"]):
        raise ValueError("v18 engine validation belongs to another evaluation authorization or training artifact")
    if v18 and slots != 9:
        raise ValueError("v18 evaluation requires measured capacity for exactly nine slots")
    proof, artifact = f"{out}/proof", f"{out}/eval-artifacts"
    run_id = (p["execution_settings"]["run_id"] + (f"-part-{p.get('evaluation_partition_index', 0)}" if p["execution_settings"]["evaluation_strategy"] == "two_pod" else "")) if v18b else "v18-ft-single" if v18 else "v17-ft-single"
    prereg, ledger, run = f"{proof}/prereg.json", f"{proof}/ledger.jsonl", f"{artifact}/runs/{run_id}"
    model_ref = f"{exported}/model_ref.json"
    binding = ["--calibration", str(validation_path), "--engine-validation", str(validation_path),
               "--training-artifacts", p["training_artifacts_path"]]
    steps = []

    def add(name: str, argv: list[str], files: list[str], kind: str = "cpu") -> None:
        steps.append(_step(name, argv, files, kind, p["timeouts"]))

    add("pin-engine-validation", [py, "-m", CHAIN, "pin", "--file", str(validation_path),
        "--sha256", sha256_file(validation_path)], [])
    for kind in ("wrong", "rest", "all"):
        add("subset-" + kind, [py, "-m", PROOF, "make-subset", "--kind", kind,
            "--output", f"{proof}/{kind}.json", "--b2-predictions", p["b2_predictions_path"]],
            [f"{proof}/{kind}.json"])
    add("preregister-v18" if v18 else "preregister-v17", [py, "-m", PROOF, "preregister", "--format", "v12", "--output", prereg,
        "--subset", f"{proof}/wrong.json", "--rest-subset", f"{proof}/rest.json", "--all-subset",
        f"{proof}/all.json", "--b2-predictions", p["b2_predictions_path"], "--ft-single-model-ref", model_ref,
        *binding], [prereg])
    all_ids = sorted(f"validation_{subject}_{i}" for subject in SUBJECTS for i in range(1, 31))
    subset_tag, ids = "all", all_ids
    if v18b:
        from .v18b_eval import partitions
        settings = check_eval_v18b(p)
        if checked.get("training_protocol_version") != "error-note-v18b" or checked.get("execution_settings") != settings:
            raise ValueError("v18b evaluation: engine receipt belongs to different settings/protocol")
        pieces = partitions(all_ids, settings)
        subset_tag = list(pieces)[p.get("evaluation_partition_index", 0)]
        ids = pieces[subset_tag]
    add("ledger-start", [py, "-m", PROOF, "ledger-start", "--prereg", prereg, "--arm", "FT-SINGLE",
        "--variant", "mmmu-val-v8-b2-seed3407-v1", "--subset", subset_tag, "--run-id", run_id,
        "--ledger", ledger], [ledger])
    evaluate = ["/usr/bin/env", f"MMDL_EVAL_SLOTS={slots}", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
        py, "-m", "fine_tuning.error_note.v17_eval", "run", "--prereg", prereg, "--ledger", ledger,
        *binding, "--preflight", str(validation_path),
        *(["--execution-settings", f"{out}/execution-settings.json", "--partition-index", str(p.get("evaluation_partition_index", 0))] if v18b else []), "--",
        "--protocol", f"{p['repo_root']}/fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml",
        "--hardware", p["hardware_path"], "--model-ref", model_ref, "--model-path", f"{exported}/model",
        "--base-path", p["model_path"], "--data-root", p["eval_data_root"], "--artifact-root", artifact,
        "--run-id", run_id, "--mode", "partial", "--run-role", "standalone", "--no-download",
        "--require-commit", p["commit"], *(arg for sid in ids for arg in ("--sample-id", sid))]
    step: dict[str, Any] = {"name": "evaluate-900", "argv": evaluate, "expect_files": [f"{run}/summary.json"]}
    if not v18b or settings.get("limit_action", "stop") != "warn":
        step["timeout_seconds"] = watchdog
    steps.append(step)
    add("v8-analyze", [py, "-m", PROOF, "analyze", "--prereg", prereg, "--ledger", ledger, "--runs", run,
        "--ft-single-model-ref", model_ref, "--output", f"{proof}/v8"], [f"{proof}/v8.json"])
    if not (v18b and settings["evaluation_strategy"] == "two_pod"):
        add("compass-separate", [py, p["cv_script_path"], "--python", py, "--predictions",
        f"{run}/predictions.jsonl", "--b2-predictions", p["b2_predictions_path"], "--output",
        f"{proof}/compass"], [f"{proof}/compass/dual-summary.json"], "compass")
    meta = _split_meta(stage, p, checked["estimated_evaluation_seconds"])
    if v18b and settings["evaluation_strategy"] == "two_pod":
        meta["partition"] = {"index": p.get("evaluation_partition_index", 0), "subset": subset_tag, "count": len(ids),
                             "full_900_requires": "v18b_eval merge of both fixed partitions; supplementary after merge only"}
    meta |= {"continuation": True, "evaluation_watchdog_seconds": watchdog,
             "estimated_evaluation_seconds": checked["estimated_evaluation_seconds"],
             "eval_engine_mode": checked["selected_mode"], "eval_concurrency": slots,
             "engine_validation_sha256": sha256_file(validation_path),
             "training_artifacts_sha256": p["training_artifacts_sha256"],
             "request_order": "sha256(v17-eval-order:<sample_id>)",
             "after_run_closed": [py, "-m", "fine_tuning.notebook_rl.pod.collect", "pack", "--out-dir",
                                   f"{out}/recovery", "--name", f"{stage}-final", p["run_dir"]]}
    return {"version": SPEC_VERSION, "meta": meta, "steps": steps}


def finalize_evaluation(params: dict[str, Any], validation_path: Path, output_dir: Path,
                        *, execute: bool = False) -> dict[str, Any]:
    from fine_tuning.notebook_rl.pod.common import sha256_file, write_json
    from fine_tuning.notebook_rl.pod.chain import check_spec, run
    from .v17_artifacts import verify_training_artifacts
    stage = "v18b-eval" if params.get("version") == PARAMS_VERSION_V18B_SPLIT else "v18-eval" if params.get("version") == PARAMS_VERSION_V18_SPLIT else "v17-eval"
    p = _check_v17_split(stage, params)
    if output_dir != Path(p["out_root"]) / "continuation":
        raise ValueError("v17 evaluation continuation must be recovered inside out_root/continuation")
    if validation_path != Path(p["out_root"]) / "engine-validation.json":
        raise ValueError("v17 evaluation continuation requires this job's engine validation")
    if sha256_file(p["training_artifacts_path"]) != p["training_artifacts_sha256"]:
        raise ValueError("v17 evaluation continuation: training artifact receipt changed")
    artifacts = verify_training_artifacts(Path(p["training_artifacts_path"]))
    if stage in ("v18-eval", "v18b-eval") and artifacts.get("training_protocol_version") != ("error-note-v18b" if stage == "v18b-eval" else "error-note-v18"):
        raise ValueError("v18 evaluation requires a v18 training seal")
    spec = check_spec(_evaluation_continuation(p, validation_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path, seal_path = output_dir / "spec.json", output_dir / "continuation-seal.json"
    for path, value in ((spec_path, spec), (seal_path, {
            "version": "v17-evaluation-continuation-seal-v1", "params_sha256": spec["meta"]["params_sha256"],
            "engine_validation_sha256": sha256_file(validation_path),
            "training_artifacts_sha256": p["training_artifacts_sha256"],
            "spec_sha256": sha256_bytes((json.dumps(spec, indent=2, sort_keys=True) + "\n").encode())})):
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError("v17 evaluation continuation: refusing to replace a different sealed artifact")
        else:
            write_json(path, value)
    result = {"spec": str(spec_path), "spec_sha256": sha256_file(spec_path), "seal": str(seal_path)}
    if execute:
        result["returncode"] = run(spec_path, result["spec_sha256"], log_name="eval_continuation_log.jsonl")
    return result


def finalize_partition_evaluation(params: dict[str, Any], shards: list[dict], output_dir: Path,
                                  *, execute: bool = False) -> dict:
    """One reviewable CPU merge then supplementary-scoring chain; never starts a pod."""
    from fine_tuning.notebook_rl.pod.chain import check_spec, run
    from fine_tuning.notebook_rl.pod.common import write_json, sha256_file

    p = _check_v17_split("v18b-eval", params)
    if p["execution_settings"]["evaluation_strategy"] != "two_pod" or len(shards) != 2:
        raise ValueError("v18b merge plan: explicitly authorized two-pod evaluation required")
    if output_dir != Path(p["out_root"]) / "partition-finalization":
        raise ValueError("v18b merge plan: finalization must be recovered inside out_root")
    py, out = p["python"], p["out_root"]
    merged = f"{out}/full-900"
    steps = []
    for name in ("b2_predictions", "cv_script"):
        steps.append(_step("pin-" + name, [py, "-m", CHAIN, "pin", "--file", p[name + "_path"],
            "--sha256", p[name + "_sha256"]], [], "cpu", p["timeouts"]))
    command = [py, "-m", "fine_tuning.error_note.v18b_eval", "merge"]
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != {"prereg", "ledger", "run_dir"}:
            raise ValueError("v18b merge plan: exact original prereg, ledger and run_dir required")
        for key, flag in (("prereg", "--prereg"), ("ledger", "--ledger"), ("run_dir", "--run-dir")):
            path = shard[key]
            if not isinstance(path, str) or not path.startswith("/") or posixpath.normpath(path) != path:
                raise ValueError("v18b merge plan: canonical absolute inputs required")
            command += [flag, path]
        if posixpath.basename(shard["run_dir"]) != p["execution_settings"]["run_id"] + f"-part-{index}":
            raise ValueError("v18b merge plan: fixed partition run IDs required")
    command += ["--output", merged]
    steps.append(_step("merge-full-900-v8", command, [f"{merged}/summary.json", f"{merged}/predictions.jsonl"], "cpu", p["timeouts"]))
    steps.append(_step("compass-separate", [py, p["cv_script_path"], "--python", py, "--predictions",
        f"{merged}/predictions.jsonl", "--b2-predictions", p["b2_predictions_path"], "--output", f"{out}/proof/compass"],
        [f"{out}/proof/compass/dual-summary.json"], "compass", p["timeouts"]))
    meta = _split_meta("v18b-eval", p, None) | {"partition_finalization": True}
    spec = check_spec({"version": SPEC_VERSION, "meta": meta, "steps": steps})
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "spec.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("v18b merge plan: refusing to replace the original finalization")
    if not path.exists():
        write_json(path, spec)
    result = {"spec": str(path), "spec_sha256": sha256_file(path), "full_900": merged}
    if execute:
        result["returncode"] = run(path, result["spec_sha256"], log_name="partition-finalization.jsonl")
    return result


def _resume_steps(spec: dict) -> dict:
    from fine_tuning.notebook_rl.pod.chain import check_spec

    steps = list(spec["steps"])
    first = next(i for i, step in enumerate(steps) if step["name"] == "evaluate-900")
    resume = deepcopy(steps[first:])
    argv = resume[0]["argv"]
    module = argv.index("fine_tuning.error_note.v17_eval")
    if argv[module + 1] != "run" or spec["meta"]["stage"] not in ("v18-eval", "v18b-eval"):
        raise ValueError("expected v18 sealed evaluation run")
    argv[module + 1] = "resume"
    return check_spec({**spec, "steps": resume})


def resume_evaluation(params: dict, *, execute: bool = False) -> dict:
    """Resume the registered run using the exact original commands and a separate attempt log."""
    import uuid
    from fine_tuning.notebook_rl.pod.chain import run

    stage = "v18b-eval" if params.get("version") == PARAMS_VERSION_V18B_SPLIT else "v18-eval"
    _check_v17_split(stage, params)
    root = Path(params["out_root"])
    directory = root / "continuation"
    if not all((directory / name).is_file() for name in ("spec.json", "continuation-seal.json")):
        raise ValueError("registered evaluation continuation is missing")
    original = finalize_evaluation(params, root / "engine-validation.json", directory)
    spec = _resume_steps(json.loads(Path(original["spec"]).read_text()))
    if not (root / "proof" / "ledger.jsonl").is_file():
        raise ValueError("original evaluation ledger is required")
    result = {"original_spec_sha256": original["spec_sha256"], "first_step": "resume",
              "downstream_steps": [s["name"] for s in spec["steps"][1:]], "execute": execute}
    if execute:
        attempt = uuid.uuid4().hex
        path = directory / f"resume-{attempt}.json"
        raw = (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode()
        with path.open("xb") as stream:
            stream.write(raw)
        result["returncode"] = run(path, sha256_bytes(raw), log_name=f"eval-resume-{attempt}.jsonl")
    return result


def plan(stage: str, params: dict[str, Any]) -> dict[str, Any]:
    p = _check(stage, params)
    if stage in ("v17-train", "v17-eval", "v18-train", "v18-eval", "v18b-train", "v18b-eval"):
        return _plan_v17_train(p) if stage.endswith("-train") else _plan_v17_eval(p)
    if stage in ("v17-calibration", "v17"):
        return _plan_v17(p, calibration=stage == "v17-calibration")
    if stage == "v16b":
        return _plan_v16b(p)
    if stage == "content-v16":
        return _spec(stage, p, _plan_content_v16(p))
    if stage in SINGLE_STAGES:
        return _spec(stage, p, _plan_single(stage, p))
    if stage in V3_STAGES:
        return _spec(stage, p, _plan_v3(stage, p))
    if stage == "note-v2":
        return _spec(stage, p, _plan_v2(p))
    py, out, model, build, commit = p["python"], p["out_root"], p["model_path"], p["build_root"], p["commit"]
    order = f"{p['repo_root']}/fine_tuning/error_note/item_order.json"
    collect_cmd, train_cmd, chain_cmd = [py, "-m", COLLECT], [py, "-m", TRAIN], [py, "-m", CHAIN]
    collect_dir, limit, timeouts = f"{out}/collect", str(p["limit"]), p["timeouts"]
    seqs = p["seqs"] if p["seqs"] == "auto" else ",".join(str(s) for s in p["seqs"])

    def common(model_path: str) -> list[str]:
        return ["--build-root", build, "--order", order, "--model-path", model_path,
                "--kv-cache-gib", str(p["kv_cache_gib"]),
                "--max-num-batched-tokens", str(p["max_num_batched_tokens"]), "--engine", "vllm"]

    def train_run(rows: str, n: str, arm: str, output: str) -> list[str]:
        return [*train_cmd, "train", "--rows", rows, "--build-root", build, "--order", order, "--limit", n,
                "--arm", arm, "--precision", "bf16", "--model-path", model, "--output", output,
                "--code-commit", commit, "--token-budget", str(p["token_budget"]),
                "--gradient-checkpointing", p["gradient_checkpointing"], "--resume"]

    def reload_run(run: str) -> list[str]:
        return [*train_cmd, "reload", "--run", run, "--model-path", model, "--precision", "bf16"]

    steps = [_step("pin-order", [*chain_cmd, "pin", "--file", order, "--sha256", p["order_sha256"]],
                   [], "collect", timeouts)]
    if p["pilot_limit"] > 0:
        pilot_limit, pilot_rows = str(p["pilot_limit"]), f"{out}/pilot-rows"
        pilot_train, pilot_export = f"{out}/pilot-train", f"{out}/pilot-export"
        steps += [
            _step("pilot-solve", [*collect_cmd, "solve", *common(model), "--limit", pilot_limit,
                                  "--output", collect_dir, "--max-num-seqs", "16"],
                  [f"{collect_dir}/t1.jsonl"], "collect", timeouts),
            _step("pilot-note", [*collect_cmd, "note", *common(model), "--limit", pilot_limit,
                                 "--output", collect_dir, "--max-num-seqs", "16"],
                  [f"{collect_dir}/note.jsonl"], "collect", timeouts),
            _step("pilot-rows", [*collect_cmd, "rows", "--build-root", build, "--order", order,
                                 "--limit", pilot_limit, "--tokenizer-path", model,
                                 "--output", collect_dir, "--rows-output", pilot_rows],
                  [f"{pilot_rows}/rows.jsonl"], "collect", timeouts),
            _step("pilot-train", train_run(f"{pilot_rows}/rows.jsonl", pilot_limit, "note", pilot_train),
                  [f"{pilot_train}/checkpoint-final/metadata.json", f"{pilot_train}/result.json"],
                  "train", timeouts),
            _step("pilot-reload", reload_run(pilot_train), [f"{pilot_train}/reload.json"], "reload", timeouts),
            _step("pilot-export", [py, "-m", PROOF, "export", "--checkpoint", f"{pilot_train}/checkpoint-final",
                                   "--binding", f"{pilot_train}/binding.json", "--base", model,
                                   "--output", pilot_export, "--code-commit", commit],
                  [f"{pilot_export}/model_ref.json", f"{pilot_export}/model/export_manifest.json"],
                  "export", timeouts),
            _step("pilot-ft-check", [*collect_cmd, "solve", *common(f"{pilot_export}/model"), "--limit", "2",
                                     "--output", f"{out}/pilot-ft-check", "--max-num-seqs", "4"],
                  [f"{out}/pilot-ft-check/t1.jsonl"], "collect", timeouts),
        ]
    steps += [
        _step("sweep", [*collect_cmd, "sweep", *common(model), "--limit", limit, "--output", collect_dir,
                        "--seqs", seqs, "--seconds", str(p["sweep_seconds"])],
              [f"{collect_dir}/sweep.json"], "collect", timeouts),
        _step("note", [*collect_cmd, "note", *common(model), "--limit", limit, "--output", collect_dir,
                       "--max-num-seqs-from", f"{collect_dir}/sweep.json"],
              [f"{collect_dir}/note.jsonl"], "collect", timeouts),
        _step("rows", [*collect_cmd, "rows", "--build-root", build, "--order", order, "--limit", limit,
                       "--tokenizer-path", model, "--output", collect_dir],
              [f"{collect_dir}/rows.jsonl"], "collect", timeouts),
    ]
    for arm in p["arms"]:
        run = f"{out}/train-{arm}"
        steps.append(_step(f"train-{arm}", train_run(f"{collect_dir}/rows.jsonl", limit, arm, run),
                           [f"{run}/checkpoint-final/metadata.json", f"{run}/result.json"], "train", timeouts))
        steps.append(_step(f"reload-{arm}", reload_run(run), [f"{run}/reload.json"], "reload", timeouts))
    return _spec(stage, p, steps)


def _spec(stage: str, p: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {"version": SPEC_VERSION,
            # single-direct runs under the existing single-main job stage (pod/job.py stays byte-identical)
            "meta": {"stage": stage, "job_stage": "single-main" if stage == "single-direct" else stage,
                     "params_sha256": sha256_bytes(canonical.encode()),
                     "generator": "fine_tuning.error_note.plan",
                     "total_timeout_seconds": sum(step["timeout_seconds"] for step in steps)},
            "steps": steps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="write the chain spec of one error-note stage")
    plan_parser.add_argument("--stage", choices=STAGES, required=True)
    plan_parser.add_argument("--params", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    source_parser = sub.add_parser("verify-source", help="verify an existing clean published v17 clone")
    source_parser.add_argument("--repo-root", type=Path, required=True)
    source_parser.add_argument("--commit", required=True)
    source_parser.add_argument("--output", type=Path, required=True)
    source_parser.add_argument("--repository", choices=("MMDL_test", "MMDL"), default="MMDL_test")
    finalizer = sub.add_parser("finalize-evaluation", help="seal evaluation after actual engine validation")
    finalizer.add_argument("--params-json", required=True)
    finalizer.add_argument("--engine-validation", type=Path, required=True)
    finalizer.add_argument("--output-dir", type=Path, required=True)
    finalizer.add_argument("--run", action="store_true")
    partitions = sub.add_parser("finalize-partitions", help="merge two fixed v18b partitions then score separately")
    partitions.add_argument("--params-json", required=True)
    partitions.add_argument("--shards-json", required=True)
    partitions.add_argument("--output-dir", type=Path, required=True)
    partitions.add_argument("--run", action="store_true")
    resumer = sub.add_parser("resume-evaluation", help="resume an already registered v18 evaluation")
    resumer.add_argument("--params", type=Path, required=True)
    resumer.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-source":
            receipt = verify_source_v17(args.repo_root, args.commit, args.output, repository=args.repository)
            print(receipt["sha256"])
            return 0
        if args.command == "resume-evaluation":
            result = resume_evaluation(json.loads(args.params.read_text()), execute=args.run)
            print(json.dumps(result, sort_keys=True))
            return result.get("returncode", 0)
        if args.command == "finalize-evaluation":
            result = finalize_evaluation(json.loads(args.params_json), args.engine_validation,
                                         args.output_dir, execute=args.run)
            print(json.dumps(result, sort_keys=True))
            return result.get("returncode", 0)
        if args.command == "finalize-partitions":
            result = finalize_partition_evaluation(json.loads(args.params_json), json.loads(args.shards_json),
                                                    args.output_dir, execute=args.run)
            print(json.dumps(result, sort_keys=True))
            return result.get("returncode", 0)
        text = json.dumps(plan(args.stage, json.loads(args.params.read_text(encoding="utf-8"))),
                          indent=2, sort_keys=True) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(sha256_bytes(text.encode()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
