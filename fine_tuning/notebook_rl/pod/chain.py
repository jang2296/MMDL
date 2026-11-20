"""A fixed multi-step chain as one Pod job's `main` argv: plan the steps from a params file, run them.

A Pod job has one `smoke` argv (hard 300 s, `pod.gpu_smoke`) and one `main` argv; the proof needs
many commands. `plan` writes the step list for a stage deterministically; the job's `main` is
`chain run --spec STEPS.json --spec-sha256 HEX`, so the job approval (which covers the argv) binds
the whole list. Nothing here starts a Pod, changes a limit or retries anything.

CLI:
  chain plan --stage {stage1,pilot-300,main} --params PARAMS.json --output STEPS.json
  chain run --spec STEPS.json --spec-sha256 HEX
  chain verify --log RUN_DIR/chain.jsonl --spec-sha256 HEX
  step helpers used inside planned argv:
  chain pin --file F --sha256 HEX                        exit 1 unless the file has that sha256
  chain with-sha256 --file F --token TOKEN -- CMD...     replace TOKEN by the file's sha256, exec CMD
  chain with-train-size --selection S --max-items N --per-update K -- CMD...
                                                         stage1: replace the size tokens by min(selected, N)
                                                         rounded down to K (and its update count), exec CMD
  chain receipt --source R --output O --step NAME        copy a throughput receipt as the stage receipt
  chain capture --output O -- CMD...                     run CMD, write its stdout to the new file O (also
                                                         echoed), exit with CMD's code (e.g. a reload receipt)

Spec `pod-chain-spec-v1` (shared with the eval stages): {"version", "steps": [{"name", "argv",
"expect_files", "timeout_seconds"?}], "meta"?}. Names are unique; expect_files are absolute paths
that must exist after the step exits 0 (existence only). Steps run in order, each in its own
session with the chain's cwd and environment unchanged. The first nonzero exit, missing expected
file (exit 1), step that cannot start (127) or timeout (124) stops the chain: no retry, no fallback,
no skipped step.
SIGTERM/SIGINT stop the running step's group (SIGTERM, then SIGKILL after 20 s, inside the
runner's 30 s grace) and exit 143/130. Records are hash-chained JSON lines in
$MMDL_RUN_DIR/chain.jsonl; the first record's prev_sha256 is the spec sha256.

Resume (local_train --resume-from) continues only a closed, unfinished segment with the identical
protocol, execution contract and preparation. The pilot is a completed run with its own N/K
protocol, so main cannot continue from it and starts at F0. A main segment stopped by its time
budget (BUDGET_STOP, exit 2) is continued by a new approved job whose params carry `resume`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import posixpath
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fine_tuning.budget import terminate_group

from .common import HEX64, read_json, sha256_bytes, sha256_file, utc_now, write_json

SPEC_VERSION = "pod-chain-spec-v1"
PARAMS_VERSION = "pod-chain-params-v1"
LOG_NAME = "chain.jsonl"
TIMEOUT_CODE = 124
MISSING_CODE = 1
STOP_GRACE_SECONDS = 20.0  # below runner.run_logged's 30 s, so the step dies before the chain is killed
POLL_SECONDS = 1.0
STEP_KEYS, OPTIONAL_STEP_KEYS = {"name", "argv", "expect_files"}, {"timeout_seconds"}
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
JOB_STAGE = {"stage1": "bf16-smoke-throughput", "pilot-300": "pilot-300", "main": "main"}
# The proof protocol. Slotless sources: exactly those of local-test-v3/chain-ten-v15.sh.
SLOTLESS_SOURCES = ("hf-chebi-role", "hf-deepweeds", "hf-geometry3k", "hf-gvlqa", "hf-mapqa", "hf-nctcrc",
                    "hf-plantvillage", "hf-scienceqa", "hf-slake", "hf-ssmr")
TRAIN_FLAGS = ["--note-contract", "repair-note-compact-v1", "--answer-verified-route", "--finalization-reward",
               *(item for source in SLOTLESS_SOURCES for item in ("--slotless-source", source))]
# local_train train takes only TRAIN_FLAGS; the turn1/retry/scale options are restored from the minimum.
# T1 cap 8192 (user decision 2026-10-31, = the proof's fast8k eval cap); retries keep 1024, the final note
# keeps its 8192 reservation in the 24576 context (repair_notebook), so the note prompt holds <= 16384 tokens.
PROOF_FLAGS = [*TRAIN_FLAGS, "--turn1-contract", "p0-eval-aligned-v1", "--turn1-max-new-tokens", "8192",
               "--retry-out-of-choice", "incorrect-v1"]
ADMISSION = "verified-functional-repair-exploration-v1"
MAIN_MAX_ITEMS = 3000
MAX_TRAIN_SECONDS = 86400  # local_train's own cap on --max-seconds
PREP_TOKEN = "@PREPARATION_SHA256@"
# stage1 trains on its own pass-rate selection (all-eligible items can carry no reward variance), sized at run time.
SIZE_TOKENS = ("@TRAIN_PROBLEMS@", "@MAX_UPDATES@")
COMMON_KEYS = {"version", "python", "run_dir", "model_path", "vault", "approval_request", "frozen_approval",
               "seed", "sentinels_per_subject", "min_free_gib", "vllm", "train", "timeouts"}
# "prefilter_vllm" (prefilter stages only): the vLLM budget of prefilter-propose/prefilter-prepare/passrate-measure,
# a process that never trains, so it may take a larger max_num_seqs/KV cache/utilization; absent = "vllm" (argv
# unchanged). The training preparation consumes only the selection and its report, never the prefilter protocol.
OPTIONAL_COMMON_KEYS = {"notes", "prefilter_vllm"}
STAGE_KEYS = {"stage1": ({"prefilter"}, set()), "pilot-300": ({"prefilter"}, set()),
              "main": (set(), {"selection", "resume"})}  # main: exactly one of the optional keys
SUB_KEYS = {"frozen_approval": {"path", "sha256"}, "selection": {"path", "sha256", "report"},
            "vllm": {"gpu_memory_utilization", "max_num_seqs", "kv_cache_memory_bytes", "enable_prefix_caching"},
            "prefilter_vllm": {"gpu_memory_utilization", "max_num_seqs", "kv_cache_memory_bytes",
                               "enable_prefix_caching"},
            "train": {"problems", "per_update", "checkpoint_every", "evidence_audit_rule", "max_seconds"},
            "timeouts": {"cpu", "passrate", "minimal", "reload", "train_close", "evaluate"},
            "prefilter": {"items", "k", "seed_namespace", "min_pass", "max_pass", "max_blocking"},
            "resume": {"from", "preparation", "minimal", "minimal_reload"}}
OPTIONAL_SUB_KEYS = {"prefilter": {"item_batch"}}  # passrate measure's --item-batch; absent = today's argv


class ChainError(ValueError):
    pass


# ---------------------------------------------------------------- params -> spec

def _path(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value.startswith("/") or value == "/"
            or posixpath.normpath(value) != value):
        raise ChainError(f"{name} must be a canonical absolute path")
    return value


def _int(value: Any, name: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ChainError(f"{name} must be an integer in [{low}, {high if high is not None else 'inf'}]")
    return value


def _keys(value: Any, keys: set[str], name: str, optional: set[str] | frozenset[str] = frozenset()) -> dict:
    if not isinstance(value, dict) or not keys <= set(value) <= keys | optional:
        raise ChainError(f"{name} must have exactly the keys {sorted(keys)}"
                         + (f" plus optionally {sorted(optional)}" if optional else ""))
    return value


def check_params(stage: str, params: Any) -> dict:
    required, optional = STAGE_KEYS[stage]
    keys = set(params) if isinstance(params, dict) else set()
    if (not COMMON_KEYS | required <= keys or keys - COMMON_KEYS - required - optional - OPTIONAL_COMMON_KEYS
            or stage == "main" and len(keys & optional) != 1 or "prefilter_vllm" in keys and "prefilter" not in keys):
        raise ChainError(f"{stage} params need {sorted(COMMON_KEYS | required)}"
                         + (" plus exactly one of selection|resume" if stage == "main" else ""))
    if params["version"] != PARAMS_VERSION:
        raise ChainError(f"params version must be {PARAMS_VERSION}")
    for key in keys & set(SUB_KEYS):
        _keys(params[key], SUB_KEYS[key], key, OPTIONAL_SUB_KEYS.get(key, set()))
    for key in ("python", "run_dir", "model_path", "vault", "approval_request"):
        _path(params[key], key)
    _path(params["frozen_approval"]["path"], "frozen_approval.path")
    for name, value in (("frozen_approval.sha256", params["frozen_approval"]["sha256"]),
                        *((("selection.sha256", params["selection"]["sha256"]),) if "selection" in params else ())):
        if not isinstance(value, str) or not HEX64.match(value):
            raise ChainError(f"{name} must be a lowercase sha256")
    _int(params["seed"], "seed", 0)
    _int(params["sentinels_per_subject"], "sentinels_per_subject", 1)
    free = params["min_free_gib"]
    if isinstance(free, bool) or not isinstance(free, (int, float)) or not math.isfinite(free) or free <= 0:
        raise ChainError("min_free_gib must be a positive number")
    for name in ("vllm", "prefilter_vllm"):
        if name not in params:
            continue
        vllm = params[name]
        if (type(vllm["gpu_memory_utilization"]) is not float or not 0 < vllm["gpu_memory_utilization"] <= 1
                or type(vllm["enable_prefix_caching"]) is not bool):
            raise ChainError(f"{name}.gpu_memory_utilization must be a float in (0, 1]; enable_prefix_caching a bool")
        _int(vllm["max_num_seqs"], f"{name}.max_num_seqs", 1)
        _int(vllm["kv_cache_memory_bytes"], f"{name}.kv_cache_memory_bytes", 1)
    train = params["train"]
    per_update = _int(train["per_update"], "train.per_update", 1)
    problems = _int(train["problems"], "train.problems", per_update, MAIN_MAX_ITEMS if stage == "main" else None)
    if problems % per_update:
        raise ChainError("train.problems must be a multiple of train.per_update (N == K * M)")
    _int(train["checkpoint_every"], "train.checkpoint_every", 1)
    _int(train["max_seconds"], "train.max_seconds", 1, MAX_TRAIN_SECONDS)
    if train["evidence_audit_rule"] is not None and not isinstance(train["evidence_audit_rule"], dict):
        raise ChainError("train.evidence_audit_rule must be null or an object")
    for key in SUB_KEYS["timeouts"]:
        _int(params["timeouts"][key], f"timeouts.{key}", 1)
    if "prefilter" in params:
        pre = params["prefilter"]
        k = _int(pre["k"], "prefilter.k", 4, 8)  # passrate.K_RANGE
        _int(pre["items"], "prefilter.items", 1)
        if not 0 <= _int(pre["min_pass"], "prefilter.min_pass", 0) <= _int(pre["max_pass"], "prefilter.max_pass", 0) <= k:
            raise ChainError("prefilter: 0 <= min_pass <= max_pass <= k")
        _int(pre["max_blocking"], "prefilter.max_blocking", 0)
        if "item_batch" in pre:
            _int(pre["item_batch"], "prefilter.item_batch", 1)
        if not isinstance(pre["seed_namespace"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
                                                                          pre["seed_namespace"]):
            raise ChainError("prefilter.seed_namespace must match passrate's namespace pattern")
    for key in ("selection", "resume"):
        for field, value in params.get(key, {}).items():
            if field != "sha256":
                _path(value, f"{key}.{field}")
    return params


def _scale(params: dict, items: int, per_update: int, *, training: bool, sized: bool = False,
           prefilter: bool = False) -> list[str]:
    """Scale flags; bound into the protocol hash, so identical for every command of one preparation.

    `sized`: N and M are the run-time SIZE_TOKENS that `with-train-size` fills from the selection.
    `prefilter`: the prefilter preparation's budget ("prefilter_vllm" when given, else "vllm")."""
    vllm, train = params.get("prefilter_vllm", params["vllm"]) if prefilter else params["vllm"], params["train"]
    flags = ["--problems-per-update", str(per_update), "--max-updates", SIZE_TOKENS[1] if sized else str(items // per_update),
             "--train-problems", SIZE_TOKENS[0] if sized else str(items)]
    if training:
        flags += ["--checkpoint-every-updates", str(train["checkpoint_every"])]
        if train["evidence_audit_rule"] is not None:
            flags += ["--evidence-audit-rule",
                      json.dumps(train["evidence_audit_rule"], sort_keys=True, separators=(",", ":"))]
    return flags + ["--policy-precision", "bf16", "--rollout-engine", "vllm",
                    "--vllm-gpu-memory-utilization", repr(vllm["gpu_memory_utilization"]),
                    "--vllm-max-num-seqs", str(vllm["max_num_seqs"]),
                    "--vllm-kv-cache-memory-bytes", str(vllm["kv_cache_memory_bytes"]),
                    *(["--vllm-enable-prefix-caching"] if vllm["enable_prefix_caching"] else [])]


def plan(stage: str, params: dict, params_sha256: str) -> dict:
    """The step list of one stage; a pure function of (stage, params)."""
    check_params(stage, params)
    out, py, t = params["run_dir"] + "/out", params["python"], params["timeouts"]
    frozen = ["--frozen-approval", params["frozen_approval"]["path"],
              "--frozen-approval-sha256", params["frozen_approval"]["sha256"]]
    train = params["train"]
    sized = stage == "stage1"
    scale = _scale(params, train["problems"], train["per_update"], training=True, sized=sized)
    steps: list[dict] = []

    def step(name: str, argv: list[str], expect: list[str], timeout: int) -> None:
        steps.append({"name": name, "argv": argv, "expect_files": expect, "timeout_seconds": timeout})

    def module(name: str, *args: str) -> list[str]:
        return [py, "-B", "-m", f"fine_tuning.notebook_rl.{name}", *args]

    def proof_rule(name: str, rule: str, prep: str, flags: list[str], selection: list[str],
                   size: list[str]) -> None:
        step(f"{name}propose", size + module("local_run", "propose-proof", "--build", params["vault"], *PROOF_FLAGS,
             *flags, "--sentinels-per-subject", str(params["sentinels_per_subject"]), "--seed", str(params["seed"]),
             *selection, *frozen, "--output", rule), [rule], t["cpu"])
        step(f"{name}prepare", size + module("local_run", "prepare-pool", "--build", params["vault"], "--approval",
             params["approval_request"], "--rule", rule, *PROOF_FLAGS, *flags, *frozen, "--output", prep),
             [prep + "/preparation.json"], t["cpu"])

    selection: list[str] = []
    size: list[str] = []
    if "prefilter" in params:  # F0 pass-rate measurement on its own label-blind preparation (K=1)
        pre = params["prefilter"]
        prefilter_scale = _scale(params, pre["items"], 1, training=False, prefilter=True)
        proof_rule("prefilter-", f"{out}/prefilter-rule.json", f"{out}/prefilter-prep", prefilter_scale, [], [])
        step("passrate-measure", module("passrate", "measure", "--preparation",
             f"{out}/prefilter-prep/preparation.json", "--model-path", params["model_path"], "--k", str(pre["k"]),
             "--seed-namespace", pre["seed_namespace"],
             *(["--item-batch", str(pre["item_batch"])] if "item_batch" in pre else []),
             *PROOF_FLAGS, *prefilter_scale, *frozen,
             "--output", f"{out}/passrate"),
             [f"{out}/passrate/summary.json", f"{out}/passrate/throughput_receipt.json"], t["passrate"])
        step("passrate-select", module("passrate", "select", "--report", f"{out}/passrate", "--min-pass",
             str(pre["min_pass"]), "--max-pass", str(pre["max_pass"]), "--max-blocking", str(pre["max_blocking"]),
             "--output", f"{out}/passrate-selection.json"), [f"{out}/passrate-selection.json"], t["cpu"])
        selection = ["--passrate-selection", f"{out}/passrate-selection.json", "--passrate-report", f"{out}/passrate"]
        if sized:  # N = min(selected, train.problems) rounded down to K; fewer than K selected refuses at run time
            size = module("pod.chain", "with-train-size", "--selection", f"{out}/passrate-selection.json",
                          "--max-items", str(train["problems"]), "--per-update", str(train["per_update"]), "--")
    if "selection" in params:  # main: the pilot's selection, pinned by sha256
        chosen = params["selection"]
        step("selection-pin", module("pod.chain", "pin", "--file", chosen["path"], "--sha256", chosen["sha256"]),
             [], t["cpu"])
        selection = ["--passrate-selection", chosen["path"], "--passrate-report", chosen["report"]]
    if "resume" in params:  # a later main segment: same preparation and minimum as the stopped segment
        resume = params["resume"]
        prep, minimal, minimal_reload = resume["preparation"], resume["minimal"], resume["minimal_reload"]
        resume_flags = ["--resume-from", resume["from"]]
    else:
        prep, minimal, minimal_reload, resume_flags = (f"{out}/prep/preparation.json", f"{out}/minimal",
                                                       f"{out}/minimal-reload", [])
        proof_rule("", f"{out}/rule.json", f"{out}/prep", scale, selection, size)
        step("minimal", size + module("local_run", "minimal", "--preparation", prep, "--model-path", params["model_path"],
             "--repair-notebook", "--context-limit", "24576", *PROOF_FLAGS, *scale, *frozen,
             "--min-free-gib", repr(float(params["min_free_gib"])), "--output", minimal),
             [f"{minimal}/result.json", f"{minimal}/reload_expected.json"], t["minimal"])
        step("minimal-reload", module("local_run", "reload", "--capture", minimal, "--output", minimal_reload),
             [f"{minimal_reload}/result.json"], t["reload"])
    capture, capture_reload = f"{out}/train", f"{out}/train-reload"
    step("train", module("local_train", "train", "--preparation", prep, "--model-path", params["model_path"],
         "--minimal", minimal, "--minimal-reload", minimal_reload, "--arm", "notebook", "--admission", ADMISSION,
         "--max-seconds", str(train["max_seconds"]), "--min-free-gib", repr(float(params["min_free_gib"])),
         *TRAIN_FLAGS, *resume_flags, *frozen, "--output", capture),
         [f"{capture}/result.json", f"{capture}/reload_expected.json", f"{capture}/throughput_receipt.json"],
         train["max_seconds"] + t["train_close"])
    step("train-reload", module("local_train", "reload", "--capture", capture, "--output", capture_reload),
         [f"{capture_reload}/result.json"], t["reload"])
    records = []
    for condition in ("F0", "FT"):
        target = f"{out}/eval-{condition.lower()}"
        step(f"eval-{condition.lower()}", module("local_eval", "evaluate-pool", "--capture", capture,
             "--preparation", prep, "--reload-result", f"{capture_reload}/result.json", "--minimal-capture", minimal,
             "--minimal-reload", minimal_reload, "--condition", condition, "--arm", "notebook", "--output", target),
             [f"{target}/result.json", f"{target}/recovery/result.json"], t["evaluate"])
        records.append((condition, f"{target}/recovery/result.json"))
    with_sha = module("pod.chain", "with-sha256", "--file", prep, "--token", PREP_TOKEN, "--")
    for condition, record in records:  # CPU regrade of each saved record (no model)
        target = f"{out}/replay-{condition.lower()}.json"
        step(f"replay-{condition.lower()}", with_sha + module("local_eval", "replay-pool", "--record", record,
             "--preparation-sha256", PREP_TOKEN, "--output", target), [target], t["cpu"])
    step("compare", with_sha + module("local_eval", "compare-pool",
         *(item for condition, record in records for item in ("--record", f"{condition}={record}")),
         "--preparation-sha256", PREP_TOKEN, "--output", f"{out}/compare.json"), [f"{out}/compare.json"], t["cpu"])
    # Every receipt writer mirrors into the run-dir root (last writer wins); the train receipt is the stage's.
    receipt = params["run_dir"] + "/throughput_receipt.json"
    step("stage-receipt", module("pod.chain", "receipt", "--source", f"{capture}/throughput_receipt.json",
         "--output", receipt, "--step", "train"), [receipt], t["cpu"])
    return {"version": SPEC_VERSION, "steps": steps,
            "meta": {"stage": stage, "job_stage": JOB_STAGE[stage], "run_dir": params["run_dir"],
                     "params_sha256": params_sha256, "protocol_flags": PROOF_FLAGS, "stage_receipt_step": "train",
                     "resume": "resume" in params,
                     "total_timeout_seconds": sum(item["timeout_seconds"] for item in steps)}}


# ---------------------------------------------------------------- spec -> run

def check_spec(spec: Any) -> dict:
    if not isinstance(spec, dict) or spec.get("version") != SPEC_VERSION or set(spec) - {"version", "steps", "meta"}:
        raise ChainError(f"spec must be {SPEC_VERSION} with keys version, steps and optional meta")
    steps, names = spec.get("steps"), set()
    if not isinstance(steps, list) or not steps or "meta" in spec and not isinstance(spec["meta"], dict):
        raise ChainError("spec.steps must be a non-empty list and meta an object")
    for item in steps:
        if not isinstance(item, dict) or not STEP_KEYS <= set(item) or set(item) - STEP_KEYS - OPTIONAL_STEP_KEYS:
            raise ChainError(f"each step has keys {sorted(STEP_KEYS)} plus optional {sorted(OPTIONAL_STEP_KEYS)}")
        if not isinstance(item["name"], str) or not item["name"] or item["name"] in names:
            raise ChainError(f"step names must be unique non-empty strings: {item['name']!r}")
        names.add(item["name"])
        argv, expect = item["argv"], item["expect_files"]
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(arg, str) and arg and "\0" not in arg for arg in argv)):
            raise ChainError(f"{item['name']}: argv must be a non-empty list of non-empty strings")
        if not isinstance(expect, list):
            raise ChainError(f"{item['name']}: expect_files must be a list")
        for path in expect:
            _path(path, f"{item['name']}: expect_files entry")
        if "timeout_seconds" in item:
            _int(item["timeout_seconds"], f"{item['name']}: timeout_seconds", 1)
    return spec


class ChainLog:
    """Append-only JSON lines; each record's sha256 covers its canonical JSON including prev_sha256."""

    def __init__(self, path: Path, genesis: str):
        self.path, self.prev, self.seq = path, genesis, 0
        with open(path, "x", encoding="utf-8"):
            pass

    def append(self, event: str, **fields: Any) -> dict:
        record = {"seq": self.seq, "event": event, "utc": utc_now(), "prev_sha256": self.prev, **fields}
        record["sha256"] = sha256_bytes(_canonical(record).encode())
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.prev, self.seq = record["sha256"], self.seq + 1
        return record


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def verify_log(path: str | Path, spec_sha256: str) -> list[dict]:
    """Every record in order, linked to its predecessor and to the spec; ValueError otherwise."""
    records, prev = [], spec_sha256
    for seq, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        body = {key: value for key, value in record.items() if key != "sha256"}
        if (record.get("seq") != seq or record.get("prev_sha256") != prev
                or record.get("sha256") != sha256_bytes(_canonical(body).encode())):
            raise ChainError(f"chain log broken at record {seq}")
        records.append(record)
        prev = record["sha256"]
    return records


def run(spec_path: str | Path, spec_sha256: str, *, grace: float = STOP_GRACE_SECONDS,
        log_name: str = LOG_NAME) -> int:
    raw = Path(spec_path).read_bytes()
    if sha256_bytes(raw) != spec_sha256:
        raise ChainError("spec sha256 differs from --spec-sha256; a changed spec needs a new approval")
    spec = check_spec(json.loads(raw))
    run_dir = os.environ.get("MMDL_RUN_DIR", "")
    if not run_dir.startswith("/") or not Path(run_dir).is_dir():
        raise ChainError("MMDL_RUN_DIR must name the existing run directory")
    meta = spec.get("meta", {})
    if meta.get("run_dir", run_dir) != run_dir:
        raise ChainError(f"spec was planned for run dir {meta['run_dir']}, not {run_dir}")
    if os.environ.get("MMDL_JOB_STAGE") and meta.get("job_stage", os.environ["MMDL_JOB_STAGE"]) != os.environ["MMDL_JOB_STAGE"]:
        raise ChainError(f"spec was planned for job stage {meta['job_stage']}, not {os.environ['MMDL_JOB_STAGE']}")
    if not NAME.fullmatch(log_name) or log_name in (".", ".."):
        raise ChainError("chain log name must be a safe basename")
    if (Path(run_dir) / log_name).exists():
        raise ChainError(f"{run_dir}/{log_name} exists; a run directory is never reused")
    stop: list[int] = []
    previous = {number: signal.signal(number, lambda signum, _frame: stop.append(signum))
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        log = ChainLog(Path(run_dir) / log_name, spec_sha256)
        log.append("CHAIN_START", spec=str(Path(spec_path).absolute()), spec_sha256=spec_sha256, pid=os.getpid(),
                   steps=[item["name"] for item in spec["steps"]])
        code, failed = 0, None
        for item in spec["steps"]:
            if stop:
                break
            if meta.get("protocol_version") == "error-note-v18b":
                from fine_tuning.error_note.v18b_execution import remaining, enforce_limits
                settings = meta["execution_settings"]
                if enforce_limits(settings):
                    seconds = int(remaining(settings, reserve_seconds=settings["recovery_reserve_seconds"])) - int(grace) - 1
                    if seconds < 1:
                        code, failed = 124, item["name"]
                        log.append("DEADLINE_REACHED", name=failed, recovery_seconds=settings["recovery_reserve_seconds"])
                        break
                    item = dict(item, timeout_seconds=min(seconds, item.get("timeout_seconds", seconds)))
            code, status, elapsed = _run_step(item, log, stop, grace)
            present = {path: os.path.exists(path) for path in item["expect_files"]}
            if status == "OK" and not all(present.values()):
                code, status = MISSING_CODE, "EXPECTED_FILES_MISSING"
            log.append("STEP_END", name=item["name"], status=status, exit_code=code,
                       elapsed_seconds=round(elapsed, 3), expect_files=present)
            if code:
                failed = item["name"]
                break
        if stop and not code:  # a signal between steps: no further step starts
            code = 128 + stop[0]
        status = "INTERRUPTED" if stop else "COMPLETE" if code == 0 else "STOPPED"
        log.append("CHAIN_END", status=status, exit_code=code, failed_step=failed)
        return code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _run_step(item: dict, log: ChainLog, stop: list[int], grace: float) -> tuple[int, str, float]:
    log.append("STEP_START", name=item["name"], argv=item["argv"], timeout_seconds=item.get("timeout_seconds"))
    print(f"CHAIN STEP {item['name']} {utc_now()}", flush=True)
    began = time.monotonic()
    try:
        process = subprocess.Popen(item["argv"], stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:  # missing/unexecutable argv[0]: a failed step, recorded like any other
        print(f"CHAIN STEP {item['name']} could not start: {exc}", file=sys.stderr, flush=True)
        return 127, "START_FAILED", time.monotonic() - began
    timeout = item.get("timeout_seconds")
    while True:
        if stop or timeout is not None and time.monotonic() - began >= timeout:
            terminate_group(process, grace_seconds=grace)
            if stop:
                return 128 + stop[0], "INTERRUPTED", time.monotonic() - began
            return TIMEOUT_CODE, "TIMEOUT", time.monotonic() - began
        try:
            code = process.wait(timeout=POLL_SECONDS)
        except subprocess.TimeoutExpired:
            continue
        code = code if code >= 0 else 128 - code
        return code, "OK" if code == 0 else "FAILED", time.monotonic() - began


# ---------------------------------------------------------------- step helpers

def stage_receipt(source: str | Path, output: str | Path, step: str) -> dict:
    from .cost import RECEIPT_VERSION

    receipt = read_json(source)
    if not isinstance(receipt, dict) or receipt.get("version") != RECEIPT_VERSION:
        raise ChainError(f"{source} is not a {RECEIPT_VERSION} receipt")
    value = {key: item for key, item in receipt.items() if key != "mirrored_from"} | {
        "stage_receipt_step": step, "stage_receipt_source": str(Path(source).absolute()),
        "stage_receipt_source_sha256": sha256_file(source)}
    write_json(output, value)
    return value


def sized_argv(selection: Any, max_items: int, per_update: int, cmd: list[str]) -> list[str]:
    """stage1's run-time training size: N = min(selected, max_items) rounded down to K, M = N / K."""
    selected = selection.get("selected_ids") if isinstance(selection, dict) else None
    if not isinstance(selected, list) or type(max_items) is not int or type(per_update) is not int or per_update < 1:
        raise ChainError("a pass-rate selection with selected_ids and positive integer sizes required")
    items = min(len(selected), max_items) // per_update * per_update
    if items < per_update:
        raise ChainError(f"the pass-rate selection keeps {len(selected)} items, fewer than one update of "
                         f"K={per_update}; refusing to train stage1 without a reward-variance selection")
    if not all(token in cmd for token in SIZE_TOKENS):
        raise ChainError("the command does not contain the size tokens")
    values = {SIZE_TOKENS[0]: str(items), SIZE_TOKENS[1]: str(items // per_update)}
    return [values.get(item, item) for item in cmd]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--stage", choices=tuple(JOB_STAGE), required=True)
    planner.add_argument("--params", required=True)
    planner.add_argument("--output", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--spec", required=True)
    runner.add_argument("--spec-sha256", required=True)
    runner.add_argument("--log-name", default=LOG_NAME)
    checker = sub.add_parser("verify")
    checker.add_argument("--log", required=True)
    checker.add_argument("--spec-sha256", required=True)
    pin = sub.add_parser("pin")
    pin.add_argument("--file", required=True)
    pin.add_argument("--sha256", required=True)
    with_sha = sub.add_parser("with-sha256")
    with_sha.add_argument("--file", required=True)
    with_sha.add_argument("--token", required=True)
    with_sha.add_argument("cmd", nargs=argparse.REMAINDER)
    with_size = sub.add_parser("with-train-size")
    with_size.add_argument("--selection", required=True)
    with_size.add_argument("--max-items", type=int, required=True)
    with_size.add_argument("--per-update", type=int, required=True)
    with_size.add_argument("cmd", nargs=argparse.REMAINDER)
    receipt = sub.add_parser("receipt")
    receipt.add_argument("--source", required=True)
    receipt.add_argument("--output", required=True)
    receipt.add_argument("--step", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--output", required=True)
    capture.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            if Path(args.output).exists():
                raise ChainError("refusing to overwrite an existing spec")
            spec = plan(args.stage, read_json(args.params), sha256_file(args.params))
            write_json(args.output, spec)
            print(json.dumps({"output": args.output, "spec_sha256": sha256_file(args.output),
                              "steps": len(spec["steps"]),
                              "total_timeout_seconds": spec["meta"]["total_timeout_seconds"]}))
            return 0
        if args.command == "run":
            return run(args.spec, args.spec_sha256, log_name=args.log_name)
        if args.command == "verify":
            print(json.dumps({"status": "VERIFIED", "records": len(verify_log(args.log, args.spec_sha256))}))
            return 0
        if args.command == "pin":
            actual = sha256_file(args.file)
            print(json.dumps({"file": args.file, "sha256": actual, "match": actual == args.sha256}))
            return 0 if actual == args.sha256 else 1
        if args.command == "with-sha256":
            cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
            if args.token not in cmd:
                raise ChainError("the command does not contain the token")
            resolved = [sha256_file(args.file) if item == args.token else item for item in cmd]
            print(json.dumps({"with_sha256": args.file, "argv": resolved}), flush=True)
            os.execvp(resolved[0], resolved)
        if args.command == "with-train-size":
            cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
            resolved = sized_argv(read_json(args.selection), args.max_items, args.per_update, cmd)
            print(json.dumps({"with_train_size": args.selection, "argv": resolved}), flush=True)
            os.execvp(resolved[0], resolved)
        if args.command == "capture":
            cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
            if not cmd:
                raise ChainError("capture needs a command")
            with open(args.output, "x", encoding="utf-8") as stream:  # write-once: a receipt is never replaced
                result = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                stream.write(result.stdout)
            sys.stdout.write(result.stdout)
            return result.returncode if result.returncode >= 0 else 128 - result.returncode
        value = stage_receipt(args.source, args.output, args.step)
        print(json.dumps({"stage_receipt": args.output, "source_sha256": value["stage_receipt_source_sha256"]}))
        return 0
    except (ChainError, OSError, ValueError, KeyError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
