"""Plan the proof's MMMU-val evaluation stages as pod/chain.py specs (`pod-chain-spec-v1`).

CLI: python -m fine_tuning.notebook_rl.pod.eval_stage plan --stage {eval-f0,eval-ft}
         --params PARAMS.json --output STEPS.json
Writes STEPS.json once (an existing file is refused) and prints its sha256, the value the approved job's
main argv passes as `chain run --spec STEPS.json --spec-sha256 HEX`. Same params -> same bytes. Executes
nothing; analysis (`proof_eval analyze`) stays local after collection.

Arms, variants, seeds and config files come from proof_eval (VARIANTS, ARMS, SEEDS, SUBSETS), the tables its
preregistration is built from: statistical runs of arm F0 (stage eval-f0) or FT-NOTE / FT-SOLUTION (stage eval-ft)
x the requested 8k seeds (32 slots, 8192 output tokens, 262144..1310720 pixels, 24576 context;
`proof_eval run-eval`) on one subset ("wrong" = the 342 B2 V8 wrong items, "rest" = the 558 right ones). Every run
is `--mode partial --run-role standalone` with one `--sample-id` per ID of the pinned subset file (the evaluator
refuses run role `evaluation` for a partial run). The course run (FT x mmmu-val-v8, plain scripts/eval.sh) keeps
its code path here but stage `eval-course` is refused: it is not in the proof plan. Every stage pins its inputs
first (`chain pin`: subset, prereg, ...), then appends all of its ledger STARTs, then runs; the ledger START of a
run always precedes the run, and an omitted or replaced seed stays visible to `analyze --ledger`.
The planned (arm, variant) pairs are the preregistration's runs of the arm, subset_tag and seeds (`_runs`): prereg
v8 lists only the B2-condition variant (proof_eval.FINAL_VARIANT_V8, 32768 tokens, 1003520..4014080 pixels, 40960
context, run with `run-eval` under the same slot rules), v3-v7 the 8k ones. Prereg v9 lists the same B2 variant on the
ID-only halves "half-a" and "half-b" (one plan per half; half-a's job first): a half-b ledger step also passes
`--artifact-root <artifact_root>`, so `ledger-start` refuses it unless the arm's finished half-a run, written under the
same artifact_root and ledger by the earlier job, passes the preregistered screen gate (`proof_eval gate-screen`).
Prereg v10 (v15 export, adapters applied unmerged): the half-a FT-SINGLE plan runs `fine_tuning.error_note.lora_preflight
check` (in-process vLLM, synthetic inputs only) after verify-single-export and before its ledger START; a failure stops
the chain before any START or MMMU item. Half-b does not repeat it (the engine's own scope assertion still runs).
Run ids: <run_prefix>-<arm lowercased>-<8k|b2>-s<seed>-<subset_tag>-evaluation (job id = run id without -evaluation).

params (`eval-stage-params-v1`), absolute canonical paths:
  all stages   commit (40-hex clone commit, --require-commit), repo_root (the clone), python
               (<volume_root>/venv-<16 hex>/bin/python), volume_root, artifact_root, prereg, prereg_sha256,
               ledger, run_prefix, subset (the subset JSON of subset_tag), subset_sha256; artifact_root, ledger,
               prereg and subset under <volume_root>/eval-data/. `plan` reads the subset only after its
               sha256 equals subset_sha256. arm ("F0" for eval-f0; "FT-NOTE" or "FT-SOLUTION" for eval-ft),
               seeds (non-empty list of distinct ints from 3407, 3408, 3409: only these runs are planned),
               subset_tag ("wrong" | "rest"; prereg v8 "all"; prereg v9 "half-a" | "half-b").
  eval-ft      + checkpoint, checkpoint_metadata_sha256 (sha256 of <checkpoint>/metadata.json, which pins
               every checkpoint file), binding, binding_sha256, training_commit, ft_export (new dir)
               optional: reuse_export true (+ ft_model_ref_sha256): ft_export already holds the export of an
               earlier stage, so the export step is skipped and <ft_export>/model_ref.json is pinned by that sha
  all stages   optional: eval_slots ("auto" or a positive int): the run-eval steps run under `/usr/bin/env
               MMDL_EVAL_SLOTS=<value>` (proof_eval.resolve_slots); absent = the environment of the job runner
  all stages   optional: vllm_inprocess (true): the run-eval steps run under `/usr/bin/env
               VLLM_ENABLE_V1_MULTIPROCESSING=0` (vLLM engine core in the evaluator's process). The protocol files and
               config digests are unchanged; the run's variant_registration.json records the value, so analyze
               refuses to pool runs with and without it. Required for arm FT-V2 (error-note version 2, prereg v4:
               seed 3407 only); its F0 runs in the same ledger must set it too.
               FT-NOTE-V3 (prereg v5) also requires seed 3407, in-process vLLM and reuse_export true; a separate
               verification step connects the declared model reference to its checkpoint before any ledger START.
               FT-HYB-V4 (prereg v6) and FT-SINGLE (prereg v7; binding = the campaign's campaign-plan.json) likewise.
  eval-course  + ft_export, ft_model_ref_sha256 (sha256 of <ft_export>/model_ref.json written by eval-ft)
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from pathlib import Path
from typing import Any

from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION

from .. import proof_eval
from .common import HEX40, HEX64, sha256_bytes
from .job import EVAL_STAGES
from .verify import EVAL_CACHE, EVAL_DATA_ROOT, EVAL_DIR

PARAMS_VERSION = "eval-stage-params-v1"
SPEC_VERSION = "pod-chain-spec-v1"
CHAIN = "fine_tuning.notebook_rl.pod.chain"
PROOF = "fine_tuning.notebook_rl.proof_eval"
PREFLIGHT = "fine_tuning.error_note.lora_preflight"
HARDWARE = "fine_tuning/notebook_rl/eval_configs/hardware/a40_48gb.yaml"
BASE_REF = "manifests/models/baseline.json"
COMMON = {"version", "commit", "repo_root", "python", "volume_root", "artifact_root", "prereg",
          "prereg_sha256", "ledger", "run_prefix", "subset", "subset_sha256", "arm", "seeds", "subset_tag"}
NOT_PATHS = {"version", "run_prefix", "arm", "seeds", "subset_tag", "reuse_export", "eval_slots", "vllm_inprocess"}
ARM_OF = {"eval-f0": ("F0",), "eval-ft": ("FT-NOTE", "FT-SOLUTION", "FT-NOTE-3417", proof_eval.ARM_V2, proof_eval.ARM_V3,
                                          proof_eval.ARM_V4, proof_eval.ARM_SINGLE)}
# declared before evaluation: reuse_export + verify-<tag>-export (FT-SINGLE: binding = its campaign-plan.json)
FINAL_ARMS = (proof_eval.ARM_V3, proof_eval.ARM_V4, proof_eval.ARM_SINGLE)
STAGES = ("eval-f0", "eval-ft")  # planned; eval-course is dropped from the proof (user decision 2026-10-31)
MMMU_ID = re.compile(r"validation_[A-Za-z_]+_[0-9]{1,3}\Z")
EXTRA = {"eval-f0": set(),
         "eval-ft": {"checkpoint", "checkpoint_metadata_sha256", "binding", "binding_sha256",
                     "training_commit", "ft_export"},
         "eval-course": {"ft_export", "ft_model_ref_sha256"}}
OPTIONAL = {"eval-f0": {"eval_slots", "vllm_inprocess"},
            "eval-ft": {"reuse_export", "ft_model_ref_sha256", "eval_slots", "vllm_inprocess"}}
PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}\Z")
VENV_PYTHON = re.compile(r"/venv-[0-9a-f]{16}/bin/python\Z")
# Raw responses (with MMMU questions and gold), the ledger and the prereg live under <volume_root>/eval-data,
# the root the training preflight proves every training-side guard forbids.
UNDER_EVAL_ROOT = ("artifact_root", "ledger", "prereg", "subset")


def _check(stage: str, params: Any) -> dict[str, Any]:
    if stage not in EVAL_STAGES:
        raise ValueError(f"stage must be one of {EVAL_STAGES}")
    if stage not in STAGES:
        raise ValueError(f"{stage} is not in the proof plan (user decision 2026-10-31: the subset runs only)")
    keys = COMMON | EXTRA[stage]
    if not isinstance(params, dict) or not keys <= set(params) <= keys | OPTIONAL[stage]:
        raise ValueError(f"{stage} params must have the keys {sorted(keys)} and may add {sorted(OPTIONAL[stage])}")
    if params["version"] != PARAMS_VERSION:
        raise ValueError(f"params version must be {PARAMS_VERSION}")
    if ("reuse_export" in params and params["reuse_export"] is not True) or (
            "reuse_export" in params) != ("ft_model_ref_sha256" in params and stage == "eval-ft"):
        raise ValueError("reuse_export (true) and ft_model_ref_sha256 are given together, for eval-ft only")
    slots = params.get("eval_slots", "auto")
    if slots != "auto" and (type(slots) is not int or slots < 1):
        raise ValueError('eval_slots must be "auto" or a positive integer')
    if params["arm"] not in ARM_OF[stage]:
        raise ValueError(f"arm of {stage} must be one of {ARM_OF[stage]}")
    if "vllm_inprocess" in params and params["vllm_inprocess"] is not True:
        raise ValueError("vllm_inprocess must be true when given")
    if params["arm"] in (proof_eval.ARM_V2, *FINAL_ARMS) and (
            params["seeds"] != [proof_eval.VARIANTS[proof_eval.GATE_62_VARIANT][1]] or params.get("vllm_inprocess") is not True):
        raise ValueError(f"arm {params['arm']} runs seed 3407 only ({proof_eval.GATE_62_VARIANT}) "
                         "and needs vllm_inprocess true")
    if params["arm"] in FINAL_ARMS and params.get("reuse_export") is not True:
        raise ValueError(f"{params['arm']} requires reuse_export true: its final export must be declared before evaluation")
    seeds = params["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(x) is not int or x not in proof_eval.SEEDS for x in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError(f"seeds must be a non-empty list of distinct values from {list(proof_eval.SEEDS)}")
    if params["subset_tag"] not in proof_eval.SUBSET_FILES:
        raise ValueError(f"subset_tag must be one of {list(proof_eval.SUBSET_FILES)}")
    for key in (keys | (OPTIONAL[stage] & set(params))) - NOT_PATHS:
        value = params[key]
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if key.endswith("commit"):
            ok = bool(HEX40.match(value))
        elif key.endswith("sha256"):
            ok = bool(HEX64.match(value))
        else:
            ok = value.startswith("/") and value != "/" and posixpath.normpath(value) == value
        if not ok:
            raise ValueError(f"{key} is malformed: {value!r}")
    if not isinstance(params["run_prefix"], str) or not PREFIX.match(params["run_prefix"]):
        raise ValueError("run_prefix must match [A-Za-z0-9][A-Za-z0-9_.-]{0,40}")
    if (not params["python"].startswith(params["volume_root"] + "/")
            or not VENV_PYTHON.match(params["python"][len(params["volume_root"]):])):
        raise ValueError("python must be <volume_root>/venv-<lock sha256[:16]>/bin/python")
    volume = params["volume_root"]
    reserved = (f"{volume}/{EVAL_DATA_ROOT}", f"{volume}/{EVAL_CACHE}")
    for key in UNDER_EVAL_ROOT:
        value = params[key]
        if (not value.startswith(f"{volume}/{EVAL_DIR}/")
                or any(value == r or value.startswith(r + "/") for r in reserved)):
            raise ValueError(f"{key} must be under {volume}/{EVAL_DIR}/ (outside mmmu/ and hf-cache/) "
                             "so that the training guards forbid it")
    return params


def _runs(stage: str, params: dict[str, Any]) -> list[tuple[str, str]]:
    """(arm, variant) of the preregistration's runs for the arm, subset_tag and requested seeds, in its order.

    The preregistration is params['prereg'] when its bytes have params['prereg_sha256']. A file that is absent,
    differs or lists no runs (fixtures) plans the 8k variants of the requested seeds, which is what every v3-v7
    preregistration lists; the b2 variant is planned only from a preregistration that lists it. Every run is
    checked again at run time: `chain pin` refuses other prereg bytes and `ledger-start` a run it does not list."""
    seed_of = {v: seed for v, (_, seed, _, role) in proof_eval.VARIANTS.items() if role == "statistical"}
    loaded = _prereg(params)
    runs = loaded.get("runs") if isinstance(loaded, dict) else None
    if not isinstance(runs, list):
        return [(params["arm"], v) for v in proof_eval.FAST_VARIANTS if seed_of[v] in params["seeds"]]
    planned = [(params["arm"], r["variant"]) for r in runs
               if isinstance(r, dict) and (r.get("arm"), r.get("subset")) == (params["arm"], params["subset_tag"])
               and seed_of.get(r.get("variant")) in params["seeds"]]
    if not planned:
        raise ValueError(f"the preregistration lists no run of {params['arm']} on {params['subset_tag']} "
                         f"for seeds {params['seeds']}")
    return planned


def _prereg(params: dict[str, Any]) -> Any:
    """The preregistration JSON when params['prereg'] has params['prereg_sha256'], else None (fixtures)."""
    path = Path(params["prereg"])
    raw = path.read_bytes() if path.is_file() else b""
    return json.loads(raw) if raw and sha256_bytes(raw) == params["prereg_sha256"] else None


def subset_ids(params: dict[str, Any]) -> list[str]:
    """The pinned subset's sorted unique MMMU-val IDs; refused unless the file has the params' sha256."""
    raw = Path(params["subset"]).read_bytes()
    if sha256_bytes(raw) != params["subset_sha256"]:
        raise ValueError("subset file sha256 differs from params.subset_sha256")
    loaded = json.loads(raw)
    ids = loaded.get("sample_ids") if isinstance(loaded, dict) else None
    version = proof_eval.SUBSET_FILES[params["subset_tag"]][0]
    if (loaded.get("version") != version or not isinstance(ids, list) or not ids
            or loaded.get("count") != len(ids) or ids != sorted(set(ids))
            or not all(isinstance(i, str) and MMMU_ID.match(i) for i in ids)):
        raise ValueError(f"subset must be {version} with sorted unique MMMU-val sample_ids and their count")
    return ids


def run_id(prefix: str, arm: str, variant: str, subset_tag: str) -> str:
    seed = proof_eval.VARIANTS[variant][1]
    tag = {proof_eval.COURSE: "course", proof_eval.FINAL_VARIANT_V8: "b2"}.get(variant, "8k")
    return f"{prefix}-{arm.lower()}-{tag}-s{seed}-{subset_tag}-evaluation"


def _step(name: str, argv: list[str], expect: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "argv": argv, "expect_files": expect or []}


def plan(stage: str, params: dict[str, Any]) -> dict[str, Any]:
    p = _check(stage, params)
    py, repo, artifacts = p["python"], p["repo_root"], p["artifact_root"]
    snapshot = (f"{p['volume_root']}/hf-cache/hub/models--{MODEL_ID.replace('/', '--')}"
                f"/snapshots/{MODEL_REVISION}")
    ids = subset_ids(p)
    pins = [("subset", p["subset"], p["subset_sha256"]), ("prereg", p["prereg"], p["prereg_sha256"])]
    if stage == "eval-ft":
        pins += [("checkpoint", f"{p['checkpoint']}/metadata.json", p["checkpoint_metadata_sha256"]),
                 ("binding", p["binding"], p["binding_sha256"])]
    reuse = stage == "eval-course" or p.get("reuse_export") is True
    if reuse:  # the export of an earlier stage, bound by sha instead of re-exported
        pins += [("ft-model-ref", f"{p['ft_export']}/model_ref.json", p["ft_model_ref_sha256"])]
    steps = [_step(f"pin-{name}", [py, "-m", CHAIN, "pin", "--file", path, "--sha256", sha])
             for name, path, sha in pins]
    if stage == "eval-f0":
        model_ref, model_path = f"{repo}/{BASE_REF}", snapshot
    else:
        model_ref, model_path = f"{p['ft_export']}/model_ref.json", f"{p['ft_export']}/model"
    if stage == "eval-ft" and not reuse:
        steps.append(_step("export-ft", [
            py, "-m", PROOF, "export", "--checkpoint", p["checkpoint"], "--binding", p["binding"],
            "--base", snapshot, "--output", p["ft_export"], "--code-commit", p["training_commit"]],
            [model_ref, f"{model_path}/export_manifest.json"]))
    if p["arm"] in FINAL_ARMS:
        command = f"verify-{proof_eval.FINAL_TAGS[p['arm']][1]}-export"
        steps.append(_step(command, [py, "-m", PROOF, command, "--model-ref", model_ref,
                                                "--checkpoint", p["checkpoint"], "--binding", p["binding"],
                                                "--prereg", p["prereg"]]))
    loaded = _prereg(p)
    if (p["arm"] == proof_eval.ARM_SINGLE and p["subset_tag"] == proof_eval.HALVES[0]
            and isinstance(loaded, dict) and loaded.get("format") == proof_eval.FORMAT_V10):
        report = f"{p['ft_export']}/lora_preflight/preflight.json"
        env = [f"{proof_eval.VLLM_INPROCESS_ENV}=0", *([f"{proof_eval.SLOTS_ENV}={p['eval_slots']}"] if "eval_slots" in p else [])]
        protocol = f"{repo}/{proof_eval._rel(proof_eval.VARIANTS[proof_eval.FINAL_VARIANT_V8][0])}"
        steps.append(_step("lora-preflight", [
            "/usr/bin/env", *env, py, "-m", PREFLIGHT, "check", "--model-ref", model_ref, "--checkpoint", p["checkpoint"],
            "--binding", p["binding"], "--base", snapshot, "--protocol", protocol, "--hardware", f"{repo}/{HARDWARE}",
            "--out", report], [report]))
    runs = [(arm, variant, run_id(p["run_prefix"], arm, variant, p["subset_tag"])) for arm, variant in _runs(stage, p)]
    for arm, variant, rid in runs:
        steps.append(_step(f"ledger-{rid}", [
            py, "-m", PROOF, "ledger-start", "--prereg", p["prereg"], "--arm", arm, "--variant", variant,
            "--run-id", rid, "--subset", p["subset_tag"], "--ledger", p["ledger"],
            # prereg v9: a half-b start re-checks the arm's finished half-a run under this root (the screen gate)
            *(("--artifact-root", artifacts) if p["subset_tag"] == "half-b" else ())], [p["ledger"]]))
    for arm, variant, rid in runs:
        protocol = f"{repo}/{proof_eval._rel(proof_eval.VARIANTS[variant][0])}"
        flags = ["--protocol", protocol, "--hardware", f"{repo}/{HARDWARE}", "--model-ref", model_ref,
                 "--model-path", model_path, "--base-path", snapshot,
                 "--data-root", f"{p['volume_root']}/{EVAL_DATA_ROOT}", "--artifact-root", artifacts,
                 "--public-root", f"{artifacts}/public", "--require-commit", p["commit"],
                 "--job-id", rid.removesuffix("-evaluation"), "--run-id", rid, "--no-download"]
        if variant == proof_eval.COURSE:
            flags += ["--run-role", "evaluation", "--mode", "full"]
        else:  # cli refuses run role evaluation for a partial run; standalone records the same identity
            flags += ["--run-role", "standalone", "--mode", "partial", *(x for i in ids for x in ("--sample-id", i))]
        run_dir = f"{artifacts}/runs/{rid}"
        expect = [f"{run_dir}/summary.json", f"{run_dir}/predictions.jsonl"]
        if variant == proof_eval.COURSE:  # the course run is the plain entry point, never run-eval
            argv = ["/usr/bin/env", f"MMDL_PYTHON={py}", "bash", f"{repo}/scripts/eval.sh", *flags]
        else:
            argv = [py, "-m", PROOF, "run-eval", *flags]
            env = ([f"{proof_eval.SLOTS_ENV}={p['eval_slots']}"] if "eval_slots" in p else []) + (
                [f"{proof_eval.VLLM_INPROCESS_ENV}=0"] if p.get("vllm_inprocess") else [])
            if env:
                argv = ["/usr/bin/env", *env, *argv]
            expect.append(f"{run_dir}/{proof_eval.REGISTRATION_FILE}")
        steps.append(_step(f"eval-{rid}", argv, expect))
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    # job_stage: chain.run refuses this spec under a job of another stage (a training-stage job, whose preflight
    # skips the eval_data check and is bound to the training-data pins, must never carry an MMMU evaluation).
    return {"version": SPEC_VERSION, "meta": {"stage": stage, "job_stage": stage,
                                              "params_sha256": sha256_bytes(canonical.encode()),
                                              "generator": "fine_tuning.notebook_rl.pod.eval_stage plan"},
            "steps": steps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="write the chain spec of one evaluation stage")
    plan_parser.add_argument("--stage", choices=STAGES, required=True)
    plan_parser.add_argument("--params", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        text = json.dumps(plan(args.stage, json.loads(args.params.read_text(encoding="utf-8"))),
                          indent=2, sort_keys=True) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(sha256_bytes(text.encode()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
