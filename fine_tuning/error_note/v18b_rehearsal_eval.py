"""Non-reportable external evaluation after a completed BF16 rehearsal campaign.

Export, vLLM inference and the optional A100 verifier use separate processes so their
model allocations cannot overlap. No MMMU dataset or official evaluation registration
is opened by this command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .prep_v16b import file_sha, read_sealed, seal
from .v18b_data import load_items

MODES = ("tiny-bf16", "a100-bf16")


def _external_panel(root: Path) -> list[dict]:
    items = load_items(root, "performance")
    if (len(items) < 9 or len({r["problem_id"] for r in items}) != len(items)
            or any(r.get("split") != "performance" or r.get("training_allowed") is not False
                   or r.get("benchmark_derived") is not False for r in items)):
        raise ValueError("rehearsal requires nine distinct held-out external performance problems")
    return items[:9]


def _boundary(args):
    from .campaign_v16b import _binding, checked

    plan = read_sealed(args.campaign_output / "campaign-plan.json")
    state = read_sealed(args.campaign_output / "campaign-state.json")
    if (plan["protocol"]["version"] != "error-note-v18b"
            or plan["protocol"]["training_contract"].get("rehearsal_profile") != args.mode
            or state["status"] != "COMPLETE" or state["plan_sha256"] != plan["sha256"]
            or Path(plan["data_root"]).resolve() != args.data_root.resolve()
            or Path(plan["model_path"]).resolve() != args.model_path.resolve()):
        raise ValueError("a completed, bound rehearsal checkpoint is required")
    return checked(state["checkpoint"]).parent, _binding(plan)


def _tiny_factory(path, arm, protocol):
    from fine_tuning.notebook_rl.policy import configure_policy
    from fine_tuning.train import load_base

    return configure_policy(load_base(path), None, arm, protocol, enforce_counts=False)


def _export(args):
    from fine_tuning.modules import BASE_ID, BASE_REVISION
    from fine_tuning.notebook_rl.checkpoint import read_metadata
    from fine_tuning.notebook_rl.export import export_merged, validate_export_identity

    ckpt, binding = _boundary(args)
    metadata = read_metadata(ckpt, binding=binding)
    kwargs = {}
    if args.mode == "tiny-bf16":
        # The synthetic fixture has the same processor and export file layout.
        # The test-only manifest never enters a production model reference.
        base = args.model_path
        cfg = json.loads((base / "config.json").read_text())
        if cfg["text_config"]["num_hidden_layers"] != 2 or cfg["vision_config"]["depth"] != 3:
            raise ValueError("tiny export requires the two-layer rehearsal fixture")
        manifest = args.output / "tiny-base-files.json"
        files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": file_sha(p)}
                 for p in sorted(base.iterdir()) if p.is_file()]
        manifest.write_text(json.dumps({"repo_id": BASE_ID, "revision": BASE_REVISION, "files": files}))
        kwargs = {"_test_policy_factory": _tiny_factory, "_test_base_manifest": manifest}
    target = args.output / "model"
    export_merged(ckpt, args.model_path, target, metadata["arm"], binding=binding, llm_lora=True, **kwargs)
    identity = validate_export_identity(target, metadata["arm"])
    seal(args.output / "rehearsal-export.json", {"reportable": False,
         "checkpoint_metadata_sha256": file_sha(ckpt / "metadata.json"), "identity": identity})


def _generate(args):
    from mmdl.evaluation.backends.vllm_backend import VLLMBackend
    from mmdl.evaluation.parsers import score_with_parser
    from mmdl.runtime.contracts import load_configs, sample_seed
    from fine_tuning.notebook_rl import proof_eval as pe
    from fine_tuning.notebook_rl.checkpoint import load_eval, read_metadata
    from fine_tuning.notebook_rl.export import validate_export_identity
    from . import llm_lora
    from .prep_v16b import images, p0
    from .v17_eval_engine import engine_execution, observe_engine

    root = Path(__file__).resolve().parents[2]
    ckpt, binding = _boundary(args)
    export = read_sealed(args.output / "rehearsal-export.json")
    target = args.output / "model"
    if (export["checkpoint_metadata_sha256"] != file_sha(ckpt / "metadata.json")
            or export["identity"] != validate_export_identity(target, read_metadata(ckpt, binding=binding)["arm"])):
        raise ValueError("rehearsal export changed after sealing")
    tiny = args.mode == "tiny-bf16"
    factory = None
    if tiny:
        from .v18b_rehearsal import policy_factory
        factory = policy_factory
    bundle = load_eval(args.model_path, ckpt, binding=binding, _test_policy_factory=factory)
    items = _external_panel(args.data_root)
    index = {item["problem_id"]: item for item in items}

    def requests():
        for item in items:
            pid = item["problem_id"]
            pictures = images(item)
            content = [{"type": "image"} for _ in pictures] + [{"type": "text", "text": p0(item)}]
            yield {"sample_id": pid, "seed": sample_seed(3407, pid),
                   "messages": [{"role": "user", "content": content}], "images": pictures}

    slots, mode = (2, "eager") if tiny else (9, "cuda_graph")
    fraction = .6 if tiny else .9
    with engine_execution(mode, slots, gpu_memory_utilization=.6 if tiny else None), pe.registered_variants(slots=slots, log_stats=True):
        cfg, hw = load_configs(pe.VARIANTS[pe.FINAL_VARIANT_V8][0], root / "configs/hardware/rtx4090_24gb.yaml")
        backend = VLLMBackend(str(target), cfg, hw, pe.LORA_KIND, str(args.model_path))
        observed = observe_engine(backend, mode, slots)
        actual_fraction = backend.llm.llm_engine.vllm_config.cache_config.gpu_memory_utilization
        if actual_fraction != fraction:
            raise ValueError("rehearsal evaluation engine did not apply its GPU memory fraction")
        identity = llm_lora.check_identity(backend.llm, bundle, backend.lora_request.lora_int_id, backend.vision_hooks)
        if not identity["pass"]:
            raise ValueError(f"evaluation engine did not load the stored rehearsal weights: {identity}")
        # Only response length is shortened after validating the normal backend configuration.
        backend.cfg = dict(backend.cfg, generation=cfg["generation"] | {"max_new_tokens": 64})
        records = {}
        for row in backend.generate_stream(requests()):
            pid = row["sample_id"]
            if pid not in index or pid in records or row["finish_reason"] not in {"stop", "length"}:
                raise ValueError("unexpected, duplicated or incomplete rehearsal response")
            spec = index[pid]["evaluation"]
            scored = score_with_parser(row["raw_response"], "multiple-choice" if spec["answer_type"] == "mcq"
                                       else "open", spec["choices"], spec["gold"], "team-final-answer-v8",
                                       row["finish_reason"], question=spec["question"])
            records[pid] = {"sample_id": pid, "seed": sample_seed(3407, pid), "generated_tokens": row["generated_tokens"],
                            "raw_response": row["raw_response"], "finish_reason": row["finish_reason"], "v8": scored}
        if records.keys() != index.keys():
            raise ValueError("missing external rehearsal responses")
    seal(args.output / "external-responses.json", {"reportable": False, "split": "external-performance",
         "max_new_tokens": 64, "engine": observed, "adapter_identity": identity,
         "gpu_memory_utilization": actual_fraction,
         "records": [records[k] for k in index], "checkpoint_metadata_sha256": export["checkpoint_metadata_sha256"]})


def _checker_rows(external: dict, items: list[dict]) -> list[dict]:
    from mmdl.evaluation.CompassVerifier_checker import selection
    from mmdl.runtime.artifacts import digest
    from mmdl.runtime.contracts import sample_seed
    from .prep_v16b import p0

    records = external["records"]
    index = {item["problem_id"]: item for item in items}
    ids = [row["sample_id"] for row in records]
    if (len(records) != 9 or len(index) != 9 or len(set(ids)) != 9 or set(ids) != index.keys()
            or external["reportable"] is not False):
        raise ValueError("external rehearsal responses are incomplete or changed")
    rows = []
    for record in records:
        pid = record["sample_id"]
        if record["seed"] != sample_seed(3407, pid):
            raise ValueError("external rehearsal seed changed")
        item, spec = index[pid], index[pid]["evaluation"]
        row = {"id": pid, "subject": item["source"],
            "question_type": "multiple-choice" if spec["answer_type"] == "mcq" else "open",
            "prompt": p0(item), "question": spec["question"], "options": spec["choices"],
            "answer": spec["gold"], "raw_response": record["raw_response"],
            "finish_reason": record["finish_reason"], "generated_tokens": record["generated_tokens"],
            **{key: record["v8"][key] for key in ("correct", "parse_status", "parsed_answer", "extraction_rule")}}
        row["_record_sha256"] = digest(row)
        rows.append(row)
    selection.replay_v8(rows)
    return rows


def _supplementary_check(rows: list[dict], judge, config: dict) -> dict:
    from mmdl.evaluation.CompassVerifier_checker import cli, prompt, scoring, selection
    from mmdl.runtime.artifacts import digest

    before = digest(rows)
    targets = selection.build_targets(rows, config, judge.encode, judge.decode)
    results = []
    for target in targets:
        probability = judge.probabilities(prompt.build(target["question"], target["gold_answer"],
                                                       target["llm_response"]))
        scoring.verdict(probability, judge.identity["label_token_ids"], config["decision"]["accept_threshold"])
        results.append({key: target[key] for key in ("id", "alias_index", "input_sha256")} | probability)
    merged = scoring.merge(rows, targets, results, config | {"label_token_ids": judge.identity["label_token_ids"]})
    summary = scoring.summarize(merged, {"mode": "smoke", "reportable": False,
        **cli._binding(config), "source_predictions_sha256": [{"sha256": before, "scope": "external-rehearsal-v8-rows"}],
        "judge": judge.identity, "scope": "external-rehearsal", "source_denominator": len(rows)})
    control = None
    if not targets:
        case = next(case for case in cli._cases() if case["id"] == "fraction-decimal")
        response, truncated = selection.tail(case["llm_response"], config["input"]["response_tail_tokens"],
                                            judge.encode, judge.decode)
        probability = judge.probabilities(prompt.build(case["question"], case["gold_answer"], response))
        verdict = scoring.verdict(probability, judge.identity["label_token_ids"], config["decision"]["accept_threshold"])
        control = {"case_id": case["id"], "case_source": "fixed synthetic checker control",
                   "truncated": truncated, "probabilities": probability, "verdict": verdict}
        if verdict != "ACCEPT":
            raise ValueError("supplementary checker did not accept its preselected equivalence control")
    if digest(rows) != before:
        raise ValueError("supplementary checker changed an original V8 record")
    return {"reportable": False, "purpose": "external-supplementary-checker", "identity": judge.identity,
        "real_eligible_count": len({target["id"] for target in targets}),
        "real_eligible_ids": list(dict.fromkeys(target["id"] for target in targets)),
        "eligible_alias_count": len(targets), "targets": targets, "verdicts": results,
        "merged": merged, "summary": summary, "synthetic_control": control,
        "judge_call_count": len(results) + (control is not None), "original_v8_rows_sha256": before}


def _judge(args):
    import torch
    from mmdl.evaluation.CompassVerifier_checker import backend, cli

    if args.mode != "a100-bf16" or "A100" not in torch.cuda.get_device_name():
        raise ValueError("the full precision supplementary rehearsal requires the A100 profile")
    external_path = args.output / "external-responses.json"
    external = read_sealed(external_path)
    rows = _checker_rows(external, _external_panel(args.data_root))
    config = cli._config()
    profile = config["judge"]["official"]
    judge = backend.load_judge(profile["model"], profile["revision"], quantization=profile["quantization"])
    result = _supplementary_check(rows, judge, config)
    if read_sealed(external_path)["sha256"] != external["sha256"]:
        raise ValueError("external rehearsal responses changed during supplementary scoring")
    seal(args.output / "external-judge.json", result | {"external_responses_sha256": external["sha256"]})


def run_rehearsal_evaluation(*, campaign_output: Path, model_path: Path, data_root: Path, output: Path,
                             mode: str, deadline_monotonic: float) -> dict:
    from .v18b_rehearsal import run_child

    if mode not in MODES:
        raise ValueError("unknown rehearsal profile")
    output.mkdir(parents=True, exist_ok=False)
    stages = ["export", "generate"] + (["judge"] if mode == "a100-bf16" else [])
    durations = {}
    for stage in stages:
        start = time.monotonic()
        if start >= deadline_monotonic:
            raise TimeoutError(f"rehearsal deadline before {stage}")
        argv = [sys.executable, "-m", "fine_tuning.error_note.v18b_rehearsal_eval", stage, "--mode", mode,
                "--campaign-output", str(campaign_output), "--model-path", str(model_path),
                "--data-root", str(data_root), "--output", str(output)]
        with (output / f"{stage}.log").open("x") as log:
            result = run_child(argv,
                env=os.environ | {"HF_HUB_OFFLINE": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
                stream=log, deadline=deadline_monotonic)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, argv)
        durations[stage] = time.monotonic() - start
    generated = read_sealed(output / "external-responses.json")
    return seal(output / "evaluation.json", {"reportable": False, "profile": mode,
        "external_count": len(generated["records"]), "response_digest": generated["sha256"],
        "judge_receipt": read_sealed(output / "external-judge.json")["sha256"] if mode == "a100-bf16" else None,
        "stage_seconds": durations})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "generate", "judge"))
    parser.add_argument("--mode", required=True, choices=MODES)
    for name in ("campaign-output", "model-path", "data-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    {"export": _export, "generate": _generate, "judge": _judge}[args.stage](args)
