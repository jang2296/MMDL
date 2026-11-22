"""Evaluation and saved-response rescoring commands; never train or augment."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from mmdl.runtime.artifacts import default_path, digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION, SUBJECTS, SUITE_COUNTS, SUITE_DATA_DIRS, load_configs, suite_of


def validate_execution(args, hw):
    if args.mode == "full" and hw["placement"] != "gpu_only":
        raise ValueError("Local CPU-offload full evaluation is prohibited; use the GPU-only reproduction path")
    if args.job_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", args.job_id):
        raise ValueError("Invalid job ID")
    if args.run_role in {"assignment", "analysis"}:
        raise ValueError("Legacy assignment/analysis roles cannot start a new evaluation")
    if args.run_role == "evaluation":
        if args.mode != "full" or not args.job_id or args.run_id != f"{args.job_id}-{args.run_role}":
            raise ValueError("Evaluation role requires a full run and the matching job-role run ID")
        if not args.require_commit:
            raise ValueError("Evaluation runs require a fixed Git commit")


def arguments(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=list(SUITE_COUNTS), default="mmmu-val")
    parser.add_argument("--model", choices=["base", "finetuned"], default="base")
    parser.add_argument("--score", choices=["v8", "both"], default="v8")
    parser.add_argument("--score-only", action="store_true",
                        help="Run only supplemental scoring on the completed saved run")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--model-ref", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--rehearsal", action="store_true",
                        help="Synthetic tiny-model smoke only; never a reportable score")
    parser.add_argument("--checker-mode", choices=["official", "smoke"])
    parser.add_argument("--checkset-receipt", type=Path)
    parser.add_argument("--checker-run", type=Path, help="Replay an existing saved judge run")
    parser.add_argument("--engine-validation", type=Path,
                        help="Completed reference/new-engine equivalence receipt")
    parser.add_argument("--base-path", type=Path, help="Pinned official base snapshot for a derived checkpoint")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts"))
    parser.add_argument("--public-root", type=Path, default=root / "results")
    parser.add_argument("--job-id")
    parser.add_argument("--run-role", choices=["standalone", "smoke", "evaluation"], default="standalone")
    parser.add_argument("--require-commit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "partial", "full"], default="full")
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    if args.score_only and args.score != "both":
        parser.error("--score-only requires --score both")
    if args.rehearsal:
        if args.mode != "smoke" or not all((args.protocol, args.model_ref, args.model_path, args.data_root)):
            parser.error("--rehearsal requires smoke mode and explicit protocol/model-ref/model-path/data-root")
        if args.checker_mode == "official":
            parser.error("Rehearsal permits only local 3B smoke checking")
        args.no_download = True
    args.checker_mode = args.checker_mode or ("smoke" if args.rehearsal else "official")
    if args.checker_mode == "smoke" and not args.rehearsal:
        parser.error("Local 3B checking requires --rehearsal; official scores require pinned 7B")
    args.hardware = args.hardware or root / "configs/hardware" / (
        "rtx4090_24gb.yaml" if args.suite == "mmmu-val" and args.model == "base" else "a100_sxm_80gb.yaml")
    profile = "4090" if "24gb" in args.hardware.name else "a100"
    args.protocol = args.protocol or root / "configs/eval" / (
        "mmmu_test_lowres2k_v8.yaml" if args.suite == "mmmu-test" else
        "mmmu_val_v8.yaml" if args.suite == "mmmu-val" and args.model == "base" else
        f"mmmu_val_v18b_ft_{profile}.yaml" if args.suite == "mmmu-val" else
        f"mmmu_pro_standard10_v8_{profile}.yaml")
    args.model_ref = args.model_ref or root / "manifests/models" / (
        "baseline.json" if args.model == "base" else "v18b.json")
    args.data_root = args.data_root or default_path("MMDL_DATA_ROOT", "mmdl-data") / SUITE_DATA_DIRS[args.suite]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        parser.error("run-id must be a safe directory name")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.sample_id and len(args.sample_id) != len(set(args.sample_id)):
        parser.error("sample-id values must be unique")
    return parser, args


def _checker_command(command, args, *extra):
    from mmdl.evaluation.CompassVerifier_checker.selection import load_config

    config = load_config()
    profile = config["judge"]["official"] if args.checker_mode == "official" else {
        **config["judge"]["smoke_allowed"][0], "quantization": "nf4"}
    return [sys.executable, "-m", "mmdl.evaluation.CompassVerifier_checker", command,
            "--mode", args.checker_mode, "--model", profile["model"],
            "--revision", profile["revision"], "--quantization", profile["quantization"],
            *map(str, extra)]


def _subprocess_directory(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    result.check_returncode()
    return Path(result.stdout.strip().splitlines()[-1])


def _checkset(args):
    if args.checker_run or args.checker_mode != "official":
        return args.checkset_receipt
    from mmdl.evaluation.CompassVerifier_checker import cli as checker

    receipt_record = args.artifact_root / "judge" / f"{args.run_id}-checkset-receipt.json"
    if args.checkset_receipt is None and receipt_record.exists():
        args.checkset_receipt = Path(read_json(receipt_record)["checkset_receipt"])
    if args.checkset_receipt is not None:
        config = checker._config()
        profile = {"mode": "official", "reportable": True, **config["judge"]["official"]}
        profile.pop("dtype")
        checker._require_checkset(args.checkset_receipt, profile, config)
        return args.checkset_receipt
    directory = _subprocess_directory(_checker_command(
        "checkset", args, "--run-label", args.run_id + "-checkset",
        "--output-root", args.artifact_root / "judge"))
    receipt = directory / "checkset.json"
    write_json(receipt_record, {"checkset_receipt": str(receipt)})
    return receipt


def _provision_checker(args):
    if args.checker_run or args.checker_mode != "official":
        return
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            check=True, text=True, capture_output=True)
    if "5060" in result.stdout:
        raise ValueError("CompassVerifier-7B download and execution are prohibited on the local RTX 5060")
    if not args.no_download:
        from huggingface_hub import snapshot_download
        from mmdl.evaluation.CompassVerifier_checker.selection import load_config

        profile = load_config()["judge"]["official"]
        snapshot_download(repo_id=profile["model"], revision=profile["revision"],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])


def load_evaluation_configs(args):
    if not args.rehearsal:
        return load_configs(args.protocol, args.hardware)
    import copy
    import yaml
    from mmdl.runtime.contracts import EXECUTION_PROFILES, PROFILE_MAX_NEW_TOKENS, SUBMISSION_PROFILES, validate_configs

    cfg = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    hw = yaml.safe_load(args.hardware.read_text(encoding="utf-8"))
    pid = cfg.get("protocol_id")
    if pid not in SUBMISSION_PROFILES or hw.get("placement") != "gpu_only":
        raise ValueError("Rehearsal requires a named submission profile and GPU-only hardware")
    if not 1 <= cfg["generation"]["max_new_tokens"] <= 128:
        raise ValueError("Rehearsal output must be between 1 and 128 tokens")
    if not 1 <= cfg["execution"]["batch_size"] <= 3:
        raise ValueError("Rehearsal permits only one to three concurrent requests")
    if not 0.1 <= cfg["execution"]["gpu_memory_utilization"] <= 0.9:
        raise ValueError("Invalid rehearsal GPU memory fraction")
    canonical = copy.deepcopy(cfg)
    canonical["generation"]["max_new_tokens"] = PROFILE_MAX_NEW_TOKENS.get(pid, 32768)
    for key in ("batch_size", "gpu_memory_utilization"):
        canonical["execution"][key] = EXECUTION_PROFILES[pid][key]
    target = SUBMISSION_PROFILES[pid][3]
    canonical_hw = hw | {"name": "a100_sxm_80gb" if target == "a100" else "rtx4090_24gb",
                         "expected_vram_gib": 80 if target == "a100" else 24}
    validate_configs(canonical, canonical_hw)
    cfg["rehearsal"] = True
    return cfg, hw


def score_saved_run(source, args):
    from mmdl.evaluation.reporting import replay_checker, saved_run, score_summary

    rows, identity = saved_run(source, args.suite)
    if args.checker_run:
        merged, checked = replay_checker(source / "predictions.jsonl", args.checker_run)
    else:
        state_path = source / "supplemental_pipeline.json"
        state = read_json(state_path) if state_path.exists() else {}
        source_hash = sha256_file(source / "predictions.jsonl")
        binding = {"source_predictions_sha256": source_hash, "checker_mode": args.checker_mode}
        if state and any(state.get(key) != value for key, value in binding.items()):
            raise ValueError("Saved supplemental pipeline differs from source or checker mode")
        if not state:
            directory = _subprocess_directory(_checker_command(
                "select", args, "--predictions", source / "predictions.jsonl",
                "--run-label", args.run_id, "--output-root", args.artifact_root / "judge", "--resume"))
            state = binding | {"checker_run": str(directory)}
            write_json(state_path, state)
        directory = Path(state["checker_run"])
        if not (directory / "judge.json").exists():
            command = _checker_command("judge", args, "--targets", directory / "targets.jsonl")
            if args.checkset_receipt:
                command += ["--checkset-receipt", str(args.checkset_receipt)]
            _subprocess_directory(command)
        if not (directory / "summary.json").exists():
            _subprocess_directory([
                sys.executable, "-m", "mmdl.evaluation.CompassVerifier_checker", "report",
                "--predictions", str(source / "predictions.jsonl"),
                "--targets", str(directory / "targets.jsonl"),
                "--verdicts", str(directory / "verdicts.jsonl")])
        merged, checked = replay_checker(source / "predictions.jsonl", directory)
        args.checker_run = directory
    if bool(checked["reportable"]) != (args.checker_mode == "official"):
        raise ValueError("Saved checker reportability differs from requested mode")
    identity["reportable"] = identity["reportable"] and checked["reportable"]
    report = score_summary(rows, identity, merged)
    report["checker_run"] = str(args.checker_run)
    write_json(source / "supplemental_summary.json", report)
    print(f"V8 {report['v8_correct']}/{report['denominator']}; "
          f"V8+checker {report['final_correct']}/{report['denominator']}; "
          f"reportable={str(report['reportable']).lower()}")
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"preview", "paired"}:
        return report_main(argv)
    parser, args = arguments(argv)
    root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("HF_HOME", str(default_path("HF_HOME", "mmdl-cache/huggingface")))
    cfg, hw = load_evaluation_configs(args)
    if suite_of(cfg) != args.suite:
        parser.error("--suite differs from the selected protocol")
    try:
        validate_execution(args, hw)
    except ValueError as exc:
        parser.error(str(exc))
    if args.score_only:
        from mmdl.evaluation.reporting import saved_run

        saved_run(args.artifact_root / "runs" / args.run_id, args.suite)
        _provision_checker(args)
        args.checkset_receipt = _checkset(args)
        return score_saved_run(args.artifact_root / "runs" / args.run_id, args)
    cache = default_path("HF_HOME", "mmdl-cache/huggingface")
    if args.base_path is None:
        args.base_path = cache / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.model == "finetuned":
        from mmdl.runtime.model_reference import resolve_export_reference

        args.model_ref, args.model_path, args.resolved_model_reference = resolve_export_reference(
            args.model_ref, Path(args.model_path) if args.model_path else None)
    args.model_path = args.model_path or os.environ.get("MMDL_MODEL_PATH", MODEL_ID)
    if args.model_path == MODEL_ID:
        args.model_path = cache / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
    else:
        args.model_path = Path(args.model_path).expanduser().resolve()
    if not args.model_ref.exists() or not (args.data_root / "manifest.json").exists():
        if args.no_download:
            parser.error("Verified model/data manifests missing; run python -m mmdl.data.download")
        standard_data = default_path("MMDL_DATA_ROOT", "mmdl-data") / SUITE_DATA_DIRS[args.suite]
        if args.data_root != standard_data:
            parser.error("Custom data-root must contain a prepared manifest and validation snapshot")
        from mmdl.data.download import prepare
        prepare(root, default_path("MMDL_DATA_ROOT", "mmdl-data"), cache, args.artifact_root,
                suite=args.suite)
    if args.score == "both":
        _provision_checker(args)
        args.checkset_receipt = _checkset(args)
        subprocess.run([sys.executable, "-m", "mmdl.evaluation.cli", *argv,
                        "--score", "v8"], check=True)
        return score_saved_run(args.artifact_root / "runs" / args.run_id, args)
    from mmdl.evaluation.engine import run
    summary = run(args, cfg, hw, root)
    print(f"{summary['status']}: {args.artifact_root / 'runs' / args.run_id / 'summary.json'}")
    return summary


def rescore_main(argv=None):
    """Re-evaluate raw responses without importing Torch or loading a model."""
    import yaml
    from collections import Counter
    from mmdl.evaluation.parsers import score_with_parser, scoring_source_files
    from mmdl.evaluation.writer import RunWriter, finalize, render
    from mmdl.runtime.artifacts import sha256_file

    parser = argparse.ArgumentParser(description="Rescore preserved responses into a separate artifact")
    parser.add_argument("--run-id")
    parser.add_argument("--source-run", type=Path, help="Read a preserved run at this path")
    parser.add_argument("--suite", choices=list(SUITE_COUNTS), default="mmmu-val")
    parser.add_argument("--score", choices=["v8", "both"], default="v8")
    parser.add_argument("--checker-run", type=Path,
                        help="Existing official checker ledger; replay only, no new judge inference")
    parser.add_argument("--artifact-root", type=Path, default=default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts"))
    parser.add_argument("--verify", action="store_true", help="Fail if current parsing differs")
    parser.add_argument("--parser", choices=["mmmu-official-no-random-v1", "team-final-answer-v4",
                                           "team-final-answer-v5", "team-final-answer-v6",
                                           "team-final-answer-v8"],
                        help="Default: the source run's saved scoring version")
    parser.add_argument("--output-dir", type=Path, help="Separate result directory; source rows remain immutable")
    args = parser.parse_args(argv)
    if bool(args.source_run) == bool(args.run_id):
        parser.error("Supply exactly one of --run-id or --source-run")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        parser.error("Invalid run ID")
    if args.score == "both" and args.checker_run is None:
        parser.error("Saved-response --score both requires --checker-run; no judge is started by rescore")
    source = args.source_run or args.artifact_root / "runs" / args.run_id
    source_cfg = yaml.safe_load((source / "resolved_eval_config.yaml").read_text())
    parser_id = args.parser or ("team-final-answer-v8" if args.score == "both" else source_cfg["parser"])
    if args.score == "both" and parser_id != "team-final-answer-v8":
        parser.error("Supplemental scoring requires the unchanged V8 parser")
    from mmdl.evaluation.reporting import saved_run

    rows, registration = saved_run(source, args.suite)
    scoring_files = scoring_source_files(parser_id)
    scoring_hash = digest([parser_id, *[sha256_file(path) for path in scoring_files]])
    destination = args.output_dir or source / "rescoring" / scoring_hash[:16]
    if args.output_dir and destination.exists():
        parser.error("Explicit output-dir must be new; refusing to overwrite")
    writer = RunWriter(destination, {"scoring_sha256": scoring_hash,
                      "parser": parser_id,
                      "source_sha256": sha256_file(source / "predictions.jsonl"),
                      **({"suite": args.suite, "id_subjects": registration["id_subjects"]}
                         if args.suite != "mmmu-val" else {})},
                       [r["id"] for r in rows], resume=destination.exists())
    changed = 0
    transitions = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
    differences = []
    stages: Counter[str] = Counter()
    try:
        write_json(destination / "source.json", {
            "source_run": str(source.resolve()), "source_predictions_sha256": sha256_file(source / "predictions.jsonl"),
            "images_resolve_relative_to": "source_run; images are not copied into this rescore-only artifact",
            "parser": parser_id, "scoring_sha256": scoring_hash,
        })
        (destination / "resolved_eval_config.yaml").write_text(yaml.safe_dump(source_cfg | {"parser": parser_id}))
        if parser_id == "team-final-answer-v8":
            import shutil
            from mmdl.evaluation.final_answer_parser_v8 import COMPARISON_POLICY
            from mmdl.runtime.artifacts import file_records

            root = Path(__file__).resolve().parents[3]
            for path in scoring_files:
                target = destination / "scorer_source" / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            write_json(destination / "scorer_source.json", {
                "parser": parser_id, "comparison_policy": COMPARISON_POLICY,
                "scoring_sha256": scoring_hash, "files": file_records(root, scoring_files),
            })
        for row in rows:
            score = score_with_parser(row["raw_response"], row["question_type"], row["options"], row["answer"],
                                      parser_id, row.get("finish_reason"), question=row.get("question", ""))
            different = any(score[key] != row[key] for key in ("parsed_answer", "parse_status", "correct"))
            changed += different
            transitions[f"{int(row['correct'])}->{int(score['correct'])}"] += 1
            stage = ("CORRECT_NOT_REASONING_CERTIFIED" if score["correct"] else
                     "LENGTH_CENSORED_UNRESOLVED" if row.get("finish_reason") == "length" else
                     "ANSWER_EXTRACTION_UNRESOLVED" if score["parse_status"] != "PARSED" else
                     "OPEN_FINAL_MISMATCH_REVIEW" if row["question_type"] == "open" else
                     "EXPLICIT_MCQ_MISMATCH_REVIEW")
            stages[stage] += 1
            if different:
                differences.append({"id": row["id"], "old_answer": row["parsed_answer"],
                                    "new_answer": score["parsed_answer"], "old_correct": row["correct"],
                                    "new_correct": score["correct"], "finish_reason": row["finish_reason"]})
            if row["id"] not in writer.completed:
                writer.save(row | score)
        summary = finalize(destination, [r["id"] for r in rows], "full", suite=args.suite,
                           id_subjects=registration["id_subjects"])
        if parser_id == "team-final-answer-v8":
            render(destination, include_images=False)
        write_json(destination / "comparison.json", {"changed_samples": changed, "scoring_sha256": scoring_hash,
                                                    "parser": parser_id, "transitions": transitions,
                                                    "observed_stage_counts": dict(stages),
                                                    "causal_status": "Stages are observable flags, not proof of visual or reasoning causes",
                                                    "differences": differences})
        print(f"Rescored {len(rows)} responses; changed={changed}; {destination / 'summary.json'}")
        if args.score == "both":
            from mmdl.evaluation.reporting import replay_checker, score_summary

            merged, checked = replay_checker(source / "predictions.jsonl", args.checker_run)
            if not checked["reportable"]:
                raise ValueError("Historical score replay requires the official 7B ledger")
            report = score_summary(rows, registration, merged)
            report["checker_run"] = str(args.checker_run.resolve())
            write_json(destination / "supplemental_summary.json", report)
        if args.verify and changed:
            raise SystemExit(1)
        return summary
    finally:
        writer.close()


def report_main(argv):
    from mmdl.evaluation.reporting import paired_report, saved_run, score_summary, write_report

    parser = argparse.ArgumentParser(description="Read-only saved-response preview and paired reports")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "paired"):
        sub = commands.add_parser(command)
        sub.add_argument("--suite", choices=list(SUITE_COUNTS), required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if command == "preview":
            sub.add_argument("--source-run", type=Path, required=True)
        else:
            sub.add_argument("--base-run", type=Path, required=True)
            sub.add_argument("--finetuned-run", type=Path, required=True)
            sub.add_argument("--base-checker", type=Path)
            sub.add_argument("--finetuned-checker", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preview":
        rows, identity = saved_run(args.source_run, args.suite, preview=True)
        report = score_summary(rows, identity)
    else:
        report = paired_report(args.base_run, args.finetuned_run, args.suite,
                               base_checker=args.base_checker, finetuned_checker=args.finetuned_checker)
    destination = write_report(args.output_dir, report)
    if args.command == "preview":
        import json
        from mmdl.runtime.artifacts import atomic_text

        atomic_text(destination / "predictions.jsonl", "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        write_json(destination / "preview_manifest.json", {
            "reportable": False, "source": identity,
            "predictions_sha256": sha256_file(destination / "predictions.jsonl"),
            "record_hashes": {row["id"]: row["_record_sha256"] for row in rows},
        })
    print(destination / "summary.json")
    return report


if __name__ == "__main__":
    main()
