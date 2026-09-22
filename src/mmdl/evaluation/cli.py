"""Evaluation and saved-response rescoring commands; never train or augment."""

import argparse
import os
import re
from pathlib import Path

from mmdl.runtime.artifacts import default_path, digest, read_json, write_json
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION, SUBJECTS, load_configs


def validate_execution(args, hw):
    if args.mode == "full" and hw["placement"] != "gpu_only":
        raise ValueError("Local CPU-offload full evaluation is prohibited; use the GPU-only reproduction path")
    if args.job_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", args.job_id):
        raise ValueError("Invalid job ID")
    if args.run_role in {"assignment", "analysis"}:
        if args.mode != "full" or not args.job_id or args.run_id != f"{args.job_id}-{args.run_role}":
            raise ValueError("Assignment/analysis roles require a full run and the matching job-role run ID")
        if not args.require_commit:
            raise ValueError("Assignment/analysis runs require a fixed Git commit")


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=root / "configs/eval/mmmu_val_v1.yaml")
    parser.add_argument("--hardware", type=Path, default=root / "configs/hardware/rtx5060_8gb.yaml")
    parser.add_argument("--model-ref", type=Path, default=root / "manifests/models/baseline.json")
    parser.add_argument("--model-path", default=os.environ.get("MMDL_MODEL_PATH", MODEL_ID))
    parser.add_argument("--base-path", type=Path, help="Pinned official base snapshot for a derived checkpoint")
    parser.add_argument("--data-root", type=Path,
                        default=default_path("MMDL_DATA_ROOT", "mmdl-data") / "evaluation/mmmu")
    parser.add_argument("--artifact-root", type=Path, default=default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts"))
    parser.add_argument("--public-root", type=Path, default=root / "results")
    parser.add_argument("--job-id")
    parser.add_argument("--run-role", choices=["standalone", "smoke", "assignment", "analysis"], default="standalone")
    parser.add_argument("--require-commit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "partial", "full"], default="full")
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        parser.error("run-id must be a safe directory name")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.sample_id and len(args.sample_id) != len(set(args.sample_id)):
        parser.error("sample-id values must be unique")
    cfg, hw = load_configs(args.protocol, args.hardware)
    try:
        validate_execution(args, hw)
    except ValueError as exc:
        parser.error(str(exc))
    cache = default_path("HF_HOME", "mmdl-cache/huggingface")
    if args.base_path is None:
        args.base_path = cache / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.model_path == MODEL_ID:
        args.model_path = cache / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
    else:
        args.model_path = Path(args.model_path).expanduser().resolve()
    if not args.model_ref.exists() or not (args.data_root / "manifest.json").exists():
        if args.no_download:
            parser.error("Verified model/data manifests missing; run python -m mmdl.data.download")
        if args.data_root != default_path("MMDL_DATA_ROOT", "mmdl-data") / "evaluation/mmmu":
            parser.error("Custom data-root must contain a prepared manifest and validation snapshot")
        from mmdl.data.download import prepare
        prepare(root, default_path("MMDL_DATA_ROOT", "mmdl-data"), cache, args.artifact_root)
    from mmdl.evaluation.engine import run
    summary = run(args, cfg, hw, root)
    print(f"{summary['status']}: {args.artifact_root / 'runs' / args.run_id / 'summary.json'}")


def rescore_main():
    """Re-evaluate raw responses without importing Torch or loading a model."""
    from mmdl.evaluation.parsers import score_response
    from mmdl.evaluation.writer import RunWriter, finalize
    from mmdl.runtime.artifacts import sha256_file

    parser = argparse.ArgumentParser(description="Rescore preserved responses into a separate artifact")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts"))
    parser.add_argument("--verify", action="store_true", help="Fail if current parsing differs")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        parser.error("Invalid run ID")
    source = args.artifact_root / "runs" / args.run_id
    rows = [read_json(p) for p in sorted((source / "samples").glob("*.json"))]
    if len(rows) != 900 or read_json(source / "summary.json")["status"] != "COMPLETE":
        parser.error("Final rescoring requires a complete original 900-sample run")
    parser_path = Path(__file__).parent / "parsers.py"
    official_path = Path(__file__).resolve().parents[3] / "third_party/mmmu/eval_utils.py"
    scoring_hash = digest([sha256_file(parser_path), sha256_file(official_path)])
    destination = source / "rescoring" / scoring_hash[:16]
    writer = RunWriter(destination, {"scoring_sha256": scoring_hash,
                      "source_sha256": sha256_file(source / "predictions.jsonl")},
                       [r["id"] for r in rows], resume=destination.exists())
    changed = 0
    try:
        for row in rows:
            score = score_response(row["raw_response"], row["question_type"], row["options"], row["answer"])
            changed += any(score[key] != row[key] for key in score)
            if row["id"] not in writer.completed:
                writer.save(row | score)
        summary = finalize(destination, [r["id"] for r in rows], "full")
        write_json(destination / "comparison.json", {"changed_samples": changed, "scoring_sha256": scoring_hash})
        print(f"Rescored {len(rows)} responses; changed={changed}; {destination / 'summary.json'}")
        if args.verify and changed:
            raise SystemExit(1)
        return summary
    finally:
        writer.close()


if __name__ == "__main__":
    main()
