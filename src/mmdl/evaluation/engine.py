"""Run local Qwen, preserve each response, and keep gold outside inference."""

import importlib.metadata
import hashlib
import io
import json
import re
import resource
import subprocess
import sys
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mmdl.evaluation.datasets.mmmu import load_validation, separate_sample
from mmdl.evaluation.parsers import score_response
from mmdl.evaluation.prompt import build_messages
from mmdl.evaluation.writer import RunWriter, finalize, render
from mmdl.runtime.artifacts import atomic_bytes, atomic_text, digest, file_records, git_commit, read_json, sha256_file, write_json
from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION, sample_seed
from mmdl.runtime.environment import check_storage, collect_environment


def verify_model(model_path, manifest):
    kind = manifest["kind"]
    if kind == "base":
        if manifest["repo_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
            raise ValueError("Wrong official baseline model manifest")
    elif kind in {"full", "merged", "adapter"}:
        if manifest.get("base") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
            raise ValueError("Derived checkpoint must identify the fixed official base")
        required = {"train_config_sha256", "data_manifest_sha256", "code_commit", "artifact_revision"}
        if any(not manifest.get(key) for key in required):
            raise ValueError("Derived checkpoint lacks training/artifact provenance")
    else:
        raise ValueError("Unknown model manifest kind")
    if manifest["dtype"] != "bfloat16" or manifest["processor_revision"] != MODEL_REVISION:
        raise ValueError("Evaluation requires BF16 and the unchanged pinned processor")
    for entry in manifest["files"]:
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise ValueError("Unsafe model artifact path")
        path = model_path / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Model file integrity failure: {entry['path']}")


def environment_identity(environment):
    return dict(python=environment["python"], platform=environment["platform"],
                packages={d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()},
                torch_cuda=environment["torch"]["cuda_runtime"],
                gpu=[{k: v for k, v in gpu.items() if k not in {"memory_free_mib"}}
                     for gpu in environment["nvidia_smi"]["gpus"]])


def code_records(root):
    paths = list((root / "src").rglob("*.py")) + list((root / "scripts").glob("*.sh"))
    paths += list((root / "scripts").glob("*.py"))
    paths += [root / "third_party/mmmu/eval_utils.py", root / "env/requirements-eval.lock",
              root / "pyproject.toml"]
    paths += list((root / "env").glob("requirements-vllm.lock"))
    return file_records(root, paths)


def validate_batch_outputs(requests, outputs):
    if len(outputs) != len(requests) or [item.get("sample_id") for item in outputs] != [
            item["sample_id"] for item in requests]:
        raise ValueError("Backend output count/order/sample IDs differ from submitted requests")


def prepare_request(row, subject, artifact_root, run_dir, cfg):
    sample, images, gold = separate_sample(row, subject)
    messages = build_messages(sample, images, run_dir)
    serialized = [{"role": "user", "content": []}]
    image_files = []
    for image in images:
        encoded_image = io.BytesIO()
        image.save(encoded_image, format="PNG")
        image_bytes = encoded_image.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        path = artifact_root / "assets/images" / f"{image_hash}.png"
        if path.exists():
            if sha256_file(path) != image_hash:
                raise ValueError("Existing inspection image differs from current model input")
        else:
            atomic_bytes(path, image_bytes)
        relative = f"../../assets/images/{image_hash}.png"
        image_files.append(relative)
        serialized[0]["content"].append({"type": "image", "image": relative, "sha256": image_hash})
    serialized[0]["content"].append(messages[0]["content"][-1])
    seed = sample_seed(cfg["generation"]["seed"], sample["id"])
    request = dict(sample_id=sample["id"], messages=messages, images=images, seed=seed)
    record = sample | gold | dict(seed=seed, messages=serialized, images=image_files,
                                 prompt=messages[0]["content"][-1]["text"],
                                 inference_id=uuid.uuid4().hex,
                                 inference_started_at=datetime.now(timezone.utc).isoformat())
    return request, record


def publish(public_root, run_dir, summary, environment, cfg, hw, identity):
    destination = public_root / run_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "summary.json", summary)
    write_json(destination / "environment.json", environment)
    atomic_text(destination / "resolved_eval_config.yaml", yaml.safe_dump(cfg, sort_keys=False))
    atomic_text(destination / "resolved_hardware_config.yaml", yaml.safe_dump(hw, sort_keys=False))
    atomic_text(destination / "subject_scores.csv", (run_dir / "subject_scores.csv").read_text())
    external = file_records(run_dir, [p for p in run_dir.iterdir() if p.is_file() and not p.name.startswith(".")])
    write_json(destination / "run_manifest.json", {"identity": identity,
               "external_root": f"MMDL_ARTIFACT_ROOT/runs/{run_dir.name}", "external_files": external})
    predictions = []
    for path in sorted((run_dir / "samples").glob("*.json")):
        row = read_json(path)
        predictions.append({k: row[k] for k in ("id", "subject", "correct", "parse_status")}
                           | {"external_record": str(path.relative_to(run_dir)), "sha256": sha256_file(path)})
    atomic_text(destination / "predictions.jsonl", "".join(json.dumps(x) + "\n" for x in predictions))
    failures = [{"external_record": str(p.relative_to(run_dir)), "sha256": sha256_file(p)}
                for p in sorted((run_dir / "external_failures").glob("*.json"))]
    atomic_text(destination / "failures.jsonl", "".join(json.dumps(x) + "\n" for x in failures))


def run(args, cfg, hw, root):
    start = time.monotonic()
    commit = git_commit(root, args.require_commit)
    environment = collect_environment()
    if not environment["torch"].get("bf16_probe", {}).get("passed"):
        raise RuntimeError("Doctor BF16 probe failed")
    storage = check_storage([args.artifact_root, args.data_root], 4 * 1024**3)
    if not storage["sufficient"]:
        raise RuntimeError("Storage headroom is below the recorded reserve + run budget")
    model_manifest = read_json(args.model_ref)
    verify_model(args.model_path, model_manifest)
    base_manifest = read_json(root / "manifests/models/baseline.json")
    if model_manifest["kind"] != "base":
        verify_model(args.base_path, base_manifest)
    datasets, coverage = load_validation(args.data_root)
    lookup = {sid: (subject, index) for subject, dataset in datasets.items()
              for index, sid in enumerate(dataset["id"])}
    ids = sorted(lookup)
    if args.mode == "full":
        if cfg["status"] != "FROZEN" or args.subject or args.limit or args.sample_id:
            raise ValueError("Full run requires frozen protocol and all 900 samples")
    else:
        if args.subject:
            ids = [sid for sid in ids if lookup[sid][0] == args.subject]
        if args.sample_id:
            if not set(args.sample_id) <= set(ids):
                raise ValueError("Smoke sample IDs not present in selected data")
            ids = sorted(args.sample_id)
        if args.limit:
            ids = ids[:args.limit]
        if not ids:
            raise ValueError("Empty smoke selection")
    records = code_records(root)
    protocol_files = file_records(root, list((root / "prompts").glob("mmmu_*_v1.txt")) +
                                 [root / "src/mmdl/evaluation/prompt.py",
                                  root / "src/mmdl/evaluation/parsers.py",
                                  root / "third_party/mmmu/eval_utils.py"])
    data_manifest = read_json(args.data_root / "manifest.json")
    processor_files = [item for item in base_manifest["files"] if not item["path"].endswith(".safetensors")]
    protocol_hash = digest(dict(config=cfg, files=protocol_files, processor=processor_files,
                                dataset=data_manifest["files"]))
    identity = dict(model_sha256=digest(model_manifest["files"]), protocol_sha256=protocol_hash,
                    code_sha256=digest(records), environment_sha256=digest(environment_identity(environment)),
                    hardware=hw, ids_sha256=digest(ids), mode=args.mode,
                    git_commit=commit, job_id=args.job_id, run_role=args.run_role)
    run_dir = args.artifact_root / "runs" / args.run_id
    writer = RunWriter(run_dir, identity, ids, resume=args.resume)
    try:
        if (run_dir / "summary.json").exists():
            previous = read_json(run_dir / "summary.json")
            if previous.get("status") in {"COMPLETE", "SMOKE", "PARTIAL"}:
                if set(writer.completed) != set(ids) or previous.get("completed_count") != len(ids):
                    raise ValueError("Previously completed run is missing immutable sample records")
                print("Existing completed run verified; no generation repeated", flush=True)
                return previous
        write_json(run_dir / "environment.json", environment)
        write_json(run_dir / "coverage.json", coverage)
        write_json(run_dir / "code_files.json", records)
        write_json(run_dir / "locations.json", dict(model=str(args.model_path), data=str(args.data_root)))
        if not (run_dir / "command.json").exists():
            reproduce = (
                f"bash scripts/eval.sh --protocol configs/eval/{Path(args.protocol).name} "
                f"--hardware configs/hardware/{hw['name']}.yaml "
                "--model-ref manifests/models/baseline.json "
                f'--model-path "$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/{MODEL_REVISION}" '
                '--data-root "$MMDL_DATA_ROOT/evaluation/mmmu" '
                f"--run-id {args.run_id} --mode {args.mode}"
            )
            if args.subject:
                reproduce += f" --subject {args.subject}"
            if args.limit:
                reproduce += f" --limit {args.limit}"
            for sid in args.sample_id or []:
                reproduce += f" --sample-id {sid}"
            if args.require_commit:
                reproduce += f" --require-commit {args.require_commit}"
            if args.job_id:
                reproduce += f" --job-id {args.job_id} --run-role {args.run_role}"
            write_json(run_dir / "command.json", {"actual_argv": sys.argv, "reproduce_command": reproduce})
        for name in ("mmmu_mcq_v1.txt", "mmmu_open_v1.txt"):
            target = run_dir / "prompts" / name
            if not target.exists():
                atomic_text(target, (root / "prompts" / name).read_text())
        for name, value in (("resolved_eval_config.yaml", cfg), ("resolved_hardware_config.yaml", hw)):
            if not (run_dir / name).exists():
                atomic_text(run_dir / name, yaml.safe_dump(value, sort_keys=False))
        # New repos may have no HEAD. Capture exact uncommitted source alongside file hashes.
        if not (run_dir / "source.patch").exists():
            patch = ""
            for item in records:
                result = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", item["path"]],
                                        cwd=root, capture_output=True, text=True)
                if result.returncode not in (0, 1):
                    raise RuntimeError("Could not preserve source patch")
                patch += result.stdout
            atomic_text(run_dir / "source.patch", patch)
        if cfg["execution"]["backend"] == "vllm":
            from mmdl.evaluation.backends.vllm_backend import VLLMBackend
            backend = VLLMBackend(str(args.model_path), cfg, hw, model_manifest["kind"], str(args.base_path))
        else:
            from mmdl.evaluation.backends.transformers_backend import TransformersBackend
            backend = TransformersBackend(str(args.model_path), cfg, hw, model_manifest["kind"],
                                          str(args.base_path))
        backend_details = backend.describe()
        backend_path = run_dir / "backend.json"
        if backend_path.exists():
            previous_backend = read_json(backend_path)
            for key in ("device_map", "generation_config", "image_processor", "chat_template_sha256"):
                if backend_details[key] != previous_backend[key]:
                    raise ValueError(f"Resume actual backend mismatch: {key}")
        else:
            write_json(backend_path, backend_details)
        evaluation_start = time.monotonic()
        invocation_id = uuid.uuid4().hex
        invocation_path = run_dir / "inference_invocations" / f"{invocation_id}.json"
        invocation = dict(id=invocation_id, run_id=args.run_id, run_role=args.run_role,
                          git_commit=commit, started_at=datetime.now(timezone.utc).isoformat(),
                          attempted_ids=[], completed_ids=[], status="RUNNING")
        write_json(invocation_path, invocation)
        failed = None
        pending = [sid for sid in ids if sid not in writer.completed]
        batch_size = cfg["execution"]["batch_size"]
        continuous = cfg["execution"].get("scheduling") == "continuous"
        active_records = {}

        def request_stream():
            for sid in pending:
                request_start = time.monotonic()
                invocation["attempted_ids"].append(sid)
                write_json(invocation_path, invocation)
                subject, index = lookup[sid]
                request, record = prepare_request(datasets[subject][index], subject,
                                                  args.artifact_root, run_dir, cfg)
                active_records[sid] = (record, request_start)
                yield request

        def completed_records():
            if continuous:
                with closing(backend.generate_stream(request_stream())) as stream:
                    for output in stream:
                        sid = output.get("sample_id")
                        if sid not in active_records:
                            raise ValueError("Streaming backend returned an unknown or duplicate sample ID")
                        record, request_start = active_records.pop(sid)
                        yield record, output, dict(
                            total_sample_seconds=time.monotonic() - request_start,
                            total_sample_timing_policy="request_latency_including_prepare; overlaps_other_requests")
            else:
                for offset in range(0, len(pending), batch_size):
                    batch_ids = pending[offset:offset + batch_size]
                    batch_start = time.monotonic()
                    requests, records_for_batch = [], []
                    for sid in batch_ids:
                        invocation["attempted_ids"].append(sid)
                        write_json(invocation_path, invocation)
                        subject, index = lookup[sid]
                        request, record = prepare_request(datasets[subject][index], subject,
                                                          args.artifact_root, run_dir, cfg)
                        requests.append(request)
                        records_for_batch.append(record)
                    outputs = backend.generate_batch(requests)
                    validate_batch_outputs(requests, outputs)
                    batch_seconds = time.monotonic() - batch_start
                    for record, output in zip(records_for_batch, outputs, strict=True):
                        yield record, output, dict(
                            total_sample_seconds=batch_seconds / len(batch_ids),
                            total_sample_timing_policy="batch_wall_divided_by_size",
                            batch_ids=batch_ids, batch_wall_seconds=batch_seconds)

        try:
            with closing(completed_records()) as completed:
                for record, output, timing in completed:
                    sid = record["id"]
                    score = score_response(output["raw_response"], record["question_type"],
                                           record["options"], record["answer"])
                    raw = output["raw_response"]
                    just_answer = bool(re.fullmatch(
                        r"\s*(?:(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?:is|:)\s*)?"
                        r"[*\s]*\(?[A-Z]\)?[*\.\s]*", raw, re.IGNORECASE
                    )) if record["question_type"] == "multiple-choice" else len(raw.split()) <= 3
                    row = record | output | score | timing | dict(
                        model_explanation=None if just_answer else raw or None,
                        inference_invocation_id=invocation_id, run_id=args.run_id,
                        explanation_policy="verbatim narrative, no synthesized explanation; answer-only/short open answer has none; full raw always retained",
                        status="COMPLETED")
                    writer.save(row)
                    invocation["completed_ids"].append(sid)
                    write_json(invocation_path, invocation)
                    print(json.dumps(dict(id=sid, completed=len(writer.completed), total=len(ids),
                                          tokens=output["generated_tokens"], seconds=round(output["generation_seconds"], 3),
                                          scheduling="continuous" if continuous else "batch",
                                          max_concurrency=batch_size)), flush=True)
            if invocation["attempted_ids"] != pending or set(invocation["completed_ids"]) != set(pending):
                raise RuntimeError("Backend stopped before every pending request completed")
        except Exception as exc:
            for sid in invocation["attempted_ids"]:
                if sid not in writer.completed and not (run_dir / "external_failures" / f"{sid}.json").exists():
                    writer.failure(sid, exc)
            failed = exc
        invocation.update(status="FAILED" if failed else "COMPLETED",
                          ended_at=datetime.now(timezone.utc).isoformat())
        write_json(invocation_path, invocation)
        summary = finalize(run_dir, ids, args.mode)
        summary.update(evaluation_seconds_this_invocation=time.monotonic() - evaluation_start,
                       model_load_seconds=backend.load_seconds,
                       total_seconds_this_invocation=time.monotonic() - start,
                       sample_timing_policy=("generation: engine_step_wall_divided_by_active_requests; "
                                             "sample: overlapping request latency; use evaluation_seconds_this_invocation for throughput"
                                             if continuous else "batch_wall_divided_by_size; not individual request latency"),
                       ram_measurement_scope="parent_process_only; excludes vLLM workers",
                       peak_ram_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       peak_vram_allocated_bytes=max((r["peak_vram_allocated_bytes"] for r in writer.completed.values()
                                                     if r.get("peak_vram_allocated_bytes") is not None), default=None),
                       peak_vram_reserved_bytes=max((r["peak_vram_reserved_bytes"] for r in writer.completed.values()
                                                    if r.get("peak_vram_reserved_bytes") is not None), default=None),
                       max_observed_device_memory_used_bytes=max((r["observed_device_memory_used_bytes"]
                            for r in writer.completed.values() if r.get("observed_device_memory_used_bytes") is not None), default=None))
        write_json(run_dir / "summary.json", summary)
        render(run_dir)
        publish(args.public_root, run_dir, summary, environment, cfg, hw, identity)
        if failed:
            raise RuntimeError(f"Run incomplete after preserved system failure: {type(failed).__name__}: {failed}") from failed
        return summary
    except Exception as exc:
        # Persist setup/load failures too, without declaring any inference completed.
        if not (run_dir / "summary.json").exists():
            pending = next((sid for sid in ids if sid not in writer.completed), None)
            if pending is not None and not (run_dir / "external_failures" / f"{pending}.json").exists():
                writer.failure(pending, exc)
            summary = finalize(run_dir, ids, args.mode)
            summary.update(total_seconds_this_invocation=time.monotonic() - start,
                           peak_ram_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            write_json(run_dir / "summary.json", summary)
            render(run_dir)
            publish(args.public_root, run_dir, summary, environment, cfg, hw, identity)
        raise
    finally:
        writer.close()
