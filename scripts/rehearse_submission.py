"""Bounded, nonreportable GPU rehearsal of public Pro commands and pinned engine parity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json

REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def _tiny_factory(path, arm, protocol):
    from fine_tuning.error_note.v18b_rehearsal_eval import _tiny_factory as factory
    return factory(path, arm, protocol)


def amplify_fixture_vision(bundle) -> dict:
    """Change only the synthetic fixture's Vision B matrices; verifier thresholds stay fixed."""
    import torch
    from fine_tuning.modules import tensor_sha256

    parameters = dict(bundle.model.named_parameters())
    names = {row["name"] for row in bundle.inventory["trainable"]
             if row["group"] == "V" and ".lora_B." in row["name"]}
    if not names:
        raise ValueError("Synthetic fixture requires Vision LoRA B tensors")
    before = {name: tensor_sha256(value) for name, value in parameters.items()}
    with torch.no_grad():
        for name in names:
            parameters[name].mul_(256)
    after = {name: tensor_sha256(value) for name, value in parameters.items()}
    changed = {name for name in before if before[name] != after[name]}
    if not changed or not changed <= names:
        raise ValueError("Synthetic fixture must change only Vision LoRA B")
    return {"kind": "synthetic-vision-positive-control", "reportable": False,
            "vision_B_gain": 256, "changed_parameters": sorted(changed),
            "before_sha256": {name: before[name] for name in sorted(names)},
            "after_sha256": {name: after[name] for name in sorted(names)}}


def prepare(repo: Path, output: Path, checkpoint: Path) -> dict:
    """Recreate deterministic fixture weights and reuse an immutable real tiny checkpoint."""
    import yaml
    from fine_tuning.tests.test_error_note_single_vllm_lora_gpu import build_tiny
    from fine_tuning.notebook_rl.checkpoint import capture_rng, load_eval, make_optimizer, read_metadata, save
    from fine_tuning.notebook_rl.export import export_merged, validate_export_identity
    from fine_tuning.notebook_rl.proof_eval import lora_section

    output.mkdir(parents=True, exist_ok=False)
    base = output / "base" / REVISION
    build_tiny(base, printable_output=True)
    base_ref = {"kind": "base", "repo_id": "Qwen/Qwen3-VL-4B-Instruct", "revision": REVISION,
                "dtype": "bfloat16", "processor_revision": REVISION, "reportable": False,
                "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                          for p in sorted(base.iterdir()) if p.is_file()]}
    manifest = output / "tiny-base.json"
    write_json(manifest, base_ref)
    raw = read_json(checkpoint / "metadata.json")
    metadata = read_metadata(checkpoint, binding=raw["binding"])
    original_checkpoint = checkpoint
    bundle = load_eval(base, original_checkpoint, binding=metadata["binding"], _test_policy_factory=_tiny_factory)
    calibration = amplify_fixture_vision(bundle)
    calibration["original_checkpoint_files"] = {p.name: sha256_file(p) for p in sorted(original_checkpoint.iterdir())}
    # A separately named step-zero fixture is never represented as another training update.
    checkpoint = output / "synthetic-checkpoint"
    optimizer, scheduler = make_optimizer(bundle, metadata["protocol"])
    metadata = save(bundle, optimizer, scheduler, 0,
                    {"epoch": 0, "position": 0, "discarded_problem_ids": [], "rehearsal_fixture": calibration},
                    capture_rng(), checkpoint, protocol=metadata["protocol"], binding=metadata["binding"])
    del bundle, optimizer, scheduler
    write_json(output / "fixture-calibration.json", calibration)
    derived = output / "derived"
    derived.mkdir()
    model = derived / "model"
    exported = export_merged(checkpoint, base, model, metadata["arm"], binding=metadata["binding"],
                             llm_lora=True, _test_policy_factory=_tiny_factory, _test_base_manifest=manifest)
    identity = validate_export_identity(model, metadata["arm"])
    reference = {"kind": "unmerged-lora", "base": {"id": base_ref["repo_id"], "revision": REVISION},
                 "dtype": "bfloat16", "processor_revision": REVISION, "usage": "evaluation_only",
                 "artifact_revision": sha256_file(model / "export_manifest.json"),
                 "train_config_sha256": metadata["resolved_protocol_sha256"],
                 "data_manifest_sha256": metadata["binding"]["data_sha256"],
                 "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                 "rehearsal_fixture": calibration,
                 "llm_lora": lora_section(),
                 "files": [{"path": p.relative_to(model).as_posix(), "bytes": p.stat().st_size,
                            "sha256": sha256_file(p)} for p in sorted(model.rglob("*")) if p.is_file()],
                 "export": {key: exported[key] for key in
                            ("arm", "binding", "checkpoint_files", "base_manifest_sha256", "mergers", "precision_policy")}}
    reference["export"].update(export_manifest_sha256=identity["manifest_sha256"],
                               checkpoint_state_sha256=digest(exported["checkpoint_state_sha256"]))
    write_json(derived / "model_ref.json", reference)
    selector = output / "tiny-derived.json"
    write_json(selector, {"kind": "derived-rehearsal", "reportable": False,
                          "reference_sha256": sha256_file(derived / "model_ref.json"),
                          "export_manifest_sha256": reference["artifact_revision"]})
    cfg = yaml.safe_load((repo / "configs/eval/mmmu_pro_standard10_v8_a100.yaml").read_text())
    cfg["generation"]["max_new_tokens"] = 16
    cfg["execution"].update(batch_size=2, gpu_memory_utilization=.62)
    (output / "tiny-pro.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    hw = yaml.safe_load((repo / "configs/hardware/rtx5060_8gb.yaml").read_text())
    hw.update(placement="gpu_only", min_free_gpu_gib=1.0)
    (output / "tiny-hardware.yaml").write_text(yaml.safe_dump(hw, sort_keys=False))
    value = {"reportable": False, "base": str(base), "base_ref": str(manifest),
             "derived": str(derived), "selector": str(selector), "checkpoint": str(checkpoint),
             "original_checkpoint": str(original_checkpoint), "calibration": calibration,
             "checkpoint_files": {p.name: sha256_file(p) for p in sorted(checkpoint.iterdir())},
             "protocol": str(output / "tiny-pro.yaml"), "hardware": str(output / "tiny-hardware.yaml")}
    write_json(output / "prepared.json", value)
    return value


def unchanged_records(before: dict[str, str], directory: Path) -> dict[str, str]:
    after = {p.name: sha256_file(p) for p in directory.glob("*.json")}
    if not before or any(after.get(name) != checksum for name, checksum in before.items()):
        raise ValueError("Completed responses changed or disappeared across resume")
    return after


def _kill_owned(process: subprocess.Popen, ticks: str) -> None:
    info = Path("/proc") / str(process.pid)
    if process.poll() is None and info.exists():
        if (info / "stat").read_text().split()[21] != ticks or os.getpgid(process.pid) != process.pid:
            raise RuntimeError("Owned rehearsal process identity changed")
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def run(repo: Path, output: Path, data_root: Path, deadline_seconds: int) -> dict:
    if not 0 < deadline_seconds <= 1800:
        raise ValueError("Rehearsal deadline must be within 30 minutes")
    prepared = read_json(output / "prepared.json")
    for name, checksum in prepared["checkpoint_files"].items():
        if sha256_file(Path(prepared["checkpoint"]) / name) != checksum:
            raise ValueError("Archived tiny checkpoint changed")
    from mmdl.evaluation.datasets.mmmu_pro import load_test
    datasets, _ = load_test(data_root)
    ids = sorted(sid for dataset in datasets.values() for sid in dataset["id"])
    selected = list(dict.fromkeys(["validation_Pharmacy_4", "validation_Pharmacy_22", "test_Geography_252", *ids[:3]]))
    if not set(selected) <= set(ids):
        raise ValueError("Required real Pro image cases are missing")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo), str(repo / "src"))),
               HF_HOME=str(Path.home() / "mmdl-cache/huggingface"), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", MMDL_PYTHON=sys.executable,
               VLLM_ENABLE_V1_MULTIPROCESSING="0")
    started = time.monotonic()
    deadline = started + deadline_seconds
    receipt = {"reportable": False, "status": "RUNNING", "stages": [], "selected_ids": selected,
               "prepared_sha256": sha256_file(output / "prepared.json")}

    def execute(label: str, command: list[str], interrupt: Path | None = None) -> dict:
        before = time.monotonic()
        with (output / f"{label}.log").open("x") as stream:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            ticks = (Path("/proc") / str(process.pid) / "stat").read_text().split()[21]
            saved = {}
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Rehearsal deadline during {label}")
                    if interrupt is not None:
                        saved = {p.name: sha256_file(p) for p in interrupt.glob("*.json")}
                        if saved:
                            _kill_owned(process, ticks)
                            break
                    time.sleep(.005 if interrupt else .1)
            finally:
                if process.poll() is None:
                    _kill_owned(process, ticks)
        result = {"stage": label, "command": command, "returncode": process.returncode,
                  "seconds": time.monotonic() - before, "saved_records": saved}
        receipt["stages"].append(result)
        write_json(output / "rehearsal-receipt.json", receipt)
        if interrupt is not None:
            if not saved or process.returncode != -signal.SIGKILL:
                raise RuntimeError(f"{label} did not reach the durable interruption boundary")
        elif process.returncode:
            raise RuntimeError(f"{label} failed; inspect {output / (label + '.log')}")
        return result

    try:
        common = ["--suite", "mmmu-pro", "--protocol", prepared["protocol"], "--hardware", prepared["hardware"],
                  "--base-path", prepared["base"], "--data-root", str(data_root), "--artifact-root", str(output / "artifacts"),
                  "--public-root", str(output / "public"), "--rehearsal", "--mode", "smoke", "--score", "v8"]
        for sid in selected:
            common += ["--sample-id", sid]
        for arm in ("base", "finetuned"):
            prefix = ["bash", str(repo / "scripts" / ("eval.sh" if arm == "base" else "eval_finetuned.sh"))]
            command = prefix + common + ["--run-id", f"rehearsal-{arm}", "--model-ref",
                prepared["base_ref"] if arm == "base" else prepared["selector"], "--model-path",
                prepared["base"] if arm == "base" else prepared["derived"]]
            samples = output / "artifacts/runs" / f"rehearsal-{arm}" / "samples"
            interrupted = execute(f"{arm}-interrupt", command, samples)
            execute(f"{arm}-resume", command + ["--resume"])
            after = unchanged_records(interrupted["saved_records"], samples)
            if len(after) != len(selected):
                raise ValueError(f"{arm} resumed response count differs")
            receipt[arm] = {"saved_records": after, "duplicate_count": 0}
            for engine in ("reference", "submission"):
                probe = [sys.executable, "-m", "mmdl.evaluation.backend_parity", "run", "--engine", engine,
                         "--protocol", prepared["protocol"], "--hardware", prepared["hardware"],
                         "--model-path", prepared["base"] if arm == "base" else str(Path(prepared["derived"]) / "model"),
                         "--rehearsal", "--output", str(output / f"{arm}-{engine}.json")]
                if arm == "finetuned":
                    probe += ["--base-path", prepared["base"], "--model-ref", str(Path(prepared["derived"]) / "model_ref.json")]
                execute(f"{arm}-parity-{engine}", probe)
            execute(f"{arm}-parity-compare", [sys.executable, "-m", "mmdl.evaluation.backend_parity", "compare",
                    "--reference", str(output / f"{arm}-reference.json"), "--candidate", str(output / f"{arm}-submission.json"),
                    "--output", str(output / f"{arm}-comparison.json")])
            if read_json(output / f"{arm}-comparison.json").get("pass") is not True:
                raise ValueError(f"{arm} engine comparison did not pass")
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "rehearsal-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.is_relative_to(repo):
        parser.error("Rehearsal artifacts must be outside the repository")
    if args.stage == "prepare":
        if args.checkpoint is None:
            parser.error("prepare requires --checkpoint")
        prepare(repo, output, args.checkpoint.resolve())
    else:
        if args.data_root is None:
            parser.error("run requires --data-root")
        run(repo, output, args.data_root.resolve(), args.deadline_seconds)


if __name__ == "__main__":
    main()
