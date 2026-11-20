"""Separate-process old/new BF16 engine probes; never reads benchmark questions.

Run ``python -m mmdl.evaluation.backend_parity --help`` for operator commands.
The reference constructor and graph/LoRA wrappers are the pinned 00fffac path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import fcntl
from importlib.metadata import version
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from typing import Any
from unittest.mock import patch

from PIL import Image, ImageDraw
import torch
import yaml

from mmdl.evaluation.backends.input_preparation import tensor_identity
from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import load_configs, sample_seed

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_MANIFEST = ROOT / "configs/eval/v18b_reference_00fffac.json"


def verify_reference_sources(path: Path = REFERENCE_MANIFEST) -> dict:
    value = read_json(path)
    for row in value["files"]:
        actual = sha256_file(ROOT / row["path"])
        if actual != row["sha256"]:
            raise ValueError(f"Pinned reference source changed: {row['path']}: {actual}")
    return value


def runtime_fingerprint() -> dict:
    """Bind each old/new pair to the same physical GPU and numerical runtime."""
    if not torch.cuda.is_available():
        raise ValueError("A real CUDA device is required for an engine comparison")
    device = torch.cuda.get_device_properties(0)
    identity = str(getattr(device, "uuid", "")).strip()
    if not identity or identity.lower() in {"none", "n/a", "[n/a]"}:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        identities = [row.strip() for row in output.stdout.splitlines() if row.strip()]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if len(identities) == 1:
            identity = identities[0]
        elif visible.split(",")[0].isdigit():
            identity = identities[int(visible.split(",")[0])]
        elif visible.startswith("GPU-") and visible.split(",")[0] in identities:
            identity = visible.split(",")[0]
        else:
            raise ValueError("Physical GPU UUID is ambiguous; expose one GPU for the comparison")
    if identity.lower() in {"none", "n/a", "[n/a]"} or not identity:
        raise ValueError("An unambiguous actual GPU UUID is required")
    return {"gpu_uuid": identity, "gpu_name": device.name, "total_memory": device.total_memory,
            "compute_capability": [device.major, device.minor], "python": sys.version.split()[0],
            "cuda_runtime": torch.version.cuda,
            "packages": {name: version(name) for name in (
                "torch", "transformers", "vllm", "peft", "safetensors", "qwen-vl-utils")}}


def candidate_sources() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in (
        "src/mmdl/evaluation/backends/vllm_backend.py",
        "src/mmdl/evaluation/backends/input_preparation.py",
        "src/mmdl/evaluation/backend_parity.py", "src/mmdl/runtime/contracts.py")}


def synthetic_requests(*, seven: bool = False, maximum_pixels: bool = False) -> list[dict]:
    """Sixteen fixed P0 prefixes, unique synthetic pixels, including 1/3/5 images."""
    result = []
    for index in range(1 if seven else 16):
        count = 7 if seven else (1, 3, 5)[index % 3]
        images = []
        for slot in range(count):
            # The last prefix tests the maximum pixel budget without changing that budget.
            size = (2240, 1792) if index == 15 or maximum_pixels else (224 + 28 * slot, 224)
            pattern = index + (100 if seven else 0)
            image = Image.new("RGB", size, ((pattern * 13 + slot * 31) % 256, 80, 160))
            draw = ImageDraw.Draw(image)
            draw.rectangle((10 + slot, 20, 70, 80), fill=(220, 20 + index, 40 + slot))
            draw.text((20, 100), f"Synthetic {pattern} / image {slot + 1}", fill="white")
            images.append(image)
        sid = f"synthetic-{'seven' if seven else 'parity'}-{index:02d}"
        text = (f"Question: Inspect the {count} synthetic diagrams in their numbered order. "
                "Which option names a visible shape?\nOptions:\nA. Rectangle\nB. Sphere\n"
                "Please select the correct answer from the options above.")
        result.append(dict(sample_id=sid, seed=sample_seed(3407, sid), images=images,
                           messages=[{"role": "user", "content":
                                      [{"type": "image"} for _ in images] +
                                      [{"type": "text", "text": text}]}]))
    return result


def _pixel_hashes(images: list) -> list[dict]:
    return [{"size": list(image.size), "mode": image.mode,
             "sha256": hashlib.sha256(image.tobytes()).hexdigest()} for image in images]


def _raw_probabilities(backend: Any, prompt: dict, seed: int, token: int, lora: Any) -> tuple[list[float], list[int]]:
    from fine_tuning.error_note.v17_eval_engine import _raw_decoded
    from vllm import SamplingParams

    params = SamplingParams(temperature=0., max_tokens=32, min_tokens=32, ignore_eos=True,
                            allowed_token_ids=[token], logprobs=0, seed=seed, detokenize=False)
    output = backend.llm.generate([prompt], params, lora_request=lora, use_tqdm=False)[0]
    return _raw_decoded(output, token, 32), list(output.prompt_token_ids)


def capture_vision_inputs(backend: Any) -> tuple[list[dict], list[Any]]:
    """Capture actual tower inputs, after vLLM preprocessing and device/dtype placement."""
    observed: list[dict] = []

    def install(model):
        def capture(module, args, kwargs):
            pixels = args[0] if args else kwargs["pixel_values"]
            grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
            if grid is None:
                raise ValueError("Actual vision tower grid unavailable")
            observed.append({"pixels": tensor_identity(pixels),
                             "grid": torch.as_tensor(grid).cpu().tolist()})
        return model.visual.register_forward_pre_hook(capture, with_kwargs=True)

    return observed, backend.llm.apply_model(install)


def probability_rows(backend: Any) -> list[dict]:
    """Use the pinned raw-logprob extractor; capture actual vision tower inputs too."""
    observed, handles = capture_vision_inputs(backend)
    token = backend.processor.tokenizer.encode(" The", add_special_tokens=False)[0]
    rows = []
    try:
        for request in synthetic_requests():
            item, prompt, _ = backend._prepare_request(request, final_only=False)
            observed.clear()
            logprobs, actual_ids = _raw_probabilities(
                backend, prompt, request["seed"], token, getattr(backend, "lora_request", None))
            if actual_ids != item["prompt_token_ids"] or not observed:
                raise ValueError("Actual engine input IDs or vision tensor capture missing")
            rows.append(dict(sample_id=request["sample_id"], seed=request["seed"],
                             input_sha256=item["input_sha256"], token_ids=[token] * 32,
                             logprobs=logprobs, prompt_token_ids=actual_ids,
                             source_pixels=_pixel_hashes(request["images"]),
                             cpu_processor_tensors=item["input_tensors"],
                             image_grid_thw=item["image_grid_thw"],
                             actual_vision_inputs=copy.deepcopy(observed)))
    finally:
        for handle in handles:
            handle.remove()
    return rows


def live_mergers(backend: Any, model_path: Path) -> dict:
    from fine_tuning.modules import tensor_sha256

    manifest = read_json(model_path / "export_manifest.json")
    names = set(manifest["merger_tensor_names"])
    expected = {row["name"].removeprefix("model."): row["sha256"]
                for row in manifest["tensors"] if row["name"] in names}

    def inspect(model):
        return {name: tensor_sha256(value) for name, value in model.named_parameters()
                if name in expected}

    actual = backend.llm.apply_model(inspect)[0]
    if not expected or actual != expected:
        raise ValueError("Live BF16 Merger tensors differ from the export manifest")
    return {"expected": expected, "actual": actual, "pass": True}


def direct_pf5(backend: Any, model_path: Path) -> dict:
    """Export-only direct PF5: real pixels, amplification, actual hook removal, restoration."""
    from fine_tuning.error_note import llm_lora

    hooks = backend.vision_hooks
    request = synthetic_requests()[0]
    item, _, _ = backend._prepare_request(request, final_only=False)
    pixels = item["inputs"]["pixel_values"]
    grid = item["inputs"]["image_grid_thw"]
    original = {name: (a.clone(), b.clone()) for name, (a, b) in hooks.buffers.items()}
    state = hooks.on
    columns = {}
    try:
        for key, factor in (("off", 0), ("off_repeat", 0), ("on", 1),
                            ("amplified", llm_lora.AMPLIFY), ("hook_removed", llm_lora.AMPLIFY)):
            hooks.load({name: (a, b * max(1, factor)) for name, (a, b) in original.items()})
            hooks.on = bool(factor)

            def probe(model):
                removed = []
                try:
                    if key == "hook_removed":
                        for name, handle in hooks.handles.items():
                            module = model.get_submodule(name)
                            removed.append((module, handle.id, module._forward_hooks.pop(handle.id)))
                    device = next(model.visual.parameters()).device
                    return llm_lora._vision_features(model.visual, pixels.to(device), grid.to(device), vllm=True)
                finally:
                    for module, identity, callback in removed:
                        module._forward_hooks[identity] = callback

            columns[key] = backend.llm.apply_model(probe)[0]
    finally:
        hooks.load(original)
        hooks.on = state
    restored = (hooks.on == state and all(torch.equal(a, hooks.buffers[name][0]) and
                torch.equal(b, hooks.buffers[name][1]) for name, (a, b) in original.items()))
    if not restored or columns["hook_removed"] != columns["off"]:
        raise ValueError("PF5 hook removal or normal-strength restoration failed")
    return {"inputs": [{"pixels_sha256": tensor_identity(pixels)["sha256"],
                        "grid_sha256": tensor_identity(grid)["sha256"]}],
            "columns": columns, "restored": restored, "mergers": live_mergers(backend, model_path)}


def language_control(backend: Any, model_path: Path) -> dict:
    """Remove only request LoRA; use an amplified export copy when the real effect is tiny."""
    from safetensors.torch import load_file, save_file
    from vllm.lora.request import LoRARequest
    from fine_tuning.error_note.llm_lora import AMPLIFY

    real = backend.lora_request
    engine = backend.llm.llm_engine
    request = synthetic_requests()[0]
    _, prompt, _ = backend._prepare_request(request, final_only=False)
    token = backend.processor.tokenizer.encode(" The", add_special_tokens=False)[0]
    on, _ = _raw_probabilities(backend, prompt, request["seed"], token, real)
    off, _ = _raw_probabilities(backend, prompt, request["seed"], token, None)
    adapter = model_path / "llm_lora"
    with tempfile.TemporaryDirectory(prefix="mmdl-parity-lora-") as temporary:
        amplified_path = Path(temporary)
        (amplified_path / "adapter_config.json").write_bytes((adapter / "adapter_config.json").read_bytes())
        weights = load_file(str(adapter / "adapter_model.safetensors"))
        save_file({name: value * AMPLIFY if "lora_B" in name else value
                   for name, value in weights.items()}, str(amplified_path / "adapter_model.safetensors"))
        amplified = LoRARequest("parity-amplified", real.lora_int_id + 100000, str(amplified_path))
        try:
            if engine.remove_lora(real.lora_int_id) is not True or engine.add_lora(amplified) is not True:
                raise ValueError("Language PF5 adapter swap failed")
            values, _ = _raw_probabilities(backend, prompt, request["seed"], token, amplified)
        finally:
            for identity in list(engine.list_loras()):
                engine.remove_lora(identity)
            if engine.add_lora(real) is not True or engine.list_loras() != {real.lora_int_id}:
                raise ValueError("Language PF5 normal adapter restoration failed")
    difference = max(abs(x - y) for x, y in zip(values, off, strict=True))
    if difference <= 1e-8:
        raise ValueError("Removing language LoRA did not expose the amplified positive control")
    return {"lora_request_id": real.lora_int_id, "real_logprobs": on, "off_logprobs": off,
            "amplified_logprobs": values, "amplified_max_difference": difference,
            "restored_adapter_ids": sorted(engine.list_loras()), "pass": True}


def compare(reference: dict, candidate: dict) -> dict:
    from fine_tuning.error_note.v17_eval_engine import compare_probabilities
    from fine_tuning.error_note.llm_lora import vision_tower_check

    if reference["engine"] != "reference" or candidate["engine"] != "submission":
        raise ValueError("Expected pinned-reference then submission receipts")
    for key in ("model_files_sha256", "base_config_sha256", "config", "hardware", "rehearsal", "model_kind", "reference_source", "runtime"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Parity bindings differ: {key}")
    for left, right in zip(reference["probabilities"], candidate["probabilities"], strict=True):
        for key in ("sample_id", "prompt_token_ids", "source_pixels", "cpu_processor_tensors",
                    "image_grid_thw", "actual_vision_inputs"):
            if left[key] != right[key]:
                raise ValueError(f"Parity actual input mismatch: {left['sample_id']}: {key}")
    result = {"version": "submission-engine-parity-v1", "reportable": False,
              "config": candidate["config"], "hardware": candidate["hardware"],
              "runtime": candidate["runtime"],
              "rehearsal": candidate["rehearsal"], "model_kind": candidate["model_kind"],
              "model_files_sha256": candidate["model_files_sha256"],
              "seven_image_acceptance": candidate.get("seven_image_acceptance"),
              "reference_source": candidate["reference_source"],
              "candidate_sources": candidate["candidate_sources"],
              "probabilities": compare_probabilities(candidate["probabilities"], reference["probabilities"])}
    if candidate.get("vision_pf5"):
        result["vision_pf5"] = vision_tower_check(reference["vision_pf5"], candidate["vision_pf5"])
        # An unpowered probe is recorded but cannot certify application for an exported FT model.
        result["pass"] = (result["probabilities"]["pass"] and result["vision_pf5"]["pass"]
                          and result["vision_pf5"]["amplified"]["pass"] is True
                          and candidate["language_control"]["pass"])
    else:
        result["pass"] = result["probabilities"]["pass"]
    return result


def validate_admission(path: Path, cfg: dict, hw: dict, model_path: Path, kind: str) -> dict:
    """Before GPU startup, bind an actual successful comparison to this model/profile."""
    value = read_json(path)
    content = dict(value)
    claimed = content.pop("content_sha256", None)
    if claimed != digest(content) or value.get("version") != "submission-engine-parity-v1":
        raise ValueError("Engine comparison receipt is unsealed or has the wrong version")
    if value.get("pass") is not True or value.get("rehearsal") is not False:
        raise ValueError("Production admission requires successful non-rehearsal engine comparison")
    if any(value.get(key) != expected for key, expected in
           (("config", cfg), ("hardware", hw), ("model_kind", kind))):
        raise ValueError("Engine comparison belongs to a different suite, profile or model kind")
    actual = {p.relative_to(model_path).as_posix(): sha256_file(p)
              for p in sorted(model_path.rglob("*")) if p.is_file()}
    if actual != value["model_files_sha256"]:
        raise ValueError("Engine comparison model file hashes differ")
    if value.get("runtime") != runtime_fingerprint():
        raise ValueError("Engine comparison GPU UUID or numerical runtime differs")
    if value.get("candidate_sources") != candidate_sources():
        raise ValueError("Engine comparison submission source hashes differ")
    if value["reference_source"] != verify_reference_sources():
        raise ValueError("Engine comparison reference sources differ")
    if cfg["execution"].get("max_images_per_prompt") == 7:
        seven = value.get("seven_image_acceptance") or {}
        rows = seven.get("rows") or []
        if (seven.get("eight_rejected") is not True or len(rows) != 1
                or rows[0].get("passed_image_count") != 7 or rows[0].get("source_image_count") != 7
                or not seven.get("kv", {}).get("pass")
                or sum(len(row["grid"]) for row in seven.get("actual_vision_inputs", [])) != 7
                or seven.get("production_reservation_seven_maximum", {}).get("status") != "CONTEXT_REJECTED_WITHOUT_TRUNCATION"
                or seven.get("production_reservation_seven_maximum", {}).get("reserved_output_tokens") != 32768):
            raise ValueError("Seven-image, eight-rejection and actual KV evidence required")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "content_sha256": claimed, "pass": True}


def ensure_admission(args: Any, cfg: dict, hw: dict, model_ref: dict) -> dict | None:
    """Run both isolated probes before model allocation, or resume completed probe stages."""
    from mmdl.runtime.contracts import SUBMISSION_PROFILES

    if (cfg.get("protocol_id") not in SUBMISSION_PROFILES or getattr(args, "rehearsal", False)
            or getattr(args, "mode", "full") != "full"):
        return None
    kind = model_ref["kind"]
    supplied = getattr(args, "engine_validation", None)
    if supplied is not None:
        return validate_admission(Path(supplied), cfg, hw, Path(args.model_path), kind)
    directory = Path(args.artifact_root) / "engine-validation" / args.run_id
    existed = directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    if existed and not getattr(args, "resume", False):
        raise FileExistsError(f"Engine admission exists; use --resume or a new run ID: {directory}")
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another engine admission process owns this run") from exc
        comparison = directory / "comparison.json"
        if comparison.exists():
            return validate_admission(comparison, cfg, hw, Path(args.model_path), kind)
        registration = {"config": cfg, "hardware": hw, "kind": kind,
                        "model_path": str(Path(args.model_path).resolve()),
                        "model_ref_sha256": sha256_file(args.model_ref) if kind == "unmerged-lora" else None,
                        "candidate_sources": candidate_sources(), "runtime": runtime_fingerprint()}
        registration_path = directory / "registration.json"
        if registration_path.exists():
            if read_json(registration_path) != registration:
                raise ValueError("Engine admission resume registration differs")
        else:
            if existed and any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError("Unregistered partial engine admission is preserved; use a new run ID")
            write_json(registration_path, registration)
        for engine in ("reference", "submission"):
            output = directory / f"{engine}.json"
            if output.exists():
                saved = read_json(output)
                if saved.get("status") != "MEASURED" or saved.get("config") != cfg:
                    raise ValueError(f"Failed or mismatched admission is preserved: {output}")
                continue
            argv = [sys.executable, "-m", "mmdl.evaluation.backend_parity", "run", "--engine", engine,
                    "--protocol", str(Path(args.protocol).resolve()), "--hardware", str(Path(args.hardware).resolve()),
                    "--model-path", str(Path(args.model_path).resolve()), "--output", str(output)]
            if kind == "unmerged-lora":
                argv += ["--base-path", str(Path(args.base_path).resolve()),
                         "--model-ref", str(Path(args.model_ref).resolve())]
            with (directory / f"{engine}.log").open("x", encoding="utf-8") as log:
                process = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode != 0 or not output.is_file():
                raise RuntimeError(f"Engine admission {engine} failed; preserved output/log: {directory}")
        value = compare(read_json(directory / "reference.json"), read_json(directory / "submission.json"))
        value["receipts"] = {name: {"path": str((directory / f"{name}.json").resolve()),
                                    "sha256": sha256_file(directory / f"{name}.json")}
                             for name in ("reference", "submission")}
        value["content_sha256"] = digest(value)
        write_json(comparison, value)
        return validate_admission(comparison, cfg, hw, Path(args.model_path), kind)


def run_probe(args: argparse.Namespace) -> dict:
    from mmdl.evaluation.backends.vllm_backend import VLLMBackend
    from fine_tuning.notebook_rl import proof_eval as pe
    from fine_tuning.error_note import v17_eval_engine as old

    source = verify_reference_sources()
    if args.rehearsal:
        cfg = yaml.safe_load(args.protocol.read_text())
        hw = yaml.safe_load(args.hardware.read_text())
    else:
        cfg, hw = load_configs(args.protocol, args.hardware)
    os.environ.setdefault(pe.VLLM_INPROCESS_ENV, "0")
    kind = "unmerged-lora" if args.model_ref else "base"
    if args.model_ref:
        from fine_tuning.notebook_rl.export import validate_export_reference
        validate_export_reference(args.model_path, read_json(args.model_ref))
    backend = None
    result = dict(version="submission-engine-probe-v1", engine=args.engine, reportable=False,
                  rehearsal=args.rehearsal, reference_source=source, candidate_sources=candidate_sources(),
                  config=cfg, hardware=hw, model_kind=kind, runtime=runtime_fingerprint(),
                  base_config_sha256=sha256_file((args.base_path or args.model_path) / "config.json"),
                  model_files_sha256={p.relative_to(args.model_path).as_posix(): sha256_file(p)
                                      for p in sorted(args.model_path.rglob("*")) if p.is_file()})
    try:
        if args.engine == "submission":
            backend = VLLMBackend(str(args.model_path), cfg, hw, kind,
                                  str(args.base_path) if args.base_path else None, rehearsal=args.rehearsal)
        else:
            from mmdl.evaluation import backends
            from mmdl.evaluation.backends import vllm_reference_00fffac as legacy
            legacy_cfg = copy.deepcopy(cfg)
            legacy_cfg["protocol_id"] = pe.FINAL_VARIANT_V8
            legacy_cfg["execution"] = pe._fast_execution(__import__("mmdl.runtime.contracts", fromlist=["*"]),
                                                       pid=pe.FINAL_VARIANT_V8)
            legacy_cfg["image"] = copy.deepcopy(cfg["image"])
            legacy_cfg["dataset"] = yaml.safe_load((ROOT / "configs/eval/mmmu_val_v8.yaml").read_text())["dataset"]
            # Scoped constants let the same immutable constructor run tiny explicit rehearsals.
            limits = pe.VARIANT_LIMITS[pe.FINAL_VARIANT_V8] | {
                "max_new_tokens": cfg["generation"]["max_new_tokens"],
                "max_model_len": cfg["execution"]["max_model_len"],
                "pixels": {key: cfg["image"][key] for key in ("min_pixels", "max_pixels")}}
            legacy_cfg["execution"]["max_model_len"] = cfg["execution"]["max_model_len"]
            with patch.object(backends, "vllm_backend", legacy), \
                    patch.dict(pe.VARIANT_LIMITS, {pe.FINAL_VARIANT_V8: limits}), \
                    old.engine_execution("eager" if cfg["execution"].get("enforce_eager", True) else "cuda_graph",
                                         cfg["execution"]["batch_size"],
                                         gpu_memory_utilization=float(cfg["execution"].get("gpu_memory_utilization", .90))), \
                    pe.registered_variants(slots=cfg["execution"]["batch_size"], log_stats=True):
                backend = legacy.VLLMBackend(str(args.model_path), legacy_cfg, hw, kind,
                                             str(args.base_path) if args.base_path else None)
        result["backend"] = backend.describe()
        if not args.rehearsal:
            result["observed_engine"] = old.observe_engine(
                backend, "eager" if cfg["execution"].get("enforce_eager", True) else "cuda_graph",
                cfg["execution"]["batch_size"])
        result["probabilities"] = probability_rows(backend)
        if args.model_ref:
            result["vision_pf5"] = direct_pf5(backend, args.model_path)
            result["language_control"] = language_control(backend, args.model_path)
        if args.engine == "submission":
            seven = synthetic_requests(seven=True)[0]
            observed, handles = capture_vision_inputs(backend)
            try:
                rows = list(backend.generate_stream([seven])) if backend.max_images_per_prompt == 7 else []
            finally:
                for handle in handles:
                    handle.remove()
            actual_seven = copy.deepcopy(observed)
            eight = seven | {"images": seven["images"] + seven["images"][:1]}
            try:
                backend._prepare_request(eight, final_only=False)
            except ValueError:
                rejected = True
            else:
                rejected = False
            maximum = {"status": "NOT_REQUESTED"}
            if backend.max_images_per_prompt == 7:
                try:
                    item, _, _ = backend._prepare_request(
                        synthetic_requests(seven=True, maximum_pixels=True)[0], final_only=False)
                except ValueError as exc:
                    if "refusing truncation" not in str(exc):
                        raise
                    maximum = {"status": "CONTEXT_REJECTED_WITHOUT_TRUNCATION", "reason": str(exc)}
                else:
                    maximum = {"status": "CONTEXT_ACCEPTED", "input_tokens": item["input_tokens"],
                               "image_grid_thw": item["image_grid_thw"]}
            production_maximum: dict[str, Any] = {"status": "NOT_REQUESTED"}
            if backend.max_images_per_prompt == 7:
                # Exercise the real refusal with the fixed production reservation; do not generate or resize down.
                original_cfg = backend.cfg
                backend.cfg = copy.deepcopy(original_cfg)
                backend.cfg["generation"]["max_new_tokens"] = 32768
                try:
                    item, _, _ = backend._prepare_request(
                        synthetic_requests(seven=True, maximum_pixels=True)[0], final_only=False)
                except ValueError as exc:
                    if "refusing truncation" not in str(exc):
                        raise
                    production_maximum = {"status": "CONTEXT_REJECTED_WITHOUT_TRUNCATION",
                                          "reserved_output_tokens": 32768, "reason": str(exc)}
                else:
                    production_maximum = {"status": "CONTEXT_ACCEPTED", "reserved_output_tokens": 32768,
                                          "input_tokens": item["input_tokens"]}
                finally:
                    backend.cfg = original_cfg
            result["seven_image_acceptance"] = {"requested": backend.max_images_per_prompt == 7,
                "rows": rows, "eight_rejected": rejected, "kv": backend.capacity_receipt,
                "seven_maximum_pixel_context": maximum,
                "production_reservation_seven_maximum": production_maximum,
                "actual_vision_inputs": actual_seven,
                "source_pixels": _pixel_hashes(seven["images"])}
            if backend.max_images_per_prompt == 7 and (len(rows) != 1 or not rejected
                    or sum(len(value["grid"]) for value in actual_seven) != 7):
                raise ValueError("Seven-image acceptance or eight-image rejection failed")
        result["status"] = "MEASURED"
        return result
    finally:
        if backend is not None:
            backend.llm.llm_engine.engine_core.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one engine in a fresh process; invoke twice then compare")
    run.add_argument("--engine", choices=("reference", "submission"), required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--hardware", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--base-path", type=Path)
    run.add_argument("--model-ref", type=Path)
    run.add_argument("--rehearsal", action="store_true", help="Explicit tiny synthetic run; never official evidence")
    run.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a fresh receipt path; previous evidence is immutable")
    try:
        result = (run_probe(args) if args.command == "run" else
                  compare(read_json(args.reference), read_json(args.candidate)))
    except Exception as exc:
        result = {"version": "submission-engine-probe-failure-v1", "reportable": False,
                  "status": "FAIL", "pass": False, "error_type": type(exc).__name__,
                  "error": str(exc)}
    if args.command == "compare":
        result["receipts"] = {"reference": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference)},
                              "candidate": {"path": str(args.candidate.resolve()), "sha256": sha256_file(args.candidate)}}
    result["content_sha256"] = digest(result)
    write_json(args.output, result)
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
