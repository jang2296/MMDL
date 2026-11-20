"""Offline NF4 cache readiness: real generation/reload, no optimizer or learning."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .checkpoint import digest, file_hash
from .dataset import _bytes, _write
from .local_profile import (DEFAULT_NOTE_CONTRACT, NOTE_CONTRACT_CHOICES, add_repair_cli_arguments, build_protocol,
                            repair_cli_error, repair_cli_keywords)
from .quantized_policy import (assert_quantized_base_unchanged, build_quantized_policy,
    load_quantized_policy, read_quantized_base_metadata, save_quantized_base)


def _sources(*, repair: bool = False) -> dict:
    if type(repair) is not bool:
        raise ValueError("NF4 cache: explicit repair source selection required")
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ("quantized_policy.py", "quantized_cache.py", "policy.py",
        "shared_policy_engine.py", "engines.py", "input_preparation.py", "local_profile.py",
        "checkpoint.py", "trainer.py", "requirements-local-nf4.txt", "resources/baseline.json")]
    paths.append(here.parent / "modules.py")
    if repair:
        paths += [here / name for name in ("compact_note.py", "repair_contract.py", "repair_credit.py", "repair_tokens.py",
            "repair_notebook.py", "repair_training.py", "repair_evidence.py", "repair_replay.py", "reasoning_trace.py", "verifier.py",
            "verifier_service.py", "schema.py", "rollout.py", "prompts.py", "losses.py", "updates.py")]
    return {str(path.relative_to(here.parent)): file_hash(path) for path in paths}


def _requests(output: Path) -> list:
    from PIL import Image, ImageDraw

    from .engines import RolloutRequest

    red, blue = Image.new("RGB", (512, 512), "white"), Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(red).rectangle((128, 128, 384, 384), fill="red")
    ImageDraw.Draw(blue).ellipse((128, 128, 384, 384), fill="blue")
    for name, image in (("red_square", red), ("blue_circle", blue)):
        image.save(output / (name + ".png"))
    requests = []
    for index, images in enumerate(([red], [red, blue])):
        text = ("What color is the large square? Answer with one word." if index == 0 else
                "Describe the colored shape in each image briefly.")
        messages = [{"role": "user", "content": [*({"type": "image"} for _ in images),
                                                       {"type": "text", "text": text}]}]
        requests.append(RolloutRequest(f"nf4-cache-{index}", None, messages, images, 1, 32, [3407 + index]))
    return requests


def _measure(bundle, protocol: dict, output: Path) -> dict:
    from .shared_policy_engine import SharedPolicyRolloutEngine
    from .trainer import SyncProbe, synchronize

    engine = SharedPolicyRolloutEngine(bundle, protocol["image"], protocol["context_limit"],
                                      logprob_chunk=protocol["model"]["logprob_chunk"])
    requests = _requests(output)
    probe_ids = bundle.processor.tokenizer.encode("measurement", add_special_tokens=False)
    sync = synchronize(bundle, engine, engine, [SyncProbe(request, probe_ids) for request in requests],
                       protocol, 0, push=False)
    generated = engine.sample(requests)
    probabilities = {request.request_id: engine.probe_logprobs(request, result.completions[0].token_ids)
                     for request, result in zip(requests, generated, strict=True)}
    assert_quantized_base_unchanged(bundle)
    return {"sync": sync, "generations": [asdict(result) for result in generated],
            "completion_logprobs": probabilities,
            "images": {p.name: file_hash(p) for p in sorted(output.glob("*.png"))}}


def _compare(before: dict, after: dict) -> dict:
    import math

    if (before["images"] != after["images"]
            or len(before["generations"]) != len(after["generations"])):
        raise ValueError("NF4 cache: readiness inputs/coverage changed")
    deltas: list[float] = []
    for old, new in zip(before["generations"], after["generations"], strict=True):
        if (old["request_id"] != new["request_id"] or old["prompt_token_ids"] != new["prompt_token_ids"]
                or len(old["completions"]) != 1 or len(new["completions"]) != 1):
            raise ValueError("NF4 cache: readiness prompt/coverage changed")
        left, right = old["completions"][0], new["completions"][0]
        if {k: v for k, v in left.items() if k != "logprobs"} != {
                k: v for k, v in right.items() if k != "logprobs"}:
            raise ValueError("NF4 cache: fresh-process generation differs")
        for first, second in ((left["logprobs"], right["logprobs"]),
                              (before["completion_logprobs"][old["request_id"]],
                               after["completion_logprobs"][old["request_id"]])):
            if not first or len(first) != len(second):
                raise ValueError("NF4 cache: incomplete readiness probabilities")
            deltas.extend(abs(a - b) for a, b in zip(first, second, strict=True))
    if any(not math.isfinite(d) or d > 1e-6 for d in deltas):
        raise ValueError("NF4 cache: fresh-process probability difference exceeds 1e-6")
    return {"generations": len(after["generations"]), "probabilities": len(deltas),
            "max_abs_logprob_delta": max(deltas), "tolerance": 1e-6, "token_ids_identical": True}


def block_original_access(original_root: Path) -> dict:
    """Process-local proof that a cache reload cannot read its original HF model."""
    original_root = original_root.resolve()
    if not original_root.is_absolute() or len(original_root.parts) < 4:
        raise ValueError("NF4 cache: explicit bounded original model directory required")
    attempts = {"denied_original_access_attempts": 0}

    def audit(event, args):
        if event not in {"open", "os.listdir", "os.scandir"} or not args:
            return
        raw = args[0]
        if isinstance(raw, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(raw)).resolve()
            if path.is_relative_to(original_root):
                attempts["denied_original_access_attempts"] += 1
                raise PermissionError("NF4 cache: original BF16 model access forbidden during reload")

    sys.addaudithook(audit)
    return attempts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("save", "check"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--context-limit", type=int, choices=(16384, 24576), default=16384)
    generation = parser.add_mutually_exclusive_group()
    generation.add_argument("--partial-notebook", action="store_true")
    generation.add_argument("--referenced-notebook", action="store_true")
    generation.add_argument("--paired-notebook", action="store_true")
    generation.add_argument("--pair-action-notebook", action="store_true")
    generation.add_argument("--pair-draft-synthesis", action="store_true")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--whole-reference-notebook", action="store_true")
    reference.add_argument("--whole-reference-pages", action="store_true")
    parser.add_argument("--verified-note-assessment-credit", action="store_true")
    objective = parser.add_mutually_exclusive_group()
    objective.add_argument("--verified-note-atoms", action="store_true")
    objective.add_argument("--verified-note-components", action="store_true")
    objective.add_argument("--verified-note-fact-credit", action="store_true")
    objective.add_argument("--verified-note-synthesis", action="store_true")
    objective.add_argument("--verified-note-synthesis-credit", action="store_true")
    objective.add_argument("--repair-notebook", action="store_true")
    parser.add_argument("--note-contract", choices=NOTE_CONTRACT_CHOICES, default=DEFAULT_NOTE_CONTRACT)
    # Every protocol option a local_run/local_train run can bind, so cache_readiness can match it.
    add_repair_cli_arguments(parser)
    args = parser.parse_args(argv)
    error = repair_cli_error(args, repair_notebook=args.repair_notebook)
    if error is not None:
        parser.error(error)
    if args.policy_precision not in (None, "nf4"):
        parser.error("an NF4 cache never serves a bf16 policy")
    if args.stage == "save" and args.model_path is None:
        parser.error("save requires --model-path")
    if args.stage == "check" and args.parent is None:
        parser.error("check requires --parent (the completed save evidence directory)")
    if args.context_limit == 24576 and not (args.verified_note_fact_credit or args.verified_note_synthesis or args.verified_note_synthesis_credit or args.repair_notebook):
        parser.error("--context-limit 24576 requires --verified-note-fact-credit or --verified-note-synthesis")
    if args.verified_note_synthesis_credit and args.context_limit != 24576:
        parser.error("--verified-note-synthesis-credit requires --context-limit 24576")
    if args.partial_notebook and not args.verified_note_synthesis_credit:
        parser.error("--partial-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.referenced_notebook and not args.verified_note_synthesis_credit:
        parser.error("--referenced-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.paired_notebook and not args.verified_note_synthesis_credit:
        parser.error("--paired-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.pair_action_notebook and not args.verified_note_synthesis_credit:
        parser.error("--pair-action-notebook requires --verified-note-synthesis-credit --context-limit 24576")
    if args.pair_draft_synthesis and not args.verified_note_synthesis_credit:
        parser.error("--pair-draft-synthesis requires --verified-note-synthesis-credit --context-limit 24576")
    if args.whole_reference_notebook and not args.pair_draft_synthesis:
        parser.error("--whole-reference-notebook requires --pair-draft-synthesis")
    if args.whole_reference_pages and not args.pair_draft_synthesis:
        parser.error("--whole-reference-pages requires --pair-draft-synthesis")
    if (args.verified_note_assessment_credit
            and (not args.verified_note_synthesis_credit or not args.pair_draft_synthesis
                 or not args.whole_reference_notebook or args.whole_reference_pages or args.context_limit != 24576)):
        parser.error("--verified-note-assessment-credit requires monolithic --whole-reference-notebook --pair-draft-synthesis --verified-note-synthesis-credit --context-limit 24576 without exploration override")
    if args.repair_notebook and (args.context_limit != 24576 or any((args.partial_notebook,
            args.referenced_notebook, args.paired_notebook, args.pair_action_notebook,
            args.pair_draft_synthesis, args.whole_reference_notebook, args.whole_reference_pages,
            args.verified_note_assessment_credit))):
        parser.error("--repair-notebook requires exclusive --context-limit 24576")
    if args.note_contract != DEFAULT_NOTE_CONTRACT and not args.repair_notebook:
        parser.error("--note-contract other than the default requires --repair-notebook")
    repair_protocol = (build_protocol(repair_notebook=True, context_limit=24576, **repair_cli_keywords(args))
                       if args.repair_notebook else None)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    import torch

    from .policy import enable_deterministic_execution

    began = time.monotonic()
    result = {"stage": args.stage, "pid": os.getpid(), "status": "INCOMPLETE",
              "optimizer_updates": 0, "training_verified": False}
    code = 0
    try:
        deterministic = enable_deterministic_execution()
        torch.cuda.reset_peak_memory_stats()
        protocol = repair_protocol if repair_protocol is not None else build_protocol(verified_note_atoms=args.verified_note_atoms,
            verified_note_components=args.verified_note_components,
            verified_note_fact_credit=args.verified_note_fact_credit,
            verified_note_synthesis=args.verified_note_synthesis,
            verified_note_synthesis_credit=args.verified_note_synthesis_credit,
            verified_note_assessment_credit=args.verified_note_assessment_credit,
            partial_notebook=args.partial_notebook, referenced_notebook=args.referenced_notebook,
            paired_notebook=args.paired_notebook,
            pair_action_notebook=args.pair_action_notebook,
            pair_draft_synthesis=args.pair_draft_synthesis,
            whole_reference_notebook=args.whole_reference_notebook,
            whole_reference_pages=args.whole_reference_pages,
            context_limit=args.context_limit)
        sources = _sources(repair=args.repair_notebook)
        _write(args.output / "sources.json", _bytes(sources) + b"\n")
        _write(args.output / "protocol.json", _bytes(protocol) + b"\n")
        torch.manual_seed(protocol["local_budget"]["seed"])
        if args.stage == "save":
            from .local_calibrate import verify_original

            original = verify_original(args.model_path)
            _write(args.output / "original_files.json", _bytes(original) + b"\n")
            source = args.model_path.resolve()
            result["original_model_root"] = str(source.parent.parent if source.parent.name == "snapshots" else source)
            bundle = build_quantized_policy(args.model_path, "MOD-RV", protocol)
            reused = args.cache.exists()
            metadata = (read_quantized_base_metadata(args.cache) if reused else
                        save_quantized_base(bundle, args.cache))
            if (bundle.base_sha256 != metadata["base_sha256"]
                    or bundle.inventory["quantization"] != metadata["quantization"]):
                raise ValueError("NF4 cache: source model differs from the existing immutable cache")
            result["cache_write"] = "REUSED_VERIFIED_UNCHANGED" if reused else "CREATED"
        else:
            parent = json.loads((args.parent / "result.json").read_text())
            if parent["status"] != "NF4_CACHE_SAVED_GENERATION_VERIFIED" or parent["pid"] == os.getpid():
                raise ValueError("NF4 cache: completed save in a different process required")
            if digest(json.loads((args.parent / "protocol.json").read_text())) != digest(protocol):
                raise ValueError("NF4 cache: saved readiness protocol changed")
            if json.loads((args.parent / "sources.json").read_text()) != sources:
                raise ValueError("NF4 cache: source differs from original readiness process")
            result["original_model_root"] = parent["original_model_root"]
            attempts = block_original_access(Path(parent["original_model_root"]))
            bundle = load_quantized_policy(args.cache, "MOD-RV", protocol)
            metadata = read_quantized_base_metadata(args.cache)
        measured = _measure(bundle, protocol, args.output)
        _write(args.output / "measurement.json", _bytes(measured) + b"\n")
        if args.stage == "check":
            before = json.loads((args.parent / "measurement.json").read_text())
            result["parity"] = _compare(before, measured)
            result.update(attempts)
            if attempts["denied_original_access_attempts"]:
                raise ValueError("NF4 cache: reload attempted original model access")
        if sources != _sources(repair=args.repair_notebook):
            raise ValueError("NF4 cache: source changed during readiness measurement")
        assert_quantized_base_unchanged(bundle)
        statuses = {"save": "NF4_CACHE_SAVED_GENERATION_VERIFIED",
                    "check": "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED"}
        result.update(status=statuses[args.stage],
            cache_metadata_sha256=file_hash(args.cache / "nf4_base.json"),
            quantized_modules=len(metadata["quantization"]["modules"]),
            deterministic=deterministic, generation_verified=True,
            device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
            sources_unchanged=True)
    except Exception as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        code = 1
    finally:
        result.update(seconds=time.monotonic() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        _write(args.output / "result.json", _bytes(result) + b"\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
