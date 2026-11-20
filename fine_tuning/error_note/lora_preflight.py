"""v15 evaluation preflight: before any ledger START, build the real in-process B2 evaluation engine for an unmerged-LoRA
export (proof_eval.registered_variants, kind unmerged-lora) and check it on synthetic inputs only (no MMMU item). Any
failure exits non-zero, so the evaluation never starts; there is no merged fallback.

CLI: VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m fine_tuning.error_note.lora_preflight check --model-ref R
     --checkpoint C --binding B --base S --protocol P --hardware H --out REPORT.json

Checks (the fine_tuning.error_note.llm_lora functions the v15 training sampler runs):
  files  every exported shard tensor outside the four Mergers is bit-identical to the pinned snapshot tensor; the LLM
         adapter file and config equal write_llm_adapter(checkpoint); the vision file equals vision_lora(checkpoint)
  pf1    LoRA wrappers are exactly the language-model modules, no vision weight renamed, two live hooks per block
  pf2    active LoRA slots == bf16(A) / bf16(B) x alpha/r, hook buffers == vision_lora, vLLM Mergers == bf16(masters)
  pf3    no non-Merger vLLM weight changed since engine start; layer-0 / vision-block-0 bases == the HF original
  pf5    teacher-forced log-probs of fixed probes (noise images at the protocol's min_pixels, 1 and 2 images, fixed
         text continuation): vLLM total / LLM / vision adapter effects against HF/PEFT within llm_lora.slope_check,
         slope tested on the adapter with every LoRA B x llm_lora.AMPLIFY (the real adapter's slope is a diagnostic)
  pf2_restored  pf2 again after the amplified probe: the export's adapter and hook buffers are back, bit-exact
  throughput  recorded only, never a pass/fail criterion: greedy 256-token generation speed of `slots` text requests with
         the adapter and with lora_request=None (base / lora ratio; the pod scripts log it for the time estimates)
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import torch

from . import llm_lora

THROUGHPUT_TOKENS = 256
THROUGHPUT_LIMIT = 1.5
CONTINUATION = "The picture is covered with dense random colored noise and shows no recognizable object or text. "


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().flatten().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a.cpu()), _bits(b.cpu()))


def _shards(root: Path) -> dict[str, Path]:
    from safetensors import safe_open

    where = {}
    for path in sorted(Path(root).glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            where.update(dict.fromkeys(handle.keys(), path))
    return where


def check_files(model_path: Path, base: Path, bundle: Any) -> dict:
    """Exported files against the pinned snapshot (bitwise) and the checkpoint's adapters (exact)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    from fine_tuning.notebook_rl.export import LLM_LORA_DIR, read_vision_lora

    exported, original = _shards(model_path), _shards(base)
    unequal, mergers = [], 0
    for name, path in sorted(exported.items()):
        if name.startswith(llm_lora.HF_MERGERS):
            mergers += 1
            continue
        if name not in original:
            unequal.append(name)
            continue
        with safe_open(str(path), framework="pt") as a, safe_open(str(original[name]), framework="pt") as b:
            if not _same(a.get_tensor(name), b.get_tensor(name)):
                unequal.append(name)
    with tempfile.TemporaryDirectory(prefix="lora-preflight-") as tmp:
        expected = Path(tmp) / "adapter"
        llm_lora.write_llm_adapter(bundle, expected)
        want, have = (load_file(str(d / "adapter_model.safetensors")) for d in (expected, Path(model_path) / LLM_LORA_DIR))
        config_equal = ((expected / "adapter_config.json").read_bytes()
                        == (Path(model_path) / LLM_LORA_DIR / "adapter_config.json").read_bytes())
    llm_bad = sorted(set(want) ^ set(have)) + [k for k in sorted(set(want) & set(have)) if not _same(want[k], have[k])]
    vision_want, vision_have = llm_lora.vision_lora(bundle), read_vision_lora(model_path)
    vision_bad = sorted(set(vision_want) ^ set(vision_have)) + [
        n for n in sorted(set(vision_want) & set(vision_have))
        if not all(_same(x, y) for x, y in zip(vision_want[n], vision_have[n]))]
    return {"base_tensors": len(exported) - mergers, "merger_tensors": mergers, "base_mismatch": unequal[:8],
            "llm_adapter_tensors": len(have), "llm_adapter_mismatch": llm_bad[:8], "adapter_config_equal": config_equal,
            "vision_modules": len(vision_have), "vision_mismatch": vision_bad[:8],
            "pass": bool(exported) and mergers > 0 and not unequal and bool(have) and not llm_bad and config_equal
            and bool(vision_have) and not vision_bad}


def probe_preparation(base: Path, model_path: Path, image: dict, context_limit: int) -> Any:
    """single_vllm's input shell (InputPreparation + VLLMRolloutEngine._prompt) over the base processor and the
    protocol's image config: HF model inputs and the vLLM token prompt of one probe come from one preparation."""
    from transformers import AutoConfig, AutoProcessor

    from fine_tuning.notebook_rl.engines import InputPreparation, VLLMRolloutEngine

    processor = AutoProcessor.from_pretrained(str(base), local_files_only=True, min_pixels=image["min_pixels"],
                                              max_pixels=image["max_pixels"])
    processor.image_processor.size = {"shortest_edge": image["min_pixels"], "longest_edge": image["max_pixels"]}
    prep = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
    InputPreparation.__init__(prep, processor, AutoConfig.from_pretrained(str(model_path), local_files_only=True),
                              image, context_limit)
    prep._call = 0
    return prep


def probe_pairs(prep: Any, image: dict) -> list[tuple[Any, list[int]]]:
    from .single_vllm import PROBE_TOKENS, probe_requests

    completion = prep.tokenizer.encode(CONTINUATION * 8, add_special_tokens=False)[:PROBE_TOKENS]
    if len(completion) != PROBE_TOKENS:
        raise ValueError("lora_preflight: probe continuation shorter than PROBE_TOKENS")
    return [(request, list(completion)) for request in probe_requests(image)]


_PROBE_IDS = itertools.count(1)  # amplified adapter ids: proof_eval.LORA_INT_ID + n, never reused in a process


def adapter_swap(engine: Any, bundle: Any, real: Any, root: Path) -> Any:
    """llm_lora.vllm_columns `swap` on the evaluation engine (one adapter slot): factor != 1 loads write_llm_adapter(
    bundle, factor) under a fresh id; factor 1 re-attaches the export's own request. That request keeps LORA_INT_ID
    because attach_lora's add_request binding and backend.json name it; it was removed before the amplified adapter
    was added, so vLLM reads it again from the export files (pf2_restored checks the bits)."""
    from vllm.lora.request import LoRARequest

    def swap(factor: float) -> Any:
        for lora_id in sorted(engine.list_loras()):
            if engine.remove_lora(lora_id) is not True:
                raise RuntimeError("lora_preflight: adapter not removed")
        request = real
        if factor != 1:
            lora_id = real.lora_int_id + next(_PROBE_IDS)
            llm_lora.write_llm_adapter(bundle, root / str(lora_id), factor)
            request = LoRARequest(f"probe-x{factor:g}", lora_id, str(root / str(lora_id)))
        if engine.add_lora(request) is not True or engine.list_loras() != {request.lora_int_id}:
            raise RuntimeError("lora_preflight: probe adapter swap failed")
        return request

    return swap


def engine_checks(backend: Any, bundle: Any, prep: Any, pairs: list, reference: dict[str, list[float]], *,
                  protocol_version: str | None = None, stage: str = "trained",
                  vision_exception: str | None = None, vision_tower: dict | None = None) -> dict:
    """pf1-pf3, pf5 and pf2_restored on a VLLMBackend built with kind unmerged-lora (proof_eval.attach_lora ran in its
    __init__); pf5 runs the training sampler's real + amplified probe (llm_lora.vllm_columns / slope_check)."""
    from fine_tuning.notebook_rl.engines import VLLMRolloutEngine
    from fine_tuning.notebook_rl.proof_eval import LORA_INT_ID

    from .single_vllm import SingleVllmSampler

    llm, config, hooks, lora = backend.llm, backend.model_config, backend.vision_hooks, backend.lora_request
    frozen = llm_lora.base_fingerprint(llm)
    report: dict[str, Any] = {
        "pf1": llm_lora.check_scope(llm, hooks, config.text_config.num_hidden_layers, config.vision_config.depth),
        "pf2": llm_lora.check_identity(llm, bundle, LORA_INT_ID, hooks)}
    prep.llm = llm
    shell: Any = SimpleNamespace(engine=prep)  # SingleVllmSampler._probe reads only self.engine
    probe = functools.partial(SingleVllmSampler._probe, shell)
    with tempfile.TemporaryDirectory(prefix="lora-preflight-amplified-") as tmp:
        columns, restored = llm_lora.vllm_columns(bundle, hooks, probe, pairs, lora,
                                                  adapter_swap(llm.llm_engine, bundle, lora, Path(tmp)))
    report["pf3"] = llm_lora.check_base(llm, bundle, frozen)
    power = llm_lora.pf5_contract_v17(bundle, stage) if protocol_version in ("error-note-v17", "error-note-v18", "error-note-v18b") else None
    if protocol_version in ("error-note-v18", "error-note-v18b") and (vision_exception is not None or vision_tower is None):
        raise ValueError("v18 PF5: direct vision tower evidence required; exception forbidden")
    report["pf5"] = llm_lora.slope_check(**columns, **reference, cap=VLLMRolloutEngine.sync_threshold_cap,
                                       require_power=power["require_power"] if power is not None else True,
                                       vision_exception=vision_exception, vision_tower=vision_tower)
    if power is not None:
        report["pf5"]["stage_contract"] = power
        report["pf5"]["effect_status"] = ("PASS" if report["pf5"]["pass"] and report["pf5"]["power"]
                                         else "UNMEASURED" if power["require_power"] else "NOT_APPLICABLE_F0")
    report["pf2_restored"] = llm_lora.check_identity(llm, bundle, restored.lora_int_id, hooks)
    report["pass"] = all(report[key]["pass"] is True for key in ("pf1", "pf2", "pf3", "pf5", "pf2_restored"))
    return report


def throughput(backend: Any, prep: Any, slots: int) -> dict:
    """Generation speed of `slots` identical text-only requests (greedy, 256 forced tokens) with the LoRA slot and without
    it (lora_request=None). Records numbers; the caller does not gate on `ratio` against `limit`."""
    from vllm import SamplingParams

    messages = [{"role": "user", "content": [{"type": "text", "text": CONTINUATION}]}]
    ids = prep.tokenizer.encode(prep.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                                add_special_tokens=False)
    params = SamplingParams(n=1, temperature=0.0, max_tokens=THROUGHPUT_TOKENS, ignore_eos=True, detokenize=False)
    speeds = []
    for lora in (backend.lora_request, None):
        start = time.perf_counter()
        outputs = backend.llm.generate([{"prompt_token_ids": ids}] * slots, params, use_tqdm=False, lora_request=lora)
        seconds = time.perf_counter() - start
        tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        if tokens != slots * THROUGHPUT_TOKENS:
            raise RuntimeError(f"lora_preflight: throughput generated {tokens} tokens, expected {slots * THROUGHPUT_TOKENS}")
        speeds.append(tokens / seconds)
    return {"slots": slots, "tokens_per_request": THROUGHPUT_TOKENS, "lora_tokens_per_second": speeds[0],
            "base_tokens_per_second": speeds[1], "ratio": speeds[1] / speeds[0], "limit": THROUGHPUT_LIMIT}


def check(model_ref: Path, checkpoint: Path, binding_file: Path, base: Path, protocol: Path, hardware: Path,
          out: Path) -> dict:
    import yaml

    from mmdl.evaluation import engine as eval_engine
    from mmdl.evaluation.backends.vllm_backend import VLLMBackend
    from mmdl.runtime import contracts

    from fine_tuning.notebook_rl import proof_eval as pe
    from fine_tuning.notebook_rl.checkpoint import load_eval

    out, model_ref = Path(out), Path(model_ref)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"lora_preflight: refusing to overwrite {out}")
    if os.environ.get(pe.VLLM_INPROCESS_ENV) != "0":
        raise RuntimeError(f"lora_preflight: the evaluation engine is in-process ({pe.VLLM_INPROCESS_ENV}=0)")
    pe._lora_model_ref(model_ref)  # the v10 contract: v15 export, unmerged-lora kind, adapter section and files
    ref, model_path = json.loads(model_ref.read_text(encoding="utf-8")), model_ref.parent / "model"
    version = ref["export"]["training_protocol"]["version"]
    pid = yaml.safe_load(Path(protocol).read_text(encoding="utf-8"))["protocol_id"]
    slots = 9 if version in ("error-note-v18", "error-note-v18b") else pe.resolve_slots(pid)["slots"]
    if version in ("error-note-v18", "error-note-v18b") and os.environ.get(pe.SLOTS_ENV) not in (None, "9"):
        raise ValueError("v18 preflight: nine slots required")
    with pe.registered_variants(slots=slots):
        eval_engine.verify_model(model_path, ref)  # every model file hash, as the evaluation checks them
        cfg, hw = contracts.load_configs(protocol, hardware)
        bundle = load_eval(Path(base), Path(checkpoint), binding=pe._binding_file(binding_file)[0])
        report: dict[str, Any] = {"files": check_files(model_path, Path(base), bundle)}
        prep = probe_preparation(Path(base), model_path, cfg["image"], cfg["execution"]["max_model_len"])
        pairs = probe_pairs(prep, cfg["image"])
        bundle.model.to("cuda")
        reference = llm_lora.hf_columns(bundle, prep, pairs)  # real and amplified HF/PEFT columns
        vision_reference = llm_lora.hf_vision_columns(bundle, prep, pairs) if version in ("error-note-v18", "error-note-v18b") else None
        bundle.model.to("cpu")  # free the GPU for the evaluation engine (gpu_memory_utilization 0.90)
        torch.cuda.empty_cache()
        backend = VLLMBackend(str(model_path), cfg, hw, pe.LORA_KIND, str(base))
        tower = (llm_lora.vision_tower_check(vision_reference, llm_lora.vllm_vision_columns(backend, bundle, prep, pairs))
                 if vision_reference is not None else None)
        report["engine"] = engine_checks(backend, bundle, prep, pairs, reference,
                                          protocol_version=version, stage="trained", vision_tower=tower)
        report["throughput"] = throughput(backend, prep, slots)  # recorded only; "pass" below ignores it
    report |= {"format": "lora-preflight-v1", "model_ref_sha256": pe.sha256_file(model_ref), "protocol_id": pid,
               "pass": report["files"]["pass"] and report["engine"]["pass"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("check", help="build the v15 evaluation engine and run the preflight; exit 1 on FAIL")
    for flag in ("model-ref", "checkpoint", "binding", "base", "protocol", "hardware", "out"):
        run.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    report = check(args.model_ref, args.checkpoint, args.binding, args.base, args.protocol, args.hardware, args.out)
    print("LORA_PREFLIGHT " + json.dumps({"pass": report["pass"], "out": str(args.out)}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # no interpreter-exit GC of the in-process vLLM engine (segfaults under WSL CuMem)
