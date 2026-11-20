"""Out-of-place FP32 fusion → sharded BF16 HF export, with immutable sources."""
from __future__ import annotations

import json
import re
import shutil
import stat
from pathlib import Path, PureWindowsPath
from typing import Any

import torch
from safetensors.torch import save_file

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from fine_tuning.notebook_rl.checkpoint import PolicyFactory, digest, file_hash, load_eval, read_metadata
from fine_tuning.notebook_rl.policy import assert_base_unchanged, merged_named_tensors
from fine_tuning.notebook_rl.resources import resource_path
from fine_tuning.notebook_rl.vault import _read, _safe_root, _unique

BASE_MANIFEST = resource_path("manifests/models/baseline.json")
_AUXILIARY = {"config.json", "generation_config.json", "chat_template.json", "merges.txt", "vocab.json",
              "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"}
FORMAT_V1 = "derived-hf-bf16-v1"
PRECISION_V1 = {"fusion": "FP32", "export": "BF16", "merger_masters": "FP32 checkpoint preserved; export rounds to BF16",
                "unmerged_inference_bit_equivalent": False}
# v15 (rollout_adapter vllm-lora-unmerged+vision-hooks): nothing is merged. The shards hold the original pinned BF16
# tensors of the language model and vision tower (bit-identical) plus the BF16-rounded Merger masters; the LLM adapter is
# a PEFT file for vLLM's LoRA path and the vision adapter a BF16 file for exact forward hooks (fine_tuning.error_note.llm_lora).
FORMAT_LORA = "derived-hf-bf16-unmerged-lora-v1"
PRECISION_LORA = {"fusion": "none: no adapter is merged into any weight", "export": "BF16",
                  "merger_masters": "FP32 checkpoint preserved; export rounds to BF16",
                  "base": "original pinned BF16 tensors of the language model and vision tower, bit-identical",
                  "llm_adapter": "unmerged PEFT file (FP32 A/B, r 16, alpha 32) applied by the vLLM 0.11.0 LoRA path "
                                 "with BF16 slots (B x alpha/r)",
                  "vision_adapter": "unmerged BF16 A and B x alpha/r applied by exact forward hooks on "
                                    "visual.blocks.N.attn.{qkv,proj}",
                  "unmerged_inference_bit_equivalent": False}
FORMATS = {FORMAT_V1: PRECISION_V1, FORMAT_LORA: PRECISION_LORA}
LLM_LORA_DIR, VISION_LORA_DIR = "llm_lora", "vision_lora"
VISION_LORA_FILE = f"{VISION_LORA_DIR}/vision_lora.safetensors"
LORA_FILES = (f"{LLM_LORA_DIR}/adapter_config.json", f"{LLM_LORA_DIR}/adapter_model.safetensors", VISION_LORA_FILE)


def model_files(root: Path) -> list[str]:
    """Sorted relative POSIX paths of every file under root (for a flat directory: the sorted file names)."""
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if not p.is_dir())


def read_vision_lora(model_path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """vLLM module name -> (BF16 A, BF16 B x alpha/r), as llm_lora.vision_lora returns it."""
    from safetensors.torch import load_file

    tensors = load_file(str(Path(model_path) / VISION_LORA_FILE))
    names = {key.rpartition(".")[0] for key in tensors}
    if set(tensors) != {f"{n}.{part}" for n in names for part in ("lora_A", "lora_B")}:
        raise ValueError("export: vision LoRA file must hold exactly lora_A/lora_B pairs")
    return {name: (tensors[f"{name}.lora_A"], tensors[f"{name}.lora_B"]) for name in sorted(names)}


def validate_export_identity(model_path: Path, arm: str) -> dict:
    """Validate a completed derived export without the historical evaluator package."""
    root = _safe_root(Path(model_path))
    manifest = json.loads(_read(root / "export_manifest.json"), object_pairs_hook=_unique)
    fmt = manifest.get("format") if isinstance(manifest, dict) else None
    if (not isinstance(arm, str) or not arm or not isinstance(manifest, dict) or manifest.get("status") != "EXPORTED"
            or fmt not in FORMATS or manifest.get("arm") != arm
            or manifest.get("base_id") != BASE_ID or manifest.get("base_revision") != BASE_REVISION
            or manifest.get("precision_policy") != FORMATS[fmt]
            or manifest["precision_policy"]["unmerged_inference_bit_equivalent"] is not False):
        raise ValueError("export: complete pinned BF16 derived identity required")
    files = manifest.get("files")
    lora = fmt == FORMAT_LORA
    if not isinstance(files, dict) or not files or (lora and not set(LORA_FILES) <= set(files)):
        raise ValueError("export: nonempty file/hash inventory required")
    for name, expected in files.items():
        if (not isinstance(name, str) or name in ("", ".", "..")
                or (len(Path(name).parts) != 1 and not (lora and name in LORA_FILES))
                or PureWindowsPath(name).is_absolute()
                or (len(PureWindowsPath(name).parts) != 1 and not (lora and name in LORA_FILES))
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("export: unsafe model file/hash inventory")
        path = root / name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not path.resolve(strict=True).is_relative_to(root.resolve()) or file_hash(path) != expected):
            raise ValueError("export: model file/hash mismatch")
    return {"arm": arm, "manifest_sha256": digest(manifest), "files": files,
            "derived_precision_policy": manifest["precision_policy"]}


def _validate_base(model_path: Path, manifest_path: Path) -> tuple[dict, dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not model_path.is_dir() or model_path.name != BASE_REVISION
            or manifest["repo_id"] != BASE_ID or manifest["revision"] != BASE_REVISION):
        raise ValueError("export: validated local pinned base required; no downloads")
    files: dict[str, str] = {}
    for row in manifest["files"]:
        name = row["path"]
        if (Path(name).is_absolute() or len(Path(name).parts) != 1 or name in files
                or (model_path / name).stat().st_size != row["bytes"]
                or file_hash(model_path / name) != row["sha256"]):
            raise ValueError("export: pinned base file mismatch")
        files[name] = row["sha256"]
    if (not _AUXILIARY <= files.keys()
            or {p.name for p in model_path.glob("*.safetensors")} != {n for n in files if n.endswith(".safetensors")}
            or json.loads((model_path / "config.json").read_text())["model_type"] != "qwen3_vl"):
        raise ValueError("export: incomplete processor/tokenizer/config or weight inventory")
    return manifest, files


def validate_export_reference(model_path: Path, reference: dict) -> dict:
    """Bind the evaluated files to their declared checkpoint, not just a file inventory."""
    root, export = Path(model_path), reference["export"]
    identity = validate_export_identity(root, export["arm"])
    manifest_path = root / "export_manifest.json"
    manifest = json.loads(_read(manifest_path), object_pairs_hook=_unique)
    artifact_sha = file_hash(manifest_path)
    files = manifest["files"] | {"export_manifest.json": artifact_sha}
    inventory = [{"path": name, "bytes": (root / name).stat().st_size, "sha256": sha}
                 for name, sha in sorted(files.items())]
    dirs = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}
    allowed = {LLM_LORA_DIR, VISION_LORA_DIR} if manifest["format"] == FORMAT_LORA else set()
    if (set(model_files(root)) != set(files) or dirs != allowed or reference.get("files") != inventory
            or reference.get("artifact_revision") != artifact_sha
            or export.get("export_manifest_sha256") != identity["manifest_sha256"]
            or reference.get("train_config_sha256") != manifest["resolved_protocol_sha256"]
            or export.get("checkpoint_state_sha256") != digest(manifest["checkpoint_state_sha256"])
            or any(export.get(key) != manifest[key] for key in
                   ("checkpoint_files", "binding", "base_manifest_sha256", "mergers", "precision_policy"))):
        raise ValueError("export: evaluated files and declared checkpoint reference differ")
    return manifest


def export_merged(ckpt: Path, model_path: Path, out: Path, arm: str, *, binding: dict,
                  max_shard_size: int = 2_000_000_000, llm_lora: bool = False,
                  _test_policy_factory: PolicyFactory | None = None,
                  _test_base_manifest: Path | None = None) -> dict:
    """Streaming shards contain canonical unique parameters, so tied weights are not duplicated.

    HF reconstructs tied lm_head/embed weights from the copied pinned config. A single
    parameter larger than max_shard_size occupies its own shard (the standard HF rule).
    Test-only injection is explicit; production always uses the pinned 4B policy builder.
    ``llm_lora`` (v15): every non-Merger tensor is the pinned snapshot tensor itself (checked equal to the policy's frozen
    base), the Mergers are their BF16-rounded masters, and the LLM / vision adapters are written unmerged (FORMAT_LORA).
    """
    ckpt, model_path, out = Path(ckpt), Path(model_path), Path(out)
    protected = Path(__file__).resolve().parents[2] / "MMDL"
    if (out.resolve().is_relative_to(protected) or out.resolve().is_relative_to(model_path.resolve())
            or out.resolve().is_relative_to(ckpt.resolve())):
        raise ValueError("export: output must be outside sources/protected submission")
    if out.exists() or out.is_symlink():
        raise FileExistsError("export: destination must be new")
    if type(max_shard_size) is not int or max_shard_size <= 0:
        raise ValueError("export: positive shard byte limit required")
    if (_test_policy_factory is None) != (_test_base_manifest is None):
        raise ValueError("export: test factory and test base manifest must be paired explicitly")
    manifest_path = _test_base_manifest or BASE_MANIFEST
    manifest_sha = file_hash(manifest_path)
    _, base_files = _validate_base(model_path, manifest_path)
    metadata = read_metadata(ckpt, binding=binding)
    if arm != metadata["arm"]:
        raise ValueError("export: checkpoint arm differs")
    source_checkpoint = {p.name: file_hash(p) for p in ckpt.iterdir()}
    bundle = load_eval(model_path, ckpt, binding=binding, _test_policy_factory=_test_policy_factory)
    before = {name: tensor_sha256(p) for name, p in bundle.model.named_parameters()}
    out.mkdir(parents=True, exist_ok=False)
    # Canonical names are deduplicated by named_parameters, matching HF safe serialization
    # for tied weights. No in-place PEFT merge and no mutation of config/source tensors.
    tensor_rows: list[dict[str, Any]] = []
    shards: list[tuple[str, list[str]]] = []
    weight_map: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f".part-{len(shards) + 1:05d}.safetensors"
        save_file(pending, str(out / filename), metadata={"format": "pt"})
        shards.append((filename, list(pending)))
        pending, pending_bytes = {}, 0

    originals = _original_tensors(model_path, base_files, bundle) if llm_lora else None
    mergers = tuple(root + "." for root in bundle.inventory["mergers"])
    for name, value in merged_named_tensors(bundle):
        if originals is not None and not name.startswith(mergers):
            value = originals(name)
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise ValueError("export: finite BF16 weights required")
        value = value.detach().cpu().contiguous()
        size = value.numel() * value.element_size()
        if pending and pending_bytes + size > max_shard_size:
            flush()
        pending[name] = value
        pending_bytes += size
        tensor_rows.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": tensor_sha256(value)})
    flush()
    for index, (filename, names) in enumerate(shards, 1):
        target = ("model.safetensors" if len(shards) == 1
                  else f"model-{index:05d}-of-{len(shards):05d}.safetensors")
        (out / filename).rename(out / target)
        weight_map.update({name: target for name in names})
    if len(shards) > 1:
        index_manifest = {"metadata": {"total_size": sum(
            2 * _numel(row["shape"]) for row in tensor_rows)}, "weight_map": weight_map}
        (out / "model.safetensors.index.json").write_text(json.dumps(index_manifest, indent=2) + "\n")
    for name in base_files:
        if not name.endswith(".safetensors") and name != "model.safetensors.index.json":
            shutil.copyfile(model_path / name, out / name)
    if llm_lora:
        from fine_tuning.error_note import llm_lora as lora

        lora.write_llm_adapter(bundle, out / LLM_LORA_DIR)
        vision = lora.vision_lora(bundle)
        (out / VISION_LORA_DIR).mkdir()
        save_file({f"{n}.{part}": t.contiguous() for n, (a, b) in vision.items() for part, t in (("lora_A", a), ("lora_B", b))},
                  str(out / VISION_LORA_FILE))
        lora_section = {"scaling": lora.adapter_scaling(bundle), "llm_targets": len(bundle.inventory["targets"]["L"]),
                        "vision_targets": len(vision), "files": list(LORA_FILES)}
    assert_base_unchanged(bundle)
    if before != {name: tensor_sha256(p) for name, p in bundle.model.named_parameters()}:
        raise ValueError("export: policy mutated during export")
    if (file_hash(manifest_path) != manifest_sha or _validate_base(model_path, manifest_path)[1] != base_files
            or source_checkpoint != {p.name: file_hash(p) for p in ckpt.iterdir()}
            or read_metadata(ckpt, binding=binding) != metadata):
        raise ValueError("export: source changed during export")
    manifest = {"status": "EXPORTED", "format": FORMAT_LORA if llm_lora else FORMAT_V1, "arm": arm,
        "base_id": BASE_ID, "base_revision": BASE_REVISION, "base_manifest_sha256": manifest_sha,
        "base_files": base_files, "checkpoint_files": source_checkpoint,
        "checkpoint_state_sha256": metadata["state_sha256"], "binding": binding,
        "resolved_protocol_sha256": metadata["resolved_protocol_sha256"],
        "mergers": bundle.inventory["mergers"],
        "merger_tensor_names": sorted(r["name"] for r in tensor_rows if any(
            r["name"].startswith(root + ".") for root in bundle.inventory["mergers"])),
        "operation_order": ["verify_pinned_base_files", "fresh_BF16_F0", "restore_all_adapters_and_FP32_masters",
                            "out_of_place_FP32_fusion", "cast_BF16", "save_HF_shards"],
        "precision_policy": dict(PRECISION_V1),
        "tensors": tensor_rows, "files": ({name: file_hash(out / name) for name in model_files(out)} if llm_lora
                                          else {p.name: file_hash(p) for p in sorted(out.iterdir())})}
    if llm_lora:
        manifest["operation_order"] = ["verify_pinned_base_files", "fresh_BF16_F0", "restore_all_adapters_and_FP32_masters",
                                       "copy_original_snapshot_tensors", "cast_merger_masters_BF16", "save_HF_shards",
                                       "save_llm_lora_peft", "save_vision_lora_bf16"]
        manifest["precision_policy"] = dict(PRECISION_LORA)
        manifest["lora"] = lora_section
    # The final manifest is the completion marker; a failed directory is never accepted.
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _original_tensors(model_path: Path, base_files: dict[str, str], bundle: Any):
    """name -> the pinned snapshot's own tensor, after checking it equals the policy's frozen base parameter."""
    from safetensors import safe_open

    from fine_tuning.notebook_rl.policy import _canonical_frozen

    frozen = dict(_canonical_frozen(bundle))
    where = {}
    for file in sorted(n for n in base_files if n.endswith(".safetensors")):
        with safe_open(str(model_path / file), framework="pt") as handle:
            where.update(dict.fromkeys(handle.keys(), file))

    def original(name: str) -> torch.Tensor:
        if name not in where or name not in frozen:
            raise ValueError(f"export: no pinned snapshot tensor for {name}")
        with safe_open(str(model_path / where[name]), framework="pt") as handle:
            value = handle.get_tensor(name)
        parameter = frozen[name].detach()
        if value.dtype != torch.bfloat16 or not torch.equal(value.to(parameter.dtype), parameter.cpu()):
            raise ValueError(f"export: policy base differs from the pinned snapshot at {name}")
        return value
    return original


def _numel(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result
