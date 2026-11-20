"""Dedicated, cache-only local v16 smoke/L0 runner. Never launches a cloud job."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

from fine_tuning.notebook_rl.input_preparation import digest

LOCAL_IMAGE = {"preprocessing": "qwen_vl_utils_0_0_14", "min_pixels": 262144,
               "max_pixels": 1310720, "max_images_per_problem": 2}
SMOKE_IMAGE = LOCAL_IMAGE | {"max_pixels": 262144}
DEFAULT_CACHE = Path.home() / "mmdl-artifacts/notebook-rl-local-nf4-20261021/base-nf4-v3"
DEFAULT_MODEL = Path.home() / Path("mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"
                     "ebb281ec70b05090aa6165b016eac8ec08e71b17")
HERE = Path(__file__).resolve().parents[2]
BINDINGS = {"source_sha256", "dataset_sha256", "f0_sha256", "protocol_sha256", "preregistration_sha256"}


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _valid_batch(value: Any) -> bool:
    return type(value) is int and 1 <= value <= 32


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_new(path: Path, value: Any) -> None:
    wire = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(wire)
        stream.flush()
        os.fsync(stream.fileno())


def exact_paired(recovered: int, lost: int) -> dict:
    """Exact two-sided McNemar test, conditional on the discordant count."""
    if any(type(value) is not int or value < 0 for value in (recovered, lost)):
        raise ValueError("v16 gate: nonnegative integer discordant counts required")
    discordant = recovered + lost
    numerator = 2 * sum(math.comb(discordant, k) for k in range(min(recovered, lost) + 1))
    return {"recovered": recovered, "lost": lost, "net": recovered - lost,
            "p_value": min(1.0, numerator / (1 << discordant))}


def _grade(record: dict) -> dict:
    from fine_tuning.notebook_rl.answers import compare, extract_eval_aligned
    from fine_tuning.notebook_rl.schema import VerifierItem

    spec = record["evaluation"]
    extracted = extract_eval_aligned(record["text"], record["finish_reason"], spec["question"],
                                     tuple(spec["choices"]), spec["answer_type"])
    item = VerifierItem(record["item_id"], spec["answer_type"], spec["gold"], spec["accept"],
                        tuple(spec["choices"]), {}, {}, (), (), ())
    if compare(item.gold, item) is None:
        raise ValueError("v16 gate: evaluation gold is not comparable")
    return {"correct": extracted.answer is not None and compare(extracted.answer, item) is True,
            "missing": extracted.answer is None, "length": record["finish_reason"] == "length",
            "tokens": len(record["completion_token_ids"])}


def _records(rows: list[dict], ids: list[str], prereg: dict, *, observed: bool = False) -> dict[str, dict]:
    from .contract import sample_seed

    if not isinstance(rows, list) or [row.get("item_id") for row in rows] != ids:
        raise ValueError("v16 gate: incomplete, reordered, or duplicate item records")
    result = {}
    for row in rows:
        body = {key: value for key, value in row.items() if key != "sha256"}
        tokens = row.get("completion_token_ids")
        if (row.get("sha256") != digest(body) or not isinstance(row.get("text"), str)
                or row.get("finish_reason") not in ("stop", "length")
                or not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or len(tokens) > prereg["max_new_tokens"]
                or row.get("seed") != sample_seed(row["item_id"], "t1", prereg["master_seed"])
                or not _sha(row.get("input_sha256"))):
            raise ValueError("v16 gate: malformed raw generation evidence")
        if observed:
            if not isinstance(row.get("target"), str) or not row["target"]:
                raise ValueError("v16 gate: missing held-out observation reference")
            result[row["item_id"]] = row
        else:
            result[row["item_id"]] = _grade(row)
    return result


def observation_summary(rows: list[dict]) -> dict:
    """Strict authored-reference coverage; unsupported paraphrases remain unmatched."""
    import re

    matched = total = exact = 0
    for row in rows:
        target, text = row["target"], row["text"]
        # Canonical target clauses are the same program-authored fact units in both arms.
        clauses = [value.strip(" .") for value in re.split(r"[;\n]|, (?=\d+(?:-|->))", target) if value.strip(" .")]
        matched += sum(clause in text for clause in clauses)
        total += len(clauses)
        exact += text.strip() == target.strip()
    return {"reference_units_matched": matched, "reference_units": total,
            "strict_reference_coverage": matched / total if total else None,
            "exact_parent_targets": exact, "parents": len(rows),
            "limitation": "Exact canonical clauses only; paraphrases and extra unsupported claims are not judged."}


def sentinel_stop(baseline: list[dict], current: list[dict]) -> list[str]:
    if len(baseline) != 16 or len(current) != 16:
        raise ValueError("v16 sentinel: exactly sixteen preregistered parents required")
    before, after = [_grade(row) for row in baseline], [_grade(row) for row in current]
    reasons = []
    if statistics.median(row["tokens"] for row in after) < 0.5 * statistics.median(row["tokens"] for row in before):
        reasons.append("SENTINEL_MEDIAN_LOW")
    if sum(row["missing"] for row in after) >= sum(row["missing"] for row in before) + 4:
        reasons.append("SENTINEL_FINAL_MISSING")
    if sum(row["length"] for row in after) >= sum(row["length"] for row in before) + 3:
        reasons.append("SENTINEL_LENGTH_LIMIT")
    return reasons


def _validate_effect_evidence(gate: Mapping[str, Any], *, expected_binding: Mapping[str, str] | None = None,
                              precision: str = "nf4") -> dict:
    """Recompute a sealed full L0 decision; smoke and caller-supplied PASS flags convey no authority."""
    try:
        body = {key: value for key, value in gate.items() if key != "sha256"}
        binding, pre, protocol = gate["binding"], gate["preregistration"], gate["protocol"]
        bf16 = precision == "bf16"
        if (gate.get("schema") != ("v16-bf16-effect-v1" if bf16 else "v16-l0-gate-v1")
                or gate.get("scope") != ("bf16-O-effect" if bf16 else "l0-full")
                or gate.get("sha256") != digest(body)
                or set(binding) != (BINDINGS | {"checkpoint_sha256"} if bf16 else BINDINGS)
                or any(not _sha(value) for value in binding.values())
                or binding["preregistration_sha256"] != digest(pre)
                or pre.get("schema") != ("v16-effect-prereg-v1" if bf16 else "v16-local-prereg-v1")
                or pre.get("mode") != ("bf16-effect" if bf16 else "l0")
                or any(binding[key] != pre[key] for key in ("source_sha256", "dataset_sha256", "f0_sha256"))
                or protocol.get("sha256") != binding["protocol_sha256"]
                or protocol["sha256"] != digest({k: v for k, v in protocol.items() if k != "sha256"})
                or protocol.get("precision") != precision
                or protocol.get("image") != (pre.get("image") if bf16 else LOCAL_IMAGE)
                or bf16 and (pre.get("image", {}).get("min_pixels") != 1003520
                             or pre.get("image", {}).get("max_pixels") != 4014080)
                or pre["source_sha256"] != digest(pre["source_files"])
                or not _valid_batch(pre.get("generation_batch_size"))
                or pre["master_seed"] != 3407 or pre["max_new_tokens"] != 8192):
            raise ValueError("v16 gate: full sealed local L0 evidence required")
        if expected_binding and any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ValueError("v16 gate: expected source/data/F0 binding differs")
        dev_ids, sentinel_ids = pre["dev_ids"], pre["sentinel_ids"]
        train_ids = {pid for block in pre["train_blocks"] for pid in block}
        if (not dev_ids or len(set(dev_ids)) != len(dev_ids) or len(sentinel_ids) != 16
                or len(set(sentinel_ids)) != 16 or set(dev_ids) & set(sentinel_ids)
                or train_ids & (set(dev_ids) | set(sentinel_ids))
                or pre["min_net"] != math.ceil(13 * len(dev_ids) / 256)
                or len(pre["train_blocks"]) != 16 or len(train_ids) != 128
                or any(len(block) != 8 or len(block) != len(set(block)) for block in pre["train_blocks"])):
            raise ValueError("v16 gate: preregistered disjoint cohorts and net-gain rule required")
        updates = gate["updates"]
        if (len(updates) != 16 or [row["step"] for row in updates] != list(range(1, 17))
                or any(row["problem_ids"] != pre["train_blocks"][row["step"] - 1]
                       or not math.isfinite(row["loss"]) for row in updates)):
            raise ValueError("v16 gate: sixteen complete finite updates required")
        contract = protocol["training_contract"]
        if (contract["pg_weight"] != 0 or bf16 and contract["stage_a_blocks"] != 16
                or not bf16 and contract["weights"] !=
                {"O": 1.0, "J": 0.0, "C": 0.0, "N": 0.0, "A1": 0.0}
                or contract["hashes"]["source"] != binding["source_sha256"]
                or contract["hashes"]["dataset"] != binding["dataset_sha256"]
                or not bf16 and contract["hashes"]["order"] != digest(pre["train_blocks"])
                or protocol["optimizer"]["lr"] != {"L": 5e-5, "M": 1e-5, "V": 1e-5}
                or (protocol["optimizer"]["total_steps"] < 16 if bf16 else protocol["optimizer"]["total_steps"] != 16)
                or protocol["optimizer"]["warmup_steps"] != 4):
            raise ValueError("v16 gate: executed training protocol differs from fixed O-only L0")
        metrics = gate["metrics"]
        f0, ft = (_records(metrics["dev"][arm], dev_ids, pre) for arm in ("F0", "FT"))
        for before, after in zip(metrics["dev"]["F0"], metrics["dev"]["FT"], strict=True):
            if any(before[key] != after[key] for key in ("input_sha256", "evaluation", "seed")):
                raise ValueError("v16 gate: P0 input, answer, or seed changed between arms")
        paired = exact_paired(sum(not f0[pid]["correct"] and ft[pid]["correct"] for pid in dev_ids),
                              sum(f0[pid]["correct"] and not ft[pid]["correct"] for pid in dev_ids))
        paired["n"] = len(dev_ids)
        reasons = []
        if paired["p_value"] >= 0.05 or paired["net"] < pre["min_net"]:
            reasons.append("P0_GAIN_NOT_ESTABLISHED")
        if paired["lost"] > paired["recovered"]:
            reasons.append("P0_LOSSES_EXCEED_RECOVERIES")
        if ((sum(row["missing"] for row in ft.values()) - sum(row["missing"] for row in f0.values()))
                / len(dev_ids) > 0.02):
            reasons.append("P0_FINAL_MISSING_INCREASE")
        if statistics.median(row["tokens"] for row in ft.values()) < 0.7 * statistics.median(
                row["tokens"] for row in f0.values()):
            reasons.append("P0_MEDIAN_LOW")
        for step in ("0", "8", "16"):
            _records(metrics["sentinel"][step], sentinel_ids, pre)
        for step in ("8", "16"):
            for before, after in zip(metrics["sentinel"]["0"], metrics["sentinel"][step], strict=True):
                if any(before[key] != after[key] for key in ("input_sha256", "evaluation", "seed")):
                    raise ValueError("v16 gate: sentinel identity changed")
            reasons.extend(f"{reason}:{step}" for reason in sentinel_stop(metrics["sentinel"]["0"],
                                                                         metrics["sentinel"][step]))
        for arm in ("F0", "FT"):
            _records(metrics["observations"][arm], dev_ids, pre, observed=True)
        for before, after in zip(metrics["observations"]["F0"], metrics["observations"]["FT"], strict=True):
            if any(before[key] != after[key] for key in ("input_sha256", "target", "seed")):
                raise ValueError("v16 gate: observation input, target, or seed changed")
        ablations = gate["diagnostics"]["ablations"]
        if (set(ablations) != {"on", "vision_off", "language_off", "merger_f0"}
                or any(len(values) != len(dev_ids) or any(type(v) not in (int, float) or not math.isfinite(v)
                                                        for v in values) for values in ablations.values())):
            raise ValueError("v16 gate: all four finite held-out ablation columns required")
        effects = {}
        for key in ("vision_off", "language_off", "merger_f0"):
            differences = [on - off for on, off in zip(ablations["on"], ablations[key], strict=True)]
            effects[key] = {"mean": statistics.mean(differences), "standard_error":
                            statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else None}
        for key in ("vision_off", "merger_f0"):
            if effects[key]["mean"] <= 0:
                reasons.append(f"NO_POSITIVE_CONTRIBUTION:{key}")
        bf16 = gate["diagnostics"]["bf16"]
        if (any(type(bf16[key]) not in (int, float) or not math.isfinite(bf16[key]) for key in
                ("changed_fraction", "cosine", "projection", "true_norm", "visible_norm"))
                or not 0 <= bf16["changed_fraction"] <= 1 or not -1 <= bf16["cosine"] <= 1):
            raise ValueError("v16 gate: finite BF16 visibility receipt required")
        reload = gate["reload"]
        if (reload.get("schema") != ("v16-effect-reload-v1" if precision == "bf16" else "v16-local-reload-v1")
                or reload.get("binding") != binding
                or reload.get("step") != 16 or reload.get("producer_pid") == reload.get("reload_pid")
                or reload.get("probe_max_abs_difference") != 0 or reload.get("greedy_equal") is not True
                or reload.get("state_equal") is not True or reload.get("runtime_equal") is not True
                or reload.get("sha256") != digest({k: v for k, v in reload.items() if k != "sha256"})):
            raise ValueError("v16 gate: fresh-process checkpoint/runtime/probe verification required")
        if precision == "bf16":
            checkpoint = gate["checkpoint"]
            if checkpoint.get("file_sha256") != binding["checkpoint_sha256"]:
                raise ValueError("v16 effect: trained checkpoint binding differs")
        correct_lengths = [_grade(row)["tokens"] for row in metrics["dev"]["F0"]
                           if row["finish_reason"] == "stop" and _grade(row)["correct"]]
        return {"verdict": "STOP" if reasons else "GO", "passed": not reasons, "paired": paired,
                "baseline_correct_length_median": statistics.median(correct_lengths) if correct_lengths else None,
                "reasons": reasons, "effects": effects,
                "merger_measurable": bf16["changed_fraction"] >= 0.2 and bf16["cosine"] >= 0.6,
                "observations": {arm: observation_summary(metrics["observations"][arm]) for arm in ("F0", "FT")},
                "scope": "bf16_O_effect_only" if precision == "bf16" else "local_nf4_only",
                "precision": precision, "benchmark_effect_established": False}
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError("v16 gate: incomplete or malformed raw evidence") from error


def validate_l0_gate(gate: Mapping[str, Any], *, expected_binding: Mapping[str, str] | None = None) -> dict:
    return _validate_effect_evidence(gate, expected_binding=expected_binding, precision="nf4")


def validate_bf16_effect_receipt(gate: Mapping[str, Any], *, expected_binding: Mapping[str, str] | None = None) -> dict:
    return _validate_effect_evidence(gate, expected_binding=expected_binding, precision="bf16")


def validate_local_functional_receipt(receipt: Mapping[str, Any], *,
                                      expected_binding: Mapping[str, str] | None = None) -> dict:
    """A real one-update/reload smoke establishes readiness, never an effect gate."""
    try:
        binding, pre, protocol = receipt["binding"], receipt["preregistration"], receipt["protocol"]
        result, reload, metadata = receipt["result"], receipt["reload"], receipt["checkpoint_metadata"]
        if (receipt.get("schema") != "v16-local-functional-v1" or receipt.get("scope") != "local-nf4-functional"
                or receipt.get("sha256") != digest({k: v for k, v in receipt.items() if k != "sha256"})
                or set(binding) != BINDINGS or any(not _sha(value) for value in binding.values())
                or pre.get("schema") != "v16-local-prereg-v1" or pre.get("mode") != "smoke"
                or pre["source_sha256"] != digest(pre["source_files"])
                or pre["f0_sha256"] != digest(pre["f0_files"])
                or not any(name.endswith(".safetensors") for name in pre["f0_files"])
                or "nf4_base.json" in pre["f0_files"]
                or binding["preregistration_sha256"] != digest(pre)
                or any(binding[key] != pre[key] for key in ("source_sha256", "dataset_sha256", "f0_sha256"))
                or protocol.get("precision") != "nf4" or protocol.get("image") != SMOKE_IMAGE
                or protocol["sha256"] != binding["protocol_sha256"]
                or protocol["sha256"] != digest({k: v for k, v in protocol.items() if k != "sha256"})
                or result.get("mode") != "smoke" or result.get("step") != 1 or result.get("binding") != binding
                or result.get("stop_reasons") or not result.get("functional_only")
                or len(pre["train_blocks"]) != 1 or not 1 <= len(pre["train_blocks"][0]) <= 2
                or len(set(pre["train_blocks"][0])) != len(pre["train_blocks"][0])
                or any(type(result["changed"][group]) is not int or result["changed"][group] <= 0 for group in ("L", "M", "V"))):
            raise ValueError("v16 functional: sealed real NF4 smoke evidence required")
        if expected_binding and any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ValueError("v16 functional: source/data/F0 binding differs")
        update, = result["updates"]
        if (update["step"] != 1 or update["problem_ids"] != pre["train_blocks"][0]
                or not math.isfinite(update["loss"])
                or any(not math.isfinite(update["gradient_norms"][group]) or update["gradient_norms"][group] <= 0
                       for group in ("L", "M", "V"))
                or metadata["step"] != 1 or metadata["cursor"]["position"] != 1
                or metadata["cursor"]["optimizer_steps"] != 1 or metadata["protocol"] != protocol
                or metadata["binding"]["code_sha256"] != binding["source_sha256"]
                or metadata["binding"]["data_sha256"] != binding["dataset_sha256"]
                or metadata["binding"]["protocol_sha256"] != binding["protocol_sha256"]
                or not metadata["state_sha256"] or not _sha(receipt["checkpoint_metadata_file_sha256"])):
            raise ValueError("v16 functional: update/checkpoint evidence differs")
        if (reload.get("schema") != "v16-local-reload-v1" or reload.get("binding") != binding
                or reload.get("sha256") != digest({k: v for k, v in reload.items() if k != "sha256"})
                or reload.get("step") != 1 or reload.get("producer_pid") != result["producer_pid"]
                or reload.get("producer_pid") == reload.get("reload_pid")
                or reload.get("probe_max_abs_difference") != 0
                or any(reload.get(key) is not True for key in ("state_equal", "runtime_equal", "greedy_equal"))):
            raise ValueError("v16 functional: exact fresh-process reload required")
        return {"ready": True, "effect_established": False, "precision": "nf4", "scope": "functional_readiness_only"}
    except (KeyError, TypeError) as error:
        raise ValueError("v16 functional: incomplete raw evidence") from error


def _source_files() -> dict:
    from .single import source_manifest

    return source_manifest()["actual_files"]


def _f0_files(model_path: Path) -> dict:
    from fine_tuning.modules import BASE_REVISION

    if model_path.name != BASE_REVISION or not (model_path / "config.json").is_file():
        raise ValueError("v16 local: original pinned BF16 snapshot required")
    files = {path.name: file_sha256(path) for path in sorted(model_path.iterdir())
             if path.is_file() and path.suffix in (".json", ".safetensors", ".jinja")}
    if not any(name.endswith(".safetensors") for name in files) or (model_path / "nf4_base.json").exists():
        raise ValueError("v16 local: F0 identity cannot come from a quantized cache")
    return files


def _verify_prereg(pre: dict, manifest_sha256: str) -> None:
    if (pre.get("schema") != "v16-local-prereg-v1" or pre.get("mode") not in ("smoke", "l0")
            or pre.get("dataset_sha256") != manifest_sha256 or pre.get("source_files") != _source_files()
            or pre.get("source_sha256") != digest(pre["source_files"])
            or not _sha(pre.get("f0_sha256")) or pre.get("f0_sha256") != digest(pre["f0_files"])
            or pre["f0_files"] != _f0_files(Path(pre["model_path"])) or pre.get("master_seed") != 3407
            or pre.get("image") != (SMOKE_IMAGE if pre.get("mode") == "smoke" else LOCAL_IMAGE)
            or not _valid_batch(pre.get("generation_batch_size"))):
        raise ValueError("v16 local: preregistration source/data/F0/image/batch binding differs")
    if file_sha256(Path(pre["cache"]) / "nf4_base.json") != pre["cache_metadata_sha256"]:
        raise ValueError("v16 local: pinned NF4 cache changed")


def prepare(args: argparse.Namespace) -> int:
    from .content_v16 import load_content_manifest

    manifest = load_content_manifest(args.manifest, args.manifest_sha256)
    training = [row["problem_id"] for row in manifest["rows"]]
    if args.mode == "smoke":
        training = training[:2]
        dev, sentinel, blocks, cap = [], [], [training], 32
    else:
        heldout = [item["problem_id"] for item in manifest["items"] if item["split"] == "dev"]
        sentinel = [item["problem_id"] for item in manifest["items"] if item["split"] == "sentinel"]
        dev = [pid for pid in heldout if pid not in sentinel]
        if not training or len(sentinel) != 16 or not dev:
            raise ValueError("v16 local: admitted train parents, sixteen sentinels, and separate dev required")
        if len(training) != 128:
            raise ValueError("v16 local: INSUFFICIENT_ELIGIBLE_DATA: sixteen unique eight-parent blocks required")
        blocks = [training[offset:offset + 8] for offset in range(0, 128, 8)]
        cap = 8192
    files = _source_files()
    cache_hash = file_sha256(args.cache / "nf4_base.json")
    f0_files = _f0_files(args.model_path)
    if args.f0_sha256 is not None and args.f0_sha256 != digest(f0_files):
        raise ValueError("v16 local: supplied F0 identity differs from the original pinned BF16 snapshot")
    pre = {"schema": "v16-local-prereg-v1", "mode": args.mode, "dataset_sha256": args.manifest_sha256,
           "source_files": files, "source_sha256": digest(files), "f0_sha256": digest(f0_files), "f0_files": f0_files,
           "cache": str(args.cache.resolve()), "model_path": str(args.model_path.resolve()),
           "cache_metadata_sha256": cache_hash, "master_seed": 3407,
           "generation_batch_size": 1 if args.mode == "smoke" else args.generation_batch_size,
           "image": SMOKE_IMAGE if args.mode == "smoke" else LOCAL_IMAGE,
           "train_blocks": blocks, "dev_ids": dev, "sentinel_ids": sentinel, "max_new_tokens": cap,
           "min_net": math.ceil(13 * len(dev) / 256), "min_net_rule": "ceil(13*n/256)",
           "claim": "local_quantized_functional_only" if args.mode == "smoke" else "local_quantized_L0_only"}
    _verify_prereg(pre, args.manifest_sha256)
    _write_new(args.out, pre)
    print(json.dumps({"preregistration": str(args.out), "file_sha256": file_sha256(args.out),
                      "canonical_sha256": digest(pre), "source_sha256": pre["source_sha256"]}))
    return 0


def _local_protocol(pre: dict) -> dict:
    from .protocol import plain_protocol_content_v16

    # Local diagnostic image limits are explicitly sealed; they are never a BF16 campaign protocol.
    sources = sorted({row["layer"] for row in pre["rows"]})
    pins = {"source": pre["source_sha256"], "dataset": pre["dataset_sha256"],
            "order": digest(pre["train_blocks"]), "data_admission": digest(pre["rows"])}
    protocol = plain_protocol_content_v16("nf4", 16, contracts=pins,
                                          source_weights={source: 1 / len(sources) for source in sources},
                                          weights={"O": 1.0, "J": 0.0, "C": 0.0, "N": 0.0, "A1": 0.0}, stage_a_blocks=16)
    protocol["image"] = dict(pre["image"])
    protocol["local_diagnostic"] = {"mode": pre["mode"], "non_reportable_benchmark": True}
    protocol["sha256"] = digest({key: value for key, value in protocol.items() if key != "sha256"})
    return protocol


def _setup_execution(seed: int) -> None:
    import random
    import numpy as np
    import torch
    from fine_tuning.notebook_rl.policy import enable_deterministic_execution

    enable_deterministic_execution()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    from .content_v16 import load_content_manifest

    manifest = load_content_manifest(args.manifest, args.manifest_sha256)
    if file_sha256(args.prereg) != args.prereg_sha256:
        raise ValueError("v16 local: preregistration file hash mismatch")
    pre = json.loads(args.prereg.read_text(encoding="utf-8"))
    _verify_prereg(pre, args.manifest_sha256)
    items = {item["problem_id"]: item for item in manifest["items"]}
    training = [pid for block in pre["train_blocks"] for pid in block]
    if (not training or any(pid not in items or items[pid]["split"] != "train" for pid in training)
            or any(pid not in items or items[pid]["split"] != "dev" for pid in pre["dev_ids"])
            or any(pid not in items or items[pid]["split"] != "sentinel" for pid in pre["sentinel_ids"])):
        raise ValueError("v16 local: preregistered cohort is not admitted to its split")
    cohorts = [training, pre["dev_ids"], pre["sentinel_ids"]]
    for key in ("parent_id", "group_id", "image_sha256"):
        groups = [{items[pid][key] for pid in cohort} for cohort in cohorts]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("v16 local: train/dev/sentinel parent, graph, or image overlap")
    return manifest, pre


def _images(item: dict, manifest: Path) -> list:
    from PIL import Image

    path = Path(item["image_path"])
    with Image.open(path if path.is_absolute() else manifest.parent / path) as image:
        return [image.convert("RGB")]


def _encode(bundle: Any, rows: list[dict], items: dict, manifest: Path, *, image: dict) -> list[dict]:
    from .train_v16 import encode_content_row

    return [encode_content_row(bundle.processor, bundle.model.config, row,
                               _images(items[row["problem_id"]], manifest), image=image) for row in rows]


def generate_many(bundle: Any, jobs: list[dict], *, manifest: Path, pre: dict, out: Path, batch_size: int) -> list[dict]:
    """Persist STARTED before each chunk is sampled; interrupted requests are never silently regenerated.

    Jobs are {"item", "prompt", "target", "phase"}; pending jobs run in consecutive chunks of at most batch_size.
    """
    from .collect import _single_generate_batch, append_jsonl
    from .contract import sample_seed
    from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
    from fine_tuning.notebook_rl.prompts import user_message

    if not _valid_batch(batch_size):
        raise ValueError("v16 local: generation batch size must be an integer from 1 to 32")
    ledger = out / "requests.jsonl"
    started = [] if not ledger.exists() else [json.loads(line) for line in ledger.read_text().splitlines()]

    def prepare(job: dict) -> dict:
        return prepare_protocol_inputs(bundle.processor, bundle.model.config, [user_message(job["prompt"], 1)],
                                       _images(job["item"], manifest), pre["image"])[0]

    records: list[dict | None] = []
    ids, pending = [], []
    for index, job in enumerate(jobs):
        item, target = job["item"], job["target"]
        request_id = digest({"phase": job["phase"], "item_id": item["problem_id"]})
        if request_id in ids:
            raise ValueError("v16 local: duplicate request within one generation batch")
        ids.append(request_id)
        record_path = out / "generations" / f"{request_id}.json"
        matching = [row for row in started if row["request_id"] == request_id]
        if matching:
            if not record_path.is_file():
                raise ValueError("v16 local: STARTED request has no durable result; manual recovery required")
            prepared = prepare(job)
            stored = json.loads(record_path.read_text(encoding="utf-8"))
            if (stored.get("input_sha256") != prepared["input_sha256"] or stored.get("sha256") != digest({k: v for k, v in stored.items() if k != "sha256"})
                    or stored.get("input_sha256") != matching[0]["input_sha256"]
                    or target is not None and stored.get("target") != target
                    or target is None and stored.get("evaluation") != item["evaluation"]
                    or any(row.get("sha256") != stored["sha256"] for row in matching if row["status"] == "COMPLETED")):
                raise ValueError("v16 local: stored single response changed")
            records.append(stored)
        else:
            if record_path.exists():
                raise ValueError("v16 local: response without request ownership ledger")
            records.append(None)
            pending.append(index)
    if not pending:
        return records
    eos = bundle.model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    if not eos_ids or any(type(token) is not int for token in eos_ids):
        raise ValueError("v16 local: explicit integer EOS tokens required")
    was_training = bundle.model.training
    bundle.model.eval()
    try:
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset:offset + batch_size]
            prepared = [prepare(jobs[index]) for index in chunk]
            requests = [{"request_id": ids[index], "problem_id": jobs[index]["item"]["problem_id"],
                         "stage": jobs[index]["phase"], "block_id": jobs[index]["phase"],
                         "seed": sample_seed(jobs[index]["item"]["problem_id"], "t1", pre["master_seed"]),
                         "max_tokens": pre["max_new_tokens"]} for index in chunk]
            for request, inputs in zip(requests, prepared, strict=True):
                append_jsonl(ledger, request | {"status": "STARTED", "input_sha256": inputs["input_sha256"]})
            sampled = _single_generate_batch(bundle, requests, prepared, [], eos=eos, eos_ids=eos_ids,
                pad=bundle.model.generation_config.pad_token_id, stop_at=None, loop_stop=False)
            for row, (index, request, inputs) in enumerate(zip(chunk, requests, prepared, strict=True)):
                job = jobs[index]
                tokens = sampled["tokens"][row]
                record = {"item_id": request["problem_id"], "seed": request["seed"], "input_sha256": inputs["input_sha256"],
                          "text": bundle.processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=False),
                          "completion_token_ids": tokens, "finish_reason": sampled["reasons"][row]}
                record.update({"target": job["target"]} if job["target"] is not None
                              else {"evaluation": job["item"]["evaluation"]})
                record["sha256"] = digest(record)
                _write_new(out / "generations" / f"{request['request_id']}.json", record)
                append_jsonl(ledger, {"request_id": request["request_id"], "status": "COMPLETED", "sha256": record["sha256"]})
                records[index] = record
    finally:
        bundle.model.train(was_training)
    return records


def generate_once(bundle: Any, item: dict, *, prompt: str, target: str | None, manifest: Path,
                  pre: dict, phase: str, out: Path) -> dict:
    """One request is a one-job, batch-size-one generate_many."""
    return generate_many(bundle, [{"item": item, "prompt": prompt, "target": target, "phase": phase}],
                         manifest=manifest, pre=pre, out=out, batch_size=1)[0]


def _p0(item: dict) -> str:
    spec = item["evaluation"]
    text = f"Question: {spec['question']}"
    if spec["answer_type"] == "mcq":
        options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(spec["choices"]))
        text += f"\nOptions:\n{options}\nPlease select the correct answer from the options above."
    if spec.get("hint"):
        text = f"Hint: {spec['hint']}\n" + text
    return text


def run(args: argparse.Namespace, *, policy_factory: Any = None, device: str = "cuda:0") -> int:
    from fine_tuning.notebook_rl import checkpoint as ckpt
    from fine_tuning.notebook_rl.quantized_policy import assert_quantized_base_unchanged, load_quantized_policy
    from . import train_v16
    from .content_v16 import observation_row, validate_content_row
    from .protocol import ARM
    from .train import NonFiniteStop, _clip_groups, compute_probe

    manifest, pre = _load_inputs(args)
    _setup_execution(pre["master_seed"])
    if pre["mode"] != args.command:
        raise ValueError("v16 local: requested scope differs from preregistration")
    items = {item["problem_id"]: item for item in manifest["items"]}
    rows = {row["problem_id"]: row for row in manifest["rows"]}
    if args.command == "smoke" and (len(pre["train_blocks"]) != 1 or not 1 <= len(pre["train_blocks"][0]) <= 2):
        raise ValueError("v16 smoke: at most two genuine parents and exactly one update")
    protocol = _local_protocol(pre | {"rows": manifest["rows"]})
    binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
               "data_sha256": pre["dataset_sha256"], "code_sha256": pre["source_sha256"]}
    gate_binding = {"source_sha256": pre["source_sha256"], "dataset_sha256": pre["dataset_sha256"],
                    "f0_sha256": pre["f0_sha256"], "protocol_sha256": protocol["sha256"],
                    "preregistration_sha256": digest(pre)}
    factory = policy_factory or (lambda path, arm, config: load_quantized_policy(Path(pre["cache"]), arm, config))
    if args.out.exists():
        if not args.resume:
            raise ValueError("v16 local: existing result directory requires explicit --resume")
        metadata = json.loads((args.out / "run.json").read_text())
        if metadata["binding"] != binding or metadata["preregistration"] != pre:
            raise ValueError("v16 local: resume binding differs")
        checkpoints = sorted(args.out.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
        if not checkpoints or (args.out / "result.json").exists():
            raise ValueError("v16 local: no resumable incomplete checkpoint")
        bundle, optimizer, scheduler, step, cursor = ckpt.load_resume(Path(pre["model_path"]), checkpoints[-1],
            protocol, binding=binding, policy_factory=factory, device=device)
        updates = json.loads((args.out / f"updates-{step}.json").read_text())
        if cursor["position"] != step:
            raise ValueError("v16 local: checkpoint cursor differs")
    else:
        args.out.mkdir(parents=True)
        (args.out / "generations").mkdir()
        _write_new(args.out / "run.json", {"binding": binding, "gate_binding": gate_binding,
            "protocol": protocol, "preregistration": pre, "manifest": str(args.manifest.resolve()),
            "manifest_sha256": args.manifest_sha256, "producer_pid": os.getpid()})
        bundle = factory(Path(pre["model_path"]), ARM, protocol)
        optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
        step, updates = 0, []
    base_mergers = {name: parameter.detach().cpu().clone() for name, parameter in bundle.model.named_parameters()
                    if name in {row["name"] for row in bundle.inventory["trainable"] if row["group"] == "M"}}
    # Resume diagnostics must compare against the original modules, never the resumed trained values.
    if step:
        base_mergers = {name: bundle.model.get_parameter(name.replace(".modules_to_save.default.", ".original_module."))
                        .detach().cpu().float().clone() for name in base_mergers}
    initial_path = args.out / "initial-state.json"
    if step:
        initial = json.loads(initial_path.read_text())
    else:
        initial = {name: ckpt.tensor_sha256(parameter) for name, parameter in bundle.model.named_parameters()
                   if name in {row["name"] for row in bundle.inventory["trainable"]}}
        _write_new(initial_path, initial)
    def evaluate(ids: list[str], phase: str, *, observation: bool = False) -> list[dict]:
        jobs = []
        for pid in ids:
            item = items[pid]
            target = None
            if observation:
                row = observation_row(item, layer=item["layer"], purpose="diagnostic")
                validate_content_row(row, item, purpose="diagnostic")
                prompt, target = row["prompt"], row["target"]
            else:
                prompt = _p0(item)
            jobs.append({"item": item, "prompt": prompt, "target": target, "phase": phase})
        return generate_many(bundle, jobs, manifest=args.manifest, pre=pre, out=args.out,
                             batch_size=pre["generation_batch_size"])
    metrics: dict[str, dict[str, list[dict]]] = {"dev": {}, "sentinel": {}, "observations": {}}
    metrics_path = args.out / "metrics-f0.json"
    if args.command == "l0":
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        elif step:
            raise ValueError("v16 local: resumed update cannot regenerate F0")
        else:
            metrics["dev"]["F0"] = evaluate(pre["dev_ids"], "F0-P0")
            metrics["observations"]["F0"] = evaluate(pre["dev_ids"], "F0-O", observation=True)
            metrics["sentinel"]["0"] = evaluate(pre["sentinel_ids"], "sentinel-0")
            _write_new(metrics_path, metrics)
    stop_reasons = []
    if args.command == "l0" and step >= 8:
        sentinel_path = args.out / "sentinel-8.json"
        if sentinel_path.exists():
            metrics["sentinel"]["8"] = json.loads(sentinel_path.read_text())
        elif step == 8:
            metrics["sentinel"]["8"] = evaluate(pre["sentinel_ids"], "sentinel-8")
            _write_new(sentinel_path, metrics["sentinel"]["8"])
        else:
            raise ValueError("v16 local: missing checkpoint-8 sentinel evidence; manual recovery required")
        stop_reasons = sentinel_stop(metrics["sentinel"]["0"], metrics["sentinel"]["8"])
        if stop_reasons:
            raise ValueError("v16 local: cannot resume past a fired sentinel")
    for index in range(step, len(pre["train_blocks"])):
        block_ids = pre["train_blocks"][index]
        if any(items[pid]["split"] != "train" for pid in block_ids):
            raise ValueError("v16 local: held-out update forbidden")
        encoded = _encode(bundle, [rows[pid] for pid in block_ids], items, args.manifest, image=pre["image"])
        optimizer.zero_grad(set_to_none=True)
        try:
            report = train_v16.backward_content_block(bundle, encoded, protocol["training_contract"], components=("O",))
            norms = _clip_groups(bundle)
        except NonFiniteStop:
            optimizer.zero_grad(set_to_none=True)
            _write_new(args.out / "stop.json", {"schema": "v16-local-stop-v1", "step": index,
                "reason": "NONFINITE_LOSS_OR_GRADIENT", "verdict": "STOP", "binding": gate_binding})
            return 2
        if not all(norms[group] is not None and norms[group] > 0 for group in ("L", "M", "V")):
            raise ValueError("v16 local: all three trainable groups require finite nonzero gradients")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        updates.append({"step": index + 1, "problem_ids": block_ids, "loss": report["loss"], "gradient_norms": norms})
        if policy_factory is None:
            assert_quantized_base_unchanged(bundle)
        step = index + 1
        checkpoint = args.out / f"checkpoint-{step}"
        ckpt.save(bundle, optimizer, scheduler, step, {"epoch": 0, "position": step, "optimizer_steps": step, "discarded_problem_ids": []},
                  ckpt.capture_rng(), checkpoint, protocol=protocol, binding=binding)
        _write_new(args.out / f"updates-{step}.json", updates)
        ckpt.read_metadata(checkpoint, binding=binding)
        import shutil

        for previous in args.out.glob("checkpoint-*"):
            if previous != checkpoint:
                ckpt.read_metadata(previous, binding=binding)
                shutil.rmtree(previous)
        if args.command == "l0" and step in (8, 16):
            metrics["sentinel"][str(step)] = evaluate(pre["sentinel_ids"], f"sentinel-{step}")
            stop_reasons = sentinel_stop(metrics["sentinel"]["0"], metrics["sentinel"][str(step)])
            _write_new(args.out / f"sentinel-{step}.json", metrics["sentinel"][str(step)])
            if stop_reasons:
                break
    probe_rows = _encode(bundle, [rows[pid] for pid in pre["train_blocks"][0][:2]], items, args.manifest, image=pre["image"])
    probe = compute_probe(bundle, probe_rows)
    _write_new(args.out / "probe.json", probe)
    parameters = dict(bundle.model.named_parameters())
    bf16 = train_v16.bf16_survival({name: parameters[name] for name in base_mergers}, base_mergers)
    changed = {group: sum(initial[row["name"]] != ckpt.tensor_sha256(parameters[row["name"]])
                          for row in bundle.inventory["trainable"] if row["group"] == group) for group in ("L", "M", "V")}
    result = {"schema": "v16-local-result-v1", "mode": args.command, "step": step, "binding": gate_binding,
              "updates": updates, "checkpoint": f"checkpoint-{step}", "bf16": bf16, "changed": changed,
              "stop_reasons": stop_reasons, "producer_pid": os.getpid(), "functional_only": args.command == "smoke",
              "benchmark_effect_established": False, "requires_fresh_process_reload": True}
    if args.command == "l0" and not stop_reasons:
        metrics["dev"]["FT"] = evaluate(pre["dev_ids"], "FT-P0")
        metrics["observations"]["FT"] = evaluate(pre["dev_ids"], "FT-O", observation=True)
        dev_rows = [observation_row(items[pid], layer=items[pid]["layer"], purpose="diagnostic") for pid in pre["dev_ids"]]
        encoded_dev = _encode(bundle, dev_rows, items, args.manifest, image=pre["image"])
        result["diagnostics"] = {"bf16": bf16, "ablations": train_v16.visual_contribution(
            bundle, encoded_dev, bundle.inventory["mergers"])}
    _write_new(args.out / "metrics.json", metrics)
    _write_new(args.out / "result.json", result)
    print(json.dumps({"mode": args.command, "step": step, "changed": changed, "stop_reasons": stop_reasons,
                      "status": "FUNCTIONAL_PENDING_FRESH_RELOAD" if args.command == "smoke" else "PENDING_FRESH_RELOAD"}))
    return 0


def _state_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
                and bool(torch.equal(left.detach().cpu(), right.detach().cpu())))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def reload_run(args: argparse.Namespace, *, policy_factory: Any = None, device: str = "cuda:0") -> int:
    from fine_tuning.notebook_rl import checkpoint as ckpt
    from fine_tuning.notebook_rl.quantized_policy import load_quantized_policy
    from .train import compute_probe

    run_meta = json.loads((args.run / "run.json").read_text())
    pre, protocol = run_meta["preregistration"], run_meta["protocol"]
    _verify_prereg(pre, run_meta["manifest_sha256"])
    _setup_execution(pre["master_seed"])
    if run_meta["producer_pid"] == os.getpid():
        raise ValueError("v16 local: reload must run in a fresh process")
    result = json.loads((args.run / "result.json").read_text())
    if result["producer_pid"] == os.getpid():
        raise ValueError("v16 local: fresh process must follow the last training process")
    factory = policy_factory or (lambda path, arm, config: load_quantized_policy(Path(pre["cache"]), arm, config))
    bundle, optimizer, scheduler, step, cursor = ckpt.load_resume(Path(pre["model_path"]),
        args.run / result["checkpoint"], protocol, binding=run_meta["binding"], policy_factory=factory, device=device)
    import torch

    runtime = torch.load(args.run / result["checkpoint"] / "runtime.pt", map_location="cpu", weights_only=True)
    runtime_equal = all(_state_equal(actual, runtime[key]) for key, actual in (
        ("optimizer", optimizer.state_dict()), ("scheduler", scheduler.state_dict()), ("rng", ckpt.capture_rng())))
    if not runtime_equal:
        raise ValueError("v16 local: loaded optimizer, scheduler, or RNG differs")
    del runtime
    if step != result["step"] or cursor.get("optimizer_steps") != step:
        raise ValueError("v16 local: reloaded step or optimizer-step cursor differs")
    from .content_v16 import load_content_manifest
    manifest_path = Path(run_meta["manifest"])
    manifest = load_content_manifest(manifest_path, run_meta["manifest_sha256"])
    items = {item["problem_id"]: item for item in manifest["items"]}
    rows = {row["problem_id"]: row for row in manifest["rows"]}
    encoded = _encode(bundle, [rows[pid] for pid in pre["train_blocks"][0][:2]], items, manifest_path, image=pre["image"])
    actual, expected = compute_probe(bundle, encoded), json.loads((args.run / "probe.json").read_text())
    differences = [abs(a - b) for aa, bb in zip(actual["logprobs"], expected["logprobs"], strict=True)
                   for a, b in zip(aa, bb, strict=True)]
    if not differences or max(differences) != 0 or actual["greedy_ids"] != expected["greedy_ids"] or cursor["position"] != step:
        raise ValueError("v16 local: fresh reload probe, greedy output, or cursor differs")
    receipt = {"schema": "v16-local-reload-v1", "binding": run_meta["gate_binding"], "step": step,
               "producer_pid": result["producer_pid"], "reload_pid": os.getpid(),
               "probe_max_abs_difference": max(differences), "greedy_equal": True, "state_equal": True,
               "runtime_equal": runtime_equal, "optimizer_groups": len(optimizer.param_groups),
               "scheduler_step": scheduler.last_epoch, "cursor": cursor}
    receipt["sha256"] = digest(receipt)
    _write_new(args.run / "reload.json", receipt)
    if pre["mode"] == "l0" and step == 16 and not result["stop_reasons"]:
        metrics = json.loads((args.run / "metrics.json").read_text())
        if (args.run / "sentinel-8.json").exists():
            metrics["sentinel"]["8"] = json.loads((args.run / "sentinel-8.json").read_text())
        gate = {"schema": "v16-l0-gate-v1", "scope": "l0-full", "binding": run_meta["gate_binding"],
                "preregistration": pre, "protocol": protocol, "metrics": metrics, "updates": result["updates"],
                "diagnostics": result["diagnostics"], "reload": receipt}
        gate["sha256"] = digest(gate)
        decision = validate_l0_gate(gate)
        _write_new(args.run / "l0-gate.json", gate)
        _write_new(args.run / "l0-decision.json", decision)
        print(json.dumps(decision))
    else:
        if pre["mode"] == "smoke":
            metadata_path = args.run / result["checkpoint"] / "metadata.json"
            functional = {"schema": "v16-local-functional-v1", "scope": "local-nf4-functional",
                "binding": run_meta["gate_binding"], "preregistration": pre, "protocol": protocol,
                "result": result, "reload": receipt,
                "checkpoint_metadata": ckpt.read_metadata(metadata_path.parent, binding=run_meta["binding"]),
                "checkpoint_metadata_file_sha256": file_sha256(metadata_path)}
            functional["sha256"] = digest(functional)
            validate_local_functional_receipt(functional)
            _write_new(args.run / "functional-receipt.json", functional)
        print(json.dumps({"status": "FUNCTIONAL_RELOAD_VERIFIED", "mode": pre["mode"], "step": step,
                          "L0_pass": False, "benchmark_effect_established": False}))
    return 0


def prepare_effect(args: argparse.Namespace) -> int:
    from .content_v16 import load_content_manifest
    from .protocol import validate_content_protocol_v16

    if not _valid_batch(args.generation_batch_size):
        raise ValueError("v16 effect: generation batch size must be an integer from 1 to 32")
    plan = json.loads(args.campaign_plan.read_text(encoding="utf-8"))
    if (plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"})
            or plan["protocol"]["precision"] != "bf16"):
        raise ValueError("v16 effect: sealed BF16 campaign plan required")
    validate_content_protocol_v16(plan["protocol"])
    descriptor = plan["inputs"]["content_manifest"]
    manifest_path = Path(descriptor["path"])
    manifest = load_content_manifest(manifest_path, descriptor["file_sha256"])
    heldout = [item for item in manifest["items"] if item["split"] == "dev"]
    sentinels = [item["problem_id"] for item in manifest["items"] if item["split"] == "sentinel"]
    if len(sentinels) != 16:
        raise ValueError("v16 effect: sixteen independently frozen sentinel parents required")
    dev = [item["problem_id"] for item in heldout if item["problem_id"] not in sentinels]
    blocks = plan["blocks"][:16]
    if (not dev or len(blocks) != 16 or any(len(block) != 8 for block in blocks)
            or len({pid for block in blocks for pid in block}) != 128
            or plan["protocol"]["training_contract"]["stage_a_blocks"] != 16):
        raise ValueError("v16 effect: INSUFFICIENT_ELIGIBLE_DATA: sixteen unique O-only eight-parent blocks required")
    pre = {"schema": "v16-effect-prereg-v1", "mode": "bf16-effect", "precision": "bf16",
        "source_files": plan["source"]["actual_files"], "source_sha256": digest(plan["source"]["actual_files"]),
        "dataset_sha256": descriptor["file_sha256"], "f0_sha256": plan["generation"]["model_files_sha256"],
        "campaign_plan_sha256": plan["sha256"], "protocol": plan["protocol"],
        "runtime_binding": {"protocol_sha256": plan["protocol"]["sha256"],
                            "prompts_sha256": plan["protocol"]["prompts_sha256"],
                            "data_sha256": plan["sha256"], "code_sha256": digest(plan["source"]["actual_files"])},
        "manifest_path": str(manifest_path.absolute()), "model_path": plan["model_path"],
        "train_blocks": blocks, "dev_ids": dev, "sentinel_ids": sentinels, "master_seed": 3407,
        "min_net": math.ceil(13 * len(dev) / 256), "min_net_rule": "ceil(13*n/256)",
        "max_new_tokens": 8192, "generation_batch_size": args.generation_batch_size,
        "image": plan["protocol"]["image"], "benchmark_effect_established": False}
    _write_new(args.out, pre)
    print(json.dumps({"preregistration": str(args.out), "file_sha256": file_sha256(args.out),
                      "canonical_sha256": digest(pre)}))
    return 0


def _effect_binding(pre: dict, checkpoint_sha256: str | None = None) -> dict:
    binding = {key: pre[key] for key in ("source_sha256", "dataset_sha256", "f0_sha256")}
    binding |= {"protocol_sha256": pre["protocol"]["sha256"], "preregistration_sha256": digest(pre)}
    if checkpoint_sha256 is not None:
        binding["checkpoint_sha256"] = checkpoint_sha256
    return binding


def evaluate_effect_bundle(bundle: Any, prereg: dict, manifest: dict, manifest_path: Path, *,
                           stage: str, output: Path, checkpoint: dict | None = None) -> dict:
    """Measure an existing BF16 policy at the sealed O-trial boundary; never perform an update."""
    from fine_tuning.notebook_rl import checkpoint as ckpt
    from .content_v16 import load_content_manifest, observation_row, validate_content_row
    from .train import compute_probe
    from .train_v16 import bf16_survival, visual_contribution

    if (stage not in ("0", "8", "16") or prereg.get("schema") != "v16-effect-prereg-v1"
            or prereg.get("precision") != "bf16" or prereg["protocol"]["precision"] != "bf16"
            or not _valid_batch(prereg.get("generation_batch_size"))
            or prereg["source_files"] != _source_files()
            or load_content_manifest(manifest_path, prereg["dataset_sha256"]) != manifest):
        raise ValueError("v16 effect: source-bound BF16 preregistration and admitted data required")
    output.mkdir(parents=True, exist_ok=True)
    (output / "generations").mkdir(exist_ok=True)
    path = output / f"phase{stage}.json"
    checkpoint_hash = None
    if stage != "0":
        if checkpoint is None:
            raise ValueError("v16 effect: durable checkpoint required before measuring its outputs")
        metadata_path = Path(checkpoint["path"])
        checkpoint_hash = file_sha256(metadata_path)
        if checkpoint_hash != checkpoint.get("file_sha256", checkpoint.get("sha256")):
            raise ValueError("v16 effect: checkpoint descriptor changed")
        checkpoint = {"path": str(metadata_path.absolute()), "file_sha256": checkpoint_hash}
        metadata = ckpt.read_metadata(metadata_path.parent, binding=prereg["runtime_binding"])
        if metadata["step"] != int(stage) or metadata["protocol"] != json.loads(json.dumps(prereg["protocol"])):
            raise ValueError("v16 effect: checkpoint stage/protocol differs")
        actual = {name: ckpt.tensor_sha256(parameter) for name, parameter in bundle.model.named_parameters()
                  if name in metadata["state_sha256"]}
        if actual != metadata["state_sha256"]:
            raise ValueError("v16 effect: live policy differs from the saved checkpoint")
    from .rl import policy_sha256

    ckpt.assert_base_unchanged(bundle)
    if stage == "0":
        import torch

        for name, parameter in bundle.model.named_parameters():
            if ".lora_B." in name and bool(torch.count_nonzero(parameter)):
                raise ValueError("v16 effect: baseline is not fresh F0")
            if ".modules_to_save.default." in name:
                original = bundle.model.get_parameter(name.replace(".modules_to_save.default.", ".original_module."))
                if not torch.equal(parameter.detach().cpu().float(), original.detach().cpu().float()):
                    raise ValueError("v16 effect: baseline Merger is not original F0")
    binding = _effect_binding(prereg, checkpoint_hash)
    start = {"schema": "v16-effect-phase-start-v1", "stage": stage, "binding": binding,
             "policy_sha256": policy_sha256(bundle), "source_sha256": prereg["source_sha256"],
             "dataset_sha256": prereg["dataset_sha256"]}
    start_path = output / f"phase{stage}-started.json"
    if start_path.exists():
        if json.loads(start_path.read_text()) != start:
            raise ValueError("v16 effect: interrupted phase policy/checkpoint binding changed")
    else:
        ledger = output / "requests.jsonl"
        if ledger.exists() and any(row.get("stage", "").startswith(f"BF16-{stage}-") for row in
                                   (json.loads(line) for line in ledger.read_text().splitlines())):
            raise ValueError("v16 effect: request history without a phase-start binding")
        _write_new(start_path, start)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["binding"] != binding or saved["sha256"] != digest({k: v for k, v in saved.items() if k != "sha256"}):
            raise ValueError("v16 effect: prior phase receipt changed")
        return saved
    items = {item["problem_id"]: item for item in manifest["items"]}
    was_training = bundle.model.training
    rng = ckpt.capture_rng()
    def evaluate(ids: list[str], *, observation: bool) -> list[dict]:
        jobs = []
        for pid in ids:
            item = items[pid]
            target = None
            if observation:
                row = observation_row(item, layer=item["layer"], purpose="diagnostic")
                validate_content_row(row, item, purpose="diagnostic")
                prompt, target = row["prompt"], row["target"]
            else:
                prompt = _p0(item)
            jobs.append({"item": item, "prompt": prompt, "target": target,
                         "phase": f"BF16-{stage}-{'O' if observation else 'P0'}"})
        return generate_many(bundle, jobs, manifest=manifest_path, pre=prereg, out=output,
                             batch_size=prereg["generation_batch_size"])
    result: dict[str, Any] = {"schema": "v16-bf16-effect-phase-v1", "stage": stage, "binding": binding,
              "producer_pid": os.getpid(), "checkpoint": checkpoint, "sentinel": [], "stop_reasons": []}
    try:
        result["sentinel"] = evaluate(prereg["sentinel_ids"], observation=False)
        if stage != "0":
            baseline = json.loads((output / "phase0.json").read_text())
            result["stop_reasons"] = sentinel_stop(baseline["sentinel"], result["sentinel"])
        if stage in ("0", "16"):
            result["dev"] = evaluate(prereg["dev_ids"], observation=False)
            result["observations"] = evaluate(prereg["dev_ids"], observation=True)
        if stage == "16":
            dev_rows = [observation_row(items[pid], layer=items[pid]["layer"], purpose="diagnostic")
                        for pid in prereg["dev_ids"]]
            encoded = _encode(bundle, dev_rows, items, manifest_path, image=prereg["image"])
            parameters = dict(bundle.model.named_parameters())
            names = [row["name"] for row in bundle.inventory["trainable"] if row["group"] == "M"]
            originals = {name: bundle.model.get_parameter(name.replace(".modules_to_save.default.", ".original_module."))
                         .detach().cpu().float() for name in names}
            result["diagnostics"] = {"ablations": visual_contribution(bundle, encoded, bundle.inventory["mergers"]),
                                      "bf16": bf16_survival({name: parameters[name] for name in names}, originals)}
            result["probe"] = compute_probe(bundle, encoded[:2])
    finally:
        bundle.model.train(was_training)
        ckpt.restore_rng(rng)
    result["sha256"] = digest(result)
    _write_new(path, result)
    return result


def evaluate_effect(args: argparse.Namespace, *, policy_factory: Any = None, device: str = "cuda:0") -> int:
    from fine_tuning.notebook_rl import checkpoint as ckpt
    from fine_tuning.notebook_rl.policy import build_policy
    from .content_v16 import load_content_manifest
    from .protocol import ARM

    if file_sha256(args.prereg) != args.prereg_sha256:
        raise ValueError("v16 effect: preregistration file hash mismatch")
    pre = json.loads(args.prereg.read_text())
    if digest(_f0_files(Path(pre["model_path"]))) != pre["f0_sha256"]:
        raise ValueError("v16 effect: original F0 model bytes changed")
    _setup_execution(pre["master_seed"])
    manifest_path = Path(pre["manifest_path"])
    manifest = load_content_manifest(manifest_path, pre["dataset_sha256"])
    descriptor = None
    if args.stage == "0":
        bundle = (policy_factory or build_policy)(Path(pre["model_path"]), ARM, pre["protocol"])
        bundle.model.to(device)
    else:
        if args.checkpoint is None:
            raise ValueError("v16 effect: --checkpoint metadata.json is required for stage 8/16")
        descriptor = {"path": str(args.checkpoint.absolute()), "file_sha256": file_sha256(args.checkpoint)}
        bundle, _, _, _, _ = ckpt.load_resume(Path(pre["model_path"]), args.checkpoint.parent, pre["protocol"],
            binding=pre["runtime_binding"], policy_factory=policy_factory, device=device)
    phase = evaluate_effect_bundle(bundle, pre, manifest, manifest_path, stage=args.stage,
                                   output=args.out, checkpoint=descriptor)
    print(json.dumps({"stage": args.stage, "sha256": phase["sha256"], "stop_reasons": phase["stop_reasons"]}))
    return 2 if phase["stop_reasons"] else 0


def seal_effect(args: argparse.Namespace, *, policy_factory: Any = None, device: str = "cuda:0") -> int:
    from fine_tuning.notebook_rl import checkpoint as ckpt
    from .content_v16 import load_content_manifest, observation_row
    from .train import compute_probe

    if file_sha256(args.prereg) != args.prereg_sha256:
        raise ValueError("v16 effect: preregistration hash mismatch")
    pre = json.loads(args.prereg.read_text())
    if digest(_f0_files(Path(pre["model_path"]))) != pre["f0_sha256"]:
        raise ValueError("v16 effect: original F0 model bytes changed")
    _setup_execution(pre["master_seed"])
    phases = {step: json.loads((args.effect_root / f"phase{step}.json").read_text()) for step in ("0", "8", "16")}
    final = phases["16"]
    for stage, phase in phases.items():
        if (phase.get("sha256") != digest({k: v for k, v in phase.items() if k != "sha256"})
                or phase.get("binding") != _effect_binding(pre, phase["binding"].get("checkpoint_sha256"))
                or phase.get("stage") != stage):
            raise ValueError("v16 effect: phase evidence changed")
    if final["producer_pid"] == os.getpid():
        raise ValueError("v16 effect: sealing requires a fresh checkpoint reload process")
    checkpoint = final["checkpoint"]
    metadata_path = Path(checkpoint["path"])
    if file_sha256(metadata_path) != checkpoint["file_sha256"]:
        raise ValueError("v16 effect: trained checkpoint changed")
    bundle, optimizer, scheduler, step, cursor = ckpt.load_resume(Path(pre["model_path"]), metadata_path.parent,
        pre["protocol"], binding=pre["runtime_binding"], policy_factory=policy_factory, device=device)
    import torch
    runtime = torch.load(metadata_path.parent / "runtime.pt", map_location="cpu", weights_only=True)
    if not all(_state_equal(value, runtime[key]) for key, value in (("optimizer", optimizer.state_dict()),
            ("scheduler", scheduler.state_dict()), ("rng", ckpt.capture_rng()))):
        raise ValueError("v16 effect: fresh runtime state differs")
    manifest_path = Path(pre["manifest_path"])
    manifest = load_content_manifest(manifest_path, pre["dataset_sha256"])
    items = {item["problem_id"]: item for item in manifest["items"]}
    rows = [observation_row(items[pid], layer=items[pid]["layer"], purpose="diagnostic") for pid in pre["dev_ids"][:2]]
    probe = compute_probe(bundle, _encode(bundle, rows, items, manifest_path, image=pre["image"]))
    if probe != final["probe"] or step != 16 or cursor["single"]["updates"] != 16:
        raise ValueError("v16 effect: fresh checkpoint probe or complete optimizer updates differ")
    binding = final["binding"]
    reload = {"schema": "v16-effect-reload-v1", "binding": binding, "step": step,
              "producer_pid": final["producer_pid"], "reload_pid": os.getpid(), "probe_max_abs_difference": 0,
              "greedy_equal": True, "state_equal": True, "runtime_equal": True}
    reload["sha256"] = digest(reload)
    updates = []
    for index, ids in enumerate(pre["train_blocks"]):
        stats = json.loads((args.campaign_output / f"block-{index}" / "update-stats.json").read_text())
        if stats["optimizer_steps"] != 1 or any(value for kind, value in stats["rows_by_kind"].items() if kind != "O"):
            raise ValueError("v16 effect: O-only update evidence required")
        updates.append({"step": index + 1, "problem_ids": ids, "loss": stats["stats"]["loss"],
                        "proof_sha256": digest(stats)})
    receipt = {"schema": "v16-bf16-effect-v1", "scope": "bf16-O-effect", "binding": binding,
        "preregistration": pre, "protocol": pre["protocol"], "checkpoint": checkpoint,
        "metrics": {"dev": {"F0": phases["0"]["dev"], "FT": final["dev"]},
                    "observations": {"F0": phases["0"]["observations"], "FT": final["observations"]},
                    "sentinel": {stage: phase["sentinel"] for stage, phase in phases.items()}},
        "updates": updates, "diagnostics": final["diagnostics"], "reload": reload}
    receipt["sha256"] = digest(receipt)
    decision = validate_bf16_effect_receipt(receipt)
    _write_new(args.out, receipt)
    _write_new(args.out.with_suffix(".decision.json"), decision)
    print(json.dumps(decision))
    return 0 if decision["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="CPU-only local run preregistration")
    prepare_parser.add_argument("--mode", choices=("smoke", "l0"), required=True)
    for name in ("manifest", "out"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--manifest-sha256", required=True)
    prepare_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    prepare_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    prepare_parser.add_argument("--f0-sha256")
    prepare_parser.add_argument("--generation-batch-size", type=int, default=2)
    for mode in ("smoke", "l0"):
        command = commands.add_parser(mode)
        for name in ("manifest", "prereg", "out"):
            command.add_argument(f"--{name}", type=Path, required=True)
        for name in ("manifest-sha256", "prereg-sha256"):
            command.add_argument(f"--{name}", required=True)
        command.add_argument("--resume", action="store_true")
    reload_parser = commands.add_parser("reload")
    reload_parser.add_argument("--run", type=Path, required=True)
    effect_pre = commands.add_parser("prepare-effect", help="CPU-only BF16 O-trial effect preregistration")
    effect_pre.add_argument("--campaign-plan", type=Path, required=True)
    effect_pre.add_argument("--out", type=Path, required=True)
    effect_pre.add_argument("--generation-batch-size", type=int, default=8)
    effect = commands.add_parser("evaluate-effect", help="BF16 measurements only, no updates")
    seal = commands.add_parser("seal-effect", help="Fresh BF16 reload and raw effect-gate validation")
    for command in (effect, seal):
        command.add_argument("--prereg", type=Path, required=True)
        command.add_argument("--prereg-sha256", required=True)
        command.add_argument("--out", type=Path, required=True)
    effect.add_argument("--stage", choices=("0", "8", "16"), required=True)
    effect.add_argument("--checkpoint", type=Path)
    seal.add_argument("--effect-root", type=Path, required=True)
    seal.add_argument("--campaign-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    if args.command == "reload":
        return reload_run(args)
    if args.command == "prepare-effect":
        return prepare_effect(args)
    if args.command == "evaluate-effect":
        return evaluate_effect(args)
    if args.command == "seal-effect":
        return seal_effect(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
