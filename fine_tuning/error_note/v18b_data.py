"""v18 inputs plus a source-reconstructed, training-forbidden reasoner monitoring panel."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

from . import v18_data as base
from .v18_recipe import _document, digest, load_recipe

VERSION = "v18b-monitor-data-v1"
ROLES = (*base.ROLES, "reasoning_monitor")
DEFAULT_PANEL = Path(__file__).with_name("v18b_reasoning_recipe.yaml")
private_verifier = base.private_verifier


def panel_recipe(path: Path | None = None) -> dict:
    from collections import Counter
    from .v18b_monitor import REASONING_SOURCES

    recipe = _document(path or DEFAULT_PANEL)
    items = recipe["items"]
    if (recipe.get("version") != "v18b-reasoning-panel-v1" or len(items) != 64
            or Counter(x["source"] for x in items) != dict.fromkeys(REASONING_SOURCES, 16)
            or len({x["problem_id"] for x in items}) != 64
            or [x["role_index"] for x in items] != list(range(64))):
        raise ValueError("V18B_REASONING_PANEL_COUNTS")
    for item in items:
        if (item.get("training_allowed") is not False or item.get("benchmark_derived") is not False
                or item.get("role") != "reasoning_monitor" or ":dev:" not in item["problem_id"]
                or item["source_revision"] != recipe["sources"][item["source"]]["revision"]):
            raise ValueError("V18B_REASONING_PANEL_PERMISSION_OR_SOURCE")
    return recipe


def _panel_receipt(root: Path) -> dict:
    receipt = json.loads((root / "reasoning-monitor-receipt.json").read_text())
    if receipt.get("sha256") != digest({k: v for k, v in receipt.items() if k != "sha256"}):
        raise ValueError("V18B_REASONING_RECEIPT_CHANGED")
    if receipt.get("source_sha256") != base.file_sha(Path(__file__)):
        raise ValueError("V18B_REASONING_DATA_SOURCE_CHANGED")
    if receipt.get("recipe_sha256") != panel_recipe()["content_sha256"]:
        raise ValueError("V18B_REASONING_RECIPE_CHANGED")
    for relative, sha in receipt["files"].items():
        if not relative.startswith(("reasoning-monitor/", "vault/reasoning_monitor/")):
            raise ValueError("V18B_REASONING_FILE_SCOPE")
        path = base._safe_path(root, relative)
        if path.is_symlink() or base.file_sha(path) != sha:
            raise ValueError("V18B_REASONING_PAYLOAD_CHANGED")
    return receipt


def load_items(root: Path, split: str) -> list[dict]:
    if split != "reasoning_monitor":
        return base.load_items(root, split)
    _panel_receipt(root)
    rows = json.loads((root / "reasoning-monitor/manifest.json").read_text())["items"]
    result = []
    for row in rows:
        item = dict(row, **{k: row["verifier"][k] for k in ("gold", "answer_type", "accept")})
        item["image_paths"] = [str(base._safe_path(root, p)) for p in row["image_paths"]]
        result.append(item)
    return result


def _isolated(panel: list[dict], existing: list[dict]) -> None:
    from .v17_data import identity_components

    all_items = existing + panel
    ids = [x["problem_id"] for x in all_items]
    if len(ids) != len(set(ids)):
        raise ValueError("V18B_REASONING_DUPLICATE_ID")
    components = identity_components(all_items)
    reserved = {components[x["problem_id"]] for x in existing}
    panel_components = [components[x["problem_id"]] for x in panel]
    if reserved & set(panel_components) or len(set(panel_components)) != len(panel_components):
        raise ValueError("V18B_REASONING_IDENTITY_OVERLAP")


def materialize_reasoning(root: Path, cache_dir: Path, *, offline: bool = False,
                          workers: int = 4) -> dict:
    from PIL import Image
    from fine_tuning.notebook_rl.adapters.hf_common import to_problem
    from fine_tuning.notebook_rl.adapters.hf_registry import sources
    from fine_tuning.notebook_rl.contamination import image_fingerprints
    from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json
    from fine_tuning.notebook_rl.vault import write_split
    from .v18_data_convert import SourceReader, candidate

    if (root / "reasoning-monitor-receipt.json").exists():
        return _panel_receipt(root)
    if (root / "reasoning-monitor").exists() or (root / "vault/reasoning_monitor").exists():
        raise ValueError("V18B_REASONING_PARTIAL_OUTPUT_PRESERVED")
    recipe = panel_recipe()
    acquisition = base.fetch(recipe, cache_dir, offline=offline, workers=workers)
    stage = Path(tempfile.mkdtemp(prefix=".v18b-reasoning-", dir=root.parent))
    reader = SourceReader(recipe, cache_dir)
    entries = []
    try:
        reader.selected_rows(recipe["items"])
        registry = sources()
        for selected in recipe["items"]:
            made, _ = candidate(selected, reader)
            problem = to_problem(made, registry[selected["source"]][0], index=selected["ordinal"])
            records = {name: json.loads(json.dumps(asdict(getattr(problem, name)))) for name in ("actor", "verifier", "lineage")}
            if problem.actor.problem_id != selected["problem_id"] or any(
                    digest(value) != selected["expected"][name + "_sha256"] for name, value in records.items()):
                raise ValueError("V18B_REASONING_NATIVE_CONTENT_CHANGED")
            images, pixels = [], []
            for raw, expected in zip(problem.image_bytes, selected["images"], strict=True):
                if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                    raise ValueError("V18B_REASONING_IMAGE_BYTES_CHANGED")
                with Image.open(io.BytesIO(raw)) as image:
                    pixel = image_fingerprints(image)["pixels_sha256"]
                if pixel != expected["pixel_sha256"]:
                    raise ValueError("V18B_REASONING_IMAGE_PIXELS_CHANGED")
                relative = "reasoning-monitor/images/" + expected["path"]
                path = base._safe_path(stage, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                images.append(relative)
                pixels.append(pixel)
            actor, truth, lineage = (records[name] for name in ("actor", "verifier", "lineage"))
            entries.append({"problem_id": selected["problem_id"], "source": selected["source"],
                "source_revision": selected["source_revision"], "native_id": selected["native_id"],
                "family_id": selected["family_id"], "split": "reasoning_monitor", "actor": actor,
                "verifier": truth, "lineage": lineage, "question": actor["question"], "choices": actor["choices"],
                "image_paths": images, "image_sha256": actor["image_sha256"], "pixel_sha256": pixels,
                "training_allowed": False, "benchmark_derived": False, "human_verified": False,
                "contamination_receipt": selected["contamination"],
                "evaluation": {"gold": truth["gold"], "answer_type": truth["answer_type"], "accept": truth["accept"],
                               "question": actor["question"], "choices": actor["choices"]}})
        existing = [x for role in base.ROLES for x in base.load_items(root, role)]
        _isolated(entries, existing)
        base._write(stage / "reasoning-monitor/manifest.json", {"version": VERSION, "items": entries,
            "training_allowed": False, "public_upload_allowed": False, "selection": recipe["selection"]})
        write_split(stage / "vault/reasoning_monitor",
            [from_json(ActorItem, json.dumps(x["actor"])) for x in entries],
            [from_json(VerifierItem, json.dumps(x["verifier"])) for x in entries],
            [from_json(Lineage, json.dumps(x["lineage"])) for x in entries])
        files = {p.relative_to(stage).as_posix(): base.file_sha(p) for p in stage.rglob("*") if p.is_file()}
        (stage / "reasoning-monitor").rename(root / "reasoning-monitor")
        (stage / "vault/reasoning_monitor").rename(root / "vault/reasoning_monitor")
        receipt = {"version": VERSION, "recipe_sha256": recipe["content_sha256"], "files": files,
            "count": 64, "training_allowed": False, "base_recipe_sha256": load_recipe()["content_sha256"],
            "acquisition": acquisition, "isolation": "ID_NATIVE_FAMILY_BYTE_PIXEL_TRANSITIVE",
            "source_sha256": base.file_sha(Path(__file__))}
        receipt["sha256"] = digest(receipt)
        base._write(root / "reasoning-monitor-receipt.json", receipt)
        shutil.rmtree(stage)
        return receipt
    except BaseException:
        if stage.exists():
            stage.rename(stage.with_name(stage.name + ".failed"))
        raise
    finally:
        reader.close()


def data_files(root: Path) -> dict[str, str]:
    return base.data_files(root) | _panel_receipt(root)["files"] | {
        "reasoning-monitor-receipt.json": base.file_sha(root / "reasoning-monitor-receipt.json")}


def validate(root: Path, *, tokenizer=None, **kwargs) -> dict:
    from fine_tuning.notebook_rl.vault import read_actors, read_lineage, read_verifier
    recipe = panel_recipe()
    receipt = _panel_receipt(root)
    additional = receipt["files"] | {"reasoning-monitor-receipt.json": base.file_sha(root / "reasoning-monitor-receipt.json")}
    result = base.validate(root, tokenizer=tokenizer, additional_files=additional, **kwargs)
    panel = load_items(root, "reasoning_monitor")
    if [x["problem_id"] for x in panel] != [x["problem_id"] for x in recipe["items"]]:
        raise ValueError("V18B_REASONING_FIXED_ORDER_CHANGED")
    existing = [x for role in base.ROLES for x in base.load_items(root, role)]
    _isolated(panel, existing)
    actors = {x.problem_id: asdict(x) for x in read_actors(root / "vault/reasoning_monitor")}
    truths = {pid: asdict(x) for pid, x in read_verifier(root / "vault/reasoning_monitor").items()}
    lineages = {pid: asdict(x) for pid, x in read_lineage(root / "vault/reasoning_monitor").items()}
    expected_ids = {x["problem_id"] for x in panel}
    if set(actors) != expected_ids or set(truths) != expected_ids or set(lineages) != expected_ids:
        raise ValueError("V18B_REASONING_VAULT_IDS_CHANGED")
    for item, selected in zip(panel, recipe["items"], strict=True):
        if item.get("training_allowed") is not False or item.get("benchmark_derived"):
            raise ValueError("V18B_REASONING_TRAINING_PERMISSION")
        if any(item.get(field) != selected[field] for field in
               ("problem_id", "source", "source_revision", "native_id", "family_id")):
            raise ValueError("V18B_REASONING_SOURCE_BINDING_CHANGED")
        actor, truth = item["actor"], item["verifier"]
        expected_evaluation = {"gold": truth["gold"], "answer_type": truth["answer_type"], "accept": truth["accept"],
                               "question": actor["question"], "choices": actor["choices"]}
        if (item["question"] != actor["question"] or item["choices"] != actor["choices"]
                or item["evaluation"] != expected_evaluation or item["split"] != "reasoning_monitor"
                or item["image_sha256"] != actor["image_sha256"]
                or item["pixel_sha256"] != [x["pixel_sha256"] for x in selected["images"]]):
            raise ValueError("V18B_REASONING_ACTOR_OR_EVALUATION_CHANGED")
        for name in ("actor", "verifier", "lineage"):
            if digest(item[name]) != selected["expected"][name + "_sha256"]:
                raise ValueError("V18B_REASONING_FROZEN_CONTENT_CHANGED")
        if (digest(actors.get(item["problem_id"])) != digest(item["actor"])
                or digest(truths.get(item["problem_id"])) != digest(item["verifier"])
                or digest(lineages.get(item["problem_id"])) != digest(item["lineage"])):
            raise ValueError("V18B_REASONING_VAULT_CHANGED")
        if item["contamination_receipt"] != selected["contamination"] or item["family_id"] != selected["family_id"]:
            raise ValueError("V18B_REASONING_EXCLUSION_CHANGED")
        for path, expected in zip(item["image_paths"], selected["images"], strict=True):
            if base.file_sha(Path(path)) != expected["sha256"]:
                raise ValueError("V18B_REASONING_FIXED_IMAGE_CHANGED")
    return result | {"version": VERSION, "reasoning_monitor_count": len(panel),
                     "reasoning_monitor_training_allowed": False, "panel_recipe_sha256": recipe["content_sha256"]}


def prepare(recipe_path: Path, cache_dir: Path, output: Path, *, offline: bool = False,
            workers: int = 4, tokenizer=None) -> dict:
    if output.exists() and (output / "reasoning-monitor-receipt.json").exists():
        return validate(output, tokenizer=tokenizer)
    base.prepare(recipe_path, cache_dir, output, offline=offline, workers=workers, tokenizer=tokenizer)
    materialize_reasoning(output, cache_dir, offline=offline, workers=workers)
    return validate(output, tokenizer=tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.command == "prepare":
        from .v18_recipe import DEFAULT_RECIPE
        if args.cache_dir is None:
            parser.error("prepare requires --cache-dir")
        result = prepare(args.recipe or DEFAULT_RECIPE, args.cache_dir, args.output,
            offline=args.offline, workers=args.workers, tokenizer=tokenizer)
    else:
        result = validate(args.output, tokenizer=tokenizer)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
