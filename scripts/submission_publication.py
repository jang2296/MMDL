"""Build the explicit v18b publication index and losslessly split its large metadata."""
from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess

from fine_tuning.error_note.v18_recipe import PACKAGE_NAME, _document, _public, digest
from scripts.check_submission import inspect_blob

INDEX = "manifests/publication/v18b_sources.json"
UPSTREAM = ("6dbc649b5f23463ff8b2fb50e97ac516e856a004", "00fffac4f569dec3adbada1790af7b1af1b72c62")
SUBMISSION_BASE = "197838c11557911a14a46d87b7f52fda0cac5b3c"
REFERENCE_BACKEND = "src/mmdl/evaluation/backends/vllm_reference_00fffac.py"
ENTRYPOINTS = {
    "fine_tuning.error_note.submission_train": "public training CLI",
    "fine_tuning.error_note.plan": "verify-published-source; plan generation",
    "fine_tuning.error_note.v17_eval": "doctor-content-gate",
    "fine_tuning.error_note.v18b_execution": "seal-execution-settings",
    "fine_tuning.error_note.v18b_data": "prepare-data",
    "fine_tuning.error_note.guard_v16b": "guard-falsealarm-gate",
    "fine_tuning.error_note.prep_v16b": "prep-select; prep-generate; prep-rows",
    "fine_tuning.error_note.campaign_v16b": "campaign-prepare; campaign-advance; require-complete; reload; export",
    "fine_tuning.error_note.v17_artifacts": "training-artifact-seal",
    "fine_tuning.notebook_rl.pod.chain": "chain runner; separate-process reload",
    "fine_tuning.notebook_rl.pod.verify": "clean source verification",
    "fine_tuning.notebook_rl.pod.collect": "artifact collection",
    "fine_tuning.notebook_rl.proof_eval": "unmerged language and vision adapter attachment",
    "fine_tuning.error_note.v17_preflight": "training input and signal validation",
    "fine_tuning.error_note.restore_v18b": "checkpoint restoration",
    "fine_tuning.tests.test_error_note_single_vllm_lora_gpu": "synthetic tiny model builder for local rehearsal",
}
ASSETS = {
    "scripts/submission_publication.py": "reviewable publication index and recipe pack tooling",
    "scripts/sync_submission.py": "publication allowlist verification",
    "fine_tuning/error_note/source_baseline_v10.json": "protocol.py source baseline",
    "fine_tuning/error_note/v18_replay_receipt.json": "v18 training replay aggregate gate",
    "fine_tuning/error_note/v18b_reasoning_recipe.yaml": "additional 64 reasoning monitor items",
    "fine_tuning/notebook_rl/protocol_v1.yaml": "vault/schema protocol",
    "fine_tuning/notebook_rl/benchmark_inventory_v1.json": "exclusion metadata only",
    "fine_tuning/notebook_rl/requirements-data.lock": "separate CPU data environment",
    "fine_tuning/notebook_rl/requirements-local-nf4.txt": "local functional validation",
    "env/requirements-proof.lock": "recorded BF16 inference/training environment",
    "env/requirements-vllm.lock": "GPU environment included by proof lock",
    "scripts/finetune.sh": "public training entrypoint",
    "scripts/eval_finetuned.sh": "public trained-model evaluation entrypoint",
    "scripts/restore_v18b.sh": "public checkpoint restoration entrypoint",
    "configs/train/v18b.yaml": "public training configuration",
    "tests/unit/test_recipe_package.py": "synthetic package trust-boundary regression",
    "tests/test_restore_v18b.py": "synthetic restoration regression",
}
RESOURCES = {
    "baseline.json": "manifests/models/baseline.json",
    "requirements-vllm.lock": "env/requirements-vllm.lock",
    "mmmu_mcq_v1.txt": "prompts/mmmu_mcq_v1.txt",
    "mmmu_open_v1.txt": "prompts/mmmu_open_v1.txt",
}


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def pack_recipe(root: Path) -> dict[str, str]:
    folder = root / "fine_tuning/error_note"
    recipe = _document(folder / "v18_recipe.json")
    sidecars = _document(folder / recipe["sidecars"]["path"])
    names = ["v18_recipe.json", recipe["sidecars"]["path"],
             *(r["path"] for r in recipe["item_shards"]), *(r["path"] for r in sidecars["item_shards"])]
    records, public = [], {}
    for name in names:
        if Path(name).name != name:
            raise ValueError("Recipe publication path is not a basename")
        path = folder / name
        data, document = path.read_bytes(), _document(path)
        ref = {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if len(data) > 2 * 1024**2:
            if not name.startswith("v18_recipe_metadata_") or encoded(document) != data:
                raise ValueError("Only canonical metadata documents may be split")
            middle = len(document["items"]) // 2
            parts = []
            for i, rows in enumerate((document["items"][:middle], document["items"][middle:])):
                part_name = path.stem + f"_part_{i}.json"
                part_document = {"version": document["version"], "items": rows}
                part_document["content_sha256"] = digest(part_document)
                part_data = encoded(part_document)
                _public(json.loads(part_data))
                errors = inspect_blob(part_name, part_data)
                if errors:
                    raise ValueError("; ".join(errors))
                (folder / part_name).write_bytes(part_data)
                parts.append({"path": part_name, "bytes": len(part_data),
                              "sha256": hashlib.sha256(part_data).hexdigest(), "count": len(rows)})
                public[f"fine_tuning/error_note/{part_name}"] = f"lossless split of {name}; part {i}"
            ref |= {"document_version": document["version"],
                    "document_content_sha256": document["content_sha256"], "parts": parts}
        else:
            public[f"fine_tuning/error_note/{name}"] = "fixed public source reconstruction recipe"
        records.append(ref)
    package = {"version": "v18-recipe-package-v1", "recipe": "v18_recipe.json", "files": records}
    package["content_sha256"] = digest(package)
    (folder / PACKAGE_NAME).write_bytes(encoded(package))
    public[f"fine_tuning/error_note/{PACKAGE_NAME}"] = "lossless recipe packing manifest"
    return public


@lru_cache(maxsize=None)
def module_path(root: Path, module: str) -> Path | None:
    path = root.joinpath(*module.split("."))
    return next((p for p in (path.with_suffix(".py"), path / "__init__.py") if p.is_file()), None)


def import_closure(root: Path, entrypoints: dict[str, str]) -> dict[str, set[str]]:
    """Include local imports, including imports inside functions, without executing them."""
    module_path.cache_clear()
    found: dict[str, set[str]] = {}
    pending = list(entrypoints.items())
    while pending:
        module, caller = pending.pop()
        path = module_path(root, module)
        if path is None:
            raise ValueError(f"Missing publication entrypoint: {module}")
        relative = path.relative_to(root).as_posix()
        if relative in found:
            found[relative].add(caller)
            continue
        found[relative] = {caller}
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                stem = (importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                        if node.level else (node.module or ""))
                names = [stem, *(stem + "." + a.name for a in node.names)]
            for name in names:
                if name.startswith("fine_tuning.") and module_path(root, name) is not None:
                    pending.append((name, relative))
        for parent in path.relative_to(root).parents:
            if parent != Path(".") and (root / parent / "__init__.py").is_file():
                pending.append((".".join(parent.parts), relative))
    return found


def _originals(root: Path, names: list[str]) -> dict[str, dict]:
    queries = [(revision, name) for revision in UPSTREAM for name in names]
    result = subprocess.run(["git", "cat-file", "--batch"], cwd=root / "MMDL_test",
        input="".join(f"{revision}:{name}\n" for revision, name in queries).encode(),
        check=True, capture_output=True).stdout
    offset = 0
    originals: dict[str, dict] = {}
    for revision, name in queries:
        end = result.index(b"\n", offset)
        header = result[offset:end].split()
        offset = end + 1
        if header[-1] == b"missing":
            continue
        size = int(header[-1])
        data = result[offset:offset + size]
        offset += size + 1
        sha = hashlib.sha256(data).hexdigest()
        original = originals.setdefault(name, {"repository": "jang2296/MMDL_test", "revision": revision,
            "path": name, "sha256": sha, "upstream_sha256": {}})
        original["upstream_sha256"][revision] = sha
    return originals


def build_index(root: Path) -> dict:
    from scripts.sync_submission import _selected_files

    selected = import_closure(root, ENTRYPOINTS)
    sources = _selected_files(root, include_index=False)
    for name in sources:
        selected.setdefault(name, set()).add("public submission evaluation, configuration or regression")
    assets = dict(ASSETS) | pack_recipe(root)
    for name, canonical in RESOURCES.items():
        relative = "fine_tuning/notebook_rl/resources/" + name
        if (root / relative).read_bytes() != (root / canonical).read_bytes():
            raise ValueError(f"Pinned bundled resource differs: {relative}")
        assets[relative] = f"resource_path({canonical}) fallback; byte-identical canonical resource"
        assets[canonical] = f"resource_path({canonical}) canonical resource"
    for name, caller in assets.items():
        if not (root / name).is_file():
            raise ValueError(f"Missing publication asset: {name}")
        selected.setdefault(name, set()).add(caller)
    part_sources = {name: name.rsplit("_part_", 1)[0] + ".json"
                    for name in selected if "v18_recipe_metadata_" in name and "_part_" in name}
    originals = _originals(root, sorted(set(selected) | set(part_sources.values())))
    records = []
    failures = []
    for name, callers in sorted(selected.items()):
        path = sources.get(name, root / name)
        data = path.read_bytes()
        errors = inspect_blob(name, data)
        failures.extend(errors)
        original = originals.get(part_sources.get(name, name), {"repository": "jang2296/MMDL", "revision": None,
            "path": name, "sha256": hashlib.sha256(data).hexdigest(), "status": "new public source; publication commit pending"})
        if name == REFERENCE_BACKEND:
            original = dict(originals["src/mmdl/evaluation/backends/vllm_backend.py"])
            original.update(revision=UPSTREAM[1], sha256=original["upstream_sha256"][UPSTREAM[1]])
            if original["sha256"] != hashlib.sha256(data).hexdigest():
                raise ValueError("Pinned reference backend is not byte-identical to 00fffac")
        if path.relative_to(root).parts[0] == "submission":
            previous = subprocess.check_output(["git", "show", f"{SUBMISSION_BASE}:{name}"], cwd=root / "MMDL")
            original = {"repository": "jang2296/MMDL", "revision": SUBMISSION_BASE,
                "path": name, "sha256": hashlib.sha256(previous).hexdigest()}
        records.append({"path": name, "source_path": path.relative_to(root).as_posix(),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "mode": stat.S_IMODE(path.stat().st_mode),
            "original": original,
            **({"transformation": "original items array split in order; no content edits"} if name in part_sources else {}),
            "used_by": sorted(callers)})
    if failures:
        raise ValueError("; ".join(failures))
    result = {"version": "v18b-submission-files-v1", "upstream_revisions": list(UPSTREAM),
              "files": records}
    result["content_sha256"] = digest(result)
    return result


def publication_files(root: Path) -> dict[str, Path]:
    index_path = root / INDEX
    if not index_path.is_file():
        if (root / "fine_tuning/error_note/submission_train.py").is_file():
            raise ValueError("Build the v18b publication index before synchronizing")
        return {}
    index = json.loads(index_path.read_text())
    if (index.get("version") != "v18b-submission-files-v1" or index.get("content_sha256") !=
            digest({k: v for k, v in index.items() if k != "content_sha256"})):
        raise ValueError("Publication index content changed")
    selected = {INDEX: index_path}
    for record in index["files"]:
        name = record["path"]
        path = root / record.get("source_path", name)
        if Path(name).is_absolute() or ".." in Path(name).parts or name in selected:
            raise ValueError("Unsafe or repeated publication index path")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Publication source escapes workspace")
        data = path.read_bytes()
        if (len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]
                or record.get("mode", stat.S_IMODE(path.stat().st_mode)) != stat.S_IMODE(path.stat().st_mode)):
            raise ValueError(f"Publication index is stale: {name}")
        selected[name] = path
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-index", action="store_true", required=True)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_index(root)
    target = root / INDEX
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded(result))
    print(json.dumps({"index": INDEX, "files": len(result["files"]), "content_sha256": result["content_sha256"]}))


if __name__ == "__main__":
    main()
