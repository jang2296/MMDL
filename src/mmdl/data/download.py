"""Acquire only pinned model files, MMMU validation, and archival MMMU-Pro 10-way."""

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from mmdl.runtime.artifacts import default_path, read_json, sha256_file, write_json
from mmdl.runtime.contracts import (
    DATA_ID, DATA_REVISION, MODEL_ID, MODEL_REVISION, PRO_ID, PRO_REVISION, SUBJECTS,
)
from mmdl.runtime.environment import check_storage

MODEL_FILES = ["README.md", "config.json", "generation_config.json", "chat_template.json",
               "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json",
               "preprocessor_config.json", "video_preprocessor_config.json",
               "model.safetensors.index.json", "model-00001-of-00002.safetensors",
               "model-00002-of-00002.safetensors"]


def acquire(repo, revision, kind, filenames, cache, destination=None):
    info = HfApi().repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hub returned an unexpected revision")
    metadata = {item.rfilename: item for item in info.siblings}
    total = sum(metadata[name].size for name in filenames)
    # Conservative bound includes downloads, Arrow conversion, and partial download files.
    storage = check_storage([cache] + ([destination] if destination else []), 3 * total)
    if not storage["sufficient"]:
        raise RuntimeError(f"Insufficient storage including configured reserve: {storage}")
    start = time.monotonic()
    records = []
    snapshot = None
    for filename in filenames:
        path = Path(hf_hub_download(repo, filename, repo_type=kind, revision=revision,
                                    cache_dir=cache / "hub"))
        relative_parts = len(Path(filename).parts)
        snapshot = path.parents[relative_parts - 1]
        checksum = sha256_file(path)
        upstream = metadata[filename]
        if path.stat().st_size != upstream.size:
            raise ValueError(f"Size mismatch: {repo}/{filename}")
        if upstream.lfs and checksum != upstream.lfs.sha256:
            raise ValueError(f"LFS hash mismatch: {repo}/{filename}")
        records.append(dict(path=filename, bytes=path.stat().st_size, sha256=checksum))
        if destination:
            link = destination / filename
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                if sha256_file(link) != checksum:
                    raise ValueError(f"Refusing to replace existing file: {filename}")
            else:
                link.symlink_to(os.path.relpath(path, link.parent))
        print(f"verified {repo} {filename}", flush=True)
    return dict(repo_id=repo, revision=revision, kind=kind, files=records,
                download_verify_seconds=time.monotonic() - start), snapshot


def prepare(root, data_root, cache, artifact_root, include_pro=True):
    cache.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    specs = [(MODEL_ID, MODEL_REVISION, "model", MODEL_FILES, None, "models/baseline.json")]
    specs.append((DATA_ID, DATA_REVISION, "dataset",
                  ["README.md"] + [f"{s}/validation-00000-of-00001.parquet" for s in SUBJECTS],
                  data_root / "evaluation/mmmu", "datasets/mmmu.json"))
    if include_pro:
        specs.append((PRO_ID, PRO_REVISION, "dataset", ["README.md"] +
                      [f"standard (10 options)/test-0000{i}-of-00002.parquet" for i in range(2)],
                      data_root / "evaluation/mmmu_pro", "datasets/mmmu_pro.json"))
    locations = {}
    for repo, revision, kind, files, destination, manifest in specs:
        result, snapshot = acquire(repo, revision, kind, files, cache, destination)
        result["usage"] = "archive_only_no_inference" if repo == PRO_ID else "evaluation_only"
        result["storage"] = ("HF_HOME/hub/" + str(snapshot.relative_to(cache / "hub"))
                             if not destination else "MMDL_DATA_ROOT/" + str(destination.relative_to(data_root)))
        result["source"] = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}/tree/{revision}"
        if kind == "model":
            result["kind"] = "base"
            result["dtype"] = "bfloat16"
            result["processor_revision"] = revision
        target = root / "manifests" / manifest
        if target.exists():
            old = read_json(target)
            if old["revision"] != revision or old["files"] != result["files"]:
                raise ValueError("Existing manifest differs; refusing overwrite")
        else:
            write_json(target, result)
        if destination:
            existing = destination / "manifest.json"
            if existing.exists() and read_json(existing)["files"] != result["files"]:
                raise ValueError("Existing local dataset manifest differs")
            if not existing.exists():
                write_json(existing, result)
        locations[repo] = dict(snapshot=str(snapshot), data_root=str(destination) if destination else None)
    write_json(artifact_root / "setup/locations.json", locations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_path("MMDL_DATA_ROOT", "mmdl-data"))
    parser.add_argument("--cache", type=Path, default=default_path("HF_HOME", "mmdl-cache/huggingface"))
    parser.add_argument("--artifact-root", type=Path, default=default_path("MMDL_ARTIFACT_ROOT", "mmdl-artifacts"))
    parser.add_argument("--skip-pro", action="store_true", help="Skip archival acquisition only")
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[3], args.data_root, args.cache, args.artifact_root,
            include_pro=not args.skip_pro)


if __name__ == "__main__":
    main()
