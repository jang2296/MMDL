"""Restore the historical slot-206 export on CPU, requiring every original file hash."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

from mmdl.runtime.artifacts import sha256_file, write_json
from mmdl.runtime.model_reference import read_reference, require_hash, resolve_export_reference


def restore(*, checkpoint: Path, base: Path, reference: Path, export_manifest: Path,
            output: Path, receipt: Path, selector: Path) -> dict:
    """Use the original checkpoint loader/exporter; optimizer state remains hash-verified."""
    started = time.monotonic()
    checkpoint, base, output, receipt = (Path(p).absolute() for p in (checkpoint, base, output, receipt))
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError("restore: output and receipt must be new")
    sources = (checkpoint, base, Path(reference).absolute(), Path(export_manifest).absolute())
    if (any(output.is_relative_to(p) or p.is_relative_to(output) for p in sources)
            or receipt.is_relative_to(output)
            or output.is_relative_to(Path(__file__).resolve().parents[2])):
        raise ValueError("restore: output/receipt must be outside source directories and repository")
    anchor = read_reference(selector)
    require_hash(reference, anchor["reference_sha256"])
    require_hash(export_manifest, anchor["export_manifest_sha256"])
    ref = read_reference(reference)
    original = read_reference(export_manifest)
    for name, expected in anchor["checkpoint_files"].items():
        require_hash(checkpoint / name, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = sum(row["bytes"] for row in ref["files"])
    if shutil.disk_usage(output.parent).free < 50 * 1024**3 + expected_bytes:
        raise ValueError("restore: output requires an additional 50 GiB free disk reserve")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("restore: run with CUDA_VISIBLE_DEVICES='' (CPU only)")
    import torch
    from fine_tuning.notebook_rl.checkpoint import read_metadata
    from fine_tuning.notebook_rl.export import export_merged

    if torch.cuda.is_initialized():
        raise ValueError("restore: CUDA was initialized")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    metadata = read_metadata(checkpoint, binding=original["binding"])
    if (metadata["step"] != anchor["slot"] or metadata["cursor"]["single"]["phase"] != "final"
            or metadata["state_sha256"] != original["checkpoint_state_sha256"]
            or metadata["binding"] != ref["export"]["binding"]):
        raise ValueError("restore: final slot206 checkpoint binding/state differs")
    output.mkdir(exist_ok=False)
    result: dict = {"status": "FAILED", "cpu_only": True, "output": str(output),
                    "reference_sha256": sha256_file(reference),
                    "versions": {name: importlib.metadata.version(name)
                                 for name in ("torch", "transformers", "safetensors", "peft")}}
    try:
        result["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
        generated = export_merged(checkpoint, base, output / "model", original["arm"],
                                  binding=original["binding"], llm_lora=True)
        if generated != original:
            raise ValueError("restore: regenerated export manifest differs from historical manifest")
        # checkpoint_files inherits filesystem iteration order; preserve the verified historical JSON order.
        shutil.copyfile(export_manifest, output / "model" / "export_manifest.json")
        require_hash(output / "model" / "export_manifest.json", anchor["export_manifest_sha256"])
        shutil.copyfile(reference, output / "model_ref.json")
        resolve_export_reference(selector, output)
        result["files"] = [{"path": row["path"], "sha256": sha256_file(output / "model" / row["path"]),
                            "expected_sha256": row["sha256"], "bytes": row["bytes"]}
                           for row in ref["files"]]
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started,
                      maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      cuda_initialized=torch.cuda.is_initialized(),
                      disk_free_bytes=shutil.disk_usage(output.parent).free)
        write_json(receipt, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "base", "reference", "export-manifest", "output", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--selector", type=Path,
                        default=Path(__file__).resolve().parents[2] / "manifests/models/v18b.json")
    args = vars(parser.parse_args())
    result = restore(**args)
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "maximum_rss_kib")}))


if __name__ == "__main__":
    main()
