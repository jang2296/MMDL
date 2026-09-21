"""Bounded host diagnostics for reproducible evaluation runs.

This module never loads a model or downloads data.  It reports only public
machine metadata; optional private output is for local artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


GIB = 1024**3
RESERVE_BYTES = 50 * GIB
PACKAGES = ("torch", "transformers", "accelerate", "datasets", "psutil")


def _run(command: list[str]) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, errors="replace", timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode:
        return None, (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
    return completed.stdout, None


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
    except OSError:
        return bool(os.environ.get("WSL_DISTRO_NAME"))


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _nvidia_smi() -> dict[str, Any]:
    summary, summary_error = _run(["nvidia-smi"])
    output, error = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if error:
        return {"available": False, "diagnostic": error or summary_error}
    cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", summary or "")
    gpus = []
    for line in (output or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            gpus.append(
                {
                    "name": fields[0],
                    "driver": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_free_mib": int(fields[3]),
                    "compute_capability": fields[4],
                }
            )
    return {"available": True, "cuda": cuda_match.group(1) if cuda_match else None, "gpus": gpus}


def _torch_details(probe: bool) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "cuda_available": False, "diagnostic": str(exc)}

    result: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_architectures": list(torch.cuda.get_arch_list()),
    }
    if not result["cuda_available"]:
        result["bf16_probe"] = {"passed": False, "diagnostic": "CUDA is unavailable"}
        return result

    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        capability = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": f"{capability[0]}.{capability[1]}",
                "memory_free_bytes": free,
                "memory_total_bytes": total,
            }
        )
    result["gpus"] = devices
    result["bf16_probe"] = _bf16_probe(torch) if probe else {"passed": None, "skipped": True}
    result["peak_cuda_memory"] = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return result


def _bf16_probe(torch: Any) -> dict[str, Any]:
    try:
        torch.cuda.reset_peak_memory_stats()
        # Deliberately create floating tensors; integer inputs must never be cast to BF16.
        left = torch.randn((32, 32), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((32, 32), device="cuda", dtype=torch.bfloat16)
        result = left @ right
        torch.cuda.synchronize()
        if result.dtype != torch.bfloat16 or not torch.isfinite(result).all().item():
            return {"passed": False, "diagnostic": "BF16 matmul returned an invalid result"}
        return {"passed": True, "shape": [32, 32]}
    except Exception as exc:  # Hardware/library failures are diagnostic output, not hidden.
        return {"passed": False, "diagnostic": f"{type(exc).__name__}: {exc}"}


def _ram() -> dict[str, int] | None:
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None
    memory = psutil.virtual_memory()
    return {"total_bytes": memory.total, "available_bytes": memory.available}


def _process_peak_rss_bytes() -> int | None:
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    return peak if sys.platform == "darwin" else peak * 1024


def collect_environment(probe: bool = True, private_artifact: Path | None = None) -> dict[str, Any]:
    """Collect sanitized public diagnostics without model loading or path disclosure."""
    wsl = _is_wsl()
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "wsl": {"detected": wsl, "distro": os.environ.get("WSL_DISTRO_NAME") if wsl else None},
        "cpu_count": os.cpu_count(),
        "ram": _ram(),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "packages": _packages(),
        "nvidia_smi": _nvidia_smi(),
        "torch": _torch_details(probe),
    }
    if private_artifact is not None:
        _atomic_json(private_artifact, {"cwd": str(Path.cwd()), "environment": result})
        result["private_artifact_written"] = True
    return result


def _windows_backing_volume() -> dict[str, Any]:
    """Find this distro's ext4 VHDX then its Windows volume; exposes no host path."""
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return {"checked": False, "diagnostic": "WSL_DISTRO_NAME is not set"}
    safe_distro = distro.replace("'", "''")
    script = (
        f"$ErrorActionPreference='Stop'; $d='{safe_distro}'; "
        "$root='HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss'; "
        "$key=Get-ChildItem $root | Where-Object { $_.GetValue('DistributionName') -eq $d } | Select-Object -First 1; "
        "if ($null -eq $key) { throw 'WSL distro registry entry not found' }; "
        "$base=$key.GetValue('BasePath'); $vhd=Join-Path $base 'ext4.vhdx'; "
        "if (!(Test-Path -LiteralPath $vhd)) { throw 'ext4.vhdx not found' }; "
        "$drive=(Get-Item -LiteralPath $vhd).PSDrive.Name; $vol=Get-PSDrive -Name $drive; "
        "[pscustomobject]@{drive=$drive;free=[int64]$vol.Free;used=[int64]$vol.Used}|ConvertTo-Json -Compress"
    )
    output, error = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
    if error:
        return {"checked": False, "diagnostic": error}
    try:
        data = json.loads(output or "{}")
        return {"checked": True, "volume": data["drive"], "free_bytes": int(data["free"])}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"checked": False, "diagnostic": f"invalid PowerShell response: {exc}"}


def check_storage(paths: list[Path], required_bytes: int = 0) -> dict[str, Any]:
    """Record explicit server headroom; local WSL always retains at least 50 GiB."""
    if required_bytes < 0:
        raise ValueError("required_bytes must be non-negative")
    reserve_gib = float(os.environ.get("MMDL_STORAGE_RESERVE_GIB", "50"))
    if not math.isfinite(reserve_gib) or reserve_gib < 1:
        raise ValueError("Storage reserve must be finite and at least 1 GiB")
    reserve = int(reserve_gib * GIB)
    if _is_wsl():
        reserve = max(reserve, RESERVE_BYTES)
    required = required_bytes + reserve
    filesystems: list[dict[str, Any]] = []
    for path in paths:
        try:
            target = path
            while not target.exists() and target != target.parent:
                target = target.parent
            usage = shutil.disk_usage(target)
            filesystems.append({"free_bytes": usage.free, "required_bytes": required, "sufficient": usage.free >= required})
        except OSError as exc:
            filesystems.append({"free_bytes": None, "required_bytes": required, "sufficient": False, "diagnostic": str(exc)})
    backing = _windows_backing_volume() if _is_wsl() else {"checked": False, "not_applicable": True}
    backing_ok = backing.get("not_applicable") or (backing.get("checked") and backing.get("free_bytes", 0) >= required)
    return {
        "reserve_bytes": reserve,
        "required_bytes": required_bytes,
        "filesystems": filesystems,
        "windows_backing_volume": backing,
        "sufficient": bool(filesystems) and all(item["sufficient"] for item in filesystems) and bool(backing_ok),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect bounded MMDL runtime diagnostics")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--storage-path", action="append", default=[], type=Path)
    parser.add_argument("--required-gib", default=0, type=float)
    parser.add_argument("--private-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.required_gib < 0:
        parser.error("--required-gib must be non-negative")
    environment = collect_environment(private_artifact=args.private_artifact)
    storage = check_storage(args.storage_path, int(args.required_gib * GIB)) if args.storage_path else None
    report = {"environment": environment, "storage": storage}
    _atomic_json(args.output, report)
    cuda_ok = environment["torch"]["cuda_available"] and environment["torch"]["bf16_probe"]["passed"] is True
    storage_ok = storage is None or storage["sufficient"]
    return 0 if cuda_ok and storage_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
