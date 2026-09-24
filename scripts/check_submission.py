"""Reject private, benchmark, weight, cache, and oversized public files."""

import argparse
import re
import subprocess
from pathlib import Path

def inspect_blob(name, data, symlink=False):
    path = Path(name)
    errors = []
    if symlink:
        errors.append(f"Public symlink is not allowed: {name}")
    if any(part in {"local_sources", ".venv", "venv", "checkpoints", "artifacts", ".cache", ".serena", ".ssh",
                   "docs", "claudedocs", "analysis_exports"} for part in path.parts):
        errors.append(f"Private directory: {name}")
    if path.name == "AGENTS.md":
        errors.append(f"Internal agent instructions: {name}")
    if path.name == ".env" or path.suffix.lower() in {".safetensors", ".parquet", ".arrow", ".pt", ".pth", ".bin", ".pdf", ".pem", ".key"}:
        errors.append(f"Private artifact type: {name}")
    if len(data) > 2 * 1024**2:
        errors.append(f"File exceeds 2 MiB: {name}")
        return errors
    text = data.decode(errors="replace")
    if re.search(r"(?:hf_|sk-|gh[pousr]_|github_pat_|rpa_|rps_)[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|/home/[a-zA-Z0-9_.-]+/|[A-Z]:\\Users\\[a-zA-Z0-9_.-]+", text):
        errors.append(f"Possible secret or personal absolute path: {name}")
    if name.startswith("results/") and any(f'"{key}"' in text for key in ("question", "raw_response", "gold_explanation")):
        errors.append(f"Benchmark text in public result: {name}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect exact index bytes selected for publication")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if args.staged
               else ["git", "ls-files", "-co", "--exclude-standard", "-z"])
    paths = subprocess.check_output(command, cwd=root).split(b"\0")
    errors = []
    for value in sorted(set(paths)):
        if not value:
            continue
        name = value.decode()
        if args.staged:
            size = int(subprocess.check_output(["git", "cat-file", "-s", f":{name}"], cwd=root))
            mode = subprocess.check_output(["git", "ls-files", "-s", "--", name], cwd=root).split()[0]
            symlink = mode == b"120000"
            data = subprocess.check_output(["git", "show", f":{name}"], cwd=root) if size <= 2 * 1024**2 else b""
        else:
            path = root / name
            if not path.exists():
                errors.append(f"Missing public file: {name}")
                continue
            symlink = path.is_symlink()
            size = path.lstat().st_size
            data = path.read_bytes() if not symlink and size <= 2 * 1024**2 else b""
        if size > 2 * 1024**2:
            errors.append(f"File exceeds 2 MiB: {name}")
        errors.extend(inspect_blob(name, data, symlink))
    if errors:
        raise SystemExit("\n".join(errors))
    print("Public-file checks passed")


if __name__ == "__main__":
    main()
