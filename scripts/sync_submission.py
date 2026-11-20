"""Sync the reviewed publication allowlist into the fixed MMDL/ folder."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.check_submission import inspect_blob
from scripts.submission_publication import publication_files


_TOP_FILES = (".gitignore", ".env.example", "pyproject.toml", "README.md")
_FIXED_FILES = (
    "env/requirements-eval.lock", "env/requirements-vllm.lock",
    "reports/mmmu_baseline.md", "assignment/assignment1.md",
    "results/scoring-v8-20260924/summary.json",
    "results/scoring-v8-20260924/subject_scores.csv",
    "results/scoring-v8-20260924/provenance.json",
)
_DEV_ONLY = {"tests/unit/test_sync_submission.py", "tests/unit/test_sync_test_repo.py"}
_REQUIRED_FILES = set(_TOP_FILES + _FIXED_FILES) | {
    "configs/eval/mmmu_val_v8.yaml",
    "src/mmdl/evaluation/final_answer_parser_v8.py",
    "third_party/mmmu/eval_utils.py",
    "third_party/mmmu/LICENSE",
}
_DIRS = (
    ("src", lambda p: p.suffix == ".py"),
    ("tests/unit", lambda p: p.suffix == ".py"),
    ("tests/unit/fixtures", lambda p: True),
    ("configs/eval", lambda p: p.suffix == ".yaml" or p.name == "v18b_reference_00fffac.json"),
    ("configs/hardware", lambda p: p.suffix == ".yaml"),
    ("prompts", lambda p: p.suffix == ".txt"),
    ("manifests/models", lambda p: p.suffix == ".json"),
    ("manifests/datasets", lambda p: p.suffix == ".json"),
    ("reports", lambda p: p.name == "finetune_v18b.md"),
    ("src/mmdl/evaluation/CompassVerifier_checker", lambda p: p.suffix in {".py", ".yaml"} or p.name == "README.md"),
    ("scripts", lambda p: p.name in {
        "doctor.sh", "eval.sh", "rescore.sh", "reproduce.sh", "test_reference.sh",
        "validate_run.py", "validate_rescore.py", "report_baseline.py", "check_submission.py",
        "rehearse_submission.py",
    }),
    ("third_party", lambda p: p.name in {"README.md", "LICENSE", "eval_utils.py", "run_mmmu.py", "prompt.py"}),
)


def _selected_files(root: Path, *, include_index: bool = True) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for relative in _TOP_FILES + _FIXED_FILES:
        path = root / relative
        if path.is_file():
            selected[relative] = path
    for directory, accept in _DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if "__pycache__" in path.parts or not path.is_file() or not accept(path):
                continue
            relative = path.relative_to(root).as_posix()
            if relative in _DEV_ONLY:
                continue
            selected[relative] = path
    # Submission prose and historical records can have a reviewed public source.
    for relative in selected:
        source = root / "submission" / relative
        if source.is_file():
            selected[relative] = source
    if include_index:
        indexed = publication_files(root)
        if indexed and selected.keys() - indexed.keys():
            raise ValueError("Publication index omits selected files: " + ", ".join(sorted(selected.keys() - indexed.keys())))
        if indexed and any(source != indexed[name] for name, source in selected.items()):
            raise ValueError("Publication index source mapping changed; regenerate it")
        selected.update(indexed)
    return dict(sorted(selected.items()))


def _destination(root: Path) -> Path:
    root = root.resolve()
    destination = root / "MMDL"
    if destination.is_symlink() or destination.resolve() != destination:
        raise ValueError("MMDL destination must be the real ROOT/MMDL directory")
    if destination.parent != root or destination.name != "MMDL":
        raise ValueError("Invalid submission destination")
    return destination


def _destination_files(destination: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not destination.exists():
        return found
    git_dir = destination / ".git"
    if git_dir.is_symlink() or (git_dir.exists() and not git_dir.is_dir()):
        raise ValueError("MMDL/.git must be a real directory (created by git clone)")
    for path in destination.rglob("*"):
        if ".git" in path.relative_to(destination).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Destination contains symlink: {path.relative_to(destination)}")
        if path.is_file():
            found[path.relative_to(destination).as_posix()] = path
        elif not path.is_dir():
            raise ValueError(f"Destination contains a non-regular file: {path.relative_to(destination)}")
    return found


def _fingerprint(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), stat.S_IMODE(path.stat().st_mode)


def _backup_directory(root: Path) -> Path:
    base = root / "analysis_exports" / "submission_backups"
    if any(path.is_symlink() for path in (root / "analysis_exports", base)):
        raise ValueError("Submission backup path must not contain symlinks")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def sync(root: Path, *, write: bool) -> list[str]:
    root = root.resolve()
    destination = _destination(root)
    selected = _selected_files(root)
    missing = sorted(_REQUIRED_FILES - selected.keys())
    if missing:
        raise ValueError("Required publication files are missing: " + ", ".join(missing))
    for relative, source in selected.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Selected source is not a regular file: {relative}")
        resolved = source.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Selected source escapes repository root: {relative}")
        ancestor = root
        for part in source.relative_to(root).parts:
            ancestor = ancestor / part
            if ancestor.is_symlink():
                raise ValueError(f"Selected source has a symlink path component: {relative}")
        errors = inspect_blob(relative, source.read_bytes())
        if errors:
            raise ValueError("; ".join(errors))

    existing = _destination_files(destination)
    expected = {name: _fingerprint(path) for name, path in selected.items()}
    actual = {name: _fingerprint(path) for name, path in existing.items()}
    expected_dirs = {parent.as_posix() for name in expected
                     for parent in Path(name).parents if parent.as_posix() != "."}
    actual_dirs = {path.relative_to(destination).as_posix() for path in destination.rglob("*")
                   if path.is_dir() and ".git" not in path.relative_to(destination).parts
                   }
    problems = []
    for name in sorted(expected.keys() | actual.keys()):
        if name not in expected:
            problems.append(f"extra: {name}")
        elif name not in actual:
            problems.append(f"missing: {name}")
        elif expected[name] != actual[name]:
            problems.append(f"different: {name}")
    problems.extend(f"extra directory: {name}" for name in sorted(actual_dirs - expected_dirs))
    if not write or not problems:
        return problems

    backup = _backup_directory(root)
    to_preserve = [name for name in actual if name not in expected or
                   (name in expected and actual[name] != expected[name])]
    for name in sorted(to_preserve):
        old = existing[name]
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(old, target)

    for directory in sorted((p for p in destination.rglob("*") if p.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        if ".git" not in directory.relative_to(destination).parts:
            try:
                directory.rmdir()
            except OSError:
                pass

    for name, source in selected.items():
        target = destination / name
        if target.exists() and target.is_dir():
            raise ValueError(f"Destination path is a directory: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report whether MMDL/ matches the allowlist")
    mode.add_argument("--sync", action="store_true", help="copy the allowlist, backing up displaced files")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        problems = sync(root, write=args.sync)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if problems:
        print("Submission tree differs:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    print("Submission tree matches allowlist" if args.check else "Submission tree synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
