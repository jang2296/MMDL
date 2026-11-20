"""CPU preparation commands. Generation and audits do not authorize GPU training."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from .dataset import SPLITS, _bytes, _sha, _write, build_splits, freeze_selection, generate_pool, load_images
from .generators.common import GeneratedProblem
from .vault import _read, _safe_root, _unique, read_actors, read_lineage, read_verifier, write_split


def artifact_path(path: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    path = Path(path)
    if (path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("CLI: private artifacts must be outside the checkout without symlink aliases")
    _safe_root(path.parent, create=True)
    return path


def write_pool(path: Path, rows: list[GeneratedProblem], *, recipe: dict) -> dict:
    path = artifact_path(path)
    if not rows or len({r.lineage.split for r in rows}) != 1:
        raise ValueError("CLI: nonempty single-split pool required")
    path.mkdir(mode=0o700, exist_ok=False)
    write_split(path/"records", [r.actor for r in rows], [r.verifier for r in rows], [r.lineage for r in rows])
    written = set()
    for row in rows:
        for rel, raw in zip(row.actor.image_paths, row.image_bytes, strict=True):
            if rel not in written:
                _write(path/"images"/rel, raw)
                written.add(rel)
        for im in load_images(row.actor, path/"images"):
            im.close()
    result = {"format": "notebook-rl-candidate-pool-v1", "status": "PREPARED_PENDING_GATES",
        "training_allowed": False, "recipe": recipe, "count": len(rows), "split": rows[0].lineage.split,
        "records_manifest_sha256": _sha(_read(path/"records/manifest.yaml"))}
    _write(path/"pool.json", _bytes(result)+b"\n")
    return result


def read_pool(path: Path) -> list[GeneratedProblem]:
    path = _safe_root(path)
    info = json.loads(_read(path/"pool.json"))
    if (info.get("format") != "notebook-rl-candidate-pool-v1" or info.get("training_allowed") is not False
            or info["records_manifest_sha256"] != _sha(_read(path/"records/manifest.yaml"))):
        raise ValueError("CLI: mismatched candidate pool identity")
    actors = read_actors(path/"records")
    verifiers, lineage = read_verifier(path/"records"), read_lineage(path/"records")
    if len(actors) != info["count"] or any(lineage[a.problem_id].split != info["split"] for a in actors):
        raise ValueError("CLI: pool count/split mismatch")
    records = []
    for actor in actors:
        for image in load_images(actor, path/"images"):
            image.close()
        records.append(GeneratedProblem(actor, verifiers[actor.problem_id], lineage[actor.problem_id],
                                        tuple(_read(path/"images"/p) for p in actor.image_paths)))
    return records


def check_measurement_data(job_path: Path) -> dict:
    """Open a public path/hash job, never its private index or answer payloads."""
    from .approval import validate_for_measurement
    from .schema import to_json

    job = json.loads(_read(job_path), object_pairs_hook=_unique)
    keys = {"format", "build_root", "index", "source_registry", "evidence", "data_report", "expected_measurement_scope"}
    if (type(job) is not dict or set(job) != keys or job["format"] != "notebook-rl-measurement-data-job-v1"
            or not isinstance(job["build_root"], str) or not Path(job["build_root"]).is_absolute()):
        raise ValueError("CLI: exact measurement data job with an absolute build path required")
    for key in ("index", "source_registry", "evidence", "data_report"):
        value = job[key]
        if (type(value) is not dict or set(value) != {"path", "file_sha256"}
                or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
                or not isinstance(value["file_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", value["file_sha256"]) is None):
            raise ValueError("CLI: measurement inputs must use explicit path/SHA descriptors")
    result = validate_for_measurement(Path(job["build_root"]), **{k: job[k] for k in keys-{"format", "build_root"}})
    return result | {"actors": [json.loads(to_json(actor)) for actor in result["actors"]]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create one deterministic, UNAPPROVED private candidate pool")
    generate.add_argument("--generator", required=True, choices=("chart", "graph", "graph_pair", "circuit", "statics", "music", "chem"))
    generate.add_argument("--split", required=True, choices=SPLITS)
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--mode-mix", type=float, default=.5)
    generate.add_argument("--out", type=Path, required=True)
    build = commands.add_parser("build", help="Combine candidate pools; require all four disjoint splits")
    build.add_argument("--inputs", type=Path, nargs="+", required=True)
    build.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser("audit", help="Read-only scan with write-once receipt; partial evidence always BLOCKED")
    audit.add_argument("--build", type=Path, required=True)
    audit.add_argument("--index", type=Path, required=True)
    audit.add_argument("--source-registry", type=Path, default=Path(__file__).with_name("source_registry_v1.yaml"))
    audit.add_argument("--evidence", type=Path)
    audit.add_argument("--partial-name")
    audit.add_argument("--out", type=Path, required=True)
    select = commands.add_parser("select", help="Freeze completed Phase A counts, not permission to train")
    select.add_argument("--phase-a", type=Path, required=True)
    select.add_argument("--rules", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("check-measurement-data", help="Read-only D1–D8 validation; NOT GPU/training authority")
    check.add_argument("--job", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # Validate output before running expensive generation or index work.
    out = artifact_path(args.out)
    if out.exists():
        raise FileExistsError("CLI: preserve existing evidence; use a new output path")
    if args.command == "generate":
        recipe = {"generator": args.generator, "count": args.count, "seed": args.seed,
                  "split": args.split, "mode_mix": args.mode_mix}
        rows = generate_pool(args.generator, args.count, args.seed, args.split, args.mode_mix)
        result = write_pool(out, rows, recipe=recipe)
    elif args.command == "build":
        pools: dict[str, list[GeneratedProblem]] = {split: [] for split in SPLITS}
        for source in args.inputs:
            for row in read_pool(source):
                pools[row.lineage.split].append(row)
        result = build_splits(out, pools)
    elif args.command == "audit":
        from .contamination import load_index, scan, write_receipt
        index = load_index(args.index, partial_name=args.partial_name)
        registry = yaml.safe_load(_read(args.source_registry))
        evidence = None if args.evidence is None else json.loads(_read(args.evidence))
        result = scan({split: args.build/split for split in SPLITS}, args.build/"images", index, registry, evidence)
        write_receipt(out, result)
    elif args.command == "select":
        rules = yaml.safe_load(_read(args.rules))
        result = freeze_selection(args.phase_a, rules, output=out)
    else:
        result = check_measurement_data(args.job)
        _write(out, _bytes(result)+b"\n")
    # Never dump private questions, answers, benchmark matches or paths in console logs.
    print(json.dumps({"command": args.command, "status": result.get("status"),
                      "count": result.get("count"), "counts": result.get("counts"),
                      "training_authorized": False}, sort_keys=True))
    return 2 if result.get("status") == "BLOCKED_NOT_CHECKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
