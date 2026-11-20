"""Stage program-generator items for the training pool. Staging is not scanning or approval."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .dataset import SPLITS, _write, generate_pool
from .generators import get_generator
from .generators.common import GeneratedProblem
from .schema import ActorItem, Lineage, VerifierItem, from_json, to_json

STAGE_VERSION = "gen-stage-v1"
_SPEC_VERSION = "gen-stage-spec-v1"
_STATUS = "STAGED_UNSCANNED_NOT_APPROVED"
_MODES = ("obs", "plain")
_HASH16 = re.compile(r"[0-9a-f]{16}\Z")
_SPEC_KEYS = frozenset({"version", "generators", "exclude_structure_hashes"})
_GENERATOR_KEYS = frozenset({"subject", "seed", "mode", "counts"})
_RECORD_KEYS = frozenset({"generator", "subject", "split", "actor", "verifier", "lineage"})
_MANIFEST_KEYS = frozenset({"version", "spec_sha256", "counts", "records_sha256",
                            "image_sha256", "excluded", "status"})


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _validate_spec(spec: Any) -> None:
    if type(spec) is not dict or set(spec) != _SPEC_KEYS or spec["version"] != _SPEC_VERSION:
        raise ValueError("gen_stage: invalid spec envelope")
    if type(spec["generators"]) is not dict:
        raise ValueError("gen_stage: generators must be an object")
    for name, gen in spec["generators"].items():
        if type(gen) is not dict or set(gen) != _GENERATOR_KEYS:
            raise ValueError(f"gen_stage: invalid spec fields for {name!r}")
        counts = gen["counts"]
        if (type(gen["subject"]) is not str or not gen["subject"].strip()
                or type(gen["seed"]) is not int or gen["seed"] < 0 or gen["mode"] not in _MODES
                or type(counts) is not dict or not set(counts) <= set(SPLITS)
                or any(type(n) is not int or n < 0 for n in counts.values())):
            raise ValueError(f"gen_stage: invalid spec values for {name!r}")
        get_generator(name)  # raises ValueError for unknown generator names
    exclusions = spec["exclude_structure_hashes"]
    if type(exclusions) is not list or any(type(h) is not str or not _HASH16.fullmatch(h)
                                           for h in exclusions):
        raise ValueError("gen_stage: exclusion hashes must be 16 lowercase hex strings")


def stage(spec: dict, output: Path) -> dict:
    _validate_spec(spec)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("gen_stage: output directory must not exist")
    exclude = set(spec["exclude_structure_hashes"])
    lines: list[bytes] = []
    images: dict[str, bytes] = {}
    excluded: dict[str, int] = {}
    for name in sorted(spec["generators"]):
        gen = spec["generators"][name]
        excluded[name] = 0
        for split in SPLITS:
            n = gen["counts"].get(split, 0)
            if n == 0:
                continue
            mode_mix = 1.0 if gen["mode"] == "obs" else 0.0
            pool = generate_pool(name, n + len(exclude), gen["seed"], split, mode_mix)
            kept = [p for p in pool if p.lineage.structure_hash not in exclude]
            excluded[name] += len(pool) - len(kept)
            if len(kept) < n:
                raise ValueError(f"gen_stage: fewer than {n} unexcluded records for {name}/{split}")
            for problem in kept[:n]:
                record = {"generator": name, "subject": gen["subject"], "split": split,
                          "actor": to_json(problem.actor), "verifier": to_json(problem.verifier),
                          "lineage": to_json(problem.lineage)}
                lines.append(_canonical(record) + b"\n")
                for rel, raw, digest in zip(problem.actor.image_paths, problem.image_bytes,
                                            problem.actor.image_sha256, strict=True):
                    if _sha(raw) != digest or images.get(rel, raw) != raw:
                        raise ValueError("gen_stage: image bytes or path identity mismatch")
                    images[rel] = raw
    records = b"".join(lines)
    manifest = {
        "version": STAGE_VERSION,
        "spec_sha256": _sha(_canonical(spec)),
        "counts": {name: dict(gen["counts"]) for name, gen in spec["generators"].items()},
        "records_sha256": _sha(records),
        "image_sha256": {rel: _sha(raw) for rel, raw in images.items()},
        "excluded": excluded,
        "status": _STATUS,
    }
    output.mkdir(mode=0o700, parents=True)  # records carry verifier gold, as vault.py private files
    for rel, raw in images.items():
        _write(output / "images" / rel, raw)
    _write(output / "records.jsonl", records)
    _write(output / "manifest.json", _canonical(manifest) + b"\n")
    return manifest


def load_stage(dir: Path) -> list[tuple[str, str, GeneratedProblem]]:
    root = Path(dir)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS or manifest["version"] != STAGE_VERSION
            or manifest["status"] != _STATUS or type(manifest["image_sha256"]) is not dict):
        raise ValueError("gen_stage: invalid manifest")
    records = (root / "records.jsonl").read_bytes()
    if _sha(records) != manifest["records_sha256"]:
        raise ValueError("gen_stage: records hash mismatch")
    images, expected = root / "images", manifest["image_sha256"]
    on_disk = ({p.relative_to(images).as_posix() for p in images.rglob("*") if p.is_file()}
               if images.is_dir() else set())
    if on_disk != set(expected) or any(_sha((images / rel).read_bytes()) != digest
                                       for rel, digest in expected.items()):
        raise ValueError("gen_stage: image files do not match manifest")
    loaded: list[tuple[str, str, GeneratedProblem]] = []
    for line in records.decode("utf-8").split("\n")[:-1]:
        record = json.loads(line)
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            raise ValueError("gen_stage: invalid record envelope")
        actor = from_json(ActorItem, record["actor"])
        verifier = from_json(VerifierItem, record["verifier"])
        lineage = from_json(Lineage, record["lineage"])
        if (type(record["generator"]) is not str or type(record["subject"]) is not str
                or record["split"] != lineage.split):
            raise ValueError("gen_stage: record identity mismatch")
        raws = []
        for rel, digest in zip(actor.image_paths, actor.image_sha256, strict=True):
            if expected.get(rel) != digest:
                raise ValueError("gen_stage: record image hash not in manifest")
            raws.append((images / rel).read_bytes())
        loaded.append((record["generator"], record["subject"],
                       GeneratedProblem(actor, verifier, lineage, tuple(raws))))
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage program-generator items for the training pool.")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = stage(json.loads(args.spec.read_bytes()), args.output)
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
