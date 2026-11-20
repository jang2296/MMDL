"""Write-once candidate builds and pre-learning selection. Preparation is not approval."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .generators import get_generator
from .generators.common import GeneratedProblem
from .schema import ActorItem, Split, to_json, validate_actor
from .vault import _read, _safe_root, read_actors, write_split

SPLITS: tuple[Split, ...] = ("train", "dev", "sealed", "style_sealed")


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def generate_pool(generator: str, n: int, seed: int, split: Split,
                  mode_mix: float = 0.5) -> list[GeneratedProblem]:
    if (type(n) is not int or n < 1 or type(seed) is not int or seed < 0 or split not in SPLITS
            or type(mode_mix) not in (float, int) or not math.isfinite(mode_mix) or not 0 <= mode_mix <= 1):
        raise ValueError("dataset: explicit positive pool size/seed/split/mode mixture required")
    module = get_generator(generator)
    if module.__file__ is None:
        raise ValueError("dataset: source-bound generator module required")
    module_source = Path(module.__file__).read_bytes()
    common_source = Path(__file__).with_name("generators").joinpath("common.py").read_bytes()
    dependencies = b"".join(b"\0" + Path(dependency.__file__).read_bytes()
                            for dependency in getattr(module, "SOURCE_DEPENDENCIES", ()))
    code_revision = _sha(module_source + b"\0" + common_source + dependencies)
    result, structures = [], set()
    for i in range(n*20):
        mode = "obs" if int.from_bytes(hashlib.sha256(f"{generator}|{seed}|{split}|{i}".encode()).digest()[:8],
                                      "big") / 2**64 < mode_mix else "plain"
        args = dict(index=i, seed=seed, split=split, mode=mode,
                    style="shift" if split == "style_sealed" else "default")
        first, second = module.generate(**args), module.generate(**args)
        module.self_check(first)
        if first != second:
            raise ValueError("dataset: nondeterministic generator output; no sample substitution")
        if first.lineage.split != split or first.actor.mode != mode:
            raise ValueError("dataset: generator changed requested split/mode")
        if first.lineage.structure_hash in structures:
            continue
        structures.add(first.lineage.structure_hash)
        result.append(replace(first, lineage=replace(first.lineage, source_revision=code_revision)))
        if len(result) == n:
            return result
    raise ValueError("dataset: insufficient unique structures within fixed candidate bound")


def load_images(actor: ActorItem, image_root: Path) -> list[Image.Image]:
    validate_actor(actor)
    root = _safe_root(image_root)
    images = []
    for rel, expected in zip(actor.image_paths, actor.image_sha256, strict=True):
        path = root / rel
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("dataset: image aliases/path escape forbidden")
        raw = _read(path)
        if _sha(raw) != expected:
            raise ValueError("dataset: image byte hash mismatch")
        with Image.open(io.BytesIO(raw)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("dataset: animated input forbidden")
            image.load()
            # Explicit white compositing, rather than silently discarding alpha to black.
            if "A" in image.getbands() or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                images.append(Image.alpha_composite(background, rgba).convert("RGB"))
            else:
                images.append(image.convert("RGB"))
    return images


def build_splits(root: Path, pools: dict[str, list[GeneratedProblem]], *,
                 index: dict | None = None, source_registry: dict | None = None,
                 evidence: dict | None = None, allow_empty_heldout: bool = False) -> dict:
    """Candidate generation/solver checks never assert human review or training permission.

    allow_empty_heldout (explicit opt-in, default off) lets a selected-training vault keep empty dev/sealed/
    style_sealed splits; train must stay nonempty and the opt-in is recorded in the build manifest."""
    if set(pools) != set(SPLITS) or not pools["train"] or (
            not allow_empty_heldout and any(not rows for rows in pools.values())):
        raise ValueError("dataset: all four nonempty splits required")
    seen_ids: set[str] = set()
    owners: dict[str, str] = {}
    image_owners: dict[str, str] = {}
    images: dict[str, bytes] = {}
    sources: set[str] = set()
    for split, rows in pools.items():
        for record in rows:
            for part in (record.actor, record.verifier, record.lineage):
                to_json(part)
            if (record.lineage.split != split or record.actor.problem_id in seen_ids
                    or record.lineage.benchmark_derived or record.lineage.model_touched):
                raise ValueError("dataset: duplicate identity, wrong split or forbidden lineage")
            seen_ids.add(record.actor.problem_id)
            family = record.lineage.structure_hash
            if family in owners and owners[family] != split:
                raise ValueError("dataset: structure family crosses splits")
            owners[family] = split
            sources.add(record.lineage.source)
            if len(record.image_bytes) != len(record.actor.image_paths):
                raise ValueError("dataset: image coverage mismatch")
            for rel, raw, digest in zip(record.actor.image_paths, record.image_bytes,
                                        record.actor.image_sha256, strict=True):
                if _sha(raw) != digest or rel in images and images[rel] != raw:
                    raise ValueError("dataset: image identity collision")
                if digest in image_owners and image_owners[digest] != split:
                    raise ValueError("dataset: same image crosses splits")
                image_owners[digest] = split
                images[rel] = raw
    root = Path(root)
    _safe_root(root.parent, create=True)
    root.mkdir(mode=0o700, exist_ok=False)
    for rel, raw in images.items():
        _write(root / "images" / rel, raw)
    manifests = {}
    for split, rows in pools.items():
        manifests[split] = write_split(root / split, [r.actor for r in rows],
                                      [r.verifier for r in rows], [r.lineage for r in rows])
        for actor in read_actors(root / split):
            for image in load_images(actor, root / "images"):
                image.close()
    if index is not None:
        if source_registry is None:
            raise ValueError("dataset: contamination scan requires source registry")
        from .contamination import scan
        exclusion = scan({s: root / s for s in SPLITS}, root / "images", index, source_registry, evidence)
    else:
        exclusion = {"status": "BLOCKED_NOT_CHECKED", "allowlist": [],
                     "limitations": ["No complete pinned exclusion index supplied."]}
    _write(root / "contamination" / "receipt_v2.json", _bytes(exclusion)+b"\n")
    manifest = {"format": "notebook-rl-candidate-build-v1", "status": "PREPARED_PENDING_GATES",
        "training_allowed": False, "sources": sorted(sources),
        "counts": {s: len(pools[s]) for s in SPLITS},
        "split_manifests": {s: _sha((root/s/"manifest.yaml").read_bytes()) for s in SPLITS},
        "image_sha256": {rel: _sha(raw) for rel, raw in images.items()},
        "exclusion_receipt_sha256": _sha((root/"contamination/receipt_v2.json").read_bytes()),
        "human_render_review": False, "human_leak_review": False,
        "structure_split_disjoint": True,
        "dataset_code_sha256": _sha(Path(__file__).read_bytes())}
    if allow_empty_heldout:
        manifest["allow_empty_heldout"] = True
    _write(root / "manifest.yaml", _bytes(manifest)+b"\n")
    return manifest


def freeze_selection(phase_a_report: Path, rules: dict, *, output: Path) -> dict:
    """No answers/teacher targets here: only pre-learning rollout counts select difficulty."""
    expected_rules = {"max_pass_rate", "max_zero_pass_share", "blind_floor", "truncation_family_max",
                      "max_selected", "min_open_answer_fraction"}
    if set(rules) != expected_rules or any(type(rules[k]) not in (int, float) or not 0 <= rules[k] <= 1
            for k in expected_rules-{"max_selected"}) or type(rules["max_selected"]) is not int or rules["max_selected"] < 1:
        raise ValueError("dataset: invalid frozen selection rules")
    raw = _read(phase_a_report)
    report = json.loads(raw)
    if (report.get("status") != "PHASE_A_COMPLETED" or report.get("policy_version") != 0
            or report.get("split") != "train" or not report.get("build_id") or not report.get("rows")
            or report.get("candidate_ids_sha256") != _sha(_bytes(sorted(r["problem_id"] for r in report["rows"])))):
        raise ValueError("dataset: completed fresh-F0 train candidate coverage required")
    caps = report["max_new_tokens"]
    if set(caps) != {"1", "2", "3"} or any(type(v) is not int or not 2048 <= v <= 4096 for v in caps.values()):
        raise ValueError("dataset: separately measured 2K..4K turn budgets required")
    ids: set[str] = set()
    family_counts: dict[str, list[int]] = {}
    for row in report["rows"]:
        if (row["problem_id"] in ids or row["n"] != 8 or row["blind_n"] != 4
                or any(type(row[k]) is not int for k in ("correct", "n", "blind_correct", "blind_n", "truncated", "n_choices"))
                or not 0 <= row["correct"] <= 8 or not 0 <= row["blind_correct"] <= 4
                or not 0 <= row["truncated"] <= 8 or row["n_choices"] not in (0, 10)
                or type(row["context_ok"]) is not bool):
            raise ValueError("dataset: malformed or duplicate Phase A observations")
        ids.add(row["problem_id"])
        counts = family_counts.setdefault(row["family"], [0, 0])
        counts[0] += row["truncated"]
        counts[1] += row["n"]
    selected: list[str] = []
    excluded: list[dict] = []
    zero: list[str] = []
    nonzero: list[str] = []
    for row in sorted(report["rows"], key=lambda r: r["problem_id"]):
        p, blind = row["correct"]/8, row["blind_correct"]/4
        chance = 1/row["n_choices"] if row["n_choices"] else 0
        a, b = family_counts[row["family"]]
        reason = ("CONTEXT" if not row["context_ok"] else "FAMILY_TRUNCATION" if a/b > rules["truncation_family_max"]
                  else "TOO_EASY" if p > rules["max_pass_rate"] else "BLIND_SOLVABLE"
                  if blind > max(chance, rules["blind_floor"]) else None)
        if reason:
            excluded.append({"problem_id": row["problem_id"], "reason": reason})
        else:
            (zero if p == 0 else nonzero).append(row["problem_id"])
    selected.extend(nonzero[:rules["max_selected"]])
    zero_cap = (len(selected) if rules["max_zero_pass_share"] == .5 else
                math.floor(len(selected)*rules["max_zero_pass_share"]/(1-rules["max_zero_pass_share"]))
                if rules["max_zero_pass_share"] < 1 else len(zero))
    selected.extend(zero[:min(zero_cap, rules["max_selected"]-len(selected))])
    for pid in set(zero+nonzero)-set(selected):
        excluded.append({"problem_id": pid, "reason": "ZERO_PASS_CAP" if pid in zero else "POOL_CAP"})
    row_by_id = {r["problem_id"]: r for r in report["rows"]}
    if not selected or sum(row_by_id[pid]["n_choices"] == 0 for pid in selected)/len(selected) < rules["min_open_answer_fraction"]:
        raise ValueError("dataset: insufficient selected/open-answer problems; do not silently change the selection policy")
    result = {"selection_id": "notebook-rl-selection-v1", "build_id": report["build_id"],
        "phase_a_report_sha256": _sha(raw), "rules": rules, "max_new_tokens": caps,
        "selected_train": sorted(selected), "excluded": sorted(excluded, key=lambda r: r["problem_id"]),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "training_allowed": False,
        "status": "SELECTION_FROZEN_NOT_TRAINING_APPROVAL"}
    _write(output, _bytes(result)+b"\n")
    return result
