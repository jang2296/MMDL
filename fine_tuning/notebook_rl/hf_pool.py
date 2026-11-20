"""Label-blind quota selection of pinned HF train rows -> an UNSCANNED candidate build.

The output is a normal notebook_rl candidate build (vault + images) produced by
dataset.build_splits with no exclusion index, so its receipt is BLOCKED_NOT_CHECKED and
nothing here grants training permission. Selection uses only a seed-keyed hash of
(source, native_id): never answers, model outputs or benchmark data.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import inspect
import io
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from PIL import Image

from .adapters.hf_common import SUBJECTS, Candidate, SourceSpec, candidate_order, pixel_sha256, split_for, to_problem
from .dataset import SPLITS, _bytes, _write, build_splits
from .generators.common import GeneratedProblem
from .schema import to_json

PLAN_VERSION = "hf-pool-plan-v1"
PLAN_V2 = "hf-pool-plan-v2"
POOL_VERSION = "hf-pool-build-v1"
ITEM_INDEX_VERSION = "hf-pool-item-index-v1"
OVERSAMPLE = 3
Registry = dict[str, tuple[SourceSpec, Callable[[Path], Iterator[Candidate]]]]
Staged = tuple[str, str, GeneratedProblem]   # (generator name, subject, problem already split-assigned)


def default_registry() -> Registry:
    from .adapters.hf_registry import sources
    return sources()


def load_plan(path: Path, registry: Registry) -> dict:
    plan = json.loads(Path(path).read_text())
    if type(plan) is not dict:
        raise ValueError("plan: must be an object")
    v1_keys = {"version", "seed", "quotas"}
    if plan.get("version") == PLAN_V2:
        if set(plan) != v1_keys | {"stage_sha256"}:
            raise ValueError("plan: keys must be exactly version, seed, quotas, stage_sha256")
        stage = plan["stage_sha256"]
        if stage is not None and not (type(stage) is str and re.fullmatch(r"[0-9a-f]{64}", stage)):
            raise ValueError("plan: stage_sha256 must be 64 lowercase hex characters or null")
    elif set(plan) != v1_keys:
        raise ValueError("plan: keys must be exactly version, seed, quotas")
    elif plan["version"] != PLAN_VERSION:
        raise ValueError("plan: unsupported version")
    if type(plan["seed"]) is not int or plan["seed"] < 0:
        raise ValueError("plan: seed must be a nonnegative integer")
    if type(plan["quotas"]) is not dict:
        raise ValueError("plan: quotas must be an object")
    totals = dict.fromkeys(SPLITS, 0)
    for source, subjects in plan["quotas"].items():
        if source not in registry:
            raise ValueError(f"plan: unknown source {source!r}")
        if type(subjects) is not dict:
            raise ValueError("plan: subjects must be an object")
        for subject, splits in subjects.items():
            if subject not in SUBJECTS:
                raise ValueError(f"plan: unknown subject {subject!r}")
            if type(splits) is not dict or set(splits) != set(SPLITS):
                raise ValueError("plan: split keys must be exactly the four splits")
            for split, n in splits.items():
                if type(n) is not int or n < 0:
                    raise ValueError("plan: quotas must be nonnegative integers")
                totals[split] += n
    empty = [split for split, n in totals.items() if n < 1]
    if empty:
        raise ValueError(f"plan: no quota for split(s) {empty}")
    return plan


def _norm_qc(question: str, choices: tuple[str, ...]) -> str:
    return " ".join((question + "\x1f" + "\x1f".join(choices)).casefold().split())


def _norm_text(c: Candidate) -> str:
    return _norm_qc(c.question, c.choices)


def select(plan: dict, raw_root: Path, registry: Registry,
           staged: tuple[Staged, ...] | list[Staged] = ()) -> tuple[dict[str, list[GeneratedProblem]], dict]:
    seed, quotas = plan["seed"], plan["quotas"]
    dropped = {"duplicate_image": 0, "duplicate_text": 0, "invalid": 0}
    # Per bucket a max-heap of the q*OVERSAMPLE smallest order keys: (-key, arrival, candidate).
    heaps: dict[tuple[str, str, str], list] = {}
    arrival = 0
    for source in sorted(quotas):
        spec, iterate = registry[source]
        wanted = quotas[source]
        for c in iterate(Path(raw_root) / spec.repo.replace("/", "__")):
            if c.subject not in wanted:
                continue
            try:
                split = split_for(c)
            except ValueError:
                dropped["invalid"] += 1
                continue
            cap = wanted[c.subject][split] * OVERSAMPLE
            if cap == 0:
                continue
            key = int(candidate_order(c, seed), 16)
            heap = heaps.setdefault((source, c.subject, split), [])
            arrival += 1
            if len(heap) < cap:
                heapq.heappush(heap, (-key, arrival, c))
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, (-key, arrival, c))
    pools: dict[str, list[GeneratedProblem]] = {split: [] for split in SPLITS}
    counts: dict[str, dict] = {split: {} for split in SPLITS}
    shortfalls: list[dict] = []
    native_ids: dict[str, list[str]] = {}
    subjects: dict[str, str] = {}   # recorded directly for item_index.json; popped before pool_manifest.json
    taken_images: set[str] = set()
    taken_texts: set[str] = set()
    next_index = 0
    for source in sorted(quotas):
        spec = registry[source][0]
        for subject in sorted(quotas[source]):
            for split in sorted(quotas[source][subject]):
                quota = quotas[source][subject][split]
                if quota == 0:
                    continue
                kept = sorted(heaps.pop((source, subject, split), ()), key=lambda e: (-e[0], e[1]))
                accepted = 0
                for _, _, c in kept:
                    if accepted == quota:
                        break
                    try:
                        problem = to_problem(c, spec, index=next_index)
                    except (ValueError, OSError):
                        dropped["invalid"] += 1
                        continue
                    # Every image, not only the first: a shared second image would otherwise cross splits.
                    image_keys = {pixel_sha256(raw) for raw in problem.image_bytes}
                    # Same question over different pictures is a different item: key on the first image too.
                    text_key = _norm_text(c) + "\x1e" + pixel_sha256(problem.image_bytes[0])
                    if image_keys & taken_images:
                        dropped["duplicate_image"] += 1
                        continue
                    if text_key in taken_texts:
                        dropped["duplicate_text"] += 1
                        continue
                    taken_images.update(image_keys)
                    taken_texts.add(text_key)
                    pools[split].append(problem)
                    native_ids[problem.actor.problem_id] = [source, c.native_id]
                    subjects[problem.actor.problem_id] = subject
                    next_index += 1
                    accepted += 1
                if accepted:
                    counts[split].setdefault(source, {})[subject] = accepted
                if accepted < quota:
                    shortfalls.append({"source": source, "subject": subject, "split": split,
                                       "quota": quota, "accepted": accepted})
    # Staged generator items join after the HF selection, under the same global dedup.
    hf_ids = {p.actor.problem_id for rows in pools.values() for p in rows}
    hf_paths = {path for rows in pools.values() for p in rows for path in p.actor.image_paths}
    for name, subject, problem in staged:
        split = problem.lineage.split
        if split not in pools:
            raise ValueError(f"hf_pool: staged problem has unknown split {split!r}")
        if problem.actor.problem_id in hf_ids or hf_paths & set(problem.actor.image_paths):
            raise ValueError("hf_pool: staged problem collides with an HF problem id or image path")
        image_keys = {pixel_sha256(raw) for raw in problem.image_bytes}
        text_key = (_norm_qc(problem.actor.question, problem.actor.choices)
                    + "\x1e" + pixel_sha256(problem.image_bytes[0]))
        if image_keys & taken_images:
            dropped["duplicate_image"] += 1
            continue
        # Text is checked against HF rows only: generator questions are fixed templates whose instance lives in
        # the image, so registering them would drop every staged item after the first per template.
        if text_key in taken_texts:
            dropped["duplicate_text"] += 1
            continue
        taken_images.update(image_keys)
        pools[split].append(problem)
        native_ids[problem.actor.problem_id] = [name, problem.actor.problem_id]
        subjects[problem.actor.problem_id] = subject
        by_subject = counts[split].setdefault(problem.actor.source, {})
        by_subject[subject] = by_subject.get(subject, 0) + 1
    return pools, {"counts": counts, "shortfalls": shortfalls, "dropped": dropped, "native_ids": native_ids,
                   "subjects": subjects}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fill_printed_text(pools: dict[str, list[GeneratedProblem]], native_ids: dict[str, list[str]],
                       registry: Registry, ocr) -> list[str]:
    """OCR printed_text for HF problems with upstream PROGRAM labels (the scan reads it instead of OCR for them).

    HUMAN problems and staged generator problems (names absent from the registry) are never touched.
    """
    todo = [(split, i) for split, rows in pools.items() for i, p in enumerate(rows)
            if native_ids[p.actor.problem_id][0] in registry
            and registry[native_ids[p.actor.problem_id][0]][0].label_provenance == "PROGRAM"
            and not p.verifier.printed_text]
    if todo and ocr is None:
        raise ValueError("external PROGRAM sources need --tesseract-root for printed_text")
    filled = []
    for split, i in todo:
        problem = pools[split][i]
        lines = []
        for raw in problem.image_bytes:
            with Image.open(io.BytesIO(raw)) as image:
                lines += [ln.strip() for ln in ocr.recognize(image)["text"].splitlines() if ln.strip()]
        verifier = dataclasses.replace(problem.verifier, printed_text=tuple(lines) or ("",))
        to_json(verifier)
        pools[split][i] = dataclasses.replace(problem, verifier=verifier)
        filled.append(problem.actor.problem_id)
    return sorted(filled)


def item_index(pools: dict[str, list[GeneratedProblem]], native_ids: dict[str, list[str]],
               subjects: dict[str, str], plan_version: str, registry: Registry, split_manifests: dict) -> dict:
    """problem_id -> provenance (ids/hashes/metadata only: no question text, choices or gold)."""
    here = Path(__file__).parent
    used = {origin for origin, _ in native_ids.values() if origin in registry}
    code = {name: _sha_file(Path(inspect.getfile(registry[name][1]))) for name in used}
    stage_sha = _sha_file(here / "gen_stage.py")   # staged generator items: the adapter is the stage loader
    items = {}
    for split, rows in pools.items():
        for p in rows:
            pid = p.actor.problem_id
            origin, native_id = native_ids[pid]
            items[pid] = {"source": p.lineage.source, "source_revision": p.lineage.source_revision,
                          "native_id": native_id, "subject": subjects[pid], "split": split,
                          "family": p.actor.family, "image_sha256": list(p.actor.image_sha256),
                          "plan_version": plan_version,
                          "adapter_code_sha256": code[origin] if origin in registry else stage_sha}
    return {"version": ITEM_INDEX_VERSION, "split_manifests": split_manifests, "items": items}


def build(plan_path: Path, raw_root: Path, output: Path, *, registry: Registry | None = None,
          stage_dir: Path | None = None, stage_loader: Callable[[Path], list[Staged]] | None = None,
          ocr=None) -> dict:
    registry = registry if registry is not None else default_registry()
    plan = load_plan(plan_path, registry)
    if plan["version"] == PLAN_V2 and plan["stage_sha256"] is not None and stage_dir is None:
        raise ValueError("hf_pool: the plan pins a generator stage but no stage directory was given")
    staged: list[Staged] = []
    stage_sha = None
    if stage_dir is not None:
        if plan["version"] != PLAN_V2:
            raise ValueError("hf_pool: a generator stage requires a v2 plan")
        stage_sha = _sha_file(Path(stage_dir) / "manifest.json")
        if plan["stage_sha256"] != stage_sha:
            raise ValueError("hf_pool: stage manifest sha256 differs from the plan")
        if stage_loader is None:
            from .gen_stage import load_stage
            stage_loader = load_stage
        staged = list(stage_loader(Path(stage_dir)))
    pools, report = select(plan, raw_root, registry, staged)
    subjects = report.pop("subjects")
    ocr_filled = _fill_printed_text(pools, report["native_ids"], registry, ocr)
    empty = [split for split, rows in pools.items() if not rows]
    if empty:
        raise ValueError(f"hf_pool: no accepted problems for split(s) {empty}; refusing to pad")
    build_manifest = build_splits(output, pools)
    raw_manifest = Path(raw_root) / "download_manifest.jsonl"
    here = Path(__file__).parent
    adapters = sorted((here / "adapters").glob("hf_*.py"))
    generators = sorted((here / "generators").glob("*.py"))
    manifest = {
        "version": POOL_VERSION, "status": "BUILT_UNSCANNED_NOT_APPROVED", "training_allowed": False,
        "plan_sha256": _sha_file(plan_path), "seed": plan["seed"], "stage_manifest_sha256": stage_sha,
        "raw_download_manifest_sha256": _sha_file(raw_manifest) if raw_manifest.exists() else None,
        "build_manifest_sha256": _sha_file(Path(output) / "manifest.yaml"),
        "adapter_code_sha256": {**{p.name: _sha_file(p) for p in [*adapters, Path(__file__)]},
                                "gen_stage.py": _sha_file(here / "gen_stage.py"),
                                **{f"generators/{p.name}": _sha_file(p) for p in generators}},
        "printed_text_ocr": {"engine_identity": getattr(ocr, "identity", None) or None, "problem_ids": ocr_filled},
        **report,
    }
    (Path(output) / "pool_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
    index = item_index(pools, report["native_ids"], subjects, plan["version"], registry, build_manifest["split_manifests"])
    _write(Path(output) / "item_index.json", _bytes(index) + b"\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fine_tuning.notebook_rl.hf_pool")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--plan", type=Path, required=True)
    b.add_argument("--raw-root", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--stage", type=Path, default=None, help="generator stage dir (gen_stage); needs a v2 plan")
    b.add_argument("--tesseract-root", type=Path, default=None,
                   help="extracted tesseract 4.1.1 root; OCRs printed_text of external PROGRAM-labelled sources")
    args = parser.parse_args(argv)
    kwargs = {"stage_dir": args.stage}
    if args.tesseract_root is not None:
        from .exclusion_engines import OCR
        kwargs["ocr"] = OCR(args.tesseract_root)
    result = build(args.plan, args.raw_root, args.output, **kwargs)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "shortfalls": len(result["shortfalls"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

