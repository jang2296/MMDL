"""Read-only exclusion scans, not dataset/training approval or contamination-free proof.

V2 index contract: schema=notebook-rl-exclusion-index-v2; purpose, training_allowed,
status, producer_sha256, coverage.files/record_count, records and images. Each record
has ref={dataset_id,configuration,split,id,file_sha256}, texts={field: [strings]},
and parallel image_sha256=[raw hashes], image_fields=[native column names]. Each
images[raw SHA] has pixels_sha256 and variants
from image_fingerprints(). Raw benchmark text stays inside the private index.

Evidence adapters never run/download models. OCR and embedding entries bind exact
image/text SHA, index SHA, pinned engine/revision/artifact, and complete coverage.
source_registry.scanner_policy pins their engines and a calibration SHA; calibration
declares frozen thresholds and measured positive/negative counts and TPR/FPR. These
upstream declarations are checked, not authenticated or measured again by this scan.
No missing prerequisite can yield EXCLUSION_CHECKED_WITH_LIMITS.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from fine_tuning import audit

from . import vault
from .schema import load_protocol

VERSION = "notebook-rl-exclusion-v2"
FINGERPRINT_VERSION = "rgb32-dct8-dhash9x8-lanczos-v1"
_HERE = Path(__file__).resolve().parent
_INVENTORY = _HERE / "benchmark_inventory_v1.json"
_INVENTORY_SHA256 = "743e03e10836e035c3cdb245d2bbc255e8c19f5fd70e00735d872435e583ab21"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_HEX64 = re.compile(r"[0-9a-f]{16}\Z")
_REVISIONS = {"MMMU/MMMU": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
              "MMMU/MMMU_Pro": "563f3e84bb3b90893083a1f039cfa13077f2302b"}
_COUNTS = {("MMMU/MMMU", "dev"): 150, ("MMMU/MMMU", "validation"): 900,
           ("MMMU/MMMU", "test"): 10500, ("MMMU/MMMU_Pro", "standard10"): 1730,
           ("MMMU/MMMU_Pro", "standard4"): 1730, ("MMMU/MMMU_Pro", "vision"): 1730}
_SPLITS = {"train", "dev", "sealed", "style_sealed"}
_VARIANTS = ("full", "center", *(f"tile{n}-{r}-{c}" for n in (2, 3) for r in range(n) for c in range(n)))
_STEPS = ("source_rights", "lineage_origin", "question_exact", "text_ngrams", "ocr_ngrams",
          "image_exact", "perceptual", "sscd", "siglip", "family", "all_splits", "frozen_manifest")


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("exclusion: text must be a string")
    value = unicodedata.normalize("NFKC", text)
    value = re.sub(r"<image(?:[\s_]*\d+)?>", " ", value, flags=re.I)
    value = re.sub(r"^\s*(?:Question\s*:\s*)+", "", value, flags=re.I)
    return " ".join(value.casefold().split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text))


def _grams(text: str, n: int) -> set[str]:
    words = _tokens(text)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class BKTree:
    """Integer Hamming metric; equal fingerprints share a node rather than a chain."""
    def __init__(self) -> None:
        self.nodes: list[tuple[int, list[dict], dict[int, int]]] = []

    def add(self, value: int, reference: dict) -> None:
        if not self.nodes:
            self.nodes.append((value, [reference], {}))
            return
        node = 0
        while True:
            key, refs, children = self.nodes[node]
            distance = (value ^ key).bit_count()
            if not distance:
                refs.append(reference)
                return
            if distance not in children:
                children[distance] = len(self.nodes)
                self.nodes.append((value, [reference], {}))
                return
            node = children[distance]

    def search(self, value: int, radius: int = 3) -> list[tuple[int, dict]]:
        if type(radius) is not int or radius < 0:
            raise ValueError("exclusion: nonnegative Hamming radius required")
        found: list[tuple[int, dict]] = []
        pending = [0] if self.nodes else []
        while pending:
            node = pending.pop()
            key, refs, children = self.nodes[node]
            distance = (value ^ key).bit_count()
            if distance <= radius:
                found.extend((distance, ref) for ref in refs)
            pending.extend(child for edge, child in children.items() if distance - radius <= edge <= distance + radius)
        return found


def _low_information(value: str) -> bool:
    return len(set(bytes.fromhex(value))) <= 1


def image_fingerprints(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if min(w, h) < 3:
        raise ValueError("exclusion: image too small for 3x3 tiles")
    regions = {"full": (0, 0, w, h), "center": (w // 4, h // 4, w - w // 4, h - h // 4)}
    regions.update({f"tile{n}-{r}-{c}": (w * c // n, h * r // n, w * (c + 1) // n, h * (r + 1) // n)
                    for n in (2, 3) for r in range(n) for c in range(n)})
    transform = np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32)[None, :] + 1) / 64)
    variants = {}
    for name, box in regions.items():
        gray = rgb.crop(box).convert("L")
        small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        bits = (small[:, :-1] > small[:, 1:]).reshape(-1)
        dhash = f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"
        pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        dct = transform @ pixels @ transform.T
        phash = f"{sum(int(bit) << i for i, bit in enumerate((dct > np.median(dct)).reshape(-1))):016x}"
        variants[name] = {"dhash": dhash, "phash": phash,
                          "low_information": bool(np.std(pixels) < 1)}
    return {"pixels_sha256": audit.image_fingerprint(rgb)["pixels_sha256"], "variants": variants}


def _coverage(data: dict) -> tuple[bool, list[str]]:
    coverage = data.get("coverage")
    if (data.get("status") != "COMPLETE" or data.get("training_allowed") is not False
            or data.get("purpose") != "exclusion_fingerprints_only"
            or data.get("coverage_verified") is not True or type(coverage) is not dict
            or type(coverage.get("files")) is not list or not coverage["files"]):
        raise ValueError("exclusion: complete exclusion-only coverage provenance required")
    counts: Counter = Counter()
    paths, digests, expected = set(), set(), {}
    subjects: dict[str, set[str]] = defaultdict(set)
    for entry in coverage["files"]:
        if (type(entry) is not dict or not isinstance(entry.get("path"), str) or not entry["path"]
                or entry["path"] in paths or type(entry.get("records")) is not int or entry["records"] <= 0
                or entry.get("dataset_id") not in _REVISIONS
                or entry.get("revision") != _REVISIONS[entry["dataset_id"]]
                or not isinstance(entry.get("configuration"), str) or not entry["configuration"]
                or not isinstance(entry.get("sha256"), str) or not _SHA.fullmatch(entry["sha256"])):
            raise ValueError("exclusion: invalid coverage file identity/revision/count/hash")
        dataset, config, split = entry["dataset_id"], entry["configuration"], entry.get("split")
        if dataset == "MMMU/MMMU":
            group = (dataset, split)
            if not isinstance(split, str):
                raise ValueError("exclusion: native split required")
            subjects[split].add(config)
        else:
            aliases = {"standard (10 options)": "standard10", "standard (4 options)": "standard4",
                       "standard10": "standard10", "standard4": "standard4", "vision": "vision"}
            if config not in aliases or split != "test":
                raise ValueError("exclusion: invalid Pro configuration/split")
            group = (dataset, aliases[config])
        if group not in _COUNTS:
            raise ValueError("exclusion: unsupported benchmark coverage scope")
        paths.add(entry["path"])
        digests.add(entry["sha256"])
        expected[entry["path"]] = entry["records"]
        counts[group] += entry["records"]
    total = sum(counts.values())
    if (type(coverage.get("record_count")) is not int or type(data.get("record_count")) is not int
            or coverage["record_count"] != total or data["record_count"] != total
            or data.get("source_files") != {e["path"]: e["sha256"] for e in coverage["files"]}
            or data.get("per_file_record_counts") != expected or type(data.get("image_count")) is not int
            or data["image_count"] <= 0):
        raise ValueError("exclusion: inconsistent coverage/source file inventory")
    complete = (dict(counts) == _COUNTS and len(paths) == 100 and len(digests) == 100
                and all(len(subjects[s]) == 30 for s in ("dev", "validation", "test"))
                and subjects["dev"] == subjects["validation"] == subjects["test"]
                and _pinned_coverage(coverage["files"]))
    return complete, [] if complete else ["Incomplete native MMMU/Pro file/record coverage"]


def _pinned_coverage(files: list[dict]) -> bool:
    """Bind native shards, including unequal/sharded test subjects, to pinned bytes."""
    raw = vault._read(_INVENTORY)
    if hashlib.sha256(raw).hexdigest() != _INVENTORY_SHA256:
        raise ValueError("exclusion: pinned native inventory changed")
    inventory = json.loads(raw, object_pairs_hook=vault._unique)
    expected = {entry["sha256"]: entry for entry in inventory["files"]}
    aliases = {"standard10": "standard (10 options)", "standard4": "standard (4 options)"}
    if len(files) != len(expected) or {entry["sha256"] for entry in files} != set(expected):
        return False
    for entry in files:
        reference = expected[entry["sha256"]]
        config = entry["configuration"]
        if entry["dataset_id"] == "MMMU/MMMU_Pro":
            config = aliases.get(config, config)
        if (config != reference["configuration"]
                or any(entry[key] != reference[key] for key in ("dataset_id", "revision", "split", "records"))
                or ("bytes" in entry and (type(entry["bytes"]) is not int or entry["bytes"] != reference["bytes"]))):
            return False
        native = reference["native_path"]
        if "native_path" in entry:
            if entry["native_path"] != native:
                return False
        else:
            # Historical receipts use either snapshot paths or SHA-named cache blobs.
            path = Path(entry["path"])
            if path.name != entry["sha256"] and path.as_posix() != native and not path.as_posix().endswith("/" + native):
                return False
    return True


def _text_cache(entries: list[dict]) -> dict:
    exact: dict[str, list[dict]] = defaultdict(list)
    sets: dict[int, dict[int, set[str]]] = {}
    inverted: dict[int, dict[str, set[int]]] = {n: defaultdict(set) for n in (5, 7, 8)}
    for i, entry in enumerate(entries):
        text = normalize_text(entry["text"])
        if not text:
            continue
        exact[hashlib.sha256(text.encode()).hexdigest()].append(entry)
        sets[i] = {n: _grams(text, n) for n in (5, 7, 8)}
        for n in inverted:
            for gram in sets[i][n]:
                inverted[n][gram].add(i)
    return {"entries": entries, "exact": exact, "sets": sets, "inverted": inverted}


def _prepare(data: dict, partial_name: str | None) -> dict:
    full, limits = _coverage(data)
    historical = data.get("schema") != "notebook-rl-exclusion-index-v2"
    if partial_name is not None and (not isinstance(partial_name, str) or not partial_name.strip()):
        raise ValueError("exclusion: a named partial scan is required")
    if (historical or not full) and partial_name is None:
        raise ValueError("exclusion: historical/incomplete index cannot authorize full exclusion; name a partial scan")
    texts, images, trees = [], {}, {"dhash": BKTree(), "phash": BKTree()}
    reverse: dict[str, list[dict]] = defaultdict(list)
    if historical:
        limits.extend(["Historical index has no record/image reverse references, normalized short/choice/OCR coverage",
                       "Historical index lacks crop/tile pHash, OCR and semantic evidence coverage"])
        for key in ("text_sha256", "pixels_sha256", "image_bytes_sha256"):
            if type(data.get(key)) is not list or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in data[key]):
                raise ValueError("exclusion: malformed historical hash inventory")
        if (type(data.get("long_text_shingles")) is not list
                or any(type(v) is not list or any(not isinstance(x, str) for x in v) for v in data["long_text_shingles"])
                or type(data.get("dhash")) is not list
                or any(not isinstance(v, str) or not _HEX64.fullmatch(v) for v in data["dhash"])):
            raise ValueError("exclusion: malformed historical near-fingerprint inventory")
        for value in data["dhash"]:
            trees["dhash"].add(int(value, 16), {"image_sha256": None, "variant": "historical-full",
                                                   "low_information": _low_information(value)})
        cache = _text_cache([])
        cache["legacy_sets"] = [set(s) for s in data["long_text_shingles"]]
        cache["legacy_inverted"] = defaultdict(set)
        for i, grams in enumerate(cache["legacy_sets"]):
            for gram in grams:
                cache["legacy_inverted"][gram].add(i)
        cache["legacy_exact"] = set(data["text_sha256"]) - {hashlib.sha256(b"").hexdigest()}
        pixel_hashes, raw_hashes = set(data["pixels_sha256"]), set(data["image_bytes_sha256"])
    else:
        if (data.get("producer_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or data.get("fingerprint_version") != FINGERPRINT_VERSION
                or data.get("normalizer_version") != VERSION
                or type(data.get("records")) is not list or len(data["records"]) != data["record_count"]
                or type(data.get("images")) is not dict or not data["images"]):
            raise ValueError("exclusion: stale producer or incomplete v2 records/images")
        images = data["images"]
        source_hashes = {e["sha256"] for e in data["coverage"]["files"]}
        record_ids, occurrences, seen_images = set(), 0, set()
        observed: Counter = Counter()
        for record in data["records"]:
            ref = record.get("ref")
            if (type(ref) is not dict or set(ref) != {"dataset_id", "configuration", "split", "id", "file_sha256"}
                    or not isinstance(ref["id"], str) or not ref["id"]
                    or ref["file_sha256"] not in source_hashes):
                raise ValueError("exclusion: missing benchmark record reverse reference")
            record_key = (ref["dataset_id"], ref["configuration"], ref["split"], ref["id"])
            owners = [f for f in data["coverage"]["files"] if f["sha256"] == ref["file_sha256"]]
            if (record_key in record_ids or len(owners) != 1 or any(ref[k] != owners[0][k]
                                                           for k in ("dataset_id", "configuration", "split"))):
                raise ValueError("exclusion: duplicate/mismatched benchmark record provenance")
            record_ids.add(record_key)
            observed[owners[0]["path"]] += 1
            fields = record.get("texts")
            if (type(fields) is not dict or set(fields) != {"question", "choices", "explanation"}
                    or any(type(v) is not list or any(not isinstance(t, str) for t in v) for v in fields.values())
                    or len(fields["question"]) != 1):
                raise ValueError("exclusion: native question/choices/explanation fields must be explicitly indexed")
            for field, values in fields.items():
                for n, value in enumerate(values):
                    texts.append({"text": value, "field": field, "ref": ref,
                                  "key": _digest([*record_key, field, n])})
            refs = record.get("image_sha256")
            fields = record.get("image_fields")
            if (type(refs) is not list or not refs or any(v not in images for v in refs)
                    or type(fields) is not list or len(fields) != len(refs)
                    or len(set(fields)) != len(fields)
                    or any(not isinstance(f, str) or not re.fullmatch(r"image(?:_[1-7])?", f) for f in fields)):
                raise ValueError("exclusion: incomplete benchmark image references")
            for sha, field in zip(refs, fields, strict=True):
                reverse[sha].append({**ref, "image_field": field})
            occurrences += len(refs)
            seen_images.update(refs)
        if observed != data["per_file_record_counts"] or occurrences != data["image_count"] or seen_images != set(images):
            raise ValueError("exclusion: benchmark record/image/file coverage mismatch")
        if full and (occurrences != 19041 or len(images) != 12168):
            raise ValueError("exclusion: full benchmark native image coverage differs from fixed inventory")
        for raw_sha, image in images.items():
            if (not isinstance(raw_sha, str) or not _SHA.fullmatch(raw_sha) or type(image) is not dict
                    or not isinstance(image.get("pixels_sha256"), str) or not _SHA.fullmatch(image["pixels_sha256"])
                    or type(image.get("variants")) is not dict or set(image["variants"]) != set(_VARIANTS)):
                raise ValueError("exclusion: incomplete full/crop/tile fingerprint inventory")
            for variant, hashes in image["variants"].items():
                if (type(hashes) is not dict or set(hashes) != {"dhash", "phash", "low_information"}
                        or type(hashes["low_information"]) is not bool
                        or any(not isinstance(hashes[k], str) or not _HEX64.fullmatch(hashes[k]) for k in trees)):
                    raise ValueError("exclusion: invalid perceptual fingerprint")
                for kind, tree in trees.items():
                    tree.add(int(hashes[kind], 16), {"image_sha256": raw_sha, "variant": variant,
                            "benchmark_refs": reverse[raw_sha],
                            "low_information": hashes["low_information"] or _low_information(hashes[kind])})
        cache = _text_cache(texts)
        pixel_hashes, raw_hashes = {v["pixels_sha256"] for v in images.values()}, set(images)
    return {"data": data, "sha256": _digest(data), "full_coverage": full and not historical and partial_name is None,
            "partial_name": partial_name, "limitations": limits, "text_cache": cache,
            "images": images, "image_refs": dict(reverse), "trees": trees,
            "pixel_hashes": pixel_hashes, "raw_hashes": raw_hashes}


def load_index(path: Path, *, partial_name: str | None = None) -> dict:
    raw = vault._read(Path(path))
    data = json.loads(raw, object_pairs_hook=vault._unique)
    prepared = _prepare(data, partial_name)
    prepared["path"], prepared["file_sha256"] = str(Path(path).absolute()), hashlib.sha256(raw).hexdigest()
    return prepared


def _text_matches(text: str, field: str, cache: dict) -> list[dict]:
    normal, words = normalize_text(text), _tokens(text)
    result: list[dict] = []
    if not normal:
        return result
    digest = hashlib.sha256(normal.encode()).hexdigest()
    # One-digit answers/options are not exact-question evidence.
    if field == "question" or len(words) >= 5:
        if digest in cache.get("legacy_exact", set()):
            result.append({"reason": "EXACT_TEXT", "action": "QUARANTINE", "ref": "historical-no-record-ref"})
        for other in cache["exact"].get(digest, []):
            result.append({"reason": "EXACT_TEXT", "action": "QUARANTINE", "ref": other.get("ref", other.get("key"))})
    for n in (5, 7, 8):
        own = _grams(text, n)
        candidates = set().union(*(cache["inverted"][n].get(g, set()) for g in own)) if own else set()
        for i in sorted(candidates):
            other = cache["sets"][i][n]
            hit = len(own & other)
            if hit and (hit / min(len(own), len(other)) >= 0.8 or (n == 8 and len(words) >= 20)):
                result.append({"reason": "SHARED_8GRAM_REVIEW" if n == 8 and hit / min(len(own), len(other)) < 0.8 else f"NEAR_TEXT_{n}GRAM",
                               "action": "REVIEW" if n == 8 and hit / min(len(own), len(other)) < 0.8 else "QUARANTINE",
                               "ref": cache["entries"][i].get("ref", cache["entries"][i].get("key"))})
        if n == 7:
            old = set().union(*(cache.get("legacy_inverted", {}).get(g, set()) for g in own)) if own else set()
            if any(len(own & cache["legacy_sets"][i]) / min(len(own), len(cache["legacy_sets"][i])) >= 0.8 for i in old):
                result.append({"reason": "NEAR_TEXT_7GRAM", "action": "QUARANTINE", "ref": "historical-no-record-ref"})
    return result


def cosine_similarities(queries, references) -> np.ndarray:
    q, r = np.asarray(queries, dtype=np.float64), np.asarray(references, dtype=np.float64)
    if (q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1] or not q.shape[1]
            or not np.isfinite(q).all() or not np.isfinite(r).all()):
        raise ValueError("exclusion: finite equal-dimensional embedding matrices required")
    qn, rn = np.linalg.norm(q, axis=1), np.linalg.norm(r, axis=1)
    if not np.isfinite(qn).all() or not np.isfinite(rn).all() or np.any(qn <= 0) or np.any(rn <= 0):
        raise ValueError("exclusion: nonzero finite embedding norms required")
    return np.clip((q / qn[:, None]) @ (r / rn[:, None]).T, -1, 1)


def family_quarantine(items: dict[str, dict], candidates: list[dict]) -> dict:
    owner: dict[str, str] = {}
    parents = {pid: pid for pid in items}
    def root(pid):
        while parents[pid] != pid:
            parents[pid] = parents[parents[pid]]
            pid = parents[pid]
        return pid
    for pid, item in items.items():
        tokens = ["root:" + item["structure_hash"], *("image:" + x for x in item["image_sha256"]),
                  *("pixel:" + x for x in item.get("pixel_sha256", []))]
        for token in tokens:
            if token in owner:
                parents[root(pid)] = root(owner[token])
            owner[token] = pid
    bad = {root(c["problem_id"]) for c in candidates if c["problem_id"] in items}
    quarantined = sorted(pid for pid in items if root(pid) in bad)
    return {"quarantined_ids": quarantined, "allowlist": sorted(set(items) - set(quarantined)),
            "families": sorted({items[pid]["structure_hash"] for pid in quarantined})}


def _source_checks(actor, lineage, registry: dict) -> list[str]:
    sources = registry.get("sources", {})
    if lineage.source not in sources:
        raise ValueError("unregistered source")
    seen, pending = set(), [lineage.source, *lineage.parents]
    blocked_origins = registry.get("upstream_origin_blocklist", [])
    if type(blocked_origins) is not list or any(not isinstance(v, str) for v in blocked_origins):
        raise ValueError("invalid upstream origin blocklist")
    reasons = []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        entry = sources.get(name)
        required = {"rights_approved", "source_known", "benchmark_derived", "model_touched",
                    "label_provenance", "source_revision", "license", "upstream_origin", "parents", "upstream_split"}
        if (type(entry) is not dict or not required <= entry.keys()
                or type(entry["parents"]) is not list or any(not isinstance(p, str) or not p for p in entry["parents"])):
            raise ValueError("incomplete registered source/rights/ancestry evidence")
        if any(type(entry[k]) is not bool for k in ("rights_approved", "source_known", "benchmark_derived", "model_touched")):
            raise ValueError("untyped source policy")
        if (any(not isinstance(entry[k], str) or not entry[k].strip()
                for k in ("source_revision", "license", "upstream_origin"))
                or entry["source_revision"] in ("main", "latest")):
            raise ValueError("unpinned source identity")
        if (entry["rights_approved"] is not True or entry["source_known"] is not True
                or entry["benchmark_derived"] or entry["model_touched"]
                or entry["label_provenance"] not in ("PROGRAM", "HUMAN")
                or entry["upstream_split"] not in ("train", "generated")):
            reasons.append("SOURCE_RIGHTS_OR_PROVENANCE_REJECTED")
        if entry["upstream_origin"] in blocked_origins:
            reasons.append("BLOCKLISTED_UPSTREAM_ORIGIN")
        pending.extend(entry["parents"])
    entry = sources[lineage.source]
    if (actor.source != lineage.source or entry["source_revision"] != lineage.source_revision
            or entry["license"] != lineage.license or entry["upstream_origin"] != lineage.upstream_origin
            or entry["label_provenance"] != lineage.label_provenance):
        reasons.append("SOURCE_IDENTITY_MISMATCH")
    # Reuse the historical cycle/known-parent check without changing its code.
    if not reasons:
        audit.check_lineage({"id": actor.problem_id, "benchmark_derived": False,
                             "lineage": [lineage.source, *lineage.parents]}, sources)
    return sorted(set(reasons))


def _snapshot(split_roots: dict[str, Path], image_root: Path) -> tuple[dict, dict, dict]:
    image_root = vault._safe_root(image_root)
    items, files, images = {}, {}, {}
    for split, raw_root in sorted(split_roots.items()):
        if split not in _SPLITS:
            raise ValueError("unsupported input split")
        root = vault._safe_root(raw_root)
        snapshot = {str(root / name): hashlib.sha256(vault._read(root / name)).hexdigest()
                    for name in ("actor_inputs.jsonl", "verifier_private.jsonl", "lineage.jsonl", "manifest.yaml")}
        actors, verifiers, lineages = vault.read_actors(root), vault.read_verifier(root), vault.read_lineage(root)
        if any(hashlib.sha256(vault._read(Path(path))).hexdigest() != digest for path, digest in snapshot.items()):
            raise ValueError("vault changed while loading scan snapshot")
        if {a.problem_id for a in actors} != set(verifiers) or set(verifiers) != set(lineages):
            raise ValueError("vault identity sets differ")
        files.update(snapshot)
        for actor in actors:
            pid, lineage, verifier = actor.problem_id, lineages[actor.problem_id], verifiers[actor.problem_id]
            if pid in items or lineage.split != split:
                raise ValueError("duplicate ID or lineage split mismatch")
            if actor.source != lineage.source or actor.answer_format != verifier.answer_type or actor.choices != verifier.choices:
                raise ValueError("inconsistent actor/verifier/lineage identity")
            for relative, expected in zip(actor.image_paths, actor.image_sha256, strict=True):
                path = image_root / relative
                vault._safe_root(path.parent)
                raw = vault._read(path)
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError("image SHA differs from actor")
                files[str(path)] = expected
                if expected not in images:
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        images[expected] = image_fingerprints(image)
            fields = [("question", actor.question), *(('choices', c) for c in actor.choices),
                      *(('printed_text', t) for t in verifier.printed_text),
                      ("printed_text", "\n".join(verifier.printed_text)),
                      *(('verifier', t) for t in verifier.oracle_fact_lines),
                      *(('verifier', t) for t in verifier.slots.values()), ("verifier", verifier.gold)]
            def strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for nested in value.values():
                        yield from strings(nested)
                elif isinstance(value, (list, tuple)):
                    for nested in value:
                        yield from strings(nested)
            fields.extend(("verifier", text) for text in strings(verifier.accept))
            combined = normalize_text("\n".join([actor.question, *actor.choices, *verifier.printed_text]))
            items[pid] = {"actor": actor, "verifier": verifier, "lineage": lineage, "split": split,
                          "structure_hash": lineage.structure_hash, "image_sha256": list(actor.image_sha256),
                          "pixel_sha256": [images[sha]["pixels_sha256"] for sha in actor.image_sha256],
                          "fields": fields, "text": combined, "text_sha256": hashlib.sha256(combined.encode()).hexdigest()}
    return items, files, images


def _calibration(evidence: dict, registry: dict, index_sha: str, protocol_sha: str) -> dict:
    calibration, policy = evidence.get("calibration"), registry.get("scanner_policy")
    if (type(calibration) is not dict or type(policy) is not dict
            or policy.get("status") != "FROZEN_BEFORE_INPUT_SCAN"
            or policy.get("calibration_sha256") != _digest(calibration)
            or policy.get("protocol_sha256") != protocol_sha
            or calibration.get("protocol_sha256") != protocol_sha
            or calibration.get("index_sha256") != index_sha
            or calibration.get("frozen_before_scan") is not True):
        raise ValueError("missing/mismatched frozen calibration policy")
    for key in ("positive_count", "negative_count"):
        if type(calibration.get(key)) is not int or calibration[key] <= 0:
            raise ValueError("calibration needs positive and negative examples")
    for key in ("tpr", "fpr"):
        if type(calibration.get(key)) not in (float, int) or not math.isfinite(calibration[key]) or not 0 <= calibration[key] <= 1:
            raise ValueError("calibration needs finite measured TPR/FPR")
    engines = calibration.get("per_engine")
    if type(engines) is not dict or set(engines) != {"dhash", "tiles", "sscd", "siglip"}:
        raise ValueError("calibration needs per-engine fixed-FPR measurements")
    for measurement in engines.values():
        if (type(measurement) is not dict
                or any(type(measurement.get(k)) is not int or measurement[k] <= 0
                       for k in ("positive_count", "negative_count"))
                or any(type(measurement.get(k)) not in (float, int) or not math.isfinite(measurement[k])
                       or not 0 <= measurement[k] <= 1 for k in ("tpr", "fpr"))
                or not isinstance(measurement.get("examples_sha256"), str)
                or not _SHA.fullmatch(measurement["examples_sha256"])):
            raise ValueError("invalid per-engine calibration counts/TPR/FPR/examples binding")
    thresholds = calibration.get("thresholds")
    if (type(thresholds) is not dict or thresholds.get("perceptual_hamming") != 3
            or thresholds.get("sscd_exclude") != 0.95 or thresholds.get("sscd_review") != 0.75
            or type(thresholds.get("siglip_review")) not in (float, int)
            or not 0 < thresholds["siglip_review"] <= 1):
        raise ValueError("missing/altered calibrated thresholds")
    return thresholds


def _engine(stage: dict, name: str, evidence: dict, registry: dict, index_sha: str) -> None:
    policy = registry["scanner_policy"].get("engines", {}).get(name)
    if (type(stage) is not dict or type(policy) is not dict
            or any(stage.get(key) != policy.get(key) for key in ("engine_id", "revision", "artifact_sha256"))
            or not isinstance(stage.get("artifact_sha256"), str) or not _SHA.fullmatch(stage["artifact_sha256"])
            or not isinstance(stage.get("revision"), str) or stage["revision"] in ("", "main", "latest")
            or stage.get("index_sha256") != index_sha
            or stage.get("calibration_sha256") != _digest(evidence["calibration"])):
        raise ValueError(f"{name}: unpinned/stale engine evidence")
    engine = stage.get("engine_id")
    if name == "sscd" and engine != "sscd_disc_mixup":
        raise ValueError("SSCD disc_mixup engine required")
    if name == "siglip" and (not isinstance(engine, str) or not engine.startswith("google/siglip")):
        raise ValueError("pinned SigLIP engine required")
    if name == "ocr" and engine not in ("Qwen/Qwen3-VL-4B-Instruct", "tesseract", "paddleocr", "easyocr"):
        raise ValueError("unapproved OCR engine or upper-model evidence")
    if name == "ocr" and engine == "Qwen/Qwen3-VL-4B-Instruct" and stage["revision"] != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
        raise ValueError("OCR Qwen must be the pinned base model")


def ocr_scan(stage: dict, index: dict, items: dict, images: dict) -> list[dict]:
    benchmark, inputs = stage.get("benchmark"), stage.get("inputs")
    human = {sha for item in items.values() if item["lineage"].label_provenance == "HUMAN" for sha in item["image_sha256"]}
    if type(benchmark) is not dict or set(benchmark) != set(index["images"]) or type(inputs) is not dict or set(inputs) != human:
        raise ValueError("OCR exact benchmark/human-input coverage missing")
    entries = []
    for group in (benchmark, inputs):
        for sha, entry in group.items():
            if (type(entry) is not dict or entry.get("image_sha256") != sha
                    or not isinstance(entry.get("text"), str)
                    or entry.get("status") not in ("COMPLETE", "NO_TEXT")
                    or (entry["status"] == "NO_TEXT") != (not entry["text"].strip())):
                raise ValueError("OCR payload has wrong image SHA/status/text")
    for sha, entry in benchmark.items():
        entries.append({"text": entry["text"], "key": sha,
                        "ref": {"image_sha256": sha, "benchmark_refs": index["image_refs"][sha]}})
    cache, found = _text_cache(entries), []
    for pid, item in items.items():
        if item["lineage"].label_provenance == "PROGRAM" and not item["verifier"].printed_text:
            raise ValueError("PROGRAM printed_text annotation coverage missing")
        texts = ([inputs[sha]["text"] for sha in item["image_sha256"]] if item["lineage"].label_provenance == "HUMAN"
                 else ["\n".join(item["verifier"].printed_text)])
        for text in texts:
            for candidate in _text_matches(text, "ocr", cache):
                found.append({"problem_id": pid, "stage": "ocr_ngrams", **candidate})
    return found


def _vectors(entries: object, keys: set[str], bindings: dict[str, str]) -> tuple[list[str], list[list[float]]]:
    if type(entries) is not dict or set(entries) != keys:
        raise ValueError("embedding coverage must exactly match input/reference identities")
    ordered, vectors = sorted(keys), []
    for key in ordered:
        entry = entries[key]
        if (type(entry) is not dict or entry.get("sha256") != bindings[key]
                or type(entry.get("vector")) is not list or not entry["vector"]
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in entry["vector"])):
            raise ValueError("embedding payload has wrong SHA/vector")
        vectors.append(entry["vector"])
    return ordered, vectors


def embed_scan(stage: dict, kind: str, index: dict, items: dict, images: dict, thresholds: dict) -> list[dict]:
    reference_ids, reference = _vectors(stage.get("benchmark_images"), set(index["images"]),
                                      {sha: sha for sha in index["images"]})
    input_ids, inputs = _vectors(stage.get("input_images"), set(images), {sha: sha for sha in images})
    jobs = [(input_ids, inputs, reference_ids, reference, "image_to_image")]
    if kind == "siglip":
        text_refs = {entry["key"]: hashlib.sha256(normalize_text(entry["text"]).encode()).hexdigest()
                     for entry in index["text_cache"]["entries"] if normalize_text(entry["text"])}
        text_ids, text_vectors = _vectors(stage.get("benchmark_texts"), set(text_refs), text_refs)
        own_text = {pid: item["text_sha256"] for pid, item in items.items()}
        own_ids, own_vectors = _vectors(stage.get("input_texts"), set(own_text), own_text)
        if not text_ids:
            raise ValueError("SigLIP benchmark text coverage missing")
        jobs.extend([(input_ids, inputs, text_ids, text_vectors, "image_to_text"),
                     (own_ids, own_vectors, reference_ids, reference, "text_to_image"),
                     (own_ids, own_vectors, text_ids, text_vectors, "text_to_text")])
    found = []
    owners = {sha: [pid for pid, item in items.items() if sha in item["image_sha256"]] for sha in images}
    for query_ids, query_vectors, ref_ids, ref_vectors, pairing in jobs:
        if not ref_ids:
            raise ValueError("empty embedding reference corpus")
        for start in range(0, len(query_ids), 128):
            scores = cosine_similarities(query_vectors[start:start + 128], ref_vectors)
            for offset, row in enumerate(scores):
                target = int(np.argmax(row))
                score = float(row[target])
                cutoff = 0.75 if kind == "sscd" else thresholds["siglip_review"]
                if score >= cutoff:
                    key = query_ids[start + offset]
                    for pid in owners[key] if pairing.startswith("image_") else [key]:
                        found.append({"problem_id": pid, "stage": kind, "reason": f"{kind.upper()}_{pairing.upper()}",
                                      "action": "QUARANTINE" if kind == "sscd" and score >= 0.95 else "REVIEW",
                                      "ref": ref_ids[target], "cosine": score})
    return found


def scan(split_roots: dict[str, Path], image_root: Path, index: dict, source_registry: dict,
         evidence: dict | None = None) -> dict:
    """Collect real candidates even in blocked/partial scans; never silently skip a stage."""
    evidence = {} if evidence is None else evidence
    result: dict = {"version": VERSION, "status": "BLOCKED_NOT_CHECKED", "training_allowed": False,
              "approval_granted": False, "allowlist": [], "candidates": [], "quarantined_ids": [],
              "blockers": [], "limitations": ["Does not certify absence of model pretraining exposure or all semantic paraphrases",
                  "Precomputed engine/calibration declarations are bound and checked, not independently rerun or authenticated"],
              "steps": {name: {"status": "BLOCKED_NOT_CHECKED", "candidate_count": 0} for name in _STEPS}}
    blockers, candidates = result["blockers"], result["candidates"]
    try:
        data = index["data"] if "data" in index else index
        partial = index.get("partial_name") if "data" in index else None
        prepared = _prepare(json.loads(_json(data)), partial)
        index_sha = prepared["sha256"]
        if "path" in index and hashlib.sha256(vault._read(Path(index["path"]))).hexdigest() != index["file_sha256"]:
            raise ValueError("index file changed since load_index")
        code_files = [Path(__file__), _INVENTORY, Path(audit.__file__), _HERE / "vault.py", _HERE / "schema.py", _HERE / "protocol_v1.yaml"]
        code_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files}
        protocol_sha = load_protocol(_HERE / "protocol_v1.yaml")["sha256"]
        registry_sha, evidence_sha = _digest(source_registry), _digest(evidence)
        items, files, images = _snapshot(split_roots, image_root)
        result.update(index_sha256=index_sha, index_file_sha256=index.get("file_sha256"), partial_name=partial,
                      source_registry_sha256=registry_sha, evidence_sha256=evidence_sha, protocol_sha256=protocol_sha,
                      code_sha256=code_hashes, input_files=files, input_files_sha256=_digest(files),
                      input_count=len(items), split_counts=dict(Counter(item["split"] for item in items.values())))
        result["limitations"].extend(prepared["limitations"])
        if not prepared["full_coverage"]:
            blockers.append("FULL_BENCHMARK_INDEX_NOT_AVAILABLE")
        if set(split_roots) != _SPLITS:
            blockers.append("ALL_FOUR_SPLITS_REQUIRED")
        if any(not result["split_counts"].get(split) for split in _SPLITS):
            blockers.append("ALL_FOUR_SPLITS_NONEMPTY_REQUIRED")
        if not items:
            blockers.append("NO_INPUT_RECORDS")
        for pid, item in items.items():
            try:
                for reason in _source_checks(item["actor"], item["lineage"], source_registry):
                    candidates.append({"problem_id": pid, "stage": "source_rights", "reason": reason, "action": "QUARANTINE"})
            except ValueError as exc:
                blockers.append(f"SOURCE_PREREQUISITE:{pid}:{exc}")
        for name in ("source_rights", "lineage_origin"):
            if not any(b.startswith("SOURCE_PREREQUISITE") for b in blockers):
                result["steps"][name]["status"] = "COMPLETE"
        shared: dict[str, list[str]] = defaultdict(list)
        for pid, item in items.items():
            for token in ["root:" + item["structure_hash"], *("image:" + sha for sha in item["image_sha256"]),
                          *("pixel:" + sha for sha in item["pixel_sha256"])]:
                shared[token].append(pid)
        for pids in shared.values():
            if len({items[pid]["split"] for pid in pids}) > 1:
                candidates.extend({"problem_id": pid, "stage": "family", "reason": "INPUT_FAMILY_SPLIT_LEAK",
                                   "action": "QUARANTINE"} for pid in pids)
        for pid, item in items.items():
            for field, text in item["fields"]:
                for candidate in _text_matches(text, field, prepared["text_cache"]):
                    candidates.append({"problem_id": pid, "stage": "question_exact" if candidate["reason"] == "EXACT_TEXT" else "text_ngrams",
                                       "field": field, **candidate})
            for sha in item["image_sha256"]:
                fp = images[sha]
                if sha in prepared["raw_hashes"] or fp["pixels_sha256"] in prepared["pixel_hashes"]:
                    matches = [key for key, value in prepared["images"].items()
                               if key == sha or value["pixels_sha256"] == fp["pixels_sha256"]]
                    candidates.append({"problem_id": pid, "stage": "image_exact", "reason": "EXACT_RAW_OR_PIXEL_IMAGE",
                                       "action": "QUARANTINE", "ref": {key: prepared["image_refs"][key] for key in matches}})
                for variant, hashes in fp["variants"].items():
                    for kind, tree in prepared["trees"].items():
                        matches = tree.search(int(hashes[kind], 16), 3)
                        for action in ("QUARANTINE", "REVIEW"):
                            options = [(distance, ref) for distance, ref in matches
                                       if ("REVIEW" if hashes["low_information"] or _low_information(hashes[kind]) or ref["low_information"] else "QUARANTINE") == action]
                            if options:
                                distance, ref = min(options, key=lambda pair: pair[0])
                                candidates.append({"problem_id": pid, "stage": "perceptual", "reason": f"{kind.upper()}_CROP_TILE_NEAR",
                                                   "action": action, "input_variant": variant, "hamming": distance, "ref": ref})
        for name in ("question_exact", "text_ngrams", "image_exact"):
            result["steps"][name]["status"] = "COMPLETE" if not prepared["limitations"] else "PARTIAL_CHECKED"
        thresholds = None
        try:
            thresholds = _calibration(evidence, source_registry, index_sha, protocol_sha)
            result["steps"]["perceptual"]["status"] = "COMPLETE" if prepared["images"] else "PARTIAL_CHECKED"
        except ValueError as exc:
            blockers.append(f"CALIBRATION_PREREQUISITE:{exc}")
        for name in ("ocr", "sscd", "siglip"):
            step = "ocr_ngrams" if name == "ocr" else name
            try:
                if thresholds is None or not prepared["images"]:
                    raise ValueError("v2 image index and frozen calibration required")
                stage = evidence.get(name)
                if type(stage) is not dict:
                    raise ValueError("complete engine evidence required")
                _engine(stage, name, evidence, source_registry, index_sha)
                hits = ocr_scan(stage, prepared, items, images) if name == "ocr" else embed_scan(stage, name, prepared, items, images, thresholds)
                candidates.extend(hits)
                result["steps"][step]["status"] = "COMPLETE"
            except (KeyError, TypeError, ValueError) as exc:
                blockers.append(f"{name.upper()}_PREREQUISITE:{exc}")
        family = family_quarantine(items, candidates)
        result.update(quarantined_ids=family["quarantined_ids"], quarantined_families=family["families"])
        result["steps"]["family"]["status"] = "COMPLETE"
        result["steps"]["all_splits"]["status"] = ("COMPLETE" if set(split_roots) == _SPLITS
            and all(result["split_counts"].get(split) for split in _SPLITS) else "PARTIAL_CHECKED")
        if (any(hashlib.sha256(vault._read(Path(path))).hexdigest() != sha for path, sha in files.items())
                or any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in code_hashes.items())
                or _digest(source_registry) != registry_sha or _digest(evidence) != evidence_sha
                or _digest(data) != index_sha
                or ("path" in index and hashlib.sha256(vault._read(Path(index["path"]))).hexdigest() != index["file_sha256"])):
            raise ValueError("inputs/index/code/evidence changed during scan")
        result["steps"]["frozen_manifest"]["status"] = "COMPLETE"
        if not blockers and all(step["status"] == "COMPLETE" for step in result["steps"].values()):
            result["status"], result["allowlist"] = "EXCLUSION_CHECKED_WITH_LIMITS", family["allowlist"]
    except (AttributeError, KeyError, OSError, TypeError, ValueError, Image.DecompressionBombError) as exc:
        blockers.append(f"INPUT_OR_INDEX_PREREQUISITE:{type(exc).__name__}:{exc}")
    for name, step in result["steps"].items():
        step["candidate_count"] = sum(c["stage"] == name for c in candidates)
    result["candidate_record_count"] = len({c["problem_id"] for c in candidates})
    result["blockers"] = sorted(set(blockers))
    result["candidate_scan_status"] = ("CANDIDATE_SCAN_COMPLETE_NOT_APPROVAL" if
        all(result["steps"][s]["status"] in ("COMPLETE", "PARTIAL_CHECKED") for s in ("question_exact", "text_ngrams", "image_exact", "family"))
        else "BLOCKED_NOT_CHECKED")
    if result["status"] != "EXCLUSION_CHECKED_WITH_LIMITS":
        result["allowlist"] = []
    result["receipt_sha256"] = _digest(result)
    return result


def write_receipt(path: Path, receipt: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _digest(body):
        raise ValueError("exclusion: receipt payload hash mismatch")
    vault._safe_root(Path(path).parent, create=True)
    raw = _json(receipt) + b"\n"
    with Path(path).open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
    return {"path": str(Path(path).absolute()), "sha256": hashlib.sha256(raw).hexdigest()}
