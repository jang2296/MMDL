"""Resumable private OCR/SSCD/SigLIP caches from fixed local artifacts only.

Cache presence is not complete benchmark coverage, calibration, or approval.
No download, training, quantization, or source-image resizing fallback is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from fine_tuning import audit

from . import contamination as c
from . import vault
from .build_exclusion_index import _private_root, _write_once

SIGLIP_ID = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
# Exact official dl.fbaipublicfiles.com/sscd-copy-detection TorchScript artifact.
SSCD_SHA256 = "9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _image(path: Path, expected: str) -> Image.Image:
    if not path.is_absolute():
        raise ValueError("engine cache: absolute image paths required")
    raw = vault._read(path)
    if _sha(raw) != expected:
        raise ValueError("engine cache: source image SHA256 changed")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return image.convert("RGB")


def _identity(name: str, revision: str, artifacts: dict[str, Path], preprocessing: dict,
              packages: tuple[str, ...] = ()) -> dict:
    hashes = {key: audit.sha(path.resolve(strict=True)) for key, path in sorted(artifacts.items())}
    return {"engine_id": name, "revision": revision, "artifact_sha256": c._digest(hashes),
            "artifact_files": hashes, "preprocessing": preprocessing,
            "producer_sha256": audit.sha(__file__), "normalizer_sha256": audit.sha(c.__file__),
            "packages": {name: importlib.metadata.version(name) for name in ("Pillow", *packages)}}


class OCR:
    def __init__(self, extracted: Path, timeout: float = 60):
        self.root = Path(extracted).resolve(strict=True)
        self.binary = self.root / "usr/bin/tesseract"
        self.tessdata = self.root / "usr/share/tesseract-ocr/4.00/tessdata"
        if not 0 < timeout <= 300:
            raise ValueError("engine cache: OCR timeout must be in (0, 300] seconds")
        self.timeout = timeout
        self.env = {**os.environ, "OMP_THREAD_LIMIT": "1", "OMP_NUM_THREADS": "1",
                    "LD_LIBRARY_PATH": str(self.root / "usr/lib/x86_64-linux-gnu")}
        version = subprocess.run([str(self.binary), "--version"], env=self.env,
                                 capture_output=True, timeout=10, check=True)
        if not (version.stdout + version.stderr).decode().startswith("tesseract 4.1.1"):
            raise ValueError("engine cache: exact tesseract 4.1.1 required")
        artifacts = {"tesseract": self.binary, "eng.traineddata": self.tessdata / "eng.traineddata"}
        artifacts.update({str(p.relative_to(self.root)): p
                          for p in (self.root / "usr/lib/x86_64-linux-gnu").glob("*.so*") if p.is_file()})
        self.identity = _identity("tesseract", "4.1.1", artifacts,
                                  {"image": "Pillow RGB lossless PNG stdin; original dimensions",
                                   "language": "eng", "psm": 3, "omp_threads": 1})

    def recognize(self, image: Image.Image) -> dict:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        process = subprocess.run([str(self.binary), "stdin", "stdout", "--tessdata-dir",
                                  str(self.tessdata), "-l", "eng", "--psm", "3"],
                                 input=stream.getvalue(), capture_output=True, timeout=self.timeout,
                                 env=self.env, check=True)
        text = process.stdout.decode("utf-8", errors="strict")
        return {"text": text, "status": "COMPLETE" if text.strip() else "NO_TEXT"}


def _torch_setup(device: str):
    import torch

    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("engine cache: requested device unavailable; no automatic fallback")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def _unit_vectors(tensor, dimension: int):
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    if value.ndim != 2 or value.shape[1] != dimension or not torch.isfinite(value).all():
        raise ValueError("engine cache: invalid embedding dimensions/values")
    norms = value.norm(dim=1, keepdim=True)
    if (norms <= 0).any():
        raise ValueError("engine cache: zero embedding")
    return value / norms


class SSCD:
    def __init__(self, artifact: Path, device: str = "cpu"):
        import torchvision.transforms as transforms

        self.torch = _torch_setup(device)
        self.device = device
        artifact = Path(artifact).resolve(strict=True)
        if audit.sha(artifact) != SSCD_SHA256:
            raise ValueError("engine cache: official SSCD disc_mixup artifact SHA256 differs")
        self.identity = _identity("sscd_disc_mixup", "sscd-disc-mixup-torchscript-v1",
                                  {"sscd_disc_mixup.torchscript.pt": artifact},
                                  {"image": "RGB skew320 bilinear antialias; ToTensor ImageNet normalization",
                                   "mean": [.485, .456, .406], "std": [.229, .224, .225],
                                   "dtype": "float32", "device": device, "dimensions": 512,
                                   "tf32": False}, ("torch", "torchvision"))
        self.model = self.torch.jit.load(str(artifact), map_location=device).float().eval()
        self.transform = transforms.Compose([transforms.Resize([320, 320], antialias=True),
                                              transforms.ToTensor(),
                                              transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def image_vectors(self, images: list[Image.Image]) -> list[dict]:
        batch = self.torch.stack([self.transform(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            vectors = _unit_vectors(self.model(batch), 512)
        return [{"vector": vector.tolist()} for vector in vectors]


def text_chunks(tokenizer, text: str) -> tuple[list[list[int]], list[int]]:
    """Keep every encoded token; SigLIP EOS doubles as the final padding token."""
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if type(eos) is not int or type(pad) is not int:
        raise ValueError("engine cache: explicit EOS/pad token IDs required")
    if not ids:
        # SentencePiece can normalize punctuation-only fields to no content tokens.
        return [[eos] + [pad] * 63], [0]
    chunks, lengths = [], []
    for start in range(0, len(ids), 63):
        part = ids[start:start + 63]
        lengths.append(len(part))
        chunks.append([*part, eos, *([pad] * (63 - len(part)))])
    return chunks, lengths


class SigLIP:
    def __init__(self, snapshot: Path, device: str = "cpu"):
        from transformers import AutoModel, AutoProcessor

        self.torch = _torch_setup(device)
        self.device = device
        snapshot = Path(snapshot).absolute()
        if snapshot.name != SIGLIP_REVISION:
            raise ValueError("engine cache: exact pinned SigLIP snapshot directory required")
        names = ("config.json", "preprocessor_config.json", "special_tokens_map.json", "spiece.model",
                 "tokenizer.json", "tokenizer_config.json", "model.safetensors")
        self.identity = _identity(SIGLIP_ID, SIGLIP_REVISION,
                                  {name: snapshot / name for name in names},
                                  {"image": "official AutoProcessor RGB 224x224",
                                   "text": "normalized full encode; 63 tokens + EOS/pad64; unit chunk mean then unit",
                                   "dtype": "float32", "device": device, "dimensions": 768,
                                   "tf32": False, "attention": "eager; official unmasked EOS padding"},
                                  ("torch", "transformers", "tokenizers", "sentencepiece", "protobuf"))
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True, torch_dtype=self.torch.float32,
                                               attn_implementation="eager").to(device).float().eval()
        config = self.processor.image_processor
        if config.size != {"height": 224, "width": 224} or self.model.config.text_config.max_position_embeddings != 64:
            raise ValueError("engine cache: pinned SigLIP image/context configuration differs")

    def image_vectors(self, images: list[Image.Image]) -> list[dict]:
        batch = self.processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        if batch.dtype != self.torch.float32 or tuple(batch.shape[1:]) != (3, 224, 224):
            raise ValueError("engine cache: SigLIP processor tensor differs from float32 224 square")
        with self.torch.inference_mode():
            vectors = _unit_vectors(self.model.get_image_features(pixel_values=batch), 768)
        return [{"vector": vector.tolist()} for vector in vectors]

    def text_vectors(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        encoded = [text_chunks(self.processor.tokenizer, text) for text in texts]
        flat = [chunk for chunks, _ in encoded for chunk in chunks]
        vectors = []
        for start in range(0, len(flat), 64):
            ids = self.torch.tensor(flat[start:start + 64], dtype=self.torch.long, device=self.device)
            with self.torch.inference_mode():
                vectors.append(_unit_vectors(self.model.get_text_features(input_ids=ids), 768))
        vectors = self.torch.cat(vectors)
        result = []
        offset = 0
        for chunks, lengths in encoded:
            mean = vectors[offset:offset + len(chunks)].mean(dim=0, keepdim=True)
            offset += len(chunks)
            vector = _unit_vectors(mean, 768)[0].tolist()
            result.append({"vector": vector, "chunks": len(chunks), "token_count": sum(lengths),
                           "chunk_token_counts": lengths, "encoded_chunks_sha256": c._digest(chunks)})
        return result


def load_manifest(path: Path, *, texts: bool = False) -> dict[str, str]:
    values = json.loads(vault._read(Path(path)), object_pairs_hook=vault._unique)
    if (type(values) is not dict or not values
            or any(not isinstance(k, str) or not _SHA.fullmatch(k) or not isinstance(v, str)
                   for k, v in values.items())):
        raise ValueError("engine cache: nonempty SHA256 to path/text manifest required")
    if texts:
        values = {key: c.normalize_text(text) for key, text in values.items()}
        if any(not text or _sha(text.encode()) != key for key, text in values.items()):
            raise ValueError("engine cache: normalized source text SHA256 differs")
    elif any(not Path(value).is_absolute() for value in values.values()):
        raise ValueError("engine cache: absolute image manifest paths required")
    return values


def _entry(path: Path, engine_sha: str, source_sha: str, kind: str) -> dict | None:
    if not path.exists() and not path.is_symlink():
        return None
    value = json.loads(vault._read(path), object_pairs_hook=vault._unique)
    if (value.get("engine_sha256") != engine_sha or value.get("source_sha256") != source_sha
            or value.get("source_kind") != kind or value.get("payload_sha256") != c._digest(value.get("payload"))):
        raise ValueError("engine cache: stale/tampered existing entry")
    return value["payload"]


def produce(engine, sources: dict[str, str], output_root: Path, *, texts: bool = False,
            max_items: int | None = None, batch_size: int = 8, workers: int = 2) -> dict:
    if (not sources or any(not isinstance(k, str) or not _SHA.fullmatch(k) or not isinstance(v, str)
                           for k, v in sources.items()) or not 1 <= batch_size <= 64 or not 1 <= workers <= 4
            or max_items is not None and (type(max_items) is not int or max_items <= 0)):
        raise ValueError("engine cache: invalid sources/work bounds")
    identity = engine.identity
    if identity["engine_id"] not in ("tesseract", "sscd_disc_mixup", SIGLIP_ID):
        raise ValueError("engine cache: unsupported engine identity")
    name = "ocr" if identity["engine_id"] == "tesseract" else "sscd" if identity["engine_id"] == "sscd_disc_mixup" else "siglip"
    engine_sha = c._digest(identity)
    kind = "text" if texts else "image"
    root = _private_root(Path(output_root), create=True) / name / engine_sha
    for directory in (root, root / "images", root / "texts", root / "failures"):
        vault._safe_root(directory, create=True)
    _write_once(root / "identity.json", c._json(identity))
    keys = sorted(sources)[:max_items]
    pending, reused = [], 0
    for sha in keys:
        if texts:
            text = c.normalize_text(sources[sha])
            if not text or _sha(text.encode()) != sha:
                raise ValueError("engine cache: normalized source text SHA256 differs")
        else:
            if not Path(sources[sha]).is_absolute():
                raise ValueError("engine cache: absolute image paths required")
            if _sha(vault._read(Path(sources[sha]))) != sha:
                raise ValueError("engine cache: source image SHA256 changed")
        entry_path = root / ("texts" if texts else "images") / (sha + ".json")
        if _entry(entry_path, engine_sha, sha, kind) is not None:
            reused += 1
        else:
            pending.append(sha)
    failed, created = 0, 0

    def commit(sha, payload=None, failure=None):
        nonlocal failed, created
        if failure is not None:
            failed += 1
            _write_once(root / "failures" / (sha + "-" + uuid.uuid4().hex + ".json"),
                        c._json({"source_sha256": sha, "engine_sha256": engine_sha,
                                 "status": "FAILED", "error_type": type(failure).__name__}))
            return
        payload = {"sha256": sha, **payload}
        if not texts:
            payload["image_sha256"] = sha
        _write_once(root / ("texts" if texts else "images") / (sha + ".json"),
                    c._json({"engine_sha256": engine_sha, "source_sha256": sha, "source_kind": kind,
                             "payload": payload, "payload_sha256": c._digest(payload)}))
        created += 1

    if name == "ocr":
        if texts:
            raise ValueError("engine cache: OCR accepts images only")

        def recognize(sha):
            try:
                with _image(Path(sources[sha]), sha) as image:
                    return sha, engine.recognize(image), None
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                return sha, None, exc

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sha, payload, failure in pool.map(recognize, pending):
                commit(sha, payload, failure)
    else:
        for start in range(0, len(pending), batch_size):
            group = pending[start:start + batch_size]
            images = []
            try:
                if texts:
                    payloads = engine.text_vectors([c.normalize_text(sources[sha]) for sha in group])
                else:
                    images = [_image(Path(sources[sha]), sha) for sha in group]
                    payloads = engine.image_vectors(images)
                if len(payloads) != len(group):
                    raise ValueError("engine cache: incomplete model batch output")
                for sha, payload in zip(group, payloads, strict=True):
                    commit(sha, payload)
            finally:
                for image in images:
                    image.close()
    return {"status": "PARTIAL_CACHE_ONLY", "cache_root": str(root), "engine_sha256": engine_sha,
            "requested_entries": len(sources), "selected_entries": len(keys), "new_entries": created,
            "reused_entries": reused, "failed_entries": failed, "training_allowed": False,
            "full_coverage_claimed": False, "calibration_produced": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ocr", "sscd", "siglip-images", "siglip-texts"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-manifest", type=Path)
    source.add_argument("--image-root", type=Path)
    source.add_argument("--text-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tesseract-root", type=Path)
    parser.add_argument("--sscd-artifact", type=Path)
    parser.add_argument("--siglip-snapshot", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    texts = args.stage == "siglip-texts"
    if texts != bool(args.text_manifest):
        parser.error("siglip-texts requires --text-manifest; image stages require image sources")
    if args.image_root:
        root = vault._safe_root(args.image_root)
        sources = {p.name: str(p) for p in root.iterdir() if _SHA.fullmatch(p.name)}
    else:
        sources = load_manifest(args.text_manifest if texts else args.image_manifest, texts=texts)
    engine: OCR | SSCD | SigLIP
    if args.stage == "ocr":
        if not args.tesseract_root:
            parser.error("--tesseract-root required")
        engine = OCR(args.tesseract_root, timeout=args.timeout)
    elif args.stage == "sscd":
        if not args.sscd_artifact:
            parser.error("--sscd-artifact required")
        engine = SSCD(args.sscd_artifact, device=args.device)
    else:
        if not args.siglip_snapshot:
            parser.error("--siglip-snapshot required")
        engine = SigLIP(args.siglip_snapshot, device=args.device)
    result = produce(engine, sources, args.output_root, texts=texts, max_items=args.max_items,
                     batch_size=args.batch_size, workers=args.workers)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["failed_entries"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
