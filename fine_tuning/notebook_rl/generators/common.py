"""Deterministic, unapproved generated records; no filesystem or model writes."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ..schema import ActorItem, AnswerType, Lineage, Mode, Split, VerifierItem, to_json


@dataclass(frozen=True)
class GeneratedProblem:
    actor: ActorItem
    verifier: VerifierItem
    lineage: Lineage
    image_bytes: tuple[bytes, ...]


def structure_hash(obj: Any) -> str:
    def check(value: Any) -> None:
        if type(value) is dict:
            if any(type(k) is not str for k in value):
                raise ValueError("structure: keys must be strings")
            for child in value.values():
                check(child)
        elif type(value) in (list, tuple):
            for child in value:
                check(child)
        elif type(value) not in (str, int, float, bool, type(None)):
            raise ValueError("structure: unsupported JSON value")
    check(obj)
    try:
        raw = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("structure: finite JSON required") from exc
    return hashlib.sha256(raw).hexdigest()[:16]


def png_chunks(raw: bytes) -> tuple[bytes, ...]:
    """Check chunk bounds and CRCs as well as the metadata allowlist."""
    if type(raw) is not bytes or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("image: PNG bytes required")
    at, kinds = 8, []
    while at < len(raw):
        if at + 12 > len(raw):
            raise ValueError("image: truncated PNG chunk")
        length = struct.unpack(">I", raw[at:at + 4])[0]
        end = at + 12 + length
        if end > len(raw):
            raise ValueError("image: truncated PNG data")
        kind = raw[at + 4:at + 8]
        if kind not in (b"IHDR", b"PLTE", b"IDAT", b"IEND"):
            raise ValueError(f"image: PNG metadata or unsupported chunk {kind!r}")
        crc = struct.unpack(">I", raw[end - 4:end])[0]
        if zlib.crc32(raw[at + 4:end - 4]) & 0xffffffff != crc:
            raise ValueError("image: invalid PNG CRC")
        kinds.append(kind)
        at = end
        if kind == b"IEND":
            if length or at != len(raw):
                raise ValueError("image: data after PNG end")
            break
    if not kinds or kinds[0] != b"IHDR" or kinds[-1] != b"IEND" or b"IDAT" not in kinds:
        raise ValueError("image: incomplete PNG")
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    return tuple(kinds)


def png_bytes(fig_or_image: Any, *, dpi: int) -> bytes:
    if type(dpi) is not int or dpi <= 0:
        raise ValueError("render: dpi must be a positive integer")
    source = io.BytesIO()
    if isinstance(fig_or_image, Image.Image):
        picture = fig_or_image
    else:
        from matplotlib.figure import Figure
        if not isinstance(fig_or_image, Figure):
            raise ValueError("render: expected Pillow image or matplotlib Figure")
        fig_or_image.savefig(source, format="png", dpi=dpi, metadata={})
        picture = Image.open(io.BytesIO(source.getvalue()))
    try:
        mode = "RGBA" if "A" in picture.getbands() or "transparency" in picture.info else "RGB"
        with picture.convert(mode) as pixels:
            pixels.info.clear()
            out = io.BytesIO()
            pixels.save(out, format="PNG", optimize=False, compress_level=9)
    finally:
        if picture is not fig_or_image:
            picture.close()
    raw = out.getvalue()
    png_chunks(raw)
    return raw


def split_of(structure_hash: str, ratios=(0.8, 0.1, 0.1)) -> Split:
    if not isinstance(structure_hash, str) or not re.fullmatch(r"[0-9a-f]{16}", structure_hash):
        raise ValueError("split: expected 16 lowercase hexadecimal characters")
    if (len(ratios) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                                for x in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-12)):
        raise ValueError("split: three positive ratios summing to one required")
    value = int(structure_hash[:8], 16) / 2**32
    return "train" if value < ratios[0] else "dev" if value < sum(ratios[:2]) else "sealed"


def matches_split(digest: str, split: Split, style: str = "default") -> bool:
    """Reserve half the sealed hash band for unseen shifted-style structures."""
    if split not in ("train", "dev", "sealed", "style_sealed"):
        raise ValueError("split: unsupported split")
    if style not in ("default", "shift") or (style == "shift") != (split == "style_sealed"):
        raise ValueError("split: shift is exclusive to style_sealed")
    base = split_of(digest)
    if base != "sealed":
        return split == base
    held = "style_sealed" if int(digest[8:], 16) % 2 else "sealed"
    return split == held


def build(source: str, family: str, index: int, split: Split, mode: Mode, question: str,
          choices, answer_type: AnswerType, gold: str, accept: dict, slots: dict[str, str],
          slot_types: dict[str, str], required_slots, oracle_fact_lines, printed_text,
          structure, template_id: str, image_bytes, generator_version: str, seed: int,
          style: str) -> GeneratedProblem:
    if not all(isinstance(x, str) and re.fullmatch(r"[a-zA-Z0-9_.-]+", x)
               for x in (source, family, generator_version, template_id)):
        raise ValueError("build: unsafe or empty identifier")
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("build: nonnegative integer index and seed required")
    digest = structure_hash(structure)
    if not matches_split(digest, split, style):
        raise ValueError("build: structure belongs to another split")
    images = tuple(image_bytes)
    if not 1 <= len(images) <= 2:
        raise ValueError("build: one or two PNG images required")
    for image in images:
        png_chunks(image)
    pid = f"{source}:{family}:{split}:{seed}:{index:06d}:{digest}"
    paths = tuple(f"{source}/{split}/{seed}-{index:06d}-{digest}-{i}.png"
                  for i in range(len(images)))
    actor = ActorItem(pid, source, family, mode, question, tuple(choices), answer_type,
                      paths, tuple(hashlib.sha256(x).hexdigest() for x in images))
    verifier = VerifierItem(pid, answer_type, gold, dict(accept), tuple(choices), dict(slots),
                            dict(slot_types), tuple(required_slots), tuple(oracle_fact_lines),
                            tuple(printed_text))
    lineage = Lineage(pid, source, generator_version, "project-authored", "PROGRAM",
                      f"rendered:{generator_version}:{style}", "procedural", generator_version,
                      seed, template_id, digest, split, ())
    for record in (actor, verifier, lineage):
        to_json(record)
    return GeneratedProblem(actor, verifier, lineage, images)
