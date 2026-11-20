"""Stdlib helpers shared by the Pod scripts; no network, no GPU, no third-party imports."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import IO, Any

from fine_tuning.budget import utc

parse_utc = utc  # tz-aware ISO parser, reused from the watchdog
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 8 << 20
QUICK_SPAN = 64 << 20  # --quick hashes the first and last 64 MiB only


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | os.PathLike[str]) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digests(path: str | os.PathLike[str], span: int = QUICK_SPAN,
                 chunk: int = CHUNK) -> tuple[int, str, str]:
    """One streaming pass -> (size, full sha256, quick sha256).

    quick = sha256(first min(size, span) bytes + last min(size - head, span) bytes), so for
    files up to `span` bytes it equals the full digest. Never report quick as a full check.
    """
    size = os.stat(path).st_size
    head_end = min(size, span)
    tail_start = max(head_end, size - span)
    full, quick = hashlib.sha256(), hashlib.sha256()
    offset = 0
    with open(path, "rb") as stream:
        while data := stream.read(chunk):
            full.update(data)
            end = offset + len(data)
            for low, high in ((0, head_end), (tail_start, size)):
                start, stop = max(low, offset), min(high, end)
                if start < stop:
                    quick.update(data[start - offset:stop - offset])
            offset = end
    if offset != size:
        raise ValueError(f"file changed while hashing: {path}")
    return size, full.hexdigest(), quick.hexdigest()


def quick_digest(path: str | os.PathLike[str], span: int = QUICK_SPAN,
                 chunk: int = CHUNK) -> tuple[int, str]:
    """Seek-based quick digest; same value as file_digests()[2] without reading the middle."""
    size = os.stat(path).st_size
    head_end = min(size, span)
    tail_start = max(head_end, size - span)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for start, stop in ((0, head_end), (tail_start, size)):
            stream.seek(start)
            remaining = stop - start
            while remaining > 0:
                data = stream.read(min(chunk, remaining))
                if not data:
                    raise ValueError(f"file shrank while hashing: {path}")
                digest.update(data)
                remaining -= len(data)
    return size, digest.hexdigest()


class HashingReader:
    """File wrapper that hashes exactly the bytes a reader (tarfile, copy loop) takes from it."""

    def __init__(self, stream: IO[bytes]) -> None:
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomic replace; allow_nan=False so a receipt can never hold NaN/inf."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + target.name, dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def process_startticks(pid: int) -> int | None:
    """/proc start time (field 22); with the PID it identifies a process despite PID reuse."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        return int(text.rsplit(")", 1)[1].split()[19])
    except (IndexError, ValueError):
        return None
