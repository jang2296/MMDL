"""Read one pinned ZIP member without transferring the full author archive."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
import tempfile
import threading
import http.client
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path


ATTEMPTS = 5
TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)


def _retryable(error: BaseException, attempt: int) -> bool:
    return attempt < ATTEMPTS - 1 and not (isinstance(error, urllib.error.HTTPError)
                                           and error.code not in (429, 500, 502, 503, 504))


def _request(request, meter):
    meter["requests"] += 1
    return urllib.request.urlopen(request, timeout=45)


def _range(spec: dict, start: int, length: int, meter: dict) -> bytes:
    end = start + length - 1
    total = spec["archive_bytes"]
    if start < 0 or not 0 < length <= 16 * 1024**2 or end >= total:
        raise ValueError("out-of-bounds ZIP member range")
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
               "If-Match": spec["etag"]}
    request = urllib.request.Request(spec["url"], headers=headers)
    for attempt in range(ATTEMPTS):
        try:
            with _request(request, meter) as response:
                if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"
                        or response.headers.get("ETag") != spec["etag"]
                        or response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise ValueError("source did not return the exact pinned byte range")
                data = response.read(length + 1)
                meter["received_bytes"] += len(data)
            break
        except TRANSIENT as error:  # a body read that times out or resets is refetched like a failed open
            if not _retryable(error, attempt):
                raise
            time.sleep(2 ** attempt)
    if len(data) != length:
        raise ValueError("incomplete ZIP member range")
    return data


class _ArchiveReader(io.RawIOBase):
    def __init__(self, spec, meter):
        self.spec, self.meter, self.pos = spec, meter, 0

    def seek(self, offset, whence=0):
        position = (0, self.pos, self.spec["archive_bytes"])[whence] + offset
        if position < 0:
            raise OSError("negative archive position")
        self.pos = position
        return position

    def tell(self):
        return self.pos

    def read(self, n=-1):
        remaining = self.spec["archive_bytes"] - self.pos
        n = remaining if n < 0 else min(n, remaining)
        if n > 64 * 1024**2:
            raise ValueError("remote ZIP directory exceeds metadata limit")
        parts = []
        while n > 0:
            chunk = min(n, 8 * 1024**2)
            parts.append(_range(self.spec, self.pos, chunk, self.meter))
            self.pos += chunk
            n -= chunk
        return b"".join(parts)


_directories: dict = {}
_directory_lock = threading.Lock()


def _locate(spec, meter):
    if "offset" in spec:
        return spec
    url = spec["url"]
    with _directory_lock:
        if url not in _directories:
            for attempt in range(ATTEMPTS):
                try:
                    with _request(urllib.request.Request(url, method="HEAD"), meter) as response:
                        headers = {"url": url, "archive_bytes": int(response.headers["Content-Length"]),
                                   "etag": response.headers["ETag"]}
                    break
                except TRANSIENT as error:
                    if not _retryable(error, attempt):
                        raise
                    time.sleep(2 ** attempt)
            if not headers["etag"]:
                raise ValueError("missing remote ZIP identity")
            reader = _ArchiveReader(headers, meter)
            with zipfile.ZipFile(reader) as archive:
                entries = {i.filename: i for i in archive.infolist()}
                if len(entries) != len(archive.infolist()):
                    raise ValueError("ambiguous ZIP member names")
            _directories[url] = headers, entries
        headers, entries = _directories[url]
    info = entries[spec["member"]]
    if info.file_size != spec["bytes"]:
        raise ValueError("pinned ZIP member size differs from directory")
    return {**spec, **headers, "offset": info.header_offset, "compression": info.compress_type,
            "compressed_bytes": info.compress_size, "crc32": info.CRC}


def fetch_member(spec: dict, cache_dir: Path) -> Path:
    member_sha = spec["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", member_sha):
        raise ValueError("invalid member digest")
    path = Path(cache_dir) / "objects" / member_sha
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == member_sha:
        return path
    meter = {"requests": 0, "received_bytes": 0}
    spec = _locate(spec, meter)
    name = spec["member"].encode("utf-8")
    header = _range(spec, spec["offset"], 30, meter)
    signature, _, flags, compression, _, _, _, _, _, nlen, elen = struct.unpack("<4s5H3I2H", header)
    if signature != b"PK\x03\x04" or flags & 1 or nlen != len(name) or compression != spec["compression"]:
        raise ValueError("foreign or encrypted ZIP header")
    body = _range(spec, spec["offset"] + 30, nlen + elen + spec["compressed_bytes"], meter)
    if body[:nlen] != name:
        raise ValueError("ZIP member locator changed")
    packed = body[nlen + elen:]
    if not 0 < spec["bytes"] <= 16 * 1024**2:
        raise ValueError("oversized ZIP image")
    if compression == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(packed, spec["bytes"] + 1)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("incomplete or oversized compressed member")
    elif compression == zipfile.ZIP_STORED:
        raw = packed
    else:
        raise ValueError("unsupported ZIP compression")
    if (len(raw) != spec["bytes"] or zlib.crc32(raw) != spec["crc32"]
            or hashlib.sha256(raw).hexdigest() != member_sha):
        raise ValueError("ZIP member size, CRC or digest mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            tmp.write(raw)
            tmp.flush()
            metrics = path.parent.parent / "range-metrics"
            metrics.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=metrics, delete=False) as receipt:
                json.dump(meter, receipt)
            os.replace(receipt.name, metrics / (member_sha + ".json"))
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return path
