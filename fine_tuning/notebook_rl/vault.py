"""Write-once split records; actor reads never open verifier or lineage payloads.

Hashes detect corruption against the manifest, not malicious replacement of both.
Use separate process/OS permissions for adversarial code and protect the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import fields
from pathlib import Path
from typing import TypeVar

from .schema import ActorItem, Lineage, VerifierItem, from_json, load_protocol, to_json

ACTOR_KEYS = frozenset(f.name for f in fields(ActorItem))
_FILES = ("actor_inputs.jsonl", "verifier_private.jsonl", "lineage.jsonl")
_CODE = ("vault.py", "schema.py", "guards.py")
_LEGACY_CODE = {
    "vault.py": "45b2e61e8eb98fe0e392a201dc9fad8382c61546b4312d40db1c81cbc243d1ce",
    "schema.py": "0605c5f1a1c0f00b08e2b0e5f7dee83e240573e7bdd1ad3b071964c173b44996",
    "guards.py": "e37f158845012e11d22e46d8965d09e422b0e59103dbb68345a857426fe5638a",
}
_COMPATIBLE_SCHEMA = "4c12aea4d0c23792afdd37e57ad62697d3cbc7766bd2700079d45512fea08c93"
_HERE = Path(__file__).resolve().parent
_SHA = re.compile(r"[0-9a-f]{64}\Z")
T = TypeVar("T", ActorItem, VerifierItem, Lineage)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _ids(ids: set[str]) -> str:
    return _sha(_json(sorted(ids)))


def _safe_root(root: Path, *, create: bool = False) -> Path:
    root = Path(root)
    if ".." in root.parts:
        raise ValueError("vault: parent path aliases forbidden")
    root = root.absolute()
    for path in (*reversed(root.parents), root):
        if path.is_symlink():
            raise ValueError("vault: symlink path forbidden")
        if path.exists() and not path.is_dir():
            raise ValueError("vault: root must be a directory")
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("vault: missing split directory")
    return root


def _read(path: Path) -> bytes:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("vault: regular unaliased file required")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("vault: file changed during open")
            return stream.read()
    except OSError as exc:
        raise ValueError("vault: missing or unreadable split file") from exc


def _code_hashes() -> dict[str, str]:
    return {name: _sha((_HERE / name).read_bytes()) for name in _CODE}


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("vault: duplicate manifest key")
        result[key] = value
    return result


def _manifest(root: Path) -> dict:
    try:
        data = json.loads(_read(root / "manifest.yaml"), object_pairs_hook=_unique)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("vault: invalid manifest") from exc
    if (type(data) is not dict or set(data) != {
        "format", "files", "problem_ids_sha256", "code_sha256", "protocol_sha256", "leak_review"
    } or data["format"] != "notebook-rl-vault-v1"):
        raise ValueError("vault: invalid manifest schema")
    if (type(data["files"]) is not dict or set(data["files"]) != set(_FILES)
            or not isinstance(data["problem_ids_sha256"], str)
            or not _SHA.fullmatch(data["problem_ids_sha256"])):
        raise ValueError("vault: invalid file manifest")
    for meta in data["files"].values():
        if (type(meta) is not dict or set(meta) != {"sha256", "rows"}
                or not isinstance(meta["sha256"], str) or not _SHA.fullmatch(meta["sha256"])
                or type(meta["rows"]) is not int or meta["rows"] < 0):
            raise ValueError("vault: invalid file metadata")
    if len({meta["rows"] for meta in data["files"].values()}) != 1:
        raise ValueError("vault: split row count mismatch")
    current_code = _code_hashes()
    # Only SlotReport changed; these pinned actor/verifier/lineage readers are unchanged.
    legacy_reader = (data["code_sha256"] == _LEGACY_CODE
                     and current_code["schema.py"] == _COMPATIBLE_SCHEMA
                     and current_code["guards.py"] == _LEGACY_CODE["guards.py"])
    if ((data["code_sha256"] != current_code and not legacy_reader)
            or data["protocol_sha256"] != load_protocol(_HERE / "protocol_v1.yaml")["sha256"]):
        raise ValueError("vault: code or protocol hash mismatch")
    if type(data["leak_review"]) is not list:
        raise ValueError("vault: invalid leak review")
    for review in data["leak_review"]:
        if (type(review) is not dict or set(review) != {"problem_id", "field"}
                or not isinstance(review["problem_id"], str)
                or review["field"] not in ("question", "choices")):
            raise ValueError("vault: invalid leak review")
    return data


def _check_lineage(lineage: Lineage) -> None:
    if lineage.benchmark_derived or lineage.model_touched:
        raise ValueError("vault: forbidden benchmark or model lineage")


def _load(root: Path, name: str, cls: type[T]) -> list[T]:
    root = _safe_root(root)
    manifest = _manifest(root)
    raw = _read(root / name)
    meta = manifest["files"][name]
    if _sha(raw) != meta["sha256"]:
        raise ValueError("vault: payload hash mismatch")
    try:
        text = raw.decode("utf-8")
        if raw and not raw.endswith(b"\n"):
            raise ValueError("vault: JSONL requires trailing newline")
        lines = text.split("\n")[:-1] if raw else []
        rows = [from_json(cls, line) for line in lines]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("vault: malformed JSONL") from exc
    ids = {row.problem_id for row in rows}
    if len(ids) != len(rows):
        raise ValueError("vault: duplicate problem ID")
    if len(rows) != meta["rows"] or _ids(ids) != manifest["problem_ids_sha256"]:
        raise ValueError("vault: payload identity/count mismatch")
    for row in rows:
        if isinstance(row, Lineage):
            _check_lineage(row)
    return rows


def write_split(root: Path, actors: list[ActorItem], verifiers: list[VerifierItem],
                lineages: list[Lineage]) -> dict:
    records: tuple[list[ActorItem] | list[VerifierItem] | list[Lineage], ...] = (actors, verifiers, lineages)
    payloads: dict[str, bytes] = {}
    id_sets = []
    for name, rows, cls in zip(_FILES, records, (ActorItem, VerifierItem, Lineage), strict=True):
        if any(type(row) is not cls for row in rows):
            raise ValueError("vault: incorrect record type")
        payloads[name] = b"".join(to_json(row).encode("utf-8") + b"\n" for row in rows)
        ids = {row.problem_id for row in rows}
        if len(ids) != len(rows):
            raise ValueError("vault: duplicate problem ID")
        id_sets.append(ids)
    if not id_sets[0] == id_sets[1] == id_sets[2]:
        raise ValueError("vault: problem ID sets differ")
    for lineage in lineages:
        _check_lineage(lineage)
    verifier_by_id = {v.problem_id: v for v in verifiers}
    lineage_by_id = {lineage.problem_id: lineage for lineage in lineages}
    leak_review = []
    for actor in actors:
        verifier = verifier_by_id[actor.problem_id]
        if (actor.answer_format != verifier.answer_type or actor.choices != verifier.choices
                or actor.source != lineage_by_id[actor.problem_id].source):
            raise ValueError("vault: inconsistent actor/verifier/lineage metadata")
        if verifier.answer_type != "mcq":
            pattern = re.compile(r"(?<!\w)" + re.escape(verifier.gold.strip()) + r"(?!\w)")
            for field, text in (("question", actor.question), ("choices", "\n".join(actor.choices))):
                if pattern.search(text):
                    leak_review.append({"problem_id": actor.problem_id, "field": field})
    manifest = {
        "format": "notebook-rl-vault-v1",
        "files": {name: {"sha256": _sha(payloads[name]), "rows": len(rows)}
                  for name, rows in zip(_FILES, records, strict=True)},
        "problem_ids_sha256": _ids(id_sets[0]),
        "code_sha256": _code_hashes(),
        "protocol_sha256": load_protocol(_HERE / "protocol_v1.yaml")["sha256"],
        "leak_review": leak_review,
    }
    root = _safe_root(root, create=True)
    for name in (*_FILES, "manifest.yaml"):
        if (root / name).exists() or (root / name).is_symlink():
            raise FileExistsError("vault: split already exists or is incomplete")
    # Manifest is the completion marker. Failed writes remain visibly incomplete.
    for name, raw in payloads.items():
        with (root / name).open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
    with (root / "manifest.yaml").open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(_json(manifest) + b"\n")
    return manifest


def read_actors(root: Path) -> list[ActorItem]:
    return _load(root, "actor_inputs.jsonl", ActorItem)


def read_verifier(root: Path) -> dict[str, VerifierItem]:
    return {row.problem_id: row for row in _load(root, "verifier_private.jsonl", VerifierItem)}


def read_lineage(root: Path) -> dict[str, Lineage]:
    return {row.problem_id: row for row in _load(root, "lineage.jsonl", Lineage)}
