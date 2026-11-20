"""Private, source-backed training answer rules; actor processes must deny this file."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from fine_tuning.notebook_rl.answers import compare_training
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import _read, _safe_root

VERSION = "training-answer-rules-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {"actor_sha256", "image_sha256", "source", "source_revision", "split", "native_id",
           "answer_type", "accept", "provenance"}
_PROVENANCE = {"label_provenance", "field", "location", "sha256", "human_verified", "review_scope"}
_NUMERIC = {"round_to", "abs_tol", "rel_tol"}
_ACCEPT = {"int": set(), "decimal": _NUMERIC, "fraction": {"accept_decimal"},
           "unit_number": _NUMERIC | {"unit_aliases", "require_unit"},
           "short_text": {"aliases", "case_sensitive"}, "range": set()}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("training rules: duplicate key")
        result[key] = value
    return result


def load_training_rules(path: Path, sha256: str, *, vault_root: Path, actors: dict,
                        items: dict, lineages: dict) -> dict[str, dict]:
    """Verify pinned bytes and original train provenance before exposing rules to the verifier."""
    if not isinstance(sha256, str) or not _SHA.fullmatch(sha256):
        raise ValueError("training rules: explicit SHA256 required")
    path = Path(path).absolute()
    _safe_root(path.parent)
    raw = _read(path)
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("training rules: SHA256 mismatch")
    data = json.loads(raw, object_pairs_hook=_unique)
    if (type(data) is not dict or set(data) != {"version", "vault_manifest_sha256", "items"}
            or data["version"] != VERSION or type(data["items"]) is not dict
            or data["vault_manifest_sha256"] != hashlib.sha256(_read(vault_root / "manifest.yaml")).hexdigest()):
        raise ValueError("training rules: invalid manifest binding")
    for pid, rule in data["items"].items():
        if pid not in actors or pid not in items or pid not in lineages or type(rule) is not dict or set(rule) != _FIELDS:
            raise ValueError("training rules: invalid item")
        actor, lineage = actors[pid], lineages[pid]
        provenance, accept, kind = rule["provenance"], rule["accept"], rule["answer_type"]
        if (rule["actor_sha256"] != hashlib.sha256(to_json(actor).encode()).hexdigest()
                or rule["image_sha256"] != list(actor.image_sha256) or rule["source"] != actor.source
                or rule["source_revision"] != lineage.source_revision or rule["split"] != lineage.split
                or rule["split"] != "train" or not isinstance(rule["native_id"], str) or not rule["native_id"].strip()
                or type(provenance) is not dict or set(provenance) != _PROVENANCE
                or provenance["label_provenance"] != lineage.label_provenance
                or provenance["label_provenance"] not in ("HUMAN", "PROGRAM")
                or type(provenance["human_verified"]) is not bool
                or any(not isinstance(provenance[key], str) or not provenance[key].strip()
                       for key in ("field", "location", "review_scope", "sha256"))
                or not _SHA.fullmatch(provenance["sha256"])):
            raise ValueError("training rules: provenance/actor binding mismatch")
        if (not isinstance(kind, str) or kind not in _ACCEPT or type(accept) is not dict
                or set(accept) - (_ACCEPT[kind] | {"variable"})
                or "variable" in accept and (kind in ("short_text", "range")
                    or not isinstance(accept["variable"], str) or not re.fullmatch(r"[A-Za-z]", accept["variable"]))):
            raise ValueError("training rules: unsupported typed policy")
        if items[pid].answer_type == "mcq" or compare_training(items[pid].gold, items[pid], rule) is not True:
            raise ValueError("training rules: policy cannot compare original gold")
    return data["items"]
