"""Source id -> (spec, row iterator, raw repo directory) for the pinned HF pool converters."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from . import (
    hf_agri, hf_art, hf_cauldron, hf_chem, hf_circuit, hf_clevr, hf_docs, hf_graph_vlqa, hf_medical, hf_music,
    hf_pathology, hf_science, hf_tatqa_table,
)
from .hf_common import Candidate, SourceSpec

_MODULES = (hf_cauldron, hf_science, hf_agri, hf_medical, hf_pathology, hf_art, hf_music, hf_chem,
            hf_docs, hf_graph_vlqa, hf_circuit, hf_clevr, hf_tatqa_table)


def repo_dir(spec: SourceSpec) -> str:
    return spec.repo.replace("/", "__")


def sources() -> dict[str, tuple[SourceSpec, Callable[[Path], Iterator[Candidate]]]]:
    out: dict[str, tuple[SourceSpec, Callable[[Path], Iterator[Candidate]]]] = {}
    for module in _MODULES:
        if set(module.SPECS) != set(module.ITERATORS):
            raise ValueError(f"hf registry: {module.__name__} specs/iterators mismatch")
        for name, spec in module.SPECS.items():
            if name in out or spec.source != name:
                raise ValueError(f"hf registry: duplicate or inconsistent source {name}")
            out[name] = (spec, module.ITERATORS[name])
    return out
