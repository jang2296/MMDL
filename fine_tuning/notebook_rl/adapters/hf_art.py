"""Art, design and catalogue pool converters: The Met Open Access, Cooper Hewitt and Crello.

Labels are the upstream catalogue fields (Met 'Object Name', Cooper Hewitt 'medium', Crello 'category').
Only the question wording and the distractor choices are program-made. Gold letters stay in
Candidate.gold and reach VerifierItem only. Nothing here reads MMMU/MMMU-Pro.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

from .hf_common import Candidate, SourceSpec, program_mcq

SOURCE_MET = "hf-met"
SOURCE_COOPER = "hf-cooperhewitt"
SOURCE_CRELLO = "hf-crello"
MET_MIN_NAME_COUNT = 15
COOPER_QUESTION = (
    "According to the museum catalogue, which material or technique was mainly used to make this object?"
)
CRELLO_QUESTION = "This is a graphic design template. What is it designed for?"
ART_TECHNIQUES = frozenset((
    "woodcut", "lithograph", "screenprint", "etching", "engraving",
    "watercolor", "gouache", "charcoal", "graphite", "pen and ink",
))
# (casefold substring of Cooper Hewitt 'medium', technique name). The first matching rule wins.
TECHNIQUE_RULES: tuple[tuple[str, str], ...] = (
    ("woodcut", "woodcut"), ("lithograph", "lithograph"), ("screenprint", "screenprint"),
    ("screen print", "screenprint"), ("etching", "etching"), ("engraving", "engraving"),
    ("watercolor", "watercolor"), ("gouache", "gouache"), ("charcoal", "charcoal"),
    ("graphite", "graphite"), ("pen and", "pen and ink"), ("machine-printed", "machine-printed paper"),
    ("block-printed", "block-printed paper"), ("tin-glazed earthenware", "tin-glazed earthenware"),
    ("porcelain", "porcelain"), ("glass", "glass"), ("silver", "silver"), ("silk", "silk"),
    ("cotton", "cotton"), ("wool", "wool"),
)
# Crello 'category' class_label names (RAW/cyberagent__crello/README.md); index = class id 0..22.
# Class 22 'all' is a catch-all, not a purpose: it is never a gold or a distractor.
CRELLO_CATCH_ALL = "all"
CRELLO_CATEGORY_NAMES: tuple[str, ...] = (
    "holidaysCelebration", "foodDrinks", "fashionStyle", "businessFinance", "homeStuff", "handcraftArt",
    "beauty", "leisureEntertainment", "natureWildlife", "educationScience", "technology", "medical",
    "socialActivityCharity", "sportExtreme", "realEstateBuilding", "travelsVacations", "pets", "religions",
    "citiesPlaces", "industry", "transportation", "kidsParents", "all",
)

SPECS: dict[str, SourceSpec] = {
    SOURCE_MET: SourceSpec(
        source=SOURCE_MET,
        repo="metmuseum/openaccess",
        revision="c65f8d6041aea7b3bc767a54d93772c3c6a365f6",
        license="CC0-1.0",
        label_provenance="HUMAN",
        upstream_origin="The Met cataloguers (Open Access metadata)",
        family="artwork",
    ),
    SOURCE_COOPER: SourceSpec(
        source=SOURCE_COOPER,
        repo="biglam/cooper-hewitt-alt-text",
        revision="a460beb10df99665253aa6f17aeaa63ffebde5a6",
        license="CC0-1.0",
        label_provenance="HUMAN",
        upstream_origin="Cooper Hewitt cataloguers",
        family="artwork",
    ),
    SOURCE_CRELLO: SourceSpec(
        source=SOURCE_CRELLO,
        repo="cyberagent/crello",
        revision="7997e2f434ee4aa73cf4cdf22c5954cb175872e1",
        license="CDLA-Permissive-2.0",
        label_provenance="HUMAN",
        upstream_origin="Crello/VistaCreate template category metadata",
        family="design_template",
    ),
}


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _image_bytes(value: object) -> bytes | None:
    # Met 'jpg' is a binary column; Cooper Hewitt and Crello store {'bytes', 'path'} structs.
    data = value.get("bytes") if isinstance(value, dict) else value
    return data if isinstance(data, bytes) and data else None


def _shards(directory: Path, pattern: str) -> list[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet shards matching {pattern} in {directory}")
    return files


def _iter_rows(files: list[Path], columns: list[str]) -> Iterator[dict]:
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def iter_met(repo_dir: Path) -> Iterator[Candidate]:
    files = _shards(Path(repo_dir) / "default" / "train", "*.parquet")
    counts: Counter[str] = Counter()
    for row in _iter_rows(files, ["Object Name", "jpg"]):
        name = _text(row["Object Name"])
        if name is not None and _image_bytes(row["jpg"]) is not None:
            counts[name] += 1
    pool = sorted(name for name, count in counts.items() if count >= MET_MIN_NAME_COUNT)
    if len(pool) < 2:  # program_mcq needs one distractor besides the gold name
        return
    pool_set = set(pool)
    for row in _iter_rows(files, ["Object ID", "Department", "Object Name", "jpg"]):
        name = _text(row["Object Name"])
        image = _image_bytes(row["jpg"])
        object_id = row["Object ID"]
        department = _text(row["Department"])
        if name not in pool_set or image is None or type(object_id) is not int or department is None:
            continue
        native_id = f"met:{object_id}"
        choices, letter = program_mcq(name, pool, key=native_id, n_options=4)
        yield Candidate(
            source=SOURCE_MET,
            native_id=native_id,
            subject="Design" if department == "Costume Institute" else "Art",
            question=f"This object is from The Metropolitan Museum of Art ({department}). What kind of object is it?",
            choices=choices,
            answer_type="mcq",
            gold=letter,
            images=(image,),
            family_key=native_id,
        )


def _techniques(medium: str) -> list[str]:
    """Every technique the medium text names, in rule priority order; the first is the gold."""
    lowered = medium.casefold()
    names: list[str] = []
    for needle, name in TECHNIQUE_RULES:
        if needle in lowered and name not in names:
            names.append(name)
    return names


def _cooper_family_key(record_id: str) -> str:
    # Parts of one object carry a trailing part suffix ('-a', '-a_c' part range, '-a-2' sub-part)
    # and share a family. Over-merging only coarsens the split; under-merging would leak an object.
    return re.sub(r"-[A-Za-z](?:_[A-Za-z])?(?:-\d+)?$", "", record_id)


def iter_cooperhewitt(repo_dir: Path) -> Iterator[Candidate]:
    files = _shards(Path(repo_dir) / "data", "train-*.parquet")
    pool = sorted({name for _, name in TECHNIQUE_RULES})
    for row in _iter_rows(files, ["image", "medium", "record_id"]):
        medium = _text(row["medium"])
        record_id = _text(row["record_id"])
        image = _image_bytes(row["image"])
        named = _techniques(medium) if medium is not None else []
        if not named or record_id is None or image is None:
            continue
        technique = named[0]
        # 'Etching and engraving': a co-named technique is also a true answer, so it is never a distractor.
        distractor_pool = [name for name in pool if name not in named[1:]]
        choices, letter = program_mcq(technique, distractor_pool, key=f"ch:{record_id}", n_options=4)
        yield Candidate(
            source=SOURCE_COOPER,
            native_id=record_id,
            subject="Art" if technique in ART_TECHNIQUES else "Design",
            question=COOPER_QUESTION,
            choices=choices,
            answer_type="mcq",
            gold=letter,
            images=(image,),
            family_key=_cooper_family_key(record_id),
        )


def _crello_text(name: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").lower()


def iter_crello(repo_dir: Path) -> Iterator[Candidate]:
    files = _shards(Path(repo_dir) / "data", "train-*.parquet")
    names = [_crello_text(x) for x in CRELLO_CATEGORY_NAMES]
    pool = sorted(set(names) - {CRELLO_CATCH_ALL})
    for row in _iter_rows(files, ["id", "category", "preview", "title"]):
        item_id = _text(row["id"])
        category = row["category"]
        image = _image_bytes(row["preview"])
        if (item_id is None or type(category) is not int or not 0 <= category < len(names)
                or names[category] == CRELLO_CATCH_ALL or image is None):
            continue
        # One design is published in several formats under one title; those ids share a family.
        title = _text(row["title"])
        family_key = f"crello:title:{title.casefold()}" if title and title != "None" else f"crello:{item_id}"
        choices, letter = program_mcq(names[category], pool, key=f"crello:{item_id}", n_options=4)
        yield Candidate(
            source=SOURCE_CRELLO,
            native_id=item_id,
            subject="Design",
            question=CRELLO_QUESTION,
            choices=choices,
            answer_type="mcq",
            gold=letter,
            images=(image,),
            family_key=family_key,
        )


ITERATORS: dict[str, Callable[[Path], Iterator[Candidate]]] = {
    SOURCE_MET: iter_met,
    SOURCE_COOPER: iter_cooperhewitt,
    SOURCE_CRELLO: iter_crello,
}
