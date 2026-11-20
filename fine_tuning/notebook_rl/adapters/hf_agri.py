"""Agriculture pool converters: PlantVillage (color train split) and DeepWeeds (train split).

Labels are the upstream expert annotations (PlantVillage class folders, DeepWeeds species column).
Only the question wording and the distractor choices are program-made. Gold letters stay in
Candidate.gold and reach VerifierItem only. Nothing here reads MMMU/MMMU-Pro.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

from .hf_common import Candidate, SourceSpec, program_mcq

SOURCE_PLANT = "hf-plantvillage"
SOURCE_WEEDS = "hf-deepweeds"
SUBJECT = "Agriculture"
HEALTHY_TEXT = "healthy (no disease)"
NO_WEED_TEXT = "none of the listed weed species"
DEEPWEEDS_QUESTION = (
    "This field photo was taken in northern Australian rangeland. "
    "Which weed species is the main plant in the photo?"
)
PLANT_SPLIT_FILE = Path("splits") / "color_train.txt"
PLANT_LEAF_MAP = Path("leaf_grouping") / "leaf-map.json"
PLANT_ZIP = Path("data.zip")
DEEPWEEDS_FILE = Path("data") / "train-00000-of-00001.parquet"


def _plant_condition(member: str) -> tuple[str, str] | None:
    """(crop, condition) from 'raw/color/<Crop>___<Condition>/<file>', None when malformed."""
    parts = member.split("/")
    if len(parts) != 4 or parts[:2] != ["raw", "color"]:
        return None
    pieces = parts[2].split("___")
    if len(pieces) != 2 or not pieces[0] or not pieces[1]:
        return None
    return pieces[0], pieces[1]


def _plant_condition_text(condition: str) -> str:
    text = condition.replace("_", " ").strip()
    return HEALTHY_TEXT if text == "healthy" else text


def _plant_crop_text(crop: str) -> str:
    return crop.replace("_", " ").replace(",", "").strip()


_COPY_SUFFIX = re.compile(r"(?: copy(?: \d+)?)+$")


def _crop_article(crop_text: str) -> str:
    return "an" if crop_text[:1].lower() in ("a", "e", "i", "o", "u") else "a"


def _plant_family_key(member: str, leaf_map: dict[str, list[str]]) -> str:
    # Tags repeat across classes ('rs_hl 6251' -> a Soybean and an Apple leaf); the entry whose class
    # equals the member's class folder is its leaf, so one physical leaf always gets one key.
    class_dir = member.split("/")[2]
    tag = os.path.splitext(member.rsplit("/", 1)[-1].rsplit("___", 1)[-1])[0].lower()
    leaves = leaf_map.get(tag) or []
    if len(leaves) > 1:
        match = [x for x in leaves if x.split(":::")[0] == class_dir]
        leaves = match if len(match) == 1 else leaves
    if leaves:
        return "|".join(sorted(leaves))
    # tag is already lowercased, so this strips ' copy' / ' copy <digits>' case-insensitively:
    # file-name copies of one photo share the fallback family.
    return f"{class_dir}/{_COPY_SUFFIX.sub('', tag)}"


def iter_plantvillage(repo_dir: Path) -> Iterator[Candidate]:
    root = Path(repo_dir)
    rows: list[tuple[str, str, str]] = []  # (member, crop, condition text) in split-file order
    for line in (root / PLANT_SPLIT_FILE).read_text(encoding="utf-8").splitlines():
        parsed = _plant_condition(line)
        if parsed is not None:
            crop, condition = parsed
            rows.append((line, crop, _plant_condition_text(condition)))
    crop_conditions: dict[str, set[str]] = {}
    for _, crop, text in rows:
        crop_conditions.setdefault(crop, set()).add(text)
    pools = {crop: sorted(texts | {HEALTHY_TEXT}) for crop, texts in crop_conditions.items()}
    leaf_map = json.loads((root / PLANT_LEAF_MAP).read_text(encoding="utf-8"))
    with zipfile.ZipFile(root / PLANT_ZIP) as archive:
        names = set(archive.namelist())
        for member, crop, text in rows:
            pool = pools[crop]
            if len(pool) < 2 or member not in names:
                continue
            crop_text = _plant_crop_text(crop)
            choices, letter = program_mcq(text, pool, key=member, n_options=4)
            yield Candidate(
                source=SOURCE_PLANT,
                native_id=member,
                subject=SUBJECT,
                question=f"This photo shows {_crop_article(crop_text)} {crop_text} leaf. Which condition does the leaf show?",
                choices=choices,
                answer_type="mcq",
                gold=letter,
                images=(archive.read(member),),
                family_key=_plant_family_key(member, leaf_map),
            )


def _weed_text(species: str) -> str:
    text = species.strip()
    return NO_WEED_TEXT if text == "Negative" else text


def iter_deepweeds(repo_dir: Path) -> Iterator[Candidate]:
    path = Path(repo_dir) / DEEPWEEDS_FILE
    species_all = pq.read_table(path, columns=["species"]).column("species").to_pylist()
    pool = sorted({_weed_text(s) for s in species_all if isinstance(s, str) and s.strip()})
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["image", "species"]):
        images = batch.column("image").to_pylist()
        labels = batch.column("species").to_pylist()
        for image, species in zip(images, labels):
            row_index = index
            index += 1
            data = image.get("bytes") if isinstance(image, dict) else None
            if not isinstance(data, bytes) or not data or not isinstance(species, str) or not species.strip():
                continue
            choices, letter = program_mcq(_weed_text(species), pool, key=f"deepweeds:{row_index}", n_options=4)
            yield Candidate(
                source=SOURCE_WEEDS,
                native_id=f"train:{row_index}",
                subject=SUBJECT,
                question=DEEPWEEDS_QUESTION,
                choices=choices,
                answer_type="mcq",
                gold=letter,
                images=(data,),
                family_key=hashlib.sha256(data).hexdigest(),
            )


SPECS: dict[str, SourceSpec] = {
    SOURCE_PLANT: SourceSpec(
        source=SOURCE_PLANT,
        repo="mohanty/PlantVillage",
        revision="9e97599868962bd0079b8db4b7f1efa9185fa1e7",
        license="CC-BY-SA-3.0",
        label_provenance="HUMAN",
        upstream_origin="PlantVillage expert class labels (Mohanty et al. 2016)",
        family="plant_photo",
    ),
    SOURCE_WEEDS: SourceSpec(
        source=SOURCE_WEEDS,
        repo="Project-AgML/DeepWeeds_classification",
        revision="8241fce63f7ec845511886cd3f2ab419ab5dba34",
        license="CC-BY-4.0",
        label_provenance="HUMAN",
        upstream_origin="DeepWeeds expert species labels (Olsen et al. 2019)",
        family="plant_photo",
    ),
}

ITERATORS: dict[str, Callable[[Path], Iterator[Candidate]]] = {
    SOURCE_PLANT: iter_plantvillage,
    SOURCE_WEEDS: iter_deepweeds,
}
