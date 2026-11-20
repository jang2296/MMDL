"""Medical pool converters: NCT-CRC-HE histology, bone marrow cytology and PTB-XL ECG.

Labels are the upstream expert annotations (tissue class, cell class, cardiologist superclass).
Only the question wording and the distractor choices are program-made; the PTB-XL image is drawn
by program from the raw signal. The German PTB-XL `report` column is never read. Gold letters
stay in Candidate.gold and reach VerifierItem only. Nothing here reads MMMU/MMMU-Pro.
"""
from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from .hf_common import Candidate, SourceSpec, program_mcq

SOURCE_NCT = "hf-nctcrc"
SOURCE_BM = "hf-bonemarrow"
SOURCE_PTBXL = "hf-ptbxl"

NCT_SUBJECT = "Basic_Medical_Science"
LAB_SUBJECT = "Diagnostics_and_Laboratory_Medicine"
NCT_QUESTION = (
    "This is an H&E-stained human tissue patch. Which tissue class does it show?"
)
BM_QUESTION = (
    "This is a single cell from a May-Grünwald-Giemsa stained bone marrow smear. Which cell type is it?"
)
PTBXL_QUESTION = (
    "This is a 12-lead resting ECG (10 s, 25 mm/s, 10 mm/mV). Which diagnostic class best describes it?"
)

# Upstream label index -> class code (ClassLabel order of the pinned revisions).
NCT_CODES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
NCT_NAMES = {
    "ADI": "adipose tissue", "BACK": "background (no tissue)", "DEB": "debris", "LYM": "lymphocytes",
    "MUC": "mucus", "MUS": "smooth muscle", "NORM": "normal colon mucosa",
    "STR": "cancer-associated stroma", "TUM": "colorectal adenocarcinoma epithelium",
}
BM_CODES = (
    "ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", "HAC", "KSC", "LYI", "LYT",
    "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO",
)
BM_NAMES = {                          # ART, NIF, OTH (artefact / not identifiable / other) have no name: skipped
    "ABE": "abnormal eosinophil", "BAS": "basophil", "BLA": "blast", "EBO": "erythroblast", "EOS": "eosinophil",
    "FGC": "faggot cell", "HAC": "hairy cell", "KSC": "smudge cell", "LYI": "immature lymphocyte",
    "LYT": "lymphocyte", "MMZ": "metamyelocyte", "MON": "monocyte", "MYB": "myelocyte",
    "NGB": "band neutrophil", "NGS": "segmented neutrophil", "PEB": "proerythroblast", "PLM": "plasma cell",
    "PMO": "promyelocyte",
}
PTBXL_NAMES = {
    "NORM": "normal ECG", "MI": "myocardial infarction", "STTC": "ST/T change",
    "CD": "conduction disturbance", "HYP": "hypertrophy",
}
PTBXL_COLUMNS = ("ecg_id", "patient_id", "strat_fold", "validated_by_human", "superclasses", "fs", "leads")
PTBXL_MAX_FOLD = 8                    # folds 9 and 10 stay out of the pool (PTB-XL test folds)
ECG_LEADS = 12
ECG_SECONDS = 10

# ECG canvas: 25 mm/s and 10 mm/mV, so one 5 mm cell is 0.2 s wide and 0.5 mV high.
ECG_W, ECG_H = 1600, 1900
ECG_LEFT, ECG_PLOT_W = 40, 1520
ECG_CELL = ECG_PLOT_W / 50
ECG_STRIP_H = 5 * ECG_CELL            # 2.5 mV per lead strip
# Real QRS peaks (record median ~2 mV, HYP voltage criteria up to ~4 mV) overlap the neighbour strip as on
# paper; only artefact excursions beyond +/-3.75 mV are cut. A per-strip clamp would erase the HYP evidence.
ECG_MAX_SWING = 1.5 * ECG_STRIP_H
ECG_TOP = (ECG_H - ECG_LEADS * ECG_STRIP_H) / 2
ECG_GRID = (255, 190, 190)
ECG_FONT_SIZE = 22


def _shards(repo_dir: Path, pattern: str) -> list[Path]:
    paths = sorted((Path(repo_dir) / "data").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet shard matches data/{pattern} in {repo_dir}")
    return paths


def _iter_labeled_images(
    repo_dir: Path, pattern: str, codes: tuple[str, ...], names: dict[str, str],
    *, source: str, subject: str, question: str, key_prefix: str,
) -> Iterator[Candidate]:
    pool = sorted(names.values())
    for path in _shards(repo_dir, pattern):
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image", "label"]):
            for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                row = row_index
                row_index += 1
                data = image.get("bytes") if isinstance(image, dict) else None
                name = names.get(codes[label]) if type(label) is int and 0 <= label < len(codes) else None
                if not isinstance(data, bytes) or not data or name is None:
                    continue
                native_id = f"{path.stem}:{row}"
                choices, letter = program_mcq(name, pool, key=f"{key_prefix}:{native_id}", n_options=4)
                yield Candidate(
                    source=source,
                    native_id=native_id,
                    subject=subject,
                    question=question,
                    choices=choices,
                    answer_type="mcq",
                    gold=letter,
                    images=(data,),
                    family_key=hashlib.sha256(data).hexdigest(),
                )


def iter_nctcrc(repo_dir: Path) -> Iterator[Candidate]:
    # NCT_CRC_HE_100K-* only: the CRC_VAL_HE_7K and NONORM shards are different files.
    return _iter_labeled_images(
        repo_dir, "NCT_CRC_HE_100K-*.parquet", NCT_CODES, NCT_NAMES,
        source=SOURCE_NCT, subject=NCT_SUBJECT, question=NCT_QUESTION, key_prefix="nct",
    )


def iter_bonemarrow(repo_dir: Path) -> Iterator[Candidate]:
    return _iter_labeled_images(
        repo_dir, "train-*.parquet", BM_CODES, BM_NAMES,
        source=SOURCE_BM, subject=LAB_SUBJECT, question=BM_QUESTION, key_prefix="bm",
    )


def render_ecg(signal: list[list[float]], leads: list[str], fs: int) -> bytes:
    """Standard-paper style PNG: light-red 0.2 s / 0.5 mV grid and one labelled strip per lead."""
    data = np.asarray(signal, dtype=np.float64)
    if (data.ndim != 2 or data.shape[0] != ECG_LEADS or len(leads) != ECG_LEADS or fs <= 0
            or data.shape[1] != fs * ECG_SECONDS or not np.isfinite(data).all()):
        raise ValueError("ecg: 12 leads of finite fs*10 samples each are required")
    image = Image.new("RGB", (ECG_W, ECG_H), "white")
    draw = ImageDraw.Draw(image)
    bottom = round(ECG_TOP + ECG_LEADS * ECG_STRIP_H)
    for k in range(51):
        x = round(ECG_LEFT + k * ECG_CELL)
        draw.line([(x, round(ECG_TOP)), (x, bottom)], fill=ECG_GRID)
    for k in range(ECG_LEADS * 5 + 1):
        y = round(ECG_TOP + k * ECG_CELL)
        draw.line([(ECG_LEFT, y), (ECG_LEFT + ECG_PLOT_W, y)], fill=ECG_GRID)
    # Downsample to one x pixel per column: keep the column's min and max so QRS spikes survive.
    edges = np.linspace(0, data.shape[1], ECG_PLOT_W + 1).astype(int)
    lo = np.minimum.reduceat(data, edges[:-1], axis=1)
    hi = np.maximum.reduceat(data, edges[:-1], axis=1)
    rising = data[:, edges[:-1]] <= data[:, edges[1:] - 1]
    first, second = np.where(rising, lo, hi), np.where(rising, hi, lo)
    px_per_mv = ECG_CELL / 0.5
    xs = np.repeat(ECG_LEFT + np.arange(ECG_PLOT_W), 2).tolist()
    font = ImageFont.load_default(size=ECG_FONT_SIZE)
    for i, name in enumerate(leads):
        base = ECG_TOP + (i + 0.5) * ECG_STRIP_H
        mv = np.stack([first[i], second[i]], axis=1).reshape(-1)
        ys = (base - np.clip(mv * px_per_mv, -ECG_MAX_SWING, ECG_MAX_SWING)).tolist()
        draw.line(list(zip(xs, ys)), fill="black", width=2, joint="curve")
        draw.text((ECG_LEFT + 6, round(ECG_TOP + i * ECG_STRIP_H) + 2), str(name), fill="black", font=font)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _ptbxl_label(row: dict) -> str | None:
    superclasses = row["superclasses"]
    labels = set(superclasses) if isinstance(superclasses, list) else set()
    fs, leads = row["fs"], row["leads"]
    if (row["validated_by_human"] is not True
            or type(row["ecg_id"]) is not int or type(row["patient_id"]) is not int
            or type(row["strat_fold"]) is not int or not 1 <= row["strat_fold"] <= PTBXL_MAX_FOLD
            or len(labels) != 1 or not labels <= PTBXL_NAMES.keys()
            or type(fs) is not int or fs <= 0
            or not isinstance(leads, list) or len(leads) != ECG_LEADS
            or not all(isinstance(x, str) and x for x in leads)):
        return None
    return next(iter(labels))


def _ptbxl_signal(value: object, fs: int) -> list[list[float]] | None:
    try:
        data = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):       # ragged or non-numeric nesting
        return None
    if data.shape != (ECG_LEADS, fs * ECG_SECONDS) or not np.isfinite(data).all():
        return None
    return value  # type: ignore[return-value]


def iter_ptbxl(repo_dir: Path) -> Iterator[Candidate]:
    pool = sorted(PTBXL_NAMES.values())
    columns = [*PTBXL_COLUMNS, "signal"]   # never "report": German free text
    for path in _shards(repo_dir, "train-*.parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=columns):
            for i in range(batch.num_rows):
                row = {name: batch.column(name)[i].as_py() for name in PTBXL_COLUMNS}
                code = _ptbxl_label(row)
                if code is None:
                    continue
                signal = _ptbxl_signal(batch.column("signal")[i].as_py(), row["fs"])
                if signal is None:
                    continue
                choices, letter = program_mcq(
                    PTBXL_NAMES[code], pool, key=f"ptbxl:{row['ecg_id']}", n_options=len(pool))
                yield Candidate(
                    source=SOURCE_PTBXL,
                    native_id=f"ecg:{row['ecg_id']}",
                    subject=LAB_SUBJECT,
                    question=PTBXL_QUESTION,
                    choices=choices,
                    answer_type="mcq",
                    gold=letter,
                    images=(render_ecg(signal, row["leads"], row["fs"]),),
                    family_key=f"patient:{row['patient_id']}",
                )


SPECS: dict[str, SourceSpec] = {
    SOURCE_NCT: SourceSpec(
        source=SOURCE_NCT,
        repo="1aurent/NCT-CRC-HE",
        revision="97765ecb14bf11ce7bb8779a1097d187844a67cb",
        license="CC-BY-4.0",
        label_provenance="HUMAN",
        upstream_origin="NCT-CRC-HE expert tissue labels (Kather et al. 2019)",
        family="histology",
    ),
    SOURCE_BM: SourceSpec(
        source=SOURCE_BM,
        repo="ekim15/bone_marrow_cell_dataset",
        revision="a9c0cf3bf5ea1f0b502f50de40fece30a82f3aed",
        license="CC-BY-4.0",
        label_provenance="HUMAN",
        upstream_origin="MLL expert cell labels (Matek et al. 2021)",
        family="cytology",
    ),
    SOURCE_PTBXL: SourceSpec(
        source=SOURCE_PTBXL,
        repo="tirandazdylan/ptb-xl-ecg",
        revision="9e69b42bdb58a3beb39354efa096676b8fb928fd",
        license="CC-BY-4.0",
        label_provenance="HUMAN",
        upstream_origin=(
            "PTB-XL cardiologist statements (Wagner et al. 2020); image rendered by program from the signal"
        ),
        family="ecg",
    ),
}

ITERATORS: dict[str, Callable[[Path], Iterator[Candidate]]] = {
    SOURCE_NCT: iter_nctcrc,
    SOURCE_BM: iter_bonemarrow,
    SOURCE_PTBXL: iter_ptbxl,
}
