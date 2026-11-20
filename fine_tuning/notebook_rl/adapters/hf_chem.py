"""Chemistry/Pharmacy pool converters over ChEBI-20-MM molecule images.

hf-chebi (retired for new plans; kept only to reproduce old builds): labels computed by RDKit, read from
derived/chebi_features_v1.jsonl (adapters/chebi_features.py). The description column is never read there.
One question template is picked per molecule from its CID alone, so selection is label-blind.

hf-chebi-role (class B, current): the gold is a ChEBI curator-assigned role, read verbatim from the
"It has a role as a X, an Y and a Z." sentence shipped in train.csv. Nothing is computed or inferred
about the molecule; only the 4-option layout is built by a fixed, label-blind program.

Gold stays in Candidate.gold and reaches VerifierItem only. Nothing here reads MMMU/MMMU-Pro.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

from .hf_common import Candidate, SourceSpec, program_mcq

SOURCE_CHEBI = "hf-chebi"
SOURCE_CHEBI_ROLE = "hf-chebi-role"
TRAIN_CSV = Path("train.csv")
FEATURES_FILE = Path("derived") / "chebi_features_v1.jsonl"
IMAGE_ZIP = Path("image.zip")
MAX_HEAVY_ATOMS = 60
MAX_RINGS = 12
FORMULA_HEAVY_ATOM_WINDOW = 2
QUESTIONS = (
    "How many rings does the molecule in the image contain?",
    "How many oxygen atoms does the molecule in the image contain?",
    "Which molecular formula matches the molecule in the image?",
)


def _template(cid: int) -> int:
    return int(hashlib.sha256(f"chebi:{cid}".encode()).hexdigest(), 16) % len(QUESTIONS)


def _load_features(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"ChEBI features missing: {path}; run adapters/chebi_features.py with the notebook-data env first"
        )
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_chebi(repo_dir: Path) -> Iterator[Candidate]:
    root = Path(repo_dir)
    records = _load_features(root / FEATURES_FILE)
    by_heavy_atoms: dict[int, set[str]] = {}
    for record in records:
        by_heavy_atoms.setdefault(record["heavy_atoms"], set()).add(record["formula"])
    with zipfile.ZipFile(root / IMAGE_ZIP) as archive:
        names = set(archive.namelist())
        for record in records:
            cid, heavy_atoms, rings = record["cid"], record["heavy_atoms"], record["rings"]
            member = f"image/CID_{cid}.png"
            if heavy_atoms > MAX_HEAVY_ATOMS or rings > MAX_RINGS or member not in names:
                continue
            choices: tuple[str, ...] = ()
            template = _template(cid)
            if template == 0:
                subject, answer_type, gold = "Chemistry", "int", str(rings)
            elif template == 1:
                subject, answer_type, gold = "Pharmacy", "int", str(record["n_O"])
            else:
                pool = sorted(set().union(*(
                    by_heavy_atoms.get(h, set())
                    for h in range(heavy_atoms - FORMULA_HEAVY_ATOM_WINDOW, heavy_atoms + FORMULA_HEAVY_ATOM_WINDOW + 1)
                )))
                if len(pool) < 2:
                    continue
                choices, gold = program_mcq(record["formula"], pool, key=f"chebi:{cid}", n_options=4)
                subject, answer_type = "Pharmacy", "mcq"
            yield Candidate(
                source=SOURCE_CHEBI,
                native_id=f"cid:{cid}",
                subject=subject,
                question=QUESTIONS[template],
                choices=choices,
                answer_type=answer_type,
                gold=gold,
                images=(archive.read(member),),
                family_key=f"cid:{cid}",
            )


# ---- hf-chebi-role -------------------------------------------------------------------------------------
ROLE_QUESTION = "Which pharmacological role is assigned to the molecule shown in the image?"
ROLE_OPTIONS = 4

# Allow-list: exact role phrases (leading "a"/"an" removed) from the train.csv vocabulary that name a drug class or
# a mechanism-of-action role. Organism/xenobiotic metabolites, toxins, agrochemicals, dyes, solvents, food and
# chemical-reaction roles are not pharmacological and are left out. The family groups near-synonyms and
# overlapping classes: distractors come only from families that contain NONE of the molecule's allow-listed roles.
ROLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "anti_infective": (
        "antibacterial agent", "antibacterial drug", "antimicrobial agent", "antiinfective agent",
        "antifungal agent", "antifungal drug", "antiviral agent", "antiviral drug", "anti-HIV agent",
        "anti-HIV-1 agent", "anticoronaviral agent", "anti-HSV-1 agent", "anti-HBV agent", "antimalarial",
        "antiplasmodial drug", "antiprotozoal drug", "antileishmanial agent", "trypanocidal drug",
        "anthelminthic drug", "antiparasitic agent", "antinematodal drug", "antitubercular agent",
        "antimycobacterial drug", "antiseptic drug", "schistosomicide drug", "HIV-1 reverse transcriptase inhibitor",
        "HIV protease inhibitor", "HIV-1 integrase inhibitor", "hepatitis C protease inhibitor",
    ),
    "antineoplastic": (
        "antineoplastic agent", "apoptosis inducer", "antimitotic", "angiogenesis inhibitor",
        "angiogenesis modulating agent", "tyrosine kinase inhibitor", "protein kinase inhibitor", "mTOR inhibitor",
        "B-Raf inhibitor", "proteasome inhibitor", "Hsp90 inhibitor", "tubulin modulator",
        "microtubule-destabilising agent", "microtubule-stabilising agent",
        "epidermal growth factor receptor antagonist", "vascular endothelial growth factor receptor antagonist",
        "fibroblast growth factor receptor antagonist", "EC 2.7.10.1 (receptor protein-tyrosine kinase) inhibitor",
        "EC 2.7.10.2 (non-specific protein-tyrosine kinase) inhibitor", "EC 2.7.11.22 (cyclin-dependent kinase) inhibitor",
        "EC 2.7.1.137 (phosphatidylinositol 3-kinase) inhibitor", "EC 3.5.1.98 (histone deacetylase) inhibitor",
        "EC 5.99.1.3 [DNA topoisomerase (ATP-hydrolysing)] inhibitor", "EC 5.99.1.2 (DNA topoisomerase) inhibitor",
        "EC 1.5.1.3 (dihydrofolate reductase) inhibitor",
    ),
    "inflammation_immune_pain": (
        "anti-inflammatory agent", "anti-inflammatory drug", "non-steroidal anti-inflammatory drug",
        "cyclooxygenase 1 inhibitor", "cyclooxygenase 2 inhibitor", "lipoxygenase inhibitor", "NF-kappaB inhibitor",
        "EC 1.14.99.1 (prostaglandin-endoperoxide synthase) inhibitor",
        "EC 1.13.11.34 (arachidonate 5-lipoxygenase) inhibitor", "antirheumatic drug", "immunosuppressive agent",
        "immunomodulator", "anti-allergic agent", "antipruritic drug", "antipsoriatic", "H1-receptor antagonist",
        "histamine antagonist", "leukotriene antagonist", "analgesic", "non-narcotic analgesic", "opioid analgesic",
        "antipyretic", "mu-opioid receptor agonist", "kappa-opioid receptor agonist", "delta-opioid receptor agonist",
        "mu-opioid receptor antagonist",
    ),
    "cardiovascular_blood": (
        "antihypertensive agent", "vasodilator agent", "vasoconstrictor agent", "anti-arrhythmia drug",
        "cardiotonic drug", "cardiovascular drug", "cardioprotective agent", "calcium channel blocker",
        "calcium channel modulator", "potassium channel blocker", "sodium channel blocker",
        "angiotensin receptor antagonist", "EC 3.4.15.1 (peptidyl-dipeptidase A) inhibitor", "diuretic",
        "nitric oxide donor", "antilipemic drug", "anticholesteremic drug", "anticoagulant",
        "platelet aggregation inhibitor", "P2Y12 receptor antagonist", "EC 3.4.21.5 (thrombin) inhibitor",
        "EC 3.4.21.6 (coagulation factor Xa) inhibitor", "alpha-adrenergic agonist", "alpha-adrenergic antagonist",
        "beta-adrenergic agonist", "beta-adrenergic antagonist", "adrenergic agent", "adrenergic antagonist",
        "adrenergic uptake inhibitor", "sympathomimetic agent", "sympatholytic agent",
    ),
    "cns_neuro_psychiatric": (
        "anticonvulsant", "antidepressant", "anxiolytic drug", "sedative", "antipsychotic agent",
        "first generation antipsychotic", "second generation antipsychotic", "phenothiazine antipsychotic drug",
        "antimanic drug", "psychotropic drug", "central nervous system depressant", "central nervous system stimulant",
        "central nervous system drug", "peripheral nervous system drug", "neuroprotective agent", "antiparkinson drug",
        "antidyskinesia agent", "NMDA receptor antagonist", "GABA modulator", "GABAA receptor agonist",
        "serotonergic agonist", "serotonergic antagonist", "serotonin uptake inhibitor", "dopaminergic agent",
        "dopaminergic antagonist", "dopamine agonist", "dopamine uptake inhibitor", "muscarinic agonist",
        "muscarinic antagonist", "cholinergic antagonist", "cholinergic drug", "nicotinic antagonist",
        "nicotinic acetylcholine receptor agonist", "parasympatholytic", "muscle relaxant", "neuromuscular agent",
        "anaesthetic", "local anaesthetic", "general anaesthetic", "intravenous anaesthetic", "inhalation anaesthetic",
        "anaesthesia adjuvant", "hallucinogen", "EC 3.1.1.7 (acetylcholinesterase) inhibitor",
        "EC 3.1.1.8 (cholinesterase) inhibitor", "EC 1.4.3.4 (monoamine oxidase) inhibitor",
    ),
    "endocrine_metabolic": (
        "hypoglycemic agent", "anti-obesity agent", "appetite depressant", "anabolic agent",
        "bone density conservation agent", "androgen", "androgen antagonist", "estrogen", "estrogen receptor agonist",
        "estrogen receptor antagonist", "estrogen receptor modulator", "anti-estrogen", "progestin",
        "contraceptive drug", "synthetic oral contraceptive", "PPARalpha agonist", "PPARgamma agonist",
        "EC 3.2.1.20 (alpha-glucosidase) inhibitor",
    ),
    "gastrointestinal_hepatic": (
        "anti-ulcer drug", "gastrointestinal drug", "antispasmodic drug", "antiemetic", "emetic", "laxative",
        "antidiarrhoeal drug", "H2-receptor antagonist", "EC 3.6.3.10 (H(+)/K(+)-exchanging ATPase) inhibitor",
        "hepatoprotective agent",
    ),
    "respiratory": (
        "bronchodilator agent", "anti-asthmatic drug", "antitussive", "expectorant", "mucolytic",
    ),
    "other_therapeutic": (
        "antidote", "diagnostic agent", "dermatologic drug", "keratolytic drug", "ophthalmology drug",
        "antiglaucoma drug", "mydriatic agent", "EC 4.2.1.1 (carbonic anhydrase) inhibitor", "oxytocic",
        "abortifacient", "antispermatogenic agent", "vulnerary", "astringent", "nephroprotective agent",
        "radiation protective agent",
    ),
}
# Not gold options, but a listed role of one of these makes its family off-limits for distractors.
ROLE_RELATED_ONLY: dict[str, tuple[str, ...]] = {
    "antifungal agrochemical": ("anti_infective",), "fungicide": ("anti_infective",),
    "disinfectant": ("anti_infective",), "antifouling biocide": ("anti_infective",),
    "antimicrobial food preservative": ("anti_infective",), "topoisomerase IV inhibitor": ("anti_infective",),
    "EC 2.5.1.15 (dihydropteroate synthase) inhibitor": ("anti_infective",),
    "EC 3.5.2.6 (beta-lactamase) inhibitor": ("anti_infective",),
    "alkylating agent": ("antineoplastic",), "radiosensitizing agent": ("antineoplastic",),
    "actin polymerisation inhibitor": ("antineoplastic",), "antimetabolite": ("antineoplastic",),
    "topoisomerase inhibitor": ("antineoplastic",),
    "DNA synthesis inhibitor": ("antineoplastic", "anti_infective"),
    "protein synthesis inhibitor": ("antineoplastic", "anti_infective"),
}
ROLE_FAMILY: dict[str, str] = {p: f for f, phrases in ROLE_FAMILIES.items() for p in phrases}
if len(ROLE_FAMILY) != sum(map(len, ROLE_FAMILIES.values())) or ROLE_FAMILY.keys() & ROLE_RELATED_ONLY.keys():
    raise ValueError("hf-chebi-role: a role phrase is listed twice")

_ROLE_SENTENCE = re.compile(r"It has a role as (.+?)\.(?=\s[A-Z]|$)")
_ROLE_SPLIT = re.compile(r", and (?=an? )|, (?=an? )| and (?=an? )")
_SMILES_HEAVY_ATOM = re.compile(r"\[(?!H\])[^\]]+\]|Br|Cl|[BCNOPSFIbcnops]")


def parse_roles(description: str) -> tuple[str, ...]:
    """Role phrases of every "It has a role as ..." sentence, as written, without the leading a/an.

    A sentence whose items do not all start with "a "/"an " is not split reliably and contributes nothing.
    """
    roles: list[str] = []
    for sentence in _ROLE_SENTENCE.finditer(description):
        items = [re.fullmatch(r"an? (.+)", part) for part in _ROLE_SPLIT.split(sentence.group(1))]
        if all(items):
            roles.extend(m.group(1) for m in items if m)
    return tuple(roles)


def _order(cid: int, phrase: str) -> str:
    return hashlib.sha256(f"chebi-role:{cid}:{phrase}".encode()).hexdigest()


def role_options(cid: int, description: str) -> tuple[tuple[str, ...], str] | None:
    """(choices, gold letter) for one molecule, or None when it has no allow-listed role / too few distractors."""
    listed = parse_roles(description)
    gold_pool = sorted({r for r in listed if r in ROLE_FAMILY})
    if not gold_pool:
        return None
    gold = min(gold_pool, key=lambda r: _order(cid, r))
    used = {ROLE_FAMILY[r] for r in gold_pool}.union(*(ROLE_RELATED_ONLY.get(r, ()) for r in listed))
    text = description.casefold()
    pool = []
    for family, phrases in ROLE_FAMILIES.items():
        free = [p for p in phrases if p not in listed and p.casefold() not in text]
        if family not in used and free:
            pool.append(min(free, key=lambda p: _order(cid, p)))
    if len(pool) < ROLE_OPTIONS - 1:
        return None
    return program_mcq(gold, pool, key=f"chebi-role:{cid}", n_options=ROLE_OPTIONS)


def iter_chebi_role(repo_dir: Path) -> Iterator[Candidate]:
    root = Path(repo_dir)
    with zipfile.ZipFile(root / IMAGE_ZIP) as archive, open(root / TRAIN_CSV, encoding="utf-8", newline="") as handle:
        names = set(archive.namelist())
        for row in csv.DictReader(handle):
            cid = int(row["CID"])
            member = f"image/CID_{cid}.png"
            # ponytail: SMILES-token proxy for RDKit heavy atoms (equal on all but 21 train rows, none across 60)
            if member not in names or len(_SMILES_HEAVY_ATOM.findall(row["SMILES"])) > MAX_HEAVY_ATOMS:
                continue
            options = role_options(cid, row["description"])
            if options is None:
                continue
            choices, gold = options
            yield Candidate(
                source=SOURCE_CHEBI_ROLE,
                native_id=f"cid:{cid}",
                subject="Pharmacy",
                question=ROLE_QUESTION,
                choices=choices,
                answer_type="mcq",
                gold=gold,
                images=(archive.read(member),),
                family_key=f"cid:{cid}",
            )


SPECS: dict[str, SourceSpec] = {
    SOURCE_CHEBI: SourceSpec(
        source=SOURCE_CHEBI,
        repo="liupf/ChEBI-20-MM",
        revision="7d44959bad65cd31e481a7561dac25a513e8957b",
        license="MIT (ChEBI CC-BY-4.0, PubChem public domain)",
        label_provenance="PROGRAM",
        upstream_origin="Labels computed by RDKit from curated ChEBI/PubChem SMILES; image from ChEBI-20-MM",
        family="molecule",
    ),
    SOURCE_CHEBI_ROLE: SourceSpec(
        source=SOURCE_CHEBI_ROLE,
        repo="liupf/ChEBI-20-MM",
        revision="7d44959bad65cd31e481a7561dac25a513e8957b",
        license="MIT (ChEBI CC-BY-4.0, PubChem public domain)",
        label_provenance="HUMAN",
        upstream_origin="ChEBI curators' has_role annotations, quoted from the ChEBI-20-MM description; image from ChEBI-20-MM",
        family="molecule_role",
    ),
}

ITERATORS: dict[str, Callable[[Path], Iterator[Candidate]]] = {
    SOURCE_CHEBI: iter_chebi,
    SOURCE_CHEBI_ROLE: iter_chebi_role,
}
