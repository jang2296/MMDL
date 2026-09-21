"""Fixed course requirements, separate from hardware placement."""

import hashlib
from pathlib import Path

import yaml

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATA_ID = "MMMU/MMMU"
DATA_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
PRO_ID = "MMMU/MMMU_Pro"
PRO_REVISION = "563f3e84bb3b90893083a1f039cfa13077f2302b"
SUBJECTS = (
    "Accounting Agriculture Architecture_and_Engineering Art Art_Theory "
    "Basic_Medical_Science Biology Chemistry Clinical_Medicine Computer_Science "
    "Design Diagnostics_and_Laboratory_Medicine Economics Electronics Energy_and_Power "
    "Finance Geography History Literature Manage Marketing Materials Math "
    "Mechanical_Engineering Music Pharmacy Physics Psychology Public_Health Sociology"
).split()
RECIPE = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
              repetition_penalty=1.0, presence_penalty=1.5, seed=3407,
              seed_policy="per_sample_sha256_v1", num_beams=1)
EXECUTION = dict(backend="transformers", batch_size=1, attention="sdpa",
                 sdpa_kernel="math", deterministic=True)
HARDWARE_KEYS = set("name placement gpu_index expected_vram_gib gpu_weight_cap_gib "
                    "gpu_reserve_gib cpu_weight_cap_gib cpu_available_fraction "
                    "min_free_gpu_gib allow_disk_offload num_workers".split())


def sample_seed(master_seed, sample_id):
    value = hashlib.sha256(f"{master_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(value[:8], "big") % (2**31)


def load_configs(protocol, hardware):
    cfg = yaml.safe_load(Path(protocol).read_text())
    hw = yaml.safe_load(Path(hardware).read_text())
    validate_configs(cfg, hw)
    return cfg, hw


def validate_configs(cfg, hw):
    if cfg["model"] != dict(id=MODEL_ID, revision=MODEL_REVISION, dtype="bfloat16"):
        raise ValueError("The official base model, revision and BF16 are fixed")
    if cfg["dataset"] != dict(id=DATA_ID, revision=DATA_REVISION, split="validation",
                              subjects=SUBJECTS, expected_total=900, per_subject=30):
        raise ValueError("Only pinned MMMU validation, all 30 subjects, is allowed")
    if set(cfg["generation"]) != set(RECIPE) | {"max_new_tokens"}:
        raise ValueError("Unknown or missing generation controls")
    if any(cfg["generation"].get(k) != v for k, v in RECIPE.items()):
        raise ValueError("The Qwen Instruct recipe and team seed policy are fixed")
    if cfg["execution"] != EXECUTION or cfg["prompt_policy"] != "P0":
        raise ValueError("Reference execution and P0 are fixed")
    if cfg["parser"] != "mmmu-official-no-random-v1":
        raise ValueError("Unknown parser")
    if cfg["status"] not in {"DRAFT", "FROZEN"}:
        raise ValueError("Invalid protocol status")
    if not isinstance(cfg["generation"]["max_new_tokens"], int) or not 1 <= cfg["generation"]["max_new_tokens"] <= 32768:
        raise ValueError("Invalid generation budget")
    image = cfg["image"]
    if set(image) != {"min_pixels", "max_pixels", "resize_owner"} or image["resize_owner"] != "official_processor":
        raise ValueError("Only the official processor may resize images")
    if not 1024 <= image["min_pixels"] <= image["max_pixels"]:
        raise ValueError("Invalid pixel budget")
    if set(hw) != HARDWARE_KEYS:
        raise ValueError("Hardware profile must contain placement controls only")
    if hw["placement"] not in {"gpu_only", "cpu_offload"} or hw["allow_disk_offload"] is not False:
        raise ValueError("Unsupported placement or disk offload")
    if hw["gpu_index"] != 0 or hw["num_workers"] != 0:
        raise ValueError("Reference uses one visible GPU and sequential data loading")
    for key in ("expected_vram_gib", "gpu_weight_cap_gib", "gpu_reserve_gib",
                "cpu_weight_cap_gib", "min_free_gpu_gib"):
        if not isinstance(hw[key], (int, float)) or hw[key] <= 0:
            raise ValueError(f"Invalid hardware budget: {key}")
    if not 0 < hw["cpu_available_fraction"] < 1:
        raise ValueError("RAM fraction must leave operating headroom")
