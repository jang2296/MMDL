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
EXECUTION_PROFILES = {
    "mmmu-val-v1": EXECUTION,
    "mmmu-val-fast-transformers-v1": EXECUTION | {"sdpa_kernel": "auto"},
    "mmmu-val-fast-vllm-32k-v1": dict(backend="vllm", batch_size=2, attention="vllm",
                                  sdpa_kernel="not_applicable", deterministic=False),
    "mmmu-val-fast-vllm-32k-continuous-v1": dict(backend="vllm", batch_size=2, attention="vllm",
                                  sdpa_kernel="not_applicable", deterministic=False,
                                  scheduling="continuous"),
}
CONTROL_IMAGES = {
    "mmmu-val-control-a-v2": dict(min_pixels=262144, max_pixels=1310720,
                                  resize_owner="official_processor", preprocessing="hf_processor"),
    "mmmu-val-control-b-v2": dict(min_pixels=262144, max_pixels=1310720,
                                  resize_owner="qwen_vl_utils", preprocessing="qwen_vl_utils_0_0_14"),
    "mmmu-val-control-c-v2": dict(min_pixels=1003520, max_pixels=4014080,
                                  resize_owner="qwen_vl_utils", preprocessing="qwen_vl_utils_0_0_14"),
    "mmmu-val-official-vllm-b2-v2": dict(min_pixels=1003520, max_pixels=4014080,
                                         resize_owner="qwen_vl_utils", preprocessing="qwen_vl_utils_0_0_14"),
}
for _protocol in CONTROL_IMAGES:
    EXECUTION_PROFILES[_protocol] = dict(backend="vllm", batch_size=1, attention="vllm",
                                       sdpa_kernel="not_applicable", deterministic=False,
                                       scheduling="continuous", max_model_len=40960)
EXECUTION_PROFILES["mmmu-val-official-vllm-b2-v2"]["batch_size"] = 2
CONTROL_IMAGES["mmmu-val-v8"] = dict(CONTROL_IMAGES["mmmu-val-official-vllm-b2-v2"])
EXECUTION_PROFILES["mmmu-val-v8"] = dict(EXECUTION_PROFILES["mmmu-val-official-vllm-b2-v2"])
# New profiles carry their own engine controls; historical configs remain byte-for-byte valid.
SUBMISSION_PROFILES = {
    "mmmu-val-v18b-ft-a100-v1": ("mmmu-val", 9, 5, "a100"),
    "mmmu-val-v18b-ft-4090-v1": ("mmmu-val", 1, 5, "4090"),
    "mmmu-pro-standard10-v8-a100-v1": ("mmmu-pro", 9, 7, "a100"),
    "mmmu-pro-standard10-v8-4090-v1": ("mmmu-pro", 1, 7, "4090"),
}
for _protocol, (_suite, _slots, _images, _hardware) in SUBMISSION_PROFILES.items():
    CONTROL_IMAGES[_protocol] = dict(CONTROL_IMAGES["mmmu-val-v8"])
    EXECUTION_PROFILES[_protocol] = dict(EXECUTION_PROFILES["mmmu-val-v8"]) | {
        "batch_size": _slots, "max_images_per_prompt": _images, "enforce_eager": False,
        "gpu_memory_utilization": 0.90, "kv_safety_fraction": 0.90,
        "logprobs_mode": "raw_logprobs",
    }
# MMMU test (10,500): its own frozen profile, registered after the loop so earlier entries stay unchanged.
# Low-resolution images, a 2,048-token cap and measured-KV admission instead of fixed full-context slots.
MMMU_TEST_PROFILE = "mmmu-test-lowres2k-v8-v1"
TEST_ANSWER_SOURCE = dict(
    url="https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/"
        "bc168a9119d986d7cdf1e07b1eeb96ed3e8f92fa/mmmu/answer_dict_test.json",
    sha256="45b639a937017d3d71f81f610df3a1471debe1887c5c1d24644fec1b33059d5c",
    commit="bc168a9119d986d7cdf1e07b1eeb96ed3e8f92fa")
SUBMISSION_PROFILES[MMMU_TEST_PROFILE] = ("mmmu-test", 64, 7, "any")
CONTROL_IMAGES[MMMU_TEST_PROFILE] = dict(CONTROL_IMAGES["mmmu-val-control-b-v2"])
EXECUTION_PROFILES[MMMU_TEST_PROFILE] = dict(EXECUTION_PROFILES["mmmu-pro-standard10-v8-a100-v1"]) | {
    "batch_size": 64, "max_model_len": 16384, "admission": "kv_reservation"}
PROFILE_MAX_NEW_TOKENS = {MMMU_TEST_PROFILE: 2048}
SUITE_COUNTS = {"mmmu-val": 900, "mmmu-pro": 1730, "mmmu-test": 10500}
SUITE_DATA_DIRS = {"mmmu-val": "evaluation/mmmu", "mmmu-pro": "evaluation/mmmu_pro",
                   "mmmu-test": "evaluation/mmmu_test"}


def suite_of(cfg):
    """The evaluation suite a protocol config names (dataset id + split)."""
    dataset = cfg.get("dataset", {})
    if dataset.get("id") == PRO_ID:
        return "mmmu-pro"
    return "mmmu-test" if dataset.get("split") == "test" else "mmmu-val"


def validate_profile_model(cfg, kind):
    """Keep historical base runs separate from nonmerged FT and Pro profiles."""
    pid = cfg.get("protocol_id")
    if kind == "unmerged-lora" and pid not in SUBMISSION_PROFILES:
        raise ValueError("Unmerged LoRA requires a named submission profile")
    if pid in SUBMISSION_PROFILES:
        if kind not in {"base", "unmerged-lora"}:
            raise ValueError("Submission profiles support base or unmerged-lora models")
        if SUBMISSION_PROFILES[pid][0] == "mmmu-val" and kind != "unmerged-lora":
            raise ValueError("The historical FT MMMU profile requires unmerged-lora")


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
    profile = SUBMISSION_PROFILES.get(cfg.get("protocol_id"))
    dataset = (dict(id=PRO_ID, revision=PRO_REVISION, config="standard (10 options)",
                    split="test", subjects=SUBJECTS, expected_total=1730)
               if profile and profile[0] == "mmmu-pro" else
               dict(id=DATA_ID, revision=DATA_REVISION, split="test", subjects=SUBJECTS,
                    expected_total=10500, answer_source=TEST_ANSWER_SOURCE)
               if profile and profile[0] == "mmmu-test" else
               dict(id=DATA_ID, revision=DATA_REVISION, split="validation",
                    subjects=SUBJECTS, expected_total=900, per_subject=30))
    if cfg["dataset"] != dataset:
        raise ValueError("Dataset must match the named pinned suite and its full coverage")
    if set(cfg["generation"]) != set(RECIPE) | {"max_new_tokens"}:
        raise ValueError("Unknown or missing generation controls")
    if any(cfg["generation"].get(k) != v for k, v in RECIPE.items()):
        raise ValueError("The Qwen Instruct recipe and team seed policy are fixed")
    if cfg["execution"] != EXECUTION_PROFILES.get(cfg.get("protocol_id")) or cfg["prompt_policy"] != "P0":
        raise ValueError("Execution must match a named fixed protocol; P0 is unchanged")
    expected_parser = "team-final-answer-v4" if cfg["protocol_id"] in CONTROL_IMAGES else "mmmu-official-no-random-v1"
    if cfg["protocol_id"] == "mmmu-val-v8" or profile:
        expected_parser = "team-final-answer-v8"
    if cfg["parser"] != expected_parser:
        raise ValueError("Unknown parser")
    if cfg["status"] not in {"DRAFT", "FROZEN"}:
        raise ValueError("Invalid protocol status")
    if not isinstance(cfg["generation"]["max_new_tokens"], int) or not 1 <= cfg["generation"]["max_new_tokens"] <= 32768:
        raise ValueError("Invalid generation budget")
    image = cfg["image"]
    if cfg["protocol_id"] in CONTROL_IMAGES:
        if (image != CONTROL_IMAGES[cfg["protocol_id"]]
                or cfg["generation"]["max_new_tokens"] != PROFILE_MAX_NEW_TOKENS.get(cfg["protocol_id"], 32768)):
            raise ValueError("Controlled image protocol and output budget are fixed")
    elif set(image) != {"min_pixels", "max_pixels", "resize_owner"} or image["resize_owner"] != "official_processor":
        raise ValueError("Only the official processor may resize images")
    if not 1024 <= image["min_pixels"] <= image["max_pixels"]:
        raise ValueError("Invalid pixel budget")
    if cfg["protocol_id"].startswith("mmmu-val-fast-"):
        output_limit = 32768 if cfg["execution"]["backend"] == "vllm" else 2048
        if cfg["generation"]["max_new_tokens"] != output_limit or image != dict(
                min_pixels=262144, max_pixels=1310720, resize_owner="official_processor"):
            raise ValueError("Named acceleration protocol length and image budgets are fixed")
    if profile:
        if profile[3] == "a100" and (hw["name"] not in {"a100_sxm_80gb", "a100_pcie_80gb"}
                                      or hw["expected_vram_gib"] != 80):
            raise ValueError("The nine-slot submission profile requires A100 80GB")
        if profile[3] == "4090" and (hw["name"] != "rtx4090_24gb" or hw["expected_vram_gib"] != 24):
            raise ValueError("The 4090 profile requires its separately registered hardware")
    if set(hw) != HARDWARE_KEYS:
        raise ValueError("Hardware profile must contain placement controls only")
    if hw["placement"] not in {"gpu_only", "cpu_offload"} or hw["allow_disk_offload"] is not False:
        raise ValueError("Unsupported placement or disk offload")
    if cfg["execution"]["backend"] == "vllm" and hw["placement"] != "gpu_only":
        raise ValueError("The vLLM protocol requires GPU-only BF16 placement")
    if hw["gpu_index"] != 0 or hw["num_workers"] != 0:
        raise ValueError("Reference uses one visible GPU and sequential data loading")
    for key in ("expected_vram_gib", "gpu_weight_cap_gib", "gpu_reserve_gib",
                "cpu_weight_cap_gib", "min_free_gpu_gib"):
        if not isinstance(hw[key], (int, float)) or hw[key] <= 0:
            raise ValueError(f"Invalid hardware budget: {key}")
    if not 0 < hw["cpu_available_fraction"] < 1:
        raise ValueError("RAM fraction must leave operating headroom")
