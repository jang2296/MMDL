import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from mmdl.evaluation import backend_parity
from mmdl.evaluation.backends.vllm_backend import VLLMBackend, allocated_kv_capacity
from mmdl.runtime.artifacts import digest, sha256_file
from mmdl.runtime.contracts import load_configs, validate_configs, validate_profile_model

ROOT = Path(__file__).resolve().parents[2]


def scheduler(blocks=30000, size=16):
    cache = NS(num_blocks=blocks, kv_cache_groups=[NS(kv_cache_spec=NS(block_size=size))])
    return NS(engine_core=NS(engine_core=NS(scheduler=NS(kv_cache_config=cache))))


class SubmissionBackendTests(unittest.TestCase):
    def test_profiles_are_scoped_and_baseline_stays_fixed(self):
        baseline, hw = load_configs(ROOT / "configs/eval/mmmu_val_v8.yaml",
                                    ROOT / "configs/hardware/rtx4090_24gb.yaml")
        self.assertNotIn("max_images_per_prompt", baseline["execution"])
        changed = copy.deepcopy(baseline)
        changed["execution"]["batch_size"] = 9
        with self.assertRaises(ValueError):
            validate_configs(changed, hw)
        for profile, hardware in (("mmmu_pro_standard10_v8_a100", "a100_sxm_80gb"),
                                  ("mmmu_pro_standard10_v8_4090", "rtx4090_24gb"),
                                  ("mmmu_val_v18b_ft_a100", "a100_sxm_80gb"),
                                  ("mmmu_val_v18b_ft_4090", "rtx4090_24gb")):
            cfg, target = load_configs(ROOT / f"configs/eval/{profile}.yaml",
                                       ROOT / f"configs/hardware/{hardware}.yaml")
            validate_profile_model(cfg, "unmerged-lora")
            if "pro" in profile:
                validate_profile_model(cfg, "base")
                self.assertEqual(cfg["dataset"]["expected_total"], 1730)
                self.assertNotIn("per_subject", cfg["dataset"])
                self.assertEqual(cfg["execution"]["max_images_per_prompt"], 7)
            else:
                with self.assertRaises(ValueError):
                    validate_profile_model(cfg, "base")
            cfg["execution"]["enforce_eager"] = True
            with self.assertRaises(ValueError):
                validate_configs(cfg, target)
        with self.assertRaises(ValueError):
            validate_profile_model(baseline, "unmerged-lora")

    def test_actual_kv_rounds_full_context_and_never_estimates_vram(self):
        value = allocated_kv_capacity(scheduler(), "0.11.0", max_model_len=40960, slots=9)
        self.assertEqual(value["required_tokens"], 9 * 40960)
        self.assertEqual(value["actual_kv_token_capacity"], 480000)
        self.assertTrue(value["pass"])
        for engine in (scheduler(blocks=25000), NS()):
            with self.assertRaises(ValueError):
                allocated_kv_capacity(engine, "0.11.0", max_model_len=40960, slots=9)
        with self.assertRaises(ValueError):
            allocated_kv_capacity(scheduler(), "0.11.0", max_model_len=40960, slots=1, safety_fraction=.91)

    def test_pro_constructor_forwards_graph_seven_images_slots_and_live_kv(self):
        cfg, hw = load_configs(ROOT / "configs/eval/mmmu_pro_standard10_v8_a100.yaml",
                              ROOT / "configs/hardware/a100_sxm_80gb.yaml")
        captured = {}

        class LLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.llm_engine = scheduler()

            def get_default_sampling_params(self):
                return "defaults"

        processor = NS(image_processor=NS(size={}), tokenizer=NS(eos_token_id=2, pad_token_id=0))
        with patch("mmdl.evaluation.backends.vllm_backend.torch.cuda.is_available", return_value=True), \
                patch("mmdl.evaluation.backends.vllm_backend.torch.cuda.is_bf16_supported", return_value=True), \
                patch("mmdl.evaluation.backends.vllm_backend.AutoProcessor.from_pretrained", return_value=processor), \
                patch("mmdl.evaluation.backends.vllm_backend.AutoConfig.from_pretrained", return_value=NS()), \
                patch("mmdl.evaluation.backends.vllm_backend._vllm", return_value=(LLM, dict, "0.11.0")), \
                patch("mmdl.evaluation.backends.vllm_backend._final_only_output_kind", return_value="FINAL_ONLY"), \
                patch("fine_tuning.error_note.v17_eval_engine.observe_engine", return_value={"pass": True}), \
                patch.dict("os.environ", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}):
            backend = VLLMBackend("/synthetic", cfg, hw)
        self.assertEqual(captured["limit_mm_per_prompt"], {"image": 7, "video": 0})
        self.assertEqual(captured["max_num_seqs"], 9)
        self.assertFalse(captured["enforce_eager"])
        self.assertEqual(captured["logprobs_mode"], "raw_logprobs")
        self.assertTrue(backend.capacity_receipt["pass"])

    def test_image_cap_is_checked_before_processor_and_context_is_not_truncated(self):
        backend = object.__new__(VLLMBackend)
        backend.max_images_per_prompt = 7
        backend.cfg = {"generation": {"max_new_tokens": 32768}}
        backend.max_model_len = 40960
        backend.processor = backend.model_config = None
        backend._sampling_params = lambda seed, **kwargs: {"seed": seed}
        request = {"images": [object()] * 7, "messages": [], "seed": 1}
        prepared = {"input_tokens": 7381, "chat_prompt": "P0"}
        with patch("mmdl.evaluation.backends.vllm_backend.prepare_protocol_inputs",
                   return_value=(prepared, request["images"])) as prepare:
            item, prompt, _ = backend._prepare_request(request, final_only=False)
            self.assertEqual(len(prompt["multi_modal_data"]["image"]), 7)
            self.assertEqual(item["passed_image_count"], 7)
            with self.assertRaises(ValueError):
                backend._prepare_request(request | {"images": [object()] * 8}, final_only=False)
            self.assertEqual(prepare.call_count, 1)
            prepared["input_tokens"] = 8193
            with self.assertRaisesRegex(ValueError, "refusing truncation"):
                backend._prepare_request(request, final_only=False)

    def test_reference_sources_and_fixed_synthetic_prefixes(self):
        self.assertEqual(backend_parity.verify_reference_sources()["commit"],
                         "00fffac4f569dec3adbada1790af7b1af1b72c62")
        requests = backend_parity.synthetic_requests()
        self.assertEqual(len(requests), 16)
        self.assertEqual({len(r["images"]) for r in requests}, {1, 3, 5})
        self.assertEqual(len(backend_parity.synthetic_requests(seven=True)[0]["images"]), 7)
        self.assertEqual([r["seed"] for r in requests],
                         [r["seed"] for r in backend_parity.synthetic_requests()])

    def test_wsl_gpu_uuid_uses_unambiguous_nvidia_smi(self):
        properties = NS(uuid="N/A", name="RTX fixture", total_memory=24, major=9, minor=0)
        with patch.object(backend_parity.torch.cuda, "is_available", return_value=True), \
                patch.object(backend_parity.torch.cuda, "get_device_properties", return_value=properties), \
                patch.object(backend_parity.subprocess, "run", return_value=NS(stdout="GPU-fixture\n")), \
                patch.object(backend_parity, "version", return_value="pinned"):
            self.assertEqual(backend_parity.runtime_fingerprint()["gpu_uuid"], "GPU-fixture")

    def test_parity_uses_strict_existing_probability_threshold(self):
        rows = [dict(sample_id=f"s{i}", seed=i, input_sha256=str(i), token_ids=[7] * 32,
                     logprobs=[-2.] * 32, prompt_token_ids=[1], source_pixels=[],
                     cpu_processor_tensors={}, image_grid_thw=[], actual_vision_inputs=[])
                for i in range(16)]
        reference = dict(engine="reference", probabilities=rows, config={}, hardware={}, rehearsal=True,
                         model_kind="base", reference_source={}, candidate_sources={}, runtime={"gpu_uuid": "GPU-fixture"},
                         model_files_sha256={}, base_config_sha256="x")
        candidate = copy.deepcopy(reference) | {"engine": "submission"}
        self.assertTrue(backend_parity.compare(reference, candidate)["pass"])
        candidate["probabilities"][0]["logprobs"] = [-2.041] * 32
        self.assertFalse(backend_parity.compare(reference, candidate)["pass"])
        candidate["runtime"]["gpu_uuid"] = "GPU-other"
        with self.assertRaisesRegex(ValueError, "runtime"):
            backend_parity.compare(reference, candidate)
        candidate["runtime"]["gpu_uuid"] = "GPU-fixture"
        candidate["probabilities"][0]["prompt_token_ids"] = [2]
        with self.assertRaises(ValueError):
            backend_parity.compare(reference, candidate)

    @patch("mmdl.evaluation.backend_parity.runtime_fingerprint", return_value={"gpu_uuid": "GPU-fixture"})
    def test_admission_rejects_rehearsal_changed_weights_and_missing_seven_evidence(self, _runtime):
        cfg, hw = load_configs(ROOT / "configs/eval/mmmu_pro_standard10_v8_4090.yaml",
                              ROOT / "configs/hardware/rtx4090_24gb.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            path = root / "receipt.json"
            value = dict(version="submission-engine-parity-v1", config=cfg, hardware=hw, model_kind="base",
                         rehearsal=False, **{"pass": True}, reference_source=backend_parity.verify_reference_sources(),
                         candidate_sources=backend_parity.candidate_sources(), runtime={"gpu_uuid": "GPU-fixture"},
                         model_files_sha256={"config.json": sha256_file(model / "config.json")})

            def save():
                value.pop("content_sha256", None)
                value["content_sha256"] = digest(value)
                path.write_text(json.dumps(value))

            save()
            with self.assertRaisesRegex(ValueError, "Seven-image"):
                backend_parity.validate_admission(path, cfg, hw, model, "base")
            value["seven_image_acceptance"] = {"eight_rejected": True, "kv": {"pass": True},
                                                "actual_vision_inputs": [{"grid": [[1, 2, 2]] * 7}],
                                                "production_reservation_seven_maximum": {
                                                    "status": "CONTEXT_REJECTED_WITHOUT_TRUNCATION",
                                                    "reserved_output_tokens": 32768},
                                                "rows": [{"source_image_count": 7, "passed_image_count": 7}]}
            save()
            self.assertTrue(backend_parity.validate_admission(path, cfg, hw, model, "base")["pass"])
            value["rehearsal"] = True
            save()
            with self.assertRaisesRegex(ValueError, "non-rehearsal"):
                backend_parity.validate_admission(path, cfg, hw, model, "base")
            value["rehearsal"] = False
            save()
            (model / "config.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "file hashes"):
                backend_parity.validate_admission(path, cfg, hw, model, "base")

    @patch("mmdl.evaluation.backend_parity.runtime_fingerprint", return_value={"gpu_uuid": "GPU-fixture"})
    def test_automatic_admission_runs_isolated_stages_once_then_resumes(self, _runtime):
        cfg, hw = load_configs(ROOT / "configs/eval/mmmu_pro_standard10_v8_4090.yaml",
                              ROOT / "configs/hardware/rtx4090_24gb.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            args = NS(artifact_root=Path(temporary), run_id="synthetic", model_path=Path(temporary),
                      protocol=ROOT / "configs/eval/mmmu_pro_standard10_v8_4090.yaml",
                      hardware=ROOT / "configs/hardware/rtx4090_24gb.yaml", resume=False,
                      rehearsal=False, engine_validation=None)
            stages = []

            def process(argv, **kwargs):
                engine = argv[argv.index("--engine") + 1]
                stages.append(engine)
                Path(argv[argv.index("--output") + 1]).write_text(json.dumps(
                    {"status": "MEASURED", "config": cfg, "engine": engine}))
                return NS(returncode=0)

            with patch.object(backend_parity.subprocess, "run", side_effect=process), \
                    patch.object(backend_parity, "compare", return_value={"pass": True}), \
                    patch.object(backend_parity, "validate_admission", return_value={"pass": True}):
                self.assertTrue(backend_parity.ensure_admission(args, cfg, hw, {"kind": "base"})["pass"])
                self.assertEqual(stages, ["reference", "submission"])
                args.resume = True
                backend_parity.ensure_admission(args, cfg, hw, {"kind": "base"})
                self.assertEqual(stages, ["reference", "submission"])
                args.resume = False
                with self.assertRaises(FileExistsError):
                    backend_parity.ensure_admission(args, cfg, hw, {"kind": "base"})


if __name__ == "__main__":
    unittest.main()
