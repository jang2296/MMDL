import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from mmdl.evaluation.engine import run, validate_batch_outputs
from mmdl.runtime.artifacts import read_json
from mmdl.runtime.contracts import load_configs, validate_configs

ROOT = Path(__file__).resolve().parents[2]


class FastProtocolTests(unittest.TestCase):
    def test_named_protocols_preserve_evaluation_and_reject_overrides(self):
        reference, _ = load_configs(ROOT / "configs/eval/mmmu_val_v1.yaml",
                                    ROOT / "configs/hardware/rtx3090_24gb.yaml")
        for backend in ("transformers", "vllm"):
            cfg, hw = load_configs(ROOT / f"configs/eval/mmmu_val_fast_{backend}_v1.yaml",
                                   ROOT / "configs/hardware/rtx3090_24gb.yaml")
            for key in ("model", "dataset", "image", "prompt_policy", "parser"):
                self.assertEqual(reference[key], cfg[key])
            self.assertEqual(cfg["generation"], reference["generation"] | {
                "max_new_tokens": 32768 if backend == "vllm" else 2048})
            for section, key, value in (("execution", "batch_size", 4),
                                        ("generation", "max_new_tokens", 1024),
                                        ("image", "max_pixels", 262144)):
                changed = copy.deepcopy(cfg)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    validate_configs(changed, hw)
        with self.assertRaises(ValueError):
            load_configs(ROOT / "configs/eval/mmmu_val_fast_vllm_v1.yaml",
                         ROOT / "configs/hardware/rtx5060_8gb.yaml")

    def test_batch_outputs_must_preserve_every_sample_and_order(self):
        requests = [{"sample_id": "one"}, {"sample_id": "two"}]
        validate_batch_outputs(requests, requests)
        for bad in (requests[:1], requests[::-1], [{"sample_id": "one"}] * 2, [{}, {}]):
            with self.assertRaises(ValueError):
                validate_batch_outputs(requests, bad)

    def test_mocked_vllm_batch_runs_through_dispatch_save_and_finalize(self):
        class Dataset(list):
            def __getitem__(self, key):
                if key == "id":
                    return [row["id"] for row in self]
                return super().__getitem__(key)

        class Backend:
            load_seconds = 0.1

            def __init__(self, *_):
                pass

            def describe(self):
                return {"device_map": {"": "cuda:0"}, "generation_config": {"fixture": True},
                        "image_processor": {"fixture": True}, "chat_template_sha256": "0" * 64}

            def generate_batch(self, requests):
                self.test_case.assertEqual([r["sample_id"] for r in requests], [
                    "validation_Accounting_1", "validation_Accounting_2"])
                return [{"sample_id": request["sample_id"], "raw_response": "A",
                         "generated_token_ids": [1], "generated_tokens": 1, "input_tokens": 2,
                         "image_grid_thw": [[1, 1, 1]], "input_tensors": {"fixture": request["sample_id"]},
                         "input_sha256": request["sample_id"], "chat_prompt": request["sample_id"],
                         "generation_seconds": 0.1, "finish_reason": "stop",
                         "peak_vram_allocated_bytes": None, "peak_vram_reserved_bytes": None,
                         "observed_device_memory_used_bytes": 123}
                        for request in requests]

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data, artifacts, public = base / "data", base / "artifacts", base / "public"
            data.mkdir()
            (data / "manifest.json").write_text(json.dumps({"files": []}))
            image = Image.new("RGB", (2, 2), "white")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            rows = Dataset([{
                "id": f"validation_Accounting_{index}", "question_type": "multiple-choice",
                "question": f"fixture {index}", "options": ["yes", "no"], "answer": "A",
                "explanation": None, "image_1": {"bytes": buffer.getvalue()},
            } for index in (1, 2)])
            cfg, hw = load_configs(ROOT / "configs/eval/mmmu_val_fast_vllm_v1.yaml",
                                   ROOT / "configs/hardware/rtx3090_24gb.yaml")
            args = SimpleNamespace(
                protocol=ROOT / "configs/eval/mmmu_val_fast_vllm_v1.yaml",
                require_commit="a" * 40, model_ref=ROOT / "manifests/models/baseline.json",
                model_path=base / "model", base_path=base / "model", data_root=data,
                artifact_root=artifacts, public_root=public, mode="smoke", subject=None, limit=None,
                sample_id=["validation_Accounting_1", "validation_Accounting_2"], run_id="smoke-fast",
                resume=False, job_id="job", run_role="smoke",
            )
            environment = {
                "python": "3.12", "platform": "fixture",
                "torch": {"cuda_runtime": "fixture", "bf16_probe": {"passed": True}},
                "nvidia_smi": {"gpus": []},
            }
            Backend.test_case = self
            with patch("mmdl.evaluation.engine.git_commit", return_value="a" * 40), \
                    patch("mmdl.evaluation.engine.collect_environment", return_value=environment), \
                    patch("mmdl.evaluation.engine.check_storage", return_value={"sufficient": True}), \
                    patch("mmdl.evaluation.engine.verify_model"), \
                    patch("mmdl.evaluation.engine.load_validation",
                          return_value=({"Accounting": rows}, {"total": 2})), \
                    patch("mmdl.evaluation.engine.code_records", return_value=[]), \
                    patch("mmdl.evaluation.backends.vllm_backend.VLLMBackend", Backend):
                summary = run(args, cfg, hw, ROOT)
            self.assertEqual(summary["status"], "SMOKE")
            self.assertEqual(summary["completed_count"], 2)
            samples = artifacts / "runs/smoke-fast/samples"
            self.assertEqual({read_json(path)["id"] for path in samples.glob("*.json")}, {
                "validation_Accounting_1", "validation_Accounting_2"})


if __name__ == "__main__":
    unittest.main()
