import copy
import unittest
from pathlib import Path

import torch
from transformers import GenerationConfig, LogitsProcessorList, PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import TemperatureLogitsWarper

from mmdl.runtime.contracts import load_configs, sample_seed, validate_configs
from mmdl.runtime.generation import GeneratedOnlyPresencePenalty
from mmdl.runtime.placement import placement_kwargs

ROOT = Path(__file__).resolve().parents[2]


class RuntimeTests(unittest.TestCase):
    def test_transformers_backend_has_no_per_token_streamer(self):
        source = (ROOT / "src/mmdl/evaluation/backends/transformers_backend.py").read_text()
        self.assertNotIn("streamer=", source)
        self.assertNotIn("BaseStreamer", source)

    def test_protocol_and_hardware_boundaries(self):
        cfg, hw = load_configs(ROOT / "configs/eval/mmmu_val_v1.yaml",
                               ROOT / "configs/hardware/rtx5060_8gb.yaml")
        for key, value in [("dtype", "float16"), ("batch_size", 2), ("generation", {})]:
            altered = hw | {key: value}
            with self.assertRaises(ValueError):
                validate_configs(cfg, altered)
        for section, key, value in [("model", "dtype", "float16"),
                                    ("dataset", "split", "test"),
                                    ("generation", "do_sample", False),
                                    ("execution", "batch_size", 2)]:
            altered = copy.deepcopy(cfg)
            altered[section][key] = value
            with self.assertRaises(ValueError):
                validate_configs(altered, hw)

    def test_seed_and_placement(self):
        self.assertEqual(sample_seed(3407, "one"), sample_seed(3407, "one"))
        self.assertNotEqual(sample_seed(3407, "one"), sample_seed(3407, "two"))
        _, hw = load_configs(ROOT / "configs/eval/mmmu_val_v1.yaml",
                            ROOT / "configs/hardware/rtx5060_8gb.yaml")
        placement = placement_kwargs(hw, 7 * 1024**3, 10 * 1024**3)
        self.assertEqual(placement["max_memory"][0], 5 * 1024**3)
        self.assertEqual(placement["max_memory"]["cpu"], int(6.5 * 1024**3))
        with self.assertRaises(RuntimeError):
            placement_kwargs(hw, 2 * 1024**3, 10 * 1024**3)

    def test_control_protocols_freeze_sampling_scoring_and_image_paths(self):
        configs = []
        for arm in ("a", "b", "c"):
            cfg, hw = load_configs(ROOT / f"configs/eval/mmmu_val_control_{arm}_v2.yaml",
                                   ROOT / "configs/hardware/rtx3090_24gb.yaml")
            configs.append(cfg)
            for section, key, value in [("image", "max_pixels", 42),
                                        ("generation", "max_new_tokens", 2048),
                                        ("execution", "max_model_len", 36864),
                                        ("execution", "batch_size", 2)]:
                changed = copy.deepcopy(cfg)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    validate_configs(changed, hw)
            changed = copy.deepcopy(cfg)
            changed["parser"] = "mmmu-official-no-random-v1"
            with self.assertRaises(ValueError):
                validate_configs(changed, hw)
        for key in ("model", "dataset", "prompt_policy", "parser", "generation", "execution"):
            self.assertEqual(configs[0][key], configs[1][key])
            self.assertEqual(configs[1][key], configs[2][key])

        b2, hw = load_configs(ROOT / "configs/eval/mmmu_val_official_vllm_b2_v2.yaml",
                              ROOT / "configs/hardware/rtx3090_24gb.yaml")
        self.assertEqual(b2["protocol_id"], "mmmu-val-official-vllm-b2-v2")
        self.assertEqual(b2["execution"]["batch_size"], 2)
        comparable = copy.deepcopy(b2)
        comparable["protocol_id"] = configs[2]["protocol_id"]
        comparable["execution"]["batch_size"] = configs[2]["execution"]["batch_size"]
        self.assertEqual(comparable, configs[2])

        old_c, old_hw = load_configs(ROOT / "configs/eval/mmmu_val_control_c_v2.yaml",
                                     ROOT / "configs/hardware/rtx3090_24gb.yaml")
        changed = copy.deepcopy(old_c)
        changed["execution"]["batch_size"] = 2
        with self.assertRaises(ValueError):
            validate_configs(changed, old_hw)
        for section, key, value in [("generation", "max_new_tokens", 2048),
                                    ("image", "max_pixels", 1310720)]:
            changed = copy.deepcopy(b2)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate_configs(changed, hw)

    def test_presence_penalty_excludes_prompt_and_counts_presence_once(self):
        penalty = GeneratedOnlyPresencePenalty(2, 1.5)
        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        self.assertTrue(torch.equal(penalty(torch.tensor([[1, 2]]), scores), scores))
        actual = penalty(torch.tensor([[1, 2, 3, 3]]), scores)
        self.assertTrue(torch.equal(actual, torch.tensor([[1.0, 2.0, 3.0, 2.5]])))

    def test_transformers_orders_custom_penalty_before_sampling_filters(self):
        # Exercise the installed generation dispatch without allocating model weights.
        model = GenerationMixin()
        model.config = PretrainedConfig()
        generation = GenerationConfig(do_sample=True, temperature=0.7, top_k=20, top_p=0.8)
        generation._eos_token_tensor = None
        generation._pad_token_tensor = None
        penalty = GeneratedOnlyPresencePenalty(2, 1.5)
        processors = model._get_logits_processor(generation, 2, None, None,
                                                 LogitsProcessorList([penalty]), device="cpu")
        self.assertEqual(sum(isinstance(p, GeneratedOnlyPresencePenalty) for p in processors), 1)
        temperature_index = next(i for i, p in enumerate(processors) if isinstance(p, TemperatureLogitsWarper))
        self.assertLess(processors.index(penalty), temperature_index)


if __name__ == "__main__":
    unittest.main()
