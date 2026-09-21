import copy
import contextlib
import io
import unittest
from pathlib import Path

import torch
from transformers import GenerationConfig, LogitsProcessorList, PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import TemperatureLogitsWarper

from mmdl.runtime.contracts import load_configs, sample_seed, validate_configs
from mmdl.runtime.generation import GeneratedOnlyPresencePenalty
from mmdl.runtime.placement import placement_kwargs
from mmdl.evaluation.backends.transformers_backend import TokenProgress

ROOT = Path(__file__).resolve().parents[2]


class RuntimeTests(unittest.TestCase):
    def test_progress_excludes_prompt_and_never_emits_token_values(self):
        stream = io.StringIO()
        progress = TokenProgress()
        with contextlib.redirect_stdout(stream):
            progress.put(torch.tensor([[98765, 98766]]))
            progress.put(torch.tensor([87654]))
            progress.put(torch.tensor([76543]))
            progress.end()
        self.assertEqual(progress.tokens, 2)
        self.assertIn('"generated_tokens": 1', stream.getvalue())
        self.assertNotIn("98765", stream.getvalue())
        self.assertNotIn("87654", stream.getvalue())

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
