from __future__ import annotations

import hashlib
import math
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from mmdl.evaluation.CompassVerifier_checker.backend import (
    FakeJudge,
    check_mode,
    load_judge,
    load_tokenizer,
)
from mmdl.evaluation.CompassVerifier_checker.prompt import CV_PROMPT, CV_PROMPT_SHA256, build


MODEL = "opencompass/CompassVerifier-3B"
REVISION = "27d6d660bdf6c2b8a9f45906d4b76c508af65ab0"


class _Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))
        self.to = Mock(return_value=self)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if index != (0, -1):
                raise AssertionError("only final input position may be read")
            return self
        return SimpleNamespace(item=lambda: self.values[index])

    def float(self):
        return self

    def argmax(self):
        return SimpleNamespace(item=lambda: max(range(len(self.values)), key=self.values.__getitem__))


def _softmax(tensor, *, dim):
    if dim != -1:
        raise AssertionError("softmax must cover the vocabulary dimension")
    values = [math.exp(value - max(tensor.values)) for value in tensor.values]
    return _Tensor([value / sum(values) for value in values])


class BackendTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "checker_v1.yaml"
        self.config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.tokenizer = Mock()
        self.tokenizer.encode.side_effect = lambda text, **_: ["ABC".index(text)]
        self.tokenizer.decode.side_effect = lambda ids, **_: ["A", "B", "C", "other"][ids[0]]
        self.tokenizer.apply_chat_template.return_value = "chat input"
        self.input_ids = _Tensor([7, 8])
        self.tokenizer.return_value = {"input_ids": self.input_ids}
        self.model = Mock()
        self.model.config = SimpleNamespace(
            model_type="qwen2", max_position_embeddings=8, vocab_size=4, _commit_hash=REVISION
        )
        self.model.dtype = "bf16"
        self.model.is_quantized = False
        self.model.is_loaded_in_4bit = False
        self.model.return_value = SimpleNamespace(logits=_Tensor([0.0, 0.0, 0.0, 10.0]))
        self.torch = SimpleNamespace(
            __version__="mock-torch",
            bfloat16="bf16",
            cuda=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 1,
                get_device_name=lambda _: "mock-gpu",
            ),
            inference_mode=Mock(side_effect=nullcontext),
            softmax=Mock(side_effect=_softmax),
            isfinite=lambda logits: SimpleNamespace(
                all=lambda: SimpleNamespace(item=lambda: all(map(math.isfinite, logits.values)))
            ),
        )
        self.transformers = SimpleNamespace(
            __version__="mock-transformers",
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=self.model)),
            BitsAndBytesConfig=Mock(side_effect=lambda **kwargs: kwargs),
        )
        self.modules = patch.dict(
            "sys.modules", {"torch": self.torch, "transformers": self.transformers}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def test_prompt_and_config_share_the_exact_pinned_hash(self):
        self.assertEqual(hashlib.sha256(CV_PROMPT.encode()).hexdigest(), CV_PROMPT_SHA256)
        self.assertEqual(self.config["judge"]["prompt"]["text_sha256"], CV_PROMPT_SHA256)
        self.assertEqual(
            build("Q {untouched}", "gold", "response"),
            CV_PROMPT.format(question="Q {untouched}", gold_answer="gold", llm_response="response"),
        )

    def test_fake_tokens_are_stable_across_instances_and_call_order(self):
        script = {"hello world": {"p_a": 0.95, "p_b": 0.02, "p_c": 0.01, "top_token_id": 0}}
        first, second = FakeJudge(script), FakeJudge(script)
        second.encode("unrelated tokens")
        text = "A coffee 한글 world"
        self.assertEqual(first.encode(text), second.encode(text))
        self.assertEqual(first.decode(first.encode(text)), text)
        self.assertEqual(first.probabilities("hello world")["top_token"], "A")
        self.assertEqual(first.probabilities("hello world")["input_tokens"], 2)
        self.assertEqual(script["hello world"].keys(), {"p_a", "p_b", "p_c", "top_token_id"})
        self.assertEqual(first.identity["model"], "FakeJudge")

    def test_mode_requires_exact_official_identity_and_pinned_smoke(self):
        official = self.config["judge"]["official"]
        self.assertTrue(check_mode("official", official["model"], official["revision"], "none", self.config))
        self.assertFalse(check_mode("smoke", MODEL, REVISION, "nf4", self.config))
        for mode, model, revision, quantization in [
            ("official", MODEL, REVISION, "none"),
            ("official", official["model"], "main", "none"),
            ("official", official["model"], official["revision"], "nf4"),
            ("smoke", MODEL, "main", "none"),
            ("smoke", MODEL, REVISION, "int8"),
            ("unknown", MODEL, REVISION, "none"),
        ]:
            with self.subTest(mode=mode, model=model, quantization=quantization):
                with self.assertRaises(ValueError):
                    check_mode(mode, model, revision, quantization, self.config)

    def test_offline_bf16_load_and_full_vocabulary_single_forward(self):
        judge = load_judge(MODEL, REVISION)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False
        )
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            MODEL, revision=REVISION, torch_dtype="bf16", device_map={"": 0},
            local_files_only=True, trust_remote_code=False,
        )
        result = judge.probabilities("synthetic prompt")
        self.assertLess(result["p_a"] + result["p_b"] + result["p_c"], 0.001)
        self.assertEqual(result["top_token_id"], 3)
        self.assertEqual(result["top_token"], "other")
        self.assertEqual(result["input_tokens"], 2)
        self.model.assert_called_once_with(
            input_ids=self.input_ids, use_cache=False, logits_to_keep=1
        )
        self.model.generate.assert_not_called()
        self.model.eval.assert_called_once_with()
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "synthetic prompt"}],
            tokenize=False, add_generation_prompt=True,
        )
        self.tokenizer.assert_called_once_with(
            "chat input", return_tensors="pt", add_special_tokens=False, truncation=False
        )
        self.assertEqual(judge.identity["torch_version"], "mock-torch")
        self.assertEqual(judge.identity["gpu_name"], "mock-gpu")

    def test_nf4_is_explicit_and_never_silent(self):
        self.model.is_loaded_in_4bit = True
        judge = load_judge(MODEL, REVISION, quantization="nf4")
        self.transformers.BitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bf16",
        )
        self.assertEqual(judge.identity["quantization"], "nf4")
        self.model.is_loaded_in_4bit = False
        with self.assertRaisesRegex(ValueError, "actually be loaded in 4 bits"):
            load_judge(MODEL, REVISION, quantization="nf4")

    def test_bad_labels_fail_before_loading_weights(self):
        for ids in ([0, 1], []):
            self.tokenizer.encode.return_value = ids
            self.tokenizer.encode.side_effect = None
            with self.assertRaisesRegex(ValueError, "exactly one token"):
                load_judge(MODEL, REVISION)
        self.tokenizer.encode.return_value = [0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            load_judge(MODEL, REVISION)
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_context_overflow_and_nonfinite_logits_fail_without_retry(self):
        judge = load_judge(MODEL, REVISION)
        self.model.config.max_position_embeddings = 1
        with self.assertRaisesRegex(ValueError, "context limit"):
            judge.probabilities("synthetic prompt")
        self.model.assert_not_called()
        self.model.config.max_position_embeddings = 8
        self.model.return_value.logits = _Tensor([0.0, float("nan"), 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            judge.probabilities("synthetic prompt")
        self.assertEqual(self.model.call_count, 1)

    def test_no_cpu_fallback_and_no_unpinned_tokenizer(self):
        self.torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(ValueError, "no CPU fallback"):
            load_judge(MODEL, REVISION)
        with self.assertRaisesRegex(ValueError, "pinned allowlist"):
            load_tokenizer(MODEL, "main")
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
