import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mmdl.evaluation.backends.input_preparation import prepare_inputs
from mmdl.evaluation.backends.vllm_backend import MAX_MODEL_LEN, VLLMBackend


class _Processor:
    image_processor = SimpleNamespace(merge_size=1)
    tokenizer = SimpleNamespace(decode=lambda ids, **_: "|".join(map(str, ids)), eos_token_id=2, pad_token_id=0)

    def apply_chat_template(self, messages, **_):
        return messages[0]["content"][-1]["text"]

    def __call__(self, **_):
        return {
            "input_ids": torch.tensor([[7, 99, 99, 99, 99]], dtype=torch.int64),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
            "pixel_values": torch.ones((4, 3), dtype=torch.float32),
        }


class _Output:
    def __init__(self, prompt_ids, token_ids=(4,)):
        self.prompt_token_ids = prompt_ids
        self.outputs = [SimpleNamespace(token_ids=token_ids, finish_reason="stop")]


class _Llm:
    def __init__(self, outputs):
        self.outputs = outputs
        self.prompts = self.params = None

    def generate(self, prompts, params, **_):
        self.prompts, self.params = prompts, params
        return self.outputs


def _request(sample_id, seed):
    return {
        "sample_id": sample_id,
        "seed": seed,
        "images": [object()],
        "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample_id}]}],
    }


class FastBackendTests(unittest.TestCase):
    def test_prepare_inputs_keeps_processor_grid_and_token_identity(self):
        prepared = prepare_inputs(_Processor(), SimpleNamespace(image_token_id=99), _request("one", 1)["messages"], [object()])
        self.assertEqual(prepared["prompt_token_ids"], [7, 99, 99, 99, 99])
        self.assertEqual(prepared["image_grid_thw"], [[1, 2, 2]])
        self.assertIn("pixel_values", prepared["input_tensors"])

    def test_vllm_rejects_engine_prompt_mismatch_before_scoring(self):
        backend = self._backend(_Llm([_Output([0])]))
        with self.assertRaisesRegex(RuntimeError, "prompt token IDs"):
            backend.generate_batch([_request("one", 11)])

    def test_official_output_budget_leaves_room_for_input_without_truncating(self):
        llm = _Llm([_Output([7, 99, 99, 99, 99])])
        backend = self._backend(llm)
        backend.cfg["generation"]["max_new_tokens"] = 32768
        with patch("mmdl.evaluation.backends.vllm_backend._device_memory_used_bytes", return_value=123):
            backend.generate_batch([_request("one", 11)])
        self.assertEqual(llm.params[0]["max_tokens"], 32768)
        backend.max_model_len = 32768
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            backend.generate_batch([_request("one", 11)])

    def test_vllm_preserves_request_order_and_per_sample_seed(self):
        llm = _Llm([_Output([7, 99, 99, 99, 99], (10,)), _Output([7, 99, 99, 99, 99], (20,))])
        backend = self._backend(llm)
        with patch("mmdl.evaluation.backends.vllm_backend.torch.cuda.max_memory_allocated", return_value=1), \
             patch("mmdl.evaluation.backends.vllm_backend.torch.cuda.max_memory_reserved", return_value=2), \
             patch("mmdl.evaluation.backends.vllm_backend._device_memory_used_bytes", return_value=123):
            rows = backend.generate_batch([_request("first", 11), _request("second", 22)])
        self.assertEqual([row["sample_id"] for row in rows], ["first", "second"])
        self.assertEqual([param["seed"] for param in llm.params], [11, 22])
        self.assertEqual(rows[0]["generation_timing_policy"], "batch_wall_divided_by_size")
        self.assertEqual(rows[0]["observed_device_memory_used_bytes"], 123)
        self.assertIsNone(rows[0]["peak_vram_allocated_bytes"])
        self.assertEqual(rows[0]["actual_engine_prompt_token_ids"], [7, 99, 99, 99, 99])
        self.assertEqual(rows[0]["input_tensors_scope"], "pinned_cpu_processor_reference_only")
        self.assertEqual(len(llm.prompts[1]["multi_modal_data"]["image"]), 1)

    def test_vllm_constructor_forwards_the_pinned_pixel_budget_to_worker_processor(self):
        captured = {}

        class _Engine:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            @staticmethod
            def get_default_sampling_params():
                return "engine-defaults"

        cfg = {
            "execution": {"backend": "vllm", "batch_size": 2},
            "image": {"min_pixels": 262144, "max_pixels": 1310720},
        }
        processor = _Processor()
        processor.image_processor.size = {}
        with patch("mmdl.evaluation.backends.vllm_backend.torch.cuda.is_available", return_value=True), \
             patch("mmdl.evaluation.backends.vllm_backend.torch.cuda.is_bf16_supported", return_value=True), \
             patch("mmdl.evaluation.backends.vllm_backend.AutoProcessor.from_pretrained", return_value=processor), \
             patch("mmdl.evaluation.backends.vllm_backend.AutoConfig.from_pretrained", return_value=SimpleNamespace()), \
             patch("mmdl.evaluation.backends.vllm_backend._vllm", return_value=(_Engine, dict, "0.11.0")):
            VLLMBackend("/model", cfg, {"placement": "gpu_only"})
        self.assertEqual(captured["mm_processor_kwargs"], {
            "size": {"shortest_edge": 262144, "longest_edge": 1310720}
        })
        self.assertEqual(captured["generation_config"], "auto")
        self.assertIsNone(captured["quantization"])

    @staticmethod
    def _backend(llm):
        backend = object.__new__(VLLMBackend)
        backend.cfg = {"generation": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                       "repetition_penalty": 1.0, "presence_penalty": 1.5, "max_new_tokens": 2}}
        backend.batch_size = 2
        backend.max_model_len = MAX_MODEL_LEN
        backend.processor = _Processor()
        backend.model_config = SimpleNamespace(image_token_id=99)
        backend.llm = llm
        backend._sampling_params_cls = lambda **kwargs: kwargs
        return backend


if __name__ == "__main__":
    unittest.main()
