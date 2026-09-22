"""BF16 Qwen3-VL Transformers evaluation backend."""

import random
import time
from contextlib import nullcontext

import numpy as np
import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoProcessor, LogitsProcessorList, Qwen3VLForConditionalGeneration

from mmdl.runtime.artifacts import digest
from mmdl.runtime.contracts import MODEL_REVISION
from mmdl.runtime.generation import GeneratedOnlyPresencePenalty
from mmdl.runtime.placement import placement_kwargs
from mmdl.evaluation.backends.input_preparation import prepare_inputs


class TransformersBackend:
    def __init__(self, model_path, cfg, hw, kind="base", base_path=None):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with native BF16 is required")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        start = time.monotonic()
        processor_path = model_path if kind == "base" else base_path
        self.processor = AutoProcessor.from_pretrained(processor_path, revision=MODEL_REVISION,
                                                       local_files_only=True, trust_remote_code=False)
        self.processor.image_processor.size = {"shortest_edge": cfg["image"]["min_pixels"],
                                               "longest_edge": cfg["image"]["max_pixels"]}
        self.placement = placement_kwargs(hw, torch.cuda.mem_get_info()[0],
                                           psutil.virtual_memory().available)
        weights_path = base_path if kind == "adapter" else model_path
        if kind in {"full", "merged"}:
            reference = AutoConfig.from_pretrained(base_path, local_files_only=True).to_dict()
            candidate = AutoConfig.from_pretrained(model_path, local_files_only=True).to_dict()
            for key in ("model_type", "text_config", "vision_config"):
                if candidate[key] != reference[key]:
                    raise ValueError("Derived checkpoint changed the original model structure")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            weights_path, revision=MODEL_REVISION, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
            **self.placement)
        if kind == "adapter":
            from peft import PeftConfig, PeftModel
            adapter_config = PeftConfig.from_pretrained(model_path, local_files_only=True)
            if adapter_config.base_model_name_or_path != cfg["model"]["id"]:
                raise ValueError("Adapter config does not declare the pinned base model")
            self.model = PeftModel.from_pretrained(
                self.model, model_path, is_trainable=False, autocast_adapter_dtype=False,
                local_files_only=True, **self.placement)
        self.model.eval()
        if "disk" in self.model.hf_device_map.values():
            raise RuntimeError("Disk offload is forbidden; insufficient RAM")
        if hw["placement"] == "gpu_only" and any(str(v) not in {"0", "cuda:0"}
                                                  for v in self.model.hf_device_map.values()):
            raise RuntimeError("GPU-only profile was not honored")
        if any(p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise RuntimeError("Model parameters are not uniformly BF16")
        self.generation = self.model.generation_config.from_dict(self.model.generation_config.to_dict())
        values = {k: v for k, v in cfg["generation"].items()
                  if k not in {"seed", "seed_policy", "presence_penalty"}}
        unused = self.generation.update(**values)
        if unused:
            raise ValueError(f"Unsupported generation settings: {sorted(unused)}")
        if getattr(self.generation, "presence_penalty", None) not in (None, 0, 0.0):
            raise ValueError("Refusing duplicate native and custom presence penalty")
        self.generation.use_cache = True
        self.generation.return_dict_in_generate = True
        self.generation.output_scores = False
        self.generation.validate()
        torch.cuda.synchronize()
        self.load_seconds = time.monotonic() - start

    def describe(self):
        return {"model_load_seconds": self.load_seconds,
                "device_map": {k: str(v) for k, v in self.model.hf_device_map.items()},
                "placement": self.placement, "generation_config": self.generation.to_dict(),
                "presence_penalty": {"implementation": "generated-only-custom", "value": 1.5,
                                     "order": "before temperature/top_k/top_p"},
                "token_progress_streamer": False,
                "attention_kernel": self.cfg["execution"]["sdpa_kernel"],
                "image_processor": self.processor.image_processor.to_dict(),
                "chat_template_sha256": digest(self.processor.chat_template)}

    def generate(self, messages, images, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prepared = prepare_inputs(self.processor, self.model.config, messages, images)
        inputs = prepared["inputs"].to(self.device)  # device only: retain integer types
        input_length = prepared["input_tokens"]
        penalty = GeneratedOnlyPresencePenalty(input_length, self.cfg["generation"]["presence_penalty"])
        start = time.monotonic()
        kernel = self.cfg["execution"]["sdpa_kernel"]
        attention = sdpa_kernel(SDPBackend.MATH) if kernel == "math" else nullcontext()
        if kernel not in {"math", "auto"}:
            raise ValueError(f"Unsupported SDPA kernel: {kernel}")
        with torch.inference_mode(), attention:
            result = self.model.generate(**inputs, generation_config=self.generation,
                                         logits_processor=LogitsProcessorList([penalty]))
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        token_ids = result.sequences[0, input_length:].tolist()
        raw = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        eos = self.generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        reason = "eos" if token_ids and token_ids[-1] in eos else "length"
        return dict(raw_response=raw, generated_token_ids=token_ids, generated_tokens=len(token_ids),
                    input_tokens=input_length, image_grid_thw=prepared["image_grid_thw"],
                    input_tensors=prepared["input_tensors"], input_sha256=prepared["input_sha256"],
                    chat_prompt=prepared["chat_prompt"], generation_seconds=seconds,
                    finish_reason=reason, peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())

    def generate_batch(self, requests):
        """Keep the reference path serial; fast concurrency belongs to vLLM."""
        if len(requests) != 1:
            raise ValueError("Transformers reference backend requires batch_size=1")
        request = requests[0]
        return [self.generate(request["messages"], request["images"], request["seed"])
                | {"sample_id": request["sample_id"]}]
