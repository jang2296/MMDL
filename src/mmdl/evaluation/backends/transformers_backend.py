"""BF16 Qwen3-VL with identical generation on either hardware profile."""

import hashlib
import json
import random
import time

import numpy as np
import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoProcessor, LogitsProcessorList, Qwen3VLForConditionalGeneration
from transformers.generation.streamers import BaseStreamer

from mmdl.runtime.artifacts import digest
from mmdl.runtime.contracts import MODEL_REVISION
from mmdl.runtime.generation import GeneratedOnlyPresencePenalty
from mmdl.runtime.placement import placement_kwargs


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


class TokenProgress(BaseStreamer):
    """Report counts, never benchmark text, while a long offloaded response runs."""

    def __init__(self):
        self.started = time.monotonic()
        self.tokens = 0
        self.prompt_pending = True

    def put(self, value):
        if self.prompt_pending:
            self.prompt_pending = False
            return
        self.tokens += value.numel()
        if self.tokens == 1 or self.tokens % 128 == 0:
            print(json.dumps({"generated_tokens": self.tokens,
                              "elapsed_seconds": round(time.monotonic() - self.started, 3)}), flush=True)

    def end(self):
        pass  # Final counts and synchronized timing are emitted by the engine.


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
                "image_processor": self.processor.image_processor.to_dict(),
                "chat_template_sha256": digest(self.processor.chat_template)}

    def generate(self, messages, images, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        if "pixel_values" not in inputs or inputs["image_grid_thw"].shape[0] != len(images):
            raise ValueError("Actual image tensors do not match message images")
        if inputs["input_ids"].dtype != torch.int64 or inputs["image_grid_thw"].is_floating_point():
            raise ValueError("Integer input types were changed")
        image_tokens = int((inputs["input_ids"] == self.model.config.image_token_id).sum())
        expected = int(inputs["image_grid_thw"].prod(-1).sum()) // self.processor.image_processor.merge_size**2
        if image_tokens != expected:
            raise ValueError("Image token count differs from processor grid")
        identity = {key: tensor_identity(value) for key, value in inputs.items()}
        grid = inputs["image_grid_thw"].tolist()
        input_length = inputs["input_ids"].shape[-1]
        inputs = inputs.to(self.device)  # device only: retain integer types
        penalty = GeneratedOnlyPresencePenalty(input_length, self.cfg["generation"]["presence_penalty"])
        start = time.monotonic()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            result = self.model.generate(**inputs, generation_config=self.generation,
                                         logits_processor=LogitsProcessorList([penalty]),
                                         streamer=TokenProgress())
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        token_ids = result.sequences[0, input_length:].tolist()
        raw = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        eos = self.generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        reason = "eos" if token_ids and token_ids[-1] in eos else "length"
        return dict(raw_response=raw, generated_token_ids=token_ids, generated_tokens=len(token_ids),
                    input_tokens=input_length, image_grid_thw=grid, input_tensors=identity,
                    input_sha256=digest(identity), chat_prompt=text, generation_seconds=seconds,
                    finish_reason=reason, peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
