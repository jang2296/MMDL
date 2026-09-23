"""Backend-neutral, pinned-processor input preparation."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from mmdl.runtime.artifacts import digest


def tensor_identity(tensor: Any) -> dict[str, Any]:
    """Return a stable CPU identity without changing the model input."""
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def prepare_inputs(processor: Any, model_config: Any, messages: list[dict[str, Any]], images: list[Any],
                   **processor_kwargs: Any) -> dict[str, Any]:
    """Apply the pinned processor once and validate its actual image tensors."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", **processor_kwargs)
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise ValueError("Official processor did not return image tensors and grid")
    if inputs["image_grid_thw"].shape[0] != len(images):
        raise ValueError("Actual image tensors do not match message images")
    if str(inputs["input_ids"].dtype) != "torch.int64" or inputs["image_grid_thw"].is_floating_point():
        raise ValueError("Integer input types were changed")
    image_tokens = int((inputs["input_ids"] == model_config.image_token_id).sum())
    expected = int(inputs["image_grid_thw"].prod(-1).sum()) // processor.image_processor.merge_size**2
    if image_tokens != expected:
        raise ValueError("Image token count differs from processor grid")
    identities = {key: tensor_identity(value) for key, value in inputs.items()}
    return {
        "chat_prompt": text,
        "inputs": inputs,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "prompt_token_ids": inputs["input_ids"][0].tolist(),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "input_tensors": identities,
        "input_sha256": digest(identities),
    }


def prepare_protocol_inputs(processor: Any, model_config: Any, messages: list[dict[str, Any]],
                            images: list[Any], image_cfg: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """Use the same image path for the CPU reference and vLLM; never resize twice."""
    policy = image_cfg.get("preprocessing", "hf_processor")
    if policy == "hf_processor":
        return prepare_inputs(processor, model_config, messages, images), images
    if policy != "qwen_vl_utils_0_0_14":
        raise ValueError(f"Unknown image preprocessing: {policy}")
    from importlib.metadata import version

    from qwen_vl_utils import process_vision_info

    if version("qwen-vl-utils") != "0.0.14":
        raise RuntimeError("The official preprocessing protocol pins qwen-vl-utils==0.0.14")
    # The utility only reads image blocks; the original P0 text and ordering are unchanged.
    vision_messages = [{"role": "user", "content": [
        {"type": "image", "image": image, "min_pixels": image_cfg["min_pixels"],
         "max_pixels": image_cfg["max_pixels"]} for image in images]}]
    processed, videos = process_vision_info(
        vision_messages, image_patch_size=processor.image_processor.patch_size)
    if videos is not None or processed is None or len(processed) != len(images):
        raise ValueError("Official image preprocessing changed image coverage")
    item = prepare_inputs(processor, model_config, messages, processed, do_resize=False)
    item["processed_image_sizes"] = [list(image.size) for image in processed]
    return item, processed
