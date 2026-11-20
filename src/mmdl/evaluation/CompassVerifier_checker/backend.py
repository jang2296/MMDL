"""Offline, single-forward CompassVerifier inference; no answer generation."""

from __future__ import annotations

import os
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol


class Judge(Protocol):
    identity: dict

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def probabilities(self, prompt: str) -> dict: ...


class FakeJudge:
    """Scripted test double with stateless, reversible whitespace tokens."""

    def __init__(self, script: Mapping[str, dict] | Callable[[str], dict]):
        self.script = script
        self.identity: dict = {
            "model": "FakeJudge",
            "revision": "synthetic-v1",
            "dtype": "synthetic",
            "quantization": "none",
            "label_token_ids": {"A": 0, "B": 1, "C": 2},
            "torch_version": None,
            "transformers_version": None,
            "gpu_name": None,
        }

    def encode(self, text: str) -> list[int]:
        labels = self.identity["label_token_ids"]
        return [
            labels[word] if word in labels else int.from_bytes(b"\x01" + word.encode(), "big")
            for word in text.split()
        ]

    def decode(self, ids: Sequence[int]) -> str:
        labels = {value: key for key, value in self.identity["label_token_ids"].items()}
        words = []
        for token in ids:
            if token in labels:
                words.append(labels[token])
            else:
                words.append(token.to_bytes((token.bit_length() + 7) // 8, "big")[1:].decode())
        return " ".join(words)

    def probabilities(self, prompt: str) -> dict:
        result = dict(self.script(prompt) if callable(self.script) else self.script[prompt])
        result.setdefault("input_tokens", len(self.encode(prompt)))
        if "top_token" not in result:
            result["top_token"] = self.decode([result["top_token_id"]])
        return result


def check_mode(mode: str, model: str, revision: str, quantization: str, config: dict) -> bool:
    if quantization not in {"none", "nf4"}:
        raise ValueError("quantization must be none or nf4")
    if mode == "official":
        expected = config["judge"]["official"]
        actual = {
            "model": model,
            "revision": revision,
            "dtype": "bfloat16",
            "quantization": quantization,
        }
        if actual != expected:
            raise ValueError("official mode requires the pinned 7B BF16 unquantized model")
        return True
    if mode == "smoke":
        if {"model": model, "revision": revision} not in config["judge"]["smoke_allowed"]:
            raise ValueError("smoke model and revision must match the pinned allowlist")
        return False
    raise ValueError("mode must be smoke or official")


class _TransformersJudge:
    def __init__(self, tokenizer, model, identity: dict, device: int):
        self.tokenizer = tokenizer
        self.model = model
        self.identity = identity
        self.device = device

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def probabilities(self, prompt: str) -> dict:
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False, truncation=False
        )
        input_tokens = inputs["input_ids"].shape[-1]
        context_limit = self.model.config.max_position_embeddings
        if input_tokens > context_limit:
            raise ValueError(f"judge input has {input_tokens} tokens; context limit is {context_limit}")
        inputs = {key: value.to(f"cuda:{self.device}") for key, value in inputs.items()}
        with torch.inference_mode():
            # Qwen2 4.57.1 slices hidden states before projecting to the full vocabulary.
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            if not torch.isfinite(logits).all().item():
                raise ValueError("judge returned non-finite logits")
            probabilities = torch.softmax(logits, dim=-1)
            top_token_id = int(probabilities.argmax().item())
            label_ids = self.identity["label_token_ids"]
            return {
                "p_a": float(probabilities[label_ids["A"]].item()),
                "p_b": float(probabilities[label_ids["B"]].item()),
                "p_c": float(probabilities[label_ids["C"]].item()),
                "top_token_id": top_token_id,
                "top_token": self.decode([top_token_id]),
                "input_tokens": input_tokens,
            }


TOKENIZER_GIT_BLOBS = {
    "added_tokens.json": "482ced4679301bf287ebb310bdd1790eb4514232",
    "merges.txt": "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
    "special_tokens_map.json": "ac23c0aaa2434523c494330aeb79c58395378103",
    "tokenizer_config.json": "78c0d7bc3209cf7cb1e3b3595df8ee6a9a1428f4",
    "vocab.json": "6bce3a0a3866c4791a74d83d78f6824c3af64ec3",
}


def verify_tokenizer_copy(path: Path) -> None:
    """Both pinned Compass releases share these five Git blobs; no weights are needed."""
    path = Path(path)
    if not path.is_dir() or {p.name for p in path.iterdir()} != set(TOKENIZER_GIT_BLOBS):
        raise ValueError("Tokenizer copy must contain exactly the five pinned tokenizer files")
    for name, expected in TOKENIZER_GIT_BLOBS.items():
        file = path / name
        if not file.is_file() or file.is_symlink():
            raise ValueError(f"Tokenizer copy requires a regular file: {name}")
        data = file.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned tokenizer Git blob mismatch: {name}")


def load_tokenizer(model: str, revision: str, *, path: Path | None = None):
    pins = {
        "opencompass/CompassVerifier-7B": "8fee654b23ec971ad4a5c6d69bd99cf0f0188197",
        "opencompass/CompassVerifier-3B": "27d6d660bdf6c2b8a9f45906d4b76c508af65ab0",
    }
    if pins.get(model) != revision:
        raise ValueError("tokenizer model and revision must match the pinned allowlist")
    os.environ["HF_HUB_OFFLINE"] = "1"

    from transformers import AutoTokenizer

    if path is None and os.environ.get("MMDL_COMPASS_TOKENIZER_PATH"):
        path = Path(os.environ["MMDL_COMPASS_TOKENIZER_PATH"])
    if path is not None:
        from transformers.utils.hub import cached_file

        try:
            cached = cached_file(model, "tokenizer_config.json", revision=revision, local_files_only=True)
        except OSError:
            cached = None
        if cached is None or Path(cached).parent.resolve() != Path(path).resolve():
            verify_tokenizer_copy(Path(path))
            return AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                                  trust_remote_code=False)
    return AutoTokenizer.from_pretrained(
        model, revision=revision, local_files_only=True, trust_remote_code=False
    )


def load_judge(
    model: str, revision: str, *, quantization: str = "none", device: int = 0
) -> Judge:
    if quantization not in {"none", "nf4"}:
        raise ValueError("quantization must be none or nf4")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a nonnegative CUDA device index")
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise ValueError("requested CUDA device is unavailable; no CPU fallback")
    tokenizer = load_tokenizer(model, revision)
    label_ids = {}
    for label in "ABC":
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"judge label {label} must encode to exactly one token")
        label_ids[label] = ids[0]
    if len(set(label_ids.values())) != 3:
        raise ValueError("judge label token IDs must be distinct")
    options = {
        "revision": revision,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if quantization == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    loaded = AutoModelForCausalLM.from_pretrained(model, **options)
    if loaded.config.model_type != "qwen2":
        raise ValueError("the pinned checker requires Qwen2ForCausalLM")
    if quantization == "none" and (
        loaded.dtype != torch.bfloat16 or getattr(loaded, "is_quantized", False)
    ):
        raise ValueError("unquantized judge must actually be BF16 without quantization")
    if quantization == "nf4" and not getattr(loaded, "is_loaded_in_4bit", False):
        raise ValueError("NF4 judge must actually be loaded in 4 bits")
    if any(token < 0 or token >= loaded.config.vocab_size for token in label_ids.values()):
        raise ValueError("judge label token ID is outside the model vocabulary")
    if getattr(loaded.config, "_commit_hash", revision) != revision:
        raise ValueError("judge loaded revision differs from the requested revision")
    loaded.eval()
    identity = {
        "model": model,
        "revision": revision,
        "dtype": "bfloat16",
        "quantization": quantization,
        "label_token_ids": label_ids,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    return _TransformersJudge(tokenizer, loaded, identity, device)
