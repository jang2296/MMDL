"""MMMU P0 prompt and image-message construction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_IMAGE_TAG = re.compile(r"<image\b[^>]*>", re.IGNORECASE)
_IMAGE_REF = re.compile(r"<image\s+(\d+)>", re.IGNORECASE)
_MCQ_TYPES = {"mcq", "multiple-choice", "multiple choice"}
_OPEN_TYPES = {"open", "open-ended", "open ended", "free-response", "free response"}


def _question_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    normalized = normalized.replace("—", "-")
    if normalized in _MCQ_TYPES:
        return "mcq"
    if normalized in _OPEN_TYPES:
        return "open"
    raise ValueError(f"Unsupported question_type: {value!r}")


def _option_values(sample: Mapping[str, Any]) -> list[str]:
    options = sample.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise TypeError("sample['options'] must be a list of option text")
    values = []
    for option in options:
        if isinstance(option, Mapping):
            option = option.get("text", option.get("value"))
        if option is None:
            raise ValueError("option text cannot be null")
        values.append(str(option))
    return values


def _template(root: Path, name: str) -> str:
    root = Path(root)
    for path in (root / "prompts" / name, root / name):
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Cannot find prompt template {name!r} below {root}")


def build_prompt(sample: dict, root: Path) -> str:
    """Render the official Qwen MMMU prompt without answer-side information."""
    question = sample.get("question")
    if question is None:
        raise ValueError("sample['question'] is required")

    kind = _question_type(sample.get("question_type"))
    hint = sample.get("hint")
    hint_prefix = "" if hint is None or not str(hint).strip() else f"Hint: {hint}\n"
    if kind == "open":
        # The official builder omits options and the selection instruction for open items.
        return _template(root, "mmmu_open_v1.txt").format(
            hint_prefix=hint_prefix,
            question=str(question),
        ).rstrip()

    options = _option_values(sample)
    if not options:
        raise ValueError("multiple-choice samples require at least one option")
    option_text = "".join(f"{chr(65 + index)}. {option}\n" for index, option in enumerate(options))
    return _template(root, "mmmu_mcq_v1.txt").format(
        hint_prefix=hint_prefix,
        question=str(question),
        options=option_text,
    ).rstrip()


def _referenced_image_numbers(sample: Mapping[str, Any]) -> set[int]:
    values = [str(sample.get("question", ""))]
    try:
        values.extend(_option_values(sample))
    except (TypeError, ValueError):
        # build_prompt will report the more useful option validation error later.
        pass

    numbers: set[int] = set()
    for value in values:
        for tag in _IMAGE_TAG.findall(value):
            match = _IMAGE_REF.fullmatch(tag)
            if match is None:
                raise ValueError(f"Invalid image reference: {tag!r}")
            numbers.add(int(match.group(1)))
    return numbers


def _validate_image_references(sample: Mapping[str, Any], image_count: int) -> None:
    numbers = _referenced_image_numbers(sample)
    if any(number < 1 or number > image_count for number in numbers):
        raise ValueError(
            f"Image references must be 1..{image_count}; found {sorted(numbers)}"
        )

    metadata = sample.get("image_references")
    if metadata is None:
        return
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise TypeError("sample['image_references'] must be a list")
    for item in metadata:
        if not isinstance(item, Mapping):
            raise TypeError("image reference metadata must be mappings")
        marker = item.get("marker")
        match = _IMAGE_REF.fullmatch(str(marker)) if marker is not None else None
        if match is None:
            raise ValueError(f"Invalid image reference metadata: {marker!r}")
        number = int(match.group(1))
        if not 1 <= number <= image_count:
            raise ValueError(f"Image reference {number} has no supplied image slot")
        content_index = item.get("content_index")
        if content_index is not None and content_index != number - 1:
            raise ValueError(f"Image reference {marker!r} is mapped to the wrong image")


def build_messages(sample: dict, images: list[Any], root: Path | None = None) -> list[dict[str, Any]]:
    """Keep raw images in supplied order, followed by the single P0 text block."""
    if not images:
        raise ValueError("MMMU samples require at least one image")
    _validate_image_references(sample, len(images))
    if root is None:
        root = Path(__file__).resolve().parents[3]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": build_prompt(sample, root)})
    return [{"role": "user", "content": content}]
