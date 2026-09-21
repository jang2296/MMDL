# Official source review

조회일: 2026-09-21. 원문은 저장소의 `third_party/` 사본과 아래 고정 commit을 대조했다. 원본 파일은 수정하지 않았다.

## P0 prompt

[Qwen3-VL technical report, arXiv:2511.21631v2](https://arxiv.org/pdf/2511.21631v2), PDF p.34 (PDF page index 33), Appendix B.1, defines:

```text
<image>
Question: {question}
Options:
{options}
Please select the correct answer from the options above.
```

The pinned Qwen builder ([`run_mmmu.py`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py), commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da`) matches this at `build_mmmu_prompt`, line 27: it serializes available A–Z fields as `A. value`, preserves source order, prepends `Hint: {hint}\n` only for a non-null source hint, and puts all image content before one final text content item. For open-ended records, the builder naturally omits options and the selection instruction; it must not coerce them into MCQ.

`prepare_inputs_for_vllm`, line 76, applies `processor.apply_chat_template(..., add_generation_prompt=True)`, then `process_vision_info` using `processor.image_processor.patch_size`. The local evaluator preserves image count/order and passes actual image objects to the pinned HF processor; image filenames alone are insufficient.

The same source uses 28-based script pixel bounds (`1280*28*28` and `5120*28*28`, lines 30–31). Qwen3-VL’s current processor guidance uses 32-based sizing; this script constant is therefore a reference-script choice, not a universal processor rule. Team draft P0 settings (`min_pixels=262144`, `max_pixels=1310720`) remain engineering settings pending smoke measurement. The actual pinned processor reports patch_size=16 and merge_size=2, a 32-pixel spatial compression factor.

The pinned paper also lists MMMUPro_Standard separately with `Please select the correct answer from the options.`; this project’s Assignment 1 P0 is MMMU validation only.

## Generation and length evidence

The [pinned Qwen repository README, Evaluation Reproduction](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction) (lines 1135–1152) directly specifies Instruct sampling: `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.0`, `presence_penalty=1.5`, and `seed=3407`. The pinned evaluation script instead passes vLLM `seed=42`. Record this as an implementation discrepancy; the team’s per-sample SHA-256 seed policy is a team design, not an official seed order.

Length values are not interchangeable: the [fixed model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md) documents 16,384 for vision-language, the Qwen evaluation README/script uses 32,768, and this project’s initial 2,048 value is only a draft pending measured truncation, speed, and memory evidence.

## Official parser/scorer

MMMU commit `268471d0d488258990025331c7528359c324aa25`, [`mmmu/utils/eval_utils.py`](https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py):

- `parse_multi_choice_response`, lines 10–62, uses heuristic label/text extraction and randomly chooses a label at lines 39–40 when no candidate is found (`random.seed(42)` at line 6).
- `parse_open_response`, lines 122–171, extracts answer-like tails and numbers, normalizes, and deduplicates.
- `eval_multi_choice`, lines 175–189, uses exact equality.
- `eval_open`, lines 191–216, uses normalized substring matching for strings and exact matching for numbers.

This project records parser failure as `NO_PARSE`/wrong and does not use the official random fallback. That is an explicit local scoring policy, not a claim about official MMMU behavior.

## Other pinned references

- Qwen3-VL: commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da`.
- MMMU: commit `268471d0d488258990025331c7528359c324aa25`.
- [VLMEvalKit prompt.py](https://github.com/open-compass/VLMEvalKit/blob/302cbce81a64b83b14fe3459a7390b3ff21b3e46/vlmeval/vlm/qwen3_vl/prompt.py): `_build_mmmu_prompt` keeps all images first and one text message, with the same hint/options structure.

The installed [Transformers model implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) is pinned to 4.57.1. The [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/) supplies torch 2.8.0 + torchvision 0.23.0 cu128; [Blackwell support](https://pytorch.org/blog/pytorch-2-7/) began with CUDA 12.8 wheels. Local doctor verified torch2.8.0+cu128, sm_120 and a BF16 matmul. This probe is not a model inference result.

[Accelerate CPU inference offload](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) supplies placement hooks, not a peak-VRAM guarantee or training support. The image processor alone owns resize; this Transformers path does not call qwen_vl_utils for another resize. Transformers4.57.1 has no native presence penalty; the custom generated-only processor subtracts 1.5 once per seen generated token, before temperature/top-k/top-p, verified against its actual processor dispatch.

Future adapter evaluation uses [PEFT from_pretrained](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model#from_pretrained), inference-only with adapter dtype autocast disabled. This neither trains nor establishes a validated trained checkpoint. Full/merged weights retain the original architecture and processor.
