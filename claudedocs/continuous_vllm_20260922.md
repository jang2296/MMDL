# Continuous vLLM validation — 2026-09-22

## Scope and fixed conditions

Implementation commit: `702ed44c4c45f454746ec4b371ad1b7403b67856`.
Protocol: `mmmu-val-fast-vllm-32k-continuous-v1`.
The pinned Qwen/model/processor revision, BF16, MMMU validation 900 IDs, P0,
image budgets, sampling recipe, per-ID seed and scorer are unchanged.
The new protocol changes scheduling; it must not resume or merge the historical batch run.

The output cap remains 32,768, following the default in the
[pinned Qwen MMMU evaluation script](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py).
This is a maximum, not a required output length. The pinned model card separately
describes VL length 16,384; these sources are not presented as identical settings.

## Diagnosis before replacement

`analysis` is a run label, not an extra reasoning prompt. The previous implementation
submitted two requests to `LLM.generate()` and waited for both before saving/refilling.
One long completion therefore prevented a finished slot from receiving new work.

For the same first 70 IDs, observed pinned CPU-reference input hashes and seeds matched:

| Observation | Historical Transformers, cap 2,048 | Historical vLLM batch 2, cap 32,768 |
|---|---:|---:|
| Output tokens | 59,265 | 399,758 |
| Completed generation-call wall seconds | 3,794.338 | 9,936.189 |
| Aggregate tokens/second | 15.619 | 40.233 |

This is not an isolated backend benchmark: output lengths, kernels and RNG consumption differ.
The 6.75-fold output volume outweighed the higher observed token throughput.
Nine outputs reached 32,768 and accounted for 294,912 tokens (73.77%).
The other 61 ended with EOS token 151645; none of the nine capped token sequences
contained either configured EOS ID (151645 or 151643). This does not indicate ignored EOS.
Inspection found repeated recalculation and literal repeated sentences, not useful reasoning
throughout every long response. The final model-level cause remains unproven without controlled tests.

The partial 70-result directory and its backup archive were deleted at the user's later
explicit request, after initial recovery verification. The table is a historical observation,
not currently retained raw evidence or an Assignment score. No partial outputs are reused.

## Implementation and CPU evidence

The [vLLM 0.11.0 engine](https://docs.vllm.ai/en/v0.11.0/api/vllm/v1/engine/llm_engine.html)
is driven through `add_request()` / `step()` with `FINAL_ONLY` output. At most two requests
are prepared/active. Every valid completion is yielded to the existing atomic writer before
the next request is prepared. Out-of-order completion is mapped by immutable sample ID.
Errors or consumer closure abort outstanding requests; already-saved outputs remain intact.
`async_scheduling=false`, eager execution and native generated-only presence penalty remain unchanged.

`generation_seconds` attributes each observed engine-step wall duration equally across its
active requests. `request_latency_seconds` records overlapping request latency separately.
Neither latency sums nor partial-run attribution are used as whole-run elapsed time;
use `evaluation_seconds_this_invocation` for actual end-to-end evaluation throughput.

Validation: `MMDL_PYTHON=<locked-eval-python> bash scripts/test_reference.sh`:
72 unit/integration tests PASS; Ruff PASS; mypy 22 source files PASS; Bash syntax PASS.
Tests include short-before-long refill, completion persistence before next preparation,
failure recovery, out-of-order IDs, consumer abort and additive timing.
CPU inspection of the installed pinned vLLM API and real `SamplingParams` passed without
loading the model or performing GPU inference. Staged public-file/secret/size checks passed.

The original reference Pod remains unchanged. The replacement checks out the fixed GitHub
commit, installs the exact vLLM lock, checks CUDA/BF16, downloads pinned artifacts, and runs
Accounting_1 / Agriculture_1 / Biology_29 smoke before the fresh analysis 900.
GPU evidence at 06:11:33 UTC: all three smoke requests completed without a system error,
with 303/5/415 tokens respectively, all EOS. Correctness was 1/3; this is functional
validation, not a full benchmark score. Evaluation wall time was 17.1464918433 seconds;
whole CLI time including initialization was 59.0283647431 seconds. The 500-ms sampled
whole-device GPU memory maximum was 22,121,807,872 bytes (20.6025 GiB), not an allocator peak.
The RTX 3090 used driver 580.65.06 and Python 3.12.3; installed lock and CUDA/BF16 checks passed.

Actual refill order (same host UTC clock): Agriculture's result was saved at 06:09:37.226;
Biology's record began at 06:09:37.324; Accounting's result was saved at 06:09:44.771.
Thus the next problem started while its older peer was still unfinished. Independent
sample hash, image hash, seed, token count, actual prompt-token hash and rescore checks passed.
Fresh analysis900 had 5 saved results and zero system-error records at the first check.
The remaining 895 were not claimed as complete. Long outputs can still consume 32,768 tokens:
continuous scheduling removes pair waiting, not the computational cost of those outputs.
