# Third-party sources

The checked-in files under `third_party/{qwen3_vl,mmmu,vlmevalkit}` are unchanged reference sources used for review and reproducibility. Runtime imports use only the MMMU parser where explicitly configured; the copied source trees are not modified.

| Source | Commit | License metadata |
|---|---|---|
| QwenLM/Qwen3-VL | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | `LICENSE`, SPDX `Apache-2.0` |
| MMMU-Benchmark/MMMU | `268471d0d488258990025331c7528359c324aa25` | `LICENSE`, SPDX `Apache-2.0` |
| open-compass/VLMEvalKit | `302cbce81a64b83b14fe3459a7390b3ff21b3e46` | `LICENSE`, SPDX `Apache-2.0` |

Repository license metadata does not settle benchmark dataset redistribution rights. MMMU data remains an external evaluation artifact and is not treated as training data. The model Hugging Face revision is tracked separately from these Git commits.
