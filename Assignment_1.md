# MMMU validation Baseline Evaluation — Qwen3-VL-4B-Instruct

- **팀명**: 미제공
- **팀원**: 미제공
- **작성일**: 2026-09-23
<!-- REPORT_STATUS:BEGIN -->
- **제출 상태**: legacy continuous 32k canonical run `COMPLETE900` 및 로컬 독립 검증 완료. 원본 parser 기준 점수는 51.00%이며, V2 CPU audit 점수 64.89%는 별도 rescoring 결과로 구분한다.
<!-- REPORT_STATUS:END -->

## 1. 환경 / 재현성

| 항목 | 확정한 내용과 상태 |
|---|---|
| 모델 | 공식 `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, 비양자화 BF16 |
| 데이터 | `MMMU/MMMU` revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation, 30개 config × 30개 = 900개 |
| canonical 평가 protocol | `mmmu-val-fast-vllm-32k-continuous-v1`: vLLM `0.11.0`, RTX 3090 24GB GPU-only, 최대 active request 2, `max_new_tokens=32768`. 한 request가 끝날 때 즉시 완료 row를 저장하고 다음 request를 refill하는 continuous scheduling을 사용했다. 900/900 완료·시스템 오류 0, 평가 57,885.846초, 생성 token 3,719,200이다. |
| 로컬 확인 | 기존 RTX 5060 8GB CPU-offload Transformers Accounting 30개와 fast local 1개는 2,048-token 역사적 확인이다. 후자는 312 tokens/EOS, generation 174.7396 s, whole 199.5493 s, allocator allocated/reserved 5,043,425,792/5,093,982,208 bytes, RSS 11,221,233,664 bytes였다. 이는 제출용 32k vLLM/900개 실행·속도 추정·점수가 아니다. |
| 연속 처리 GPU 검증 | commit `702ed44c4c45f454746ec4b371ad1b7403b67856`에서 3문제 SMOKE와 900문제 full run을 완료했다. full run 장치 전체 메모리 관측 최대는 22,885,171,200 bytes이며, worker allocator peak는 미측정이다. |
| RunPod 재현 | GitHub 고정 commit을 새 checkout에 받는 [reproduce.sh](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/scripts/reproduce.sh)와 [eval.sh](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/scripts/eval.sh)가 모델·데이터 경로를 인자/환경변수로 받는다. 환경 doctor, lock receipt, BF16 probe와 결과 hash를 남긴다. |

<!-- REPORT_RUNTIME:BEGIN -->
legacy canonical run의 평가 wall time은 57,885.846초(모델 로드 제외), 장치 전체 메모리 관측 최대는 22,885,171,200 bytes다. 부모 프로세스 RAM peak는 2,555,043,840 bytes로 worker를 포함하지 않는다.
<!-- REPORT_RUNTIME:END -->

제출용 실행 예시는 다음과 같다. `<...>`은 실행자가 바꾸는 경로이며 source 파일을 수정하지 않는다.

```bash
bash scripts/eval.sh \
  --protocol configs/eval/mmmu_val_continuous_vllm_v1.yaml \
  --hardware configs/hardware/rtx3090_24gb.yaml \
  --model-ref manifests/models/baseline.json \
  --model-path "$MMDL_MODEL_PATH" \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" \
  --job-id "$MMDL_JOB_ID" --run-role evaluation --require-commit "$MMDL_CODE_COMMIT" \
  --run-id "${MMDL_JOB_ID}-evaluation" --mode full
```

백엔드와 batch는 수업 지침 §1.3·§5가 문서화를 조건으로 자유/권장한 엔지니어링 선택이다. 모델·revision·BF16·데이터·P0·sampling·이미지 budget·채점은 바꾸지 않는다. 연속 protocol은 `async_scheduling=false`, prefix cache 비활성, eager 실행을 유지하면서 `LLMEngine.add_request()`와 `step()`으로 최대 2개 request를 active 상태로 둔다. 완료한 request는 최종 출력만 즉시 저장하고 같은 slot에 다음 독립 request를 넣는다. 이는 두 문제의 완료를 함께 기다리는 고정 batch가 아니다. vLLM protocol의 `deterministic=false`는 PyTorch deterministic-algorithm 강제를 주장하지 않는다는 뜻이며, request seed는 여전히 master 3407에서 ID별로 고정한다. vLLM worker의 image processor에도 같은 pixel budget을 전달하고, 반환된 engine prompt token IDs가 pinned CPU processor의 IDs와 같을 때만 결과를 수용한다. 이 검사는 engine 내부 pixel tensor가 직접 검증되었다는 뜻은 아니다. 3문제 GPU SMOKE와 실제 slot refill, 900문제 전체 실행을 완료했다.

연속 실행의 `generation_seconds`는 각 `step()` wall time을 해당 step의 active request 수로 나누어 request에 배분한 분석용 시간이고, 겹쳐 실행되는 request의 실제 지연시간은 `request_latency_seconds`로 별도 기록한다. 따라서 request 시간 합은 전체 elapsed time과 같다고 해석하지 않으며, throughput은 engine invocation wall time으로 계산한다. 기존 `mmmu-val-fast-vllm-32k-v1` 고정 batch protocol은 과거 분석 기록으로 보존하고 새 protocol과 결과를 합치지 않는다.

## 2. 프롬프트

실제 P0 템플릿은 다음이며 [Qwen 공식 MMMU builder `build_mmmu_prompt`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)를 기준으로 저장소에 고정했다. 상세 대조 위치는 [SOURCE_REVIEW](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/docs/SOURCE_REVIEW.md)에 기록했다.

객관식:

```text
{hint_prefix}Question: {question}
Options:
{options}Please select the correct answer from the options above.
```

주관식:

```text
{hint_prefix}Question: {question}
```

`hint`가 원본에 있고 비어 있지 않을 때만 `Hint: {hint}\n`을 앞에 붙인다. 객관식 보기는 원래 순서대로 `A. ...`부터 직렬화한다. 실제 user content는 `image_1, …, image_N, text` 순서이고, 이 **전체 messages list**에 pinned [Qwen chat template](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/chat_template.json)을 `apply_chat_template(add_generation_prompt=True)`로 적용한다. 따라서 wrapper의 ChatML role/assistant-generation prefix와 vision placeholder는 processor가 만들며 텍스트 마지막 항목만 template에 적용하는 것이 아니다. 정답·해설은 입력에서 분리된다. 공식 builder와 같은 P0를 쓰는 것은 protocol 비교 가능성을 위해서이며, extra system/few-shot/CoT/JSON 지시는 넣지 않는다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 | 근거/구분 |
|---|---:|---|
| `do_sample` | `true` | Qwen Evaluation Reproduction recipe |
| `temperature` | `0.7` | 동일 |
| `top_p` | `0.8` | 동일 |
| `top_k` | `20` | 동일 |
| `repetition_penalty` | `1.0` | 동일 |
| `presence_penalty` | `1.5` | 동일; Transformers는 generated-only custom processor, vLLM은 native penalty를 사용하고 적용 순서를 backend metadata에 기록 |
| master seed | `3407` | Qwen README recipe |
| request seed | `int.from_bytes(SHA256("3407:{sample_id}").digest()[:8], "big") mod 2^31` | 팀의 `per_sample_sha256_v1` 정책 |

[Qwen pinned README Evaluation Reproduction](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction)는 위 recipe와 seed 3407을 제시한다. 반면 공식 `run_mmmu.py`는 vLLM `seed=42`를 사용한다. 본 프로젝트의 sample-ID seed는 resume·batch concurrency에도 ID별 난수 정책을 유지하려는 **팀 설계**이며, 공식 구현과 난수 소비 순서까지 같다고 주장하지 않는다. run ID를 seed에 넣지 않으며 점수와 실패 검토는 한 번 생성한 응답을 함께 사용한다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 팀 protocol 값 | 상태와 이유 |
|---|---:|---|
| `max_new_tokens` | `32768` | [Qwen 공식 평가 source](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의 `out_seq_length=32768`을 따르는 최종 가속 protocol 값. 잘림률은 `finish_reason=length` 개수로 보고한다. |
| `max_model_len` | `36864` | 입력+생성 전체 문맥의 팀 설정. 고정 processor로 900개 입력을 CPU 전처리한 최대 2,655 + 출력 32,768 = 35,423보다 크게 확보했다. 초과 입력을 자동 자르지 않는다. GPU OOM-free 보장은 아니다. |
| `min_pixels` | `262144` | pinned official processor가 resize owner |
| `max_pixels` | `1310720` | 동일 |

고정 [Qwen 모델 card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md)의 VL length 16,384과 공식 평가 source의 출력 상한 32,768은 서로 다른 권장 설정이며, 여기서는 사용자 승인으로 후자를 채택한다. 기존 2,048-token reference/local 경로는 Accounting 30개 중 EOS 15개(정답 13), length 15개(정답 5)라는 높은 cutoff 관측이 있어 제출용으로 채택하지 않는다. 이 관측은 길이와 정답의 인과 증명은 아니다. 새 32k 가속 protocol은 sampling·seed·P0·이미지 budget을 그대로 두고 출력 상한을 공식 평가 source에 맞춘다. 다른 길이의 local 1개와 32k 900개 사이의 속도 배수는 추정하지 않는다.

이미지 budget은 [고정 Qwen README](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md)의 256–1280 visual-token 예시를 채택한 팀 선택이다. 수업 강제값이나 공식 MMMU 평가 script의 더 큰 28 기반 budget과 같다고 주장하지 않는다. patch=16, merge=2이므로 한 visual token은 32×32 pixel 격자에 대응한다. 24GB 장비의 메모리 여유와 비교 protocol의 일관성을 위해 이 범위를 고정했고, 실제 크기는 aspect ratio와 processor smart resize에 따라 달라지므로 고정 폭·높이가 아니다.

## 4. 채점(파싱) 방식

[MMMU 공식 `eval_utils.py`](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)의 open-ended parser/evaluator를 고정 사본으로 사용하고, 객관식은 같은 순서를 따르되 random fallback만 제거한 `mmmu-official-no-random-v1`이다.

- 빈 raw 응답은 즉시 `EMPTY`, 오답이다.
- 객관식은 끝 구두점을 제거한 뒤 `(A)` 같은 괄호 label → 공백으로 분리된 label → 응답이 5단어 초과일 때 option text 포함 순으로 후보를 찾는다. 후보가 여러 개면 label/text의 마지막 출현 후보를 선택한다.
- 후보가 없으면 무작위 label을 고르지 않고 `NO_PARSE`, 오답으로 남긴다. `EMPTY`와 `NO_PARSE`는 모두 900개 분모에 포함한다.
- 주관식은 공식 `parse_open_response`의 answer-tail/number normalization·dedup 결과를 정렬해 `eval_open`에 전달한다. 문자열은 정규화된 substring, 수치는 exact match 규칙을 따른다.

이는 공식 fallback을 그대로 썼다는 주장이 아니라, 재실행 시 같은 실패 처리와 분모를 보장하는 명시적 팀 scoring 정책이다.

<!-- REPORT_DYNAMIC:BEGIN -->
## 5. 결과

canonical R2 evaluation은 `COMPLETE900`으로 완료되어 로컬 독립 검증했다. legacy R1도
동일하게 900/900 검증되었다. R1은 2,048-token Transformers reference, R2는 32,768-token
vLLM continuous baseline이다. 아래 표의 R2 원본 parser 점수는 459/900=51.00%이며,
별도 CPU V2 audit 점수 584/900=64.89%는 새 추론이 아닌 rescoring 결과다.

| No. | Subject | Data Num | Acc |
|---:|---|---:|---:|
| 1 | Accounting | 30 | 80.00% |
| 2 | Agriculture | 30 | 33.33% |
| 3 | Architecture_and_Engineering | 30 | 36.67% |
| 4 | Art | 30 | 40.00% |
| 5 | Art_Theory | 30 | 46.67% |
| 6 | Basic_Medical_Science | 30 | 56.67% |
| 7 | Biology | 30 | 43.33% |
| 8 | Chemistry | 30 | 26.67% |
| 9 | Clinical_Medicine | 30 | 46.67% |
| 10 | Computer_Science | 30 | 50.00% |
| 11 | Design | 30 | 56.67% |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 30.00% |
| 13 | Economics | 30 | 60.00% |
| 14 | Electronics | 30 | 43.33% |
| 15 | Energy_and_Power | 30 | 50.00% |
| 16 | Finance | 30 | 66.67% |
| 17 | Geography | 30 | 46.67% |
| 18 | History | 30 | 70.00% |
| 19 | Literature | 30 | 60.00% |
| 20 | Manage | 30 | 63.33% |
| 21 | Marketing | 30 | 83.33% |
| 22 | Materials | 30 | 36.67% |
| 23 | Math | 30 | 56.67% |
| 24 | Mechanical_Engineering | 30 | 33.33% |
| 25 | Music | 30 | 30.00% |
| 26 | Pharmacy | 30 | 53.33% |
| 27 | Physics | 30 | 53.33% |
| 28 | Psychology | 30 | 56.67% |
| 29 | Public_Health | 30 | 80.00% |
| 30 | Sociology | 30 | 40.00% |
|  | **Overall (macro avg)** | **900** | **51.00% (459/900)** |

계산식은 `mean(30개 과목 accuracy) = correct/900`이다. 모든 과목이 30개로 같아 R2는 `459/900=51.00%`다. V2 audit은 동일 raw response의 CPU rescoring이며 표의 원본 parser 점수를 대체하지 않는다.

## 6. 공식 수치와의 비교

| | Overall (MMMU validation) |
|---|---:|
| 수업이 제시한 공식 비교값 | 67.4 |
| 우리 재현 결과 (R2 원본 parser) | 51.00% |
| 차이 (Δ, percentage points) | −16.40pp |

## 7. 격차 분석

R2 원본 parser 점수는 51.00%로 수업 비교값 67.4보다 16.40pp 낮다. CPU V2 audit은 저장 raw response를 독립 재생해 64.89%를 얻었고, 900개 입력·answer·raw hash 및 8개 audit test를 확인했다. 이는 새 모델 점수가 아니라 parser/scoring 차이의 측정이다. V2에서 주관식 53개와 `length` 종료 75개는 바꾸지 않았으며, 파싱 개선으로 모든 차이가 설명된다고 단정하지 않는다. 공식 Qwen 경로와의 image budget·전처리·채점/seed 차이, 모델의 시각·수리 추론 오류가 남은 후보이며, 새 team-final-answer-v4와 ABC 입력 조건은 CPU 검증 후 고정했으며, GPU 통제 실험과 신규900은 아직 미실행이다.
<!-- REPORT_DYNAMIC:END -->

## 8. 기타 특이사항 / 한계

2026-09-23 채점/입력 통제: `team-final-answer-v4`로 저장 R1/R2를 전수 재채점한 값은
479/900(53.22%)·556/900(61.78%)이다. V2 감사의 기존 length/open 판정 유지와 다른 정책이며
새 추론 점수는 아니다. 공식 open evaluator를 최종답 구간에만 적용하는 이유와 이미지 A/B/C,
고정값·코드 hash·회귀검사·차이는 [EVALUATION_V2](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/docs/EVALUATION_V2.md)에 기록했다.
새3090 GPU 통제 실험/신규900이 완료되기 전에는 위 원래 baseline 표를 대체하지 않는다.

- 공식 비교값 67.4는 수업 지침이 제시한 비교 기준이며, 우리 측정값이 아니다.
- R1 reference는 427/900=47.44%, R2 continuous baseline은 459/900=51.00%다. 두 실행은 backend·출력 상한·난수 소비가 달라 순수 길이 효과로 해석하지 않는다.
- R2 V2 audit 584/900=64.89%는 동일 저장 응답의 CPU rescoring 결과다. 새 추론/공식 judge 결과가 아니며, 새 scoring version을 최종 고정하기 전까지 R2 원본 51.00%를 canonical baseline으로 유지한다.
- R2는 847개 객관식과 53개 주관식으로 구성되고, `length` 75개, `NO_PARSE` 69개를 기록했다. 새 parser·ABC 통제 실험은 별도 검증 후 갱신한다.
- 다음 승인 작업은 CPU scoring 고정 → CPU 입력 audit → ABC 통제 protocol → RTX 3090 full900 순서로 진행 중이다. 새 `team-final-answer-v4` parser와 ABC 결과는 아직 보고서 점수에 반영하지 않는다.
- final update는 `summary.json`, 900개 sample records, 30개 subject count와 재채점 산술을 독립 검증해야만 허용한다. 대형 raw response·이미지는 외부 artifact에 두고 이 문서는 작은 제출 보고서만 보존한다.
