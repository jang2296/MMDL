# MMMU validation Baseline Evaluation — Qwen3-VL-4B-Instruct

- **팀명**: 미제공
- **팀원**: 미제공
- **작성일**: 2026-09-22
<!-- REPORT_STATUS:BEGIN -->
- **제출 상태**: `COMPLETE900` 검증 전. 아래의 미실행 표는 점수가 아니며, 완료된 제출용 run만 `scripts/report_baseline.py`가 갱신한다.
<!-- REPORT_STATUS:END -->

## 1. 환경 / 재현성

| 항목 | 확정한 내용과 상태 |
|---|---|
| 모델 | 공식 `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, 비양자화 BF16 |
| 데이터 | `MMMU/MMMU` revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation, 30개 config × 30개 = 900개 |
| 제출용 가속 protocol | `mmmu-val-fast-vllm-32k-continuous-v1`: vLLM `0.11.0`, GPU-only, 최대 active request 2, `max_new_tokens=32768`. 한 request가 끝날 때 즉시 완료 row를 저장하고 다음 request를 refill하는 continuous scheduling을 사용한다. 설치 lock은 [requirements-vllm.lock](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/env/requirements-vllm.lock)이다. GPU 900개 실행·메모리·시간은 아직 미검증이다. |
| 로컬 확인 | 기존 RTX 5060 8GB CPU-offload Transformers Accounting 30개와 fast local 1개는 2,048-token 역사적 확인이다. 후자는 312 tokens/EOS, generation 174.7396 s, whole 199.5493 s, allocator allocated/reserved 5,043,425,792/5,093,982,208 bytes, RSS 11,221,233,664 bytes였다. 이는 제출용 32k vLLM/900개 실행·속도 추정·점수가 아니다. |
| RunPod 재현 | GitHub 고정 commit을 새 checkout에 받는 [reproduce.sh](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/scripts/reproduce.sh)와 [eval.sh](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/scripts/eval.sh)가 모델·데이터 경로를 인자/환경변수로 받는다. 환경 doctor, lock receipt, BF16 probe와 결과 hash를 남긴다. |

<!-- REPORT_RUNTIME:BEGIN -->
제출용 `COMPLETE900` run의 GPU·시간·장치 메모리 관측치는 아직 미측정이다.
<!-- REPORT_RUNTIME:END -->

제출용 실행 예시는 다음과 같다. `<...>`은 실행자가 바꾸는 경로이며 source 파일을 수정하지 않는다.

```bash
bash scripts/eval.sh \
  --protocol configs/eval/mmmu_val_continuous_vllm_v1.yaml \
  --hardware configs/hardware/rtx3090_24gb.yaml \
  --model-ref manifests/models/baseline.json \
  --model-path "$MMDL_MODEL_PATH" \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" \
  --run-id baseline-accelerated-3090 --mode full
```

백엔드와 batch는 수업 지침 §1.3·§5가 문서화를 조건으로 자유/권장한 엔지니어링 선택이다. 모델·revision·BF16·데이터·P0·sampling·이미지 budget·채점은 바꾸지 않는다. 연속 protocol은 `async_scheduling=false`, prefix cache 비활성, eager 실행을 유지하면서 `LLMEngine.add_request()`와 `step()`으로 최대 2개 request를 active 상태로 둔다. 완료한 request는 최종 출력만 즉시 저장하고 같은 slot에 다음 독립 request를 넣는다. 이는 두 문제의 완료를 함께 기다리는 고정 batch가 아니다. vLLM protocol의 `deterministic=false`는 PyTorch deterministic-algorithm 강제를 주장하지 않는다는 뜻이며, request seed는 여전히 master 3407에서 ID별로 고정한다. vLLM worker의 image processor에도 같은 pixel budget을 전달하고, 반환된 engine prompt token IDs가 pinned CPU processor의 IDs와 같을 때만 결과를 수용한다. 이 검사는 engine 내부 pixel tensor가 직접 검증되었다는 뜻은 아니다. 새 continuous protocol은 아직 실제 GPU에서 검증되지 않았다.

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

[Qwen pinned README Evaluation Reproduction](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction)는 위 recipe와 seed 3407을 제시한다. 반면 공식 `run_mmmu.py`는 vLLM `seed=42`를 사용한다. 본 프로젝트의 sample-ID seed는 resume·batch concurrency에도 ID별 난수 정책을 유지하려는 **팀 설계**이며, 공식 구현과 난수 소비 순서까지 같다고 주장하지 않는다. run ID를 seed에 넣지 않으므로 제출용/분석용 별도 실행도 같은 ID에는 같은 seed를 사용한다.

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

현재 제출용 A의 `COMPLETE900` 결과는 없다. 별도 분석 B의 중단된 partial run은 제출 점수에
사용하지 않는다: 70/900, 시스템 실패 0, EOS 61, `finish_reason=length` 9, 생성 token
399,758, generation wall 9,936.189초였다. 9개 length row가 294,912 token(73.77%)을
차지했으며, 이 현상 때문에 continuous scheduling protocol을 별도 설계했다. 이 partial
관측은 새 protocol의 GPU 검증이나 900개 점수가 아니다.

| No. | Subject | Data Num | Acc |
|---:|---|---:|---:|
| 1 | Accounting | 30 | 미실행 |
| 2 | Agriculture | 30 | 미실행 |
| 3 | Architecture_and_Engineering | 30 | 미실행 |
| 4 | Art | 30 | 미실행 |
| 5 | Art_Theory | 30 | 미실행 |
| 6 | Basic_Medical_Science | 30 | 미실행 |
| 7 | Biology | 30 | 미실행 |
| 8 | Chemistry | 30 | 미실행 |
| 9 | Clinical_Medicine | 30 | 미실행 |
| 10 | Computer_Science | 30 | 미실행 |
| 11 | Design | 30 | 미실행 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 미실행 |
| 13 | Economics | 30 | 미실행 |
| 14 | Electronics | 30 | 미실행 |
| 15 | Energy_and_Power | 30 | 미실행 |
| 16 | Finance | 30 | 미실행 |
| 17 | Geography | 30 | 미실행 |
| 18 | History | 30 | 미실행 |
| 19 | Literature | 30 | 미실행 |
| 20 | Manage | 30 | 미실행 |
| 21 | Marketing | 30 | 미실행 |
| 22 | Materials | 30 | 미실행 |
| 23 | Math | 30 | 미실행 |
| 24 | Mechanical_Engineering | 30 | 미실행 |
| 25 | Music | 30 | 미실행 |
| 26 | Pharmacy | 30 | 미실행 |
| 27 | Physics | 30 | 미실행 |
| 28 | Psychology | 30 | 미실행 |
| 29 | Public_Health | 30 | 미실행 |
| 30 | Sociology | 30 | 미실행 |
|  | **Overall (macro avg)** | **900** | **미실행** |

계산식은 `mean(30개 과목 accuracy) = correct/900`이다. 현재 `COMPLETE900` 검증 run이 없으므로 수치를 적지 않는다.

## 6. 공식 수치와의 비교

| | Overall (MMMU validation) |
|---|---:|
| 수업이 제시한 공식 비교값 | 67.4 |
| 우리 재현 결과 | 미실행 |
| 차이 (Δ, percentage points) | 미실행 |

## 7. 격차 분석

`COMPLETE900`이 없으므로 67.4와의 격차 원인을 관측했다고 말할 수 없다. 최종 생성기는 저장 raw response의 `NO_PARSE`/`EMPTY`, `finish_reason=length`, resolved backend·length 설정만으로 1,000자 이내 진단한다. 기존 2,048-token Accounting 15/30 length 종료는 32k protocol 채택의 제한 관측일 뿐 전체 점수 차이의 원인이라고 단정하지 않는다.
<!-- REPORT_DYNAMIC:END -->

## 8. 기타 특이사항 / 한계

- 공식 비교값 67.4는 수업 지침이 제시한 비교 기준이며, 우리 측정값이 아니다.
- 결과가 완전한 뒤에도 제출용 A만 score로 사용한다. 별도 B900 분석 run과 평균/교체/고득점 선택을 하지 않는다.
- final update는 `summary.json`, 900개 sample records, 30개 subject count와 재채점 산술을 독립 검증해야만 허용한다. 대형 raw response·이미지는 외부 artifact에 두고 이 문서는 작은 제출 보고서만 보존한다.
