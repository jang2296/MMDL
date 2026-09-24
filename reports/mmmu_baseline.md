# MMMU validation Baseline Evaluation — Qwen3-VL-4B-Instruct

- **팀명**: 미제공
- **팀원**: 미제공
- **작성일**: 2026-09-24
- **재현 커맨드**: 아래 GPU 평가 명령과 `bash scripts/rescore.sh ... --parser team-final-answer-v8`
<!-- REPORT_STATUS:BEGIN -->
- **제출 상태**: `COMPLETE900` 독립 검증 완료; job `mmdl-official-b2-20260923`, run `mmdl-official-b2-20260923-evaluation`, 900/900 completed, denominator `900`.
- **원본 실행 표식**: `evaluation` (기존 기록 보존; 점수·실패 분석은 같은 응답을 사용).
- **원본 GPU 추론 명령**: `bash scripts/eval.sh --protocol configs/eval/mmmu_val_official_vllm_b2_v2.yaml --hardware configs/hardware/rtx3090_24gb.yaml --model-ref manifests/models/baseline.json --model-path "$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17" --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --run-id mmdl-official-b2-20260923-evaluation --mode full --require-commit 6d6d4a1a4bed1785e8770902d65e12b54340ba75 --job-id mmdl-official-b2-20260923 --run-role evaluation`
- **사후 CPU 채점**: `team-final-answer-v8`, scorer SHA256 `b80a28a4b66906d5c4f1ece046bd7c4be2b447445da4b072144e52ce7ee047fa`. 원본 GPU 응답/입력/종료 사유를 보존하여 재채점했으며 아래 시간·메모리는 원본 추론 실측이다.
- **재채점 원장**: 별도 보존된 `source.json`, `scorer_source.json`, `samples/`, `predictions.jsonl`; 새 GPU 추론이 아니며 원래 inference protocol을 소급 변경하지 않았다.
<!-- REPORT_STATUS:END -->

## 1. 환경 / 재현성

| 항목 | 실제 실행과 확인 범위 |
|---|---|
| 모델 | 공식 `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, 비양자화 BF16. processor/tokenizer/chat template도 동일 revision |
| 데이터 | `MMMU/MMMU` revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation30과목×30=900. 각 실행 객관식847·주관식53 |
| 최신 B2 추론 | `mmmu-val-official-vllm-b2-v2`, Git commit `6d6d4a1a4bed1785e8770902d65e12b54340ba75`, RTX3090 24GB GPU-only, vLLM0.11.0, 최대 동시2 continuous requests |
| 환경 | Linux x86_64, Python3.12.3, torch2.8.0+cu128, transformers4.57.1, qwen-vl-utils0.0.14. driver580.126.09, torch CUDA12.8. vLLM 재현 lock: `env/requirements-vllm.lock`(검사 도구 pin 추가); 원본 실행별 `environment.json`·lock receipt는 별도 보존 |
| 실측 VRAM | B2 장치 전체 메모리 **관측 최대24,552,407,040 bytes (22.87GiB)**. worker allocator peak는 미측정이며 주기적 장치 관측을 정확한 allocator peak라고 부르지 않는다 |
| 시간 | B2 평가89,561.304초(약24시간53분), 모델 로드27.700초, 전체89,608.979초. 이는 기존 GPU 추론 시간이며 이번 CPU 재채점 시간이 아님 |
| RAM | B2 부모 프로세스 peak RSS2,956,546,048bytes. worker 전체 RAM을 포함하는 수치가 아님 |
| 현재 CPU 채점 | `team-final-answer-v8`로 세900 전수 재생. 새 추론/GPU/LLM/학습 없이 원문과 입력·정답·종료 사유 불변 검증 |

MMMU-Pro는 Assignment #1의900문항에 포함하지 않는다. 준비기가 별도로 취득하는
`MMMU/MMMU_Pro` Standard 10-way의 revision은 `563f3e84bb3b90893083a1f039cfa13077f2302b`다.
이는 팀의 보관 버전 고정이며 수업의 MMMU revision 지정과 다르다. 다운로드/hash 확인만 하고 평가·학습하지 않는다.

<!-- REPORT_RUNTIME:BEGIN -->
### COMPLETE900 실행 실측

- backend: `vllm` `0.11.0`, batch `2`; GPU: NVIDIA GeForce RTX 3090 (24576 MiB)
- effective generation record: `{'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.0, 'presence_penalty': 1.5, 'max_new_tokens': 32768, 'num_beams': 1, 'max_model_len': 40960}`
- evaluation loop: 89561.304 s; model load: 27.700 s; total invocation: 89608.979 s
- GPU memory: 22.87 GiB; source=job stage `evaluation`, 500 ms 간격 179106회, scope=`whole_device_sampled_not_allocator_peak`. CUDA/vLLM allocator peak가 아니라 해당 single-GPU device의 sampled used-memory 관측 최대치이며 driver/other process overhead를 포함할 수 있다.
<!-- REPORT_RUNTIME:END -->

아래 명령은 **실행 예시**이며 이번 작업에서 GPU를 다시 실행하지 않았다.
모델·데이터 경로는 실행자가 환경변수로 지정한다. 초기 설치/doctor는 `scripts/reproduce.sh`,
실행은 `scripts/eval.sh`가 담당한다. B2 원래 추론은 V4를 사용했고,
V8은 원본을 보존하는 별도 CPU 후처리 단계다. 향후 새 추론에는 B2와 채점기만 다른
`mmmu_val_v8.yaml`을 사용한다. 아래 명령을 실행했다고 보고하지 않는다.

```bash
bash scripts/eval.sh \
  --protocol configs/eval/mmmu_val_v8.yaml \
  --hardware configs/hardware/rtx3090_24gb.yaml \
  --model-ref manifests/models/baseline.json \
  --model-path "$MMDL_MODEL_PATH" \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" \
  --job-id "$MMDL_JOB_ID" --run-role evaluation --require-commit "$MMDL_CODE_COMMIT" \
  --run-id "${MMDL_JOB_ID}-evaluation" --mode full

bash scripts/rescore.sh --artifact-root "$MMDL_ARTIFACT_ROOT" \
  --run-id "${MMDL_JOB_ID}-evaluation" --parser team-final-answer-v8 \
  --output-dir "$NEW_RESCORE_DIR"

python -m scripts.validate_rescore \
  --source-run "$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation" --scored-dir "$NEW_RESCORE_DIR"
python -m scripts.report_baseline \
  --run-dir "$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation" \
  --scored-dir "$NEW_RESCORE_DIR" --output Assignment_1.md
```

기존 FROZEN GPU protocol의 parser를 사후에 바꾸지 않았다. V8 실행 소스 snapshot과 hash는
§4에 고정했다. 원본 GPU 추론 commit과 제출 코드 commit은 구분하며,
제출 checkout의 전체 SHA는 `git rev-parse HEAD`로 확인한다. 대용량 기존 응답 원장은 Git에
포함되지 않으며, 기존 응답의 동일 재채점과 새900 추론 재현은 구분한다.

수업 §1.3·§5는 backend/batch 선택을 문서화하면 허용한다. vLLM 선택 이유는 장문 응답의
연속 배치로 GPU 작업을 겹치기 위해서다. `LLMEngine.add_request()/step()`에서 최대2개를 유지하고
끝난 요청은 저장한 뒤 다음 독립 문제를 즉시 투입한다. 두 문제 완료를 같이 기다리는 방식이 아니다.
eager 실행, `async_scheduling=false`, prefix cache 비활성을 유지했다.
`deterministic=false`는 PyTorch 결정론 강제를 주장하지 않는다는 뜻이며 ID별 seed는 고정한다.
입력 hash와 engine prompt IDs 대조를 통과했지만 engine 내부 pixel tensor나 교차 GPU 동일 출력까지 보증하지 않는다.
요청별 `generation_seconds`는 step 시간을 활성 요청 수로 나눈 분석값이고 실제 겹친 지연시간은
`request_latency_seconds`다. 시간 합을 전체 wall time과 같다고 해석하지 않는다.
30과목별 점수·생성/샘플 시간은 [공개 CSV](https://github.com/jang2296/MMDL/blob/feat/mmmu-baseline/results/scoring-v8-20260924/subject_scores.csv)에
세 실행별로 기록했다. `sample_seconds`도 동시 실행이 겹치는 샘플 시간의 합이다.

## 2. 프롬프트

실제 P0 템플릿은 다음이며 [Qwen 공식 MMMU builder `build_mmmu_prompt`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)를 기준으로 저장소에 고정했다. 참고 원본은 `third_party/qwen3_vl/run_mmmu.py`, 실제 템플릿은 `prompts/`에 있다.

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

| 설정 | R1:2K | R2:이전32K | B2:고해상도32K |
|---|---:|---:|---:|
| 출력 상한 `max_new_tokens` | 2048 | 32768 | 32768 |
| 입력+생성 문맥 `max_model_len` | Transformers 경로 | 36864 | 40960 |
| `min_pixels` | 262144 | 262144 | 1003520 |
| `max_pixels` | 1310720 | 1310720 | 4014080 |
| resize 경로 | pinned HF processor | pinned HF processor | qwen-vl-utils0.0.14 → pinned HF processor |
| 추론 | Transformers reference | vLLM 동시2 연속 처리 | vLLM 동시2 연속 처리 |
| `finish_reason=length` | 226/900 | 75/900 | 90/900 |

출력32768은 [Qwen 공식 MMMU 평가 source](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의
평가 설정을 따른다. [지정 모델 card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md)의
VL 길이16384와 구분하며, 수업이32768을 강제했다고 하지 않는다.
R1의2048은 초기 엔지니어링 제한이었다. 긴 출력의 잘림과 비용/시간을 함께 관측하기 위해 사용자 승인으로 상향했고
이번 V8 작업에서는 생성 길이·sampling·이미지·프롬프트를 전혀 바꾸지 않았다.

R1/R2 이미지 budget은 고정 Qwen README의256–1280 visual-token 예시에서 채택한 팀 선택이었다.
B2는 같은 공식 `run_mmmu.py/build_mmmu_prompt`의 `1280×28×28`과 `5120×28×28`을
pixel budget으로 그대로 적용하고 공식 이미지 도구를 먼저 거치는 경로로 통제했다.
28계수를 Qwen3-VL 실제 patch격자라고 해석하지 않는다. 실제 patch16·merge2에 따른 resize 격자는32배수다.
가로·세로 고정 해상도가 아니며 aspect ratio별 smart resize를 적용하고 HF 단계에서 이중 resize를 막는다.
CPU900 대조에서 B2 최대입력5627이므로5627+32768=38395를 수용하도록 context40960을 선택했다.
해상도 확대는 시각 정보 손실을 줄이려는 실험 조건이지 성능 향상 보장이 아니고 VRAM/추론시간 부담을 늘린다.

## 4. 채점(파싱) 방식

**사용한 로직:** 팀 자체 `team-final-answer-v8`
(`src/mmdl/evaluation/final_answer_parser_v8.py`), 비교 정책 `single-final-mmmu-round2-v1`.
[MMMU 공식 eval_utils.py](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)
commit `268471d0d488258990025331c7528359c324aa25`(Apache-2.0)을 부분 차용했다.
최종답 구간 선택/MCQ 추출/단일 주관식 비교/실패 정책은 자체 구현이다. 공식
`normalize_str`의 숫자 소수점 두 자리 반올림 정책을 재사용한다.
여러 숫자를 후보로 만드는 `parse_open_response`와 문자열 부분 일치의 `eval_open`은
V8 비교에 사용하지 않는다. V6 모듈에서는 표기·유형·선택지 검증 보조 함수만 재사용한다.
Qwen 공식 LLM judge나 MMMU 전체 파서를 그대로 재현한 것은 아니다.

**자체 구현 이유:** 풀이 전체에서 gold와 우연히 같은 수치/단어를 찾는 문제를 줄이기 위해
답 선언/boxed/말미 답으로 비교 범위를 제한한다. V6의 조기 label 반환·여러 수치 중
하나 일치·문자열 부분 일치 경로를 제거하면서, V7이 잃었던 정상 표기와 부정/철회를
구별하기 위한 자체 구현이다. 완전한 자연어 의미 판정기라고 주장하지 않는다.

1. 원문을 보존하고 추출용 복사본에서 Unicode/Markdown/지원 LaTeX 표기를 정리한다.
2. Final/Answer/Correct/Best 선언을 응답 순서로 읽는다. 가정·타인의 인용은 선언과 구별한다.
   답의 여러 줄 수식·boxed·자기 결론 blockquote를 지원한다. 선언 사이의 다른 답은
   명시적인 정정 관계가 확인되어야 갱신하며, 해결되지 않은 충돌은 `NO_PARSE`다.
3. MCQ는 단일 유효 label 또는 유일한 보기 내용을 추출한다. `B or C`, `B is not correct`,
   label과 다른 보기 내용의 충돌을 차단한다. `A. L sin θ`, 전류 단위 `A`, 보기 자체의
   `Not enough information`은 다른 label/철회로 오인하지 않도록 제한된 문법을 사용한다.
   정확한 보기 일치 뒤의 철회도 검사한다.
4. 선언이 없을 때만 말미의 단독 답·boxed·정확한 보기·검증된 결론 패턴을 확인한다.
   명시적 마지막 선언이 잘못되었으면 풀이 중간 답이나 과거 파서로 돌아가지 않는다.
5. gold 없이 추출한 후 비교한다. 주관식은 단일 수치 또는 정규화한 문자열 전체를 비교한다.
   숫자는 MMMU의 float/소수점 두 자리 반올림을 유지한다. 예:0.9632와0.963은 둘 다0.96으로
   비교된다. 정확 소수 정책은 이번 버전이 아니다. 유리수/간단한 LaTeX 분수는 지원하지만
   튜플/일반 수식 동치·자동 단위 환산은 지원하지 않는다. 데이터 gold 목록은 별칭 목록이다.
6. 질문으로 확인한 요청 단위·단계·point 등 제한된 형식만 해석한다. `10 V`를 전압 질문의
   숫자10과 비교할 수 있지만 `10 A`는 인정하지 않는다. `Step 2`는 step을 요청한 질문에서만
   2로 해석한다. 임의 `question_schema`나 gold 기반 단위 추정은 쓰지 않는다.
7. 원문·추출값·상태·규칙·근거를 별도 필드로 보존한다. `EMPTY/NO_PARSE`는0점이며 분모900 유지.
   `length`에서 완성된 답 뒤 설명만 잘린 경우 수용할 수 있지만, 철회/미완성 정정은 미확정이다.
   마지막 구두점 유무만으로 판단하지 않는다.

LLM·무작위 fallback·모델 재생성·gold 맞춤 예외는 없다. 합성33개 사양은 위 정책에 맞게
수정하여 실행한다(기존 미구현 schema 제거, ordered pair 미지원, P17 반올림 정책 유지).
기존 V6/V7 변경64건은 **개발 회귀 자료**이며 독립 holdout이 아니다. 그 감사도 주로 답 주변의
텍스트를 검토했으므로, V8에서 전체 선언의 충돌을 확인하면 이전의 '명확' 분류와 달라질 수 있다.
소스·집계 검증과 사람에 의한 모든 응답의 의미 인증은 구분한다.

- 통합 scoring SHA256: `b80a28a4b66906d5c4f1ece046bd7c4be2b447445da4b072144e52ce7ee047fa`.
- V8 module SHA256: `2beb1230283cb882f47d6e8af72b01180847fd7aa4b7ac640a5173e391f3f891`.
- 공개 검증 기록: `results/scoring-v8-20260924/provenance.json`.
  별도 보존 원장의 `scorer_source.json`과 `scorer_source/`에는 의존 파일 snapshot/hash를 고정했다.
- 원본과 재채점의900 ID,30과목,record/JSONL hash,비채점 필드 불변,전체 재실행 결과,
  과목 CSV·분모 산술을 독립 검사한다. gold 교체 시 추출 불변도2700개 확인한다.
- V6/V7 결과와 원래 GPU protocol은 보존한다. V8의 온라인 연결은 CPU mock으로 검사했으며
  새 GPU900/RTX4090 실행을 주장하지 않는다.

<!-- REPORT_DYNAMIC:BEGIN -->
## 5. 결과

| No. | Subject | Data Num | Acc |
|---:|---|---:|---:|
| 1 | Accounting | 30 | 66.67% |
| 2 | Agriculture | 30 | 50.00% |
| 3 | Architecture_and_Engineering | 30 | 43.33% |
| 4 | Art | 30 | 60.00% |
| 5 | Art_Theory | 30 | 80.00% |
| 6 | Basic_Medical_Science | 30 | 76.67% |
| 7 | Biology | 30 | 46.67% |
| 8 | Chemistry | 30 | 50.00% |
| 9 | Clinical_Medicine | 30 | 76.67% |
| 10 | Computer_Science | 30 | 53.33% |
| 11 | Design | 30 | 73.33% |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67% |
| 13 | Economics | 30 | 83.33% |
| 14 | Electronics | 30 | 40.00% |
| 15 | Energy_and_Power | 30 | 46.67% |
| 16 | Finance | 30 | 70.00% |
| 17 | Geography | 30 | 56.67% |
| 18 | History | 30 | 73.33% |
| 19 | Literature | 30 | 80.00% |
| 20 | Manage | 30 | 63.33% |
| 21 | Marketing | 30 | 86.67% |
| 22 | Materials | 30 | 40.00% |
| 23 | Math | 30 | 63.33% |
| 24 | Mechanical_Engineering | 30 | 43.33% |
| 25 | Music | 30 | 20.00% |
| 26 | Pharmacy | 30 | 63.33% |
| 27 | Physics | 30 | 80.00% |
| 28 | Psychology | 30 | 83.33% |
| 29 | Public_Health | 30 | 86.67% |
| 30 | Sociology | 30 | 66.67% |
|  | **Overall (macro avg)** | **900** | **62.00%** |

계산식: `mean(30개 과목 accuracy) = correct/900 = 558/900`.

## 6. 공식 수치와의 비교

| | Overall (MMMU validation) |
|---|---:|
| 수업이 제시한 공식 비교값 | 67.4 |
| 우리 재현 결과 | 62.00 |
| 차이 (Δ, percentage points) | -5.40 |

## 7. 격차 분석

관측값: NO_PARSE 97개, EMPTY 0개, length 종료 90개이며, 이 run의 backend는 vllm, max_new_tokens는 32768이다. 이 값들은 공식 67.4와의 차이를 설명할 후보 관측치일 뿐 인과를 증명하지 않는다. 프롬프트/processor/revision/seed 정책, 생성 길이, parser와 runtime의 차이는 저장된 resolved config·raw response로만 검토한다.
<!-- REPORT_DYNAMIC:END -->

## 8. 기타 특이사항 / 한계

| 동일 저장 응답 | V6 | V7 (이전 개발 버전) | V8 (현재) |
|---|---:|---:|---:|
| R1: 2K | 479/900 · 53.22% | 463/900 · 51.44% | 478/900 · 53.11% |
| R2: 이전32K | 558/900 · 62.00% | 539/900 · 59.89% | 554/900 · 61.56% |
| B2: 고해상도32K | 560/900 · 62.22% | 539/900 · 59.89% | 558/900 · 62.00% |

- 미리 선택한 B2 응답을 제출 baseline으로 유지했다. 파서별 최고 점수를 골라 제출하지 않는다.
  V2의 R2 64.89%는 다른 fallback/비교 정책의 진단값이며 현재 점수와 동일 척도가 아니다.
- V8은 기존 감사의 명확한 V7 누락46건 중39건, 별도 step 정책 누락2건 및 V7의 유효 회수4건을 수용했다.
  남은7건은 qualifier를 보존하는 주관식3건과 전체 응답의 미해결 최종답 충돌4건이다.
  특히 `2960 unfavorable`을 숫자2960으로 축약하지 않는 것은 **정상 표현을 놓칠 수 있는 알려진 한계**다.
  이를 V6의 거짓 정답을 제거한 성과라고 주장하지 않는다.
- V6→V8은 세 실행 합산4건 증가·11건 감소다. 감소11건은 위7건 및 모호/잘린 충돌4건이다.
  점수 하락 전부를 모델 오답 교정으로 해석하지 않는다. 합성 부정/복수 후보/부분일치 반례 차단과
  실제 응답에서의 의미 정확성 인증은 별개다.
- 167개 파이프라인 CPU 단위 검사, Ruff·mypy·Bash 문법 검사와2700개 원장 재생/무변조 검증을 통과했다.
  validation 응답을 개발 회귀 자료로 사용했으므로 독립 holdout이 아니며,
  모든 이미지 재풀이·전체 응답의 사람 의미 검증을 완료한 것은 아니다.
- 생성 조건은 변경하지 않았다. 새 GPU 추론·4090 검증·fine-tuning·상세 모델 오답 분석은 미실행이다.
  R1↔R2는 backend/길이, R2↔B2는 전처리/해상도/context/호스트 등이 달라 해상도의 인과효과로 단정하지 않는다.
- 재채점 원장·V8 HTML·원본 응답은 별도 보존했다. 공개 저장소에는 작은 확정 집계/hash와
  새 실행·재채점에 필요한 코드만 포함한다. 기존 원장 전체의 배포/접근은 별도이며 Git clone만으로 복원되지 않는다.
- 제출 정본은 `Assignment_1.md`; `reports/mmmu_baseline.md`는 동일 본문,
  `assignment/assignment1.md`는 본문 링크다. 실행·설치 방법은 `README.md`로 모았다.
