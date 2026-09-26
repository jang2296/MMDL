# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: Team 2
- **팀원**: 장우석, 정예찬, 김기원, 김정현
- **작성일**: 2026-09-25
- **재현 커맨드**: 환경 설치 후 아래 §1의 `bash scripts/eval.sh ...` 실행

---

## 1. 환경 / 재현성

<!-- REPORT_RUNTIME:BEGIN -->
| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 모델 정밀도 | `bfloat16`, 비양자화. Processor·Tokenizer도 모델과 동일 revision |
| 평가 데이터 | `MMMU/MMMU`, revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, `validation` 900문항 |
| 추론 백엔드 | `vllm` `0.11.0`, 최대 2개 요청의 연속 배치 |
| 사용 GPU | NVIDIA GeForce RTX 3090 (24576 MiB) |
| 실측 peak VRAM | 장치 전체 사용량의 관측 최대 22.87 GiB (500 ms 간격). vLLM allocator의 정확한 peak는 미측정; driver·다른 프로세스 사용량이 포함될 수 있음 |
| 총 소요 시간 | 평가 89561.304 s; 모델 로드 27.700 s; 전체 호출 89608.979 s (설치·다운로드 제외) |
| 주요 환경 | Linux, Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.57.1 |
| 의존성 | [env/requirements-vllm.lock](../env/requirements-vllm.lock) |
<!-- REPORT_RUNTIME:END -->

설치는 [README.md](../README.md)를 따른다. vLLM은 한 문항이 끝나면 다음 문항을 바로 투입하여,
문항별 응답 길이가 다른 평가에서 동시 처리를 활용하기 위해 선택했다.
GPU에는 모델 전체를 올리며 CPU·디스크 가중치 offload는 사용하지 않는다.

README의 환경 설치와 `HF_HOME`, `MMDL_DATA_ROOT`, `MMDL_ARTIFACT_ROOT` 설정을 완료한 뒤,
깨끗한 제출 commit의 저장소 루트에서 실행한다.

```bash
bash scripts/eval.sh \
  --protocol configs/eval/mmmu_val_v8.yaml \
  --hardware configs/hardware/rtx3090_24gb.yaml \
  --model-ref manifests/models/baseline.json \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" \
  --artifact-root "$MMDL_ARTIFACT_ROOT" \
  --public-root "$MMDL_ARTIFACT_ROOT/public" \
  --job-id baseline-submission \
  --run-role evaluation \
  --require-commit "$(git rev-parse HEAD)" \
  --run-id baseline-submission-evaluation \
  --mode full
```

`--model-path`는 지정 revision의 로컬 모델 경로로 교체할 수 있다. 데이터와 결과 저장 위치는
환경변수로 변경한다. 기본 데이터 위치에서는 고정 revision을 자동 취득하며, 임의의 별도
`--data-root`는 검증 manifest와 validation 데이터가 준비되어 있어야 한다.
재실행 시에는 `--job-id`와 그 뒤에 `-evaluation`을 붙인 `--run-id`를 함께 새 이름으로 바꾼다.

RTX 4090에서는 하드웨어 설정을 `configs/hardware/rtx4090_24gb.yaml`로 변경한다.
실제 측정 장비와 교차 GPU 검증 범위는 §8에 명시한다.

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문**은 다음과 같다.

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

변수 구성:

- `{question}`: 데이터셋의 원본 질문.
- `{hint_prefix}`: hint가 있는 경우 `Hint: {hint}`와 줄바꿈을 삽입하며, 없으면 빈 문자열.
- `{options}`: 원본 보기 순서에 따라 `A. 보기 내용`, `B. 보기 내용` 등을 각각 줄바꿈하여 구성.
  보기 개수를 A–D로 제한하지 않는다.

이미지를 원본 참조 순서대로 먼저 전달하고, 마지막에 위 텍스트를 붙인다. 전체 메시지에는
지정 모델 revision의 chat template을 `apply_chat_template(add_generation_prompt=True)`로 적용한다.
정답·해설은 입력에 포함하지 않으며, 추가 system·few-shot·단계별 풀이 지시는 사용하지 않는다.

- **출처**: [Qwen 공식 MMMU 평가 코드의 `build_mmmu_prompt`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py).
  실제 템플릿은 [객관식](../prompts/mmmu_mcq_v1.txt)과 [주관식](../prompts/mmmu_open_v1.txt)에 있다.
- **선택 이유**: 모델 제공사가 공개한 평가 코드에 근거하여 입력을 구성하고 임의의 추가 지시를
  최소화하기 위해 선택했다. 공개 프롬프트를 따른다는 의미이며, 공식 점수를 산출한 전체 실행을
  동일하게 재현했다는 의미는 아니다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | `true` |
| `temperature` | `0.7` |
| `top_p` | `0.8` |
| `top_k` | `20` |
| `repetition_penalty` | `1.0` |
| `presence_penalty` | `1.5` |
| `seed` | 기준값 `3407`; 문항별 seed는 아래 규칙으로 계산 |

- **출처**: [Qwen 공식 README의 Evaluation Reproduction — Instruct models](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction).
- **선택 이유**: 임의의 greedy 설정으로 바꾸지 않고 모델 제공사가 안내한 Instruct 평가 recipe를 사용했다.

문항별 seed는 다음과 같이 계산한다.

```text
SHA256("3407:{sample_id}")의 첫 8바이트를
big-endian 정수로 해석한 값 mod 2^31
```

이 방식은 실행 순서나 재개 여부와 관계없이 문항별 seed를 유지하기 위한 **팀 자체 정책**이다.
공식 README의 기준 seed는 3407이지만, 공개 MMMU 실행 코드에는 엔진 seed 42가 사용된다.
따라서 공식 실행과 난수 사용 순서까지 동일하다고 주장하지 않는다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | `32768` |
| `min_pixels` | `1003520` |
| `max_pixels` | `4014080` |
| 이미지 처리 | qwen-vl-utils 0.0.14로 비율을 유지하여 resize한 뒤 지정 revision의 Processor에 전달 |
| 전체 문맥 한도 | `40960` tokens |

**선택 근거**

출력 상한 32,768은 Qwen의 공개 평가 설정을 따랐다. 긴 풀이가 중간에 잘리는 위험을 줄이기 위한
선택이며, 항상 32,768토큰을 생성하도록 요구하는 것은 아니다. 실제 상한 도달 건수는 §7에 기록했다.
[공식 생성 설정](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction)

이미지 픽셀 범위는 공개 MMMU 코드의 `1280×28×28`과 `5120×28×28`을 적용했다.
고정된 가로·세로 크기가 아니라 이미지별 픽셀 예산이며, 세부 시각 정보 손실을 줄이려는 선택이다.
28이라는 계수를 이 모델의 실제 patch 격자로 간주하지 않으며 실제 resize는 Processor 규격에 맞춘다.
이미지 입력 확대는 메모리와 처리 시간 부담을 늘리며 정확도 향상을 보장하지 않는다.
[공식 이미지 처리 설정](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)

입력 사전 점검에서 최대 5,627토큰을 확인하여, 입력과 최대 출력의 합인 38,395토큰을 수용하도록
문맥 한도를 40,960으로 설정했다. 메모리가 부족하더라도 해상도·출력 길이·모델 정밀도를 자동으로 낮추지 않는다.

## 4. 채점(파싱) 방식

- **사용한 파서/로직**: 팀 자체 규칙 기반 최종답 추출기. 구현 식별자는 `team-final-answer-v8`이며,
  [구현 코드](../src/mmdl/evaluation/final_answer_parser_v8.py)를 저장소에 포함했다.
- **참고·부분 차용**: [MMMU 공식 `eval_utils.py`](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)의
  숫자 정규화 함수를 사용한다. 최종답 구간 선택, 모호성 검사와 문자열 전체 비교는 자체 구현이다.

**자체 구현 이유**

LLM 판정 없이 일정한 규칙으로 평가하되, 풀이 중간에 정답과 같은 숫자나 단어가 우연히 등장했다는
이유로 정답 처리하지 않기 위해 최종답을 먼저 추출하도록 설계했다.

**동작 방식 요약**

1. 원본 응답을 보존하고 추출용 복사본에서 Markdown·공백·지원하는 수식 표기를 정규화한다.
2. `Final answer`, `Answer` 등의 선언을 응답 순서대로 확인한다. 부정·가정·인용을 구별하고
   답의 철회나 정정을 검사한다.
3. 명시적인 선언이 없을 때만 말미의 단독 답, `boxed` 답, 정확한 보기 내용 또는 지원하는 결론 표현을
   확인한다. 모호한 최종 선언이 있으면 앞의 풀이에서 다른 답을 찾아 구제하지 않는다.
4. **객관식**은 실제 보기 중 하나를 추출하여 정답과 비교한다. 복수 후보나 해결되지 않은 답 선언
   충돌은 미확정으로 처리한다.
5. **주관식**은 단일 수치 또는 정규화한 문자열 전체를 비교한다. 숫자는 공식 함수의 소수점 두 자리
   반올림 규칙을 사용한다. 단위·단계 번호는 질문에서 확인되는 제한된 형식만 처리하며,
   일반적인 단위 환산이나 의미상 동의어 추정은 하지 않는다.
6. 빈 응답과 추출 실패는 오답으로 계산하되 별도 상태로 기록한다. 출력 상한에 도달했더라도 완성된
   답이 있고 관측된 철회·충돌이 없다면 수용할 수 있다.

정답 추출 단계에는 기준 정답을 전달하지 않는다. LLM 호출, 무작위 선택, 모델 재생성 또는 이전
파서로의 fallback은 사용하지 않으며 분모는 항상 900문항으로 유지한다.

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

계산식: `Overall = mean(30개 과목 accuracy) = 558/900 × 100 = 62.00%`.
종합 점수는 반올림 전 과목별 정확도로 계산했다. 전체 900문항을 평가했으며 누락·시스템 오류는 없다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| 공식 — 과제에서 제시한 Qwen3-VL Technical Report 수치 | 67.4% |
| 우리 재현 결과 | 62.00% |
| 차이 (우리 − 공식) | **-5.40%p** |

## 7. 격차 분석

Qwen 공개 평가 코드는 규칙으로 답을 추출하지 못하면 LLM 판정을 사용할 수 있지만, 본 프로젝트에서는 고정된 규칙만 사용한다. 따라서 자유로운 표현이나 답변 내 충돌을 처리하는 기준이 다르다. [공식 채점 방식](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/README.md#custom-evaluation-logic)

저장 응답에서 상한 종료와 자연 종료는 97개, 출력 상한 종료는 90개이고 두 조건의 겹침은 76개다. 따라서 독립적인 손실로 합산할 수 없다. 현재 발생한 오답의 경우 출력 상한에 도달하고 오답이 된 경우 85건, 자연 종료했지만 최종답 추출 실패 건 21건, 자연 종료·객관식 답 추출 성공·정답과 불일치 218건, 자연 종료·주관식 답 추출 성공·정답과 불일치 18건이다. 현재 실제로 채점 규칙 때문에 놓친 정답 사례로 Manage_20이 존재하며 데이터셋 정답: 2960, 모델의 최종답: $2,960 unfavorable 이지만 오답 처리 되었으며 이는 단순한 모델 오답이라기보다 비교 규칙의 지원 범위 때문에 생긴 손실이다. 반면 Qwen 공개 평가 코드는 규칙으로 선택지를 연결하지 못하면 LLM을 통한 판정이 가능하다. Qwen 공식 Instruct 평가에서는 gpt-3.5-turbo-0125을 활용하였지만 본 프로젝트에서는 외부 상용 LLM API를 사용하지 않으며 이는 공식 Qwen과는 다르게 불특정한 장소에서 모델을 평가하고 재현하는 특성상 외부 LLM API의 경우 맞지 않는 것으로 판단하여 제외하였다.

또한 32,768토큰에서  상한에 도달한 90문항은 전체 문항의 10%인데 토큰 부족으로 문제를 풀지 못 한 경우 이외에도 Accounting_9은 재고 수령을 현금 유입으로 볼 것인지 같은 해석을 반복하다가 종료되었으며 이는 파서가 명확한 정답을 놓쳤다기보다, 모델이 불확실성을 해결하지 못 하고 반복하면서 유효한 결론 도출을 못 한 것으로 판단되며 공개 Instruct 설정도 출력 상한 32,768을 사용한다는 점에서 토큰양의 유무 때문에 점수가 차이 나는 것은 아님을 알 수 있다.

그 외에도 난수 사용 방식이 공식은 seed=42라는 점에서 차이가 존재하며 이미지 토큰을 증가시켜 확인한 경우에도 이전 32K는 554개, 고해상도 32K는 558개가 정답인 점에서 해상도가 커져도 관계 해석, 계산, 반복 생성 문제가 자동으로 해결되지 않았다. 
<!-- REPORT_DYNAMIC:END -->

## 8. 기타 특이사항 / 한계

<!-- REPORT_STATUS:BEGIN -->
- 보고한 점수는 저장된 900개 응답을 변경하지 않고 §4의 최종 채점 규칙으로 다시 평가한 결과다. 시간·메모리는 해당 GPU 추론 실행의 측정값이다.
<!-- REPORT_STATUS:END -->
- 현재 채점 규칙을 포함한 새 900문항 GPU 실행과 RTX 4090 실측은 아직 수행하지 않았다.
  동일 seed라도 장비·실행환경에 따라 응답과 점수가 달라질 수 있다.
- 규칙 기반 파서는 모든 자연어 표현을 이해하지 못한다. 예를 들어 수치에 붙은 `unfavorable` 같은
  방향 표현을 처리하지 못해 유효한 답을 놓칠 수 있다. validation 응답을 파서 개발·검증에
  활용했으므로 독립적인 검증 자료로 간주하지 않는다.
- 코드·설정·작은 결과 집계는 GitHub에 포함하고 대용량 원본 응답과 HTML은 별도 보관한다.
  저장소의 명령으로 새 평가를 실행할 수 있지만 과거 응답을 그대로 재채점하려면 별도 보관된
  원본 결과가 필요하다. [점수 집계](../results/scoring-v8-20260924/summary.json),
  [과목별 집계](../results/scoring-v8-20260924/subject_scores.csv),
  [코드·검증 기록](../results/scoring-v8-20260924/provenance.json)을 함께 제공한다.
