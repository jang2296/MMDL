# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 미제공
- **팀원**: 미제공
- **작성일**: 2026-09-21
- **재현 커맨드**: 실행 전 — `bash scripts/eval.sh`의 parameterized command는 README에 기록됨

Run status: **NOT RUN**. A COMPLETE900 artifact does not exist yet; no score is fabricated here. The final report is generated from a verified external run with `python scripts/report_baseline.py --run-dir <MMDL_ARTIFACT_ROOT>/runs/<run-id> --output reports/mmmu_baseline.md`.

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (`ebb281ec70b05090aa6165b016eac8ec08e71b17`), dtype=`bfloat16` (로컬 PARTIAL 검증 완료; 제출용 900문제 미실행) |
| 추론 백엔드 | `transformers` (resolved protocol target; execution not run) |
| 사용 GPU | 미실측 |
| 실측 peak VRAM | 미실측 |
| 총 소요 시간 | 미실측; download verification, model load, evaluation, and whole-run timings are not available |
| 의존성 | [env/requirements-eval.lock](../env/requirements-eval.lock) (installed/runtime versions not yet used for a COMPLETE900 run) |
| 실행 커맨드 | ```bash
bash scripts/eval.sh --protocol configs/eval/mmmu_val_v1.yaml \
  --hardware configs/hardware/rtx4090_24gb.yaml \
  --model-ref manifests/models/baseline.json \
  --model-path "$MMDL_MODEL_PATH" \
  --data-root <MMDL_DATA_ROOT>/evaluation/mmmu \
  --run-id baseline-4090 --mode full
``` |

## 2. 프롬프트

실행 전 확정된 P0 source template 전문:

### MCQ template (`prompts/mmmu_mcq_v1.txt`)

```text
{hint_prefix}Question: {question}
Options:
{options}Please select the correct answer from the options above.
```

### Open-ended template (`prompts/mmmu_open_v1.txt`)

```text
{hint_prefix}Question: {question}
```

Hint policy: non-empty sanitized source `hint` is rendered as `Hint: {hint}\n` before `Question`; absent or blank hints render nothing. Open-ended prompts keep that hint policy, omit options and the MCQ selection instruction, and are not converted to MCQ. Gold answer and explanation fields are excluded from inference input.

- **출처**: Qwen3-VL official `evaluation/mmmu/run_mmmu.py`의 `build_mmmu_prompt`; pinned source comparison is recorded in [docs/SOURCE_REVIEW.md](../docs/SOURCE_REVIEW.md).
- **선택 이유**: official MMMU option order and selection instruction are retained. Actual image objects are placed before one final text content item, with original `<image n>` references preserved.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | `true` (protocol target; execution not run) |
| `temperature` | `0.7` |
| `top_p` | `0.8` |
| `top_k` | `20` |
| `repetition_penalty` | `1.0` |
| `presence_penalty` | `1.5` (generated-only custom implementation) |
| `seed` | master `3407`; `per_sample_sha256_v1` team policy |

- **출처**: Qwen3-VL pinned model-card Evaluation Reproduction recipe. The reference script seed and this team seed policy are distinguished in [docs/SOURCE_REVIEW.md](../docs/SOURCE_REVIEW.md).

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | `2048` (팀의 제한된 생성 예산; 로컬 Accounting30 중 15개가 상한 도달) |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | `262144` / `1310720`; resize owner `official_processor` (initial protocol target) |

**선택 근거**: 초기 팀 설정을 유지한 로컬 Accounting30에서 15/30이 길이 상한에 도달했다. 충분한 길이라고 주장하지 않으며, 상한 도달은 공식값 비교의 제한으로 기록한다. 제출용 4090 전체 평가의 잘림률·시간·VRAM은 아직 미실측이다. 이 값들은 공식 지정값이 아니며 부분 정답률에 따라 변경하지 않는다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: `third_party/mmmu/eval_utils.py`의 official open parser/evaluator와 local `mmmu-official-no-random-v1` MCQ adaptation.
- 동작 방식 요약: MCQ applies bracketed-label, separated-label, and option-content rules. The official random fallback is removed; unparseable output is `NO_PARSE` and scores false. Empty output is `EMPTY` and remains in the 900-sample denominator. Open responses use official normalization and evaluation.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
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
| | **Overall (macro avg)** | **900** | **미실행** |

계산식: `Overall = mean(30개 과목 accuracy) = correct/900`; 실행 전이므로 값은 미실행이다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 미실행 |
| 차이 (Δ) | 미실행 |

## 7. 격차 분석

아직 COMPLETE900 실행 결과가 없으므로 공식 67.4와의 차이를 진단할 관측치가 없다. 최종 보고서에서는 저장된 `finish_reason`, `NO_PARSE`, `EMPTY` 집계와 resolved runtime/length settings만으로 1000자 이내 진단하며, 관측되지 않은 원인을 단정하지 않는다.

## 8. 기타 특이사항 / 한계 (Optional)

- 이 문서는 실행 전 상태를 기록한 초기 보고서이며, 결과 표·peak 자원·실행 시간은 비워 두었다.
- 모델/데이터 다운로드, 1문제·1과목 smoke, 900문제 COMPLETE900 실행, 4090 실측은 각각 별도 증거가 필요하다.
- 상세 실패 사례 분석, 학습 데이터·학습법 제안, 개선 계획은 사용자 검토 후에 작성한다.
