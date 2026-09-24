# MMDL — Assignment #1

Qwen3-VL-4B-Instruct의 MMMU validation **900문항 평가 파이프라인**입니다.
현재 기준은 `team-final-answer-v8`입니다. 공식 점수에 맞춰 답을 고르지 않으며,
이후 학습 모델도 같은 프롬프트·생성·이미지·채점 조건으로 비교합니다.

- **보고서:** [Assignment_1.md](Assignment_1.md)
- **수업 지정 경로:** [reports/mmmu_baseline.md](reports/mmmu_baseline.md), [assignment/assignment1.md](assignment/assignment1.md)
- **확정 결과:** [점수·검증 요약](results/scoring-v8-20260924/summary.json), [30과목 × 세 실행 표](results/scoring-v8-20260924/subject_scores.csv)

| 저장된 추론 실행 | V8 CPU 재채점 |
|---|---:|
| 2,048토큰 | 478/900 · 53.11% |
| 이전 32,768토큰 | 554/900 · 61.56% |
| 고해상도 32,768토큰 — 제출 baseline | 558/900 · 62.00% |

새 GPU 추론이나 fine-tuning 점수가 아닙니다. 원래 RTX3090 실행의 응답은 보존했고
V8으로만 재채점했습니다. 공식 비교값67.4%와의 격차, 구현 출처와 한계는 보고서에 있습니다.
V8의 새 GPU 실행·실제 RTX4090 검증·학습은 아직 하지 않았습니다.

## 1. 환경과 저장 위치

Linux/WSL, Python3.12, NVIDIA CUDA 사용 가능 환경이 필요합니다.
권장 경로는 RTX4090/3090 24GB의 vLLM·BF16·동시2 연속 처리입니다.
가중치·MMMU 원문·이미지·대형 응답은 Git 저장소 밖에 저장합니다.

```bash
python3.12 -m venv "$HOME/mmdl-env"
source "$HOME/mmdl-env/bin/activate"
pip install -r env/requirements-vllm.lock
pip install --no-deps -e .
export HF_HOME="$HOME/mmdl-cache/huggingface"
export MMDL_DATA_ROOT="$HOME/mmdl-data"
export MMDL_ARTIFACT_ROOT="$HOME/mmdl-artifacts"
export CUDA_VISIBLE_DEVICES=0
export MMDL_JOB_ID=baseline-v8
bash scripts/test_reference.sh
```

vLLM lock은 실제 추론 환경에 검사 도구(Ruff/mypy 및 의존 패키지)의 고정 버전을 추가한 것이다.
원본 실행 당시 환경은 별도 `environment.json` 기록과 구분한다.

`--model-path`는 아래 공식 ID 또는 동일 revision의 로컬 snapshot을 받습니다.
`--data-root`는 기본 위치에서 자동 취득하거나, 준비된 `manifest.json`과 30개
validation parquet가 있는 경로를 받습니다. 다른 경로는 환경변수/인자로 바꾸면 됩니다.
고정 revision·파일 hash·BF16을 검사하며, OOM이라고 모델/해상도/길이를 자동 변경하지 않습니다.

| 대상 | 고정 revision | 현재 용도 |
|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | BF16·비양자화; processor/tokenizer도 동일 revision |
| `MMMU/MMMU` | `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` | validation 30과목 × 30문항 = 900 평가 |
| `MMMU/MMMU_Pro` | `563f3e84bb3b90893083a1f039cfa13077f2302b` | Standard 10-way 다운로드·hash 확인만; 평가하지 않음 |

앞의 두 revision은 Assignment #1 지정값이다. MMMU-Pro는 별도 저장소이며 그 revision은
팀이 취득한 버전을 고정한 값이지 과제의 MMMU revision을 재사용한 것이 아니다.
준비 단계는 모델·데이터를 Hugging Face에서 취득하거나 동일 revision의 캐시를 검증해 재사용한다.
GitHub 웹에서 실행 버튼을 누르는 방식이 아니라, GPU 환경에서 저장소를 clone한 뒤 아래 Bash 명령을 실행한다.

## 2. 900문항 평가 — 한 명령

깨끗한 Git checkout에서 실행합니다. 아래 명령은 실제 GPU 추론을 시작합니다.
3090에서는 hardware 파일만 `configs/hardware/rtx3090_24gb.yaml`로 바꿉니다.

```bash
bash scripts/eval.sh --protocol configs/eval/mmmu_val_v8.yaml --hardware configs/hardware/rtx4090_24gb.yaml --model-ref manifests/models/baseline.json --model-path Qwen/Qwen3-VL-4B-Instruct --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --artifact-root "$MMDL_ARTIFACT_ROOT" --public-root "$MMDL_ARTIFACT_ROOT/public" --job-id "$MMDL_JOB_ID" --run-role evaluation --require-commit "$(git rev-parse HEAD)" --run-id "${MMDL_JOB_ID}-evaluation" --mode full
```

프롬프트P0, 출력32768, sampling0.7/0.8/20·presence1.5, 이미지1003520–4014080,
context40960을 고정합니다. 다른 YAML은 기존 실행 재현·회귀 검사에 필요한 버전이며
현재 제출용 진입점은 **mmmu_val_v8.yaml 하나**입니다.

학습 후에는 `--model-ref`와 `--model-path`를 해당 checkpoint manifest/path로 바꿉니다.
`full/merged/adapter`의 base revision·학습 설정·데이터 hash·artifact 출처를 요구합니다.
실제 학습 코드/학습 결과는 아직 완료되지 않았으며, 빈 학습 구현을 제출하지 않습니다.

## 3. 결과 확인·CPU 재채점·보고서

`$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation/`에 원문 JSONL, 샘플 JSON,
과목별 점수/시간 CSV, 환경·설정·입력/소스 hash와 `review.html`이 저장됩니다.
공유 이미지는 외부 `assets/images/`에 있습니다. 시스템 오류/누락은 `INCOMPLETE`이며,
빈 답/추출 실패는 전체900 분모에 포함합니다.

```bash
export NEW_RESCORE_DIR="$MMDL_ARTIFACT_ROOT/rescores/${MMDL_JOB_ID}-v8"
bash scripts/rescore.sh --artifact-root "$MMDL_ARTIFACT_ROOT" --run-id "${MMDL_JOB_ID}-evaluation" --parser team-final-answer-v8 --output-dir "$NEW_RESCORE_DIR"
python -m scripts.validate_rescore --source-run "$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation" --scored-dir "$NEW_RESCORE_DIR"
python -m scripts.report_baseline --run-dir "$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation" --scored-dir "$NEW_RESCORE_DIR" --output Assignment_1.md
```

재채점은 GPU/LLM 없이 수행하며 기존 결과 폴더를 덮어쓰지 않습니다.
보고서 생성은 원본 GPU 실측과 새 CPU 점수를 분리하고 두 제출 경로를 동기화합니다.
원본 응답을 배포하지 않아도 위 평가 명령으로 데이터를 정식 취득해 새 실행할 수 있습니다.
기존 응답의 완전 동일 재채점에는 별도 보존 원장이 필요하며 이 저장소에 포함되지 않습니다.

## 4. 저장소 범위

`src/` 평가기, `scripts/` 실행·검증, `configs/` 고정 조건, `env/` exact lock,
`prompts/` P0, `manifests/` 모델/데이터 hash, `third_party/` 고정 출처·license,
`tests/` CPU 회귀 검사, `results/` 작은 확정 집계만 포함합니다.
내부 에이전트 지침, 개발 대화/감사 폴더, 강의 PDF, 모델·데이터 원문은 공개하지 않습니다.
