# MMDL — Assignment #1

Qwen3-VL-4B-Instruct의 MMMU validation **900문항 평가 파이프라인**입니다.
현재 기준은 `team-final-answer-v8`입니다. 공식 점수에 맞춰 답을 고르지 않으며,
이후 학습 모델도 같은 프롬프트·생성·이미지·채점 조건으로 비교합니다.

- **Assignment #1 보고서:** [reports/mmmu_baseline.md](reports/mmmu_baseline.md)
- **제출 형식·기한:** 제공된 `SUBMISSION_TEMPLATE.md`의8개 항목, 2026-09-28 23:59(과제 안내 기준)
- **강의 PDF의 안내 경로:** [assignment/assignment1.md](assignment/assignment1.md) → 위 보고서 링크
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
다음 명령들은 저장소 루트에서 실행합니다. 처음 받는 경우:

```bash
git clone --branch feat/mmmu-baseline https://github.com/jang2296/MMDL.git
cd MMDL
git rev-parse HEAD
```

정확한 제출 버전을 재현할 때는 제출 기록의 전체 commit SHA로 `git checkout <commit-sha>`한 뒤
진행합니다. 이후 환경을 설치합니다.

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
기본 데이터 경로가 비어 있으면 평가 명령이 모델·데이터를 준비합니다. 데이터 `manifest.json`만
있고 모델 cache가 없는 사용자 경로라면, 평가 전에 다음 명령으로 모델과 데이터 manifest를 맞춥니다.

```bash
python -m mmdl.data.download --data-root "$MMDL_DATA_ROOT" --cache "$HF_HOME" --artifact-root "$MMDL_ARTIFACT_ROOT" --skip-pro
```

이 명령은 고정 revision을 취득·검증하고 MMMU-Pro 취득은 건너뜁니다. 위 준비 명령의
`--data-root`는 데이터 저장 루트이고, 아래 평가 명령의 `--data-root`는 그 아래
`evaluation/mmmu`입니다. 평가에 별도 데이터 경로를 지정한다면 그곳에 준비된
`manifest.json`과 30개 validation parquet가 있어야 합니다.
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

보고서의 프롬프트, 출력32768, sampling0.7/0.8/20·presence1.5, 이미지1003520–4014080,
context40960을 고정합니다.
CLI의 기본 설정과 현재 제출용 진입점은 **`configs/eval/mmmu_val_v8.yaml`**입니다.
다른 `configs/eval/` YAML은 이전 실행이나 회귀 확인용 설정입니다.

위 명령은 환경 설치 후 모델·데이터 준비 → 900문항 추론 → V8 채점 → 결과 저장을 수행합니다.
`scripts/reproduce.sh`는 별도의 RunPod 운영용 wrapper이며, persistent volume·예산/종료 정책
등 추가 환경 설정을 검사하는 경로입니다. 위 직접 평가 명령에는 그 운영용 설정이 필요하지 않습니다.

현재 공개 범위는 평가·저장 응답 재채점입니다. 학습과 데이터 증강은 구현·실행하지 않았습니다.
향후 학습 checkpoint 평가에는 `--model-ref`와 `--model-path`를 바꾸고,
`full/merged/adapter`의 base revision·학습 설정·데이터 hash·artifact 출처를 제공해야 합니다.

## 3. 결과 확인·CPU 재채점·보고서

`$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation/`에 원문 JSONL, 샘플 JSON,
과목별 점수/시간 CSV, 환경·설정·입력/소스 hash와 `review.html`이 저장됩니다.
공유 이미지는 외부 `assets/images/`에 있습니다. 시스템 오류/누락은 `INCOMPLETE`이며,
빈 답/추출 실패는 전체900 분모에 포함합니다.

```bash
export NEW_RESCORE_DIR="$MMDL_ARTIFACT_ROOT/rescores/${MMDL_JOB_ID}-v8"
bash scripts/rescore.sh --artifact-root "$MMDL_ARTIFACT_ROOT" --run-id "${MMDL_JOB_ID}-evaluation" --parser team-final-answer-v8 --output-dir "$NEW_RESCORE_DIR"
python -m scripts.validate_rescore --source-run "$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation" --scored-dir "$NEW_RESCORE_DIR"
python -m scripts.report_baseline --run-dir "$MMDL_ARTIFACT_ROOT/runs/${MMDL_JOB_ID}-evaluation" --scored-dir "$NEW_RESCORE_DIR" --output reports/mmmu_baseline.md
```

재채점은 GPU/LLM 없이 수행하며 기존 결과 폴더를 덮어쓰지 않습니다.
보고서 생성은 원본 GPU 실측과 새 CPU 점수를 분리하여 `reports/mmmu_baseline.md`만 갱신합니다.
보고서는 제공된 `SUBMISSION_TEMPLATE.md`의8개 항목을 따라 작성했으며, 루트에 중복 본문을 만들지 않습니다.
원본 응답을 배포하지 않아도 위 평가 명령으로 데이터를 정식 취득해 새 실행할 수 있습니다.
기존 응답의 완전 동일 재채점에는 별도 보존 원장이 필요하며 이 저장소에 포함되지 않습니다.

## 4. 저장소 범위와 디렉터리

```text
MMDL/
├── README.md         설치·평가·결과 확인 안내
├── pyproject.toml    Python 패키지 설치 정보
├── .env.example      외부 저장 경로·운영 환경변수 예시
├── .gitignore        대용량·비공개 파일 제외
├── assignment/       과제 지정 경로의 보고서 링크
├── configs/
│   ├── eval/         현재 기본은 mmmu_val_v8.yaml; 나머지는 이전 실행·회귀용
│   └── hardware/     GPU별 실행 자원 프로필
├── env/              고정된 평가·vLLM 의존성 lock
├── manifests/        모델·데이터 revision과 파일 hash
├── prompts/          평가 프롬프트 템플릿
├── reports/          제출 baseline 보고서
├── results/          작은 공개 점수·검증 요약
├── scripts/          평가·재채점·검증 진입점
├── src/mmdl/
│   ├── data/          고정 모델·데이터 취득
│   ├── evaluation/    MMMU 데이터, 프롬프트, backend, 채점, 결과 저장
│   └── runtime/       계약·환경·artifact·재현 지원
├── tests/unit/        CPU 회귀 검사
└── third_party/       출처가 고정된 evaluator·참고 코드와 license
```

결과 원문, 개별 응답, 이미지와 HTML 열람본은 `MMDL_ARTIFACT_ROOT` 아래에 저장하며
GitHub에는 작은 확정 집계만 포함합니다. 내부 에이전트 지침, 개발 대화/감사 폴더,
강의 PDF, 모델·데이터 원문은 공개하지 않습니다.
