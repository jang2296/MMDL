# Team 2 · v18b 학습과 평가 재현

| 시험과 모델 | V8 | V8 + CompassVerifier-7B |
|---|---:|---:|
| MMMU validation · Qwen 원본 | 558/900 (62.00%) | 566/900 (62.89%) |
| MMMU validation · v18b, 최종 slot 206 | 573/900 (63.67%) | 591/900 (65.67%) |
| MMMU-Pro Standard · Qwen 원본 | 미측정 | 미측정 |
| MMMU-Pro Standard · v18b, 최종 slot 206 | 미측정 | 미측정 |

65%를 넘은 값은 **일반 MMMU validation의 보완 점수**다. V8과 보완 판정은 같은 응답을 쓴다. 각 문항에 한 번만 답하고, 채점 단계에서 고정한 7B 판정기가 정상 종료한 V8 오답 중 정해진 조건에 해당하는 답만 확인한다. 원본 8개, v18b 18개가 추가로 인정됐다. 동일 문항의 짝비교에서 V8은 v18b만 정답 75개·원본만 정답 60개(양측 정확 McNemar p=0.228), 보완 기준은 80개·55개(p=0.038, 다중비교 보정 전)였다.

## 기존 모델로 평가하기

1. README의 환경변수를 설정하고 고정 Python 및 모델·자료를 준비한다. BF16 평가 환경은 `env/requirements-proof.lock`, 원자료 준비 환경은 `fine_tuning/notebook_rl/requirements-data.lock`을 사용한다. 저장 경로에는 일반 사용자에게 쓰기 권한이 있어야 한다.
2. 공개 Qwen과 MMMU·MMMU-Pro를 취득한다. 아래에서는 `MMDL_MODEL_PATH`에 고정 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`의 snapshot 디렉터리를 지정한다. 기준 장비는 학습·빠른 평가에 A100 80GB, 작은 동시 수의 평가에 24GB 프로필이다.
3. 이미 사용 장비에 평가용 export가 있으면 그 디렉터리를 `MMDL_V18B_EXPORT_ROOT`로 지정한다. 이 디렉터리에는 `model_ref.json`과 `model/`이 나란히 있다. 체크포인트에서 시작할 때만 아래 복원 명령을 실행한다. 입력 디렉터리에는 원래의 `checkpoint-206-rl/` 다섯 파일과 회수 시 보존한 `model_ref.json`, `export_manifest.json`이 필요하다. `MMDL_V18B_EXPORT_ROOT`에는 새 출력 디렉터리를 지정한다. 직접 재학습했을 때는 그 실행이 출력한 모델 참조와 그 실행의 export 경로를 쓴다.

별도의 두 환경은 아래와 같이 준비할 수 있다. 학습용 모델과 결과 파일은 Git 디렉터리 밖에 둔다.

```bash
uv venv --python 3.12 "$MMDL_ARTIFACT_ROOT/envs/proof"
uv venv --python 3.12 "$MMDL_ARTIFACT_ROOT/envs/data"
export MMDL_PYTHON="$MMDL_ARTIFACT_ROOT/envs/proof/bin/python"
export MMDL_TRAIN_PYTHON="$MMDL_PYTHON"
export MMDL_DATA_PYTHON="$MMDL_ARTIFACT_ROOT/envs/data/bin/python"
uv pip install --index-strategy unsafe-best-match --python "$MMDL_PYTHON" -r env/requirements-proof.lock
uv pip install --python "$MMDL_DATA_PYTHON" -r fine_tuning/notebook_rl/requirements-data.lock
```

```bash
PYTHONPATH=src "$MMDL_PYTHON" -m mmdl.data.download --suite mmmu-val --data-root "$MMDL_DATA_ROOT" --cache "$HF_HOME" --artifact-root "$MMDL_ARTIFACT_ROOT"
PYTHONPATH=src "$MMDL_PYTHON" -m mmdl.data.download --suite mmmu-pro --data-root "$MMDL_DATA_ROOT" --cache "$HF_HOME" --artifact-root "$MMDL_ARTIFACT_ROOT"
export MMDL_MODEL_PATH="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
export MMDL_V18B_EXPORT_ROOT="$MMDL_ARTIFACT_ROOT/models/v18b"
bash scripts/restore_v18b.sh --checkpoint "$CHECKPOINT_DIR" --base "$MMDL_MODEL_PATH" \
  --reference "$REFERENCE_DIR/model_ref.json" --export-manifest "$REFERENCE_DIR/export_manifest.json" \
  --output "$MMDL_V18B_EXPORT_ROOT" --receipt "$MMDL_ARTIFACT_ROOT/restore-v18b.json"
```

복원기는 원본 BF16 기저와 학습된 네 개의 시각 연결부(Merger)로 가중치 shard를 재구성한다. 언어 및 시각 LoRA는 분리한 채 불러온다. 최종 체크포인트를 CPU에서 복원했을 때 가중치 shard 5개를 포함한 파일 20개가 원래 평가 manifest의 SHA256과 모두 일치했다. 보관된 두 모델의 MMMU 응답과 당시 공식 7B 판정을 재생하면 위 일반 MMMU 표의 네 점수가 나온다. 새 기계에서 응답을 다시 생성하거나 학습을 반복하면 GPU·라이브러리·요청 처리 순서에 따라 결과가 달라질 수 있다.

## 두 시험 실행하기

새 Pro와 학습 모델 프로필은 최초 실행 시 기준 엔진과 제출 엔진을 같은 비벤치마크 입력으로 각각 검사한다. 해당 장비·가중치·프로필에 맞는 비교 영수증이 있으면 `--engine-validation PATH`로 지정할 수 있다.

아래 네 실행에는 서로 다른 run ID와 결과 디렉터리를 사용한다. 보완 판정은 고정된 CompassVerifier-7B BF16와 공식 checkset을 사용한다. 자체 GPU에서 실행하며 원본 답안은 그대로 유지한다. 점수는 완결된 900개 또는 1,730개의 응답에서 계산한다. 재개할 때는 같은 명령·ID에 `--resume`을 붙인다.

```bash
bash scripts/eval.sh --suite mmmu-val --model base --score both --hardware configs/hardware/rtx3090_24gb.yaml --run-id val-base
bash scripts/eval_finetuned.sh --suite mmmu-val --score both --hardware configs/hardware/a100_sxm_80gb.yaml --run-id val-v18b
bash scripts/eval.sh --suite mmmu-pro --model base --score both --hardware configs/hardware/a100_sxm_80gb.yaml --run-id pro-base
bash scripts/eval_finetuned.sh --suite mmmu-pro --score both --hardware configs/hardware/a100_sxm_80gb.yaml --run-id pro-v18b
```

일반 MMMU 원본은 역사적 eager·동시 2 설정이다. 일반 MMMU v18b의 역사적 A100 프로필과 Pro A100 프로필은 CUDA graph·동시 9를 사용한다. Pro는 문제에서 실제로 참조한 그림 최대 7장, 보기 최대 12개를 보존한다. 두 모델 모두 P0, B2 샘플링, 출력 32,768토큰, 문항별 seed(마스터 3407), 같은 Pro 문제 순서를 쓴다. 각 문항을 기준으로 한 전체 정확도와 과목별 정답 수·분모를 기록한다.

RTX 4090에서는 해당 명령의 장비 파일을 `configs/hardware/rtx4090_24gb.yaml`로 지정한다. 평가기는 이에 맞는 낮은 동시 요청 수를 선택한다. 고정 Pro 1,730문항을 지정 프로세서로 전수 확인한 결과 최대 입력은 7,381토큰이었다. 여기에 출력 상한을 더한 40,149토큰은 문맥 한도 40,960 안에 든다. 이 입력 검사는 CPU에서 수행했다. 4090에서 두 모델의 7장 생성과 전체 소요 시간은 실기 확인이 남아 있다.

RunPod 운영자는 게시된 제출 checkout에서 아래 두 명령으로 실행 전에 대상 모델·문항 수·채점기·단계별 명령을 확인할 수 있다. 기본 동작은 계획 출력이다. `MMDL_BOOTSTRAP_PYTHON`에는 앞서 설치한 환경의 Python을 지정한다. `MMDL_VENV_ROOT`는 운영 스크립트가 실행 단계에서 uv로 생성하고 설치 기록을 관리할 별도 경로다.

```bash
export MMDL_CODE_COMMIT="$(git rev-parse HEAD)"
export MMDL_BOOTSTRAP_PYTHON="$MMDL_PYTHON"
export MMDL_VENV_ROOT="$MMDL_VOLUME_ROOT/envs/pro-78cb73de10bb90d2"
bash scripts/reproduce.sh --suite mmmu-pro --model base --score both --hardware configs/hardware/a100_sxm_80gb.yaml --job-id pro-base-3407 --commit "$MMDL_CODE_COMMIT"
bash scripts/reproduce.sh --suite mmmu-pro --model finetuned --score both --hardware configs/hardware/a100_sxm_80gb.yaml --job-id pro-v18b-3407 --commit "$MMDL_CODE_COMMIT"
```

## 원본부터 재학습하기

별도 학습 환경의 Python을 `MMDL_TRAIN_PYTHON`, 자료 준비 환경의 Python을 `MMDL_DATA_PYTHON`으로 설정한다. 고정 목록에 기록된 원자료 1,067개 중 1,031개는 Hugging Face, 36개는 원 저자 배포처에서 취득한다. 문항 순서와 픽셀·질문·선택지 해시를 대조한다. 검증된 본학습 ID는 3,000개이며 준비 후보 512개는 별도다. 감시 200개, 효과 확인 128개, 속도 측정 16개 및 별도 추론 감시 64개는 본학습에 들어가지 않는다.

```bash
bash scripts/finetune.sh --prepare-only --config configs/train/v18b.yaml
bash scripts/finetune.sh --plan-only --config configs/train/v18b.yaml --run-id v18b-plan-inspection
bash scripts/finetune.sh --train --config configs/train/v18b.yaml
```

배포 설정은 각 실행의 감시 오경보율을 다시 측정한다. 95% 상한이 2%를 넘어 자동 통과하지 못하면 그 실행 기록에서 수치를 확인한다. 기존 규칙이 운영자 판단을 허용하는 범위(상한 5% 이하)에서만 새 실행에 `--guard-risk-acceptance "이번 측정에 대한 판단 근거"`를 줄 수 있다. 완료·중단한 기록을 보존하므로 별도의 실행에는 새 `--run-id`를 지정한다.

원본이 준비 문항을 풀고 검사를 통과한 자기 풀이와 허용된 사람 풀이로 준비 SFT를 진행한다. 본학습에서는 문항당 첫 풀이 한 번에 답과 검증 가능한 관측의 RL 신호를 주며, 채점 후 자기 오답노트의 검증된 수정과 성공한 재풀이를 SFT에 활용한다. 16문항씩 188묶음과 마무리 갱신을 처리한다. MMMU·MMMU-Pro의 모든 split과 파생물은 학습에서 제외한다. 원 실행은 49묶음에서 잘림 경보로 일시 정지한 뒤 경고 기록 방식으로 끝까지 진행했다. 새 공개 학습은 경고 기록을 처음부터 적용하고 고정된 원본에서 준비 풀이를 새로 생성한다. 새 체크포인트와 점수는 새로운 실행 결과로 기록한다. 같은 모델을 직접 확인하려면 당시 로컬 export 또는 위 방식으로 복원한 slot 206을 지정한다. 공개 저장소만 받은 환경에서는 재학습으로 새로운 체크포인트를 만들 수 있다. 새로 학습한 모델은 그 실행이 출력한 `model_ref.json`을 `--model-ref`에 넘기고 별도의 run ID를 사용한다.

기존 실행에서는 답 RL 3,000행과 관측 RL 261행, 정답 첫 풀이 SFT 1,153행, 검증된 오답노트 교정 SFT 35행, 재풀이 SFT 139행을 사용했다. 준비 SFT에는 자기 풀이 158행과 사람 풀이 77행이 들어갔다. 한 번의 비교에서 이들 개별 학습 항목의 점수 기여는 따로 측정하지 않았다.

## 보관한 응답을 다시 채점하기

보관한 원 응답·실행 목록·공식 판정 기록이 있으면 기존 결과를 새로 생성할 필요 없이 대조할 수 있다. 아래 경로 변수에는 동일한 모델·문항 집합의 완료 디렉터리를 지정한다. 결과는 별도의 새 디렉터리에 쓴다.

```bash
bash scripts/rescore.sh --source-run "$SAVED_RUN" --suite mmmu-val --parser team-final-answer-v8 \
  --score both --checker-run "$SAVED_CHECKER" --output-dir "$MMDL_ARTIFACT_ROOT/replay-val"
bash scripts/eval.sh paired --suite mmmu-pro --base-run "$BASE_RUN" --finetuned-run "$FT_RUN" \
  --base-checker "$BASE_CHECKER" --finetuned-checker "$FT_CHECKER" --output-dir "$MMDL_ARTIFACT_ROOT/pro-comparison"
```
