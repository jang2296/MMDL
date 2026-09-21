# Qwen3-VL-4B MMMU validation baseline

Assignment 1의 고정 BF16 모델과 validation 900문제를 평가한다.
현재 실행 상태와 GPU 검증 증거는 [PROJECT_STATUS](docs/PROJECT_STATUS.md),
P0·generation·공식 parser 근거는 [SOURCE_REVIEW](docs/SOURCE_REVIEW.md)에 있다.
MMMU-Pro 10-way는 다운로드·해시 확인만 하며 평가하지 않는다.

## 환경

Python 3.12 Linux/WSL에서 저장소 밖에 환경을 만든다. Windows GPU driver를 유지한다.

```bash
python3.12 -m venv "$HOME/mmdl-env"
source "$HOME/mmdl-env/bin/activate"
pip install -r env/requirements-eval.lock
pip install --no-deps -e .
export HF_HOME="$HOME/mmdl-cache/huggingface"
export MMDL_DATA_ROOT="$HOME/mmdl-data"
export MMDL_ARTIFACT_ROOT="$HOME/mmdl-artifacts"
export CUDA_VISIBLE_DEVICES=0
bash scripts/doctor.sh --output "$MMDL_ARTIFACT_ROOT/doctor.json" \
  --storage-path "$HF_HOME" --storage-path "$MMDL_DATA_ROOT" --required-gib 35
bash scripts/test_reference.sh
```

`nvidia-smi`의 CUDA 표시는 driver 한계, torch의 CUDA는 wheel runtime이다.
doctor는 BF16 실제 연산, `sm_120`, WSL 가용 RAM과 VHDX의 Windows 잔여량을 검사한다.
다운로드/변환 추정 peak 공간 외에 50 GiB 여유를 요구한다.

## 자료와 평가

```bash
python -m mmdl.data.download
```

공식 model revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`의 필요한 파일,
MMMU revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`의 30개 validation parquet,
MMMU-Pro revision `563f3e84bb3b90893083a1f039cfa13077f2302b`의 Standard 10-way
test parquet 두 개만 취득한다. 데이터 파일은 표준 HF cache를 가리키는 링크로 보관한다.
30개 원본 subject 파일을 검증한 로컬 snapshot에서 각각 `name=subject`인 Parquet config로
로드한다. 원본 README의 split metadata는 dev/test도 요구하므로, validation 전용 취득에는
이 로더를 쓰고 파일 hash·과목당 30개·전체 900개 ID를 별도로 강제 검증한다.
`data_files`로 validation을 명시하여 dev/test를 로드하지 않는다. 파일 해시는 `manifests/`에,
실제 개인 저장 위치는 외부 `setup/locations.json`에 둔다.

로컬 5060은 아래 30문제 검증까지만 실행한다. CPU-offload profile의 `full`은 거부한다.
누락된 기본 자료는 자동 준비하며 학습/증강은 실행하지 않는다. 전체 평가는 사전 검증 후
`FROZEN` protocol과 GPU-only 장비에서만 실행한다.

```bash
bash scripts/eval.sh --protocol configs/eval/mmmu_val_v1.yaml \
  --hardware configs/hardware/rtx5060_8gb.yaml \
  --model-ref manifests/models/baseline.json \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --run-id smoke-accounting \
  --mode partial --subject Accounting
```

`--model-path`는 검증된 로컬 snapshot 경로로 바꿀 수 있다. `--data-root`도
manifest와 30개 validation parquet가 있는 위치로 교체한다. baseline은 공식 base만 받는다.
이후 학습 모델은 `--model-ref`와 `--model-path`를 교체한다. `kind`는
`full`/`merged`/`adapter`를 지원하며 `base: {id, revision}`, `dtype: bfloat16`,
`processor_revision`, 파일별 `path/bytes/sha256`, `train_config_sha256`,
`data_manifest_sha256`, `code_commit`, `artifact_revision`을 요구한다.
`--base-path`로 원본 위치를 지정한다. processor/P0/생성/채점은 원본과 동일하다.
실제 학습 checkpoint 검증은 해당 artifact가 생긴 뒤 수행해야 한다.

4090에서는 같은 평가 설정에 다음 profile과 별도 run ID를 쓴다. 실제 4090 검증은 미실행이다.

```bash
bash scripts/eval.sh --protocol configs/eval/mmmu_val_v1.yaml \
  --hardware configs/hardware/rtx4090_24gb.yaml \
  --model-ref manifests/models/baseline.json --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --run-id baseline-4090 --mode full
```

사전 검증은 별도 ID와 `--mode smoke --limit 1`,
`--mode partial --subject Accounting`을 사용한다. 추가 다중 이미지/주관식은
`--sample-id`를 반복 지정한다. 정답률은 설정 변경 기준이 아니다.
중단한 실행은 같은 명령에 `--resume --no-download`를 더한다.
모델·protocol·코드·환경·장비·ID가 달라지면 재개를 거부한다.
완료된 샘플은 재생성하지 않는다. OOM에 자동 fallback은 없다.

## 결과 확인과 재채점

외부 `MMDL_ARTIFACT_ROOT/runs/<run-id>/`에 샘플별 JSON, 원문 전체 `predictions.jsonl`,
이미지 PNG, 실제 messages/입력 tensor hash, `review.html`을 보존한다.
HTML을 로컬 브라우저로 열면 질문·모델 출력·정답을 확인할 수 있다.
설명 필드는 생성된 응답 원문이며 별도 풀이를 만들어 추가하지 않는다.
`results/<run-id>/`에는 작은 성적표·설정·환경·외부 artifact 상대참조/해시만 둔다.

```bash
bash scripts/rescore.sh --run-id baseline-4090 --verify
python scripts/validate_run.py --run-dir "$MMDL_ARTIFACT_ROOT/runs/baseline-4090" \
  --public-dir results/baseline-4090
python scripts/check_submission.py
```

재채점은 저장 응답만 사용해 별도 `rescoring/<parser-hash>/`에 쓴다.
파싱 실패와 빈 응답은 900개 분모에 포함되는 오답이다. 시스템 오류/누락이 남으면
`INCOMPLETE`이고 제출 가능한 최종 점수가 아니다.
공식 비교값 67.4는 수업 제시값이며 실측값이 아니다.
실제 게시 전에는 명시적으로 선택한 파일만 stage하고
`python scripts/check_submission.py --staged`로 **index의 실제 내용**을 검사한다.
가중치·cache·venv·강의 원본·전체 응답·이미지는 게시하지 않는다.

## RunPod: GitHub 고정 commit에서 두 번의 실제 추론

팀 저장소는 `https://github.com/jang2296/MMDL.git`이다. 로컬 30문제와 필요한 구조별
smoke 통과 후 공통 protocol을 동결하고 검사된 feature commit을 게시한다.
RunPod에는 로컬 코드/venv/모델/cache/결과를 복사하지 않는다. 아래 경로는 현재
코드·CPU 검사 대상이며 **실제 RunPod clean-clone/4090/A900/B900 검증은 아직 미실행**이다.

배포 전 총 예산·GPU/디스크 실단가·회수 여유를 포함한 최대시간과 독립적인 STOP
watchdog을 정한다. 승인된 전체 예산을 job manifest에 기록하며 승인 없는 초과,
자동 충전·추가 Pod·고가 GPU 전환은 금지한다. 기본은 4090 한 대와 작업 전용 Pod volume,
별도 Network Volume은 생성하지 않는다. 아래 watchdog 값은 실제 외부 감시를 가동한
제어측의 기록이어야 하며 임의 문자열로 gate를 통과시키면 안 된다. 재현 스크립트는
Pod를 생성하거나 과금을 중지하지 않으므로, 프로세스 종료를 Pod stop으로 간주하지 않는다.
승인 예산과 실제 계정 잔액은 별개다. 배포 전 잔액에도 회수/보존 여유가 있는지 확인한다.
[공식 과금 안내](https://docs.runpod.io/pods/pricing)에 따르면 잔액 소진 시 별도 network volume이
없는 Pod는 종료되어 데이터가 소실될 수 있다. 자동 충전/결제 설정은 변경하지 않는다.

공식 이미지 후보는 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`의 Linux amd64
digest `sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`이다.
[공개 registry metadata](https://hub.docker.com/v2/repositories/runpod/pytorch/tags/1.0.2-cu1281-torch280-ubuntu2404/)와
해당 digest의 build history에서 Python 3.12/venv 설치를 확인했다. 실제 Pod 검증 증거는 아니며,
대여에는 `runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`로
이미지를 고정하고, 기본 이미지의 Python 패키지 대신 별도 venv에 프로젝트 lock을 설치한다.

Git과 Python 3.12가 있는 승인 Pod에서, 실제 영구 볼륨 mount를 `MMDL_VOLUME_ROOT`로
지정하고 그 아래 **서로 분리된** `HF_HOME`, `MMDL_DATA_ROOT`, `MMDL_ARTIFACT_ROOT`,
`MMDL_VENV_ROOT`를 지정한다. 체크아웃 밖에 cache/env/결과를 둔다.
`MMDL_STORAGE_RESERVE_GIB`는 설치·다운로드·Arrow 변환·두 결과·압축 peak 외의 여유다.
서버 여유는 대여 용량에 맞춰 명시하며 로컬 WSL의 최소 50 GiB는 낮출 수 없다.
`MMDL_BUDGET_USD`, `MMDL_STOP_DEADLINE_UTC`, `MMDL_WATCHDOG_ID`와 provider가 주는
`RUNPOD_POD_ID`를 기록한다. 비밀 API 키는 저장소나 명령줄에 넣지 않는다.

`MMDL_CODE_COMMIT`은 게시된 **40자리 전체 SHA**, `MMDL_JOB_ID`는 새 job ID,
`MMDL_CHECKOUT`은 영구 볼륨 아래 새 디렉터리다. 다음 **한 shell 호출**이 GitHub 취득부터
SHA 확인·lock 설치·doctor·공식 pinned 다운로드·최소 GPU smoke·A900→B900·검증/포장까지 실행한다.

```bash
bash -c '
  set -euo pipefail
  : "${MMDL_CODE_COMMIT:?full commit required}" "${MMDL_JOB_ID:?job ID required}" "${MMDL_CHECKOUT:?new checkout required}"
  [[ "$MMDL_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  test ! -e "$MMDL_CHECKOUT"
  git clone --filter=blob:none --no-checkout https://github.com/jang2296/MMDL.git "$MMDL_CHECKOUT"
  git -C "$MMDL_CHECKOUT" fetch --depth=1 origin "$MMDL_CODE_COMMIT"
  git -C "$MMDL_CHECKOUT" checkout --detach "$MMDL_CODE_COMMIT"
  test "$(git -C "$MMDL_CHECKOUT" rev-parse HEAD)" = "$MMDL_CODE_COMMIT"
  exec bash "$MMDL_CHECKOUT/scripts/reproduce.sh" --stage evaluate --target runpod \
    --suite mmmu-val-two-runs --commit "$MMDL_CODE_COMMIT" --job-id "$MMDL_JOB_ID" --execute
'
```

`reproduce.sh`만 `--execute` 없이 호출하면 dry-run이며 파일·환경·GPU를 변경하지 않는다.
위 전체 shell 호출은 Git clone/fetch를 수행하므로 dry-run에 사용하지 않는다.
중단된 동일 job은 이미 취득한 checkout에서 아래처럼 재개한다. 완료 run은 다시 추론하지 않는다.

```bash
bash "$MMDL_CHECKOUT/scripts/reproduce.sh" --commit "$MMDL_CODE_COMMIT" \
  --job-id "$MMDL_JOB_ID" --resume --execute
```

`--model-path`/`MMDL_MODEL_PATH` 및 이미 준비된 `--data-root`도 받을 수 있으며, 새 취득 위치는
HF_HOME/MMDL_DATA_ROOT로 바꾼다. 환경·commit·입력/설정이 다르면 재개를 거부한다.

- `<job>-assignment`: 제출용 A900. 사전에 지정한 이 run만 보고서 점수로 사용한다.
- `<job>-analysis`: 같은 조건으로 실제 `generate`를 새로 900회 호출하는 B900. A 응답 복사/
  rescore/KV·대화 상태 재사용은 금지한다. 같은 seed이므로 같은 응답이 나와도 정상이다.
- `<job>-smoke`: 별도 최소 GPU 검증이며 1,800개 full inference 집계에 포함하지 않는다.

코드는 두 full run을 순차적인 별도 프로세스로 실행한다. 읽기 전용 모델/데이터 cache와
lock 환경만 재사용하며, 이미지 PNG는 외부 `assets/images/<sha256>.png`에 한 번 보존한다.
샘플별 원문/실제 모델 설명/토큰/추론 ID/호출 기록은 각 run에 별도로 보존한다.
공개용 작은 결과도 실행 중 checkout을 더럽히지 않도록 외부 `public/<run>/`에 쓴다.
설치/doctor/다운로드/추론/재채점/검사 로그와 시간이 `jobs/<job>.*`에 남는다.

## 결과 회수와 정리

원격 `bundles/<job>.tar.gz`와 `.sha256`를 Full SSH의 SCP/rsync로 로컬
`MMDL_ARTIFACT_ROOT` 아래로 **pull**한다. 모델/venv/HF 전체 cache는 묶지 않는다.
예를 들어 전송 설정을 확인한 뒤 `scp -P "$SSH_PORT" "$SSH_TARGET:$REMOTE_BUNDLE" "$LOCAL_BUNDLE"`
및 hash sidecar를 회수하고, 원격에서 기록한 SHA와 비교한다.

```bash
python -m mmdl.runtime.bundle verify --archive "$LOCAL_BUNDLE" --sha256 "$BUNDLE_SHA256" \
  --destination "$MMDL_ARTIFACT_ROOT/imports/$MMDL_JOB_ID" --repository "$MMDL_CHECKOUT"
```

검증용 로컬 checkout도 실행 commit과 code hash가 같아야 한다. 검증은 안전한 압축해제,
파일별 SHA, A/B 각900 ID·30과목 산술·저장 raw 재채점, 독립 호출 기록,
이미지 decode와 viewer 링크까지 확인한 뒤 `local_receipt.json`에 `LOCAL_VERIFIED`를 쓴다.
`imports/<job>/runs/<job>-analysis/review.html`을 브라우저로 실제 확인한다.

로컬 제어측은 생성 전 계정 자원 목록과 이 job의 정확한 Pod/전용 storage ID를 resource
ledger에 보관한다. `python -m mmdl.runtime.bundle cleanup-targets --receipt "$LOCAL_RECEIPT" \
--resources "$RESOURCE_LEDGER" --pod-id "$OWNED_POD_ID"`는 검증된 **정확한 삭제 대상만**
출력하며 그 자체로 삭제하지 않는다. 이후 연결된 관리 API로 해당 Pod를 Delete/Terminate,
별도 생성한 미공유 전용 Network Volume이 있다면 삭제하고 계정에서 소멸을 재조회한다.
Stop만으로 cleanup 완료를 선언하지 않는다. API 결과·시각·잔존 조회는 로컬에 남긴다.

회수/검증 실패 시 원격 유일 사본을 삭제하지 않는다. Pod volume의 보존을 확인하고
외부 watchdog/관리 API로 GPU를 Stop한 뒤 `RECOVERY_REQUIRED/CLEANUP_PENDING`, 남은
storage 비용과 정확한 복구 경로를 보고한다. 원인별 재시도는 최대3회다.
첫 Assignment 보고서는 회수·검증된 A만으로 `scripts/report_baseline.py`에서 생성한다.

향후 승인된 MMMU test/MMMU-Pro도 같은 GitHub clean-clone·GPU-only·회수/삭제 정책을 쓴다.
현재 suite는 validation A/B만 허용하며 final suite는 거부한다. final protocol/revision/정답
경로를 별도 확정하기 전에는 최종 평가 성공이나 지원 완료를 주장하지 않는다.
