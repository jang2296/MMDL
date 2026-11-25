# Team 2 · Qwen3-VL-4B-Instruct 평가와 파인튜닝

수업 지정 모델 Qwen3-VL-4B-Instruct를 MMMU validation으로 평가하고(Assignment #1), 같은 모델이 스스로 푼 풀이와 자기 교정으로 추가 학습한 최종 모델(v18b)을 MMMU validation, MMMU-Pro, MMMU test로 평가했습니다.

## 최종 결과

| 시험 | 문항 | 원본 · V8 | 원본 · V8+판정 | 최종 · V8 | 최종 · V8+판정 |
|---|---:|---:|---:|---:|---:|
| MMMU validation | 900 | 558 (62.00%) | 566 (62.89%) | **573 (63.67%)** | **591 (65.67%)** |
| MMMU-Pro Standard 10-way | 1,730 | 905 (52.31%) | 926 (53.53%) | **932 (53.87%)** | **953 (55.09%)** |
| MMMU test | 10,500 | 측정 안 함 | 측정 안 함 | **5,232 (49.83%)** | **5,380 (51.24%)** |

- **V8**: 과제 기준 점수입니다. 규칙 기반 최종답 파서(`team-final-answer-v8`)로 채점합니다.
- **V8+판정**: 같은 답안의 보조 점수입니다. V8 오답 중 정상 종료했지만 답을 읽지 못했거나 주관식 표기가 달랐던 답안만 CompassVerifier-7B로 다시 판정해, 같은 답일 확률이 0.90 이상이면 정답으로 인정했습니다. 정답을 오답으로 바꾸지는 않습니다.
- 모든 점수는 문항마다 답을 한 번 생성한 결과입니다(기준 seed 3407).
- MMMU validation과 MMMU-Pro는 출력 32,768토큰, 이미지 1,003,520~4,014,080픽셀로 원본과 최종 모델을 같은 조건에서 평가했습니다. MMMU test는 비용 때문에 출력 2,048토큰, 이미지 262,144~1,310,720픽셀로 최종 모델만 평가했으므로 위 두 시험과 직접 비교하지 않습니다.
- 원본 MMMU validation 558/900은 Assignment #1의 제출 baseline입니다.

## 학습 방법

추가 학습에는 같은 Qwen3-VL-4B-Instruct만 사용했고, 더 큰 모델의 풀이는 쓰지 않았습니다.

1. 학습 문제를 정답 없이 한 번 풀고, 정답 여부를 보상으로 강화학습합니다.
2. 맞힌 풀이 중 품질 검사를 통과한 풀이를 지도학습합니다.
3. 틀린 문제는 정답을 보고 오답노트를 쓰게 하고, 노트에서 고친 값이 원래 자료와 맞을 때만 그 부분을 지도학습합니다.
4. 정답을 가리고 노트의 교훈만 준 채 다시 풀어 맞히면, 그 풀이를 원래 질문의 답으로 지도학습합니다.
5. 원본 모델에서 너무 멀어지지 않도록 KL 항을 함께 둡니다.

- 학습 자료는 공개 데이터셋 15종에서 고른 3,000문항입니다(본학습 전 준비 학습은 별도 512문항). MMMU와 MMMU-Pro는 학습에 쓰지 않았습니다.
- 본학습은 16문항마다 가중치를 갱신했고, A100 80GB 한 대에서 약 6시간 14분 걸렸습니다.
- 평가할 때는 오답노트나 정답 없이 문항마다 한 번만 답합니다.

## 학습한 부분

언어 모델과 시각 인코더의 원래 가중치는 고정하고, 아래 부분만 학습했습니다.

| 부분 | 대상 | 방식 | 학습률 | 파라미터 |
|---|---|---|---:|---:|
| 언어 모델 | 선형층 252곳 | LoRA (rank 16, α 32) | 5×10⁻⁵ | 33,030,144 |
| 시각 인코더 | 선형층 48곳 | LoRA (rank 16, α 32) | 1×10⁻⁵ | 2,359,296 |
| 기본 Merger | `visual.merger` | 전체 가중치 | 1×10⁻⁵ | 27,271,680 |
| DeepStack Merger ×3 | `visual.deepstack_merger_list.{0,1,2}` | 전체 가중치 | 1×10⁻⁵ | 81,833,472 (각 27,277,824) |
| **합계** | | | | **144,494,592** (전체 약 44억 개의 약 3.3%) |

- **기본 Merger**: 시각 인코더 마지막 층의 특징을 이웃한 2×2 조각씩 묶어(1024×4 = 4096차원) 언어 모델의 입력 차원(2560)으로 바꾸는 연결부입니다. 구조는 LayerNorm(1024) → Linear(4096→4096) → GELU → Linear(4096→2560)입니다.
- **DeepStack Merger**: Qwen3-VL의 DeepStack 구조에 있는 연결부 3개로, 시각 인코더 5·11·17번째 층의 특징을 같은 방식으로 바꿔 언어 모델 처음 세 층의 이미지 토큰 위치에 더합니다. 구조는 LayerNorm(4096) → Linear(4096→4096) → GELU → Linear(4096→2560)입니다.
- 네 Merger의 크기는 원래 모델 구조 그대로입니다(`config.json`의 `deepstack_visual_indexes: [5, 11, 17]`). LoRA 없이 전체 가중치를 학습했기 때문에 학습 파라미터의 대부분을 차지합니다.

## 실행과 재현

```bash
git clone --branch feat/mmmu-baseline https://github.com/jang2296/MMDL.git
cd MMDL
python3.12 -m venv "$HOME/mmdl-env" && source "$HOME/mmdl-env/bin/activate"
pip install -r env/requirements-vllm.lock
pip install --no-deps -e .
export HF_HOME="$HOME/mmdl-cache/huggingface"
export MMDL_DATA_ROOT="$HOME/mmdl-data"
export MMDL_ARTIFACT_ROOT="$HOME/mmdl-artifacts"
export MMDL_JOB_ID=baseline-v8
```

원본 모델의 MMMU validation 900문항 평가(Assignment #1)는 아래 한 명령으로 모델·데이터 준비, 추론, V8 채점, 결과 저장까지 수행합니다. RTX 3090에서는 hardware 파일만 `configs/hardware/rtx3090_24gb.yaml`로 바꿉니다.

```bash
bash scripts/eval.sh --protocol configs/eval/mmmu_val_v8.yaml --hardware configs/hardware/rtx4090_24gb.yaml --model-ref manifests/models/baseline.json --model-path Qwen/Qwen3-VL-4B-Instruct --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --artifact-root "$MMDL_ARTIFACT_ROOT" --public-root "$MMDL_ARTIFACT_ROOT/public" --job-id "$MMDL_JOB_ID" --run-role evaluation --require-commit "$(git rev-parse HEAD)" --run-id "${MMDL_JOB_ID}-evaluation" --mode full
```

- **Assignment #1 보고서**(평가 조건, 저장 응답 재채점, 공식 비교값과의 차이): [reports/mmmu_baseline.md](reports/mmmu_baseline.md)
- **최종 모델(v18b) 재현**(체크포인트 복원, 원본부터 재학습, MMMU validation·MMMU-Pro·MMMU test 평가 명령): [reports/finetune_v18b.md](reports/finetune_v18b.md)
- 최종 체크포인트는 학습 자료의 이용 조건 때문에 공개 저장소에 올리지 않았습니다. 제공 범위는 v18b 보고서에 적었습니다.

| 고정 버전 | revision |
|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` (BF16, 비양자화) |
| `MMMU/MMMU` | `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| `MMMU/MMMU_Pro` (Standard 10-way) | `563f3e84bb3b90893083a1f039cfa13077f2302b` |

## 저장소 구성

```text
MMDL/
├── assignment/   과제 지정 경로의 보고서 링크
├── configs/      평가 설정(eval), GPU 자원 프로필(hardware), 학습 설정(train)
├── env/          고정 의존성 lock
├── fine_tuning/  v18b 학습 코드
├── manifests/    모델·데이터 revision과 파일 hash
├── prompts/      평가 프롬프트
├── reports/      Assignment #1 보고서와 v18b 재현 보고서
├── results/      작은 공개 점수 집계
├── scripts/      평가·학습·복원·재채점 진입점
├── src/mmdl/     데이터 취득, 평가, 채점(V8·CompassVerifier), 실행 지원
├── tests/unit/   CPU 회귀 검사
└── third_party/  출처를 고정한 평가·참고 코드와 license
```

가중치, MMMU 원문과 이미지, 개별 응답은 이 저장소에 포함하지 않습니다.
