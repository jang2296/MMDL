# CompassVerifier로 V8 오답 일부 보완 판정하기

저장된 `predictions.jsonl`을 읽어 **V8 오답 → 정답**만 보완한다. 새 답이나 풀이를 생성하지 않는다.
과제 점수는 `team-final-answer-v8` 그대로이며, `V8+checker`는 별도 참고 결과다.
2026-11-14 기준 새 CPU 검사 55개와 기존 평가 검사 181개가 통과했다.
로컬 RTX 5060에서 **3B NF4 실제 실행**을 확인했다. 공식 7B 판정은 미실행이다.

이 폴더는 설정·질문문·점검 사례·시험을 함께 둔다. 실행에는 같은 저장소의
`mmdl.evaluation.parsers`, `mmdl.runtime.artifacts`, 고정 V8 소스도 필요하다.
기존 V8 파일 변경, `MMDL/` 동기화, commit·push는 이 사용 절차에 포함하지 않는다.

## 확인된 실행 범위

- 고정 3B 가중치 두 파일의 크기·SHA256 일치 확인 후, 완전 오프라인으로 모델을 로드했다.
- 자체 합성 사례 30개: 오수락 0/16, 수락 대상 11/14 인정. 긴 응답을 자르는 사례도 실행했다.
- 실제 저장 B2 900행을 CPU로 재채점해 V8 정답 558개와 적격 39개를 확인했다. 39개 중 15개는 판정기 입력 한도(3,072토큰)를 넘어 응답 앞부분을 잘라내고 뒤쪽 접미부만 남겼다(`response_truncated`). 길이 한도로 끝난 응답은 적격 대상에 한 건도 없다.
- 그중 첫 4개를 두 개의 새 프로세스에서 `run --limit 4`로 처리했다. `targets.jsonl`, `verdicts.jsonl`, `merged.jsonl`, `summary.json`이 바이트 단위로 같았다.
- 로컬 결과는 `mode=smoke`, `reportable=false`, 분모 4, `source_denominator=900`이다. 과제 점수·공식 보완 점수가 아니다.
- Ruff·범위 mypy(13파일) 통과. 기존 V8 네 파일과 MMDL 제출본은 유지했다.

3B BF16·7B·클라우드 판정은 실행하지 않았다. 합성 사례의 오수락 0건은 실제 벤치마크 판정 정확성의 보증이 아니다.
실행 증거는 비공개 아티팩트 루트의 `compassverifier_checker/implementation-20261114/`에 보관한다.

## 처리 대상과 명령

V8 오답이면서 정상 종료(`finish_reason == "stop"`)한 다음 두 종류만 판정한다.

- `NO_PARSE_STOP`: V8이 최종 답을 찾지 못한 응답.
- `OPEN_MISMATCH_STOP`: 주관식 답을 추출했지만 정답과 불일치한 응답.

V8 정답, 길이 한도 종료, 빈 응답, 명시적으로 다른 보기를 고른 객관식 오답은 보내지 않는다.
학습 보상·학습 행 선별·여러 응답 중 답 선택에 사용하지 않는다.

| 명령 | 하는 일 | 필요한 자원 |
|---|---|---|
| `select` | 행 해시·ID·V8 재채점 대조 후 판정 입력 저장 | CPU, 캐시된 토크나이저 |
| `judge` | 대상과 정답 별칭마다 순전파 1회, A/B/C 확률 저장 | CUDA GPU, 캐시된 모델 |
| `report` | 대상·확률·영수증 재검증, V8과 병합 | CPU, 같은 토크나이저 |
| `run` | `select → judge → report` 실행 | 위 자원 모두 |
| `checkset` | 직접 만든 점검 사례 판정, 영수증 저장 | CUDA GPU, 캐시된 모델 |

## 실행 환경과 CPU 검사

저장소 루트에서 기존 Python 3.12 환경을 사용한다. 프로젝트 고정 환경은
Torch 2.8.0, Transformers 4.57.1이며 NF4에는 `env/requirements-proof.lock`의
bitsandbytes 0.48.2가 추가로 필요하다. 전역 환경을 바꾸거나 패키지를 자동 업그레이드하지 않는다.

```bash
PYTHON="${MMDL_PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/mmdl-cache/huggingface}"
export HF_HUB_OFFLINE=1
CHECKER=("$PYTHON" -m mmdl.evaluation.CompassVerifier_checker)
OUT="${MMDL_ARTIFACT_ROOT:-$HOME/mmdl-artifacts}/compassverifier_checker"
INPUT="$PWD/analysis_exports/mmmu_20260924/team_final_v8/official_b2/predictions.jsonl"

"$PYTHON" -m unittest discover \
  -s src/mmdl/evaluation/CompassVerifier_checker/tests -t src -v
"${CHECKER[@]}" --help
```

CPU 검사는 합성 행·가짜 판정기·모형 backend로 무결성과 분기 동작을 확인한다.
실제 CompassVerifier의 정확성이나 GPU 실행 성공을 증명하지 않는다.

## 모델 캐시 준비 예시

판정 시 네트워크를 사용하지 않는다. 모든 로드는 `local_files_only=True`,
`trust_remote_code=False`이며, 캐시가 없으면 실패한다. 모델 ID와 전체 리비전을 고정한다.

| 용도 | 모델 | 리비전 | 정밀도 |
|---|---|---|---|
| 로컬 작동 시험 | `opencompass/CompassVerifier-3B` | `27d6d660bdf6c2b8a9f45906d4b76c508af65ab0` | NF4, BF16 연산 |
| 공식 보완 판정 | `opencompass/CompassVerifier-7B` | `8fee654b23ec971ad4a5c6d69bd99cf0f0188197` | 비양자화 BF16 |

아래는 **캐시가 없을 때의 별도 취득 단계 예시**다. 실행하면 3B 모델을 다운로드한다.
공식 7B 취득은 추후 사용자 명령을 받은 런포드 작업에서만 진행한다.

```bash
env -u HF_HUB_OFFLINE "$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="opencompass/CompassVerifier-3B",
    revision="27d6d660bdf6c2b8a9f45906d4b76c508af65ab0",
)
PY
```

다운로드 뒤에는 `HF_HUB_OFFLINE=1`을 유지한다. `select --tokenizer PATH`를 쓸 경우
임의 복사본이 아니라 지정 모델·리비전의 **정확한 HF 캐시 snapshot 디렉터리**여야 한다.

## 로컬 3B NF4 작동 시험

다음 명령은 위 작동 검증을 재현하는 예시다. 이미 유효한 결과가 있으면 불필요하게 반복하지 않는다. 로컬 시험은 항상 `reportable=false`다.
로컬에서는 **3B만** 다운로드·실행한다(사용자 정정 2026-11-14). 7B는 추후 사용자 명령에 따른 런포드 작업이다.
비양자화 3B를 쓰더라도 `smoke` 결과를 공식 점수로 보고하지 않는다. 설정의 기술적 모델 허용 목록은 실행 승인을 뜻하지 않는다.

```bash
SMOKE=(--mode smoke --model opencompass/CompassVerifier-3B
  --revision 27d6d660bdf6c2b8a9f45906d4b76c508af65ab0 --quantization nf4)

# 먼저 합성 사례로 실행 경로 확인
"${CHECKER[@]}" checkset "${SMOKE[@]}" \
  --run-label local-3b-checkset --output-root "$OUT"

# 저장 원본에서 처음 등장하는 적격 문항 4개만 확인
"${CHECKER[@]}" run "${SMOKE[@]}" \
  --predictions "$INPUT" --run-label local-3b-smoke --output-root "$OUT" \
  --limit 4
```

`--limit N`은 `judge`·`run`의 **smoke에서만** 허용한다. 원본 순서의 적격 ID 중 처음 N개를
고르며, 선택한 문항의 정답 별칭은 모두 판정한다. N회 순전파라는 뜻은 아니다.
`select`는 전체 대상을 저장하고 `judge`에서 범위를 줄인다.
이때 보고서의 `denominator`는 실제 시험한 문항 수이며, 원본 수는 `source_denominator`에 따로 남긴다.
900문항 원본으로 2개만 시험했다면 2개 진단 결과이지 900문항 점수가 아니다.

## 공식 7B BF16 판정 절차

공식 판정용 클라우드 실행은 **사용자의 별도 실행 명령 후에만** 진행한다.
이 절의 명령은 포드를 만들거나 비용 승인을 대신하지 않는다. 고정 commit의 clean clone,
모델 캐시·CUDA·BF16·메모리와 비용 한도를 먼저 확인한다. 계획상 24GB 이상 GPU를 사용한다.
OOM 시 양자화·길이·임계값을 자동으로 낮추지 않는다.

먼저 같은 7B BF16 설정으로 `checkset`을 실행한다. `--run-label`과 `--output-root`를 명시한다.

```bash
"${CHECKER[@]}" checkset --mode official \
  --run-label official-7b-checkset --output-root "$OUT"
```

마지막에 출력된 디렉터리의 `checkset.json` 경로를 다음 명령에 넣는다.
아래 경로의 `<식별해시>`는 실제 출력 이름으로 바꾼다.

```bash
RECEIPT="$OUT/official-7b-checkset/official-CompassVerifier-7B-none-<식별해시>/checkset.json"
"${CHECKER[@]}" run --mode official \
  --predictions "$INPUT" --run-label b2-official --output-root "$OUT" \
  --checkset-receipt "$RECEIPT"
```

`official`의 모델·리비전·양자화 기본값은 표의 **7B·고정 리비전·none**이다.
공식 `judge`·`run`에는 같은 프로필의 완전한 점검 영수증이 필요하다.
오수락(`not_accept` 사례를 `ACCEPT`)이 하나라도 있으면 중단하고 영수증은 보존한다.
`accept_recall`도 기록하지만, 오수락 0이라는 점검 통과가 실제 벤치마크 정확성을 보장하지는 않는다.
영수증은 판정 폴더의 `checkset.json`으로 복사되며 `report`가 해시와 각 사례 확률을 다시 검사한다.

## 단계를 나눠 실행하기

`select`·`judge`에는 **같은 mode·model·revision·quantization**을 지정해야 한다.
`select`에서 옵션을 생략하면 공식 7B 프로필로 저장되므로 이후 3B smoke와 결합할 수 없다.
`report`는 저장 프로필을 읽는다. 입력 파일이 여러 개라면 `select`와 `report`에 같은 순서로 모두 준다.

```bash
"${CHECKER[@]}" select "${SMOKE[@]}" \
  --predictions "$INPUT" --run-label local-3b-stages --output-root "$OUT"
# 마지막에 출력된 디렉터리를 그대로 지정
SELECTED="$OUT/local-3b-stages/smoke-CompassVerifier-3B-nf4-<식별해시>"
"${CHECKER[@]}" judge "${SMOKE[@]}" --targets "$SELECTED/targets.jsonl" --limit 2
"${CHECKER[@]}" report --predictions "$INPUT" \
  --targets "$SELECTED/targets.jsonl" --verdicts "$SELECTED/verdicts.jsonl"
```

공식 단계별 실행의 `judge`에는 `--checkset-receipt "$RECEIPT"`도 필요하고 `--limit`은 금지다.
정상 완료는 종료 코드 0, 검사 실패는 2다. 실패 원인을 해결한 뒤 새 run label을 쓴다.
기존 산출물 덮어쓰기·불완전 결과의 자동 재개는 지원하지 않는다.

## 고정 판정 규칙과 산출물

`checker_v1.yaml`은 사용자가 자유롭게 조절하는 설정 파일이 아니라 **v1 계약**이다.
내용이 동결값과 다르면 중단한다. 결과를 본 뒤 바꾸지 않으며 변경에는 새 버전이 필요하다.
버전은 `compassverifier-checker-v1`, 채점 ID는 `team-final-answer-v8+compassverifier-checker-v1`이다.

- 입력은 원본 `prompt`·정답·응답이다. 이미지 자체를 판정기에 전달하지 않는다.
- 긴 응답은 판정기 토크나이저 기준 마지막 **3072토큰 이내**의 원문 접미부만 남기고 잘림을 기록한다.
- 공식 `CV_PROMPT`를 단일 user 메시지로 넣고 토크나이저의 기본 chat template을 적용한다.
- 생성 없이 마지막 위치의 float32 로짓에 **전체 어휘 softmax**를 적용한다. A/B/C만 재정규화하지 않는다.
- 전체 어휘 1위가 `A`이고 `p_a >= 0.90`일 때만 `ACCEPT`다. 약한 A와 그 밖 토큰은 `UNSURE`, B는 `REJECT`, C는 `INVALID`다.
- 별칭 중 하나라도 `ACCEPT`면 문항을 수락한다. 최종 정오는 `v8_correct or ACCEPT`이며 `1→0`은 항상 0이다.

출력은 `<output-root>/<run-label>/<mode>-<모델명>-<양자화>-<식별해시12자>/`에 저장한다.
버전·설정·질문문·모델 프로필·입력 해시를 묶으며 기존 폴더나 단계 파일을 덮어쓰지 않는다.
`--run-label`은 영숫자로 시작하는 영숫자·`_`·`.`·`-` 조합, 최대 61자다.
출력 위치가 저장소 내부이면 `analysis_exports/`라도 거부한다. 전체 산출물을 비공개로 보관한다.

| 파일 | 내용 |
|---|---|
| `selection.json`, `targets.jsonl` | 원본 파일 SHA256, V8·설정·질문문 식별 정보, 대상과 정답별 입력 |
| `judge.json`, `verdicts.jsonl` | 모델·리비전·정밀도·GPU·패키지 버전, 확률·1위 토큰·입력 길이 |
| `checkset.json` | 점검 사례 결과·오수락 수·수락 재현율; 공식 판정에는 영수증 사본 |
| `merged.jsonl`, `summary.json` | V8/보완 정오, 전환·분류·과목별 개수, 실제 분모와 식별 정보 |
| `review.md` | `ACCEPT`·`UNSURE` 문항의 정답·확률·응답 끝부분을 담은 비공개 검토 목록 |

원본 행의 `_record_sha256`, ID 중복, V8 재채점 일치와 대상·별칭·확률 파일 해시를 검사한다.
기존 네 파일(`final_answer_parser.py`, `final_answer_parser_v8.py`, `parsers.py`,
`third_party/mmmu/eval_utils.py`)의 SHA256도 동결값과 대조한다. 불일치를 우회하지 않는다.
공식 결과는 V8와 V8+checker를 나란히 제시하고, 900개일 때만 화면에 백분율을 붙인다.
450개 결과는 450개 개수로만 보고하며 900개 점수로 환산하지 않는다.

## 질문문 출처와 라이선스 확인

질문문 원본은 [CompassVerifier의 고정 `src/prompts.py`](https://github.com/open-compass/CompassVerifier/blob/2d7cba6df0b21f9c6121786ac1e5770c68473598/src/prompts.py)의 `CV_PROMPT`다.
`CV_COT_PROMPT`는 사용하지 않는다. 파일과 문자열 해시는 서로 다른 대상을 가리킨다.

| 대상 | 고정값 |
|---|---|
| upstream commit | `2d7cba6df0b21f9c6121786ac1e5770c68473598` |
| `src/prompts.py` 파일 SHA256 | `6a4bde9a47fc670ab7a5d93b9c7e1d5aa9bf13c00fedf05cc2a87ac03811411a` |
| `CV_PROMPT` UTF-8 문자열 SHA256 | `aee9c27520639ca543f0b5077164dc32c1976be0f16c87bee7055f3b7ba45e72` |

라이선스 표기가 일치하지 않는다. 고정 [GitHub 루트](https://github.com/open-compass/CompassVerifier/tree/2d7cba6df0b21f9c6121786ac1e5770c68473598)에는 루트 `LICENSE` 파일이 없다.
고정 [7B 모델 카드](https://huggingface.co/opencompass/CompassVerifier-7B/blob/8fee654b23ec971ad4a5c6d69bd99cf0f0188197/README.md)와
[3B 모델 카드](https://huggingface.co/opencompass/CompassVerifier-3B/blob/27d6d660bdf6c2b8a9f45906d4b76c508af65ab0/README.md)는 모두 YAML 메타데이터에 `apache-2.0`을 적었지만,
본문 배지·링크는 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)을 가리킨다.
여기서는 어느 표기가 우선한다고 판정하지 않는다. 재배포 전 원저작자의 조건을 확인해야 한다.

## 해석의 한계

정답을 본 판정기이며 V8 오답만 재검토하므로 참고 점수는 한쪽 방향으로만 움직인다.
V8 정답의 오류율·추론 과정 정확성·이미지 이해 능력을 검증하지 않는다. `p_a`도 정확성 보증이 아니다.
3072토큰 접미부가 앞선 문맥을 잃거나 응답 속 지시문이 판정에 영향을 줄 수 있다.
`review.md`의 모든 `ACCEPT`·`UNSURE`를 사람이 검토해야 하며, `UNSURE`는 자동으로 오답에 남긴다.
사람 검토를 했다는 기록이나 점수 변경을 도구가 대신 만들지 않는다.
