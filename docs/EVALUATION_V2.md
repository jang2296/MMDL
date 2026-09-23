# 원본 점수 재현 전 평가 계약 점검

## 범위와 상태

2026-09-23 사용자 승인: 기존 두900 결과 보존, CPU 재채점/입력 대조,
소규모 A/B/C 통제 실험, GitHub 고정 commit의 RTX3090 신규900 평가.
LLM judge·fine-tuning·최종 test는 포함하지 않는다. GPU A/B/C는18개·오류0으로 완료했다.
이후 동시1 부분900은 사용자 요청으로 Pod와 함께 폐기했다. 새 동시2 protocol의900은 별도 실행한다.
기존900 결과는47.44%(427/900)와51.00%(459/900)이며 새 채점/추론과 섞지 않는다.

## V2 원본 감사 재현

사용자 제공 `docs/P0/MMMU_audit/audit_parser.py`:
`final-scope-mcq-audit-v2.0`, SHA256
`2e86359da8c47951e084851b6add0a6c20d9116aab2d986f36a823b93e96903c`.
원본 합성 검사8개가 통과하고 두 원본 `predictions.jsonl`을 재채점했다.

| 저장 실행 | 원래 정답 | V2 정답 | 오답→정답 | 정답→오답 |
|---|---:|---:|---:|---:|
| R1:2048 |427|536(59.56%)|128|19|
| R2:32768 |459|584(64.89%)|148|23|

각900개 응답 hash, 추출 결과와 정오 판정이 제공된 감사 원장과 일치한다.
번들 summary의 source 파일 hash는 회수 원본 파일 hash와 다르므로 바이트 동일 파일이라고
주장하지 않는다. 이번 재실행 summary에는 실제 회수본의 파일 hash를 기록했다.
결과 위치는 `analysis_exports/mmmu_20260923/p0_v2_replayed/{baseline_2048,continuous_32k}/`다.
P0 번들에는 문제/정답/원응답이 있으므로 전체를 공개 commit하지 않는다.

V2의 추출은 gold-blind이며 좋아진 판정만 선택하지 않는다. 다만 length MCQ는 옛 추출,
open53은 저장된 정오 판정을 유지한다. 일반 fallback도 옛 전 응답 검색이다.
그러므로64.89는 재현 가능한 **진단 점수**이지 확정 채점/공식 judge 재현이 아니다.

## 새 채점 계약: team-final-answer-v4

팀 자체 구현 `src/mmdl/evaluation/final_answer_parser.py`와 합성 회귀 검사로 관리한다.
V2의 명시적 최종답 우선 원칙과 형식 정규화 사례를 검토하여 구현했으며 V2와 동일 코드/점수를
주장하지 않는다. 질문·ID·gold는 추출 함수에 전달하지 않고 추출 후에만 정답을 비교한다.

- 명시적 final/answer 문장, boxed 답, 답만 있는 label/정확한 option text를 지원한다.
- 우선순위는 Final Answer/Decision 구간 → 마지막 Correct/Best Answer 선언 → 마지막 Answer 선언이다.
  최종 구간 안의 뒤쪽 명시적 선언은 계산값을 보기 label로 연결할 수 있다. 일반 Conclusion 설명은
  답 선언으로 취급하지 않는다. 마지막 명시적 답이 유효하지 않으면 앞의 답으로 되돌아가지 않는다.
- 사고 과정 전체에서 괄호 문자나 숫자 substring을 찾아 답으로 삼지 않는다. 임의추측/LLM 호출 없음.
- 서로 다른 유효한 Final 구간들의 답이 충돌하거나 마지막 명시적 답이 유효하지 않으면 NO_PARSE.
  잘린 응답도 같은 규칙을 적용한다.
- open은 명시적 최종 구간만 추출하고 통화기호/천단위/decimal/scientific/유리수/간단한 LaTeX분수의
  구문 차이를 정규화한 뒤 저장소의 pinned MMMU `normalize_str`/`parse_open_response`/`eval_open`을
  사용한다. 숫자 전체를 나타내는 분수는 그 문자열과 단일 수치 표현을 전달하며, 분자/분모 각각을
  별도 답으로 만들지 않는다. 숫자가 아닌 최종 구간은 공식 open parser에 전달한다.
- 공식 open의 소수2자리 정규화와 문자열 포함 판정은 유지한다. 자체 허용오차·의미 추론·단위 변환·
  각도 modulo·일반 식 계산은 추가하지 않는다. 공식 parser도 최종 구간 내부의 의미 모호성까지
  해결하지는 못한다. 공식 source는 [MMMU eval_utils](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)다.
- EMPTY/NO_PARSE/length를 분모에서 제외하지 않는다. 원점수·V2·새 scorer를 별도 기록한다.

이는 개발 validation 응답을 검토한 뒤 정한 정책이다. 독립 test 성능처럼 주장하지 않으며,
확정 후 가중치만 바꿔 fine-tuned 모델에도 동일하게 적용해야 한다.

개발 후보 v1/v2/v3의 결과와 당시 파서 사본은 `analysis_exports/mmmu_20260923/team_final_v*/`에
보존했다. v1은 일반 추론 문장을 최종답 선언으로 오인했고, v2의 일반 설명/결론 처리도 보완했다.
v3의 자체 exact-only open 비교는 공식 유형별 evaluator를 재사용하는 수업/프로젝트 방침과 달라
v4에서 위 공식 open 비교로 교체했다. 후보 중 점수가 높은 버전을 고른 것이 아니며 후보들은 baseline이 아니다.

v4 확정 전 두900 전체 재채점 결과는 R1 **479/900(53.22%)**, R2 **556/900(61.78%)**다.
원래 판정 대비 각각 +133/−81, +151/−54로 양방향 수정했다. 분모900은 유지한다.
R2 감사V2의584보다28개 낮은 것은 모델이 다시 틀린 것이 아니다. V2-only 정답32개 중23개는
length에서 옛 추출을 유지한 경우,6개는 open,3개는 MCQ 해석 차이이고, 새 방식만 정답인4개가 있다.
R2 잔여344오답의 관측 단계는 length74, 종료 후 추출 미해결15, 명시적 MCQ 불일치229,
open 최종답 불일치26이다. 시각/지식/계산 원인의 자동 확정이 아니며 원응답을 따로 검토해야 한다.
결과는 `analysis_exports/mmmu_20260923/team_final_v4/`에 원본과 분리해 저장했다.
파서 source SHA256: `4ee95ba0d5d9ed27acc01911c27235caf7f192d9187e32e339863f140629dc08`.
채점 코드 집합 hash: `bff377373e66755b3a2e8e1ed9a40b715ba9f7fdb9c9ffe72d6358b3b57be794`.
확정 후 소규모/900의 정답을 보고 이 규칙을 바꾸지 않는다. 완료된 통제 결과는 아래에 기록한다.

## A/B/C 통제 조건과 공식 출처

모든 arm은 P0, 지정 checkpoint/processor, BF16 비양자화, MMMU HF 지정 revision,
T=0.7/top_p=0.8/top_k=20/repetition=1.0/presence=1.5,
master3407+per-sample SHA256 seed, 출력32768, 동일 scorer를 유지한다.

| arm | 경로 | min_pixels | max_pixels |
|---|---|---:|---:|
| A | 기존 HF processor 직접 |262144|1310720|
| B | qwen-vl-utils0.0.14→고정 HF processor |262144|1310720|
| C | qwen-vl-utils0.0.14→고정 HF processor |1003520|4014080|

A→B는 같은 budget에서 전처리 경로, B→C는 같은 경로에서 budget의 차이를 점검한다.
C의 수치는 [Qwen MMMU source](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의
`1280*28*28`, `5120*28*28`이다. Qwen3-VL utility에는 processor의 patch_size16을 전달하여
실제32배수 smart resize를 사용한다. 28계수를 임의로32로 바꾸어 budget을 키우지 않는다.
utility가 resize한 이미지는 HF와 vLLM 양쪽에 `do_resize=False`로 전달하여 실제 재resize를 막는다.
토큰 ID가 CPU processor와 실제 engine에서 일치해야 저장하며, engine 내부 pixel tensor를 직접
검증한 것은 아니라는 한계는 유지한다.

[공식 infer script](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh)와
[공식 README](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction)를
sampling/길이 근거로 사용한다. 공식 run script seed42와 README3407은 다르며 ID별seed는 팀설계다.
공식TSV 대신 수업의 HF900을 유지하고 공식 LLM judge를 사용하지 않으므로 공식67.4와 완전 동일평가가 아니다.

고해상도 다중 이미지와32768응답의24GiB 메모리 여유를 확인하기 위해 A/B/C 모두 동시요청1이다.
CPU900 대조에서 A는 기존 모든 token수/grid/tensorhash와 일치했다. A→B 텐서 차이는897개,
A→C는900개였다. C 최대입력5627+출력32768=38395이므로 A/B/C 모두 문맥40960을 고정했다.
입력/출력을 자동 축소하지 않는다. CPU 입력 차이만으로 점수 개선을 증명하지 않는다.
동시성이나 해상도를 달리한 결과를 같은 protocol인 것처럼 합치지 않는다.

## 실행과 판정

CPU audit: `python -m mmdl.evaluation.input_audit --model-path "$MMDL_MODEL_PATH" --run-dir "$SOURCE_RUN" --output "$NEW_AUDIT_DIR"`.
이 명령은 processor/config만 읽고 모델 가중치/GPU 추론은 실행하지 않는다.
기본 CPU 환경에는 먼저 `pip install qwen-vl-utils==0.0.14 av==16.0.1`을 추가한다.
신규 전처리 의존성은 qwen-vl-utils0.0.14/av16.0.1로 고정했다. Python3.12 av14.4의 source build는
FFmpeg 개발도구가 없어 실패했으며, wheel이 제공되는16.0.1을 검증하여 OS 설치 의존을 피했다.

```bash
bash scripts/reproduce.sh --commit "$MMDL_CODE_COMMIT" --job-id "$PREP_JOB_ID" \
  --protocol configs/eval/mmmu_val_control_c_v2.yaml \
  --hardware configs/hardware/rtx3090_24gb.yaml --prepare-only --execute
bash scripts/controlled_eval.sh "$MMDL_CODE_COMMIT" "$CONTROL_JOB_ID"
```

두 번째 명령은 미리 선택한6개×3arm만 실행하고 멈춘다. Accounting1(단일 이미지),
Biology29(5개 이미지), Computer_Science19(그래프), Chemistry13(open 계산),
Music21(기존 CPU 입력에서 처리 pixel 합 최대), Electronics21(open 오판정 사례)다.
일부 알려진 실패를 포함한 **진단 표본**이며 대표 정확도/통계 유의성을 추정하지 않는다.
소규모 점수가 가장 높은 arm을 고르는 것이 목적이 아니다. 공식 입력에 가까운 C의 입력 완전성,
OOM/시스템 오류, 잘림, 속도/메모리를 확인하고 통과하면 C로 단일 신규900을 평가한다.
오류가 있으면 보존·원인 확인 후 새 protocol을 기록하며 자동 조건 완화는 없다.

최신 사용자 승인에 따른 신규900은 아래 동시2 명령을 사용한다. 위 batch1 통제 명령은 역사/재현용이다.
다운로드·driver/UVM/BF16·exact lock·smoke·900·재채점·독립 감사·bundle 순서로 실행한다.
전체 결과 회수/hash 검증 전에는 Pod/결과 볼륨을 삭제하지 않는다. 단, 사용자가 명시적으로 폐기한
중단된 C 부분900은 예외이며 새 완료 결과의 보존 원칙을 바꾸지 않는다.

## 과거 배포 checkpoint — 2026-09-23 05:05 UTC

실행 코드는 GitHub commit `a7fb5fec8d6b2a96c29e4577b351926e1aaba4b9`로 고정했다.
RTX3090 Pod `pdszzbcroakws1`에서 exact lock·pip check·BF16·다운로드를 통과했으며,
A/B/C 통제 실험 진행 중이다. 05:05 UTC에는 A5/6을 완료했고 C full900은 아직 대기 상태였다.

기존 `controlled_eval.sh`는 종료 후 멈추며, 이번 승인 작업용 별도 continuation이 종료를 기다린다.
18개 SMOKE의6/6 coverage·오류0·고정 commit/설정·코드/레코드/이미지 hash·seed·저장 token metadata와
CPU 재채점을 검사한다. 검사를 통과하면 control archive를 보존하고 기존 reproduce 명령으로
`mmdl-input-v2-full`의 C smoke→단일900→재채점→독립 감사→bundle을 실행한다.
정확도를 기준으로 arm/설정을 선택하지 않으며, 중간 오류가 있으면900을 시작하지 않는다.
원격 작업 checkout/평가 코드를 고치거나 실행 중 Git pull하지 않았다.

운영 파일은 비공개 `analysis_exports/mmmu_20260923/new_baseline/`에 보존했다.
guard `verify_controls.py` SHA256: `3ca32a5456b16e1107e09ac94c7c7f8d09c6768ce32e5d0c22950223dfe2b4b2`.
continuation `continue_full.sh` SHA256: `e5b30dcec4591a36fd329efd3caf00767dab10321456eda15310e35fc3d24dcd`.
불완전한 A 결과로 guard가 후속 실행을 거부하는 음성 검사를 수행했고 문법/lint와 별도 schema 검토를 통과했다.
이 운영 guard는 protocol hash를 별도 재구성하지 않으며, CPU/engine token 값의 동일성은 저장 전
backend assertion에 의존한다. 저장 metadata의 길이/hash 검사와 engine 내부 pixel tensor 검증을 혼동하지 않는다.
예정했던 full bundle은 생성하지 못했다. 중단된 `/workspace/artifacts/runs/mmdl-input-v2-full-evaluation/`과
해당 Pod는 아래 사용자 지시에 따라 폐기했다. 위 대기/진행중 서술은05:05 UTC 당시 기록이다.

## 통제 완료·부분900 폐기·동시2 전환 — 2026-09-23

05:55 UTC guard가18개 coverage·오류0·hash·독립 재채점을 통과했다.

| arm | 완료 | 정답 | length | evaluation_seconds |
|---|---:|---:|---:|---:|
| A |6/6|2|1|1259.522555|
| B |6/6|2|1|1346.895201|
| C |6/6|2|1|1188.179577|

각 arm의 Music21이32768 상한에 도달했다. 2/6은 진단 표본의 관측이며900 정확도나 개선의 증거가 아니다.
C smoke 후05:57 UTC 동시1 full900을 시작했으나, 사용자가 동시2 전환과 부분 결과 폐기를 요청했다.
평가 프로세스를 중단하고06:14:18 UTC Pod `pdszzbcroakws1`를 삭제했다(204→조회404).
부분900은 다운로드하지 않았으며 Pod의30GB container/80GB workspace와 함께 복구 불가능하게 삭제됐다.
기존 완결된 R1/R2는 건드리지 않았다.

완료 통제18개만 비공개 `analysis_exports/mmmu_20260923/new_baseline/mmdl-input-v2-controls.tar.gz`에
보존했다(3,400,152 bytes). SHA256:
`993e14deac00d8a1ba681818075cc8bfabcb1ee304830d253899150341751a56`.
`controls_recovered/`에 안전 경로 검사 후 풀었고 로컬18개 record hash와6/6 summaries를 확인했다.
archive에는 중단된 full900 결과가 없다.

새 `mmmu-val-official-vllm-b2-v2`는 C와 **protocol ID와 동시 요청 수2만** 다르다.
qwen-vl-utils0.0.14·pixel1003520..4014080·출력32768·context40960·P0·sampling·v4 scorer는 그대로다.
`official`은 공식 입력/생성 recipe 참조를 뜻하며 공식 LLM judge/점수 재현을 뜻하지 않는다.
수업 §1.3·§5는 backend/batch 자유와 vLLM 활용을 허용·권장한다. 이는 수업 필수batch2가 아닌 팀 선택이다.
기존 `LLMEngine.add_request/step`에서 최대2개 독립 요청을 유지하고 하나가 끝나면 즉시 다음 것을 넣는다.
`async_scheduling=false`여도 이 연속 처리는 작동한다. batch1 부분 결과와 섞지 않고 새900을 시작한다.

CPU111 tests·Ruff·mypy24source·Bash 문법 검사 통과. C와 batch/ID 이외 전체 설정 동일성,
원래 C의 무단batch 변경 거부, 새 출력/이미지 budget 축소 거부를 검사했다.
새 GPU smoke와 throughput/peak VRAM은 배포 후 측정한다.3문제 smoke 통과가 모든 장문 조합의
OOM/preemption 부재를 보장하지 않으며, KV 부족/재계산 경고를 별도로 확인한다.2배 가속을 미리 주장하지 않는다.

추가 후보는 [vLLM0.11.0 CUDA Graph와 experimental async scheduling](https://docs.vllm.ai/en/v0.11.0/configuration/engine_args.html)이다.
현재 eager=true/async=false를 유지하고 한 번에 동시성만 바꾼다. Graph는 CPU launch overhead 절감 후보지만
추가 VRAM/호환성 검사가 필요하다. async도 CPU overhead 절감 후보이지 GPU 병목 해법의 보장은 아니다.
[vLLM 성능 문서](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html)에 따라 preemption/recompute는
throughput을 해칠 수 있다. 이미 켜진 V1 chunked prefill을 새로 도입한 가속이라고 보고하지 않는다.

```bash
bash scripts/reproduce.sh --commit "$MMDL_CODE_COMMIT" --job-id mmdl-official-b2-20260923 \
  --protocol configs/eval/mmmu_val_official_vllm_b2_v2.yaml \
  --hardware configs/hardware/rtx3090_24gb.yaml --execute
# 새 Pod의 실행 단계 로그
tail -f /workspace/launcher.log
```

### 새 동시2 GPU 검증·900 시작

추론 commit `6d6d4a1a4bed1785e8770902d65e12b54340ba75`를 새 Pod `58nm0qrpgwc4ub`에서 clean clone했다.
RTX3090 24576MiB/driver580.126.09/Python3.12.3이며 API$0.22/h에 storage는 별도다.
검증된 image `runpod/base@sha256:a5aead56b5ed7754235250afface107a8a19646ac34f62c87e5f41eb147fa7b2`와
30GB container/80GB workspace를 사용했다. SSH에는 Pod 환경변수가 자동 상속되지 않아 API로 확인한
Pod ID를 launcher에 명시했다. 설치/driver/BF16 검사를 건너뛰지 않았고 모두 통과했다.

3문제 smoke는3/3·오류0으로 완료했다. Accounting526/Agriculture5/Biology547tokens,
평가30.717559s,500ms 장치 전체 메모리 관측 최대23,707,254,784bytes(약22.08GiB)다.
Agriculture가 먼저 끝나고 Biology를 보충하여 Accounting과 겹쳐 실행했다. worker allocator peak는 미측정이다.
저장 backend는 vLLM0.11.0/max_num_seqs2/context40960/지정sampling·이미지와 일치한다.
KV cache79376tokens이며 이는 가상의40960문맥 두 개를 모두 채우는 용량은 아니다.
CPU900의 최대입력5627+출력32768 두 요청 합76790은 그보다 작지만 full900의 오류 부재를 미리 보장하지 않는다.

06:31:57 UTC 새900을 시작했다.06:33:13 UTC 확인 시3/900 완료·요청2개 진행 중·실패기록0이다.
속도 배수나 최종 점수는 아직 확정하지 않는다. 원격 저장 경로는
`/workspace/artifacts/runs/mmdl-official-b2-20260923-evaluation/`, 완료 bundle 예정 경로는
`/workspace/artifacts/bundles/mmdl-official-b2-20260923.tar.gz`다.
local 운영 증거는 `analysis_exports/mmmu_20260923/new_baseline/b2_provision.json`과 `b2_smoke_evidence/`에 있다.

```bash
# RunPod 웹 터미널: 각 문항 완료/오류 확인
tail -f /workspace/artifacts/jobs/mmdl-official-b2-20260923.008-evaluation.log
```
