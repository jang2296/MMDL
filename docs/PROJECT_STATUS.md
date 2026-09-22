# Assignment 1 baseline — 작업 상태

2026-09-21. 새 요청: 로컬 30문제 검증 → 팀 GitHub 고정 commit의 clean clone →
RunPod Assignment A900 + analysis B900 독립 추론 → 로컬 회수·검증 → 전용 자원 삭제.
제출 점수는 A만 사용한다. 로컬900·학습·증강·MMMU test/Pro 추론·심층 분석은 금지한다.
팀 저장소는 `https://github.com/jang2296/MMDL`, 최초 총 RunPod 비용 상한은 USD 6.80이었다.
현재 상한 철회는 아래 최신 승인 기록을 따른다.
대여 직전 GPU+storage 실단가에서 회수/복구 여유를 제외한 최대시간을 계산·기록한다.

## 최신 실행 전환 — continuous scheduling (2026-09-22)

사용자 승인에 따라 기존 reference Pod는 유지하고, 새 분석 Pod의 partial 결과를 보존한 뒤
일시 중지하여 장문 생성 원인을 점검했다. 새 분석 run은 70/900, 시스템 실패 0으로 보존되었고
그 중 61개는 EOS, 9개는 `finish_reason=length`(각 32,768 token)였다. 9개가 전체 생성
399,758 token의 294,912 token(73.77%)을 차지했다. 예시 장문은 Architecture_and_Engineering
1에서 반복 문장이 약335회, 15에서 동일 유형 문장이 약79회 관측되었으나, 이는 원인 가설의
근거이지 모델 일반 동작의 증명은 아니다. 이후 사용자 요청으로 해당 분석 Pod와 로컬
70문항 결과 디렉터리·압축 백업을 삭제했다. 수치는 삭제 전 관측이며 새 실행에 재사용하지 않는다.

새 protocol 설정은 `configs/eval/mmmu_val_continuous_vllm_v1.yaml`의
`mmmu-val-fast-vllm-32k-continuous-v1`이다. vLLM 0.11.0 `LLMEngine.add_request()`/`step()`으로
최대 2개 request를 유지하고, 하나가 끝나면 최종 row를 즉시 저장한 뒤 다음 독립 문제를 refill한다.
`async_scheduling=false`, eager, BF16, P0, seed/image/parser, 출력 상한 32,768은 유지한다.
고정 batch protocol `mmmu-val-fast-vllm-32k-v1`은 역사 기록으로 보존하며 결과를 섞지 않는다.
구현 commit `702ed44c4c45f454746ec4b371ad1b7403b67856`을 GitHub에 게시했다.
새 RTX 3090/24GB Pod가 같은 commit을 clean checkout하고 lock 설치·CUDA/BF16·고정 다운로드·
3문제 GPU SMOKE를 통과했다. 분석 900문제를 새로 시작했으며 첫 확인은 5/900, 시스템 오류 0이었다.
이전 reference Pod는 그대로 유지했다. [상세 검증](../claudedocs/continuous_vllm_20260922.md).

이번 새 배포에 한해 사용자가 비용을 더 지불할 수 있다고 명시적으로 승인하여
`MMDL_COST_POLICY=user_waived` opt-in을 사용한다. 이는 결제/자동충전 변경이나 비용 0을
의미하지 않으며, storage reserve·실제 단가·자원 식별·회수/정리 기록은 계속 요구한다.
기본 strict 비용 정책은 유지한다. 이번 두 Pod의 기존 비용 제한 watchdog은 사용자 상한
철회에 따라 해제했으며, 새 배포도 자동 중지 예산을 임의로 만들어 기록하지 않는다.

새 continuous engine의 CPU mock 검사는 완료 row를 다음 request보다 먼저 보존하고,
완료 순서가 입력 순서와 달라도 기록하며, 후속 오류가 앞선 완료 결과를 지우지 않는지 통과했다.
전체 72 tests, Ruff, mypy 22 source files, Bash syntax, staged 공개 파일 검사가 통과했다.
설치된 vLLM 0.11.0 API/실제 SamplingParams의 CPU 검사도 통과했으며 로컬 GPU 추론은 추가하지 않았다.
Accounting_1, Agriculture_1, Biology_29를 `max_new_tokens=32768`로 실행했다.
모두 EOS 종료했으며 출력은 각각 303/5/415 tokens였다. 정답 1/3은 기능 SMOKE의 관측일 뿐
최종 성능이 아니다. 평가 구간 17.1465 s, 500 ms GPU 전체 메모리 관측 최대 20.6025 GiB였다.
Agriculture 결과 저장 후 Biology가 시작됐을 때 Accounting은 미완료여서 실제 rolling refill을 확인했다.

연속 실행의 `generation_seconds`는 각 engine `step()` wall time을 당시 active request 수로
나눈 attribution 값이고, 실제 요청 지연은 `request_latency_seconds`로 별도 기록한다.
따라서 request 시간 합을 전체 elapsed time으로 해석하지 않으며, throughput은 engine invocation
wall time으로 산출한다. 32,768은 Qwen 공식 MMMU 평가 source의 `out_seq_length` 값이며,
모델 card의 VL 16,384와 다른 평가 recipe 값이다. 어떤 설정도 자동으로 낮추지 않는다.

## 2026-09-22 가속 전환 — 진행 중

### 게시 및 별도3090 실행 기록

- 가속 코드/보고서 게시 SHA: `34c5cdb5559e00d1bda92e4932914344d112aba4`.
  GitHub feature ref의 SHA 일치 확인. 최종 CPU 단위검사61개, Ruff, mypy(22 source files),
  shell syntax와 staged 공개 파일 검사를 통과했다. 로컬 추가 모델 추론은 하지 않았다.
- 새 분석용3090은 GitHub에서 위 commit을 clean clone했다. 2026-09-22 02:26 UTC 부근
  `mmmu-val-analysis` bootstrap을 시작했다. GPU3090 24576MiB, driver555.42.02,
  Python3.12.3, `/workspace` XFS 영구 mount, 설치 전 CUDA/UVM 검사 PASS.
  별도 exact vLLM lock 설치→doctor→고정 model/MMMU 다운로드→2문제 GPU smoke→900 순서다.
  이 기록 시점에는 설치 중이며 vLLM GPU 추론 성공이나900 완료로 보고하지 않는다.
- 해당 base image의 SSH 환경에는 `RUNPOD_POD_ID`가 없어, provider MCP에서 확인한
  실제 Pod ID를 실행 shell에서 명시했다. 비밀키를 환경·저장소에 복사하지 않았다.
- 비용은 이전 failed Pod와 기존/새3090을 합산한다. 두3090 GPU 실단가는 각각$0.22/h이며,
  storage를 별도 보수적으로 합산한12:00 UTC 중지 안전장치가 Pod 생성 전에 설정되었다.
  그 시각까지 보수적 합산 상한$5.20, 회수·정리 여유$1.60이며 총$6.80을 넘기는 승인이 아니다.
  billing API는 지연되므로 이미 청구된 금액만으로 잔액을 추정하지 않는다.
- 기존3090 읽기 전용 재점검(02:24:55 UTC):118/900, 시스템 실패0,
  EOS80/length38,103,823 생성tokens/6,641.079초, GPU33%, VRAM19,696/24,576MiB.
  기존 실행/코드/설정은 바꾸지 않았다. 분석900 검증·회수 전 기존 Pod를 삭제하지 않는다.

이 절은 아래 이전 실행 순서보다 우선한다. 사용자는 기존 느린3090 유지,
가속 구현→추가 로컬1문제→GitHub→별도3090 분석900→검증·회수 후 기존Pod 정리→
새3090 제출900을 승인했다. 두 Pod가 잠시 겹쳐도 총 비용은 USD6.80 이내다.

- 기존 `mmmu-val-v1` 설정과 실행 중인 Pod/코드는 보존한다.
- 새 `mmmu-val-fast-transformers-v1`: batch1, 자동 SDPA, 매토큰 CPU streamer 없음.
- 새 `mmmu-val-fast-vllm-32k-v1`: batch2, 요청별 seed, BF16, native generated-only presence penalty,
  prefix cache 없음. 모델/processor revision, 데이터900, P0, sampling, 이미지 범위, parser는 유지한다.
  사용자가 잘림 문제를 지적하고 공식 생성 상한32768을 명시 승인했다. 전체 문맥36864와 구별한다.
- vLLM 전용 exact lock과 CUDA/UVM 설치 전 probe, single-role suite/bundle을 구현했다.
  로컬 격리 환경의157개 패키지 호환 검사와 vLLM0.11.0 import/CPU processor 검사가 통과했다.
  사용자 최신 지시에 따라 로컬 추가 모델 추론은 하지 않고 새3090에서 GPU 검증한다.
- vLLM 내부 입력 token IDs와 pinned processor reference IDs를 비교한다. reference pixel tensor hash를
  실제 엔진 내부 tensor 검증으로 표현하지 않는다. 동시처리 시간은 batch walltime과 배분 시간을 구별한다.
- 추가 로컬1문제 완료: `smoke-fast-local-one`, 312 tokens/EOS, 생성174.740초/전체199.549초,
  오류0. peak VRAM allocated5,043,425,792/reserved5,093,982,208 bytes, RSS11,221,233,664 bytes.
  이 실행의2048 상한은 새32768 승인 전 조건이다. 새3090 vLLM/32768 검증을 대체하지 않는다.
  출력 길이가 달라졌으므로 이전 실행 대비 전체시간 비율을 순수 가속 배율로 주장하지 않는다.
  기존34개 로컬 결과는 재실행하지 않았다. 새3090 smoke/900은 아직 미실행이다.
- 기존 Pod 읽기 전용 점검(2026-09-22 01:48 UTC 부근):75/900, 시스템 실패0,
  EOS50/length25,69,505 생성tokens/4,449.426초. GPU40%, VRAM19,696/24,576MiB,59도.
  입력에 따라 메모리는 증가하지만 이 점검에서 OOM은 없었다. 길이 제한의 품질 제약은 존재한다.
- 확인 순서: CPU 검사 → 로컬1문제 → 게시/합산 비용 watchdog → 새3090 analysis900 →
  로컬 완전 검증/열람 → 기존 partial 회수/정리 → 새3090 assignment900 → 회수/최종 정리.

## 순서와 증거

1. 발견/공식 자료 확인: 완료. 초기 폴더는 AGENTS.md와 과제 지침·템플릿
   3개뿐이며 Git 저장소·기존 코드·local_sources는 없었다.
   루트의 실제 과제 원문을 읽고 hash 기록 후 local_sources로 이동·ignore했다.
   feat/mmmu-baseline 로컬 브랜치 생성, commit/push 없음.
2. 환경/다운로드: 완료. Python 3.12.13, RTX 5060 Laptop 8151 MiB,
   Windows driver 592.82, WSL available RAM 약 21 GiB 확인.
   torch 2.8.0+cu128/CUDA12.8, transformers4.57.1, sm_120 및 BF16 matmul PASS.
   증거: 외부 setup/doctor.json 및 setup/*install.log, env/requirements-eval.lock.
   WSL ext4 available 655 GiB, 실제 VHDX 호스트 C: 잔여 378,914,590,720 bytes.
   다운로드 전 추정 peak와 별도로 50 GiB 이상 운영 여유를 유지한다.
   기본 Hugging Face cache는 비어 있고 지정 mmdl 경로는 없었다.
3. 평가 구현/CPU 검사: 평가 CLI·P0·채점·이미지·저장·재개·열람본 구현.
   22개 CPU 검사 PASS, lint PASS, typecheck PASS. 실제 2-image processor tensor/grid 검사 PASS.
   모델 두 shard와 MMMU validation 30파일, MMMU-Pro 10-way test 2파일 해시 검증 완료.
   최초 Hugging Face CDN DNS 오류는 자체 회복, 기존 다운로드 프로세스 이어받기 확인.
4. 1문제 → 1과목 30문제 → 최소 다중이미지/주관식 smoke: 첫 1문제·30문제 완료/검증 PASS.
   실행: `bash scripts/eval.sh --run-id smoke-one --mode smoke --limit 1 --no-download`.
   실제 BF16 모델 로드 2.464초. GPU에 vision/embedding/head와 decoder16개,
   나머지 decoder20개 RAM offload. 첫 응답 2,048 tokens / length 종료,
   생성 1,237.489초, 전체 1,260.610초, peak VRAM reserved 5,460,983,808 bytes,
   allocated 5,399,902,720 bytes, process RAM RSS 11,315,146,752 bytes.
   SMOKE 1/1 완료, 시스템 실패 0. 길이 상한에 도달했으므로 30문제 잘림률 실측 전
   초안 길이를 확정하지 않는다. 점수에 따른 설정 변경은 하지 않는다.
   외부 `runs/smoke-one/backend.json`, `setup/smoke-one.log` 참조.
   완료 전 해당 프로세스 swap 0 bytes 확인. 원문 JSONL/이미지/hash/HTML 보존 검증.
   독립 사후 검사: effective generation 설정, 실제 token→raw decode, 입력 정수형/hash,
   이미지 hash, JSONL 동등성, public artifact 참조 hash 모두 PASS.
   증거: 외부 `setup/smoke-one-audit.json` (1문제만 검증한 증거; 900문제 완료 증거 아님).
   다음 실행: `bash scripts/eval.sh --run-id smoke-accounting --mode partial --subject Accounting --no-download`.
   2026-09-21 18:01 UTC 완료 확인: Accounting 30/30, EOS 15/length 15, 시스템 실패·누락 0.
   프로세스 exit0. 전체 22,222.849초, 생성 포함 평가 22,194.080초, 37,848 생성 tokens.
   peak VRAM reserved 5,733,613,568 / allocated 5,433,335,296 bytes, RAM RSS 11,303,710,720 bytes.
   입력 tensor 전체 CPU 재계산/hash·P0/chat template·PNG decode/hash·token→raw decode·seed·
   raw 재채점·JSONL·public 참조/hash·escaped HTML 구조를 30개 모두 검증했다.
   통합 전 실행 code_files 28개 원본 일치. 증거: 외부 `setup/smoke-accounting-audit.json`.
   PARTIAL 18/30은 제출용 점수가 아니다. 추가 모델 추론 없이 사후 검증했으며 재실행하지 않는다.
   50%가 2,048 token 상한에 도달했다. 이 제한을 숨기거나 길이가 충분하다고 간주하지 않는다.
   중복된 첫 샘플의 seed/messages/input hash/generated token IDs/raw response/parsed answer가
   최초 smoke와 모두 일치. 현재 형식의 샘플 자체 hash도 검증 완료.
   증거: 외부 `setup/smoke-repeat-audit.json`; 같은 GPU의 1문제 관측에 한정한다.
   데이터 실제 coverage 900 고유 ID, MCQ847/open53. 이미지 개수 1:857/2:24/3:5/4:8/5:6.
   원본 README를 직접 로드하면 미취득 dev/test metadata를 요구하는
   ExpectedMoreSplitsError가 발생하여 검증한 과목별 원본 Parquet를 name=subject로 로드.
   검증을 끄지 않았으며 30×30/ID/hash 강제 검사 통과.
   증거: 외부 setup/data-coverage.json, setup/data-check.log.
5. 공통 protocol: 로컬30+구조3 검증 후 FROZEN. 변경은 상태 표기뿐이며 생성/이미지/P0/채점은 동일하다.
   2,048 token 상한은 실측된 잘림 제한을 명시한 팀 예산이다. 충분한 길이나 공식 지정값으로 주장하지 않는다.
   GitHub feature 게시 완료: `13b44e4c0f768d84f8c4d153b323371ff7168217`.
   원격 refs/heads/feat/mmmu-baseline의 SHA 일치 확인. 실제 RunPod 실행 SHA는 이후 수정 시 다시 고정한다.
   공개 대상 86개 파일의 실제 index bytes 비밀키/개인경로/크기 검사 PASS.
   개인 .serena 설정과 별도 방법론 문서는 제외·보존했다. 자체 코드의 whitespace 검사 PASS;
   vendored 공식 소스의 기존 공백 경고는 원본 hash 보존을 위해 수정하지 않았다.
   origin fetch=팀 HTTPS / push=동일 팀 SSH. 초기 빈 저장소의 feature 브랜치에 첫 게시를 완료했다.
   기존 SSH 인증에서 jang2296 계정 확인. 별도 GitHub 플러그인 설치는 필요하지 않다.
   공개키에 대응하는 기존 RSA 키로 비대화형 인증 성공, 비밀키 권한 0600 확인.
   Pod에는 공개키만 전달했다. 실제 SSH 실행·파일 쓰기·SCP 회수와 파일 해시 검증을 완료했다.
6. RunPod A/B: A 실행 중 / B 대기. 이전 실행 SHA `08e00ca802bb25f6915464df1f29ceb17b7eaa7e` 게시 확인.
   작업 전 Pod 0 / Network Volume 0. 5090 전용 Pod `0l1rrpy7m41lkq` 생성 성공(2026-09-21 19:12 UTC).
   Network Volume 없음. 60GB Pod volume/30GB container. 실제 GPU 단가 $0.69/h.
   독립 STOP watchdog을 먼저 가동했다. CUDA 진단 실패 후 조기 STOP/EXITED 확인, 감시 해제.
   SSH·SCP용 직접 TCP 포트 연결과 실제 5090 32607MiB/driver580.178.04/Python3.12.3,
   /workspace 영구 XFS mount 60GiB 확인. GitHub exact SHA 직접 clone·격리 lock 설치·pip check 성공.
   doctor는 CUDA 초기화 오류로 실패했다. 직접 libcuda cuInit(0)=999, /dev/nvidia-uvm open=EIO(5).
   CUDA control/GPU 장치는 열렸고 libcuda와 kernel driver 버전은 일치한다. 호스트 UVM 계층 장애이며
   정확한 커널 원인은 미확정이다. wheel/모델/평가 조건을 바꾸지 않았고 원격 추론은 0회다.
   진단/설치 로그 9개 SCP 회수·원격/로컬 SHA256 전부 일치. 외부 recovery/<job>-<pod>/에 보존.
   모델·benchmark 다운로드 전 실패였으므로 900개 결과는 없다. 진단 회수를 FULL LOCAL_VERIFIED로 쓰지 않는다.
   유일한 진단 사본이 로컬 검증된 뒤 장애 Pod를 삭제했다. 2026-09-21 19:27 UTC 조회:
   계정 Pod 0 / Network Volume 0. 실패한 준비 작업의 자원 정리만 완료; 전체 평가 완료는 아니다.
   Community 4090 생성 요청 2회와 교체용 EU-RO-1 5090 요청 1회가 재고 부족 HTTP400으로 거절됐다.
   최초 재고 부족3회 후 같은 조건의 재시도를 중단했다. 19:31 UTC 새 US-IL-1 4090 표시가 나타나
   공급 조건 변경을 근거로 해당 위치에만 1회 추가 요청했지만 역시 HTTP400이었다(재고 거절 누적4회).
   표시 재고와 실제 배정의 차단이 지속된다. 19:33 UTC Pod 0 / Network Volume 0 재확인.
   당시 목표를 BLOCKED로 전환했다. 2026-09-22 사용자가 GitHub 직접 취득 실행을 재지시하고
   4090 또는 3090 사용을 명시적으로 허용했다. 팀명/팀원은 미제공을 유지한다.
   재고 API의 LOW 표시와 실제 배정 가능 여부는 다르다. 동일 조건의 무근거 재시도는 하지 않는다.
   목표가 허용한 동급 이상 단일 GPU 대안으로 5090 32GB의 명시적 hardware/doctor guard를 추가했다.
   CPU 단위검사 41개·Ruff·mypy·shell syntax PASS. 평가 조건 변경이나 추가 로컬 추론은 없다.
   원래 운영 상한7.5시간 전에 중단했다. 청구 조회에서 이전 Pod GPU+디스크 합계 $0.1447284467 확인.
   재개 전 Pod/Network Volume 모두0 확인. 3090은 표시 재고 MEDIUM, Community $0.22/h;
   3090 명시적 profile/실제 장비 검사를 추가하되 공통 평가 조건과 dependency lock은 바꾸지 않는다.
   3090 추가 후 CPU42/Ruff/mypy/shell syntax 및 FROZEN/GPU-only profile 검사 PASS. 추가 로컬 추론 없음.
   새 게시 SHA `af961a36b2471e978e96efd7347e150cb0bbce00` 원격 일치 확인.
   2026-09-22 00:14 UTC Community 3090 Pod `2j4knzy36jtzmv` 배정, 실제 GPU $0.22/h.
   container30GB/volume60GB, 별도 Network Volume 없음. 외부 STOP 2026-09-22 22:14:12 UTC 가동.
   새 실행22시간+보수적 디스크 비용+이전 청구 합계 약$5.271, 회수 여유 약$1.529.
   배정 호스트 CUDA12.7이 cu128 컨테이너의 cuda>=12.8 조건을 만족하지 못해 부팅이 거부됐다.
   Pod STOP/EXITED 확인 후 공식 base 이미지 digest로 교체하여 00:26 UTC 재시작했다.
   `runpod/base@sha256:a5aead56b5ed7754235250afface107a8a19646ac34f62c87e5f41eb147fa7b2`.
   SSH에서 실제3090 24,576MiB/driver565.57.01/Python3.12.3/영구 XFS volume60GiB 확인.
   libcuda cuInit(0)=0 및 UVM 장치 열기 PASS. GitHub af961a3 직접 clone, exact lock 설치131.440초,
   pip check/FROZEN config/doctor BF16 PASS, 고정 모델·MMMU 다운로드45.835초 완료.
   실행 job=mmdl-val-20260922-3090. 사전 smoke1/1 완료/시스템 실패0, 실제 모델 GPU-only device map.
   smoke 생성50.679초/전체80.206초, peak VRAM allocated9,162,626,560/reserved9,275,703,296 bytes.
   00:33 UTC 제출용 A900 실제 평가 시작. 완료 후 별도 B900→검증/포장 순차 실행이다.
   외부 STOP deadline은 그대로 유지한다. 사전1문제는 전체900 결과나 시간 추정의 충분한 표본이 아니다.
   검사를 우회하거나 평가용 Torch2.8.0+cu128 lock/프로토콜을 바꾸지는 않는다.
   총 상한$6.80은 유지하며 재개 시 이미 사용한 비용·보존비·회수 여유를 먼저 차감한다.
7. 무료 준비: Accounting30 종료·사후 검증 후 외부 setup/runpod-prep.DK1IeD의
   검증된 19개 파일을 원본에 통합했다. A/B role·Git SHA·개별 추론 호출 기록·공유 이미지,
   과목별 시간·안전한 bundle/로컬 receipt와 clean-clone 재현 명령이 포함된다.
   CPU40/lint/typecheck PASS. bundle·LOCAL_VERIFIED receipt·정확한 삭제 대상 gate,
   staged-byte secret 검사까지 검증했다. clean-clone 재현 스크립트·README를 작성했고
   중단된 stage의 로그 충돌 방지·실제 단일4090/VRAM 확인·팀 origin 제한을 구현/CPU 검증했다.
   명령: 준비본에서 `PYTHONPATH=src MMDL_PYTHON=<검증 venv>/bin/python bash scripts/test_reference.sh`.
   README는 clone shell과 무변경 dry-run을 구별하고 기존 checkout의 재개 명령을 명시한다.
   검증 venv를 bootstrap Python으로 지정한 `reproduce.sh` shell 진입점 dry-run도 exit0 확인.
   기존 SMOKE 열람본을 Windows Edge headless로 실제 렌더링했다(외부 setup/smoke-one-viewer.png).
   토큰 ID가 질문/응답보다 먼저 표시되는 사용성 문제를 준비본 render에서 보완했다.
   합성 2건의 질문·A/B 보기·공유 PNG·응답·null 설명·접힌 metadata를 Edge에서 확인했다.
   증거: 외부 setup/viewer-preflight.{png,json}, setup/runpod-prep-checks.log.
   원본 샘플/HTML은 수정하지 않았고 실제 A/B viewer 검증은 아직 아니다.
   통합 원본에서도 CPU40/Ruff/mypy/bash 검사 PASS. 환경·backend·parser·P0·generation 값은 그대로다.
   `smoke-structural` PARTIAL 3/3 완료, exit0/시스템 실패0. 사전 선택한 각 ID를 한 번씩 실행했다.
   Architecture_and_Engineering_14: open/1 image, 2,048 tokens/length/1,183.360초.
   Biology_29: 5 images, 457 tokens/EOS/269.916초.
   Music_21: 2 images/처리 pixel 합 2,591,744, 346 tokens/EOS/273.765초.
   전체 1,752.717초. 입력 tensor CPU 재구성/hash, P0/chat, 실제 token decode, PNG 8개,
   이미지 순서, raw 재채점, 과목별 산술/시간, JSONL/public hashes, inline HTML, 3개 독립 호출 기록 PASS.
   통합된 source 31개 hash 일치. 증거: 외부 `setup/smoke-structural-audit.json`, `setup/smoke-structural.log`.
   peak RAM RSS 11,503,804,416 bytes. CUDA allocator allocated 9,001,974,784 / reserved 9,347,006,464 bytes.
   이 allocator peak는 WSL이 보고한 물리 VRAM 8,546,484,224 bytes보다 크다. OOM은 없었으나
   실제 물리 residency/host paging은 계측하지 않았으며, allocator 수치를 물리 VRAM 상한으로 해석하지 않는다.

## 결정

- sc-load/메모리 조회에 기존 작업 기록 없음. 현재 파일을 기준으로 시작.
- 자료 조사와 장비 확인은 병렬, 설치→probe→GPU 실행은 순차 수행.
- 코드와 작은 결과는 저장소, 환경·모델·벤치마크·대형 결과는 WSL Linux 저장.
- P0 한 종류만 사용. sampling 및 benchmark 조건은 점수에 따라 변경하지 않는다.
- Ponytail: 기존 표준 라이브러리와 Transformers/Datasets를 사용하며 웹앱은 만들지 않는다.
- 진행 상태는 이 문서 하나에 기록. 정답/benchmark 내용을 대화로 출력하지 않는다.
- 새 요청이 이전 클라우드/게시 금지를 대체한다. 단일 RunPod 용도·총 USD 6.80 예산과
  팀 URL·SSH 인증은 확인했다. 실제 단가/최대시간·SSH 회수·watchdog/삭제 gate 전에는 배포하지 않는다.
  RunPod에는 로컬 코드/환경/모델/cache를 업로드하지 않고 GitHub 고정 commit만 취득한다.
- RunPod 공식 manage-pods/storage/types/pricing 문서를 확인했다. Stop은 볼륨을
  보존하지만 storage 과금이 남으며, 검증 완료 후 정확한 작업 Pod를 Delete해야 한다.
  Network Volume은 기본 생성하지 않는다. MCP/v2에 예약 stop 옵션은 확인되지 않았다.
  SSH와 독립된 외부 MCP watchdog의 시간 한도/실제 stop 증거를 배포 전에 준비한다.
  외부 실행 셀의 5초 타이머 모의 검사는 완료했으나 실제 Pod stop 검증은 아니다.
  4090 조회 가격은 community $0.34/h·secure $0.74/h, stock LOW였으며 확정 견적은 아니다.
  공식 `runpod-torch-v280` template과 Docker registry metadata 확인 완료:
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Linux amd64 digest
  `sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`.
  해당 image build history의 Python3.12/venv 설치를 확인했으며 실제 Pod 실행은 미검증이다.
  사용자가 이번 작업 예산의 사용 가능 여부를 확인했다.
  설치·평가·회수·storage·복구 여유를 합친 총 상한은 USD 6.80이다.
  잔액 소진 전에 GPU를 중지할 여유를 확보하며 자동 충전/결제 설정은 변경하지 않는다.
  MCP 사용내역 조회에는 계정 잔액 필드가 없으므로 이를 잔액 조회 증거로 사용하지 않는다.
- 샘플 JSON과 자체 hash를 한 번의 원자적 저장으로 함께 보존한다.
  최초 smoke에서 이미 실행 중이던 이전 sidecar 형식은 읽기만 호환한다.
- CUDA peak는 해당 프로세스 시작 후 누적 최대치이며 샘플별 독립 peak가 아니다.
- 첫 smoke 이후 실행에는 benchmark 본문 없이 생성 token 수/경과시간만 로그에 표시한다.
- 추가 smoke는 정답 무관 구조 기준으로 Architecture_and_Engineering_14(open),
  Biology_29(5 images), Music_21(처리 pixel 합 최대)을 선택한다.
- 속도 점검(읽기 전용): Accelerate 1.11.0 `dispatch_model`/`AlignDevicesHook`는
  CPU decoder weight를 각 forward 전에 GPU로 보내고 이후 meta로 돌린다.
  20개 decoder weight 합 약 3.760 GiB/token은 safetensors 크기로 계산한 전송량이며
  실제 PCIe 계측값은 아니다. 표준 pinned/non-blocking inference switch는 없으므로
  현재 실행을 유지한다. 추가 device-map 튜닝·A/B 실험은 실행하지 않았다.

## 다음 행동

1. Accounting30 종료·사후 검사 완료. 원본 결과 보존, 재추론하지 않음.
2. 격리 준비본 통합·CPU 검사·주관식/다중 이미지 3개 검증 완료. 추가 로컬 추론 없음.
   build_messages가 run_dir/prompts 사본을 쓰는 보완도 포함하며 P0는 바꾸지 않는다.
3. 3090 명시적 프로필/장비 검사 CPU 검증·새 고정 commit 게시. 이전 5090은 호스트 CUDA 실패로 정리 완료.
   기존 실행 commit의 깨끗한 별도 local worktree를 보관했다. 최종 bundle 검증에는 실제 실행 SHA의 worktree를 쓴다.
4. 사용자의 재개 지시에 따라 4090/3090을 사용한다. 대여 직후 libcuda cuInit/UVM 열기부터 확인하여 설치 낭비를 막는다.
   삭제된 Pod는 --resume하지 않는다. 정상 새 Pod/남은 예산/독립 watchdog을 확인한 뒤 README의 clone 명령을
   MMDL_CODE_COMMIT에는 새 게시 전체 SHA, MMDL_JOB_ID=mmdl-val-20260922-3090,
   --hardware configs/hardware/rtx3090_24gb.yaml을 지정한다(4090 배정 시 대응 profile 사용).
   doctor·최소 smoke 완료. 현재 A900 진행 중이며 B900·검증/포장/회수/삭제가 남았다.
전체 protocol은 FROZEN이다. 상태는 RUNPOD_ASSIGNMENT_RUNNING.
A/B·전체 결과 로컬 회수·실측 보고서·최종 CLEANUP_VERIFIED는 아직 달성하지 않았다.
