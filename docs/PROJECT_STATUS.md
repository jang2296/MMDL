# Assignment 1 baseline — 작업 상태

2026-09-21. 새 요청: 로컬 30문제 검증 → 팀 GitHub 고정 commit의 clean clone →
RunPod Assignment A900 + analysis B900 독립 추론 → 로컬 회수·검증 → 전용 자원 삭제.
제출 점수는 A만 사용한다. 로컬900·학습·증강·MMMU test/Pro 추론·심층 분석은 금지한다.
팀 저장소는 `https://github.com/jang2296/MMDL`, 총 RunPod 비용 상한은 USD 6.80이다.
대여 직전 GPU+storage 실단가에서 회수/복구 여유를 제외한 최대시간을 계산·기록한다.

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
   Pod에는 공개키만 전달하며 실제 Pod SSH/SCP 연결은 대여 후 검증해야 한다.
6. RunPod A/B: 각각 NOT_STARTED. 실제 4090 검증·회수·cleanup도 미실행.
   계정 조회 Pod 0 / Network Volume 0. 유료 자원을 만들지 않았다.
   Community 4090 생성 요청 2회가 실제 재고 부족 HTTP400으로 거절됐다(전체 위치, 표시된 EU-RO-1).
   재고 API의 LOW 표시와 실제 배정 가능 여부는 다르다. 동일 조건의 무근거 재시도는 하지 않는다.
   목표가 허용한 동급 이상 단일 GPU 대안으로 5090 32GB의 명시적 hardware/doctor guard를 추가했다.
   CPU 단위검사 41개·Ruff·mypy·shell syntax PASS. 평가 조건 변경이나 추가 로컬 추론은 없다.
   현재 표시 단가 community $0.69/h. container30GB+volume60GB 포함 보수적 $0.703/h,
   운영 최대7.5시간/약$5.273, 회수·복구 여유 약$1.527. 총 상한$6.80은 변경하지 않는다.
   실제 생성 단가가 다르면 운영 시간을 낮추고, 생성된 정확한 ID에 외부 STOP 감시를 연결한다.
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
3. 5090 명시적 프로필/실제 장비 guard 검사 → 새 고정 commit 게시 → 단일 Pod 배정·독립 STOP 감시.
   기존 실행 commit의 깨끗한 별도 local worktree를 보관했다. 최종 bundle 검증에는 실제 실행 SHA의 worktree를 쓴다.
4. 준비가 모두 끝나면 한 Pod에서 doctor·최소 smoke·A900→B900·검증/포장/회수/삭제.
전체 protocol은 FROZEN이다. A/B·로컬 회수·보고서·CLEANUP_VERIFIED는 아직 달성하지 않았다.
