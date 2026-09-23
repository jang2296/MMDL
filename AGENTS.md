# AGENTS.md — MMDL: Qwen3-VL-4B 평가·개선·재현

> **2026-09-23 최신 사용자 승인 — 아래 과거 변경 금지/추가 추론 제한보다 우선:**
> `docs/P0/MMMU_audit` 원본 V2 확보·검토 → 두 저장 실행 CPU 재채점 → 실패 단계 분류 →
> CPU 입력 대조와 A/B/C 소규모 통제 실험 → 조건 고정·GitHub 게시 → RTX3090 신규900 평가를 진행한다.
> P0, 지정 모델/processor/revision/BF16/데이터와 공식 sampling recipe, 출력32768은 유지한다.
> 이미지 전처리/해상도 및 LLM 없는 채점은 별도 version으로 고정하며 원본 결과는 덮어쓰지 않는다.
> 외부 LLM judge·학습·MMMU test는 승인하지 않았다. 새 baseline 완료 전 기존51.00%를 대체하지 않는다.
> 감사 V2 코드는 확보되었으며 두 원본900에 독립 재적용하여536/900,584/900을 재현했다(8 tests PASS).
> V2는 length/open 판정 유지 및 legacy fallback 한계가 있는 진단용 버전이다.64.89%는 새 모델 추론 결과가 아니다.
> 새 `team-final-answer-v4`는 명시적 최종답 추출+공식 open evaluator를 고정했다. 저장 R1/R2 재채점은
> 479/900·556/900이며 개발 후보와 원본/V2 원장은 모두 보존했다. 전체111 CPU tests 통과.
> 다음 조건·근거·확정 상태는 `docs/EVALUATION_V2.md`를 확인한다. 유료 실행은 로컬검사 통과 후에만 시작한다.
> 코드 `a7fb5fec8d6b2a96c29e4577b351926e1aaba4b9`의 A/B/C는 각6/6·오류0으로 완료했다.
> 각2정답·Music21 length1이며 정확도 개선을 입증한 표본이 아니다. 완료된 통제18개만 로컬 회수·hash 확인했다.
> **06:14 UTC 최신 사용자 지시:** 동시1 C full900을 중단하고 부분 결과를 회수하지 않은 채
> Pod `pdszzbcroakws1`를 삭제했다(API204→조회404). 해당 부분 결과/Pod 디스크는 복구할 수 없다.
> 새 `mmmu-val-official-vllm-b2-v2`는 C와 protocol ID·동시 요청2만 다르다. 기존 continuous refill을 재사용한다.
> 모델·P0·이미지1003520..4014080·출력32768·context40960·sampling·scorer는 유지한다.
> 로컬111 tests/Ruff/mypy 통과 후 GitHub 게시→새3090 clean clone→3문제 smoke→새900 순서다.
> CUDA Graph/async는 이번 변경에 포함하지 않는다. 새 GPU 성공/속도는 실행 증거 전 주장하지 않는다.
> 폐기 지시는 중단된 C 부분 실행에만 적용하며, 새 완료 결과는 기존 회수·검증 후 삭제 원칙을 따른다.
> **06:33 UTC 실행 확인:** 게시 commit `6d6d4a1a4bed1785e8770902d65e12b54340ba75`를 새3090 Pod
> `58nm0qrpgwc4ub`가 clean clone했다. 설치·BF16·다운로드·3/3 smoke(오류0)를 통과했다.
> `mmdl-official-b2-20260923-evaluation`은06:31:57 UTC 시작, 확인 시3/900·진행 중2·오류0이다.
> 로그 `/workspace/artifacts/jobs/mmdl-official-b2-20260923.008-evaluation.log`. 새 전체 점수/가속률은 미확정이다.

> 갱신일: 2026-09-23. 저장소 루트 지침. 실행정책: **로컬 smoke/부분 검증 → GitHub 고정 commit → RunPod 단일 900문제 평가 → 로컬 회수·검증 → 작업 전용 클라우드 자원 삭제**.
> **현재 확정 상태:** 2,048-token 실행과 32,768-token 연속 처리 실행 모두 900/900 완료·로컬 회수·독립 검증·해당 Pod 삭제 완료. 점수는 각각 **47.44% / 51.00%**다. 최신 실험 사실·한계·후속 추적 기준은 **§0.1**을 우선한다. 아래 과거의 실행 중/초안 표기는 당시 기록이며 새 실행 지시가 아니다.

> **2026-09-22 사용자 변경 승인(아래 이전 운영 제한보다 우선):** 기존 실행은 역사 자료로 보존한다.
> 별도 가속 protocol(vLLM continuous scheduling, 토큰 streamer 제거)을 구현하고
> 로컬 smoke 검증→GitHub 고정 commit→단일 3090 평가 run으로 900문제를 검증·회수한다.
> ‘분석용’과 ‘테스트용/제출용’은 새 평가 역할로 만들지 않으며, 하나의 저장 결과를 점수와 실패 검토에 함께 사용한다.
> legacy continuous32k run은 전체900과 독립 검증을 완료했으며 현재 32k baseline 결과의 근거다. 새 이름으로 같은 900개를 다시 생성하지 않는다.
> 코드 변경은 실행 중인 Pod에 자동 적용하지 않는다. 추가 사용자 승인에 따라 먼저 끝나는 Pod는 900개 완료 후 로컬 회수·독립 검증을 거쳐 삭제하며, 불필요한 후속 900회는 차단한다.
> 최초 USD6.80 상한은 과거 승인 기록이며, 사용자가 이후 비용 제한을 철회한 현재 운영 승인과 구분한다.
> 기존 reference protocol은 보존하며 모델/revision/BF16/데이터/P0/생성/이미지/채점은 변경하지 않는다.
> 로컬 CPU-offload 1문제는 Transformers 경로 검증이고 vLLM 검증이 아니다. continuous vLLM은 기존3090의3문제 smoke로 검증했다.
> 추가 사용자 지정: 제출 보고서 정본은 `Assignment_1.md`. 수업 §4의 `reports/mmmu_baseline.md`에도
> 동일 본문을 유지하며 템플릿8절, 공식 출처/revision, 자체 설계 이유·차이·한계를 정확히 적는다.
> **최신 승인:** 단일 평가 run은 공식 생성 상한32768 (`mmmu-val-fast-vllm-32k-continuous-v1`)로 전환한다.
> 기존 실행/완료한 로컬1 smoke의2048은 역사적 결과로 보존한다. 새 출력 상한을 자동 축소하지 않는다.
> 로컬 추가 모델 추론은 하지 않는다. 가속 코드 검사 후32768 설정을 GitHub에 게시하고 새3090에서 검증한다.
> `[수업]`은 첨부 규정, `[사용자]`는 이번 실행범위, `[설계]`는 구현 정책, `[외부]`는 공식 기술 근거다. 팀 설계를 수업 규정으로 말하지 않는다.
> 문서·참고 코드의 존재는 구현·실행·평가 완료 증거가 아니다. 과거 미검증 표기도 현재 상태로 단정하지 말고 실제 코드·로그·manifest를 확인한다.

## 0. 에이전트 작업 규칙과 현재 범위

- 적용 지침 전체, Git 루트/상태, 기존 코드·README·설정·테스트·진행 기록, `local_sources/assignment_guidance.md`·`SUBMISSION_TEMPLATE.md`를 확인한다. 기존 작업을 reset하거나 덮어쓰지 않는다.
- `[사용자]` **로컬은 smoke/필요한 부분 검증, RunPod는 단일 `mmmu-val` 평가 run의 실제 추론 900회**다. 별도 분석용/테스트용 full run을 새로 만들지 않는다. 완료·독립검증된 legacy continuous32k 저장 결과를 baseline 평가와 실패 검토에 함께 사용한다.
- 확인된 팀 GitHub `MMDL`로 검사된 파일의 commit/push, 승인 예산 내 RunPod 실행·회수, 로컬 검증 후 작업 전용 자원 삭제를 허용한다. 정상 단계별 재승인은 없다. 계정/권한·저장소·가격/예산 상한/최대시간·회수경로 미확정은 유료 배포 전에 확인한다. 무제한 지출 승인이 아니다.
- 이번에는 평가만 완성한다. MMMU-Pro는 취득/hash 확인만 한다. **향후 승인된 MMMU test/MMMU-Pro full도 RunPod**에서 하되 현재 evaluation에는 포함하지 않는다. 모든 benchmark split/파생본은 학습·증강 금지다.
- 기존 파일을 우선한다. 중복 구현·빈 학습/증강기·가짜PASS·임시v2/final/new·불필요한 프레임워크/대시보드/문서를 만들지 않는다. 전역/다른 프로젝트 환경, OS/WSL/driver, Codex 모델·추론 강도·지침 한도를 임의 변경하지 않는다.
- 상태·결정·검증 명령·증거·다음 행동은 `docs/PROJECT_STATUS.md`에 갱신한다. 변경 없는 전체 재독, 대형 로그/900응답 대화 출력, 자동 에이전트 토론/무한 리팩터링을 금지한다. 같은 차단 원인의 수정·검증은 최대3회다. 장시간 작업은 실행장비 프로세스/로그와 완료 대기·드문 확인으로 운영한다.
- 단일 평가 검증·로컬 회수·전용 자원 삭제 후 새 유료 실행은 자동 시작하지 않는다. 2026-09-23 사용자 요청으로 1차 결과·설정·문제·개선 추적을 이 파일에 기록한다. 이 문서 갱신은 파서 수정·새 채점 결과 확정·모델 재추론·학습·유료 judge 호출의 승인이 아니다.

## 0.1. 1차 평가 실측·문제·개선 추적 원장 — 2026-09-23

`[실측]`은 저장 결과/receipt로 확인한 사실, `[진단]`은 저장 응답의 읽기 전용 분석, `[미검증]`은 원인 가설/후속 실험이다. **측정값, 수정 시의 산술 예상, 향후 계획을 섞지 않는다.** 이 절은 발표와 이후 비교의 출발점이며 900개 원문을 복제하는 장소가 아니다. 원본 증거 위치는 F항에 고정한다. 다른 문서의 과거 진행 중 표기보다 이 절의 완료 상태가 최신이다.

### A. 두 완료 실행과 실제 설정

| 항목 | R1: 2,048-token reference | R2: 32,768-token continuous baseline |
|---|---|---|
| 원본 run ID | `mmdl-val-20260922-3090-assignment` | `mmdl-cont32k-analysis-20260922-analysis` |
| 실제 추론 commit | `af961a36b2471e978e96efd7347e150cb0bbce00` | `702ed44c4c45f454746ec4b371ad1b7403b67856` |
| protocol | `mmmu-val-v1` | `mmmu-val-fast-vllm-32k-continuous-v1` |
| 설정 파일 | `configs/eval/mmmu_val_v1.yaml` | `configs/eval/mmmu_val_continuous_vllm_v1.yaml` |
| 실행 | Transformers, batch 1, SDPA MATH, deterministic=true | vLLM 0.11.0, 연속 처리, 최대 동시 요청 2, deterministic=false |
| 생성 상한 / 전체 문맥 | `max_new_tokens=2048` | `max_new_tokens=32768` / `max_model_len=36864` |
| 완료 / 시스템 오류·누락 | 900/900 / 0 | 900/900 / 0 |
| 정답 / macro accuracy | **427/900 = 47.44%** | **459/900 = 51.00%** |
| 수업 비교값 67.4%와 차이 | −19.96 percentage points | −16.40 percentage points |
| 상한 종료 `length` | **226/900 = 25.11%** | **75/900 = 8.33%** |
| 자연 종료 | `eos` 674개 | `stop` 825개 |
| 파싱 실패 `NO_PARSE` | **96개** | **69개**, 모두 객관식·오답 처리 |
| 생성 token 합 | 776,359 | 3,719,200 |
| 평가 wall time, 모델 로드 제외 | 49,741.672초 ≈ 13시간 49분 | 57,885.846초 ≈ 16시간 05분 |
| GPU 메모리 관측 | allocator allocated 13,037,445,120 / reserved 23,595,057,152 bytes | 장치 전체 사용량 표본 최대 22,885,171,200 bytes, 500ms 간격; worker allocator peak는 미측정 |

- **공통 고정 조건:** §2의 지정 Qwen3-VL-4B-Instruct와 모델/processor/tokenizer revision, MMMU revision, BF16·비양자화·미학습 원본 가중치. validation **30과목×30개=900**, 실제 유형은 **객관식 847 / 주관식 53**이다. 전부 객관식이라고 설명하지 않는다. 두 실행의 ID·질문·보기·정답·seed·이미지 순서/grid 및 입력 token 합 **534,018**이 일치함을 확인했다.
- **프롬프트:** 둘 다 §5의 **P0**. 원본 이미지들을 참조 순서대로 먼저 전달하고 질문/보기를 마지막 user text로 전달한다. MCQ는 `Question: ...\nOptions:\nA. ...\n...\nPlease select the correct answer from the options above.`, open은 `Question: ...`; 원본 hint가 있을 때만 앞에 붙인다. 추가 system/few-shot/CoT 지시 없음. 별도 분석용 프롬프트가 아니다.
- **생성 공통값:** `do_sample=true, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.5, num_beams=1`. master seed **3407**, 샘플 seed는 `SHA256("3407:{sample_id}")`의 첫 8 bytes를 big-endian 정수로 바꿔 `mod (2**31)` 하는 **팀 정책**이다. Qwen의 난수 소비 순서와 동일하다고 주장하지 않는다.
- **이미지 공통값:** `min_pixels=262144, max_pixels=1310720`, official processor 소유 resize. 수업 고정 해상도가 아니라 팀이 택한 pixel budget이며 R2에서도 유지했다.
- **파싱 공통값:** `mmmu-official-no-random-v1`. pinned MMMU 규칙 기반 MCQ 파서에서 무작위 추측을 제거했고, 실패는 `NO_PARSE`·오답으로 처리한다. open은 공식 문자열/수치 정규화 evaluator를 쓴다. **Qwen 평가 코드의 LLM 보조 판정은 사용하지 않았다.** 원본의 임의 추측 대신 실패를 명시하려던 설계이나, 아래 형식 인식 누락이 확인됐다.
- **R2 실행 세부:** 독립 문제 하나가 끝나면 저장 후 다음 문제를 넣는다. `max_num_seqs=2`, `async_scheduling=false`, eager, prefix cache=false, GPU memory utilization=0.90, tensor parallel=1. presence penalty는 vLLM native generated-only 적용; R1은 custom generated-only 구현이다. EOS/stop/기본값을 포함한 전체 실효 설정은 각 `backend.json`·샘플 기록을 근거로 한다.
- **실제 장비/환경:** 두 full run 모두 **RTX 3090 24GB GPU-only**, CPU/disk weight offload 없음. Python 3.12.3, Torch 2.8.0+cu128, Transformers 4.57.1. R1 driver 565.57.01/Linux 6.8.0-50, R2 driver 580.65.06/Linux 6.8.0-57. **4090 실측 완료가 아니다.** R2 RAM peak는 부모 프로세스만 포함하므로 worker 포함 총 RAM으로 쓰지 않는다. 표의 GPU 메모리 두 지표도 직접 비교하지 않는다.
- **paired 결과:** 둘 다 정답 347, 둘 다 오답 361, R1 오답→R2 정답 112, 정답→오답 80. 순증 **32개 / +3.56pp**다. 길이·백엔드·코드·환경이 함께 바뀌었으므로 **순수 토큰 효과나 fine-tuning 효과가 아니다.** R2 생성량은 약 4.79배이고 전체 평가 시간은 더 길다. 이 비교만으로 가속 배율을 주장하지 않는다.

### B. 출력 상한·장문 반복에서 확인한 한계

- 2,048은 초기 자원/시간 예산을 위한 팀 선택이며 공식 권장값으로 쓰지 않는다. R1 226개가 상한에 도달해 답변이 완결되지 않을 위험이 실제로 컸다. 이후 사용자 승인으로 Qwen MMMU 평가 recipe의 출력 상한 **32,768**을 채택했다. 이는 **최대 생성량**이지 항상 그만큼 생성하라는 뜻이 아니다. 지정 model card의 일반 VL 안내 **16,384**와 평가 recipe **32,768**은 출처가 다름을 남긴다.
- R2도 75개가 **정확히 32,768 token**에서 잘렸다. 그중 정답 24 / 오답 51, 파싱 실패 13이다. 자연 종료 825개는 정답 435 / 오답 390이다. 잘린 응답이 항상 오답인 것도, 상한을 올리면 반드시 정답이 되는 것도 아니다.
- 저장 응답에서 같은 문장/계산을 반복하는 장문 사례를 관측했다. **반복 현상은 관측, 발생 원인과 penalty/이미지/샘플링의 기여도는 미검증**이다. 출력 길이 부족만으로 모든 오답과 긴 실행 시간을 설명하거나 더 큰 상한을 자동 적용하지 않는다.
- R2 최대 입력은 2,655 token, **동일 샘플의 입력+출력 실측 최대는 35,390**으로 문맥 한도 36,864 안이다. 입력 최댓값과 출력 상한을 단순 합친 35,423은 보수적 합계이지 실제 단일 샘플 최댓값이 아니다. 따라서 이번 75개의 상한 종료를 작은 전체 문맥 때문에 출력 32,768을 확보하지 못한 사례라고 설명하지 않는다.
- `length`는 종료 사유, `NO_PARSE`는 추출 상태다. 두 집합은 겹치므로 단순 합산하지 않는다. R2 오답 중 `(length 또는 NO_PARSE)`는 **51+69−13=107개**이며, 이는 수정하면 107개가 모두 정답이 된다는 뜻이 아니다.

### C. 확정된 파싱 문제와 점수 해석

- 원인 위치: `src/mmdl/evaluation/parsers.py::parse_mcq_response`. `(A)` → 공백으로 둘러싸인 `A` → **응답이 5단어를 초과할 때만** 보기 문구 검색 순이다. 끝의 punctuation 제거는 내부 `A. 내용`의 점을 없애지 못한다. 짧은 `LETTER. option text` 응답을 놓칠 수 있고, 여러 후보가 나오면 마지막 출현을 택한다.
- `[진단]` R2 `NO_PARSE` 69개 중 **32개**는 응답 전체가 유효한 `LETTER. 해당 보기 원문`과 정확히 일치했다. 그중 **24개는 정답 label과도 일치했지만 오답 처리**됐다. 32개 모두 `stop` 종료라서 이 24개의 손실은 토큰 잘림 때문이 아니다. 재현 기준: `re.fullmatch(r'([A-Z])\.\s*(.+)', raw.strip(), re.S)` 후 label 범위와 본문/해당 보기의 `strip()` 일치 검사; 정답은 추출 후 비교에만 사용한다.
- 같은 기준으로 R1 `NO_PARSE` 96개 중 **31개**가 형식 일치, 그중 **23개**가 정답 label 일치였다. R1에도 같은 추출 문제가 있었으므로 후속 SC1은 R2만이 아니라 두 실행을 같은 scoring version으로 대조한다. 원래 R1 427/900은 변경하지 않는다.
- 진단 사례 ID: `validation_Agriculture_1`, `validation_Art_10`. 원문/보기는 F항의 비공개 원본에서 확인한다. 이 두 사례만으로 만든 예외 처리나 정답 기반 파싱은 금지한다.
- **24개만 바로잡고 나머지가 같다는 산술 가정**이면 `(459+24)/900 = 53.67%`, +2.67pp다. **새 파서로 전체 재채점한 실측 성적이 아니다. 현재 확정 원본 점수는 51.00%다.** 수정 규칙이 이미 PARSED인 응답에 미치는 영향도 검증해야 한다.
- 69개 NO_PARSE가 전부 정답으로 바뀐다고 가정해도 `(459+69)/900 = 58.67%`다. 이는 **NO_PARSE 집합만의 산술 상한**이지 모든 파서 오류의 상한이나 개선 예측이 아니다. 이미 PARSED인 오답의 잘못된 답 추출 여부는 별도 검토 대상이다.
- R2 유형별 결과: 객관식 **440/847**, 주관식 **19/53**. 모델 지식/시각 이해의 오답과 출력 형식/채점 손실을 분리해야 한다. 공식 점수와 격차 전부를 모델 능력 부족이나 parser 하나에 귀속하지 않는다.

### D. 공식 Qwen 평가와 같은 것·다른 것·아직 모르는 것

| 비교 항목 | 이번 실행과 공식 공개 코드의 관계 | 해석 / 다음 확인 |
|---|---|---|
| 모델·P0·sampling | 지정 Instruct/BF16, 기본 P0, T/p/k/penalty 및 R2 출력 32,768은 채택 근거가 있다 | Thinking 모델/CoT 프롬프트로 실행한 것이 아님 |
| 이미지 budget | 우리 262,144–1,310,720 pixels / Qwen `infer_instruct.sh` 1,003,520–4,014,080 pixels | 더 작은 시각 입력이 세부 문자/도형 인식에 영향을 줄 가능성; 실제 점수 영향은 미측정 |
| 이미지 처리 경로 | Qwen `qwen_vl_utils.process_vision_info` 경로와 우리 pinned processor/vLLM 경로가 다름 | 우리 reference 입력 token/grid 검증과 **엔진 내부 pixel tensor 동등성 검증은 별개**; 후자는 미완료 |
| 객관식 채점 | Qwen은 option/text 규칙 추출 후 실패 시 LLM 보조 판정, 재시도 소진 시 random fallback; 우리는 MMMU 규칙 변형, 실패 오답 | 저장 raw response로 추출 차이부터 분리. 공식 방식이면 반드시 정답이 된다고 가정하지 않음 |
| 주관식 채점 | Qwen 공개 코드는 **채점 단계에서** 정답/Other Answers 후보로 판정; 우리는 공식 MMMU open 정규화 | gold를 baseline 추론 프롬프트에 넣었다는 뜻이 아님; 두 점수의 직접 동등성 미검증 |
| seed·실행 환경 | Qwen README seed3407, 참고 실행 코드 seed42; 우리는 per-sample SHA seed / 고정 vLLM 0.11.0 | 같은 sampling 숫자만으로 응답 동일성을 보장하지 않음 |
| 데이터 취득 | 수업 지정 HF validation 900 / Qwen 공개 코드 `MMMU_DEV_VAL.tsv` | TSV와 HF의 ID·이미지·split 전체 동등성 대조 미완료. 임의로 dev 포함본으로 교체하지 않음 |

공식의 `</think>` 이후 추출 처리 차이도 확인했으나 R2 900개에는 해당 종료 태그가 **0개**여서 관측된 원인으로 세지 않는다. **67.4는 과제에 제시된 비교값**이며, 그 수치를 만든 모든 내부 실행 조건을 재현·입증한 상태가 아니다. 위 차이는 확인된 설정 차이이지 각각의 점수 손실을 정량 입증한 목록이 아니다.

출처 고정(2026-09-23 대조): Qwen commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da`의 [prompt/추론](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py), [Instruct 실행 설정](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh), [채점](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/eval_utils.py), [데이터](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/dataset_utils.py). 우리 parser의 원류는 [MMMU evaluator](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py), commit `268471d0d488258990025331c7528359c324aa25`다. 고정 model card는 W2, 수업 조건은 S1, 추가 대조 근거는 `docs/SOURCE_REVIEW.md`다.

### E. 구현·환경 문제의 경과와 재현 한계

- **로컬 사전 검사(역사 기록):** RTX 5060 8GB의 CPU weight offload에서 Accounting 30개는 18/30, `length` 15/30, 평가 22,194.080초였다. 추가 다중이미지/open 3개도 검증했다. 이는 full900 점수가 아니다. 가속 Transformers 로컬 1문제는 312 token/EOS, 생성 174.740초로 완료했으나 **vLLM 검증을 대신하지 않는다**. 상세 증거는 `docs/PROJECT_STATUS.md`의 로컬 smoke/partial 및 외부 `setup/*audit.json` 참조.
- **설치와 호스트 문제 분리:** 초기 5090의 `cuInit=999`, UVM open `EIO(5)`는 호스트 CUDA/UVM 접근 문제로 진단됐고 정확한 커널 원인은 미확정이다. 모델 추론 전 실패·로그 회수 후 삭제했다. 초기 3090은 호스트 CUDA12.7과 cu128 컨테이너의 `cuda>=12.8` 부팅 조건 불일치로 시작하지 못했다. 공식 base image digest로 교체한 뒤 같은 평가 lock으로 CUDA/UVM/BF16 검사를 통과했다. 모델/BF16을 낮춰 해결한 것이 아니다.
- 성공한 R1 base image: `runpod/base@sha256:a5aead56b5ed7754235250afface107a8a19646ac34f62c87e5f41eb147fa7b2`. 재현 shell은 clone commit·exact lock·`pip check`·실제 GPU/driver·CUDA/UVM·BF16·mount/disk·pinned model/data 검사를 수행한다. **부팅 자체가 안 되는 이미지는 shell이 고칠 수 없으므로 배포 전 호스트/이미지 조건 확인이 필요하다.** `nvidia-smi` CUDA 표시는 설치된 Torch runtime 버전과 구별한다.
- **가속 경과:** MATH attention/매 token CPU 진행 전송 경로의 비용을 검토하여 별도 fast protocol을 만들었다. 초기 vLLM 고정 batch2는 두 요청 묶음이 끝날 때까지 기다리는 구조였고, 이후 `LLMEngine.add_request()/step()`로 최대2 요청을 유지·개별 완료 즉시 refill하도록 바꿨다. 3문제 GPU smoke에서 앞 요청 미완료 중 다음 요청 투입을 관측했다. kernel 변경·로그 전송 제거·연속 처리 각각의 단독 속도 기여는 측정하지 않았다.
- 이전 고정 batch 32k **70문항 partial은 사용자 요청으로 Pod·로컬 결과/압축 사본까지 삭제**했다. 삭제 전 관측은 61 EOS / 9 length, 생성 399,758 token 중 상한9개가 294,912 token이었다. 이는 과거 운영 관측이며 현재 원문 재검증이 가능한 R1/R2 결과가 아니다. 900개와 합치거나 복원 가능하다고 하지 않는다.
- R1/R2는 각각 전체 raw 재채점·900 고유 ID/30과목·입력/코드/레코드 hash·PNG 959개·HTML 참조를 로컬에서 검증한 뒤 해당 Pod를 삭제했다. R1 Pod `2j4knzy36jtzmv`는 **2026-09-22 14:36:23 UTC**, R2 `vf8wla9q5v9gm8`는 **2026-09-23 01:03:17 UTC** 삭제 기록이 있다. 삭제 후 조회 부재를 확인했으며 추가 full900은 시작하지 않았다. 이 시점의 자원 정리 기록을 미래 계정 상태로 가정하지 않는다.

### F. 원본·공유 묶음·보고서 상태

Windows 보관 루트는 **`C:\projects\multimodel\analysis_exports\mmmu_20260923\`**이며 Git 제외 대상이다. `originals/`는 검증한 WSL `$MMDL_ARTIFACT_ROOT/recovered/`의 두 회수 폴더를 복사한 것으로 WSL 원본도 보존했다.

| 증거 | 보관 루트 아래 상대경로 |
|---|---|
| R1 전체 회수본 | `originals/baseline-2048-complete-20260922/` |
| R1 개별 결과 | 위 폴더의 `extracted/artifacts/runs/mmdl-val-20260922-3090-assignment/` |
| R2 전체 회수본 | `originals/continuous32k-complete-20260923/` |
| R2 개별 결과 | 위 폴더의 `extracted/runs/mmdl-cont32k-analysis-20260922-analysis/` |
| 먼저 웹 GPT에 올릴 자료 | `upload/mmmu_analysis_text.zip` — **2,955,323 bytes / 2.82 MiB**, 두 실행 각900 raw 응답·paired900·과목별30·설정/환경/프롬프트 |
| 필요할 때 추가할 이미지 | `upload/mmmu_images.zip` — **324,943,322 bytes / 309.89 MiB**, 두 실행이 공유한 고유 PNG **959개** |
| 분할 열람·무결성 | `upload/by_subject/` 60개 파일, `SHA256SUMS.txt`, `README.md` |

개별 run의 `summary.json`, `subject_scores.csv`, `predictions.jsonl`, `resolved_eval_config.yaml`, `resolved_hardware_config.yaml`, `backend.json`, `environment.json`, `run_manifest.json`과 회수 루트의 `recovery_receipt.json`을 근거로 사용한다. 공유용 텍스트 ZIP은 raw 응답을 자르지 않았지만 token ID 배열·tensor·전체 로그를 생략했다. 상세 검증에는 `originals/`를 사용하며 benchmark/이미지를 학습·증강에 재사용하지 않는다.

- R1 원본 archive `results.tar.gz`: **332,612,462 bytes**, SHA-256 `11cba0dff07a228c7ee70332f3b10859b66894a9349687507017c5072a3444e6`.
- R2 원본 archive `mmdl-cont32k-analysis-20260922.tar.gz`: **350,207,727 bytes**, SHA-256 `67d2734d909aca42043d12ae17ed5364cf50d3cb19fbbec25504d073b7cc7416`. supplement/receipt도 함께 보존한다. 모델 가중치·전체 HF cache·venv가 담긴 배포본은 아니다.
- 원본 ID의 `assignment`/`analysis` 문자열은 추적용으로 유지한다. 지금도 별도 테스트용/분석용 모델·프롬프트가 있다는 의미가 아니다.
- **보고서 반영은 별도 미완료 항목:** 이 원장 작성 시 `Assignment_1.md`와 `reports/mmmu_baseline.md`에는 최종 R2 수치·파싱 진단이 아직 반영되지 않았다. 이후 S1/S2를 따라 함께 갱신하고, 공식값과 차이 진단의 **1000자 제한은 제출 보고서**에 적용한다. 이 원장을 보고서 제출 완료로 간주하지 않는다.

### G. 다음 개선을 추적하는 방법과 발표 원칙

| 추적 ID / 상태 | 변경·확인 대상 | 완료에 필요한 증거 |
|---|---|---|
| R1 / COMPLETE | 초기2048 평가 | 위 원본427/900·commit·receipt 보존 |
| R2 / COMPLETE | 32768+vLLM 연속 처리 평가 | 위 원본459/900·commit·receipt 보존; 현 baseline |
| D1 / DIAGNOSED | 짧은 `LETTER. 보기` 파싱 누락 | 32개 형식 일치/24개 정답 손실, 원인 함수와 읽기 전용 진단 기준; 수정 완료 아님 |
| SC1 / PLANNED | 일반화한 parser 수정·새 scoring version | 합성 회귀 검사 후 **동일 저장 raw900 전체** 재채점, old/new 답·점수 차이 및 PARSED 회귀 확인; R1/R2 원본 덮어쓰기 금지 |
| SC2 / PLANNED | Qwen 규칙/LLM 보조 채점과 대조 | MCQ/open별 차이 분리, judge 모델/버전/프롬프트/비용 기록; 유료 API는 별도 승인 전 호출 금지 |
| I1 / PLANNED | 이미지 budget/전처리 동등성·장문 반복 원인 | 사전 고정 표본/가설, 한 요인씩 바꾼 protocol, 길이/반복/점수/시간/메모리 함께 측정; 900 재추론은 새 승인 필요 |
| T1 / NOT STARTED | 실패 유형→외부 데이터→실제 fine-tuning | MMMU/MMMU-Pro 오염 차단·학습 manifest·parameter update·고정 checkpoint·동일 평가 조건 비교 |

후속 기록은 **날짜 → 문제/근거 ID → 가설 → 변경한 단일 요인과 유지 조건 → 코드 commit·protocol/scoring version·환경 → 전후 정답 수/파싱 실패/length/시간·메모리 → 회귀·한계 → 증거 경로·상태** 순으로 이 원장에 덧붙인다. 실패한 실험도 삭제해 성공만 남기지 않는다. 현재 요청은 기록이며 PLANNED 항목의 실행 승인이 아니다.

발표에서는 **① 초기 상한의 한계 → ② 연속 처리와32k 적용 후 관측 변화 → ③ 남은 반복/파싱/공식 평가 차이 → ④ 원인을 분리할 다음 실험 → ⑤ 외부 데이터와 학습 계획** 순서로 연결한다. **47.44→51.00%는 실행 조건 묶음의 관측 변화**, **53.67%는 아직 가정 계산**, **학습 개선 성과는 아직 없음**을 명시한다. 점수 상승을 위해 프롬프트/파서를 바꿨다면 모델 학습 성능과 분리하고, baseline과 fine-tuned 모두 같은 최종 평가/채점 조건으로 비교한다.

## 1. 프로젝트·근거 자료

`[수업:S3 pp.4–5; S4 pp.14–20]` Qwen3-VL-4B-Instruct baseline→실패 분석→외부 데이터/학습법→fine-tuning→동일 평가·개선→최종 MMMU test/MMMU-Pro Standard10-way 순서다. Fine-tuning은 실제 파라미터 업데이트이며 전체/일부 학습, LoRA/QLoRA/adapter가 후보다. 원본 구조 유지가 기본이고 구조 변경은 필수가 아니다. 프롬프트/RAG/앙상블/파싱만으로 완료하지 않는다. [S4 pp.15,18–19] 초파리 신경망·압축·트레이딩을 섞지 않는다.

| ID | 자료 | 근거 |
|---|---|---|
| S1 | `assignment_guidance.md` / `(1)` 사본 | 고정스펙·채점·경로·마감 |
| S2 | `SUBMISSION_TEMPLATE.md` / `(1)` 사본 | 보고항목·실측·결과표 |
| S3 | `02_MMDL_Team_Project_Introduction.pdf` | 흐름·GitHub·평가·발표 |
| S4 | `03_MMDL_VLM_Introduction.pdf` | 입력/파싱·학습·데이터금지 |
| S5 | `01_MMDL_Course_Introduction.pdf` | 범위·HW·Standard10-way |
| S6 | `04_MMDL_Deep_Learning_1(2).pdf` | 파라미터·손실·최적화 |
| S7 | `05_MMDL_Deep_Learning_2.pdf` | 역전파·목적함수·softmax |
| S8 | 사용자 PPP#1 공지 | 발표일정·시간·LMS |

사본은 이름이 아니라 hash로 확인한다. S3 p.15의 `assignment/assignment1.md`와 S1 §4의 `reports/mmmu_baseline.md` 차이는 후자에 본문, 전자에 링크를 두고 기록한다. “validation으로 tune”은 정답 학습 허용이 아니다. 모든 MMMU/MMMU-Pro는 학습 금지다. [S4 pp.14,16,21] S6/S7은 이론이지 특정 학습률/배치/레이어 지시가 아니다.

## 2. 변경하면 안 되는 수업 조건

`[수업: S1 §§0–4; S2; S4 pp.14,16,21]`

```yaml
model_id: Qwen/Qwen3-VL-4B-Instruct
model_revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
dtype: bfloat16
mmmu_dataset_id: MMMU/MMMU
mmmu_dataset_revision: 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
assignment_split: validation
assignment_subjects: 30
assignment_samples_per_subject: 30
assignment_samples_total: 900
assignment_report: reports/mmmu_baseline.md
assignment_deadline: "2026-09-28 23:59 (수업 공지 기준)"
```

- 공식 Qwen 배포본만 사용한다. 커뮤니티 재업로드·GGUF·AWQ 등은 baseline 대체물이 아니다.
- 모델뿐 아니라 processor/tokenizer/chat template도 동일한 지정 revision으로 받는다.
- 30개 subject config를 각각 로드한다. 로드 오류가 난 과목을 버리거나 900개 중 일부만 제출하지 않는다.
- 과목별 accuracy 30개와 macro average를 기록한다. 전체가 과목당 30개이면 `정답 수 / 900`과 같다.
- 공식 비교값은 **과제 지침에 제시된 67.4**이다. 이를 우리 측정값처럼 쓰지 않는다.
- 공식값과의 차이 진단은 **1000자 이내**다. 표면적 추측보다 응답·설정·보조 실험의 근거를 쓴다.
- 한 명령으로 평가를 재실행하고, 모델 위치와 데이터 위치를 인자/환경변수로 교체할 수 있게 한다.
- 프롬프트 전문·출처, 생성 설정·근거, 파싱 규칙·출처, 환경·의존성·시간·peak VRAM을 기록한다.
- 4090 24GB에서도 실행 가능한 경로를 제공한다. 24GB가 강제 최대치라는 뜻은 아니다. [S1 §1.3]

현재 Assignment 1은 **MMMU validation만 필수**다. MMMU-Pro와 MMMU test는 최종 단계다.
MMMU-Pro는 수업이 지정한 **Standard 10-way**를 기준으로 삼고 vision-only를 대신 보고하지 않는다. [S5 p.11]

## 3. 저장소 전체 목표 구조

`[설계]` 폴더명은 수업 강제규정이 아니다. 기존 코드를 보존하고 필요한 파일만 구현한다. 예정 파일을 완료로 표시하지 않는다.

```text
MMDL/
├── AGENTS.md / README.md / pyproject.toml
├── .gitignore / .gitattributes / .env.example
├── env/requirements.in / requirements-{eval,train}.lock / environment.json
│       Dockerfile                         # 선택, image digest 기록
├── configs/eval/{mmmu_val_v1,mmmu_test_v1,mmmu_pro_standard10_v1}.yaml
│           hardware/{rtx5060_8gb,rtx4090_24gb}.yaml
│           data/{sources_v1,augmentation_v1}.yaml
│           train/{lora_pilot,qlora_pilot,final}.yaml
├── prompts/{mmmu_mcq_v1,mmmu_open_v1,mmmu_pro_mcq_v1}.txt
├── scripts/check_config.sh / doctor.sh / test_reference.sh
│           prepare_data.sh / augment_data.sh / train.sh
│           eval.sh / rescore.sh / reproduce.sh
├── src/mmdl/runtime/                      # 계약·환경·배치·생성·hash·경로·artifact
│           evaluation/                   # CLI·engine·prompt·이미지·parser·저장·재개
│             datasets/{mmmu,mmmu_pro}.py
│             backends/{transformers_backend,vllm_backend}.py
│           data/ / augmentation/ / training/
├── manifests/{sources,datasets,models,runs}/
├── third_party/ / tests/{unit,integration,fixtures}/
├── results/<run_id>/                      # 작은 공개 결과
├── reports/{mmmu_baseline,project_plan,progress,final_report}.md
├── assignment/assignment1.md
├── docs/{PROJECT_IMPLEMENTATION_GUIDE,SOURCE_REVIEW,REPRODUCIBILITY,PROJECT_STATUS}.md
└── .github/workflows/ci.yml               # CPU 검사만
```

`src/mmdl`은 필요한 `__init__.py`를 갖춘 설치 가능한 패키지로 만든다. 평가/학습/증강 CLI를 분리하고 학습기에서 평가로더를 import하지 않는다. RunPod는 기존 `reproduce.sh`와 runtime을 확장한다.

## 4. 코드·모델·데이터·결과 저장 위치

GitHub에는 코드·설정·lock·출처/manifest·작은 결과만 둔다. 모델/benchmark 원문/cache/venv/강의 원본/secret/개인경로/전체 응답·이미지는 공개하지 않는다. 대용량 구조는 다음과 같다.

```text
HF_HOME/                                  # pinned cache
MMDL_DATA_ROOT/evaluation/{mmmu,mmmu_pro}/  # 학습·증강 접근 금지
MMDL_DATA_ROOT/training/{raw,clean,augmented,final}/
MMDL_ARTIFACT_ROOT/{checkpoints,generations,runs,imports}/<version_or_id>/
MMDL_VENV_ROOT/                            # 저장소 밖 격리 환경
```

로컬 저장소는 현재 위치, 대용량/venv는 WSL Linux 파일시스템에 둔다. 인자/환경변수로 경로를 바꿀 수 있게 한다.

```bash
export HF_HOME="${HF_HOME:-$HOME/mmdl-cache/huggingface}"
export MMDL_DATA_ROOT="${MMDL_DATA_ROOT:-$HOME/mmdl-data}"
export MMDL_ARTIFACT_ROOT="${MMDL_ARTIFACT_ROOT:-$HOME/mmdl-artifacts}"
export MMDL_VENV_ROOT="${MMDL_VENV_ROOT:-$HOME/mmdl-envs/eval}"
```

유효 cache를 먼저 확인하고 재다운로드·실물복제·전체 snapshot을 피한다. WSL 내부와 VHDX Windows 볼륨, 다운로드/변환/회수 압축해제 peak를 검사하여 **로컬50GiB 여유**를 남긴다. 부족하면 임의 삭제하지 않는다.

RunPod는 GitHub 새 checkout/격리 환경에서 공식 pinned 모델·데이터를 받는다. **로컬 코드/venv/모델/cache 업로드는 금지**하고 평가 run 안에서 cache를 재사용한다. 경로의 실제 mount를 검사하며 미회수 결과는 stop에도 남는 작업 전용 Pod volume disk에 둔다. container disk만을 유일 저장소로 쓰지 않고 Network Volume은 기본 생성하지 않는다. [W10,W11] 대여 용량은 설치/취득/변환/단일 결과/bundle peak+기록된 안전여유로 정하며 로컬50GiB를 무조건 복사하지 않는다.

MMMU는 지정 validation30config×30 고유ID/유형을 검증한다. `MMMU_DEV_VAL` TSV·dev 포함본·다른 revision으로 대체하지 않는다. Pro `standard (10 options)` test는 취득 시 공식 전체 commit SHA를 한 번 고정해 보관/hash 확인만 한다. 향후 Pro 표본수를900으로 가정하지 않는다.

출처/license/revision·파일 목록/크기/hash와 실제 비공개 경로를 추적하고 다운로드/평가 완료를 구별한다. 채택한 외부 생성 학습 데이터는 서비스 재호출 대신 artifact를 고정·복원하며 접근권한/보존기간/hash/복원 명령을 남긴다. 강의 원본은 Git 제외 `local_sources/`에 두고 파일명/페이지/hash만 추적한다. `.gitignore`는 경로 단위로 적용하고 `results/` 전체를 숨기지 않는다. 이미 추적된 비공개/대용량 파일도 검사한다.

## 5. 공통 평가 조건과 P0

`[설계]` 로컬 smoke와 RunPod 단일 평가 run, 이후 fine-tuned MMMU 평가에 **같은 평가 protocol**을 사용한다. 모델/processor/BF16, 데이터 revision/순서, 이미지 변환, P0, 생성/seed/길이, parser/scoring/실패 처리를 고정한다. backend·batch·attention은 선택한 protocol에 기록한다. smoke/evaluation은 실행 범위이며 서로 다른 점수 역할이 아니다. legacy A/B run은 역사 자료로 보존하고 새 run과 섞지 않는다.

### 역사적 초안: `configs/eval/mmmu_val_v1.yaml`

아래 DRAFT/2048 발췌는 초기 설계 이력이다. 실제 R1은 고정 설정으로 완료했고, 현재 채택한 R2는 `configs/eval/mmmu_val_continuous_vllm_v1.yaml`의 FROZEN/32768이다. 최신 실측·선택 근거는 §0.1을 따른다.

```yaml
status: DRAFT
protocol_id: mmmu-val-v1
model:
  id: Qwen/Qwen3-VL-4B-Instruct
  revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
  dtype: bfloat16
dataset:
  id: MMMU/MMMU
  revision: 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
  split: validation
  expected_total: 900
generation:
  do_sample: true
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  repetition_penalty: 1.0
  presence_penalty: 1.5
  seed: 3407
  seed_policy: per_sample_sha256_v1
  max_new_tokens: 2048
  num_beams: 1
image:
  min_pixels: 262144
  max_pixels: 1310720
  resize_owner: official_processor
execution:
  backend: transformers
  batch_size: 1
  attention: sdpa
  sdpa_kernel: math
  deterministic: true
```

sampling은 Qwen Instruct Evaluation Reproduction, seed3407은 공식 README를 근거로 확인한다. [W1,W2] 참고 실행 코드 seed42와의 차이, per-sample seed가 팀 설계임을 기록한다. 지정 model card VL 길이16384와 GitHub 평가안내32768의 차이를 숨기지 않는다. **2048과 pixel budget은 실측 전 팀 초안**이며 공식 정답값이 아니다. 기존 smoke 증거를 우선 활용하여 메모리·속도·잘림으로 공통값을 확정한다. 자동 fallback이나 점수 기반 조정을 하지 않는다.

### P0의 근거·전문·이미지 입력

Qwen3-VL Technical Report `arXiv:2511.21631v2` 부록 B.1 p.34, Qwen `evaluation/mmmu/run_mmmu.py`의 `build_mmmu_prompt`/`prepare_inputs_for_vllm`, VLMEvalKit `vlmeval/vlm/qwen3_vl/prompt.py`의 `_build_mmmu_prompt`를 대조한다. 실제 흐름·URL·조회일·각 코드 commit·파일/함수/관련 위치·채택 근거를 `docs/SOURCE_REVIEW.md`에 한 번 기록한다. 이미 확인된 근거를 반복 탐색하거나 후보별 900문제 경쟁평가하지 않는다.

객관식은 원래 순서의 **모든** 보기를 `A. 내용` 형식으로 한 줄씩 직렬화한다.

```text
Question: {question}
Options:
{options}
Please select the correct answer from the options above.
```

주관식은 `Question: {question}`만 쓰고 Options/선택 지시문을 생략한다. 유효한 원본 hint가 있을 때만 `Hint: {hint}\n`을 앞에 붙인다. 정답·해설에서 hint를 만들지 않는다. 실제 이미지 content를 참조 관계/순서대로 먼저 넣고 P0 텍스트를 마지막에 둔 단일 user message에 지정 processor의 `apply_chat_template(add_generation_prompt=True)`를 적용한다.

P0 파일·chat template·입력 변환을 hash로 고정한다. 이미지 누락/문자열 대체, 이중 resize, Qwen2.5의 28계수 가정 복사를 막고 실제 image tensor·크기·grid·token hash를 확인한다. Qwen3-VL processor의 pixel budget/32계수 안내와 강의 예시를 구분한다. [W1] 추가 system/few-shot/CoT/JSON/answer-reason 강제/도구/대화 이력을 넣지 않는다. 분석용/테스트용 별도 P1을 만들지 않는다. Transformers reference는 공식 vLLM 실행과 동일하다고 주장하지 않는다.

## 6. 로컬 5060 / RunPod 4090 하드웨어 profile

`[사용자 전제]` 노트북 RTX 5060 8GB·RAM24GB, 재현 대상은 **단일 RTX 4090 24GB GPU-only**다. 사용자 승인으로 이번 full900 두 실행은 `configs/hardware/rtx3090_24gb.yaml`의 **RTX 3090 24GB**에서 완료했다(§0.1). 아래 4090 값은 weight 배치 초안이며 실제 측정/성공 보장이 아니다.

| profile 항목 | rtx5060_8gb | rtx4090_24gb |
|---|---|---|
| placement | cpu_offload | gpu_only |
| 허용 평가 | 로컬 SMOKE/PARTIAL, 30문제 규모 | RunPod 단일 full evaluation, 향후 승인된 final |
| gpu_index / expected_vram_gib | 0 / 8 | 0 / 24 |
| gpu_weight_cap_gib / gpu_reserve_gib | 5.5 / 2.0 | 20.0 / 3.0 |
| cpu_weight_cap_gib / cpu_available_fraction | 12.0 / 0.65 | 12.0 / 0.65 (weight offload에 사용 안 함) |
| min_free_gpu_gib | 3.0 | 12.0 |
| allow_disk_offload / num_workers | false / 0 | false / 0 |

RAM과 VRAM은 합산 VRAM이 아니다. Accelerate CPU offload의 `max_memory`는 주로 가중치 배치 예산이며 KV cache/활성값/workspace peak 제한이 아니다. [W3] 5060은 CPU 전처리와 RAM weight offload+GPU 추론을 사용한다. GPU cap은 실가용량에서 reserve를 빼고, CPU cap은 **WSL available RAM** 비율로 제한한다. 4090은 실제 전체 가중치 GPU 배치를 검사하며 부족하면 중단한다.

`device_map="auto"`는 OOM 방지/학습 성공 보장이 아니다. offload 뒤 `model.to("cuda")`를 호출하지 않는다. 정수 `input_ids`/`image_grid_thw`를 BF16으로 바꾸지 않는다. `inference_mode`, RNG, determinism, SDPA kernel과 실제 device map을 확인한다. disk offload는 금지다.

4090 품절/환경 실패를 이유로 고가 GPU·다른 backend·양자화·greedy·batch 확대에 자동 전환하지 않는다. 다른 GPU는 가격/예산과 별도 hardware profile을 승인·검증하고 실제 모델명으로 기록한다. 다른 24GB+ GPU 실행을 **4090 검증 완료**라고 하지 않는다.

## 7. 환경·GitHub clean-clone·교차 GPU 재현

`doctor`는 GPU/freeVRAM·CPU/availableRAM·OS/WSL·driver·Python·torch/CUDA runtime·compiled architectures·패키지·CUDA/BF16 실제 연산·disk/mount를 기록한다. `nvidia-smi` 표시와 torch runtime을 구별하고 wheel의5060지원도 공식 확인한다. WSL에 Linux display driver를 설치하지 않는다. [W4,W5]

같은 Python minor와 성공한 exact lock을 사용한다. 재현에 `latest`/무조건 `pip install -U`를 쓰지 않는다. GitHub 참고코드 commit·HF revision·Transformers 구현 버전을 구별한다. 필요한 외부 코드만 license/출처/commit/수정점과 관리한다. Docker image digest도 driver 차이를 없애지는 않는다.

**유료 배포 전** 로컬 검증·공개/secret 검사·README/lock·회수/정리를 준비한다. Git remote로 팀MMDL을 확인해 명시적 파일만 commit/push한다. force push/이력 삭제/공개범위·계정설정 변경을 금지한다. 강의 원본 없이 공개된 고정 계약으로 실행되어야 한다.

RunPod는 GitHub **전체 commit SHA**·clean tracked files·lock hash를 검사한다. 미commit 파일 전송·Pod 수동 패치·실행 중 branch 이동은 금지한다. 수정은 로컬 검사→새 commit/push→새 checkout으로 처리하고 결과는 외부 root에 쓴다.

같은 코드/seed도 GPU 간 응답·점수 동일성을 보장하지 않는다. [W6] 절차/결과/재학습 재현을 구별하고 입력 hash→raw→parsed→score를 비교한다. 이후 baseline/fine-tuned는 가능한 같은4090 환경·P0로 비교한다.

## 8. 한 명령 재현과 실제 평가 1회

아래는 **구현 후 검증할 CLI 계약**이다. 기존 동등 인자를 재사용하면 README도 일치시킨다. 파일 존재를 실행 성공으로 표시하지 않는다.

```bash
# 로컬30문제: 유효한 기존 실행은 반복하지 않는다.
bash scripts/eval.sh --protocol configs/eval/mmmu_val_v1.yaml \
  --hardware configs/hardware/rtx5060_8gb.yaml --model-ref manifests/models/baseline.json \
  --model-path "${MMDL_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --run-id smoke-local-30 \
  --mode partial --subject Accounting --limit 30
# RunPod: 설치→doctor→취득→최소GPU검증→단일900→검증/포장
bash scripts/reproduce.sh --stage evaluate --target runpod --suite mmmu-val \
  --commit "$MMDL_CODE_COMMIT" --protocol configs/eval/mmmu_val_continuous_vllm_v1.yaml \
  --hardware configs/hardware/rtx4090_24gb.yaml --model-ref manifests/models/baseline.json \
  --model-path "${MMDL_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --job-id "$MMDL_JOB_ID" --execute
# 저장raw만 재채점
bash scripts/rescore.sh --artifact-root "$MMDL_ARTIFACT_ROOT" --run-id "$RUN_ID"
```

README에는 Git 설치된 승인 Pod에서 **GitHub 취득→전체 SHA 검사→reproduce를 한 번의 shell 호출**로 수행하는 완결 명령도 둔다. 수동 코드수정/설치 탐색 없이 경로만 바꿔 실행한다. `--model-path`/`MMDL_MODEL_PATH`로 모델 위치를 바꾸되 baseline revision/hash를 검사한다. `reproduce.sh`는 기본 dry-run, `--execute`로 실행하며 학습/증강/final을 자동 시작하지 않는다.

새 실행은 `<job_id>-evaluation` 하나로 식별한다. `--suite mmmu-val`은 smoke 후 하나의 900문제 평가를 수행하며, 저장된 raw response와 sample record를 점수 산출 및 실패 검토에 함께 사용한다. 별도 분석 900회를 새로 만들지 않는다. 이미 실행 중인 legacy continuous32k run은 전체900·독립검증 완료 후 같은 목적의 결과로 채택할 수 있으나, 다른 run과 합치거나 평균내지 않는다. 같은 run 재개에서는 성공 ID 중복생성을 막는다. smoke/중단 시도는 본 평가와 별도 상태로 기록한다.

기존 `<job_id>-assignment`/`<job_id>-analysis` artifact는 legacy 형식으로 읽을 수 있게 보존한다. 새 단일 run으로 이름을 바꾸거나 서로 합치지 않는다. legacy run을 재개할 때는 기록된 원래 commit/protocol을 사용하며 최신 CLI의 새 role로 재실행하지 않는다. continuous32k legacy run이 900개와 독립검증을 완료하면 보고서 full score 근거로 사용할 수 있다. 새 CLI는 legacy role을 새 run으로 재분류하지 않는다.

Bash는 `set -euo pipefail`, script 기준 root, 안전한 인자/실패 전파를 적용한다. 로컬 제어측에서 회수/정리를 연결해 Pod 삭제 후에도 증거를 남긴다. 교수님은 RunPod 계정 없이4090에서 `eval.sh` 한 명령으로 평가를 재현할 수 있어야 한다.

## 9. 평가 engine·파싱·모델 풀이 기록

30과목각30·고유ID/유형·원래 보기·이미지 참조/순서를 검증한다. 주관식을 버리거나 객관식으로 바꾸지 않는다. 정답/해설은 채점·저장에만 전달한다. [S4 pp.8,11;W7]

`presence_penalty` 실제 적용을 검사하고 미지원이면 테스트된 **generated-only logits processor**를 쓴다. 무시/중복 적용을 금지하고 대상/순서·실제 generation config(eos/pad/stop/default 포함)를 저장한다. 동일 evaluation seed는 다음 팀 정책이다.

```python
import hashlib
value = hashlib.sha256(f"{master_seed}:{sample_id}".encode()).digest()
seed = int.from_bytes(value[:8], "big") % (2**31)  # master_seed=3407
```

매 샘플 Python/NumPy/Torch RNG를 설정하고 ID 순서를 고정한다. 공식 `mmmu/utils/eval_utils.py` parser/open-ended evaluator의 commit/license/수정점을 기록한다. **무작위 추측 fallback은 NO_PARSE=오답**으로 바꾸고 생성 답변만 파싱한다. 빈답/파싱 실패도 완료 추론이면900 분모에 포함한다. OOM/입력 오류/누락은 시스템 실패이고 INCOMPLETE다.

단일 evaluation은 ID·과목·유형·질문·원래 보기·이미지 참조·실제 prompt/messages·seed·입력 hash·**raw response 전체**·추출답·gold·정오·파싱 상태·종료 사유·token 수·시간을 외부 JSONL에 저장한다. `model_explanation`은 실제 출력 원문 구간/offset·추출 규칙만 보존한다. 없으면 `null/NOT_GENERATED`, 불명확하면 상태와 raw를 남긴다. **전문제 기록은 전문제 풀이 생성 보장이 아니다.** Codex/외부 모델이 답/풀이를 채우지 않는다. `gold_explanation`은 데이터셋에 있을 때만 따로 둔다. 설명 사실성과 정답 일치는 별개다. [S4 p.12]

같은900개 질문/보기/이미지/응답/실제 풀이/정답으로 단순 로컬 HTML/Markdown 열람본을 만든다. 추가 추론이나 원문 요약/삭제 없이 HTML escape·안전한 상대경로를 적용한다. 사용 이미지만 공유 assets로 한 번 보존하거나 검증된 로컬 data root에 연결해 Pod 삭제 후 열람을 보장한다. base64 중복·무거운 웹앱·불필요한 hidden state/attention 전체 저장은 금지한다.

공개 `results/<run_id>/`에는 작은 summary·30과목 scores·resolved eval/hardware config·sanitized environment·run manifest·외부 상대참조/hash만 둔다. 전체 predictions/failures JSONL·열람본/이미지는 외부 root에 둔다. run별 peakGPU/RAM·전체시간과 다운로드/설치/모델로드/평가/포장/전송을 구별한다. 반올림 전 macro=정답수/900, 공식 차이의 퍼센트포인트를 검사한다.

## 10. manifest·재개·비용·회수·정리

`source revision→data/code/config→checkpoint→eval protocol→run→성적표`를 연결한다. 모델은 base/adapter/full/merged·base revision·artifact/hash·processor·train/data config·code commit을, 데이터는 source/revision/split/license·부모ID·변환/seed·상대경로/크기/hash·count·검수/오염검사를 기록한다. LoRA 외 vision/projector도 저장하며 학습 재개에는 optimizer/scheduler/step/RNG/data 순서를 포함한다.

protocol hash는 P0/chat template/입력변환/dataset 내용 manifest/parser/evaluator를 묶는다. 절대경로/run_id/시각/로그 상세도는 execution manifest로 분리한다. 모델hash·code commit·env lock·device map/kernel·dirty patch와 이후 report commit을 구별한다. atomic/append-only 저장 후 ID·모델/protocol/코드/안정적 환경/하드웨어 일치를 검사해 재개한다. 원본 덮어쓰기·5060/4090/A/B 혼합은 금지한다. 재채점 변경은 새 scoring version에 남긴다.

### 비용 gate

새 유료 작업은 승인된 Pod 범위에서만 수행한다. 당시 가격/GPU/disk·비용 정책·전송/복구 여유를 기록한다. 가격은 공식 console에서 확인한다. [W12] 자동충전/결제·계정한도 변경, 다른 유료 서비스/LLM judge·미승인 Pod·고가 GPU 확대는 금지한다. 평가에 필요한 cache를 재사용하며 인적 분석/문서 편집을 기다리며 idle시키지 않는다. 설치/취득/전송/재시도 비용도 기록하며, 현재 비용 상한 철회와 기본 strict 정책을 구분한다.

대여 전 로컬 수신/여유 공간·SSH 전송·작업 전용 Pod/volume ID·정리 권한을 확인한다. SSH 단절에도 작동하는 최대시간/진행정지 watchdog을 준비하되 GPU이용률0만으로 중지하지 않는다.

### 회수와 삭제 gate

**REMOTE_VERIFIED→LOCAL_VERIFIED→POD_TERMINATED→OWNED_STORAGE_DELETED→CLEANUP_VERIFIED** 순서다.

평가 결과·공유 이미지·JSONL/열람본·환경/설정/출처/로그/명령을 회수한다. 과거 서로 다른 run은 합치지 않는다. 파일 path/크기/SHA-256·archive hash·job/run/code/protocol을 기록한다. 모델/venv/전체 HF·benchmark cache/강의 원본은 회수하지 않는다.

로컬에서 SSH/SFTP/rsync로 pull한다. 점진 회수는 닫힌 shard만 하고 최종 전체 hash를 검사한다. 안전한 압축해제(경로이탈/외부 symlink 금지), 각900 고유ID/30과목/필수필드/산술·저장raw 재채점·이미지 열람을 검증해 **로컬 receipt**를 만든다. 원격 자체검사/전송 exit code만으로 삭제하지 않는다.

검증 후 해당 작업 Pod를 **Terminate/Delete**하고 별도 생성한 Network Volume 등은 정확한 ID/소유권/미공유를 확인해 삭제한다. **Stop/파일rm만으로 정리 완료라 하지 않는다**. [W10,W11] 기존/공유 volume·다른 프로젝트·전체 계정 자원은 삭제하지 않는다. 승인 범위의 정리는 조건 충족 시 재승인 없이 수행하되 식별/권한이 불명확하면 막는다. 계정 API/console에서 잔존 자원을 확인하고 ID·시각·응답은 로컬에 보존한다. 이미 발생한 비용까지0이라고 하지 않는다.

회수 실패는 최대3회 복구 후 **미검증 원격 결과를 삭제하지 않는다**. 결과가 Pod volume disk에 안전한지 확인하고 GPU를 stop하여 idle compute를 줄인 뒤 `RECOVERY_REQUIRED/CLEANUP_PENDING`과 storage 과금 가능성을 보고한다. 무기한 방치/무검증 자동삭제/자동 새Pod 재실행을 금지한다. 복구·로컬 검증 후 정리한다. 보존과 비용 완전 중단을 동시에 달성할 수 없는 장애는 차단 사항이다. [W10,W11]

## 11. 증강·학습 계획

`[수업: S4 pp.14–21]` 외부 공개/합성/타 모델 생성 데이터가 후보이다.
데이터·학습법의 구체값은 아직 미확정이다.
**실패 분석→관련 연구→원인 가설→데이터/학습법→실험 계획**으로 정한다.
증강은 의미 보존을 검증한다. crop/회전/반전이 글자·축·방향·화학 구조·보기 참조를 바꿀 수 있으므로 무조건 적용하지 않는다.
차트/도형은 수치·구조를 먼저 만들고 렌더링·정답 검증하는 방식을 후보로 둔다.
MMMU 원본을 paraphrase하거나 수치/이미지만 바꾼 파생 학습 데이터도 금지한다.
오염검사에는 benchmark 정보를 제외 신호로만 사용하고 학습기로 전달하지 않는다.

**추론용 `device_map='auto'`를 학습용 offload로 그대로 재사용하지 않는다.** [W3]
이번 5060 작업은 평가기·데이터 처리·CPU 검사·30문제 검증까지다. 아래 학습 pilot은 이후 별도 승인 단계다.
8GB BF16 full fine-tuning 성공을 전제하지 않는다. 실제 학습은 자원 확보와 memory smoke 후 결정한다.
LoRA rank/대상 모듈/freeze 범위는 실험 조건이며 GPU에 따라 몰래 바꾸지 않는다.
QLoRA는 별도 학습 실험이다. 수업의 QLoRA 허용 예시와 baseline BF16 무양자화 규칙을 구분한다. [S4 p.18; S1 §1.1]
양자화 학습 후 평가의 base/adapter/merge/precision 정책을 명시한다.

```text
effective batch = micro_batch × accumulation steps × world_size
1×16×1 = 4×4×1 = 16 (표본 수 기준)
```

같은 effective batch만으로 동일 학습은 보장되지 않는다.
가변길이 token loss 정규화, 샘플 순서, update 횟수, scheduler, dropout, accumulation을 함께 맞춘다.
학습 collator의 prompt/padding/image token loss mask와 답변 supervision을 검사한다.
작은 외부 데이터로 실제 parameter update와 checkpoint 저장/재로딩을 확인한다.
원본 학습→증강 추가→학습 영역 변경을 분리하고 데이터량/step이 같이 바뀌면 교란 요인으로 보고한다.

## 12. 테스트·완료 기준·교수님 재현

CPU CI는 고정계약/금지override·정답누출·이미지/주관식·parser/penalty·coverage/산술·hash/재개·secret/대용량·archive 경로를 검사한다. fixture는 자체 제작한다. hardware의 dtype/prompt/generation/image/backend/batch override를 거부한다. 로컬full·A/B응답cache·receipt 없는 삭제·잘못된Pod/공유volume 삭제 거부는 mock으로 검사한다.

`로컬1→1과목30→GitHub고정→RunPod최소smoke→단일 evaluation900→회수→정리` 순서다. 다중이미지/주관식은30문제/fixture 안에서 우선 확인하고 부족한 유형만 최소 추가SMOKE로 분리한다. 유효30/완료full을 반복하지 않는다.

| 독립 상태 | 증거 |
|---|---|
| 구현/CPU | 실제 검사 명령/결과 |
| 로컬SMOKE/PARTIAL | 30ID·protocol·peak·시간·입력/파싱/재개 |
| GitHub 재현 | commit·clean checkout·lock·한 명령 로그 |
| RunPod/실제4090 | doctor·BF16·device map·실GPU명 |
| evaluation FULL | 900/900·실제생성·raw·30과목·산술 |
| 로컬 회수 | hash·재채점·coverage·열람·receipt |
| 자원 정리 | 전용Pod/storage 삭제·잔존조회 |

미실행/실패를 PASS로 채우지 않는다. 제출은 검증된 단일900 결과와1000자 이내 근거 기반 격차 진단이다. 사용자 요청으로 확인한 1차 진단과 후속 미실행 항목은 §0.1에 구분한다. 교수님은 공개코드commit→lock→4090 doctor→pinned artifact/hash→같은eval→재채점으로 재현한다. RunPod는 수업 강제 provider가 아니다. checkpoint 교체 재평가에는 재학습이 필요 없고 재학습 재현은 별도 요청이다.

## 13. 진행·제출·발표

`[수업: S3 pp.7,19–21; S1 §4; S8]` 팀장 public repo, 전체 팀원 collaborator,
각자 계정 commit/PR, default branch에 최종 병합을 따른다.
Assignment 본문은 `reports/mmmu_baseline.md`, 기존 경로에는 링크를 둔다.
전체 수업 흐름은 환경/입력/채점→baseline→실패분석→보고서/발표다. 평가·회수·클라우드 정리는 완료했고 현재 요청은 1차 실험 원장 작성이다. 템플릿 보고서 최종 반영·발표 제작·학습은 각각의 완료 증거와 요청 범위를 확인한다.
Assignment 마감은 **9/28 23:59**, 첫 발표는 학습 완료 보고가 아니라 개선 계획이다.

| PPP #1 | 팀 순서 |
|---|---|
| 9/29 화 | 9, 10, 11 |
| 9/30 수 | 8, 7, 1 |
| 10/6 화 | 12, 6, 3 |
| 10/7 수 | 13, 5, 4, 2 |

팀 번호는 미확인이다. 발표 **최소 10분+Q&A 5분**이며 공지에 상한은 없다.
PDF를 LMS `모델 개선 계획안 발표 - PPP #1`에 사전 제출한다. GitHub 제출과 별개다.
PPT/PDF·HDMI 노트북/강의실 PC·다음 팀 대기를 준비한다.
내용은 평가→실패사례→목표→관련연구→데이터/학습법→HW→실험/일정 순으로 연결한다.
전문용어를 설명하고 인용을 표시한다. 아직 실행하지 않은 목표를 성과로 쓰지 않는다.
`docs/PROJECT_STATUS.md`에 상태·담당·증거를 남긴다. 기존 팀 저장소의 완성도는 조사 전 추정하지 않는다.

## 14. 외부 근거·검증 상태

실제 Qwen/MMMU/VLMEvalKit commit·문헌 위치는 `docs/SOURCE_REVIEW.md`에서 고정한다. 기술 근거는 수업 규정을 대체하지 않는다.

| ID | 공식 근거 |
|---|---|
| W1 | Qwen recipe/processor: `https://github.com/QwenLM/Qwen3-VL` |
| W2 | 고정 model card: `https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md` |
| W3 | Accelerate: `https://huggingface.co/docs/accelerate/concept_guides/big_model_inference` |
| W4 | WSL: `https://docs.nvidia.com/cuda/wsl-user-guide/index.html` |
| W5 | Blackwell 배경: `https://pytorch.org/blog/pytorch-2-7/` |
| W6 | 재현 한계: `https://docs.pytorch.org/docs/2.7/notes/randomness.html` |
| W7 | MMMU 평가: `https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu` |
| W8 | Generation API: `https://huggingface.co/docs/transformers/main_classes/text_generation` |
| W9 | AGENTS.md: `https://developers.openai.com/codex/guides/agents-md` |
| W10 | RunPod Stop/Terminate: `https://docs.runpod.io/pods/manage-pods` |
| W11 | RunPod storage: `https://docs.runpod.io/pods/storage/types` |
| W12 | RunPod pricing: `https://docs.runpod.io/pods/pricing` |

W10–W12는2026-09-21 공식 문서 확인 근거다. 배포 시 가격/CLI/API/정리 동작을 다시 확인한다. 과거 구현·로컬30·환경 문제는 `docs/PROJECT_STATUS.md`, 2026-09-23 완료 상태·결과·진단·회수/삭제 증거는 **§0.1과 연결된 원본 receipt**로 판정한다. 실제4090·학습/final은 미실행이다. 사용자 실행 요약은 재확인 전 사용자 보고값이다. test 정답/revision/절차 미확정 시 prediction export와 점수를 구별한다.

기존 구현가이드/검증 기록의 필요 구간만 활용한다. 지침 byte·상위/하위 합산 한도·실제 로드 목록을 확인한다. 자동로딩에서 잘렸다면 최초 작업 때 누락 구간을 명시적으로 읽되 한도를 임의 변경하거나 미로딩 구간을 읽었다고 하지 않는다.
