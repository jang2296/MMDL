#!/usr/bin/env python3
"""Render the submission report from one independently verified COMPLETE900 run."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCORE = 67.4
FINAL_PROTOCOL_ID = "mmmu-val-fast-vllm-32k-continuous-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _validate_final_protocol(cfg: dict[str, Any]) -> None:
    execution = cfg.get("execution", {})
    if (cfg.get("protocol_id") not in {FINAL_PROTOCOL_ID, "mmmu-val-official-vllm-b2-v2", "mmmu-val-v8"}
            or cfg.get("generation", {}).get("max_new_tokens") != 32768
            or execution.get("backend") != "vllm"
            or execution.get("batch_size") != 2
            or execution.get("scheduling") != "continuous"):
        raise ValueError("Assignment report accepts only the frozen continuous 32k vLLM submission protocol")


def _config_path(run_dir: Path, manifest: dict[str, Any]) -> Path:
    public = ROOT / "results" / run_dir.name
    candidates = [run_dir / "resolved_eval_config.yaml", public / "resolved_eval_config.yaml"]
    for owner in (manifest, manifest.get("identity", {})):
        if not isinstance(owner, dict):
            continue
        for key in ("resolved_eval_config", "protocol_config", "protocol", "protocol_path"):
            value = owner.get(key)
            if not value:
                continue
            path = Path(str(value))
            candidates.extend((run_dir / path, ROOT / path))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No resolved evaluation config was found beside the run/public manifest")


def _hardware_path(run_dir: Path, manifest: dict[str, Any]) -> Path | None:
    public = ROOT / "results" / run_dir.name
    candidates = [run_dir / "resolved_hardware_config.yaml", public / "resolved_hardware_config.yaml"]
    for owner in (manifest, manifest.get("identity", {})):
        if not isinstance(owner, dict):
            continue
        for key in ("resolved_hardware_config", "hardware_config", "hardware_path"):
            value = owner.get(key)
            if value:
                path = Path(str(value))
                candidates.extend((run_dir / path, ROOT / path))
    return next((path for path in candidates if path.is_file()), None)


def _seconds(value: Any) -> str:
    if value is None:
        return "미제공"
    assert value is not None
    return f"{float(value):.3f} s"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _bytes_gib(value: Any) -> str:
    if value is None:
        return "미제공"
    return f"{float(value) / 1024**3:.2f} GiB"


def _load_complete_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    summary = _read_json(run_dir / "summary.json")
    environment = _read_json(run_dir / "environment.json")
    backend = _read_json(run_dir / "backend.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    # Legacy role labels are provenance, not a different evaluation protocol.
    if manifest.get("identity", {}).get("run_role") not in {"evaluation", "assignment", "analysis"}:
        raise ValueError("Report requires a managed evaluation with fixed-commit inference provenance")
    cfg_path = _config_path(run_dir, manifest)
    cfg = _read_yaml(cfg_path)
    _validate_final_protocol(cfg)
    if summary.get("status") != "COMPLETE" or str(summary.get("mode", "")).lower() != "full":
        raise ValueError("Report generation requires a COMPLETE full run")
    if summary.get("expected_count") != 900 or summary.get("completed_count") != 900:
        raise ValueError("Report generation requires 900 completed samples")
    if summary.get("denominator") != 900 or summary.get("missing_count") != 0:
        raise ValueError("Report generation requires a 900-sample denominator with no missing rows")
    if summary.get("unresolved_failure_count", 0) != 0:
        raise ValueError("Report generation requires no unresolved system failures")

    subjects = cfg.get("dataset", {}).get("subjects")
    if not isinstance(subjects, list) or len(subjects) != 30 or cfg["dataset"].get("expected_total") != 900:
        raise ValueError("Resolved protocol does not describe all 30 MMMU subjects")
    paths = sorted((run_dir / "samples").glob("*.json"))
    if len(paths) != 900:
        raise ValueError(f"Expected 900 sample records, found {len(paths)}")
    rows = [_read_json(path) for path in paths]
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != 900 or any(row.get("status") != "COMPLETED" for row in rows):
        raise ValueError("Sample records are not 900 unique completed rows")
    expected_ids = manifest.get("expected_ids")
    if isinstance(expected_ids, list) and set(expected_ids) != set(ids):
        raise ValueError("Sample IDs differ from the run manifest")
    counts = Counter(row.get("subject") for row in rows)
    expected_subjects = set(subjects)
    if set(counts) != expected_subjects or any(counts[name] != 30 for name in subjects):
        raise ValueError("Independent subject count is not 30 for each configured subject")
    correct = sum(row.get("correct") is True for row in rows)
    macro = correct / 900
    reported_macro = summary.get("macro_average")
    if reported_macro is None or summary.get("correct") != correct or not math.isclose(float(reported_macro), macro, abs_tol=1e-12):
        raise ValueError("Summary score does not match independent row arithmetic")
    public_dir = run_dir.parent.parent / "public" / run_dir.name
    try:
        from scripts.validate_run import validate
        validate(run_dir, public_dir)
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        raise ValueError(f"Report generation requires an independent recovered-run audit: {exc}") from exc
    return summary, environment, backend, manifest, cfg, rows, subjects


def _gpu_text(environment: dict[str, Any]) -> str:
    gpus = environment.get("nvidia_smi", {}).get("gpus") or environment.get("torch", {}).get("gpus") or []
    if not gpus:
        return "미제공"
    values = []
    for gpu in gpus:
        name = gpu.get("name", "unknown")
        if gpu.get("memory_total_mib") is not None:
            memory = f"{gpu['memory_total_mib']} MiB"
        elif gpu.get("memory_total_bytes") is not None:
            memory = _bytes_gib(gpu["memory_total_bytes"])
        else:
            memory = "VRAM 미제공"
        values.append(f"{name} ({memory})")
    return "; ".join(values)


def _preflight_text(run_dir: Path) -> str:
    candidates = [
        run_dir / "doctor.json",
        run_dir.parent.parent / "setup/doctor.json",
        run_dir.parent / "setup/doctor.json",
        ROOT / "setup/doctor.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        data = _read_json(path)
        environment = data.get("environment", data)
        if not isinstance(environment, dict):
            continue
        measured = []
        probe = environment.get("torch", {}).get("bf16_probe", {}).get("passed")
        if probe is not None:
            measured.append(f"BF16 probe={'PASS' if probe else 'FAIL'}")
        ram = environment.get("ram", {}).get("available_bytes")
        if ram is not None:
            measured.append(f"RAM available={_bytes_gib(ram)}")
        gpus = environment.get("nvidia_smi", {}).get("gpus", [])
        if gpus and gpus[0].get("compute_capability"):
            measured.append(f"GPU compute capability={gpus[0]['compute_capability']}")
        storage = data.get("storage")
        if isinstance(storage, dict) and storage.get("sufficient") is not None:
            measured.append(f"storage={'PASS' if storage['sufficient'] else 'FAIL'}")
        if measured:
            return "; ".join(measured) + " (measured preflight: doctor.json)"
    return "미기록 (doctor.json 없음)"


def _package_text(environment: dict[str, Any]) -> str:
    packages = environment.get("packages", {})
    if not isinstance(packages, dict):
        return "미제공"
    return "; ".join(f"{name}={packages[name]}" for name in sorted(packages)) or "미제공"


def _download_text() -> str:
    files = [ROOT / "manifests/models/baseline.json", ROOT / "manifests/datasets/mmmu.json"]
    values = []
    for path in files:
        if not path.is_file():
            continue
        data = _read_json(path)
        seconds = data.get("download_verify_seconds")
        if seconds is not None:
            values.append(f"{path.stem}: {_seconds(seconds)}")
    return "; ".join(values) if values else "미기록 (다운로드 검증 manifest에 시간 없음)"


def _command(run_dir: Path, manifest: dict[str, Any], cfg_path: Path | None = None) -> str:
    command_path = run_dir / "command.json"
    if command_path.is_file():
        command = _read_json(command_path).get("reproduce_command")
        if command:
            return str(command)
    for key in ("command", "reproduce_command", "eval_command"):
        if manifest.get(key):
            return str(manifest[key])
    protocol = "configs/eval/mmmu_val_v1.yaml"
    if cfg_path is not None:
        try:
            protocol = str(cfg_path.relative_to(ROOT))
        except ValueError:
            protocol = "configs/eval/mmmu_val_v1.yaml"
    return (
        "# run manifest에 실제 호출 문자열이 없을 때의 parameterized 재현 명령\n"
        f"bash scripts/eval.sh --protocol {protocol} --hardware configs/hardware/rtx5060_8gb.yaml "
        f"--model-ref manifests/models/baseline.json --model-path \"$MMDL_MODEL_PATH\" "
        f"--data-root \"$MMDL_DATA_ROOT/evaluation/mmmu\" --run-id {run_dir.name} --mode full"
    )


def _prompt_source(run_dir: Path) -> str:
    def read(name: str) -> str:
        snapshot = run_dir / "prompts" / name
        return (snapshot if snapshot.is_file() else ROOT / "prompts" / name).read_text(encoding="utf-8")

    mcq = read("mmmu_mcq_v1.txt")
    open_prompt = read("mmmu_open_v1.txt")
    return (
        "### MCQ template (`prompts/mmmu_mcq_v1.txt`)\n\n"
        "```text\n" + mcq + "```\n\n"
        "### Open-ended template (`prompts/mmmu_open_v1.txt`)\n\n"
        "```text\n" + open_prompt + "```"
    )


def _command_header(run_dir: Path, manifest: dict[str, Any]) -> str:
    command_path = run_dir / "command.json"
    if command_path.is_file() and _read_json(command_path).get("reproduce_command"):
        return "저장된 sanitized `command.json`의 재현 명령"
    if any(manifest.get(key) for key in ("command", "reproduce_command", "eval_command")):
        return "run manifest의 재현 명령"
    return "run artifact에 호출 문자열이 없어 parameterized 명령을 사용"


def _gap_analysis(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    no_parse = sum(row.get("parse_status") == "NO_PARSE" for row in rows)
    empty = sum(row.get("parse_status") == "EMPTY" for row in rows)
    truncated = sum(row.get("finish_reason") == "length" for row in rows)
    observations = [f"Observed {no_parse} NO_PARSE and {empty} EMPTY responses; {truncated} ended at the generation length limit."]
    max_tokens = cfg.get("generation", {}).get("max_new_tokens")
    if max_tokens != 32768:
        observations.append(f"This run used max_new_tokens={max_tokens}; the Qwen evaluation material uses 32768.")
    backend = cfg.get("execution", {}).get("backend")
    if backend and backend != "vllm":
        observations.append(f"This run used {backend}; the pinned Qwen reference script uses vLLM.")
    observations.append("These are observable protocol/runtime differences, not proof of causation.")
    text = " ".join(observations)
    if len(text) > 1000:
        raise AssertionError("Gap diagnosis exceeded 1000 characters")
    return text


def _legacy_render(run_dir: Path, output: Path) -> None:
    summary, environment, backend, manifest, cfg, rows, subjects = _load_complete_run(run_dir)
    model = cfg.get("model", {})
    generation = cfg.get("generation", {})
    image = cfg.get("image", {})
    execution = cfg.get("execution", {})
    cfg_path = _config_path(run_dir, manifest)
    hardware_path = _hardware_path(run_dir, manifest)
    hardware = _read_yaml(hardware_path) if hardware_path else {}
    by_subject = {
        subject: [row for row in rows if row.get("subject") == subject]
        for subject in subjects
    }
    actual_accuracy = sum(row.get("correct") is True for row in rows) / 900
    delta = actual_accuracy * 100 - OFFICIAL_SCORE
    timing = (
        f"evaluation loop {_seconds(summary.get('evaluation_seconds_this_invocation'))}; "
        f"model load {_seconds(summary.get('model_load_seconds'))}; "
        f"whole evaluation invocation {_seconds(summary.get('total_seconds_this_invocation'))}; "
        f"download verification (separate) {_download_text()}"
    )
    packages = environment.get("packages", {})
    backend_name = execution.get("backend", "transformers")
    backend_version = (backend.get("vllm", {}).get("version", "미제공") if backend_name == "vllm"
                       else packages.get("transformers", "미제공"))
    peak_vram = (
        f"reserved {_bytes_gib(summary.get('peak_vram_reserved_bytes'))}; "
        f"allocated {_bytes_gib(summary.get('peak_vram_allocated_bytes'))}"
    )
    if backend_name == "vllm":
        peak_vram += ("; worker allocator peak 미계측; 최대 batch 후 device snapshot "
                      f"{_bytes_gib(summary.get('max_observed_device_memory_used_bytes'))} (true peak 아님)")
    peak_ram = _bytes_gib(summary.get("peak_ram_rss_bytes"))
    if backend_name == "vllm":
        peak_ram += " (parent process만; worker RSS 제외)"
    command = _command(run_dir, manifest, cfg_path)
    rows_md = "\n".join(
        f"| {index} | {subject} | 30 | {_percent(sum(row.get('correct') is True for row in by_subject[subject]) / 30)} |"
        for index, subject in enumerate(subjects, 1)
    )
    lock_name = "requirements-vllm.lock" if backend_name == "vllm" else "requirements-eval.lock"
    package_link = f"[env/{lock_name}](../env/{lock_name})"
    report = f'''# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 미제공
- **팀원**: 미제공
- **작성일**: {date.today().isoformat()}
- **재현 커맨드**: `{_command_header(run_dir, manifest)}`

Run status: **COMPLETE900**. Independent audit: 900 unique completed rows, 30 subjects × 30 rows, and macro arithmetic `{sum(row.get("correct") is True for row in rows)}/900`.

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `{model.get('id', '미제공')}` (`{model.get('revision', '미제공')}`), dtype=`{model.get('dtype', '미제공')}` |
| 추론 백엔드 | `{backend_name}` `{backend_version}`, batch `{execution.get('batch_size')}`, attention `{execution.get('attention')}` / `{execution.get('sdpa_kernel')}` |
| 사용 GPU | {_gpu_text(environment)} |
| 실측 peak VRAM | {peak_vram} |
| 실측 peak RAM (process RSS) | {peak_ram} |
| 사전 환경 검사 | {_preflight_text(run_dir)} |
| 총 소요 시간 | {timing} |
| 의존성 | {package_link}; {_package_text(environment)} |
| 하드웨어 profile | `{hardware.get('name', hardware_path.name if hardware_path else '미제공')}` |
| 실행 커맨드 | 아래 code block |

실행 커맨드:

```bash
{command}
```

## 2. 프롬프트

실제 P0 source template 전문:

{_prompt_source(run_dir)}

Hint policy: non-empty sanitized source `hint` is rendered as `Hint: {{hint}}\\n` before `Question`; absent or blank hints render nothing. The open-ended counterpart keeps that hint policy, emits `Question: {{question}}`, and omits options and the MCQ selection instruction. Gold answer/explanation fields are excluded from inference input.

- **출처**: Qwen3-VL official `evaluation/mmmu/run_mmmu.py` `build_mmmu_prompt`, pinned in `third_party/README.md`; repository templates are the hashed P0 protocol inputs.
- **선택 이유**: official MMMU option order and instruction are retained; image content is supplied before one final text content item.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | {str(generation.get('do_sample', '미제공')).lower()} |
| `temperature` | {generation.get('temperature', '미제공')} |
| `top_p` | {generation.get('top_p', '미제공')} |
| `top_k` | {generation.get('top_k', '미제공')} |
| `repetition_penalty` | {generation.get('repetition_penalty', '미제공')} |
| `presence_penalty` | {generation.get('presence_penalty', '미제공')} (generated-only; backend record: {backend.get('presence_penalty', {}).get('implementation', '미제공')}) |
| `seed` | master `{generation.get('seed', '미제공')}`, policy `{generation.get('seed_policy', '미제공')}` |

- **출처**: Qwen3-VL repository `README.md`의 Evaluation Reproduction recipe; seed policy is the team protocol and is recorded separately from the reference script seed.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | {generation.get('max_new_tokens', '미제공')} |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | `{image.get('min_pixels', '미제공')}` / `{image.get('max_pixels', '미제공')}`, resize owner `{image.get('resize_owner', '미제공')}` |

**선택 근거**: values are the resolved run protocol. The generation budget and pixel bounds are engineering settings; this report does not present them as official answer values.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: `third_party/mmmu/eval_utils.py`의 official open parser/evaluator와 local exact MCQ heuristic (`mmmu-official-no-random-v1`).
- 동작 방식 요약: MCQ는 bracketed label, separated label, then option-content rules를 적용하고 random fallback 없이 `NO_PARSE`를 기록한다. Empty responses are `EMPTY` and count as incorrect. Open responses use the official normalization/evaluator.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
{rows_md}
| | **Overall (macro avg)** | **900** | **{_percent(actual_accuracy)}** |

계산식: `Overall = mean(30개 과목 accuracy) = correct/900 = {sum(row.get("correct") is True for row in rows)}/900`.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 수업이 제시한 공식 비교값 | 67.4 |
| 우리 재현 결과 | {actual_accuracy * 100:.2f} |
| 차이 (Δ) | {delta:+.2f} percentage points |

## 7. 격차 분석

{_gap_analysis(rows, cfg)}

## 8. 기타 특이사항 / 한계 (Optional)

- Report values are read from the external run's `summary.json`, `environment.json`, `backend.json`, and resolved protocol; no benchmark row content is reproduced here.
- Download verification timing is read separately from manifests when available; it is not merged into model-load or evaluation-loop timing.
- This report records aggregate observable evidence only. Detailed failure-case analysis, learning data, and improvement planning remain for user review.
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


_DYNAMIC = re.compile(
    r"(<!-- REPORT_DYNAMIC:BEGIN -->\n)(.*?)(<!-- REPORT_DYNAMIC:END -->)", re.DOTALL
)
_RUNTIME = re.compile(
    r"(<!-- REPORT_RUNTIME:BEGIN -->\n)(.*?)(<!-- REPORT_RUNTIME:END -->)", re.DOTALL
)
_STATUS = re.compile(
    r"(<!-- REPORT_STATUS:BEGIN -->\n)(.*?)(<!-- REPORT_STATUS:END -->)", re.DOTALL
)


def _sampled_device_memory(run_dir: Path, manifest: dict[str, Any], backend_name: str) -> str:
    if backend_name != "vllm":
        return "vLLM worker 장치 메모리 샘플 대상 아님"
    identity = manifest.get("identity", {})
    job_id = identity.get("job_id") if isinstance(identity, dict) else None
    role = identity.get("run_role") if isinstance(identity, dict) else None
    if not isinstance(job_id, str) or not isinstance(role, str):
        return "미측정: run manifest에 job_id/run_role이 없어 job-level device memory를 연결할 수 없음"
    job_path = run_dir.parent.parent / "jobs" / f"{job_id}.json"
    if not job_path.is_file():
        return "미측정: 회수된 job manifest가 없어 vLLM worker allocator 밖 장치 메모리를 확인할 수 없음"
    job = _read_json(job_path)
    stages = job.get("stages", [])
    matches = [stage for stage in stages if isinstance(stage, dict)
               and stage.get("run_role", stage.get("stage")) == role]
    values = [float(stage["peak_observed_device_memory_used_bytes"]) for stage in matches
              if isinstance(stage.get("peak_observed_device_memory_used_bytes"), (int, float))]
    if not values:
        return "미측정: 이 평가의 900-run device memory sampling record가 없음"
    stage = next(stage for stage in matches if stage.get("peak_observed_device_memory_used_bytes") == max(values))
    interval = stage.get("gpu_memory_sample_interval_ms", "미제공")
    return (
        f"장치 전체 사용량의 관측 최대 {_bytes_gib(max(values))} ({interval} ms 간격). "
        "vLLM allocator의 정확한 peak는 미측정; driver·다른 프로세스 사용량이 포함될 수 있음"
    )


def _runtime_metrics(summary: dict[str, Any], environment: dict[str, Any], backend: dict[str, Any],
                     manifest: dict[str, Any], cfg: dict[str, Any], run_dir: Path) -> str:
    """Return the complete authored section-1 Markdown table."""
    model = cfg.get("model", {})
    dataset = cfg.get("dataset", {})
    execution = cfg.get("execution", {})
    backend_name = execution.get("backend", "미제공")
    version = (backend.get("vllm", {}).get("version", "미제공") if backend_name == "vllm"
               else environment.get("packages", {}).get("transformers", "미제공"))
    device_memory = _sampled_device_memory(run_dir, manifest, backend_name)
    if device_memory.startswith("미측정"):
        observed = summary.get("max_observed_device_memory_used_bytes")
        if isinstance(observed, (int, float)):
            device_memory = (
                f"{_bytes_gib(observed)}; summary의 observed device-memory snapshot 최대치 "
                "(allocator peak가 아님)"
            )
    timing = (
        f"평가 {_seconds(summary.get('evaluation_seconds_this_invocation'))}; "
        f"모델 로드 {_seconds(summary.get('model_load_seconds'))}; "
        f"전체 호출 {_seconds(summary.get('total_seconds_this_invocation'))}"
    )
    packages = environment.get("packages", {})
    lock = "requirements-vllm.lock" if backend_name == "vllm" else "requirements-eval.lock"
    return (
        "| 항목 | 값 |\n"
        "|---|---|\n"
        f"| 모델 checkpoint | `{model.get('id', '미제공')}`, revision `{model.get('revision', '미제공')}` |\n"
        f"| 모델 정밀도 | `{model.get('dtype', '미제공')}`, 비양자화. Processor·Tokenizer도 모델과 동일 revision |\n"
        f"| 평가 데이터 | `{dataset.get('id', '미제공')}`, revision `{dataset.get('revision', '미제공')}`, `{dataset.get('split', '미제공')}` {dataset.get('expected_total', '미제공')}문항 |\n"
        f"| 추론 백엔드 | `{backend_name}` `{version}`, 최대 {execution.get('batch_size', '미제공')}개 요청의 연속 배치 |\n"
        f"| 사용 GPU | {_gpu_text(environment)} |\n"
        f"| 실측 peak VRAM | {device_memory} |\n"
        f"| 총 소요 시간 | {timing} (설치·다운로드 제외) |\n"
        f"| 주요 환경 | {environment.get('platform', {}).get('system', '미제공')}, Python {environment.get('python', '미제공')}, PyTorch {packages.get('torch', '미제공')}, Transformers {packages.get('transformers', '미제공')} |\n"
        f"| 의존성 | [env/{lock}](../env/{lock}) |\n"
    )


def _validate_authored_v8_config(cfg: dict[str, Any], scoring: dict[str, Any] | None) -> None:
    """Reject a run that would make the authored v8 static sections inaccurate."""
    v8_path = ROOT / "configs/eval/mmmu_val_v8.yaml"
    if not v8_path.is_file():
        return
    v8 = _read_yaml(v8_path)
    for section in ("model", "dataset", "generation", "image", "execution"):
        if cfg.get(section) != v8.get(section):
            raise ValueError(f"Authored v8 report requires matching resolved {section} settings")
    if cfg.get("prompt_policy") != v8.get("prompt_policy"):
        raise ValueError("Authored v8 report requires the P0 prompt policy")
    parser = (scoring or cfg).get("parser")
    if parser != v8.get("parser"):
        raise ValueError("Authored v8 report requires team-final-answer-v8 scoring")


def _status_metrics(summary: dict[str, Any], manifest: dict[str, Any], run_dir: Path) -> str:
    del summary, manifest, run_dir
    return "- 보고한 점수와 시간·메모리는 검증된 동일 실행의 기록이다.\n"


def _dynamic_results(rows: list[dict[str, Any]], subjects: list[str], cfg: dict[str, Any]) -> str:
    by_subject = {subject: [row for row in rows if row.get("subject") == subject] for subject in subjects}
    correct = sum(row.get("correct") is True for row in rows)
    accuracy = correct / 900
    table = "\n".join(
        f"| {index} | {subject} | 30 | {_percent(sum(row.get('correct') is True for row in by_subject[subject]) / 30)} |"
        for index, subject in enumerate(subjects, 1)
    )
    no_parse = sum(row.get("parse_status") == "NO_PARSE" for row in rows)
    length = sum(row.get("finish_reason") == "length" for row in rows)
    overlap = sum(row.get("parse_status") == "NO_PARSE" and row.get("finish_reason") == "length"
                  for row in rows)
    diagnosis = (
        "Qwen 공개 평가 코드는 규칙으로 답을 추출하지 못하면 LLM 판정을 사용할 수 있지만, "
        "우리는 고정된 규칙만 사용한다. 따라서 자유로운 표현이나 답변 내 충돌을 처리하는 기준이 다르다. "
        "[공식 채점 방식](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/README.md#custom-evaluation-logic)\n\n"
        f"저장 응답에서 NO_PARSE는 {no_parse}개, 출력 상한 종료는 {length}개이고 두 조건의 겹침은 {overlap}개다. "
        "따라서 독립적인 손실로 합산할 수 없다. 문항 ID별 seed 정책과 추론 환경도 공식 실행과 동일하지 않다. "
        "이미지 입력을 확대했어도 추론 오류나 반복 생성이 해결된다는 보장은 없다. "
        "공식 실행의 문항별 원본 응답을 확보하지 못했으므로 각 요인의 점수 기여도는 확정하지 않는다. "
        "추출 실패를 모두 모델 오답이나 모두 파서 오류로 해석하지 않는다."
    )
    if len(diagnosis) > 1000:
        raise AssertionError("Gap diagnosis exceeded 1000 characters")
    return f'''## 5. 결과

| No. | Subject | Data Num | Acc |
|---:|---|---:|---:|
{table}
|  | **Overall (macro avg)** | **900** | **{_percent(accuracy)}** |

계산식: `Overall = mean(30개 과목 accuracy) = {correct}/900 × 100 = {accuracy * 100:.2f}%`.
종합 점수는 반올림 전 과목별 정확도로 계산했다. 전체 900문항을 평가했으며 누락·시스템 오류는 없다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| 공식 — 과제에서 제시한 Qwen3-VL Technical Report 수치 | 67.4% |
| 우리 재현 결과 | {accuracy * 100:.2f}% |
| 차이 (우리 − 공식) | **{accuracy * 100 - OFFICIAL_SCORE:+.2f}%p** |

## 7. 격차 분석

{diagnosis}
'''


def render(run_dir: Path, output: Path, scored_dir: Path | None = None) -> None:
    """Replace only the marked metrics in the authored Korean submission report."""
    summary, environment, backend, manifest, cfg, rows, subjects = _load_complete_run(run_dir)
    scoring_note = ""
    scoring: dict[str, Any] | None = None
    if scored_dir is not None:
        from scripts.validate_rescore import validate
        validate(run_dir, scored_dir)
        rows = [_read_json(path) for path in sorted((scored_dir / "samples").glob("*.json"))]
        scoring = _read_json(scored_dir / "scorer_source.json")
        scoring_note = (
            "- 보고한 점수는 저장된 900개 응답을 변경하지 않고 §4의 최종 채점 규칙으로 다시 평가한 결과다. "
            "시간·메모리는 해당 GPU 추론 실행의 측정값이다.\n"
        )
    _validate_authored_v8_config(cfg, scoring)
    template = (ROOT / "reports/mmmu_baseline.md").read_text(encoding="utf-8")
    report, count = _DYNAMIC.subn(
        lambda match: match.group(1) + _dynamic_results(rows, subjects, cfg) + match.group(3), template
    )
    if count != 1:
        raise ValueError("reports/mmmu_baseline.md must contain exactly one dynamic report marker")
    report, count = _RUNTIME.subn(
        lambda match: match.group(1) + _runtime_metrics(summary, environment, backend, manifest, cfg, run_dir) + match.group(3), report
    )
    if count != 1:
        raise ValueError("reports/mmmu_baseline.md must contain exactly one runtime report marker")
    report, count = _STATUS.subn(
        lambda match: match.group(1) + (scoring_note or _status_metrics(summary, manifest, run_dir)) + match.group(3), report
    )
    if count != 1:
        raise ValueError("reports/mmmu_baseline.md must contain exactly one status report marker")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mmmu_baseline.md")
    parser.add_argument("--scored-dir", type=Path, help="verified CPU rescore overlay; preserve original inference metrics")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    render(args.run_dir.expanduser().resolve(), output,
           args.scored_dir.expanduser().resolve() if args.scored_dir else None)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
