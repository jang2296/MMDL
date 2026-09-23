#!/usr/bin/env bash
# Predeclared structural diagnostics, not a validation-score selection contest.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
commit=${1:?Usage: controlled_eval.sh FULL_COMMIT JOB_ID}
job_id=${2:?Usage: controlled_eval.sh FULL_COMMIT JOB_ID}
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$job_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,50}$ ]] || exit 2
: "${MMDL_ARTIFACT_ROOT:?}" "${MMDL_DATA_ROOT:?}" "${MMDL_PYTHON:?}"
sample_ids=(validation_Accounting_1 validation_Biology_29 validation_Computer_Science_19
            validation_Chemistry_13 validation_Music_21 validation_Electronics_21)
samples=()
for sample_id in "${sample_ids[@]}"; do samples+=(--sample-id "$sample_id"); done
for arm in a b c; do
  bash scripts/eval.sh \
    --protocol "configs/eval/mmmu_val_control_${arm}_v2.yaml" \
    --hardware configs/hardware/rtx3090_24gb.yaml \
    --require-commit "$commit" --job-id "$job_id" \
    --run-id "${job_id}-control-${arm}" --mode smoke \
    --artifact-root "$MMDL_ARTIFACT_ROOT" \
    --public-root "$MMDL_ARTIFACT_ROOT/control-public" \
    --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --no-download "${samples[@]}"
done
# Deliberately stop here: inspect all three arms before starting the full 900.
