#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$root:$root/src${PYTHONPATH:+:$PYTHONPATH}"
export MMDL_TRAIN_PYTHON="${MMDL_TRAIN_PYTHON:-${MMDL_PYTHON:-python}}"
export MMDL_DATA_PYTHON="${MMDL_DATA_PYTHON:-}"
export MMDL_DATA_ROOT="${MMDL_DATA_ROOT:-$HOME/mmdl-data}"
export MMDL_ARTIFACT_ROOT="${MMDL_ARTIFACT_ROOT:-$HOME/mmdl-artifacts}"
exec "$MMDL_TRAIN_PYTHON" -m fine_tuning.error_note.submission_train "$@"
