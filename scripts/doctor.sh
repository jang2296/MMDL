#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${MMDL_PYTHON:-python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m mmdl.runtime.environment "$@"
