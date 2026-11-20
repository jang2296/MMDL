#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$root:$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${MMDL_PYTHON:-python}" -c 'from mmdl.evaluation.cli import rescore_main; rescore_main()' "$@"
