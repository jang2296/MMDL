#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${MMDL_BOOTSTRAP_PYTHON:-python3.12}" -m mmdl.runtime.reproduce "$@"
