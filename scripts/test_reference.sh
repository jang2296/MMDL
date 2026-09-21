#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python_bin=${MMDL_PYTHON:-python}
"$python_bin" -m unittest discover -s tests/unit -v
"$python_bin" -m ruff check src tests scripts
"$python_bin" -m mypy --follow-imports=skip --ignore-missing-imports src
bash -n scripts/*.sh
