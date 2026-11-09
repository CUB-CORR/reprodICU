#!/usr/bin/env bash
# Runs build_db.py in a local Python environment, with a timestamped log.
# Usage: ./run.sh check | verify | prepare | build | concepts | summary   (see ./run.sh --help)
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

# 1) Python environment (tools/build_db/.venv, created on first run, refreshed when
#    requirements.txt changes). reprodICU itself is installed in editable mode from
#    this repository, so the build always uses the checked-out code.
if [ ! -x .venv/bin/python ] || [ requirements.txt -nt .venv/.installed ]; then
  echo ">> Creating / updating the Python environment (.venv)..."
  if command -v uv >/dev/null 2>&1; then
    [ -x .venv/bin/python ] || uv venv .venv --python 3.12 --quiet
    uv pip install --python .venv/bin/python --quiet -r requirements.txt -e "$REPO_ROOT"
  else
    PY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)"
    if [ -z "$PY" ]; then
      echo "!! Python >= 3.10 not found. Install uv (recommended), e.g.: brew install uv"
      exit 1
    fi
    [ -x .venv/bin/python ] || "$PY" -m venv .venv
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt -e "$REPO_ROOT"
  fi
  touch .venv/.installed
fi

# 2) Run, with a log in logs/
mkdir -p logs
LOG="logs/$(date +%Y%m%d_%H%M%S)_${1:-help}.log"
echo ">> Log: $PWD/$LOG"
set +e
PYTHONWARNINGS="ignore::SyntaxWarning" .venv/bin/python -u build_db.py "$@" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo ">> Done (exit code $STATUS), log: $PWD/$LOG"
exit $STATUS
