#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/sanity_check
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG="outputs/sanity_check/sanity_${RUN_TS}.log"
PYTHON="./.venv/bin/python"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

nohup "$PYTHON" -u sanity_check.py > "$LOG" 2>&1 &
PID=$!

echo "sanity_check started"
echo "PID=$PID"
echo "LOG=$LOG"
echo "$PID" > outputs/sanity_check/latest.pid
echo "$LOG" > outputs/sanity_check/latest.log
