#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="outputs/meaning_prompt_sweep"
mkdir -p "$OUT_ROOT"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG="$OUT_ROOT/meaning_prompt_sweep_${RUN_TS}.log"
PYTHON="./.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

setsid "$PYTHON" -u studies/11_sweep_meaning_prompts.py > "$LOG" 2>&1 < /dev/null &
PID=$!

echo "meaning_prompt_sweep started"
echo "PID=$PID"
echo "LOG=$LOG"
echo "$PID" > "$OUT_ROOT/latest.pid"
echo "$LOG" > "$OUT_ROOT/latest.log"
