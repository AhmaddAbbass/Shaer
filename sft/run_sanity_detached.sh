#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/sanity_check
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG="outputs/sanity_check/sanity_${RUN_TS}.log"
WATCHER_LOG="outputs/sanity_check/watcher_${RUN_TS}.log"
PYTHON="./.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

setsid "$PYTHON" -u sanity_check.py > "$LOG" 2>&1 < /dev/null &
PID=$!

echo "sanity_check started"
echo "PID=$PID"
echo "LOG=$LOG"
echo "$PID" > outputs/sanity_check/latest.pid
echo "$LOG" > outputs/sanity_check/latest.log

setsid "$PYTHON" -u watcher.py "$LOG" > "$WATCHER_LOG" 2>&1 < /dev/null &
WATCHER_PID=$!
echo "WATCHER_PID=$WATCHER_PID"
echo "WATCHER_LOG=$WATCHER_LOG"
echo "$WATCHER_PID" > outputs/sanity_check/latest.watcher.pid
echo "$WATCHER_LOG" > outputs/sanity_check/latest.watcher.log
