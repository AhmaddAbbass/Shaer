#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <outer-log-or-run-dir>" >&2
  exit 2
fi

mkdir -p outputs/watchers
RUN_TS=$(date +"%Y%m%d_%H%M%S")
TARGET="$1"
WATCHER_LOG="outputs/watchers/watcher_${RUN_TS}.log"
PYTHON="./.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

setsid "$PYTHON" -u watcher.py "$TARGET" > "$WATCHER_LOG" 2>&1 < /dev/null &
WATCHER_PID=$!

echo "watcher started"
echo "TARGET=$TARGET"
echo "WATCHER_PID=$WATCHER_PID"
echo "WATCHER_LOG=$WATCHER_LOG"
echo "$WATCHER_PID" > outputs/watchers/latest.pid
echo "$WATCHER_LOG" > outputs/watchers/latest.log
printf '%s\n' "$TARGET" > outputs/watchers/latest.target
