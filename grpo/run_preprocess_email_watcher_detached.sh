#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  echo "usage: $0 /abs/path/to/preprocess_run_dir" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_PATH="$RUN_DIR/preprocess_email_watcher.log"
PID_PATH="$RUN_DIR/preprocess_email_watcher.pid"

nohup ../sft/.venv/bin/python -u preprocess_email_watcher.py --run-dir "$RUN_DIR" > "$LOG_PATH" 2>&1 &
echo $! > "$PID_PATH"
echo "watcher_pid=$(cat "$PID_PATH")"
echo "watcher_log=$LOG_PATH"
