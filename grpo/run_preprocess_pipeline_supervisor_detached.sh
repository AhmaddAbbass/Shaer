#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  echo "usage: $0 <run_dir> [judge_workers]" >&2
  exit 1
fi
JUDGE_WORKERS="${2:-8}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
LOG_PATH="$RUN_DIR/preprocess_supervisor.log"
PID_PATH="$RUN_DIR/preprocess_supervisor.pid"
setsid ../sft/.venv/bin/python -u preprocess_pipeline_supervisor.py --run-dir "$RUN_DIR" --judge-workers "$JUDGE_WORKERS" >> "$LOG_PATH" 2>&1 < /dev/null &
echo $! > "$PID_PATH"
echo "supervisor_pid=$(cat "$PID_PATH")"
echo "supervisor_log=$LOG_PATH"
