#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  echo "usage: $0 <run_dir> [num_workers]" >&2
  exit 1
fi

NUM_WORKERS="${2:-4}"
PYTHON="../sft/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing python at $PYTHON" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/judge_worker_pids.txt"
LOG_FILE="$RUN_DIR/judge_worker_logs.txt"
: > "$PID_FILE"
: > "$LOG_FILE"

for ((i=0; i<NUM_WORKERS; i++)); do
  WORKER_ID=$(printf "judge%02d" "$i")
  WORKER_LOG="$RUN_DIR/${WORKER_ID}_detached.log"
  EXTRA_ARGS=()
  if [[ "$i" == "0" ]]; then
    EXTRA_ARGS+=(--publish)
  fi
  setsid "$PYTHON" -u judge_grpo_candidates.py --run-dir "$RUN_DIR" --worker-id "$WORKER_ID" "${EXTRA_ARGS[@]}" > "$WORKER_LOG" 2>&1 < /dev/null &
  PID="$!"
  echo "$PID" >> "$PID_FILE"
  echo "$WORKER_LOG" >> "$LOG_FILE"
  echo "started $WORKER_ID pid=$PID log=$WORKER_LOG"
done

