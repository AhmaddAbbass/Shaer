#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"
RUN_BASENAME="$(basename "$RUN_DIR")"
LOG_PATH="$ROOT_DIR/outputs/train/best_probe_meter_keeper_${RUN_BASENAME}.log"
PID_PATH="$ROOT_DIR/outputs/train/best_probe_meter_keeper_${RUN_BASENAME}.pid"

mkdir -p "$ROOT_DIR/outputs/train"

setsid "$ROOT_DIR/.venv/bin/python" -u "$ROOT_DIR/best_probe_meter_keeper.py" \
  --run-dir "$RUN_DIR" \
  --watch \
  >"$LOG_PATH" 2>&1 < /dev/null &

KEEPER_PID=$!
echo "$KEEPER_PID" >"$PID_PATH"
echo "Started best-probe-meter keeper for $RUN_DIR"
echo "PID: $KEEPER_PID"
echo "Log: $LOG_PATH"
