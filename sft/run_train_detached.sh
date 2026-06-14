#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/train
RUN_TS=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="train_${RUN_TS}"
LOG="outputs/train/train_${RUN_TS}.log"
WATCHER_LOG="outputs/train/watcher_${RUN_TS}.log"
PLOTTER_LOG="outputs/train/plotter_${RUN_TS}.log"
PYTHON="./.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

setsid "$PYTHON" -u train_sft.py --mode train --run_name "$RUN_NAME" > "$LOG" 2>&1 < /dev/null &
PID=$!

echo "train_sft started"
echo "PID=$PID"
echo "LOG=$LOG"
echo "$PID" > outputs/train/latest.pid
echo "$LOG" > outputs/train/latest.log

setsid "$PYTHON" -u watcher.py "$LOG" > "$WATCHER_LOG" 2>&1 < /dev/null &
WATCHER_PID=$!
echo "WATCHER_PID=$WATCHER_PID"
echo "WATCHER_LOG=$WATCHER_LOG"
echo "$WATCHER_PID" > outputs/train/latest.watcher.pid
echo "$WATCHER_LOG" > outputs/train/latest.watcher.log

RUN_DIR="outputs/train/${RUN_NAME}"
for _ in $(seq 1 60); do
  if [[ -f "${RUN_DIR}/train.log" ]]; then
    break
  fi
  sleep 1
done

if [[ -n "${RUN_DIR}" && -d "${RUN_DIR}" ]]; then
  setsid "$PYTHON" -u plot_live_sft.py --run-dir "$RUN_DIR" --follow-chain --watch > "$PLOTTER_LOG" 2>&1 < /dev/null &
  PLOTTER_PID=$!
  echo "PLOTTER_PID=$PLOTTER_PID"
  echo "PLOTTER_LOG=$PLOTTER_LOG"
  echo "$PLOTTER_PID" > outputs/train/latest.plotter.pid
  echo "$PLOTTER_LOG" > outputs/train/latest.plotter.log
else
  echo "plotter not started: run dir not discovered in time" >&2
fi
