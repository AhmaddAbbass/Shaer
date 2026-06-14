#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/root/workspace/Shaer/sft/.venv/bin/python"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_NAME [WORKERS_MANIFEST.json] [extra worker args...]" >&2
  exit 1
fi

RUN_NAME="$1"
shift

DEFAULT_MANIFEST="$ROOT_DIR/outputs/$RUN_NAME/workers_manifest.json"
if [[ $# -gt 0 && "$1" == *.json ]]; then
  MANIFEST_PATH="$1"
  shift
else
  MANIFEST_PATH="$DEFAULT_MANIFEST"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "missing workers manifest at $MANIFEST_PATH" >&2
  exit 1
fi

WORKER_COUNT="${WORKER_COUNT:-10}"
SYNC_EVERY="${SYNC_EVERY:-25}"
LOG_DIR="$ROOT_DIR/outputs/$RUN_NAME/launch_logs"
mkdir -p "$LOG_DIR"

for ((worker_id=0; worker_id<WORKER_COUNT; worker_id++)); do
  worker_tag="$(printf '%02d' "$worker_id")"
  log_path="$LOG_DIR/worker_${worker_tag}.log"
  pid_path="$LOG_DIR/worker_${worker_tag}.pid"
  cmd=(
    "$PYTHON"
    -u
    "$ROOT_DIR/regenerate_descriptions.py"
    --run-name "$RUN_NAME"
    --workers-manifest "$MANIFEST_PATH"
    --worker-id "$worker_id"
    --resume
    --sync-every "$SYNC_EVERY"
  )
  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi
  setsid "${cmd[@]}" >"$log_path" 2>&1 < /dev/null &
  pid="$!"
  echo "$pid" >"$pid_path"
  echo "worker=$worker_tag pid=$pid log=$log_path"
done
