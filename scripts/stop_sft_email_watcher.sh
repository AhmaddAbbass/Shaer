#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WATCHER_PID_FILE="${WATCHER_PID_FILE:-$ROOT_DIR/logs/sft_email_watcher.pid}"

if [[ ! -f "$WATCHER_PID_FILE" ]]; then
  echo "Watcher PID file not found: $WATCHER_PID_FILE"
  exit 0
fi

pid="$(cat "$WATCHER_PID_FILE" 2>/dev/null || true)"
if [[ -z "$pid" ]]; then
  echo "PID file is empty: $WATCHER_PID_FILE"
  rm -f "$WATCHER_PID_FILE"
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped watcher PID $pid"
else
  echo "Watcher process not running (PID $pid)"
fi

rm -f "$WATCHER_PID_FILE"
