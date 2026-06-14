#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

WATCHER_TO_EMAIL="${WATCHER_TO_EMAIL:-abbassahmad050@gmail.com}"
WATCHER_LOG_FILE="${WATCHER_LOG_FILE:-$ROOT_DIR/logs/sft_email_watcher.log}"
WATCHER_PID_FILE="${WATCHER_PID_FILE:-$ROOT_DIR/logs/sft_email_watcher.pid}"

if [[ -z "${WATCHER_SMTP_EMAIL:-}" ]]; then
  echo "Missing WATCHER_SMTP_EMAIL in .env"
  exit 1
fi
if [[ -z "${WATCHER_SMTP_APP_PASSWORD:-}" ]]; then
  echo "Missing WATCHER_SMTP_APP_PASSWORD in .env"
  exit 1
fi

mkdir -p "$(dirname "$WATCHER_LOG_FILE")"

if [[ -f "$WATCHER_PID_FILE" ]]; then
  old_pid="$(cat "$WATCHER_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Watcher already running (PID $old_pid)."
    echo "Log: $WATCHER_LOG_FILE"
    exit 0
  fi
fi

setsid "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/watch_sft_email.py" \
  --to-email "$WATCHER_TO_EMAIL" \
  --smtp-email "$WATCHER_SMTP_EMAIL" \
  --smtp-app-password "$WATCHER_SMTP_APP_PASSWORD" \
  >> "$WATCHER_LOG_FILE" 2>&1 < /dev/null &

echo $! > "$WATCHER_PID_FILE"
echo "Watcher started with PID $(cat "$WATCHER_PID_FILE")"
echo "Tail logs: tail -f $WATCHER_LOG_FILE"
