#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop a running full judge orchestrator after a dataset is completed.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_path)
    target_line = f"dataset={args.dataset_key} status=completed"
    printed_wait = False
    while True:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if target_line in text:
                os.kill(args.pid, signal.SIGTERM)
                print(f"stopped pid={args.pid} after {args.dataset_key} completion", flush=True)
                return 0
        if not printed_wait:
            print(f"watching {log_path} for '{target_line}'", flush=True)
            printed_wait = True
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
