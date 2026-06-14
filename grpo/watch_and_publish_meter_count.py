#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from preprocess_meter_count_common import (
    append_jsonl,
    atomic_write_json,
    ensure_dir,
    load_dotenv_if_present,
    load_run_config,
    read_json_or_none,
    run_counts,
    status_path,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a meter/count preprocess run and publish checkpoint snapshots to Hugging Face.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--every-rows", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def event_path(run_dir: Path) -> Path:
    return run_dir / "events" / "checkpoint_publisher.jsonl"


def state_path(run_dir: Path) -> Path:
    return status_path(run_dir, "checkpoint_publisher")


def next_checkpoint_rows(last_published_rows: int, every_rows: int, manifest_rows: int) -> int:
    if manifest_rows <= 0:
        return max(1, every_rows)
    if last_published_rows >= manifest_rows:
        return manifest_rows
    step = max(1, int(every_rows))
    return min(manifest_rows, ((int(last_published_rows) // step) + 1) * step)


def write_state(run_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(state_path(run_dir), payload)


def publish_snapshot(
    *,
    python_bin: str,
    run_dir: Path,
    repo_id: str,
    allow_incomplete: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        python_bin,
        str((Path(__file__).resolve().parent / "assemble_meter_count_dataset.py")),
        "--run-dir",
        str(run_dir),
        "--repo-id",
        repo_id,
        "--push",
    ]
    if allow_incomplete:
        cmd.append("--allow-incomplete")
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    run_dir = ensure_dir(args.run_dir)
    cfg = load_run_config(run_dir)
    manifest_rows = int(cfg.get("manifest_rows") or 0) or int(run_counts(run_dir)["manifest_rows"])
    every_rows = max(1, int(args.every_rows))
    poll_seconds = max(10, int(args.poll_seconds))
    state = read_json_or_none(state_path(run_dir)) or {}
    last_published_rows = int(state.get("last_published_rows") or 0)
    publish_count = int(state.get("publish_count") or 0)
    last_kind = str(state.get("last_publish_kind") or "")

    append_jsonl(
        event_path(run_dir),
        {
            "event": "watcher_started",
            "timestamp": utc_now_iso(),
            "repo_id": args.repo_id,
            "every_rows": every_rows,
            "poll_seconds": poll_seconds,
            "manifest_rows": manifest_rows,
            "resumed_last_published_rows": last_published_rows,
        },
    )

    while True:
        counts = run_counts(run_dir)
        scored_rows = int(counts["scored_rows"])
        generated_rows = int(counts["generated_rows"])
        manifest_rows = int(counts["manifest_rows"])
        final_ready = bool(counts["all_generators_done"]) and scored_rows >= manifest_rows > 0
        target_checkpoint = next_checkpoint_rows(last_published_rows, every_rows, manifest_rows)

        state_payload = {
            "status": "watching",
            "timestamp": utc_now_iso(),
            "repo_id": args.repo_id,
            "manifest_rows": manifest_rows,
            "generated_rows": generated_rows,
            "scored_rows": scored_rows,
            "last_published_rows": last_published_rows,
            "next_checkpoint_rows": target_checkpoint,
            "publish_count": publish_count,
            "last_publish_kind": last_kind,
            "final_ready": final_ready,
            "all_generators_done": bool(counts["all_generators_done"]),
        }
        write_state(run_dir, state_payload)

        should_publish_checkpoint = (
            not final_ready
            and manifest_rows > 0
            and target_checkpoint < manifest_rows
            and scored_rows >= target_checkpoint
            and scored_rows > last_published_rows
        )
        should_publish_final = final_ready and (last_published_rows < manifest_rows or last_kind != "final")

        if should_publish_checkpoint or should_publish_final:
            kind = "final" if should_publish_final else "checkpoint"
            allow_incomplete = kind != "final"
            append_jsonl(
                event_path(run_dir),
                {
                    "event": "publish_started",
                    "timestamp": utc_now_iso(),
                    "kind": kind,
                    "repo_id": args.repo_id,
                    "scored_rows_visible": scored_rows,
                    "generated_rows_visible": generated_rows,
                    "manifest_rows": manifest_rows,
                    "allow_incomplete": allow_incomplete,
                },
            )
            proc = publish_snapshot(
                python_bin=args.python_bin,
                run_dir=run_dir,
                repo_id=args.repo_id,
                allow_incomplete=allow_incomplete,
            )
            if proc.returncode == 0:
                publish_count += 1
                last_published_rows = manifest_rows if kind == "final" else scored_rows
                last_kind = kind
                append_jsonl(
                    event_path(run_dir),
                    {
                        "event": "publish_succeeded",
                        "timestamp": utc_now_iso(),
                        "kind": kind,
                        "repo_id": args.repo_id,
                        "published_rows": last_published_rows,
                        "publish_count": publish_count,
                    },
                )
                write_state(
                    run_dir,
                    {
                        "status": "published" if kind == "final" else "watching",
                        "timestamp": utc_now_iso(),
                        "repo_id": args.repo_id,
                        "manifest_rows": manifest_rows,
                        "generated_rows": generated_rows,
                        "scored_rows": scored_rows,
                        "last_published_rows": last_published_rows,
                        "next_checkpoint_rows": next_checkpoint_rows(last_published_rows, every_rows, manifest_rows),
                        "publish_count": publish_count,
                        "last_publish_kind": last_kind,
                        "final_ready": final_ready,
                        "all_generators_done": bool(counts["all_generators_done"]),
                    },
                )
                if kind == "final":
                    return 0
            else:
                append_jsonl(
                    event_path(run_dir),
                    {
                        "event": "publish_failed",
                        "timestamp": utc_now_iso(),
                        "kind": kind,
                        "repo_id": args.repo_id,
                        "returncode": int(proc.returncode),
                        "stdout_tail": proc.stdout[-4000:],
                        "stderr_tail": proc.stderr[-4000:],
                    },
                )
                write_state(
                    run_dir,
                    {
                        **state_payload,
                        "status": "publish_failed",
                        "last_publish_returncode": int(proc.returncode),
                    },
                )

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
