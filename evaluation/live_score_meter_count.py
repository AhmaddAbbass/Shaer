#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics import score_generation_row


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_processed(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            generation_id = str(row.get("generation_id") or "").strip()
            if generation_id:
                processed.add(generation_id)
    return processed


def iter_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously score generated rows with meter/count metrics.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--expected-rows", type=int, default=17405)
    parser.add_argument("--stop-when-generation-complete", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_path = run_dir / "generations.jsonl"
    output_path = run_dir / "meter_count_scored.jsonl"
    status_path = run_dir / "meter_count_status.json"
    failure_path = run_dir / "meter_count_failures.jsonl"
    gen_status_path = run_dir / "status.json"
    processed = load_processed(output_path)
    started = time.time()

    while True:
        did = 0
        for row in iter_rows(input_path):
            generation_id = str(row.get("generation_id") or "").strip()
            if not generation_id or generation_id in processed:
                continue
            try:
                scored = dict(row)
                scored.update(score_generation_row(row))
                scored["meter_count_scored_at_utc"] = utc_now_iso()
                append_jsonl(output_path, scored)
                processed.add(generation_id)
                did += 1
            except Exception as exc:
                append_jsonl(
                    failure_path,
                    {
                        "generation_id": generation_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp_utc": utc_now_iso(),
                    },
                )

        gen_status = {}
        if gen_status_path.exists():
            try:
                gen_status = json.loads(gen_status_path.read_text(encoding="utf-8"))
            except Exception:
                gen_status = {}
        atomic_write_json(
            status_path,
            {
                "status": "running",
                "scored_rows": len(processed),
                "expected_rows": int(args.expected_rows),
                "remaining_rows": max(0, int(args.expected_rows) - len(processed)),
                "last_loop_scored": did,
                "generation_status": gen_status.get("status", ""),
                "generation_healthy_rows": gen_status.get("healthy_generations", None),
                "updated_at_utc": utc_now_iso(),
                "elapsed_seconds": time.time() - started,
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "meter_model_id": os.getenv("METER_MODEL_ID", ""),
            },
        )

        if (
            args.stop_when_generation_complete
            and gen_status.get("status") == "completed"
            and len(processed) >= int(args.expected_rows)
        ):
            atomic_write_json(
                status_path,
                {
                    "status": "completed",
                    "scored_rows": len(processed),
                    "expected_rows": int(args.expected_rows),
                    "completed_at_utc": utc_now_iso(),
                    "elapsed_seconds": time.time() - started,
                },
            )
            return 0

        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
