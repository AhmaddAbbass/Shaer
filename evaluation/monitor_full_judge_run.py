#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from judge_dataset_registry import DATASET_ORDER


@dataclass
class WorkerProgress:
    worker_index: int
    worker_count: int
    processed: int
    completed: int
    failures: int
    total_cost_usd: float
    elapsed_sec: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously monitor a detached full judge run and mirror status into repo logs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="evaluation/outputs/full_judge_run/logs")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status_json = output_dir / "status.json"
    status_md = output_dir / "status.md"
    events_jsonl = output_dir / "events.jsonl"
    heartbeat_log = output_dir / "heartbeat.log"

    last_signature = ""
    while True:
        snapshot = collect_snapshot(run_dir)
        signature = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
        if signature != last_signature:
            status_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            status_md.write_text(render_markdown(snapshot) + "\n", encoding="utf-8")
            with events_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ts": time.time(), "snapshot": snapshot}, ensure_ascii=False) + "\n")
            append_heartbeat(heartbeat_log, snapshot)
            last_signature = signature
        if snapshot.get("run_status") in {"completed", "failed", "stopped"}:
            return 0
        time.sleep(float(args.interval_seconds))


def collect_snapshot(run_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "run_dir": str(run_dir),
        "captured_at_unix": time.time(),
        "datasets": {},
        "current_dataset": "",
        "run_status": "running",
    }
    completed_datasets = 0
    any_running = False
    any_failed = False

    for dataset_key in DATASET_ORDER:
        dataset_dir = run_dir / dataset_key
        if not dataset_dir.exists():
            snapshot["datasets"][dataset_key] = {"status": "pending"}
            continue

        input_summary_path = dataset_dir / "input_rows.summary.json"
        merged_summary_path = dataset_dir / "scored_rows.summary.json"
        worker_dir = dataset_dir / "workers"
        expected_rows = read_json_field(input_summary_path, "rows", 0)
        worker_progress = read_worker_progress(worker_dir)
        completed_rows = sum(item.completed for item in worker_progress)
        failures = sum(item.failures for item in worker_progress)
        total_cost = sum(item.total_cost_usd for item in worker_progress)
        avg_elapsed = safe_mean([item.elapsed_sec for item in worker_progress])
        rows_per_sec = completed_rows / avg_elapsed if avg_elapsed > 0 else 0.0
        eta_sec = ((expected_rows - completed_rows) / rows_per_sec) if rows_per_sec > 0 and expected_rows > completed_rows else 0.0

        if merged_summary_path.exists():
            status = "completed"
            completed_datasets += 1
        elif worker_progress:
            status = "failed" if failures > 0 else "running"
            any_running = status == "running"
            any_failed = any_failed or status == "failed"
            if not snapshot["current_dataset"] and status == "running":
                snapshot["current_dataset"] = dataset_key
        else:
            status = "materializing" if input_summary_path.exists() else "pending"

        snapshot["datasets"][dataset_key] = {
            "status": status,
            "expected_rows": expected_rows,
            "completed_rows": completed_rows,
            "failures": failures,
            "workers": [item.__dict__ for item in worker_progress],
            "total_cost_usd": round(total_cost, 6),
            "avg_elapsed_sec": avg_elapsed,
            "rows_per_sec": rows_per_sec,
            "eta_sec": eta_sec,
            "merged_summary_exists": merged_summary_path.exists(),
        }

    if any_failed:
        snapshot["run_status"] = "failed"
    elif completed_datasets == len(DATASET_ORDER):
        snapshot["run_status"] = "completed"
    elif any_running:
        snapshot["run_status"] = "running"
    else:
        snapshot["run_status"] = "stopped"
    return snapshot


def read_worker_progress(worker_dir: Path) -> list[WorkerProgress]:
    items: list[WorkerProgress] = []
    if not worker_dir.exists():
        return items
    for path in sorted(worker_dir.glob("worker_*.progress.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                WorkerProgress(
                    worker_index=int(data.get("worker_index") or 0),
                    worker_count=int(data.get("worker_count") or 0),
                    processed=int(data.get("processed") or 0),
                    completed=int(data.get("completed") or 0),
                    failures=int(data.get("failures") or 0),
                    total_cost_usd=float(data.get("total_cost_usd") or 0.0),
                    elapsed_sec=float(data.get("elapsed_sec") or 0.0),
                    status=str(data.get("status") or ""),
                )
            )
        except Exception:
            continue
    return items


def read_json_field(path: Path, field: str, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(field, default)
    except Exception:
        return default


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = []
    lines.append("# Full Judge Run Status")
    lines.append("")
    lines.append(f"- Run dir: `{snapshot['run_dir']}`")
    lines.append(f"- Status: `{snapshot['run_status']}`")
    lines.append(f"- Current dataset: `{snapshot.get('current_dataset') or 'none'}`")
    lines.append(f"- Captured at unix: `{snapshot['captured_at_unix']}`")
    lines.append("")
    lines.append("| Dataset | Status | Completed / Expected | Failures | Cost ($) | Rows/sec | ETA |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset_key in DATASET_ORDER:
        data = snapshot["datasets"].get(dataset_key, {})
        lines.append(
            "| {dataset} | {status} | {completed}/{expected} | {failures} | {cost:.6f} | {rps:.4f} | {eta} |".format(
                dataset=dataset_key,
                status=data.get("status", "pending"),
                completed=data.get("completed_rows", 0),
                expected=data.get("expected_rows", 0),
                failures=data.get("failures", 0),
                cost=float(data.get("total_cost_usd", 0.0) or 0.0),
                rps=float(data.get("rows_per_sec", 0.0) or 0.0),
                eta=format_eta(float(data.get("eta_sec", 0.0) or 0.0)),
            )
        )
    return "\n".join(lines)


def append_heartbeat(path: Path, snapshot: dict[str, Any]) -> None:
    current = snapshot.get("current_dataset") or "none"
    current_data = snapshot["datasets"].get(current, {}) if current != "none" else {}
    line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"run_status={snapshot['run_status']} "
        f"current_dataset={current} "
        f"completed={current_data.get('completed_rows', 0)}/{current_data.get('expected_rows', 0)} "
        f"failures={current_data.get('failures', 0)} "
        f"cost_usd={float(current_data.get('total_cost_usd', 0.0) or 0.0):.6f} "
        f"eta={format_eta(float(current_data.get('eta_sec', 0.0) or 0.0))}"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    minutes = int(seconds // 60)
    hours = minutes // 60
    if hours > 0:
        return f"{hours}h{minutes % 60:02d}m"
    return f"{minutes}m"


if __name__ == "__main__":
    raise SystemExit(main())
