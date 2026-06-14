#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preprocess_meter_count_common import (
    atomic_write_json,
    ensure_dir,
    event_log_path,
    expected_shard_ids,
    generated_path,
    load_dotenv_if_present,
    load_manifest,
    load_run_config,
    read_jsonl,
    read_json_or_none,
    run_counts,
    scored_path,
    utc_now_iso,
)


def parse_ts(value: str) -> float | None:
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def load_events(run_dir: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    events_dir = run_dir / "events"
    if not events_dir.exists():
        return rows
    for path in sorted(events_dir.glob(f"{prefix}*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def parallel_rate_from_events(events: list[dict[str, Any]], *, id_key: str, count_key: str) -> dict[str, Any]:
    counts = Counter()
    active_seconds = defaultdict(float)
    first_ts = None
    last_ts = None
    for event in events:
        worker = str(event.get(id_key) or "unknown")
        counts[worker] += int(event.get(count_key) or 0)
        active_seconds[worker] += float(event.get("seconds") or event.get("runtime_seconds") or 0.0)
        ts = parse_ts(str(event.get("timestamp") or ""))
        if ts is not None:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
    total = sum(counts.values())
    max_active = max(active_seconds.values()) if active_seconds else 0.0
    wall_seconds = max(0.0, (last_ts - first_ts) if first_ts is not None and last_ts is not None else 0.0)
    return {
        "total_count": int(total),
        "count_by_id": {key: int(value) for key, value in sorted(counts.items())},
        "active_seconds_by_id": {key: float(value) for key, value in sorted(active_seconds.items())},
        "max_parallel_active_seconds": float(max_active),
        "observed_wall_event_span_seconds": float(wall_seconds),
        "parallel_active_rate": float(total / max_active) if max_active > 0 else 0.0,
        "event_span_rate": float(total / wall_seconds) if wall_seconds > 0 else 0.0,
    }


def summarize_run(run_dir: Path, full_train_rows: int) -> dict[str, Any]:
    cfg = load_run_config(run_dir)
    manifest = load_manifest(run_dir)
    counts = run_counts(run_dir)
    generated_rows_by_shard = {}
    for shard_id in expected_shard_ids(run_dir):
        generated_rows_by_shard[str(shard_id)] = sum(1 for _ in (run_dir / "generated" / f"shard_{shard_id:02d}").glob("*.json"))

    scored_rows = []
    difficulty = Counter()
    candidate_total = 0
    for path in sorted((run_dir / "scored").glob("*.json")) if (run_dir / "scored").exists() else []:
        row = read_json_or_none(path)
        if row is None:
            continue
        scored_rows.append(row)
        difficulty[str(row.get("difficulty") or "")] += 1
        candidate_total += int(row.get("num_candidates_scored") or 0)

    generator_events = load_events(run_dir, "generator_")
    score_events = load_events(run_dir, "score_worker_")
    gen_rate = parallel_rate_from_events(generator_events, id_key="generator_id", count_key="candidates")
    score_rate = parallel_rate_from_events(score_events, id_key="worker_id", count_key="candidates")

    num_candidates = int(cfg.get("num_candidates") or 0)
    full_candidates = int(full_train_rows) * num_candidates
    generator_rate = float(gen_rate["parallel_active_rate"])
    score_worker_rate = float(score_rate["parallel_active_rate"])
    bottleneck_rate = min(x for x in [generator_rate, score_worker_rate] if x > 0.0) if generator_rate > 0.0 or score_worker_rate > 0.0 else 0.0
    full_eta_seconds = float(1.15 * full_candidates / bottleneck_rate) if bottleneck_rate > 0 else None
    return {
        "timestamp": utc_now_iso(),
        "run_dir": str(run_dir),
        "dataset_id": cfg.get("dataset_id"),
        "source_split": cfg.get("source_split"),
        "num_candidates": num_candidates,
        "num_generator_shards": int(cfg.get("num_generator_shards") or 0),
        **counts,
        "generated_rows_by_shard": generated_rows_by_shard,
        "candidate_total_scored": int(candidate_total),
        "difficulty_counts": dict(sorted(difficulty.items())),
        "generator_rate": gen_rate,
        "score_worker_rate": score_rate,
        "full_train_rows_for_eta": int(full_train_rows),
        "full_candidate_count_for_eta": int(full_candidates),
        "bottleneck_candidates_per_second": bottleneck_rate,
        "full_eta_seconds": full_eta_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a meter/count preprocess run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--full-train-rows", type=int, default=109070)
    return parser.parse_args()


def print_brief(summary: dict[str, Any]) -> None:
    eta = summary["full_eta_seconds"]
    eta_text = "unknown" if eta is None else f"{eta:.1f}s ({eta / 3600.0:.2f}h)"
    print(
        " | ".join(
            [
                summary["timestamp"],
                f"rows manifest/gen/scored={summary['manifest_rows']}/{summary['generated_rows']}/{summary['scored_rows']}",
                f"scored_candidates={summary['candidate_total_scored']}",
                f"gen_rate={summary['generator_rate']['parallel_active_rate']:.3f} cand/s",
                f"score_rate={summary['score_worker_rate']['parallel_active_rate']:.3f} cand/s",
                f"eta={eta_text}",
            ]
        ),
        flush=True,
    )


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    run_dir = Path(args.run_dir)
    reports_dir = ensure_dir(run_dir / "reports")
    while True:
        summary = summarize_run(run_dir, full_train_rows=int(args.full_train_rows))
        atomic_write_json(reports_dir / "throughput_summary.json", summary)
        print_brief(summary)
        if not args.watch:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            break
        time.sleep(max(1.0, float(args.interval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
