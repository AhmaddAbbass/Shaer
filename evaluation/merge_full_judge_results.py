#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from judge_common import read_jsonl
from judge_llm import DEFAULT_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-worker full judge outputs for one dataset.")
    parser.add_argument("--workers-dir", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers_dir = Path(args.workers_dir)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_ids: set[str] = set()
    for worker_file in sorted(workers_dir.glob("worker_*.jsonl")):
        for row in read_jsonl(worker_file):
            row_id = str(row.get("judge_row_id") or "")
            if not row_id:
                raise RuntimeError(f"Missing judge_row_id in {worker_file}")
            if row_id in seen_ids:
                raise RuntimeError(f"Duplicate judge_row_id detected: {row_id}")
            seen_ids.add(row_id)
            rows.append(row)

    if len(rows) != int(args.expected_rows):
        raise RuntimeError(f"Merged rows={len(rows)} but expected_rows={args.expected_rows}")

    rows.sort(key=lambda row: int(row.get("judge_row_index") or 0))
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(rows)
    summary["rows"] = len(rows)
    summary["expected_rows"] = int(args.expected_rows)
    summary["output_jsonl"] = str(output_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rows={len(rows)} output={output_path} summary={summary_path}")
    return 0


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_stats = {}
    total_cost = 0.0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for metric in DEFAULT_METRICS:
        scores = []
        for row in rows:
            value = row.get(metric)
            if isinstance(value, (int, float)) and float(value) > 0:
                scores.append(int(value))
        metric_stats[metric] = {
            "mean": safe_mean(scores),
            "median": safe_median(scores),
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "counts": dict(sorted(Counter(scores).items())),
            "scored_rows": len(scores),
            "missing_or_error_rows": len(rows) - len(scores),
        }
    for row in rows:
        total_cost += float(row.get("judge_total_cost_usd") or 0.0)
        total_latency += float(row.get("judge_total_latency_sec") or 0.0)
        total_prompt_tokens += int(row.get("judge_total_prompt_tokens") or 0)
        total_completion_tokens += int(row.get("judge_total_completion_tokens") or 0)
    return {
        "judge_model": str(rows[0].get("judge_model") or "") if rows else "",
        "judge_prompt_version": str(rows[0].get("judge_prompt_version") or "") if rows else "",
        "metric_stats": metric_stats,
        "total_cost_usd": total_cost,
        "avg_cost_usd_per_row": total_cost / len(rows) if rows else 0.0,
        "total_latency_sec": total_latency,
        "avg_latency_sec_per_row": total_latency / len(rows) if rows else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }


def safe_mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


if __name__ == "__main__":
    raise SystemExit(main())
