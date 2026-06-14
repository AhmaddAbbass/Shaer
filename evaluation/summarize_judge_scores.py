#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import median
from pathlib import Path
from typing import Any

from judge_common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize judge score JSONL and project full-run cost.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--full-rows", type=int, default=13924)
    parser.add_argument("--metrics-per-row", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(Path(args.input_jsonl))
    scores = [int(row.get("score") or 0) for row in rows]
    costs = [float(row.get("reported_cost_usd") or 0.0) for row in rows]
    prompt_tokens = [int(row.get("prompt_tokens") or 0) for row in rows]
    completion_tokens = [int(row.get("completion_tokens") or 0) for row in rows]
    latencies = [float(row.get("judge_latency_sec") or 0.0) for row in rows]
    score_counts = Counter(scores)

    avg_cost = safe_mean(costs)
    avg_prompt_tokens = safe_mean(prompt_tokens)
    avg_completion_tokens = safe_mean(completion_tokens)
    avg_latency = safe_mean(latencies)

    projected_calls = int(args.full_rows) * int(args.metrics_per_row)
    projected_cost = avg_cost * projected_calls

    summary = {
        "rows": len(rows),
        "score_mean": safe_mean(scores),
        "score_median": safe_median(scores),
        "score_min": safe_min(scores),
        "score_max": safe_max(scores),
        "score_counts": {str(key): score_counts[key] for key in sorted(score_counts)},
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_prompt_tokens + avg_completion_tokens,
        "avg_latency_sec": avg_latency,
        "total_latency_sec": sum(latencies),
        "avg_reported_cost_usd": avg_cost,
        "total_reported_cost_usd": sum(costs),
        "projected_full_rows": int(args.full_rows),
        "projected_metrics_per_row": int(args.metrics_per_row),
        "projected_total_calls": projected_calls,
        "projected_total_cost_usd": projected_cost,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rows={len(rows)} projected_total_cost_usd={projected_cost:.6f} output={output_path}")
    return 0


def safe_mean(values: list[Any]) -> float:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def safe_median(values: list[Any]) -> float:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except Exception:
            pass
    return float(median(vals)) if vals else 0.0


def safe_min(values: list[Any]) -> float:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except Exception:
            pass
    return min(vals) if vals else 0.0


def safe_max(values: list[Any]) -> float:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except Exception:
            pass
    return max(vals) if vals else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
