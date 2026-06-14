#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METRICS = [
    "description_adherence",
    "meaning",
    "fluency",
    "coherence",
    "poeticness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple judge calibration folders into a paper-ready table.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Entries in the form label=folder_path")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for item in args.inputs:
        label, folder = parse_input(item)
        folder_path = Path(folder)
        for metric in DEFAULT_METRICS:
            summary = json.loads((folder_path / f"{metric}.summary.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "judge_model": label,
                    "metric": metric,
                    "rows": summary["rows"],
                    "mean": summary["score_mean"],
                    "median": summary["score_median"],
                    "min": summary["score_min"],
                    "max": summary["score_max"],
                    "avg_latency_sec": summary["avg_latency_sec"],
                    "total_latency_sec": summary["total_latency_sec"],
                    "avg_cost_usd": summary["avg_reported_cost_usd"],
                    "total_cost_usd": summary["total_reported_cost_usd"],
                    "projected_total_cost_usd": summary["projected_total_cost_usd"],
                }
            )

    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows)
    print(f"rows={len(rows)} csv={args.output_csv} md={args.output_md}")
    return 0


def parse_input(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Expected label=folder_path, got: {item}")
    label, folder = item.split("=", 1)
    return label.strip(), folder.strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "judge_model",
        "metric",
        "rows",
        "mean",
        "median",
        "min",
        "max",
        "avg_latency_sec",
        "total_latency_sec",
        "avg_cost_usd",
        "total_cost_usd",
        "projected_total_cost_usd",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Judge Model | Metric | Rows | Mean | Median | Min | Max | Avg Latency (s) | Total Time (s) | Avg Cost ($) | Total Cost ($) | Projected Full Cost ($) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {judge_model} | {metric} | {rows} | {mean:.2f} | {median:.2f} | {min:.2f} | {max:.2f} | {avg_latency_sec:.2f} | {total_latency_sec:.2f} | {avg_cost_usd:.6f} | {total_cost_usd:.6f} | {projected_total_cost_usd:.2f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
