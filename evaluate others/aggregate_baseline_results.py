#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate meter/count scored baseline generations.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"model:{row.get('model_name', '')}"].append(row)
        groups[f"group:{row.get('model_group', '')}"].append(row)
        groups[f"meter:{row.get('base_meter', '')}"].append(row)
        groups[f"form:{row.get('form', '')}"].append(row)
        groups[f"model_meter:{row.get('model_name', '')}|{row.get('base_meter', '')}"].append(row)

    summary = {group_name: summarize(group_rows) for group_name, group_rows in groups.items()}
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"aggregated_groups={len(summary)} output={args.output_json}")
    return 0


def summarize(rows: list[dict]) -> dict:
    meter_vals = [float(row.get("meter", 0.0) or 0.0) for row in rows]
    count_vals = [float(row.get("count_adherence", 0.0) or 0.0) for row in rows]
    exact_count = [
        int(int(row.get("parsed_num_lines") or 0) == int(row.get("requested_num_lines") or -1))
        for row in rows
        if row.get("requested_num_lines") is not None
    ]
    return {
        "rows": len(rows),
        "meter_mean": avg(meter_vals),
        "count_adherence_mean": avg(count_vals),
        "exact_count_rate": avg(exact_count),
    }


def avg(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
