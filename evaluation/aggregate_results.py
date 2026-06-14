#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate meter/count scored Shaer generations.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    rows = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"meter:{row.get('base_meter','')}"].append(row)
        groups[f"form:{row.get('form','')}"].append(row)
    summary = {name: summarize(items) for name, items in groups.items()}
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"aggregated_groups={len(summary)} output={args.output_json}")
    return 0


def summarize(rows):
    def avg(key):
        vals = [float(row.get(key, 0.0) or 0.0) for row in rows]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "rows": len(rows),
        "meter_mean": avg("meter"),
        "count_adherence_mean": avg("count_adherence"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
