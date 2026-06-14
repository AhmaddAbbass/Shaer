#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import atomic_write_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate baseline generation JSONL outputs.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--expected-models", default="")
    parser.add_argument("--expected-source-rows", type=int, default=0)
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(Path(args.input_jsonl))
    key_counts = Counter((row.get("model_name"), input_row_id(row), row.get("sample_index")) for row in rows)
    duplicates = [key for key, count in key_counts.items() if count > 1]
    bad_rows = [
        row
        for row in rows
        if row.get("generation_status") != "ok"
        or row.get("health_status") != "ok"
        or not str(row.get("generated_text") or "").strip()
    ]
    per_model = defaultdict(list)
    for row in rows:
        per_model[str(row.get("model_name") or "")].append(row)

    expected_models = [name.strip() for name in str(args.expected_models or "").split(",") if name.strip()]
    models_summary: dict[str, dict[str, int | bool]] = {}
    overall_valid = not duplicates and not bad_rows
    for model_name, model_rows in per_model.items():
        unique_pairs = {
            (input_row_id(row), row.get("sample_index"))
            for row in model_rows
            if row.get("generation_status") == "ok" and row.get("health_status") == "ok"
        }
        expected_rows = 0
        model_valid = True
        if args.expected_source_rows:
            expected_rows = int(args.expected_source_rows) * int(args.samples_per_row)
            model_valid = len(unique_pairs) == expected_rows
            overall_valid = overall_valid and model_valid
        models_summary[model_name] = {
            "rows": len(model_rows),
            "valid_pairs": len(unique_pairs),
            "expected_rows": expected_rows,
            "valid": bool(model_valid),
        }

    for model_name in expected_models:
        models_summary.setdefault(
            model_name,
            {
                "rows": 0,
                "valid_pairs": 0,
                "expected_rows": int(args.expected_source_rows) * int(args.samples_per_row)
                if args.expected_source_rows
                else 0,
                "valid": False,
            },
        )
        overall_valid = overall_valid and bool(models_summary[model_name]["valid"])

    summary = {
        "row_count": len(rows),
        "duplicate_key_count": len(duplicates),
        "bad_row_count": len(bad_rows),
        "model_count": len(per_model),
        "models": models_summary,
        "valid": bool(overall_valid),
    }
    summary_path = Path(args.summary_json) if args.summary_json else Path(args.input_jsonl).with_suffix(".validation.json")
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["valid"] else 1


def input_row_id(row: dict) -> str:
    value = str(row.get("input_row_id") or row.get("manifest_row_id") or row.get("selected_shaer_generation_id") or "").strip()
    if value:
        return value
    return str(row.get("source_row_index"))


if __name__ == "__main__":
    raise SystemExit(main())
