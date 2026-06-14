#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import atomic_write_json, read_jsonl, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple per-model JSONL result files into long and wide comparison outputs.")
    parser.add_argument("--inputs", required=True, help="Comma-separated JSONL files")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--wide-csv", default="")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = [Path(item.strip()) for item in str(args.inputs).split(",") if item.strip()]
    if not input_paths:
        raise RuntimeError("No input files were provided.")

    merged_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for path in input_paths:
        for row in read_jsonl(path):
            key = (
                str(row.get("model_name") or ""),
                baseline_row_id(row),
                int(row.get("sample_index") or 0),
            )
            if key in seen_keys:
                raise RuntimeError(f"Duplicate merged key detected: {key}")
            seen_keys.add(key)
            merged_rows.append(row)

    write_jsonl(Path(args.output_jsonl), merged_rows)
    if args.output_csv:
        write_csv(Path(args.output_csv), merged_rows)
    if args.wide_csv:
        write_wide_csv(Path(args.wide_csv), merged_rows)

    summary = {
        "created_at_utc": utc_now_iso(),
        "input_files": [str(path) for path in input_paths],
        "row_count": len(merged_rows),
        "model_count": len({str(row.get('model_name') or '') for row in merged_rows}),
        "source_row_count": len({baseline_row_id(row) for row in merged_rows}),
    }
    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_jsonl).with_suffix(".summary.json")
    atomic_write_json(summary_path, summary)
    print(f"merged_rows={len(merged_rows)} output={args.output_jsonl}")
    return 0


def baseline_row_id(row: dict[str, Any]) -> str:
    for key in ("input_row_id", "manifest_row_id", "selected_shaer_generation_id", "generation_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(row.get("source_row_index") or "")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify(row.get(key)) for key in fieldnames})


def write_wide_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    model_names = sorted({str(row.get("model_name") or "") for row in rows})
    for row in rows:
        row_id = baseline_row_id(row)
        bucket = grouped.setdefault(
            row_id,
            {
                "row_id": row_id,
                "source_row_index": row.get("source_row_index"),
                "base_meter": row.get("base_meter"),
                "form": row.get("form"),
                "requested_num_lines": row.get("requested_num_lines") or row.get("sft_num_lines"),
                "reference_completion": row.get("reference_completion"),
            },
        )
        model_name = str(row.get("model_name") or "")
        bucket[f"{model_name}__generated_text"] = row.get("generated_text")
        bucket[f"{model_name}__meter"] = row.get("meter")
        bucket[f"{model_name}__count_adherence"] = row.get("count_adherence")
        bucket[f"{model_name}__health_status"] = row.get("health_status")

    fieldnames = [
        "row_id",
        "source_row_index",
        "base_meter",
        "form",
        "requested_num_lines",
        "reference_completion",
    ]
    for model_name in model_names:
        fieldnames.extend(
            [
                f"{model_name}__generated_text",
                f"{model_name}__meter",
                f"{model_name}__count_adherence",
                f"{model_name}__health_status",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row_id in sorted(grouped):
            writer.writerow({key: stringify(grouped[row_id].get(key)) for key in fieldnames})


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
