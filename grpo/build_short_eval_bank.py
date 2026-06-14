#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "eval_bank_short_no_trio"
DROP_METERS = {"المديد", "المنسرح", "الهزج"}
ALLOWED_BUCKETS = ("1-3", "4-6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short-only no-trio eval bank.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--per-meter-per-bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(seed: int, *parts: Any) -> int:
    blob = "|".join(str(part) for part in (seed, *parts))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def bucket_for_bayts(value: Any) -> str:
    try:
        bayts = int(value or 0)
    except Exception:
        bayts = 0
    if 1 <= bayts <= 3:
        return "1-3"
    if 4 <= bayts <= 6:
        return "4-6"
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    manifest_path = output_root / "selected_eval_bank.csv"
    summary_path = output_root / "selected_eval_bank_summary.json"

    ds = load_dataset(args.dataset_id, split=args.split)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for idx, raw in enumerate(ds):
        meter = str(raw.get("base_meter", "") or "").strip()
        if not meter or meter in DROP_METERS:
            continue
        bucket = bucket_for_bayts(raw.get("requested_bayts"))
        if bucket not in ALLOWED_BUCKETS:
            continue
        row = {
            "source_index": idx,
            "base_meter": meter,
            "meter_label": str(raw.get("meter_label", "") or "").strip(),
            "requested_bayts": int(raw.get("requested_bayts", 0) or 0),
            "requested_lines": int(raw.get("requested_lines", 0) or 0),
            "length_bucket": bucket,
            "row_uid": str(raw.get("row_uid", "") or "").strip(),
            "source_id": str(raw.get("source_id", raw.get("id", "")) or "").strip(),
            "source_split": str(raw.get("source_split", args.split) or "").strip(),
            "description": str(raw.get("description", "") or "").strip(),
            "poem_text": str(raw.get("poem_text", "") or "").strip(),
        }
        grouped[(meter, bucket)].append(row)

    selected: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for meter in sorted({key[0] for key in grouped}):
        for bucket in ALLOWED_BUCKETS:
            rows = grouped.get((meter, bucket), [])
            ordered = sorted(
                rows,
                key=lambda row: (
                    stable_hash(args.seed, meter, bucket, row.get("row_uid", row.get("source_id", row.get("source_index", "")))),
                    int(row.get("source_index", 0) or 0),
                ),
            )
            chosen = ordered[: min(int(args.per_meter_per_bucket), len(ordered))]
            selected.extend(chosen)
            summary_rows.append(
                {
                    "base_meter": meter,
                    "length_bucket": bucket,
                    "available": len(rows),
                    "selected": len(chosen),
                    "target": int(args.per_meter_per_bucket),
                }
            )

    selected.sort(key=lambda row: (row["base_meter"], row["length_bucket"], int(row["source_index"])))
    write_csv(
        manifest_path,
        selected,
        fieldnames=[
            "source_index",
            "base_meter",
            "meter_label",
            "requested_bayts",
            "requested_lines",
            "length_bucket",
            "row_uid",
            "source_id",
            "source_split",
            "description",
            "poem_text",
        ],
    )

    payload = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "output_manifest": str(manifest_path),
        "per_meter_per_bucket": int(args.per_meter_per_bucket),
        "drop_meters": sorted(DROP_METERS),
        "allowed_length_buckets": list(ALLOWED_BUCKETS),
        "selected_total": len(selected),
        "meter_bucket_counts": summary_rows,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
