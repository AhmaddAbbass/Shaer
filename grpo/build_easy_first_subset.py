#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ID = "Shaer-AI/ashaar-enhanced-desc-baseform-final-sft-lte20-min500-splits-grpo-meter-count-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "curated_meter_count_easyfirst_short_drop_trio"
DEFAULT_HARD_MANIFEST = ROOT / "outputs" / "curated_meter_count_drop_trio" / "hard_diagnostic_cap_256" / "selected_manifest.csv"
DROP_METERS = {"المديد", "المنسرح", "الهزج"}
DEFAULT_ALLOWED_LENGTH_BUCKETS = ("1-3", "4-6")
MANIFEST_FIELDS = [
    "row_uid",
    "source_dataset_id",
    "source_split",
    "source_id",
    "source_index",
    "source_row_index_in_split",
    "difficulty",
    "base_meter",
    "form",
    "meter_label",
    "requested_bayts",
    "requested_lines",
    "length_bucket",
    "sampler_group",
    "split_group",
    "mean_meter_score",
    "mean_count_adherence_score",
    "num_candidates_requested",
    "num_candidates_generated",
    "num_candidates_scored",
    "num_strong_meter_candidates",
    "num_strong_exact_candidates",
    "num_bad_meter_candidates",
    "strong_meter_rate",
    "strong_exact_rate",
    "bad_meter_rate",
    "meter_count_success_rate",
    "count_exact_rate",
    "manifest_order",
    "selection_pool",
    "selection_reason",
    "cap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build short easy-first GRPO subset with medium top-up.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cap-per-meter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-manifest-source", default=str(DEFAULT_HARD_MANIFEST))
    parser.add_argument(
        "--allowed-length-buckets",
        nargs="+",
        default=list(DEFAULT_ALLOWED_LENGTH_BUCKETS),
        help="Only keep rows from these length buckets.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()[:16], 16)


def normalized_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    for key in ["row_uid", "source_dataset_id", "source_split", "source_id", "difficulty", "base_meter", "form", "meter_label", "length_bucket", "sampler_group", "split_group"]:
        row[key] = str(row.get(key, "") or "")
    for key in [
        "source_index",
        "source_row_index_in_split",
        "requested_bayts",
        "requested_lines",
        "num_candidates_requested",
        "num_candidates_generated",
        "num_candidates_scored",
        "num_strong_meter_candidates",
        "num_strong_exact_candidates",
        "num_bad_meter_candidates",
        "manifest_order",
    ]:
        try:
            row[key] = int(row.get(key, 0) or 0)
        except Exception:
            row[key] = 0
    for key in [
        "mean_meter_score",
        "mean_count_adherence_score",
        "strong_meter_rate",
        "strong_exact_rate",
        "bad_meter_rate",
        "meter_count_success_rate",
        "count_exact_rate",
    ]:
        try:
            row[key] = float(row.get(key, 0.0) or 0.0)
        except Exception:
            row[key] = 0.0
    return row


def row_sort_key(seed: int, row: dict[str, Any]) -> tuple[int, int]:
    return (stable_hash(seed, row["row_uid"]), int(row.get("source_index", 0) or 0))


def round_robin_take(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    if cap <= 0 or not rows:
        return []
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row.get("length_bucket", "") or "")].append(row)
    bucket_names = sorted(by_bucket.keys(), key=lambda name: (name not in {"1-3", "4-6", "7-10", "11-20"}, name))
    for bucket in bucket_names:
        by_bucket[bucket] = sorted(by_bucket[bucket], key=lambda row: row_sort_key(seed, row))
    positions = {bucket: 0 for bucket in bucket_names}
    selected: list[dict[str, Any]] = []
    while len(selected) < cap:
        took_any = False
        for bucket in bucket_names:
            pos = positions[bucket]
            if pos >= len(by_bucket[bucket]):
                continue
            selected.append(by_bucket[bucket][pos])
            positions[bucket] += 1
            took_any = True
            if len(selected) >= cap:
                break
        if not took_any:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    chosen_fields = fieldnames or MANIFEST_FIELDS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=chosen_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in chosen_fields})


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    cap_dir = output_root / f"cap_{int(args.cap_per_meter)}"
    manifest_path = cap_dir / "selected_manifest.csv"
    summary_path = output_root / "easyfirst_summary.json"
    meter_summary_path = cap_dir / "meter_summary.csv"
    hard_manifest_target = output_root / "hard_diagnostic_cap_256" / "selected_manifest.csv"
    ensure_dir(output_root)
    allowed_length_buckets = {str(value).strip() for value in args.allowed_length_buckets if str(value).strip()}

    ds = load_dataset(args.dataset_id, split="train")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"easy": [], "medium": []})
    for raw in ds:
        row = normalized_row(raw)
        meter = row["base_meter"]
        difficulty = row["difficulty"]
        if meter in DROP_METERS:
            continue
        if difficulty not in {"easy", "medium"}:
            continue
        if allowed_length_buckets and row["length_bucket"] not in allowed_length_buckets:
            continue
        grouped[meter][difficulty].append(row)

    selected_rows: list[dict[str, Any]] = []
    meter_summaries: list[dict[str, Any]] = []
    for meter in sorted(grouped):
        easy_rows = grouped[meter]["easy"]
        medium_rows = grouped[meter]["medium"]
        selected_easy = round_robin_take(easy_rows, min(args.cap_per_meter, len(easy_rows)), args.seed)
        remaining = max(0, args.cap_per_meter - len(selected_easy))
        selected_medium = round_robin_take(medium_rows, min(remaining, len(medium_rows)), args.seed + 1)
        for row in selected_easy:
            row["selection_pool"] = "easy"
            row["selection_reason"] = "easy_first_kept"
            row["cap"] = args.cap_per_meter
        for row in selected_medium:
            row["selection_pool"] = "medium"
            row["selection_reason"] = "medium_topup"
            row["cap"] = args.cap_per_meter
        selected_rows.extend(selected_easy)
        selected_rows.extend(selected_medium)
        meter_summaries.append(
            {
                "base_meter": meter,
                "cap": args.cap_per_meter,
                "available_easy": len(easy_rows),
                "available_medium": len(medium_rows),
                "selected_easy": len(selected_easy),
                "selected_medium": len(selected_medium),
                "selected_total": len(selected_easy) + len(selected_medium),
                "reached_cap": len(selected_easy) + len(selected_medium) >= args.cap_per_meter,
            }
        )

    selected_rows = sorted(selected_rows, key=lambda row: (row["base_meter"], row["selection_pool"], row_sort_key(args.seed, row)))
    write_csv(manifest_path, selected_rows)
    write_csv(
        meter_summary_path,
        meter_summaries,
        fieldnames=[
            "base_meter",
            "cap",
            "available_easy",
            "available_medium",
            "selected_easy",
            "selected_medium",
            "selected_total",
            "reached_cap",
        ],
    )

    hard_source = Path(args.hard_manifest_source).resolve()
    ensure_dir(hard_manifest_target.parent)
    shutil.copy2(hard_source, hard_manifest_target)

    payload = {
        "dataset_id": args.dataset_id,
        "output_manifest": str(manifest_path),
        "hard_manifest_source": str(hard_source),
        "hard_manifest_target": str(hard_manifest_target),
        "cap_per_meter": args.cap_per_meter,
        "drop_meters": sorted(DROP_METERS),
        "allowed_length_buckets": sorted(allowed_length_buckets),
        "selected_total": len(selected_rows),
        "selected_easy": sum(1 for row in selected_rows if row.get("selection_pool") == "easy"),
        "selected_medium": sum(1 for row in selected_rows if row.get("selection_pool") == "medium"),
        "base_meter_counts": {row["base_meter"]: row["selected_total"] for row in meter_summaries},
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
