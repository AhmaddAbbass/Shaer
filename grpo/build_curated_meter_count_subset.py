#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import httpx


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ID = "Shaer-AI/ashaar-enhanced-desc-baseform-final-sft-lte20-min500-splits-grpo-meter-count-v1"
DEFAULT_CAPS = [2500, 3000]
DEFAULT_MEDIUM_TARGET_SHARE = 0.60
DEFAULT_HARD_BANK_CAP = 256
LENGTH_BUCKET_ORDER = ["1-3", "4-6", "7-10", "11-20"]
POOL_ORDER = ["easy", "medium_strict", "medium_fallback", "hard_diagnostic"]
RECOVERABLE_MEDIUM_RULES = {"default"}


def utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_dir() -> Path:
    return ROOT / "outputs" / f"curated_meter_count_subset_{utc_stamp()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed curated GRPO subset analysis artifacts from the derived meter+count dataset.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--caps", nargs="+", type=int, default=DEFAULT_CAPS)
    parser.add_argument("--cap-per-meter", type=int, default=None, help="Alias for a single cap in --caps.")
    parser.add_argument("--medium-target-share", type=float, default=DEFAULT_MEDIUM_TARGET_SHARE)
    parser.add_argument("--hard-bank-cap", type=int, default=DEFAULT_HARD_BANK_CAP)
    parser.add_argument("--hard-cap-per-meter", type=int, default=None, help="Alias for --hard-bank-cap.")
    parser.add_argument(
        "--recoverable-medium-rule",
        default="default",
        choices=sorted(RECOVERABLE_MEDIUM_RULES),
        help="Locked rule spec for which medium rows count as recoverable.",
    )
    parser.add_argument(
        "--balance-length-buckets",
        default="true",
        choices=["true", "false"],
        help="Whether capped selection should preserve length-bucket diversity.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="", help="Alias for --output-dir.")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def fetch_train_parquet_urls(dataset_id: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{dataset_id}/parquet"
    with httpx.Client(timeout=60.0) as client:
        res = client.get(url)
        res.raise_for_status()
        payload = res.json()
    if not payload:
        raise RuntimeError(f"No parquet payload returned for dataset: {dataset_id}")
    first_config = next(iter(payload.values()))
    train_urls = list(first_config.get("train") or [])
    if not train_urls:
        raise RuntimeError(f"No train parquet URLs found for dataset: {dataset_id}")
    return train_urls


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs;")
    except Exception:
        pass
    con.execute("LOAD httpfs;")
    return con


def load_rows(train_urls: list[str]) -> list[dict[str, Any]]:
    con = connect_duckdb()
    query = """
        select
            row_uid,
            source_dataset_id,
            source_split,
            source_index,
            source_row_index_in_split,
            source_id,
            difficulty,
            base_meter,
            form,
            meter_label,
            requested_bayts,
            requested_lines,
            length_bucket,
            sampler_group,
            split_group,
            num_candidates_requested,
            num_candidates_generated,
            num_candidates_scored,
            mean_meter_score,
            mean_count_adherence_score,
            num_strong_meter_candidates,
            num_strong_exact_candidates,
            num_bad_meter_candidates,
            strong_meter_rate,
            strong_exact_rate,
            bad_meter_rate,
            meter_count_success_rate,
            count_exact_rate,
            manifest_order
        from read_parquet(?)
    """
    result = con.execute(query, [train_urls])
    columns = [item[0] for item in result.description]
    rows = []
    for values in result.fetchall():
        row = dict(zip(columns, values))
        row["row_uid"] = str(row["row_uid"])
        row["source_dataset_id"] = str(row["source_dataset_id"])
        row["source_split"] = str(row["source_split"])
        row["source_id"] = str(row["source_id"])
        row["difficulty"] = str(row["difficulty"])
        row["base_meter"] = str(row["base_meter"])
        row["form"] = str(row["form"])
        row["meter_label"] = str(row["meter_label"])
        row["length_bucket"] = str(row["length_bucket"])
        row["sampler_group"] = str(row["sampler_group"])
        row["split_group"] = str(row["split_group"])
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
            row[key] = int(row[key])
        for key in [
            "mean_meter_score",
            "mean_count_adherence_score",
            "strong_meter_rate",
            "strong_exact_rate",
            "bad_meter_rate",
            "meter_count_success_rate",
            "count_exact_rate",
        ]:
            row[key] = float(row[key])
        rows.append(row)
    return rows


def stable_hash(seed: int, value: str) -> int:
    blob = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(blob[:16], 16)


def length_bucket_sort_key(bucket: str) -> tuple[int, str]:
    if bucket in LENGTH_BUCKET_ORDER:
        return (LENGTH_BUCKET_ORDER.index(bucket), bucket)
    return (len(LENGTH_BUCKET_ORDER), bucket)


def classify_medium_pool(row: dict[str, Any]) -> str:
    if row["difficulty"] != "medium":
        return ""
    if row["num_strong_meter_candidates"] >= 2:
        return "medium_strict"
    if (
        row["num_strong_meter_candidates"] == 1
        and row["num_strong_exact_candidates"] >= 1
        and row["mean_meter_score"] >= 0.40
        and row["bad_meter_rate"] <= 0.50
    ):
        return "medium_strict"
    if (
        row["num_strong_meter_candidates"] >= 1
        and row["mean_meter_score"] >= 0.35
        and row["bad_meter_rate"] <= 0.625
    ):
        return "medium_fallback"
    return ""


def normal_sort_key(seed: int, row: dict[str, Any]) -> tuple[int, int]:
    return (stable_hash(seed, row["row_uid"]), int(row["source_index"]))


def fallback_sort_key(seed: int, row: dict[str, Any]) -> tuple[float, ...]:
    return (
        -float(row["num_strong_meter_candidates"]),
        -float(row["num_strong_exact_candidates"]),
        -float(row["mean_meter_score"]),
        float(row["bad_meter_rate"]),
        -float(row["count_exact_rate"]),
        float(stable_hash(seed, row["row_uid"])),
        float(row["source_index"]),
    )


def row_to_manifest(row: dict[str, Any], selection_pool: str, selection_reason: str, cap: int | None = None) -> dict[str, Any]:
    return {
        "row_uid": row["row_uid"],
        "source_dataset_id": row["source_dataset_id"],
        "source_split": row["source_split"],
        "source_id": row["source_id"],
        "source_index": row["source_index"],
        "source_row_index_in_split": row["source_row_index_in_split"],
        "difficulty": row["difficulty"],
        "base_meter": row["base_meter"],
        "form": row["form"],
        "meter_label": row["meter_label"],
        "requested_bayts": row["requested_bayts"],
        "requested_lines": row["requested_lines"],
        "length_bucket": row["length_bucket"],
        "sampler_group": row["sampler_group"],
        "split_group": row["split_group"],
        "mean_meter_score": row["mean_meter_score"],
        "mean_count_adherence_score": row["mean_count_adherence_score"],
        "num_strong_meter_candidates": row["num_strong_meter_candidates"],
        "num_strong_exact_candidates": row["num_strong_exact_candidates"],
        "num_bad_meter_candidates": row["num_bad_meter_candidates"],
        "strong_meter_rate": row["strong_meter_rate"],
        "strong_exact_rate": row["strong_exact_rate"],
        "bad_meter_rate": row["bad_meter_rate"],
        "meter_count_success_rate": row["meter_count_success_rate"],
        "count_exact_rate": row["count_exact_rate"],
        "selection_pool": selection_pool,
        "selection_reason": selection_reason,
        "cap": cap if cap is not None else "",
    }


def allocate_bucket_counts(grouped: dict[str, list[dict[str, Any]]], target: int) -> dict[str, int]:
    active = [bucket for bucket, rows in grouped.items() if rows]
    target = min(int(target), sum(len(rows) for rows in grouped.values()))
    alloc = {bucket: 0 for bucket in grouped}
    if target <= 0 or not active:
        return alloc

    if target >= len(active):
        for bucket in active:
            alloc[bucket] = 1
    remaining = target - sum(alloc.values())
    if remaining <= 0:
        return alloc

    capacities = {bucket: len(grouped[bucket]) - alloc[bucket] for bucket in active}
    total_capacity = sum(max(0, value) for value in capacities.values())
    if total_capacity <= 0:
        return alloc

    provisional: dict[str, int] = {}
    remainders: list[tuple[float, tuple[int, str], str]] = []
    for bucket in active:
        cap = max(0, capacities[bucket])
        raw = remaining * cap / total_capacity if total_capacity else 0.0
        take = min(cap, int(math.floor(raw)))
        provisional[bucket] = take
        remainder = raw - math.floor(raw)
        remainders.append((remainder, length_bucket_sort_key(bucket), bucket))
    for bucket, take in provisional.items():
        alloc[bucket] += take
    leftover = target - sum(alloc.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    while leftover > 0:
        moved = False
        for _remainder, _bucket_key, bucket in remainders:
            if alloc[bucket] >= len(grouped[bucket]):
                continue
            alloc[bucket] += 1
            leftover -= 1
            moved = True
            if leftover <= 0:
                break
        if not moved:
            break
    return alloc


def choose_rows(pool: list[dict[str, Any]], target: int, *, seed: int, ranking: str, balance_length_buckets: bool) -> list[dict[str, Any]]:
    if target <= 0 or not pool:
        return []
    if ranking == "fallback":
        ordered = sorted(pool, key=lambda row: fallback_sort_key(seed, row))
    else:
        ordered = sorted(pool, key=lambda row: normal_sort_key(seed, row))

    if not balance_length_buckets:
        return ordered[:target]

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_bucket[str(row["length_bucket"] or "")].append(row)
    allocations = allocate_bucket_counts(by_bucket, target)
    selected: list[dict[str, Any]] = []
    for bucket, _bucket_rows in sorted(by_bucket.items(), key=lambda item: length_bucket_sort_key(item[0])):
        take = allocations.get(bucket, 0)
        if take <= 0:
            continue
        selected.extend(by_bucket[bucket][:take])
    return selected


def sort_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["base_meter"]),
            POOL_ORDER.index(row["selection_pool"]) if row["selection_pool"] in POOL_ORDER else len(POOL_ORDER),
            length_bucket_sort_key(str(row["length_bucket"])),
            int(row["source_index"]),
            str(row["row_uid"]),
        ),
    )


def select_curated_rows(
    rows: list[dict[str, Any]],
    *,
    cap: int,
    seed: int,
    medium_target_share: float,
    recoverable_medium_rule: str,
    balance_length_buckets: bool,
) -> dict[str, Any]:
    available_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    meter_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        meter = row["base_meter"]
        if row["difficulty"] == "easy":
            meter_rows[meter]["easy"].append(row)
            available_rows[meter].append(row)
            continue
        if recoverable_medium_rule not in RECOVERABLE_MEDIUM_RULES:
            raise ValueError(f"Unsupported recoverable medium rule: {recoverable_medium_rule}")
        medium_pool = classify_medium_pool(row)
        if medium_pool:
            meter_rows[meter][medium_pool].append(row)
            available_rows[meter].append(row)

    selected_manifest: list[dict[str, Any]] = []
    meter_summary_rows: list[dict[str, Any]] = []

    for meter in sorted(available_rows):
        easy_pool = list(meter_rows[meter].get("easy", []))
        strict_pool = list(meter_rows[meter].get("medium_strict", []))
        fallback_pool = list(meter_rows[meter].get("medium_fallback", []))
        eligible_total = len(easy_pool) + len(strict_pool) + len(fallback_pool)
        quota = min(int(cap), eligible_total)
        selected: list[dict[str, Any]] = []

        if eligible_total <= cap:
            selected = easy_pool + strict_pool + fallback_pool
        else:
            medium_target = int(math.ceil(float(medium_target_share) * quota))
            selected_strict = choose_rows(
                strict_pool,
                medium_target,
                seed=seed,
                ranking="normal",
                balance_length_buckets=balance_length_buckets,
            )
            selected.extend(selected_strict)

            selected_uids = {row["row_uid"] for row in selected}
            remaining_medium_needed = max(0, medium_target - len(selected))
            if remaining_medium_needed > 0:
                remaining_fallback = [row for row in fallback_pool if row["row_uid"] not in selected_uids]
                selected_fallback = choose_rows(
                    remaining_fallback,
                    remaining_medium_needed,
                    seed=seed,
                    ranking="fallback",
                    balance_length_buckets=balance_length_buckets,
                )
                selected.extend(selected_fallback)
                selected_uids = {row["row_uid"] for row in selected}

            easy_target = max(0, quota - len(selected))
            selected_easy = choose_rows(
                easy_pool,
                easy_target,
                seed=seed,
                ranking="normal",
                balance_length_buckets=balance_length_buckets,
            )
            selected.extend(selected_easy)
            selected_uids = {row["row_uid"] for row in selected}

            remaining_needed = max(0, quota - len(selected))
            if remaining_needed > 0:
                remaining_strict = [row for row in strict_pool if row["row_uid"] not in selected_uids]
                extra_strict = choose_rows(
                    remaining_strict,
                    remaining_needed,
                    seed=seed,
                    ranking="normal",
                    balance_length_buckets=balance_length_buckets,
                )
                selected.extend(extra_strict)
                selected_uids = {row["row_uid"] for row in selected}
                remaining_needed = max(0, quota - len(selected))

            if remaining_needed > 0:
                remaining_fallback = [row for row in fallback_pool if row["row_uid"] not in selected_uids]
                extra_fallback = choose_rows(
                    remaining_fallback,
                    remaining_needed,
                    seed=seed,
                    ranking="fallback",
                    balance_length_buckets=balance_length_buckets,
                )
                selected.extend(extra_fallback)
                selected_uids = {row["row_uid"] for row in selected}
                remaining_needed = max(0, quota - len(selected))

            if remaining_needed > 0:
                remaining_easy = [row for row in easy_pool if row["row_uid"] not in selected_uids]
                extra_easy = choose_rows(
                    remaining_easy,
                    remaining_needed,
                    seed=seed,
                    ranking="normal",
                    balance_length_buckets=balance_length_buckets,
                )
                selected.extend(extra_easy)

        selected_by_uid = {row["row_uid"]: row for row in selected}
        selected_easy = [row for row in easy_pool if row["row_uid"] in selected_by_uid]
        selected_strict = [row for row in strict_pool if row["row_uid"] in selected_by_uid]
        selected_fallback = [row for row in fallback_pool if row["row_uid"] in selected_by_uid]

        for row in selected_easy:
            selected_manifest.append(row_to_manifest(row, "easy", "easy_kept", cap=cap))
        for row in selected_strict:
            selected_manifest.append(row_to_manifest(row, "medium_strict", "medium_strict_kept", cap=cap))
        for row in selected_fallback:
            selected_manifest.append(row_to_manifest(row, "medium_fallback", "medium_fallback_kept", cap=cap))

        meter_summary_rows.append(
            {
                "base_meter": meter,
                "available_easy": len(easy_pool),
                "available_medium_strict": len(strict_pool),
                "available_medium_fallback": len(fallback_pool),
                "available_total": eligible_total,
                "selected_easy": len(selected_easy),
                "selected_medium_strict": len(selected_strict),
                "selected_medium_fallback": len(selected_fallback),
                "selected_medium_total": len(selected_strict) + len(selected_fallback),
                "selected_total": len(selected),
                "cap": int(cap),
                "quota": int(quota),
                "was_capped": bool(eligible_total > cap),
                "selected_medium_share": (len(selected_strict) + len(selected_fallback)) / max(1, len(selected)),
                "selected_easy_share": len(selected_easy) / max(1, len(selected)),
            }
        )

    selected_manifest = sort_manifest_rows(selected_manifest)
    return {
        "manifest_rows": selected_manifest,
        "meter_summary_rows": sorted(meter_summary_rows, key=lambda row: row["base_meter"]),
    }


def select_hard_bank(rows: list[dict[str, Any]], *, cap: int, seed: int, balance_length_buckets: bool) -> dict[str, Any]:
    by_meter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["difficulty"] == "hard":
            by_meter[row["base_meter"]].append(row)

    manifest_rows: list[dict[str, Any]] = []
    meter_summary_rows: list[dict[str, Any]] = []
    for meter in sorted(by_meter):
        pool = by_meter[meter]
        quota = min(int(cap), len(pool))
        selected = (
            pool
            if len(pool) <= cap
            else choose_rows(
                pool,
                quota,
                seed=seed,
                ranking="normal",
                balance_length_buckets=balance_length_buckets,
            )
        )
        for row in selected:
            manifest_rows.append(row_to_manifest(row, "hard_diagnostic", "hard_eval_only", cap=cap))
        meter_summary_rows.append(
            {
                "base_meter": meter,
                "available_hard": len(pool),
                "selected_hard": len(selected),
                "cap": int(cap),
                "quota": int(quota),
                "was_capped": bool(len(pool) > cap),
            }
        )
    return {
        "manifest_rows": sort_manifest_rows(manifest_rows),
        "meter_summary_rows": sorted(meter_summary_rows, key=lambda row: row["base_meter"]),
    }


def build_length_tables(manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meter_pool_bucket: dict[tuple[str, str, str], int] = defaultdict(int)
    meter_bucket_total: dict[tuple[str, str], int] = defaultdict(int)
    for row in manifest_rows:
        meter = row["base_meter"]
        pool = row["selection_pool"]
        bucket = row["length_bucket"]
        meter_pool_bucket[(meter, pool, bucket)] += 1
        meter_bucket_total[(meter, bucket)] += 1

    meter_pool_rows = [
        {
            "base_meter": meter,
            "selection_pool": pool,
            "length_bucket": bucket,
            "rows": count,
        }
        for (meter, pool, bucket), count in sorted(
            meter_pool_bucket.items(),
            key=lambda item: (item[0][0], POOL_ORDER.index(item[0][1]) if item[0][1] in POOL_ORDER else len(POOL_ORDER), length_bucket_sort_key(item[0][2])),
        )
    ]
    meter_total_rows = [
        {
            "base_meter": meter,
            "selection_pool": "total",
            "length_bucket": bucket,
            "rows": count,
        }
        for (meter, bucket), count in sorted(
            meter_bucket_total.items(),
            key=lambda item: (item[0][0], length_bucket_sort_key(item[0][1])),
        )
    ]
    return meter_pool_rows, meter_total_rows


def build_overall_summary(manifest_rows: list[dict[str, Any]], meter_summary_rows: list[dict[str, Any]], *, cap: int | None, medium_target_share: float | None, seed: int) -> dict[str, Any]:
    by_pool: dict[str, int] = defaultdict(int)
    by_bucket: dict[str, int] = defaultdict(int)
    by_meter: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        by_pool[row["selection_pool"]] += 1
        by_bucket[row["length_bucket"]] += 1
        by_meter[row["base_meter"]] += 1
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "cap": int(cap) if cap is not None else None,
        "medium_target_share_when_capped": medium_target_share,
        "total_rows": len(manifest_rows),
        "by_selection_pool": dict(sorted(by_pool.items())),
        "by_length_bucket": dict(sorted(by_bucket.items(), key=lambda item: length_bucket_sort_key(item[0]))),
        "by_base_meter": dict(sorted(by_meter.items())),
        "meters_capped": [row["base_meter"] for row in meter_summary_rows if row.get("was_capped")],
    }


def build_compare_rows(cap_to_meter_rows: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    meters = sorted({row["base_meter"] for rows in cap_to_meter_rows.values() for row in rows})
    by_cap_and_meter = {
        (cap, row["base_meter"]): row
        for cap, rows in cap_to_meter_rows.items()
        for row in rows
    }
    compare_rows: list[dict[str, Any]] = []
    for meter in meters:
        row: dict[str, Any] = {"base_meter": meter}
        for cap in sorted(cap_to_meter_rows):
            source = by_cap_and_meter[(cap, meter)]
            row[f"cap_{cap}_selected_total"] = source["selected_total"]
            row[f"cap_{cap}_selected_easy"] = source["selected_easy"]
            row[f"cap_{cap}_selected_medium_strict"] = source["selected_medium_strict"]
            row[f"cap_{cap}_selected_medium_fallback"] = source["selected_medium_fallback"]
            row[f"cap_{cap}_selected_medium_total"] = source["selected_medium_total"]
            row[f"cap_{cap}_selected_medium_share"] = round(float(source["selected_medium_share"]), 6)
            row[f"cap_{cap}_was_capped"] = source["was_capped"]
        compare_rows.append(row)
    return compare_rows


def write_readme(
    path: Path,
    *,
    dataset_id: str,
    caps: list[int],
    medium_target_share: float,
    hard_bank_cap: int,
    cap_summaries: dict[int, dict[str, Any]],
    hard_summary: dict[str, Any],
) -> None:
    lines = [
        "# Curated Meter+Count GRPO Subset Analysis",
        "",
        f"- Source dataset: `{dataset_id}`",
        f"- Selection caps compared: `{', '.join(str(cap) for cap in caps)}`",
        f"- Medium target share when a meter is capped: `{medium_target_share:.2f}`",
        f"- Hard diagnostic bank cap per meter: `{hard_bank_cap}`",
        "",
        "## Medium rules",
        "",
        "- `medium_strict`: `num_strong_meter_candidates >= 2`, or exactly one strong meter candidate with at least one strong exact candidate, `mean_meter_score >= 0.40`, and `bad_meter_rate <= 0.50`.",
        "- `medium_fallback`: remaining medium rows with `num_strong_meter_candidates >= 1`, `mean_meter_score >= 0.35`, and `bad_meter_rate <= 0.625`.",
        "",
        "## Cap comparison",
        "",
    ]
    for cap in sorted(cap_summaries):
        summary = cap_summaries[cap]
        lines.extend(
            [
                f"### Cap `{cap}`",
                "",
                f"- Total rows: `{summary['total_rows']:,}`",
                f"- Easy rows: `{int(summary['by_selection_pool'].get('easy', 0)):,}`",
                f"- Medium strict rows: `{int(summary['by_selection_pool'].get('medium_strict', 0)):,}`",
                f"- Medium fallback rows: `{int(summary['by_selection_pool'].get('medium_fallback', 0)):,}`",
                f"- Meters capped: `{', '.join(summary['meters_capped'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Hard diagnostic bank",
            "",
            f"- Total rows: `{hard_summary['total_rows']:,}`",
            f"- Meters covered: `{len(hard_summary['by_base_meter'])}`",
            f"- Capped meters: `{', '.join(hard_summary['meters_capped'])}`",
            "",
            "See the CSV/JSONL manifests and the meter x difficulty x length-bucket tables in this directory.",
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_arg = args.output_root or args.output_dir
    output_dir = Path(output_arg).expanduser().resolve() if output_arg else default_output_dir().resolve()
    ensure_dir(output_dir)
    caps = [int(args.cap_per_meter)] if args.cap_per_meter is not None else [int(cap) for cap in args.caps]
    hard_cap = int(args.hard_cap_per_meter) if args.hard_cap_per_meter is not None else int(args.hard_bank_cap)
    balance_length_buckets = str(args.balance_length_buckets).strip().lower() == "true"

    train_urls = fetch_train_parquet_urls(args.dataset_id)
    rows = load_rows(train_urls)

    config_payload = {
        "dataset_id": args.dataset_id,
        "train_parquet_urls": train_urls,
        "caps": caps,
        "medium_target_share": float(args.medium_target_share),
        "hard_bank_cap": hard_cap,
        "seed": int(args.seed),
        "recoverable_medium_rule": str(args.recoverable_medium_rule),
        "balance_length_buckets": bool(balance_length_buckets),
        "strict_medium_rule": {
            "num_strong_meter_candidates_gte": 2,
            "or_single_strong_with": {
                "num_strong_exact_candidates_gte": 1,
                "mean_meter_score_gte": 0.40,
                "bad_meter_rate_lte": 0.50,
            },
        },
        "fallback_medium_rule": {
            "num_strong_meter_candidates_gte": 1,
            "mean_meter_score_gte": 0.35,
            "bad_meter_rate_lte": 0.625,
        },
    }
    save_json(output_dir / "config.json", config_payload)

    cap_summaries: dict[int, dict[str, Any]] = {}
    cap_to_meter_rows: dict[int, list[dict[str, Any]]] = {}

    for cap in sorted(set(caps)):
        cap_dir = ensure_dir(output_dir / f"cap_{cap}")
        result = select_curated_rows(
            rows,
            cap=cap,
            seed=int(args.seed),
            medium_target_share=float(args.medium_target_share),
            recoverable_medium_rule=str(args.recoverable_medium_rule),
            balance_length_buckets=balance_length_buckets,
        )
        manifest_rows = result["manifest_rows"]
        meter_summary_rows = result["meter_summary_rows"]
        meter_pool_rows, meter_total_rows = build_length_tables(manifest_rows)
        overall_summary = build_overall_summary(
            manifest_rows,
            meter_summary_rows,
            cap=cap,
            medium_target_share=float(args.medium_target_share),
            seed=int(args.seed),
        )
        cap_summaries[cap] = overall_summary
        cap_to_meter_rows[cap] = meter_summary_rows

        write_jsonl(cap_dir / "selected_manifest.jsonl", manifest_rows)
        write_csv(cap_dir / "selected_manifest.csv", manifest_rows)
        write_csv(cap_dir / "meter_summary.csv", meter_summary_rows)
        write_csv(cap_dir / "meter_difficulty_length_summary.csv", meter_pool_rows + meter_total_rows)
        save_json(cap_dir / "summary.json", overall_summary)

    compare_rows = build_compare_rows(cap_to_meter_rows)
    write_csv(output_dir / "cap_compare_by_meter.csv", compare_rows)

    hard_dir = ensure_dir(output_dir / f"hard_diagnostic_cap_{hard_cap}")
    hard_result = select_hard_bank(rows, cap=hard_cap, seed=int(args.seed), balance_length_buckets=balance_length_buckets)
    hard_manifest_rows = hard_result["manifest_rows"]
    hard_meter_summary_rows = hard_result["meter_summary_rows"]
    hard_meter_pool_rows, hard_meter_total_rows = build_length_tables(hard_manifest_rows)
    hard_summary = build_overall_summary(
        hard_manifest_rows,
        hard_meter_summary_rows,
        cap=hard_cap,
        medium_target_share=None,
        seed=int(args.seed),
    )
    write_jsonl(hard_dir / "selected_manifest.jsonl", hard_manifest_rows)
    write_csv(hard_dir / "selected_manifest.csv", hard_manifest_rows)
    write_csv(hard_dir / "meter_summary.csv", hard_meter_summary_rows)
    write_csv(hard_dir / "meter_length_summary.csv", hard_meter_pool_rows + hard_meter_total_rows)
    save_json(hard_dir / "summary.json", hard_summary)

    write_readme(
        output_dir / "README.md",
        dataset_id=args.dataset_id,
        caps=sorted(set(caps)),
        medium_target_share=float(args.medium_target_share),
        hard_bank_cap=hard_cap,
        cap_summaries=cap_summaries,
        hard_summary=hard_summary,
    )

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "caps": cap_summaries,
            "hard_diagnostic": hard_summary,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
