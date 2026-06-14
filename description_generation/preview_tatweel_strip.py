#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import load_dataset


DEFAULT_SOURCE_DATASET = "Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed"
TATWEEL = "\u0640"


def normalize_poem_verses(verses: Any) -> list[str]:
    if not isinstance(verses, list):
        return []
    out: list[str] = []
    for item in verses:
        text = str(item).strip()
        if not text:
            return []
        out.append(text)
    return out


def strip_tatweel(text: str) -> str:
    return str(text).replace(TATWEEL, "")


def iter_rows(dataset_name: str) -> Any:
    ds = load_dataset(dataset_name, split="train")
    for row in ds:
        yield row


def find_row(dataset_name: str, row_id: int) -> dict[str, Any]:
    for row in iter_rows(dataset_name):
        if int(row.get("id", -1)) == row_id:
            return dict(row)
    raise SystemExit(f"row id {row_id} not found in {dataset_name}")


def build_preview(row: dict[str, Any]) -> dict[str, Any]:
    verses = normalize_poem_verses(row.get("poem verses"))
    verse_items: list[dict[str, Any]] = []
    boundary_warnings: list[dict[str, Any]] = []
    for index, verse in enumerate(verses):
        stripped = strip_tatweel(verse)
        verse_items.append(
            {
                "index": index,
                "original": verse,
                "stripped": stripped,
                "tatweel_count": verse.count(TATWEEL),
                "starts_with_tatweel": verse.startswith(TATWEEL),
                "ends_with_tatweel": verse.endswith(TATWEEL),
            }
        )
        if index == 0:
            continue
        left = verses[index - 1]
        right = verse
        if left.endswith(TATWEEL) or right.startswith(TATWEEL):
            boundary_warnings.append(
                {
                    "left_index": index - 1,
                    "right_index": index,
                    "left_original": left,
                    "right_original": right,
                    "left_stripped": strip_tatweel(left),
                    "right_stripped": strip_tatweel(right),
                    "warning": "Removing tatweel alone will not reconnect a word split across hemistich boundaries.",
                }
            )

    bayts: list[dict[str, Any]] = []
    for index in range(0, len(verses) - 1, 2):
        left = verses[index]
        right = verses[index + 1]
        bayts.append(
            {
                "bayt_index": index // 2,
                "original": f"{left} // {right}",
                "stripped": f"{strip_tatweel(left)} // {strip_tatweel(right)}",
            }
        )

    return {
        "id": row.get("id"),
        "base_meter": str(row.get("base_meter", "")).strip(),
        "form": str(row.get("form", "")).strip(),
        "verse_count": len(verses),
        "tatweel_removed_total": sum(item["tatweel_count"] for item in verse_items),
        "has_boundary_split_risk": bool(boundary_warnings),
        "verses": verse_items,
        "boundary_warnings": boundary_warnings,
        "bayts": bayts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview deterministic tatweel stripping for one poem.")
    parser.add_argument("--row-id", type=int, required=True, help="Dataset row id to inspect.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_SOURCE_DATASET,
        help=f"Hugging Face dataset id (default: {DEFAULT_SOURCE_DATASET})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = find_row(args.dataset, args.row_id)
    preview = build_preview(row)
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
