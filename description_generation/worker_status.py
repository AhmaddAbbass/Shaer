#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local worker progress for final SFT regeneration")
    parser.add_argument("--workers-manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(Path(args.workers_manifest))
    worker_rows = manifest["worker_shards"]
    all_indices: set[int] = set()
    duplicates: list[int] = []
    rows: list[dict[str, Any]] = []

    for worker in worker_rows:
        output_dir = Path(worker["output_dir"])
        results_path = output_dir / "results.jsonl"
        summary_path = output_dir / "summary.json"
        results = load_jsonl(results_path)
        summary = load_json(summary_path) if summary_path.exists() else {}
        local_indices: list[int] = []
        ok_rows = 0
        error_rows = 0
        for rec in results:
            idx = int(rec["source_index"])
            local_indices.append(idx)
            if idx in all_indices:
                duplicates.append(idx)
            all_indices.add(idx)
            if rec.get("status") == "ok" and str(rec.get("enhanced_description", "")).strip():
                ok_rows += 1
            else:
                error_rows += 1
        rows.append(
            {
                "worker_id": worker["worker_id"],
                "assigned_rows": worker["row_count"],
                "completed_rows": len(results),
                "ok_rows": ok_rows,
                "error_rows": error_rows,
                "pending_rows": max(worker["row_count"] - len(results), 0),
                "source_index_min": worker["source_index_min"],
                "source_index_max": worker["source_index_max"],
                "sync_count": summary.get("sync_count"),
                "skipped_existing": summary.get("skipped_existing"),
            }
        )

    completed = sum(row["completed_rows"] for row in rows)
    assigned = sum(row["assigned_rows"] for row in rows)
    payload = {
        "run_name": manifest["run_name"],
        "repo_id": manifest["repo_id"],
        "assigned_rows": assigned,
        "completed_rows": completed,
        "pending_rows": max(assigned - completed, 0),
        "worker_count": len(rows),
        "has_duplicate_source_indices": bool(duplicates),
        "duplicate_source_indices": sorted(set(duplicates))[:20],
        "workers": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
