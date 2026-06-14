#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess_meter_count_common import ensure_dir, read_json_or_none, scored_path, utc_now_iso


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def summarize(run_dir: Path, output_dir: Path) -> dict:
    rows = []
    for path in sorted((run_dir / "scored").glob("*.json")):
        scored = read_json_or_none(path)
        if scored is None:
            continue
        for cand in scored["candidate_scores"]:
            rows.append(
                {
                    "row_uid": scored["row_uid"],
                    "source_index": int(scored["source_index"]),
                    "base_meter": scored["base_meter"],
                    "requested_bayts": int(scored["requested_bayts"]),
                    "candidate_id": cand["candidate_id"],
                    "meter_score": float(cand["meter_score"]),
                    "count_adherence_score": float(cand["count_adherence_score"]),
                    "generated_bayts": int(cand["generated_bayts"]),
                    "completion_preview": str(cand["completion"]).replace("\n", " / ")[:500],
                }
            )

    by_meter = defaultdict(list)
    for row in rows:
        by_meter[row["base_meter"]].append(row)

    summary_rows = []
    for meter in sorted(by_meter):
        items = by_meter[meter]
        meter_scores = [float(row["meter_score"]) for row in items]
        count_scores = [float(row["count_adherence_score"]) for row in items]
        summary_rows.append(
            {
                "base_meter": meter,
                "candidate_count": len(items),
                "meter_mean": sum(meter_scores) / len(meter_scores) if meter_scores else 0.0,
                "meter_max": max(meter_scores) if meter_scores else 0.0,
                "count_mean": sum(count_scores) / len(count_scores) if count_scores else 0.0,
                "count_exact_rate": sum(score >= 0.999999 for score in count_scores) / len(count_scores) if count_scores else 0.0,
            }
        )

    ensure_dir(output_dir)
    results_csv = output_dir / "candidate_results.csv"
    summary_csv = output_dir / "per_base_meter_summary.csv"
    examples_jsonl = output_dir / "examples.jsonl"
    if rows:
        with results_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with examples_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if summary_rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    summary = {
        "timestamp": utc_now_iso(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "scored_candidates": len(rows),
        "base_meters": len(by_meter),
        "candidate_results_csv": str(results_csv),
        "per_base_meter_summary_csv": str(summary_csv),
        "examples_jsonl": str(examples_jsonl),
        "per_base_meter_summary": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick baseline sanity: 5 train rows per base meter by meter/count.")
    parser.add_argument("--dataset-id", default=os.getenv("GRPO_PREPROCESS_DATASET_ID", "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"))
    parser.add_argument("--rows-per-base-meter", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", default="grpo/outputs/meter_count_baseline_sanity/latest")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--generator-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    py = sys.executable
    run_dir = Path(args.run_dir or Path(args.output_dir) / "pipeline_run").resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(run_dir)
    ensure_dir(output_dir)

    run(
        [
            py,
            str(ROOT / "prepare_meter_count_manifest.py"),
            "--dataset-id",
            args.dataset_id,
            "--source-split",
            "train",
            "--run-dir",
            str(run_dir),
            "--balanced-rows-per-base-meter",
            str(args.rows_per_base_meter),
            "--num-generator-shards",
            "1",
            "--num-candidates",
            str(args.num_candidates),
        ]
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(
        [
            py,
            str(ROOT / "generate_meter_count_candidates.py"),
            "--run-dir",
            str(run_dir),
            "--shard-id",
            "0",
            "--generator-id",
            "sanity_gen0",
            "--batch-size",
            str(args.generator_batch_size),
        ],
        env=env,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["METER_DEVICE"] = env.get("METER_DEVICE", "cpu")
    run(
        [
            py,
            str(ROOT / "score_meter_count_candidates.py"),
            "--run-dir",
            str(run_dir),
            "--worker-id",
            "sanity_score0",
            "--exit-when-generators-done",
        ],
        env=env,
    )
    summary = summarize(run_dir, output_dir)
    expected_rows = int(args.rows_per_base_meter) * summary["base_meters"]
    expected_candidates = expected_rows * int(args.num_candidates)
    if summary["scored_candidates"] != expected_candidates:
        raise RuntimeError(f"expected {expected_candidates} scored candidates, got {summary['scored_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
