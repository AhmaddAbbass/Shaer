#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

from judge_dataset_registry import DATASET_ORDER, DATASET_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full paid judge evaluation dataset by dataset with worker sharding.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge-model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--datasets", nargs="*", default=DATASET_ORDER)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--allow-row-errors", action="store_true")
    parser.add_argument("--prompt-file", default="evaluation/judge_prompts.yaml")
    parser.add_argument("--prepared-input-root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    main_log = run_dir / "main.log"

    log(main_log, f"run_dir={run_dir} workers={args.workers} judge_model={args.judge_model} datasets={','.join(args.datasets)}")

    for dataset_key in args.datasets:
        dataset_dir = run_dir / dataset_key
        (dataset_dir / "logs").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "workers").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "cache").mkdir(parents=True, exist_ok=True)
        input_jsonl = dataset_dir / "input_rows.jsonl"
        input_summary = dataset_dir / "input_rows.summary.json"
        merged_jsonl = dataset_dir / "scored_rows.jsonl"
        merged_summary = dataset_dir / "scored_rows.summary.json"
        hf_cache_root = os.environ.get("JUDGE_HF_CACHE_ROOT", "").strip()
        if hf_cache_root:
            hf_cache_dir = Path(hf_cache_root) / dataset_key
        elif os.name == "nt":
            hf_cache_dir = Path("C:/sjc") / dataset_key
        else:
            hf_cache_dir = Path(tempfile.gettempdir()) / "shaer_judge_hf_cache" / dataset_key
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

        prepared_root = Path(args.prepared_input_root).expanduser().resolve() if args.prepared_input_root else None
        prepared_input_jsonl = prepared_root / dataset_key / "input_rows.jsonl" if prepared_root else None
        prepared_input_summary = prepared_root / dataset_key / "input_rows.summary.json" if prepared_root else None
        if prepared_input_jsonl and prepared_input_jsonl.exists() and prepared_input_summary and prepared_input_summary.exists():
            log(main_log, f"using prepared input dataset={dataset_key} from={prepared_input_jsonl}")
            input_jsonl.write_text(prepared_input_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
            input_summary.write_text(prepared_input_summary.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            log(main_log, f"materializing dataset={dataset_key}")
            run_checked(
                [
                    sys.executable,
                    "evaluation/materialize_judge_dataset.py",
                    "--dataset-key",
                    dataset_key,
                    "--output-jsonl",
                    str(input_jsonl),
                    "--summary-json",
                    str(input_summary),
                    "--cache-dir",
                    str(hf_cache_dir),
                ],
                log_path=main_log,
            )
        expected_rows = json.loads(input_summary.read_text(encoding="utf-8"))["rows"]
        dataset_metrics = list(DATASET_SPECS[dataset_key].get("judge_metrics") or [])
        if args.limit_rows:
            expected_rows = min(expected_rows, int(args.limit_rows))
            truncate_input(input_jsonl, expected_rows)
        log(main_log, f"dataset={dataset_key} expected_rows={expected_rows} metrics={','.join(dataset_metrics)}")

        worker_procs: list[subprocess.Popen] = []
        for worker_index in range(int(args.workers)):
            worker_jsonl = dataset_dir / "workers" / f"worker_{worker_index}.jsonl"
            worker_progress = dataset_dir / "workers" / f"worker_{worker_index}.progress.json"
            worker_log = dataset_dir / "logs" / f"worker_{worker_index}.log"
            cache_dir = dataset_dir / "cache"
            cmd = [
                sys.executable,
                "-u",
                "evaluation/full_judge_worker.py",
                "--input-jsonl",
                str(input_jsonl),
                "--output-jsonl",
                str(worker_jsonl),
                "--progress-json",
                str(worker_progress),
                "--worker-index",
                str(worker_index),
                "--worker-count",
                str(args.workers),
                "--judge-model",
                args.judge_model,
                "--cache-dir",
                str(cache_dir),
                "--prompt-file",
                str(args.prompt_file),
                "--metrics",
                *dataset_metrics,
            ]
            if not args.allow_row_errors:
                cmd.append("--fail-on-error")
            log(main_log, f"launch dataset={dataset_key} worker={worker_index}/{args.workers}")
            log_handle = worker_log.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            proc._log_handle = log_handle  # type: ignore[attr-defined]
            worker_procs.append(proc)

        status = 0
        for proc in worker_procs:
            rc = proc.wait()
            proc._log_handle.close()  # type: ignore[attr-defined]
            if rc != 0:
                status = rc or 1
        if status != 0:
            log(main_log, f"dataset={dataset_key} status=failed rc={status}")
            return status

        log(main_log, f"merging dataset={dataset_key}")
        run_checked(
            [
                sys.executable,
                "evaluation/merge_full_judge_results.py",
                "--workers-dir",
                str(dataset_dir / "workers"),
                "--expected-rows",
                str(expected_rows),
                "--output-jsonl",
                str(merged_jsonl),
                "--summary-json",
                str(merged_summary),
            ],
            log_path=main_log,
        )
        log(main_log, f"dataset={dataset_key} status=completed merged_jsonl={merged_jsonl}")

    log(main_log, "all_datasets_completed")
    return 0


def truncate_input(input_jsonl: Path, keep_rows: int) -> None:
    truncated_path = input_jsonl.with_suffix(".truncated.tmp")
    rows = input_jsonl.read_text(encoding="utf-8").splitlines()
    subset = [line for line in rows[:keep_rows] if line.strip()]
    truncated_path.write_text("\n".join(subset) + ("\n" if subset else ""), encoding="utf-8")
    truncated_path.replace(input_jsonl)


def run_checked(cmd: list[str], *, log_path: Path) -> None:
    log(log_path, "run " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def log(log_path: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
