#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from judge_common import load_env, read_jsonl
from judge_llm import (
    DEFAULT_METRICS,
    JudgeClient,
    extract_judge_json,
    extract_usage,
    load_prompt_config,
    render_user_prompt,
    safe_int_score,
)


PROMPT_FILE = Path("evaluation/judge_prompts.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one shard of one dataset across all judge metrics.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prompt-file", default=str(PROMPT_FILE))
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    prompt_config = load_prompt_config(Path(args.prompt_file))
    metrics = list(args.metrics or DEFAULT_METRICS)
    rows = read_jsonl(Path(args.input_jsonl))
    shard_rows = [row for idx, row in enumerate(rows) if idx % args.worker_count == args.worker_index]
    output_path = Path(args.output_jsonl)
    progress_path = Path(args.progress_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    completed = read_completed_ids(output_path)
    client = JudgeClient(
        cache_dir=args.cache_dir,
        cache_namespace=f"worker_{args.worker_index}",
        model=args.judge_model,
        json_mode=True,
        max_tokens=128,
    )

    start_time = time.time()
    processed = 0
    newly_written = 0
    failures = 0
    total_cost = 0.0

    with output_path.open("a", encoding="utf-8") as handle:
        for row in shard_rows:
            row_id = str(row["judge_row_id"])
            if row_id in completed:
                processed += 1
                continue
            record, cost = evaluate_row(row=row, prompt_config=prompt_config, client=client, metrics=metrics)
            if has_errors(record):
                failures += 1
                log_event(
                    "row_failure",
                    worker_index=args.worker_index,
                    row_id=row_id,
                    errors=record["judge_metric_errors"],
                )
                if args.fail_on_error:
                    write_progress(
                        progress_path,
                        worker_index=args.worker_index,
                        worker_count=args.worker_count,
                        processed=processed,
                        completed=len(completed) + newly_written,
                        failures=failures,
                        total_cost_usd=total_cost,
                        elapsed_sec=time.time() - start_time,
                        status="failed",
                        failed_row_id=row_id,
                    )
                    raise RuntimeError(f"worker {args.worker_index} failed on row {row_id}")

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(row_id)
            processed += 1
            newly_written += 1
            total_cost += cost

            if newly_written == 1 or newly_written % 10 == 0:
                write_progress(
                    progress_path,
                    worker_index=args.worker_index,
                    worker_count=args.worker_count,
                    processed=processed,
                    completed=len(completed),
                    failures=failures,
                    total_cost_usd=total_cost,
                    elapsed_sec=time.time() - start_time,
                    status="running",
                )
                log_event(
                    "row_batch",
                    worker_index=args.worker_index,
                    processed=processed,
                    completed=len(completed),
                    failures=failures,
                    total_cost_usd=round(total_cost, 6),
                )

    write_progress(
        progress_path,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        processed=processed,
        completed=len(completed),
        failures=failures,
        total_cost_usd=total_cost,
        elapsed_sec=time.time() - start_time,
        status="completed",
    )
    log_event(
        "worker_done",
        worker_index=args.worker_index,
        processed=processed,
        completed=len(completed),
        failures=failures,
        total_cost_usd=round(total_cost, 6),
        elapsed_sec=round(time.time() - start_time, 3),
    )
    return 0


def evaluate_row(
    *,
    row: dict[str, Any],
    prompt_config: dict[str, Any],
    client: JudgeClient,
    metrics: list[str],
) -> tuple[dict[str, Any], float]:
    metric_scores: dict[str, int] = {}
    metric_details: dict[str, dict[str, Any]] = {}
    metric_errors: dict[str, str] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    total_latency_sec = 0.0

    for metric in metrics:
        metric_cfg = prompt_config["metrics"][metric]
        user_prompt = render_user_prompt(
            metric_cfg,
            poem=str(row.get("judge_poem") or ""),
            description=str(row.get("judge_description") or ""),
        )
        response_obj, cache_hit = client.judge(prompt_config["shared_system_prompt"], user_prompt)
        parsed = extract_judge_json(response_obj, score_key="score")
        usage = extract_usage(response_obj.get("response"))
        score = safe_int_score(parsed.get("score"))
        error = str(parsed.get("error") or "")
        metric_scores[metric] = score if score > 0 else None
        metric_details[metric] = {
            "score": score,
            "cache_hit": bool(cache_hit),
            "error": error,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "reported_cost_usd": float(usage.get("reported_cost_usd") or 0.0),
            "latency_sec": float(response_obj.get("latency_sec") or 0.0),
        }
        if error or score == 0:
            metric_errors[metric] = error or "score_zero"
        total_prompt_tokens += metric_details[metric]["prompt_tokens"]
        total_completion_tokens += metric_details[metric]["completion_tokens"]
        total_cost += metric_details[metric]["reported_cost_usd"]
        total_latency_sec += metric_details[metric]["latency_sec"]

    record = dict(row)
    record.update(metric_scores)
    record["judge_model"] = client.model
    record["judge_prompt_version"] = str(prompt_config.get("version") or "")
    record["judge_metrics"] = list(metrics)
    record["judge_metric_details"] = metric_details
    record["judge_metric_errors"] = metric_errors
    record["judge_total_prompt_tokens"] = total_prompt_tokens
    record["judge_total_completion_tokens"] = total_completion_tokens
    record["judge_total_tokens"] = total_prompt_tokens + total_completion_tokens
    record["judge_total_cost_usd"] = total_cost
    record["judge_total_latency_sec"] = total_latency_sec
    record["judge_evaluated_at_unix"] = time.time()
    return record, total_cost


def has_errors(record: dict[str, Any]) -> bool:
    return bool(record.get("judge_metric_errors"))


def read_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            row_id = str(row.get("judge_row_id") or "")
            if row_id:
                ids.add(row_id)
    return ids


def write_progress(progress_path: Path, **payload: Any) -> None:
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log_event(event: str, **payload: Any) -> None:
    data = {"event": event, "ts": time.time(), **payload}
    print(json.dumps(data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"event": "fatal_error", "ts": time.time(), "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise
