#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from preprocess_meter_count_common import (
    all_generators_done,
    append_jsonl,
    atomic_write_json,
    ensure_dir,
    event_log_path,
    failure_log_path,
    generated_path,
    load_manifest,
    load_dotenv_if_present,
    read_json_or_none,
    release_claim,
    row_score_summary,
    scored_path,
    status_path,
    try_claim_score,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated candidates with meter and count adherence.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--worker-id", default="score00")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--claim-stale-after-seconds", type=int, default=1800)
    parser.add_argument("--exit-when-generators-done", action="store_true")
    parser.add_argument("--oneshot", action="store_true")
    return parser.parse_args()


def configure_cpu_threads() -> None:
    torch_threads = int(os.getenv("GRPO_MC_TORCH_NUM_THREADS", "1") or 1)
    interop_threads = int(os.getenv("GRPO_MC_TORCH_INTEROP_THREADS", "1") or 1)
    torch_threads = max(1, torch_threads)
    interop_threads = max(1, interop_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(torch_threads))
    try:
        import torch

        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass


def meter_scoring_text(completion: str, requested_bayts: int) -> tuple[str, dict[str, Any]]:
    lines = [line.strip() for line in str(completion or "").replace("\r", "\n").split("\n") if line.strip()]
    complete_bayts = len(lines) // 2
    extra_bayts = int(os.getenv("GRPO_MC_METER_SCORE_EXTRA_BAYTS", "4") or 4)
    hard_cap = int(os.getenv("GRPO_MC_METER_SCORE_MAX_BAYTS", "24") or 24)
    cap = max(int(requested_bayts), int(requested_bayts) + extra_bayts)
    if hard_cap > 0:
        cap = min(cap, max(int(requested_bayts), hard_cap))
    if complete_bayts <= cap:
        return completion, {
            "meter_scoring_truncated": False,
            "meter_scoring_bayts_limit": int(cap),
            "meter_scoring_input_complete_bayts": int(complete_bayts),
            "meter_scoring_complete_bayts_used": int(complete_bayts),
        }
    keep_lines = lines[: int(cap) * 2]
    return "\n".join(keep_lines), {
        "meter_scoring_truncated": True,
        "meter_scoring_bayts_limit": int(cap),
        "meter_scoring_input_complete_bayts": int(complete_bayts),
        "meter_scoring_complete_bayts_used": int(cap),
    }


def score_candidate(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    from rewards.common import score_count_adherence
    from rewards.meter import score_meter_poem

    completion = str(candidate.get("completion") or "")
    count = score_count_adherence(int(row["requested_bayts"]), completion)
    meter_text, meter_text_meta = meter_scoring_text(completion, int(row["requested_bayts"]))
    meter = score_meter_poem(meter_text, row["meter_label"], base_meter=row["base_meter"], aggregator="logmean")
    return {
        **candidate,
        "meter_score": float(meter["score"]),
        "meter_mean_score": float(meter["mean_score"]),
        "meter_logmean_score": float(meter["logmean_score"]),
        "meter_target_used": str(meter["target_meter_used"]),
        "meter_target_resolution": str(meter["target_resolution"]),
        "meter_num_valid_bayts": int(meter["num_valid_bayts"]),
        "meter_num_skipped_bayts": int(meter["num_skipped_bayts"]),
        "meter_num_lines": int(meter["num_lines"]),
        "meter_complete_bayts": int(meter["complete_bayts"]),
        "meter_has_odd_tail": bool(meter["has_odd_tail"]),
        "per_bayt_meter_scores": [float(x) for x in meter["per_bayt_scores"]],
        "per_bayt_meter_details": meter["per_bayt_details"],
        "count_adherence_score": float(count["score"]),
        "generated_bayts": int(count["generated_bayts"]),
        "count_requested_bayts": int(count["requested_bayts"]),
        "count_has_odd_tail": bool(count.get("has_odd_tail", False)),
        "count_num_lines": int(count.get("num_lines", 0)),
        **meter_text_meta,
        "scored_at": utc_now_iso(),
    }


def write_scored(run_dir: Path, generated: dict[str, Any], worker_id: str) -> dict[str, Any]:
    row = dict(generated["row"])
    t0 = time.time()
    candidate_scores = [score_candidate(row, candidate) for candidate in generated["candidates"]]
    summary = row_score_summary(candidate_scores)
    payload = {
        "row_uid": generated["row_uid"],
        "source_dataset_id": row.get("source_dataset_id", ""),
        "source_split": row["source_split"],
        "source_index": int(row["source_index"]),
        "source_row_index_in_split": int(row["source_row_index_in_split"]),
        "source_id": row.get("source_id", ""),
        "row_preprocess_status": "scored",
        "difficulty": summary["difficulty"],
        "base_meter": row["base_meter"],
        "form": row["form"],
        "meter_label": row["meter_label"],
        "requested_bayts": int(row["requested_bayts"]),
        "requested_lines": int(row["requested_lines"]),
        "length_bucket": row.get("length_bucket", ""),
        "sampler_group": row.get("sampler_group", ""),
        "split_group": row.get("split_group", ""),
        "description": row.get("description", ""),
        "enhanced_description": row.get("enhanced_description", ""),
        "sft_prompt": row.get("sft_prompt", ""),
        "poem_url": row.get("poem_url", ""),
        "num_candidates_requested": int(generated["num_candidates_requested"]),
        "num_candidates_generated": int(generated["num_candidates_generated"]),
        "num_candidates_scored": len(candidate_scores),
        "candidate_scores": candidate_scores,
        "meter_summary": summary["meter_summary"],
        "count_adherence_summary": summary["count_adherence_summary"],
        "meter_count_combined_summary": summary["meter_count_combined_summary"],
        "meter_count_success_rate": float(summary["meter_count_success_rate"]),
        "count_exact_rate": float(summary["count_exact_rate"]),
        "num_strong_meter_candidates": int(summary["num_strong_meter_candidates"]),
        "num_strong_exact_candidates": int(summary["num_strong_exact_candidates"]),
        "num_bad_meter_candidates": int(summary["num_bad_meter_candidates"]),
        "strong_meter_rate": float(summary["strong_meter_rate"]),
        "strong_exact_rate": float(summary["strong_exact_rate"]),
        "bad_meter_rate": float(summary["bad_meter_rate"]),
        "mean_meter_score": float(summary["mean_meter_score"]),
        "mean_count_adherence_score": float(summary["mean_count_adherence_score"]),
        "difficulty_rule": summary["difficulty_rule"],
        "generator_shard_id": int(generated["generator_shard_id"]),
        "manifest_order": int(generated["manifest_order"]),
        "generator_metadata": generated["generator_metadata"],
        "scoring_metadata": {
            "worker_id": worker_id,
            "pid": os.getpid(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "runtime_seconds": time.time() - t0,
            "scored_at": utc_now_iso(),
        },
    }
    atomic_write_json(scored_path(run_dir, payload["row_uid"]), payload)
    return payload


def generated_ready_rows(run_dir: Path, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = []
    for row in manifest:
        uid = str(row["row_uid"])
        if scored_path(run_dir, uid).exists():
            continue
        gen = read_json_or_none(generated_path(run_dir, int(row["generator_shard_id"]), uid))
        if gen is not None:
            ready.append(gen)
    return ready


def main() -> int:
    load_dotenv_if_present()
    configure_cpu_threads()
    args = parse_args()
    run_dir = ensure_dir(args.run_dir)
    worker_id = args.worker_id
    manifest = load_manifest(run_dir)
    processed_this_worker = 0
    candidates_this_worker = 0
    t0_run = time.time()
    atomic_write_json(status_path(run_dir, f"score_worker_{worker_id}"), {"status": "starting", "worker_id": worker_id, "started_at": utc_now_iso()})

    while True:
        did_work = False
        ready = generated_ready_rows(run_dir, manifest)
        for generated in ready:
            uid = str(generated["row_uid"])
            if scored_path(run_dir, uid).exists():
                continue
            claim = try_claim_score(run_dir, uid, worker_id, args.claim_stale_after_seconds)
            if claim is None:
                continue
            try:
                if scored_path(run_dir, uid).exists():
                    continue
                payload = write_scored(run_dir, generated, worker_id)
                processed_this_worker += 1
                candidates_this_worker += int(payload["num_candidates_scored"])
                did_work = True
                append_jsonl(
                    event_log_path(run_dir, f"score_worker_{worker_id}"),
                    {
                        "event": "row_scored",
                        "timestamp": utc_now_iso(),
                        "worker_id": worker_id,
                        "row_uid": uid,
                        "candidates": int(payload["num_candidates_scored"]),
                        "runtime_seconds": float(payload["scoring_metadata"]["runtime_seconds"]),
                        "difficulty": payload["difficulty"],
                    },
                )
            except Exception as exc:
                append_jsonl(
                    failure_log_path(run_dir, f"score_worker_{worker_id}"),
                    {
                        "event": "score_failed",
                        "timestamp": utc_now_iso(),
                        "worker_id": worker_id,
                        "row_uid": uid,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
            finally:
                release_claim(claim)

            atomic_write_json(
                status_path(run_dir, f"score_worker_{worker_id}"),
                {
                    "status": "running",
                    "worker_id": worker_id,
                    "processed_this_worker": processed_this_worker,
                    "candidates_this_worker": candidates_this_worker,
                    "updated_at": utc_now_iso(),
                    "run_seconds": time.time() - t0_run,
                },
            )

        if args.oneshot:
            break
        if args.exit_when_generators_done and all_generators_done(run_dir) and not generated_ready_rows(run_dir, manifest):
            break
        if not did_work:
            atomic_write_json(
                status_path(run_dir, f"score_worker_{worker_id}"),
                {
                    "status": "waiting",
                    "worker_id": worker_id,
                    "processed_this_worker": processed_this_worker,
                    "candidates_this_worker": candidates_this_worker,
                    "all_generators_done": all_generators_done(run_dir),
                    "updated_at": utc_now_iso(),
                    "run_seconds": time.time() - t0_run,
                },
            )
            time.sleep(max(0.25, float(args.poll_seconds)))

    atomic_write_json(
        status_path(run_dir, f"score_worker_{worker_id}"),
        {
            "status": "completed",
            "worker_id": worker_id,
            "processed_this_worker": processed_this_worker,
            "candidates_this_worker": candidates_this_worker,
            "completed_at": utc_now_iso(),
            "run_seconds": time.time() - t0_run,
        },
    )
    print(f"SCORE_WORKER_DONE worker_id={worker_id} rows={processed_this_worker} candidates={candidates_this_worker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
