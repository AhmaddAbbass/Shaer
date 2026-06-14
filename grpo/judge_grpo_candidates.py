import argparse
import time
from pathlib import Path

from preprocess_grpo_pipeline_common import (
    DEFAULT_FIT_PROMPT,
    DEFAULT_SUBSTANCE_PROMPT,
    append_jsonl,
    compute_split_summary,
    ensure_dir,
    load_processed_source_indexes,
    load_runtime,
    pipeline_paths,
    publish_dataset,
    read_json,
    read_jsonl,
    release_claim,
    row_result_record,
    save_json,
    seed_done_markers,
    score_single_completion,
    setup_logger,
    try_mark_done,
    try_claim_row,
    utc_now_iso,
    write_status,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--worker-id", default="judge0")
    args = parser.parse_args()

    loaded = load_runtime()
    cfg = loaded["cfg"]
    runtime = loaded["runtime"]
    run_dir = ensure_dir(Path(args.run_dir))
    worker_id = str(args.worker_id)
    logger = setup_logger("judge_grpo_candidates", run_dir / f"judge_{worker_id}.log", also_stdout=True)

    save_json(
        {
            **runtime,
            "started_at": utc_now_iso(),
            "role": "judge",
            "publish": bool(args.publish),
            "worker_id": worker_id,
        },
        run_dir / f"judge_runtime_snapshot_{worker_id}.json",
    )

    fit_prompt_file = str((Path.cwd() / (Path(__file__).resolve().parent / DEFAULT_FIT_PROMPT)).resolve()) if not Path(DEFAULT_FIT_PROMPT).is_absolute() else DEFAULT_FIT_PROMPT
    substance_prompt_file = str((Path.cwd() / (Path(__file__).resolve().parent / DEFAULT_SUBSTANCE_PROMPT)).resolve()) if not Path(DEFAULT_SUBSTANCE_PROMPT).is_absolute() else DEFAULT_SUBSTANCE_PROMPT
    # Simpler and consistent with existing relative usage from grpo cwd:
    fit_prompt_file = "prompts/meaning_fit.yaml"
    substance_prompt_file = "prompts/meaning_substance.yaml"
    cache_dir = "./cache/reward_cache"
    reward_weights = cfg["phase1"]["reward_weights"]
    model_meta = read_json(run_dir / "model_meta.json") or {}
    status_path = run_dir / f"judge_status_{worker_id}.json"
    poll_seconds = int(cfg["preprocess_grpo_pipeline"].get("judge_poll_seconds", 30))
    claim_stale_after_seconds = int(cfg["preprocess_grpo_pipeline"].get("judge_claim_stale_after_seconds", 1800))
    seeded_done_splits: set[str] = set()

    split_order = ["train", "eval", "test"]
    while True:
        did_work = False
        generator_status = read_json(run_dir / "generator_status.json") or {}
        generator_done = bool(generator_status.get("done", False))

        for split_name in split_order:
            paths = pipeline_paths(run_dir, split_name)
            generated_rows = read_jsonl(paths["generated"])
            judged = load_processed_source_indexes(paths["judged"])
            if split_name not in seeded_done_splits:
                seed_done_markers(run_dir, split_name, judged)
                seeded_done_splits.add(split_name)
            available = [row for row in generated_rows if int(row["source_index"]) not in judged]

            if available:
                logger.info("split=%s available_to_judge=%d already_judged=%d", split_name, len(available), len(judged))

            for row in available:
                source_index = int(row["source_index"])
                claim = try_claim_row(
                    run_dir=run_dir,
                    split_name=split_name,
                    source_index=source_index,
                    worker_id=worker_id,
                    stale_after_seconds=claim_stale_after_seconds,
                )
                if claim is None:
                    continue
                runtime_state = {"started_at": utc_now_iso(), "t0": time.time()}
                done_marker = None
                try:
                    done_marker = try_mark_done(run_dir, split_name, source_index)
                    if done_marker is None:
                        judged.add(source_index)
                        continue
                    reward_rows = [
                        score_single_completion(
                            row=row,
                            completion_text=completion,
                            fit_prompt_file=fit_prompt_file,
                            substance_prompt_file=substance_prompt_file,
                            cache_dir=cache_dir,
                        )
                        for completion in row["sample_completions"]
                    ]
                    record = row_result_record(
                        row=row,
                        reward_rows=reward_rows,
                        reward_weights=reward_weights,
                        model_meta=model_meta,
                        runtime=runtime_state,
                    )
                    append_jsonl(paths["judged"], record)
                    judged.add(source_index)
                    did_work = True
                except Exception as exc:
                    append_jsonl(paths["judge_failures"], {"split": split_name, "source_index": source_index, "error": f"{type(exc).__name__}: {exc}", "timestamp": utc_now_iso()})
                    logger.exception("judge_failed split=%s source_index=%s", split_name, source_index)
                    if done_marker is not None:
                        try:
                            done_marker.unlink()
                        except FileNotFoundError:
                            pass
                finally:
                    release_claim(claim)

                if len(judged) % int(cfg["preprocess_grpo_pipeline"].get("progress_every", 10)) == 0:
                    write_status(
                        status_path,
                        {
                            "status": "running",
                            "worker_id": worker_id,
                            "split": split_name,
                            "judged_rows": len(judged),
                            "available_generated_rows": len(generated_rows),
                            "done": False,
                            "timestamp": utc_now_iso(),
                        },
                    )
                    logger.info("progress split=%s judged=%d available_generated=%d", split_name, len(judged), len(generated_rows))

        all_done = True
        for split_name in split_order:
            paths = pipeline_paths(run_dir, split_name)
            generated_count = len(read_jsonl(paths["generated"]))
            judged_count = len(read_jsonl(paths["judged"]))
            if generated_count == 0 or judged_count < generated_count:
                all_done = False
                break

        if generator_done and all_done:
            break

        if not did_work:
            write_status(
                status_path,
                {
                    "status": "waiting",
                    "worker_id": worker_id,
                    "done": False,
                    "generator_done": generator_done,
                    "timestamp": utc_now_iso(),
                },
            )
            time.sleep(poll_seconds)

    save_json(
        {
            "dataset_id": runtime["dataset_id"],
            "output_dataset_id": runtime["output_dataset_id"],
            "completed_at": utc_now_iso(),
            "splits": {split: compute_split_summary(pipeline_paths(run_dir, split)["judged"]) for split in split_order},
        },
        run_dir / "run_summary.json",
    )
    if args.publish:
        publish_dataset(run_dir, runtime["output_dataset_id"], logger)
    write_status(status_path, {"status": "completed", "worker_id": worker_id, "done": True, "completed_at": utc_now_iso()})
    logger.info("JUDGE_DONE run_dir=%s worker_id=%s", run_dir, worker_id)


if __name__ == "__main__":
    main()
