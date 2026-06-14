import argparse
from datetime import datetime
from pathlib import Path

from preprocess_grpo_pipeline_common import (
    append_jsonl,
    build_vllm_engine,
    generate_batch_k_vllm,
    iter_meter_balanced_pending_rows,
    load_processed_source_indexes,
    load_runtime,
    load_and_prepare_dataset,
    maybe_limit_dataset,
    pending_meter_counts,
    pipeline_paths,
    save_json,
    setup_logger,
    utc_now_iso,
    write_status,
    ensure_dir,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    args = parser.parse_args()

    loaded = load_runtime()
    cfg = loaded["cfg"]
    runtime = loaded["runtime"]
    run_dir = ensure_dir(Path(args.run_dir))
    logger = setup_logger("generate_grpo_candidates", run_dir / "generator.log", also_stdout=True)

    prep_cfg = cfg["preprocess_grpo_pipeline"]
    batch_size = int(prep_cfg.get("generator_batch_size", 16))
    save_json(cfg, run_dir / "config_snapshot.json")
    save_json(
        {
            **runtime,
            "started_at": utc_now_iso(),
            "max_rows_per_split": args.max_rows_per_split,
            "role": "generator",
        },
        run_dir / "generator_runtime_snapshot.json",
    )

    llm, lora_request, model_meta = build_vllm_engine(cfg)
    save_json(model_meta, run_dir / "model_meta.json")
    logger.info(
        "Resolved adapter | repo=%s mode=%s path=%s",
        runtime["sft_adapter_repo"],
        runtime["sft_adapter_mode"],
        model_meta.get("adapter_path", ""),
    )

    split_map = {"train": "train", "eval": "eval", "test": "test"}
    status_path = run_dir / "generator_status.json"
    for split_name, source_split in split_map.items():
        ds = load_and_prepare_dataset(dataset_id=runtime["dataset_id"], split=source_split, hf_token=None)
        ds = maybe_limit_dataset(ds, args.max_rows_per_split)
        ds = ds.map(lambda row: {**row, "dataset_split": split_name})
        paths = pipeline_paths(run_dir, split_name)
        generated = load_processed_source_indexes(paths["generated"])
        total_rows = len(ds)
        meter_counts = pending_meter_counts(ds, generated)
        logger.info(
            "split=%s total_rows=%d already_generated=%d pending_rows=%d pending_meters=%d order=meter_round_robin",
            split_name,
            total_rows,
            len(generated),
            sum(meter_counts.values()),
            len(meter_counts),
        )
        logger.info("split=%s pending_meter_counts=%s", split_name, meter_counts)
        write_status(status_path, {"status": "running", "split": split_name, "generated_rows": len(generated), "total_rows": total_rows, "done": False})

        batch_rows = []

        def flush_batch(rows_batch):
            nonlocal generated
            if not rows_batch:
                return
            prompts = [row["prompt"] for row in rows_batch]
            completions_batch = generate_batch_k_vllm(
                llm=llm,
                lora_request=lora_request,
                prompts=prompts,
                k=int(prep_cfg["num_generations"]),
                max_new_tokens=int(prep_cfg["max_completion_length"]),
                temperature=float(prep_cfg["temperature"]),
                top_p=float(prep_cfg["top_p"]),
            )
            for row, completions in zip(rows_batch, completions_batch):
                append_jsonl(
                    paths["generated"],
                    {
                        "source_index": int(row["source_index"]),
                        "dataset_split": split_name,
                        "prompt": row["prompt"],
                        "description": row["description"],
                        "base_meter": row["base_meter"],
                        "form": row.get("form", ""),
                        "meter_label": row["meter_label"],
                        "requested_bayts": int(row["requested_bayts"]),
                        "requested_lines": int(row["requested_lines"]),
                        "length_bucket": row["length_bucket"],
                        "sample_completions": completions,
                        "num_generations": len(completions),
                        "generated_at": utc_now_iso(),
                    },
                )
                generated.add(int(row["source_index"]))

                if len(generated) % int(prep_cfg.get("progress_every", 10)) == 0:
                    write_status(
                        status_path,
                        {
                            "status": "running",
                            "split": split_name,
                            "generated_rows": len(generated),
                            "total_rows": total_rows,
                            "remaining_rows": max(0, total_rows - len(generated)),
                            "done": False,
                            "timestamp": utc_now_iso(),
                        },
                    )
                    logger.info("progress split=%s generated=%d/%d remaining=%d", split_name, len(generated), total_rows, max(0, total_rows - len(generated)))

        for row in iter_meter_balanced_pending_rows(ds, generated):
            source_index = int(row["source_index"])
            batch_rows.append(row)
            if len(batch_rows) < batch_size:
                continue
            try:
                flush_batch(batch_rows)
                batch_rows = []
            except Exception as exc:
                append_jsonl(paths["generated_failures"], {"split": split_name, "source_index": source_index, "error": f"{type(exc).__name__}: {exc}", "timestamp": utc_now_iso()})
                logger.exception("generation_failed split=%s source_index=%s batch_size=%d", split_name, source_index, len(batch_rows))
                batch_rows = []

        if batch_rows:
            try:
                flush_batch(batch_rows)
            except Exception as exc:
                for row in batch_rows:
                    append_jsonl(paths["generated_failures"], {"split": split_name, "source_index": int(row["source_index"]), "error": f"{type(exc).__name__}: {exc}", "timestamp": utc_now_iso()})
                logger.exception("generation_failed split=%s final_batch_size=%d", split_name, len(batch_rows))

    write_status(status_path, {"status": "completed", "done": True, "completed_at": utc_now_iso()})
    logger.info("GENERATOR_DONE run_dir=%s", run_dir)


if __name__ == "__main__":
    main()
