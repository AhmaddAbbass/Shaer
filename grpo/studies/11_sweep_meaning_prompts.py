import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import (
    ensure_dir,
    append_jsonl,
    get_dataset_id_for_phase,
    load_and_prepare_dataset,
    load_yaml,
    parse_allowed_meters_env,
    save_json,
    setup_logger,
)
from rewards.meaning import score_meaning

BENCHMARK_CSV = "fixed_gold_benchmark.csv"
BENCHMARK_META_JSON = "fixed_gold_benchmark_metadata.json"


def make_bad_variants(poem_text: str):
    lines = [x for x in str(poem_text).split("\n") if x.strip()]
    out = []

    if len(lines) >= 2:
        out.append(("bad_truncate_half", "\n".join(lines[: max(1, len(lines) // 2)])))
        out.append(("bad_reverse_lines", "\n".join(list(reversed(lines)))))

    if lines:
        out.append(("bad_duplicate_line", "\n".join([lines[0]] * min(4, len(lines)))))

    return out


def discover_prompt_files():
    env_value = os.getenv("MEANING_PROMPT_CANDIDATES", "").strip()
    if env_value:
        return [x.strip() for x in env_value.split(",") if x.strip()]

    prompt_dir = Path("prompts")
    files = []
    for path in sorted(prompt_dir.glob("meaning*.yaml")):
        if path.stem in {"meaning_fit", "meaning_substance"}:
            continue
        files.append(str(path))
    if not files:
        raise ValueError(
            "No legacy merged meaning prompt files found under prompts/. "
            "This sweep is intentionally quarantined from the active split-reward prompts."
        )
    return files


def validate_prompt_files(prompt_files):
    records = []
    for path in prompt_files:
        cfg = load_yaml(path)
        if "system" not in cfg or "user_template" not in cfg:
            raise ValueError(f"Prompt file missing required keys: {path}")
        records.append({
            "prompt_file": path,
            "system_preview": str(cfg["system"])[:200],
            "user_template_preview": str(cfg["user_template"])[:200],
        })
    return records


def prompt_tag(path: str, rank: int) -> str:
    stem = Path(path).stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{rank:02d}_{safe}"


def select_fixed_sample(ds, n: int, seed: int, root_dir: Path, strategy: str, reuse_saved_sample: bool, logger):
    manifest_path = root_dir / "sample_source_indices.json"

    if reuse_saved_sample and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        indices = [int(x) for x in manifest.get("source_indices", [])]
        target_set = set(indices)
        found = {}
        logger.info(f"attempting to reuse fixed sample of {len(indices)} poems from {manifest_path}")
        for i, row in enumerate(ds, start=1):
            src = int(row["source_index"])
            if src in target_set:
                found[src] = row
                if len(found) == len(target_set):
                    break
            if i % 20000 == 0:
                logger.info(f"reuse_sample_scan_rows={i} found={len(found)}")
        selected = [found[idx] for idx in indices if idx in found]
        if len(selected) == len(indices) and selected:
            logger.info(f"reusing fixed sample from {manifest_path}")
            return selected, manifest
        logger.info("saved fixed sample could not be fully reused; rebuilding")

    meter_groups = {}
    logger.info(f"building new fixed sample n={n} strategy={strategy}")
    for i, row in enumerate(ds, start=1):
        meter_groups.setdefault(str(row["base_meter"]), []).append(row)
        if i % 20000 == 0:
            logger.info(f"build_sample_rows={i} base_meters_seen={len(meter_groups)}")

    rng = random.Random(seed)
    for rows in meter_groups.values():
        rng.shuffle(rows)

    if strategy == "balanced_by_base_meter":
        meter_order = sorted(meter_groups.keys(), key=lambda meter: (len(meter_groups[meter]), meter))
    else:
        meter_order = sorted(meter_groups.keys())

    selected = []
    pointers = {meter: 0 for meter in meter_order}
    while len(selected) < n:
        progressed = False
        for meter in meter_order:
            ptr = pointers[meter]
            rows = meter_groups[meter]
            if ptr >= len(rows):
                continue
            selected.append(rows[ptr])
            pointers[meter] += 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break

    manifest = {
        "strategy": strategy,
        "seed": seed,
        "source_indices": [int(row["source_index"]) for row in selected],
        "base_meters": [str(row["base_meter"]) for row in selected],
    }
    save_json(manifest, manifest_path)
    logger.info(f"saved fixed sample manifest to {manifest_path}")
    logger.info(f"selected_source_indices={manifest['source_indices']}")
    logger.info(f"selected_base_meters={manifest['base_meters']}")
    return selected, manifest


def freeze_sample(sample, root_dir: Path, dataset_id: str, strategy: str, seed: int, logger):
    benchmark_rows = []
    for row in sample:
        benchmark_rows.append({
            "source_index": int(row["source_index"]),
            "base_meter": str(row["base_meter"]),
            "meter_label": str(row["meter_label"]),
            "requested_bayts": int(row["requested_bayts"]),
            "requested_lines": int(row["requested_lines"]),
            "description": str(row["description"]),
            "poem_text": str(row["poem_text"]),
        })

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(root_dir / BENCHMARK_CSV, index=False)
    benchmark_meta = {
        "dataset_id": dataset_id,
        "selection_strategy": strategy,
        "seed": seed,
        "num_poems": len(benchmark_rows),
        "source_indices": [row["source_index"] for row in benchmark_rows],
        "base_meters": [row["base_meter"] for row in benchmark_rows],
    }
    save_json(benchmark_meta, root_dir / BENCHMARK_META_JSON)
    logger.info(f"froze benchmark to {root_dir / BENCHMARK_CSV}")
    return benchmark_rows, benchmark_meta


def load_frozen_benchmark(root_dir: Path, logger):
    benchmark_csv = root_dir / BENCHMARK_CSV
    if not benchmark_csv.exists():
        return None, None

    df = pd.read_csv(benchmark_csv)
    required_cols = [
        "source_index",
        "base_meter",
        "meter_label",
        "requested_bayts",
        "requested_lines",
        "description",
        "poem_text",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Frozen benchmark missing required columns: {missing}")

    rows = []
    for record in df.to_dict(orient="records"):
        rows.append({
            "source_index": int(record["source_index"]),
            "base_meter": str(record["base_meter"]),
            "meter_label": str(record["meter_label"]),
            "requested_bayts": int(record["requested_bayts"]),
            "requested_lines": int(record["requested_lines"]),
            "description": str(record["description"]),
            "poem_text": str(record["poem_text"]),
        })

    meta_path = root_dir / BENCHMARK_META_JSON
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    logger.info(f"reusing frozen benchmark from {benchmark_csv}")
    return rows, meta


def score_task(task):
    out = score_meaning(task["description"], task["poem"], prompt_file=task["prompt_file"], cache_dir=task["cache_dir"])
    return {
        "prompt_file": task["prompt_file"],
        "prompt_tag": task["prompt_tag"],
        "source_index": int(task["source_index"]),
        "base_meter": task["base_meter"],
        "meter_label": task["meter_label"],
        "requested_bayts": int(task["requested_bayts"]),
        "variant_tag": task["variant_tag"],
        "is_gold": bool(task["is_gold"]),
        "score": float(out["score"]),
        "cache_hit": bool(out["cache_hit"]),
        "latency_sec": out["latency_sec"],
        "description_preview": str(task["description"])[:220],
        "poem_preview": str(task["poem"])[:320],
        "notes": out["notes"],
    }


def main():
    load_dotenv(override=False)

    cfg = load_yaml("grpo_config.yaml")
    study_cfg = cfg["studies"]["meaning_prompt_sweep"]
    root_dir = Path(study_cfg["output_root"])
    ensure_dir(root_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = ensure_dir(root_dir / f"meaning_prompt_sweep_{ts}")
    logger = setup_logger("meaning_prompt_sweep", out_root / "meaning_prompt_sweep.log")
    (root_dir / "latest_internal.log").write_text(str(out_root / "meaning_prompt_sweep.log"), encoding="utf-8")
    (root_dir / "latest_run_dir.txt").write_text(str(out_root), encoding="utf-8")

    random.seed(int(study_cfg["seed"]))

    dataset_id = get_dataset_id_for_phase()
    max_bayts = os.getenv("PHASE1_MAX_BAYTS", "").strip() or None
    allowed_meters = parse_allowed_meters_env()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    cache_dir = os.getenv("REWARD_CACHE_DIR", "./cache/reward_cache")
    good_score_threshold = float(os.getenv("MEANING_SWEEP_GOOD_SCORE_THRESHOLD", str(study_cfg.get("good_score_threshold", 0.80))))
    parallel_workers = int(os.getenv("MEANING_SWEEP_PARALLEL_WORKERS", str(study_cfg.get("parallel_workers", 6))))
    sample_strategy = str(study_cfg.get("sample_strategy", "balanced_by_base_meter"))
    reuse_saved_sample = bool(study_cfg.get("reuse_saved_sample", True))
    rebuild_benchmark = os.getenv("MEANING_SWEEP_REBUILD_BENCHMARK", "").strip() == "1"

    prompt_files = discover_prompt_files()
    prompt_meta = validate_prompt_files(prompt_files)
    pd.DataFrame(prompt_meta).to_csv(out_root / "prompt_files.csv", index=False)

    sample = None
    sample_manifest = {}
    benchmark_meta = {}
    if not rebuild_benchmark:
        sample, benchmark_meta = load_frozen_benchmark(root_dir, logger)

    if sample is None:
        ds = load_and_prepare_dataset(
            dataset_id=dataset_id,
            split="train",
            max_bayts=max_bayts,
            allowed_meters=allowed_meters or None,
            hf_token=hf_token,
        )
        logger.info(f"dataset_size_after_prep={len(ds)}")

        n = min(int(study_cfg["golden_samples"]), len(ds))
        sample, sample_manifest = select_fixed_sample(
            ds=ds,
            n=n,
            seed=int(study_cfg["seed"]),
            root_dir=root_dir,
            strategy=sample_strategy,
            reuse_saved_sample=reuse_saved_sample,
            logger=logger,
        )
        sample, benchmark_meta = freeze_sample(
            sample=sample,
            root_dir=root_dir,
            dataset_id=dataset_id,
            strategy=sample_strategy,
            seed=int(study_cfg["seed"]),
            logger=logger,
        )
    else:
        sample_manifest = {
            "source_indices": [row["source_index"] for row in sample],
            "base_meters": [row["base_meter"] for row in sample],
            "reused_frozen_benchmark": True,
        }

    save_json(sample_manifest, out_root / "sample_source_indices.json")
    save_json(benchmark_meta, out_root / BENCHMARK_META_JSON)

    selected_rows = []
    for row in sample:
        selected_rows.append({
            "source_index": int(row["source_index"]),
            "base_meter": row["base_meter"],
            "meter_label": row["meter_label"],
            "requested_bayts": int(row["requested_bayts"]),
            "requested_lines": int(row["requested_lines"]),
            "description_preview": str(row["description"])[:220],
            "poem_preview": str(row["poem_text"])[:320],
        })
    pd.DataFrame(selected_rows).to_csv(out_root / "selected_gold_poems.csv", index=False)

    tasks = []
    prompt_dir = ensure_dir(out_root / "per_prompt")
    for prompt_rank, prompt_file in enumerate(prompt_files, start=1):
        tag = prompt_tag(prompt_file, prompt_rank)
        ensure_dir(prompt_dir / tag)
        for row in sample:
            gold_poem = row["poem_text"]
            desc = row["description"]
            variants = [("gold", gold_poem)]
            variants.extend(make_bad_variants(gold_poem)[: int(study_cfg["bad_variants_per_sample"])])

            for variant_tag, poem in variants:
                tasks.append({
                    "prompt_file": prompt_file,
                    "prompt_tag": tag,
                    "source_index": int(row["source_index"]),
                    "base_meter": row["base_meter"],
                    "meter_label": row["meter_label"],
                    "requested_bayts": int(row["requested_bayts"]),
                    "variant_tag": variant_tag,
                    "is_gold": variant_tag == "gold",
                    "description": desc,
                    "poem": poem,
                    "cache_dir": cache_dir,
                })

    logger.info(f"selected_gold_poems={len(sample)}")
    logger.info(f"prompt_files={prompt_files}")
    logger.info(f"tasks_total={len(tasks)}")
    logger.info(f"parallel_workers={parallel_workers}")

    rows = []
    event_path = out_root / "meaning_prompt_events.jsonl"
    with ThreadPoolExecutor(max_workers=max(1, parallel_workers)) as executor:
        future_to_task = {executor.submit(score_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            record = future.result()
            rows.append(record)
            append_jsonl(event_path, record)
            logger.info(json.dumps({
                "prompt_file": record["prompt_file"],
                "prompt_tag": record["prompt_tag"],
                "source_index": record["source_index"],
                "variant_tag": record["variant_tag"],
                "score": record["score"],
                "cache_hit": record["cache_hit"],
            }, ensure_ascii=False))

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "meaning_prompt_rows.csv", index=False)

    summary_rows = []
    for prompt_file, g in df.groupby("prompt_file"):
        gold = g[g["is_gold"]].copy()
        bad = g[~g["is_gold"]].copy()
        prompt_tag_value = gold["prompt_tag"].iloc[0] if len(gold) else g["prompt_tag"].iloc[0]
        per_prompt_dir = ensure_dir(prompt_dir / prompt_tag_value)
        g.sort_values(["source_index", "variant_tag"]).to_csv(per_prompt_dir / "all_rows.csv", index=False)
        gold.sort_values("score", ascending=False).to_csv(per_prompt_dir / "gold_scores.csv", index=False)
        bad.sort_values("score", ascending=False).to_csv(per_prompt_dir / "bad_variant_scores.csv", index=False)

        gold_failures = gold[gold["score"] < good_score_threshold].sort_values("score", ascending=True)
        gold_failures.to_csv(per_prompt_dir / "gold_failures.csv", index=False)

        bad_by_source = bad.groupby("source_index")["score"].mean().rename("bad_mean_per_source")
        gold_by_source = gold.groupby("source_index")["score"].mean().rename("gold_mean_per_source")
        paired = pd.concat([gold_by_source, bad_by_source], axis=1).dropna()

        summary_row = {
            "prompt_file": prompt_file,
            "prompt_tag": prompt_tag_value,
            "num_gold_rows": int(len(gold)),
            "num_bad_rows": int(len(bad)),
            "gold_mean_score": round(float(gold["score"].mean()), 4),
            "gold_median_score": round(float(gold["score"].median()), 4),
            "gold_min_score": round(float(gold["score"].min()), 4),
            "gold_max_score": round(float(gold["score"].max()), 4),
            "gold_pass_rate_0_7": round(float((gold["score"] >= 0.7).mean()), 4),
            "gold_pass_rate_0_8": round(float((gold["score"] >= 0.8).mean()), 4),
            "good_score_threshold": round(float(good_score_threshold), 4),
            "num_gold_below_threshold": int((gold["score"] < good_score_threshold).sum()),
            "all_gold_ge_threshold": bool((gold["score"] >= good_score_threshold).all()) if len(gold) else False,
            "bad_mean_score": round(float(bad["score"].mean()), 4) if len(bad) else 0.0,
            "bad_median_score": round(float(bad["score"].median()), 4) if len(bad) else 0.0,
            "bad_min_score": round(float(bad["score"].min()), 4) if len(bad) else 0.0,
            "bad_max_score": round(float(bad["score"].max()), 4) if len(bad) else 0.0,
            "bad_pass_rate_0_7": round(float((bad["score"] >= 0.7).mean()), 4) if len(bad) else 0.0,
            "mean_separation_gap": round(float(gold["score"].mean() - bad["score"].mean()), 4) if len(bad) else 0.0,
            "pairwise_gold_gt_bad_rate": round(float((paired["gold_mean_per_source"] > paired["bad_mean_per_source"]).mean()), 4) if len(paired) else 0.0,
            "pairwise_margin_mean": round(float((paired["gold_mean_per_source"] - paired["bad_mean_per_source"]).mean()), 4) if len(paired) else 0.0,
            "cache_hit_rate": round(float(g["cache_hit"].mean()), 4),
        }
        summary_rows.append(summary_row)

        save_json(summary_row, per_prompt_dir / "summary.json")
        (per_prompt_dir / "README.md").write_text(
            f"# {prompt_file}\n\n"
            f"- `good_score_threshold`: {good_score_threshold}\n"
            f"- `all_gold_ge_threshold`: {summary_row['all_gold_ge_threshold']}\n"
            f"- `num_gold_below_threshold`: {summary_row['num_gold_below_threshold']}\n"
            f"- `gold_mean_score`: {summary_row['gold_mean_score']}\n"
            f"- `bad_mean_score`: {summary_row['bad_mean_score']}\n"
            f"- `pairwise_gold_gt_bad_rate`: {summary_row['pairwise_gold_gt_bad_rate']}\n\n"
            "Files:\n"
            "- `gold_scores.csv`: the 20 gold poems sorted by meaning score.\n"
            "- `gold_failures.csv`: the subset of gold poems still below the chosen threshold.\n"
            "- `bad_variant_scores.csv`: corrupted variants for separation checking.\n"
            "- `all_rows.csv`: all scored rows for this prompt.\n",
            encoding="utf-8",
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["all_gold_ge_threshold", "num_gold_below_threshold", "pairwise_gold_gt_bad_rate", "mean_separation_gap", "gold_mean_score"],
        ascending=[False, True, False, False, False],
    )
    summary_df.to_csv(out_root / "meaning_prompt_summary.csv", index=False)

    best_prompt = summary_df.iloc[0].to_dict() if len(summary_df) else {}
    save_json({
        "dataset_id": dataset_id,
        "selected_gold_poems": len(sample),
        "good_score_threshold": good_score_threshold,
        "num_prompt_files": len(prompt_files),
        "prompt_files": prompt_files,
        "sample_manifest": sample_manifest,
        "benchmark_metadata": benchmark_meta,
        "best_prompt_by_current_sort": best_prompt,
    }, out_root / "report.json")

    (out_root / "README.md").write_text(
        "# Meaning Prompt Sweep\n\n"
        "This study compares multiple meaning reward prompt files on a fixed 20-poem benchmark drawn from the current prepared phase-1 dataset.\n\n"
        "How to use it:\n"
        "- Add or edit `prompts/meaning*.yaml` files.\n"
        "- Optionally set `MEANING_PROMPT_CANDIDATES` to a comma-separated subset.\n"
        "- Optionally set `MEANING_SWEEP_GOOD_SCORE_THRESHOLD` to change what counts as a successful gold score.\n"
        "- Run `python studies/11_sweep_meaning_prompts.py`.\n\n"
        "Main files:\n"
        "- `meaning_prompt_summary.csv`: one row per prompt file with gold quality and gold-vs-bad separation.\n"
        "- `meaning_prompt_rows.csv`: row-level raw scores and notes.\n"
        "- `meaning_prompt_events.jsonl`: streaming event log as each judge call finishes.\n"
        "- `selected_gold_poems.csv`: the fixed 20-poem benchmark set used in this run.\n"
        f"- `{BENCHMARK_CSV}` in the root output folder is the frozen benchmark used across runs.\n"
        "- `prompt_files.csv`: previews of the tested prompt files.\n"
        "- `report.json`: compact top-level result.\n\n"
        "How to choose the best prompt:\n"
        "- first check `all_gold_ge_threshold`\n"
        "- then check `num_gold_below_threshold`\n"
        "- prefer high `gold_mean_score`\n"
        "- prefer low `bad_mean_score`\n"
        "- prefer high `pairwise_gold_gt_bad_rate`\n"
        "- prefer high `mean_separation_gap`\n"
        "- avoid prompts that simply score everything high\n\n"
        "Detachment notes:\n"
        "- the root folder stores `latest.log`, `latest_internal.log`, `latest.pid`, and `latest_run_dir.txt`\n"
        "- each prompt also gets its own folder under `per_prompt/`\n",
        encoding="utf-8",
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
