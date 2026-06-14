import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import (
    ensure_dir,
    get_dataset_source_id_for_phase,
    load_and_prepare_dataset,
    load_yaml,
    parse_allowed_meters_env,
    save_json,
    score_count_adherence,
    setup_logger,
)
from rewards.meaning_fit import score_meaning_fit
from rewards.meaning_substance import score_meaning_substance
from rewards.meter import score_meter_poem


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_BANK = ROOT / "choices" / "building_dataset_for_benchmark" / "benchmark_bank_v1.csv"


def summarize_group(df: pd.DataFrame, metric: str) -> dict[str, float]:
    return {
        "mean": float(df[metric].mean()),
        "median": float(df[metric].median()),
        "min": float(df[metric].min()),
        "max": float(df[metric].max()),
    }


def main():
    load_dotenv(override=False)

    cfg = load_yaml(ROOT / "grpo_config.yaml")
    study_cfg = cfg["studies"]["sanity_check_rewards"]
    out_root = ensure_dir(ROOT / study_cfg["output_root"])
    logger = setup_logger("sanity_check_rewards", out_root / "sanity_check_rewards.log")

    fit_prompt = os.getenv("MEANING_FIT_PROMPT_FILE", "prompts/meaning_fit.yaml").strip() or "prompts/meaning_fit.yaml"
    substance_prompt = os.getenv("MEANING_SUBSTANCE_PROMPT_FILE", "prompts/meaning_substance.yaml").strip() or "prompts/meaning_substance.yaml"
    cache_dir = os.getenv("REWARD_CACHE_DIR", "./cache/reward_cache")
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    if not BENCHMARK_BANK.exists():
        raise FileNotFoundError(f"Missing benchmark bank: {BENCHMARK_BANK}")

    benchmark_df = pd.read_csv(BENCHMARK_BANK)
    logger.info("loaded benchmark rows=%d from %s", len(benchmark_df), BENCHMARK_BANK)

    source_dataset_id = get_dataset_source_id_for_phase()
    source_ds = load_and_prepare_dataset(
        dataset_id=source_dataset_id,
        split="train",
        max_bayts=os.getenv("PHASE1_MAX_BAYTS", "").strip() or None,
        allowed_meters=parse_allowed_meters_env() or None,
        hf_token=hf_token,
    )
    source_lookup = {int(row["source_index"]): row for row in source_ds}
    logger.info("loaded source lookup rows=%d from %s", len(source_lookup), source_dataset_id)

    rows = []
    for _, row in benchmark_df.iterrows():
        source_index = int(row["source_index"])
        poem_source_index = int(row["poem_source_index"])
        source_meta = source_lookup.get(poem_source_index) or source_lookup.get(source_index) or {}

        poem_text = str(row["poem_text"])
        description = str(row["description"])
        meter_label = str(source_meta.get("meter_label") or row.get("poem_meter_label") or "")
        base_meter = str(source_meta.get("base_meter") or "")
        requested_bayts = int(source_meta.get("requested_bayts") or row.get("num_bayts") or 0)

        meter_out = score_meter_poem(poem_text, meter_label, base_meter=base_meter, aggregator="logmean")
        fit_out = score_meaning_fit(description, poem_text, prompt_file=fit_prompt, cache_dir=cache_dir)
        substance_out = score_meaning_substance(poem_text, prompt_file=substance_prompt, cache_dir=cache_dir)
        count_out = score_count_adherence(requested_bayts, poem_text)

        rec = {
            "pair_id": int(row["pair_id"]),
            "group_id": str(row["group_id"]),
            "group_label": str(row["group_label"]),
            "source_index": source_index,
            "description_source_index": int(row["description_source_index"]),
            "poem_source_index": poem_source_index,
            "requested_bayts": requested_bayts,
            "meter_label": meter_label,
            "base_meter": base_meter,
            "corruption_note": str(row.get("corruption_note") or ""),
            "meter_score": meter_out["score"],
            "meter_target_resolution": meter_out["target_resolution"],
            "meter_target_used": meter_out["target_meter_used"],
            "meter_num_valid_bayts": meter_out["num_valid_bayts"],
            "fit_score": fit_out["score"],
            "fit_notes": fit_out["notes"],
            "fit_error": fit_out["error"],
            "fit_cache_hit": fit_out["cache_hit"],
            "fit_mode_respected": fit_out["mode_respected"],
            "fit_reasoning_tokens": fit_out["reasoning_tokens"],
            "substance_score": substance_out["score"],
            "substance_notes": substance_out["notes"],
            "substance_error": substance_out["error"],
            "substance_cache_hit": substance_out["cache_hit"],
            "substance_mode_respected": substance_out["mode_respected"],
            "substance_reasoning_tokens": substance_out["reasoning_tokens"],
            "count_adherence_score": count_out["score"],
            "generated_bayts": count_out["generated_bayts"],
            "description_preview": description[:300],
            "poem_preview": poem_text[:400],
        }
        rows.append(rec)
        logger.info(json.dumps(rec, ensure_ascii=False))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_root / "reward_sanity_rows.csv", index=False, encoding="utf-8-sig")
    save_json(results_df.replace({pd.NA: None}).to_dict(orient="records"), out_root / "reward_sanity_rows.json")

    summary_df = (
        results_df.groupby("group_id", as_index=False)
        .agg(
            n=("pair_id", "count"),
            meter_mean=("meter_score", "mean"),
            fit_mean=("fit_score", "mean"),
            fit_median=("fit_score", "median"),
            substance_mean=("substance_score", "mean"),
            substance_median=("substance_score", "median"),
            count_mean=("count_adherence_score", "mean"),
            fit_error_count=("fit_error", lambda s: int(pd.Series(s).notna().sum())),
            substance_error_count=("substance_error", lambda s: int(pd.Series(s).notna().sum())),
            fit_mode_fail_count=("fit_mode_respected", lambda s: int((pd.Series(s) == False).sum())),
            substance_mode_fail_count=("substance_mode_respected", lambda s: int((pd.Series(s) == False).sum())),
        )
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    summary_df.to_csv(out_root / "reward_sanity_summary.csv", index=False, encoding="utf-8-sig")
    save_json(summary_df.to_dict(orient="records"), out_root / "reward_sanity_summary.json")

    by_group = {row["group_id"]: row for row in summary_df.to_dict(orient="records")}
    g1 = by_group["group1"]
    g2 = by_group["group2"]
    g3 = by_group["group3"]

    assertions = [
        {
            "name": "fit_group1_beats_group2",
            "lhs": float(g1["fit_mean"]),
            "rhs": float(g2["fit_mean"]),
            "delta": float(g1["fit_mean"] - g2["fit_mean"]),
            "passed": bool((g1["fit_mean"] - g2["fit_mean"]) >= 0.12),
        },
        {
            "name": "substance_group1_beats_group3",
            "lhs": float(g1["substance_mean"]),
            "rhs": float(g3["substance_mean"]),
            "delta": float(g1["substance_mean"] - g3["substance_mean"]),
            "passed": bool((g1["substance_mean"] - g3["substance_mean"]) >= 0.10),
        },
        {
            "name": "substance_group2_stays_above_group3",
            "lhs": float(g2["substance_mean"]),
            "rhs": float(g3["substance_mean"]),
            "delta": float(g2["substance_mean"] - g3["substance_mean"]),
            "passed": bool((g2["substance_mean"] - g3["substance_mean"]) >= 0.03),
        },
        {
            "name": "non_thinking_mode_respected",
            "lhs": int(summary_df["fit_mode_fail_count"].sum() + summary_df["substance_mode_fail_count"].sum()),
            "rhs": 0,
            "delta": int(summary_df["fit_mode_fail_count"].sum() + summary_df["substance_mode_fail_count"].sum()),
            "passed": bool(
                int(summary_df["fit_mode_fail_count"].sum() + summary_df["substance_mode_fail_count"].sum()) == 0
            ),
        },
        {
            "name": "judge_parse_errors_absent",
            "lhs": int(summary_df["fit_error_count"].sum() + summary_df["substance_error_count"].sum()),
            "rhs": 0,
            "delta": int(summary_df["fit_error_count"].sum() + summary_df["substance_error_count"].sum()),
            "passed": bool(int(summary_df["fit_error_count"].sum() + summary_df["substance_error_count"].sum()) == 0),
        },
    ]
    save_json(assertions, out_root / "reward_sanity_assertions.json")

    print(summary_df.to_string(index=False))
    print("\nAssertions:")
    for item in assertions:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"- {status} | {item['name']} | delta={item['delta']}")

    failed = [item for item in assertions if not item["passed"]]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
