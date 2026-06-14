import os
import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import (
    ensure_dir,
    save_json,
    setup_logger,
    row_to_poem_text,
    row_to_meter_fields,
    poem_structure,
)

load_dotenv(override=False)


def main():
    dataset_id = os.getenv("SOURCE_DATASET_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    out_root = Path("outputs/inspect_lengths")
    ensure_dir(out_root)
    logger = setup_logger("inspect_lengths", out_root / "inspect_lengths.log")

    if not dataset_id:
        raise ValueError("SOURCE_DATASET_ID missing in .env")

    logger.info(f"loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", token=hf_token)

    rows = []
    for i, row in enumerate(ds):
        poem_text = row_to_poem_text(row)
        base_meter, form, meter_label = row_to_meter_fields(row)
        s = poem_structure(poem_text)
        rows.append({
            "source_index": i,
            "meter_label": meter_label,
            "base_meter": base_meter,
            "form": form,
            "num_lines": s["num_lines"],
            "num_bayts": s["complete_bayts"],
            "has_odd_tail": s["has_odd_tail"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "lengths_full.csv", index=False)

    logger.info(f"num_rows={len(df)}")
    logger.info(f"columns={list(df.columns)}")

    summary = {
        "num_rows": int(len(df)),
        "num_lines_describe": df["num_lines"].describe().to_dict(),
        "num_bayts_describe": df["num_bayts"].describe().to_dict(),
        "num_odd_tail": int(df["has_odd_tail"].sum()),
        "meters": int(df["meter_label"].nunique()),
    }
    save_json(summary, out_root / "summary.json")

    counts = df["num_bayts"].value_counts().sort_index()
    counts.to_csv(out_root / "bayt_count_histogram.csv")

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Bayt Count Distribution")
    plt.xlabel("Number of Bayts")
    plt.ylabel("Number of Poems")
    plt.tight_layout()
    plt.savefig(out_root / "bayt_count_distribution.png")
    plt.close()

    coverage_target = 0.90
    cumsum = counts.cumsum() / counts.sum()
    chosen_cutoff = int(cumsum[cumsum >= coverage_target].index.min())

    logger.info(f"coverage_target={coverage_target}")
    logger.info(f"chosen_cutoff={chosen_cutoff}")

    coverage_df = pd.DataFrame({
        "num_bayts": counts.index,
        "count": counts.values,
        "coverage": cumsum.values,
    })
    coverage_df.to_csv(out_root / "coverage_table.csv", index=False)

    meter_stats = (
        df.groupby("meter_label")["num_bayts"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    meter_stats.to_csv(out_root / "meter_length_stats.csv", index=False)

    filtered_df = df[df["num_bayts"] <= chosen_cutoff].copy()
    filtered_df.to_csv(out_root / f"filtered_preview_maxlen{chosen_cutoff}.csv", index=False)

    next_steps = {
        "recommended_max_bayts_phase1": chosen_cutoff,
        "set_in_env": {
            "PHASE1_MAX_BAYTS": chosen_cutoff
        },
        "recommended_filtered_dataset_suffix": f"maxlen{chosen_cutoff}",
    }
    save_json(next_steps, out_root / "next_steps.json")

    print(json.dumps(next_steps, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()