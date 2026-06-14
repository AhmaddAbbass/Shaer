import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import ensure_dir, get_dataset_id_for_phase, load_and_prepare_dataset, load_env, save_json, setup_logger


THRESHOLDS = [4, 6, 8, 10, 12, 16, 20]


def safe_plot_style():
    for style in ["seaborn-v0_8-whitegrid", "ggplot"]:
        try:
            plt.style.use(style)
            return
        except Exception:
            pass


def meter_dir_name(rank: int, base_meter: str) -> str:
    return f"{rank:02d}_{base_meter}"


def write_text(path: Path, text: str):
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def plot_meter_histogram(df_meter: pd.DataFrame, meter: str, out_path: Path):
    counts = (
        df_meter["requested_bayts"]
        .value_counts()
        .sort_index()
        .reindex(range(1, int(df_meter["requested_bayts"].max()) + 1), fill_value=0)
    )
    cumulative = counts.cumsum() / counts.sum()

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(counts.index, counts.values, color="#2E5B88", edgecolor="white", width=0.85)
    ax1.set_title(f"{meter}: Requested Bayt Count Distribution")
    ax1.set_xlabel("Requested Bayts")
    ax1.set_ylabel("Poem Count")
    ax1.set_xticks(list(counts.index))

    ymax = max(counts.values) if len(counts.values) else 1
    ax1.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

    for cutoff, color in [(8, "#C44E52"), (10, "#55A868")]:
        if cutoff <= counts.index.max():
            ax1.axvline(cutoff, color=color, linestyle="--", linewidth=1.8, alpha=0.9)

    ax2 = ax1.twinx()
    ax2.plot(counts.index, cumulative.values, color="#8172B3", marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative Share")
    ax2.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    load_env()
    dataset_id = get_dataset_id_for_phase()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    out_root = Path("outputs/phase1_length_by_base_meter")
    per_meter_root = ensure_dir(out_root / "per_meter")
    ensure_dir(out_root)
    logger = setup_logger("phase1_length_by_base_meter", out_root / "phase1_length_by_base_meter.log")
    safe_plot_style()

    if not dataset_id:
        raise ValueError("No dataset id resolved from PHASE1_DATASET_ID or SOURCE_DATASET_ID")

    logger.info(f"loading prepared phase-1 dataset: {dataset_id}")
    ds = load_and_prepare_dataset(dataset_id, split="train", hf_token=hf_token)
    rows = []
    for row in ds:
        rows.append({
            "source_index": int(row["source_index"]),
            "base_meter": str(row["base_meter"]),
            "meter_label": str(row["meter_label"]),
            "form": str(row["form"]),
            "requested_lines": int(row["requested_lines"]),
            "requested_bayts": int(row["requested_bayts"]),
            "has_odd_tail": bool(row["has_odd_tail"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Prepared dataset is empty")

    df.to_csv(out_root / "all_rows.csv", index=False)

    base_meter_order = (
        df.groupby("base_meter")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    summary_rows = []
    threshold_rows = []
    index_rows = []

    for rank, meter in enumerate(base_meter_order, start=1):
        df_meter = df[df["base_meter"] == meter].copy()
        meter_dir = ensure_dir(per_meter_root / meter_dir_name(rank, meter))

        counts = (
            df_meter["requested_bayts"]
            .value_counts()
            .sort_index()
            .rename_axis("requested_bayts")
            .reset_index(name="count")
        )
        counts["share"] = counts["count"] / counts["count"].sum()
        counts["cumulative_share"] = counts["share"].cumsum()
        counts.to_csv(meter_dir / "histogram_counts.csv", index=False)

        plot_meter_histogram(df_meter, meter, meter_dir / "length_histogram.png")

        summary = {
            "base_meter": meter,
            "num_poems": int(len(df_meter)),
            "min_requested_bayts": int(df_meter["requested_bayts"].min()),
            "max_requested_bayts": int(df_meter["requested_bayts"].max()),
            "mean_requested_bayts": round(float(df_meter["requested_bayts"].mean()), 4),
            "median_requested_bayts": round(float(df_meter["requested_bayts"].median()), 4),
            "odd_tail_rate": round(float(df_meter["has_odd_tail"].mean()), 4),
            "requested_labels_present": sorted(df_meter["meter_label"].value_counts().index.tolist()),
        }
        save_json(summary, meter_dir / "summary.json")

        write_text(
            meter_dir / "README.md",
            (
                f"# {meter}\n\n"
                f"- `num_poems`: {summary['num_poems']}\n"
                f"- `min_requested_bayts`: {summary['min_requested_bayts']}\n"
                f"- `median_requested_bayts`: {summary['median_requested_bayts']}\n"
                f"- `max_requested_bayts`: {summary['max_requested_bayts']}\n"
                f"- `mean_requested_bayts`: {summary['mean_requested_bayts']}\n\n"
                "Files:\n"
                "- `length_histogram.png`: count bars plus cumulative share line.\n"
                "- `histogram_counts.csv`: exact counts and cumulative shares by requested bayt count.\n"
                "- `summary.json`: compact meter summary plus which prompt labels feed this base meter.\n"
            ),
        )

        row = {
            "rank_by_num_poems": rank,
            "base_meter": meter,
            "num_poems": int(len(df_meter)),
            "min_requested_bayts": int(df_meter["requested_bayts"].min()),
            "median_requested_bayts": round(float(df_meter["requested_bayts"].median()), 4),
            "mean_requested_bayts": round(float(df_meter["requested_bayts"].mean()), 4),
            "max_requested_bayts": int(df_meter["requested_bayts"].max()),
            "odd_tail_rate": round(float(df_meter["has_odd_tail"].mean()), 4),
            "num_requested_labels_present": int(df_meter["meter_label"].nunique()),
        }
        for threshold in THRESHOLDS:
            count_leq = int((df_meter["requested_bayts"] <= threshold).sum())
            row[f"count_leq_{threshold}"] = count_leq
            row[f"share_leq_{threshold}"] = round(count_leq / len(df_meter), 4)
        summary_rows.append(row)

        for threshold in THRESHOLDS:
            count_leq = int((df_meter["requested_bayts"] <= threshold).sum())
            threshold_rows.append({
                "base_meter": meter,
                "threshold_bayts": threshold,
                "num_poems_leq_threshold": count_leq,
                "share_leq_threshold": round(count_leq / len(df_meter), 4),
            })

        index_rows.append({
            "rank_by_num_poems": rank,
            "base_meter": meter,
            "folder": meter_dir_name(rank, meter),
        })

    summary_df = pd.DataFrame(summary_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    index_df = pd.DataFrame(index_rows)

    summary_df.to_csv(out_root / "base_meter_length_summary.csv", index=False)
    threshold_df.to_csv(out_root / "threshold_retention_by_base_meter.csv", index=False)
    index_df.to_csv(out_root / "per_meter_index.csv", index=False)

    report = {
        "dataset_id": dataset_id,
        "num_rows": int(len(df)),
        "num_base_meters": int(df["base_meter"].nunique()),
        "base_meters": base_meter_order,
        "thresholds": THRESHOLDS,
        "largest_base_meter_by_rows": base_meter_order[0],
        "smallest_base_meter_by_rows": summary_df.sort_values("num_poems").iloc[0]["base_meter"],
    }
    save_json(report, out_root / "report.json")

    write_text(
        out_root / "README.md",
        (
            "# Phase-1 Lengths by Base Meter\n\n"
            "This package answers a curriculum question for the current active training dataset: "
            "if we start GRPO on shorter poems first, will every base meter still be exposed often enough?\n\n"
            "Data source:\n"
            f"- `dataset_id`: `{dataset_id}`\n"
            f"- `prepared_rows`: `{len(df)}`\n"
            f"- `base_meters`: `{df['base_meter'].nunique()}`\n\n"
            "Main files:\n"
            "- `base_meter_length_summary.csv`: one row per base meter with min/median/mean/max requested bayt counts plus retention under several short-length cutoffs.\n"
            "- `threshold_retention_by_base_meter.csv`: long-format version of the cutoff retention table.\n"
            "- `per_meter/`: one folder per base meter with a histogram and exact counts.\n"
            "- `all_rows.csv`: row-level prepared dataset view used for this analysis.\n\n"
            "How to read the per-meter histogram:\n"
            "- bars show how many training poems ask for each requested bayt count.\n"
            "- the purple line shows cumulative share.\n"
            "- the dashed red line marks `8` bayts.\n"
            "- the dashed green line marks `10` bayts.\n\n"
            "How to reason about curriculum exposure:\n"
            "- if `share_leq_8` or `share_leq_10` is high for a base meter, a short-poem warmup would still expose that meter often.\n"
            "- if those shares are very low, a short-poem-only stage could underexpose that meter early.\n"
            "- `count_leq_*` matters as much as `share_leq_*`: a rare meter can have a decent share but still too few absolute examples.\n\n"
            "Important scope note:\n"
            "- this is based on the prepared phase-1 training dataset, not the raw unfiltered upstream corpus.\n"
            "- this package is about requested length exposure, not generation quality.\n"
        ),
    )

    logger.info(f"saved analysis to {out_root}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
