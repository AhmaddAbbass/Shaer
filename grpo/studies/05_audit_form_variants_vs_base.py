import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import (
    ensure_dir,
    get_dataset_id_for_phase,
    load_and_prepare_dataset,
    load_env,
    save_json,
    setup_logger,
)
from rewards.meter import score_meter_poem

load_dotenv(override=False)


def safe_filename(text):
    text = str(text or "").strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text) or "plot"


def plot_ranked_bar(summary, value_col, title, ylabel, path, color="#1f77b4"):
    data = summary.sort_values(value_col, ascending=False)
    plt.figure(figsize=(14, 7))
    plt.bar(data["meter_label"], data[value_col], color=color)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_form_summary(summary, value_col, title, ylabel, path, color="#2ca02c"):
    data = summary.sort_values(value_col, ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(data["form"], data[value_col], color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def majority_vote(labels):
    labels = [str(x).strip() for x in labels if str(x).strip()]
    if not labels:
        return "", 0.0
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    return label, count / len(labels)


def main():
    out_root = ensure_dir(Path("outputs/audit_form_variants_vs_base"))
    plots_root = ensure_dir(out_root / "plots")
    logger = setup_logger("audit_form_variants_vs_base", out_root / "audit_form_variants_vs_base.log")

    load_env()
    dataset_id = get_dataset_id_for_phase()
    max_bayts = os.getenv("PHASE1_MAX_BAYTS", "").strip() or None
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    ds = load_and_prepare_dataset(
        dataset_id=dataset_id,
        split="train",
        max_bayts=max_bayts,
        hf_token=hf_token,
    )
    df = ds.to_pandas()
    form_series = df["form"].fillna("").astype(str).str.strip()
    sub = df[form_series != "تام"].copy()

    logger.info(f"dataset_id={dataset_id}")
    logger.info(f"rows_total={len(df)}")
    logger.info(f"rows_non_tam={len(sub)}")
    logger.info(f"forms={sorted(sub['form'].dropna().astype(str).str.strip().unique().tolist())}")

    rows = []
    for row_id, (_, row) in enumerate(sub.iterrows(), 1):
        meter_out = score_meter_poem(
            generated_poem=row["poem_text"],
            target_meter=row["base_meter"],
            base_meter=row["base_meter"],
            aggregator="logmean",
        )

        valid_details = [d for d in meter_out["per_bayt_details"] if not d.get("skipped")]
        bayt_preds = [d.get("pred", "") for d in valid_details]
        bayt_pred_base_matches = [int(pred == row["base_meter"]) for pred in bayt_preds]

        majority_pred, majority_frac = majority_vote(bayt_preds)
        frac_bayts_pred_base = (
            sum(bayt_pred_base_matches) / len(bayt_pred_base_matches)
            if bayt_pred_base_matches
            else 0.0
        )

        rows.append({
            "audit_row_id": row_id,
            "source_index": int(row["source_index"]),
            "form": row["form"],
            "base_meter": row["base_meter"],
            "meter_label": row["meter_label"],
            "requested_bayts": int(row["requested_bayts"]),
            "requested_lines": int(row["requested_lines"]),
            "has_odd_tail": bool(row["has_odd_tail"]),
            "target_meter_used": meter_out["target_meter_used"],
            "target_resolution": meter_out["target_resolution"],
            "meter_score_vs_base": meter_out["score"],
            "meter_score_mean_vs_base": meter_out["mean_score"],
            "meter_score_std_vs_base": meter_out["std_score"],
            "num_valid_bayts": meter_out["num_valid_bayts"],
            "num_skipped_bayts": meter_out["num_skipped_bayts"],
            "majority_pred_label": majority_pred,
            "majority_pred_fraction": majority_frac,
            "majority_pred_is_base": int(majority_pred == row["base_meter"]) if majority_pred else 0,
            "fraction_bayts_pred_as_base": frac_bayts_pred_base,
            "all_valid_bayts_pred_as_base": int(
                bool(bayt_pred_base_matches) and all(bayt_pred_base_matches)
            ),
            "prompt_preview": str(row["prompt"])[:220],
            "poem_preview": str(row["poem_text"])[:500],
        })

        if row_id % 250 == 0:
            logger.info(f"scored_non_tam_rows={row_id}")

    results = pd.DataFrame(rows)
    results.to_csv(out_root / "all_non_tam_poems_scored_vs_base.csv", index=False)

    per_label = (
        results.groupby(["meter_label", "base_meter", "form"])
        .agg(
            poems=("meter_score_vs_base", "size"),
            mean_score_vs_base=("meter_score_vs_base", "mean"),
            median_score_vs_base=("meter_score_vs_base", "median"),
            std_score_vs_base=("meter_score_vs_base", "std"),
            mean_num_valid_bayts=("num_valid_bayts", "mean"),
            odd_tail_fraction=("has_odd_tail", "mean"),
            majority_pred_is_base_rate=("majority_pred_is_base", "mean"),
            all_valid_bayts_pred_as_base_rate=("all_valid_bayts_pred_as_base", "mean"),
            mean_fraction_bayts_pred_as_base=("fraction_bayts_pred_as_base", "mean"),
        )
        .reset_index()
        .fillna(0.0)
        .sort_values(["mean_score_vs_base", "majority_pred_is_base_rate"], ascending=[False, False])
    )
    per_label.to_csv(out_root / "summary_by_meter_label.csv", index=False)

    per_form = (
        results.groupby("form")
        .agg(
            poems=("meter_score_vs_base", "size"),
            mean_score_vs_base=("meter_score_vs_base", "mean"),
            median_score_vs_base=("meter_score_vs_base", "median"),
            majority_pred_is_base_rate=("majority_pred_is_base", "mean"),
            all_valid_bayts_pred_as_base_rate=("all_valid_bayts_pred_as_base", "mean"),
            mean_fraction_bayts_pred_as_base=("fraction_bayts_pred_as_base", "mean"),
            odd_tail_fraction=("has_odd_tail", "mean"),
        )
        .reset_index()
        .sort_values("mean_score_vs_base", ascending=False)
    )
    per_form.to_csv(out_root / "summary_by_form.csv", index=False)

    confusion = results[
        (results["majority_pred_label"].fillna("") != "") &
        (results["majority_pred_label"] != results["base_meter"])
    ].copy()
    confusion_summary = (
        confusion.groupby(["meter_label", "base_meter", "majority_pred_label"])
        .size()
        .reset_index(name="poems")
        .sort_values("poems", ascending=False)
    )
    confusion_summary.to_csv(out_root / "majority_label_confusions.csv", index=False)

    worst = results.sort_values("meter_score_vs_base", ascending=True).head(50)
    worst.to_csv(out_root / "lowest_scoring_non_tam_examples.csv", index=False)

    plot_ranked_bar(
        per_label,
        value_col="mean_score_vs_base",
        title="Non-Tam Poems: Mean Meter Score Against Base Meter",
        ylabel="Mean Score vs Base Meter",
        path=plots_root / "mean_score_vs_base_by_meter_label.png",
        color="#1f77b4",
    )
    plot_ranked_bar(
        per_label,
        value_col="majority_pred_is_base_rate",
        title="Non-Tam Poems: Majority Prediction Matches Base Meter",
        ylabel="Rate",
        path=plots_root / "majority_pred_is_base_rate_by_meter_label.png",
        color="#ff7f0e",
    )
    plot_ranked_bar(
        per_label,
        value_col="mean_fraction_bayts_pred_as_base",
        title="Non-Tam Poems: Fraction of Bayts Predicted as Base Meter",
        ylabel="Mean Fraction",
        path=plots_root / "fraction_bayts_pred_as_base_by_meter_label.png",
        color="#2ca02c",
    )
    plot_form_summary(
        per_form,
        value_col="mean_score_vs_base",
        title="Non-Tam Poems: Mean Score vs Base by Form",
        ylabel="Mean Score vs Base Meter",
        path=plots_root / "mean_score_vs_base_by_form.png",
    )

    report = {
        "dataset_id": dataset_id,
        "rows_total_after_filter": int(len(df)),
        "rows_non_tam": int(len(sub)),
        "forms_present": sorted(sub["form"].dropna().astype(str).str.strip().unique().tolist()),
        "overall_mean_score_vs_base": float(results["meter_score_vs_base"].mean()),
        "overall_median_score_vs_base": float(results["meter_score_vs_base"].median()),
        "overall_majority_pred_is_base_rate": float(results["majority_pred_is_base"].mean()),
        "overall_all_valid_bayts_pred_as_base_rate": float(results["all_valid_bayts_pred_as_base"].mean()),
        "overall_mean_fraction_bayts_pred_as_base": float(results["fraction_bayts_pred_as_base"].mean()),
        "top_majority_confusions": confusion_summary.head(15).to_dict(orient="records"),
    }
    save_json(report, out_root / "report.json")

    logger.info(f"overall_mean_score_vs_base={report['overall_mean_score_vs_base']:.4f}")
    logger.info(f"overall_majority_pred_is_base_rate={report['overall_majority_pred_is_base_rate']:.4f}")
    logger.info(
        f"overall_mean_fraction_bayts_pred_as_base={report['overall_mean_fraction_bayts_pred_as_base']:.4f}"
    )
    logger.info(f"saved_out_root={out_root}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
