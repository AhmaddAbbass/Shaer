#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DIFFICULTY_COLORS = {
    "easy": "#2E8B57",
    "medium": "#D18F00",
    "hard": "#B23A48",
}
LENGTH_BUCKET_ORDER = ["1-3", "4-6", "7-10", "11-20"]
METER_DISPLAY = {
    "البسيط": "Basit",
    "الخفيف": "Khafif",
    "الرجز": "Rajaz",
    "الرمل": "Raml",
    "السريع": "Sari",
    "الطويل": "Tawil",
    "الكامل": "Kamil",
    "المتقارب": "Mutaqarib",
    "المجتث": "Mujtath",
    "المديد": "Madid",
    "المنسرح": "Munsarih",
    "الهزج": "Hazaj",
    "الوافر": "Wafir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-oriented plots for the finished meter+count preprocess run.")
    parser.add_argument("--run-dir", required=True, help="Full preprocess run directory.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to repo_root/artifacts/analysis/grpo_meter_count_full_k8_paper_figures",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir(run_dir: Path) -> Path:
    stamp = run_dir.name.replace("full_train_", "")
    return repo_root() / "artifacts" / "analysis" / f"grpo_meter_count_paper_figures_{stamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_bucket() -> dict[str, float]:
    return {
        "rows": 0.0,
        "sum_mean_meter": 0.0,
        "sumsq_mean_meter": 0.0,
        "sum_count_exact_rate": 0.0,
        "sum_strong_exact_rate": 0.0,
        "sum_strong_meter_rate": 0.0,
        "sum_spread": 0.0,
    }


def update_bucket(bucket: dict[str, float], *, mean_meter: float, count_exact_rate: float, strong_exact_rate: float, strong_meter_rate: float, spread: float) -> None:
    bucket["rows"] += 1.0
    bucket["sum_mean_meter"] += float(mean_meter)
    bucket["sumsq_mean_meter"] += float(mean_meter) ** 2
    bucket["sum_count_exact_rate"] += float(count_exact_rate)
    bucket["sum_strong_exact_rate"] += float(strong_exact_rate)
    bucket["sum_strong_meter_rate"] += float(strong_meter_rate)
    bucket["sum_spread"] += float(spread)


def finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    rows = int(bucket["rows"])
    if rows <= 0:
        return {
            "rows": 0,
            "mean_row_meter_score": 0.0,
            "std_row_meter_score": 0.0,
            "mean_count_exact_rate": 0.0,
            "mean_strong_exact_rate": 0.0,
            "mean_strong_meter_rate": 0.0,
            "mean_within_row_meter_spread": 0.0,
        }
    mean_meter = bucket["sum_mean_meter"] / rows
    var_meter = max(0.0, (bucket["sumsq_mean_meter"] / rows) - (mean_meter ** 2))
    return {
        "rows": rows,
        "mean_row_meter_score": mean_meter,
        "std_row_meter_score": math.sqrt(var_meter),
        "mean_count_exact_rate": bucket["sum_count_exact_rate"] / rows,
        "mean_strong_exact_rate": bucket["sum_strong_exact_rate"] / rows,
        "mean_strong_meter_rate": bucket["sum_strong_meter_rate"] / rows,
        "mean_within_row_meter_spread": bucket["sum_spread"] / rows,
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_percent_stacked_barh(df: pd.DataFrame, category_col: str, out_path: Path, *, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(df) + 1.5)))
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for difficulty in DIFFICULTY_ORDER:
        vals = df[f"{difficulty}_share"].to_numpy() * 100.0
        ax.barh(y, vals, left=left, color=DIFFICULTY_COLORS[difficulty], label=difficulty, height=0.78)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(df[category_col].tolist(), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, out_path)


def render_difficulty_counts(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = df["rows"].tolist()
    colors = [DIFFICULTY_COLORS[name] for name in df["difficulty"]]
    bars = ax.bar(df["difficulty"], vals, color=colors, width=0.65)
    total = max(1, int(sum(vals)))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}\n({100.0 * val / total:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Rows")
    ax.set_title("Overall Difficulty Distribution")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_path)


def render_line_shares(df: pd.DataFrame, x_col: str, out_path: Path, *, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for difficulty in DIFFICULTY_ORDER:
        ax.plot(df[x_col], df[f"{difficulty}_share"] * 100.0, marker="o", linewidth=2.0, label=difficulty, color=DIFFICULTY_COLORS[difficulty])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of rows (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, out_path)


def render_barh(df: pd.DataFrame, label_col: str, value_col: str, out_path: Path, *, title: str, xlabel: str, color: str, fmt: str = "{:.3f}") -> None:
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(df) + 1.5)))
    y = np.arange(len(df))
    vals = df[value_col].to_numpy()
    bars = ax.barh(y, vals, color=color, height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col].tolist(), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    xmax = max(vals) if len(vals) else 1.0
    ax.set_xlim(0, max(1.0, xmax * 1.08))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(0.01, ax.get_xlim()[1] * 0.008), bar.get_y() + bar.get_height() / 2, fmt.format(val), va="center", fontsize=9)
    save_figure(fig, out_path)


def render_histogram(values: list[float], out_path: Path, *, title: str, xlabel: str, bins: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
    for x, label, color in [(0.30, "bad<0.30", "#B23A48"), (0.70, "strong>=0.70", "#2E8B57")]:
        ax.axvline(x, color=color, linestyle="--", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Candidate count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, out_path)


def render_distribution(counter: Counter[int], out_path: Path, *, title: str, xlabel: str, color: str) -> None:
    xs = sorted(counter)
    ys = [counter[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(xs, ys, color=color, width=0.75)
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rows")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, out_path)


def render_rate_line(df: pd.DataFrame, x_col: str, y_cols: list[tuple[str, str, str]], out_path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for column, label, color in y_cols:
        ax.plot(df[x_col], df[column] * 100.0, marker="o", linewidth=2.0, label=label, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, out_path)


def render_heatmap(df: pd.DataFrame, row_col: str, col_col: str, value_col: str, out_path: Path, *, title: str, cmap: str = "magma") -> None:
    pivot = df.pivot(index=row_col, columns=col_col, values=value_col)
    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.42 * len(pivot) + 1.2)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist())
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=10)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{100.0 * value:.1f}%", ha="center", va="center", fontsize=8, color="white" if value > 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share")
    save_figure(fig, out_path)


def build_summary_tables(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[float], Counter[int]]:
    assembled_path = run_dir / "assembled" / "train.jsonl"
    difficulty_counts: Counter[str] = Counter()
    difficulty_by_meter: dict[str, Counter[str]] = defaultdict(Counter)
    difficulty_by_bayts: dict[int, Counter[str]] = defaultdict(Counter)
    difficulty_by_length_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    meter_stats: dict[str, dict[str, float]] = defaultdict(init_bucket)
    meter_difficulty_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(init_bucket)
    bayts_stats: dict[int, dict[str, float]] = defaultdict(init_bucket)
    difficulty_stats: dict[str, dict[str, float]] = defaultdict(init_bucket)
    meter_length_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    candidate_meter_scores: list[float] = []
    strong_exact_dist: Counter[int] = Counter()
    row_count = 0
    candidate_count = 0

    with assembled_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_count += 1
            difficulty = str(row["difficulty"])
            meter = str(row["base_meter"])
            requested_bayts = int(row["requested_bayts"])
            length_bucket = str(row["length_bucket"])
            mean_meter = float(row["mean_meter_score"])
            count_exact_rate = float(row["count_exact_rate"])
            strong_exact_rate = float(row["strong_exact_rate"])
            strong_meter_rate = float(row["strong_meter_rate"])
            num_strong_exact = int(row["num_strong_exact_candidates"])
            candidates = row["candidates"]
            scores = [float(candidate["meter_score"]) for candidate in candidates]
            candidate_count += len(scores)
            candidate_meter_scores.extend(scores)
            spread = max(scores) - min(scores) if scores else 0.0

            difficulty_counts[difficulty] += 1
            difficulty_by_meter[meter][difficulty] += 1
            difficulty_by_bayts[requested_bayts][difficulty] += 1
            difficulty_by_length_bucket[length_bucket][difficulty] += 1
            meter_length_counts[(meter, length_bucket)][difficulty] += 1
            strong_exact_dist[num_strong_exact] += 1

            update_bucket(meter_stats[meter], mean_meter=mean_meter, count_exact_rate=count_exact_rate, strong_exact_rate=strong_exact_rate, strong_meter_rate=strong_meter_rate, spread=spread)
            update_bucket(meter_difficulty_stats[(meter, difficulty)], mean_meter=mean_meter, count_exact_rate=count_exact_rate, strong_exact_rate=strong_exact_rate, strong_meter_rate=strong_meter_rate, spread=spread)
            update_bucket(bayts_stats[requested_bayts], mean_meter=mean_meter, count_exact_rate=count_exact_rate, strong_exact_rate=strong_exact_rate, strong_meter_rate=strong_meter_rate, spread=spread)
            update_bucket(difficulty_stats[difficulty], mean_meter=mean_meter, count_exact_rate=count_exact_rate, strong_exact_rate=strong_exact_rate, strong_meter_rate=strong_meter_rate, spread=spread)

    difficulty_rows = []
    for difficulty in DIFFICULTY_ORDER:
        rows = difficulty_counts[difficulty]
        share = rows / max(1, row_count)
        stats = finalize_bucket(difficulty_stats[difficulty])
        difficulty_rows.append({"difficulty": difficulty, "rows": rows, "share": share, **stats})
    difficulty_df = pd.DataFrame(difficulty_rows)

    meter_rows = []
    for meter, stats_bucket in meter_stats.items():
        stats = finalize_bucket(stats_bucket)
        diff_counts = difficulty_by_meter[meter]
        rows = max(1, int(stats["rows"]))
        meter_rows.append(
            {
                "base_meter": meter,
                "meter_display": METER_DISPLAY.get(meter, meter),
                **stats,
                "easy_rows": int(diff_counts["easy"]),
                "medium_rows": int(diff_counts["medium"]),
                "hard_rows": int(diff_counts["hard"]),
                "easy_share": diff_counts["easy"] / rows,
                "medium_share": diff_counts["medium"] / rows,
                "hard_share": diff_counts["hard"] / rows,
            }
        )
    meter_df = pd.DataFrame(meter_rows)

    bayt_rows = []
    for requested_bayts, stats_bucket in sorted(bayts_stats.items()):
        stats = finalize_bucket(stats_bucket)
        diff_counts = difficulty_by_bayts[requested_bayts]
        rows = max(1, int(stats["rows"]))
        bayt_rows.append(
            {
                "requested_bayts": int(requested_bayts),
                **stats,
                "easy_rows": int(diff_counts["easy"]),
                "medium_rows": int(diff_counts["medium"]),
                "hard_rows": int(diff_counts["hard"]),
                "easy_share": diff_counts["easy"] / rows,
                "medium_share": diff_counts["medium"] / rows,
                "hard_share": diff_counts["hard"] / rows,
            }
        )
    bayt_df = pd.DataFrame(bayt_rows).sort_values("requested_bayts")

    meter_difficulty_rows = []
    for (meter, difficulty), stats_bucket in meter_difficulty_stats.items():
        stats = finalize_bucket(stats_bucket)
        meter_difficulty_rows.append(
            {
                "base_meter": meter,
                "meter_display": METER_DISPLAY.get(meter, meter),
                "difficulty": difficulty,
                **stats,
            }
        )
    meter_difficulty_df = pd.DataFrame(meter_difficulty_rows)

    meter_length_rows = []
    for (meter, length_bucket), diff_counts in meter_length_counts.items():
        rows = int(sum(diff_counts.values()))
        meter_length_rows.append(
            {
                "base_meter": meter,
                "meter_display": METER_DISPLAY.get(meter, meter),
                "length_bucket": length_bucket,
                "rows": rows,
                "easy_rows": int(diff_counts["easy"]),
                "medium_rows": int(diff_counts["medium"]),
                "hard_rows": int(diff_counts["hard"]),
                "easy_share": diff_counts["easy"] / max(1, rows),
                "medium_share": diff_counts["medium"] / max(1, rows),
                "hard_share": diff_counts["hard"] / max(1, rows),
            }
        )
    meter_length_df = pd.DataFrame(meter_length_rows)
    if not meter_length_df.empty:
        meter_order = meter_df.sort_values(["hard_share", "rows"], ascending=[False, False])["base_meter"].tolist()
        meter_length_df["base_meter"] = pd.Categorical(meter_length_df["base_meter"], categories=meter_order, ordered=True)
        display_order = [METER_DISPLAY.get(name, name) for name in meter_order]
        meter_length_df["meter_display"] = pd.Categorical(meter_length_df["meter_display"], categories=display_order, ordered=True)
        meter_length_df["length_bucket"] = pd.Categorical(meter_length_df["length_bucket"], categories=LENGTH_BUCKET_ORDER, ordered=True)
        meter_length_df = meter_length_df.sort_values(["base_meter", "length_bucket"])

    summary = {
        "run_dir": str(run_dir),
        "row_count": int(row_count),
        "candidate_count": int(candidate_count),
        "difficulty_counts": {k: int(v) for k, v in difficulty_counts.items()},
    }
    return summary, difficulty_df, meter_df, bayt_df, meter_length_df, meter_difficulty_df, candidate_meter_scores, strong_exact_dist


def write_report(path: Path, *, summary: dict[str, Any], difficulty_df: pd.DataFrame, meter_df: pd.DataFrame, bayt_df: pd.DataFrame) -> None:
    total_rows = max(1, int(summary["row_count"]))
    hardest_meters = meter_df.sort_values(["hard_share", "rows"], ascending=[False, False]).head(5)
    strongest_meters = meter_df.sort_values(["mean_strong_exact_rate", "rows"], ascending=[False, False]).head(5)
    weakest_meters = meter_df.sort_values(["mean_strong_exact_rate", "rows"], ascending=[True, False]).head(5)
    hardest_bayts = bayt_df.sort_values(["hard_share", "requested_bayts"], ascending=[False, True]).head(5)
    easiest_bayts = bayt_df.sort_values(["mean_row_meter_score", "requested_bayts"], ascending=[False, True]).head(5)

    lines = [
        "# Meter+Count Paper Figure Pack",
        "",
        f"- run dir: `{summary['run_dir']}`",
        f"- rows analyzed: **{summary['row_count']:,}**",
        f"- candidate rows analyzed: **{summary['candidate_count']:,}**",
        "",
        "## Where the figures are",
        "",
        "- plots directory: `plots/`",
        "- table directory: `tables/`",
        "- machine-readable summary: `summary.json`",
        "",
        "Core figure files:",
        "- `plots/01_overall_difficulty_distribution.png`",
        "- `plots/02_difficulty_share_by_meter.png`",
        "- `plots/03_difficulty_share_by_requested_bayts.png`",
        "- `plots/04_mean_row_meter_score_by_meter.png`",
        "- `plots/05_mean_strong_exact_rate_by_meter.png`",
        "- `plots/06_rate_curves_by_requested_bayts.png`",
        "- `plots/07_candidate_meter_score_histogram.png`",
        "- `plots/08_num_strong_exact_candidates_distribution.png`",
        "- `plots/09_mean_within_row_meter_spread_by_meter.png`",
        "- `plots/10_mean_row_meter_score_by_meter_easy.png`",
        "- `plots/11_mean_row_meter_score_by_meter_medium.png`",
        "- `plots/12_mean_row_meter_score_by_meter_hard.png`",
        "- `plots/13_hard_share_heatmap_meter_x_length_bucket.png`",
        "",
        "## How to regenerate",
        "",
        "```bash",
        "cd /root/workspace/Shaer",
        "source grpo/.venv/bin/activate",
        "python grpo/plot_meter_count_paper_figures.py \\",
        "  --run-dir /root/workspace/Shaer/grpo/outputs/preprocess_meter_count/full_train_k8_20260409_192820",
        "```",
        "",
        "## Overall difficulty mix",
    ]
    for _, row in difficulty_df.iterrows():
        lines.append(f"- `{row['difficulty']}`: **{int(row['rows']):,}** rows ({100.0 * float(row['share']):.1f}%), mean row meter={float(row['mean_row_meter_score']):.3f}")

    lines.extend(["", "## Hardest meters by hard-share"])
    for _, row in hardest_meters.iterrows():
        lines.append(f"- `{row['base_meter']}` ({row['meter_display']}): hard={100.0 * float(row['hard_share']):.1f}%, mean row meter={float(row['mean_row_meter_score']):.3f}, strong exact rate={100.0 * float(row['mean_strong_exact_rate']):.1f}%")

    lines.extend(["", "## Strongest meters by strong-exact rate"])
    for _, row in strongest_meters.iterrows():
        lines.append(f"- `{row['base_meter']}` ({row['meter_display']}): strong exact rate={100.0 * float(row['mean_strong_exact_rate']):.1f}%, mean row meter={float(row['mean_row_meter_score']):.3f}, hard={100.0 * float(row['hard_share']):.1f}%")

    lines.extend(["", "## Weakest meters by strong-exact rate"])
    for _, row in weakest_meters.iterrows():
        lines.append(f"- `{row['base_meter']}` ({row['meter_display']}): strong exact rate={100.0 * float(row['mean_strong_exact_rate']):.1f}%, mean row meter={float(row['mean_row_meter_score']):.3f}, hard={100.0 * float(row['hard_share']):.1f}%")

    lines.extend(["", "## Most difficult requested-bayt counts by hard-share"])
    for _, row in hardest_bayts.iterrows():
        lines.append(f"- `{int(row['requested_bayts'])}` bayts: hard={100.0 * float(row['hard_share']):.1f}%, mean row meter={float(row['mean_row_meter_score']):.3f}, exact-count rate={100.0 * float(row['mean_count_exact_rate']):.1f}%")

    lines.extend(["", "## Strongest requested-bayt counts by mean row meter"])
    for _, row in easiest_bayts.iterrows():
        lines.append(f"- `{int(row['requested_bayts'])}` bayts: mean row meter={float(row['mean_row_meter_score']):.3f}, strong exact rate={100.0 * float(row['mean_strong_exact_rate']):.1f}%, hard={100.0 * float(row['hard_share']):.1f}%")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(run_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    tables_dir = ensure_dir(out_dir / "tables")

    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    summary, difficulty_df, meter_df, bayt_df, meter_length_df, meter_difficulty_df, candidate_meter_scores, strong_exact_dist = build_summary_tables(run_dir)

    meter_df = meter_df.sort_values(["rows", "base_meter"], ascending=[False, True]).reset_index(drop=True)
    difficulty_df.to_csv(tables_dir / "difficulty_summary.csv", index=False)
    meter_df.to_csv(tables_dir / "meter_summary.csv", index=False)
    bayt_df.to_csv(tables_dir / "requested_bayts_summary.csv", index=False)
    meter_length_df.to_csv(tables_dir / "meter_length_bucket_summary.csv", index=False)
    meter_difficulty_df.to_csv(tables_dir / "meter_difficulty_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    render_difficulty_counts(
        difficulty_df[["difficulty", "rows"]],
        plots_dir / "01_overall_difficulty_distribution.png",
    )

    render_percent_stacked_barh(
        meter_df.sort_values(["hard_share", "rows"], ascending=[False, False]).reset_index(drop=True),
        "meter_display",
        plots_dir / "02_difficulty_share_by_meter.png",
        title="Difficulty Composition by Base Meter",
        xlabel="Share of rows (%)",
    )

    render_line_shares(
        bayt_df,
        "requested_bayts",
        plots_dir / "03_difficulty_share_by_requested_bayts.png",
        title="Difficulty Share by Requested Bayts",
        xlabel="Requested bayts",
    )

    render_barh(
        meter_df.sort_values(["mean_row_meter_score", "rows"], ascending=[True, False]).reset_index(drop=True),
        "meter_display",
        "mean_row_meter_score",
        plots_dir / "04_mean_row_meter_score_by_meter.png",
        title="Mean Row Meter Score by Base Meter",
        xlabel="Mean row meter score",
        color="#4C78A8",
    )

    render_barh(
        meter_df.sort_values(["mean_strong_exact_rate", "rows"], ascending=[True, False]).reset_index(drop=True),
        "meter_display",
        "mean_strong_exact_rate",
        plots_dir / "05_mean_strong_exact_rate_by_meter.png",
        title="Mean Strong-Exact Rate by Base Meter",
        xlabel="Mean strong-exact rate",
        color="#2E8B57",
        fmt="{:.1%}",
    )

    render_rate_line(
        bayt_df,
        "requested_bayts",
        [
            ("mean_count_exact_rate", "exact-count rate", "#6C5CE7"),
            ("mean_strong_exact_rate", "strong-exact rate", "#2E8B57"),
            ("mean_strong_meter_rate", "strong-meter rate", "#4C78A8"),
        ],
        plots_dir / "06_rate_curves_by_requested_bayts.png",
        title="Count and Strong-Hit Rates by Requested Bayts",
        xlabel="Requested bayts",
        ylabel="Average rate (%)",
    )

    render_histogram(
        candidate_meter_scores,
        plots_dir / "07_candidate_meter_score_histogram.png",
        title="Candidate Meter Score Distribution",
        xlabel="Candidate meter score",
        bins=45,
    )

    render_distribution(
        strong_exact_dist,
        plots_dir / "08_num_strong_exact_candidates_distribution.png",
        title="Distribution of Strong-Exact Hits per Row (K=8)",
        xlabel="Number of strong-exact candidates in row",
        color="#2E8B57",
    )

    render_barh(
        meter_df.sort_values(["mean_within_row_meter_spread", "rows"], ascending=[False, False]).reset_index(drop=True),
        "meter_display",
        "mean_within_row_meter_spread",
        plots_dir / "09_mean_within_row_meter_spread_by_meter.png",
        title="Mean Within-Row Meter Spread by Base Meter",
        xlabel="Mean max(candidate meter) - min(candidate meter)",
        color="#B279A2",
    )

    plot_idx = 10
    for difficulty_name in DIFFICULTY_ORDER:
        subset = meter_difficulty_df[meter_difficulty_df["difficulty"] == difficulty_name].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["mean_row_meter_score", "rows"], ascending=[True, False]).reset_index(drop=True)
        render_barh(
            subset,
            "meter_display",
            "mean_row_meter_score",
            plots_dir / f"{plot_idx:02d}_mean_row_meter_score_by_meter_{difficulty_name}.png",
            title=f"Mean Row Meter Score by Base Meter ({difficulty_name.capitalize()} Rows)",
            xlabel="Mean row meter score within difficulty bucket",
            color=DIFFICULTY_COLORS[difficulty_name],
        )
        plot_idx += 1

    if not meter_length_df.empty:
        heat_df = meter_length_df.copy()
        heat_df["meter_display"] = heat_df["meter_display"].astype(str)
        heat_df["length_bucket"] = heat_df["length_bucket"].astype(str)
        render_heatmap(
            heat_df,
            "meter_display",
            "length_bucket",
            "hard_share",
            plots_dir / f"{plot_idx:02d}_hard_share_heatmap_meter_x_length_bucket.png",
            title="Hard-Row Share by Base Meter and Length Bucket",
        )

    write_report(out_dir / "README.md", summary=summary, difficulty_df=difficulty_df, meter_df=meter_df, bayt_df=bayt_df)
    print(f"Wrote figure pack to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
