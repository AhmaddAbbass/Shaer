#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import math


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def to_df(path: Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "global_step" in df.columns:
        df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    return df


def parse_resume_step_and_parent(run_dir: Path) -> tuple[int | None, Path | None]:
    decision_path = run_dir / "resume_decision.json"
    if not decision_path.exists():
        return None, None
    try:
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    checkpoint_path = str(payload.get("checkpoint_repo_path") or "")
    if not checkpoint_path:
        checkpoint_path = str(payload.get("local_resume_path") or "")
    if not checkpoint_path:
        return None, None

    step_match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    step = int(step_match.group(1)) if step_match else None

    run_match = re.search(r"checkpoints/[^/]+/[^/]+/([^/]+)/checkpoint-\d+", checkpoint_path)
    parent = run_dir.parent / run_match.group(1) if run_match else None
    if parent is None:
        local_parent = Path(checkpoint_path).expanduser()
        if local_parent.name.startswith("checkpoint-"):
            parent = local_parent.parent
    if parent is not None and not parent.exists():
        parent = None
    return step, parent


def resolve_chain(run_dir: Path) -> list[Path]:
    chain: list[Path] = []
    seen: set[Path] = set()
    current = run_dir.resolve()
    while current not in seen and current.exists():
        chain.append(current)
        seen.add(current)
        _, parent = parse_resume_step_and_parent(current)
        if parent is None:
            break
        current = parent.resolve()
    chain.reverse()
    return chain


def combined_df(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, run_dir in enumerate(run_dirs):
        df = to_df(run_dir / filename)
        if df.empty:
            continue
        df = df.copy()
        df["source_run_name"] = run_dir.name
        resume_step, _ = parse_resume_step_and_parent(run_dir)
        if idx > 0 and resume_step is not None and "global_step" in df.columns:
            df.loc[df["global_step"] == 0, "global_step"] = resume_step
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "mode" in merged.columns and "global_step" in merged.columns:
        merged = merged.drop_duplicates(subset=["mode", "global_step"], keep="last")
    elif "global_step" in merged.columns:
        merged = merged.drop_duplicates(subset=["global_step"], keep="last")
    return merged.sort_values("global_step") if "global_step" in merged.columns else merged


def plot_loss_panels(run_dir: Path, metrics: pd.DataFrame, filename: str = "loss_panels.png", title_suffix: str = "") -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    train = metrics[metrics.get("mode").eq("train") if "mode" in metrics.columns else []].copy()
    live_eval = metrics[metrics.get("mode").eq("eval") if "mode" in metrics.columns else []].copy()
    test_eval = metrics[metrics.get("mode").eq("test_eval") if "mode" in metrics.columns else []].copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    if not train.empty and "loss" in train.columns:
        train = train.dropna(subset=["global_step", "loss"]).sort_values("global_step")
        train["loss"] = pd.to_numeric(train["loss"], errors="coerce")
        train["loss_ema"] = train["loss"].ewm(span=50, adjust=False).mean()
        axes[0].plot(train["global_step"], train["loss"], linewidth=0.8, alpha=0.25, label="train loss (raw)")
        axes[0].plot(train["global_step"], train["loss_ema"], linewidth=2.0, label="train loss (EMA span=50)")
        axes[0].legend()
    axes[0].set_title(f"Train Loss{title_suffix}")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    if not live_eval.empty and "eval_loss" in live_eval.columns:
        live_eval = live_eval.dropna(subset=["global_step", "eval_loss"]).sort_values("global_step")
        live_eval["eval_loss"] = pd.to_numeric(live_eval["eval_loss"], errors="coerce")
        axes[1].plot(live_eval["global_step"], live_eval["eval_loss"], marker="o", linewidth=1.4, label="eval loss")
    if not test_eval.empty and "test_loss" in test_eval.columns:
        test_eval = test_eval.dropna(subset=["global_step", "test_loss"]).sort_values("global_step")
        test_eval["test_loss"] = pd.to_numeric(test_eval["test_loss"], errors="coerce")
        axes[1].plot(test_eval["global_step"], test_eval["test_loss"], marker="o", linewidth=1.8, label="final test loss")
    axes[1].set_title(f"Eval Loss{title_suffix}")
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=160)
    plt.close(fig)


def plot_aux_loss_panels(run_dir: Path, metrics: pd.DataFrame, filename: str = "aux_loss_panels.png", title_suffix: str = "") -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty or "mode" not in metrics.columns:
        return

    train = metrics[metrics["mode"].eq("train")].copy()
    eval_aux = metrics[metrics["mode"].eq("eval_aux")].copy()
    test_aux = metrics[metrics["mode"].eq("test_aux_eval")].copy()

    train_cols = [c for c in ["train_ce_loss", "train_meter_loss", "train_total_loss"] if c in train.columns]
    eval_cols = [c for c in ["eval_ce_loss", "eval_meter_loss", "eval_total_loss"] if c in eval_aux.columns]
    test_cols = [c for c in ["test_ce_loss", "test_meter_loss", "test_total_loss"] if c in test_aux.columns]
    if not train_cols and not eval_cols and not test_cols:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    if not train.empty and train_cols:
        train = train.dropna(subset=["global_step"]).sort_values("global_step")
        for col in train_cols:
            series = pd.to_numeric(train[col], errors="coerce")
            if series.notna().any():
                axes[0].plot(train["global_step"], series, linewidth=1.4, marker="o", markersize=2.5, label=col.replace("train_", ""))
        axes[0].legend()
    axes[0].set_title(f"Train Auxiliary Losses{title_suffix}")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    if not eval_aux.empty and eval_cols:
        eval_aux = eval_aux.dropna(subset=["global_step"]).sort_values("global_step")
        for col in eval_cols:
            series = pd.to_numeric(eval_aux[col], errors="coerce")
            if series.notna().any():
                axes[1].plot(eval_aux["global_step"], series, linewidth=1.6, marker="o", label=col.replace("eval_", "eval "))
    if not test_aux.empty and test_cols:
        test_aux = test_aux.dropna(subset=["global_step"]).sort_values("global_step")
        for col in test_cols:
            series = pd.to_numeric(test_aux[col], errors="coerce")
            if series.notna().any():
                axes[1].plot(test_aux["global_step"], series, linewidth=1.6, marker="x", linestyle="--", label=col.replace("test_", "test "))
    if axes[1].has_data():
        axes[1].legend()
    axes[1].set_title(f"Eval/Test Auxiliary Losses{title_suffix}")
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=160)
    plt.close(fig)


def plot_probe_panels(run_dir: Path, probe_metrics: pd.DataFrame, filename: str = "probe_panels.png", heatmap_filename: str = "probe_meter_heatmap.png", title_suffix: str = "") -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if probe_metrics.empty:
        return
    probe_metrics = probe_metrics.dropna(subset=["global_step"]).sort_values("global_step")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    if "probe_meter_mean" in probe_metrics.columns:
        axes[0].plot(probe_metrics["global_step"], probe_metrics["probe_meter_mean"], marker="o", linewidth=1.8)
    axes[0].set_title(f"Probe Meter Mean{title_suffix}")
    axes[0].set_ylabel("Meter Score")
    axes[0].grid(True, alpha=0.3)

    if "probe_count_adherence_mean" in probe_metrics.columns:
        axes[1].plot(probe_metrics["global_step"], probe_metrics["probe_count_adherence_mean"], marker="o", linewidth=1.8)
    axes[1].set_title(f"Probe Count Adherence Mean{title_suffix}")
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Count Score")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=160)
    plt.close(fig)

    records = []
    for _, row in probe_metrics.iterrows():
        per_meter = row.get("per_meter_probe_meter_mean", {})
        if isinstance(per_meter, str):
            try:
                per_meter = json.loads(per_meter)
            except Exception:
                per_meter = {}
        if not isinstance(per_meter, dict):
            continue
        for meter, value in per_meter.items():
            records.append({"global_step": row["global_step"], "base_meter": meter, "probe_meter_mean": float(value)})
    if not records:
        return
    heat = pd.DataFrame(records).pivot(index="base_meter", columns="global_step", values="probe_meter_mean").sort_index()
    fig, ax = plt.subplots(figsize=(14, max(6, len(heat) * 0.45)))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([str(int(x)) for x in heat.columns], rotation=45, ha="right")
    ax.set_title(f"Per-Meter Probe Meter Mean{title_suffix}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / heatmap_filename, dpi=160)
    plt.close(fig)


def plot_probe_per_meter_lines(
    run_dir: Path,
    probe_metrics: pd.DataFrame,
    filename: str = "probe_meter_small_multiples.png",
    title_suffix: str = "",
) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if probe_metrics.empty:
        return
    probe_metrics = probe_metrics.dropna(subset=["global_step"]).sort_values("global_step")

    records = []
    for _, row in probe_metrics.iterrows():
        per_meter = row.get("per_meter_probe_meter_mean", {})
        if isinstance(per_meter, str):
            try:
                per_meter = json.loads(per_meter)
            except Exception:
                per_meter = {}
        if not isinstance(per_meter, dict):
            continue
        for meter, value in per_meter.items():
            records.append(
                {
                    "global_step": float(row["global_step"]),
                    "base_meter": meter,
                    "probe_meter_mean": float(value),
                }
            )
    if not records:
        return

    per_meter_df = pd.DataFrame(records).sort_values(["base_meter", "global_step"])
    meters = sorted(per_meter_df["base_meter"].dropna().unique().tolist())
    ncols = 3
    nrows = math.ceil(len(meters) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(9, nrows * 3.5)), sharex=True, sharey=True)
    axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for ax, meter in zip(axes_list, meters):
        meter_df = per_meter_df[per_meter_df["base_meter"] == meter]
        ax.plot(
            meter_df["global_step"],
            meter_df["probe_meter_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4,
        )
        best_idx = meter_df["probe_meter_mean"].idxmax()
        if pd.notna(best_idx):
            best_row = meter_df.loc[best_idx]
            ax.scatter(
                [best_row["global_step"]],
                [best_row["probe_meter_mean"]],
                s=36,
                color="crimson",
                zorder=3,
            )
        ax.set_title(meter)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    for ax in axes_list[len(meters) :]:
        ax.axis("off")

    fig.suptitle(f"Per-Meter Probe Accuracy Over Time{title_suffix}", fontsize=14)
    for row_idx in range(nrows):
        axes_list[row_idx * ncols].set_ylabel("Meter Score")
    for ax in axes_list[max(0, (nrows - 1) * ncols) : min(len(axes_list), nrows * ncols)]:
        if ax.has_data():
            ax.set_xlabel("Global Step")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plots_dir / filename, dpi=160)
    plt.close(fig)


def render(run_dir: Path, follow_chain: bool = False) -> None:
    metrics = to_df(run_dir / "metrics.jsonl")
    probe_metrics = to_df(run_dir / "probe_metrics.jsonl")
    plot_loss_panels(run_dir, metrics)
    plot_aux_loss_panels(run_dir, metrics)
    plot_probe_panels(run_dir, probe_metrics)
    plot_probe_per_meter_lines(run_dir, probe_metrics)
    if not follow_chain:
        return

    chain = resolve_chain(run_dir)
    if len(chain) <= 1:
        return
    chain_metrics = combined_df(chain, "metrics.jsonl")
    chain_probe_metrics = combined_df(chain, "probe_metrics.jsonl")
    plot_loss_panels(run_dir, chain_metrics, filename="loss_panels_chain.png", title_suffix=" (Resume Chain)")
    plot_aux_loss_panels(run_dir, chain_metrics, filename="aux_loss_panels_chain.png", title_suffix=" (Resume Chain)")
    plot_probe_panels(
        run_dir,
        chain_probe_metrics,
        filename="probe_panels_chain.png",
        heatmap_filename="probe_meter_heatmap_chain.png",
        title_suffix=" (Resume Chain)",
    )
    plot_probe_per_meter_lines(
        run_dir,
        chain_probe_metrics,
        filename="probe_meter_small_multiples_chain.png",
        title_suffix=" (Resume Chain)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Live plotter for SFT")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--follow-chain", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=30)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    while True:
        render(run_dir, follow_chain=args.follow_chain)
        if not args.watch:
            return 0
        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
