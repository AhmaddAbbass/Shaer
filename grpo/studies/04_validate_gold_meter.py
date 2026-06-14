import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import ensure_dir, get_dataset_id_for_phase, load_and_prepare_dataset, save_json, setup_logger
from rewards.meter import get_meter_predictor, score_meter_poem

load_dotenv(override=False)


def score_candidates(frame, require_odd_tail, seed=42, max_per_label=8):
    if len(frame) == 0:
        return []

    parts = []
    for meter_label, group in frame.groupby("meter_label"):
        take = min(max_per_label, len(group))
        parts.append(group.sample(n=take, random_state=seed))

    candidate_df = pd.concat(parts, ignore_index=True)
    rows = []
    for _, row in candidate_df.iterrows():
        score = score_meter_poem(
            row["poem_text"],
            target_meter=row["meter_label"],
            base_meter=row["base_meter"],
            aggregator="logmean",
        )
        rows.append({
            **row.to_dict(),
            "score": score["score"],
            "mean_score": score["mean_score"],
            "std_score": score["std_score"],
            "num_valid_bayts": score["num_valid_bayts"],
            "num_skipped_bayts": score["num_skipped_bayts"],
            "target_meter_used": score["target_meter_used"],
            "target_resolution": score["target_resolution"],
            "has_odd_tail": bool(row["has_odd_tail"]),
            "selection_group": "odd" if require_odd_tail else "even",
        })

    rows.sort(
        key=lambda r: (
            r["score"],
            r["mean_score"],
            -r["std_score"],
            r["num_valid_bayts"],
        ),
        reverse=True,
    )
    return rows


def pick_samples(ds, total_samples=10, odd_tail_samples=3, seed=42):
    df = ds.to_pandas()

    # Favor examples with at least 2 complete bayts so "stable across bayts"
    # means something in the sanity artifact.
    df = df[df["requested_bayts"] >= 2].copy()

    odd_df = df[df["has_odd_tail"]].copy()
    even_df = df[~df["has_odd_tail"]].copy()

    odd_candidates = score_candidates(odd_df, require_odd_tail=True, seed=seed, max_per_label=6)
    even_candidates = score_candidates(even_df, require_odd_tail=False, seed=seed, max_per_label=8)

    picks = []
    used_indices = set()
    used_labels = set()

    def _take_rows(candidates, count):
        nonlocal picks
        if count <= 0 or not candidates:
            return

        for row in candidates:
            if len(picks) >= total_samples:
                return
            source_index = int(row["source_index"])
            if source_index in used_indices:
                continue
            meter_label = str(row["meter_label"])
            if meter_label in used_labels and len(used_labels) < total_samples:
                continue
            if row["num_valid_bayts"] <= 0:
                continue
            used_indices.add(source_index)
            used_labels.add(meter_label)
            picks.append(row)
            if sum(1 for x in picks if x["has_odd_tail"]) >= odd_tail_samples and row["selection_group"] == "odd":
                return
            if row["selection_group"] != "odd" and len(picks) >= total_samples:
                return

    _take_rows(odd_candidates, odd_tail_samples)
    _take_rows(even_candidates, total_samples - len(picks))

    if len(picks) < total_samples:
        remaining = odd_candidates + even_candidates
        for row in remaining:
            if int(row["source_index"]) in used_indices:
                continue
            picks.append(row)
            if len(picks) >= total_samples:
                break

    return picks[:total_samples]


def main():
    dataset_id = get_dataset_id_for_phase()
    out_root = Path("outputs/validate_gold_meter")
    ensure_dir(out_root)
    logger = setup_logger("validate_gold_meter", out_root / "validate_gold_meter.log")

    ds = load_and_prepare_dataset(
        dataset_id=dataset_id,
        split="train",
        max_bayts=os.getenv("PHASE1_MAX_BAYTS", "").strip() or None,
        allowed_meters=None,
        hf_token=os.getenv("HF_TOKEN", "").strip() or None,
    )
    predictor = get_meter_predictor()

    dataset_labels = sorted(set(ds["meter_label"]))
    model_labels = sorted(predictor.classes)
    compatibility = {
        "dataset_id": dataset_id,
        "dataset_meter_labels": dataset_labels,
        "model_meter_labels": model_labels,
        "missing_in_model": sorted(set(dataset_labels) - set(model_labels)),
        "extra_in_model": sorted(set(model_labels) - set(dataset_labels)),
    }
    save_json(compatibility, out_root / "label_compatibility.json")

    rows = pick_samples(ds, total_samples=10, odd_tail_samples=3, seed=42)
    logger.info(f"selected_samples={len(rows)}")

    sample_records = []
    bayt_records = []

    for sample_idx, row in enumerate(rows):
        score = score_meter_poem(
            row["poem_text"],
            target_meter=row["meter_label"],
            base_meter=row["base_meter"],
            aggregator="logmean",
        )

        rec = {
            "sample_index": sample_idx,
            "source_index": int(row["source_index"]),
            "meter_label": row["meter_label"],
            "base_meter": row["base_meter"],
            "form": row["form"],
            "requested_lines": int(row["requested_lines"]),
            "requested_bayts": int(row["requested_bayts"]),
            "has_odd_tail": bool(row["has_odd_tail"]),
            "target_meter_used": score["target_meter_used"],
            "target_resolution": score["target_resolution"],
            "score": score["score"],
            "mean_score": score["mean_score"],
            "logmean_score": score["logmean_score"],
            "std_score": score["std_score"],
            "min_score": score["min_score"],
            "max_score": score["max_score"],
            "num_valid_bayts": score["num_valid_bayts"],
            "num_skipped_bayts": score["num_skipped_bayts"],
            "odd_tail_line": score["odd_tail_line"],
            "description_preview": str(row["description"])[:220],
            "poem_preview": str(row["poem_text"])[:320],
        }
        sample_records.append(rec)
        logger.info(json.dumps(rec, ensure_ascii=False))

        for detail in score["per_bayt_details"]:
            bayt_rec = {
                "sample_index": sample_idx,
                "source_index": int(row["source_index"]),
                "meter_label": row["meter_label"],
                "base_meter": row["base_meter"],
                **detail,
            }
            bayt_records.append(bayt_rec)

    samples_df = pd.DataFrame(sample_records)
    bayts_df = pd.DataFrame(bayt_records)
    samples_df.to_csv(out_root / "gold_meter_scores.csv", index=False)
    bayts_df.to_csv(out_root / "gold_meter_per_bayt.csv", index=False)

    summary = {
        "dataset_id": dataset_id,
        "num_samples": int(len(samples_df)),
        "num_odd_tail_samples": int(samples_df["has_odd_tail"].sum()) if len(samples_df) else 0,
        "mean_score": float(samples_df["score"].mean()) if len(samples_df) else 0.0,
        "min_score": float(samples_df["score"].min()) if len(samples_df) else 0.0,
        "max_score": float(samples_df["score"].max()) if len(samples_df) else 0.0,
        "mean_std_score": float(samples_df["std_score"].mean()) if len(samples_df) else 0.0,
        "target_resolution_counts": samples_df["target_resolution"].value_counts().to_dict() if len(samples_df) else {},
    }
    save_json(summary, out_root / "summary.json")
    save_json(sample_records, out_root / "selected_samples.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
