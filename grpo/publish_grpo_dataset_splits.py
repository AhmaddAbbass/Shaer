import json
import os
import random
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

from rewards.common import ensure_dir, get_dataset_source_id_for_phase, load_and_prepare_dataset, save_json


ROOT = Path(__file__).resolve().parent


def infer_output_repo(source_id: str) -> str:
    configured = os.getenv("PHASE1_DATASET_ID", "").strip()
    if configured and configured != source_id:
        return configured

    if "/" in source_id:
        owner, name = source_id.split("/", 1)
    else:
        owner, name = "", source_id

    suffix = "-grpo-splits"
    candidate_name = name if name.endswith(suffix) else f"{name}{suffix}"
    max_name_len = 96 - (len(owner) + 1 if owner else 0)
    if len(candidate_name) <= max_name_len:
        return f"{owner}/{candidate_name}" if owner else candidate_name

    digest = hashlib.sha256(candidate_name.encode("utf-8")).hexdigest()[:8]
    base_budget = max_name_len - len(suffix) - len(digest) - 1
    if base_budget <= 8:
        raise RuntimeError(
            f"Cannot derive a valid output repo name from source dataset id: {source_id}"
        )
    truncated = name[:base_budget].rstrip("-._")
    candidate_name = f"{truncated}-{digest}{suffix}"
    return f"{owner}/{candidate_name}" if owner else candidate_name


def shuffled(values, rng):
    values = list(values)
    rng.shuffle(values)
    return values


def take_with_train_reserve(indices, want: int):
    indices = list(indices)
    max_take = max(0, len(indices) - 1)
    take = min(int(want), max_take)
    return indices[:take], indices[take:]


def allocate_bucket(indices, val_target, test_target):
    selected_val, remaining = take_with_train_reserve(indices, val_target)
    selected_test, remaining = take_with_train_reserve(remaining, test_target)
    return selected_val, selected_test, remaining


def add_records(rows, split_name, meter, bucket, indices):
    for source_index in indices:
        rows.append(
            {
                "split": split_name,
                "base_meter": meter,
                "length_bucket": bucket,
                "source_index": int(source_index),
            }
        )


def main():
    load_dotenv(override=False)

    cfg_path = ROOT / "grpo_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        import yaml

        cfg = yaml.safe_load(f)

    out_root = ensure_dir(ROOT / cfg["studies"]["publish_grpo_splits"]["output_root"])
    study_cfg = cfg["studies"]["publish_grpo_splits"]
    seed = int(study_cfg["seed"])
    rng = random.Random(seed)

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    source_id = get_dataset_source_id_for_phase()
    output_repo = infer_output_repo(source_id)

    prepared = load_and_prepare_dataset(
        dataset_id=source_id,
        split="train",
        hf_token=hf_token,
    )
    prepared_df = prepared.to_pandas()
    prepared_df["source_index"] = prepared_df["source_index"].astype(int)
    prepared_df["requested_bayts"] = prepared_df["requested_bayts"].astype(int)
    prepared_df["length_bucket"] = prepared_df["requested_bayts"].apply(
        lambda x: "short_le_8" if int(x) <= 8 else "full_gt_8"
    )

    raw = load_dataset(source_id, split="train", token=hf_token)
    source_indices = set(prepared_df["source_index"].tolist())
    raw_prepared_df = prepared_df.sort_values("source_index").reset_index(drop=True)

    meters = sorted(raw_prepared_df["base_meter"].dropna().unique().tolist())
    val_total_target = int(study_cfg["validation_total"])
    test_total_target = int(study_cfg["test_total"])
    val_short_target = int(study_cfg["validation_short_per_meter"])
    val_full_target = int(study_cfg["validation_full_per_meter"])
    test_short_target = int(study_cfg["test_short_per_meter"])
    test_full_target = int(study_cfg["test_full_per_meter"])

    per_meter_bucket = defaultdict(lambda: {"short_le_8": [], "full_gt_8": []})
    for row in raw_prepared_df.to_dict(orient="records"):
        per_meter_bucket[row["base_meter"]][row["length_bucket"]].append(int(row["source_index"]))

    for meter in meters:
        for bucket in ["short_le_8", "full_gt_8"]:
            per_meter_bucket[meter][bucket] = shuffled(per_meter_bucket[meter][bucket], rng)

    val_indices = set()
    test_indices = set()
    selection_rows = []
    meter_shortfalls = []

    leftovers_by_meter = defaultdict(list)

    for meter in meters:
        short_indices = per_meter_bucket[meter]["short_le_8"]
        full_indices = per_meter_bucket[meter]["full_gt_8"]

        val_short, test_short, rem_short = allocate_bucket(short_indices, val_short_target, test_short_target)
        val_full, test_full, rem_full = allocate_bucket(full_indices, val_full_target, test_full_target)

        add_records(selection_rows, "validation", meter, "short_le_8", val_short)
        add_records(selection_rows, "validation", meter, "full_gt_8", val_full)
        add_records(selection_rows, "test", meter, "short_le_8", test_short)
        add_records(selection_rows, "test", meter, "full_gt_8", test_full)

        val_indices.update(val_short)
        val_indices.update(val_full)
        test_indices.update(test_short)
        test_indices.update(test_full)

        shortfall = {
            "base_meter": meter,
            "validation_short_shortfall": max(0, val_short_target - len(val_short)),
            "validation_full_shortfall": max(0, val_full_target - len(val_full)),
            "test_short_shortfall": max(0, test_short_target - len(test_short)),
            "test_full_shortfall": max(0, test_full_target - len(test_full)),
        }

        leftovers_by_meter[meter].extend(rem_short)
        leftovers_by_meter[meter].extend(rem_full)

        for split_name, bucket_name, missing in [
            ("validation", "short_le_8", shortfall["validation_short_shortfall"]),
            ("validation", "full_gt_8", shortfall["validation_full_shortfall"]),
            ("test", "short_le_8", shortfall["test_short_shortfall"]),
            ("test", "full_gt_8", shortfall["test_full_shortfall"]),
        ]:
            if missing <= 0:
                continue
            pool = leftovers_by_meter[meter]
            if not pool:
                continue
            take_n = min(missing, max(0, len(pool)))
            taken = pool[:take_n]
            leftovers_by_meter[meter] = pool[take_n:]
            if split_name == "validation":
                val_indices.update(taken)
            else:
                test_indices.update(taken)
            add_records(selection_rows, split_name, meter, "backfill_same_meter", taken)
            shortfall_key = f"{split_name}_{bucket_name}_after_backfill_shortfall"
            shortfall[shortfall_key] = max(0, missing - len(taken))

        meter_shortfalls.append(shortfall)

    global_leftovers = []
    for meter in meters:
        global_leftovers.extend(leftovers_by_meter[meter])
    global_leftovers = shuffled(global_leftovers, rng)

    def fill_global(target_set, split_name, total_target):
        if len(target_set) >= total_target:
            return
        needed = total_target - len(target_set)
        taken = []
        while global_leftovers and len(taken) < needed:
            source_index = global_leftovers.pop(0)
            if source_index in val_indices or source_index in test_indices:
                continue
            taken.append(source_index)
        target_set.update(taken)
        add_records(selection_rows, split_name, "__global__", "global_backfill", taken)

    fill_global(val_indices, "validation", val_total_target)
    fill_global(test_indices, "test", test_total_target)

    overlap = sorted(val_indices & test_indices)
    if overlap:
        raise RuntimeError(f"validation/test overlap detected for source indices: {overlap[:10]}")

    train_indices = sorted(source_indices - val_indices - test_indices)
    if not train_indices:
        raise RuntimeError("Train split is empty after eval/test allocation.")

    validation_df = raw_prepared_df[raw_prepared_df["source_index"].isin(sorted(val_indices))]
    test_df = raw_prepared_df[raw_prepared_df["source_index"].isin(sorted(test_indices))]
    train_df = raw_prepared_df[raw_prepared_df["source_index"].isin(train_indices)]

    missing_val = sorted(set(meters) - set(validation_df["base_meter"].tolist()))
    missing_test = sorted(set(meters) - set(test_df["base_meter"].tolist()))
    if missing_val:
        raise RuntimeError(f"validation split missing base meters: {missing_val}")
    if missing_test:
        raise RuntimeError(f"test split missing base meters: {missing_test}")

    split_summary = {
        "source_dataset": source_id,
        "output_repo": output_repo,
        "seed": seed,
        "counts": {
            "train": int(len(train_indices)),
            "validation": int(len(val_indices)),
            "test": int(len(test_indices)),
        },
        "targets": {
            "validation_total": val_total_target,
            "test_total": test_total_target,
            "validation_short_per_meter": val_short_target,
            "validation_full_per_meter": val_full_target,
            "test_short_per_meter": test_short_target,
            "test_full_per_meter": test_full_target,
        },
        "meter_coverage": {
            "validation": sorted(validation_df["base_meter"].dropna().unique().tolist()),
            "test": sorted(test_df["base_meter"].dropna().unique().tolist()),
        },
        "length_bucket_counts": {
            "train": train_df["length_bucket"].value_counts().to_dict(),
            "validation": validation_df["length_bucket"].value_counts().to_dict(),
            "test": test_df["length_bucket"].value_counts().to_dict(),
        },
        "meter_shortfalls": meter_shortfalls,
    }

    save_json(split_summary, out_root / cfg["logging"]["split_summary_name"])
    pd.DataFrame(selection_rows).to_csv(out_root / "split_selection_rows.csv", index=False)
    pd.DataFrame(meter_shortfalls).to_csv(out_root / "meter_shortfalls.csv", index=False)

    prepared_lookup = {
        int(row["source_index"]): row
        for row in raw_prepared_df.to_dict(orient="records")
    }

    def build_split_rows(indices):
        rows = []
        for source_index in indices:
            raw_row = dict(raw[int(source_index)])
            prepared_row = prepared_lookup[int(source_index)]
            raw_row["source_index"] = int(source_index)
            raw_row["base_meter"] = prepared_row["base_meter"]
            raw_row["form"] = prepared_row["form"]
            raw_row["meter_label"] = prepared_row["meter_label"]
            raw_row["requested_bayts"] = int(prepared_row["requested_bayts"])
            raw_row["requested_lines"] = int(prepared_row["requested_lines"])
            raw_row["length_bucket"] = prepared_row["length_bucket"]
            rows.append(raw_row)
        return rows

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(build_split_rows(train_indices)),
            "validation": Dataset.from_list(build_split_rows(sorted(val_indices))),
            "test": Dataset.from_list(build_split_rows(sorted(test_indices))),
        }
    )

    create_repo(repo_id=output_repo, repo_type="dataset", token=hf_token, exist_ok=True)
    dataset_dict.push_to_hub(output_repo, token=hf_token)

    api = HfApi(token=hf_token)
    readme = "\n".join(
        [
            "---",
            "language:",
            "- ar",
            "license: apache-2.0",
            "---",
            "",
            "# GRPO Split Dataset",
            "",
            f"Source dataset: `{source_id}`",
            f"Validation size: `{len(val_indices)}`",
            f"Test size: `{len(test_indices)}`",
            "",
            "This split preserves the original schema and locked SFT prompt contract.",
            "",
            "Coverage policy:",
            "- validation contains at least one sample from every remaining phase-1 base meter",
            "- test contains at least one sample from every remaining phase-1 base meter",
            "- splits were built deterministically with seed 42",
            "",
        ]
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=output_repo,
        repo_type="dataset",
        token=hf_token,
    )

    print(json.dumps(split_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
