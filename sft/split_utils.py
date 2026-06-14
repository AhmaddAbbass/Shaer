from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

from datasets import Dataset, DatasetDict


TRAIN_FRACTION = 0.94
EVAL_FRACTION = 0.03
TEST_FRACTION = 0.03
WEIGHT_ALPHA = 0.6
MIN_SAFE_STRATIFY_GROUP_SIZE = 8


def length_bucket_from_bayts(requested_bayts: int) -> str:
    bayts = max(1, int(requested_bayts))
    if bayts <= 3:
        return "1-3"
    if bayts <= 6:
        return "4-6"
    if bayts <= 10:
        return "7-10"
    return "11-20"


def _requested_bayts_from_row(row: dict[str, Any]) -> int:
    raw = row.get("requested_bayts")
    if raw is not None and str(raw).strip():
        return max(1, int(raw))
    raw_num_lines = row.get("sft_num_lines")
    if raw_num_lines is not None and str(raw_num_lines).strip():
        return max(1, int(raw_num_lines) // 2)
    verses = row.get("poem verses") or []
    if isinstance(verses, list) and verses:
        return max(1, len(verses) // 2)
    return 1


def _meter_from_row(row: dict[str, Any]) -> str:
    return str(row.get("base_meter") or "").strip() or "UNKNOWN_METER"


def _form_from_row(row: dict[str, Any]) -> str:
    return str(row.get("form") or "").strip() or "UNKNOWN_FORM"


def _fine_group_key(base_meter: str, form: str, length_bucket: str) -> str:
    return f"{base_meter}||{form}||{length_bucket}"


def annotate_split_columns(ds: Dataset, min_group_size: int = MIN_SAFE_STRATIFY_GROUP_SIZE) -> Dataset:
    fine_counts = Counter()
    meter_form_counts = Counter()
    meter_counts = Counter()

    for row in ds:
        base_meter = _meter_from_row(row)
        form = _form_from_row(row)
        requested_bayts = _requested_bayts_from_row(row)
        length_bucket = length_bucket_from_bayts(requested_bayts)
        fine_key = _fine_group_key(base_meter, form, length_bucket)
        meter_form_key = f"{base_meter}||{form}"
        fine_counts[fine_key] += 1
        meter_form_counts[meter_form_key] += 1
        meter_counts[base_meter] += 1

    def _annotate(row: dict[str, Any], idx: int) -> dict[str, Any]:
        source_index = row.get("source_index")
        if source_index is None or str(source_index).strip() == "":
            source_index = idx
        base_meter = _meter_from_row(row)
        form = _form_from_row(row)
        requested_bayts = _requested_bayts_from_row(row)
        length_bucket = length_bucket_from_bayts(requested_bayts)
        fine_key = _fine_group_key(base_meter, form, length_bucket)
        meter_form_key = f"{base_meter}||{form}"
        if fine_counts[fine_key] >= min_group_size:
            split_group = fine_key
            split_group_level = "base_meter_form_length_bucket"
        elif meter_form_counts[meter_form_key] >= min_group_size:
            split_group = meter_form_key
            split_group_level = "base_meter_form"
        elif meter_counts[base_meter] >= min_group_size:
            split_group = base_meter
            split_group_level = "base_meter"
        else:
            split_group = "__global__"
            split_group_level = "global"
        return {
            "source_index": int(source_index),
            "base_meter": base_meter,
            "form": form,
            "requested_bayts": int(requested_bayts),
            "length_bucket": length_bucket,
            "sampler_group": fine_key,
            "split_group": split_group,
            "split_group_level": split_group_level,
        }

    return ds.map(_annotate, with_indices=True, desc="annotate_split_columns")


def _allocate_quotas(
    counts_by_group: dict[str, int],
    target_total: int,
    fraction: float,
    capacity_by_group: dict[str, int] | None = None,
) -> dict[str, int]:
    quotas = {}
    residuals = []
    capacity = capacity_by_group or counts_by_group
    for group, count in counts_by_group.items():
        cap = int(capacity.get(group, count))
        exact = float(count) * float(fraction)
        base_quota = min(cap, int(math.floor(exact)))
        quotas[group] = base_quota
        residuals.append((exact - math.floor(exact), count, group))

    remaining = int(target_total) - sum(quotas.values())
    for _, _, group in sorted(residuals, key=lambda item: (item[0], item[1], item[2]), reverse=True):
        if remaining <= 0:
            break
        if quotas[group] >= int(capacity.get(group, counts_by_group[group])):
            continue
        quotas[group] += 1
        remaining -= 1

    if remaining > 0:
        for group, count in sorted(counts_by_group.items(), key=lambda item: (item[1], item[0]), reverse=True):
            if remaining <= 0:
                break
            cap = int(capacity.get(group, count))
            while remaining > 0 and quotas[group] < cap:
                quotas[group] += 1
                remaining -= 1

    if remaining != 0:
        raise RuntimeError(f"failed to allocate split quotas exactly; remaining={remaining}")
    return quotas


def _distribution_counts(ds: Dataset, column: str) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in ds[column]).items()))


def build_distribution_report(train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset) -> dict[str, Any]:
    report = {}
    for column in ["base_meter", "form", "length_bucket"]:
        report[column] = {
            "train": _distribution_counts(train_ds, column),
            "eval": _distribution_counts(eval_ds, column),
            "test": _distribution_counts(test_ds, column),
        }
    return report


def split_dataset(
    ds: Dataset,
    seed: int,
    train_fraction: float = TRAIN_FRACTION,
    eval_fraction: float = EVAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    min_group_size: int = MIN_SAFE_STRATIFY_GROUP_SIZE,
) -> tuple[DatasetDict, dict[str, Any]]:
    if not math.isclose(train_fraction + eval_fraction + test_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1.0")

    annotated = annotate_split_columns(ds, min_group_size=min_group_size)
    split_groups = [str(item) for item in annotated["split_group"]]
    levels = [str(item) for item in annotated["split_group_level"]]
    counts_by_group = Counter(split_groups)
    counts_by_level = Counter(levels)
    total_rows = len(annotated)

    eval_target = int(round(total_rows * eval_fraction))
    test_target = int(round(total_rows * test_fraction))
    train_target = total_rows - eval_target - test_target

    eval_quotas = _allocate_quotas(dict(counts_by_group), eval_target, eval_fraction)
    remaining_capacity = {group: counts_by_group[group] - eval_quotas[group] for group in counts_by_group}
    test_quotas = _allocate_quotas(dict(counts_by_group), test_target, test_fraction, capacity_by_group=remaining_capacity)

    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(split_groups):
        by_group[group].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    test_indices: list[int] = []

    for group in sorted(by_group):
        idxs = list(by_group[group])
        rng.shuffle(idxs)
        eval_count = int(eval_quotas[group])
        test_count = int(test_quotas[group])
        eval_indices.extend(idxs[:eval_count])
        test_indices.extend(idxs[eval_count : eval_count + test_count])
        train_indices.extend(idxs[eval_count + test_count :])

    train_indices = sorted(train_indices)
    eval_indices = sorted(eval_indices)
    test_indices = sorted(test_indices)

    dataset_dict = DatasetDict(
        {
            "train": annotated.select(train_indices),
            "eval": annotated.select(eval_indices),
            "test": annotated.select(test_indices),
        }
    )

    split_counts = {name: len(split) for name, split in dataset_dict.items()}
    if split_counts["train"] != train_target or split_counts["eval"] != eval_target or split_counts["test"] != test_target:
        raise RuntimeError(
            "split counts do not match targets: "
            f"actual={split_counts} target={{'train': {train_target}, 'eval': {eval_target}, 'test': {test_target}}}"
        )

    _, train_weight_report = build_train_weights(dataset_dict["train"], alpha=WEIGHT_ALPHA)
    summary = {
        "seed": int(seed),
        "rows_total": total_rows,
        "train_fraction": float(train_fraction),
        "eval_fraction": float(eval_fraction),
        "test_fraction": float(test_fraction),
        "split_counts": split_counts,
        "stratify_group_count": len(counts_by_group),
        "stratify_group_level_counts": dict(sorted(counts_by_level.items())),
        "distribution_tables": build_distribution_report(dataset_dict["train"], dataset_dict["eval"], dataset_dict["test"]),
        "train_weight_report": train_weight_report,
        "top_split_groups_by_size": [
            {"group": group, "count": int(counts_by_group[group])}
            for group, _ in counts_by_group.most_common(20)
        ],
        "small_group_examples": [
            {"group": group, "count": int(count)}
            for group, count in sorted(counts_by_group.items(), key=lambda item: (item[1], item[0]))[:20]
        ],
    }
    return dataset_dict, summary


def build_train_weights(train_ds: Dataset, alpha: float = WEIGHT_ALPHA) -> tuple[list[float], dict[str, Any]]:
    group_counts = Counter(str(group) for group in train_ds["sampler_group"])
    raw_weights = {group: (1.0 / float(count)) ** float(alpha) for group, count in group_counts.items()}
    mean_weight = sum(raw_weights.values()) / len(raw_weights) if raw_weights else 1.0
    normalized_weights = {group: (weight / mean_weight) for group, weight in raw_weights.items()}
    sample_weights = [float(normalized_weights[str(group)]) for group in train_ds["sampler_group"]]

    ranked = sorted(normalized_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)
    report = {
        "alpha": float(alpha),
        "num_groups": len(group_counts),
        "top_weighted_groups": [
            {"group": group, "weight": float(weight), "count": int(group_counts[group])}
            for group, weight in ranked[:20]
        ],
        "bottom_weighted_groups": [
            {"group": group, "weight": float(weight), "count": int(group_counts[group])}
            for group, weight in list(reversed(ranked[-20:]))
        ],
    }
    return sample_weights, report
