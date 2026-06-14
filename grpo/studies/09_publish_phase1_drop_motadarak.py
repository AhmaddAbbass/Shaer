import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import ensure_dir, load_and_prepare_dataset, save_json, setup_logger

load_dotenv(override=False)


TARGET_BASE_METER = "المتدارك"
TARGET_REPO_DEFAULT = (
    "Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed-maxlen20-drop-majzuu-wafer-drop-motadarak"
)


def make_readme(
    upstream_repo: str,
    source_rows: int,
    removed_rows: int,
    final_rows: int,
    prepared_source_rows: int,
    prepared_final_rows: int,
) -> str:
    retention = 100.0 * final_rows / source_rows if source_rows else 0.0
    return f"""---
language:
- ar
license: apache-2.0
pretty_name: Ashaar v1 SFT Ready Locked Prompt Maxlen20 Drop Majzuu Wafer Drop Motadarak (2026-04-03)
task_categories:
- text-generation
size_categories:
- 100K<n<1M
---

# Ashaar v1 SFT-Ready (Locked Prompt, <= 2048 tokens, max 20 bayts, drop مجزوء الوافر, drop المتدارك)

This dataset is derived from `{upstream_repo}` and keeps the same schema, columns, locked prompt format, and general structure as the upstream phase-1 dataset.

The only additional change is the removal of rows where:

- `base_meter == "{TARGET_BASE_METER}"`

This removes all poems whose base meter is `المتدارك` from the published phase-1 subset.

## Why this variant exists

The phase-1 meter reward is explicitly base-meter-first. A dedicated base-only sanity study with multiple prompts per base meter and multiple sampled candidates per prompt showed that `المتدارك` remained a clear outlier even under a multi-sample selection view.

In that study:

- `المتدارك` had very weak candidate-level scores
- `المتدارك` also failed the prompt-level best-of-`k` criterion
- no prompt produced a candidate that crossed the useful quality thresholds used for review

Because the goal of this phase-1 subset is to align the training data with the available meter reward and keep RL signal reasonably clean, this derivative removes `المتدارك` from the current phase-1 dataset.

## Locked Prompt
### SYSTEM_PROMPT
أنت شاعر عربي تكتب الشعر العمودي الكلاسيكي.
التزم بالبحر المحدد في كل شطر، واستلهم من الموضوع دون نقله حرفياً.
أخرج الأبيات فقط دون مقدمة أو تعليق.

### USER_TEMPLATE
البحر الأساسي: {{base_meter}}
الصيغة: {{form}}
اسم البحر المطلوب: {{meter_label}}
الموضوع: {{description}}

اكتب {{num_lines}} شطراً ملتزماً بصيغة {{form}} من بحر {{base_meter}} دون أي شرح إضافي.

## Conditioning Rule
- `meter_label = base_meter` if `form == "تام"`
- else `meter_label = "{{form}} {{base_meter}}"`

## Added Columns
- `sft_prompt`
- `sft_completion`
- `sft_full_text`
- `sft_num_lines`
- `sft_total_tokens`

## Target Formatting
- `sft_completion` is built from `poem verses` using real newline characters.

## Filtering
- Inherits all filtering already present in `{upstream_repo}`.
- Additional filter applied here: drop rows where `base_meter == "{TARGET_BASE_METER}"`.

## Counts
- Source rows from `{upstream_repo}`: **{source_rows}**
- Removed by dropping `المتدارك`: **{removed_rows}**
- Final rows kept: **{final_rows}**
- Retention from upstream phase-1 subset: **{retention:.2f}%**

## Training-Prep Note
- After the repo's current `load_and_prepare_dataset(...)` preprocessing, the upstream phase-1 dataset yields **{prepared_source_rows}** usable rows.
- This derivative yields **{prepared_final_rows}** usable rows after the same preparation path.

## Important Meter-Reward Caveat
- The current meter reward is primarily a **base-meter correctness** signal.
- Form-sensitive meter realization is not directly validated when the classifier lacks that exact form label.
- This dataset change should therefore be understood as alignment with a **base-meter-first** reward, not as a claim that the dropped meter is impossible in general.

## Notes
- This is a derived dataset repo; upstream datasets are unchanged.
- The schema and column names are kept identical to the upstream dataset.
- This subset is intended for phase-1 GRPO experiments where the active meter-reward signal is more reliable on the retained base-meter set.
"""


def main():
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    source_repo = os.getenv("PHASE1_DATASET_ID", "").strip()
    target_repo = os.getenv("PHASE1_DATASET_ID_NEXT", "").strip() or TARGET_REPO_DEFAULT

    if not source_repo:
        raise ValueError("PHASE1_DATASET_ID is missing in .env")

    out_root = ensure_dir(Path("outputs/publish_phase1_drop_motadarak"))
    logger = setup_logger(
        "publish_phase1_drop_motadarak",
        out_root / "publish_phase1_drop_motadarak.log",
    )

    logger.info(f"source_repo={source_repo}")
    logger.info(f"target_repo={target_repo}")

    ds = load_dataset(source_repo, split="train", token=hf_token)
    source_rows = len(ds)
    logger.info(f"source_rows={source_rows}")

    filtered = ds.filter(
        lambda row: str(row.get("base_meter", "")).strip() != TARGET_BASE_METER
    )
    final_rows = len(filtered)
    removed_rows = source_rows - final_rows
    logger.info(f"removed_rows={removed_rows}")
    logger.info(f"final_rows={final_rows}")

    prepared_source = load_and_prepare_dataset(
        dataset_id=source_repo,
        split="train",
        max_bayts=os.getenv("PHASE1_MAX_BAYTS", "").strip() or None,
        hf_token=hf_token,
    )
    prepared_source_rows = len(prepared_source)
    prepared_final_rows = sum(
        1
        for row in prepared_source
        if str(row["base_meter"]).strip() != TARGET_BASE_METER
    )
    logger.info(f"prepared_source_rows={prepared_source_rows}")
    logger.info(f"prepared_final_rows={prepared_final_rows}")

    current_readme_path = hf_hub_download(
        repo_id=source_repo,
        filename="README.md",
        repo_type="dataset",
        token=hf_token,
    )
    current_readme = Path(current_readme_path).read_text(encoding="utf-8")
    Path(out_root / "upstream_README.md").write_text(current_readme, encoding="utf-8")

    readme_text = make_readme(
        upstream_repo=source_repo,
        source_rows=source_rows,
        removed_rows=removed_rows,
        final_rows=final_rows,
        prepared_source_rows=prepared_source_rows,
        prepared_final_rows=prepared_final_rows,
    )
    local_readme_path = out_root / "README.md"
    local_readme_path.write_text(readme_text, encoding="utf-8")

    summary = {
        "source_repo": source_repo,
        "target_repo": target_repo,
        "raw_source_rows": source_rows,
        "raw_removed_rows": removed_rows,
        "raw_final_rows": final_rows,
        "prepared_source_rows": prepared_source_rows,
        "prepared_removed_rows": prepared_source_rows - prepared_final_rows,
        "prepared_final_rows": prepared_final_rows,
        "drop_rule": {
            "base_meter": TARGET_BASE_METER,
        },
    }
    save_json(summary, out_root / "summary.json")

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)

    logger.info("pushing dataset shards to hub")
    filtered.push_to_hub(target_repo, token=hf_token)

    logger.info("uploading README to hub")
    api.upload_file(
        path_or_fileobj=str(local_readme_path),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="dataset",
    )

    logger.info("verifying remote dataset")
    verified = load_dataset(target_repo, split="train", token=hf_token)
    verified_rows = len(verified)
    if verified_rows != final_rows:
        raise RuntimeError(f"Verification failed: expected {final_rows}, got {verified_rows}")
    verified_motadarak_rows = sum(
        1 for row in verified if str(row.get("base_meter", "")).strip() == TARGET_BASE_METER
    )
    if verified_motadarak_rows != 0:
        raise RuntimeError(f"Verification failed: found {verified_motadarak_rows} remaining المتدارك rows")

    logger.info(f"verified_rows={verified_rows}")
    logger.info("verified_motadarak_rows=0")

    summary["verified_rows"] = verified_rows
    summary["verified_motadarak_rows"] = 0
    save_json(summary, out_root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
