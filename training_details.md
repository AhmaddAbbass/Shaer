# Shaer SFT Training Details

Last verified from local repo artifacts and Hugging Face repositories on 2026-05-26.

This document records the SFT-only lineage for the Shaer model. It is intended as a paper-writing reference, so it favors explicit provenance, concrete counts, and reproducible configuration over narrative brevity.

## 1. Project Purpose

Shaer is a classical Arabic vertical-poetry generation project. The SFT stage trains a base Arabic language model to produce poem hemistichs conditioned on:

- `base_meter`: the Arabic meter family.
- `form`: the metrical form, such as `تام`, `مجزوء`, `مخلع`, or `أحذ`.
- requested line count in hemistichs.
- a semantic topic/description.

The target output is only the poem text, with no prose introduction or explanation. The SFT model is a LoRA/QLoRA adapter over `Navid-AI/Yehia-7B-preview`, published as `Shaer-AI/Shaer-adapters`.

GRPO exists elsewhere in the repo, but the SFT paper baseline is the completed fresh SFT run:

- local run: `sft/outputs/train/train_20260407_231929`
- HF adapter repo: `Shaer-AI/Shaer-adapters`
- base model: `Navid-AI/Yehia-7B-preview`
- final training dataset actually used: `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

Important clarification: the dataset named in the prompt, `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`, is the final unsplit SFT-ready dataset. The completed SFT run trained on the deterministic split derivative, `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`.

## 2. Core Ideas In This Document

If you only review the high-level claims before approving/disproving details, these are the core ideas:

1. The final SFT adapter is a fresh QLoRA/LoRA run over `Navid-AI/Yehia-7B-preview`, published as `Shaer-AI/Shaer-adapters`.
2. The final SFT run did not train directly on the unsplit dataset `...lte20-min500`; it trained on `...lte20-min500-splits`.
3. The unsplit `...lte20-min500` dataset is still the central final SFT source because the split repo is derived from it without changing rows, except adding split annotations.
4. The main data improvement was replacing the old `description` conditioning field with Qwen-generated `enhanced_description`.
5. The final SFT prompt was rebuilt from `enhanced_description`, while the target completion stayed the original poem hemistichs.
6. The final dataset was filtered to valid even-shatr poems, `requested_bayts <= 20`, and base meters with at least 500 examples.
7. The final split was deterministic 94/3/3, stratified by `base_meter || form || length_bucket`.
8. Training used train-only weighted sampling over the same joint group, with weight proportional to `(1 / group_count) ** 0.6`.
9. Eval/test were natural and unweighted.
10. The run used completion-only loss, sequence packing for train, max length 1024, bf16 4-bit QLoRA, `r=64`, `alpha=128`, `dropout=0.05`, and RS-LoRA.
11. The best eval checkpoint was step 3000, with eval loss 2.207448.
12. The best generation-side probe-meter checkpoint was step 2800, so the best CE checkpoint and best meter checkpoint are not identical.
13. Final held-out test loss was 2.193206.
14. Final probe count adherence was 1.0, while final probe meter mean was 0.508705.
15. Current HF retained checkpoints are 4000, 4200, and 4236, but the best adapter export is preserved separately under `adapters/fresh_sft/train/best`.

The main thing to approve or challenge is the dataset lineage. The final SFT part is strongly supported by local run artifacts and the HF model repo. The earlier dataset chain is partly documented in HF dataset cards and partly reconstructed from dataset names, schemas, row counts, and creation order.

## 3. Full Upstream Dataset Lineage

This section traces the dataset farther back than the final SFT source. There are two levels of confidence:

- **direct chain**: explicitly stated in HF cards or implemented by local scripts.
- **visible upstream/prehistory**: visible Shaer-AI datasets that explain the older raw/description pipeline, but where the current repo snapshot does not contain every transformation script.

The strongest direct chain for the final SFT baseline is:

```text
original Ashaar corpus
  -> Shaer-AI/ashaar-v1-base-form
  -> Shaer-AI/ashaar-v1-base-form-with-descriptions
  -> Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed
  -> Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500
  -> Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits
  -> Shaer-AI/Shaer-adapters
```

The visible earlier description/preprocessing prehistory is:

```text
original Ashaar corpus
  -> Shaer-AI/ashaar-preprocessed
  -> Shaer-AI/ashaar-preprocessed-desc
  -> Shaer-AI/ashaar-full-desc / Shaer-AI/ashaar-full-desc-preprocessed
```

The part from `ashaar-v1-base-form-with-descriptions` onward is directly stated in dataset cards and scripts. The earlier `ashaar-preprocessed*` / `ashaar-full-desc*` path is useful for explaining how Ashaar text and old descriptions existed before the final SFT pipeline, but it should be described as reconstructed from HF repo sequence, schemas, and counts unless older notebooks/scripts are recovered.

### 3.1 Original Ashaar Corpus

The `ashaar-v1-base-form` dataset card says it is derived from the original Ashaar corpus. In this repo/HF lineage, the earliest visible Shaer-AI dataset artifact representing that raw corpus is around:

- `Shaer-AI/ashaar-with-poem-id` (private, created 2025-11-19)
- `Shaer-AI/ashaar-preprocessed` (private, created 2025-11-19)

The `ashaar-preprocessed` schema shows raw-ish fields with spaces and nested description structures:

- `poem title`
- `poem meter`
- `poem verses`
- `poem theme`
- `poem url`
- `poet name`
- `poet description`
- `poet url`
- `poet era`
- `poet location`
- `poem description` as a nested/list/struct field
- `poem language type`
- `num_verses`
- `poem id`

Published metadata for `ashaar-preprocessed`:

- split: train
- rows: 145,167
- dataset size: 367,158,108 bytes

Interpretation: this is the earliest inspected preprocessed corpus artifact. It still preserves source-style metadata and a nested `poem description` representation, rather than the final simplified SFT schema.

Evidence status:

- row count and schema are verified from HF dataset metadata.
- exact scraper/source extraction code is not present in this repo snapshot.
- paper wording should say "derived from the original Ashaar corpus" because that is what the `ashaar-v1-base-form` card states, not that this repo fully documents raw scraping.

### 3.2 Description Flattening / Description Preprocessing

Next visible public artifact:

- `Shaer-AI/ashaar-preprocessed-desc`

Published metadata:

- created: 2025-11-21
- rows: 145,167
- split: train

Schema changed from space-separated names and nested description structures into cleaner snake_case fields:

- `poem_title`
- `poem_meter`
- `poem_verses`
- `poem_theme`
- `poem_url`
- `poet_name`
- `poet_description`
- `poet_url`
- `poet_era`
- `poet_location`
- `poem_language_type`
- `num_verses`
- `poem_id`
- `poem_description`

Interpretation: this step appears to flatten or normalize the original nested `poem description` into a string-like `poem_description`, while preserving the same 145,167-row scale.

Related worker artifacts visible on HF:

- `Shaer-AI/ashaar-preprocessed-desc-worker0`
- `Shaer-AI/ashaar-preprocessed-desc-worker1`
- `Shaer-AI/ashaar-preprocessed-desc-worker2`

These indicate the description preprocessing was likely sharded, but the exact local worker code for this older 2025 stage is not present in the current repo snapshot.

Evidence status:

- same row count as `ashaar-preprocessed` is verified: 145,167.
- schema simplification is verified.
- the exact flattening implementation is inferred from schema change.

### 3.3 Postworker / Full Description Datasets

Visible follow-up datasets:

- `Shaer-AI/ashaar-preprocessed-postworkers`
- `Shaer-AI/ashaar-preprocessed-postworkers-2`
- `Shaer-AI/ashaar-preprocessed-postworkers-worker0`
- `Shaer-AI/ashaar-preprocessed-postworkers-worker1`
- `Shaer-AI/ashaar-preprocessed-postworkers-worker2`
- `Shaer-AI/ashaar-full-desc`

Published metadata for `ashaar-preprocessed-postworkers`:

- rows: 143,774
- split: train
- same general schema as `ashaar-preprocessed-desc`

Published metadata for `ashaar-full-desc`:

- rows: 143,774
- split: train
- fields include the same poem and poet metadata plus:
  - `poem_description`
  - `second_pass_candidate`

Interpretation: the postworker/full-desc stage reduced the corpus from 145,167 to 143,774 rows and consolidated description outputs. The presence of `second_pass_candidate` suggests quality-control or retry/second-pass bookkeeping. This stage predates the April 2026 Qwen `enhanced_description` regeneration described later; it should be treated as the older description pipeline.

Evidence status:

- row count drop from 145,167 to 143,774 is verified.
- `second_pass_candidate` in `ashaar-full-desc` is verified.
- reason for each removed row is not documented in the current repo.

### 3.4 Full Description Preprocessed

Dataset:

- `Shaer-AI/ashaar-full-desc-preprocessed`

Published metadata:

- rows: 118,304
- split: train
- same broad schema as `ashaar-full-desc`, but without `second_pass_candidate`

Interpretation: this appears to be a filtered/cleaned subset of `ashaar-full-desc`. It is not the final SFT source, but it is part of the older description-data history.

Evidence status:

- row count drop from 143,774 to 118,304 is verified.
- exact filtering rules for this older artifact are not present in the current repo.
- do not present this as the direct source of the final SFT dataset unless older build scripts are found.

### 3.5 Meter-Control Base-Form Dataset

Dataset:

- `Shaer-AI/ashaar-v1-base-form`

This is the first clearly documented meter-control dataset. Its card states it is a cleaned and structured Arabic poetry dataset derived from the original Ashaar corpus for meter-conditioned generation.

Its key transformation is the introduction/stabilization of:

- `base_meter`: canonical classical meter
- `form`: controlled form such as `تام`, `مجزوء`, `مخلع`, `أحذ`

Published structure:

- splits: train and validation
- train rows: 143,025
- validation rows: 882
- columns:
  - `id`
  - `poem verses`
  - `base_meter`
  - `form`
  - `poem theme`
  - `poem meter`
  - `poem url`

Supported base meters in V1:

- `الطويل`
- `الكامل`
- `البسيط`
- `الوافر`
- `الخفيف`
- `السريع`
- `الرجز`
- `الرمل`
- `المتقارب`
- `المجتث`
- `المنسرح`
- `المديد`
- `الهزج`
- `المتدارك`

Excluded rare meters:

- `المضارع`
- `المقتضب`

Supported forms:

- `تام`
- `مجزوء`
- `مخلع` only for `البسيط`
- `أحذ` only for `الكامل`

Other rare/inconsistent forms were removed, including examples such as `مشطور`, `منهوك`, `مربع`, and `تفعيلة`.

The validation split rule in the dataset card:

```text
take = min(50, ceil(5% of each (meter, form) group))
```

Interpretation: this is the key dataset where the raw Ashaar meter labels were converted into the controlled `base_meter` and `form` setup used by all later SFT/RL datasets.

Important curation choices recorded in the card:

- the dataset keeps only learnable meter/form combinations.
- two rare meters, `المضارع` and `المقتضب`, were excluded due to insufficient data.
- rare/inconsistent forms were removed to avoid noisy control signals.
- original raw meter labels are preserved separately as `poem meter`.
- canonical control labels are stored as `base_meter` and `form`.

Evidence status:

- supported meters/forms and validation split rule are directly stated in the HF card.
- exact code mapping every raw `poem meter` string to `base_meter` and `form` is not present in this repo snapshot.
- for the paper, state the categories and filtering principles, not a nonexistent exact regex/mapping table unless recovered separately.

### 3.6 Base-Form With Descriptions

Dataset:

- `Shaer-AI/ashaar-v1-base-form-with-descriptions`

Published metadata:

- rows: 143,022
- split: train
- dataset size: 381,742,406 bytes

Schema:

- `id`
- `poem verses`
- `base_meter`
- `form`
- `poem theme`
- `poem meter`
- `poem url`
- `description`

Interpretation: this joins the controlled meter/form dataset with an older description field. The row count, 143,022, is lower than the 143,774 full-desc row count, suggesting additional joining/filtering constraints before the description-enhanced base-form dataset was published.

Compared with `ashaar-v1-base-form`, this dataset:

- removes the explicit `validation` split and publishes a single `train` split.
- keeps the controlled meter fields:
  - `base_meter`
  - `form`
- keeps poem metadata:
  - `poem theme`
  - `poem meter`
  - `poem url`
- adds `description`.

Evidence status:

- the schema and 143,022-row count are verified.
- the join source for descriptions is not fully documented in local code; likely tied to the older `ashaar-full-desc`/description branch.

### 3.7 Locked-Prompt SFT Dataset With Old Descriptions

Dataset:

- `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`

The dataset card directly states it is derived from:

- `Shaer-AI/ashaar-v1-base-form-with-descriptions`

It was prepared for SFT with a locked prompt from `prompt_testing.ipynb`.

Filtering and construction:

- source rows: 143,022
- valid-row filter: keep rows where `poem verses` is a non-empty list of non-empty strings
- optional line cap: `sft_num_lines <= 68`
- after row processing: 129,688
- token-length filter: `sft_total_tokens <= 2048`
- removed by token filter: 78
- final rows kept: 129,610

Added columns:

- `sft_prompt`
- `sft_completion`
- `sft_full_text`
- `sft_num_lines`
- `sft_total_tokens`

This dataset still used the old `description` field as conditioning text. The final April 2026 SFT baseline did not train on this dataset directly; it used this dataset only as the source for regenerated `enhanced_description`.

Compared with `ashaar-v1-base-form-with-descriptions`, this dataset:

- removes invalid poem rows.
- applies an optional line cap of 68 shatr strings.
- builds the initial locked SFT prompt.
- builds completion text from `poem verses` with real newline separators.
- computes tokenizer length with `Navid-AI/Yehia-7B-preview`.
- keeps only examples with `sft_total_tokens <= 2048`.
- reduces rows from 143,022 to 129,610.

The build script in this repo, `build_push_sft_dataset.py`, implements this kind of conversion:

- source: `Shaer-AI/ashaar-v1-base-form-with-descriptions`
- required source fields:
  - `base_meter`
  - `form`
  - `description`
  - `poem verses`
- added fields:
  - `sft_prompt`
  - `sft_completion`
  - `sft_full_text`
  - `sft_num_lines`
  - `sft_total_tokens`

The locked prompt at this stage included a `meter_label` rule:

```text
meter_label = base_meter if form == "تام"
else meter_label = "{form} {base_meter}"
```

This older prompt still used the old `description` field.

### 3.8 Enhanced-Description Final SFT Dataset

Dataset:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`

Direct source:

- `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`

This is the Qwen-enhanced final unsplit dataset. It:

- keeps the old `description`
- adds `enhanced_description`
- rebuilds SFT prompt fields from `enhanced_description`
- filters to valid even-shatr rows
- filters to `requested_bayts <= 20`
- drops meters with support `< 500`
- drops `المتدارك`
- applies tatweel cleanup
- publishes 116,032 rows

Compared with `ashaar-with-descriptions-baseform-final-trimmed`, this dataset:

- changes the conditioning text from old `description` to generated `enhanced_description`.
- changes the line-count logic from broad `sft_num_lines <= 68` to strict SFT final filtering on `requested_bayts <= 20`.
- requires even shatr count, because each bayt is two shatrs.
- removes malformed odd-shatr rows.
- removes meters whose post-filter support is below 500.
- drops `المتدارك`.
- preserves the old `description` only as a comparison/reference column.
- rebuilds all SFT fields after regeneration and again after tatweel cleanup.

The row-count transition is:

| Step | Rows | Drop From Previous |
|---|---:|---:|
| old trimmed source | 129,610 | - |
| after valid even-shatr filter | 127,722 | 1,888 |
| after `requested_bayts <= 20` | 116,167 | 11,555 |
| after meter support `>= 500` | 116,032 | 135 |

The 135-row final drop corresponds to `المتدارك` after the length/even-shatr filtering stage.

### 3.9 Stratified Split Dataset Used By SFT

Dataset:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

Direct source:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`

This adds deterministic train/eval/test splits:

- train: 109,070
- eval: 3,481
- test: 3,481

This is the dataset ID in the final SFT run config.

Compared with the unsplit enhanced dataset, this split dataset:

- does not regenerate descriptions.
- does not rewrite prompt/completion text.
- adds split/grouping columns:
  - `length_bucket`
  - `sampler_group`
  - `split_group`
  - `split_group_level`
- publishes three splits:
  - train: 109,070
  - eval: 3,481
  - test: 3,481
- uses deterministic seed 42.
- stratifies by `base_meter || form || length_bucket`.

### 3.10 Dataset-by-Dataset Transition Ledger

This table is the compact paper-facing ledger of what changed at every visible stage.

| Stage | Dataset | Rows / Splits | Main Columns | What Changed | Evidence |
|---|---|---:|---|---|---|
| 0 | Original Ashaar corpus | not directly inspected | raw poem/source fields | Original Arabic poetry corpus referenced by `ashaar-v1-base-form` card. | HF card statement |
| 1 | `Shaer-AI/ashaar-preprocessed` | train 145,167 | source-style fields with spaces; nested `poem description` | Early preprocessed corpus with poem/poet metadata and nested description structures. | HF schema/count |
| 2 | `Shaer-AI/ashaar-preprocessed-desc` | train 145,167 | snake_case fields; string-like `poem_description` | Schema normalized; description flattened/cleaned without row-count change. | HF schema/count; transformation inferred |
| 3 | `Shaer-AI/ashaar-preprocessed-postworkers` | train 143,774 | same cleaned metadata schema | Worker outputs consolidated; 1,393 fewer rows than `ashaar-preprocessed-desc`. | HF schema/count; exact drop reason unavailable |
| 4 | `Shaer-AI/ashaar-full-desc` | train 143,774 | adds `second_pass_candidate` | Full old-description dataset, preserving quality-control/second-pass marker. | HF schema/count |
| 5 | `Shaer-AI/ashaar-full-desc-preprocessed` | train 118,304 | old-description metadata schema | Further filtered/cleaned old-description subset. | HF schema/count; exact rules unavailable |
| 6 | `Shaer-AI/ashaar-v1-base-form` | train 143,025; validation 882 | `id`, `poem verses`, `base_meter`, `form`, source metadata | Canonical meter-control dataset; introduces `base_meter`/`form`; removes rare meters/forms; stratified validation split. | HF card/schema/count |
| 7 | `Shaer-AI/ashaar-v1-base-form-with-descriptions` | train 143,022 | base-form columns + `description` | Adds old description field to controlled meter/form rows. | HF schema/count |
| 8 | `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed` | train 129,610 | adds old SFT fields | Builds locked prompt SFT rows from old `description`; valid-row filter; line cap 68; token cap 2048. | HF card + local `build_push_sft_dataset.py` |
| 9 | `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500` | train 116,032 | adds `source_index`, `requested_bayts`, `enhanced_description`; rebuilt SFT fields | Filters valid even-shatr rows, `<=20` bayts, meter support `>=500`; Qwen regenerates descriptions; prompt rebuilt; tatweel cleaned. | HF progress summaries + local scripts |
| 10 | `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits` | train 109,070; eval 3,481; test 3,481 | adds `length_bucket`, `sampler_group`, `split_group`, `split_group_level` | Deterministic 94/3/3 stratified splits; no row content change intended. | HF split summary + local split code |
| 11 | `Shaer-AI/Shaer-adapters` | model repo | LoRA adapter files, reports, manifests | SFT adapter trained on split dataset. | HF model repo + local run artifacts |

### 3.11 Row-Count Drop Ledger

The final paper can use this to explain what was removed and where.

| Transition | Before | After | Delta | Known Reason |
|---|---:|---:|---:|---|
| `ashaar-preprocessed` -> `ashaar-preprocessed-desc` | 145,167 | 145,167 | 0 | schema/description normalization; no row drop |
| `ashaar-preprocessed-desc` -> `ashaar-preprocessed-postworkers` | 145,167 | 143,774 | -1,393 | older worker/description consolidation; exact row reasons not in current repo |
| `ashaar-preprocessed-postworkers` -> `ashaar-full-desc` | 143,774 | 143,774 | 0 | old full-description packaging; adds `second_pass_candidate` |
| `ashaar-full-desc` -> `ashaar-full-desc-preprocessed` | 143,774 | 118,304 | -25,470 | older full-description filtering; exact rules not in current repo |
| `ashaar-v1-base-form` train -> `ashaar-v1-base-form-with-descriptions` | 143,025 | 143,022 | -3 | description join / publication difference; exact three-row reason not documented |
| `ashaar-v1-base-form-with-descriptions` -> `ashaar-with-descriptions-baseform-final-trimmed` | 143,022 | 129,610 | -13,412 | valid poem rows, line cap `<=68`, token cap `<=2048`; card reports 129,688 after row processing and 78 token drops |
| old trimmed source -> enhanced staged valid rows | 129,610 | 127,722 | -1,888 | odd-shatr rows removed |
| enhanced valid rows -> `<=20` bayts | 127,722 | 116,167 | -11,555 | poems longer than 20 bayts removed |
| enhanced `<=20` bayts -> final meter cutoff | 116,167 | 116,032 | -135 | `المتدارك` dropped because support `<500` |
| enhanced unsplit -> split dataset | 116,032 | 116,032 | 0 | split annotation only; train/eval/test partition |

Note: `ashaar-full-desc-preprocessed` is not documented as the direct parent of `ashaar-v1-base-form`. The row-count ledger includes it because it is part of the visible old-description lineage, not because it is a proven direct input to final SFT.

### 3.12 Exact Final Curation Algorithm

For the final SFT dataset, the exact script-backed algorithm is:

1. Load `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`.
2. For each row, normalize `poem verses`:
   - must be a list
   - each item is converted to string and stripped
   - empty item rejects the row
3. Reject rows with odd shatr count.
4. Compute `requested_bayts = len(poem verses) // 2`.
5. Reject rows where `requested_bayts > 20`.
6. Count `base_meter` after the above filters.
7. Drop meters with count `<500`; in the final run this drops only `المتدارك`.
8. Preserve:
   - all source columns
   - `source_index`
   - `requested_bayts`
9. Shard the 116,032 staged rows across 24 workers, balanced to within one row.
10. For every row, call Qwen/OpenRouter with the poem hemistichs only and ask for `{"new_description":"..."}`.
11. Validate generated description:
    - non-empty
    - no raw JSON leakage
    - no `new_description` key leakage
    - valid parsed candidate or acceptable cleaned raw text in retry path
12. Merge all successful worker and retry outputs by `source_index`.
13. Add `enhanced_description`.
14. Rebuild `sft_prompt` from `enhanced_description`.
15. Rebuild `sft_completion` from cleaned poem hemistichs joined by newline.
16. Rebuild `sft_full_text`.
17. Compute `sft_num_lines`.
18. Tokenize `sft_full_text` with `Navid-AI/Yehia-7B-preview`.
19. Drop rows with `sft_total_tokens > 2048`; final run drops zero rows here.
20. Strip tatweel:
    - all tatweel from `description`
    - all tatweel from `enhanced_description`
    - internal tatweel from poems
    - preserve edge tatweel at shatr boundaries
21. Rebuild prompt fields again after tatweel cleanup.
22. Push final unsplit dataset.
23. Build deterministic 94/3/3 splits with seed 42.
24. Add split/sampler columns.
25. Push final split dataset used by SFT.

### 3.13 What Is Fully Verified vs. Reconstructed

Fully verified by local scripts and HF artifacts:

- final enhanced dataset source: `ashaar-with-descriptions-baseform-final-trimmed`
- final enhanced dataset filtering counts
- Qwen/OpenRouter description-generation prompt and validation behavior
- worker count and final merge/publish counts
- tatweel cleanup behavior and counts
- split algorithm, seed, groups, and counts
- SFT training dataset ID
- SFT model/training configuration
- SFT checkpoints, metrics, and adapter repo paths

Verified by HF cards/metadata but without local transformation code:

- `ashaar-v1-base-form` motivation, supported meters/forms, excluded rare meters, validation split policy
- `ashaar-v1-base-form-with-descriptions` schema and row count
- `ashaar-with-descriptions-baseform-final-trimmed` source, SFT fields, line cap, token cap, and counts

Reconstructed from HF sequence/schema/counts:

- `ashaar-preprocessed` -> `ashaar-preprocessed-desc`
- worker description consolidation into `ashaar-preprocessed-postworkers`
- packaging into `ashaar-full-desc`
- filtering into `ashaar-full-desc-preprocessed`

For a paper, the safest wording is:

> Earlier Ashaar preprocessing and description artifacts show the evolution from raw poem/poet metadata to flattened description-bearing tables. The final SFT pipeline is directly grounded from `ashaar-v1-base-form-with-descriptions` onward, where meter/form control labels and old descriptions are present and where all subsequent filtering, prompt construction, description regeneration, splitting, and training are reproducible from the released scripts and artifacts.

## 4. Final Dataset Lineage Overview

The SFT dataset lineage is:

1. Raw/original Ashaar corpus, visible in this workspace's HF history through early `ashaar-*` preprocessing datasets.
2. Meter-control base-form dataset:
   `Shaer-AI/ashaar-v1-base-form`
3. Base-form dataset with old descriptions:
   `Shaer-AI/ashaar-v1-base-form-with-descriptions`
4. Earlier SFT-ready locked-prompt dataset with old descriptions:
   `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`
5. Filtered and staged SFT source:
   valid even-length poems, `requested_bayts <= 20`, and base meters with at least 500 examples.
6. Qwen/OpenRouter description regeneration:
   one new `enhanced_description` generated per staged poem.
7. Final SFT prompt rebuild:
   `sft_prompt`, `sft_completion`, `sft_full_text`, `sft_num_lines`, and `sft_total_tokens` rebuilt from `enhanced_description`.
8. Tatweel cleanup:
   internal tatweel removed from poems and all tatweel removed from description fields; edge tatweel at shatr boundaries preserved.
9. Final unsplit dataset:
   `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`
10. Deterministic train/eval/test split dataset:
   `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`
11. Main SFT adapter training:
   `Shaer-AI/Shaer-adapters`

The older SFT dataset `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed` was explicitly not used for the final baseline training. It was only the source that was filtered and enhanced.

## 5. Source Dataset Filtering

The final enhanced-description dataset was built by `description_generation/final_sft_dataset.py`.

Source dataset:

- `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`

Target final dataset:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`

The prepare step used:

- `max_bayts = 20`
- `min_meter_count = 500`
- `num_workers = 24` for the final full regeneration run
- `requested_bayts = len(poem verses) // 2`

Filtering steps:

1. Read all source rows.
2. Normalize `poem verses` as a list of stripped, non-empty strings.
3. Drop rows where `poem verses` is missing, empty, or contains empty strings.
4. Drop rows with an odd number of shatr strings, since bayt count requires pairs.
5. Compute `requested_bayts = len(poem verses) // 2`.
6. Keep only rows where `requested_bayts <= 20`.
7. Count `base_meter` after the validity and length filters.
8. Drop base meters whose post-filter support is `< 500`.
9. Preserve `source_index` so later worker outputs can be merged deterministically.

Published preprocessing counts from HF:

| Stage | Rows |
|---|---:|
| Source rows | 129,610 |
| After valid even-shatr filtering | 127,722 |
| After `requested_bayts <= 20` | 116,167 |
| Final staged rows after meter cutoff | 116,032 |

Invalid/dropped counts:

- odd-shatr rows: 1,888
- rows over 20 bayts: 11,555
- dropped base meter: `المتدارك`

The only meter dropped by the `< 500` cutoff was `المتدارك`.

## 6. Qwen Enhanced-Description Generation

The old `description` field was preserved, but it was not trusted as the final conditioning text. The repo notes state that old descriptions were often weak or noisy enough to hurt prompt quality, so a new `enhanced_description` field was generated for every staged poem.

Generation script:

- `description_generation/regenerate_descriptions.py`

Final run:

- run name: `final_sft_regen_v1_20260407_w24`
- workers: 24
- worker shard sizes: 4,834 or 4,835 rows
- total assigned rows: 116,032
- description prompt version: `pure_description_v8_poem_only_antidrift`

The OpenRouter client in `regenerate_descriptions.py` defaulted to:

- API base: `https://openrouter.ai/api/v1`
- model env default: `qwen/qwen3.5-35b-a3b`
- temperature: `0.0`
- `response_format = {"type": "json_object"}`
- `max_tokens = 300`
- provider fallbacks disabled with `allow_fallbacks = False`
- reasoning disabled with `reasoning = {"effort": "none"}`
- row retry limit default: 4 attempts

Each request sent the poem hemistichs and required a JSON object:

```json
{"new_description":"..."}
```

Validation accepted only non-empty natural text. It rejected:

- empty `new_description`
- raw JSON leakage
- output beginning with `{` or `[`
- output that still contained the `new_description` key text
- malformed or missing JSON key when the JSON path was used

Worker reliability behavior:

- each worker had a disjoint shard from `workers_manifest.json`
- each worker wrote only to its own `results.jsonl`
- progress was uploaded to the dataset repo under `progress/<run_name>/workers/worker_XX/`
- resume mode loaded already completed local and remote rows and skipped them
- failed rows could be retried through extra retry manifests/results during merge

Final worker summaries for the 24-worker run showed:

- assigned rows: 116,032
- completed in the original worker outputs: 115,992
- errors in the original worker outputs: 40
- merge used extra retry artifacts:
  - `final_sft_regen_v1_20260407_w24_retry40_w4`
  - `final_sft_regen_v1_20260407_w24_retry15_plaintext`

The published merge summary confirms all staged rows were eventually merged:

- merged rows: 116,032
- rows removed by final token cap: 0
- final published rows: 116,032
- token cap: 2,048 tokens

## 7. Final SFT Prompt Rebuild

The final dataset preserves original source columns and adds/rebuilds SFT fields.

Important columns:

- `description`: original old description, preserved for comparison.
- `enhanced_description`: regenerated Qwen description.
- `sft_prompt`: prompt rebuilt from `enhanced_description`.
- `sft_completion`: reference poem lines joined with real newline characters.
- `sft_full_text`: prompt, completion, and closing `</s>`.
- `sft_num_lines`: number of shatr strings in `poem verses`.
- `sft_total_tokens`: tokenizer length of `sft_full_text`.
- `requested_bayts`: `len(poem verses) // 2`.
- `source_index`: original source-row index used for deterministic merging.

The final HF dataset card reports prompt version:

- `final_sft_meter_emphasis_v2_num_lines`

The final SFT system prompt:

```text
أنت شاعر عربي تكتب الشعر العمودي الكلاسيكي.
التزم بالبحر المحدد في كل شطر، واستلهم من الموضوع دون نقله حرفياً.
أخرج الأبيات فقط دون مقدمة أو تعليق.
التزم التزاماً صارماً بالبحر المطلوب، ولا تخرج عنه.
```

The final SFT user template:

```text
البحر الأساسي: {base_meter}
الصيغة: {form}
عدد الأشطر المطلوب: {num_lines}
الموضوع: {description}

اكتب {num_lines} أشطارًا ملتزمة بصيغة {form} من بحر {base_meter} دون أي شرح إضافي.
```

In construction, `{description}` is filled with the regenerated `enhanced_description`, not the original `description`.

The final prompt wrapper is:

```text
<s> [INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]
```

The completion is:

```text
{poem_shatr_1}
{poem_shatr_2}
...
```

The full training text is:

```text
{sft_prompt} {sft_completion} </s>
```

## 8. Tatweel Cleanup

After the final rows were built, `description_generation/tatweel_cleanup_publish.py` normalized tatweel.

Rules:

- preserve tatweel only when it appears at the start or end of a shatr boundary in `poem verses`
- remove internal tatweel from `poem verses`
- remove all tatweel from `description`
- remove all tatweel from `enhanced_description`
- rebuild prompt fields after cleanup

Published cleanup summary:

| Metric | Value |
|---|---:|
| rows processed | 116,032 |
| tatweel characters removed | 5,461,645 |
| rows with enhanced-description tatweel before | 2,350 |
| rows with old-description tatweel before | 3,847 |
| rows with poem tatweel before | 66,361 |
| rows with internal poem tatweel before | 65,965 |
| rows changed | 65,965 |
| rows with poem tatweel after | 514 |
| rows with edge poem tatweel after | 514 |
| validation failures | 0 |

The validation report had `failure_count = 0`.

## 9. Deterministic Stratified Splits

The main SFT run did not train on the unsplit final dataset directly. It trained on:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

Split script:

- `sft/publish_sft_stratified_splits.py`
- split logic: `sft/split_utils.py`

Split fractions:

- train: 94%
- eval: 3%
- test: 3%

Seed:

- 42

Final split counts:

| Split | Rows |
|---|---:|
| train | 109,070 |
| eval | 3,481 |
| test | 3,481 |
| total | 116,032 |

Primary stratification key:

```text
base_meter || form || length_bucket
```

Length buckets:

- `1-3`
- `4-6`
- `7-10`
- `11-20`

The split code computes:

- `sampler_group = base_meter||form||length_bucket`
- `split_group = sampler_group` if that fine group has at least `MIN_SAFE_STRATIFY_GROUP_SIZE = 8`
- otherwise fallback to `base_meter||form`
- otherwise fallback to `base_meter`
- otherwise fallback to `__global__`

For the final published split, all rows stayed at the finest level:

| Split group level | Rows |
|---|---:|
| `base_meter_form_length_bucket` | 116,032 |
| `base_meter_form` | 0 |
| `base_meter` | 0 |
| `global` | 0 |

There were 84 fine stratification groups.

## 10. Split Distributions

Base-meter counts:

| Meter | Train | Eval | Test |
|---|---:|---:|---:|
| `البسيط` | 16,563 | 529 | 529 |
| `الخفيف` | 9,263 | 295 | 295 |
| `الرجز` | 4,311 | 137 | 137 |
| `الرمل` | 4,316 | 138 | 138 |
| `السريع` | 6,758 | 216 | 216 |
| `الطويل` | 26,346 | 841 | 841 |
| `الكامل` | 20,375 | 649 | 649 |
| `المتقارب` | 4,877 | 156 | 156 |
| `المجتث` | 1,938 | 61 | 61 |
| `المديد` | 504 | 17 | 17 |
| `المنسرح` | 2,579 | 84 | 84 |
| `الهزج` | 573 | 18 | 18 |
| `الوافر` | 10,667 | 340 | 340 |

Form counts:

| Form | Train | Eval | Test |
|---|---:|---:|---:|
| `أحذ` | 351 | 11 | 11 |
| `تام` | 104,224 | 3,327 | 3,327 |
| `مجزوء` | 3,882 | 124 | 124 |
| `مخلع` | 613 | 19 | 19 |

Length-bucket counts:

| Length bucket | Train | Eval | Test |
|---|---:|---:|---:|
| `1-3` | 52,202 | 1,668 | 1,668 |
| `4-6` | 25,266 | 805 | 805 |
| `7-10` | 14,824 | 473 | 473 |
| `11-20` | 16,778 | 535 | 535 |

## 11. Train-Only Weighted Sampling

The SFT run used weighted sampling only on the train split.

Config flag:

- `use_weighted_sampler = true`

Eval and test remained natural, unweighted splits.

Sampler group:

```text
base_meter || form || length_bucket
```

Weight formula from `sft/split_utils.py`:

```text
raw_weight(g) = (1 / count(g)) ** 0.6
normalized_weight(g) = raw_weight(g) / mean(raw_weight over groups)
```

The trainer used PyTorch `WeightedRandomSampler` with:

- `weights = sample_weights`
- `num_samples = len(train_dataset)`
- `replacement = True`

The purpose was to soften dominance from very large groups without fully flattening the dataset. Rare combinations such as `البسيط||مجزوء||11-20` received higher sampling weight; very common combinations such as `الطويل||تام||1-3` received lower sampling weight.

Examples from the split summary:

Top-weighted groups:

| Group | Train count | Weight |
|---|---:|---:|
| `البسيط||مجزوء||11-20` | 17 | 4.5319 |
| `البسيط||مجزوء||7-10` | 22 | 3.8824 |
| `الوافر||مجزوء||7-10` | 37 | 2.8420 |
| `الخفيف||مجزوء||7-10` | 39 | 2.7537 |
| `الوافر||مجزوء||11-20` | 46 | 2.4940 |

Lowest-weighted groups:

| Group | Train count | Weight |
|---|---:|---:|
| `الطويل||تام||1-3` | 12,908 | 0.0847 |
| `الكامل||تام||1-3` | 8,336 | 0.1101 |
| `البسيط||تام||1-3` | 7,529 | 0.1171 |
| `الطويل||تام||4-6` | 5,805 | 0.1369 |
| `الوافر||تام||1-3` | 5,187 | 0.1464 |

## 12. Main SFT Training Run

Canonical run:

- run name: `train_20260407_231929`
- local run dir: `sft/outputs/train/train_20260407_231929`
- HF repo: `Shaer-AI/Shaer-adapters`
- namespace: `fresh_sft`
- resume mode: `fresh`
- init from adapter: `false`
- seed: 42
- pushed to Hub: true

Model:

- base model: `Navid-AI/Yehia-7B-preview`
- `trust_remote_code = true`
- `torch_dtype = bfloat16`
- `use_cache = false`
- gradient checkpointing: true

Quantization:

- QLoRA with `load_in_4bit = true`
- quant type: `nf4`
- double quantization: true
- compute dtype: `bfloat16`
- `prepare_model_for_kbit_training = true`

LoRA configuration:

- PEFT type: `LORA`
- task type: `CAUSAL_LM`
- target modules: all linear layers, materialized in adapter config as:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- rank `r = 64`
- `lora_alpha = 128`
- `lora_dropout = 0.05`
- `bias = none`
- `use_rslora = true`
- base model in adapter config: `Navid-AI/Yehia-7B-preview`

Optimizer and scheduler:

- optimizer: `paged_adamw_8bit`
- learning rate: `8e-5`
- scheduler: cosine
- warmup ratio: `0.03`
- weight decay: `0.0`
- max grad norm: `1.0`

Training shape:

- per-device train batch size: 1
- per-device eval batch size: 1
- gradient accumulation steps: 16
- effective update batch size: 16 sequences before distributed scaling
- number of epochs: 2
- max steps: `-1` in config; resolved trainer state max steps: 4,236
- logging steps: 10
- eval steps: 200
- save steps: 200
- save total limit: 3
- remote keep-last checkpoints: 3
- `load_best_model_at_end = true`
- best-model metric: `eval_loss`
- `greater_is_better = false`

Data preprocessing:

- max sequence length: 1,024
- packing enabled for train
- completion-only loss enabled
- truncation mode: `keep_end`
- train/eval/test row caps: 0, meaning no cap
- no additional run-time meter drop
- no additional min/max bayt filter during this run

Loss masking:

- The prompt tokens are masked with `-100`.
- Only completion tokens contribute to the causal language-modeling loss.
- `first_batch_debug.json` was written to validate that both masked prompt tokens and supervised completion tokens existed.

Packing:

- training examples were tokenized from `sft_prompt` and `sft_full_text`
- labels were built with prompt tokens masked when `completion_only_loss = true`
- packed train sequences were assembled up to `max_seq_length = 1024`
- eval/test preprocessing was not packed in the same way as train

## 13. Monitoring and Probe Evaluation

The SFT run used three evaluation concepts:

- `eval`: the 3% validation split, used during training for eval loss and best checkpoint selection.
- `test`: the 3% held-out split, evaluated only at the end.
- `probe`: a small generation-based monitor built from the test split.

Probe settings:

- `probe_per_meter = 2`
- final probe bank rows: 26
- generation max new tokens: 192
- generation batch size: 4
- `do_sample = false`
- configured temperature: 0.8, but deterministic generation means sampling was disabled

Probe metrics:

- `probe_meter_mean`: generation-side meter score from the repo meter scorer.
- `probe_count_adherence_mean`: whether generated output obeyed requested line count.
- per-meter probe means were also logged.

The probe scorer is evaluation-only and non-differentiable. It was not used as the SFT training loss.

## 14. Training Outcomes

The run completed cleanly:

- global optimizer steps: 4,236
- epochs: 1.999881971, effectively 2.0
- train runtime: 45,102.4652 seconds, about 12.53 hours
- reported train loss: 1.620038229863074
- total FLOPs in trainer state: `2.8588846620388884e18`

Eval loss trajectory:

| Step | Epoch | Eval loss |
|---:|---:|---:|
| 0 | before training | 3.0803117752 |
| 200 | 0.0944 | 2.3407740593 |
| 400 | 0.1888 | 2.2992038727 |
| 600 | 0.2833 | 2.2702960968 |
| 800 | 0.3777 | 2.2622749805 |
| 1,000 | 0.4721 | 2.2709798813 |
| 2,800 | 1.3219 | 2.2295434475 |
| 3,000 | 1.4163 | 2.2074480057 |
| 3,200 | 1.5108 | 2.2103283405 |
| 3,400 | 1.6052 | 2.2326102257 |
| 3,600 | 1.6996 | 2.2287876606 |
| 3,800 | 1.7940 | 2.2308671474 |
| 4,000 | 1.8885 | 2.2301211357 |
| 4,200 | 1.9829 | 2.2317516804 |
| 4,236 | final explicit eval | 2.2074480057 |

Best eval checkpoint:

- local: `sft/outputs/train/train_20260407_231929/checkpoint-3000`
- HF path in best manifest: `checkpoints/fresh_sft/train/train_20260407_231929/checkpoint-3000`
- best eval loss: 2.2074480056762695

Final held-out test:

- final test loss: 2.1932055950164795
- test runtime: 357.0562 seconds
- test samples/sec: 9.749

Probe outcomes:

| Metric | Step | Value |
|---|---:|---:|
| initial probe meter mean | 0 | 0.0226480712 |
| initial probe count adherence mean | 0 | 0.6923076923 |
| best probe meter mean | 2,800 | 0.6042156156 |
| best probe count adherence mean | 1,400 | 1.0 |
| final probe meter mean | 4,236 | 0.5087050690 |
| final probe count adherence mean | 4,236 | 1.0 |

The best eval checkpoint and best generation-side meter checkpoint did not coincide:

- best eval loss: step 3,000
- best probe meter: step 2,800

The run preserved a local best-meter alias:

- `sft/outputs/train/train_20260407_231929/best_probe_meter_checkpoint`

The key metrics file records the best-meter remote checkpoint as:

- `checkpoints/fresh_sft/train/train_20260407_231929/checkpoint-2800`

However, due to the remote keep-last policy, the current HF model repo file listing retains only checkpoints 4,000, 4,200, and 4,236 under the main `fresh_sft` checkpoint path, while the best adapter export is separately preserved.

## 15. Final Per-Meter Probe Snapshot

Final probe meter mean by meter at step 4,236:

| Meter | Final probe meter mean |
|---|---:|
| `الطويل` | 0.9907925137 |
| `الكامل` | 0.9638710435 |
| `المنسرح` | 0.8246771979 |
| `الوافر` | 0.7789316796 |
| `الخفيف` | 0.5783225187 |
| `البسيط` | 0.5194733848 |
| `السريع` | 0.5107348435 |
| `المجتث` | 0.5091973305 |
| `المتقارب` | 0.5016234686 |
| `الرمل` | 0.3504121841 |
| `الرجز` | 0.0556526308 |
| `الهزج` | 0.0234434797 |
| `المديد` | 0.0060336212 |

Final probe count adherence was 1.0 for every meter in the probe bank.

## 16. Hugging Face Adapter Repository Contents

Main model repo:

- `Shaer-AI/Shaer-adapters`

Important exported adapter paths:

- best adapter: `adapters/fresh_sft/train/best`
- latest adapter: `adapters/fresh_sft/train/latest`

Important manifest paths:

- `manifests/fresh_sft/train/best.json`
- `manifests/fresh_sft/train/latest.json`
- `manifests/fresh_sft/train/history/train_20260407_231929.json`

Main report bundle:

- `reports/fresh_sft/train_20260407_231929/summary.md`
- `reports/fresh_sft/train_20260407_231929/key_metrics.json`
- `reports/fresh_sft/train_20260407_231929/run_summary.json`
- `reports/fresh_sft/train_20260407_231929/config_snapshot.json`
- `reports/fresh_sft/train_20260407_231929/effective_runtime_config.json`
- `reports/fresh_sft/train_20260407_231929/dataset_summary.json`
- `reports/fresh_sft/train_20260407_231929/split_summary.json`
- `reports/fresh_sft/train_20260407_231929/metrics.csv`
- `reports/fresh_sft/train_20260407_231929/metrics.jsonl`
- `reports/fresh_sft/train_20260407_231929/probe_metrics.csv`
- `reports/fresh_sft/train_20260407_231929/probe_metrics.jsonl`
- `reports/fresh_sft/train_20260407_231929/final_probe_per_meter.csv`
- `reports/fresh_sft/train_20260407_231929/probe_meter_history_long.csv`
- `reports/fresh_sft/train_20260407_231929/best_probe_meter_summary.json`
- `reports/fresh_sft/train_20260407_231929/plots/loss_panels.png`
- `reports/fresh_sft/train_20260407_231929/plots/probe_panels.png`
- `reports/fresh_sft/train_20260407_231929/plots/probe_meter_heatmap.png`
- `reports/fresh_sft/train_20260407_231929/plots/probe_meter_small_multiples.png`

Current retained main checkpoints in HF listing:

- `checkpoints/fresh_sft/train/train_20260407_231929/checkpoint-4000`
- `checkpoints/fresh_sft/train/train_20260407_231929/checkpoint-4200`
- `checkpoints/fresh_sft/train/train_20260407_231929/checkpoint-4236`

Each retained checkpoint includes adapter weights, tokenizer files, optimizer/scheduler state, RNG state, trainer state, and training args.

The best manifest says:

```json
{
  "global_step": 3000,
  "best_eval_loss": 2.2074480056762695,
  "adapter_best_path_in_repo": "adapters/fresh_sft/train/best",
  "checkpoint_path_in_repo": "checkpoints/fresh_sft/train/train_20260407_231929/checkpoint-3000"
}
```

The latest manifest says:

```json
{
  "global_step": 4236,
  "checkpoint_path_in_repo": "checkpoints/fresh_sft/train/train_20260407_231929/checkpoint-4236",
  "adapter_latest_path_in_repo": "adapters/fresh_sft/train/latest"
}
```

## 17. Paper-Facing Interpretation

The final SFT baseline improved held-out CE substantially:

- pre-training eval loss: 3.0803
- best/final eval loss: 2.2074
- final test loss: 2.1932

It also learned strong structural behavior under generation probes:

- probe meter mean rose from 0.0226 to a best of 0.6042
- final probe meter mean remained 0.5087
- count adherence reached and ended at 1.0

The final model is best described as:

> a fresh QLoRA SFT adapter over `Navid-AI/Yehia-7B-preview`, trained for two epochs on 109,070 weighted-sampled training examples from a deterministic 94/3/3 stratified split of a 116,032-row enhanced-description Ashaar SFT dataset.

The key dataset contribution is not merely filtering Ashaar. It is the replacement of noisy old conditioning descriptions with Qwen-generated `enhanced_description` fields, followed by a rebuild of the prompt/completion text and deterministic stratification by meter, form, and length bucket.

## 18. Reproducibility Anchors

Local files:

- `description_generation/final_sft_dataset.py`
- `description_generation/regenerate_descriptions.py`
- `description_generation/tatweel_cleanup_publish.py`
- `description_generation/prompt_contracts.py`
- `sft/publish_sft_stratified_splits.py`
- `sft/split_utils.py`
- `sft/train_sft.py`
- `sft/sft_config.yaml`
- `sft/outputs/train/train_20260407_231929/run_summary.json`
- `sft/outputs/train/train_20260407_231929/dataset_summary.json`
- `sft/outputs/train/train_20260407_231929/split_summary.json`
- `artifacts/sft/train_20260407_231929_summary.md`
- `artifacts/sft/train_20260407_231929_key_metrics.json`

Hugging Face repos:

- final unsplit dataset: `https://huggingface.co/datasets/Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`
- final split dataset used for training: `https://huggingface.co/datasets/Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`
- final SFT adapters: `https://huggingface.co/Shaer-AI/Shaer-adapters`
