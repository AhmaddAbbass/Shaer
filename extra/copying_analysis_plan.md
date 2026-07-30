# Poem Copying Analysis Plan

This plan addresses reviewer concern:

> No memorization or copying analysis is provided for generated poems. The paper checks copying in generated descriptions, but it does not appear to check whether Shaer's generated poems copy training poems or held-out source poems. This matters because the model is fine-tuned on a large poetry corpus and evaluated on prompts derived from held-out poems. Possible train/test leakage, and no memorization/copying analysis of generated poems.

## Available Data

Training dataset used for SFT:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

Local cached files:

- `train-00000-of-00002.parquet`: `54,535` rows
- `train-00001-of-00002.parquet`: `54,535` rows
- train total: `109,070` rows
- `eval-00000-of-00001.parquet`: `3,481` rows
- `test-00000-of-00001.parquet`: `3,481` rows

Relevant fields:

- Training/reference poem text: `sft_completion`
- Generated poem text in evaluation datasets: `generated_text`
- Metadata: `id`, `source_index`, `base_meter`, `form`, `requested_bayts`, `enhanced_description`

Generated Shaer evaluation dataset:

- `Shaer-AI/shaer-sft-test`
- rows: `3,481`
- generated poem field: `generated_text`
- held-out reference field: `reference_completion`

## Core Comparisons

Run copying checks for each Shaer generated poem against:

- `train` split poems: detects training memorization or copying.
- held-out `test` source/reference poems: detects prompt-derived source copying.
- held-out `eval` split poems: optional leakage sanity check against the other held-out split.

The main paper-facing comparison should be:

- `Shaer generated_text` vs `train.sft_completion`
- `Shaer generated_text` vs `test.sft_completion` / `reference_completion`

## Text Normalization

Use two normalization levels:

- `surface`: light cleanup only, preserves Arabic letters and most spelling.
- `normalized`: removes diacritics, tatweel, punctuation, extra whitespace, and normalizes common Arabic variants.

Recommended Arabic normalization:

- remove tashkeel/harakat
- remove tatweel
- normalize `أ`, `إ`, `آ` to `ا`
- normalize `ى` to `ي`
- normalize `ة` to `ه` only in the aggressive setting, not the main setting
- remove punctuation
- collapse whitespace

Report both surface and normalized results if space allows. Use normalized results as the main robustness check.

## Metrics

### 1. Exact Duplication

For each generated poem:

- exact full-poem match against any training poem
- normalized full-poem match against any training poem
- exact/normalized full-poem match against held-out reference poem

Report:

- number of exact full-poem copies
- percentage of generated poems copied
- matched source row IDs

### 2. Longest Common Substring / Span

Compute the longest contiguous copied span between each generated poem and its nearest source poem.

Recommended units:

- word tokens
- hemistich lines

Report:

- max copied word span
- average max copied word span
- percentage of generations with a copied span of at least `10`, `20`, and `30` words
- examples of the top matches

### 3. N-Gram Overlap

Compute word n-gram overlap against the nearest source poem.

Recommended n-grams:

- `5`-grams
- `8`-grams
- `13`-grams

For each generated poem:

- tokenize generated poem
- tokenize candidate source poems
- compute maximum source overlap for each n
- record nearest source row

Report:

- percent of generated poems with any shared `13`-gram with training poems
- percent with at least `5`, `10`, `20` shared `8`-grams
- maximum n-gram overlap rates
- top copied-looking examples

### 4. Near-Duplicate Similarity

Use MinHash or sparse character/word n-gram retrieval to avoid comparing every generated poem against every training poem exhaustively.

Recommended practical approach:

- build an inverted index of normalized word `5`-grams from training poems
- for each generation, retrieve candidate training poems sharing at least one rare `5`-gram
- compute exact metrics only for candidates
- separately run direct comparison against the paired held-out source/reference poem

Report:

- highest Jaccard similarity over word `5`-grams
- highest Jaccard similarity over character `5`-grams
- top nearest training poem for each generated poem

## Thresholds For Flagging

Flag a generation as suspicious if any condition holds:

- normalized full-poem exact match
- copied contiguous span of at least `30` words
- at least one shared normalized word `13`-gram
- word `5`-gram Jaccard similarity >= `0.50`
- character `5`-gram Jaccard similarity >= `0.70`

These thresholds are conservative enough to catch likely copying while allowing formulaic classical poetic language.

## Outputs

Create row-level output:

- `generation_id`
- `base_meter`
- `form`
- `generated_text`
- `nearest_train_id`
- `nearest_train_source_index`
- `nearest_train_score`
- `max_word_5gram_jaccard_train`
- `max_char_5gram_jaccard_train`
- `max_copied_word_span_train`
- `shared_13gram_count_train`
- `paired_reference_score`
- `max_copied_word_span_reference`
- `shared_13gram_count_reference`
- `copy_flag`
- `copy_flag_reasons`

Create aggregate summary:

- total generations checked
- full-poem duplicate count
- near-duplicate count
- suspicious-copy count
- top-10 nearest training matches
- top-10 nearest held-out reference matches
- metric distributions

## Paper-Ready Reporting

Recommended table:

| Comparison Source | Exact Copies | Any Shared 13-Gram | Span >= 30 Words | Mean Max 5-Gram Jaccard | Flagged Rows |
|---|---:|---:|---:|---:|---:|
| Train split | TBD | TBD | TBD | TBD | TBD |
| Held-out source/reference | TBD | TBD | TBD | TBD | TBD |

Recommended text:

> To assess memorization, we compared all 3,481 Shaer generations against the 109,070 training poems and the paired held-out source poems using normalized exact matching, long-span overlap, and word/character n-gram similarity. We found [TBD], indicating [TBD].

## Implementation Notes

Use cached parquet files where possible to avoid network dependency.

Potential script:

- `evaluation/poem_copying_analysis.py`

Recommended CLI:

```bash
python evaluation/poem_copying_analysis.py \
  --train-parquet-root <cached_split_dataset_data_dir> \
  --shaer-eval-dataset Shaer-AI/shaer-sft-test \
  --output-dir evaluation/outputs/poem_copying_analysis
```

Use `HF_TOKEN` only if the Shaer evaluation dataset is not already cached locally.
