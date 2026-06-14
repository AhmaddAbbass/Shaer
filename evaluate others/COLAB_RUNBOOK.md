# Colab Parallel Runbook

This workflow is for running the continuation models in parallel on separate Colab accounts and publishing each model's results as its own Hugging Face dataset.

## Shared Input

Use the same input file for all runs:

- `evaluation/outputs/full_baseline_manifest_unique_prompts.jsonl`

This file contains exactly one row per original Shaer test prompt:

- `3,481` rows
- `3,481` unique `source_row_index`

## Recommended Split

Run one model per Colab session:

- `ashaar_model`
- `fanar_2_diwan_prefix` (experimental)
- `gpt2_small_arabic_poetry`
- `gpt2_medium_arabic_poetry`

## Colab Setup

Clone the repo and install dependencies:

```bash
git clone <your_repo_url>
cd Shaer
pip install -U pip
pip install transformers datasets huggingface_hub python-dotenv pandas
```

If the scoring step is needed in the same notebook, install the project dependencies required by `evaluation/metrics.py` and `grpo/rewards`.

If your Colab environment only runs generation and does not include the scoring dependencies, you can still package the raw outputs first with `--skip-score` and score them later in a fuller environment.

Authenticate with Hugging Face:

```python
from huggingface_hub import login
login()
```

## Generate One Model

Run exactly one model per session:

```bash
python evaluation/generate_continuation_baselines.py \
  --subset-jsonl evaluation/outputs/full_baseline_manifest_unique_prompts.jsonl \
  --models gpt2_small_arabic_poetry \
  --samples-per-row 1 \
  --run-dir evaluation/outputs/gpt2_small_unique_prompts
```

Replace `gpt2_small_arabic_poetry` with:

- `ashaar_model`
- `fanar_2_diwan_prefix`
- `gpt2_medium_arabic_poetry`

For the Fanar experimental continuation run, add `--trust-remote-code`:

```bash
python evaluation/generate_continuation_baselines.py \
  --subset-jsonl evaluation/outputs/full_baseline_manifest_unique_prompts.jsonl \
  --models fanar_2_diwan_prefix \
  --samples-per-row 1 \
  --run-dir evaluation/outputs/fanar_prefix_unique_prompts \
  --trust-remote-code
```

## Package the Results

After generation completes, build the portable bundle:

```bash
python evaluation/export_model_results_bundle.py \
  --input-jsonl evaluation/outputs/gpt2_small_unique_prompts/continuation_generations.jsonl \
  --output-dir evaluation/outputs/gpt2_small_unique_prompts_bundle \
  --expected-source-rows 3481 \
  --samples-per-row 1 \
  --dataset-repo-id Shaer-AI/shaer-eval-gpt2-small-arabic-poetry
```

If scoring dependencies are missing in Colab, use:

```bash
python evaluation/export_model_results_bundle.py \
  --input-jsonl evaluation/outputs/gpt2_small_unique_prompts/continuation_generations.jsonl \
  --output-dir evaluation/outputs/gpt2_small_unique_prompts_bundle \
  --expected-source-rows 3481 \
  --samples-per-row 1 \
  --dataset-repo-id Shaer-AI/shaer-eval-gpt2-small-arabic-poetry \
  --skip-score
```

This creates:

- `generations.jsonl`
- `generations.csv`
- `generations_scored.jsonl`
- `generations_scored.csv`
- `validation.json`
- `aggregate.json`
- `bundle_metadata.json`
- `README.md`

## Upload to Hugging Face

Upload the bundle folder to a dataset repo:

```bash
python evaluation/upload_dataset_bundle.py \
  --folder evaluation/outputs/gpt2_small_unique_prompts_bundle \
  --repo-id Shaer-AI/shaer-eval-gpt2-small-arabic-poetry
```

Suggested repo names:

- `Shaer-AI/shaer-eval-ashaar-model`
- `Shaer-AI/shaer-eval-fanar-diwan-prefix`
- `Shaer-AI/shaer-eval-gpt2-small-arabic-poetry`
- `Shaer-AI/shaer-eval-gpt2-medium-arabic-poetry`

## Combine Later

After all three datasets are downloaded locally again, combine the scored JSONL files:

```bash
python evaluation/combine_model_results.py \
  --inputs \
evaluation/outputs/ashaar_bundle/generations_scored.jsonl,evaluation/outputs/gpt2_small_bundle/generations_scored.jsonl,evaluation/outputs/gpt2_medium_bundle/generations_scored.jsonl \
  --output-jsonl evaluation/outputs/combined_continuation_models.jsonl \
  --output-csv evaluation/outputs/combined_continuation_models.csv \
  --wide-csv evaluation/outputs/combined_continuation_models_wide.csv
```

Use the wide CSV for inspection and the long JSONL/CSV for analysis scripts.
