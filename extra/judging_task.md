# Additional Judge Run Task

Goal: run the locked strict `v2` judge prompts on all released row-level evaluation datasets with two new OpenRouter judges, then append prefixed metric columns without changing the existing unprefixed scores.

Use the new standalone runner only:

- `extra/run_additional_judges.py`
- `extra/run_additional_judges.ps1`, optional Windows launcher
- `extra/judge_prompts_v2_strict.yaml`

Do not use the old evaluation runner scripts for this task.

## Judge Models

- `qwen/qwen3.7-max`, prefix `qwen3_7_max`
- `openai/gpt-5.6-terra`, prefix `gpt5_6_terra`

## Datasets

- `shaer`: `Shaer-AI/shaer-sft-test`, `3481` rows
- `ashaar`: `Shaer-AI/shaer-eval-ashaar-native-controls`, `3481` rows
- `yehia`: `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`, `3481` rows
- `fanar`: `Shaer-AI/fanar-eval-native-prompt`, `3481` rows

## Metrics

For `shaer` and `yehia`:

- `description_adherence`
- `meaning`
- `fluency`
- `coherence`
- `poeticness`

For `ashaar` and `fanar`:

- `meaning`
- `fluency`
- `coherence`
- `poeticness`

## Columns To Add

Qwen 3.7 Max:

- `qwen3_7_max_description_adherence`
- `qwen3_7_max_meaning`
- `qwen3_7_max_fluency`
- `qwen3_7_max_coherence`
- `qwen3_7_max_poeticness`

GPT-5.6 Terra:

- `gpt5_6_terra_description_adherence`
- `gpt5_6_terra_meaning`
- `gpt5_6_terra_fluency`
- `gpt5_6_terra_coherence`
- `gpt5_6_terra_poeticness`

Only applicable columns should be filled per dataset. Existing columns like `meaning`, `fluency`, `coherence`, `poeticness`, and `description_adherence` must stay unchanged.

## Environment

Required variables:

- `OPENROUTER_API_KEY`
- `HF_TOKEN`, if the Hugging Face datasets require authenticated access

The runner searches for `.env` in the current directory and parent directories. On this machine the existing env file was at:

```text
C:\Users\Ahmad Abbas\Desktop\Shaer-Project\.env
```

On a new machine, either place `.env` in the repo tree or export the variables in the shell.

## Commands

Install/check dependencies if needed:

```powershell
python -m pip install pandas pyarrow pyyaml huggingface_hub requests curl_cffi
```

Pilot run before spending money:

```powershell
python extra/run_additional_judges.py --limit-rows 2 --workers 2
```

Full run:

```powershell
python extra/run_additional_judges.py --workers 4
```

Equivalent Windows launcher:

```powershell
.\extra\run_additional_judges.ps1 --workers 4
```

If PowerShell blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\extra\run_additional_judges.ps1 --workers 4
```

Optional upload after validation:

```powershell
python extra/run_additional_judges.py --workers 4 --push-to-hf
```

Do not use `--push-to-hf` with `--limit-rows`.

## Output

Default output root:

```text
extra/judge_runs/additional_judges
```

Important files:

- `validation_summary.json`
- `additional_judge_results.md`
- `updated_datasets/<dataset>/<split-file>`
- `scored_rows/<model-prefix>/<dataset>/scores.jsonl`
- `cache/<model-prefix>/<dataset>/<metric>/*.json`

The cache makes the run resumable. If the process stops, rerun the same command and completed calls should be reused.

## Validation Checklist

- All four datasets have exactly `3481` rows in the full run.
- No duplicate row IDs.
- Existing unprefixed metric columns remain present.
- Every applicable prefixed score exists.
- Every applicable prefixed score is an integer in `[1, 5]`.
- `description_adherence` is filled only for `shaer` and `yehia`.
- `ashaar` and `fanar` do not get non-null description-adherence judge columns.
- Aggregate means are written to `additional_judge_results.md`.
- `validation_summary.json` reports zero missing/invalid applicable scores.

## Cost And Scale

Per judge:

- Rows: `13,924`
- Calls: `62,658`
- Estimated input tokens: `32,447,138`

Estimated cost:

- `qwen/qwen3.7-max`: `$49.52 / $66.44 / $83.35` min/avg/max
- `openai/gpt-5.6-terra`: `$43.38 / $72.04 / $100.71` min/avg/max
- Both together: about `$92.90 / $138.48 / $184.06`

Expected wall time with `4` workers is roughly `8-15` hours per model, depending on OpenRouter/provider rate limits.
