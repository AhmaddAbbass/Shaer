# Additional Judge Run Handoff

Goal: rerun the locked strict `v2` judge prompts on the four released row-level evaluation datasets using two new OpenRouter judge models, and append prefixed metric columns without overwriting the existing `results2.md` columns.

## Target Judge Models

- `qwen/qwen3.7-max`
- `openai/gpt-5.6-terra`

Use these exact model IDs unless OpenRouter reports they are unavailable.

## Inputs

- Repo: `C:\Users\Ahmad Abbas\Desktop\Shaer-Project\paper\Shaer`
- Existing env file on this machine: `C:\Users\Ahmad Abbas\Desktop\Shaer-Project\.env`
- Required env var: `OPENROUTER_API_KEY`
- Important: the current judge code loads `.env` from the repo root, `C:\Users\Ahmad Abbas\Desktop\Shaer-Project\paper\Shaer\.env`. If that file does not exist on the new machine, either export `OPENROUTER_API_KEY` in the shell before running or copy only the needed variables into the repo-root `.env`.
- Prompt pack: `evaluation/judge_prompts_v2_strict.yaml`
- Existing result table to compare against: `evaluation/results2.md`

Evaluation datasets:

- `Shaer-AI/shaer-sft-test`, key `shaer`, rows `3481`
- `Shaer-AI/shaer-eval-ashaar-native-controls`, key `ashaar`, rows `3481`
- `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`, key `yehia`, rows `3481`
- `Shaer-AI/fanar-eval-native-prompt`, key `fanar`, rows `3481`

## Metrics

For `shaer` and `yehia`, run:

- `description_adherence`
- `meaning`
- `fluency`
- `coherence`
- `poeticness`

For `ashaar` and `fanar`, run:

- `meaning`
- `fluency`
- `coherence`
- `poeticness`

Do not run `description_adherence` for `ashaar` or `fanar`.

## Required New Columns

For Qwen 3.7 Max:

- `qwen3_7_max_description_adherence`
- `qwen3_7_max_meaning`
- `qwen3_7_max_fluency`
- `qwen3_7_max_coherence`
- `qwen3_7_max_poeticness`

For GPT-5.6 Terra:

- `gpt5_6_terra_description_adherence`
- `gpt5_6_terra_meaning`
- `gpt5_6_terra_fluency`
- `gpt5_6_terra_coherence`
- `gpt5_6_terra_poeticness`

Only create/fill the applicable columns per dataset. Preserve the existing unprefixed columns such as `meaning`, `fluency`, `coherence`, `poeticness`, and `description_adherence`.

## Existing Scripts To Inspect

- `evaluation/full_judge_orchestrator.py`
- `evaluation/full_judge_worker.py`
- `evaluation/judge_llm.py`
- `evaluation/judge_dataset_registry.py`
- `evaluation/merge_full_judge_results.py`
- `evaluation/push_judge_metrics_to_hf_datasets.py`

Important: `push_judge_metrics_to_hf_datasets.py` currently writes unprefixed metric names. Adapt or replace it so it appends prefixed judge columns instead of overwriting canonical metric columns.

## Recommended Execution Plan

1. Verify `.env` contains a valid `OPENROUTER_API_KEY`.
2. Run a tiny pilot for each model, for example `--limit-rows 2`, using `evaluation/judge_prompts_v2_strict.yaml`.
3. Confirm the pilot produces valid integer scores from `1` to `5` for every applicable metric.
4. Run the full judge for `qwen/qwen3.7-max` into a dedicated run directory.
5. Run the full judge for `openai/gpt-5.6-terra` into a separate dedicated run directory.
6. Merge worker outputs for each dataset/model.
7. Materialize updated dataset files with new prefixed columns.
8. Validate every dataset before any upload or final handoff.

Suggested run directory pattern:

```powershell
evaluation/outputs/judge_qwen3_7_max_YYYYMMDD_HHMM
evaluation/outputs/judge_gpt5_6_terra_YYYYMMDD_HHMM
```

Suggested full-run command shape:

```powershell
python evaluation/full_judge_orchestrator.py `
  --run-dir evaluation/outputs/judge_qwen3_7_max_FULL `
  --workers 4 `
  --judge-model qwen/qwen3.7-max `
  --prompt-file evaluation/judge_prompts_v2_strict.yaml
```

Use the same command for Terra with `--judge-model openai/gpt-5.6-terra` and a different `--run-dir`.

## Scale And Cost

Per judge model:

- Rows: `13,924`
- Calls: `62,658`
- Estimated input tokens: `32,447,138`
- Estimated output tokens: `375,948` minimum, `8,020,224` max cap

Estimated full-run cost from `extra/llms.md`:

- `qwen/qwen3.7-max`: `$49.52 / $66.44 / $83.35` min/avg/max
- `openai/gpt-5.6-terra`: `$43.38 / $72.04 / $100.71` min/avg/max
- Both models: about `$92.90 / $138.48 / $184.06`

Expected runtime with `4` workers:

- Large non-reasoning judge: plan around `8-15` hours per model unless the provider is rate-limited.

## Validation Checklist

- All four datasets still have exactly `3481` rows.
- Existing unprefixed judge columns are preserved unchanged.
- All applicable prefixed columns exist for each model.
- No applicable prefixed score is null/missing.
- Every prefixed score is an integer in `[1, 5]`.
- `description_adherence` prefixed columns are present for `shaer` and `yehia`.
- `description_adherence` prefixed columns are absent or null-only for `ashaar` and `fanar`.
- No duplicate row IDs appear after merging.
- Worker logs contain no repeated parsing/API failures.
- Aggregate means are recomputed per dataset/model/metric.
- New aggregates are saved in a compact markdown or JSON summary.

## Final Deliverables

- Full run directories for both judges.
- Updated local dataset split files with prefixed columns.
- Validation summary showing row counts, missing counts, score ranges, and aggregate means.
- If explicitly requested, push the updated dataset files to the same Hugging Face dataset repos.

Do not overwrite `evaluation/results2.md`; create a new summary file for the additional judges.
