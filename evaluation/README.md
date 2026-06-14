# Evaluation

This folder contains the paper-faithful benchmark code for Shaer and the comparison baselines.

## Evaluated systems

The final benchmark uses direct one-row-per-prompt released datasets:

- `Shaer-AI/shaer-sft-test`
- `Shaer-AI/shaer-eval-ashaar-native-controls`
- `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`
- `Shaer-AI/fanar-eval-native-prompt`

## Structural evaluation

Core files:

- [metrics.py](./metrics.py)
- [score_meter_count.py](./score_meter_count.py)
- [aggregate_results.py](./aggregate_results.py)

These scripts compute:

- `meter`
- `count_adherence`

The meter score is produced by the repo's established BiLSTM-based meter-classification path.

## Strict LLM-judge evaluation

Core files:

- [judge_prompts_v2_strict.yaml](./judge_prompts_v2_strict.yaml)
- [judge_llm.py](./judge_llm.py)
- [judge_dataset_registry.py](./judge_dataset_registry.py)
- [full_judge_orchestrator.py](./full_judge_orchestrator.py)
- [full_judge_worker.py](./full_judge_worker.py)
- [merge_full_judge_results.py](./merge_full_judge_results.py)
- [push_judge_metrics_to_hf_datasets.py](./push_judge_metrics_to_hf_datasets.py)

Judge metrics:

- `description_adherence`
- `meaning`
- `fluency`
- `coherence`
- `poeticness`

Metric applicability:

- `description_adherence` is evaluated only for `Shaer` and `Yehia`
- `Ashaar` and `Fanar` are evaluated on the four shared literary metrics only

Final judge model used in the paper:

- `qwen/qwen3-235b-a22b-2507`

## Key result notes

- [judge_choice.md](./judge_choice.md) - judge-model selection study
- [results2.md](./results2.md) - final strict `v2` results used in reporting

The older [results.md](./results.md) note is kept only as historical context for the earlier judge setup.

## Detached runners

- [launch_full_judge_detached.ps1](./launch_full_judge_detached.ps1)
- [launch_monitor_detached.ps1](./launch_monitor_detached.ps1)
- [run_full_judge_all.sh](./run_full_judge_all.sh)

## Scope

This folder preserves the final reproducible evaluation workflow that matches the paper:

- direct benchmark datasets
- structural meter and count scoring
- strict `v2` judge prompts
- final dataset augmentation with released judge metrics
