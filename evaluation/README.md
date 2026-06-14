# Evaluation

This folder contains the paper-faithful evaluation code for Shaer and the comparison baselines.

## Final benchmark shape

The benchmark uses direct one-row-per-prompt published datasets:

- `Shaer-AI/shaer-sft-test`
- `Shaer-AI/shaer-eval-ashaar-native-controls`
- `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`
- `Shaer-AI/fanar-eval-native-prompt`

The repo intentionally does not use an older multi-sample selection workflow in the public evaluation story.

## Structural metrics

Core structural scoring files:

- [metrics.py](./metrics.py)
- [score_meter_count.py](./score_meter_count.py)
- [aggregate_results.py](./aggregate_results.py)

These scripts reuse the established 4BiLSTM meter-classifier path from the repo and compute:

- `meter`
- `count_adherence`

## Strict judge evaluation

Strict judge evaluation files:

- [judge_llm.py](./judge_llm.py)
- [judge_prompts_v2_strict.yaml](./judge_prompts_v2_strict.yaml)
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

- `description_adherence` applies to `Shaer` and `Yehia`
- `Ashaar` and `Fanar` are scored only on the shared literary metrics

## Result notes

- [judge_choice.md](./judge_choice.md) — 50-reference judge-model selection study
- [results.md](./results.md) — earlier judge notes
- [results2.md](./results2.md) — final strict `v2` results used for reporting

## Detached execution helpers

- [launch_full_judge_detached.ps1](./launch_full_judge_detached.ps1)
- [launch_monitor_detached.ps1](./launch_monitor_detached.ps1)
- [run_full_judge_all.sh](./run_full_judge_all.sh)

## Scope

This folder is meant to preserve the final reproducible evaluation workflow that matches the reported benchmark:

- structural scoring from the direct evaluation datasets
- strict `v2` LLM judging
- final dataset augmentation with judge metrics
