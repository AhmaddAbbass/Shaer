# Non-Shaer Baselines

This folder contains the baseline-generation and normalization code used for the paper comparisons against Shaer.

## Released comparison datasets

- `Shaer-AI/shaer-eval-ashaar-native-controls`
- `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`
- `Shaer-AI/fanar-eval-native-prompt`

Each released dataset is aligned one-row-per-prompt with the final Shaer evaluation set.

## Benchmark interpretation

- `Ashaar` is evaluated in its native control-token format.
- `Yehia` is evaluated with the same description-conditioned instruction template used for Shaer.
- `Fanar` is evaluated in its native prompt format.

## Exact scripts

- [build_ashaar_native_manifest.py](./build_ashaar_native_manifest.py)
- [generate_ashaar_native_baselines.py](./generate_ashaar_native_baselines.py)
- [generate_instruction_baselines.py](./generate_instruction_baselines.py)
- [generate_instruction_vllm.py](./generate_instruction_vllm.py)
- [generate_continuation_baselines.py](./generate_continuation_baselines.py)
- [normalize_ashaar_generations.py](./normalize_ashaar_generations.py)
- [normalize_fanar_prefix_generations.py](./normalize_fanar_prefix_generations.py)
- [score_baseline_meter_count.py](./score_baseline_meter_count.py)
- [aggregate_baseline_results.py](./aggregate_baseline_results.py)
- [combine_model_results.py](./combine_model_results.py)
- [export_model_results_bundle.py](./export_model_results_bundle.py)

## Structural scoring

Baseline structural scoring uses the same metric path as Shaer:

```bash
python "evaluate others/score_baseline_meter_count.py" \
  --input-jsonl <baseline_rows.jsonl> \
  --output-jsonl <baseline_rows_scored.jsonl>
```

The strict literary metrics are then added by the workflow in [../evaluation/](../evaluation/).
