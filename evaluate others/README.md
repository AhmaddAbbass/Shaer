# Non-Shaer Baselines

This folder contains the utilities used to generate and score non-Shaer comparison systems for the final direct benchmark.

## Final published comparison datasets

The paper-facing comparison set is:

- `Shaer-AI/shaer-eval-ashaar-native-controls`
- `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`
- `Shaer-AI/fanar-eval-native-prompt`

Each of these datasets is one-row-per-prompt and is aligned to the final Shaer evaluation set.

## Main scripts

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

## Current benchmark interpretation

- `Ashaar` is evaluated in its native control-token setup.
- `Yehia` is evaluated with the description-conditioned instruction template.
- `Fanar` is evaluated in its native prompt setup.

Because the final benchmark is already direct one-row-per-prompt, this folder does not rely on an older multi-sample Shaer source dataset in the public workflow.

## Ashaar native manifest

The Ashaar-native manifest is reconstructed from the final source split dataset:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`, split `test`

Example:

```bash
python "evaluate others/build_ashaar_native_manifest.py" \
  --prompt-mode controls \
  --output-jsonl evaluation/outputs/ashaar_native_manifest_controls.jsonl
```

## Scoring

Baseline scoring reuses the same structural metric path as Shaer:

```bash
python "evaluate others/score_baseline_meter_count.py" \
  --input-jsonl <baseline_rows.jsonl> \
  --output-jsonl <baseline_rows_scored.jsonl>
```

The final literary metrics are added by the strict judge workflow in `evaluation/`.
