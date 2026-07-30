# Extra Evaluation Provenance

This folder records the locked-in evaluation artifacts that match the final reported results in `evaluation/results2.md`.

## Confirmed Results

These are the current verified results:

| Model | Meter | Count Adherence | Description Adherence | Meaning | Fluency | Coherence | Poeticness |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Shaer` | `0.9064` | `0.9792` | `4.75` | `3.72` | `4.31` | `3.69` | `3.82` |
| `Yehia-7B` | `0.1494` | `0.7680` | `4.42` | `3.75` | `4.21` | `3.82` | `3.73` |
| `Fanar-Diwan` | `0.6488` | `N/A` | `N/A` | `3.70` | `4.26` | `3.69` | `3.79` |
| `Ashaar` | `0.8535` | `N/A` | `N/A` | `2.84` | `3.46` | `2.84` | `3.00` |

Shared 4-metric averages:

- `Shaer`: `3.89`
- `Yehia-7B`: `3.88`
- `Fanar-Diwan`: `3.86`
- `Ashaar`: `3.04`

Description-conditioned 5-metric averages:

- `Shaer`: `4.06`
- `Yehia-7B`: `3.99`

## Judge Model

The locked judge model used for the strict `v2` results is:

- `qwen/qwen3-235b-a22b-2507`

## Verified Row-Level Evaluation Datasets

These are the exact Hugging Face datasets containing the evaluated rows. Each row contains a generated poem and its stored evaluation fields. We inspected these datasets directly and recomputed the metric means from the row-level columns to confirm that they match `evaluation/results2.md` after rounding.

- `Shaer-AI/shaer-sft-test`
- `Shaer-AI/shaer-eval-ashaar-native-controls`
- `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`
- `Shaer-AI/fanar-eval-native-prompt`

Verification notes:

- Hugging Face access was confirmed using the local `HF_TOKEN` from `C:\Users\Ahmad Abbas\Desktop\Shaer-Project\.env`.
- All four datasets were accessible.
- Recomputed means matched the values reported in `evaluation/results2.md`.
- Missing row-level judge values are excluded from metric means, consistent with the note in `evaluation/results2.md`.

## Metrics and Prompt Pack

The semantic and literary metrics were computed by using the judge model above together with the strict `v2` judge prompt pack:

- `description_adherence`
- `meaning`
- `fluency`
- `coherence`
- `poeticness`

The exact locked prompt pack is copied in this folder as:

- [judge_prompts_v2_strict.yaml](./judge_prompts_v2_strict.yaml)

This YAML contains:

- the shared system prompt
- the exact per-metric user prompt templates
- the scoring anchors for `1` through `5`
- the applicability of `description_adherence` to description-conditioned models

## Notes

- `meter` and `count_adherence` are structural metrics and are not produced by the LLM judge prompt pack.
- `description_adherence` applies only to the description-conditioned setups used for `Shaer` and `Yehia-7B`.
- `Ashaar` and `Fanar-Diwan` were evaluated only on the four shared literary metrics plus structural meter.
