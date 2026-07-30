# Evaluation Numbers

This file records the current planning numbers for running additional judge LLMs on the released evaluation datasets.

## Dataset Sizes

Each released evaluation dataset contains `3481` rows.

- `Shaer-AI/shaer-sft-test`: `3481`
- `Shaer-AI/shaer-eval-ashaar-native-controls`: `3481`
- `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`: `3481`
- `Shaer-AI/fanar-eval-native-prompt`: `3481`

Total rows across all four datasets:

- `13,924`

## Judge Calls Per Row

Metric applicability:

- `Shaer`: `5` judge calls per row
- `Yehia`: `5` judge calls per row
- `Ashaar`: `4` judge calls per row
- `Fanar`: `4` judge calls per row

This follows the current metric setup:

- `Shaer` and `Yehia` use:
  - `description_adherence`
  - `meaning`
  - `fluency`
  - `coherence`
  - `poeticness`
- `Ashaar` and `Fanar` use:
  - `meaning`
  - `fluency`
  - `coherence`
  - `poeticness`

## Judge Calls Per Dataset

- `Shaer`: `3481 x 5 = 17,405`
- `Yehia`: `3481 x 5 = 17,405`
- `Ashaar`: `3481 x 4 = 13,924`
- `Fanar`: `3481 x 4 = 13,924`

Total calls for one additional judge over all four datasets:

- `62,658`

Breakdown:

- Shared 4-metric calls over all rows: `13,924 x 4 = 55,696`
- Extra `description_adherence` calls for `Shaer` and `Yehia`: `6,962`

## Prompt Token Estimates

Important note:

- The released datasets do not preserve the original row-level judge token usage.
- The numbers below are prompt-token estimates computed from the exact locked `judge_v2_strict` prompts and the real released poem rows.
- Tokenization was estimated with a Qwen-family tokenizer (`Qwen/Qwen3-8B`) as a proxy.
- These are suitable for planning and pricing estimates, but they are not exact API billing numbers.

### Per-Dataset Total Estimated Prompt Tokens

- `Shaer`: `10,289,695`
- `Yehia`: `8,836,285`
- `Ashaar`: `5,551,701`
- `Fanar`: `7,769,457`

Total estimated prompt tokens for one additional judge over all four datasets:

- `32,447,138`

### Average Estimated Prompt Tokens Per Call

- `Shaer`: `591.19`
- `Yehia`: `507.69`
- `Ashaar`: `398.71`
- `Fanar`: `557.99`
- Overall: `517.85`

### Per-Metric Average Estimated Prompt Tokens

#### Shaer

- `description_adherence`: `717.92`
- `meaning`: `560.26`
- `fluency`: `546.26`
- `coherence`: `547.26`
- `poeticness`: `584.26`

#### Yehia

- `description_adherence`: `634.41`
- `meaning`: `476.75`
- `fluency`: `462.75`
- `coherence`: `463.75`
- `poeticness`: `500.75`

#### Ashaar

- `meaning`: `399.46`
- `fluency`: `385.46`
- `coherence`: `386.46`
- `poeticness`: `423.46`

#### Fanar

- `meaning`: `558.74`
- `fluency`: `544.74`
- `coherence`: `545.74`
- `poeticness`: `582.74`

## Completion Token Estimates

The judge pipeline uses score-only structured output and sets `max_tokens=128` per call.

Practical interpretation:

- hard upper bound per call: `128` completion tokens
- realistic completion should be much smaller because the response is effectively just a score payload

Minimal JSON-only estimate:

- average completion tokens per call: about `6`
- total completion tokens for one additional judge over all four datasets:
  - `62,658 x 6 = 375,948`

This `6`-token number should be treated as a lower-bound style estimate, not an exact billing figure.

## Planning Summary

For one additional judge model over all four released evaluation datasets:

- Rows: `13,924`
- Judge calls: `62,658`
- Estimated prompt tokens: `32,447,138`
- Minimal JSON-only completion estimate: `375,948`

## Source Notes

These numbers were derived from:

- the released Hugging Face evaluation datasets
- the locked strict `v2` judge prompt pack in [judge_prompts_v2_strict.yaml](./judge_prompts_v2_strict.yaml)
- the current evaluation code in `evaluation/`

The main provenance overview is in [README.md](./README.md).
