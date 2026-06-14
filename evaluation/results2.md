# Results V2

## Scope

This file records the final results of the **strict v2 judge setup**. It is intentionally separate from [results.md](./results.md), which documents the earlier `judge_v1` run.

The evaluation has two components:

1. **Formal structural metrics**
   - `meter`
   - `count_adherence`

2. **LLM-judge semantic and literary metrics**
   - `description_adherence`
   - `meaning`
   - `fluency`
   - `coherence`
   - `poeticness`

The `v2` judge prompts were introduced because the earlier `v1` setup showed clear score compression and excessive generosity across stronger models.

## Evaluated systems

- `Shaer`: `Shaer-AI/shaer-sft-test`
- `Ashaar`: `Shaer-AI/shaer-eval-ashaar-native-controls`
- `Yehia`: `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`
- `Fanar`: `Shaer-AI/fanar-eval-native-prompt`

## Judge model

The judge model is:

- `qwen/qwen3-235b-a22b-2507`

This model was selected earlier from the calibration study documented in [judge_choice.md](./judge_choice.md).

## Why a v2 rerun was needed

The first judge run (`judge_v1`) produced results that were too compressed. In particular:

- `Shaer`, `Yehia`, and `Fanar` often landed in nearly identical score ranges
- `fluency` in particular showed a ceiling effect
- the judge appeared too willing to assign `4` and `5` to merely competent outputs

To address this, the `v2` prompts were rewritten to:

- make the scoring scale harsher
- reserve `5` for rare, exceptional outputs
- explicitly distinguish `acceptable` from `strong`
- reduce the tendency to inflate grammatically correct but ordinary poems

## Strict v2 judge prompts

The frozen prompt pack is:

- [judge_prompts_v2_strict.yaml](./judge_prompts_v2_strict.yaml)

### Shared system prompt

```text
You are a strict expert evaluator of classical-style Arabic poetry.

Evaluate only the requested metric.
Use the full 1-5 scale critically.

General scale:
1 = failed or very poor
2 = weak
3 = acceptable but ordinary
4 = strong
5 = exceptional

A score of 5 should be rare and reserved for poems that strongly satisfy the metric at a high classical-style Arabic poetry standard.
Do not give 4 or 5 for merely understandable, grammatical, or pleasant text.

Return valid JSON only in this exact format:
{"score": <integer from 1 to 5>}
```

### Metric prompts

The metric-specific prompts are stored in [judge_prompts_v2_strict.yaml](./judge_prompts_v2_strict.yaml). They preserve the same five evaluation dimensions as `v1`, but use stricter anchors for `3`, `4`, and `5`.

## Metric applicability by model

`description_adherence` is evaluated only for models that receive the description-conditioned task:

- `Shaer`: yes
- `Yehia`: yes
- `Ashaar`: no
- `Fanar`: no

This avoids unfairly scoring native-prompt baselines on a metric they were not designed to satisfy.

## Diagnostic validation

Before launching the full rerun, a strict `v2` diagnostic was executed on `50` rows per model.

The diagnostic behaved as intended:

- `Ashaar` dropped clearly across the semantic and literary metrics
- `Fanar` no longer tied `Shaer` across everything
- `Yehia` remained competitive
- the scale used more `2/3/4` and fewer inflated `5`s

## Final strict v2 run

All four datasets completed successfully.

## Structural metric context

The main structural metric is meter adherence. The current comparison set gives:

| Model | Meter Mean | Evaluation setup |
|---|---:|---|
| `Shaer` | `0.9064` | selected best generation per held-out prompt |
| `Ashaar` | `0.8535` | native controls |
| `Fanar` | `0.6488` | native prompt |
| `Yehia` | `0.1494` | instruction baseline |

These scores are essential for interpreting the judge results below. The judge metrics evaluate semantic and literary quality, while the meter metric captures formal prosodic control.

## Final v2 strict judge results

### Main table

| Model | Rows | Description Adherence | Meaning | Fluency | Coherence | Poeticness |
|---|---:|---:|---:|---:|---:|---:|
| `Shaer` | `3481` | `4.75` | `3.72` | `4.31` | `3.69` | `3.82` |
| `Ashaar` | `3481` | `n/a` | `2.84` | `3.46` | `2.84` | `3.00` |
| `Yehia` | `3481` | `4.42` | `3.75` | `4.21` | `3.82` | `3.73` |
| `Fanar` | `3481` | `n/a` | `3.70` | `4.26` | `3.69` | `3.79` |

### Common 4-metric average

To compare all four systems fairly on the shared semantic/literary metrics, we average:

- `meaning`
- `fluency`
- `coherence`
- `poeticness`

| Model | 4-Metric Average |
|---|---:|
| `Shaer` | `3.89` |
| `Yehia` | `3.88` |
| `Fanar` | `3.86` |
| `Ashaar` | `3.04` |

### 5-metric average for description-conditioned models

For the two models that actually receive the description-conditioned task, we also report the 5-metric average:

| Model | 5-Metric Average |
|---|---:|
| `Shaer` | `4.06` |
| `Yehia` | `3.99` |

## Interpretation

### 1. Shaer vs Ashaar

`Ashaar` is now clearly separated downward across all four shared judge metrics:

- `Meaning`: `2.84`
- `Fluency`: `3.46`
- `Coherence`: `2.84`
- `Poeticness`: `3.00`

This is a much more believable result than the earlier compressed `v1` scores. Combined with the structural metrics, it suggests that `Ashaar` remains capable of metrically strong native verse generation, but it is substantially weaker than `Shaer` in semantic and literary quality under this evaluation.

### 2. Shaer vs Yehia

`Yehia` remains a strong base model. The strict rerun still shows that:

- `Yehia` is competitive on semantic and literary quality
- `Shaer` remains slightly ahead overall
- `Shaer` is stronger on `description_adherence`
- `Shaer` and `Yehia` are close on the other literary axes

This supports the cleanest interpretation of the SFT result:

> `Shaer` preserves the strong literary quality of the `Yehia` base model while substantially improving formal poetic control, especially metrical adherence.

### 3. Fanar

Under the stricter prompts, `Fanar` no longer collapses into the exact same score band as `Shaer`. It remains a relatively strong native poetry baseline on literary surface quality, but it does not overtake `Shaer` in the overall 4-metric average.

### 4. Overall ranking

Under the strict rerun:

1. `Shaer`
2. `Yehia`
3. `Fanar`
4. `Ashaar`

The difference between `Shaer` and `Yehia` is narrow on the common judge metrics, but once the structural metric context is incorporated, `Shaer` remains the strongest overall system in the benchmark.

## Recommended paper takeaway

The strongest paper sentence from these results is:

> `Shaer` achieves the strongest overall strict-judge average while substantially outperforming the baselines on meter adherence, indicating that the SFT adapter improves formal control without sacrificing broader poetic quality.

For the base-model comparison specifically:

> Relative to `Yehia`, `Shaer` preserves comparable semantic and literary quality while delivering much stronger metrical control and structured generation behavior.

## Notes on missing rows

The strict rerun tolerated isolated row-level judge failures instead of killing the whole run. As a result:

- all datasets completed
- any metric with a missing value is excluded from that metric's mean instead of being silently treated as zero

This makes the final aggregation more robust than the earlier fail-fast behavior.
