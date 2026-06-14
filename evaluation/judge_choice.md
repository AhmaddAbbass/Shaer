# Judge Choice

## Setup

We evaluated candidate judge LLMs on a stratified bank of `50` reference poems sampled from the held-out Shaer test set.
The bank covers all meters in the test distribution, with most meters contributing `4` poems and the rarest meters contributing `3` poems.
Each judge scored every reference poem on five metrics: `Description Adherence`, `Meaning`, `Fluency`, `Coherence`, and `Poeticness`.
All runs used the same prompts, the same tool-calling output schema, and the same one-metric-at-a-time evaluation pipeline.

### Meter Distribution

- `البسيط`: `4`
- `الخفيف`: `4`
- `الرجز`: `4`
- `الرمل`: `4`
- `السريع`: `4`
- `الطويل`: `4`
- `الكامل`: `4`
- `المتقارب`: `4`
- `المجتث`: `4`
- `المديد`: `3`
- `المنسرح`: `4`
- `الهزج`: `3`
- `الوافر`: `4`

## Candidates

- `Claude Sonnet 4.6`
- `Qwen 3.5 35B A3B`
- `Qwen 3 235B A22B 2507`

## Per-Metric Results

| Judge Model | Metric | Rows | Mean | Median | Min | Max | Avg Latency (s) | Total Time (s) | Avg Cost ($) | Total Cost ($) | Projected Full Cost, 4 datasets ($) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | Description Adherence | 50 | 4.88 | 5.00 | 3.00 | 5.00 | 3.15 | 157.68 | 0.004397 | 0.219849 | 61.22 |
| Claude Sonnet 4.6 | Meaning | 50 | 4.04 | 4.00 | 2.00 | 5.00 | 3.41 | 170.51 | 0.003912 | 0.195582 | 54.47 |
| Claude Sonnet 4.6 | Fluency | 50 | 4.20 | 4.00 | 2.00 | 5.00 | 3.50 | 174.87 | 0.003849 | 0.192432 | 53.59 |
| Claude Sonnet 4.6 | Coherence | 50 | 4.18 | 4.00 | 2.00 | 5.00 | 3.31 | 165.70 | 0.003852 | 0.192582 | 53.63 |
| Claude Sonnet 4.6 | Poeticness | 50 | 3.82 | 4.00 | 2.00 | 5.00 | 3.35 | 167.39 | 0.004005 | 0.200232 | 55.76 |
| Qwen 3.5 35B A3B | Description Adherence | 50 | 4.60 | 5.00 | 2.00 | 5.00 | 0.93 | 46.48 | 0.000117 | 0.005834 | 1.62 |
| Qwen 3.5 35B A3B | Meaning | 50 | 3.14 | 3.00 | 2.00 | 5.00 | 0.88 | 44.22 | 0.000102 | 0.005109 | 1.42 |
| Qwen 3.5 35B A3B | Fluency | 50 | 2.30 | 2.00 | 2.00 | 5.00 | 0.91 | 45.51 | 0.000100 | 0.004980 | 1.39 |
| Qwen 3.5 35B A3B | Coherence | 50 | 3.62 | 4.00 | 2.00 | 5.00 | 0.90 | 45.14 | 0.000097 | 0.004861 | 1.35 |
| Qwen 3.5 35B A3B | Poeticness | 50 | 2.92 | 3.00 | 2.00 | 4.00 | 0.89 | 44.50 | 0.000104 | 0.005183 | 1.44 |
| Qwen 3 235B A22B 2507 | Description Adherence | 50 | 5.00 | 5.00 | 5.00 | 5.00 | 1.94 | 96.85 | 0.000064 | 0.003189 | 0.89 |
| Qwen 3 235B A22B 2507 | Meaning | 50 | 4.34 | 4.00 | 3.00 | 5.00 | 1.89 | 94.37 | 0.000055 | 0.002733 | 0.76 |
| Qwen 3 235B A22B 2507 | Fluency | 50 | 4.88 | 5.00 | 4.00 | 5.00 | 1.90 | 95.19 | 0.000053 | 0.002652 | 0.74 |
| Qwen 3 235B A22B 2507 | Coherence | 50 | 4.76 | 5.00 | 2.00 | 5.00 | 1.88 | 93.93 | 0.000053 | 0.002647 | 0.74 |
| Qwen 3 235B A22B 2507 | Poeticness | 50 | 4.40 | 4.00 | 3.00 | 5.00 | 1.98 | 98.84 | 0.000057 | 0.002832 | 0.79 |

## Overall Totals

| Judge Model | Actual Time on 50 Poems x 5 Metrics | Actual Cost on 50 Poems x 5 Metrics ($) | Projected Time for 1 dataset, 3,481 rows (serial) | Projected Cost for 1 dataset, 3,481 rows ($) | Projected Time for 4 datasets, 13,924 rows (serial) | Projected Cost for 4 datasets ($) |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 13.94 min | 1.000677 | 16.17 h | 69.67 | 64.68 h | 278.67 |
| Qwen 3.5 35B A3B | 3.76 min | 0.025966 | 4.37 h | 1.81 | 17.47 h | 7.23 |
| Qwen 3 235B A22B 2507 | 7.99 min | 0.014052 | 9.27 h | 0.98 | 37.07 h | 3.91 |

## Decision

We select `Qwen 3 235B A22B 2507` as the final judge model.

Reasons:

- It achieved the strongest reference-poem scores among the tested affordable models.
- It matched or exceeded Claude Sonnet 4.6 on the 50-poem calibration bank across the five evaluation metrics.
- It completed the entire calibration cleanly with `0` bad rows.
- Its projected full-benchmark cost is negligible relative to Claude.

In particular, the projected full evaluation cost is approximately:

- `Qwen 3 235B A22B 2507`: `$3.91` for all four datasets
- `Claude Sonnet 4.6`: `$278.67` for all four datasets

This makes the chosen Qwen judge both the best-value and the most practical option for full-scale row-level evaluation.
