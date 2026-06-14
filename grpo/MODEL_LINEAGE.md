# Shaer GRPO Model Lineage

Last updated: 2026-04-13 UTC

## Published Model Order

1. `Shaer-AI/Shaer-adapters`
   - SFT baseline
2. `Shaer-AI/Shaer-adapters-grpo`
   - first GRPO artifact, historically hacked
3. `Shaer-AI/Shaer-adapters-grpo-vnext`
   - stricter structure-side repair stage
4. `Shaer-AI/Shaer-adapters-grpo-friend-v1`
   - first judge-centered run
5. `Shaer-AI/Shaer-adapters-grpo-friend-v1-easyfirst`
   - healthier multiplicative judge-centered rerun
6. `Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v2`
   - best completed GRPO run
7. `Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v3`
   - stopped `meter * judge` stage
8. `Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v4`
   - stopped judge-gated-aux stage

## Best Current Interpretations

- trusted baseline:
  `Shaer-AI/Shaer-adapters`
- best completed GRPO stage:
  `Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v2`
- most important reward-hack cautionary stage:
  `Shaer-AI/Shaer-adapters-grpo`
- most important stopped-but-useful diagnostic stage:
  `Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v3`

## Current Next Step

There is no launched `v5` model yet.

The repo is currently prepared for an unlaunched dual-judge candidate:

```text
hard_gate * (
    0.45 * (meter * judge_meaning_fit)
  + 0.20 * judge_naturalness
  + 0.20 * (count_adherence * judge_meaning_fit)
  + 0.15 * (repeat_soft * judge_naturalness)
)
```

This file is the compact public lineage note for the exploratory GRPO branch of the project.
