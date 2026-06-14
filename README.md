# Shaer

Shaer is a classical Arabic poetry generation project built around supervised fine-tuning of `Navid-AI/Yehia-7B-preview` for controlled `الشعر العمودي`.

The paper-facing project centers on:

- curated Arabic poetry data with normalized `base_meter` and `form`
- regenerated semantic descriptions for conditioning
- QLoRA SFT training over Yehia-7B
- evaluation on structural and literary metrics

The public repository intentionally reflects the paper-faithful workflow only.

## Read order

1. [shaer.md](./shaer.md) — project dossier and paper-writing context
2. [training_details.md](./training_details.md) — dataset and training lineage
3. [evaluation/results2.md](./evaluation/results2.md) — final strict judge results
4. [evaluation/judge_choice.md](./evaluation/judge_choice.md) — judge-model selection study

## Repository layout

- [description_generation/](./description_generation/) — description regeneration and final SFT dataset assembly
- [sft/](./sft/) — supervised fine-tuning configuration and training code
- [evaluation/](./evaluation/) — meter/count scoring, strict judge evaluation, and final results notes
- [evaluate others/](./evaluate%20others/) — baseline-generation and baseline-scoring utilities for non-Shaer models
- [grpo/](./grpo/) — exploratory post-SFT reinforcement-learning code, not part of the paper’s core reported method

## Released artifacts referenced in the paper

- final split dataset for SFT
- `Shaer-AI/Shaer-adapters`
- direct one-row-per-prompt evaluation datasets
- strict `v2` LLM-judge prompts and evaluation scripts

## Notes

- The paper story is SFT-first. GRPO code remains in the repo as exploratory follow-up work, not as part of the main reported method.
- The final evaluation path in this repo uses direct numbered evaluation datasets. Older multi-sample selection workflows are intentionally not part of the public history.
