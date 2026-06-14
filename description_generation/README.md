# Description Generation and Final SFT Dataset Assembly

This folder contains the dataset-construction path that turns the cleaned Ashaar-derived corpus into the final Shaer SFT dataset.

## What this stage does

It performs the paper-facing upstream work behind the final SFT release:

1. filter and normalize the classical-poetry corpus
2. generate `enhanced_description` fields with Qwen 3.5
3. verify and, when needed, regenerate weak descriptions
4. rebuild the final SFT prompt/completion fields
5. publish the final training datasets

## Paper-facing corpus counts

The paper distinguishes two related corpus sizes:

- **116,032 poems** after structural validation, length filtering, and meter-support filtering
- **114,065 poems** retained after description verification for downstream training

Those two numbers are not contradictory:

- `116,032` is the cleaned metrical corpus
- `114,065` is the verified-description subset reported for downstream training after the description-quality pipeline

## Released datasets

Main released training datasets:

- [`Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`](https://huggingface.co/datasets/Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500)
- [`Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`](https://huggingface.co/datasets/Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits)

Downstream model release:

- [`Shaer-AI/Shaer-adapters`](https://huggingface.co/Shaer-AI/Shaer-adapters)

## Exact scripts

Core files:

- [final_sft_dataset.py](./final_sft_dataset.py)
- [regenerate_descriptions.py](./regenerate_descriptions.py)
- [prompt_contracts.py](./prompt_contracts.py)

Supporting launch and monitoring helpers:

- [launch_final_sft_workers.sh](./launch_final_sft_workers.sh)
- [worker_status.py](./worker_status.py)
- [run_detached.sh](./run_detached.sh)

## Filtering rules

The cleaned metrical corpus is built by:

1. validating `poem verses` as a non-empty even-length list
2. computing `requested_bayts = len(poem verses) // 2`
3. dropping malformed odd-hemistich rows
4. keeping only rows with `requested_bayts <= 20`
5. computing post-filter meter counts
6. dropping base meters with support `< 500`

The final base meter removed by the support threshold was:

- `المتدارك`

## Prompt contracts

Prompt versions are defined in [prompt_contracts.py](./prompt_contracts.py):

- description generation prompt:
  - `pure_description_v8_poem_only_antidrift`
- final SFT prompt:
  - `final_sft_meter_emphasis_v2_num_lines`

### Description generation

Qwen 3.5 is used to produce `enhanced_description` as a faithful semantic summary of the poem, while explicitly avoiding:

- meter mentions
- rhyme mentions
- explicit poem-length mentions
- imperative or instruction-like phrasing

The exact system and user prompt builders live in:

- `DESCRIPTION_SYSTEM_PROMPT`
- `build_description_user_prompt(...)`

inside [prompt_contracts.py](./prompt_contracts.py).

### Final SFT prompt

The final SFT prompt conditions generation on:

- `base_meter`
- `form`
- requested number of hemistichs
- `enhanced_description`

The exact builder lives in:

- `build_sft_prompt(...)`

inside [prompt_contracts.py](./prompt_contracts.py).

## Output fields

The final dataset assembly keeps the original poem text and rebuilds:

- `enhanced_description`
- `sft_prompt`
- `sft_completion`
- `sft_full_text`
- `sft_num_lines`
- `sft_total_tokens`

## Recommended reading order

1. [prompt_contracts.py](./prompt_contracts.py)
2. [final_sft_dataset.py](./final_sft_dataset.py)
3. [regenerate_descriptions.py](./regenerate_descriptions.py)

Then continue to:

- [../sft/README.md](../sft/README.md)
