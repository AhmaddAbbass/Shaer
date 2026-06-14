# SFT

This folder contains the final supervised fine-tuning code for the paper-reported Shaer baseline.

## Released baseline

- Base model: [`Navid-AI/Yehia-7B-preview`](https://huggingface.co/Navid-AI/Yehia-7B-preview)
- Released adapter: [`Shaer-AI/Shaer-adapters`](https://huggingface.co/Shaer-AI/Shaer-adapters)
- Training split dataset: [`Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`](https://huggingface.co/datasets/Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits)

## Exact files

Main training files:

- [train_sft.py](./train_sft.py)
- [sft_config.yaml](./sft_config.yaml)
- [split_utils.py](./split_utils.py)

Supporting utilities:

- [best_probe_meter_keeper.py](./best_probe_meter_keeper.py)
- [publish_sft_stratified_splits.py](./publish_sft_stratified_splits.py)
- [publish_adapter_model_cards.py](./publish_adapter_model_cards.py)
- [../train_sft_qlora.py](../train_sft_qlora.py)

## Paper-facing training setup

The final SFT run uses:

- QLoRA with 4-bit quantization
- completion-only loss
- LoRA over all linear layers
- deterministic `94/3/3` splits
- train-only weighted sampling by:
  - `base_meter || form || length_bucket`

The released adapter is the SFT baseline reported in the paper. GRPO experiments are separate follow-up work and are not the core reported method.

## Prompt contract

The final prompt contract used for SFT is defined in:

- [../description_generation/prompt_contracts.py](../description_generation/prompt_contracts.py)

Relevant builder:

- `build_sft_prompt(...)`

## Typical usage

Train:

```bash
python sft/train_sft.py --config sft/sft_config.yaml
```

Top-level wrapper:

```bash
python train_sft_qlora.py
```

## Scope

This folder preserves the code used to train and publish the released SFT baseline. Runtime outputs and checkpoints are intentionally excluded from the public repository history.
