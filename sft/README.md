# SFT

This folder contains the supervised fine-tuning code and configuration for the paper-faithful Shaer baseline.

## Final baseline

- base model: `Navid-AI/Yehia-7B-preview`
- adapter repo: `Shaer-AI/Shaer-adapters`
- training dataset: `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

## Main files

- [train_sft.py](./train_sft.py)
- [sft_config.yaml](./sft_config.yaml)
- [split_utils.py](./split_utils.py)
- [best_probe_meter_keeper.py](./best_probe_meter_keeper.py)
- [publish_sft_stratified_splits.py](./publish_sft_stratified_splits.py)
- [publish_adapter_model_cards.py](./publish_adapter_model_cards.py)

## Training shape

The final SFT setup uses:

- QLoRA over all linear layers
- completion-only supervised fine-tuning
- weighted train sampling by meter/form/length bucket
- deterministic train/eval/test splits

This folder preserves the code used to train and publish the SFT baseline. Generated run outputs are intentionally not part of the public repo history.
