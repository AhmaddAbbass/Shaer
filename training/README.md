# GRPO notebook setup

- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps (CUDA wheels index is baked in): `pip install --upgrade pip && pip install -r training/requirements.txt`
- Register the kernel: `python -m ipykernel install --user --name shaer-grpo --display-name "shaer-grpo"`
- Run Jupyter with the new kernel selected. The notebook defaults to `CUDA_VISIBLE_DEVICES=1`; override before launch if you need another GPU.
- Export your HF token: `export HF_TOKEN=xxxx`

Notes:
- `meter_reward` now forces CPU and the saved BiLSTM `.keras` file was patched for Keras 3 compatibility (backup at `poem_meter_bilstm.keras.bak`).
- `meaning_reward` uses vLLM; adjust `YEHIA_VLLM_GPU_UTIL` if you need to lower GPU usage (defaults to 0.2).
