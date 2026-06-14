import json
import os
import re
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
NOTEBOOK_PATH = PROJECT_DIR / "prompt_testing.ipynb"

SOURCE_DATASET = "Shaer-AI/ashaar-v1-base-form-with-descriptions"
DEFAULT_MODEL_ID = "Navid-AI/Yehia-7B-preview"
MAX_TOKENS = 2048


def extract_prompts_from_notebook(path: Path) -> tuple[str, str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    target = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "SYSTEM_PROMPT" in src and "USER_TEMPLATE" in src:
            target = src
            break
    if target is None:
        raise RuntimeError("Could not find SYSTEM_PROMPT and USER_TEMPLATE in notebook.")

    sm = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', target, flags=re.S)
    um = re.search(r'USER_TEMPLATE\s*=\s*"""(.*?)"""', target, flags=re.S)
    if not sm or not um:
        raise RuntimeError("Failed to parse SYSTEM_PROMPT or USER_TEMPLATE from notebook cell.")

    return sm.group(1).strip(), um.group(1).strip()


def build_meter_label(base_meter: str, form: str) -> str:
    if form == "تام":
        return base_meter
    return f"{form} {base_meter}"


def main() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required in .env or environment.")

    source_dataset = os.getenv("DATASET_ID", SOURCE_DATASET)
    model_id = os.getenv("YEHIA_MODEL_ID", DEFAULT_MODEL_ID)
    max_tokens = int(os.getenv("MAX_SFT_TOKENS", str(MAX_TOKENS)))
    max_lines_cap_raw = os.getenv("SFT_MAX_LINES", "").strip()
    max_lines_cap = int(max_lines_cap_raw) if max_lines_cap_raw else None

    system_prompt, user_template = extract_prompts_from_notebook(NOTEBOOK_PATH)

    print(f"Loading source dataset: {source_dataset}")
    ds = load_dataset(source_dataset, split="train", download_mode="force_redownload")
    source_rows = len(ds)
    print(f"Source rows: {source_rows}")

    print(f"Loading tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    def valid_row(ex):
        verses = ex["poem verses"]
        if not isinstance(verses, list) or len(verses) == 0:
            return False
        for x in verses:
            if not isinstance(x, str) or not x.strip():
                return False
        return True

    print("Filtering invalid poem rows...")
    ds_valid = ds.filter(valid_row)
    print(f"Rows after validity filter: {len(ds_valid)}")

    if max_lines_cap is not None:
        print(f"Applying optional line cap: SFT_MAX_LINES={max_lines_cap}")
        ds_valid = ds_valid.filter(lambda ex: len(ex["poem verses"]) <= max_lines_cap)
        print(f"Rows after line cap: {len(ds_valid)}")

    def add_sft_columns(batch):
        prompts = []
        completions = []
        full_texts = []
        n_lines_list = []

        for base_meter, form, desc, verses in zip(
            batch["base_meter"],
            batch["form"],
            batch["description"],
            batch["poem verses"],
        ):
            lines = [x.strip() for x in verses if isinstance(x, str) and x.strip()]
            n_lines = len(lines)
            meter_label = build_meter_label(base_meter, form)

            user = user_template.format(
                base_meter=base_meter,
                form=form,
                meter_label=meter_label,
                description=(desc or "").strip(),
                num_lines=n_lines,
            )

            prompt = f"<s> [INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user.strip()} [/INST]"
            completion = "\n".join(lines).strip()
            full_text = f"{prompt} {completion} </s>"

            prompts.append(prompt)
            completions.append(completion)
            full_texts.append(full_text)
            n_lines_list.append(n_lines)

        return {
            "sft_prompt": prompts,
            "sft_completion": completions,
            "sft_full_text": full_texts,
            "sft_num_lines": n_lines_list,
        }

    print("Adding SFT columns...")
    ds_sft = ds_valid.map(add_sft_columns, batched=True, batch_size=256)

    print("Filtering empty SFT completions (safety)...")
    ds_sft = ds_sft.filter(lambda ex: isinstance(ex["sft_completion"], str) and len(ex["sft_completion"].strip()) > 0)

    def add_token_length(batch):
        enc = tok(batch["sft_full_text"], add_special_tokens=False, return_attention_mask=False)
        return {"sft_total_tokens": [len(ids) for ids in enc["input_ids"]]}

    print("Computing token lengths...")
    ds_len = ds_sft.map(add_token_length, batched=True, batch_size=128)

    total_after_processing = len(ds_len)
    over_limit = sum(1 for x in ds_len["sft_total_tokens"] if x > max_tokens)
    kept = total_after_processing - over_limit

    print(f"Rows after processing: {total_after_processing}")
    print(f"Rows over {max_tokens} tokens: {over_limit}")
    print(f"Rows kept: {kept}")

    print("Applying token-length filter...")
    ds_filtered = ds_len.filter(lambda ex: ex["sft_total_tokens"] <= max_tokens)

    # Validation checks
    required_cols = {"sft_prompt", "sft_completion", "sft_full_text", "sft_num_lines", "sft_total_tokens"}
    missing = required_cols.difference(set(ds_filtered.column_names))
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    if len(ds_filtered) == 0:
        raise RuntimeError("Filtered dataset is empty.")

    max_tokens_seen = max(ds_filtered["sft_total_tokens"])
    if max_tokens_seen > max_tokens:
        raise RuntimeError(f"Token filter failed: max tokens seen {max_tokens_seen} > {max_tokens}")

    # Ensure we stored real newlines for multiline rows
    newline_check_ok = True
    for ex in ds_filtered.select(range(min(1000, len(ds_filtered)))):
        if ex["sft_num_lines"] > 1 and "\n" not in ex["sft_completion"]:
            newline_check_ok = False
            break
    if not newline_check_ok:
        raise RuntimeError("Newline validation failed in sft_completion.")

    print("Validation checks passed.")

    api = HfApi(token=hf_token)
    who = api.whoami(token=hf_token)
    org_names = [o["name"] for o in who.get("orgs", [])]
    namespace = "Shaer-AI" if "Shaer-AI" in org_names else who["name"]

    explicit_repo = os.getenv("OUTPUT_DATASET_ID", "").strip()
    if explicit_repo:
        repo_id = explicit_repo
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        repo_id = f"{namespace}/ashaar-v1-sft-ready-locked-prompt-2048-{stamp}"

    print(f"Creating dataset repo: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token, exist_ok=True)

    print("Pushing dataset...")
    ds_filtered.push_to_hub(repo_id=repo_id, token=hf_token)

    readme = f"""---
language:
- ar
license: apache-2.0
pretty_name: Ashaar v1 SFT Ready Locked Prompt ({datetime.utcnow().date().isoformat()})
task_categories:
- text-generation
size_categories:
- 100K<n<1M
---

# Ashaar v1 SFT-Ready (Locked Prompt, <= {max_tokens} tokens)

This dataset is derived from `{source_dataset}` and prepared for supervised fine-tuning (SFT) with the locked prompt from `prompt_testing.ipynb`.

## Locked Prompt
### SYSTEM_PROMPT
{system_prompt}

### USER_TEMPLATE
{user_template}

## Conditioning Rule
- `meter_label = base_meter` if `form == "تام"`
- else `meter_label = "{{form}} {{base_meter}}"`

## Added Columns
- `sft_prompt`
- `sft_completion`
- `sft_full_text`
- `sft_num_lines`
- `sft_total_tokens`

## Target Formatting
- `sft_completion` is built from `poem verses` using **real newline characters**.

## Filtering
- Valid-row filtering: keep rows where `poem verses` is non-empty list of non-empty strings.
{f"- Optional line cap applied: `sft_num_lines <= {max_lines_cap}`." if max_lines_cap is not None else "- No line-cap filter applied."}
- Token length filtering: keep rows with `sft_total_tokens <= {max_tokens}`.

## Counts
- Source rows: **{source_rows}**
- After row processing: **{total_after_processing}**
- Removed by token filter (`> {max_tokens}`): **{over_limit}**
- Final rows kept: **{kept}**

## Notes
- This is a derived dataset repo; source dataset is unchanged.
"""

    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    print("Done.")
    print(f"DATASET_REPO={repo_id}")
    print(f"DATASET_URL=https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
