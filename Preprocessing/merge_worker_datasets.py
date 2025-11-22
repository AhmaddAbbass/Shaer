# first_word_eda.py
#
# Run with:
#   python first_word_eda.py
#
# It will create: first_word_frequencies.json in the current directory.

from datasets import load_dataset
from collections import Counter
import json
from huggingface_hub import login
DATASET = "Shaer-AI/ashaar-preprocessed-postworkers"
DESCRIPTION_COLUMN = "poem_description"
OUTPUT_JSON = "first_word_frequencies.json"


def extract_first_word(text):
    if not isinstance(text, str):
        return ""

    # collapse whitespace/newlines
    text = " ".join(text.strip().split())
    if not text:
        return ""

    # just take the first whitespace-separated token
    return text.split(" ", 1)[0]


def main():
    print(f"🔍 Loading dataset: {DATASET}") 
    HF_TOKEN = "" # <-- put your token here

    # -----------------------------
    # 1) Login
    # -----------------------------
    print("🔐 Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    print("✓ Logged in.\n")

    ds = load_dataset(DATASET, split="train")

    counter = Counter()
    total_rows = len(ds)
    used_rows = 0
    skipped_rows = 0

    for row in ds:
        desc = row.get(DESCRIPTION_COLUMN)

        first_word = extract_first_word(desc)
        if not first_word:
            skipped_rows += 1
            continue

        counter[first_word] += 1
        used_rows += 1

    # sort by frequency desc
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    freq_dict = {word: int(count) for word, count in sorted_items}

    print("📊 Stats:")
    print(f"  • Total rows: {total_rows}")
    print(f"  • Rows with a usable first word: {used_rows}")
    print(f"  • Skipped rows (None / empty / non-str): {skipped_rows}")
    print(f"  • Unique first words: {len(freq_dict)}")

    print(f"\n💾 Writing frequencies to {OUTPUT_JSON} ...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(freq_dict, f, ensure_ascii=False, indent=2)

    print("✅ Done. Open the JSON and inspect the top entries manually.")


if __name__ == "__main__":
    main()
