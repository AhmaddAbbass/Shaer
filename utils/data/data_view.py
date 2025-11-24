# ================================
# EDA for Shaer-AI poem-level data
# ================================

# If needed:
# !pip install datasets huggingface_hub

import random
from collections import Counter

from datasets import load_dataset
from huggingface_hub import login

# ----------------
# 1. HF auth
# ----------------
HF_TOKEN = "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" # e.g. os.environ.get("HF_TOKEN")
  # <-- PUT YOUR TOKEN HERE
login(HF_TOKEN)

# ----------------
# 2. Load dataset
# ----------------
# Adjust this if your dataset id or split is different
DATASET_ID = "Shaer-AI/ashaar-full-desc"
SPLIT = "train"

print(f"Loading dataset: {DATASET_ID} [{SPLIT}]...")
ds = load_dataset(DATASET_ID, split=SPLIT)

print("\n=== DATASET BASIC INFO ===")
print(ds)
print("\nColumns:", ds.column_names)
print("Number of rows:", len(ds))

# ----------------
# 3. Helpers
# ----------------
def is_null_text(x):
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

def is_unknown_desc(text):
    if text is None:
        return False
    t = text.strip()
    patterns = [
        "لا أعلم",
        "لا اعلم",
        "لا أعرف",
        "لا اعرف",
        "لاأعلم",
        "لاأعرف",
    ]
    return any(p in t for p in patterns)

GENERIC_PATTERNS = [
    "هذه القصيدة من نظم",
    "هذه القصيدة من تأليف",
    "هذه القصيدة من كتابة",
    "الشاعر هو",
    "قصيدة من قصائد",
    "من قصائد الشاعر",
    "في هذه القصيدة",
]

def has_generic_intro(text):
    if text is None:
        return False
    t = text.strip()
    return any(p in t for p in GENERIC_PATTERNS)

# ----------------
# 4. Description quality stats
# ----------------
desc_col = "poem_description"

null_desc_count = sum(is_null_text(x) for x in ds[desc_col])
unknown_desc_mask = [is_unknown_desc(x) for x in ds[desc_col]]
unknown_desc_count = sum(unknown_desc_mask)

generic_desc_mask = [has_generic_intro(x) for x in ds[desc_col]]
generic_desc_count = sum(generic_desc_mask)

print("\n=== DESCRIPTION QUALITY STATS ===")
print(f"Total rows: {len(ds)}")
print(f"Null / empty poem_description: {null_desc_count}")
print(f"'لا أعلم' / 'لا أعرف' style descriptions: {unknown_desc_count}")
print(f"Generic / intro-ish descriptions (e.g. 'هذه القصيدة من نظم...'): {generic_desc_count}")

# ----------------
# 5. Basic distributions of key fields
# ----------------
def print_top_counts(column_name, top_n=15):
    values = ds[column_name]
    cnt = Counter(values)
    print(f"\n=== Top {top_n} values for '{column_name}' ===")
    for value, c in cnt.most_common(top_n):
        print(f"{repr(value)}: {c}")

for col in ["poem_meter", "poem_theme", "poet_era", "poem_language_type"]:
    if col in ds.column_names:
        print_top_counts(col, top_n=15)

# ----------------
# 6. Print random examples (full rows)
# ----------------
def print_row(idx, title="Example"):
    row = ds[idx]
    print(f"\n=== {title} (index={idx}) ===")
    for col in ds.column_names:
        val = row[col]
        # Shorten very long strings a bit
        if isinstance(val, str) and len(val) > 400:
            display_val = val[:400] + "... [TRUNCATED]"
        else:
            display_val = val
        print(f"{col}: {repr(display_val)}")

# 6a) Random general examples
num_random_examples = 5
indices = random.sample(range(len(ds)), k=min(num_random_examples, len(ds)))

print("\n=== RANDOM GENERAL EXAMPLES ===")
for i, idx in enumerate(indices, start=1):
    print_row(idx, title=f"Random example #{i}")

# 6b) Examples with null / empty descriptions
print("\n=== EXAMPLES: NULL / EMPTY DESCRIPTIONS ===")
null_indices = [i for i, x in enumerate(ds[desc_col]) if is_null_text(x)]
for i, idx in enumerate(null_indices[:5], start=1):
    print_row(idx, title=f"Null/empty desc example #{i}")
if not null_indices:
    print("No null/empty poem_description rows found.")

# 6c) Examples with 'لا أعلم' style descriptions
print("\n=== EXAMPLES: 'لا أعلم' STYLE DESCRIPTIONS ===")
unknown_indices = [i for i, flag in enumerate(unknown_desc_mask) if flag]
for i, idx in enumerate(unknown_indices[:5], start=1):
    print_row(idx, title=f"'لا أعلم' desc example #{i}")
if not unknown_indices:
    print("No 'لا أعلم' style poem_description rows found.")

# 6d) Examples with generic intros
print("\n=== EXAMPLES: GENERIC / INTRO-ISH DESCRIPTIONS ===")
generic_indices = [i for i, flag in enumerate(generic_desc_mask) if flag]
for i, idx in enumerate(generic_indices[:5], start=1):
    print_row(idx, title=f"Generic desc example #{i}")
if not generic_indices:
    print("No generic/intro-ish poem_description rows found.")

print("\n=== EDA DONE ===")
