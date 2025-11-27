from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from shaer_rag.dataset import load_cleaned_parquet


def test_load_cleaned_parquet_handles_arrays(tmp_path: Path):
    import pandas as pd

    data = {
        "poem_id": [1],
        "poem_title": ["demo"],
        "poem_meter": ["الطويل"],
        "poem_theme": ["الشوق"],
        "poem_url": [None],
        "poet_name": ["شاعر"],
        "poet_url": [None],
        "poet_description": [None],
        "poet_era": ["العصر العباسي"],
        "poet_location": [None],
        "poem_language_type": ["arabic"],
        # Use numpy array to ensure _to_list handles non-list iterables
        "poem_verses": [np.array(["بيت ١", "بيت ٢"])],
        "num_verses": [0],
        "description_raw": ["raw"],
        "description_clean": ["clean"],
        "has_bad_description": [False],
        "needs_resummarization": [False],
        "issues": [np.array(["placeholder"])],
        "source": ["test"],
        "verse_preview": ["بيت ١"],
        "poet_key": ["شاعر|العصر العباسي"],
    }

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path, index=False)

    poems = load_cleaned_parquet(parquet_path)
    assert len(poems) == 1
    poem = poems[0]
    assert poem.poem_verses == ["بيت ١", "بيت ٢"]
    assert poem.num_verses == 2  # derived from verses when num_verses was 0
    assert poem.issues == ["placeholder"]
