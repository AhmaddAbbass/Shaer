from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from .cleaning import CleanedPoem, clean_poem_row
from .settings import Settings
from .utils import ensure_dir


def load_raw_dataset(settings: Settings, *, limit: Optional[int] = None):
    kwargs = {}
    if settings.hf_token:
        kwargs["token"] = settings.hf_token
    ds = load_dataset(settings.dataset_id, split=settings.dataset_split, **kwargs)
    if limit is not None:
        ds = ds.select(range(limit))
    return ds


def clean_dataset(settings: Settings, *, limit: Optional[int] = None) -> List[CleanedPoem]:
    dataset = load_raw_dataset(settings, limit=limit)
    cleaned: List[CleanedPoem] = []
    total = len(dataset)
    for row in tqdm(dataset, desc="Cleaning poems", total=total):
        cleaned.append(clean_poem_row(row, settings))
    return cleaned


def to_dataframe(cleaned: Iterable[CleanedPoem]) -> pd.DataFrame:
    rows = [p.as_dict() for p in cleaned]
    return pd.DataFrame(rows)


def save_cleaned_parquet(cleaned: Iterable[CleanedPoem], path) -> None:
    df = to_dataframe(cleaned)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def _to_list(value) -> List:
    """Safely convert parquet-loaded list/array-like values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (np.ndarray, pd.Series)):
        return value.tolist()
    return list(value) if hasattr(value, "__iter__") else []


def load_cleaned_parquet(path) -> List[CleanedPoem]:
    df = pd.read_parquet(path)
    records: List[CleanedPoem] = []
    for _, row in df.iterrows():
        poem_verses = _to_list(row.get("poem_verses"))
        issues = _to_list(row.get("issues"))
        records.append(
            CleanedPoem(
                poem_id=int(row["poem_id"]),
                poem_title=row.get("poem_title"),
                poem_meter=row.get("poem_meter"),
                poem_theme=row.get("poem_theme"),
                poem_url=row.get("poem_url"),
                poet_name=row.get("poet_name"),
                poet_url=row.get("poet_url"),
                poet_description=row.get("poet_description"),
                poet_era=row.get("poet_era"),
                poet_location=row.get("poet_location"),
                poem_language_type=row.get("poem_language_type"),
                poem_verses=poem_verses,
                num_verses=int(row.get("num_verses") or len(poem_verses) or 0),
                description_raw=row.get("description_raw") or "",
                description_clean=row.get("description_clean")
                if pd.notna(row.get("description_clean"))
                else None,
                has_bad_description=bool(row.get("has_bad_description")),
                needs_resummarization=bool(row.get("needs_resummarization")),
                issues=issues,
                source=row.get("source"),
                verse_preview=row.get("verse_preview"),
                poet_key=row.get("poet_key"),
            )
        )
    return records


def filter_good_descriptions(cleaned: Iterable[CleanedPoem]) -> List[CleanedPoem]:
    return [
        poem
        for poem in cleaned
        if poem.description_clean and not poem.has_bad_description
    ]
