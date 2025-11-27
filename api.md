# API Surface (Blackbox Overview)

This file documents the stable entrypoints you can import from `api.py` at the repo root. They wrap the RAG internals and provide a minimal, long-lived surface for agents or services.

## Functions

### `build_index(limit=None, reuse_cleaned=False, skip_neo4j=False, skip_chroma=False) -> dict`
- Runs the HF → clean → Neo4j → Chroma pipeline.
- `limit`: cap rows for dry runs (omit for full ingest).
- `reuse_cleaned`: load `artifacts/clean_poems.parquet` if present instead of downloading HF again.
- `skip_neo4j` / `skip_chroma`: disable either target if you only need the other.
- Returns a summary dict (row counts, vectors indexed, parquet path).

### `search_descriptions(query: str, top_k: int = 5) -> list[dict]`
- Semantic search over poem descriptions in Chroma.
- Returns a list of hits with `id` (poem_id), `distance`, `document` (description), and `metadata` (meter, era, poet, etc.).

### `chroma_count() -> int`
- Returns the current number of vectors in the Chroma collection (uses settings from `.env`).

### `neo4j_counts() -> dict`
- Returns `{"poems": <total>, "good_descriptions": <count>}` from Neo4j.
- Requires the Neo4j driver and reachable Neo4j per `.env`.

### `summary() -> dict`
- Combined quick stats; includes Neo4j counts and Chroma vector count (or errors if unreachable).

### `load_cleaned_cache(limit=None, only_good=True) -> list[CleanedPoem]`
- Loads rows from `artifacts/clean_poems.parquet` (cached clean dataset).
- `only_good=True` filters to poems with usable descriptions; `limit` slices for quick inspection.

### `get_poems_by_id(poem_ids: list[int]) -> list[dict]`
- Fetches poem metadata from Neo4j by `poem_id` (title, meter, era, theme, poet, description_clean, etc.).
- Requires Neo4j connectivity.

## Notes
- The API auto-loads `RAG` into `sys.path` so it can be imported from repo root.
- `.env` drives everything (HF_TOKEN, Neo4j URI/auth, Chroma mode/host/port, etc.).
- Chroma “persistent” mode uses `artifacts/chroma`; set `CHROMA_RESET=0` when querying to avoid wiping the index on client init.
