# Shaer RAG Folder Guide

This folder contains the end-to-end Retrieval-Augmented Generation (RAG) stack for Shaer, driven by the Hugging Face dataset `Shaer-AI/ashaar-full-desc-preprocessed`. It cleans poem descriptions, loads metadata into Neo4j, builds a Chroma vector index over good descriptions, and exposes simple CLIs to build and query.

## Layout
- `rag.md` — specification of the dataset, cleaning rules, graph/vector design.
- `shaer_rag/` — pipeline code:
  - `settings.py` — loads config from `.env` (HF token, Neo4j, Chroma mode/host/port, thresholds).
  - `dataset.py` — load HF dataset, clean, save/load parquet.
  - `cleaning.py` — flags placeholders, low-Arabic-ratio text, verse-duplicate descriptions, boilerplate trimming.
  - `graph.py` — upserts Poem/Poet/Era/Meter/Theme/SourceSite nodes + relationships into Neo4j (constraints included).
  - `vector.py` — Chroma indexer/query helper (SentenceTransformer embeddings).
  - `pipeline.py` — orchestrates cleaning → Neo4j → Chroma.
  - `utils.py` — batching, URL domain extraction, Arabic normalization helpers.
- `scripts/`
  - `build_pipeline.py` — run the full pipeline from CLI.
  - `query_chroma.py` — ad-hoc semantic search against the Chroma collection.
- `docker-compose.yml` — spins up Neo4j + Chroma (volumes under `artifacts/`).
- `tests/test_cleaning.py` — unit tests for cleaning heuristics.
- `artifacts/` (gitignored) — cleaned parquet, Chroma store, Neo4j volumes.
- `nano_graphrag/` — optional graph-RAG toolkit (left intact, not wired into the new pipeline).

## Environment (.env)
Key variables:
```
HF_TOKEN=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DB=neo4j

# Chroma: use persistent mode by default (local JSON store under artifacts/chroma)
CHROMA_MODE=persistent
CHROMA_RESET=0   # set to 1 only if you want to wipe/rebuild the local Chroma store
# If you want to use the Chroma container: set CHROMA_MODE=http and CHROMA_HOST/CHROMA_PORT accordingly.
```

## Quickstart (Windows, from `RAG/`)
1) Activate venv and install deps:
```
..\ .venv\Scripts\activate
pip install -r requirements-dev.txt
```
2) Start services (Neo4j + Chroma, optional if you stay in persistent mode):
```
docker compose up -d
```
3) Build the pipeline (example with a subset for speed):
```
python scripts/build_pipeline.py --limit 2000
```
- Loads HF, cleans descriptions, writes Neo4j, builds Chroma.
- Outputs a summary (counts and artifact path).
4) Query Chroma:
```
python scripts/query_chroma.py --text "قصيدة عن الشوق في العصر العباسي" --top-k 5
```
5) Inspect Neo4j (browser): http://localhost:7474
- Example Cypher checks:
  - `MATCH (p:Poem) RETURN count(p);`
  - `MATCH (p:Poem {has_bad_description:false}) RETURN count(p);`
  - `MATCH (p:Poem)-[:WRITTEN_BY]->(poet) RETURN poet.name, count(*) AS n ORDER BY n DESC LIMIT 5;`
6) Tests:
```
python -m pytest tests/test_cleaning.py
```

## Cleaning rules (from `cleaning.py`)
- Strip whitespace/boilerplate prefixes, normalize Arabic.
- Flag bad descriptions if:
  - placeholder text (e.g., “لا أعرف”, “غير معروف”),
  - low Arabic-character ratio (configurable),
  - description duplicates poem verses (fuzzy match),
  - empty after cleaning.
- `has_bad_description=True` keeps the poem in Neo4j but excludes it from Chroma.
- `needs_resummarization=True` when descriptions are excessively long (configurable).

## Current state (latest run)
- Ingested 2,000 rows (with `--limit 2000`), all marked as good; Neo4j has 2,000 Poem nodes; Chroma has 2,000 vectors.
- To scale to full ~118k, rerun `scripts/build_pipeline.py` without `--limit` (will take longer and require more resources).

## Notes/Tips
- Leave `CHROMA_RESET=0` when querying so the collection is not wiped on client init; set to `1` only before a rebuild.
- If you switch to the Chroma container, set `CHROMA_MODE=http` and ensure the container is up (`docker compose up -d`).
- All artifacts live under `artifacts/` (gitignored) so rebuilds don’t dirty git.
