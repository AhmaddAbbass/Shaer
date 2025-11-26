# Shaer RAG Workspace Guide

This folder now contains a clean, purpose-built stack for the Shaer poetry RAG described in `rag.md`. Everything revolves around the Hugging Face dataset `Shaer-AI/ashaar-full-desc-preprocessed` and the graph/vector split outlined there.

## What Lives Where
- `shaer_rag/` — pipeline code.
  - `settings.py` — pulls config from `RAG/.env` (HF token, Neo4j/Chroma endpoints, thresholds).
  - `cleaning.py` — description cleaning/quality flags (placeholder detection, Arabic-ratio, verse-duplication check).
  - `dataset.py` — loads HF data, applies cleaning, saves parquet under `artifacts/`.
  - `graph.py` — upserts Poem/Poet/Era/Meter/SourceSite nodes and relationships into Neo4j.
  - `vector.py` — builds a Chroma collection of good descriptions using `SentenceTransformerEmbeddingFunction`.
  - `pipeline.py` — orchestration helper used by the CLI.
- `scripts/`
  - `build_pipeline.py` — end-to-end: load HF → clean → write Neo4j graph → build Chroma vectors (toggleable).
  - `query_chroma.py` — quick semantic search against the built Chroma collection.
- `docker-compose.yml` — spins up Neo4j + Chroma with volumes under `artifacts/`.
- `tests/` — unit tests for the cleaning heuristics.
- `nano_graphrag/` — kept for optional concept/entity graph extraction. Not wired into the new pipeline yet.
- `rag.md` — canonical requirements/spec for the dataset and retrieval strategy.

## Running The Pipeline
1) Install deps (inside the repo venv is recommended):
```
pip install -r requirements-dev.txt
```

2) Start services (Docker Desktop must be running):
```
cd RAG
docker compose up -d
```
- Set `CHROMA_MODE=http` (and optionally `CHROMA_HOST/CHROMA_PORT`) if you want the pipeline to talk to the dockerized Chroma server. Otherwise it will use a local persistent client under `artifacts/chroma`.

3) Build graph + vectors (use `--limit` for dry runs):
```
python scripts/build_pipeline.py --limit 500
```
- Cleans descriptions, writes `artifacts/clean_poems.parquet`.
- Upserts Poem/Poet/Era/Meter/SourceSite into Neo4j (constraints are created automatically).
- Indexes only good descriptions into Chroma; document IDs match `poem_id`.

4) Smoke-test retrieval:
```
python scripts/query_chroma.py --text "قصيدة عن الشوق في العصر العباسي" --top-k 5
```

## Cleaning And Quality Rules (from `cleaning.py`)
- Flags placeholders (`لا أعرف`, `غير معروف`, etc.).
- Drops/flags descriptions with low Arabic character ratio (threshold is configurable, default 0.6).
- Marks descriptions that essentially copy the poem verses (fuzzy match vs. concatenated verses).
- Trims boilerplate prefixes and collapses whitespace.
- `has_bad_description=True` if any issue is found or the cleaned text is empty; these are kept in Neo4j but excluded from Chroma.
- `needs_resummarization=True` when a description is excessively long (default >1200 chars).

## Outputs And State
- `artifacts/clean_poems.parquet` — cleaned dataset cache (reused when `--reuse-cleaned` is set).
- `artifacts/neo4j/` — Neo4j data/log/import volumes (created by Docker).
- `artifacts/chroma/` — Chroma persisted store (or use HTTP mode if you point at a running service).

## Notes
- Environment is read from `RAG/.env`; keep your HF token and OpenAI keys there. Defaults fall back to localhost Neo4j/Chroma if not set.
- `nano_graphrag` remains available for concept/entity extraction; we can hook it up later to enrich Neo4j with `:Concept` nodes once the core pipeline is stable.
- The root-level `requirements.txt` (outside `RAG/`) is untouched and still belongs to the rest of the project.
