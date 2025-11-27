
## RAG Service

FastAPI wrapper over Neo4j + Chroma for poem retrieval.

### Endpoints
- `GET /health` – checks Neo4j and Chroma connectivity.
- `GET /search?text=...&top_k=5` – semantic search over poem descriptions (Chroma).
- `GET /poem/{poem_id}` – fetch full poem record from Neo4j.
- `GET /filter?poet_name=&meter=&era=&theme=&limit=` – metadata filters via Neo4j.
- `GET /similar/{poem_id}` – query similar poems using the stored description.

### Env vars
- `NEO4J_URI` (default `bolt://localhost:7687`)
- `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DB`
- `CHROMA_MODE` (`persistent`|`http`, default `persistent`)
- `CHROMA_PERSIST_DIR` (default `RAG/artifacts/chroma`)
- `CHROMA_HOST`, `CHROMA_PORT` (HTTP mode)
- `CHROMA_COLLECTION` (default `shaer_poems`)
- `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL` (for query embeddings)

### Local run
```bash
uvicorn app.main:app --reload --port 8003
```

### Docker
```bash
docker build -t rag-service -f services/rag_service/Dockerfile .
docker run --rm -p 8003:8000 \
  -e NEO4J_URI=bolt://host.docker.internal:7687 \
  -e NEO4J_USER=neo4j -e NEO4J_PASSWORD=... \
  -e CHROMA_MODE=persistent \
  -e CHROMA_PERSIST_DIR=/data/chroma \
  -v $(pwd)/RAG/artifacts/chroma:/data/chroma \
  rag-service
```

### Tests
```bash
pytest services/rag_service/tests
```
