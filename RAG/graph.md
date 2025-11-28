# Neo4j Quick Checks (Shaer RAG)

Use these Cypher snippets in Neo4j Browser to inspect what the pipeline loaded.

## Basic introspection
- List labels: `CALL db.labels();`
- List relationship types: `CALL db.relationshipTypes();`
- Show constraints: `SHOW CONSTRAINTS;`
- Show indexes: `SHOW INDEXES;`

## Counts
- All nodes by label:
  ```cypher
  MATCH (n) RETURN labels(n) AS labels, count(*) AS c ORDER BY c DESC;
  ```
- Core labels:
  ```cypher
  MATCH (n:Poem) RETURN count(n) AS poems;
  MATCH (n:Poet) RETURN count(n) AS poets;
  MATCH (n:Meter) RETURN count(n) AS meters;
  MATCH (n:Era) RETURN count(n) AS eras;
  MATCH (n:Theme) RETURN count(n) AS themes;
  MATCH (n:SourceSite) RETURN count(n) AS sources;
  ```
- Core relationships:
  ```cypher
  MATCH (:Poem)-[r:WRITTEN_BY]->(:Poet) RETURN count(r) AS poem_poet;
  MATCH (:Poem)-[r:IN_METER]->(:Meter) RETURN count(r) AS poem_meter;
  MATCH (:Poem)-[r:IN_ERA]->(:Era) RETURN count(r) AS poem_era;
  MATCH (:Poem)-[r:HAS_THEME]->(:Theme) RETURN count(r) AS poem_theme;
  MATCH (:Poem)-[r:FROM_SITE]->(:SourceSite) RETURN count(r) AS poem_source;
  ```

## Sampling data
- Sample poems:
  ```cypher
  MATCH (p:Poem)
  OPTIONAL MATCH (p)-[:WRITTEN_BY]->(poet:Poet)
  OPTIONAL MATCH (p)-[:IN_METER]->(m:Meter)
  RETURN p.poem_id AS id, p.title AS title, poet.name AS poet, m.name AS meter
  LIMIT 10;
  ```
- Find poems missing descriptions:
  ```cypher
  MATCH (p:Poem)
  WHERE coalesce(p.description_clean, "") = ""
  RETURN count(p) AS missing_desc, collect(p.poem_id)[0..10] AS sample_ids;
  ```
- Find poems flagged bad descriptions:
  ```cypher
  MATCH (p:Poem {has_bad_description: true})
  RETURN count(p) AS bad_desc, collect(p.poem_id)[0..10] AS sample_ids;
  ```

## If you see “label does not exist” warnings
That means the database is empty or you are connected to the wrong database. Run the pipeline to populate:
```
cd RAG
python scripts/build_pipeline.py --limit 500   # or drop --limit for full ingest
```
Ensure `NEO4J_URI/USER/PASSWORD/DB` in `RAG/.env` (or environment) point to your running Neo4j instance. In Neo4j Browser, select the same database name shown in `NEO4J_DB`.***
