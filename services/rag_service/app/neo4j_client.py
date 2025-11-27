
from __future__ import annotations

from typing import List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from services.common_schemas.schemas import RagPoemRecord, RagSearchHit, RagSearchResponse

from .config import Settings
from .logging import get_logger


class Neo4jClient:
    """
    Minimal Neo4j wrapper for poem retrieval and filtering.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("rag_service.neo4j")
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def ping(self) -> bool:
        try:
            with self.driver.session(database=self.settings.neo4j_db) as session:
                session.run("RETURN 1 AS ok").single()
            return True
        except Exception:
            return False

    def get_poem(self, poem_id: str) -> Optional[RagPoemRecord]:
        query = """
        MATCH (p:Poem {poem_id: $poem_id})
        OPTIONAL MATCH (p)-[:WRITTEN_BY]->(poet:Poet)
        OPTIONAL MATCH (p)-[:OF_METER]->(m:Meter)
        OPTIONAL MATCH (p)-[:OF_THEME]->(t:Theme)
        OPTIONAL MATCH (p)-[:OF_ERA]->(e:Era)
        RETURN p, poet.name AS poet_name, m.name AS poem_meter, t.name AS poem_theme, e.name AS poem_era
        """
        with self.driver.session(database=self.settings.neo4j_db) as session:
            record = session.run(query, poem_id=poem_id).single()
            if not record:
                return None

            poem = record["p"]
            verses = poem.get("poem_verses") or poem.get("verses") or []
            if isinstance(verses, str):
                verses = [v.strip() for v in verses.split("\n") if v.strip()]

            num_verses = poem.get("num_verses") or len(verses) or 0

            return RagPoemRecord(
                poem_id=str(poem.get("poem_id", poem_id)),
                poem_title=poem.get("poem_title"),
                poet_name=record.get("poet_name") or poem.get("poet_name"),
                poem_meter=record.get("poem_meter") or poem.get("poem_meter"),
                poem_era=record.get("poem_era") or poem.get("poet_era"),
                poem_theme=record.get("poem_theme") or poem.get("poem_theme"),
                num_verses=int(num_verses),
                verses=verses,
                source_url=poem.get("poem_url"),
            )

    def filter_poems(
        self,
        *,
        poet_name: Optional[str] = None,
        meter: Optional[str] = None,
        era: Optional[str] = None,
        theme: Optional[str] = None,
        limit: int = 20,
    ) -> RagSearchResponse:
        query = """
        MATCH (p:Poem)
        WHERE ($poet_name IS NULL OR toLower(p.poet_name) CONTAINS toLower($poet_name))
          AND ($meter IS NULL OR toLower(p.poem_meter) CONTAINS toLower($meter))
          AND ($era IS NULL OR toLower(p.poet_era) CONTAINS toLower($era))
          AND ($theme IS NULL OR toLower(p.poem_theme) CONTAINS toLower($theme))
        RETURN p
        LIMIT $limit
        """

        with self.driver.session(database=self.settings.neo4j_db) as session:
            result = session.run(
                query,
                poet_name=poet_name,
                meter=meter,
                era=era,
                theme=theme,
                limit=limit,
            )
            hits: List[RagSearchHit] = []
            for record in result:
                poem = record["p"]
                hits.append(
                    RagSearchHit(
                        poem_id=str(poem.get("poem_id")),
                        poem_title=poem.get("poem_title"),
                        poet_name=poem.get("poet_name"),
                        poem_description=poem.get("description_clean")
                        or poem.get("description_raw")
                        or "",
                        poem_meter=poem.get("poem_meter"),
                        poem_era=poem.get("poet_era"),
                        poem_theme=poem.get("poem_theme"),
                        has_bad_description=bool(poem.get("has_bad_description", False)),
                    )
                )

        return RagSearchResponse(hits=hits)
