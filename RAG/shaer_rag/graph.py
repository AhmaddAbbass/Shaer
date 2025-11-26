from __future__ import annotations

from typing import Iterable, List

from neo4j import GraphDatabase

from .cleaning import CleanedPoem
from .settings import Settings
from .utils import batched


class Neo4jPoemGraph:
    """Handles graph persistence for poems, poets, meters, and eras."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def setup_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT poem_id_unique IF NOT EXISTS FOR (p:Poem) REQUIRE p.poem_id IS UNIQUE",
            "CREATE CONSTRAINT poet_key_unique IF NOT EXISTS FOR (p:Poet) REQUIRE p.poet_key IS UNIQUE",
            "CREATE CONSTRAINT meter_name_unique IF NOT EXISTS FOR (m:Meter) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT era_name_unique IF NOT EXISTS FOR (e:Era) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT source_name_unique IF NOT EXISTS FOR (s:SourceSite) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT theme_name_unique IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE",
        ]
        with self.driver.session(database=self.settings.neo4j_database) as session:
            for stmt in statements:
                session.run(stmt)

    def load(self, poems: Iterable[CleanedPoem], *, batch_size: int = 500) -> None:
        rows = [self._poem_to_row(p) for p in poems]
        query = """
        UNWIND $rows AS row
        MERGE (p:Poem {poem_id: row.poem_id})
        SET
            p.title = row.poem_title,
            p.meter = row.poem_meter,
            p.theme_raw = row.poem_theme,
            p.url = row.poem_url,
            p.num_verses = row.num_verses,
            p.description_raw = row.description_raw,
            p.description_clean = row.description_clean,
            p.has_bad_description = row.has_bad_description,
            p.needs_resummarization = row.needs_resummarization,
            p.source = row.source,
            p.verse_preview = row.verse_preview

        MERGE (meter:Meter {name: row.poem_meter})
        MERGE (p)-[:IN_METER]->(meter)

        FOREACH (_ IN CASE WHEN row.poet_key IS NULL THEN [] ELSE [1] END |
            MERGE (poet:Poet {poet_key: row.poet_key})
            SET poet.name = row.poet_name,
                poet.url = row.poet_url,
                poet.description = row.poet_description,
                poet.era = row.poet_era,
                poet.location = row.poet_location
            MERGE (p)-[:WRITTEN_BY]->(poet)
        )

        FOREACH (_ IN CASE WHEN row.poet_era IS NULL THEN [] ELSE [1] END |
            MERGE (era:Era {name: row.poet_era})
            MERGE (p)-[:IN_ERA]->(era)
        )

        FOREACH (_ IN CASE WHEN row.poem_theme IS NULL THEN [] ELSE [1] END |
            MERGE (theme:Theme {name: row.poem_theme})
            MERGE (p)-[:HAS_THEME]->(theme)
        )

        MERGE (src:SourceSite {name: row.source})
        MERGE (p)-[:FROM_SITE]->(src)
        """

        with self.driver.session(database=self.settings.neo4j_database) as session:
            for batch in batched(rows, batch_size):
                session.run(query, rows=batch)

    @staticmethod
    def _poem_to_row(poem: CleanedPoem) -> dict:
        return {
            "poem_id": int(poem.poem_id),
            "poem_title": poem.poem_title,
            "poem_meter": poem.poem_meter,
            "poem_theme": poem.poem_theme,
            "poem_url": poem.poem_url,
            "num_verses": poem.num_verses,
            "description_raw": poem.description_raw,
            "description_clean": poem.description_clean,
            "has_bad_description": poem.has_bad_description,
            "needs_resummarization": poem.needs_resummarization,
            "source": poem.source,
            "verse_preview": poem.verse_preview[:500] if poem.verse_preview else None,
            "poet_key": poem.poet_key,
            "poet_name": poem.poet_name,
            "poet_url": poem.poet_url,
            "poet_description": poem.poet_description,
            "poet_era": poem.poet_era,
            "poet_location": poem.poet_location,
        }

