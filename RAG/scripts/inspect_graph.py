from __future__ import annotations

"""
Lightweight interactive CLI to inspect the Neo4j graph built by the RAG pipeline.

Examples:
  python scripts/inspect_graph.py --counts
  python scripts/inspect_graph.py --list Poem --limit 5
  python scripts/inspect_graph.py --poem-id 12345
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase

# Ensure RAG package is importable when run directly.
RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from shaer_rag.settings import Settings


def _connect(settings: Settings):
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    return driver


def print_counts(session) -> None:
    labels = ["Poem", "Poet", "Meter", "Era", "Theme", "SourceSite"]
    for label in labels:
        res = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
        count = res["c"] if res else 0
        print(f"{label:12}: {count}")

    # Relationship counts
    rels = [
        ("Poem", "WRITTEN_BY", "Poet"),
        ("Poem", "IN_METER", "Meter"),
        ("Poem", "IN_ERA", "Era"),
        ("Poem", "HAS_THEME", "Theme"),
        ("Poem", "FROM_SITE", "SourceSite"),
    ]
    for src, rel, dst in rels:
        res = session.run(
            f"MATCH (:{src})-[:{rel}]->(:{dst}) RETURN count(*) AS c"
        ).single()
        count = res["c"] if res else 0
        print(f"{src}-{rel}->{dst}: {count}")


def list_nodes(session, label: str, limit: int) -> None:
    query = f"MATCH (n:{label}) RETURN n LIMIT $limit"
    result = session.run(query, limit=limit)
    rows = [record["n"] for record in result]
    for idx, row in enumerate(rows, 1):
        print(f"[{idx}] {row}")
    if not rows:
        print(f"No nodes found for label '{label}'.")


def show_poem(session, poem_id: str) -> None:
    query = """
    MATCH (p:Poem {poem_id: $poem_id})
    OPTIONAL MATCH (p)-[:WRITTEN_BY]->(poet:Poet)
    OPTIONAL MATCH (p)-[:IN_METER]->(m:Meter)
    OPTIONAL MATCH (p)-[:HAS_THEME]->(t:Theme)
    OPTIONAL MATCH (p)-[:IN_ERA]->(e:Era)
    RETURN p, poet, m, t, e
    """
    record = session.run(query, poem_id=poem_id).single()
    if not record:
        print(f"Poem {poem_id} not found.")
        return

    poem = record["p"]
    poet = record["poet"]
    meter = record["m"]
    theme = record["t"]
    era = record["e"]

    print("Poem:")
    for key, val in poem.items():
        print(f"  {key}: {val}")
    print(f"Poet: {poet['name'] if poet else 'n/a'}")
    print(f"Meter: {meter['name'] if meter else 'n/a'}")
    print(f"Theme: {theme['name'] if theme else 'n/a'}")
    print(f"Era: {era['name'] if era else 'n/a'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Shaer Neo4j graph.")
    parser.add_argument("--counts", action="store_true", help="Print counts of nodes and key relationships.")
    parser.add_argument("--list", dest="label", help="List nodes for a given label (e.g., Poem, Poet).")
    parser.add_argument("--limit", type=int, default=5, help="Limit for --list (default: 5).")
    parser.add_argument("--poem-id", dest="poem_id", help="Show a single poem by poem_id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()

    if not (args.counts or args.label or args.poem_id):
        print("Nothing to do. Use --counts, --list <Label>, or --poem-id <id>.")
        return

    driver = _connect(settings)
    try:
        with driver.session(database=settings.neo4j_database) as session:
            if args.counts:
                print_counts(session)
            if args.label:
                list_nodes(session, args.label, args.limit)
            if args.poem_id:
                show_poem(session, args.poem_id)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
