from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

from common.embedding_client import get_embedding_client
from core.config import settings


_MANUALS_PATH    = Path("data/processed/manual_sections.json")


def build_manual_text(manual: dict) -> str:
    """
    Serialize ManualSection fields into a retrieval-friendly text chunk.
    related_errors are prepended to improve error-code-aware semantic search.
    """
    errors = ", ".join(manual.get("related_errors", []))
    return (
        f"Manual: {manual['title']}. "
        f"Related errors: {errors}. "
        f"{manual['content']}"
    )


def embed_manuals() -> None:
    client = get_embedding_client()
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )

    manuals = json.loads(_MANUALS_PATH.read_text(encoding="utf-8"))
    now     = datetime.now(timezone.utc).isoformat()

    print(f"Embedding {len(manuals)} manual sections...")

    with driver.session() as session:
        for manual in manuals:
            mid  = manual["id"]
            text = build_manual_text(manual)

            response  = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=text.strip())
            embedding = response.data[0].embedding

            session.run(
                """
                MATCH (m:ManualSection {section_id: $mid})
                SET
                    m.embedding       = $embedding,
                    m.embedding_text  = $text,
                    m.embedding_model = $model,
                    m.embedded_at     = $now
                """,
                mid=mid,
                embedding=embedding,
                text=text,
                model=settings.EMBEDDING_MODEL,
                now=now,
            )
            print(f"  ✓ {mid}: {manual['title']}")

    # Create vector index if it does not already exist.
    with driver.session() as session:
        session.run(
            """
            CREATE VECTOR INDEX manual_embedding IF NOT EXISTS
            FOR (m:ManualSection) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )
        print("Vector index 'manual_embedding' ready.")

    driver.close()
    print("Manual embedding pipeline completed.")


if __name__ == "__main__":
    embed_manuals()
