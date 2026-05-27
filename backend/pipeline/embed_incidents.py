from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

from common.embedding_client import get_embedding_client
from core.config import settings


_INCIDENTS_PATH   = Path("data/processed/incidents.json")


def build_incident_text(inc: dict) -> str:
    """
    Serialize structured Incident fields into a retrieval-friendly text chunk.

    Sensor values are included to preserve failure-state patterns that may not
    appear clearly in the raw description.
    """
    failure_types = ", ".join(inc.get("failure_types", []))
    components    = ", ".join(inc.get("affected_components", []))
    symptoms      = ", ".join(inc.get("symptoms", []))

    sv = inc.get("sensor_values", {})
    sensor_str = (
        f"air={sv.get('air_temperature_c')}°C, "
        f"process={sv.get('process_temperature_c')}°C, "
        f"rpm={sv.get('rotational_speed_rpm')}, "
        f"torque={sv.get('torque_nm')} Nm, "
        f"tool_wear={sv.get('tool_wear_min')} min"
    ) if sv else ""

    return (
        f"{inc['title']}. "
        f"Failure type: {failure_types}. "
        f"Error code: {inc.get('error_code', '')}. "
        f"Severity: {inc.get('severity', '')}. "
        f"Affected components: {components}. "
        f"Symptoms: {symptoms}. "
        f"Sensor values at failure: {sensor_str}. "
        f"{inc.get('description', '')}"
    )


def embed_incidents() -> None:
    client = get_embedding_client()
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )

    incidents = json.loads(_INCIDENTS_PATH.read_text(encoding="utf-8"))
    now       = datetime.now(timezone.utc).isoformat()

    print(f"Embedding {len(incidents)} incidents...")

    with driver.session() as session:
        for inc in incidents:
            iid  = inc["id"]
            text = build_incident_text(inc)

            response  = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=text.strip())
            embedding = response.data[0].embedding

            session.run(
                """
                MATCH (i:Incident {incident_id: $iid})
                SET
                    i.embedding       = $embedding,
                    i.embedding_text  = $text,
                    i.embedding_model = $model,
                    i.embedded_at     = $now
                """,
                iid=iid,
                embedding=embedding,
                text=text,
                model=settings.EMBEDDING_MODEL,
                now=now,
            )
            print(f"  ✓ {iid}: {inc['title'][:60]}")

    # Create vector index if it does not already exist.
    with driver.session() as session:
        session.run(
            """
            CREATE VECTOR INDEX incident_embedding IF NOT EXISTS
            FOR (i:Incident) ON (i.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )
        print("Vector index 'incident_embedding' ready.")

    driver.close()
    print("Incident embedding pipeline completed.")


if __name__ == "__main__":
    embed_incidents()
