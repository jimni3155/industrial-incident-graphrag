from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

from core.config import settings
from common.clip_client import get_clip_image_embedding, get_clip_text_embedding


_IMAGE_METADATA_PATH = Path("data/processed/image_metadata.json")
_CLIP_MODEL_NAME     = "openai/clip-vit-base-patch32"
_CLIP_DIM            = 512


def build_image_caption(image: dict) -> str:
    analysis = image.get("analysis", {})
    category = image.get("category", "")
    parts    = []

    if defect := analysis.get("visual_defect"):
        parts.append(defect)
    if component := (analysis.get("detected_component") or analysis.get("depicted_component")):
        parts.append(component)
    if symptoms := analysis.get("visible_symptoms"):
        parts.extend(symptoms[:2])
    if desc := analysis.get("description"):
        parts.append(desc[:150])
    if category == "diagrams":
        if dtype := analysis.get("diagram_type"):
            parts.append(dtype)
        if flow := analysis.get("process_flow"):
            parts.append(flow[:100])

    error_codes = image.get("graph_links", {}).get("related_error_codes", [])
    if error_codes:
        parts.append(f"related errors: {', '.join(error_codes)}")

    return ". ".join(p.strip() for p in parts if p.strip())


def embed_images() -> None:
    driver  = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    images  = json.loads(_IMAGE_METADATA_PATH.read_text(encoding="utf-8"))
    now     = datetime.now(timezone.utc).isoformat()
    success = 0
    failed  = 0

    print(f"Starting CLIP embedding for {len(images)} images\n")

    with driver.session() as session:
        for image in images:
            image_id  = image["image_id"]
            file_path = image["file_path"]
            caption   = build_image_caption(image)

            try:
                image_embedding = get_clip_image_embedding(file_path)
                text_embedding  = get_clip_text_embedding(caption)

                session.run(
                    """
                    MATCH (img:Image {image_id: $image_id})
                    SET
                        img.clip_embedding      = $image_embedding,
                        img.clip_text_embedding = $text_embedding,
                        img.clip_caption        = $caption,
                        img.clip_model          = $model_name,
                        img.clip_embedded_at    = $now
                    """,
                    image_id        = image_id,
                    image_embedding = image_embedding,
                    text_embedding  = text_embedding,
                    caption         = caption,
                    model_name      = _CLIP_MODEL_NAME,
                    now             = now,
                )
                print(f"  ✓ {image_id}")
                success += 1

            except FileNotFoundError:
                print(f"  ✗ {image_id} — file not found: {file_path}")
                failed += 1
            except Exception as e:
                print(f"  ✗ {image_id} — error: {e}")
                failed += 1

    with driver.session() as session:
        session.run(
            """
            CREATE VECTOR INDEX image_clip_embedding IF NOT EXISTS
            FOR (img:Image) ON (img.clip_embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )
        session.run(
            """
            CREATE VECTOR INDEX image_clip_text_embedding IF NOT EXISTS
            FOR (img:Image) ON (img.clip_text_embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )
        print("\nVector index ready.")

    driver.close()
    print(f"\nDone: {success} succeeded / {failed} failed")


if __name__ == "__main__":
    embed_images()