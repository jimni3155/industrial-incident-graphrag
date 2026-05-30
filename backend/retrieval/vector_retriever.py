from __future__ import annotations

import re
import logging
from typing import Any

from neo4j import Session
from openai import OpenAI

from core.config import settings
from common.embedding_client import get_embedding_client as _get_emb_client

logger = logging.getLogger(__name__)

_W_VECTOR   = 0.5
_W_SEVERITY = 0.3
_W_GRAPH    = 0.2

_SEVERITY_SCORE = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}

_MANUAL_KEYWORDS   = {"fix", "repair", "procedure", "maintenance", "action", "how to",
                       "step", "recommend", "what should", "inspect", "replace", "check"}
_INCIDENT_KEYWORDS = {"incident", "occurred", "failure", "fault", "history",
                       "similar", "past", "before", "previous", "case"}


_embedding_client: OpenAI | None = None


def _get_embedding_client() -> OpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = _get_emb_client()
    return _embedding_client


def get_embedding(client: OpenAI, text: str) -> list[float]:
    response = _get_embedding_client().embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text.strip(),
    )
    return response.data[0].embedding


def _route_query(query: str) -> tuple[bool, bool]:
    """
    Route the query to manual and/or incident retrieval.

    Returns:
        (search_manuals, search_incidents)
    """
    q      = query.lower()
    tokens = set(re.findall(r"[a-z]+", q))

    is_manual_query   = bool(tokens & _MANUAL_KEYWORDS)
    is_incident_query = bool(tokens & _INCIDENT_KEYWORDS)

    if is_manual_query and not is_incident_query:
        return True, False
    if is_incident_query and not is_manual_query:
        return False, True
    return True, True



def _rerank_manuals(
    results:           list[dict],
    discovered_errors: set[str],
) -> list[dict]:
    for r in results:
        related_errors = r.get("related_errors", [])

        if related_errors and discovered_errors:
            overlap     = set(related_errors) & discovered_errors
            graph_score = len(overlap) / len(related_errors)
        else:
            graph_score = 0.0

        r["graph_score"]    = round(graph_score, 4)
        r["severity_score"] = 0.0
        r["final_score"]    = round(
            _W_VECTOR * r["score"] + _W_GRAPH * graph_score, 4,
        )

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


def _rerank_incidents(
    results:           list[dict],
    discovered_errors: set[str],
) -> list[dict]:
    for r in results:
        severity_score = _SEVERITY_SCORE.get(r.get("severity", "").lower(), 0.0)
        graph_score    = 1.0 if r.get("error_code") in discovered_errors else 0.0

        r["graph_score"]    = graph_score
        r["severity_score"] = severity_score
        r["final_score"]    = round(
            _W_VECTOR   * r["score"] +
            _W_SEVERITY * severity_score +
            _W_GRAPH    * graph_score,
            4,
        )

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


def search_similar_manuals(
    session:           Session,
    client:            OpenAI,
    query:             str,
    top_k:             int      = 5,
    min_score:         float    = 0.70,
    discovered_errors: set[str] | None = None,
) -> list[dict[str, Any]]:
    embedding = get_embedding(client, query)

    rows = session.run(
        """
        CALL db.index.vector.queryNodes('manual_embedding', $top_k, $embedding)
        YIELD node AS m, score
        WHERE score >= $min_score
        RETURN
            m.section_id     AS section_id,
            m.title          AS title,
            m.embedding_text AS embedding_text,
            m.related_errors AS related_errors,
            score
        ORDER BY score DESC
        """,
        top_k     = top_k,
        embedding = embedding,
        min_score = min_score,
    ).data()

    results = [
        {
            "section_id":     r["section_id"],
            "title":          r["title"],
            "embedding_text": r["embedding_text"],
            "related_errors": r["related_errors"] or [],
            "score":          round(r["score"], 4),
            "source":         "vector",
        }
        for r in rows
    ]

    return _rerank_manuals(results, discovered_errors or set())


def search_similar_incidents(
    session:           Session,
    client:            OpenAI,
    query:             str,
    top_k:             int      = 5,
    min_score:         float    = 0.70,
    discovered_errors: set[str] | None = None,
) -> list[dict[str, Any]]:
    embedding = get_embedding(client, query)

    rows = session.run(
        """
        CALL db.index.vector.queryNodes('incident_embedding', $top_k, $embedding)
        YIELD node AS i, score
        WHERE score >= $min_score
        RETURN
            i.incident_id    AS incident_id,
            i.title          AS title,
            i.error_code     AS error_code,
            i.severity       AS severity,
            i.embedding_text AS embedding_text,
            score
        ORDER BY score DESC
        """,
        top_k     = top_k,
        embedding = embedding,
        min_score = min_score,
    ).data()

    results = [
        {
            "incident_id":    r["incident_id"],
            "title":          r["title"],
            "error_code":     r["error_code"],
            "severity":       r["severity"],
            "embedding_text": r["embedding_text"],
            "score":          round(r["score"], 4),
            "source":         "vector",
        }
        for r in rows
    ]

    return _rerank_incidents(results, discovered_errors or set())


def search_similar_images(
    session:           Session,
    top_k:             int      = 3,
    min_score:         float    = 0.20,
    query:             str      = "",
    discovered_errors: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not query:
        return []

    try:
        from common.clip_client import get_clip_text_embedding
        embedding = get_clip_text_embedding(query)
    except Exception as e:
        logger.exception("CLIP text embedding failed: %s", e)
        return []

    rows = session.run(
        """
        CALL db.index.vector.queryNodes('image_clip_text_embedding', $top_k, $embedding)
        YIELD node AS img, score
        WHERE score >= $min_score
        RETURN
            img.image_id            AS image_id,
            img.category            AS category,
            img.clip_caption        AS caption,
            img.related_error_codes AS related_error_codes,
            img.severity            AS severity,
            score
        ORDER BY score DESC
        """,
        top_k     = top_k,
        embedding = embedding,
        min_score = min_score,
    ).data()

    def _ext(category: str) -> str:
        return "png" if category in ("sensor_graphs", "dashboards") else "jpg"

    results = [
        {
            "image_id":            r["image_id"],
            "category":            r["category"],
            "caption":             r["caption"],
            "related_error_codes": r["related_error_codes"] or [],
            "severity":            r.get("severity") or "medium",
            "file_url":            f"/images/{r['category']}/{r['image_id']}.{_ext(r['category'])}",
            "score":               round(r["score"], 4),
            "source":              "clip",
        }
        for r in rows
    ]

    discovered_errors = discovered_errors or set()
    for r in results:
        error_match     = len(set(r["related_error_codes"]) & discovered_errors)
        graph_score     = min(1.0, error_match * 0.3)
        r["graph_score"]  = graph_score
        r["final_score"]  = round(0.7 * r["score"] + 0.3 * graph_score, 4)

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


# main entry

def run_vector_retrieval(
    session:           Session,
    client:            OpenAI,
    query:             str,
    discovered_errors: set[str] | None = None,
    top_k:             int             = 5,
    min_score:         float           = 0.70,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Run query routing, vector search, and graph-aware reranking.

    Returns:
        (vector_manuals, vector_incidents, visual_matches)
    """
    search_manuals, search_incidents = _route_query(query)
    discovered_errors = discovered_errors or set()

    vector_manuals:   list[dict] = []
    vector_incidents: list[dict] = []
    visual_matches:   list[dict] = []

    if search_manuals:
        vector_manuals = search_similar_manuals(
            session, client, query,
            top_k             = top_k,
            min_score         = min_score,
            discovered_errors = discovered_errors,
        )

    if search_incidents:
        vector_incidents = search_similar_incidents(
            session, client, query,
            top_k             = top_k,
            min_score         = min_score,
            discovered_errors = discovered_errors,
        )

    visual_matches = search_similar_images(
        session,
        top_k             = 3,
        min_score         = 0.20,
        query             = query,
        discovered_errors = discovered_errors,
    )

    return vector_manuals, vector_incidents, visual_matches