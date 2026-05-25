from __future__ import annotations

import re
from typing import Any

from neo4j import GraphDatabase, Session

from core.config import settings
from graph.queries import (
    get_context_by_error_code,
    get_failure_pattern_context,
    get_similar_incidents,
    get_component_incident_chain,
    get_manual_procedures,
)
from graph.context_builder import build_graphrag_context


_ERROR_CODE_RE = re.compile(r"\bE-\d{3}\b")
_FAILURE_TYPE_RE = re.compile(r"\b(TWF|HDF|PWF|OSF|RNF)\b")
_COMPONENT_RE = re.compile(r"\bC-\d{3}\b")
_INCIDENT_RE = re.compile(r"\bINC-[\w-]+\b")

# keyword to error-code fallback when the query does not name a code explicitly
_KEYWORD_TO_CODE: dict[str, str] = {
    "overheat": "E-101",
    "temperature": "E-101",
    "hot": "E-101",
    "pressure": "E-102",
    "hydraulic": "E-102",
    "vibration": "E-103",
    "rpm": "E-103",
    "torque": "E-103",
    "lubrication": "E-104",
    "oil": "E-104",
    "coolant": "E-105",
    "leak": "E-105",
    "sensor": "E-201",
    "power": "E-202",
    "voltage": "E-202",
    "current": "E-202",
    "conveyor": "E-203",
    "jam": "E-203",
    "bearing": "E-204",
    "wear": "E-204",
    "corrosion": "E-205",
    "rust": "E-205",
}


def _extract_seeds(query: str) -> dict[str, list[str]]:
    q = query.upper()
    return {
        "error_codes": _ERROR_CODE_RE.findall(q),
        "failure_types": _FAILURE_TYPE_RE.findall(q),
        "component_ids": _COMPONENT_RE.findall(q),
        "incident_ids": _INCIDENT_RE.findall(query),
    }


def _infer_error_codes(query: str) -> list[str]:
    lower = query.lower()
    found = []
    for keyword, code in _KEYWORD_TO_CODE.items():
        if keyword in lower and code not in found:
            found.append(code)
    return found


class RetrievalService:
    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "RetrievalService":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def retrieve(self, query: str, max_evidence: int = 5) -> dict[str, Any]:
        seeds = _extract_seeds(query)

        error_codes = seeds["error_codes"]
        failure_types = seeds["failure_types"]
        component_ids = seeds["component_ids"]
        incident_ids = seeds["incident_ids"]

        if not error_codes and not failure_types:
            inferred = _infer_error_codes(query)
            error_codes = inferred
            seeds["inferred_error_codes"] = inferred

        with self._driver.session() as session:
            contexts = []

            for code in error_codes:
                ctx = get_context_by_error_code(session, code)
                if ctx:
                    contexts.append(
                        {
                            "seed": code,
                            "seed_type": "error_code",
                            "context": ctx,
                            "formatted": build_graphrag_context(ctx, max_evidence),
                        }
                    )

            for ft in failure_types:
                ctx = get_failure_pattern_context(session, ft)
                if ctx:
                    contexts.append(
                        {
                            "seed": ft,
                            "seed_type": "failure_type",
                            "context": ctx,
                            "formatted": build_graphrag_context(ctx, max_evidence),
                        }
                    )

            for cid in component_ids:
                ctx = get_component_incident_chain(session, cid)
                if ctx:
                    contexts.append(
                        {
                            "seed": cid,
                            "seed_type": "component",
                            "context": ctx,
                            "formatted": build_graphrag_context(ctx, max_evidence),
                        }
                    )

            for iid in incident_ids:
                ctx = get_similar_incidents(session, iid)
                if ctx:
                    contexts.append(
                        {
                            "seed": iid,
                            "seed_type": "incident",
                            "context": ctx,
                            "formatted": build_graphrag_context(ctx, max_evidence),
                        }
                    )

        return {
            "query": query,
            "seeds": seeds,
            "contexts": contexts,
            "combined": _combine_formatted(contexts),
        }

    def retrieve_by_error_code(self, code: str, max_evidence: int = 5) -> str:
        with self._driver.session() as session:
            ctx = get_context_by_error_code(session, code)
        return build_graphrag_context(ctx, max_evidence)

    def retrieve_by_failure_type(self, failure_type: str, max_evidence: int = 5) -> str:
        with self._driver.session() as session:
            ctx = get_failure_pattern_context(session, failure_type)
        return build_graphrag_context(ctx, max_evidence)

    def retrieve_similar(self, incident_id: str) -> dict[str, Any]:
        with self._driver.session() as session:
            return get_similar_incidents(session, incident_id)

    def retrieve_procedures(self, error_code: str) -> list[dict]:
        with self._driver.session() as session:
            return get_manual_procedures(session, error_code)


def _combine_formatted(contexts: list[dict]) -> str:
    if not contexts:
        return ""
    parts = []
    for i, c in enumerate(contexts, 1):
        parts.append(f"=== Context {i} (seed: {c['seed']}) ===")
        parts.append(c["formatted"])
    return "\n\n".join(parts)
