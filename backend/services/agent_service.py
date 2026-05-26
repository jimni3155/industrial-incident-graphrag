from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from core.config import settings
from common.llm_client import get_client
from agent.planner import plan
from agent.explorer import explore, ToolCall, EvidenceState
from agent.validator import validate
from agent.synthesizer import synthesize


class AgentService:
    def __init__(self) -> None:
        self._client = get_client()
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "AgentService":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def investigate(
        self,
        query:                 str,
        max_iterations:        int   = 5,
        use_reflection:        bool  = True,
        sufficiency_threshold: float = 0.55,
    ) -> dict[str, Any]:
        with self._driver.session() as session:
            investigation_plan = plan(self._client, query, session=session)
            calls, state = explore(
                session,
                self._client,
                query,
                investigation_plan,
                max_iterations,
                use_reflection=use_reflection,
                sufficiency_threshold=sufficiency_threshold,
            )

        validation = validate(self._client, query, calls, state=state)
        answer     = synthesize(self._client, query, investigation_plan, calls, validation, state=state)

        return {
            "query":      query,
            "plan":       investigation_plan,
            "tool_trace": [c.to_trace() for c in calls],
            "evidence_state": {
                "discovered_error_codes": sorted(state.discovered_error_codes),
                "discovered_components":  sorted(state.discovered_components),
                "discovered_incidents":   sorted(state.discovered_incidents),
                "discovered_manuals":     sorted(state.discovered_manuals),
                "reasoning_path_count":   len(state.reasoning_paths),
                "confidence":             round(state.confidence, 3),
                "contradictions":         state.contradictions,
            },
            "validation": validation,
            "answer":     answer,
        }