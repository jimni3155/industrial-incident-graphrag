from __future__ import annotations

import json
import re
from typing import Any

from neo4j import Session
from openai import OpenAI

from core.config import settings
from graph.queries import get_graph_preview, format_graph_preview


_TOOL_SCHEMA = """\
Available tools and argument constraints:
- get_context_by_error_code(error_code: str)
    error_code must match pattern E-\\d{3}, e.g. E-101, E-204
    Use when a specific error code is identified or suspected.

- get_failure_pattern_context(failure_type: str)
    failure_type must be one of: TWF, HDF, PWF, OSF, RNF
    Use when the failure mode category is known.

- get_similar_incidents(incident_id: str)
    incident_id must match pattern INC-\\d{4}, e.g. INC-0003
    NEVER pass an error code (E-xxx) as incident_id.
    Only call if a specific incident ID is already known from prior evidence.

- get_manual_procedures(error_code: str)
    error_code must match pattern E-\\d{3}
    Use to retrieve maintenance procedures for a confirmed error code.
    Always include this step for maintenance/repair/action queries.

- get_component_incident_chain(component_id: str)
    component_id must match pattern C-\\d{3}, e.g. C-008
    NEVER pass an error code as component_id.
    Use when a specific component is suspected.
"""

_ERROR_FAILURE_MAP = """\
ErrorCode to FailureType mapping (use this to set suspected_failure_types):
  E-101 -> HDF  (Heat Dissipation Failure)
  E-102 -> PWF  (Power Failure)
  E-103 -> OSF  (Overstrain Failure)
  E-104 -> TWF  (Tool Wear Failure)
  E-105 -> PWF  (Power Failure)
  E-201 -> RNF  (Random Failure)
  E-202 -> PWF  (Power Failure)
  E-203 -> OSF  (Overstrain Failure)
  E-204 -> TWF  (Tool Wear Failure)
  E-205 -> RNF  (Random Failure)
"""

_PLANNER_PROMPT = f"""\
You are an industrial incident investigation planner.
Given a user query about equipment failure or anomaly, produce a structured investigation plan.

{_TOOL_SCHEMA}

{_ERROR_FAILURE_MAP}

Retrieval Strategy:
- LOCAL retrieval: node-level, specific evidence
    tools: get_context_by_error_code, get_component_incident_chain, get_manual_procedures
    → use when query contains a specific error code, component, or asks for procedures

- GLOBAL retrieval: pattern-level, cross-incident aggregated evidence
    tools: get_failure_pattern_context
    → use when query describes a symptom or failure mode without a specific error code

Selection rule:
    specific error code in query  → LOCAL first, then GLOBAL to confirm pattern
    symptom-based query           → GLOBAL first, then LOCAL to narrow down
    maintenance/action query      → LOCAL first (manual procedures required)

Respond ONLY with a valid JSON object:
{{
  "suspected_error_codes": ["E-xxx", ...],
  "suspected_failure_types": ["TWF"|"HDF"|"PWF"|"OSF"|"RNF", ...],
  "suspected_components": ["C-xxx", ...],
  "retrieval_strategy": "local_first"|"global_first",
  "investigation_steps": [
    {{"step": 1, "tool": "<tool_name>", "args": {{}}, "reason": "<why>"}},
    ...
  ],
  "max_iterations": <int 2-5>
}}

Rules:
- investigation_steps: 3-5 steps, ordered by retrieval strategy
- Always include get_context_by_error_code as first LOCAL step
- Always include get_manual_procedures for maintenance/repair/action/procedure queries
- Do not call get_similar_incidents unless you have a confirmed INC-xxxx ID
- Do not invent component_id or incident_id values
- Only use IDs visible in the Graph Preview below
- Use ErrorCode to FailureType mapping above to set suspected_failure_types
"""

_FALLBACK_TOOL_PRIORITY = [
    ("overheat",    "E-101", "HDF"),
    ("heat",        "E-101", "HDF"),
    ("temperature", "E-101", "HDF"),
    ("pressure",    "E-102", "PWF"),
    ("vibration",   "E-103", "OSF"),
    ("crack",       "E-103", "OSF"),
    ("oil",         "E-104", "TWF"),
    ("lubrication", "E-104", "TWF"),
    ("leak",        "E-105", "PWF"),
    ("coolant",     "E-105", "PWF"),
    ("sensor",      "E-201", "RNF"),
    ("power",       "E-202", "PWF"),
    ("conveyor",    "E-203", "OSF"),
    ("bearing",     "E-204", "TWF"),
    ("wear",        "E-204", "TWF"),
    ("corrosion",   "E-205", "RNF"),
    ("rust",        "E-205", "RNF"),
]


def _keyword_fallback(query: str) -> dict[str, Any]:
    q = query.lower()
    error_code   = "E-101"
    failure_type = "HDF"
    for keyword, code, ft in _FALLBACK_TOOL_PRIORITY:
        if keyword in q:
            error_code   = code
            failure_type = ft
            break
    return {
        "suspected_error_codes":   [error_code],
        "suspected_failure_types": [failure_type],
        "suspected_components":    [],
        "investigation_steps": [
            {
                "step":   1,
                "tool":   "get_context_by_error_code",
                "args":   {"error_code": error_code},
                "reason": "keyword-based fallback: retrieve base context",
            },
            {
                "step":   2,
                "tool":   "get_manual_procedures",
                "args":   {"error_code": error_code},
                "reason": "keyword-based fallback: retrieve maintenance procedures",
            },
            {
                "step":   3,
                "tool":   "get_failure_pattern_context",
                "args":   {"failure_type": failure_type},
                "reason": "keyword-based fallback: retrieve failure pattern",
            },
        ],
        "max_iterations": 3,
    }


def _extract_seed_codes(query: str) -> list[str]:
    explicit = re.findall(r"E-\d{3}", query, re.IGNORECASE)
    if explicit:
        return list(dict.fromkeys(c.upper() for c in explicit))

    q = query.lower()
    found = []
    for keyword, code, _ in _FALLBACK_TOOL_PRIORITY:
        if keyword in q and code not in found:
            found.append(code)
    return found[:3]


def plan(client: OpenAI, query: str, session: Session | None = None) -> dict[str, Any]:
    preview_text = ""
    if session is not None:
        seed_codes = _extract_seed_codes(query)
        print(f"[planner] seed_codes: {seed_codes}")
        if seed_codes:
            try:
                preview      = get_graph_preview(session, seed_codes)
                preview_text = format_graph_preview(preview)
                print(f"[planner] preview:\n{preview_text}")
            except Exception as e:
                print(f"[planner] graph preview failed: {e}")

    system_prompt = _PLANNER_PROMPT
    if preview_text:
        system_prompt = f"{_PLANNER_PROMPT}\n\n{preview_text}"

    response = client.chat.completions.create(
        model=settings.DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    raw = (response.choices[0].message.content or "").strip()
    print(f"[planner] raw LLM response:\n{raw}") 

    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()

    try:
        result = json.loads(raw)
        print(f"[planner] parsed plan: {json.dumps(result, indent=2)}") 
        return result
    except json.JSONDecodeError as e:
        print(f"[planner] JSON parse failed: {e} → fallback")
        return _keyword_fallback(query)