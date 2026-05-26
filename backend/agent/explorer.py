from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from neo4j import Session
from openai import OpenAI

from core.config import settings
from graph.queries import (
    get_context_by_error_code,
    get_failure_pattern_context,
    get_similar_incidents,
    get_manual_procedures,
    get_component_incident_chain,
)
from graph.context_builder import build_graphrag_context


# arg validation 

_ARG_VALIDATORS: dict[str, dict[str, Any]] = {
    "get_context_by_error_code": {
        "required": ["error_code"],
        "patterns": {"error_code": r"^E-\d{3}$"},
    },
    "get_failure_pattern_context": {
        "required": ["failure_type"],
        "patterns": {"failure_type": r"^(TWF|HDF|PWF|OSF|RNF)$"},
    },
    "get_similar_incidents": {
        "required": ["incident_id"],
        "patterns": {"incident_id": r"^INC-\d{4}$"},
    },
    "get_manual_procedures": {
        "required": ["error_code"],
        "patterns": {"error_code": r"^E-\d{3}$"},
    },
    "get_component_incident_chain": {
        "required": ["component_id"],
        "patterns": {"component_id": r"^C-\d{3}$"},
    },
}

_TOOL_MAP = {
    "get_context_by_error_code":    lambda s, a: get_context_by_error_code(s, a["error_code"]),
    "get_failure_pattern_context":  lambda s, a: get_failure_pattern_context(s, a["failure_type"]),
    "get_similar_incidents":        lambda s, a: get_similar_incidents(s, a["incident_id"]),
    "get_manual_procedures":        lambda s, a: get_manual_procedures(s, a["error_code"]),
    "get_component_incident_chain": lambda s, a: get_component_incident_chain(s, a["component_id"]),
}

_REFLECTION_PROMPT = """\
You are an industrial investigation agent reviewing structured evidence state.
Decide if additional tool calls are needed based on the current state.

Respond ONLY with valid JSON:
{
  "sufficient": true|false,
  "reasoning": "<brief explanation>",
  "additional_steps": [
    {"tool": "<tool_name>", "args": {...}, "reason": "<why needed>"}
  ]
}

Available tools and argument constraints:
- get_context_by_error_code(error_code: str)  — error_code: E-\\d{3}
- get_failure_pattern_context(failure_type: str)  — one of: TWF, HDF, PWF, OSF, RNF
- get_similar_incidents(incident_id: str)  — incident_id: INC-\\d{4} only
- get_manual_procedures(error_code: str)  — error_code: E-\\d{3}
- get_component_incident_chain(component_id: str)  — component_id: C-\\d{3}

Rules:
- Only request tools not already in visited_tools.
- Only use IDs present in discovered_error_codes / discovered_components / discovered_incidents.
- Max 2 additional steps.
- If sufficient or contradictions are unresolvable, set sufficient=true.
"""


# EvidenceState

@dataclass
class EvidenceState:
    """
    탐색 중 누적되는 structured evidence.
    reflection이 문자열 대신 이 state를 보고 판단.
    """
    discovered_error_codes: set[str]  = field(default_factory=set)
    discovered_components:  set[str]  = field(default_factory=set)
    discovered_incidents:   set[str]  = field(default_factory=set)
    discovered_manuals:     set[str]  = field(default_factory=set)

    confidence_by_source: dict[str, float] = field(default_factory=dict)
    reasoning_paths: list[dict[str, str]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    
    # 방문한 tool+args 조합 (중복 방지)
    visited_tools: list[dict[str, Any]] = field(default_factory=list)

    confidence: float = 0.0

    def has_manual_evidence(self) -> bool:
        return bool(self.discovered_manuals)

    def has_error_evidence(self) -> bool:
        return bool(self.discovered_error_codes)

    def already_visited(self, tool: str, args: dict) -> bool:
        return {"tool": tool, "args": args} in self.visited_tools

    def to_reflection_context(self) -> str:
        """reflection prompt에 넘길 structured summary."""
        return json.dumps({
            "discovered_error_codes": sorted(self.discovered_error_codes),
            "discovered_components":  sorted(self.discovered_components),
            "discovered_incidents":   sorted(self.discovered_incidents),
            "discovered_manuals":     sorted(self.discovered_manuals),
            "contradictions":         self.contradictions,
            "confidence":             round(self.confidence, 3),
            "visited_tools":          self.visited_tools,
            "reasoning_path_count":   len(self.reasoning_paths),
        }, ensure_ascii=False, indent=2)


def _update_state(
    state:     EvidenceState,
    tool:      str,
    args:      dict,
    result:    Any,
    relevance: float,
    support:   float,
) -> None:
    """
    tool 결과의 structured field만 읽어서 state 업데이트.
    re.findall로 전체 텍스트 스캔하지 않음 — noise 방지.
    """
    if not result:
        return

    # structured field 기반 추출
    if isinstance(result, dict):
        # primary error code (queried)
        if ec := result.get("error_code"):
            if code := ec.get("code"):
                state.discovered_error_codes.add(code)

        # connected components
        for c in result.get("components", []):
            if cid := c.get("component_id"):
                state.discovered_components.add(cid)

        # component (get_component_incident_chain)
        if comp := result.get("component"):
            if cid := comp.get("component_id"):
                state.discovered_components.add(cid)

        # incidents
        for i in result.get("incidents", []):
            if iid := i.get("incident_id"):
                state.discovered_incidents.add(iid)

        # manuals
        for m in result.get("manuals", []):
            if mid := m.get("section_id"):
                state.discovered_manuals.add(mid)

        # chains (get_component_incident_chain)
        for chain in result.get("chains", []):
            if ec := chain.get("error_code"):
                if code := ec.get("code"):
                    state.discovered_error_codes.add(code)
            for i in chain.get("incidents", []):
                if iid := i.get("incident_id"):
                    state.discovered_incidents.add(iid)

        # failure_pattern → error_code
        if fp_ec := result.get("error_code"):
            if code := fp_ec.get("code"):
                state.discovered_error_codes.add(code)

        # reasoning paths
        for p in result.get("reasoning_paths", []):
            if p not in state.reasoning_paths:
                state.reasoning_paths.append(p)

    elif isinstance(result, list):
        # get_manual_procedures 반환값
        for item in result:
            if not isinstance(item, dict):
                continue
            if sec := item.get("section"):
                if mid := sec.get("section_id"):
                    state.discovered_manuals.add(mid)
            for p in item.get("reasoning_paths", []):
                if p not in state.reasoning_paths:
                    state.reasoning_paths.append(p)

    source_score = (relevance + support) / 2
    state.confidence_by_source[tool] = source_score
    state.confidence = sum(state.confidence_by_source.values()) / len(state.confidence_by_source)

    state.visited_tools.append({"tool": tool, "args": args})


# ToolCall

@dataclass
class ToolCall:
    tool:            str
    args:            dict
    reason:          str
    result:          Any   = field(default=None)
    formatted:       str   = field(default="")
    skipped:         bool  = field(default=False)
    skip_reason:     str   = field(default="")
    relevance_score: float = field(default=0.0)
    support_score:   float = field(default=0.0)
    from_reflection: bool  = field(default=False)

    def to_trace(self) -> dict:
        return {
            "tool":            self.tool,
            "args":            self.args,
            "reason":          self.reason,
            "skipped":         self.skipped,
            "skip_reason":     self.skip_reason,
            "relevance_score": self.relevance_score,
            "support_score":   self.support_score,
            "from_reflection": self.from_reflection,
        }


# helpers

def _validate_args(tool: str, args: dict) -> tuple[bool, str]:
    spec = _ARG_VALIDATORS.get(tool)
    if not spec:
        return False, f"unknown tool: {tool}"
    for key in spec["required"]:
        if key not in args:
            return False, f"missing required arg: {key}"
        val     = str(args[key])
        pattern = spec["patterns"].get(key)
        if pattern and not re.match(pattern, val):
            return False, f"invalid {key}='{val}' — expected pattern {pattern}"
    return True, ""


def _call_tool(session: Session, tool: str, args: dict) -> tuple[Any, str]:
    fn        = _TOOL_MAP[tool]
    result    = fn(session, args)
    formatted = build_graphrag_context(result) if isinstance(result, (dict, list)) else str(result)
    return result, formatted


def _score_relevance(query: str, result: Any, tool: str) -> tuple[float, float]:
    """
    (relevance_score, support_score).
    rule-based — 나중에 CLIP/embedding으로 교체 가능한 자리.
    """
    if not result:
        return 0.0, 0.0

    result_str    = json.dumps(result, default=str).lower()
    query_tokens  = set(re.findall(r"[a-z0-9\-]+", query.lower()))
    result_tokens = set(re.findall(r"[a-z0-9\-]+", result_str))
    overlap       = query_tokens & result_tokens

    relevance = min(1.0, len(overlap) * 0.1) if overlap else 0.0
    if re.findall(r"e-\d{3}", result_str):
        relevance = min(1.0, relevance + 0.2)
    if re.findall(r"c-\d{3}|inc-\d{4}", result_str):
        relevance = min(1.0, relevance + 0.1)

    support = 0.0
    if "manual" in result_str or "procedure" in result_str:
        support += 0.2
    if "image" in result_str or "sensor" in result_str or "dashboard" in result_str:
        support += 0.2
    if "critical" in result_str or "high" in result_str:
        support += 0.2
    if tool in ("get_context_by_error_code", "get_component_incident_chain"):
        support += 0.1

    return round(relevance, 3), round(min(1.0, support), 3)


def _requires_manual(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in [
        "what should i do", "fix", "repair", "procedure",
        "maintenance", "action", "how to", "recommend", "step",
    ])

def _has_tool(calls: list[ToolCall], tool_name: str) -> bool:
    return any(
        c.tool == tool_name and not c.skipped and c.result is not None
        for c in calls
    )

def _is_sufficient(
    query:     str,
    state:     EvidenceState,
    calls:     list[ToolCall],
    threshold: float = 0.55,
) -> bool:
    active = [c for c in calls if not c.skipped]
    if not active:
        return False

    if not state.has_error_evidence():
        return False
    if _requires_manual(query) and not _has_tool(calls, "get_manual_procedures"):
        return False

    if state.contradictions:
        return False

    return state.confidence >= threshold


def _reflect(
    client: OpenAI,
    query:  str,
    state:  EvidenceState,
    calls:  list[ToolCall],
) -> list[dict]:
    """EvidenceState를 기반으로 추가 탐색 여부 판단."""
    evidence_summary = "\n\n".join(
        f"[{c.tool}({c.args})] relevance={c.relevance_score} support={c.support_score}\n{c.formatted[:300]}"
        for c in calls if not c.skipped and c.formatted
    )

    response = client.chat.completions.create(
        model=settings.DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": _REFLECTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Evidence State:\n{state.to_reflection_context()}\n\n"
                    f"Evidence Summary:\n{evidence_summary}"
                ),
            },
        ],
        max_tokens=256,
        temperature=0.1,
    )

    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()

    try:
        parsed = json.loads(raw)
        if parsed.get("sufficient"):
            return []
        return parsed.get("additional_steps", [])
    except Exception:
        return []


# main entry

def explore(
    session:               Session,
    client:                OpenAI,
    query:                 str,
    plan:                  dict[str, Any],
    max_iterations:        int   = 5,
    use_reflection:        bool  = True,
    sufficiency_threshold: float = 0.55,
) -> tuple[list[ToolCall], EvidenceState]:
    """
    반환값: (calls, state)
    use_reflection=False → single-pass baseline (ablation용).
    """
    all_calls: list[ToolCall] = []
    state     = EvidenceState()
    pending   = list(plan.get("investigation_steps", []))[:max_iterations]
    iteration = 0

    while pending and iteration < max_iterations:
        cfg  = pending.pop(0)
        tool = cfg.get("tool", "")
        args = cfg.get("args", {})

        if state.already_visited(tool, args):
            iteration += 1
            continue

        call = ToolCall(
            tool            = tool,
            args            = args,
            reason          = cfg.get("reason", "plan step"),
            from_reflection = cfg.get("from_reflection", False),
        )

        # 1. args validation
        valid, reason = _validate_args(tool, args)
        if not valid:
            call.skipped     = True
            call.skip_reason = reason
            all_calls.append(call)
            iteration += 1
            continue

        # 2. tool 실행
        try:
            call.result, call.formatted = _call_tool(session, tool, args)
        except Exception as exc:
            call.skipped     = True
            call.skip_reason = f"execution error: {exc}"
            all_calls.append(call)
            iteration += 1
            continue

        # 3. relevance / support scoring
        call.relevance_score, call.support_score = _score_relevance(query, call.result, tool)

        # 4. state 업데이트
        _update_state(state, tool, args, call.result, call.relevance_score, call.support_score)

        all_calls.append(call)
        iteration += 1

        # 5. sufficiency check
        if use_reflection and _is_sufficient(query, state, all_calls, sufficiency_threshold):
            break

        # 6. reflection — pending 소진 후 트리거
        if use_reflection and not pending and iteration < max_iterations:
            remaining  = max_iterations - iteration
            additional = _reflect(client, query, state, all_calls)
            for extra in additional[:remaining]:
                extra["from_reflection"] = True
                pending.append(extra)

    return all_calls, state