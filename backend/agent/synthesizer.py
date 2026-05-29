from __future__ import annotations

from typing import Any

from openai import OpenAI

from core.config import settings
from agent.explorer import ToolCall, EvidenceState


_SYNTHESIZER_PROMPT = """\
You are a concise operational diagnostic assistant for industrial equipment.
Given investigation evidence, provide a direct, actionable diagnosis.

Structure your response as:
1. Likely Diagnosis (2-3 sentences max)
2. Key Evidence (bullet points, node IDs only)
3. Affected Components (IDs and names)
4. Actions (prioritized, manual-grounded only)
5. Evidence Path (reasoning paths as-is)
6. Risk Level (one line)

Rules:
- Be concise. Avoid narrative prose.
- Only recommend actions traceable to ManualSection or Procedure evidence.
- If no manual procedure was retrieved, state: "Procedure evidence insufficient."
- Do not invent component names, IDs, or steps not present in evidence.
- Section 5: list reasoning paths exactly as provided — do not add or infer new ones.
"""

def _format_reasoning_paths(state: EvidenceState) -> str:
    if not state.reasoning_paths:
        return "  (no reasoning paths collected)"
    seen  = set()
    lines = []
    for p in state.reasoning_paths:
        src = str(p.get("source", "")).strip()
        rel = str(p.get("relation", "")).strip()
        tgt = str(p.get("target", "")).strip()
        if not src or not tgt:
            continue
        key = (src, rel, tgt)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {src} -[:{rel}]-> {tgt}")
    return "\n".join(lines) if lines else "  (no valid paths)"


def synthesize(
    client:     OpenAI,
    query:      str,
    plan:       dict[str, Any],
    steps:      list[ToolCall],
    validation: dict[str, Any],
    state:      EvidenceState | None = None,
) -> str:
    
    # summarize collected evidence for final diagnosis
    evidence_summary = "\n\n".join(
        f"[{step.tool} step {i}] relevance={step.relevance_score} support={step.support_score}\n{step.formatted[:600]}"
        for i, step in enumerate(steps, 1)
        if not step.skipped and step.formatted
    )

    reasoning_paths = _format_reasoning_paths(state) if state else "  (state not provided)"

    # inject structured runtime state into synthesis prompt
    state_summary = ""
    if state:
        state_summary = (
            f"Evidence State:\n"
            f"  confirmed error codes : {sorted(state.discovered_error_codes)}\n"
            f"  confirmed components  : {sorted(state.discovered_components)}\n"
            f"  confirmed manuals     : {sorted(state.discovered_manuals)}\n"
            f"  overall confidence    : {round(state.confidence, 3)}\n"
            f"  contradictions        : {state.contradictions}\n\n"
        )
    
    vector_summary = ""
    if state:
        if state.vector_manuals:
            vm_lines = "\n".join(
                f"  - {m.get('section_id', 'UNKNOWN')} "
                f"{m.get('title', 'Untitled')} "
                f"(similarity={m.get('score', 0)})"
                for m in state.vector_manuals
            )
            vector_summary += f"Vector-Retrieved Manuals:\n{vm_lines}\n"
        if state.vector_incidents:
            vi_lines = "\n".join(
                f"  - {i.get('incident_id', 'UNKNOWN')} "
                f"[{i.get('error_code', 'UNKNOWN')}] "
                f"{i.get('title', 'Untitled')} "
                f"(similarity={i.get('score', 0)})"
                for i in state.vector_incidents
            )
            vector_summary += f"Vector-Retrieved Incidents:\n{vi_lines}\n"
        if state.visual_matches:
            vi_lines = "\n".join(
                f"  - {v['image_id']} [{v['category']}] "
                f"errors={v['related_error_codes']} "
                f"(similarity={v['score']}, final={v['final_score']})"
                for v in state.visual_matches
            )
            vector_summary += f"Visual Evidence (CLIP):\n{vi_lines}\n"

    user_content = (
        f"Query: {query}\n\n"
        f"Investigation Plan:\n"
        f"  suspected error codes  : {plan.get('suspected_error_codes')}\n"
        f"  suspected failure types: {plan.get('suspected_failure_types')}\n\n"
        f"{state_summary}"
        f"Validation:\n"
        f"  consistent={validation.get('consistent')}  "
        f"confidence={validation.get('confidence')}\n"
        f"  confirmed codes : {validation.get('confirmed_error_codes')}\n"
        f"  key findings    : {validation.get('key_findings')}\n"
        f"  conflicts       : {validation.get('conflicts')}\n"
        f"  hallucination_risk: {validation.get('hallucination_risk')}\n\n"
        + (f"Vector Evidence:\n{vector_summary}\n" if vector_summary else "")
        + f"Reasoning Paths:\n{reasoning_paths}\n\n"
        f"Evidence:\n{evidence_summary}"
    )

    response = client.chat.completions.create(
        model=settings.DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYNTHESIZER_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    return response.choices[0].message.content or ""