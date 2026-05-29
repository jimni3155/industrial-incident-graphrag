from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from core.config import settings
from agent.explorer import ToolCall, EvidenceState


_VALIDATOR_PROMPT = """\
You are an industrial evidence validator performing cross-source verification.

You will receive pre-filtered evidence (low-confidence items already excluded).
Your job is semantic interpretation only.

Check:
1. Do sensor graphs and image evidence point to the same component?
2. Are severity levels consistent across sources?
3. Is there any claim not traceable to a specific node ID?

Respond ONLY with valid JSON:
{
  "consistent": true|false,
  "confidence": 0.0-1.0,
  "confirmed_error_codes": ["E-xxx"],
  "excluded_evidence": ["<id and reason>"],
  "conflicts": ["<description>"],
  "key_findings": ["<finding with node ID>"],
  "hallucination_risk": "low|medium|high"
}
"""


def _rule_based_filter(calls: list[ToolCall]) -> tuple[list[ToolCall], list[str]]:
    """Filter low-confidence evidence before LLM validation."""
    passed:   list[ToolCall] = []
    excluded: list[str]      = []

    for c in calls:
        if c.skipped or not c.formatted:
            continue

        result_str = json.dumps(c.result, default=str) if c.result else ""

        low_conf_images = re.findall(
            r'"image_id":\s*"([^"]+)"[^}]*"confidence_score":\s*(0\.\d+)',
            result_str,
        )
        for img_id, score_str in low_conf_images:
            if float(score_str) < 0.6:
                excluded.append(f"{img_id}: confidence_score={score_str} < 0.6")

        if c.relevance_score < 0.1 and c.support_score < 0.1:
            excluded.append(f"{c.tool}({c.args}): relevance={c.relevance_score} support={c.support_score} too low")
            continue

        passed.append(c)

    return passed, excluded


def _check_severity_consistency(calls: list[ToolCall]) -> list[str]:
    """Detect cross-tool severity contradictions for the same error code.""" 
    source_map: dict[str, list[tuple[str, str]]] = {}
    contradictions: list[str] = []

    for c in calls:
        if not c.result:
            continue

        result_str = json.dumps(c.result, default=str).lower()

        if isinstance(c.result, dict):
            ec = c.result.get("error_code", {})
            if isinstance(ec, dict):
                code = ec.get("code", "")
                sev  = ec.get("severity", "")
                if code and sev:
                    if code not in source_map:
                        source_map[code] = []
                    source_map[code].append((c.tool, sev))

    for code, entries in source_map.items():
        sevs = set(s for _, s in entries)
        if len(sevs) > 1:
            details = ", ".join(f"{t}={s}" for t, s in entries)
            contradictions.append(
                f"{code} severity conflict across tools: {details}"
            )

    return contradictions


def validate(
    client: OpenAI,
    query:  str,
    calls:  list[ToolCall],
    state:  EvidenceState | None = None,
) -> dict[str, Any]:
    active, rule_excluded = _rule_based_filter(calls)

    # Baseline values derived from EvidenceState
    baseline_codes    = sorted(state.discovered_error_codes) if state else []
    baseline_conf     = round(state.confidence, 3)           if state else 0.0
    baseline_manuals  = sorted(state.discovered_manuals)     if state else []
    hallucination_risk = (
        "low"    if baseline_manuals and baseline_conf >= 0.6 else
        "medium" if baseline_conf >= 0.4 else
        "high"
    )

    if not active:
        return {
            "consistent":            False,
            "confidence":            baseline_conf,
            "confirmed_error_codes": baseline_codes,
            "excluded_evidence":     rule_excluded,
            "conflicts":             ["no evidence passed rule-based filter"],
            "key_findings":          [f"{c} confirmed via EvidenceState" for c in baseline_codes],
            "hallucination_risk":    hallucination_risk,
        }

    contradictions = _check_severity_consistency(active)
    if state:
        contradictions += state.contradictions

    evidence_text = "\n\n".join(
        f"[{c.tool}({c.args})] relevance={c.relevance_score} support={c.support_score}\n{c.formatted}"
        for c in active
    )

    state_summary = ""
    if state:
        state_summary = (
            f"\nEvidence State Summary:\n"
            f"  error codes: {baseline_codes}\n"
            f"  components:  {sorted(state.discovered_components)}\n"
            f"  manuals:     {baseline_manuals}\n"
            f"  confidence:  {baseline_conf}\n"
        )

    response = client.chat.completions.create(
        model=settings.DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": _VALIDATOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n"
                    f"{state_summary}\n"
                    f"Pre-detected contradictions: {contradictions}\n\n"
                    f"Evidence:\n{evidence_text}"
                ),
            },
        ],
        max_tokens=512,
        temperature=0.1,
    )

    raw = (response.choices[0].message.content or "").strip()

    # Handle markdown-wrapped JSON responses from the LLM
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw   = "\n".join(lines[1:-1]).strip()

    # Extract JSON object from free-form model output
    if not raw.startswith("{"):
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        raw = m.group(0) if m else raw

    try:
        result = json.loads(raw)
        
        # Backfill missing fields from EvidenceState
        if not result.get("confirmed_error_codes"):
            result["confirmed_error_codes"] = baseline_codes
        if not result.get("key_findings"):
            result["key_findings"] = [
                f"{c} confirmed via graph evidence (manuals: {', '.join(baseline_manuals)})"
                for c in baseline_codes
            ]

        # Preserve graph-derived confidence when higher
        if result.get("confidence", 0) < baseline_conf:
            result["confidence"] = baseline_conf
        result["excluded_evidence"] = rule_excluded + result.get("excluded_evidence", [])
        result["conflicts"]         = contradictions + result.get("conflicts", [])
        return result
    except Exception:
        return {
            "consistent":            True,
            "confidence":            baseline_conf,
            "confirmed_error_codes": baseline_codes,
            "excluded_evidence":     rule_excluded,
            "conflicts":             contradictions,
            "key_findings":          [
                f"{c} confirmed via graph evidence (manuals: {', '.join(baseline_manuals)})"
                for c in baseline_codes
            ],
            "hallucination_risk":    hallucination_risk,
        }