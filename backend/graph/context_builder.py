from __future__ import annotations

from datetime import datetime
from typing import Any


_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _neo4j_val(val: Any) -> Any:
    """Convert Neo4j integer objects to native Python values."""
    if isinstance(val, dict) and "low" in val and "high" in val:
        return val["low"] + val["high"] * (2 ** 32)
    return val


def _get(node: dict, key: str, default: Any = "") -> Any:
    """Get a value from a node and apply Neo4j type conversion."""
    return _neo4j_val(node.get(key, default))



def _sev(node: dict) -> int:
    return _SEV_RANK.get(str(_get(node, "severity")).lower(), 9)


def _parse_ts(node: dict) -> datetime:
    raw = _get(node, "timestamp") or _get(node, "timestamp_anomaly") or ""
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return datetime.min


def rank_evidence(
    images:        list[dict],
    sensor_graphs: list[dict],
    dashboards:    list[dict],
    max_each: int = 5,
) -> dict[str, list[dict]]:
    return {
        "images": sorted(
            images,
            key=lambda x: (_sev(x), -(_neo4j_val(x.get("confidence_score") or 0))),
        )[:max_each],
        "sensor_graphs": sorted(
            sensor_graphs,
            key=lambda x: (_sev(x), -(_neo4j_val(x.get("anomaly_score") or 0))),
        )[:max_each],
        "dashboards": sorted(
            dashboards,
            key=lambda x: (_sev(x), _parse_ts(x)),
            reverse=True,
        )[:max_each],
    }


def format_context(ctx: dict[str, Any]) -> str:
    lines: list[str] = []

    if ec := ctx.get("error_code"):
        lines += [
            "[ErrorCode]",
            f"{_get(ec, 'code')}: {_get(ec, 'name')} (severity={_get(ec, 'severity')})",
            _get(ec, "description"),
            "",
        ]

    if fp := ctx.get("failure_pattern"):
        count           = _get(fp, "count")
        total           = _get(fp, "total_failures")
        occurrence_rate = _neo4j_val(fp.get("occurrence_rate", 0))
        lines += [
            "[FailurePattern]",
            f"{_get(fp, 'pattern_id')} — {_get(fp, 'label')}",
            (
                f"source: AI4I2020 | count={count} / {total} "
                f"({float(occurrence_rate) * 100:.1f}%)"
            ),
            (
                f"avg sensor: air={_get(fp, 'avg_air_temperature_c')}°C  "
                f"process={_get(fp, 'avg_process_temperature_c')}°C  "
                f"rpm={_get(fp, 'avg_rotational_speed_rpm')}  "
                f"torque={_get(fp, 'avg_torque_nm')} Nm  "
                f"tool_wear={_get(fp, 'avg_tool_wear_min')} min"
            ),
            "",
        ]

    if incidents := ctx.get("incidents"):
        lines.append("[Incidents]")
        for i in incidents:
            iid = _get(i, "incident_id") or _get(i, "id")
            lines.append(f"- {iid}: {_get(i, 'title')} [{_get(i, 'severity')}]")
            if desc := _get(i, "description"):
                lines.append(f"  {str(desc)[:120]}")
        lines.append("")

    if components := ctx.get("components"):
        lines.append("[Components]")
        for c in components:
            lines.append(
                f"- {_get(c, 'component_id')}: {_get(c, 'name')} ({_get(c, 'component_type')})"
            )
        lines.append("")

    if manuals := ctx.get("manuals"):
        lines.append("[Manual Procedures]")
        for m in manuals:
            if isinstance(m, dict) and "section" in m:
                sec   = m.get("section", {})
                sid   = _get(sec, "section_id")
                title = _get(sec, "title")
                lines.append(f"{sid} {title}")
                for p in m.get("procedures", []):
                    step = _neo4j_val(p.get("step_number", ""))
                    desc = _get(p, "description") or _get(p, "action")
                    if desc:
                        lines.append(f"  - Step {step}: {desc}")
            else:
                sid   = _get(m, "section_id")
                title = _get(m, "title")
                lines.append(f"- {sid}: {title}")
        lines.append("")

    if evidence := ctx.get("evidence"):
        if any(evidence.values()):
            lines.append("[Evidence]")
            for img in evidence.get("images", []):
                lines.append(
                    f"- Image {_get(img, 'image_id')} [{_get(img, 'category')}] "
                    f"defect={_get(img, 'visual_defect')} "
                    f"severity={_get(img, 'severity')} "
                    f"confidence={_neo4j_val(img.get('confidence_score'))}"
                )
            for sg in evidence.get("sensor_graphs", []):
                lines.append(
                    f"- SensorGraph {_get(sg, 'graph_id')} "
                    f"type={_get(sg, 'graph_type')} "
                    f"anomaly_score={_neo4j_val(sg.get('anomaly_score'))} "
                    f"severity={_get(sg, 'severity')}"
                )
            for d in evidence.get("dashboards", []):
                lines.append(
                    f"- Dashboard {_get(d, 'dashboard_id')} "
                    f"type={_get(d, 'dashboard_type')} "
                    f"severity={_get(d, 'severity')}"
                )
            lines.append("")

    if vector_manuals := ctx.get("vector_manuals"):
        lines.append("[Vector-Retrieved Manuals]")
        for m in vector_manuals:
            score = _neo4j_val(m.get("score", 0))
            lines.append(f"- {_get(m, 'section_id')} {_get(m, 'title')} (similarity={score})")
            if text := m.get("embedding_text", ""):
                content = text.split(". ", 2)[-1] if ". " in text else text
                lines.append(f"  {content[:300]}")
        lines.append("")

    if vector_incidents := ctx.get("vector_incidents"):
        lines.append("[Vector-Retrieved Incidents]")
        for i in vector_incidents:
            score = _neo4j_val(i.get("score", 0))
            lines.append(
                f"- {_get(i, 'incident_id')} [{_get(i, 'error_code')}] "
                f"{_get(i, 'title')} (severity={_get(i, 'severity')}, similarity={score})"
            )
        lines.append("")

    raw_paths = ctx.get("reasoning_paths") or []

    if raw_paths and isinstance(raw_paths[0], dict) and "section" in raw_paths[0]:
        flat_paths = [p for item in raw_paths for p in item.get("reasoning_paths", [])]
    else:
        flat_paths = list(raw_paths)

    for chain in ctx.get("chains", []):
        flat_paths += chain.get("reasoning_paths", [])

    if flat_paths:
        seen = set()
        lines.append("[Reasoning Paths]")
        for p in flat_paths:
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
        lines.append("")

    return "\n".join(lines).strip()



def build_graphrag_context(
    ctx:              dict[str, Any] | list,
    max_evidence:     int             = 5,
    vector_manuals:   list[dict] | None = None,
    vector_incidents: list[dict] | None = None,
) -> str:
    if isinstance(ctx, list):
        merged_paths   = [p for item in ctx for p in item.get("reasoning_paths", [])]
        merged_manuals = ctx
        ctx = {
            "manuals":         merged_manuals,
            "reasoning_paths": merged_paths,
        }
    else:
        ctx = dict(ctx)

    if evidence := ctx.get("evidence"):
        ctx["evidence"] = rank_evidence(
            images        = evidence.get("images", []),
            sensor_graphs = evidence.get("sensor_graphs", []),
            dashboards    = evidence.get("dashboards", []),
            max_each      = max_evidence,
        )

    if vector_manuals:
        ctx["vector_manuals"] = vector_manuals
    if vector_incidents:
        ctx["vector_incidents"] = vector_incidents

    return format_context(ctx)