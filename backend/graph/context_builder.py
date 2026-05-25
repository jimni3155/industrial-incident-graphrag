from __future__ import annotations

from datetime import datetime
from typing import Any


_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _sev(node: dict) -> int:
    return _SEV_RANK.get(str(node.get("severity", "")).lower(), 9)


def _parse_ts(node: dict) -> datetime:
    raw = node.get("timestamp") or node.get("timestamp_anomaly") or ""
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
            key=lambda x: (_sev(x), -(x.get("confidence_score") or 0)),
        )[:max_each],
        "sensor_graphs": sorted(
            sensor_graphs,
            key=lambda x: (_sev(x), -(x.get("anomaly_score") or 0)),
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
            f"{ec.get('code')}: {ec.get('name')} (severity={ec.get('severity')})",
            ec.get("description", ""),
            "",
        ]

    if fp := ctx.get("failure_pattern"):
        lines += [
            "[FailurePattern]",
            f"{fp.get('pattern_id')} — {fp.get('label')}",
            (
                f"source: AI4I2020 | count={fp.get('count')} / {fp.get('total_failures')} "
                f"({fp.get('occurrence_rate', 0) * 100:.1f}%)"
            ),
            (
                f"avg sensor: air={fp.get('avg_air_temperature_c')}°C  "
                f"process={fp.get('avg_process_temperature_c')}°C  "
                f"rpm={fp.get('avg_rotational_speed_rpm')}  "
                f"torque={fp.get('avg_torque_nm')} Nm  "
                f"tool_wear={fp.get('avg_tool_wear_min')} min"
            ),
            "",
        ]

    if incidents := ctx.get("incidents"):
        lines.append("[Incidents]")
        for i in incidents:
            iid = i.get("incident_id") or i.get("id", "")
            lines.append(f"- {iid}: {i.get('title')} [{i.get('severity')}]")
            if desc := i.get("description"):
                lines.append(f"  {desc[:120]}")
        lines.append("")

    if components := ctx.get("components"):
        lines.append("[Components]")
        for c in components:
            lines.append(f"- {c.get('component_id')}: {c.get('name')} ({c.get('component_type')})")
        lines.append("")

    if manuals := ctx.get("manuals"):
        lines.append("[Manuals]")
        for m in manuals:
            lines.append(f"- {m.get('section_id')}: {m.get('title')}")
        lines.append("")

    if evidence := ctx.get("evidence"):
        if any(evidence.values()):
            lines.append("[Evidence]")
            for img in evidence.get("images", []):
                lines.append(
                    f"- Image {img.get('image_id')} [{img.get('category')}] "
                    f"defect={img.get('visual_defect')} "
                    f"severity={img.get('severity')} "
                    f"confidence={img.get('confidence_score')}"
                )
            for sg in evidence.get("sensor_graphs", []):
                lines.append(
                    f"- SensorGraph {sg.get('graph_id')} "
                    f"type={sg.get('graph_type')} "
                    f"anomaly_score={sg.get('anomaly_score')} "
                    f"severity={sg.get('severity')}"
                )
            for d in evidence.get("dashboards", []):
                lines.append(
                    f"- Dashboard {d.get('dashboard_id')} "
                    f"type={d.get('dashboard_type')} "
                    f"severity={d.get('severity')}"
                )
            lines.append("")

    return "\n".join(lines).strip()


def build_graphrag_context(ctx: dict[str, Any], max_evidence: int = 5) -> str:
    ctx = dict(ctx)  # 원본 보호
    if evidence := ctx.get("evidence"):
        ctx["evidence"] = rank_evidence(
            images        = evidence.get("images", []),
            sensor_graphs = evidence.get("sensor_graphs", []),
            dashboards    = evidence.get("dashboards", []),
            max_each      = max_evidence,
        )
    return format_context(ctx)