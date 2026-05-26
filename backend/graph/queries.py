from __future__ import annotations

from neo4j import Session


def get_context_by_error_code(session: Session, code: str) -> dict:
    error = session.run(
        "MATCH (e:ErrorCode {code:$code}) RETURN e", code=code
    ).single()
    if not error:
        return {}

    incidents = session.run(
        """
        MATCH (i:Incident)-[:HAS_ERROR]->(e:ErrorCode {code:$code})
        RETURN i
        ORDER BY
            CASE i.severity
                WHEN 'critical' THEN 0
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                ELSE 3
            END,
            coalesce(i.occurred_at, '') DESC
        LIMIT 10
        """, code=code
    ).data()

    components = session.run(
        """
        MATCH (e:ErrorCode {code:$code})-[:INVOLVES]->(c:Component)
        RETURN c
        """, code=code
    ).data()

    manuals = session.run(
        """
        MATCH (e:ErrorCode {code:$code})-[:REFERENCES]->(m:ManualSection)
        RETURN m
        """, code=code
    ).data()

    images = session.run(
        """
        MATCH (img:Image)-[:SHOWS_ERROR]->(e:ErrorCode {code:$code})
        RETURN img
        ORDER BY
            CASE img.severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            img.confidence_score DESC
        LIMIT 5
        """, code=code
    ).data()

    sensor_graphs = session.run(
        """
        MATCH (sg:SensorGraph)-[:SHOWS_ERROR]->(e:ErrorCode {code:$code})
        RETURN sg ORDER BY sg.anomaly_score DESC
        LIMIT 5
        """, code=code
    ).data()

    dashboards = session.run(
        """
        MATCH (d:Dashboard)-[:SHOWS_ERROR]->(e:ErrorCode {code:$code})
        RETURN d ORDER BY d.timestamp DESC
        LIMIT 3
        """, code=code
    ).data()

    return {
        "error_code": dict(error["e"]),
        "incidents":  [dict(r["i"]) for r in incidents],
        "components": [dict(r["c"]) for r in components],
        "manuals":    [dict(r["m"]) for r in manuals],
        "evidence": {
            "images":        [dict(r["img"]) for r in images],
            "sensor_graphs": [dict(r["sg"])  for r in sensor_graphs],
            "dashboards":    [dict(r["d"])   for r in dashboards],
        },
        "reasoning_paths": (
            [{"source": code, "relation": "INVOLVES",   "target": f"{c['c'].get('component_id')} {c['c'].get('name', '')}"}  for c in components] +
            [{"source": code, "relation": "REFERENCES", "target": f"{m['m'].get('section_id')} {m['m'].get('title', '')}"}   for m in manuals] +
            [{"source": i["i"].get("incident_id", ""), "relation": "HAS_ERROR", "target": code}                              for i in incidents[:3]] +
            [{"source": r["img"].get("image_id", ""),  "relation": "SHOWS_ERROR", "target": code}                           for r in images[:3]] +
            [{"source": r["sg"].get("graph_id", ""),   "relation": "SHOWS_ERROR", "target": code}                           for r in sensor_graphs[:3]]
        ),
    }


def get_failure_pattern_context(session: Session, failure_type: str) -> dict:
    pattern = session.run(
        "MATCH (fp:FailurePattern {failure_type:$ft}) RETURN fp", ft=failure_type
    ).single()
    if not pattern:
        return {}

    pid = pattern["fp"]["pattern_id"]

    incidents = session.run(
        """
        MATCH (i:Incident)-[:BELONGS_TO]->(fp:FailurePattern {pattern_id:$pid})
        RETURN i ORDER BY i.severity DESC
        LIMIT 5
        """, pid=pid
    ).data()

    result = session.run(
        """
        MATCH (fp:FailurePattern {pattern_id:$pid})-[:MAPS_TO]->(e:ErrorCode)
        OPTIONAL MATCH (e)-[:INVOLVES]->(c:Component)
        OPTIONAL MATCH (e)-[:REFERENCES]->(m:ManualSection)
        RETURN e, collect(DISTINCT c) AS components, collect(DISTINCT m) AS manuals
        """, pid=pid
    ).single()

    pid_str = pattern["fp"].get("pattern_id", "")
    ec_str  = dict(result["e"]).get("code", "") if result else ""

    return {
        "failure_pattern": dict(pattern["fp"]),
        "incidents":       [dict(r["i"]) for r in incidents],
        "error_code":      dict(result["e"]) if result else {},
        "components":      [dict(c) for c in (result["components"] or []) if c],
        "manuals":         [dict(m) for m in (result["manuals"]    or []) if m],
        "reasoning_paths": (
            ([{"source": pid_str, "relation": "MAPS_TO", "target": ec_str}] if ec_str else []) +
            [{"source": ec_str,   "relation": "INVOLVES",   "target": f"{c.get('component_id')} {c.get('name', '')}"}  for c in ([dict(c) for c in (result["components"] or []) if c] if result else [])] +
            [{"source": ec_str,   "relation": "REFERENCES", "target": f"{m.get('section_id')} {m.get('title', '')}"}   for m in ([dict(m) for m in (result["manuals"]    or []) if m] if result else [])] +
            [{"source": r["i"].get("incident_id", ""), "relation": "BELONGS_TO", "target": pid_str}                    for r in incidents[:3]]
        ),
    }


def get_similar_incidents(session: Session, incident_id: str, limit: int = 5) -> dict:
    base = session.run(
        "MATCH (i:Incident {incident_id:$id}) RETURN i", id=incident_id
    ).single()
    if not base:
        return {}

    similar = session.run(
        """
        MATCH (a:Incident {incident_id:$id})-[:SIMILAR_TO]-(b:Incident)
        RETURN b ORDER BY b.severity DESC
        LIMIT $limit
        """, id=incident_id, limit=limit
    ).data()

    # 센서값 기반 유사도 — AI4I incidents끼리만 의미있음
    sensor_similar: list[dict] = []
    base_node = dict(base["i"])
    if base_node.get("source") == "AI4I2020":
        sensor_similar = session.run(
            """
            MATCH (b:Incident)
            WHERE b.incident_id <> $id
              AND b.error_code = $code
              AND b.source = 'AI4I2020'
            RETURN b
            ORDER BY b.severity DESC
            LIMIT $limit
            """, id=incident_id, code=base_node.get("error_code", ""), limit=limit
        ).data()

    return {
        "base":           base_node,
        "similar":        [dict(r["b"]) for r in similar],
        "sensor_similar": [dict(r["b"]) for r in sensor_similar],
        "reasoning_paths": (
            [{"source": incident_id, "relation": "SIMILAR_TO", "target": r["b"].get("incident_id", "")} for r in similar] +
            [{"source": incident_id, "relation": "SENSOR_SIMILAR_TO", "target": r["b"].get("incident_id", "")} for r in sensor_similar[:3]]
        ),
    }


def get_component_incident_chain(session: Session, component_id: str) -> dict:
    """component → error code → incident 멀티홉 체인"""
    rows = session.run(
        """
        MATCH (c:Component {component_id:$cid})<-[:INVOLVES]-(e:ErrorCode)
        OPTIONAL MATCH (i:Incident)-[:HAS_ERROR]->(e)
        OPTIONAL MATCH (sg:SensorGraph)-[:SHOWS_ANOMALY]->(c)
        OPTIONAL MATCH (img:Image)-[:DEPICTS]->(c)
        RETURN e, collect(DISTINCT i) AS incidents,
               collect(DISTINCT sg) AS graphs,
               collect(DISTINCT img) AS images
        """, cid=component_id
    ).data()

    component = session.run(
        "MATCH (c:Component {component_id:$cid}) RETURN c", cid=component_id
    ).single()

    chains = []
    for row in rows:
        chains.append({
            "error_code": dict(row["e"]),
            "incidents":  [dict(i) for i in row["incidents"] if i],
            "sensor_graphs": [dict(sg) for sg in row["graphs"] if sg],
            "images":     [dict(img) for img in row["images"] if img],
        })

    return {
        "component": dict(component["c"]) if component else {},
        "chains":    chains,
        "reasoning_paths": [
            path
            for row in rows
            for path in (
                [{"source": component_id, "relation": "INVOLVES", "target": dict(row["e"]).get("code", "")}] +
                [{"source": dict(i).get("incident_id", ""), "relation": "HAS_ERROR", "target": dict(row["e"]).get("code", "")} for i in row["incidents"] if i] +
                [{"source": dict(sg).get("graph_id", ""),  "relation": "SHOWS_ANOMALY", "target": component_id}               for sg in row["graphs"]    if sg] +
                [{"source": dict(img).get("image_id", ""), "relation": "DEPICTS",       "target": component_id}               for img in row["images"]   if img]
            )
        ],
    }


def get_manual_procedures(session: Session, error_code: str) -> list[dict]:
    rows = session.run(
        """
        MATCH (e:ErrorCode {code:$code})-[:REFERENCES]->(m:ManualSection)
        OPTIONAL MATCH (m)-[:HAS_PROCEDURE]->(p:Procedure)
        RETURN m, collect(p) AS procedures ORDER BY m.section_id
        """, code=error_code
    ).data()

    return [
        {
            "section":    dict(r["m"]),
            "procedures": [dict(p) for p in r["procedures"] if p],
            "reasoning_paths": (
                [{"source": error_code, "relation": "REFERENCES", "target": dict(r["m"]).get("section_id", "")}] +
                [{"source": dict(r["m"]).get("section_id", ""), "relation": "HAS_PROCEDURE", "target": dict(p).get("procedure_id", "")} for p in r["procedures"] if p]
            ),
        }
        for r in rows
    ]

def get_graph_preview(session: Session, error_codes: list[str]) -> dict:
    """
    planner에게 넘길 graph topology 미리보기.
    error_code 목록 기준으로 1-hop 이웃 노드 요약 반환.
    """
    if not error_codes:
        return {}

    preview: dict[str, Any] = {}

    for code in error_codes:
        row = session.run(
            """
            MATCH (e:ErrorCode {code: $code})
            OPTIONAL MATCH (e)-[:INVOLVES]->(c:Component)
            OPTIONAL MATCH (e)-[:REFERENCES]->(m:ManualSection)
            OPTIONAL MATCH (i:Incident)-[:HAS_ERROR]->(e)
            RETURN
                e,
                collect(DISTINCT c)[..5]  AS components,
                collect(DISTINCT m)[..5]  AS manuals,
                collect(DISTINCT i)[..5]  AS incidents
            """,
            code=code,
        ).single()

        if not row or not row["e"]:
            continue

        preview[code] = {
            "name":       row["e"].get("name", ""),
            "severity":   row["e"].get("severity", ""),
            "components": [
                f"{c.get('component_id')} {c.get('name', '')}"
                for c in [dict(c) for c in row["components"] if c]
            ],
            "manuals": [
                f"{m.get('section_id')} {m.get('title', '')}"
                for m in [dict(m) for m in row["manuals"] if m]
            ],
            "incidents": [
                i.get("incident_id", "")
                for i in [dict(i) for i in row["incidents"] if i]
            ],
        }

    return preview


def format_graph_preview(preview: dict) -> str:
    """Format graph preview for planner prompt."""
    if not preview:
        return ""

    lines = ["[Graph Preview — use these IDs in your plan]"]
    for code, info in preview.items():
        lines.append(f"\n{code}: {info['name']} (severity={info['severity']})")
        if info["components"]:
            lines.append(f"  Components : {', '.join(info['components'])}")
        if info["manuals"]:
            lines.append(f"  Manuals    : {', '.join(info['manuals'])}")
        if info["incidents"]:
            lines.append(f"  Incidents  : {', '.join(info['incidents'])}")
    return "\n".join(lines)