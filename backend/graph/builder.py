from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from core.config import settings

BACKEND_DIR   = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def load_json(path: Path) -> Any:
    if not path.exists():
        print(f"[WARN] not found: {path}")
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def insert_error_codes(session, data: list[dict]) -> int:
    for item in data:
        session.run(
            "MERGE (e:ErrorCode {code:$code}) SET e.name=$name, e.severity=$severity, e.description=$description",
            code=item["code"], name=item["name"],
            severity=item["severity"], description=item["description"],
        )
    return len(data)


def insert_components(session, data: list[dict]) -> int:
    for item in data:
        session.run(
            "MERGE (c:Component {component_id:$id}) SET c.name=$name, c.component_type=$type, c.location=$location",
            id=item["id"], name=item["name"], type=item["type"], location=item["location"],
        )
    return len(data)


def insert_manual_sections(session, data: list[dict]) -> int:
    for item in data:
        session.run(
            "MERGE (m:ManualSection {section_id:$id}) SET m.title=$title, m.content=$content, m.related_error_codes=$codes",
            id=item["id"], title=item["title"],
            content=item["content"], codes=item.get("related_errors", []),
        )
        steps = [l.strip() for l in item["content"].splitlines() if l.strip() and l.strip()[0].isdigit()]
        for i, step in enumerate(steps, 1):
            session.run(
                """
                MERGE (p:Procedure {procedure_id:$pid})
                SET p.step_number=$n, p.description=$desc, p.section_id=$sid
                WITH p MATCH (m:ManualSection {section_id:$sid}) MERGE (m)-[:HAS_PROCEDURE]->(p)
                """,
                pid=f"{item['id']}-P{i:02d}", n=i, desc=step, sid=item["id"],
            )
    return len(data)


def insert_incidents(session, data: list[dict]) -> int:
    for item in data:
        session.run(
            """
            MERGE (i:Incident {incident_id:$id})
            SET i.title=$title, i.department=$dept, i.occurred_at=$at,
                i.severity=$sev, i.error_code=$code, i.description=$desc,
                i.resolved=$resolved, i.resolution=$resolution,
                i.downtime_hours=$downtime, i.symptoms=$symptoms,
                i.source=$source, i.product_type=$ptype
            """,
            id=item["id"], title=item["title"], dept=item["department"],
            at=item.get("occurred_at"), sev=item["severity"], code=item["error_code"],
            desc=item["description"], resolved=item.get("resolved", False),
            resolution=item.get("resolution"), downtime=item.get("downtime_hours"),
            symptoms=item.get("symptoms", []),
            source=item.get("source", ""),
            ptype=item.get("product_type", ""),
        )
    return len(data)


def insert_failure_patterns(session, data: list[dict]) -> int:
    for item in data:
        session.run(
            """
            MERGE (fp:FailurePattern {pattern_id:$id})
            SET fp.source=$source,
                fp.failure_type=$ftype,
                fp.label=$label,
                fp.error_code=$code,
                fp.primary_component=$component,
                fp.count=$count,
                fp.total_failures=$total,
                fp.occurrence_rate=$rate,
                fp.avg_air_temperature_c=$air,
                fp.avg_process_temperature_c=$proc,
                fp.avg_rotational_speed_rpm=$rpm,
                fp.avg_torque_nm=$torque,
                fp.avg_tool_wear_min=$wear,
                fp.std_torque_nm=$std_torque,
                fp.std_tool_wear_min=$std_wear
            """,
            id=item["pattern_id"], source=item["source"],
            ftype=item["failure_type"], label=item["label"],
            code=item["error_code"], component=item["primary_component"],
            count=item["count"], total=item["total_failures"],
            rate=item["occurrence_rate"],
            air=item["avg_air_temperature_c"],
            proc=item["avg_process_temperature_c"],
            rpm=item["avg_rotational_speed_rpm"],
            torque=item["avg_torque_nm"], wear=item["avg_tool_wear_min"],
            std_torque=item["std_torque_nm"], std_wear=item["std_tool_wear_min"],
        )
    return len(data)


def insert_images(session, data: list[dict]) -> int:
    for item in data:
        a = item.get("analysis", {})
        session.run(
            """
            MERGE (img:Image {image_id:$id})
            SET img.file_path=$path, img.category=$cat,
                img.detected_component=$component, img.visual_defect=$defect,
                img.severity=$sev, img.condition=$cond,
                img.maintenance_urgency=$urgency, img.description=$desc,
                img.confidence_score=$conf, img.needs_review=$review,
                img.related_error_codes=$codes
            """,
            id=item["image_id"], path=item["file_path"], cat=item["category"],
            component=a.get("detected_component", ""), defect=a.get("visual_defect", ""),
            sev=a.get("severity", ""), cond=a.get("condition", ""),
            urgency=a.get("maintenance_urgency", ""), desc=a.get("description", ""),
            conf=item.get("confidence_score", -1), review=item.get("needs_review", False),
            codes=item.get("graph_links", {}).get("related_error_codes", []),
        )
    return len(data)


def insert_sensor_graphs(session, data: list[dict]) -> int:
    for item in data:
        session.run(
            """
            MERGE (sg:SensorGraph {graph_id:$id})
            SET sg.file_path=$path, sg.graph_type=$gtype, sg.sensor_type=$stype,
                sg.anomaly_type=$atype, sg.anomaly_score=$score, sg.severity=$sev,
                sg.related_error_code=$code, sg.related_incident=$incident,
                sg.related_component=$component,
                sg.timestamp_start=$ts_start, sg.timestamp_anomaly=$ts_anomaly
            """,
            id=item["graph_id"], path=item["file_path"],
            gtype=item["graph_type"], stype=item["sensor_type"],
            atype=item["anomaly_type"], score=item.get("anomaly_score", 0.0),
            sev=item.get("severity", ""), code=item.get("related_error_code", ""),
            incident=item.get("related_incident", ""),
            component=item.get("related_component", ""),
            ts_start=item.get("timestamp_start", ""),
            ts_anomaly=item.get("timestamp_anomaly", ""),
        )
    return len(data)


def insert_dashboards(session, data: list[dict]) -> int:
    for item in data:
        session.run(
            """
            MERGE (d:Dashboard {dashboard_id:$id})
            SET d.file_path=$path, d.dashboard_type=$dtype, d.severity=$sev,
                d.anomaly_panels=$panels, d.related_incidents=$incidents,
                d.related_error_codes=$codes, d.timestamp=$ts
            """,
            id=item["dashboard_id"], path=item["file_path"],
            dtype=item["dashboard_type"], sev=item.get("severity", ""),
            panels=item.get("anomaly_panels", []),
            incidents=item.get("related_incidents", []),
            codes=item.get("related_error_codes", []),
            ts=item.get("timestamp", ""),
        )
    return len(data)


def insert_edges(session, incidents, failure_patterns, error_component_map,
                 image_metadata, sensor_metadata, dashboard_data) -> None:
    run = session.run

    # Incident → ErrorCode, Component
    for inc in incidents:
        if code := inc.get("error_code"):
            run("MATCH (i:Incident {incident_id:$iid}), (e:ErrorCode {code:$code}) MERGE (i)-[:HAS_ERROR]->(e)",
                iid=inc["id"], code=code)
        for cid in inc.get("affected_components", []):
            run("MATCH (i:Incident {incident_id:$iid}), (c:Component {component_id:$cid}) MERGE (i)-[:AFFECTS]->(c)",
                iid=inc["id"], cid=cid)

    # Incident → FailurePattern (AI4I incidents만)
    for inc in incidents:
        if inc.get("source") != "AI4I2020":
            continue
        for ft in inc.get("failure_types", []):
            run("MATCH (i:Incident {incident_id:$iid}), (fp:FailurePattern {pattern_id:$pid}) MERGE (i)-[:BELONGS_TO]->(fp)",
                iid=inc["id"], pid=f"FP-{ft}")

    # FailurePattern → ErrorCode, Component
    for fp in failure_patterns:
        run("MATCH (fp:FailurePattern {pattern_id:$pid}), (e:ErrorCode {code:$code}) MERGE (fp)-[:MAPS_TO]->(e)",
            pid=fp["pattern_id"], code=fp["error_code"])
        run("MATCH (fp:FailurePattern {pattern_id:$pid}), (c:Component {component_id:$cid}) MERGE (fp)-[:INVOLVES]->(c)",
            pid=fp["pattern_id"], cid=fp["primary_component"])

    # ErrorCode → Component, ManualSection
    for code, comp_ids in error_component_map.items():
        for cid in comp_ids:
            run("MATCH (e:ErrorCode {code:$code}), (c:Component {component_id:$cid}) MERGE (e)-[:INVOLVES]->(c)",
                code=code, cid=cid)

    run("MATCH (e:ErrorCode), (m:ManualSection) WHERE e.code IN m.related_error_codes MERGE (e)-[:REFERENCES]->(m)")

    # Incident → Incident (같은 error_code)
    run("""
        MATCH (a:Incident), (b:Incident)
        WHERE a.incident_id < b.incident_id AND a.error_code = b.error_code
        MERGE (a)-[:SIMILAR_TO]->(b)
    """)

    # Image → ErrorCode, Component, Incident
    for item in image_metadata:
        iid   = item["image_id"]
        links = item.get("graph_links", {})
        for code in links.get("related_error_codes", []):
            run("MATCH (img:Image {image_id:$iid}), (e:ErrorCode {code:$code}) MERGE (img)-[:SHOWS_ERROR]->(e)",
                iid=iid, code=code)
        for cid in links.get("related_components", []):
            run("MATCH (img:Image {image_id:$iid}), (c:Component {component_id:$cid}) MERGE (img)-[:DEPICTS]->(c)",
                iid=iid, cid=cid)
        for inc_id in links.get("related_incidents", []):
            run("MATCH (i:Incident {incident_id:$inc_id}), (img:Image {image_id:$iid}) MERGE (i)-[:LINKED_IMAGE]->(img)",
                inc_id=inc_id, iid=iid)

    # SensorGraph → ErrorCode, Incident, Component
    for item in sensor_metadata:
        gid  = item["graph_id"]
        code = item.get("related_error_code", "")
        inc  = item.get("related_incident", "")
        comp = item.get("related_component", "")
        if code:
            run("MATCH (sg:SensorGraph {graph_id:$gid}), (e:ErrorCode {code:$code}) MERGE (sg)-[:SHOWS_ERROR]->(e)",
                gid=gid, code=code)
        if inc:
            run("MATCH (i:Incident {incident_id:$inc}), (sg:SensorGraph {graph_id:$gid}) MERGE (i)-[:LINKED_GRAPH]->(sg)",
                inc=inc, gid=gid)
        if comp:
            run("""
                MATCH (sg:SensorGraph {graph_id:$gid}), (c:Component)
                WHERE toLower(c.name) CONTAINS toLower($comp) OR toLower($comp) CONTAINS toLower(c.name)
                MERGE (sg)-[:SHOWS_ANOMALY]->(c)
                """, gid=gid, comp=comp)

    # Dashboard → Incident, ErrorCode
    for item in dashboard_data:
        did = item["dashboard_id"]
        for inc_id in item.get("related_incidents", []):
            run("MATCH (i:Incident {incident_id:$inc}), (d:Dashboard {dashboard_id:$did}) MERGE (i)-[:LINKED_DASHBOARD]->(d)",
                inc=inc_id, did=did)
        for code in item.get("related_error_codes", []):
            run("MATCH (d:Dashboard {dashboard_id:$did}), (e:ErrorCode {code:$code}) MERGE (d)-[:SHOWS_ERROR]->(e)",
                did=did, code=code)


def main() -> None:
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()

        error_codes      = load_json(PROCESSED_DIR / "error_codes.json")
        components       = load_json(PROCESSED_DIR / "components.json")
        manual_sections  = load_json(PROCESSED_DIR / "manual_sections.json")
        error_comp_map   = load_json(PROCESSED_DIR / "error_component_map.json")
        incidents        = load_json(PROCESSED_DIR / "incidents.json")
        failure_patterns = load_json(PROCESSED_DIR / "failure_patterns.json")
        image_metadata   = load_json(PROCESSED_DIR / "image_metadata.json")
        sensor_metadata  = load_json(PROCESSED_DIR / "sensor_graph_metadata.json")
        dashboard_data   = load_json(PROCESSED_DIR / "dashboard_metadata.json")

        with driver.session() as session:
            print(f"ErrorCode       {insert_error_codes(session, error_codes)}")
            print(f"Component       {insert_components(session, components)}")
            print(f"ManualSection   {insert_manual_sections(session, manual_sections)}")
            print(f"Incident        {insert_incidents(session, incidents)}")
            print(f"FailurePattern  {insert_failure_patterns(session, failure_patterns)}")
            print(f"Image           {insert_images(session, image_metadata)}")
            print(f"SensorGraph     {insert_sensor_graphs(session, sensor_metadata)}")
            print(f"Dashboard       {insert_dashboards(session, dashboard_data)}")
            insert_edges(session, incidents, failure_patterns, error_comp_map,
                         image_metadata, sensor_metadata, dashboard_data)

        with driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            print(f"\nnodes {nodes}  edges {edges}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()