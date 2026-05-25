"""
3개 metadata 파일을 읽어서 등장한 ID 기반으로
보조 정의 JSON을 data/processed/ 에 생성한다.
incidents.json은 generate_incidents_from_ai4i.py 가 별도 생성.

실행: python -m scripts.generate_processed_data
출력: backend/data/processed/error_codes.json
    backend/data/processed/components.json
    backend/data/processed/error_component_map.json
    backend/data/processed/manual_sections.json
"""

from __future__ import annotations

import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROCESSED   = BACKEND_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

IMAGE_META  = PROCESSED / "image_metadata.json"
SENSOR_META = PROCESSED / "sensor_graph_metadata.json"
DASH_META   = PROCESSED / "dashboard_metadata.json"


# error_component_map은 mappings.py가 source of truth
# 여기선 역생성 불가능하므로 직접 정의
ERROR_COMPONENT_MAP: dict[str, list[str]] = {
    "E-101": ["C-002", "C-008", "C-011"],
    "E-102": ["C-001", "C-005", "C-014"],
    "E-103": ["C-003", "C-010", "C-013"],
    "E-104": ["C-007", "C-003", "C-012"],
    "E-105": ["C-011", "C-001"],
    "E-201": ["C-008", "C-015"],
    "E-202": ["C-015", "C-006"],
    "E-203": ["C-004", "C-012"],
    "E-204": ["C-003", "C-010", "C-013"],
    "E-205": ["C-009", "C-011"],
}

ERROR_CODE_META: dict[str, dict] = {
    "E-101": {"name": "Overheat Warning",    "severity": "high",     "description": "Component temperature exceeded safe operating threshold."},
    "E-102": {"name": "Pressure Drop",       "severity": "high",     "description": "System pressure fell below minimum operating level."},
    "E-103": {"name": "Vibration Anomaly",   "severity": "medium",   "description": "Abnormal vibration detected; possible misalignment or imbalance."},
    "E-104": {"name": "Lubrication Failure", "severity": "high",     "description": "Insufficient lubrication detected in rotating components."},
    "E-105": {"name": "Coolant Leak",        "severity": "critical", "description": "Coolant loss detected; risk of thermal runaway."},
    "E-201": {"name": "Sensor Malfunction",  "severity": "medium",   "description": "Sensor reading out of expected range or unresponsive."},
    "E-202": {"name": "Power Fluctuation",   "severity": "medium",   "description": "Voltage or current instability detected in supply line."},
    "E-203": {"name": "Conveyor Jam",        "severity": "high",     "description": "Conveyor belt stalled or speed below threshold."},
    "E-204": {"name": "Bearing Wear",        "severity": "high",     "description": "Bearing degradation detected via vibration or temperature signature."},
    "E-205": {"name": "Corrosion Alert",     "severity": "medium",   "description": "Surface corrosion identified; structural integrity may be compromised."},
}

# component name → ID, type, location
# derived from sensor_graph related_component names + image_metadata C-xxx
COMPONENT_REGISTRY: dict[str, dict] = {
    "C-001": {"name": "Hydraulic Pump",          "type": "pump",       "location": "Assembly Line 1 — Station AL-01"},
    "C-002": {"name": "Cooling Fan Motor",        "type": "motor",      "location": "Assembly Line 1 — Cooling Bay"},
    "C-003": {"name": "Main Bearing",             "type": "bearing",    "location": "Welding Station — Spindle Unit"},
    "C-004": {"name": "Conveyor Belt",            "type": "conveyor",   "location": "Assembly Line 2 — Transport Section"},
    "C-005": {"name": "Pressure Relief Valve",    "type": "valve",      "location": "Hydraulic Circuit — Line A"},
    "C-006": {"name": "Power Distribution Unit",  "type": "electrical", "location": "Control Room — Panel B"},
    "C-007": {"name": "Lubrication Pump",         "type": "pump",       "location": "Centralized Lube Station"},
    "C-008": {"name": "Coolant Pump",             "type": "pump",       "location": "Cooling Circuit — Primary Loop"},
    "C-009": {"name": "Pipe Manifold",            "type": "piping",     "location": "Assembly Line 1 — Junction Box"},
    "C-010": {"name": "Spindle Assembly",         "type": "mechanical", "location": "Welding Station — Rotary Head"},
    "C-011": {"name": "Coolant Tank",             "type": "tank",       "location": "Cooling Circuit — Reservoir"},
    "C-012": {"name": "Gear Box",                 "type": "mechanical", "location": "Drive Train — Secondary Shaft"},
    "C-013": {"name": "Vibration Sensor",         "type": "sensor",     "location": "Main Bearing Housing"},
    "C-014": {"name": "Flow Control Valve",       "type": "valve",      "location": "Hydraulic Circuit — Line B"},
    "C-015": {"name": "Motor Controller",         "type": "electrical", "location": "Assembly Line 1 — Drive Cabinet"},
}

# sensor_graph의 related_component 문자열 → C-xxx 매핑
COMP_NAME_TO_ID: dict[str, str] = {v["name"]: k for k, v in COMPONENT_REGISTRY.items()}


MANUAL_TEMPLATES: dict[str, dict] = {
    "M-001": {"title": "Hydraulic System Pressure Check",    "errors": ["E-102"],                   "content": "Hydraulic pressure inspection procedure.\n1. Verify system is depressurized before opening any line.\n2. Connect calibrated pressure gauge to test port on line A.\n3. Start pump and record pressure at idle (target: 130–150 bar).\n4. Increase load to 80% and verify pressure holds above 120 bar.\n5. Inspect all fittings and seals for leakage during load test.\n6. If pressure drops below 110 bar under load, isolate and replace pump seal."},
    "M-002": {"title": "Bearing Inspection and Replacement", "errors": ["E-204"],                   "content": "Bearing wear assessment and replacement procedure.\n1. Shut down and lock out the affected machine.\n2. Remove bearing housing cover and visually inspect for pitting or scoring.\n3. Measure radial play with feeler gauge (limit: 0.05 mm).\n4. Check vibration baseline with portable analyzer (alarm > 3.5 mm/s RMS).\n5. If wear exceeds threshold, remove bearing using extraction tool.\n6. Clean housing bore and install new bearing with correct interference fit.\n7. Apply specified grease quantity and reassemble housing.\n8. Run at low speed for 30 min and recheck temperature and vibration."},
    "M-003": {"title": "Coolant System Inspection",          "errors": ["E-101","E-105"],            "content": "Coolant level and leak inspection procedure.\n1. Allow system to cool to below 40°C before opening reservoir cap.\n2. Check coolant level against min/max marks on tank C-011.\n3. Inspect all hose connections and manifold joints for seepage.\n4. Measure coolant pH (acceptable range: 7.5–9.0).\n5. If level is below 60%, identify leak source before refilling.\n6. Flush and replace coolant if pH is out of range."},
    "M-004": {"title": "Lubrication System Maintenance",     "errors": ["E-104"],                   "content": "Centralized lubrication system service procedure.\n1. Check lube pump reservoir level (minimum 30% full).\n2. Inspect distribution lines for blockage or leaks.\n3. Verify pump output pressure (target: 2.5–4.0 bar).\n4. Replace filter element every 500 operating hours.\n5. Sample oil for metal particle analysis every 1,000 hours.\n6. If metal particles detected, inspect downstream gear box and bearings."},
    "M-007": {"title": "Pressure Relief Valve Testing",      "errors": ["E-102"],                   "content": "Pressure relief valve set-point verification.\n1. Isolate the relief valve from the system using upstream block valve.\n2. Connect test bench and increase pressure slowly until valve opens.\n3. Record cracking pressure (should be within ±5% of nameplate set-point).\n4. Allow valve to reseat and verify no leakage at 90% of set-point.\n5. If cracking pressure deviates, replace valve.\n6. Reinstall and perform system pressure test before returning to service."},
    "M-008": {"title": "Corrosion Inspection and Treatment", "errors": ["E-205"],                   "content": "Corrosion assessment and remediation procedure.\n1. Visually inspect all exposed metal surfaces under adequate lighting.\n2. Use wire brush to remove loose rust and assess base metal condition.\n3. Measure remaining wall thickness with ultrasonic gauge at corroded areas.\n4. If wall thickness is below 80% of nominal, replace the section.\n5. Apply rust converter and allow to cure 4 hours.\n6. Apply two coats of epoxy primer followed by topcoat."},
    "M-011": {"title": "Pipe and Manifold Integrity Check",  "errors": ["E-205","E-102"],            "content": "Pipe and manifold inspection for leaks and corrosion.\n1. Pressurize system to 110% of working pressure and hold for 15 minutes.\n2. Inspect all joints, welds, and flanges for leakage with leak detection spray.\n3. Check flange bolt torque against specification table.\n4. Visually inspect external surfaces for corrosion, pitting, or coating damage.\n5. Replace gaskets at any flange showing seepage.\n6. Document all findings with photographs and location reference."},
    "M-013": {"title": "Vibration Analysis and Balancing",   "errors": ["E-103"],                   "content": "Vibration analysis and dynamic balancing procedure.\n1. Record baseline vibration spectrum with FFT analyzer on all bearing points.\n2. Compare against alarm levels: warning 2.5 mm/s, alarm 4.0 mm/s RMS.\n3. Identify dominant frequency peaks and correlate to rotational speed harmonics.\n4. If 1x RPM peak dominant, perform dynamic balancing on rotor.\n5. If bearing frequencies present, replace bearing and retest.\n6. Record new baseline after corrective action."},
    "M-014": {"title": "Gear Box Oil Analysis and Service",  "errors": ["E-104","E-204"],            "content": "Gear box oil sampling and service procedure.\n1. Draw oil sample from drain port after at least 1 hour of operation.\n2. Send sample for ferrography and particle count analysis.\n3. If iron particle count exceeds 50 ppm, drain and inspect gear box.\n4. Check oil level via sight glass (fill to center of glass).\n5. Replace oil every 2,000 operating hours or annually, whichever is sooner.\n6. If large particles found, disassemble and inspect gear mesh and bearings."},
    "M-016": {"title": "Hydraulic Oil Sampling",             "errors": ["E-102"],                   "content": "Hydraulic oil condition monitoring procedure.\n1. Draw oil sample from system return line sampling valve.\n2. Submit for ISO cleanliness particle count (target: ISO 16/14/11 or better).\n3. Replace high-pressure filter element if differential pressure > 4 bar.\n4. Drain and replace oil if ISO cleanliness exceeds 19/17/14.\n5. Flush system with clean oil for 30 minutes before refilling."},
    "M-017": {"title": "Thermal Imaging Inspection",         "errors": ["E-101","E-105"],            "content": "Thermal imaging inspection of electrical and mechanical systems.\n1. Conduct scan with calibrated IR camera during normal operating load.\n2. Compare equipment surface temperature against ambient (delta-T alarm: +25°C).\n3. Focus on motor windings, bearing housings, and electrical panel connections.\n4. Flag any hotspot exceeding delta-T 15°C for follow-up investigation.\n5. Re-scan within 48 hours after any corrective action to verify resolution."},
    "M-018": {"title": "Spindle Alignment Verification",     "errors": ["E-103"],                   "content": "Spindle alignment and runout verification procedure.\n1. Mount dial indicator on spindle nose and rotate slowly by hand.\n2. Record TIR (Total Indicator Reading); alarm limit: 0.02 mm.\n3. If runout exceeds limit, check spindle bearing preload and taper fit.\n4. Perform laser alignment between spindle axis and guide rail.\n5. Run spindle at 50% speed and verify vibration < 1.5 mm/s RMS."},
    "M-019": {"title": "Emergency Shutdown and Recovery",    "errors": ["E-101","E-102","E-103","E-104","E-105"], "content": "Emergency shutdown and safe recovery procedure.\n1. Activate E-stop and verify all motion has ceased within 3 seconds.\n2. Isolate main power supply and apply lockout/tagout.\n3. Identify the fault code and consult the relevant error code manual section.\n4. Do not restart until root cause is confirmed and corrective action taken.\n5. Perform a controlled restart at reduced speed and monitor for 5 minutes.\n6. Document incident in the maintenance management system with timeline."},
}


def load(path: Path) -> list:
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def collect_ids(image: list, sensor: list, dash: list) -> tuple[set, set, set]:
    error_codes, comp_ids, manuals = set(), set(), set()

    for item in image:
        links = item.get("graph_links", {})
        error_codes.update(links.get("related_error_codes", []))
        comp_ids.update(links.get("related_components", []))
        manuals.update(links.get("related_manuals", []))

    for g in sensor:
        if c := g.get("related_error_code"):
            error_codes.add(c)
        if name := g.get("related_component"):
            if cid := COMP_NAME_TO_ID.get(name):
                comp_ids.add(cid)

    for d in dash:
        error_codes.update(d.get("related_error_codes", []))

    return error_codes, comp_ids, manuals


def build_error_codes(codes: set) -> list:
    return [{"code": c, **ERROR_CODE_META[c]} for c in sorted(codes) if c in ERROR_CODE_META]


def build_components(comp_ids: set) -> list:
    return [{"id": cid, **COMPONENT_REGISTRY[cid]} for cid in sorted(comp_ids) if cid in COMPONENT_REGISTRY]


def build_error_component_map(codes: set) -> dict:
    return {c: ERROR_COMPONENT_MAP[c] for c in sorted(codes) if c in ERROR_COMPONENT_MAP}




def build_manuals(manual_ids: set) -> list:
    result = []
    for mid in sorted(manual_ids):
        if mid not in MANUAL_TEMPLATES:
            continue
        t = MANUAL_TEMPLATES[mid]
        result.append({"id": mid, "title": t["title"], "related_errors": t["errors"], "content": t["content"]})
    return result


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    n = len(data) if isinstance(data, (list, dict)) else "?"
    print(f"  {path.name:<35} {n}")


def main() -> None:
    image  = load(IMAGE_META)
    sensor = load(SENSOR_META)
    dash   = load(DASH_META)

    error_codes, comp_ids, manuals = collect_ids(image, sensor, dash)

    dump(PROCESSED / "error_codes.json",         build_error_codes(error_codes))
    dump(PROCESSED / "components.json",          build_components(comp_ids))
    dump(PROCESSED / "error_component_map.json", build_error_component_map(error_codes))
    dump(PROCESSED / "manual_sections.json",     build_manuals(manuals))


if __name__ == "__main__":
    main()