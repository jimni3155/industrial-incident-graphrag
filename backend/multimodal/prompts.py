ERROR_CODE_REFERENCE = """\
Error code reference. Pick all that apply:
E-101 Overheat Warning
E-102 Pressure Drop
E-103 Vibration Anomaly
E-104 Lubrication Failure
E-105 Coolant Leak
E-201 Sensor Malfunction
E-202 Power Fluctuation
E-203 Conveyor Jam
E-204 Bearing Wear
E-205 Corrosion Alert

You must select at least one. If uncertain, choose the closest match."""

PROMPTS: dict[str, str] = {
    "defects": f"""\
You are an industrial maintenance expert analyzing equipment defect images.
Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.

{{
  "detected_component": "<e.g. pipe flange, bearing housing, pump seal>",
  "visual_defect": "<primary defect, e.g. severe corrosion, surface crack, leakage>",
  "severity": "<exactly one of: low / medium / high / critical>",
  "visible_symptoms": ["<symptom1>", "<symptom2>", "<symptom3>"],
  "related_error_codes": ["<from list below>"],
  "maintenance_urgency": "<exactly one of: routine / soon / immediate>",
  "description": "<2-3 factual sentences describing what is visible>"
}}

{ERROR_CODE_REFERENCE}""",

    "equipment": f"""\
You are an industrial equipment identification expert.
Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.

{{
  "detected_component": "<primary equipment name, e.g. centrifugal pump, induction motor>",
  "equipment_type": "<exactly one of: pump / valve / motor / bearing / compressor / conveyor / sensor / other>",
  "operational_role": "<what this equipment does, e.g. coolant circulation, pressure regulation>",
  "condition": "<exactly one of: good / fair / worn / damaged>",
  "visible_features": ["<feature1>", "<feature2>", "<feature3>"],
  "potential_issues": ["<issue1>", "<issue2>"],
  "related_error_codes": ["<from list below>"],
  "description": "<2-3 factual sentences describing the equipment>"
}}

{ERROR_CODE_REFERENCE}""",

    "diagrams": f"""\
You are an industrial engineering expert analyzing technical schematics.
Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.

{{
  "diagram_type": "<e.g. hydraulic schematic, bearing assembly, valve cross-section>",
  "depicted_component": "<main component shown>",
  "key_parts": ["<part1>", "<part2>", "<part3>"],
  "process_flow": "<brief description of the process or assembly shown>",
  "related_error_codes": ["<from list below>"],
  "maintenance_relevance": "<which maintenance tasks this diagram supports>",
  "description": "<2-3 factual sentences describing the diagram>"
}}

{ERROR_CODE_REFERENCE}""",
}

VERIFIER_PROMPT = """\
You are a quality checker for industrial image analysis results.
Given the original image and the analysis below, verify its accuracy.
Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.

Analysis to verify:
{analysis_json}

{{
  "component_correct": true,
  "defect_correct": true,
  "error_codes_plausible": true,
  "severity_plausible": true,
  "confidence_score": 0.0,
  "issues": []
}}

Replace the placeholder values with your actual assessment.
confidence_score below 0.6 means the analysis needs human review.\
"""