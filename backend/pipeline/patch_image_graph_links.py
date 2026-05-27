from __future__ import annotations

import json
from pathlib import Path


_IMAGE_METADATA_PATH = Path("data/processed/image_metadata.json")

# Keyword-to-error mappings
_DEFECT_TO_ERRORS: list[tuple[list[str], list[str]]] = [
    (["overheat", "thermal", "smoke", "heat"],          ["E-101"]),
    (["pressure", "hydraulic"],                         ["E-102"]),
    (["vibration", "crack", "spindle", "alignment"],    ["E-103"]),
    (["lubrication", "oil", "gear"],                    ["E-104"]),
    (["leak", "coolant", "fluid"],                      ["E-105"]),
    (["corrosion", "rust", "pitting"],                  ["E-205"]),
    (["bearing", "wear"],                               ["E-204"]),
    (["sensor", "random"],                              ["E-201"]),
    (["power", "voltage", "current", "electrical"],     ["E-202"]),
    (["conveyor", "jam"],                               ["E-203"]),
]

# Keyword-to-component mappings
_COMPONENT_TO_IDS: list[tuple[list[str], list[str]]] = [
    (["hydraulic pump", "pump"],                        ["C-001"]),
    (["cooling fan", "fan motor"],                      ["C-002"]),
    (["bearing", "main bearing"],                       ["C-003"]),
    (["pressure relief valve", "pressure valve"],       ["C-005"]),
    (["lubrication pump", "lube"],                      ["C-007"]),
    (["coolant pump"],                                  ["C-008"]),
    (["pipe manifold", "manifold"],                     ["C-009"]),
    (["spindle"],                                       ["C-010"]),
    (["coolant tank", "tank", "reservoir"],             ["C-011"]),
    (["gear box", "gear"],                              ["C-012"]),
    (["vibration sensor", "sensor"],                    ["C-013"]),
    (["flow control valve"],                            ["C-014"]),
    (["motor controller", "controller", "motor"],       ["C-015"]),
]

# Error-to-manual mappings
_ERROR_TO_MANUALS: dict[str, list[str]] = {
    "E-101": ["M-003", "M-017", "M-019"],
    "E-102": ["M-001", "M-007", "M-016"],
    "E-103": ["M-013", "M-018"],
    "E-104": ["M-004", "M-014"],
    "E-105": ["M-003", "M-017"],
    "E-201": [],
    "E-202": [],
    "E-203": [],
    "E-204": ["M-002", "M-014"],
    "E-205": ["M-008", "M-011"],
}

# Error-to-component mappings
_ERROR_TO_COMPONENTS: dict[str, list[str]] = {
    "E-101": ["C-002", "C-008", "C-011"],
    "E-102": ["C-001", "C-005", "C-014"],
    "E-103": ["C-003", "C-010", "C-013"],
    "E-104": ["C-007", "C-003", "C-012"],
    "E-105": ["C-011", "C-001"],
    "E-201": ["C-008", "C-015"],
    "E-202": ["C-015"],
    "E-203": [],
    "E-204": ["C-003", "C-012"],
    "E-205": ["C-009"],
}


def _extract_text(image: dict) -> str:
    """Extract searchable text from image analysis metadata."""
    analysis = image.get("analysis", {})
    parts = [
        analysis.get("detected_component", ""),
        analysis.get("visual_defect", ""),
        analysis.get("description", ""),
        " ".join(analysis.get("visible_symptoms", [])),
        " ".join(analysis.get("visible_features", [])),
        " ".join(analysis.get("potential_issues", [])),
        analysis.get("depicted_component", ""),
        analysis.get("diagram_type", ""),
        " ".join(analysis.get("key_parts", [])),
        analysis.get("process_flow", ""),
        analysis.get("maintenance_relevance", ""),
        analysis.get("equipment_type", ""),
    ]
    return " ".join(p for p in parts if p).lower()


def _match_error_codes(text: str, existing: list[str]) -> list[str]:
    codes: list[str] = list(existing)
    for keywords, error_codes in _DEFECT_TO_ERRORS:
        if any(kw in text for kw in keywords):
            for ec in error_codes:
                if ec not in codes:
                    codes.append(ec)
    return codes


def _match_components(text: str, error_codes: list[str]) -> list[str]:
    components: list[str] = []

    for keywords, comp_ids in _COMPONENT_TO_IDS:
        if any(kw in text for kw in keywords):
            for cid in comp_ids:
                if cid not in components:
                    components.append(cid)

    for ec in error_codes:
        for cid in _ERROR_TO_COMPONENTS.get(ec, []):
            if cid not in components:
                components.append(cid)

    return components


def _match_manuals(error_codes: list[str]) -> list[str]:
    manuals: list[str] = []
    for ec in error_codes:
        for mid in _ERROR_TO_MANUALS.get(ec, []):
            if mid not in manuals:
                manuals.append(mid)
    return manuals


def patch_graph_links() -> None:
    images = json.loads(_IMAGE_METADATA_PATH.read_text(encoding="utf-8"))
    patched = 0

    for image in images:
        existing_links = image.get("graph_links", {})

        existing_errors = (
            image.get("analysis", {}).get("related_error_codes", [])
            or existing_links.get("related_error_codes", [])
        )

        text = _extract_text(image)

        error_codes = _match_error_codes(text, existing_errors)
        components = _match_components(text, error_codes)
        manuals = _match_manuals(error_codes)

        updated_links = {
            "related_error_codes": error_codes,
            "related_components": components,
            "related_incidents": existing_links.get("related_incidents", []),
            "related_manuals": manuals,
        }

        changed = updated_links != existing_links
        image["graph_links"] = updated_links

        if "analysis" in image:
            image["analysis"]["related_error_codes"] = error_codes

        if changed:
            patched += 1
            print(f"  ✓ {image['image_id']}")
            print(f"    errors     : {error_codes}")
            print(f"    components : {components}")
            print(f"    manuals    : {manuals}")

    _IMAGE_METADATA_PATH.write_text(
        json.dumps(images, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nCompleted: {patched} images updated / {len(images)} total")


if __name__ == "__main__":
    patch_graph_links()