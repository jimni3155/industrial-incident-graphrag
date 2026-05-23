"""분석 결과 정규화 및 graph link 생성"""

from __future__ import annotations

import json
from typing import Any
import re

from backend.multimodal.mappings import (
    DEFECT_KEYWORD_TO_ERROR,
    ERROR_TO_COMPONENTS,
    ERROR_TO_INCIDENTS,
    ERROR_TO_MANUALS,
    VALID_ERROR_CODES,
)


# 정규화
def normalize_error_codes(codes: Any) -> list[str]:
    if not isinstance(codes, list):
        return []
    return [c for c in (str(x).strip() for x in codes) if c in VALID_ERROR_CODES]


def normalize_detected_component(component: Any, category: str) -> str:
    raw = str(component or "").strip()
    if category != "defects":
        return raw

    normalized = raw.lower()
    for keyword, target in [
        ("flange",  "pipe flange"),
        ("pipe",    "pipe flange"),
        ("valve",   "valve"),
        ("pump",    "pump"),
        ("motor",   "motor"),
        ("bearing", "bearing"),
    ]:
        if keyword in normalized:
            return target
    return raw


def infer_error_codes(analysis: dict[str, Any], category: str) -> list[str]:
    """related_error_codes가 비었을 때 텍스트 기반으로 추론."""
    corpus = " ".join([
        str(analysis.get("visual_defect", "")),
        str(analysis.get("description", "")),
        str(analysis.get("detected_component", "")),
        *[str(s) for s in analysis.get("visible_symptoms", [])],
        *[str(s) for s in analysis.get("potential_issues", [])],
    ]).lower()

    inferred = []
    for keyword, code in DEFECT_KEYWORD_TO_ERROR:
        if keyword in corpus and code not in inferred:
            inferred.append(code)
    if inferred:
        return inferred

    component = str(analysis.get("detected_component", "")).lower()
    if "bearing" in component:
        return ["E-204"]
    if "motor" in component:
        return ["E-101"]
    if any(k in component for k in ("pump", "valve", "pipe", "flange")):
        return ["E-102"]
    return []


def resolve_graph_links(error_codes: list[str]) -> dict[str, list[str]]:
    components, incidents, manuals = [], [], []
    for code in error_codes:
        components.extend(ERROR_TO_COMPONENTS.get(code, []))
        incidents.extend(ERROR_TO_INCIDENTS.get(code, []))
        manuals.extend(ERROR_TO_MANUALS.get(code, []))
    return {
        "related_error_codes": error_codes,
        "related_components":  list(dict.fromkeys(components)),
        "related_incidents":   list(dict.fromkeys(incidents)),
        "related_manuals":     list(dict.fromkeys(manuals)),
    }


# needs_review 판단
_AMBIGUOUS = {"unknown", "other", "n/a", ""}


def needs_review(
    analysis: dict[str, Any],
    verification: dict[str, Any] | None = None,
) -> bool:
    if verification:
        if float(verification.get("confidence_score", 1.0)) < 0.6:
            return True
        if verification.get("issues"):
            return True

    if not analysis.get("related_error_codes"):
        return True

    component = str(analysis.get("detected_component", "")).lower().strip()
    return component in _AMBIGUOUS

# json 파싱
def parse_json_response(text: str | None) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("모델이 빈 응답 반환")

    text = text.strip()

    # 코드펜스 제거 (```json ... ``` 또는 ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise
