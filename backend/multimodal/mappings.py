"""Error-code based graph linking rules for demo industrial dataset.

detected error code를 기존 graph node ID와 연결하기 위한 매핑 테이블이다.
각 ID는 components.json, incidents.json, manual_sections.json과 매칭되어야 한다.
"""

# error_code → component ID
ERROR_TO_COMPONENTS: dict[str, list[str]] = {
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

# error_code → incident ID
ERROR_TO_INCIDENTS: dict[str, list[str]] = {
    "E-101": ["INC-0003", "INC-0011"],
    "E-102": ["INC-0001", "INC-0013"],
    "E-103": ["INC-0005", "INC-0017"],
    "E-104": ["INC-0004", "INC-0008"],
    "E-105": ["INC-0006", "INC-0014"],
    "E-201": ["INC-0009", "INC-0022"],
    "E-202": ["INC-0010", "INC-0023"],
    "E-203": ["INC-0012", "INC-0016"],
    "E-204": ["INC-0002", "INC-0015"],
    "E-205": ["INC-0011", "INC-0019"],
}

# error_code → manual section ID
ERROR_TO_MANUALS: dict[str, list[str]] = {
    "E-101": ["M-003", "M-017", "M-019"],
    "E-102": ["M-001", "M-007", "M-016"],
    "E-103": ["M-018", "M-013"],
    "E-104": ["M-004", "M-014"],
    "E-105": ["M-003", "M-017"],
    "E-201": ["M-006", "M-015"],
    "E-202": ["M-009", "M-015"],
    "E-203": ["M-005", "M-012"],
    "E-204": ["M-002", "M-018"],
    "E-205": ["M-008", "M-011"],
}

# defect 텍스트 키워드 → error code 추론용
DEFECT_KEYWORD_TO_ERROR: list[tuple[str, str]] = [
    ("overheat",     "E-101"),
    ("hot",          "E-101"),
    ("temperature",  "E-101"),
    ("pressure",     "E-102"),
    ("leak",         "E-105"),
    ("coolant",      "E-105"),
    ("vibration",    "E-103"),
    ("crack",        "E-103"),
    ("bearing",      "E-204"),
    ("wear",         "E-204"),
    ("corrosion",    "E-205"),
    ("rust",         "E-205"),
    ("lubrication",  "E-104"),
    ("oil",          "E-104"),
]

VALID_ERROR_CODES: set[str] = set(ERROR_TO_COMPONENTS.keys())