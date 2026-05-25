"""
AI4I 2020 predictive maintenance dataset 기반으로
- FailurePattern 노드용 통계 JSON (전체 340개 집계)
- Incident 노드용 대표 샘플 JSON (type별 센서값 구간 균등 샘플링)

실행: python -m data.generate_incidents_from_ai4i
입력: ai4i2020.csv 
출력:
  backend/data/processed/incidents.json
  backend/data/processed/failure_patterns.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "ai4i2020.csv"
OUT_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_MAP: dict[str, dict] = {
    "TWF": {"error_code": "E-204", "component": "C-003", "component_name": "Main Bearing",     "label": "Tool Wear Failure"},
    "HDF": {"error_code": "E-101", "component": "C-008", "component_name": "Coolant Pump",     "label": "Heat Dissipation Failure"},
    "PWF": {"error_code": "E-202", "component": "C-015", "component_name": "Motor Controller", "label": "Power Failure"},
    "OSF": {"error_code": "E-103", "component": "C-010", "component_name": "Spindle Assembly", "label": "Overstrain Failure"},
    "RNF": {"error_code": "E-201", "component": "C-013", "component_name": "Vibration Sensor", "label": "Random / Sensor Failure"},
}

SEVERITY_BY_ERROR = {
    "E-101": "high", "E-103": "high", "E-204": "high",
    "E-201": "medium", "E-202": "medium",
}

PRODUCT_TYPE = {"H": "High", "M": "Medium", "L": "Low"}

# type별 센서값 기반 구간 정의 → 각 구간에서 몇 개 뽑을지
# (partition_col, bins, labels, counts_per_bin)
SAMPLING_CONFIG: dict[str, dict] = {
    "TWF": {"col": "Tool wear [min]",         "bins": [0, 9999],             "labels": ["any"],              "per_bin": [5]},
    "HDF": {"col": "Process temperature [K]", "bins": [0, 309, 311, 9999],   "labels": ["low","mid","high"], "per_bin": [3, 3, 2]},
    "PWF": {"col": "Rotational speed [rpm]",  "bins": [0, 1500, 1800, 9999], "labels": ["low","mid","high"], "per_bin": [3, 3, 2]},
    "OSF": {"col": "Torque [Nm]",             "bins": [0, 45, 60, 9999],     "labels": ["low","mid","high"], "per_bin": [3, 3, 2]},
    "RNF": {"col": "Tool wear [min]",         "bins": [0, 9999],             "labels": ["any"],              "per_bin": [1]},
}


def primary_failure(row: pd.Series) -> str | None:
    for col in FAILURE_MAP:
        if row[col] == 1:
            return col
    return None


def sensor_values(row: pd.Series) -> dict:
    return {
        "air_temperature_c":     round(float(row["Air temperature [K]"]) - 273.15, 1),
        "process_temperature_c": round(float(row["Process temperature [K]"]) - 273.15, 1),
        "rotational_speed_rpm":  int(row["Rotational speed [rpm]"]),
        "torque_nm":             round(float(row["Torque [Nm]"]), 1),
        "tool_wear_min":         int(row["Tool wear [min]"]),
    }


def all_failure_types(row: pd.Series) -> list[str]:
    return [col for col in FAILURE_MAP if row[col] == 1]


def sample_incidents(failures: pd.DataFrame, rnf_rows: pd.DataFrame) -> list[dict]:
    failures = failures.copy()
    failures["_primary"] = failures.apply(primary_failure, axis=1)
    failures = failures[failures["_primary"].notna()]

    incidents = []
    seq = 1

    for ftype, cfg in SAMPLING_CONFIG.items():
        if ftype == "RNF":
            subset = rnf_rows.copy()
        else:
            subset = failures[failures["_primary"] == ftype].copy()
        if subset.empty:
            continue

        col      = cfg["col"]
        bins     = cfg["bins"]
        labels   = cfg["labels"]
        per_bin  = cfg["per_bin"]

        subset["_bin"] = pd.cut(subset[col], bins=bins, labels=labels, right=False)
        meta = FAILURE_MAP[ftype]

        for label, n in zip(labels, per_bin):
            bucket = subset[subset["_bin"] == label]
            sampled = bucket.sample(n=min(n, len(bucket)), random_state=42)
            for _, row in sampled.iterrows():
                sv   = sensor_values(row)
                ftypes = all_failure_types(row)
                affected = list({meta["component"]} | {FAILURE_MAP[f]["component"] for f in ftypes[1:]})
                ptype = PRODUCT_TYPE.get(str(row["Type"]), str(row["Type"]))

                incidents.append({
                    "id":                  f"INC-AI4I-{seq:04d}",
                    "source":              "AI4I2020",
                    "udi":                 int(row["UDI"]),
                    "product_id":          str(row["Product ID"]),
                    "product_type":        ptype,
                    "title":               f"{ptype} Quality — {meta['label']} ({label} range)",
                    "department":          "Manufacturing",
                    "occurred_at":         None,
                    "severity":            SEVERITY_BY_ERROR.get(meta["error_code"], "medium"),
                    "error_code":          meta["error_code"],
                    "failure_types":       ftypes,
                    "affected_components": affected,
                    "description": (
                        f"AI4I record UDI={int(row['UDI'])} ({ptype} quality). "
                        f"Primary failure: {meta['label']}. "
                        f"Sensor at failure — air={sv['air_temperature_c']}°C, "
                        f"process={sv['process_temperature_c']}°C, "
                        f"rpm={sv['rotational_speed_rpm']}, "
                        f"torque={sv['torque_nm']} Nm, "
                        f"tool_wear={sv['tool_wear_min']} min."
                    ),
                    "resolved":      True,
                    "resolution":    None,
                    "downtime_hours":None,
                    "symptoms": [
                        f"air_temp={sv['air_temperature_c']}°C",
                        f"process_temp={sv['process_temperature_c']}°C",
                        f"rpm={sv['rotational_speed_rpm']}",
                        f"torque={sv['torque_nm']} Nm",
                        f"tool_wear={sv['tool_wear_min']} min",
                    ],
                    "sensor_values": sv,
                })
                seq += 1

    return incidents


def build_failure_patterns(failures: pd.DataFrame, rnf_rows: pd.DataFrame) -> list[dict]:
    failures = failures.copy()
    failures["_primary"] = failures.apply(primary_failure, axis=1)
    failures = failures[failures["_primary"].notna()]

    patterns = []
    for ftype, meta in FAILURE_MAP.items():
        if ftype == "RNF":
            subset = rnf_rows.copy()
        else:
            subset = failures[failures["_primary"] == ftype]
        if subset.empty:
            continue

        def pct(col: str) -> float:
            return round(float(subset[col].mean()), 2)

        patterns.append({
            "pattern_id":              f"FP-{ftype}",
            "source":                  "AI4I2020",
            "failure_type":            ftype,
            "label":                   meta["label"],
            "error_code":              meta["error_code"],
            "primary_component":       meta["component"],
            "count":                   len(subset),
            "total_failures":          len(failures),
            "occurrence_rate":         round(len(subset) / len(failures), 3),
            "avg_air_temperature_c":   round(pct("Air temperature [K]") - 273.15, 2),
            "avg_process_temperature_c": round(pct("Process temperature [K]") - 273.15, 2),
            "avg_rotational_speed_rpm":pct("Rotational speed [rpm]"),
            "avg_torque_nm":           pct("Torque [Nm]"),
            "avg_tool_wear_min":       pct("Tool wear [min]"),
            "std_torque_nm":           round(float(subset["Torque [Nm]"].std()), 2),
            "std_tool_wear_min":       round(float(subset["Tool wear [min]"].std()), 2),
        })

    return patterns


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found — place ai4i2020.csv in project root")

    df       = pd.read_csv(CSV_PATH)
    failures = df[df["Machine failure"] == 1].copy()
    print(f"total rows: {len(df)}  failures: {len(failures)}")
    for col in FAILURE_MAP:
        print(f"  {col}: {int(failures[col].sum())}")

    rnf_rows  = df[df["RNF"] == 1].copy()
    incidents = sample_incidents(failures, rnf_rows)
    patterns  = build_failure_patterns(failures, rnf_rows)

    (OUT_DIR / "incidents.json").write_text(
        json.dumps(incidents, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "failure_patterns.json").write_text(
        json.dumps(patterns, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nincidents.json       {len(incidents)} sampled")
    print(f"failure_patterns.json {len(patterns)} patterns")


if __name__ == "__main__":
    main()
