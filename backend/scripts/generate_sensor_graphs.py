import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = "data/raw/images/sensor_graphs"
METADATA_PATH = os.path.join(OUTPUT_DIR, "sensor_graph_metadata.json")

GRAPHS = [
    dict(id=1,  type="overheat",      title="Bearing Temperature Rise",              ylabel="Temperature (°C)",     low=40,   high=65,   anomaly_type="spike",        sensor_type="temperature", component="Main Bearing",     incident="INC-0003"),
    dict(id=2,  type="overheat",      title="Coolant Temperature Anomaly",            ylabel="Temperature (°C)",     low=35,   high=55,   anomaly_type="gradual_rise",  sensor_type="temperature", component="Coolant Tank",      incident="INC-0011"),
    dict(id=3,  type="torque_spike",  title="Torque Fluctuation — Welding Station",   ylabel="Torque (Nm)",          low=30,   high=50,   anomaly_type="fluctuation",   sensor_type="torque",      component="Motor Controller",  incident="INC-0007"),
    dict(id=4,  type="torque_spike",  title="Overstrain Detected — Assembly Line",    ylabel="Torque (Nm)",          low=25,   high=45,   anomaly_type="spike",        sensor_type="torque",      component="Spindle Assembly",  incident="INC-0019"),
    dict(id=5,  type="rpm_drop",      title="Rotational Speed Drop — Motor",          ylabel="RPM",                  low=1400, high=1500, anomaly_type="drop",         sensor_type="rpm",         component="Motor Controller",  incident="INC-0002"),
    dict(id=6,  type="rpm_drop",      title="Spindle RPM Instability",                ylabel="RPM",                  low=2800, high=3200, anomaly_type="fluctuation",  sensor_type="rpm",         component="Spindle Assembly",  incident="INC-0024"),
    dict(id=7,  type="tool_wear",     title="Tool Wear Progression",                  ylabel="Wear (min)",           low=0,    high=150,  anomaly_type="gradual_rise",  sensor_type="wear",        component="Drive Belt",        incident="INC-0008"),
    dict(id=8,  type="tool_wear",     title="Tool Wear — Critical Zone",              ylabel="Wear (min)",           low=0,    high=200,  anomaly_type="gradual_rise",  sensor_type="wear",        component="Gear Box",          incident="INC-0015"),
    dict(id=9,  type="pressure_drop", title="Hydraulic Pressure Drop",                ylabel="Pressure (bar)",       low=120,  high=150,  anomaly_type="drop",         sensor_type="pressure",    component="Hydraulic Pump",    incident="INC-0001"),
    dict(id=10, type="pressure_drop", title="Air Compressor Pressure Loss",           ylabel="Pressure (bar)",       low=6,    high=8,    anomaly_type="decline",      sensor_type="pressure",    component="Air Compressor",    incident="INC-0013"),
    dict(id=11, type="vibration",     title="Bearing Vibration Before Failure",       ylabel="Vibration (mm/s RMS)", low=0.5,  high=2.5,  anomaly_type="gradual_rise",  sensor_type="vibration",   component="Main Bearing",      incident="INC-0005"),
    dict(id=12, type="vibration",     title="Gearbox Vibration Anomaly",              ylabel="Vibration (g)",        low=0.1,  high=0.5,  anomaly_type="spike",        sensor_type="vibration",   component="Gear Box",          incident="INC-0017"),
    dict(id=13, type="power",         title="Voltage Fluctuation — Control Panel",    ylabel="Voltage (V)",          low=380,  high=420,  anomaly_type="fluctuation",  sensor_type="voltage",     component="Control Panel",     incident="INC-0009"),
    dict(id=14, type="power",         title="Power Instability — Motor Controller",   ylabel="Current (A)",          low=10,   high=14,   anomaly_type="fluctuation",  sensor_type="current",     component="Motor Controller",  incident="INC-0022"),
    dict(id=15, type="coolant",       title="Coolant Level Decline",                  ylabel="Coolant Level (%)",    low=70,   high=100,  anomaly_type="decline",      sensor_type="level",       component="Coolant Tank",      incident="INC-0006"),
]

SEVERITY_MAP = {
    "spike": "high",
    "drop": "high",
    "gradual_rise": "medium",
    "decline": "medium",
    "fluctuation": "medium",
}


def apply_anomaly(signal, anomaly_type, low, high, anomaly_start, rng):
    n_anomaly = len(signal) - anomaly_start
    seg = signal[anomaly_start:].copy()

    if anomaly_type == "spike":
        peak_idx = n_anomaly // 3
        spike_val = high + rng.uniform(0.60, 0.80) * (high - low)
        for i in range(n_anomaly):
            if i <= peak_idx:
                seg[i] = low + (spike_val - low) * (i / peak_idx)
            else:
                recovery = low + (high - low) * 0.6
                frac = (i - peak_idx) / (n_anomaly - peak_idx)
                seg[i] = spike_val - (spike_val - recovery) * frac

    elif anomaly_type == "gradual_rise":
        target = high + rng.uniform(0.40, 0.70) * (high - low)
        for i in range(n_anomaly):
            frac = i / max(n_anomaly - 1, 1)
            seg[i] = seg[i] + (target - high) * frac

    elif anomaly_type == "drop":
        drop_val = low * rng.uniform(0.30, 0.40) if low > 0 else (high - low) * 0.10
        recovery = low * 0.7 if low > 0 else (high - low) * 0.20
        trough_idx = n_anomaly // 4
        for i in range(n_anomaly):
            if i <= trough_idx:
                seg[i] = seg[0] - (seg[0] - drop_val) * (i / trough_idx)
            else:
                frac = (i - trough_idx) / (n_anomaly - trough_idx)
                seg[i] = drop_val + (recovery - drop_val) * frac

    elif anomaly_type == "decline":
        floor = low - rng.uniform(0.15, 0.30) * (high - low)
        for i in range(n_anomaly):
            frac = i / max(n_anomaly - 1, 1)
            seg[i] = seg[i] - (seg[i] - floor) * frac

    elif anomaly_type == "fluctuation":
        swing = (high - low) * rng.uniform(0.30, 0.35)
        for i in range(n_anomaly):
            direction = 1 if i % 2 == 0 else -1
            seg[i] = seg[i] + direction * swing * rng.uniform(0.8, 1.2)

    signal[anomaly_start:] = seg
    return signal


def generate_graph(cfg):
    g_id       = cfg["id"]
    g_type     = cfg["type"]
    title      = cfg["title"]
    ylabel     = cfg["ylabel"]
    low        = cfg["low"]
    high       = cfg["high"]
    a_type     = cfg["anomaly_type"]
    sensor_type = cfg["sensor_type"]
    component  = cfg["component"]
    incident   = cfg["incident"]

    random.seed(g_id)
    np.random.seed(42)
    rng = np.random.default_rng(g_id)

    anomaly_start = random.randint(70, 90)

    times = pd.date_range(start="2026-01-15 08:00", periods=120, freq="5min")

    mid   = (low + high) / 2
    noise = (high - low) * 0.05
    signal = np.random.normal(mid, noise, 120)
    signal = np.clip(signal, low, high)

    signal = apply_anomaly(signal, a_type, low, high, anomaly_start, rng)

    anomaly_score = round(random.uniform(0.82, 0.97), 2)
    severity      = SEVERITY_MAP[a_type]

    filename  = f"graph_{g_type}_{g_id:03d}.png"
    out_path  = os.path.join(OUTPUT_DIR, filename)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.grid(color="#263238", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhspan(low, high, alpha=0.08, color="#4caf50", label="Normal range")
    ax.axvspan(times[anomaly_start], times[-1], alpha=0.12, color="#f44336")

    ax.plot(times[:anomaly_start], signal[:anomaly_start],
            color="#64b5f6", linewidth=1.5, label="Normal")
    ax.plot(times[anomaly_start:], signal[anomaly_start:],
            color="#ef5350", linewidth=1.8, label="Anomaly")

    ax.axhline(low,  color="#81c784", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(high, color="#81c784", linewidth=0.8, linestyle="--", alpha=0.7)

    annotation_idx = min(anomaly_start + 4, 119)
    ax.annotate(
        "⚠ ANOMALY DETECTED",
        xy=(times[anomaly_start], signal[anomaly_start]),
        xytext=(times[annotation_idx], signal[anomaly_start] * 1.06),
        color="#ffcc80", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#ffcc80", lw=0.8),
    )

    ax.text(0.98, 0.95, f"ANOMALY SCORE: {anomaly_score}",
            transform=ax.transAxes, ha="right", va="top",
            color="#ffcc80", fontsize=8)

    sensor_id = f"S-{g_id:03d} | {sensor_type.upper()}"
    fig.suptitle(sensor_id, fontsize=9, color="#90a4ae", y=0.98)

    ax.set_title(title, color="#eceff1", fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color="#b0bec5", fontsize=9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=30, color="#b0bec5", fontsize=8)
    plt.yticks(color="#b0bec5", fontsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#37474f")

    legend = ax.legend(loc="upper left", fontsize=8,
                       facecolor="#1e272e", edgecolor="#37474f")
    for text in legend.get_texts():
        text.set_color("#b0bec5")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    timestamp_anomaly = times[anomaly_start].isoformat()

    metadata = {
        "graph_id":             f"graph_{g_type}_{g_id:03d}",
        "file_path":            f"data/raw/images/sensor_graphs/{filename}",
        "graph_type":           g_type,
        "sensor_type":          sensor_type,
        "anomaly_type":         a_type,
        "anomaly_score":        anomaly_score,
        "anomaly_start_index":  anomaly_start,
        "related_incident":     incident,
        "related_component":    component,
        "severity":             severity,
        "timestamp_start":      "2026-01-15T08:00:00",
        "timestamp_anomaly":    timestamp_anomaly,
    }

    return out_path, metadata


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metadata = []

    for cfg in GRAPHS:
        out_path, metadata = generate_graph(cfg)
        all_metadata.append(metadata)
        print(f"[OK] {out_path}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n--- Summary ---")
    print(f"Generated : {len(all_metadata)} graphs")
    print(f"Metadata  : {METADATA_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
