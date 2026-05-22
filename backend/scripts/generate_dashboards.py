import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.dates as mdates
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "backend/data/dashboards"
METADATA_PATH = os.path.join(OUTPUT_DIR, "dashboard_metadata.json")

BG_MAIN   = "#111217"
BG_PANEL  = "#181b1f"
BG_CARD   = "#1f2329"
BORDER    = "#2c3038"
TEXT_PRI  = "#d0d5dd"
TEXT_SEC  = "#8b949e"
TEXT_DIM  = "#4d5562"
GREEN     = "#73bf69"
YELLOW    = "#f5a623"
RED       = "#f2495c"
BLUE      = "#5794f2"
PURPLE    = "#b877d9"
CYAN      = "#37872d"
ORANGE    = "#ff7f50"

def make_times(n=120, freq="5min", start="2026-01-15 06:00"):
    return pd.date_range(start=start, periods=n, freq=freq)

def smooth_signal(arr, w=5):
    return np.convolve(arr, np.ones(w)/w, mode="same")

def make_normal_signal(n, low, high):
    mid = (low + high) / 2
    sig = np.random.normal(mid, (high - low) * 0.05, n)
    return np.clip(sig, low * 0.95, high * 1.05)

def inject_anomaly(sig, anomaly_start, anomaly_type, low, high):
    s = sig.copy()
    n = len(s)
    for i in range(anomaly_start, n):
        p = (i - anomaly_start) / max(n - anomaly_start, 1)
        if anomaly_type == "spike":
            s[i] = high * (1.4 + p * 0.4) + np.random.normal(0, (high-low)*0.03)
        elif anomaly_type == "drop":
            s[i] = low * max(0.25, 1.0 - p * 0.75)
        elif anomaly_type == "gradual_rise":
            s[i] = sig[i] + p * (high - low) * 0.7
        elif anomaly_type == "decline":
            s[i] = sig[i] - p * (high - low) * 0.55
        elif anomaly_type == "fluctuation":
            s[i] = sig[i] + random.choice([-1,1]) * (high-low) * random.uniform(0.2, 0.38)
    return s

def set_grafana_axes(ax, times=None):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#23272e", linestyle="--", linewidth=0.5, alpha=0.8)
    if times is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, color=TEXT_SEC, fontsize=7)

def stat_card(ax, label, value, unit, color, status=None):
    ax.set_facecolor(BG_CARD)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.78, label, transform=ax.transAxes,
            ha="center", va="center", color=TEXT_SEC, fontsize=8)
    ax.text(0.5, 0.42, f"{value}", transform=ax.transAxes,
            ha="center", va="center", color=color, fontsize=22, fontweight="bold")
    ax.text(0.5, 0.14, unit, transform=ax.transAxes,
            ha="center", va="center", color=TEXT_DIM, fontsize=7)
    if status:
        s_color = RED if status == "ALERT" else (YELLOW if status == "WARN" else GREEN)
        ax.text(0.93, 0.93, f"● {status}", transform=ax.transAxes,
                ha="right", va="top", color=s_color, fontsize=6.5)

def header_bar(fig, title, subtitle, time_str):
    fig.text(0.012, 0.968, title, color=TEXT_PRI, fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.012, 0.945, subtitle, color=TEXT_SEC, fontsize=8, va="top")
    fig.text(0.988, 0.968, time_str, color=TEXT_DIM, fontsize=7.5,
             va="top", ha="right")
    fig.text(0.988, 0.952, "● LIVE", color=GREEN, fontsize=7,
             va="top", ha="right")


#  Assembly Line Overview
def dashboard_assembly_line():
    fig = plt.figure(figsize=(16, 9), facecolor=BG_MAIN)
    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        left=0.04, right=0.98, top=0.91, bottom=0.06,
        hspace=0.52, wspace=0.35
    )

    header_bar(fig,
               "Assembly Line 1 — Equipment Monitor",
               "Dept: Manufacturing / Station: AL-01 / Shift: Day",
               "2026-01-15  14:37:22")

    times = make_times(120)
    anom = 82

    # stat cards (row 0, col 0-4)
    cards = [
        ("Hydraulic Pressure", "138.4", "bar",   BLUE,   "OK"),
        ("Oil Temperature",    "71.2",  "°C",    YELLOW, "WARN"),
        ("RPM — Motor",        "1,487", "rpm",   GREEN,  "OK"),
        ("Torque",             "43.8",  "Nm",    ORANGE, "OK"),
        ("Vibration",          "3.21",  "mm/s",  RED,    "ALERT"),
    ]
    for col, (lbl, val, unit, col_, st) in enumerate(cards):
        ax = fig.add_subplot(gs[0, col])
        stat_card(ax, lbl, val, unit, col_, st)

    # Hydraulic pressure time series
    ax1 = fig.add_subplot(gs[1, :3])
    sig = make_normal_signal(120, 120, 150)
    sig = inject_anomaly(sig, anom, "drop", 120, 150)
    ax1.fill_between(times, 120, 150, alpha=0.06, color=GREEN)
    ax1.fill_between(times[anom:], sig[anom:], 120,
                     alpha=0.15, color=RED)
    ax1.plot(times[:anom], sig[:anom], color=BLUE, lw=1.4, label="Pressure")
    ax1.plot(times[anom:], sig[anom:], color=RED,  lw=1.6)
    ax1.axhline(120, color=YELLOW, lw=0.8, ls="--", alpha=0.7)
    ax1.axhline(150, color=YELLOW, lw=0.8, ls="--", alpha=0.7)
    ax1.annotate("⚠ PRESSURE DROP", xy=(times[anom], sig[anom]),
                 xytext=(times[anom+6], sig[anom]-12),
                 color=YELLOW, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.8))
    ax1.set_ylabel("Pressure (bar)", color=TEXT_SEC, fontsize=8)
    ax1.set_title("Hydraulic Pressure — C-001", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    set_grafana_axes(ax1, times)

    # Vibration
    ax2 = fig.add_subplot(gs[1, 3:])
    sig2 = make_normal_signal(120, 0.5, 2.5)
    sig2 = inject_anomaly(sig2, 75, "gradual_rise", 0.5, 2.5)
    ax2.fill_between(times, 0.5, 2.5, alpha=0.06, color=GREEN)
    ax2.plot(times[:75], sig2[:75], color=PURPLE, lw=1.4)
    ax2.plot(times[75:],  sig2[75:],  color=RED,    lw=1.6)
    ax2.axhline(2.5, color=RED, lw=0.8, ls="--", alpha=0.7)
    ax2.set_ylabel("mm/s RMS", color=TEXT_SEC, fontsize=8)
    ax2.set_title("Vibration — Main Bearing", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    set_grafana_axes(ax2, times)

    # Torque
    ax3 = fig.add_subplot(gs[2, :2])
    sig3 = make_normal_signal(120, 30, 50)
    sig3 = inject_anomaly(sig3, 88, "fluctuation", 30, 50)
    ax3.plot(times, sig3, color=ORANGE, lw=1.2)
    ax3.fill_between(times, 30, 50, alpha=0.05, color=GREEN)
    ax3.set_ylabel("Torque (Nm)", color=TEXT_SEC, fontsize=8)
    ax3.set_title("Torque — Motor Controller", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    set_grafana_axes(ax3, times)

    # Oil temp
    ax4 = fig.add_subplot(gs[2, 2:4])
    sig4 = make_normal_signal(120, 40, 65)
    sig4 = inject_anomaly(sig4, 85, "spike", 40, 65)
    ax4.plot(times[:85], sig4[:85], color=CYAN, lw=1.4)
    ax4.plot(times[85:],  sig4[85:],  color=RED,  lw=1.6)
    ax4.axhline(65, color=RED, lw=0.8, ls="--", alpha=0.7)
    ax4.set_ylabel("°C", color=TEXT_SEC, fontsize=8)
    ax4.set_title("Oil Temperature — Lubrication", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    set_grafana_axes(ax4, times)

    #  Alert log
    ax5 = fig.add_subplot(gs[2, 4])
    ax5.set_facecolor(BG_CARD)
    for sp in ax5.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.8)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.set_title("Recent Alerts", color=TEXT_PRI, fontsize=9, loc="left", pad=6)
    alerts = [
        ("14:31", "ALERT",  "Vibration > 3.0"),
        ("14:19", "WARN",   "Oil Temp rise"),
        ("13:55", "ALERT",  "Pressure drop"),
        ("13:22", "INFO",   "Maintenance due"),
        ("12:48", "WARN",   "RPM fluctuation"),
    ]
    for i, (t, lvl, msg) in enumerate(alerts):
        c = RED if lvl == "ALERT" else (YELLOW if lvl == "WARN" else BLUE)
        y = 0.88 - i * 0.18
        ax5.text(0.03, y, f"[{t}]", transform=ax5.transAxes,
                 color=TEXT_DIM, fontsize=6.5, va="top")
        ax5.text(0.32, y, lvl, transform=ax5.transAxes,
                 color=c, fontsize=6.5, va="top", fontweight="bold")
        ax5.text(0.60, y, msg, transform=ax5.transAxes,
                 color=TEXT_SEC, fontsize=6.5, va="top")

    fname = "dashboard_assembly_line_001.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"[OK] {fname}")
    return {
        "dashboard_id": "dashboard_assembly_line_001",
        "file_path": f"backend/data/dashboards/{fname}",
        "dashboard_type": "assembly_line_overview",
        "related_incidents": ["INC-0001", "INC-0003", "INC-0007"],
        "anomaly_panels": ["Hydraulic Pressure", "Vibration", "Oil Temperature"],
        "severity": "high",
        "timestamp": "2026-01-15T14:37:22"
    }


#  Bearing Failure Analysis
def dashboard_bearing_failure():
    fig = plt.figure(figsize=(16, 9), facecolor=BG_MAIN)
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        left=0.05, right=0.98, top=0.91, bottom=0.06,
        hspace=0.5, wspace=0.35
    )

    header_bar(fig,
               "Bearing Health Monitor — Predictive Maintenance",
               "Component: Main Bearing C-003 / Location: Welding Station",
               "2026-01-15  09:14:05")

    times = make_times(120, start="2026-01-15 03:00")

    # stat cards
    cards = [
        ("Vibration RMS",  "4.87", "mm/s", RED,    "ALERT"),
        ("Temperature",    "78.3", "°C",   RED,    "ALERT"),
        ("RPM",           "1,442", "rpm",  YELLOW, "WARN"),
        ("Health Score",   "23%",  "",     RED,    "CRITICAL"),
    ]
    for col, (lbl, val, unit, col_, st) in enumerate(cards):
        ax = fig.add_subplot(gs[0, col])
        stat_card(ax, lbl, val, unit, col_, st)

    # vibration spectrum (simulated freq domain feel)
    ax1 = fig.add_subplot(gs[1, :2])
    sig = make_normal_signal(120, 0.3, 2.0)
    sig = inject_anomaly(sig, 70, "gradual_rise", 0.3, 2.0)
    ax1.fill_between(times, 0, sig, where=(np.arange(120) < 70),
                     alpha=0.3, color=BLUE)
    ax1.fill_between(times, 0, sig, where=(np.arange(120) >= 70),
                     alpha=0.35, color=RED)
    ax1.plot(times[:70], sig[:70], color=BLUE, lw=1.4)
    ax1.plot(times[70:],  sig[70:],  color=RED,  lw=1.7)
    ax1.axhline(2.0, color=RED, lw=0.9, ls="--", label="Alarm limit")
    ax1.axhline(1.5, color=YELLOW, lw=0.8, ls="--", label="Warning limit")
    ax1.set_title("Vibration Trend — 12h Window", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax1.set_ylabel("mm/s RMS", color=TEXT_SEC, fontsize=8)
    ax1.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BORDER,
               labelcolor=TEXT_SEC)
    set_grafana_axes(ax1, times)

    # temperature
    ax2 = fig.add_subplot(gs[1, 2:])
    sig2 = make_normal_signal(120, 40, 65)
    sig2 = inject_anomaly(sig2, 75, "spike", 40, 65)
    ax2.fill_between(times, sig2, 40, where=(np.arange(120) >= 75),
                     alpha=0.2, color=RED)
    ax2.plot(times[:75], sig2[:75], color=CYAN,  lw=1.4)
    ax2.plot(times[75:],  sig2[75:],  color=RED,   lw=1.7)
    ax2.axhline(70, color=RED,    lw=0.9, ls="--", alpha=0.8, label="Critical")
    ax2.axhline(60, color=YELLOW, lw=0.8, ls="--", alpha=0.7, label="Warning")
    ax2.set_title("Bearing Temperature", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax2.set_ylabel("°C", color=TEXT_SEC, fontsize=8)
    ax2.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BORDER,
               labelcolor=TEXT_SEC)
    set_grafana_axes(ax2, times)

    # health score decay
    ax3 = fig.add_subplot(gs[2, :2])
    health = np.linspace(95, 23, 120) + np.random.normal(0, 1.5, 120)
    health = np.clip(health, 0, 100)
    colors_h = [RED if v < 30 else (YELLOW if v < 60 else GREEN) for v in health]
    ax3.scatter(times, health, c=colors_h, s=4, zorder=3)
    ax3.plot(times, smooth_signal(health, 8), color=PURPLE, lw=1.5, zorder=2)
    ax3.axhline(30, color=RED,    lw=0.8, ls="--", alpha=0.7)
    ax3.axhline(60, color=YELLOW, lw=0.8, ls="--", alpha=0.7)
    ax3.set_ylim(0, 110)
    ax3.set_title("Health Score Decay", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax3.set_ylabel("Health (%)", color=TEXT_SEC, fontsize=8)
    set_grafana_axes(ax3, times)

    # RPM drop
    ax4 = fig.add_subplot(gs[2, 2:])
    sig4 = make_normal_signal(120, 1400, 1500)
    sig4 = inject_anomaly(sig4, 80, "drop", 1400, 1500)
    ax4.fill_between(times, 1400, 1500, alpha=0.06, color=GREEN)
    ax4.plot(times[:80], sig4[:80], color=GREEN, lw=1.4)
    ax4.plot(times[80:],  sig4[80:],  color=RED,   lw=1.7)
    ax4.axhline(1400, color=YELLOW, lw=0.8, ls="--", alpha=0.7)
    ax4.set_title("Rotational Speed", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax4.set_ylabel("RPM", color=TEXT_SEC, fontsize=8)
    set_grafana_axes(ax4, times)

    fname = "dashboard_bearing_failure_002.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"[OK] {fname}")
    return {
        "dashboard_id": "dashboard_bearing_failure_002",
        "file_path": f"backend/data/dashboards/{fname}",
        "dashboard_type": "bearing_failure_analysis",
        "related_incidents": ["INC-0003", "INC-0005", "INC-0017"],
        "anomaly_panels": ["Vibration", "Temperature", "Health Score", "RPM"],
        "severity": "high",
        "timestamp": "2026-01-15T09:14:05"
    }



# Multi-Equipment Anomaly Alert
def dashboard_anomaly_alert():
    fig = plt.figure(figsize=(16, 9), facecolor=BG_MAIN)
    gs = gridspec.GridSpec(
        3, 6, figure=fig,
        left=0.04, right=0.98, top=0.91, bottom=0.06,
        hspace=0.55, wspace=0.4
    )

    header_bar(fig,
               "⚠  Multi-Equipment Anomaly Alert — Shift Report",
               "Dept: Manufacturing / Date: 2026-01-15 / Alert Level: HIGH",
               "2026-01-15  16:52:10")

    times = make_times(60, freq="10min", start="2026-01-15 06:00")

    # stat cards
    cards = [
        ("Active Alerts",  "7",      "incidents", RED,    "ALERT"),
        ("Downtime",       "2.4h",   "today",     YELLOW, "WARN"),
        ("Equip Online",   "11/15",  "units",     YELLOW, "WARN"),
        ("MTBF",           "312h",   "avg",       BLUE,   "OK"),
        ("Maintenance Due","3",      "units",     ORANGE, "WARN"),
        ("Shift OEE",      "71.3%",  "",          YELLOW, "WARN"),
    ]
    for col, (lbl, val, unit, col_, st) in enumerate(cards):
        ax = fig.add_subplot(gs[0, col])
        stat_card(ax, lbl, val, unit, col_, st)

    equip = [
        ("Hydraulic Pump C-001",  (120,150), "pressure", "drop",       BLUE,   78),
        ("Main Bearing C-003",    (0.5,2.5), "vibration","gradual_rise",PURPLE, 55),
        ("Cooling Fan C-002",     (1400,1500),"rpm",     "drop",        CYAN,   85),
        ("Conveyor Belt C-004",   (18,22),   "speed",    "fluctuation", ORANGE, 88),
    ]
    ylabels = ["bar", "mm/s", "RPM", "m/min"]

    for i, ((name, rng, _, atype, color, anom_i), ylabel) in enumerate(zip(equip, ylabels)):
        row = 1 + i // 2
        col_start = (i % 2) * 3
        ax = fig.add_subplot(gs[row, col_start:col_start+3])

        sig = make_normal_signal(60, rng[0], rng[1])
        sig = inject_anomaly(sig, anom_i, atype, rng[0], rng[1])
        anom_idx = min(anom_i, 59)

        ax.fill_between(times, rng[0], rng[1], alpha=0.06, color=GREEN)
        ax.plot(times[:anom_idx], sig[:anom_idx], color=color, lw=1.4)
        if anom_idx < 60:
            ax.plot(times[anom_idx:], sig[anom_idx:], color=RED, lw=1.6)
            ax.axvspan(times[anom_idx], times[-1], alpha=0.08, color=RED)

        ax.axhline(rng[0], color=YELLOW, lw=0.7, ls="--", alpha=0.6)
        ax.axhline(rng[1], color=YELLOW, lw=0.7, ls="--", alpha=0.6)
        ax.set_title(name, color=TEXT_PRI, fontsize=8.5, loc="left", pad=5)
        ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=7.5)
        set_grafana_axes(ax, times)

    fname = "dashboard_anomaly_alert_003.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"[OK] {fname}")
    return {
        "dashboard_id": "dashboard_anomaly_alert_003",
        "file_path": f"backend/data/dashboards/{fname}",
        "dashboard_type": "multi_equipment_anomaly_alert",
        "related_incidents": ["INC-0001","INC-0002","INC-0005","INC-0009","INC-0013"],
        "anomaly_panels": ["Hydraulic Pump","Main Bearing","Cooling Fan","Conveyor Belt"],
        "severity": "high",
        "timestamp": "2026-01-15T16:52:10"
    }


# Predictive Maintenance Timeline
def dashboard_predictive_maintenance():
    fig = plt.figure(figsize=(16, 9), facecolor=BG_MAIN)
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        left=0.05, right=0.98, top=0.91, bottom=0.06,
        hspace=0.52, wspace=0.38
    )

    header_bar(fig,
               "Predictive Maintenance — Tool Wear & Degradation Monitor",
               "AI4I Predictive Model / Confidence: 94.2% / Next maintenance in: 6.3h",
               "2026-01-15  11:28:44")

    times = make_times(120, start="2026-01-15 00:00")

    # stat cards
    cards = [
        ("Tool Wear",     "187min",  "of 200 limit", RED,    "ALERT"),
        ("RUL Estimate",  "6.3h",    "remaining",    YELLOW, "WARN"),
        ("Torque Trend",  "+18.4%",  "vs baseline",  YELLOW, "WARN"),
        ("Pred. Failure", "94.2%",   "probability",  RED,    "ALERT"),
    ]
    for col, (lbl, val, unit, col_, st) in enumerate(cards):
        ax = fig.add_subplot(gs[0, col])
        stat_card(ax, lbl, val, unit, col_, st)

    # tool wear progression
    ax1 = fig.add_subplot(gs[1, :2])
    wear = np.linspace(0, 187, 120) + np.random.normal(0, 2, 120)
    wear = np.clip(wear, 0, 210)
    ax1.fill_between(times, 0, 150, alpha=0.06, color=GREEN, label="Safe zone")
    ax1.fill_between(times, 150, 200, alpha=0.06, color=YELLOW, label="Warning zone")
    ax1.fill_between(times, 200, 220, alpha=0.06, color=RED, label="Critical zone")
    ax1.plot(times, smooth_signal(wear, 5), color=ORANGE, lw=1.8)
    ax1.axhline(150, color=YELLOW, lw=0.8, ls="--")
    ax1.axhline(200, color=RED,    lw=0.9, ls="--")
    ax1.set_ylim(0, 220)
    ax1.set_title("Tool Wear Progression (min)", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax1.set_ylabel("Wear (min)", color=TEXT_SEC, fontsize=8)
    ax1.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BORDER,
               labelcolor=TEXT_SEC, loc="upper left")
    set_grafana_axes(ax1, times)

    # failure probability
    ax2 = fig.add_subplot(gs[1, 2:])
    prob = 1 / (1 + np.exp(-0.07 * (np.arange(120) - 60)))
    prob += np.random.normal(0, 0.015, 120)
    prob = np.clip(prob, 0, 1) * 100
    prob_smooth = smooth_signal(prob, 6)
    ax2.fill_between(times, prob_smooth, alpha=0.25,
                     color=[RED if p > 70 else (YELLOW if p > 40 else BLUE)
                            for p in prob_smooth])
    ax2.plot(times, prob_smooth, color=RED, lw=1.6)
    ax2.axhline(70, color=RED,    lw=0.8, ls="--", label="Alert threshold")
    ax2.axhline(40, color=YELLOW, lw=0.8, ls="--", label="Warn threshold")
    ax2.set_ylim(0, 110)
    ax2.set_title("Failure Probability (%)", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax2.set_ylabel("Probability (%)", color=TEXT_SEC, fontsize=8)
    ax2.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BORDER,
               labelcolor=TEXT_SEC)
    set_grafana_axes(ax2, times)

    # torque trend
    ax3 = fig.add_subplot(gs[2, :2])
    sig3 = make_normal_signal(120, 30, 50)
    sig3 = inject_anomaly(sig3, 80, "gradual_rise", 30, 50)
    ax3.plot(times, smooth_signal(sig3, 4), color=ORANGE, lw=1.5)
    ax3.fill_between(times, 30, 50, alpha=0.06, color=GREEN)
    ax3.axhline(50, color=YELLOW, lw=0.8, ls="--", alpha=0.7)
    ax3.set_title("Torque — Gradual Increase", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax3.set_ylabel("Torque (Nm)", color=TEXT_SEC, fontsize=8)
    set_grafana_axes(ax3, times)

    # RUL bar chart
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.set_facecolor(BG_PANEL)
    components = ["Drive Belt", "Main Bearing", "Hydraulic Pump",
                  "Spindle Assy", "Gear Box"]
    rul_hours  = [6.3, 18.7, 42.1, 65.4, 88.2]
    bar_colors = [RED if r < 10 else (YELLOW if r < 30 else GREEN)
                  for r in rul_hours]
    bars = ax4.barh(components, rul_hours, color=bar_colors,
                    alpha=0.8, edgecolor=BORDER, height=0.55)
    for bar, val in zip(bars, rul_hours):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val}h", va="center", color=TEXT_SEC, fontsize=7.5)
    ax4.axvline(24, color=YELLOW, lw=0.8, ls="--", alpha=0.7)
    ax4.set_xlim(0, 110)
    ax4.set_title("Remaining Useful Life (RUL)", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax4.set_xlabel("Hours", color=TEXT_SEC, fontsize=8)
    for sp in ax4.spines.values():
        sp.set_color(BORDER)
    ax4.tick_params(colors=TEXT_SEC, labelsize=7.5)
    ax4.grid(axis="x", color="#23272e", linestyle="--", lw=0.5, alpha=0.8)

    fname = "dashboard_predictive_maintenance_004.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"[OK] {fname}")
    return {
        "dashboard_id": "dashboard_predictive_maintenance_004",
        "file_path": f"backend/data/dashboards/{fname}",
        "dashboard_type": "predictive_maintenance_timeline",
        "related_incidents": ["INC-0008","INC-0015","INC-0019"],
        "anomaly_panels": ["Tool Wear","Failure Probability","Torque","RUL"],
        "severity": "high",
        "timestamp": "2026-01-15T11:28:44"
    }



# Coolant & Thermal Overview
def dashboard_thermal_coolant():
    fig = plt.figure(figsize=(16, 9), facecolor=BG_MAIN)
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        left=0.05, right=0.98, top=0.91, bottom=0.06,
        hspace=0.52, wspace=0.38
    )

    header_bar(fig,
               "Thermal & Coolant System Monitor — Assembly Line 2",
               "Component Group: Cooling / Dept: Manufacturing / Station: AL-02",
               "2026-01-15  13:05:31")

    times = make_times(120, start="2026-01-15 07:00")

    cards = [
        ("Coolant Level",  "54%",   "of 100%",  RED,    "ALERT"),
        ("Coolant Temp",   "62.8",  "°C",       YELLOW, "WARN"),
        ("Air Temp",       "34.1",  "°C",       ORANGE, "WARN"),
        ("Process Temp",   "311K",  "",         BLUE,   "OK"),
    ]
    for col, (lbl, val, unit, col_, st) in enumerate(cards):
        ax = fig.add_subplot(gs[0, col])
        stat_card(ax, lbl, val, unit, col_, st)

    # coolant level
    ax1 = fig.add_subplot(gs[1, :2])
    level = np.linspace(100, 54, 120) + np.random.normal(0, 0.8, 120)
    level = np.clip(level, 0, 105)
    ax1.fill_between(times, 70, 100, alpha=0.07, color=GREEN, label="Normal")
    ax1.fill_between(times, 40, 70,  alpha=0.07, color=YELLOW)
    ax1.fill_between(times, 0,  40,  alpha=0.07, color=RED)
    ax1.plot(times, smooth_signal(level, 6), color=BLUE, lw=1.8)
    ax1.axhline(70, color=YELLOW, lw=0.8, ls="--", alpha=0.7)
    ax1.axhline(40, color=RED,    lw=0.9, ls="--", alpha=0.8)
    ax1.set_ylim(0, 115)
    ax1.set_title("Coolant Level Decline — C-011", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax1.set_ylabel("Level (%)", color=TEXT_SEC, fontsize=8)
    set_grafana_axes(ax1, times)

    # air vs process temp
    ax2 = fig.add_subplot(gs[1, 2:])
    air_t = make_normal_signal(120, 25, 35)
    air_t = inject_anomaly(air_t, 85, "spike", 25, 35)
    proc_t = make_normal_signal(120, 300, 320)
    proc_t = inject_anomaly(proc_t, 85, "gradual_rise", 300, 320)

    ax2b = ax2.twinx()
    ax2.plot(times, air_t,  color=ORANGE, lw=1.4, label="Air Temp (°C)")
    ax2b.plot(times, proc_t, color=CYAN,   lw=1.4, ls="--",
              label="Process Temp (K)")
    ax2.set_ylabel("Air Temp (°C)", color=ORANGE, fontsize=8)
    ax2b.set_ylabel("Process Temp (K)", color=CYAN, fontsize=8)
    ax2b.tick_params(colors=TEXT_SEC, labelsize=7)
    ax2b.spines["right"].set_color(BORDER)
    ax2b.spines["top"].set_visible(False)
    ax2.set_title("Thermal Comparison", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labs1+labs2, fontsize=7,
               facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_SEC)
    set_grafana_axes(ax2, times)

    # coolant temp
    ax3 = fig.add_subplot(gs[2, :2])
    sig3 = make_normal_signal(120, 35, 55)
    sig3 = inject_anomaly(sig3, 80, "gradual_rise", 35, 55)
    ax3.plot(times[:80], sig3[:80], color=CYAN, lw=1.4)
    ax3.plot(times[80:],  sig3[80:],  color=RED,  lw=1.7)
    ax3.fill_between(times, 35, 55, alpha=0.06, color=GREEN)
    ax3.axhline(55, color=RED, lw=0.9, ls="--", alpha=0.8)
    ax3.set_title("Coolant Temperature Rise", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    ax3.set_ylabel("°C", color=TEXT_SEC, fontsize=8)
    set_grafana_axes(ax3, times)

    # maintenance log
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.set_facecolor(BG_CARD)
    for sp in ax4.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.8)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_title("Maintenance Log", color=TEXT_PRI,
                  fontsize=9, loc="left", pad=6)
    logs = [
        ("2026-01-08", "Coolant flush completed"),
        ("2026-01-10", "Coolant level: 98%"),
        ("2026-01-12", "Minor leak detected"),
        ("2026-01-13", "Temporary seal applied"),
        ("2026-01-15", "⚠ Level at 54% — refill required"),
    ]
    for i, (date, note) in enumerate(logs):
        y = 0.88 - i * 0.17
        c = RED if "⚠" in note else TEXT_SEC
        ax4.text(0.03, y, date, transform=ax4.transAxes,
                 color=TEXT_DIM, fontsize=6.5, va="top")
        ax4.text(0.38, y, note, transform=ax4.transAxes,
                 color=c, fontsize=6.5, va="top")

    fname = "dashboard_thermal_coolant_005.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"[OK] {fname}")
    return {
        "dashboard_id": "dashboard_thermal_coolant_005",
        "file_path": f"backend/data/dashboards/{fname}",
        "dashboard_type": "thermal_coolant_overview",
        "related_incidents": ["INC-0006","INC-0011","INC-0017"],
        "anomaly_panels": ["Coolant Level","Air Temp","Process Temp","Coolant Temp"],
        "severity": "medium",
        "timestamp": "2026-01-15T13:05:31"
    }



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output → {OUTPUT_DIR}\n")

    generators = [
        dashboard_assembly_line,
        dashboard_bearing_failure,
        dashboard_anomaly_alert,
        dashboard_predictive_maintenance,
        dashboard_thermal_coolant,
    ]

    metadata_all = []
    for gen in generators:
        meta = gen()
        metadata_all.append(meta)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_all, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] dashboard_metadata.json — {len(metadata_all)} entries")
    print(f"\nAll done. {len(metadata_all)} dashboards → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()