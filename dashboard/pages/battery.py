"""
PolarGrid AI — Dashboard Battery Storage Page v2
BESS analytics: SOC telemetry, charge/discharge dynamics, cycle health, and cold derating.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.glass_theme import apply_glass_theme, render_header, render_section_header
from dashboard.components.kpi_cards import render_kpi_row, render_status_badge
from dashboard.components.charts import battery_soc_chart, LAYOUT_BASE, COLORS

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIGS = PROJECT_ROOT / "configs"


from dashboard.utils.data_loader import load_master_dataset, load_system_configs


def load_battery_data():
    master = load_master_dataset()
    battery_cfg = load_system_configs().get("battery", {})
    return master, battery_cfg


def _charge_discharge_bar(df: pd.DataFrame, capacity_kwh: float, height: int = 320) -> go.Figure:
    """Daily charge vs discharge energy bar chart."""
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
    else:
        df = df.copy()
        df["date"] = pd.RangeIndex(len(df)) // 24

    soc_diff = df["battery_soc"].diff().fillna(0)
    df["charge_kwh"]    = (soc_diff.clip(lower=0) * capacity_kwh).values
    df["discharge_kwh"] = ((-soc_diff).clip(lower=0) * capacity_kwh).values

    daily = df.groupby("date")[["charge_kwh", "discharge_kwh"]].sum().reset_index()
    daily = daily.tail(30)  # last 30 days

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"].astype(str), y=daily["charge_kwh"],
        name="🔋 Charged", marker_color="rgba(52,211,153,0.75)",
        marker_line=dict(color="rgba(52,211,153,0.9)", width=1),
    ))
    fig.add_trace(go.Bar(
        x=daily["date"].astype(str), y=-daily["discharge_kwh"],
        name="🔌 Discharged", marker_color="rgba(167,139,250,0.70)",
        marker_line=dict(color="rgba(167,139,250,0.9)", width=1),
    ))
    fig.update_layout(
        barmode="overlay",
        height=height,
        title=dict(text="Daily Charge / Discharge Energy (kWh)", font=dict(size=14, color="#e2e8f0")),
        **LAYOUT_BASE,
    )
    fig.update_yaxes(title_text="Energy (kWh)")
    return fig


def render_battery_page():
    apply_glass_theme()
    render_header(
        title="Battery Energy Storage System (BESS)",
        subtitle="500 kWh Lithium-Ion polar battery bank — SOC telemetry, sub-zero derating, and cycle health analytics",
        badge="STORAGE • BESS TELEMETRY",
    )

    master_df, battery_cfg = load_battery_data()

    window_options = {
        "Last 7 Days (168h)":      168,
        "Last 30 Days (720h)":     720,
        "Last 90 Days (2,160h)": 2160,
        "Full 14-Year Record":   len(master_df),
    }
    col_win, _ = st.columns([1, 2])
    with col_win:
        window = st.selectbox("Inspection Horizon", list(window_options.keys()), index=0)
    n_hours  = window_options[window]
    display_df = master_df.tail(n_hours).copy()

    # ── Compute metrics ────────────────────────────────────────────────────
    soc      = display_df["battery_soc"].values
    avg_soc  = float(np.mean(soc))
    min_soc  = float(np.min(soc))
    max_soc  = float(np.max(soc))

    soc_diffs     = pd.Series(soc).diff().fillna(0)
    charge_energy  = float(soc_diffs[soc_diffs > 0].sum()  * battery_cfg["capacity_kwh"])
    discharge_energy = float(abs(soc_diffs[soc_diffs < 0].sum()) * battery_cfg["capacity_kwh"])
    throughput     = charge_energy + discharge_energy
    usable_cap     = battery_cfg["capacity_kwh"] * (battery_cfg["soc_max"] - battery_cfg["soc_min"])
    eq_cycles      = throughput / (2.0 * usable_cap) if usable_cap > 0 else 0.0

    cold_threshold = battery_cfg["cold_derating_threshold_c"]
    cold_hours     = int((display_df["temperature"] < cold_threshold).sum())
    cold_pct       = cold_hours / max(1, len(display_df)) * 100.0

    # ── KPI Cards ──────────────────────────────────────────────────────────
    render_section_header("Storage Capacity & Health Metrics",
                          "Battery state of charge, cumulative throughput, and cycle count", icon="🔋")

    render_kpi_row([
        {"title": "Mean Battery SOC",
         "value": f"{avg_soc:.0%}",
         "subtitle": f"Range: {min_soc:.0%} – {max_soc:.0%}",
         "delta": "Healthy Buffer",
         "delta_color": "good",
         "icon": "🔋",
         "glow_color": "rgba(34,211,238,0.35)"},
        {"title": "Energy Charged",
         "value": f"{charge_energy:,.0f} kWh",
         "subtitle": "Surplus renewable stored",
         "icon": "⚡",
         "glow_color": "rgba(52,211,153,0.30)"},
        {"title": "Energy Discharged",
         "value": f"{discharge_energy:,.0f} kWh",
         "subtitle": "Dispatched during deficit",
         "icon": "🔌",
         "glow_color": "rgba(167,139,250,0.30)"},
        {"title": "Equivalent Cycles",
         "value": f"{eq_cycles:.1f}",
         "subtitle": f"Based on {usable_cap:.0f} kWh usable bank",
         "delta": "Low Degradation",
         "delta_color": "good",
         "icon": "🔄",
         "glow_color": "rgba(56,189,248,0.30)"},
    ])

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ── SOC Chart ──────────────────────────────────────────────────────────
    render_section_header("State of Charge (SOC) Time Series",
                          "Continuous SOC trajectory with 10% minimum and 90% maximum safety bounds", icon="📈")
    st.plotly_chart(battery_soc_chart(display_df, height=360), use_container_width=True)

    # ── Charge / Discharge Chart ───────────────────────────────────────────
    render_section_header("Daily Energy Throughput",
                          "Charge and discharge energy per day (last 30 days of selected window)", icon="⚡")
    st.plotly_chart(_charge_discharge_bar(display_df, battery_cfg["capacity_kwh"]), use_container_width=True)

    # ── Specs & Cold Derating ──────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        render_section_header("Hardware Specifications", "Calibrated battery bank design parameters", icon="⚙️")
        rte = battery_cfg["charge_efficiency"] * battery_cfg["discharge_efficiency"]
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | **Nameplate Capacity** | `{battery_cfg['capacity_kwh']} kWh` |
        | **Usable Reserve** | `{usable_cap:.0f} kWh` (10–90% SOC) |
        | **Max Charge Power** | `{battery_cfg['max_charge_power_kw']} kW` |
        | **Max Discharge Power** | `{battery_cfg['max_discharge_power_kw']} kW` |
        | **Round-Trip Efficiency** | `{rte*100:.1f}%` (η_c × η_d) |
        | **Chemistry** | Lithium-Ion (LFP) |
        """)

    with col2:
        render_section_header("Polar Thermal Derating",
                              "Capacity reduction below threshold ambient temperature", icon="❄️")
        cold_status = "critical" if cold_pct > 30 else ("warning" if cold_pct > 10 else "ok")
        render_status_badge(f"{cold_hours}h below {cold_threshold}°C  ({cold_pct:.1f}%)", cold_status)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        - **Derating Threshold:** `{cold_threshold}°C`
        - **Derating Gradient:** `{battery_cfg['cold_derating_factor']*100:.1f}%` capacity loss per °C below threshold
        - **Hours Below Threshold:** `{cold_hours}h` of `{len(display_df)}h` selected (`{cold_pct:.1f}%`)
        - **Protection:** Dynamic power limiting in `safety_constraints.py`
        - **Impact:** Max usable capacity reduced to `{usable_cap * (1 - battery_cfg['cold_derating_factor'] * max(0, 0 - cold_threshold)):.0f}+  kWh` at extreme cold
        """)


if __name__ == "__main__":
    render_battery_page()
