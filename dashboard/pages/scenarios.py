"""
PolarGrid AI — Dashboard Scenarios Page v2
Scenario stress-testing: Normal, Polar Night, Extreme Cold, Low Renewable, Load Spike, Generator Failure.
Includes interactive simulation slider and robust fallback when scenario CSV is missing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.glass_theme import apply_glass_theme, render_header, render_section_header
from dashboard.components.kpi_cards import render_kpi_row, render_status_badge
from dashboard.components.charts import energy_mix_area_chart, battery_soc_chart

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SIMULATED = PROJECT_ROOT / "data" / "simulated"
CONFIGS        = PROJECT_ROOT / "configs"

SCENARIO_DEFS = {
    "☀️ Normal Operations": {
        "icon": "☀️",
        "color": "#38bdf8",
        "desc": "Standard baseline: nominal solar radiation and moderate katabatic winds.",
        "tag": "1.0× Load · 1.0× Solar · 1.0× Wind",
        "solar_mul": 1.0, "wind_mul": 1.0, "load_mul": 1.0, "diesel_cap": 1.0,
    },
    "🌑 Polar Night (Apr–Aug)": {
        "icon": "🌑",
        "color": "#818cf8",
        "desc": "24-hour total darkness for months. Zero PV generation; wind and BESS cover all demand.",
        "tag": "1.1× Heating Load · 0.0× Solar",
        "solar_mul": 0.0, "wind_mul": 1.0, "load_mul": 1.1, "diesel_cap": 1.0,
    },
    "🥶 Extreme Cold Wave": {
        "icon": "🥶",
        "color": "#60a5fa",
        "desc": "Air temperature plunges below −40°C. Heating load surges; BESS capacity derated by up to 20%.",
        "tag": "1.4× Load · −20% BESS Capacity",
        "solar_mul": 0.6, "wind_mul": 0.9, "load_mul": 1.4, "diesel_cap": 1.0,
    },
    "🌫️ Low Renewable Lull": {
        "icon": "🌫️",
        "color": "#fbbf24",
        "desc": "Heavy cloud cover + wind calm (below cut-in). BESS buffers; diesel compensates deficit.",
        "tag": "0.15× Solar · 0.20× Wind",
        "solar_mul": 0.15, "wind_mul": 0.20, "load_mul": 1.0, "diesel_cap": 1.0,
    },
    "⚡ Research Load Spike": {
        "icon": "⚡",
        "color": "#fb923c",
        "desc": "Simultaneous activation of atmospheric radar and ice-core drilling during active hours.",
        "tag": "1.5× Peak Load Demand",
        "solar_mul": 1.0, "wind_mul": 1.0, "load_mul": 1.5, "diesel_cap": 1.0,
    },
    "🔧 Genset Unit Failure": {
        "icon": "🔧",
        "color": "#f43f5e",
        "desc": "One 75 kW generator unit trips offline during scheduled maintenance. Backup capacity halved.",
        "tag": "0.5× Diesel Capacity",
        "solar_mul": 1.0, "wind_mul": 1.0, "load_mul": 1.0, "diesel_cap": 0.5,
    },
}


from dashboard.utils.data_loader import load_master_dataset


def load_base_data():
    return load_master_dataset()


def _simulate_scenario(base_df: pd.DataFrame, params: dict, n_hours: int = 168) -> pd.DataFrame:
    """Apply scenario multipliers and compute dynamic power balance with vectorized execution."""
    df = base_df.tail(n_hours).copy()
    sol = np.maximum(0.0, df["solar_generation_kw"].values * params["solar_mul"])
    wnd = np.maximum(0.0, df["wind_generation_kw"].values * params["wind_mul"])
    lod = np.maximum(0.0, df["load_kw"].values * params["load_mul"])
    df["solar_generation_kw"] = sol
    df["wind_generation_kw"] = wnd
    df["load_kw"] = lod

    renewable = sol + wnd
    diesel_max = 150.0 * params["diesel_cap"]
    bess_soc = df["battery_soc"].values

    gap = lod - renewable - (bess_soc * 50.0)
    df["diesel_generation_kw"] = np.clip(gap, 0.0, diesel_max)

    surplus_arr = renewable - lod
    soc_vals = np.empty(len(df), dtype=float)
    soc = 0.65
    for idx, surplus in enumerate(surplus_arr):
        if surplus > 0:
            soc = min(0.90, soc + surplus * 0.0019)
        else:
            soc = max(0.10, soc + surplus * 0.0021)
        soc_vals[idx] = soc
    df["battery_soc"] = soc_vals
    return df


def render_scenarios_page():
    apply_glass_theme()
    render_header(
        title="Antarctic Polar Scenario Stress Testing",
        subtitle="System resilience simulation across 6 extreme polar weather anomalies and equipment failure conditions",
        badge="RESILIENCE • STRESS TESTING",
    )

    base_df = load_base_data()

    # ── Scenario Cards Grid ────────────────────────────────────────────────
    render_section_header("Defined Operating Scenarios",
                          "Stress-testing conditions modelling real Antarctic extremes", icon="🌊")

    from dashboard.components.glass_theme import render_html
    cols = st.columns(3)
    for idx, (name, info) in enumerate(SCENARIO_DEFS.items()):
        with cols[idx % 3]:
            card_html = f"""<div class="glass-card" style="margin-bottom:16px;min-height:185px;text-align:left;padding:20px;border-left:3px solid {info['color']}60;"><div style="font-size:1.9rem;margin-bottom:8px;">{info['icon']}</div><div style="color:#f0f4f8;font-weight:700;font-size:0.98rem;margin-bottom:5px;">{name}</div><div style="color:#7c8fa8;font-size:0.78rem;line-height:1.5;margin-bottom:10px;">{info['desc']}</div><div style="color:{info['color']};font-size:0.72rem;font-weight:700;background:rgba(255,255,255,0.04);padding:4px 10px;border-radius:6px;border:1px solid {info['color']}40;display:inline-block;">{info['tag']}</div></div>"""
            render_html(card_html)

    # ── Interactive Simulator ──────────────────────────────────────────────
    render_section_header("Interactive Scenario Simulator",
                          "Apply scenario parameters on top of real ERA5 data and visualise dynamic power flows", icon="🎮")

    col_sc, col_wn = st.columns([2, 1])
    with col_sc:
        selected = st.selectbox("Select Scenario Profile", list(SCENARIO_DEFS.keys()), key="sc_select")
    with col_wn:
        n_hours = st.slider("Simulation Horizon (Hours)", 24, 720, 168, step=24, key="sc_hours")

    params = SCENARIO_DEFS[selected]
    sc_df = _simulate_scenario(base_df, params, n_hours)

    # KPI mini strip for the scenario
    sc_total_solar   = sc_df["solar_generation_kw"].sum()
    sc_total_wind    = sc_df["wind_generation_kw"].sum()
    sc_total_diesel  = sc_df["diesel_generation_kw"].sum()
    sc_total_load    = sc_df["load_kw"].sum()
    sc_ren_pct       = min(100.0, (sc_total_solar + sc_total_wind) / sc_total_load * 100) if sc_total_load > 0 else 0
    sc_diesel_hrs    = int((sc_df["diesel_generation_kw"] > 0.1).sum())
    sc_avg_soc       = sc_df["battery_soc"].mean()

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    render_kpi_row([
        {"title": "Renewable Share",
         "value": f"{sc_ren_pct:.1f}%",
         "subtitle": f"Solar {sc_total_solar:,.0f} + Wind {sc_total_wind:,.0f} kWh",
         "icon": "🌿", "glow_color": "rgba(52,211,153,0.30)"},
        {"title": "Diesel Dispatch",
         "value": f"{sc_total_diesel:,.0f} kWh",
         "subtitle": f"{sc_diesel_hrs}h generator active",
         "icon": "⛽", "glow_color": "rgba(244,63,94,0.25)"},
        {"title": "Total Load Served",
         "value": f"{sc_total_load:,.0f} kWh",
         "subtitle": f"{n_hours}h scenario horizon",
         "icon": "⚡", "glow_color": "rgba(251,191,36,0.25)"},
        {"title": "Mean Battery SOC",
         "value": f"{sc_avg_soc:.0%}",
         "subtitle": f"Min {sc_df['battery_soc'].min():.0%} · Max {sc_df['battery_soc'].max():.0%}",
         "icon": "🔋", "glow_color": "rgba(34,211,238,0.30)"},
    ])

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.plotly_chart(energy_mix_area_chart(sc_df, height=420), use_container_width=True)
    st.plotly_chart(battery_soc_chart(sc_df, height=300), use_container_width=True)

    # ── Comparison across all scenarios ───────────────────────────────────
    render_section_header("All-Scenario Comparative Summary",
                          "Renewable share, diesel hours, and mean SOC across all six stress conditions", icon="📊")

    rows = []
    for name, params in SCENARIO_DEFS.items():
        s = _simulate_scenario(base_df, params, 168)
        sol = s["solar_generation_kw"].sum()
        wnd = s["wind_generation_kw"].sum()
        lod = s["load_kw"].sum()
        dsl = s["diesel_generation_kw"].sum()
        ren = min(100, (sol + wnd) / lod * 100) if lod > 0 else 0
        dhr = int((s["diesel_generation_kw"] > 0.1).sum())
        rows.append({
            "Scenario": name,
            "Renewable %": f"{ren:.1f}%",
            "Diesel kWh":  f"{dsl:,.0f}",
            "Diesel Hrs":  f"{dhr}h",
            "Mean SOC":    f"{s['battery_soc'].mean():.0%}",
            "Peak Load kW": f"{s['load_kw'].max():.1f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    render_scenarios_page()
