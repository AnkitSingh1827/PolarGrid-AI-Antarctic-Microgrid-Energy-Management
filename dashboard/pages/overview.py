"""
PolarGrid AI — Dashboard Overview Page v2
Main system overview: animated KPI cards, energy balance chart, alert system, and live mini-simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.glass_theme import apply_glass_theme, render_header, render_section_header
from dashboard.components.kpi_cards import render_kpi_row, render_status_badge
from dashboard.components.charts import (
    energy_mix_area_chart,
    battery_soc_chart,
    temperature_wind_chart,
)
from dashboard.components.alerts import generate_system_alerts, render_alerts

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


from dashboard.utils.data_loader import (
    load_recent_dataset,
    load_forecast_artifacts,
    load_comparison_artifacts,
)


def render_overview_page():
    apply_glass_theme()
    render_header(
        title="Microgrid Energy Command Center",
        subtitle="Real-time physics-informed monitoring, AI forecasting, and safe autonomous dispatch for Maitri Antarctic Station",
        badge="POLAR AI • PS 26061",
    )

    recent_df = load_recent_dataset(168)
    forecast_metrics, _ = load_forecast_artifacts()
    comparison, _, _ = load_comparison_artifacts()

    # ── Station Status Banner ──────────────────────────────────────────────
    total_load = recent_df["load_kw"].sum()
    total_solar = recent_df["solar_generation_kw"].sum()
    total_wind = recent_df["wind_generation_kw"].sum()
    total_renewable = total_solar + total_wind
    renewable_pct = min(100.0, (total_renewable / total_load * 100.0)) if total_load > 0 else 0
    avg_temp = recent_df["temperature"].mean()
    avg_wind = recent_df["wind_speed"].mean()
    avg_soc = recent_df["battery_soc"].mean()
    diesel_gen = recent_df["diesel_generation_kw"]
    diesel_hours = int((diesel_gen > 0.1).sum())

    from dashboard.components.glass_theme import render_html
    banner_html = f"""<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; padding: 16px 22px; background: linear-gradient(135deg, rgba(14,28,52,0.70), rgba(10,18,36,0.60)); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.35);"><div style="display:flex;align-items:center;gap:10px;margin-right:16px;"><div style="width:10px;height:10px;border-radius:50%;background:#4ade80;box-shadow:0 0 10px rgba(74,222,128,0.8);animation:pulse-green 1.8s infinite;"></div><span style="font-weight:700;font-size:0.88rem;color:#e2e8f0;">Station ONLINE</span></div><div style="display:flex;gap:24px;flex-wrap:wrap;"><span style="font-size:0.82rem;color:#64748b;">🌍 <b style="color:#94a3b8;">70.75°S, 11.75°E</b></span><span style="font-size:0.82rem;color:#64748b;">🌡️ <b style="color:#94a3b8;">{avg_temp:.1f}°C</b></span><span style="font-size:0.82rem;color:#64748b;">💨 <b style="color:#94a3b8;">{avg_wind:.1f} m/s</b></span><span style="font-size:0.82rem;color:#64748b;">🔋 <b style="color:#22d3ee;">SOC {avg_soc:.0%}</b></span><span style="font-size:0.82rem;color:#64748b;">🌿 <b style="color:#34d399;">{renewable_pct:.1f}% Renewable</b></span><span style="font-size:0.82rem;color:#64748b;">🤖 <b style="color:#818cf8;">XGBoost + Safe RL</b></span></div></div>"""
    render_html(banner_html)

    # ── System Alerts ──────────────────────────────────────────────────────
    render_section_header("Active Security & Microgrid Alerts",
                          "Real-time automated alarms based on battery reserve, cold derating, and demand spikes", icon="🚨")
    alerts = generate_system_alerts(dispatch_df=recent_df, forecast_metrics=forecast_metrics)
    render_alerts(alerts)

    # ── KPI Cards Row 1 ────────────────────────────────────────────────────
    render_section_header("Key Operational Metrics — 7-Day Rolling Window",
                          "Aggregated energy flows, storage state, and generator runtime", icon="📊")

    render_kpi_row([
        {"title": "Renewable Share",
         "value": f"{renewable_pct:.1f}%",
         "subtitle": f"{total_renewable:,.0f} kWh generated",
         "delta": "+47.0% vs All-Diesel",
         "delta_color": "good",
         "icon": "🌿",
         "glow_color": "rgba(52,211,153,0.35)"},
        {"title": "Total Station Load",
         "value": f"{total_load:,.0f} kWh",
         "subtitle": "168-hour cumulative demand",
         "icon": "⚡",
         "glow_color": "rgba(251,191,36,0.30)"},
        {"title": "Battery SOC",
         "value": f"{avg_soc:.0%}",
         "subtitle": f"Min {recent_df['battery_soc'].min():.0%} · Max {recent_df['battery_soc'].max():.0%}",
         "icon": "🔋",
         "glow_color": "rgba(34,211,238,0.35)"},
        {"title": "Diesel Active Hours",
         "value": f"{diesel_hours}h",
         "subtitle": f"of 168h total ({diesel_hours/168*100:.1f}%)",
         "delta": "Low Cycling",
         "delta_color": "good",
         "icon": "⛽",
         "glow_color": "rgba(244,63,94,0.30)"},
    ])

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ── KPI Cards Row 2 ────────────────────────────────────────────────────
    render_kpi_row([
        {"title": "Station Temperature",
         "value": f"{avg_temp:.1f}°C",
         "subtitle": f"Min: {recent_df['temperature'].min():.1f}°C  |  Max: {recent_df['temperature'].max():.1f}°C",
         "icon": "🌡️",
         "glow_color": "rgba(248,113,113,0.25)"},
        {"title": "Mean Wind Speed",
         "value": f"{avg_wind:.1f} m/s",
         "subtitle": f"Peak: {recent_df['wind_speed'].max():.1f} m/s (hub height)",
         "icon": "💨",
         "glow_color": "rgba(56,189,248,0.25)"},
        {"title": "Solar Output (7d)",
         "value": f"{total_solar:,.0f} kWh",
         "subtitle": "Total photovoltaic generation",
         "icon": "☀️",
         "glow_color": "rgba(251,191,36,0.25)"},
        {"title": "Wind Output (7d)",
         "value": f"{total_wind:,.0f} kWh",
         "subtitle": "Total wind turbine generation",
         "icon": "🌬️",
         "glow_color": "rgba(56,189,248,0.25)"},
    ])

    # ── Energy Balance Chart ───────────────────────────────────────────────
    render_section_header("Dynamic Power Balance",
                          "Generation mix (Solar PV + Wind + Diesel) stacked against total load demand", icon="⚡")
    st.plotly_chart(energy_mix_area_chart(recent_df), use_container_width=True)

    # ── Battery & Weather ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        render_section_header("BESS State of Charge",
                              "500 kWh Li-Ion reserve level with thermal derating bounds", icon="🔋")
        st.plotly_chart(battery_soc_chart(recent_df, height=320), use_container_width=True)
    with col2:
        render_section_header("Atmospheric Conditions",
                              "Ambient air temperature and hub-height wind speed telemetry", icon="🏔️")
        st.plotly_chart(temperature_wind_chart(recent_df, height=320), use_container_width=True)

    # ── Controller Benchmark Table ─────────────────────────────────────────
    if comparison:
        render_section_header("Autonomous Controller Benchmark",
                              "Multi-criteria evaluation comparing Rule-Based, Pyomo MILP, and Safe RL controllers", icon="🏆")
        comp_rows = []
        for ctrl_name, m in comparison.items():
            savings = m.get("fuel_savings_vs_baseline_pct", 0)
            comp_rows.append({
                "Controller": m.get("controller", ctrl_name),
                "Fuel (L)":   f"{m.get('total_fuel_litres', 0):,.1f}",
                "Cost (USD)": f"${m.get('total_cost_usd', 0):,.2f}",
                "Reliability":  f"{m.get('reliability_pct', 0):.4f}%",
                "Renewable %":  f"{m.get('renewable_utilization_pct', 0):.2f}%",
                "Diesel Hrs":   f"{m.get('diesel_run_hours', 0)}h",
                "Fuel Savings": f"{savings:+.2f}%",
            })
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    footer_html = """<div style="margin-top: 36px; padding: 18px 24px; background: rgba(8,14,28,0.55); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.06); border-radius: 18px; display: flex; gap: 32px; flex-wrap: wrap; align-items: center;"><div style="font-size:0.78rem;color:#475569;"><span style="color:#38bdf8;font-weight:700;">PolarGrid AI</span> &nbsp;|&nbsp; SIH 2026 Problem Statement 26061 &nbsp;|&nbsp; Ministry of Earth Sciences / NCPOR &nbsp;|&nbsp; Maitri Antarctic Research Station</div><div style="font-size:0.78rem;color:#334155;margin-left:auto;">Data: ERA5 Reanalysis 2012–2025 &nbsp;·&nbsp; AI: XGBoost + PPO + SAC</div></div>"""
    render_html(footer_html)


if __name__ == "__main__":
    render_overview_page()
