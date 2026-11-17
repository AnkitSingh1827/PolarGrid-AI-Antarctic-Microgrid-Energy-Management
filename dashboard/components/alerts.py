"""
PolarGrid AI — Dashboard Alerts
Data-driven alert system with glassmorphic cards and pulsating severity indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict


def generate_system_alerts(
    dispatch_df: pd.DataFrame = None,
    forecast_metrics: dict = None,
    params: dict = None,
) -> List[Dict[str, str]]:
    """
    Generate data-driven alerts based on current system state.
    Returns list of alert dicts with keys: level, title, message, icon.
    """
    alerts = []

    if dispatch_df is not None and len(dispatch_df) > 0:
        recent = dispatch_df.tail(24)

        # Battery SOC alerts
        soc_col = "battery_soc" if "battery_soc" in recent.columns else None
        if soc_col:
            min_soc = recent[soc_col].min()
            if min_soc < 0.15:
                alerts.append({
                    "level": "critical",
                    "title": "Critical Battery Reserve",
                    "message": f"Battery SOC dropped to {min_soc:.1%} within the last 24 hours. Cold derating protection active.",
                    "icon": "⚡",
                })
            elif min_soc < 0.25:
                alerts.append({
                    "level": "warning",
                    "title": "Low Battery Reserve Advisory",
                    "message": f"Battery SOC reached {min_soc:.1%}. Energy management recommends reducing non-critical laboratory loads.",
                    "icon": "🔋",
                })

        # Unmet load alerts
        unmet_col = "unmet_load_kw" if "unmet_load_kw" in recent.columns else None
        if unmet_col:
            total_unmet = recent[unmet_col].sum()
            hours_unmet = (recent[unmet_col] > 0.01).sum()
            if total_unmet > 0.1:
                alerts.append({
                    "level": "critical",
                    "title": "Unmet Load Event",
                    "message": f"{total_unmet:.1f} kWh unserved load across {hours_unmet} hours. Station microgrid reliability threshold compromised.",
                    "icon": "⚠️",
                })

        # Diesel overuse alert
        diesel_col = "diesel_status" if "diesel_status" in recent.columns else None
        if diesel_col:
            diesel_hours = (recent[diesel_col] > 0).sum()
            if diesel_hours > 18:
                alerts.append({
                    "level": "warning",
                    "title": "High Diesel Run-Time",
                    "message": f"Diesel generator active for {diesel_hours}/24 hours. Fuel conservation policy recommended.",
                    "icon": "⛽",
                })

        # Temperature alert
        if "temperature" in recent.columns:
            min_temp = recent["temperature"].min()
            if min_temp < -40:
                alerts.append({
                    "level": "critical",
                    "title": "Severe Polar Cold Wave",
                    "message": f"Station temperature reached {min_temp:.1f}°C. Battery capacity derated by >10%. Life-support heating loads prioritized.",
                    "icon": "🥶",
                })
            elif min_temp < -25:
                alerts.append({
                    "level": "warning",
                    "title": "Sub-Zero Thermal Advisory",
                    "message": f"Temperature dropped to {min_temp:.1f}°C. Lithium-ion battery heating jackets active.",
                    "icon": "❄️",
                })

    # Forecasting quality alerts
    if forecast_metrics:
        for model_name, metrics in forecast_metrics.items():
            if metrics.get("R2", 1.0) < 0.85:
                alerts.append({
                    "level": "warning",
                    "title": f"{model_name.title()} Forecast Variance",
                    "message": f"R² metric is {metrics['R2']:.3f}. Retraining recommended under current weather pattern.",
                    "icon": "📉",
                })

    if not alerts:
        alerts.append({
            "level": "ok",
            "title": "All Microgrid Systems Nominal",
            "message": "Power balance, battery SOC limits, and renewables integration operating within safe design limits.",
            "icon": "🛡️",
        })

    return alerts


def render_alerts(alerts: List[Dict[str, str]]):
    """Render alerts in the dashboard with ultra-glassmorphic cards."""
    for alert in alerts:
        level = alert["level"]

        if level == "critical":
            bg = "rgba(127, 29, 29, 0.4)"
            border = "rgba(248, 113, 113, 0.5)"
            glow = "rgba(239, 68, 68, 0.15)"
            title_color = "#fca5a5"
        elif level == "warning":
            bg = "rgba(120, 53, 15, 0.35)"
            border = "rgba(251, 191, 36, 0.45)"
            glow = "rgba(245, 158, 11, 0.15)"
            title_color = "#fde68a"
        elif level == "ok":
            bg = "rgba(6, 78, 59, 0.35)"
            border = "rgba(74, 222, 128, 0.4)"
            glow = "rgba(16, 185, 129, 0.15)"
            title_color = "#86efac"
        else:
            bg = "rgba(12, 74, 110, 0.35)"
            border = "rgba(56, 189, 248, 0.4)"
            glow = "rgba(14, 165, 233, 0.15)"
            title_color = "#93c5fd"

        from dashboard.components.glass_theme import render_html
        alert_html = f"""<div style="background: {bg}; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid {border}; border-left: 5px solid {title_color}; border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 4px 20px {glow}; transition: all 0.2s ease;"><div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;"><span style="font-size:1.1rem;">{alert['icon']}</span><span style="font-size:0.92rem;font-weight:700;color:{title_color};letter-spacing:-0.2px;">{alert['title']}</span></div><div style="font-size:0.83rem;color:#cbd5e1;line-height:1.4;margin-left:26px;">{alert['message']}</div></div>"""
        render_html(alert_html)
