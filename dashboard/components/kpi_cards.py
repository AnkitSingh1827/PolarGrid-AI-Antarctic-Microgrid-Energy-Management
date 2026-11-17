"""
PolarGrid AI — Dashboard KPI Cards v2
Reusable Streamlit KPI card components with next-level glassmorphism, animated borders, and glow accents.
"""

import streamlit as st
from typing import Optional


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    delta: Optional[str] = None,
    delta_color: str = "normal",
    icon: str = "⚡",
    glow_color: str = "rgba(56, 189, 248, 0.25)",
):
    """Render a single KPI card with premium glassmorphism and dynamic glow accents."""
    delta_html = ""
    if delta:
        if delta_color in ("normal", "good"):
            fg = "#4ade80"
            arrow = "↑"
        elif delta_color == "warning":
            fg = "#fbbf24"
            arrow = "◆"
        elif delta_color == "inverse":
            fg = "#f87171"
            arrow = "↓"
        else:
            fg = "#94a3b8"
            arrow = "•"
        delta_html = (
            f'<div style="color:{fg};font-size:0.80rem;font-weight:600;margin-top:8px;'
            f'display:inline-flex;align-items:center;gap:5px;'
            f'background:rgba(255,255,255,0.04);padding:3px 10px;border-radius:9999px;'
            f'border:1px solid {fg}30;">'
            f'<span>{arrow}</span><span>{delta}</span></div>'
        )

    from dashboard.components.glass_theme import render_html
    html_card = f"""<div class="glass-card" style="text-align:center;padding:20px 14px;"><div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; background: radial-gradient(circle at 30% 30%, {glow_color}, rgba(255,255,255,0.02) 80%); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; font-size: 1.5rem; margin-bottom: 12px; box-shadow: 0 4px 20px {glow_color};">{icon}</div><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div><div class="kpi-subtitle">{subtitle}</div>{delta_html}</div>"""
    render_html(html_card)


def render_kpi_row(kpis: list):
    """
    Render a row of KPI cards.
    kpis: list of dicts with keys matching render_kpi_card parameters.
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            render_kpi_card(**kpi)


def render_status_badge(label: str, status: str = "ok"):
    """Render a colored glassmorphism status badge."""
    colors = {
        "ok":       ("#4ade80", "rgba(6, 78, 59, 0.5)",   "rgba(74, 222, 128, 0.4)"),
        "warning":  ("#fbbf24", "rgba(120, 53, 15, 0.5)", "rgba(251, 191, 36, 0.4)"),
        "critical": ("#f87171", "rgba(127, 29, 29, 0.5)", "rgba(248, 113, 113, 0.4)"),
        "info":     ("#38bdf8", "rgba(12, 74, 110, 0.5)", "rgba(56, 189, 248, 0.4)"),
    }
    fg, bg, border = colors.get(status, colors["info"])
    from dashboard.components.glass_theme import render_html
    html_badge = f"""<span style="background: {bg}; backdrop-filter: blur(10px); border: 1px solid {border}; color: {fg}; padding: 4px 14px; border-radius: 9999px; font-size: 0.73rem; font-weight: 700; letter-spacing: 0.8px; display: inline-flex; align-items: center; gap: 7px; box-shadow: 0 0 12px {border};"><span style="width:7px;height:7px;border-radius:50%;background:{fg};box-shadow:0 0 6px {fg};"></span>{label}</span>"""
    render_html(html_badge)


def render_section_header(title: str, description: str = "", icon: str = ""):
    """Render a styled section header with gradient accent bar (delegates to glass_theme)."""
    from dashboard.components.glass_theme import render_section_header as _rsh
    _rsh(title, description, icon)
