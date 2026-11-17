"""
PolarGrid AI — Main Interactive Microgrid Dashboard v2
AI-Driven Smart Energy Management System for Polar Research Stations
SIH 2026 • Problem Statement 26061 • Ministry of Earth Sciences / NCPOR
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.glass_theme import apply_glass_theme, render_html

# ── Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PolarGrid AI — Polar Energy Command Center",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_glass_theme()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    render_html("""
    <div style="text-align:center;padding:24px 0 18px 0;">
        <div style="
            display:inline-flex;align-items:center;justify-content:center;
            width:62px;height:62px;
            background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(99,102,241,0.22));
            border: 1px solid rgba(56,189,248,0.40);
            border-radius: 20px;
            font-size: 2rem;
            margin-bottom: 12px;
            box-shadow: 0 8px 28px rgba(14,165,233,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
        ">🏔️</div>
        <h1 style="
            font-size:1.5rem;font-weight:800;margin:0;
            background:linear-gradient(135deg,#38bdf8 0%,#818cf8 50%,#34d399 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;letter-spacing:-0.5px;
        ">PolarGrid AI</h1>
        <p style="font-size:0.75rem;color:#64748b;margin:4px 0 0;font-weight:500;">
            Maitri Antarctic Microgrid
        </p>
        <p style="font-size:0.66rem;color:#334155;margin:2px 0 0;">SIH 2026 • PS 26061</p>
    </div>
    """)

    render_html("<hr style='border-color:rgba(255,255,255,0.07);margin:12px 0;'>")

    # Navigation label
    render_html("""
    <div style="color:#334155;font-size:0.68rem;font-weight:800;
        text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;padding:0 4px;">
        Navigation
    </div>
    """)

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "🏠 Overview"

    PAGES = [
        "🏠 Overview",
        "🤖 AI Forecasting",
        "⚡ Energy Dispatch",
        "🔋 Battery Storage",
        "⛽ Diesel Generator",
        "🌊 Polar Scenarios",
    ]

    selected_page = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state["current_page"])
              if st.session_state["current_page"] in PAGES else 0,
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = selected_page

    render_html("<hr style='border-color:rgba(255,255,255,0.07);margin:14px 0;'>")

    # Station status card
    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")
    from dashboard.utils.api_client import check_api_health
    api_status = check_api_health()
    api_badge_color = "#4ade80" if api_status["online"] else "#f59e0b"
    api_text = "Online (Port 8000)" if api_status["online"] else "Standby (Local Fallback)"

    render_html(f"""
    <div style="
        background:rgba(10,16,32,0.65);
        backdrop-filter:blur(16px);
        border:1px solid rgba(255,255,255,0.07);
        border-radius:16px;
        padding:16px;
        font-size:0.76rem;
        color:#94a3b8;
    ">
        <div style="display:flex;align-items:center;gap:7px;font-weight:700;color:#f1f5f9;margin-bottom:10px;">
            <span style="width:8px;height:8px;border-radius:50%;background:#4ade80;
                box-shadow:0 0 8px rgba(74,222,128,0.8);display:inline-block;"></span>
            Station Online
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span>Location</span><span style="color:#cbd5e1;font-weight:600;">70.75°S, 11.75°E</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span>Dataset</span><span style="color:#cbd5e1;font-weight:600;">ERA5 2012–2025</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span>AI Models</span><span style="color:#38bdf8;font-weight:600;">XGBoost + RL</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span>Backend API</span><span style="color:{api_badge_color};font-weight:600;">{api_text}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span>Time (UTC)</span><span style="color:#818cf8;font-weight:600;font-family:'JetBrains Mono',monospace;">{now_utc}</span>
        </div>
    </div>
    """)

    render_html("<div style='height:14px;'></div>")

    # Quick nav links
    render_html("""
    <div style="font-size:0.70rem;color:#334155;font-weight:600;letter-spacing:0.5px;
        margin-bottom:6px;">Quick Access</div>
    """)

    if st.button("📊 Run Full Benchmark", use_container_width=True):
        st.toast("Launch `python -m src.reinforcement_learning.evaluate_rl` in your terminal.", icon="🤖")
    if st.button("🔄 Refresh Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.toast("All cached data cleared — pages will reload fresh data.", icon="✅")


# ── Page Routing ──────────────────────────────────────────────────────────
if selected_page == "🏠 Overview":
    from dashboard.pages.overview import render_overview_page
    render_overview_page()
elif selected_page == "🤖 AI Forecasting":
    from dashboard.pages.forecasting import render_forecasting_page
    render_forecasting_page()
elif selected_page == "⚡ Energy Dispatch":
    from dashboard.pages.energy_dispatch import render_energy_dispatch_page
    render_energy_dispatch_page()
elif selected_page == "🔋 Battery Storage":
    from dashboard.pages.battery import render_battery_page
    render_battery_page()
elif selected_page == "⛽ Diesel Generator":
    from dashboard.pages.diesel import render_diesel_page
    render_diesel_page()
elif selected_page == "🌊 Polar Scenarios":
    from dashboard.pages.scenarios import render_scenarios_page
    render_scenarios_page()
