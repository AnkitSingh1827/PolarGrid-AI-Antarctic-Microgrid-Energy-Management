"""
PolarGrid AI — Next-Level Ultra-Premium Glassmorphism Design System v2
Provides reusable glassmorphic styles, animated aurora, custom badges, and layout utilities.
"""

import streamlit as st
from datetime import datetime, timezone


GLASS_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200;0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── Base Reset & Fonts ─────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Animated Aurora Background ─────────────────────────────────── */
    @keyframes aurora-drift-1 {
        0%   { transform: translate(0%,    0%)    scale(1);    opacity: 0.12; }
        25%  { transform: translate(3%,    -4%)   scale(1.08); opacity: 0.18; }
        50%  { transform: translate(-2%,   2%)    scale(0.95); opacity: 0.13; }
        75%  { transform: translate(4%,    1%)    scale(1.05); opacity: 0.20; }
        100% { transform: translate(0%,    0%)    scale(1);    opacity: 0.12; }
    }
    @keyframes aurora-drift-2 {
        0%   { transform: translate(0%,    0%)    scale(1);    opacity: 0.14; }
        33%  { transform: translate(-5%,   3%)    scale(1.1);  opacity: 0.22; }
        66%  { transform: translate(4%,    -2%)   scale(0.92); opacity: 0.10; }
        100% { transform: translate(0%,    0%)    scale(1);    opacity: 0.14; }
    }
    @keyframes aurora-drift-3 {
        0%   { transform: translate(0%,    0%)    scale(1);    opacity: 0.08; }
        50%  { transform: translate(-3%,   -3%)   scale(1.15); opacity: 0.16; }
        100% { transform: translate(0%,    0%)    scale(1);    opacity: 0.08; }
    }

    .stApp {
        background:
            radial-gradient(ellipse 70% 50% at 10% 20%, rgba(14, 165, 233, 0.14) 0%, transparent 60%),
            radial-gradient(ellipse 60% 60% at 88% 75%, rgba(99, 102, 241, 0.18) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 55% 45%, rgba(16, 185, 129, 0.10) 0%, transparent 65%),
            radial-gradient(ellipse 80% 30% at 50% 100%, rgba(56, 189, 248, 0.06) 0%, transparent 50%),
            linear-gradient(160deg, #050a14 0%, #0a1020 45%, #060d1a 100%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9;
        min-height: 100vh;
    }

    /* Floating orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -20%;
        left: -10%;
        width: 65%;
        height: 65%;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.10) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        animation: aurora-drift-1 18s ease-in-out infinite;
        z-index: 0;
    }
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -15%;
        right: -8%;
        width: 55%;
        height: 55%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        animation: aurora-drift-2 22s ease-in-out infinite;
        z-index: 0;
    }

    /* ── Hide Streamlit Chrome ──────────────────────────────────────── */
    [data-testid="stSidebarNav"] { display: none !important; }
    #MainMenu, footer, header { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    /* ── Glass Sidebar ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(8, 14, 28, 0.82) !important;
        backdrop-filter: blur(32px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        box-shadow: 4px 0 40px rgba(0, 0, 0, 0.5) !important;
    }

    /* ── Ultra Glass Cards ──────────────────────────────────────────── */
    @keyframes card-shimmer {
        0%   { left: -100%; }
        100% { left: 200%; }
    }

    .glass-card {
        background: linear-gradient(
            135deg,
            rgba(18, 28, 52, 0.65) 0%,
            rgba(14, 22, 42, 0.55) 100%
        ) !important;
        backdrop-filter: blur(24px) saturate(200%) brightness(1.1) !important;
        -webkit-backdrop-filter: blur(24px) saturate(200%) brightness(1.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 22px !important;
        padding: 24px 20px !important;
        box-shadow:
            0 8px 40px rgba(0, 0, 0, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            inset 0 -1px 0 rgba(0, 0, 0, 0.15) !important;
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.35s ease,
                    border-color 0.35s ease !important;
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 60%;
        height: 100%;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.04) 40%,
            rgba(255, 255, 255, 0.08) 50%,
            rgba(255, 255, 255, 0.04) 60%,
            transparent 100%);
        pointer-events: none;
    }

    .glass-card:hover {
        transform: translateY(-6px) scale(1.015);
        border-color: rgba(56, 189, 248, 0.45) !important;
        box-shadow:
            0 20px 60px rgba(14, 165, 233, 0.25),
            0 4px 16px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    .glass-card:hover::before {
        animation: card-shimmer 0.7s ease forwards;
    }

    /* ── KPI Typography ─────────────────────────────────────────────── */
    .kpi-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 60%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.05;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }
    .kpi-subtitle {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 500;
        line-height: 1.4;
    }

    /* ── Pulsating Live Status ──────────────────────────────────────── */
    @keyframes pulse-green {
        0%   { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
        70%  { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
        100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
    }
    @keyframes pulse-red {
        0%   { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.7); }
        70%  { box-shadow: 0 0 0 8px rgba(248, 113, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
    }
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse-green 1.8s infinite;
        margin-right: 6px;
        vertical-align: middle;
    }
    .live-dot-red {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #f87171;
        border-radius: 50%;
        animation: pulse-red 1.5s infinite;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* ── Buttons ────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg,
            rgba(28, 40, 68, 0.85) 0%,
            rgba(14, 22, 42, 0.85) 100%) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.13) !important;
        border-radius: 14px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 10px 22px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg,
            rgba(56, 189, 248, 0.25) 0%,
            rgba(99, 102, 241, 0.25) 100%) !important;
        border-color: rgba(56, 189, 248, 0.55) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Sidebar Nav Radio ──────────────────────────────────────────── */
    div[role="radiogroup"] > label {
        background: rgba(14, 22, 42, 0.5) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 11px 16px !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    div[role="radiogroup"] > label:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        color: #e2e8f0 !important;
    }
    div[role="radiogroup"] > label[data-testid="stMarkdownContainer"] {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
    }

    /* ── Selectbox ──────────────────────────────────────────────────── */
    .stSelectbox > div > div {
        background: rgba(14, 22, 42, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 14px !important;
        color: #e2e8f0 !important;
    }
    .stSelectbox > div > div:hover {
        border-color: rgba(56, 189, 248, 0.4) !important;
    }

    /* ── Sliders ────────────────────────────────────────────────────── */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    }

    /* ── Plotly Charts ──────────────────────────────────────────────── */
    .stPlotlyChart {
        background: rgba(10, 16, 32, 0.55) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 22px !important;
        padding: 18px !important;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.08) !important;
        transition: box-shadow 0.3s ease !important;
    }
    .stPlotlyChart:hover {
        box-shadow: 0 12px 48px rgba(14, 165, 233, 0.18), inset 0 1px 0 rgba(255,255,255,0.10) !important;
    }

    /* ── DataFrames / Tables ────────────────────────────────────────── */
    .stDataFrame, [data-testid="stDataFrame"] {
        background: rgba(10, 16, 32, 0.55) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }

    /* ── Alerts / Info Boxes ────────────────────────────────────────── */
    .stAlert {
        background: rgba(14, 22, 42, 0.7) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        border-left-width: 4px !important;
    }

    /* ── Metric elements ────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: rgba(14, 22, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        backdrop-filter: blur(16px) !important;
    }

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); border-radius: 3px; }
    ::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.25); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.5); }

    /* ── Progress bar ───────────────────────────────────────────────── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399) !important;
    }

    /* ── Divider ────────────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* ── Page fade-in ───────────────────────────────────────────────── */
    @keyframes page-in {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: page-in 0.5s ease forwards;
        padding-top: 1.5rem !important;
    }
</style>
"""


def render_html(html_code: str):
    """Render HTML cleanly without Markdown indentation interpreting it as code blocks."""
    if hasattr(st, "html"):
        st.html(html_code)
    else:
        import textwrap
        st.markdown(textwrap.dedent(html_code), unsafe_allow_html=True)


def apply_glass_theme():
    """Inject the global ultra-glassmorphism CSS stylesheet."""
    render_html(GLASS_CSS)


def render_header(title: str, subtitle: str = "", badge: str = "POLAR AI"):
    """Render a premium glassmorphic page header with aurora glow and live clock."""
    now = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    html_content = f"""<div style="margin-bottom: 24px; padding: 24px 28px; background: linear-gradient(135deg, rgba(18,28,52,0.60) 0%, rgba(14,22,42,0.45) 100%); backdrop-filter: blur(24px) saturate(200%); -webkit-backdrop-filter: blur(24px) saturate(200%); border: 1px solid rgba(255,255,255,0.09); border-radius: 24px; box-shadow: 0 8px 40px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.10); position: relative; overflow: hidden;"><div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;background:radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);pointer-events:none;border-radius:50%;"></div><div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:14px;"><div style="display:flex;align-items:center;gap:10px;"><span style="background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(99,102,241,0.18)); border: 1px solid rgba(56,189,248,0.45); color: #38bdf8; padding: 4px 14px; border-radius: 9999px; font-size: 0.70rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;">{badge}</span><span style="font-size:0.82rem;color:#64748b;"><span class="live-dot"></span>Maitri Antarctic Station — Live Telemetry</span></div><span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#475569;background:rgba(255,255,255,0.04);padding:4px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.06);">🕐 {now}</span></div><h1 style="font-size: 2.2rem; font-weight: 800; margin: 0 0 6px 0; background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 45%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -1px; line-height: 1.15;">{title}</h1><p style="color:#7c8fa8;font-size:0.92rem;margin:0;font-weight:400;line-height:1.5;">{subtitle}</p></div>"""
    render_html(html_content)


def render_section_header(title: str, description: str = "", icon: str = ""):
    """Render a styled section header with gradient accent bar."""
    icon_str = f"{icon} " if icon else ""
    desc_html = f'<p style="color:#4e6080;font-size:0.83rem;margin:3px 0 0 16px;font-weight:400;line-height:1.5;">{description}</p>' if description else ""
    html_content = f"""<div style="margin: 2rem 0 1.2rem 0;"><div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;"><div style="width: 4px; min-height: 24px; background: linear-gradient(180deg, #38bdf8 0%, #818cf8 60%, #34d399 100%); border-radius: 4px; box-shadow: 0 0 12px rgba(56,189,248,0.5);"></div><h2 style="color: #f0f4f8; font-size: 1.2rem; font-weight: 700; margin: 0; letter-spacing: -0.3px;">{icon_str}{title}</h2></div>{desc_html}</div>"""
    render_html(html_content)
