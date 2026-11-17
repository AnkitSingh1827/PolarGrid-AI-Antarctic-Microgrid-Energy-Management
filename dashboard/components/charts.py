"""
PolarGrid AI — Dashboard Charts v2
Next-level Plotly charts with aurora polar palettes, animated traces, and transparent glass styling.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Aurora Neon Palette ────────────────────────────────────────────────────────
COLORS = {
    "solar":             "#fbbf24",           # Warm Sun Gold
    "wind":              "#38bdf8",           # Polar Cyan
    "diesel":            "#f43f5e",           # Coral Red
    "battery_charge":    "#34d399",           # Emerald Mint
    "battery_discharge": "#a78bfa",           # Aurora Violet
    "load":              "#f8fafc",           # Crisp White
    "critical_load":     "#fb923c",           # Safety Amber
    "unmet":             "#ef4444",           # Alert Crimson
    "soc":               "#22d3ee",           # Electric Cyan
    "temp":              "#f87171",           # Soft Coral
    "wind_speed":        "#38bdf8",           # Polar Cyan
    "grid":              "rgba(255,255,255,0.055)",
    "bg":                "rgba(0,0,0,0)",
}

_FONT = dict(color="#94a3b8", family="Plus Jakarta Sans, sans-serif", size=12)

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=_FONT,
    margin=dict(l=48, r=20, t=44, b=40),
    xaxis=dict(
        gridcolor=COLORS["grid"],
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.09)",
        tickfont=dict(color="#64748b", size=11),
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"],
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.09)",
        tickfont=dict(color="#64748b", size=11),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.03,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
        itemsizing="constant",
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(10,16,32,0.92)",
        bordercolor="rgba(56,189,248,0.35)",
        font=dict(family="Plus Jakarta Sans", color="#f0f4f8", size=12),
    ),
)


def _apply_layout(fig: go.Figure, title: str = "", yaxis_title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color="#e2e8f0", family="Plus Jakarta Sans", weight=700),
            x=0,
            xanchor="left",
        ),
        yaxis_title=yaxis_title,
        height=height,
        **LAYOUT_BASE,
    )
    return fig


# ── Energy Mix Area Chart ──────────────────────────────────────────────────────
def energy_mix_area_chart(df: pd.DataFrame, x_col: str = "timestamp", height: int = 440) -> go.Figure:
    """Stacked area chart showing generation mix with aurora gradients."""
    fig = go.Figure()

    # Resolve column names
    solar_col = next((c for c in ["solar_generation_kw", "solar_kw", "solar_available_kw"] if c in df.columns), None)
    wind_col  = next((c for c in ["wind_generation_kw",  "wind_kw",  "wind_available_kw"]  if c in df.columns), None)
    diesel_col = next((c for c in ["diesel_generation_kw", "diesel_kw"] if c in df.columns), None)

    x = df[x_col] if x_col in df.columns else df.index

    if solar_col:
        fig.add_trace(go.Scatter(
            x=x, y=df[solar_col], name="☀️ Solar PV",
            fill="tonexty",
            fillcolor="rgba(251,191,36,0.28)",
            line=dict(color=COLORS["solar"], width=2),
            stackgroup="gen",
            hovertemplate="<b>Solar:</b> %{y:.1f} kW<extra></extra>",
        ))
    if wind_col:
        fig.add_trace(go.Scatter(
            x=x, y=df[wind_col], name="💨 Wind Turbine",
            fill="tonexty",
            fillcolor="rgba(56,189,248,0.28)",
            line=dict(color=COLORS["wind"], width=2),
            stackgroup="gen",
            hovertemplate="<b>Wind:</b> %{y:.1f} kW<extra></extra>",
        ))
    if diesel_col:
        fig.add_trace(go.Scatter(
            x=x, y=df[diesel_col], name="⛽ Diesel Gen",
            fill="tonexty",
            fillcolor="rgba(244,63,94,0.25)",
            line=dict(color=COLORS["diesel"], width=2),
            stackgroup="gen",
            hovertemplate="<b>Diesel:</b> %{y:.1f} kW<extra></extra>",
        ))
    if "load_kw" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["load_kw"], name="⚡ Total Demand",
            line=dict(color=COLORS["load"], width=2.5, dash="solid"),
            hovertemplate="<b>Load:</b> %{y:.1f} kW<extra></extra>",
        ))

    return _apply_layout(fig, "Microgrid Power Generation & Load Balance", "Power (kW)", height)


# ── Battery SOC Chart ──────────────────────────────────────────────────────────
def battery_soc_chart(df: pd.DataFrame, x_col: str = "timestamp",
                      soc_col: str = "battery_soc", height: int = 340) -> go.Figure:
    """Battery SOC with glowing zones and threshold lines."""
    fig = go.Figure()
    x = df[x_col] if x_col in df.columns else df.index

    # Background zone fills
    fig.add_hrect(y0=0.00, y1=0.10, fillcolor="rgba(239,68,68,0.10)",  line_width=0, annotation_text="Critical")
    fig.add_hrect(y0=0.10, y1=0.25, fillcolor="rgba(251,191,36,0.07)", line_width=0)
    fig.add_hrect(y0=0.75, y1=0.90, fillcolor="rgba(52,211,153,0.07)", line_width=0)
    fig.add_hrect(y0=0.90, y1=1.00, fillcolor="rgba(56,189,248,0.10)", line_width=0, annotation_text="Full")

    fig.add_trace(go.Scatter(
        x=x,
        y=df[soc_col] if soc_col in df.columns else [0] * len(df),
        name="🔋 Battery SOC",
        fill="tozeroy",
        fillcolor="rgba(34,211,238,0.18)",
        line=dict(color=COLORS["soc"], width=2.5),
        hovertemplate="<b>SOC:</b> %{y:.1%}<extra></extra>",
    ))

    fig.add_hline(y=0.10, line=dict(color="rgba(239,68,68,0.7)",   dash="dot", width=1.5),
                  annotation_text="Min 10%", annotation_font_color="#f87171", annotation_position="top right")
    fig.add_hline(y=0.90, line=dict(color="rgba(56,189,248,0.65)", dash="dot", width=1.5),
                  annotation_text="Max 90%", annotation_font_color="#38bdf8", annotation_position="top right")

    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _apply_layout(fig, "Battery State of Charge (SOC)", "SOC Level", height)


# ── Forecast Comparison Chart ──────────────────────────────────────────────────
def forecast_comparison_chart(
    df: pd.DataFrame,
    actual_col: str = "load_actual",
    pred_col: str = "load_pred",
    x_col: str = "timestamp",
    title: str = "Forecast vs Actual",
    y_label: str = "kW",
    height: int = 380,
) -> go.Figure:
    """Dual line chart: ground truth vs XGBoost predictions with shaded confidence band."""
    fig = go.Figure()
    x = df[x_col] if x_col in df.columns else df.index

    if actual_col in df.columns and pred_col in df.columns:
        residuals = (df[actual_col] - df[pred_col]).abs()
        # Shaded band (±1 sigma)
        sigma = residuals.std()
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x.iloc[::-1]]),
            y=pd.concat([df[pred_col] + sigma, (df[pred_col] - sigma).iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(245,158,11,0.10)",
            line=dict(width=0),
            name="±1σ Confidence",
            hoverinfo="skip",
            showlegend=True,
        ))

        fig.add_trace(go.Scatter(
            x=x, y=df[actual_col], name="Ground Truth",
            line=dict(color="#38bdf8", width=2.5),
            hovertemplate=f"<b>Actual:</b> %{{y:.2f}} {y_label}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df[pred_col], name="XGBoost Predicted",
            line=dict(color="#f59e0b", width=2.2, dash="dash"),
            hovertemplate=f"<b>Predicted:</b> %{{y:.2f}} {y_label}<extra></extra>",
        ))

    return _apply_layout(fig, title, y_label, height)


# ── Diesel Usage Chart ─────────────────────────────────────────────────────────
def diesel_usage_chart(df: pd.DataFrame, x_col: str = "timestamp", height: int = 360) -> go.Figure:
    """Diesel power output + on/off status dual sub-panel chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
    )

    diesel_col = next((c for c in ["diesel_generation_kw", "diesel_kw"] if c in df.columns), None)
    x = df[x_col] if x_col in df.columns else df.index

    if diesel_col:
        fig.add_trace(go.Scatter(
            x=x, y=df[diesel_col], name="⛽ Diesel Power Output",
            fill="tozeroy",
            fillcolor="rgba(244,63,94,0.20)",
            line=dict(color=COLORS["diesel"], width=2),
            hovertemplate="<b>Diesel:</b> %{y:.1f} kW<extra></extra>",
        ), row=1, col=1)

    if "diesel_status" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["diesel_status"], name="Generator ON/OFF",
            fill="tozeroy",
            fillcolor="rgba(244,63,94,0.12)",
            line=dict(color="rgba(244,63,94,0.7)", width=1.5, shape="hv"),
        ), row=2, col=1)

    fig.update_layout(height=height, **LAYOUT_BASE)
    fig.update_layout(title=dict(text="Diesel Generator Operational Dynamics",
                                 font=dict(size=14, color="#e2e8f0", family="Plus Jakarta Sans")))
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Status",     row=2, col=1,
                     tickvals=[0, 1], ticktext=["OFF", "ON"], gridcolor=COLORS["grid"])
    return fig


# ── Cost Breakdown Donut ────────────────────────────────────────────────────────
def cost_breakdown_pie(cost_metrics: dict, height: int = 380) -> go.Figure:
    """Donut chart of operational cost breakdown with aurora colours."""
    labels = ["Fuel Cost", "Start Events", "Maintenance", "Battery Aging"]
    values = [
        cost_metrics.get("fuel_cost_usd", 0),
        cost_metrics.get("startup_cost_usd", 0),
        cost_metrics.get("maintenance_cost_usd", 0),
        cost_metrics.get("degradation_cost_usd", 0),
    ]
    clrs = ["#f43f5e", "#fb923c", "#facc15", "#818cf8"]
    pull  = [0.05, 0, 0, 0]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        marker=dict(colors=clrs, line=dict(color="rgba(10,16,32,0.8)", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#f8fafc", family="Plus Jakarta Sans"),
        hole=0.58,
        pull=pull,
        rotation=45,
    )])
    # Center annotation
    total = sum(values)
    fig.update_layout(
        height=height,
        annotations=[dict(
            text=f"<b>${total:,.0f}</b><br><span style='font-size:10px'>Total USD</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#f1f5f9", family="Plus Jakarta Sans"),
        )],
        title=dict(text="Operational Cost Breakdown", font=dict(size=14, color="#e2e8f0")),
        **LAYOUT_BASE,
    )
    return fig


# ── Controller Comparison Bar ──────────────────────────────────────────────────
def controller_comparison_bar(comparison_df: pd.DataFrame, height: int = 380) -> go.Figure:
    """Grouped bar chart comparing controller performance metrics."""
    fig = go.Figure()
    metrics = [
        ("total_fuel_litres",         "Fuel (L)",      "#f43f5e"),
        ("total_cost_usd",            "Cost ($ USD)",  "#facc15"),
        ("renewable_utilization_pct", "Renew. Util %", "#34d399"),
    ]
    for metric, label, color in metrics:
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                x=comparison_df["controller"],
                y=comparison_df[metric],
                name=label,
                marker_color=color,
                opacity=0.85,
                marker_line=dict(color="rgba(255,255,255,0.08)", width=1),
            ))

    fig.update_layout(
        barmode="group",
        height=height,
        bargap=0.22,
        bargroupgap=0.08,
        title=dict(text="Controller Benchmark Comparison", font=dict(size=14, color="#e2e8f0")),
        **LAYOUT_BASE,
    )
    return fig


# ── Temperature & Wind Dual-Axis ────────────────────────────────────────────────
def temperature_wind_chart(df: pd.DataFrame, x_col: str = "timestamp", height: int = 340) -> go.Figure:
    """Dual-axis chart: temperature (°C) + wind speed (m/s)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df[x_col] if x_col in df.columns else df.index

    if "temperature" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["temperature"], name="🌡️ Air Temp (°C)",
            line=dict(color=COLORS["temp"], width=2),
            hovertemplate="<b>Temp:</b> %{y:.1f}°C<extra></extra>",
        ), secondary_y=False)

    if "wind_speed" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["wind_speed"], name="💨 Wind Speed (m/s)",
            line=dict(color=COLORS["wind_speed"], width=2),
            hovertemplate="<b>Wind:</b> %{y:.1f} m/s<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(height=height, **LAYOUT_BASE)
    fig.update_layout(title=dict(text="Meteorological Telemetry (Maitri Station)",
                                 font=dict(size=14, color="#e2e8f0")))
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Wind Speed (m/s)",  secondary_y=True,  gridcolor=COLORS["grid"])
    return fig


# ── RL Reward Curve ─────────────────────────────────────────────────────────────
def rl_reward_curve(rewards: list, algorithm: str = "PPO", height: int = 340) -> go.Figure:
    """Smoothed RL episode reward curve with EMA."""
    fig = go.Figure()
    eps = list(range(1, len(rewards) + 1))

    # Raw
    fig.add_trace(go.Scatter(
        x=eps, y=rewards, name="Episode Reward (raw)",
        line=dict(color="rgba(56,189,248,0.25)", width=1),
        mode="lines",
    ))

    # EMA
    if len(rewards) > 5:
        ema = pd.Series(rewards).ewm(span=max(5, len(rewards) // 10)).mean().tolist()
        fig.add_trace(go.Scatter(
            x=eps, y=ema, name="EMA (smoothed)",
            line=dict(color="#38bdf8", width=2.5),
        ))

    return _apply_layout(fig, f"{algorithm} Training Reward Curve", "Cumulative Episode Reward", height)
