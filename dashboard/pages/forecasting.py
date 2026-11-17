"""
PolarGrid AI — Dashboard Forecasting Page v2
AI forecasting model performance, confidence-band charts, and actual vs predicted profiles.
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.glass_theme import apply_glass_theme, render_header, render_section_header, render_html
from dashboard.components.kpi_cards import render_kpi_row, render_status_badge
from dashboard.components.charts import forecast_comparison_chart

RESULTS_DIR = PROJECT_ROOT / "results" / "forecasts"


from dashboard.utils.data_loader import load_forecast_artifacts


def render_forecasting_page():
    apply_glass_theme()
    render_header(
        title="24-Hour AI Generation & Load Forecasting",
        subtitle="XGBoost gradient-boosted models predicting Antarctic electrical demand and variable renewable availability",
        badge="XGBOOST AI • PREDICTIVE DISPATCH",
    )

    metrics, preds = load_forecast_artifacts()

    if not metrics:
        render_html("""
        <div style="
            padding:32px;text-align:center;
            background:rgba(14,22,42,0.65);
            backdrop-filter:blur(20px);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:20px;
        ">
            <div style="font-size:2.5rem;margin-bottom:12px;">📊</div>
            <div style="color:#e2e8f0;font-weight:700;font-size:1.1rem;margin-bottom:6px;">No Forecast Results Found</div>
            <div style="color:#64748b;font-size:0.85rem;">Run <code style="color:#38bdf8;">python -m src.forecasting.evaluation</code> to generate results.</div>
        </div>
        """)
        return

    # ── Model Accuracy KPIs ─────────────────────────────────────────────────
    render_section_header("Model Accuracy Benchmarks",
                          "Out-of-sample test performance on 18,400+ hourly validation timesteps", icon="🎯")

    render_kpi_row([
        {"title": "Load Forecast R²",
         "value": f"{metrics['load']['R2']:.4f}",
         "subtitle": f"MAE: {metrics['load']['MAE']:.2f} kW  |  MAPE: {metrics['load']['MAPE']:.1f}%",
         "delta": "High Confidence",
         "delta_color": "good",
         "icon": "⚡",
         "glow_color": "rgba(251,191,36,0.35)"},
        {"title": "Solar Forecast R²",
         "value": f"{metrics['solar']['R2']:.4f}",
         "subtitle": f"MAE: {metrics['solar']['MAE']:.3f} kW  (STC calibrated)",
         "delta": "Near Perfect",
         "delta_color": "good",
         "icon": "☀️",
         "glow_color": "rgba(251,191,36,0.30)"},
        {"title": "Wind Forecast R²",
         "value": f"{metrics['wind']['R2']:.4f}",
         "subtitle": f"MAE: {metrics['wind']['MAE']:.2f} kW  (Cubic Power Curve)",
         "delta": "Robust",
         "delta_color": "good",
         "icon": "🌬️",
         "glow_color": "rgba(56,189,248,0.30)"},
    ])

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    from dashboard.components.glass_theme import render_html
    info_html = """<div style="display:flex;gap:28px;flex-wrap:wrap;padding:20px 26px;background:linear-gradient(135deg,rgba(14,28,52,0.65),rgba(10,18,36,0.55));backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.08);border-radius:20px;margin-bottom:8px;"><div><div style="font-size:0.70rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Algorithm</div><div style="color:#e2e8f0;font-weight:700;">XGBoost Regressor (Tuned)</div></div><div><div style="font-size:0.70rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Features</div><div style="color:#e2e8f0;font-weight:700;">42 Lag + Calendar + Weather Features</div></div><div><div style="font-size:0.70rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Training Data</div><div style="color:#e2e8f0;font-weight:700;">ERA5 Reanalysis 2012–2022</div></div><div><div style="font-size:0.70rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Test Set</div><div style="color:#e2e8f0;font-weight:700;">2023–2025 (18,400+ hrs)</div></div><div><div style="font-size:0.70rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Dispatch Integration</div><div style="color:#38bdf8;font-weight:700;">24h Look-Ahead MPC</div></div></div>"""
    render_html(info_html)

    # ── Detailed Error Table ────────────────────────────────────────────────
    render_section_header("Detailed Error Statistics",
                          "Precision error breakdown across test evaluation horizons", icon="📋")

    metrics_df = pd.DataFrame({
        "Predictive Target": [
            "Station Electrical & Heating Load",
            "Solar PV Farm Generation",
            "Wind Turbine Generator Set",
        ],
        "Algorithm": ["XGBoost (Tuned)"] * 3,
        "MAE (kW)":  [f"{metrics['load']['MAE']:.3f}",  f"{metrics['solar']['MAE']:.3f}",  f"{metrics['wind']['MAE']:.3f}"],
        "RMSE (kW)": [f"{metrics['load']['RMSE']:.3f}", f"{metrics['solar']['RMSE']:.3f}", f"{metrics['wind']['RMSE']:.3f}"],
        "R²":        [f"{metrics['load']['R2']:.4f}",   f"{metrics['solar']['R2']:.4f}",   f"{metrics['wind']['R2']:.4f}"],
        "MAPE (%)":  [f"{metrics['load']['MAPE']:.2f}", f"{metrics['solar']['MAPE']:.2f}", f"{metrics['wind']['MAPE']:.2f}"],
    })
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    # ── Interactive Forecast Horizon Viewer ─────────────────────────────────
    if not preds.empty:
        render_section_header("Interactive Horizon Visualizer",
                              "Compare actual sensor measurements against 24h predictive curves with ±1σ confidence bands", icon="📈")

        n_hours = st.slider("Horizon Window (Hours)", 24, min(720, len(preds)), 168,
                            step=24, help="Adjust window to inspect multi-day forecast precision")
        display_df = preds.head(n_hours)

        # Load forecast
        if "load_actual" in display_df.columns and "load_pred" in display_df.columns:
            st.plotly_chart(
                forecast_comparison_chart(
                    display_df, "load_actual", "load_pred",
                    title="Station Load Demand — Ground Truth vs XGBoost Prediction",
                    y_label="Demand (kW)",
                ),
                use_container_width=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            if "solar_actual" in display_df.columns:
                st.plotly_chart(
                    forecast_comparison_chart(
                        display_df, "solar_actual", "solar_pred",
                        title="Solar PV Output (kW)", y_label="Solar (kW)", height=320,
                    ),
                    use_container_width=True,
                )
        with col2:
            if "wind_actual" in display_df.columns:
                st.plotly_chart(
                    forecast_comparison_chart(
                        display_df, "wind_actual", "wind_pred",
                        title="Wind Turbine Output (kW)", y_label="Wind (kW)", height=320,
                    ),
                    use_container_width=True,
                )


if __name__ == "__main__":
    render_forecasting_page()
