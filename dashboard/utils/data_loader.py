"""
PolarGrid AI — Centralized High-Performance Data Loader
Caches datasets, configurations, and results in memory to provide instantaneous page navigation.
"""

import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS = PROJECT_ROOT / "configs"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_data(ttl=600, show_spinner=False)
def load_master_dataset() -> pd.DataFrame:
    """Load the full 14-year master dataset once into memory."""
    path = DATA_PROCESSED / "master_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_recent_dataset(hours: int = 168) -> pd.DataFrame:
    """Return a slice of the recent N hours of the master dataset."""
    df = load_master_dataset()
    if not df.empty:
        return df.tail(hours).copy()
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_system_configs() -> Dict[str, Any]:
    """Load all YAML configuration files."""
    battery_cfg, diesel_cfg, station_cfg = {}, {}, {}
    b_path = CONFIGS / "battery_config.yaml"
    if b_path.exists():
        with open(b_path, "r") as f:
            battery_cfg = yaml.safe_load(f).get("battery", {})
    d_path = CONFIGS / "diesel_config.yaml"
    if d_path.exists():
        with open(d_path, "r") as f:
            diesel_cfg = yaml.safe_load(f).get("diesel", {})
    s_path = CONFIGS / "station_config.yaml"
    if s_path.exists():
        with open(s_path, "r") as f:
            station_cfg = yaml.safe_load(f)
    return {"battery": battery_cfg, "diesel": diesel_cfg, "station": station_cfg}


@st.cache_data(ttl=600, show_spinner=False)
def load_forecast_artifacts() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load forecast metrics and predictions."""
    metrics = {}
    m_path = RESULTS_DIR / "forecasts" / "forecast_metrics.json"
    if m_path.exists():
        with open(m_path, "r") as f:
            metrics = json.load(f)
    preds = pd.DataFrame()
    p_path = RESULTS_DIR / "forecasts" / "forecast_predictions.csv"
    if p_path.exists():
        preds = pd.read_csv(p_path)
    return metrics, preds


@st.cache_data(ttl=600, show_spinner=False)
def load_comparison_artifacts() -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load controller benchmark metrics, comparison CSV, and RL dispatch logs."""
    comparison = {}
    cmp_path = RESULTS_DIR / "comparison" / "controller_comparison.json"
    if cmp_path.exists():
        with open(cmp_path, "r") as f:
            comparison = json.load(f)

    comparison_df = pd.DataFrame()
    cmp_csv = RESULTS_DIR / "comparison" / "controller_comparison.csv"
    if cmp_csv.exists():
        comparison_df = pd.read_csv(cmp_csv)

    dispatch_logs = {}
    rl_dir = RESULTS_DIR / "rl"
    if rl_dir.exists():
        for log_file in sorted(rl_dir.glob("*_dispatch_log.csv")):
            name = log_file.stem.replace("_dispatch_log", "")
            dispatch_logs[name] = pd.read_csv(log_file)

    return comparison, comparison_df, dispatch_logs
