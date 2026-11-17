"""
PolarGrid AI — Forecasting Tests
Tests for XGBoost forecasting models and evaluation pipeline.
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "forecasts"
MODELS_DIR = PROJECT_ROOT / "models" / "forecasting"


class TestForecastModels:
    def test_model_files_exist(self):
        """Trained model pickle files must exist."""
        for name in ["load_xgb.pkl", "solar_xgb.pkl", "wind_xgb.pkl"]:
            path = MODELS_DIR / name
            assert path.exists(), f"Model file not found: {name}"
            assert path.stat().st_size > 100, f"Model file too small (likely placeholder): {name}"

    def test_forecast_metrics_exist(self):
        """Forecast metrics JSON must exist."""
        path = RESULTS_DIR / "forecast_metrics.json"
        assert path.exists(), "forecast_metrics.json not found"

    def test_forecast_metrics_valid(self):
        """Forecast metrics must have valid values."""
        path = RESULTS_DIR / "forecast_metrics.json"
        if not path.exists():
            pytest.skip("Metrics file not found")
        
        with open(path) as f:
            metrics = json.load(f)
        
        for model_name in ["load", "solar", "wind"]:
            assert model_name in metrics, f"Missing model: {model_name}"
            m = metrics[model_name]
            assert "MAE" in m and m["MAE"] >= 0, f"Invalid MAE for {model_name}"
            assert "RMSE" in m and m["RMSE"] >= 0, f"Invalid RMSE for {model_name}"
            assert "R2" in m, f"Missing R2 for {model_name}"

    def test_forecast_r2_acceptable(self):
        """All forecast models should achieve R² > 0.80 on test data."""
        path = RESULTS_DIR / "forecast_metrics.json"
        if not path.exists():
            pytest.skip("Metrics file not found")
        
        with open(path) as f:
            metrics = json.load(f)
        
        for model_name in ["load", "solar", "wind"]:
            r2 = metrics[model_name]["R2"]
            assert r2 > 0.80, f"{model_name} R² too low: {r2:.4f}"

    def test_forecast_predictions_exist(self):
        """Forecast prediction CSV must exist and have expected columns."""
        path = RESULTS_DIR / "forecast_predictions.csv"
        assert path.exists(), "forecast_predictions.csv not found"
        
        df = pd.read_csv(path, nrows=5)
        for col in ["load_actual", "load_pred", "solar_actual", "solar_pred",
                     "wind_actual", "wind_pred"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_predictions_non_negative(self):
        """Predicted values should be non-negative."""
        path = RESULTS_DIR / "forecast_predictions.csv"
        if not path.exists():
            pytest.skip("Predictions file not found")
        
        df = pd.read_csv(path)
        for col in ["load_pred", "solar_pred", "wind_pred"]:
            # Allow small negative values from model imprecision but flag large negatives
            min_val = df[col].min()
            assert min_val > -5.0, f"Large negative prediction in {col}: {min_val}"

    def test_forecast_plots_exist(self):
        """Evaluation plots must be generated."""
        for name in ["load_forecast_plot.png", "solar_forecast_plot.png", "wind_forecast_plot.png"]:
            path = RESULTS_DIR / name
            assert path.exists(), f"Plot not found: {name}"
