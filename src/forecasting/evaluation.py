"""
PolarGrid AI — Forecasting Evaluation
Runs all forecasting models, generates evaluation reports, and plots actual vs predicted.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from src.forecasting.load_model import train_load_forecasting_model
from src.forecasting.solar_model import train_solar_forecasting_model
from src.forecasting.wind_model import train_wind_forecasting_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "forecasts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (skipping zero values)."""
    mask = actual > 1e-5
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)

def evaluate_all_forecasts():
    logger.info("Evaluating all forecasting models...")
    
    # Train and evaluate each model
    load_results = train_load_forecasting_model()
    solar_results = train_solar_forecasting_model()
    wind_results = train_wind_forecasting_model()
    
    # Compile metrics
    metrics = {
        "load": {
            "MAE": float(load_results["mae"]),
            "RMSE": float(load_results["rmse"]),
            "R2": float(load_results["r2"]),
            "MAPE": calculate_mape(load_results["test_actuals"], load_results["test_preds"])
        },
        "solar": {
            "MAE": float(solar_results["mae"]),
            "RMSE": float(solar_results["rmse"]),
            "R2": float(solar_results["r2"]),
            "MAPE": calculate_mape(solar_results["test_actuals"], solar_results["test_preds"])
        },
        "wind": {
            "MAE": float(wind_results["mae"]),
            "RMSE": float(wind_results["rmse"]),
            "R2": float(wind_results["r2"]),
            "MAPE": calculate_mape(wind_results["test_actuals"], wind_results["test_preds"])
        }
    }
    
    # Save metrics JSON
    metrics_path = RESULTS_DIR / "forecast_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved forecast metrics to {metrics_path}")
    
    # Save test set predictions to CSV for dashboard to load
    forecast_results_df = pd.DataFrame({
        "timestamp": load_results["test_timestamps"],
        "load_actual": load_results["test_actuals"],
        "load_pred": load_results["test_preds"],
        "solar_actual": solar_results["test_actuals"],
        "solar_pred": solar_results["test_preds"],
        "wind_actual": wind_results["test_actuals"],
        "wind_pred": wind_results["test_preds"]
    })
    
    predictions_path = RESULTS_DIR / "forecast_predictions.csv"
    forecast_results_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved forecast predictions to {predictions_path}")
    
    # Generate actual vs predicted plots (take a slice for visualization, e.g., first 168 hours)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    
    # Plot Load
    plt.figure(figsize=(12, 5))
    plt.plot(forecast_results_df["timestamp"].iloc[:168], forecast_results_df["load_actual"].iloc[:168], label="Actual Load", alpha=0.8)
    plt.plot(forecast_results_df["timestamp"].iloc[:168], forecast_results_df["load_pred"].iloc[:168], label="Predicted Load", alpha=0.8, linestyle="--")
    plt.title("24-Hour Load Forecasting Performance (Sample Week)")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "load_forecast_plot.png")
    plt.close()
    
    # Plot Solar
    plt.figure(figsize=(12, 5))
    plt.plot(forecast_results_df["timestamp"].iloc[:168], forecast_results_df["solar_actual"].iloc[:168], label="Actual Solar", alpha=0.8, color="orange")
    plt.plot(forecast_results_df["timestamp"].iloc[:168], forecast_results_df["solar_pred"].iloc[:168], label="Predicted Solar", alpha=0.8, linestyle="--", color="red")
    plt.title("24-Hour Solar Generation Forecasting Performance (Sample Week)")
    plt.xlabel("Timestamp")
    plt.ylabel("Solar Generation (kW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "solar_forecast_plot.png")
    plt.close()
    
    # Plot Wind
    plt.figure(figsize=(12, 5))
    plt.plot(forecast_results_df["timestamp"].iloc[:168], forecast_results_df["wind_actual"].iloc[:168], label="Actual Wind", alpha=0.8, color="blue")
    plt.plot(forecast_results_df["timestamp"].iloc[:168], forecast_results_df["wind_pred"].iloc[:168], label="Predicted Wind", alpha=0.8, linestyle="--", color="cyan")
    plt.title("24-Hour Wind Generation Forecasting Performance (Sample Week)")
    plt.xlabel("Timestamp")
    plt.ylabel("Wind Generation (kW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "wind_forecast_plot.png")
    plt.close()
    
    logger.info("Saved all evaluation plots under results/forecasts/")
    return metrics

if __name__ == "__main__":
    evaluate_all_forecasts()
