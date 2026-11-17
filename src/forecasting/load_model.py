"""
PolarGrid AI — Load Forecasting Model
Trains an XGBoost model to predict station load 24–48 hours ahead.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "forecasting"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def create_forecasting_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create features for forecasting: temporal, weather, and lags."""
    df = df.copy()
    
    # Feature columns
    feature_cols = [
        "temperature", "pressure", "humidity", "wind_speed", "wind_direction", 
        "solar_radiation", "cloud_cover", "hour", "day", "month", "day_of_year", 
        "day_of_week", "season", "polar_night", "extreme_cold"
    ]
    
    # Lags for target (24h and 48h lags are standard for next-day forecast)
    df[f"{target_col}_lag_24"] = df[target_col].shift(24)
    df[f"{target_col}_lag_48"] = df[target_col].shift(48)
    
    # Rolling features on lag_24 to avoid leakage (e.g. mean load of previous day)
    df[f"{target_col}_roll_mean_24"] = df[f"{target_col}_lag_24"].rolling(window=24).mean()
    df[f"{target_col}_roll_std_24"] = df[f"{target_col}_lag_24"].rolling(window=24).std()
    
    all_features = feature_cols + [
        f"{target_col}_lag_24", f"{target_col}_lag_48", 
        f"{target_col}_roll_mean_24", f"{target_col}_roll_std_24"
    ]
    
    # Drop rows with NaNs caused by lags/rolling
    df = df.dropna(subset=all_features + [target_col])
    
    return df, all_features

def train_load_forecasting_model() -> dict:
    logger.info("Training Load Forecasting Model...")
    
    # Load dataset
    master_df = pd.read_csv(DATA_PROCESSED / "master_dataset.csv")
    
    # Build features
    df_feat, feature_names = create_forecasting_features(master_df, "load_kw")
    
    # Chronological split: 70% Train, 15% Val, 15% Test
    n = len(df_feat)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:]
    
    X_train, y_train = train_df[feature_names], train_df["load_kw"]
    X_val, y_val = val_df[feature_names], val_df["load_kw"]
    X_test, y_test = test_df[feature_names], test_df["load_kw"]
    
    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Save model
    model_path = MODELS_DIR / "load_xgb.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Load Forecasting Model saved at {model_path}")
    
    # Predictions
    preds = model.predict(X_test)
    
    # Calculate simple metrics
    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    r2 = 1 - (np.sum((y_test - preds)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    logger.info(f"Load Forecast Test Set Metrics: MAE = {mae:.2f} kW, RMSE = {rmse:.2f} kW, R2 = {r2:.4f}")
    
    return {
        "model": model,
        "features": feature_names,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "test_actuals": y_test.values,
        "test_preds": preds,
        "test_timestamps": test_df["timestamp"].values
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_load_forecasting_model()
