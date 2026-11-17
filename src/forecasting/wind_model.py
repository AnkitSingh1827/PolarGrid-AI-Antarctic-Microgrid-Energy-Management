"""
PolarGrid AI — Wind Generation Forecasting Model
Trains an XGBoost model to predict wind generation 24–48 hours ahead.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from src.forecasting.load_model import create_forecasting_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "forecasting"

def train_wind_forecasting_model() -> dict:
    logger.info("Training Wind Generation Forecasting Model...")
    
    # Load dataset
    master_df = pd.read_csv(DATA_PROCESSED / "master_dataset.csv")
    
    # Build features
    df_feat, feature_names = create_forecasting_features(master_df, "wind_generation_kw")
    
    # Chronological split: 70% Train, 15% Val, 15% Test
    n = len(df_feat)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:]
    
    X_train, y_train = train_df[feature_names], train_df["wind_generation_kw"]
    X_val, y_val = val_df[feature_names], val_df["wind_generation_kw"]
    X_test, y_test = test_df[feature_names], test_df["wind_generation_kw"]
    
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
    model_path = MODELS_DIR / "wind_xgb.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Wind Forecasting Model saved at {model_path}")
    
    # Predictions
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, None) # Generation cannot be negative
    
    # Calculate simple metrics
    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    r2 = 1 - (np.sum((y_test - preds)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    logger.info(f"Wind Forecast Test Set Metrics: MAE = {mae:.2f} kW, RMSE = {rmse:.2f} kW, R2 = {r2:.4f}")
    
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
    train_wind_forecasting_model()
