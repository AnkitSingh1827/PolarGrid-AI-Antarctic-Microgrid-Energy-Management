"""
PolarGrid AI — Forecasting API Routes
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import ForecastRequest, ForecastResponse, ForecastPoint

logger = logging.getLogger(__name__)
RESULTS_DIR = PROJECT_ROOT / "results"

router = APIRouter(tags=["Forecasting"])


@router.post("/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest):
    """Generate energy forecasts for the specified horizon."""
    try:
        preds_path = RESULTS_DIR / "forecasts" / "forecast_predictions.csv"
        if not preds_path.exists():
            raise HTTPException(status_code=503, detail="Forecast predictions not found. Run model evaluation first.")
        
        preds = pd.read_csv(preds_path)
        n = min(request.hours_ahead, len(preds))
        sample = preds.head(n)
        
        forecasts = []
        for _, row in sample.iterrows():
            forecasts.append(ForecastPoint(
                timestamp=str(row.get("timestamp", "")),
                load_kw=float(row.get("load_pred", 0)),
                solar_kw=float(row.get("solar_pred", 0)),
                wind_kw=float(row.get("wind_pred", 0)),
            ))
        
        metrics = None
        metrics_path = RESULTS_DIR / "forecasts" / "forecast_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        
        return ForecastResponse(
            model_type=request.model_type,
            hours_ahead=n,
            forecasts=forecasts,
            metrics=metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
