"""
PolarGrid AI — Energy Dispatch & Metrics API Routes
"""

import sys
import json
import logging
from pathlib import Path
import yaml

from fastapi import APIRouter, HTTPException
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    DispatchRequest, DispatchResponse, MetricsResponse,
    AlertsResponse, AlertItem,
)

logger = logging.getLogger(__name__)

CONFIGS = PROJECT_ROOT / "configs"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

router = APIRouter(tags=["Dispatch & Metrics"])


def _load_configs():
    with open(CONFIGS / "battery_config.yaml", "r") as f:
        b = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml", "r") as f:
        d = yaml.safe_load(f)["diesel"]
    return {"battery": b, "diesel": d}


@router.post("/dispatch", response_model=DispatchResponse)
def execute_dispatch(request: DispatchRequest):
    """Execute a single-step safe energy dispatch decision."""
    try:
        params = _load_configs()
        mode = request.controller_mode.lower()
        
        if mode in ("rule_based", "rule"):
            from src.dispatch.fallback_controller import run_rule_based_dispatch
            result = run_rule_based_dispatch(
                load_kw=request.load_kw,
                critical_load_kw=request.load_kw * 0.4,
                solar_gen_kw=request.solar_gen_kw,
                wind_gen_kw=request.wind_gen_kw,
                current_soc=request.battery_soc,
                temp_c=request.temperature,
                params=params,
            )
        else:
            from src.dispatch.controller import DispatchController
            ctrl_mode = mode if mode.startswith("rl_") else f"rl_{mode}"
            ctrl = DispatchController(mode=ctrl_mode, params=params)
            ctrl.soc = request.battery_soc
            result = ctrl.dispatch_step(
                load_kw=request.load_kw,
                critical_load_kw=request.load_kw * 0.4,
                solar_gen_kw=request.solar_gen_kw,
                wind_gen_kw=request.wind_gen_kw,
                temperature=request.temperature,
            )
        
        return DispatchResponse(
            battery_power_kw=result.get("battery_power_kw", 0.0),
            battery_charge_kw=result.get("battery_charge_kw", 0.0),
            battery_discharge_kw=result.get("battery_discharge_kw", 0.0),
            diesel_generation_kw=result.get("diesel_generation_kw", 0.0),
            diesel_status=int(result.get("diesel_status", 0)),
            unmet_load_kw=result.get("unmet_load_kw", 0.0),
            curtailment_kw=result.get("curtailment_kw", 0.0),
            controller_mode=request.controller_mode,
        )
    except Exception as e:
        logger.error(f"Dispatch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Retrieve system performance and reliability metrics."""
    try:
        params = _load_configs()
        master_path = DATA_PROCESSED / "master_dataset.csv"
        if not master_path.exists():
            raise HTTPException(status_code=503, detail="Processed dataset not found.")
        
        master = pd.read_csv(master_path)
        recent = master.tail(720)  # Last 30 days
        
        from src.metrics.fuel_savings import compute_fuel_savings
        from src.metrics.renewable_utilization import compute_renewable_utilization
        from src.metrics.reliability import compute_reliability
        from src.metrics.battery_metrics import compute_battery_metrics
        from src.metrics.cost_metrics import compute_cost_metrics
        
        return MetricsResponse(
            fuel_savings=compute_fuel_savings(recent, params),
            renewable_utilization=compute_renewable_utilization(recent),
            reliability=compute_reliability(recent),
            battery=compute_battery_metrics(recent, params),
            cost=compute_cost_metrics(recent, params),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=AlertsResponse)
def get_alerts():
    """Retrieve data-driven system alerts."""
    try:
        from dashboard.components.alerts import generate_system_alerts
        
        master_path = DATA_PROCESSED / "master_dataset.csv"
        if not master_path.exists():
            return AlertsResponse(alerts=[], count=0)
        
        master = pd.read_csv(master_path)
        recent = master.tail(24)
        
        forecast_metrics = None
        fm_path = RESULTS_DIR / "forecasts" / "forecast_metrics.json"
        if fm_path.exists():
            with open(fm_path, "r") as f:
                forecast_metrics = json.load(f)
        
        alert_list = generate_system_alerts(dispatch_df=recent, forecast_metrics=forecast_metrics)
        
        return AlertsResponse(
            alerts=[AlertItem(**a) for a in alert_list],
            count=len(alert_list),
        )
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
