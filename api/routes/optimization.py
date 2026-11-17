"""
PolarGrid AI — Optimization API Routes
"""

import sys
import logging
from pathlib import Path
import yaml

from fastapi import APIRouter, HTTPException
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import OptimizeRequest, OptimizeResponse

logger = logging.getLogger(__name__)

CONFIGS = PROJECT_ROOT / "configs"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

router = APIRouter(tags=["Optimization"])


def _load_configs():
    with open(CONFIGS / "battery_config.yaml", "r") as f:
        b = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml", "r") as f:
        d = yaml.safe_load(f)["diesel"]
    return {"battery": b, "diesel": d}


@router.post("/optimize", response_model=OptimizeResponse)
def run_optimization(request: OptimizeRequest):
    """Run MILP optimization for 24h energy scheduling."""
    try:
        from src.optimization.milp_model import solve_milp
        
        params = _load_configs()
        master_path = DATA_PROCESSED / "master_dataset.csv"
        if not master_path.exists():
            raise HTTPException(status_code=503, detail="Processed dataset not found.")
        
        master = pd.read_csv(master_path)
        sample = master.tail(request.horizon_hours)
        
        load = sample["load_kw"].values
        critical = sample["critical_load_kw"].values
        solar = sample["solar_generation_kw"].values
        wind = sample["wind_generation_kw"].values
        
        schedule_df, solved = solve_milp(
            load, critical, solar, wind, params,
            solver_name=request.solver,
        )
        
        fuel_intercept = params["diesel"]["fuel_intercept_lph"]
        fuel_slope = params["diesel"]["fuel_slope_lph_per_kw"]
        fuel_cost_l = params["diesel"]["fuel_cost_per_liter"]
        
        total_cost = 0.0
        for _, row in schedule_df.iterrows():
            if row.get("diesel_on", 0) > 0:
                fuel = fuel_intercept + fuel_slope * row.get("diesel_gen", 0)
                total_cost += fuel * fuel_cost_l
        
        return OptimizeResponse(
            solved_optimally=solved,
            solver_used=request.solver if solved else "heuristic",
            schedule=schedule_df.to_dict(orient="records"),
            total_cost=round(total_cost, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
