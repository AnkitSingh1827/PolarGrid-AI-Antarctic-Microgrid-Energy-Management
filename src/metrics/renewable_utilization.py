"""
PolarGrid AI — Renewable Utilization Metrics
Measures how effectively available renewable energy (solar + wind) is used.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_renewable_utilization(
    dispatch_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute renewable energy utilization metrics.
    
    Returns:
        dict with keys: solar_available_kwh, wind_available_kwh, total_renewable_available_kwh,
                        renewable_used_kwh, curtailment_kwh, utilization_pct,
                        solar_utilization_pct, wind_utilization_pct,
                        renewable_fraction_of_load_pct
    """
    # Get available renewable generation
    solar_avail = dispatch_df.get(
        "solar_available_kw",
        dispatch_df.get("solar_kw", dispatch_df.get("solar_generation_kw", pd.Series(np.zeros(len(dispatch_df)))))
    ).values
    
    wind_avail = dispatch_df.get(
        "wind_available_kw",
        dispatch_df.get("wind_kw", dispatch_df.get("wind_generation_kw", pd.Series(np.zeros(len(dispatch_df)))))
    ).values
    
    total_renewable = solar_avail + wind_avail
    
    # Curtailment
    curtailment = dispatch_df.get("curtailment_kw", pd.Series(np.zeros(len(dispatch_df)))).values
    
    # Renewable used = available - curtailed
    renewable_used = total_renewable - curtailment
    renewable_used = np.clip(renewable_used, 0, None)
    
    # Totals (each timestep is 1 hour, so kW = kWh)
    total_solar_kwh = float(np.sum(solar_avail))
    total_wind_kwh = float(np.sum(wind_avail))
    total_avail_kwh = float(np.sum(total_renewable))
    total_used_kwh = float(np.sum(renewable_used))
    total_curtail_kwh = float(np.sum(curtailment))
    
    # Load served
    load = dispatch_df["load_kw"].values
    total_load_kwh = float(np.sum(load))
    
    utilization_pct = (total_used_kwh / total_avail_kwh * 100.0) if total_avail_kwh > 0 else 0.0
    
    # Per-source utilization
    solar_curtail_est = curtailment * (solar_avail / np.maximum(total_renewable, 1e-6))
    wind_curtail_est = curtailment * (wind_avail / np.maximum(total_renewable, 1e-6))
    
    solar_used = float(np.sum(solar_avail - solar_curtail_est))
    wind_used = float(np.sum(wind_avail - wind_curtail_est))
    
    solar_util = (solar_used / total_solar_kwh * 100.0) if total_solar_kwh > 0 else 0.0
    wind_util = (wind_used / total_wind_kwh * 100.0) if total_wind_kwh > 0 else 0.0
    
    renewable_fraction = (total_used_kwh / total_load_kwh * 100.0) if total_load_kwh > 0 else 0.0
    
    return {
        "solar_available_kwh": round(total_solar_kwh, 2),
        "wind_available_kwh": round(total_wind_kwh, 2),
        "total_renewable_available_kwh": round(total_avail_kwh, 2),
        "renewable_used_kwh": round(total_used_kwh, 2),
        "curtailment_kwh": round(total_curtail_kwh, 2),
        "utilization_pct": round(utilization_pct, 2),
        "solar_utilization_pct": round(solar_util, 2),
        "wind_utilization_pct": round(wind_util, 2),
        "renewable_fraction_of_load_pct": round(min(100.0, renewable_fraction), 2),
    }
