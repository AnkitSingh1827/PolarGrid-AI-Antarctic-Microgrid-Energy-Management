"""
PolarGrid AI — Cost Metrics
Overall operational cost analysis for energy dispatch.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_cost_metrics(
    dispatch_df: pd.DataFrame,
    params: dict,
) -> Dict[str, float]:
    """
    Compute overall operational cost metrics from dispatch results.
    
    Returns:
        dict with keys: fuel_cost_usd, startup_cost_usd, maintenance_cost_usd,
                        degradation_cost_usd, total_cost_usd, avg_cost_per_hour,
                        avg_cost_per_kwh, baseline_cost_usd, cost_savings_pct
    """
    fuel_intercept = params["diesel"]["fuel_intercept_lph"]
    fuel_slope = params["diesel"]["fuel_slope_lph_per_kw"]
    fuel_cost_per_l = params["diesel"]["fuel_cost_per_liter"]
    startup_cost_usd = params["diesel"]["startup_cost_usd"]
    maint_cost_ph = params["diesel"]["maintenance_cost_per_hour"]
    deg_cost_kwh = params["battery"]["degradation_cost_per_kwh"]
    
    n_steps = len(dispatch_df)
    
    # Diesel generation and status
    diesel_gen = dispatch_df.get("diesel_generation_kw", dispatch_df.get("diesel_kw", pd.Series(np.zeros(n_steps)))).values
    diesel_status = dispatch_df.get("diesel_status", pd.Series(np.zeros(n_steps))).values
    
    # Fuel consumption
    fuel_lph = np.where(diesel_status > 0, fuel_intercept + fuel_slope * diesel_gen, 0.0)
    total_fuel_litres = float(np.sum(fuel_lph))
    total_fuel_cost = total_fuel_litres * fuel_cost_per_l
    
    # Startup events
    startup_events = 0
    for i in range(1, n_steps):
        if diesel_status[i] > 0 and diesel_status[i-1] == 0:
            startup_events += 1
    if n_steps > 0 and diesel_status[0] > 0:
        startup_events += 1
    total_startup_cost = startup_events * startup_cost_usd
    
    # Maintenance cost
    diesel_hours = int(np.sum(diesel_status > 0))
    total_maint_cost = diesel_hours * maint_cost_ph
    
    # Battery degradation cost
    if "battery_power_kw" in dispatch_df.columns:
        bat_throughput = float(np.sum(np.abs(dispatch_df["battery_power_kw"].values)))
    elif "battery_charge_kw" in dispatch_df.columns:
        bat_throughput = float(np.sum(dispatch_df["battery_charge_kw"].values) +
                               np.sum(dispatch_df.get("battery_discharge_kw", pd.Series(np.zeros(n_steps))).values))
    else:
        bat_throughput = 0.0
    total_deg_cost = bat_throughput * deg_cost_kwh
    
    # Total operational cost
    total_cost = total_fuel_cost + total_startup_cost + total_maint_cost + total_deg_cost
    
    # Per-unit metrics
    load = dispatch_df["load_kw"].values
    total_load_kwh = float(np.sum(load))
    
    avg_cost_per_hour = total_cost / max(1, n_steps)
    avg_cost_per_kwh = total_cost / max(1, total_load_kwh)
    
    # Baseline cost (all diesel)
    baseline_fuel = np.sum(fuel_intercept + fuel_slope * load)
    baseline_cost = baseline_fuel * fuel_cost_per_l + n_steps * maint_cost_ph
    
    cost_savings = (1.0 - total_cost / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
    
    return {
        "fuel_cost_usd": round(total_fuel_cost, 2),
        "startup_cost_usd": round(total_startup_cost, 2),
        "maintenance_cost_usd": round(total_maint_cost, 2),
        "degradation_cost_usd": round(total_deg_cost, 2),
        "total_cost_usd": round(total_cost, 2),
        "avg_cost_per_hour": round(avg_cost_per_hour, 4),
        "avg_cost_per_kwh": round(avg_cost_per_kwh, 4),
        "baseline_cost_usd": round(baseline_cost, 2),
        "cost_savings_pct": round(cost_savings, 2),
        "diesel_run_hours": diesel_hours,
        "diesel_startups": startup_events,
        "total_fuel_litres": round(total_fuel_litres, 2),
    }
