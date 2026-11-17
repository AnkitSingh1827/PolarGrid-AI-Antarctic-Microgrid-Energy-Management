"""
PolarGrid AI — Battery Metrics
Battery health, utilization, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_battery_metrics(
    dispatch_df: pd.DataFrame,
    params: dict,
) -> Dict[str, float]:
    """
    Compute battery storage performance metrics.
    
    Returns:
        dict with keys: total_charge_kwh, total_discharge_kwh, throughput_kwh,
                        equivalent_full_cycles, avg_soc, min_soc, max_soc,
                        soc_std, charge_hours, discharge_hours, idle_hours,
                        avg_depth_of_discharge, roundtrip_efficiency_pct
    """
    capacity = params["battery"]["capacity_kwh"]
    soc_min = params["battery"]["soc_min"]
    soc_max = params["battery"]["soc_max"]
    usable_capacity = capacity * (soc_max - soc_min)
    
    # Get battery power flows
    if "battery_charge_kw" in dispatch_df.columns:
        charge = dispatch_df["battery_charge_kw"].values
        discharge = dispatch_df["battery_discharge_kw"].values
    elif "battery_power_kw" in dispatch_df.columns:
        bat_power = dispatch_df["battery_power_kw"].values
        charge = np.where(bat_power > 0, bat_power, 0.0)
        discharge = np.where(bat_power < 0, -bat_power, 0.0)
    else:
        charge = np.zeros(len(dispatch_df))
        discharge = np.zeros(len(dispatch_df))
    
    # SOC profile
    soc = dispatch_df.get("battery_soc", pd.Series(np.full(len(dispatch_df), 0.5))).values
    
    # Totals (hourly data, so kW = kWh)
    total_charge = float(np.sum(charge))
    total_discharge = float(np.sum(discharge))
    throughput = total_charge + total_discharge
    
    # Equivalent full cycles
    eq_cycles = throughput / (2.0 * usable_capacity) if usable_capacity > 0 else 0.0
    
    # SOC statistics
    avg_soc = float(np.mean(soc))
    min_soc = float(np.min(soc))
    max_soc = float(np.max(soc))
    soc_std = float(np.std(soc))
    
    # Hours in each state
    charge_hours = int(np.sum(charge > 0.01))
    discharge_hours = int(np.sum(discharge > 0.01))
    idle_hours = len(dispatch_df) - charge_hours - discharge_hours
    
    # Average depth of discharge per cycle
    # Approximate: total discharge / (number of discharge events * usable capacity)
    n_discharge_events = max(1, discharge_hours)
    avg_dod = (total_discharge / (n_discharge_events * usable_capacity)) if usable_capacity > 0 else 0.0
    avg_dod = min(1.0, avg_dod)
    
    # Roundtrip efficiency
    roundtrip_eff = (total_discharge / total_charge * 100.0) if total_charge > 0 else 0.0
    
    # Degradation cost
    deg_cost = params["battery"]["degradation_cost_per_kwh"]
    degradation_cost = throughput * deg_cost
    
    return {
        "total_charge_kwh": round(total_charge, 2),
        "total_discharge_kwh": round(total_discharge, 2),
        "throughput_kwh": round(throughput, 2),
        "equivalent_full_cycles": round(eq_cycles, 2),
        "avg_soc": round(avg_soc, 4),
        "min_soc": round(min_soc, 4),
        "max_soc": round(max_soc, 4),
        "soc_std": round(soc_std, 4),
        "charge_hours": charge_hours,
        "discharge_hours": discharge_hours,
        "idle_hours": idle_hours,
        "avg_depth_of_discharge": round(avg_dod, 4),
        "roundtrip_efficiency_pct": round(roundtrip_eff, 2),
        "degradation_cost_usd": round(degradation_cost, 2),
    }
