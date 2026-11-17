"""
PolarGrid AI — Fuel Savings Metrics
Calculates fuel consumption and savings relative to diesel-only baseline.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_fuel_consumption(
    diesel_generation_kw: np.ndarray,
    diesel_status: np.ndarray,
    fuel_intercept_lph: float = 5.0,
    fuel_slope_lph_per_kw: float = 0.25,
) -> np.ndarray:
    """Calculate hourly fuel consumption from diesel generation profile."""
    fuel_lph = np.where(
        diesel_status > 0,
        fuel_intercept_lph + fuel_slope_lph_per_kw * diesel_generation_kw,
        0.0,
    )
    return fuel_lph


def calculate_diesel_only_fuel(
    load_kw: np.ndarray,
    fuel_intercept_lph: float = 5.0,
    fuel_slope_lph_per_kw: float = 0.25,
) -> np.ndarray:
    """Calculate fuel if all load were served by diesel (no renewables/battery)."""
    return fuel_intercept_lph + fuel_slope_lph_per_kw * load_kw


def compute_fuel_savings(
    dispatch_df: pd.DataFrame,
    params: dict,
) -> Dict[str, float]:
    """
    Compute fuel savings metrics comparing dispatched results to a diesel-only baseline.
    
    Returns:
        dict with keys: total_fuel_litres, baseline_fuel_litres, fuel_saved_litres,
                        fuel_savings_pct, avg_fuel_per_hour, co2_avoided_kg
    """
    fuel_intercept = params["diesel"]["fuel_intercept_lph"]
    fuel_slope = params["diesel"]["fuel_slope_lph_per_kw"]
    
    # Actual fuel consumed
    if "fuel_lph" in dispatch_df.columns:
        actual_fuel = dispatch_df["fuel_lph"].values
    else:
        diesel_gen = dispatch_df.get("diesel_generation_kw", dispatch_df.get("diesel_kw", np.zeros(len(dispatch_df)))).values
        diesel_status = dispatch_df.get("diesel_status", np.ones(len(dispatch_df))).values
        actual_fuel = calculate_fuel_consumption(diesel_gen, diesel_status, fuel_intercept, fuel_slope)
    
    total_fuel = float(np.sum(actual_fuel))
    
    # Baseline: diesel serves full load
    load = dispatch_df["load_kw"].values
    baseline_fuel = calculate_diesel_only_fuel(load, fuel_intercept, fuel_slope)
    total_baseline = float(np.sum(baseline_fuel))
    
    fuel_saved = total_baseline - total_fuel
    savings_pct = (fuel_saved / total_baseline * 100.0) if total_baseline > 0 else 0.0
    
    # CO2 factor: ~2.68 kg CO2 per litre of diesel
    co2_factor = 2.68
    
    n_hours = len(dispatch_df)
    
    return {
        "total_fuel_litres": round(total_fuel, 2),
        "baseline_fuel_litres": round(total_baseline, 2),
        "fuel_saved_litres": round(fuel_saved, 2),
        "fuel_savings_pct": round(savings_pct, 2),
        "avg_fuel_per_hour": round(total_fuel / max(1, n_hours), 4),
        "co2_avoided_kg": round(fuel_saved * co2_factor, 2),
        "co2_total_kg": round(total_fuel * co2_factor, 2),
    }
