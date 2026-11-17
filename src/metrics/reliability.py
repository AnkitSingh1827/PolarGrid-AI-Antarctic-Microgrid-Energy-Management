"""
PolarGrid AI — Reliability Metrics
Measures system reliability: unmet load, critical load protection, availability.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_reliability(
    dispatch_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute system reliability metrics.
    
    Returns:
        dict with keys: total_load_kwh, unmet_load_kwh, load_served_kwh,
                        reliability_pct, loss_of_load_probability,
                        loss_of_load_hours, loss_of_load_expectation,
                        critical_load_violations, max_unmet_kw
    """
    load = dispatch_df["load_kw"].values
    unmet = dispatch_df.get("unmet_load_kw", pd.Series(np.zeros(len(dispatch_df)))).values
    
    total_load = float(np.sum(load))
    total_unmet = float(np.sum(unmet))
    total_served = total_load - total_unmet
    
    n_steps = len(dispatch_df)
    
    # Reliability percentage
    reliability = (total_served / total_load * 100.0) if total_load > 0 else 100.0
    
    # Loss of Load Probability (LOLP): fraction of hours with any unmet load
    lol_hours = int(np.sum(unmet > 0.01))
    lolp = lol_hours / max(1, n_steps)
    
    # Loss of Load Expectation (LOLE): expected hours of unmet load per year
    # If evaluation period != 8760h, normalize
    hours_per_year = 8760
    lole = lolp * hours_per_year
    
    # Critical load violations
    critical_load = dispatch_df.get(
        "critical_load_kw",
        pd.Series(load * 0.4)  # Default critical fraction if not available
    ).values
    
    # A critical violation occurs when unmet > non-critical portion
    non_critical = load - critical_load
    critical_violations = int(np.sum(unmet > non_critical + 0.01))
    
    # Maximum single-hour unmet
    max_unmet = float(np.max(unmet)) if len(unmet) > 0 else 0.0
    
    # Energy Not Served (ENS) — same as total_unmet but named for industry standard
    ens = total_unmet
    
    return {
        "total_load_kwh": round(total_load, 2),
        "unmet_load_kwh": round(total_unmet, 2),
        "load_served_kwh": round(total_served, 2),
        "reliability_pct": round(reliability, 4),
        "loss_of_load_probability": round(lolp, 6),
        "loss_of_load_hours": lol_hours,
        "loss_of_load_expectation_hours_per_year": round(lole, 2),
        "critical_load_violations": critical_violations,
        "max_unmet_kw": round(max_unmet, 2),
        "energy_not_served_kwh": round(ens, 2),
    }
