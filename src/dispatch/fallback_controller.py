"""
PolarGrid AI — Fallback Controller
Rule-based logic to handle dispatch when optimization models fail or are bypassed.
Enforces: Renewable -> Battery -> Diesel priority.
"""

import logging
from typing import Dict, Any
from src.dispatch.safety_constraints import enforce_safety_constraints

logger = logging.getLogger(__name__)

def run_rule_based_dispatch(
    load_kw: float,
    critical_load_kw: float,
    solar_gen_kw: float,
    wind_gen_kw: float,
    current_soc: float,
    temp_c: float,
    params: dict,
    diesel_status: int = 0
) -> Dict[str, Any]:
    """
    Standard Rule-Based Dispatch Controller.
    Defines fallback target actions and wraps them in hard safety constraints.
    """
    net_deficit = load_kw - (solar_gen_kw + wind_gen_kw)
    
    # Target values before safety layer
    target_battery = 0.0
    target_diesel = 0.0
    target_diesel_status = False
    
    if net_deficit <= 0:
        # Renewable surplus: charge battery
        target_battery = -net_deficit # Positive charging request
        target_diesel = 0.0
        target_diesel_status = False
    else:
        # Deficit: discharge battery
        target_battery = -net_deficit # Negative discharging request
        
        # Estimate if battery is enough
        capacity = params["battery"]["capacity_kwh"]
        soc_min = params["battery"]["soc_min"]
        discharge_eff = params["battery"]["discharge_efficiency"]
        max_discharge = params["battery"]["max_discharge_power_kw"]
        
        max_p_discharge_by_soc = (current_soc - soc_min) * capacity * discharge_eff
        avail_discharge = min(max_discharge, max_p_discharge_by_soc)
        
        rem_deficit = net_deficit - avail_discharge
        
        if rem_deficit > 0:
            # Need diesel generator backup
            target_diesel_status = True
            target_diesel = rem_deficit
        else:
            target_diesel_status = False
            target_diesel = 0.0
            
    # Run through the safety layer to project/clip to feasible limits
    safe_dispatch = enforce_safety_constraints(
        battery_action_kw=target_battery,
        diesel_action_kw=target_diesel,
        diesel_on_off=target_diesel_status,
        current_soc=current_soc,
        current_load_kw=load_kw,
        critical_load_kw=critical_load_kw,
        solar_gen_kw=solar_gen_kw,
        wind_gen_kw=wind_gen_kw,
        temp_c=temp_c,
        params=params
    )
    
    return safe_dispatch
