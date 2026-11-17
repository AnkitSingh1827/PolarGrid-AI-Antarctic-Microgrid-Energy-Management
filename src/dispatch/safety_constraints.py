"""
PolarGrid AI — Safe Energy Dispatch hard constraints
Hard safety layer to clip or project RL controller decisions to feasible physical limits.
Prevents critical load shedding, battery SOC violations, and generator capability limits.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def enforce_safety_constraints(
    battery_action_kw: float,  # >0: charge, <0: discharge
    diesel_action_kw: float,
    diesel_on_off: bool,
    current_soc: float,
    current_load_kw: float,
    critical_load_kw: float,
    solar_gen_kw: float,
    wind_gen_kw: float,
    temp_c: float,
    params: dict
) -> Dict[str, Any]:
    """
    Enforces physical and security constraints on the dispatched power values.
    Returns a dictionary of safe, feasible actions.
    """
    # 1. Load parameters
    soc_min = params["battery"]["soc_min"]
    soc_max = params["battery"]["soc_max"]
    max_charge = params["battery"]["max_charge_power_kw"]
    max_discharge = params["battery"]["max_discharge_power_kw"]
    charge_eff = params["battery"]["charge_efficiency"]
    discharge_eff = params["battery"]["discharge_efficiency"]
    
    # Cold derate battery capacity if applicable
    cold_threshold = params["battery"]["cold_derating_threshold_c"]
    cold_factor = params["battery"]["cold_derating_factor"]
    capacity = params["battery"]["capacity_kwh"]
    if temp_c < cold_threshold:
        loss = (cold_threshold - temp_c) * cold_factor
        capacity = capacity * max(0.5, 1.0 - loss)
        
    d_capacity = params["diesel"]["capacity_kw"]
    d_min_load = d_capacity * params["diesel"]["min_load_fraction"]
    
    # 2. Enforce battery charging/discharging power boundaries based on current SOC
    # If battery is full, cannot charge. If empty, cannot discharge.
    max_charge_by_soc = ((soc_max - current_soc) * capacity) / charge_eff
    max_discharge_by_soc = (current_soc - soc_min) * capacity * discharge_eff
    
    safe_max_charge = min(max_charge, max_charge_by_soc)
    safe_max_discharge = min(max_discharge, max_discharge_by_soc)
    
    if battery_action_kw > 0:
        safe_battery_kw = min(battery_action_kw, safe_max_charge)
    else:
        safe_battery_kw = -min(abs(battery_action_kw), safe_max_discharge)
        
    # 3. Enforce diesel generator constraints
    # If turned off, generation is 0. If turned on, must be between min_load and capacity.
    safe_diesel_status = diesel_on_off
    if safe_diesel_status:
        safe_diesel_kw = np.clip(diesel_action_kw, d_min_load, d_capacity)
    else:
        safe_diesel_kw = 0.0
        
    # 4. Critical load protection check:
    # Essential load MUST be met. If renewable + battery discharge + diesel gen is not enough,
    # we must override diesel status to ON and run it at sufficient power.
    net_renewable = solar_gen_kw + wind_gen_kw
    
    # Battery maximum possible discharge power in this instant
    avail_battery_discharge = safe_max_discharge
    
    max_instant_power = net_renewable + avail_battery_discharge + safe_diesel_kw
    
    # If we are undershooting critical load, we MUST force diesel generator ON
    if max_instant_power < critical_load_kw:
        # Force diesel ON
        safe_diesel_status = True
        needed_from_diesel = critical_load_kw - net_renewable - avail_battery_discharge
        safe_diesel_kw = np.clip(needed_from_diesel, d_min_load, d_capacity)
        
    # Recalculate power balance to find actual unmet load and curtailment
    # generation = solar + wind + diesel + battery_discharge (if net power negative)
    # demand = load + battery_charge (if net power positive)
    net_used_renewables = min(current_load_kw, net_renewable)
    deficit = current_load_kw - net_renewable
    
    actual_battery_charge = 0.0
    actual_battery_discharge = 0.0
    
    if deficit <= 0:
        # Renewable surplus
        surplus = -deficit
        # Charging battery
        actual_battery_charge = min(surplus, safe_max_charge)
        curtailment = surplus - actual_battery_charge
        actual_diesel_kw = 0.0
        actual_diesel_status = False
        unmet_load = 0.0
    else:
        # Renewable deficit
        # Discharge battery
        actual_battery_discharge = min(deficit, safe_max_discharge)
        rem_deficit = deficit - actual_battery_discharge
        
        if rem_deficit > 0:
            # Check if diesel is running
            if safe_diesel_status:
                actual_diesel_kw = min(rem_deficit, safe_diesel_kw)
                # If we operate below min load, clip it to min load and curtail excess
                if actual_diesel_kw < d_min_load:
                    actual_diesel_kw = d_min_load
                unmet_load = max(0.0, rem_deficit - actual_diesel_kw)
                curtailment = max(0.0, actual_diesel_kw - rem_deficit) # Generator excess curtailed
            else:
                # Force diesel on if unmet load threatens critical load
                actual_diesel_status = True
                actual_diesel_kw = np.clip(rem_deficit, d_min_load, d_capacity)
                unmet_load = max(0.0, rem_deficit - actual_diesel_kw)
                curtailment = max(0.0, actual_diesel_kw - rem_deficit)
                safe_diesel_status = True
        else:
            actual_diesel_kw = 0.0
            actual_diesel_status = False
            unmet_load = 0.0
            curtailment = 0.0
            
    # Set final battery net power
    # If charging: actual_battery_charge. If discharging: -actual_battery_discharge
    net_battery_power = actual_battery_charge - actual_battery_discharge
    
    return {
        "battery_power_kw": net_battery_power,
        "battery_charge_kw": actual_battery_charge,
        "battery_discharge_kw": actual_battery_discharge,
        "diesel_generation_kw": actual_diesel_kw,
        "diesel_status": int(safe_diesel_status),
        "unmet_load_kw": unmet_load,
        "curtailment_kw": curtailment
    }
