"""
PolarGrid AI — Station Load Simulation Model
Simulates electrical and thermal heating demand of the research station,
incorporating baseline, critical systems, heating demand, and research schedules.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Tuple

logger = logging.getLogger(__name__)

def simulate_station_load(
    ambient_temp_c: Union[float, np.ndarray],
    hour_utc: Union[int, np.ndarray],
    base_load_kw: float = 35.0,
    peak_load_kw: float = 120.0,
    heating_coefficient: float = 0.8, # kW increase per degree below -10°C
    heating_threshold_c: float = -10.0,
    research_load_peak_kw: float = 25.0,
    research_hours: list = None,
    critical_load_fraction: float = 0.40,
    noise_std: float = 3.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Simulate total station load (load_kw) and critical load (critical_load_kw).
    Total load = base_load + heating_load + research_load + random_noise
    - Heating load: heating_coefficient * max(0, heating_threshold_c - ambient_temp_c)
    - Research load: active during research_hours (default: 8 to 17 UTC)
    - Critical load: fraction of total load that is critical (comms, life support)
    """
    if research_hours is None:
        research_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        
    # Temperature heating demand
    heating_demand = np.maximum(0.0, heating_threshold_c - ambient_temp_c) * heating_coefficient
    
    # Research equipment load
    if isinstance(hour_utc, np.ndarray):
        research_demand = np.isin(hour_utc, research_hours).astype(float) * research_load_peak_kw
        # Add random noise
        noise = np.random.normal(0, noise_std, size=len(ambient_temp_c))
    else:
        research_demand = research_load_peak_kw if hour_utc in research_hours else 0.0
        noise = np.random.normal(0, noise_std)
        
    total_load = base_load_kw + heating_demand + research_demand + noise
    
    # Clip to physical limits [base_load_kw, peak_load_kw]
    total_load = np.clip(total_load, base_load_kw, peak_load_kw)
    
    # Critical load is the essential fraction of base load + essential portion of heating
    # Life support heating and essential systems are prioritised
    critical_load = total_load * critical_load_fraction
    
    return total_load, critical_load
