"""
PolarGrid AI — Wind Turbine Power Simulation Model
Calculates wind generation based on wind speed, turbine capacity, and a power curve.
"""

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

def simulate_wind_power(
    wind_speed_ms: Union[float, np.ndarray],
    capacity_kw: float = 100.0,
    cut_in_speed: float = 3.0,
    rated_speed: float = 12.0,
    cut_out_speed: float = 25.0
) -> Union[float, np.ndarray]:
    """
    Simulate Wind Turbine Power Output using a standard physics-informed power curve.
    - wind_speed < cut_in_speed: P = 0
    - cut_in_speed <= wind_speed < rated_speed: cubic growth P = capacity * (w^3 - cut_in^3) / (rated^3 - cut_in^3)
    - rated_speed <= wind_speed < cut_out_speed: P = capacity
    - wind_speed >= cut_out_speed: P = 0 (furling/safety shutoff)
    """
    if isinstance(wind_speed_ms, np.ndarray):
        power = np.zeros_like(wind_speed_ms)
        
        # Region 1: Cubic ramp-up between cut-in and rated speed
        ramp_mask = (wind_speed_ms >= cut_in_speed) & (wind_speed_ms < rated_speed)
        w = wind_speed_ms[ramp_mask]
        power[ramp_mask] = capacity_kw * (w**3 - cut_in_speed**3) / (rated_speed**3 - cut_in_speed**3)
        
        # Region 2: Rated power output
        rated_mask = (wind_speed_ms >= rated_speed) & (wind_speed_ms < cut_out_speed)
        power[rated_mask] = capacity_kw
        
        # Region 3: Cut-out (storm protection) -> remains 0
        # Region 4: Below cut-in -> remains 0
        
        return np.clip(power, 0.0, capacity_kw)
    else:
        if wind_speed_ms < cut_in_speed or wind_speed_ms >= cut_out_speed:
            return 0.0
        elif wind_speed_ms >= rated_speed:
            return capacity_kw
        else:
            w = wind_speed_ms
            power = capacity_kw * (w**3 - cut_in_speed**3) / (rated_speed**3 - cut_in_speed**3)
            return float(np.clip(power, 0.0, capacity_kw))
            
def wind_speed_at_hub_height(
    wind_speed_10m: Union[float, np.ndarray],
    hub_height: float = 30.0,
    measurement_height: float = 10.0,
    roughness_length: float = 0.001
) -> Union[float, np.ndarray]:
    """
    Adjust wind speed to hub height using logarithmic wind profile law:
    v_hub = v_meas * ln(h_hub / z0) / ln(h_meas / z0)
    z0 is the roughness length (0.001m for ice/snow).
    """
    factor = np.log(hub_height / roughness_length) / np.log(measurement_height / roughness_length)
    return wind_speed_10m * factor
