"""
PolarGrid AI — Solar PV Power Simulation Model
Calculates solar generation based on solar radiation, panel capacity, efficiency, and temperature.
"""

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

def simulate_solar_power(
    solar_irradiance_wm2: Union[float, np.ndarray],
    ambient_temp_c: Union[float, np.ndarray],
    capacity_kw: float = 50.0,
    efficiency: float = 0.18,
    temp_coefficient: float = -0.004, # -0.4% per °C
    stc_temp_c: float = 25.0,
    inverter_efficiency: float = 0.96,
    polar_night: Union[int, float, np.ndarray] = 0
) -> Union[float, np.ndarray]:
    """
    Simulate Solar PV Power Output.
    P = Capacity (kW) * (Irradiance / 1000) * [1 + Temp_Coeff * (Cell_Temp - STC_Temp)] * Inverter_Eff
    Cell temperature is approximated as: Cell_Temp = Ambient_Temp + (Irradiance / 800) * (NOCT - 20)
    We will simplify cell temp to Ambient_Temp for polar regions or use the simple approximation.
    """
    # NOCT (Nominal Operating Cell Temperature) is typically 45°C
    noct = 45.0
    cell_temp = ambient_temp_c + (solar_irradiance_wm2 / 800.0) * (noct - 20.0)
    
    # Temperature correction factor
    temp_corr = 1.0 + temp_coefficient * (cell_temp - stc_temp_c)
    
    # Calculate output power in kW
    # solar_irradiance_wm2 / 1000.0 converts W/m² to standard suns (1000 W/m²)
    raw_power = capacity_kw * (solar_irradiance_wm2 / 1000.0) * temp_corr * inverter_efficiency
    
    # Ensure power is non-negative and zero during polar night or zero irradiance
    power = np.clip(raw_power, 0.0, capacity_kw)
    
    # Force zero if solar irradiance is negligible
    if isinstance(solar_irradiance_wm2, np.ndarray):
        power[solar_irradiance_wm2 < 10.0] = 0.0
        if isinstance(polar_night, np.ndarray):
            power[polar_night == 1] = 0.0
    else:
        if solar_irradiance_wm2 < 10.0 or polar_night == 1:
            power = 0.0
            
    return power
