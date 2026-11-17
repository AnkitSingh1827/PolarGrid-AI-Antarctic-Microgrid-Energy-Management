"""
PolarGrid AI — Battery Energy Storage System (BESS) Model
Simulates SOC dynamics, charge/discharge constraints, degradation, and temperature derating.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BatterySimulation:
    def __init__(
        self,
        capacity_kwh: float = 500.0,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        soc_initial: float = 0.50,
        charge_efficiency: float = 0.92,
        discharge_efficiency: float = 0.92,
        max_charge_power_kw: float = 100.0,
        max_discharge_power_kw: float = 100.0,
        degradation_cost_per_kwh: float = 0.02,
        cold_threshold_c: float = -20.0,
        cold_derating_factor: float = 0.005
    ):
        self.capacity_kwh = capacity_kwh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc = soc_initial
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_charge_power_kw = max_charge_power_kw
        self.max_discharge_power_kw = max_discharge_power_kw
        self.degradation_cost_per_kwh = degradation_cost_per_kwh
        self.cold_threshold_c = cold_threshold_c
        self.cold_derating_factor = cold_derating_factor
        
    def get_effective_capacity(self, temp_c: float) -> float:
        """Apply cold derating to capacity if temperature is below threshold."""
        if temp_c < self.cold_threshold_c:
            loss = (self.cold_threshold_c - temp_c) * self.cold_derating_factor
            return self.capacity_kwh * max(0.5, 1.0 - loss)
        return self.capacity_kwh
        
    def step(self, power_kw: float, temp_c: float, dt_hours: float = 1.0) -> Dict[str, float]:
        """
        Step the battery simulation by one timestep.
        - power_kw > 0: charging
        - power_kw < 0: discharging
        Returns dictionary of actual charged/discharged power and new SOC.
        """
        effective_capacity = self.get_effective_capacity(temp_c)
        
        # Check charge/discharge limits
        if power_kw > 0:
            # Limit power by max charge rate and headroom to SOC max
            max_p_by_soc = ((self.soc_max - self.soc) * effective_capacity) / (self.charge_efficiency * dt_hours)
            actual_power = min(power_kw, self.max_charge_power_kw, max_p_by_soc)
            actual_power = max(0.0, actual_power)
            
            # Update SOC
            self.soc += (actual_power * self.charge_efficiency * dt_hours) / effective_capacity
            charge_power = actual_power
            discharge_power = 0.0
        else:
            # Discharging (power_kw is negative)
            power_needed = abs(power_kw)
            # Limit power by max discharge rate and footroom to SOC min
            max_p_by_soc = ((self.soc - self.soc_min) * effective_capacity * self.discharge_efficiency) / dt_hours
            actual_power = min(power_needed, self.max_discharge_power_kw, max_p_by_soc)
            actual_power = max(0.0, actual_power)
            
            # Update SOC
            self.soc -= (actual_power * dt_hours) / (self.discharge_efficiency * effective_capacity)
            charge_power = 0.0
            discharge_power = actual_power
            
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)
        
        # Calculate cycle degradation cost (USD)
        throughput_kwh = (charge_power + discharge_power) * dt_hours
        deg_cost = throughput_kwh * self.degradation_cost_per_kwh
        
        return {
            "battery_soc": self.soc,
            "battery_charge_kw": charge_power,
            "battery_discharge_kw": discharge_power,
            "battery_net_power_kw": charge_power - discharge_power,
            "degradation_cost_usd": deg_cost,
            "effective_capacity_kwh": effective_capacity
        }
        
    def reset(self, soc_initial: float = 0.50):
        self.soc = soc_initial
