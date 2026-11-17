"""
PolarGrid AI — Diesel Generator Simulation Model
Simulates power generation, fuel consumption curve, maintenance costs, and startup/shutdown logic.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DieselSimulation:
    def __init__(
        self,
        capacity_kw: float = 150.0,
        min_load_fraction: float = 0.30, # Cannot operate below 30% of capacity
        fuel_intercept_lph: float = 5.0, # L/h fuel at no load (when running)
        fuel_slope_lph_per_kw: float = 0.25, # L/h fuel per kW generated
        fuel_cost_per_liter: float = 2.50,
        startup_cost_usd: float = 15.0,
        maintenance_cost_per_hour: float = 3.0
    ):
        self.capacity_kw = capacity_kw
        self.min_load = capacity_kw * min_load_fraction
        self.fuel_intercept = fuel_intercept_lph
        self.fuel_slope = fuel_slope_lph_per_kw
        self.fuel_cost = fuel_cost_per_liter
        self.startup_cost = startup_cost_usd
        self.maint_cost_per_hour = maintenance_cost_per_hour
        
        # State tracking
        self.is_running = False
        self.runtime_hours = 0.0
        
    def step(self, power_request_kw: float, turn_on: bool, dt_hours: float = 1.0) -> Dict[str, Any]:
        """
        Step the generator by one timestep.
        - power_request_kw: power output requested.
        - turn_on: Boolean indicating if the generator is commanded ON.
        Returns dictionary containing generation, fuel usage, and costs.
        """
        was_running = self.is_running
        self.is_running = turn_on
        
        startup_event = self.is_running and not was_running
        
        if self.is_running:
            # Constrain power between minimum load and maximum capacity
            generation_kw = np.clip(power_request_kw, self.min_load, self.capacity_kw)
            # Fuel curve: fuel_lph = intercept + slope * power_kw
            fuel_lph = self.fuel_intercept + self.fuel_slope * generation_kw
            self.runtime_hours += dt_hours
        else:
            generation_kw = 0.0
            fuel_lph = 0.0
            self.runtime_hours = 0.0
            
        fuel_consumed_l = fuel_lph * dt_hours
        fuel_cost_usd = fuel_consumed_l * self.fuel_cost
        
        startup_cost_usd = self.startup_cost if startup_event else 0.0
        maint_cost_usd = self.maint_cost_per_hour * dt_hours if self.is_running else 0.0
        
        total_operating_cost_usd = fuel_cost_usd + startup_cost_usd + maint_cost_usd
        
        return {
            "diesel_generation_kw": generation_kw,
            "diesel_status": int(self.is_running),
            "fuel_consumption_lph": fuel_lph,
            "fuel_consumed_litres": fuel_consumed_l,
            "fuel_cost_usd": fuel_cost_usd,
            "startup_cost_usd": startup_cost_usd,
            "maintenance_cost_usd": maint_cost_usd,
            "operating_cost_usd": total_operating_cost_usd
        }
        
    def reset(self):
        self.is_running = False
        self.runtime_hours = 0.0
