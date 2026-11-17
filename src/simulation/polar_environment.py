"""
PolarGrid AI — Polar Environment Scenario Engine
Applies scenario modifications (e.g. Polar Night, Extreme Cold, Load Spike, Diesel Failure)
to raw weather/system conditions to evaluate the energy system's robustness.
"""

import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ScenarioEngine:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "scenarios.yaml"
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.scenarios = config.get("scenarios", {})
        logger.info(f"Loaded scenarios: {list(self.scenarios.keys())}")
        
    def apply_scenario(self, df_weather: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
        """
        Applies a scenario's weather and load modifications to the weather DataFrame.
        Returns a modified DataFrame.
        """
        if scenario_name not in self.scenarios:
            logger.warning(f"Scenario '{scenario_name}' not defined. Returning unmodified data.")
            return df_weather.copy()
            
        mods = self.scenarios[scenario_name].get("modifications", {})
        df = df_weather.copy()
        
        # Add scenario identifier
        df["scenario_id"] = scenario_name
        
        # Temperature modifications
        if "temperature_offset_c" in mods and mods["temperature_offset_c"] is not None:
            offset = mods["temperature_offset_c"]
            col = "temperature" if "temperature" in df.columns else "temperature_c"
            df[col] = df[col] + offset
            
        if "temperature_override_min_c" in mods and mods["temperature_override_min_c"] is not None:
            override_min = mods["temperature_override_min_c"]
            col = "temperature" if "temperature" in df.columns else "temperature_c"
            df[col] = np.minimum(df[col], override_min)
            
        # Solar radiation modifications
        if "solar_irradiance_multiplier" in mods and mods["solar_irradiance_multiplier"] is not None:
            mult = mods["solar_irradiance_multiplier"]
            col = "solar_radiation" if "solar_radiation" in df.columns else "solar_irradiance_wm2"
            df[col] = df[col] * mult
            
        if "cloud_cover_override" in mods and mods["cloud_cover_override"] is not None:
            df["cloud_cover"] = mods["cloud_cover_override"]
            
        # Wind speed modifications
        if "wind_speed_multiplier" in mods and mods["wind_speed_multiplier"] is not None:
            mult = mods["wind_speed_multiplier"]
            col = "wind_speed" if "wind_speed" in df.columns else "wind_speed_ms"
            df[col] = df[col] * mult
            
        # Load multipliers (to be used when load is calculated)
        df["load_multiplier"] = 1.0
        df["critical_load_offset_kw"] = 0.0
        
        if "load_increase_fraction" in mods and mods["load_increase_fraction"] is not None:
            df["load_multiplier"] = 1.0 + mods["load_increase_fraction"]
            
        if "critical_load_increase_kw" in mods and mods["critical_load_increase_kw"] is not None:
            # Check spike start hour if specified
            if "spike_start_hour" in mods:
                start_hour = mods["spike_start_hour"]
                # Apply only if hour of day >= start_hour
                df["critical_load_offset_kw"] = np.where(df["hour"] >= start_hour, mods["critical_load_increase_kw"], 0.0)
            else:
                df["critical_load_offset_kw"] = mods["critical_load_increase_kw"]
                
        # Generator failure capacity reductions
        df["generator_capacity_multiplier"] = 1.0
        if "diesel_capacity_multiplier" in mods and mods["diesel_capacity_multiplier"] is not None:
            if "failure_start_hour" in mods:
                start_hour = mods["failure_start_hour"]
                df["generator_capacity_multiplier"] = np.where(df["hour"] >= start_hour, mods["diesel_capacity_multiplier"], 1.0)
            else:
                df["generator_capacity_multiplier"] = mods["diesel_capacity_multiplier"]
                
        return df
