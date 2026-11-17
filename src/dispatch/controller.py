"""
PolarGrid AI — Unified Dispatch Controller
Integrates MILP schedule, RL adaptive control, safety layer, and rule-based fallback.
Provides a single dispatch interface for the system.
"""

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.dispatch.safety_constraints import enforce_safety_constraints
from src.dispatch.fallback_controller import run_rule_based_dispatch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS = PROJECT_ROOT / "configs"
RL_MODELS_DIR = PROJECT_ROOT / "models" / "rl"


def load_system_configs() -> dict:
    """Load all system configuration parameters."""
    with open(CONFIGS / "battery_config.yaml", "r") as f:
        b_cfg = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml", "r") as f:
        d_cfg = yaml.safe_load(f)["diesel"]
    with open(CONFIGS / "station_config.yaml", "r") as f:
        s_cfg = yaml.safe_load(f)
    return {"battery": b_cfg, "diesel": d_cfg, "station": s_cfg}


class DispatchController:
    """
    Unified energy dispatch controller for polar research stations.
    
    Dispatch priority hierarchy:
    1. Renewable energy (solar + wind) → used first
    2. Battery storage → charge from surplus, discharge during deficit
    3. Diesel generator → backup power
    
    Controller modes:
    - 'rule_based': Priority-based dispatch
    - 'milp': MILP-optimized schedule
    - 'rl_ppo': PPO RL controller
    - 'rl_sac': SAC RL controller
    """
    
    _model_cache: Dict[str, Any] = {}

    def __init__(self, mode: str = "rule_based", params: dict = None):
        self.mode = mode
        self.params = params or load_system_configs()
        self.soc = self.params["battery"]["soc_initial"]
        self.diesel_status = 0
        self.rl_model = None
        self.milp_schedule = None
        self.milp_index = 0
        
        # Try to load RL model if requested
        norm_mode = mode if mode.startswith("rl_") else f"rl_{mode}"
        if norm_mode in ("rl_ppo", "rl_sac"):
            self._load_rl_model(norm_mode)
            
    def _load_rl_model(self, mode: str):
        """Load a trained RL model with in-memory caching."""
        if mode in self._model_cache:
            self.rl_model = self._model_cache[mode]
            self.mode = mode
            return

        try:
            if mode == "rl_ppo":
                from stable_baselines3 import PPO
                model_path = RL_MODELS_DIR / "ppo_polargrid.zip"
                if model_path.exists():
                    self.rl_model = PPO.load(model_path)
                    self._model_cache[mode] = self.rl_model
                    self.mode = mode
                    logger.info("Loaded and cached PPO model for dispatch.")
                else:
                    logger.warning(f"PPO model not found at {model_path}. Falling back to rule-based.")
                    self.mode = "rule_based"
            elif mode == "rl_sac":
                from stable_baselines3 import SAC
                model_path = RL_MODELS_DIR / "sac_polargrid.zip"
                if model_path.exists():
                    self.rl_model = SAC.load(model_path)
                    self._model_cache[mode] = self.rl_model
                    self.mode = mode
                    logger.info("Loaded and cached SAC model for dispatch.")
                else:
                    logger.warning(f"SAC model not found at {model_path}. Falling back to rule-based.")
                    self.mode = "rule_based"
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}. Falling back to rule-based.")
            self.mode = "rule_based"
            
    def set_milp_schedule(self, schedule_df: pd.DataFrame):
        """Set a pre-computed MILP dispatch schedule."""
        self.milp_schedule = schedule_df
        self.milp_index = 0
        logger.info(f"MILP schedule loaded with {len(schedule_df)} timesteps.")
        
    def reset(self):
        """Reset controller state."""
        self.soc = self.params["battery"]["soc_initial"]
        self.diesel_status = 0
        self.milp_index = 0
        
    def dispatch_step(
        self,
        load_kw: float,
        critical_load_kw: float,
        solar_gen_kw: float,
        wind_gen_kw: float,
        temperature: float,
        hour: float = 0.0,
        season: float = 0.0,
        polar_night: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute one dispatch step.
        Returns a dispatch dictionary with all power flows and status.
        """
        if self.mode == "milp" and self.milp_schedule is not None:
            dispatch = self._dispatch_milp(load_kw, critical_load_kw, solar_gen_kw,
                                           wind_gen_kw, temperature)
        elif self.mode in ("rl_ppo", "rl_sac") and self.rl_model is not None:
            dispatch = self._dispatch_rl(load_kw, critical_load_kw, solar_gen_kw,
                                         wind_gen_kw, temperature, hour, season, polar_night)
        else:
            dispatch = self._dispatch_rule_based(load_kw, critical_load_kw, solar_gen_kw,
                                                  wind_gen_kw, temperature)
            
        # Update internal state
        self._update_soc(dispatch["battery_power_kw"], temperature)
        self.diesel_status = dispatch["diesel_status"]
        
        # Add metadata
        dispatch["controller_mode"] = self.mode
        dispatch["battery_soc"] = self.soc
        
        return dispatch
        
    def _dispatch_rule_based(self, load_kw, critical_load_kw, solar_gen_kw,
                              wind_gen_kw, temperature) -> Dict[str, Any]:
        """Rule-based dispatch: Renewable → Battery → Diesel."""
        return run_rule_based_dispatch(
            load_kw=load_kw,
            critical_load_kw=critical_load_kw,
            solar_gen_kw=solar_gen_kw,
            wind_gen_kw=wind_gen_kw,
            current_soc=self.soc,
            temp_c=temperature,
            params=self.params,
            diesel_status=self.diesel_status,
        )
        
    def _dispatch_milp(self, load_kw, critical_load_kw, solar_gen_kw,
                        wind_gen_kw, temperature) -> Dict[str, Any]:
        """Use pre-computed MILP schedule."""
        if self.milp_index >= len(self.milp_schedule):
            logger.warning("MILP schedule exhausted. Falling back to rule-based dispatch.")
            return self._dispatch_rule_based(load_kw, critical_load_kw, solar_gen_kw,
                                              wind_gen_kw, temperature)
            
        row = self.milp_schedule.iloc[self.milp_index]
        self.milp_index += 1
        
        battery_target = row.get("battery_charge", 0.0) - row.get("battery_discharge", 0.0)
        diesel_target = row.get("diesel_gen", 0.0)
        diesel_on = bool(row.get("diesel_on", 0))
        
        return enforce_safety_constraints(
            battery_action_kw=battery_target,
            diesel_action_kw=diesel_target,
            diesel_on_off=diesel_on,
            current_soc=self.soc,
            current_load_kw=load_kw,
            critical_load_kw=critical_load_kw,
            solar_gen_kw=solar_gen_kw,
            wind_gen_kw=wind_gen_kw,
            temp_c=temperature,
            params=self.params,
        )
        
    def _dispatch_rl(self, load_kw, critical_load_kw, solar_gen_kw,
                      wind_gen_kw, temperature, hour, season, polar_night) -> Dict[str, Any]:
        """RL model dispatch with generator modulation and safety layer."""
        deficit = (load_kw - (solar_gen_kw + wind_gen_kw)) / 150.0
        
        obs = np.array([
            np.clip(load_kw / 150.0, 0.0, 1.5),
            np.clip(solar_gen_kw / 100.0, 0.0, 1.5),
            np.clip(wind_gen_kw / 150.0, 0.0, 2.0),
            float(self.soc),
            float(self.diesel_status),
            np.clip((temperature + 60.0) / 80.0, 0.0, 1.0),
            float(hour) / 23.0,
            float(season) / 3.0,
            float(polar_night),
            np.clip(deficit, -1.0, 1.5)
        ], dtype=np.float32)
        
        action, _ = self.rl_model.predict(obs, deterministic=True)
        
        b_max = self.params["battery"]["max_charge_power_kw"]
        d_max = self.params["diesel"]["capacity_kw"]
        
        net_renewable = solar_gen_kw + wind_gen_kw
        if net_renewable >= load_kw:
            battery_target = net_renewable - load_kw
            diesel_target = 0.0
            diesel_on_off = False
        else:
            raw_deficit = load_kw - net_renewable
            avail_battery = (self.soc - self.params["battery"]["soc_min"]) * self.params["battery"]["capacity_kwh"] * self.params["battery"]["discharge_efficiency"]
            
            rl_wants_diesel = float(action[2]) > 0.0
            need_diesel = (avail_battery < raw_deficit) or (self.soc < 0.25) or (rl_wants_diesel and self.soc < 0.50)
            
            if need_diesel:
                diesel_on_off = True
                mod_factor = 0.6 + 0.3 * (float(action[1]) + 1.0) * 0.5
                diesel_target = np.clip(raw_deficit * mod_factor, self.params["diesel"]["capacity_kw"] * 0.3, d_max)
                battery_target = float(action[0]) * b_max
            else:
                diesel_on_off = False
                diesel_target = 0.0
                battery_target = -raw_deficit
        
        return enforce_safety_constraints(
            battery_action_kw=battery_target,
            diesel_action_kw=diesel_target,
            diesel_on_off=diesel_on_off,
            current_soc=self.soc,
            current_load_kw=load_kw,
            critical_load_kw=critical_load_kw,
            solar_gen_kw=solar_gen_kw,
            wind_gen_kw=wind_gen_kw,
            temp_c=temperature,
            params=self.params,
        )
        
    def _update_soc(self, net_battery_kw: float, temperature: float):
        """Update internal SOC state after dispatch."""
        capacity = self.params["battery"]["capacity_kwh"]
        cold_threshold = self.params["battery"]["cold_derating_threshold_c"]
        cold_factor = self.params["battery"]["cold_derating_factor"]
        
        if temperature < cold_threshold:
            loss = (cold_threshold - temperature) * cold_factor
            capacity = capacity * max(0.5, 1.0 - loss)
            
        charge_eff = self.params["battery"]["charge_efficiency"]
        discharge_eff = self.params["battery"]["discharge_efficiency"]
        
        if net_battery_kw > 0:
            self.soc += (net_battery_kw * charge_eff) / capacity
        else:
            self.soc += net_battery_kw / (discharge_eff * capacity)
            
        self.soc = np.clip(
            self.soc,
            self.params["battery"]["soc_min"],
            self.params["battery"]["soc_max"],
        )


def run_dispatch_simulation(
    df: pd.DataFrame,
    mode: str = "rule_based",
    params: dict = None,
) -> pd.DataFrame:
    """
    Run full dispatch simulation over a DataFrame.
    Returns DataFrame with dispatch results for every timestep.
    """
    controller = DispatchController(mode=mode, params=params)
    
    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        dispatch = controller.dispatch_step(
            load_kw=row["load_kw"],
            critical_load_kw=row["critical_load_kw"],
            solar_gen_kw=row["solar_generation_kw"],
            wind_gen_kw=row["wind_generation_kw"],
            temperature=row["temperature"],
            hour=float(row.get("hour", 0)),
            season=float(row.get("season", 0)),
            polar_night=float(row.get("polar_night", 0)),
        )
        dispatch["timestamp"] = row.get("timestamp", i)
        dispatch["load_kw"] = row["load_kw"]
        dispatch["solar_available_kw"] = row["solar_generation_kw"]
        dispatch["wind_available_kw"] = row["wind_generation_kw"]
        dispatch["temperature"] = row["temperature"]
        records.append(dispatch)
        
    return pd.DataFrame(records)
