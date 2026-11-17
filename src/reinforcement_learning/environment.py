"""
PolarGrid AI — Gymnasium Energy Dispatch Environment
Reinforcement learning environment for training PPO and SAC controllers.
Ensures hard safety bounds are enforced using safety_constraints.py.
Features normalized observations, random episode start sampling across seasons,
and stabilized reward shaping for optimal convergence.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from src.dispatch.safety_constraints import enforce_safety_constraints

class PolarGridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df_weather: pd.DataFrame,
        config_params: dict = None,
        episode_length: Optional[int] = 168, # 1 week episodes during training
        random_start: bool = True
    ):
        super().__init__()
        
        self.df = df_weather.copy().reset_index(drop=True)
        self.n_steps_total = len(self.df)
        self.episode_length = episode_length if episode_length is not None else self.n_steps_total
        self.random_start = random_start
        
        # Load configs if not provided
        if config_params is None:
            root = Path(__file__).resolve().parent.parent.parent
            with open(root / "configs" / "battery_config.yaml", "r") as f:
                b_cfg = yaml.safe_load(f)["battery"]
            with open(root / "configs" / "diesel_config.yaml", "r") as f:
                d_cfg = yaml.safe_load(f)["diesel"]
            self.params = {"battery": b_cfg, "diesel": d_cfg}
        else:
            self.params = config_params
            
        # Normalized continuous observation space [0, 1] or [-1, 1]:
        # 1. load_kw / 150.0
        # 2. solar_gen_kw / 100.0
        # 3. wind_gen_kw / 150.0
        # 4. battery_soc (0 to 1)
        # 5. diesel_status (0 or 1)
        # 6. (temperature + 60.0) / 80.0
        # 7. hour / 23.0
        # 8. season / 3.0
        # 9. polar_night (0 or 1)
        # 10. net_deficit / 150.0
        low_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        high_obs = np.array([1.5, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        # Action space: Box[-1, 1] containing:
        # 1. Battery power target (positive: charge, negative: discharge)
        # 2. Diesel power target
        # 3. Diesel ON/OFF command (> 0 means ON)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Track current state
        self.start_idx = 0
        self.current_step = 0
        self.soc = self.params["battery"]["soc_initial"]
        self.diesel_running = 0
        
    def _get_obs(self) -> np.ndarray:
        idx = min(self.start_idx + self.current_step, self.n_steps_total - 1)
        row = self.df.iloc[idx]
        
        load = float(row["load_kw"])
        solar = float(row["solar_generation_kw"])
        wind = float(row["wind_generation_kw"])
        deficit = (load - (solar + wind)) / 150.0
        
        return np.array([
            np.clip(load / 150.0, 0.0, 1.5),
            np.clip(solar / 100.0, 0.0, 1.5),
            np.clip(wind / 150.0, 0.0, 2.0),
            float(self.soc),
            float(self.diesel_running),
            np.clip((float(row["temperature"]) + 60.0) / 80.0, 0.0, 1.0),
            float(row["hour"]) / 23.0,
            float(row.get("season", 0.0)) / 3.0,
            float(row.get("polar_night", 0.0)),
            np.clip(deficit, -1.0, 1.5)
        ], dtype=np.float32)
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if self.random_start and self.n_steps_total > self.episode_length:
            self.start_idx = np.random.randint(0, self.n_steps_total - self.episode_length)
        else:
            self.start_idx = 0
            
        self.current_step = 0
        self.soc = self.params["battery"]["soc_initial"]
        self.diesel_running = 0
        return self._get_obs(), {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map continuous actions to physical quantities
        b_max = self.params["battery"]["max_charge_power_kw"]
        battery_target = float(action[0]) * b_max
        
        d_max = self.params["diesel"]["capacity_kw"]
        diesel_target = (float(action[1]) + 1.0) * 0.5 * d_max
        diesel_status_target = float(action[2]) > 0.0
        
        idx = min(self.start_idx + self.current_step, self.n_steps_total - 1)
        row = self.df.iloc[idx]
        load = float(row["load_kw"])
        crit_load = float(row["critical_load_kw"])
        solar = float(row["solar_generation_kw"])
        wind = float(row["wind_generation_kw"])
        temp = float(row["temperature"])
        
        # Enforce safe physical boundaries
        dispatch = enforce_safety_constraints(
            battery_action_kw=battery_target,
            diesel_action_kw=diesel_target,
            diesel_on_off=diesel_status_target,
            current_soc=self.soc,
            current_load_kw=load,
            critical_load_kw=crit_load,
            solar_gen_kw=solar,
            wind_gen_kw=wind,
            temp_c=temp,
            params=self.params
        )
        
        startup_event = (dispatch["diesel_status"] == 1) and (self.diesel_running == 0)
        self.diesel_running = dispatch["diesel_status"]
        
        # Step battery SOC
        capacity = self.params["battery"]["capacity_kwh"]
        cold_threshold = self.params["battery"]["cold_derating_threshold_c"]
        if temp < cold_threshold:
            loss = (cold_threshold - temp) * self.params["battery"]["cold_derating_factor"]
            capacity = capacity * max(0.5, 1.0 - loss)
            
        net_battery = dispatch["battery_power_kw"]
        if net_battery > 0:
            self.soc += (net_battery * self.params["battery"]["charge_efficiency"]) / capacity
        else:
            self.soc += net_battery / (self.params["battery"]["discharge_efficiency"] * capacity)
        self.soc = np.clip(self.soc, self.params["battery"]["soc_min"], self.params["battery"]["soc_max"])
        
        # Fuel consumption calculation
        fuel_intercept = self.params["diesel"]["fuel_intercept_lph"]
        fuel_slope = self.params["diesel"]["fuel_slope_lph_per_kw"]
        fuel_cost = self.params["diesel"]["fuel_cost_per_liter"]
        
        fuel_used = 0.0
        if self.diesel_running == 1:
            fuel_used = fuel_intercept + fuel_slope * dispatch["diesel_generation_kw"]
            
        fuel_cost_val = fuel_used * fuel_cost
        startup_cost = self.params["diesel"]["startup_cost_usd"] if startup_event else 0.0
        maint_cost = self.params["diesel"]["maintenance_cost_per_hour"] if self.diesel_running == 1 else 0.0
        deg_cost = abs(net_battery) * self.params["battery"]["degradation_cost_per_kwh"]
        unmet_penalty = dispatch["unmet_load_kw"] * 50.0
        
        # Baseline diesel-only cost for normalization
        baseline_fuel = fuel_intercept + fuel_slope * load
        baseline_cost = baseline_fuel * fuel_cost + self.params["diesel"]["maintenance_cost_per_hour"]
        
        # Actual step cost
        step_cost = fuel_cost_val + startup_cost + maint_cost + deg_cost + unmet_penalty
        
        # Normalized reward: positive for saving cost vs diesel-only baseline, scaled cleanly
        # Reward is proportional to % savings achieved
        reward = (baseline_cost - step_cost) / max(1.0, baseline_cost)
        
        # Small penalty if battery SOC is dangerously close to empty without need
        if self.soc < 0.15:
            reward -= 0.1
        elif 0.30 <= self.soc <= 0.80:
            reward += 0.05  # Slight healthy SOC buffer bonus
            
        self.current_step += 1
        terminated = self.current_step >= self.episode_length or (self.start_idx + self.current_step) >= (self.n_steps_total - 1)
        truncated = False
        
        next_obs = self._get_obs() if not terminated else np.zeros_like(self._get_obs())
        
        info = {
            "battery_power_kw": dispatch["battery_power_kw"],
            "diesel_generation_kw": dispatch["diesel_generation_kw"],
            "diesel_status": dispatch["diesel_status"],
            "unmet_load_kw": dispatch["unmet_load_kw"],
            "curtailment_kw": dispatch["curtailment_kw"],
            "fuel_consumed_litres": fuel_used,
            "cost_usd": step_cost
        }
        
        return next_obs, float(reward), terminated, truncated, info
