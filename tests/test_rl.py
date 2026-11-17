"""
PolarGrid AI — RL Controller Tests
Tests for the Gymnasium environment and RL agent interfaces.
"""

import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIGS = PROJECT_ROOT / "configs"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def make_test_df(n=100):
    """Create a small test DataFrame with required columns."""
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
        "temperature": np.random.uniform(-30, 0, n),
        "wind_speed": np.random.uniform(2, 20, n),
        "solar_radiation": np.random.uniform(0, 200, n),
        "load_kw": np.random.uniform(35, 90, n),
        "critical_load_kw": np.random.uniform(14, 36, n),
        "solar_generation_kw": np.random.uniform(0, 30, n),
        "wind_generation_kw": np.random.uniform(0, 100, n),
        "battery_soc": np.full(n, 0.5),
        "hour": np.tile(np.arange(24), n // 24 + 1)[:n],
        "season": np.zeros(n),
        "polar_night": np.zeros(n),
    })


class TestPolarGridEnv:
    def test_env_creation(self):
        """Environment should instantiate without errors."""
        from src.reinforcement_learning.environment import PolarGridEnv
        df = make_test_df()
        env = PolarGridEnv(df_weather=df)
        assert env is not None

    def test_env_reset(self):
        """Reset should return valid observation."""
        from src.reinforcement_learning.environment import PolarGridEnv
        df = make_test_df()
        env = PolarGridEnv(df_weather=df)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

    def test_env_step(self):
        """Step should return valid (obs, reward, terminated, truncated, info)."""
        from src.reinforcement_learning.environment import PolarGridEnv
        df = make_test_df()
        env = PolarGridEnv(df_weather=df)
        env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(info, dict)
        assert "battery_power_kw" in info
        assert "diesel_generation_kw" in info

    def test_env_episode_completes(self):
        """Running through all timesteps should terminate."""
        from src.reinforcement_learning.environment import PolarGridEnv
        df = make_test_df(50)
        env = PolarGridEnv(df_weather=df)
        obs, _ = env.reset()
        
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert steps == 49, f"Expected 49 steps, got {steps}"  # n_steps - 1

    def test_env_soc_stays_bounded(self):
        """SOC should stay within configured bounds throughout an episode."""
        from src.reinforcement_learning.environment import PolarGridEnv
        df = make_test_df(50)
        env = PolarGridEnv(df_weather=df)
        env.reset()
        
        soc_min = env.params["battery"]["soc_min"]
        soc_max = env.params["battery"]["soc_max"]
        
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            assert soc_min - 0.001 <= env.soc <= soc_max + 0.001, \
                f"SOC out of bounds: {env.soc}"


class TestSafetyConstraints:
    def test_critical_load_protection(self):
        """Safety constraints must not shed critical load."""
        from src.dispatch.safety_constraints import enforce_safety_constraints
        
        with open(CONFIGS / "battery_config.yaml") as f:
            b = yaml.safe_load(f)["battery"]
        with open(CONFIGS / "diesel_config.yaml") as f:
            d = yaml.safe_load(f)["diesel"]
        params = {"battery": b, "diesel": d}
        
        result = enforce_safety_constraints(
            battery_action_kw=0.0,
            diesel_action_kw=0.0,
            diesel_on_off=False,
            current_soc=0.5,
            current_load_kw=100.0,
            critical_load_kw=40.0,
            solar_gen_kw=10.0,
            wind_gen_kw=10.0,
            temp_c=-10.0,
            params=params,
        )
        
        # Total available = 10 solar + 10 wind + battery + diesel should cover critical 40kW
        total_supply = (10 + 10 +
                       result["battery_discharge_kw"] +
                       result["diesel_generation_kw"] -
                       result["battery_charge_kw"])
        unmet = result["unmet_load_kw"]
        
        # Unmet should not exceed non-critical portion (60 kW)
        assert unmet <= 60.0 + 0.1, f"Critical load shed! Unmet: {unmet}"
