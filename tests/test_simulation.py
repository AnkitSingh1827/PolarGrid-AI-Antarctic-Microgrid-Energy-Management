"""
PolarGrid AI — Simulation Tests
Tests for solar, wind, load, battery, and diesel simulation modules.
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


def load_test_configs():
    """Load configs for testing."""
    with open(CONFIGS / "battery_config.yaml") as f:
        b = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml") as f:
        d = yaml.safe_load(f)["diesel"]
    with open(CONFIGS / "renewable_config.yaml") as f:
        r = yaml.safe_load(f)
    with open(CONFIGS / "station_config.yaml") as f:
        s = yaml.safe_load(f)
    return {"battery": b, "diesel": d, "renewable": r, "station": s}


class TestSolarSimulation:
    def test_solar_zero_during_polar_night(self):
        """Solar generation must be zero during polar night."""
        from src.simulation.solar import simulate_solar_power
        result = simulate_solar_power(solar_irradiance_wm2=0.0, ambient_temp_c=-20.0, capacity_kw=50.0)
        assert result == 0.0, f"Solar should be 0 during polar night, got {result}"

    def test_solar_zero_low_irradiance(self):
        """Solar generation must be zero when irradiance < 10 W/m²."""
        from src.simulation.solar import simulate_solar_power
        result = simulate_solar_power(solar_irradiance_wm2=5.0, ambient_temp_c=-10.0, capacity_kw=50.0)
        assert result == 0.0, f"Solar should be 0 at low irradiance, got {result}"

    def test_solar_non_negative(self):
        """Solar generation must never be negative."""
        from src.simulation.solar import simulate_solar_power
        for ghi in [0, 50, 200, 500, 1000]:
            for temp in [-40, -20, 0, 10]:
                result = simulate_solar_power(solar_irradiance_wm2=float(ghi), ambient_temp_c=float(temp), capacity_kw=50.0)
                assert result >= 0, f"Solar generation negative: {result} at GHI={ghi}, T={temp}"

    def test_solar_increases_with_ghi(self):
        """Higher GHI should produce more solar power."""
        from src.simulation.solar import simulate_solar_power
        low = simulate_solar_power(solar_irradiance_wm2=100.0, ambient_temp_c=0.0, capacity_kw=50.0)
        high = simulate_solar_power(solar_irradiance_wm2=500.0, ambient_temp_c=0.0, capacity_kw=50.0)
        assert high >= low, f"Higher GHI should produce more power: {low} vs {high}"

    def test_solar_array_input(self):
        """Solar model should handle numpy array inputs."""
        from src.simulation.solar import simulate_solar_power
        ghi = np.array([0, 100, 500, 0, 200])
        temp = np.array([-20, -10, 0, -30, -5])
        result = simulate_solar_power(solar_irradiance_wm2=ghi, ambient_temp_c=temp, capacity_kw=50.0)
        assert len(result) == 5
        assert result[0] == 0.0  # Zero irradiance
        assert (result >= 0).all()


class TestWindSimulation:
    def test_wind_zero_below_cutin(self):
        """Wind generation zero below cut-in speed."""
        from src.simulation.wind import simulate_wind_power
        result = simulate_wind_power(wind_speed_ms=2.0, capacity_kw=100.0)
        assert result == 0.0, f"Wind should be 0 below cut-in, got {result}"

    def test_wind_zero_above_cutout(self):
        """Wind generation zero above cut-out speed."""
        from src.simulation.wind import simulate_wind_power
        result = simulate_wind_power(wind_speed_ms=30.0, capacity_kw=100.0)
        assert result == 0.0, f"Wind should be 0 above cut-out, got {result}"

    def test_wind_non_negative(self):
        """Wind generation must never be negative."""
        from src.simulation.wind import simulate_wind_power
        for ws in [0, 3, 5, 10, 15, 20, 25, 30, 40]:
            result = simulate_wind_power(wind_speed_ms=float(ws), capacity_kw=100.0)
            assert result >= 0, f"Wind negative at {ws} m/s: {result}"

    def test_wind_at_rated(self):
        """Wind at rated speed should equal capacity."""
        from src.simulation.wind import simulate_wind_power
        result = simulate_wind_power(wind_speed_ms=12.0, capacity_kw=100.0)
        assert result == 100.0, f"Wind at rated speed should be capacity, got {result}"

    def test_wind_cubic_region(self):
        """Wind power should increase with speed in cubic region."""
        from src.simulation.wind import simulate_wind_power
        p_low = simulate_wind_power(wind_speed_ms=5.0, capacity_kw=100.0)
        p_high = simulate_wind_power(wind_speed_ms=10.0, capacity_kw=100.0)
        assert p_high > p_low, f"Wind should increase with speed: {p_low} vs {p_high}"

    def test_wind_array_input(self):
        """Wind model should handle numpy array inputs."""
        from src.simulation.wind import simulate_wind_power
        speeds = np.array([0, 2, 5, 12, 20, 26])
        result = simulate_wind_power(wind_speed_ms=speeds, capacity_kw=100.0)
        assert len(result) == 6
        assert result[0] == 0.0  # Below cut-in
        assert result[1] == 0.0  # Below cut-in
        assert result[3] == 100.0  # At rated
        assert result[5] == 0.0  # Above cut-out


class TestBatterySimulation:
    def test_soc_within_bounds(self):
        """Battery SOC must stay within configured bounds."""
        from src.simulation.battery import BatterySimulation
        cfgs = load_test_configs()
        battery = BatterySimulation(**{k: v for k, v in cfgs["battery"].items() 
                                       if k in BatterySimulation.__init__.__code__.co_varnames})

        for power in [-100, -50, 0, 50, 100, -200, 200]:
            result = battery.step(power_kw=float(power), temp_c=-10.0)
            soc = result["battery_soc"]
            assert cfgs["battery"]["soc_min"] <= soc <= cfgs["battery"]["soc_max"], \
                f"SOC out of bounds: {soc}"
        
    def test_soc_within_bounds_simple(self):
        """Battery SOC stays bounded over many steps."""
        from src.simulation.battery import BatterySimulation
        battery = BatterySimulation(capacity_kwh=500, soc_min=0.1, soc_max=0.9, soc_initial=0.5)
        
        np.random.seed(42)
        for _ in range(100):
            power = np.random.uniform(-150, 150)
            result = battery.step(power_kw=power, temp_c=-25.0)
            assert 0.1 <= result["battery_soc"] <= 0.9, f"SOC out of bounds: {result['battery_soc']}"

    def test_cold_derating(self):
        """Battery effective capacity should decrease in extreme cold."""
        from src.simulation.battery import BatterySimulation
        battery = BatterySimulation(capacity_kwh=500, cold_threshold_c=-20.0, cold_derating_factor=0.005)
        
        normal_cap = battery.get_effective_capacity(temp_c=0.0)
        cold_cap = battery.get_effective_capacity(temp_c=-40.0)
        
        assert cold_cap < normal_cap, f"Cold capacity should be less: {cold_cap} vs {normal_cap}"
        assert cold_cap >= 250.0, "Cold capacity should not drop below 50% minimum"


class TestDieselSimulation:
    def test_diesel_within_capacity(self):
        """Diesel generation must not exceed capacity."""
        from src.simulation.diesel import DieselSimulation
        gen = DieselSimulation(capacity_kw=150.0, min_load_fraction=0.3)
        
        result = gen.step(power_request_kw=200.0, turn_on=True)
        assert result["diesel_generation_kw"] <= 150.0, \
            f"Diesel exceeds capacity: {result['diesel_generation_kw']}"

    def test_diesel_non_negative(self):
        """Diesel generation must be non-negative."""
        from src.simulation.diesel import DieselSimulation
        gen = DieselSimulation(capacity_kw=150.0)
        
        result = gen.step(power_request_kw=-50.0, turn_on=True)
        assert result["diesel_generation_kw"] >= 0, f"Diesel negative: {result['diesel_generation_kw']}"

    def test_diesel_min_load_when_on(self):
        """When diesel is on, output must be at least min load."""
        from src.simulation.diesel import DieselSimulation
        gen = DieselSimulation(capacity_kw=150.0, min_load_fraction=0.3)
        
        result = gen.step(power_request_kw=10.0, turn_on=True)
        assert result["diesel_generation_kw"] >= 150 * 0.3, \
            f"Diesel below min load: {result['diesel_generation_kw']}"

    def test_diesel_zero_when_off(self):
        """When diesel is off, output must be zero."""
        from src.simulation.diesel import DieselSimulation
        gen = DieselSimulation(capacity_kw=150.0)
        
        result = gen.step(power_request_kw=100.0, turn_on=False)
        assert result["diesel_generation_kw"] == 0.0
        assert result["fuel_consumption_lph"] == 0.0

    def test_startup_cost(self):
        """Startup event should incur startup cost."""
        from src.simulation.diesel import DieselSimulation
        gen = DieselSimulation(capacity_kw=150.0, startup_cost_usd=15.0)
        
        result = gen.step(power_request_kw=100.0, turn_on=True)
        assert result["startup_cost_usd"] == 15.0, "First start should cost $15"
        
        result2 = gen.step(power_request_kw=100.0, turn_on=True)
        assert result2["startup_cost_usd"] == 0.0, "Continued running should not incur startup cost"


class TestDataIntegrity:
    def test_master_dataset_exists(self):
        """Master dataset must exist and have data."""
        path = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
        assert path.exists(), "master_dataset.csv not found"
        df = pd.read_csv(path, nrows=10)
        assert len(df) > 0, "master_dataset.csv is empty"

    def test_master_dataset_columns(self):
        """Master dataset must have required columns."""
        path = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
        df = pd.read_csv(path, nrows=5)
        required = ["timestamp", "temperature", "wind_speed", "load_kw",
                     "solar_generation_kw", "wind_generation_kw", "battery_soc"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_negative_generation(self):
        """All generation values must be non-negative."""
        path = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
        df = pd.read_csv(path, nrows=1000)
        assert (df["solar_generation_kw"] >= 0).all(), "Negative solar generation found"
        assert (df["wind_generation_kw"] >= 0).all(), "Negative wind generation found"
        assert (df["load_kw"] >= 0).all(), "Negative load found"

    def test_simulated_files_exist(self):
        """All simulated output files should exist with data."""
        sim_dir = PROJECT_ROOT / "data" / "simulated"
        for name in ["solar_simulation.csv", "wind_simulation.csv", "load_simulation.csv",
                      "battery_simulation.csv", "diesel_simulation.csv"]:
            path = sim_dir / name
            assert path.exists(), f"Missing simulated file: {name}"
            assert path.stat().st_size > 100, f"Simulated file too small: {name}"
