"""
PolarGrid AI — Dispatch Controller Tests
Tests for the unified dispatch controller, fallback controller, and metrics.
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


def load_params():
    with open(CONFIGS / "battery_config.yaml") as f:
        b = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml") as f:
        d = yaml.safe_load(f)["diesel"]
    return {"battery": b, "diesel": d}


class TestFallbackController:
    def test_renewable_surplus_charges_battery(self):
        """When renewables exceed load, battery should charge."""
        from src.dispatch.fallback_controller import run_rule_based_dispatch
        params = load_params()
        
        result = run_rule_based_dispatch(
            load_kw=30.0,
            critical_load_kw=12.0,
            solar_gen_kw=40.0,
            wind_gen_kw=20.0,
            current_soc=0.5,
            temp_c=-10.0,
            params=params,
        )
        
        assert result["battery_charge_kw"] > 0 or result["battery_power_kw"] > 0, \
            "Battery should charge when renewables exceed load"
        assert result["diesel_generation_kw"] == 0.0, "Diesel should be off"
        assert result["unmet_load_kw"] == 0.0, "No unmet load expected"

    def test_deficit_discharges_battery(self):
        """When renewables insufficient, battery should discharge."""
        from src.dispatch.fallback_controller import run_rule_based_dispatch
        params = load_params()
        
        result = run_rule_based_dispatch(
            load_kw=80.0,
            critical_load_kw=32.0,
            solar_gen_kw=5.0,
            wind_gen_kw=5.0,
            current_soc=0.8,
            temp_c=-10.0,
            params=params,
        )
        
        assert result["battery_discharge_kw"] > 0, "Battery should discharge during deficit"

    def test_empty_battery_triggers_diesel(self):
        """When battery is near empty and renewables insufficient, diesel must start."""
        from src.dispatch.fallback_controller import run_rule_based_dispatch
        params = load_params()
        
        result = run_rule_based_dispatch(
            load_kw=80.0,
            critical_load_kw=32.0,
            solar_gen_kw=0.0,
            wind_gen_kw=0.0,
            current_soc=0.11,  # near SOC minimum
            temp_c=-10.0,
            params=params,
        )
        
        assert result["diesel_status"] == 1 or result["diesel_generation_kw"] > 0, \
            "Diesel should start when battery is near empty and no renewables"


class TestUnifiedController:
    def test_controller_creation(self):
        """DispatchController should instantiate in rule_based mode."""
        from src.dispatch.controller import DispatchController
        ctrl = DispatchController(mode="rule_based")
        assert ctrl.mode == "rule_based"
        assert ctrl.soc == ctrl.params["battery"]["soc_initial"]

    def test_controller_dispatch_step(self):
        """dispatch_step should return valid dispatch dictionary."""
        from src.dispatch.controller import DispatchController
        ctrl = DispatchController(mode="rule_based")
        
        result = ctrl.dispatch_step(
            load_kw=60.0,
            critical_load_kw=24.0,
            solar_gen_kw=15.0,
            wind_gen_kw=20.0,
            temperature=-15.0,
        )
        
        assert "battery_power_kw" in result
        assert "diesel_generation_kw" in result
        assert "diesel_status" in result
        assert "unmet_load_kw" in result
        assert "controller_mode" in result

    def test_controller_reset(self):
        """Reset should restore initial state."""
        from src.dispatch.controller import DispatchController
        ctrl = DispatchController(mode="rule_based")
        
        ctrl.dispatch_step(60.0, 24.0, 15.0, 20.0, -15.0)
        ctrl.reset()
        
        assert ctrl.soc == ctrl.params["battery"]["soc_initial"]
        assert ctrl.diesel_status == 0


class TestMetrics:
    def test_fuel_savings_computation(self):
        """Fuel savings metrics should produce valid results."""
        from src.metrics.fuel_savings import compute_fuel_savings
        params = load_params()
        
        df = pd.DataFrame({
            "load_kw": [60, 70, 50],
            "diesel_generation_kw": [20, 30, 10],
            "diesel_status": [1, 1, 0],
        })
        
        result = compute_fuel_savings(df, params)
        assert result["total_fuel_litres"] >= 0
        assert result["fuel_savings_pct"] >= 0
        assert result["co2_avoided_kg"] >= 0

    def test_reliability_computation(self):
        """Reliability metrics should compute correctly."""
        from src.metrics.reliability import compute_reliability
        
        df = pd.DataFrame({
            "load_kw": [60, 70, 50],
            "unmet_load_kw": [0, 0, 0],
        })
        
        result = compute_reliability(df)
        assert result["reliability_pct"] == 100.0
        assert result["unmet_load_kwh"] == 0.0

    def test_renewable_utilization(self):
        """Renewable utilization should be valid percentage."""
        from src.metrics.renewable_utilization import compute_renewable_utilization
        
        df = pd.DataFrame({
            "load_kw": [60, 70, 50],
            "solar_generation_kw": [20, 25, 15],
            "wind_generation_kw": [30, 35, 25],
            "curtailment_kw": [5, 5, 5],
        })
        
        result = compute_renewable_utilization(df)
        assert 0 <= result["utilization_pct"] <= 100
        assert result["curtailment_kwh"] == 15.0
