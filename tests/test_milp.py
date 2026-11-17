"""
PolarGrid AI — MILP Optimization Tests
Tests for Pyomo MILP model construction, constraints, and heuristic fallback.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIGS = PROJECT_ROOT / "configs"


def load_test_params():
    with open(CONFIGS / "battery_config.yaml") as f:
        b = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml") as f:
        d = yaml.safe_load(f)["diesel"]
    return {"battery": b, "diesel": d}


def make_test_data(n=24):
    """Generate synthetic test data for MILP tests."""
    np.random.seed(42)
    load = np.random.uniform(40.0, 80.0, n)
    critical = load * 0.4
    solar = np.random.uniform(0.0, 20.0, n)
    wind = np.random.uniform(0.0, 60.0, n)
    return load, critical, solar, wind


class TestMILPModel:
    def test_model_builds(self):
        """Pyomo model should build without errors."""
        from src.optimization.milp_model import build_pyomo_model
        params = load_test_params()
        load, critical, solar, wind = make_test_data(24)
        
        model = build_pyomo_model(load, critical, solar, wind, params)
        assert model is not None
        assert hasattr(model, "T")
        assert hasattr(model, "solar_use")
        assert hasattr(model, "diesel_gen")
        assert hasattr(model, "battery_soc")
        assert hasattr(model, "objective")

    def test_heuristic_dispatch_runs(self):
        """Heuristic fallback dispatcher must produce valid output."""
        from src.optimization.milp_model import run_heuristic_dispatch
        params = load_test_params()
        load, critical, solar, wind = make_test_data(24)
        
        result = run_heuristic_dispatch(load, critical, solar, wind, params)
        assert len(result) == 24, f"Expected 24 rows, got {len(result)}"
        assert "solar_use" in result.columns
        assert "battery_soc" in result.columns
        assert "diesel_gen" in result.columns
        assert "unmet_load" in result.columns

    def test_heuristic_soc_within_bounds(self):
        """Heuristic SOC must stay within configured bounds."""
        from src.optimization.milp_model import run_heuristic_dispatch
        params = load_test_params()
        load, critical, solar, wind = make_test_data(48)
        
        result = run_heuristic_dispatch(load, critical, solar, wind, params)
        soc_min = params["battery"]["soc_min"]
        soc_max = params["battery"]["soc_max"]
        
        assert result["battery_soc"].min() >= soc_min - 0.001, \
            f"SOC below minimum: {result['battery_soc'].min()}"
        assert result["battery_soc"].max() <= soc_max + 0.001, \
            f"SOC above maximum: {result['battery_soc'].max()}"

    def test_heuristic_non_negative_generation(self):
        """All generation values must be non-negative."""
        from src.optimization.milp_model import run_heuristic_dispatch
        params = load_test_params()
        load, critical, solar, wind = make_test_data(24)
        
        result = run_heuristic_dispatch(load, critical, solar, wind, params)
        assert (result["solar_use"] >= -0.001).all(), "Negative solar use"
        assert (result["wind_use"] >= -0.001).all(), "Negative wind use"
        assert (result["diesel_gen"] >= -0.001).all(), "Negative diesel gen"

    def test_heuristic_diesel_within_capacity(self):
        """Diesel generation must not exceed capacity."""
        from src.optimization.milp_model import run_heuristic_dispatch
        params = load_test_params()
        load, critical, solar, wind = make_test_data(24)
        
        result = run_heuristic_dispatch(load, critical, solar, wind, params)
        capacity = params["diesel"]["capacity_kw"]
        assert result["diesel_gen"].max() <= capacity + 0.001, \
            f"Diesel exceeds capacity: {result['diesel_gen'].max()}"

    def test_solve_milp_with_fallback(self):
        """solve_milp should succeed even if GLPK is not installed (heuristic fallback)."""
        from src.optimization.milp_model import solve_milp
        params = load_test_params()
        load, critical, solar, wind = make_test_data(24)
        
        result_df, solved = solve_milp(load, critical, solar, wind, params)
        assert result_df is not None
        assert len(result_df) == 24
        # solved can be True or False depending on GLPK availability
