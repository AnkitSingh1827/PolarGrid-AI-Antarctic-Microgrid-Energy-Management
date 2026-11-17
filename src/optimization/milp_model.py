"""
PolarGrid AI — MILP Optimization Model
Constructs and solves the MILP scheduling problem using Pyomo.
Includes a robust fallback heuristic solver if external MILP solvers (e.g. GLPK) are not installed.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from src.optimization.constraints import apply_system_constraints
from src.optimization.objective import apply_optimization_objective

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS = PROJECT_ROOT / "configs"

def build_pyomo_model(
    load_forecast: np.ndarray,
    critical_load_forecast: np.ndarray,
    solar_gen_forecast: np.ndarray,
    wind_gen_forecast: np.ndarray,
    params: dict,
    initial_soc: float = 0.50,
    initial_diesel_status: int = 0,
    capacity_derating: float = 1.0,
    generator_capacity_mult: np.ndarray = None
) -> ConcreteModel:
    """Builds the Pyomo ConcreteModel with variables, params, objective, and constraints."""
    
    n_steps = len(load_forecast)
    model = ConcreteModel()
    
    # Time indices (1-indexed for Pyomo)
    model.T = RangeSet(1, n_steps)
    
    # Store parameters in model for constraint referencing
    model.soc_initial = initial_soc
    model.diesel_initial_status = initial_diesel_status
    
    # Setup weather and capacity parameters
    model.load_forecast = Param(model.T, initialize={t+1: load_forecast[t] for t in range(n_steps)})
    model.critical_load_forecast = Param(model.T, initialize={t+1: critical_load_forecast[t] for t in range(n_steps)})
    model.solar_gen_forecast = Param(model.T, initialize={t+1: solar_gen_forecast[t] for t in range(n_steps)})
    model.wind_gen_forecast = Param(model.T, initialize={t+1: wind_gen_forecast[t] for t in range(n_steps)})
    
    # Cold derated battery capacity over time
    model.effective_capacity = Param(model.T, initialize={t+1: params["battery"]["capacity_kwh"] * capacity_derating for t in range(n_steps)})
    
    # Generator capacity multipliers
    if generator_capacity_mult is None:
        generator_capacity_mult = np.ones(n_steps)
    model.gen_cap_mult = Param(model.T, initialize={t+1: generator_capacity_mult[t] for t in range(n_steps)})
    
    # ----------------------------------------------------
    # Decision Variables
    # ----------------------------------------------------
    model.solar_use = Var(model.T, domain=NonNegativeReals)
    model.wind_use = Var(model.T, domain=NonNegativeReals)
    
    model.battery_charge = Var(model.T, domain=NonNegativeReals)
    model.battery_discharge = Var(model.T, domain=NonNegativeReals)
    model.battery_soc = Var(model.T, domain=NonNegativeReals, 
                            bounds=(params["battery"]["soc_min"], params["battery"]["soc_max"]))
    
    model.diesel_gen = Var(model.T, domain=NonNegativeReals)
    model.diesel_on = Var(model.T, domain=Binary) # 1 if generator is running
    model.diesel_startup = Var(model.T, domain=Binary) # 1 if generator starts up in this period
    
    model.unmet_load = Var(model.T, domain=NonNegativeReals)
    
    # ----------------------------------------------------
    # Constraints & Objective
    # ----------------------------------------------------
    apply_system_constraints(model, params)
    apply_optimization_objective(model, params)
    
    return model


def run_heuristic_dispatch(
    load_forecast: np.ndarray,
    critical_load_forecast: np.ndarray,
    solar_gen_forecast: np.ndarray,
    wind_gen_forecast: np.ndarray,
    params: dict,
    initial_soc: float = 0.50,
    initial_diesel_status: int = 0,
    capacity_derating: float = 1.0,
    generator_capacity_mult: np.ndarray = None
) -> pd.DataFrame:
    """
    Rule-Based / Heuristic fallback dispatcher in case Pyomo solver is unavailable.
    Guarantees that the project continues working end-to-end.
    """
    logger.info("Running Heuristic Dispatch Fallback Solver...")
    n_steps = len(load_forecast)
    
    soc = initial_soc
    diesel_status = initial_diesel_status
    
    # Output arrays
    solar_use = np.zeros(n_steps)
    wind_use = np.zeros(n_steps)
    bat_charge = np.zeros(n_steps)
    bat_discharge = np.zeros(n_steps)
    bat_soc = np.zeros(n_steps)
    d_gen = np.zeros(n_steps)
    d_on = np.zeros(n_steps)
    d_startup = np.zeros(n_steps)
    unmet = np.zeros(n_steps)
    
    bat_capacity = params["battery"]["capacity_kwh"] * capacity_derating
    soc_min = params["battery"]["soc_min"]
    soc_max = params["battery"]["soc_max"]
    charge_eff = params["battery"]["charge_efficiency"]
    discharge_eff = params["battery"]["discharge_efficiency"]
    max_charge = params["battery"]["max_charge_power_kw"]
    max_discharge = params["battery"]["max_discharge_power_kw"]
    
    d_capacity = params["diesel"]["capacity_kw"]
    d_min_load = d_capacity * params["diesel"]["min_load_fraction"]
    
    if generator_capacity_mult is None:
        generator_capacity_mult = np.ones(n_steps)
        
    for t in range(n_steps):
        p_load = load_forecast[t]
        p_solar = solar_gen_forecast[t]
        p_wind = wind_gen_forecast[t]
        p_renew = p_solar + p_wind
        
        gen_mult = generator_capacity_mult[t]
        
        # 1. Renewable utilization
        renew_use = min(p_load, p_renew)
        if p_renew > 0:
            solar_use[t] = p_solar * (renew_use / p_renew)
            wind_use[t] = p_wind * (renew_use / p_renew)
        else:
            solar_use[t] = 0.0
            wind_use[t] = 0.0
            
        net_deficit = p_load - renew_use
        
        if net_deficit <= 0:
            # Renewable surplus
            surplus = -net_deficit
            # Charge battery
            max_p_charge_by_soc = ((soc_max - soc) * bat_capacity) / charge_eff
            charge_power = min(surplus, max_charge, max_p_charge_by_soc)
            
            bat_charge[t] = charge_power
            bat_discharge[t] = 0.0
            soc += (charge_power * charge_eff) / bat_capacity
            
            d_gen[t] = 0.0
            d_on[t] = 0
            d_startup[t] = 0
            diesel_status = 0
        else:
            # Deficit
            # Try discharging battery
            max_p_discharge_by_soc = (soc - soc_min) * bat_capacity * discharge_eff
            discharge_power = min(net_deficit, max_discharge, max_p_discharge_by_soc)
            
            bat_discharge[t] = discharge_power
            bat_charge[t] = 0.0
            soc -= discharge_power / (discharge_eff * bat_capacity)
            
            rem_deficit = net_deficit - discharge_power
            
            if rem_deficit > 0:
                # Turn diesel generator on
                d_on[t] = 1
                d_startup[t] = 1 if diesel_status == 0 else 0
                diesel_status = 1
                
                # Constrain diesel generation
                active_cap = d_capacity * gen_mult
                active_min = d_min_load * gen_mult
                
                gen_power = np.clip(rem_deficit, active_min, active_cap)
                d_gen[t] = gen_power
                
                unmet[t] = max(0.0, rem_deficit - gen_power)
            else:
                d_gen[t] = 0.0
                d_on[t] = 0
                d_startup[t] = 0
                diesel_status = 0
                
        bat_soc[t] = np.clip(soc, soc_min, soc_max)
        
    return pd.DataFrame({
        "timestamp_idx": range(1, n_steps + 1),
        "solar_use": solar_use,
        "wind_use": wind_use,
        "battery_charge": bat_charge,
        "battery_discharge": bat_discharge,
        "battery_soc": bat_soc,
        "diesel_gen": d_gen,
        "diesel_on": d_on,
        "diesel_startup": d_startup,
        "unmet_load": unmet
    })


def solve_milp(
    load_forecast: np.ndarray,
    critical_load_forecast: np.ndarray,
    solar_gen_forecast: np.ndarray,
    wind_gen_forecast: np.ndarray,
    params: dict,
    initial_soc: float = 0.50,
    initial_diesel_status: int = 0,
    capacity_derating: float = 1.0,
    generator_capacity_mult: np.ndarray = None,
    solver_name: str = "glpk"
) -> Tuple[pd.DataFrame, bool]:
    """
    Solves the MILP ConcreteModel using Pyomo and returns dispatch schedule DataFrame.
    Automatically falls back to Heuristic Dispatch if the solver is missing.
    """
    # Create the ConcreteModel
    model = build_pyomo_model(
        load_forecast, critical_load_forecast, solar_gen_forecast, wind_gen_forecast,
        params, initial_soc, initial_diesel_status, capacity_derating, generator_capacity_mult
    )
    
    # Attempt solving
    solver = SolverFactory(solver_name)
    solver_available = False
    try:
        solver_available = solver.available()
    except Exception as e:
        logger.warning(f"Could not check solver availability: {e}")
    
    # Check if actually runnable (sometimes solver returns True but fails to run)
    is_solved = False
    if solver_available:
        try:
            logger.info(f"Attempting to solve MILP with '{solver_name}'...")
            results = solver.solve(model, tee=False)
            if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
                is_solved = True
                logger.info("MILP Optimization solved to optimality.")
            else:
                logger.warning(f"Solver returned termination: {results.solver.termination_condition}")
        except Exception as e:
            logger.warning(f"Solver run failed: {e}. Falling back to heuristic dispatch.")
            
    if is_solved:
        # Extract variables
        n_steps = len(load_forecast)
        dispatch_data = {
            "timestamp_idx": list(model.T),
            "solar_use": [value(model.solar_use[t]) for t in model.T],
            "wind_use": [value(model.wind_use[t]) for t in model.T],
            "battery_charge": [value(model.battery_charge[t]) for t in model.T],
            "battery_discharge": [value(model.battery_discharge[t]) for t in model.T],
            "battery_soc": [value(model.battery_soc[t]) for t in model.T],
            "diesel_gen": [value(model.diesel_gen[t]) for t in model.T],
            "diesel_on": [int(value(model.diesel_on[t])) for t in model.T],
            "diesel_startup": [int(value(model.diesel_startup[t])) for t in model.T],
            "unmet_load": [value(model.unmet_load[t]) for t in model.T],
        }
        return pd.DataFrame(dispatch_data), True
    else:
        # Heuristic dispatch fallback
        df_heuristic = run_heuristic_dispatch(
            load_forecast, critical_load_forecast, solar_gen_forecast, wind_gen_forecast,
            params, initial_soc, initial_diesel_status, capacity_derating, generator_capacity_mult
        )
        return df_heuristic, False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Simple test run with dummy data
    with open(CONFIGS / "station_config.yaml", "r") as f:
        station_cfg = yaml.safe_load(f)
    with open(CONFIGS / "battery_config.yaml", "r") as f:
        battery_cfg = yaml.safe_load(f)
    with open(CONFIGS / "diesel_config.yaml", "r") as f:
        diesel_cfg = yaml.safe_load(f)
        
    params = {
        "battery": battery_cfg["battery"],
        "diesel": diesel_cfg["diesel"]
    }
    
    # 24 steps
    np.random.seed(42)
    load = np.random.uniform(40.0, 80.0, 24)
    crit = load * 0.4
    solar = np.random.uniform(0.0, 20.0, 24)
    wind = np.random.uniform(0.0, 60.0, 24)
    
    df, solved = solve_milp(load, crit, solar, wind, params)
    print("Optimization Result (Solved?):", solved)
    print(df.head(5).to_string())
