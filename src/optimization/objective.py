"""
PolarGrid AI — MILP Optimization Objective
Defines the objective function to minimize diesel fuel, startup, maintenance,
battery degradation, and unmet load costs while maximizing renewable usage.
"""

from pyomo.environ import *

def apply_optimization_objective(model, params: dict):
    """
    Defines and applies the objective function to the Pyomo ConcreteModel.
    We minimize total operational costs:
    Total Cost = Fuel Cost + Startup Cost + Maintenance Cost + Battery Degradation Cost + Unmet Load Penalty
    """
    
    # Cost parameters
    fuel_cost = params["diesel"]["fuel_cost_per_liter"]
    fuel_intercept = params["diesel"]["fuel_intercept_lph"]
    fuel_slope = params["diesel"]["fuel_slope_lph_per_kw"]
    
    maint_cost = params["diesel"]["maintenance_cost_per_hour"]
    startup_cost = params["diesel"]["startup_cost_usd"]
    
    deg_cost = params["battery"]["degradation_cost_per_kwh"]
    
    # Heavy penalty for unmet non-critical load ($500/kWh)
    unmet_penalty = 500.0
    
    # Small penalty for curtailment to encourage renewable usage ($0.10/kWh)
    curtailment_penalty = 0.10
    
    def objective_rule(model):
        cost_fuel = sum(
            (fuel_intercept * model.diesel_on[t] + fuel_slope * model.diesel_gen[t]) * fuel_cost 
            for t in model.T
        )
        cost_maint = sum(model.diesel_on[t] * maint_cost for t in model.T)
        cost_startup = sum(model.diesel_startup[t] * startup_cost for t in model.T)
        cost_degradation = sum((model.battery_charge[t] + model.battery_discharge[t]) * deg_cost for t in model.T)
        cost_unmet = sum(model.unmet_load[t] * unmet_penalty for t in model.T)
        
        # Calculate curtailment for each step: Curtailment = Solar_forecast + Wind_forecast - Solar_use - Wind_use
        cost_curtail = sum(
            ((model.solar_gen_forecast[t] - model.solar_use[t]) + (model.wind_gen_forecast[t] - model.wind_use[t])) * curtailment_penalty
            for t in model.T
        )
        
        return cost_fuel + cost_maint + cost_startup + cost_degradation + cost_unmet + cost_curtail
        
    model.objective = Objective(rule=objective_rule, sense=minimize)
