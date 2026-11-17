"""
PolarGrid AI — MILP Optimization Constraints
Defines Pyomo constraints for power balance, battery dynamics, and generator limits.
"""

from pyomo.environ import *

def apply_system_constraints(model, params: dict):
    """
    Applies all physical and operational constraints to the Pyomo ConcreteModel.
    - params: dictionary containing battery, diesel, and station configuration values.
    """
    
    # 1. Renewable availability limits
    def solar_limit_rule(model, t):
        return model.solar_use[t] <= model.solar_gen_forecast[t]
    model.solar_limit = Constraint(model.T, rule=solar_limit_rule)
    
    def wind_limit_rule(model, t):
        return model.wind_use[t] <= model.wind_gen_forecast[t]
    model.wind_limit = Constraint(model.T, rule=wind_limit_rule)
    
    # 2. Generator capacity and minimum operating load limits
    # If diesel_on[t] == 1, then min_load <= diesel_gen[t] <= capacity
    # If diesel_on[t] == 0, then diesel_gen[t] == 0
    def diesel_min_rule(model, t):
        min_load = params["diesel"]["capacity_kw"] * params["diesel"]["min_load_fraction"]
        # Apply generator capacity multiplier if scenario modifies it
        cap_mult = model.gen_cap_mult[t]
        return model.diesel_gen[t] >= min_load * model.diesel_on[t] * cap_mult
    model.diesel_min = Constraint(model.T, rule=diesel_min_rule)
    
    def diesel_max_rule(model, t):
        max_cap = params["diesel"]["capacity_kw"]
        cap_mult = model.gen_cap_mult[t]
        return model.diesel_gen[t] <= max_cap * model.diesel_on[t] * cap_mult
    model.diesel_max = Constraint(model.T, rule=diesel_max_rule)
    
    # 3. Battery charging/discharging limits
    def battery_charge_limit_rule(model, t):
        return model.battery_charge[t] <= params["battery"]["max_charge_power_kw"]
    model.battery_charge_limit = Constraint(model.T, rule=battery_charge_limit_rule)
    
    def battery_discharge_limit_rule(model, t):
        return model.battery_discharge[t] <= params["battery"]["max_discharge_power_kw"]
    model.battery_discharge_limit = Constraint(model.T, rule=battery_discharge_limit_rule)
    
    # 4. Battery SOC dynamics:
    # SOC[t] = SOC[t-1] + (charge * eff_charge - discharge / eff_discharge) / capacity
    def battery_soc_dynamics_rule(model, t):
        eff_c = params["battery"]["charge_efficiency"]
        eff_d = params["battery"]["discharge_efficiency"]
        cap = model.effective_capacity[t]
        
        if t == 1:
            # Initial SOC boundary
            return model.battery_soc[t] == model.soc_initial + (model.battery_charge[t] * eff_c - model.battery_discharge[t] / eff_d) / cap
        else:
            return model.battery_soc[t] == model.battery_soc[t-1] + (model.battery_charge[t] * eff_c - model.battery_discharge[t] / eff_d) / cap
    model.battery_soc_dynamics = Constraint(model.T, rule=battery_soc_dynamics_rule)
    
    # 5. Power balance equation:
    # Solar_use + Wind_use + Battery_discharge + Diesel_gen + Unmet = Load + Battery_charge + Curtailment
    def power_balance_rule(model, t):
        generation = model.solar_use[t] + model.wind_use[t] + model.diesel_gen[t] + model.battery_discharge[t] + model.unmet_load[t]
        demand = model.load_forecast[t] + model.battery_charge[t]
        return generation == demand
    model.power_balance = Constraint(model.T, rule=power_balance_rule)
    
    # 6. Critical load reliability constraint:
    # Essential/critical load cannot be unmet. Unmet load is capped at the non-critical portion.
    def critical_load_protection_rule(model, t):
        return model.unmet_load[t] <= (model.load_forecast[t] - model.critical_load_forecast[t])
    model.critical_load_protection = Constraint(model.T, rule=critical_load_protection_rule)
    
    # 7. Generator startup tracking
    # startup[t] >= diesel_on[t] - diesel_on[t-1]
    def startup_tracking_rule(model, t):
        if t == 1:
            return model.diesel_startup[t] >= model.diesel_on[t] - model.diesel_initial_status
        else:
            return model.diesel_startup[t] >= model.diesel_on[t] - model.diesel_on[t-1]
    model.startup_tracking = Constraint(model.T, rule=startup_tracking_rule)
