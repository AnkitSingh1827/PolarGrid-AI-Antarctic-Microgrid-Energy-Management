"""
PolarGrid AI — Physics-Informed Energy Simulation Runner
Runs the physics simulation for solar, wind, load, battery, and diesel generator,
generating simulated datasets and evaluation scenarios.
"""

import pandas as pd
import numpy as np
import yaml
import os
import logging
from pathlib import Path
from src.simulation.solar import simulate_solar_power
from src.simulation.wind import simulate_wind_power, wind_speed_at_hub_height
from src.simulation.load import simulate_station_load
from src.simulation.battery import BatterySimulation
from src.simulation.diesel import DieselSimulation
from src.simulation.polar_environment import ScenarioEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SIMULATED = PROJECT_ROOT / "data" / "simulated"
CONFIGS = PROJECT_ROOT / "configs"

def run_physics_simulation():
    logger.info("Starting Physics-Informed Energy Simulation...")
    
    # Load configurations
    with open(CONFIGS / "station_config.yaml", "r") as f:
        station_cfg = yaml.safe_load(f)
    with open(CONFIGS / "battery_config.yaml", "r") as f:
        battery_cfg = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml", "r") as f:
        diesel_cfg = yaml.safe_load(f)["diesel"]
    with open(CONFIGS / "renewable_config.yaml", "r") as f:
        renew_cfg = yaml.safe_load(f)
        
    solar_cfg = renew_cfg["solar"]
    wind_cfg = renew_cfg["wind"]
    
    # Load weather foundation from master dataset
    master_path = DATA_PROCESSED / "master_dataset.csv"
    if not master_path.exists():
        raise FileNotFoundError("master_dataset.csv not found. Please run the data pipeline first.")
        
    master_df = pd.read_csv(master_path)
    logger.info(f"Loaded master dataset: {master_df.shape} rows")
    
    # Standardize time
    master_df["timestamp"] = pd.to_datetime(master_df["timestamp"], utc=True)
    
    # 1. Simulate Renewables & Load
    logger.info("Simulating Solar PV generation...")
    solar_gen = simulate_solar_power(
        solar_irradiance_wm2=master_df["solar_radiation"].values,
        ambient_temp_c=master_df["temperature"].values,
        capacity_kw=solar_cfg["capacity_kw"],
        efficiency=solar_cfg["efficiency"],
        temp_coefficient=solar_cfg["temp_coefficient"],
        stc_temp_c=solar_cfg["stc_temperature_c"],
        inverter_efficiency=solar_cfg["inverter_efficiency"]
    )
    
    logger.info("Simulating Wind Turbine generation...")
    # Scale wind speed to hub height
    hub_wind_speed = wind_speed_at_hub_height(
        wind_speed_10m=master_df["wind_speed"].values,
        hub_height=wind_cfg["hub_height_m"],
        measurement_height=wind_cfg["measurement_height_m"],
        roughness_length=wind_cfg["roughness_length_m"]
    )
    # Simulate power for 2 wind turbines
    wind_gen = simulate_wind_power(
        wind_speed_ms=hub_wind_speed,
        capacity_kw=wind_cfg["capacity_kw"] * wind_cfg["num_turbines"],
        cut_in_speed=wind_cfg["cut_in_speed_ms"],
        rated_speed=wind_cfg["rated_speed_ms"],
        cut_out_speed=wind_cfg["cut_out_speed_ms"]
    )
    
    logger.info("Simulating Station electrical and heating Load...")
    total_load, critical_load = simulate_station_load(
        ambient_temp_c=master_df["temperature"].values,
        hour_utc=master_df["hour"].values,
        base_load_kw=station_cfg["load"]["base_load_kw"],
        peak_load_kw=station_cfg["load"]["peak_load_kw"],
        heating_coefficient=station_cfg["load"]["heating_coefficient"],
        heating_threshold_c=station_cfg["load"]["heating_threshold_c"],
        research_load_peak_kw=station_cfg["load"]["research_load_peak_kw"],
        research_hours=station_cfg["load"]["research_hours"],
        critical_load_fraction=station_cfg["load"]["critical_load_fraction"]
    )
    
    # Save base simulated components
    pd.DataFrame({
        "timestamp": master_df["timestamp"],
        "solar_generation_kw": solar_gen
    }).to_csv(DATA_SIMULATED / "solar_simulation.csv", index=False)
    
    pd.DataFrame({
        "timestamp": master_df["timestamp"],
        "wind_generation_kw": wind_gen
    }).to_csv(DATA_SIMULATED / "wind_simulation.csv", index=False)
    
    pd.DataFrame({
        "timestamp": master_df["timestamp"],
        "load_kw": total_load,
        "critical_load_kw": critical_load
    }).to_csv(DATA_SIMULATED / "load_simulation.csv", index=False)
    
    # 2. Run rule-based baseline simulation for battery and diesel generator
    # Priority dispatch logic:
    # 1. Use solar + wind to meet load.
    # 2. If surplus: charge battery. Remaining surplus is curtailed.
    # 3. If deficit: discharge battery.
    # 4. If battery SOC reaches minimum or discharge rate limit: start diesel generator.
    logger.info("Running hour-by-hour battery and diesel baseline simulation (Rule-Based)...")
    
    battery = BatterySimulation(
        capacity_kwh=battery_cfg["capacity_kwh"],
        soc_min=battery_cfg["soc_min"],
        soc_max=battery_cfg["soc_max"],
        soc_initial=battery_cfg["soc_initial"],
        charge_efficiency=battery_cfg["charge_efficiency"],
        discharge_efficiency=battery_cfg["discharge_efficiency"],
        max_charge_power_kw=battery_cfg["max_charge_power_kw"],
        max_discharge_power_kw=battery_cfg["max_discharge_power_kw"],
        degradation_cost_per_kwh=battery_cfg["degradation_cost_per_kwh"],
        cold_threshold_c=battery_cfg["cold_derating_threshold_c"],
        cold_derating_factor=battery_cfg["cold_derating_factor"]
    )
    
    diesel = DieselSimulation(
        capacity_kw=diesel_cfg["capacity_kw"] * diesel_cfg["num_generators"],
        min_load_fraction=diesel_cfg["min_load_fraction"],
        fuel_intercept_lph=diesel_cfg["fuel_intercept_lph"],
        fuel_slope_lph_per_kw=diesel_cfg["fuel_slope_lph_per_kw"],
        fuel_cost_per_liter=diesel_cfg["fuel_cost_per_liter"],
        startup_cost_usd=diesel_cfg["startup_cost_usd"],
        maintenance_cost_per_hour=diesel_cfg["maintenance_cost_per_hour"]
    )
    
    n_steps = len(master_df)
    battery_socs = np.zeros(n_steps)
    battery_charge = np.zeros(n_steps)
    battery_discharge = np.zeros(n_steps)
    diesel_gen = np.zeros(n_steps)
    diesel_status = np.zeros(n_steps)
    fuel_lph = np.zeros(n_steps)
    unmet_load = np.zeros(n_steps)
    curtailment = np.zeros(n_steps)
    
    for t in range(n_steps):
        net_renewable = solar_gen[t] + wind_gen[t]
        demand = total_load[t]
        crit_demand = critical_load[t]
        temp = master_df["temperature"].values[t]
        
        net_deficit = demand - net_renewable
        
        if net_deficit <= 0:
            # Renewable surplus
            surplus = -net_deficit
            # Charge battery
            bat_res = battery.step(surplus, temp)
            battery_socs[t] = bat_res["battery_soc"]
            battery_charge[t] = bat_res["battery_charge_kw"]
            battery_discharge[t] = 0.0
            
            # Curtailed power
            curtailment[t] = surplus - bat_res["battery_charge_kw"]
            
            # Diesel is off
            d_res = diesel.step(0.0, False)
            diesel_gen[t] = 0.0
            diesel_status[t] = 0
            fuel_lph[t] = 0.0
            unmet_load[t] = 0.0
        else:
            # Renewable deficit
            deficit = net_deficit
            # Try discharging battery
            bat_res = battery.step(-deficit, temp)
            battery_socs[t] = bat_res["battery_soc"]
            battery_charge[t] = 0.0
            battery_discharge[t] = bat_res["battery_discharge_kw"]
            
            remaining_deficit = deficit - bat_res["battery_discharge_kw"]
            
            if remaining_deficit > 0:
                # Need diesel generator backup
                d_res = diesel.step(remaining_deficit, True)
                diesel_gen[t] = d_res["diesel_generation_kw"]
                diesel_status[t] = d_res["diesel_status"]
                fuel_lph[t] = d_res["fuel_consumption_lph"]
                
                # Check for unmet load
                unmet = remaining_deficit - d_res["diesel_generation_kw"]
                unmet_load[t] = max(0.0, unmet)
            else:
                # Deficit fully met by battery
                d_res = diesel.step(0.0, False)
                diesel_gen[t] = 0.0
                diesel_status[t] = 0
                fuel_lph[t] = 0.0
                unmet_load[t] = 0.0
                
    # Save simulated battery and diesel datasets
    pd.DataFrame({
        "timestamp": master_df["timestamp"],
        "battery_soc": battery_socs,
        "battery_charge_kw": battery_charge,
        "battery_discharge_kw": battery_discharge,
        "battery_capacity_kwh": battery.capacity_kwh
    }).to_csv(DATA_SIMULATED / "battery_simulation.csv", index=False)
    
    pd.DataFrame({
        "timestamp": master_df["timestamp"],
        "diesel_generation_kw": diesel_gen,
        "diesel_status": diesel_status,
        "fuel_consumption_lph": fuel_lph
    }).to_csv(DATA_SIMULATED / "diesel_simulation.csv", index=False)
    
    # 3. Scenario Engine Evaluation
    logger.info("Generating scenarios using ScenarioEngine...")
    engine = ScenarioEngine()
    
    # We will generate a scenarios CSV file with all scenarios stacked
    scenario_list = ["normal", "polar_night", "extreme_cold", "low_renewable", "load_spike", "generator_failure"]
    scenario_dfs = []
    
    # For scenario evaluation, we take a slice of weather data (e.g., first 168 hours / 1 week)
    eval_slice = master_df.head(168).copy()
    
    for s_name in scenario_list:
        logger.info(f"Generating physical conditions for scenario: {s_name}")
        scen_weather = engine.apply_scenario(eval_slice, s_name)
        scenario_dfs.append(scen_weather)
        
    stacked_scenarios = pd.concat(scenario_dfs, ignore_index=True)
    stacked_scenarios.to_csv(DATA_SIMULATED / "polar_scenarios.csv", index=False)
    logger.info(f"Polar scenarios dataset saved: {stacked_scenarios.shape} rows")
    
    # 4. Update master dataset energy columns with baseline simulation values
    logger.info("Updating master dataset with clean baseline simulation values...")
    master_df["load_kw"] = total_load
    master_df["critical_load_kw"] = critical_load
    master_df["solar_generation_kw"] = solar_gen
    master_df["wind_generation_kw"] = wind_gen
    master_df["battery_soc"] = battery_socs
    master_df["diesel_generation_kw"] = diesel_gen
    master_df["fuel_consumption_lph"] = fuel_lph
    master_df["diesel_status"] = diesel_status
    
    master_df.to_csv(DATA_PROCESSED / "master_dataset.csv", index=False)
    logger.info("Master dataset updated successfully.")
    
    logger.info("Physics-Informed Energy Simulation Complete!")

if __name__ == "__main__":
    run_physics_simulation()
