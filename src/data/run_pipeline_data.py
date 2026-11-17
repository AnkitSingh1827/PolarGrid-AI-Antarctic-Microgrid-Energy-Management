"""
PolarGrid AI — Data Pipeline Execution Script
Ingests, cleans, validates, engineers features, and creates the master dataset.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.data.loader import load_era5, load_ncpor_maitri, load_antarctic_electricity, load_antarctic_fuel
from src.data.cleaning import clean_era5, clean_ncpor_maitri, clean_australian_electricity, clean_australian_fuel
from src.data.validation import validate_weather_ranges, compare_era5_vs_ncpor
from src.data.feature_engineering import add_temporal_features, calculate_relative_humidity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def run_data_pipeline():
    logger.info("Starting Data Pipeline...")
    
    # 1. Ingest
    logger.info("Ingesting raw datasets...")
    era5_raw = load_era5()
    ncpor_raw = load_ncpor_maitri()
    elec_raw = load_antarctic_electricity()
    fuel_raw = load_antarctic_fuel()
    
    # 2. Clean
    logger.info("Cleaning raw datasets...")
    era5_clean = clean_era5(era5_raw)
    ncpor_clean = clean_ncpor_maitri(ncpor_raw)
    elec_clean = clean_australian_electricity(elec_raw)
    fuel_clean = clean_australian_fuel(fuel_raw)
    
    # Save individual processed files
    era5_clean.to_csv(DATA_PROCESSED / "weather_processed.csv", index=False)
    elec_clean.to_csv(DATA_PROCESSED / "electricity_processed.csv", index=False)
    fuel_clean.to_csv(DATA_PROCESSED / "fuel_processed.csv", index=False)
    
    logger.info("Saved intermediate processed files: weather_processed.csv, electricity_processed.csv, fuel_processed.csv")
    
    # 3. Validate
    logger.info("Validating datasets...")
    era5_report = validate_weather_ranges(era5_clean, "ERA5")
    ncpor_report = validate_weather_ranges(ncpor_clean, "NCPOR Maitri")
    compare_report = compare_era5_vs_ncpor(era5_clean, ncpor_clean)
    
    # 4. Feature Engineering & Master Dataset Creation
    logger.info("Performing feature engineering on weather...")
    # Use ERA5 clean as our base since it spans the full 2012-2025 period
    master_df = add_temporal_features(era5_clean, "valid_time")
    
    # Add relative humidity using temperature and dewpoint
    master_df["humidity"] = calculate_relative_humidity(master_df["temperature_c"], master_df["dewpoint_c"])
    
    # Let's map standard weather columns
    # timestamp, temperature, pressure, humidity, wind_speed, wind_direction, solar_radiation, cloud_cover
    master_df = master_df.rename(columns={
        "valid_time": "timestamp",
        "temperature_c": "temperature",
        "pressure_hpa": "pressure",
        "wind_speed_ms": "wind_speed",
        "solar_irradiance_wm2": "solar_radiation"
    })
    
    # Wind direction: we will approximate wind direction from u10 and v10: atan2(u10, v10) in degrees
    # Wind direction (degrees) = (270 - atan2(v10, u10) * 180 / pi) % 360
    u10 = master_df["u10"]
    v10 = master_df["v10"]
    wind_dir_rad = np.arctan2(u10, v10)
    wind_dir_deg = np.degrees(wind_dir_rad) % 360.0
    master_df["wind_direction"] = wind_dir_deg
    
    # Drop raw u10, v10, dewpoint_c, precipitation_mm
    master_df = master_df.drop(columns=["u10", "v10", "dewpoint_c", "precipitation_mm"], errors="ignore")
    
    # Load energy simulation variables from baseline dataset to maintain consistency with historical baseline
    # Let's read PolarGrid_AI_Baseline_2012_2025.csv columns for energy simulation mapping
    baseline_path = DATA_PROCESSED / "PolarGrid_AI_Baseline_2012_2025.csv"
    if baseline_path.exists():
        logger.info(f"Merging simulated energy variables from baseline: {baseline_path}")
        baseline_df = pd.read_csv(baseline_path, usecols=[
            "valid_time", "load_kw", "critical_load_kw", "solar_generation_kw", 
            "wind_generation_kw", "battery_soc", "diesel_generation_kw", 
            "diesel_fuel_lph", "diesel_running"
        ])
        baseline_df["valid_time"] = pd.to_datetime(baseline_df["valid_time"], utc=True)
        # Rename valid_time to timestamp for alignment
        baseline_df = baseline_df.rename(columns={"valid_time": "timestamp"})
        
        # Merge
        master_df = pd.merge(master_df, baseline_df, on="timestamp", how="left")
    else:
        logger.warning("Baseline dataset PolarGrid_AI_Baseline_2012_2025.csv not found. Leaving energy columns empty for simulation phase.")
        # Create empty columns to be populated during simulation
        energy_cols = [
            "load_kw", "critical_load_kw", "solar_generation_kw", "wind_generation_kw",
            "battery_soc", "diesel_generation_kw", "fuel_consumption_lph", "diesel_status"
        ]
        for col in energy_cols:
            master_df[col] = np.nan
            
    # Standardize column naming
    master_df = master_df.rename(columns={
        "diesel_fuel_lph": "fuel_consumption_lph",
        "diesel_running": "diesel_status"
    })
    
    # Reorder columns to match request:
    # timestamp, temperature, pressure, humidity, wind_speed, wind_direction, solar_radiation, cloud_cover,
    # hour, day, month, day_of_year, day_of_week, season, polar_night, extreme_cold,
    # load_kw, critical_load_kw, solar_generation_kw, wind_generation_kw, battery_soc,
    # diesel_generation_kw, fuel_consumption_lph, diesel_status
    requested_cols = [
        "timestamp", "temperature", "pressure", "humidity", "wind_speed", "wind_direction", 
        "solar_radiation", "cloud_cover", "hour", "day", "month", "day_of_year", "day_of_week", 
        "season", "polar_night", "extreme_cold", "load_kw", "critical_load_kw", 
        "solar_generation_kw", "wind_generation_kw", "battery_soc", "diesel_generation_kw", 
        "fuel_consumption_lph", "diesel_status"
    ]
    
    # Ensure all requested columns exist
    for col in requested_cols:
        if col not in master_df.columns:
            master_df[col] = np.nan
            
    master_df = master_df[requested_cols]
    
    # Save master dataset
    master_df.to_csv(DATA_PROCESSED / "master_dataset.csv", index=False)
    logger.info(f"Master dataset generated at {DATA_PROCESSED / 'master_dataset.csv'} with shape {master_df.shape}")
    
    logger.info("Data Pipeline Complete!")
    return master_df

if __name__ == "__main__":
    run_data_pipeline()
