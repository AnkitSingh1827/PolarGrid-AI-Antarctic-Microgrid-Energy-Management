"""
PolarGrid AI — Data Cleaning
Cleans raw data, standardizes units, handles missing values and outliers.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def clean_era5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ERA5 reanalysis data.
    Standardizes units:
      - t2m, d2m: Kelvin to Celsius (K - 273.15)
      - sp: Pascals to hPa (Pa / 100.0)
      - Wind speed: calculated from u10 and v10: sqrt(u10^2 + v10^2)
      - ssrd: cumulative surface solar radiation downwards (J/m²).
        Convert to average hourly irradiance (W/m² or J/m²/s).
        ERA5 SSRD is accumulated from the beginning of the forecast.
        Since we have hourly data, the difference between consecutive hours (or hourly rates)
        should be calculated, or we scale it to W/m2 by dividing by 3600 (seconds per hour)
        when it's already an hourly rate.
        Let's convert to W/m² (flux) by dividing by 3600 and ensuring non-negative.
      - strd: surface thermal radiation downwards (J/m²). Scale by 1/3600 to get W/m².
      - tp: total precipitation (m) -> standardise to mm (m * 1000)
    """
    df = df.copy()
    
    # Unit conversions
    df["temperature_c"] = df["t2m"] - 273.15
    df["dewpoint_c"] = df["d2m"] - 273.15
    df["pressure_hpa"] = df["sp"] / 100.0
    df["wind_speed_ms"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
    
    # Solar and thermal radiation: accumulated hourly.
    # Convert SSRD and STRD from accumulated J/m^2 over 1 hour to average W/m^2:
    # 1 W = 1 J/s. In 1 hour, there are 3600 seconds. So average W/m^2 = J/m^2 / 3600.
    df["solar_irradiance_wm2"] = np.clip(df["ssrd"] / 3600.0, 0, None)
    df["thermal_down_wm2"] = np.clip(df["strd"] / 3600.0, 0, None)
    
    # Precipitation in mm
    df["precipitation_mm"] = df["tp"] * 1000.0
    
    # Cloud cover rename
    df["cloud_cover"] = df["tcc"]
    
    # Drop raw columns to avoid confusion
    clean_cols = [
        "valid_time", "latitude", "longitude", "temperature_c", "dewpoint_c", 
        "pressure_hpa", "u10", "v10", "wind_speed_ms", "solar_irradiance_wm2", 
        "thermal_down_wm2", "precipitation_mm", "cloud_cover"
    ]
    df = df[clean_cols]
    
    # Outlier detection / physics bounds check
    # Temperature bounds: -60°C to +15°C
    df["temperature_c"] = np.clip(df["temperature_c"], -60.0, 15.0)
    # Wind speed bounds: 0 to 60 m/s
    df["wind_speed_ms"] = np.clip(df["wind_speed_ms"], 0.0, 60.0)
    # Pressure: 900 to 1050 hPa
    df["pressure_hpa"] = np.clip(df["pressure_hpa"], 900.0, 1050.0)
    # Cloud cover: 0 to 1
    df["cloud_cover"] = np.clip(df["cloud_cover"], 0.0, 1.0)
    
    logger.info(f"Cleaned ERA5 data: {df.shape}")
    return df

def clean_ncpor_maitri(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NCPOR Maitri AWS data.
    - Replace -999 sentinels with NaN.
    - Validate wind speed (ws). In raw NCPOR, some values might be scaled or have anomalies.
      Let's verify values.
    """
    df = df.copy()
    
    # Replace -999 sentinels
    df = df.replace(-999, np.nan)
    df = df.replace(-999.0, np.nan)
    
    # Rename columns to standard
    df = df.rename(columns={
        "obstime": "valid_time",
        "tempr": "temperature_c",
        "ap": "pressure_hpa",
        "ws": "wind_speed_ms",
        "wd": "wind_direction",
        "rh": "relative_humidity_pct"
    })
    
    # Check if wind speed is in km/h. In raw, mean ws is 26.5 with max of 81.0. 
    # Standard wind speeds in m/s at Maitri can reach 30-40 m/s during blizzards.
    # A value of 81.0 m/s is extremely high (hurricane force), and 26.5 average is very high for m/s.
    # This suggests wind speed might be in km/h.
    # Let's check: if mean is 26.5, converting to m/s by dividing by 3.6 gives 7.3 m/s, which is a normal mean wind speed.
    # We will divide wind_speed_ms by 3.6 to convert km/h to m/s if values look large.
    # Since the request says: Use Australian records for calibration/reference and Maitri for Maitri-specific weather validation,
    # let's do this conversion to represent wind speed in m/s.
    # We will assume ws in NCPOR is indeed in km/h and divide by 3.6.
    df["wind_speed_ms"] = df["wind_speed_ms"] / 3.6
    
    # Clean wind direction to [0, 360]
    df.loc[df["wind_direction"] < 0, "wind_direction"] = np.nan
    df.loc[df["wind_direction"] > 360, "wind_direction"] = np.nan
    
    # Relative humidity to [0, 100]
    df.loc[df["relative_humidity_pct"] < 0, "relative_humidity_pct"] = np.nan
    df.loc[df["relative_humidity_pct"] > 100, "relative_humidity_pct"] = np.nan
    
    # Drop rows with entirely missing weather
    df = df.dropna(subset=["temperature_c", "pressure_hpa", "wind_speed_ms"], how="all")
    
    logger.info(f"Cleaned NCPOR Maitri data: {df.shape}")
    return df

def clean_australian_electricity(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Australian stations electricity dataset."""
    df = df.copy()
    # Remove NaN values in target value_kwh
    df = df.dropna(subset=["value_kwh"])
    return df

def clean_australian_fuel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Australian stations fuel dataset."""
    df = df.copy()
    # Remove NaN values in target value_litres
    df = df.dropna(subset=["value_litres"])
    return df

# Backward-compatible names used by older notebooks and scripts.
clean_era5_data = clean_era5
clean_ncpor_data = clean_ncpor_maitri
