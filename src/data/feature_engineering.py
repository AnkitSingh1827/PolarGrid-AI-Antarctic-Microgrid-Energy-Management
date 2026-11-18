"""
PolarGrid AI — Feature Engineering
Creates temporal features, polar night flags, extreme cold events, and lag variables.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_temporal_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Add timezone-aware or naive datetime features, seasons, and daylight metrics."""
    df = df.copy()
    
    # Parse to datetime if not already
    ts = pd.to_datetime(df[time_col])
    
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["day_of_week"] = ts.dt.dayofweek
    df["year"] = ts.dt.year
    
    # Season mapping in Antarctica:
    # Nov-Feb: Summer, Mar-Apr: Autumn, May-Aug: Winter, Sep-Oct: Spring
    def get_season(m):
        if m in [11, 12, 1, 2]:
            return 0  # Summer
        elif m in [3, 4]:
            return 1  # Autumn
        elif m in [5, 6, 7, 8]:
            return 2  # Winter
        else:
            return 3  # Spring
            
    df["season"] = df["month"].apply(get_season)
    
    # Polar night flag (May 20 to July 21 roughly at Maitri -70.75 latitude)
    # Simply flag months 5, 6, 7 as polar night proxy, or check day of year between 140 and 205
    df["polar_night"] = ((df["day_of_year"] >= 140) & (df["day_of_year"] <= 205)).astype(int)
    
    # Extreme cold indicator: temperature below -30 C
    if "temperature_c" in df.columns:
        df["extreme_cold"] = (df["temperature_c"] < -30.0).astype(int)
    elif "temperature" in df.columns:
        df["extreme_cold"] = (df["temperature"] < -30.0).astype(int)
    else:
        df["extreme_cold"] = 0
        
    return df

def calculate_relative_humidity(temp_c: pd.Series, dew_c: pd.Series) -> pd.Series:
    """
    Estimate relative humidity from temperature and dewpoint temperature.
    Using Magnus-Tetens formula.
    """
    a = 17.625
    b = 243.04
    alpha = ((a * dew_c) / (b + dew_c))
    beta = ((a * temp_c) / (b + temp_c))
    rh = 100.0 * np.exp(alpha - beta)
    return np.clip(rh, 0.0, 100.0)

def generate_lags_and_rolling(df: pd.DataFrame, target_cols: list, lags: list, windows: list) -> pd.DataFrame:
    """Create lag features and rolling averages to capture temporal dependencies."""
    df = df.copy()
    for col in target_cols:
        if col in df.columns:
            # Lags
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            # Rolling means
            for window in windows:
                df[f"{col}_roll_mean_{window}"] = df[col].shift(1).rolling(window=window).mean()
                df[f"{col}_roll_std_{window}"] = df[col].shift(1).rolling(window=window).std()
                
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create notebook-ready temporal and weather features from cleaned ERA5 data."""
    df = df.copy()

    time_col = "valid_time" if "valid_time" in df.columns else "timestamp"
    if time_col not in df.columns:
        raise KeyError("Weather data must contain 'valid_time' or 'timestamp'")

    if time_col == "valid_time":
        df = df.rename(columns={"valid_time": "timestamp"})

    df = add_temporal_features(df, "timestamp")

    if "temperature" not in df.columns and "temperature_c" in df.columns:
        df["temperature"] = df["temperature_c"]
    if "pressure" not in df.columns and "pressure_hpa" in df.columns:
        df["pressure"] = df["pressure_hpa"]
    if "wind_speed" not in df.columns and "wind_speed_ms" in df.columns:
        df["wind_speed"] = df["wind_speed_ms"]
    if "solar_radiation" not in df.columns and "solar_irradiance_wm2" in df.columns:
        df["solar_radiation"] = df["solar_irradiance_wm2"]

    if "wind_direction" not in df.columns and {"u10", "v10"}.issubset(df.columns):
        df["wind_direction"] = (
            np.degrees(np.arctan2(df["u10"], df["v10"])) % 360.0
        )

    if "humidity" not in df.columns and {"temperature_c", "dewpoint_c"}.issubset(df.columns):
        df["humidity"] = calculate_relative_humidity(
            df["temperature_c"], df["dewpoint_c"]
        )

    return df
