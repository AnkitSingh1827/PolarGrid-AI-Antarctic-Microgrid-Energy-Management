"""
PolarGrid AI — Data Loader
Loads and parses all raw datasets with automatic schema detection.

Data Provenance:
    REAL/REANALYSIS: ERA5, NCPOR Maitri AWS, AntAWS, Antarctic Electricity, Antarctic Fuel
    These are observational or reanalysis datasets — NOT simulated.
"""

import pandas as pd
import numpy as np
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SIMULATED = PROJECT_ROOT / "data" / "simulated"
CONFIGS = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    """Load a YAML configuration file."""
    path = CONFIGS / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_antarctic_date(val) -> pd.Timestamp:
    """Helper to parse dates that might have formats like Jan-86, Jan-00, or be Timestamp objects."""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        return val
    # If it is a datetime object
    if hasattr(val, "strftime"):
        return pd.Timestamp(val)
        
    val_str = str(val).strip()
    # Check if format is like 'Jan-86'
    parts = val_str.split('-')
    if len(parts) == 2:
        mmm, yy = parts[0], parts[1]
        try:
            yy_val = int(yy)
            # If year is e.g. 86 -> 1986, if 00 -> 2000, if 22 -> 2022
            year = 1900 + yy_val if yy_val >= 80 else 2000 + yy_val
            months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            month = months.get(mmm[:3].title(), 1)
            return pd.Timestamp(year=year, month=month, day=1)
        except ValueError:
            pass
            
    try:
        return pd.to_datetime(val_str, errors='coerce')
    except Exception:
        return pd.NaT


def load_era5(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load ERA5 hourly reanalysis data.

    Source: ECMWF ERA5 reanalysis at Maitri coordinates (-70.75, 11.75)
    Provenance: REAL REANALYSIS DATA
    Period: 2012-01-01 to 2025-12-31 (hourly)

    Raw columns:
        valid_time: timestamp (UTC)
        u10: 10m U-component of wind (m/s)
        v10: 10m V-component of wind (m/s)
        d2m: 2m dewpoint temperature (K)
        t2m: 2m temperature (K)
        sp: surface pressure (Pa)
        ssrd: surface solar radiation downwards (J/m², accumulated)
        strd: surface thermal radiation downwards (J/m², accumulated)
        tcc: total cloud cover (0-1)
        tp: total precipitation (m)
        latitude, longitude: coordinates
    """
    if filepath is None:
        filepath = DATA_RAW / "ERA5" / "ERA5_raw_original.csv"

    logger.info(f"Loading ERA5 data from {filepath}")
    df = pd.read_csv(filepath)

    # Parse timestamp
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df = df.sort_values("valid_time").reset_index(drop=True)

    logger.info(f"ERA5 loaded: {df.shape[0]} rows, {df['valid_time'].min()} to {df['valid_time'].max()}")
    return df


def load_ncpor_maitri(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load NCPOR Maitri Automatic Weather Station data.

    Source: NCPOR (National Centre for Polar and Ocean Research) Maitri AWS
    Provenance: REAL OBSERVATIONAL DATA
    Period: 2012-01-01 to 2016-12-19 (hourly)

    Raw columns:
        obstime: observation timestamp (UTC)
        tempr: temperature (°C)
        ap: atmospheric pressure (hPa)
        ws: wind speed (m/s, but values up to 81 m/s need validation — likely km/h)
        wd: wind direction (degrees, -999 = missing)
        rh: relative humidity (%, -999 = missing)

    Note: -999 is used as sentinel for missing values.
    """
    if filepath is None:
        filepath = DATA_RAW / "NCPOR_Maitri" / "NCPOR_Maitri_AWS_raw_combined.csv"

    logger.info(f"Loading NCPOR Maitri AWS data from {filepath}")
    df = pd.read_csv(filepath)

    # Parse timestamp
    df["obstime"] = pd.to_datetime(df["obstime"], utc=True)
    df = df.sort_values("obstime").reset_index(drop=True)

    logger.info(f"NCPOR Maitri loaded: {df.shape[0]} rows, {df['obstime'].min()} to {df['obstime'].max()}")
    return df


def load_antarctic_electricity(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load Australian Antarctic station electricity usage data.

    Source: Australian Antarctic Division — Indicator 59
    Provenance: REAL OBSERVATIONAL DATA (Australian stations, NOT Maitri)
    Stations: Mawson, Davis, Casey, Macquarie Island
    Period: ~1986 to ~2022 (monthly)
    Unit: kWh per month

    WARNING: This is NOT Indian Maitri station data.
    Use for calibration, reference patterns, and independent validation only.
    """
    if filepath is None:
        filepath = DATA_RAW / "Antarctic_Electricity" / "indicator_59.csv"

    logger.info(f"Loading Antarctic electricity data from {filepath}")
    df = pd.read_csv(filepath)

    # The file has header-in-data format. First row contains actual column names.
    actual_columns = df.iloc[0].values
    df = df.iloc[1:].reset_index(drop=True)
    # Assign meaningful names
    col_map = {
        df.columns[0]: "date",
        df.columns[1]: "latitude",
        df.columns[2]: "longitude",
        df.columns[3]: "station",
        df.columns[4]: "parameter",
        df.columns[5]: "value_kwh",
        df.columns[6]: "error",
        df.columns[7]: "unit",
    }
    df = df.rename(columns=col_map)
    # Keep only useful columns
    df = df[["date", "latitude", "longitude", "station", "value_kwh", "unit"]].copy()

    # Parse date (format: "Jan-86", "Feb-86", etc.)
    df["date"] = df["date"].apply(parse_antarctic_date)

    # Convert value to numeric
    df["value_kwh"] = pd.to_numeric(df["value_kwh"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    logger.info(f"Antarctic electricity loaded: {df.shape[0]} rows, stations: {df['station'].unique()}")
    return df


def load_antarctic_fuel(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load Australian Antarctic station fuel usage data.

    Source: Australian Antarctic Division — Indicator 56
    Provenance: REAL OBSERVATIONAL DATA (Australian stations, NOT Maitri)
    Stations: Mawson, Davis, Casey, Macquarie Island
    Period: ~1993 to ~2022 (monthly)
    Unit: litres per month

    WARNING: This is NOT Indian Maitri station data.
    Use for calibration and reference patterns only.
    """
    if filepath is None:
        filepath = DATA_RAW / "Antarctic_Fuel" / "SOE_generator_boiler_fuel_usage.xlsx"

    logger.info(f"Loading Antarctic fuel data from {filepath}")
    df = pd.read_excel(filepath, sheet_name="indicator_56")

    # Header-in-data format
    actual_columns = df.iloc[0].values
    df = df.iloc[1:].reset_index(drop=True)

    col_map = {
        df.columns[0]: "date",
        df.columns[1]: "latitude",
        df.columns[2]: "longitude",
        df.columns[3]: "station",
        df.columns[4]: "parameter",
        df.columns[5]: "value_litres",
        df.columns[6]: "error",
        df.columns[7]: "unit",
    }
    df = df.rename(columns=col_map)
    df = df[["date", "latitude", "longitude", "station", "value_litres", "unit"]].copy()

    # Parse date
    df["date"] = df["date"].apply(parse_antarctic_date)
    df["value_litres"] = pd.to_numeric(df["value_litres"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    logger.info(f"Antarctic fuel loaded: {df.shape[0]} rows, stations: {df['station'].unique()}")
    return df


def load_antaws_station(station_name: str, encoding: str = "latin1") -> pd.DataFrame:
    """
    Load a single AntAWS station file.

    Source: Antarctic Automatic Weather Stations (AntAWS) network
    Provenance: REAL OBSERVATIONAL DATA (various Antarctic stations, NOT Maitri)
    Resolution: 3-hourly

    Raw columns:
        Year, Month, Day: date components
        Three-hourly observation time(UTC): hour (0,3,6,9,12,15,18,21)
        Temperature(°): temperature in Celsius
        Pressure(hPa): atmospheric pressure
        Wind Speed(m/s): wind speed
        Wind Direction: wind direction in degrees
        Relative Humidity(%): relative humidity

    WARNING: AntAWS stations are NOT Maitri. Use for Antarctic-wide validation only.
    """
    filepath = DATA_RAW / "AntAWS" / "AntAWS_3hourly" / f"{station_name}_3h.csv"
    if not filepath.exists():
        # Try with exact filename
        filepath = DATA_RAW / "AntAWS" / "AntAWS_3hourly" / station_name
        if not filepath.exists():
            raise FileNotFoundError(f"AntAWS station file not found: {station_name}")

    df = pd.read_csv(filepath, encoding=encoding)

    # Standardize column names
    col_renames = {}
    for col in df.columns:
        col_lower = col.lower()
        if "temperature" in col_lower:
            col_renames[col] = "temperature_c"
        elif "pressure" in col_lower:
            col_renames[col] = "pressure_hpa"
        elif "wind speed" in col_lower:
            col_renames[col] = "wind_speed_ms"
        elif "wind direction" in col_lower:
            col_renames[col] = "wind_direction"
        elif "humidity" in col_lower:
            col_renames[col] = "relative_humidity_pct"
        elif "three-hourly" in col_lower or "observation time" in col_lower:
            col_renames[col] = "hour_utc"

    df = df.rename(columns=col_renames)

    # Create proper timestamp
    if all(c in df.columns for c in ["Year", "Month", "Day", "hour_utc"]):
        df["timestamp"] = pd.to_datetime(
            df[["Year", "Month", "Day"]].assign(hour=df["hour_utc"]),
            errors="coerce"
        )
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    return df


def load_antaws_metadata() -> pd.DataFrame:
    """Load AntAWS station metadata (locations, names)."""
    filepath = DATA_RAW / "AntAWS" / "AntAWS_3hourly" / "Main characteristics of automatic weather station.csv"
    df = pd.read_csv(filepath, encoding="latin1")
    return df


def get_antaws_station_list() -> list:
    """Get list of available AntAWS station files."""
    aws_dir = DATA_RAW / "AntAWS" / "AntAWS_3hourly"
    stations = []
    for f in aws_dir.glob("*_3h.csv"):
        name = f.stem.replace("_3h", "")
        stations.append(name)
    return sorted(stations)


def load_all_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Load all raw datasets and return as a dictionary.
    Does NOT merge unrelated datasets.
    """
    data = {}
    data["era5"] = load_era5()
    data["ncpor_maitri"] = load_ncpor_maitri()
    data["electricity"] = load_antarctic_electricity()
    data["fuel"] = load_antarctic_fuel()
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Loading all raw datasets...")

    data = load_all_raw_data()

    for name, df in data.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"First 2 rows:\n{df.head(2)}")

    # Also list available AntAWS stations
    stations = get_antaws_station_list()
    print(f"\nAntAWS stations available: {len(stations)}")
    print(f"Sample: {stations[:5]}")
