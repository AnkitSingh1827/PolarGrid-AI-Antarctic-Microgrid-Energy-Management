"""
PolarGrid AI — Data Validation
Performs physics-informed checks on values and logs statistics/anomalies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def validate_weather_ranges(df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
    """
    Validate that weather features lie within physically possible bounds in Antarctica.
    Returns validation summary dictionary.
    """
    report = {
        "source": source_name,
        "total_records": len(df),
        "anomalies": {}
    }
    
    # Physics bounds check dictionary
    bounds = {
        "temperature_c": (-60.0, 15.0),
        "pressure_hpa": (900.0, 1050.0),
        "wind_speed_ms": (0.0, 50.0),
    }
    
    for col, (min_val, max_val) in bounds.items():
        if col in df.columns:
            # Count values outside bounds
            series = df[col].dropna()
            out_of_bounds = series[(series < min_val) | (series > max_val)]
            count = len(out_of_bounds)
            pct = (count / len(series)) * 100 if len(series) > 0 else 0
            
            report["anomalies"][col] = {
                "missing_count": df[col].isnull().sum(),
                "out_of_bounds_count": count,
                "out_of_bounds_pct": pct,
                "min_observed": float(series.min()) if len(series) > 0 else None,
                "max_observed": float(series.max()) if len(series) > 0 else None,
            }
            if count > 0:
                logger.warning(
                    f"[{source_name}] Found {count} out-of-bounds values ({pct:.2f}%) for '{col}' "
                    f"outside [{min_val}, {max_val}] range. Min: {series.min()}, Max: {series.max()}"
                )
    
    return report

def compare_era5_vs_ncpor(era5_df: pd.DataFrame, ncpor_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare overlapping data between ERA5 and NCPOR Maitri AWS to check consistency.
    Both must have 'valid_time' aligned.
    """
    report = {}
    
    # Align on time
    merged = pd.merge(
        era5_df[["valid_time", "temperature_c", "wind_speed_ms", "pressure_hpa"]],
        ncpor_df[["valid_time", "temperature_c", "wind_speed_ms", "pressure_hpa"]],
        on="valid_time",
        suffixes=("_era5", "_ncpor")
    )
    
    if merged.empty:
        logger.warning("No overlapping time coverage found between ERA5 and NCPOR Maitri for validation comparison.")
        return {"aligned_records": 0}
        
    # Calculate difference metrics
    temp_mae = np.mean(np.abs(merged["temperature_c_era5"] - merged["temperature_c_ncpor"]))
    temp_corr = merged["temperature_c_era5"].corr(merged["temperature_c_ncpor"])
    
    wind_mae = np.mean(np.abs(merged["wind_speed_ms_era5"] - merged["wind_speed_ms_ncpor"]))
    wind_corr = merged["wind_speed_ms_era5"].corr(merged["wind_speed_ms_ncpor"])
    
    press_mae = np.mean(np.abs(merged["pressure_hpa_era5"] - merged["pressure_hpa_ncpor"]))
    press_corr = merged["pressure_hpa_era5"].corr(merged["pressure_hpa_ncpor"])
    
    report = {
        "aligned_records": len(merged),
        "temperature": {"mae": float(temp_mae), "correlation": float(temp_corr)},
        "wind_speed": {"mae": float(wind_mae), "correlation": float(wind_corr)},
        "pressure": {"mae": float(press_mae), "correlation": float(press_corr)}
    }
    
    logger.info(
        f"ERA5 vs NCPOR Weather Alignment check: Temp MAE = {temp_mae:.2f}°C (corr={temp_corr:.2f}), "
        f"Wind Speed MAE = {wind_mae:.2f} m/s (corr={wind_corr:.2f})"
    )
    
    return report
