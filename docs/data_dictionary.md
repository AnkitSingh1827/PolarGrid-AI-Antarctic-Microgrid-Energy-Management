# PolarGrid AI — Data Dictionary

## Overview
This document describes all data sources, processed datasets, and simulated outputs used in the PolarGrid AI system.

---

## Raw Data Sources

### ERA5 Reanalysis (`data/raw/ERA5/ERA5_raw_original.csv`)
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `valid_time` | datetime | UTC | Timestamp |
| `u10` | float | m/s | 10m U-component of wind |
| `v10` | float | m/s | 10m V-component of wind |
| `d2m` | float | K | 2m dewpoint temperature |
| `t2m` | float | K | 2m air temperature |
| `sp` | float | Pa | Surface pressure |
| `ssrd` | float | J/m² | Surface solar radiation downward (cumulative) |
| `strd` | float | J/m² | Surface thermal radiation downward |
| `tcc` | float | 0-1 | Total cloud cover |
| `tp` | float | m | Total precipitation |
| `latitude` | float | ° | -70.75 (Maitri) |
| `longitude` | float | ° | 11.75 (Maitri) |

**Time Range:** 2012-01-01 to 2025-12-31 (hourly)  
**Shape:** 122,736 rows × 12 columns

### NCPOR Maitri AWS (`data/raw/NCPOR_Maitri/NCPOR_Maitri_AWS_raw_combined.csv`)
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `obstime` | datetime | UTC | Observation timestamp |
| `tempr` | float | °C | Air temperature |
| `ap` | float | hPa | Atmospheric pressure |
| `ws` | float | m/s | Wind speed |
| `wd` | float | ° | Wind direction |
| `rh` | float | % | Relative humidity |

**Time Range:** 2012-01-01 to 2016-12-19 (hourly)  
**Sentinel:** -999 for missing values

---

## Processed Datasets

### Master Dataset (`data/processed/master_dataset.csv`)
The primary dataset combining cleaned weather with simulated energy variables.

| Column | Type | Unit | Source |
|--------|------|------|--------|
| `timestamp` | datetime | UTC | ERA5 valid_time |
| `temperature` | float | °C | ERA5 t2m (converted from K) |
| `pressure` | float | hPa | ERA5 sp (converted from Pa) |
| `humidity` | float | % | Calculated from ERA5 d2m and t2m |
| `wind_speed` | float | m/s | Calculated from ERA5 u10, v10 |
| `wind_direction` | float | ° | Calculated from ERA5 u10, v10 |
| `solar_radiation` | float | W/m² | ERA5 ssrd (converted from J/m²) |
| `cloud_cover` | float | 0-1 | ERA5 tcc |
| `hour` | int | 0-23 | Temporal feature |
| `day` | int | 1-31 | Temporal feature |
| `month` | int | 1-12 | Temporal feature |
| `day_of_year` | int | 1-366 | Temporal feature |
| `day_of_week` | int | 0-6 | Temporal feature |
| `season` | int | 0-3 | 0=Summer, 1=Autumn, 2=Winter, 3=Spring |
| `polar_night` | int | 0/1 | 1 if continuous darkness |
| `extreme_cold` | int | 0/1 | 1 if temperature < -30°C |
| `load_kw` | float | kW | Simulated station load |
| `critical_load_kw` | float | kW | Critical (unsheddable) load |
| `solar_generation_kw` | float | kW | Simulated solar PV output |
| `wind_generation_kw` | float | kW | Simulated wind turbine output |
| `battery_soc` | float | 0-1 | Battery state of charge |
| `diesel_generation_kw` | float | kW | Diesel generator output |
| `fuel_consumption_lph` | float | L/h | Diesel fuel consumption |
| `diesel_status` | float | 0/1 | Generator on/off |

---

## Simulated Outputs

All simulated files are in `data/simulated/`:

| File | Description | Key Columns |
|------|-------------|-------------|
| `solar_simulation.csv` | Hourly solar PV generation | timestamp, solar_generation_kw, ghi_wm2 |
| `wind_simulation.csv` | Hourly wind turbine output | timestamp, wind_generation_kw, wind_speed_ms |
| `load_simulation.csv` | Hourly station load demand | timestamp, load_kw, critical_load_kw |
| `battery_simulation.csv` | Battery charge/discharge/SOC | timestamp, battery_soc, charge_kw, discharge_kw |
| `diesel_simulation.csv` | Diesel generation and fuel | timestamp, diesel_kw, fuel_lph, status |
| `polar_scenarios.csv` | Scenario simulation results | scenario, timestamp, load_kw, solar_kw, wind_kw |

---

## Model Outputs

### Forecasting (`results/forecasts/`)
- `forecast_metrics.json`: MAE, RMSE, R², MAPE for load, solar, wind
- `forecast_predictions.csv`: Test set actual vs predicted values
- `*_forecast_plot.png`: Visualization plots

### Optimization (`results/optimization/`)
- `milp_schedule_24h.csv`: MILP dispatch schedule

### RL Evaluation (`results/rl/`)
- `*_dispatch_log.csv`: Hourly dispatch logs per controller

### Comparison (`results/comparison/`)
- `controller_comparison.csv`: Side-by-side metrics
- `controller_comparison.json`: Structured comparison data
