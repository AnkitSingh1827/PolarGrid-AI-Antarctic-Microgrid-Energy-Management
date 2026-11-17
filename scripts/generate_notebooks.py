"""
PolarGrid AI — Jupyter Notebooks Generator
Generates all 8 interactive, fully documented Jupyter notebooks in the notebooks/ folder.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


# ==============================================================================
# Notebook 1: Data Exploration
# ==============================================================================
nb1_cells = [
    md_cell("""# 01 — PolarGrid AI: Exploratory Data Analysis (EDA)
**SIH 2026 • Problem Statement 26061: AI-Driven Smart Energy Management System for Polar Research Stations**

This notebook inspects the real Antarctic meteorological datasets and historical reference data:
1. **ERA5 Reanalysis** (2012–2025 hourly at Maitri coordinates: $70.75^\\circ\\text{S}, 11.75^\\circ\\text{E}$)
2. **NCPOR Maitri AWS** ground truth observations (2012–2016)
3. **Australian Antarctic Reference Energy Records** (Electricity & Fuel monthly usage)
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure project path
PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
print("Project root:", PROJECT_ROOT)
"""),
    md_cell("## 1. Load and Inspect Raw ERA5 Reanalysis Weather Data"),
    code_cell("""era5_path = PROJECT_ROOT / "data" / "raw" / "ERA5" / "ERA5_raw_original.csv"
df_era5 = pd.read_csv(era5_path, nrows=5000)
print(f"ERA5 Sample Shape: {df_era5.shape}")
df_era5.head()
"""),
    code_cell("""# Summary statistics of raw meteorological variables
df_era5.describe().T
"""),
    md_cell("## 2. Load NCPOR Maitri AWS Ground Truth Data"),
    code_cell("""ncpor_path = PROJECT_ROOT / "data" / "raw" / "NCPOR_Maitri" / "NCPOR_Maitri_AWS_raw_combined.csv"
df_ncpor = pd.read_csv(ncpor_path, nrows=5000)
print(f"NCPOR AWS Sample Shape: {df_ncpor.shape}")
df_ncpor.head()
"""),
    md_cell("## 3. Weather Distributions and Seasonal Trends at Maitri Station"),
    code_cell("""# Load clean processed dataset for visualization
master_path = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
df_master = pd.read_csv(master_path, nrows=8760) # 1 full year sample

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df_master['temperature'], kde=True, ax=axes[0, 0], color='tomato')
axes[0, 0].set_title("Temperature Distribution (°C)")

sns.histplot(df_master['wind_speed'], kde=True, ax=axes[0, 1], color='dodgerblue')
axes[0, 1].set_title("Wind Speed Distribution (m/s)")

sns.histplot(df_master[df_master['solar_radiation'] > 0]['solar_radiation'], kde=True, ax=axes[1, 0], color='goldenrod')
axes[1, 0].set_title("Daylight Solar Radiation (W/m²)")

sns.boxplot(x='month', y='temperature', data=df_master, ax=axes[1, 1], palette='coolwarm')
axes[1, 1].set_title("Monthly Temperature Variation at Maitri")
plt.tight_layout()
plt.show()
""")
]

# ==============================================================================
# Notebook 2: Data Cleaning & Feature Engineering
# ==============================================================================
nb2_cells = [
    md_cell("""# 02 — Data Cleaning & Feature Engineering
Demonstrates unit conversions (Kelvin to Celsius, Pa to hPa, J/m² to W/m²), sentinel value (-999) replacement, wind vector resolution, polar night detection, and temporal features.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
"""),
    md_cell("## 1. Load Processed Clean Weather Data"),
    code_cell("""from src.data.cleaning import clean_era5_data, clean_ncpor_data
weather_path = PROJECT_ROOT / "data" / "processed" / "weather_processed.csv"
df_weather = pd.read_csv(weather_path, nrows=2000)
df_weather.head()
"""),
    md_cell("## 2. Feature Engineering: Wind Components & Solar Radiance"),
    code_cell("""from src.data.feature_engineering import create_features
df_features = create_features(df_weather)
print("Engineered Columns:", [c for c in df_features.columns if c not in df_weather.columns])
df_features[['hour', 'season', 'polar_night', 'extreme_cold']].head(10)
"""),
    md_cell("## 3. Polar Night and Extreme Cold Indicators"),
    code_cell("""plt.figure(figsize=(12, 4))
plt.plot(df_features['timestamp'].iloc[:720], df_features['solar_radiation'].iloc[:720], label='Solar Radiation (W/m²)', color='orange')
plt.plot(df_features['timestamp'].iloc[:720], df_features['polar_night'].iloc[:720] * 100, label='Polar Night Flag (x100)', color='navy', linestyle='--')
plt.title("Solar Irradiance and Polar Night Detection")
plt.xlabel("Timestep")
plt.legend()
plt.tight_layout()
plt.show()
""")
]

# ==============================================================================
# Notebook 3: Physics Simulation
# ==============================================================================
nb3_cells = [
    md_cell("""# 03 — Physics-Informed Microgrid Energy Simulation
Simulates:
1. **Solar PV Generation** (Temperature derating + Polar Night shutoff)
2. **Wind Turbine Generation** (Piecewise cubic power curve, cut-in, rated, cut-out)
3. **Station Load Profile** (Base load, temperature-dependent heating demand, research load)
4. **Lithium-Ion Battery Storage (BESS)** (Charge/discharge efficiency, sub-zero cold derating)
5. **Diesel Generator Sets** (Fuel consumption curves, startup costs, minimum loading)
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
"""),
    md_cell("## 1. Wind Turbine Power Curve Simulation"),
    code_cell("""from src.simulation.wind import simulate_wind_power
speeds = np.linspace(0, 30, 300)
power_curve = [simulate_wind_power(s, capacity_kw=100.0, cut_in_speed=3.0, rated_speed=12.0, cut_out_speed=25.0) for s in speeds]

plt.figure(figsize=(9, 4.5))
plt.plot(speeds, power_curve, color='dodgerblue', lw=2.5)
plt.axvline(3.0, color='gray', linestyle=':', label='Cut-in (3 m/s)')
plt.axvline(12.0, color='green', linestyle=':', label='Rated (12 m/s)')
plt.axvline(25.0, color='red', linestyle=':', label='Cut-out (25 m/s)')
plt.title("Polar Wind Turbine Power Curve (100 kW Rated)")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (kW)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
"""),
    md_cell("## 2. Solar PV Output with Thermal Derating"),
    code_cell("""from src.simulation.solar import simulate_solar_power
irradiance = np.linspace(0, 1000, 200)
p_cold = [simulate_solar_power(g, ambient_temp_c=-20.0, capacity_kw=50.0) for g in irradiance]
p_warm = [simulate_solar_power(g, ambient_temp_c=25.0, capacity_kw=50.0) for g in irradiance]

plt.figure(figsize=(9, 4.5))
plt.plot(irradiance, p_cold, label='T = -20°C (High Efficiency)', color='deepskyblue', lw=2)
plt.plot(irradiance, p_warm, label='T = +25°C (STC Reference)', color='darkorange', lw=2)
plt.title("Solar PV Power Output vs Irradiance")
plt.xlabel("Solar Irradiance (W/m²)")
plt.ylabel("Power Output (kW)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
"""),
    md_cell("## 3. Battery Storage & Cold Weather Derating"),
    code_cell("""from src.simulation.battery import BatterySimulation
temps = np.linspace(-50, 10, 100)
battery = BatterySimulation(capacity_kwh=500.0, cold_threshold_c=-20.0, cold_derating_factor=0.005)
effective_caps = [battery.get_effective_capacity(t) for t in temps]

plt.figure(figsize=(9, 4))
plt.plot(temps, effective_caps, color='purple', lw=2.5)
plt.axvline(-20.0, color='crimson', linestyle='--', label='Cold Derating Threshold (-20°C)')
plt.title("500 kWh Battery Bank: Effective Capacity vs Ambient Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Available Capacity (kWh)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
""")
]

# ==============================================================================
# Notebook 4: Load Forecasting
# ==============================================================================
nb4_cells = [
    md_cell("""# 04 — 24-Hour Station Load Demand Forecasting
Uses **XGBoost Regressor** with temporal, meteorological, and lag features to predict hourly station electrical and heating load demand.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
"""),
    md_cell("## 1. Load Evaluation Metrics and Predictions"),
    code_cell("""preds_path = PROJECT_ROOT / "results" / "forecasts" / "forecast_predictions.csv"
metrics_path = PROJECT_ROOT / "results" / "forecasts" / "forecast_metrics.json"

df_preds = pd.read_csv(preds_path)
with open(metrics_path) as f:
    metrics = json.load(f)

print("Load Forecast Metrics:")
print(f"  MAE:  {metrics['load']['MAE']:.3f} kW")
print(f"  RMSE: {metrics['load']['RMSE']:.3f} kW")
print(f"  R²:   {metrics['load']['R2']:.4f}")
print(f"  MAPE: {metrics['load']['MAPE']:.2f}%")
"""),
    md_cell("## 2. Actual vs Predicted Load (Sample Week)"),
    code_cell("""sample = df_preds.head(168)
plt.figure(figsize=(14, 5))
plt.plot(sample['load_actual'], label='Actual Load Demand', color='#2563eb', lw=2)
plt.plot(sample['load_pred'], label='XGBoost Predicted Load', color='#f59e0b', linestyle='--', lw=2)
plt.title("24-Hour Ahead Load Demand Forecast (1 Week Test Horizon)")
plt.xlabel("Hour")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""")
]

# ==============================================================================
# Notebook 5: Renewable Forecasting
# ==============================================================================
nb5_cells = [
    md_cell("""# 05 — Solar and Wind Generation Forecasting
Demonstrates machine learning forecasting for variable renewable resources in Antarctica.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
"""),
    code_cell("""preds_path = PROJECT_ROOT / "results" / "forecasts" / "forecast_predictions.csv"
df_preds = pd.read_csv(preds_path)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Solar
axes[0].plot(df_preds['solar_actual'].iloc[:168], label='Actual Solar PV Output', color='goldenrod', lw=2)
axes[0].plot(df_preds['solar_pred'].iloc[:168], label='Predicted Solar Output', color='red', linestyle='--', lw=2)
axes[0].set_title("Solar PV Generation Forecast (Sample Week)")
axes[0].set_ylabel("Power (kW)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Wind
axes[1].plot(df_preds['wind_actual'].iloc[:168], label='Actual Wind Turbine Output', color='dodgerblue', lw=2)
axes[1].plot(df_preds['wind_pred'].iloc[:168], label='Predicted Wind Output', color='darkblue', linestyle='--', lw=2)
axes[1].set_title("Wind Turbine Generation Forecast (Sample Week)")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Power (kW)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")
]

# ==============================================================================
# Notebook 6: MILP Optimization
# ==============================================================================
nb6_cells = [
    md_cell("""# 06 — Mixed-Integer Linear Programming (MILP) Energy Scheduling
Formulates the 24-hour ahead microgrid dispatch schedule with Pyomo to minimize fuel consumption, battery degradation, and generator cycling costs.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.milp_model import solve_milp
"""),
    code_cell("""with open(PROJECT_ROOT / "configs" / "battery_config.yaml") as f:
    b_cfg = yaml.safe_load(f)["battery"]
with open(PROJECT_ROOT / "configs" / "diesel_config.yaml") as f:
    d_cfg = yaml.safe_load(f)["diesel"]
params = {"battery": b_cfg, "diesel": d_cfg}

# Load a 24-hour forecast window
master_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "master_dataset.csv")
sample_24h = master_df.iloc[1000:1024].copy()

load = sample_24h["load_kw"].values
critical = sample_24h["critical_load_kw"].values
solar = sample_24h["solar_generation_kw"].values
wind = sample_24h["wind_generation_kw"].values

schedule_df, solved = solve_milp(load, critical, solar, wind, params)
print("MILP Solved Optimally?", solved)
schedule_df.head()
"""),
    code_cell("""# Visualizing the 24-Hour Dispatch Schedule
plt.figure(figsize=(12, 6))
plt.plot(schedule_df['solar_use'], label='Solar Used (kW)', color='orange')
plt.plot(schedule_df['wind_use'], label='Wind Used (kW)', color='dodgerblue')
plt.plot(schedule_df['diesel_gen'], label='Diesel Gen (kW)', color='crimson')
plt.plot(schedule_df['battery_discharge'], label='Battery Discharge (kW)', color='purple', linestyle='--')
plt.plot(load, label='Total Demand (kW)', color='black', lw=2)
plt.title("24-Hour Optimal Energy Scheduling (MILP)")
plt.xlabel("Hour of Horizon")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""")
]

# ==============================================================================
# Notebook 7: RL Training
# ==============================================================================
nb7_cells = [
    md_cell("""# 07 — Reinforcement Learning Controller Training
Trains **PPO** (Proximal Policy Optimization) and **SAC** (Soft Actor-Critic) controllers inside a custom Gymnasium polar microgrid environment with hard physical safety projection constraints.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.reinforcement_learning.environment import PolarGridEnv
"""),
    code_cell("""master_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "master_dataset.csv")
train_df = master_df.iloc[:2000].copy()

env = PolarGridEnv(train_df)
obs, info = env.reset()
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Initial Observation:", obs)
"""),
    code_cell("""# Inspect test step with sample action
action = np.array([0.5, -0.2, 0.0]) # Charge battery, lower diesel target, diesel off
next_obs, reward, term, trunc, info = env.step(action)
print("Step Reward:", reward)
print("Dispatch Info:", info)
""")
]

# ==============================================================================
# Notebook 8: Model Comparison
# ==============================================================================
nb8_cells = [
    md_cell("""# 08 — Comprehensive Model Comparison & Benchmarks
Compares **Rule-Based**, **PPO Safe RL**, and **Soft Actor-Critic (SAC)** controllers across:
- Fuel consumption (Litres) and Fuel Savings (%)
- Reliability and Critical Load Protection (%)
- Renewable Utilization Rate (%)
- Battery Throughput and Cycle Degradation
- Total Operating Cost ($ USD)
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
"""),
    code_cell("""comp_path = PROJECT_ROOT / "results" / "comparison" / "controller_comparison.csv"
df_comp = pd.read_csv(comp_path)
df_comp
"""),
    code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Fuel
axes[0].bar(df_comp['controller'], df_comp['total_fuel_litres'], color=['#3b82f6', '#10b981', '#8b5cf6'])
axes[0].set_title("Total Fuel Consumed (Litres)")
axes[0].set_ylabel("Litres")

# Cost
axes[1].bar(df_comp['controller'], df_comp['total_cost_usd'], color=['#3b82f6', '#10b981', '#8b5cf6'])
axes[1].set_title("Total Operating Cost ($ USD)")
axes[1].set_ylabel("$ USD")

# Reliability
axes[2].bar(df_comp['controller'], df_comp['reliability_pct'], color=['#3b82f6', '#10b981', '#8b5cf6'])
axes[2].set_title("System Reliability (%)")
axes[2].set_ylim(95, 100.5)
axes[2].set_ylabel("Reliability %")

plt.tight_layout()
plt.show()
""")
]

notebooks = {
    "01_data_exploration.ipynb": nb1_cells,
    "02_data_cleaning.ipynb": nb2_cells,
    "03_physics_simulation.ipynb": nb3_cells,
    "04_load_forecasting.ipynb": nb4_cells,
    "05_renewable_forecasting.ipynb": nb5_cells,
    "06_milp_optimization.ipynb": nb6_cells,
    "07_rl_training.ipynb": nb7_cells,
    "08_model_comparison.ipynb": nb8_cells,
}

for filename, cells in notebooks.items():
    nb_dict = make_nb(cells)
    target = NOTEBOOKS_DIR / filename
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)
    print(f"Generated {target.name} ({len(cells)} cells)")

print("All 8 notebooks generated successfully.")
