# PolarGrid AI — Methodology

## 1. Problem Statement
SIH 2026 Problem Statement 26061: Design an AI-driven smart energy management system for Indian polar research stations to minimize diesel dependency while ensuring uninterrupted power supply under extreme Antarctic conditions.

## 2. System Architecture

```
ERA5 Reanalysis → Data Pipeline → Physics Simulation → Master Dataset
                                                            ↓
                                        XGBoost Forecasting (Load, Solar, Wind)
                                                            ↓
                                          MILP Optimization (24h scheduling)
                                                            ↓
                                        RL Controllers (PPO, SAC) + Safety Layer
                                                            ↓
                                              Dispatch Controller
                                                            ↓
                                     Dashboard (Streamlit) + API (FastAPI)
```

## 3. Data Pipeline

### 3.1 Data Sources
- **ERA5 Hourly Reanalysis** (ECMWF): Primary weather data at Maitri coordinates (70.75°S, 11.75°E), 2012-2025
- **NCPOR Maitri AWS**: Ground-truth weather validation (2012-2016)
- **Australian Antarctic Division**: Reference electricity/fuel consumption patterns

### 3.2 Processing
1. **Cleaning**: Sentinel value (-999) replacement, unit conversions (K→°C, Pa→hPa, J/m²→W/m²)
2. **Validation**: Physics-based bounds checking, cross-dataset consistency
3. **Feature Engineering**: Wind speed/direction from u10/v10, solar irradiance, humidity from dewpoint, temporal features, lag features

## 4. Physics-Informed Energy Simulation

### 4.1 Solar PV Model
```
P_solar = capacity × η × (GHI / 1000) × temp_derating
temp_derating = 1 + γ × (T_cell - 25°C)     (γ = -0.004/°C typical)
P_solar = 0  during polar night (GHI = 0)
```

### 4.2 Wind Turbine Model
Piecewise power curve:
- Below cut-in (3 m/s): P = 0
- Cut-in to rated (3–12 m/s): P = capacity × ((v - v_cutin)/(v_rated - v_cutin))³
- Rated to cut-out (12–25 m/s): P = capacity
- Above cut-out (>25 m/s): P = 0

Air density correction: ρ = ρ₀ × (T₀/T) × (P/P₀)

### 4.3 Station Load Model
```
P_load = P_base + P_heating + P_research + noise
P_heating = k × max(0, T_threshold - T)     (temperature-dependent heating)
P_research = P_research_peak × activity_schedule
P_critical = P_load × critical_fraction       (40% unsheddable)
```

### 4.4 Battery Model
```
SOC(t+1) = SOC(t) + η_c × P_charge / C  -  P_discharge / (η_d × C)
SOC ∈ [SOC_min, SOC_max] = [0.10, 0.90]
Cold derating: C_eff = C × max(0.5, 1 - 0.005 × max(0, -20 - T))
```

### 4.5 Diesel Generator Model
```
Fuel(t) = a + b × P_diesel     (L/h)    [a=5.0, b=0.25]
P_diesel ∈ [P_min_load, P_capacity] when ON
P_diesel = 0 when OFF
```

## 5. AI Forecasting

### 5.1 Model Architecture
- **Algorithm**: XGBoost (gradient boosted trees)
- **Targets**: 24-hour ahead load, solar generation, wind generation
- **Features**: Weather (temperature, wind, solar radiation, humidity), temporal (hour, day, month, season), lag features, rolling statistics

### 5.2 Training Strategy
- Chronological split: 70% train / 15% validation / 15% test
- No data leakage: future data never used for prediction
- Hyperparameters: n_estimators=300, max_depth=6, learning_rate=0.05

### 5.3 Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

## 6. MILP Optimization

### 6.1 Formulation
Mixed-Integer Linear Program solved with Pyomo + GLPK (with heuristic fallback).

**Decision Variables**: solar_use, wind_use, battery_charge, battery_discharge, diesel_gen (continuous); diesel_on, diesel_startup (binary)

**Objective**: Minimize total operational cost
```
min Σ [fuel_cost(t) + startup_cost(t) + maintenance_cost(t) + degradation_cost(t) + unmet_penalty(t)]
```

**Constraints**:
1. Power balance: generation = demand
2. Renewable availability limits
3. Battery SOC dynamics and bounds
4. Diesel min/max operating load
5. Critical load protection
6. Generator startup tracking

## 7. Reinforcement Learning

### 7.1 Environment (Gymnasium)
- **State**: [load, solar, wind, SOC, diesel_status, temperature, hour, season, polar_night]
- **Action**: [battery_power, diesel_power, diesel_on_off] ∈ [-1, 1]³
- **Reward**: -(fuel_cost + startup + maintenance + degradation + unmet_penalty) + renewable_bonus

### 7.2 Agents
- **PPO** (Proximal Policy Optimization): On-policy, stable training
- **SAC** (Soft Actor-Critic): Off-policy, entropy-regularized, handles continuous actions

### 7.3 Safety Layer
Hard constraint projection applied after every RL action:
- SOC bounds enforcement
- Diesel capacity limits
- Critical load guarantee (forces diesel ON if needed)

## 8. Dispatch Controller

Priority hierarchy: **Renewable → Battery → Diesel**

Modes:
1. **Rule-Based**: Simple priority dispatch (baseline)
2. **MILP**: Pre-computed optimal schedule
3. **RL (PPO/SAC)**: Adaptive learned policy + safety layer

## 9. Evaluation

### Metrics
- **Fuel Savings**: % reduction vs diesel-only baseline
- **Renewable Utilization**: % of available renewable energy used
- **Reliability**: % of load served (target: 99.9%+)
- **Cost**: Total operational cost (fuel + maintenance + degradation)
- **LOLP**: Loss of Load Probability
- **CO₂ Reduction**: Based on 2.68 kg CO₂/L diesel

### Comparison
Three controllers (Rule-Based, PPO, SAC) evaluated on identical test data with identical safety constraints.
