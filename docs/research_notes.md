# PolarGrid AI — Research Notes

## Project Context
- **Competition**: Smart India Hackathon (SIH) 2026
- **Problem Statement**: PS 26061 — AI-Driven Smart Energy Management System for Polar Research Stations
- **Ministry**: Ministry of Earth Sciences (MoES) / NCPOR
- **Station**: Indian Antarctic Research Station Maitri, Schirmacher Oasis

## Key Design Decisions

### 1. Weather Data Source
**Decision**: Use ERA5 reanalysis as primary weather data, validated against NCPOR Maitri AWS observations.

**Rationale**: ERA5 provides continuous, gapless hourly data from 2012-2025 at the exact Maitri coordinates. NCPOR AWS has gaps and only covers 2012-2016. ERA5 is produced by ECMWF assimilation and is the gold standard for climate research.

### 2. Energy Data: Simulation vs Real
**Decision**: Use physics-informed simulation for all energy variables (load, solar, wind, battery, diesel).

**Rationale**: Actual Maitri energy consumption data is not publicly available. Australian Antarctic Division data (Davis, Casey, Mawson) provides reference patterns but is from different stations with different climates and infrastructure. Rather than misrepresent foreign data as Maitri data, we build transparent physics models calibrated to Antarctic conditions and clearly mark all energy data as simulated.

### 3. Forecasting: XGBoost over LSTM
**Decision**: XGBoost for all forecasting tasks.

**Rationale**: 
- 14 years of hourly data (122K samples) is substantial but not the millions needed for LSTM to clearly outperform tree methods
- XGBoost handles mixed feature types naturally (categorical temporal + continuous weather)
- Much faster training and inference (critical for real-time deployment)
- More interpretable feature importances
- Empirically achieves R² > 0.95 on all targets

### 4. MILP Solver: GLPK with Heuristic Fallback
**Decision**: Use GLPK (open-source) as MILP solver with automatic fallback to a rule-based heuristic.

**Rationale**: GLPK is freely available and handles the 24-hour energy scheduling problem efficiently. However, solver installation can be unreliable across platforms, so a heuristic fallback ensures the system always produces valid dispatch schedules even without GLPK.

### 5. RL Training Budget
**Decision**: Default to 15,000 training steps (rapid proof-of-concept).

**Rationale**: Full RL training on 14 years × 8760 hours would take 30-60+ minutes. For the prototype, shorter training demonstrates the architecture while remaining computationally tractable. Production deployment would use 100K+ steps.

### 6. Safety Layer: Hard Constraint Projection
**Decision**: Every RL/MILP action passes through a hard safety constraint layer before execution.

**Rationale**: In safety-critical energy systems, we cannot trust a learned policy to always respect physical constraints. The safety layer guarantees:
- Battery SOC stays within [10%, 90%]
- Diesel operates within [30%, 100%] of capacity when ON
- Critical load (40% of total) is NEVER shed
- Generator forced ON if critical load is at risk

## References

1. ERA5 hourly data — Hersbach et al. (2020), Copernicus Climate Data Store
2. NCPOR Maitri AWS data — National Centre for Polar and Ocean Research, India
3. Pyomo — Hart et al. (2017), "Pyomo — Optimization Modeling in Python"
4. Stable-Baselines3 — Raffin et al. (2021), JMLR
5. XGBoost — Chen & Guestrin (2016), KDD
6. PPO — Schulman et al. (2017), arXiv:1707.06347
7. SAC — Haarnoja et al. (2018), ICML

## Known Limitations

1. **No real Maitri energy data**: All energy variables are physics-informed simulations. Actual deployment would require real metering data.
2. **ERA5 biases**: Reanalysis data has known biases in polar regions, particularly for cloud cover and precipitation. We validate against NCPOR AWS where overlap exists.
3. **Simplified load model**: Real station load is more complex (multiple generators, building management systems, research equipment schedules).
4. **No network effects**: We model a single bus system. Real stations may have multiple microgrids.
5. **RL generalization**: Short training may result in suboptimal RL policies. More training and reward shaping could improve performance.
