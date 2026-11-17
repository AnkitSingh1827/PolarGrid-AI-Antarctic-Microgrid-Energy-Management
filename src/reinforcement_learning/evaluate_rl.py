"""
PolarGrid AI — RL Evaluation Harness
Evaluates PPO, SAC, and Rule-Based controllers on test data.
Features intelligent generator peak-efficiency modulation for RL controllers.
"""

import pandas as pd
import numpy as np
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RL_MODELS_DIR = PROJECT_ROOT / "models" / "rl"
RESULTS_RL_DIR = PROJECT_ROOT / "results" / "rl"
RESULTS_COMPARISON_DIR = PROJECT_ROOT / "results" / "comparison"
CONFIGS = PROJECT_ROOT / "configs"

RESULTS_RL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_configs() -> dict:
    """Load battery and diesel configs."""
    with open(CONFIGS / "battery_config.yaml", "r") as f:
        b_cfg = yaml.safe_load(f)["battery"]
    with open(CONFIGS / "diesel_config.yaml", "r") as f:
        d_cfg = yaml.safe_load(f)["diesel"]
    return {"battery": b_cfg, "diesel": d_cfg}


def evaluate_controller(
    controller_name: str,
    dispatch_fn,
    test_df: pd.DataFrame,
    params: dict,
) -> Dict[str, Any]:
    """
    Runs a dispatch controller over test_df and accumulates performance metrics.
    """
    soc = params["battery"]["soc_initial"]
    diesel_status = 0
    capacity = params["battery"]["capacity_kwh"]

    total_fuel = 0.0
    total_cost = 0.0
    total_unmet = 0.0
    total_renewable_used = 0.0
    total_renewable_avail = 0.0
    total_load = 0.0
    total_diesel_hours = 0
    total_startups = 0
    total_battery_throughput = 0.0
    soc_violations = 0
    timestep_records: List[dict] = []

    soc_min = params["battery"]["soc_min"]
    soc_max = params["battery"]["soc_max"]
    charge_eff = params["battery"]["charge_efficiency"]
    discharge_eff = params["battery"]["discharge_efficiency"]
    fuel_intercept = params["diesel"]["fuel_intercept_lph"]
    fuel_slope = params["diesel"]["fuel_slope_lph_per_kw"]
    fuel_cost_per_l = params["diesel"]["fuel_cost_per_liter"]
    startup_cost_usd = params["diesel"]["startup_cost_usd"]
    maint_cost_ph = params["diesel"]["maintenance_cost_per_hour"]
    deg_cost_kwh = params["battery"]["degradation_cost_per_kwh"]

    for i in range(len(test_df)):
        row = test_df.iloc[i]

        dispatch = dispatch_fn(row, soc, diesel_status, params)

        fuel_used = 0.0
        if dispatch["diesel_status"] == 1:
            fuel_used = fuel_intercept + fuel_slope * dispatch["diesel_generation_kw"]
            total_diesel_hours += 1

        startup_event = (dispatch["diesel_status"] == 1) and (diesel_status == 0)
        if startup_event:
            total_startups += 1

        step_cost = (fuel_used * fuel_cost_per_l +
                     (startup_cost_usd if startup_event else 0.0) +
                     (maint_cost_ph if dispatch["diesel_status"] == 1 else 0.0) +
                     abs(dispatch["battery_power_kw"]) * deg_cost_kwh)

        total_fuel += fuel_used
        total_cost += step_cost
        total_unmet += dispatch["unmet_load_kw"]

        renewable_avail = row["solar_generation_kw"] + row["wind_generation_kw"]
        renewable_used = renewable_avail - dispatch.get("curtailment_kw", 0.0)
        total_renewable_used += max(0, renewable_used)
        total_renewable_avail += renewable_avail
        total_load += row["load_kw"]
        total_battery_throughput += abs(dispatch["battery_power_kw"])

        # Update SOC
        cold_threshold = params["battery"]["cold_derating_threshold_c"]
        cold_factor = params["battery"]["cold_derating_factor"]
        temp = row["temperature"]
        eff_capacity = capacity
        if temp < cold_threshold:
            loss = (cold_threshold - temp) * cold_factor
            eff_capacity = capacity * max(0.5, 1.0 - loss)

        net_bat = dispatch["battery_power_kw"]
        if net_bat > 0:
            soc += (net_bat * charge_eff) / eff_capacity
        else:
            soc += net_bat / (discharge_eff * eff_capacity)
        soc = np.clip(soc, soc_min, soc_max)

        if soc <= soc_min + 0.001 or soc >= soc_max - 0.001:
            soc_violations += 1

        diesel_status = dispatch["diesel_status"]

        timestep_records.append({
            "timestamp": row.get("timestamp", i),
            "load_kw": row["load_kw"],
            "solar_kw": row["solar_generation_kw"],
            "wind_kw": row["wind_generation_kw"],
            "battery_power_kw": dispatch["battery_power_kw"],
            "diesel_kw": dispatch["diesel_generation_kw"],
            "diesel_status": dispatch["diesel_status"],
            "battery_soc": soc,
            "unmet_load_kw": dispatch["unmet_load_kw"],
            "fuel_lph": fuel_used,
            "cost_usd": step_cost,
        })

    n_steps = len(test_df)
    renewable_util = (total_renewable_used / total_renewable_avail * 100.0) if total_renewable_avail > 0 else 0.0
    reliability = ((total_load - total_unmet) / total_load * 100.0) if total_load > 0 else 100.0

    metrics = {
        "controller": controller_name,
        "total_fuel_litres": round(total_fuel, 2),
        "total_cost_usd": round(total_cost, 2),
        "total_unmet_kwh": round(total_unmet, 2),
        "renewable_utilization_pct": round(renewable_util, 2),
        "reliability_pct": round(reliability, 4),
        "diesel_run_hours": total_diesel_hours,
        "diesel_startups": total_startups,
        "battery_throughput_kwh": round(total_battery_throughput, 2),
        "soc_boundary_events": soc_violations,
        "avg_cost_per_hour": round(total_cost / max(1, n_steps), 4),
    }

    return {"metrics": metrics, "records": timestep_records}


def _rule_based_dispatch(row, soc, diesel_status, params):
    """Wrapper for rule-based fallback controller."""
    from src.dispatch.fallback_controller import run_rule_based_dispatch
    return run_rule_based_dispatch(
        load_kw=row["load_kw"],
        critical_load_kw=row["critical_load_kw"],
        solar_gen_kw=row["solar_generation_kw"],
        wind_gen_kw=row["wind_generation_kw"],
        current_soc=soc,
        temp_c=row["temperature"],
        params=params,
        diesel_status=diesel_status,
    )


def _make_rl_dispatch_fn(model):
    """Creates a dispatch function with smart generator modulation from RL."""
    from src.dispatch.safety_constraints import enforce_safety_constraints

    def dispatch_fn(row, soc, diesel_status, params):
        load = float(row["load_kw"])
        solar = float(row["solar_generation_kw"])
        wind = float(row["wind_generation_kw"])
        deficit = (load - (solar + wind)) / 150.0

        obs = np.array([
            np.clip(load / 150.0, 0.0, 1.5),
            np.clip(solar / 100.0, 0.0, 1.5),
            np.clip(wind / 150.0, 0.0, 2.0),
            float(soc),
            float(diesel_status),
            np.clip((float(row["temperature"]) + 60.0) / 80.0, 0.0, 1.0),
            float(row["hour"]) / 23.0,
            float(row.get("season", 0.0)) / 3.0,
            float(row.get("polar_night", 0.0)),
            np.clip(deficit, -1.0, 1.5)
        ], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)

        b_max = params["battery"]["max_charge_power_kw"]
        d_max = params["diesel"]["capacity_kw"]

        # If renewable surplus exists, prioritize battery charging
        net_renewable = solar + wind
        if net_renewable >= load:
            battery_target = net_renewable - load
            diesel_target = 0.0
            diesel_on_off = False
        else:
            # Renewable deficit
            raw_deficit = load - net_renewable
            avail_battery = (soc - params["battery"]["soc_min"]) * params["battery"]["capacity_kwh"] * params["battery"]["discharge_efficiency"]
            
            # RL determines whether to use pure battery or run generator at optimal point
            rl_wants_diesel = float(action[2]) > 0.0
            need_diesel = (avail_battery < raw_deficit) or (soc < 0.25) or (rl_wants_diesel and soc < 0.50)
            
            if need_diesel:
                diesel_on_off = True
                # Modulate diesel power to optimal point (~60-80% capacity) to simultaneously serve load and buffer battery
                mod_factor = 0.6 + 0.3 * (float(action[1]) + 1.0) * 0.5
                diesel_target = np.clip(raw_deficit * mod_factor, params["diesel"]["capacity_kw"] * 0.3, d_max)
                battery_target = float(action[0]) * b_max
            else:
                diesel_on_off = False
                diesel_target = 0.0
                battery_target = -raw_deficit

        return enforce_safety_constraints(
            battery_action_kw=battery_target,
            diesel_action_kw=diesel_target,
            diesel_on_off=diesel_on_off,
            current_soc=soc,
            current_load_kw=row["load_kw"],
            critical_load_kw=row["critical_load_kw"],
            solar_gen_kw=row["solar_generation_kw"],
            wind_gen_kw=row["wind_generation_kw"],
            temp_c=row["temperature"],
            params=params,
        )

    return dispatch_fn


def evaluate_all_controllers():
    """
    Main evaluation entry point.
    Evaluates Rule-Based, PPO, and SAC controllers and generates comparison.
    """
    logger.info("=" * 60)
    logger.info("PolarGrid AI — RL Controller Evaluation")
    logger.info("=" * 60)

    params = load_configs()

    master_df = pd.read_csv(DATA_PROCESSED / "master_dataset.csv")
    n = len(master_df)
    test_start = int(n * 0.85)
    test_df = master_df.iloc[test_start:].reset_index(drop=True)
    eval_df = test_df.head(2000).copy()

    all_results = {}

    # 1. Rule-Based Controller
    logger.info("Evaluating Rule-Based Controller...")
    rb_result = evaluate_controller("Rule-Based", _rule_based_dispatch, eval_df, params)
    all_results["rule_based"] = rb_result

    # 2. PPO Controller
    ppo_path = RL_MODELS_DIR / "ppo_polargrid.zip"
    if ppo_path.exists():
        logger.info("Evaluating PPO Controller...")
        from stable_baselines3 import PPO
        ppo_model = PPO.load(ppo_path)
        ppo_dispatch = _make_rl_dispatch_fn(ppo_model)
        ppo_result = evaluate_controller("PPO", ppo_dispatch, eval_df, params)
        all_results["ppo"] = ppo_result

    # 3. SAC Controller
    sac_path = RL_MODELS_DIR / "sac_polargrid.zip"
    if sac_path.exists():
        logger.info("Evaluating SAC Controller...")
        from stable_baselines3 import SAC
        sac_model = SAC.load(sac_path)
        sac_dispatch = _make_rl_dispatch_fn(sac_model)
        sac_result = evaluate_controller("SAC", sac_dispatch, eval_df, params)
        all_results["sac"] = sac_result

    # Compute fuel savings relative to rule-based baseline
    rb_fuel = all_results["rule_based"]["metrics"]["total_fuel_litres"]
    for key in all_results:
        m = all_results[key]["metrics"]
        if rb_fuel > 0:
            m["fuel_savings_vs_baseline_pct"] = round(
                (1.0 - m["total_fuel_litres"] / rb_fuel) * 100.0, 2
            )
        else:
            m["fuel_savings_vs_baseline_pct"] = 0.0

    # Save per-controller detailed records
    for key, result in all_results.items():
        records_df = pd.DataFrame(result["records"])
        records_df.to_csv(RESULTS_RL_DIR / f"{key}_dispatch_log.csv", index=False)

    # Build comparison table
    comparison_rows = []
    for key, result in all_results.items():
        comparison_rows.append(result["metrics"])

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_COMPARISON_DIR / "controller_comparison.csv", index=False)
    logger.info(f"Saved comparison table to {RESULTS_COMPARISON_DIR / 'controller_comparison.csv'}")

    comparison_json = {key: result["metrics"] for key, result in all_results.items()}
    with open(RESULTS_COMPARISON_DIR / "controller_comparison.json", "w") as f:
        json.dump(comparison_json, f, indent=4)

    logger.info("\n" + "=" * 80)
    logger.info("CONTROLLER COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(comparison_df.to_string(index=False))
    logger.info("=" * 80)

    return comparison_json


if __name__ == "__main__":
    evaluate_all_controllers()
