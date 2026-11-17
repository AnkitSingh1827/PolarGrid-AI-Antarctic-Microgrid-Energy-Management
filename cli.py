"""
PolarGrid AI — Interactive Command Line Interface (CLI)
Convenient terminal utility for station operators and developers.
"""

import argparse
import sys
import json
import io
import pandas as pd
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def status_command(args):
    """Print microgrid real-time status summary."""
    master_path = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
    if not master_path.exists():
        print("Error: master_dataset.csv not found. Run pipeline first.")
        return
    
    df = pd.read_csv(master_path).tail(24)
    last = df.iloc[-1]
    
    print("\n" + "=" * 60)
    print("POLARGRID AI — MAITRI STATION MICROGRID STATUS")
    print("=" * 60)
    print(f"Timestamp:             {last.get('timestamp', 'N/A')}")
    print(f"Ambient Temperature:   {last.get('temperature', 0):.1f} °C")
    print(f"Wind Speed:            {last.get('wind_speed', 0):.1f} m/s")
    print(f"Current Demand:        {last.get('load_kw', 0):.1f} kW (Critical: {last.get('critical_load_kw', 0):.1f} kW)")
    print(f"Solar PV Output:       {last.get('solar_generation_kw', 0):.1f} kW")
    print(f"Wind Turbine Output:   {last.get('wind_generation_kw', 0):.1f} kW")
    print(f"Battery Bank SOC:      {last.get('battery_soc', 0)*100:.1f} %")
    diesel_val = last.get('diesel_generation_kw', 0)
    print(f"Diesel Generator:      {'RUNNING (' + str(round(diesel_val, 1)) + ' kW)' if diesel_val > 0.1 else 'OFF (Standby)'}")
    print("=" * 60 + "\n")


def forecast_command(args):
    """Print forecast metrics and next 24h predictions."""
    metrics_path = PROJECT_ROOT / "results" / "forecasts" / "forecast_metrics.json"
    preds_path = PROJECT_ROOT / "results" / "forecasts" / "forecast_predictions.csv"
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print("\n" + "=" * 60)
        print("AI FORECASTING MODEL BENCHMARKS")
        print("=" * 60)
        for target, m in metrics.items():
            print(f"{target.upper():10s} | R²: {m['R2']:.4f} | MAE: {m['MAE']:.3f} kW | RMSE: {m['RMSE']:.3f} kW | MAPE: {m['MAPE']:.2f}%")
        print("=" * 60)
        
    if preds_path.exists():
        preds = pd.read_csv(preds_path).head(args.hours)
        print(f"\nNext {args.hours} Hours Predicted Demand & Generation:")
        print(preds[['load_pred', 'solar_pred', 'wind_pred']].round(2).to_string())
        print()


def dispatch_command(args):
    """Execute single-step safe dispatch."""
    from src.dispatch.controller import DispatchController
    ctrl = DispatchController(mode=args.mode)
    ctrl.soc = args.soc
    
    result = ctrl.dispatch_step(
        load_kw=args.load,
        critical_load_kw=args.load * 0.4,
        solar_gen_kw=args.solar,
        wind_gen_kw=args.wind,
        temperature=args.temp,
    )
    
    print("\n" + "=" * 60)
    print(f"DISPATCH DECISION ({args.mode.upper()})")
    print("=" * 60)
    b_power = result['battery_power_kw']
    state_str = 'Charging' if b_power > 0 else 'Discharging' if b_power < 0 else 'Idle'
    print(f"Battery Power:     {b_power:+.1f} kW ({state_str})")
    print(f"Diesel Output:     {result['diesel_generation_kw']:.1f} kW ({'ACTIVE' if result['diesel_status'] == 1 else 'OFF'})")
    print(f"Unmet Load:        {result['unmet_load_kw']:.1f} kW")
    print(f"Curtailed Power:   {result['curtailment_kw']:.1f} kW")
    print(f"Resulting SOC:     {result['battery_soc']*100:.1f} %")
    print("=" * 60 + "\n")


def benchmark_command(args):
    """Print controller comparison summary."""
    comp_path = PROJECT_ROOT / "results" / "comparison" / "controller_comparison.csv"
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        print("\n" + "=" * 80)
        print("CONTROLLER PERFORMANCE BENCHMARK MATRIX")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80 + "\n")
    else:
        print("Running controller benchmark...")
        from src.reinforcement_learning.evaluate_rl import evaluate_all_controllers
        evaluate_all_controllers()


def main():
    parser = argparse.ArgumentParser(description="PolarGrid AI — Smart Energy Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # status
    subparsers.add_parser("status", help="Print live microgrid status")
    
    # forecast
    fc_parser = subparsers.add_parser("forecast", help="Display forecasting model metrics and predictions")
    fc_parser.add_argument("--hours", type=int, default=12, help="Number of hours to display (default: 12)")
    
    # dispatch
    dp_parser = subparsers.add_parser("dispatch", help="Calculate optimal safe dispatch for specific condition")
    dp_parser.add_argument("--load", type=float, default=65.0, help="Station load demand in kW")
    dp_parser.add_argument("--solar", type=float, default=20.0, help="Solar PV generation in kW")
    dp_parser.add_argument("--wind", type=float, default=30.0, help="Wind turbine generation in kW")
    dp_parser.add_argument("--soc", type=float, default=0.5, help="Current battery state of charge (0 to 1)")
    dp_parser.add_argument("--temp", type=float, default=-15.0, help="Ambient temperature in °C")
    dp_parser.add_argument("--mode", type=str, default="rule_based", choices=["rule_based", "rl_ppo", "rl_sac"], help="Controller mode")
    
    # benchmark
    subparsers.add_parser("benchmark", help="Display controller comparison table")
    
    args = parser.parse_args()
    
    if args.command == "status":
        status_command(args)
    elif args.command == "forecast":
        forecast_command(args)
    elif args.command == "dispatch":
        dispatch_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
