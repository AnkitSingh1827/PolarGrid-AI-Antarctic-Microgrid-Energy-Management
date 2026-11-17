"""
PolarGrid AI — Full Pipeline Runner
Single command to run: data → simulation → forecasting → optimization → evaluation
"""

import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PolarGrid Pipeline")


def run_full_pipeline():
    """Execute the complete PolarGrid AI pipeline end-to-end."""
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("PolarGrid AI — Full Pipeline Execution")
    logger.info("=" * 70)
    
    # Phase 1: Data Pipeline
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: Data Pipeline")
    logger.info("=" * 50)
    try:
        from src.data.run_pipeline_data import run_data_pipeline
        run_data_pipeline()
        logger.info("✅ Data pipeline completed.")
    except Exception as e:
        logger.warning(f"⚠️ Data pipeline skipped (data may already exist): {e}")
    
    # Phase 2: Simulation
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: Physics Simulation")
    logger.info("=" * 50)
    try:
        from src.simulation.run_simulation import run_all_simulations
        run_all_simulations()
        logger.info("✅ Simulation completed.")
    except Exception as e:
        logger.warning(f"⚠️ Simulation skipped (results may already exist): {e}")
    
    # Phase 3: Forecasting
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: AI Forecasting (XGBoost)")
    logger.info("=" * 50)
    try:
        from src.forecasting.evaluation import evaluate_all_forecasts
        metrics = evaluate_all_forecasts()
        logger.info(f"✅ Forecasting completed. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"❌ Forecasting failed: {e}")
        raise
    
    # Phase 4: MILP Optimization
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: MILP Optimization")
    logger.info("=" * 50)
    try:
        import pandas as pd
        import numpy as np
        import yaml
        import json
        
        from src.optimization.milp_model import solve_milp
        
        with open(PROJECT_ROOT / "configs" / "battery_config.yaml") as f:
            b = yaml.safe_load(f)["battery"]
        with open(PROJECT_ROOT / "configs" / "diesel_config.yaml") as f:
            d = yaml.safe_load(f)["diesel"]
        params = {"battery": b, "diesel": d}
        
        master = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "master_dataset.csv")
        # Run optimization on a representative 24-hour window
        sample = master.iloc[1000:1024]
        
        schedule_df, solved = solve_milp(
            sample["load_kw"].values,
            sample["critical_load_kw"].values,
            sample["solar_generation_kw"].values,
            sample["wind_generation_kw"].values,
            params,
        )
        
        results_dir = PROJECT_ROOT / "results" / "optimization"
        results_dir.mkdir(parents=True, exist_ok=True)
        schedule_df.to_csv(results_dir / "milp_schedule_24h.csv", index=False)
        
        logger.info(f"✅ MILP Optimization completed (Solved optimally: {solved}).")
    except Exception as e:
        logger.error(f"❌ MILP Optimization failed: {e}")
    
    # Phase 5: RL Training
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: RL Controller Training (PPO + SAC)")
    logger.info("=" * 50)
    try:
        from src.reinforcement_learning.train_ppo import train_ppo_agent
        train_ppo_agent(total_timesteps=15000)
        logger.info("✅ PPO training completed.")
    except Exception as e:
        logger.error(f"❌ PPO training failed: {e}")
    
    try:
        from src.reinforcement_learning.train_sac import train_sac_agent
        train_sac_agent(total_timesteps=15000)
        logger.info("✅ SAC training completed.")
    except Exception as e:
        logger.error(f"❌ SAC training failed: {e}")
    
    # Phase 6: RL Evaluation & Comparison
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 6: Controller Evaluation & Comparison")
    logger.info("=" * 50)
    try:
        from src.reinforcement_learning.evaluate_rl import evaluate_all_controllers
        comparison = evaluate_all_controllers()
        logger.info("✅ Controller evaluation completed.")
    except Exception as e:
        logger.error(f"❌ Controller evaluation failed: {e}")
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETE — Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)
    logger.info("Outputs:")
    logger.info(f"  📊 Data: data/processed/master_dataset.csv")
    logger.info(f"  🔬 Simulations: data/simulated/*.csv")
    logger.info(f"  🤖 Forecasts: results/forecasts/")
    logger.info(f"  ⚙️ Optimization: results/optimization/")
    logger.info(f"  🎮 RL Models: models/rl/")
    logger.info(f"  📈 Comparison: results/comparison/")


if __name__ == "__main__":
    run_full_pipeline()
