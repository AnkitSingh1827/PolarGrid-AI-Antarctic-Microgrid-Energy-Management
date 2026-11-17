"""
PolarGrid AI — Model Training Script
Train all forecasting and RL models.
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Model Training")


def train_all_models():
    """Train all forecasting and RL models."""
    logger.info("=" * 50)
    logger.info("PolarGrid AI — Model Training")
    logger.info("=" * 50)
    
    # Forecasting Models
    logger.info("\n--- Training XGBoost Forecasting Models ---")
    from src.forecasting.evaluation import evaluate_all_forecasts
    metrics = evaluate_all_forecasts()
    
    for model, m in metrics.items():
        logger.info(f"  {model}: R²={m['R2']:.4f}, MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}")
    
    # RL Models
    logger.info("\n--- Training PPO Controller ---")
    from src.reinforcement_learning.train_ppo import train_ppo_agent
    train_ppo_agent(total_timesteps=15000)
    
    logger.info("\n--- Training SAC Controller ---")
    from src.reinforcement_learning.train_sac import train_sac_agent
    train_sac_agent(total_timesteps=15000)
    
    logger.info("\n✅ All models trained successfully.")


if __name__ == "__main__":
    train_all_models()
