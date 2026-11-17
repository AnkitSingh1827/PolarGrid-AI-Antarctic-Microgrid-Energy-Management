"""
PolarGrid AI — RL Training and Evaluation Script
Train PPO/SAC agents and evaluate against rule-based baseline.
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RL Runner")


def run_rl_pipeline(timesteps: int = 15000):
    """Train RL agents and evaluate all controllers."""
    logger.info("=" * 50)
    logger.info("PolarGrid AI — RL Pipeline")
    logger.info("=" * 50)
    
    # Train PPO
    logger.info("\n--- Training PPO Agent ---")
    from src.reinforcement_learning.train_ppo import train_ppo_agent
    train_ppo_agent(total_timesteps=timesteps)
    
    # Train SAC
    logger.info("\n--- Training SAC Agent ---")
    from src.reinforcement_learning.train_sac import train_sac_agent
    train_sac_agent(total_timesteps=timesteps)
    
    # Evaluate all controllers
    logger.info("\n--- Evaluating All Controllers ---")
    from src.reinforcement_learning.evaluate_rl import evaluate_all_controllers
    comparison = evaluate_all_controllers()
    
    logger.info("\n✅ RL pipeline completed.")
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PolarGrid AI RL Pipeline")
    parser.add_argument("--timesteps", type=int, default=15000, help="Training timesteps")
    args = parser.parse_args()
    
    run_rl_pipeline(timesteps=args.timesteps)
