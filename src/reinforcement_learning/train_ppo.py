"""
PolarGrid AI — Train PPO RL Controller
Trains a Proximal Policy Optimization agent to manage energy dispatch.
Uses diverse multi-season training data and normalized observations.
"""

import pandas as pd
import yaml
import logging
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.reinforcement_learning.environment import PolarGridEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RL_MODELS_DIR = PROJECT_ROOT / "models" / "rl"

RL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_ppo_agent(total_timesteps: int = 30000):
    logger.info("Training PPO Reinforcement Learning Controller across multi-year data...")
    
    # Load dataset - use 85% train split covering all seasons across multiple years
    master_df = pd.read_csv(DATA_PROCESSED / "master_dataset.csv")
    n_train = int(len(master_df) * 0.85)
    train_df = master_df.iloc[:n_train].copy().reset_index(drop=True)
    
    logger.info(f"PPO training dataset size: {len(train_df)} hours ({len(train_df)/8760:.1f} years)")
    
    # Instantiate environment with random 168h (1 week) episode starts
    env = PolarGridEnv(df_weather=train_df, episode_length=168, random_start=True)
    env = Monitor(env)
    
    # Initialize PPO model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=42
    )
    
    logger.info(f"Starting PPO training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save PPO agent
    model_path = RL_MODELS_DIR / "ppo_polargrid"
    model.save(model_path)
    logger.info(f"PPO agent model saved successfully at {model_path}")
    
    return model

if __name__ == "__main__":
    train_ppo_agent()
