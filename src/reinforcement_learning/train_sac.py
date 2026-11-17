"""
PolarGrid AI — Train SAC RL Controller
Trains a Soft Actor-Critic (SAC) agent to manage energy dispatch.
Uses diverse multi-season training data and normalized observations.
"""

import pandas as pd
import yaml
import logging
import os
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from src.reinforcement_learning.environment import PolarGridEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RL_MODELS_DIR = PROJECT_ROOT / "models" / "rl"

def train_sac_agent(total_timesteps: int = 25000):
    logger.info("Training SAC Reinforcement Learning Controller across multi-year data...")
    
    # Load dataset - use 85% train split covering all seasons
    master_df = pd.read_csv(DATA_PROCESSED / "master_dataset.csv")
    n_train = int(len(master_df) * 0.85)
    train_df = master_df.iloc[:n_train].copy().reset_index(drop=True)
    
    # Instantiate environment with random 168h episode starts
    env = PolarGridEnv(df_weather=train_df, episode_length=168, random_start=True)
    env = Monitor(env)
    
    # Initialize SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=20000,
        learning_starts=1000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        seed=42
    )
    
    logger.info(f"Starting SAC training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save SAC agent
    model_path = RL_MODELS_DIR / "sac_polargrid"
    model.save(model_path)
    logger.info(f"SAC agent model saved successfully at {model_path}")
    
    return model

if __name__ == "__main__":
    train_sac_agent()
