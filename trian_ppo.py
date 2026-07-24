import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from environment.warehouse_dataset_env import WarehouseDatasetEnvironment
from planners.ppo import PPOTrainer


############################################################
# Dataset Path
############################################################

DATASET_PATH = "/content/drive/MyDrive/Dataset/bin"


############################################################
# Create Environment
############################################################

def make_env():

    env = WarehouseDatasetEnvironment(
        dataset_path=DATASET_PATH
    )

    env = Monitor(env)

    return env


train_env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])


############################################################
# Directories
############################################################

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


############################################################
# Optional Callbacks
############################################################

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo_warehouse"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)


############################################################
# PPO Trainer
############################################################

trainer = PPOTrainer(train_env)


############################################################
# Train
############################################################

trainer.train(
    total_timesteps=500000,
    save_path="models/ppo_warehouse_final"
)

print("\nTraining Complete!")