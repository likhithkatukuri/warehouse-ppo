import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from environment.warehouse_dataset_env import WarehouseDatasetEnvironment


############################################################
# Create Environment
############################################################

def make_env():

    env = WarehouseDatasetEnvironment()

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
# Callbacks
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
# PPO Model
############################################################

model = PPO(

    policy="CnnPolicy",

    env=train_env,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=64,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    vf_coef=0.5,

    max_grad_norm=0.5,

    tensorboard_log="./logs/",

    verbose=1,

    device="auto"

)


############################################################
# Train
############################################################

TOTAL_TIMESTEPS = 500000

model.learn(

    total_timesteps=TOTAL_TIMESTEPS,

    callback=[checkpoint_callback, eval_callback],

    progress_bar=True

)


############################################################
# Save Final Model
############################################################

model.save("models/ppo_warehouse_final")

print("\nTraining Complete!")