import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from lsy_drone_racing.utils import load_config
from rl.rl_env_wrapper import FlattenedObsWrapper
import importlib
from stable_baselines3.common.callbacks import CheckpointCallback
from rl.first_gate_reward_wrapper import FirstGateRewardWrapper
from rl.distance_to_gate_reward_wrapper import DistanceToGateRewardWrapper
from rl.gate_logging_callback import GateLoggingCallback

# --- Config ---
CONFIG_PATH = "config/level0.toml"
ENV_ID = "DroneRacing-v0"
CONTINUE_TIMESTEPS = 500_000  # Number of additional steps
MODEL_PATH = "ppo_drone_racing.zip"

# --- Load config and environment ---
config = load_config(Path(CONFIG_PATH))

# Dynamically import the env registration (if not already imported)
importlib.import_module("lsy_drone_racing.envs")

def make_env():
    env = gym.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = FirstGateRewardWrapper(env)
    env = DistanceToGateRewardWrapper(env)
    env = FlattenedObsWrapper(env)
    return env

vec_env = DummyVecEnv([make_env])

# --- Load the trained PPO model ---
model = PPO.load(MODEL_PATH, env=vec_env)

# --- Checkpoint callback ---
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,  # Save every 10,000 steps
    save_path='./checkpoints/',
    name_prefix='ppo_drone_racing'
)
callback = [checkpoint_callback, GateLoggingCallback()]

# --- Continue training ---
try:
    model.learn(total_timesteps=CONTINUE_TIMESTEPS, callback=callback)
except KeyboardInterrupt:
    print("Training interrupted! Saving model...")
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# --- Save the trained model (if not already saved) ---
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}") 