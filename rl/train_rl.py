import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.envs.rl.rl_env_wrapper import FlattenedObsWrapper, MaskYawActionWrapper
import importlib
from stable_baselines3.common.callbacks import CheckpointCallback
from lsy_drone_racing.envs.rl.center_line_reward_wrapper import CenterLineRewardWrapper
from lsy_drone_racing.envs.rl.gate_logging_callback import GateLoggingCallback
from datetime import datetime

# --- Naming ---
MODEL_NAME = "ppo_att_center"  # Change this to describe your experiment

# --- Config ---
CONFIG_PATH = "config/level0.toml"
ENV_ID = "DroneRacing-v0"
TOTAL_TIMESTEPS = 1_000_000  # Adjust as needed
MODEL_PATH = f"{MODEL_NAME}.zip"
N_ENVS = 1  # Number of parallel environments for GPU training

# --- Reward config ---
STEP_PENALTY = -0.01
DISTANCE_PENALTY_COEFF = 0.1  # Positive: penalizes distance from gate center

# --- Load config and environment ---
config = load_config(Path(CONFIG_PATH))

# Dynamically import the env registration (if not already imported)
importlib.import_module("lsy_drone_racing.envs")

def make_env():
    def _init():
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
        env = CenterLineRewardWrapper(env, step_penalty=STEP_PENALTY, distance_penalty_coeff=DISTANCE_PENALTY_COEFF)
        if config.env.control_mode == 'attitude':
            env = MaskYawActionWrapper(env)
        env = FlattenedObsWrapper(env)
        return env
    return _init

if N_ENVS > 1:
    vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
else:
    vec_env = DummyVecEnv([make_env()])

# --- Model loading/continuing logic ---
original_model_path = MODEL_PATH
original_model_name = MODEL_NAME
original_tb_log = f"./{MODEL_NAME}_tensorboard/"
original_ckpt_path = f'./checkpoints/{MODEL_NAME}/'

if os.path.exists(MODEL_PATH):
    print(f"Model file {MODEL_PATH} already exists.")
    choice = input("Do you want to [c]ontinue training or [o]verride? [c/o]: ").strip().lower()
    if choice == 'c':
        print(f"Continuing training from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH, env=vec_env, tensorboard_log=original_tb_log)
        tb_log = original_tb_log
        ckpt_path = original_ckpt_path
        model_save_path = original_model_path
        model_name = original_model_name
    elif choice == 'o':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{MODEL_NAME}_{timestamp}"
        model_save_path = f"{model_name}.zip"
        tb_log = f"./{model_name}_tensorboard/"
        ckpt_path = f'./checkpoints/{model_name}/'
        print(f"Overriding: starting new training as {model_name}...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tb_log,
        )
    else:
        print("Invalid choice. Exiting.")
        exit(1)
else:
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=original_tb_log,
    )
    tb_log = original_tb_log
    ckpt_path = original_ckpt_path
    model_save_path = original_model_path
    model_name = original_model_name

# --- Checkpoint callback ---
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,  # Save every 10,000 steps
    save_path=ckpt_path,
    name_prefix=model_name
)

callback = [checkpoint_callback, GateLoggingCallback()]

print(f"\n==== Training: {model_name} ====")
try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
except KeyboardInterrupt:
    print("Training interrupted! Saving model...")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# --- Save the trained model (if not already saved) ---
model.save(model_save_path)
print(f"Model saved to {model_save_path}") 