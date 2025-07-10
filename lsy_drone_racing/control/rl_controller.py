import numpy as np
from lsy_drone_racing.control import Controller
from stable_baselines3 import PPO
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_PATH = Path(__file__).parents[2] / "ppo_drone_racing.zip"  # Adjust if needed

class RLController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self.control_mode = config.env.control_mode
        # Load the trained PPO model
        if MODEL_PATH.exists():
            self.model = PPO.load(str(MODEL_PATH))
        else:
            self.model = None  # Fallback to random actions if model not found

    def flatten_obs(self, obs):
        # Flatten the observation dict to a 1D numpy array for the policy
        flat = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, np.ndarray):
                flat.append(v.flatten())
            else:
                flat.append(np.array([v]))
        return np.concatenate(flat, axis=0).astype(np.float32)

    def compute_control(self, obs, info=None):
        obs_vec = self.flatten_obs(obs)
        if self.model is None:
            # Random action for testing; replace with model(obs_vec)
            if self.control_mode == "state":
                return np.random.uniform(-1, 1, size=(13,)).astype(np.float32)
            else:
                return np.random.uniform(-1, 1, size=(4,)).astype(np.float32)
        else:
            # Use the trained PPO model to predict the action
            action, _ = self.model.predict(obs_vec, deterministic=True)
            return action.astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        # Optionally update model or log data here
        return terminated or truncated

    def episode_callback(self):
        # Optionally save model or statistics here
        pass 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            print(f"Episode ended. Reward: {reward}, Info: {info}")
        return obs, reward, terminated, truncated, info