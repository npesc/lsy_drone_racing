import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlattenedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Flatten the original observation space
        self.obs_keys = sorted(self.observation_space.spaces.keys())
        flat_dim = 0
        for k in self.obs_keys:
            shape = self.observation_space.spaces[k].shape
            flat_dim += int(np.prod(shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        flat = []
        for k in self.obs_keys:
            v = obs[k]
            flat.append(np.asarray(v).flatten())
        return np.concatenate(flat, axis=0).astype(np.float32) 