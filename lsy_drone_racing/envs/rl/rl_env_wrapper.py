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

class MaskYawActionWrapper(gym.ActionWrapper):
    """
    If the action space is 4D (attitude control), mask the yaw action (index 3) to zero.
    """
    def action(self, action):
        if hasattr(self.env, 'action_space') and isinstance(self.env.action_space, spaces.Box):
            if self.env.action_space.shape == (4,):
                action = np.array(action, dtype=np.float32)
                action[3] = 0.0  # Mask yaw
        return action 