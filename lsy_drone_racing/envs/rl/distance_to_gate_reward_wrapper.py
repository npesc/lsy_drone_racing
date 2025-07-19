import gymnasium as gym
import numpy as np

class DistanceToGateRewardWrapper(gym.Wrapper):
    """
    Adds a negative reward proportional to the distance from the drone to the first gate.
    reward -= 0.1 * distance_to_first_gate
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Extract drone position and first gate position
        drone_pos = obs["pos"]
        first_gate_pos = obs["gates_pos"][0]
        # Compute Euclidean distance
        distance = np.linalg.norm(drone_pos - first_gate_pos)
        # Subtract distance penalty
        reward -= 0.1 * distance
        return obs, reward, terminated, truncated, info 