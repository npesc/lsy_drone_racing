import gymnasium as gym
import numpy as np

class CenterLineRewardWrapper(gym.Wrapper):
    """
    Reward wrapper for drone racing:
    - Negative reward proportional to distance from drone to center of current target gate
    - +10 reward for passing a gate
    - -10 reward and terminate if the drone crashes
    - Small step penalty (configurable)
    - Works for all gates (not just the first)
    """
    def __init__(self, env, step_penalty=-0.01, distance_penalty_coeff=-0.1):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.distance_penalty_coeff = distance_penalty_coeff
        self._last_target_gate = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_target_gate = obs["target_gate"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        target_gate = obs["target_gate"]
        drone_pos = obs["pos"]
        gates_pos = obs["gates_pos"]
        n_gates = len(gates_pos)
        # Clamp target_gate to valid range
        if target_gate < 0 or target_gate >= n_gates:
            # All gates passed or invalid, no distance penalty
            distance = 0.0
        else:
            gate_pos = gates_pos[target_gate]
            distance = np.linalg.norm(drone_pos - gate_pos)
        # Distance penalty
        reward -= self.distance_penalty_coeff * distance
        # Step penalty
        reward += self.step_penalty
        # Gate passing logic: reward +10 for passing a gate
        if self._last_target_gate is not None and target_gate > self._last_target_gate:
            reward += 10.0
        # Crash detection: if 'crash' or 'disabled' in info, penalize and terminate
        crashed = False
        if isinstance(info, dict):
            crashed = info.get("crash", False) or info.get("disabled", False)
        if not crashed:
            # Fallback: if terminated and not all gates passed, treat as crash
            crashed = terminated and (target_gate != n_gates)
        if crashed:
            reward -= 10.0
            terminated = True
        self._last_target_gate = target_gate
        return obs, reward, terminated, truncated, info 