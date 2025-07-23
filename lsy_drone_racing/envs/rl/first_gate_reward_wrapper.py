import gymnasium as gym
import numpy as np
from lsy_drone_racing.envs.gate_logging_callback import GateLoggingCallback

class FirstGateRewardWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for first-gate curriculum:
    - +10 reward and terminate when the first gate is passed (target_gate == 1)
    - -10 reward and terminate if the drone crashes before passing the first gate
    - -0.01 per step otherwise
    - Adds 'curriculum_terminated' to info when wrapper terminates the episode
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Extract target_gate and crash info robustly
        target_gate = obs["target_gate"] if isinstance(obs, dict) else obs[-1]
        # Handle both array and scalar
        if isinstance(target_gate, np.ndarray):
            target_gate = target_gate.item() if target_gate.size == 1 else target_gate
        # Try to detect crash (terminated by core env before passing first gate)
        crashed = False
        if isinstance(info, dict):
            # If info contains a 'crash' or 'disabled' key, use it; else fallback to terminated
            crashed = info.get("crash", False) or info.get("disabled", False)
        if not crashed:
            # Fallback: if terminated and not passed first gate, treat as crash
            crashed = terminated and target_gate == 0
        # Curriculum logic
        curriculum_terminated = False
        if target_gate == 1:
            reward = 10.0
            terminated = True
            curriculum_terminated = True
        elif crashed and target_gate == 0:
            reward = -10.0
            terminated = True
            curriculum_terminated = True
        else:
            reward = -0.01
        if isinstance(info, dict):
            info = dict(info)  # Copy to avoid mutating original
            if curriculum_terminated:
                info["curriculum_terminated"] = True
        if terminated or truncated:
            first_gate_pos = obs["gates_pos"][0] if isinstance(obs, dict) and "gates_pos" in obs else None
            drone_pos = obs["pos"] if isinstance(obs, dict) and "pos" in obs else None
            print(f"Episode ended! Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}, Target gate: {target_gate}, Drone pos: {drone_pos}")
        return obs, reward, terminated, truncated, info 