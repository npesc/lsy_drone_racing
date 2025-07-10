from stable_baselines3.common.callbacks import BaseCallback

class GateLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.gates_passed = []
        self.episode_rewards = []
        self._reward_buffer = None  # Will be initialized on first call

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        n_envs = len(dones)
        # Initialize reward buffer if needed
        if self._reward_buffer is None:
            self._reward_buffer = [0.0 for _ in range(n_envs)]
        for i, (info, done, reward) in enumerate(zip(infos, dones, rewards)):
            self._reward_buffer[i] += reward
            if done and info is not None:
                # Log gates passed
                gates_passed = info.get("gates_passed")
                if gates_passed is None:
                    target_gate = info.get("target_gate", 0)
                    if target_gate == -1:
                        gates_passed = info.get("n_gates", 0)
                    else:
                        gates_passed = target_gate
                self.gates_passed.append(gates_passed)
                self.logger.record("rollout/gates_passed", gates_passed)
                # Log episode reward
                ep_rew = self._reward_buffer[i]
                self.episode_rewards.append(ep_rew)
                self.logger.record("rollout/ep_rew", ep_rew)
                self._reward_buffer[i] = 0.0  # Reset for next episode
        return True 