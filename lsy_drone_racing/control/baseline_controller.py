"""Controller that follows a pre-defined trajectory and performs on the fly replanning.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import minsnap_trajectories as ms
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(Controller):
    """Trajectory controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        self.t_total = 13
        self._tick = 0
        self._freq = config.env.freq
        self.obstacle_size = 0.05
        self._finished = False
        self.initial_pos = obs["pos"]
        self.gates_flags = [False] * len(obs["gates_visited"])
        self.obstacles_flags = [False] * len(obs["obstacles_visited"])
        self.index_gate = 0

        self.waypoints = np.array(
            [
                # Find more optimal waypoints
                self.initial_pos,
                [0.8, 1.0, 0.3],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.65],
                obs["gates_pos"][2] + [+0.2, -0.3, 0],
                obs["obstacles_pos"][2] + [0.2, -0.5, -1],
                obs["gates_pos"][3] + [0, -0.3, 0],
                obs["gates_pos"][3] + [-0, -0.4, 0],
                obs["gates_pos"][3] + [0.1, -0.4, 0],
            ]
        )
        self.generate_trajectory()

    def update_gate_trajectory(self, gate_index: int):
        """Update the trajectory for a visited gate."""
        distances = np.linalg.norm(self.waypoints - self.gates_pos[gate_index], axis=1)

        # Identify indices of waypoints within a 0.5 radius
        waypoints_to_replace = np.where(distances <= 0.4)[0]

        # Get gate orientation and forward direction
        rotation = R.from_quat(self.gates_quat[gate_index])
        gate_forward = rotation.apply([0, 1, 0])  # Gate's forward direction in local frame
        # print(f"{gate_forward}")
        # Define positions for before and after the gate to guide through the center
        center_of_gate = self.gates_pos[gate_index]  # Center of the gate
        before_gate = center_of_gate - 0.3 * gate_forward  # Approach position
        after_gate = center_of_gate + 0.3 * gate_forward  # Exit position

        # Replace waypoints affected by the gate
        if len(waypoints_to_replace) > 0:
            first_idx = waypoints_to_replace[0]
            last_idx = waypoints_to_replace[-1] + 1

            # Special handling for gate 1 (second gate, index=1): add X-offset to exit
            if gate_index == 1:
                self.waypoints = np.vstack(
                    (
                        self.waypoints[:first_idx],  # Keep waypoints before affected region
                        before_gate[np.newaxis, :],  # Insert before_gate
                        center_of_gate[np.newaxis, :],  # Insert gate center
                        after_gate[np.newaxis, :] - [0.2, 0.0, 0.0],  # Insert after_gate
                        self.waypoints[last_idx:],  # Keep waypoints after affected region
                    )
                )
            elif gate_index == 2:
                after_gate = center_of_gate + 0.05 * gate_forward  # Exit position
                self.waypoints = np.vstack(
                    (
                        self.waypoints[:first_idx],  # Keep waypoints before affected region
                        before_gate[np.newaxis, :]
                        + [0.1, -0.1, 0.0],  # Insert before_gate -[0.2 , 0.0 , 0.0]
                        center_of_gate[np.newaxis, :],  # Insert gate center
                        after_gate[np.newaxis, :],  # Insert after_gate
                        self.waypoints[last_idx:],  # Keep waypoints after affected region
                    )
                )
            else:
                # Update waypoints to ensure the drone passes through the center
                self.waypoints = np.vstack(
                    (
                        self.waypoints[:first_idx],  # Keep waypoints before affected region
                        before_gate[np.newaxis, :],  # Insert before_gate
                        center_of_gate[np.newaxis, :],  # Insert gate center
                        after_gate[np.newaxis, :],  # Insert after_gate
                        self.waypoints[last_idx:],  # Keep waypoints after affected region
                    )
                )
            # print(f"Updated waypoints to center: {self.waypoints}")

        # Regenerate the trajectory
        self.generate_trajectory()

    def update_obstacle_trajectory(self, obstacle_index: int):
        """Update the trajectory for a visited obstacle."""
        distances = np.linalg.norm(
            self.waypoints[:, :2] - self.obstacles[obstacle_index, :2], axis=1
        )
        index_obstacle = np.argmin(distances)
        # print(f"{distances}")

        if distances[index_obstacle] <= self.obstacle_size:
            # Calculate midpoint and adjust direction
            before_point = self.waypoints[index_obstacle - 1]
            after_point = self.waypoints[index_obstacle + 1]
            obstacle_pos = self.obstacles[obstacle_index]

            direction = after_point[:2] - before_point[:2]
            direction /= np.linalg.norm(direction)
            # Move the waypoint slightly around the obstacle

            adjusted_wp = self.replan_waypoint(
                self.waypoints[index_obstacle], obstacle_pos, safety_radius=0.3
            )
            self.waypoints[index_obstacle] = adjusted_wp
        self.generate_trajectory()

    def generate_trajectory(self):
        t = np.linspace(0, self.t_total, len(self.waypoints))
        # Build minsnap Waypoint objects
        refs = [ms.Waypoint(time=ti, position=pos) for ti, pos in zip(t, self.waypoints)]
        # Generate minsnap trajectory (degree=5 for fast/safe, or 7/8 for more smoothness)
        self.minsnap_traj = ms.generate_trajectory(
            refs,
            degree=6,
            idx_minimized_orders=(4,),
            num_continuous_orders=3,
            algorithm="closed-form",
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        i_target = obs["target_gate"]
        self.current_pos = obs["pos"]
        self.gates_pos = obs["gates_pos"]
        self.gates_quat = obs["gates_quat"]
        self.gates_visited = obs["gates_visited"]
        self.obstacles = obs["obstacles_pos"]
        self.obstacles_visited = obs["obstacles_visited"]

        if self.gates_visited[i_target] and not self.gates_flags[i_target]:
            self.update_gate_trajectory(i_target)
            self.gates_flags[i_target] = True

        for i, visited in enumerate(self.obstacles_visited):
            if visited and not self.obstacles_flags[i]:
                self.update_obstacle_trajectory(i)
                self.obstacles_flags[i] = True

        tau = min(self._tick / self._freq, self.t_total)
        # Sample minsnap trajectory at time tau
        pva = ms.compute_trajectory_derivatives(self.minsnap_traj, np.array([tau]), 1)
        target_pos = pva[0, 0]  # position at time tau
        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

    def replan_waypoint(x0, obstacle_center, safety_radius):
        """Adjusts a 2D waypoint to avoid a circular obstacle using constrained optimization."""

        def objective(x):
            return np.sum((x - x0) ** 2)  # Minimize distance from original point

        def constraint(x):
            return (
                np.linalg.norm(x - obstacle_center) - safety_radius
            )  # Must be outside safety circle

        cons = {"type": "ineq", "fun": constraint}

        result = minimize(objective, x0, constraints=cons, method="SLSQP")

        if result.success:
            return result.x
        else:
            print("Replanning failed:", result.message)
            return x0  # fallback to original

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished
        """Reset the time step counter."""
        self._tick = 0
