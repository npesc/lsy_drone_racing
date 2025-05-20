"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

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
        self.t_total = 12
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
                self.initial_pos,
                [0.8, 1.0, 0.3],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.65],
                [0.0, 1.2, 0.525],
                [0.0, 1.1, 1.1],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )
        self.generate_trajectory()

    def update_gate_trajectory(self, gate_index: int):
        """Update the trajectory for a visited gate."""
        distances = np.linalg.norm(self.waypoints - self.gates_pos[gate_index], axis=1)
        
        # Identify indices of waypoints within a 0.5 radius
        waypoints_to_replace = np.where(distances <= 0.5)[0]

        # Get gate orientation and forward direction
        rotation = R.from_quat(self.gates_quat[gate_index])
        gate_forward = rotation.apply([0, 1, 0])  # Gate's forward direction in local frame
        #print(f"{gate_forward}")
        # Define positions for before and after the gate to guide through the center
        center_of_gate = self.gates_pos[gate_index]  # Center of the gate
        before_gate = center_of_gate - 0.3 * gate_forward  # Approach position
        after_gate = center_of_gate + 0.3 * gate_forward  # Exit position

        # Replace waypoints affected by the gate
        if len(waypoints_to_replace) > 0:
            first_idx = waypoints_to_replace[0]
            last_idx = waypoints_to_replace[-1] + 1
            if gate_index == 1 : 
                self.waypoints = np.vstack((
                self.waypoints[:first_idx],  # Keep waypoints before affected region
                before_gate[np.newaxis, :],  # Insert before_gate
                center_of_gate[np.newaxis, :],  # Insert gate center
                after_gate[np.newaxis, :]-[0.2 , 0.0 , 0.0],  # Insert after_gate
                self.waypoints[last_idx:]  # Keep waypoints after affected region
            ))
            else : 
            # Update waypoints to ensure the drone passes through the center
                self.waypoints = np.vstack((
                    self.waypoints[:first_idx],  # Keep waypoints before affected region
                    before_gate[np.newaxis, :],  # Insert before_gate
                    center_of_gate[np.newaxis, :],  # Insert gate center
                    after_gate[np.newaxis, :],  # Insert after_gate
                    self.waypoints[last_idx:]  # Keep waypoints after affected region
                ))
            #print(f"Updated waypoints to center: {self.waypoints}")
        
        # Regenerate the trajectory
        self.generate_trajectory()

    def update_obstacle_trajectory(self, obstacle_index: int):
        """Update the trajectory for a visited obstacle."""
        marge = 0.075
        distances = np.linalg.norm(self.waypoints[:,:2] - self.obstacles[obstacle_index,:2], axis=1)
        index_obstacle = np.argmin(distances)
        #print(f"{distances}")
        
        if distances[index_obstacle] <= self.obstacle_size - 0.2 :
            # Calculate midpoint and adjust direction
            before_point = self.waypoints[index_obstacle[0] - 1]
            after_point = self.waypoints[index_obstacle[0] + 1]
            obstacle_pos = self.obstacles[obstacle_index]

            direction = after_point[:2] - before_point[:2]
            direction /= np.linalg.norm(direction)
            direction_3d = np.array([-direction[1], direction[0], 0.0])
            # Move the waypoint slightly around the obstacle
            adjustment = self.obstacle_size * 1.2 * direction_3d
            adjusted_waypoint = obstacle_pos + adjustment
            self.waypoints[index_obstacle] = adjusted_waypoint
        self.generate_trajectory()

    def generate_trajectory(self):
        t = np.linspace(0, self.t_total, len(self.waypoints))
        self.trajectory = CubicSpline(t, self.waypoints)

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

        if self.gates_visited[i_target] and not self.gates_flags[i_target] :
            self.update_gate_trajectory(i_target)
            self.gates_flags[i_target] = True
        

        for i, visited in enumerate(self.obstacles_visited):
            if visited and not self.obstacles_flags[i] :
                self.update_obstacle_trajectory(i)
        

        tau = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(tau)
        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

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
