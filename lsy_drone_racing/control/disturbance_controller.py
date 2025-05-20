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
        self._tick = 0
        self._freq = config.env.freq
        self.sensor_range = config.env.sensor_range
        self.obstacle_size = 0.05
        self._finished = False
        self.t_total = 11
        self.current_pos = obs["pos"]
        self.i_gate = obs["target_gate"]
        self.gates_pos = obs["gates_pos"]
        self.gates_quat = obs["gates_quat"]
        self.obstacles = obs["obstacles_pos"]
        self.obstacles_flags = [False] * len(self.obstacles)
        self.gates_flags = [False] * len(self.obstacles)
        self.index_gate = 0
         # Same waypoints as in the trajectory controller. Determined by trial and error.
        self.waypoints = np.array(
            [
                obs["pos"],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.65],
                [0.0, 1.2, 0.525],
                [0.0, 1.2, 1.1],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )

        self.update_trajectory()

    def compute_detour(self, wp1, wp2):
        
        direction = wp2[:2] - wp1[:2]
        if np.linalg.norm(direction) < 1e-8:  # avoid division by zero or tiny norm
            # fallback: create some fixed perpendicular direction, for example pointing upward in XY
            perp = np.array([0.05, 0.05, 0])
        else:
            direction /= np.linalg.norm(direction)  # normalize
            # Vector perpendicular to direction in XY plane (to side-step)
            perp = np.array([-direction[1], direction[0], 0])
        
        safe_distance = 0.1  # some margin bigger than obstacle size 0.05
        
        # Detour point is midpoint between wp1 and wp2, shifted perpendicularly by safe_distance
        midpoint = (wp1 + wp2) / 2
        detour_point = midpoint + perp * safe_distance

        return detour_point


    def update_trajectory(self):
        # Generate the spline trajectory
        t = np.linspace(0, self.t_total, len(self.waypoints))
        self.trajectory = CubicSpline(t, self.waypoints)


    def adjust_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        for i, gate_visited in enumerate(obs["gates_visited"]):
            if gate_visited and not self.gates_flags[i] :  # Check only detected
                distances = np.linalg.norm(self.waypoints - self.gates_pos[i], axis=1)
                self.index_gate = np.argmin(distances)

                rotation = R.from_quat(self.gates_quat[i])
                gate_forward = rotation.apply([1, 0, 0])  # Gate's forward direction
                before_gate = self.gates_pos[i] - 0.2 * gate_forward
                after_gate = self.gates_pos[i] + 0.2 * gate_forward

                
                self.waypoints[self.index_gate] = self.gates_pos[i]
                self.waypoints = np.insert(self.waypoints, self.index_gate, after_gate, axis=0)
                self.waypoints = np.insert(self.waypoints, self.index_gate-1, before_gate, axis=0)
                self.obstacles_flags[i] = True

        for i, obstacle_visited in enumerate(obs["obstacles_visited"]):
            if obstacle_visited and not self.obstacles_flags[i] :  # Check only detected obstacles
                obstacle_xy = self.obstacles[i, :2]
                distances = np.linalg.norm(self.waypoints[:, :2] - obstacle_xy, axis=1)
                closest_indices = np.argsort(distances)[:2]

                # Ensure closest_indices are sorted ascending
                if closest_indices[0] > closest_indices[1]:
                    closest_indices = closest_indices[::-1]

                wp1 = self.waypoints[closest_indices[0], :2]
                wp2 = self.waypoints[closest_indices[1], :2]

                # Check shortest distance from obstacle to segment wp1-wp2
                norm_sq = np.dot(obstacle_xy - wp1, wp2 - wp1)
                if norm_sq == 0:
                    dist_to_segment = np.linalg.norm(obstacle_xy - wp1)  # a and b are the same point
                else:
                    t = np.clip(np.dot(obstacle_xy - wp1, wp2 - wp1) / norm_sq, 0, 1)
                    projection = wp1 + t * (wp2 - wp1)
                    dist_to_segment = np.linalg.norm(obstacle_xy - projection)
                if dist_to_segment <= self.obstacle_size + 0.02:
                    # Line segment passes close to obstacle — adjust waypoints
                    # Check if obstacle is very close to wp1 for replacement
                    perp = np.array([0.5, 0.5, 0])
                    safe_distance = 0.05
                    midpoint = (wp1 + wp2) / 2
                    self.waypoints[closest_indices[0]] = midpoint + perp * safe_distance

          
        self.update_trajectory()

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
        self.current_pos = obs["pos"]
        self.i_gate = obs["target_gate"]
        self.gates_pos = obs["gates_pos"]
        self.gates_quat = obs["gates_quat"]
        self.obstacles = obs["obstacles_pos"]
        self.adjust_trajectory(obs)

        tau = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(tau)

        #print(f"Tick: {self._tick}, Tau: {tau:.3f}, Target waypoint: {target_pos}")

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
