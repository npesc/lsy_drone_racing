from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
import math

from lsy_drone_racing.control.controller import Controller
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray


import heapq
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

import heapq
import math
from collections import defaultdict

class MultiGoalDStarLite3D:
    def __init__(self, start, goals, grid_size):
        self.start = start
        self.goals = set(map(tuple, goals))  # Ensure hashable tuples
        self.grid_size = grid_size
        self.km = 0

        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))

        for goal in self.goals:
            self.rhs[goal] = 0

        self.U = []
        self.obstacles = set()

        for goal in self.goals:
            self.insert(goal, self.calculate_key(goal))

    def heuristic(self, a, b):
        return math.dist(a, b)

    def calculate_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        h = min(self.heuristic(self.start, goal) for goal in self.goals)
        return (g_rhs + h + self.km, g_rhs)

    def insert(self, s, key):
        heapq.heappush(self.U, (key, s))

    def update_vertex(self, u):
        if u not in self.goals:
            self.rhs[u] = min(
                self.g[s] + self.cost(u, s)
                for s in self.get_neighbors(u)
                if self.is_valid(s)
            )
        self.U = [(k, s) for (k, s) in self.U if s != u]
        heapq.heapify(self.U)
        if self.g[u] != self.rhs[u]:
            self.insert(u, self.calculate_key(u))

    def cost(self, a, b):
        return float('inf') if not self.is_valid(b) else self.heuristic(a, b)

    def is_valid(self, pos):
        return pos not in self.obstacles

    def get_neighbors(self, pos):
        x, y, z = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and 0 <= nz < self.grid_size[2]:
                        yield (nx, ny, nz)

    def compute_shortest_path(self):
        while self.U and (
            self.U[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]
        ):
            k_old, u = heapq.heappop(self.U)
            k_new = self.calculate_key(u)
            if k_old < k_new:
                self.insert(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

    def update_obstacles(self, new_obstacles):
        for obs in new_obstacles:
            if obs not in self.obstacles:
                self.obstacles.add(obs)
                for s in self.get_neighbors(obs):
                    self.update_vertex(s)

    def get_path(self, max_steps=500):
        path = [self.start]
        current = self.start
        steps = 0
        while current not in self.goals and steps < max_steps:
            neighbors = list(self.get_neighbors(current))
            neighbors = [n for n in neighbors if self.is_valid(n)]
            if not neighbors:
                return None
            current = min(neighbors, key=lambda s: self.g[s] + self.heuristic(current, s))
            path.append(current)
            steps += 1
        return path if current in self.goals else None


class DStarTrajectoryController(Controller):
    """
    Controller that plans via D* Lite in 3D, then follows
    a smooth spline through the resulting waypoints.
    """

    def __init__(self,
                 obs: dict[str, NDArray[np.floating]],
                 info: dict,
                 config: dict):
        super().__init__(obs, info, config)

        # --- 1) Read start, gate positions, grid size from config or info ---
        self.start = tuple(obs["pos"])            # e.g. (0,0,0)
        self.gates = [tuple(g) for g in obs["gates_pos"]]  # e.g. [(5,5,0),(10,10,1)...]
        print(self.gates)
        self.grid_size = (50, 50, 20)

        # --- 2) Initialize D* Lite planner to first gate ---
        self.current_gate_idx = 0
        self.planner = MultiGoalDStarLite3D(self.start,
                                   self.gates,
                                   self.grid_size)

        # --- 3) Plan initial discrete path ---
        self.planner.compute_shortest_path()
        raw_path = self.planner.get_path()

        # --- 4) Smooth it with a cubic spline ---
        try:
            smooth_path = self._make_smooth_traj(raw_path)
        except Exception as e:
            print(f"❗Smoothing failed: {e}. Using raw path.")
            smooth_path = raw_path

        # timestep tracking
        self._tick = 0
        self._freq = config.env.freq

    def _make_smooth_traj(self, waypoints: list[tuple[float, float, float]]):
        # collapse collinear segments first if you like...
        pts = np.array(waypoints)
        T = np.linspace(0, len(pts)-1, len(pts))
        self._traj = CubicSpline(T, pts, axis=0)
        self._t_end = T[-1]

    def compute_control(self,
                        obs: dict[str, NDArray[np.floating]],
                        info: dict | None = None
                       ) -> NDArray[np.floating]:
        """
        Called at each environment step. Returns
        [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate].
        """
        # 1) If we’re at the end of current spline, see if we reached gate
        tau = min(self._tick / self._freq, self._t_end)
        pos = self._traj(tau)

        # 2) If we’ve just arrived at end of spline, move to next gate
        if tau >= self._t_end:
            self.current_gate_idx += 1
            if self.current_gate_idx < len(self.gates):
                # reinitialize planner for next gate
                start = tuple(pos.tolist())
                goal  = self.gates[self.current_gate_idx]
                self.planner.reinitialize(start, goal)
                self.planner.compute_shortest_path()
                raw_path = self.planner.get_path()
                self._make_smooth_traj(raw_path)
                self._tick = 0
                tau = 0
                pos = self._traj(0.0)

        # 3) Build full state vector (zero velocity/attitude rates for simplicity)
        state = np.zeros(13, dtype=np.float32)
        state[0:3] = pos
        return state

    def step_callback(self,
                      action: NDArray[np.floating],
                      obs: dict[str, NDArray[np.floating]],
                      reward: float,
                      terminated: bool,
                      truncated: bool,
                      info: dict,
                     ) -> bool:
        """
        Called once per step. Here we could incorporate
        online obstacle detection:
        """
        # Example: if lidar sees a new obstacle, replan mid‐spline.
        new_obs = info.get("new_obstacles")  # your env should fill this
        if new_obs:
            self.planner.update_obstacles(new_obs)
            self.planner.compute_shortest_path()
            path = self.planner.get_path()
            self._make_smooth_traj(path)
            # reset tau so we follow the new spline from current pos
            self._tick = 0

        self._tick += 1
        # Return True only when all gates are done
        return self.current_gate_idx >= len(self.gates)
