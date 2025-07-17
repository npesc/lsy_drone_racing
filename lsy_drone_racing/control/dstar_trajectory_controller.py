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

        self.grid = np.ones(grid_size, dtype=np.uint8)

        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))

        for goal in self.goals:
            self.rhs[goal] = 0

        self.U = []
        self.entry_finder = {}  # maps node -> key
        self.visited = set()

        for goal in self.goals:
            self.insert(goal, self.calculate_key(goal))

    def heuristic(self, a, b):
        return math.dist(a, b)

    def calculate_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        h = min(self.heuristic(s, goal) for goal in self.goals)
        return (g_rhs + h + self.km, g_rhs)

    def insert(self, s, key):
        self.entry_finder[s] = key
        heapq.heappush(self.U, (key, s))


    def update_vertex(self, u):
        if u in self.visited:
            return
        self.visited.add(u)

        if u not in self.goals:
            min_rhs = float('inf')
            for s in self.get_neighbors(u):
                if self.is_valid(s):
                    min_rhs = min(min_rhs, self.g[s] + self.cost(u, s))
            self.rhs[u] = min_rhs

        if self.g[u] != self.rhs[u]:
            key = self.calculate_key(u)
            # Only insert if key improved
            if u not in self.entry_finder or key < self.entry_finder[u]:
                self.insert(u, key)

    def cost(self, a, b):
        return float('inf') if not self.is_valid(b) else self.heuristic(a, b)

    def is_valid(self, pos):
        x, y, z = pos
        return (
            0 <= x < self.grid_size[0] and
            0 <= y < self.grid_size[1] and
            0 <= z < self.grid_size[2] and
            self.grid[x, y, z] == 1  # walkable
        )

    def update_grid(self, grid_point, walkable):
        #print(grid_point)
        x, y, z = grid_point
        self.grid[x, y, z] = walkable
        self.update_vertex(grid_point)
        for s in self.get_neighbors(grid_point):
            self.update_vertex(s)


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
            self.U[0][0] < self.calculate_key(self.start) or
            self.rhs[self.start] != self.g[self.start]
        ):
            k_old, u = heapq.heappop(self.U)
            
            # Lazy removal: skip outdated entries
            if u in self.entry_finder and self.entry_finder[u] != k_old:
                continue  # outdated, skip
            
            self.entry_finder.pop(u, None)

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
        
        #print(self.gates)
        self.resolution = 0.05
        self.grid_size = (np.array([3, 3, 2]) / self.resolution).astype(int).tolist()
        
        self.start = tuple(self.world_to_grid(*obs["pos"]))            
        self.gates = [tuple(self.world_to_grid(*g)) for g in obs["gates_pos"]]

        # --- 2) Initialize D* Lite planner to first gate ---
        self.current_gate_idx = 0
        self.planner = MultiGoalDStarLite3D(self.start,
                                   self.gates,
                                   self.grid_size)

        # --- 3) Plan initial discrete path ---
        self.planner.compute_shortest_path()
        grid_raw_path = self.planner.get_path()
        world_raw_path = [self.grid_to_world(*pt) for pt in grid_raw_path]
        print(world_raw_path)
        # --- 4) Smooth it with a cubic spline ---
        try:
            smooth_path = self._make_smooth_traj(world_raw_path)
        except Exception as e:
            print(f"❗Smoothing failed: {e}. Using raw path.")
            smooth_path = world_raw_path

        # timestep tracking
        self._tick = 0
        self._freq = config.env.freq

    def _make_smooth_traj(self, waypoints: list[tuple[float, float, float]]):
        # collapse collinear segments first if you like...
        pts = np.array(waypoints)
        T = np.linspace(0, len(pts)-1, len(pts))
        self._traj = CubicSpline(T, pts, axis=0)
        self._t_end = T[-1]

    def compute_obstacle_boxes(self, index):
        """
        Compute bounding boxes [xmin, ymin, zmin, xmax, ymax, zmax] for each obstacle.

        Parameters:
            obstacle_pos (list of [x, y, z]): list of obstacle positions
            obstacle_size (tuple) : (obstacle_width, obstacle_height)

        Returns:
            np.ndarray: Flattened list of bounding box coordinates for all obstacles
        """
        obstacle_width, obstacle_height = self.obstacles_size
        
        x, y, z = self.obstacles[index]
        xmin = x - obstacle_width
        xmax = x + obstacle_width
        ymin = y - obstacle_width
        ymax = y + obstacle_width
        zmin = z - obstacle_height
        zmax = z

        return np.array([xmin, ymin, zmin, xmax, ymax, zmax])
        
    def compute_gate_edge_boxes(self, index):
        """
        Compute bounding boxes for all edges of all gates.

        Parameters:
            gate_pos (np.ndarray): (N_gates, 3) array of gate center positions
            gate_quat (np.ndarray): (N_gates, 4) array of quaternions [x, y, z, w]
            gate_width (float): width of the gate center opening
            gate_edge (tuple): (edge_width, edge_thickness, edge_height)

        Returns:
            np.ndarray: (N_gates * 4, 6) array of bounding boxes per edge:
                        [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        edge_width, edge_thickness, edge_height = self.gates_edges_size
        gate_width = self.gates_size
        pos = self.gates_pos[index]
        quat = self.gates_quat[index]
        rotation = R.from_quat(quat)
        #print(quat)
        #print(rotation)
        
        all_boxes = []

        for edge in range(4):
            i = 1 if edge % 2 == 0 else -1

            if edge < 2:
                # Left/right vertical edge
                p0 = rotation.apply([
                    -i * (gate_width + 2*edge_width),
                    -edge_thickness,
                    -edge_height
                ]) + pos

                p1 = rotation.apply([
                    -i * (gate_width),
                    edge_thickness,
                    edge_height
                ]) + pos
            else:
                # Top/bottom horizontal edge
                p0 = rotation.apply([
                    -edge_height,
                    -edge_thickness,
                    -i * (gate_width + 2*edge_width),
                ]) + pos

                p1 = rotation.apply([
                    edge_height,
                    edge_thickness,
                    -i * (gate_width),
                ]) + pos

            min_edge = np.minimum(p0, p1)
            max_edge = np.maximum(p0, p1)
            #print(min_edge, max_edge)
            all_boxes.append(np.array([*min_edge, *max_edge]))

        return all_boxes

    def world_to_grid(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int((x + 1.5) / self.resolution)
        gy = int((y + 1.5) / self.resolution)
        gz = int(z / self.resolution)
        return gx, gy, gz

    def grid_to_world(self, gx: int, gy: int, gz: int) -> tuple[float, float, float]:
        """Convert grid coordinates to world coordinates."""
        x = gx * self.resolution - 1.5
        y = gy * self.resolution - 1.5
        z = gz * self.resolution
        return x, y, z

    def _modify_grid(self, walkable : bool, pos_min : NDArray[np.floating], pos_max : NDArray[np.floating]) -> None:
        """Update the occupancy grid with obstacles."""
        gx_min, gy_min, gz_min = self.world_to_grid(*pos_min)
        gx_max, gy_max, gz_max = self.world_to_grid(*pos_max)
        for gx in range(gx_min, gx_max):
            for gy in range(gy_min, gy_max):
                for gz in range(gz_min, gz_max):
                    self.planner.update_grid((gx, gy, gz), walkable)


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
