from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
import minsnap_trajectories as ms

from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller

# Import pathfinding libraries
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.theta_star import ThetaStarFinder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ThetaStarController(Controller):
    """Trajectory controller using Theta* for dynamic replanning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._tick = 0
        self.t_total = 10
        self._freq = config.env.freq
        self._finished = False
        self.resolution = 0.05
        self.grid_size = (np.array([4, 4, 2]) / self.resolution).astype(int).tolist() # Grid size in cells
        self.grid = Grid(matrix = np.ones(self.grid_size, dtype=np.uint8))
        self.obstacles_size = [0.05, 1.4] 
        self.gates_edges_size = [0.045, 0.0125, 0.29]
        self.gates_size = 0.245

        self.initial_pos = self.current_pos = obs["pos"]
        self.gates_pos = obs["gates_pos"]
        self.gates_quat = obs["gates_quat"]
        self.gates_visited = obs["gates_visited"]
        self.obstacles = obs["obstacles_pos"]
        self.obstacles_visited = obs["obstacles_visited"]
        self.target_gate_index = obs["target_gate"]

        self.waypoints = np.empty((0, 3))  
        self.trajectory = []
        self.path_index = 0
        self.current_target = None
        self._generate_trajectory()
        print(f"Planned path: {self.waypoints}")
        

    def world_to_grid(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int((x + 2) / self.resolution)
        gy = int((y + 2) / self.resolution)
        gz = int(z / self.resolution)
        return gx, gy, gz

    def grid_to_world(self, gx: int, gy: int, gz: int) -> tuple[float, float, float]:
        """Convert grid coordinates to world coordinates."""
        x = gx * self.resolution - 2
        y = gy * self.resolution - 2
        z = gz * self.resolution
        return x, y, z

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
        print(f"{rotation.apply([1, 1, 1])}")
        #print(quat)
        #print(rotation)
        
        all_boxes = []

        for edge in range(4):
            i = 1 if edge % 2 == 0 else -1

            if edge < 2:
                # Left/right vertical edge
                p0 = rotation.apply([
                    -edge_thickness,
                    -i * (gate_width + 2*edge_width),
                    -edge_height
                ]) + pos

                p1 = rotation.apply([
                    edge_thickness,
                    -i * (gate_width),
                    edge_height
                ]) + pos
            else:
                # Top/bottom horizontal edge
                p0 = rotation.apply([
                    -edge_thickness,
                    -edge_height,
                    -i * (gate_width + 2*edge_width),
                ]) + pos

                p1 = rotation.apply([
                    edge_thickness,
                    edge_height,
                    -i * (gate_width),
                ]) + pos

            edge_min = np.floor(np.minimum(p0, p1) * 10) / 10
            edge_max =  np.ceil(np.maximum(p0, p1) * 10) / 10
            print(edge_min, edge_max)
            all_boxes.append(np.array([*edge_min, *edge_max]))

        return all_boxes

    def _modify_grid(self, walkable : bool, pos_min : NDArray[np.floating], pos_max : NDArray[np.floating]) -> None:
        """Update the occupancy grid with obstacles."""
        gx_min, gy_min, gz_min = self.world_to_grid(*pos_min)
        gx_max, gy_max, gz_max = self.world_to_grid(*pos_max)
        for gx in range(gx_min, gx_max):
            for gy in range(gy_min, gy_max):
                for gz in range(gz_min, gz_max):
                    self.grid.nodes[gx][gy][gz].walkable = walkable

    def _generate_trajectory(self) -> None:
        """Plan a new trajectory using Theta*."""
        
        for i in range(len(self.gates_pos)) :
            # Get the end node for this gate 
            finder = ThetaStarFinder(diagonal_movement=DiagonalMovement.always)

            for j in range(len(self.obstacles)) : 
                xmin, ymin, zmin, xmax, ymax, zmax = self.compute_obstacle_boxes(j)
                self._modify_grid(False, np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]))
            
            for k in range(len(self.gates_pos)) :
                edges = self.compute_gate_edge_boxes(k)
                for edge in edges :
                    xmin, ymin, zmin, xmax, ymax, zmax = edge
                    self._modify_grid(False, np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]))

            #print(self.gates_pos[i])
            end_node = self.grid.node(*self.world_to_grid(*self.gates_pos[i]))
            start_node = (self.grid.node(*self.world_to_grid(*self.initial_pos))
                            if i == 0
                            else self.grid.node(*self.world_to_grid(*self.gates_pos[i - 1])))
            print(end_node, start_node)
            # Find the path between start and end nodes
            path, _ = finder.find_path(start_node, end_node, self.grid)    
            self.grid.cleanup() 
            print(path)
            # Convert path to world coordinates and round
            processed_path = [
                tuple(round(coord, 2) for coord in self.grid_to_world(*p.identifier))
                for p in path
            ]

            # Append to planned path; skip duplicate start node for intermediate segments
            if i == 0:
                self.waypoints = np.vstack((self.current_pos, np.array(processed_path[1:])))
            elif i == len(self.gates_pos) -1 :
                self.waypoints = np.vstack((self.waypoints, np.array(processed_path[1:]), np.array([-0.5, -0.5, 1.1])))
            else:
                self.waypoints = np.vstack((self.waypoints, np.array(processed_path[1:])))

                           
            print("path:", self.waypoints)

    
        #print("operations:", runs, "path length:", len(self.planned_path))
        
        # Distribute time across waypoints linearly (or use smarter time allocation if available)
        num_pts = len(self.waypoints)
        times = np.linspace(0, self.t_total, num_pts)

        # Create ms.Waypoint list
        refs = [
            ms.Waypoint(time=t, position=pos)
            for t, pos in zip(times, self.waypoints)
        ]

        # Generate minimum snap trajectory
        self.polys = ms.generate_trajectory(
            refs,
            degree=12,
            idx_minimized_orders=(3, 4),       # Minimize jerk and snap
            num_continuous_orders=3,           # Position, velocity, acceleration continuity
            algorithm="closed-form",           # Roy & Bry formulation
        )

        print("path:", self.polys)

    def update_partial_trajectory(self, gate_index: int) -> None:
        """Replan the trajectory from the current position to a specific gate."""
        gate_pos = self.gates_pos[gate_index]

        # 1. Find the closest waypoints in the current trajectory
        dists_to_start = np.linalg.norm(self.waypoints - self.current_pos, axis=1)
        start_idx = int(np.argmin(dists_to_start))

        dists_to_goal = np.linalg.norm(self.waypoints - gate_pos, axis=1)
        end_idx = int(np.argmin(dists_to_goal))

        if start_idx >= end_idx:
            start_idx, end_idx = end_idx, start_idx

        # 2. Replan using Theta* between current pos and gate_pos
        finder = ThetaStarFinder(diagonal_movement=DiagonalMovement.always)
        self.grid.cleanup()  # Reset walkable flags

        for j in range(len(self.obstacles)):
            xmin, ymin, zmin, xmax, ymax, zmax = self.compute_obstacle_boxes(j)
            self._modify_grid(False, np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]))

        for k in range(len(self.gates_pos)):
            edges = self.compute_gate_edge_boxes(k)
            for edge in edges:
                xmin, ymin, zmin, xmax, ymax, zmax = edge
                self._modify_grid(False, np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]))
        if self.target_gate_index == 0 : 
            start_node = self.grid.node(*self.world_to_grid(*self.initial_pos))
        else : 
            start_node = self.grid.node(*self.world_to_grid(*self.gates_pos[self.target_gate_index-1]))
        end_node = self.grid.node(*self.world_to_grid(*gate_pos))

        path, _ = finder.find_path(start_node, end_node, self.grid)
        self.grid.cleanup()

        processed_path = [
            tuple(round(coord, 2) for coord in self.grid_to_world(*p.identifier))
            for p in path
        ]

        new_segment = np.array([self.current_pos] + processed_path[1:])

        # 3. Replace the relevant segment in waypoints
        self.waypoints = np.vstack((
            self.waypoints[:start_idx],
            new_segment,
            self.waypoints[end_idx+1:]
        ))

        # 4. Rebuild the min-snap trajectory
        num_pts = len(self.waypoints)
        times = np.linspace(0, self.t_total, num_pts)
        refs = [
            ms.Waypoint(time=t, position=pos)
            for t, pos in zip(times, self.waypoints)
        ]

        self.polys = ms.generate_trajectory(
            refs,
            degree=8,
            idx_minimized_orders=(3, 4),
            num_continuous_orders=3,
            algorithm="closed-form",
        )
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""
        self.current_pos = obs["pos"]
        self.obstacles = obs["obstacles_pos"]
        new_detected = False

        for i in range(4):
            if obs["obstacles_visited"][i] and not self.obstacles_visited[i]:
                new_detected = True
                self.obstacles_visited = obs["obstacles_visited"]
            if obs["gates_visited"][i] and not self.gates_visited[i]:
                new_detected = True
                self.gates_visited = obs["gates_visited"]
        
        if new_detected or self.target_gate_index != obs["target_gate"]:
            self.target_gate_index = obs["target_gate"]
            #self.update_partial_trajectory(self.target_gate_index)

        tau = min(self._tick / self._freq, self.t_total)

        # Sample position from minimum snap trajectory
        pva = ms.compute_trajectory_derivatives(self.polys, np.array([tau]), 3)  # up to jerk
        target_pos = pva[0, 0]  # position at time tau

        if tau == self.t_total:  # Maximum duration reached
            self._finished = True

        # Return target_pos + zeros for rest of the state vector (10 zeros)
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
        """Increment the time step counter and check if finished."""
        self._tick += 1
        return self._finished

    def reset(self):
        """Reset the controller state."""
        self._tick = 0
        self._finished = False
        self.path_index = 0
        self._generate_trajectory()
