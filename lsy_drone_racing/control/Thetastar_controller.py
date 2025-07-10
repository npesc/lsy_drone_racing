from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline

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
        self.t_total = 12
        self._freq = config.env.freq
        self._finished = False
        self.resolution = 0.05
        self.grid_size = (np.array([4, 4, 2]) / self.resolution).astype(int).tolist() # Grid size in cells
        self.grid = Grid(matrix = np.ones(self.grid_size, dtype=np.uint8))
        self.obstacle_size = 0.05
        self.gates_size = 0.125

        self.current_pos = obs["pos"]
        self.gates_pos = obs["gates_pos"]
        self.gates_quat = obs["gates_quat"]
        self.obstacles = obs["obstacles_pos"]

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

    def _generate_obstacle_map(self) -> None:
        """Update the occupancy grid with obstacles."""
        margin = int(self.obstacle_size / self.resolution)
        for obs_pos in self.obstacles:
            gx, gy, gz = self.world_to_grid(*obs_pos)
            for dx in range(gx - margin, gx + margin):
                for dy in range(gy - margin, gy + margin):
                    for dz in range(0, gz):
                        self.grid.nodes[dx][dy][dz].walkable = False
    
    def _generate_gate_map(self) -> None:
        """Update the occupancy grid with gates as obstacles."""
        rot_90_z = R.from_euler('z', np.pi / 2)  # 90 degrees around Z
        buffer_cells = int(self.gates_size / self.resolution)  # safety margin in cells

        for gate_pos, gate_quat in zip(self.gates_pos, self.gates_quat):
            center = np.array(gate_pos)
            rotation = R.from_quat(gate_quat)
            rpy = rotation.as_euler("xyz")
            
            # Define gate corners in XZ plane (Z = height)
            half_size = 0.245 + 0.0125  # meters (includes thickness)
            local_corners = np.array([
                [-half_size, 0.0, -half_size],  # bottom left
                [-half_size, 0.0,  half_size],  # top left
                [ half_size, 0.0,  half_size],  # top right
                [ half_size, 0.0, -half_size],  # bottom right
            ])
            
            # Rotate and translate to world
            rotated_corners = rotation.apply(local_corners)
            print(rotated_corners, local_corners)
            world_corners = rotated_corners + center

            edge_pairs = [
            (0, 1),  # left vertical (bottom-left to top-left)
            (3, 2),  # right vertical (bottom-right to top-right)
            (0, 3),  # bottom horizontal (left to right)
            (1, 2),  # top horizontal (left to right)
            ]

            for start_idx, end_idx in edge_pairs:
                p0 = world_corners[start_idx]
                p1 = world_corners[end_idx]

                # Sample points along the edge
                num_samples = int(np.linalg.norm(p1 - p0) / self.resolution) + 1
                for t in np.linspace(0, 1, num_samples):
                    pt = (1 - t) * p0 + t * p1
                    gx, gy, gz = self.world_to_grid(*pt)
                     # Inflate obstacle by marking neighbors unwalkable
                    for dx in range(-buffer_cells, buffer_cells + 1):
                        for dy in range(-buffer_cells, buffer_cells + 1):
                            for dz in range(-buffer_cells, buffer_cells + 1):
                                nx, ny, nz = gx + dx, gy + dy, gz + dz
                                self.grid.nodes[nx][ny][nz].walkable = False

    def _generate_trajectory(self) -> None:
        """Plan a new trajectory using Theta*."""
        
        for i in range(len(self.gates_pos)):
            # Get the end node for this gate 
            finder = ThetaStarFinder(diagonal_movement=DiagonalMovement.always)
            self._generate_obstacle_map()
            self._generate_gate_map()
            end_node = self.grid.node(*self.world_to_grid(*self.gates_pos[i]))
            start_node = (self.grid.node(*self.world_to_grid(*self.current_pos))
                            if i == 0
                            else self.grid.node(*self.world_to_grid(*self.gates_pos[i - 1])))
            
            # Find the path between start and end nodes
            path, _ = finder.find_path(start_node, end_node, self.grid)    
            self.grid.cleanup() 

            # Convert path to world coordinates and round
            processed_path = [
                tuple(round(coord, 2) for coord in self.grid_to_world(*p.identifier))
                for p in path
            ]

            # Append to planned path; skip duplicate start node for intermediate segments
            if i == 0:
                self.waypoints= processed_path
            else:
                self.waypoints = np.vstack((self.waypoints, np.array(processed_path[1:])))

                           
            print("path:", self.waypoints)

    
        #print("operations:", runs, "path length:", len(self.planned_path))
        
        t = np.linspace(0, self.t_total, len(self.waypoints))
        self.trajectory = CubicSpline(t, self.waypoints)
        #print("path:", self.trajectory)

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""
        self.current_pos = obs["pos"]
        self.obstacles = obs["obstacles_pos"]

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
        """Increment the time step counter and check if finished."""
        self._tick += 1
        return self._finished

    def reset(self):
        """Reset the controller state."""
        self._tick = 0
        self._finished = False
        self.path_index = 0
        self._generate_trajectory()
