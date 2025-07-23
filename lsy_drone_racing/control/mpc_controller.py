"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import minsnap_trajectories as ms
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, fmax, fmin, horzcat, sin, sqrt, vertcat
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


PARAMS_RPY = np.array([[-12.7, 10.15], [-12.7, 10.15], [-8.117, 14.36]])
PARAMS_ACC = np.array([0.1906, 0.4903])
MASS = 0.033
GRAVITY = 9.81
THRUST_MIN = 0.05
THRUST_MAX = 0.1425
OBSTACLE_SIZE = 0.05
GATE_SIZE = [0.245, 0.045, 0.0125, 0.29]


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "drone_mpc"

    """Model setting"""
    # define basic variables in state and input vector
    pos = vertcat(MX.sym("x"), MX.sym("y"), MX.sym("z"))
    vel = vertcat(MX.sym("vx"), MX.sym("vy"), MX.sym("vz"))
    rpy = vertcat(MX.sym("r"), MX.sym("p"), MX.sym("y"))

    r_cmd, p_cmd, y_cmd = MX.sym("r_cmd"), MX.sym("p_cmd"), MX.sym("y_cmd")
    thrust_cmd = MX.sym("thrust_cmd")

    obstacle_pos = MX.sym("obstacles_params", 3)  # [x, y, z]
    gate_param = MX.sym("gates_params", 7)  # [x, y, z, qx, qy, qz, qw]

    d_obs_safe = 0.1 + OBSTACLE_SIZE
    d_gate_safe = 0.2
    obs_cost = fmax(
        d_obs_safe - sqrt((pos[0] - obstacle_pos[0]) ** 2 + (pos[1] - obstacle_pos[1]) ** 2), 0
    )
    gate_cost = []

    rotation = quat_to_rot(gate_param[3:])
    for i in range(4):
        edges = compute_gate_edges(gate_param[:3], rotation, i)

        xmin = edges[0]
        ymin = edges[1]
        zmin = edges[2]
        xmax = edges[3]
        ymax = edges[4]
        zmax = edges[5]

        dx = fmax(xmin - pos[0], fmax(pos[0] - xmax, 0))
        dy = fmax(ymin - pos[1], fmax(pos[1] - ymax, 0))
        dz = fmax(zmin - pos[2], fmax(pos[2] - zmax, 0))

        dist_gate = fmax(d_gate_safe - sqrt(dx**2 + dy**2 + dz**2), 0)
        gate_cost.append(dist_gate)

    # define state and input vector
    states = vertcat(pos, vel, rpy)
    inputs = vertcat(thrust_cmd, r_cmd, p_cmd, y_cmd)

    # Define nonlinear system dynamics
    pos_dot = vel
    z_axis = vertcat(
        cos(rpy[0]) * sin(rpy[1]) * cos(rpy[2]) + sin(rpy[0]) * sin(rpy[2]),
        cos(rpy[0]) * sin(rpy[1]) * sin(rpy[2]) - sin(rpy[0]) * cos(rpy[2]),
        cos(rpy[0]) * cos(rpy[1]),
    )
    thrust = PARAMS_ACC[0] + PARAMS_ACC[1] * inputs[0]
    vel_dot = thrust * z_axis / MASS - vertcat([0.0, 0.0, GRAVITY])
    rpy_dot = PARAMS_RPY[:, 0] * rpy + PARAMS_RPY[:, 1] * inputs[1:]
    f = vertcat(pos_dot, vel_dot, rpy_dot)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.cost_y_expr = vertcat(states, inputs, obs_cost, *gate_cost)
    model.cost_y_expr_e = states
    model.x = states
    model.u = inputs
    model.p = vertcat(obstacle_pos, gate_param)

    return model


def quat_to_rot(q):
    """Convert CasADi quaternion [x, y, z, w] to rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return vertcat(
        horzcat(1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w),
        horzcat(2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w),
        horzcat(2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2),
    )


def compute_gate_edges(gate_pos, rotation, edge):
    """Compute bounding boxes for all edges of all gates.

    Parameters:
        gate_pos (np.ndarray): (N_gates, 3) array of gate center positions
        gate_quat (np.ndarray): (N_gates, 4) array of quaternions [x, y, z, w]
        gate_width (float): width of the gate center opening
        gate_edge (tuple): (edge_width, edge_thickness, edge_height)

    Returns:
        np.ndarray: (N_gates * 4, 6) array of bounding boxes per edge:
                    [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    i = 1 if edge % 2 == 0 else -1

    if edge < 2:
        # Left/right vertical edge
        p0 = rotation @ vertcat(
            -i * (GATE_SIZE[0] + 2 * GATE_SIZE[1]), -GATE_SIZE[2], -GATE_SIZE[3]
        )

        p1 = rotation @ vertcat(-i * (GATE_SIZE[0]), GATE_SIZE[2], GATE_SIZE[3])
    else:
        # Top/bottom horizontal edge
        p0 = rotation @ vertcat(
            -GATE_SIZE[3], -GATE_SIZE[2], -i * (GATE_SIZE[0] + 2 * GATE_SIZE[1])
        )

        p1 = rotation @ vertcat(GATE_SIZE[3], GATE_SIZE[2], -i * (GATE_SIZE[0]))

    min_edge = fmin(p0, p1) + gate_pos
    max_edge = fmax(p0, p1) + gate_pos
    # print(min_edge, max_edge)

    return vertcat(min_edge, max_edge)


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu + 5
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.cost_y_expr = model.cost_y_expr
    ocp.cost_y_expr_e = model.cost_y_expr_e

    # Weights (we only give pos reference anyway)
    Q = np.diag(
        [
            10.0,  # pos
            10.0,  # pos
            20.0,  # pos
            0.0,  # vel
            0.0,  # vel
            0.0,  # vel
            0.0,  # rpy
            0.0,  # rpy
            10.0,  # rpy
        ]
    )

    R = np.diag(
        [
            25.0,  # thrust
            10.0,  # rpy
            10.0,  # rpy
            15.0,  # rpy
            10000.0,  # obstacle
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 60°)
    ocp.constraints.lbx = np.array([0.03, -1.0, -1.0, -0.1])
    ocp.constraints.ubx = np.array([2.0, 1.0, 1.0, 0.1])
    ocp.constraints.idxbx = np.array([2, 6, 7, 8])

    # Set Input Constraints (rpy < 60°)
    ocp.constraints.lbu = np.array([THRUST_MIN * 4, -1.0, -1.0, -1.0])
    ocp.constraints.ubu = np.array([THRUST_MAX * 4, 1.0, 1.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))
    p = model.p.rows()
    ocp.parameter_values = np.zeros((p))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="c_generated_code/lsy_example_mpc.json", verbose=verbose
    )

    return acados_ocp_solver, ocp


class MPC_Controller(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 20
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self._t_total = 10.0

        self.initial_pos = obs["pos"]
        self.gates_pos = obs["gates_pos"]
        self.gates_quat = obs["gates_quat"]
        self.gates_visited = obs["gates_visited"]
        self.gates_edges = np.zeros((4, 4, 6))
        self.obstacles_visited = obs["obstacles_visited"]

        # Same waypoints as in the trajectory controller. Determined by trial and error.

        waypoints = [self.initial_pos] + list(self.gates_pos) + [[-0.5, -0.5, 1.1]]
        segment_times = np.linspace(0, self._t_total, len(waypoints))

        refs = [ms.Waypoint(position=pos, time=t) for pos, t in zip(waypoints, segment_times)]

        polys = ms.generate_trajectory(
            refs,
            degree=13,
            idx_minimized_orders=(3, 4),
            num_continuous_orders=3,
            algorithm="closed-form",
        )

        sample_times = np.linspace(0, self._t_total, int(self._t_total / self._dt))

        pva = ms.compute_trajectory_derivatives(polys, sample_times, order=3)
        # Then you fill your reference arrays:
        x_des = pva[0, :, 0]  # position + velocity x
        y_des = pva[0, :, 1]  # position + velocity y
        z_des = pva[0, :, 2]  # position + velocity z

        x_pos = np.concatenate((x_des, [x_des[-1]] * (self._N + 1)))
        y_pos = np.concatenate((y_des, [y_des[-1]] * (self._N + 1)))
        z_pos = np.concatenate((z_des, [z_des[-1]] * (self._N + 1)))
        print(x_pos.shape)
        # x_vel = np.concatenate((x_des[1], [x_des[1, -1]] * (self._N + 1)))
        # y_vel = np.concatenate((y_des[1], [y_des[1, -1]] * (self._N + 1)))
        # z_vel = np.concatenate((z_des[1], [z_des[1, -1]] * (self._N + 1)))
        self._waypoints_pos = np.stack((x_pos, y_pos, z_pos)).T
        # self._waypoints_vel = np.stack((x_vel, y_vel, z_vel)).T
        self._waypoints_yaw = x_pos * 0

        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T_HORIZON, self._N)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu + 5
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(x_des) - 1 - self._N
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = Rot.from_quat(obs["quat"]).as_euler("xyz")
        x0 = np.concatenate((obs["pos"], obs["vel"], obs["rpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)
        i_target = obs["target_gate"]
        self.gates_pos = obs["gates_pos"]
        self.gates_quat = obs["gates_quat"]
        self.obstacles_pos = obs["obstacles_pos"]

        if self.gates_visited[i_target] != obs["gates_visited"][i_target]:
            waypoints = [x0[:3], self.gates_pos[i_target], self._waypoints_pos[i + self._N]]
            t_start = i * self._dt
            t_end = t_start + self._N * self._dt
            segment_times = np.linspace(t_start, t_end, len(waypoints))

            refs = [ms.Waypoint(position=pos, time=t) for pos, t in zip(waypoints, segment_times)]

            polys = ms.generate_trajectory(
                refs,
                degree=13,
                idx_minimized_orders=(3, 4),
                num_continuous_orders=3,
                algorithm="closed-form",
            )

            sample_times = np.linspace(t_start, t_end, int((t_end - t_start) / self._dt))
            pva = ms.compute_trajectory_derivatives(polys, sample_times, order=3)

            # Then you fill your reference arrays:
            x_des = pva[0, :, 0]  # position x
            y_des = pva[0, :, 1]  # position y
            z_des = pva[0, :, 2]  # position z
            x_pos = self._waypoints_pos[:, 0]
            y_pos = self._waypoints_pos[:, 1]
            z_pos = self._waypoints_pos[:, 2]

            x_pos = np.concatenate((x_pos[:i], x_des, x_pos[i + self._N + 1 :]))
            y_pos = np.concatenate((y_pos[:i], y_des, y_pos[i + self._N + 1 :]))
            z_pos = np.concatenate((z_pos[:i], z_des, z_pos[i + self._N + 1 :]))
            # print(x_pos.shape)

            self._waypoints_pos = np.stack((x_pos, y_pos, z_pos)).T
            # self._waypoints_vel = np.stack((x_vel, y_vel, z_vel)).T
            self._waypoints_yaw = x_pos * 0
            self.gates_visited = obs["gates_visited"]
            self.obstacles_visited = obs["obstacles_visited"]

        # Setting reference

        for j in range(self._N):
            idx_closest_gate = np.argmin(
                np.linalg.norm(self._waypoints_pos[i + j] - self.gates_pos, axis=1)
            )
            idx_closest_obs = np.argmin(
                np.linalg.norm(self._waypoints_pos[i + j, :2] - self.obstacles_pos[:, :2], axis=1)
            )
            params = np.concatenate(
                [
                    self.obstacles_pos[idx_closest_obs],
                    self.gates_pos[idx_closest_gate],
                    self.gates_quat[idx_closest_gate],
                ]
            )
            self._acados_ocp_solver.set(j, "p", params)
            yref = np.zeros((self._ny))
            yref[0:3] = self._waypoints_pos[i + j]  # position
            # yref[3:6] = self._waypoints_vel[i + j]  # velocity
            yref[8] = self._waypoints_yaw[i + j]  # yaw
            yref[9] = MASS * GRAVITY  # hover thrust
            self._acados_ocp_solver.set(j, "yref", yref)

        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        # yref[3:6] = self._waypoints_vel[i + self._N]  # velocity
        yref_e[8] = self._waypoints_yaw[i + self._N]  # yaw
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u1 = self._acados_ocp_solver.get(1, "u")

        # cost_value = self._acados_ocp_solver.get_cost()
        # print(f"Cost after solve: {cost_value}")
        # print(f"thrust = {self._acados_ocp_solver.get(1, "u")}")

        # WARNING: The following line is only for legacy reason!
        # The Crazyflie uses the rpyt command format, the environment
        # take trpy format. Remove this line as soon as the env
        # also works with rpyt!
        # u0 = np.array([u0[3], *u0[:3]], dtype=np.float32)

        return u1

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
