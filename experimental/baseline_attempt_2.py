import os
import time
import argparse
import numpy as np
from datetime import datetime
import pybullet as p
from scipy.linalg import expm
from scipy.optimize import minimize

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


class QuadrotorKinodynamicRRTStar:
    def __init__(self, A, B, R, x_start, x_goal, x_bounds, u_bounds, max_iter=1000, goal_sample_rate=0.2):
        self.A = A  # Dynamics matrix
        self.B = B  # Control input matrix
        self.R = R  # Cost weight matrix
        self.x_start = np.array(x_start)
        self.x_goal = np.array(x_goal)
        self.x_bounds = x_bounds
        self.u_bounds = u_bounds
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate

        # Tree and costs
        self.tree = [self.x_start]
        self.parents = {tuple(self.x_start): None}
        self.costs = {tuple(self.x_start): 0.0}

    def cost_function(self, trajectory_duration, control_input):
        return trajectory_duration + control_input.T @ self.R @ control_input

    def sample_random_state(self):
        if np.random.rand() < self.goal_sample_rate:
            return self.x_goal
        return np.random.uniform(self.x_bounds[:, 0], self.x_bounds[:, 1])

    def find_nearest_neighbor(self, x_rand):
        distances = [np.linalg.norm(x_rand - x) for x in self.tree]
        return self.tree[np.argmin(distances)]

    def find_optimal_time(self, x0, x1):
        def objective(t):
            G = self.compute_gramian(t)
            x_bar = expm(self.A * t) @ x0
            delta = x1 - x_bar
            reg_term = 1e-6 * np.eye(G.shape[0])  # Add regularization
            G = G + reg_term
            return t + delta.T @ np.linalg.inv(G) @ delta

        result = minimize(objective, x0=1.0, bounds=[(0.5, 10)])  # Adjust bounds
        if not result.success:
            print(f"Optimization failed: {result.message}")
        return result.x[0] if result.success else None

    def compute_gramian(self, t):
        G = expm(self.A * t) @ self.B @ np.linalg.inv(self.R) @ self.B.T @ expm(self.A.T * t)
        reg_term = 1e-3 * np.eye(G.shape[0])  # Add regularization to avoid singularity
        return G + reg_term


    def optimal_trajectory(self, x0, x1):
        t_opt = self.find_optimal_time(x0, x1)
        if t_opt is None:
            print("Falling back to heuristic time estimation.")
            t_opt = self.fallback_time_estimation(x0, x1)

        G = self.compute_gramian(t_opt)
        x_bar = expm(self.A * t_opt) @ x0
        delta = x1 - x_bar
        G += 1e-6 * np.eye(G.shape[0])  # Regularization
        d_tau = np.linalg.inv(G) @ delta
        return d_tau, t_opt
    
    def fallback_time_estimation(self, x0, x1):
        delta = np.linalg.norm(x1[:3] - x0[:3])
        return delta / 5.0


    def simulate_dynamics(self, x0, u, t):
        return expm(self.A * t) @ x0 + expm(self.A * t) @ self.B @ u

    def is_valid_state(self, x):
        return np.all((self.x_bounds[:, 0] <= x) & (x <= self.x_bounds[:, 1]))

    def expand_tree(self):
        for _ in range(self.max_iter):
            x_rand = self.sample_random_state()
            nearest_node = self.find_nearest_neighbor(x_rand)
            u_opt, t_opt = self.optimal_trajectory(nearest_node, x_rand)

            if u_opt is None or t_opt is None:
                continue

            x_new = self.simulate_dynamics(nearest_node, u_opt, t_opt)
            if not self.is_valid_state(x_new):
                continue

            self.tree.append(x_new)
            self.parents[tuple(x_new)] = nearest_node
            self.costs[tuple(x_new)] = self.costs[tuple(nearest_node)] + self.cost_function(t_opt, u_opt)

            if np.linalg.norm(x_new[:3] - self.x_goal[:3]) < 0.1:  # Goal proximity
                print("Goal reached!")
                return True
        print("Failed to find a trajectory.")
        return False

    def extract_path(self):
        path = [self.x_goal]
        current = self.x_goal
        while current is not None:
            path.append(current)
            current = self.parents.get(tuple(current))
        return path[::-1]

    def plan(self):
        if self.expand_tree():
            return self.extract_path()
        return None


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize Simulation #################################
    INIT_XYZS = np.array([[0, 0, 1]])
    INIT_RPYS = np.array([[0, 0, 0]])

    #### Define Quadrotor Dynamics ############################
    m = 0.027
    I_x, I_y, I_z = 1.4e-5, 1.4e-5, 2.2e-5
    g = 9.81

    A = np.block([
        [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
        [np.zeros((3, 3)), np.array([[0, g, 0], [-g, 0, 0], [0, 0, 0]]), np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
    ])

    B = np.block([
        [np.zeros((3, 4))],
        [np.zeros((3, 4))],
        [np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1 / m, 0, 0, 0]])],
        [np.array([[0, 1 / I_x, 0, 0], [0, 0, 1 / I_y, 0], [0, 0, 0, 1 / I_z]])]
    ])

    R = np.eye(4)
    x_start = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_goal = np.array([5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_bounds = np.array([[-10, 10], [-10, 10], [0, 10], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
                         [-5, 5], [-5, 5], [-5, 5], [-10, 10], [-10, 10], [-10, 10]])
    u_bounds = np.array([[-1, 1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]])

    #### Plan Trajectory Using RRT* ###########################
    planner = QuadrotorKinodynamicRRTStar(A, B, R, x_start, x_goal, x_bounds, u_bounds, max_iter=1000, goal_sample_rate=0.3)
    waypoints = planner.plan()
    if waypoints is None:
        print("No trajectory found!")
        return

    #### Initialize Simulation Environment ####################
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui
                     )

    #### Follow Waypoints #####################################
    TARGET_POS = np.array([wp[:3] for wp in waypoints])
    ctrl = DSLPIDControl(drone_model=drone)

    for i in range(len(TARGET_POS)):
        target = TARGET_POS[i]
        action, _, _ = ctrl.computeControlFromState(
            control_timestep=1 / control_freq_hz,
            state=np.zeros(12),
            target_pos=target
        )
        obs, _, _, _, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RRT* with PyBullet Drones')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int)
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool)
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool)
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool)
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool)
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool)
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int)
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int)
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int)
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool)
    ARGS = parser.parse_args()

    run(**vars(ARGS))
