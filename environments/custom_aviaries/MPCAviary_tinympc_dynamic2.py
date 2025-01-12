import os
import numpy as np
import pybullet as p
from gymnasium import spaces
import logging
import time
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from environments.shapes import create_box_shape

import tinympc


class MPCAviaryDynamicTinyMPC(BaseAviary):
    """
    An extension of BaseAviary that uses TinyMPC for the MPC, 
    ensuring arrays are fortran-contiguous and dimensions match TinyMPC's expectations.
    """

    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 neighbourhood_radius=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 ctrl_freq=240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results',
                 mpc_params=None,
                 x_target=None,
                 obstacle_config={},
                 seed=42):

        self.moving_bodies = []
        self.crane = None
        self.crane_velocity = 5.0

        self.x_target_config = x_target
        self.obstacle_config = obstacle_config
        self.seed = seed
        self.obstacles = obstacles
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=False,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=vision_attributes,
                         output_folder=output_folder)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if self.NUM_DRONES != 1:
            raise NotImplementedError("MPCAviary currently supports only a single drone.")

        np.random.seed(self.seed)

        if mpc_params is None:
            mpc_params = {
                'dt': 0.1,
                'N': 10,
                'sim_time': 100,
                'proximity_threshold': 0.01
            }

        self.dt = mpc_params['dt']
        self.N = mpc_params['N']
        self.sim_time = mpc_params['sim_time']
        self.proximity_threshold = mpc_params['proximity_threshold']
        self.max_steps = int(self.sim_time / self.dt)

        # Build A,B,c,Q,R, etc.
        self._initialize_mpc_matrices()

        if self.x_target_config is not None:
            if self.x_target_config.shape != (12,):
                raise ValueError("x_target must be a 12-dimensional vector.")
            self.x_target = self.x_target_config
        else:
            self.x_target = np.array([2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.state_history = np.zeros((self.max_steps + 1, 12))
        self.control_history = np.zeros((self.max_steps, 4))

        initial_state = self._get_current_state()
        self.state_history[0, :] = initial_state
        logging.info(f"Target set to: {self.x_target}")

        self.obstacle_ids = []

        # ---- Setup TinyMPC properly ----
        self._tinympc_setup()

    def _tinympc_setup(self):
        # Convert everything to Fortran as you did
        A_f = np.asfortranarray(self.A, dtype=np.float64)
        B_f = np.asfortranarray(self.B_full, dtype=np.float64)
        Q_f = np.asfortranarray(self.Q, dtype=np.float64)
        R_f = np.asfortranarray(self.R, dtype=np.float64)

        # Some large (±1000) bounds for states and inputs
        nx = A_f.shape[0]  # 12
        nu = B_f.shape[1]  # 4
        x_min_f = -1e3 * np.ones(nx, dtype=np.float64)
        x_max_f = +1e3 * np.ones(nx, dtype=np.float64)
        u_min_f = -1e3 * np.ones(nu, dtype=np.float64)
        u_max_f = +1e3 * np.ones(nu, dtype=np.float64)

        self.tinympc_prob = tinympc.TinyMPC()

        # For the new signature, just do:
        # setup(A, B, Q, R, N, x_min=None, x_max=None, u_min=None, u_max=None, xf_min=None, xf_max=None, settings=None)
        self.tinympc_prob.setup(
            A_f, B_f, Q_f, R_f,
            self.N,  # horizon length
            x_min=x_min_f,
            x_max=x_max_f,
            u_min=u_min_f,
            u_max=u_max_f,
            xf_min=None,  # final-state bounds, if you want
            xf_max=None
        )

    def set_target(self, new_target):
        if new_target.shape != (12,):
            raise ValueError("new_target must be a 12-dimensional vector.")
        self.x_target = new_target
        logging.info(f"Target updated to: {self.x_target}")

    def _initialize_mpc_matrices(self):
        mass = self.M
        Ixx, Iyy, Izz = self.J[0, 0], self.J[1, 1], self.J[2, 2]
        g = self.G

        d_x = 9.1785e-7
        d_y = 9.1785e-7
        d_z = 10.311e-7

        A = np.eye(12)
        A[0, 3] = self.dt
        A[1, 4] = self.dt
        A[2, 5] = self.dt

        # Some scaled example
        A[3, 7] = (self.dt / mass)
        A[4, 6] = -(self.dt / mass)
        A[3, 3] = 1 - d_x * self.dt
        A[4, 4] = 1 - d_y * self.dt
        A[5, 5] = 1 - d_z * self.dt

        A[6, 9] = self.dt
        A[7, 10] = self.dt
        A[8, 11] = self.dt

        B_full = np.zeros((12, 4))
        B_full[5, 0] = self.dt / mass
        B_full[9, 1] = self.dt / Ixx
        B_full[10, 2] = self.dt / Iyy
        B_full[11, 3] = self.dt / Izz

        c = np.zeros(12)

        c[5] = -self.dt * g

        Q = np.diag([
            2000, 2000, 3000,
            100, 100, 500,
            5, 5, 5,
            1, 1, 1
        ])
        R = np.diag([1.0, 0.2, 0.2, 0.2])

        u_min = np.array([0, -np.pi / 3, -np.pi / 3, -np.pi / 3])
        u_max = np.array([20, np.pi / 3, np.pi / 3, np.pi / 3])

        self.A = A
        self.B_full = B_full
        self.c = c
        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max

    def _get_current_state(self):
        drone_id = self.DRONE_IDS[0]
        pos, orn = p.getBasePositionAndOrientation(drone_id)
        linear_vel, angular_vel = p.getBaseVelocity(drone_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        state = np.array([
            pos[0], pos[1], pos[2],
            linear_vel[0], linear_vel[1], linear_vel[2],
            roll, pitch, yaw,
            angular_vel[0], angular_vel[1], angular_vel[2]
        ])
        return state

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        initial_state = self._get_current_state()
        self.state_history[0, :] = initial_state

        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []

        if self.obstacle_config and self.obstacles:
            self._addObstacles()

        return initial_state, info

    def _create_environment(self, environment_grid, cell_size=1.0, wall_thickness=0.2, wall_height=2.0):
        """
        Creates an environment based on a 3D grid without optimization for continuous lines.

        Parameters:
        - environment_grid: 3D grid representing obstacles at different heights.
        - cell_size: The size of each cell in the grid.
        - wall_thickness: Thickness of the walls/obstacles.
        - wall_height: Height of the walls/obstacles.
        """
        self.obstacle_ids = []

        for z, layer in enumerate(environment_grid):
            if not isinstance(layer, list):
                raise TypeError(f"Expected layer to be a list, got {type(layer)}")

            rows = len(layer)
            cols = len(layer[0])  # Ensure each row is a list

            for row in range(rows):
                for col in range(cols):
                    cell = layer[row][col]
                    if cell != 0:
                        self._create_obstacle(
                            row=row, col=col, z=z, cell_type=cell, cell_size=cell_size,
                            wall_thickness=wall_thickness, wall_height=wall_height
                        )

        print(f"Environment created with {len(self.obstacle_ids)} obstacles.")

    def _create_obstacle(self, row, col, z, cell_type, cell_size, wall_thickness, wall_height):
        """
        Creates a single obstacle for a cell.

        Parameters:
        - row: The row index of the cell.
        - col: The column index of the cell.
        - z: The height layer index.
        - cell_type: The type of the cell (e.g., 1, 2, 3).
        - cell_size: The size of each cell.
        - wall_thickness: Thickness of the walls/obstacles.
        - wall_height: Height of the walls/obstacles.
        """
        x_center = col * cell_size
        y_center = row * cell_size
        z_center = z * cell_size + wall_height / 2

        # Define box dimensions and color based on cell type
        size = [cell_size / 2, cell_size / 2, wall_thickness / 2]
        color = [0.9, 0.9, 0.9, 0.5]  # Default color

        if cell_type == 1:  # Thin walls with configurable height
            size[2] = wall_height / 2
        elif cell_type == 0:  # Regular cubes
            return
        # Create the obstacle
        obstacle_id = create_box_shape(size=size, color=color, client_id=self.CLIENT)
        p.resetBasePositionAndOrientation(
            obstacle_id, [x_center, y_center, z_center], [0, 0, 0, 1], physicsClientId=self.CLIENT
        )
        self.obstacle_ids.append(obstacle_id)

    def _addObstacles(self):
        """
        Overrides the `_addObstacles` method to create an environment based on a grid layout.
        Combines adjacent cubes into larger rectangular cuboids when possible.
        """
        self.obstacle_ids = []

        # # Define the environment grid
        # environment_grid = [
        #     # LAYER 1
        #     [
        #         [0, 0, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 0, 1],
        #         [1, 0, 0, 0, 1, 1, 0, 1],
        #         [1, 0, 1, 1, 1, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #     # roof layer 1
        #     [
        #         [0, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 0, 0, 1, 1, 1, 1],
        #         [1, 1, 0, 0, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #     #layer 2
        #     [
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 1, 0, 1, 0, 1, 1],
        #         [1, 0, 0, 0, 1, 0, 0, 1],
        #         [1, 1, 0, 0, 1, 1, 0, 1],
        #         [1, 1, 1, 1, 1, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #
        #     # roof layer 2
        #     [
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 0, 0, 0, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #     # layer 3
        #     [
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 0, 1, 0, 1, 0, 0, 1],
        #         [1, 0, 1, 0, 1, 0, 1, 1],
        #         [1, 0, 1, 0, 0, 0, 1, 1],
        #         [1, 0, 1, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 1, 0, 0, 1],
        #         [1, 0, 0, 0, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 0, 0]
        #     ],
        #
        #     # roof layer 3
        #     [
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 0, 0]
        #     ],
        #     # LAST LAYER (empty)
        #     [
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0]
        #     ]
        #     # END
        # ]

        ########### edges from roof layers removed to same ~80 blocks

        # Define the environment grid
        environment_grid = [
            # LAYER 1
            [
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ],
            # roof layer 1
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ],
            # layer 2
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ],

            # roof layer 2
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 0]
            ],
            # layer 3
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0]
            ],

            # roof layer 3
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ],
            # LAST LAYER (empty)
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]
            # END
        ]
################ hollow box
        # environment_grid = [
        #     # LAYER 1
        #     [
        #         [0, 0, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #     [
        #         [0, 0, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #     # layer 2
        #     [
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #     [
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #     [
        #         [0, 0, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1]
        #     ],
        #
        #     # roof layer 3
        #     [
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 0, 0]
        #     ],
        #     # LAST LAYER (empty)
        #     [
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0]
        #     ]
        #     # END
        # ]

        wall_thickness = self.obstacle_config.get('wall_thickness', 1.0)
        wall_height = self.obstacle_config.get('wall_height', 2.0)

        # Create the environment based on the grid
        self._create_environment(
            environment_grid=environment_grid,
            wall_thickness=wall_thickness,
            wall_height=wall_height
        )


    def step(self, action=None):

        t = self.step_counter // self.PYB_STEPS_PER_CTRL
        if t >= self.max_steps:
            terminated = True
            truncated = False
            reward = 0.0
            info = self._computeInfo()
            return self._computeObs(), reward, terminated, truncated, info

        # # Update moving shapes
        # timestep = 1 / self.PYB_FREQ
        # for body_id, vel_x, vel_y, vel_z, bounds, reset_position in self.moving_bodies:
        #     move_shape_reset(
        #         body_id, vel_x, vel_y, vel_z,
        #         bounds, timestep, reset_position,
        #         client_id=self.CLIENT
        #     )

        # self.crane_velocity_x, _, _ = move_shape_dynamic(
        #     self.crane,
        #     self.crane_velocity_x, 0, 0,
        #     self.crane_bounds, timestep,
        #     self.CLIENT
        # )

        # for person in self.people:
        #     person_id = person["id"]
        #     bounds = person["bounds"]
        #     max_speed = person["max_speed"]
        #     change_interval = person["change_interval"]

        #     move_shape_random(
        #         obstacle_id=person_id,
        #         bounds=bounds,
        #         max_speed=max_speed,
        #         timestep=timestep,
        #         change_interval=change_interval,
        #         client_id=self.CLIENT
        #     )

        current_state = self._get_current_state()
        self.state_history[t, :] = current_state

        u_opt = self._solve_mpc(current_state)
        self.control_history[t, :] = u_opt

        # Apply
        thrust_z, torque_x, torque_y, torque_z = u_opt
        # print(
        #     f"Time {t * self.dt:.1f}s - Control: Tz={thrust_z:.2f}, Tx={torque_x:.2f}, Ty={torque_y:.2f}, Tz={torque_z:.2f}")
        self._apply_control_inputs(thrust_z, torque_x, torque_y, torque_z)

        # for _ in range(self.PYB_STEPS_PER_CTRL):
        #     p.stepSimulation(physicsClientId=self.CLIENT)
        #     print(self.PYB_STEPS_PER_CTRL)
        #     print(self.PYB_TIMESTEP)
        #     time.sleep(self.PYB_TIMESTEP)

        for _ in range(self.PYB_STEPS_PER_CTRL):  # now 240
            p.stepSimulation()
            time.sleep(1 / 60)  # 1/240

        self._updateAndStoreKinematicInformation()

        if t + 1 <= self.max_steps:
            self.state_history[t + 1, :] = self._get_current_state()

        distance = np.linalg.norm(current_state[:3] - self.x_target[:3])
        if distance < self.proximity_threshold:
            terminated = False
            truncated = False
            logging.info(f"Target reached at step {t}, time {t * self.dt:.1f} s.")
        else:
            terminated = False
            truncated = False

        reward = -distance
        self.step_counter += self.PYB_STEPS_PER_CTRL
        return self._computeObs(), reward, terminated, truncated, self._computeInfo()

    def _solve_mpc(self, current_state):
        """
        TinyMPC solve step. We'll store the 'error' e = x - x_target, 
        run the solver, clamp solution, apply +c in sim externally.
        """
        e_current = current_state - self.x_target
        self.tinympc_prob.set_x0(e_current)
        solution = self.tinympc_prob.solve()
        if solution is None:
            logging.warning("TinyMPC returned None. Using zero control.")
            u_opt = np.zeros(4)
        else:
            u_opt = solution["controls"]

        # clamp to the environment's bounds
        u_opt = np.clip(u_opt, self.u_min, self.u_max)
        return u_opt

    def _apply_control_inputs(self, thrust_z, torque_x, torque_y, torque_z):
        thrust_body = np.array([0, 0, thrust_z])
        p.applyExternalForce(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,
            forceObj=thrust_body.tolist(),
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME
        )
        torque = np.array([torque_x, torque_y, torque_z])
        p.applyExternalTorque(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,
            torqueObj=torque.tolist(),
            flags=p.LINK_FRAME
        )
    

    def close(self):
        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []
            logging.info("Removed all obstacles from the environment.")
        super().close()

    def render(self, mode='human', close=False):
        super().render(mode=mode, close=close)

    def _actionSpace(self):
        return 0

    def _observationSpace(self):
        return 0

    def _computeObs(self):
        return self._get_current_state()

    def _preprocessAction(self, action):
        return np.zeros(4)

    def _computeReward(self):
        return -1

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {"answer": 42}

    def plot_results(self):
        steps = min(self.step_counter // self.PYB_STEPS_PER_CTRL, self.max_steps)
        time_array = np.linspace(0, steps * self.dt, steps + 1)

        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(self.state_history[:steps + 1, 0],
                self.state_history[:steps + 1, 1],
                self.state_history[:steps + 1, 2], label='Trajectory')
        ax.scatter(self.x_target[0], self.x_target[1], self.x_target[2],
                   color='r', marker='*', s=100, label='Target')
        ax.set_title('Trajectory 3D')
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)

        ax2 = fig.add_subplot(132)
        ax2.plot(self.state_history[:steps + 1, 0],
                 self.state_history[:steps + 1, 1], 'b-', label='XY Path')
        ax2.plot(self.x_target[0], self.x_target[1], 'ro', label='Target')
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax2.set_title('Top view')
        ax2.grid(True)
        ax2.axis('equal')
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.plot(time_array, self.state_history[:steps + 1, 6], label='Roll')
        ax3.plot(time_array, self.state_history[:steps + 1, 7], label='Pitch')
        ax3.plot(time_array, self.state_history[:steps + 1, 8], label='Yaw')
        ax3.set_title('Orientation (rad)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (rad)')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot inputs
        plt.figure(figsize=(14, 6))
        time_ctrl = np.linspace(0, steps * self.dt, steps)
        plt.subplot(2, 1, 1)
        plt.plot(time_ctrl, self.control_history[:steps, 0], label='Thrust Z')
        plt.title('Control Inputs Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_ctrl, self.control_history[:steps, 1], label='Torque X')
        plt.plot(time_ctrl, self.control_history[:steps, 2], label='Torque Y')
        plt.plot(time_ctrl, self.control_history[:steps, 3], label='Torque Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()