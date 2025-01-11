import os
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from environments.shapes import create_box_shape

class StaticFactory(VelocityAviary):


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
        
        self.obstacle_config = obstacle_config
        self.seed = seed
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
                         output_folder=output_folder)
        

    def reset(self):
        obs, info = super().reset(seed=self.seed)

        if hasattr(self, 'obstacle_ids'):
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)

        self._addObstacles()
        print(self.obstacle_ids)

        return obs, info

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
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 0]
            ],
            # layer 3
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1],
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

        
