import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from environments.shapes import create_box_shape

class StaticFactory(VelocityAviary):
    def __init__(self, obstacle_config={}, seed=42, **kwargs):
        self.obstacle_config = obstacle_config
        self.seed = seed
        self.obstacle_ids = []
        self.moving_bodies = []
        self.crane = None
        self.crane_velocity = 5.0
        super().__init__(**kwargs)

    def reset(self):
        obs, info = super().reset(seed=self.seed)

        # Remove all existing obstacles
        for body_id in self.obstacle_ids + ([self.crane] if self.crane else []):
            p.removeBody(body_id, physicsClientId=self.CLIENT)

        self._addObstacles()
        return obs, info

    def _addObstacles(self):
        alpha = 1
        height = 2.0
        width = 0.2

        # Wall configurations
        wallx2 = self.obstacle_config.get('wall_x2', [1.4, width, height * 2])
        wallx5 = self.obstacle_config.get('wall_x5', [2.5, width, height * 2])
        wallx7 = self.obstacle_config.get('wall_x7', [3.8, width, height * 2])
        wallx72 = self.obstacle_config.get('wall_x72', [wallx7[0] * 2, width, height * 2])
        wally2 = self.obstacle_config.get('wall_y2', [width, 1.4, height * 2])
        wally22 = self.obstacle_config.get('wall_y22', [width, wally2[1] * 2 - width, height * 2])

        # Define wall sizes
        wall_sizes = {
            'wall_x2': wallx2,
            'wall_x5': wallx5,
            'wall_x7': wallx7,
            'wall_x72': wallx72,
            'wall_y2': wally2,
            'wall_y22': wally22
        }

        # Define wall positions
        wall_positions = {
            'wall_x2': [[wallx2[0] + wallx5[0] * 3 + wallx7[0] - width * 1.5, wally2[1] - width, height]],
            'wall_x5': [
                [wallx5[0] / 2, wally2[1] - width, height],
                [wallx5[0] / 2, -wally2[1] + width, height],
                [wallx5[0] / 2 + width * 0.5, 3 * (wally2[1] - width), height],
                [wallx72[0] + wallx2[0] - width, 3 * (wally2[1] - width), height],
                [wallx5[0] * 2 + wallx7[0] + width / 1.5, -wally2[1] + width, height]
            ],
            'wall_x7': [[wallx5[0] * (4 / 3) + wallx7[0] + width / 1.5, -wallx5[0] - wally2[1] + width * 1.5, height]],
            'wall_x72': [[wallx5[0] - width * 2 + wallx2[0] * 3, 4 * (wallx2[0] - width) + wallx2[0] - width, height]],
            'wall_y2': [
                [-wallx5[0] / 2 - width / 2, 0, height],
                [wallx5[0] * 2 + (-wallx5[0] / 2 - width / 1.5), -wallx5[0] + width / 2, height],
                [wallx5[0] * 2 + (-wallx5[0] / 2 - width / 1.5) + wallx7[0] * 2, -wallx5[0] + width / 2, height],
                [wallx5[0] * 2 + (-wallx5[0] / 2 - width / 1.5), wallx5[0] - width / 2, height],
                [-wallx5[0] / 2 - width / 2, 4 * (wallx2[0] - width), height],
                [wallx5[0] * 3 + wallx7[0] - width, 2.4, height]
            ],
            'wall_y22': [
                [wallx5[0] * 2 + (-wallx5[0] / 2 - width / 1.5) + wally2[1] * 2, -wally2[1] + wally22[1], height],
                [wallx2[0] * 2 + wallx5[0] * 3 + wallx7[0] - width * 2, 3 * (wally2[1] - width), height]
            ]
        }

        # Create obstacles
        self.obstacle_ids = []
        for wall_name, positions in wall_positions.items():
            size = wall_sizes[wall_name]
            for pos in positions:
                wall_id = create_box_shape(
                    size=size,
                    color=[0.6, 0.4, 0.2, alpha],  # Brown box
                    client_id=self.CLIENT
                )
                p.resetBasePositionAndOrientation(wall_id, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
                self.obstacle_ids.append(wall_id)


    def initialize_planning(self):
        start_pos = np.array([2.0, 0.0, 1.0])  # Starting position of the drone
        goal_pos = np.array([13.1, 2.4, 1.0])  # Example goal position

        obstacles = []
        for obs_id in self.obstacle_ids:
            aabb_min, aabb_max = p.getAABB(obs_id, physicsClientId=self.CLIENT)
            pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.CLIENT)
            size = np.array(aabb_max) - np.array(aabb_min)
            obstacles.append({
                'position': np.array(pos),
                'size': np.array(size),
                'aabb_min': np.array(aabb_min),
                'aabb_max': np.array(aabb_max),
            })

        x_range = [-1, 50]
        y_range = [-6, 6]
        z_range = [0.5, 3.0]

        return start_pos, goal_pos, obstacles, (x_range, y_range, z_range)

def create_env(duration_sec=50, simulation_freq_hz=240, control_freq_hz=48, gui=True):
    num_steps = int(duration_sec * control_freq_hz)

    env = StaticFactory(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.PYB,
        neighbourhood_radius=np.inf,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=True,
        user_debug_gui=False,
        seed=40
    )
    return env, num_steps
