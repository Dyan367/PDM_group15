import numpy as np
import pybullet as p
import logging

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from environments.shapes import create_box_shape, create_cylinder_shape, add_bounding_box, move_shape_dynamic, move_shape_reset, move_shape_random

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
        shelve_size = [0.1, 0.1, 4.0]
        shelve_positions = [4, 4, shelve_size[2] / 2]
        
        for pos in shelve_positions:
            pillar_id = create_box_shape(
                size=shelve_size,
                color=[0.6, 0.4, 0.2, 1],  # Brown box
                client_id=self.CLIENT
            )
            p.resetBasePositionAndOrientation(pillar_id, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
            self.obstacle_ids.append(pillar_id)


    def initialize_planning(self):
        start_pos = np.copy(self.pos[0])
        goal_pos = np.array([-2.0, 0.0, 1.0])

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
