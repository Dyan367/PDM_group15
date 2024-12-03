import os
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class StaticFactory(VelocityAviary):


    def __init__(self, obstacle_config={}, seed=42, **kwargs):
        self.obstacle_config = obstacle_config
        self.seed = seed
        super().__init__(**kwargs)

    def reset(self):
        obs, info = super().reset(seed=self.seed)

        if hasattr(self, 'obstacle_ids'):
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)

        self._addObstacles()
        print(self.obstacle_ids)

        return obs, info

    def _addObstacles(self):
        num_obstacles = self.obstacle_config.get('num_obstacles', 5)
        obstacle_size = self.obstacle_config.get('obstacle_size', [0.5, 0.5, 0.5])
        arena_size = self.obstacle_config.get('arena_size', 5.0)

        self.obstacle_ids = []
        drone_start_pos = self.INIT_XYZS[0]

        rows = int(np.sqrt(num_obstacles))
        cols = rows if rows > 0 else 1  
        shelf_spacing_x = arena_size / cols
        shelf_spacing_y = arena_size / rows

        for i in range(rows):
            for j in range(cols):
                if len(self.obstacle_ids) >= num_obstacles:
                    break

                x = -arena_size / 2 + (j + 0.5) * shelf_spacing_x
                y = -arena_size / 2 + (i + 0.5) * shelf_spacing_y
                z = obstacle_size[2] / 2

                distance = np.linalg.norm(np.array([x, y]) - drone_start_pos[:2])
                if distance <= obstacle_size[0]:
                    continue

                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[obstacle_size[0] / 2, obstacle_size[1] / 2, obstacle_size[2] / 2],
                    physicsClientId=self.CLIENT
                )
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[obstacle_size[0] / 2, obstacle_size[1] / 2, obstacle_size[2] / 2],
                    rgbaColor=[0.6, 0.4, 0.2, 1],
                    physicsClientId=self.CLIENT
                )
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, z],
                    physicsClientId=self.CLIENT
                )
                self.obstacle_ids.append(obstacle_id)
