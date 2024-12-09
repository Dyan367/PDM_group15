import os
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.MPCAviary import MPCAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class StaticFactory(MPCAviary):


    def __init__(self, dt = 0.01,obstacle_config={}, seed=42, **kwargs):
        self.obstacle_config = obstacle_config
        self.seed = seed
        self.dt = dt ## dt for discretization of state space
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
        """
        Add small spheres as obstacles at random positions within the arena.
        """
        num_obstacles = self.obstacle_config.get('num_obstacles', 5)
        obstacle_radius = self.obstacle_config.get('obstacle_radius', 0.2)  # Radius of the sphere
        arena_size = self.obstacle_config.get('arena_size', 5.0)

        self.obstacle_ids = []
        drone_start_pos = self.INIT_XYZS[0]

        for _ in range(num_obstacles):
            while True:
                # Randomize position within the arena boundaries
                x = np.random.uniform(-arena_size / 2, arena_size / 2)
                y = np.random.uniform(-arena_size / 2, arena_size / 2)
                z = np.random.uniform(0.5, 2.5)  

                # Ensure the sphere is not too close to the drone start position
                distance = np.linalg.norm(np.array([x, y]) - drone_start_pos[:2])
                if distance <= obstacle_radius * 2:  # Avoid too-close placement
                    continue

                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=obstacle_radius,
                    physicsClientId=self.CLIENT
                )
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=obstacle_radius,
                    rgbaColor=[1.0, 0.0, 0.0, 1],  # Sphere color
                    physicsClientId=self.CLIENT
                )
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, z],
                    physicsClientId=self.CLIENT
                )

                # Add the sphere to the list of obstacles and break out of the loop
                self.obstacle_ids.append(obstacle_id)
                break



        
