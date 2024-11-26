import numpy as np
import pybullet as p
import pybullet_data
import time
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class StaticFactory(BaseAviary):

    def __init__(self, obstacle_config={}, seed=42, **kwargs):
        self.obstacle_config = obstacle_config
        self.seed = seed
        
        super().__init__(**kwargs)

    def reset(self):
        obs, info = super().reset(seed=42)

        time.sleep(1)

        if hasattr(self, 'obstacle_ids'):
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)


        self._addObstacles()

        return obs, info

    def _addObstacles(self):
        num_obstacles = self.obstacle_config.get('num_obstacles', 5)
        obstacle_size = self.obstacle_config.get('obstacle_size', [0.5, 0.5, 0.5])
        arena_size = self.obstacle_config.get('arena_size', 5.0)

        self.obstacle_ids = []
        drone_start_pos = self.INIT_XYZS[0]  

        for _ in range(num_obstacles):
            while True:
                x = np.random.uniform(-arena_size, arena_size)
                y = np.random.uniform(-arena_size, arena_size)
                z = obstacle_size[2] / 2


                distance = np.linalg.norm(np.array([x, y]) - drone_start_pos[:2])
                if distance > obstacle_size[0]: 
                    break  

            size_x = np.random.uniform(0.1, obstacle_size[0])
            size_y = np.random.uniform(0.1, obstacle_size[1])
            size_z = obstacle_size[2]

            collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                        halfExtents=[size_x / 2, size_y / 2, size_z / 2],
                                                        physicsClientId=self.CLIENT)
            visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[size_x / 2, size_y / 2, size_z / 2],
                                                rgbaColor=[1, 0, 0, 1],
                                                physicsClientId=self.CLIENT)
            obstacle_id = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=collision_shape,
                                            baseVisualShapeIndex=visual_shape,
                                            basePosition=[x, y, z],
                                            physicsClientId=self.CLIENT)
            self.obstacle_ids.append(obstacle_id)

    def _actionSpace(self):
        """Defines the action space."""
        act_lower_bound = np.array([-1, -1, -1, -1])
        act_upper_bound = np.array([1, 1, 1, 1])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """Defines the observation space."""
        obs_lower_bound = np.array([-np.inf] * 12)  
        obs_upper_bound = np.array([np.inf] * 12)
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Computes the current observation."""
        obs = np.hstack([self.pos[0],
                         self.vel[0],
                         self.rpy[0],
                         self.ang_v[0]])
        return obs

    def _preprocessAction(self, action):
        """Converts action into motor RPMs."""
        rpm = self._normalizedActionToRPM(action)
        return rpm


    def _computeTerminated(self):
        """Checks if the episode should terminate."""
        # Terminate if the drone crashes or flies out of bounds
        pos = self.pos[0]
        if pos[2] < 0.0 or pos[2] > 10.0:
            return True
        return False

    def _computeTruncated(self):
        """Checks if the episode should be truncated."""
        # Truncate after a fixed number of steps
        max_steps = 2400  # 10 seconds at 240 Hz
        return self.step_counter >= max_steps

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        return 
    
    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        return