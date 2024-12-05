import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary


class DynamicFactory(BaseAviary):
    def __init__(self, obstacle_config={}, seed=42, **kwargs):
        self.obstacle_config = obstacle_config
        self.seed = seed
        super().__init__(**kwargs)

    def reset(self):
        obs, info = super().reset(seed=self.seed)

        if hasattr(self, 'obstacle_ids'):
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)

        self._addConveyorBelt()
        return obs, info

    def _addConveyorBelt(self):

        beam_size = [5.0, 0.5, 0.2]
        beam_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[beam_size[0] / 2, beam_size[1] / 2, beam_size[2] / 2],
            physicsClientId=self.CLIENT
        )
        beam_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[beam_size[0] / 2, beam_size[1] / 2, beam_size[2] / 2],
            rgbaColor=[0.6, 0.6, 0.6, 1],
            physicsClientId=self.CLIENT
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=beam_collision_shape,
            baseVisualShapeIndex=beam_visual_shape,
            basePosition=[0, -2, 0.1],
            physicsClientId=self.CLIENT
        )


        box_size = [0.5, 0.5, 0.5]
        box_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[box_size[0] / 2, box_size[1] / 2, box_size[2] / 2],
            physicsClientId=self.CLIENT
        )
        box_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[box_size[0] / 2, box_size[1] / 2, box_size[2] / 2],
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.CLIENT
        )
        self.moving_box_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_collision_shape,
            baseVisualShapeIndex=box_visual_shape,
            basePosition=[-2, -2, (0.2 + (0.5 / 2))],
            physicsClientId=self.CLIENT
        )

        self.box_velocity = 0.4  
        self.box_bounds = [-2.5, 2.5]  

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)


        box_pos, _ = p.getBasePositionAndOrientation(self.moving_box_id, physicsClientId=self.CLIENT)
        new_x = box_pos[0] + self.box_velocity * self.CTRL_TIMESTEP

        if new_x < self.box_bounds[0] or new_x > self.box_bounds[1]:
            self.box_velocity *= -1
            new_x = box_pos[0] + self.box_velocity * self.CTRL_TIMESTEP

        p.resetBasePositionAndOrientation(
            self.moving_box_id,
            [new_x, box_pos[1], box_pos[2]],
            [0, 0, 0, 1],  
            physicsClientId=self.CLIENT
        )

        return obs, reward, terminated, truncated, info


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
        pos = self.pos[0]
        if pos[2] < 0.0 or pos[2] > 10.0:
            return True
        return False

    def _computeTruncated(self):
        """Checks if the episode should be truncated."""
        max_steps = 2400  
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