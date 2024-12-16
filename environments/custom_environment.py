import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from environments.shapes import create_box_shape, create_sphere_shape, move_shape_dynamic

class StaticFactory(VelocityAviary):
    def __init__(self, obstacle_config={}, seed=42, **kwargs):
        self.obstacle_config = obstacle_config
        self.seed = seed
        self.obstacle_ids = []
        self.moving_bodies = [] 
        super().__init__(**kwargs)

    def reset(self):
        obs, info = super().reset(seed=self.seed)

        # Remove existing obstacles if they exist
        if hasattr(self, 'obstacle_ids'):
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)

        self._addObstacles()
        print(self.obstacle_ids)

        return obs, info

    def _addObstacles(self):
        obstacle_size = self.obstacle_config.get('obstacle_size', [0.5, 0.5, 3.0])
        sphere_radius = self.obstacle_config.get('sphere_radius', 1.0)
        moving_sphere_radius_1 = self.obstacle_config.get('moving_sphere_radius_1', 0.5)
        moving_sphere_radius_2 = self.obstacle_config.get('moving_sphere_radius_2', 0.75) 
        arena_size = self.obstacle_config.get('arena_size', 10.0)

        self.obstacle_ids = []
        self.moving_bodies = []

        # Add two static pillars
        pillar_positions = [
            [arena_size / 16, arena_size / 4, obstacle_size[2] / 2],
            [-arena_size / 4, arena_size / 4, obstacle_size[2] / 2]
        ]

        for pos in pillar_positions:
            pillar_id = create_box_shape(
                size=[obstacle_size[0] / 2, obstacle_size[1] / 2, obstacle_size[2] / 2],
                color=[0.6, 0.4, 0.2, 1],  # Brown color
                client_id=self.CLIENT
            )
            p.resetBasePositionAndOrientation(
                pillar_id,
                pos,
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )
            self.obstacle_ids.append(pillar_id)

        # Add a static sphere
        sphere_position = [arena_size / 4, arena_size / 4, sphere_radius]
        sphere_id = create_sphere_shape(
            radius=sphere_radius,
            color=[0.2, 0.6, 0.8, 1],  # Blue color
            client_id=self.CLIENT
        )
        p.resetBasePositionAndOrientation(
            sphere_id,
            sphere_position,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )
        self.obstacle_ids.append(sphere_id)

        # Add multiple moving spheres with different radii, velocities, and bounds
        moving_bodies_config = [
            # First moving sphere
            {
                "type": "sphere",
                "radius": moving_sphere_radius_1,
                "initial_position": [0, -arena_size / 8, moving_sphere_radius_1],
                "velocity": [0.2, -0.4, 0.0],
                "bounds": [-2.0, 2.0, -2.0, 2.0, 0.5, 2.5]
            },
            # Second moving sphere
            {
                "type": "sphere",
                "radius": moving_sphere_radius_2,
                "initial_position": [0, arena_size / 8, moving_sphere_radius_2],
                "velocity": [-0.3, 0.5, 0.0],
                "bounds": [-2.5, 2.5, -2.5, 2.5, 0.5, 2.5]
            }
        ]

        for body_config in moving_bodies_config:
            if body_config["type"] == "sphere":
                body_id = create_sphere_shape(
                    radius=body_config["radius"],
                    color=[0, 1, 0, 1],  # Green color for moving spheres
                    client_id=self.CLIENT
                )

            p.resetBasePositionAndOrientation(
                body_id,
                body_config["initial_position"],
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )
            self.obstacle_ids.append(body_id)
            self.moving_bodies.append({
                "id": body_id,
                "velocity": body_config["velocity"],
                "bounds": body_config["bounds"]
            })

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Move each body dynamically
        for body in self.moving_bodies:
            body["velocity"][0], body["velocity"][1], body["velocity"][2] = move_shape_dynamic(
                body["id"],
                body["velocity"][0],
                body["velocity"][1],
                body["velocity"][2],
                body["bounds"],
                self.CTRL_TIMESTEP,
                client_id=self.CLIENT
            )

        return obs, reward, terminated, truncated, info


def create_env(duration_sec=50, simulation_freq_hz=240, control_freq_hz=48, gui=True):
    """
    Parameters:
    - duration_sec: Duration of the simulation in seconds.
    - simulation_freq_hz: Frequency of the simulation steps.
    - control_freq_hz: Frequency of the control updates.
    - gui: Boolean to enable or disable GUI.

    Returns:
    - env: The initialized StaticFactory environment.
    - num_steps: Total number of control steps.
    """
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
        obstacle_config={
            'obstacle_size': [2, 0.5, 10.0],
            'sphere_radius': 1.0,
            'arena_size': 10.0,
            'moving_sphere_radius_1': 0.5,
            'moving_sphere_radius_2': 0.75
        },
        seed=40
    )
    return env, num_steps
