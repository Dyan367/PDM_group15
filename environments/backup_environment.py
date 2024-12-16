import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from environments.shapes import create_box_shape, create_cylinder_shape, move_shape_reset

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
        return obs, info

    def _addObstacles(self):
        shelve_size = self.obstacle_config.get('shelve_size', [1.0, 1.0, 1.0])
        conveyor_size = self.obstacle_config.get('conveyor_size', [1.0, 1.0, 1.0])
        arena_size = self.obstacle_config.get('arena_size', 10.0)

        self.obstacle_ids = []
        self.moving_bodies = []

        # Add static shelves
        shelve_positions = [
            [2, -5, shelve_size[2] / 2],
            [6, -5, shelve_size[2] / 2],
            [2, 0, shelve_size[2] / 2],
            [6, 0, shelve_size[2] / 2],
            [2, 5, shelve_size[2] / 2],
            [6, 5, shelve_size[2] / 2]
        ]

        for pos in shelve_positions:
            pillar_id = create_box_shape(
                size=[shelve_size[0], shelve_size[1], shelve_size[2]],
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

        conveyor_pos = [10, 0, conveyor_size[2] / 2]
        conveyor_id = create_box_shape(
            size=[conveyor_size[0], conveyor_size[1], conveyor_size[2]],
            color=[0.5, 0.5, 0.5, 1],  # Gray color
            client_id=self.CLIENT
        )

        p.resetBasePositionAndOrientation(
            conveyor_id,
            conveyor_pos,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        self.obstacle_ids.append(conveyor_id)

        num_cylinders = 8  # Number of cylinders to add
        cylinder_radius = 0.3
        cylinder_height = 5
        y_spacing = conveyor_size[1] * 2 / (num_cylinders - 1)
        cylinder_bounds = [-np.inf, np.inf, -conveyor_size[1], conveyor_size[1], -np.inf, np.inf]
        cylinder_reset_position = [conveyor_pos[0], conveyor_pos[1] - conveyor_size[1], conveyor_pos[2] + cylinder_height / 2]
        cylinder_velocity = [0.0, 10.0, 0.0]  # Movement along the y-axis

        for i in range(num_cylinders):
            cylinder_position = [
                conveyor_pos[0],
                conveyor_pos[1] - conveyor_size[1] + i * y_spacing,
                conveyor_pos[2] + cylinder_height / 2
            ]

            cylinder_id = create_cylinder_shape(
                radius=cylinder_radius,
                height=cylinder_height,
                color=[0, 0, 1, 1],  # Blue color
                client_id=self.CLIENT
            )

            p.resetBasePositionAndOrientation(
                cylinder_id,
                cylinder_position,
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )

            self.moving_bodies.append(
                (cylinder_id, *cylinder_velocity, cylinder_bounds, cylinder_reset_position)
            )

            self.obstacle_ids.append(cylinder_id)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update moving shapes
        timestep = 1 / self.PYB_FREQ
        for body in self.moving_bodies:
            body_id, velocity_x, velocity_y, velocity_z, bounds, reset_position = body
            move_shape_reset(
                body_id,
                velocity_x,
                velocity_y,
                velocity_z,
                bounds,
                timestep,
                reset_position,
                client_id=self.CLIENT
            )

        return obs, reward, terminated, truncated, info

    def initialize_planning(self):
        """
        Initialize the planning-related attributes: start position, goal position, obstacles, and arena ranges.
        Returns:
        - start_pos: Starting position of the drone
        - goal_pos: Goal position
        - obstacles: List of obstacles with their positions and sizes
        - arena_ranges: Ranges for x, y, and z dimensions
        """
        start_pos = np.copy(self.pos[0])
        goal_pos = np.array([-5.0, 0.0, 1.0]) 

        obstacles = []
        for obs_id in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.CLIENT)
            size = p.getVisualShapeData(obs_id, physicsClientId=self.CLIENT)[0][3]
            print(f"Obstacle Position: {pos}, Size: {size}")
            obstacles.append({'position': np.array(pos), 'size': np.array(size)})

        arena_size = self.obstacle_config['arena_size']
        x_range = [-arena_size / 2, arena_size / 2]
        y_range = [-arena_size / 2, arena_size / 2]
        z_range = [0.5, 2.0]

        return start_pos, goal_pos, obstacles, (x_range, y_range, z_range)

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
            'shelve_size': [0.4, 1.5, 4.0],
            'conveyor_size': [0.5, 6, 0.5],
            'arena_size': 35.0,
        },
        seed=40
    )
    return env, num_steps
