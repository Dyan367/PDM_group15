import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from environments.shapes import create_box_shape, create_cylinder_shape, move_shape_dynamic, move_shape_reset


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

        # Remove existing obstacles if they exist
        if hasattr(self, 'obstacle_ids'):
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)

        if self.crane:
            p.removeBody(self.crane, physicsClientId=self.CLIENT)

        self._addObstacles()
        return obs, info

    def _addObstacles(self):
        shelve_size = self.obstacle_config.get('shelve_size', [1.0, 1.0, 1.0])
        conveyor_size = self.obstacle_config.get('conveyor_size', [1.0, 1.0, 1.0])
        arena_size = self.obstacle_config.get('arena_size', 10.0)

        self.obstacle_ids = []
        self.moving_bodies = []

        shelve_positions = [
            [2, -5, shelve_size[2] / 2],
            [6, -5, shelve_size[2] / 2],
            [2, 0, shelve_size[2] / 2],
            [6, 0, shelve_size[2] / 2],
            [2, 5, shelve_size[2] / 2],
            [6, 5, shelve_size[2] / 2]
        ]

        # Add shelves
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

        conveyor_pos1 = [10, 0, conveyor_size[2] / 2]
        conveyor_id1 = create_box_shape(
            size=[conveyor_size[0], conveyor_size[1], conveyor_size[2]],
            color=[0.5, 0.5, 0.5, 1],  # Gray color
            client_id=self.CLIENT
        )

        p.resetBasePositionAndOrientation(
            conveyor_id1,
            conveyor_pos1,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        self.obstacle_ids.append(conveyor_id1)

        conveyor_pos2 = [14, 0, conveyor_size[2] / 2]
        conveyor_id2 = create_box_shape(
            size=[conveyor_size[0], conveyor_size[1], conveyor_size[2]],
            color=[0.5, 0.5, 0.5, 1],  # Gray color
            client_id=self.CLIENT
        )

        p.resetBasePositionAndOrientation(
            conveyor_id2,
            conveyor_pos2,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        self.obstacle_ids.append(conveyor_id2)

        # Add cylinders on the first conveyor moving in the positive y direction
        num_cylinders = 8
        cylinder_radius = 0.3
        cylinder_height = 3.5
        y_spacing = conveyor_size[1] * 2 / (num_cylinders - 1)
        cylinder_bounds1 = [-np.inf, np.inf, -conveyor_size[1], conveyor_size[1], -np.inf, np.inf]
        cylinder_reset_position1 = [conveyor_pos1[0], conveyor_pos1[1] - conveyor_size[1], conveyor_pos1[2] + cylinder_height / 2]
        cylinder_velocity1 = [0.0, 10.0, 0.0]

        for i in range(num_cylinders):
            cylinder_position1 = [
                conveyor_pos1[0],
                conveyor_pos1[1] - conveyor_size[1] + i * y_spacing,
                conveyor_pos1[2] + cylinder_height / 2
            ]

            cylinder_id1 = create_cylinder_shape(
                radius=cylinder_radius,
                height=cylinder_height,
                color=[0, 0, 1, 1],  # Blue color
                client_id=self.CLIENT
            )

            p.resetBasePositionAndOrientation(
                cylinder_id1,
                cylinder_position1,
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )

            self.moving_bodies.append(
                (cylinder_id1, *cylinder_velocity1, cylinder_bounds1, cylinder_reset_position1)
            )

            self.obstacle_ids.append(cylinder_id1)

        # Add cylinders on the second conveyor moving in the negative y direction
        cylinder_bounds2 = [-np.inf, np.inf, -conveyor_size[1], conveyor_size[1], -np.inf, np.inf]
        cylinder_reset_position2 = [conveyor_pos2[0], conveyor_pos2[1] + conveyor_size[1], conveyor_pos2[2] + cylinder_height / 2]
        cylinder_velocity2 = [0.0, -10.0, 0.0]

        for i in range(num_cylinders):
            cylinder_position2 = [
                conveyor_pos2[0],
                conveyor_pos2[1] + conveyor_size[1] - i * y_spacing,
                conveyor_pos2[2] + cylinder_height / 2
            ]

            cylinder_id2 = create_cylinder_shape(
                radius=cylinder_radius,
                height=cylinder_height,
                color=[0, 1, 0, 1],  # Green color
                client_id=self.CLIENT
            )

            p.resetBasePositionAndOrientation(
                cylinder_id2,
                cylinder_position2,
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )

            self.moving_bodies.append(
                (cylinder_id2, *cylinder_velocity2, cylinder_bounds2, cylinder_reset_position2)
            )

            self.obstacle_ids.append(cylinder_id2)

        crane_size = [0.4, conveyor_size[1] * 2, 0.8]
        crane_position = [10, 0, 5.0]

        self.crane = create_box_shape(
            size=crane_size,
            color=[1.0, 0.0, 0.0, 1],  # Red color
            client_id=self.CLIENT
        )

        p.resetBasePositionAndOrientation(
            self.crane,
            crane_position,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        self.crane_bounds = [conveyor_pos1[0] - 1, conveyor_pos2[0] + 1, -2, 2, 2.5, 3.5]
        self.crane_velocity_x = self.crane_velocity


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

        self.crane_velocity_x, _, _ = move_shape_dynamic(
        self.crane,
        self.crane_velocity_x, 0, 0,
        self.crane_bounds,
        timestep,
        self.CLIENT
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
            print(f"Obstacle Position: {pos}, Size: {size}, AABB Min: {aabb_min}, AABB Max: {aabb_max}")

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
            'arena_size': 100.0,
        },
        seed=40
    )
    return env, num_steps
