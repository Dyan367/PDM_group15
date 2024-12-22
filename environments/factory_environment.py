import numpy as np
import pybullet as p

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
        shelve_size = self.obstacle_config.get('shelve_size', [0.4, 1.5, 4.0])
        conveyor_size = self.obstacle_config.get('conveyor_size', [0.5, 6, 0.5])

        self.obstacle_ids = []
        self.moving_bodies = []
        self.people = []

        # Shelf positions
        shelve_positions = []
        for x in [2, 6]:
            for y in [-5, 0, 5]:
                shelve_positions.append([x, y, shelve_size[2] / 2])

        # Add shelves
        for pos in shelve_positions:
            pillar_id = create_box_shape(
                size=shelve_size,
                color=[0.6, 0.4, 0.2, 1],  # Brown box
                client_id=self.CLIENT
            )
            p.resetBasePositionAndOrientation(pillar_id, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
            self.obstacle_ids.append(pillar_id)

        # Add conveyors
        conveyor_positions = [[10, 0, conveyor_size[2] / 2], [14, 0, conveyor_size[2] / 2]]
        for pos in conveyor_positions:
            conveyor_id = create_box_shape(
                size=conveyor_size,
                color=[0.5, 0.5, 0.5, 1],  # Gray box
                client_id=self.CLIENT
            )
            p.resetBasePositionAndOrientation(conveyor_id, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
            self.obstacle_ids.append(conveyor_id)

        # Cylinder parameters
        num_cylinders = 8
        cylinder_radius = 0.3
        cylinder_height = 3.5
        y_spacing = conveyor_size[1] * 2 / (num_cylinders - 1)

        def add_cylinders(conveyor_pos, y_direction, color):
            cylinder_bounds = [-np.inf, np.inf, -conveyor_size[1], conveyor_size[1], -np.inf, np.inf]
            reset_position = [conveyor_pos[0], conveyor_pos[1] - y_direction * conveyor_size[1], conveyor_pos[2] + cylinder_height / 2]
            velocity = [0.0, y_direction * 10.0, 0.0]

            for i in range(num_cylinders):
                cylinder_position = [
                    conveyor_pos[0],
                    conveyor_pos[1] - y_direction * (conveyor_size[1] - i * y_spacing),
                    conveyor_pos[2] + cylinder_height / 2
                ]
                cylinder_id = create_cylinder_shape(
                    radius=cylinder_radius,
                    height=cylinder_height,
                    color=color,
                    client_id=self.CLIENT
                )
                p.resetBasePositionAndOrientation(cylinder_id, cylinder_position, [0, 0, 0, 1], physicsClientId=self.CLIENT)
                self.moving_bodies.append((cylinder_id, *velocity, cylinder_bounds, reset_position))
                self.obstacle_ids.append(cylinder_id)

        # Add cylinders to conveyors
        add_cylinders([10, 0, conveyor_size[2] / 2], 1, [0, 0, 1, 1])  # Blue cylinders
        add_cylinders([14, 0, conveyor_size[2] / 2], -1, [0, 1, 0, 1])  # Green cylinders

        # Add crane
        crane_size = [0.4, conveyor_size[1], 0.8]
        self.crane = create_box_shape(
            size=crane_size,
            color=[1.0, 0.0, 0.0, 1],  # Red box
            client_id=self.CLIENT
        )
        p.resetBasePositionAndOrientation(self.crane, [10, 0, 5.0], [0, 0, 0, 1], physicsClientId=self.CLIENT)
        self.crane_bounds = [9, 15, -2, 2, 2.5, 3.5]
        self.crane_velocity_x = self.crane_velocity

        # Add moving people
        num_people = 8
        person_size = [0.4, 0.4, 3]
        person_bounds = [18, 28, -6, 6, 0, person_size[2]*2]
        add_bounding_box(person_bounds, client_id=0)
        max_speed = 10.0  
        change_interval = 2.0

        for i in range(num_people):
            person_position = [
                np.random.uniform(person_bounds[0], person_bounds[1]),
                np.random.uniform(person_bounds[2], person_bounds[3]),
                person_size[2]
            ]
            person_id = create_box_shape(
                size=person_size,
                color=[1, 0.75, 0.8, 1], # Pink box
                client_id=self.CLIENT
            )
            p.resetBasePositionAndOrientation(person_id, person_position, [0, 0, 0, 1], physicsClientId=self.CLIENT)
            self.people.append({
                "id": person_id,
                "bounds": person_bounds,
                "max_speed": max_speed,
                "change_interval": change_interval
            })
            self.obstacle_ids.append(person_id)


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update moving shapes
        timestep = 1 / self.PYB_FREQ
        for body_id, vel_x, vel_y, vel_z, bounds, reset_position in self.moving_bodies:
            move_shape_reset(
                body_id, vel_x, vel_y, vel_z,
                bounds, timestep, reset_position,
                client_id=self.CLIENT
            )

        self.crane_velocity_x, _, _ = move_shape_dynamic(
            self.crane,
            self.crane_velocity_x, 0, 0,
            self.crane_bounds, timestep,
            self.CLIENT
        )

        for person in self.people:
            person_id = person["id"]
            bounds = person["bounds"]
            max_speed = person["max_speed"]
            change_interval = person["change_interval"]

            move_shape_random(
                obstacle_id=person_id,
                bounds=bounds,
                max_speed=max_speed,
                timestep=timestep,
                change_interval=change_interval,
                client_id=self.CLIENT
            )

        return obs, reward, terminated, truncated, info




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
