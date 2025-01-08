import os
import numpy as np
import pybullet as p
from gymnasium import spaces
import logging
import time
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from environments.shapes import create_box_shape, create_cylinder_shape, add_bounding_box, move_shape_dynamic, move_shape_reset, move_shape_random

import tinympc

class MPCAviaryStaticTinyMPC(BaseAviary):
    """
    An extension of BaseAviary that uses TinyMPC for the MPC, 
    ensuring arrays are fortran-contiguous and dimensions match TinyMPC's expectations.
    """

    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 neighbourhood_radius=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 ctrl_freq=240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results',
                 mpc_params=None,
                 x_target=None,
                 obstacle_config={},
                 seed=42):

        self.x_target_config = x_target
        self.obstacle_config = obstacle_config
        self.seed = seed
        self.obstacles = obstacles
        self.obstacle_ids = []
        self.moving_bodies = []
        self.crane = None
        self.crane_velocity = 5.0
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=vision_attributes,
                         output_folder=output_folder)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if self.NUM_DRONES != 1:
            raise NotImplementedError("MPCAviary currently supports only a single drone.")

        np.random.seed(self.seed)

        if mpc_params is None:
            mpc_params = {
                'dt': 0.1,
                'N': 10,
                'sim_time': 10000,
                'proximity_threshold': 0.01
            }

        self.dt = mpc_params['dt']
        self.N = mpc_params['N']
        self.sim_time = mpc_params['sim_time']
        self.proximity_threshold = mpc_params['proximity_threshold']
        self.max_steps = int(self.sim_time / self.dt)

        # Build A,B,c,Q,R, etc.
        self._initialize_mpc_matrices()

        if self.x_target_config is not None:
            if self.x_target_config.shape != (12,):
                raise ValueError("x_target must be a 12-dimensional vector.")
            self.x_target = self.x_target_config
        else:
            self.x_target = np.array([2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.state_history = np.zeros((self.max_steps + 1, 12))
        self.control_history = np.zeros((self.max_steps, 4))

        initial_state = self._get_current_state()
        self.state_history[0, :] = initial_state
        logging.info(f"Target set to: {self.x_target}")

        self.obstacle_ids = []
        if self.obstacles:
            self._addObstacles()

        # ---- Setup TinyMPC properly ----
        self._tinympc_setup()

    def _tinympc_setup(self):
        # Convert everything to Fortran as you did
        A_f = np.asfortranarray(self.A, dtype=np.float64)
        B_f = np.asfortranarray(self.B_full, dtype=np.float64)
        Q_f = np.asfortranarray(self.Q, dtype=np.float64)
        R_f = np.asfortranarray(self.R, dtype=np.float64)

        # Some large (±1000) bounds for states and inputs
        nx = A_f.shape[0]  # 12
        nu = B_f.shape[1]  # 4
        x_min_f = -1e3 * np.ones(nx, dtype=np.float64)
        x_max_f = +1e3 * np.ones(nx, dtype=np.float64)
        u_min_f = -1e3 * np.ones(nu, dtype=np.float64)
        u_max_f = +1e3 * np.ones(nu, dtype=np.float64)

        self.tinympc_prob = tinympc.TinyMPC()

        # For the new signature, just do:
        # setup(A, B, Q, R, N, x_min=None, x_max=None, u_min=None, u_max=None, xf_min=None, xf_max=None, settings=None)
        self.tinympc_prob.setup(
            A_f, B_f, Q_f, R_f,
            self.N,                            # horizon length
            x_min=x_min_f,
            x_max=x_max_f,
            u_min=u_min_f,
            u_max=u_max_f,
            xf_min=None,                       # final-state bounds, if you want
            xf_max=None,
            settings=None
        )



    def set_target(self, new_target):
        if new_target.shape != (12,):
            raise ValueError("new_target must be a 12-dimensional vector.")
        self.x_target = new_target
        logging.info(f"Target updated to: {self.x_target}")

    def _initialize_mpc_matrices(self):
        mass = self.M
        Ixx, Iyy, Izz = self.J[0,0], self.J[1,1], self.J[2,2]
        g = self.G

        d_x = 9.1785e-7
        d_y = 9.1785e-7
        d_z = 10.311e-7

        A = np.eye(12)
        A[0,3] = self.dt
        A[1,4] = self.dt
        A[2,5] = self.dt

        # Some scaled example
        A[3,7] = (self.dt/mass)
        A[4,6] = -(self.dt/mass)
        A[3,3] = 1 - d_x*self.dt
        A[4,4] = 1 - d_y*self.dt
        A[5,5] = 1 - d_z*self.dt

        A[6,9]  = self.dt
        A[7,10] = self.dt
        A[8,11] = self.dt

        B_full = np.zeros((12,4))
        B_full[5,0]  = self.dt/mass
        B_full[9,1]  = self.dt/Ixx
        B_full[10,2] = self.dt/Iyy
        B_full[11,3] = self.dt/Izz

        c = np.zeros(12)
        
        c[5] = -self.dt*g

        Q = np.diag([
            2000, 2000, 3000,
            500,   500,   500,
            5,    5,    5,
            1,    1,    1
        ])
        R = np.diag([1.0, 0.2, 0.2, 0.2])

        u_min = np.array([0, -np.pi/3, -np.pi/3, -np.pi/3])
        u_max = np.array([20, np.pi/3, np.pi/3, np.pi/3])

        self.A = A
        self.B_full = B_full
        self.c = c
        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max

    def _get_current_state(self):
        drone_id = self.DRONE_IDS[0]
        pos, orn = p.getBasePositionAndOrientation(drone_id)
        linear_vel, angular_vel = p.getBaseVelocity(drone_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        state = np.array([
            pos[0], pos[1], pos[2],
            linear_vel[0], linear_vel[1], linear_vel[2],
            roll, pitch, yaw,
            angular_vel[0], angular_vel[1], angular_vel[2]
        ])
        return state

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        initial_state = self._get_current_state()
        self.state_history[0, :] = initial_state

        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []

        if self.obstacle_config and self.obstacles:
            self._addObstacles()

        return initial_state, info

    # def _addObstacles(self):
    #     num_obstacles = self.obstacle_config.get('num_obstacles', 5)
    #     obstacle_size = self.obstacle_config.get('obstacle_size', [0.5,0.5,0.5])
    #     arena_size = self.obstacle_config.get('arena_size', 5.0)
    #     obstacle_density = self.obstacle_config.get('obstacle_density',0.5)
    #
    #     self.obstacle_ids = []
    #     drone_start_pos = self.INIT_XYZS[0]
    #
    #     rows = int(np.sqrt(num_obstacles))
    #     cols = rows if rows>0 else 1
    #     shelf_spacing_x = arena_size/cols
    #     shelf_spacing_y = arena_size/rows
    #
    #     for i in range(rows):
    #         for j in range(cols):
    #             if len(self.obstacle_ids)>=num_obstacles:
    #                 break
    #             x = -arena_size/2 + (j+0.5)*shelf_spacing_x + np.random.uniform(-obstacle_density, obstacle_density)
    #             y = -arena_size/2 + (i+0.5)*shelf_spacing_y + np.random.uniform(-obstacle_density, obstacle_density)
    #             z = obstacle_size[2]/2
    #             distance = np.linalg.norm(np.array([x, y]) - drone_start_pos[:2])
    #             if distance <= obstacle_size[0]:
    #                 continue
    #
    #             collision_shape = p.createCollisionShape(
    #                 shapeType=p.GEOM_BOX,
    #                 halfExtents=[dim/2 for dim in obstacle_size],
    #                 physicsClientId=self.CLIENT
    #             )
    #             visual_shape = p.createVisualShape(
    #                 shapeType=p.GEOM_BOX,
    #                 halfExtents=[dim/2 for dim in obstacle_size],
    #                 rgbaColor=[0.6,0.4,0.2,1],
    #                 physicsClientId=self.CLIENT
    #             )
    #             obs_id = p.createMultiBody(
    #                 baseMass=0,
    #                 baseCollisionShapeIndex=collision_shape,
    #                 baseVisualShapeIndex=visual_shape,
    #                 basePosition=[x,y,z],
    #                 physicsClientId=self.CLIENT
    #             )
    #             self.obstacle_ids.append(obs_id)
    #
    #     logging.info(f"Added {len(self.obstacle_ids)} obstacles to the environment.")
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

    def step(self, action=None):
        t = self.step_counter // self.PYB_STEPS_PER_CTRL
        if t >= self.max_steps:
            terminated = True
            truncated = False
            reward = 0.0
            info = self._computeInfo()
            return self._computeObs(), reward, terminated, truncated, info

        current_state = self._get_current_state()
        self.state_history[t, :] = current_state

        u_opt = self._solve_mpc(current_state)
        self.control_history[t, :] = u_opt

        # Apply
        thrust_z, torque_x, torque_y, torque_z = u_opt
        print(f"Time {t*self.dt:.1f}s - Control: Tz={thrust_z:.2f}, Tx={torque_x:.2f}, Ty={torque_y:.2f}, Tz={torque_z:.2f}")
        self._apply_control_inputs(thrust_z, torque_x, torque_y, torque_z)

        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.CLIENT)
            time.sleep(self.PYB_TIMESTEP)

        self._updateAndStoreKinematicInformation()

        if t+1 <= self.max_steps:
            self.state_history[t+1, :] = self._get_current_state()

        distance = np.linalg.norm(current_state[:3] - self.x_target[:3])
        if distance < self.proximity_threshold:
            terminated = False
            truncated = False
            logging.info(f"Target reached at step {t}, time {t*self.dt:.1f} s.")
        else:
            terminated = False
            truncated = False

        reward = -distance
        self.step_counter += self.PYB_STEPS_PER_CTRL

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




        return self._computeObs(), reward, terminated, truncated, self._computeInfo()

    def _solve_mpc(self, current_state):
        """
        TinyMPC solve step. We'll store the 'error' e = x - x_target, 
        run the solver, clamp solution, apply +c in sim externally.
        """
        e_current = current_state - self.x_target
        self.tinympc_prob.set_x0(e_current)
        solution = self.tinympc_prob.solve()
        if solution is None:
            logging.warning("TinyMPC returned None. Using zero control.")
            u_opt = np.zeros(4)
        else:
            u_opt = solution["controls"]

        # clamp to the environment's bounds
        u_opt = np.clip(u_opt, self.u_min, self.u_max)
        return u_opt

    def _apply_control_inputs(self, thrust_z, torque_x, torque_y, torque_z):
        thrust_body = np.array([0, 0, thrust_z])
        p.applyExternalForce(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,
            forceObj=thrust_body.tolist(),
            posObj=[0,0,0],
            flags=p.LINK_FRAME
        )
        torque = np.array([torque_x, torque_y, torque_z])
        p.applyExternalTorque(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,
            torqueObj=torque.tolist(),
            flags=p.LINK_FRAME
        )

    def close(self):
        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []
            logging.info("Removed all obstacles from the environment.")
        super().close()

    def render(self, mode='human', close=False):
        super().render(mode=mode, close=close)

    def _actionSpace(self):
        return 0

    def _observationSpace(self):
        return 0

    def _computeObs(self):
        return self._get_current_state()

    def _preprocessAction(self, action):
        return np.zeros(4)

    def _computeReward(self):
        return -1

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {"answer": 42}

    def plot_results(self):
        steps = min(self.step_counter // self.PYB_STEPS_PER_CTRL, self.max_steps)
        time_array = np.linspace(0, steps*self.dt, steps+1)

        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(self.state_history[:steps+1,0],
                self.state_history[:steps+1,1],
                self.state_history[:steps+1,2], label='Trajectory')
        ax.scatter(self.x_target[0], self.x_target[1], self.x_target[2],
                   color='r', marker='*', s=100, label='Target')
        ax.set_title('Trajectory 3D')
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)

        ax2 = fig.add_subplot(132)
        ax2.plot(self.state_history[:steps+1,0],
                 self.state_history[:steps+1,1], 'b-', label='XY Path')
        ax2.plot(self.x_target[0], self.x_target[1], 'ro', label='Target')
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax2.set_title('Top view')
        ax2.grid(True)
        ax2.axis('equal')
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.plot(time_array, self.state_history[:steps+1,6], label='Roll')
        ax3.plot(time_array, self.state_history[:steps+1,7], label='Pitch')
        ax3.plot(time_array, self.state_history[:steps+1,8], label='Yaw')
        ax3.set_title('Orientation (rad)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (rad)')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot inputs
        plt.figure(figsize=(14,6))
        time_ctrl = np.linspace(0, steps*self.dt, steps)
        plt.subplot(2,1,1)
        plt.plot(time_ctrl, self.control_history[:steps,0], label='Thrust Z')
        plt.title('Control Inputs Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(time_ctrl, self.control_history[:steps,1], label='Torque X')
        plt.plot(time_ctrl, self.control_history[:steps,2], label='Torque Y')
        plt.plot(time_ctrl, self.control_history[:steps,3], label='Torque Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
