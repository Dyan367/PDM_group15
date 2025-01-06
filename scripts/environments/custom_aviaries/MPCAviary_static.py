

import os
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
import cvxpy as cp
import matplotlib.pyplot as plt
import logging
import time

class MPCAviaryStatic(BaseAviary):
    """
    An extension of BaseAviary that incorporates a Model Predictive Controller (MPC)
    and allows dynamic obstacle placement.
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
                 x_target=None,  # External target
                 obstacle_config={},  # Obstacle configuration
                 seed=42  # Seed for reproducibility
                 ):
        """
        Initializes the MPCAviary environment with MPC controller parameters and obstacle configurations.

        Parameters
        ----------
        All parameters are inherited from BaseAviary except:
        x_target : np.array, optional
            Target state vector [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz].
            If None, a default target is used.
        obstacle_config : dict, optional
            Configuration dictionary for obstacles (e.g., number, size, arena size).
        seed : int, optional
            Seed for random number generators to ensure reproducibility.
        """
        self.x_target_config = x_target  # Store the target configuration
        self.obstacle_config = obstacle_config  # Store obstacle configuration
        self.seed = seed  # Store seed for reproducibility
        self.obstacles = obstacles  # **Add this line to define the 'obstacles' attribute**
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
                         obstacles=obstacles,  # Pass to superclass
                         user_debug_gui=user_debug_gui,
                         vision_attributes=vision_attributes,
                         output_folder=output_folder)

        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Ensure single drone for simplicity
        if self.NUM_DRONES != 1:
            raise NotImplementedError("MPCAviary currently supports only a single drone.")

        # Set the random seed for reproducibility
        np.random.seed(self.seed)

        # MPC Controller Parameters
        if mpc_params is None:
            mpc_params = {
                'dt': 0.1,          # Time step (seconds)
                'N': 10,            # Prediction horizon (steps)
                'sim_time': 100,    # Total simulation time (seconds)
                'proximity_threshold': 0.01  # meters
            }

        self.dt = mpc_params['dt']
        self.N = mpc_params['N']
        self.sim_time = mpc_params['sim_time']
        self.proximity_threshold = mpc_params['proximity_threshold']
        self.max_steps = int(self.sim_time / self.dt)

        # Initialize MPC matrices
        self._initialize_mpc_matrices()

        # Initialize target
        if self.x_target_config is not None:
            if self.x_target_config.shape != (12,):
                raise ValueError("x_target must be a 12-dimensional vector.")
            self.x_target = self.x_target_config
        else:
            self.x_target = np.array([
                2, 2, 2,      # Target position
                0, 0, 0,      # Target velocity
                0, 0, 0,      # Target orientation (roll, pitch, yaw)
                0, 0, 0       # Target angular velocity
            ])

        # Initialize state and control histories
        self.state_history = np.zeros((self.max_steps + 1, 12))
        self.control_history = np.zeros((self.max_steps, 4))

        # Reset initial state
        initial_state = self._get_current_state()
        self.state_history[0, :] = initial_state

        # Log the target for debugging
        logging.info(f"Target set to: {self.x_target}")

        # Initialize obstacle IDs list
        self.obstacle_ids = []

        # Add obstacles if enabled
        if self.obstacles:
            self._addObstacles()

    def set_target(self, new_target):
        """
        Updates the target state vector.

        Parameters
        ----------
        new_target : np.array
            New target state vector [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz].
        """
        if new_target.shape != (12,):
            raise ValueError("new_target must be a 12-dimensional vector.")
        self.x_target = new_target
        logging.info(f"Target updated to: {self.x_target}")

    def _initialize_mpc_matrices(self):
        """Initializes the state-space matrices for MPC."""
        # Drone properties
        mass = self.M
        Ixx, Iyy, Izz = self.J[0,0], self.J[1,1], self.J[2,2]
        g = self.G

        # Drag coefficients (Assumed or can be adjusted)
        d_x = 9.1785e-7
        d_y = 9.1785e-7
        d_z = 10.311e-7

        # State-space matrices for 12D state: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        A = np.eye(12)
        A[0,3] = self.dt  # x_dot = vx
        A[1,4] = self.dt  # y_dot = vy
        A[2,5] = self.dt  # z_dot = vz

        # Influence of orientation on velocities
        A[3,7] = self.dt / mass    # Pitch influences vx
        A[4,6] = -self.dt / mass   # Roll influences vy
        A[3,3] = 1 - d_x * self.dt  # vx_dot includes drag
        A[4,4] = 1 - d_y * self.dt  # vy_dot includes drag
        A[5,5] = 1 - d_z * self.dt  # vz_dot includes drag

        # Rotational derivatives
        A[6,9] = self.dt  # roll_dot = wx
        A[7,10] = self.dt # pitch_dot = wy
        A[8,11] = self.dt # yaw_dot = wz

        # Input matrix
        B_full = np.zeros((12, 4))
        B_full[5,0] = self.dt / mass  # Thrust_z influences vz_dot
        B_full[9,1] = self.dt / Ixx  # Torque_x influences wx_dot
        B_full[10,2] = self.dt / Iyy # Torque_y influences wy_dot
        B_full[11,3] = self.dt / Izz # Torque_z influences wz_dot

        # Disturbance vector (gravity)
        c = np.zeros(12)
        c[5] = -self.dt * g  # Negative sign for downward acceleration

        # Cost matrices
        Q = np.diag([
            200, 200, 300,    # Position
            10, 10, 10,        # Velocity
            50, 50, 50,        # Orientation
            10, 10, 10         # Angular Velocity
        ])
        R = np.diag([1.0, 0.2, 0.2, 0.2])  # [Thrust_z, Torque_x, Torque_y, Torque_z]

        # Control constraints
        u_min = np.array([0, -np.pi/3, -np.pi/3, -np.pi/3])  # [Thrust_z_min, Torque_x_min, Torque_y_min, Torque_z_min]
        u_max = np.array([20, np.pi/3, np.pi/3, np.pi/3])    # [Thrust_z_max, Torque_x_max, Torque_y_max, Torque_z_max]

        # Assign to instance variables
        self.A = A
        self.B_full = B_full
        self.c = c
        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max

        # State and target vectors
        self.x0 = np.zeros(12)
        # self.x_target is already initialized in __init__

    def _get_current_state(self):
        """
        Retrieves the current state of the drone in the format required by MPC.

        Returns
        -------
        state (np.array): [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        """
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

    def reset(self, 
              seed: int = None, 
              options: dict = None):
        """
        Resets the environment, initializes MPC state history, and adds obstacles.

        Returns
        -------
        observation (np.array): The initial observation.
        info (dict): Additional information.
        """
        obs, info = super().reset(seed=seed)
        initial_state = self._get_current_state()
        self.state_history[0, :] = initial_state

        # Remove existing obstacles if any
        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []

        # Add obstacles
        if self.obstacle_config and self.obstacles:
            self._addObstacles()

        return initial_state, info

    def _addObstacles(self):
        """
        Adds obstacles to the simulation based on the provided obstacle configuration.
        """
        num_obstacles = self.obstacle_config.get('num_obstacles', 5)
        obstacle_size = self.obstacle_config.get('obstacle_size', [0.5, 0.5, 0.5])
        arena_size = self.obstacle_config.get('arena_size', 5.0)
        obstacle_density = self.obstacle_config.get('obstacle_density', 0.5)  # Optional: To control obstacle placement randomness

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

                # Calculate obstacle positions with some randomness based on obstacle_density
                x = -arena_size / 2 + (j + 0.5) * shelf_spacing_x + np.random.uniform(-obstacle_density, obstacle_density)
                y = -arena_size / 2 + (i + 0.5) * shelf_spacing_y + np.random.uniform(-obstacle_density, obstacle_density)
                z = obstacle_size[2] / 2

                distance = np.linalg.norm(np.array([x, y]) - drone_start_pos[:2])
                if distance <= obstacle_size[0]:
                    continue  # Avoid placing obstacles too close to the drone's start position

                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[dim / 2 for dim in obstacle_size],
                    physicsClientId=self.CLIENT
                )
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[dim / 2 for dim in obstacle_size],
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

        logging.info(f"Added {len(self.obstacle_ids)} obstacles to the environment.")

    def step(self, action=None):
        """
        Advances the environment by one simulation step using MPC to compute control inputs.

        Parameters
        ----------
        action : None
            Not used since MPC computes control inputs internally.

        Returns
        -------
        obs (np.array): The current observation.
        reward (float): The computed reward.
        terminated (bool): Whether the episode has terminated.
        truncated (bool): Whether the episode has been truncated.
        info (dict): Additional information.
        """
        t = self.step_counter // self.PYB_STEPS_PER_CTRL  # Current step index for MPC

        if t >= self.max_steps:
            # Episode termination condition
            terminated = True
            truncated = False
            reward = 0.0
            info = self._computeInfo()
            return self._computeObs(), reward, terminated, truncated, info

        # Get current state
        current_state = self._get_current_state()
        self.state_history[t, :] = current_state

        # MPC Optimization
        u_opt = self._solve_mpc(current_state)
        self.control_history[t, :] = u_opt

        # Debug: Print control inputs
        thrust_z, torque_x, torque_y, torque_z = u_opt
        print(f"Time {t*self.dt:.1f}s - Control Inputs: Thrust_z={thrust_z:.2f} N, Torque_x={torque_x:.2f} Nm, Torque_y={torque_y:.2f} Nm, Torque_z={torque_z:.2f} Nm")

        # Apply control inputs directly as thrust and torques
        self._apply_control_inputs(thrust_z, torque_x, torque_y, torque_z)

        # Step the simulation
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.CLIENT)
            time.sleep(self.PYB_TIMESTEP)

        # Update kinematic information
        self._updateAndStoreKinematicInformation()

        # Store the final state after the step
        if t + 1 <= self.max_steps:
            self.state_history[t + 1, :] = self._get_current_state()

        # Check termination condition
        distance = np.linalg.norm(current_state[:3] - self.x_target[:3])
        if distance < self.proximity_threshold:
            terminated = True
            truncated = False
            logging.info(f"Target reached at step {t}, time {t*self.dt:.1f} seconds.")
        else:
            terminated = False
            truncated = False

        # Compute reward (optional, can be customized)
        reward = -distance  # Example: negative distance as reward

        # Increment step counter
        self.step_counter += self.PYB_STEPS_PER_CTRL

        # Return observation, reward, terminated, truncated, info
        return self._computeObs(), reward, terminated, truncated, self._computeInfo()

    def _solve_mpc(self, current_state):
        """
        Solves the MPC optimization problem to compute the optimal control inputs.

        Parameters
        ----------
        current_state : np.array
            The current state of the drone.

        Returns
        -------
        u_opt (np.array): Optimal control inputs [Thrust_z, Torque_x, Torque_y, Torque_z].
        """
        N = self.N
        A = self.A
        B = self.B_full
        c = self.c
        Q = self.Q
        R = self.R
        u_min = self.u_min
        u_max = self.u_max
        x_target = self.x_target

        # Define optimization variables
        u = cp.Variable((4, N))
        x = cp.Variable((12, N + 1))

        # Define the cost function and constraints
        cost = 0
        constraints = []

        # Initial condition
        constraints += [x[:, 0] == current_state]

        for k in range(N):
            # Cost accumulation
            state_error = x[:, k] - x_target
            cost += cp.quad_form(state_error, Q) + cp.quad_form(u[:, k], R)

            # Control input smoothing (rate constraints)
            if k > 0:
                delta_u = u[:, k] - u[:, k - 1]
            else:
                delta_u = u[:, k] - np.zeros(4)  # Assuming initial control inputs are zero
            # Define maximum allowed change per timestep
            max_delta = np.array([2, np.pi/12, np.pi/12, np.pi/12])  # Adjusted for smoother control
            constraints += [delta_u <= max_delta]
            constraints += [delta_u >= -max_delta]

            # Penalize large changes in control inputs
            cost += cp.quad_form(delta_u, np.diag([0.1, 0.1, 0.1, 0.1]))

            # Dynamics constraints with constant disturbance (gravity)
            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + c]

            # Control constraints
            constraints += [u_min <= u[:, k], u[:, k] <= u_max]

        # Terminal cost
        state_error_terminal = x[:, N] - x_target
        cost += cp.quad_form(state_error_terminal, Q)

        # Define and solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                logging.warning(f"MPC Optimization failed with status {prob.status}. Trying ECOS...")
                prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                logging.error(f"MPC Optimization failed with status {prob.status}. Using zero control inputs.")
                u_opt = np.zeros(4)
            else:
                u_opt = u.value[:, 0]
        except cp.SolverError as e:
            logging.error(f"MPC SolverError: {e}. Using zero control inputs.")
            u_opt = np.zeros(4)

        # Ensure u_opt is within bounds
        u_opt = np.clip(u_opt, self.u_min, self.u_max)

        return u_opt

    def _apply_control_inputs(self, thrust_z, torque_x, torque_y, torque_z):
        """
        Applies the computed thrust and torques directly to the drone.

        Parameters
        ----------
        thrust_z : float
            Thrust force along the z-axis in Newtons.
        torque_x : float
            Torque around the x-axis in Nm.
        torque_y : float
            Torque around the y-axis in Nm.
        torque_z : float
            Torque around the z-axis in Nm.
        """
        # Define thrust in the body frame (assuming thrust is along the drone's positive z-axis)
        thrust_body = np.array([0, 0, thrust_z])

        # Apply the thrust in the LOCAL_FRAME
        p.applyExternalForce(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,  # Apply to base
            forceObj=thrust_body.tolist(),
            posObj=[0, 0, 0],  # Center of mass
            flags=p.LINK_FRAME  # Apply in local frame
        )

        # Define torques in the LOCAL_FRAME
        torque = np.array([torque_x, torque_y, torque_z])
        p.applyExternalTorque(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,
            torqueObj=torque.tolist(),
            flags=p.LINK_FRAME  # Apply in local frame
        )

    def render(self, mode='human', close=False):
        """
        Overrides the render method to include MPC-specific rendering if needed.
        """
        super().render(mode=mode, close=close)

    def close(self):
        """
        Closes the environment, removes obstacles, and disconnects PyBullet.
        """
        # Remove obstacles if any
        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []
            logging.info("Removed all obstacles from the environment.")
        super().close()

    def _actionSpace(self):
        """
        Defines the action space. Since MPC computes actions internally,
        external actions are not used.
        """
        # Define a dummy action space with shape (0,)
        return 0

    def _observationSpace(self):
        """
        Defines the observation space based on the drone's state vector.
        """
        return 0

    def _computeObs(self):
        """
        Returns the current observation, which is the drone's state vector.
        """
        return self._get_current_state()

    def _preprocessAction(self, action):
        """
        Preprocesses the action. Since MPC handles control internally,
        this method can return a default or zero action.
        """
        # Ignore external actions and return zero or previous control inputs
        return np.zeros(4)

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.
        """
        return -1

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.
        """
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.
        """
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.
        """
        return {"answer": 42}  # Calculated by the Deep Thought supercomputer in 7.5M years

    def plot_results(self):
        """
        Plots the trajectory, orientation, and control inputs after simulation.
        """
        steps = min(self.step_counter // self.PYB_STEPS_PER_CTRL, self.max_steps)
        time_array = np.linspace(0, steps * self.dt, steps + 1)

        # Plotting the trajectory
        fig = plt.figure(figsize=(18, 6))

        # 3D Trajectory Plot
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(self.state_history[:steps+1, 0], 
                self.state_history[:steps+1, 1], 
                self.state_history[:steps+1, 2], label='Drone Trajectory')
        ax.scatter(self.x_target[0], self.x_target[1], self.x_target[2], 
                   color='red', marker='*', s=100, label='Target')
        ax.set_title('Drone Trajectory in 3D Space')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.legend()
        ax.grid(True)

        # 2D Projection (Top View)
        ax2 = fig.add_subplot(132)
        ax2.plot(self.state_history[:steps+1, 0], 
                 self.state_history[:steps+1, 1], 'b-', label='Drone Trajectory')
        ax2.plot(self.x_target[0], self.x_target[1], 'ro', label='Target')
        ax2.set_title('Drone Trajectory - Top View')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')

        # Orientation Plot
        ax3 = fig.add_subplot(133)
        ax3.plot(time_array, self.state_history[:steps+1, 6], label='Roll (rad)')
        ax3.plot(time_array, self.state_history[:steps+1, 7], label='Pitch (rad)')
        ax3.plot(time_array, self.state_history[:steps+1, 8], label='Yaw (rad)')
        ax3.set_title('Drone Orientation Over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (rad)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot control inputs over time
        plt.figure(figsize=(14, 6))
        time_array_control = np.linspace(0, steps * self.dt, steps)
        plt.subplot(2,1,1)
        plt.plot(time_array_control, self.control_history[:, 0], label='Thrust Z (N)')
        plt.title('Control Inputs Over Time')
        plt.ylabel('Thrust (N)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(time_array_control, self.control_history[:, 1], label='Torque X (Nm)')
        plt.plot(time_array_control, self.control_history[:, 2], label='Torque Y (Nm)')
        plt.plot(time_array_control, self.control_history[:, 3], label='Torque Z (Nm)')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
