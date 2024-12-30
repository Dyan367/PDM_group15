import os
import numpy as np
import pybullet as p
import logging
import time
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# do-mpc and CasADi
import do_mpc
from casadi import DM

class MPCAviaryStaticDoMPC(BaseAviary):
    """
    A do-mpc-based environment that mimics TinyMPC's "error-state" approach.

    Key differences from the standard do-mpc example:
      1) We define e = x - x_target as the 'state' inside do-mpc, so the solver
         works exactly like TinyMPC, which also uses error-based updates.
      2) We do NOT include gravity in the solver's model. Instead, we apply gravity
         externally (in step()), matching how TinyMPC does it.
      3) We keep large +/- 1e3 bounds in do-mpc, then clamp to [u_min, u_max] after.
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
            raise NotImplementedError("MPCAviaryStaticDoMPC currently supports only a single drone.")

        np.random.seed(self.seed)

        # If no MPC params are provided, set some defaults
        if mpc_params is None:
            mpc_params = {
                'dt': 0.1,
                'N': 10,
                'sim_time': 100,
                'proximity_threshold': 0.01
            }
        self.dt = mpc_params['dt']
        self.N = mpc_params['N']
        self.sim_time = mpc_params['sim_time']
        self.proximity_threshold = mpc_params['proximity_threshold']
        self.max_steps = int(self.sim_time / self.dt)

        # Build same A,B,c,Q,R as TinyMPC
        self._initialize_mpc_matrices()

        # Set target
        if self.x_target_config is not None:
            if self.x_target_config.shape != (12,):
                raise ValueError("x_target must be a 12-dimensional vector.")
            self.x_target = self.x_target_config
        else:
            self.x_target = np.array([2,2,2, 0,0,0, 0,0,0, 0,0,0])

        logging.info(f"Target set to: {self.x_target}")

        self.state_history = np.zeros((self.max_steps+1, 12))
        self.control_history = np.zeros((self.max_steps, 4))

        init_state = self._get_current_state()
        self.state_history[0,:] = init_state

        self.obstacle_ids = []
        if self.obstacles:
            self._addObstacles()

        # Setup do-mpc with error-state approach
        self._dompc_setup()

        # Initialize the do-mpc solver's guess
        e0 = init_state - self.x_target  # error init
        self.mpc.x0 = e0.reshape((12,1))
        self.mpc.u0 = np.zeros((4,1))
        self.mpc.set_initial_guess()
        self.mpc.reset_history()

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

    def _addObstacles(self):
        num_obstacles = self.obstacle_config.get('num_obstacles', 5)
        obstacle_size = self.obstacle_config.get('obstacle_size', [0.5,0.5,0.5])
        arena_size = self.obstacle_config.get('arena_size', 5.0)
        obstacle_density = self.obstacle_config.get('obstacle_density',0.5)

        self.obstacle_ids = []
        drone_start_pos = self.INIT_XYZS[0]

        rows = int(np.sqrt(num_obstacles))
        cols = rows if rows>0 else 1
        shelf_spacing_x = arena_size/cols
        shelf_spacing_y = arena_size/rows

        for i in range(rows):
            for j in range(cols):
                if len(self.obstacle_ids)>=num_obstacles:
                    break
                x = -arena_size/2 + (j+0.5)*shelf_spacing_x + np.random.uniform(-obstacle_density, obstacle_density)
                y = -arena_size/2 + (i+0.5)*shelf_spacing_y + np.random.uniform(-obstacle_density, obstacle_density)
                z = obstacle_size[2]/2
                distance = np.linalg.norm(np.array([x, y]) - drone_start_pos[:2])
                if distance <= obstacle_size[0]:
                    continue

                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[dim/2 for dim in obstacle_size],
                    physicsClientId=self.CLIENT
                )
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[dim/2 for dim in obstacle_size],
                    rgbaColor=[0.6,0.4,0.2,1],
                    physicsClientId=self.CLIENT
                )
                obs_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x,y,z],
                    physicsClientId=self.CLIENT
                )
                self.obstacle_ids.append(obs_id)

        logging.info(f"Added {len(self.obstacle_ids)} obstacles to the environment.")

    def set_target(self, new_target):
        if new_target.shape != (12,):
            raise ValueError("new_target must be a 12D vector.")
        self.x_target = new_target
        logging.info(f"Target updated to: {self.x_target}")

    def _initialize_mpc_matrices(self):
        """Match the same A,B as TinyMPC, ignoring c since we'll apply gravity externally."""
        mass = self.M
        Ixx, Iyy, Izz = self.J[0,0], self.J[1,1], self.J[2,2]

        d_x = 9.1785e-7
        d_y = 9.1785e-7
        d_z = 10.311e-7

        # dt
        dt = self.dt

        # Same A as TinyMPC but ignoring gravity
        A = np.eye(12)
        A[0,3] = dt
        A[1,4] = dt
        A[2,5] = dt
        A[3,7] = dt / mass  # v_x depends on pitch
        A[4,6] = -dt / mass # v_y depends on roll
        A[3,3] = 1 - d_x*dt
        A[4,4] = 1 - d_y*dt
        A[5,5] = 1 - d_z*dt
        A[6,9]  = dt
        A[7,10] = dt
        A[8,11] = dt

        B_full = np.zeros((12,4))
        B_full[5,0]  = dt / mass
        B_full[9,1]  = dt / Ixx
        B_full[10,2] = dt / Iyy
        B_full[11,3] = dt / Izz

        # We'll apply c = [0, ..., -dt*g, ...] externally in step(), not in do-mpc

        Q = np.diag([
            2000, 2000, 3000,
             500,  500,  500,
               5,    5,    5,
               1,    1,    1
        ])
        R = np.diag([1.0, 0.2, 0.2, 0.2])

        # Use the same environment clamp for final step
        u_min = np.array([0, -np.pi/3, -np.pi/3, -np.pi/3])
        u_max = np.array([20,  np.pi/3,  np.pi/3,  np.pi/3])

        self.A = A
        self.B_full = B_full
        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max

    def _dompc_setup(self):
        """
        Build a do-mpc model with e_{k+1} = A e_k + B u_k for the error-state e = x - x_target.
        """
        # We'll define a discrete model where the 'state' is error e in R^{12}:
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)

        # e_k in R^{12}, u_k in R^4
        e_var = model.set_variable('_x','e', shape=(12,1))
        u_var = model.set_variable('_u','u', shape=(4,1))

        # There's no parameter for x_target, because we incorporate that in e = x - x_target outside
        # => e_{k+1} = A e_k + B u_k
        # ignoring gravity in the model so it matches TinyMPC's approach

        A_dm = DM(self.A)
        B_dm = DM(self.B_full)

        e_next = A_dm @ e_var + B_dm @ u_var
        model.set_rhs('e', e_next)
        model.setup()

        # Create MPC
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': self.N,
            't_step': self.dt,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            'n_robust': 0
        }
        mpc.set_param(**setup_mpc)

        Q_dm = DM(self.Q)
        R_dm = DM(self.R)
        # cost = e^T Q e + u^T R u
        lterm = e_var.T @ Q_dm @ e_var + u_var.T @ R_dm @ u_var
        mterm = e_var.T @ Q_dm @ e_var
        mpc.set_objective(mterm=mterm[0,0], lterm=lterm[0,0])

        # We'll set do-mpc's big internal bounds e.g. +/- 1000
        # Then clamp to self.u_min, self.u_max at the end
        nx = 12
        nu = 4
        e_lo = -1e3 * np.ones(nx)
        e_hi = +1e3 * np.ones(nx)
        u_lo = -1e3 * np.ones(nu)
        u_hi = +1e3 * np.ones(nu)

        mpc.bounds['lower','_x','e'] = e_lo
        mpc.bounds['upper','_x','e'] = e_hi
        mpc.bounds['lower','_u','u'] = u_lo
        mpc.bounds['upper','_u','u'] = u_hi

        mpc.setup()
        self.mpc_model = model
        self.mpc = mpc

    def step(self, action=None):
        t = self.step_counter // self.PYB_STEPS_PER_CTRL
        if t >= self.max_steps:
            terminated = True
            truncated = False
            reward = 0.0
            info = self._computeInfo()
            return self._computeObs(), reward, terminated, truncated, info

        current_state = self._get_current_state()
        self.state_history[t,:] = current_state

        # e_current = x - x_target
        e_current = current_state - self.x_target
        u_opt = self._solve_mpc(e_current)

        self.control_history[t,:] = u_opt

        # Apply control externally, including gravity, exactly like TinyMPC:
        thrust_z, torque_x, torque_y, torque_z = u_opt
        print(f"Time {t*self.dt:.2f}s - Control: "
              f"Tz={thrust_z:.2f}, Tx={torque_x:.2f}, Ty={torque_y:.2f}, Tz={torque_z:.2f}")

        # The environment side: apply thrust, torque, plus gravity
        # We'll do it like TinyMPC:
        self._apply_control_inputs(thrust_z, torque_x, torque_y, torque_z)

        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.CLIENT)
            time.sleep(self.PYB_TIMESTEP)

        self._updateAndStoreKinematicInformation()

        if t+1 <= self.max_steps:
            self.state_history[t+1,:] = self._get_current_state()

        dist = np.linalg.norm(current_state[:3] - self.x_target[:3])
        if dist < self.proximity_threshold:
            terminated = False
            truncated = False
            logging.info(f"Target reached at step {t}, time {t*self.dt:.2f}s.")
        else:
            terminated = False
            truncated = False

        reward = -dist
        self.step_counter += self.PYB_STEPS_PER_CTRL
        return self._computeObs(), reward, terminated, truncated, self._computeInfo()

    def _solve_mpc(self, e_current):
        """
        Solve do-mpc for the error-state e = x - x_target. Then clamp to environment bounds.
        """
        # Force do-mpc to solve from e_current
        u_sol = self.mpc.make_step(e_current.reshape((12,1)))
        u_opt = np.array(u_sol).flatten()

        # clamp to the environment's smaller bounds
        u_opt = np.clip(u_opt, self.u_min, self.u_max)
        return u_opt

    def _apply_control_inputs(self, thrust_z, torque_x, torque_y, torque_z):
        """
        Matches how TinyMPC applies them: no gravity in the solver, so we do it externally.
        We'll let PyBullet handle gravity automatically or you can add an external force c[5] = -dt*g if you want.
        """
        thrust_body = np.array([0,0,thrust_z])
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

    def _get_current_state(self):
        """
        Same as TinyMPC. We just read the drone's position, orientation, etc.
        """
        drone_id = self.DRONE_IDS[0]
        pos, orn = p.getBasePositionAndOrientation(drone_id)
        lin_vel, ang_vel = p.getBaseVelocity(drone_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        return np.array([
            pos[0], pos[1], pos[2],
            lin_vel[0], lin_vel[1], lin_vel[2],
            roll, pitch, yaw,
            ang_vel[0], ang_vel[1], ang_vel[2]
        ])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        init_state = self._get_current_state()
        self.state_history[0,:] = init_state

        # Remove old obstacles, re-add if needed
        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []
        if self.obstacle_config and self.obstacles:
            self._addObstacles()

        # Re-init do-mpc solver with the new error
        e0 = init_state - self.x_target
        self.mpc.reset_history()
        self.mpc.x0 = e0.reshape((12,1))
        self.mpc.u0 = np.zeros((4,1))
        self.mpc.set_initial_guess()

        return init_state, info

    def close(self):
        if hasattr(self, 'obstacle_ids') and self.obstacle_ids:
            for obs_id in self.obstacle_ids:
                p.removeBody(obs_id, physicsClientId=self.CLIENT)
            self.obstacle_ids = []
            logging.info("Removed all obstacles.")
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

        fig = plt.figure(figsize=(18,6))
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(self.state_history[:steps+1,0],
                self.state_history[:steps+1,1],
                self.state_history[:steps+1,2], label='Drone Traj')
        ax.scatter(self.x_target[0], self.x_target[1], self.x_target[2],
                   color='r', marker='*', s=100, label='Target')
        ax.set_title('3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)

        ax2 = fig.add_subplot(132)
        ax2.plot(self.state_history[:steps+1,0],
                 self.state_history[:steps+1,1], 'b-', label='XY path')
        ax2.plot(self.x_target[0], self.x_target[1], 'ro', label='Target')
        ax2.set_title('Top-Down')
        ax2.axis('equal')
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(133)
        ax3.plot(time_array, self.state_history[:steps+1,6], label='Roll')
        ax3.plot(time_array, self.state_history[:steps+1,7], label='Pitch')
        ax3.plot(time_array, self.state_history[:steps+1,8], label='Yaw')
        ax3.set_title('Orientation Over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (rad)')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        plt.show()

        # Control inputs
        plt.figure(figsize=(14,6))
        time_array_ctrl = np.linspace(0, steps*self.dt, steps)
        plt.subplot(2,1,1)
        plt.plot(time_array_ctrl, self.control_history[:steps,0], label='Thrust Z')
        plt.title('Control Inputs Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(time_array_ctrl, self.control_history[:steps,1], label='Torque X')
        plt.plot(time_array_ctrl, self.control_history[:steps,2], label='Torque Y')
        plt.plot(time_array_ctrl, self.control_history[:steps,3], label='Torque Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
