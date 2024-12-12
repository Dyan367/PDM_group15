import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from scipy.spatial.transform import Rotation
import cvxpy as cp

from scipy.linalg import expm

class MPCControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 dt: float=0.01 ## dt for discretization of state space
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in MPCControl.__init__(), MPCControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        # Quadrotor Parameters
        # [INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:
        # [INFO] m 0.027000, L 0.039700,
        # [INFO] ixx 0.000014, iyy 0.000014, izz 0.000022,
        # [INFO] kf 0.000000, km 0.000000,
        # [INFO] t2w 2.250000, max_speed_kmh 30.000000,
        # [INFO] gnd_eff_coeff 11.368590, prop_radius 0.023135,
        # [INFO] drag_xy_coeff 0.000001, drag_z_coeff 0.000001,
        # [INFO] dw_coeff_1 2267.180000, dw_coeff_2 0.160000, dw_coeff_3 -0.110000

        self.m = 0.027000   # Mass of the quadrotor (kg)
        self.l = 0.039700 # Length of the quadrotor arm (m)
        self.I_x = 0.000014  # Moment of inertia about x-axis (kg·m^2)
        self.I_y = 0.000014  # Moment of inertia about y-axis (kg·m^2)
        self.I_z = 0.000022  # Moment of inertia about z-axis (kg·m^2)

        # State-space matrices DIFFERENTIALLY FLAT INPUTS !!state vector = [x y z yaw xdot ydot zdot yawdot]!!
        self.linear_drone_A = np.zeros((8, 8))
        self.linear_drone_B = np.zeros((8, 2))

        self.linear_drone_A[0:4, 4:8] = np.eye(4)  # Position and yaw kinematics

        self.linear_drone_B[6, 0] = 1 / self.m  # Thrust affects z acceleration
        self.linear_drone_B[7, 1] = 1 / self.I_z  # Torque affects yaw acceleration

        self.dt = 1/48
        #self.Ad, self.Bd = discretize_tustin(self.linear_drone_A,self.linear_drone_B, 0.01)

        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy,
                       thrust_vector,
                       target_vel,
                       target_rpy_rates=np.zeros(3),
                       use_MPC=True
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        # #des_total_thrust, des_tau_z, des_roll, des_pitch = self._MPCController( cur_pos,
        #                                                                         cur_quat,
        #                                                                         cur_vel,
        #                                                                         cur_ang_vel,
        #                                                                         target_pos=target_pos,
        #                                                                         target_vel=target_vel,
        #                                                                         target_acc=target_acc,
        #                                                                         dt=self.dt
        #                                                                         )
        
        pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        if use_MPC:
            thrust, computed_target_rpy = self._MPCPositionControl(control_timestep,
                                                                            cur_pos,
                                                                            cur_quat,
                                                                            cur_vel,
                                                                            
                                                                            target_rpy,
                                                                            
                                                                            thrust_vector
                                                                            )
            rpm = self._MPCAttitudeControl(control_timestep,
                                            thrust,
                                            cur_quat,
                                            computed_target_rpy,
                                            target_rpy_rates
                                            )
            cur_rpy = p.getEulerFromQuaternion(cur_quat)
            return rpm, computed_target_rpy[2] - cur_rpy[2]
        else:
            rpm,_,_ =pid_controller.computeControl(control_timestep=control_timestep,
                                                    cur_pos=cur_pos,
                                                    cur_quat=cur_quat,
                                                    cur_vel=cur_vel,
                                                    cur_ang_vel=cur_ang_vel,
                                                    target_pos=target_pos, # same as the current position
                                                    target_rpy=target_rpy, # keep current yaw
                                                    target_vel=target_vel # target the desired velocity vector
                                                    )
            return rpm, 0.0
    
    ################################################################################

    def _MPCPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               
                               target_rpy,
                               
                               thrust_vector
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        
        
        #### MPC target thrust #####################################
        target_thrust = thrust_vector[0:3]
        print("thrust:",target_thrust)
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))

        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        #### Here we compute the desired rotation matrix aka roll, pitch, yaw angles to feed to attitude controller
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        print("rpy",target_euler)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._MPCPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler
    
    ################################################################################

    def _MPCAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in MPCControl._one23DInterface()")
            exit()
    
    ###############################################################################

#     def _MPCController(self, 
#                         cur_pos,
#                         cur_quat,
#                         cur_vel,
#                         cur_ang_vel,
#                         dt=0.01,
#                         target_pos=np.zeros(3),
#                         target_vel=np.zeros(3),
#                         target_acc=np.zeros(3),
#                         horizon=5
#                         ):
#         """
#         Improved MPC controller with correctly dimensioned weight matrices.

#         Parameters
#         ----------
#         target_state : ndarray
#             (13,1)-shaped array of floats containing the desired state.

#         Returns
#         -------
#         ndarray
#             (4,1)-shaped array of integers containing the target thrust and torque.
#         """
#         # Extract current roll, pitch, yaw from quaternion
#         cur_rpy = p.getEulerFromQuaternion(cur_quat)

#         # Current state / initial state ! 
#         x0 = np.array([cur_pos[0], cur_pos[1], cur_pos[2], cur_rpy[2], cur_vel[0], cur_vel[1], cur_vel[2], cur_ang_vel[2]]) 

        
#         desired_yaw = 0.0  # Desired yaw angle (rad)
#         x_des = np.array([target_pos[0], target_pos[1], target_pos[2], desired_yaw, target_vel[0], target_vel[1], target_vel[2], 0.0])

#         print("MPC x_des: ", x_des)
#         # Get state and input dimensions from the discretized matrices
#         nx = self.Ad.shape[0]  # State dimension (12)
#         nu = self.Bd.shape[1]  # Control input dimension (4)

#         # Correctly dimensioned cost matrices
#         # Define optimization variables
#         X = cp.Variable((8, horizon + 1))  # States
#         U = cp.Variable((2, horizon))  # Inputs: thrust and yaw torque

#         # Cost function and constraints
#         Q = np.diag([1000, 1000, 1000, 1, 1, 1, 1, 1])  # State weights
#         R = np.diag([0.1, 0.1])  # Input weights

#         # Initial state constraint
#         constraints = [X[:, 0] == x0]
#         thrust_max = 2 * self.m * 9.81  # Maximum thrust (e.g., 2g)

#         # Cost function
#         cost = 0
#         for k in range(horizon):
#             # Cost function
#             cost += cp.quad_form(X[:, k] - x_des, Q) + cp.quad_form(U[:, k], R)

#             # Dynamics constraint
#             constraints += [X[:, k + 1] == self.Ad @ X[:, k] + self.Bd @ U[:, k]]
#             constraints += [U[0, k] >= 0, U[0, k] <= thrust_max]  # Thrust

        

#         cost += cp.quad_form(X[:, horizon] - x_des, Q)

#         # Initial condition
#         constraints += [X[:, 0] == x0]

#         # Solve the problem
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve()

#         # Check for solution
#         if problem.status == cp.OPTIMAL:
#             optimal_states = X.value
#             optimal_inputs = U.value
#             #print("MPC U: ", optimal_inputs[:,0])
#         else:
#             print("No optimal solution found.")

#         ## Optimal thrust and yaw torque
#         des_total_thrust = optimal_inputs[0, 0]
#         des_tau_z = optimal_inputs[1, 0]
#         ## Using differential flatness !!!!!!!!! so we can compute roll and pitch and feed this to attitude controller
#         des_roll, des_pitch = compute_roll_pitch(target_acc, 0.0, self.m)
#         print("MPC des_roll: ", des_roll, "MPC des_pitch: ", des_pitch,"MPC des_total_thrust: ", des_total_thrust)

#         return des_total_thrust, des_tau_z, des_roll, des_pitch

# ## discretize the state-space matrices using the Tustin method for MPC controller
# def discretize_tustin(A, B, dt):
#     """Discretize the state-space matrices using the Tustin method."""
#     dt = 1/48
#     n = A.shape[0]
#     I = np.eye(n)
#     Ad = np.linalg.inv(I - 0.5 * A * dt) @ (I + 0.5 * A * dt)
#     Bd = np.linalg.inv(I - 0.5 * A * dt) @ (B * dt)
#     return Ad, Bd

# ## computes the desired roll and pitch angles based on output of MPC
# def compute_roll_pitch(acc_des, yaw, mass):
#     ax, ay, az = acc_des
#     Fz = mass * (9.81 + az)
#     Fx = mass * ax
#     Fy = mass * ay

#     # Transform forces to body frame
#     Fx_b = Fx * np.cos(yaw) + Fy * np.sin(yaw)
#     Fy_b = -Fx * np.sin(yaw) + Fy * np.cos(yaw)

#     # Roll and Pitch calculations
#     des_roll = np.arcsin(Fy_b / np.sqrt(Fx_b**2 + Fy_b**2 + Fz**2))
#     des_pitch = np.arctan(Fx_b / Fz)
    
#     return des_roll, des_pitch

