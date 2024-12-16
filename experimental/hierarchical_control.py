import numpy as np
import cvxpy as cp
from scipy.linalg import expm

class hierarchical_control():
    def __init__(self, Q_pos, R_pos,Q_attitude,R_attitude, N):
        self.Q_pos = Q_pos
        self.R_pos = R_pos
        self.Q_attitude = Q_attitude
        self.R_attitude = R_attitude
        self.N = N

        self.KM = 7.94e-12 # moment coefficient
        self.KF = 3.16e-10 #thrust coefficient
        
        self.mixer_matrix = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        
        self.pwm2rpm_scale = 0.2685
        self.pwm2rpm_const = 4070.3
        self.min_pwm = 20000
        self.max_pwm = 65535
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.m = 0.027   # Mass of the quadrotor (kg)
        self.I_x = 14e-5  # Moment of inertia about x-axis (kg·m^2)
        self.I_y = 14e-5  # Moment of inertia about y-axis (kg·m^2)
        self.I_z = 22e-5

        self.Inertia = np.array([[self.I_x, 0, 0], [0, self.I_y, 0], [0, 0, self.I_z]])

        # x = [x, y, z, vx, vy, vz]
        # u = [u1, r, p ]
        self.A_position = np.zeros((6, 6))
        self.A_position[0:3, 3:6] = np.eye(3)
        self.B_position = np.zeros((6, 3))
        self.B_position[5, 0] = 1 / self.m
        self.B_position[4, 1] = -self.g
        self.B_position[3, 2] = self.g

        # x = [r, p, y, wx, wy, wz]
        # u = [u2, u3, u4]
        self.A_attitude = np.zeros((6, 6))
        self.A_attitude[0:3, 3:6] = np.eye(3)
        self.B_attitude = np.zeros((6, 3))
        self.B_attitude[3,0] = 1/self.I_x
        self.B_attitude[4,1] = 1/self.I_y
        self.B_attitude[5,2] = 1/self.I_z

        self.Ad_position, self.Bd_position = discretize_zoh(self.A_position, self.B_position, 0.01)
        self.Ad_attitude, self.Bd_attitude = discretize_zoh(self.A_attitude, self.B_attitude, 0.01)

        
    def compute_position_mpc(self, x0, x_des):
        """
        Position MPC: Computes thrust and desired angles based on desired position.
        Args:
            x0: Initial state [x, y, z, vx, vy, vz].
            x_des: Desired position [x_des, y_des, z_des,vx_des,vy_des,vz_des].
        Returns:
            u_position: Optimal control inputs [u1, r, p].
        """
        x = cp.Variable((6, self.N + 1))  # States
        u = cp.Variable((3, self.N))  # Inputs [u1, r, p]
        v = 5.0  # Maximum velocity
        max_velocity = np.array([v, v, v])  # Maximum velocity
        min_velocity = -max_velocity  # Minimum velocity
        cost = 0
        constraints = [x[:, 0] == x0]

        

        for t in range(self.N):
            cost += cp.quad_form(x[:, t] - x_des, self.Q_pos)
            cost += cp.quad_form(u[:, t], self.R_pos)
            constraints += [x[:, t + 1] == self.Ad_position @ x[:, t] + self.Bd_position @ u[:, t]]
            #constraints += [u[0, t] >= 0]  # Thrust must be non-negative
            # constraints += [
            #     u[1, t] >= -np.pi / 12,  # Constrain roll (r) lower bound
            #     u[1, t] <= np.pi / 12,   # Constrain roll (r) upper bound
            #     u[2, t] >= -np.pi / 12,  # Constrain pitch (p) lower bound
            #     u[2, t] <= np.pi / 12    # Constrain pitch (p) upper bound
            # ]
        #     constraints += [
        #     x[3, t] >= min_velocity[0],  # vx lower bound
        #     x[3, t] <= max_velocity[0],  # vx upper bound
        #     x[4, t] >= min_velocity[1],  # vy lower bound
        #     x[4, t] <= max_velocity[1],  # vy upper bound
        #     x[5, t] >= min_velocity[2],  # vz lower bound
        #     x[5, t] <= max_velocity[2]   # vz upper bound
        # ]

        # Terminal cost
        cost += cp.quad_form(x[:, self.N] - x_des, self.Q_pos)

        

        # Solve the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # Return the first control input
        return u[:, 0].value , x.value

    def compute_attitude_mpc(self, x0, attitude_des):
        """
        Attitude MPC: Computes torques to track desired angles.
        Args:
            x0: Initial state [phi, theta, psi, wx, wy, wz].
            attitude_des: Desired angles [phi_des, theta_des, psi_des,wx,wy,wz].
        Returns:
            u_attitude: Optimal control inputs [u2, u3, u4].
        """
        x = cp.Variable((6, self.N + 1))  # States
        u = cp.Variable((3, self.N))  # Inputs [u2, u3, u4]
        v = 5.0  # Maximum velocity
        max_velocity = np.array([v, v, v])  # Maximum angular velocity
        min_velocity = -max_velocity

        cost = 0
        constraints = [x[:, 0] == x0]
        
        for t in range(self.N):
            cost += cp.quad_form(x[:, t] - attitude_des, self.Q_attitude)
            cost += cp.quad_form(u[:, t], self.R_attitude)
            constraints += [x[:, t + 1] == self.Ad_attitude @ x[:, t] + self.Bd_attitude @ u[:, t]]
        #     constraints += [
        #     x[3, t] >= min_velocity[0],  # wx lower bound
        #     x[3, t] <= max_velocity[0],  # wx upper bound
        #     x[4, t] >= min_velocity[1],  # wy lower bound
        #     x[4, t] <= max_velocity[1],  # wy upper bound
        #     x[5, t] >= min_velocity[2],  # wz lower bound
        #     x[5, t] <= max_velocity[2]   # wz upper bound
        # ]
            

        # Terminal cost
        cost += cp.quad_form(x[:, self.N] - attitude_des, self.Q_attitude)

        # Terminal constraint: Ensure terminal state lies in terminal set
        #constraints += [cp.quad_form(x[:, self.N], self.P_attitude) <= 1.0]  # Example terminal set bound


        # Solve the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # Return the first control input
        return u[:, 0].value, x.value

    def compute_mpc(self, x0, x_des):
        """
        Computes the hierarchical control solution by calling position and attitude MPCs.
        Args:
            x0_position: Initial state for position [x, y, z, vx, vy, vz].
            x_des: Desired position [x_des, y_des, z_des].
            x0_attitude: Initial state for attitude [phi, theta, psi, wx, wy, wz].
            attitude_des: Desired angles [phi_des, theta_des, psi_des].
        Returns:
            u_position: Optimal control inputs for position [u1, r, p].
            u_attitude: Optimal control inputs for attitude [u2, u3, u4].
        """
        x0_position = x0[:6]
        x_des_position = x_des[:6]
        x0_attitude = x0[6:]
        u_position ,x_predicted_position= self.compute_position_mpc(x0_position, x_des_position)
        attitude_des = np.hstack((u_position[1:],[0.0],x_des[9:]))  # Extract desired angles [r, p] from position MPC
        u_attitude,x_predicted_attitude = self.compute_attitude_mpc(x0_attitude, attitude_des)

        total_thrust = u_position[0] #+self.m*self.g
        tau = u_attitude 
        print("Optimal thrust: ", total_thrust)
        print("Optimal tau: ", tau)
        return total_thrust, tau , np.hstack((x_predicted_position,x_predicted_attitude))
    
    def u2rpms(self,total_thrust, tau):
        total_thrust = (np.sqrt(total_thrust / (4*self.KF)) - self.pwm2rpm_const) / self.pwm2rpm_scale
    
        pwm = total_thrust + np.dot(self.mixer_matrix,tau)
        print("Optimal pwm: ", pwm)
        pwm = np.clip(pwm, self.min_pwm, self.max_pwm)
        
        rpms = self.pwm2rpm_scale * pwm + self.pwm2rpm_const

        return rpms

def discretize_zoh(A, B, dt):
    """
    Discretize the state-space matrices using the Zero-Order Hold (ZOH) method.
    
    Parameters:
        A (numpy.ndarray): Continuous-time state matrix (n x n)
        B (numpy.ndarray): Continuous-time input matrix (n x m)
        dt (float): Sampling time

    Returns:
        A_d (numpy.ndarray): Discrete-time state matrix
        B_d (numpy.ndarray): Discrete-time input matrix
    """
    # Combine A and B into one matrix for computation
    n = A.shape[0]
    augmented_matrix = np.block([
        [A, B],
        [np.zeros((B.shape[1], A.shape[1] + B.shape[1]))]
    ])

    # Exponential of the augmented matrix
    exp_matrix = expm(augmented_matrix * dt)

    # Extract A_d and B_d from the result
    A_d = exp_matrix[:n, :n]
    B_d = exp_matrix[:n, n:]
    
    return A_d, B_d

