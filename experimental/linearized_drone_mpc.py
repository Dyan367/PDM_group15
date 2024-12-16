import numpy as np
import cvxpy as cp
from scipy.linalg import expm

class rpm_calc():
    def __init__(self, Q, R, N, dt=0.01):
        """
        Initializes the MPC controller.
        
        Parameters:
        - A, B: Linearized state-space matrices
        - mixer_matrix: Mixer matrix for mapping thrust and torques to motor RPMs
        - pwm2rpm_scale, pwm2rpm_const: Conversion parameters for PWM to RPM
        - min_pwm, max_pwm: Minimum and maximum PWM values
        - Q, R: Cost matrices for state and input penalties
        - N: Prediction horizon
        - dt: Time step
        """
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.m = 0.027000   # Mass of the quadrotor (kg)
        self.I_x = 1.4e-1  # Moment of inertia about x-axis (kg·m^2)
        self.I_y = 1.4e-1  # Moment of inertia about y-axis (kg·m^2)
        self.I_z = 2.17e-1
        self.l = 0.039700
        self.A = np.zeros((12, 12))
        self.A[0:3, 3:6] = np.eye(3)  # Position derivatives (velocity)
        self.A[3:6, 6:9] = np.array([[0, self.g, 0], [-self.g, 0, 0], [0, 0, 0]])  # Gravity coupling
        #self.A[6:9, 9:12] = np.eye(3)  # Angular position derivatives (angular velocities)

        # Define the B matrix
        self.B = np.zeros((12, 4))
        self.B[5, 0] = 1 / self.m  # Thrust affects vertical acceleration
        self.B[9, 1] = 1 / self.I_x  # Roll torque affects roll angular acceleration
        self.B[10, 2] = 1 / self.I_y  # Pitch torque affects pitch angular acceleration
        self.B[11, 3] = 1/ self.I_z  # Yaw torque affects yaw angular acceleration
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
        self.Q = Q
        self.R = R
        self.N = N
        self.dt = dt
        self.Ad, self.Bd = discretize_zoh(self.A, self.B, self.dt)

    def solve_mpc(self, current_state, desired_state):
        """
        Solves the MPC optimization problem to compute the control inputs.
        
        Parameters:
        - current_state: Current state vector
        - desired_state: Desired state vector
        
        Returns:
        - thrust: Total thrust
        - tau: [tau_x, tau_y, tau_z] torques
        - rpms: Motor RPMs
        """
        nx = self.Ad.shape[0]  # State dimension
        nu = self.Bd.shape[1]  # Input dimension

        # Decision variables
        x = cp.Variable((nx, self.N + 1))  # States over the horizon
        u = cp.Variable((nu, self.N))     # Inputs over the horizon

        # Objective function
        thrust = 0
        tau = 0
        cost = 0
        constraints = []

        # Initial state constraint
        constraints.append(x[:, 0] == current_state)

        for k in range(self.N):
            # Dynamics constraint
            constraints.append(x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ (u[:, k]+ np.array([self.m * self.g,0,0,0])))
            #input constraints
            
            constraints.append(u[0, k] >= 0)  # Thrust >= 0
            constraints.append(u[0, k] >= 0)  # Thrust >= 0
            

            # Accumulate cost
            cost += cp.quad_form(x[:, k] - desired_state, self.Q)
            cost += cp.quad_form(u[:, k], self.R)

        # Terminal cost
        cost += cp.quad_form(x[:, self.N] - desired_state, self.Q)

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError("MPC optimization did not converge.")

        # Extract first control input
        optimal_u = u[:, 0].value
        predicted_states = x.value
        thrust = optimal_u[0]
        tau = optimal_u[1:4]

        print("Optimal tau: ", optimal_u)
        

        # Map to motor RPMs using mixer matrix
        #thrust += self.m * self.g
        # thrust = np.maximum(0, thrust)
        
        thrust += self.m * self.g
        
       
        thrust = (np.sqrt(thrust / (4*self.KF)) - self.pwm2rpm_const) / self.pwm2rpm_scale
        
        pwm = thrust + np.dot(self.mixer_matrix,tau)
        print("Optimal pwm: ", pwm)
        pwm = np.clip(pwm, self.min_pwm, self.max_pwm)
        
        rpms = self.pwm2rpm_scale * pwm + self.pwm2rpm_const
        
        return rpms, predicted_states, thrust, tau
    

def discretize_tustin(A, B, dt):
    """Discretize the state-space matrices using the Tustin method."""
    
    n = A.shape[0]
    I = np.eye(n)
    Ad = np.linalg.inv(I - 0.5 * A * dt) @ (I + 0.5 * A * dt)
    Bd = np.linalg.inv(I - 0.5 * A * dt) @ (B * dt)
    return Ad, Bd

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





