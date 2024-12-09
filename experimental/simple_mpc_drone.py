import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import cvxpy as cp


# Quadrotor Parameters from pybullet
# [INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:
# [INFO] m 0.027000, L 0.039700,
# [INFO] ixx 0.000014, iyy 0.000014, izz 0.000022,
# [INFO] kf 0.000000, km 0.000000,
# [INFO] t2w 2.250000, max_speed_kmh 30.000000,
# [INFO] gnd_eff_coeff 11.368590, prop_radius 0.023135,
# [INFO] drag_xy_coeff 0.000001, drag_z_coeff 0.000001,
# [INFO] dw_coeff_1 2267.180000, dw_coeff_2 0.160000, dw_coeff_3 -0.110000
# Quadrotor Parameters
g = 9.81  # Gravity (m/s^2)
m = 0.027000   # Mass of the quadrotor (kg)
I_x = 0.000014  # Moment of inertia about x-axis (kg·m^2)
I_y = 0.000014  # Moment of inertia about y-axis (kg·m^2)
I_z = 0.000022  # Moment of inertia about z-axis (kg·m^2)
l = 0.039700  # Length of the quadrotor arm (m)

# Continuous-time state-space matrices
A = np.block([
    [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
    [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), np.array([[0, g, 0], [-g, 0, 0], [0, 0, 0]]), np.zeros((3, 3)), np.zeros((3, 3))],
    [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
])

B = np.block([
    [np.zeros((3, 4))],
    [np.zeros((3, 4))],
    [np.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1 / m, 0, 0, 0]])],
    [np.array([[0, 1*l / I_x, 0, 0],
               [0, 0, 1*l / I_y, 0],
               [0, 0, 0, 1*l / I_z]])]
])

# Tustin discretization function
def tustin_discretization(A, B, dt):
    I = np.eye(A.shape[0])  # Identity matrix of size A
    A_d = np.linalg.inv(I - 0.5 * A * dt) @ (I + 0.5 * A * dt)
    B_d = np.linalg.inv(I - 0.5 * A * dt) @ (B * dt)
    return A_d, B_d

# Discretize the matrices
dt = 0.05  # Time step
A_d, B_d = tustin_discretization(A, B, dt)

# MPC parameters
N_horizon = 10  # Prediction horizon
Q_mpc = np.eye(12) * 5  # State cost
R_mpc = np.eye(4)        # Input cost
u_min = np.array([-10, -10, -10, -10])  # Control input bounds
u_max = np.array([10, 10, 10, 10])

# Define MPC control function
def mpc_control(state, desired_state):
    x = cp.Variable((12, N_horizon + 1))
    u = cp.Variable((4, N_horizon))
    
    cost = 0
    constraints = []
    constraints += [x[:, 0] == state]  # Initial state constraint
    
    for t in range(N_horizon):
        # Cost function
        cost += cp.quad_form(x[:, t] - desired_state, Q_mpc)
        cost += cp.quad_form(u[:, t], R_mpc)
        
        # Dynamics and input constraints
        constraints += [
            x[:, t+1] == A_d @ x[:, t] + B_d @ u[:, t],
            u_min <= u[:, t],
            u[:, t] <= u_max
        ]
    
    # Terminal cost
    cost += cp.quad_form(x[:, N_horizon] - desired_state, Q_mpc)
    
    # Solve the optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    # Return the first control input
    return u[:, 0].value

