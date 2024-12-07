import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Quadrotor Parameters
g = 9.81  # Gravity (m/s^2)
m = 1.0   # Mass of the quadrotor (kg)
I_x = 0.01  # Moment of inertia about x-axis (kg·m^2)
I_y = 0.01  # Moment of inertia about y-axis (kg·m^2)
I_z = 0.02  # Moment of inertia about z-axis (kg·m^2)

# State-space matrices
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
    [np.array([[0, 1 / I_x, 0, 0],
               [0, 0, 1 / I_y, 0],
               [0, 0, 0, 1 / I_z]])]
])

# Define Q and R matrices for LQR
pq = 80
vq = 1
aq = 10
wq = 1
rr1 = 1
rr = 10
Q = np.diag([pq, pq, pq, vq, vq, vq, aq, aq, aq, wq, wq, wq])
R = np.diag([rr1, rr, rr, rr])

# Solve the Riccati equation to get P
P = solve_continuous_are(A, B, Q, R)

# Compute the LQR gain matrix K
K = np.linalg.inv(R) @ B.T @ P

# Define the LQR control law
def lqr_control(state, desired_state):
    # Compute state error
    error = state - desired_state
    # Compute control input
    u = -K @ error
    return u

# Trajectory Generator
def generate_trajectory(start_point, end_point, total_time, dt):
    """
    Generate a smooth trajectory between two 3D points.
    
    Args:
        start_point (array-like): [x, y, z] starting position.
        end_point (array-like): [x, y, z] ending position.
        total_time (float): Time duration for the trajectory.
        dt (float): Time step.
    
    Returns:
        np.ndarray: Trajectory array of shape (N, 3), where N is the number of timesteps.
    """
    timesteps = int(total_time / dt)
    time = np.linspace(0, total_time, timesteps)
    trajectory = np.zeros((timesteps, 3))
    for i in range(3):  # x, y, z
        trajectory[:, i] = np.linspace(start_point[i], end_point[i], timesteps)
    return trajectory

# Simulation parameters
dt = 0.01  # Time step
T_total = 20  # Total simulation time
timesteps = int(T_total / dt)
time = np.linspace(0, T_total, timesteps)

# Initial state: [x, y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi]
state = np.zeros(12)  # Start at rest
trajectory_states = []

# Generate a trajectory between two points
start_point = [0, 0, 5]
end_point = [5, 5, 5]
total_time_per_trajectory = 5  # Time to move between points (seconds)
trajectory = generate_trajectory(start_point, end_point, total_time_per_trajectory, dt)

# Simulate dynamics
trajectory_idx = 0
trajectory_length = len(trajectory)

for t in range(timesteps - 1):
    if trajectory_idx < trajectory_length:
        # Get the current desired position from the trajectory
        desired_position = trajectory[trajectory_idx]
        desired_state = np.hstack((desired_position, [0, 0, 0], [0, 0, 0], [0, 0, 0]))
        
        # Compute control input using LQR
        u = lqr_control(state, desired_state)
        
        # Clamp control inputs to avoid instability
        u = np.clip(u, -10, 10)

        # Advance the dynamics
        sol = solve_ivp(lambda t, x: A @ x + B @ u, [t * dt, (t + 1) * dt], state, t_eval=[(t + 1) * dt])
        state = sol.y.flatten()
        trajectory_idx += 1  # Move along the trajectory
    else:
        # Hover at the last point
        desired_state = np.hstack((end_point, [0, 0, 0], [0, 0, 0], [0, 0, 0]))
        u = lqr_control(state, desired_state)
        
    # Record the state
    trajectory_states.append(state)

trajectory_states = np.array(trajectory_states)

# 3D Animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_zlim(0, 6)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Drone Following Trajectory with LQR Controller")

# Initial plot
trajectory_line, = ax.plot([], [], [], lw=2, label="Drone Path")
drone_point, = ax.plot([], [], [], 'ro', label="Drone")
trajectory_path, = ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'go', label="Trajectory", markersize=1)


# Animation update function
def update(frame):
    trajectory_line.set_data(trajectory_states[:frame + 1, 0], trajectory_states[:frame + 1, 1])
    trajectory_line.set_3d_properties(trajectory_states[:frame + 1, 2])
    
    drone_point.set_data([trajectory_states[frame, 0]], [trajectory_states[frame, 1]])
    drone_point.set_3d_properties([trajectory_states[frame, 2]])
    return trajectory_line, drone_point

# Animate
ani = FuncAnimation(fig, update, frames=timesteps, interval=dt * 1000, blit=False)

# Show plot
plt.legend()
plt.show()
