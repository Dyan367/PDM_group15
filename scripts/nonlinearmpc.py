import sympy as sp
import numpy as np
import casadi as ca
import minsnap_trajectories as ms

def quadrotor_state_and_jacobian():
    # Define symbolic variables
    t = sp.symbols('t')
    x, y, z = sp.Function('x')(t), sp.Function('y')(t), sp.Function('z')(t)
    phi, theta, psi = sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t)
    omega1, omega2, omega3, omega4 = sp.symbols('omega1 omega2 omega3 omega4')

    # Fixed parameters
    Ixx, Iyy, Izz = 1.4e-5, 1.4e-5, 2.17e-5
    k, l, m, b, g = 3.16e-10, 0.0397, 0.027, 7.94e-12, 9.81

    # Rotation matrix R_ZYX
    Rx = sp.Matrix([[1, 0, 0],
                    [0, sp.cos(phi), -sp.sin(phi)],
                    [0, sp.sin(phi), sp.cos(phi)]])

    Ry = sp.Matrix([[sp.cos(theta), 0, sp.sin(theta)],
                    [0, 1, 0],
                    [-sp.sin(theta), 0, sp.cos(theta)]])

    Rz = sp.Matrix([[sp.cos(psi), -sp.sin(psi), 0],
                    [sp.sin(psi), sp.cos(psi), 0],
                    [0, 0, 1]])

    R = Rz * Ry * Rx

    # Control inputs as squared motor speeds
    u1 = k * (omega1**2 + omega2**2 + omega3**2 + omega4**2)
    u2 = l * k * (-omega2**2 + omega4**2)
    u3 = l * k * (-omega1**2 + omega3**2)
    u4 = b * (-omega1**2 + omega2**2 - omega3**2 + omega4**2)

    # Define the state vector and control inputs
    state = sp.Matrix([x, y, z, phi, theta, psi, sp.diff(x, t), sp.diff(y, t), sp.diff(z, t),
                       sp.diff(phi, t), sp.diff(theta, t), sp.diff(psi, t)])
    control = sp.Matrix([omega1, omega2, omega3, omega4])

    # Define the dynamics equations
    f = sp.Matrix([
        sp.diff(x, t),
        sp.diff(y, t),
        sp.diff(z, t),
        sp.diff(phi, t),
        sp.diff(theta, t),
        sp.diff(psi, t),
        -g * sp.sin(theta),
        g * sp.cos(theta) * sp.sin(phi),
        -g * sp.cos(theta) * sp.cos(phi) + u1 / m,
        (Iyy - Izz) / Ixx * sp.diff(theta, t) * sp.diff(psi, t) + u2 / Ixx,
        (Izz - Ixx) / Iyy * sp.diff(phi, t) * sp.diff(psi, t) + u3 / Iyy,
        (Ixx - Iyy) / Izz * sp.diff(phi, t) * sp.diff(theta, t) + u4 / Izz
    ])

    # Jacobians
    A = f.jacobian(state)
    B = f.jacobian(control)

    # Lambdify the functions for numerical evaluation
    state_vars = sp.Matrix([x, y, z, phi, theta, psi, sp.diff(x, t), sp.diff(y, t), sp.diff(z, t),
                            sp.diff(phi, t), sp.diff(theta, t), sp.diff(psi, t)])

    f_func = sp.lambdify((state_vars, control), f)
    A_func = sp.lambdify((state_vars, control), A)
    B_func = sp.lambdify((state_vars, control), B)

    return f_func, A_func, B_func

# Define the MPC controller using CasADi
def mpc_controller(x0, x_ref):
    # Parameters
    horizon = 5  # Prediction horizon
    dt = 1/48 # Time step

    # State and control dimensions
    nx = 12  # Number of state variables
    nu = 4   # Number of control inputs

    # Define symbolic variables for CasADi
    x = ca.MX.sym('x', nx)  # State vector
    u = ca.MX.sym('u', nu)  # Control vector
    Ixx, Iyy, Izz = 1.4e-5, 1.4e-5, 2.17e-5
    k, l, m, b, g = 3.16e-10, 0.0397, 0.027, 7.94e-12, 9.81

    # Dynamics function (converted to CasADi)
    f = ca.Function('f', [x, u], [
        ca.vertcat(
            x[6], x[7], x[8], x[9], x[10], x[11],
            -g * ca.sin(x[4]),
            g * ca.cos(x[4]) * ca.sin(x[3]),
            -g * ca.cos(x[4]) * ca.cos(x[3]) + (k * (u[0] + u[1] + u[2] + u[3])) / m,
            (Iyy - Izz) / Ixx * x[10] * x[11] + (l * k * (-u[1] + u[3])) / Ixx,
            (Izz - Ixx) / Iyy * x[9] * x[11] + (l * k * (-u[0] + u[2])) / Iyy,
            (Ixx - Iyy) / Izz * x[9] * x[10] + (b * (-u[0] + u[1] - u[2] + u[3])) / Izz
        )
    ])

    # Cost function weights
    Q = np.diag([1, 1, 1000, 1, 1, 1, 1, 1, 1000, 1, 1, 1])  # Emphasize position, attitude
    R = np.diag([1, 1, 1, 1])  # Penalize large control inputs


    # Optimization variables
    X = ca.MX.sym('X', nx, horizon+1)  # Predicted states
    U = ca.MX.sym('U', nu, horizon)    # Predicted controls

    # Objective and constraints
    cost = 0
    constraints = []
    constraints.append(X[:, 0] == x0)

    for t in range(horizon):
        cost += ca.mtimes([(X[:, t] - x_ref).T, Q, (X[:, t] - x_ref)])
        cost += ca.mtimes([U[:, t].T, R, U[:, t]])

        # Dynamics constraints
        x_next = f(X[:, t], U[:, t]) * dt + X[:, t]
        constraints.append(X[:, t+1] - x_next)

    # Terminal cost
    cost += ca.mtimes([(X[:, -1] - x_ref).T, Q, (X[:, -1] - x_ref)])

    # Flatten constraints
    constraints = ca.vertcat(*constraints)

    # Define optimization problem
    nlp = {
        'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
        'f': cost,
        'g': constraints
    }

    # Create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    lbx = np.concatenate([np.full(X.shape, -np.inf).flatten(), np.full(U.shape, 30000).flatten()])
    max_motor_speed_rad2 = (30000 * (2 * np.pi / 60))**2
    ubx = np.concatenate([np.full(X.shape, np.inf).flatten(), np.full(U.shape, np.inf).flatten()])

    lbg = np.zeros(constraints.shape)
    ubg = np.zeros(constraints.shape)
    X_guess = np.tile(x0.reshape(-1, 1), (1, horizon + 1))
    U_guess = np.zeros((nu, horizon))  # Zero initial guess for controls
    initial_guess = np.concatenate([X_guess.flatten(), U_guess.flatten()])
    
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=initial_guess)

    return sol

def generate_full_state_trajectory(waypoints, duration, sample_rate, vehicle_mass):
    # Generate a trajectory using minsnap_trajectories
    trajectory = ms.generate_trajectory(
        references=waypoints,
        degree=7,  # Minimum snap requires at least degree 7
        idx_minimized_orders=4,  # Minimize snap
        num_continuous_orders=3  # Ensure continuity of position, velocity, acceleration
    )

    t_sample = np.linspace(0, duration, int(duration * sample_rate))
    traj_samples = ms.compute_trajectory_derivatives(trajectory, t_sample, order=4)

    yaw = np.zeros_like(t_sample)  # Example: No yaw motion
    yaw_rate = np.zeros_like(t_sample)
    drag_params = ms.RotorDragParameters(
        cp=0.000001,   # drag_xy_coeff
        dh=2267.18,    # dw_coeff_1
        dv=0.16,       # dw_coeff_2
    )

    quadrotor_traj = ms.compute_quadrotor_trajectory(
        polys=trajectory,
        t_sample=t_sample,
        vehicle_mass=vehicle_mass,
        yaw=None,
        yaw_rate=None,
        drag_params=drag_params
    )

    return quadrotor_traj

def generate_full_state_trajectory_from_array(waypoints_array, total_duration, sample_rate, vehicle_mass):
    """
    Generate a full state trajectory from an np.array of waypoints.

    Parameters:
        waypoints_array (np.array): Array of shape (N, 3) where N is the number of waypoints.
        total_duration (float): Total time for the trajectory.
        sample_rate (float): Sampling rate in Hz.
        vehicle_mass (float): Mass of the quadrotor.

    Returns:
        QuadrotorTrajectory: Full state trajectory including position, velocity, acceleration, and inputs.
    """
    # Calculate time intervals between waypoints
    num_waypoints = waypoints_array.shape[0]
    time_per_segment = total_duration / (num_waypoints - 1)
    times = np.linspace(0, total_duration, num_waypoints)

    # Create Waypoint objects
    waypoints = [
        ms.Waypoint(time=t, position=waypoints_array[i].tolist()) for i, t in enumerate(times)
    ]

    # Generate trajectory
    trajectory = generate_full_state_trajectory(
        waypoints=waypoints,
        duration=total_duration,
        sample_rate=sample_rate,
        vehicle_mass=vehicle_mass
    )

    return trajectory


# Example usage
x0 = np.zeros(12)
x_ref = np.zeros(12)
solution = mpc_controller(x0, x_ref)
print("MPC Solution:", solution)

def mpc_to_motor_rpm(omega_squared):
    """
    Convert MPC outputs (omega squared) to motor RPMs using PID controller methods.

    Parameters:
    - omega_squared: (4, ) array of motor angular velocities squared (rad^2/s^2).
    - KF: Thrust coefficient.
    - KM: Drag coefficient.
    - MIXER_MATRIX: Mixer matrix for Crazyflie.
    - PWM2RPM_SCALE: Scaling factor from PWM to RPM.
    - PWM2RPM_CONST: Offset constant for PWM to RPM.
    - l: Arm length of the drone.

    Returns:
    - rpm: (4, ) array of motor RPMs.
    """
    KF= 3.16e-10
    KM =  7.94e-12
    l = 0.0397
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    MIN_PWM = 20000
    MAX_PWM = 65535
    MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
    # Compute thrust and torques
    u1 = KF * np.sum(omega_squared)
    u2 = l * KF * (-omega_squared[1] + omega_squared[3])
    u3 = l * KF * (-omega_squared[0] + omega_squared[2])
    u4 = KM * (-omega_squared[0] + omega_squared[1] - omega_squared[2] + omega_squared[3])
    
    # Combine thrust and torques into a vector
    thrust_and_torques = np.array([u1, u2, u3, u4])

    # Use mixer matrix to compute PWM
    pwm = thrust_and_torques[0] + np.dot(MIXER_MATRIX, thrust_and_torques[1:])
    pwm = np.clip(pwm, MIN_PWM, MAX_PWM)  # Ensure PWM is within operational bounds

    # Convert PWM to RPM
    rpm = PWM2RPM_SCALE * pwm + PWM2RPM_CONST
    return rpm

