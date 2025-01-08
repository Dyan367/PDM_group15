import numpy as np
import casadi as ca

class NonlinearMPC:
    def __init__(self, dt, horizon, mass=0.027, g=9.81):
        self.dt = dt
        self.horizon = horizon
        self.mass = mass
        self.g = g

        self.nx = 12  # State: position(3), velocity(3), roll/pitch/yaw(3), angular velocity(3)
        self.nu = 4   # Input: thrust, torques (3)

        # Define optimization problem
        self.opti = ca.Opti()
        self._build_mpc()

    def _build_mpc(self):
        # Decision variables
        X = self.opti.variable(self.nx, self.horizon + 1)  # States
        U = self.opti.variable(self.nu, self.horizon)      # Inputs
        X_ref = self.opti.parameter(self.nx, self.horizon + 1)  # Reference trajectory
        x0 = self.opti.parameter(self.nx)  # Initial state

        # Cost weights
        Q = np.diag([100, 100, 100, 10, 10, 10, 1, 1, 1, 0.1, 0.1, 0.1])
        R = np.diag([1, 5, 5, 5])

        # Cost and constraints
        cost = 0
        for k in range(self.horizon):
            cost += ca.mtimes((X[:, k] - X_ref[:, k]).T, Q @ (X[:, k] - X_ref[:, k]))
            cost += ca.mtimes(U[:, k].T, R @ U[:, k])
            x_next = self._quadrotor_dynamics(X[:, k], U[:, k])
            self.opti.subject_to(X[:, k + 1] == x_next)

        # Initial state constraint
        self.opti.subject_to(X[:, 0] == x0)

        # Thrust constraint
        self.opti.subject_to(U[0, :] >= 0.0)

        self.opti.minimize(cost)
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

        self.X, self.U, self.X_ref, self.x0 = X, U, X_ref, x0

    def _quadrotor_dynamics(self, x, u):
        # Access components of x and u using indexing
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        roll, pitch, yaw = x[6], x[7], x[8]
        wx, wy, wz = x[9], x[10], x[11]
        T, tau_x, tau_y, tau_z = u[0], u[1], u[2], u[3]

        g, m = self.g, self.mass

        # Rotation matrix from roll, pitch, yaw
        R = ca.vertcat(
            ca.horzcat(ca.cos(yaw) * ca.cos(pitch), ca.sin(yaw) * ca.cos(pitch), -ca.sin(pitch)),
            ca.horzcat(ca.cos(yaw) * ca.sin(pitch) * ca.sin(roll) - ca.sin(yaw) * ca.cos(roll),
                       ca.sin(yaw) * ca.sin(pitch) * ca.sin(roll) + ca.cos(yaw) * ca.cos(roll),
                       ca.cos(pitch) * ca.sin(roll)),
            ca.horzcat(ca.cos(yaw) * ca.sin(pitch) * ca.cos(roll) + ca.sin(yaw) * ca.sin(roll),
                       ca.sin(yaw) * ca.sin(pitch) * ca.cos(roll) - ca.cos(yaw) * ca.sin(roll),
                       ca.cos(pitch) * ca.cos(roll))
        )

        # Acceleration and angular velocity derivatives
        acc = R @ ca.vertcat(0, 0, T / m) - ca.vertcat(0, 0, g)
        omega_dot = ca.vertcat(tau_x, tau_y, tau_z)

        # State update equations
        x_next = ca.vertcat(
            px + vx * self.dt,
            py + vy * self.dt,
            pz + vz * self.dt,
            vx + acc[0] * self.dt,
            vy + acc[1] * self.dt,
            vz + acc[2] * self.dt,
            roll + wx * self.dt,
            pitch + wy * self.dt,
            yaw + wz * self.dt,
            wx + omega_dot[0] * self.dt,
            wy + omega_dot[1] * self.dt,
            wz + omega_dot[2] * self.dt
        )
        return x_next

    def solve(self, x0, x_ref):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.X_ref, x_ref)
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])