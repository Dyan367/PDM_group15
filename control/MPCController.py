import numpy as np
import cvxpy as cp

def discretize_tustin(A,B,dt):
        """Discretize the state-space matrices using the Tustin method."""
        dt = 1/48
        n = A.shape[0]
        I = np.eye(n)
        Ad = np.linalg.inv(I - 0.5 * A * dt) @ (I + 0.5 * A * dt)
        Bd = np.linalg.inv(I - 0.5 * A * dt) @ (B * dt)
        return Ad, Bd

class Simple_MPC():
    def __init__(self,Q, R):
        ## Simple state space
        ## x = [x,y,z,vx,vy,vz]
        ## u = [ax,ay,az]

        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.Q = Q
        self.R = R
        self.Ad, self.Bd = discretize_tustin(self.A, self.B, 0.01)

    def compute_mpc_control(self, x0, x_ref, N):
        """
        Compute the desired accelerations using MPC.

        Args:
            x0: Initial state vector [x, y, z, vx, vy, vz].
            x_ref: Reference state vector [x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref].
            N: Prediction horizon.

        Returns:
            u_opt: Optimal control input [ax, ay, az] for the first timestep.
        """

        # Dimensions
        n_states = self.Ad.shape[0]
        n_inputs = self.Bd.shape[1]

        # Optimization variables
        x = cp.Variable((n_states, N + 1))
        u = cp.Variable((n_inputs, N))

        # Objective function
        cost = 0
        constraints = []

        # Initial state constraint
        constraints.append(x[:, 0] == x0)

        for k in range(N):
            # Cost function (tracking + control effort)
            cost += cp.quad_form(x[:, k] - x_ref, self.Q) + cp.quad_form(u[:, k], self.R)

            # Dynamics constraint
            constraints.append(x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k])

            constraints.append(u[:, k] >= -0.5)  # Minimum acceleration
            constraints.append(u[:, k] <= 0.5)  # Maximum acceleration 
            constraints.append(x[3:, k] >= [-1.0, -1.0, -1.0])  # Minimum velocity 
            constraints.append(x[3:, k] <= [1.0, 1.0, 1.0])   # Maximum velocity 
            constraints.append(x[:3, k] >= [-5.0, -5.0, 0.0])  # Minimum position 
            constraints.append(x[:3, k] <= [5.0,5.0,3.0])   # Maximum position 

        # Terminal cost
        cost += cp.quad_form(x[:, N] - x_ref, self.Q)

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError("MPC optimization failed.")

        # Return the first control input
        u_opt = u[:, 0].value
        return u_opt
    


