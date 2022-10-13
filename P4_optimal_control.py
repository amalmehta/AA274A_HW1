import typing as T

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from scipy.optimize import minimize, Bounds  # type: ignore
from utils import save_dict, maybe_makedirs

N = 20  # Number of time discretization nodes (0, 1, ... N).
s_dim = 3  # State dimension; 3 for (x, y, th).
u_dim = 2  # Control dimension; 2 for (V, om).
v_max = 0.5  # Maximum linear velocity.
om_max = 1.0  # Maximum angular velocity.

s_0 = np.array([0, 0, -np.pi / 2])  # Initial state.
s_f = np.array([5, 5, -np.pi / 2])  # Final state.


def pack_decision_variables(t_f: float, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Packs decision variables (final time, states, controls) into a 1D vector.

    Args:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).

    Returns:
        An array `z` of shape (1 + (N + 1) * s_dim + N * u_dim,).
    """
    return np.concatenate([[t_f], s.ravel(), u.ravel()])


def unpack_decision_variables(z: np.ndarray) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Unpacks a 1D vector into decision variables (final time, states, controls).

    Args:
        z: An array of shape (1 + (N + 1) * s_dim + N * u_dim,).

    Returns:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).
    """
    t_f = float(z[0])
    s = z[1:1 + (N + 1) * s_dim].reshape(N + 1, s_dim)
    u = z[-N * u_dim:].reshape(N, u_dim)
    return t_f, s, u


def optimize_trajectory(
    time_weight: float = 1.0,
    verbose: bool = True
) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Computes the optimal trajectory as a function of `time_weight`.

    Args:
        time_weight: \lambda in the HW writeup.

    Returns:
        t_f_opt: Optimal final time, a scalar.
        s_opt: Optimal states, an array of shape (N + 1, s_dim).
        u_opt: Optimal controls, an array of shape (N, u_dim).
    """

    # NOTE: When using `minimize`, you may find the utilities
    # `pack_decision_variables` and `unpack_decision_variables` useful.

    # WRITE YOUR CODE BELOW ###################################################
    z_shape = 1 + (N + 1) * s_dim + N * u_dim
    def get_bounds():
        bounds = [0]*z_shape
        bounds[0] = (0, None)
        bounds[1:1+(N+1)*s_dim] = [(None, None)]*int((N+1)*s_dim)
        bounds[1+(N+1)*s_dim:1+(N+1)*s_dim + (N)*u_dim:2] = [(-v_max, v_max)]*int((N)*u_dim/2) # clean this up with om_max later
        bounds[2+(N+1)*s_dim:1+(N+1)*s_dim + (N)*u_dim:2] = [(-om_max, om_max)]*int((N)*u_dim/2)
        return bounds

    #constraints
    def x0_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        return x[0]
    def xf_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        return x[-1]-5
    def y0_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        return y[0]
    def yf_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        return y[-1]-5
    def th0_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        return th[0]+np.pi/2
    def thf_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        return th[-1]+np.pi/2
    def x_dyn_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        t_interval = t_f/N
        V = u[:, 0]
        om = u[:, 1]
        total = 0 
        for i in range(1,N):
            xdot_i = V[i-1]*np.cos(th[i-1])
            total += x[i]-x[i-1] -t_interval*xdot_i
        return total




    def y_dyn_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        t_interval = t_f/N
        V = u[:, 0]
        om = u[:, 1]
        total = 0 
        for i in range(1,N):
            ydot_i = V[i-1]*np.sin(th[i-1])
            total += y[i]-y[i-1] -t_interval*ydot_i
        return total

    def th_dyn_constraint(z):
        t_f, s, u = unpack_decision_variables(z)
        x, y, th = s[:, 0], s[:, 1], s[:,2]
        t_interval = t_f/N
        V = u[:, 0]
        om = u[:, 1]
        total = 0
        for i in range(1,N):
            total += th[i]-th[i-1]-t_interval*om[i-1]
        return total


    



    def cost(z):
        t_f, s, u = unpack_decision_variables(z)
        t_interval = t_f/N
        J = 0

        for idx in range(N):
            u_curr = u[idx]
            J += t_interval*(time_weight + u_curr[0]**2 + u_curr[1]**2)
        return J
    init_guess = np.zeros((z_shape,))
    init_guess[0] = 10
    bds = get_bounds()
    cons = ({'type':'eq', 'fun': x0_constraint},
                {'type':'eq', 'fun': xf_constraint},
                {'type':'eq', 'fun': y0_constraint},
                {'type':'eq', 'fun': yf_constraint},
                {'type':'eq', 'fun': th0_constraint},
                {'type':'eq', 'fun': thf_constraint},
                {'type':'eq', 'fun': x_dyn_constraint},
                {'type':'eq', 'fun': y_dyn_constraint},
                {'type':'eq', 'fun': th_dyn_constraint},
                )
    res = minimize(cost, x0=init_guess, bounds =bds, constraints=cons)
    t_f, s, u = unpack_decision_variables(res.x)
    print(s)
    return t_f, s, u
    #lambda t_f, s, u: time_weight + 
    #raise NotImplementedError
    ###########################################################################


if __name__ == '__main__':
    for time_weight in (1.0, 0.2):
        t_f, s, u = optimize_trajectory(time_weight)
        V = u[:, 0]
        om = u[:, 1]
        t = np.linspace(0, t_f, N + 1)[:-1]
        x = s[:, 0]
        y = s[:, 1]
        th = s[:, 2]
        data = {'t_f': t_f, 's': s, 'u': u}
        save_dict(data, f'data/optimal_control_{time_weight}.pkl')
        maybe_makedirs('plots')

        # plotting
        # plt.rc('font', weight='bold', size=16)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'k-', linewidth=2)
        plt.quiver(x, y, np.cos(th), np.sin(th))
        plt.grid(True)
        plt.plot(0, 0, 'go', markerfacecolor='green', markersize=15)
        plt.plot(5, 5, 'ro', markerfacecolor='red', markersize=15)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis([-1, 6, -1, 6])
        plt.title(f'Optimal Control Trajectory (lambda = {time_weight})')

        plt.subplot(1, 2, 2)
        plt.plot(t, V, linewidth=2)
        plt.plot(t, om, linewidth=2)
        plt.grid(True)
        plt.xlabel('Time [s]')
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
        plt.title(f'Optimal control sequence (lambda = {time_weight})')
        plt.tight_layout()
        plt.savefig(f'plots/optimal_control_{time_weight}.png')
        plt.show()
