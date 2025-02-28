import math
import typing as T

import numpy as np
from numpy import linalg
from scipy.integrate import cumtrapz  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils import save_dict, maybe_makedirs

class State:
    def __init__(self, x: float, y: float, V: float, th: float) -> None:
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self) -> float:
        return self.V*np.cos(self.th)

    @property
    def yd(self) -> float:
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state: State, final_state: State, tf: float) -> np.ndarray:
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########
    x_i, y_i, V_i, th_i, xd_i, yd_i = initial_state.x, initial_state.y, initial_state.V, initial_state.th, initial_state.xd, initial_state.yd
    x_f, y_f, V_f, th_f, xd_f, yd_f = final_state.x, final_state.y, final_state.V, final_state.th, final_state.xd, final_state.yd
    A = np.array([[1, 0, 0, 0, 0,0,0,0],
                  [0, 1, 0, 0, 0, 0, 0,0],
                  [1, tf, tf**2, tf**3, 0,0,0,0],
                  [0, 1, 2*tf, 3*tf**2, 0, 0, 0,0],
                  [0,0,0,0,1,0, 0, 0],
                  [0,0,0,0,0,1, 0, 0],
                  [0,0,0,0,1,tf, tf**2, tf**3],
                  [0,0,0,0,0,1, 2*tf, 3*tf**2]])
    b = np.array([x_i, xd_i, x_f, xd_f, y_i, yd_i, y_f, yd_f])
    b = np.reshape(b,(8,))
    coeffs = np.linalg.solve(A, b)
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs: np.ndarray, tf: float, N: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        t (np.array shape [N]) evenly spaced time points from 0 to tf
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0, tf, N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N, 7))
    ########## Code starts here ##########
    x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4 = coeffs
    for idx, timestep in enumerate(t):
        x = x_1 + x_2*timestep + x_3*timestep**2 + x_4*timestep**3
        y = y_1 + y_2*timestep + y_3*timestep**2 + y_4*timestep**3
        x_dot = x_2 + 2*x_3*timestep + 3*x_4*timestep**2
        y_dot = y_2 + 2*y_3*timestep + 3*y_4*timestep**2
        th = np.arctan2(y_dot,x_dot)
        x_ddot = 2*x_3 + 6*x_4*timestep
        y_ddot = 2*y_3 + 6*y_4*timestep
        traj[idx] = [x, y, th, x_dot, y_dot, x_ddot, y_ddot]

    ########## Code ends here ##########

    return t, traj

def compute_controls(traj: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    N = traj.shape[0]
    V = np.zeros((N))
    om =  np.zeros((N))
    for t in range(N):
        x, y, th, x_dot, y_dot, x_ddot, y_ddot = traj[t]

        J = np.array([[np.cos(th), -y_dot],[np.sin(th), x_dot]])
        J_inv = np.linalg.inv(J)
        b = np.array([x_ddot, y_ddot])

        a_curr, om_curr = J_inv @ b
        V[t] = np.linalg.det(J)
        om[t] = om_curr

    ########## Code ends here ##########

    return V, om

def compute_arc_length(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    This function computes arc-length s as a function of t.
    Inputs:
        V: a vector of velocities of length T
        t: a vector of time of length T
    Output:
        s: the arc-length as a function of time. s[i] is the arc-length at time
            t[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    s = None
    ########## Code starts here ##########
    s = cumtrapz(V, x= t, initial = 0)
    ########## Code ends here ##########
    return s

def rescale_V(V: np.ndarray, om: np.ndarray, V_max: float, om_max: float) -> np.ndarray:
    """
    This function computes V_tilde, given the unconstrained solution V, and om.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained,
            differential flatness problem.
        om: vector of angular velocities of length T. Solution from the
            unconstrained, differential flatness problem.
        V_max: maximum absolute linear velocity
        om_max: maximum absolute angular velocity
    Output:
        V_tilde: Rescaled velocity that satisfies the control constraints.

    Hint: At each timestep V_tilde should be computed as a minimum of the
    original value V, and values required to ensure _both_ constraints are
    satisfied.
    Hint: This should only take one or two lines.
    Hint: If you run into division-by-zero runtime warnings, try adding a small
          epsilon (e.g. 1e-6) to the denomenator
    """
    ########## Code starts here ##########
    V_tilde = np.clip(V, -V_max, V_max)
    V_tilde = np.where(om > om_max, V_tilde*(om_max/(om+1e-6)), V_tilde)
    V_tilde = np.where(om < -om_max, V_tilde*(om_max/(-om+1e-6)), V_tilde)
    ########## Code ends here ##########
    return V_tilde


def compute_tau(V_tilde: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    This function computes the new time history tau as a function of s.
    Inputs:
        V_tilde: a sequence of scaled velocities of length T.
        s: a sequence of arc-length of length T.
    Output:
        tau: the new time history for the sequence. tau[i] is the time at s[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    ########## Code starts here ##########
    tau = cumtrapz(1.0/V_tilde, s, initial = 0)
    ########## Code ends here ##########
    return tau

def rescale_om(V: np.ndarray, om: np.ndarray, V_tilde: np.ndarray) -> np.ndarray:
    """
    This function computes the rescaled om control.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
        V_tilde: vector of scaled velocities of length T.
    Output:
        om_tilde: vector of scaled angular velocities

    Hint: This should take one line.
    """
    ########## Code starts here ##########
    om_tilde = V_tilde*(om/V)
    ########## Code ends here ##########
    return om_tilde

def compute_traj_with_limits(
    z_0: State,
    z_f: State,
    tf: float,
    N: int,
    V_max: float,
    om_max: float
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coeffs = compute_traj_coeffs(initial_state=z_0, final_state=z_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    return traj, tau, V_tilde, om_tilde

def interpolate_traj(
    traj: np.ndarray,
    tau: np.ndarray,
    V_tilde: np.ndarray,
    om_tilde: np.ndarray,
    dt: float,
    s_f: State
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation
        s_f (State) final state

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new/dt)
    t_new = dt*np.array(range(N_new+1))

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new+1,7))
    traj_scaled[:,0] = np.interp(t_new,tau,traj[:,0])   # x
    traj_scaled[:,1] = np.interp(t_new,tau,traj[:,1])   # y
    traj_scaled[:,2] = np.interp(t_new,tau,traj[:,2])   # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)           # V
    om_scaled = np.interp(t_new, tau, om_tilde)         # om
    # Compute xy velocities
    traj_scaled[:,3] = V_scaled*np.cos(traj_scaled[:,2])    # xd
    traj_scaled[:,4] = V_scaled*np.sin(traj_scaled[:,2])    # yd
    # Compute xy acclerations
    traj_scaled[:,5] = np.append(np.diff(traj_scaled[:,3])/dt,-s_f.V*om_scaled[-1]*np.sin(s_f.th)) # xdd
    traj_scaled[:,6] = np.append(np.diff(traj_scaled[:,4])/dt, s_f.V*om_scaled[-1]*np.cos(s_f.th)) # ydd

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    # Constants
    tf = 15.
    V_max = 0.5
    om_max = 1

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=V_max, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=V_max, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)

    part_b_complete = False
    s = compute_arc_length(V, t)
    if s is not None:
        part_b_complete = True
        V_tilde = rescale_V(V, om, V_max, om_max)
        tau = compute_tau(V_tilde, s)
        om_tilde = rescale_om(V, om, V_tilde)

        t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

        # Save trajectory data
        data = {'z': traj_scaled, 'V': V_scaled, 'om': om_scaled}
        save_dict(data, "data/differential_flatness.pkl")

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(2, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 4, sharex=ax)
    if part_b_complete:
        plt.plot(t_new, V_scaled, linewidth=2)
        plt.plot(t_new, om_scaled, linewidth=2)
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
        plt.grid(True)
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('Time [s]')
    plt.title('Scaled Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    if part_b_complete:
        h, = plt.plot(t, s, 'b-', linewidth=2)
        handles = [h]
        labels = ["Original"]
        h, = plt.plot(tau, s, 'r-', linewidth=2)
        handles.append(h)
        labels.append("Scaled")
        plt.legend(handles, labels, loc="best")
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Arc-length [m]')
    plt.title('Original and scaled arc-length')
    plt.tight_layout()
    plt.savefig("plots/differential_flatness.png")
    plt.show()
