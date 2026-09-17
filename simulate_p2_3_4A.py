import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from rcam_model import xdot

# SELECT SIMULATION

# 2 -> Base simulation
# 3 -> Aileron pulse
# 4 -> Engine shutdown
SIMULATION_TO_RUN = 4

# Only used for simulation 4
FAILED_ENGINE = 2
# 1 -> shutdown throttle 1
# 2 -> shutdown throttle 2


TF = 180.0       # total simulation time [s]
DT_OUT = 1.0     # output/plot time step [s]
DT_INT = 0.05    # internal integration time step [s]


X0 = np.array([
    85.0,   # u [m/s]
    0.0,    # v [m/s]
    0.0,    # w [m/s]
    0.0,    # p [rad/s]
    0.0,    # q [rad/s]
    0.0,    # r [rad/s]
    0.0,    # phi [rad]
    0.1,    # theta [rad]
    0.0     # psi [rad]
], dtype=float)


# ================================================================
# BASE CONTROL
# U = [aileron, stabilizer, rudder, throttle1, throttle2]
# ================================================================

U0 = np.array([
    0.0,    # aileron [rad]
   -0.1,    # stabilizer/elevator [rad]
    0.0,    # rudder [rad]
    0.08,   # throttle 1
    0.08    # throttle 2
], dtype=float)


# ================================================================
# CONTROL INPUTS
# ================================================================

def control_input(t: float, simulation: int) -> np.ndarray:

    U = U0.copy()

    # ============================================================
    # SIMULATION 3 - AILERON PULSE
    # ============================================================

    if simulation == 3:

        if 30.0 <= t < 32.0:
            U[0] = np.deg2rad(5.0)

    # ============================================================
    # SIMULATION 4 - ENGINE FAILURE
    # ============================================================

    elif simulation == 4:

        if FAILED_ENGINE == 1:

            U[3] = 0.0      # throttle 1 OFF
            U[4] = 0.08     # throttle 2 ON

        elif FAILED_ENGINE == 2:

            U[3] = 0.08     # throttle 1 ON
            U[4] = 0.0      # throttle 2 OFF

        else:
            raise ValueError("FAILED_ENGINE must be 1 or 2")

    return U


# ================================================================
# BODY TO NED ROTATION MATRIX
# ================================================================

def dcm_body_to_ned(phi: float, theta: float, psi: float) -> np.ndarray:

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    cth = np.cos(theta)
    sth = np.sin(theta)

    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    C_nb = np.array([
        [cth * cpsi,
         sphi * sth * cpsi - cphi * spsi,
         cphi * sth * cpsi + sphi * spsi],

        [cth * spsi,
         sphi * sth * spsi + cphi * cpsi,
         cphi * sth * spsi - sphi * cpsi],

        [-sth,
         sphi * cth,
         cphi * cth]
    ])

    return C_nb


# ================================================================
# RK4 INTEGRATION
# ================================================================

def rk4_step(X: np.ndarray, U: np.ndarray, h: float) -> np.ndarray:

    k1 = xdot(X, U)

    k2 = xdot(
        X + 0.5 * h * k1,
        U
    )

    k3 = xdot(
        X + 0.5 * h * k2,
        U
    )

    k4 = xdot(
        X + h * k3,
        U
    )

    X_next = X + (h / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )

    return X_next


# ================================================================
# MAIN SIMULATION FUNCTION
# ================================================================

def simulate_rcam(
    simulation: int,
    X_initial: np.ndarray,
    tf: float = 180.0,
    dt_out: float = 1.0,
    dt_int: float = 0.05
):

    if simulation not in [2, 3, 4]:
        raise ValueError("simulation must be 2, 3 or 4")

    n_out = int(tf / dt_out) + 1

    t_hist = np.linspace(0.0, tf, n_out)

    X_hist = np.zeros((n_out, 9))
    P_hist = np.zeros((n_out, 3))
    U_hist = np.zeros((n_out, 5))

    X = X_initial.copy()

    P = np.array([
        0.0,   # North
        0.0,   # East
        0.0    # Down
    ], dtype=float)

    X_hist[0] = X
    P_hist[0] = P
    U_hist[0] = control_input(0.0, simulation)

    current_time = 0.0

    for k in range(n_out - 1):

        t_next = t_hist[k + 1]

        while current_time < t_next - 1e-12:

            h = min(dt_int, t_next - current_time)

            U_now = control_input(current_time, simulation)

            # ====================================================
            # POSITION UPDATE
            # ====================================================

            phi = X[6]
            theta = X[7]
            psi = X[8]

            V_body = X[0:3]

            C_nb = dcm_body_to_ned(phi, theta, psi)

            P_dot = C_nb @ V_body

            # ====================================================
            # STATE UPDATE
            # ====================================================

            X_new = rk4_step(X, U_now, h)

            # ====================================================
            # POSITION INTEGRATION
            # ====================================================

            P_new = P + h * P_dot

            X = X_new
            P = P_new

            current_time += h

        X_hist[k + 1] = X
        P_hist[k + 1] = P

        U_hist[k + 1] = control_input(
            t_hist[k + 1],
            simulation
        )

    return t_hist, X_hist, P_hist, U_hist


# ================================================================
# PLOT STATES
# ================================================================

def plot_states(
    t_hist: np.ndarray,
    X_hist: np.ndarray,
    simulation: int
):

    state_names = [
        "u [m/s]",
        "v [m/s]",
        "w [m/s]",
        "p [rad/s]",
        "q [rad/s]",
        "r [rad/s]",
        "phi [rad]",
        "theta [rad]",
        "psi [rad]"
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14, 10)
    )

    axes = axes.flatten()

    for i in range(9):

        axes[i].plot(
            t_hist,
            X_hist[:, i],
            linewidth=1.8
        )

        axes[i].set_title(state_names[i])

        axes[i].set_xlabel("Time [s]")

        axes[i].set_ylabel(state_names[i])

        axes[i].grid(True)

    fig.suptitle(
        f"RCAM State Variables vs Time - Simulation {simulation}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ================================================================
# PLOT CONTROLS
# ================================================================

def plot_controls(
    t_hist: np.ndarray,
    U_hist: np.ndarray,
    simulation: int
):

    control_names = [
        "aileron [rad]",
        "stabilizer/elevator [rad]",
        "rudder [rad]",
        "throttle 1",
        "throttle 2"
    ]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10, 10),
        sharex=True
    )

    for i in range(5):

        axes[i].plot(
            t_hist,
            U_hist[:, i],
            linewidth=1.8
        )

        axes[i].set_ylabel(control_names[i])

        axes[i].grid(True)

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        f"Control Inputs vs Time - Simulation {simulation}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ================================================================
# PLOT 2D TRAJECTORY
# ================================================================

def plot_trajectory_2d(
    P_hist: np.ndarray,
    simulation: int
):

    N = P_hist[:, 0]
    E = P_hist[:, 1]

    plt.figure(figsize=(8, 6))

    plt.plot(
        E,
        N,
        linewidth=2.0
    )

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")

    plt.title(
        f"RCAM Horizontal Trajectory - Simulation {simulation}"
    )

    plt.grid(True)

    plt.axis("equal")

    plt.tight_layout()

    plt.show()


# ================================================================
# PLOT 3D TRAJECTORY
# ================================================================

def plot_trajectory_3d(
    P_hist: np.ndarray,
    simulation: int
):

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    N = P_hist[:, 0]
    E = P_hist[:, 1]
    D = P_hist[:, 2]

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot(
        E,
        N,
        -D,
        linewidth=2.0
    )

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")

    ax.set_title(
        f"RCAM 3D Trajectory - Simulation {simulation}"
    )

    ax.grid(True)

    plt.tight_layout()

    plt.show()


# ================================================================
# GIF FUNCTION
# ================================================================

def create_trajectory_gif(
    P_hist: np.ndarray,
    X_hist: np.ndarray,
    simulation: int,
    filename: str = None,
    step: int = 2,
    fps: int = 10
):

    if filename is None:
        filename = f"rcam_simulation_{simulation}.gif"

    N = P_hist[:, 0]
    E = P_hist[:, 1]

    psi = X_hist[:, 8]

    e_min, e_max = np.min(E), np.max(E)
    n_min, n_max = np.min(N), np.max(N)

    e_margin = max(
        50.0,
        0.05 * max(1.0, e_max - e_min)
    )

    n_margin = max(
        50.0,
        0.05 * max(1.0, n_max - n_min)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlim(
        e_min - e_margin,
        e_max + e_margin
    )

    ax.set_ylim(
        n_min - n_margin,
        n_max + n_margin
    )

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")

    ax.set_title(
        f"RCAM Trajectory Animation - Simulation {simulation}"
    )

    ax.grid(True)

    ax.axis("equal")

    line, = ax.plot([], [], linewidth=2.0)

    point, = ax.plot(
        [],
        [],
        marker="o",
        markersize=6
    )

    heading_line, = ax.plot([], [], linewidth=2.0)

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes
    )

    frame_indices = list(
        range(0, len(P_hist), step)
    )

    if frame_indices[-1] != len(P_hist) - 1:
        frame_indices.append(len(P_hist) - 1)

    def init():

        line.set_data([], [])
        point.set_data([], [])

        heading_line.set_data([], [])

        time_text.set_text("")

        return (
            line,
            point,
            heading_line,
            time_text
        )

    def update(frame_number):

        i = frame_indices[frame_number]

        line.set_data(
            E[:i + 1],
            N[:i + 1]
        )

        point.set_data(
            [E[i]],
            [N[i]]
        )

        arrow_length = max(
            100.0,
            0.04 * max(
                e_max - e_min,
                n_max - n_min,
                1.0
            )
        )

        e_head = E[i] + arrow_length * np.sin(psi[i])

        n_head = N[i] + arrow_length * np.cos(psi[i])

        heading_line.set_data(
            [E[i], e_head],
            [N[i], n_head]
        )

        time_text.set_text(
            f"t = {i:.0f} s"
        )

        return (
            line,
            point,
            heading_line,
            time_text
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        blit=True,
        interval=1000 / fps
    )

    writer = PillowWriter(fps=fps)

    animation.save(
        filename,
        writer=writer
    )

    plt.close(fig)

    return filename


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    if SIMULATION_TO_RUN not in [2, 3, 4]:
        raise ValueError(
            "SIMULATION_TO_RUN must be 2, 3 or 4"
        )

    print(
        f"Running RCAM Simulation {SIMULATION_TO_RUN}..."
    )

    if SIMULATION_TO_RUN == 4:

        print(
            f"Engine failure selected: Engine {FAILED_ENGINE}"
        )

    t_hist, X_hist, P_hist, U_hist = simulate_rcam(
        simulation=SIMULATION_TO_RUN,
        X_initial=X0,
        tf=TF,
        dt_out=DT_OUT,
        dt_int=DT_INT
    )

    # ============================================================
    # PLOTS
    # ============================================================

    plot_states(
        t_hist,
        X_hist,
        SIMULATION_TO_RUN
    )

    plot_controls(
        t_hist,
        U_hist,
        SIMULATION_TO_RUN
    )

    plot_trajectory_2d(
        P_hist,
        SIMULATION_TO_RUN
    )

    plot_trajectory_3d(
        P_hist,
        SIMULATION_TO_RUN
    )

    # ============================================================
    # GIF
    # ============================================================

    gif_name = create_trajectory_gif(
        P_hist=P_hist,
        X_hist=X_hist,
        simulation=SIMULATION_TO_RUN,
        filename=f"rcam_simulation_{SIMULATION_TO_RUN}.gif",
        step=2,
        fps=10
    )

    print(
        f"Finished Simulation {SIMULATION_TO_RUN}."
    )

    print(
        f"GIF saved as: {os.path.abspath(gif_name)}"
    )