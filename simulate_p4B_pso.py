# ================================================================
# RCAM - PART 4B
# PSO TRIM OPTIMIZATION WITH ENGINE FAILURE
#
# This script performs:
#
# 1. Particle Swarm Optimization (PSO)
# 2. One engine remains permanently OFF
# 3. Search for trimmed flight near 78 m/s
# 4. Visualization of optimization process
# 5. Visualization of aircraft response
#
# ================================================================

# ================================================================
# IMPORT LIBRARIES
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

# 3D plotting tools
from mpl_toolkits.mplot3d import Axes3D

# Import RCAM dynamics model
from rcam_model import xdot


# ================================================================
# USER SETTINGS
# ================================================================

# Which engine fails?
FAILED_ENGINE = 1

# 1 -> throttle 1 OFF
# 2 -> throttle 2 OFF

# Simulation parameters
TF = 180.0
DT = 0.05
DT_OUT = 1.0

# Desired trim speed
TARGET_SPEED = 78.0


# ================================================================
# INITIAL GUESS
# ================================================================

# State vector:
# X = [u, v, w, p, q, r, phi, theta, psi]

X0 = np.array([

    85.0,                  # u -> forward speed
    0.0,                   # v -> lateral speed
    0.0,                   # w -> vertical speed

    0.0,                   # p -> roll rate
    0.0,                   # q -> pitch rate
    0.0,                   # r -> yaw rate

    0.0,                   # phi -> roll angle
    0.1,                   # theta -> pitch angle
    np.deg2rad(45.0)       # psi -> heading

])

# Control vector:
# U = [aileron, elevator, rudder, throttle1, throttle2]

U0 = np.array([

    0.0,       # aileron
    -0.1,      # elevator
    0.0,       # rudder

    0.08,      # throttle 1
    0.08       # throttle 2

])


# ================================================================
# RK4 INTEGRATOR
# ================================================================
# Integrates aircraft states using Runge-Kutta 4th order
# ================================================================

def rk4_step(X, U, h):

    # Slope at beginning
    k1 = xdot(X, U)

    # Slope at midpoint using k1
    k2 = xdot(
        X + 0.5*h*k1,
        U
    )

    # Another midpoint slope using k2
    k3 = xdot(
        X + 0.5*h*k2,
        U
    )

    # Slope at end
    k4 = xdot(
        X + h*k3,
        U
    )

    # Weighted RK4 integration
    return X + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ================================================================
# BUILD STATE + CONTROL FROM PARTICLE
# ================================================================
# Each particle represents:
#
# [u, v, w, p, q, r, phi, theta,
#  aileron, elevator, rudder,
#  alive_engine_throttle]
#
# ================================================================

def build_state_control(particle):

    # ============================================================
    # EXTRACT STATES
    # ============================================================

    u = particle[0]
    v = particle[1]
    w = particle[2]

    p = particle[3]
    q = particle[4]
    r = particle[5]

    phi = particle[6]
    theta = particle[7]

    # Fixed heading to northeast
    psi = np.deg2rad(45.0)

    # ============================================================
    # EXTRACT CONTROLS
    # ============================================================

    da = particle[8]
    de = particle[9]
    dr = particle[10]

    # Throttle of surviving engine
    throttle_alive = particle[11]

    # ============================================================
    # ENGINE FAILURE LOGIC
    # ============================================================

    # One engine is forced to remain OFF permanently

    if FAILED_ENGINE == 1:

        T1 = 0.0
        T2 = throttle_alive

    else:

        T1 = throttle_alive
        T2 = 0.0

    # ============================================================
    # BUILD STATE VECTOR
    # ============================================================

    X = np.array([

        u, v, w,
        p, q, r,
        phi, theta, psi

    ])

    # ============================================================
    # BUILD CONTROL VECTOR
    # ============================================================

    U = np.array([

        da,
        de,
        dr,

        T1,
        T2

    ])

    return X, U


# ================================================================
# COST FUNCTION
# ================================================================
# Goal:
#
# 1. Minimize state derivatives -> steady flight
# 2. Reach target speed of 78 m/s
# 3. Minimize rotational rates
# 4. Maintain straight-and-level flight
#
# ================================================================

def cost_function(particles):

    costs = []

    # Evaluate every particle
    for particle in particles:

        # Build aircraft state and controls
        X, U = build_state_control(particle)

        # Aircraft derivatives
        Xdot_val = xdot(X, U)

        # ========================================================
        # EXTRACT STATES
        # ========================================================

        u = X[0]
        v = X[1]
        w = X[2]

        p = X[3]
        q = X[4]
        r = X[5]

        phi = X[6]
        theta = X[7]

        # Total airspeed
        V = np.sqrt(
            u**2 + v**2 + w**2
        )

        # ========================================================
        # COST FUNCTION
        # ========================================================
        # Each term penalizes undesired behavior
        # ========================================================

        J = (

            # Minimize aircraft dynamics
            800*np.sum(Xdot_val**2)

            # Reach target speed
            + 500*(V - TARGET_SPEED)**2

            # Minimize lateral motion
            + 200*v**2

            # Minimize vertical motion
            + 200*w**2

            # Minimize angular rates
            + 200*p**2
            + 200*q**2
            + 200*r**2

            # Keep wings level
            + 100*phi**2

            # Small pitch angle
            + 50*theta**2
        )

        costs.append(J)

    return np.array(costs)


# ================================================================
# RUN PSO OPTIMIZATION
# ================================================================

def run_pso():

    # ============================================================
    # PSO PARAMETERS
    # ============================================================

    options = {

        # Cognitive coefficient
        'c1': 1.5,

        # Social coefficient
        'c2': 1.5,

        # Inertia coefficient
        'w': 0.7

    }

    # ============================================================
    # LOWER BOUNDS
    # ============================================================

    lower_bounds = np.array([

        70.0,      # u
        -5.0,      # v
        -5.0,      # w

        -0.15,     # p
        -0.15,     # q
        -0.15,     # r

        -0.30,     # phi
        -0.30,     # theta

        -0.25,     # aileron
        -0.35,     # elevator
        -0.25,     # rudder

        0.00       # alive engine throttle

    ])

    # ============================================================
    # UPPER BOUNDS
    # ============================================================

    upper_bounds = np.array([

        90.0,      # u
        5.0,       # v
        5.0,       # w

        0.15,      # p
        0.15,      # q
        0.15,      # r

        0.30,      # phi
        0.30,      # theta

        0.25,      # aileron
        0.35,      # elevator
        0.25,      # rudder

        0.20       # alive engine throttle

    ])

    # ============================================================
    # CREATE PSO OPTIMIZER
    # ============================================================

    optimizer = ps.single.GlobalBestPSO(

        n_particles=45,
        dimensions=12,

        options=options,

        bounds=(
            lower_bounds,
            upper_bounds
        )

    )

    # ============================================================
    # RUN OPTIMIZATION
    # ============================================================

    best_cost, best_position = optimizer.optimize(

        cost_function,
        iters=120,
        verbose=True

    )

    return best_cost, best_position, optimizer


# ================================================================
# COST HISTORY PLOT
# ================================================================
# Shows convergence of PSO
# ================================================================

def plot_cost_history(optimizer):

    plt.figure(figsize=(9,5))

    plt.plot(

        optimizer.cost_history,
        linewidth=2.5,
        label='Best Cost'

    )

    plt.xlabel("Iteration")
    plt.ylabel("Cost Function")

    plt.title("PSO Convergence History")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()
    plt.show()


# ================================================================
# 2D + 3D PSO VISUALIZATION
# ================================================================

def plot_pso_surface(best_position, optimizer):

    # ============================================================
    # VARIABLES TO VISUALIZE
    # ============================================================
    # X-axis -> forward speed u
    # Y-axis -> active engine throttle
    # ============================================================

    u_vals = np.linspace(70, 90, 80)

    throttle_vals = np.linspace(
        0,
        0.2,
        80
    )

    # Create grid
    U_grid, T_grid = np.meshgrid(
        u_vals,
        throttle_vals
    )

    # Empty cost map
    J_grid = np.zeros_like(U_grid)

    # ============================================================
    # BUILD COST SURFACE
    # ============================================================

    for i in range(U_grid.shape[0]):

        for j in range(U_grid.shape[1]):

            # Start from best solution
            particle = best_position.copy()

            # Modify only:
            # forward speed and throttle
            particle[0] = U_grid[i, j]
            particle[11] = T_grid[i, j]

            # Evaluate cost
            J_grid[i, j] = cost_function(
                np.array([particle])
            )[0]

    # ============================================================
    # PARTICLE POSITIONS
    # ============================================================

    particles = optimizer.swarm.position

    # Cost of each particle
    particle_costs = cost_function(particles)

    # ============================================================
    # 2D CONTOUR PLOT
    # ============================================================

    plt.figure(figsize=(11, 8))

    # Filled contour
    contour = plt.contourf(

        U_grid,
        T_grid,
        J_grid,

        levels=40,
        cmap='viridis'

    )

    # Colorbar
    cbar = plt.colorbar(contour)

    cbar.set_label("Cost Function")

    # ============================================================
    # PARTICLES
    # ============================================================

    plt.scatter(

        particles[:,0],
        particles[:,11],

        c=particle_costs,

        cmap='coolwarm',

        s=60,

        edgecolors='black',

        label='Particles'

    )

    # ============================================================
    # BEST SOLUTION
    # ============================================================

    plt.scatter(

        best_position[0],
        best_position[11],

        c='white',

        s=350,

        marker='*',

        edgecolors='black',

        linewidths=2,

        label='Best Solution'

    )

    # ============================================================
    # LABEL BEST SOLUTION
    # ============================================================

    plt.annotate(

        'Global Minimum',

        xy=(
            best_position[0],
            best_position[11]
        ),

        xytext=(
            best_position[0] + 1.5,
            best_position[11] + 0.02
        ),

        arrowprops=dict(
            arrowstyle='->',
            linewidth=2
        ),

        fontsize=11

    )

    # ============================================================
    # AXIS LABELS
    # ============================================================

    plt.xlabel(
        "Forward Velocity u [m/s]"
    )

    plt.ylabel(
        "Active Engine Throttle"
    )

    plt.title(
        "2D PSO Optimization Surface"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # ============================================================
    # 3D COST SURFACE
    # ============================================================

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    # ============================================================
    # SURFACE
    # ============================================================

    surface = ax.plot_surface(

        U_grid,
        T_grid,
        J_grid,

        cmap='viridis',

        alpha=0.85,

        linewidth=0

    )

    # ============================================================
    # COLORBAR
    # ============================================================

    fig.colorbar(

        surface,

        shrink=0.6,
        aspect=12,

        label='Cost Function'

    )

    # ============================================================
    # PARTICLES IN 3D
    # ============================================================

    ax.scatter(

        particles[:,0],
        particles[:,11],
        particle_costs,

        c='red',

        s=50,

        label='Particles'

    )

    # ============================================================
    # BEST SOLUTION
    # ============================================================

    best_cost = cost_function(
        np.array([best_position])
    )[0]

    ax.scatter(

        best_position[0],
        best_position[11],
        best_cost,

        c='white',

        edgecolors='black',

        s=300,

        marker='*',

        label='Best Solution'

    )

    # ============================================================
    # LABELS
    # ============================================================

    ax.set_xlabel(
        'Forward Velocity u [m/s]'
    )

    ax.set_ylabel(
        'Active Engine Throttle'
    )

    ax.set_zlabel(
        'Cost Function'
    )

    ax.set_title(
        '3D PSO Optimization Surface'
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


# ================================================================
# SIMULATE OPTIMAL TRIM SOLUTION
# ================================================================

def simulate_trim(X_trim, U_trim):

    n_out = int(TF / DT_OUT) + 1

    t_hist = np.linspace(
        0.0,
        TF,
        n_out
    )

    X_hist = np.zeros((n_out, 9))

    X = X_trim.copy()

    X_hist[0] = X

    current_time = 0.0

    for k in range(n_out - 1):

        t_next = t_hist[k+1]

        while current_time < t_next - 1e-12:

            h = min(
                DT,
                t_next - current_time
            )

            # ====================================================
            # KEEP FAILED ENGINE OFF
            # ====================================================

            U_now = U_trim.copy()

            if FAILED_ENGINE == 1:
                U_now[3] = 0.0

            else:
                U_now[4] = 0.0

            # Integrate aircraft
            X = rk4_step(
                X,
                U_now,
                h
            )

            current_time += h

        X_hist[k+1] = X

    return t_hist, X_hist


# ================================================================
# PLOT AIRCRAFT RESPONSE
# ================================================================

def plot_trim_states(t_hist, X_hist):

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
        figsize=(14,10)
    )

    axes = axes.flatten()

    for i in range(9):

        axes[i].plot(

            t_hist,
            X_hist[:,i],

            linewidth=1.8,

            label=state_names[i]

        )

        axes[i].set_title(state_names[i])

        axes[i].set_xlabel("Time [s]")

        axes[i].grid(True)

        axes[i].legend()

    fig.suptitle(
        "Aircraft Response Using PSO Trim Solution",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ================================================================
# MAIN PROGRAM
# ================================================================

if __name__ == "__main__":

    print("\n================================================")
    print("RUNNING PSO OPTIMIZATION")
    print("================================================")

    print(f"\nFailed engine: {FAILED_ENGINE}")

    # ============================================================
    # RUN PSO
    # ============================================================

    best_cost, best_position, optimizer = run_pso()

    # ============================================================
    # BUILD OPTIMAL STATE + CONTROL
    # ============================================================

    X_trim, U_trim = build_state_control(
        best_position
    )

    # Evaluate derivatives
    Xdot_trim = xdot(
        X_trim,
        U_trim
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print("\n================================================")
    print("OPTIMIZATION RESULTS")
    print("================================================")

    print("\nBest Cost:")
    print(best_cost)

    print("\nOptimal State X_trim:")
    print(X_trim)

    print("\nOptimal Control U_trim:")
    print(U_trim)

    print("\nAircraft Derivatives X_dot:")
    print(Xdot_trim)

    # ============================================================
    # PLOTS
    # ============================================================

    plot_cost_history(optimizer)

    plot_pso_surface(
        best_position,
        optimizer
    )

    # ============================================================
    # SIMULATE OPTIMAL TRIM
    # ============================================================

    t_hist, X_hist = simulate_trim(
        X_trim,
        U_trim
    )

    # ============================================================
    # SHOW AIRCRAFT RESPONSE
    # ============================================================

    plot_trim_states(
        t_hist,
        X_hist
    )

    print("\nOptimization completed.")