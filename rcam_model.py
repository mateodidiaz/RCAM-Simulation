import numpy as np


def xdot(X, U):
    """
    RCAM nonlinear model
    States:  X = [u, v, w, p, q, r, phi, theta, psi]
    Controls: U = [da, de, dr, dth1, dth2]
    """

    x1 = X[0]   # u
    x2 = X[1]   # v
    x3 = X[2]   # w
    x4 = X[3]   # p
    x5 = X[4]   # q
    x6 = X[5]   # r
    x7 = X[6]   # phi
    x8 = X[7]   # theta
    x9 = X[8]   # psi

    u1 = U[0]   # aileron
    u2 = U[1]   # stabilizer
    u3 = U[2]   # rudder
    u4 = U[3]   # throttle 1
    u5 = U[4]   # throttle 2

    g = 9.81
    rho = 1.225
    m = 120000.0
    cbar = 6.6
    lt = 24.8
    S = 260.0
    St = 64.0
    b = 44.8

    Ixx = 40.07 * m
    Iyy = 64.00 * m
    Izz = 99.92 * m
    Ixz = -2.0923 * m

    I_body = np.array([
        [Ixx, 0.0, Ixz],
        [0.0, Iyy, 0.0],
        [Ixz, 0.0, Izz]
    ])

    r_eng1 = np.array([0.0,  7.94, -1.90])
    r_eng2 = np.array([0.0, -7.94, -1.90])

    r_cg = np.array([0.23 * cbar, 0.0, 0.10 * cbar])
    r_ac = np.array([0.12 * cbar, 0.0, 0.0])

    alpha0 = -11.5 * np.pi / 180.0

    # Step 1: saturation
    u1 = np.clip(u1, -25.0 * np.pi / 180.0,  25.0 * np.pi / 180.0)
    u2 = np.clip(u2, -25.0 * np.pi / 180.0,  10.0 * np.pi / 180.0)
    u3 = np.clip(u3, -30.0 * np.pi / 180.0,  30.0 * np.pi / 180.0)
    u4 = np.clip(u4, 0.0, 10.0 * np.pi / 180.0)
    u5 = np.clip(u5, 0.0, 10.0 * np.pi / 180.0)

    # Step 2: intermediate variables
    Vb = np.array([x1, x2, x3])
    omega = np.array([x4, x5, x6])

    Va = np.sqrt(x1**2 + x2**2 + x3**2)
    Va = max(Va, 1e-6)

    alpha = np.arctan2(x3, x1)
    beta = np.arcsin(np.clip(x2 / Va, -1.0, 1.0))
    Qdyn = 0.5 * rho * Va**2

    # Step 3: aerodynamic coefficients
    eps_down = 0.25 * (alpha - alpha0)
    alpha_t = alpha - eps_down + u2 + 1.3 * x5 * lt / Va

    CL_wb = 5.5 * (alpha - alpha0)
    CL_t = 3.1 * (St / S) * alpha_t
    CL = CL_wb + CL_t

    CD = 0.13 + 0.07 * (5.5 * alpha + 0.654)**2
    CY = -1.6 * beta + 0.24 * u3

    # Step 4: aerodynamic force in body
    C_Fs = np.array([-CD, CY, -CL])

    C_w_s = np.array([
        [ np.cos(beta), np.sin(beta), 0.0],
        [-np.sin(beta), np.cos(beta), 0.0],
        [ 0.0,          0.0,          1.0]
    ])

    C_Fw = C_w_s @ C_Fs

    C_b_w = np.array([
        [ np.cos(alpha), 0.0, -np.sin(alpha)],
        [ 0.0,           1.0,  0.0],
        [ np.sin(alpha), 0.0,  np.cos(alpha)]
    ])

    F_A_b = C_b_w @ (Qdyn * S * C_Fw)

    # Step 5: aerodynamic moment coefficients about AC
    eta = np.array([
        -1.4 * beta,
        -0.59 - 3.1 * (St * lt / (S * cbar)) * (alpha - eps_down),
        (1.0 - alpha * 180.0 / (15.0 * np.pi)) * beta
    ])

    A_rate = np.array([
        [-11.0, 0.0,   5.0],
        [  0.0, -4.03 * (St * lt**2) / (S * cbar**2), 0.0],
        [  1.7, 0.0, -11.5]
    ])

    A_ctrl = np.array([
        [-0.6, 0.0,  0.22],
        [ 0.0, -3.1 * (St * lt) / (S * cbar), 0.0],
        [ 0.0, 0.0, -0.63]
    ])

    delta_ctrl = np.array([u1, u2, u3])
    C_M_ac = eta + (A_rate @ omega) / Va + A_ctrl @ delta_ctrl

    # Step 6: aerodynamic moment about AC
    M_A_ac_b = Qdyn * S * np.array([
        b * C_M_ac[0],
        cbar * C_M_ac[1],
        b * C_M_ac[2]
    ])

    # Step 7: aerodynamic moment about CG
    M_A_cg_b = M_A_ac_b + np.cross(F_A_b, (r_cg - r_ac))

    # Step 8: propulsion
    F1 = u4 * m * g
    F2 = u5 * m * g

    F_E1_b = np.array([F1, 0.0, 0.0])
    F_E2_b = np.array([F2, 0.0, 0.0])
    F_E_b = F_E1_b + F_E2_b

    M_E1_cg_b = np.cross(r_eng1, F_E1_b)
    M_E2_cg_b = np.cross(r_eng2, F_E2_b)
    M_E_cg_b = M_E1_cg_b + M_E2_cg_b

    # Step 9: gravity
    F_g_b = m * np.array([
        -g * np.sin(x8),
         g * np.cos(x8) * np.sin(x7),
         g * np.cos(x8) * np.cos(x7)
    ])

    # Step 10: explicit first-order form
    F_total_b = F_g_b + F_E_b + F_A_b
    M_total_cg_b = M_E_cg_b + M_A_cg_b

    V_dot = (1.0 / m) * F_total_b - np.cross(omega, Vb)
    omega_dot = np.linalg.solve(I_body, M_total_cg_b - np.cross(omega, I_body @ omega))

    phi = x7
    theta = x8
    cos_theta = np.cos(theta)
    if abs(cos_theta) < 1e-6:
        cos_theta = 1e-6

    H = np.array([
        [1.0, np.sin(phi) * np.tan(theta),  np.cos(phi) * np.tan(theta)],
        [0.0, np.cos(phi),                 -np.sin(phi)],
        [0.0, np.sin(phi) / cos_theta,      np.cos(phi) / cos_theta]
    ])

    euler_dot = H @ omega

    X_dot = np.array([
        V_dot[0], V_dot[1], V_dot[2],
        omega_dot[0], omega_dot[1], omega_dot[2],
        euler_dot[0], euler_dot[1], euler_dot[2]
    ])

    return X_dot