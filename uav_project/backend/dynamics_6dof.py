"""
6-DOF rigid-body dynamics for the fixed-wing UAV.

State vector (12): [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    pn, pe, pd : NED position (m)      (pd is DOWN-positive)
    u, v, w    : body-frame velocity (m/s)
    phi,theta,psi : Euler angles, 3-2-1 (roll, pitch, yaw) (rad)
    p, q, r    : body-frame angular rates (rad/s)

Controls (4): [de, da, dr, throttle]   de,da,dr in RAD, throttle in [0,1]

Body axes: x=forward, y=right, z=down.

Equations follow the standard small-UAV rigid-body formulation (see
Beard & McLain, "Small Unmanned Aircraft: Theory and Practice").
"""
import numpy as np
import aircraft_params as P


# ---------------------------------------------------------------------------
# Payload -> physical effect mapping
# ---------------------------------------------------------------------------
def payload_effects(payload_level):
    """Return mass-derived payload effects for ``payload_level`` in [0, 1].

    The normalized input is converted to payload mass and position first.
    The aerodynamic model uses the longitudinal CG shift in chord units.
    Elevator effectiveness is not silently derated: its value is an
    aerodynamic derivative and must be identified independently.
    """
    pl = float(np.clip(payload_level, 0.0, 1.0))
    payload_mass = P.payload_mass_max * pl
    mass_eff = P.mass + payload_mass
    cg_x_m = payload_mass * P.payload_x_m / max(mass_eff, 1e-9)
    # Positive cg_shift denotes aft motion because body x is forward.
    cg_shift = -cg_x_m / P.c
    return cg_shift, cg_x_m, 1.0, mass_eff


def payload_inertias(payload_level):
    """Return a self-consistent equivalent-payload inertia tensor.

    The project does not yet have measured payload geometry, so the payload is
    represented as an equivalent mass-distribution scale. All inertia terms
    change together; changing only Jy would be internally inconsistent.
    """
    pl = float(np.clip(payload_level, 0.0, 1.0))
    mp = P.payload_mass_max * pl
    total_mass = P.mass + mp
    cg_x = mp * P.payload_x_m / max(total_mass, 1e-9)
    # Parallel-axis theorem: shift the airframe and payload inertias to the
    # combined CG, then add the payload's intrinsic box inertia.
    air_dx = -cg_x
    pay_dx = P.payload_x_m - cg_x
    Jx = P.Jx + mp * P.payload_Jx_per_kg
    Jy = P.Jy + P.mass * air_dx ** 2 + mp * pay_dx ** 2 + mp * P.payload_Jy_per_kg
    Jz = P.Jz + P.mass * air_dx ** 2 + mp * pay_dx ** 2 + mp * P.payload_Jz_per_kg
    Jxz = P.Jxz
    return Jx, Jy, Jz, Jxz


def _dcm_body_to_ned(phi, theta, psi):
    """Direction-cosine matrix for body vectors to NED vectors."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cth * cpsi, sphi * sth * cpsi - cphi * spsi,
         cphi * sth * cpsi + sphi * spsi],
        [cth * spsi, sphi * sth * spsi + cphi * cpsi,
         cphi * sth * spsi - sphi * cpsi],
        [-sth, sphi * cth, cphi * cth],
    ])


def cg_effect_on_derivatives(cg_shift, cm_de_derate):
    """Apply CG-shift and elevator-derating effects to the pitching-moment
    derivatives. An aft CG shift REDUCES static stability (Cm_alpha becomes
    less negative / can cross zero).

    Sanity check: instability (Cm_alpha_new >= 0) occurs near
    cg_shift ~= |Cm_alpha| / CL_alpha ~= 0.60/4.9 ~= 0.122
    """
    Cm_alpha_new = P.Cm_alpha + P.CL_alpha * cg_shift
    Cm0_new = P.Cm0 + P.CL0 * cg_shift
    Cm_de_new = P.Cm_de * cm_de_derate
    return Cm0_new, Cm_alpha_new, Cm_de_new


def payload_cg_position(payload_level):
    """Return combined CG longitudinal position relative to empty CG (m)."""
    pl = float(np.clip(payload_level, 0.0, 1.0))
    mp = P.payload_mass_max * pl
    return float(mp * P.payload_x_m / max(P.mass + mp, 1e-9))


def static_margin(payload_level):
    """Linear longitudinal static margin, positive for static stability."""
    cg_shift, _, _, _ = payload_effects(payload_level)
    _, cm_alpha, _ = cg_effect_on_derivatives(cg_shift, 1.0)
    return float(-cm_alpha / P.CL_alpha)


# ---------------------------------------------------------------------------
# Aerodynamics
# ---------------------------------------------------------------------------
def aero_forces_moments(state, controls, payload_level, wind=(0.0, 0.0, 0.0),
                         tail_incidence=0.0):
    """Compute aerodynamic + thrust forces/moments in the body frame.

    Parameters
    ----------
    state : array-like (12,)
    controls : array-like (4,)  [de, da, dr, throttle]
    payload_level : float in [0,1]
    wind : 3-vector, atmospheric wind in NED coordinates. It is rotated into
           body coordinates before computing relative airspeed.
    tail_incidence : float, radians (morphing-tail incidence angle)

    Returns
    -------
    forces  : np.array([Fx, Fy, Fz])   (body frame, N)  -- aero + thrust only
    moments : np.array([l, m, n])      (body frame, N*m)
    alpha, beta, Va : floats
    Jy_eff, mass_eff : floats (payload-adjusted inertia / mass)
    D : float (drag force, N) for logging/metrics
    """
    state = np.asarray(state, dtype=float)
    controls = np.asarray(controls, dtype=float)
    wind = np.asarray(wind, dtype=float)

    u, v, w = state[3], state[4], state[5]
    p, q, r = state[9], state[10], state[11]
    de, da, dr, throttle = controls

    wind_body = _dcm_body_to_ned(state[6], state[7], state[8]).T @ wind
    ur = u - wind_body[0]
    vr = v - wind_body[1]
    wr = w - wind_body[2]

    Va = float(np.sqrt(ur ** 2 + vr ** 2 + wr ** 2))
    Va_safe = max(Va, 1e-6)
    alpha = np.arctan2(wr, ur)
    beta = np.arcsin(np.clip(vr / Va_safe, -1.0, 1.0))
    qbar = 0.5 * P.rho * Va ** 2

    cg_shift, _, cm_de_derate, mass_eff = payload_effects(payload_level)
    _, Jy_eff, _, _ = payload_inertias(payload_level)
    Cm0_e, Cm_alpha_e, Cm_de_e = cg_effect_on_derivatives(cg_shift, cm_de_derate)

    p_hat = p * P.b / (2.0 * Va_safe)
    q_hat = q * P.c / (2.0 * Va_safe)
    r_hat = r * P.b / (2.0 * Va_safe)

    CL = P.CL0 + P.CL_alpha * alpha + P.CL_q * q_hat + P.CL_de * de
    # Include the defined profile-drag rise with angle of attack. Without
    # this term P.CD_alpha was dead data and trim-drag/efficiency effects
    # were understated.
    CD = (P.CD0 + P.CD_alpha * alpha ** 2
          + P.k_induced * CL ** 2 + P.CD_de_k * de ** 2)
    Cm = (Cm0_e + Cm_alpha_e * alpha + P.Cm_q * q_hat + Cm_de_e * de
          + P.Cm_dtail * tail_incidence)

    CY = P.CY_beta * beta + P.CY_dr * dr
    Cl = (P.Cl_beta * beta + P.Cl_p * p_hat + P.Cl_r * r_hat
          + P.Cl_da * da + P.Cl_dr * dr)
    Cn = (P.Cn_beta * beta + P.Cn_p * p_hat + P.Cn_r * r_hat
          + P.Cn_da * da + P.Cn_dr * dr)

    L = qbar * P.S * CL
    D = qbar * P.S * CD
    Y = qbar * P.S * CY

    l = qbar * P.S * P.b * Cl
    m = qbar * P.S * P.c * Cm
    n = qbar * P.S * P.b * Cn

    ca, sa = np.cos(alpha), np.sin(alpha)
    Fx = -D * ca + L * sa
    Fz = -D * sa - L * ca
    Fy = Y

    # Morphing incidence changes tail lift and induced drag as well as the
    # pitching moment above. The baseline tail contribution is absorbed into
    # the identified aircraft coefficients; only the commanded increment is
    # added here.
    tail_cl_delta = -P.CL_alpha_tail * tail_incidence
    tail_lift_delta = qbar * P.S_tail * P.tail_efficiency * tail_cl_delta
    tail_drag_delta = qbar * P.S_tail * P.CD_tail_induced * tail_cl_delta ** 2
    Fx += -tail_drag_delta * ca + tail_lift_delta * sa
    Fz += -tail_drag_delta * sa - tail_lift_delta * ca
    D += tail_drag_delta

    Fx += throttle * 12.0  # simple thrust model, N (max 12N at full throttle)

    forces = np.array([Fx, Fy, Fz])
    moments = np.array([l, m, n])

    return forces, moments, alpha, beta, Va, Jy_eff, mass_eff, D


# ---------------------------------------------------------------------------
# Full 12-state derivative
# ---------------------------------------------------------------------------
def dynamics(state, controls, payload_level, wind=(0.0, 0.0, 0.0),
             tail_incidence=0.0):
    """Return d(state)/dt, a 12-vector, for the given state/controls."""
    state = np.asarray(state, dtype=float)
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state

    forces, moments, alpha, beta, Va, Jy_eff, mass_eff, D = aero_forces_moments(
        state, controls, payload_level, wind, tail_incidence)
    Fx, Fy, Fz = forces
    l, m, n = moments

    # --- gravity, resolved into body axes ---
    gx = -P.g * np.sin(theta)
    gy = P.g * np.cos(theta) * np.sin(phi)
    gz = P.g * np.cos(theta) * np.cos(phi)

    # --- translational kinematics: body vel -> NED position rate ---
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    pn_dot = (cth * cpsi) * u + (sphi * sth * cpsi - cphi * spsi) * v + \
             (cphi * sth * cpsi + sphi * spsi) * w
    pe_dot = (cth * spsi) * u + (sphi * sth * spsi + cphi * cpsi) * v + \
             (cphi * sth * spsi - sphi * cpsi) * w
    pd_dot = (-sth) * u + (sphi * cth) * v + (cphi * cth) * w

    # --- translational dynamics (body frame) ---
    u_dot = r * v - q * w + Fx / mass_eff + gx
    v_dot = p * w - r * u + Fy / mass_eff + gy
    w_dot = q * u - p * v + Fz / mass_eff + gz

    # --- Euler-angle kinematics ---
    tth = np.tan(theta)
    phi_dot = p + q * sphi * tth + r * cphi * tth
    theta_dot = q * cphi - r * sphi
    psi_dot = (q * sphi + r * cphi) / cth

    # --- rotational dynamics with Jxz coupling, Jy_eff for pitch axis ---
    Jx, Jy_eff, Jz, Jxz = payload_inertias(payload_level)
    Gamma = Jx * Jz - Jxz ** 2
    Gamma1 = (Jxz * (Jx - Jy_eff + Jz)) / Gamma
    Gamma2 = (Jz * (Jz - Jy_eff) + Jxz ** 2) / Gamma
    Gamma3 = Jz / Gamma
    Gamma4 = Jxz / Gamma
    Gamma5 = (Jz - Jx) / Jy_eff
    Gamma6 = Jxz / Jy_eff
    Gamma7 = ((Jx - Jy_eff) * Jx + Jxz ** 2) / Gamma
    Gamma8 = Jx / Gamma

    p_dot = Gamma1 * p * q - Gamma2 * q * r + Gamma3 * l + Gamma4 * n
    q_dot = Gamma5 * p * r - Gamma6 * (p ** 2 - r ** 2) + m / Jy_eff
    r_dot = Gamma7 * p * q - Gamma1 * q * r + Gamma4 * l + Gamma8 * n

    return np.array([pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                      phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])


def rk4_step(state, controls, payload_level, dt, wind=(0.0, 0.0, 0.0),
             tail_incidence=0.0):
    """Single RK4 integration step."""
    k1 = dynamics(state, controls, payload_level, wind, tail_incidence)
    k2 = dynamics(state + 0.5 * dt * k1, controls, payload_level, wind, tail_incidence)
    k3 = dynamics(state + 0.5 * dt * k2, controls, payload_level, wind, tail_incidence)
    k4 = dynamics(state + dt * k3, controls, payload_level, wind, tail_incidence)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
