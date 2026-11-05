"""
Trim solver: finds alpha, elevator, throttle for steady, straight-and-level
flight at a given airspeed and payload level (zero du, dw, dq).
"""
import numpy as np
from scipy.optimize import least_squares

import aircraft_params as P
from dynamics_6dof import dynamics


TRIM_ALPHA_LIMIT = np.radians(25.0)


def _trim_state_controls(x, Va):
    alpha, de, throttle = x
    theta = alpha  # straight-and-level => flight path angle gamma = theta-alpha = 0
    u = Va * np.cos(alpha)
    w = Va * np.sin(alpha)
    v = 0.0
    state = np.array([0.0, 0.0, -P.alt_trim, u, v, w,
                       0.0, theta, 0.0, 0.0, 0.0, 0.0])
    controls = np.array([de, 0.0, 0.0, throttle])
    return state, controls


def find_trim(Va, payload_level=0.0, tail_incidence=0.0, x0=None):
    """Solve for (alpha, elevator, throttle) giving steady, straight-and-level
    flight at airspeed `Va` and the given `payload_level`.

    Returns a dict: alpha, elevator, throttle, theta, state, controls,
    residual (should be ~0), converged (bool).
    """
    if x0 is None:
        x0 = [0.05, 0.0, 0.5]

    def residual(x):
        state, controls = _trim_state_controls(x, Va)
        deriv = dynamics(state, controls, payload_level,
                          wind=(0.0, 0.0, 0.0), tail_incidence=tail_incidence)
        return [deriv[3], deriv[5], deriv[10]]  # u_dot, w_dot, q_dot

    # Solve the bounded problem directly. Solving unconstrained and clipping
    # afterward can return a state whose controls no longer satisfy trim.
    soln = least_squares(
        residual,
        np.asarray(x0, dtype=float),
        bounds=([-TRIM_ALPHA_LIMIT, -P.ELEVATOR_LIMIT, 0.0],
                [TRIM_ALPHA_LIMIT, P.ELEVATOR_LIMIT, 1.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )
    sol = soln.x
    alpha, de, throttle = sol
    de_clipped = float(de)
    throttle_clipped = float(throttle)

    state, controls = _trim_state_controls(
        [alpha, de_clipped, throttle_clipped], Va)
    res = np.asarray(residual([alpha, de_clipped, throttle_clipped]), dtype=float)
    residual_norm = float(np.linalg.norm(res, ord=np.inf))

    return {
        "alpha": float(alpha),
        "elevator": de_clipped,
        "elevator_raw": float(de),
        "throttle": throttle_clipped,
        "throttle_raw": float(throttle),
        "theta": float(alpha),
        "state": state,
        "controls": controls,
        "residual": res,
        "converged": bool(soln.success and residual_norm <= 1e-6),
        "residual_norm": residual_norm,
        "message": str(soln.message),
    }


if __name__ == "__main__":
    for pl in (0.0, 0.5, 1.0):
        t = find_trim(P.Va_trim, payload_level=pl)
        print(f"payload={pl}: alpha={np.degrees(t['alpha']):.3f}deg  "
              f"de={np.degrees(t['elevator']):.3f}deg  throttle={t['throttle']:.3f}  "
              f"converged={t['converged']}  residual={t['residual']}")
