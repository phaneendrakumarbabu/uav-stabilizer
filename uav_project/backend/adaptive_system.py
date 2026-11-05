"""
Adaptive-control support: morphing-tail incidence command (solved via a
precomputed lookup table) and gain-scheduling functions for pitch/roll/yaw.
"""
import numpy as np
from scipy.optimize import least_squares

import aircraft_params as P
from dynamics_6dof import dynamics
from trim import find_trim

TAIL_INC_LIMIT = np.radians(15.0)
TRIM_ALPHA_LIMIT = np.radians(25.0)

# The estimator is deliberately built from causal IMU/command summaries.
# Keep this list and the implementation below shared by dataset generation
# and deployment; a silent train/deploy feature mismatch is unacceptable in a
# paper experiment.
ESTIMATOR_FEATURE_NAMES = (
    "q_mean", "q_std", "q_max", "q_min",
    "az_mean", "az_std", "az_max", "az_min",
    "elev_max", "elev_min", "va_mean", "va_std",
    "tail_mean", "tail_std",
    "q_half_delta", "az_half_delta", "elev_half_delta",
    "va_half_delta", "tail_half_delta", "turbulence",
    "elev_mean",   # mean commanded-elevator offset: the direct trim signature of a CG shift
)
ESTIMATOR_FEATURE_COUNT = len(ESTIMATOR_FEATURE_NAMES)


def estimate_turbulence(q_recent, az_recent):
    """Estimate disturbance intensity using only recent IMU signals.

    The constants are sensor-calibration scales, not the scenario's gust
    input.  The estimator is intentionally conservative: steady bias is
    handled by the payload estimator while high-frequency q/vertical-force
    variation is treated as turbulence.
    """
    q = np.asarray(q_recent, dtype=float)
    az = np.asarray(az_recent, dtype=float)
    if q.size < 20 or az.size < 20:
        return 0.0
    q_std = float(np.std(q[-min(100, q.size):]))
    az_std = float(np.std(az[-min(100, az.size):]))
    q_component = q_std / 0.08       # rad/s, calibrated IMU scale
    az_component = az_std / 0.80     # m/s^2, calibrated accelerometer scale
    return float(np.clip(1.5 * 0.5 * (q_component + az_component), 0.0, 2.0))


def estimator_features(q, az, elevator, va, tail):
    """Return the exact 20-feature vector used by train and deployment."""
    arrays = [np.asarray(x, dtype=float) for x in (q, az, elevator, va, tail)]
    if any(x.size < 2 for x in arrays):
        raise ValueError("estimator window must contain at least two samples")
    q, az, elevator, va, tail = arrays
    n = min(x.size for x in arrays)
    q, az, elevator, va, tail = [x[-n:] for x in arrays]
    mid = n // 2
    mean_delta = lambda x: float(np.mean(x[mid:]) - np.mean(x[:mid]))
    turb = estimate_turbulence(q, az)
    return np.asarray([
        q.mean(), q.std(), q.max(), q.min(),
        az.mean(), az.std(), az.max(), az.min(),
        elevator.max(), elevator.min(), va.mean(), va.std(),
        tail.mean(), tail.std(),
        mean_delta(q), mean_delta(az), mean_delta(elevator),
        mean_delta(va), mean_delta(tail), turb, elevator.mean(),
    ], dtype=float)


def tail_identification_command(t, phase=0.0):
    """Small bounded excitation used in both training and deployment.

    It makes the estimator robust to ordinary tail motion without encoding the
    true payload label.  The adaptive trim command is added separately.
    """
    return float(
        np.radians(0.20) * np.sin(2.0 * np.pi * 0.08 * t + phase)
        + np.radians(0.10) * np.sin(2.0 * np.pi * 0.21 * t + 0.7 * phase)
    )

# Grid used to precompute the morphing-tail lookup table.
VA_GRID = np.linspace(10.0, 22.0, 13)
PAYLOAD_GRID = np.linspace(0.0, 1.0, 11)


def _solve_tail_incidence_for_zero_elevator(Va, payload_level, x0=None):
    """Solve for (alpha, tail_incidence, throttle) such that the aircraft
    trims in level flight with elevator EXACTLY zero. This is a genuine
    re-solve of the trim equations with tail_incidence as a free variable
    replacing elevator (NOT a fitted/interpolated formula)."""
    if x0 is None:
        x0 = [0.05, 0.0, 0.5]

    def residual(x):
        alpha, tail_inc, throttle = x
        theta = alpha
        u = Va * np.cos(alpha)
        w = Va * np.sin(alpha)
        state = np.array([0, 0, -P.alt_trim, u, 0.0, w,
                           0.0, theta, 0.0, 0.0, 0.0, 0.0])
        controls = np.array([0.0, 0.0, 0.0, throttle])  # elevator fixed at 0
        deriv = dynamics(state, controls, payload_level,
                          wind=(0.0, 0.0, 0.0), tail_incidence=tail_inc)
        return [deriv[3], deriv[5], deriv[10]]

    soln = least_squares(
        residual,
        np.asarray(x0, dtype=float),
        bounds=([-TRIM_ALPHA_LIMIT, -TAIL_INC_LIMIT, 0.0],
                [TRIM_ALPHA_LIMIT, TAIL_INC_LIMIT, 1.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )
    alpha, tail_inc, throttle = soln.x
    residual_norm = float(np.linalg.norm(residual(soln.x), ord=np.inf))
    return float(tail_inc), bool(soln.success and residual_norm <= 1e-6)


def build_morphing_tail_table(va_grid=VA_GRID, payload_grid=PAYLOAD_GRID, verbose=False):
    """Precompute the (Va x payload_level) -> tail_incidence table."""
    table = np.zeros((len(va_grid), len(payload_grid)))
    failures = []
    x0 = [0.05, 0.0, 0.5]
    for i, Va in enumerate(va_grid):
        for j, pl in enumerate(payload_grid):
            tail_inc, converged = _solve_tail_incidence_for_zero_elevator(Va, pl, x0=x0)
            table[i, j] = tail_inc
            if not converged:
                failures.append((float(Va), float(pl)))
            if converged:
                x0 = [x0[0], tail_inc, x0[2]]  # warm-start next solve
            if verbose:
                print(f"Va={Va:.1f} pl={pl:.2f} -> tail_inc={np.degrees(tail_inc):.3f}deg "
                      f"converged={converged}")
    if failures:
        raise RuntimeError(f"unconverged morphing-tail trim points: {failures}")
    return va_grid, payload_grid, table


_TABLE_CACHE = None


def _get_table():
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        _TABLE_CACHE = build_morphing_tail_table()
    return _TABLE_CACHE


def morphing_tail_command(payload_level_est, Va_est):
    """Bilinear interpolation of the precomputed lookup table, clipped to
    +-15 degrees."""
    va_grid, pl_grid, table = _get_table()

    Va_c = np.clip(Va_est, va_grid[0], va_grid[-1])
    pl_c = np.clip(payload_level_est, pl_grid[0], pl_grid[-1])

    i1 = np.searchsorted(va_grid, Va_c, side="right") - 1
    i1 = int(np.clip(i1, 0, len(va_grid) - 2))
    i2 = i1 + 1
    j1 = np.searchsorted(pl_grid, pl_c, side="right") - 1
    j1 = int(np.clip(j1, 0, len(pl_grid) - 2))
    j2 = j1 + 1

    va0, va1 = va_grid[i1], va_grid[i2]
    pl0, pl1 = pl_grid[j1], pl_grid[j2]

    tv = 0.0 if va1 == va0 else (Va_c - va0) / (va1 - va0)
    tp = 0.0 if pl1 == pl0 else (pl_c - pl0) / (pl1 - pl0)

    f00 = table[i1, j1]
    f10 = table[i2, j1]
    f01 = table[i1, j2]
    f11 = table[i2, j2]

    val = (f00 * (1 - tv) * (1 - tp) + f10 * tv * (1 - tp) +
           f01 * (1 - tv) * tp + f11 * tv * tp)
    # The aircraft controller retains the nominal elevator trim. Therefore
    # only the payload-induced change from the nominal tail incidence belongs
    # in the command; applying the absolute zero-elevator-trim value would
    # double-count the baseline trim and bias pitch.
    # BASELINE = zero-payload column (index 0). The earlier code used the table
    # at the lower payload neighbour (j1), which cancelled almost the whole
    # payload correction and left only the interpolation inside one grid cell.
    baseline = table[i1, 0] * (1 - tv) + table[i2, 0] * tv
    return float(np.clip(val - baseline, -TAIL_INC_LIMIT, TAIL_INC_LIMIT))


_ELEV_TABLE_CACHE = None
def _get_elev_table():
    global _ELEV_TABLE_CACHE
    if _ELEV_TABLE_CACHE is None:
        tab = np.zeros((len(VA_GRID), len(PAYLOAD_GRID)))
        for i, Va in enumerate(VA_GRID):
            for j, pl in enumerate(PAYLOAD_GRID):
                tr = find_trim(Va, payload_level=pl)
                if not tr["converged"]:
                    raise RuntimeError(f"elevator trim failed at Va={Va}, pl={pl}")
                tab[i, j] = tr["elevator"]
        _ELEV_TABLE_CACHE = tab
    return _ELEV_TABLE_CACHE
def elevator_trim_delta(payload_level_est, Va_est):
    """Elevator trim change (rad) needed for the estimated payload, relative
    to payload 0 at the same airspeed (bilinear table from the trim solver).
    This is the elevator-only baseline for the morphing-tail trim command."""
    tab = _get_elev_table()
    Va_c = float(np.clip(Va_est, VA_GRID[0], VA_GRID[-1]))
    pl_c = float(np.clip(payload_level_est, 0.0, 1.0))
    col0 = np.array([np.interp(pl_c, PAYLOAD_GRID, tab[i]) for i in range(len(VA_GRID))])
    return float(np.interp(Va_c, VA_GRID, col0 - tab[:, 0]))
# ---------------------------------------------------------------------------
# Gain scheduling
# ---------------------------------------------------------------------------
def adaptive_gains(pl_est, turb_est, base=(-2.0, -0.3, -0.4)):
    """Pitch-axis PID gain scheduling based on estimated payload level and
    turbulence level."""
    kp0, ki0, kd0 = base
    pl_est = float(np.clip(pl_est, 0.0, 1.0))
    turb_est = float(np.clip(turb_est, 0.0, 2.0))
    # Conservative scheduling: the payload model changes slowly; large gain
    # jumps would turn estimator error into control chatter.
    comp = min(1.0 + 0.35 * pl_est, 1.35)
    turb_frac = min(turb_est / 1.5, 1.0)
    # Turbulence must not turn measurement noise into derivative chatter.
    # Keep the scheduled boost deliberately modest and rate-limit the gains
    # in final_comparison.py before applying them to the PID.
    kd_f = min(1.0 + 0.10 * turb_frac, 1.10)
    ki_f = 1.0 - 0.5 * turb_frac
    kp = float(np.clip(kp0 * comp, -5.0, -0.5))
    kd = float(np.clip(kd0 * comp * kd_f, -2.5, -0.05))
    ki = float(np.clip(ki0 * ki_f, -0.5, -0.01))
    return kp, ki, kd


def adaptive_lateral_gains(turb_est, roll_base=(3.0, 0.1, 0.6),
                            yaw_base=(2.5, 0.3, 0.5)):
    """Roll- and yaw-axis PID gain scheduling based on estimated turbulence."""
    turb_frac = min(float(np.clip(turb_est, 0.0, 2.0)) / 1.5, 1.0)
    kp_f = 1.0 + 0.15 * turb_frac
    ki_f = 1.0 - 0.15 * turb_frac
    kd_f = 1.0 

    rkp, rki, rkd = roll_base
    ykp, yki, ykd = yaw_base

    roll_gains = (float(np.clip(rkp * kp_f, 0.5, 5.0)),
                  float(np.clip(rki * ki_f, 0.01, 0.5)),
                  float(np.clip(rkd * kd_f, 0.1, 1.5)))
    yaw_gains = (float(np.clip(ykp * kp_f, 0.5, 5.0)),
                 float(np.clip(yki * ki_f, 0.01, 0.5)),
                 float(np.clip(ykd * kd_f, 0.1, 1.5)))
    return roll_gains, yaw_gains


if __name__ == "__main__":
    print("Building morphing-tail lookup table (bounded trim solve at every grid point)...")
    va_grid, pl_grid, table = build_morphing_tail_table(verbose=True)
    print("Table shape:", table.shape)
    print("Sample query morphing_tail_command(0.5, 15.0) ->",
          np.degrees(morphing_tail_command(0.5, 15.0)), "deg")
