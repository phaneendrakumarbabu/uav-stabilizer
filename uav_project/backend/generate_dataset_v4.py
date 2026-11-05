"""
Generates the training dataset for the payload-level neural-net estimator.

Each of ~5000 runs simulates 3 seconds (150 steps @ dt=0.02) of a
PID-stabilized aircraft using the same common airspeed/altitude outer hold as
deployment, with the pitch baseline derived from the NOMINAL (payload=0) trim.
The true payload_level is applied to
the physics at a random changeover time, but is NEVER used to set the
reference/baseline (no label leakage). Twenty window-summary features are
extracted per run; the label is the true payload_level active at the end
of the 3s window (0 if it hasn't changed yet).
"""
import numpy as np

import aircraft_params as P
from dynamics_6dof import rk4_step, dynamics
from trim import find_trim
from pid_controller import PID, RateLimitedActuator
from flight_hold import FlightHold
from adaptive_system import (ESTIMATOR_FEATURE_COUNT, estimator_features,
                             tail_identification_command)

DT = 0.02
N_STEPS = 150
WINDOW_T = N_STEPS * DT  # 3.0 s

PITCH_GAINS = (-2.0, -0.3, -0.4)


def simulate_one_run(rng):
    payload_level = rng.uniform(0.0, 1.0)
    Va = rng.uniform(12.0, 20.0)
    gust_intensity = rng.uniform(0.0, 2.0)

    # nominal (payload=0) trim used as the ENTIRE run's reference -- never
    # leaks the true payload_level into the baseline.
    trim0 = find_trim(Va, payload_level=0.0)
    theta_cmd = trim0["theta"]
    nominal_elevator = trim0["elevator"]
    flight_hold = FlightHold(trim0["theta"], trim0["throttle"], Va)

    state = trim0["state"].copy()

    pitch_pid = PID(*PITCH_GAINS, limit=P.ELEVATOR_LIMIT)
    elevator_actuator = RateLimitedActuator(max_rate_deg_s=np.radians(90.0),
                                             initial_position=nominal_elevator)
    tail_actuator = RateLimitedActuator(max_rate_deg_s=np.radians(30.0),
                                        initial_position=0.0)

    from gust_model import DrydenGust
    gust = DrydenGust(Va, gust_intensity, seed=int(rng.integers(0, 2**31 - 1)))

    # Include fully active, transition, and no-change windows so the deployed
    # estimator is calibrated against gust-only data as well as payload data.
    split = rng.random()
    if split < 0.40:
        change_t_rel = 4.0       # explicit no-change / false-alarm examples
    elif split < 0.80:
        change_t_rel = rng.uniform(-3.0, 0.0)
    else:
        change_t_rel = rng.uniform(0.0, 3.0)

    # If change_t_rel < 0, we must ACTUALLY simulate that pre-roll period
    # first (real settling), not just relabel an untouched state.
    # Deployment may evaluate a window well after startup, so include a
    # randomized settled pre-roll. This prevents startup-only features from
    # being mistaken for a later payload event.
    # Deployment starts at a trimmed state, so include startup windows as
    # well as settled windows.  Otherwise the first live window is out of
    # distribution and can create a false payload estimate.
    preroll_t = 0.0 if rng.random() < 0.35 else rng.uniform(0.0, 10.0)
    total_sim_t = preroll_t + WINDOW_T
    # absolute time (from start of full simulation, including preroll) at
    # which payload becomes active:
    change_t_abs = preroll_t + change_t_rel

    n_total_steps = int(round(total_sim_t / DT))
    window_start_step = int(round(preroll_t / DT))

    log_q = []
    log_az = []
    log_elev_err = []  # raw pre-rate-limit commanded elevator - nominal trim elevator
    log_Va = []
    log_tail = []
    tail_phase = float(rng.uniform(0.0, 2.0 * np.pi))

    t = 0.0
    for step in range(n_total_steps):
        payload_active = payload_level if t >= change_t_abs else 0.0

        wind_vec = gust.step(DT)
        theta_cmd, throttle, _ = flight_hold.update(state, wind_vec, DT)
        theta = state[7]
        q = state[10]
        error = theta_cmd - theta
        pid_out = pitch_pid.update(error, DT)
        # The deployment estimator sees the controller's raw elevator
        # command, not an artificial training-only excitation signal.
        commanded_elevator_raw = pid_out + nominal_elevator
        commanded_elevator_raw = float(np.clip(commanded_elevator_raw,
                                                -P.ELEVATOR_LIMIT, P.ELEVATOR_LIMIT))

        # Fixed hardware actuator limit; the true payload label must not enter
        # the estimator/controller path.
        elevator_actuator.max_rate = np.radians(90.0)
        de = elevator_actuator.update(commanded_elevator_raw, DT)

        # Bounded causal tail motion is present in both training and
        # deployment; it is independent of the payload label.
        tail_target = tail_identification_command(t, tail_phase)
        tail_incidence = tail_actuator.update(tail_target, DT)

        controls = np.array([de, 0.0, 0.0, throttle])
        gust_vec = wind_vec

        state = rk4_step(state, controls, payload_active, DT, wind=gust_vec,
                         tail_incidence=tail_incidence)

        # az (vertical body-axis specific force, approx from w_dot term) --
        # compute via finite-difference-free direct force evaluation:
        forces, moments, alpha, beta, Va_now, Jy_eff, mass_eff, D = \
            dynamics_forces_helper(state, controls, payload_active, gust_vec,
                                   tail_incidence)
        az = forces[2] / mass_eff  # body-frame z specific force (down-positive)

        if step >= window_start_step:
            log_q.append(state[10])
            log_az.append(az)
            log_elev_err.append(commanded_elevator_raw - nominal_elevator)
            log_Va.append(Va_now)
            log_tail.append(tail_incidence)

        t += DT

    q_arr = np.array(log_q)
    az_arr = np.array(log_az)
    elev_arr = np.array(log_elev_err)
    va_arr = np.array(log_Va)
    tail_arr = np.array(log_tail)

    features = estimator_features(q_arr, az_arr, elev_arr, va_arr, tail_arr)

    # true payload level active AT WINDOW END:
    label = payload_level if (t - DT) >= change_t_abs else 0.0

    return features, label


def dynamics_forces_helper(state, controls, payload_level, wind, tail_incidence):
    from dynamics_6dof import aero_forces_moments
    forces, moments, alpha, beta, Va, Jy_eff, mass_eff, D = aero_forces_moments(
        state, controls, payload_level, wind=wind,
        tail_incidence=tail_incidence)
    return forces, moments, alpha, beta, Va, Jy_eff, mass_eff, D


def generate_dataset(n_runs=5000, seed=0, verbose_every=500):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_runs, ESTIMATOR_FEATURE_COUNT))
    y = np.zeros(n_runs)
    groups = np.arange(n_runs, dtype=int)
    for i in range(n_runs):
        feats, label = simulate_one_run(rng)
        X[i] = feats
        y[i] = label
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"generated {i + 1}/{n_runs} runs")
    return X, y, groups


if __name__ == "__main__":
    import sys
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    X, y, groups = generate_dataset(n_runs=n_runs)
    from pathlib import Path
    out_path = Path(__file__).resolve().with_name("payload_dataset.npz")
    np.savez(out_path, X=X, y=y, groups=groups)
    print("Saved payload_dataset.npz", X.shape, y.shape)
    print("Label stats: mean=%.3f std=%.3f min=%.3f max=%.3f" %
          (y.mean(), y.std(), y.min(), y.max()))
