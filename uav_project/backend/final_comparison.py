"""
final_comparison.py -- the MAIN ENGINE.

run_conventional(): fixed-gain PID, no payload compensation, no morphing
tail. run_intelligent(): same base controller architecture, plus a
runtime NN payload estimator, morphing-tail trim compensation, and
gain-scheduled PID gains.

Both integrate the SAME 6-DOF dynamics (dynamics_6dof.rk4_step) under
IDENTICAL disturbances for a given seed, so the two runs are a fair,
apples-to-apples comparison.
"""
import numpy as np
import joblib
from pathlib import Path

import aircraft_params as P
from dynamics_6dof import rk4_step, aero_forces_moments, _dcm_body_to_ned
from trim import find_trim
from pid_controller import PID, RateLimitedActuator
from gust_model import DrydenGust
from flight_hold import FlightHold
from adaptive_system import (morphing_tail_command, adaptive_gains,
                              adaptive_lateral_gains, estimator_features, elevator_trim_delta,
                              estimate_turbulence, tail_identification_command,
                              ESTIMATOR_FEATURE_COUNT)
from realism import SensorSuite, SafetySupervisor, make_actuator

DT = 0.02
EULER_VALID_LIMIT = np.radians(85.0)
AERO_VALID_ALPHA_LIMIT = np.radians(15.0)
AERO_VALID_BETA_LIMIT = np.radians(15.0)

PITCH_BASE = (-2.0, -0.3, -0.4)
ROLL_BASE = (3.0, 0.1, 0.6)
YAW_BASE = (2.5, 0.3, 0.5)
ELEVATOR_RATE_LIMIT = np.radians(90.0)  # fixed physical actuator limit
PITCH_GAIN_RATE_LIMIT = np.array([0.25, 0.03, 0.02])  # gain-units/s
DEFAULT_GUST_MODEL = "ou"  # legacy evidence; robustness studies use "dryden"

_ESTIMATOR = None
_ESTIMATOR_ERROR = None
_BASE_DIR = Path(__file__).resolve().parent


def _load_estimator():
    global _ESTIMATOR, _ESTIMATOR_ERROR
    if _ESTIMATOR is None:
        try:
            bundle = joblib.load(_BASE_DIR / "payload_estimator.joblib")
            # Never run an old artifact with a different feature schema.
            if (bundle.get("feature_count") != ESTIMATOR_FEATURE_COUNT or
                    getattr(bundle.get("model"), "n_features_in_", None)
                    != ESTIMATOR_FEATURE_COUNT):
                _ESTIMATOR = None
                _ESTIMATOR_ERROR = (
                    "payload_estimator.joblib is incompatible with the current "
                    f"{ESTIMATOR_FEATURE_COUNT}-feature schema"
                )
            else:
                _ESTIMATOR = bundle
                _ESTIMATOR_ERROR = None
        except FileNotFoundError:
            _ESTIMATOR = None
            _ESTIMATOR_ERROR = "payload_estimator.joblib is missing"
        except Exception as exc:
            _ESTIMATOR = None
            _ESTIMATOR_ERROR = f"payload estimator could not be loaded: {exc}"
    return _ESTIMATOR


def _verify_yaw_sign():
    """Apply a small constant test rudder and confirm which way psi moves,
    so the heading-hold error sign can be set correctly (avoids spiral
    divergence from a flipped sign)."""
    trim0 = find_trim(P.Va_trim, payload_level=0.0)
    state = trim0["state"].copy()
    controls = np.array([trim0["elevator"], 0.0, np.radians(5.0), trim0["throttle"]])
    psi0 = state[8]
    for _ in range(50):  # 1 second
        state = rk4_step(state, controls, 0.0, DT)
    dpsi = state[8] - psi0
    # dr > 0 with our convention (CY_dr>0, Cn_dr<0) yaws the nose in the
    # -psi direction (nose left) -- record the sign so heading control uses
    # the correct feedback polarity.
    return np.sign(dpsi) if dpsi != 0 else -1.0


_YAW_SIGN = None


def _get_yaw_sign():
    global _YAW_SIGN
    if _YAW_SIGN is None:
        _YAW_SIGN = _verify_yaw_sign()
    return _YAW_SIGN


def _lateral_disturbance_wind(t, change_t):
    if t < change_t:
        return np.array([0.0, 0.0, 0.0])
    v_gust = 1.0 * np.sin(2 * np.pi * (0.3 / (2 * np.pi)) * (t - change_t))
    return np.array([0.0, v_gust, 0.0])


def _init_run(Va_init, seed):
    if not np.isfinite(Va_init) or Va_init < P.Va_min_valid:
        raise ValueError(f"airspeed must be >= {P.Va_min_valid:.1f} m/s for the validated model")
    trim0 = find_trim(Va_init, payload_level=0.0)
    if not trim0["converged"]:
        raise ValueError(f"nominal trim failed at airspeed {Va_init:.3f} m/s")
    state = trim0["state"].copy()
    return trim0, state


def _metrics_from_log(log, change_t, theta_ref_deg):
    log = np.asarray(log)
    if log.size == 0:
        return {"max_pitch_dev": float("nan"), "max_roll_dev": float("nan"),
                "mean_elev": float("nan"), "mean_aileron": float("nan"),
                "mean_rudder": float("nan"), "mean_drag": float("nan")}
    t = log[:, 0]
    mask = t >= change_t
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)
    # New logs carry the actual (altitude-hold) pitch command in the final
    # column. Keep the argument fallback for old/test logs.
    theta_ref = log[mask, -1] if log.shape[1] >= 13 else theta_ref_deg
    theta = log[mask, 1] - theta_ref
    phi = log[mask, 2]
    psi = log[mask, 3]
    de = log[mask, 4]
    da = log[mask, 5]
    dr = log[mask, 6]
    drag = log[mask, 8]
    tt = t[mask] - t[mask][0]
    dt = float(np.median(np.diff(t[mask]))) if len(tt) > 1 else DT
    tail = log[mask, 9] if log.shape[1] >= 15 else np.zeros_like(de)
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))
    def itae(x):
        return float(np.trapezoid(tt * np.abs(x), tt)) if len(tt) > 1 else 0.0
    def activity(x):
        return float(np.sum(np.abs(np.diff(x))) / max(dt, 1e-9)) if len(x) > 1 else 0.0
    def sat_time(x, limit_deg):
        return float(np.sum(np.abs(x) >= 0.999 * limit_deg) * dt)
    altitude = log[mask, 7]
    altitude_error = altitude - P.alt_trim
    # Transient / stability metrics for the response after the payload step.
    # settling band = max(0.1 deg, 5 % of the peak error); "settled" is 0 when
    # the error is still outside the band at the end of the window (always the
    # case under continuing gusts, so interpret these at gust = 0).
    peak = float(np.max(np.abs(theta)))
    band = max(0.1, 0.05 * peak)
    outside = np.where(np.abs(theta) > band)[0]
    settled = bool(len(outside) == 0 or outside[-1] < len(theta) - 1)
    settling_time = float(tt[outside[-1]]) if len(outside) else 0.0
    tail_n = max(1, len(theta) // 5)
    return {
        "pitch_settling_time_s": settling_time,
        "pitch_settled": 1.0 if settled else 0.0,
        "pitch_steady_state_error": float(np.mean(theta[-tail_n:])),
        "max_pitch_dev": float(np.max(np.abs(theta))),
        "max_roll_dev": float(np.max(np.abs(phi))),
        "max_yaw_dev": float(np.max(np.abs(psi))),
        "rms_pitch_error": rms(theta),
        "rms_roll_error": rms(phi),
        "rms_yaw_error": rms(psi),
        "itae_pitch": itae(theta),
        "itae_roll": itae(phi),
        "itae_yaw": itae(psi),
        "mean_elev": float(np.mean(np.abs(de))),
        "mean_aileron": float(np.mean(np.abs(da))),
        "mean_rudder": float(np.mean(np.abs(dr))),
        "mean_signed_elev": float(np.mean(de)),
        "mean_signed_aileron": float(np.mean(da)),
        "mean_signed_rudder": float(np.mean(dr)),
        "rms_elev": rms(de),
        "rms_aileron": rms(da),
        "rms_rudder": rms(dr),
        "elev_activity": activity(de),
        "aileron_activity": activity(da),
        "rudder_activity": activity(dr),
        "elev_saturation_s": sat_time(de, np.degrees(P.ELEVATOR_LIMIT)),
        "aileron_saturation_s": sat_time(da, np.degrees(P.AILERON_LIMIT)),
        "rudder_saturation_s": sat_time(dr, np.degrees(P.RUDDER_LIMIT)),
        "rms_altitude_error": rms(altitude_error),
        "max_altitude_drop_m": float(max(0.0, altitude[0] - np.min(altitude))),
        "tail_mean_abs": float(np.mean(np.abs(tail))),
        "tail_rms": rms(tail),
        "tail_activity": activity(tail),
        "mean_drag": float(np.mean(drag)),
    }


def _airspeed_from_state_wind(state, wind_ned):
    wind_body = _dcm_body_to_ned(state[6], state[7], state[8]).T @ np.asarray(wind_ned)
    return float(np.linalg.norm(np.asarray(state[3:6]) - wind_body))


def _sensor_sample(sensor, state, wind, previous_accel):
    true_va = _airspeed_from_state_wind(state, wind)
    measured_state, measured_va, measured_accel = sensor.measure(
        state, true_va, previous_accel)
    measurements = {
        "state": measured_state,
        "airspeed_mps": measured_va,
        "altitude_m": -float(measured_state[2]),
        "acceleration_z_mps2": measured_accel,
    }
    return measurements


def _status_with_realism(status, gust_model, sensor, supervisor):
    status["gust_model"] = str(gust_model)
    status["sensor_realism"] = bool(sensor.config.enabled)
    status["sensor_config"] = dict(sensor.config.__dict__)
    status["safety_supervisor"] = supervisor.status()


def run_conventional(payload_level, Va_init, gust_intensity, seed, T=40.0,
                      change_t=10.0, lateral_disturbance=False,
                      gust_model=DEFAULT_GUST_MODEL, sensor_config=None,
                      actuator_config=None, safety_config=None):
    if float(T) < DT:
        raise ValueError(f"T must be >= integration step {DT:.3f} s")
    trim0, state = _init_run(Va_init, seed)
    theta_cmd = trim0["theta"]
    nominal_elevator = trim0["elevator"]
    throttle = trim0["throttle"]
    flight_hold = FlightHold(theta_cmd, throttle, Va_init)
    yaw_sign = _get_yaw_sign()

    pitch_pid = PID(*PITCH_BASE, limit=P.ELEVATOR_LIMIT)
    roll_pid = PID(*ROLL_BASE, limit=P.AILERON_LIMIT)
    yaw_pid = PID(*YAW_BASE, limit=P.RUDDER_LIMIT)

    actuator_config = actuator_config or {}
    elevator_act = make_actuator(
        "elevator", np.radians(90.0), nominal_elevator,
        -P.ELEVATOR_LIMIT, P.ELEVATOR_LIMIT,
        actuator_config.get("elevator"), seed + 101)
    aileron_act = make_actuator(
        "aileron", np.radians(120.0), 0.0,
        -P.AILERON_LIMIT, P.AILERON_LIMIT,
        actuator_config.get("aileron"), seed + 102)
    rudder_act = make_actuator(
        "rudder", np.radians(120.0), 0.0,
        -P.RUDDER_LIMIT, P.RUDDER_LIMIT,
        actuator_config.get("rudder"), seed + 103)

    gust = DrydenGust(Va_init, gust_intensity, seed=seed, model=gust_model)
    sensor = SensorSuite(sensor_config, DT, initial_state=state, seed=seed + 1001)
    supervisor = SafetySupervisor(safety_config)

    n_steps = int(round(T / DT))
    log = []

    theta_ref = theta_cmd
    phi_ref = 0.0
    psi_ref = 0.0

    t = 0.0
    status = {"valid": True, "terminated_reason": None, "simulated_T": 0.0,
              "lateral_disturbance": bool(lateral_disturbance),
              "random_seed": int(seed), "payload_level": float(payload_level),
              "airspeed_init": float(Va_init),
              "gust_intensity": float(gust_intensity),
              "change_t": float(change_t), "requested_T": float(T),
              "max_abs_alpha_deg": 0.0, "max_abs_beta_deg": 0.0,
              "min_airspeed_mps": float("inf"),
              "model_version": "uav-6dof-2026-10",
              "airframe_regime": ("main_stable_margin"
                                   if payload_level <= P.Main_results_payload_max
                                   else "unstable_airframe_stress")}
    previous_accel = 0.0
    for step in range(n_steps):
        payload_active = payload_level if t >= change_t else 0.0
        wind = gust.step(DT)
        if lateral_disturbance:
            wind = wind + _lateral_disturbance_wind(t, change_t)

        measurements = _sensor_sample(sensor, state, wind, previous_accel)
        measured_state = measurements["state"]
        theta_ref, throttle, Va_now = flight_hold.update(
            measured_state, wind, DT, measurements=measurements)
        theta, phi, psi = measured_state[7], measured_state[6], measured_state[8]
        _, _, alpha_pre, beta_pre, _, _, _, _ = aero_forces_moments(
            measured_state, np.zeros(4), payload_active, wind=wind,
            tail_incidence=0.0)
        supervisor.evaluate(measured_state, Va_now, alpha=alpha_pre,
                            beta=beta_pre, t=t)
        theta_ref = supervisor.protect_pitch_reference(
            theta_ref, theta, alpha_pre)

        pitch_err = theta_ref - theta
        de_cmd = pitch_pid.update(pitch_err, DT) + nominal_elevator
        de_cmd = supervisor.protect_elevator(de_cmd, alpha_pre)
        de_cmd = float(np.clip(de_cmd, -P.ELEVATOR_LIMIT, P.ELEVATOR_LIMIT))

        roll_err = phi_ref - phi
        da_cmd = roll_pid.update(roll_err, DT)
        da_cmd = float(np.clip(da_cmd, -P.AILERON_LIMIT, P.AILERON_LIMIT))

        # heading-hold yaw controller (verified sign)
        yaw_err = yaw_sign * (psi_ref - psi)
        dr_cmd = yaw_pid.update(yaw_err, DT)
        dr_cmd = float(np.clip(dr_cmd, -P.RUDDER_LIMIT, P.RUDDER_LIMIT))

        elevator_act.max_rate = ELEVATOR_RATE_LIMIT
        de = elevator_act.update(de_cmd, DT)
        da = aileron_act.update(da_cmd, DT)
        dr = rudder_act.update(dr_cmd, DT)

        controls = np.array([de, da, dr, throttle])
        controls[:3] = supervisor.enforce_controls(
            controls[:3], [P.ELEVATOR_LIMIT, P.AILERON_LIMIT, P.RUDDER_LIMIT])

        state = rk4_step(state, controls, payload_active, DT, wind=wind,
                          tail_incidence=0.0)

        forces, moments, alpha, beta, Va_now, Jy_eff, mass_eff, D = \
            aero_forces_moments(state, controls, payload_active, wind=wind,
                                tail_incidence=0.0)
        status["max_abs_alpha_deg"] = max(
            status["max_abs_alpha_deg"], abs(float(np.degrees(alpha))))
        status["max_abs_beta_deg"] = max(
            status["max_abs_beta_deg"], abs(float(np.degrees(beta))))
        status["min_airspeed_mps"] = min(status["min_airspeed_mps"], float(Va_now))
        if (not np.all(np.isfinite(state)) or
                abs(state[7]) >= EULER_VALID_LIMIT):
            status.update({"valid": False, "terminated_reason": "euler_validity_limit",
                           "simulated_T": float(t)})
            break
        if abs(alpha) >= AERO_VALID_ALPHA_LIMIT or abs(beta) >= AERO_VALID_BETA_LIMIT:
            status.update({"valid": False, "terminated_reason": "aerodynamic_validity_limit",
                           "validity_detail": {"alpha_deg": float(np.degrees(alpha)),
                                               "beta_deg": float(np.degrees(beta)),
                                               "airspeed_mps": float(Va_now)},
                           "simulated_T": float(t)})
            break
        previous_accel = forces[2] / mass_eff
        safe, reason = supervisor.evaluate(
            measured_state, Va_now, alpha=alpha, beta=beta, t=t)
        if not safe and supervisor.terminate_on_violation:
            status.update({"valid": False,
                           "terminated_reason": "safety_supervisor:" + str(reason),
                           "simulated_T": float(t)})
            break

        log.append([t + DT, np.degrees(state[7]), np.degrees(state[6]),
                    np.degrees(state[8]), np.degrees(de), np.degrees(da),
                    np.degrees(dr), -state[2], D,
                    state[0], state[1], state[2], np.degrees(theta_ref)])
        t += DT

    status["simulated_T"] = float(t)
    _status_with_realism(status, gust_model, sensor, supervisor)
    metrics = _metrics_from_log(log, change_t, np.degrees(theta_ref))
    log_arr = np.array(log)
    return metrics, log_arr, status


def run_intelligent(payload_level, Va_init, gust_intensity, seed, T=40.0,
                     change_t=10.0, lateral_disturbance=False, debug=False,
                     payload_source="nn", tail_enabled=True,
                     gain_schedule=True, tail_dither=True, pitch_gain_schedule=False,
                     trim_mode=None, gust_model=DEFAULT_GUST_MODEL,
                     sensor_config=None, actuator_config=None,
                     safety_config=None):
    if float(T) < DT:
        raise ValueError(f"T must be >= integration step {DT:.3f} s")
    if trim_mode is None:
        trim_mode = "tail" if tail_enabled else "none"
    if trim_mode not in ("tail", "elevator_ff", "none"):
        raise ValueError("trim_mode must be 'tail', 'elevator_ff' or 'none'")
    if payload_source not in ("nn", "oracle", "off"):
        raise ValueError("payload_source must be 'nn', 'oracle' or 'off'")
    trim0, state = _init_run(Va_init, seed)
    theta_cmd = trim0["theta"]
    nominal_elevator = trim0["elevator"]
    throttle = trim0["throttle"]
    flight_hold = FlightHold(theta_cmd, throttle, Va_init)
    yaw_sign = _get_yaw_sign()
    est_bundle = _load_estimator()

    # Derivative filtering is DISABLED (tau=0): tau>=0.02 s destabilised pitch at 20 m/s (diagnosed Oct 2026);
    # this suppresses sensor/gain-schedule chatter without changing the
    # conventional baseline definition.
    pitch_pid = PID(*PITCH_BASE, limit=P.ELEVATOR_LIMIT,
                    derivative_tau=0.0)
    roll_pid = PID(*ROLL_BASE, limit=P.AILERON_LIMIT)
    yaw_pid = PID(*YAW_BASE, limit=P.RUDDER_LIMIT)

    actuator_config = actuator_config or {}
    elevator_act = make_actuator(
        "elevator", np.radians(90.0), nominal_elevator,
        -P.ELEVATOR_LIMIT, P.ELEVATOR_LIMIT,
        actuator_config.get("elevator"), seed + 101)
    aileron_act = make_actuator(
        "aileron", np.radians(120.0), 0.0,
        -P.AILERON_LIMIT, P.AILERON_LIMIT,
        actuator_config.get("aileron"), seed + 102)
    rudder_act = make_actuator(
        "rudder", np.radians(120.0), 0.0,
        -P.RUDDER_LIMIT, P.RUDDER_LIMIT,
        actuator_config.get("rudder"), seed + 103)
    tail_act = make_actuator(
        "tail", np.radians(30.0), 0.0,
        -np.radians(15.0), np.radians(15.0),
        actuator_config.get("tail"), seed + 104)

    gust = DrydenGust(Va_init, gust_intensity, seed=seed, model=gust_model)
    sensor = SensorSuite(sensor_config, DT, initial_state=state, seed=seed + 1001)
    supervisor = SafetySupervisor(safety_config)

    n_steps = int(round(T / DT))
    log = []

    theta_ref = theta_cmd
    phi_ref = 0.0
    psi_ref = 0.0

    buf_q, buf_az, buf_elev, buf_va, buf_tail = [], [], [], [], []
    BUF_LEN = 150

    pl_est = 0.0
    turb_est = 0.0
    applied_pitch_gains = np.asarray(PITCH_BASE, dtype=float)
    last_update_step = -10_000
    estimator_update_period = 25  # 0.5 s, after the first complete window
    estimator_smoothing = 0.15

    log_pl_est_t = []
    debug_raw_estimates = []
    debug_turbulence = []
    debug_feature_rows = []

    t = 0.0
    status = {"valid": True, "terminated_reason": None, "simulated_T": 0.0,
              "lateral_disturbance": bool(lateral_disturbance),
              "random_seed": int(seed), "payload_level": float(payload_level),
              "airspeed_init": float(Va_init),
              "gust_intensity": float(gust_intensity),
              "change_t": float(change_t), "requested_T": float(T),
              "max_abs_alpha_deg": 0.0, "max_abs_beta_deg": 0.0,
              "min_airspeed_mps": float("inf"),
              "model_version": "uav-6dof-2026-10",
              "airframe_regime": ("main_stable_margin"
                                   if payload_level <= P.Main_results_payload_max
                                   else "unstable_airframe_stress"),
              "estimator_available": est_bundle is not None,
              "estimator_error": _ESTIMATOR_ERROR,
              "payload_source": payload_source,
              "tail_enabled": bool(tail_enabled),
              "tail_dither": bool(tail_dither),
              "trim_mode": trim_mode,
              "gain_schedule": bool(gain_schedule),
              "fallback_mode": "conventional"}
    previous_accel = 0.0
    for step in range(n_steps):
        payload_active = payload_level if t >= change_t else 0.0
        wind = gust.step(DT)
        if lateral_disturbance:
            wind = wind + _lateral_disturbance_wind(t, change_t)

        measurements = _sensor_sample(sensor, state, wind, previous_accel)
        measured_state = measurements["state"]
        theta_ref, throttle, Va_now = flight_hold.update(
            measured_state, wind, DT, measurements=measurements)
        theta, phi, psi = measured_state[7], measured_state[6], measured_state[8]
        _, _, alpha_pre, beta_pre, _, _, _, _ = aero_forces_moments(
            measured_state, np.zeros(4), payload_active, wind=wind,
            tail_incidence=0.0)
        supervisor.evaluate(measured_state, Va_now, alpha=alpha_pre,
                            beta=beta_pre, t=t)
        theta_ref = supervisor.protect_pitch_reference(
            theta_ref, theta, alpha_pre)
        if payload_source == "oracle":
            pl_est = float(payload_active)
        elif payload_source == "off":
            pl_est = 0.0
        do_update = (len(buf_q) >= BUF_LEN and
                     payload_source == "nn" and
                     step - last_update_step >= estimator_update_period)

        # Continuous sliding-window estimation handles partial payloads and
        # payload changes hidden inside a window; no fragile onset detector or
        # one-shot lock is used.
        scaled_features = None
        raw_feature_vector = None
        feature_bounds = None
        if do_update:
            qa = np.array(buf_q[-BUF_LEN:])
            aza = np.array(buf_az[-BUF_LEN:])
            elva = np.array(buf_elev[-BUF_LEN:])
            vaa = np.array(buf_va[-BUF_LEN:])
            taila = np.array(buf_tail[-BUF_LEN:])

            # Features are recorded whenever a window is complete, even with no
            # estimator loaded (needed to collect stage-0 training windows).
            feats = estimator_features(qa, aza, elva, vaa, taila)[None, :]
            if debug:
                debug_feature_rows.append((float(t), feats[0].tolist()))
            if est_bundle is not None and len(qa) >= BUF_LEN:
                raw_feature_vector = feats[0].copy()
                if "clip_low" in est_bundle and "clip_high" in est_bundle:
                    feature_bounds = (est_bundle["clip_low"], est_bundle["clip_high"])
                    feats = np.clip(feats, est_bundle["clip_low"],
                                    est_bundle["clip_high"])
                feats_s = est_bundle["scaler"].transform(feats)
                scaled_features = feats_s[0]
                new_est = float(np.clip(est_bundle["model"].predict(feats_s)[0], 0.0, 1.0))
            else:
                new_est = pl_est

            if debug:
                debug_raw_estimates.append((t, float(new_est)))
            # Deadband removes tiny no-payload predictions while retaining
            # observability of the 0.2-level partial-payload case.
            target_est = 0.0 if new_est < 0.10 else new_est
            pl_est = float(np.clip(
                (1.0 - estimator_smoothing) * pl_est
                + estimator_smoothing * target_est, 0.0, 1.0))
            last_update_step = step
            log_pl_est_t.append((t, pl_est))

        supervisor.evaluate(
            measured_state, Va_now, estimate=pl_est,
            scaled_features=scaled_features,
            feature_vector=raw_feature_vector,
            feature_bounds=feature_bounds, t=t)
        fallback_active = supervisor.fallback_active
        control_payload = 0.0 if fallback_active else pl_est

        if buf_q:
            turb_est = estimate_turbulence(buf_q, buf_az)
        if debug:
            debug_turbulence.append((float(t), float(turb_est)))

        if gain_schedule and pitch_gain_schedule and not fallback_active:
            target_pitch_gains = np.asarray(
                adaptive_gains(control_payload, turb_est, base=PITCH_BASE), dtype=float)
        else:
            target_pitch_gains = np.asarray(PITCH_BASE, dtype=float)
        # Prevent a noisy estimate from changing the controller gains
        # instantaneously. The limits are gain-units per second.
        max_gain_step = PITCH_GAIN_RATE_LIMIT * DT
        applied_pitch_gains += np.clip(
            target_pitch_gains - applied_pitch_gains,
            -max_gain_step, max_gain_step)
        kp, ki, kd = applied_pitch_gains.tolist()
        pitch_pid.kp, pitch_pid.ki, pitch_pid.kd = kp, ki, kd
        if gain_schedule and not fallback_active:
            (rkp, rki, rkd), (ykp, yki, ykd) = adaptive_lateral_gains(
                turb_est, roll_base=ROLL_BASE, yaw_base=YAW_BASE)
        else:
            rkp, rki, rkd = ROLL_BASE
            ykp, yki, ykd = YAW_BASE
        roll_pid.kp, roll_pid.ki, roll_pid.kd = rkp, rki, rkd
        yaw_pid.kp, yaw_pid.ki, yaw_pid.kd = ykp, yki, ykd

        dither = (tail_identification_command(t)
                  if tail_dither and not fallback_active else 0.0)
        # The identification dither is part of the estimator's input
        # distribution, so it is kept whatever trim_mode is selected.
        morph = (morphing_tail_command(control_payload, Va_now)
                 if trim_mode == "tail" and not fallback_active else 0.0)
        tail_cmd = morph + dither
        tail_cmd = supervisor.enforce_tail(tail_cmd, np.radians(15.0))
        tail_inc = tail_act.update(tail_cmd, DT)

        pitch_err = theta_ref - theta
        de_cmd_raw = pitch_pid.update(pitch_err, DT) + nominal_elevator
        if trim_mode == "elevator_ff" and not fallback_active:
            # Baseline for the morphing tail: same payload estimate, but the
            # trim shift is applied with the ordinary elevator instead.
            de_cmd_raw += elevator_trim_delta(pl_est, Va_now)
        de_cmd_raw = supervisor.protect_elevator(de_cmd_raw, alpha_pre)
        de_cmd = float(np.clip(de_cmd_raw, -P.ELEVATOR_LIMIT, P.ELEVATOR_LIMIT))

        roll_err = phi_ref - phi
        da_cmd = float(np.clip(roll_pid.update(roll_err, DT), -P.AILERON_LIMIT, P.AILERON_LIMIT))

        yaw_err = yaw_sign * (psi_ref - psi)
        dr_cmd = float(np.clip(yaw_pid.update(yaw_err, DT), -P.RUDDER_LIMIT, P.RUDDER_LIMIT))

        elevator_act.max_rate = ELEVATOR_RATE_LIMIT
        de = elevator_act.update(de_cmd, DT)
        da = aileron_act.update(da_cmd, DT)
        dr = rudder_act.update(dr_cmd, DT)

        controls = np.array([de, da, dr, throttle])
        controls[:3] = supervisor.enforce_controls(
            controls[:3], [P.ELEVATOR_LIMIT, P.AILERON_LIMIT, P.RUDDER_LIMIT])

        state = rk4_step(state, controls, payload_active, DT, wind=wind,
                          tail_incidence=tail_inc)

        forces, moments, alpha, beta, Va_now, Jy_eff, mass_eff, D = \
            aero_forces_moments(state, controls, payload_active, wind=wind,
                                tail_incidence=tail_inc)
        status["max_abs_alpha_deg"] = max(
            status["max_abs_alpha_deg"], abs(float(np.degrees(alpha))))
        status["max_abs_beta_deg"] = max(
            status["max_abs_beta_deg"], abs(float(np.degrees(beta))))
        status["min_airspeed_mps"] = min(status["min_airspeed_mps"], float(Va_now))
        if (not np.all(np.isfinite(state)) or
                abs(state[7]) >= EULER_VALID_LIMIT):
            status.update({"valid": False, "terminated_reason": "euler_validity_limit",
                           "simulated_T": float(t)})
            break
        if abs(alpha) >= AERO_VALID_ALPHA_LIMIT or abs(beta) >= AERO_VALID_BETA_LIMIT:
            status.update({"valid": False, "terminated_reason": "aerodynamic_validity_limit",
                           "validity_detail": {"alpha_deg": float(np.degrees(alpha)),
                                               "beta_deg": float(np.degrees(beta)),
                                               "airspeed_mps": float(Va_now)},
                           "simulated_T": float(t)})
            break
        az = forces[2] / mass_eff
        previous_accel = az
        safe, reason = supervisor.evaluate(
            measured_state, Va_now, alpha=alpha, beta=beta,
            estimate=pl_est, t=t)
        if (not safe and supervisor.terminate_on_violation):
            status.update({"valid": False,
                           "terminated_reason": "safety_supervisor:" + str(reason),
                           "simulated_T": float(t)})
            break

        # causal buffer: post-step state paired with the raw pre-rate-limit
        # commanded elevator that caused it (matches training construction)
        # Match training: use the post-step measured pitch rate.
        if sensor.config.enabled:
            buf_q.append(measured_state[10])
            buf_az.append(measurements["acceleration_z_mps2"])
            buf_va.append(measurements["airspeed_mps"])
        else:
            # Preserve the original nominal-training/evaluation pairing when
            # ideal sensors are selected: the post-step truth measurement is
            # the label used by the original estimator dataset.
            buf_q.append(state[10])
            buf_az.append(az)
            buf_va.append(Va_now)
        buf_elev.append(de_cmd_raw - nominal_elevator)
        buf_tail.append(tail_inc)
        if len(buf_q) > BUF_LEN:
            buf_q.pop(0); buf_az.pop(0); buf_elev.pop(0); buf_va.pop(0); buf_tail.pop(0)

        log.append([t + DT, np.degrees(state[7]), np.degrees(state[6]),
                    np.degrees(state[8]), np.degrees(de), np.degrees(da),
                    np.degrees(dr), -state[2], D, np.degrees(tail_inc), pl_est,
                    state[0], state[1], state[2], np.degrees(theta_ref)])
        t += DT

    status["simulated_T"] = float(t)
    _status_with_realism(status, gust_model, sensor, supervisor)
    status["fallback_mode"] = "conventional"
    if debug:
        status["raw_estimates"] = debug_raw_estimates
        status["turbulence_estimates"] = debug_turbulence
        status["estimator_features"] = debug_feature_rows
    metrics = _metrics_from_log(log, change_t, np.degrees(theta_ref))
    log_arr = np.array(log)
    return metrics, log_arr, log_pl_est_t, status


if __name__ == "__main__":
    print("=== Severe scenario: payload=1.0, Va=13, gust=2.0 ===")
    conv_metrics, conv_log, conv_status = run_conventional(1.0, 13.0, 2.0, seed=42)
    int_metrics, int_log, pl_est_t, int_status = run_intelligent(1.0, 13.0, 2.0, seed=42)

    print("\nConventional:")
    for k, v in conv_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nIntelligent:")
    for k, v in int_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nImprovement (positive = intelligent better):")
    for k in conv_metrics:
        c, i = conv_metrics[k], int_metrics[k]
        if c != 0:
            pct = 100.0 * (c - i) / abs(c)
            print(f"  {k}: {pct:+.1f}%")
    print("\nValidity:")
    print("  conventional:", conv_status)
    print("  intelligent:", int_status)
