"""Sensor, actuator, and safety-supervisor models for robustness studies.

The nominal experiments can keep their deterministic ideal-sensor path.  The
robustness path uses these models explicitly so that every non-ideal effect is
visible in the run configuration and status output.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

import aircraft_params as P


def _as_float(value: Any, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


@dataclass
class SensorConfig:
    """Simple IMU/airspeed/altitude sensor error model.

    Angles are radians, angular rates are rad/s, velocities are m/s, and
    altitude/position are metres.  Biases are sampled once per flight and
    white noise is sampled at the configured sensor rate.
    """

    enabled: bool = False
    delay_s: float = 0.0
    sample_period_s: float = 0.02
    angle_noise_deg: float = 0.0
    rate_noise_deg_s: float = 0.0
    velocity_noise_mps: float = 0.0
    position_noise_m: float = 0.0
    airspeed_noise_mps: float = 0.0
    acceleration_noise_mps2: float = 0.0
    angle_bias_deg: float = 0.0
    rate_bias_deg_s: float = 0.0
    velocity_bias_mps: float = 0.0
    position_bias_m: float = 0.0
    airspeed_bias_mps: float = 0.0
    acceleration_bias_mps2: float = 0.0
    seed: int = 0

    @classmethod
    def from_mapping(cls, mapping=None, seed=None):
        if isinstance(mapping, cls):
            cfg = cls(**mapping.__dict__)
        else:
            values = dict(mapping or {})
            allowed = set(cls.__dataclass_fields__)
            cfg = cls(**{k: v for k, v in values.items() if k in allowed})
        if seed is not None:
            cfg.seed = int(seed)
        cfg.delay_s = max(0.0, _as_float(cfg.delay_s, 0.0))
        cfg.sample_period_s = max(1e-6, _as_float(cfg.sample_period_s, 0.02))
        return cfg


class SensorSuite:
    """Causal delayed/noisy state, airspeed, and acceleration measurements."""

    def __init__(self, config=None, dt=0.02, initial_state=None, seed=None):
        self.config = SensorConfig.from_mapping(config, seed=seed)
        self.dt = max(float(dt), 1e-6)
        self.rng = np.random.default_rng(self.config.seed)
        self.delay_steps = int(round(self.config.delay_s / self.dt))
        self.sample_steps = max(1, int(round(self.config.sample_period_s / self.dt)))
        initial = np.zeros(12) if initial_state is None else np.asarray(initial_state, dtype=float)
        self._state_queue = deque([initial.copy()] * (self.delay_steps + 1),
                                  maxlen=self.delay_steps + 1)
        self._last_state = initial.copy()
        self._last_airspeed = 0.0
        self._last_accel = 0.0
        self._step_count = 0
        self._truth_accel = 0.0

        c = self.config
        self._state_noise = np.zeros(12)
        self._state_bias = np.zeros(12)
        self._state_noise[0:3] = c.position_noise_m
        self._state_noise[3:6] = c.velocity_noise_mps
        self._state_noise[6:9] = np.radians(c.angle_noise_deg)
        self._state_noise[9:12] = np.radians(c.rate_noise_deg_s)
        self._state_bias[0:3] = c.position_bias_m
        self._state_bias[3:6] = c.velocity_bias_mps
        self._state_bias[6:9] = np.radians(c.angle_bias_deg)
        self._state_bias[9:12] = np.radians(c.rate_bias_deg_s)

    def _sample_or_hold(self, state, airspeed, acceleration):
        if not self.config.enabled:
            return np.asarray(state, dtype=float).copy(), float(airspeed), float(acceleration)

        if self._step_count % self.sample_steps == 0:
            raw = np.asarray(state, dtype=float) + self._state_bias
            raw = raw + self.rng.normal(0.0, self._state_noise, size=12)
            self._state_queue.append(raw)
            if self.delay_steps:
                measured_state = self._state_queue[0].copy()
            else:
                measured_state = raw.copy()
            self._last_state = measured_state
            self._last_airspeed = (
                float(airspeed) + self.config.airspeed_bias_mps
                + self.rng.normal(0.0, self.config.airspeed_noise_mps)
            )
            self._last_accel = (
                float(acceleration) + self.config.acceleration_bias_mps2
                + self.rng.normal(0.0, self.config.acceleration_noise_mps2)
            )
        self._step_count += 1
        return self._last_state.copy(), self._last_airspeed, float(self._last_accel)

    def measure(self, state, airspeed, acceleration):
        """Return one causal sensor sample."""
        return self._sample_or_hold(state, airspeed, acceleration)

    def set_acceleration_truth(self, acceleration):
        self._truth_accel = float(acceleration)

    @property
    def truth_acceleration(self):
        return self._truth_accel


class ActuatorChannel:
    """First-order, rate-limited, bounded actuator with optional errors."""

    def __init__(self, max_rate, initial_position=0.0, minimum=-np.inf,
                 maximum=np.inf, time_constant=0.0, bias=0.0,
                 deadband=0.0, noise_std=0.0, seed=0):
        self.max_rate = max(0.0, float(max_rate))
        self.position = float(initial_position)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.time_constant = max(0.0, float(time_constant))
        self.bias = float(bias)
        self.deadband = max(0.0, float(deadband))
        self.noise_std = max(0.0, float(noise_std))
        self.rng = np.random.default_rng(seed)

    def update(self, command, dt):
        command = float(np.clip(command + self.bias, self.minimum, self.maximum))
        error = command - self.position
        if abs(error) <= self.deadband:
            target = self.position
        else:
            target = command
        if self.time_constant > 0.0:
            target = self.position + (float(dt) / self.time_constant) * (target - self.position)
        delta = np.clip(target - self.position,
                        -self.max_rate * float(dt), self.max_rate * float(dt))
        self.position = float(np.clip(self.position + delta,
                                      self.minimum, self.maximum))
        output = self.position
        if self.noise_std > 0.0:
            output += float(self.rng.normal(0.0, self.noise_std))
        return float(np.clip(output, self.minimum, self.maximum))

    def reset(self, position=0.0):
        self.position = float(np.clip(position, self.minimum, self.maximum))


def make_actuator(name, max_rate, initial_position, minimum, maximum,
                  config=None, seed=0):
    """Create an ideal or realistic actuator from a JSON-friendly mapping."""
    values = dict(config or {})
    if not values.get("enabled", False):
        from pid_controller import RateLimitedActuator
        return RateLimitedActuator(max_rate, initial_position)
    return ActuatorChannel(
        max_rate=max_rate,
        initial_position=initial_position,
        minimum=minimum,
        maximum=maximum,
        time_constant=_as_float(values.get("time_constant_s"), 0.0),
        bias=_as_float(values.get("bias"), 0.0),
        deadband=_as_float(values.get("deadband"), 0.0),
        noise_std=_as_float(values.get("noise_std"), 0.0),
        seed=int(values.get("seed", seed)),
    )


class SafetySupervisor:
    """Envelope monitor and fallback trigger for adaptive control.

    The supervisor does not certify an aircraft.  It detects excursions from
    the declared simulation envelope, latches a fallback request, and records
    the first reason so the run cannot hide a safety event.
    """

    def __init__(self, config=None):
        values = dict(config or {})
        self.enabled = bool(values.get("enabled", False))
        self.terminate_on_violation = bool(values.get("terminate_on_violation", False))
        self.theta_limit = np.radians(_as_float(values.get("theta_limit_deg"), 35.0))
        self.phi_limit = np.radians(_as_float(values.get("phi_limit_deg"), 50.0))
        self.alpha_limit = np.radians(_as_float(values.get("alpha_limit_deg"), 12.0))
        self.beta_limit = np.radians(_as_float(values.get("beta_limit_deg"), 12.0))
        self.alpha_protection_margin = np.radians(
            max(0.0, _as_float(values.get("alpha_protection_margin_deg"), 1.0)))
        self.alpha_protection_limit = max(
            0.0, self.alpha_limit - self.alpha_protection_margin)
        self.alpha_recovery_pitch = np.radians(
            max(0.0, _as_float(values.get("alpha_recovery_pitch_deg"), 2.0)))
        self.alpha_recovery_elevator = np.radians(
            max(0.0, _as_float(values.get("alpha_recovery_elevator_deg"), 5.0)))
        self.min_airspeed = _as_float(values.get("min_airspeed_mps"), P.Va_min_valid)
        self.max_airspeed = _as_float(values.get("max_airspeed_mps"), 25.0)
        self.max_estimator_step = max(0.0, _as_float(values.get("max_estimator_step"), 0.20))
        self.estimator_ood_fraction = float(np.clip(
            _as_float(values.get("estimator_ood_fraction"), 0.20), 0.0, 1.0))
        self.triggered = False
        self.fallback_active = False
        self.first_reason = None
        self.events = []
        self._previous_estimate = None

    def evaluate(self, state, airspeed, alpha=None, beta=None, estimate=None,
                 feature_vector=None, scaled_features=None, feature_bounds=None,
                 t=0.0):
        if not self.enabled:
            return True, None
        state = np.asarray(state, dtype=float)
        reasons = []
        if not np.all(np.isfinite(state)) or not np.isfinite(airspeed):
            reasons.append("non_finite_measurement")
        else:
            if abs(state[7]) > self.theta_limit:
                reasons.append("pitch_envelope")
            if abs(state[6]) > self.phi_limit:
                reasons.append("roll_envelope")
            if airspeed < self.min_airspeed or airspeed > self.max_airspeed:
                reasons.append("airspeed_envelope")
        if alpha is not None and abs(float(alpha)) > self.alpha_limit:
            reasons.append("angle_of_attack_envelope")
        if beta is not None and abs(float(beta)) > self.beta_limit:
            reasons.append("sideslip_envelope")
        if scaled_features is not None and not np.all(np.isfinite(scaled_features)):
            reasons.append("non_finite_estimator_features")
        if feature_vector is not None and feature_bounds is not None:
            raw = np.asarray(feature_vector, dtype=float)
            low, high = (np.asarray(feature_bounds[0], dtype=float),
                         np.asarray(feature_bounds[1], dtype=float))
            if (raw.shape != low.shape or raw.shape != high.shape or
                    not np.all(np.isfinite(raw))):
                reasons.append("non_finite_estimator_features")
            else:
                outside = (raw < low) | (raw > high)
                if float(np.mean(outside)) >= self.estimator_ood_fraction:
                    reasons.append("estimator_out_of_distribution")
        elif scaled_features is not None and np.max(np.abs(scaled_features)) > 5.0:
            # Backward-compatible gate for callers that do not provide raw
            # training-support bounds.
            reasons.append("estimator_out_of_distribution")
        if estimate is not None and self._previous_estimate is not None:
            if abs(float(estimate) - self._previous_estimate) > self.max_estimator_step:
                reasons.append("estimator_rate_limit")
        if estimate is not None:
            self._previous_estimate = float(estimate)

        if reasons:
            reason = "+".join(reasons)
            self.triggered = True
            self.fallback_active = True
            if self.first_reason is None:
                self.first_reason = reason
            self.events.append({"t": float(t), "reason": reason})
            return False, reason
        return True, None

    def enforce_tail(self, command, limit):
        return float(np.clip(command, -abs(limit), abs(limit)))

    def protect_elevator(self, command, alpha):
        """Remove elevator commands that drive further toward an alpha limit.

        With this model's sign convention positive elevator produces a
        nose-down moment.  The clamp therefore preserves nose-down authority
        at positive alpha and nose-up authority at negative alpha.  It is only
        active for an enabled supervisor and is intentionally conservative.
        """
        command = float(command)
        if not self.enabled or not np.isfinite(alpha):
            return command
        if alpha >= self.alpha_protection_limit:
            return max(command, self.alpha_recovery_elevator)
        if alpha <= -self.alpha_protection_limit:
            return min(command, -self.alpha_recovery_elevator)
        return command

    def protect_pitch_reference(self, reference, current_pitch, alpha):
        """Bias pitch demand toward recovery when alpha nears its limit."""
        reference = float(reference)
        current_pitch = float(current_pitch)
        if not self.enabled or not np.isfinite(alpha):
            return reference
        if alpha >= self.alpha_protection_limit:
            return min(reference, current_pitch - self.alpha_recovery_pitch)
        if alpha <= -self.alpha_protection_limit:
            return max(reference, current_pitch + self.alpha_recovery_pitch)
        return reference

    def enforce_controls(self, controls, limits):
        controls = np.asarray(controls, dtype=float).copy()
        return np.clip(controls, -np.asarray(limits), np.asarray(limits))

    def status(self):
        return {
            "enabled": self.enabled,
            "triggered": self.triggered,
            "fallback_active": self.fallback_active,
            "first_reason": self.first_reason,
            "alpha_protection_margin_deg": float(
                np.degrees(self.alpha_protection_margin)),
            "alpha_recovery_pitch_deg": float(
                np.degrees(self.alpha_recovery_pitch)),
            "alpha_recovery_elevator_deg": float(
                np.degrees(self.alpha_recovery_elevator)),
            "estimator_ood_fraction": self.estimator_ood_fraction,
            "event_count": len(self.events),
            "events": self.events[:20],
            "terminate_on_violation": self.terminate_on_violation,
        }
