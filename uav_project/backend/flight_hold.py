"""Simple common airspeed and altitude hold used by both experiments."""
import numpy as np

import aircraft_params as P
from dynamics_6dof import _dcm_body_to_ned
from pid_controller import PID, RateLimitedActuator


class FlightHold:
    """Sensor-based outer loops; neither loop uses the true payload label."""

    def __init__(self, theta_trim, throttle_trim, Va_ref, altitude_ref=None):
        self.theta_trim = float(theta_trim)
        self.throttle_trim = float(throttle_trim)
        self.Va_ref = float(Va_ref)
        self.altitude_ref = P.alt_trim if altitude_ref is None else float(altitude_ref)
        # The derivative term damps climb/descent transients; without it the
        # payload step can create a noticeable altitude excursion before the
        # pitch loop catches up.
        self.alt_pid = PID(0.012, 0.0008, 0.025, np.radians(10.0))
        self.speed_pid = PID(0.03, 0.005, 0.0, 0.60)
        self.throttle_act = RateLimitedActuator(0.50, self.throttle_trim)

    @staticmethod
    def airspeed(state, wind_ned):
        wind_body = _dcm_body_to_ned(state[6], state[7], state[8]).T @ np.asarray(wind_ned)
        return float(np.linalg.norm(np.asarray(state[3:6]) - wind_body))

    def update(self, state, wind_ned, dt, measurements=None):
        """Update the outer loops using truth or a sensor measurement.

        ``measurements`` is optional to preserve the original ideal-sensor
        behavior.  Robustness studies pass ``altitude_m`` and
        ``airspeed_mps`` from :class:`realism.SensorSuite`.
        """
        measurements = measurements or {}
        altitude = float(measurements.get("altitude_m", -float(state[2])))
        altitude_error = self.altitude_ref - altitude
        theta_ref = self.theta_trim + self.alt_pid.update(altitude_error, dt)
        va = float(measurements.get("airspeed_mps", self.airspeed(state, wind_ned)))
        throttle_cmd = np.clip(
            self.throttle_trim + self.speed_pid.update(self.Va_ref - va, dt),
            0.0, 1.0,
        )
        throttle = self.throttle_act.update(float(throttle_cmd), dt)
        return float(theta_ref), float(throttle), va
