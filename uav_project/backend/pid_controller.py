"""
Standard PID controller with integral clipping and output saturation, plus
a rate-limited actuator model (moves toward a commanded position at a
maximum slew rate).
"""


class PID:
    """Standard PID controller.

    output = clip( kp*e + ki*integral(e) + kd*de/dt , -limit, +limit )
    integral term is clipped to [-5, 5] before being multiplied by ki.
    """

    def __init__(self, kp, ki, kd, limit, derivative_tau=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.derivative_tau = max(0.0, float(derivative_tau))
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0

    def update(self, error, dt):
        self._integral += error * dt
        self._integral = max(-5.0, min(5.0, self._integral))

        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        if self.derivative_tau > 0.0 and dt > 0.0:
            alpha = dt / (self.derivative_tau + dt)
            self._filtered_derivative += alpha * (derivative - self._filtered_derivative)
            derivative = self._filtered_derivative
        else:
            self._filtered_derivative = derivative
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = max(-self.limit, min(self.limit, output))
        return output


class RateLimitedActuator:
    """Models an actuator that can only move toward a commanded position at
    a bounded slew rate (deg/s internally converted to rad/s at construction
    time is NOT assumed -- caller passes whatever units it wants to work in
    consistently; `max_rate_deg_s` name kept for spec-fidelity but the value
    is treated as "units per second" of whatever unit `position` is in).
    """

    def __init__(self, max_rate_deg_s, initial_position=0.0):
        self.max_rate = max_rate_deg_s
        self.position = initial_position

    def update(self, cmd, dt):
        max_delta = self.max_rate * dt
        delta = cmd - self.position
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        self.position += delta
        return self.position

    def reset(self, position=0.0):
        self.position = position
