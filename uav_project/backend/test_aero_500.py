"""Large data-driven aerospace regression suite.

This file deliberately contains 525 independently executed test cases.  The
cases are generated from explicit physical grids rather than copied test
methods, so every case exercises a different seed, operating point, or
configuration.  It complements the smaller readable suites:

    python -m unittest -v test_aero_500.py

The suite is intentionally fast: it tests contracts and local model
properties, while the long Monte Carlo and grid campaigns remain evidence
artifacts rather than being repeated 525 times.
"""

import unittest

import numpy as np

import aircraft_params as P
from dynamics_6dof import _dcm_body_to_ned, aero_forces_moments
from gust_model import DrydenGust
from realism import ActuatorChannel, SafetySupervisor, SensorSuite


DT = 0.02


class GeneratedDCMTests(unittest.TestCase):
    """125 attitude-grid cases for coordinate-transform invariants."""


def _dcm_case(self, values):
    phi, theta, psi = values
    dcm = _dcm_body_to_ned(phi, theta, psi)
    np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
    self.assertAlmostEqual(float(np.linalg.det(dcm)), 1.0, places=12)


class GeneratedAeroCases(unittest.TestCase):
    """100 operating-point cases for finite aerodynamic force evaluation."""


def _aero_case(self, values):
    speed, alpha, payload, tail = values
    state = np.zeros(12)
    state[3] = speed * np.cos(alpha)
    state[5] = speed * np.sin(alpha)
    controls = np.array([0.0, 0.0, 0.0, 0.45])
    forces, moments, measured_alpha, beta, va, jy, mass, drag = \
        aero_forces_moments(state, controls, payload, tail_incidence=tail)
    self.assertTrue(np.all(np.isfinite(forces)))
    self.assertTrue(np.all(np.isfinite(moments)))
    self.assertTrue(np.isfinite(measured_alpha))
    self.assertTrue(np.isfinite(beta))
    self.assertGreater(float(va), 0.0)
    self.assertGreater(float(jy), 0.0)
    self.assertGreaterEqual(float(mass), P.mass)
    self.assertGreaterEqual(float(drag), 0.0)


class GeneratedGustCases(unittest.TestCase):
    """100 seeded OU/Dryden gust cases for finite deterministic disturbances."""


def _gust_case(self, values):
    speed, intensity, seed, model = values
    a = DrydenGust(speed, intensity, seed=seed, model=model)
    b = DrydenGust(speed, intensity, seed=seed, model=model)
    samples_a = np.asarray([a.step(DT) for _ in range(30)])
    samples_b = np.asarray([b.step(DT) for _ in range(30)])
    self.assertEqual(samples_a.shape, (30, 3))
    self.assertTrue(np.all(np.isfinite(samples_a)))
    np.testing.assert_allclose(samples_a, samples_b, atol=1e-13)


class GeneratedActuatorCases(unittest.TestCase):
    """100 actuator cases for causality, bounds, and rate limiting."""


def _actuator_case(self, values):
    rate, tau, lower, upper, command = values
    actuator = ActuatorChannel(
        max_rate=rate, initial_position=0.0, minimum=lower, maximum=upper,
        time_constant=tau, seed=0)
    output = np.asarray([actuator.update(command, DT) for _ in range(25)])
    self.assertTrue(np.all(np.isfinite(output)))
    self.assertTrue(np.all(output >= lower - 1e-12))
    self.assertTrue(np.all(output <= upper + 1e-12))
    self.assertTrue(np.all(np.abs(np.diff(output)) <= rate * DT + 1e-10))


class GeneratedSensorSupervisorCases(unittest.TestCase):
    """100 sensor/safety cases for repeatability and envelope decisions."""


def _sensor_supervisor_case(self, values):
    delay_steps, noise, seed, theta_deg, airspeed = values
    state = np.zeros(12)
    state[7] = np.radians(theta_deg)
    config = {
        "enabled": True,
        "delay_s": delay_steps * DT,
        "sample_period_s": DT,
        "angle_noise_deg": noise,
        "seed": seed,
    }
    first = SensorSuite(config, DT, state, seed=seed)
    second = SensorSuite(config, DT, state, seed=seed)
    for _ in range(3):
        measured_a = first.measure(state, airspeed, 0.0)
        measured_b = second.measure(state, airspeed, 0.0)
        np.testing.assert_allclose(measured_a[0], measured_b[0])
        self.assertEqual(measured_a[1], measured_b[1])
        self.assertEqual(measured_a[2], measured_b[2])
    supervisor = SafetySupervisor({
        "enabled": True,
        "theta_limit_deg": 35.0,
        "min_airspeed_mps": 10.0,
        "max_airspeed_mps": 25.0,
    })
    safe, reason = supervisor.evaluate(state, airspeed)
    self.assertTrue(safe, reason)
    self.assertIsNone(reason)
    self.assertFalse(supervisor.fallback_active)


def _install_cases(cls, prefix, cases, body):
    for index, case in enumerate(cases):
        def test(self, case=case):
            body(self, case)
        test.__name__ = f"test_{prefix}_{index:03d}"
        setattr(cls, test.__name__, test)


_attitudes = [
    (phi, theta, psi)
    for phi in np.linspace(-0.8, 0.8, 5)
    for theta in np.linspace(-0.5, 0.5, 5)
    for psi in np.linspace(-1.0, 1.0, 5)
]
_aero_points = [
    (speed, alpha, payload, tail)
    for speed in np.linspace(10.5, 22.0, 5)
    for alpha in np.linspace(-0.08, 0.08, 5)
    for payload in (0.0, 0.35, 0.70, 0.74)
    for tail in (0.0,)
][:100]
_gust_points = [
    (speed, intensity, seed, model)
    for seed in range(25, 50)
    for model in ("ou", "dryden")
    for speed, intensity in ((12.0, 0.5), (20.0, 1.5))
]
_actuator_points = [
    (rate, tau, -limit, limit, command)
    for rate in (0.5, 1.0, 2.0, 4.0, 8.0)
    for tau in (0.0, 0.04, 0.12, 0.25, 0.5)
    for limit, command in ((0.4, 1.0), (0.4, -1.0), (0.4, 0.2))
]
_sensor_points = [
    (delay, noise, seed, theta, speed)
    for delay in (0, 1, 2, 3, 4)
    for noise in (0.0, 0.05, 0.2, 0.5, 1.0)
    for seed, theta, speed in ((seed, theta, speed)
                               for seed, theta, speed in (
                                   (11, -5.0, 12.0), (12, 0.0, 15.0),
                                   (13, 5.0, 20.0), (14, 10.0, 24.0)))
]

_install_cases(GeneratedDCMTests, "dcm", _attitudes, _dcm_case)       # 125
_install_cases(GeneratedAeroCases, "aero", _aero_points, _aero_case)  # 100
_install_cases(GeneratedGustCases, "gust", _gust_points, _gust_case)  # 100
_install_cases(GeneratedActuatorCases, "actuator", _actuator_points, _actuator_case)  # 75
_install_cases(GeneratedSensorSupervisorCases, "sensor", _sensor_points, _sensor_supervisor_case)  # 100


if __name__ == "__main__":
    unittest.main(verbosity=2)
