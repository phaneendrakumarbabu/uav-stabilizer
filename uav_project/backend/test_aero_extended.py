"""Extended QA and aeronautical-analysis tests.

This suite is intentionally organized by evidence type:

* physics invariants and coordinate conventions,
* numerical convergence and repeatability,
* gust/sensor/actuator realism,
* controller/supervisor integration,
* API and evidence-artifact contracts,
* stability-envelope and Monte Carlo reporting checks.
"""

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

import aircraft_params as P
import api
from adaptive_system import morphing_tail_command
from dynamics_6dof import (
    _dcm_body_to_ned,
    aero_forces_moments,
    payload_inertias,
    payload_effects,
    rk4_step,
    static_margin,
)
from final_comparison import DT, run_conventional, run_intelligent
from gust_model import DrydenGust
from linear_stability import closed_loop_matrix
from realism import ActuatorChannel, SafetySupervisor, SensorSuite
from trim import find_trim


ROOT = Path(__file__).resolve().parent


class PhysicsInvariantTests(unittest.TestCase):
    def test_dcm_is_orthogonal_right_handed_over_attitude_grid(self):
        for phi in np.radians((-30.0, 0.0, 30.0)):
            for theta in np.radians((-20.0, 0.0, 20.0)):
                for psi in np.radians((-45.0, 0.0, 45.0)):
                    dcm = _dcm_body_to_ned(phi, theta, psi)
                    np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
                    self.assertAlmostEqual(float(np.linalg.det(dcm)), 1.0, places=12)

    def test_dynamic_pressure_scaling_is_physical(self):
        state12 = np.zeros(12)
        state18 = np.zeros(12)
        state12[3] = 12.0
        state18[3] = 18.0
        controls = np.zeros(4)
        drag12 = aero_forces_moments(state12, controls, 0.0)[-1]
        drag18 = aero_forces_moments(state18, controls, 0.0)[-1]
        self.assertGreater(drag12, 0.0)
        self.assertAlmostEqual(drag18 / drag12, (18.0 / 12.0) ** 2, delta=0.08)

    def test_payload_mass_inertia_and_static_margin_have_expected_trends(self):
        masses = [payload_effects(level)[3] for level in np.linspace(0.0, 1.0, 6)]
        self.assertTrue(np.all(np.diff(masses) > 0.0))
        margins = [static_margin(level) for level in np.linspace(0.0, 1.0, 6)]
        self.assertTrue(np.all(np.diff(margins) < 0.0))
        for level in np.linspace(0.0, 1.0, 11):
            jx, jy, jz, jxz = payload_inertias(level)
            inertia = np.array([[jx, 0.0, -jxz], [0.0, jy, 0.0], [-jxz, 0.0, jz]])
            self.assertTrue(np.all(np.linalg.eigvalsh(inertia) > 0.0))

    def test_tail_command_is_bounded_and_increases_with_payload(self):
        values = [morphing_tail_command(level, 15.0) for level in (0.0, 0.2, 0.5, 0.7)]
        self.assertAlmostEqual(values[0], 0.0, places=12)
        self.assertTrue(np.all(np.abs(values) <= np.radians(15.0)))
        self.assertTrue(np.all(np.diff(values) >= -1e-12))

    def test_trim_residual_is_small_across_operating_envelope(self):
        for va in (12.0, 15.0, 20.0):
            for payload in (0.0, 0.35, 0.7):
                result = find_trim(va, payload)
                self.assertTrue(result["converged"])
                self.assertTrue(np.isfinite(result["residual_norm"]))
                self.assertLess(result["residual_norm"], 1e-5)

    def test_closed_loop_pitch_stability_covers_open_loop_stress_payloads(self):
        for va in (12.0, 15.0, 20.0):
            for payload in (0.0, 0.7, 0.8, 1.0):
                result = find_trim(va, payload)
                matrix = closed_loop_matrix(result["state"], result["controls"], payload)
                self.assertTrue(np.all(np.isfinite(matrix)))
                self.assertLess(float(np.linalg.eigvals(matrix).real.max()), 0.0)


class NumericalConvergenceTests(unittest.TestCase):
    def test_rk4_repeatability_and_timestep_convergence(self):
        trim = find_trim(15.0, 0.2)
        state = trim["state"].copy()
        controls = trim["controls"].copy()

        def integrate(dt):
            current = state.copy()
            for _ in range(round(1.0 / dt)):
                current = rk4_step(current, controls, 0.2, dt)
            return current

        coarse = integrate(0.02)
        fine = integrate(0.01)
        np.testing.assert_allclose(coarse, integrate(0.02), atol=1e-12)
        self.assertLess(float(np.linalg.norm(coarse - fine)), 0.10)
        self.assertTrue(np.all(np.isfinite(fine)))

    def test_nominal_controller_log_time_is_monotonic_and_uniform(self):
        _, log, status = run_conventional(0.2, 15.0, 0.0, seed=41, T=1.0, change_t=0.2)
        self.assertTrue(status["valid"], status)
        times = log[:, 0]
        self.assertTrue(np.all(np.diff(times) > 0.0))
        np.testing.assert_allclose(np.diff(times), DT, atol=1e-12)


class RealismModelTests(unittest.TestCase):
    def test_dryden_zero_intensity_is_zero(self):
        gust = DrydenGust(15.0, 0.0, seed=4, model="dryden")
        history = np.asarray([gust.step(DT) for _ in range(100)])
        np.testing.assert_allclose(history, 0.0, atol=1e-12)

    def test_dryden_axes_are_correlated_but_not_identical(self):
        gust = DrydenGust(15.0, 1.0, seed=5, model="dryden")
        history = np.asarray([gust.step(DT) for _ in range(500)])
        self.assertTrue(np.all(np.isfinite(history)))
        self.assertGreater(float(np.std(history[:, 0])), 0.0)
        self.assertGreater(float(np.std(history[:, 1])), 0.0)
        self.assertFalse(np.allclose(history[:, 0], history[:, 1]))

    def test_sensor_noise_is_seed_reproducible(self):
        config = {"enabled": True, "delay_s": 0.02, "angle_noise_deg": 0.2,
                  "airspeed_noise_mps": 0.1, "seed": 77}
        state = np.zeros(12)
        a = SensorSuite(config, DT, state, seed=77)
        b = SensorSuite(config, DT, state, seed=77)
        for _ in range(10):
            measured_a = a.measure(state, 15.0, 0.0)
            measured_b = b.measure(state, 15.0, 0.0)
            np.testing.assert_allclose(measured_a[0], measured_b[0])
            self.assertEqual(measured_a[1], measured_b[1])
            self.assertEqual(measured_a[2], measured_b[2])

    def test_actuator_has_monotonic_lag_rate_and_position_limits(self):
        actuator = ActuatorChannel(
            max_rate=1.0, initial_position=0.0, minimum=-0.5, maximum=0.5,
            time_constant=0.2)
        values = [actuator.update(1.0, 0.05) for _ in range(20)]
        self.assertTrue(np.all(np.diff(values) >= -1e-12))
        self.assertTrue(np.all(np.abs(values) <= 0.5 + 1e-12))
        self.assertTrue(np.all(np.diff(values) <= 0.05 + 1e-12))


class SupervisorAndIntegrationTests(unittest.TestCase):
    def test_supervisor_latches_each_primary_envelope_violation(self):
        cases = (
            (dict(theta_limit_deg=10.0), np.radians(11.0), None, None, "pitch_envelope"),
            (dict(alpha_limit_deg=5.0), 0.0, np.radians(6.0), None, "angle_of_attack_envelope"),
            (dict(beta_limit_deg=5.0), 0.0, None, np.radians(6.0), "sideslip_envelope"),
        )
        for config, theta, alpha, beta, expected in cases:
            supervisor = SafetySupervisor({"enabled": True, **config})
            state = np.zeros(12)
            state[7] = theta
            safe, reason = supervisor.evaluate(state, 15.0, alpha=alpha, beta=beta)
            self.assertFalse(safe)
            self.assertIn(expected, reason)
            self.assertTrue(supervisor.status()["fallback_active"])

    def test_supervisor_alpha_protection_preserves_recovery_direction(self):
        supervisor = SafetySupervisor({
            "enabled": True,
            "alpha_limit_deg": 12.0,
            "alpha_protection_margin_deg": 1.0,
        })
        self.assertAlmostEqual(
            supervisor.protect_elevator(-0.2, np.radians(11.5)), np.radians(5.0))
        self.assertAlmostEqual(
            supervisor.protect_elevator(0.2, np.radians(-11.5)), -np.radians(5.0))
        self.assertAlmostEqual(
            supervisor.protect_elevator(-0.2, np.radians(5.0)), -0.2)
        protected_ref = supervisor.protect_pitch_reference(
            np.radians(10.0), np.radians(5.0), np.radians(11.5))
        self.assertLessEqual(protected_ref, np.radians(3.0) + 1e-12)

    def test_supervisor_uses_raw_feature_support_fraction(self):
        supervisor = SafetySupervisor({
            "enabled": True,
            "estimator_ood_fraction": 0.4,
        })
        state = np.zeros(12)
        low = np.zeros(5)
        high = np.ones(5)
        one_outside = np.array([2.0, 0.5, 0.5, 0.5, 0.5])
        safe, reason = supervisor.evaluate(
            state, 15.0, feature_vector=one_outside,
            feature_bounds=(low, high), scaled_features=np.zeros(5))
        self.assertTrue(safe, reason)
        two_outside = np.array([2.0, 2.0, 0.5, 0.5, 0.5])
        safe, reason = supervisor.evaluate(
            state, 15.0, feature_vector=two_outside,
            feature_bounds=(low, high), scaled_features=np.zeros(5))
        self.assertFalse(safe)
        self.assertIn("estimator_out_of_distribution", reason)

    def test_realism_options_flow_through_intelligent_controller(self):
        sensor = {"enabled": True, "delay_s": 0.02, "angle_noise_deg": 0.05,
                  "rate_noise_deg_s": 0.2, "airspeed_noise_mps": 0.05, "seed": 9}
        actuator = {name: {"enabled": True, "time_constant_s": 0.05}
                    for name in ("elevator", "aileron", "rudder", "tail")}
        metrics, log, estimates, status = run_intelligent(
            0.2, 15.0, 0.2, seed=9, T=1.0, change_t=0.2,
            gust_model="dryden", sensor_config=sensor,
            actuator_config=actuator, safety_config={"enabled": True}, debug=True)
        self.assertTrue(status["sensor_realism"])
        self.assertEqual(status["gust_model"], "dryden")
        self.assertTrue(status["safety_supervisor"]["enabled"])
        self.assertTrue(np.all(np.isfinite(log)))
        self.assertTrue(all(math.isfinite(float(v)) for v in metrics.values()))

    def test_api_exposes_realism_flags_and_finite_response(self):
        request = api.SimRequest(
            mode="intelligent", payload_level=0.2, airspeed=15.0,
            gust_intensity=0.2, seed=10, change_t=0.1, T=0.5,
            gust_model="dryden", sensor_realism=True,
            actuator_realism=True, safety_supervisor=True)
        response = api._run("intelligent", request)
        metrics, _, status = response
        self.assertEqual(status["gust_model"], "dryden")
        self.assertTrue(status["sensor_realism"])
        self.assertTrue(status["safety_supervisor"]["enabled"])
        self.assertTrue(all(math.isfinite(float(v)) for v in metrics.values()))


class EvidenceAndReleaseTests(unittest.TestCase):
    def test_grid_manifests_are_complete_unique_and_hashed(self):
        for directory in ("results_grid_final", "results_grid_final2"):
            grid = ROOT / directory
            config = json.loads((grid / "grid_config.json").read_text())
            rows = [json.loads(line) for line in
                    (grid / "raw_runs.jsonl").read_text().splitlines() if line]
            keys = {(r["payload"], r["gust"], r["airspeed"], r["seed"], r["config"])
                    for r in rows}
            self.assertTrue(config["grid_complete"])
            self.assertEqual(len(rows), config["expected_runs"])
            self.assertEqual(len(keys), len(rows))
            for relative, expected in config["sha256"].items():
                candidate = grid / relative
                if not candidate.exists():
                    candidate = ROOT / relative
                self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(), expected)

    def test_each_summary_row_accounts_for_all_trials(self):
        for directory in ("results_grid_final", "results_grid_final2"):
            with (ROOT / directory / "summary.csv").open(encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    self.assertEqual(
                        int(row["n_valid"]) + int(row["invalid_count"]), 30)

    def test_uncertainty_artifact_has_no_dropped_trials(self):
        result = json.loads((ROOT / "uncertainty_monte_carlo_results.json").read_text())
        self.assertEqual(result["trials"], len(result["records"]))
        self.assertEqual(result["disturbance_model"], "dryden")
        self.assertTrue(result["lateral_disturbance"])
        self.assertEqual(result["conventional_summary"]["completed"], result["trials"])
        self.assertEqual(result["intelligent_summary"]["completed"], result["trials"])

    def test_api_rejects_sub_timestep_and_cors_is_restricted(self):
        with self.assertRaises(ValidationError):
            api.SimRequest(mode="conventional", payload_level=0.2,
                           airspeed=15.0, gust_intensity=0.0,
                           change_t=0.0, T=DT / 2.0)
        for middleware in api.app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                self.assertNotEqual(middleware.kwargs.get("allow_origins"), ["*"])


if __name__ == "__main__":
    unittest.main()
