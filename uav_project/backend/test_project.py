"""Fast, deterministic checks for the conference experiment model."""
import unittest

import numpy as np

import aircraft_params as P
from adaptive_system import build_morphing_tail_table, morphing_tail_command
from dynamics_6dof import dynamics
from dynamics_6dof import (aero_forces_moments, payload_inertias,
                           payload_cg_position, static_margin)
from final_comparison import _metrics_from_log, run_conventional, run_intelligent
from gust_model import DrydenGust, one_cosine_gust
from pid_controller import RateLimitedActuator
from trim import find_trim
from api import SimRequest


class ProjectTests(unittest.TestCase):
    def test_trim_is_bounded_and_residual_is_small(self):
        for va in (12.0, 15.0, 20.0):
            for payload in (0.0, 0.5, 1.0):
                trim = find_trim(va, payload)
                self.assertTrue(trim["converged"])
                self.assertLessEqual(abs(trim["elevator"]), P.ELEVATOR_LIMIT)
                self.assertGreaterEqual(trim["throttle"], 0.0)
                self.assertLessEqual(trim["throttle"], 1.0)
                self.assertLess(trim["residual_norm"], 1e-6)

    def test_morphing_tail_grid_converges(self):
        va, payload, table = build_morphing_tail_table(
            va_grid=np.array([12.0, 15.0, 20.0]),
            payload_grid=np.array([0.0, 0.5, 1.0]),
        )
        self.assertEqual(table.shape, (3, 3))
        self.assertTrue(np.all(np.abs(table) <= np.radians(15.0)))

    def test_zero_payload_tail_command_is_zero_delta(self):
        self.assertAlmostEqual(morphing_tail_command(0.0, 15.0), 0.0, places=10)

    def test_wind_is_rotated_from_ned_to_body(self):
        state = np.zeros(12)
        state[7] = np.radians(20.0)
        controls = np.array([0.0, 0.0, 0.0, 0.0])
        d0 = dynamics(state, controls, 0.0, wind=np.zeros(3))
        d1 = dynamics(state, controls, 0.0, wind=np.array([2.0, 0.0, 0.0]))
        self.assertFalse(np.allclose(d0[3:6], d1[3:6]))

    def test_gust_is_reproducible(self):
        a = DrydenGust(15.0, 0.5, seed=7)
        b = DrydenGust(15.0, 0.5, seed=7)
        for _ in range(20):
            np.testing.assert_allclose(a.step(0.02), b.step(0.02))

    def test_pitch_metric_uses_reference(self):
        log = np.zeros((2, 12))
        log[:, 0] = (0.0, 1.0)
        log[:, 1] = (5.0, 7.0)
        log[:, 7] = 100.0
        log[:, 8] = 1.0
        metrics = _metrics_from_log(log, 0.0, theta_ref_deg=5.0)
        self.assertAlmostEqual(metrics["max_pitch_dev"], 2.0)

    def test_metrics_use_degree_saturation_limits_and_altitude_metrics(self):
        log = np.zeros((2, 12))
        log[:, 0] = (0.0, 1.0)
        log[:, 7] = (P.alt_trim, P.alt_trim - 30.0)
        log[:, 4] = np.degrees(P.ELEVATOR_LIMIT)
        log[:, 8] = 1.0
        metrics = _metrics_from_log(log, 0.0, theta_ref_deg=0.0)
        self.assertGreater(metrics["elev_saturation_s"], 0.0)
        self.assertAlmostEqual(metrics["max_altitude_drop_m"], 30.0)
        self.assertGreater(metrics["rms_altitude_error"], 0.0)

    def test_actuator_rate_and_position_limits(self):
        actuator = RateLimitedActuator(0.5, 0.0)
        self.assertAlmostEqual(actuator.update(2.0, 0.1), 0.05)
        self.assertAlmostEqual(actuator.update(-2.0, 0.1), 0.0)

    def test_tail_incidence_changes_force_and_drag(self):
        state = find_trim(15.0, 0.0)["state"]
        controls = find_trim(15.0, 0.0)["controls"]
        f0 = aero_forces_moments(state, controls, 0.0, tail_incidence=0.0)
        f1 = aero_forces_moments(state, controls, 0.0, tail_incidence=np.radians(5.0))
        self.assertFalse(np.allclose(f0[0], f1[0]))
        self.assertGreater(f1[-1], f0[-1])

    def test_payload_changes_all_inertia_axes(self):
        base = np.asarray(payload_inertias(0.0))
        loaded = np.asarray(payload_inertias(1.0))
        self.assertTrue(np.all(loaded[:3] > base[:3]))
        self.assertTrue(np.all(np.isfinite(loaded)))

    def test_payload_cg_and_static_margin_are_mass_derived(self):
        self.assertLess(payload_cg_position(1.0), 0.0)  # aft in body x
        self.assertLess(static_margin(1.0), static_margin(0.0))

    def test_one_cosine_gust_is_finite_and_axis_correct(self):
        g0 = one_cosine_gust(0.0, 1.0, 2.0, 3.0, axis=2)
        gm = one_cosine_gust(2.0, 1.0, 2.0, 3.0, axis=2)
        self.assertTrue(np.allclose(g0, 0.0))
        self.assertAlmostEqual(gm[2], 3.0)
        self.assertAlmostEqual(gm[0], 0.0)

    def test_api_rejects_invalid_time_order(self):
        with self.assertRaises(ValueError):
            SimRequest(mode="conventional", payload_level=0.5, airspeed=15.0,
                       gust_intensity=1.0, change_t=5.0, T=4.0)

    def test_api_rejects_out_of_envelope_airspeed(self):
        with self.assertRaises(ValueError):
            SimRequest(mode="conventional", payload_level=0.5, airspeed=5.0,
                       gust_intensity=1.0, change_t=1.0, T=4.0)
        with self.assertRaises(ValueError):
            SimRequest(mode="conventional", payload_level=0.5, airspeed=10.0,
                       gust_intensity=1.0, change_t=1.0, T=4.0)

    def test_gust_only_window_does_not_create_large_payload_estimate(self):
        _, _, estimates, status = run_intelligent(
            payload_level=0.0, Va_init=15.0, gust_intensity=1.0,
            seed=0, T=8.0, debug=True)
        self.assertTrue(status["valid"])
        self.assertTrue(estimates)
        # The estimator is continuous and intentionally remains conservative
        # rather than forcing a false zero under a strong gust.
        self.assertLess(max(v for _, v in estimates), 0.40)
        self.assertTrue(status["turbulence_estimates"])

    def test_lateral_disturbance_changes_run(self):
        off = run_conventional(0.5, 15.0, 0.5, seed=2, T=8.0,
                               change_t=1.0, lateral_disturbance=False)
        on = run_conventional(0.5, 15.0, 0.5, seed=2, T=8.0,
                              change_t=1.0, lateral_disturbance=True)
        self.assertFalse(off[-1]["lateral_disturbance"])
        self.assertTrue(on[-1]["lateral_disturbance"])
        self.assertNotAlmostEqual(off[0]["mean_rudder"], on[0]["mean_rudder"], places=5)

    def test_intelligent_status_records_configuration(self):
        _, _, _, status = run_intelligent(
            0.2, 15.0, 0.0, seed=4, T=0.5, change_t=0.2,
            payload_source="oracle", tail_enabled=False, gain_schedule=False)
        self.assertEqual(status["payload_source"], "oracle")
        self.assertFalse(status["tail_enabled"])
        self.assertFalse(status["gain_schedule"])
        self.assertEqual(status["random_seed"], 4)


if __name__ == "__main__":
    unittest.main()
