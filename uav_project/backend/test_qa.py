"""QA-level contract and integration tests for the UAV project.

These tests intentionally exercise the public API wrapper and the data shape
used by the frontend in addition to the physics checks in ``test_project.py``.
They are deterministic and keep runs short so they can be used as a smoke
suite in CI.
"""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError
from fastapi import HTTPException

import api
import aircraft_params as P
from final_comparison import run_conventional, run_intelligent


class QaContractTests(unittest.TestCase):
    def test_health_and_geometry_contract(self):
        self.assertEqual(api.health(), {"status": "ok"})
        geometry = api.get_geometry()
        self.assertIn("real_from_backend", geometry)
        self.assertIn("cosmetic_only", geometry)
        real = geometry["real_from_backend"]
        for key in ("b", "c", "S_tail", "l_tail"):
            self.assertIn(key, real)
            self.assertTrue(math.isfinite(float(real[key])))

    def test_request_accepts_documented_envelope_edges(self):
        for airspeed in (P.Va_train_min, P.Va_train_max):
            request = api.SimRequest(
                mode="conventional",
                payload_level=0.0,
                airspeed=airspeed,
                gust_intensity=2.0,
                change_t=0.0,
                T=1.0,
            )
            self.assertEqual(request.airspeed, airspeed)

    def test_request_rejects_non_finite_and_invalid_time_values(self):
        invalid_cases = (
            {"payload_level": float("nan")},
            {"airspeed": float("inf")},
            {"gust_intensity": float("-inf")},
            {"change_t": 2.0, "T": 1.0},
        )
        for overrides in invalid_cases:
            values = dict(
                mode="conventional",
                payload_level=0.5,
                airspeed=15.0,
                gust_intensity=0.5,
                change_t=0.5,
                T=1.0,
            )
            values.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises((ValidationError, ValueError)):
                    api.SimRequest(**values)

    def test_simulate_endpoint_returns_frontend_log_contract(self):
        request = api.SimRequest(
            mode="conventional", payload_level=0.5, airspeed=15.0,
            gust_intensity=0.0, seed=11, change_t=0.1, T=0.2,
        )
        with patch.object(api, "_save_run_artifacts", return_value="qa-artifact"):
            response = api.simulate(request)

        self.assertEqual(response["saved_to"], "qa-artifact")
        self.assertIn("metrics", response)
        self.assertIn("status", response)
        log = response["log"]
        required = {
            "t", "theta_deg", "phi_deg", "psi_deg", "de_deg", "da_deg",
            "dr_deg", "alt_m", "drag", "pn_m", "pe_m", "pd_m",
            "theta_ref_deg",
        }
        self.assertTrue(required.issubset(log))
        lengths = {len(log[key]) for key in required}
        self.assertEqual(len(lengths), 1)

    def test_compare_endpoint_returns_both_controller_results(self):
        request = api.SimRequest(
            mode="compare", payload_level=0.2, airspeed=15.0,
            gust_intensity=0.0, seed=12, change_t=0.1, T=0.2,
        )
        with patch.object(api, "_save_run_artifacts", return_value="qa-artifact") as save:
            response = api.simulate_compare(request)

        self.assertEqual(set(response), {"conventional", "intelligent"})
        self.assertEqual(save.call_count, 2)
        for mode in ("conventional", "intelligent"):
            self.assertIn("metrics", response[mode])
            self.assertIn("log", response[mode])
            self.assertIn("status", response[mode])

    def test_single_simulate_rejects_compare_mode(self):
        request = api.SimRequest(
            mode="compare", payload_level=0.2, airspeed=15.0,
            gust_intensity=0.0, change_t=0.0, T=0.2,
        )
        with self.assertRaises(HTTPException) as raised:
            api.simulate(request)
        self.assertEqual(raised.exception.status_code, 422)

    def test_log_shape_validation_catches_backend_frontend_mismatch(self):
        with self.assertRaises(ValueError):
            api._log_to_dict(np.zeros((3, 12)), "conventional")
        with self.assertRaises(ValueError):
            api._log_to_dict(np.zeros((3, 13)), "intelligent")


class QaSimulationTests(unittest.TestCase):
    def test_short_runs_are_finite_and_reproducible(self):
        first = run_conventional(0.5, 15.0, 0.0, seed=21, T=0.5, change_t=0.2)
        second = run_conventional(0.5, 15.0, 0.0, seed=21, T=0.5, change_t=0.2)
        for result in (first, second):
            metrics, log, status = result
            self.assertTrue(status["valid"])
            self.assertEqual(log.ndim, 2)
            self.assertEqual(log.shape[1], 13)
            self.assertTrue(np.all(np.isfinite(log)))
            self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))
        np.testing.assert_allclose(first[1], second[1])

    def test_intelligent_run_exposes_estimator_and_controller_status(self):
        metrics, log, estimates, status = run_intelligent(
            0.5, 15.0, 0.0, seed=22, T=0.8, change_t=0.2, debug=True,
        )
        self.assertTrue(status["valid"])
        self.assertEqual(log.shape[1], 15)
        self.assertTrue(np.all(np.isfinite(log)))
        self.assertIn("payload_source", status)
        self.assertIn("tail_enabled", status)
        self.assertIn("gain_schedule", status)
        self.assertTrue(status["turbulence_estimates"] is not None)
        self.assertTrue(all(math.isfinite(float(t)) and math.isfinite(float(v))
                            for t, v in estimates))
        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))


class QaFrontendContractTests(unittest.TestCase):
    def test_frontend_contains_expected_runtime_scripts_and_api_routes(self):
        frontend = Path(__file__).resolve().parents[1] / "frontend"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        api_js = (frontend / "js" / "api.js").read_text(encoding="utf-8")
        app_js = (frontend / "js" / "app.js").read_text(encoding="utf-8")
        for script in ("js/geometry.js", "js/api.js", "js/viewport.js",
                       "js/charts.js", "js/app.js"):
            self.assertIn(f'src="{script}', html)
        for route in ("/api/health", "/api/geometry", "/api/simulate",
                      "/api/simulate/compare"):
            self.assertIn(route, api_js)

    def test_frontend_exposes_reproducible_seed_control(self):
        frontend = Path(__file__).resolve().parents[1] / "frontend"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        app_js = (frontend / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="seed-input"', html)
        self.assertIn('const seedInput = el("seed-input")', app_js)
        self.assertIn("seed: Number.isFinite(enteredSeed)", app_js)
        self.assertNotIn("seed: Math.floor(Math.random()", app_js)


if __name__ == "__main__":
    unittest.main()
