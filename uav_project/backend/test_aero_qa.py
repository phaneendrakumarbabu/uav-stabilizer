"""Aeronautical release-gate tests for the updated UAV project.

The ordinary smoke tests should remain green while this suite checks claims
that matter for flight safety and conference reproducibility.  A failing
test is an explicit release blocker, not a reason to weaken the test.
"""

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from pydantic import ValidationError

import api
from final_comparison import run_intelligent
from gust_model import DrydenGust
from realism import ActuatorChannel, SafetySupervisor, SensorSuite


ROOT = Path(__file__).resolve().parent


class AeronauticalReleaseGateTests(unittest.TestCase):
    def test_estimator_dataset_and_model_have_matching_21_feature_contract(self):
        dataset = np.load(ROOT / "payload_dataset_adaptive.npz", allow_pickle=False)
        bundle = joblib.load(ROOT / "payload_estimator.joblib")
        self.assertEqual(dataset["X"].shape[1], 21)
        self.assertEqual(bundle["feature_count"], 21)
        self.assertEqual(len(bundle["feature_names"]), 21)
        self.assertEqual(bundle["model"].n_features_in_, 21)

    def test_intelligent_controller_is_finite_in_nominal_envelope(self):
        metrics, log, estimates, status = run_intelligent(
            payload_level=0.5,
            Va_init=15.0,
            gust_intensity=0.5,
            seed=100000,
            T=2.0,
            change_t=0.5,
            debug=True,
        )
        self.assertTrue(status["valid"], status)
        self.assertTrue(np.all(np.isfinite(log)))
        self.assertTrue(all(math.isfinite(float(v)) for v in metrics.values()))
        self.assertTrue(all(math.isfinite(float(v)) for _, v in estimates))

    def test_dryden_gust_is_reproducible_and_finite(self):
        first = DrydenGust(15.0, 1.0, seed=99, model="dryden")
        second = DrydenGust(15.0, 1.0, seed=99, model="dryden")
        a = np.asarray([first.step(0.02) for _ in range(200)])
        b = np.asarray([second.step(0.02) for _ in range(200)])
        self.assertTrue(np.all(np.isfinite(a)))
        np.testing.assert_allclose(a, b)
        self.assertGreater(float(np.std(a[:, 1])), 0.0)

    def test_sensor_delay_and_actuator_dynamics_are_causal_and_bounded(self):
        initial = np.zeros(12)
        sensor = SensorSuite(
            {"enabled": True, "delay_s": 0.04, "sample_period_s": 0.02},
            dt=0.02, initial_state=initial, seed=7)
        changed = initial.copy()
        changed[7] = 1.0
        first = sensor.measure(changed, 15.0, 0.0)[0]
        second = sensor.measure(changed, 15.0, 0.0)[0]
        self.assertAlmostEqual(first[7], 0.0, places=6)
        self.assertAlmostEqual(second[7], 0.0, places=6)

        actuator = ActuatorChannel(
            max_rate=0.5, initial_position=0.0, minimum=-1.0, maximum=1.0,
            time_constant=0.2)
        value = actuator.update(10.0, 0.1)
        self.assertLessEqual(abs(value), 0.05 + 1e-12)
        self.assertLessEqual(value, 1.0)

    def test_safety_supervisor_latches_fallback_on_ood_features(self):
        supervisor = SafetySupervisor({"enabled": True})
        safe, reason = supervisor.evaluate(
            np.zeros(12), 15.0, scaled_features=np.full(21, 6.0), t=1.0)
        self.assertFalse(safe)
        self.assertEqual(reason, "estimator_out_of_distribution")
        self.assertTrue(supervisor.status()["fallback_active"])

    def test_api_rejects_or_handles_sub_timestep_requests_without_nan(self):
        """A live endpoint must never return an empty log with NaN metrics."""
        with self.assertRaises(ValidationError):
            api.SimRequest(
                mode="conventional",
                payload_level=0.2,
                airspeed=15.0,
                gust_intensity=0.0,
                change_t=0.0,
                T=0.001,
            )

    def test_grid_manifests_match_the_current_source_revision(self):
        mismatches = []
        for directory in ("results_grid_final", "results_grid_final2"):
            grid_dir = ROOT / directory
            config = json.loads((grid_dir / "grid_config.json").read_text())
            self.assertTrue(config["grid_complete"], directory)
            self.assertEqual(config["expected_runs"], config["completed_runs"])
            for relative, expected in config["sha256"].items():
                candidate = grid_dir / relative
                if not candidate.exists():
                    candidate = ROOT / relative
                actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if actual != expected:
                    mismatches.append(f"{directory}/{relative}")
        self.assertEqual(
            mismatches,
            [],
            "Evidence was generated with a different source revision: "
            + ", ".join(mismatches),
        )

    def test_high_gust_high_payload_nn_does_not_lose_more_runs_than_baseline(self):
        """Safety acceptance gate for the hardest supplied grid condition."""
        with (ROOT / "results_grid_final" / "summary.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        selected = [
            row for row in rows
            if row["payload"] == "0.7"
            and row["gust"] == "1.5"
            and row["airspeed"] == "20.0"
            and row["metric"] == "rms_pitch_error"
            and row["config"] in {"conventional", "nn"}
        ]
        valid = {row["config"]: int(row["n_valid"]) for row in selected}
        self.assertEqual(set(valid), {"conventional", "nn"})
        self.assertGreaterEqual(
            valid["nn"], valid["conventional"],
            "NN controller has fewer valid runs than conventional at the "
            "high-payload/high-gust/high-speed boundary",
        )

    def test_evidence_grid_includes_crosswind_cases(self):
        """Crosswind must be represented before claiming lateral robustness."""
        configs = []
        for directory in ("results_grid_final", "results_grid_final2"):
            config = json.loads(
                (ROOT / directory / "grid_config.json").read_text()
            )
            configs.append(config["crosswind"])
        mc_path = ROOT / "uncertainty_monte_carlo_results.json"
        mc_crosswind = False
        if mc_path.exists():
            mc_crosswind = bool(json.loads(mc_path.read_text())["lateral_disturbance"])
        self.assertTrue(
            any(configs) or mc_crosswind,
            "No supplied evidence campaign exercises crosswind/lateral disturbance",
        )

    def test_uncertainty_campaign_reports_all_trials(self):
        path = ROOT / "uncertainty_monte_carlo_results.json"
        self.assertTrue(path.exists(), "Run the uncertainty campaign first")
        result = json.loads(path.read_text())
        self.assertEqual(result["trials"], len(result["records"]))
        self.assertTrue(result["lateral_disturbance"])
        self.assertEqual(result["disturbance_model"], "dryden")
        for controller in ("conventional_summary", "intelligent_summary"):
            summary = result[controller]
            self.assertEqual(summary["completed"], result["trials"])
            self.assertGreaterEqual(summary["fallback_count"], 0)

    def test_public_api_does_not_use_wildcard_cors_with_credentials(self):
        insecure = any(
            middleware.cls.__name__ == "CORSMiddleware"
            and middleware.kwargs.get("allow_origins") == ["*"]
            and middleware.kwargs.get("allow_credentials") is True
            for middleware in api.app.user_middleware
        )
        self.assertFalse(
            insecure,
            "Wildcard CORS with credentials is unsafe for a deployed API",
        )


if __name__ == "__main__":
    unittest.main()
