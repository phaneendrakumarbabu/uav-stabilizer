"""Uncertainty and robustness Monte Carlo campaign.

This is intentionally separate from the published nominal grids.  It samples
uncertain aerodynamic, mass-property, atmosphere, sensor, actuator, and
operating-condition inputs, then records failures and safety-supervisor
fallbacks instead of filtering them out.

Example:
    python uncertainty_monte_carlo.py --trials 100
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import aircraft_params as P
from final_comparison import run_conventional, run_intelligent


ROOT = Path(__file__).resolve().parent


@contextmanager
def temporary_parameter_set(values):
    """Temporarily apply sampled physical parameters and always restore them."""
    names = tuple(values)
    previous = {name: getattr(P, name) for name in names}
    try:
        for name, value in values.items():
            setattr(P, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(P, name, value)


def sample_parameters(rng):
    """Sample deliberately documented 1-sigma-like engineering ranges."""
    return {
        "rho": float(rng.uniform(1.18, 1.27)),
        "CL_alpha": float(rng.uniform(4.6, 5.2)),
        "Cm_alpha": float(rng.uniform(-0.66, -0.54)),
        "Cm_de": float(rng.uniform(-1.21, -0.99)),
        "CD0": float(rng.uniform(0.027, 0.033)),
        "payload_mass_max": float(rng.uniform(1.35, 1.65)),
        "payload_x_m": float(rng.uniform(-0.115, -0.093)),
    }


def robustness_configs(seed):
    sensor = {
        "enabled": True,
        "delay_s": 0.02,
        "sample_period_s": 0.02,
        "angle_noise_deg": 0.08,
        "rate_noise_deg_s": 0.30,
        "velocity_noise_mps": 0.05,
        "airspeed_noise_mps": 0.08,
        "acceleration_noise_mps2": 0.15,
        "angle_bias_deg": 0.10,
        "rate_bias_deg_s": 0.15,
        "airspeed_bias_mps": 0.05,
        "seed": seed + 1001,
    }
    actuator = {
        name: {
            "enabled": True,
            "time_constant_s": tau,
            "deadband": 0.0005,
            "noise_std": 0.0002,
            "seed": seed + offset,
        }
        for name, tau, offset in (
            ("elevator", 0.06, 101),
            ("aileron", 0.05, 102),
            ("rudder", 0.05, 103),
            ("tail", 0.08, 104),
        )
    }
    safety = {
        "enabled": True,
        "terminate_on_violation": False,
        "theta_limit_deg": 35.0,
        "phi_limit_deg": 50.0,
        "alpha_limit_deg": 12.0,
        "beta_limit_deg": 12.0,
        "min_airspeed_mps": 10.0,
        "max_airspeed_mps": 25.0,
        "max_estimator_step": 0.20,
        "estimator_ood_fraction": 0.20,
    }
    return sensor, actuator, safety


def _finite_metrics(metrics):
    return bool(metrics) and all(np.isfinite(float(value)) for value in metrics.values())


def _sampled_static_margin(payload, sampled):
    """Evaluate the sampled longitudinal static margin without mutating P."""
    payload_mass = float(sampled["payload_mass_max"]) * float(payload)
    total_mass = P.mass + payload_mass
    cg_shift = (payload_mass * (-float(sampled["payload_x_m"]))) / total_mass / P.c
    cm_alpha = float(sampled["Cm_alpha"]) + float(sampled["CL_alpha"]) * cg_shift
    return float(-cm_alpha / float(sampled["CL_alpha"]))


def _controller_record(metrics, status):
    """Keep validity, termination, and safety evidence in one auditable record."""
    finite = _finite_metrics(metrics)
    valid = bool(status.get("valid", False))
    supervisor = status.get("safety_supervisor", {})
    return {
        "valid": valid,
        "finite_metrics": finite,
        "metrics": metrics,
        "terminated_reason": status.get("terminated_reason"),
        "validity_detail": status.get("validity_detail"),
        "simulated_T": float(status.get("simulated_T", 0.0)),
        "max_abs_alpha_deg": float(status.get("max_abs_alpha_deg", 0.0)),
        "max_abs_beta_deg": float(status.get("max_abs_beta_deg", 0.0)),
        "min_airspeed_mps": float(status.get("min_airspeed_mps", 0.0)),
        "requested_T": float(status.get("requested_T", 0.0)),
        "completed_to_requested": bool(
            valid and float(status.get("simulated_T", 0.0)) + 1e-9
            >= float(status.get("requested_T", 0.0))),
        "supervisor": supervisor,
    }


def run_campaign(trials=20, seed_base=300000, T=15.0, output=None):
    rng = np.random.default_rng(seed_base)
    records = []
    for index in range(int(trials)):
        seed = seed_base + index
        payload = float(rng.uniform(0.2, 0.7))
        airspeed = float(rng.uniform(12.0, 20.0))
        gust = float(rng.uniform(0.5, 1.5))
        sampled = sample_parameters(rng)
        sensor, actuator, safety = robustness_configs(seed)
        record = {
            "trial": index,
            "seed": seed,
            "payload": payload,
            "airspeed": airspeed,
            "gust_intensity": gust,
            "parameters": sampled,
            "sampled_static_margin": _sampled_static_margin(payload, sampled),
        }
        record["positive_static_margin"] = bool(record["sampled_static_margin"] > 0.0)
        try:
            with temporary_parameter_set(sampled):
                conv_metrics, _, conv_status = run_conventional(
                    payload, airspeed, gust, seed=seed, T=T, change_t=4.0,
                    lateral_disturbance=True, gust_model="dryden",
                    sensor_config=sensor, actuator_config=actuator,
                    safety_config=safety)
                int_metrics, _, _, int_status = run_intelligent(
                    payload, airspeed, gust, seed=seed, T=T, change_t=4.0,
                    lateral_disturbance=True, debug=False,
                    gust_model="dryden", sensor_config=sensor,
                    actuator_config=actuator, safety_config=safety)
            record.update({
                "conventional": _controller_record(conv_metrics, conv_status),
                "intelligent": _controller_record(int_metrics, int_status),
            })
        except Exception as exc:  # failures are evidence, not dropped trials
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    def summarize(controller):
        completed = [r[controller] for r in records if controller in r]
        valid = [r for r in completed if r["valid"] and r["finite_metrics"]]
        metric_names = sorted({name for row in valid for name in row["metrics"]})
        quantiles = {}
        for name in metric_names:
            values = [float(row["metrics"][name]) for row in valid
                      if np.isfinite(float(row["metrics"][name]))]
            if values:
                quantiles[name] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(np.max(values)),
                }
        fallback = sum(bool(row["supervisor"]["fallback_active"])
                       for row in completed)
        invalid_reasons = {}
        for row in completed:
            if not row["valid"]:
                reason = row.get("terminated_reason") or "unknown"
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        first_reasons = {}
        for row in completed:
            reason = row["supervisor"].get("first_reason") or "none"
            first_reasons[reason] = first_reasons.get(reason, 0) + 1
        in_physical_envelope = [
            row for row in records
            if row.get("positive_static_margin") and controller in row
        ]
        valid_in_physical_envelope = [
            row[controller] for row in in_physical_envelope
            if row[controller]["valid"] and row[controller]["finite_metrics"]
        ]
        return {
            "trials": len(records),
            "completed": len(completed),
            "valid_finite": len(valid),
            "valid_finite_rate": len(valid) / max(len(records), 1),
            "fallback_count": fallback,
            "fallback_rate": fallback / max(len(records), 1),
            "invalid_reasons": invalid_reasons,
            "supervisor_first_reasons": first_reasons,
            "positive_static_margin_trials": len(in_physical_envelope),
            "valid_finite_in_positive_static_margin": len(valid_in_physical_envelope),
            "valid_finite_rate_in_positive_static_margin": (
                len(valid_in_physical_envelope) / max(len(in_physical_envelope), 1)),
            "metrics": quantiles,
        }

    result = {
        "campaign": "uncertainty_monte_carlo",
        "seed_base": int(seed_base),
        "trials": int(trials),
        "duration_s": float(T),
        "disturbance_model": "dryden",
        "lateral_disturbance": True,
        "uncertainty_ranges": {
            "rho": [1.18, 1.27],
            "CL_alpha": [4.6, 5.2],
            "Cm_alpha": [-0.66, -0.54],
            "Cm_de": [-1.21, -0.99],
            "CD0": [0.027, 0.033],
            "payload_mass_max": [1.35, 1.65],
            "payload_x_m": [-0.115, -0.093],
        },
        "conventional_summary": summarize("conventional"),
        "intelligent_summary": summarize("intelligent"),
        "records": records,
    }
    if output is None:
        output = ROOT / "uncertainty_monte_carlo_results.json"
    output = Path(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=300000)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_campaign(args.trials, args.seed_base, args.duration, args.output)
    print(json.dumps({
        "conventional": result["conventional_summary"],
        "intelligent": result["intelligent_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
