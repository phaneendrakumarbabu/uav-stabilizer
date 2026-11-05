"""Matched-seed controller, oracle and ablation evaluation for paper tables.

Stores each seed's result once and reuses it for paired comparisons.
The aircraft/controller implementation is not modified.
"""
import argparse
from collections import Counter
import numpy as np
from scipy.stats import wilcoxon
from final_comparison import run_conventional, run_intelligent

METRICS = (
    "max_pitch_dev", "max_roll_dev", "max_yaw_dev",
    "rms_pitch_error", "rms_roll_error", "rms_yaw_error",
    "itae_pitch", "itae_roll", "itae_yaw",
    "rms_altitude_error", "max_altitude_drop_m",
    "mean_elev", "mean_aileron", "mean_rudder",
    "rms_elev", "rms_aileron", "rms_rudder",
    "elev_activity", "aileron_activity", "rudder_activity",
    "elev_saturation_s", "aileron_saturation_s", "rudder_saturation_s",
    "tail_mean_abs", "tail_rms", "tail_activity", "mean_drag",
)
CONFIGS = ("conventional", "nn", "oracle", "estimator_off", "tail_off", "tail_no_dither", "fixed_gain")


def run_config(name, payload, va, gust, seed, T, change_t, crosswind):
    if name == "conventional":
        return run_conventional(payload, va, gust, seed, T=T, change_t=change_t,
                                lateral_disturbance=crosswind)
    options = {
        "nn": dict(payload_source="nn", tail_enabled=True, gain_schedule=True),
        "oracle": dict(payload_source="oracle", tail_enabled=True, gain_schedule=True),
        "estimator_off": dict(payload_source="off", tail_enabled=True, gain_schedule=True),
        "tail_off": dict(payload_source="nn", tail_enabled=False, gain_schedule=True),
        "tail_no_dither": dict(payload_source="nn", tail_enabled=True, tail_dither=False, gain_schedule=True),
        "fixed_gain": dict(payload_source="off", tail_enabled=False, gain_schedule=False),
    }
    if name not in options:
        raise ValueError(f"Unknown configuration: {name}")
    return run_intelligent(payload, va, gust, seed, T=T, change_t=change_t,
                           lateral_disturbance=crosswind, **options[name])


def paired_test(results, left_name, right_name, metric):
    left, right = results[left_name], results[right_name]
    seeds = sorted(s for s in left if s in right and left[s]["valid"] and right[s]["valid"])
    diffs = np.asarray([left[s]["metrics"][metric] - right[s]["metrics"][metric] for s in seeds], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 5:
        return len(diffs), None, None
    p = 1.0 if np.allclose(diffs, 0.0) else float(wilcoxon(diffs).pvalue)
    return len(diffs), float(diffs.mean()), p


def evaluate(trials, payload, va, gust, T, change_t, crosswind):
    results = {name: {} for name in CONFIGS}
    reasons = {name: Counter() for name in CONFIGS}
    # Each configuration/seed is run once; stored results are reused below.
    for seed in range(trials):
        for name in CONFIGS:
            result = run_config(name, payload, va, gust, seed, T, change_t, crosswind)
            metrics, status = result[0], result[-1]
            results[name][seed] = {
                "valid": bool(status["valid"]),
                "metrics": {m: float(metrics[m]) for m in METRICS},
            }
            if not status["valid"]:
                reasons[name][status.get("terminated_reason", "unknown")] += 1

    print(f"matched trials={trials}; payload={payload}; Va={va}; gust={gust}; T={T}; change_t={change_t}; crosswind={crosswind}")
    for name in CONFIGS:
        valid = [r["metrics"] for r in results[name].values() if r["valid"]]
        print(f"\n{name}: valid={len(valid)}/{trials}, invalid={dict(reasons[name])}")
        for metric in METRICS:
            vals = np.asarray([r[metric] for r in valid], dtype=float)
            if len(vals):
                sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
                print(f"  {metric}: {vals.mean():.4f} +/- {sd:.4f}")

    for left, right in (("conventional", "nn"), ("nn", "tail_no_dither"), ("nn", "oracle"), ("nn", "tail_off")):
        print(f"\nPAIRED COMPARISON: {left} - {right} (positive means left metric is larger)")
        for metric in METRICS:
            n, diff, p = paired_test(results, left, right, metric)
            if diff is None:
                print(f"  {metric}: insufficient paired valid trials (n={n})")
            else:
                print(f"  {metric}: diff={diff:.4f}, Wilcoxon p={p:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--payload", type=float, default=0.5)
    ap.add_argument("--airspeed", type=float, default=15.0)
    ap.add_argument("--gust", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--change-t", type=float, default=5.0)
    ap.add_argument("--crosswind", action="store_true")
    ap.add_argument("--stress-case", action="store_true", help="allow payload > 0.74; labels it unstable-airframe stress")
    args = ap.parse_args()
    if args.trials < 5:
        ap.error("--trials must be at least 5 for paired statistical tests")
    if args.payload > 0.74 and not args.stress_case:
        ap.error("payload > 0.74 is an unstable-airframe stress case; use --stress-case")
    evaluate(args.trials, args.payload, args.airspeed, args.gust, args.T, args.change_t, args.crosswind)
