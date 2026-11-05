"""Matched-seed controller, oracle and ablation evaluation for paper tables."""
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
    "pitch_settling_time_s", "pitch_settled", "pitch_steady_state_error",
)


def run_config(name, payload, va, gust, seed, T, change_t, crosswind):
    if name == "conventional":
        return run_conventional(payload, va, gust, seed, T=T,
                                change_t=change_t,
                                lateral_disturbance=crosswind)
    options = {
        "nn": dict(payload_source="nn", tail_enabled=True, gain_schedule=True),
        "oracle": dict(payload_source="oracle", tail_enabled=True, gain_schedule=True),
        "estimator_off": dict(payload_source="off", tail_enabled=True, gain_schedule=True),
        "tail_off": dict(payload_source="nn", tail_enabled=False, gain_schedule=True),
        "tail_no_dither": dict(payload_source="nn", tail_enabled=True,
                                tail_dither=False, gain_schedule=True),
        "fixed_gain": dict(payload_source="off", tail_enabled=False, gain_schedule=False),
        "nn_pitch_gain": dict(payload_source="nn", trim_mode="tail", gain_schedule=True, pitch_gain_schedule=True),  # old design, kept as ablation
        "tail_lateral": dict(payload_source="nn", trim_mode="tail", gain_schedule=True, pitch_gain_schedule=False),
        "tail_only": dict(payload_source="nn", trim_mode="tail", gain_schedule=False),
        "elev_ff": dict(payload_source="nn", trim_mode="elevator_ff", gain_schedule=False),
        "oracle_tail_only": dict(payload_source="oracle", trim_mode="tail", gain_schedule=False),
        "oracle_elev_ff": dict(payload_source="oracle", trim_mode="elevator_ff", gain_schedule=False),
    }
    return run_intelligent(payload, va, gust, seed, T=T, change_t=change_t,
                           lateral_disturbance=crosswind, **options[name])


def evaluate(trials, payload, va, gust, T, change_t, crosswind):
    names = ("conventional", "nn", "oracle", "estimator_off", "tail_off",
             "tail_no_dither", "fixed_gain")
    values = {name: [] for name in names}
    reasons = {name: Counter() for name in names}
    for seed in range(trials):
        for name in names:
            result = run_config(name, payload, va, gust, seed, T, change_t, crosswind)
            metrics, status = result[0], result[-1]
            if status["valid"]:
                values[name].append([metrics[m] for m in METRICS])
            else:
                reasons[name][status.get("terminated_reason", "unknown")] += 1
    print(f"matched trials={trials}; payload={payload}; Va={va}; gust={gust}; "
          f"crosswind={crosswind}")
    for name in names:
        arr = np.asarray(values[name])
        print(f"\n{name}: valid={len(arr)}/{trials}, invalid={dict(reasons[name])}")
        if len(arr):
            for j, metric in enumerate(METRICS):
                sd = arr[:, j].std(ddof=1) if len(arr) > 1 else 0.0
                print(f"  {metric}: {arr[:, j].mean():.4f} +/- {sd:.4f}")
    if len(values["conventional"]) and len(values["nn"]):
        # Only compare paired trials where both flights were valid.
        paired = []
        for seed in range(trials):
            c = run_config("conventional", payload, va, gust, seed, T, change_t, crosswind)
            n = run_config("nn", payload, va, gust, seed, T, change_t, crosswind)
            if c[-1]["valid"] and n[-1]["valid"]:
                paired.append(([c[0][m] for m in METRICS], [n[0][m] for m in METRICS]))
        if len(paired) >= 5:
            print(f"\npaired-valid={len(paired)}")
            for j, metric in enumerate(METRICS):
                diff = np.asarray([x[0][j] - x[1][j] for x in paired])
                p = wilcoxon(diff).pvalue if np.any(diff) else 1.0
                print(f"  {metric}: diff={diff.mean():.4f}, Wilcoxon p={p:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--payload", type=float, default=0.5)
    ap.add_argument("--airspeed", type=float, default=15.0)
    ap.add_argument("--gust", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--change-t", type=float, default=5.0)
    ap.add_argument("--crosswind", action="store_true")
    ap.add_argument("--stress-case", action="store_true",
                    help="allow payload > 0.74; labels it unstable-airframe stress")
    args = ap.parse_args()
    if args.payload > 0.74 and not args.stress_case:
        ap.error("payload > 0.74 is an unstable-airframe stress case; use --stress-case")
    evaluate(args.trials, args.payload, args.airspeed, args.gust, args.T,
             args.change_t, args.crosswind)
