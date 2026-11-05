"""Operational holdout test for payload estimation.

These intermediate payload/gust/airspeed values are intentionally different
from the exact corner grid used during adaptive-dataset collection. Results
are saved with every condition and seed instead of being folded into training.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from final_comparison import run_intelligent


def evaluate(seeds=5, T=20.0, change_t=5.0, output="estimator_holdout.csv",
             include_stress=False):
    payloads = [0.10, 0.35, 0.65, 0.70]
    if include_stress:
        payloads.append(0.90)
    gusts = (0.25, 0.75, 1.25, 1.75)
    airspeeds = (12.5, 17.5, 19.5)
    rows = []
    for payload in payloads:
        for gust in gusts:
            for va in airspeeds:
                estimates, valid_estimates, invalid = [], [], 0
                for seed in range(seeds):
                    _, _, online, status = run_intelligent(
                        payload, va, gust, seed=seed, T=T,
                        change_t=change_t, debug=False)
                    invalid += int(not status["valid"])
                    post = [v for t, v in online if t >= change_t]
                    estimate = float(np.mean(post[-3:])) if post else 0.0
                    estimates.append(estimate)
                    if status["valid"]:
                        valid_estimates.append(estimate)
                arr = np.asarray(estimates)
                valid_arr = np.asarray(valid_estimates)
                rows.append({
                    "payload": payload, "gust": gust, "airspeed": va,
                    "n": seeds, "invalid": invalid,
                    "mean_estimate": float(arr.mean()),
                    "sd_estimate": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "valid_n": len(valid_arr),
                    "valid_mae": (float(np.mean(np.abs(valid_arr - payload)))
                                  if len(valid_arr) else float("nan")),
                    "invalid_rate": invalid / seeds,
                    "estimates": ";".join(f"{x:.6f}" for x in arr),
                })
    out = Path(output)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out} ({len(rows)} held-out conditions)")
    valid_mae = [r["valid_mae"] for r in rows if np.isfinite(r["valid_mae"])]
    print(f"Valid-flight MAE: {np.mean(valid_mae):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--change-t", type=float, default=5.0)
    parser.add_argument("--output", default="estimator_holdout.csv")
    parser.add_argument("--include-stress", action="store_true")
    args = parser.parse_args()
    evaluate(args.seeds, args.T, args.change_t, args.output, args.include_stress)
