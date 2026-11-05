"""Evaluate online payload-estimator behaviour by payload/gust bin.

This reports false alarms, estimate error, detection rate, and invalid runs;
it is intentionally separate from controller-performance evaluation.
"""
import argparse
import numpy as np

from final_comparison import run_intelligent


def evaluate(seeds=10, T=20.0, change_t=5.0):
    rows = []
    for payload in (0.0, 0.2, 0.5, 0.8, 1.0):
        for gust in (0.0, 0.5, 1.0, 1.5, 2.0):
            final = []
            delays = []
            invalid = 0
            for seed in range(seeds):
                _, _, estimates, status = run_intelligent(
                    payload, 15.0, gust, seed=seed, T=T,
                    change_t=change_t, debug=True)
                if not status["valid"]:
                    invalid += 1
                values = [(t, v) for t, v in estimates if t >= change_t]
                final_value = float(np.mean([v for _, v in values[-3:]])) if values else 0.0
                final.append(final_value)
                if payload > 0.0:
                    threshold = 0.5 * payload
                    hit = next((t for t, v in values if v >= threshold), None)
                    if hit is not None:
                        delays.append(hit - change_t)
            arr = np.asarray(final)
            if payload == 0.0:
                detection = float(np.mean(arr >= 0.20))
                mae = float(np.mean(arr))
            else:
                detection = float(np.mean(arr >= 0.5 * payload))
                mae = float(np.mean(np.abs(arr - payload)))
            rows.append((payload, gust, mae, detection,
                         float(np.mean(delays)) if delays else np.nan, invalid))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    print("payload gust MAE/false-alarm  detection  delay_s  invalid")
    for row in evaluate(seeds=args.seeds):
        print(f"{row[0]:.1f} {row[1]:.1f} {row[2]:.3f} {row[3]:.2f} "
              f"{row[4]:.2f} {row[5]}")
