"""Matched-seed repeated-trial summary for paper tables."""
import argparse
from collections import Counter
import numpy as np
from scipy.stats import wilcoxon

from final_comparison import run_conventional, run_intelligent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--payload", type=float, default=0.5)
    parser.add_argument("--airspeed", type=float, default=15.0)
    parser.add_argument("--gust", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=20.0)
    args = parser.parse_args()

    names = ("max_pitch_dev", "max_roll_dev", "rms_pitch_error",
             "itae_pitch", "rms_elev", "elev_saturation_s", "mean_drag")
    rows = {"conventional": [], "intelligent": []}
    valid = {"conventional": 0, "intelligent": 0}
    invalid_reasons = {"conventional": Counter(), "intelligent": Counter()}
    paired = []
    for seed in range(args.trials):
        c = run_conventional(args.payload, args.airspeed, args.gust, seed, T=args.T)
        i = run_intelligent(args.payload, args.airspeed, args.gust, seed, T=args.T)
        if c[-1]["valid"] and i[-1]["valid"]:
            paired.append(([c[0][name] for name in names],
                           [i[0][name] for name in names]))
        for key, result in (("conventional", c), ("intelligent", i)):
            metrics = result[0]
            status = result[-1]
            valid[key] += int(status["valid"])
            if status["valid"]:
                rows[key].append([metrics[name] for name in names])
            else:
                invalid_reasons[key][status.get("terminated_reason", "unknown")] += 1

    print(f"matched trials={args.trials}, payload={args.payload}, Va={args.airspeed}, gust={args.gust}")
    for key in rows:
        values = np.asarray(rows[key])
        print(f"{key}: valid={valid[key]}/{args.trials}, "
              f"invalid={args.trials - valid[key]} {dict(invalid_reasons[key])}")
        if len(values):
            for j, name in enumerate(names):
                sd = values[:, j].std(ddof=1) if len(values) > 1 else 0.0
                print(f"  {name}: mean={values[:, j].mean():.4f} std={sd:.4f}")
        else:
            print("  no valid flights; metrics omitted")
    if len(paired) >= 5:
        cvals = np.asarray([p[0] for p in paired])
        ivals = np.asarray([p[1] for p in paired])
        print(f"paired valid flights={len(paired)}")
        for j, name in enumerate(names):
            diff = cvals[:, j] - ivals[:, j]
            try:
                p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            print(f"  {name}: paired mean diff={diff.mean():.4f}, "
                  f"Wilcoxon p={p_value:.4g}")


if __name__ == "__main__":
    main()
