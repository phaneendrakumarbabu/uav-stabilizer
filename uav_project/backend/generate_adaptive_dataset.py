"""Collect estimator windows from the closed adaptive loop (payload label never given to the controller).
Stage 0: no estimator loaded (controller runs with payload estimate 0) -> train stage-0 estimator.
Stage 1: loop driven by the stage-0 estimator -> final training set (deployment distribution).
Payload is sampled only inside the claimed envelope (<= Main_results_payload_max).
Training run seeds are >= 1,000,000; evaluation scripts use seeds >= 100,000 and
< 1,000,000, so train/eval seeds can never coincide. Per-window metadata is saved
for held-out-condition validation."""
import argparse
from pathlib import Path
import numpy as np
import aircraft_params as P
from final_comparison import run_intelligent
from adaptive_system import ESTIMATOR_FEATURE_COUNT

def generate(n_runs=400, seed=17, T=20.0):
    rng = np.random.default_rng(seed)
    rows, labels, groups, meta = [], [], [], []
    invalid = 0
    pmax = P.Main_results_payload_max
    grid_p, grid_g = (0.0, 0.2, 0.5, pmax), (0.0, 0.5, 1.0, 1.5, 2.0)
    corners = [(p, g) for p in grid_p for g in grid_g] * 3   # envelope corners first (zero payload + strong gust is a required negative class)
    for i in range(n_runs):
        if i < len(corners):
            payload, gust = corners[i]
        else:
            payload, gust = float(rng.uniform(0.0, pmax)), float(rng.uniform(0.0, 2.0))
        va = float(rng.uniform(P.Va_train_min, P.Va_train_max))
        change_t = float(rng.uniform(4.0, 12.0))
        lateral = bool(rng.random() < 0.25)
        _, _, _, status = run_intelligent(payload, va, gust, seed=int(rng.integers(1_000_000, 2**31 - 1)),
                                          T=T, change_t=change_t, lateral_disturbance=lateral, debug=True)
        invalid += int(not status["valid"])
        for t, f in status.get("estimator_features", []):
            rows.append(f)
            # window ends at state time t; last step used payload active at t - DT
            labels.append(payload if (t - 0.02) >= change_t else 0.0)
            groups.append(i)
            meta.append((payload, gust, va, t - change_t))
        if (i + 1) % 50 == 0:
            print(f"adaptive runs {i + 1}/{n_runs}", flush=True)
    X = np.asarray(rows, dtype=float); y = np.asarray(labels, dtype=float)
    if X.ndim != 2 or X.shape[1] != ESTIMATOR_FEATURE_COUNT:
        raise RuntimeError(f"unexpected dataset shape {X.shape}")
    print(f"Collected {len(y)} windows from {n_runs} flights; invalid runs={invalid}")
    return X, y, np.asarray(groups, dtype=int), np.asarray(meta, dtype=float)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="payload_dataset_adaptive.npz")
    a = ap.parse_args()
    X, y, g, m = generate(a.runs, a.seed)
    np.savez(Path(__file__).resolve().with_name(a.out), X=X, y=y, groups=g,
             run_payload=m[:, 0], run_gust=m[:, 1], run_va=m[:, 2], t_rel=m[:, 3])
    print("Saved", a.out, X.shape)
