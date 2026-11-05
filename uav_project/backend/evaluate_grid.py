"""Reproducible matched-seed grid evaluation for paper tables.

Main results use only payload <= 0.74 (positive static margin). Payloads above
that limit are optional stress cases and are labelled separately. Every run,
seed, configuration, metric, invalid reason, mean/SD and paired Wilcoxon
value is saved so the tables can be regenerated exactly.
"""
import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import aircraft_params as P
from evaluate_matched import METRICS, run_config


CONFIGS = ("conventional", "nn", "oracle", "estimator_off", "tail_off",
           "tail_no_dither", "fixed_gain")


def _run_seed(task):
    payload, gust, va, seed, T, change_t, crosswind, config_names = task
    regime = ("main_stable_margin"
              if payload <= P.Main_results_payload_max
              else "unstable_airframe_stress")
    records = []
    condition = {"payload": payload, "gust": gust, "airspeed": va,
                 "seed": seed, "regime": regime}
    # ProcessPool workers on Windows import this module afresh. Pass the
    # selected list in the task so custom grids cannot silently use the
    # seven-config default from the worker process.
    for config in config_names:
        result = run_config(config, payload, va, gust, seed, T, change_t,
                            crosswind)
        metrics, status = result[0], result[-1]
        record = {**condition, "config": config,
                  "valid": bool(status["valid"]),
                  "terminated_reason": status.get("terminated_reason"),
                  "simulated_T": status.get("simulated_T")}
        record["metrics"] = {
            key: float(metrics.get(key, np.nan)) for key in METRICS
        }
        records.append(record)
    return records


def _safe_pvalue(diffs):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 5 or not np.any(np.abs(diffs) > 0.0):
        return 1.0 if len(diffs) else float("nan")
    return float(wilcoxon(diffs).pvalue)


def evaluate_grid(trials=30, T=20.0, change_t=5.0, crosswind=False,
                  output_dir="results_grid", include_stress=False, workers=1,
                  payloads=None, gusts=None, airspeeds=None, configs=None,
                  seed_base=100000):
    global CONFIGS
    if configs:
        CONFIGS = tuple(configs)
    payloads = list(payloads) if payloads else [0.0, 0.2, 0.5, 0.7]
    if include_stress:
        payloads += [0.8, 1.0]
    gusts = list(gusts) if gusts else [0.0, 0.5, 1.0, 1.5, 2.0]
    airspeeds = list(airspeeds) if airspeeds else [12.0, 15.0, 20.0]
    # evaluation seeds are disjoint from estimator-training seeds (>= 1,000,000)
    seeds = list(range(seed_base, seed_base + trials))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trials": trials, "seeds": seeds, "T": T, "change_t": change_t,
        "crosswind": bool(crosswind), "payloads": payloads,
        "gusts": gusts, "airspeeds": airspeeds,
        "workers": workers,
        "main_payload_max": P.Main_results_payload_max,
        "configs": CONFIGS, "metrics": METRICS,
        "code": "evaluate_grid.py + evaluate_matched.py",
    }
    (out / "grid_config.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")

    raw_path = out / "raw_runs.jsonl"
    selected_configs = tuple(CONFIGS)
    tasks = [(payload, gust, va, seed, T, change_t, crosswind, selected_configs)
             for payload in payloads for gust in gusts
             for va in airspeeds for seed in seeds]
    records = []
    total = len(tasks)
    with raw_path.open("w", encoding="utf-8") as raw_file:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(_run_seed, tasks, chunksize=1)
                for done, seed_records in enumerate(iterator, start=1):
                    for record in seed_records:
                        raw_file.write(json.dumps(record, allow_nan=True) + "\n")
                        records.append(record)
                    if done % max(1, total // 20) == 0:
                        print(f"grid conditions {done}/{total}")
        else:
            for done, seed_records in enumerate(map(_run_seed, tasks), start=1):
                for record in seed_records:
                    raw_file.write(json.dumps(record, allow_nan=True) + "\n")
                    records.append(record)
                if done % max(1, total // 20) == 0:
                    print(f"grid conditions {done}/{total}")

    summary_rows = []
    for payload in payloads:
        regime = ("main_stable_margin"
                  if payload <= P.Main_results_payload_max
                  else "unstable_airframe_stress")
        for gust in gusts:
            for va in airspeeds:
                condition_records = [r for r in records
                                     if r["payload"] == payload
                                     and r["gust"] == gust
                                     and r["airspeed"] == va]
                by_config = {name: [r for r in condition_records
                                    if r["config"] == name]
                             for name in CONFIGS}
                for config in CONFIGS:
                    rows = by_config[config]
                    valid_rows = [r for r in rows if r["valid"]]
                    reasons = Counter(r["terminated_reason"] or "unknown"
                                      for r in rows if not r["valid"])
                    for metric in METRICS:
                        vals = np.asarray([r["metrics"][metric]
                                           for r in valid_rows], dtype=float)
                        vals = vals[np.isfinite(vals)]
                        conventional = {
                            r["seed"]: r["metrics"][metric]
                            for r in by_config["conventional"] if r["valid"]
                        }
                        current = {
                            r["seed"]: r["metrics"][metric]
                            for r in rows if r["valid"]
                        }
                        paired_diffs = [conventional[s] - current[s]
                                         for s in sorted(set(conventional) & set(current))]
                        summary_rows.append({
                            "payload": payload, "gust": gust, "airspeed": va,
                            "regime": regime, "config": config,
                            "metric": metric, "n_valid": int(len(vals)),
                            "invalid_count": int(len(rows) - len(valid_rows)),
                            "invalid_reasons": json.dumps(dict(reasons), sort_keys=True),
                            "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "wilcoxon_p_vs_conventional": _safe_pvalue(paired_diffs)
                            if config != "conventional" else 1.0,
                        })

    summary_path = out / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    import hashlib
    expected = len(tasks) * len(selected_configs)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest.update({"expected_runs": expected, "completed_runs": len(records),
                     "grid_complete": len(records) == expected,
                     "seed_base": seed_base,
                     "sha256": {"raw_runs.jsonl": sha(raw_path),
                                **{f: sha(f) for f in ("final_comparison.py", "adaptive_system.py",
                                   "evaluate_matched.py", "evaluate_grid.py", "payload_estimator.joblib")}}})
    (out / "grid_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {out / 'grid_config.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--change-t", type=float, default=5.0)
    parser.add_argument("--crosswind", action="store_true")
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--payloads", type=float, nargs="+")
    parser.add_argument("--gusts", type=float, nargs="+")
    parser.add_argument("--airspeeds", type=float, nargs="+")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--seed-base", type=int, default=100000)
    parser.add_argument("--output-dir", default="results_grid")
    args = parser.parse_args()
    if args.trials < 30:
        parser.error("at least 30 matched seeds are required")
    if args.workers < 1:
        parser.error("workers must be >= 1")
    evaluate_grid(args.trials, args.T, args.change_t, args.crosswind,
                  args.output_dir, args.include_stress, args.workers,
                  args.payloads, args.gusts, args.airspeeds, args.configs, args.seed_base)
