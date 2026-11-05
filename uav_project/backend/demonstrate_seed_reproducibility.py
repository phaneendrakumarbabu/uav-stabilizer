"""Create an auditable proof that the seed controls the gust realization.

This is a demonstration artifact, not a controller-performance campaign.
It runs the same scenario twice with the same seed and once with a different
seed, then compares the returned logs and metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from final_comparison import run_conventional, run_intelligent


ROOT = Path(__file__).resolve().parent
SCENARIO = {
    "payload": 0.7,
    "airspeed_mps": 20.0,
    "gust_intensity": 1.5,
    "T_s": 15.0,
    "change_t_s": 4.0,
    "crosswind": False,
    "gust_model": "ou",
}


def _hash_log(log: np.ndarray) -> str:
    array = np.asarray(log, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _run(controller: str, seed: int):
    if controller == "conventional":
        metrics, log, status = run_conventional(
            SCENARIO["payload"], SCENARIO["airspeed_mps"],
            SCENARIO["gust_intensity"], seed=seed,
            T=SCENARIO["T_s"], change_t=SCENARIO["change_t_s"],
            lateral_disturbance=SCENARIO["crosswind"],
            gust_model=SCENARIO["gust_model"],
        )
    else:
        metrics, log, _, status = run_intelligent(
            SCENARIO["payload"], SCENARIO["airspeed_mps"],
            SCENARIO["gust_intensity"], seed=seed,
            T=SCENARIO["T_s"], change_t=SCENARIO["change_t_s"],
            lateral_disturbance=SCENARIO["crosswind"],
            gust_model=SCENARIO["gust_model"],
        )
    selected = {
        name: float(value)
        for name, value in metrics.items()
        if name in {
            "max_pitch_dev", "rms_pitch_error", "max_roll_dev",
            "rms_roll_error", "max_yaw_dev", "rms_yaw_error",
            "pitch_settling_time_s", "pitch_settled",
        }
    }
    return {
        "controller": controller,
        "seed": int(seed),
        "status_random_seed": int(status["random_seed"]),
        "valid": bool(status["valid"]),
        "terminated_reason": status.get("terminated_reason"),
        "simulated_T": float(status["simulated_T"]),
        "log_sha256": _hash_log(log),
        "metrics": selected,
    }


def main() -> None:
    records = []
    for controller in ("conventional", "intelligent"):
        records.append(_run(controller, 42))
        records.append(_run(controller, 42))
        records.append(_run(controller, 43))

    by_controller = {}
    for row in records:
        by_controller.setdefault(row["controller"], []).append(row)

    assertions = {}
    for controller, rows in by_controller.items():
        same_seed = rows[0]["seed"] == rows[1]["seed"] == 42
        same_hash = rows[0]["log_sha256"] == rows[1]["log_sha256"]
        same_metrics = rows[0]["metrics"] == rows[1]["metrics"]
        different_seed = rows[2]["seed"] == 43
        different_hash = rows[0]["log_sha256"] != rows[2]["log_sha256"]
        assertions[f"{controller}_same_seed_repeats_exactly"] = bool(
            same_seed and same_hash and same_metrics)
        assertions[f"{controller}_different_seed_changes_log"] = bool(
            different_seed and different_hash)

    assertions["matched_controller_seed_is_same"] = bool(
        records[0]["seed"] == records[3]["seed"] == 42
        and records[0]["status_random_seed"] == records[3]["status_random_seed"])

    result = {
        "scenario": SCENARIO,
        "purpose": "Demonstrate deterministic replay for a fixed seed and different gust realization for a different seed.",
        "records": records,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
        "interpretation": (
            "The visible scenario parameters define disturbance strength and the aircraft condition. "
            "The seed defines the exact stochastic gust realization. Same seed reproduces the exact log; "
            "a different seed creates a different gust realization. Compare mode uses the same seed for both controllers."
        ),
    }

    json_path = ROOT / "seed_reproducibility_evidence.json"
    md_path = ROOT / "seed_reproducibility_evidence.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Seed reproducibility evidence",
        "",
        "This artifact uses the real simulator with the same payload, airspeed, gust intensity, duration, payload-change time, and crosswind setting.",
        "",
        "| Controller | Seed | Valid | Log SHA-256 (first 16) | RMS pitch error | Max pitch deviation |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['controller']} | {row['seed']} | {row['valid']} | "
            f"`{row['log_sha256'][:16]}` | "
            f"{row['metrics']['rms_pitch_error']:.6f} | "
            f"{row['metrics']['max_pitch_dev']:.6f} |"
        )
    lines += [
        "",
        "## Checks",
        "",
        *[f"- `{name}`: **{'PASS' if value else 'FAIL'}**" for name, value in assertions.items()],
        "",
        "## Conclusion",
        "",
        result["interpretation"],
        "",
        "The conventional and intelligent runs with seed 42 are a matched comparison. The dashboard now exposes the seed field, so entering the same seed reproduces the same scenario.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), **result}, indent=2))


if __name__ == "__main__":
    main()
