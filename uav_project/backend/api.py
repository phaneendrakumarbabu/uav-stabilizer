"""
FastAPI backend. Thin wrapper around final_comparison.py -- no
reimplementation of any physics/control logic here.

Run with:  uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import geometry
import aircraft_params as P
from final_comparison import DT, run_conventional, run_intelligent

app = FastAPI(title="UAV Stability Simulation API")

_default_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_allowed_origins = [item.strip() for item in os.getenv(
    "UAV_ALLOWED_ORIGINS", ",".join(_default_origins)
).split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimRequest(BaseModel):
    mode: Literal["conventional", "intelligent", "compare"]
    payload_level: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    airspeed: float = Field(ge=P.Va_train_min, le=P.Va_train_max,
                            allow_inf_nan=False)
    gust_intensity: float = Field(ge=0.0, le=2.0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)
    change_t: float = Field(default=10.0, ge=0.0, le=120.0, allow_inf_nan=False)
    lateral_disturbance: bool = False
    gust_model: Literal["ou", "dryden"] = "ou"
    sensor_realism: bool = False
    actuator_realism: bool = False
    safety_supervisor: bool = False
    T: float = Field(default=40.0, gt=0.0, le=120.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_time_order(self) -> "SimRequest":
        for name in ("payload_level", "airspeed", "gust_intensity", "change_t", "T"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.change_t > self.T:
            raise ValueError("change_t must be <= T")
        if self.T < DT:
            raise ValueError(f"T must be >= integration step {DT:.3f} s")
        return self


def _log_to_dict(log_arr: Any, mode: str) -> dict[str, Any]:
    log_arr = np.asarray(log_arr)
    if log_arr.size == 0:
        result: dict[str, Any] = {
            "t": [], "theta_deg": [], "phi_deg": [], "psi_deg": [],
            "de_deg": [], "da_deg": [], "dr_deg": [], "alt_m": [],
            "drag": [], "pn_m": [], "pe_m": [], "pd_m": [],
            "theta_ref_deg": [],
        }
        if mode == "intelligent":
            result.update({
                "tail_inc_deg": [], "pl_est": [], "theta_ref_deg": []
            })
        return result
    expected_columns = 15 if mode == "intelligent" else 13
    if log_arr.ndim != 2 or log_arr.shape[1] != expected_columns:
        raise ValueError(
            f"{mode} log has shape {log_arr.shape}; "
            f"expected {expected_columns} columns"
        )
    result = {
        "t": log_arr[:, 0].tolist(),
        "theta_deg": log_arr[:, 1].tolist(),
        "phi_deg": log_arr[:, 2].tolist(),
        "psi_deg": log_arr[:, 3].tolist(),
        "de_deg": log_arr[:, 4].tolist(),
        "da_deg": log_arr[:, 5].tolist(),
        "dr_deg": log_arr[:, 6].tolist(),
        "alt_m": log_arr[:, 7].tolist(),
        "drag": log_arr[:, 8].tolist(),
    }
    if mode == "conventional":
        result.update({
            "pn_m": log_arr[:, 9].tolist(),
            "pe_m": log_arr[:, 10].tolist(),
            "pd_m": log_arr[:, 11].tolist(),
            "theta_ref_deg": log_arr[:, 12].tolist(),
        })
    else:
        result.update({
            "tail_inc_deg": log_arr[:, 9].tolist(),
            "pl_est": log_arr[:, 10].tolist(),
            "pn_m": log_arr[:, 11].tolist(),
            "pe_m": log_arr[:, 12].tolist(),
            "pd_m": log_arr[:, 13].tolist(),
            "theta_ref_deg": log_arr[:, 14].tolist(),
        })
    return result


def _save_run_artifacts(mode: str, req: SimRequest, metrics: dict,
                        log: dict, status: dict) -> str:
    """Save the actual simulation log, configuration, metrics and status."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = results_dir / f"{run_id}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=False)

    series = {
        key: values for key, values in log.items()
        if key != "pl_est_t"
        and isinstance(values, list)
        and all(not isinstance(v, (dict, list)) for v in values)
    }
    lengths = {len(values) for values in series.values()}
    if len(lengths) > 1:
        raise ValueError("Log columns have inconsistent lengths")
    if series:
        with (run_dir / "simulation_log.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(series.keys())
            writer.writerows(zip(*series.values()))

    estimates = log.get("pl_est_t", [])
    if estimates:
        with (run_dir / "nn_estimates.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("t", "pl_est"))
            writer.writerows((item["t"], item["pl_est"]) for item in estimates)

    with (run_dir / "configuration.json").open("w", encoding="utf-8") as f:
        json.dump(req.model_dump(mode="json"), f, indent=2)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)
    with (run_dir / "status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, allow_nan=True)
    return str(run_dir)


def _run(mode, req: SimRequest):
    sensor_config = None
    actuator_config = None
    safety_config = None
    if req.sensor_realism:
        sensor_config = {
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
            "seed": req.seed + 1001,
        }
    if req.actuator_realism:
        actuator_config = {
            "elevator": {"enabled": True, "time_constant_s": 0.06,
                         "deadband": 0.0005, "noise_std": 0.0002},
            "aileron": {"enabled": True, "time_constant_s": 0.05,
                        "deadband": 0.0005, "noise_std": 0.0002},
            "rudder": {"enabled": True, "time_constant_s": 0.05,
                       "deadband": 0.0005, "noise_std": 0.0002},
            "tail": {"enabled": True, "time_constant_s": 0.08,
                     "deadband": 0.0005, "noise_std": 0.0002},
        }
    if req.safety_supervisor:
        safety_config = {"enabled": True, "terminate_on_violation": False}
    if mode == "conventional":
        metrics, log, status = run_conventional(
            payload_level=req.payload_level,
            Va_init=req.airspeed,
            gust_intensity=req.gust_intensity,
            seed=req.seed,
            T=req.T,
            change_t=req.change_t,
            lateral_disturbance=bool(req.lateral_disturbance),
            gust_model=req.gust_model,
            sensor_config=sensor_config,
            actuator_config=actuator_config,
            safety_config=safety_config,
        )
        return metrics, _log_to_dict(log, mode), status
    elif mode == "intelligent":
        metrics, log, pl_est_t, status = run_intelligent(
            payload_level=req.payload_level,
            Va_init=req.airspeed,
            gust_intensity=req.gust_intensity,
            seed=req.seed,
            T=req.T,
            change_t=req.change_t,
            lateral_disturbance=bool(req.lateral_disturbance),
            debug=False,
            payload_source="nn",
            tail_enabled=True,
            gain_schedule=True,
            gust_model=req.gust_model,
            sensor_config=sensor_config,
            actuator_config=actuator_config,
            safety_config=safety_config,
        )
        log_dict = _log_to_dict(log, mode)
        log_dict["pl_est_t"] = [{"t": t, "pl_est": v} for t, v in pl_est_t]
        return metrics, log_dict, status
    else:
        raise ValueError(f"unknown mode: {mode}")


@app.post("/api/simulate")
def simulate(req: SimRequest):
    if req.mode == "compare":
        raise HTTPException(status_code=422, detail="use /api/simulate/compare for compare mode")
    try:
        metrics, log, status = _run(req.mode, req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    saved_to = _save_run_artifacts(req.mode, req, metrics, log, status)
    return {
        "metrics": metrics,
        "log": log,
        "status": status,
        "saved_to": saved_to,
    }


@app.post("/api/simulate/compare")
def simulate_compare(req: SimRequest):
    try:
        conv_metrics, conv_log, conv_status = _run("conventional", req)
        int_metrics, int_log, int_status = _run("intelligent", req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    conv_saved_to = _save_run_artifacts(
        "conventional", req, conv_metrics, conv_log, conv_status
    )
    int_saved_to = _save_run_artifacts(
        "intelligent", req, int_metrics, int_log, int_status
    )
    return {
        "conventional": {
            "metrics": conv_metrics,
            "log": conv_log,
            "status": conv_status,
            "saved_to": conv_saved_to,
        },
        "intelligent": {
            "metrics": int_metrics,
            "log": int_log,
            "status": int_status,
            "saved_to": int_saved_to,
        },
    }


@app.get("/api/geometry")
def get_geometry():
    return {
        "real_from_backend": geometry.real_from_backend(),
        "cosmetic_only": geometry.cosmetic_only(),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

