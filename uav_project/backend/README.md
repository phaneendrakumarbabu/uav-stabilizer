# UAV Stability Simulation - Backend

Python backend implementing 6-DOF flight dynamics, conventional PID control,
and an adaptive controller with payload estimation, morphing-tail trim
compensation, and gain scheduling.

## Setup

Use Python 3.11–3.13 from this `backend/` folder. Install the exact pinned
versions so the serialized estimator remains compatible:

```bash
python -m pip install -r requirements.txt
```

Generate and train the version-matched estimator artifacts:

```bash
python generate_dataset_v4.py 5000
python train_payload_estimator.py
# collect causal windows from the adaptive controller and retrain deployment
python generate_adaptive_dataset.py --runs 400
python train_payload_estimator.py payload_dataset_adaptive.npz --output payload_estimator.joblib
```

Regenerate both artifacts whenever the dynamics or estimator features change.

## Verification

```bash
python trim.py
python geometry.py
python adaptive_system.py
python -m unittest -v test_project.py
python final_comparison.py
python evaluate_trials.py --trials 20
python evaluate_estimator.py --seeds 10
python evaluate_matched.py --trials 30 --crosswind
python evaluate_estimator_holdout.py --seeds 5
python evaluate_grid.py --trials 30 --crosswind
python uncertainty_monte_carlo.py --trials 100 --seed-base 300000 --duration 15
python report_aircraft_properties.py
```

## Run the API

```bash
uvicorn api:app --reload --port 8000
```

Health check: `http://127.0.0.1:8000/api/health`

Endpoints:

- `POST /api/simulate`
- `POST /api/simulate/compare`
- `GET /api/geometry`
- `GET /api/health`

## Model scope and honesty

- The severe scenario is an envelope-exit stress test, not a validated
  post-stall flight simulation. Runs stop and return a validity status when
  Euler angles approach 90 degrees or when angle of attack/sideslip exceeds
  the linear-model validity limit.
- Payload mass, CG, and inertia are an equivalent assumed model. Measured
  payload geometry is required for hardware-accurate claims.
- The current assumed payload is 1.5 kg maximum at an aft arm of 0.104 m;
  `report_aircraft_properties.py` prints the resulting CG/static-margin table.
- The static-margin model crosses zero near payload level 0.74; levels above
  that are longitudinally unstable in this assumed model and must be reported
  as an unstable-airframe stress case.
- Main controller tables use payload levels <= 0.74. Higher payloads require
  `--include-stress` and are labelled `unstable_airframe_stress`.
- The validated estimator/controller airspeed envelope is 12--20 m/s. The
  API and frontend reject or prevent operation outside that trained range.
- Aerodynamic coefficients are assumed values, not measured airframe data.
- The 3D replay uses backend NED position logs, not a synthetic flight path.
- Rudder geometry sign is independently checked by `geometry.py`.
- The common outer loops use measured airspeed and altitude; the payload label
  is not supplied to either controller.
- The intelligent controller estimates turbulence from recent q and vertical
  specific-force measurements; it is not given the scenario gust magnitude.
- The estimator uses a shared 21-feature schema, mixed pre/post-change windows,
  moving-tail data, training-envelope clipping, and smoothed sliding updates.
- The final deployment artifact is trained on causal windows collected from the
  adaptive loop itself; its held-out error must be reported from the training
  command and must not be replaced by the easier fixed-PID result. Splits are
  grouped by flight/run to prevent overlapping-window leakage.
- Very small payload changes may remain unobservable when their response is
  indistinguishable from turbulence; report this as an estimator limitation,
  not as a successful detection.
- The elevator actuator rate limit is fixed hardware data; neither controller
  receives the true payload level.
- `evaluate_matched.py` provides oracle-payload and estimator/tail/gain
  ablations. Do not report the NN comparison without these controls.
- `tail_no_dither` removes the bounded 0.2/0.1-degree identification sine
  excitation and is reported beside `tail_off`.
- `evaluate_grid.py` saves `grid_config.json`, `raw_runs.jsonl` and
  `summary.csv`, including every metric, invalid reason, mean +/- SD and
  paired Wilcoxon p-value for the fixed seed list.
- `gust_model.py` retains the original `ou` disturbance mode for reproducible
  legacy evidence and adds an explicit Dryden-style mode for robustness
  campaigns.
- `realism.py` provides causal sensor noise/bias/delay, first-order bounded
  actuator channels, and a safety supervisor that records envelope/OOD events
  and requests fallback to conventional control.
- `uncertainty_monte_carlo.py` samples documented parameter ranges and retains
  invalid trials and supervisor fallbacks in its JSON output. Its uncertainty
  ranges are engineering assumptions until replaced by measured or published
  parameter pedigree.
- API requests can opt into `gust_model="dryden"`, `sensor_realism`,
  `actuator_realism`, and `safety_supervisor`. These are off by default so
  nominal evidence remains comparable to the original experiments.
