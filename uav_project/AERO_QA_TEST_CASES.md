# Aeronautical QA and Release-Gate Test Cases

These tests are separate from the fast physics/API smoke suite. They check
whether the updated project is safe to describe as a deployable UAV system or
as a reproducible international-conference experiment.

Run from `backend`:

```powershell
python -m unittest -v test_aero_qa.py test_aero_extended.py
python -m unittest -q test_aero_500.py
```

| ID | Priority | Test | Release expectation |
|---|---:|---|---|
| AERO-01 | P0 | Estimator and dataset feature contract | Dataset, scaler, and model must have the same feature count and names |
| AERO-02 | P0 | Nominal intelligent-controller run | States, commands, estimates, status, and metrics must remain finite |
| AERO-03 | P0 | Sub-timestep API request | Reject `T < DT` or return a valid finite initial sample; never return NaN metrics |
| AERO-04 | P0 | Result-manifest source hashes | Every evidence file must be tied to the exact source revision used to generate it |
| AERO-05 | P0 | High-payload/high-gust/high-speed safety gate | NN must not lose more valid trials than the conventional baseline without an approved fallback |
| AERO-06 | P1 | Crosswind evidence | At least one completed grid must exercise crosswind/lateral disturbance |
| AERO-07 | P0 | Public API CORS policy | No wildcard origin combined with credentialed requests |
| AERO-08 | P0 | Dryden gust reproducibility | Same seed produces identical finite disturbance history |
| AERO-09 | P0 | Sensor/actuator causality | Measurements are delayed/noisy only through the declared configuration; actuators remain bounded and rate-limited |
| AERO-10 | P0 | Safety supervisor | Out-of-distribution estimator inputs latch fallback and record the first event |
| AERO-11 | P1 | Uncertainty campaign completeness | Every sampled trial is represented, including invalid and fallback outcomes |
| AERO-12 | P0 | Coordinate and force-model invariants | DCMs remain orthogonal/right-handed and dynamic pressure follows the expected V² trend |
| AERO-13 | P0 | Numerical repeatability/convergence | Repeated seeded runs match and timestep refinement does not materially change the trajectory |
| AERO-14 | P0 | Trim and payload trends | Trim residuals remain bounded and payload mass changes inertia/static-margin quantities in the expected direction |
| AERO-15 | P0 | Integrated realism path | Dryden gusts, sensors, actuators, supervisor, API flags, and finite metrics work together |
| AERO-16 | P1 | Evidence accounting | Grid manifests are complete, unique, hashed, and summary rows account for every trial |
| AERO-17 | P0 | Large regression matrix | 500 generated physical/configuration cases execute independently and pass |
| AERO-18 | P0 | Monte Carlo failure attribution | Invalid runs report termination causes and safety-trigger causes separately |
| AERO-19 | P0 | Closed-loop stability envelope | Local PID linearisation is finite and stable across the open-loop stress-payload cases |

## Interpretation

The existing smoke suite may pass while a release gate fails. A release-gate
failure is an issue to fix or explicitly qualify in the paper; it should not be
removed by changing the test acceptance criterion.

This suite does not replace hardware-in-the-loop, wind-tunnel, or flight-test
qualification. It only prevents unsupported simulation and deployment claims.

The 500-case suite is a regression-strength check of mathematical and software
properties. It does not create physical validation data and must not be
presented as 500 flight tests.
