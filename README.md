---
title: UAV 6-DOF Adaptive Flight Simulation
emoji: ✈️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 8000
pinned: false
---

# UAV 6-DOF Stability & Adaptive Control Simulation

A high-fidelity 6-DOF fixed-wing UAV flight dynamics simulation and adaptive flight control platform.

## Features
- **Nonlinear 6-DOF Dynamics**: Rigid-body flight dynamics, aerodynamic force and moment derivatives, trim solving, and stall/crosswind envelope validation.
- **Conventional vs. Intelligent Control**:
  - *Baseline*: Conventional cascaded longitudinal and lateral PID loops.
  - *Intelligent*: Real-time causal neural network payload estimator + morphing-tail trim compensation + lateral gain scheduling.
- **Interactive 3D Viewport**: Built with Three.js, featuring 360° orbit controls, chase camera, synced telemetry HUD, and an X-ray see-through mode revealing the internal payload cargo module and dynamic Center-of-Gravity (CG) shift.
- **Deterministic Turbulence**: Dryden and Ornstein-Uhlenbeck turbulence spectra with reproducible random seeds.
- **Synced Telemetry & Analytics**: Real-time charts for pitch, roll, yaw, altitude, elevator, aileron, rudder, and tail incidence.

## API Documentation
The underlying FastAPI service exposes OpenAPI / Swagger documentation at `/docs`:
- `POST /api/simulate`: Runs a single controller flight trajectory.
- `POST /api/simulate/compare`: Runs conventional and intelligent controllers on identical wind disturbance.
- `GET /api/geometry`: Airframe specifications and control surface boundaries.
- `GET /api/health`: Service health check.

