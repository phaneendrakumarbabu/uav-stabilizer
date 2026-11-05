# Robustness validation additions

The project now has an explicit robustness path in addition to the original
nominal experiment path.

## Added models

- `backend/gust_model.py`: retains the original `ou` mode for old evidence and
  adds a Dryden-style mode with a first-order longitudinal filter and
  second-order lateral/vertical shaping filters.
- `backend/realism.py`: causal delayed/noisy sensor suite, bounded first-order
  actuator channels, and an envelope/supervisor monitor.
- `backend/final_comparison.py`: optional `gust_model`, `sensor_config`,
  `actuator_config`, and `safety_config` arguments. The default path remains
  compatible with the original nominal results.
- `backend/uncertainty_monte_carlo.py`: samples aerodynamic, mass-property,
  atmosphere, operating-condition, sensor, actuator, gust, and crosswind
  uncertainties. It records invalid trials and supervisor fallbacks rather
  than discarding them.

## Reproduce the robustness campaign

From `backend`:

```powershell
python uncertainty_monte_carlo.py --trials 100 --seed-base 300000 --duration 15
```

The output is written to `backend/uncertainty_monte_carlo_results.json`.
The current smoke campaign uses 20 trials and is a software-verification
artifact, not a statistically sufficient flight-safety demonstration. Increase
the trial count before using the results in a paper.

## Interpretation rules

- A valid run is not automatically a validated aircraft prediction.
- A supervisor fallback is an event to report, not a successful adaptive run.
- Invalid runs remain in the denominator.
- The sampled ranges are engineering assumptions and must be replaced or
  justified with measured/literature parameter uncertainty.
- The Dryden-style model is stronger than the old OU model, but it still does
  not replace atmospheric or flight-test data.
