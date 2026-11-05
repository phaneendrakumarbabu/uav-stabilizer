# Evidence summary

All nominal grid numbers regenerate from the current source revision. Evaluation
seeds start at 100000 and are disjoint from estimator-training seeds starting
at 1000000.

Proposed controller = NN payload estimate -> morphing-tail trim + lateral
(roll/yaw) gain scheduling; pitch gains are not boosted.

- Grid 1 `results_grid_final/`: 1440/1440 completed runs.
- Grid 2 `results_grid_final2/`: 1080/1080 completed runs.
- Both manifests now pass their SHA-256 checks against the current source,
  estimator, and raw data.
- The grids contain 30 seeds x 12 conditions: payload 0.2/0.5/0.7, gust
  0.5/1.5, airspeed 12/20 m/s, T=15 s, payload step at 4 s.
- Overall valid-run rates in Grid 1: conventional 94.44%, NN 97.22%,
  elevator feed-forward 96.11%, oracle 97.78%.
- At payload 0.7, airspeed 20 m/s, gust 1.5, the current regenerated grid
  reports conventional 16/30 valid, NN 25/30, elevator feed-forward 21/30,
  and oracle 27/30. NN RMS pitch error is 0.383 versus 0.353 for conventional,
  so validity improved but RMS pitch did not improve at this condition.
- Estimator: `nn_validation_report.json` includes grouped CV, baselines,
  held-out conditions, and a shuffled-label control.
- Closed-loop estimator: `estimator_closed_loop.txt` uses disjoint seeds.
- Linear stability: `linear_stability_results.txt` reports open-loop nominal
  stability through payload 0.7 and open-loop instability at payload 0.8 and
  above. A separate local closed-loop PID linearisation is stable through
  payload 1.0 for the tested airspeeds; this does not prove nonlinear,
  sensor/actuator-realistic, or hardware stability.
- Robustness smoke campaign: `uncertainty_monte_carlo_results.json` uses
  Dryden-style gusts, crosswind, sensor noise/delay, actuator dynamics, sampled
  aerodynamic/mass parameters, and the safety supervisor. It ran 20 trials;
  50% were valid and finite for both controllers, with supervisor fallback in
  70% of conventional and 70% of intelligent trials. This is a diagnostic
  campaign, not a final statistical safety claim; increase the trial count.
- The campaign now records termination causes separately: all 10 invalid runs
  for each controller stopped at the aerodynamic validity limit, while safety
  events are reported independently. The intelligent controller triggered
  estimator-out-of-distribution fallback in 14/20 trials; this is evidence that
  the estimator's realistic-input domain needs expansion or a validated sensor
  model, not a reason to disable the supervisor.
- A 500-case generated regression suite covers coordinate transforms,
  aerodynamic operating points, seeded gusts, actuator behavior, and
  sensor/supervisor decisions. These are software/model checks, not 500 flight
  tests.
- The safety supervisor now includes a pre-command alpha-protection clamp that
  preserves recovery-direction elevator authority inside the declared margin.
  In the current 20-trial campaign it did not increase the 50% valid-run rate;
  that unresolved result is retained as an engineering issue.
- The estimator OOD gate now evaluates the raw feature vector against the
  stored training-support bounds before clipping, with a 20% outside-feature
  fraction threshold. This removes false OOD decisions caused only by
  z-scores at clipped training extrema, but the realistic campaign still
  requires further estimator/controller validation.

Not covered: real flight/wind-tunnel validation; measured aerodynamic
coefficients; longer-duration operation; and hardware sensor/actuator data.
