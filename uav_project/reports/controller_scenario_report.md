# Controller Scenario Report

Source: `backend/results_grid_final/raw_runs.jsonl`; 30 matched seeds per scenario.

## Scenarios

| Payload | Gust | Airspeed | Conv valid | Intelligent valid | Delta (pp) | Tracking wins | Effort wins | Interpretation |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.2 | 0.5 | 12 | 100.0% | 100.0% | 0.0 | 6/9 | 1/3 | Mixed result |
| 0.2 | 0.5 | 20 | 100.0% | 100.0% | 0.0 | 2/9 | 1/3 | Mixed result |
| 0.2 | 1.5 | 12 | 96.7% | 96.7% | 0.0 | 5/9 | 0/3 | Mixed result |
| 0.2 | 1.5 | 20 | 100.0% | 100.0% | 0.0 | 7/9 | 1/3 | Mixed result |
| 0.5 | 0.5 | 12 | 100.0% | 100.0% | 0.0 | 6/9 | 2/3 | Mixed result |
| 0.5 | 0.5 | 20 | 100.0% | 100.0% | 0.0 | 4/9 | 1/3 | Mixed result |
| 0.5 | 1.5 | 12 | 93.3% | 93.3% | 0.0 | 6/9 | 1/3 | Mixed result |
| 0.5 | 1.5 | 20 | 100.0% | 100.0% | 0.0 | 5/9 | 1/3 | Mixed result |
| 0.7 | 0.5 | 12 | 100.0% | 100.0% | 0.0 | 7/9 | 3/3 | Mixed result |
| 0.7 | 0.5 | 20 | 100.0% | 100.0% | 0.0 | 4/9 | 1/3 | Mixed result |
| 0.7 | 1.5 | 12 | 90.0% | 93.3% | 3.3 | 7/9 | 1/3 | Intelligent candidate |
| 0.7 | 1.5 | 20 | 53.3% | 83.3% | 30.0 | 6/9 | 1/3 | Intelligent candidate |

## Interpretation rule

An `Intelligent candidate` has a higher valid-run rate and wins at least 5 of 9 tracking metrics on common valid seeds. A `Mixed result` means there is a trade-off. These labels are decision aids, not proof of hardware superiority.

## QA result

The automated test suite ran 559 tests and passed. Existing nominal grids were complete and their manifests were hash-checked. The 20-trial uncertainty campaign remains diagnostic: both controllers had 50% valid-finite rate, so it should not be presented as final safety evidence.