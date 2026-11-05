# Seed reproducibility evidence

This artifact uses the real simulator with the same payload, airspeed, gust intensity, duration, payload-change time, and crosswind setting.

| Controller | Seed | Valid | Log SHA-256 (first 16) | RMS pitch error | Max pitch deviation |
|---|---:|---|---|---:|---:|
| conventional | 42 | True | `01a444ea8e349126` | 3.660521 | 19.671023 |
| conventional | 42 | True | `01a444ea8e349126` | 3.660521 | 19.671023 |
| conventional | 43 | True | `04c1222b9afe0ac1` | 0.287534 | 0.569481 |
| intelligent | 42 | True | `d467da86d7b7e308` | 0.615375 | 3.527311 |
| intelligent | 42 | True | `d467da86d7b7e308` | 0.615375 | 3.527311 |
| intelligent | 43 | False | `1c5ce288def42517` | 15.995743 | 35.608821 |

## Checks

- `conventional_same_seed_repeats_exactly`: **PASS**
- `conventional_different_seed_changes_log`: **PASS**
- `intelligent_same_seed_repeats_exactly`: **PASS**
- `intelligent_different_seed_changes_log`: **PASS**
- `matched_controller_seed_is_same`: **PASS**

## Conclusion

The visible scenario parameters define disturbance strength and the aircraft condition. The seed defines the exact stochastic gust realization. Same seed reproduces the exact log; a different seed creates a different gust realization. Compare mode uses the same seed for both controllers.

The conventional and intelligent runs with seed 42 are a matched comparison. The dashboard now exposes the seed field, so entering the same seed reproduces the same scenario.
