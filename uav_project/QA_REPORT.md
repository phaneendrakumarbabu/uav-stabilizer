# QA Report — UAV Stability Simulation

Date: 2026-10-19

## Result

Automated backend and static frontend checks pass. No functional failure was
found in the exercised scenarios.

## Evidence

- Existing backend suite: 17/17 passed.
- QA contract/integration suite: added in `backend/test_qa.py`.
- Python `compileall`: passed.
- Node syntax checks for all frontend JavaScript files: passed.
- `trim.py`: passed for payload levels 0.0, 0.5, and 1.0.
- `geometry.py`: all reported geometry sign/DCM checks passed.
- `adaptive_system.py`: all 13×11 morphing-tail trim points converged.
- Short conventional and intelligent simulation smoke runs: passed and valid.
- Live HTTP smoke: health/geometry/frontend returned 200; conventional
  simulation returned 200; invalid airspeed returned 422.

## Findings

1. The documented unittest command is directory-sensitive. Running it from
   `backend/` passes; running the same module path from the project parent
   fails because the backend modules are imported as top-level modules. This
   is a test-runner/CI usability issue, not a runtime simulation failure.
2. Browser cases UI-02 through UI-08 require a live browser session and are
   listed in `QA_TEST_CASES.md`; the browser automation surface was not
   available in this session, so they are not treated as passed.

## Recommendation

Use the backend directory as the working directory for automated execution:

```text
python -m unittest -v test_project.py test_qa.py
```

Before release, execute the manual browser cases with the backend and static
frontend servers running, then record screenshots or browser-console errors
for any failure.
