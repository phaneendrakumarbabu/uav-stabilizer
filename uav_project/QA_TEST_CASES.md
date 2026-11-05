# UAV Stability Simulation QA Test Cases

## Scope

Backend physics/control behavior, FastAPI request/response contracts, saved
run artifacts, frontend data binding, and basic browser usability.

## Automated test cases

| ID | Priority | Test | Expected result |
|---|---|---|---|
| API-01 | P0 | GET `/api/health` | HTTP 200 and `{"status":"ok"}` |
| API-02 | P0 | GET `/api/geometry` | Real aircraft dimensions are present and finite |
| API-03 | P0 | POST conventional simulation at payload 0/1 and airspeed 12/20 | Request is accepted inside the documented envelope |
| API-04 | P0 | Submit NaN/Infinity, airspeed outside 12–20, or `change_t > T` | Request is rejected with validation error |
| API-05 | P0 | POST `/api/simulate` | Metrics, status, and equal-length frontend log arrays are returned |
| API-06 | P0 | POST `/api/simulate/compare` | Conventional and intelligent result objects are both returned |
| API-07 | P1 | POST `/api/simulate` with `mode=compare` | API rejects the wrong endpoint with HTTP 422 |
| API-08 | P0 | Run same conventional seed twice | Logs and metrics are reproducible |
| API-09 | P0 | Run intelligent controller | 15-column log, finite values, estimator/controller status fields |
| API-10 | P1 | Pass malformed backend log to serializer | Shape mismatch raises a clear `ValueError` |
| UI-01 | P1 | Load frontend and inspect script/API contract | Required scripts and API routes are present |

Automated implementation: `backend/test_project.py` covers core physics and
`backend/test_qa.py` covers the cases above.

## Manual/browser test cases

| ID | Priority | Steps | Expected result |
|---|---|---|---|
| UI-02 | P0 | Start backend and frontend; open the frontend | Status changes to “Backend connected”; Run button enables |
| UI-03 | P0 | Run Conventional with default sliders | Summary, charts, scrubber, and 3D replay populate |
| UI-04 | P0 | Switch to Intelligent and run | NN estimate readout appears; intelligent controls render |
| UI-05 | P0 | Run Compare with crosswind enabled | Two synchronized viewports and both metric columns render |
| UI-06 | P1 | Scrub to start, middle, and end | Time label, charts, and aircraft pose stay synchronized |
| UI-07 | P1 | Stop/restart backend while page is open | Connection indicator and Run button recover automatically |
| UI-08 | P1 | Run high gust/payload stress case | If envelope is exceeded, a clear validity/termination note is shown |

## Exit criteria

P0 cases must pass. Any failed case needs a reproducible input, observed
result, expected result, and severity before release.
