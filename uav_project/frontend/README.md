# UAV Stability Simulation — Frontend

Plain HTML/CSS/JavaScript (no bundler, no build step) using Three.js
(loaded from a CDN `<script>` tag) for the 3D viewport. This was chosen
over a React+Vite setup specifically so the whole app can be opened with
zero install step beyond a static file server — appropriate for a
no-coding-background user following a README.

## 1. Start the backend first

This frontend has **zero mock, hardcoded, or fallback data**. Every number
on screen — every chart point, every simulated metric, every 3D pose — is
read from a live response of the backend API for the exact parameters you
selected. If the backend isn't running, the connection indicator will show
red and the "Run Simulation" button will stay disabled.

Follow `../backend/README.md` first, and leave the backend server (`uvicorn
api:app --reload --port 8000`) running in its own terminal.

## 2. Serve this folder

Because this app makes `fetch()` calls, opening `index.html` directly via
`file://` can hit browser CORS restrictions. Serve it over local HTTP
instead. From this `frontend/` folder, with Python already installed:

```bash
python -m http.server 5500
```

Then open http://127.0.0.1:5500 in your browser.

## 3. Using the app

1. Wait for the top-right status dot to turn green ("Backend connected").
2. Adjust the payload / airspeed / gust sliders and pick a mode
   (Conventional / Intelligent / Compare).
3. Click **Run Simulation**. The 3D aircraft, charts, and results table are
   populated entirely from that response.
4. Use the play/pause button and scrubber to replay the flight. Nothing is
   simulated in the browser — it's all just replaying the returned time
   log.
5. In Compare mode, both aircraft replay the SAME timeline (identical
   disturbance) side by side so you can see the difference directly.

## Notes / limitations

- The NN payload-estimate readout uses a **held-value (step) lookup** of
  the sparse, irregularly-timed `pl_est_t` samples returned by the
  backend — it does not interpolate between samples, matching how the
  estimate is actually produced at runtime.
- If you change the backend's `api.py` host/port, set
  `window.UAV_API_BASE` in a small inline `<script>` before the other
  scripts in `index.html` (defaults to `http://127.0.0.1:8000`).
- `js/geometry.js` is a hand-written, line-by-line port of
  `backend/geometry.py`. Real dimensions (`b`, `c`, `S_tail`, `l_tail`)
  are always fetched from `/api/geometry` at runtime, never hardcoded here.
  Open the browser devtools console — `UAVGeometry.runSelfTests(real)`
  reproduces the same nose-tip/elevator/aileron sign checks as the Python
  test suite, for manual cross-verification once you have a live `real`
  dims object (e.g. `UAVGeometry.runSelfTests(await Api.geometry().then(g=>g.real_from_backend))`
  in the console).
