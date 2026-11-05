// app.js
(function () {
  const state = {
    connected: false,
    real: null, // real_from_backend dims
    mode: "conventional", // conventional | intelligent | compare
    playing: false,
    playT: 0,
    playSpeed: 1.0,
    result: null, // {metrics, log} or {conventional:{...}, intelligent:{...}}
    lastRunParams: null,
    viewMode: "chase",
    renderMode: "solid",
  };

  const el = (id) => document.getElementById(id);

  const statusDot = el("status-dot");
  const statusText = el("status-text");
  const runBtn = el("run-btn");
  const modeButtons = document.querySelectorAll(".mode-btn");
  const summaryEl = el("results-summary");
  const scrubber = el("scrubber");
  const playBtn = el("play-btn");
  const timeLabel = el("time-label");
  const speedSelect = el("speed-select");
  const stepBackBtn = el("step-back-btn");
  const stepFwdBtn = el("step-fwd-btn");
  const jumpEventBtn = el("jump-event-btn");
  const exportCsvBtn = el("export-csv-btn");

  const sliders = {
    payload: el("payload-slider"),
    airspeed: el("airspeed-slider"),
    gust: el("gust-slider"),
  };
  const sliderVals = {
    payload: el("payload-val"),
    airspeed: el("airspeed-val"),
    gust: el("gust-val"),
  };
  const lateralToggle = el("lateral-toggle");
  const seedInput = el("seed-input");

  // HUD elements
  const hudAirspeed = el("hud-airspeed");
  const hudAlt = el("hud-alt");
  const hudAttitude = el("hud-attitude");
  const hudSurfaces = el("hud-surfaces");
  const hudCg = el("hud-cg");

  // KPI elements
  const kpiPitchRms = el("kpi-pitch-rms");
  const kpiPitchDiff = el("kpi-pitch-diff");
  const kpiPitchPeak = el("kpi-pitch-peak");
  const kpiControlEffort = el("kpi-control-effort");
  const kpiValidity = el("kpi-validity");

  // Live Slider listeners + 3D preview
  for (const key of Object.keys(sliders)) {
    sliders[key].addEventListener("input", () => {
      sliderVals[key].textContent = sliders[key].value;
      if (key === "payload") {
        const pl = parseFloat(sliders.payload.value);
        for (const vp of [viewportSingle, viewportConvCompare, viewportIntlCompare]) {
          if (vp) vp.setPayloadPreview(pl);
        }
        updateStaticMarginHud(pl);
      }
    });
  }

  function updateStaticMarginHud(pl) {
    if (!hudCg) return;
    const plMass = pl * 1.5;
    const totalMass = 2.5 + plMass;
    const xcg = (plMass * -0.104) / totalMass;
    const margin = 0.065 - xcg;
    if (margin < 0.02) {
      hudCg.textContent = `Aft CG (Marginal ${(margin * 100).toFixed(1)}%)`;
      hudCg.style.color = "#ff6b6b";
    } else {
      hudCg.textContent = `Stable (+${(margin * 100).toFixed(1)}%)`;
      hudCg.style.color = "#8affc1";
    }
  }

  // Preset buttons
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      sliders.payload.value = btn.dataset.pl;
      sliderVals.payload.textContent = btn.dataset.pl;

      sliders.airspeed.value = btn.dataset.va;
      sliderVals.airspeed.textContent = btn.dataset.va;

      sliders.gust.value = btn.dataset.gust;
      sliderVals.gust.textContent = btn.dataset.gust;

      lateralToggle.checked = btn.dataset.lat === "true";

      const pl = parseFloat(btn.dataset.pl);
      for (const vp of [viewportSingle, viewportConvCompare, viewportIntlCompare]) {
        if (vp) vp.setPayloadPreview(pl);
      }
      updateStaticMarginHud(pl);

      if (hudAirspeed) hudAirspeed.textContent = parseFloat(btn.dataset.va).toFixed(1);
    });
  });

  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      state.result = null;
      state.lastRunParams = null;
      state.playT = 0;
      state.playing = false;
      summaryEl.innerHTML = "";
      el("nn-estimate-readout").textContent = "";
      resetKpiCards();
      if (exportCsvBtn) exportCsvBtn.disabled = true;

      chartAttitude.setData(null);
      chartControls.setData(null);
      chartAlt.setData(null);
      chartAttitude.setEventTime(null);
      chartControls.setEventTime(null);
      chartAlt.setEventTime(null);
      scrubber.value = 0;
      timeLabel.textContent = "t = 0.00s";
      ensureViewportForMode();
      for (const vp of [viewportSingle, viewportConvCompare, viewportIntlCompare]) {
        if (vp) vp.reset();
      }
    });
  });

  // ------------------------------------------------------------------
  // Viewports & Toolbar Controls
  // ------------------------------------------------------------------
  let viewportSingle = null;
  let viewportConvCompare = null;
  let viewportIntlCompare = null;

  function renderViewportLayout() {
    const singleWrap = el("viewport-single-wrap");
    const compareWrap = el("viewport-compare-wrap");
    if (state.mode === "compare") {
      singleWrap.style.display = "none";
      compareWrap.style.display = "grid";
      if (!viewportConvCompare) {
        viewportConvCompare = new AircraftViewport(el("viewport-conv"), 0xffb454);
        viewportIntlCompare = new AircraftViewport(el("viewport-intl"), 0x4dd0e1);
        if (state.real) {
          viewportConvCompare.setRealDims(state.real);
          viewportIntlCompare.setRealDims(state.real);
        }
      }
    } else {
      singleWrap.style.display = "block";
      compareWrap.style.display = "none";
      if (!viewportSingle) {
        const color = state.mode === "intelligent" ? 0x4dd0e1 : 0xffb454;
        viewportSingle = new AircraftViewport(el("viewport-single"), color);
        if (state.real) viewportSingle.setRealDims(state.real);
      }
    }
  }

  function getActiveViewports() {
    if (state.mode === "compare") {
      return [viewportConvCompare, viewportIntlCompare].filter(Boolean);
    }
    return [viewportSingle].filter(Boolean);
  }

  function ensureViewportForMode() {
    renderViewportLayout();
    if (state.real) {
      getActiveViewports().forEach((vp) => {
        vp.setRealDims(state.real);
        vp.setViewMode(state.viewMode);
        vp.setRenderMode(state.renderMode);
      });
    }
  }

  // Viewport Toolbar button handlers
  const viewBtns = {
    center: el("btn-view-center"),
    chase: el("btn-view-chase"),
    orbit: el("btn-view-orbit"),
    top: el("btn-view-top"),
    side: el("btn-view-side"),
    front: el("btn-view-front"),
  };

  const modeBtns = {
    solid: el("btn-mode-solid"),
    seethrough: el("btn-mode-seethrough"),
    wireframe: el("btn-mode-wireframe"),
  };

  if (viewBtns.center) {
    viewBtns.center.addEventListener("click", () => {
      getActiveViewports().forEach((vp) => vp.centerView());
    });
  }

  ["chase", "orbit", "top", "side", "front"].forEach((vMode) => {
    const btn = viewBtns[vMode];
    if (btn) {
      btn.addEventListener("click", () => {
        Object.values(viewBtns).forEach((b) => b?.classList.remove("active"));
        btn.classList.add("active");
        state.viewMode = vMode;
        getActiveViewports().forEach((vp) => vp.setViewMode(vMode));
      });
    }
  });

  ["solid", "seethrough", "wireframe"].forEach((rMode) => {
    const btn = modeBtns[rMode];
    if (btn) {
      btn.addEventListener("click", () => {
        Object.values(modeBtns).forEach((b) => b?.classList.remove("active"));
        btn.classList.add("active");
        state.renderMode = rMode;
        getActiveViewports().forEach((vp) => vp.setRenderMode(rMode));
      });
    }
  });

  const snapshotBtn = el("btn-snapshot");
  if (snapshotBtn) {
    snapshotBtn.addEventListener("click", () => {
      const vp = viewportSingle || viewportIntlCompare;
      if (!vp) return;
      const dataUrl = vp.captureSnapshot();
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `uav_simulation_snapshot_${Date.now()}.png`;
      a.click();
    });
  }

  // ------------------------------------------------------------------
  // Charts
  // ------------------------------------------------------------------
  const chartAttitude = new MiniChart(el("chart-attitude"), {
    seriesConfig: [
      { key: "theta_deg", color: "#4dd0e1", label: "pitch" },
      { key: "phi_deg", color: "#ffb454", label: "roll" },
      { key: "psi_deg", color: "#8affc1", label: "yaw" },
    ],
  });
  const chartControls = new MiniChart(el("chart-controls"), {
    seriesConfig: [
      { key: "de_deg", color: "#4dd0e1", label: "elevator" },
      { key: "da_deg", color: "#ffb454", label: "aileron" },
      { key: "dr_deg", color: "#ff6b6b", label: "rudder" },
      { key: "tail_inc_deg", color: "#8affc1", label: "tail inc" },
    ],
  });
  const chartAlt = new MiniChart(el("chart-alt"), {
    seriesConfig: [{ key: "alt_m", color: "#4dd0e1", label: "altitude" }],
  });

  // ------------------------------------------------------------------
  // Connection polling & Initial 3D UAV Render
  // ------------------------------------------------------------------
  async function pollHealth() {
    try {
      await Api.health();
      if (!state.real) {
        try {
          const geo = await Api.geometry();
          state.real = geo.real_from_backend;
          ensureViewportForMode();
          // Initial center render
          const defaultFrame = {
            t: 0, theta_deg: 0, phi_deg: 0, psi_deg: 0,
            de_deg: 0, da_deg: 0, dr_deg: 0, alt_m: 100, tail_inc_deg: 0,
            pn_m: 0, pe_m: 0, pd_m: 0,
          };
          getActiveViewports().forEach((vp) => {
            vp.updateFrame(defaultFrame);
            vp.centerView();
            vp.render();
          });
          updateStaticMarginHud(parseFloat(sliders.payload.value));
        } catch (e) { /* retry next poll */ }
      }
      state.connected = true;
    } catch (e) {
      state.connected = false;
    }
    statusDot.className = state.connected ? "status-dot connected" : "status-dot disconnected";
    statusText.textContent = state.connected
      ? "Backend connected"
      : "Backend disconnected — start it (see backend/README.md) then this will reconnect automatically.";
    runBtn.disabled = !state.connected;
  }
  pollHealth();
  setInterval(pollHealth, 3000);

  // ------------------------------------------------------------------
  // Run simulation
  // ------------------------------------------------------------------
  function currentParams() {
    const enteredSeed = Number.parseInt(seedInput.value, 10);
    return {
      payload_level: parseFloat(sliders.payload.value),
      airspeed: parseFloat(sliders.airspeed.value),
      gust_intensity: parseFloat(sliders.gust.value),
      seed: Number.isFinite(enteredSeed) && enteredSeed >= 0
        ? enteredSeed
        : 42,
      lateral_disturbance: lateralToggle.checked,
      change_t: 10.0,
      T: 40.0,
    };
  }

  runBtn.addEventListener("click", async () => {
    if (!state.connected) return;
    runBtn.disabled = true;
    runBtn.textContent = "Running…";
    try {
      const params = currentParams();
      state.lastRunParams = params;
      if (state.mode === "compare") {
        const res = await Api.simulateCompare({ ...params, mode: "compare" });
        state.result = res;
      } else {
        const res = await Api.simulate({ ...params, mode: state.mode });
        state.result = res;
      }
      getActiveViewports().forEach((vp) => {
        vp.reset();
        vp.setChangeTime(params.change_t);
      });
      state.playT = 0;
      state.playing = false;
      playBtn.textContent = "▶";
      renderSummary();
      updateKpiCards();
      setupScrubber();
      drawCharts();
      updateViewportsAtT(state.playT);
      if (exportCsvBtn) exportCsvBtn.disabled = false;
    } catch (e) {
      summaryEl.innerHTML = `<div class="error">Simulation failed: ${e.message}</div>`;
    } finally {
      runBtn.disabled = !state.connected;
      runBtn.textContent = "Run Simulation";
    }
  });

  function fmt(n, d = 3) {
    return typeof n === "number" ? n.toFixed(d) : n;
  }

  function resetKpiCards() {
    if (kpiPitchRms) kpiPitchRms.textContent = "--";
    if (kpiPitchDiff) kpiPitchDiff.textContent = "Conventional vs Intelligent";
    if (kpiPitchPeak) kpiPitchPeak.textContent = "--";
    if (kpiControlEffort) kpiControlEffort.textContent = "--";
    if (kpiValidity) kpiValidity.textContent = "--";
  }

  function updateKpiCards() {
    if (!state.result) return;
    if (state.mode === "compare") {
      const conv = state.result.conventional.metrics;
      const intl = state.result.intelligent.metrics;
      const cRms = conv.rms_pitch_error, iRms = intl.rms_pitch_error;
      const diff = cRms !== 0 ? (100 * (cRms - iRms) / cRms) : 0;

      if (kpiPitchRms) kpiPitchRms.textContent = `${intl.rms_pitch_error.toFixed(3)}°`;
      if (kpiPitchDiff) {
        kpiPitchDiff.textContent = `${diff >= 0 ? "+" : ""}${diff.toFixed(1)}% vs Conventional`;
        kpiPitchDiff.style.color = diff >= 0 ? "#8affc1" : "#ff6b6b";
      }
      if (kpiPitchPeak) kpiPitchPeak.textContent = `${intl.max_pitch_dev.toFixed(2)}°`;
      if (kpiControlEffort) {
        const totalEffort = (intl.elev_activity + (intl.tail_activity || 0)).toFixed(2);
        kpiControlEffort.textContent = totalEffort;
      }
      const valid = state.result.intelligent.status?.valid;
      if (kpiValidity) {
        kpiValidity.textContent = valid ? "100% Valid" : "Envelope Limit";
        kpiValidity.style.color = valid ? "#8affc1" : "#ff6b6b";
      }
    } else {
      const m = state.result.metrics;
      if (kpiPitchRms) kpiPitchRms.textContent = `${m.rms_pitch_error.toFixed(3)}°`;
      if (kpiPitchDiff) kpiPitchDiff.textContent = `Mode: ${state.mode}`;
      if (kpiPitchPeak) kpiPitchPeak.textContent = `${m.max_pitch_dev.toFixed(2)}°`;
      if (kpiControlEffort) kpiControlEffort.textContent = `${m.elev_activity.toFixed(2)}`;
      const valid = state.result.status?.valid;
      if (kpiValidity) {
        kpiValidity.textContent = valid ? "100% Valid" : "Envelope Limit";
        kpiValidity.style.color = valid ? "#8affc1" : "#ff6b6b";
      }
    }
  }

  function metricsTable(metrics) {
    const rows = Object.entries(metrics).map(
      ([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`
    ).join("");
    return `<table class="metrics-table"><tbody>${rows}</tbody></table>`;
  }

  function statusNote(status) {
    if (!status) return "";
    let note = "";
    if (!status.valid) {
      note += `<p class="note error">Simulation stopped at the validated flight-envelope limit: ${status.terminated_reason} (t = ${Number(status.simulated_T).toFixed(2)}s).</p>`;
    }
    if (status.estimator_error) {
      note += `<p class="note error">Estimator fallback active: ${status.estimator_error}</p>`;
    }
    return note;
  }

  function renderSummary() {
    if (!state.result) { summaryEl.innerHTML = ""; return; }
    if (state.mode === "compare") {
      const { conventional, intelligent } = state.result;
      let rows = "";
      for (const k of Object.keys(conventional.metrics)) {
        const c = conventional.metrics[k], i = intelligent.metrics[k];
        const pct = c !== 0 ? (100 * (c - i) / Math.abs(c)) : 0;
        const cls = pct >= 0 ? "better" : "worse";
        rows += `<tr><td>${k}</td><td>${fmt(c)}</td><td>${fmt(i)}</td>
          <td class="${cls}">${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%</td></tr>`;
      }
      summaryEl.innerHTML = `
        <table class="metrics-table compare">
          <thead><tr><th>metric</th><th>conventional</th><th>intelligent</th><th>Δ (intelligent better = +)</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <p class="note">Numbers shown exactly as returned by the backend for these parameters — including any metric where intelligent underperforms.</p>
        <p class="note">Random seed used: ${conventional.status?.random_seed ?? state.lastRunParams?.seed ?? "unknown"}</p>
        <p class="note">Crosswind disturbance: ${conventional.status?.lateral_disturbance ? "ON" : "OFF"}</p>
        ${statusNote(conventional.status)}${statusNote(intelligent.status)}`;
    } else {
      summaryEl.innerHTML = metricsTable(state.result.metrics)
        + `<p class="note">Random seed used: ${state.result.status?.random_seed ?? state.lastRunParams?.seed ?? "unknown"}</p>`
        + `<p class="note">Crosswind disturbance: ${state.result.status?.lateral_disturbance ? "ON" : "OFF"}</p>`
        + statusNote(state.result.status);
    }
  }

  // ------------------------------------------------------------------
  // Playback Controls
  // ------------------------------------------------------------------
  function activeLog() {
    if (!state.result) return null;
    if (state.mode === "compare") return state.result.intelligent.log;
    return state.result.log;
  }

  function setupScrubber() {
    const log = activeLog();
    if (!log) return;
    scrubber.min = 0;
    scrubber.max = log.t.length - 1;
    scrubber.value = 0;
    timeLabel.textContent = `t = ${log.t[0].toFixed(2)}s`;
  }

  scrubber.addEventListener("input", () => {
    state.playing = false;
    playBtn.textContent = "▶";
    const idx = parseInt(scrubber.value, 10);
    const log = activeLog();
    if (!log) return;
    state.playT = log.t[idx];
    updateViewportsAtT(state.playT);
    drawChartsCursor();
  });

  playBtn.addEventListener("click", () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? "⏸" : "▶";
  });

  if (speedSelect) {
    speedSelect.addEventListener("change", () => {
      state.playSpeed = parseFloat(speedSelect.value) || 1.0;
    });
  }

  if (stepBackBtn) {
    stepBackBtn.addEventListener("click", () => {
      const log = activeLog();
      if (!log) return;
      state.playT = Math.max(log.t[0], state.playT - 0.1);
      const idx = findIndexForT(log.t, state.playT);
      scrubber.value = idx;
      updateViewportsAtT(state.playT);
      drawChartsCursor();
    });
  }

  if (stepFwdBtn) {
    stepFwdBtn.addEventListener("click", () => {
      const log = activeLog();
      if (!log) return;
      state.playT = Math.min(log.t[log.t.length - 1], state.playT + 0.1);
      const idx = findIndexForT(log.t, state.playT);
      scrubber.value = idx;
      updateViewportsAtT(state.playT);
      drawChartsCursor();
    });
  }

  if (jumpEventBtn) {
    jumpEventBtn.addEventListener("click", () => {
      const log = activeLog();
      if (!log) return;
      const targetT = state.lastRunParams?.change_t ?? 10.0;
      state.playT = targetT;
      const idx = findIndexForT(log.t, state.playT);
      scrubber.value = idx;
      updateViewportsAtT(state.playT);
      drawChartsCursor();
    });
  }

  // Export CSV
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener("click", () => {
      const log = activeLog();
      if (!log) return;
      const headers = Object.keys(log).filter((k) => Array.isArray(log[k]) && typeof log[k][0] === "number");
      const rows = [headers.join(",")];
      const count = log.t.length;
      for (let i = 0; i < count; i++) {
        rows.push(headers.map((h) => log[h][i]).join(","));
      }
      const blob = new Blob([rows.join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `uav_flight_telemetry_${state.mode}_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  function findIndexForT(tArr, t) {
    let lo = 0, hi = tArr.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (tArr[mid] < t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function frameAt(log, t) {
    if (!log || !log.t || log.t.length === 0) return {};
    if (t <= log.t[0]) return frameAtIndex(log, 0);
    if (t >= log.t[log.t.length - 1]) return frameAtIndex(log, log.t.length - 1);
    const hi = findIndexForT(log.t, t);
    const lo = Math.max(0, hi - 1);
    const span = log.t[hi] - log.t[lo];
    const f = span > 0 ? (t - log.t[lo]) / span : 0;
    const frame = {};
    for (const k of Object.keys(log)) {
      if (k === "pl_est_t") continue;
      if (!Array.isArray(log[k])) { frame[k] = log[k]; continue; }
      const a = Number(log[k][lo]);
      const b = Number(log[k][hi]);
      frame[k] = Number.isFinite(a) && Number.isFinite(b) ? a + f * (b - a) : log[k][lo];
    }
    return frame;
  }

  function frameAtIndex(log, idx) {
    const frame = {};
    for (const k of Object.keys(log)) {
      if (k !== "pl_est_t") frame[k] = Array.isArray(log[k]) ? log[k][idx] : log[k];
    }
    return frame;
  }

  function plEstAt(log, t) {
    if (!log.pl_est_t || log.pl_est_t.length === 0) return null;
    let val = null;
    for (const entry of log.pl_est_t) {
      if (entry.t <= t) val = entry.pl_est; else break;
    }
    return val;
  }

  function updateHud(frame) {
    if (!frame) return;
    if (hudAirspeed && state.lastRunParams) {
      hudAirspeed.textContent = Number(state.lastRunParams.airspeed).toFixed(1);
    }
    if (hudAlt && frame.alt_m !== undefined) {
      hudAlt.textContent = Number(frame.alt_m).toFixed(1);
    }
    if (hudAttitude && frame.theta_deg !== undefined && frame.phi_deg !== undefined) {
      hudAttitude.textContent = `P: ${frame.theta_deg.toFixed(1)}° / R: ${frame.phi_deg.toFixed(1)}°`;
    }
    if (hudSurfaces && frame.de_deg !== undefined) {
      const tail = frame.tail_inc_deg !== undefined ? `${frame.tail_inc_deg.toFixed(1)}°` : "0.0°";
      hudSurfaces.textContent = `δe: ${frame.de_deg.toFixed(1)}° | δtail: ${tail}`;
    }
  }

  function updateViewportsAtT(t) {
    if (!state.result) return;
    ensureViewportForMode();
    if (state.mode === "compare") {
      const cLog = state.result.conventional.log;
      const iLog = state.result.intelligent.log;
      const cFrame = frameAt(cLog, t);
      const iFrame = frameAt(iLog, t);
      if (viewportConvCompare) { viewportConvCompare.updateFrame(cFrame); viewportConvCompare.render(); }
      if (viewportIntlCompare) { viewportIntlCompare.updateFrame(iFrame); viewportIntlCompare.render(); }
      updateHud(iFrame);
    } else {
      const log = state.result.log;
      const f = frameAt(log, t);
      if (viewportSingle) { viewportSingle.updateFrame(f); viewportSingle.render(); }
      updateHud(f);
    }
    timeLabel.textContent = `t = ${t.toFixed(2)}s`;

    // NN estimate vs true payload readout
    const nnEl = el("nn-estimate-readout");
    if (nnEl) {
      let log = state.mode === "compare" ? state.result.intelligent.log
        : (state.mode === "intelligent" ? state.result.log : null);
      if (log) {
        const est = plEstAt(log, t);
        const truePl = t >= state.lastRunParams.change_t ? state.lastRunParams.payload_level : 0.0;
        nnEl.textContent = est !== null
          ? `NN estimate: ${est.toFixed(3)}   |   true payload_level: ${truePl.toFixed(3)}`
          : `NN estimate: (no update yet)   |   true payload_level: ${truePl.toFixed(3)}`;
      } else {
        nnEl.textContent = "NN estimate: n/a in conventional mode";
      }
    }
  }

  function drawCharts() {
    const log = activeLog();
    if (!log) return;
    chartAttitude.setData(log);
    chartControls.setData(log);
    chartAlt.setData(log);
    const eventT = state.lastRunParams ? state.lastRunParams.change_t : null;
    chartAttitude.setEventTime(eventT);
    chartControls.setEventTime(eventT);
    chartAlt.setEventTime(eventT);
    drawChartsCursor();
  }

  function drawChartsCursor() {
    chartAttitude.setCursor(state.playT);
    chartControls.setCursor(state.playT);
    chartAlt.setCursor(state.playT);
    chartAttitude.draw();
    chartControls.draw();
    chartAlt.draw();
  }

  // ------------------------------------------------------------------
  // Animation loop
  // ------------------------------------------------------------------
  let lastFrameTime = performance.now();
  function tick(now) {
    const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
    lastFrameTime = now;

    if (state.playing && state.result) {
      const log = activeLog();
      const tMax = log.t[log.t.length - 1];
      state.playT += dt * state.playSpeed;
      if (state.playT >= tMax) { state.playT = tMax; state.playing = false; playBtn.textContent = "▶"; }
      const idx = findIndexForT(log.t, state.playT);
      scrubber.value = idx;
      updateViewportsAtT(state.playT);
      drawChartsCursor();
    } else {
      getActiveViewports().forEach((vp) => vp.render());
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  renderViewportLayout();
})();
