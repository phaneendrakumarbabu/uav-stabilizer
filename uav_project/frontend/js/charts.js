// charts.js
// Lightweight canvas-based line chart with a moving time cursor. No heavy
// chart library -- just enough to show synced readouts during playback.

class MiniChart {
  constructor(canvasEl, { seriesConfig, yLabel = "" }) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    this.seriesConfig = seriesConfig; // [{key, color, label}]
    this.yLabel = yLabel;
    this.data = null; // {t: [...], seriesKey: [...] }
    this.cursorT = 0;
    this.eventT = null;
    this._resize();
    window.addEventListener("resize", () => { this._resize(); this.draw(); });
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, rect.width * dpr);
    this.canvas.height = Math.max(1, rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = rect.width;
    this.h = rect.height;
  }

  setData(data) {
    this.data = data;
  }

  setCursor(t) {
    this.cursorT = t;
  }

  setEventTime(t) {
    this.eventT = Number.isFinite(Number(t)) ? Number(t) : null;
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    if (!this.data || !this.data.t || this.data.t.length === 0) return;

    const t = this.data.t;
    const tMin = t[0], tMax = t[t.length - 1];

    let yMin = Infinity, yMax = -Infinity;
    for (const s of this.seriesConfig) {
      const arr = this.data[s.key];
      if (!arr) continue;
      for (const v of arr) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
    if (!isFinite(yMin) || !isFinite(yMax)) return;
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const pad = (yMax - yMin) * 0.1;
    yMin -= pad; yMax += pad;

    const marginL = 34, marginB = 14, marginT = 6, marginR = 6;
    const plotW = this.w - marginL - marginR;
    const plotH = this.h - marginT - marginB;

    const xOf = (tt) => marginL + ((tt - tMin) / (tMax - tMin || 1)) * plotW;
    const yOf = (vv) => marginT + plotH - ((vv - yMin) / (yMax - yMin || 1)) * plotH;

    // gridlines
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const yy = marginT + (plotH * i) / 3;
      ctx.beginPath(); ctx.moveTo(marginL, yy); ctx.lineTo(this.w - marginR, yy); ctx.stroke();
    }

    // y-axis labels
    ctx.fillStyle = "rgba(220,235,245,0.55)";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(yMax.toFixed(1), marginL - 4, marginT + 8);
    ctx.fillText(yMin.toFixed(1), marginL - 4, marginT + plotH);

    // series lines
    for (const s of this.seriesConfig) {
      const arr = this.data[s.key];
      if (!arr) continue;
      ctx.beginPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.75;
      for (let i = 0; i < t.length; i++) {
        const x = xOf(t[i]), y = yOf(arr[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Payload-change event marker, shared with the 3D replay marker.
    if (this.eventT !== null && this.eventT >= tMin && this.eventT <= tMax) {
      const ex = xOf(this.eventT);
      ctx.strokeStyle = "rgba(255,107,107,0.85)";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ex, marginT); ctx.lineTo(ex, marginT + plotH); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,150,150,0.9)";
      ctx.textAlign = "center";
      ctx.fillText("payload", ex, marginT + 10);
    }

    // cursor
    const cx = xOf(this.cursorT);
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, marginT); ctx.lineTo(cx, marginT + plotH); ctx.stroke();
  }
}

window.MiniChart = MiniChart;
