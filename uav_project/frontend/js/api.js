// api.js
const API_BASE = window.UAV_API_BASE || (window.location.port === "5500" || window.location.origin.startsWith("file:") ? "http://127.0.0.1:8000" : "");

async function requestError(res, label) {
  let detail = "";
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? `: ${body.detail}`
      : body.detail ? `: ${JSON.stringify(body.detail)}` : "";
  } catch (_) { /* keep the HTTP status when the body is not JSON */ }
  return new Error(`${label} failed (${res.status})${detail}`);
}

const Api = {
  async health() {
    const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    if (!res.ok) throw new Error(`health check failed: ${res.status}`);
    return res.json();
  },

  async geometry() {
    const res = await fetch(`${API_BASE}/api/geometry`);
    if (!res.ok) throw new Error(`geometry fetch failed: ${res.status}`);
    return res.json();
  },

  async simulate(params) {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw await requestError(res, "simulate");
    return res.json();
  },

  async simulateCompare(params) {
    const res = await fetch(`${API_BASE}/api/simulate/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw await requestError(res, "compare");
    return res.json();
  },
};

window.Api = Api;
