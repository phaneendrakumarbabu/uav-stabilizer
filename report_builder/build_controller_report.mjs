import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = path.resolve(process.cwd(), "..", "uav_project");
const backendRoot = path.join(projectRoot, "backend");
const sourcePath = path.join(backendRoot, "results_grid_final", "raw_runs.jsonl");
const outputDir = path.join(projectRoot, "reports");
await fs.mkdir(outputDir, { recursive: true });

const raw = (await fs.readFile(sourcePath, "utf8"))
  .trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
const configs = new Map();
for (const row of raw) {
  const key = `${row.payload}|${row.gust}|${row.airspeed}|${row.seed}`;
  if (!configs.has(key)) configs.set(key, {});
  configs.get(key)[row.config] = row;
}

const lowerBetter = new Set([
  "max_pitch_dev", "max_roll_dev", "max_yaw_dev",
  "rms_pitch_error", "rms_roll_error", "rms_yaw_error",
  "rms_altitude_error", "max_altitude_drop_m", "mean_elev",
  "mean_aileron", "mean_rudder", "pitch_settling_time_s",
  "abs_pitch_steady_state_error",
]);
const trackingMetrics = [
  "max_pitch_dev", "max_roll_dev", "max_yaw_dev",
  "rms_pitch_error", "rms_roll_error", "rms_yaw_error",
  "rms_altitude_error", "max_altitude_drop_m",
  "abs_pitch_steady_state_error",
];
const effortMetrics = ["mean_elev", "mean_aileron", "mean_rudder"];
const allMetrics = [
  ...trackingMetrics, ...effortMetrics,
  "pitch_settling_time_s", "pitch_settled",
];

function mean(values) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}
function sd(values) {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((s, x) => s + (x - m) ** 2, 0) / (values.length - 1));
}
function rowMetric(row, metric) {
  if (metric === "abs_pitch_steady_state_error") return Math.abs(Number(row.metrics.pitch_steady_state_error));
  return Number(row.metrics[metric]);
}
function better(metric, c, i) {
  if (metric === "pitch_settled") return i > c;
  return lowerBetter.has(metric) ? i < c : i > c;
}
function pct(c, i) {
  if (c === null || i === null || Math.abs(c) < 1e-12) return null;
  return 100 * (c - i) / Math.abs(c);
}

const scenarioKeys = [...new Set(raw.filter(r => r.config === "conventional")
  .map(r => `${r.payload}|${r.gust}|${r.airspeed}`))].sort((a, b) => {
    const aa = a.split("|").map(Number), bb = b.split("|").map(Number);
    return aa[0] - bb[0] || aa[1] - bb[1] || aa[2] - bb[2];
  });

const scenarioRows = [];
const detailRows = [];
for (const skey of scenarioKeys) {
  const [payload, gust, airspeed] = skey.split("|").map(Number);
  const pairs = [...configs.entries()]
    .filter(([key]) => key.startsWith(`${skey}|`))
    .map(([, byConfig]) => byConfig)
    .filter(x => x.conventional && x.nn);
  const convAll = pairs.map(x => x.conventional);
  const nnAll = pairs.map(x => x.nn);
  const convValid = convAll.filter(x => x.valid);
  const nnValid = nnAll.filter(x => x.valid);
  const common = pairs.filter(x => x.conventional.valid && x.nn.valid);
  const trackingWins = trackingMetrics.filter(metric => {
    const c = mean(common.map(x => rowMetric(x.conventional, metric)));
    const i = mean(common.map(x => rowMetric(x.nn, metric)));
    return c !== null && i !== null && better(metric, c, i);
  }).length;
  const effortWins = effortMetrics.filter(metric => {
    const c = mean(common.map(x => rowMetric(x.conventional, metric)));
    const i = mean(common.map(x => rowMetric(x.nn, metric)));
    return c !== null && i !== null && better(metric, c, i);
  }).length;
  const validDelta = 100 * (nnValid.length - convValid.length) / convAll.length;
  const classification = validDelta > 0 && trackingWins >= 5
    ? "Intelligent candidate"
    : validDelta < 0 && trackingWins <= 3
      ? "Conventional candidate"
      : "Mixed result";
  const note = classification === "Intelligent candidate"
    ? "Higher valid-run rate and wins most tracking metrics."
    : classification === "Conventional candidate"
      ? "Conventional has the stronger completion/tracking balance here."
      : "Controller trade-off; inspect detailed metrics before claiming a winner.";
  scenarioRows.push({
    payload, gust, airspeed,
    conventional_valid_rate: convValid.length / convAll.length,
    intelligent_valid_rate: nnValid.length / nnAll.length,
    valid_rate_delta_pp: validDelta,
    paired_valid_n: common.length,
    intelligent_tracking_wins: trackingWins,
    tracking_metric_count: trackingMetrics.length,
    intelligent_effort_wins: effortWins,
    effort_metric_count: effortMetrics.length,
    classification, note,
  });
  for (const metric of allMetrics) {
    const cVals = common.map(x => rowMetric(x.conventional, metric));
    const iVals = common.map(x => rowMetric(x.nn, metric));
    const c = mean(cVals), i = mean(iVals);
    detailRows.push({
      payload, gust, airspeed, metric, paired_valid_n: common.length,
      conventional_mean: c, intelligent_mean: i,
      intelligent_change_percent: pct(c, i),
      intelligent_better: c !== null && i !== null ? better(metric, c, i) : null,
      conventional_sd: sd(cVals), intelligent_sd: sd(iVals),
    });
  }
}

const qaTests = [
  ["Automated QA/regression suite", 559, "PASS", "All 559 tests passed, including 500 generated physics/realism cases."],
  ["Grid 1 completeness", 1440, "PASS", "30 seeds × 12 nominal conditions × 4 configurations."],
  ["Grid 2 completeness", 1080, "PASS", "30 seeds × 12 nominal conditions × 3 ablation configurations."],
  ["Grid reproducibility manifests", 2, "PASS", "SHA-256 manifests match the current source revision."],
  ["Robustness Monte Carlo", 20, "DIAGNOSTIC", "50% valid-finite rate for both controllers; not a final safety claim."],
  ["Linear stability", null, "LIMITATION", "Open-loop instability above approximately payload 0.7; closed-loop local result is not hardware proof."],
];

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Scenario Summary");
const details = workbook.worksheets.add("Metric Details");
const qa = workbook.worksheets.add("QA Evidence");
const method = workbook.worksheets.add("Methodology");
for (const sheet of [summary, details, qa, method]) sheet.showGridLines = false;

const navy = "#17365D", blue = "#D9EAF7", green = "#E2F0D9", amber = "#FFF2CC", red = "#FCE4D6", gray = "#666666";
const headerFormat = { fill: navy, font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
const titleFormat = { font: { name: "Arial", size: 14, bold: true, color: navy } };
const noteFormat = { font: { name: "Arial", size: 10, italic: true, color: gray }, wrapText: true };

summary.getRange("A1:L1").merge();
summary.getRange("A1").values = [["Where the Intelligent Controller Is Better"]];
summary.getRange("A1:L1").format = titleFormat;
summary.getRange("A2:L2").merge();
summary.getRange("A2").values = [["30 matched seeds per scenario. Positive valid-rate delta means more completed valid runs for the intelligent controller. Tracking and effort wins are counts of lower-is-better metrics on common valid seeds."]];
summary.getRange("A2:L2").format = noteFormat;
const summaryHeaders = ["Payload level", "Gust", "Airspeed (m/s)", "Conventional valid rate", "Intelligent valid rate", "Valid-rate delta (percentage points)", "Paired valid runs", "Intelligent tracking wins", "Tracking metrics", "Intelligent effort wins", "Effort metrics", "Interpretation"];
summary.getRange("A4:L4").values = [summaryHeaders];
summary.getRange("A4:L4").format = headerFormat;
summary.getRange(`A5:L${4 + scenarioRows.length}`).values = scenarioRows.map(r => [r.payload, r.gust, r.airspeed, r.conventional_valid_rate, r.intelligent_valid_rate, r.valid_rate_delta_pp, r.paired_valid_n, r.intelligent_tracking_wins, r.tracking_metric_count, r.intelligent_effort_wins, r.effort_metric_count, `${r.classification}: ${r.note}`]);
summary.getRange(`D5:E${4 + scenarioRows.length}`).format.numberFormat = "0.0%";
summary.getRange(`F5:F${4 + scenarioRows.length}`).format.numberFormat = "0.0";
summary.getRange(`A4:L${4 + scenarioRows.length}`).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
summary.getRange(`A5:L${4 + scenarioRows.length}`).format.wrapText = true;
summary.getRange(`L5:L${4 + scenarioRows.length}`).format.columnWidth = 55;
summary.getRange("A:A").format.columnWidth = 13;
summary.getRange("B:C").format.columnWidth = 14;
summary.getRange("D:F").format.columnWidth = 18;
summary.getRange("G:K").format.columnWidth = 15;
summary.freezePanes.freezeRows(4);

details.getRange("A1:K1").merge();
details.getRange("A1").values = [["Metric-level matched comparison"]];
details.getRange("A1:K1").format = titleFormat;
details.getRange("A2:K2").merge();
details.getRange("A2").values = [["Means use only seeds where both controllers completed validly. For pitch_settled, 1 means settled and 0 means not settled. For pitch steady-state error, absolute value is used because closeness to zero is better."]];
details.getRange("A2:K2").format = noteFormat;
const detailHeaders = ["Payload", "Gust", "Airspeed (m/s)", "Metric", "Paired valid runs", "Conventional mean", "Intelligent mean", "Intelligent change (%)", "Intelligent better", "Conventional SD", "Intelligent SD"];
details.getRange("A4:K4").values = [detailHeaders];
details.getRange("A4:K4").format = headerFormat;
details.getRange(`A5:K${4 + detailRows.length}`).values = detailRows.map(r => [r.payload, r.gust, r.airspeed, r.metric, r.paired_valid_n, r.conventional_mean, r.intelligent_mean, r.intelligent_change_percent === null ? null : r.intelligent_change_percent / 100, r.intelligent_better === null ? "N/A" : r.intelligent_better ? "YES" : "NO", r.conventional_sd, r.intelligent_sd]);
details.getRange(`F5:G${4 + detailRows.length}`).format.numberFormat = "0.000";
details.getRange(`H5:H${4 + detailRows.length}`).format.numberFormat = "0.0%";
details.getRange(`A4:K${4 + detailRows.length}`).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
details.getRange(`A5:K${4 + detailRows.length}`).format.wrapText = false;
details.getRange("A:C").format.columnWidth = 14;
details.getRange("D:D").format.columnWidth = 28;
details.getRange("E:K").format.columnWidth = 17;
details.freezePanes.freezeRows(4);

qa.getRange("A1:D1").merge();
qa.getRange("A1").values = [["Testing and evidence completed"]];
qa.getRange("A1:D1").format = titleFormat;
qa.getRange("A3:D3").values = [["Evidence item", "Count/size", "Status", "Interpretation"]];
qa.getRange("A3:D3").format = headerFormat;
qa.getRange(`A4:D${3 + qaTests.length}`).values = qaTests;
qa.getRange(`A4:D${3 + qaTests.length}`).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
qa.getRange(`D4:D${3 + qaTests.length}`).format.wrapText = true;
qa.getRange("A:A").format.columnWidth = 32;
qa.getRange("B:B").format.columnWidth = 14;
qa.getRange("C:C").format.columnWidth = 15;
qa.getRange("D:D").format.columnWidth = 75;
qa.freezePanes.freezeRows(3);

method.getRange("A1:B1").merge();
method.getRange("A1").values = [["How to read this report"]];
method.getRange("A1:B1").format = titleFormat;
method.getRange("A3:B11").values = [
  ["Controller comparison", "Conventional is fixed-gain PID. Intelligent is the NN payload estimate plus adaptive tail/gain logic."],
  ["Scenario grid", "Payload 0.2/0.5/0.7, gust 0.5/1.5, airspeed 12/20 m/s, 30 matched seeds, 15-second runs."],
  ["Fairness", "Each conventional/intelligent pair uses the same seed and scenario, so the disturbance realization is matched."],
  ["Valid rate", "A run is valid only if it completes without the declared Euler, aerodynamic, or safety termination condition."],
  ["Tracking metrics", "Lower maximum deviation, RMS error, altitude error/drop, and absolute steady-state error is better."],
  ["Effort metrics", "Lower average elevator, aileron, and rudder magnitude is generally better, but may trade against tracking quality."],
  ["Interpretation rule", "Intelligent candidate means better valid rate and at least 5 of 9 tracking metrics. Mixed means no single controller dominates."],
  ["Source", "backend/results_grid_final/raw_runs.jsonl and grid_config.json; current source revision is hash-checked in the manifest."],
  ["Limitation", "These are simulation results under assumed aerodynamic parameters. They are not hardware or flight-test evidence."],
];
method.getRange("A3:B11").format.wrapText = true;
method.getRange("A3:B11").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
method.getRange("A:A").format.columnWidth = 25;
method.getRange("B:B").format.columnWidth = 100;

for (const sheet of [summary, details, qa, method]) {
  const used = sheet.getUsedRange();
  used.format.font = { name: "Arial", size: 10 };
  used.format.verticalAlignment = "center";
}
summary.getRange("A1:L1").format = titleFormat;
summary.getRange("A2:L2").format = noteFormat;
details.getRange("A1:K1").format = titleFormat;
details.getRange("A2:K2").format = noteFormat;
qa.getRange("A1:D1").format = titleFormat;
method.getRange("A1:B1").format = titleFormat;

workbook.recalculate();
const preview = await workbook.render({ sheetName: "Scenario Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "controller_scenario_report_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(outputDir, "controller_scenario_report.xlsx"));

const md = [
  "# Controller Scenario Report",
  "",
  "Source: `backend/results_grid_final/raw_runs.jsonl`; 30 matched seeds per scenario.",
  "",
  "## Scenarios",
  "",
  "| Payload | Gust | Airspeed | Conv valid | Intelligent valid | Delta (pp) | Tracking wins | Effort wins | Interpretation |",
  "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
  ...scenarioRows.map(r => `| ${r.payload} | ${r.gust} | ${r.airspeed} | ${(100*r.conventional_valid_rate).toFixed(1)}% | ${(100*r.intelligent_valid_rate).toFixed(1)}% | ${r.valid_rate_delta_pp.toFixed(1)} | ${r.intelligent_tracking_wins}/${r.tracking_metric_count} | ${r.intelligent_effort_wins}/${r.effort_metric_count} | ${r.classification} |`),
  "",
  "## Interpretation rule",
  "",
  "An `Intelligent candidate` has a higher valid-run rate and wins at least 5 of 9 tracking metrics on common valid seeds. A `Mixed result` means there is a trade-off. These labels are decision aids, not proof of hardware superiority.",
  "",
  "## QA result",
  "",
  "The automated test suite ran 559 tests and passed. Existing nominal grids were complete and their manifests were hash-checked. The 20-trial uncertainty campaign remains diagnostic: both controllers had 50% valid-finite rate, so it should not be presented as final safety evidence.",
].join("\n");
await fs.writeFile(path.join(outputDir, "controller_scenario_report.md"), md, "utf8");

console.log(JSON.stringify({ output: path.join(outputDir, "controller_scenario_report.xlsx"), markdown: path.join(outputDir, "controller_scenario_report.md"), preview: path.join(outputDir, "controller_scenario_report_preview.png"), scenarios: scenarioRows }, null, 2));
