// Chart.js wrappers. Plots only: every number drawn here was computed by
// the server. Colours are read from the CSS variables at draw time, and
// every chart redraws when the theme changes.
import { colors, onThemeChange } from "./theme.js";
import { fmtHzTick } from "./util.js";

const FONT = '"Martian Mono", ui-monospace, monospace';
const LOG_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

Chart.defaults.font.family = FONT;
Chart.defaults.font.size = 12;

const charts = new Set();
onThemeChange(() => charts.forEach((chart) => chart.render()));

// A chart built by build(colors, data) -> Chart.js config. set(data)
// redraws it; a theme change rebuilds it with the new colours.
export class ThemedChart {
  constructor(canvas, build) {
    this.canvas = canvas;
    this.build = build;
    this.data = null;
    this.chart = null;
    charts.add(this);
  }

  set(data) {
    this.data = data;
    this.render();
  }

  render() {
    if (this.data === null) return;
    const config = this.build(colors(), this.data);
    if (this.chart && this.chart.config.type === config.type) {
      this.chart.data = config.data;
      this.chart.options = config.options;
      this.chart.update("none");
    } else {
      if (this.chart) this.chart.destroy();
      this.chart = new Chart(this.canvas, config);
    }
  }

  resize() {
    if (this.chart) this.chart.resize();
  }

  destroy() {
    if (this.chart) this.chart.destroy();
    charts.delete(this);
  }
}

export const xy = (xs, ys) => xs.map((x, i) => ({ x, y: ys[i] }));

export const fmt = {
  hz: (v) => fmtHzTick(v),
  db: (v) => `${v > 0 ? "+" : ""}${+v.toFixed(1)}`,
  s: (v) => `${+v.toFixed(3)}`,
  num: (v) => `${+v.toPrecision(3)}`,
};

// spec: { title, type?, min?, max?, step?, suggested?, format?, zeroLine? }
// suggested: min / max are only hints, so Chart.js may round them to
// nice tick values. step: fixed tick spacing (e.g. to put a tick on 0 dB).
function axis(c, spec) {
  const zero = (ctx) => spec.zeroLine && ctx.tick && ctx.tick.value === 0;
  const ticks = { color: c.fgDim, font: { size: 12 }, maxRotation: 0, autoSkipPadding: 14, padding: 6 };
  if (spec.format) ticks.callback = (value) => spec.format(value);
  if (spec.step) {
    ticks.stepSize = spec.step;
    ticks.autoSkip = false; // skipping could drop the 0 dB tick and its reference line
  }
  const result = {
    type: spec.type || "linear",
    title: { display: true, text: spec.title, color: c.fgDim, font: { size: 12 } },
    ticks,
    grid: {
      color: (ctx) => (zero(ctx) ? c.fgDim : c.line),
      lineWidth: (ctx) => (zero(ctx) ? 2 : 1),
      drawTicks: false,
    },
    border: { color: c.line, width: 2 },
  };
  const bound = (key) => (spec.suggested ? `suggested${key[0].toUpperCase()}${key.slice(1)}` : key);
  if (spec.min !== undefined) result[bound("min")] = spec.min;
  if (spec.max !== undefined) result[bound("max")] = spec.max;
  if (spec.type === "logarithmic") {
    // Only 1-2-5 decades, so the labels never collide.
    result.afterBuildTicks = (scale) => {
      scale.ticks = LOG_TICKS.filter((v) => v >= scale.min && v <= scale.max).map((value) => ({ value }));
    };
  }
  return result;
}

// Options shared by every line plot. x / y: axis specs (above).
// legend: show the dataset labels (for before/after pairs).
export function lineOptions(c, { x, y, legend = false, mode = "index" }) {
  const xFormat = x.format || fmt.num;
  const yFormat = y.format || fmt.num;
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    normalized: true,
    interaction: { mode, axis: "x", intersect: false },
    layout: { padding: { top: 4, right: 6 } },
    elements: {
      point: { radius: 0, hoverRadius: 3, hitRadius: 6 },
      line: { borderWidth: 2, tension: 0 },
    },
    scales: { x: axis(c, x), y: axis(c, y) },
    plugins: {
      legend: legend
        ? {
            display: true,
            position: "top",
            align: "end",
            labels: {
              color: c.fg,
              boxWidth: 18,
              boxHeight: 2,
              padding: 12,
              font: { size: 12 },
              filter: (item, data) => !data.datasets[item.datasetIndex].hideInLegend,
              sort: (a, b) => a.datasetIndex - b.datasetIndex, // original first, whatever the draw order
            },
          }
        : { display: false },
      tooltip: {
        backgroundColor: c.panel,
        borderColor: c.line,
        borderWidth: 2,
        cornerRadius: 0,
        caretSize: 0,
        padding: 8,
        titleColor: c.fg,
        bodyColor: c.fg,
        titleFont: { family: FONT, size: 12, weight: "500" },
        bodyFont: { family: FONT, size: 12 },
        displayColors: legend,
        boxWidth: 10,
        boxHeight: 2,
        filter: (item) => !item.dataset.hideInTooltip,
        itemSort: (a, b) => a.datasetIndex - b.datasetIndex,
        callbacks: {
          title: (items) => (items.length ? `${xFormat(items[0].parsed.x)} ${x.unit || ""}` : ""),
          label: (item) => `${item.dataset.label}: ${yFormat(item.parsed.y)} ${y.unit || ""}`,
        },
      },
    },
  };
}

// One 2px trace. role: "before" (--fg-dim) | "after" (--accent). After
// traces are drawn on top (Chart.js draws lower `order` last).
export function trace(c, label, data, role = "after", extra = {}) {
  return {
    label,
    data,
    borderColor: role === "before" ? c.fgDim : c.accent,
    backgroundColor: role === "before" ? c.fgDim : c.accent,
    borderWidth: 2,
    pointRadius: 0,
    order: role === "before" ? 2 : 1,
    ...extra,
  };
}
