// Backstage: one global tab with a wall of analysis figures for one run.
// Every figure has the same shape: "FIG 0N / TITLE", a one-line
// "LOOK FOR:" caption and a 2px border. All numbers come from
// GET /backstage/<run_id>; the page only draws them.
import { CONFIG, toolInfo } from "./config.js";
import { ThemedChart, fmt, lineOptions, trace, xy } from "./charts.js";
import { Segmented, formatValue } from "./controls.js";
import { ResponsePlot, responseQuery } from "./responses.js";
import { withAlpha } from "./theme.js";
import { el, fmtClock, requestJSON, storageGet, storageSet } from "./util.js";

const SESSION_KEY = "spectra-runs"; // this browser tab's runs, kept across reloads

// ---------------------------------------------------------------------
// Run labels: "REVERB · 14:32 · RT60 1.5s WET 0.3"
// ---------------------------------------------------------------------

const num = (value, digits = 2) => +Number(value).toFixed(digits);

function keyParams(run) {
  const p = run.params;
  switch (run.tool) {
    case "reverb":
      return `RT60 ${num(p.rt60)}s WET ${num(p.wet)}${p.circular ? " CIRCULAR" : ""}`;
    case "echo":
      return `${p.mode.toUpperCase()} D ${num(p.delay_ms, 0)}ms g ${num(p.gain)}`;
    case "flanger":
      return `RATE ${num(p.rate_hz)}Hz SWEEP ${num(p.sweep_ms, 1)}ms g ${num(p.gain)}`;
    case "eq": {
      const parts = [p.mode.toUpperCase()];
      if (p.mode === "hum") parts.push(`${num(p.hum_freq, 0)}Hz ×${num(p.harmonics, 0)}`);
      if (p.mode === "filter" || p.mode === "both") {
        const two = p.filter_type === "bandpass" || p.filter_type === "bandstop";
        const cut = two ? `${num(p.low_cutoff, 0)}-${num(p.high_cutoff, 0)}Hz` : `${num(p.cutoff, 0)}Hz`;
        parts.push(`${p.filter_type.toUpperCase()} ${cut} N${num(p.order, 0)}`);
      }
      if (p.mode === "eq" || p.mode === "both") {
        const bands = (p.bands || [])
          .map((b, i) => [CONFIG.eq.bands[i], b.gain_db])
          .filter(([, g]) => g !== 0)
          .map(([spec, g]) => `${spec.label.split(" ")[0].toUpperCase()} ${g > 0 ? "+" : ""}${num(g, 1)}`);
        parts.push(bands.length ? bands.join(" ") : "FLAT");
      }
      return parts.join(" ");
    }
    case "separate":
      return `${run.outputs.stems.length} STEMS${p.mock ? " (MOCK)" : ""}`;
    default:
      return "";
  }
}

const toolName = (tool) => (toolInfo(tool) ? toolInfo(tool).name.toUpperCase() : tool.toUpperCase());
const runLabel = (run) => `${toolName(run.tool)} · ${fmtClock(run.created)} · ${keyParams(run)}`;

// Run card rows: [label, value, unit] with the sliders' labels and units.
function paramRows(run) {
  const p = run.params;
  const specFor = (list, name) => list.find((s) => s.name === name);
  // The equation letter keeps its case: "GAIN g", not "GAIN G".
  const label = (spec) => (spec.symbol ? [spec.label, " ", el("span", { class: "sym" }, spec.symbol)] : spec.label);
  const fromSpec = (spec, value) => [label(spec), ...formatValue(spec, value)];
  const option = (list, value) => (list.find((o) => o.value === value) || { label: value }).label;

  if (run.tool === "eq") {
    const cfg = CONFIG.eq;
    const rows = [["Mode", option(cfg.modes, p.mode), ""]];
    if (p.mode === "eq" || p.mode === "both") {
      (p.bands || []).forEach((band, i) => rows.push(fromSpec(cfg.bands[i], band.gain_db)));
    }
    if (p.mode === "filter" || p.mode === "both") {
      rows.push(["Filter", option(cfg.filter_types, p.filter_type), ""]);
      const two = p.filter_type === "bandpass" || p.filter_type === "bandstop";
      for (const name of two ? ["low_cutoff", "high_cutoff", "order"] : ["cutoff", "order"]) {
        rows.push(fromSpec(specFor(cfg.filter, name), p[name]));
      }
    }
    if (p.mode === "hum") cfg.hum.forEach((spec) => rows.push(fromSpec(spec, p[spec.name])));
    return rows;
  }
  if (run.tool === "separate") {
    const rows = [
      ["Stems", run.outputs.stems.map((s) => s.name).join(", "), ""],
      ["Module", p.mock ? "mock (copies of the input)" : "NMF separation", ""],
    ];
    if (run.analysed_seconds !== undefined) {
      rows.push(["Analysed", run.truncated ? `first ${num(run.analysed_seconds, 0)}` : "whole track",
        run.truncated ? "s" : ""]);
    }
    return rows;
  }
  const cfg = CONFIG[run.tool];
  const rows = [];
  if (run.tool === "echo") rows.push(["Mode", p.mode, ""]);
  cfg.sliders.forEach((spec) => rows.push(fromSpec(spec, p[spec.name])));
  if (run.tool === "reverb") rows.push(["Convolution", p.circular ? "circular (wrong way)" : "linear", ""]);
  return rows;
}

// ---------------------------------------------------------------------
// Captions
// ---------------------------------------------------------------------

const LOOK = {
  spectrograms: {
    eq: "where the processed side goes darker (cut) or brighter (boosted) in the filtered bands.",
    reverb: "energy smeared to the right of every note: the room's tail.",
    echo: "every hit repeated D later, fainter each time.",
    flanger: "thin dark notches drifting up and down the spectrum.",
  },
  response: {
    eq: "the gain curve the audio was multiplied by; 0 dB leaves a frequency untouched.",
    reverb: "h(t), the room's echo pattern: silence for the pre-delay, then a decay over RT60.",
    echo: "comb teeth every 1/D Hz; feedback makes them sharper.",
    flanger: "the notches moving as D(n) sweeps (same animation as the tool).",
  },
  spectrum: {
    eq: "the gap between the curves follows the gain curve in FIG 03.",
    reverb: "reverb adds energy; the tail fills the gaps between notes.",
    echo: "the combined repeats raise the level; fine comb ripples average out.",
    flanger: "moving notches average out, so the average spectrum barely changes.",
  },
};

// ---------------------------------------------------------------------
// Figure helpers
// ---------------------------------------------------------------------

function table(head, rows, { changeColumn = -1 } = {}) {
  return el("table", { class: "data-table" },
    el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { class: i ? "num" : null }, h)))),
    el("tbody", {}, rows.map((row) => el("tr", {},
      row.map((cell, i) => el(i ? "td" : "th", {
        class: [i ? "num" : null, i === changeColumn ? "change" : null].filter(Boolean).join(" ") || null,
        scope: i ? null : "row",
      }, cell))))));
}

const dbText = (v) => (v === null || v === undefined ? "−∞" : v.toFixed(2));
const change = (a, b) => {
  if (a === null || b === null || a === undefined || b === undefined) return "–";
  const d = b - a;
  return `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(2)}`;
};

// Envelope {duration, max, min} -> two datasets: the max line, and the
// min line filled up to it (or an unfilled outline, to overlay on top).
// order: lower is drawn later, i.e. on top.
function envelopeSets(c, label, env, color, { order = 1, filled = true } = {}) {
  const t = (i) => (i * env.duration) / env.max.length;
  const common = { borderColor: color, borderWidth: 1.5, pointRadius: 0, order };
  return [
    { ...common, label, data: env.max.map((y, i) => ({ x: t(i), y })), backgroundColor: color, fill: false },
    {
      ...common, label: `${label} min`, data: env.min.map((y, i) => ({ x: t(i), y })),
      backgroundColor: withAlpha(color, 0.28), fill: filled ? "-1" : false,
      hideInLegend: true, hideInTooltip: true,
    },
  ];
}

export class Backstage {
  constructor(root) {
    this.root = root;
    this.picker = root.querySelector("[data-bs-runs]");
    this.wall = root.querySelector("[data-bs-wall]");
    this.selected = null;
    this.seq = 0;
    this.disposables = [];
    try {
      this.runIds = JSON.parse(storageGet("sessionStorage", SESSION_KEY) || "[]");
    } catch {
      this.runIds = [];
    }
  }

  record(runId) {
    this.runIds.push(runId);
    storageSet("sessionStorage", SESSION_KEY, JSON.stringify(this.runIds));
  }

  // The tab became visible; runId: select this run (→ BACKSTAGE).
  async shown(runId = null) {
    let runs;
    try {
      const data = await requestJSON("/backstage/runs");
      const mine = new Set(this.runIds);
      runs = data.runs.filter((run) => mine.has(run.run_id));
    } catch (err) {
      this.showMessage(`Could not load the runs: ${err.message}`, true);
      return;
    }
    this.runs = runs;
    this.renderPicker();
    if (!runs.length) {
      this.selected = null;
      this.clearWall();
      this.showMessage("No runs yet. Process a track in any tool, then come back here.");
      return;
    }
    const known = (id) => runs.some((run) => run.run_id === id);
    const target = runId && known(runId) ? runId : known(this.selected) ? this.selected : runs[0].run_id;
    if (target !== this.selected || !this.wall.childElementCount) this.select(target);
    else this.markSelected();
  }

  renderPicker() {
    this.picker.replaceChildren(...this.runs.map((run) => el("button", {
      type: "button", class: "run-chip", "data-run": run.run_id, "aria-pressed": "false",
      onclick: () => this.select(run.run_id),
    }, runLabel(run))));
    this.markSelected();
  }

  markSelected() {
    for (const chip of this.picker.children) {
      const on = chip.dataset.run === this.selected;
      chip.setAttribute("aria-pressed", String(on));
      if (on) chip.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }

  clearWall() {
    this.disposables.forEach((d) => d.destroy());
    this.disposables = [];
    this.wall.replaceChildren();
  }

  showMessage(text, warn = false) {
    this.wall.replaceChildren(el("p", { class: warn ? "msg msg-warn bs-message" : "empty bs-message" }, text));
  }

  async select(runId) {
    this.selected = runId;
    this.markSelected();
    const seq = ++this.seq;
    this.clearWall();
    this.showMessage("Analyzing this run… (the first look at a long track takes a few seconds)");
    let data;
    try {
      data = await requestJSON(`/backstage/${runId}`);
    } catch (err) {
      if (seq === this.seq) this.showMessage(`Could not analyze this run: ${err.message}`, true);
      return;
    }
    if (seq !== this.seq) return; // another run was picked meanwhile
    this.wall.replaceChildren();
    this.figureCount = 0;
    if (data.run.tool === "separate") this.renderSeparation(data);
    else this.renderEffect(data);
  }

  // One figure: "FIG 0N / TITLE", LOOK FOR caption, content. width: grid columns of 12.
  figure(title, look, width, ...content) {
    this.figureCount += 1;
    const n = String(this.figureCount).padStart(2, "0");
    const fig = el("section", { class: `fig w-${width}`, "aria-label": `Figure ${n}: ${title}` },
      el("header", { class: "fig-head" },
        el("h3", { class: "fig-title" }, el("span", { class: "fig-num" }, `FIG ${n}`), ` / ${title}`),
        el("p", { class: "look" }, el("b", {}, "LOOK FOR: "), look)),
      ...content);
    this.wall.append(fig);
    return fig;
  }

  chart(build, data, { height = null } = {}) {
    const canvas = el("canvas", { role: "img" });
    const box = el("div", { class: "chart-box" }, canvas);
    if (height) box.style.height = `${height}px`;
    // Build after the box is in the page, so Chart.js can size it.
    requestAnimationFrame(() => {
      const chart = new ThemedChart(canvas, build);
      this.disposables.push(chart);
      chart.set(data);
    });
    return box;
  }

  // -------------------------------------------------------------------
  // Effect runs: Tier 1 (1-6), Tier 2 (7-8), Tier 3 (9-11)
  // -------------------------------------------------------------------

  renderEffect(data) {
    const { run } = data;
    const tool = run.tool;

    // 1. Spectrograms
    this.figure("Spectrograms", `${LOOK.spectrograms[tool]} Grey frame = original, cyan = processed.`, 12,
      el("div", { class: "spec-pair" },
        el("figure", { class: "spec spec-before" },
          el("figcaption", { class: "label" }, "Original"),
          el("img", { src: data.images.before, alt: "Spectrogram of the original", loading: "lazy" })),
        el("figure", { class: "spec spec-after" },
          el("figcaption", { class: "label spec-label-after" }, "Processed"),
          el("img", { src: data.images.after, alt: "Spectrogram of the processed audio", loading: "lazy" }))));

    // 2. Difference spectrogram
    const d = data.diff;
    let diffLook = "cyan = frequencies boosted, magenta = cut, black = unchanged.";
    if (d.output_longer_s > 0.01) {
      diffLook += ` The output is ${d.output_longer_s.toFixed(2)} s longer (its tail); this covers the original ${d.seconds.toFixed(1)} s only.`;
    }
    this.figure("Difference", diffLook, 7,
      el("img", { class: "fig-img", src: data.images.diff, alt: "Difference spectrogram, processed minus original, in dB" }),
      el("p", { class: "fig-stats" },
        `Clipped to ±${d.clip_db} dB · largest change ${d.max_abs_db.toFixed(1)} dB · `,
        `${d.boosted_pct}% of bins boosted, ${d.cut_pct}% cut (by more than 1 dB)`));

    // 3. System response: the same plot as the tool, for this run's params
    this.responseFigure(run);

    // 4. Average spectrum
    const spec = data.avg_spectrum;
    const top = Math.max(...spec.before_db, ...spec.after_db);
    this.figure("Average spectrum", LOOK.spectrum[tool], 6,
      this.chart((c) => ({
        type: "line",
        data: {
          datasets: [
            trace(c, "ORIGINAL", xy(spec.freqs, spec.before_db), "before"),
            trace(c, "PROCESSED", xy(spec.freqs, spec.after_db), "after"),
          ],
        },
        options: lineOptions(c, {
          legend: true,
          x: { type: "logarithmic", min: 20, max: 20000, title: "FREQUENCY (Hz)", format: fmt.hz, unit: "Hz" },
          y: { min: Math.floor((top - 80) / 20) * 20, max: Math.ceil((top + 5) / 20) * 20, step: 20,
               title: "LEVEL (dBFS)", format: fmt.db, unit: "dB" },
        }),
      }), spec),
      el("p", { class: "fig-stats" }, "Both averaged over the original's length, so a tail's energy counts."));

    // 5. Waveform overlay
    const env = data.envelopes;
    this.figure("Waveform overlay",
      "level changes, and tails: the cyan envelope runs past the grey one when the effect adds a tail.", 6,
      this.chart((c) => ({
        type: "line",
        data: {
          datasets: [
            ...envelopeSets(c, "ORIGINAL", env.before, c.fgDim, { order: 2 }),
            ...envelopeSets(c, "PROCESSED", env.after, c.accent, { order: 1 }),
          ],
        },
        options: lineOptions(c, {
          legend: true,
          mode: "nearest",
          x: { min: 0, max: Math.max(env.before.duration, env.after.duration), title: "TIME (s)", format: fmt.s, unit: "s" },
          y: { min: -1, max: 1, step: 0.5, title: "AMPLITUDE (FULL SCALE)", zeroLine: true },
        }),
      }), env));

    // 6. Run card
    this.runCard(run, 4);

    // 7. Level meters
    const L = data.levels;
    this.figure("Levels", "peak and RMS before vs after; crest factor = peak − RMS, how spiky the sound is.", 4,
      table(["", "Original", "Processed", "Change"], [
        ["Peak (dBFS)", dbText(L.before.peak_dbfs), dbText(L.after.peak_dbfs), change(L.before.peak_dbfs, L.after.peak_dbfs)],
        ["RMS (dBFS)", dbText(L.before.rms_dbfs), dbText(L.after.rms_dbfs), change(L.before.rms_dbfs, L.after.rms_dbfs)],
        ["Crest (dB)", dbText(L.before.crest_db), dbText(L.after.crest_db), change(L.before.crest_db, L.after.crest_db)],
        ["Duration (s)", L.before.duration.toFixed(2), L.after.duration.toFixed(2), change(L.before.duration, L.after.duration)],
      ], { changeColumn: 3 }));

    // 8. STFT settings
    this.stftFigure(data.stft, tool, 4);

    // 9-11. Tool-specific
    const extra = data.extra || {};
    if (extra.reverb_start) this.reverbStartFigure(extra.reverb_start);
    if (extra.echo_ir) this.echoIrFigure(extra.echo_ir);
    if (extra.flanger_lfo) this.flangerLfoFigure(extra.flanger_lfo, run.params);
  }

  responseFigure(run) {
    const tool = run.tool;
    const tools = el("div", { class: "block-tools" });
    const canvas = el("canvas", { role: "img", "aria-label": "System response" });
    const msg = el("p", { class: "chart-msg", hidden: true });
    this.figure("System response", LOOK.response[tool], 5,
      tools, el("div", { class: "chart-box" }, canvas, msg));

    const plot = new ResponsePlot(canvas, tool, { tools, rate: () => run.params.rate_hz });
    this.disposables.push(plot);
    let echoZoom = tool === "echo" ? CONFIG.echo.zooms[0].f_max : null;
    let seq = 0;
    const load = async () => {
      const mine = ++seq;
      const query = responseQuery(tool, run.params, { echoZoom });
      try {
        const data = await requestJSON(`/response/${tool}?${new URLSearchParams(query)}`);
        if (mine === seq) plot.set(data, query);
      } catch (err) {
        msg.textContent = err.message;
        msg.hidden = false;
      }
    };
    if (tool === "echo") {
      const zoom = new Segmented(
        CONFIG.echo.zooms.map((z) => ({ value: String(z.f_max), label: z.label })), String(echoZoom),
        (value) => {
          echoZoom = value === "null" ? null : Number(value);
          load();
        }, { label: "Frequency zoom", compact: true });
      tools.append(el("span", { class: "label" }, "Zoom"), zoom.root);
    }
    requestAnimationFrame(load);
  }

  runCard(run, width) {
    const info = toolInfo(run.tool);
    const source = run.source && run.source.filename ? run.source.filename : "–";
    this.figure("Run card", "the exact settings behind every other figure on this wall.", width,
      el("dl", { class: "kv" },
        el("dt", {}, "Tool"), el("dd", {}, info ? info.name : run.tool),
        el("dt", {}, "Source"), el("dd", {}, source),
        el("dt", {}, "Run"), el("dd", {}, `${fmtClock(run.created)} · ${run.run_id.slice(0, 8)}`),
        paramRows(run).flatMap(([label, value, unit]) => [
          el("dt", {}, label),
          el("dd", {}, el("span", { class: "kv-value" }, value), unit ? ` ${unit}` : ""),
        ])),
      el("pre", { class: "equation" }, (info ? info.equations : []).join("\n")));
  }

  stftFigure(stft, tool, width) {
    const cols = [["Analysis", stft.analysis]];
    if (stft.processing) cols.push(["EQ processing", stft.processing]);
    const row = (label, get) => [label, ...cols.map(([, s]) => get(s))];
    const look = stft.processing
      ? "long frames give fine frequency resolution but poor time resolution."
      : "Δf = sr/N is how close two tones can be and still be told apart; N/sr is how much time one frame blurs.";
    this.figure("STFT settings", look, width,
      table(["", ...cols.map(([name]) => name)], [
        row("Frame size N", (s) => `${s.frame_size}`),
        row("Hop", (s) => `${s.hop_size} (${s.hop_ms.toFixed(1)} ms)`),
        row("Window", (s) => s.window),
        row("Sample rate", (s) => `${s.sr} Hz`),
        row("Δf = sr/N", (s) => `${s.df_hz.toFixed(2)} Hz`),
        row("N/sr", (s) => `${s.dt_ms.toFixed(1)} ms`),
      ]));
  }

  reverbStartFigure(cmp) {
    this.figure("Linear vs circular: first 2 s",
      "in circular mode the tail wraps onto the start.", 12,
      this.chart((c) => ({
        type: "line",
        data: {
          datasets: [
            ...envelopeSets(c, "LINEAR (CORRECT)", cmp.linear, c.accent, { order: 2 }),
            // An outline on top, so the wrapped tail shows where it leaves the linear fill.
            ...envelopeSets(c, "CIRCULAR (WRONG WAY)", cmp.circular, c.warn, { order: 1, filled: false }),
          ],
        },
        options: lineOptions(c, {
          legend: true,
          mode: "nearest",
          x: { min: 0, max: cmp.duration, title: "TIME (s)", format: fmt.s, unit: "s" },
          y: { title: "AMPLITUDE (FULL SCALE)", zeroLine: true },
        }),
      }), cmp),
      el("p", { class: "fig-stats" },
        `Mono mix, before the clip protection. This run used ${cmp.run_circular ? "circular" : "linear"} convolution.`));
  }

  echoIrFigure(ir) {
    // A stem plot: a vertical line from 0 to g^k at each k·D.
    const stems = ir.t.flatMap((t, i) => [{ x: t, y: 0 }, { x: t, y: ir.h[i] }, { x: t, y: NaN }]);
    const heads = ir.t.map((t, i) => ({ x: t, y: ir.h[i] }));
    const span = ir.t[ir.t.length - 1] || ir.delay_ms / 1000;
    const xMax = Math.ceil((span * 1.05 + 0.01) * 4) / 4; // a round number of seconds
    this.figure("Impulse response",
      ir.mode === "feedforward"
        ? "one repeat of height g at D: the feedforward echo is an FIR system."
        : "a decaying train g, g², g³… every D: feedback makes an IIR system.", 12,
      this.chart((c) => {
        const options = lineOptions(c, {
          x: { min: 0, max: xMax, title: "TIME (s)", format: fmt.s, unit: "s" },
          y: { min: 0, max: 1.2, step: 0.2, title: "h[n] (LINEAR)", format: (v) => v.toFixed(1) },
        });
        options.normalized = false;
        options.spanGaps = false;
        return {
          type: "line",
          data: {
            datasets: [
              trace(c, "h[n]", stems, "after", { hideInTooltip: true }),
              trace(c, "h[n]", heads, "after", {
                showLine: false, pointRadius: 4, pointHoverRadius: 5, pointStyle: "rect",
                pointBackgroundColor: c.accent,
              }),
            ],
          },
          options,
        };
      }, ir));
  }

  flangerLfoFigure(lfo, params) {
    this.figure("Delay sweep D(n)",
      `the delay rises and falls once per LFO cycle (${num(params.rate_hz)} Hz); the notches follow it.`, 12,
      this.chart((c) => ({
        type: "line",
        data: { datasets: [trace(c, "D(n)", xy(lfo.t, lfo.delay_ms))] },
        options: lineOptions(c, {
          x: { min: 0, max: lfo.t[lfo.t.length - 1], title: "TIME (s)", format: fmt.s, unit: "s" },
          y: { min: 0, max: Math.ceil(Math.max(...lfo.delay_ms) + 0.5), title: "DELAY D (ms)", unit: "ms" },
        }),
      }), lfo));
  }

  // -------------------------------------------------------------------
  // Separation runs
  // -------------------------------------------------------------------

  renderSeparation(data) {
    const { run } = data;
    const mock = run.params.mock;
    this.figure("Mixture spectrogram", "where each instrument's energy sits before it is split.", 6,
      el("img", { class: "fig-img", src: data.images.mixture, alt: "Spectrogram of the mixture" }));

    const L = data.levels.mixture;
    this.figure("Stem levels", "how much of the mixture's level each stem carries.", 6,
      table(["", "Peak", "RMS", "Crest", ""], [
        ["Mixture", dbText(L.peak_dbfs), dbText(L.rms_dbfs), dbText(L.crest_db), ""],
        ...data.stems.map((stem) => [
          stem.name, dbText(stem.levels.peak_dbfs), dbText(stem.levels.rms_dbfs), dbText(stem.levels.crest_db),
          el("a", { href: data.audio.stems[stem.name].download_url, download: "" }, "WAV"),
        ]),
      ]),
      el("p", { class: "fig-stats" }, "Peak and RMS in dBFS, crest in dB."));

    this.figure("Stem masks",
      `bright = the stem owns that bin (|stem| / |mixture|), black = silent in the mixture; mid values are where stems overlap and bleed.${mock ? " Mock stems are copies of the mixture, so every audible bin is 1." : ""}`, 12,
      el("div", { class: "mask-list" }, data.stems.map((stem) => el("figure", { class: "mask" },
        el("figcaption", { class: "stem-name" }, stem.name),
        el("img", { class: "fig-img", src: data.images.masks[stem.name], alt: `Mask of ${stem.name}`, loading: "lazy" })))));

    this.runCard(run, 6);
    this.stftFigure(data.stft, run.tool, 6);
  }
}
