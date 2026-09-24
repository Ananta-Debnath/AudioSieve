// System-response plots, shared by the Studio tools and Backstage so both
// draw exactly the same figure. Data always comes from /response/<tool>.
import { CONFIG } from "./config.js";
import { ThemedChart, fmt, lineOptions, trace, xy } from "./charts.js";
import { el, isVisible } from "./util.js";

// The /response/<tool> query for a tool's parameters. Studio sliders give
// flat EQ fields; a registry run stores the parsed params (a bands
// list), which are flattened back the same way. echoZoom: f_max in Hz,
// or null for the whole spectrum.
export function responseQuery(tool, params, { echoZoom = null } = {}) {
  if (tool === "eq") {
    const query = {};
    for (const [key, value] of Object.entries(params)) {
      if (key !== "bands" && value !== null && value !== undefined) query[key] = value;
    }
    (params.bands || []).forEach((band, i) => {
      query[`band_${i}_gain_db`] = band.gain_db;
    });
    return query;
  }
  if (tool === "reverb") {
    return { rt60: params.rt60, pre_delay_ms: params.pre_delay_ms };
  }
  if (tool === "echo") {
    const { delay_ms, gain, mode } = params;
    return echoZoom ? { delay_ms, gain, mode, f_max: echoZoom } : { delay_ms, gain, mode };
  }
  if (tool === "flanger") {
    const { min_delay_ms: min, sweep_ms: sweep, gain } = params;
    // The first few comb teeth at the longest delay.
    const f_max = (CONFIG.flanger.plot_teeth * 1000) / (Number(min) + Number(sweep));
    return { min_delay_ms: min, sweep_ms: sweep, gain, f_max };
  }
  return {};
}

const BUILDERS = {
  eq(c, { freqs, gain_db: gain, query }) {
    const eqOnly = query.mode === "eq";
    return {
      type: "line",
      data: { datasets: [trace(c, "Gain", xy(freqs, gain))] },
      options: lineOptions(c, {
        x: { type: "logarithmic", min: 20, max: 20000, title: "FREQUENCY (Hz)", format: fmt.hz, unit: "Hz" },
        y: eqOnly
          ? { min: -18, max: 18, step: 6, title: "GAIN (dB)", format: fmt.db, unit: "dB", zeroLine: true }
          : { min: -60, max: 24, step: 12, title: "GAIN (dB)", format: fmt.db, unit: "dB", zeroLine: true },
      }),
    };
  },

  reverb(c, { t, h }) {
    const peak = h.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
    return {
      type: "line",
      data: { datasets: [trace(c, "h(t)", xy(t, h))] },
      options: lineOptions(c, {
        x: { min: 0, max: t[t.length - 1], title: "TIME (s)", format: fmt.s, unit: "s" },
        y: { min: -peak, max: peak, suggested: true, title: "h(t) (LINEAR)", format: fmt.num, zeroLine: true },
      }),
    };
  },

  echo(c, { freqs, mag_db: mag }) {
    return {
      type: "line",
      data: { datasets: [trace(c, "|H|", xy(freqs, mag))] },
      options: lineOptions(c, {
        x: { min: 0, max: freqs[freqs.length - 1], title: "FREQUENCY (Hz)", format: fmt.hz, unit: "Hz" },
        y: { min: -24, max: 24, step: 6, title: "|H| (dB)", format: fmt.db, unit: "dB", zeroLine: true },
      }),
    };
  },

  flanger(c, { freqs, frame }) {
    return {
      type: "line",
      data: { datasets: [trace(c, "|H|", frame)] },
      options: lineOptions(c, {
        x: { min: 0, max: freqs[freqs.length - 1], title: "FREQUENCY (Hz)", format: fmt.hz, unit: "Hz" },
        y: { min: -30, max: 10, step: 10, title: "|H| (dB)", format: fmt.db, unit: "dB", zeroLine: true },
      }),
    };
  },
};

// One system-response chart. For the flanger it also animates: one
// curve per delay of the sweep, played back at the LFO rate (rate() Hz),
// with a D readout and a play/pause button appended to `tools`.
export class ResponsePlot {
  constructor(canvas, tool, { tools = null, rate = () => 1 } = {}) {
    this.canvas = canvas;
    this.tool = tool;
    this.chart = new ThemedChart(canvas, (c, data) => BUILDERS[tool](c, data));

    if (tool === "flanger") {
      this.rate = rate;
      this.phase = 0;
      this.frameIndex = 0;
      this.lastTick = null;
      this.readout = el("span", { class: "readout-inline" });
      this.animButton = el("button", {
        type: "button", class: "btn btn-small", onclick: () => this.setAnimating(!this.animating),
      });
      if (tools) tools.append(this.readout, this.animButton);
      this.setAnimating(!matchMedia("(prefers-reduced-motion: reduce)").matches);
    }
  }

  set(data, query) {
    if (this.tool !== "flanger") {
      this.chart.set({ ...data, query });
      return;
    }
    this.frames = data.mag_db.map((row) => xy(data.freqs, row));
    this.delays = data.delays_ms;
    this.frameIndex = Math.min(this.frameIndex, this.frames.length - 1);
    this.chart.set({ freqs: data.freqs, frame: this.frames[this.frameIndex] });
    this.showDelay();
    this.startLoop();
  }

  setAnimating(on) {
    this.animating = on;
    this.animButton.textContent = on ? "❚❚ Pause sweep" : "▶ Play sweep";
    this.animButton.setAttribute("aria-pressed", String(on));
    this.lastTick = null;
  }

  showDelay() {
    if (!this.delays) return;
    this.readout.replaceChildren("D = ", el("b", {}, this.delays[this.frameIndex].toFixed(2)), " ms");
  }

  // One requestAnimationFrame loop per plot; it only redraws while the
  // plot is visible and the sweep is playing.
  startLoop() {
    if (this.loopStarted) return;
    this.loopStarted = true;
    const tick = (now) => {
      if (this.destroyed) return;
      requestAnimationFrame(tick);
      const last = this.lastTick;
      this.lastTick = now;
      if (!this.animating || !this.frames || last === null || !isVisible(this.canvas)) return;
      this.phase = (this.phase + ((now - last) / 1000) * Number(this.rate())) % 1;
      const index = Math.floor(this.phase * this.frames.length);
      const chart = this.chart.chart;
      if (index === this.frameIndex || !chart) return;
      this.frameIndex = index;
      this.chart.data = { ...this.chart.data, frame: this.frames[index] }; // for theme redraws
      chart.data.datasets[0].data = this.frames[index];
      chart.update("none");
      this.showDelay();
    };
    requestAnimationFrame(tick);
  }

  resize() {
    this.chart.resize();
  }

  destroy() {
    this.destroyed = true;
    this.chart.destroy();
  }
}
