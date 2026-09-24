// The Studio: one view per tool. Each keeps its own file, parameters and
// last result while other tabs are open. The page never processes
// audio: it sends parameters to the server and plays / plots what comes back.
import { CONFIG, toolInfo } from "./config.js";
import { ThemedChart, fmt, lineOptions, trace, xy } from "./charts.js";
import { Segmented, Slider, field } from "./controls.js";
import { Player, StemGroup } from "./player.js";
import { SourceBox } from "./source.js";
import { el, fmtClock, isVisible, requestJSON } from "./util.js";

const group = (label, ...content) => el("div", { class: "group" }, el("p", { class: "label" }, label), ...content);

class Tool {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.id = root.dataset.tool;
    this.info = toolInfo(this.id);
    this.values = {};
    this.running = false;
    this.responseSeq = 0;
    this.players = [];

    this.source = new SourceBox(root.querySelector("[data-source]"), () => this.sourceChanged());
    this.runButton = root.querySelector("[data-run]");
    this.runHint = root.querySelector("[data-run-hint]");
    this.runError = root.querySelector("[data-run-error]");
    this.output = root.querySelector("[data-output]");
    this.runButton.addEventListener("click", () => this.run());
    root.querySelector("[data-how]").addEventListener("click", () => app.drawer.toggle(this.id));

    const canvas = root.querySelector("[data-response]");
    if (canvas) {
      this.chart = new ThemedChart(canvas, (c, data) => this.responseChart(c, data));
      this.responseMsg = root.querySelector("[data-response-msg]");
      this.plotTools = root.querySelector("[data-plot-tools]");
    }

    this.buildControls(root.querySelector("[data-params]"));
    this.showEmptyOutput();
    this.updateRunButton();
    if (this.chart) this.refreshResponse(0);
  }

  // ---- hooks for each tool ----------------------------------------
  buildControls(_container) {}

  params() {
    return { ...this.values };
  }

  responseQuery() {
    return this.params();
  }

  responseChart(_colors, _data) {
    return null;
  }

  // ---- parameters and the live response plot ----------------------
  slider(spec) {
    const slider = new Slider(spec, (value) => {
      this.values[spec.name] = value;
      this.changed(spec.name);
    });
    this.values[spec.name] = slider.value;
    return slider;
  }

  // The user changed a parameter (name: the slider's, if it was one).
  changed(_name) {
    this.refreshResponse();
  }

  refreshResponse(delay = CONFIG.responseDebounceMs) {
    if (!this.chart) return;
    clearTimeout(this.responseTimer);
    this.responseTimer = setTimeout(() => this.fetchResponse(), delay);
  }

  async fetchResponse() {
    // Only the newest request may draw: responses can arrive out of order.
    const seq = ++this.responseSeq;
    const query = this.responseQuery();
    try {
      const data = await requestJSON(`/response/${this.id}?${new URLSearchParams(query)}`);
      if (seq !== this.responseSeq) return;
      this.responseMsg.hidden = true;
      this.drawResponse(data, query);
    } catch (err) {
      if (seq !== this.responseSeq) return;
      this.responseMsg.textContent = err.message;
      this.responseMsg.hidden = false;
    }
  }

  drawResponse(data, query) {
    this.chart.set({ ...data, query });
  }

  // Called when this tab becomes visible.
  shown() {
    if (this.chart) this.chart.resize();
  }

  // ---- source and RUN ---------------------------------------------
  sourceChanged() {
    this.clearOutput();
    this.showEmptyOutput();
    this.runError.hidden = true;
    this.updateRunButton();
  }

  updateRunButton() {
    const ready = Boolean(this.source.meta);
    this.runButton.disabled = !ready || this.running;
    this.runHint.hidden = ready || this.running;
  }

  async run() {
    if (this.running || !this.source.meta) return;
    const fileId = this.source.meta.file_id;
    this.running = true;
    this.runError.hidden = true;
    this.updateRunButton();
    this.runButton.setAttribute("aria-busy", "true");

    const started = performance.now();
    const elapsed = () => (performance.now() - started) / 1000;
    const tick = () => this.runButton.replaceChildren(
      "Processing… ", el("span", { style: "text-transform: none" }, `${Math.floor(elapsed())}s`));
    tick();
    const timer = setInterval(tick, 250);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), CONFIG.runTimeoutMs);

    try {
      const data = await requestJSON(`/process/${this.id}`, {
        method: "POST",
        json: { ...this.params(), file_id: fileId },
        signal: controller.signal,
      });
      this.app.recordRun(data.run_id);
      // The user may have loaded another file meanwhile; this result isn't for it.
      if (this.source.meta && this.source.meta.file_id === fileId) this.showResult(data, elapsed());
    } catch (err) {
      this.showRunError(err.name === "AbortError"
        ? new Error("the server took too long. Try a shorter track.")
        : err);
    } finally {
      clearInterval(timer);
      clearTimeout(timeout);
      this.running = false;
      this.runButton.removeAttribute("aria-busy");
      this.runButton.textContent = "Run";
      this.updateRunButton();
    }
  }

  showRunError(err) {
    this.runError.textContent = `Run failed: ${err.message}`;
    this.runError.hidden = false;
  }

  // ---- output -------------------------------------------------------
  clearOutput() {
    this.players.forEach((p) => p.destroy());
    this.players = [];
  }

  showEmptyOutput() {
    this.output.replaceChildren(el("p", { class: "empty" }, "No output yet. Load a track and press RUN."));
  }

  outputActions(data, seconds, ...extra) {
    return el("div", { class: "output-actions" },
      ...extra,
      el("button", {
        type: "button", class: "btn", onclick: () => this.app.openBackstage(data.run_id),
      }, "→ Backstage"),
      el("span", { class: "run-stamp" }, `Run ${fmtClock(Date.now() / 1000)} · took ${seconds.toFixed(1)} s`));
  }

  showResult(data, seconds) {
    this.clearOutput();
    const wrap = el("div", { class: "output" });
    this.output.replaceChildren(wrap);
    const waves = data.waveforms || {};
    this.players = [
      new Player(wrap, { label: "Original", variant: "original", source: { url: data.input_url, ...waves.before } }),
      new Player(wrap, { label: "Processed", variant: "processed", source: { url: data.url, ...waves.after } }),
    ];
    wrap.append(this.outputActions(data, seconds,
      el("a", { class: "btn btn-accent", href: data.download_url, download: "" }, "Download WAV")));
  }
}

// ---------------------------------------------------------------------
// 01 EQ + Filter
// ---------------------------------------------------------------------

class EqTool extends Tool {
  buildControls(container) {
    const cfg = CONFIG.eq;
    this.values.mode = cfg.default_mode;
    this.values.filter_type = cfg.default_filter_type;
    this.sliders = {};
    const make = (spec) => {
      this.sliders[spec.name] = this.slider(spec);
      return this.sliders[spec.name].root;
    };

    this.modeSeg = new Segmented(cfg.modes, this.values.mode, (mode) => {
      this.values.mode = mode;
      this.layout();
      this.changed();
    }, { label: "Mode" });
    this.typeSeg = new Segmented(cfg.filter_types, this.values.filter_type, (type) => {
      this.values.filter_type = type;
      this.layout();
      this.changed();
    }, { label: "Filter type" });

    this.bandGroup = group("EQ bands (dB)", cfg.bands.map(make));
    this.filterGroup = group("Butterworth filter", this.typeSeg.root, cfg.filter.map(make));
    this.humGroup = group("Hum notches", cfg.hum.map(make));
    this.presetButtons = cfg.presets.map((preset) => el("button", {
      type: "button", class: "btn btn-small", "aria-pressed": "false",
      "data-preset": preset.name, onclick: () => this.applyPreset(preset),
    }, preset.label));

    container.append(
      el("div", { class: "group" }, field("Mode", this.modeSeg.root)),
      this.bandGroup, this.filterGroup, this.humGroup,
      el("div", { class: "group" }, field("Presets", el("div", { class: "presets" }, this.presetButtons))));
    this.layout();
  }

  // Show only the controls the current mode / filter type uses.
  layout() {
    const { mode, filter_type: type } = this.values;
    this.bandGroup.hidden = !(mode === "eq" || mode === "both");
    this.filterGroup.hidden = !(mode === "filter" || mode === "both");
    this.humGroup.hidden = mode !== "hum";
    const twoCutoffs = type === "bandpass" || type === "bandstop";
    this.sliders.cutoff.root.hidden = twoCutoffs;
    this.sliders.low_cutoff.root.hidden = !twoCutoffs;
    this.sliders.high_cutoff.root.hidden = !twoCutoffs;
  }

  changed(name) {
    if (name === "low_cutoff" || name === "high_cutoff") this.keepCutoffsApart(name);
    this.highlightPreset(null); // a manual change: it's not the preset any more
    super.changed(name);
  }

  // Band-pass / band-stop need low < high: dragging one cutoff past the
  // other pushes the other one along instead of making an invalid band.
  keepCutoffsApart(moved) {
    const GAP = 1.12; // ~ a sixth of an octave
    const low = this.sliders.low_cutoff;
    const high = this.sliders.high_cutoff;
    if (low.value * GAP <= high.value) return;
    if (moved === "low_cutoff") {
      high.set(low.value * GAP);
      if (high.value < low.value * GAP) low.set(high.value / GAP); // high hit its maximum
    } else {
      low.set(high.value / GAP);
      if (low.value > high.value / GAP) high.set(low.value * GAP); // low hit its minimum
    }
    this.values.low_cutoff = low.value;
    this.values.high_cutoff = high.value;
  }

  applyPreset(preset) {
    for (const [name, value] of Object.entries(preset.values)) {
      if (name === "mode" || name === "filter_type") {
        this.values[name] = value;
        (name === "mode" ? this.modeSeg : this.typeSeg).set(value);
      } else if (this.sliders[name]) {
        this.sliders[name].set(value);
        this.values[name] = this.sliders[name].value;
      }
    }
    this.layout();
    this.highlightPreset(preset.name);
    this.refreshResponse(0);
  }

  highlightPreset(name) {
    this.presetButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.preset === name)));
  }

  responseChart(c, { freqs, gain_db: gain, query }) {
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
  }
}

// ---------------------------------------------------------------------
// 02 Reverb
// ---------------------------------------------------------------------

class ReverbTool extends Tool {
  buildControls(container) {
    const cfg = CONFIG.reverb;
    this.values.circular = "false";
    this.circularNote = el("p", { class: "note note-warn", hidden: true },
      "Wrong on purpose: no zero-padding, so the tail wraps onto the start. Listen to the first second.");
    const circular = new Segmented(cfg.circular, this.values.circular, (value) => {
      this.values.circular = value;
      this.circularNote.hidden = value !== "true";
      this.changed();
    }, { label: "Convolution", warnValue: "true" });

    container.append(
      group("Room", cfg.sliders.map((spec) => this.slider(spec).root)),
      el("div", { class: "group" }, field("Convolution", circular.root), this.circularNote));
  }

  responseChart(c, { t, h }) {
    const peak = h.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
    return {
      type: "line",
      data: { datasets: [trace(c, "h(t)", xy(t, h))] },
      options: lineOptions(c, {
        x: { min: 0, max: t[t.length - 1], title: "TIME (s)", format: fmt.s, unit: "s" },
        y: { min: -peak, max: peak, suggested: true, title: "h(t) (LINEAR)", format: fmt.num, zeroLine: true },
      }),
    };
  }
}

// ---------------------------------------------------------------------
// 03 Echo + Delay
// ---------------------------------------------------------------------

class EchoTool extends Tool {
  buildControls(container) {
    const cfg = CONFIG.echo;
    this.values.mode = cfg.default_mode;
    this.zoom = cfg.zooms[0].f_max;
    const mode = new Segmented(cfg.modes, this.values.mode, (value) => {
      this.values.mode = value;
      this.changed();
    }, { label: "Echo mode" });
    container.append(
      el("div", { class: "group" }, field("Mode", mode.root)),
      group("Echo", cfg.sliders.map((spec) => this.slider(spec).root)));

    const zoom = new Segmented(
      cfg.zooms.map((z) => ({ value: String(z.f_max), label: z.label })),
      String(this.zoom),
      (value) => {
        this.zoom = value === "null" ? null : Number(value);
        this.refreshResponse(0);
      },
      { label: "Frequency zoom", compact: true });
    this.plotTools.append(el("span", { class: "label" }, "Zoom"), zoom.root);
  }

  responseQuery() {
    const { delay_ms, gain, mode } = this.values;
    return this.zoom ? { delay_ms, gain, mode, f_max: this.zoom } : { delay_ms, gain, mode };
  }

  responseChart(c, { freqs, mag_db: mag }) {
    return {
      type: "line",
      data: { datasets: [trace(c, "|H|", xy(freqs, mag))] },
      options: lineOptions(c, {
        x: { min: 0, max: freqs[freqs.length - 1], title: "FREQUENCY (Hz)", format: fmt.hz, unit: "Hz" },
        y: { min: -24, max: 24, step: 6, title: "|H| (dB)", format: fmt.db, unit: "dB", zeroLine: true },
      }),
    };
  }
}

// ---------------------------------------------------------------------
// 04 Flanger: the comb's response at each delay of one sweep, played
// back at the LFO rate.
// ---------------------------------------------------------------------

class FlangerTool extends Tool {
  buildControls(container) {
    container.append(group("Flanger", CONFIG.flanger.sliders.map((spec) => this.slider(spec).root)));

    this.animating = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.phase = 0;
    this.frameIndex = 0;
    this.delayReadout = el("span", { class: "readout-inline" });
    this.animButton = el("button", {
      type: "button", class: "btn btn-small", onclick: () => this.setAnimating(!this.animating),
    });
    this.plotTools.append(this.delayReadout, this.animButton);
    this.setAnimating(this.animating);
  }

  responseQuery() {
    const { min_delay_ms: min, sweep_ms: sweep, gain } = this.values;
    // The first few comb teeth at the longest delay.
    const f_max = (CONFIG.flanger.plot_teeth * 1000) / (min + sweep);
    return { min_delay_ms: min, sweep_ms: sweep, gain, f_max };
  }

  drawResponse(data, query) {
    this.frames = data.mag_db.map((row) => xy(data.freqs, row));
    this.delays = data.delays_ms;
    this.frameIndex = Math.min(this.frameIndex, this.frames.length - 1);
    super.drawResponse(data, query);
    this.showDelay();
    this.startLoop();
  }

  responseChart(c, { freqs }) {
    return {
      type: "line",
      data: { datasets: [trace(c, "|H|", this.frames[this.frameIndex])] },
      options: lineOptions(c, {
        x: { min: 0, max: freqs[freqs.length - 1], title: "FREQUENCY (Hz)", format: fmt.hz, unit: "Hz" },
        y: { min: -30, max: 10, step: 10, title: "|H| (dB)", format: fmt.db, unit: "dB", zeroLine: true },
      }),
    };
  }

  setAnimating(on) {
    this.animating = on;
    this.animButton.textContent = on ? "❚❚ Pause sweep" : "▶ Play sweep";
    this.animButton.setAttribute("aria-pressed", String(on));
    this.lastTick = null;
  }

  showDelay() {
    if (!this.delays) return;
    this.delayReadout.replaceChildren("D = ", el("b", {}, this.delays[this.frameIndex].toFixed(2)), " ms");
  }

  // One requestAnimationFrame loop for the page's lifetime; it only
  // redraws while the tab is visible and the sweep is playing.
  startLoop() {
    if (this.loopStarted) return;
    this.loopStarted = true;
    const tick = (now) => {
      requestAnimationFrame(tick);
      const last = this.lastTick;
      this.lastTick = now;
      if (!this.animating || !this.frames || last === null || !isVisible(this.root)) return;
      this.phase = (this.phase + ((now - last) / 1000) * this.values.rate_hz) % 1;
      const index = Math.floor(this.phase * this.frames.length);
      if (index === this.frameIndex || !this.chart.chart) return;
      this.frameIndex = index;
      this.chart.chart.data.datasets[0].data = this.frames[index];
      this.chart.chart.update("none");
      this.showDelay();
    };
    this.lastTick = null;
    requestAnimationFrame(tick);
  }
}

// ---------------------------------------------------------------------
// 05 Separation (built against the /process/separate contract)
// ---------------------------------------------------------------------

class SeparationTool extends Tool {
  buildControls(container) {
    container.append(el("p", { class: "note" },
      "No parameters: the separation module decides the stems. Press RUN, then mute, solo or play them all together."));
  }

  params() {
    return {};
  }

  showRunError(err) {
    if (err.status === 501) {
      this.clearOutput();
      this.output.replaceChildren(el("div", { class: "msg msg-warn" }, "Separation module offline, pending merge."));
      return;
    }
    super.showRunError(err);
  }

  async showResult(data, seconds) {
    this.clearOutput();
    const wrap = el("div", { class: "output" });
    this.output.replaceChildren(wrap);

    const originalSlot = el("div");
    wrap.append(originalSlot);
    this.addOriginal(originalSlot, this.source.meta);

    const playAll = el("button", { type: "button", class: "btn btn-accent" }, "▶ Play all");
    wrap.append(el("div", { class: "stems-head" },
      playAll, el("span", { class: "note" }, `${data.stems.length} stems · mute and solo only change playback volume`)));

    const stems = data.stems.map((stem) => {
      const body = el("div", { class: "stem-body" });
      const row = el("div", { class: "stem" }, el("div", { class: "stem-name" }, stem.name), body);
      wrap.append(row);
      const player = new Player(body, { label: stem.name, variant: "stem", source: stem, height: 44, showLabel: false });
      const entry = { player, row, muted: false, soloed: false };
      entry.mute = el("button", {
        type: "button", class: "btn btn-small", "aria-pressed": "false",
        onclick: () => { entry.muted = !entry.muted; applyVolumes(); },
      }, "Mute");
      entry.solo = el("button", {
        type: "button", class: "btn btn-small", "aria-pressed": "false",
        onclick: () => { entry.soloed = !entry.soloed; applyVolumes(); },
      }, "Solo");
      player.root.querySelector(".player-head").append(el("div", { class: "stem-controls" },
        entry.mute, entry.solo,
        el("a", { class: "btn btn-small", href: stem.download_url, download: "" }, "Download")));
      this.players.push(player);
      return entry;
    });

    // Solo wins over mute; volume only, the audio itself is untouched.
    const applyVolumes = () => {
      const anySolo = stems.some((s) => s.soloed);
      for (const s of stems) {
        const audible = anySolo ? s.soloed : !s.muted;
        s.player.setVolume(audible ? 1 : 0);
        s.row.classList.toggle("is-silent", !audible);
        s.mute.setAttribute("aria-pressed", String(s.muted));
        s.solo.setAttribute("aria-pressed", String(s.soloed));
      }
    };

    const stemGroup = new StemGroup(stems.map((s) => s.player), () => {
      playAll.textContent = stemGroup.isPlaying() ? "❚❚ Pause all" : "▶ Play all";
    });
    playAll.addEventListener("click", () => stemGroup.toggle());

    wrap.append(this.outputActions(data, seconds));
  }

  // The contract has no waveform for the original, so ask for it; if that
  // fails, WaveSurfer decodes the file itself.
  async addOriginal(slot, meta) {
    let wave = {};
    try {
      wave = await requestJSON(`${meta.url}/waveform`);
    } catch {
      /* fall back to decoding */
    }
    if (!slot.isConnected) return; // the output was replaced meanwhile
    this.players.push(new Player(slot, {
      label: "Original", variant: "original", source: { url: meta.url, ...wave },
    }));
  }
}

const TOOL_CLASSES = { eq: EqTool, reverb: ReverbTool, echo: EchoTool, flanger: FlangerTool, separate: SeparationTool };

export function createTools(app) {
  const tools = new Map();
  document.querySelectorAll("section[data-tool]").forEach((root) => {
    const ToolClass = TOOL_CLASSES[root.dataset.tool] || Tool;
    tools.set(root.dataset.tool, new ToolClass(root, app));
  });
  return tools;
}
