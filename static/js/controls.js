// Sliders and segmented buttons, built from the server's slider specs:
// { name, label, unit, min, max, default, step, decimals, scale }.
import { el } from "./util.js";

const LOG_STEPS = 1000; // slider positions for a log-scale (frequency) slider

// A parameter value as the sliders show it: [number, unit], e.g.
// ["+8.0", "dB"], ["3.40", "kHz"].
export function formatValue(spec, value) {
  let shown = Number(value);
  let unit = spec.unit;
  if (unit === "Hz" && shown >= 1000) {
    shown /= 1000;
    unit = "kHz";
  }
  const digits = unit === "kHz" ? 2 : spec.decimals;
  const sign = spec.min < 0 && value > 0 ? "+" : "";
  return [`${sign}${shown.toFixed(digits)}`, unit];
}

export class Slider {
  // onChange(value) fires on user input (dragging, keys, double-click reset).
  constructor(spec, onChange) {
    this.spec = spec;
    this.onChange = onChange;
    this.log = spec.scale === "log";

    const id = `slider-${spec.name}-${Math.random().toString(36).slice(2, 8)}`;
    this.input = el("input", {
      type: "range", id,
      min: this.log ? 0 : spec.min,
      max: this.log ? LOG_STEPS : spec.max,
      step: this.log ? 1 : spec.step,
      "aria-label": spec.label,
      oninput: () => {
        this.value = this.fromPosition(Number(this.input.value));
        this.show();
        this.onChange(this.value);
      },
    });
    this.readout = el("output", {
      class: "readout", for: id, title: "Double-click to reset",
      ondblclick: () => this.reset(),
    });
    this.fill = el("span", { class: "slider-fill" });
    this.slider = el("div", { class: "slider" },
      el("span", { class: "slider-track" }), this.fill, this.input);
    const symbol = spec.symbol ? el("span", { class: "sym" }, spec.symbol) : null;
    this.root = el("div", { class: "slider-row" },
      el("label", { class: "label", for: id }, spec.label, symbol ? " " : null, symbol),
      this.readout, this.slider);

    this.set(spec.default);
  }

  fromPosition(pos) {
    if (!this.log) return Number(pos);
    const { min, max, step } = this.spec;
    const hz = min * (max / min) ** (pos / LOG_STEPS);
    return Math.min(max, Math.max(min, Math.round(hz / step) * step));
  }

  toPosition(value) {
    if (!this.log) return value;
    const { min, max } = this.spec;
    return Math.round((LOG_STEPS * Math.log(value / min)) / Math.log(max / min));
  }

  set(value) {
    const { min, max } = this.spec;
    this.value = Math.min(max, Math.max(min, Number(value)));
    this.input.value = this.toPosition(this.value);
    this.show();
  }

  reset() {
    this.set(this.spec.default);
    this.onChange(this.value);
  }

  show() {
    const [number, unit] = formatValue(this.spec, this.value);
    this.readout.replaceChildren(number, unit ? el("span", { class: "unit" }, unit) : "");
    this.input.setAttribute("aria-valuetext", `${number} ${unit}`.trim());
    const lo = Number(this.input.min);
    const hi = Number(this.input.max);
    this.slider.style.setProperty("--pct", (Number(this.input.value) - lo) / (hi - lo || 1));
  }
}

// A row of block buttons, one pressed. options: [{ value, label }].
export class Segmented {
  constructor(options, value, onChange, { label, warnValue, compact = false } = {}) {
    this.onChange = onChange;
    this.buttons = options.map((option) =>
      el("button", {
        type: "button",
        class: option.value === warnValue ? "is-warn" : null,
        "data-value": option.value,
        onclick: () => {
          if (option.value === this.value) return;
          this.set(option.value);
          this.onChange(option.value);
        },
      }, option.label));
    this.root = el("div", {
      class: compact ? "seg seg-compact" : "seg", role: "group", "aria-label": label,
    }, this.buttons);
    this.set(value);
  }

  set(value) {
    this.value = value;
    this.buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === String(value))));
  }
}

// A labelled block: <div class="field"><p class="label">..</p>content</div>
export function field(label, ...content) {
  return el("div", { class: "field" }, el("p", { class: "label" }, label), ...content);
}
