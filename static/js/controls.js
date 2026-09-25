// Sliders and segmented buttons, built from the server's slider specs:
// { name, label, unit, min, max, default, step, decimals, scale }.
import { el } from "./util.js";

const LOG_STEPS = 1000; // slider positions for a log-scale (frequency) slider

// A parameter value as the sliders show it: [number, unit], e.g.
// ["+8.0", "dB"], ["3.40", "kHz"]. kHz: false keeps frequencies in Hz.
export function formatValue(spec, value, { kHz = true } = {}) {
  let shown = Number(value);
  let unit = spec.unit;
  if (kHz && unit === "Hz" && shown >= 1000) {
    shown /= 1000;
    unit = "kHz";
  }
  const digits = unit === "kHz" ? 2 : spec.decimals;
  const sign = spec.min < 0 && value > 0 ? "+" : "";
  return [`${sign}${shown.toFixed(digits)}`, unit];
}

// Typed text -> a number in the spec's unit, or NaN. Takes a sign (+, -, −),
// a decimal comma, the unit, and on frequencies a "k" / "kHz" suffix:
// "3.4k" = 3400 Hz.
export function parseValue(spec, text) {
  let s = text.trim().replace(/\s+/g, "").replace(/−/g, "-").replace(",", ".").toLowerCase();
  let scale = 1;
  const unit = spec.unit.toLowerCase();
  if (unit === "hz" && /k(hz)?$/.test(s)) {
    s = s.replace(/k(hz)?$/, "");
    scale = 1000;
  } else if (unit && s.endsWith(unit)) {
    s = s.slice(0, -unit.length);
  }
  return /^[+-]?(\d+\.?\d*|\.\d+)$/.test(s) ? Number(s) * scale : NaN;
}

export class Slider {
  // onChange(value) fires on user input (dragging, keys, typing, double-click reset).
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
      title: "Double-click to reset",
      oninput: () => {
        this.value = this.fromPosition(Number(this.input.value));
        this.warn(null);
        this.show();
        this.onChange(this.value);
      },
      ondblclick: () => this.reset(),
    });

    // The readout is also a text box: type a value, ENTER or click away to
    // apply it, ESC to cancel. While it has focus it holds the plain number
    // in the spec's unit (Hz, never kHz).
    this.warning = el("p", { class: "slider-warn", id: `${id}-warn`, "aria-live": "polite", hidden: true });
    this.box = el("input", {
      type: "text", class: "readout-box", inputmode: "decimal",
      autocomplete: "off", spellcheck: "false",
      "aria-label": `${spec.label} value`, "aria-describedby": this.warning.id,
      onfocus: () => {
        this.show();
        this.box.select();
        this.keepSelection = true;
      },
      // A click's mouseup would drop the selection made on focus.
      onmouseup: (e) => {
        if (this.keepSelection) e.preventDefault();
        this.keepSelection = false;
      },
      oninput: () => this.check(),
      onkeydown: (e) => {
        if (e.key === "Enter") {
          this.commit();
          this.box.select();
        } else if (e.key === "Escape") {
          e.stopPropagation(); // cancel the edit, don't close the drawer too
          this.warn(null);
          this.show();
          this.box.select();
        }
      },
      onblur: () => this.commit(),
    });
    const chars = Math.max(...[spec.min, spec.max].flatMap((v) =>
      [formatValue(spec, v)[0].length, formatValue(spec, v, { kHz: false })[0].length]));
    this.box.style.width = `${chars}ch`;
    this.unit = el("span", { class: "unit" });
    this.readout = el("span", { class: "readout" }, this.box, this.unit);

    this.fill = el("span", { class: "slider-fill" });
    this.slider = el("div", { class: "slider" },
      el("span", { class: "slider-track" }), this.fill, this.input);
    const symbol = spec.symbol ? el("span", { class: "sym" }, spec.symbol) : null;
    this.root = el("div", { class: "slider-row" },
      el("label", { class: "label", for: id }, spec.label, symbol ? " " : null, symbol),
      this.readout, this.slider, this.warning);

    this.set(spec.default);
  }

  // "0.20 s to 5.00 s", "20 Hz to 20.00 kHz"
  rangeText() {
    const text = (v) => formatValue(this.spec, v).join(" ").trim();
    return `${text(this.spec.min)} to ${text(this.spec.max)}`;
  }

  // invalid: the text in the box can't be used (false: just a note).
  warn(message, { invalid = Boolean(message) } = {}) {
    this.warning.textContent = message || "";
    this.warning.hidden = !message;
    this.box.setAttribute("aria-invalid", String(invalid));
  }

  // Live, while typing: say so as soon as the text can't be used as is.
  check() {
    const text = this.box.value.trim();
    const typed = parseValue(this.spec, text);
    if (!text) this.warn(null);
    else if (Number.isNaN(typed)) this.warn(`Not a number. Allowed: ${this.rangeText()}.`);
    else if (typed < this.spec.min || typed > this.spec.max) this.warn(`Out of range. Allowed: ${this.rangeText()}.`);
    else this.warn(null);
  }

  // Apply the typed text. Out of range: clamp to the limit and keep the
  // warning up. Not a number (or empty): put the current value back.
  commit() {
    const typed = parseValue(this.spec, this.box.value);
    const { min, max } = this.spec;
    if (Number.isNaN(typed)) {
      this.warn(null);
      this.show();
      return;
    }
    if (typed < min || typed > max) {
      const limit = typed < min ? min : max;
      this.warn(`Out of range, set to the ${typed < min ? "minimum" : "maximum"} ${formatValue(this.spec, limit).join(" ").trim()}.`,
        { invalid: false });
    } else {
      this.warn(null);
    }
    const value = this.snap(Math.min(max, Math.max(min, typed)));
    if (value === this.value) {
      this.show();
      return;
    }
    this.set(value);
    this.onChange(this.value);
  }

  // A typed value lands on the slider's steps, so the readout shows what runs.
  snap(value) {
    const { min, step, decimals } = this.spec;
    return Number((min + Math.round((value - min) / step) * step).toFixed(decimals));
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
    this.warn(null);
    this.set(this.spec.default);
    this.onChange(this.value);
  }

  show() {
    const [number, unit] = formatValue(this.spec, this.value);
    const editing = document.activeElement === this.box;
    const [editNumber, editUnit] = formatValue(this.spec, this.value, { kHz: false });
    this.box.value = editing ? editNumber : number;
    this.unit.textContent = editing ? editUnit : unit;
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
