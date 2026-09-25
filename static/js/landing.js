// SPECTRA landing page: the text below, the sound toggle and the tools
// reel. No DSP and no backend calls: every clip, its timing and each card
// background is a static file made by scripts/make_landing_assets.py,
// listed by the server in the #landing-data JSON.
import { initThemeToggle } from "./theme.js";
import { el, storageGet, storageSet } from "./util.js";

// All the page's editable text.
const CONFIG = {
  appName: "SPECTRA",
  tagline: "Interactive audio signal processing.",
  teamName: "TEAM NAME",                          // placeholder, fill in
  githubUrl: "https://github.com/OWNER/REPO",     // placeholder, fill in
  // A slide with a preview clip lasts as long as the clip (the asset
  // script makes them about 6 s), so the audio fills it; this is for the rest.
  slideMs: 6000,
  // One reel card per tool; `tool` is also its /lab#<tool> deep link.
  slides: [
    {
      tool: "eq", number: "01", name: "EQ + Filter",
      // ‑: a hyphen the line never breaks at
      text: "Reshape the spectrum: Butterworth filters remove bands, a 5‑band EQ rebalances them.",
    },
    {
      tool: "reverb", number: "02", name: "Reverb",
      text: "Place the sound in a room by convolving it with an impulse response, via the FFT.",
    },
    {
      tool: "echo", number: "03", name: "Echo + Delay",
      text: "Repeats built from difference equations: one echo (FIR) or decaying echoes (IIR).",
    },
    {
      tool: "flanger", number: "04", name: "Flanger",
      text: "A comb filter whose notches sweep, driven by a slowly varying delay.",
    },
    {
      tool: "separation", number: "05", name: "Separation",
      text: "Split a mix into stems with non-negative matrix factorization and spectral masks.",
    },
  ],
};

const HOLD_MS = 200;       // a press this long is a hold; shorter is a tap
const TAP_SLOP_PX = 10;    // a press that travels further is a drag, not a tap
const PREV_ZONE = 0.3;     // taps on the card's left 30% go back
const MIN_VISIBLE = 0.4;   // with less of the reel on screen, it pauses
const MAX_FRAME_MS = 100;  // a longer gap between frames (a stalled tab) counts as this
const CLIP_BEAT_MS = 300;  // after a slide's clip ends, before the next slide
const SOUND_KEY = "spectra-sound";

const page = JSON.parse(document.getElementById("landing-data").textContent);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// ---------------------------------------------------------------------
// Tools reel: stories-style slides that advance every CONFIG.slideMs.
// Hold the card (or press Space) to pause it and loop the slide's clip;
// tap its left 30% to go back, anywhere else to go on.
// ---------------------------------------------------------------------

class Reel {
  constructor(root, slides, previews, sound) {
    this.root = root;
    this.slides = slides.map((slide) => ({ ...slide, preview: previews[slide.tool] || null, audio: null }));
    this.index = 0;
    this.elapsed = 0;         // ms the current slide has run
    this.holds = new Set();   // why it is held: "pointer" (press and hold), "key" (Space)
    this.onScreen = false;
    this.pageVisible = document.visibilityState === "visible";
    this.sound = sound;
    // The current slide's clip as a timeline: its position (s) and whether
    // it runs. With sound on, the audio element is its clock; with sound
    // off it runs silently on the frame clock, so the ORIGINAL / PROCESSED
    // label switches on the same timing either way.
    this.clipTime = 0;
    this.clipRunning = false;
    this.labelB = null;
    this.lastFrame = null;
    this.waitingForGesture = false;

    const part = (name) => root.querySelector(`[data-${name}]`);
    this.card = part("card");
    this.bg = part("card-bg");
    this.num = part("card-num");
    this.name = part("card-name");
    this.text = part("card-text");
    this.ab = part("ab");
    this.link = part("card-link");
    this.live = part("reel-live");
    this.segments = this.slides.map((slide, i) => el("button", {
      type: "button", class: "reel-seg", tabindex: "-1",
      "aria-label": `${slide.number} ${slide.name}`,
      onclick: () => this.show(i, { manual: true }),
    }, el("span", { class: "reel-seg-bar" }, el("span", { class: "reel-seg-fill" }))));
    part("reel-progress").append(...this.segments);
    this.fills = this.segments.map((seg) => seg.querySelector(".reel-seg-fill"));

    // Load every background up front: a slide never waits for its image.
    for (const { preview } of this.slides) {
      if (preview && preview.background_url) new Image().src = preview.background_url;
    }

    this.bindPointer();
    this.bindKeys();
    this.watchVisibility();
    reducedMotion.addEventListener("change", () => this.renderProgress());
    this.show(0);
    requestAnimationFrame((now) => this.frame(now));
  }

  get current() {
    return this.slides[this.index];
  }

  get active() {
    return this.onScreen && this.pageVisible;
  }

  get held() {
    return this.holds.size > 0;
  }

  // ms the current slide runs: its clip and a short beat, so there is no
  // silence before the next one; CONFIG.slideMs without a clip.
  get duration() {
    const { preview } = this.current;
    return preview ? preview.duration_s * 1000 + CLIP_BEAT_MS : CONFIG.slideMs;
  }

  get audioPlaying() {
    const { audio } = this.current;
    return Boolean(this.sound && audio && !audio.paused && !audio.ended);
  }

  // manual: the visitor chose this slide (tap, arrow, progress segment).
  show(index, { manual = false } = {}) {
    const count = this.slides.length;
    if (this.current.audio) this.current.audio.pause();
    this.index = (index + count) % count;
    this.elapsed = 0;
    if (manual) this.holds.delete("key"); // stepping ends a Space pause
    this.clipTime = 0;
    this.clipRunning = Boolean(this.current.preview);
    this.renderCard();
    this.segments.forEach((seg, i) => {
      if (i === this.index) seg.setAttribute("aria-current", "true");
      else seg.removeAttribute("aria-current");
    });
    // Announced only when the visitor changes slide: announcing every
    // automatic advance would talk over everything else.
    if (manual) this.live.textContent = `${this.current.number} ${this.current.name}`;
    this.preload();
    this.syncAudio();
    this.renderProgress();
  }

  // Press and hold / Space: pause, and loop the clip (restarting it if
  // it already finished).
  hold(reason) {
    this.holds.add(reason);
    if (this.current.preview && !this.clipRunning) {
      this.clipTime = 0;
      this.clipRunning = true;
    }
    this.syncAudio();
  }

  // Letting go stops the clip; the slide's progress carries on.
  release(reason) {
    if (!this.holds.delete(reason) || this.held) return;
    this.clipRunning = false;
    this.syncAudio();
  }

  setSound(on) {
    this.sound = on;
    if (on) {
      // Start the slide over, so the whole clip is heard.
      this.elapsed = 0;
      this.clipTime = 0;
      this.clipRunning = Boolean(this.current.preview);
      this.preload();
    }
    this.syncAudio();
  }

  // ---- audio ------------------------------------------------------
  audioFor(slide) {
    if (!slide.audio) {
      slide.audio = new Audio(slide.preview.clip_url);
      slide.audio.preload = "auto";
    }
    return slide.audio;
  }

  // With sound on, keep the current and the next slide's clips loaded.
  preload() {
    if (!this.sound) return;
    for (const slide of [this.current, this.slides[(this.index + 1) % this.slides.length]]) {
      if (slide.preview) this.audioFor(slide);
    }
  }

  // Make the current clip's audio match the timeline: it plays only
  // while sound is on, the clip runs and the reel can be seen.
  syncAudio() {
    const slide = this.current;
    if (!(this.sound && this.clipRunning && this.active)) {
      if (slide.audio) slide.audio.pause();
      return;
    }
    const audio = this.audioFor(slide);
    audio.loop = this.held;
    if (Math.abs(audio.currentTime - this.clipTime) > 0.05) audio.currentTime = this.clipTime;
    if (!audio.paused) return;
    audio.play().catch((err) => {
      // Browsers block audio until the visitor interacts with the page
      // (sound remembered as ON, on a fresh load). The timeline carries
      // on silently until then.
      if (err.name === "NotAllowedError") this.waitForGesture();
    });
  }

  waitForGesture() {
    if (this.waitingForGesture) return;
    this.waitingForGesture = true;
    const retry = () => {
      this.waitingForGesture = false;
      document.removeEventListener("pointerup", retry, true);
      document.removeEventListener("keydown", retry, true);
      this.syncAudio();
    };
    document.addEventListener("pointerup", retry, true);
    document.addEventListener("keydown", retry, true);
  }

  // ---- animation frame --------------------------------------------
  frame(now) {
    const dt = this.lastFrame === null ? 0 : Math.min(now - this.lastFrame, MAX_FRAME_MS);
    this.lastFrame = now;
    // Reduced motion: no auto-advance; slides change only when asked.
    if (this.active && !this.held && !reducedMotion.matches) {
      this.elapsed = Math.min(this.elapsed + dt, this.duration);
      // Never cut a clip off: if its audio started late, wait for the end.
      if (this.elapsed >= this.duration && !this.audioPlaying) this.show(this.index + 1);
    }
    if (this.active && this.clipRunning) this.tickClip(dt);
    this.renderProgress();
    requestAnimationFrame((t) => this.frame(t));
  }

  tickClip(dt) {
    const { preview, audio } = this.current;
    if (audio && !audio.paused) {
      this.clipTime = audio.currentTime; // sound on: the audio is the clock
    } else {
      this.clipTime += dt / 1000; // sound off: the same timeline, silently
      if (this.clipTime >= preview.duration_s) {
        if (this.held) this.clipTime %= preview.duration_s;
        else this.clipRunning = false;
      }
    }
    this.renderLabel();
  }

  // ---- rendering ----------------------------------------------------
  renderCard() {
    const { number, name, text, tool, preview } = this.current;
    this.num.textContent = number;
    this.name.textContent = name;
    this.text.textContent = text;
    this.link.href = `${page.lab_url}#${tool}`;
    this.link.setAttribute("aria-label", `Open tool: ${name}`);
    this.card.setAttribute("aria-label", `${this.index + 1} of ${this.slides.length}`);

    const bg = preview && preview.background_url;
    this.bg.hidden = !bg;
    if (bg) {
      const line = bg.endsWith(".svg");
      this.bg.style.setProperty("--bg-url", `url("${bg}")`);
      this.bg.classList.toggle("is-line", line);
      this.bg.classList.toggle("is-image", !line);
    }

    this.ab.classList.toggle("is-soon", !preview);
    this.labelB = null;
    if (preview) {
      this.renderLabel();
    } else {
      this.ab.classList.remove("is-b");
      this.ab.textContent = "Preview coming soon";
    }

    // Replay the short fade-in (style.css turns it off for reduced motion).
    this.card.classList.remove("is-entering");
    void this.card.offsetWidth;
    this.card.classList.add("is-entering");
  }

  renderLabel() {
    const { preview } = this.current;
    const b = this.clipTime >= preview.switch_at_s;
    if (b === this.labelB) return;
    this.labelB = b;
    this.ab.textContent = b ? preview.label_b : "Original";
    this.ab.classList.toggle("is-b", b);
  }

  // Done slides full, the current one filling, the rest empty. With
  // reduced motion they only show position.
  renderProgress() {
    this.fills.forEach((fill, i) => {
      let amount = i < this.index ? 1 : 0;
      if (i === this.index) amount = reducedMotion.matches ? 1 : this.elapsed / this.duration;
      fill.style.transform = `scaleX(${amount})`;
    });
  }

  // ---- input ----------------------------------------------------------
  bindPointer() {
    const card = this.card;
    let press = null; // { id, x, y, moved, held, timer }

    const end = (e, cancelled) => {
      if (!press || e.pointerId !== press.id) return;
      clearTimeout(press.timer);
      if (press.held) {
        this.release("pointer");
      } else if (!cancelled && !press.moved) {
        const rect = card.getBoundingClientRect();
        const back = e.clientX - rect.left < rect.width * PREV_ZONE;
        this.show(this.index + (back ? -1 : 1), { manual: true });
      }
      press = null;
    };

    card.addEventListener("pointerdown", (e) => {
      // OPEN TOOL → only follows its link.
      if (press || !e.isPrimary || e.button !== 0 || e.target.closest("a")) return;
      const current = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: false, held: false };
      current.timer = setTimeout(() => {
        current.held = true;
        this.hold("pointer");
      }, HOLD_MS);
      press = current;
      card.setPointerCapture(e.pointerId);
    });
    card.addEventListener("pointermove", (e) => {
      if (press && e.pointerId === press.id
          && Math.hypot(e.clientX - press.x, e.clientY - press.y) > TAP_SLOP_PX) press.moved = true;
    });
    card.addEventListener("pointerup", (e) => end(e, false));
    // A touch that turns into a scroll is cancelled: neither tap nor hold.
    card.addEventListener("pointercancel", (e) => end(e, true));
    card.addEventListener("lostpointercapture", (e) => end(e, true));
    card.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  bindKeys() {
    this.root.addEventListener("keydown", (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        this.show(this.index + (e.key === "ArrowLeft" ? -1 : 1), { manual: true });
      } else if (e.key === " " && e.target === this.root) {
        e.preventDefault(); // not a page scroll
        if (e.repeat) return;
        if (this.holds.has("key")) this.release("key");
        else this.hold("key");
      }
    });
  }

  // Scrolled mostly out of view, or the browser tab hidden: everything
  // stops, audio included, and picks up again when it is back.
  watchVisibility() {
    new IntersectionObserver((entries) => {
      const entry = entries[entries.length - 1];
      this.onScreen = entry.isIntersecting && entry.intersectionRatio >= MIN_VISIBLE;
      this.syncAudio();
    }, { threshold: [0, MIN_VISIBLE] }).observe(this.root);
    document.addEventListener("visibilitychange", () => {
      this.pageVisible = document.visibilityState === "visible";
      this.syncAudio();
    });
  }
}

// ---------------------------------------------------------------------
// Hero backdrop: a log-frequency grid from 20 Hz to 20 kHz, with the
// demo track's spectrum drifting across it in real time. The spectra
// come precomputed (scripts/make_landing_assets.py); this only draws them.
// ---------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";
const PLOT_LABELS_HZ = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
const PLOT_DB_LINES = 6;    // horizontal gridlines: every 10 dB of the 60 dB range
const PLOT_HEADROOM = 0.1;  // the loudest level stops this far below the top

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

class HeroPlot {
  // data: {hz: [low, high], frame_s, frames: [[level 0..100, ...], ...]} or null.
  constructor(root, data) {
    this.root = root;
    this.data = data && data.frames.length ? data : null;
    const [low, high] = data ? data.hz : [20, 20000];
    const x = (hz) => (1000 * Math.log(hz / low)) / Math.log(high / low);

    // 1-9 times each power of ten; the powers of ten are the major lines.
    let minor = "";
    let major = "";
    for (let decade = 1; decade <= high; decade *= 10) {
      for (let m = 1; m <= 9; m++) {
        const hz = m * decade;
        if (hz < low || hz > high) continue;
        if (m === 1) major += `M${x(hz).toFixed(1)} 0V1000`;
        else minor += `M${x(hz).toFixed(1)} 0V1000`;
      }
    }
    for (let i = 0; i <= PLOT_DB_LINES; i++) minor += `M0 ${((1000 * i) / PLOT_DB_LINES).toFixed(1)}H1000`;

    this.svg = svgEl("svg", { viewBox: "0 0 1000 1000", preserveAspectRatio: "none" });
    this.svg.append(
      svgEl("path", { class: "plot-grid plot-grid-minor", d: minor }),
      svgEl("path", { class: "plot-grid", d: major }));
    root.append(this.svg, el("div", { class: "plot-labels" }, PLOT_LABELS_HZ.map((hz) =>
      el("span", { style: `left: ${x(hz) / 10}%` }, hz >= 1000 ? `${hz / 1000}k` : String(hz)))));
    if (!this.data) return;

    root.append(el("p", { class: "plot-caption" }, "Spectrum · demo track"));
    this.fill = svgEl("path", { class: "plot-fill" });
    this.line = svgEl("path", { class: "plot-line" });
    this.svg.append(this.fill, this.line);
    const { frames } = this.data;
    this.xs = frames[0].map((_, i) => (1000 * i) / (frames[0].length - 1));
    // Reduced motion shows the whole track's average instead.
    this.average = frames[0].map((_, k) => frames.reduce((sum, row) => sum + row[k], 0) / frames.length);
    this.raf = null;
    this.onScreen = true;

    new IntersectionObserver(([entry]) => {
      this.onScreen = entry.isIntersecting;
      this.update();
    }).observe(root);
    document.addEventListener("visibilitychange", () => this.update());
    reducedMotion.addEventListener("change", () => this.update());
    this.draw(this.average);
    this.update();
  }

  // Animate only while the hero can be seen and motion is welcome.
  update() {
    const run = this.onScreen && document.visibilityState === "visible" && !reducedMotion.matches;
    if (run && this.raf === null) {
      this.raf = requestAnimationFrame((now) => this.frame(now));
    } else if (!run && this.raf !== null) {
      cancelAnimationFrame(this.raf);
      this.raf = null;
    }
    if (reducedMotion.matches) this.draw(this.average);
  }

  // The spectrum at this moment of the (looping) track, eased between
  // its two nearest frames.
  frame(now) {
    this.raf = null;
    const { frames, frame_s: step } = this.data;
    const pos = (now / 1000 / step) % frames.length;
    const i = Math.floor(pos);
    const t = pos - i;
    const a = frames[i];
    const b = frames[(i + 1) % frames.length];
    this.draw(a.map((level, k) => level + (b[k] - level) * t));
    this.update();
  }

  draw(levels) {
    const scale = 10 * (1 - PLOT_HEADROOM);
    const points = levels.map((level, i) => `${this.xs[i].toFixed(1)} ${(1000 - level * scale).toFixed(1)}`);
    const line = `M${points.join("L")}`;
    this.line.setAttribute("d", line);
    this.fill.setAttribute("d", `${line}L1000 1000L0 1000Z`);
  }
}

// ---------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------

// SOUND: off by default; the choice is remembered. Returns the current state.
function initSoundToggle(button, onChange) {
  const state = button.querySelector("[data-sound-state]");
  let on = storageGet("localStorage", SOUND_KEY) === "on";
  const render = () => {
    button.setAttribute("aria-pressed", String(on));
    state.textContent = on ? "On" : "Off";
  };
  render();
  button.addEventListener("click", () => {
    on = !on;
    storageSet("localStorage", SOUND_KEY, on ? "on" : "off");
    render();
    onChange(on);
  });
  return on;
}

document.title = CONFIG.appName;
document.querySelectorAll("[data-app-name]").forEach((node) => {
  node.textContent = CONFIG.appName;
});
document.querySelector("[data-tagline]").textContent = CONFIG.tagline;
document.querySelector("[data-team-name]").textContent = CONFIG.teamName;
document.querySelector("[data-github]").href = CONFIG.githubUrl;
initThemeToggle(document.querySelector("[data-theme-toggle]"));
new HeroPlot(document.querySelector("[data-hero-plot]"), page.hero);

const reelRoot = document.querySelector("[data-reel]");
let reel = null;
const sound = initSoundToggle(document.querySelector("[data-sound-toggle]"), (on) => reel.setSound(on));
reel = new Reel(reelRoot, CONFIG.slides, page.previews, sound);

// SEE THE TOOLS ↓: bring the whole reel into view (smoothly, unless
// reduced motion) and focus it, so ← → and Space work straight away.
document.querySelector("[data-see-tools]").addEventListener("click", (e) => {
  e.preventDefault();
  reelRoot.scrollIntoView({ block: "center" });
  reelRoot.focus({ preventScroll: true });
});
