// Audio players: WaveSurfer for display and playback only (no effects).
// The server sends each waveform's min/max envelope ("peaks") and
// duration, so WaveSurfer streams the file instead of decoding it.
import { colors, onThemeChange, withAlpha } from "./theme.js";
import { el, fmtTime, isVisible } from "./util.js";

const players = new Set();
let lastUsed = null; // the Player (or StemGroup) the Space bar toggles

function waveColors(variant) {
  const c = colors();
  if (variant === "original") {
    return { waveColor: c.fgDim, progressColor: c.fg, cursorColor: c.fg };
  }
  return { waveColor: withAlpha(c.accent, 0.45), progressColor: c.accent, cursorColor: c.fg };
}

onThemeChange(() => players.forEach((p) => p.ws.setOptions(waveColors(p.variant))));

// Starting one player pauses every other one, except the members of a
// stem group that is playing in sync.
function pauseOthers(started) {
  for (const p of players) {
    const sameSyncedGroup = started.group && started.group.syncing && p.group === started.group;
    if (p !== started && !sameSyncedGroup && p.isPlaying()) p.pause();
  }
}

export class Player {
  // variant: "original" | "processed" | "stem"
  // source: { url, peaks?, duration? }
  constructor(parent, { label, variant, source, height = 64, showLabel = true }) {
    this.variant = variant;
    this.group = null;

    this.button = el("button", {
      class: "play-btn", type: "button", "aria-pressed": "false",
      "aria-label": `Play ${label.toLowerCase()}`,
      onclick: () => {
        lastUsed = this;
        if (this.group) this.group.syncing = false; // a single stem on its own
        this.toggle();
      },
    }, "▶");
    this.time = el("span", { class: "player-time" }, "0:00.0 / –");
    this.wave = el("div", { class: "player-wave" });
    this.root = el("div", { class: `player player-${variant}` },
      el("div", { class: "player-head" },
        this.button,
        showLabel ? el("span", { class: "player-label" }, label) : null,
        this.time),
      this.wave);
    parent.append(this.root);

    const options = {
      container: this.wave,
      height,
      url: source.url,
      normalize: false, // same scale for before and after, so level changes show
      cursorWidth: 2,
      dragToSeek: true,
      hideScrollbar: true,
      ...waveColors(variant),
    };
    if (source.peaks && source.duration) {
      options.peaks = source.peaks;
      options.duration = source.duration;
    }
    this.ws = WaveSurfer.create(options);
    this.ws.on("ready", () => this.showTime());
    this.ws.on("timeupdate", () => this.showTime());
    this.ws.on("play", () => {
      pauseOthers(this);
      this.showState();
    });
    this.ws.on("pause", () => this.showState());
    this.ws.on("finish", () => this.showState());
    this.ws.on("interaction", (newTime) => {
      lastUsed = this.group && this.group.syncing ? this.group : this;
      if (this.group && this.group.syncing) this.group.seek(newTime, this);
    });
    this.ws.on("error", () => {
      this.wave.replaceChildren(el("p", { class: "player-error" }, "Could not load this audio."));
    });
    players.add(this);
  }

  showTime() {
    this.time.textContent = `${fmtTime(this.ws.getCurrentTime())} / ${fmtTime(this.ws.getDuration())}`;
  }

  showState() {
    const playing = this.ws.isPlaying();
    this.button.textContent = playing ? "❚❚" : "▶";
    this.button.setAttribute("aria-pressed", String(playing));
  }

  isPlaying() {
    return this.ws.isPlaying();
  }

  play() {
    return this.ws.play().catch(() => {});
  }

  pause() {
    this.ws.pause();
  }

  toggle() {
    if (this.isPlaying()) this.pause();
    else this.play();
  }

  currentTime() {
    return this.ws.getCurrentTime();
  }

  setTime(seconds) {
    this.ws.setTime(seconds);
  }

  setVolume(volume) {
    this.ws.setVolume(volume);
  }

  isVisible() {
    return isVisible(this.root);
  }

  destroy() {
    if (lastUsed === this) lastUsed = null;
    players.delete(this);
    this.ws.destroy();
    this.root.remove();
  }
}

// Stems that PLAY ALL starts together and seeks together. Volume, mute
// and solo only set each player's volume (playback, not processing).
export class StemGroup {
  constructor(members, onChange) {
    this.members = members;
    this.syncing = false;
    this.onChange = onChange;
    members.forEach((p) => {
      p.group = this;
      p.ws.on("play", onChange);
      p.ws.on("pause", onChange);
      p.ws.on("finish", () => {
        if (this.syncing && !this.isPlaying()) this.syncing = false;
        onChange();
      });
    });
    // Media elements drift apart a little; pull them back onto the first one.
    members[0].ws.on("timeupdate", (t) => {
      if (!this.syncing) return;
      for (const p of members.slice(1)) {
        if (p.isPlaying() && Math.abs(p.currentTime() - t) > 0.08) p.setTime(t);
      }
    });
  }

  isPlaying() {
    return this.syncing && this.members.some((p) => p.isPlaying());
  }

  play() {
    lastUsed = this;
    const t = Math.max(...this.members.map((p) => p.currentTime()));
    this.syncing = true;
    this.members.forEach((p) => p.setTime(t));
    this.members.forEach((p) => p.play());
    this.onChange();
  }

  pause() {
    this.syncing = false;
    this.members.forEach((p) => p.pause());
    this.onChange();
  }

  toggle() {
    lastUsed = this;
    if (this.isPlaying()) this.pause();
    else this.play();
  }

  seek(seconds, from) {
    this.members.forEach((p) => {
      if (p !== from) p.setTime(seconds);
    });
  }

  isVisible() {
    return this.members.some((p) => p.isVisible());
  }
}

// Space plays / pauses the last-used player, unless focus is in an input,
// or a keyboard user has focused a button (Space activates that button).
export function installSpaceShortcut() {
  let handled = false;
  document.addEventListener("keydown", (e) => {
    if (e.key !== " " || e.ctrlKey || e.metaKey || e.altKey) return;
    const target = e.target;
    if (target.closest("input, textarea, select, [contenteditable]")) return;
    if (target.closest("button, a") && target.matches(":focus-visible")) return;
    if (!lastUsed || !lastUsed.isVisible()) return;
    e.preventDefault();
    handled = true;
    if (!e.repeat) lastUsed.toggle();
  });
  // A mouse-focused button would otherwise "click" on keyup.
  document.addEventListener("keyup", (e) => {
    if (e.key === " " && handled) {
      e.preventDefault();
      handled = false;
    }
  });
}
