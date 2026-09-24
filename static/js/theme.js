// Dark / light theme: data-theme on <html>, remembered in localStorage.
// Charts and waveforms read their colours from the CSS variables at draw
// time and redraw through onThemeChange.
import { storageSet } from "./util.js";

const STORAGE_KEY = "spectra-theme";
const listeners = new Set();

export function currentTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

export function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  storageSet("localStorage", STORAGE_KEY, theme);
  listeners.forEach((fn) => fn(theme));
}

export function onThemeChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function colors() {
  return {
    bg: cssVar("--bg"),
    panel: cssVar("--panel"),
    line: cssVar("--line"),
    fg: cssVar("--fg"),
    fgDim: cssVar("--fg-dim"),
    accent: cssVar("--accent"),
    accentText: cssVar("--accent-text"),
    warn: cssVar("--warn"),
  };
}

// "#22D3EE", 0.4 -> "rgba(34, 211, 238, 0.4)"
export function withAlpha(hex, alpha) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export function initThemeToggle(button) {
  const label = () => `Switch to ${currentTheme() === "dark" ? "light" : "dark"} theme`;
  button.setAttribute("aria-label", label());
  button.addEventListener("click", () => {
    setTheme(currentTheme() === "dark" ? "light" : "dark");
    button.setAttribute("aria-label", label());
  });
}
