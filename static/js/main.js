// SPECTRA page bootstrap: header, tabs, drawer, tools, Backstage.
import { Backstage } from "./backstage.js";
import { CONFIG, toolInfo } from "./config.js";
import { initDrawer } from "./drawer.js";
import { installSpaceShortcut } from "./player.js";
import { initThemeToggle } from "./theme.js";
import { createTools } from "./tools.js";

document.querySelector("[data-app-name]").textContent = CONFIG.appName;
document.title = CONFIG.appName;
initThemeToggle(document.querySelector("[data-theme-toggle]"));
installSpaceShortcut();

const tabs = [...document.querySelectorAll('.tabbar [role="tab"]')];
const views = new Map(tabs.map((tab) => [tab.dataset.view, document.getElementById(`view-${tab.dataset.view}`)]));
let currentView = null;
const backButton = document.querySelector("[data-bs-back]");
let returnTo = null; // { view, scroll }: the tool tab Backstage was opened from

// Deep links: /lab#reverb opens that tab, and the hash follows the open
// tab. Links say #separation for the "separate" view.
const HASH_ALIASES = { separation: "separate" };
const viewFromHash = () => {
  const name = decodeURIComponent(window.location.hash.slice(1)).toLowerCase();
  return HASH_ALIASES[name] || name;
};
const hashFor = (id) => Object.keys(HASH_ALIASES).find((name) => HASH_ALIASES[name] === id) || id;

const app = {
  drawer: initDrawer(),
  backstage: new Backstage(document.getElementById("view-backstage")),
  tools: null,

  // runId (Backstage only): the run to select.
  showView(id, { focusTab = false, runId = null } = {}) {
    if (id === currentView || !views.has(id)) return;
    if (id === "backstage" && currentView) {
      returnTo = { view: currentView, scroll: window.scrollY };
      backButton.querySelector("[data-bs-back-name]").textContent = toolInfo(currentView).name;
      backButton.hidden = false;
    }
    currentView = id;
    // replaceState: switching tabs adds no history entries and fires no hashchange.
    history.replaceState(null, "", `#${hashFor(id)}`);
    for (const tab of tabs) {
      const active = tab.dataset.view === id;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active && focusTab) tab.focus();
    }
    views.forEach((view, key) => {
      view.hidden = key !== id;
    });
    this.drawer.viewChanged(id);
    const tool = this.tools.get(id);
    if (tool) tool.shown();
    if (id === "backstage") this.backstage.shown(runId);
  },

  recordRun(runId) {
    this.backstage.record(runId);
  },

  // → BACKSTAGE under a tool's output: open Backstage on that run.
  openBackstage(runId) {
    this.showView("backstage", { runId });
  },
};

app.tools = createTools(app);

// ← BACK TO <TOOL> in Backstage: that tab, scrolled to where it was left.
backButton.addEventListener("click", () => {
  app.showView(returnTo.view, { focusTab: true });
  window.scrollTo(0, returnTo.scroll);
});

// Tabs: click, or arrow keys / Home / End once a tab has focus.
tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => app.showView(tab.dataset.view));
  tab.addEventListener("keydown", (e) => {
    const moves = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 };
    if (!(e.key in moves)) return;
    e.preventDefault();
    const next = tabs[(moves[e.key] + tabs.length) % tabs.length];
    app.showView(next.dataset.view, { focusTab: true });
  });
});

window.addEventListener("hashchange", () => {
  if (views.has(viewFromHash())) app.showView(viewFromHash());
});

app.showView(views.has(viewFromHash()) ? viewFromHash() : tabs[0].dataset.view);
