// SPECTRA page bootstrap: header, tabs, drawer, tools, Backstage.
import { Backstage } from "./backstage.js";
import { CONFIG } from "./config.js";
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

const app = {
  drawer: initDrawer(),
  backstage: new Backstage(document.getElementById("view-backstage")),
  tools: null,

  // runId (Backstage only): the run to select.
  showView(id, { focusTab = false, runId = null } = {}) {
    if (id === currentView || !views.has(id)) return;
    currentView = id;
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

app.showView(tabs[0].dataset.view);
