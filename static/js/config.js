// App-wide settings. Everything tool-specific (slider ranges and
// defaults, labels, presets, upload limits) comes from the server's
// PARAMS tables through the #spectra-config JSON, never from here.
const page = JSON.parse(document.getElementById("spectra-config").textContent);

export const CONFIG = {
  appName: "SPECTRA",
  responseDebounceMs: 150,
  // A stuck request must never leave RUN disabled for good.
  runTimeoutMs: 5 * 60 * 1000,
  ...page,
};

export const toolInfo = (id) => CONFIG.tools.find((tool) => tool.id === id);
