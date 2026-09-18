// Tab switching
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) => {
      p.hidden = p.id !== `panel-${btn.dataset.tab}`;
    });
  });
});

async function postJSON(url, options) {
  const res = await fetch(url, { method: "POST", ...options });
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// Gain curve (dB vs. log frequency) returned by /process/eq
function drawCurve(canvas, { freqs, gain_db }) {
  const ctx = canvas.getContext("2d");
  const { width: W, height: H } = canvas;
  const pad = { left: 44, right: 10, top: 10, bottom: 26 };
  const fMin = 20, fMax = freqs[freqs.length - 1];
  const dbMin = -40, dbMax = 20;

  const x = (f) => pad.left + (Math.log10(f / fMin) / Math.log10(fMax / fMin)) * (W - pad.left - pad.right);
  const y = (db) => pad.top + ((dbMax - Math.max(dbMin, Math.min(dbMax, db))) / (dbMax - dbMin)) * (H - pad.top - pad.bottom);

  ctx.clearRect(0, 0, W, H);
  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;

  // Grid + labels
  ctx.strokeStyle = "#e3e3e3";
  ctx.fillStyle = "#666";
  ctx.textAlign = "center";
  for (const f of [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]) {
    if (f > fMax) continue;
    ctx.beginPath(); ctx.moveTo(x(f), pad.top); ctx.lineTo(x(f), H - pad.bottom); ctx.stroke();
    ctx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, x(f), H - 8);
  }
  ctx.textAlign = "right";
  for (let db = dbMin; db <= dbMax; db += 10) {
    ctx.strokeStyle = db === 0 ? "#999" : "#e3e3e3";
    ctx.beginPath(); ctx.moveTo(pad.left, y(db)); ctx.lineTo(W - pad.right, y(db)); ctx.stroke();
    ctx.fillText(`${db} dB`, pad.left - 4, y(db) + 4);
  }

  // Curve
  ctx.strokeStyle = "#1565c0";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  freqs.forEach((f, i) => {
    if (f < fMin) return;
    if (!started) { ctx.moveTo(x(f), y(gain_db[i])); started = true; }
    else ctx.lineTo(x(f), y(gain_db[i]));
  });
  ctx.stroke();
}

// EQ presets: choosing one fills in the form fields with its values;
// editing any field afterwards switches the preset back to "(none)",
// so what the form shows is always what gets processed.
const eqForm = document.querySelector('form[data-tool="eq"]');
const presetData = document.getElementById("eq-presets");
if (eqForm && presetData) {
  const presets = JSON.parse(presetData.textContent);
  const presetSelect = eqForm.elements.preset;

  const setField = (name, value) => {
    const field = eqForm.elements[name];
    if (field && value !== undefined && value !== null) field.value = value;
  };

  presetSelect.addEventListener("change", () => {
    const preset = presets[presetSelect.value];
    if (!preset) return;

    setField("mode", preset.mode);
    // Band gains: the preset's bands, or flat for filter-only presets.
    for (let i = 0; eqForm.elements[`band_${i}_gain_db`]; i++) {
      setField(`band_${i}_gain_db`, preset.bands ? preset.bands[i].gain_db : 0);
    }
    for (const key of ["filter_type", "cutoff", "low_cutoff", "high_cutoff", "order"]) {
      setField(key, preset[key]);
    }
  });

  eqForm.addEventListener("input", (e) => {
    if (e.target !== presetSelect && e.target.type !== "file") presetSelect.value = "";
  });
}

// Upload -> process -> play
document.querySelectorAll("form[data-tool]").forEach((form) => {
  const panel = form.closest(".panel");
  const status = panel.querySelector(".status");
  const result = panel.querySelector(".result");

  const setStatus = (msg, isError = false) => {
    status.textContent = msg;
    status.classList.toggle("error", isError);
  };

  // Play the chosen file straight from disk so it can be compared with
  // the processed result (no upload needed).
  const fileInput = form.querySelector('input[type="file"]');
  const original = form.querySelector(".original");
  const originalAudio = form.querySelector(".audio-original");
  let originalUrl = null;
  fileInput.addEventListener("change", () => {
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    originalUrl = null;
    const file = fileInput.files[0];
    if (file) {
      originalUrl = URL.createObjectURL(file);
      originalAudio.src = originalUrl;
    } else {
      originalAudio.removeAttribute("src");
    }
    original.hidden = !file;
    result.hidden = true; // old result belongs to the previous file
  });

  // A/B comparison: starting one player pauses the other.
  const players = panel.querySelectorAll("audio");
  players.forEach((player) => {
    player.addEventListener("play", () => {
      players.forEach((other) => { if (other !== player) other.pause(); });
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const button = form.querySelector("button");
    button.disabled = true;
    result.hidden = true;
    try {
      setStatus("Uploading...");
      const upload = await postJSON("/upload", { body: new FormData(form) });

      // Every form field except the file itself is an effect parameter.
      const params = {};
      for (const [key, value] of new FormData(form)) {
        if (key !== "file") params[key] = value;
      }

      setStatus("Processing...");
      const processed = await postJSON(`/process/${form.dataset.tool}`, {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...params, file_id: upload.file_id }),
      });

      panel.querySelector(".audio-processed").src = processed.url;
      panel.querySelector(".download").href = processed.download_url;
      if (processed.curve) drawCurve(panel.querySelector("canvas.curve"), processed.curve);
      if (processed.spectrograms) {
        panel.querySelector(".spec-before").src = processed.spectrograms.before;
        panel.querySelector(".spec-after").src = processed.spectrograms.after;
      }
      result.hidden = false;
      setStatus(`Done: ${upload.filename} (${upload.duration.toFixed(1)} s)`);
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      button.disabled = false;
    }
  });
});
