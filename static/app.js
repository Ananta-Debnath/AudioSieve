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

// Upload -> process -> play
document.querySelectorAll("form[data-tool]").forEach((form) => {
  const panel = form.closest(".panel");
  const status = panel.querySelector(".status");
  const result = panel.querySelector(".result");

  const setStatus = (msg, isError = false) => {
    status.textContent = msg;
    status.classList.toggle("error", isError);
  };

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const button = form.querySelector("button");
    button.disabled = true;
    result.hidden = true;
    try {
      setStatus("Uploading...");
      const upload = await postJSON("/upload", { body: new FormData(form) });

      setStatus("Processing...");
      const processed = await postJSON(`/process/${form.dataset.tool}`, {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_id: upload.file_id }),
      });

      panel.querySelector("audio").src = processed.url;
      panel.querySelector(".download").href = processed.download_url;
      result.hidden = false;
      setStatus(`Done: ${upload.filename} (${upload.duration.toFixed(1)} s)`);
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      button.disabled = false;
    }
  });
});
