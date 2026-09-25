// The source box inside every tool: each tool has its own file. Upload
// (drag-and-drop or browse), USE SAME FILE (the last file loaded in any
// tool, reused by file_id without re-uploading) or the tool's demo track.
import { CONFIG } from "./config.js";
import { el, fmtTime, requestJSON } from "./util.js";

// The most recent file uploaded (or demo loaded) in any tool.
const shared = { last: null, boxes: new Set() };

function setLastFile(meta) {
  shared.last = meta;
  shared.boxes.forEach((box) => box.showSameButton());
}

export const lastFileId = () => (shared.last ? shared.last.file_id : null);

function describe(meta) {
  const channels = meta.channels === 1 ? "mono" : meta.channels === 2 ? "stereo" : `${meta.channels} ch`;
  return `${fmtTime(meta.duration, 0)} · ${+(meta.samplerate / 1000).toFixed(1)} kHz · ${channels}`;
}

// The demo track most tools use; Separation's is a song with vocals.
export const DEFAULT_DEMO = { url: "/upload/demo", label: "Use demo track" };

export class SourceBox {
  // onLoad(meta) runs whenever this tool gets a new file.
  // demo: { url, label } of the demo button (the route that registers it).
  constructor(root, onLoad, demo = DEFAULT_DEMO) {
    this.root = root;
    this.onLoad = onLoad;
    this.demo = demo;
    this.meta = null;
    const { extensions, max_upload_mb: maxMb, max_duration_sec: maxSec } = CONFIG.limits;

    this.fileInput = el("input", {
      type: "file", hidden: true,
      accept: extensions.map((ext) => `.${ext}`).join(","),
      onchange: () => {
        const file = this.fileInput.files[0];
        this.fileInput.value = "";
        if (file) this.upload(file);
      },
    });
    this.dropMain = el("span", { class: "dropzone-main" }, "Drop a track here");
    this.dropzone = el("button", {
      type: "button", class: "dropzone", onclick: () => this.fileInput.click(),
    },
    this.dropMain,
    el("span", { class: "dropzone-sub" },
      `or click to browse · ${extensions.join(", ").toUpperCase()} · up to ${maxMb} MB, ${fmtTime(maxSec, 0)}`));

    this.sameButton = el("button", {
      type: "button", class: "btn btn-small", hidden: true,
      onclick: () => this.load(shared.last),
    });
    this.demoButton = el("button", {
      type: "button", class: "btn btn-small", onclick: () => this.useDemo(),
    }, demo.label);
    this.keepButton = el("button", {
      type: "button", class: "btn btn-small", hidden: true,
      onclick: () => this.showLoaded(),
    }, "Keep current");
    this.chooser = el("div", { class: "field" },
      this.dropzone,
      el("div", { class: "source-actions" }, this.sameButton, this.demoButton, this.keepButton));

    this.name = el("div", { class: "source-name" });
    this.details = el("div", { class: "source-meta" });
    this.loaded = el("div", { class: "source-loaded", hidden: true },
      el("div", { class: "source-text" }, this.name, this.details),
      el("button", {
        type: "button", class: "btn btn-small", onclick: () => this.showChooser(),
      }, "Replace"));

    this.error = el("div", { class: "msg msg-warn", role: "alert", hidden: true });

    root.append(el("p", { class: "label" }, "Source"), this.chooser, this.loaded, this.error, this.fileInput);

    // Dropping a file anywhere on the box loads it (also replaces a loaded one).
    root.addEventListener("dragover", (e) => {
      if (![...e.dataTransfer.types].includes("Files")) return;
      e.preventDefault();
      root.classList.add("is-dragover");
    });
    root.addEventListener("dragleave", (e) => {
      if (!root.contains(e.relatedTarget)) root.classList.remove("is-dragover");
    });
    root.addEventListener("drop", (e) => {
      e.preventDefault();
      root.classList.remove("is-dragover");
      const file = e.dataTransfer.files[0];
      if (file) this.upload(file);
    });

    shared.boxes.add(this);
    this.showSameButton();
  }

  showSameButton() {
    const last = shared.last;
    const usable = last && (!this.meta || last.file_id !== this.meta.file_id);
    this.sameButton.hidden = !usable;
    if (usable) {
      this.sameButton.replaceChildren("Use same file ", el("span", { class: "dim" }, last.filename));
    }
  }

  showChooser() {
    this.chooser.hidden = false;
    this.loaded.hidden = true;
    this.keepButton.hidden = !this.meta;
    this.showSameButton();
  }

  showLoaded() {
    this.chooser.hidden = true;
    this.loaded.hidden = false;
    this.error.hidden = true;
  }

  showError(message) {
    this.error.textContent = message;
    this.error.hidden = false;
  }

  setBusy(text) {
    const busy = Boolean(text);
    this.dropMain.textContent = busy ? text : "Drop a track here";
    [this.dropzone, this.sameButton, this.demoButton, this.keepButton].forEach((b) => {
      b.disabled = busy;
    });
  }

  load(meta) {
    this.meta = meta;
    this.name.textContent = meta.filename;
    this.name.title = meta.filename;
    this.details.textContent = describe(meta);
    this.showLoaded();
    this.showSameButton();
    this.onLoad(meta);
  }

  async upload(file) {
    const { extensions, max_upload_mb: maxMb } = CONFIG.limits;
    this.error.hidden = true;
    // Same checks as the server, done first so a huge file isn't sent at all.
    const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!extensions.includes(ext)) {
      const allowed = extensions.map((e) => `.${e}`).join(", ");
      this.showError(`File rejected: unsupported format '.${ext}'. Allowed: ${allowed}.`);
      return;
    }
    if (file.size > maxMb * 1024 * 1024) {
      this.showError(`File rejected: larger than ${maxMb} MB.`);
      return;
    }

    this.setBusy(`Uploading ${file.name}…`);
    try {
      const body = new FormData();
      body.append("file", file);
      const meta = await requestJSON("/upload", { method: "POST", body });
      setLastFile(meta);
      this.load(meta);
    } catch (err) {
      this.showError(err.status ? `File rejected: ${err.message}` : `Upload failed: ${err.message}`);
    } finally {
      this.setBusy(null);
    }
  }

  async useDemo() {
    this.error.hidden = true;
    this.setBusy("Loading demo…");
    try {
      const meta = await requestJSON(this.demo.url, { method: "POST" });
      setLastFile(meta);
      this.load(meta);
    } catch (err) {
      this.showError(`Demo track unavailable: ${err.message}`);
    } finally {
      this.setBusy(null);
    }
  }
}
