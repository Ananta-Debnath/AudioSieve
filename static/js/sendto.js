// SEND TO under a result (an effect's output or one stem): a menu of
// every tool, this one included. Picking one registers the result as an
// upload on the server and loads it as that tool's source, so a result
// can go through another effect, or the same one again, without a
// download and re-upload.
import { CONFIG } from "./config.js";
import { el, requestJSON } from "./util.js";

let openMenu = null; // at most one menu is open

// A press outside the open menu closes it.
document.addEventListener("pointerdown", (e) => {
  if (openMenu && !openMenu.root.contains(e.target)) openMenu.close();
});

export class SendTo {
  // resultId: the result to send; name: what the new source is called
  // (the server adds ".wav"); fromTool: the tool the result came from.
  constructor(app, { resultId, name, fromTool, small = false }) {
    this.app = app;
    this.resultId = resultId;
    this.name = name;
    this.sending = false;

    this.button = el("button", {
      type: "button", class: small ? "btn btn-small" : "btn",
      "aria-haspopup": "menu", "aria-expanded": "false",
      onclick: () => (this.isOpen() ? this.close() : this.open()),
    }, "Send to ▾");
    this.items = CONFIG.tools.map((tool) => el("button", {
      type: "button", class: "menu-item", role: "menuitem", tabindex: "-1",
      onclick: () => this.send(tool.id),
    },
    el("span", { class: "tab-num" }, tool.number),
    el("span", {}, tool.name),
    tool.id === fromTool ? el("span", { class: "dim menu-here" }, "again") : null));
    this.error = el("p", { class: "msg msg-warn menu-msg", role: "alert", hidden: true });
    this.menu = el("div", { class: "menu", role: "menu", "aria-label": "Send to", hidden: true },
      el("p", { class: "label menu-label" }, "Use as the source of"), this.items, this.error);
    this.root = el("div", { class: "menu-wrap" }, this.button, this.menu);

    this.root.addEventListener("keydown", (e) => this.keydown(e));
    // Tabbing out of the menu closes it.
    this.root.addEventListener("focusout", (e) => {
      if (this.isOpen() && !this.root.contains(e.relatedTarget)) this.close();
    });
  }

  isOpen() {
    return !this.menu.hidden;
  }

  open() {
    if (openMenu) openMenu.close();
    openMenu = this;
    this.error.hidden = true;
    this.menu.hidden = false;
    this.button.setAttribute("aria-expanded", "true");
    // Open to the left if it would run off the right edge, and scroll
    // so all of it shows (the output is often at the bottom of the page).
    this.menu.classList.remove("menu-right");
    if (this.menu.getBoundingClientRect().right > document.documentElement.clientWidth) {
      this.menu.classList.add("menu-right");
    }
    this.menu.scrollIntoView({ block: "nearest" });
    this.items[0].focus({ preventScroll: true });
  }

  close(refocus = false) {
    if (openMenu === this) openMenu = null;
    this.menu.hidden = true;
    this.button.setAttribute("aria-expanded", "false");
    if (refocus) this.button.focus();
  }

  // Arrows / Home / End move through the tools; Esc closes (and doesn't
  // also close the HOW IT WORKS drawer).
  keydown(e) {
    if (!this.isOpen()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      this.close(true);
      return;
    }
    const i = this.items.indexOf(document.activeElement);
    const last = this.items.length - 1;
    const moves = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: last };
    if (!(e.key in moves)) return;
    e.preventDefault();
    this.items[(moves[e.key] + this.items.length) % this.items.length].focus();
  }

  async send(toolId) {
    // Not disabling the items: that would take focus out of the menu.
    if (this.sending) return;
    this.sending = true;
    this.error.hidden = true;
    this.button.textContent = "Sending…";
    try {
      const meta = await requestJSON(`/upload/result/${this.resultId}`, {
        method: "POST", json: { name: this.name },
      });
      this.close();
      // Sending to its own tool replaces this output (and this menu).
      this.app.sendTo(toolId, meta);
    } catch (err) {
      this.error.textContent = `Could not send: ${err.message}`;
      this.error.hidden = false;
    } finally {
      this.sending = false;
      this.button.textContent = "Send to ▾";
    }
  }
}
