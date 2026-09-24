// Small DOM, formatting and fetch helpers.

// el("div", { class: "x", onclick: fn, hidden: true }, child, "text", ...)
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// 83.4 -> "1:23.4"; digits = decimals on the seconds.
export function fmtTime(seconds, digits = 1) {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  const width = digits ? 3 + digits : 2;
  return `${minutes}:${rest.toFixed(digits).padStart(width, "0")}`;
}

export function fmtHz(hz) {
  if (hz >= 1000) return `${+(hz / 1000).toFixed(hz >= 10000 ? 1 : 2)} kHz`;
  return `${+hz.toFixed(hz < 100 ? 1 : 0)} Hz`;
}

// Axis ticks: 20, 50, 200, 1k, 2k, 20k
export function fmtHzTick(hz) {
  return hz >= 1000 ? `${+(hz / 1000).toFixed(1)}k` : `${+hz.toFixed(1)}`;
}

export function fmtClock(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}

export class RequestError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// fetch -> JSON, throwing RequestError with the server's {"error"} message.
export async function requestJSON(url, { method = "GET", json, body, signal } = {}) {
  const init = { method, signal };
  if (json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }

  let res;
  try {
    res = await fetch(url, init);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new RequestError("Could not reach the server.", 0);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new RequestError((data && data.error) || `HTTP ${res.status}`, res.status);
  return data;
}

// localStorage / sessionStorage can throw (private mode, blocked site data).
export function storageGet(storage, key) {
  try {
    return window[storage].getItem(key);
  } catch {
    return null;
  }
}

export function storageSet(storage, key, value) {
  try {
    window[storage].setItem(key, value);
  } catch {
    /* not persisted; the page still works */
  }
}

export function isVisible(node) {
  return Boolean(node && node.isConnected && node.offsetParent !== null);
}
