"""Warm-up before the demo: run every tool once on the demo track with
the demo's showcase settings, through the app's own routes, and print
PASS / FAIL with timings.

    python scripts/warmup.py            # the running app if there is one, else in-process
    python scripts/warmup.py --url http://127.0.0.1:5000
    python scripts/warmup.py --local    # in-process only (Flask's test client)

Against the running app (start it first with run.sh / run.bat), this
warms up that server process: imports, first-call costs, the OS file
cache. Without one, the same routes run inside this process instead,
which still shows any error before the presentation.

For each tool: POST /process/<tool>, fetch its audio, then its Backstage
analysis and images. Before that it checks the pages load and only use
local files (the lab may be offline). Exit code 1 if anything failed.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_URL = "http://127.0.0.1:5000"
TIMEOUT_S = 300  # separation of a long track takes a while

# The demo's showcase settings (everything else at its default).
WARMUP = {
    "eq": {"preset": "telephone"},
    "reverb": {"rt60": 2.5, "wet": 0.5},
    "echo": {"mode": "feedback", "delay_ms": 300, "gain": 0.5},
    "flanger": {},
    "separate": {},
}

# <script src="http..."> or <link href="http...">: anything the page
# would load from the internet.
EXTERNAL = re.compile(r"<(?:script|link)\b[^>]*\b(?:src|href)\s*=\s*[\"']?(?:https?:)?//", re.I)
LOCAL_ASSET = re.compile(r"<(?:script|link)\b[^>]*\b(?:src|href)=\"(/static/[^\"]+)\"", re.I)


class Failed(Exception):
    pass


# ---------------------------------------------------------------------
# Clients: the running server over HTTP, or the app in this process
# ---------------------------------------------------------------------

class HttpClient:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.where = f"{self.base} (the running app)"

    def request(self, method, path, body=None, timeout=TIMEOUT_S):
        headers, data = {}, None
        if method == "POST":
            data = json.dumps(body or {}).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return res.status, res.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()


class LocalClient:
    def __init__(self, flask_app=None):
        if flask_app is None:
            import app as app_module
            flask_app = app_module.app
        self.client = flask_app.test_client()
        self.where = "in-process (no server running)"

    def request(self, method, path, body=None, timeout=None):
        res = self.client.open(path, method=method, json=body if method == "POST" else None)
        return res.status_code, res.data


def server_is_up(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/lab", timeout=2) as res:
            return res.status == 200
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def fetch(client, path, method="GET", body=None):
    status, data = client.request(method, path, body)
    if status != 200:
        try:
            reason = json.loads(data)["error"]
        except (ValueError, KeyError, TypeError):
            reason = f"HTTP {status}"
        raise Failed(f"{method} {path}: {reason}")
    return data


def fetch_json(client, path, method="GET", body=None):
    return json.loads(fetch(client, path, method, body))


def check_pages(client):
    """/ and /lab load, reference no external URLs, and every local
    script, stylesheet and font they use is served."""
    assets = set()
    for page in ("/", "/lab"):
        html = fetch(client, page).decode("utf-8")
        if EXTERNAL.search(html):
            raise Failed(f"{page} loads something from the internet: {EXTERNAL.search(html).group(0)}")
        assets.update(LOCAL_ASSET.findall(html))
    for path in sorted(assets):
        body = fetch(client, path)
        if path.endswith(".css"):
            css = body.decode("utf-8")
            if re.search(r"url\(\s*[\"']?(?:https?:)?//", css) or "@import" in css:
                raise Failed(f"{path} loads something from the internet")
            folder = path.rsplit("/", 1)[0]
            for font in re.findall(r"url\(\"?([^\")]+\.woff2)\"?\)", css):
                fetch(client, f"{folder}/{font}")
                assets.add(font)
    return f"/, /lab and {len(assets)} local assets; no external URLs"


def run_tool(client, tool, file_id):
    """One run plus everything the Studio and Backstage then load for it."""
    start = time.perf_counter()
    data = fetch_json(client, f"/process/{tool}", "POST", {"file_id": file_id, **WARMUP[tool]})
    processed = time.perf_counter() - start

    if tool == "separate":
        for stem in data["stems"]:
            fetch(client, stem["url"])
    else:
        fetch(client, data["url"])
        for url in data["spectrograms"].values():
            fetch(client, url)

    start = time.perf_counter()
    analysis = fetch_json(client, f"/backstage/{data['run_id']}")
    images = analysis["images"]
    for url in [images["mixture"], *images["masks"].values()] if tool == "separate" else images.values():
        fetch(client, url)
    backstage = time.perf_counter() - start

    detail = f"run {processed:.1f} s, backstage {backstage:.1f} s"
    if tool == "separate":
        detail += f", {len(data['stems'])} stems"
        if data.get("note"):
            detail += f" ({data['note']})"
    return detail


def step(results, name, fn, *args):
    start = time.perf_counter()
    try:
        detail = fn(*args)
        ok = True
    except Failed as e:
        detail, ok = str(e), False
    except Exception as e:  # anything else is a failure too, not a crash of the warm-up
        detail, ok = f"{type(e).__name__}: {e}", False
    elapsed = time.perf_counter() - start
    results.append((name, ok, elapsed, detail))
    print(f"  {name:9} {'PASS' if ok else 'FAIL'}  {elapsed:6.1f} s  {detail}", flush=True)
    return ok


def warm_up(client):
    """Run every check; returns [(name, ok, seconds, detail)]."""
    print(f"SPECTRA warm-up against {client.where}", flush=True)
    results = []
    step(results, "pages", check_pages, client)

    upload = {}

    def upload_demo():
        upload.update(fetch_json(client, "/upload/demo", "POST"))
        return f"{upload['filename']}, {upload['duration']:.1f} s"

    if step(results, "demo", upload_demo):
        for tool in WARMUP:
            step(results, tool, run_tool, client, tool, upload["file_id"])
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--url", help=f"the running app (default: {DEFAULT_URL} if it answers)")
    where.add_argument("--local", action="store_true", help="run in-process, even if a server is up")
    args = parser.parse_args(argv)

    if args.local:
        client = LocalClient()
    elif args.url:
        client = HttpClient(args.url)
    else:
        client = HttpClient(DEFAULT_URL) if server_is_up(DEFAULT_URL) else LocalClient()

    results = warm_up(client)
    tools = [r for r in results if r[0] in WARMUP]
    failed = [r[0] for r in results if not r[1]]
    print()
    if failed:
        print(f"FAIL: {', '.join(failed)}. Fix these before the demo.")
        return 1
    print(f"PASS: all {len(tools)} tools, in {sum(r[2] for r in results):.1f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
