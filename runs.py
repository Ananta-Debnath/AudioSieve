"""Run registry: what each /process/* run did, for the Backstage tab.

A run is a dict: run_id, tool, params, file_id (the input upload),
outputs (file names inside the registry's directory), created (Unix
time), plus whatever extra fields the route adds.

Runs are kept in memory, and each one also gets a JSON sidecar next to
its output files (<run_id>.run.json), so they survive a server restart
(e.g. Flask's debug reloader).
"""

import json
import threading
import time
from pathlib import Path

SIDECAR_SUFFIX = ".run.json"


class RunRegistry:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._runs = None  # loaded from the sidecars on first use
        self._lock = threading.Lock()

    def _loaded(self):
        if self._runs is None:
            self._runs = {}
            for path in self.directory.glob(f"*{SIDECAR_SUFFIX}"):
                try:
                    run = json.loads(path.read_text(encoding="utf-8"))
                    self._runs[run["run_id"]] = run
                except (OSError, ValueError, KeyError, TypeError):
                    continue  # unreadable sidecar: skip it, don't break the list
        return self._runs

    def add(self, run_id, tool, params, file_id, outputs, **extra):
        """Record a run and write its sidecar. Returns the run dict."""
        run = {
            "run_id": run_id,
            "tool": tool,
            "params": params,
            "file_id": file_id,
            "outputs": outputs,
            "created": time.time(),
            **extra,
        }
        with self._lock:
            self._loaded()[run_id] = run
            sidecar = self.directory / f"{run_id}{SIDECAR_SUFFIX}"
            sidecar.write_text(json.dumps(run), encoding="utf-8")
        return run

    def get(self, run_id):
        with self._lock:
            return self._loaded().get(run_id)

    def list(self):
        """All runs, newest first."""
        with self._lock:
            runs = list(self._loaded().values())
        return sorted(runs, key=lambda run: run["created"], reverse=True)
