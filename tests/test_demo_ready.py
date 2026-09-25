"""Tests for demo readiness (pivot stage 06): the pages work offline,
scripts/warmup.py, scripts/clean_runs.py and the default run settings.

Run with: python -m pytest tests -m "not slow"
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    import app as app_module

    for name in ("uploads", "processed"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.delenv("SEPARATION_MOCK", raising=False)
    return app_module


@pytest.fixture
def warmup():
    return load_script("warmup")


@pytest.fixture
def short_demo(app_module, tmp_path, monkeypatch):
    """A 1 s demo track, so a whole warm-up takes a few seconds."""
    sr = 22050
    t = np.arange(sr) / sr
    audio = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.05 * np.random.default_rng(0).standard_normal(sr)
    path = tmp_path / "demo.wav"
    sf.write(str(path), audio, sr)
    monkeypatch.setattr(app_module, "DEMO_PATH", path)
    return path


# ---------------------------------------------------------------------
# Offline: no CDN, no Google Fonts
# ---------------------------------------------------------------------

def test_pages_load_nothing_from_the_internet(app_module, warmup):
    detail = warmup.check_pages(warmup.LocalClient(app_module.app))
    assert "no external URLs" in detail


@pytest.mark.parametrize("html, external", [
    ('<script src="https://cdn.jsdelivr.net/npm/x.js"></script>', True),
    ('<link rel="stylesheet" href="//fonts.googleapis.com/css2">', True),
    ("<link rel='preconnect' href='https://fonts.gstatic.com'>", True),
    ('<script src="/static/vendor/chart.umd.min.js"></script>', False),
    ('<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\'">', False),
    ('<a href="https://github.com/">code</a>', False),  # a link, not a download
])
def test_the_external_url_check(warmup, html, external):
    assert bool(warmup.EXTERNAL.search(html)) == external


def test_vendored_files_and_fonts_are_there():
    vendor = ROOT / "static" / "vendor"
    assert b"Chart.js v4.5.1" in (vendor / "chart.umd.min.js").read_bytes()[:200]
    assert b"WaveSurfer" in (vendor / "wavesurfer.min.js").read_bytes()
    for family in ("big-shoulders-display", "martian-mono"):
        folder = ROOT / "static" / "fonts" / family
        assert "SIL Open Font License" in (folder / "OFL.txt").read_text(encoding="utf-8")
        fonts = list(folder.glob("*.woff2"))
        assert fonts and all(f.read_bytes()[:4] == b"wOF2" for f in fonts)


# ---------------------------------------------------------------------
# scripts/warmup.py
# ---------------------------------------------------------------------

def test_warmup_passes_every_tool_with_the_showcase_settings(app_module, warmup, short_demo):
    results = warmup.warm_up(warmup.LocalClient(app_module.app))
    assert [name for name, *_ in results] == ["pages", "demo", *warmup.WARMUP]
    assert all(ok for _, ok, _, _ in results), results

    # One registered run per tool, with the demo's settings.
    runs = app_module.run_registry().list()
    assert sorted(r["tool"] for r in runs) == sorted(warmup.WARMUP)
    params = {r["tool"]: r["params"] for r in runs}
    assert params["eq"] == {"preset": "telephone"}
    assert params["reverb"]["rt60"] == 2.5 and params["reverb"]["wet"] == 0.5
    assert (params["echo"]["mode"], params["echo"]["delay_ms"], params["echo"]["gain"]) == ("feedback", 300, 0.5)
    from effects import flanger
    assert params["flanger"] == {name: limits[2] for name, limits in flanger.PARAMS.items()}
    assert params["separate"]["mock"] is False


def test_warmup_reports_a_failing_tool(app_module, warmup, short_demo, monkeypatch, capsys):
    def broken(_audio, _sr):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.separation, "separate", broken)
    monkeypatch.setattr(warmup, "WARMUP", {tool: warmup.WARMUP[tool] for tool in ("echo", "separate")})
    results = {name: (ok, detail) for name, ok, _, detail in warmup.warm_up(warmup.LocalClient(app_module.app))}
    assert results["separate"][0] is False and "boom" in results["separate"][1]
    assert all(ok for name, (ok, _) in results.items() if name != "separate")
    assert "separate  FAIL" in capsys.readouterr().out


def test_warmup_without_a_demo_track_fails_cleanly(app_module, warmup, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DEMO_PATH", tmp_path / "missing.wav")
    results = warmup.warm_up(warmup.LocalClient(app_module.app))
    assert [(name, ok) for name, ok, *_ in results] == [("pages", True), ("demo", False)]


# ---------------------------------------------------------------------
# scripts/clean_runs.py and the run configuration
# ---------------------------------------------------------------------

def test_clean_runs_empties_uploads_and_results(tmp_path):
    clean_runs = load_script("clean_runs")
    uploads, processed = tmp_path / "uploads", tmp_path / "processed"
    uploads.mkdir()
    (uploads / "a.wav").write_bytes(b"x")
    (uploads / "a.json").write_text("{}")
    (tmp_path / "keep.txt").write_text("not in a run folder")

    assert clean_runs.clean((uploads, processed)) == 2
    assert uploads.is_dir() and not list(uploads.iterdir())
    assert processed.is_dir()  # created when missing
    assert (tmp_path / "keep.txt").exists()


def test_debug_is_off_by_default(app_module):
    assert app_module.app.debug is False
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'debug = os.environ.get("SPECTRA_DEBUG") == "1"' in source
    assert (app_module.HOST, app_module.PORT) == ("127.0.0.1", 5000)
    assert not app_module.app.secret_key  # no sessions, so no secret to leak


def test_run_scripts_start_the_app_on_port_5000():
    for script in ("run.sh", "run.bat"):
        text = (ROOT / script).read_text(encoding="utf-8")
        assert "http://127.0.0.1:5000" in text and "--clean" in text and "clean_runs.py" in text
        assert "requirements.txt" in text and ".venv" in text
    # cmd.exe needs CRLF for its labels; .gitattributes keeps it on checkout.
    assert b"\r\n" in (ROOT / "run.bat").read_bytes()
    assert b"\r" not in (ROOT / "run.sh").read_bytes()
