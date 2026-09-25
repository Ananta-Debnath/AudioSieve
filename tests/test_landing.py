"""Tests for the landing page (pivot stage 05): routes, the preview list
it embeds, and the clip assembly in scripts/make_landing_assets.py.

Run with: python -m pytest tests -m "not slow"
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app_module():
    import app as app_module
    return app_module


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


def landing_data(client):
    html = client.get("/").get_data(as_text=True)
    start = html.index('<script id="landing-data" type="application/json">')
    start = html.index(">", start) + 1
    return json.loads(html[start:html.index("</script>", start)])


def test_landing_is_at_the_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "SPECTRA" in res.get_data(as_text=True)


def test_the_app_is_at_lab(client):
    res = client.get("/lab")
    assert res.status_code == 200
    assert 'id="spectra-config"' in res.get_data(as_text=True)


def test_landing_links_into_the_lab(client):
    assert landing_data(client)["lab_url"] == "/lab"
    assert 'href="/lab"' in client.get("/").get_data(as_text=True)


def test_every_listed_preview_is_served(client):
    previews = landing_data(client)["previews"]
    assert set(previews) <= {"eq", "reverb", "echo", "flanger", "separation"}
    for tool, preview in previews.items():
        assert preview["tool"] == tool
        assert 0 < preview["switch_at_s"] < preview["duration_s"]
        for key in ("clip_url", "background_url"):
            assert client.get(preview[key]).status_code == 200, (tool, key)


def test_hero_spectrum_is_embedded(client):
    hero = landing_data(client)["hero"]
    if hero is None:
        pytest.skip("static/landing/hero.json not generated")
    frames = np.array(hero["frames"])
    assert frames.ndim == 2 and len(frames) > 1
    assert frames.min() >= 0 and frames.max() == 100
    assert hero["frame_s"] > 0 and hero["hz"] == [20, 20000]


def test_missing_assets_leave_the_page_working(client, app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "LANDING_DIR", tmp_path)
    assert client.get("/").status_code == 200
    data = landing_data(client)
    assert data["previews"] == {} and data["hero"] is None


# ---------------------------------------------------------------------
# scripts/make_landing_assets.py
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def assets():
    spec = importlib.util.spec_from_file_location(
        "make_landing_assets", ROOT / "scripts" / "make_landing_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clip_is_original_gap_processed_at_matched_loudness(assets):
    sr = 8000
    rng = np.random.default_rng(0)
    original = rng.standard_normal(sr) * 0.1
    processed = original * 0.05  # much quieter, like the telephone preset
    clip, switch_at = assets.make_clip(original, processed, sr)

    gap = int(assets.GAP_SEC * sr)
    assert len(clip) == 2 * sr + gap
    assert switch_at == pytest.approx((sr + gap) / sr)
    assert np.all(clip[sr:sr + gap] == 0)
    a, b = clip[:sr], clip[sr + gap:]
    assert assets.rms(b) == pytest.approx(assets.rms(a), rel=1e-6)


def test_clip_stays_under_full_scale(assets):
    sr = 8000
    original = np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.9
    spiky = np.zeros(sr)
    spiky[::400] = 1.0  # same RMS as the original needs peaks far above 1
    clip, _ = assets.make_clip(original, spiky, sr)
    assert np.max(np.abs(clip)) <= assets.PEAK + 1e-9


def test_hero_spectrum_peaks_at_the_tone(assets):
    sr = 44100
    t = np.arange(3 * sr) / sr
    hero = assets.hero_spectrum(0.5 * np.sin(2 * np.pi * 1000 * t), sr)
    frames = np.array(hero["frames"])
    freqs = np.geomspace(*assets.HERO_HZ, assets.HERO_POINTS)
    assert frames.shape[1] == assets.HERO_POINTS
    assert len(frames) == pytest.approx(3 / hero["frame_s"], abs=1)
    peaks = freqs[frames.argmax(axis=1)]
    assert np.all(np.abs(np.log2(peaks / 1000)) < 0.2)  # within a fifth of an octave
    assert frames[:, 0].max() == 0  # nothing at 20 Hz


def test_loudest_start_finds_the_loud_part(assets):
    sr = 1000
    audio = np.full(10 * sr, 0.01)
    audio[6 * sr:7 * sr] = 0.5
    assert assets.loudest_start(audio, sr, 1.0) == pytest.approx(6.0, abs=assets.RMS_HOP_SEC)
