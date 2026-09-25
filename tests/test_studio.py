"""Tests for the SPECTRA studio backend (pivot stage 04a): /response/eq,
uploads and the demo track, run ids and the run registry, and the
separation mock.

Run with: python -m pytest tests -m "not slow"
"""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import runs
import ui_config
from effects import echo, eq_filter, flanger, reverb, separation

SR = 8000
ROOT = Path(__file__).resolve().parent.parent


def noise(shape, seed=0, scale=0.1):
    return np.random.default_rng(seed).standard_normal(shape) * scale


def wav_bytes(audio, sr, subtype="FLOAT"):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


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
def client(app_module):
    return app_module.app.test_client()


def upload(client, audio, sr, name="clip.wav"):
    res = client.post("/upload", data={"file": (io.BytesIO(wav_bytes(audio, sr)), name)},
                      content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()
    return res.get_json()


def eq_response(client, query=""):
    res = client.get(f"/response/eq?{query}")
    assert res.status_code == 200, res.get_json()
    data = res.get_json()
    return np.array(data["freqs"]), np.array(data["gain_db"])


# ---------------------------------------------------------------------
# /response/eq
# ---------------------------------------------------------------------

def test_eq_response_is_a_log_grid_over_the_audible_range(client):
    freqs, gain_db = eq_response(client)
    assert len(freqs) == len(gain_db) == 1000
    assert freqs[0] == pytest.approx(20) and freqs[-1] == pytest.approx(20000)
    assert np.allclose(freqs, np.geomspace(20, 20000, 1000), atol=0.005)  # evenly spaced in log
    assert np.all(gain_db == 0)  # default: flat EQ


def test_eq_response_lowpass_is_the_butterworth_curve(client):
    freqs, gain_db = eq_response(client, "mode=filter&filter_type=lowpass&cutoff=1000&order=4")
    expected = eq_filter.create_filter_curve(freqs, "lowpass", cutoff=1000, order=4)
    assert np.allclose(gain_db, 20 * np.log10(np.maximum(expected, 1e-6)), atol=0.01)
    assert np.interp(np.log(1000), np.log(freqs), gain_db) == pytest.approx(-3.01, abs=0.05)


def test_eq_response_uses_the_band_gains(client):
    freqs, gain_db = eq_response(client, "mode=eq&band_0_gain_db=8&band_4_gain_db=-6")
    assert gain_db[0] == pytest.approx(8, abs=0.1)     # bass shelf at 20 Hz
    assert gain_db[-1] == pytest.approx(-6, abs=0.5)   # treble shelf at 20 kHz
    assert abs(np.interp(1000, freqs, gain_db)) < 0.5  # untouched middle


@pytest.mark.parametrize("preset", [p for p in eq_filter.PRESETS if p != "remove_hum_50hz"])
def test_eq_response_matches_the_processing_curve(client, preset):
    # The same curve the audio gets (build_curve on the 65536-point FFT
    # bins), just evaluated on the plot's grid.
    freqs, gain_db = eq_response(client, f"preset={preset}")
    bins, curve = eq_filter.build_curve(44100, {"preset": preset})
    assert np.allclose(10 ** (gain_db / 20), np.interp(freqs, bins, curve), atol=2e-3)


def test_eq_response_shows_the_full_depth_of_narrow_hum_notches(client):
    freqs, gain_db = eq_response(client, "mode=hum&hum_freq=60&harmonics=10&notch_width=1")
    for k in range(1, 11):
        at = np.flatnonzero(np.isclose(freqs, 60 * k))
        assert len(at) == 1 and gain_db[at[0]] <= -100  # the grid hits every notch centre
    assert abs(np.interp(90, freqs, gain_db)) < 0.1     # and leaves the rest alone


@pytest.mark.parametrize("query", [
    "mode=bogus",
    "preset=nope",
    "mode=filter&filter_type=lowpass&cutoff=30000",       # above Nyquist
    "mode=filter&filter_type=bandpass&low_cutoff=3000&high_cutoff=300",
    "mode=hum&harmonics=2.5",
    "mode=hum&hum_freq=100",                                # outside HUM_PARAMS
])
def test_eq_response_rejects_bad_params(client, query):
    res = client.get(f"/response/eq?{query}")
    assert res.status_code == 400
    assert "Invalid" in res.get_json()["error"]


# ---------------------------------------------------------------------
# Uploads and the demo track
# ---------------------------------------------------------------------

def test_upload_returns_metadata_and_serves_the_file(client):
    audio = noise((SR // 2, 2))
    meta = upload(client, audio, SR)
    assert meta["duration"] == pytest.approx(0.5)
    assert meta["samplerate"] == SR and meta["channels"] == 2
    assert meta["filename"] == "clip.wav"

    served = client.get(meta["url"])
    assert served.status_code == 200
    assert served.data == wav_bytes(audio, SR)

    wave = client.get(f"{meta['url']}/waveform").get_json()
    assert wave["duration"] == pytest.approx(0.5)
    high, low = (np.array(p) for p in wave["peaks"])
    assert len(high) == len(low) == 2000
    assert high.max() == pytest.approx(audio.max(), abs=1e-4)
    assert low.min() == pytest.approx(audio.min(), abs=1e-4)


def test_unknown_upload_is_404(client):
    assert client.get("/upload/not-an-id").status_code == 404
    assert client.get("/upload/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/upload/00000000-0000-0000-0000-000000000000/waveform").status_code == 404


def test_demo_upload_goes_through_the_normal_upload_path(client, app_module, tmp_path, monkeypatch):
    demo = tmp_path / "demo.wav"
    sf.write(str(demo), noise(SR), SR)
    monkeypatch.setattr(app_module, "DEMO_PATH", demo)

    res = client.post("/upload/demo")
    assert res.status_code == 200, res.get_json()
    meta = res.get_json()
    assert meta["filename"] == "demo.wav"
    assert meta["duration"] == pytest.approx(1.0) and meta["samplerate"] == SR

    # An ordinary file_id: any tool can process it.
    run = client.post("/process/echo", json={"file_id": meta["file_id"], "gain": 0})
    assert run.status_code == 200, run.get_json()


def test_missing_demo_track_is_a_clear_404(client, app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DEMO_PATH", tmp_path / "missing.wav")
    res = client.post("/upload/demo")
    assert res.status_code == 404
    assert "make_placeholder_demo" in res.get_json()["error"]


def test_placeholder_demo_script(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "make_placeholder_demo", ROOT / "scripts" / "make_placeholder_demo.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    audio = script.make_demo()
    assert len(audio) == 20 * 44100
    assert np.max(np.abs(audio)) == pytest.approx(0.8)

    out = tmp_path / "demo" / "demo.wav"
    monkeypatch.setattr(script, "OUT_PATH", out)
    assert script.main([]) == 0 and sf.info(str(out)).duration == pytest.approx(20)
    assert script.main([]) == 1  # never overwrites the real track by accident
    assert script.main(["--force"]) == 0


# ---------------------------------------------------------------------
# Run ids and the run registry
# ---------------------------------------------------------------------

@pytest.mark.parametrize("tool, params", [
    ("eq", {"mode": "filter", "filter_type": "highpass", "cutoff": 200}),
    ("reverb", {"rt60": 0.5, "circular": "true"}),
    ("echo", {"delay_ms": 100, "mode": "feedforward"}),
    ("flanger", {"sweep_ms": 2}),
])
def test_every_run_returns_a_run_id_and_is_registered(client, app_module, tool, params):
    audio = noise(SR)
    meta = upload(client, audio, SR)
    res = client.post(f"/process/{tool}", json={"file_id": meta["file_id"], **params})
    assert res.status_code == 200, res.get_json()
    data = res.get_json()

    # Additions only: the old fields are still there.
    assert {"result_id", "url", "download_url", "spectrograms"} <= set(data)
    assert data["run_id"] == data["result_id"]
    assert client.get(data["input_url"]).status_code == 200
    before, after = data["waveforms"]["before"], data["waveforms"]["after"]
    assert before["duration"] == pytest.approx(1.0) and after["duration"] >= 1.0
    assert all(len(p) == 2000 for p in before["peaks"] + after["peaks"])

    run = app_module.run_registry().get(data["run_id"])
    assert run["tool"] == tool and run["file_id"] == meta["file_id"]
    assert run["source"]["filename"] == "clip.wav"
    assert run["outputs"]["audio"] == f"{data['run_id']}.wav"
    assert (app_module.PROCESSED_DIR / run["outputs"]["audio"]).exists()
    assert set(run["outputs"]["spectrograms"]) == {"before", "after"}
    for key, value in params.items():
        if key in run["params"] and key != "circular":
            assert run["params"][key] == value
    if tool == "reverb":
        assert run["params"]["circular"] is True

    # The sidecar lets a fresh registry (e.g. after a restart) find it.
    sidecar = app_module.PROCESSED_DIR / f"{data['run_id']}{runs.SIDECAR_SUFFIX}"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == run
    assert runs.RunRegistry(app_module.PROCESSED_DIR).get(data["run_id"]) == run


def test_registry_lists_newest_first_and_skips_broken_sidecars(tmp_path, monkeypatch):
    registry = runs.RunRegistry(tmp_path)
    clock = iter([100.0, 200.0, 300.0])
    monkeypatch.setattr(runs.time, "time", lambda: next(clock))
    for run_id in ("a", "b", "c"):
        registry.add(run_id, "echo", {"gain": 0.5}, "file", {"audio": f"{run_id}.wav"})
    (tmp_path / f"broken{runs.SIDECAR_SUFFIX}").write_text("{not json", encoding="utf-8")

    assert [r["run_id"] for r in registry.list()] == ["c", "b", "a"]
    reloaded = runs.RunRegistry(tmp_path)
    assert [r["run_id"] for r in reloaded.list()] == ["c", "b", "a"]
    assert reloaded.get("b")["created"] == 200.0
    assert reloaded.get("missing") is None


# ---------------------------------------------------------------------
# Separation: the real module by default, fake stems with the mock
# ---------------------------------------------------------------------

def test_separation_returns_the_contract(client, app_module):
    audio = noise((SR, 2))
    meta = upload(client, audio, SR)

    res = client.post("/process/separate", json={"file_id": meta["file_id"]})
    assert res.status_code == 200, res.get_json()
    data = res.get_json()
    assert {"run_id", "stems"} <= set(data)
    assert data["truncated"] is False and data["note"] is None
    assert [s["name"] for s in data["stems"]] == list(separation.STEMS)
    total = 0
    for stem in data["stems"]:
        assert {"name", "url", "download_url"} <= set(stem)
        stem_audio, sr = sf.read(io.BytesIO(client.get(stem["url"]).data))
        assert sr == SR and stem_audio.shape == audio.shape
        assert np.all(np.isfinite(stem_audio))
        total = total + stem_audio
        download = client.get(stem["download_url"])
        assert download.headers["Content-Disposition"].startswith("attachment")
    assert np.allclose(total, audio, atol=1e-5)  # the stems add up to the mix

    run = app_module.run_registry().get(data["run_id"])
    assert run["tool"] == "separate" and run["file_id"] == meta["file_id"]
    assert run["params"]["mock"] is False
    assert [s["name"] for s in run["outputs"]["stems"]] == list(separation.STEMS)


def test_separation_mock_returns_the_contract(client, app_module, monkeypatch):
    monkeypatch.setenv("SEPARATION_MOCK", "1")
    audio = noise((SR, 2))
    meta = upload(client, audio, SR)

    res = client.post("/process/separate", json={"file_id": meta["file_id"]})
    assert res.status_code == 200, res.get_json()
    data = res.get_json()
    assert [s["name"] for s in data["stems"]] == list(separation.STEMS)
    for stem in data["stems"]:
        assert {"name", "url", "download_url"} <= set(stem)
        stem_audio, sr = sf.read(io.BytesIO(client.get(stem["url"]).data))
        assert sr == SR and np.allclose(stem_audio, audio, atol=1e-6)  # a copy of the input
        download = client.get(stem["download_url"])
        assert download.headers["Content-Disposition"].startswith("attachment")

    run = app_module.run_registry().get(data["run_id"])
    assert run["tool"] == "separate" and run["file_id"] == meta["file_id"]
    assert run["params"]["mock"] is True
    assert [s["name"] for s in run["outputs"]["stems"]] == list(separation.STEMS)


def test_long_tracks_are_separated_up_to_the_cap(client, app_module, monkeypatch):
    monkeypatch.setenv("SEPARATION_MOCK", "1")
    monkeypatch.setattr(app_module, "SEPARATION_MAX_SECONDS", 0.25)
    meta = upload(client, noise(SR), SR)

    data = client.post("/process/separate", json={"file_id": meta["file_id"]}).get_json()
    assert data["truncated"] is True
    assert data["analysed_seconds"] == pytest.approx(0.25) and data["input_seconds"] == pytest.approx(1.0)
    assert data["note"] == "Separation analyses the first 0.25 s of the track."
    for stem in data["stems"]:
        assert sf.info(io.BytesIO(client.get(stem["url"]).data)).duration == pytest.approx(0.25)
        assert stem["duration"] == pytest.approx(0.25)

    # The page gets the cap, to warn before RUN.
    assert page_config(client)["limits"]["separation_max_sec"] == 0.25
    # 0 turns the cap off.
    monkeypatch.setattr(app_module, "SEPARATION_MAX_SECONDS", 0)
    data = client.post("/process/separate", json={"file_id": meta["file_id"]}).get_json()
    assert data["truncated"] is False and data["analysed_seconds"] == pytest.approx(1.0)
    assert page_config(client)["limits"]["separation_max_sec"] is None


def test_separation_cap_defaults_to_60_seconds(monkeypatch):
    import app as app_module

    assert app_module._env_seconds("SEPARATION_MAX_SECONDS_UNSET_FOR_TEST", 60) == 60
    monkeypatch.setenv("SEPARATION_MAX_SECONDS", "30")
    assert app_module._env_seconds("SEPARATION_MAX_SECONDS", 60) == 30
    monkeypatch.setenv("SEPARATION_MAX_SECONDS", "lots")
    assert app_module._env_seconds("SEPARATION_MAX_SECONDS", 60) == 60


def test_a_server_error_is_json_with_the_reason(client, app_module, monkeypatch):
    def broken(_audio, _sr):
        raise MemoryError("out of memory")

    monkeypatch.setattr(app_module.separation, "separate", broken)
    meta = upload(client, noise(SR), SR)
    res = client.post("/process/separate", json={"file_id": meta["file_id"]})
    assert res.status_code == 500
    assert res.get_json()["error"] == "Server error: MemoryError: out of memory"


# ---------------------------------------------------------------------
# MP3 needs a libsndfile with MP3 support; WAV and FLAC always work
# ---------------------------------------------------------------------

def test_mp3_without_decoder_is_a_clear_error(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MP3_SUPPORTED", False)
    res = client.post("/upload", data={"file": (io.BytesIO(b"ID3 not really mp3"), "song.mp3")},
                      content_type="multipart/form-data")
    assert res.status_code == 415
    assert "MP3 is not supported" in res.get_json()["error"]
    assert "WAV or FLAC" in res.get_json()["error"]
    assert not list(app_module.UPLOAD_DIR.iterdir())  # nothing was saved

    # The other formats still work.
    assert upload(client, noise(SR), SR)["samplerate"] == SR
    buf = io.BytesIO()
    sf.write(buf, noise(SR), SR, format="FLAC")
    res = client.post("/upload", data={"file": (io.BytesIO(buf.getvalue()), "clip.flac")},
                      content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()


@pytest.mark.skipif("MP3" not in sf.available_formats(), reason="this libsndfile has no MP3 support")
def test_mp3_upload_works_without_ffmpeg(client, app_module):
    assert app_module.MP3_SUPPORTED
    buf = io.BytesIO()
    sf.write(buf, noise(44100), 44100, format="MP3", subtype="MPEG_LAYER_III")
    res = client.post("/upload", data={"file": (io.BytesIO(buf.getvalue()), "clip.mp3")},
                      content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()
    run = client.post("/process/echo", json={"file_id": res.get_json()["file_id"]})
    assert run.status_code == 200, run.get_json()


def test_separation_mock_needs_a_known_file(client, monkeypatch):
    monkeypatch.setenv("SEPARATION_MOCK", "1")
    res = client.post("/process/separate", json={"file_id": "nope"})
    assert res.status_code == 404


# ---------------------------------------------------------------------
# The page: slider ranges come from the effects' PARAMS
# ---------------------------------------------------------------------

def page_config(client):
    html = client.get("/lab").get_data(as_text=True)
    assert "<title>SPECTRA</title>" in html
    start = html.index('<script id="spectra-config" type="application/json">')
    start = html.index(">", start) + 1
    return json.loads(html[start:html.index("</script>", start)])


def test_page_sliders_come_from_params(client):
    config = page_config(client)
    for tool, module in (("reverb", reverb), ("echo", echo), ("flanger", flanger)):
        for s in config[tool]["sliders"]:
            assert (s["min"], s["max"], s["default"]) == tuple(module.PARAMS[s["name"]])
    for s in config["eq"]["hum"]:
        assert (s["min"], s["max"], s["default"]) == tuple(eq_filter.HUM_PARAMS[s["name"]])
    for s in config["eq"]["bands"]:
        assert (s["min"], s["max"]) == (-eq_filter.MAX_BAND_GAIN_DB, eq_filter.MAX_BAND_GAIN_DB)
    assert [p["name"] for p in config["eq"]["presets"]] == list(eq_filter.PRESETS)
    assert [t["id"] for t in config["tools"]] == [t["id"] for t in ui_config.TOOLS]


def test_preset_slider_values_reproduce_the_preset(client):
    # Running with the values a preset puts on the sliders gives the
    # preset's own curve.
    for preset in page_config(client)["eq"]["presets"]:
        query = "&".join(f"{k}={v}" for k, v in preset["values"].items())
        _, from_sliders = eq_response(client, query)
        _, from_preset = eq_response(client, f"preset={preset['name']}")
        assert np.allclose(from_sliders, from_preset), preset["name"]
