"""Tests for Backstage (pivot stage 04b): analysis/backstage.py and the
/backstage routes.

Run with: python -m pytest tests -m "not slow"  (drop -m to include the
6-minute performance test).
"""

import io
import json
import time

import numpy as np
import pytest
import soundfile as sf

import stft
from analysis import backstage
from effects import echo, flanger, reverb, separation

SR = 8000


def noise(shape, seed=0, scale=0.1):
    return np.random.default_rng(seed).standard_normal(shape) * scale


def sine(freq, seconds, sr=SR, amplitude=0.5):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(int(seconds * sr)) / sr)


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


def upload(client, audio, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    res = client.post("/upload", data={"file": (buf, "clip.wav")}, content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()
    return res.get_json()["file_id"]


def run(client, tool, file_id, **params):
    res = client.post(f"/process/{tool}", json={"file_id": file_id, **params})
    assert res.status_code == 200, res.get_json()
    return res.get_json()["run_id"]


def backstage_json(client, run_id):
    res = client.get(f"/backstage/{run_id}")
    assert res.status_code == 200, res.get_json()
    return res.get_json()


# ---------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------

def test_level_stats_of_a_known_sine():
    stats = backstage.level_stats(sine(1000, 2.0, amplitude=0.5), SR)
    assert stats["peak_dbfs"] == pytest.approx(20 * np.log10(0.5), abs=0.01)           # -6.02
    assert stats["rms_dbfs"] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.01)  # -9.03
    assert stats["crest_db"] == pytest.approx(20 * np.log10(np.sqrt(2)), abs=0.02)      # 3.01
    assert stats["duration"] == pytest.approx(2.0)


def test_level_stats_count_every_channel_and_survive_silence():
    stereo = np.stack([sine(500, 1.0, amplitude=0.5), np.zeros(SR)], axis=1)
    stats = backstage.level_stats(stereo, SR)
    assert stats["peak_dbfs"] == pytest.approx(-6.02, abs=0.01)
    assert stats["rms_dbfs"] == pytest.approx(20 * np.log10(0.25), abs=0.01)  # half the samples are 0
    silent = backstage.level_stats(np.zeros(SR), SR)
    assert silent["peak_dbfs"] is None and silent["rms_dbfs"] is None and silent["crest_db"] is None
    json.dumps(silent)  # no -inf in the JSON


def test_chunked_magnitudes_equal_the_one_shot_stft(monkeypatch):
    x = noise(SR * 3, seed=5)
    expected = np.abs(stft.calculatr_stft(x, backstage.FRAME_SIZE, backstage.HOP_SIZE))
    monkeypatch.setattr(backstage, "CHUNK_FRAMES", 7)  # many chunks, last one partial
    mags = backstage.magnitude_frames(x)
    assert mags.shape == expected.shape
    assert np.allclose(mags, expected, rtol=1e-5, atol=1e-5)


def test_difference_is_zero_for_identical_audio_and_gain_for_a_scaled_copy():
    mag = backstage.magnitude_frames(noise(SR * 2))
    diff, stats = backstage.difference_db(mag, mag)
    assert np.all(diff == 0) and stats["max_abs_db"] == 0 and stats["boosted_pct"] == 0

    diff, stats = backstage.difference_db(mag, mag * 2)  # +6.02 dB wherever well above the floor
    loud = mag > mag.max() * 0.1  # 20 dB above the floor ε barely matters
    assert np.allclose(diff[loud], 20 * np.log10(2), atol=0.02)
    diff, _ = backstage.difference_db(mag, mag * 100)  # +40 dB: clipped
    assert diff.max() == backstage.DIFF_CLIP_DB


@pytest.mark.parametrize("circular", [False, True])
def test_reverb_start_comparison_matches_the_real_effect(circular):
    x = noise(SR * 2, seed=2, scale=0.05)
    params = {"rt60": 0.6, "pre_delay_ms": 10, "wet": 0.4}
    linear, wrapped = backstage.reverb_start_comparison(x, SR, params, seconds=1.0)
    full = reverb.apply_reverb(x, SR, circular=circular, **params)  # quiet: no clip scaling
    ours = wrapped if circular else linear
    assert len(ours) == SR
    assert np.allclose(ours, full[:SR], atol=1e-9)


def test_reverb_comparison_works_when_the_ir_is_longer_than_the_track():
    x = noise(SR // 4, seed=3, scale=0.05)  # 0.25 s track, 1.5 s IR
    params = {"rt60": 1.5, "pre_delay_ms": 0, "wet": 1.0}
    _, circular = backstage.reverb_start_comparison(x, SR, params)
    assert np.allclose(circular, reverb.apply_reverb(x, SR, circular=True, **params), atol=1e-9)


# ---------------------------------------------------------------------
# /backstage/<run_id>: JSON shape per tool
# ---------------------------------------------------------------------

EFFECT_KEYS = {"run", "levels", "envelopes", "avg_spectrum", "diff", "stft", "extra", "images", "audio"}


@pytest.mark.parametrize("tool, params, extra_key", [
    ("eq", {"mode": "eq", "band_0_gain_db": 6}, None),
    ("reverb", {"rt60": 0.5}, "reverb_start"),
    ("echo", {"delay_ms": 100, "gain": 0.5}, "echo_ir"),
    ("flanger", {"rate_hz": 1}, "flanger_lfo"),
])
def test_backstage_json_shape(client, tool, params, extra_key):
    audio = noise((SR, 2))
    data = backstage_json(client, run(client, tool, upload(client, audio), **params))
    assert set(data) == EFFECT_KEYS
    assert data["run"]["tool"] == tool

    for side in ("before", "after"):
        levels = data["levels"][side]
        assert set(levels) == {"peak_dbfs", "rms_dbfs", "crest_db", "duration"}
        env = data["envelopes"][side]
        assert len(env["max"]) == len(env["min"]) == backstage.ENVELOPE_POINTS
        assert np.all(np.array(env["max"]) >= np.array(env["min"]))
    assert data["levels"]["before"]["duration"] == pytest.approx(1.0)
    assert data["levels"]["after"]["duration"] >= 1.0

    spectrum = data["avg_spectrum"]
    assert len(spectrum["freqs"]) == len(spectrum["before_db"]) == len(spectrum["after_db"]) \
        == backstage.FRAME_SIZE // 2
    assert spectrum["freqs"][0] > 0  # no DC on a log axis

    assert data["diff"]["clip_db"] == backstage.DIFF_CLIP_DB
    assert data["diff"]["seconds"] == pytest.approx(1.0)
    analysis = data["stft"]["analysis"]
    assert analysis["frame_size"] == 1024 and analysis["sr"] == SR
    assert analysis["df_hz"] == pytest.approx(SR / 1024)
    assert analysis["dt_ms"] == pytest.approx(1024 / SR * 1000)
    assert ("processing" in data["stft"]) == (tool == "eq")
    if tool == "eq":
        assert data["stft"]["processing"]["frame_size"] == 65536
    assert list(data["extra"]) == ([extra_key] if extra_key else [])

    for kind in ("before", "after", "diff"):
        image = client.get(data["images"][kind])
        assert image.status_code == 200 and image.mimetype == "image/png"
    assert client.get(data["audio"]["input"]).status_code == 200
    assert client.get(data["audio"]["output"]["url"]).status_code == 200


@pytest.mark.parametrize("tool, params", [
    ("echo", {"gain": 0}),
    ("reverb", {"wet": 0}),
])
def test_difference_is_about_0_db_when_the_effect_does_nothing(client, tool, params):
    data = backstage_json(client, run(client, tool, upload(client, noise((SR, 2))), **params))
    assert data["diff"]["max_abs_db"] < 0.01
    assert data["diff"]["boosted_pct"] == data["diff"]["cut_pct"] == 0
    before, after = data["levels"]["before"], data["levels"]["after"]
    assert after["peak_dbfs"] == pytest.approx(before["peak_dbfs"], abs=0.01)
    assert np.allclose(data["avg_spectrum"]["before_db"], data["avg_spectrum"]["after_db"], atol=0.01)


def test_a_boost_shows_up_in_the_levels_and_the_difference(client):
    tone = sine(1000, 1.0, amplitude=0.1)
    data = backstage_json(client, run(client, "echo", upload(client, tone), delay_ms=20, gain=0.9, mode="feedback"))
    assert data["levels"]["after"]["rms_dbfs"] > data["levels"]["before"]["rms_dbfs"]
    assert data["diff"]["boosted_pct"] > 0
    assert data["diff"]["output_longer_s"] > 0  # the feedback tail


def test_echo_impulse_response_is_g_to_the_k_at_k_times_d(client):
    file_id = upload(client, noise(SR))
    ir = backstage_json(client, run(client, "echo", file_id, delay_ms=100, gain=0.5, mode="feedback"))["extra"]["echo_ir"]
    delay = echo.delay_samples(SR, 100)
    k = np.arange(len(ir["t"]))
    assert np.allclose(ir["t"], k * delay / SR)
    assert np.allclose(ir["h"], 0.5 ** k, atol=1e-5)  # sent rounded to 5 decimals
    assert 0.5 ** k[-1] <= 0.002  # runs until the echoes are ~60 dB down

    ff = backstage_json(client, run(client, "echo", file_id, delay_ms=100, gain=0.5, mode="feedforward"))
    assert ff["extra"]["echo_ir"]["h"] == [1.0, 0.5]


def test_flanger_lfo_sweeps_between_min_and_min_plus_sweep(client):
    lfo = backstage_json(client, run(client, "flanger", upload(client, noise(SR * 2)),
                                     min_delay_ms=2, sweep_ms=4, rate_hz=2))["extra"]["flanger_lfo"]
    delays = np.array(lfo["delay_ms"])
    assert delays.min() == pytest.approx(2, abs=0.01) and delays.max() == pytest.approx(6, abs=0.01)
    assert np.allclose(delays, flanger.sweep_delay_ms(2 * np.array(lfo["t"]), 2, 4), atol=1e-3)


def test_reverb_extra_reports_the_run_mode(client):
    data = backstage_json(client, run(client, "reverb", upload(client, noise(SR)), rt60=0.3, circular="true"))
    start = data["extra"]["reverb_start"]
    assert start["run_circular"] is True
    assert len(start["linear"]["max"]) == len(start["circular"]["max"])


# ---------------------------------------------------------------------
# Separation runs
# ---------------------------------------------------------------------

def test_separation_run_shows_its_stems(client, monkeypatch):
    monkeypatch.setenv("SEPARATION_MOCK", "1")
    file_id = upload(client, noise((SR, 2)))
    res = client.post("/process/separate", json={"file_id": file_id})
    data = backstage_json(client, res.get_json()["run_id"])

    assert [s["name"] for s in data["stems"]] == list(separation.STEMS)
    for stem in data["stems"]:
        assert stem["levels"] == data["levels"]["mixture"]  # mock stems are copies
        assert len(stem["envelope"]["max"]) == backstage.ENVELOPE_POINTS
        mask = client.get(data["images"]["masks"][stem["name"]])
        assert mask.status_code == 200 and mask.mimetype == "image/png"
        assert client.get(data["audio"]["stems"][stem["name"]]["url"]).status_code == 200
    assert client.get(data["images"]["mixture"]).status_code == 200

    run_id = data["run"]["run_id"]
    assert client.get(f"/backstage/{run_id}/mask/guitar.png").status_code == 404
    assert client.get(f"/backstage/{run_id}/diff.png").status_code == 404


def test_real_separation_run_shows_its_stems(client):
    mix = sine(110, 1.0) + noise(SR, scale=0.05)
    data = backstage_json(client, run(client, "separate", upload(client, mix)))

    assert [s["name"] for s in data["stems"]] == list(separation.STEMS)
    assert data["run"]["params"]["mock"] is False
    assert data["levels"]["mixture"]["duration"] == pytest.approx(1.0)
    for stem in data["stems"]:
        assert stem["levels"]["duration"] == pytest.approx(1.0)
        mask = client.get(data["images"]["masks"][stem["name"]])
        assert mask.status_code == 200 and mask.data.startswith(b"\x89PNG")
    assert client.get(data["images"]["mixture"]).status_code == 200


# ---------------------------------------------------------------------
# Run list, caching, errors
# ---------------------------------------------------------------------

def test_runs_are_listed_newest_first(client):
    file_id = upload(client, noise(SR))
    first = run(client, "echo", file_id, gain=0.2)
    second = run(client, "reverb", file_id, rt60=0.3)
    runs = client.get("/backstage/runs").get_json()["runs"]
    assert [r["run_id"] for r in runs] == [second, first]
    assert runs[0]["tool"] == "reverb" and runs[0]["params"]["rt60"] == 0.3


def test_analysis_and_images_are_cached(client, app_module, monkeypatch):
    run_id = run(client, "echo", upload(client, noise(SR)), gain=0.5)
    first = backstage_json(client, run_id)
    diff_png = client.get(first["images"]["diff"]).data

    def fail(*_args, **_kwargs):
        raise AssertionError("recomputed instead of using the cache")

    monkeypatch.setattr(backstage, "analyze_effect", fail)
    monkeypatch.setattr(backstage, "render_diff", fail)
    assert backstage_json(client, run_id) == first
    assert client.get(first["images"]["diff"]).data == diff_png


def test_diff_image_can_be_requested_before_the_json(client):
    run_id = run(client, "flanger", upload(client, noise(SR)))
    image = client.get(f"/backstage/{run_id}/diff.png")
    assert image.status_code == 200 and image.data.startswith(b"\x89PNG")


def test_unknown_runs_and_missing_inputs_are_404(client, app_module):
    assert client.get("/backstage/not-a-run").status_code == 404
    assert client.get("/backstage/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/backstage/00000000-0000-0000-0000-000000000000/diff.png").status_code == 404

    file_id = upload(client, noise(SR))
    run_id = run(client, "echo", file_id)
    for path in app_module.UPLOAD_DIR.glob(f"{file_id}.*"):
        path.unlink()
    res = client.get(f"/backstage/{run_id}")
    assert res.status_code == 404
    assert "input" in res.get_json()["error"]


# ---------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------

@pytest.mark.slow
def test_six_minutes_of_stereo_analyzes_in_about_3_seconds(tmp_path):
    sr = 44100
    audio = noise((6 * 60 * sr, 2), scale=0.2).astype(np.float32)
    src, out = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(str(src), audio, sr, subtype="PCM_16")
    sf.write(str(out), echo.apply_echo(audio, sr, delay_ms=350, gain=0.5), sr, subtype="FLOAT")
    run_record = {"tool": "echo", "params": {"delay_ms": 350, "gain": 0.5, "mix": 0.5, "mode": "feedback"}}

    start = time.perf_counter()
    data = backstage.analyze_effect(run_record, src, out, tmp_path / "diff.png")
    elapsed = time.perf_counter() - start

    assert (tmp_path / "diff.png").exists()
    assert len(json.dumps(data)) < 500_000  # everything sent for plotting is downsampled
    assert elapsed < 3.5, f"analysis took {elapsed:.1f} s"
