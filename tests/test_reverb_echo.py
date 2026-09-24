"""Tests for effects/reverb.py, effects/echo.py and their routes.

Run with: python -m pytest tests -m "not slow"  (drop -m to include the
performance test). np.convolve and per-sample loops appear here only as
reference implementations.
"""

import ast
import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from effects import echo, reverb

SR = 8000


def noise(shape, seed=0, scale=0.1):
    return np.random.default_rng(seed).standard_normal(shape) * scale


def impulse(length, channels=None):
    x = np.zeros(length if channels is None else (length, channels))
    x[0] = 1.0
    return x


# ---------------------------------------------------------------------
# Reverb: convolution
# ---------------------------------------------------------------------

@pytest.mark.parametrize("len_x, len_h", [(1000, 300), (37, 512)])
def test_fft_convolve_matches_direct_convolution(len_x, len_h):
    x, h = noise(len_x, seed=1), noise(len_h, seed=2)
    y = reverb.fft_convolve(x, h)
    assert len(y) == len_x + len_h - 1
    assert np.allclose(y, np.convolve(x, h), rtol=0, atol=1e-9)


def test_circular_convolve_wraps_the_tail_onto_the_start():
    x, h = noise(1000, seed=1), noise(300, seed=2)
    linear = np.convolve(x, h)
    circular = reverb.circular_convolve(x, h)

    assert len(circular) == len(x)
    assert not np.allclose(circular, linear[:len(x)])

    # Exactly the linear result with the part past len(x) folded back.
    folded = linear[:len(x)].copy()
    folded[:len(h) - 1] += linear[len(x):]
    assert np.allclose(circular, folded, atol=1e-9)


def test_circular_convolve_truncates_long_ir():
    x, h = noise(37, seed=1), noise(512, seed=2)
    assert len(reverb.circular_convolve(x, h)) == 37


# ---------------------------------------------------------------------
# Reverb: impulse response
# ---------------------------------------------------------------------

def test_impulse_response_has_unit_energy_and_60db_decay_at_rt60():
    rt60, pre_delay_ms, seed = 0.5, 25, 3
    h = reverb.generate_impulse_response(SR, rt60, pre_delay_ms, seed=seed)

    n_pre, n_decay = int(pre_delay_ms / 1000 * SR), int(rt60 * SR)
    assert h.dtype == np.float64 and h.ndim == 1
    assert len(h) == n_pre + n_decay
    assert np.sum(h ** 2) == pytest.approx(1.0)
    assert np.all(h[:n_pre] == 0)

    # Check the envelope, not the noise: the noise is reproducible from
    # the seed, so dividing it out leaves the (scaled) envelope exactly.
    env = h[n_pre:] / np.random.default_rng(seed).standard_normal(n_decay)
    env_db = 20 * np.log10(env / env[0])
    t = np.arange(n_decay) / SR
    assert np.allclose(env_db, -60 * t / rt60, atol=0.01)  # a straight line in dB...
    slope_db_per_s = np.polyfit(t, env_db, 1)[0]
    assert slope_db_per_s * rt60 == pytest.approx(-60, abs=0.01)  # ...60 dB down at rt60


def test_impulse_response_is_reproducible_and_capped():
    a = reverb.generate_impulse_response(SR, 1.0, 0, seed=0)
    assert np.array_equal(a, reverb.generate_impulse_response(SR, 1.0, 0, seed=0))
    assert not np.allclose(a, reverb.generate_impulse_response(SR, 1.0, 0, seed=1))
    assert len(reverb.generate_impulse_response(SR, 60.0, 0)) == reverb.MAX_IR_SEC * SR


# ---------------------------------------------------------------------
# Reverb: apply_reverb
# ---------------------------------------------------------------------

def test_impulse_through_reverb_reproduces_the_ir():
    ir = reverb.generate_impulse_response(SR, 0.4, 10, seed=0)
    out = reverb.apply_reverb(impulse(SR), SR, rt60=0.4, pre_delay_ms=10, wet=1.0)

    assert out.shape == (SR + len(ir) - 1,)
    assert np.allclose(out[:len(ir)], ir, atol=1e-12)
    assert np.allclose(out[len(ir):], 0, atol=1e-12)


def test_stereo_channels_get_different_irs():
    out = reverb.apply_reverb(impulse(SR, channels=2), SR, rt60=0.4, pre_delay_ms=10, wet=1.0)
    for ch in range(2):
        ir = reverb.generate_impulse_response(SR, 0.4, 10, seed=ch)
        assert np.allclose(out[:len(ir), ch], ir, atol=1e-12)
    assert not np.allclose(out[:, 0], out[:, 1])


def test_reverb_wet_zero_is_dry_plus_zero_padded_tail():
    x = noise(SR)
    ir_len = len(reverb.generate_impulse_response(SR, 1.0, 20))
    out = reverb.apply_reverb(x, SR, rt60=1.0, pre_delay_ms=20, wet=0.0)

    assert len(out) == len(x) + ir_len - 1
    assert np.allclose(out[:len(x)], x, atol=1e-12)
    assert np.all(out[len(x):] == 0)


def test_circular_reverb_keeps_length_and_smears_tail_onto_start():
    x = np.zeros(SR)
    x[-100] = 1.0  # a click right before the end: its tail has nowhere to go
    kwargs = dict(rt60=0.5, pre_delay_ms=0, wet=1.0)

    linear = reverb.apply_reverb(x, SR, **kwargs)
    circular = reverb.apply_reverb(x, SR, circular=True, **kwargs)

    assert len(circular) == len(x)
    assert np.allclose(linear[:len(x) - 100], 0, atol=1e-12)  # causal: silence before the click
    assert np.sum(circular[:SR // 4] ** 2) > 0.1  # the wrapped tail


def test_reverb_output_never_clips():
    out = reverb.apply_reverb(noise((SR, 2), scale=0.9), SR, wet=1.0)
    assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= 1.0


# ---------------------------------------------------------------------
# Echo: difference equations
# ---------------------------------------------------------------------

def naive_feedback(x, delay, gain):
    """y[n] = x[n] + g * y[n - D], one sample at a time."""
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = x[n] + (gain * y[n - delay] if n >= delay else 0.0)
    return y


def test_feedforward_impulse_gives_one_echo():
    delay_ms, gain = 100, 0.5
    delay = echo.delay_samples(SR, delay_ms)
    out = echo.apply_echo(impulse(SR), SR, delay_ms=delay_ms, gain=gain,
                          mix=1.0, mode="feedforward")

    assert np.flatnonzero(out).tolist() == [0, delay]
    assert out[0] == 1.0
    assert out[delay] == pytest.approx(gain)


def test_feedback_impulse_gives_geometric_echoes():
    delay_ms, gain = 50, 0.6
    delay = echo.delay_samples(SR, delay_ms)
    out = echo.apply_echo(impulse(SR), SR, delay_ms=delay_ms, gain=gain,
                          mix=1.0, mode="feedback")

    for k in range(6):
        assert out[k * delay] == pytest.approx(gain ** k, abs=1e-12)
    assert np.all(np.flatnonzero(out) % delay == 0)


def test_block_feedback_matches_per_sample_loop():
    x = noise(5000, seed=4)  # not a multiple of the delay
    for delay in (123, 1, 4999, 6000):
        assert np.allclose(echo.feedback_comb(x, delay, 0.7), naive_feedback(x, delay, 0.7),
                           rtol=0, atol=1e-12)


def test_gain_is_clamped():
    x = noise(SR)
    too_high = echo.apply_echo(x, SR, gain=5.0, mix=1.0, mode="feedback")
    assert np.array_equal(too_high, echo.apply_echo(x, SR, gain=0.9, mix=1.0, mode="feedback"))
    assert np.all(np.isfinite(too_high)) and np.max(np.abs(too_high)) <= 1.0

    # Negative gain clamps to 0: no echo and no tail.
    assert np.allclose(echo.apply_echo(x, SR, gain=-1.0, mix=1.0), x)


@pytest.mark.parametrize("mode", echo.MODES)
def test_echo_mix_zero_is_dry(mode):
    x = noise((SR, 2))
    out = echo.apply_echo(x, SR, mix=0.0, mode=mode)
    assert out.shape[1] == 2
    assert np.allclose(out[:len(x)], x, atol=1e-12)
    assert np.all(out[len(x):] == 0)


def test_tail_lengths():
    delay = echo.delay_samples(SR, 100)
    x = noise(SR)
    assert len(echo.apply_echo(x, SR, delay_ms=100, mode="feedforward")) == SR + delay
    # 0.5^10 < 0.001 < 0.5^9: ten repeats until 60 dB down
    assert len(echo.apply_echo(x, SR, delay_ms=100, gain=0.5, mode="feedback")) == SR + 10 * delay
    # 0.9 needs 66 repeats of 2 s: capped
    capped = echo.apply_echo(x, SR, delay_ms=2000, gain=0.9, mode="feedback")
    assert len(capped) == SR + echo.MAX_TAIL_SEC * SR
    assert len(echo.apply_echo(x, SR, gain=0.0)) == SR


def test_echo_rejects_unknown_mode():
    with pytest.raises(ValueError):
        echo.apply_echo(noise(100), SR, mode="pingpong")
    with pytest.raises(ValueError):
        echo.echo_frequency_response(SR, 100, 0.5, "pingpong")


# ---------------------------------------------------------------------
# Echo: frequency response
# ---------------------------------------------------------------------

def test_frequency_response_peaks_and_notches():
    delay_ms, g = 10, 0.5
    delay = echo.delay_samples(SR, delay_ms)
    notch_hz = SR / (2 * delay)  # halfway between the peaks at 0 and SR / D

    freqs, ff = echo.echo_frequency_response(SR, delay_ms, g, "feedforward", n_points=4001)
    _, fb = echo.echo_frequency_response(SR, delay_ms, g, "feedback", n_points=4001)
    notch = np.argmin(np.abs(freqs - notch_hz))

    assert len(freqs) == len(ff) == 4001
    assert freqs[0] == 0 and freqs[-1] == pytest.approx(SR / 2)
    assert freqs[notch] == pytest.approx(notch_hz)
    assert ff[0] == pytest.approx(20 * np.log10(1 + g))
    assert ff[notch] == pytest.approx(20 * np.log10(1 - g))
    assert fb[0] == pytest.approx(-20 * np.log10(1 - g))
    assert fb[notch] == pytest.approx(-20 * np.log10(1 + g))


def test_frequency_response_zoom_is_the_same_curve():
    _, full = echo.echo_frequency_response(SR, 350, 0.7, "feedback", n_points=4001)  # 1 Hz steps
    freqs, zoomed = echo.echo_frequency_response(SR, 350, 0.7, "feedback", n_points=101, f_max=100)
    assert np.allclose(freqs, np.arange(101))
    assert np.allclose(zoomed, full[:101])
    clamped, _ = echo.echo_frequency_response(SR, 350, 0.7, "feedback", f_max=1e9)
    assert clamped[-1] == pytest.approx(SR / 2)  # never past Nyquist


@pytest.mark.parametrize("mode", echo.MODES)
def test_frequency_response_matches_dft_of_impulse_response(mode):
    # The formula's grid w = pi * i / (n - 1) is exactly the rfft grid
    # for a 2 * (n - 1) point DFT of the difference equation's output.
    # (Long enough that the truncated feedback IR has decayed to ~g^100.)
    n_points, delay_ms, g = 4097, 10, 0.5
    delay = echo.delay_samples(SR, delay_ms)
    comb = echo.feedforward_comb if mode == "feedforward" else echo.feedback_comb
    ir = comb(impulse(2 * (n_points - 1)), delay, g)

    _, mag_db = echo.echo_frequency_response(SR, delay_ms, g, mode, n_points=n_points)
    assert np.allclose(mag_db, 20 * np.log10(np.abs(np.fft.rfft(ir))), atol=1e-9)


# ---------------------------------------------------------------------
# No black-box effect implementations
# ---------------------------------------------------------------------

FORBIDDEN = {"fftconvolve", "convolve", "lfilter", "scipy.signal", "signal"}


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).resolve().parent.parent / "effects").glob("*.py")),
    ids=lambda p: p.name,
)
def test_effects_use_no_forbidden_calls(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    assert not names & FORBIDDEN


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(app_module, "PROCESSED_DIR", tmp_path)
    return app_module.app.test_client()


def upload(client, audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    res = client.post("/upload", data={"file": (buf, "clip.wav")},
                      content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()
    return res.get_json()["file_id"]


def process(client, tool, file_id, **params):
    res = client.post(f"/process/{tool}", json={"file_id": file_id, **params})
    assert res.status_code == 200, res.get_json()
    audio, sr = sf.read(io.BytesIO(client.get(res.get_json()["url"]).data))
    return audio, sr


def test_reverb_route_linear_keeps_tail_circular_does_not(client):
    file_id = upload(client, noise((SR, 2)), SR)
    ir_len = len(reverb.generate_impulse_response(SR, 0.5, 20))

    linear, sr = process(client, "reverb", file_id, rt60=0.5, pre_delay_ms=20, wet=0.5)
    circular, _ = process(client, "reverb", file_id, rt60="0.5", circular=True)

    assert sr == SR
    assert linear.shape == (SR + ir_len - 1, 2)
    assert circular.shape == (SR, 2)


def test_echo_route_uses_mode_and_delay(client):
    file_id = upload(client, noise(SR), SR)
    out, _ = process(client, "echo", file_id, delay_ms=100, gain=0.5, mode="feedforward")
    assert out.shape == (SR + echo.delay_samples(SR, 100),)


@pytest.mark.parametrize("tool, params", [
    ("reverb", {"rt60": 10}),
    ("reverb", {"wet": -0.1}),
    ("reverb", {"pre_delay_ms": "lots"}),
    ("echo", {"delay_ms": 5}),
    ("echo", {"gain": 0.95}),
    ("echo", {"mode": "pingpong"}),
    ("eq", {"mode": "hum", "hum_freq": 100}),
    ("eq", {"mode": "hum", "harmonics": 2.5}),
])
def test_process_routes_reject_bad_params(client, tool, params):
    file_id = upload(client, noise(SR), SR)
    res = client.post(f"/process/{tool}", json={"file_id": file_id, **params})
    assert res.status_code == 400
    assert "Invalid" in res.get_json()["error"]


@pytest.mark.parametrize("tool, params", [
    ("eq", {"mode": "hum"}),
    ("reverb", {"rt60": 0.5}),
    ("echo", {"delay_ms": 100}),
])
def test_process_routes_return_before_after_spectrograms(client, tool, params):
    file_id = upload(client, noise(SR), SR)
    res = client.post(f"/process/{tool}", json={"file_id": file_id, **params})
    assert res.status_code == 200, res.get_json()
    urls = res.get_json()["spectrograms"]
    for kind in ("before", "after"):
        png = client.get(urls[kind])
        assert png.status_code == 200
        assert png.mimetype == "image/png"


def test_reverb_response_route(client):
    res = client.get("/response/reverb?rt60=5&pre_delay_ms=100")
    assert res.status_code == 200
    data = res.get_json()
    assert set(data) == {"t", "h"}
    assert 0 < len(data["t"]) == len(data["h"]) <= 4000
    assert data["t"][0] == 0 and data["t"][-1] <= 5.1
    assert all(v == 0 for v in data["h"][:10])  # pre-delay


def test_reverb_response_route_uses_defaults(client):
    data = client.get("/response/reverb").get_json()
    assert data["t"][-1] == pytest.approx(1.52, abs=0.01)  # 1.5 s decay + 20 ms pre-delay


@pytest.mark.parametrize("query, f_max", [
    ("delay_ms=350&gain=0.5&mode=feedback", 22050),
    ("delay_ms=350&gain=0.5&mode=feedforward&f_max=200", 200),
    ("delay_ms=2000&gain=0.9&f_max=5", 5),
    ("f_max=99999", 22050),  # clamped to Nyquist
])
def test_echo_response_route(client, query, f_max):
    res = client.get(f"/response/echo?{query}")
    assert res.status_code == 200
    data = res.get_json()
    assert set(data) == {"freqs", "mag_db"}
    assert len(data["freqs"]) == len(data["mag_db"]) == 4000
    assert data["freqs"][0] == 0
    assert data["freqs"][-1] == pytest.approx(f_max, rel=0.01)


@pytest.mark.parametrize("url", [
    "/response/reverb?rt60=0.1",
    "/response/reverb?pre_delay_ms=abc",
    "/response/echo?gain=1.5",
    "/response/echo?mode=pingpong",
    "/response/echo?f_max=-10",
])
def test_response_routes_reject_bad_params(client, url):
    assert client.get(url).status_code == 400


# ---------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def six_minutes_stereo():
    return noise((6 * 60 * 44100, 2)), 44100


@pytest.mark.slow
@pytest.mark.parametrize("effect, params", [
    (reverb.apply_reverb, {"rt60": 5.0, "pre_delay_ms": 100}),     # longest IR
    (echo.apply_echo, {"delay_ms": 20, "gain": 0.9}),              # most feedback blocks
])
def test_six_minutes_of_stereo_takes_under_5_seconds(six_minutes_stereo, effect, params):
    audio, sr = six_minutes_stereo
    start = time.perf_counter()
    out = effect(audio, sr, **params)
    elapsed = time.perf_counter() - start

    assert out.shape[1] == 2
    assert elapsed < 5.0, f"{effect.__name__} took {elapsed:.1f} s"
