"""Unit tests for effects/eq_filter.py. Run with: python -m pytest tests"""

import numpy as np
import pytest

from effects import eq_filter

SR = 44100
FREQS = np.fft.rfftfreq(eq_filter.FRAME_SIZE, d=1 / SR)
MINUS_3_DB = 1 / np.sqrt(2)


def db(gain):
    return 20 * np.log10(gain)


# ---------------------------------------------------------------------
# Filter curves
# ---------------------------------------------------------------------

@pytest.mark.parametrize("filter_type", ["lowpass", "highpass"])
@pytest.mark.parametrize("order", [1, 2, 4, 8])
def test_lowpass_highpass_are_minus_3db_at_cutoff(filter_type, order):
    gain = eq_filter.create_filter_curve([1000.0], filter_type, cutoff=1000, order=order)
    assert gain[0] == pytest.approx(MINUS_3_DB)


def test_lowpass_shape():
    gain = eq_filter.create_filter_curve(FREQS, "lowpass", cutoff=1000, order=4)
    assert gain[0] == pytest.approx(1.0)
    assert np.all(np.diff(gain) <= 0)  # monotonically falling
    assert np.all((gain >= 0) & (gain <= 1))


def test_highpass_dc_is_zero_and_shape():
    gain = eq_filter.create_filter_curve(FREQS, "highpass", cutoff=1000, order=4)
    assert gain[0] == 0
    assert np.all(np.diff(gain) >= 0)  # monotonically rising
    assert gain[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_rolloff_is_6db_per_octave_per_order(order):
    # Far into the stopband, doubling the distance loses 6n dB.
    gain = eq_filter.create_filter_curve([8000.0, 16000.0], "lowpass", cutoff=100, order=order)
    assert db(gain[0]) - db(gain[1]) == pytest.approx(6.02 * order, abs=0.05)


def test_bandpass_passes_middle_rejects_edges():
    gain = eq_filter.create_filter_curve(
        [0.0, 50.0, 1000.0, 20000.0], "bandpass", low_cutoff=300, high_cutoff=3000, order=4
    )
    dc, low, mid, high = gain
    assert dc == 0
    assert mid > 0.99
    assert low < 0.01 and high < 0.01


def test_bandstop_is_minus_3db_at_edges_and_zero_at_center():
    low, high = 800.0, 1200.0
    center = np.sqrt(low * high)
    gain = eq_filter.create_filter_curve(
        [0.0, low, center, high, 20000.0], "bandstop", low_cutoff=low, high_cutoff=high, order=2
    )
    assert gain[0] == pytest.approx(1.0)
    assert gain[1] == pytest.approx(MINUS_3_DB)
    assert gain[2] == pytest.approx(0, abs=1e-12)
    assert gain[3] == pytest.approx(MINUS_3_DB)
    assert gain[4] > 0.999


def test_hum_curve_notches_each_harmonic_only():
    harmonics = [50.0 * k for k in range(1, 6)]
    gain = eq_filter.create_hum_curve(harmonics + [75.0, 300.0, 1000.0], 50, 5, 4)
    assert np.allclose(gain[:5], 0, atol=1e-12)
    assert np.all(gain[5:] > 0.99)  # between harmonics, and past the last one


def test_hum_notch_is_minus_3db_at_its_edges():
    centre, width = 50.0, 4.0
    low = np.sqrt(centre ** 2 + width ** 2 / 4) - width / 2
    gain = eq_filter.create_hum_curve([low, centre, low + width], centre, 1, width)
    assert gain[0] == pytest.approx(MINUS_3_DB)
    assert gain[1] == pytest.approx(0, abs=1e-12)
    assert gain[2] == pytest.approx(MINUS_3_DB)


@pytest.mark.parametrize("hum_freq, harmonics, notch_width", [
    (0, 5, 4), (50, 0, 4), (50, 2.5, 4), (50, 5, 0),
])
def test_hum_curve_rejects_bad_params(hum_freq, harmonics, notch_width):
    with pytest.raises(ValueError):
        eq_filter.create_hum_curve(FREQS, hum_freq, harmonics, notch_width)


@pytest.mark.parametrize("kwargs", [
    {"filter_type": "lowpass"},                                    # no cutoff
    {"filter_type": "highpass", "cutoff": -5},
    {"filter_type": "bandpass", "low_cutoff": 3000, "high_cutoff": 300},
    {"filter_type": "bandstop", "low_cutoff": 100},                # missing high
    {"filter_type": "lowpass", "cutoff": 1000, "order": 0},
    {"filter_type": "notafilter", "cutoff": 1000},
])
def test_filter_curve_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        eq_filter.create_filter_curve(FREQS, **kwargs)


# ---------------------------------------------------------------------
# EQ curve
# ---------------------------------------------------------------------

def test_flat_eq_is_unity():
    gain = eq_filter.create_eq_curve(FREQS, eq_filter.DEFAULT_BANDS)
    assert np.allclose(gain, 1.0)


def test_peak_band_hits_target_and_leaves_far_bins_alone():
    band = {"type": "peak", "center": 1000, "gain_db": 6, "bandwidth": 400}
    gain = eq_filter.create_eq_curve([1000.0, 1200.0, 5000.0], [band])
    assert db(gain[0]) == pytest.approx(6)
    assert db(gain[1]) == pytest.approx(3)  # half the dB at half-width
    assert db(gain[2]) == pytest.approx(0, abs=0.01)


def test_band_type_defaults_to_peak():
    band = {"center": 1000, "gain_db": 6, "bandwidth": 400}
    assert eq_filter.create_eq_curve([1000.0], [band])[0] == pytest.approx(10 ** (6 / 20))


def test_lowshelf_and_highshelf():
    low = {"type": "lowshelf", "center": 100, "gain_db": 10}
    high = {"type": "highshelf", "center": 10000, "gain_db": -10}
    freqs = [0.0, 100.0, 1000.0, 10000.0, 20000.0]

    low_db = db(eq_filter.create_eq_curve(freqs, [low]))
    assert low_db[0] == pytest.approx(10)
    assert low_db[1] == pytest.approx(5)          # half the gain at the corner
    assert low_db[2] == pytest.approx(0, abs=0.01)

    high_db = db(eq_filter.create_eq_curve(freqs, [high]))
    assert high_db[0] == pytest.approx(0)
    assert high_db[3] == pytest.approx(-5)
    assert high_db[4] < -9


def test_default_bass_band_reaches_its_gain_despite_bin_spacing():
    bands = eq_filter._eq_bands([6, 0, 0, 0, 0])
    gain = eq_filter.create_eq_curve(FREQS, bands)
    assert db(gain.max()) == pytest.approx(6, abs=0.1)


def test_band_gain_is_limited():
    limit = eq_filter.MAX_BAND_GAIN_DB
    band = {"type": "peak", "center": 1000, "gain_db": 100, "bandwidth": 400}
    assert db(eq_filter.create_eq_curve([1000.0], [band])[0]) == pytest.approx(limit)


@pytest.mark.parametrize("band", [
    {"type": "peak", "center": 1000, "gain_db": 3, "bandwidth": 0},
    {"type": "peak", "center": 0, "gain_db": 3, "bandwidth": 100},
    {"type": "wobble", "center": 1000, "gain_db": 3, "bandwidth": 100},
])
def test_eq_curve_rejects_bad_bands(band):
    with pytest.raises(ValueError):
        eq_filter.create_eq_curve(FREQS, [band])


# ---------------------------------------------------------------------
# build_curve / presets / modes
# ---------------------------------------------------------------------

def test_both_mode_is_product_of_eq_and_filter():
    bands = eq_filter._eq_bands([6, -3, 0, 3, -6])
    filt = {"filter_type": "highpass", "cutoff": 200, "order": 4}

    _, eq_curve = eq_filter.build_curve(SR, {"mode": "eq", "bands": bands})
    _, f_curve = eq_filter.build_curve(SR, {"mode": "filter", **filt})
    _, both = eq_filter.build_curve(SR, {"mode": "both", "bands": bands, **filt})

    assert np.allclose(both, eq_curve * f_curve)


@pytest.mark.parametrize("name", list(eq_filter.PRESETS))
def test_presets_build(name):
    freqs, curve = eq_filter.build_curve(SR, {"preset": name})
    assert curve.shape == freqs.shape
    assert np.all(np.isfinite(curve)) and np.all(curve >= 0)


@pytest.mark.parametrize("params", [
    {"preset": "nope"},
    {"mode": "karaoke"},
    {"mode": "filter", "filter_type": "lowpass", "cutoff": SR},  # above Nyquist
])
def test_build_curve_rejects_bad_params(params):
    with pytest.raises(ValueError):
        eq_filter.build_curve(SR, params)


def test_hum_mode_uses_sub_hz_bins():
    freqs, _ = eq_filter.build_curve(SR, {"preset": "remove_hum_50hz"})
    assert len(freqs) == eq_filter.HUM_FRAME_SIZE // 2 + 1
    assert freqs[1] < 1


def test_hum_harmonics_must_stay_below_nyquist():
    params = {"mode": "hum", "hum_freq": 50, "harmonics": 10, "notch_width": 4}
    eq_filter.build_curve(SR, params)  # 500 Hz: fine
    with pytest.raises(ValueError):
        eq_filter.build_curve(800, params)


# ---------------------------------------------------------------------
# process()
# ---------------------------------------------------------------------

def noise(shape, seed=0):
    return (np.random.default_rng(seed).standard_normal(shape) * 0.1).astype(np.float32)


def test_flat_eq_reconstructs_input_exactly():
    audio = noise(SR)
    out, sr = eq_filter.process(audio, SR, {"mode": "eq"})
    assert sr == SR
    assert out.shape == audio.shape
    assert np.allclose(out, audio, atol=1e-6)


@pytest.mark.parametrize("preset", ["telephone", "remove_hum_50hz"])
@pytest.mark.parametrize("length", [1, 100, 1023, 1024, 1537, 44100])
def test_output_length_matches_input(length, preset):
    out, _ = eq_filter.process(noise(length), SR, {"preset": preset})
    assert out.shape == (length,)


def test_stereo_keeps_shape_and_channels_independent():
    audio = noise((SR, 2))
    audio[:, 1] = 0
    out, _ = eq_filter.process(audio, SR, {"preset": "bass_boost"})
    assert out.shape == audio.shape
    assert out.dtype == np.float32
    assert np.allclose(out[:, 1], 0)


def test_lowpass_removes_high_tone():
    t = np.arange(SR) / SR
    low, high = np.sin(2 * np.pi * 200 * t), np.sin(2 * np.pi * 8000 * t)
    out, _ = eq_filter.process((low + high) * 0.4, SR,
                               {"mode": "filter", "filter_type": "lowpass", "cutoff": 1000})
    residual = out - low * 0.4
    assert np.sqrt(np.mean(residual[2048:-2048] ** 2)) < 0.01


def test_hum_removal_removes_hum_and_keeps_music():
    t = np.arange(8 * SR) / SR
    hum = 0.1 * sum(np.sin(2 * np.pi * 50 * k * t + k) / k for k in range(1, 6))
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    out, _ = eq_filter.process(hum + tone, SR, {"preset": "remove_hum_50hz"})

    # Away from the edges, where the hum is only partly removed (see HUM_FRAME_SIZE).
    mid = slice(SR, -SR)
    rms = lambda x: np.sqrt(np.mean(x ** 2))
    assert db(rms(out[mid] - tone[mid]) / rms(hum[mid])) < -40


def test_boost_never_clips():
    out, _ = eq_filter.process(noise(SR) * 9, SR, {"mode": "eq", "bands": eq_filter._eq_bands([15] * 5)})
    assert np.max(np.abs(out)) <= 1.0
