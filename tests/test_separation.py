"""Tests for effects/separation.py, the wrapper the app calls around the
NMF separation pipeline (semi_supervised_nmf + nmf_sep).

Run with: python -m pytest tests -m "not slow"
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from effects import separation

SR = 8000
DEMO = Path(__file__).resolve().parent.parent / "static" / "demo" / "demo.wav"


def mix(seconds=1.0, channels=None, seed=0):
    """A bass tone, a few clicks and some noise: something to split."""
    t = np.arange(int(seconds * SR)) / SR
    x = 0.4 * np.sin(2 * np.pi * 80 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    x[:: SR // 4] += 0.8
    x += np.random.default_rng(seed).standard_normal(len(t)) * 0.02
    if channels:
        x = np.stack([x * (1 - 0.3 * ch) for ch in range(channels)], axis=1)
    return x


@pytest.mark.parametrize("channels", [None, 2])
def test_stems_have_the_input_shape_and_are_finite(channels):
    audio = mix(channels=channels)
    stems = separation.separate(audio, SR)
    assert list(stems) == list(separation.STEMS) == ["percussion", "bass", "vocals", "harmonics"]
    for name, stem in stems.items():
        assert stem.shape == audio.shape, name
        assert np.all(np.isfinite(stem)), name


@pytest.mark.parametrize("channels", [None, 2])
def test_stems_add_up_to_the_mix(channels):
    # The masks sum to 1 in every bin, so nothing is lost or doubled,
    # up to the first and last sample (the padding keeps those exact).
    audio = mix(channels=channels)
    total = sum(separation.separate(audio, SR).values())
    assert np.allclose(total, audio, atol=1e-9)


def test_the_same_track_gives_the_same_stems_and_keeps_the_global_rng():
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)

    first = separation.separate(mix(), SR)
    assert np.random.random() == expected_next  # the fixed seed didn't leak out
    second = separation.separate(mix(), SR)
    for name in separation.STEMS:
        assert np.array_equal(first[name], second[name])


def test_scale_does_not_change_the_split():
    # main() normalises the mono mix before the NMF; stems follow the input level.
    audio = mix()
    quiet = separation.separate(audio * 0.01, SR)
    loud = separation.separate(audio, SR)
    for name in separation.STEMS:
        assert np.allclose(quiet[name] * 100, loud[name], atol=1e-6)


@pytest.mark.parametrize("n", [0, 1, 100, 2000])
def test_silence_and_very_short_input(n):
    for audio in (np.zeros(n), mix()[:n]):
        stems = separation.separate(audio, SR)
        for stem in stems.values():
            assert stem.shape == audio.shape and np.all(np.isfinite(stem))


def test_frames_match_main():
    # main_semi_nmf.main(): 40 ms rounded up to a power of two, 50% overlap.
    assert separation.frame_sizes(44100) == (2048, 1024)
    assert separation.frame_sizes(48000) == (2048, 1024)
    assert separation.frame_sizes(8000) == (512, 256)


def test_component_groups_follow_main():
    n = separation.NUM_COMPONENTS
    df = pd.DataFrame({
        "percussion_score": np.linspace(0, 1, n),       # 19, 18 score highest
        "bass_score": [0.9, 0.7] + [0.1] * (n - 2),       # 0 and 1 pass 0.6
        "vocal_score": np.linspace(1, 0, n),            # 0, 1, 2, ... score highest
    })
    groups = separation.component_groups(df)
    assert groups["percussion"] == [19, 18]
    assert groups["bass"] == [0, 1]
    # 3/5 of the 16 components the first two groups left free
    assert groups["vocal"] == list(range(9))
    assert groups["remaining"] == list(range(9, 18))


@pytest.mark.slow
def test_the_demo_track_separates_in_well_under_a_minute():
    audio, sr = sf.read(str(DEMO), dtype="float32")
    start = time.perf_counter()
    stems = separation.separate(audio, sr)
    elapsed = time.perf_counter() - start
    for stem in stems.values():
        assert stem.shape == audio.shape and np.all(np.isfinite(stem))
    assert np.allclose(sum(stems.values()), audio, atol=1e-6)
    assert elapsed < 30, f"{elapsed:.1f} s for {len(audio) / sr:.0f} s of audio"


def test_an_empty_group_gives_a_silent_stem(monkeypatch):
    def no_bass(df, num_components=separation.NUM_COMPONENTS):
        groups = {"percussion": [0, 1], "bass": [], "vocal": [2, 3]}
        groups["remaining"] = list(range(4, num_components))
        return groups

    monkeypatch.setattr(separation, "component_groups", no_bass)
    audio = mix()
    stems = separation.separate(audio, SR)
    assert np.all(stems["bass"] == 0)
    assert np.allclose(sum(stems.values()), audio, atol=1e-9)
