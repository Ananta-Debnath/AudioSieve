"""Write a 20-second placeholder demo track to static/demo/demo.wav.

A sine melody, a low bass tone and noise bursts: enough for every tool
to have something to act on (highs for the filters, lows for the rumble
filter, transients for the reverb and echo tails) until the real demo
track replaces it.

Run from the repo root:  python scripts/make_placeholder_demo.py [--force]
It refuses to overwrite an existing demo.wav without --force.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

OUT_PATH = Path(__file__).resolve().parent.parent / "static" / "demo" / "demo.wav"

SR = 44100
SECONDS = 20
BEAT_SEC = 0.5  # 120 BPM

# A minor pentatonic, one note per half beat, looping.
MELODY_HZ = [440.0, 523.25, 587.33, 659.26, 783.99, 659.26, 587.33, 523.25,
             440.0, 523.25, 659.26, 880.0, 783.99, 659.26, 587.33, 523.25]
BASS_HZ = [55.0, 43.65, 65.41, 49.0]  # A1 F1 C2 G1, two seconds each


def _fade(n, sr, fade_sec=0.005):
    """1 in the middle, short linear ramps at both ends (no clicks)."""
    ramp = min(int(fade_sec * sr), n // 2)
    env = np.ones(n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[n - ramp:] = np.linspace(1, 0, ramp)
    return env


def make_demo(sr=SR, seconds=SECONDS, seed=0):
    """Returns the placeholder track: mono float64, peak 0.8."""
    n = int(sr * seconds)
    out = np.zeros(n)

    note_len = int(BEAT_SEC / 2 * sr)
    t = np.arange(note_len) / sr
    note_env = np.exp(-t / 0.12) * _fade(note_len, sr)
    for i, start in enumerate(range(0, n - note_len + 1, note_len)):
        freq = MELODY_HZ[i % len(MELODY_HZ)]
        out[start:start + note_len] += 0.35 * np.sin(2 * np.pi * freq * t) * note_env

    bass_len = int(4 * BEAT_SEC * sr)
    t = np.arange(bass_len) / sr
    for i, start in enumerate(range(0, n, bass_len)):
        seg = min(bass_len, n - start)
        freq = BASS_HZ[i % len(BASS_HZ)]
        out[start:start + seg] += (0.4 * np.sin(2 * np.pi * freq * t[:seg])
                                   * _fade(seg, sr, fade_sec=0.02))

    rng = np.random.default_rng(seed)
    burst_len = int(0.08 * sr)
    burst_env = np.exp(-np.arange(burst_len) / sr / 0.015)
    for k, start in enumerate(range(0, n - burst_len + 1, int(BEAT_SEC * sr))):
        level = 0.45 if k % 2 else 0.25  # louder on the off-beat
        out[start:start + burst_len] += level * rng.standard_normal(burst_len) * burst_env

    return out * 0.8 / np.max(np.abs(out))


def main(argv):
    if OUT_PATH.exists() and "--force" not in argv:
        print(f"{OUT_PATH} already exists; pass --force to overwrite it.")
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_PATH), make_demo(), SR, subtype="PCM_16")
    print(f"Wrote {OUT_PATH} ({SECONDS} s at {SR} Hz).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
