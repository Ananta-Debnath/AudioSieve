"""Flanger: a feedforward comb filter whose delay sweeps back and forth.

    y[n] = x[n] + g · x[n − D(n)]

Same difference equation as the feedforward echo (effects/echo.py), but
D(n) is only a few ms and is moved slowly by a low-frequency oscillator
(LFO). The comb's notches, at odd multiples of sr / (2D) Hz, sweep up
and down the spectrum as D changes: the "jet plane" whoosh. g = 1 makes
the delayed copy cancel the input completely at the notches.
"""

import numpy as np

from effects.common import per_channel, prevent_clipping

# Parameter -> (min, max, default); the route validates against these.
PARAMS = {
    "min_delay_ms": (0.1, 10, 1),
    "sweep_ms": (0, 10, 3),
    "rate_hz": (0.05, 5, 0.25),
    "gain": (0, 1, 0.9),
}


def sweep_delay_ms(phase, min_delay_ms, sweep_ms):
    """D at LFO phase (in cycles): min_delay_ms at phase 0, min_delay_ms +
    sweep_ms at phase 0.5, back at phase 1 (a raised cosine)."""
    return min_delay_ms + sweep_ms * (1 - np.cos(2 * np.pi * phase)) / 2


def fractional_delay(x, delays):
    """x[n − D(n)] for a time-varying D(n) in (fractional) samples.

    n − D(n) usually falls between two samples, so the value is linearly
    interpolated from those two neighbours. Samples before the start of
    x are 0.
    """
    pos = np.arange(len(x)) - delays
    left = np.floor(pos).astype(int)
    frac = pos - left

    # Prepend zeros so reads before the start land on them.
    lead = int(np.ceil(np.max(delays, initial=0))) + 1
    padded = np.concatenate([np.zeros(lead), x, [0.0]])
    return (1 - frac) * padded[left + lead] + frac * padded[left + lead + 1]


def apply_flanger(audio, sr, min_delay_ms=1, sweep_ms=3, rate_hz=0.25, gain=0.9):
    """Run the flanger's difference equation on each channel.

    D(n) follows sweep_delay_ms at rate_hz sweeps per second. Because D
    keeps changing, the delayed copy is also slightly pitch-shifted, like
    a Doppler shift.

    audio: shape (frames,) or (frames, channels); every channel uses the
    same sweep. The output is ceil(min_delay_ms + sweep_ms) ms longer than
    the input, so the delayed copy can finish.

    Returns float64 audio, scaled down if it would clip.
    """
    audio = np.asarray(audio)
    tail = int(np.ceil((min_delay_ms + sweep_ms) / 1000 * sr))
    phase = rate_hz * np.arange(len(audio) + tail) / sr
    delays = sweep_delay_ms(phase, min_delay_ms, sweep_ms) / 1000 * sr

    def flanger_channel(x, _ch):
        dry = np.pad(x, (0, tail))
        return dry + gain * fractional_delay(dry, delays)

    return prevent_clipping(per_channel(audio, flanger_channel))


def flanger_frequency_response(sr, min_delay_ms, sweep_ms, gain,
                               n_points=600, n_frames=48, f_max=None):
    """Magnitude response at n_frames delays spread evenly over one sweep.

    For a fixed D the flanger is a feedforward comb:

        H(e^jw) = 1 + g · e^(−jwD)

    with notches where the delayed copy arrives in antiphase, at odd
    multiples of sr / (2D) Hz. Evaluated at n_points frequencies from
    0 Hz to Nyquist, or to f_max (Hz) if given.

    Returns (freqs_hz, delays_ms, mag_db); mag_db has one row per delay,
    shape (n_frames, n_points).
    """
    w_max = np.pi if f_max is None else 2 * np.pi * min(f_max, sr / 2) / sr
    w = np.linspace(0, w_max, n_points)
    delays_ms = sweep_delay_ms(np.arange(n_frames) / n_frames, min_delay_ms, sweep_ms)

    delays = delays_ms[:, None] / 1000 * sr
    response = 1 + gain * np.exp(-1j * w[None, :] * delays)

    freqs_hz = w / (2 * np.pi) * sr
    return freqs_hz, delays_ms, 20 * np.log10(np.abs(response) + 1e-12)


def process(audio, sr, **params):
    """params: apply_flanger keyword arguments. Returns (audio, sr)."""
    return apply_flanger(audio, sr, **params), sr
