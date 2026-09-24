"""Waveform envelopes: what the players and plots draw instead of every sample."""

import numpy as np

ENVELOPE_POINTS = 2000


def envelope(audio, n_points=ENVELOPE_POINTS):
    """Min/max envelope of audio in n_points equal time buckets.

    audio: shape (frames,) or (frames, channels). Each bucket keeps the
    highest and lowest sample of any channel, so nothing out of phase
    between channels cancels (as it would in a mono mix).

    Returns (maxima, minima), float32 arrays of min(n_points, frames)
    values each (empty for empty audio).
    """
    audio = np.asarray(audio)
    if audio.ndim == 2:
        high, low = audio.max(axis=1), audio.min(axis=1)
    else:
        high = low = audio

    n_points = min(n_points, len(high))
    if n_points == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    starts = np.linspace(0, len(high), n_points, endpoint=False).astype(int)
    return (np.maximum.reduceat(high, starts).astype(np.float32),
            np.minimum.reduceat(low, starts).astype(np.float32))


def waveform_json(audio, sr, n_points=ENVELOPE_POINTS):
    """Envelope + duration in the shape WaveSurfer takes as pre-computed
    peaks: {"duration": s, "peaks": [maxima, minima]} (it draws the first
    list above the centre line and the second below it)."""
    high, low = envelope(audio, n_points)
    return {
        "duration": len(audio) / sr,
        "peaks": [high.round(4).tolist(), low.round(4).tolist()],
    }
