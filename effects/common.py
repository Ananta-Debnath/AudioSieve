"""Helpers shared by the time-domain effects (reverb, echo)."""

import numpy as np


def per_channel(audio, process_channel):
    """Run process_channel(x, channel_index) on each channel independently.

    audio: shape (frames,) or (frames, channels). Channel outputs may be
    longer than the input (effect tails), but must all have the same length.
    Returns float64 audio in the same layout.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        return process_channel(audio, 0)
    return np.stack(
        [process_channel(audio[:, ch], ch) for ch in range(audio.shape[1])],
        axis=1,
    )


def prevent_clipping(out):
    """Scale down (instead of hard-clipping) if peaks exceed full scale.

    Same protection as eq_filter.process; the clip is the same safety net
    utils.save_audio uses.
    """
    peak = np.max(np.abs(out)) if out.size else 0
    if peak > 1.0:
        out = out / peak
    return np.clip(out, -1.0, 1.0)
