"""Echo / delay effect, defined by difference equations.

D is the delay in samples, g the gain of each repeat:

- feedforward (one echo, FIR comb filter):

      y[n] = x[n] + g · x[n − D]

- feedback (repeating, decaying echoes, IIR comb filter):

      y[n] = x[n] + g · y[n − D]

  Its impulse response is 1, g, g^2, ... at 0, D, 2D, ..., which only
  decays for |g| < 1, so g is clamped to [0, MAX_GAIN].
"""

import numpy as np

from effects.common import per_channel, prevent_clipping

MODES = ("feedforward", "feedback")
MAX_GAIN = 0.9
MAX_TAIL_SEC = 5

# Parameter -> (min, max, default); the route validates against these.
PARAMS = {
    "delay_ms": (20, 2000, 350),
    "gain": (0.0, MAX_GAIN, 0.5),
    "mix": (0.0, 1.0, 0.5),
}


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: {', '.join(MODES)}."
        )


def _clamp_gain(gain):
    return float(np.clip(gain, 0.0, MAX_GAIN))


def delay_samples(sr, delay_ms):
    """Delay D in whole samples (at least 1)."""
    return max(1, int(delay_ms / 1000 * sr))


def tail_length(sr, delay, gain, mode):
    """Zeros to append so the echoes can ring out after the input ends.

    feedforward: the single echo, D samples.
    feedback: K repeats, where g^K <= 0.001 (the echo is 60 dB down):
        K = ceil(log(0.001) / log(g)).
    Capped at MAX_TAIL_SEC; 0 when g == 0 (no echo at all).
    """
    if gain == 0:
        return 0
    if mode == "feedforward":
        tail = delay
    else:
        repeats = int(np.ceil(np.log(0.001) / np.log(gain)))
        tail = delay * repeats
    return min(tail, int(MAX_TAIL_SEC * sr))


def feedforward_comb(x, delay, gain):
    """y[n] = x[n] + g · x[n − D], as one vectorized slice addition."""
    y = x.copy()
    y[delay:] += gain * x[:-delay]
    return y


def feedback_comb(x, delay, gain):
    """y[n] = x[n] + g · y[n − D], vectorized in blocks of D samples.

    y[n] only depends on y[n − D], so every sample in a block of D
    consecutive samples depends only on the block before it:

        y[0:D]          = x[0:D]
        y[kD:(k+1)D]    = x[kD:(k+1)D] + g · y[(k−1)D:kD]

    Blocks are computed in order, so the previous block is always final.
    This is exact (not an approximation) and loops len(x) / D times
    instead of once per sample. The last block may be shorter than D.
    """
    y = x.copy()
    for start in range(delay, len(y), delay):
        stop = min(start + delay, len(y))
        y[start:stop] += gain * y[start - delay:stop - delay]
    return y


def apply_echo(audio, sr, delay_ms=350, gain=0.5, mix=0.5, mode="feedback"):
    """Run the difference equation and mix the echoes with the dry signal.

        out = dry + mix · (y − dry)

    mix = 0 gives the dry input, mix = 1 the full difference-equation
    output y. gain is clamped to [0, MAX_GAIN].

    audio: shape (frames,) or (frames, channels); channels are processed
    independently. The output is tail_length() samples longer than the
    input, so the echoes can ring out.

    Returns float64 audio, scaled down if it would clip.
    """
    _check_mode(mode)
    gain = _clamp_gain(gain)
    delay = delay_samples(sr, delay_ms)
    tail = tail_length(sr, delay, gain, mode)
    comb = feedforward_comb if mode == "feedforward" else feedback_comb

    def echo_channel(x, _ch):
        dry = np.pad(x, (0, tail))
        y = comb(dry, delay, gain)
        return dry + mix * (y - dry)

    return prevent_clipping(per_channel(audio, echo_channel))


def echo_frequency_response(sr, delay_ms, gain, mode, n_points=2048, f_max=None):
    """Magnitude response of the echo's difference equation.

    Evaluated at n_points frequencies w in [0, pi] (rad/sample), i.e.
    0 Hz to Nyquist:

        feedforward: H(e^jw) = 1 + g · e^(−jwD)
        feedback:    H(e^jw) = 1 / (1 − g · e^(−jwD))

    Both are comb filters: peaks where the delayed copy adds in phase
    (every sr / D Hz, starting at 0 Hz), notches halfway between.

    f_max (Hz, optional) evaluates [0, f_max] instead of up to Nyquist,
    so plots can zoom in on the teeth, which are very dense for long delays.

    Returns (freqs_hz, mag_db).
    """
    _check_mode(mode)
    gain = _clamp_gain(gain)
    delay = delay_samples(sr, delay_ms)

    w_max = np.pi if f_max is None else 2 * np.pi * min(f_max, sr / 2) / sr
    w = np.linspace(0, w_max, n_points)
    delayed = gain * np.exp(-1j * w * delay)
    if mode == "feedforward":
        response = 1 + delayed
    else:
        response = 1 / (1 - delayed)

    freqs_hz = w / (2 * np.pi) * sr
    return freqs_hz, 20 * np.log10(np.abs(response) + 1e-12)


def process(audio, sr, **params):
    """params: apply_echo keyword arguments. Returns (audio, sr)."""
    return apply_echo(audio, sr, **params), sr
