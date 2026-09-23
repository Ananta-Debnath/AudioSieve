"""Reverb effect: FFT convolution with a synthetic room impulse response.

A room is modelled as an LTI system, so the reverberant signal is the
input convolved with the room's impulse response (IR):

    y = x * h

Convolution is computed via the convolution theorem, i.e. as a
multiplication in the frequency domain:

    Y(k) = X(k) · H(k)

The IR is synthetic: exponentially decaying white noise (dense, random
reflections whose energy dies away), preceded by a short silent gap
(pre-delay, the time before the first reflection arrives).
"""

import numpy as np

from effects.common import per_channel, prevent_clipping

# ln(1000): exp(-LN_1000 * t / rt60) is 1/1000 (-60 dB) at t = rt60.
LN_1000 = 6.9078
MAX_IR_SEC = 5

# Parameter -> (min, max, default); the route validates against these.
PARAMS = {
    "rt60": (0.2, 5.0, 1.5),
    "pre_delay_ms": (0, 100, 20),
    "wet": (0.0, 1.0, 0.3),
}


def generate_impulse_response(sr, rt60, pre_delay_ms, seed=0):
    """Synthetic room IR: exponentially decaying white noise.

    sr: sample rate (Hz)
    rt60: time (s) for the reverb to decay by 60 dB; the decay part is
        rt60 seconds long, capped at MAX_IR_SEC.
    pre_delay_ms: silence before the decay starts.
    seed: noise seed; the same seed always gives the same IR.

    Normalised to unit energy (sum of h^2 == 1), so the wet level stays
    predictable whatever rt60 is.
    Returns a float64 1-D array.
    """
    if rt60 <= 0:
        raise ValueError("rt60 must be positive.")

    length = int(min(rt60, MAX_IR_SEC) * sr)
    t = np.arange(length) / sr
    env = np.exp(-LN_1000 * t / rt60)

    noise = np.random.default_rng(seed).standard_normal(length)
    decay = noise * env

    pre_delay = np.zeros(int(pre_delay_ms / 1000 * sr))
    h = np.concatenate([pre_delay, decay])
    return h / np.sqrt(np.sum(h ** 2))


def _next_pow2(n):
    return 1 << max(n - 1, 0).bit_length()


def fft_convolve(x, h):
    """LINEAR convolution of x and h via the FFT.

    Multiplying two N-point DFTs gives the CIRCULAR convolution of the
    two sequences: output that would run past the end wraps around onto
    the start. The linear convolution has N = len(x) + len(h) - 1
    samples, so zero-padding both inputs to at least N points leaves
    room for the whole result and nothing wraps; the extra zeros are
    trimmed off afterwards. (The FFT length is rounded up to a power of
    two, where the FFT is fastest.)

    Returns y with len(y) == len(x) + len(h) - 1.
    """
    n = len(x) + len(h) - 1
    nfft = _next_pow2(n)

    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)
    return np.fft.irfft(X * H, nfft)[:n]


def circular_convolve(x, h):
    """CIRCULAR convolution of x and h: the WRONG way to do reverb.

    Only here for the educational toggle in the app. The FFT length is
    len(x), with no zero-padding, so the reverb tail that should ring on
    after the end of x wraps around and is smeared onto its start.

    h is truncated to len(x) if longer. Returns len(x) samples.
    """
    n = len(x)
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h[:n], n), n)


def apply_reverb(audio, sr, rt60=1.5, pre_delay_ms=20, wet=0.3, circular=False):
    """Convolve audio with a synthetic room IR and mix it with the dry signal.

        out = (1 - wet) · dry + wet · (x * h)

    audio: shape (frames,) or (frames, channels); channels are processed
        independently, each with its own IR (seed = channel index), so
        stereo gets slightly different tails, which sounds wider.
    circular: False (linear convolution) keeps the tail, so the output
        is len(h) - 1 samples longer than the input. True uses
        circular_convolve (the wrong way, for demonstration): same
        length as the input, tail wrapped onto the start.

    Returns float64 audio, scaled down if it would clip.
    """
    def reverb_channel(x, ch):
        h = generate_impulse_response(sr, rt60, pre_delay_ms, seed=ch)
        if circular:
            wet_sig = circular_convolve(x, h)
            dry = x
        else:
            wet_sig = fft_convolve(x, h)
            dry = np.pad(x, (0, len(wet_sig) - len(x)))
        return (1 - wet) * dry + wet * wet_sig

    return prevent_clipping(per_channel(audio, reverb_channel))


def process(audio, sr, **params):
    """params: apply_reverb keyword arguments. Returns (audio, sr)."""
    return apply_reverb(audio, sr, **params), sr
