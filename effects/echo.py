"""Echo / delay effect.

Stub: returns the input unchanged. Real DSP lands in a later stage;
keep the signature so app.py routing doesn't need to change.
"""


def process(audio, sr, **params):
    """Apply the effect.

    audio: float ndarray, shape (frames,) or (frames, channels)
    sr: sample rate in Hz
    params: effect parameters from the request (ignored for now)

    Returns (audio, sr).
    """
    return audio, sr
