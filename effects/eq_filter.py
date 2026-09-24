"""EQ / filter effect.

Both modes are a static per-frequency-bin gain curve multiplied onto the
STFT spectrum (same mechanism as the drum/bass masks, but the curve is
the same for every frame):

- filter: Butterworth magnitude response (lowpass / highpass / bandpass /
  bandstop), computed from its closed-form formula, ~1 in the passband,
  ~0 outside.
- eq: sum of per-band boost/cut shapes in dB (Gaussian peaks, low/high
  shelves), converted to linear gain.
- both: the EQ curve and the filter curve multiplied together.
- hum: narrow notches at a mains-hum frequency and its harmonics, on a
  much longer frame (see HUM_FRAME_SIZE).
"""

import numpy as np

import stft
import utils

FRAME_SIZE = 1024
HOP_SIZE = 512

MODES = ("eq", "filter", "both", "hum")
FILTER_TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
BAND_TYPES = ("peak", "lowshelf", "highshelf")
MAX_BAND_GAIN_DB = 15

# Steepness of the shelf transition (same role as a filter order).
SHELF_ORDER = 2

# Hum notches are a few Hz wide, but FRAME_SIZE bins are sr / 1024
# (~43 Hz) apart, far too coarse to cut one: a 50 Hz notch there only
# reduces a 50 Hz tone by ~4.5 dB. Frequency resolution is sr /
# frame_size, so hum mode uses a 64x longer frame: ~0.67 Hz bins at
# 44.1 kHz. The price is time resolution (~1.5 s frames), which doesn't
# matter here because the curve is flat everywhere except the notches.
# Only the first and last ~0.5 s keep some hum: telling hum apart from
# notes a few Hz away takes that much signal (true of any notch this
# narrow, IIR ones included).
HUM_FRAME_SIZE = 65536
HUM_HOP_SIZE = HUM_FRAME_SIZE // 2
NOTCH_ORDER = 4

# Hum parameter -> (min, max, default); the route validates against these.
# Mains hum is 50 Hz (e.g. Bangladesh, Europe) or 60 Hz (e.g. the US).
HUM_PARAMS = {
    "hum_freq": (40, 70, 50),
    "harmonics": (1, 10, 5),
    "notch_width": (1, 10, 4),
}

# 5-band EQ.
# - peak: Gaussian bump centred on 'center'; 'bandwidth' is its full
#   width at half maximum, in Hz.
# - lowshelf / highshelf: boost/cut everything below / above 'center',
#   reaching half the gain (in dB) at 'center'. 'bandwidth' is unused.
# Bins are sr / FRAME_SIZE (~43 Hz) apart, so peak bands narrower than
# ~150 Hz only touch a couple of bins and never reach their full gain.
DEFAULT_BANDS = [
    {"name": "Bass", "type": "lowshelf", "center": 100, "gain_db": 0, "bandwidth": 150},
    {"name": "Low-mid", "type": "peak", "center": 350, "gain_db": 0, "bandwidth": 250},
    {"name": "Mid", "type": "peak", "center": 1000, "gain_db": 0, "bandwidth": 700},
    {"name": "High-mid", "type": "peak", "center": 3500, "gain_db": 0, "bandwidth": 2500},
    {"name": "Treble", "type": "highshelf", "center": 10000, "gain_db": 0, "bandwidth": 6000},
]


def _eq_bands(gains_db):
    """DEFAULT_BANDS with the given per-band gains (dB), in order."""
    return [{**band, "gain_db": g} for band, g in zip(DEFAULT_BANDS, gains_db)]


# Ready-made settings; process({'preset': name}) uses these params as-is.
PRESETS = {
    "telephone": {"mode": "filter", "filter_type": "bandpass",
                  "low_cutoff": 300, "high_cutoff": 3400, "order": 4},
    "radio": {"mode": "filter", "filter_type": "bandpass",
              "low_cutoff": 500, "high_cutoff": 5000, "order": 2},
    "remove_rumble": {"mode": "filter", "filter_type": "highpass",
                      "cutoff": 80, "order": 4},
    "remove_hum_50hz": {"mode": "hum", "hum_freq": 50, "harmonics": 5,
                        "notch_width": 4},
    "bass_boost": {"mode": "eq", "bands": _eq_bands([8, 2, 0, 0, 0])},
}


# ---------------------------------------------------------------------
# Filter curves (Butterworth magnitude responses)
# ---------------------------------------------------------------------
#
# Butterworth is defined by its power response |H(f)|^2, e.g. for
# low-pass 1 / (1 + (f/fc)^(2n)). We multiply the gain onto the complex
# spectrum, i.e. onto amplitudes, so the curve is the square root of
# that: -3 dB at the cutoff and 6n dB/octave rolloff.

def _lowpass(freq_bins, cutoff, order):
    return 1 / np.sqrt(1 + (freq_bins / cutoff) ** (2 * order))


def _highpass(freq_bins, cutoff, order):
    gain = np.zeros_like(freq_bins, dtype=float)
    nonzero = freq_bins > 0  # DC bin stays at 0
    gain[nonzero] = 1 / np.sqrt(1 + (cutoff / freq_bins[nonzero]) ** (2 * order))
    return gain


def _bandstop(freq_bins, low_cutoff, high_cutoff, order):
    """Butterworth band-stop via the low-pass -> band-stop transform
    f/fc -> B*f / (f0^2 - f^2), with f0 = sqrt(low*high), B = high - low.

    -3 dB exactly at low_cutoff and high_cutoff, 0 at the centre f0.
    (1 - bandpass would not reach 0 for narrow notches, because the
    cascaded bandpass itself doesn't reach 1 there.)
    """
    f0_sq = low_cutoff * high_cutoff
    bandwidth = high_cutoff - low_cutoff

    gain = np.zeros_like(freq_bins, dtype=float)
    denom = f0_sq - freq_bins ** 2
    away = denom != 0  # exactly at f0 the gain is 0
    ratio = bandwidth * freq_bins[away] / denom[away]
    gain[away] = 1 / np.sqrt(1 + ratio ** (2 * order))
    return gain


def create_filter_curve(freq_bins, filter_type, cutoff=None,
                        low_cutoff=None, high_cutoff=None, order=4):
    """
    freq_bins: array of bin center frequencies (Hz), from rfft bin count.
    filter_type: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
    Returns: array of gain values in [0, 1], same length as freq_bins.
    """
    freq_bins = np.asarray(freq_bins, dtype=float)

    if order <= 0:
        raise ValueError("order must be positive.")

    if filter_type in ("lowpass", "highpass"):
        if cutoff is None or cutoff <= 0:
            raise ValueError(f"{filter_type} needs a positive cutoff.")
        if filter_type == "lowpass":
            return _lowpass(freq_bins, cutoff, order)
        return _highpass(freq_bins, cutoff, order)

    if filter_type in ("bandpass", "bandstop"):
        if low_cutoff is None or high_cutoff is None:
            raise ValueError(f"{filter_type} needs low_cutoff and high_cutoff.")
        if not 0 < low_cutoff < high_cutoff:
            raise ValueError(f"{filter_type} needs 0 < low_cutoff < high_cutoff.")
        if filter_type == "bandpass":
            return (_lowpass(freq_bins, high_cutoff, order)
                    * _highpass(freq_bins, low_cutoff, order))
        return _bandstop(freq_bins, low_cutoff, high_cutoff, order)

    raise ValueError(
        f"Unknown filter_type '{filter_type}'. "
        f"Expected one of: {', '.join(FILTER_TYPES)}."
    )


def create_hum_curve(freq_bins, hum_freq, harmonics, notch_width, order=NOTCH_ORDER):
    """Notches at hum_freq, 2·hum_freq, ..., harmonics·hum_freq.

    Each notch is a Butterworth band-stop (_bandstop), 0 exactly on the
    harmonic and notch_width Hz wide between its -3 dB points. _bandstop
    centres on sqrt(low·high), so for centre c the edges solve
    low·high = c^2 and high - low = notch_width.

    Returns: array of gain values in [0, 1], same length as freq_bins.
    """
    freq_bins = np.asarray(freq_bins, dtype=float)

    if hum_freq <= 0 or notch_width <= 0:
        raise ValueError("hum_freq and notch_width must be positive.")
    if harmonics < 1 or harmonics != int(harmonics):
        raise ValueError("harmonics must be a whole number, at least 1.")

    gain = np.ones_like(freq_bins)
    for k in range(1, int(harmonics) + 1):
        centre = k * hum_freq
        low = np.sqrt(centre ** 2 + notch_width ** 2 / 4) - notch_width / 2
        gain *= _bandstop(freq_bins, low, low + notch_width, order)
    return gain


# ---------------------------------------------------------------------
# EQ curve
# ---------------------------------------------------------------------

def _band_shape(freq_bins, band):
    """0..1 weight of one band across frequency (multiplied by gain_db)."""
    band_type = band.get("type", "peak")
    center = band["center"]

    if center <= 0:
        raise ValueError("EQ band center must be positive.")

    if band_type == "peak":
        if band["bandwidth"] <= 0:
            raise ValueError("EQ band bandwidth must be positive.")
        # FWHM -> standard deviation
        sigma = band["bandwidth"] / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-0.5 * ((freq_bins - center) / sigma) ** 2)

    # Shelves: a smooth 1 -> 0 (low) or 0 -> 1 (high) step in dB,
    # halfway at 'center'. Same x^(2n) form as the Butterworth curves.
    x = (freq_bins / center) ** (2 * SHELF_ORDER)
    if band_type == "lowshelf":
        return 1 / (1 + x)
    if band_type == "highshelf":
        return x / (1 + x)

    raise ValueError(
        f"Unknown EQ band type '{band_type}'. "
        f"Expected one of: {', '.join(BAND_TYPES)}."
    )


def create_eq_curve(freq_bins, bands):
    """
    bands: list of dicts, e.g.
      [{'center': 100, 'gain_db': 6, 'bandwidth': 80}, ...]
      optional 'type': 'peak' (default) | 'lowshelf' | 'highshelf'
    Returns: array of linear gain values (can be >1 for boost, <1 for cut).
    """
    freq_bins = np.asarray(freq_bins, dtype=float)
    total_db = np.zeros_like(freq_bins)

    for band in bands:
        gain_db = np.clip(band["gain_db"], -MAX_BAND_GAIN_DB, MAX_BAND_GAIN_DB)
        total_db += gain_db * _band_shape(freq_bins, band)

    return 10 ** (total_db / 20)


# ---------------------------------------------------------------------
# Params -> curve -> audio
# ---------------------------------------------------------------------

def resolve_params(params):
    """Expand {'preset': name} into that preset's params."""
    preset = params.get("preset")
    if not preset:
        return params
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Expected one of: {', '.join(PRESETS)}."
        )
    return PRESETS[preset]


def _filter_curve_from_params(freq_bins, sr, params):
    nyquist = sr / 2
    for key in ("cutoff", "low_cutoff", "high_cutoff"):
        value = params.get(key)
        if value is not None and value >= nyquist:
            raise ValueError(
                f"{key} ({value} Hz) must be below Nyquist ({nyquist:g} Hz)."
            )
    return create_filter_curve(
        freq_bins,
        params.get("filter_type", "lowpass"),
        cutoff=params.get("cutoff"),
        low_cutoff=params.get("low_cutoff"),
        high_cutoff=params.get("high_cutoff"),
        order=params.get("order", 4),
    )


def _hum_curve_from_params(freq_bins, sr, params):
    top = params["harmonics"] * params["hum_freq"]
    if top >= sr / 2:
        raise ValueError(
            f"Highest harmonic ({top:g} Hz) must be below Nyquist ({sr / 2:g} Hz)."
        )
    return create_hum_curve(
        freq_bins, params["hum_freq"], params["harmonics"], params["notch_width"]
    )


def frame_and_hop(params):
    """(frame_size, hop_size) for the params' mode: hum needs long frames."""
    if resolve_params(params).get("mode") == "hum":
        return HUM_FRAME_SIZE, HUM_HOP_SIZE
    return FRAME_SIZE, HOP_SIZE


def build_curve(sr, params, frame_size=None):
    """Build the gain curve for the rfft bins of one frame from params.

    frame_size defaults to the mode's own (see frame_and_hop).
    Returns (freq_bins, curve).
    """
    params = resolve_params(params)
    if frame_size is None:
        frame_size, _ = frame_and_hop(params)
    freq_bins = np.fft.rfftfreq(frame_size, d=1 / sr)
    mode = params.get("mode", "eq")

    if mode not in MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: {', '.join(MODES)}."
        )

    if mode == "hum":
        return freq_bins, _hum_curve_from_params(freq_bins, sr, params)

    curve = np.ones_like(freq_bins)
    if mode in ("eq", "both"):
        curve *= create_eq_curve(freq_bins, params.get("bands", DEFAULT_BANDS))
    if mode in ("filter", "both"):
        curve *= _filter_curve_from_params(freq_bins, sr, params)

    return freq_bins, curve


def _process_channel(audio, curve, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    """STFT -> multiply by the static curve -> ISTFT, for one mono channel."""
    # Pad so the first/last samples sit under full window overlap and
    # get_frames doesn't drop the tail; trimmed back afterwards.
    padded = np.pad(audio, (frame_size, frame_size + hop_size))

    window = stft.create_window(frame_size)
    frames = stft.get_frames(padded, frame_size, hop_size)
    spectra = stft.calculate_fft(stft.apply_window(frames, window))

    # (frames, bins) * (bins,) broadcasts the one curve across every frame
    filtered = utils.apply_mask(spectra, curve)

    out_frames = stft.calculate_ifft(filtered, frame_size)
    out = stft.overlap_add(out_frames, window, hop_size)

    return out[frame_size:frame_size + len(audio)]


def process(audio, sr, params):
    """
    params example for EQ mode:
      {'mode': 'eq', 'bands': [{'center': 100, 'gain_db': 6, 'bandwidth': 80}, ...]}
    params example for Filter mode:
      {'mode': 'filter', 'filter_type': 'lowpass', 'cutoff': 4000, 'order': 4}
      {'mode': 'filter', 'filter_type': 'bandpass', 'low_cutoff': 300, 'high_cutoff': 3000, 'order': 4}
    params example for both at once (EQ curve x filter curve):
      {'mode': 'both', 'bands': [...], 'filter_type': 'highpass', 'cutoff': 80}
    params example for hum removal (notches at 50, 100, ..., 250 Hz):
      {'mode': 'hum', 'hum_freq': 50, 'harmonics': 5, 'notch_width': 4}
    params example for a preset (see PRESETS):
      {'preset': 'telephone'}
    Returns: (processed_audio, sr)

    audio: float ndarray, shape (samples,) or (samples, channels).
    """
    audio = np.asarray(audio, dtype=np.float64)
    frame_size, hop_size = frame_and_hop(params)
    _, curve = build_curve(sr, params, frame_size)

    def run(x):
        return _process_channel(x, curve, frame_size, hop_size)

    if audio.ndim == 1:
        out = run(audio)
    else:
        out = np.stack([run(audio[:, ch]) for ch in range(audio.shape[1])], axis=1)

    # EQ boosts can push peaks past full scale. Scale down instead of
    # letting the export clamp (distort) them; the clip is the same
    # safety net utils.save_audio uses.
    peak = np.max(np.abs(out)) if out.size else 0
    if peak > 1.0:
        out /= peak
    out = np.clip(out, -1.0, 1.0)

    return out.astype(np.float32), sr
