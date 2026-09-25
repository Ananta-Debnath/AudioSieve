"""What the SPECTRA page shows for each tool: labels, units, slider steps
and the HOW IT WORKS text.

Slider ranges and defaults are NOT written here: they are read from each
effect module's PARAMS (and eq_filter's HUM_PARAMS / DEFAULT_BANDS), the
same tables the routes validate against. The page gets all of this as
one JSON blob (page_config), so the JS never hard-codes a range.
"""

import math

from effects import echo, eq_filter, flanger, reverb

# The EQ's filter fields have no PARAMS table in eq_filter: the DSP takes
# any positive cutoff below Nyquist. These are the slider ranges only.
EQ_FILTER_PARAMS = {
    "cutoff": (20, 20000, 4000),
    "low_cutoff": (20, 20000, 300),
    "high_cutoff": (20, 20000, 3000),
    "order": (1, 8, 4),
}

# Echo plot zoom: teeth are 1000 / delay_ms Hz apart, far too dense to
# see over the whole spectrum. None = up to Nyquist.
ECHO_ZOOMS = [("200 Hz", 200), ("2 kHz", 2000), ("FULL", None)]

# The flanger plot shows this many comb teeth at the longest delay.
FLANGER_PLOT_TEETH = 10


def _decimals(step):
    return max(0, -math.floor(math.log10(step)))


def slider(name, limits, label, unit, step, scale="linear", symbol=None):
    """One slider; limits = (min, max, default) straight from a PARAMS table.
    symbol: the parameter's letter in the equations (shown as-is, not uppercased)."""
    low, high, default = limits
    return {
        "name": name, "label": label, "symbol": symbol, "unit": unit,
        "min": low, "max": high, "default": default,
        "step": step, "decimals": _decimals(step), "scale": scale,
    }


def options(values, labels):
    return [{"value": v, "label": labels.get(v, v.upper())} for v in values]


def _band_label(band):
    center = band["center"]
    return f"{band['name']} {center / 1000:g}k" if center >= 1000 else f"{band['name']} {center}"


def _preset_values(params):
    """A preset as flat slider/option values. Filter-only and hum presets
    reset the band gains to flat, so what the sliders show is what runs."""
    values = {k: v for k, v in params.items() if k != "bands"}
    gains = [b["gain_db"] for b in params.get("bands", eq_filter.DEFAULT_BANDS)]
    if "bands" not in params:
        gains = [0] * len(eq_filter.DEFAULT_BANDS)
    values.update({f"band_{i}_gain_db": g for i, g in enumerate(gains)})
    return values


PRESET_LABELS = {
    "telephone": "Telephone",
    "radio": "Radio",
    "remove_rumble": "Remove rumble",
    "remove_hum_50hz": "Remove hum 50 Hz",
    "bass_boost": "Bass boost",
}


def _eq_config():
    hum = eq_filter.HUM_PARAMS
    return {
        "modes": options(eq_filter.MODES, {"both": "EQ + FILTER"}),
        "default_mode": "eq",
        "filter_types": options(eq_filter.FILTER_TYPES, {
            "lowpass": "LOW-PASS", "highpass": "HIGH-PASS",
            "bandpass": "BAND-PASS", "bandstop": "BAND-STOP",
        }),
        "default_filter_type": "lowpass",
        "bands": [
            slider(f"band_{i}_gain_db",
                   (-eq_filter.MAX_BAND_GAIN_DB, eq_filter.MAX_BAND_GAIN_DB, band["gain_db"]),
                   _band_label(band), "dB", 0.5)
            for i, band in enumerate(eq_filter.DEFAULT_BANDS)
        ],
        "filter": [
            slider("cutoff", EQ_FILTER_PARAMS["cutoff"], "Cutoff", "Hz", 1, scale="log", symbol="fc"),
            slider("low_cutoff", EQ_FILTER_PARAMS["low_cutoff"], "Low cutoff", "Hz", 1, scale="log"),
            slider("high_cutoff", EQ_FILTER_PARAMS["high_cutoff"], "High cutoff", "Hz", 1, scale="log"),
            slider("order", EQ_FILTER_PARAMS["order"], "Order", "", 1, symbol="N"),
        ],
        "hum": [
            slider("hum_freq", hum["hum_freq"], "Hum frequency", "Hz", 1),
            slider("harmonics", hum["harmonics"], "Harmonics", "", 1),
            slider("notch_width", hum["notch_width"], "Notch width", "Hz", 0.5),
        ],
        "presets": [
            {"name": name, "label": PRESET_LABELS.get(name, name.replace("_", " ")),
             "values": _preset_values(params)}
            for name, params in eq_filter.PRESETS.items()
        ],
        "processing_frame": {"frame_size": eq_filter.FRAME_SIZE, "hop_size": eq_filter.HOP_SIZE},
    }


# Per tool: tab number and name, subtitle, governing equation(s), and
# the HOW IT WORKS drawer text (a few sentences + one line per parameter).
TOOLS = [
    {
        "id": "eq",
        "number": "01",
        "name": "EQ + Filter",
        "subtitle": "Multiply every frequency by a gain curve.",
        "equations": [
            "Y[k] = H[k] · X[k]",
            "|H(f)| = 1 / √(1 + (f/fc)^(2N))",
        ],
        "explain": [
            "The track is cut into long overlapping frames, and the FFT turns each "
            "one into a spectrum X[k].",
            "Every frequency bin is multiplied by one gain curve H[k], then the inverse "
            "FFT and overlap-add turn the frames back into sound.",
            "A filter removes a band: past the cutoff, the Butterworth curve falls "
            "6N dB per octave.",
            "An EQ reshapes the balance instead, with gentle boosts and cuts of at "
            "most 15 dB.",
            "Frames are 65536 samples long, so bins sit 0.67 Hz apart and even a "
            "50 Hz hum notch lands where the plot shows it.",
        ],
        "param_notes": [
            ("Mode", "EQ bands, a filter, both curves multiplied, or hum notches."),
            ("Band gains", "Boost (+) or cut (−) around each band; bass and treble are shelves."),
            ("Filter type", "Which side of the cutoff survives (band-stop removes the middle)."),
            ("Cutoff", "Where the filter curve is −3 dB."),
            ("Order N", "Steepness: each step adds 6 dB per octave of roll-off."),
            ("Hum", "Mains frequency (50 or 60 Hz), how many harmonics, notch width."),
        ],
    },
    {
        "id": "reverb",
        "number": "02",
        "name": "Reverb",
        "subtitle": "Convolve the track with a room's impulse response.",
        "equations": ["y = x * h   ⟷   Y = X · H"],
        "explain": [
            "A room is a linear, time-invariant system, so its impulse response h "
            "(the echo pattern of one click) describes everything it does to sound.",
            "Reverb is the track convolved with h, computed with the convolution "
            "theorem: convolution in time is multiplication of spectra.",
            "Our h is synthetic: a silent pre-delay, then white noise that decays "
            "by 60 dB over RT60 seconds.",
            "The FFT computes circular convolution, so both signals are zero-padded "
            "to N + M − 1 samples; without that padding, the tail wraps onto the start.",
        ],
        "param_notes": [
            ("RT60", "Time for the reverb to decay by 60 dB: bigger means a larger room."),
            ("Pre-delay", "Gap before the first reflection arrives."),
            ("Wet", "Reverb against dry track: 0 is dry only, 1 is reverb only."),
            ("Linear / circular", "Correct zero-padded convolution, or the wrong way."),
        ],
    },
    {
        "id": "echo",
        "number": "03",
        "name": "Echo + Delay",
        "subtitle": "Add delayed copies with a difference equation.",
        "equations": [
            "feedforward:",
            "  y[n] = x[n] + g · x[n − D]",
            "feedback:",
            "  y[n] = x[n] + g · y[n − D]",
        ],
        "explain": [
            "An echo is a delayed, quieter copy of the signal, written as a "
            "difference equation over samples n.",
            "Feedforward adds one copy of the input D samples later: a single "
            "repeat, an FIR system.",
            "Feedback adds a copy of the output instead, so every echo is echoed "
            "again and the repeats decay as g, g², g³… (an IIR system).",
            "Feedback is stable only for g < 1; at g ≥ 1 the repeats would never die out.",
            "In frequency both are comb filters, with peaks every 1/D Hz where the "
            "copies add in phase.",
        ],
        "param_notes": [
            ("Delay", "Time D between the signal and each repeat."),
            ("Gain g", "Level of each repeat relative to the one before."),
            ("Mix", "How much of the echoed signal replaces the dry one."),
            ("Mode", "Feedforward (one repeat) or feedback (decaying repeats)."),
        ],
    },
    {
        "id": "flanger",
        "number": "04",
        "name": "Flanger",
        "subtitle": "A comb filter whose notches sweep the spectrum.",
        "equations": ["y[n] = x[n] + g · x[n − D(n)]"],
        "explain": [
            "A flanger is the feedforward echo with a delay of only a few "
            "milliseconds, too short to hear as a repeat.",
            "At such short delays the comb's notches fall in the audible range, at "
            "odd multiples of 1/(2D) Hz.",
            "A low-frequency oscillator (LFO) sweeps D(n) up and down, so the notches "
            "sweep through the spectrum: the jet-plane whoosh.",
            "D(n) usually falls between two samples, so x[n − D(n)] is interpolated "
            "linearly from its neighbours.",
        ],
        "param_notes": [
            ("Min delay", "Shortest delay of the sweep: sets the highest notches."),
            ("Sweep", "How far D travels above the minimum."),
            ("Rate", "LFO sweeps per second."),
            ("Gain g", "Notch depth: g = 1 cancels completely at each notch."),
        ],
    },
    {
        "id": "separate",
        "number": "05",
        "name": "Separation",
        "subtitle": "Split a mix into stems with NMF and spectral masks.",
        "equations": [
            "V ≈ W · H",
            "mask = stem estimate / mixture",
        ],
        "explain": [
            "Non-negative matrix factorisation (NMF) splits the mixture's magnitude "
            "spectrogram V into W · H.",
            "Each column of W is a spectral template, such as a drum hit or a bass "
            "note, and each row of H says how strongly it plays over time.",
            "Each instrument's share of the mixture becomes a soft mask from 0 to 1, "
            "applied to the mixture's STFT to give that stem.",
            "The known limitation is bleed: where two instruments overlap in "
            "frequency at the same moment, the mask cannot split them.",
        ],
        "param_notes": [
            ("Stems", "Percussion, bass, vocals and harmonics: the pitched sound that is left "
                      "(guitar, keys, strings)."),
            ("Volume", "Each stem's playback level, 0 to 100 %; nothing is re-processed."),
            ("Mute / solo", "Silence a stem, or hear only the soloed ones."),
            ("Play all", "Plays every stem in sync, so muting one reveals the rest."),
        ],
    },
]


def page_config(limits):
    """Everything the page's JS needs, as one JSON-serializable dict.

    limits: the upload limits the /upload route enforces.
    """
    return {
        "limits": limits,
        "tools": [
            {k: tool[k] for k in ("id", "number", "name", "subtitle", "equations")}
            for tool in TOOLS
        ],
        "eq": _eq_config(),
        "reverb": {
            "sliders": [
                slider("rt60", reverb.PARAMS["rt60"], "RT60", "s", 0.01),
                slider("pre_delay_ms", reverb.PARAMS["pre_delay_ms"], "Pre-delay", "ms", 1),
                slider("wet", reverb.PARAMS["wet"], "Wet", "", 0.01),
            ],
            "circular": options(["false", "true"], {
                "false": "LINEAR", "true": "CIRCULAR (DEMO: WRONG WAY)",
            }),
        },
        "echo": {
            "sliders": [
                slider("delay_ms", echo.PARAMS["delay_ms"], "Delay", "ms", 1, symbol="D"),
                slider("gain", echo.PARAMS["gain"], "Gain", "", 0.01, symbol="g"),
                slider("mix", echo.PARAMS["mix"], "Mix", "", 0.01),
            ],
            "modes": options(echo.MODES, {}),
            "default_mode": "feedback",
            "zooms": [{"label": label, "f_max": f_max} for label, f_max in ECHO_ZOOMS],
        },
        "flanger": {
            "sliders": [
                slider("min_delay_ms", flanger.PARAMS["min_delay_ms"], "Min delay", "ms", 0.1),
                slider("sweep_ms", flanger.PARAMS["sweep_ms"], "Sweep", "ms", 0.1),
                slider("rate_hz", flanger.PARAMS["rate_hz"], "Rate", "Hz", 0.05),
                slider("gain", flanger.PARAMS["gain"], "Gain", "", 0.01, symbol="g"),
            ],
            "plot_teeth": FLANGER_PLOT_TEETH,
        },
    }
