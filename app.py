"""SPECTRA signal processing lab: Flask app.

Synchronous request -> process -> response. Each tool is independent:
upload a file once, then run any tool on it by id.
"""

import json
import os
import shutil
import threading
import uuid
from pathlib import Path

import matplotlib

# Spectrograms are rendered inside request threads; no GUI backend.
# Must be set before anything imports pyplot (utils does).
matplotlib.use("Agg")

import numpy as np
import soundfile as sf
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

import runs
import stft
import ui_config
import utils
from analysis.waveform import waveform_json
from effects import echo, eq_filter, flanger, reverb

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DEMO_PATH = BASE_DIR / "static" / "demo" / "demo.wav"

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}
MAX_UPLOAD_MB = 50
MAX_DURATION_SEC = 6 * 60

# /response/* plots depend only on the parameters, not on an upload.
RESPONSE_SR = 44100
MAX_PLOT_POINTS = 4000
# The flanger plot is one curve per delay over a sweep, so fewer points each.
FLANGER_PLOT_FRAMES = 48
FLANGER_PLOT_POINTS = 600

# Before/after spectrograms: short frames for time detail (not the EQ's
# long processing frames), and their colour range (see save_spectrograms).
SPECTROGRAM_FRAME_SIZE = 1024
SPECTROGRAM_HOP_SIZE = 512
SPECTROGRAM_RANGE_DB = 90

# The shared plot code (utils.save_spectrogram) takes its colours from
# matplotlib's defaults; these make it a magma image on a dark
# background, which the UI shows in both themes.
SPECTROGRAM_STYLE = {
    "image.cmap": "magma",
    "figure.facecolor": "#111315",
    "axes.facecolor": "#111315",
    "savefig.facecolor": "#111315",
    "text.color": "#E8EAEC",
    "axes.labelcolor": "#E8EAEC",
    "axes.edgecolor": "#9AA0A6",
    "xtick.color": "#9AA0A6",
    "ytick.color": "#9AA0A6",
}
# pyplot's current-figure state (and rcParams) are global, so only one
# request thread may draw at a time.
PLOT_LOCK = threading.Lock()

# /response/eq: log-spaced grid over the audible range.
EQ_RESPONSE_POINTS = 1000
EQ_RESPONSE_RANGE_HZ = (20, 20000)

# Tool name (as used in /process/<tool>) -> effect module.
# Later stages only fill in the modules' process() functions.
EFFECTS = {
    "eq": eq_filter,
    "reverb": reverb,
    "echo": echo,
    "flanger": flanger,
}

# SEPARATION_MOCK=1: /process/separate returns the real contract, with
# copies of the input as these stems, until the separation module is merged.
SEPARATION_MOCK_STEMS = ("drums", "bass", "rest")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# One registry per output directory (tests point PROCESSED_DIR elsewhere).
_registries = {}


def run_registry():
    if PROCESSED_DIR not in _registries:
        _registries[PROCESSED_DIR] = runs.RunRegistry(PROCESSED_DIR)
    return _registries[PROCESSED_DIR]


def error(message, status=400):
    return jsonify({"error": message}), status


def is_valid_id(file_id):
    try:
        return str(uuid.UUID(file_id)) == file_id
    except (ValueError, TypeError, AttributeError):
        return False


def find_upload(file_id):
    """Return the path of an uploaded file by id, or None."""
    if not is_valid_id(file_id):
        return None
    for ext in ALLOWED_EXTENSIONS:
        path = UPLOAD_DIR / f"{file_id}.{ext}"
        if path.exists():
            return path
    return None


def upload_info(file_id):
    """What /upload returned for file_id (kept in a JSON sidecar), or {}."""
    try:
        return json.loads((UPLOAD_DIR / f"{file_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def request_params():
    """Collect request parameters from a JSON body or form fields."""
    if request.is_json:
        return dict(request.get_json(silent=True) or {})
    return request.form.to_dict()


def _optional_float(params, key):
    value = params.get(key)
    if value is None or value == "":
        return None
    return float(value)


def eq_params(raw):
    """Turn raw request params into the dict eq_filter.process() expects.

    A non-empty 'preset' wins over everything else.
    Filter fields (modes 'filter' and 'both'): filter_type, cutoff,
    low_cutoff, high_cutoff, order.
    EQ bands (modes 'eq' and 'both'): either a JSON 'bands' list, or flat
    form fields band_<i>_gain_db applied to eq_filter.DEFAULT_BANDS.
    Hum fields (mode 'hum'): see eq_filter.HUM_PARAMS.
    """
    if raw.get("preset"):
        return {"preset": raw["preset"]}

    mode = raw.get("mode") or "eq"
    if mode not in eq_filter.MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: {', '.join(eq_filter.MODES)}."
        )
    params = {"mode": mode}

    if mode == "hum":
        return {**params, **ranged_params(raw, eq_filter.HUM_PARAMS)}

    if mode in ("filter", "both"):
        order = _optional_float(raw, "order")
        params.update({
            "filter_type": raw.get("filter_type") or "lowpass",
            "cutoff": _optional_float(raw, "cutoff"),
            "low_cutoff": _optional_float(raw, "low_cutoff"),
            "high_cutoff": _optional_float(raw, "high_cutoff"),
            "order": 4 if order is None else order,
        })

    if mode in ("eq", "both"):
        if isinstance(raw.get("bands"), list):
            params["bands"] = [
                {
                    "type": b.get("type", "peak"),
                    "center": float(b["center"]),
                    "gain_db": float(b.get("gain_db", 0)),
                    "bandwidth": float(b.get("bandwidth", 0)),
                }
                for b in raw["bands"]
            ]
        else:
            params["bands"] = [
                {**band, "gain_db": _optional_float(raw, f"band_{i}_gain_db") or 0.0}
                for i, band in enumerate(eq_filter.DEFAULT_BANDS)
            ]

    return params


def ranged_params(raw, ranges):
    """Read float params described by ranges = {name: (min, max, default)}.

    Missing params get their default; values outside [min, max] raise
    ValueError.
    """
    params = {}
    for key, (low, high, default) in ranges.items():
        value = _optional_float(raw, key)
        if value is None:
            value = default
        elif not low <= value <= high:
            raise ValueError(f"{key} must be between {low:g} and {high:g} (got {value:g}).")
        params[key] = value
    return params


def reverb_params(raw):
    """rt60, pre_delay_ms, wet (see reverb.PARAMS) and the circular toggle."""
    circular = str(raw.get("circular", "")).lower() in ("1", "true", "on", "yes")
    return {**ranged_params(raw, reverb.PARAMS), "circular": circular}


def flanger_params(raw):
    """min_delay_ms, sweep_ms, rate_hz, gain (see flanger.PARAMS)."""
    return ranged_params(raw, flanger.PARAMS)


def f_max_param(raw):
    """Optional f_max (Hz) for the /response/* zoom; must be positive."""
    f_max = _optional_float(raw, "f_max")
    if f_max is not None and not f_max > 0:
        raise ValueError("f_max must be positive.")
    return f_max


def echo_params(raw):
    """delay_ms, gain, mix (see echo.PARAMS) and mode."""
    mode = raw.get("mode") or "feedback"
    if mode not in echo.MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: {', '.join(echo.MODES)}."
        )
    return {**ranged_params(raw, echo.PARAMS), "mode": mode}


def _mono_magnitude(audio):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    spectra = stft.calculatr_stft(mono, SPECTROGRAM_FRAME_SIZE, SPECTROGRAM_HOP_SIZE)
    return utils.calculate_magnitude(spectra)


def save_spectrograms(before, after, sr, result_id):
    """Before/after spectrograms (mono mix) via the shared STFT/plot code.

    Reverb and echo tails make 'after' longer, so the shorter one is
    padded with silence to put both on the same time axis. The plot
    scales its colours from its quietest to its loudest bin, and digital
    silence (e.g. that padding) is -200 dB, which would wash out
    everything else; both are floored at the same level,
    SPECTROGRAM_RANGE_DB below the louder peak, instead.
    """
    length = max(len(before), len(after))
    mags = {}
    for kind, audio in (("before", before), ("after", after)):
        pad = [(0, length - len(audio))] + [(0, 0)] * (audio.ndim - 1)
        mags[kind] = _mono_magnitude(np.pad(audio, pad))

    floor = max(mag.max() for mag in mags.values()) * 10 ** (-SPECTROGRAM_RANGE_DB / 20)
    with PLOT_LOCK, matplotlib.rc_context(SPECTROGRAM_STYLE):
        for kind, mag in mags.items():
            utils.save_spectrogram(
                np.maximum(mag, floor), sr, SPECTROGRAM_HOP_SIZE,
                str(PROCESSED_DIR / f"{result_id}_{kind}.png"),
            )


def write_result(result_id, audio, sr):
    sf.write(str(PROCESSED_DIR / f"{result_id}.wav"), audio, sr, subtype="FLOAT")


def result_urls(result_id):
    return {
        "url": url_for("result", result_id=result_id),
        "download_url": url_for("result", result_id=result_id, download=1),
    }


def eq_response_freqs(params):
    """Log-spaced grid over EQ_RESPONSE_RANGE_HZ. In hum mode the notch
    centres and -3 dB edges are added, so notches only a few Hz wide
    show up at their full depth however coarse the grid is there."""
    freqs = np.geomspace(*EQ_RESPONSE_RANGE_HZ, EQ_RESPONSE_POINTS)
    if params.get("mode") == "hum":
        centres = params["hum_freq"] * np.arange(1, int(params["harmonics"]) + 1)
        half = params["notch_width"] / 2
        freqs = np.union1d(freqs, np.concatenate([centres - half, centres, centres + half]))
    return freqs


def eq_curve_at(freqs, sr, params):
    """eq_filter.build_curve's gain curve, evaluated at any frequencies
    (build_curve only evaluates at one frame's FFT bins). Same functions,
    same validation."""
    mode = params.get("mode", "eq")
    if mode == "hum":
        return eq_filter._hum_curve_from_params(freqs, sr, params)
    curve = np.ones_like(freqs)
    if mode in ("eq", "both"):
        curve *= eq_filter.create_eq_curve(freqs, params.get("bands", eq_filter.DEFAULT_BANDS))
    if mode in ("filter", "both"):
        curve *= eq_filter._filter_curve_from_params(freqs, sr, params)
    return curve


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    return error(f"File too large (max {MAX_UPLOAD_MB} MB).", 413)


@app.get("/")
def index():
    tools = {tool["id"]: tool for tool in ui_config.TOOLS}
    config = ui_config.page_config({
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_duration_sec": MAX_DURATION_SEC,
        "extensions": sorted(ALLOWED_EXTENSIONS),
    })
    return render_template("index.html", tools=tools, config=config)


def register_upload(filename, save):
    """Validate and register one audio file; save(path) writes it to path.

    Shared by /upload and /upload/demo, so the demo track goes through
    exactly the same checks and gets an ordinary file_id.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(f".{e}" for e in sorted(ALLOWED_EXTENSIONS))
        return error(f"Unsupported format '.{ext}'. Allowed: {allowed}.")

    file_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{file_id}.{ext}"
    save(path)

    try:
        info = sf.info(str(path))
    except Exception:
        path.unlink(missing_ok=True)
        return error("Could not read the file as audio. Is it a valid WAV/MP3/FLAC?")

    if info.duration > MAX_DURATION_SEC:
        path.unlink(missing_ok=True)
        return error(
            f"Audio too long ({info.duration / 60:.1f} min). "
            f"Max is {MAX_DURATION_SEC // 60} min."
        )

    meta = {
        "file_id": file_id,
        "filename": filename,
        "duration": info.duration,
        "samplerate": info.samplerate,
        "channels": info.channels,
    }
    (UPLOAD_DIR / f"{file_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return jsonify({**meta, "url": url_for("uploaded_audio", file_id=file_id)})


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return error("No file provided (expected form field 'file').")
    return register_upload(file.filename, file.save)


@app.post("/upload/demo")
def upload_demo():
    """Register the shared demo track like any upload; returns a file_id."""
    if not DEMO_PATH.exists():
        return error(
            "Demo track missing (static/demo/demo.wav). "
            "Run: python scripts/make_placeholder_demo.py", 404,
        )
    return register_upload(DEMO_PATH.name, lambda path: shutil.copyfile(DEMO_PATH, path))


@app.get("/upload/<file_id>")
def uploaded_audio(file_id):
    """The uploaded file itself, for the ORIGINAL player."""
    path = find_upload(file_id)
    if path is None:
        abort(404)
    return send_file(path)


@app.get("/upload/<file_id>/waveform")
def uploaded_waveform(file_id):
    """Min/max envelope and duration of an upload, for its ORIGINAL player."""
    path = find_upload(file_id)
    if path is None:
        abort(404)
    audio, sr = sf.read(str(path), dtype="float32")
    return jsonify(waveform_json(audio, sr))


@app.post("/process/separate")
def process_separate():
    if os.environ.get("SEPARATION_MOCK") != "1":
        return error(
            "Separation is not implemented in the app shell yet; "
            "it is being developed on the separation branch.",
            501,
        )

    params = request_params()
    file_id = params.pop("file_id", None)
    src = find_upload(file_id)
    if src is None:
        return error("Unknown or missing file_id. Upload a file first.", 404)

    # Mock: every "stem" is a copy of the input.
    audio, sr = sf.read(str(src), dtype="float32")
    run_id = str(uuid.uuid4())
    stems = []
    for name in SEPARATION_MOCK_STEMS:
        stem_id = str(uuid.uuid4())
        write_result(stem_id, audio, sr)
        stems.append({"name": name, "file": f"{stem_id}.wav", **result_urls(stem_id)})

    run_registry().add(
        run_id, "separate", {"mock": True}, file_id,
        {"stems": [{"name": s["name"], "file": s["file"]} for s in stems]},
        sr=sr, source=upload_info(file_id),
    )
    wave = waveform_json(audio, sr)  # the same for every copy
    return jsonify({
        "run_id": run_id,
        "input_url": url_for("uploaded_audio", file_id=file_id),
        "stems": [
            {"name": s["name"], "url": s["url"], "download_url": s["download_url"], **wave}
            for s in stems
        ],
    })


@app.post("/process/<tool>")
def process_tool(tool):
    effect = EFFECTS.get(tool)
    if effect is None:
        return error(f"Unknown tool '{tool}'.", 404)

    params = request_params()
    file_id = params.pop("file_id", None)
    src = find_upload(file_id)
    if src is None:
        return error("Unknown or missing file_id. Upload a file first.", 404)

    # float32 keeps decoded samples exact for 16/24-bit sources and
    # leaves headroom (no wrap-around) once effects can exceed [-1, 1].
    audio, sr = sf.read(str(src), dtype="float32")
    result_id = str(uuid.uuid4())
    extra = {}

    if tool == "eq":
        try:
            params = eq_params(params)
            freq_bins, curve = eq_filter.build_curve(sr, params)
            processed, sr = eq_filter.process(audio, sr, params)
        except (ValueError, TypeError, KeyError) as e:
            return error(f"Invalid EQ/filter parameters: {e}")

        extra["curve"] = {
            "freqs": freq_bins.round(1).tolist(),
            "gain_db": (20 * np.log10(np.maximum(curve, 1e-6))).round(2).tolist(),
        }
    else:
        parse_params = {
            "reverb": reverb_params, "echo": echo_params, "flanger": flanger_params,
        }[tool]
        try:
            params = parse_params(params)
        except (ValueError, TypeError) as e:
            return error(f"Invalid {tool} parameters: {e}")
        processed, sr = effect.process(audio, sr, **params)

    save_spectrograms(audio, processed, sr, result_id)
    write_result(result_id, processed, sr)

    # The result id doubles as the run id.
    run_registry().add(
        result_id, tool, params, file_id,
        {
            "audio": f"{result_id}.wav",
            "spectrograms": {kind: f"{result_id}_{kind}.png" for kind in ("before", "after")},
        },
        sr=sr, source=upload_info(file_id),
    )

    return jsonify({
        "result_id": result_id,
        "run_id": result_id,
        **result_urls(result_id),
        "input_url": url_for("uploaded_audio", file_id=file_id),
        "spectrograms": {
            kind: url_for("result_spectrogram", result_id=result_id, kind=kind)
            for kind in ("before", "after")
        },
        "waveforms": {
            "before": waveform_json(audio, sr),
            "after": waveform_json(processed, sr),
        },
        **extra,
    })


@app.get("/result/<result_id>")
def result(result_id):
    if not is_valid_id(result_id):
        abort(404)
    path = PROCESSED_DIR / f"{result_id}.wav"
    if not path.exists():
        abort(404)
    return send_file(
        path,
        mimetype="audio/wav",
        as_attachment=request.args.get("download") == "1",
        download_name=f"spectra_{result_id[:8]}.wav",
    )


@app.get("/result/<result_id>/spectrogram/<kind>.png")
def result_spectrogram(result_id, kind):
    if not is_valid_id(result_id) or kind not in ("before", "after"):
        abort(404)
    path = PROCESSED_DIR / f"{result_id}_{kind}.png"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.get("/response/eq")
def response_eq():
    """Gain curve for the same params as /process/eq (preset, mode, band
    gains, filter and hum fields), as {"freqs": [Hz], "gain_db": [...]}
    on a log-spaced grid from 20 Hz to 20 kHz."""
    try:
        params = eq_filter.resolve_params(eq_params(request.args))
        freqs = eq_response_freqs(params)
        curve = eq_curve_at(freqs, RESPONSE_SR, params)
    except (ValueError, TypeError, KeyError) as e:
        return error(f"Invalid EQ/filter parameters: {e}")

    return jsonify({
        "freqs": freqs.round(2).tolist(),
        "gain_db": (20 * np.log10(np.maximum(curve, 1e-6))).round(2).tolist(),
    })


@app.get("/response/reverb")
def response_reverb():
    """Impulse response for ?rt60=&pre_delay_ms=, as {"t": [s], "h": [...]}."""
    try:
        params = reverb_params(request.args)
    except (ValueError, TypeError) as e:
        return error(f"Invalid reverb parameters: {e}")

    h = reverb.generate_impulse_response(
        RESPONSE_SR, params["rt60"], params["pre_delay_ms"]
    )
    step = -(-len(h) // MAX_PLOT_POINTS)  # ceil: plain decimation to <= MAX_PLOT_POINTS
    t = np.arange(0, len(h), step) / RESPONSE_SR
    return jsonify({"t": t.round(5).tolist(), "h": h[::step].round(6).tolist()})


@app.get("/response/echo")
def response_echo():
    """Frequency response for ?delay_ms=&gain=&mode=, as {"freqs": [Hz], "mag_db": [...]}.

    Optional f_max (Hz) zooms into [0, f_max], so the comb teeth stay
    visible for long delays; default is up to Nyquist.
    """
    try:
        params = echo_params(request.args)
        f_max = f_max_param(request.args)
    except (ValueError, TypeError) as e:
        return error(f"Invalid echo parameters: {e}")

    freqs, mag_db = echo.echo_frequency_response(
        RESPONSE_SR, params["delay_ms"], params["gain"], params["mode"],
        n_points=MAX_PLOT_POINTS, f_max=f_max,
    )
    return jsonify({"freqs": freqs.round(3).tolist(), "mag_db": mag_db.round(2).tolist()})


@app.get("/response/flanger")
def response_flanger():
    """Frequency responses over one sweep, for ?min_delay_ms=&sweep_ms=&gain=,
    as {"freqs": [Hz], "delays_ms": [...], "mag_db": [[...], ...]}, one
    mag_db row per delay (the page animates through them).

    Optional f_max (Hz) zooms into [0, f_max], as for /response/echo.
    """
    try:
        params = flanger_params(request.args)
        f_max = f_max_param(request.args)
    except (ValueError, TypeError) as e:
        return error(f"Invalid flanger parameters: {e}")

    freqs, delays_ms, mag_db = flanger.flanger_frequency_response(
        RESPONSE_SR, params["min_delay_ms"], params["sweep_ms"], params["gain"],
        n_points=FLANGER_PLOT_POINTS, n_frames=FLANGER_PLOT_FRAMES, f_max=f_max,
    )
    return jsonify({
        "freqs": freqs.round(3).tolist(),
        "delays_ms": delays_ms.round(4).tolist(),
        "mag_db": mag_db.round(1).tolist(),
    })


if __name__ == "__main__":
    app.run(debug=True)
