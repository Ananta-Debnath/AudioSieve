"""SPECTRA signal processing lab: Flask app.

Synchronous request -> process -> response. Each tool is independent:
upload a file once, then run any tool on it by id.
"""

import json
import logging
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
from werkzeug.exceptions import InternalServerError, RequestEntityTooLarge

import runs
import stft
import ui_config
import utils
from analysis import backstage
from analysis.waveform import waveform_json
from effects import echo, eq_filter, flanger, reverb, separation

log = logging.getLogger("spectra")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DEMO_PATH = BASE_DIR / "static" / "demo" / "demo.wav"
# Separation's demo is a song with vocals (the placeholder above has
# none): 30 s of Karissa Hobbs' "Let's Go Fishin'" (static/demo/CREDITS.txt).
VOCALS_DEMO_PATH = BASE_DIR / "static" / "demo" / "demo_vocals.flac"
VOCALS_DEMO_NAME = "Let's Go Fishin' - Karissa Hobbs.flac"
# Landing page previews (static files; see scripts/make_landing_assets.py).
# Separation is "separation" here and in /lab#separation links.
LANDING_DIR = BASE_DIR / "static" / "landing"
LANDING_TOOLS = ("eq", "reverb", "echo", "flanger", "separation")

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}
MAX_UPLOAD_MB = 200
MAX_DURATION_SEC = 20 * 60

# Every file is decoded by soundfile's libsndfile, ffmpeg is never used.
# libsndfile reads MP3 from version 1.1 on (the soundfile wheels bundle
# such a build); WAV and FLAC work with any build.
MP3_SUPPORTED = "MP3" in sf.available_formats()
MP3_MISSING = (
    "MP3 is not supported on this server: its libsndfile has no MP3 decoder "
    "(it needs libsndfile 1.1 or newer; pip install -r requirements.txt brings one)."
)

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

# SEPARATION_MOCK=1: /process/separate skips the real separation and
# returns copies of the input as the stems (a fallback, off by default).
SEPARATION_MOCK_STEMS = separation.STEMS

# Backstage JSON is cached per run; bump this when its shape changes so
# old caches are ignored.
BACKSTAGE_VERSION = 1

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

if not MP3_SUPPORTED:
    log.warning("%s WAV and FLAC uploads still work.", MP3_MISSING)

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


@app.errorhandler(InternalServerError)
def internal_error(e):
    """A crash inside a route: the page shows the reason, not "HTTP 500"
    (the traceback is still logged)."""
    cause = e.original_exception or e
    return error(f"Server error: {type(cause).__name__}: {cause}", 500)


def landing_previews():
    """The reel's preview assets (made by scripts/make_landing_assets.py),
    per tool: the clip's metadata plus the clip and card background URLs.
    A tool without a generated clip is left out, and its card says so.
    Read here, so the page itself makes no requests for them."""
    previews = {}
    for tool in LANDING_TOOLS:
        try:
            meta = json.loads((LANDING_DIR / "clips" / f"{tool}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        preview = {**meta, "clip_url": url_for("static", filename=f"landing/clips/{tool}.mp3")}
        for ext in ("svg", "png"):
            if (LANDING_DIR / "responses" / f"{tool}.{ext}").exists():
                preview["background_url"] = url_for("static", filename=f"landing/responses/{tool}.{ext}")
                break
        previews[tool] = preview
    return previews


def landing_hero():
    """The demo track's spectrum over time for the hero's backdrop (see
    make_landing_assets.py), or None: the page then draws the grid only."""
    try:
        return json.loads((LANDING_DIR / "hero.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@app.get("/")
def landing():
    page = {"lab_url": url_for("lab"), "previews": landing_previews(), "hero": landing_hero()}
    return render_template("landing.html", page=page)


@app.get("/lab")
def lab():
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
    if ext == "mp3" and not MP3_SUPPORTED:
        return error(f"{MP3_MISSING} Upload a WAV or FLAC file instead.", 415)

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


@app.post("/upload/demo/vocals")
def upload_vocals_demo():
    """Register Separation's demo song like any upload; returns a file_id."""
    if not VOCALS_DEMO_PATH.exists():
        return error("Demo song missing (static/demo/demo_vocals.flac).", 404)
    return register_upload(VOCALS_DEMO_NAME, lambda path: shutil.copyfile(VOCALS_DEMO_PATH, path))


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
    """Separate the whole track. That takes about 0.4 s per second of
    stereo audio here, so a long track keeps the request busy for minutes."""
    params = request_params()
    file_id = params.pop("file_id", None)
    src = find_upload(file_id)
    if src is None:
        return error("Unknown or missing file_id. Upload a file first.", 404)

    audio, sr = sf.read(str(src), dtype="float32")

    mock = os.environ.get("SEPARATION_MOCK") == "1"
    if mock:
        # Every "stem" is a copy of the input.
        stems = {name: audio for name in SEPARATION_MOCK_STEMS}
    else:
        stems = separation.separate(audio, sr)

    run_id = str(uuid.uuid4())
    saved = []
    for name, stem in stems.items():
        stem_id = str(uuid.uuid4())
        write_result(stem_id, stem, sr)
        saved.append({"name": name, "file": f"{stem_id}.wav", "audio": stem, **result_urls(stem_id)})

    run_registry().add(
        run_id, "separate", {"mock": mock}, file_id,
        {"stems": [{"name": s["name"], "file": s["file"]} for s in saved]},
        sr=sr, source=upload_info(file_id),
    )
    return jsonify({
        "run_id": run_id,
        "input_url": url_for("uploaded_audio", file_id=file_id),
        "stems": [
            {"name": s["name"], "url": s["url"], "download_url": s["download_url"],
             **waveform_json(s["audio"], sr)}
            for s in saved
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


# ---------------------------------------------------------------------
# Backstage: analysis of one run (read-only; see analysis/backstage.py)
# ---------------------------------------------------------------------

def registered_run(run_id):
    run = run_registry().get(run_id) if is_valid_id(run_id) else None
    if run is None:
        abort(404)
    return run


def run_input(run):
    src = find_upload(run["file_id"])
    if src is None:
        abort(404)
    return src


def stem_files(run):
    return [(s["name"], PROCESSED_DIR / s["file"]) for s in run["outputs"]["stems"]]


def backstage_urls(run):
    """Image and audio URLs for the run's panels."""
    run_id = run["run_id"]
    if run["tool"] == "separate":
        return {
            "images": {
                "mixture": url_for("backstage_mixture", run_id=run_id),
                "masks": {name: url_for("backstage_mask", run_id=run_id, stem=name)
                          for name, _ in stem_files(run)},
            },
            "audio": {
                "input": url_for("uploaded_audio", file_id=run["file_id"]),
                "stems": {name: result_urls(path.stem) for name, path in stem_files(run)},
            },
        }
    return {
        # The before/after spectrograms saved when the run was processed.
        "images": {
            "before": url_for("result_spectrogram", result_id=run_id, kind="before"),
            "after": url_for("result_spectrogram", result_id=run_id, kind="after"),
            "diff": url_for("backstage_diff", run_id=run_id),
        },
        "audio": {
            "input": url_for("uploaded_audio", file_id=run["file_id"]),
            "output": result_urls(run_id),
        },
    }


def write_atomic(path, text):
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@app.get("/backstage/runs")
def backstage_runs():
    """Every registered run, newest first."""
    return jsonify({"runs": run_registry().list()})


@app.get("/backstage/<run_id>")
def backstage_run(run_id):
    """Analysis of one run: levels, envelopes, average spectra, the
    difference stats, STFT settings and tool-specific extras, plus the
    run itself and its image URLs. Computed once, then cached on disk."""
    run = run_registry().get(run_id) if is_valid_id(run_id) else None
    if run is None:
        return error("Unknown run.", 404)
    src = find_upload(run["file_id"])
    if src is None:
        return error("This run's input file is no longer on the server.", 404)

    cache = PROCESSED_DIR / f"{run_id}.backstage-v{BACKSTAGE_VERSION}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        try:
            if run["tool"] == "separate":
                data = backstage.analyze_separation(run, src, stem_files(run))
            else:
                data = backstage.analyze_effect(
                    run, src, PROCESSED_DIR / run["outputs"]["audio"],
                    PROCESSED_DIR / f"{run_id}_diff.png",
                )
        except (OSError, RuntimeError) as e:  # includes soundfile's read errors
            return error(f"Could not read this run's audio files ({e}).", 404)
        write_atomic(cache, json.dumps(data))

    return jsonify({"run": run, **data, **backstage_urls(run)})


@app.get("/backstage/<run_id>/diff.png")
def backstage_diff(run_id):
    """Difference spectrogram of an effect run (cached after the first request)."""
    run = registered_run(run_id)
    if run["tool"] == "separate":
        abort(404)
    path = PROCESSED_DIR / f"{run_id}_diff.png"
    if not path.exists():
        backstage.render_diff(run_input(run), PROCESSED_DIR / run["outputs"]["audio"], path)
    return send_file(path, mimetype="image/png")


@app.get("/backstage/<run_id>/mixture.png")
def backstage_mixture(run_id):
    """Spectrogram of a separation run's mixture (cached)."""
    run = registered_run(run_id)
    if run["tool"] != "separate":
        abort(404)
    path = PROCESSED_DIR / f"{run_id}_mixture.png"
    if not path.exists():
        audio, sr = backstage.read(run_input(run))
        backstage.save_mixture_png(backstage.mono(audio), sr, path)
    return send_file(path, mimetype="image/png")


@app.get("/backstage/<run_id>/mask/<stem>.png")
def backstage_mask(run_id, stem):
    """Mask heatmap |stem| / |mixture| of one separated stem (cached)."""
    run = registered_run(run_id)
    if run["tool"] != "separate":
        abort(404)
    files = dict(stem_files(run))
    if stem not in files:
        abort(404)
    index = list(files).index(stem)  # the stem name is user-facing; the index names the file
    path = PROCESSED_DIR / f"{run_id}_mask{index}.png"
    if not path.exists():
        mix, sr = backstage.read(run_input(run))
        audio, _ = backstage.read(files[stem])
        backstage.save_mask_png(backstage.mono(mix), backstage.mono(audio), sr, path)
    return send_file(path, mimetype="image/png")


HOST = "127.0.0.1"
PORT = 5000

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Debug mode (reloader, in-browser debugger) only on request: SPECTRA_DEBUG=1.
    debug = os.environ.get("SPECTRA_DEBUG") == "1"
    print(f"SPECTRA running at http://{HOST}:{PORT}  (Ctrl+C to stop)", flush=True)
    app.run(host=HOST, port=PORT, debug=debug)
