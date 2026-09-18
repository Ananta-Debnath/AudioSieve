"""AudioSieve Audio Lab: Flask app shell.

Synchronous request -> process -> response. Each tool is independent:
upload a file once, then run any tool on it by id.
"""

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

import stft
import utils
from effects import echo, eq_filter, reverb

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}
MAX_UPLOAD_MB = 50
MAX_DURATION_SEC = 6 * 60

# Tool name (as used in /process/<tool>) -> effect module.
# Later stages only fill in the modules' process() functions.
EFFECTS = {
    "eq": eq_filter,
    "reverb": reverb,
    "echo": echo,
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)


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
    """
    if raw.get("preset"):
        return {"preset": raw["preset"]}

    mode = raw.get("mode") or "eq"
    if mode not in eq_filter.MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: {', '.join(eq_filter.MODES)}."
        )
    params = {"mode": mode}

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


def save_spectrogram_png(audio, sr, path):
    """Spectrogram of the (mono-mixed) audio, via the shared STFT/plot code."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    spectra = stft.calculatr_stft(mono, eq_filter.FRAME_SIZE, eq_filter.HOP_SIZE)
    utils.save_spectrogram(
        utils.calculate_magnitude(spectra), sr, eq_filter.HOP_SIZE, str(path)
    )


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    return error(f"File too large (max {MAX_UPLOAD_MB} MB).", 413)


@app.get("/")
def index():
    return render_template(
        "index.html",
        eq_presets=eq_filter.PRESETS,
        eq_bands=eq_filter.DEFAULT_BANDS,
        eq_max_gain=eq_filter.MAX_BAND_GAIN_DB,
    )


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return error("No file provided (expected form field 'file').")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(f".{e}" for e in sorted(ALLOWED_EXTENSIONS))
        return error(f"Unsupported format '.{ext}'. Allowed: {allowed}.")

    file_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{file_id}.{ext}"
    file.save(path)

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

    return jsonify({
        "file_id": file_id,
        "filename": file.filename,
        "duration": info.duration,
        "samplerate": info.samplerate,
        "channels": info.channels,
    })


@app.post("/process/separate")
def process_separate():
    return error(
        "Separation is not implemented in the app shell yet; "
        "it is being developed on the separation branch.",
        501,
    )


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

        for kind, data in (("before", audio), ("after", processed)):
            save_spectrogram_png(data, sr, PROCESSED_DIR / f"{result_id}_{kind}.png")

        extra = {
            "curve": {
                "freqs": freq_bins.round(1).tolist(),
                "gain_db": (20 * np.log10(np.maximum(curve, 1e-6))).round(2).tolist(),
            },
            "spectrograms": {
                kind: url_for("result_spectrogram", result_id=result_id, kind=kind)
                for kind in ("before", "after")
            },
        }
        audio = processed
    else:
        audio, sr = effect.process(audio, sr, **params)

    sf.write(str(PROCESSED_DIR / f"{result_id}.wav"), audio, sr, subtype="FLOAT")

    return jsonify({
        "result_id": result_id,
        "url": url_for("result", result_id=result_id),
        "download_url": url_for("result", result_id=result_id, download=1),
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
        download_name=f"audiosieve_{result_id[:8]}.wav",
    )


@app.get("/result/<result_id>/spectrogram/<kind>.png")
def result_spectrogram(result_id, kind):
    if not is_valid_id(result_id) or kind not in ("before", "after"):
        abort(404)
    path = PROCESSED_DIR / f"{result_id}_{kind}.png"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
