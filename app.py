"""AudioSieve Audio Lab: Flask app shell.

Synchronous request -> process -> response. Each tool is independent:
upload a file once, then run any tool on it by id.
"""

import uuid
from pathlib import Path

import soundfile as sf
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

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


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    return error(f"File too large (max {MAX_UPLOAD_MB} MB).", 413)


@app.get("/")
def index():
    return render_template("index.html")


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
    audio, sr = effect.process(audio, sr, **params)

    result_id = str(uuid.uuid4())
    sf.write(str(PROCESSED_DIR / f"{result_id}.wav"), audio, sr, subtype="FLOAT")

    return jsonify({
        "result_id": result_id,
        "url": url_for("result", result_id=result_id),
        "download_url": url_for("result", result_id=result_id, download=1),
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


if __name__ == "__main__":
    app.run(debug=True)
