"""Make the landing page's preview assets. A one-off: run it by hand and
commit what it writes under static/landing/.

For each tool it takes one short phrase of the demo track and writes:
  clips/<tool>.mp3         the phrase twice: original, then processed
  clips/<tool>.json        when the processed half starts (switch_at_s), ...
  responses/<tool>.svg     the tool's response as one bare line, for the
                           reel card's background (separation: a .png
                           spectrogram of the stem instead)
and, for the hero's backdrop:
  hero.json                the whole track's spectrum over time

Everything comes from the real DSP: the effect modules' process()
functions, the app's own /response/* routes for the lines, and the
shared STFT for the spectra.

Run from the repo root:
    python scripts/make_landing_assets.py [--start SEC] [--length SEC]
The phrase defaults to the loudest 2.8 s of static/demo/demo.wav. A reel
slide lasts as long as its clip (the phrase twice, plus reverb and echo
tails), so 2.8 s keeps the slides at about 6 s.

Separation's clip is the drums stem, from the same
effects.separation.separate() that /process/separate calls.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib import image as mpimg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  (the /response/* routes, through its test client)
from analysis import backstage  # noqa: E402
from effects import echo, eq_filter, flanger, reverb, separation  # noqa: E402

SOURCE = ROOT / "static" / "demo" / "demo.wav"
OUT_DIR = ROOT / "static" / "landing"

PHRASE_SEC = 2.8        # clips of 5.7 s (6.3 s with a tail): about one 6 s slide
PRE_ROLL_SEC = 1.0      # processed with the phrase, then cut off: no onset artifacts
TAIL_SEC = 0.6          # reverb and echo: extra processed length, so the tail is heard
GAP_SEC = 0.12          # silence between the two halves
EDGE_FADE_SEC = 0.03
TAIL_FADE_SEC = 0.15
RMS_HOP_SEC = 0.05      # loudest-phrase scan step
PEAK = 0.98             # the whole clip is scaled down if loudness matching pushed it past this

MP3_KBPS = 128
MP3_RATES = (32000, 44100, 48000)  # MPEG-1 Layer III, where the bitrate mapping below holds

SVG_WIDTH, SVG_HEIGHT = 1000, 400
SVG_MARGIN = 12         # keeps the stroke inside the box
BACKGROUND_TEETH = 12   # echo / flanger lines: zoom so this many comb teeth show
SPECTROGRAM_SIZE = (800, 300)  # separation background (width, height)
SPECTROGRAM_RANGE_DB = 80
SPECTROGRAM_MIN_HZ = 40  # lower rows would all repeat the first FFT bins

# Hero backdrop: the track's spectrum, one row per HERO_FRAME_SEC, at
# HERO_POINTS log-spaced frequencies, as levels 0..100 over HERO_RANGE_DB.
HERO_HZ = (20, 20000)
HERO_POINTS = 128
HERO_FRAME_SEC = 0.25
HERO_RANGE_DB = 60
HERO_FRAME_SIZE = 4096  # 10.8 Hz bins at 44.1 kHz: enough detail at the log axis' low end
HERO_SMOOTH_POINTS = 5  # moving average along frequency: a shape, not noise
HERO_SMOOTH_ROWS = 3    # and along time (wrapping round, as the page loops): it drifts, not twitches

SIZE_BUDGET_MB = 1.5

# Showcase settings, chosen to be clearly audible.
SHOWCASE = {
    "eq": {"preset": "telephone"},
    "reverb": {"rt60": 2.5, "pre_delay_ms": 20, "wet": 0.5, "circular": False},
    "echo": {"mode": "feedback", "delay_ms": 300, "gain": 0.5, "mix": 0.6},
    # Defaults, with the notch depth g near its maximum of 1.
    "flanger": {**{name: limits[2] for name, limits in flanger.PARAMS.items()}, "gain": 0.95},
    "separation": {"stem": "drums"},
}
TAILED = ("reverb", "echo")
LABEL_B = {"separation": "STEM: DRUMS"}


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------

def separated_stem(audio, sr, stem):
    """One stem of audio, from the separation module /process/separate uses."""
    return np.asarray(separation.separate(audio, sr)[stem])


def process(tool, audio, sr):
    """The tool's output for audio, with its showcase settings (None if unavailable)."""
    params = SHOWCASE[tool]
    if tool == "eq":
        return eq_filter.process(audio, sr, params)[0]
    if tool == "separation":
        return separated_stem(audio, sr, params["stem"])
    effect = {"reverb": reverb, "echo": echo, "flanger": flanger}[tool]
    return effect.process(audio, sr, **params)[0]


def loudest_start(audio, sr, length):
    """Start (s) of the loudest `length`-second window, by RMS."""
    mono = backstage.mono(audio).astype(np.float64)
    n = int(length * sr)
    if len(mono) <= n:
        return 0.0
    energy = np.concatenate([[0], np.cumsum(mono ** 2)])
    starts = np.arange(0, len(mono) - n + 1, max(1, int(RMS_HOP_SEC * sr)))
    return starts[np.argmax(energy[starts + n] - energy[starts])] / sr


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def fade(x, sr, fade_in, fade_out):
    """Linear fade-in / fade-out (seconds) at the ends of x."""
    env = np.ones(len(x))
    n_in, n_out = int(fade_in * sr), int(fade_out * sr)
    env[:n_in] = np.linspace(0, 1, n_in)
    env[len(x) - n_out:] = np.linspace(1, 0, n_out)
    return x * (env[:, None] if x.ndim == 2 else env)


def make_clip(original, processed, sr, tail_fade=EDGE_FADE_SEC):
    """original + GAP_SEC of silence + processed, the processed half
    scaled to the original half's RMS, so "processed" never just means
    "quieter" (the telephone preset loses most of the energy).

    The RMS is compared over the phrase only: a reverb or echo tail
    would pull the processed half's average down. If the match pushes
    the peak past PEAK, the whole clip is scaled down, so the halves
    stay matched.

    Returns (clip, switch_at_s): switch_at_s is where the processed half starts.
    """
    a = fade(original, sr, EDGE_FADE_SEC, EDGE_FADE_SEC)
    b = fade(processed, sr, EDGE_FADE_SEC, tail_fade)
    b = b * (rms(a) / max(rms(b[:len(a)]), 1e-9))
    gap = np.zeros((int(GAP_SEC * sr),) + a.shape[1:])
    clip = np.concatenate([a, gap, b])
    peak = np.max(np.abs(clip))
    if peak > PEAK:
        clip *= PEAK / peak
    return clip, (len(a) + len(gap)) / sr


def tool_clip(tool, audio, sr, start, length):
    """The A/B clip for one tool, or None if the tool is unavailable.

    The effect runs on the phrase plus up to PRE_ROLL_SEC before it, and
    the pre-roll is cut off afterwards. Reverb and echo keep TAIL_SEC of
    tail after the phrase.
    """
    first, n = int(start * sr), int(length * sr)
    pre = min(first, int(PRE_ROLL_SEC * sr))
    out = process(tool, audio[first - pre:first + n], sr)
    if out is None:
        return None
    tail = int(TAIL_SEC * sr) if tool in TAILED else 0
    processed = out[pre:pre + n + tail]
    return make_clip(audio[first:first + n], processed,
                     sr, TAIL_FADE_SEC if tool in TAILED else EDGE_FADE_SEC)


def write_mp3(path, clip, sr):
    """Constant MP3_KBPS through libsndfile's MP3 encoder (soundfile);
    no ffmpeg needed. For MPEG-1 rates it maps compression 0..1 linearly
    onto 320..32 kbps."""
    if sr not in MP3_RATES:
        raise SystemExit(f"Resample the demo track to one of {MP3_RATES} Hz (it is {sr} Hz).")
    sf.write(str(path), clip, sr, format="MP3", subtype="MPEG_LAYER_III",
             bitrate_mode="CONSTANT", compression_level=(320 - MP3_KBPS) / (320 - 32))


# ---------------------------------------------------------------------
# Card backgrounds
# ---------------------------------------------------------------------

def response(route, **params):
    res = app.app.test_client().get(f"/response/{route}", query_string=params)
    if res.status_code != 200:
        raise SystemExit(f"/response/{route} failed: {res.get_json()}")
    return res.get_json()


def response_line(tool):
    """(x, y, y_range) of the tool's response from its /response/* route,
    with the showcase params, clipped to the Studio plot's y range.
    y_range: the y values to fill the box with (None: y's own min and max)."""
    params = SHOWCASE[tool]
    if tool == "eq":
        data = response("eq", **params)
        # Log-spaced frequencies: even steps on the Studio's log axis.
        return np.log10(data["freqs"]), np.maximum(data["gain_db"], -60), None
    if tool == "reverb":
        data = response("reverb", rt60=params["rt60"], pre_delay_ms=params["pre_delay_ms"])
        peak = np.max(np.abs(data["h"]))
        return data["t"], data["h"], (-peak, peak)  # centred on h = 0
    if tool == "echo":
        f_max = BACKGROUND_TEETH * 1000 / params["delay_ms"]
        data = response("echo", delay_ms=params["delay_ms"], gain=params["gain"],
                        mode=params["mode"], f_max=f_max)
        return data["freqs"], np.clip(data["mag_db"], -24, 24), None
    if tool == "flanger":
        longest = params["min_delay_ms"] + params["sweep_ms"]
        data = response("flanger", min_delay_ms=params["min_delay_ms"], sweep_ms=params["sweep_ms"],
                        gain=params["gain"], f_max=BACKGROUND_TEETH * 1000 / longest)
        # The comb at the longest delay of the sweep: the most notches.
        row = int(np.argmax(data["delays_ms"]))
        return data["freqs"], np.clip(data["mag_db"][row], -30, 10), None
    raise ValueError(tool)


def write_svg(path, x, y, y_range=None):
    """One polyline, normalised to fill the viewBox: no axes, no text.
    stroke="currentColor", so the page decides its colour."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    y_low, y_high = (y.min(), y.max()) if y_range is None else y_range
    px = SVG_MARGIN + (x - x.min()) / max(x.max() - x.min(), 1e-12) * (SVG_WIDTH - 2 * SVG_MARGIN)
    py = SVG_HEIGHT - SVG_MARGIN - (y - y_low) / max(y_high - y_low, 1e-12) * (SVG_HEIGHT - 2 * SVG_MARGIN)
    points = []
    for point in (f"{a:.1f},{b:.1f}" for a, b in zip(px, py)):
        if not points or point != points[-1]:
            points.append(point)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" '
        f'preserveAspectRatio="none">\n'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="currentColor" '
        f'stroke-width="3" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>\n'
        f"</svg>\n",
        encoding="utf-8")


def write_spectrogram_png(path, audio, sr):
    """Bare magma spectrogram (log frequency, no axes) of audio."""
    mag = backstage.magnitude_frames(backstage.mono(audio))
    db = 20 * np.log10(np.maximum(mag, mag.max() * 1e-9))
    width, height = SPECTROGRAM_SIZE
    df = sr / backstage.FRAME_SIZE
    freqs = np.geomspace(max(df, SPECTROGRAM_MIN_HZ), sr / 2, height)
    rows = np.clip(np.round(freqs / df).astype(int), 0, mag.shape[1] - 1)
    cols = np.linspace(0, len(db) - 1, width).round().astype(int)
    mpimg.imsave(str(path), db[cols][:, rows].T, cmap="magma", origin="lower",
                 vmin=db.max() - SPECTROGRAM_RANGE_DB, vmax=db.max())


def hero_spectrum(audio, sr):
    """The track's spectrum over time, for the hero's backdrop (the page
    only draws it). Power is averaged over HERO_FRAME_SEC rows and over
    log-spaced frequency bands; a band narrower than one FFT bin is
    interpolated instead. Levels: 0..100 over the top HERO_RANGE_DB."""
    hop = HERO_FRAME_SIZE // 4
    power = backstage.magnitude_frames(backstage.mono(audio), HERO_FRAME_SIZE, hop).astype(np.float64) ** 2
    per_row = max(1, round(HERO_FRAME_SEC * sr / hop))
    rows = len(power) // per_row
    power = power[:rows * per_row].reshape(rows, per_row, -1).mean(axis=1)

    bins = np.fft.rfftfreq(HERO_FRAME_SIZE, 1 / sr)
    freqs = np.geomspace(*HERO_HZ, HERO_POINTS)
    ratio = freqs[1] / freqs[0]
    edges = np.concatenate([[freqs[0] / np.sqrt(ratio)], np.sqrt(freqs[:-1] * freqs[1:]),
                            [freqs[-1] * np.sqrt(ratio)]])
    first, stop = np.searchsorted(bins, edges[:-1]), np.searchsorted(bins, edges[1:])
    total = np.concatenate([np.zeros((rows, 1)), np.cumsum(power, axis=1)], axis=1)
    count = stop - first
    band = (total[:, stop] - total[:, first]) / np.maximum(count, 1)
    narrow = count == 0
    for row in range(rows):
        band[row, narrow] = np.interp(freqs[narrow], bins, power[row])

    db = 10 * np.log10(np.maximum(band, 1e-20))
    n = HERO_SMOOTH_POINTS
    db = np.array([np.convolve(np.pad(r, n // 2, mode="edge"), np.ones(n) / n, mode="valid") for r in db])
    shifts = range(-(HERO_SMOOTH_ROWS // 2), HERO_SMOOTH_ROWS // 2 + 1)
    db = np.mean([np.roll(db, s, axis=0) for s in shifts], axis=0)
    levels = np.clip((db - (db.max() - HERO_RANGE_DB)) / HERO_RANGE_DB, 0, 1)
    return {
        "hz": list(HERO_HZ),
        "frame_s": round(per_row * hop / sr, 5),
        "range_db": HERO_RANGE_DB,
        "frames": np.round(levels * 100).astype(int).tolist(),
    }


# ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=float, help="phrase start (s); default: the loudest region")
    parser.add_argument("--length", type=float, default=PHRASE_SEC, help="phrase length (s)")
    args = parser.parse_args(argv)

    audio, sr = sf.read(str(SOURCE), dtype="float64")
    duration = len(audio) / sr
    start = loudest_start(audio, sr, args.length) if args.start is None else args.start
    if not (args.length > 0 and 0 <= start and start + args.length <= duration):
        raise SystemExit(f"The phrase {start:g}-{start + args.length:g} s must lie "
                         f"inside the track (0-{duration:.2f} s).")
    print(f"Phrase: {start:.2f}-{start + args.length:.2f} s of {SOURCE.name}")

    for folder in ("clips", "responses"):
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    for tool in SHOWCASE:
        result = tool_clip(tool, audio, sr, start, args.length)
        if result is None:
            continue
        clip, switch_at = result
        write_mp3(OUT_DIR / "clips" / f"{tool}.mp3", clip, sr)
        meta = {
            "tool": tool,
            "switch_at_s": round(switch_at, 3),
            "duration_s": round(len(clip) / sr, 3),
            "label_b": LABEL_B.get(tool, "PROCESSED"),
            "params": SHOWCASE[tool],
            "phrase": {"source": SOURCE.name, "start_s": round(start, 3), "length_s": args.length},
        }
        (OUT_DIR / "clips" / f"{tool}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        if tool == "separation":
            write_spectrogram_png(OUT_DIR / "responses" / "separation.png",
                                  clip[int(round(switch_at * sr)):], sr)
        else:
            write_svg(OUT_DIR / "responses" / f"{tool}.svg", *response_line(tool))
        print(f"  {tool}: switch at {meta['switch_at_s']} s, {meta['duration_s']} s long")

    hero = hero_spectrum(audio, sr)
    (OUT_DIR / "hero.json").write_text(json.dumps(hero, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"  hero: {len(hero['frames'])} spectra, {hero['frame_s']} s apart")

    files = sorted(p for p in OUT_DIR.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print("\nSizes:")
    for p in files:
        print(f"  {p.relative_to(OUT_DIR).as_posix():28} {p.stat().st_size / 1024:7.1f} KB")
    print(f"  {'total':28} {total / 1024:7.1f} KB")
    if total > SIZE_BUDGET_MB * 1024 * 1024:
        print(f"WARNING: over the {SIZE_BUDGET_MB} MB budget.")


if __name__ == "__main__":
    main()
