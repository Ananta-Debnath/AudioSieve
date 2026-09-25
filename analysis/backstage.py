"""Backstage: read-only analysis of one run, for the analysis wall.

Everything is computed from a run registry record: its input upload and
its output file(s). The STFT comes from stft.py, used as-is; nothing
here changes any audio. Plot data is downsampled before it is sent, and
the images are drawn with matplotlib's object API (no pyplot state).
"""

import os
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

import stft
from analysis.waveform import envelope
from effects import echo, eq_filter, flanger, reverb

# Analysis frames: the same as the before/after spectrograms.
FRAME_SIZE = 1024
HOP_SIZE = 512
WINDOW_NAME = "Hann"
CHUNK_FRAMES = 4096  # frames per stft.calculatr_stft call, to bound memory

RANGE_DB = 90        # floor below the loudest bin, as in the spectrograms
DIFF_CLIP_DB = 24
ENVELOPE_POINTS = 2000
LFO_POINTS = 1000
REVERB_COMPARE_SEC = 2.0
REVERB_COMPARE_POINTS = 1000
MAX_ECHO_STEMS = 40

# Images: time columns and log-spaced frequency rows.
IMAGE_COLUMNS = 1600
IMAGE_ROWS = 320
IMAGE_FREQ_TICKS = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)

# Images keep a dark background in both UI themes.
BG = "#111315"
FG = "#E8EAEC"
DIM = "#9AA0A6"
# Difference map: magenta (cut) -> black (unchanged) -> cyan (boosted).
DIFF_CMAP = LinearSegmentedColormap.from_list("cut_boost", ["#D946EF", "#000000", "#22D3EE"])


# ---------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------

def read(path):
    audio, sr = sf.read(str(path), dtype="float32")
    return audio, sr


def read_mixture(run, path):
    """A separation run's input, cut to the part that was separated (a
    long track is only separated up to the route's length cap)."""
    audio, sr = read(path)
    seconds = run.get("analysed_seconds")
    if seconds is not None:
        audio = audio[:int(round(seconds * sr))]
    return audio, sr


def mono(audio):
    """Mean of the channels (as a matrix product: much faster than
    mean(axis=1) on a long track)."""
    if audio.ndim == 1:
        return audio
    channels = audio.shape[1]
    return audio @ np.full(channels, 1 / channels, dtype=audio.dtype)


def n_frames(n_samples, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    """How many frames stft.get_frames makes from n_samples."""
    return len(range(0, n_samples - frame_size, hop_size))


def magnitude_frames(x, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    """|STFT| of mono x, shape (frames, frame_size // 2 + 1), float32.

    The same frames, window and FFT as stft.calculatr_stft (its window,
    apply_window and calculate_fft are used as-is), but the frames are a
    zero-copy view in float32, computed CHUNK_FRAMES at a time, so a
    6-minute track neither copies itself into frames nor holds its whole
    complex STFT in memory.
    """
    x = np.asarray(x, dtype=np.float32)
    count = n_frames(len(x), frame_size, hop_size)
    out = np.empty((count, frame_size // 2 + 1), np.float32)
    if count == 0:
        return out
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_size)[::hop_size][:count]
    window = stft.create_window(frame_size).astype(np.float32)
    for first in range(0, count, CHUNK_FRAMES):
        chunk = frames[first:first + CHUNK_FRAMES]
        out[first:first + len(chunk)] = np.abs(stft.calculate_fft(stft.apply_window(chunk, window)))
    return out


def _to_db(value):
    """20·log10(value), or None for digital silence (JSON has no -inf)."""
    return round(float(20 * np.log10(value)), 2) if value > 0 else None


def level_stats(audio, sr):
    """Peak and RMS over all samples of all channels, in dBFS (a
    full-scale square wave is 0 dBFS RMS; a full-scale sine is -3.01),
    crest factor (peak / RMS, in dB), and duration in seconds."""
    flat = np.asarray(audio).reshape(-1)
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    sum_sq = 0.0
    for start in range(0, flat.size, 2**16):  # short blocks, summed in float64
        block = flat[start:start + 2**16]
        sum_sq += float(np.dot(block, block))
    rms = np.sqrt(sum_sq / flat.size) if flat.size else 0.0
    peak_db, rms_db = _to_db(peak), _to_db(rms)
    return {
        "peak_dbfs": peak_db,
        "rms_dbfs": rms_db,
        "crest_db": round(peak_db - rms_db, 2) if peak_db is not None and rms_db is not None else None,
        "duration": len(audio) / sr,
    }


def envelope_json(audio, sr, n_points=ENVELOPE_POINTS):
    """Min/max envelope for the waveform overlay; bucket i starts at
    i · duration / len(max) seconds."""
    high, low = envelope(audio, n_points)
    return {"duration": len(audio) / sr, "max": high.round(4).tolist(), "min": low.round(4).tolist()}


# ---------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------

def _floor(*mags):
    peak = max((float(m.max()) for m in mags if m.size), default=0.0)
    return peak * 10 ** (-RANGE_DB / 20) if peak > 0 else 1e-10


def average_spectra(mag_x, mag_y, sr, original_frames):
    """Mean magnitude over all frames, in dBFS (the Hann window's gain is
    divided out, so a full-scale sine peaks near 0 dB). DC is dropped
    (the plot's frequency axis is logarithmic).

    mag_x / mag_y cover the same (padded) length. Both are averaged over
    the ORIGINAL's frame count: a silent tail then doesn't dilute the
    processed curve, and a tail with energy (reverb, echo) adds to it.
    """
    window_gain = stft.create_window(FRAME_SIZE).sum() / 2
    freqs = np.fft.rfftfreq(FRAME_SIZE, 1 / sr)[1:]
    frames = max(original_frames, 1)
    means = [m.sum(axis=0, dtype=np.float64)[1:] / frames / window_gain if len(m) else np.zeros(len(freqs))
             for m in (mag_x, mag_y)]
    floor = _floor(*means)
    before, after = (20 * np.log10(np.maximum(m, floor)) for m in means)
    return {
        "freqs": freqs.round(2).tolist(),
        "before_db": before.round(2).tolist(),
        "after_db": after.round(2).tolist(),
    }


def difference_db(mag_x, mag_y, frames=None):
    """20·log10(|Y| + ε) − 20·log10(|X| + ε) over the first `frames`
    frames (the original's length; default: all shared frames), clipped
    to ±DIFF_CLIP_DB. ε sits RANGE_DB below the loudest bin (the
    spectrograms' floor), so bins that are silent in both read 0 dB
    instead of amplifying rounding noise. Returns (diff, stats)."""
    n = min(len(mag_x), len(mag_y), len(mag_x) if frames is None else frames)
    x, y = mag_x[:n], mag_y[:n]
    eps = _floor(x, y)
    diff = np.clip(20 * np.log10((y + eps) / (x + eps)), -DIFF_CLIP_DB, DIFF_CLIP_DB)
    stats = {
        "max_abs_db": round(float(np.max(np.abs(diff))), 2) if diff.size else 0.0,
        "mean_abs_db": round(float(np.mean(np.abs(diff))), 2) if diff.size else 0.0,
        "boosted_pct": round(float(np.mean(diff > 1) * 100), 1) if diff.size else 0.0,
        "cut_pct": round(float(np.mean(diff < -1) * 100), 1) if diff.size else 0.0,
    }
    return diff.astype(np.float32), stats


# ---------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------

def _log_rows(sr, n_bins):
    """For each image row (log-spaced frequency), the nearest FFT bin."""
    df = sr / FRAME_SIZE
    freqs = np.geomspace(df, sr / 2, IMAGE_ROWS)
    return freqs, np.clip(np.round(freqs / df).astype(int), 0, n_bins - 1)


def _shrink_time(frames, columns=IMAGE_COLUMNS):
    """Average groups of frames so at most `columns` remain."""
    group = -(-len(frames) // columns) if len(frames) > columns else 1
    usable = len(frames) // group * group
    shrunk = frames[:usable].reshape(-1, group, frames.shape[1]).mean(axis=1)
    return shrunk if len(shrunk) else frames[:1]


def _save_image(data, sr, path, cmap, vmin, vmax, label, ticks=None, tick_labels=None):
    """data: (frames, bins) on the analysis frames. Drawn against time and
    a log frequency axis, dark background, with a colorbar."""
    freqs, rows = _log_rows(sr, data.shape[1])
    image = _shrink_time(data)[:, rows].T
    seconds = len(data) * HOP_SIZE / sr

    fig = Figure(figsize=(10, 3.4), dpi=130, facecolor=BG)
    ax = fig.add_subplot()
    ax.set_facecolor(BG)
    im = ax.imshow(image, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", extent=[0, seconds, 0, IMAGE_ROWS])
    positions = [np.interp(np.log(f), np.log(freqs), np.arange(IMAGE_ROWS))
                 for f in IMAGE_FREQ_TICKS if freqs[0] <= f <= freqs[-1]]
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{f // 1000}k" if f >= 1000 else str(f)
                        for f in IMAGE_FREQ_TICKS if freqs[0] <= f <= freqs[-1]])
    ax.set_xlabel("Time (s)", color=FG)
    ax.set_ylabel("Frequency (Hz, log)", color=FG)
    ax.tick_params(colors=DIM)
    for spine in ax.spines.values():
        spine.set_color(DIM)

    bar = fig.colorbar(im, ax=ax, pad=0.015)
    bar.set_label(label, color=FG)
    bar.ax.tick_params(colors=DIM)
    bar.outline.set_edgecolor(DIM)
    if ticks is not None:
        bar.set_ticks(ticks)
        bar.set_ticklabels(tick_labels)
    # Write under a unique name, then rename: a concurrent request never
    # sees (or serves) a half-written image.
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    fig.savefig(str(tmp), format="png", facecolor=BG, bbox_inches="tight")
    os.replace(tmp, path)


def save_diff_png(diff, sr, path):
    _save_image(diff, sr, path, DIFF_CMAP, -DIFF_CLIP_DB, DIFF_CLIP_DB,
                "Change (dB)", ticks=[-24, -12, 0, 12, 24],
                tick_labels=["−24 cut", "−12", "0", "+12", "+24 boost"])


def save_mixture_png(mix, sr, path):
    mag = magnitude_frames(mix)
    db = 20 * np.log10(np.maximum(mag, _floor(mag)))
    _save_image(db, sr, path, "magma", db.max() - RANGE_DB, db.max(), "Magnitude (dB)")


def save_mask_png(mix, stem, sr, path):
    """Mask heatmap |stem| / |mixture|, clipped to 0..1. Bins where the
    mixture itself is silent (below the spectrogram floor) have no
    meaningful ratio and are drawn as 0."""
    mag_mix = magnitude_frames(mix)
    mag_stem = magnitude_frames(stem[:len(mix)])
    n = min(len(mag_mix), len(mag_stem))
    mix_n, stem_n = mag_mix[:n], mag_stem[:n]
    audible = mix_n > _floor(mix_n)
    mask = np.zeros_like(mix_n)
    mask[audible] = np.clip(stem_n[audible] / mix_n[audible], 0, 1)
    _save_image(mask, sr, path, "magma", 0, 1, "|stem| / |mixture|")


# ---------------------------------------------------------------------
# Settings and tool-specific extras
# ---------------------------------------------------------------------

def _frame_settings(frame_size, hop_size, sr):
    return {
        "frame_size": frame_size,
        "hop_size": hop_size,
        "window": WINDOW_NAME,
        "sr": sr,
        "df_hz": sr / frame_size,
        "dt_ms": frame_size / sr * 1000,
        "hop_ms": hop_size / sr * 1000,
    }


def stft_settings(tool, sr):
    """The analysis frames; for EQ runs also the long processing frames."""
    settings = {"analysis": _frame_settings(FRAME_SIZE, HOP_SIZE, sr)}
    if tool == "eq":
        settings["processing"] = _frame_settings(eq_filter.FRAME_SIZE, eq_filter.HOP_SIZE, sr)
    return settings


def reverb_start_comparison(x, sr, params, seconds=REVERB_COMPARE_SEC):
    """The first `seconds` of the reverb output (mono, before the clip
    protection's scaling), linear vs circular convolution.

    Linear convolution is causal, so its start only needs the start of x.
    Circular convolution (FFT length len(x), h truncated to len(x)) is
    the linear result with everything past len(x) folded back onto the
    start: y_c[k] = y[k] + y[len(x) + k], and y[len(x) + k] only depends
    on the last len(h) − 1 input samples. So both come from short
    convolutions, not from re-running the whole track.
    """
    h = reverb.generate_impulse_response(sr, params["rt60"], params["pre_delay_ms"], seed=0)
    total = len(x)
    n = min(total, int(seconds * sr))
    h_c = h[:min(len(h), total)]

    linear_wet = reverb.fft_convolve(x[:n], h)[:n]
    wrapped = np.zeros(n)
    if len(h_c) > 1:
        tail = reverb.fft_convolve(x[total - len(h_c) + 1:], h_c)[len(h_c) - 1:]
        wrapped[:min(n, len(tail))] = tail[:n]
    circular_wet = reverb.fft_convolve(x[:n], h_c)[:n] + wrapped

    wet = params["wet"]
    dry = x[:n]
    return (1 - wet) * dry + wet * linear_wet, (1 - wet) * dry + wet * circular_wet


def _extra(run, x, sr, out_seconds):
    """Tier 3: tool-specific data (small; computed from params and input)."""
    tool, params = run["tool"], run["params"]

    if tool == "reverb":
        linear, circular = reverb_start_comparison(x, sr, params)
        return {"reverb_start": {
            "duration": len(linear) / sr,
            "linear": envelope_json(linear, sr, REVERB_COMPARE_POINTS),
            "circular": envelope_json(circular, sr, REVERB_COMPARE_POINTS),
            "run_circular": bool(params.get("circular")),
        }}

    if tool == "echo":
        gain = float(np.clip(params["gain"], 0, echo.MAX_GAIN))
        delay = echo.delay_samples(sr, params["delay_ms"])
        if params["mode"] == "feedforward" or gain == 0:
            k = np.arange(2 if gain > 0 else 1)
        else:
            repeats = echo.tail_length(sr, delay, gain, "feedback") // delay
            k = np.arange(min(repeats, MAX_ECHO_STEMS) + 1)
        return {"echo_ir": {"t": (k * delay / sr).round(5).tolist(), "h": (gain ** k).round(5).tolist(),
                            "mode": params["mode"], "delay_ms": params["delay_ms"], "gain": gain}}

    if tool == "flanger":
        rate = params["rate_hz"]
        span = min(out_seconds, max(4 / rate, 1.0))  # a few LFO cycles
        t = np.linspace(0, span, LFO_POINTS)
        delay_ms = flanger.sweep_delay_ms(rate * t, params["min_delay_ms"], params["sweep_ms"])
        return {"flanger_lfo": {"t": t.round(5).tolist(), "delay_ms": delay_ms.round(4).tolist()}}

    return {}


# ---------------------------------------------------------------------
# Whole runs
# ---------------------------------------------------------------------

def _padded_magnitudes(x, y):
    """|STFT| of mono x and y framed identically: the shorter one is
    zero-padded to the longer one's length first (as for the before /
    after spectrograms), so frame k covers the same samples in both.
    Also returns the original's own frame count."""
    length = max(len(x), len(y))

    def padded(signal):
        return np.pad(signal, (0, length - len(signal))) if len(signal) < length else signal

    return magnitude_frames(padded(x)), magnitude_frames(padded(y)), n_frames(len(x))


def analyze_effect(run, input_path, output_path, diff_png=None):
    """Tier 1 + 2 (+ 3) data for an effect run. Also writes the
    difference image to diff_png, if given and not there yet (the
    difference is already computed here)."""
    x_audio, sr = read(input_path)
    y_audio, _ = read(output_path)
    x, y = mono(x_audio), mono(y_audio)

    mag_x, mag_y, original_frames = _padded_magnitudes(x, y)
    diff, diff_stats = difference_db(mag_x, mag_y, original_frames)
    if diff_png is not None and not diff_png.exists() and diff.size:
        save_diff_png(diff, sr, diff_png)

    return {
        "levels": {"before": level_stats(x_audio, sr), "after": level_stats(y_audio, sr)},
        "envelopes": {"before": envelope_json(x_audio, sr), "after": envelope_json(y_audio, sr)},
        "avg_spectrum": average_spectra(mag_x, mag_y, sr, original_frames),
        "diff": {
            **diff_stats,
            "clip_db": DIFF_CLIP_DB,
            "seconds": len(x) / sr,
            "output_longer_s": max(0.0, (len(y) - len(x)) / sr),
        },
        "stft": stft_settings(run["tool"], sr),
        "extra": _extra(run, x, sr, len(y) / sr),
    }


def render_diff(input_path, output_path, diff_png):
    """Just the difference image (when it is requested before the JSON)."""
    x, sr = read(input_path)
    y, _ = read(output_path)
    diff, _ = difference_db(*_padded_magnitudes(mono(x), mono(y)))
    save_diff_png(diff, sr, diff_png)


def analyze_separation(run, input_path, stem_paths):
    """stem_paths: [(name, path)]. Levels and envelopes of the mixture
    and each stem; the mask images are drawn on request."""
    mix, sr = read_mixture(run, input_path)
    stems = []
    for name, path in stem_paths:
        audio, _ = read(path)
        stems.append({"name": name, "levels": level_stats(audio, sr), "envelope": envelope_json(audio, sr)})
    return {
        "levels": {"mixture": level_stats(mix, sr)},
        "envelopes": {"mixture": envelope_json(mix, sr)},
        "stems": stems,
        "stft": stft_settings(run["tool"], sr),
    }
