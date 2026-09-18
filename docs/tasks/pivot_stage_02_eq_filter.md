# Pivot Stage 2: EQ / Filter (Real DSP)

## Context

Stage 1 built the Flask app shell with a stub `/process/eq` route that
currently just passes audio through unchanged (`effects/eq_filter.py`).
This stage replaces the stub with real signal processing.

Reuse the existing shared STFT infrastructure — don't reimplement
framing, windowing, FFT, or overlap-add:

- `stft.py` — `get_frames`, `apply_window` (Hann), `calculate_fft`
  (`np.fft.rfft`), `calculate_ifft` (`np.fft.irfft`), `overlap_add`.
- `utils.py` — audio I/O, `apply_mask` (spectra × mask), spectrogram
  plotting/saving.

Keep frame size 1024 / hop size 512, consistent with the rest of the
project (drum/bass mask work).

**Explicitly do not use `scipy.signal.butter` or any other filter-design
library function.** The whole project is built on STFT-domain masking —
computing the gain curve directly from its closed-form frequency-response
formula and applying it to STFT bins keeps this consistent with the rest
of the pipeline and is the actual signal theory this stage is meant to
demonstrate. `scipy.signal` filter design targets time-domain IIR
recursion, which is a different technique that doesn't fit how the rest
of this codebase processes audio.

## Concept Recap

EQ and Filter are the same mechanism — a per-frequency-bin gain curve
multiplied onto the STFT spectrum, same shape as the drum/bass masks —
just with a different curve shape, and **static across all frames**
(unlike the adaptive per-frame drum/bass masks, this curve doesn't change
over time):

- **Filter**: near-binary curve — ≈1 inside the passband, ≈0 outside,
  smooth rolloff at the edge.
- **EQ**: smooth curve with several boost/cut bumps, nothing fully
  zeroed.

## Requirements

### 1. Filter curve generation

In `effects/eq_filter.py`, add:

```python
def create_filter_curve(freq_bins, filter_type, cutoff=None,
                         low_cutoff=None, high_cutoff=None, order=4):
    """
    freq_bins: array of bin center frequencies (Hz), from rfft bin count.
    filter_type: 'lowpass' | 'highpass' | 'bandpass'
    Returns: array of gain values in [0, 1], same length as freq_bins.
    """
```

Use the closed-form Butterworth magnitude-response formula directly:

- Low-pass: `gain(f) = 1 / (1 + (f / cutoff) ** (2 * order))`
- High-pass: `gain(f) = 1 / (1 + (cutoff / f) ** (2 * order))` — guard
  the `f = 0` (DC) bin against division by zero (set gain to 0 there).
- Band-pass: multiply a low-pass curve (at `high_cutoff`) by a high-pass
  curve (at `low_cutoff`) — cascading the two gives you the band in the
  middle.

`order` controls rolloff steepness (higher = sharper cutoff). Default 4.

### 2. EQ curve generation

```python
def create_eq_curve(freq_bins, bands):
    """
    bands: list of dicts, e.g.
      [{'center': 100, 'gain_db': 6, 'bandwidth': 80}, ...]
    Returns: array of linear gain values (can be >1 for boost, <1 for cut).
    """
```

For each band, compute a Gaussian bump in dB centered at `center` with
the given `bandwidth`, scaled by `gain_db`, sum all bands' contributions
in dB, then convert the total to linear gain: `10 ** (total_db / 20)`.

Default band set (5 bands, ±12 dB range each):
`Bass (100 Hz), Low-mid (350 Hz), Mid (1000 Hz), High-mid (3500 Hz), Treble (10000 Hz)`.

### 3. Applying the curve

Since the curve is static (same for every frame), apply it by
multiplying each frame's complex spectrum directly — this both scales
magnitude and leaves phase untouched automatically, same math as
`utils.apply_mask`, just with one curve reused across all frames instead
of a per-frame adaptive mask. Reuse `apply_mask` if its signature
supports a single 1D curve broadcast across frames; otherwise write a
small wrapper rather than duplicating the multiply logic.

### 4. Top-level process function

Replace the stub in `effects/eq_filter.py`:

```python
def process(audio, sr, params):
    """
    params example for EQ mode:
      {'mode': 'eq', 'bands': [{'center': 100, 'gain_db': 6, 'bandwidth': 80}, ...]}
    params example for Filter mode:
      {'mode': 'filter', 'filter_type': 'lowpass', 'cutoff': 4000, 'order': 4}
      {'mode': 'filter', 'filter_type': 'bandpass', 'low_cutoff': 300, 'high_cutoff': 3000, 'order': 4}
    Returns: (processed_audio, sr)
    """
```

Pipeline inside: frame + window (via `stft.py`) → FFT per frame → build
the gain curve once (via the appropriate function above) → apply it to
every frame's spectrum → inverse FFT + overlap-add → return.

### 5. Wire into the existing route

Update the `/process/eq` handler in `app.py` (already exists as a stub
from Stage 1) to read `mode` and the relevant params from the request,
call `effects.eq_filter.process(audio, sr, params)`, and return the
result the same way the stub did. Don't add a new route — EQ and Filter
share this one endpoint via the `mode` param, per how the app shell was
scoped.

### 6. Minimal test input

Keep the frontend form barebones for now (plain number inputs for
cutoff/order or band gains is fine — no sliders yet, that's a later
polish stage). Follow the existing `test_drum_mask.py` pattern in the
repo and add a `test_eq_filter.py` that runs both modes on a test clip
and saves before/after spectrograms to `Spectograms/` (via the existing
plotting utilities in `utils.py`), so results can be sanity-checked
visually without needing the web UI.

## Acceptance Criteria

- [ ] `create_filter_curve` produces correct-shaped curves for lowpass,
      highpass, and bandpass — verify by plotting the curve itself (gain
      vs. frequency) and confirming the expected shape.
- [ ] `create_eq_curve` correctly boosts/cuts the target band without
      significantly affecting distant frequencies.
- [ ] Running filter mode with a low cutoff (e.g. 500 Hz lowpass)
      audibly muffles a test clip; a highpass at the same cutoff
      audibly thins it out. Confirm via listening and via spectrogram
      comparison.
- [ ] Running EQ mode with a boosted bass band visibly raises low-
      frequency energy in the output spectrogram relative to the input,
      without distorting/clipping (check output isn't clamped — apply
      the same clipping protection `utils.py` already uses for other
      exports).
- [ ] `/process/eq` in `app.py` calls the real `process()` function; the
      stub no-op is gone for this route.
- [ ] `test_eq_filter.py` exists and produces before/after spectrograms
      for both modes.

## Out of Scope (don't touch yet)

- Reverb, Echo/Delay DSP (Stages 3 and 4).
- Real slider-based frontend UI (later polish stage) — plain form inputs
  are enough for this stage.
- Real-time/streaming processing — this stays batch, upload → process →
  return, same as the rest of the app.
- Any changes to `stft.py` or `utils.py` beyond importing from them — if
  something's missing there, flag it rather than editing shared files
  your partner is also using.
