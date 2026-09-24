# `wasek` branch: status

**As of:** 2026-09-24 · **Last commit:** `403fbb1` added flanger tab with animated frequency response
**Pushed:** yes, `origin/wasek` is up to date · **Working tree:** clean
**Tests:** 166 passed, 3 slow deselected (`python -m pytest tests -m "not slow"`, ~23 s)
**Deadline:** ~2026-09-27

## What this branch is

The project pivoted from "separation only" to an **Audio Lab**: a Flask app where you
upload a track once and run independent DSP tools on it, one per tab. This branch holds
the app and every effect. Separation stays on Ananta's `ananta` branch and gets merged
in at the end.

Rule for every effect: the DSP is written from first principles. No `scipy.signal`
filter design, `lfilter`, `fftconvolve` or `np.convolve` in `effects/`, and no Web Audio
effect nodes. Only `np.fft` and plain numpy.

## Done

| Tab | Module | How it works | Status |
|---|---|---|---|
| EQ + Filter | [effects/eq_filter.py](effects/eq_filter.py) | Static gain curve × STFT spectrum. Butterworth low/high/band-pass/band-stop from the closed-form formula; 5-band EQ (low shelf, 3 Gaussian peaks, high shelf); EQ + filter combined; hum removal (notches at 50/60 Hz and harmonics) | ✅ |
| Reverb | [effects/reverb.py](effects/reverb.py) | Linear FFT convolution with a synthetic IR (decaying noise + pre-delay). A "circular convolution" toggle shows the wrong way (the tail wraps onto the start) | ✅ |
| Echo + Delay | [effects/echo.py](effects/echo.py) | Difference equations: feedforward `y[n] = x[n] + g·x[n−D]` and feedback `y[n] = x[n] + g·y[n−D]` (vectorised in blocks of D, exact) | ✅ |
| Flanger | [effects/flanger.py](effects/flanger.py) | Feedforward comb with an LFO-swept delay D(n), with linear-interpolated fractional delay | ✅ |
| Separation | none | `/process/separate` returns **501** until the merge | ⏳ |

**EQ presets:** telephone, radio, remove_rumble, remove_hum_50hz, bass_boost
(bright and warm were removed).

**Shared app features**

- Upload validation: `.wav/.mp3/.flac` only, 50 MB max, 6 min max.
- Each result returns playable and downloadable audio plus **before/after spectrograms**,
  drawn with the shared `stft.py`/`utils.py` code.
- Response plots, which need no upload:
  - EQ: gain curve (dB against log frequency)
  - Reverb: impulse response
  - Echo: comb-filter frequency response
  - Flanger: frequency response **animated** over one sweep
- Parameter ranges are defined once in each module's `PARAMS` and used by both the route
  validation and the HTML inputs.

**EQ uses long frames:** 65536 samples, not 1024. Low frequencies now get the curve that
the plot shows. For example, the rumble filter cuts 30 Hz by 34 dB (it was 18 dB).
The reasoning and measurements are in
[docs/long frames for EQ,filters and hum.md](docs/long%20frames%20for%20EQ,filters%20and%20hum.md).

## Routes ([app.py](app.py))

| Route | Purpose |
|---|---|
| `POST /upload` | Returns a `file_id` |
| `POST /process/<eq\|reverb\|echo\|flanger>` | Runs the effect and returns the result URL, download URL, spectrograms and (for EQ) the curve |
| `POST /process/separate` | 501 stub |
| `GET /result/<id>` and `/result/<id>/spectrogram/<before\|after>.png` | Output files |
| `GET /response/<reverb\|echo\|flanger>` | Plot data from the parameters alone |

## Tests

- [tests/test_eq_filter_unit.py](tests/test_eq_filter_unit.py) (30 tests): curve shapes;
  the audio really gets the plotted curve on sine tones; each preset does what its name
  says.
- [tests/test_reverb_echo.py](tests/test_reverb_echo.py) (40 tests):
  - reverb, echo and flanger against naive reference implementations
  - impulse responses, gain clamping, dry/wet edge cases, routes
  - `@pytest.mark.slow` performance tests on 6 min of stereo audio
- [test_eq_filter.py](test_eq_filter.py) (root) is a manual script, not pytest. It saves
  curves, spectrograms and WAVs to check by eye and by ear.

## Known issues

- **`bass_mask.py` is broken:** it imports `drum_mask`, but that file is now
  `drum_mask_gen.py`. Nothing in the app uses it yet.
- The stage-1 bass plan in
  [STAGE_01_bass_mask_and_quality_gate.md](STAGE_01_bass_mask_and_quality_gate.md) is
  only partly done:
  - `bass_mask.py` exists.
  - `main.py` has no `--stem` flag.
  - There is no `EVALUATION.md`.
- [PROGRESS.md](PROGRESS.md) is out of date (last updated 2026-08-29, drums only).
- Hum removal leaves a little hum in the first and last ~0.5 s. Any notch this narrow has
  this limit, so it is not a bug.
- The frontend is still barebones: plain number inputs and minimal CSS.

## Branch relationships

- **`main`** is far behind at `d8c9f65` (2026-08-26). Nothing from this branch is on
  `main` yet.
- **`ananta`** has 2 commits that `wasek` doesn't:
  - `57a3fe0`: mask scaling, rest mask, NMF refactor
  - `af80719`: semi-supervised NMF
- A trial merge (`git merge-tree`) conflicts **only in `.gitignore`**.
- `stft.py` and `utils.py` are identical on both branches. The functions `app.py` imports
  (`calculatr_stft`, `calculate_magnitude`, `save_spectrogram`, `apply_mask`, plus the
  framing/FFT helpers) all exist on `ananta`.

## Next (in order)

1. **Frontend polish:** sliders, layout and styling of the tabs and plots.
2. **Merge `ananta` into `wasek`:**
   - resolve the `.gitignore` conflict
   - wire separation into `/process/separate`
   - fix the `bass_mask.py` import
3. Merge into `main`.
