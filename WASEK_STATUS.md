# `wasek` branch: status

**As of:** 2026-09-25 · **Last commit:** `6a13bf6` pivot stage 04b: SPECTRA backstage analysis
**Pushed:** no, 2 commits ahead of `origin/wasek` (`89b1294`, `6a13bf6`) · **Working tree:** clean
**Tests:** 217 passed, 4 slow deselected (`python -m pytest tests -m "not slow"`, ~85 s)
**Deadline:** code freeze Sun 2026-09-27, projector demo Mon 2026-09-28

## What this branch is

The project pivoted from "separation only" to an **Audio Lab**, now named **SPECTRA**:
a Flask app where you upload a track once and run independent DSP tools on it, one per
tab, plus a **Backstage** tab that analyses any run. This branch holds the app, every
effect and the whole frontend. Separation stays on Ananta's `ananta` branch and gets
merged in at the end.

Rule for every effect: the DSP is written from first principles. No `scipy.signal`
filter design, `lfilter`, `fftconvolve` or `np.convolve` in `effects/`, and no Web Audio
effect nodes. Only `np.fft` and plain numpy. The frontend only plays and plots; it never
processes audio.

Run it with `python app.py` (add `SEPARATION_MOCK=1` for fake separation stems).

## Done

| Tab | Module | How it works | Status |
| --- | --- | --- | --- |
| EQ + Filter | [effects/eq_filter.py](effects/eq_filter.py) | Static gain curve × STFT spectrum. Butterworth low/high/band-pass/band-stop from the closed-form formula; 5-band EQ (low shelf, 3 Gaussian peaks, high shelf); EQ + filter combined; hum removal (notches at 50/60 Hz and harmonics) | ✅ |
| Reverb | [effects/reverb.py](effects/reverb.py) | Linear FFT convolution with a synthetic IR (decaying noise + pre-delay). A "circular convolution" toggle shows the wrong way (the tail wraps onto the start) | ✅ |
| Echo + Delay | [effects/echo.py](effects/echo.py) | Difference equations: feedforward `y[n] = x[n] + g·x[n−D]` and feedback `y[n] = x[n] + g·y[n−D]` (vectorised in blocks of D, exact) | ✅ |
| Flanger | [effects/flanger.py](effects/flanger.py) | Feedforward comb with an LFO-swept delay D(n), with linear-interpolated fractional delay | ✅ |
| Separation | none yet | UI built against the contract below. `/process/separate` returns **501** until the merge; `SEPARATION_MOCK=1` returns copies of the input as drums/bass/rest | ⏳ merge |
| Backstage | [analysis/backstage.py](analysis/backstage.py) | Analysis wall for any run (see below) | ✅ |

### Effects notes

**EQ presets:** telephone, radio, remove_rumble, remove_hum_50hz, bass_boost.

**EQ uses long frames:** 65536 samples, not 1024. Low frequencies now get the curve that
the plot shows. For example, the rumble filter cuts 30 Hz by 34 dB (it was 18 dB).
The reasoning and measurements are in
[docs/long frames for EQ,filters and hum.md](docs/long%20frames%20for%20EQ,filters%20and%20hum.md).

### Stage 04a: Studio UI

Files: [templates/index.html](templates/index.html),
[static/style.css](static/style.css), [static/js/](static/js/).

- Plain HTML/CSS/vanilla JS modules, no build step. Chart.js 4.5.1 and WaveSurfer 7.12.12
  are pinned CDN scripts; fonts are Big Shoulders Display + Martian Mono (Google Fonts).
- Dark theme by default, light-theme toggle remembered in `localStorage`. Charts and
  waveforms recolour on toggle. Cyan = processed/after, grey = original/before.
- Each tool keeps its own file, sliders and last result. The source box has upload
  (drag-and-drop or browse), **USE SAME FILE** and **USE DEMO TRACK**.
- Sliders get their min/max/default from each module's `PARAMS`. Labels, units and steps
  live in [ui_config.py](ui_config.py), not in the JS.
- Live system-response plot for every effect (debounced, stale responses dropped). The
  flanger's plot animates the sweep, with play/pause.
- Output: ORIGINAL and PROCESSED WaveSurfer players (one plays at a time, Space toggles
  the last used), DOWNLOAD WAV, → BACKSTAGE.
- Separation: ORIGINAL player plus one row per stem with play, MUTE, SOLO, DOWNLOAD and a
  synced PLAY ALL. Mute/solo only change playback volume.
- HOW IT WORKS drawer per tool (equation, a short explanation, one line per parameter).
  It is non-modal and closes with ✕, Esc or the button.
- Placeholder demo track: [static/demo/demo.wav](static/demo/demo.wav), made by
  [scripts/make_placeholder_demo.py](scripts/make_placeholder_demo.py) (sine melody, bass
  tone, noise bursts). The script refuses to overwrite an existing `demo.wav` without
  `--force`.
- Every run is recorded by the run registry ([runs.py](runs.py)): in memory, plus a
  `<run_id>.run.json` sidecar in `processed/`, so runs survive a restart.

### Stage 04b: Backstage

- A run picker lists this browser tab's runs (e.g. `REVERB · 14:32 · RT60 1.5s WET 0.3`);
  → BACKSTAGE from a tool opens that run.
- Every figure has a `FIG 0N / TITLE` header and a `LOOK FOR:` caption.
- Effect runs:
  - Tier 1: before/after spectrograms, difference spectrogram (magenta = cut,
    cyan = boost, ±24 dB), system response, average spectrum, waveform overlay, run card.
  - Tier 2: level meters (peak, RMS, crest, duration) and STFT settings. EQ runs also show
    the 65536-sample processing frames.
  - Tier 3: reverb linear vs circular (first 2 s), echo impulse response (stem plot),
    flanger delay sweep D(n).
- Separation runs: mixture spectrogram, per-stem mask heatmaps (|stem| / |mixture|),
  stem levels, run card, STFT settings.
- Analysis JSON is cached on disk after the first look, and so are the images. A 6-minute
  stereo track takes ~2.3–2.7 s the first time.

## Routes ([app.py](app.py))

API responses only gained fields in stage 04; nothing was renamed or removed.

| Route | Purpose |
| --- | --- |
| `POST /upload` | Returns `file_id`, filename, duration, sample rate, channels and a playable `url` |
| `POST /upload/demo` | Registers `static/demo/demo.wav` like a normal upload |
| `GET /upload/<id>` and `/upload/<id>/waveform` | The uploaded file, and its min/max envelope for the ORIGINAL player |
| `POST /process/<eq\|reverb\|echo\|flanger>` | Runs the effect. Returns `run_id`, result and download URLs, spectrograms, waveform peaks and (for EQ) the curve |
| `POST /process/separate` | 501 until the merge; the contract (below) with `SEPARATION_MOCK=1` |
| `GET /result/<id>` and `/result/<id>/spectrogram/<before\|after>.png` | Output files |
| `GET /response/<eq\|reverb\|echo\|flanger>` | Plot data from the parameters alone |
| `GET /backstage/runs` and `/backstage/<run_id>` | Run list, and one run's analysis JSON |
| `GET /backstage/<run_id>/diff.png`, `/mixture.png`, `/mask/<stem>.png` | Backstage images |

**Separation contract** (Ananta's code must match):

```text
POST /process/separate  (same file_id pattern as the other tools)
200 → {"run_id": "...", "stems": [{"name": "drums", "url": "...", "download_url": "..."}, ...]}
```

The stem count comes from the response; the UI never assumes 3.

## Tests

- [tests/test_eq_filter_unit.py](tests/test_eq_filter_unit.py) (97 tests): curve shapes;
  the audio really gets the plotted curve on sine tones; each preset does what its name
  says.
- [tests/test_reverb_echo.py](tests/test_reverb_echo.py) (72 tests, 3 slow):
  - reverb, echo and flanger against naive reference implementations
  - impulse responses, gain clamping, dry/wet edge cases, routes
  - `@pytest.mark.slow` performance tests on 6 min of stereo audio
- [tests/test_studio.py](tests/test_studio.py) (29 tests): `/response/eq`, uploads and the
  demo track, run ids and the registry, the separation mock, and slider ranges coming
  from `PARAMS`.
- [tests/test_backstage.py](tests/test_backstage.py) (23 tests, 1 slow): JSON shape per
  tool, ~0 dB difference when an effect does nothing, level stats on a known sine,
  caching, 404s, and a 6-minute timing test.
- [test_eq_filter.py](test_eq_filter.py) (root) is a manual script, not pytest. It saves
  curves, spectrograms and WAVs to check by eye and by ear.

## Known issues

- **The demo needs internet:** Chart.js, WaveSurfer and the fonts come from CDNs. On a
  machine with no network the charts and waveforms won't load. Fix if needed: save copies
  into `static/` as a fallback.
- **Flaky slow test:** `test_six_minutes_of_stereo_takes_under_5_seconds[apply_reverb]`
  takes 5.2–5.6 s on this laptop against a 5 s limit. It fails on `07cdacf` too, before
  stage 04, so it is not a regression. Only affects the slow tests.
- **`demo.wav` is a placeholder:** replace it with the real demo track (same path).
- **Light theme:** cyan traces on white are low contrast (1.8:1). The spec keeps cyan
  the same in both themes; every before/after chart has a text legend.
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

## Branch relationships

- **`main`** is far behind at `d8c9f65` (2026-08-26). Nothing from this branch is on
  `main` yet.
- **`ananta`** (as of the last fetch) has 6 commits that `wasek` doesn't, latest
  `2f7122f` "Quite good separation" (2026-09-25). They include `57a3fe0` (mask scaling,
  rest mask, NMF refactor) and `af80719` (semi-supervised NMF).
- A trial merge (`git merge-tree`) still conflicts **only in `.gitignore`**.
- `stft.py` and `utils.py` are still identical on both branches. The functions `app.py`
  and `analysis/backstage.py` use (`calculatr_stft`, `calculate_magnitude`,
  `save_spectrogram`, `create_window`, `apply_window`, `calculate_fft`) all exist on
  `ananta`.
- `ananta` has no `app.py`, so its separation code needs wiring into the route.

## Next (in order)

1. **Push** the two stage-04 commits to `origin/wasek`.
2. **Replace the placeholder demo track** with the real one.
3. **Merge `ananta` into `wasek`:**
   - resolve the `.gitignore` conflict
   - wire separation into `/process/separate` so it returns the contract above, and
     record the run with `run_registry().add(...)` (as the mock does), so real
     separation runs appear in Backstage
   - fix the `bass_mask.py` import
4. Merge into `main`.
5. Deployment configuration (the next stage).
