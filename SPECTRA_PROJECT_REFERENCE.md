# SPECTRA: complete project reference

> **For the assistant reading this:** this file describes a finished student project
> so that you can answer questions about it without seeing the code. It was written
> from the source on **2026-09-27**, at commit `5bc113e` on branch `merged`. File
> paths are relative to the repo root. The owner will present it on **2026-09-28** on
> a local PC and then deploy it. Most follow-up questions will be about the
> presentation, deployment, and how the DSP works. Section 11 lists the facts that
> matter for deployment, and section 12 lists known loose ends.

---

## 1. What the project is

**SPECTRA** is an interactive audio signal-processing lab that runs in the browser.
The user uploads a track once, then runs independent DSP tools on it, one per tab:

| # | Tool | Core idea |
|---|------|-----------|
| 01 | **EQ + Filter** | Multiply every frequency by a gain curve (STFT → × H[k] → ISTFT) |
| 02 | **Reverb** | Convolve with a synthetic room impulse response via the FFT |
| 03 | **Echo + Delay** | Difference equations (feedforward FIR comb, feedback IIR comb) |
| 04 | **Flanger** | A feedforward comb whose delay is swept by an LFO |
| 05 | **Separation** | Split a mix into 4 stems with NMF and soft spectral masks |
| — | **Backstage** | Analysis wall for any run: spectrograms, difference map, levels, and more |

It also has a landing page at `/` with an animated hero, a carousel of before/after
audio previews, and a "how it's built" diagram.

- **Course:** CSE 220 (signal processing), BUET. The lab header reads
  "signal processing lab · CSE 220 · BUET".
- **Team:** two people. Wasik Samin (the repo owner) wrote the Flask app, all four
  effects, Backstage, the frontend, the landing page and the integration. The partner,
  Ananta, wrote the separation algorithm (NMF and component scoring) on the `ananta`
  branch. It was merged in at the end.
- **Names:** the repo folder is `AudioSieve`, the old README calls it **Stemify**
  (from when it was only a separation project), and the app itself is **SPECTRA**.
  SPECTRA is the current name.
- **Design rule:** all DSP is written from first principles with numpy (`np.fft` and
  array math). The effects don't use `scipy.signal` filter design, `lfilter`,
  `fftconvolve` or `np.convolve`, and the browser doesn't use Web Audio effect nodes.
  The browser never processes audio; it only plays and plots what the server returns.
  (One exception: the partner's separation *feature scoring* uses
  `scipy.signal.find_peaks` to measure rhythm. That is analysis, not filtering.)
- **No machine learning.** Separation uses NMF, a matrix factorisation that runs on the
  track itself, with no training data or trained model.

---

## 2. Tech stack

| Layer | What |
|-------|------|
| Backend | Python 3.12+ (the dev machine has 3.14), **Flask 3.1.3**, Werkzeug 3.1.8 |
| DSP | numpy 2.5.1, scipy 1.18.0 (only `find_peaks` in separation scoring), pandas 3.0.3 (separation's component table) |
| Audio I/O | **soundfile 0.14.0** (libsndfile) reads and writes WAV, FLAC **and MP3**. **No ffmpeg is needed.** |
| Images | matplotlib 3.11.1, Agg backend (spectrogram PNGs rendered on the server) |
| Frontend | Plain HTML, CSS and vanilla ES modules. **No build step and no framework.** |
| Charts | **Chart.js 4.5.1**, vendored in `static/vendor/chart.umd.min.js` |
| Waveforms/players | **WaveSurfer.js 7.12.12**, vendored in `static/vendor/wavesurfer.min.js` |
| Fonts | Big Shoulders Display (800) for headings and Martian Mono (400/500/700), self-hosted `.woff2` files in `static/fonts/` with their OFL licences |
| Tests | pytest 9.1.1 |

**The app runs fully offline.** After the first `pip install`, no page loads anything
from the internet. A test checks this: `test_pages_load_nothing_from_the_internet`.

`requirements.txt` also lists `pydub` and `audioop-lts`, but only `trim.py` (an old
helper script the app doesn't use) needs them.

---

## 3. How to run it (local)

```text
run.bat             Windows
./run.sh            macOS / Linux / Git Bash   (or: bash run.sh; see §12)
run.bat --clean     also delete old uploads and results first
```

What the run scripts do:
1. Find a Python ≥ 3.12. `run.bat` tries the `py -3` launcher, then `python`.
   `run.sh` tries `python3`, then `python`.
2. Create `.venv` if it's missing.
3. `pip install -r requirements.txt`, but only when `requirements.txt` has changed
   since the last install. They keep a copy in `.venv/requirements.installed`, so later
   starts skip pip and work offline.
4. `--clean` runs `scripts/clean_runs.py`, which empties `uploads/` and `processed/`.
5. `python app.py` serves on **http://127.0.0.1:5000** with debug off.

Manual equivalent: `pip install -r requirements.txt && python app.py`.

**Environment variables** (the only two):
- `SPECTRA_DEBUG=1`: Flask debug mode (reloader and in-browser debugger). Off by default.
- `SEPARATION_MOCK=1`: `/process/separate` skips the real separation and returns copies
  of the input as the 4 stems. This is a fallback for demos and is off by default.

**Helper scripts** (`scripts/`):
- `warmup.py`: run it with the app running. It runs every tool once on the demo track
  with the showcase settings, through the real HTTP routes, then fetches the audio, the
  Backstage analysis and the images, and prints PASS/FAIL with timings. It also checks
  that the pages load only local files. Options: `--url`, `--local` (in-process through
  Flask's test client). Exit code 1 if anything fails. Showcase settings: EQ
  `telephone` preset; reverb RT60 2.5 s, wet 0.5; echo feedback, 300 ms, g 0.5;
  flanger defaults; separation on the demo song.
- `clean_runs.py`: empties `uploads/` and `processed/`.
- `make_landing_assets.py`: a one-off that regenerates the landing page's preview clips,
  response-line SVGs and hero spectrum from the real DSP. Its output is committed.
- `make_placeholder_demo.py`: writes the 20 s synthetic `static/demo/demo.wav`: a sine
  melody in A minor pentatonic, a low bass tone and noise bursts. It refuses to
  overwrite the file without `--force`.

**Tests:** `python -m pytest tests -m "not slow"` runs the fast suite
(281 tests pass in about 2 minutes; 5 slow tests are deselected). `-m slow` runs the performance tests on 6 minutes of stereo audio.

---

## 4. Repository layout

```text
app.py                 Flask app: every route, parameter parsing, result naming, spectrograms
ui_config.py           What the page shows per tool: labels, units, slider steps, HOW IT WORKS
                       text, presets. Slider RANGES come from each effect's PARAMS table.
runs.py                Run registry: every /process run in memory + <run_id>.run.json sidecar
effects/
  common.py            per_channel() and prevent_clipping() shared by reverb/echo/flanger
  eq_filter.py         EQ, Butterworth filters, hum notches, presets (STFT with 65536-sample frames)
  reverb.py            Synthetic IR + linear FFT convolution (+ the "wrong" circular version)
  echo.py              Feedforward / feedback comb difference equations + frequency response
  flanger.py           LFO-swept fractional delay + animated frequency responses
  separation.py        Wrapper that runs the partner's NMF pipeline without file I/O
analysis/
  backstage.py         Backstage analysis: levels, envelopes, average spectra, difference map,
                       mask heatmaps, STFT settings, tool-specific extras
  waveform.py          Min/max envelopes ("peaks") sent to WaveSurfer
stft.py                Shared STFT/ISTFT (Hann window, framing, rfft, overlap-add). Partner's.
utils.py               Shared helpers (magnitude, apply_mask, save_spectrogram, load/save audio). Partner's.
semi_supervised_nmf.py NMF with temporal continuity + sparseness (multiplicative updates). Partner's.
nmf_sep.py             Per-component feature extraction, scoring, Wiener-style masks. Partner's.
main_semi_nmf.py       Partner's original command-line separation script (reads Audio/audio.wav).
                       effects/separation.py reproduces its steps and settings.
bass_mask.py           Early experiment (band-limited/energy-gated bass mask); not used by the app
trim.py                Old pydub clip-cutting helper; not used by the app
templates/index.html   The lab (/lab)
templates/landing.html The landing page (/)
static/js/             main.js (tabs, routing), tools.js (the 5 tools), source.js (upload box),
                       controls.js (sliders), player.js (WaveSurfer players, stem sync),
                       responses.js (system-response plots), charts.js (Chart.js wrapper),
                       backstage.js, sendto.js, drawer.js, theme.js, util.js, config.js, landing.js
static/style.css       Lab styles; static/landing.css landing styles
static/vendor/         Chart.js + WaveSurfer (offline)
static/fonts/          Self-hosted fonts + fonts.css
static/demo/           demo.wav (synthetic), demo_vocals.flac (the Separation demo song), CREDITS.txt
static/landing/        Landing assets: clips/<tool>.mp3 + .json, responses/<tool>.svg|png, hero.json
scripts/               warmup.py, clean_runs.py, make_landing_assets.py, make_placeholder_demo.py
tests/                 pytest suite (see §10)
test_eq_filter.py      Manual script (not pytest): saves EQ curves, spectrograms and WAVs to inspect
docs/                  Stage specs (docs/tasks/pivot_stage_01..06) and a note on EQ frame length
uploads/, processed/   Runtime data, created at startup and gitignored
run.sh, run.bat        Launchers
PROGRESS.md, WASEK_STATUS.md, STAGE_01_*.md   Older status notes, partly outdated (see §12)
```

---

## 5. The lab page (`/lab`): shared features

### 5.1 Layout and navigation
- **Masthead:** the SPECTRA wordmark, the subtitle "signal processing lab · CSE 220 ·
  BUET", and a **dark/light theme toggle** (☾ / ☀).
- **Tab bar:** 01 EQ + Filter, 02 Reverb, 03 Echo + Delay, 04 Flanger, 05 Separation,
  and Backstage. Tabs work with the mouse or with the arrow keys, Home and End.
- **Deep links:** `/lab#reverb`, `/lab#echo`, and so on open that tab, and the URL hash
  follows the open tab. `#separation` is an alias for the `separate` view. Switching
  tabs doesn't add browser history entries.
- Each tool tab has two columns. The left one holds the title, subtitle, **How it
  works** button, source box, parameters and **Run** button. The right one holds the
  **System response** plot (not on Separation) and the **Output**.
- **Every tool is independent:** each tab keeps its own file, slider values and last
  result while you switch tabs.

### 5.2 Source box (in every tool)
- **Drag and drop** a file anywhere on the box, or **click to browse**. Accepted: WAV,
  MP3, FLAC, **up to 200 MB and 20 minutes**. The browser checks the extension and size
  before uploading, and the server checks everything again.
- **Use same file <name>:** reuses the last file loaded in *any* tool by its `file_id`,
  without uploading it again.
- **Use demo track:** the 20 s synthetic `demo.wav`. On Separation the button is
  **Use demo song**: 30 s of "Let's Go Fishin'" by Karissa Hobbs (CC BY-NC-SA 4.0,
  credits in `static/demo/CREDITS.txt`), because a separation demo needs vocals.
- Once a file is loaded, the box shows its name and "duration · kHz · mono/stereo", plus
  **Replace**. After pressing Replace there is a **Keep current** button to cancel.
- Errors show inline: unsupported format, too large, too long, unreadable file.

### 5.3 Parameters (sliders)
- Every slider's **min, max and default come from the server**: the effect module's
  `PARAMS` table, which the route also validates against. The JavaScript hard-codes no
  ranges; `ui_config.py` supplies labels, units and steps.
- Frequency sliders use a **log scale**.
- The **readout is an editable text box**. Type a value and press Enter or click away
  to apply it; Esc cancels. It accepts `3.4k`, `3.4 kHz`, a decimal comma and `−`.
  Out-of-range values are clamped with a warning, and anything else gets a "Not a
  number" warning. Typed values snap to the slider step.
- **Double-click a slider to reset** it to its default.
- **Segmented buttons** choose modes (for example EQ mode, filter type and echo mode).

### 5.4 Live system-response plot
- For EQ, Reverb, Echo and Flanger, the plot redraws **as you move sliders** (debounced
  by 150 ms, and a late reply to an older request is dropped). The data comes from
  `GET /response/<tool>`, computed on the server from the parameters alone, before
  anything is run.
- EQ shows the gain curve in dB on a log-frequency axis. Reverb shows the impulse
  response h(t). Echo shows |H| in dB, with zoom buttons **200 Hz / 2 kHz / FULL**.
  Flanger shows an **animated** |H| as the delay sweeps, with a "D = x.xx ms" readout
  and **Play/Pause sweep**. The animation follows the LFO rate and starts paused when
  the OS asks for reduced motion.

### 5.5 Run and output
- **Run** stays disabled until a track is loaded. While a run is going it shows
  "Processing… Ns" with a live seconds counter. The browser gives up after **20
  minutes**. If you load another file during a run, the late result is discarded.
- **Output for an effect:** two **WaveSurfer players**, **Original** (grey) and
  **Processed** (accent colour), drawn at the same scale so level changes are visible.
  Below them:
  - **Download WAV**
  - **Send to ▾**
  - **→ Backstage**
  - a "Run HH:MM · took X.X s" stamp
- **Players:**
  - click or drag the waveform to seek
  - starting one player pauses all the others
  - **the Space bar** plays or pauses the last player used (not while typing, and not
    when a keyboard user has focused a button)
  - the server sends the waveform envelope, so the browser streams the file instead of
    decoding it
- **Download names** are built on the server as `<source>_<what made it>.wav`. Examples:
  `song_radio filter.wav`, `song_lowpass filter.wav`, `song_eq + highpass filter.wav`,
  `song_hum filter 50 Hz.wav`, `song_circular reverb.wav`, `song_feedback echo.wav`,
  `song_flanger.wav`, `song_vocals.wav`. Results that were sent on and processed again
  chain their names: `song_radio filter_reverb.wav`. Characters that no OS allows in a
  file name are replaced with `-`.

### 5.6 Send to ▾ (move a result between tools)
- Under every effect output and every separation stem there is a **Send to ▾** menu
  that lists all five tools. The result's own tool is included and marked "again".
- Picking a tool registers that result on the server as a normal upload
  (`POST /upload/result/<result_id>`, a server-side copy, no download needed), loads it
  as that tool's source, and switches to its tab. The source box briefly flashes an
  outline.
- This lets you chain effects (for example EQ → Reverb → Echo), apply an effect twice,
  or run an effect on one stem (for example vocals → Reverb).
- The menu works from the keyboard (arrows, Home/End, Esc) and closes when you click
  outside it or tab away.
- A browser can't put audio on the OS clipboard, so the feature is an in-app menu
  instead.

### 5.7 How it works drawer
- A non-modal side drawer for each tool, with its governing equation(s), a short
  explanation and one line per parameter. The text lives in `ui_config.py`; §6 quotes it.
- It closes with ✕, Esc, or the tool's button. It follows the tabs as you switch and
  closes on Backstage.

### 5.8 Theme and accessibility
- Dark by default. The **light theme** choice is saved in `localStorage`
  (`spectra-theme`) and applied before the first paint, so there is no dark flash.
  Charts and waveforms recolour on toggle.
- Colours: dark theme background `#111315`, accent cyan `#22D3EE`, warning orange
  `#F97316`. Light theme background `#F2F3F1`, accent blue `#3A83F7`, warning
  `#C2410C`. Spectrogram images keep a dark background in both themes.
- ARIA roles and labels are set throughout (tabs, menus, pressed states, live regions).
  `prefers-reduced-motion` is respected. The layout is responsive, with breakpoints at
  1279 px and 899 px.

---

## 6. The tools in detail

All effects keep the sample rate and the channel layout. Stereo channels are processed
independently. Audio is decoded as float32. If the output's peak would exceed full
scale, the whole signal is **scaled down** so nothing hard-clips (`prevent_clipping`).
Results are written as **32-bit float WAV**. For every effect run, the server also
saves **before/after spectrogram PNGs** (magma colour map, 1024/512 STFT, 90 dB range).
Backstage shows them.

### 6.1 EQ + Filter (`effects/eq_filter.py`)

**Equations:** `Y[k] = H[k] · X[k]` and `|H(f)| = 1 / √(1 + (f/fc)^(2N))`

**Mechanism:** STFT, then multiply every frame by one static gain curve H[k], then
inverse FFT and overlap-add. **Frames are 65,536 samples (hop 32,768)**, so bins sit
0.67 Hz apart at 44.1 kHz and even a 50 Hz hum notch lands where the plot shows it.
With 1024-sample frames, frequencies below about 100 Hz were smeared: the 80 Hz rumble
filter cut 30 Hz by only 18 dB instead of 34 dB, and a 50 Hz notch managed 4.5 dB
instead of 48 dB. Long frames cost nothing here because the curve never changes.
`docs/long frames for EQ,filters and hum.md` has the measurements. The audio is padded
by one frame before processing and trimmed afterwards.

**Modes:** `eq`, `filter`, `both` (EQ curve × filter curve), `hum`. The default is `eq`.

- **Filter:** a Butterworth magnitude response computed from the closed-form formula,
  as an amplitude curve (the square root of the power response). It is −3 dB at the
  cutoff and rolls off at 6N dB/octave.
  - Types: **low-pass, high-pass, band-pass** (a low-pass times a high-pass) and
    **band-stop**, which uses the low-pass-to-band-stop transform
    `f/fc → B·f/(f0² − f²)` with `f0 = √(low·high)` and `B = high − low`.
  - Sliders: Cutoff 20 Hz–20 kHz (default 4000), Low cutoff (default 300), High cutoff
    (default 3000), Order N 1–8 (default 4).
  - Band types show two cutoffs. Dragging one past the other **pushes the other along**,
    so a band always keeps a gap of about 1/6 octave and can't become invalid.
  - Every cutoff must be below Nyquist.
- **5-band EQ:** gains in dB are summed across bands and converted to linear gain.
  Each band gain is limited to ±15 dB, in 0.5 dB steps.

  | Band | Type | Centre | Width |
  |---|---|---|---|
  | Bass | low shelf | 100 Hz | n/a |
  | Low-mid | Gaussian peak | 350 Hz | FWHM 250 Hz |
  | Mid | Gaussian peak | 1 kHz | FWHM 700 Hz |
  | High-mid | Gaussian peak | 3.5 kHz | FWHM 2500 Hz |
  | Treble | high shelf | 10 kHz | n/a |

  Shelves are smooth steps of the form `x/(1+x)`, with `x = (f/fc)^4`, reaching half
  the gain at their centre.
- **Hum removal:** Butterworth band-stop notches (order 4) at the hum frequency and its
  harmonics. Sliders: hum frequency 40–70 Hz (default 50, since mains hum is 50 Hz in
  Bangladesh and Europe and 60 Hz in the US), harmonics 1–10 (default 5), notch width
  1–10 Hz (default 4, measured between the −3 dB points). The highest harmonic must be
  below Nyquist. Known limit: the first and last ~0.5 s keep a little hum, because any
  notch this narrow needs that much signal to resolve.
- **Presets** (buttons; the active one is highlighted, and the highlight clears when
  you move a slider by hand):

  | Preset | Settings |
  |---|---|
  | Telephone | band-pass 300–3400 Hz, order 4 |
  | Radio | band-pass 500–5000 Hz, order 2 |
  | Remove rumble | high-pass 80 Hz, order 4 |
  | Remove hum 50 Hz | hum mode, 50 Hz, 5 harmonics, 4 Hz notches |
  | Bass boost | EQ: Bass +8 dB, Low-mid +2 dB |

  Applying a preset sets the sliders. A filter-only or hum preset resets the band gains
  to flat, so what you see is what runs. The preset name is sent only to name the
  download.
- The response plot shows the curve on a log grid from 20 Hz to 20 kHz. In hum mode the
  notch centres and edges are added to the grid, so the notches show at full depth.

### 6.2 Reverb (`effects/reverb.py`)

**Equation:** `y = x * h ⟷ Y = X · H`, the convolution theorem.

- **Impulse response:** synthetic. White noise times `exp(−ln(1000)·t/RT60)`, which is
  −60 dB at t = RT60, preceded by a silent pre-delay. Its length is RT60, capped at
  5 s. It is normalised to unit energy so the wet level stays predictable.
  - The noise is seeded, so the same settings always give the same IR.
  - Each channel uses a different seed (the channel index), which makes stereo sound
    wider.
- **Linear FFT convolution:** both signals are zero-padded to at least `N + M − 1`
  samples (rounded up to a power of two), multiplied as spectra, inverse-FFT'd and
  trimmed. The output is `len(h) − 1` samples longer than the input, because the tail
  rings out.
- **Circular toggle ("CIRCULAR (DEMO: WRONG WAY)")** is an educational switch. It
  applies the FFT at length `len(x)` with **no zero-padding**, so the reverb tail wraps
  around onto the **start** of the track. You can hear it in the first second. The UI
  shows a warning note when this is on.
- **Mix:** `out = (1 − wet)·dry + wet·(x * h)`
- **Parameters:**

  | Parameter | Range | Default |
  |---|---|---|
  | RT60 | 0.2–5.0 s | 1.5 |
  | Pre-delay | 0–100 ms | 20 |
  | Wet | 0–1 | 0.3 |
  | Convolution | linear / circular | linear |

- The response plot shows h(t), decimated to at most 4000 points.

### 6.3 Echo + Delay (`effects/echo.py`)

**Equations:**
- feedforward (one echo, FIR): `y[n] = x[n] + g·x[n − D]`
- feedback (decaying repeats, IIR): `y[n] = x[n] + g·y[n − D]`

- **Feedback is stable only for g < 1**, so the gain is limited to **0.9**. The impulse
  response is 1, g, g², … at 0, D, 2D, …
- **Tail** (zeros appended so the echoes can ring out):
  - feedforward: D samples
  - feedback: `D·K`, where `K = ceil(log(0.001)/log g)`, the number of repeats until
    the echo is 60 dB down
  - capped at 5 s, and 0 when g = 0
- **Vectorised exactly:** the feedback comb is computed in blocks of D samples, since
  each block depends only on the previous one. This is exact, not an approximation, and
  loops `len/D` times instead of once per sample.
- **Mix:** `out = dry + mix·(y − dry)`
- **Parameters:**

  | Parameter | Range | Default |
  |---|---|---|
  | Delay D | 20–2000 ms | 350 |
  | Gain g | 0–0.9 | 0.5 |
  | Mix | 0–1 | 0.5 |
  | Mode | feedforward / feedback | feedback |

- **Frequency response**, from the closed form:
  - feedforward: `H = 1 + g·e^(−jωD)`
  - feedback: `H = 1/(1 − g·e^(−jωD))`
  - Both are comb filters with peaks every sr/D Hz.
  - The plot has zoom buttons (200 Hz / 2 kHz / FULL), because for long delays the
    teeth are too dense to see across the whole spectrum.

### 6.4 Flanger (`effects/flanger.py`)

**Equation:** `y[n] = x[n] + g·x[n − D(n)]`

- This is the feedforward echo with a delay of only a few milliseconds, swept by an LFO:
  `D(n) = min + sweep·(1 − cos(2π·rate·t))/2`, a raised cosine.
- The comb's notches sit at odd multiples of `1/(2D)` Hz and **sweep up and down the
  spectrum**, which makes the "jet plane" whoosh. At g = 1 the notches cancel
  completely.
- **Fractional delay:** `n − D(n)` usually falls between two samples, so the value is
  **linearly interpolated** from its two neighbours. Because D keeps changing, the
  delayed copy is also slightly pitch-shifted, like a Doppler shift.
- The output is `ceil(min + sweep)` ms longer than the input.
- **Parameters:**

  | Parameter | Range | Default |
  |---|---|---|
  | Min delay | 0.1–10 ms | 1 |
  | Sweep | 0–10 ms | 3 |
  | Rate | 0.05–5 Hz | 0.25 |
  | Gain g | 0–1 | 0.9 |

- **Response plot:** the server computes |H| at 48 delays over one sweep, 600 points
  each. The browser animates through them at the LFO rate. The x-axis shows the first
  10 comb teeth at the longest delay.

### 6.5 Separation (`effects/separation.py`, which wraps the partner's code)

**Equations:** `V ≈ W·H` and `mask = stem estimate / mixture`

**Stems (4):** **percussion, bass, vocals, harmonics**. "Harmonics" is the pitched
sound that is left, such as guitar, keys and strings. The tool has **no parameters**.

**Pipeline**, which mirrors the partner's `main_semi_nmf.main()` settings exactly but
without its file I/O:
1. Pad the track by one STFT frame on each side (so the edges reconstruct cleanly),
   mix it down to mono, and normalise that mono mix to a peak of 1.
2. STFT: a 40 ms frame rounded up to a power of two (**2048 samples at 44.1 kHz**),
   50 % overlap (hop 1024), Hann window. Take the magnitude V.
3. **NMF**, `semi_supervised_nmf.separate_sources_nmf`:
   - 20 components, 300 multiplicative-update iterations.
   - The reconstruction term is KL-divergence style.
   - A temporal-continuity penalty (α = 100) favours smooth activations over time. A
     sparseness term exists but has weight β = 0, so it is off.
   - Initialisation is random with a fixed seed (0), so the same track always gives the
     same stems. The global RNG is restored afterwards.
   - This matches Virtanen's temporal-continuity NMF formulation (the code's comments
     refer to "the paper" defaults).
   - `initialize_B`, a hand-designed spectral-template initialisation, exists but is
     commented out. It is the reason for the "semi-supervised" name.
4. **Score each component** (`nmf_sep`):
   - **Spectral features:** energy, low_ratio (below 250 Hz), bass_ratio (below
     150 Hz), high_ratio (above 4 kHz), spectral centroid, bandwidth, flatness,
     peakiness, harmonicity (strong spectral peaks), number of peaks.
   - **Activation features:** transientness, sustain, activation variance, rhythmicity
     (regular peak spacing, found with `find_peaks`).
   - **Vocal features:** mid-band ratio, formant strength (energy in 300–1000,
     800–2500 and 1800–3500 Hz), spectral-envelope smoothness.
   - These are combined into weighted scores:
     - `percussion = 0.40·transient + 0.20·high + 0.15·flatness + 0.10·(1−sustain) + 0.10·rhythm + 0.05·act_var`
     - `bass = 0.40·bass_ratio + 0.20·low + 0.15·(1−high) + 0.10·sustain + 0.10·harmonicity + 0.05·(1−transient)`
     - `vocal = 0.45·mid_band + 0.35·formant + 0.20·envelope_smoothness`
5. **Group components into stems:**
   - percussion: the **top 2** by percussion score
   - bass: every component with a **bass score above 0.6**
   - vocals: the top `int((20 − used)·3/5)` by vocal score
   - harmonics: everything not used
   - An empty group gives a silent stem. As in the partner's script, the vocal pick
     doesn't exclude components that were already used, so a component can count
     toward two groups.
6. **Masks:** a Wiener-style soft mask for each group,
   `(Σ W·H of its components)² / |X|²`, clipped to 0–1. All masks are then divided by
   their sum so they **sum to 1 at every bin**. As a result the **stems add up to the
   original mix**, and a test checks this.
7. The masks are computed once from the mono mix and **applied to each channel's STFT**,
   so stereo stays stereo. Then ISTFT and overlap-add, and the padding is trimmed.
   Every stem has exactly the input's length and channels.

**Known limitation:** bleed. Where two instruments overlap in frequency at the same
moment, the mask cannot split them.

**Runtime:** about 0.4 s per second of stereo audio. It processes the **whole track**,
with no length cap. Measured on the dev laptop:

| Track | Time |
|---|---|
| Demo song (30 s) | 11–12 s |
| 2:13 song | 51 s |
| 6 min stereo | ~150 s |

Only **one separation runs at a time** per server process (a lock), because it is
CPU- and memory-heavy and uses NumPy's global RNG.

**Separation output UI:**
- An **Original** player, then a **stem mixer** with one row per stem. Each row has:
  - a waveform player
  - a **Volume slider (0–100 %)**
  - **Mute** and **Solo** (solo wins over mute)
  - **Download**
  - **Send to ▾**
- **▶ Play all / ❚❚ Pause all** plays every stem in sync. Seeking one stem seeks all of
  them, and players that drift more than 80 ms are pulled back. With Play all running,
  muting a stem reveals what the rest sound like without it.
- Volume, mute and solo **only change playback**. Nothing is re-processed.
- **→ Backstage** opens the separation analysis.

---

## 7. Backstage (analysis wall)

A global tab that analyses one run. It is read-only and never changes any audio.

- **Run picker:** chips such as `REVERB · 14:32 · RT60 1.5s WET 0.3` or
  `EQ · 14:35 · FILTER BANDPASS 300-3400Hz N4`. It lists only **this browser tab's
  runs**: their ids are kept in `sessionStorage` and filtered from `/backstage/runs`.
  **→ Backstage** under an output opens that run, and a **← Back to <tool>** button
  returns to the tab at the same scroll position.
- Every figure has a `FIG 0N / TITLE` header and a one-line **LOOK FOR:** caption that
  tells the viewer what to notice.
- The analysis JSON and images are **computed once and cached on disk**. The first look
  at a 6-minute stereo track takes about 2.5 s.

**Effect runs (EQ / Reverb / Echo / Flanger):**

| Fig | Content |
|---|---|
| 01 | **Spectrograms**, original vs processed side by side |
| 02 | **Difference spectrogram**, `20·log10(|Y|/|X|)`, clipped to ±24 dB. Magenta = cut, black = unchanged, cyan = boosted. Stats: largest change, % of bins boosted or cut by more than 1 dB. Notes when the output is longer (a tail). |
| 03 | **System response**, the same plot as the tool, for this run's exact parameters (the flanger one animates, echo has zoom buttons) |
| 04 | **Average spectrum**, before vs after in dBFS on a log-frequency axis, averaged over the original's length so a tail's energy counts |
| 05 | **Waveform overlay**, min/max envelopes of original (grey) and processed (accent); shows level changes and tails |
| 06 | **Run card**: tool, source file, time, run id, every parameter with its unit, and the equations |
| 07 | **Levels table**: peak dBFS, RMS dBFS, crest factor (dB), duration; before, after and the change |
| 08 | **STFT settings**: frame size, hop, window, sample rate, Δf = sr/N, N/sr. EQ runs also show the 65,536-sample processing frames. |
| 09 | Tool-specific. **Reverb:** linear vs circular convolution over the first 2 s, computed cheaply from short convolutions; it shows the wrapped tail. **Echo:** impulse-response stem plot (one repeat for feedforward, the train g, g², … for feedback). **Flanger:** the delay sweep D(n) over a few LFO cycles. |

**Separation runs:**

| Fig | Content |
|---|---|
| 01 | **Mixture spectrogram** |
| 02 | **Stem levels table** (peak, RMS and crest for the mixture and each stem) with WAV links |
| 03 | **Stem mask heatmaps**: `|stem| / |mixture|` per stem. Bright = the stem owns that bin, mid values = overlap and bleed. |
| 04 | Run card |
| 05 | STFT settings |

---

## 8. Landing page (`/`)

- **Top bar:** the name, a **Sound** toggle (off by default and remembered), the theme
  toggle, and **Enter the lab →**.
- **Hero:** the SPECTRA wordmark, the tagline "Interactive audio signal processing.",
  Enter the lab and "See the tools ↓". The backdrop is a 20 Hz–20 kHz log-frequency
  grid with the **demo track's real spectrum drifting across it in real time**. The
  spectra are precomputed in `hero.json`. With reduced motion the backdrop shows a
  static average, and the animation pauses when the hero is off screen or the tab is
  hidden.
- **"The tools" reel:** a stories-style carousel with one card per tool.
  - Each card plays a ~6 s **before/after clip**: a 2.8 s phrase played original, then
    processed, with the label switching from "Original" to "Processed" ("STEM: VOCALS"
    for separation).
  - Its background is the tool's response line as an SVG (for separation, a spectrogram
    PNG of the stem).
  - Progress segments run along the top.
  - **Controls:**
    - hold the card, or press Space, to pause and loop its clip
    - tap the left 30 % of the card to go back, anywhere else to go forward
    - ← and → also work
    - "Open tool →" deep-links into `/lab#<tool>`
  - It pauses when scrolled mostly out of view or when the tab is hidden, and doesn't
    auto-advance with reduced motion.
  - The clips were made by the real DSP with these settings:
    - EQ: telephone preset
    - reverb: RT60 2.5, wet 0.5
    - echo: feedback, 300 ms, g 0.5, mix 0.6
    - flanger: 1 ms / 3 ms / 0.25 Hz / g 0.95
    - separation: the demo song's vocals stem from 13.7 s, with a credit line on the card
- **"How it's built":** an SVG flow diagram showing three paths from x[n] to y[n]. It
  has a wide and a narrow version.
  1. EQ and Separation: STFT → gain/mask H[k] → ISTFT
  2. Reverb: zero-pad → FFT → × H → IFFT
  3. Echo and Flanger: a difference equation in the time domain

  Below it are four fact cards: "No black boxes", "Tested against references (160+
  tests)", "Every system is visible" and "Backstage analysis".
- **Footer:** team name and a "Source on GitHub ↗" link. **Both are still placeholders**
  (see §12).
- The landing page makes no backend calls. The server embeds the asset list and the
  hero data in the page, and a missing asset leaves the page working ("Preview coming
  soon").

---

## 9. HTTP API (`app.py`)

All JSON errors look like `{"error": "message"}`. Status codes: 400 for bad
parameters, 404 for an unknown id, 413 for a file that is too large, 415 for MP3 when
there is no decoder, and 500 as `"Server error: Type: msg"`. Ids are UUID4 strings and
are validated before any file path is built.

| Method & route | Purpose |
|---|---|
| `GET /` | Landing page |
| `GET /lab` | The lab. The tool config (slider specs, presets, limits) is embedded as JSON. |
| `POST /upload` | Multipart field `file`. Returns `{file_id, filename, duration, samplerate, channels, url}` and writes a `<file_id>.json` sidecar. |
| `POST /upload/demo` | Registers `static/demo/demo.wav` like an upload |
| `POST /upload/demo/vocals` | Registers the Separation demo song like an upload |
| `POST /upload/result/<result_id>` | **Send to**: copies a result or stem into `uploads/` as a new upload, named like its download |
| `GET /upload/<file_id>` | The uploaded file (for the Original player) |
| `GET /upload/<file_id>/waveform` | Its min/max envelope and duration |
| `POST /process/eq` `/reverb` `/echo` `/flanger` | JSON or form body `{file_id, ...params}`. Returns `result_id`, `run_id` (the same value), `url`, `download_url`, `input_url`, spectrogram URLs, before/after waveform peaks, and for EQ also `curve`. |
| `POST /process/separate` | `{file_id}` → `{run_id, input_url, stems: [{name, result_id, url, download_url, duration, peaks}]}` |
| `GET /result/<id>` | Result WAV. `?download=1` sends it as an attachment with its generated name. |
| `GET /result/<id>/spectrogram/<before\|after>.png` | Spectrogram images |
| `GET /response/eq` `/reverb` `/echo` `/flanger` | Plot data from the query parameters only (`f_max` zooms echo and flanger) |
| `GET /backstage/runs` | Every registered run, newest first |
| `GET /backstage/<run_id>` | Analysis JSON (cached as `<run_id>.backstage-v1.json`) |
| `GET /backstage/<run_id>/diff.png` `/mixture.png` `/mask/<stem>.png` | Backstage images (cached) |

**Data on disk:**
- `uploads/`: `<file_id>.<ext>` plus `<file_id>.json` with the metadata.
- `processed/`:
  - `<result_id>.wav`
  - `_before.png` and `_after.png`
  - `<run_id>.run.json`, the run registry sidecar
  - `.backstage-v1.json`
  - `_diff.png`, `_mixture.png`, `_mask<i>.png`

  Separation stems each get their own UUID WAV.
- Nothing is ever deleted automatically. On the dev machine the folders currently hold
  about 120 MB and 770 MB.

**Limits:**
- Upload: 200 MB (Flask's `MAX_CONTENT_LENGTH`), 20 minutes, extensions wav/mp3/flac.
- MP3 support is detected at startup: `"MP3" in soundfile.available_formats()`, which
  needs libsndfile ≥ 1.1. The soundfile wheels bundle it. Without it the app logs a
  warning, and MP3 uploads return 415 while WAV and FLAC still work.

---

## 10. Tests

Run with `python -m pytest tests -m "not slow"`. Result on 2026-09-27: **281 passed, 5 slow deselected** (about 110 s).

| File | Covers |
|---|---|
| `test_eq_filter_unit.py` | Curve shapes (cutoff −3 dB, roll-off, band-stop, shelves, hum notches); **the audio really gets the plotted curve**, measured on sine tones; each preset does what its name says |
| `test_reverb_echo.py` | Reverb, echo and flanger checked **against naive reference implementations** (direct convolution, per-sample loops); impulse responses, gain clamping, dry/wet edge cases, routes; slow performance tests on 6 min of stereo |
| `test_separation.py` | Stems have the input's shape and are finite; **stems add up to the mix**; deterministic and leaves the global RNG alone; scale-invariant; silence and very short input; frame sizes and component grouping match the partner's `main()`; an empty group gives a silent stem; the demo separates well under a minute |
| `test_studio.py` | `/response/eq`, uploads and the demo, run ids and the registry, the separation mock, slider ranges coming from `PARAMS` |
| `test_backstage.py` | JSON shape per tool, ~0 dB difference when an effect does nothing, level stats on a known sine, caching, 404s, 6-minute timing |
| `test_landing.py` | `/` and `/lab`, links, every preview asset is served, the separation preview carries its credit, missing assets don't break the page, clip loudness matching |
| `test_demo_ready.py` | **Pages load nothing from the internet**, vendored files and fonts exist, warmup passes and reports failures, clean_runs, debug off by default, run scripts use port 5000 |

`test_eq_filter.py` in the repo root is a manual inspection script, not part of pytest.
As of 2026-09-25, one slow test (6-minute reverb under 5 s) was borderline at 5.2–5.6 s
on the dev laptop. It is a timing limit, not a correctness problem.

---

## 11. Facts that matter for deployment

The app has only been run locally so far and has **no deployment configuration** (no
Dockerfile, Procfile, WSGI server config or cloud config). These are the facts from the
code that a deployment has to account for:

1. **Entry point and host binding.** `app = Flask(__name__)` sits at module level in
   `app.py`, so a WSGI server can import it as **`app:app`**, for example gunicorn on
   Linux or waitress on Windows. `python app.py` uses Flask's development server and is
   **hard-coded to `127.0.0.1:5000`** (`HOST` and `PORT` constants at the bottom of
   `app.py`). That address is unreachable from other machines, so a deployment has to
   either use a WSGI server or change the bind to `0.0.0.0` and/or read `PORT` from the
   environment.
2. **Long synchronous requests.** Every `/process/*` call runs the DSP inside the HTTP
   request.
   - Effects take seconds.
   - **Separation takes about 0.4 s per second of audio**, so a 6-minute track holds
     one request for about 2.5 minutes and a 20-minute track (the upload limit) for
     longer.
   - The browser waits up to 20 minutes.
   - Any WSGI worker timeout (gunicorn's default is 30 s), reverse-proxy read timeout
     (nginx's default is 60 s) or platform request limit must be raised to match.
     Otherwise, lower the upload/duration limits for the hosted version.
   - There is no job queue and no progress reporting beyond the client-side seconds
     counter.
3. **Upload size.** Flask accepts up to 200 MB. A reverse proxy in front of it has its
   own limit (nginx `client_max_body_size` defaults to 1 MB), which must be raised.
4. **Run the app as one process, with threads if needed.** The run registry
   (`runs.py`):
   - keeps runs in memory
   - loads the `*.run.json` sidecars from disk **only once**, on first use
   - never re-reads them

   With several worker processes, a run made in worker A is invisible to worker B once
   B has loaded its registry. Backstage (`/backstage/<run_id>` returns "Unknown run")
   and download naming (the name falls back to `spectra_<id>.wav`) would then fail
   intermittently. The separation lock and the matplotlib plot lock are also
   per-process. A single process with several threads is safe as written. Multiple
   processes would need a registry change (for example, re-reading the sidecar on a
   cache miss).
5. **Storage.** Everything is on local disk in `uploads/` and `processed/`, next to
   `app.py`. These folders are created at startup.
   - Nothing expires them; only `scripts/clean_runs.py` (or `run.sh --clean`) empties
     them.
   - A hosted version needs a writable disk, and ideally a persistent one, because
     results and Backstage caches disappear on hosts with ephemeral filesystems.
   - It also needs periodic cleanup, for example a cron job running `clean_runs.py`
     or an age-based deletion.
6. **CPU and memory.** Separation (NMF with 20 components × 300 iterations on a
   2048-point STFT of the whole track, plus pandas) and the long-frame EQ are CPU-bound
   numpy work. Very small free-tier instances may be slow or run out of memory on long
   tracks. Only one separation runs at a time per process; others wait on the lock.
7. **No authentication and no per-user isolation.** Anyone who can reach the server can
   upload and run tools. Result and upload URLs are unguessable UUIDs.
   `GET /backstage/runs` returns **every** run from every visitor, including source
   filenames and parameters; the page filters it to the current tab's runs, but the
   API doesn't. This is fine for a class demo, but worth knowing for a public
   deployment.
8. **No secrets.** The app uses no sessions or cookies and no secret key, and debug is
   off unless `SPECTRA_DEBUG=1`.
9. **Static files** are served by Flask from `static/`: about 7 MB, no CDN, no build
   step. A reverse proxy can serve `static/` directly if wanted.
10. **Dependencies.** Python ≥ 3.12 (numpy 2.5 and scipy 1.18 require it). No ffmpeg is
    needed. soundfile's wheels bundle libsndfile with MP3 support, and the app checks
    this at startup. `pydub` and `audioop-lts` in `requirements.txt` can be dropped
    from a deployment image because only `trim.py` uses them.
11. **Licensing of the demo song:** CC BY-NC-SA 4.0. It needs attribution (already on
    the landing card and in `CREDITS.txt`) and non-commercial use only.

---

## 12. Known loose ends (as of 2026-09-27)

1. **`main` is 47 commits behind `merged`.** The finished app is on branch `merged`
   (pushed to `origin/merged`). `main` locally and on origin still points at the
   pre-pivot code from 2026-08-26. `main` is an ancestor of `merged`, so it can
   fast-forward: `git checkout main && git merge --ff-only merged && git push`.
   Evaluators or a deploy pipeline that clones `main` would get the old code.
2. **`README.md` is outdated.** Its "Run it" section is current, but the rest still
   describes "Stemify", a separation-only, "real-time", "in development" project with
   vocals/drums/bass/other stems. `PROGRESS.md` (2026-08-29, drums only) and
   `WASEK_STATUS.md` are also stale in places. For example, `WASEK_STATUS.md` still
   says CDN scripts and "separation returns 501 until the merge"; both are fixed.
3. **Landing page placeholders:** `static/js/landing.js` still has
   `teamName: "TEAM NAME"` and `githubUrl: "https://github.com/OWNER/REPO"`.
4. **`run.sh` is not executable in git** (mode 100644), so `./run.sh` fails with
   "permission denied" on a fresh clone on Linux or macOS. Workarounds: use
   `bash run.sh`, or fix it once with `git update-index --chmod=+x run.sh` and commit.
5. **Hum removal** leaves a little hum in the first and last ~0.5 s (inherent to very
   narrow notches, not a bug).
6. **Separation quality** depends on the track; bleed between overlapping instruments
   is the known limitation. It is also slow on long tracks (§6.5).
7. **`uploads/` and `processed/` grow without limit** (§11.5). Run `run.bat --clean`
   before the presentation for an empty Backstage run list.
8. Old experiments remain in the repo but the app doesn't use them: `bass_mask.py`,
   `trim.py` (needs pydub, plus ffmpeg for non-WAV), `main_semi_nmf.py` (a CLI script
   with a hard-coded `Audio/audio.wav` path) and `test_eq_filter.py`.

---

## 13. Presentation-day checklist (local PC)

1. On the lab PC, with internet: clone or copy the repo, check out `merged` (or `main`
   after the fast-forward), and run `run.bat` once so `.venv` gets installed.
2. From then on it works **offline**. Before presenting, run `run.bat --clean`, then in
   a second terminal run `.venv\Scripts\python scripts\warmup.py`. It should print
   PASS for all five tools. The warm-up runs also appear in `/backstage/runs`, but the
   Backstage tab only lists runs from the current browser tab, so the demo still starts
   with an empty list.
3. Open http://127.0.0.1:5000. The landing page is at `/` and the lab at `/lab`.
4. A demo flow that shows every feature:
   1. Landing reel with Sound on.
   2. EQ: Use demo track → Telephone preset → Run → compare Original and Processed →
      → Backstage.
   3. Reverb: show the circular toggle ("wrong way") and its Backstage FIG 09.
   4. Echo: feedforward vs feedback and the comb zoom.
   5. Flanger: the animated notch sweep.
   6. Send to: EQ output → Reverb.
   7. Separation: Use demo song (about 12 s) → Play all, then mute/solo stems and use
      the volume sliders → Backstage mask heatmaps.
   8. Theme toggle.
5. If separation ever fails during the demo, restart with `SEPARATION_MOCK=1`
   (Windows: `set SEPARATION_MOCK=1` then `python app.py` inside the venv). The stems
   are then copies of the input, but the whole UI still works.
