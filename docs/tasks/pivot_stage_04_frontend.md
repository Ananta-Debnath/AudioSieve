# PIVOT STAGE 04 — SPECTRA frontend (Studio + Backstage)

Branch: `wasek` · Code freeze: **Sun 2026-09-27** · Shown on a projector Mon 2026-09-28.

This stage has **two parts. Do them in order, with one commit each.**

- **Part A — Studio:** the tool screens. Do this first, because it is the minimum demoable product.
- **Part B — Backstage:** the analysis tab.

If time runs out, Part A alone must be a complete, working app.

---

## 0. Ground rules

1. **Do not change any DSP** in `effects/`, `stft.py`, or `utils.py`. `utils.py` and `stft.py` are
   shared with the partner's branch, so leave them untouched. New analysis code goes in new
   modules (see Part B).
2. **The frontend only plays and visualizes. It never processes audio.**
   - Allowed: `<audio>`, WaveSurfer.js (display and playback only), Chart.js (plots only).
   - Forbidden: `BiquadFilterNode`, `ConvolverNode`, `DelayNode`, Tone.js effects, or any
     client-side DSP.
3. **Stack:** plain HTML, CSS, and vanilla JS served by Flask.
   - No framework, no build step.
   - CDN scripts must be pinned to exact versions.
4. **Read the existing frontend, routes, and each module's `PARAMS` first.**
   - Slider min, max, step, and default values come from `PARAMS`. Never hard-code ranges in JS.
   - Keep existing API responses backward-compatible. Only add fields; never rename or remove them.
5. **The app name is `SPECTRA`.** Put it in one JS `CONFIG` constant and in `<title>`.

---

## 1. Design system

**Direction:** industrial but polished, with dark mode as the default. This is an **educational**
app, so clarity beats rawness: consistent spacing, clear labels, and readable text everywhere.

### 1.1 Color tokens (CSS custom properties)

The theme is set with `data-theme` on `<html>`. A header toggle switches between dark and light,
and the choice persists in `localStorage` (wrap every access in try/catch).

Dark theme (the default):

    --bg:#111315; --panel:#181B1E; --line:#2E3338; --fg:#E8EAEC; --fg-dim:#9AA0A6;
    --accent:#22D3EE;        /* electric cyan: fills, lines, slider fill, RUN */
    --accent-text:#22D3EE;   /* cyan used as text */
    --on-accent:#0B1215;     /* text on cyan fills */
    --warn:#F97316;          /* errors / the circular-convolution warning only */

Light theme:

    --bg:#F2F3F1; --panel:#FFFFFF; --line:#C9CDD1; --fg:#15181B; --fg-dim:#5B6168;
    --accent:#22D3EE; --accent-text:#0E7490; --on-accent:#0B1215; --warn:#C2410C;

**Rules:**
- Cyan is the only accent.
- **Encoding: cyan = processed / after / active; `--fg` or `--fg-dim` = original / before.**
  Use this everywhere, in players, plots, and labels, so the audience learns it once.
- Borders are 2px solid `--line`.
- `border-radius: 0` everywhere.
- No gradients and no shadows.

### 1.2 Typography (Google Fonts)

**Headings:** `Big Shoulders Display`, weight 800, UPPERCASE, letter-spacing `0.01em`,
line-height `0.95`.
- The app title in the header uses `clamp(2.5rem, 5vw, 4.5rem)`.
- Tool titles are around 2.25rem.

**Everything else:** `Martian Mono` (labels, values, body, buttons).
- Labels are UPPERCASE with letter-spacing `0.06em`.
- Body and explanatory text is sentence case.
- **Minimum size 13px.** Body text is 14px, and slider readouts are 17–18px in `--accent-text`.

Martian Mono is wide, so keep text lines short (about 60 characters or fewer).

### 1.3 Motion (subtle)

- Transitions of 150–200ms on hover and focus states and on tab switches (fade).
- The drawer slides in over about 220ms.
- While a RUN is in flight, its button reads `PROCESSING…` with an elapsed-seconds counter.
  There is no blinking or pulsing.
- Honor `prefers-reduced-motion` by turning off all transitions.

### 1.4 Plots (Chart.js)

- Before/original traces are drawn in `--fg-dim`; after/processed traces in `--accent`. Lines are
  2px, with no point markers.
- Gridlines use `--line`. Tick labels are Martian Mono at 12–13px. Axis titles always include units.
- **Read the colors from CSS variables at draw time, and redraw on theme toggle**, so charts
  follow the theme.

### 1.5 Spectrograms

- Standard spectrograms use the matplotlib `magma` colormap.
- The **difference spectrogram** uses a custom diverging map:
  magenta `#D946EF` (cut) → black (unchanged) → cyan `#22D3EE` (boosted).
- Spectrogram images keep a dark background in both themes (that's expected), framed with a 2px
  border: `--fg-dim` for before, `--accent` for after.

---

## Part A — Studio

### A1. Page layout (desktop first; must work at 1366×768 and 1920×1080)

    ┌───────────────────────────────────────────────────────────────────────┐
    │ SPECTRA   signal processing lab · CSE 220 · BUET          [☾ / ☀]     │
    ├────────┬────────┬──────────┬─────────┬────────────┬────────────────────┤
    │01 EQ + │02      │03 ECHO + │04       │05          │      ▌BACKSTAGE ▐  │ ← pinned right,
    │ FILTER │ REVERB │  DELAY   │ FLANGER │ SEPARATION │                    │   visually distinct
    ├────────┴────────┴──────────┴─────────┴────────────┴────────────────────┤
    │ LEFT ≈ 36%: CONTROLS         │ RIGHT ≈ 64%: VISUALS                   │
    │ tool title + 1-line subtitle │ system response plot (live)            │
    │ [ HOW IT WORKS ]             │                                        │
    │ source box (upload)          │ OUTPUT                                 │
    │ sliders / segmented buttons  │  ORIGINAL  player + waveform           │
    │ presets (EQ only)            │  PROCESSED player + waveform           │
    │ [ RUN ]                      │  [ DOWNLOAD WAV ]  [ → BACKSTAGE ]     │
    └──────────────────────────────┴────────────────────────────────────────┘

**Header:** app title, a subtitle, and the theme toggle.

**Tab bar:**
- Tool tabs show a number and a name. The active tab has a cyan background with `--on-accent` text.
- **BACKSTAGE** is pinned to the far right, separated from the tool tabs by a gap. Its default
  state is an outline style, and it has its own active style, so it reads as a separate room
  rather than a sixth tool.
- Arrow keys move between tabs. Focus outlines must be visible.

**Tab independence:** tabs keep their state (file, sliders, last result) when you switch away.

**Narrow screens:** below 900px wide, stack the layout into one column. Mobile is low priority,
but the page must not break.

### A2. Source box (inside every tool)

Each tool has **its own source file**. Inside the box:
- A drag-and-drop zone that is also click-to-browse, using the existing `POST /upload`.
- **`USE SAME FILE`**: reuses the most recent file uploaded in *any* tool. The frontend keeps
  `lastFileId` globally and simply reuses that `file_id`, with no re-upload. Hidden if nothing has
  been uploaded yet.
- **`USE DEMO TRACK`**: loads the shared demo track (see A7).

**Loaded state:** filename, duration, sample rate, and a small `REPLACE` button.

**Errors:** backend validation errors (format, 50 MB limit, 6 min limit) appear inline in a `--warn`
bordered message. For example: `File rejected: longer than 6:00.`

**Before a file is loaded:**
- RUN is disabled with the hint `Load a track first`.
- The response plot still works, since it needs no file.

### A3. Controls

**Sliders:**
- Every numeric parameter is a slider, custom-styled: a square thumb, a 6px track, and the filled
  part of the track in `--accent`.
- Each slider row shows a LABEL and a live readout with units, e.g. `RT60 1.50 s`.
- Double-clicking a readout resets that slider to its default.

**Discrete options** (filter type, echo mode, EQ vs filter mode, the reverb circular toggle) are
**segmented block buttons**, not native selects.

**Circular-convolution toggle:**
- The label reads `CIRCULAR (DEMO: WRONG WAY)`.
- When it is ON, the option shows a `--warn` border.

**EQ presets:**
- A row of block buttons: telephone, radio, remove rumble, remove hum 50 Hz, bass boost.
- Clicking a preset sets the sliders, updates the plot, and highlights that preset.
- Any manual slider change afterwards un-highlights it.

**RUN button:**
- Full width, cyan, label `RUN`.
- One request per tool at a time.
- It must never stay stuck disabled after an error.

### A4. HOW IT WORKS drawer

- A `HOW IT WORKS` button sits under each tool's title.
- It opens a **right-side drawer** 38% of the width (100% below 900px). The drawer is
  **non-modal**: the controls stay visible and usable while it is open.
- It closes with ✕, with Esc, or by clicking the button again.

**Drawer contents** (write real text for every tool, including separation):
1. The governing equation in a monospace block.
2. Three to five plain-language sentences.
3. What each parameter does, one line each.

| Tool | Equation(s) | Key idea for the text |
|---|---|---|
| EQ/Filter | `Y[k] = H[k]·X[k]`; Butterworth `|H(f)| = 1/√(1+(f/fc)^(2N))` | A filter removes a band; EQ reshapes the balance with gentle boosts and cuts |
| Reverb | `y = x * h  ⟷  Y = X·H` | Convolution theorem; zero-pad to N+M−1 or the tail wraps around (circular) |
| Echo | `y[n] = x[n] + g·x[n−D]` / `y[n] = x[n] + g·y[n−D]` | Feedforward = one repeat (FIR); feedback = decaying repeats (IIR), stable only for g < 1 |
| Flanger | `y[n] = x[n] + g·x[n−D(n)]` | A comb filter whose notches sweep because D(n) is driven by an LFO |
| Separation | `V ≈ W·H` (NMF); mask = stem estimate / mixture | Spectral templates × activations; the known limitation is bleed when sources overlap in frequency |

### A5. Visuals (right column)

**System response plot:**
- Redraws live as sliders move, debounced by about 150ms. It uses `GET /response/reverb`,
  `/response/echo`, `/response/flanger`, and the new `/response/eq` (see A8).
- Ignore stale responses: keep a request counter per tool and drop any response older than the
  newest request.

**Per-tool plots:**
- **EQ:** gain (dB) against frequency on a **log axis**, 20 Hz–20 kHz, with a 0 dB reference line.
- **Reverb:** h(t) against seconds.
- **Echo:** |H| in dB against frequency. It defaults to 0–200 Hz, with a zoom control:
  `200 Hz | 2 kHz | FULL` (segmented buttons).
- **Flanger:** keep the existing animated sweep, restyled, with a play/pause for the animation.
- **Separation:** no response plot. Show the stem list instead (A6).

**Output area:**
- Before the first run, it shows `No output yet. Load a track and press RUN.`
- After a run, **two stacked players**:
  - `ORIGINAL`: WaveSurfer waveform in `--fg-dim`, with its own play/pause and time readout.
  - `PROCESSED`: WaveSurfer waveform in `--accent`, with its own play/pause and time readout.
- Starting one player pauses the other.
- Space bar plays/pauses the last-used player (only when focus is not in an input).
- Buttons under the players: `DOWNLOAD WAV` and `→ BACKSTAGE`. The second switches to Backstage
  with this run selected.
- The processed output can be longer than the input (reverb and echo tails). Each waveform simply
  shows its own length.

### A6. Separation tool (built against a contract; merged later)

**Contract** the partner's code must match:

    POST /process/separate  (same file_id pattern as the other tools)
    200 → {"run_id": "...",
           "stems": [{"name":"drums","url":"...","download_url":"..."}, ...]}

The stem count comes from the response. Never assume 3.

**Mock mode:** when `SEPARATION_MOCK=1` is set, the route returns this contract with copies of the
input as fake stems (drums, bass, rest). Otherwise it keeps returning 501. On 501, the UI shows
`Separation module offline, pending merge.`

**Stem list:**
- One row per stem: the stem name in the heading font, a compact WaveSurfer waveform,
  play/pause, `MUTE`, `SOLO`, and `DOWNLOAD`.
- A master `PLAY ALL` plays all unmuted stems in sync (start them together and seek them together).
- Mute and solo only set playback `volume` 0/1. That is playback control, not processing.
- An `ORIGINAL` player sits above the stems.
- `→ BACKSTAGE` works the same as in the other tools.

### A7. Demo track

- Path: `static/demo/demo.wav`.
- **I (Wasik) will add the file.** Until then, generate a 20-second placeholder: a mix of a sine
  melody, a low bass tone, and noise bursts, created once by a small script
  `scripts/make_placeholder_demo.py`.
- New route `POST /upload/demo`: registers the demo file through the same path as a normal upload
  and returns a normal `file_id`, so the demo goes through the same pipeline as any upload.

### A8. Backend additions for Part A

1. **`GET /response/eq`** takes the same query params as the EQ tool and returns
   `{"freqs": [...], "gain_db": [...]}`, computed by the **existing** functions in
   `effects/eq_filter.py` on a log-spaced grid of about 1000 points.
2. **`POST /upload/demo`**, as described in A7.
3. **`SEPARATION_MOCK=1`** mock mode, as described in A6.
4. **Every `/process/*` response also returns a `run_id`.** It can reuse the existing result id.
5. **Run registry:** the server records, per run, the tool, params, input `file_id`, output path(s),
   and a timestamp.
   - In memory, with a JSON sidecar next to the output files so it survives a restart.
   - Part B needs this. Add it now.
6. **Upload metadata:** the upload response includes duration and sample rate, if it doesn't already.
7. **Tests** for each item above, in the existing test style.

### A9. Part A acceptance

- [ ] Each of the 4 effect tools: upload → sliders → RUN → two stacked players → download.
- [ ] `USE SAME FILE` and `USE DEMO TRACK` work in every tool.
- [ ] Response plots update live while dragging a slider, without a file loaded, and without
      stale-response flicker.
- [ ] The HOW IT WORKS drawer opens and closes (✕, Esc, button). Controls stay usable while it is open.
- [ ] The theme toggle switches everything, including the charts, and persists after a reload.
- [ ] Separation mock (3 stems, PLAY ALL/mute/solo/download) works; without the mock variable,
      the offline message shows.
- [ ] Bad uploads show inline errors. RUN never stays stuck.
- [ ] Readable at 1366×768 and at 125% browser zoom. No console errors.
- [ ] `python -m pytest tests -m "not slow"` passes.
- [ ] Commit: `pivot stage 04a: SPECTRA studio UI`.

---

## Part B — Backstage

### B1. Concept

Backstage is **one global tab** that shows a dense "instrument wall" of analysis for a single run.
It should look like a busy lab bench but stay readable, because **every panel follows the same
rules:**
- A header `FIG 0N / TITLE` in the heading font.
- A one-line **`LOOK FOR:`** caption in `--fg-dim` telling the viewer what to notice.
  For example: `LOOK FOR: cyan = frequencies boosted, magenta = cut.`
- A 2px border.

**Layout:** a CSS grid of panels in mixed sizes (some span 2 columns). The wall scrolls
vertically. Each panel stays compact.

### B2. Run picker

- A top bar lists this session's runs: tool, time, and key params, e.g.
  `REVERB · 14:32 · RT60 1.5s WET 0.3`.
- The newest run is selected by default. `→ BACKSTAGE` from a tool selects that tool's run.
- With no runs yet, it shows `No runs yet. Process a track in any tool, then come back here.`

### B3. Panels

Build them in tier order. **Tiers 1 and 2 are required; Tier 3 is stretch, in the listed order.**

**Tier 1 — core (every effect run)**
1. **Spectrograms, before and after** (magma), side by side.
2. **Difference spectrogram:** `20·log10(|Y|+ε) − 20·log10(|X|+ε)`, clipped to ±24 dB, with the
   diverging map and a colorbar. Compute it over the original length. If the output is longer,
   say so in the caption.
3. **System response:** the same plot as the Studio tab, for this run's params.
4. **Average spectrum:** the mean magnitude over all frames, in dB against log frequency. Before
   and after on one chart.
5. **Waveform overlay:** the before and after min/max envelopes (about 2000 points each) on one chart.
6. **Run card:** tool, all parameter values with units, and the governing equation.

**Tier 2 — readouts**

7. **Level meters:** peak dBFS, RMS dBFS, crest factor, and duration, before and after, as a
   compact table with the change highlighted in cyan.
8. **STFT settings:** frame size N, hop, window, sample rate, Δf = sr/N, and time resolution
   N/sr, for the analysis frames. For EQ runs, also show the long 65536-sample processing frames
   next to them, with the caption:
   `LOOK FOR: long frames give fine frequency resolution but poor time resolution.`

**Tier 3 — tool-specific (stretch)**

9. **Reverb:** the first 2 s of output in linear mode against circular mode, as a waveform overlay.
   Caption: `LOOK FOR: in circular mode the tail wraps onto the start.`
10. **Echo:** the impulse response as a stem plot (gᵏ at k·D).
11. **Flanger:** delay D(n) over time (the LFO curve).
12. **Separation:** the mixture spectrogram plus a per-stem mask heatmap (`|stem|/|mixture|`,
    clipped to 0–1). Optional: the NMF W and H, if the partner's code exposes them.

### B4. Backend for Part B

- **New module `analysis/backstage.py`.** It uses `stft.py` functions read-only and computes
  everything from the run registry: input file plus output file(s).
- **Main route:** `GET /backstage/runs` lists the runs; `GET /backstage/<run_id>` returns JSON
  with the average spectra, envelopes, level stats, STFT settings, and the params.
- **Image routes:** `GET /backstage/<run_id>/diff.png` and the per-stem mask PNGs, generated with
  matplotlib and cached on disk after the first request.
- Reuse the existing before/after spectrogram URLs where they already exist.
- **Performance target:** the Backstage JSON arrives in under about 3 s for a 6-minute track.
  Downsample everything that is sent for plotting.
- **Tests:** JSON shape for each tool; a known case where the difference spectrogram is about
  0 dB everywhere (gain 0 / wet 0); and level stats on a known sine.

### B5. Part B acceptance

- [ ] Every effect tool's run shows all Tier 1 and Tier 2 panels with correct captions.
- [ ] The picker switches between runs. `→ BACKSTAGE` from a tool lands on that run.
- [ ] A separation (mock) run shows its stems without crashing, even if the Tier 3 masks aren't built.
- [ ] Both themes are readable; charts recolor on toggle.
- [ ] Tests pass. Commit: `pivot stage 04b: SPECTRA backstage analysis`.

---

## Out of scope

- New effects
- Client-side audio processing
- Accounts
- Deployment configuration (the next stage)
