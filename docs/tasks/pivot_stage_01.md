# Pivot Stage 1: Flask App Shell (Upload/Download Skeleton)

## Context

The project has pivoted per supervisor feedback: source separation is no
longer the main deliverable. It's now one tool inside a broader **Audio
Lab** — independent signal-processing tools (Separation, EQ/Filter,
Reverb, Echo/Delay) the user can run on an uploaded track. Each tool is
its own tab/mode, not chained together.

This is a **separate track from the separation work** — your partner is
independently improving the NMF separation quality on their own branch,
building on the existing `stft.py` / `utils.py` / `drum_mask.py` /
`bass_mask.py` / `main.py` scaffolding. Don't modify those files in this
stage; this stage builds the shared app shell they'll eventually plug
into, plus your own effects modules on a separate path.

**Architecture decision for this pivot:** synchronous Flask, no job
queue, no WebSockets. Classical DSP effects (filtering, EQ, convolution
reverb) and NMF separation all run in well under a few seconds on a short
clip in Python — there's no long-running inference to report progress on
anymore, so a plain request → process → response cycle is enough. Keep
it simple.

## Goal

Stand up a minimal, working Flask app that proves the full upload → route
to a tool → process (stub for now) → return/download loop, before any
real DSP logic is added. This stage deliberately does **not** implement
real EQ, filter, or reverb math — it wires the skeleton so Stage 2+ can
drop real processing functions in without touching routing again.

## Requirements

### 1. Project structure

```
app.py                  # Flask app + routes
effects/
  __init__.py
  eq_filter.py           # stub for now: process(audio, sr) -> (audio, sr) unchanged
  reverb.py               # stub for now: same signature
  echo.py                  # stub for now: same signature
uploads/                 # incoming files (gitignored)
processed/               # output files (gitignored)
templates/
  index.html             # minimal tab-style nav: Separation / EQ+Filter / Reverb / Echo+Delay
static/                  # css/js as needed (keep barebones for now)
```

Don't create a `separation/` module yet — that's your partner's branch;
just leave a clearly marked stub route for it (see below) so the shell
is complete without colliding with their work.

### 2. Upload handling

- `POST /upload` — accepts a file (`multipart/form-data`), validates:
  - Format: `.wav`, `.mp3`, `.flac` only — reject anything else with a
    clear error message.
  - Length: reject clips over ~6 minutes.
  - Size: reject files over ~50MB.
- On success, save to `uploads/<uuid>.<ext>`, return the generated id
  (used to reference the file in subsequent processing calls).
- These limits exist for sane processing/demo behavior, not GPU/inference
  reasons anymore — keep them, but they're no longer hard architectural
  constraints, just good hygiene.

### 3. Processing routes (stubs)

- `POST /process/eq` — takes an uploaded file id + placeholder params,
  calls `effects.eq_filter.process()` (currently a no-op passthrough:
  load audio, return it unchanged), saves result to `processed/`,
  returns a reference to the result.
- `POST /process/reverb` — same pattern, calling
  `effects.reverb.process()` (also a no-op stub for now).
- `POST /process/echo` — same pattern, calling `effects.echo.process()`
  (also a no-op stub for now).
- `POST /process/separate` — same pattern, but literally just returns a
  "not yet implemented, see separation branch" placeholder response.
  Don't build real separation logic here — that's your partner's track.

### 4. Result retrieval

- `GET /result/<id>` — serves the processed audio file (for playback or
  download) referenced by the id returned from a `/process/*` call.

### 5. Minimal frontend

- `templates/index.html` — simple tab navigation (Separation / EQ+Filter
  / Reverb / Echo+Delay), each with: a file upload form, a "process" button hitting
  the relevant `/process/*` route, and an `<audio>` player pointed at the
  returned result once processing completes.
- Keep this barebones (plain HTML forms, no WaveSurfer.js/Tone.js yet) —
  the goal here is to prove the pipeline works end-to-end, not to build
  the real UI. Rich playback/visualization is a later stage.

## Acceptance Criteria

- [ ] `flask run` starts the app locally with no errors.
- [ ] Uploading a valid WAV/MP3/FLAC file succeeds; uploading an invalid
      format or oversized/overlong file is rejected with a clear message.
- [ ] Hitting `/process/eq`, `/process/reverb`, or `/process/echo` on an
      uploaded file returns audio identical to the input (since the
      effect functions are no-op stubs at this stage) and it's playable
      via the returned result URL.
- [ ] `/process/separate` returns a clear "not implemented here" response
      without touching the partner's separation branch/files.
- [ ] Directory/module structure matches the layout above so later stages
      (EQ, reverb, echo logic) can fill in `effects/eq_filter.py`,
      `effects/reverb.py`, and `effects/echo.py` without changing
      `app.py` routing.

## Out of Scope (don't touch yet)

- Real EQ/filter DSP math (Stage 2).
- Real convolution reverb DSP math (Stage 3).
- Real echo/delay DSP math (Stage 4).
- WaveSurfer.js / Web Audio / Tone.js frontend (later stage — playback
  and visualization only, never for the actual signal processing).
- Wiring in the partner's actual separation pipeline.
- Deployment.
