# Stage 1: Bass Classical Mask + Quality Gate (Drums & Bass)

## Context

This repo (working name AudioSieve/UNMASK) already has a working STFT/ISTFT
foundation and a drum separation module:

- `stft.py` — Hann windowing, framing, forward FFT (`np.fft.rfft`), inverse
  FFT (`np.fft.irfft`), overlap-add reconstruction with window-sum
  normalization.
- `utils.py` — audio I/O, magnitude computation, spectrogram
  plotting/saving, mask application (`spectra * mask`).
- `drum_mask.py` — multiple drum transient-detection approaches. The
  active one in `main.py` is `create_drum_mask_frequency_soft` (soft 0-1
  per-bin mask) + `smooth_mask_frequency` (moving-average smoothing).
- `main.py` — hardcoded pipeline on `Audio/trimmed.wav`, frame size 1024,
  hop size 512, drums only.

**Decision context:** the project uses a middle-ground strategy — classical
DSP masks for drums and bass where feasible, and htdemucs_6s-derived masks
(future stage) for vocals/guitar/other, or as a fallback for drums/bass if
the classical result isn't good enough. This stage produces the bass mask
and the mechanism to make that classical-vs-fallback call, on paper, per
stem.

## Goal

1. Implement a classical bass separation mask, mirroring the structure of
   `drum_mask.py`.
2. Generalize `main.py` so it isn't drum-hardcoded (accepts a stem
   argument: `drums` or `bass`).
3. Produce a documented, repeatable quality-gate process and a written
   GO/NO-GO decision for both drums and bass classical masks.

## Part A — `bass_mask.py`

Bass has different signal characteristics than drums, so don't reuse the
transient-detection approach as-is:

- **Drums:** broadband transients, short attack, energy jump across many
  frequencies simultaneously.
- **Bass:** narrow low-frequency band (~40-400 Hz fundamental + a few
  harmonics), sustained notes rather than transients, typically low
  polyphony (often near-monophonic).

Implement, at minimum:

- `create_bass_mask_bandlimit(spectrum, sr, freq_low=40, freq_high=400, rolloff_bins=5)`
  — a soft band-pass-style mask: bins within `[freq_low, freq_high]` get
  weight approaching 1, with a smooth (not hard-cutoff) rolloff outside
  the band to avoid ringing artifacts. Return a per-bin 0-1 mask, same
  shape convention as the drum masks.
- `create_bass_mask_energy(frame_spectra, sr, freq_low=40, freq_high=250, threshold=...)`
  — a refinement that also gates on *relative* low-band energy per frame
  (not just static frequency range), so the mask suppresses frames where
  the low band is quiet (e.g. a drum-only breakdown) even if they fall in
  the target frequency range. This reduces bleed from kick drum
  fundamentals, which overlap the bass band.
- `smooth_mask_frequency` from `drum_mask.py` can be reused as-is (import
  it) for post-smoothing — don't duplicate it.

Follow the existing code style/conventions in `drum_mask.py` (function
signature shape, return types, docstring format).

## Part B — Generalize `main.py`

- Replace the hardcoded drum-only pipeline with a `--stem` CLI argument
  (`drums` or `bass`), dispatching to the right mask function.
- Keep the existing spectrogram/mask PNG dump to `Spectograms/` — extend
  the filename to include the stem name (e.g. `mask_bass.png`,
  `reconstructed_bass.wav`) so drum and bass outputs don't overwrite each
  other.
- Don't do a full refactor into an importable `separate()` function yet —
  that's Stage 3, once htdemucs is in the mix too. A CLI flag is enough
  for this stage.

## Part C — Quality Gate

No ground-truth isolated stems are available, so evaluation has to be
proxy-based. Create `EVALUATION.md` at the repo root with:

1. **A fixed rubric**, scored per stem (drums, bass) on a simple scale
   (e.g. 1-5 or pass/fail) across:
   - **Bleed** — how much of *other* instruments leaks into this stem's
     reconstruction (assessed by ear).
   - **Completeness** — how much of the target instrument's own content
     is missing/thinned out.
   - **Artifacts** — musical noise, warbling, or discontinuities
     introduced by the mask (common failure mode of hard/binary masks).
   - **Spectrogram sanity check** — does the masked spectrogram visually
     track the expected frequency profile (e.g. bass mask should show a
     clean low-band strip, not scattered energy).
2. **A documented decision** at the end: for each of drums and bass,
   state PASS (keep classical mask) or FAIL (defer to htdemucs-derived
   mask in Stage 2), with 1-2 sentences of rationale referencing the
   rubric scores.
3. Run the evaluation on at least 2-3 different test clips (not just the
   current 10-second `Audio/trimmed.wav`) covering different genres/mixes,
   using `trim.py` to produce them, since a single clip isn't a reliable
   basis for a pipeline-wide decision.

## Acceptance Criteria

- [ ] `bass_mask.py` exists with at least the two functions above,
      producing a soft (non-binary) bass mask.
- [ ] `main.py` runs end-to-end for both `--stem drums` and `--stem bass`
      without hardcoded stem-specific paths.
- [ ] `Spectograms/` contains before/after spectrograms and mask
      visualizations for both stems, across multiple test clips.
- [ ] `EVALUATION.md` exists with the rubric, scores per clip, and a final
      explicit PASS/FAIL decision for drums and for bass.

## Out of Scope (don't touch yet)

- htdemucs_6s integration (Stage 2).
- Vocals, guitar, "other" stems.
- Django/web app work.
- The unused `find_drum_stripes` / `create_drum_mask_simple` experiments
  — leave them where they are for now (cleanup is a later pass).
