# AudioSieve (Stemify) — Progress Notes

**Last updated:** 2026-08-29

A DSP-only (no ML) music source separation project. Goal: split a mixed
track into stems (vocals, drums, bass, other) using classic time-frequency
analysis instead of trained models. Currently the pipeline only implements
**drum isolation**; vocals/bass/other are future work per the README roadmap.

## Pipeline implemented so far

```
Audio/trimmed.wav
    → load & normalize (utils.load_audio)
    → framing + Hann window (stft.get_frames / apply_window)
    → FFT per frame (stft.calculate_fft)
    → drum mask estimation + smoothing (drum_mask.py)
    → mask applied to spectra (utils.apply_mask)
    → IFFT + overlap-add reconstruction (stft.calculate_ifft / overlap_add)
    → Audio/reconstructed.wav
```
`main.py` wires this end-to-end and also dumps a spectrogram of the
original and masked signal to `Spectograms/`.

## File-by-file summary

- **`stft.py`** — Core STFT/ISTFT machinery: Hann window creation, frame
  splitting, windowing, forward FFT (`np.fft.rfft`), inverse FFT
  (`np.fft.irfft`), and overlap-add reconstruction with window-sum
  normalization to avoid amplitude artifacts at frame boundaries.

- **`utils.py`** — I/O and support utilities: load/save WAV (mono
  conversion, normalization, clipping protection, 16-bit PCM export),
  magnitude computation, spectrogram plotting/saving (as PNG via
  matplotlib), mask application (`spectra * mask`), and mask
  visualization/saving.

- **`drum_mask.py`** — The actively-developed part ("Add mask for drums,
  in progress"). Contains several increasingly refined approaches to
  detecting drum transients, all exploring frame-to-frame or local-median
  energy jumps as a proxy for percussive hits:
  - `create_drum_mask` — whole-frame broadband energy transient detector
    (binary, all-or-nothing per frame).
  - `create_drum_mask_median` — same idea but compares against a local
    rolling median instead of just the previous frame (more robust to
    slow energy drift).
  - `create_drum_mask_frequency` — per-frequency-bin transient detection
    (binary) instead of whole-frame energy.
  - `create_drum_mask_frequency_soft` — soft/graded version of the above,
    producing a continuous 0–1 mask between a `threshold` and
    `full_strength` ratio. **This is the version currently used in
    `main.py`.**
  - `smooth_mask_frequency` — smooths a mask across neighboring frequency
    bins (moving average) to reduce a "patchy" mask.
  - `find_drum_stripes` / `create_drum_mask_simple` — a more elaborate,
    currently-unused (commented out in `main.py`) approach that looks for
    vertical broadband "stripes" in the spectrogram (energy simultaneously
    elevated across many frequencies, especially high frequencies) as a
    percussive-event candidate, then estimates per-bin drum strength
    relative to neighboring time frames.

- **`main.py`** — Orchestrates the full forward → mask → inverse pipeline
  described above. Currently hard-coded to a 10-second clip
  (`Audio/trimmed.wav`), frame size 1024, hop size 512. The soft
  frequency mask + smoothing (kernel size 10) is the active configuration;
  the "simple" stripe-based mask is present but commented out for later
  comparison/experimentation.

- **`trim.py`** — Small standalone script (uses `pydub`) to cut a section
  out of a source file, currently hard-coded to extract seconds 85–95 from
  `Audio/audio.wav` into `Audio/trimmed.wav`. Used to produce test clips.

- **`requirements.txt`** — `numpy`, `matplotlib`, `pydub`, `scipy`,
  `soundfile`.

- **`README.md`** — Project pitch/overview, states the project is
  🚧 in development and DSP-only (no ML) by design.

## Git history (chronological)

1. `9075240` Initial Commit — README + .gitignore.
2. `ba3d16d` Base STFT and ISTFT — `stft.py`, `utils.py`, `main.py`,
   `trim.py` scaffolding (forward/inverse transform + reconstruction,
   no separation yet).
3. `b0ff1e9` Add mask for drums (In progress) — introduced `drum_mask.py`
   with the multiple detection strategies above, wired masking into
   `main.py`, added mask-saving utility.
4. `e4c02b8` Add requirements file with essential dependencies.
5. `d8c9f65` Minor change — removed 2 stray lines from `stft.py`.

## Where things stand / open threads

- Only **drum** separation is attempted; vocals, bass, and "other" from
  the README's feature list are not yet started.
- Drum detection is still experimental — `drum_mask.py` has multiple
  competing implementations left in place (not cleaned up), and `main.py`
  currently uses the soft frequency-ratio approach while the stripe-based
  `create_drum_mask_simple` sits commented out as an alternative to
  revisit.
- No automated tests; verification appears to be visual (spectrogram/mask
  PNGs) and by ear (listening to `reconstructed.wav`).
- `Audio/` and `Spectograms/` are gitignored — sample audio and generated
  images aren't tracked, so the repo alone doesn't include test material.
