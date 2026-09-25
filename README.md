# Stemify

**Stemify** is a real-time music source separation project that separates a mixed audio track into individual components such as **vocals, drums, bass, and other instruments**.

## Run it

Needs Python 3.12 or newer. From the repo folder:

```text
run.bat            (Windows)
./run.sh           (macOS / Linux / Git Bash)
```

Then open <http://127.0.0.1:5000>. The first start creates `.venv` and installs
`requirements.txt` (this needs the internet). Later starts skip that, and the app
itself never loads anything from the internet (scripts and fonts are in `static/`).

* `--clean` deletes old uploads and results first.
* `python scripts/warmup.py`, with the app running, runs every tool once on the
  demo track and prints PASS / FAIL per tool.
* `python -m pytest tests -m "not slow"` runs the fast tests.
* Long tracks: separation works on the first 60 s only
  (`SEPARATION_MAX_SECONDS` changes that; `0` means the whole track).
* Audio is decoded by soundfile's libsndfile: WAV, FLAC and MP3, no ffmpeg needed.

## Overview

The project analyzes an input music signal and generates separate audio stems for its major sources.

The system is designed around traditional **digital signal processing (DSP)** techniques rather than machine learning.

## Features

* 🎵 Separate a mixed music track into individual sources
* 🎤 Vocal isolation
* 🥁 Drum isolation
* 🎸 Bass and instrumental separation
* ⚡ Real-time processing
* 🔊 Audio reconstruction from separated components
* 🚫 No machine learning

## Approach

Stemify uses time-frequency analysis to identify and isolate different components of a music signal.

```text
Input Audio
    ↓
Audio Framing
    ↓
Time-Frequency Analysis
    ↓
Source Identification
    ↓
Source Separation
    ↓
Audio Reconstruction
    ↓
Separated Stems
```

## Project Status

🚧 **In Development**

The separation algorithms are actively being developed and improved. The quality of separation depends on the characteristics of the input audio.

## Goals

The main goal is to explore how far music source separation can be achieved using **signal processing techniques alone**, without relying on trained models.

## License

License information will be added as the project develops.
