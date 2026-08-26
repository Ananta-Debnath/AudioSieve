# Stemify

**Stemify** is a real-time music source separation project that separates a mixed audio track into individual components such as **vocals, drums, bass, and other instruments**.

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
